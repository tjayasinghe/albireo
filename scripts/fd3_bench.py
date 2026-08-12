"""fd3 comparison harness (M5): export, run both codes, compare (docs/design.md §1).

Simulates an SB2 on a common ln-lambda grid (fd3's required sampling — the native
grid IS the model grid here, so neither code resamples), writes fd3 v3.1 input
files (format verified against the official examples; see docs/benchmarks.md M5),
runs the albireo fixed-orbit solve, and — when an fd3 binary is available — runs
fd3 in separation mode on identical data and compares recovered component spectra
and wall time.

Without an fd3 binary the export + albireo side still runs and writes everything
needed; point --fd3 at the binary once built (source:
http://sail.zpf.fer.hr/fdbinary/fd3.tar.gz, ~1.9 MB, needs GSL; no license stated
on the page — contact the author before redistribution).

fd3 format facts this exporter respects (all verified against the fd3 v3.1 source
and official example files):
- master file header "# <ncols> X <nrows>", '#' at byte 0, uppercase X;
- column 1 is ln(wavelength), *exactly* equidistant, ascending (fd3 derives the
  step from the two endpoints only and never checks uniformity);
- control stream is a flat whitespace token list with NO comments: master, z0, z1,
  root, 3 component switches, M x (t[d], rv_corr[km/s], sigma, lf per enabled
  component), 13 (value, step) orbital pairs (wide orbit first; omega in DEGREES;
  step 0 = fixed), nruns, niter, stoprat;
- component B's RV gets the opposite sign internally (matches albireo's omega+pi
  convention for component 2), K's are entered positive;
- fd3 output .mod: same "# ncols X nrows" layout, col 1 = ln lambda.

Run: python scripts/fd3_bench.py [--fd3 PATH] [--outdir DIR] [--fit]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import albireo as ab
from albireo.forward import build_problem
from albireo.kepler import t_peri_from_t_conj
from albireo.likelihood import marginal_loglikelihood
from albireo.priors import SmoothnessPrior

# --- benchmark configuration (deterministic) --------------------------------
GRID = ab.LogGrid.from_wavelength_range(4400.0, 4600.0, dv_kms=3.0)
P, TCONJ, ECC, OMEGA = 6.31, 2.05, 0.2, 0.7  # omega in rad
K1, K2 = 45.0, 70.0
ELL = np.array([0.62, 0.38])
N_EP, SNR, LSF_V = 20, 100.0, 6.0
TAU, ETA = 300.0, 5.0


def simulate():
    rng = np.random.default_rng(42)
    bjd = np.sort(rng.uniform(0.0, 2.7 * P, N_EP))
    comps = [
        ab.synthetic_deviation_spectrum(
            GRID, n_lines=45, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 22.0), seed=s
        )
        for s in (1, 2)
    ]
    tperi = float(t_peri_from_t_conj(TCONJ, period=P, ecc=ECC, omega=OMEGA))
    orbit = ab.OrbitParams(period=P, t_peri=tperi, ecc=ECC, omega=OMEGA, k=(K1, K2))
    # native grid = model grid interior (fd3 needs the common log grid; edge pixels
    # are dropped so shifted spectra stay clear of the zero-padded boundary)
    interior = GRID.wave[60:-60]
    spec = ab.InstrumentSpec(wave=interior, sigma_v_lsf=LSF_V, snr=SNR)
    ds, truth = ab.simulate_dataset(
        GRID,
        comps,
        bjd=bjd,
        instruments={"i": spec},
        light_fractions=ELL,
        orbit=orbit,
        v_bary=np.zeros(N_EP),
        frame="barycentric",
        seed=7,
    )
    return ds, truth, tperi


def write_fd3_inputs(outdir: Path, ds, tperi: float, fit: bool) -> tuple[Path, Path]:
    """Write master.obs + control file; returns (control_path, root)."""
    # ln-lambda of the observed (interior) pixels: exact arithmetic progression
    i0 = 60
    n_native = ds[0].wave.size
    lnlam = GRID.x0 + GRID.dx * (i0 + np.arange(n_native))
    flux = np.stack([ep.flux for ep in ds])  # (M, n)
    m = flux.shape[0]

    master = outdir / "bench.master.obs"
    with open(master, "w", newline="\n") as fh:
        fh.write(f"# {m + 1} X {n_native}\n")
        for row in range(n_native):
            vals = " ".join(f"{v:.10f}" for v in flux[:, row])
            fh.write(f"{lnlam[row]:.16e} {vals}\n")

    root = "bench_fit" if fit else "bench_fixed"
    lines: list[str] = [
        "bench.master.obs",
        f"{lnlam[0] - 1e-9:.16e}  {lnlam[-1] + 1e-9:.16e}",
        root,
        "1 1 0",
    ]
    sigma = 1.0 / SNR
    for ep in ds:
        lines.append(f"{ep.bjd:.10f}  0.0  {sigma:.6e}  {ELL[0]:.6f}  {ELL[1]:.6f}")
    om_deg = np.degrees(OMEGA)
    if fit:
        close = f"{P} 0   {tperi} 0.2   {ECC} 0.05   {om_deg} 20   {K1 * 0.93} 5   {K2 * 1.05} 5"
        opt = "40 2000 0.001"
    else:
        close = f"{P} 0   {tperi} 0   {ECC} 0   {om_deg} 0   {K1} 0   {K2} 0"
        opt = "1 1 0.001"
    wide = "1 0   0 0   0 0   0 0   0 0   0 0"
    lines.append(f"{wide}   {close}   0 0")
    lines.append(opt)
    control = outdir / f"{root}.in"
    control.write_text("\n".join(lines) + "\n", newline="\n")
    return control, outdir / root


def read_fd3_matrix(path: Path) -> np.ndarray:
    with open(path) as fh:
        header = fh.readline().split()
        ncols, nrows = int(header[1]), int(header[3])
        data = np.loadtxt(fh)
    assert data.shape == (nrows, ncols), f"{path}: {data.shape} != {(nrows, ncols)}"
    return data


def run_albireo(ds, truth):
    problem = build_problem(
        GRID,
        ds,
        velocities=truth.velocities,
        light_fractions=ELL,
        lsf_sigma_v={"i": LSF_V},
    )
    prior = SmoothnessPrior(jnp.full(2, TAU), jnp.full(2, ETA))
    t0 = time.time()
    result = marginal_loglikelihood(problem, prior)
    d_hat = np.asarray(result.d_hat)
    wall = time.time() - t0
    return d_hat, wall


def spectrum_metrics(recovered: np.ndarray, truth_d: np.ndarray, label: str):
    """RMS in line cores, raw and mean-aligned (both codes have a k~0 freedom)."""
    core = truth_d < -0.15
    core[:80] = core[-80:] = False
    err = recovered[core] - truth_d[core]
    raw = float(np.sqrt(np.mean(err**2)))
    aligned = float(np.sqrt(np.mean((err - err.mean()) ** 2)))
    print(f"  {label}: core RMS {raw:.4f} (mean-aligned {aligned:.4f}, {core.sum()} px)")
    return raw, aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd3", default=None, help="path to the fd3 binary (else $PATH lookup)")
    ap.add_argument("--outdir", default="fd3_bench_out")
    ap.add_argument("--fit", action="store_true", help="also write/run the orbit-fitting mode")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    ds, truth, tperi = simulate()
    truth_d = np.stack([np.asarray(c) for c in truth.components])[:, 60:-60]

    control_fixed, root_fixed = write_fd3_inputs(outdir, ds, tperi, fit=False)
    if args.fit:
        write_fd3_inputs(outdir, ds, tperi, fit=True)
    np.savez(
        outdir / "truth.npz",
        components=truth_d,
        velocities=truth.velocities,
        bjd=ds.bjd,
        light=ELL,
    )
    print(f"fd3 inputs written to {outdir}/ (master + control files)")

    print("albireo fixed-orbit solve (same data, native masks/weights unused: parity):")
    d_hat, wall = run_albireo(ds, truth)
    for i in range(2):
        spectrum_metrics(d_hat[i][60:-60], truth_d[i], f"albireo comp {i + 1}")
    print(f"  albireo wall (un-jitted single solve): {wall:.2f} s")

    fd3_bin = args.fd3 or shutil.which("fd3")
    if fd3_bin is None:
        print(
            "\nfd3 binary not found - export complete, comparison pending.\n"
            "Build it (Linux/WSL): download http://sail.zpf.fer.hr/fdbinary/fd3.tar.gz\n"
            "(~1.9 MB, needs libgsl-dev), `make`, then rerun with --fd3 path/to/fd3\n"
            f"or run manually:  fd3 < {control_fixed.name} > {root_fixed.name}.out  (in {outdir}/)"
        )
        return

    print(f"\nrunning fd3 ({fd3_bin}), separation mode:")
    t0 = time.time()
    with open(control_fixed) as fin, open(f"{root_fixed}.out", "w") as fout:
        subprocess.run([fd3_bin], stdin=fin, stdout=fout, cwd=outdir, check=True)
    wall_fd3 = time.time() - t0
    mod = read_fd3_matrix(Path(f"{root_fixed}.mod"))
    # fd3 components are normalized flux; ours are deviations. Interpolate fd3's
    # output (its ln-lambda column) onto the interior pixel grid for comparison.
    i0 = 60
    lnlam = GRID.x0 + GRID.dx * (i0 + np.arange(truth_d.shape[1]))
    for i in range(2):
        comp = np.interp(lnlam, mod[:, 0], mod[:, i + 1]) - 1.0
        spectrum_metrics(comp, truth_d[i], f"fd3 comp {i + 1}")
    print(f"  fd3 wall (separation): {wall_fd3:.2f} s")


if __name__ == "__main__":
    main()
