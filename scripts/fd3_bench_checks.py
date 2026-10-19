"""Checks of the three-code comparison at a common resolution.

docs/benchmarks.md, "The three-code comparison at a common resolution (2026-10-15)", quotes
the numbers this script prints:

1. albireo, fd3 and shift-and-add (7 and 50 sweeps, and the limit of its iteration) on
   twenty noise realizations of the benchmark SB2 of ``scripts/fd3_bench.py``. Only the
   noise seed changes; seed 7 is the recorded benchmark.
2. albireo given no LSF, the information the other two codes receive, and albireo with its
   smoothing weights fitted by ML-II instead of held at the harness's 300 and 5.
3. The error of each code in three frequency bands.
4. The bias of each code, from a noiseless copy of the seed-7 data with its weights kept.
5. Shift-and-add against the number of sweeps, and the limit of its iteration.

The scoring is that of fd3_bench.py: against B d, the injected spectra convolved with the
simulated LSF, in the line cores (B d < -0.15, 80 px at either end left out), raw and after
the mean of the error is removed. albireo's deconvolved spectra are convolved with the same
LSF first.

The limit of shift-and-add. With the plain mean, ``shift_and_add.disentangle`` is the
Gauss-Seidel iteration ``A <- a - P B``, ``B <- b - Q A`` with ``a = <T(-va_j) S_j>``,
``b = <T(-vb_j) S_j>``, ``P = <T(vb_j - va_j)>`` and ``Q = <T(va_j - vb_j)>``, where
``T(s)`` is the module's zero-padded linear-interpolation shift. Its fixed point solves
``(I - P Q) A = a - P b`` and ``B = b - Q A``, which is solved here directly and compared
with 20,000 sweeps of the iteration.

fd3 runs once per realization, from the inputs that ``fd3_bench.write_fd3_inputs`` writes.
``--fd3`` takes the command that runs the binary, split as a shell would split it:
``--fd3 /path/to/fd3`` on Linux, ``--fd3 "wsl -e env OMP_NUM_THREADS=1 /home/<user>/fd3/fd3"``
from Windows. Without ``--fd3`` the ``.mod`` files of an earlier run in ``--outdir`` are read
where present, and the fd3 rows are left out otherwise. ``--time`` adds the wall-clock time
of shift-and-add against the number of sweeps, each in a fresh interpreter that never
imports jax (run it with nothing else running).

Run: python scripts/fd3_bench_checks.py [--fd3 CMD] [--outdir DIR] [--seeds N] [--time]
About 11 minutes on the desktop, most of it the 42 ML-II fits and the 20,000-sweep run;
``--time`` adds about 5 minutes.
"""

from __future__ import annotations

import argparse
import dataclasses
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import fd3_bench as fb
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.sparse.linalg import splu
from shift_and_add import disentangle

import albireo as ab
from albireo.forward import build_problem
from albireo.likelihood import marginal_loglikelihood
from albireo.priors import SmoothnessPrior

EDGE = 60  # model-grid margin outside the observed (interior) grid, as in fd3_bench
NO_LSF_KMS = 1e-3  # a Gaussian this narrow is the identity kernel on the 3 km/s grid
SWEEPS = (1, 7, 50, 500, 5000, 20000)
ALBIREO = ("albireo LSF, weights held", "albireo no LSF, weights held")
ALBIREO_MLII = ("albireo LSF, ML-II weights", "albireo no LSF, ML-II weights")


# --- the codes --------------------------------------------------------------------------


def albireo_solve(ds, truth, *, lsf: bool, tau=None, eta=None) -> np.ndarray:
    """Posterior-mean spectra on the full model grid, orbit fixed at the injected values."""
    problem = albireo_problem(ds, truth, lsf=lsf)
    tau = np.full(2, fb.TAU) if tau is None else np.asarray(tau, dtype=np.float64)
    eta = np.full(2, fb.ETA) if eta is None else np.asarray(eta, dtype=np.float64)
    prior = SmoothnessPrior(jnp.asarray(tau), jnp.asarray(eta))
    return np.asarray(marginal_loglikelihood(problem, prior).d_hat)


def albireo_problem(ds, truth, *, lsf: bool):
    width = fb.LSF_V if lsf else NO_LSF_KMS
    return build_problem(
        fb.GRID,
        ds,
        velocities=truth.velocities,
        light_fractions=fb.ELL,
        lsf_sigma_v={"i": width},
    )


def fit_weights(ds, truth, *, lsf: bool) -> tuple[np.ndarray, np.ndarray]:
    """ML-II: (tau_i, eta_i) maximizing the marginal likelihood at the injected orbit."""
    problem = albireo_problem(ds, truth, lsf=lsf)
    half_bandwidth = problem.natural_half_bandwidth

    def negative(x):
        prior = SmoothnessPrior(jnp.exp(x[:2]), jnp.exp(x[2:]))
        return -marginal_loglikelihood(problem, prior, half_bandwidth=half_bandwidth).log_likelihood

    value_and_grad = jax.jit(jax.value_and_grad(negative))

    def fun(x):
        value, grad = value_and_grad(jnp.asarray(x))
        return float(value), np.asarray(grad, dtype=np.float64)

    x0 = np.log([fb.TAU, fb.TAU, fb.ETA, fb.ETA])
    opt = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=[(-5.0, 20.0)] * 4)
    if not opt.success:
        print(f"  ML-II did not converge: {opt.message}")
    return np.exp(opt.x[:2]), np.exp(opt.x[2:])


def shift_matrix(s: float, n: int) -> sp.csr_matrix:
    """``shift_and_add._shift`` as a matrix: out[p] = (1-f) x[p+k] + f x[p+k+1]."""
    k = int(np.floor(-s))
    f = (-s) - k
    return sp.diags([1.0 - f, f], [k, k + 1], shape=(n, n), format="csr")


class ShiftAndAddLimit:
    """The fixed point of ``disentangle`` (plain mean) for one set of shifts."""

    def __init__(self, shifts_pix: np.ndarray, n_pix: int):
        va, vb = shifts_pix
        n_ep = va.size
        self.p = sum(shift_matrix(vb[j] - va[j], n_pix) for j in range(n_ep)) / n_ep
        self.q = sum(shift_matrix(va[j] - vb[j], n_pix) for j in range(n_ep)) / n_ep
        self.ta = [shift_matrix(-va[j], n_pix) for j in range(n_ep)]
        self.tb = [shift_matrix(-vb[j], n_pix) for j in range(n_ep)]
        identity = sp.identity(n_pix, format="csc")
        self.lu = splu((identity - self.p @ self.q).tocsc())

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Light-weighted components, as ``disentangle`` returns them."""
        n_ep = obs.shape[0]
        a = sum(self.ta[j] @ obs[j] for j in range(n_ep)) / n_ep
        b = sum(self.tb[j] @ obs[j] for j in range(n_ep)) / n_ep
        comp_a = self.lu.solve(a - self.p @ b)
        return np.stack([comp_a, b - self.q @ comp_a])


def noiseless(ds, truth):
    """The dataset with each flux replaced by its noiseless value, the weights kept."""
    epochs = [
        dataclasses.replace(ep, flux=np.asarray(f, dtype=np.float64))
        for ep, f in zip(ds, truth.noiseless_flux, strict=True)
    ]
    return ab.Dataset(epochs, frame=ds.frame)


def read_fd3(case_dir: Path) -> np.ndarray | None:
    path = case_dir / "bench_fixed.mod"
    if not path.exists():
        return None
    mod = fb.read_fd3_matrix(path)
    n_native = fb.GRID.n - 2 * EDGE
    lnlam = fb.GRID.x0 + fb.GRID.dx * (EDGE + np.arange(n_native))
    return np.stack([np.interp(lnlam, mod[:, 0], mod[:, i + 1]) - 1.0 for i in range(2)])


def run_fd3(command: list[str], case_dir: Path) -> None:
    with open(case_dir / "bench_fixed.in") as fin, open(case_dir / "bench_fixed.out", "w") as fout:
        subprocess.run(command, stdin=fin, stdout=fout, cwd=case_dir, check=True)


# --- one realization ----------------------------------------------------------------------


def run_case(
    ds, truth, tperi, case_dir: Path, fd3_command, limit: ShiftAndAddLimit
) -> tuple[dict, dict]:
    """Every estimate for one dataset, keyed by row label, and the ML-II weights."""
    case_dir.mkdir(parents=True, exist_ok=True)
    fb.write_fd3_inputs(case_dir, ds, tperi, fit=False)
    if fd3_command:
        run_fd3(fd3_command, case_dir)

    def instrumental(d_hat, lsf):
        full = np.stack([fb.lsf_convolve(c) for c in d_hat]) if lsf else d_hat
        return full[:, EDGE:-EDGE]

    out, weights = {}, {}
    for label, lsf in zip(ALBIREO, (True, False), strict=True):
        d_hat = albireo_solve(ds, truth, lsf=lsf)
        out[label] = instrumental(d_hat, lsf)
        if lsf:
            out["albireo deconvolved, weights held"] = d_hat[:, EDGE:-EDGE]
    for label, lsf in zip(ALBIREO_MLII, (True, False), strict=True):
        tau, eta = fit_weights(ds, truth, lsf=lsf)
        weights[label] = (tau, eta)
        d_hat = albireo_solve(ds, truth, lsf=lsf, tau=tau, eta=eta)
        out[label] = instrumental(d_hat, lsf)
        if lsf:
            out["albireo deconvolved, ML-II weights"] = d_hat[:, EDGE:-EDGE]
    fd3 = read_fd3(case_dir)
    if fd3 is not None:
        out["fd3"] = fd3
    obs, shifts = fb.shift_and_add_inputs(ds, truth)
    for n in (7, 50):
        out[f"shift-and-add, {n} sweeps"] = disentangle(obs, shifts, n_iter=n) / fb.ELL[:, None]
    out["shift-and-add, limit"] = limit(obs) / fb.ELL[:, None]
    return out, weights


def score(outputs: dict, truth) -> dict:
    """Core RMS (raw, aligned) per component: the deconvolved rows against d, the rest B d."""
    d = np.stack([np.asarray(c) for c in truth.components])[:, EDGE:-EDGE]
    bd = np.stack([fb.lsf_convolve(c) for c in truth.components])[:, EDGE:-EDGE]
    return {
        label: [fb.core_rms(rec[i], (d if "deconvolved" in label else bd)[i]) for i in range(2)]
        for label, rec in outputs.items()
    }


# --- frequency bands ------------------------------------------------------------------------


def band_edges(shifts_pix: np.ndarray) -> tuple[float, float]:
    """The low band (difference-mode information below one half, docs/math.md 5.1) and the
    LSF cutoff (transfer one half), in cycles per pixel."""
    delta = shifts_pix[0] - shifts_pix[1]
    f = np.linspace(0.0, 0.5, 200001)
    info = 1.0 - np.abs(np.exp(2j * np.pi * np.outer(f, delta)).mean(axis=1))
    f_low = float(f[np.argmax(info >= 0.5)])
    f_lsf = float(np.sqrt(np.log(2.0) / 2.0) / (np.pi * fb.LSF_V / fb.GRID.dv_kms))
    return f_low, f_lsf


def band_rms(error: np.ndarray, core: np.ndarray, f_low: float, f_lsf: float) -> list[float]:
    """Aligned core RMS of the error: total, low band, line band, above the LSF cutoff."""

    def lowpass(e, f_edge):
        return gaussian_filter1d(e, np.sqrt(np.log(2.0) / 2.0) / (np.pi * f_edge), mode="reflect")

    low = lowpass(error, f_low)
    middle = lowpass(error, f_lsf) - low
    high = error - low - middle

    def aligned(x):
        v = x[core]
        return float(np.sqrt(np.mean((v - v.mean()) ** 2)))

    return [aligned(x) for x in (error, low, middle, high)]


# --- output -----------------------------------------------------------------------------


def print_scores(title: str, scores: dict) -> None:
    print(f"\n{title}\n  {'':36s} comp 1 raw / aligned   comp 2 raw / aligned")
    for label, ((r1, a1), (r2, a2)) in scores.items():
        print(f"  {label:36s} {r1:.4f} / {a1:.4f}        {r2:.4f} / {a2:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd3", default=None, help="command that runs fd3 (split as a shell would)")
    ap.add_argument("--outdir", default="fd3_bench_checks_out")
    ap.add_argument("--seeds", type=int, default=20, help="noise seeds 1..N (7 is the benchmark)")
    ap.add_argument("--time", action="store_true", help="also time shift-and-add per sweep count")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    fd3_command = shlex.split(args.fd3) if args.fd3 else None
    t_start = time.time()

    ds7, truth7, tperi7 = fb.simulate(7)
    obs7, shifts7 = fb.shift_and_add_inputs(ds7, truth7)
    limit = ShiftAndAddLimit(shifts7, obs7.shape[1])

    # 1-2. Every seed, every code.
    per_seed, weights = {}, {}
    for seed in sorted({*range(1, args.seeds + 1), 7}):
        ds, truth, tperi = fb.simulate(seed)
        outputs, weights[seed] = run_case(
            ds, truth, tperi, outdir / f"seed{seed:02d}", fd3_command, limit
        )
        per_seed[seed] = score(outputs, truth)
        if seed == 7:
            outputs7 = outputs
        print(f"seed {seed} done ({time.time() - t_start:.0f} s)", flush=True)
    print_scores(
        "Seed 7, the recorded benchmark: core RMS against B d (deconvolved rows: against d)",
        per_seed[7],
    )

    labels = list(per_seed[7])
    al = {k: np.array([[per_seed[s][k][i][1] for i in range(2)] for s in per_seed]) for k in labels}
    print(f"\n{len(per_seed)} noise seeds: aligned core RMS, mean and standard deviation")
    for k in labels:
        v = al[k]
        print(
            f"  {k:36s} {v[:, 0].mean():.4f} +- {v[:, 0].std(ddof=1):.4f}   "
            f"{v[:, 1].mean():.4f} +- {v[:, 1].std(ddof=1):.4f}"
        )
    if "fd3" in al:
        print("\nAligned error relative to fd3, per seed: mean [min, max]")
        for k in labels:
            if "deconvolved" in k or k == "fd3":
                continue
            r = al[k] / al["fd3"]
            print(
                f"  {k:36s} {r[:, 0].mean():.2f} [{r[:, 0].min():.2f}, {r[:, 0].max():.2f}]   "
                f"{r[:, 1].mean():.2f} [{r[:, 1].min():.2f}, {r[:, 1].max():.2f}]"
            )
    base = al[ALBIREO[0]]
    print(f"\nSeeds on which '{ALBIREO[0]}' has the lower aligned error on both components")
    for k in labels:
        if "deconvolved" in k or k == ALBIREO[0]:
            continue
        lower = int(np.sum(np.all(base < al[k], axis=1)))
        print(f"  than {k:36s} {lower} of {len(per_seed)}")
    print("\nML-II weights over the seeds: mean [min, max]")
    for label in ALBIREO_MLII:
        tau = np.array([weights[s][label][0] for s in weights])
        eta = np.array([weights[s][label][1] for s in weights])
        print(
            f"  {label}: tau {np.round(tau.mean(0))} "
            f"[{np.round(tau.min(0))}, {np.round(tau.max(0))}], "
            f"eta {np.round(eta.mean(0), 2)} [{np.round(eta.min(0), 2)}, {np.round(eta.max(0), 2)}]"
        )

    # 3. Frequency bands, seed 7.
    f_low, f_lsf = band_edges(shifts7)
    print(
        f"\nFrequency bands (seed 7): low below {f_low:.5f} cycles/px (periods above "
        f"{1 / f_low:.0f} px, {fb.GRID.dv_kms / f_low:.0f} km/s), where the epochs give the "
        f"difference mode less than half its information; the LSF cutoff at {f_lsf:.4f} "
        f"(period {1 / f_lsf:.1f} px)\n  aligned core RMS against B d: total | low | lines | "
        "above the LSF cutoff"
    )
    bd7 = np.stack([fb.lsf_convolve(c) for c in truth7.components])[:, EDGE:-EDGE]
    for label, rec in outputs7.items():
        if "deconvolved" in label:
            continue
        parts = [band_rms(rec[i] - bd7[i], fb.core_mask(bd7[i]), f_low, f_lsf) for i in range(2)]
        print(
            f"  {label:36s} "
            + "   ".join(" | ".join(f"{p:.4f}" for p in parts[i]) for i in range(2))
        )

    # 4. Bias: the noiseless copy of seed 7.
    outputs0, _ = run_case(
        noiseless(ds7, truth7), truth7, tperi7, outdir / "noiseless07", fd3_command, limit
    )
    print_scores(
        "Noiseless seed 7 (the bias): core RMS against B d (deconvolved rows: against d)",
        score(outputs0, truth7),
    )

    # 5. Shift-and-add against the number of sweeps, seed 7.
    d7 = np.stack([np.asarray(c) for c in truth7.components])[:, EDGE:-EDGE]
    lim7 = limit(obs7) / fb.ELL[:, None]
    print(
        "\nShift-and-add, seed 7: core RMS against B d (raw / aligned), aligned against d, "
        "and max |iterate - limit|"
    )
    for n in (*SWEEPS, "limit"):
        rec = lim7 if n == "limit" else disentangle(obs7, shifts7, n_iter=n) / fb.ELL[:, None]
        b = [fb.core_rms(rec[i], bd7[i]) for i in range(2)]
        a = [fb.core_rms(rec[i], d7[i])[1] for i in range(2)]
        gap = float(np.abs(rec - lim7).max())
        print(
            f"  {n!s:>6}  {b[0][0]:.4f} / {b[0][1]:.4f}   {b[1][0]:.4f} / {b[1][1]:.4f}   "
            f"against d {a[0]:.4f} / {a[1]:.4f}   {gap:.1e}"
        )

    if args.time:
        print("\nShift-and-add wall time, fresh process (min of 5 after a warmup; 20,000 once)")
        for n in SWEEPS[:-1]:
            print(
                f"  {n:6d} sweeps: {fb._time_in_fresh_process(obs7, shifts7, n):.4f} s", flush=True
            )
        print(f"  {SWEEPS[-1]:6d} sweeps: {time_once(obs7, shifts7, SWEEPS[-1]):.1f} s")
    print(f"\ntotal {time.time() - t_start:.0f} s")


def time_once(obs: np.ndarray, shifts_pix: np.ndarray, n_iter: int) -> float:
    """One timed call of ``disentangle`` in a fresh interpreter, after a 7-sweep warmup."""
    child = (
        "import sys, time\n"
        "import numpy as np\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from shift_and_add import disentangle\n"
        "d = np.load(sys.argv[1])\n"
        "disentangle(d['obs'], d['shifts'], n_iter=7)\n"
        "t0 = time.perf_counter()\n"
        "disentangle(d['obs'], d['shifts'], n_iter=int(d['n_iter']))\n"
        "print(time.perf_counter() - t0)\n"
    )
    with tempfile.TemporaryDirectory() as td:
        payload = Path(td) / "saa_timing.npz"
        np.savez(payload, obs=obs, shifts=shifts_pix, n_iter=n_iter)
        out = subprocess.run(
            [sys.executable, "-c", child, str(payload), str(Path(__file__).resolve().parent)],
            capture_output=True,
            text=True,
            check=True,
        )
    return float(out.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    main()
