"""AI Phoenicis: the three disentangling codes on real archival spectra.

The simulated benchmark in ``fd3_bench.py`` compares albireo, fd3 and shift-and-add against
an injected truth spectrum. This script runs the comparison on real data, where no truth
spectrum exists but the orbit is known to a precision no disentangling code approaches.

AI Phe (HD 6980) eclipses, so its geometry is pinned by the TESS light curve, and its
spectroscopic semi-amplitudes are published from several independent studies agreeing to
0.1 per cent:

    K1 = 51.164 +/- 0.007 km/s      K2 = 49.106 +/- 0.010 km/s
    P  = 24.5924 d                  e  = 0.1878 +/- 0.0006
    omega = 110.30 +/- 0.06 deg     T0 = BJD_TDB 2458362.82847  (primary eclipse)
    -- Maxted et al. (2020), MNRAS 498, 332, Tables 2-3

The external ground truth is therefore the orbit rather than the component spectra: recover
K1 and K2 from 36 archival HARPS spectra and compare against values good to 0.014 and
0.020 per cent.

The light ratio is not supplied by the eclipse. The eclipse pins the fractional radii, the
inclination and the surface-brightness ratio in the photometric band, TESS, centred near
7860 A. The light ratio in the spectroscopic window is a different quantity and has to be
computed from the radii and the two temperatures. It is also strongly wavelength dependent:
for AI Phe's 6310 K + 5010 K pair with R2/R1 = 1.624, a blackbody estimate gives l2 = 0.375
at 4000 A and 0.510 at 6500 A. The light ratio is far better constrained here than for a
non-eclipsing system, but quoting the TESS-band value at 5200 A would be a 10 per cent error
in the quantity every recovered line depth scales by.

Data: 36 HARPS spectra (ESO programme archive, R = 115,000, 3782-6913 A), SNR 41-129,
covering all ten phase bins. Fetched with :mod:`albireo.archive`.

Run:  python scripts/aiphe_bench.py --data DIR [--fd3 PATH] [--fit]
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import albireo as ab
from albireo.forward import build_problem
from albireo.kepler import t_peri_from_t_conj
from albireo.likelihood import marginal_loglikelihood
from albireo.priors import SmoothnessPrior

# --- the published system (Maxted et al. 2020, MNRAS 498, 332) ---------------
P_PUB = 24.5924
K1_PUB, K1_ERR = 51.164, 0.007
K2_PUB, K2_ERR = 49.106, 0.010
ECC_PUB, ECC_ERR = 0.1878, 0.0006
OMEGA_PUB_DEG = 110.30
T0_PUB = 2458362.82847  # BJD_TDB, primary eclipse
TEFF1, TEFF2 = 6310.0, 5010.0
R1_FRAC, R2_FRAC = 0.037724, 0.061253  # r = R/a, run C

# --- the analysis window -----------------------------------------------------
# 5150-5250 A: metal-rich, no telluric bands (those start beyond ~6270 A), and it
# carries the Mg I b triplet, which both an F7 V and a K0 IV show strongly.
WINDOW = (5150.0, 5250.0)
DV_KMS = 0.8  # native HARPS sampling is 0.577 km/s; the LSF sigma is 1.107
LSF_SIGMA_V = 299792.458 / 115000.0 / 2.3548  # R = 115,000
TAU, ETA = 300.0, 5.0


def light_fractions(wave_angstrom: float) -> np.ndarray:
    """Blackbody light ratio at ``wave_angstrom`` from the published radii and Teffs.

    An estimate, not a measurement: real stars are not blackbodies, and the error on this
    is several per cent. Every recovered line depth scales as 1/l_i, so the value is an
    assumption the results inherit; ``scripts/m5_light_ratio_demo.py`` quantifies that
    systematic.
    """
    h, c, kb = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    lam = wave_angstrom * 1e-10

    def planck(t):
        return 1.0 / lam**5 / (np.exp(h * c / (lam * kb * t)) - 1.0)

    ratio = (R2_FRAC / R1_FRAC) ** 2 * planck(TEFF2) / planck(TEFF1)
    l2 = ratio / (1.0 + ratio)
    return np.array([1.0 - l2, l2])


def load(data_dir: Path):
    """Read the archival spectra into a Dataset on the analysis window."""
    ds = ab.read_dataset(
        str(data_dir / "*.fits"),
        region=WINDOW,
        region_pad_angstrom=3.0,
        smooth_angstrom=25.0,
    )
    return ds


def published_velocities(bjd: np.ndarray) -> np.ndarray:
    """Radial velocities of both components at ``bjd`` from the published orbit.

    The systemic velocity is absent: albireo's gamma is identically zero (D14), because a
    common shift of both components is exactly degenerate with translating the component
    spectra. The disentangling does not see it.
    """
    omega = np.radians(OMEGA_PUB_DEG)
    tperi = float(t_peri_from_t_conj(T0_PUB, period=P_PUB, ecc=ECC_PUB, omega=omega))
    t = np.asarray(bjd)
    common = dict(period=P_PUB, t_peri=tperi, ecc=ECC_PUB)
    # Component 2 moves in antiphase: omega + pi, which is albireo's own convention for
    # odd-indexed stellar components.
    v1 = np.asarray(ab.radial_velocity(t, omega=omega, k=K1_PUB, **common))
    v2 = np.asarray(ab.radial_velocity(t, omega=omega + np.pi, k=K2_PUB, **common))
    return np.stack([v1, v2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory of HARPS .fits spectra")
    ap.add_argument("--fd3", default=None, help="path to an fd3 binary")
    ap.add_argument("--fit", action="store_true", help="also fit K1/K2 and test the orbit")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument(
        "--window",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="analysis window in angstrom; the recovered orbit's dependence on this is "
        "itself a measurement (see the module docstring)",
    )
    args = ap.parse_args()
    if args.window:
        global WINDOW
        WINDOW = tuple(args.window)

    ell = light_fractions(np.mean(WINDOW))
    print(f"albireo {ab.__version__} — AI Phoenicis (HD 6980), real HARPS spectra")
    print(f"window {WINDOW[0]:.0f}-{WINDOW[1]:.0f} A, dv = {DV_KMS} km/s")
    print(f"light fractions at {np.mean(WINDOW):.0f} A: {ell[0]:.4f} / {ell[1]:.4f} (blackbody)")

    ds = load(Path(args.data))
    print(f"\n{ds.summary()}")

    grid = ab.LogGrid.covering(ds, dv_kms=DV_KMS, v_margin_kms=140.0, lsf_sigma_kms=LSF_SIGMA_V)
    vel = published_velocities(np.asarray(ds.bjd))
    print(f"model grid: {grid.n} pixels")
    print(
        f"published RVs: comp 1 {vel[0].min():+.1f}..{vel[0].max():+.1f} km/s, "
        f"comp 2 {vel[1].min():+.1f}..{vel[1].max():+.1f}"
    )
    print(f"max separation |v1 - v2| = {np.abs(vel[0] - vel[1]).max():.1f} km/s")

    # ---- albireo, orbit fixed at the published solution ----------------------
    problem = build_problem(
        grid,
        ds,
        velocities=vel,
        light_fractions=ell,
        lsf_sigma_v={name: LSF_SIGMA_V for name in ds.instruments},
    )
    prior = SmoothnessPrior(jnp.full(2, TAU), jnp.full(2, ETA))
    t0 = time.perf_counter()
    result = marginal_loglikelihood(problem, prior)
    d_hat = np.asarray(result.d_hat)
    wall = time.perf_counter() - t0
    std = np.asarray(ab.spectra_std(result))
    z = np.asarray(ab.data_residual_zscores(problem, result.d_hat))
    print(f"\nalbireo (orbit fixed at published): {wall:.2f} s")
    print(f"  residual z-RMS {np.sqrt(np.mean(z**2)):.4f}  (1.0 = the noise model is right)")
    print(f"  median band    {np.median(std):.5f} in normalized flux")
    print(f"  deepest line   comp 1 {d_hat[0].min():.4f}, comp 2 {d_hat[1].min():.4f}")

    # ---- shift-and-add on the same data --------------------------------------
    from shift_and_add import disentangle

    shifts = np.asarray(
        [[float(ab.log_doppler_shift(v) / grid.dx) for v in vel[i]] for i in range(2)]
    )
    obs = np.stack(
        [
            np.interp(np.asarray(grid.wave), np.asarray(ep.wave), np.asarray(ep.flux)) - 1.0
            for ep in ds
        ]
    )
    t0 = time.perf_counter()
    sa = disentangle(obs, shifts, n_iter=7)
    sa_wall = time.perf_counter() - t0
    print(f"\nshift-and-add (7 sweeps): {sa_wall:.3f} s")
    print(
        f"  deepest line   comp 1 {(sa[0] / ell[0]).min():.4f}, comp 2 {(sa[1] / ell[1]).min():.4f}"
    )

    # Agreement between the two reconstructions in the line cores.
    core = d_hat[0] < -0.10
    interior = np.zeros_like(core)
    interior[grid.n // 12 : -grid.n // 12] = True
    core &= interior
    for i in range(2):
        diff = (sa[i] / ell[i] - d_hat[i])[core]
        print(
            f"  comp {i + 1} vs albireo in line cores: RMS {np.sqrt(np.mean(diff**2)):.4f} "
            f"(mean-aligned {np.std(diff):.4f}, {core.sum()} px)"
        )

    # ---- the orbit, which carries the external ground truth -------------------
    if args.fit:
        print("\n--- fitting the orbit, and holding it against the published values ---")
        model = ab.MarginalOrbitModel(
            grid,
            ds,
            light_fractions=ell,
            lsf_sigma_v={name: LSF_SIGMA_V for name in ds.instruments},
            v_rel_max_kms=140.0,
        )
        omega = np.radians(OMEGA_PUB_DEG)
        priors = {
            "period": dist.Uniform(P_PUB - 0.01, P_PUB + 0.01),
            "t_conj": dist.Uniform(T0_PUB - 0.05, T0_PUB + 0.05),
            "secosw": dist.Uniform(-0.8, 0.8),
            "sesinw": dist.Uniform(-0.8, 0.8),
            "k": dist.Uniform(jnp.array([30.0, 30.0]), jnp.array([70.0, 70.0])),
            "log_tau": dist.Normal(jnp.log(TAU) * jnp.ones(2), 3.0),
            "log_eta": dist.Normal(jnp.log(ETA) * jnp.ones(2), 3.0),
        }
        init = {
            "period": P_PUB,
            "t_conj": T0_PUB,
            # Start away from the published solution, so agreement is not an echo of the init.
            "secosw": float(np.sqrt(ECC_PUB) * np.cos(omega)) * 0.85,
            "sesinw": float(np.sqrt(ECC_PUB) * np.sin(omega)) * 0.85,
            "k": jnp.array([K1_PUB * 0.92, K2_PUB * 1.08]),
            "log_tau": jnp.log(TAU) * jnp.ones(2),
            "log_eta": jnp.log(ETA) * jnp.ones(2),
        }
        t0 = time.perf_counter()
        fit = ab.run_map(model.model(priors), init=init, max_steps=args.steps)
        print(f"  {fit.num_steps} L-BFGS steps in {time.perf_counter() - t0:.1f} s")
        # A fit that stopped at max_steps has not found the optimum, so convergence is
        # printed: semi-amplitudes from an unconverged fit cannot be compared with a
        # published value good to 0.02%.
        print(f"  converged: {bool(fit.converged)}   |grad| = {float(fit.grad_norm):.3e}")
        if not bool(fit.converged):
            print("  *** NOT CONVERGED — the numbers below are a waypoint, not a result ***")

        k = np.asarray(fit.params["k"])
        sec, ses = float(fit.params["secosw"]), float(fit.params["sesinw"])
        ecc = sec**2 + ses**2
        print(f"\n  {'':10s} {'albireo':>12s} {'published':>14s} {'difference':>14s}")
        print(
            f"  {'K1':10s} {k[0]:12.3f} {K1_PUB:>9.3f} ± {K1_ERR:.3f} "
            f"{k[0] - K1_PUB:>+10.3f} ({(k[0] - K1_PUB) / K1_PUB * 100:+.2f}%)"
        )
        print(
            f"  {'K2':10s} {k[1]:12.3f} {K2_PUB:>9.3f} ± {K2_ERR:.3f} "
            f"{k[1] - K2_PUB:>+10.3f} ({(k[1] - K2_PUB) / K2_PUB * 100:+.2f}%)"
        )
        print(f"  {'ecc':10s} {ecc:12.4f} {ECC_PUB:>9.4f} ± {ECC_ERR:.4f} {ecc - ECC_PUB:>+10.4f}")
        print(
            f"  {'q=K1/K2':10s} {k[0] / k[1]:12.4f} {K1_PUB / K2_PUB:>14.4f} "
            f"{k[0] / k[1] - K1_PUB / K2_PUB:>+10.4f}"
        )

    if args.fd3:
        print(f"\n(fd3 at {args.fd3} — export path is scripts/fd3_bench.py; not run here)")
        subprocess.run([args.fd3, "--version"], check=False)


if __name__ == "__main__":
    main()
