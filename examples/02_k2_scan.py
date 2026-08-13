"""Find a hidden companion with the K2 scan: SB1 + faint secondary (``docs/math.md`` §6).

The dormant-compact-object / faint-companion workflow. You already have an SB1
solution — period, conjunction time, eccentricity vector and ``K_1`` from the
single-lined orbit — and you want to know whether a second set of lines is buried in
the composite spectra at some unknown ``K_2``.

At each trial ``K_2`` the companion's deviation spectrum is a *linear* component, so it
marginalizes analytically: one linear solve per grid point buys the optimal matched
filter integrated over every possible companion spectrum, with no template library.
The detection statistic is

    D(K_2) = 2 [ log p(y | K_2) - log p(y | no companion) ].

This script runs the scan twice with everything else held fixed: once on a dataset
with a companion injected at K_2 = 38 km/s, and once on a companion-free dataset. The
second run is the point. Because both marginal likelihoods carry their ``1/2 log det``
Occam terms, the extra marginalized component *costs* likelihood unless coherent signal
pays for it, so on companion-free data D is negative at every trial — the honest
baseline that makes a positive D mean something.

Two caveats the package will not paper over:

* ``D`` is **not** asymptotically chi-squared. Its null distribution depends on the
  companion's prior scale ``(tau_2, eta_2)`` and must be calibrated by injection-
  recovery with :mod:`albireo.simulate` (``docs/math.md`` §6). Absolute D values here
  are illustrative; the contrast between peak, edges, and null is the message.
* ``ell_2`` is a *choice*, not a fit. The observable is ``ell_2 * d_2``, so the
  companion's light fraction trades exactly against its line depths (``docs/math.md``
  §5.2), and at ``ell_2 = 0.1`` the recovered spectrum's smooth envelope is
  prior-dominated by a factor ~ ell_1/ell_2. The line *pattern* is what the scan
  recovers, not an absolute depth scale.

Environment
-----------
ALBIREO_EXAMPLE_FAST=1
    CI-sized run: 10 epochs and a 4 km/s K_2 grid. Unset (the default) gives 12 epochs
    and a 2 km/s grid. The injected K_2 = 38 km/s lands exactly on both grids.

Usage
-----
    python examples/02_k2_scan.py
"""

from __future__ import annotations

import importlib.util
import os
import time

import jax.numpy as jnp
import numpy as np

import albireo as ab

FAST = bool(os.environ.get("ALBIREO_EXAMPLE_FAST"))

# --- the system: a bright primary and a faint, never-directly-seen companion ------
GRID = ab.LogGrid.from_wavelength_range(5000.0, 5045.0, dv_kms=5.5)
P_TRUE = 6.31  # orbital period [d]
ECC_TRUE = 0.15
OMEGA_TRUE = 0.70  # argument of periastron of the primary [rad]
T_PERI_TRUE = 2.0
K1_TRUE = 12.0  # primary semi-amplitude [km/s] -- known from the SB1 solution
K2_TRUE = 38.0  # companion semi-amplitude [km/s] -- what the scan must find
ELL = (0.9, 0.1)  # (ell_1, ell_2): a CHOICE, see the module docstring
LSF = {"a": 7.0}
SNR = 150.0
N_EPOCHS = 10 if FAST else 12
K2_STEP = 4.0 if FAST else 2.0
K2_GRID = np.arange(10.0, 70.0, K2_STEP)
SEED = 7

# The scan holds the spectral hyperparameters fixed (it is a profile over K_2, not a
# joint fit): a stiffer, shallower prior for the faint companion than for the primary.
PRIOR = ab.SmoothnessPrior(jnp.asarray([300.0, 30.0]), jnp.asarray([5.0, 5.0]))

# (K_1 + max K_2)(1 + e) with headroom -- the static solver bandwidth for the
# two-component model. The null (one-component) model inherits it and needs less.
V_REL_MAX = 105.0


def sb1_solution() -> dict:
    """The fixed SB1 orbit, in albireo's (period, t_conj, secosw, sesinw) parameterization.

    ``k1`` is passed separately to :func:`albireo.k2_scan` — the scan profiles over the
    companion's semi-amplitude only, with everything else nailed down by the SB1 fit.
    """
    nu_conj = 0.5 * np.pi - OMEGA_TRUE
    e_conj = 2.0 * np.arctan2(
        np.sqrt(1.0 - ECC_TRUE) * np.sin(0.5 * nu_conj),
        np.sqrt(1.0 + ECC_TRUE) * np.cos(0.5 * nu_conj),
    )
    t_conj = T_PERI_TRUE + (e_conj - ECC_TRUE * np.sin(e_conj)) * P_TRUE / (2.0 * np.pi)
    return {
        "period": P_TRUE,
        "t_conj": t_conj,
        "secosw": np.sqrt(ECC_TRUE) * np.cos(OMEGA_TRUE),
        "sesinw": np.sqrt(ECC_TRUE) * np.sin(OMEGA_TRUE),
    }


def simulate(*, with_companion: bool):
    """Two datasets from one generator: with the companion injected, and without it."""
    rng = np.random.default_rng(SEED)
    bjd = np.sort(rng.uniform(0.0, 21.0, size=N_EPOCHS))
    primary = ab.synthetic_deviation_spectrum(GRID, seed=21)
    companion = ab.synthetic_deviation_spectrum(GRID, seed=22)
    instruments = {
        "a": ab.InstrumentSpec(wave=np.arange(5003.0, 5042.0, 0.11), sigma_v_lsf=LSF["a"], snr=SNR)
    }
    if with_companion:
        orbit = ab.OrbitParams(
            period=P_TRUE,
            t_peri=T_PERI_TRUE,
            ecc=ECC_TRUE,
            omega=OMEGA_TRUE,
            k=(K1_TRUE, K2_TRUE),
        )
        components, light = [primary, companion], ELL
    else:
        # Identical primary, identical epochs, identical noise seed -- only the
        # companion is removed, so any difference in D is the companion's doing.
        orbit = ab.OrbitParams(
            period=P_TRUE, t_peri=T_PERI_TRUE, ecc=ECC_TRUE, omega=OMEGA_TRUE, k=(K1_TRUE,)
        )
        components, light = [primary], (1.0,)
    dataset, _ = ab.simulate_dataset(
        GRID,
        components,
        bjd=bjd,
        instruments=instruments,
        light_fractions=light,
        orbit=orbit,
        seed=5,
    )
    return dataset, companion


def run_scan(dataset) -> ab.K2ScanResult:
    """One detection scan over ``K2_GRID`` at the fixed SB1 solution."""
    return ab.k2_scan(
        GRID,
        dataset,
        orbit=sb1_solution(),
        k1=K1_TRUE,
        k2_grid=K2_GRID,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        prior=PRIOR,
        v_rel_max_kms=V_REL_MAX,
    )


def print_curve(result: ab.K2ScanResult, label: str) -> None:
    """Compact summary of a detection curve: peak, edges, and the shape around the peak."""
    detection = np.asarray(result.detection)
    edge = max(float(detection[0]), float(detection[-1]))
    print(f"\n--- {label} ---")
    print(f"  peak at K_2 = {result.k2_peak:.1f} km/s   D(peak) = {result.detection_peak:12.1f}")
    print(f"  worst scan edge          D(edge) = {edge:12.1f}")
    print(f"  contrast peak - edge             = {result.detection_peak - edge:12.1f}")
    print(f"  log p(y | no companion)          = {result.log_likelihood_null:12.1f}")
    top = np.argsort(detection)[::-1][:5]
    ranked = "  ".join(f"{result.k2_grid[i]:.0f}:{detection[i]:.0f}" for i in sorted(top))
    print(f"  five highest trials (K_2 : D)    = {ranked}")


def plot_detection(injected: ab.K2ScanResult, null: ab.K2ScanResult, path: str) -> None:
    """D(K_2) for both datasets on one axis, with the injected value marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True, constrained_layout=True)
    ab.plot_detection(injected, injected_k2=K2_TRUE, ax=axes[0])
    axes[0].set_title("Companion injected at $K_2$ = 38 km/s, $\\ell_2$ = 0.1")

    ab.plot_detection(null, ax=axes[1])
    axes[1].set_title("No companion: the Occam term keeps $D$ below zero everywhere")
    for ax in axes[:-1]:
        ax.set_xlabel("")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"albireo {ab.__version__} | fast mode: {FAST} | grid: {GRID.n} px @ {GRID.dv_kms:.2f}")
    print(f"K_2 trials: {K2_GRID.size} from {K2_GRID[0]:.0f} to {K2_GRID[-1]:.0f} km/s")
    t_start = time.perf_counter()

    # 1. The positive: a companion really is there ---------------------------------
    dataset, companion_truth = simulate(with_companion=True)
    print(dataset.summary())
    t0 = time.perf_counter()
    injected = run_scan(dataset)
    t_injected = time.perf_counter() - t0
    print_curve(injected, f"companion injected at K_2 = {K2_TRUE:.0f} km/s  [{t_injected:.1f} s]")

    # The recovered companion spectrum comes free from the conditional Gaussian at the
    # peak. Compare the line *pattern*: the mean level is prior-set (math.md 5.2, 6).
    core = np.asarray(companion_truth) < -0.05
    margin = int(0.05 * GRID.n)  # ignore the zero-padded grid edges
    core[:margin] = False
    core[-margin:] = False
    err = np.asarray(injected.companion)[core] - np.asarray(companion_truth)[core]
    offset = float(err.mean())
    corr = float(
        np.corrcoef(np.asarray(injected.companion)[core], np.asarray(companion_truth)[core])[0, 1]
    )
    print(
        f"  companion spectrum at the peak: r = {corr:.3f} with truth over line cores, "
        f"offset {offset:+.3f}, RMS about the offset {np.sqrt(np.mean((err - offset) ** 2)):.3f}"
    )
    print(
        f"  median formal sigma on d_2: {float(np.median(np.asarray(injected.companion_std))):.3f}"
    )

    # 2. The negative control: same everything, no companion ------------------------
    null_dataset, _ = simulate(with_companion=False)
    t0 = time.perf_counter()
    null = run_scan(null_dataset)
    t_null = time.perf_counter() - t0
    print_curve(null, f"companion-free control  [{t_null:.1f} s]")
    print(
        f"  max over the whole grid: D = {float(np.max(null.detection)):.1f} "
        "(negative everywhere -- nothing to detect, and the Occam term says so)"
    )

    # 3. Figure, only if matplotlib happens to be installed --------------------------
    if importlib.util.find_spec("matplotlib") is not None:
        plot_detection(injected, null, "k2_scan_detection.png")
        print("\nwrote k2_scan_detection.png")
    else:
        print("\nmatplotlib not installed - skipping the figure (it is not a dependency)")

    # 4. The gate --------------------------------------------------------------------
    assert injected.k2_peak == K2_TRUE, (
        f"peak at K_2 = {injected.k2_peak} km/s, injected {K2_TRUE} km/s"
    )
    edge = max(float(injected.detection[0]), float(injected.detection[-1]))
    assert injected.detection_peak - edge > 1e3, "peak is not decisively above the scan edges"
    null_max = float(np.max(null.detection))
    assert null_max < 0.0, f"companion-free scan reached D = {null_max:.1f} >= 0"
    print(
        f"\nOK - peak on the injected K_2, null curve strictly negative. "
        f"Total wall: {time.perf_counter() - t_start:.1f} s"
    )


if __name__ == "__main__":
    main()
