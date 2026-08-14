"""Nebular contamination: leave it in, mask it out, or model it (``docs/math.md`` §1.3).

Massive stars are born in H II regions, so their spectra carry emission lines that
belong to neither star: they sit still while the stars move, and they change strength
from night to night with seeing, slit losses and sky subtraction. This script takes one
simulated SB2 whose H-beta absorption carries exactly such a line and disentangles it
three ways, with everything else — data, orbit, priors, grid — held fixed:

1. **Leave it in.** Two stellar components and nothing else. The emission has to go
   somewhere, and where it goes is into the stars: it fills the line core, so the
   disentangled profile comes out too shallow and too narrow.
2. **Mask it.** Zero the inverse variance across the nebular window
   (:func:`albireo.mask_ranges`). This is what the literature does, and it is honest —
   but it deletes the core of the very line a Balmer gravity diagnostic is measured
   from, so what comes back has a hole in the middle and the wings alone carry the
   answer.
3. **Model it.** A third component at rest in the barycentric frame with a free
   per-epoch amplitude (D40), confined by the prior to the nebular window
   (:func:`albireo.window_profile`). The stellar spectra come back uncontaminated *and*
   complete, and the nebular line comes back as its own measured product.

The orbit is held at truth throughout, because the claim being demonstrated is about
the spectra, not about whether the orbit survives (``tests/test_nebular.py`` covers the
joint fit). The number to watch is the **equivalent width**: it is what reaches an
atmosphere code, so an error there is an error in the temperature and gravity that come
out the far end — the one nobody currently propagates.

Two honest notes. The amplitude scale is a convention: only the product
``amplitude * spectrum`` is observable, so the recovered amplitudes are compared after
centering (:func:`albireo.nebular_amplitudes` pins their geometric mean to 1). And the
nebular velocity is not identified either — it decides where the component's lines land
on the model grid, which is what the prior windows have to agree with, and nothing else.

Environment
-----------
ALBIREO_EXAMPLE_FAST=1
    CI-sized run: 8 epochs instead of 14. The comparison is unchanged; the numbers move
    slightly.

Usage
-----
    python examples/04_nebular.py
"""

from __future__ import annotations

import importlib.util
import os
import time

import numpy as np

import albireo as ab

FAST = bool(os.environ.get("ALBIREO_EXAMPLE_FAST"))

# --- the system: an SB2 in an H II region ------------------------------------------
HBETA = 4861.33  # air wavelength [A]
GRID = ab.LogGrid.from_wavelength_range(4838.0, 4886.0, dv_kms=5.5)
P_TRUE, T_PERI_TRUE = 5.7, 0.0
K_TRUE = (58.0, 41.0)  # semi-amplitudes [km/s]
ELL = np.array([0.7, 0.3])  # light fractions
LSF = {"vlt": 7.0}  # Gaussian LSF width [km/s]
SNR = 220.0
N_EPOCHS = 8 if FAST else 14
SEED = 5

# The *prior* window: generous on purpose. Confinement is soft, so too wide only returns
# some of the freedom the profile takes away, while too narrow clips real emission and
# pushes the residual back into the stars — the failure being fixed.
WINDOWS = ab.nebular_windows(lines=[HBETA], halfwidth_kms=500.0)
# The *mask* window, for treatment 2. Deliberately tighter than the prior window, because
# a mask costs pixels outright and nobody masking a nebular line by hand would be as
# generous as a soft prior can afford to be. +-150 km/s is ~2.4 A at H-beta.
MASK = ab.nebular_windows(lines=[HBETA], halfwidth_kms=150.0)

# (K_1 + K_2) plus the nebula at rest, with headroom for the static solver bandwidth.
V_REL_MAX = 150.0


def stellar_components() -> list[np.ndarray]:
    """Two stars: a broad H-beta absorption each, plus a scatter of metal lines."""
    px = np.arange(GRID.n, dtype=np.float64)
    center = float(np.interp(HBETA, GRID.wave, px))

    def balmer(depth: float, sigma_kms: float) -> np.ndarray:
        return depth * np.exp(-0.5 * ((px - center) / (sigma_kms / GRID.dv_kms)) ** 2)

    primary = balmer(-0.55, 95.0) + ab.synthetic_deviation_spectrum(
        GRID, n_lines=6, depth_range=(0.03, 0.12), sigma_v_range=(12.0, 25.0), seed=11
    )
    secondary = balmer(-0.40, 70.0) + ab.synthetic_deviation_spectrum(
        GRID, n_lines=5, depth_range=(0.03, 0.10), sigma_v_range=(12.0, 25.0), seed=12
    )
    return [np.maximum(primary, -0.95), np.maximum(secondary, -0.95)]


def simulate():
    """One dataset with a nebular H-beta line whose strength varies per epoch."""
    rng = np.random.default_rng(SEED)
    bjd = np.sort(rng.uniform(0.0, 2.0 * P_TRUE, N_EPOCHS))
    orbit = ab.OrbitParams(period=P_TRUE, t_peri=T_PERI_TRUE, ecc=0.0, omega=0.0, k=K_TRUE)
    nebular = ab.synthetic_nebular_spectrum(
        GRID, lines=[HBETA], amplitude_range=(0.45, 0.45), sigma_v_kms=12.0, seed=3
    )
    return ab.simulate_dataset(
        GRID,
        stellar_components(),
        bjd=bjd,
        instruments={
            "vlt": ab.InstrumentSpec(wave=np.arange(4841.0, 4883.0, 0.10), sigma_v_lsf=7.0, snr=SNR)
        },
        light_fractions=ELL,
        orbit=orbit,
        v_bary=np.linspace(-24.0, 26.0, N_EPOCHS),
        frame="barycentric",
        nebular=nebular,
        nebular_amplitudes=np.exp(rng.normal(0.0, 0.28, N_EPOCHS)),
        seed=SEED,
    )


def stellar_prior(n_comp: int, *, confine: bool) -> ab.SmoothnessPrior:
    """(tau, eta) per component; the nebular entry is stiffer-free and window-confined."""
    tau = np.full(n_comp, 200.0)
    eta = np.full(n_comp, 2.0)
    eta_profile = None
    if confine:
        tau[-1] = 8.0  # the nebular line is narrow -- do not smooth it away
        eta_profile = np.ones((n_comp, GRID.n))
        eta_profile[-1] = ab.window_profile(GRID.wave, WINDOWS, inside=1.0, outside=1e6)
    return ab.SmoothnessPrior(tau, eta, None, eta_profile)


def disentangle(dataset, truth, *, nebular: bool):
    """Posterior-mean spectra at the true orbit. Returns (d_hat, log-likelihood)."""
    problem = ab.build_problem(
        GRID,
        dataset,
        velocities=truth.velocities,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        nebular=nebular,
        nebular_amplitudes=truth.nebular_amplitudes if nebular else None,
    )
    n_comp = 3 if nebular else 2
    result = ab.marginal_loglikelihood(
        problem,
        stellar_prior(n_comp, confine=nebular),
        half_bandwidth=problem.half_bandwidth_bound(V_REL_MAX),
    )
    return np.asarray(result.d_hat), float(result.log_likelihood)


def combination(d_hat) -> np.ndarray:
    """The light-weighted stellar composite, the quantity constant light fractions leave
    observable (``docs/math.md`` §5.1), with its prior-set offset removed."""
    comb = ELL @ d_hat[:2]
    interior = slice(int(0.06 * GRID.n), GRID.n - int(0.06 * GRID.n))
    return comb - np.mean(comb[interior])


def equivalent_width(comb, half_width_angstrom: float = 8.0) -> float:
    """H-beta equivalent width [A] -- the quantity an atmosphere code actually consumes.

    Measured on the offset-removed composite, because the overall level of a disentangled
    spectrum is set by the ridge rather than by the data (the low-frequency degeneracy);
    every treatment below gets the identical treatment, truth included, so the comparison
    is exact even though the absolute number carries that convention.
    """
    inside = np.abs(GRID.wave - HBETA) < half_width_angstrom
    return float(-np.sum(comb[inside] * np.gradient(GRID.wave)[inside]))


def plot(truth_comb, results, masked_range, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.0, 6.4), sharex=True, height_ratios=[2.2, 1.0], constrained_layout=True
    )
    top.axvspan(*masked_range, color="0.92", zorder=0, label="masked window (treatment 2)")
    top.plot(GRID.wave, 1.0 + truth_comb, "k-", lw=2.4, alpha=0.75, label="truth")
    for (name, comb, _), style in zip(results, ("C3--", "C1-.", "C0-"), strict=True):
        top.plot(GRID.wave, 1.0 + comb, style, lw=1.5, label=name)
    top.set_ylabel("light-weighted stellar composite")
    top.legend(loc="lower left", fontsize=9)
    top.set_title(r"Disentangled H$\beta$ under nebular contamination")

    bottom.axvspan(*masked_range, color="0.92", zorder=0)
    bottom.axhline(0.0, color="k", lw=0.8)
    for (name, comb, _), style in zip(results, ("C3--", "C1-.", "C0-"), strict=True):
        bottom.plot(GRID.wave, comb - truth_comb, style, lw=1.5, label=name)
    bottom.set_xlabel(r"wavelength [$\AA$]")
    bottom.set_ylabel("error")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    t_start = time.perf_counter()
    dataset, truth = simulate()
    truth_comb = ELL @ np.stack([np.asarray(c) for c in truth.components])
    interior = slice(int(0.06 * GRID.n), GRID.n - int(0.06 * GRID.n))
    truth_comb = truth_comb - np.mean(truth_comb[interior])
    ew_true = equivalent_width(truth_comb)

    # 1. Leave it in ---------------------------------------------------------------
    d_left, ll_left = disentangle(dataset, truth, nebular=False)

    # 2. Mask it -------------------------------------------------------------------
    lo, hi = MASK[0]
    masked = ab.Dataset(
        epochs=tuple(ab.mask_ranges(epoch, [(lo, hi)]) for epoch in dataset),
        frame=dataset.frame,
    )
    d_masked, _ = disentangle(masked, truth, nebular=False)

    # 3. Model it ------------------------------------------------------------------
    d_modelled, ll_modelled = disentangle(dataset, truth, nebular=True)

    results = [
        ("left in", combination(d_left), ll_left),
        ("masked out", combination(d_masked), None),
        ("modelled", combination(d_modelled), ll_modelled),
    ]

    core = np.abs(GRID.wave - HBETA) < HBETA * 30.0 / ab.C_KMS
    print(f"{N_EPOCHS} epochs, SNR {SNR:.0f}, {GRID.n} model pixels")
    print(f"\ntrue H-beta: core depth {truth_comb[core].min():+.3f}, EW {ew_true:.3f} A\n")
    print(f"{'treatment':<12} {'core depth':>11} {'core error':>11} {'EW [A]':>9} {'EW error':>9}")
    for name, comb, _ in results:
        # A masked core has no data behind it; the prior interpolates, so report what
        # the pixels say and let the figure show that they are not measurements.
        ew = equivalent_width(comb)
        err = np.mean(comb[core] - truth_comb[core])
        print(
            f"{name:<12} {comb[core].min():>+11.3f} {err:>+11.3f} "
            f"{ew:>9.3f} {100 * (ew - ew_true) / ew_true:>8.1f}%"
        )
    print(
        "\n'masked out' is not a fair equivalent width and is not meant to be: with the core\n"
        "deleted there is nothing behind those pixels but the prior, so the number measures\n"
        "the size of the hole. That is the cost of masking — the product is incomplete\n"
        "exactly where a Balmer gravity diagnostic is read."
    )
    print(
        f"\nmarginal log-likelihood, modelled minus left-in: "
        f"{ll_modelled - ll_left:+.1f} nats (a Bayes factor: both integrate over the spectra)"
    )

    # The nebular component is a measurement in its own right.
    injected = np.asarray(truth.nebular) * float(np.exp(np.mean(np.log(truth.nebular_amplitudes))))
    inside = np.asarray(ab.window_profile(GRID.wave, WINDOWS)) == 1.0
    peak_err = abs(d_modelled[2][inside].max() - injected[inside].max())
    print(
        f"recovered nebular line: peak {d_modelled[2][inside].max():.3f} "
        f"(injected {injected[inside].max():.3f}, error {peak_err:.3f})"
    )

    # 4. Figure, only if matplotlib happens to be installed --------------------------
    if importlib.util.find_spec("matplotlib") is not None:
        plot(truth_comb, results, (lo, hi), "nebular_comparison.png")
        print("\nwrote nebular_comparison.png")
    else:
        print("\nmatplotlib not installed - skipping the figure (it is not a dependency)")

    # 5. The gate --------------------------------------------------------------------
    err = {name: abs(equivalent_width(comb) - ew_true) / ew_true for name, comb, _ in results}
    assert err["left in"] > 0.05, f"expected a visible EW deficit, got {100 * err['left in']:.1f}%"
    assert err["modelled"] < 0.01, f"EW still off by {100 * err['modelled']:.1f}% when modelled"
    assert err["modelled"] < err["masked out"], "modelling did not beat masking on EW"
    assert ll_modelled > ll_left, "the data did not prefer the nebular model"
    print(
        f"\nOK - modelling the nebula recovers the equivalent width to "
        f"{100 * err['modelled']:.2f}% where leaving it in costs "
        f"{100 * err['left in']:.1f}%. Total wall: {time.perf_counter() - t_start:.1f} s"
    )


if __name__ == "__main__":
    main()
