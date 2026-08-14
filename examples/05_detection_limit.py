"""From a peak to a claim: false-alarm rates and detection limits (``docs/math.md`` §6).

``examples/02_k2_scan.py`` finds a companion. This one asks the two questions that turn
that peak into something a referee will accept:

1. **How often would noise alone have done that?** The detection statistic ``D`` has no
   closed-form null distribution — it depends on the companion's prior scale, on the
   epoch sampling, on the masks — so the only honest answer is measured. Draw many
   datasets from the fitted no-companion model, scan each one exactly as the real data
   were scanned, and look at how large ``D`` gets by chance.
2. **What would I have found if it were there?** Inject a companion at a ladder of light
   fractions, scan again, and count how often each rung clears the threshold. Where that
   completeness curve crosses 95% is the limit — the sentence the Gaia BH and
   stripped-star communities have to write.

Both loops run on the *observed* dataset's own operators
(:func:`albireo.simulate.resimulate`): same epochs, same barycentric velocities, same
chip gaps, same weights, same response. Only the noise and the injected spectra change.
That is what makes a few hundred scans cost seconds rather than hours, and it is also
what makes the answer specific to *this* dataset rather than to a plausible imitation.

The script runs it twice, on the same simulated system:

* a genuine **SB1** — nothing to find, and the calibration says how faint a companion
  would have had to be to hide;
* the **SB2** — a real companion, whose peak is then read against the null distribution
  the SB1 run produced.

Two things worth watching. First, the null peaks come out **negative**: the marginal
likelihood charges an Occam term for the companion's free spectrum, and with nothing to
find, nothing pays for it. So the calibrated threshold is negative too, and "D > 0" would
have been a *conservative* test here — on another dataset it might not be, which is the
whole argument for calibrating rather than assuming. Second, the limit is conditional on
the assumed companion spectrum: the observable is ``ell_2 * d_2``, so a companion with no
lines is invisible at any light fraction. The default template is the primary's own
recovered spectrum, and that assumption belongs next to the number.

Environment
-----------
ALBIREO_EXAMPLE_FAST=1
    CI-sized run: fewer trials and a coarser K_2 grid. The structure of the result is
    unchanged, but the false-alarm resolution floor is much coarser -- 1/(n_null + 1) --
    so the quoted limit is correspondingly rougher.

Usage
-----
    python examples/05_detection_limit.py
"""

from __future__ import annotations

import importlib.util
import os
import time

import jax.numpy as jnp
import numpy as np

import albireo as ab

FAST = bool(os.environ.get("ALBIREO_EXAMPLE_FAST"))

# --- the system, matching examples/02_k2_scan.py ----------------------------------
GRID = ab.LogGrid.from_wavelength_range(5000.0, 5045.0, dv_kms=5.5)
P_TRUE, ECC_TRUE, OMEGA_TRUE, T_PERI_TRUE = 6.31, 0.15, 0.70, 2.0
K1_TRUE, K2_TRUE = 12.0, 38.0
ELL = (0.9, 0.1)  # the ASSUMED pair the scan runs with (D13)
LSF = {"a": 7.0}
SNR = 150.0
N_EPOCHS = 9 if FAST else 12
K2_GRID = np.arange(14.0, 66.0, 6.0 if FAST else 4.0)
PRIOR = ab.SmoothnessPrior(jnp.asarray([300.0, 30.0]), jnp.asarray([5.0, 5.0]))
V_REL_MAX = 105.0
SEED = 7

# The ladder of injected companion light fractions, and how many trials each rung gets.
# n_null sets the finest false-alarm probability the calibration can resolve, at
# 1 / (n_null + 1): 100 trials cannot substantiate a claim below about 1%.
ELL2_LADDER = np.array([0.005, 0.01, 0.02, 0.04])
N_NULL = 24 if FAST else 120
N_TRIALS = 8 if FAST else 40
FALSE_ALARM = 0.05 if FAST else 0.01
CONFIDENCE = 0.95


def sb1_solution() -> dict:
    """The fixed SB1 orbit in albireo's (period, t_conj, secosw, sesinw) parameterization."""
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
        comps, ell = [primary, companion], ELL
    else:
        orbit = ab.OrbitParams(
            period=P_TRUE, t_peri=T_PERI_TRUE, ecc=ECC_TRUE, omega=OMEGA_TRUE, k=(K1_TRUE,)
        )
        comps, ell = [primary], (1.0,)
    dataset, _ = ab.simulate_dataset(
        GRID,
        comps,
        bjd=bjd,
        instruments=instruments,
        light_fractions=ell,
        orbit=orbit,
        seed=5,
    )
    return dataset


def scan_kwargs() -> dict:
    return {
        "orbit": sb1_solution(),
        "k1": K1_TRUE,
        "k2_grid": K2_GRID,
        "light_fractions": ELL,
        "lsf_sigma_v": LSF,
        "prior": PRIOR,
        "v_rel_max_kms": V_REL_MAX,
    }


def ticker(total_hint: str):
    """A progress callback that stays readable in a CI log: one line per 10%."""
    state = {"next": 0.0}

    def report(done: int, total: int) -> None:
        frac = done / total
        if frac >= state["next"] or done == total:
            state["next"] = frac + 0.1
            print(f"    {total_hint}: {done:4d}/{total} trials", flush=True)

    return report


def plot(limit, injected_scan, path: str) -> None:
    """Both panels come from :func:`albireo.plot_detection_limit` — the figure is library
    code, not example code, so a user gets it without copying anything out of here."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, _ = ab.plot_detection_limit(limit, observed=injected_scan.detection_peak)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    t_start = time.perf_counter()
    print(f"model grid: {GRID.n} pixels, {N_EPOCHS} epochs, SNR {SNR:.0f}")
    print(
        f"calibration: {N_NULL} null trials + {ELL2_LADDER.size} x {N_TRIALS} injected, "
        f"each a full {K2_GRID.size}-point scan\n"
    )

    sb1 = simulate(with_companion=False)
    sb2 = simulate(with_companion=True)
    common = scan_kwargs()

    # 1. The calibration, on the companion-free dataset -------------------------------
    print("1. calibrating on the SB1 (nothing to find)")
    t0 = time.perf_counter()
    limit = ab.detection_limit(
        GRID,
        sb1,
        k2_true=K2_TRUE,
        ell2_grid=ELL2_LADDER,
        n_null=N_NULL,
        n_trials=N_TRIALS,
        false_alarm=FALSE_ALARM,
        confidence=CONFIDENCE,
        seed=17,
        progress=ticker("calibrating"),
        **common,
    )
    t_cal = time.perf_counter() - t0
    n_scans = N_NULL + N_TRIALS * ELL2_LADDER.size
    print(
        f"  {n_scans} scans x {K2_GRID.size + 1} marginal solves in {t_cal:.1f} s "
        f"({1e3 * t_cal / n_scans:.0f} ms per scan)"
    )
    print(
        f"  null peak D: min {limit.null_peaks.min():.1f}, median "
        f"{np.median(limit.null_peaks):.1f}, max {limit.null_peaks.max():.1f}"
    )
    print(
        f"  threshold D > {limit.threshold:.1f}; realized null exceedance "
        f"{np.mean(limit.null_peaks > limit.threshold):.3f} (budget {FALSE_ALARM:g})"
    )
    for ell2, comp, peaks in zip(ELL2_LADDER, limit.completeness, limit.signal_peaks, strict=True):
        print(
            f"    ell_2 = {100 * ell2:5.2f}%  detected {comp:5.1%}  "
            f"median D = {np.median(peaks):10.1f}"
        )
    print(f"\n  >>> {limit.summary()}\n")

    # 2. The real detection, read against that null distribution ----------------------
    print("2. the SB2, scanned identically")
    t0 = time.perf_counter()
    injected = ab.k2_scan(GRID, sb2, **common)
    print(
        f"  peak D = {injected.detection_peak:.1f} at K_2 = {injected.k2_peak:.1f} km/s "
        f"[{time.perf_counter() - t0:.1f} s]"
    )
    fap = limit.false_alarm_probability(injected.detection_peak)
    print(
        f"  false-alarm probability {fap:.4f}"
        + (
            f" -- the 1/(n_null+1) = {limit.fap_floor:.4f} floor, i.e. no companion-free "
            "trial came close"
            if fap == limit.fap_floor
            else ""
        )
    )

    # 3. K_1 marginalized, since the SB1 solution has an error bar --------------------
    print("\n3. the same scan with K_1 integrated out rather than assumed")
    t0 = time.perf_counter()
    marginal = ab.k2_scan(GRID, sb2, k1_sigma=0.05 * K1_TRUE, k1_nodes=7, **common)
    print(
        f"  peak D = {marginal.detection_peak:.1f} at K_2 = {marginal.k2_peak:.1f} km/s, "
        f"best node K_1 = {marginal.k1_peak:.2f} km/s "
        f"[{time.perf_counter() - t0:.1f} s for a "
        f"{marginal.log_likelihood_grid.shape[0]}x{K2_GRID.size} grid]"
    )
    print(
        "  the 2-D surface is in .log_likelihood_grid -- its ridge is the K_1-K_2 "
        "covariance a fixed-K_1 scan assumes away"
    )

    # 4. Figure, only if matplotlib happens to be installed ----------------------------
    if importlib.util.find_spec("matplotlib") is not None:
        plot(limit, injected, "detection_limit.png")
        print("\nwrote detection_limit.png")
    else:
        print("\nmatplotlib not installed - skipping the figure (it is not a dependency)")

    # 5. The gate ----------------------------------------------------------------------
    assert np.all(limit.null_peaks < 0.0), "a companion-free trial reached D >= 0"
    assert np.mean(limit.null_peaks > limit.threshold) <= FALSE_ALARM + 1e-12, (
        "the threshold is anti-conservative against its own null trials"
    )
    assert np.all(np.diff(limit.completeness) >= 0.0), "completeness is not monotone in ell_2"
    assert limit.completeness[-1] == 1.0, "the brightest injected rung was not always found"
    assert injected.detection_peak > limit.null_peaks.max(), (
        "the injected companion did not clear every null trial"
    )
    assert injected.k2_peak == K2_TRUE, f"peak at K_2 = {injected.k2_peak}, injected {K2_TRUE}"
    print(
        f"\nOK - null strictly negative, threshold conservative, completeness monotone, "
        f"the real companion above every null trial. "
        f"Total wall: {time.perf_counter() - t_start:.1f} s"
    )


if __name__ == "__main__":
    main()
