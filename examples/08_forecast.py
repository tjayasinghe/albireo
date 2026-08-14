"""Which twelve nights should you ask for? (``docs/math.md`` §5.5)

Every other example in this directory analyses data that exist. This one is the question
that comes *before* the data: you have eight epochs, the time allocation committee will
give you twelve more, and you have to say which phases and why.

It is answerable exactly, and by nothing else in the field, because the posterior
covariance of the component spectra

    Sigma = (Lambda_p + A^T W A)^-1

contains no fluxes at all. Only the epoch times (through the velocities, hence the
shifts), the weights, the masks, the line-spread function, the light fractions and the
prior. The observed fluxes move the posterior *mean* and the evidence; they never touch
the covariance. So a night that has not happened has a computable error bar.

The script builds one system and puts three plans of twelve nights against it:

* **aliased** — nights spaced at half the orbital period. This is what a naive reading of
  ``docs/math.md`` §5.1 recommends, because it maximizes the spread of the differential
  velocity ``Var_j(Delta)``, and it is the trap: those nights visit the *two* extreme
  values of ``Delta`` over and over, and two values leave the separation degenerate at a
  whole comb of feature scales.
* **quadrature** — nights spread evenly over orbital phase.
* **more of the same** — twelve nights continuing the existing cadence, the plan that gets
  written when nobody checks.

Three things to watch in the output.

1. The aliased plan wins on RMS differential velocity and loses on everything that
   matters. That is the correction to §5.1's own reading, and it is why albireo computes
   the exact covariance rather than the closed-form proxy.
2. The **worst-determined mode barely moves** under any plan, and sits at ~1x the prior.
   That is not a failure — it is the ``k = 0`` exchange mode, degenerate for *every*
   design, and the forecast reports it rather than hiding it. What a good plan does is
   drag the *rest* of the mode ladder down.
3. Every number is quoted against the same quantity under the prior alone. A forecast
   band that has relaxed onto the prior looks exactly as convincing as one the data
   earned, which is the whole reason the comparison is printed.

What this deliberately does not forecast is the orbit. The Fisher information for a
velocity runs through the derivative of the component spectrum, so an error bar on K_2
needs the line depths — the thing that has not been measured yet.

Environment
-----------
ALBIREO_EXAMPLE_FAST=1
    CI-sized run: a shorter model grid and fewer modes. The ranking of the plans is
    unchanged; the numbers are coarser.

Usage
-----
    python examples/08_forecast.py
"""

from __future__ import annotations

import importlib.util
import os
import time

import numpy as np

import albireo as ab

FAST = bool(os.environ.get("ALBIREO_EXAMPLE_FAST"))

# --- the system -------------------------------------------------------------------
GRID = ab.LogGrid.from_wavelength_range(4480.0, 4520.0 if FAST else 4545.0, dv_kms=5.0)
PERIOD, ECC, OMEGA, T_PERI = 13.7, 0.0, 0.0, 0.0
K1, K2 = 48.0, 71.0
ELL = (0.65, 0.35)
LSF = {"a": 5.5}
SNR = 80.0
PRIOR = ab.SmoothnessPrior(tau=np.array([3e2, 3e2]), eta=np.array([1e-2, 1e-2]))
N_MODES = 3 if FAST else 4
N_PLANNED = 12

# Eight nights already taken, in four tight pairs a fortnight apart — the cadence a
# service-mode queue produces on its own, and it is aliased to the period.
HAVE_BJD = np.array([0.1, 0.3, 6.9, 7.1, 13.8, 14.0, 20.7, 20.9])

ORBIT = ab.OrbitParams(period=PERIOD, t_peri=T_PERI, ecc=ECC, omega=OMEGA, k=(K1, K2))


def observed() -> ab.Dataset:
    """The eight epochs in hand.

    Simulated here so the example runs offline, but note that *nothing below reads the
    flux*: the forecast needs this dataset only for its wavelength grid, its inverse
    variances, its times and its barycentric velocities.
    """
    wave = np.arange(4490.0, GRID.wave[-1] - 12.0, 0.05)
    dataset, _ = ab.simulate_dataset(
        GRID,
        [ab.synthetic_deviation_spectrum(GRID, n_lines=25, seed=s, margin=0.08) for s in (1, 2)],
        bjd=HAVE_BJD,
        instruments={"a": ab.InstrumentSpec(wave=wave, sigma_v_lsf=LSF["a"], snr=SNR)},
        light_fractions=ELL,
        orbit=ORBIT,
        v_bary=np.zeros(HAVE_BJD.size),
        frame="barycentric",
        seed=3,
    )
    return dataset


def plans(t_last: float) -> dict[str, np.ndarray]:
    """Three candidate sets of twelve nights, as BJDs."""
    return {
        "aliased (P/2 spacing)": t_last + PERIOD * np.arange(1, N_PLANNED + 1) / 2.0,
        "quadrature (even in phase)": t_last + PERIOD * (np.arange(N_PLANNED) + 0.5) / N_PLANNED,
        "more of the same cadence": t_last
        + np.repeat(np.arange(1, 7) * 6.9, 2)
        + np.tile([0.0, 0.2], 6),
    }


def forecast(design: ab.Dataset, n_have: int) -> ab.SensitivityForecast:
    return ab.sensitivity_forecast(
        GRID,
        design,
        orbit=ORBIT,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        prior=PRIOR,
        baseline=range(n_have),
        n_modes=N_MODES,
    )


def plot(best: ab.SensitivityForecast, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, _ = ab.plot_forecast(best)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t_start = time.perf_counter()
    dataset = observed()
    n_have = dataset.n_epochs

    # 1. What the eight nights in hand are worth on their own ---------------------------
    base = ab.sensitivity_forecast(
        GRID,
        dataset,
        orbit=ORBIT,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        prior=PRIOR,
        n_modes=N_MODES,
    )
    print(f"Model grid: {GRID.n} pixels at {GRID.dv_kms:.1f} km/s\n")
    print(base.summary())

    # 2. The three plans ---------------------------------------------------------------
    results: dict[str, ab.SensitivityForecast] = {}
    for name, t_new in plans(float(HAVE_BJD[-1])).items():
        design = ab.Dataset([*dataset, *ab.plan_epochs(dataset[0], t_new)], frame=dataset.frame)
        results[name] = forecast(design, n_have)
        print(f"\n=== {name} ===")
        print(results[name].summary())

    # 3. The ranking, and the statistic that gets it wrong ------------------------------
    print(f"\n{'plan':<28} {'gain [nats]':>12} {'RMS dv [km/s]':>14} {'blind':>7} {'mode 2':>9}")
    for name, fc in results.items():
        print(
            f"{name:<28} {fc.gain_nats:>12.1f} {fc.rms_delta_kms:>14.1f} "
            f"{100 * fc.blind_fraction:>6.0f}% {fc.mode_std[1]:>9.4f}"
        )
    best = max(results, key=lambda n: results[n].gain_nats)
    naive = max(results, key=lambda n: results[n].rms_delta_kms)
    print(f"\nBest by information gain     : {best}")
    print(f"Best by RMS differential dv  : {naive}  <- the proxy, and it disagrees")

    # 4. Figure, only if matplotlib happens to be installed -----------------------------
    if importlib.util.find_spec("matplotlib") is not None:
        plot(results[best], "forecast.png")
        print("\nwrote forecast.png")
    else:
        print("\nmatplotlib not installed - skipping the figure (it is not a dependency)")

    # 5. The gate ----------------------------------------------------------------------
    quad = results["quadrature (even in phase)"]
    alias = results["aliased (P/2 spacing)"]
    assert quad.gain_nats > alias.gain_nats, "the aliased plan should not win on information"
    assert alias.rms_delta_kms > quad.rms_delta_kms, (
        "the aliased plan should win on the naive Var(Delta) proxy — that is the point"
    )
    assert quad.mode_std[1] < alias.mode_std[1], "the second mode should fall further"
    assert quad.blind_fraction < alias.blind_fraction
    for fc in results.values():
        assert fc.gain_nats > 0.0, "more epochs cannot be worth less than none"
        assert np.all(fc.component_std <= fc.baseline.component_std + 1e-12), (
            "adding epochs raised a posterior standard deviation somewhere"
        )
        assert 0.8 < fc.worst_mode_gain < 1.5, (
            "the leading mode should sit at the prior: it is the k=0 exchange"
        )
        assert fc.mode_residual < 1e-6, "the mode iteration did not converge"
    print(
        f"\nOK - quadrature wins on information while losing on RMS dv, every plan improves "
        f"every band, and the k=0 mode stays at the prior in all three. "
        f"Total wall: {time.perf_counter() - t_start:.1f} s"
    )


if __name__ == "__main__":
    main()
