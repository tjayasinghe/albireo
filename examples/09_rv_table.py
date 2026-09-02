"""Radial velocities without an orbit: the free per-epoch table (``docs/math.md`` §7.6).

Every other fit in this directory imposes a Keplerian. This one does not: each epoch's
velocity is its own free parameter, the component spectra are still marginalized out
analytically, and no orbital element is assumed. The mode exists for three reasons.

* A per-epoch RV table with uncertainties is the standard product of a spectroscopic binary
  analysis, and it is the point of comparison for a user arriving from a cross-correlation
  or shift-and-add pipeline.
* It is the model check for the Keplerian mode. Fit free velocities, then ask whether a
  Keplerian threads them (:func:`albireo.keplerian_residuals`). A slightly wrong period, an
  unmodelled third body, or line-profile variability that the orbit would have absorbed
  into ``e`` all appear as structured residuals where noise alone would not.
* Two of its properties are counter-intuitive, and the script demonstrates them rather than
  stating them.

Property one: there is one arbitrary zero point per component, not one in total. With no
orbit tying the stars together, each component's spectrum is a free vector, so translating
it absorbs a constant added to that component's shifts and leaves the likelihood unchanged.
It is the systemic velocity (D14) once per star rather than once in total. The script
demonstrates the invariance directly, and shows that removing it in velocity space is only
first-order correct while pixel space is exact, because ``xi = artanh(v/c)`` turns
relativistic velocity addition into ordinary addition.

Property two: the raw Laplace error bars are the prior. Each zero point is an exactly flat
direction, so its posterior width is the prior width, and every epoch's marginal variance
inherits it. The script prints both: the raw diagonal, which comes out at
``prior_sigma / sqrt(n_epochs)`` on every entry and would take the same value on an
uninformative dataset, and the projected value from
:func:`albireo.relative_velocity_errors`, which is smaller by a factor of several hundred
and responds to the data.

The mode also has a failure that the script exercises. From a cold start the problem is
multimodal: with every epoch initialized at the same velocity the two components are
indistinguishable, so the free table requires a warm start. The failure is loud rather than
silent, in that the cold fit ends at a potential tens of thousands of nats worse.

Environment
-----------
ALBIREO_EXAMPLE_FAST=1
    CI-sized run: fewer L-BFGS steps. The recovery is slightly looser and the printed
    numbers move a little; every conclusion is unchanged.

Usage
-----
    python examples/09_rv_table.py
"""

from __future__ import annotations

import importlib.util
import os
import time

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import albireo as ab

FAST = bool(os.environ.get("ALBIREO_EXAMPLE_FAST"))

# --- the system, matching tests/test_velocity_table.py ----------------------------
GRID = ab.LogGrid.from_wavelength_range(5000.0, 5040.0, dv_kms=6.0)
P_TRUE, ECC_TRUE, OMEGA_TRUE, T_PERI_TRUE = 6.31, 0.15, 0.70, 2.0
K1_TRUE, K2_TRUE = 30.0, 55.0
ELL = (0.6, 0.4)
LSF = {"a": 7.0}
SNR = 200.0
N_EPOCHS = 10
V_REL_MAX = 320.0
PRIOR = ab.SmoothnessPrior(jnp.asarray([300.0, 300.0]), jnp.asarray([5.0, 5.0]))
# The prior on the free velocities. Its width is what the raw Laplace diagonal returns
# when the zero point is not projected out.
V_PRIOR_SIGMA = 120.0
MAX_STEPS = 120 if FAST else 250
SEED = 5


def kepler_theta(period: float = P_TRUE) -> dict:
    """The injected orbit in albireo's (period, t_conj, secosw, sesinw) parameterization."""
    nu_c = 0.5 * np.pi - OMEGA_TRUE
    e_c = 2.0 * np.arctan2(
        np.sqrt(1.0 - ECC_TRUE) * np.sin(0.5 * nu_c),
        np.sqrt(1.0 + ECC_TRUE) * np.cos(0.5 * nu_c),
    )
    t_conj = T_PERI_TRUE + (e_c - ECC_TRUE * np.sin(e_c)) * period / (2.0 * np.pi)
    return {
        "period": jnp.asarray(period),
        "t_conj": jnp.asarray(t_conj),
        "secosw": jnp.asarray(np.sqrt(ECC_TRUE) * np.cos(OMEGA_TRUE)),
        "sesinw": jnp.asarray(np.sqrt(ECC_TRUE) * np.sin(OMEGA_TRUE)),
        "k": jnp.asarray([K1_TRUE, K2_TRUE]),
    }


def simulate():
    rng = np.random.default_rng(7)
    bjd = np.sort(rng.uniform(0.0, 21.0, size=N_EPOCHS))
    components = [ab.synthetic_deviation_spectrum(GRID, seed=s) for s in (21, 22)]
    instruments = {
        "a": ab.InstrumentSpec(wave=np.arange(5003.0, 5037.0, 0.12), sigma_v_lsf=LSF["a"], snr=SNR)
    }
    orbit = ab.OrbitParams(
        period=P_TRUE, t_peri=T_PERI_TRUE, ecc=ECC_TRUE, omega=OMEGA_TRUE, k=(K1_TRUE, K2_TRUE)
    )
    dataset, truth = ab.simulate_dataset(
        GRID,
        components,
        bjd=bjd,
        instruments=instruments,
        light_fractions=ELL,
        orbit=orbit,
        seed=SEED,
    )
    model = ab.MarginalOrbitModel(
        GRID,
        dataset,
        light_fractions=np.asarray(ELL),
        lsf_sigma_v=LSF,
        v_rel_max_kms=V_REL_MAX,
        prior=PRIOR,
    )
    return model, np.asarray(truth.velocities), bjd


def priors(n_epochs: int) -> dict:
    """No orbital sites: ``velocity`` replaces them, and may not coexist with them."""
    return {
        "velocity": dist.Normal(0.0, V_PRIOR_SIGMA).expand([2, n_epochs]).to_event(2),
        "log_tau": dist.Normal(5.7, 1.5).expand([2]).to_event(1),
        "log_eta": dist.Normal(1.6, 1.0).expand([2]).to_event(1),
    }


def fit(model, init_v, n_epochs):
    return ab.run_map(
        model.model(priors(n_epochs)),
        init={
            "velocity": jnp.asarray(init_v),
            "log_tau": jnp.full(2, 5.7),
            "log_eta": jnp.full(2, 1.6),
        },
        max_steps=MAX_STEPS,
        model_args=(model.problem,),
    )


def relativistic_add(v, c):
    """The exact group operation on velocities: the effect of a constant pixel offset."""
    b1, b2 = np.asarray(v) / ab.C_KMS, c / ab.C_KMS
    return ab.C_KMS * (b1 + b2) / (1.0 + b1 * b2)


def plot(bjd, rel_true, rel_fit, sigma, resid, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = kepler_theta()
    t_conj = float(theta["t_conj"])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    # Overplotting a Keplerian on a free table requires adopting the table's zero point,
    # which is the mean over the observed epochs taken in pixel space. Using the curve's
    # own phase average instead offsets it by several km/s, and the residual panel would
    # then disagree with the left panel about the same fit.
    dense = np.linspace(0.0, 1.0, 400)
    kep_pix = np.asarray(GRID.velocity_to_pixels(ab.orbit_velocities(theta, jnp.asarray(bjd))))
    zero_point = kep_pix.mean(axis=1)
    for i, color in enumerate(("C0", "C3")):
        t_dense = t_conj + dense * P_TRUE
        v_dense = ab.orbit_velocities(theta, jnp.asarray(t_dense))[i]
        centered = GRID.pixels_to_velocity(GRID.velocity_to_pixels(v_dense) - zero_point[i])
        axes[0].plot(dense, np.asarray(centered), color=color, lw=0.9, alpha=0.5)
        ab.plot_phase_fold(
            bjd,
            rel_fit[i],
            P_TRUE,
            t_conj,
            yerr=sigma[i],
            ax=axes[0],
            color=color,
            label=f"star {i + 1}",
        )
    axes[0].set_ylabel("relative RV [km/s]")
    axes[0].set_title("free per-epoch table (no orbit fitted)")
    axes[0].legend(fontsize=8)

    # Wilson diagram: the slope is -K_1/K_2 = the mass ratio, and it is invariant to both
    # zero points because a slope is not a location.
    axes[1].errorbar(
        rel_fit[1], rel_fit[0], xerr=sigma[1], yerr=sigma[0], fmt="o", ms=4, capsize=2, color="C2"
    )
    span = np.array([rel_fit[1].min(), rel_fit[1].max()])
    slope = -K1_TRUE / K2_TRUE
    axes[1].plot(span, slope * span, "k--", lw=0.9, label=f"$-K_1/K_2$ = {slope:.3f}")
    axes[1].set_xlabel("star 2 relative RV [km/s]")
    axes[1].set_ylabel("star 1 relative RV [km/s]")
    axes[1].set_title("Wilson diagram (slope = mass ratio)")
    axes[1].legend(fontsize=8)

    for i, color in enumerate(("C0", "C3")):
        ab.plot_phase_fold(
            bjd,
            resid[i],
            P_TRUE,
            t_conj,
            yerr=sigma[i],
            ax=axes[2],
            color=color,
            label=f"star {i + 1}",
        )
    axes[2].axhline(0.0, color="0.6", lw=0.8)
    axes[2].set_ylabel("free table $-$ Keplerian [km/s]")
    axes[2].set_title("the model check: does a Keplerian thread it?")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t_start = time.perf_counter()
    model, v_true, bjd = simulate()
    rel_true = np.asarray(ab.relative_velocities(v_true, GRID))

    # 1. The zero point: one per component, and exactly flat ---------------------------
    ref = float(model.log_likelihood({"velocity": jnp.asarray(v_true)}))
    exact = np.array(v_true, dtype=float)
    exact[0] = relativistic_add(exact[0], 50.0)
    naive = np.array(v_true, dtype=float)
    naive[0] = naive[0] + 50.0  # ordinary addition, which is NOT the group operation
    d_exact = abs(float(model.log_likelihood({"velocity": jnp.asarray(exact)})) - ref)
    d_naive = abs(float(model.log_likelihood({"velocity": jnp.asarray(naive)})) - ref)
    print("1. The arbitrary zero point, one per component")
    print(f"   log-likelihood at the injected velocities : {ref:.4f}")
    print(f"   ... after a +50 km/s *relativistic* shift of star 1 : moved {d_exact:.3e} nats")
    print(f"   ... after a +50 km/s *ordinary*     shift of star 1 : moved {d_naive:.3e} nats")
    print("   The first is the exact degeneracy; the second is the first-order")
    print("   approximation to it, and the difference is why the centering is done in")
    print("   pixel space (docs/math.md 7.6).")

    # 2. Fit the table, warm-started from a badly wrong Keplerian ----------------------
    start = np.stack([v_true[0] * 1.3, v_true[1] * 0.7])  # 30% off in both semi-amplitudes
    warm = fit(model, start, bjd.size)
    rel_fit = np.asarray(ab.relative_velocities(warm.params["velocity"], GRID))
    rms = np.sqrt(np.mean((rel_fit - rel_true) ** 2, axis=1))
    print(f"\n2. Warm start from a Keplerian 30% wrong in both K ({warm.num_steps} L-BFGS steps)")
    for i in range(2):
        print(
            f"   star {i + 1}: per-epoch RV rms {rms[i]:.4f} km/s "
            f"= 1/{GRID.dv_kms / max(rms[i], 1e-9):.0f} of a model pixel"
        )
    wilson = np.polyfit(rel_fit[1], rel_fit[0], 1)[0]
    print(
        f"   Wilson slope {wilson:.4f} against -K_1/K_2 = {-K1_TRUE / K2_TRUE:.4f} "
        f"({100 * abs(wilson / (-K1_TRUE / K2_TRUE) - 1):.2f}% off); a slope, so both "
        "zero points cancel"
    )

    # 3. The error bars, raw and projected ---------------------------------------------
    cov = ab.laplace_inverse_mass(
        model.model(priors(bjd.size)), warm.params, model_args=(model.problem,)
    )
    sigma = ab.relative_velocity_errors(cov, warm.unconstrained)
    from jax.flatten_util import ravel_pytree

    marks, _ = ravel_pytree(
        {
            name: jnp.full(jnp.shape(jnp.asarray(v)), 1.0 if name == "velocity" else 0.0)
            for name, v in warm.unconstrained.items()
        }
    )
    raw = np.sqrt(np.diag(cov)[np.asarray(marks) > 0.5])
    realized = np.abs(rel_fit - rel_true)
    print("\n3. The error bars: why the raw Laplace diagonal must not be quoted")
    print(
        f"   raw diagonal      : {raw.min():.3f} to {raw.max():.3f} km/s "
        f"(the prior, {V_PRIOR_SIGMA:g}/sqrt({bjd.size}) = "
        f"{V_PRIOR_SIGMA / np.sqrt(bjd.size):.3f})"
    )
    print(f"   projected         : {sigma.min():.4f} to {sigma.max():.4f} km/s")
    print(f"   realized |error|  : {realized.min():.4f} to {realized.max():.4f} km/s")
    print(
        f"   The raw bars are {raw.mean() / sigma.mean():.0f}x too large and identical to "
        "several digits\n   across every epoch and both stars: the signature of reading a "
        "flat direction."
    )

    # 4. The model check ----------------------------------------------------------------
    resid = np.asarray(ab.keplerian_residuals(warm.params["velocity"], kepler_theta(), bjd, GRID))
    bad = np.asarray(
        ab.keplerian_residuals(
            warm.params["velocity"], kepler_theta(period=P_TRUE * 1.005), bjd, GRID
        )
    )
    print("\n4. Does a Keplerian thread the free table?")
    print(
        f"   against the true orbit      : max |residual| {np.max(np.abs(resid)):.4f} km/s "
        f"= {np.max(np.abs(resid) / sigma):.1f} sigma"
    )
    print(
        f"   against a period 0.5% wrong : max |residual| {np.max(np.abs(bad)):.4f} km/s "
        f"= {np.max(np.abs(bad) / sigma):.0f} sigma"
    )
    print("   That gap is what the mode is for: a Keplerian is a strong constraint, and a")
    print("   table fitted without one says whether the data support it.")

    # 5. The failure mode, demonstrated rather than described ---------------------------
    cold = fit(model, np.zeros((2, bjd.size)), bjd.size)
    print("\n5. The cold start, which does not work")
    print(f"   warm potential {warm.potential:.1f}   cold potential {cold.potential:.1f}")
    print(
        f"   The cold fit is {cold.potential - warm.potential:.0f} nats worse. The mode needs "
        "a warm\n   start (a Keplerian fit, or cross-correlation velocities), and the "
        "failure\n   is loud rather than silent."
    )

    # 6. Figure, if matplotlib is installed ----------------------------------------------
    if importlib.util.find_spec("matplotlib") is not None:
        plot(bjd, rel_true, rel_fit, sigma, resid, "rv_table.png")
        print("\nwrote rv_table.png")
    else:
        print("\nmatplotlib not installed - skipping the figure (it is not a dependency)")

    # 7. The gate -------------------------------------------------------------------------
    assert d_exact < 1e-6, f"the relativistic zero-point shift moved the likelihood by {d_exact}"
    assert d_naive > 100 * max(d_exact, 1e-12), "the naive shift should be measurably worse"
    assert np.all(rms < 0.3), f"per-epoch RV rms {rms} km/s"
    assert abs(wilson / (-K1_TRUE / K2_TRUE) - 1) < 0.02, f"Wilson slope {wilson}"
    assert np.ptp(raw) / raw.mean() < 1e-3, "the raw bars should be the prior on every entry"
    assert raw.mean() > 50.0 * sigma.mean(), "the projection should shrink the bars enormously"
    # The residual against the true orbit is the fit's own noise rather than zero, so the
    # two are compared in units of the per-epoch error: a few sigma against tens of them.
    assert np.max(np.abs(resid) / sigma) < 10.0, "the true orbit should thread the table"
    assert np.max(np.abs(bad) / sigma) > 10.0 * np.max(np.abs(resid) / sigma), (
        "a 0.5% period error should be an order of magnitude more visible than the fit residual"
    )
    assert cold.potential > warm.potential + 1e3, (
        f"the cold start should fail loudly: {cold.potential:.1f} vs {warm.potential:.1f}"
    )
    print(
        f"\nOK - zero point exactly flat per component, table recovered to "
        f"{rms.max():.3f} km/s, raw bars {raw.mean() / sigma.mean():.0f}x the projected ones, "
        f"cold start {cold.potential - warm.potential:.0f} nats worse. "
        f"Total wall: {time.perf_counter() - t_start:.1f} s"
    )


if __name__ == "__main__":
    main()
