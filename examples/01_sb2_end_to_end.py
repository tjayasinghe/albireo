"""Disentangle an SB2 end to end: simulate, fit, and check against the injected truth.

The shortest complete path through albireo. A synthetic SB2 is generated with the
pathologies the forward model advertises — topocentric wavelengths with per-epoch
barycentric corrections, a chip gap, cosmic hits, finite SNR — and then handed to the
supported inference pipeline:

    MarginalOrbitModel        the marginal posterior over the orbit (spectra integrated out)
      -> run_map              MAP over theta *and* the spectral hyperparameters (ML-II)
      -> laplace_inverse_mass the Hessian at the MAP, as the NUTS mass matrix
      -> run_nuts             sample the orbit with the hyperparameters held fixed
      -> posterior_spectra    component spectra drawn from the *joint* posterior

The script prints posterior mean +/- sd against truth and ends in ``assert``
statements (K_1 and K_2 within 2%), so it doubles as a slow smoke test of the stack.

Honest caveats, both structural rather than numerical (``docs/math.md`` §5):

* the continuum light fractions are *assumed*, not inferred — with constant light the
  likelihood only ever sees the products ``ell_i * d_i`` (§5.2);
* the k = 0 mode of the component spectra is exactly unconstrained (§5.1), so each
  component's smooth envelope is set by the prior. The light-weighted *sum* is the
  quantity the data actually measure.

Environment
-----------
ALBIREO_EXAMPLE_FAST=1
    CI-sized run: 10 epochs, 100 warmup / 150 samples. Unset (the default) gives
    12 epochs and 150 / 250.

Usage
-----
    python examples/01_sb2_end_to_end.py
"""

from __future__ import annotations

import importlib.util
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import albireo as ab

FAST = bool(os.environ.get("ALBIREO_EXAMPLE_FAST"))

# --- the binary we are about to "observe" ----------------------------------------
GRID = ab.LogGrid.from_wavelength_range(5000.0, 5060.0, dv_kms=5.5)
P_TRUE = 6.31  # orbital period [d]
TCONJ_TRUE = 2.05  # conjunction of component 1 (nu + omega = pi/2) [d]
ECC_TRUE = 0.20
OMEGA_TRUE = 0.70  # argument of periastron of component 1 [rad]
K_TRUE = np.array([32.0, 24.0])  # (K_1, K_2) [km/s]
ELL = np.array([0.62, 0.38])  # continuum light fractions -- assumed, never inferred here
LSF = {"HERMES": 7.0}  # Gaussian LSF width [km/s], per instrument
SNR = 120.0  # per-pixel continuum signal-to-noise
N_EPOCHS = 10 if FAST else 12
SEED = 20260811

# The solver bandwidth is *static*, so it is set from a bound on the largest relative
# velocity the two components ever reach: (K_1 + K_2)(1 + e), with headroom. Orbits
# that would exceed it are rejected (-inf) by the model rather than mis-solved.
V_REL_MAX = float(K_TRUE.sum()) * (1.0 + ECC_TRUE) * 1.35

NUM_WARMUP, NUM_SAMPLES = (100, 150) if FAST else (150, 250)
NUM_CHAINS = 1  # one chain keeps the example short; use >= 2 and check r_hat for science

# Priors. The period and conjunction time come from an external ephemeris (tight
# Gaussians, deliberately offset from truth so nothing is initialized at the answer);
# (secosw, sesinw) carry the uniform-on-the-disk prior that maps to uniform in e; the
# hyperparameter priors are weak, and only steer the ML-II fit away from silly scales.
PRIORS = {
    "period": dist.Normal(P_TRUE + 0.001, 0.003),
    "t_conj": dist.Normal(TCONJ_TRUE + 0.005, 0.02),
    "secosw": dist.Uniform(-1.0, 1.0),
    "sesinw": dist.Uniform(-1.0, 1.0),
    "k": dist.Uniform(jnp.array([10.0, 5.0]), jnp.array([45.0, 40.0])),
    "log_tau": dist.Normal(jnp.full(2, np.log(300.0)), 3.0),
    "log_eta": dist.Normal(jnp.full(2, np.log(5.0)), 3.0),
}

# Starting point for L-BFGS: the ephemeris values, a deliberately wrong eccentricity,
# and semi-amplitudes that are merely the right order of magnitude. Circular orbits
# must start slightly off (secosw, sesinw) = (0, 0) -- the one non-smooth point.
INIT = {
    "period": P_TRUE + 0.001,
    "t_conj": TCONJ_TRUE + 0.005,
    "secosw": float(np.sqrt(0.15) * np.cos(0.5)),
    "sesinw": float(np.sqrt(0.15) * np.sin(0.5)),
    "k": jnp.array([27.0, 27.0]),
    "log_tau": jnp.full(2, np.log(300.0)),
    "log_eta": jnp.full(2, np.log(5.0)),
}


def simulate():
    """A 12-epoch (10 in fast mode) SB2 time series with gaps, cosmics and noise."""
    rng = np.random.default_rng(SEED)
    components = [
        ab.synthetic_deviation_spectrum(
            GRID, n_lines=30, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=1
        ),
        ab.synthetic_deviation_spectrum(
            GRID, n_lines=25, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=2
        ),
    ]
    bjd = np.sort(rng.uniform(0.0, 2.4 * P_TRUE, N_EPOCHS))
    v_bary = rng.uniform(-25.0, 25.0, N_EPOCHS)
    t_peri = float(ab.t_peri_from_t_conj(TCONJ_TRUE, period=P_TRUE, ecc=ECC_TRUE, omega=OMEGA_TRUE))
    orbit = ab.OrbitParams(
        period=P_TRUE, t_peri=t_peri, ecc=ECC_TRUE, omega=OMEGA_TRUE, k=tuple(K_TRUE)
    )
    instrument = ab.InstrumentSpec(
        wave=np.arange(5003.0, 5057.0, 0.11), sigma_v_lsf=LSF["HERMES"], snr=SNR
    )
    return ab.simulate_dataset(
        GRID,
        components,
        bjd=bjd,
        instruments={"HERMES": instrument},
        light_fractions=ELL,
        orbit=orbit,
        v_bary=v_bary,
        frame="topocentric",
        gap_fraction=0.01,  # one contiguous chip gap per epoch (ivar = 0, flux = garbage)
        cosmic_fraction=0.002,  # a few masked cosmic hits per epoch
        seed=11,
    )


def print_table(samples: dict) -> None:
    """Posterior mean +/- sd against truth for the parameters that matter."""
    rows = [
        ("P [d]", P_TRUE, np.asarray(samples["period"])),
        ("t_conj [d]", TCONJ_TRUE, np.asarray(samples["t_conj"])),
        ("K_1 [km/s]", K_TRUE[0], np.asarray(samples["k"])[:, 0]),
        ("K_2 [km/s]", K_TRUE[1], np.asarray(samples["k"])[:, 1]),
        ("e", ECC_TRUE, np.asarray(samples["ecc"])),
    ]
    head = f"{'parameter':>11} {'truth':>10} {'post. mean':>12} {'sd':>10} {'rel. err':>9} {'z':>6}"
    print(head)
    print("-" * len(head))
    for name, truth_value, draws in rows:
        mean, sd = float(draws.mean()), float(draws.std())
        rel = abs(mean - truth_value) / max(abs(truth_value), 1e-12)
        print(
            f"{name:>11} {truth_value:10.5f} {mean:12.5f} {sd:10.5f} "
            f"{100 * rel:8.3f}% {(mean - truth_value) / sd:6.2f}"
        )


def plot_rv_curve(samples: dict, truth, bjd: np.ndarray, path: str) -> None:
    """Posterior orbit draws (phase-folded) against the injected Keplerian."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    period = float(np.asarray(samples["period"]).mean())
    t_conj = float(np.asarray(samples["t_conj"]).mean())
    phase = np.linspace(0.0, 1.0, 400)
    t_dense = t_conj + phase * period

    n_post = int(np.asarray(samples["period"]).shape[0])
    idx = np.linspace(0, n_post - 1, min(60, n_post)).astype(int)
    sites = ("period", "t_conj", "secosw", "sesinw", "k")

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for i in idx:
        theta = {s: jnp.asarray(np.asarray(samples[s])[i]) for s in sites}
        vel = np.asarray(ab.orbit_velocities(theta, t_dense))
        ax.plot(phase, vel[0], color="C0", alpha=0.10, lw=0.9)
        ax.plot(phase, vel[1], color="C3", alpha=0.10, lw=0.9)
    vel_true = truth.orbit.component_velocities(t_dense)
    ax.plot(phase, vel_true[0], "k--", lw=1.3, label="truth, component 1")
    ax.plot(phase, vel_true[1], "k:", lw=1.3, label="truth, component 2")
    ax.plot([], [], color="C0", lw=2, label="posterior draws, component 1")
    ax.plot([], [], color="C3", lw=2, label="posterior draws, component 2")

    ep_phase = ((bjd - t_conj) / period) % 1.0
    ymin = float(min(vel_true.min(), -1.0))
    ax.plot(ep_phase, np.full_like(ep_phase, ymin), "|", color="0.3", ms=12, label="epochs")
    ax.set_xlabel("phase from conjunction of component 1")
    ax.set_ylabel("radial velocity [km/s]")
    ax.set_title("SB2 orbit: posterior vs truth (no per-epoch RVs were ever measured)")
    ax.legend(fontsize=8, loc="best")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spectra(draws, truth, path: str) -> None:
    """Joint-posterior component spectra against truth, with the k = 0 caveat visible."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    draws = np.asarray(draws)
    mean, sd = draws.mean(axis=0), draws.std(axis=0)
    wave = GRID.wave
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True, constrained_layout=True)
    for i, ax in enumerate(axes):
        color = f"C{0 if i == 0 else 3}"
        ax.fill_between(
            wave,
            mean[i] - 2 * sd[i],
            mean[i] + 2 * sd[i],
            color=color,
            alpha=0.35,
            lw=0,
            label="posterior +/- 2 sd",
        )
        ax.plot(wave, mean[i], color=color, lw=1.0, label="posterior mean")
        ax.plot(wave, truth.components[i], "k--", lw=0.8, label="truth")
        ax.set_ylabel(f"$d_{i + 1}$")
        ax.legend(fontsize=8, loc="lower right", ncol=3)
    axes[0].set_title(
        "Disentangled deviation spectra (the smooth envelope is prior-set: math.md 5.1)"
    )
    axes[-1].set_xlabel("wavelength [A]")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"albireo {ab.__version__} | fast mode: {FAST} | grid: {GRID.n} px @ {GRID.dv_kms:.2f}")
    t_start = time.perf_counter()

    # 1. Simulate ------------------------------------------------------------------
    dataset, truth = simulate()
    print(dataset.summary())

    # 2. Build the marginal model. The component spectra never appear as parameters:
    #    they are integrated out analytically inside every likelihood evaluation.
    model = ab.MarginalOrbitModel(
        GRID,
        dataset,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        v_rel_max_kms=V_REL_MAX,
    )

    # 3. MAP + ML-II. With log_tau/log_eta among the sampled sites this maximization
    #    *is* the empirical-Bayes hyperparameter fit -- the spectra are already
    #    marginalized, so their prior scales are the only thing left to estimate.
    #    (max_steps=300: the flat hyperparameter directions need ~215 L-BFGS steps to
    #    reach the default |grad| < 1e-2 here, just past the 200-step default cap.)
    t0 = time.perf_counter()
    map_fit = ab.run_map(model.model(PRIORS), init=INIT, max_steps=300)
    t_map = time.perf_counter() - t0
    print(
        f"\nMAP: {map_fit.num_steps} L-BFGS steps, converged={map_fit.converged}, "
        f"|grad|={map_fit.grad_norm:.2e}, potential={map_fit.potential:.2f}  [{t_map:.1f} s]"
    )
    print(
        f"  K = {np.asarray(map_fit.params['k']).round(3)}  "
        f"e = {float(map_fit.params['ecc']):.4f}  "
        f"tau = {np.exp(np.asarray(map_fit.params['log_tau'])).round(1)}  "
        f"eta = {np.exp(np.asarray(map_fit.params['log_eta'])).round(2)}"
    )

    # 4. Freeze the hyperparameters at their ML-II values and sample the orbit.
    hyper = {s: map_fit.params[s] for s in ("log_tau", "log_eta")}
    orbit_priors = {s: d for s, d in PRIORS.items() if s not in hyper}
    nuts_model = model.model(orbit_priors, fixed=hyper)

    # The Laplace covariance at the MAP is a ready-made mass matrix: warmup then only
    # has to tune the step size instead of discovering the parameter scales.
    t0 = time.perf_counter()
    inverse_mass = ab.laplace_inverse_mass(nuts_model, map_fit.params)
    t_laplace = time.perf_counter() - t0
    print(f"Laplace mass matrix: {inverse_mass.shape} [{t_laplace:.1f} s]")

    t0 = time.perf_counter()
    mcmc = ab.run_nuts(
        nuts_model,
        rng_key=jax.random.PRNGKey(3),
        init=map_fit.params,
        inverse_mass_matrix=inverse_mass,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
    )
    samples = mcmc.get_samples()
    jax.block_until_ready(samples["k"])  # mcmc.run dispatches asynchronously
    t_nuts = time.perf_counter() - t0
    extra = mcmc.get_extra_fields()
    n_div = int(np.sum(np.asarray(extra["diverging"])))
    print(
        f"NUTS: {NUM_WARMUP} warmup + {NUM_SAMPLES} samples x {NUM_CHAINS} chain(s), "
        f"{n_div} divergences, {float(np.mean(np.asarray(extra['num_steps']))):.0f} "
        f"leapfrogs/sample  [{t_nuts:.1f} s]"
    )

    # 5. The recovered orbit -------------------------------------------------------
    print("\n=== orbital posterior vs truth ===")
    print_table(samples)

    # 6. The recovered spectra. Each draw picks a posterior theta and then draws once
    #    from the conditional Gaussian over the spectra, so the scatter carries both
    #    the spectral and the orbital uncertainty.
    t0 = time.perf_counter()
    spectra = ab.posterior_spectra(model, samples, jax.random.PRNGKey(9), num_draws=24, extra=hyper)
    spectra_np = np.asarray(spectra)
    t_spec = time.perf_counter() - t0
    mean = spectra_np.mean(axis=0)
    truth_d = np.stack([np.asarray(c) for c in truth.components])
    core = (truth_d[0] < -0.15) | (truth_d[1] < -0.15)  # pixels with real line cores
    visible = ELL @ mean - ELL @ truth_d
    print(f"\nposterior spectra: {spectra_np.shape} draws  [{t_spec:.1f} s]")
    print(
        f"  RMS error in line cores: component 1 "
        f"{np.sqrt(np.mean((mean[0][core] - truth_d[0][core]) ** 2)):.4f}, "
        f"component 2 {np.sqrt(np.mean((mean[1][core] - truth_d[1][core]) ** 2)):.4f}"
    )
    print(
        f"  RMS error in the light-weighted sum (the observable): "
        f"{np.sqrt(np.mean(visible[core] ** 2)):.4f}  <- k=0 degeneracy cancels here"
    )

    # 7. Figures, only if matplotlib happens to be installed ------------------------
    if importlib.util.find_spec("matplotlib") is not None:
        plot_rv_curve(samples, truth, dataset.bjd, "sb2_rv_curve.png")
        plot_spectra(spectra_np, truth, "sb2_spectra.png")
        print("\nwrote sb2_rv_curve.png and sb2_spectra.png")
    else:
        print("\nmatplotlib not installed - skipping figures (it is not a dependency)")

    # 8. The gate ------------------------------------------------------------------
    k_draws = np.asarray(samples["k"])
    for i in range(2):
        rel = abs(float(k_draws[:, i].mean()) - K_TRUE[i]) / K_TRUE[i]
        assert rel < 0.02, f"K_{i + 1} off by {100 * rel:.2f}% (tolerance: 2%)"
    period_rel = abs(float(np.asarray(samples["period"]).mean()) - P_TRUE) / P_TRUE
    assert period_rel < 1e-3, f"period off by {100 * period_rel:.3f}%"
    assert np.isfinite(map_fit.potential)
    print(
        f"\nOK - K_1 and K_2 recovered within 2%. Total wall: {time.perf_counter() - t_start:.1f} s"
    )


if __name__ == "__main__":
    main()
