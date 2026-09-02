"""The same fit twice: the expert path, and the `Disentangler` façade over it.

This script is the before/after for ``docs/design.md`` D46. Both halves fit the packaged
example (a simulated SB2 with a known injected truth), and the two results are asserted to
agree with each other and with the truth.

    python examples/07_facade.py

What the façade derives
-----------------------
The expert path requires four quantities to be supplied correctly, and an error in any of
them produces a converged fit with a wrong answer:

* Velocity budget (``v_rel_max_kms``). It must bound the largest relative velocity the
  priors allow, not the value the fit converges to. Too small a budget stalls the sampler
  against a guard it cannot see; through ``log_likelihood`` directly, too small a budget
  returns a wrong value with no error. The information is already in the ``k`` priors.
* Grid margin. It must cover that budget plus the LSF kernel radius. Short of it, the
  shifted model runs off the end of the grid and the fit silently loses flux.
* Conjunction phase. It must be located before optimizing. The marginal likelihood is
  sharply multimodal in phase (the scan below spans 10⁵ nats), and L-BFGS started in the
  wrong trough converges to the wrong answer.
* Smoothness hyperparameters. They are fitted by ML-II and must then be carried into every
  downstream call; omitting them raises no error.

A fifth difference is structural: in the expert path ``priors`` and ``init`` are two dicts
that must be kept consistent by hand. A façade spec carries both.

What the façade does not derive
-------------------------------
The light fractions. With constant light fractions the likelihood depends only on the
products ``l_i * d_i`` (``docs/math.md`` §5.2), so every recovered depth scales as
``1 / l_i`` and no diagnostic in the fit can detect a wrong value. ``Star(light=...)`` is
required, and every summary repeats it under ``Assumed, not measured``.

Usage
-----
    python examples/07_facade.py [--steps N]
"""

from __future__ import annotations

import argparse
import time

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import albireo as ab

LIGHT = (0.62, 0.38)
LSF_SIGMA = 6.5


def expert_path(dataset, *, max_steps: int) -> dict:
    """The fit through the low-level API: grid, model, priors, scan and MAP by hand."""
    # 1. The velocity budget, by hand. The k priors below reach 90 km/s each, at up to
    #    e = 0.64 (the secosw/sesinw bounds), on topocentric data, so the bound is
    #    (90 + 90) * 1.64 + 2 * 30 = 355. A smaller value such as 160 truncates the prior
    #    against the solver's guard instead of raising an error.
    v_rel_max = 355.0
    grid = ab.LogGrid.covering(dataset, dv_kms=4.0, v_margin_kms=v_rel_max, lsf_sigma_kms=LSF_SIGMA)
    model = ab.MarginalOrbitModel(
        grid,
        dataset,
        light_fractions=LIGHT,
        lsf_sigma_v={"DEMO": LSF_SIGMA},
        v_rel_max_kms=v_rel_max,
        prior=ab.SmoothnessPrior(tau=np.full(2, 300.0), eta=np.full(2, 5.0)),
    )

    # 2. The priors and the starting values: two dicts whose keys must agree.
    priors = {
        "period": dist.Uniform(5.5, 6.5),
        "t_conj": dist.Uniform(-0.5, 6.0),
        "secosw": dist.Uniform(-0.8, 0.8),
        "sesinw": dist.Uniform(-0.8, 0.8),
        "k": dist.Uniform(jnp.array([10.0, 10.0]), jnp.array([90.0, 90.0])),
        "log_tau": dist.Normal(jnp.log(300.0) * jnp.ones(2), 3.0),
        "log_eta": dist.Normal(jnp.log(5.0) * jnp.ones(2), 3.0),
    }
    init = {
        "period": 6.0,
        "t_conj": 0.0,
        # Not exactly (0, 0): the parameterization is singular there, the gradient is
        # NaN, and numpyro reports only "Cannot find valid initial parameters".
        "secosw": 0.2,
        "sesinw": 0.1,
        "k": jnp.array([50.0, 50.0]),
        "log_tau": jnp.log(300.0) * jnp.ones(2),
        "log_eta": jnp.log(5.0) * jnp.ones(2),
    }
    assert set(priors) == set(init), "the two dicts have to carry the same keys"

    # 3. Locate the conjunction phase before optimizing.
    trials = float(np.min(dataset.bjd)) + np.linspace(0.0, 6.0, 41, endpoint=False)
    theta = {k: jnp.asarray(v) for k, v in init.items()}
    scan = []
    for t in trials:
        theta["t_conj"] = jnp.asarray(float(t))
        scan.append(float(model.log_likelihood(theta)))
    init["t_conj"] = float(trials[int(np.argmax(scan))])
    priors["t_conj"] = dist.Uniform(init["t_conj"] - 3.0, init["t_conj"] + 3.0)

    # 4. MAP, which with log_tau/log_eta among the sites is also the ML-II step.
    fit = ab.run_map(
        model.model(priors),
        init=init,
        max_steps=max_steps,
        model_args=(model.problem,),
    )

    # 5. Carry the fitted hyperparameters downstream, by hand.
    params = ab.orbit_parameters(fit.params)
    return {
        "period": float(params["period"]),
        "k": np.asarray(params["k"], dtype=float),
        "ecc": float(params["ecc"]),
        "scan_contrast": float(np.max(scan) - np.min(scan)),
    }


def facade_path(dataset, *, max_steps: int):
    """The same fit through the façade declaration."""
    dis = ab.Disentangler(
        dataset,
        components=[ab.Star("primary", light=LIGHT[0]), ab.Star("secondary", light=LIGHT[1])],
        orbit=ab.Orbit(
            period=ab.Between(5.5, 6.5),
            k=ab.Between([10.0, 10.0], [90.0, 90.0]),
            ecc=ab.Between(0.0, 0.64),
        ),
        lsf={"DEMO": LSF_SIGMA},
        dv_kms=4.0,
    )
    return dis, dis.fit(max_steps=max_steps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=150, help="L-BFGS iteration cap")
    args = parser.parse_args(argv)

    dataset, truth = ab.load_example("sb2_sim", with_truth=True)
    print(dataset.summary())
    print(f"\ninjected truth: P = {truth['period']} d, K = {truth['k']}, e = {truth['ecc']}\n")

    print("=" * 78)
    print("the expert path")
    print("=" * 78)
    t0 = time.time()
    expert = expert_path(dataset, max_steps=args.steps)
    print(
        f"[{time.time() - t0:5.1f}s] P = {expert['period']:.6f} d, "
        f"K = {expert['k'][0]:.3f} / {expert['k'][1]:.3f} km/s, e = {expert['ecc']:.4f}"
    )
    print(f"          conjunction scan spanned {expert['scan_contrast']:.3g} nats")

    print()
    print("=" * 78)
    print("the façade declaration")
    print("=" * 78)
    t0 = time.time()
    dis, fit = facade_path(dataset, max_steps=args.steps)
    print(dis.explain())
    print()
    print(fit.summary())
    print(f"\n[{time.time() - t0:5.1f}s] elapsed")

    # The check this example exists for: the two paths must agree.
    got = np.asarray([fit.star(n)["k"] for n in ("primary", "secondary")])
    print()
    print("=" * 78)
    print(
        f"agreement: dK = {np.abs(got - expert['k'])} km/s, "
        f"dP = {abs(float(fit.orbit()['period']) - expert['period']):.2e} d"
    )
    assert np.allclose(got, expert["k"], atol=0.25), (got, expert["k"])
    assert np.allclose(got, truth["k"], atol=0.3), (got, truth["k"])
    print("both paths agree with each other and with the injected truth.")
    print(
        "\nThe façade derived the velocity budget, the grid margin, the conjunction phase\n"
        "and the ML-II hyperparameters; dis.expert() returns the (model, priors, init)\n"
        "triple the first half of this script builds by hand."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
