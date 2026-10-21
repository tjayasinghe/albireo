"""What holding the regularization weights at their ML-II values leaves out (2026-10-20).

`Fit.sample` holds the curvature and continuum weights of each component at their ML-II
values, a plug-in approximation. On the packaged example, with the quickstart declaration,
this script measures the omission at the best fit, which is joint over the orbit and the
weights:

1. The orbit. The Hessian of the potential is taken with the weights held, as for the
   mass matrix of `Fit.sample`, and with the weights free under their hyperpriors
   (`Fit._plug_in_widening`, which `Posterior.summary` reports for every fit). Printed:
   the ratio of the free to the held standard deviation of each sampled coordinate, its
   largest correlation with a weight, the standard deviations of the weights, and the
   check that the held covariance is the conditional of the free one.
2. The spectra. At the best orbit, the four weights are drawn from their Laplace
   covariance and the spectra are solved at each draw. The band marginalized over the
   weights, the square root of the mean conditional variance plus the variance of the
   conditional means, is compared with the band at the ML-II weights inside the data
   range, and the movement of the conditional mean is divided into a constant, a slope
   and the rest.

Run: ``python scripts/plug_in_bench.py`` (about 5 minutes on the desktop CPU, most of it
the fit). The record is in docs/benchmarks.md, "The plug-in of the regularization
weights, measured (2026-10-20)".
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np

import albireo as ab
from albireo.inference import laplace_check

LIGHT = (0.62, 0.38)
N_DRAWS = 64
SEED = 0


def declaration(dataset):
    """The quickstart declaration (``docs/quickstart.md``)."""
    return ab.Disentangler(
        dataset,
        components=[ab.Star("primary", light=LIGHT[0]), ab.Star("secondary", light=LIGHT[1])],
        orbit=ab.Orbit(period=ab.Between(5.5, 6.5), k=ab.Between([10.0, 10.0], [90.0, 90.0])),
        lsf={"DEMO": 6.5},
    )


def main() -> None:
    dataset = ab.load_example("sb2_sim")
    dis = declaration(dataset)
    start = time.perf_counter()
    fit = dis.fit()
    print(f"fit: {time.perf_counter() - start:.0f} s")
    problem = dis.model.problem

    # -- the orbit ---------------------------------------------------------------
    model, params = fit._orbit_model()
    held = laplace_check(model, params, model_args=(problem,))
    model, params = fit._orbit_model(hold_weights=False)
    free = laplace_check(model, params, model_args=(problem,))
    print(f"Laplace approximation valid: held {held.valid}, free {free.valid}")
    factors, note = fit._plug_in_widening(held)
    assert factors is not None, note

    index = {label: i for i, label in enumerate(free.labels)}
    weights = [index[label] for label in free.labels if label.startswith(("log_tau", "log_eta"))]
    sd = np.sqrt(np.diag(free.covariance))
    corr = free.covariance / np.outer(sd, sd)
    print(f"\n{'coordinate':<12s}{'sd free / sd held':>19s}{'largest |corr| with a weight':>31s}")
    for label in held.labels:
        largest = np.max(np.abs(corr[index[label], weights]))
        print(f"{label:<12s}{factors[label]:19.5f}{largest:31.3f}")
    print()
    for label in free.labels:
        if label.startswith(("log_tau", "log_eta")):
            print(f"sd of {label:<11s}{sd[index[label]]:.4f}")
    orbit = [index[label] for label in held.labels]
    conditional = np.linalg.inv(np.linalg.inv(free.covariance)[np.ix_(orbit, orbit)])
    print(
        "held covariance against the conditional of the free one, largest relative "
        f"difference: {np.max(np.abs(conditional / held.covariance - 1.0)):.1e}"
    )

    # -- the spectra --------------------------------------------------------------
    theta = {**{k: jnp.asarray(v) for k, v in fit.result.params.items()}, **dis.fixed}
    order = [index[f"{site}[{i}]"] for site in ("log_tau", "log_eta") for i in (0, 1)]
    mean_w = np.concatenate(
        [np.asarray(theta["log_tau"], dtype=float), np.asarray(theta["log_eta"], dtype=float)]
    )
    cov_w = free.covariance[np.ix_(order, order)]
    base = dis.model.marginal(theta)
    d0 = np.asarray(base.d_hat)
    sd0 = np.asarray(ab.spectra_std(base))
    means, variances = [], []
    for z in np.random.default_rng(SEED).multivariate_normal(mean_w, cov_w, size=N_DRAWS):
        result = dis.model.marginal(
            {**theta, "log_tau": jnp.asarray(z[:2]), "log_eta": jnp.asarray(z[2:])}
        )
        means.append(np.asarray(result.d_hat))
        variances.append(np.asarray(ab.spectra_std(result)) ** 2)
    means, variances = np.stack(means), np.stack(variances)
    band = np.sqrt(variances.mean(axis=0) + means.var(axis=0))

    wave = np.asarray(dis.grid.wave)
    inside = (wave >= float(dataset[0].wave.min())) & (wave <= float(dataset[0].wave.max()))
    x = (wave[inside] - wave[inside].mean()) / np.ptp(wave[inside])
    basis = np.stack([np.ones_like(x), x], axis=1)
    delta = means[:, :, inside] - d0[None, :, inside]
    print(f"\nspectra, {N_DRAWS} draws of the weights, inside the data range:")
    for i, name in enumerate(dis.component_names):
        ratio = band[i][inside] / sd0[i][inside]
        cores = d0[i][inside] < -0.10
        continuum = np.abs(d0[i][inside]) < 0.02
        coef, *_ = np.linalg.lstsq(basis, delta[:, i, :].T, rcond=None)
        rest = delta[:, i, :] - (basis @ coef).T
        without = np.sqrt(variances[:, i, inside].mean(axis=0) + rest.var(axis=0))
        print(
            f"  {name}: band free / band held median {np.median(ratio):.4f}, largest "
            f"{ratio.max():.4f}; in line cores {np.median(ratio[cores]):.4f}, in the "
            f"continuum {np.median(ratio[continuum]):.4f}"
        )
        print(
            f"  {name}: movement of the mean, rms over the draws: constant {np.std(coef[0]):.5f}, "
            f"slope {np.std(coef[1]) * np.sqrt(np.mean(x**2)):.5f}, rest "
            f"{np.sqrt(np.mean(rest**2)):.5f}; band held median {np.median(sd0[i][inside]):.4f}; "
            f"band free without the constant and the slope, median "
            f"{np.median(without / sd0[i][inside]):.4f}"
        )
    light = np.asarray(LIGHT)[None, :, None]
    parts = light * delta
    print(
        f"  light-weighted sum: movement rms {np.sqrt(np.mean(parts.sum(axis=1) ** 2)):.5f}, "
        f"against {np.sqrt(np.mean(parts[:, 0] ** 2)):.5f} for the primary's part alone"
    )


if __name__ == "__main__":
    main()
