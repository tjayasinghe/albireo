"""AI Phoenicis: the (K1, K2) map by three methods, with a contour of defined confidence level.

The shift-and-add literature plots a chi-square surface over (K1, K2) and takes its
minimum. No published map has a contour level, because the recovered spectra are re-solved
at every grid point and treated as free parameters, so the number of degrees of freedom the
surface uses is not known (Hynes & Maxted 1998). albireo's marginal likelihood integrates
the spectra out, so the level ``Delta(-2 ln L) = 2.30`` is the two-parameter 68.3 per cent
region in the usual sense. ``p_eff = tr[(Lambda + A^T W A)^-1 A^T W A]`` is the number of
spectral degrees of freedom the data constrain, which is the number a contour level on the
chi-square surface would require.

Three panels on one dataset and one grid:

1. the marginal log-likelihood over (K1, K2) with every other site held at its MAP value
   (period, conjunction, eccentricity vector, hyperparameters, noise scale): the
   like-for-like comparison with the figure in the shift-and-add literature, with the
   spectra integrated out and ``p_eff`` printed;
2. the same two parameters from the NUTS posterior, everything else marginalised: the
   sample cloud with its 68.3 and 95.4 per cent ellipses;
3. the clean-room shift-and-add chi-square on the same grid, with the naive
   ``Delta chi^2 = 2.30`` contour.

The difference between the first two contours is the width that the nuisance parameters
add. The difference between the first and the third is what the figure is meant to show.

Run:  python scripts/aiphe_k1k2_map.py --data data/aiphe --results offset_tests.json
          [--posterior nuts_eggs_keyed.json] [--config eggs_keyed] [--out FIG.png]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import albireo as ab
from albireo.forecast import _effective_parameters

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiphe_bench import DV_KMS, K1_PUB, K2_PUB, WINDOW, light_fractions
from aiphe_offset_tests import load_raws, lsf_for, make_dataset
from shift_and_add import _shift, disentangle

LEVELS = {2.30: "68.3%", 6.17: "95.4%", 11.8: "99.7%"}


def dataset_for(raws, config: str):
    if config == "baseline":
        return make_dataset(raws)
    if config == "eggs_keyed":
        return make_dataset(raws, key_eggs=True)
    if config == "eggs_dropped":
        return make_dataset(raws, drop_eggs=True)
    raise SystemExit(f"unknown configuration {config!r}")


def theta_from(record: dict) -> dict:
    theta = {
        "period": record["period"],
        "t_conj": record["t_conj"],
        "secosw": record["secosw"],
        "sesinw": record["sesinw"],
        "k": jnp.array([record["K1"], record["K2"]]),
        "log_tau": jnp.array(record["log_tau"]),
        "log_eta": jnp.array(record["log_eta"]),
    }
    if record.get("jitter"):
        theta["log_jitter"] = float(np.log(record["jitter"]))
    return theta


def marginal_map(model, theta, k1, k2):
    """``-2 ln L`` above its minimum on the grid, everything but K held."""
    kk1, kk2 = np.meshgrid(k1, k2, indexing="ij")
    trials = np.stack([kk1.ravel(), kk2.ravel()], axis=1)
    t0 = time.perf_counter()
    ll = np.asarray(model.log_likelihood_sweep(theta, {"k": jnp.asarray(trials)}))
    print(f"  marginal sweep: {trials.shape[0]} points in {time.perf_counter() - t0:.0f} s")
    surface = 2.0 * (ll.max() - ll)
    return surface.reshape(kk1.shape)


def effective_parameters(model, theta) -> float:
    """``p_eff`` at ``theta``, exactly, by the one-derivative rule of the forecast module."""
    problem = model.problem_at(theta)
    prior = model._prior(theta)
    # with_jitter sets the factor rather than compounding it, so the fitted noise scale
    # has to be passed in again here or the derivative would be taken at alpha = 1.
    alpha = jnp.exp(jnp.asarray(theta.get("log_jitter", 0.0)))
    return _effective_parameters(problem, prior, alpha, model.half_bandwidth, model.block_size)


def shift_and_add_map(ds, grid, theta, ell, k1, k2, *, n_iter=7):
    """Weighted chi-square of the shift-and-add reconstruction over the same grid."""
    from albireo.inference import orbit_velocities

    obs = np.stack(
        [
            np.interp(np.asarray(grid.wave), np.asarray(ep.wave), np.asarray(ep.flux)) - 1.0
            for ep in ds
        ]
    )
    w = np.stack(
        [
            np.interp(np.asarray(grid.wave), np.asarray(ep.wave), np.asarray(ep.effective_ivar))
            for ep in ds
        ]
    )
    surface = np.empty((k1.size, k2.size))
    t0 = time.perf_counter()
    for a, ka in enumerate(k1):
        for b, kb in enumerate(k2):
            th = {**theta, "k": jnp.array([ka, kb])}
            vel = np.asarray(orbit_velocities(th, jnp.asarray(ds.bjd)))
            shifts = np.asarray(grid.velocity_to_pixels(vel))
            sa = disentangle(obs, shifts, n_iter=n_iter)
            model = np.stack(
                [sum(_shift(sa[i], shifts[i, j]) for i in range(2)) for j in range(ds.n_epochs)]
            )
            surface[a, b] = float(np.sum(w * (obs - model) ** 2))
    print(f"  shift-and-add sweep: {k1.size * k2.size} points in {time.perf_counter() - t0:.0f} s")
    return surface - surface.min()


def ellipse(mean, cov, level):
    """Points on the iso-density ellipse enclosing ``level`` of a bivariate normal."""
    from numpy.linalg import cholesky

    r = np.sqrt(-2.0 * np.log(1.0 - level))
    t = np.linspace(0.0, 2.0 * np.pi, 200)
    circle = np.stack([np.cos(t), np.sin(t)]) * r
    return (mean[:, None] + cholesky(cov) @ circle).T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True, help="JSON from aiphe_offset_tests.py")
    ap.add_argument("--posterior", default=None, help="JSON from aiphe_offset_tests.py --nuts")
    ap.add_argument("--config", default="eggs_keyed")
    ap.add_argument("--n", type=int, default=31, help="grid points per axis")
    ap.add_argument("--span", type=float, default=1.2, help="half-width of the grid, km/s")
    ap.add_argument("--out", default="docs/figures/aiphe_k1k2.png")
    ap.add_argument("--summary", default=None, help="write the numbers here as JSON")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    posterior = json.loads(Path(args.posterior).read_text()) if args.posterior else None
    # The MAP with the noise scale fitted, if a posterior run exists; else the plain MAP,
    # whose contour is too tight by the residual z-RMS (the panel is labelled accordingly).
    record = posterior["map"] if posterior else results[args.config]
    raws = load_raws(Path(args.data))
    ds = dataset_for(raws, args.config)
    ell = light_fractions(float(np.mean(WINDOW)))
    lsf = lsf_for(ds)
    grid = ab.LogGrid.covering(
        ds, dv_kms=DV_KMS, v_margin_kms=140.0, lsf_sigma_kms=max(lsf.values())
    )
    model = ab.MarginalOrbitModel(
        grid, ds, light_fractions=ell, lsf_sigma_v=lsf, v_rel_max_kms=140.0
    )
    theta = theta_from(record)
    k1 = record["K1"] + np.linspace(-args.span, args.span, args.n)
    k2 = record["K2"] + np.linspace(-args.span, args.span, args.n)
    print(f"{args.config}: {ds.n_epochs} epochs, MAP K1 {record['K1']:.3f} K2 {record['K2']:.3f}")

    held = marginal_map(model, theta, k1, k2)
    p_eff = effective_parameters(model, theta)
    n_good = sum(int(ep.good.sum()) for ep in ds)
    print(f"  p_eff = {p_eff:.0f} of {2 * grid.n} spectral parameters; {n_good} weighted pixels")
    sa = shift_and_add_map(ds, grid, theta, ell, k1, k2)

    # The marginal contour's extent, read off the surface along each axis.
    def half_width(surface, axis_vals, axis, level=2.30):
        """Half the extent of the ``level`` region along ``axis`` (0 = K1, 1 = K2)."""
        inside = (surface <= level).any(axis=1 - axis)
        idx = np.where(inside)[0]
        return 0.5 * (axis_vals[idx.max()] - axis_vals[idx.min()])

    summary = {
        "config": args.config,
        "n_epochs": int(ds.n_epochs),
        "map": {"K1": record["K1"], "K2": record["K2"], "jitter": record.get("jitter")},
        "p_eff": float(p_eff),
        "n_spectral_parameters": int(2 * grid.n),
        "n_weighted_pixels": int(n_good),
        "held_68_halfwidth_kms": {
            "K1": float(half_width(held, k1, 0)),
            "K2": float(half_width(held, k2, 1)),
        },
        "shift_and_add_min": {
            "K1": float(k1[np.unravel_index(sa.argmin(), sa.shape)[0]]),
            "K2": float(k2[np.unravel_index(sa.argmin(), sa.shape)[1]]),
        },
        "shift_and_add_naive_68_halfwidth_kms": {
            "K1": float(half_width(sa, k1, 0)),
            "K2": float(half_width(sa, k2, 1)),
        },
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    extent = (k2[0], k2[-1], k1[0], k1[-1])
    levels = sorted(LEVELS)
    for ax, surface, title in (
        (axes[0], held, "marginal likelihood, nuisances held at MAP"),
        (axes[2], sa, "shift-and-add chi-square, spectra re-solved"),
    ):
        ax.imshow(
            np.minimum(surface, 30.0),
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis_r",
        )
        cs = ax.contour(k2, k1, surface, levels=levels, colors="w", linewidths=1.0)
        ax.clabel(cs, fmt={lv: LEVELS[lv] for lv in levels}, fontsize=7)
        ax.set_title(title, fontsize=10)
    axes[0].text(
        0.03,
        0.97,
        f"$\\Delta(-2\\ln L)$ contours\n$p_\\mathrm{{eff}}$ = {p_eff:.0f} of {2 * grid.n}\n"
        f"{n_good} pixels" + ("" if record.get("jitter") else "\n(noise scale not fitted)"),
        transform=axes[0].transAxes,
        va="top",
        color="w",
        fontsize=8,
    )
    axes[2].text(
        0.03,
        0.97,
        "$\\Delta\\chi^2$ contours at the same levels;\nno published map states a level",
        transform=axes[2].transAxes,
        va="top",
        color="w",
        fontsize=8,
    )
    ax = axes[1]
    if posterior is not None:
        samples = np.load(Path(args.posterior).with_suffix(".npz"))
        k = np.asarray(samples["k"]).reshape(-1, 2)
        ax.scatter(k[:, 1], k[:, 0], s=2, alpha=0.25, color="0.4", rasterized=True)
        mean, cov = k.mean(axis=0), np.cov(k.T)
        for level, style in ((0.683, "-"), (0.954, "--")):
            pts = ellipse(mean, cov, level)
            ax.plot(pts[:, 1], pts[:, 0], "k" + style, lw=1.2, label=f"{100 * level:.1f}%")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("NUTS posterior, nuisances marginalised", fontsize=10)
        summary["posterior"] = {
            "K1": [float(mean[0]), float(np.sqrt(cov[0, 0]))],
            "K2": [float(mean[1]), float(np.sqrt(cov[1, 1]))],
            "corr": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
            "n": int(k.shape[0]),
        }
    else:
        ax.text(
            0.5, 0.5, "no posterior file given", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("NUTS posterior (not run)", fontsize=10)
    ax.set_xlim(k2[0], k2[-1])
    ax.set_ylim(k1[0], k1[-1])
    for ax in axes:
        ax.plot(K2_PUB, K1_PUB, "r*", ms=11, label="Maxted et al. 2020")
        ax.plot(record["K2"], record["K1"], "w+", ms=9, mew=1.5)
        ax.set_xlabel("$K_2$ [km/s]")
        ax.set_ylabel("$K_1$ [km/s]")
    fig.suptitle(
        f"AI Phe, {ds.n_epochs} HARPS epochs, {WINDOW[0]:.0f}-{WINDOW[1]:.0f} A ({args.config}); "
        "red star: published; white cross: MAP",
        fontsize=10,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"  figure: {out}")
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
