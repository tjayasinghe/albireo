"""Build the figures of ``docs/method.md`` from the packaged example ``sb2_sim``.

The script fits the quickstart declaration once (the best fit with ML-II, then NUTS),
computes every array that a figure plots, and writes each figure in a light and a dark
rendering to ``docs/method/figures/``:

1. ``fig01-epochs``: two epochs of the composite spectrum, the model, and the
   contribution of each star;
2. ``fig02-structure``: the design matrix ``A`` and the posterior precision
   ``Lambda + A^T W A`` of a small simulated problem;
3. ``fig03-regularization``: the component spectra at the fitted orbit for three
   curvature weights;
4. ``fig04-evidence``: the terms of the marginal likelihood against the curvature weight;
5. ``fig05-semi-amplitudes``: the joint posterior of the two semi-amplitudes, with the
   Laplace approximation.

The script asserts the identities that the figures rely on: the contributions of the
two stars sum to the model, the dense posterior precision equals the prior precision plus
``A^T W A``, and the four terms of the marginal likelihood sum to the value that
:func:`albireo.marginal_loglikelihood` returns. The run is also the quickstart's run, so
its printed reports are written to ``.cache/method_figures/reports.txt`` for
``docs/quickstart.md``.

Results are cached in ``.cache/method_figures/``, keyed on the source of the package and
of this script. Usage::

    python scripts/build_method_figures.py                 # fit, sample, compute, draw
    python scripts/build_method_figures.py --figures-only  # redraw from the cache
    python scripts/build_method_figures.py --arrays        # recompute the arrays only

A cold run takes about 15 minutes on one CPU, most of it the NUTS run.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "method" / "figures"
CACHE = REPO / ".cache" / "method_figures"
SEED = 0
TAU_SCALES = (1e-2, 1.0, 1e2)  # Fig. 3: the curvature weight relative to its ML-II value
SWEEP = np.logspace(-4.0, 4.0, 33)  # Fig. 4: the same ratio, on a grid
NUTS = {"num_warmup": 500, "num_samples": 500, "num_chains": 2}  # Fit.sample defaults

THEMES = {
    "light": {
        "surface": "#ffffff",
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e1",
        "neutral": "#9a9892",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
        "ramp": ("#eef4fc", "#b3d0f5", "#5598e7", "#256abf", "#0e3a73"),
        "soft": "#a9cbf3",
        "faint": "#d3d2cd",
    },
    "dark": {
        "surface": "#1e2129",
        "ink": "#f2f1ee",
        "muted": "#b5b3ad",
        "grid": "#3a3d47",
        "neutral": "#80828c",
        "series": ("#3987e5", "#d95926", "#199e70"),
        "ramp": ("#232836", "#1d3f72", "#2a69bd", "#5d9fee", "#c2dcfb"),
        "soft": "#1f4f8f",
        "faint": "#4d505b",
    },
}


def style(theme: dict) -> dict:
    """Matplotlib settings of one theme (as in ``scripts/rvs_eb_report.py``)."""
    return {
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "axes.edgecolor": theme["grid"],
        "axes.labelcolor": theme["muted"],
        "axes.titlecolor": theme["ink"],
        "text.color": theme["ink"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "grid.color": theme["grid"],
        "axes.grid": True,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "axes.axisbelow": True,
    }


# ---------------------------------------------------------------------------
# The run: the quickstart declaration, fitted and sampled
# ---------------------------------------------------------------------------


def declaration(dataset):
    """The quickstart declaration (``docs/quickstart.md``)."""
    import albireo as ab

    return ab.Disentangler(
        dataset,
        components=[ab.Star("primary", light=0.62), ab.Star("secondary", light=0.38)],
        orbit=ab.Orbit(period=ab.Between(5.5, 6.5), k=ab.Between([10.0, 10.0], [90.0, 90.0])),
        lsf={"DEMO": 6.5},
    )


def cache_key() -> str:
    """Hash of what the run depends on: the package, the declaration, seed and sampler.

    The drawing code is left out, so that a change to a figure does not repeat the run.
    """
    digest = hashlib.sha256()
    for path in sorted((REPO / "src" / "albireo").glob("*.py")):
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    digest.update(inspect.getsource(declaration).encode())
    digest.update(json.dumps({"seed": SEED, **NUTS}, sort_keys=True).encode())
    return digest.hexdigest()[:16]


def run_fit_and_sample() -> None:
    """Fit and sample once; cache the parameters, the draws and the printed reports."""
    import albireo as ab

    dataset = ab.load_example("sb2_sim")
    dis = declaration(dataset)
    start = time.perf_counter()
    fit = dis.fit()
    fit_seconds = time.perf_counter() - start
    print(f"fit: {fit_seconds:.0f} s", flush=True)
    start = time.perf_counter()
    post = fit.sample(seed=SEED, **NUTS)
    sample_seconds = time.perf_counter() - start
    print(f"sample: {sample_seconds:.0f} s", flush=True)

    reports = {
        "dataset.summary()": dataset.summary(),
        "fit.summary()": fit.summary(),
        "dis.explain()": dis.explain(),
        "post.summary()": post.summary(),
        'post.star("secondary")': repr(post.star("secondary")),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "reports.txt").write_text(
        "".join(f"# {name}\n{text}\n\n" for name, text in reports.items()), encoding="utf-8"
    )
    grouped = post.mcmc.get_samples(group_by_chain=True)
    extra = post.mcmc.get_extra_fields()
    arrays = {f"param_{k}": np.asarray(v) for k, v in fit.params.items()}
    arrays |= {f"draw_{k}": np.asarray(v) for k, v in grouped.items()}
    arrays["diverging"] = np.asarray(extra["diverging"])
    np.savez_compressed(CACHE / "run.npz", **arrays)
    meta = {
        "key": cache_key(),
        "hyper": {name: {k: float(v) for k, v in h.items()} for name, h in fit.hyper.items()},
        "scan_period": float(fit.phase_scan.period),
        "fit_seconds": fit_seconds,
        "sample_seconds": sample_seconds,
    }
    (CACHE / "run.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(reports["fit.summary()"], reports["post.summary()"], sep="\n\n", flush=True)


def load_run() -> tuple[dict, dict]:
    with np.load(CACHE / "run.npz") as data:
        arrays = {name: data[name] for name in data.files}
    meta = json.loads((CACHE / "run.json").read_text(encoding="utf-8"))
    return arrays, meta


# ---------------------------------------------------------------------------
# The arrays of the figures
# ---------------------------------------------------------------------------


def compute_arrays() -> None:
    """Every plotted array, from the cached run, with the identities asserted."""
    import jax
    import jax.numpy as jnp

    import albireo as ab

    run, meta = load_run()
    dataset, truth = ab.load_example("sb2_sim", with_truth=True)
    dis = declaration(dataset)
    params = {k[6:]: jnp.asarray(v) for k, v in run.items() if k.startswith("param_")}
    theta = {**params, **{k: jnp.asarray(v) for k, v in dis.fixed.items()}}
    out: dict[str, np.ndarray] = {}

    # Fig. 1: the epochs of widest and narrowest separation, decomposed.
    problem = dis.model.problem_at(theta)
    d_hat = np.asarray(dis.model.marginal(theta).d_hat)
    velocities = np.asarray(
        ab.orbit_velocities({k: v for k, v in theta.items()}, np.asarray(dataset.bjd))
    )
    separation = np.abs(velocities[0] - velocities[1])
    for tag, j in (("wide", int(np.argmax(separation))), ("close", int(np.argmin(separation)))):
        group_index, row = next(
            (k, g.epoch_indices.index(j))
            for k, g in enumerate(problem.groups)
            if j in g.epoch_indices
        )
        group = problem.groups[group_index]
        linear = np.asarray(ab.forward.apply_model(problem, d_hat)[group_index][row])
        parts = [
            np.asarray(
                ab.forward.apply_model(problem, np.where(np.arange(2)[:, None] == i, d_hat, 0.0))[
                    group_index
                ][row]
            )
            for i in range(2)
        ]
        assert np.allclose(parts[0] + parts[1], linear, atol=1e-12), "parts do not sum"
        r = np.asarray(group.r[row])
        base = np.asarray(group.base)
        light = np.asarray(group.light[row])
        good = np.asarray(group.w[row]) > 0
        epoch = dataset[j]
        assert epoch.wave.shape == base.shape, "the epoch is not the group's native grid"
        model = r * (base + linear)
        out[f"f1_{tag}_wave"] = np.asarray(epoch.wave)
        out[f"f1_{tag}_flux"] = np.where(good, epoch.flux, np.nan)
        out[f"f1_{tag}_model"] = np.where(good, model, np.nan)
        for i in range(2):
            out[f"f1_{tag}_star{i}"] = np.where(good, r * (light[i] * base + parts[i]), np.nan)
        sigma = np.where(good, 1.0 / np.sqrt(np.where(good, epoch.ivar, 1.0)), np.nan)
        out[f"f1_{tag}_z"] = (out[f"f1_{tag}_flux"] - out[f"f1_{tag}_model"]) / sigma
        out[f"f1_{tag}_meta"] = np.array([j, epoch.bjd, *velocities[:, j]])

    # Fig. 2: a small problem, with A and the posterior precision written out densely.
    from albireo.solver import dense_from_block_tridiagonal

    # Velocity differences of up to 83 km/s (21 model pixels) make the bands move visibly;
    # the grid keeps 1.5 A of margin for the largest shift plus the LSF.
    grid = ab.LogGrid.from_wavelength_range(4998.0, 5006.0, dv_kms=4.0)
    comps = [ab.synthetic_deviation_spectrum(grid, n_lines=3, seed=s, margin=0.2) for s in (1, 2)]
    small_v = np.array([[30.0, -25.0, 10.0], [-45.0, 38.0, -15.0]])
    instrument = {
        "A": ab.InstrumentSpec(wave=np.arange(4999.5, 5004.5, 0.1), sigma_v_lsf=3.0, snr=50.0)
    }
    small, _ = ab.simulate_dataset(
        grid,
        comps,
        bjd=np.arange(3.0),
        velocities=small_v,
        light_fractions=[0.6, 0.4],
        instruments=instrument,
        v_bary=np.zeros(3),
        frame="barycentric",
        seed=0,
    )
    small_problem = ab.build_problem(
        grid, small, velocities=small_v, light_fractions=[0.6, 0.4], lsf_sigma_v={"A": 3.0}
    )
    small_prior = ab.SmoothnessPrior(tau=[5.0, 5.0], eta=[1e-2, 1e-2])
    small_result = ab.marginal_loglikelihood(small_problem, small_prior, validate=True)

    def linear_map(d):
        return jnp.concatenate(
            [
                (g.r * m).ravel()
                for g, m in zip(
                    small_problem.groups, ab.forward.apply_model(small_problem, d), strict=True
                )
            ]
        )

    n = grid.n
    design = np.asarray(jax.jacfwd(linear_map)(jnp.zeros((2, n)))).reshape(-1, 2 * n)
    weights = np.concatenate([np.asarray(g.effective_w).ravel() for g in small_problem.groups])
    by_star = small_prior.dense(n) + (design.T * weights) @ design
    interleaved = dense_from_block_tridiagonal(small_result.precision)
    order = (np.arange(2 * n) % n) * 2 + np.arange(2 * n) // n  # by star -> interleaved
    assert np.allclose(
        interleaved[np.ix_(order, order)], by_star, rtol=0.0, atol=1e-9 * np.abs(by_star).max()
    )
    out["f2_design"] = design
    out["f2_by_star"] = by_star
    out["f2_interleaved"] = interleaved
    out["f2_meta"] = np.array(
        [n, small.n_epochs, design.shape[0] // small.n_epochs, small_result.precision.block_size]
    )

    # Fig. 3 and Fig. 4: the spectra and the marginal likelihood against the curvature
    # weight, at the fitted orbit, with eta at its ML-II value.
    from albireo.assembly import prior_logdet
    from albireo.forward import weighted_data_terms
    from albireo.solver import logdet

    base_prior = dis.model.fixed_prior
    _zwz, logw, n_good = (float(x) for x in weighted_data_terms(problem))
    constant = 0.5 * logw - 0.5 * n_good * math.log(2.0 * math.pi)

    def at(scale: float):
        log_tau = jnp.asarray(theta["log_tau"]) + math.log(scale)
        result = dis.model.marginal({**theta, "log_tau": log_tau})
        prior = ab.SmoothnessPrior(
            jnp.exp(log_tau),
            jnp.exp(jnp.asarray(theta["log_eta"])),
            None if base_prior is None else base_prior.tau_profile,
            None if base_prior is None else base_prior.eta_profile,
        )
        d = result.d_hat
        chi2 = float(jnp.sum(ab.data_residual_zscores(problem, d) ** 2))
        penalty = float(jnp.sum(d * prior.apply(d)))
        ld_post = float(logdet(result.chol))
        ld_prior = float(prior_logdet(prior, grid_n))
        total = float(result.log_likelihood)
        rebuilt = -0.5 * (chi2 + penalty) - 0.5 * ld_post + 0.5 * ld_prior + constant
        assert abs(rebuilt - total) < 1e-8 * max(1.0, abs(total)), (scale, rebuilt, total)
        return result, (chi2, penalty, ld_post, ld_prior, total)

    grid_n = dis.grid.n
    wave = np.asarray(dis.grid.wave)
    truth_grid = ab.LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))
    out["f3_wave"] = wave
    out["f3_truth"] = np.stack(
        [
            np.interp(wave, truth_grid.wave, component, left=0.0, right=0.0)
            for component in truth["components"]
        ]
    )
    out["f3_data_range"] = np.array([float(dataset[0].wave.min()), float(dataset[0].wave.max())])
    for k, scale in enumerate(TAU_SCALES):
        result, _ = at(scale)
        out[f"f3_d_{k}"] = np.asarray(result.d_hat)
        out[f"f3_sd_{k}"] = np.asarray(ab.spectra_std(result))
    terms = np.array([at(float(scale))[1] for scale in SWEEP])
    out["f4_scale"] = SWEEP
    out["f4_terms"] = terms  # chi2, penalty, ln det posterior, ln det prior, ln p
    out["f4_constant"] = np.array([constant, n_good])

    # Fig. 5: the semi-amplitudes, drawn and in the Laplace approximation at the best fit.
    import numpyro.distributions as dist
    from jax.flatten_util import ravel_pytree
    from numpyro.distributions.transforms import biject_to

    names = dis.component_names
    hyper = meta["hyper"]
    fixed = {
        **dis.fixed,
        "log_tau": jnp.log(jnp.array([hyper[name]["tau"] for name in names])),
        "log_eta": jnp.log(jnp.array([hyper[name]["eta"] for name in names])),
    }
    priors = {k: v for k, v in dis.priors.items() if k not in fixed}
    centre, period = float(params["t_conj"]), meta["scan_period"]
    priors["t_conj"] = dist.Uniform(centre - 0.5 * period, centre + 0.5 * period)
    start = {k: v for k, v in params.items() if k in priors}
    cov_u = np.asarray(
        ab.laplace_inverse_mass(
            dis.model.model(priors, fixed=fixed), start, model_args=(dis.model.problem,)
        )
    )
    _, unravel = ravel_pytree({k: jnp.zeros(jnp.shape(v)) for k, v in start.items()})
    index = np.asarray(unravel(jnp.arange(cov_u.shape[0], dtype=float))["k"]).astype(int)
    transform = biject_to(priors["k"].support)
    jacobian = np.asarray(jax.jacfwd(transform)(transform.inv(jnp.asarray(params["k"]))))
    out["f5_draws"] = run["draw_k"].reshape(-1, 2)
    out["f5_map"] = np.asarray(params["k"])
    out["f5_cov"] = jacobian @ cov_u[np.ix_(index, index)] @ jacobian.T
    out["f5_truth"] = np.asarray(truth["k"], dtype=float)
    out["hyper_tau"] = np.array([hyper[name]["tau"] for name in names])

    np.savez_compressed(CACHE / "arrays.npz", **out)
    summary = {
        "f4_best_scale": float(SWEEP[int(np.argmax(terms[:, 4]))]),
        "f4_chi2_monotone": bool(np.all(np.diff(terms[:, 0]) >= -1e-6 * np.abs(terms[1:, 0]))),
        "f4_ln_p_at_ml2_minus_best": float(
            terms[np.argmin(np.abs(SWEEP - 1.0)), 4] - terms[:, 4].max()
        ),
    }
    (CACHE / "arrays.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


# ---------------------------------------------------------------------------
# The figures
# ---------------------------------------------------------------------------


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig01_epochs(a: dict, theme: dict):
    plt = _mpl()
    fig, axes = plt.subplots(
        2, 2, figsize=(8.2, 4.6), sharex=True, gridspec_kw={"height_ratios": (3.2, 1.0)}
    )
    blue, orange = theme["series"][:2]
    for col, (tag, title) in enumerate(
        (("wide", "Widest separation"), ("close", "Narrowest separation"))
    ):
        j, _bjd, v1, v2 = a[f"f1_{tag}_meta"]
        wave = a[f"f1_{tag}_wave"]
        top, bottom = axes[0, col], axes[1, col]
        top.plot(wave, a[f"f1_{tag}_flux"], ".", color=theme["neutral"], ms=1.8, label="data")
        top.plot(wave, a[f"f1_{tag}_model"], color=theme["ink"], lw=1.0, label="model")
        top.plot(wave, a[f"f1_{tag}_star0"], color=blue, label="primary")
        top.plot(wave, a[f"f1_{tag}_star1"], color=orange, label="secondary")
        top.set_title(f"{title} (epoch {int(j)}: {v1:+.0f}, {v2:+.0f} km/s)", loc="left")
        bottom.plot(wave, a[f"f1_{tag}_z"], ".", color=theme["neutral"], ms=1.8)
        bottom.axhline(0.0, color=theme["ink"], lw=0.8)
        bottom.set_xlabel("wavelength [Å]")
        if col == 0:
            top.set_ylabel("normalized flux")
            bottom.set_ylabel(r"residual / $\sigma$")
        top.set_ylim(0.0, 1.15)
        bottom.set_ylim(-4.5, 4.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, markerscale=3.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


def fig02_structure(a: dict, theme: dict):
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("ramp", theme["ramp"][1:])
    cmap.set_bad(theme["surface"])
    n, n_epochs, rows, block = (int(x) for x in a["f2_meta"])
    fig, axes = plt.subplots(
        1, 3, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": (1.9, 1.4, 1.4)}
    )
    panels = (
        (
            "Design matrix A",
            a["f2_design"],
            "model pixels: primary | secondary",
            "data pixels by epoch",
        ),
        (
            r"$\Lambda + A^{T} W A$, stars in blocks",
            a["f2_by_star"],
            "primary | secondary",
            "primary | secondary",
        ),
        (
            r"$\Lambda + A^{T} W A$, pixels interleaved",
            a["f2_interleaved"],
            "interleaved pixels",
            "",
        ),
    )
    for ax, (title, matrix, xlabel, ylabel) in zip(axes, panels, strict=True):
        shown = np.where(
            matrix != 0.0, np.log10(np.abs(np.where(matrix != 0.0, matrix, 1.0))), np.nan
        )
        ax.imshow(shown, cmap=cmap, interpolation="nearest", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(theme["grid"])
    for boundary in range(1, n_epochs):
        axes[0].axhline(boundary * rows - 0.5, color=theme["muted"], lw=0.6)
    for ax in axes[:2]:
        ax.axvline(n - 0.5, color=theme["muted"], lw=0.6)
    axes[1].axhline(n - 0.5, color=theme["muted"], lw=0.6)
    for edge in range(block, 2 * n, block):
        axes[2].axhline(edge - 0.5, color=theme["muted"], lw=0.6, ls=":")
        axes[2].axvline(edge - 0.5, color=theme["muted"], lw=0.6, ls=":")
    fig.tight_layout()
    return fig


def fig03_regularization(a: dict, theme: dict):
    plt = _mpl()
    low, high = a["f3_data_range"]
    wave = a["f3_wave"]
    inside = (wave >= low) & (wave <= high)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 4.6), sharex=True, sharey="row")
    titles = (
        r"Curvature weight $\times$ 0.01",
        "ML-II curvature weight",
        r"Curvature weight $\times$ 100",
    )
    for col, title in enumerate(titles):
        for row, name in enumerate(("primary", "secondary")):
            ax = axes[row, col]
            colour = theme["series"][row]
            d, sd = a[f"f3_d_{col}"][row], a[f"f3_sd_{col}"][row]
            ax.fill_between(
                wave[inside], (d - sd)[inside], (d + sd)[inside], color=theme["soft"], lw=0
            )
            ax.plot(wave[inside], a["f3_truth"][row][inside], color=theme["muted"], lw=0.9, ls="--")
            ax.plot(wave[inside], d[inside], color=colour, lw=1.1)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{name}: deviation d")
            if row == 1:
                ax.set_xlabel("wavelength [Å]")
    handles = [
        plt.Line2D([], [], color=theme["series"][0], lw=1.1, label="recovered"),
        plt.Line2D([], [], color=theme["muted"], lw=0.9, ls="--", label="injected"),
        plt.Rectangle((0, 0), 1, 1, color=theme["soft"], label=r"$\pm 1\sigma$ band"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=3)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


def fig04_evidence(a: dict, theme: dict):
    plt = _mpl()
    scale = a["f4_scale"]
    chi2, penalty, ld_post, ld_prior, ln_p = a["f4_terms"].T
    n_good = a["f4_constant"][1]
    one = int(np.argmin(np.abs(scale - 1.0)))
    window = (scale >= 1e-3 * 0.99) & (scale <= 1e2 * 1.01)
    fit_term = -0.5 * (chi2 + penalty)
    occam = -0.5 * (ld_post - ld_prior)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    left, right = axes
    blue, orange, green = theme["series"]
    left.plot(scale[window], chi2[window] / n_good, color=blue, marker="o")
    left.axhline(1.0, color=theme["muted"], lw=1.0, ls="--")
    left.axvline(1.0, color=green, lw=1.0, ls=":")
    left.set_xscale("log")
    left.set_xlabel("curvature weight / fitted value")
    left.set_ylabel(rf"$\chi^2 / N$ of the data ($N$ = {n_good:.0f} pixels)")
    left.set_title("Data misfit at the regularized solution")
    shown = {
        r"$-(\chi^2 + \mathrm{penalty})/2$": (fit_term, blue, 1.4),
        "Occam term": (occam, orange, 1.4),
        r"$\ln p(y\,|\,\theta)$": (ln_p, theme["ink"], 2.2),
    }
    for label, (values, colour, width) in shown.items():
        right.plot(
            scale[window], (values - values[one])[window], color=colour, lw=width, label=label
        )
    right.axvline(1.0, color=green, lw=1.0, ls=":", label="fitted weight (ML-II)")
    right.set_xscale("log")
    right.set_ylim(-1600.0, 1100.0)
    right.set_xlabel("curvature weight / fitted value")
    right.set_ylabel(r"change in $\ln L$ [nats]")
    right.set_title("The marginal likelihood and its two parts")
    right.legend(loc="lower left")
    fig.tight_layout()
    return fig


def fig05_semi_amplitudes(a: dict, theme: dict):
    plt = _mpl()
    draws, centre, cov, truth = a["f5_draws"], a["f5_map"], a["f5_cov"], a["f5_truth"]
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot(
        draws[:, 0],
        draws[:, 1],
        ".",
        color=theme["series"][0],
        ms=2.0,
        alpha=0.5,
        label="NUTS draws",
    )
    values, vectors = np.linalg.eigh(cov)
    angle = np.linspace(0.0, 2.0 * np.pi, 200)
    for level, label in ((2.30, "Laplace 68%"), (6.18, "Laplace 95%")):
        ring = vectors @ (
            np.sqrt(level * values)[:, None] * np.stack([np.cos(angle), np.sin(angle)])
        )
        ax.plot(
            centre[0] + ring[0],
            centre[1] + ring[1],
            color=theme["series"][1],
            lw=1.2,
            ls="-" if level < 3 else "--",
            label=label,
        )
    ax.plot(*truth, marker="+", color=theme["ink"], ms=12, mew=1.6, ls="none", label="injected")
    ax.set_xlabel(r"$K_1$ [km/s]")
    ax.set_ylabel(r"$K_2$ [km/s]")
    ax.set_title("Semi-amplitudes")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


FIGURES = (
    ("fig01-epochs", fig01_epochs),
    ("fig02-structure", fig02_structure),
    ("fig03-regularization", fig03_regularization),
    ("fig04-evidence", fig04_evidence),
    ("fig05-semi-amplitudes", fig05_semi_amplitudes),
)


def draw_all() -> None:
    plt = _mpl()
    with np.load(CACHE / "arrays.npz") as data:
        arrays = {name: data[name] for name in data.files}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw in FIGURES:
        for theme_name, theme in THEMES.items():
            with plt.rc_context(style(theme)):
                fig = draw(arrays, theme)
                path = OUT / f"{name}-{theme_name}.png"
                fig.savefig(path, dpi=150)
                plt.close(fig)
            size = path.stat().st_size
            assert size < 500_000, f"{path.name} is {size} bytes, over the pre-commit limit"
        print("drew", name, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figures-only", action="store_true", help="redraw from the cache")
    parser.add_argument("--arrays", action="store_true", help="recompute the arrays only")
    args = parser.parse_args()
    if not args.figures_only:
        stale = not (CACHE / "run.json").exists() or load_run()[1]["key"] != cache_key()
        if stale and not args.arrays:
            run_fit_and_sample()
        compute_arrays()
    draw_all()


if __name__ == "__main__":
    main()
