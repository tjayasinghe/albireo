"""Regenerate the executed showcase notebook, ``docs/tutorials/showcase.ipynb``.

The docs build renders the notebook with ``execute: false`` (see ``mkdocs.yml``): the
committed outputs are what the site shows, so building the docs stays cheap, offline and
free of a JAX-plus-sampling dependency. The price is that the outputs go stale when the
API or the packaged example changes, and this script is how they are refreshed:

    python scripts/build_showcase_notebook.py

Expect roughly ten minutes of wall time, nearly all of it the NUTS cell. Everything in
the notebook is seeded, so apart from the printed timings a rebuild is reproducible.

Two post-processing passes run after execution, and ``--postprocess-only`` applies them
to the existing file without re-executing:

* **Environment noise is stripped.** A kernel without ipywidgets emits a TqdmWarning
  ("IProgress not found") on stderr at import time. It is a fact about the executing
  environment, not about albireo, and it would sit as a red block at the top of the
  rendered page.
* **Figures are palette-quantized.** Matplotlib's inline PNGs are 32-bit RGBA; flattened
  onto white and quantized to a 256-color palette they are visually identical for line
  plots and roughly a third the size. This is what keeps the notebook under the 500 kB
  pre-commit file-size limit — the gate at the end fails the build if it is exceeded.
"""

from __future__ import annotations

import argparse
import base64
import io
import pathlib
import sys
import time

import nbformat

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "tutorials" / "showcase.ipynb"
SIZE_LIMIT = 500 * 1024  # the pre-commit check-added-large-files default

MD = "markdown"
PY = "code"

CELLS: list[tuple[str, str]] = [
    (
        MD,
        """\
# albireo, end to end: a tour of the outputs

One notebook, every headline figure. It runs the example dataset that ships inside the
package — no download, no network, no data of your own — through the `Disentangler`
façade, and shows what comes back at each stage: the declaration's derivations, the
MAP + ML-II fit summary, the disentangled spectra with their uncertainty band, the
residual diagnostics, the NUTS posterior over the orbit, spectra drawn from the joint
posterior, and a sensitivity forecast for epochs that have not been taken yet.

Install with the plotting extra:

```
pip install "albireo[plots]"
```

Two honesty notes, both of which the package will repeat at you in its own summaries:

- The continuum light fractions are **assumed, not measured**. With constant light the
  data only ever constrain the products `l_i * d_i`, so the fractions are an input the
  fit cannot contradict.
- Each component's smooth envelope is set by the prior, not by the data (the `k = 0`
  degeneracy). The uncertainty band is what says so — read the band, not just the mean.

The saved outputs, timings included, are from a single run on a 16-core desktop; the
absolute times will differ on your machine, and every first call includes JAX
compilation. Everything is seeded, so the numbers reproduce.""",
    ),
    (
        PY,
        """\
import time

import matplotlib
import numpy as np

import albireo as ab

matplotlib.rcParams["figure.dpi"] = 110

print(f"albireo {ab.__version__}")

dataset, truth = ab.load_example("sb2_sim", with_truth=True)
print(dataset.summary())
print(f"\\ninjected truth: P = {truth['period']} d, e = {truth['ecc']}, K = {truth['k']} km/s")""",
    ),
    (
        MD,
        """\
## Declare the system

The façade is a compiler, not a shortcut. You declare the *system* — components, orbit
priors, instrument LSF — and it derives the machinery the expert path makes you supply
by hand: the velocity budget from the priors' own support, the grid margin, the
conjunction phase (scanned before anything is optimized), and the smoothness
hyperparameters by ML-II. `explain()` prints every derivation, and `expert()` hands back
the exact `(model, priors, init)` triple, so nothing is hidden behind the convenience.""",
    ),
    (
        PY,
        """\
light = truth["light_fractions"]  # assumed, not measured: only l_i * d_i is observable

dis = ab.Disentangler(
    dataset,
    components=[
        ab.Star("primary", light=float(light[0])),
        ab.Star("secondary", light=float(light[1])),
    ],
    orbit=ab.Orbit(
        period=ab.Between(5.5, 6.5),
        k=ab.Between([10.0, 10.0], [90.0, 90.0]),
        ecc=ab.Between(0.0, 0.64),
    ),
    lsf={"DEMO": 6.5},
    dv_kms=4.0,
)
print(dis.explain())""",
    ),
    (
        MD,
        """\
## Fit: MAP plus ML-II, in one call

The summary is the artifact: the optimizer's own report, the conjunction scan contrast,
the orbit, the ML-II smoothness table (with a flag on any hyperparameter the data did
not move), the residual z-score RMS — near 1 when the noise model describes the data —
and, always, the assumptions block.""",
    ),
    (
        PY,
        """\
t0 = time.perf_counter()
fit = dis.fit()
print(f"[{time.perf_counter() - t0:.1f} s, including JAX compilation]\\n")
print(fit.summary())""",
    ),
    (
        MD,
        """\
## The disentangled spectra, with the band that keeps them honest

The component spectra are recovered as deviations from a unit continuum, conditional on
the MAP orbit, with a pointwise uncertainty band. Wherever the epochs give little
leverage the band widens toward the prior — that is the figure saying "the data did not
decide this", which a mean line on its own never admits. The model grid is deliberately
wider than the data (a velocity-budget-plus-LSF margin the façade derived above), so the
band climbing in the wings is expected rather than alarming: those pixels are prior-only.

The injected truth lives on the grid the example was generated on, so it is resampled
onto the model grid for the overlay — zero deviation outside its window, which is
exactly what the simulation put there.""",
    ),
    (
        PY,
        """\
truth_grid = ab.LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))
truth_on_model = np.stack(
    [
        np.interp(dis.grid.wave, truth_grid.wave, component, left=0.0, right=0.0)
        for component in truth["components"]
    ]
)

fig, axes = ab.plot_spectra(dis.grid, fit.spectra(), std=fit.std(), truth=truth_on_model)
axes[0].set_title("MAP component spectra with the +/- 2 sigma band, against the injected truth")
fig.set_layout_engine("constrained")""",
    ),
    (
        MD,
        """\
## Residual diagnostics

Whitened residuals, three ways: their distribution against a unit normal, the per-epoch
RMS, and the lag-1 autocorrelation within each exposure. A z-RMS near 1 with flat
per-epoch structure is the model check; phase-dependent structure would mean the model
is absorbing something it should be describing.""",
    ),
    (
        PY,
        """\
fig, _ = ab.plot_residual_zscores(
    dis.model.problem_at(fit.theta), fit.marginal().d_hat, bjd=dataset.bjd
)
fig.set_layout_engine("constrained")""",
    ),
    (
        MD,
        """\
## The posterior over the orbit

This is the reason the package exists: the component spectra are marginalized out
analytically, so NUTS samples only the orbital sites. `fit.sample()` freezes the
smoothness hyperparameters at their ML-II values (a plug-in approximation — the summary
says so rather than hoping you knew) and uses the Laplace covariance at the MAP as the
mass matrix, so warmup only has to tune the step size. Two chains run sequentially on a
CPU.""",
    ),
    (
        PY,
        """\
t0 = time.perf_counter()
post = fit.sample(seed=0)
print(f"[{time.perf_counter() - t0:.1f} s]\\n")
print(post.summary())""",
    ),
    (
        MD,
        """\
## The orbit, drawn rather than tabulated

albireo never measures a per-epoch radial velocity — the orbit is inferred from the
spectra directly — so this is the posterior *curve*, not a fit through RV points. The
open circles are the injected component velocities at the observed epochs, which the
draws should (and do) thread; the ticks along the bottom mark the epochs' phase
coverage, the thing that actually determines how well the orbit is constrained.""",
    ),
    (
        PY,
        """\
samples = post.samples
fig, ax = ab.plot_rv_curve(samples, dataset.bjd)

period = float(np.mean(np.asarray(samples["period"])))
t_conj = float(np.mean(np.asarray(samples["t_conj"])))
phase = ((dataset.bjd - t_conj) / period) % 1.0
for i in range(2):
    ax.plot(
        phase,
        truth["velocities"][i],
        "o",
        ms=5,
        mfc="none",
        mec="k",
        label="injected truth at the epochs" if i == 0 else None,
    )
ax.legend(fontsize=8)
ax.set_title("posterior orbit draws; no per-epoch RV was ever measured")""",
    ),
    (
        MD,
        """\
## The pairwise posterior

The sampled space is small — the spectra are integrated out — so the corner plot is
readable by default. Shown here: period, eccentricity, and both semi-amplitudes.""",
    ),
    (
        PY,
        """\
idata = post.to_inference_data()
_ = ab.plot_corner(idata, var_names=["period", "ecc", "k"])""",
    ),
    (
        MD,
        """\
## Spectra from the joint posterior

Each draw picks a posterior orbit and then draws once from the conditional Gaussian over
the spectra, so the scatter carries the orbital *and* the spectral uncertainty together.
This is the object to propagate downstream: equivalent widths measured on these draws
inherit the `k = 0` exchange between the components, which independent per-pixel error
bars would miss entirely.""",
    ),
    (
        PY,
        """\
draws = post.spectra(num_draws=32)
fig, axes = ab.plot_spectra(dis.grid, draws, truth=truth_on_model)
axes[0].set_title("32 joint-posterior draws: the band is spectral + orbital uncertainty")
fig.set_layout_engine("constrained")""",
    ),
    (
        MD,
        """\
## Forecast: what would six more nights buy?

The posterior covariance of the spectra contains no flux — only the epochs, their
phases, weights and the prior — so it can be computed for observations that do not exist
yet. Planned epochs carry a placeholder flux of exactly 1.0, so a planned dataset fed to
a *fit* by mistake returns featureless spectra: visibly wrong rather than plausible.
Here: the twelve epochs in hand, plus six more spread over one period, forecast with the
fitted orbit and the ML-II smoothness the fit just measured.""",
    ),
    (
        PY,
        """\
theta_orbit = {site: fit.params[site] for site in ("period", "t_conj", "secosw", "sesinw", "k")}
stars = [fit.star(name) for name in ("primary", "secondary")]
prior = ab.SmoothnessPrior(
    tau=np.array([s["tau"] for s in stars]),
    eta=np.array([s["eta"] for s in stars]),
)

planned_bjd = dataset.bjd.max() + 1.0 + period * np.linspace(0.0, 5.0 / 6.0, 6)
planned = ab.plan_epochs(dataset.epochs[0], planned_bjd)
design = ab.Dataset(dataset.epochs + planned, frame=dataset.frame)

fc = ab.sensitivity_forecast(
    dis.grid,
    design,
    orbit=theta_orbit,
    light_fractions=truth["light_fractions"],
    lsf_sigma_v={"DEMO": 6.5},
    prior=prior,
    baseline=list(range(dataset.n_epochs)),
)
print(fc.summary())
fig, _ = ab.plot_forecast(fc)""",
    ),
    (
        MD,
        """\
## What this notebook did not show

Kept out to stay small, not because it is exotic — each has a runnable example in the
repository:

- the SB1 faint-companion scan and its calibrated detection limit
  ([`examples/02_k2_scan.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/02_k2_scan.py),
  [`examples/05_detection_limit.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/05_detection_limit.py));
- the nebular component — unmodelled contamination reaches the *masses*, not just the
  line depths
  ([`examples/04_nebular.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/04_nebular.py));
- the free per-epoch RV table for systems with no known period
  ([`examples/09_rv_table.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/09_rv_table.py)),
  and real survey data end to end in the BLOeM tutorial;
- the handoff to atmosphere codes with the uncertainty attached
  ([`examples/10_downstream.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/10_downstream.py)).

Every claim in the prose above is measured somewhere in `docs/benchmarks.md`, and the
reasoning behind the design lives in `docs/design.md` and `docs/math.md`.""",
    ),
]

# stderr fragments that describe the executing kernel rather than albireo. Anything
# matching is dropped from the saved outputs; everything else on stderr is kept, because
# a real warning from the package belongs on the rendered page.
ENVIRONMENT_NOISE = (
    "IProgress not found",
    "Proactor event loop does not implement add_reader",
)


def build() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    for kind, source in CELLS:
        if kind == MD:
            nb.cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(source))
    return nb


def strip_environment_noise(nb: nbformat.NotebookNode) -> int:
    """Drop stderr outputs that are about the kernel environment, not the package."""
    removed = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        kept = []
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", "")) if out.get("output_type") == "stream" else ""
            if out.get("name") == "stderr" and any(m in text for m in ENVIRONMENT_NOISE):
                removed += 1
                continue
            kept.append(out)
        cell["outputs"] = kept
    return removed


def quantize_pngs(nb: nbformat.NotebookNode) -> int:
    """Flatten inline PNGs onto white and quantize to a 256-color palette.

    Visually identical for line plots, roughly a third the bytes, and the flattening is
    deliberate: matplotlib's inline figures have a transparent background, on which the
    default black text is unreadable in a dark-themed viewer anyway.
    """
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed; skipping PNG quantization (the size gate may fail).")
        return 0

    saved = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            payload = out.get("data", {}).get("image/png")
            if not payload:
                continue
            raw = base64.b64decode(payload)
            image = Image.open(io.BytesIO(raw))
            if image.mode in ("RGBA", "LA"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")
            quantized = image.convert("P", palette=Image.ADAPTIVE, colors=256)
            buffer = io.BytesIO()
            quantized.save(buffer, format="PNG", optimize=True)
            packed = buffer.getvalue()
            if len(packed) < len(raw):
                out["data"]["image/png"] = base64.b64encode(packed).decode("ascii")
                saved += len(raw) - len(packed)
    return saved


def postprocess(nb: nbformat.NotebookNode) -> None:
    removed = strip_environment_noise(nb)
    saved = quantize_pngs(nb)
    print(
        f"post-process: {removed} environment-noise output(s) removed, "
        f"{saved / 1024:.0f} KiB saved by PNG quantization"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="re-apply the post-processing passes to the existing notebook, no execution",
    )
    args = parser.parse_args(argv)

    if args.postprocess_only:
        nb = nbformat.read(OUT, as_version=4)
    else:
        from nbclient import NotebookClient

        nb = build()
        t0 = time.perf_counter()
        client = NotebookClient(
            nb,
            timeout=3600,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO)}},
        )
        client.execute()
        print(f"executed in {time.perf_counter() - t0:.1f} s")

    postprocess(nb)
    nbformat.write(nb, OUT)
    size = OUT.stat().st_size
    print(f"wrote {OUT.relative_to(REPO)} ({size / 1024:.0f} KiB)")
    if size >= SIZE_LIMIT:
        print(f"FAIL: {size} bytes is over the {SIZE_LIMIT} pre-commit file-size limit.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
