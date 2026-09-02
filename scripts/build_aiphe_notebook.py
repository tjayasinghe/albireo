"""Regenerate the executed AI Phoenicis notebook, ``docs/tutorials/aiphe-labels.ipynb``.

Like ``build_showcase_notebook.py``, the docs render this with ``execute: false``: the
committed outputs are what the site shows, so the docs build stays offline and free of a
JAX dependency. Unlike the showcase, this notebook **cannot** be re-executed without two
downloads that are far too large to ship:

    python scripts/download_aiphe.py          # 36 HARPS spectra, ~194 MB
    python -c "import albireo; albireo.fetch_library('bosz2024-fgk-r20000')"   # ~645 MB

That is the same reason ``examples/03_hr6819_real_data.py`` and ``06_bloem.py`` are excluded
from the examples job in CI, and it is why this file exists: so a reader can see the result
on real data without acquiring the data.

    python scripts/build_aiphe_notebook.py [--postprocess-only]

Expect roughly five minutes, nearly all of it the three label fits. Everything is seeded.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import nbformat

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_showcase_notebook import SIZE_LIMIT, postprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "tutorials" / "aiphe-labels.ipynb"

MD = "markdown"
PY = "code"

CELLS: list[tuple[str, str]] = [
    (
        MD,
        """\
# Labels on real data: AI Phoenicis

Every other check on `albireo.match` scores it against a spectrum this package injected
itself. This one scores it against a star.

AI Phe (HD 6980) is a detached eclipsing binary whose orbit is published to better than
0.02 per cent, and — the part that matters here — whose components have independently
known temperatures, gravities and radii. So every quantity the label fit produces has an
external number waiting for it:

| quantity | published | source |
|---|---|---|
| Teff | 6310 K and 5010 K | Maxted et al. (2020), MNRAS 498, 332 |
| log g | 4.001 and 3.598 | derived below, from the same solution |
| R₂/R₁ | 1.6237 | from the fractional radii, run C |

The third is the one to watch. `RadiusRatio` fits both components jointly through a single
shared scalar, so the radius ratio is something the label fit *returns* — and here it can
be held against a number measured photometrically, from eclipses, which nothing in the
spectroscopic fit is told.

**This notebook cannot be re-run without the data.** It needs 36 HARPS spectra (~194 MB,
`scripts/download_aiphe.py`) and the BOSZ library (~645 MB on first use). The committed
outputs are the record.""",
    ),
    (
        PY,
        """\
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
matplotlib.rcParams["figure.dpi"] = 110

# After the path insert on purpose: the two bench modules live in scripts/.
from aiphe_bench import R1_FRAC, R2_FRAC, TEFF1, TEFF2, WINDOW, load  # noqa: E402
from aiphe_labels_bench import LIBRARY, disentangle, published_logg  # noqa: E402

import albireo as ab  # noqa: E402

print(f"albireo {ab.__version__}")""",
    ),
    (
        MD,
        """\
## log g, derived rather than remembered

For a double-lined eclipsing binary the surface gravity follows from the spectroscopic and
photometric elements alone — no absolute masses, no radii, no distance:

$$g_1 = \\frac{2\\pi\\sqrt{1-e^2}\\,K_2}{P\\,r_1^2 \\sin i}$$

which is Kepler's third law and the mass ratio with everything that cancels cancelled. It
reproduces the published absolute masses and radii to 0.002 dex, which is worth checking
rather than trusting.""",
    ),
    (
        PY,
        """\
logg_pub = published_logg()
print(f"log g  primary {logg_pub[0]:.4f}   secondary {logg_pub[1]:.4f}")
print(f"R2/R1  {R2_FRAC / R1_FRAC:.4f}")
print(f"Teff   {TEFF1:.0f} K and {TEFF2:.0f} K")""",
    ),
    (
        MD,
        """\
## Disentangle at the published orbit

The orbit is not re-derived here. It is known far better than any disentangling could
recover it, so fixing the velocities at the published solution isolates what is actually
under test: the label fit.

Note the light fractions. They are a **blackbody estimate** from the published radii and
temperatures — honest to a few per cent at best. That makes this a fair test of the claim
that a wrong assumed dilution comes back as dilution rather than as temperature.""",
    ),
    (
        PY,
        """\
dataset = load(Path("data/aiphe"))
print(dataset.summary())

t0 = time.perf_counter()
grid, d_hat, std, ell = disentangle(dataset)
print(f"\\ndisentangled {grid.n} px in {time.perf_counter() - t0:.1f} s")
print(f"light fractions assumed: {ell[0]:.4f} / {ell[1]:.4f}")""",
    ),
    (
        PY,
        """\
fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.0), sharex=True)
for i, (ax, name) in enumerate(zip(axes, ("primary (F7 V)", "secondary (K0 IV)"), strict=True)):
    ax.fill_between(grid.wave, 1 + d_hat[i] - 2 * std[i], 1 + d_hat[i] + 2 * std[i],
                    color=f"C{i}", alpha=0.3, lw=0)
    ax.plot(grid.wave, 1 + d_hat[i], color=f"C{i}", lw=0.8)
    ax.set_ylabel(name, fontsize=9)
    ax.set_xlim(*WINDOW)
axes[-1].set_xlabel("wavelength [Å]")
axes[0].set_title(r"AI Phe, disentangled on the Mg I b window, with the $\\pm 2\\sigma$ band")
fig.set_layout_engine("constrained")""",
    ),
    (
        MD,
        """\
## Fit the labels

`log g` is declared, not fitted. On an eclipsing binary it is known to ~0.003 dex, an order
of magnitude better than any spectroscopic determination — and Teff and log g correlate at
about 0.98 when both are free. Declaring it is the single most consequential choice on the
page, and the next cell shows what happens without it.""",
    ),
    (
        PY,
        """\
pad = 2.0
library = ab.fetch_library(
    LIBRARY, wave_range=(float(grid.wave[0]) - pad, float(grid.wave[-1]) + pad), progress=False
)
print(f"{library.nodes.shape[0]} nodes x {library.wave.size} px, medium {library.medium!r}")


def stars(logg_spec):
    return {
        "primary": ab.StarLabels(
            library=library, teff=ab.Between(5400.0, 6900.0), logg=logg_spec(logg_pub[0]),
            vsini=ab.Between(0.1, 30.0), v_kms=ab.Between(-15.0, 15.0)),
        "secondary": ab.StarLabels(
            library=library, teff=ab.Between(4300.0, 5900.0), logg=logg_spec(logg_pub[1]),
            vsini=ab.Between(0.1, 30.0), v_kms=ab.Between(-15.0, 15.0)),
    }


common = dict(medium=dataset.epochs[0].medium, light_fractions=ell,
              lsf_sigma_kms=ab.LSF.from_resolution(115000.0).sigma_kms,
              std=std, mh=ab.Between(-0.9, 0.4), max_steps=400)

fixed = ab.match_labels(grid, d_hat, stars=stars(ab.Fixed),
                        dilution=ab.RadiusRatio(), **common)
print(fixed.summary())""",
    ),
    (
        PY,
        """\
def score(match, label):
    rows = []
    for name, truth in (("primary", TEFF1), ("secondary", TEFF2)):
        got = match.labels[name]["teff"]
        rows.append((label, name, got, got - truth, 100 * (got - truth) / truth))
    return rows


table = score(fixed, "log g fixed")
print(f"{'configuration':<16}{'component':<12}{'Teff [K]':>10}{'error':>10}{'':>8}")
for label, name, got, dt, pct in table:
    print(f"{label:<16}{name:<12}{got:10.1f}{dt:+10.1f}{pct:+7.2f}%")
print(f"\\nR2/R1 fitted {fixed.radius_ratio['secondary']:.4f}  "
      f"published {R2_FRAC / R1_FRAC:.4f}  "
      f"({100 * (fixed.radius_ratio['secondary'] / (R2_FRAC / R1_FRAC) - 1):+.1f}%)")""",
    ),
    (
        MD,
        """\
## What happens when log g is free

This is the failure the tutorial warns about, on real data. Teff and log g trade against
each other; the fit slides down the degeneracy, log g runs to the bottom of its prior, and
the temperatures follow it down. χ² *improves* while it happens — a better fit and worse
physics.

And note what does **not** catch it: the correlation report comes back empty. Once log g has
reached the edge of the grid it stops varying, so the curvature at the optimum no longer
shows the degeneracy that produced the answer. A flagged correlation is evidence; an empty
one is not absence of evidence. What catches this is having an external log g to compare
against — which is the whole argument for declaring it.""",
    ),
    (
        PY,
        """\
free = ab.match_labels(grid, d_hat, stars=stars(lambda g: ab.Between(3.0, 4.9)),
                       dilution=ab.RadiusRatio(), **common)
for name in ("primary", "secondary"):
    got, pub = free.labels[name], logg_pub["primary secondary".split().index(name)]
    print(f"{name:<10} Teff {got['teff']:7.1f}   log g {got['logg']:.3f} "
          f"(published {pub:.3f}, {got['logg'] - pub:+.3f})")
print(f"\\nchi2 free {free.chi2:.0f}  vs  fixed {fixed.chi2:.0f}")
print("strong correlations:", free.flagged_correlations())""",
    ),
    (
        MD,
        """\
## What the assumed dilution was doing

`FixedDilution` freezes the light fractions at the blackbody estimate. The difference
between the two fits measures how hard that assumption was bending the temperatures — which
is the whole reason the dilution is fitted jointly rather than taken on trust.""",
    ),
    (
        PY,
        """\
rigid = ab.match_labels(grid, d_hat, stars=stars(ab.Fixed),
                        dilution=ab.FixedDilution(), **common)
print(f"{'':<18}{'primary':>22}{'secondary':>22}")
for label, m in (("radius ratio fitted", fixed), ("dilution frozen", rigid)):
    cells = "".join(f"{m.labels[n]['teff']:10.1f} ({m.labels[n]['teff'] - t:+6.1f})"
                    for n, t in (("primary", TEFF1), ("secondary", TEFF2)))
    print(f"{label:<18}{cells}")
print(f"\\nlight fractions   fitted {fixed.flux_ratio['primary']:.4f}/"
      f"{fixed.flux_ratio['secondary']:.4f}   assumed {ell[0]:.4f}/{ell[1]:.4f}")
print(f"chi2              fitted {fixed.chi2:.0f}   frozen {rigid.chi2:.0f}")""",
    ),
    (
        MD,
        """\
## Reading the result honestly

The primary lands within a per cent of its published temperature. The secondary does not:
it comes back several hundred kelvin hot, outside the 2–3 per cent this mode claims, and
the fitted radius ratio is about 6 per cent low. Two things were tested and one of them
explained something.

**Microturbulence was the obvious suspect, and it is not the answer.** The library is
pinned at ξ = 2 km s⁻¹, and a K subgiant wants nearer 1.3; too much microturbulence makes
the model's lines too strong, which the fit could answer by raising Teff. Rebuilding the
library at ξ = 1 km s⁻¹ moves the lines by 8.5 per cent in equivalent width — the right
direction — but the secondary's temperature got *worse*, not better, and χ² with it. The
ξ change went into [M/H] instead (+0.10 dex), which is the documented [M/H]–ξ degeneracy
doing what it is documented to do.

**The comparison mode was not the obvious suspect, and it changed the default.** `matched`
convolves both the model and the data with the LSF before comparing, on the argument that
`d_hat` is a partial deconvolution. That argument is right about the deconvolution and
wrong about the cost: convolving the residuals correlates them, while the likelihood stays
diagonal. On this dataset it inflated χ² by 4.26× where the kernel predicts 4.91×, and
`v sin i` absorbed the mis-specification — both components pinned to the floor of their
prior. `native` returns 2.2 km s⁻¹ for both, which is at least physical. The default is now
`native`; the closed-loop test never saw this, because its rows have no LSF and no
disentangling behind them.

What is left unexplained is most of the secondary's offset. Candidates in order: a 100 Å
window carrying far more temperature leverage for an F star than for a K subgiant; the
published 5010 K itself being a photometric/SED temperature rather than a spectroscopic
one; and the assumed light fractions, which the radius ratio only partly absorbs.""",
    ),
    (
        PY,
        """\
for name in ("primary", "secondary"):
    print(f"{name:<10} vsini {fixed.labels[name]['vsini']:5.2f} km/s   "
          f"[M/H] {fixed.labels[name]['mh']:+.3f}   "
          f"v {fixed.labels[name]['v_kms']:+6.2f} km/s")
print(f"\\nnulls   chi2 {fixed.chi2:.4g}   nearest node {fixed.chi2_nearest_node:.4g}   "
      f"no template {fixed.chi2_continuum:.4g}")""",
    ),
    (
        MD,
        """\
## The template, which is the point

The labels exist to select a template, and `LabelMatch.template` renders one: the
interpolated model at the fitted labels, broadened and shifted as fitted, **undiluted** —
a template is the star, not the star's share of the system's light.

It comes back as flux on the fit's own grid. Writing it to the file formats the downstream
cross-correlation codes read is `albireo.handoff`'s job and is not built yet.""",
    ),
    (
        PY,
        """\
template = fixed.template("primary")
print(f"{template.shape[0]} px on the fit grid, flux {template.min():.3f}-{template.max():.3f}")

fig, ax = plt.subplots(figsize=(9.5, 3.0))
ax.plot(grid.wave, 1 + d_hat[0] / ell[0], color="0.55", lw=0.8,
        label="disentangled primary, undiluted")
ax.plot(grid.wave, template, color="C3", lw=0.9, label="fitted template")
ax.set_xlim(5180, 5220)
ax.set_xlabel("wavelength [Å]")
ax.legend(fontsize=8)
ax.set_title("The recovered component and the template the labels select")
fig.set_layout_engine("constrained")""",
    ),
    (
        MD,
        """\
## What to take from this

The mode does what it was scoped to do — pick a template — and the validation says how
well, on a real star, with the failures named rather than tuned away:

- The **primary** is recovered to within 1 per cent in Teff.
- The **secondary** is several hundred kelvin hot, outside the claimed accuracy, and the
  reason is not yet fully established.
- The **radius ratio**, which the fit was never told, comes back about 5 per cent low —
  from spectroscopy alone, against a photometric measurement.
- **log g must be declared** when the system gives it to you. Free, it takes the
  temperatures with it, reports a lower χ² while doing so, and leaves the correlation
  report empty because it ended up against a bound.
- Formal errors here are sub-kelvin and should be ignored. They are the curvature of an
  optimum, not an uncertainty; `refit_draws` is the number to quote.

The last point is not a caveat added for modesty. `summary()` prints it every time.""",
    ),
]


def build() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    for kind, source in CELLS:
        nb.cells.append(
            nbformat.v4.new_markdown_cell(source)
            if kind == MD
            else nbformat.v4.new_code_cell(source)
        )
    return nb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args(argv)

    if args.postprocess_only:
        nb = nbformat.read(OUT, as_version=4)
    else:
        from nbclient import NotebookClient

        nb = build()
        t0 = time.perf_counter()
        NotebookClient(
            nb,
            timeout=3600,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO)}},
        ).execute()
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
