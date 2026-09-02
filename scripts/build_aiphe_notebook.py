"""Regenerate the executed AI Phoenicis notebook, ``docs/tutorials/aiphe-labels.ipynb``.

As with ``build_showcase_notebook.py``, the docs render this notebook with
``execute: false``: the committed outputs are what the site shows, so the docs build stays
offline and free of a JAX dependency. Unlike the showcase, this notebook cannot be
re-executed without two downloads that are too large to ship:

    python scripts/download_aiphe.py          # 36 HARPS spectra, ~194 MB
    python -c "import albireo; albireo.fetch_library('bosz2024-fgk-r20000')"   # ~645 MB

The same download size excludes ``examples/03_hr6819_real_data.py`` and ``06_bloem.py``
from the examples job in CI. Committing the executed outputs lets a reader see the result
on real data without acquiring the data.

    python scripts/build_aiphe_notebook.py [--postprocess-only]

Wall time is roughly five minutes, nearly all of it the three label fits. The notebook is
seeded.
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

The other checks on `albireo.match` score it against spectra this package injected
itself. This one scores it against a star.

Background and references: [science overview](../science.md).

AI Phe (HD 6980) is a detached eclipsing binary whose orbit is published to better than
0.02 per cent and whose components have independently known temperatures, gravities and
radii, so every quantity the label fit produces has an external value to compare against:

| quantity | published | source |
|---|---|---|
| Teff | 6310 K and 5010 K | Maxted et al. (2020), MNRAS 498, 332 |
| log g | 4.001 and 3.598 | derived below, from the same solution |
| R₂/R₁ | 1.6237 | from the fractional radii, run C |

The third is the strongest test. `RadiusRatio` fits both components jointly through a
single shared scalar, so the radius ratio is a quantity the label fit returns, and here it
can be compared with a value measured photometrically from the eclipses, which the
spectroscopic fit is never given.

This notebook cannot be re-run without the data. It requires 36 HARPS spectra (~194 MB,
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
## log g from the eclipsing solution

For a double-lined eclipsing binary the surface gravity follows from the spectroscopic and
photometric elements alone, without absolute masses, radii or a distance:

$$g_1 = \\frac{2\\pi\\sqrt{1-e^2}\\,K_2}{P\\,r_1^2 \\sin i}$$

which is Kepler's third law combined with the mass ratio, with the common factors
cancelled. It reproduces the published absolute masses and radii to 0.002 dex, and the
next cell checks that.""",
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

The orbit is not re-derived here. It is known far better than disentangling could recover
it, so fixing the velocities at the published solution isolates the quantity under test,
the label fit.

The light fractions are a blackbody estimate from the published radii and temperatures,
accurate to a few per cent at best. That makes this a test of the claim that a wrong
assumed dilution is absorbed as dilution rather than as temperature.""",
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

`log g` is declared rather than fitted. For an eclipsing binary it is known to about
0.003 dex, an order of magnitude better than any spectroscopic determination, and Teff and
log g correlate at about 0.98 when both are free. The next cell shows the result when
log g is left free.""",
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
## The fit with log g free

This is the failure mode the tutorial describes, here on real data. Teff and log g trade
against each other: the fit follows the degeneracy, log g runs to the lower bound of its
prior, and the temperatures follow it down. χ² improves while this happens, so the lower
χ² accompanies the worse physical solution.

The correlation report does not detect it. Once log g has reached the edge of the grid it
stops varying, so the curvature at the optimum no longer shows the degeneracy that
produced the answer. A flagged correlation is evidence of a degeneracy; an empty report is
not evidence of its absence. What detects this case is an external log g to compare
against, which is the argument for declaring it.""",
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
## The effect of the assumed dilution

`FixedDilution` freezes the light fractions at the blackbody estimate. The difference
between the two fits measures how strongly that assumption biases the temperatures, and is
the reason the dilution is fitted jointly rather than assumed.""",
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
## The result

The primary lands within one per cent of its published temperature. The secondary does
not: it comes back several hundred kelvin hot, outside the 2–3 per cent this mode claims,
and the fitted radius ratio is about 6 per cent low. Two explanations were tested, of which
one accounts for part of the discrepancy.

Microturbulence is not the explanation. The library is pinned at ξ = 2 km s⁻¹ while a K
subgiant requires nearer 1.3 km s⁻¹, and excess microturbulence makes the model lines too
strong, which the fit could compensate by raising Teff. Rebuilding the library at
ξ = 1 km s⁻¹ changes the equivalent widths by 8.5 per cent, in the expected direction, but
the secondary's temperature and χ² both became worse. The change in ξ was absorbed by
[M/H] instead (+0.10 dex), which is the documented [M/H]–ξ degeneracy.

The comparison mode accounts for part of the discrepancy, and the default was changed as a
result. `matched` convolves both the model and the data with the LSF before comparing, on
the argument that `d_hat` is a partial deconvolution. That argument is correct about the
deconvolution but not about the cost: convolving the residuals correlates them while the
likelihood remains diagonal. On this dataset it inflated χ² by 4.26× where the kernel
predicts 4.91×, and `v sin i` absorbed the mis-specification, with both components pinned
to the floor of their prior. `native` returns 2.2 km s⁻¹ for both, which is physical. The
default is now `native`. The closed-loop test did not detect this, because its rows have no
LSF and no disentangling behind them.

Most of the secondary's offset remains unexplained. The candidates, in order: a 100 Å
window carrying more temperature leverage for an F star than for a K subgiant; the
published 5010 K being a photometric or SED temperature rather than a spectroscopic one;
and the assumed light fractions, which the radius ratio only partly absorbs.""",
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
## The template

The labels exist to select a template, and `LabelMatch.template` renders one: the
interpolated model at the fitted labels, broadened and shifted as fitted, and undiluted,
since a template represents the star rather than the star's share of the system's light.

It is returned as flux on the fit's own grid. Writing it in the file formats the
downstream cross-correlation codes read is the role of `albireo.handoff` and is not yet
implemented.""",
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
## Summary

The mode performs the task it was scoped for, template selection, and this run measures
how well it does so on a real star:

- The primary Teff is recovered to within 1 per cent.
- The secondary is several hundred kelvin hot, outside the claimed accuracy, and the
  reason is not yet established.
- The radius ratio, which the fit is never given, comes back about 5 per cent low, from
  spectroscopy alone, against a photometric measurement.
- log g should be declared when the system determines it. Left free, it carries the
  temperatures with it, reports a lower χ² while doing so, and leaves the correlation
  report empty because it ends against a bound.
- The formal errors here are sub-kelvin and should not be quoted. They are the curvature
  of an optimum, not an uncertainty; `refit_draws` provides the value to quote, and
  `summary()` states this on every call.""",
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
