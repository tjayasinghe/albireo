# From a disentangled spectrum to an honest error bar on log *g*

The disentangled spectrum is almost never the result. It is fed to an atmosphere code —
GSSP, iSpec, Korg.jl, PySME — and what comes out of *that* is what ends up in the table:
effective temperature, surface gravity, abundances. The uncertainty on the disentangled
spectrum is dropped at exactly that joint, and the papers say so themselves:

> We stress that the uncertainties that could arise from the normalisation procedure are not
> taken into account in the global uncertainties on the presented properties
>
> — Mahy et al. (2020), TMBM III, §3.1

> Propagation of uncertainties through this process is difficult so must be tackled
> numerically.
>
> — Pavlovski, Southworth & Tamajo (2018)

This page is about closing that gap. It has two halves: the half albireo can run for you,
and the half that needs an atmosphere code installed.

!!! tip "When four labels are enough"

    If what you need is a *template* — the right synthetic spectrum to cross-correlate your
    individual epochs against, not an abundance table — there is a shorter route that stays
    inside albireo and never drops the uncertainty at a file boundary:
    [Turn a component into an RV template](labels.md). It fits Teff, log g, [M/H] and
    *v* sin *i* against a published grid, propagates the spectral posterior by refitting its
    draws, and measures the light ratio on the way.

    It does not replace this page. It fits four labels and nothing else: no abundances, no
    microturbulence, no bespoke synthesis. When the answer belongs in a table, you still want
    an atmosphere code, and everything below is how to reach one without losing the error bar.

## Why the file format is the hard part

GSSP and iSpec disagree about nearly everything that can be got wrong quietly.

| | GSSP | iSpec |
|---|---|---|
| Layout | 2 columns, whitespace | 3 columns, **tab**, one header line |
| Wavelength unit | **ångström** | **nanometre** |
| Grid | must be **equidistant** | as given |
| Per-pixel error | **none — there is no column** | `err`, absolute 1σ |
| Flux | normalized | normalized |

Two of those are traps rather than differences.

**iSpec does no unit conversion on the text path.** Its entire internal scale is nanometres,
line lists included. A wavelength written in ångström lands a factor of ten outside every
model grid — and still fits *something*. `write_ispec` divides by ten, and both the unit and
the round trip are regression-tested.

**GSSP infers its synthetic step from your file**: *"the step width in wavelength that will
be used for the calculation of synthetic spectra is computed from the observations"*
(Tkachenko 2015, Appendix B.2 — which is the entire manual; there is no separate document
and no source repository). albireo solves on a log-wavelength grid, whose linear spacing
drifts across the window — 1.32% on the packaged example. Dumped as-is, GSSP would take the
first pixel pair as the step for the whole spectrum. `write_gssp` resamples onto an
equidistant grid, and applies the identical grid to every draw so they stay comparable.

```python
import albireo as ab

ab.write_gssp("component.dat", grid, d_hat)              # -> component_1.dat, component_2.dat
ab.write_ispec("component.txt", grid, d_hat, std)        # -> component_1.txt, component_2.txt
```

## The half that matters: N draws, not one spectrum

GSSP has nowhere to put a per-pixel uncertainty. Not "it ignores one" — its configuration
files contain no error path, no signal-to-noise entry and no weighting entry, and its own
quoted error bars come from χ² on the fit residuals. So the posterior band **cannot** reach
a temperature through the file. It can only get there by fitting many spectra:

```python
draws = ab.draw_spectra(marginal, jax.random.key(0), 100)   # (100, n_comp, n_pix)
paths = ab.export_draws("draws/", grid, draws, format="gssp")
# draws/draw_0000_1.dat, draws/draw_0000_2.dat, draws/draw_0001_1.dat, ...
```

Then fit every file with the same grid, the same line list, the same masks and the same
starting guess, and take the spread of the resulting parameters. That spread is the
disentangling contribution.

`N = 100` is the number to use. The relative standard error of a sample standard deviation
is `1/sqrt(2(N-1))`: 7% at 100, 12.7% at 32, and 18% at 16. Below about 32 the spread is too
noisy to quote. One hard check before trusting it — **the atmosphere grid step must be
smaller than the spread you are measuring**, or every draw lands in one grid cell and the
answer is zero for a reason that has nothing to do with your data.

## This is not the same as adding noise to the spectrum

The loop is not new. Kiran et al. (2016, §3.5) added *"artificial Gaussian noise with
sigma = sigma_c"* to a disentangled profile, refitted 500 times, and took the scatter. Cite
them. What is new is what gets drawn.

`draw_spectra` returns `d_hat + L⁻ᵀz` computed on the vector **stacked over all
components**, so a draw is correlated across wavelength *and* across the two stars, and
draw *i* of component A is the same posterior sample as draw *i* of component B. Adding
independent noise per pixel assumes the error is white. Disentangling error is not: it has
a genuine low-frequency null space (Pavlovski & Hensberge 2011), and low-frequency error is
exactly what moves a continuum, and through it a surface gravity.

[`examples/10_downstream.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/10_downstream.py)
measures the difference rather than asserting it. Equivalent width is the stand-in for the
atmosphere code, which is the right choice and not just the cheap one: D40 established that
EW is the quantity that reaches the atmosphere code, and that an 11.5% EW error is a
systematic in log *g*. On the packaged example, over 24 draws:

| | EW (Å) | joint draws | independent per-pixel noise | ratio |
|---|---|---|---|---|
| component 1 | 0.2568 | 0.01873 | 0.01041 | **1.80×** |
| component 2 | 0.0417 | 0.02974 | 0.00881 | **3.38×** |

**White noise understates the integrated uncertainty by a factor of two to three.** For a
pointwise question the band is fine; for anything that integrates the spectrum it is not,
and every atmospheric parameter integrates the spectrum.

The second number is sharper still. The correlation between the two components' equivalent
widths across draws is **−0.992**, against **−0.052** for the same statistic under
independent per-pixel noise. That is D47's *k* = 0 exchange mode — the delocalized see-saw
that sits at ~1× the prior for every observing design — arriving in a derived quantity. The
two stars trade line depth almost exactly, so their **difference** is far better determined
than either one alone, and fitting the components separately with independent error bars
misstates the answer in both directions at once. Keep the draw index: plot *T*<sub>eff,A</sub>
against *T*<sub>eff,B</sub> per draw and the structure is visible. Pool the draws per
component and it is gone.

## What the spread does not contain

A caveat list, not a victory lap.

- **The atmosphere code's own model error** — grid coarseness, LTE, line-list quality, its
  own continuum placement. Entirely outside albireo's posterior, and unaffected by the draws.
- **Anything albireo conditions on rather than marginalizes.** The light fractions are
  *assumed*, not inferred, and the marginal likelihood is flat in them under constant light
  (`scripts/m5_light_ratio_demo.py`). That is precisely the systematic Pavlovski & Hensberge
  identify as dominant, and the draw spread is silent about it. Quote the assumed light
  ratio next to the number.
- **Double counting.** iSpec's `errors['teff']` is a within-draw fit error computed from the
  `err` column you supplied. Adding it in quadrature to a spread that came from the same
  posterior counts part of the uncertainty twice. Pick one: either give iSpec the band and
  read its error bar, or run the draws and take the spread — the draws are the better answer
  because iSpec weights by `sqrt(1/err)` rather than `1/err²`, a deliberate hand-calibration
  in its own source, so its reported error does not scale linearly with the band you give it.

## Practical notes

iSpec has no batch runner; it is a plain Python library and the loop is yours, with
`multiprocessing.Pool` over `ispec.model_spectrum` and a distinct `tmp_dir` per worker.
GSSP is Intel-Fortran and OpenMPI and runs on Linux; Tkachenko (2015) reports 5–6 minutes
per fit on 8 CPUs against a pre-computed grid, so `N = 100` is an overnight job on a
desktop, not a cluster job — provided you generate the grid once and reuse it.

Neither writer converts between air and vacuum, deliberately. iSpec ships `air_to_vacuum` /
`vacuum_to_air` as explicit user steps and does no conversion on read; albireo does the same,
for the reason in D43 — the offset is a nearly constant 83 km/s, the same order as the orbits
being measured, so guessing it is worse than declining to.
