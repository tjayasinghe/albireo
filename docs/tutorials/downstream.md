# Propagate a disentangled spectrum into Teff and log *g*

See the [science overview](../science.md) for background and references.

A disentangled spectrum is rarely the result. It is fed to an atmosphere code (GSSP, iSpec,
Korg.jl, PySME), and the parameters that code returns are what is tabulated: effective
temperature, surface gravity, abundances. The uncertainty on the disentangled spectrum is
usually dropped at that boundary, and the papers say so:

> We stress that the uncertainties that could arise from the normalisation procedure are not
> taken into account in the global uncertainties on the presented properties
>
> — Mahy et al. (2020), TMBM III, §3.1

> Propagation of uncertainties through this process is difficult so must be tackled
> numerically.
>
> — Pavlovski, Southworth & Tamajo (2018)

This page describes how to carry the uncertainty across that boundary. It has two halves: the
half albireo runs, and the half that needs an atmosphere code installed.

!!! tip "When four labels are enough"

    If what is needed is a *template*, the synthetic spectrum against which individual epochs
    are cross-correlated, rather than an abundance table, there is a shorter route that stays
    inside albireo and does not drop the uncertainty at a file boundary:
    [Turn a component into an RV template](labels.md). It fits Teff, log g, [M/H] and
    *v* sin *i* against a published grid, propagates the spectral posterior by refitting its
    draws, and measures the light ratio along the way.

    It does not replace this page. It fits four labels and nothing else: no abundances, no
    microturbulence, no bespoke synthesis. A parameter that belongs in a published table still
    needs an atmosphere code, and everything below is how to reach one without losing the
    error bar.

## The file formats

GSSP and iSpec differ in most of the conventions that can fail without an error message.

| | GSSP | iSpec |
|---|---|---|
| Layout | 2 columns, whitespace | 3 columns, tab, one header line |
| Wavelength unit | ångström | nanometre |
| Grid | must be equidistant | as given |
| Per-pixel error | none: there is no column | `err`, absolute 1σ |
| Flux | normalized | normalized |

Two of those rows fail silently rather than loudly.

**iSpec performs no unit conversion on the text path.** Its internal scale is nanometres, line
lists included. A wavelength written in ångström lands a factor of ten outside every model
grid and still fits something. `write_ispec` divides by ten, and both the unit and the round
trip are regression-tested.

**GSSP infers its synthetic step from the input file**: *"the step width in wavelength that
will be used for the calculation of synthetic spectra is computed from the observations"*
(Tkachenko 2015, Appendix B.2, which is the entire manual; there is no separate document and
no source repository). albireo solves on a log-wavelength grid, whose linear spacing drifts
across the window by 1.32% on the packaged example. Written out as it stands, GSSP would take
the first pixel pair as the step for the whole spectrum. `write_gssp` resamples onto an
equidistant grid, and applies the identical grid to every draw so that the draws stay
comparable.

```python
import albireo as ab

ab.write_gssp("component.dat", grid, d_hat)              # -> component_1.dat, component_2.dat
ab.write_ispec("component.txt", grid, d_hat, std)        # -> component_1.txt, component_2.txt
```

## Fit N draws, not one spectrum

GSSP accepts no per-pixel uncertainty. Its configuration files contain no error path, no
signal-to-noise entry and no weighting entry, and its own quoted error bars come from χ² on
the fit residuals. The posterior band can therefore reach a temperature only through repeated
fits:

```python
draws = ab.draw_spectra(marginal, jax.random.key(0), 100)   # (100, n_comp, n_pix)
paths = ab.export_draws("draws/", grid, draws, format="gssp")
# draws/draw_0000_1.dat, draws/draw_0000_2.dat, draws/draw_0001_1.dat, ...
```

Then fit every file with the same grid, the same line list, the same masks and the same
starting guess, and take the spread of the resulting parameters. That spread is the
disentangling contribution.

`N = 100` is the recommended count. The relative standard error of a sample standard deviation
is `1/sqrt(2(N-1))`: 7% at 100, 12.7% at 32, and 18% at 16. Below about 32 the spread is too
noisy to quote. One check before trusting it: the atmosphere grid step must be smaller than the
spread being measured, or every draw lands in one grid cell and the answer is zero for a reason
unrelated to the data.

## Joint draws are not independent per-pixel noise

The loop itself is not new. Kiran et al. (2016, §3.5) added *"artificial Gaussian noise with
sigma = sigma_c"* to a disentangled profile, refitted 500 times, and took the scatter; that
work should be cited. The difference lies in what is drawn.

`draw_spectra` returns `d_hat + L⁻ᵀz` computed on the vector stacked over all components, so a
draw is correlated across wavelength and across the two stars, and draw *i* of component A is
the same posterior sample as draw *i* of component B. Adding independent noise per pixel
assumes the error is white. Disentangling error is not white: it has a low-frequency null space
(Pavlovski & Hensberge 2010), and low-frequency error is what moves a continuum and, through
it, a surface gravity.

[`examples/10_downstream.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/10_downstream.py)
measures the difference. Equivalent width stands in for the atmosphere code: D40 established
that EW is the quantity that reaches the atmosphere code, and that an 11.5% EW error is a
systematic in log *g*. On the packaged example, over 24 draws:

| | EW (Å) | joint draws | independent per-pixel noise | ratio |
|---|---|---|---|---|
| component 1 | 0.2568 | 0.01873 | 0.01041 | 1.80× |
| component 2 | 0.0417 | 0.02974 | 0.00881 | 3.38× |

Independent per-pixel noise understates the integrated uncertainty by a factor of two to
three. The pointwise band answers a pointwise question; every atmospheric parameter integrates
the spectrum, and for an integrated quantity the band is not the right input.

The second result concerns the two components jointly. The correlation between their
equivalent widths across draws is −0.992, against −0.052 for the same statistic under
independent per-pixel noise. That is D47's *k* = 0 exchange mode, the delocalized see-saw that
sits at about 1× the prior for every observing design, appearing in a derived quantity. The two
stars trade line depth almost exactly, so their difference is better determined than either one
alone, and fitting the components separately with independent error bars misstates both. Keep
the draw index: a plot of *T*<sub>eff,A</sub> against *T*<sub>eff,B</sub> per draw shows the
structure, and pooling the draws per component discards it.

## What the spread does not contain

- **The atmosphere code's own model error**: grid coarseness, LTE, line-list quality, its own
  continuum placement. These lie outside albireo's posterior and are unaffected by the draws.
- **Anything albireo conditions on rather than marginalizes.** The light fractions are
  assumed, not inferred, and the marginal likelihood is flat in them under constant light
  (`scripts/m5_light_ratio_demo.py`). Pavlovski & Hensberge identify this as the dominant
  systematic, and the draw spread says nothing about it. Quote the assumed light ratio beside
  the result.
- **Double counting.** iSpec's `errors['teff']` is a within-draw fit error computed from the
  supplied `err` column. Adding it in quadrature to a spread that came from the same posterior
  counts part of the uncertainty twice. Use one or the other: either give iSpec the band and
  read its error bar, or run the draws and take the spread. The draws are preferable because
  iSpec weights by `sqrt(1/err)` rather than `1/err²`, a hand-calibration in its own source, so
  its reported error does not scale linearly with the band it is given.

## Practical notes

iSpec has no batch runner; it is a plain Python library, so the loop is the caller's, with
`multiprocessing.Pool` over `ispec.model_spectrum` and a distinct `tmp_dir` per worker. GSSP is
Intel-Fortran and OpenMPI and runs on Linux; Tkachenko (2015) reports 5–6 minutes per fit on
8 CPUs against a pre-computed grid, so `N = 100` is an overnight job on a desktop rather than a
cluster job, provided the grid is generated once and reused.

Neither writer converts between air and vacuum, by design. iSpec ships `air_to_vacuum` /
`vacuum_to_air` as explicit user steps and does no conversion on read; albireo does the same,
for the reason in D43: the offset is a nearly constant 83 km/s, the same order as the orbits
being measured, so guessing it is worse than declining to.
