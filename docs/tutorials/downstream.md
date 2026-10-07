# Propagate a disentangled spectrum into Teff and log *g*

This page propagates the uncertainty of a disentangled spectrum into the parameters returned
by an atmosphere code, with numbers from the packaged example. It has two parts: the part
albireo runs, and the part that needs an atmosphere code installed.

See the [science overview](../science.md) for background and references.

A disentangled spectrum is rarely the final result. It is passed to an atmosphere code (GSSP,
iSpec, Korg.jl, PySME), whose returned parameters are tabulated: effective temperature,
surface gravity, abundances. The uncertainty on the disentangled spectrum is usually dropped
at that step. Mahy et al. (2020, TMBM III, §3.1) state:

> We stress that the uncertainties that could arise from the normalisation procedure are not
> taken into account in the global uncertainties on the presented properties

Pavlovski, Southworth & Tamajo (2018) state:

> Propagation of uncertainties through this process is difficult so must be tackled
> numerically.

!!! tip "When four labels are enough"

    If the goal is a *template* (the synthetic spectrum against which individual epochs are
    cross-correlated) rather than an abundance table, a shorter procedure runs within albireo
    and keeps the uncertainty: [Turn a component into an RV template](labels.md). It fits
    Teff, log g, [M/H] and *v* sin *i* against a published grid, propagates the spectral
    posterior by refitting its draws, and measures the light ratio.

    It does not replace this page. It fits four labels only: no abundances, no
    microturbulence, no bespoke synthesis. A parameter intended for a published table still
    needs an atmosphere code, run as described below so that the uncertainty is kept.

## The file formats

GSSP and iSpec differ in most of the conventions that can fail without an error message.

| | GSSP | iSpec |
|---|---|---|
| Layout | 2 columns, whitespace | 3 columns, tab, one header line |
| Wavelength unit | ångström | nanometre |
| Grid | must be equidistant | as given |
| Per-pixel error | none: there is no column | `err`, absolute 1σ |
| Flux | normalized | normalized |

Two of those conventions fail silently.

**iSpec performs no unit conversion on the text path.** Its internal scale is nanometres, line
lists included. A wavelength written in ångström is a factor of ten outside every model grid,
and the fit still returns a result. `write_ispec` divides by ten, and both the unit and the
round trip are regression-tested.

**GSSP infers its synthetic step from the input file**: *"the step width in wavelength that
will be used for the calculation of synthetic spectra is computed from the observations"*
(Tkachenko 2015, Appendix B.2, which is the entire manual; there is no separate document and
no source repository). albireo solves on a log-wavelength grid, whose linear spacing varies
across the window by 1.32% on the packaged example. If the spectrum were written on that grid,
GSSP would take the first pixel pair as the step for the whole spectrum. `write_gssp`
resamples onto an equidistant grid, and applies the identical grid to every draw so the draws
stay comparable.

```python
import albireo as ab

ab.write_gssp("component.dat", grid, d_hat)              # -> component_1.dat, component_2.dat
ab.write_ispec("component.txt", grid, d_hat, std)        # -> component_1.txt, component_2.txt
```

## Fit N draws

GSSP accepts no per-pixel uncertainty. Its configuration files contain no error path, no
signal-to-noise entry and no weighting entry, and its own quoted error bars come from χ² on
the fit residuals. The posterior band can therefore be propagated to a temperature only
through repeated fits:

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
noisy to quote. Before using the spread, check that the atmosphere grid step is smaller than
the spread being measured. Otherwise every draw falls in one grid cell and the spread is zero
for a reason unrelated to the data.

## Joint draws compared with independent per-pixel noise

The loop is not new. Kiran et al. (2016, §3.5) added *"artificial Gaussian noise with
sigma = sigma_c"* to a disentangled profile, refitted 500 times, and took the scatter. That
work should be cited. The difference is in what is drawn.

`draw_spectra` returns `d_hat + L⁻ᵀz` computed on the vector stacked over all components, so a
draw is correlated across wavelength and across the two stars, and draw *i* of component A is
the same posterior sample as draw *i* of component B. Adding independent noise per pixel
assumes the error is white. Disentangling error is not white: it has a low-frequency null space
(Pavlovski & Hensberge 2010), and low-frequency error shifts a continuum and, through it, a
surface gravity.

[`examples/10_downstream.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/10_downstream.py)
measures the difference. Equivalent width is a proxy for the atmosphere code: the nebular
benchmark established that EW is the quantity that propagates into that code, and that an
11.5% EW error is a systematic in log *g*. The results on the packaged example, over 24
draws, are:

| | EW (Å) | joint draws | independent per-pixel noise | ratio |
|---|---|---|---|---|
| component 1 | 0.2568 | 0.01873 | 0.01041 | 1.80× |
| component 2 | 0.0417 | 0.02974 | 0.00881 | 3.38× |

Independent per-pixel noise understates the integrated uncertainty by a factor of two to
three. The pointwise band is the uncertainty of each pixel separately. Every atmospheric
parameter integrates the spectrum, and for an integrated quantity the band is not the right
input.

Across draws, the correlation between the two components' equivalent widths is −0.992,
against −0.052 under independent per-pixel noise. This is the *k* = 0 exchange mode of the
forecast (a delocalized exchange between the components, at about 1× the prior for every
observing design) appearing in a derived quantity. The two stars exchange line depth almost
exactly, so their difference is better determined than either one alone, and fitting the
components separately with independent error bars misstates both. Keep the draw index. A plot
of *T*<sub>eff,A</sub> against *T*<sub>eff,B</sub> per draw shows the correlation, and pooling
the draws per component discards it.

## What the spread does not contain

- **The atmosphere code's own model error**: grid coarseness, LTE, line-list quality, its own
  continuum placement. These are outside albireo's posterior and are unaffected by the draws.
- **Anything albireo conditions on rather than marginalizes.** The light fractions are
  assumed, not inferred, and the marginal likelihood is flat in them under constant light
  (`scripts/m5_light_ratio_demo.py`). Pavlovski & Hensberge identify this as the dominant
  systematic, and the draw spread does not include it. Quote the assumed light ratio beside
  the result.
- **Double counting.** iSpec's `errors['teff']` is a within-draw fit error computed from the
  supplied `err` column. Adding it in quadrature to a spread derived from the same posterior
  counts part of the uncertainty twice. Use one or the other: either give iSpec the band and
  read its error bar, or run the draws and take the spread. The draws are preferable because
  iSpec weights by `sqrt(1/err)` rather than `1/err²`, a manual calibration in its own source
  code, so its reported error does not scale linearly with the band it is given.

## Practical notes

iSpec has no batch runner. It is a plain Python library, so the caller writes the loop, with
`multiprocessing.Pool` over `ispec.model_spectrum` and a distinct `tmp_dir` per worker. GSSP is
Intel-Fortran and OpenMPI and runs on Linux. Tkachenko (2015) reports 5–6 minutes per fit on
8 CPUs against a pre-computed grid, so `N = 100` is an overnight job on a desktop rather than a
cluster job, provided the grid is generated once and reused.

Neither writer converts between air and vacuum. iSpec ships `air_to_vacuum` /
`vacuum_to_air` as explicit user steps and does no conversion on read. albireo does the same
because the offset is a nearly constant 83 km/s, the same order as the orbits being measured,
so a wrong guess would be worse than no conversion.
