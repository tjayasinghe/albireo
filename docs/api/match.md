# Stellar labels for template selection

Fits Teff, log g, [M/H] and *v* sin *i* to disentangled component spectra against a
published synthetic grid, so that each component can be rendered as a template for
measuring epoch radial velocities in TODCOR, saphires, iSpec or a survey pipeline.

The module performs template selection, not stellar synthesis: it synthesizes no spectrum,
carries no line list, solves no radiative transfer and fits no individual abundances.
Questions that require those are addressed through [`albireo.handoff`](handoff.md) and
GSSP, iSpec, Korg.jl or PySME.

Its purpose is the preceding step: choosing the right template, pinning the per-component
velocity zero point, and checking an assumed flux ratio. The literature supports this
scope. A wrong template mostly injects a constant velocity offset per component rather
than degrading precision, and the accuracy required before a template stops limiting the
velocities is loose: Teff to 2–3%, log g and [M/H] to 0.15 dex, *v* sin *i* to 10%
([§9.6](../math.md#96-the-accuracy-this-has-to-reach)). Labels from this mode are template
coordinates. A label for an abundance table is a different measurement with a different
error budget.

## Model choices

**Dilution is fitted.** Disentangling returns components scaled by light fractions that
were assumed, and the likelihood saw only the products, so an error in the light fractions
rescales every line depth and is indistinguishable from a temperature error. Both
components are therefore fitted together through one shared radius ratio, with
wavelength-dependent light fractions that sum to one at every pixel by construction: the
parameterization of GSSP's binary mode (Tkachenko 2015). The resulting spectroscopic light
ratio is a product in its own right; downstream cross-correlation codes are far more
sensitive to a wrong flux ratio than to a wrong temperature.

**The unconstrained zero point is modelled and reported.** Each component's constant
offset lies in the null space of the disentangling problem
([§5.1](../math.md#51-the-low-frequency-degeneracy-the-undulations-theorem)) and is
constrained only by the smoothness ridge. Left unmodelled, it shifts the line depths and
returns a biased Teff. The additive Chebyshev nuisance absorbs it, and its zeroth term is
that zero point, which is fitted and then printed.

**Uncertainties are quoted twice.** The Laplace error is the curvature at the optimum,
which is not the relevant quantity on correlated residuals; every code that has checked
finds the formal errors optimistic by five to ten times (Czekala et al. 2015; Gebruers et
al. 2022). `refit_draws` therefore refits the labels once per joint posterior draw of the
component spectra, and `summary()` prints both numbers side by side with the ratio between
them. This applies to the two comparisons against the disentangled components; the epoch
comparison below has its own measured calibration.

## Three comparisons

`compare` selects what the template is compared with.

| `compare` | compared with | likelihood | optimiser | default of |
|---|---|---|---|---|
| `"epochs"` | the epoch spectra, through `EpochStatistics` | exact epoch chi-square $`L_{\mathrm{data}}`$ | bounded Levenberg-Marquardt with restarts | `Fit.match_labels` |
| `"native"` | the disentangled components $`\hat d`$ | diagonal, with a jitter site | L-BFGS (`run_map`) | `match_labels` |
| `"matched"` | $`\hat d`$ and the template, both convolved | diagonal, with a jitter site | L-BFGS (`run_map`) | |

**The epoch comparison is the one to use when a disentangling is available.** With the orbit,
the declared light fractions and the noise model held at the MAP, the chi-square of a template
composite against the epochs is
$`L_{\mathrm{data}}(m) = z^\top W z - 2\, m^\top h + m^\top G\, m`$ with
$`h = A^\top W z`$ and $`G = A^\top W A`$, an identity that is exact for the disentangling's
noise model, AR(1) included, and that equals the exact likelihood of $`\hat d`$ up to a
constant; the comparisons against $`\hat d`$ replace that likelihood's dense covariance with a
diagonal one ([§9.2a](../math.md#92a-comparing-in-the-epoch-space)). `Fit.epoch_statistics`
builds $`h`$, $`z^\top W z`$ and the band of $`G`$ in a few seconds, holding any telluric or
nebular row at its posterior mean, and `Fit.match_labels` does so by default:

```python
labels = fit.match_labels(stars)                       # compare="epochs"
stats = fit.epoch_statistics(resolving_power=library.resolving_power)
labels = ab.match_labels(grid, d_hat, stars=stars, medium="vacuum",
                         light_fractions=lights, lsf_sigma_kms=sigma,
                         compare="epochs", statistics=stats)
```

The low-level `match_labels` keeps `"native"` as its default, because on its own it has only
$`\hat d`$; passing `statistics` without `compare="epochs"` raises rather than ignoring them.
The statistics must have been built for the same stars, grid, medium and declared light
fractions, at the resolving power the stars' libraries declare, and from the instrument width
the call declares. `EpochStatistics.summary()` prints the operator they carry.

Five properties follow from the construction and were measured on simulated binaries
([Mathematical foundations §9.2a](../math.md#92a-comparing-in-the-epoch-space) and the
research notes it cites):

- **The declared light cancels.** The label rows carry $`w_i/\ell^0_i`$ and the operator
  $`\ell^0_i`$, so the fit measures the light fraction itself. In the benchmark's orbit tier it
  cut the median error of the declared light from 0.043 to 0.011.
- **The template is broadened only by what the library lacks.** A published grid is already at
  its resolving power (`SpectralLibrary.resolving_power`, 20,000 for BOSZ), so the stellar
  operator carries $`\sigma_q = \sqrt{\sigma_{\mathrm{inst}}^2 - \sigma_{\mathrm{lib}}^2}`$, and a library
  at or below the instrument's resolving power is refused. `"matched"` applies the same
  reduction to its template.
- **The model grid's own smoothing is given back.** A template reaches the epoch pixels
  through five discrete steps on the model grid (the library's box average, the rotation
  kernel's pixel integration, the frame shift and the epoch shifts by linear interpolation,
  and the model pixel in the rebin), which add $`\tfrac{7}{12}\,\Delta v^2`$ in variance and
  which the epochs do not carry. `Fit.epoch_statistics` therefore builds the operator of
  $`\sigma_{\mathrm{op}}^2 = \sigma_{\mathrm{inst}}^2 - \sigma_{\mathrm{lib}}^2 - \tfrac{7}{12}\,\Delta v^2`$
  by default and records the variance and both widths (`grid_variance_kms2`, `lsf_sigma_kms`,
  `operator_sigma_kms`). On the pipeline's closed-loop fixture (native pixel 4.63 km/s, LSF
  5.5 km/s) the mean *v* sin *i* error over five noise draws was -3.02 km/s for a star at
  11 km/s without it and -0.24 km/s with it, with Teff, log g and the light unchanged.
  Where $`\sigma_{\mathrm{op}}`$ would fall below half a model pixel, which happens once
  $`\Delta v > 1.10\,\sigma_q`$, it is floored there and a warning names the grid spacing
  that avoids the floor; a grid coarser than $`\sigma_q`$ is warned about as well, and both
  are kept in `notes`. `grid_compensation=False` builds the operator of $`\sigma_q`$ alone
  ([§9.2a](../math.md#92a-comparing-in-the-epoch-space)).
- **The optimiser converges.** A bounded Levenberg-Marquardt in the constrained parameters
  stops when the undamped step predicts less than 0.001 in chi-square, in a median of four
  iterations. Restart rounds scan every component's *v* sin *i*, since the rotation kernel is
  an exact delta below half a model pixel and the objective is flat there, and the faintest
  component's (Teff, log g) nodes, and add a joint restart when that component's light has
  collapsed; `LabelMatch.restarts` records each round and `epoch_fit.notes` any collapse
  that survived.
- **Its formal errors are near calibration for three labels.** The 68th percentile of
  $`|\mathrm{pull}|`$ was 1.40 for Teff, 0.97 for log g and 1.38 for [M/H], and 3.52 for
  *v* sin *i* and 2.72 for the light fraction, where the native comparison's ran from 2.95 to
  35.1. No inflation is applied in code; the rotation and the light fraction should be
  quoted with an error enlarged by about three
  ([§9.5](../math.md#95-uncertainties-and-why-the-formal-one-is-not-enough)).
  `refit_draws` refuses an epoch fit.

Converged the same way, the epoch comparison returned median errors of 60 K, 0.062 dex, 0.006
dex, 2.7 km/s and 0.012 in the light fraction, against 119 K, 0.180 dex, 0.024 dex, 6.9 km/s
and 0.035 for the native comparison. In epochs mode `chi2`, `chi2_continuum` and
`chi2_nearest_node` are epoch chi-squares over `n_pixels_used` epoch pixels, so the report's
nulls keep their meaning.

The warm-start scan's default *v* sin *i* trials are 1, 5 and 10 km/s and a fifth and a
half of the prior's upper bound, in every mode (the previous default started every fit at
37.5 km/s or more on a 0 to 150 km/s prior).

## The summary report

`LabelMatch.summary()` leads with the caveats, which are part of the result. Every number is
quoted against a null: the chi-square against a fit with no template and against the best
raw grid node, each label's posterior width against its prior width, and the formal error
against the spread of the draws. A Teff–log g correlation near 0.98 with both free is
expected and is labelled as such; the remedy is to fix log g from the eclipsing solution,
not to discount the number.

The theory is in [§9](../math.md#9-stellar-labels-from-disentangled-components).

Background and references: [science overview](../science.md).

::: albireo.match
