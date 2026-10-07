# Turn a disentangled component into an RV template

This page fits four labels, Teff, log g, [M/H] and *v* sin *i*, to the components of a
disentangling fit against a published synthetic grid, so that a template can be selected or
rendered. The calls start from an existing fit, and the working example runs offline against
a toy grid.

See the [science overview](../science.md) for background and references.

Disentangling returns two spectra but does not identify the synthetic template against which
the individual epochs should be cross-correlated. Choosing it is the next step in a pipeline.

The mode is [`albireo.match`](../api/match.md), and its scope is narrow: it synthesizes
nothing, has no line list, and fits no abundances. For those,
[`albireo.handoff`](../api/handoff.md) exports to GSSP, iSpec, Korg.jl and PySME, and
[Propagate into Teff and log g](downstream.md) covers the procedure.

## Required accuracy of the labels

The criterion is whether a better template would change the epoch velocities, not whether a
label equals the star's true temperature. The literature indicates that the gain saturates
quickly:

| Label | Enough for template selection |
|---|---|
| Teff | 2–3% |
| log g | 0.15 dex |
| [M/H] | 0.15 dex |
| *v* sin *i* | 10% |

Posbic et al. (2012) measured that a template 400–1000 K too warm biases solar-type velocities
by about 0.2 km/s, roughly FWHM/60, with no loss of precision. The effect of a wrong template
is a constant velocity zero point per component, which is the quantity albireo already treats
as unidentified ([§5.3](../math.md#53-systemic-velocity-zero-point)). This mode therefore
fixes zero points and flux ratios. It does not improve velocity precision.

A label from this mode is a template coordinate. A label for a published abundance table is a
different measurement with a different error budget.

## The minimum call

```python
import albireo as ab

fit = dis.fit()                       # your disentangling, as usual

labels = fit.match_labels({
    "A": ab.StarLabels(library=grid_a, teff=ab.Between(5000, 7000),
                       logg=ab.Fixed(4.12), vsini=ab.Between(0, 40)),
    "B": ab.StarLabels(library=grid_b, teff=ab.Between(4000, 6000),
                       logg=ab.Fixed(4.31), vsini=ab.Between(0, 40)),
})
print(labels.summary())
```

`Fit.match_labels` takes the model grid, the recovered spectra, their uncertainty band, the
assumed light fractions, the instrument width and the dataset's wavelength medium from the
fit, so all of them are consistent with the disentangling. The module-level
[`match_labels`](../api/match.md) takes arrays instead, for spectra from another source.

By default `Fit.match_labels` compares the template composite with the epoch spectra rather
than with the disentangled components (`compare="epochs"`,
[§9.2a](../math.md#92a-comparing-in-the-epoch-space)). That chi-square is exact for the fit's
noise model, it does not depend on the declared light fractions, and it takes the libraries'
own resolving power into account, so a grid at R = 20,000 is not broadened twice. On simulated
Gaia RVS binaries it returned temperatures twice as close to the injected values as the
comparison against the components. `compare="native"` remains available, and the module-level
`match_labels` defaults to it because it has only the component spectra.

`logg=ab.Fixed(...)` in that example is the most consequential choice on the page.

## Fix log g where possible

Teff and log g correlate at about 0.98 when both are free. This is the published behaviour of
the problem, not an albireo artifact (Tamajo et al. 2011), and `summary()` flags the pair.

For an eclipsing binary the light curve and the orbit give masses and radii, hence log g, to
0.01 dex, an order of magnitude better than any spectroscopic determination. Declaring it
makes the fit well posed.

For a non-eclipsing SB2 no such constraint exists. Run the fit three ways and report the
spread as the uncertainty:

```python
free   = fit.match_labels(stars_all_free)
fixed  = fit.match_labels(stars_with_logg_fixed)
rigid  = fit.match_labels(stars_with_logg_fixed, dilution=ab.FixedDilution())
```

## The fitted light ratio

Disentangling returns component spectra scaled by assumed light fractions, and the likelihood
depends only on the products (`ℓ_i d_i`). An error in the assumed ratio therefore rescales
every line depth, and a uniform rescaling of line depths is indistinguishable from a change in
temperature unless the fit has another parameter to absorb it.

The default `RadiusRatio` provides that parameter. Both components are fitted together through
one shared scalar, with wavelength-dependent light fractions derived from the grids' own
continua, constructed so that they sum to one at every wavelength. This is GSSP's binary-mode
parameterization, and the light ratio is a result of the fit:

```python
labels.flux_ratio           # {"A": 0.62, "B": 0.38} - measured, not assumed
labels.light_fractions()    # (n_star, n_pix), summing to 1 at every pixel
```

Published spectroscopic light ratios of this kind agree with light-curve ratios to a few
percent, and are competitive with them when the photometric solution is degenerate. Quote the
value, because downstream cross-correlation codes are more sensitive to a wrong flux ratio
than to a wrong temperature, as the saphires documentation states.

`FixedDilution()` holds the dilution at the assumed light fractions. Run it as a diagnostic.
The difference between the two fits measures how much the assumed light fractions biased the
temperatures.

## Read the report against its nulls

`summary()` prints the caveats first and quotes every number against a reference:

- **`chi2` against `chi2_continuum`**: a fit with no template, only the nuisance. A `chi2` not
  far below it means the spectrum contained no label information and the result is the prior.
- **`chi2` against `chi2_nearest_node`**: the best raw grid node. This mode replaces choosing
  the closest node by eye, and the difference measures what continuous interpolation, fitted
  broadening and fitted dilution contributed.
- **Posterior width against prior width**, per label. Widths at 80% or more of the prior are
  listed under "learned nothing here".
- **Formal error against the draws spread** (below).

## Quote the wider error bar

For the default epoch comparison the formal errors were tested against the injected values on
simulated Gaia RVS binaries. They are close to calibrated for Teff, log g and [M/H]
(68th-percentile pulls of 1.4, 1.0 and 1.4) and too small by a factor of about three for
*v* sin *i* and the light fraction (3.5 and 2.7). The remaining errors of those two come from
smoothing and orbit errors the formal covariance does not include
([§9.5](../math.md#95-uncertainties-the-formal-covariance-and-the-spread-over-posterior-draws)).
Quote those two with an error enlarged by about three. `labels.flux_ratio_errors` holds the
formal light-fraction errors, and `summary()` prints the calibration beside them.

For the comparisons against the disentangled components (`compare="native"` or `"matched"`)
the Laplace covariance understates the error more, because the residuals are correlated rather
than white: disentangling artifacts are structured across wavelength by construction. In
every code where this has been checked, the formal errors are five to ten times too small.
Gebruers et al. (2022) report 70 K formal against 425 K realistic for B stars at S/N 150. For
those fits, refit the draws of the disentangling posterior:

```python
native = fit.match_labels(stars, compare="native")
draws = posterior.spectra(num_draws=32)          # joint draws, correlated across components
native = ab.refit_draws(native, draws[:, :2])    # stellar rows only
native.errors("draws")     # the number to quote
native.errors("laplace")   # the number to quote it beside
```

The draws must be joint. Independent per-component draws would omit the exchange modes, the
low-*k* directions that transfer flux between the two stars, and propagating those modes is
the purpose of the refit. After the refit, `summary()` prints both errors and the ratio
between them. `refit_draws` rejects an epoch-comparison fit, whose spread would come from
draws of the epoch noise, which is not implemented.

## Working example

[`examples/11_labels.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/11_labels.py)
runs the procedure offline against a toy grid built in the file. It injects two components at
off-node labels, gives the fit light fractions that are wrong by a factor of 1.3, and checks
that the error is absorbed by the dilution rather than by the temperature. On the packaged
run both components are within a few K of the injected Teff, and the light fractions recover
0.62/0.38 from an assumed 0.72/0.28.

## Getting a real grid

For real work, `fetch_library` downloads and caches a published grid:

```python
ab.library_names()
# ['bosz2024-fgk-r20000', 'bosz2024-fgk-rvs', 'bosz2024-hot-r20000',
#  'bosz2024-hot-rvs', 'pollux-ob-smc24']

ab.library_info("bosz2024-fgk-r20000")          # coverage, licence, citation, sizes
library = ab.fetch_library("bosz2024-fgk-r20000")
```

The first call downloads about 621 MB from MAST and leaves a ~51 MB cache. Every later call
reads the cache. `$ALBIREO_DATA_DIR` moves the cache to a volume with enough space, or points
to a directory that is already populated. A subset of the band can be requested with
`fetch_library(name, wave_range=(5150.0, 5250.0))`, which slices the cached grid.

For the SMC OB regime `pollux-ob-smc24` is registered with its coverage and citation, but
POLLUX has no stable download URL (its collections are served through a web form), so the
archive must be fetched manually. `ingest_pollux` reports this, and no parser is shipped for a
format that has not been inspected.

Cite whichever library is used. `library_info(name)["citation"]` is the citation string, and
both shipped grids are CC BY 4.0, which requires attribution.

## Choosing a grid, and the wavelength medium

`SpectralLibrary.medium` is required and has no default. Air and vacuum wavelengths differ by
about 83 km/s across the optical, the same order as the semi-amplitudes being measured.

The distribution's README is not a reliable source for the medium. BOSZ 2017 was vacuum
throughout, and BOSZ 2024 is air above 200 nm, under the same name. A cached copy from the
wrong year produces an 80 km/s error that no downstream step detects. `line_core_medium`
measures the convention from the spectra:

```python
verdict = ab.line_core_medium(wave, flux)
verdict["medium"]   # "air" or "vacuum", or a refusal if the lines are too blended to tell
```

Before using an unfamiliar grid, measure its interpolation error:

```python
ab.crossval_library(library)   # rms flux error at doubled node spacing
```

For context, on the 250 K / 0.5 dex spacing BOSZ uses, linear flux interpolation has an error
of about 0.05% and a cubic about 0.03%, against roughly 0.1% for a Payne-style neural
emulator. On a well-sampled grid the differentiable cubic used here is more accurate, so
albireo ships no neural emulator. On a coarse, strongly non-linear grid the ordering may
differ, and `crossval_library` measures it.
