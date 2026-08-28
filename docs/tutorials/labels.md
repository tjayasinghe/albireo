# Turn a disentangled component into an RV template

Disentangling gives you two spectra. It does not tell you which synthetic template to
cross-correlate the individual epochs against, and that choice is where the next tool in your
pipeline begins. This page closes that gap.

The mode is [`albireo.match`](../api/match.md), and it is scoped narrowly on purpose: it fits
four labels — Teff, log g, [M/H] and *v* sin *i* — against a published synthetic grid, so you
can pick or render a template. It synthesizes nothing, carries no line list, and fits no
abundances. For those, [`albireo.handoff`](../api/handoff.md) still exports to GSSP, iSpec,
Korg.jl and PySME, and [Propagate into Teff and log g](downstream.md) is still the page for it.

## What you actually need the labels to be good for

Less than instinct suggests. The question is not "is this the star's temperature" but "would a
better template change my epoch velocities", and the literature is fairly clear that the answer
goes flat quickly:

| Label | Enough for template selection |
|---|---|
| Teff | 2–3% |
| log g | 0.15 dex |
| [M/H] | 0.15 dex |
| *v* sin *i* | 10% |

Posbic et al. (2012) measured a template 400–1000 K too warm to bias solar-type RVs by about
0.2 km/s — roughly FWHM/60 — with no loss of precision. What a wrong template *does* cost is a
constant velocity zero point per component, which is exactly the quantity albireo already
tracks as unidentified ([§5.3](../math.md#53-systemic-velocity-zero-point)). So the useful
claim for this mode is that it fixes zero points and flux ratios, not that it improves RV
precision.

Treat a label from here as a template coordinate. A label for a paper's abundance table is a
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
assumed light fractions, the instrument width and the dataset's wavelength medium from the fit
itself, so none of them can disagree with what was actually solved. The module-level
[`match_labels`](../api/match.md) takes arrays instead, if your spectra came from somewhere
else.

Note `logg=ab.Fixed(...)` in that example. It is the single most consequential choice on the
page — see below.

## Fix log g if you possibly can

Teff and log g correlate at about 0.98 when both are free. That is not an albireo artifact; it
is the published behaviour of this problem (Tamajo et al. 2011), and `summary()` will flag the
pair and say so rather than quietly reporting two confident numbers.

For an **eclipsing** binary you do not have to accept it: the light curve plus the orbit give
masses and radii, hence log g, to 0.01 dex — an order of magnitude better than any spectroscopic
determination. Declare it and the fit becomes well posed.

For a **non-eclipsing** SB2 there is no such anchor, and the honest move is to stop pretending
there is. Run the fit three ways and report the spread as your uncertainty:

```python
free   = fit.match_labels(stars_all_free)
fixed  = fit.match_labels(stars_with_logg_fixed)
rigid  = fit.match_labels(stars_with_logg_fixed, dilution=ab.FixedDilution())
```

## The light ratio is a result, not just an input

This is the part worth understanding before trusting any number the mode produces.

Disentangling returns component spectra scaled by light fractions you *assumed*, and the
likelihood only ever saw the products (`ℓ_i d_i`). So an error in the assumed ratio rescales
every line depth — and a uniform rescaling of line depths is indistinguishable from a change in
temperature, unless the fit is given somewhere else to put it.

The default `RadiusRatio` gives it that somewhere else. Both components are fitted **together**
through one shared scalar, with wavelength-dependent light fractions derived from the grids' own
continua, constructed so they sum to one at every wavelength. This is GSSP's binary-mode
parameterization, and it means the light ratio comes *out* of the fit:

```python
labels.flux_ratio           # {"A": 0.62, "B": 0.38} - measured, not assumed
labels.light_fractions()    # (n_star, n_pix), summing to 1 at every pixel
```

Published spectroscopic light ratios of this kind agree with light-curve ratios to a few
percent, and are competitive with them when the photometric solution is degenerate. It is worth
quoting: downstream cross-correlation codes are considerably more sensitive to a wrong flux
ratio than to a wrong temperature — saphires says so in its own documentation.

`FixedDilution()` freezes the dilution at what you assumed. Run it as a **diagnostic**: the
difference between the two fits is a direct measure of how hard your assumed light fractions
were bending the temperatures.

## Read the report against its nulls

`summary()` leads with caveats because the caveats are the finding. Every number is quoted
against something:

- **`chi2` against `chi2_continuum`** — a fit with no template at all, only the nuisance. If
  yours is not far below it, the spectrum carried no label information and what came back is
  the prior wearing a number.
- **`chi2` against `chi2_nearest_node`** — the best raw grid node. This mode replaces "snap to
  the closest node by eye"; the gap is what continuous interpolation, fitted broadening and
  fitted dilution actually bought.
- **Posterior width against prior width**, per label. Anything at 80% or more of its prior is
  listed under "learned nothing here".
- **Formal error against the draws spread** (below).

## Quote the wider error bar

The Laplace covariance answers "how sharp is the optimum". On disentangled components that is
the wrong question, because the residuals are correlated rather than white — disentangling
artifacts are structured across wavelength by construction. Every code that has checked finds
formal errors optimistic by five to ten times; Gebruers et al. (2022) report 70 K formal against
425 K realistic for B stars at S/N 150.

So refit the disentangling posterior's own draws:

```python
draws = posterior.spectra(num_draws=32)     # joint draws, correlated across components
labels = ab.refit_draws(labels, draws[:, :2])   # stellar rows only
labels.errors("draws")     # the number to quote
labels.errors("laplace")   # the number to quote it beside
```

They must be **joint** draws. Independent per-component draws would miss exactly the exchange
modes — the low-*k* directions that trade flux between the two stars — that this is here to
propagate. `summary()` prints both errors and the ratio between them once you have done this.

## Working example

[`examples/11_labels.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/11_labels.py)
runs the whole loop offline against a toy grid built in the file: it injects two components at
off-node labels, hands the fit light fractions that are *wrong by a factor of 1.3*, and checks
that the error comes back as dilution rather than as temperature. On the packaged run both
components land within a few K of the injected Teff, and the light fractions recover 0.62/0.38
from an assumed 0.72/0.28.

## Getting a real grid

The toy grid above keeps the example offline. For real work, `fetch_library` downloads and
caches a published one:

```python
ab.library_names()
# ['bosz2024-fgk-r20000', 'bosz2024-fgk-rvs', 'pollux-ob-smc24']

ab.library_info("bosz2024-fgk-r20000")          # coverage, licence, citation, sizes
library = ab.fetch_library("bosz2024-fgk-r20000")
```

The first call downloads about 645 MB from MAST and leaves a ~95 MB cache; every call after
that reads the cache. `$ALBIREO_DATA_DIR` moves the cache somewhere with room, or points at a
directory somebody has already populated. If you only need part of the band, ask for it —
`fetch_library(name, wave_range=(5150.0, 5250.0))` — which is free, because it slices what was
already cached.

For the SMC OB regime `pollux-ob-smc24` is registered with its coverage and citation, but
POLLUX has no stable download URL — its collections come through a web form — so the archive
has to be fetched by hand and `ingest_pollux` tells you so instead of shipping a parser for a
format nobody has inspected.

Whatever you use, cite it. `library_info(name)["citation"]` is the string, and both shipped
grids are CC BY 4.0, which obliges attribution.

## Choosing a grid, and the trap in it

`SpectralLibrary.medium` is required and has no default. Air and vacuum wavelengths differ by
about 83 km/s across the optical — the same order as the semi-amplitudes you are trying to
measure — so this is not bookkeeping.

And do not trust the distribution's own README for it. BOSZ 2017 was vacuum throughout; BOSZ
2024 is air above 200 nm, **under the same name**. A cached copy from the wrong year is an
80 km/s error that nothing downstream will catch. `line_core_medium` measures the convention
from the spectra themselves, and is the right way to settle it:

```python
verdict = ab.line_core_medium(wave, flux)
verdict["medium"]   # "air" or "vacuum", or a refusal if the lines are too blended to tell
```

Before trusting a grid you have not used before, measure what interpolating it costs:

```python
ab.crossval_library(library)   # rms flux error at doubled node spacing
```

For context, on the 250 K / 0.5 dex spacing BOSZ actually uses, linear flux interpolation scores
about 0.05% and a cubic about 0.03%, against roughly 0.1% for a Payne-style neural emulator. On
a well-sampled grid the differentiable cubic here is the more accurate option, not a compromise
— which is why albireo does not ship a neural emulator for it. On a coarse, strongly non-linear
grid the answer may differ, and `crossval_library` is how you find out rather than assume.
