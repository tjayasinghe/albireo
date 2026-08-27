# API reference

These pages are generated from the docstrings in the source. They are the reference; the
[design](../design.md) and [mathematical foundations](../math.md) pages are the explanation,
and the [tutorials](../tutorials/sb2-end-to-end.md) are the worked examples.

Nearly everything below is re-exported at the top level, so `albireo.build_problem` and
`albireo.forward.build_problem` are the same function. Prefer the short form.

## How the modules fit together

The path from a directory of FITS files to a posterior runs through them in roughly this order:

| Stage | Module | What it does |
|---|---|---|
| Declare | [`albireo.facade`](facade.md) | `Disentangler` — the experimental front end that compiles a declaration into everything below |
| Fetch | [`albireo.archive`](archive.md) | ESO ObsCore/TAP queries and resumable downloads; BLOeM by name |
| Read | [`albireo.io`](io.md) | FITS → `RawSpectrum` → `EpochData`, with every assumption warned about |
| Repair | [`albireo.preprocess`](preprocess.md) | continuum, inverse variance, masks, one shared grid |
| Hold | [`albireo.data`](data.md) | `EpochData` / `Dataset` — the pure-NumPy user boundary, plus `LogGrid` |
| Build | [`albireo.forward`](forward.md) | `Dataset` + grid → `Problem`, the fixed-θ forward model |
| Evaluate | [`albireo.likelihood`](likelihood.md) | the marginal likelihood with the component spectra integrated out |
| Infer | [`albireo.inference`](inference.md) | the numpyro model, MAP, Laplace mass matrix, NUTS |
| Search | [`albireo.scan`](scan.md) | the SB1 faint-companion K₂ scan |
| Calibrate | [`albireo.calibrate`](calibrate.md) | injection–recovery detection limits and false-alarm probabilities |
| Plan again | [`albireo.forecast`](forecast.md) | what the *next* epochs would buy — computable before they are taken |
| Stock | [`albireo.library`](library.md) | published synthetic grids, standardized, with the wavelength medium declared and differentiable interpolation over them |
| Label | [`albireo.match`](match.md) | Teff / log g / [M/H] / *v* sin *i* against those grids — for choosing a template, not for an abundance table |

[`albireo.simulate`](simulate.md) sits outside that path and feeds it: it is the oracle every
closed-loop test is written against.

Three modules are lower-level than the rest and are documented because the numerics are the
point, not because they are the intended entry point:
[`albireo.operators`](operators.md) (shifts, convolutions, rebinning — each with an exact
adjoint), [`albireo.solver`](solver.md) (the block-tridiagonal Cholesky, selected inverse,
and sampling), and [`albireo.kepler`](kepler.md) (a differentiable Kepler solver).

## Conventions that apply everywhere

- **Wavelengths are ångström** and **velocities are km/s**, throughout.
- **Masking is `ivar == 0`.** There is no separate mask array in the model, and data are
  never deleted or resampled to work around a gap.
- **Deviation spectra, not fluxes.** The component spectra the model solves for are
  deviations `d` from a normalized continuum, so the modelled flux is `1 + d`; absorption is
  negative and emission is positive.
- **Guards return `-inf`, never an approximation.** Where a parameter leaves the regime a
  build-time-static structure was built for — solver bandwidth, LSF kernel radius,
  eccentricity — the log-density is non-finite rather than quietly wrong.
