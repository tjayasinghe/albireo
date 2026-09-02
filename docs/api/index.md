# API reference

These pages are generated from the docstrings in the source and are the reference for the
package. The [scientific background](../science.md) and the
[mathematical foundations](../math.md) explain the choices behind the code, and the
[tutorials](../tutorials/sb2-end-to-end.md) are worked examples.

Nearly every public name is re-exported at the top level, so `albireo.build_problem` and
`albireo.forward.build_problem` are the same function. The short form is preferred.

## Module overview

The path from a directory of FITS files to a posterior passes through the modules in
approximately this order:

| Stage | Module | Contents |
|---|---|---|
| Declare | [`albireo.facade`](facade.md) | `Disentangler`, the experimental front end that compiles a declaration into the modules below |
| Fetch | [`albireo.archive`](archive.md) | ESO ObsCore/TAP queries and resumable downloads; BLOeM targets resolved by name |
| Read | [`albireo.io`](io.md) | FITS to `RawSpectrum` to `EpochData`, with a warning for every assumption made |
| Repair | [`albireo.preprocess`](preprocess.md) | continuum, inverse variance, masks, one shared grid |
| Hold | [`albireo.data`](data.md) | `EpochData` / `Dataset`, the pure-NumPy user boundary, and `LogGrid` |
| Build | [`albireo.forward`](forward.md) | `Dataset` + grid to `Problem`, the fixed-θ forward model |
| Evaluate | [`albireo.likelihood`](likelihood.md) | the marginal likelihood with the component spectra integrated out |
| Infer | [`albireo.inference`](inference.md) | the numpyro model, MAP, Laplace mass matrix, NUTS |
| Search | [`albireo.scan`](scan.md) | the SB1 faint-companion K₂ scan |
| Calibrate | [`albireo.calibrate`](calibrate.md) | injection–recovery detection limits and false-alarm probabilities |
| Forecast | [`albireo.forecast`](forecast.md) | the information that planned epochs would add, computable before they are taken |
| Stock | [`albireo.library`](library.md) | published synthetic grids in a standard form, with the wavelength medium declared and differentiable interpolation over them |
| Label | [`albireo.match`](match.md) | Teff / log g / [M/H] / *v* sin *i* against those grids, for template selection rather than abundance analysis |
| Measure | [`albireo.todcor`](todcor.md) | one velocity per component per epoch by N-dimensional correlation against templates (disentangled components, a label match, or a library), with calibrated errors and a batch driver |
| Solve | [`albireo.rvorbit`](rvorbit.md) | a Keplerian fitted to that table with the same solver and conventions as the joint model, and a period search |
| Run | [`albireo.pipeline`](pipeline.md) | every stage above for a list of stars from one declaration (`albireo run config.toml`), in worker processes, with structured products, figures and a record of failures |

[`albireo.simulate`](simulate.md) sits outside that path and feeds it: every closed-loop test
is written against the truths it generates.

Three modules are lower-level than the rest and are documented for their numerics rather
than as entry points: [`albireo.operators`](operators.md) (shifts, convolutions and
rebinning, each with an exact adjoint), [`albireo.solver`](solver.md) (the block-tridiagonal
Cholesky, selected inverse and sampling), and [`albireo.kepler`](kepler.md) (a
differentiable Kepler solver).

## Conventions

- **Units.** Wavelengths are ångström and velocities are km/s throughout.
- **Masking.** A pixel is masked by setting `ivar == 0`. The model has no separate mask
  array, and data are never deleted or resampled to work around a gap.
- **Deviation spectra.** The component spectra the model solves for are deviations `d` from
  a normalized continuum, so the modelled flux is `1 + d`; absorption is negative and
  emission is positive.
- **Guards.** Where a parameter leaves the regime a build-time-static structure was built
  for (solver bandwidth, LSF kernel radius, eccentricity), the log-density returns `-inf`
  rather than an approximation.

Background and references: [science overview](../science.md).
