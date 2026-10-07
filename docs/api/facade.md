# Disentangler (experimental)

This class is a declarative front end. The caller states which components are present, how
bright each one is, the line-spread function of each instrument, and what is already known
about the orbit, and albireo assembles the fit from that declaration.

!!! warning "Experimental"
    This class defines a vocabulary, which is difficult to change once other code depends
    on it, so it stays experimental until it has been used on a problem outside this
    project. [`MarginalOrbitModel`](inference.md) and the functions around it are the
    supported interface and are stable. `Disentangler.expert()` returns that interface, so
    moving to the low-level API takes three lines.

```python
import albireo as ab

dataset = ab.load_example("sb2_sim")
dis = ab.Disentangler(
    dataset,
    components=[ab.Star("primary", light=0.62), ab.Star("secondary", light=0.38)],
    orbit=ab.Orbit(period=ab.Between(5.5, 6.5), k=ab.Between([10.0, 10.0], [90.0, 90.0])),
    lsf={"DEMO": 6.5},
)
fit = dis.fit()
post = fit.sample(seed=0)
```

## Derived quantities

The `Disentangler` interface assembles the low-level fit and exposes it. Four quantities
that the low-level API requires the caller to supply correctly are derived instead, and
`dis.explain()` prints each one:

| Derived | Basis |
|---|---|
| The **velocity budget** | It must bound the largest relative velocity that the priors allow, not that of the fitted solution. With too small a budget the sampler stalls at the bandwidth guard, and a direct call to `log_likelihood` returns a wrong result without an error. The budget is derived from the support of the `k` priors. |
| The **model grid** | Wide enough for that budget plus the LSF kernel radius. With a smaller margin the shifted model extends past the end of the grid, and the fit loses flux there without a warning. |
| The **conjunction phase** | Located by a 42-point scan over one period before anything is optimized. The likelihood is sharply multimodal in phase, and L-BFGS started in the wrong mode converges tightly to the wrong solution. The count is even so that the grid contains the antipode of every trial, which an odd count leaves midway between two trials. For a near-equal pair a phase and its antipode have nearly equal likelihoods, and the scan can choose between them only if both are on the grid. |
| The **semi-amplitude basin** | When a semi-amplitude is declared as a `Between` range, a scan of the marginal likelihood precedes the fit. The first ranged component is scanned over a geometric grid across its range (a factor of 1.25 between neighbours), crossed with eight conjunction phases. Each further ranged component is scanned over its own grid at the values located so far. Two joint refinements halve the spacings around the best trial. A final pass scans each component once more over its whole grid at the refined best. For two ranged components the exchange-symmetric twin of the best trial is tried. L-BFGS starts from the best trial when its likelihood exceeds that of the declared start. Between the two scan passes each star's prior amplitude is profiled at the orbit located so far. The marginal likelihood is exactly invariant under scaling a star's light by a factor and its two smoothness hyperparameters by the factor squared, so a profile over the light at fixed hyperparameters is a profile over the star's prior amplitude. It is not a measurement of the star's light, for which the profile maximum is at 0.3 to 0.7 of the injected value. Where the profile maximum differs from the declared light by a factor of two or more, the hyperparameter starts are moved and the scan is repeated (`fit.k_scan.prior_scales`). With a companion declared at six times its light, the scan's maximum was at a static secondary, and the declared amplitude was 150 nats below the profile maximum. A companion with a few percent of the light has its likelihood maximum at the true semi-amplitude only when the primary's semi-amplitude is within about ten percent of the true value. No affordable product grid satisfies that condition, so the axes are scanned one at a time rather than as a product. The likelihood is also multimodal in the semi-amplitudes: a start at half the true value converges at half, and a start far above it converges in the static-component minimum. A phase located at semi-amplitudes wrong by a factor of three can be a quarter of a period from the true phase, so the two are scanned together. The scans run on a copy of the declaration with the model grid at twice the pixel size, which is a quarter of the cost at a dozen epochs. The fit runs on the full grid. Two checks limit the harm of scanning a coarse model of the wrong shape. A best trial with every ranged semi-amplitude at the floor of its grid is the static-component minimum and is rejected. The start moves only when the best trial exceeds it by more than 25 nats jointly. A component whose own move then gains less than 5 nats, with the others at the best trial, returns to its start. A companion with a few percent of the light changes the log-likelihood by about 25 nats over its whole range, so the evidence is assessed jointly, and a starting value from a template table is kept where the data do not distinguish the two. `fit.k_scan` retains the trials, the hold losses and the notes. `fit(k_scan=False)` skips the scan. |
| The **smoothness hyperparameters** | Fitted by empirical Bayes, then frozen for sampling, and reported per component with a flag on any that did not move from its start. |

A fifth quantity is structural rather than derived. A spec such as `Between(5.5, 6.5)`
specifies both its prior and its starting value, so `priors` and `init` cannot diverge. In
the low-level API they are two dictionaries written separately and compared by an
assertion.

## Declaring velocities instead of an orbit

Many binaries of interest have no published period: BLOeM's 59 double-lined systems have
no orbital solutions. Measured velocities can be declared in place of an orbit:

```python
dis = ab.Disentangler(
    dataset,
    components=[ab.Star("A", light=0.6), ab.Star("B", light=0.4)],
    velocities=ccf_velocities,        # (n_stellar, n_epochs) km/s, instead of orbit=
    lsf={"GIRAFFE": ab.LSF.from_resolution(6300)},
)
table = dis.fit()                     # a velocity-mode Fit; no orbital sites are sampled
rv, err = table.velocities(), table.velocity_errors()
```

Exactly one of `orbit=` and `velocities=` is required. For a system with no published
period the free per-epoch table (`docs/math.md` §7.6) produces the period, and it requires
a warm start: a cold start is 122,000 nats worse. The alternative warm start,
`Fit.free_velocities()`, requires a Keplerian fit and therefore a period, which an unsolved
system does not have.

The declaration has three properties:

- The velocities are a starting point, not a constraint. The per-component zero point
  remains unidentified, so the declaration must be correct in its epoch-to-epoch pattern
  but need not be correct in its level. A systemic offset of +150 km/s changes neither the
  result nor the solver's bandwidth, because the budget is derived from the centred table.
- A declaration whose components never separate is equivalent to a cold start and raises
  an error. Velocities that never resolve the pair beyond the LSF width produce a warning.
- `scan()` and `detection_limit()` require a known SB1 orbit and raise an error without
  one.

## One instrument name, two resolving powers

The line-spread width is a property of the exposure, not of the instrument name. HARPS
observes at R = 115,000 in its high-accuracy mode and at R = 80,000 in its high-efficiency
mode, and both write `INSTRUME = 'HARPS'`. FEROS, UVES and X-shooter settings differ between
programmes on one target. The reader records each file's resolving power on its epoch
(`EpochData.lsf_sigma_kms`), `Dataset.summary()` lists the distinct widths under each key,
and the declaration can use these widths:

```python
dis = ab.Disentangler(dataset, components=[...], orbit=...,
                      lsf={"HARPS": ab.LSF.per_epoch()})   # or the string "per-epoch"
```

Each epoch is then modelled at its own width, and `dis.explain()` prints the widths and how
many epochs have each. A width given as one number for a key whose epochs declare several
is applied to all of them. This produces a warning that names the epochs, because on AI
Phoenicis six EGGS exposures modelled at the HAM width were not noticed for a month. A
per-epoch width is the value that the file declared, so it cannot be inferred. An
`lsf_sigma` site on such an instrument raises an error in the low-level API.

## Required declarations

In each of the following a default would amount to a scientific claim, so the value is
required or the call is rejected.

- **Light fractions have no default.** `Star(light=...)` is required and the stellar light
  fractions must sum to 1. With constant light fractions the likelihood depends only on
  `l_i * d_i`, so every recovered depth scales as `1/l_i` and the fit is not sensitive to
  a wrong value. The assumed value is repeated in every summary under
  `Assumed, not measured`.
- **A phase scan is not a period search.** The scan resolves phase at one period, the
  prior's midpoint, so a prior wide enough to constitute a search produces a warning that
  names the remedy. It is a warning and not an error because the result degrades gradually.
- **Air versus vacuum must be declared when it matters.** A `Nebular` or `Telluric`
  component depends on absolute line positions, so an undeclared wavelength scale raises
  an error. The difference between the two scales is a nearly constant 83 km/s.
- **A budget override may not fall below what the priors reach**, and the error names the
  terms that exceed it.
- **The eccentricity singularity is handled by construction.** A free `ecc` never starts at
  the origin, and `ecc=ab.Fixed(0.0)` does not sample those sites.

## The noise model

The pixels are independent by default. A pipeline that resampled the spectra onto a
common step correlates neighbouring pixels (Gaia's RVS grids have lag-one correlations of
0.27 and 0.81, which the archive's errors do not express), and the declaration can state
this:

```python
dis = ab.Disentangler(dataset, components=[...], orbit=..., lsf={"RVS": ab.LSF.from_resolution(11_500)},
                      noise_correlation={"RVS": 0.27})       # or ab.Between(-0.9, 0.9) to fit one value
```

The noise model becomes AR(1) along the pixel index
([§1.4a](../math.md#14a-correlated-noise-the-ar1-chain)), held at the declared values per
epoch or fitted as one shared `ar1_phi` site. `dis.explain()` states which, and a declared
value is listed among the assumptions. The same value is passed to
`fit.measure_velocities()`, whose errors include it through the sandwich of
[§10.4](../math.md#104-uncertainties-and-detection).

## Stellar labels from a fit

`fit.match_labels(stars)` fits Teff, log g, [M/H] and *v* sin *i* to the stellar components,
taking the grid, the spectra, the declared light fractions, the LSF and the medium from the
fit. It compares the template composite with the epoch spectra by default
(`compare="epochs"`), through the statistics that `fit.epoch_statistics(resolving_power=...)`
returns: $`h = A^\top W z`$, $`z^\top W z`$ and the band of $`G = A^\top W A`$ at the MAP.
In these statistics the stellar operator's LSF is reduced by the libraries' resolving power
and by the model grid's smoothing, $`\tfrac{7}{12}\,\Delta v^2`$ in variance, and any
telluric or nebular row is held at its posterior mean
([§9.2a](../math.md#92a-comparing-in-the-epoch-space)). The stars' libraries must declare
one resolving power between them. The statistics are built once per call, in a few seconds,
and can be reused through `statistics=`. A compensated width that would fall below half a
model pixel is floored there with a warning that names the grid spacing that avoids it.

```python
labels = fit.match_labels(stars)                               # epoch comparison
native = fit.match_labels(stars, compare="native")             # against the components
stats = fit.epoch_statistics(resolving_power=20_000)           # reuse across calls
print(stats.summary())                                         # the operator's widths
again = fit.match_labels(stars, statistics=stats, restart_rounds=0)
```

The epoch comparison does not depend on the declared light fractions, so its
`labels.flux_ratio` is a measurement of the light. [`albireo.match`](match.md) describes the
optimiser, the restarts and the measured calibration of the formal errors.

## Outside the scope of v1

Per-epoch jitter, inferred light fractions and inferred LSF widths are not offered through
this interface. Each is a one-line site in the low-level API and a scientific claim, which
a keyword such as `jitter=True` would present as a convenience. `fit.z_rms` is always
printed as the diagnostic for whether they are needed, and they can be added through
`dis.expert()`.

Three further declarations are rejected rather than approximated:

- **Hierarchical triples.** The model has the `period_out`/`t_conj_out`/`k_out` sites.
  `Orbit(outer=...)` raises `NotImplementedError` and refers to `expert()`.
- **Gauss-Hermite `h3`.** It enters the kernel through `build_problem` rather than through
  the model class that this interface builds, so an `LSF(h3=...)` field would be discarded
  without an error. There is no such field.
- **A lower bound on eccentricity.** The sampled pair is (√e·cos ω, √e·sin ω), in which a
  lower bound on *e* is an annulus rather than a box. `ecc=Between(lo, hi)` with `lo > 0`
  raises an error. Without this check, a fit with a declared lower bound was measured to
  return half that bound.

`Disentangler` does not wrap plotting. [`albireo.plotting`](results.md) already covers it,
and wrapping it would duplicate the interface without adding a guarantee.

Background and references: [science overview](../science.md).

::: albireo.facade
