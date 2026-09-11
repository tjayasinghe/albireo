# Disentangler (experimental)

A declarative front end. The caller states which components are present, how bright each
one is, what the spectrograph does, and what is already known about the orbit, and albireo
assembles the fit from that declaration.

!!! warning "Experimental"
    This class defines a vocabulary, and a vocabulary is expensive to change once other
    code depends on it, so it remains marked experimental until it has been used on a
    problem outside this project. [`MarginalOrbitModel`](inference.md) and the functions
    around it are the supported surface and are stable. `Disentangler.expert()` returns
    exactly that surface, so moving to the low-level path costs three lines rather than a
    rewrite.

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

The façade emits the expert path rather than hiding it. Four quantities that the low-level
path requires the caller to supply correctly are derived instead, and `dis.explain()`
prints each one:

| Derived | Basis |
|---|---|
| The **velocity budget** | It must bound the largest relative velocity the priors allow, not the one the answer turns out to have. Too small a budget stalls the sampler against a guard it cannot see; reached through `log_likelihood` directly, too small a budget gives a wrong result without an error. The information is already in the support of the `k` priors. |
| The **model grid** | Wide enough for that budget plus the LSF kernel radius. Short of that margin the shifted model runs off the grid and the fit silently loses flux there. |
| The **conjunction phase** | Located by a 42-point scan over one period before anything is optimized. The likelihood is sharply multimodal in phase, and L-BFGS started in the wrong trough converges tightly on the wrong answer. The count is even so that the grid holds the antipode of every trial it samples, which an odd count leaves midway between two trials; for a near-equal pair the two mirrors are nearly equally good and the scan can only choose between them if both are on the grid. |
| The **semi-amplitude basin** | When a semi-amplitude is declared as a `Between` range, a scan of the marginal likelihood precedes the fit: the first ranged component crosses a geometric grid over its range (a factor of 1.25 between neighbours) together with eight conjunction phases, each further ranged component crosses its own grid at what has been located so far, two joint refinements halve the spacings around the best trial, a final pass takes each component once more over its whole grid at the refined best, and for two ranged components the exchange-symmetric twin of the best is tried; L-BFGS starts from the best trial when it beats the declared start. Between the two passes each star's prior amplitude is profiled at the orbit located so far: the marginal likelihood is exactly invariant under scaling a star's light by a factor and its two smoothness hyperparameters by the factor squared, so a profile over the light at fixed hyperparameters is a profile over the amplitude the data want for that star (not a measurement of its light, which it puts at 0.3 to 0.7 of the truth), and where it asks for a factor of two or more the hyperparameter starts are moved and the scan repeated (`fit.k_scan.prior_scales`); a companion declared at six times its light had made the scan prefer a static secondary, and the profile rejected that amplitude by 150 nats. The axes are taken in turn rather than as a product because a companion of a few percent of the light prefers its true semi-amplitude only while the primary's is within about ten percent of the truth, which no product grid coarse enough to afford holds. The likelihood is multimodal in the semi-amplitudes too: a start at half the true value settles at half, a start far above it in the static-component minimum, and a phase located at semi-amplitudes a factor of three off can sit a quarter of a period from the truth, which is why the two are scanned together. The scans run on a copy of the declaration with the model grid at twice the pixel, a quarter of the cost at a dozen epochs, and the fit itself on the full grid. Two guards keep a scan on a coarse model of the wrong shape from doing harm: a best trial with every ranged semi-amplitude at the floor of its grid is the static-component minimum and is refused, and the start moves only when the best trial beats it by more than 25 nats jointly, after which a component whose own move is worth less than 5 nats, with the others at the best trial, returns to its start (a companion of a few percent of the light commands some 25 nats over its whole range, so the evidence is judged jointly and a template table's seed is kept where the data are indifferent). `fit.k_scan` retains the trials, the hold losses and the notes; `fit(k_scan=False)` skips it. |
| The **smoothness hyperparameters** | Fitted by empirical Bayes, then frozen for sampling, and reported per component with a flag on any that did not move from its start. |

A fifth quantity is structural rather than derived: a spec such as `Between(5.5, 6.5)`
carries both its prior and its starting value, so `priors` and `init` cannot diverge. In
the low-level path they are two dictionaries written separately with an assertion between
them.

## Declaring velocities instead of an orbit

Not every binary has a published period, and for many systems of interest none exists:
BLOeM's 59 double-lined systems have no orbital solutions. Measured velocities can be
declared in place of an orbit:

```python
dis = ab.Disentangler(
    dataset,
    components=[ab.Star("A", light=0.6), ab.Star("B", light=0.4)],
    velocities=ccf_velocities,        # (n_stellar, n_epochs) km/s — instead of orbit=
    lsf={"GIRAFFE": ab.LSF.from_resolution(6300)},
)
table = dis.fit()                     # a velocity-mode Fit; no orbital sites are sampled
rv, err = table.velocities(), table.velocity_errors()
```

Exactly one of `orbit=` and `velocities=` is required. For a system with no published
period the free per-epoch table (`docs/math.md` §7.6) is what produces the period, and it
requires a warm start: a cold start is 122,000 nats worse. The alternative warm start,
`Fit.free_velocities()`, is reached through a Keplerian fit and therefore through a period,
which is what an unsolved system lacks.

Three properties of the declaration:

- The velocities are a starting point, not a constraint. The per-component zero point
  remains unidentified, so the declaration must be right about the epoch-to-epoch pattern
  rather than about the level: a systemic offset of +150 km/s changes neither the result nor
  the solver's bandwidth, because the budget is derived from the centred table.
- A declaration whose components never separate is equivalent to a cold start and raises.
  Velocities that never resolve the pair beyond the LSF width produce a warning.
- `scan()` and `detection_limit()` require a known SB1 orbit and refuse without one.

## One instrument name, two resolving powers

The line-spread width is a property of the exposure, not of the instrument name. HARPS
observes at R = 115,000 in its high-accuracy mode and at R = 80,000 in its high-efficiency
mode, and both write `INSTRUME = 'HARPS'`; FEROS, UVES and X-shooter settings differ between
programmes on one target. The reader records each file's resolving power on its epoch
(`EpochData.lsf_sigma_kms`), `Dataset.summary()` lists the distinct widths under each key,
and the declaration can defer to them:

```python
dis = ab.Disentangler(dataset, components=[...], orbit=...,
                      lsf={"HARPS": ab.LSF.per_epoch()})   # or the string "per-epoch"
```

Each epoch is then modelled at its own width, and `dis.explain()` prints the widths and how
many epochs carry each. A width given as one number for a key whose epochs declare several
is applied to all of them, and produces a warning naming the epochs, because on AI Phoenicis
six EGGS exposures modelled at the HAM width went unnoticed for a month. A per-epoch width
is what the file declared, so it cannot be inferred: an `lsf_sigma` site on such an
instrument is refused at the low level.

## Required declarations

In each of the following a default would amount to a scientific claim, so the value is
required or the call is refused.

- **Light fractions have no default.** `Star(light=...)` is required and the stellar light
  fractions must sum to 1. With constant light fractions the likelihood depends only on
  `l_i * d_i`, so every recovered depth scales as `1/l_i` and nothing in the fit is
  sensitive to a wrong value. The assumed value is repeated in every summary under
  `Assumed, not measured`.
- **A phase scan is not a period search.** The scan resolves phase at one period, the
  prior's midpoint, so a prior wide enough to constitute a search warns and names the
  remedy. The result degrades rather than failing, which is why this is a warning rather
  than a refusal.
- **Air versus vacuum must be declared when it matters.** A `Nebular` or `Telluric`
  component is keyed to absolute line positions, so an undeclared wavelength scale raises
  rather than being assumed. The difference between the two scales is a nearly constant
  83 km/s.
- **A budget override may not fall below what the priors reach**, and the error names the
  terms that overflowed it.
- **The eccentricity singularity is handled by construction.** A free `ecc` never starts at
  the origin, and `ecc=ab.Fixed(0.0)` does not sample those sites at all.

## The noise model

The pixels are independent by default. A pipeline that resampled the spectra onto a
common step correlates neighbouring pixels (Gaia's RVS grids carry lag-one correlations of
0.27 and 0.81, which the archive's errors do not express), and the declaration can say so:

```python
dis = ab.Disentangler(dataset, components=[...], orbit=..., lsf={"RVS": ab.LSF.from_resolution(11_500)},
                      noise_correlation={"RVS": 0.27})       # or ab.Between(-0.9, 0.9) to fit one value
```

The noise model becomes AR(1) along the pixel index
([§1.4a](../math.md#14a-correlated-noise-the-ar1-chain)), held at the declared values per
epoch or fitted as one shared `ar1_phi` site; `dis.explain()` states which, and a declared
value is listed among the assumptions. The same value reaches `fit.measure_velocities()`,
whose errors carry it through the sandwich of
[§10.4](../math.md#104-uncertainties-and-detection).

## Outside the scope of v1

Per-epoch jitter, inferred light fractions and inferred LSF widths are not offered through
the façade. Each is a one-line site in the low-level API, and each is a scientific claim
rather than a convenience, which a keyword such as `jitter=True` would present as the
latter. `fit.z_rms` is printed unconditionally as the diagnostic for whether they are
needed, and `dis.expert()` is the route to adding them.

Three further declarations are refused rather than approximated:

- **Hierarchical triples.** The model carries the `period_out`/`t_conj_out`/`k_out` sites;
  `Orbit(outer=...)` raises `NotImplementedError` and points at `expert()`.
- **Gauss-Hermite `h3`.** It reaches the kernel through `build_problem` rather than through
  the model class the façade builds, so an `LSF(h3=...)` field would have been discarded
  without an error. There is no such field.
- **A lower bound on eccentricity.** The sampled pair is (√e·cos ω, √e·sin ω), in which a
  lower bound on *e* is an annulus rather than a box. `ecc=Between(lo, hi)` with `lo > 0`
  raises; before the refusal was added, a declared lower bound was measured returning half
  its value.

`Disentangler` does not wrap plotting: [`albireo.plotting`](results.md) already covers it,
and wrapping it would double the surface without adding a guarantee.

Background and references: [science overview](../science.md).

::: albireo.facade
