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
| The **conjunction phase** | Located by a 41-point scan before anything is optimized. The likelihood is sharply multimodal in phase, and L-BFGS started in the wrong trough converges tightly on the wrong answer. |
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

## Outside the scope of v1

Per-epoch jitter, AR(1) correlated noise, inferred light fractions and inferred LSF widths
are not offered through the façade. Each is a one-line site in the low-level API, and each
is a scientific claim rather than a convenience, which a keyword such as `jitter=True`
would present as the latter. `fit.z_rms` is printed unconditionally as the diagnostic for
whether they are needed, and `dis.expert()` is the route to adding them.

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
