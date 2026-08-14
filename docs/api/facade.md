# Disentangler (experimental)

A declarative front end. You describe the system — which components exist, how bright each
one is, what the spectrograph does, what is already known about the orbit — and albireo
assembles the fit.

!!! warning "Experimental"
    This is a *vocabulary*, and a vocabulary is expensive to change once people depend on
    it, so it stays marked experimental until it has been used on somebody else's problem.
    [`MarginalOrbitModel`](inference.md) and the functions around it are the supported
    surface and are not going anywhere. `Disentangler.expert()` hands you exactly that
    surface, so dropping down costs three lines rather than a rewrite.

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

## It is a compiler, not a shortcut

The façade emits the expert path rather than hiding it. Four things it *derives* are four
things the low-level path makes you supply correctly, and `dis.explain()` prints every one:

| Derived | Why it is not yours to type |
|---|---|
| The **velocity budget** | It must bound the largest relative velocity the *priors* allow, not the one the answer has. Too small and the sampler stalls against a guard it cannot see; reached through `log_likelihood` directly, too small is quietly *wrong*. The information is already in the `k` priors' support. |
| The **model grid** | Wide enough for that budget plus the LSF kernel radius. Short of that margin the shifted model runs off the grid and the fit silently loses flux there. |
| The **conjunction phase** | Located by a 41-point scan before anything is optimized. The likelihood is sharply multimodal in phase, and L-BFGS in the wrong trough converges confidently to the wrong answer. |
| The **smoothness hyperparameters** | Fitted by empirical Bayes, then frozen for sampling, and reported per component with a flag on any that did not move from its start. |

A fifth thing is structural rather than derived: a spec such as `Between(5.5, 6.5)` carries
both its prior *and* its starting value, so `priors` and `init` cannot drift apart — in the
low-level path they are two dicts written twice with an assertion between them.

## When there is no orbit to declare

Not every binary has a published period, and for the ones that matter most it is the point
that there isn't — BLOeM's 59 double-lined systems have no orbital solutions at all. Declare
the velocities you measured instead:

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

Exactly one of `orbit=` and `velocities=` is required. This exists because the ordering an
unsolved system forces cannot be met the other way round: the free per-epoch table
(`docs/math.md` §7.6) is what *produces* the period, but it needs a warm start — a cold one
is 122,000 nats worse — and the only warm start on offer used to be `Fit.free_velocities()`,
which needs a Keplerian fit, which needs a period.

Three things to know about the declaration:

- The velocities are a **starting point, not a constraint**. The per-component zero point
  stays unidentified, so what they must be right about is the epoch-to-epoch *pattern*, not
  the level — a systemic +150 km/s changes neither the answer nor the solver's bandwidth,
  because the budget is derived from the centred table.
- The one failure mode is **refused rather than discovered**. A declaration whose components
  never separate *is* the cold start, and it raises; velocities that never resolve the pair
  beyond the LSF width warn.
- `scan()` and `detection_limit()` need a known SB1 orbit and say so.

## Where it refuses to be convenient

Each of these is a place where a default would be a scientific claim.

- **Light fractions have no default.** `Star(light=...)` is required and the star lights
  must sum to 1. With constant light fractions the likelihood sees only `l_i * d_i`, so
  every recovered depth scales as `1/l_i` and nothing in the fit can tell you the value was
  wrong. It is repeated in every summary under `Assumed, not measured`.
- **It will not pretend a phase scan is a period search.** The scan resolves phase at *one*
  period — the prior's midpoint — so a prior wide enough to be a search warns and names the
  fix. It degrades rather than failing, so it is a warning rather than a refusal.
- **Air versus vacuum is refused when it matters.** A `Nebular` or `Telluric` component is
  keyed to absolute line positions, so an undeclared wavelength scale raises rather than
  picking one. The difference is a nearly constant 83 km/s.
- **A budget override may not shrink below what the priors reach**, and the error names the
  terms that overflowed it.
- **The eccentricity singularity is designed around, not warned about.** A free `ecc` never
  starts at the origin; `ecc=ab.Fixed(0.0)` does not sample those sites at all.

## What is deliberately not in v1

Per-epoch **jitter**, **AR(1)** correlated noise, **inferred light fractions**, and
**inferred LSF widths**. Each is a one-line site in the low-level API, and each is a
scientific claim rather than a convenience — a `jitter=True` keyword would offer one as the
other. `fit.z_rms` is printed unconditionally so you can see whether you need them, and
`dis.expert()` is how you add them.

Three more are absent because the façade could not deliver them honestly, and each says so
rather than doing something approximate:

- **Hierarchical triples.** The model has the `period_out`/`t_conj_out`/`k_out` sites;
  `Orbit(outer=...)` raises `NotImplementedError` pointing at `expert()`.
- **Gauss-Hermite `h3`.** It reaches the kernel through `build_problem`, not through the
  model class the façade builds, so an `LSF(h3=...)` field would have been silently
  discarded. There is no such field.
- **A lower bound on eccentricity.** The sampled pair is (√e·cos ω, √e·sin ω), in which a
  lower bound on *e* is an annulus rather than a box. `ecc=Between(lo, hi)` with `lo > 0`
  raises instead of quietly dropping it — measured returning half the declared lower bound
  before it did.

`Disentangler` also does not wrap plotting: [`albireo.plotting`](results.md) already covers
it, and wrapping it would double the surface for no new guarantee.

::: albireo.facade
