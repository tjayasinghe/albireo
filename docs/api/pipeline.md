# Pipeline and command line

One declaration in, structured products out. For every star in a list the pipeline reads
the epochs, disentangles them, fits atmospheric labels to the components against a
synthetic grid, measures one velocity per component per epoch against those components,
fits a Keplerian to the table, and writes the products (tables, spectra with their
uncertainty bands, a JSON report, diagnostic figures) into one directory per star, with a
batch-level table and a record of every failure.

```bash
albireo init            # writes an annotated albireo.toml
albireo run albireo.toml --jobs 4
albireo demo            # two simulated stars with known answers, offline
```

The same from Python:

```python
import albireo as ab

run = ab.run_pipeline("albireo.toml", jobs=4)
run.results["AI Phe"].report["orbit"]["k"]     # {"primary": ..., "secondary": ...}
```

## Scope

The pipeline is a driver. Every scientific decision is made by the stage that owns it
([`Disentangler`](facade.md), [`match_labels`](match.md), [`todcor`](todcor.md),
[`fit_rv_orbit`](rvorbit.md)), and the two rules those stages enforce apply unchanged: light
fractions are declared, never defaulted, and the wavelength medium is declared before a
synthetic grid is consulted. Where the pipeline cannot honour a request, it records a flag
on the star's report and continues rather than guessing or stopping the batch.

Three routes to the orbit:

| Declaration | What runs |
|---|---|
| `period = [lo, hi]` (or a value, or `{value, sigma}`) | the Keplerian is inferred from the spectra; the epochs are measured back against the components |
| `period = "search"` | library templates at the declared starting labels measure a first table, four periodograms propose candidate periods, a Keplerian is fitted from each and the chi-square ranks them, the disentangling decides among the best few by its own marginal likelihood, and that orbit warm-starts the disentangling |
| `velocities = "file"` | the free per-epoch table is fitted from measured velocities; the period comes from the table afterwards. An entry that is not a number is refused by name (the epoch, the component): the free fit has one velocity site per component per epoch and no site for an unmeasured one, so such an epoch is removed from the dataset or measured |

**Period search.** The candidates come from four periodograms of the same table
([`find_period`](rvorbit.md)): the twenty highest peaks of the floating-mean generalized
Lomb-Scargle of the relative velocity, the twenty highest of its two-harmonic form, which
ranks an eccentric orbit's period higher, the twenty highest of the first component alone,
the source that survives a companion the templates could not follow (its relative velocity
is then noise while the curve of the primary is intact), and, for two components, the first
four peaks of the swap-invariant search with their doubles, the only source that survives
components exchanged between epochs. The union is deduplicated at 2% in period, which leaves
about four dozen starting periods. An eccentric Keplerian is fitted from each, the epochs are
re-assigned by it where two alike components were exchanged, every orbit above the declared
semi-amplitude ceiling or eccentricity maximum is set aside (the velocities of a companion
the templates lost let a wrong period win at 1537 km/s and e = 0.94; the flag names such an
orbit; a semi-amplitude below the floor is what an unmeasurable companion leaves at any
period and is not tested), and the lowest chi-square wins; fits that converged to the same
period, within 2%, count once. Over the 32 oracle velocity
tables of the [D62 benchmark](../benchmarks.md) this route recovers the period of 28,
against 23 for the six peaks of the classical periodogram it used before. Every other
fitted period within a chi-square difference of 25 of the winner is named in one flag, with
its difference, and recorded in `result.json` under `bootstrap.ambiguous`: the bootstrap
hands the disentangling a period known to 3%, which is unrecoverable if it is the wrong
one, and a table of ten or eleven epochs that a Keplerian fits at a reduced chi-square above
about five cannot be searched at all. On those 32 tables the flag fired on three systems,
which were three of the four the route got wrong, and on none of the 28 it got right.

**The decision by the disentangling.** A table's chi-square cannot always make that choice,
so the best few candidates are put to the disentangling itself. Two blind systems of the
[D62 population](../benchmarks.md) are the case for it: on a table of fifteen epochs an
eccentric Keplerian at 0.2485 d with semi-amplitudes of 233 and 239 km/s at $`e = 0.73`$
fitted better than the true 6.104 d, and on one of twelve epochs the same happened at
1.1294 d with 182 and 185 km/s at $`e = 0.72`$ against a true 5.356 d. Both orbits lie
inside the declared ranges, so the filter above does not reach them; five free parameters
fitted to a dozen noisy velocities have that much freedom. The disentangling has none of
it. Its marginal likelihood uses every pixel of every epoch, one pair of component spectra
has to serve all of them, and there is no per-epoch freedom to absorb a wrong period.

`period_decision_candidates` under `[analysis]` (default 4; `0` leaves the decision to the
table) sets how many of the chi-square ranking's distinct orbits are compared. Each is
declared as the winner would be, with two differences: the conjunction is scanned rather
than held at the table's, which is as uncertain as the period it came from, and every
semi-amplitude is the declared range started at the table's value rather than a Gaussian
around it, which puts every candidate on one velocity budget and therefore one model grid,
a marginal likelihood read off two grids of different length not being one number twice. A
candidate's value is the best its coarse declaration reaches (`_scan_declaration`, twice
the model pixel, a quarter to an eighth of the full model's cost per solve): the
conjunction-phase scan over one period, then, where a semi-amplitude is a range, the coarse
semi-amplitude scan, whose best is taken when it beat the start. The values are comparable
because the data, the noise model, the model grid and the coarse scan grids are the same
for every candidate, and because the marginal log-likelihood carries no prior at all, so
nothing rewards a period for being where a prior expected it. The prior-amplitude profile
and the second scan pass of [`Disentangler.fit`](facade.md) are not run here: the question
at this point is which basin, not where within it. The cost is 42 phase trials plus about
300 semi-amplitude trials per candidate for two components over a hundredfold
semi-amplitude range, order a minute or two per candidate on a dozen epochs, and each
candidate's declaration is released as soon as its value is in hand, since it holds a
compiled model.

When the comparison overrules the table, a flag names both periods and the margin ("the
table preferred P ... at chi2 ...; the disentangling prefers P ... by N nats"). `result.json`
records the comparison under `bootstrap.decision`: `by`, which is `"disentangling"` when it
overruled and `"table"` when the table's choice stood or no comparison ran; one entry per
candidate with its period, table chi-square, marginal value in nats, the scan's best
semi-amplitudes and conjunction, the number of trials and the seconds it took; the chosen
period; and the margins over the runner-up and over the table's choice.

**Component order.** Components must be declared in order of decreasing mass, which for a
main-sequence pair is the brighter star first. The likelihood cannot tell which spectrum
belongs to which star: with a symmetric semi-amplitude prior the conjunction scan sees two
equally deep troughs, the declared assignment and its mirror with the spectra swapped and
rescaled by the light ratio. The pipeline starts the fit with $`K_1 < K_2 < \dots`$ (the
first star moves least), which is a convention rather than a constraint, and the label
stage checks it: a fitted light fraction far from the declared one is flagged as the
signature of a reversed order. With a library configured the starting values come from a
template table fitted at the declared period (the `k-start` stage, or the `light = "measure"`
stage's table), and the convention only breaks the ties the data leave; a component the
table cannot measure starts from its nearest measured neighbour, at 1.5 times its
semi-amplitude for a later-declared (lighter) star and at two thirds for an earlier
(heavier) one, every start held strictly inside the range. The same table's orbit starts a
free eccentricity and argument of periastron where it measured them. A table of a dozen
epochs is not a reliable seed, so before the disentangling's L-BFGS the façade scans the
marginal likelihood over the ranged semi-amplitudes in turn, the phase with the first, on a coarser
grid and starts from the best trial when it beats the seed (`k_scan`, on by default); the
report states what the scan chose, and a flag records a start the scan moved by more than
a factor of 1.5.

**Noise model.** `noise_correlation` under `[analysis]` (a number, a table per instrument,
or `"fit"`) declares the lag-one correlation of each epoch's noise along its pixels, the
signature of resampled spectra; the disentangling then runs the AR(1) noise model and the
velocity table's errors carry the correlation. Without it the pixels are taken as
independent, which is what most archives' errors say and what resampled spectra are not.

**Divergence.** After the disentangling the residual z-score rms is compared against
`z_rms_max` under `[analysis]` (default 10), and a star above it stops with an error before
the label and velocity stages run. That number sits near 1 wherever the noise model holds
and near 2 to 3 for a fit that settled at the wrong period on the blind route; it reached
45 for the one diverged fit of the [D62 benchmark](../benchmarks.md), whose components
correlated with the injected spectra at 0.07 and 0.02 and whose semi-amplitudes were held
by their priors alone. Everything downstream is measured against those components, so a fit
that far from the data still yields a full set of products, each of them arithmetic on a
failure. The star's directory keeps `log.txt`, `error.txt` and any table measured before
the disentangling; the noise model (`noise_correlation` included, where the spectra were
resampled onto a common grid) and the priors are the first things to look at.

**Zero points.** Velocities measured against a disentangled component are differential,
because its rest frame is not identified
([§5.3](../math.md#53-systemic-velocity-zero-point)), unless the label fit measured the
frame's offset, in which case the pipeline applies it to the templates and the velocities
are absolute. A fitted offset that the label stage disowned is refused rather than applied:
when the fit beats neither of its nulls, when a component's offset lies within one trial
step of the bound of the scan over `v_zero_range` (the scan was cut off there rather than
turned round), or when two components' offsets differ by more than the declared velocity
budget on a Keplerian fit, where every component has the same systemic velocity by
construction. The table then stays differential. An offset the fit never learned, its
posterior as wide as the prior it started from (above 0.8 of that width), is refused for
that component alone while the other component's offset is still applied: the mostly-noise
secondary of a third benchmark run kept 11 percent of its equivalent width and had its
offset placed 46 km/s from the systemic velocity, and every velocity of that component
carried it. `result.json` records which case applies (`velocities.absolute`), the offsets
and any reason they were refused (`velocities.zero_points`), and so does the orbit fit,
which uses one systemic velocity per component whenever a component is differential.

**Epoch velocities.** The epochs are correlated against the disentangled components with
the template amplitudes held at the declared light fractions, because the disentangling
recovers a component as $`(w / l_0)\,t`$ at its declared fraction $`l_0`$, which makes
$`l_0`$ the amplitude consistent with the template. That holds only while the posterior
mean is not shrunk, and a smoothness precision far from where ML-II started says it is: a
secondary whose $`\tau`$ the fit raised from 400 to 2000 lost a quarter of its line depth,
and under the declared fraction the table then under-measured its semi-amplitude by 89
percent, against 25 percent with a freely fitted amplitude. Where a component's precision
has left its start by more than a factor of ten the correlation fits the amplitudes freely
instead, the table's light column is that fitted scale rather than a light fraction, and a
flag names the component, its starting value and the fitted one.

**The exchange by the orbit.** Two alike spectra cannot be told apart in a single epoch, so
where the correlation's two peaks are equivalent solutions the disentangling's orbit
decides which component each row belongs to, and a flag counts the epochs it exchanged. At
light fractions a factor of several apart, with the amplitudes held, the peaks are not
equivalent solutions: an exchanged row carries the other component's amplitude, and the
swap is decided on a noise draw of the faint component's velocity. On a 95/5 pair the step
swapped 19 of 80 epochs and moved the primary's semi-amplitude from 5 to 56 percent off, so
the exchange is skipped, with a flag, wherever the two declared fractions differ by more
than a factor of three. Where it runs and an epoch was exchanged, the table as measured is
kept beside the delivered one as `velocities_unexchanged.rv`, which is what separates a
genuine exchange from a swap made on noise.

## Batches

Stars are independent, so `jobs > 1` runs them in a spawn-based process pool. On the
development desktop, four workers finish a batch of eight simulated stars 2.0× faster than
one process and eight workers 2.5× faster. The scaling is sub-linear because a single star
already occupies several cores, so the workers overlap only each star's serial part. Each
worker's XLA and BLAS thread counts are capped at `cpu_count // jobs` as a precaution
against oversubscription; on that benchmark the cap made no measurable difference. A
worker returns a plain-data report; the live objects (the `Fit`, the velocity table, the
label match) are retained only on an in-process run (`jobs=1`), since they carry compiled
JAX programs that cannot cross a pipe. A star that fails is recorded in `failures.txt` and
does not stop the others. The measurements are in
[the benchmark record](../benchmarks.md#the-pipeline-in-worker-processes-2026-09-01).

## Products

Per star, `<output>/<star>/`:

| File | What |
|---|---|
| `summary.txt` | every stage's own report, the assumptions block, the flags |
| `result.json` | the machine-readable report: dataset, declaration (with the noise model), the orbit from the spectra with its Keplerian at every epoch and the scans that preceded it, labels with both error bars, the velocity table (every epoch's velocity, error, light and detection statistic per component), the orbit from the table with errors, timings, flags, files; for a simulation, the truth block with the injected velocity of every epoch |
| `velocities.rv`, `velocities.csv` | the epoch velocity table, twice (commented ASCII with the zero-point status in its header; a CSV): one measured velocity per component per epoch by TODCOR against the disentangled components, absolute when the label fit pinned the zero points. A table whose median R-squared is negative (the templates fit the epochs worse than no template at all), or which has no usable epoch, is marked failed: `# FAILED: <reason>` on the line under the format line, `velocities.status` in `result.json`, a flag, and no orbit fitted to it |
| `velocities_unexchanged.rv` | the table as measured, before the exchange by the orbit, written when an epoch was exchanged |
| `template_velocities.rv`, `template_velocities.csv` | the table measured against library templates at the declared starting labels before the disentangling, on the routes that measure one (the search route's bootstrap, the light measurement, the semi-amplitude start); written as soon as it exists, so it is there when a later stage fails. On the search route its component assignment is the winning orbit's rather than the correlation's, which the header states |
| `template_velocities_unexchanged.rv` | that table as the period search itself saw it, before the bootstrap's winning orbit re-assigned any component, written when the bootstrap re-assigned an epoch; it is what reproduces the periodogram |
| `spectrum_<component>.txt`, `spectra.fits` | the disentangled components with their uncertainty band |
| `orbit.txt`, `labels.txt`, `template_<component>.txt` | the Keplerian from the table; the label report; the label fit's model spectra |
| `fit.npz`, `posterior.npz` | the MAP result for `load_fit`; the NUTS draws when sampling was asked for |
| `spectra.png`, `residuals.png`, `velocities.png`, `phase_scan.png`, `todcor_surface.png`, `rv_curve.png` | the figures |
| `log.txt` | the progress lines with timings, and every warning |

Per batch: `results.json` (every report), `results.csv` (one row per star: period,
eccentricity, semi-amplitudes and systemic velocities with errors, labels, flags),
`summary.txt`, `failures.txt`, and `run.json` (the declaration as run).

Background and references: [science overview](../science.md).

::: albireo.pipeline

::: albireo.cli
