# Pipeline and command line

From one declaration, for every star in a list, the pipeline reads the epochs,
disentangles them, fits atmospheric labels to the components against a synthetic grid,
measures one velocity per component per epoch against those components, and fits a
Keplerian to the table. It writes the products (tables, spectra with their uncertainty
bands, a JSON report, diagnostic figures) into one directory per star, with a batch-level
table and a record of every failure.

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
[`fit_rv_orbit`](rvorbit.md)), and their two rules apply unchanged: light fractions are
declared, never defaulted, and the wavelength medium is declared before a synthetic grid is
consulted. A request the pipeline cannot honour is recorded as a flag on the star's report,
and the batch continues without guessing.

Three routes to the orbit:

| Declaration | What runs |
|---|---|
| `period = [lo, hi]` (or a value, or `{value, sigma}`) | the Keplerian is inferred from the spectra; the epochs are measured back against the components |
| `period = "search"` | library templates at the declared starting labels measure a first table, four periodograms propose candidate periods, a Keplerian is fitted from each and the chi-square ranks them, the disentangling decides among the best few by its own marginal likelihood, and that orbit warm-starts the disentangling |
| `velocities = "file"` | the free per-epoch table is fitted from measured velocities; the period comes from the table afterwards. A non-numeric entry is refused by name (epoch and component), because the free fit has one velocity site per component per epoch and none for an unmeasured one; such an epoch must be measured or removed from the dataset |

**Period search.** The candidates come from four periodograms of the same table
([`find_period`](rvorbit.md)): the fifty highest peaks of the floating-mean generalized
Lomb-Scargle of the relative velocity; the fifty highest of its two-harmonic form, which
ranks an eccentric orbit's period higher; the fifty highest of the first component alone,
which survives a companion the templates could not follow (the relative velocity is then
noise while the primary's curve is intact, and every epoch at which the primary was
measured is kept); and, for two components, the first four peaks of the swap-invariant
search with their doubles, the only source that survives components exchanged between
epochs. The lists are merged round robin by rank and deduplicated at 2% in period, which
leaves 60 to 89 starting periods on the blind Gaia-like tables of the benchmark's third run
and 70 to 98 on its field tables. A search with too few epochs for its model contributes
nothing and the others stand. In the bootstrap only, a table of at most 25 usable epochs
gets a fifth source after the merge, the leave-one-out search (below).

An eccentric Keplerian is fitted from each start, and the epochs are re-assigned by it
where two alike components were exchanged and the light allows it (below). Every orbit
above the declared semi-amplitude ceiling or eccentricity maximum is set aside and named in
a flag: the velocities of a companion the templates lost let a wrong period win at
1537 km/s and e = 0.94. A semi-amplitude below the floor is what an unmeasurable companion
leaves at any period and is not tested. The lowest chi-square wins; fits that converged to
the same period within 2% count once. Over the 32 oracle velocity tables of the
[D62 benchmark](../benchmarks.md) this route recovers the period of 28, against 23 for the
six peaks of the classical periodogram it replaced. Every other fitted period within a
chi-square difference of 25 of the winner is named, with its difference, in one flag and
recorded in `result.json` under `bootstrap.ambiguous`. The bootstrap hands the
disentangling a period known to 3%, which is unrecoverable if wrong, and a table of ten or
eleven epochs that a Keplerian fits at a reduced chi-square above about five cannot be
searched at all. On those 32 tables the flag fired on three systems, three of the four the
route got wrong, and on none of the 28 it got right.

**The detection gate.** In the bootstrap's period search and its candidate fits, and nowhere
else, a companion velocity whose detection statistic (the table's `delta_chi2`, the rise in
chi-square when that component is removed) is below `detection_min` under `[analysis]`
(default 100; `0` turns the gate off) carries no weight. It is set to `nan` in a copy of the
table that only the search and the fits see, and the epoch's first component is kept. Only
the components after the first (declared in order of decreasing mass) are gated. The gate
reaches past the search in one respect: the winning orbit is fitted to the gated copy and
seeds the disentangling's semi-amplitude, conjunction and eccentricity starts. It does not
reach the table written as `template_velocities.rv`, the light measured from that table,
the template table of the known-period route, or the velocities measured after the
disentangling, which all stay as measured. `result.json` counts the velocities removed per
component under `bootstrap.detection_gate`, and the stage's log line says the same.

The threshold was set by measurement on the 33 blind tables of the Gaia RVS benchmark's
third run: at 100 the gate recovers the period of one field system whose 7% secondary the
library templates detected at no epoch, and costs no system its rank 1; at 25, the table
summary's own threshold for a weak detection, it fails on that system. Gating the first
component as well cost a system, so the first component is never gated. The measurement is
in [Benchmarks](../benchmarks.md#after-the-third-run-d65-2026-09-17-measured-on-its-archive-not-rerun).

**Leave-one-epoch-out candidates.** A periodogram cannot outvote one wrong epoch among ten
to twenty-five; a Keplerian fit with re-assignment can. On a Gaia system of twins whose
correlation exchanged the components at one epoch of twelve, with detection statistics of
15,641 and 15,577 that mark nothing, the true period sat at peak 108 of the one-harmonic
search and at no start, while the orbit fitted from it beats the run's winner by 514 in
chi-square. In the bootstrap, on a table of at most 25 usable epochs as measured (before
the detection gate), the one-harmonic search is therefore repeated with each of those
epochs left out in turn, and the three highest peaks of each are appended after the round
robin, in epoch order, wherever they lie more than 2% from every start already listed. The
route that fits declared velocities (`velocities = "file"`) does not run it. Over the 17
blind tables of at most 25 usable epochs this adds 18 starts in total, none on eight of
them and seven on that system, which it takes from absent to rank 1; one other system moves
from rank 37 to 38 and none loses its rank 1. The block never removes a periodogram start,
and the round robin before it keeps its property that a larger peak count only appends. A
deeper periodogram list can, however, take a leave-one-out start's place with a periodogram
start within 2% of it. `bootstrap.n_leave_one_out` counts the starts added.

**Few nights.** A bootstrap table whose usable epochs fall on fewer than eight nights
(distinct integer parts of the BJD) is flagged as one whose velocities cannot be expected to
decide the period; `bootstrap.n_nights` records the count. The flag changes nothing. Three
of the 33 blind tables were that sparse, and the search recovered none of them. On one,
nine usable epochs on seven nights, the injected velocities with noise at the quoted errors
put the true period first in 2 of 10 draws, every loss inside a chi-square difference of 5.

**The search changes together (D65).** With the detection gate, the leave-one-out source, the
exchange rule and the anchored frequency grid of [`find_period`](rvorbit.md) all in place, the
chi-square ranking of the 33 blind tables of the third run puts the true period first on 22
systems against 20, gaining the Gaia twins and the field system with the 7% secondary and
losing none. The other ranks move among near-degenerate candidates: Gaia systems from 39 to 12,
23 to 24, 20 to 9, 3 to 4 and 37 to 39, and field systems from 40 to 41, 2 to 6 and 24 to
absent. The move from 2 to 6 comes from the grid, and it takes that system's true period out
of the four candidates the decision by the disentangling compares (below); that decision has
converted no miss into a hit on either population (D64).

**The decision by the disentangling.** The table's chi-square cannot always choose, so the
best few candidates are put to the disentangling itself. Two blind systems of the
[D62 population](../benchmarks.md) show why: on a table of fifteen epochs an eccentric
Keplerian at 0.2485 d with semi-amplitudes of 233 and 239 km/s at $`e = 0.73`$ fitted
better than the true 6.104 d, and on one of twelve epochs the same happened at 1.1294 d
with 182 and 185 km/s at $`e = 0.72`$ against a true 5.356 d. Both orbits lie inside the
declared ranges, so the filter above does not reach them; five free parameters fitted to a
dozen noisy velocities have that much freedom. The disentangling does not: its marginal
likelihood uses every pixel of every epoch, one pair of component spectra must serve all of
them, and no per-epoch freedom can absorb a wrong period.

`period_decision_candidates` under `[analysis]` (default 4; `0` leaves the decision to the
table) sets how many of the chi-square ranking's distinct orbits are compared. Each is
declared as the winner would be, with two differences. The conjunction is scanned rather
than held at the table's value, which is as uncertain as the period it came from. Every
semi-amplitude is the declared range started at the table's value rather than a Gaussian
around it, which puts every candidate on one velocity budget and therefore one model grid;
marginal likelihoods read off grids of different length are not comparable. A candidate's
value is the best its coarse declaration reaches (`_scan_declaration`, twice the model
pixel, a quarter to an eighth of the full model's cost per solve): the conjunction-phase
scan over one period, then, where a semi-amplitude is a range, the coarse semi-amplitude
scan, whose best is taken when it beats the start. The values are comparable because the
data, the noise model, the model grid and the coarse scan grids are the same for every
candidate, and the marginal log-likelihood carries no prior, so nothing rewards a period for
being where a prior expected it. The prior-amplitude profile and the second scan pass of
[`Disentangler.fit`](facade.md) are not run: the question here is which basin, not where
within it. The cost is 42 phase trials plus about 300 semi-amplitude trials per candidate
for two components over a hundredfold semi-amplitude range, of order a minute or two per
candidate on a dozen epochs. Each candidate's declaration holds a compiled model and is
released as soon as its value is in hand.

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
first star moves least), a convention rather than a constraint, and the label stage checks
it: a fitted light fraction that differs from the declared one by more than a factor of
1.5, or by more than 0.15, is flagged as the signature of a wrong declaration or a reversed
order. With a library configured, the starting values come from a template table fitted at
the declared period (the `k-start` stage, or the `light = "measure"` stage's table), and
the convention only breaks the ties the data leave. A component the table cannot measure
starts from its nearest measured neighbour, at 1.5 times its semi-amplitude for a
later-declared (lighter) star and at two thirds for an earlier (heavier) one, every start
held strictly inside the range. The same table's orbit starts a free eccentricity and
argument of periastron where it measured them. A table of a dozen epochs is not a reliable
seed, so before the disentangling's L-BFGS the façade scans the marginal likelihood on a
coarser grid over the ranged semi-amplitudes in turn, and the phase with the first, and
starts from the best trial when it beats the seed (`k_scan`, on by default). The report
states what the scan chose, and a flag records a start the scan moved by more than a factor
of 1.5.

**Noise model.** `noise_correlation` under `[analysis]` (a number, a table per instrument,
or `"fit"`) declares the lag-one correlation of each epoch's noise along its pixels, the
signature of resampled spectra; the disentangling then runs the AR(1) noise model and the
velocity table's errors carry the correlation. Without it the pixels are taken as
independent, which is what most archives' errors state and what resampled spectra are not.

**Divergence.** After the disentangling the residual z-score rms is compared with
`z_rms_max` under `[analysis]` (default 10); a star above it stops with an error before the
label and velocity stages. The rms sits near 1 wherever the noise model holds and near 2 to
3 for a fit that settled at the wrong period on the blind route. It reached 45 for the one
diverged fit of the [D62 benchmark](../benchmarks.md), whose components correlated with the
injected spectra at 0.07 and 0.02 and whose semi-amplitudes were held by their priors
alone. Everything downstream is measured against those components, so such a fit would
still yield a full set of products, each of them arithmetic on a failure. The star's
directory keeps `log.txt`, `error.txt` and any table measured before the disentangling; the
noise model (`noise_correlation` included, where the spectra were resampled onto a common
grid) and the priors are the first things to check.

**Labels.** The label stage fits Teff, log g, [M/H] and v sin i to every star against the
library ([`match_labels`](match.md)). `compare` under `[labels]` (or `label_compare` under
`[analysis]`) chooses what the templates are compared with: `"epochs"` (the default), the
template composite against the epoch spectra through the disentangling's sufficient
statistics at its MAP, which is exact for the fit's noise model and independent of the
declared light fractions; or `"native"` and `"matched"`, the templates against the
disentangled components under a diagonal likelihood. Converged the same way on simulated
Gaia RVS binaries, the epoch comparison halved the native comparison's temperature error
and cut its light-fraction error by a factor of three
([§9.2a](../math.md#92a-comparing-in-the-epoch-space)). `steps` (`label_steps`) caps each
Levenberg-Marquardt run of the epoch comparison and the L-BFGS steps of the other two.

`result.json` records the comparison (`labels.compare`), the light fraction the fit
measured with its formal error (`labels.flux_ratio`, `labels.flux_ratio_errors`) beside the
declared one (`labels.light_declared`), the sites the epoch fit left without a formal error
and why (`labels.at_bounds`), anything its restarts could not resolve (`labels.notes`), a
summary of its optimisation (`labels.epoch_fit`), and the operator it compared through
(`labels.epoch_operator`): the model grid's pixel (`dv_kms`), the library's resolving power,
the declared widths reduced for it (`quadrature_sigma_kms`), the discretisation variance of
the model grid, $`\tfrac{7}{12}\,\Delta v^2`$, removed from them (`grid_variance_kms2`), the
widths the operator applied (`operator_sigma_kms`), and what that compensation could not do
(`notes`). The stage raises five flags: a label whose posterior is as wide as its prior; a
label on a bound or on the rotation plateau, which in the epoch comparison has no formal
error and so no posterior width for the first flag to read; a note from the restarts, such
as a faint component whose light collapsed; a grid compensation that was floored at half a
model pixel, or that ran on a model grid coarser than the width it compensates, each naming
the grid spacing that avoids it; and a light fraction a factor of 1.5 or 0.15 from the
declared one. The model grid stays the native pixel by default: with the compensation it
returned 9.82 km/s for a star rotating at 11 km/s on the closed-loop test's 4.63 km/s grid,
against 6.91 without it, and a 2 km/s grid without it returned 9.78 in 1.5 times the time
([§9.5](../math.md#95-uncertainties-and-why-the-formal-one-is-not-enough)). The formal errors
of the light and of v sin i were measured too small by about three
([§9.5](../math.md#95-uncertainties-and-why-the-formal-one-is-not-enough)).

**Zero points.** Velocities measured against a disentangled component are differential,
because its rest frame is not identified
([§5.3](../math.md#53-systemic-velocity-zero-point)), unless the label fit measured the
frame's offset; the pipeline then applies it to the templates and the velocities are
absolute. A fitted offset that the label stage disowned is refused rather than applied:
when the fit beats neither of its nulls, when a component's offset lies within one trial
step of the bound of the scan over `v_zero_range` (the scan was cut off there rather than
turned round), or when two components' offsets differ by more than the declared velocity
budget on a Keplerian fit, where every component has the same systemic velocity by
construction. The table then stays differential. An offset the fit never learned (its
posterior above 0.8 of the width of the prior it started from) is refused for that
component alone, and the other component's offset is still applied. The mostly-noise
secondary of a third benchmark run kept 11 percent of its equivalent width, had its offset
placed 46 km/s from the systemic velocity, and every velocity of that component carried it.
`result.json` records which case applies (`velocities.absolute`), the offsets and any
reason they were refused (`velocities.zero_points`), and so does the orbit fit, which uses
one systemic velocity per component whenever a component is differential.

**Epoch velocities.** The epochs are correlated against the disentangled components with
the template amplitudes held at the declared light fractions: the disentangling recovers a
component as $`(w / l_0)\,t`$ at its declared fraction $`l_0`$, so $`l_0`$ is the amplitude
consistent with the template. This holds only while the posterior mean is not shrunk, and a
smoothness precision far from where ML-II started indicates that it is. A secondary whose
$`\tau`$ the fit raised from 400 to 2000 lost a quarter of its line depth, and under the
declared fraction the table then under-measured its semi-amplitude by 89 percent, against
25 percent with a freely fitted amplitude. Where a component's precision has left its start
by more than a factor of ten, the correlation fits the amplitudes freely instead, the
table's light column is that fitted scale rather than a light fraction, and a flag names
the component, its starting value and the fitted one. `velocities.light_source` in
`result.json` records which applied (`"declared"` or `"global re-measure"`).

The light fraction the label fit measures is not used here, although in the epoch
comparison it is the better measurement of the light: on the orbit tier of the D65
benchmark products it erred by a median 0.011 against 0.043 for the declared value. The
correlation needs the amplitude that reproduces the epochs with these templates, and a
disentangled component scaled to $`l_0`$ is reproduced by $`l_0`$, not by the true
fraction. On a simulated pair with true light 0.70 and 0.30 and templates from a
disentangling at each of four declarations, holding the true fractions instead of the
declared ones raised the fainter component's rms velocity error from 0.35 to 0.84 km/s at a
declaration of 0.6 and 0.4, from 0.34 to 14.3 km/s at 0.45 and 0.55, and to 71 km/s at 0.9
and 0.1, where the components exchanged. With the templates rescaled by $`l_0 / w`$ the
velocities were identical to the declared case at every declaration.

**The exchange by the orbit.** Two alike spectra cannot be told apart in a single epoch, so
where the correlation's two peaks are equivalent solutions the disentangling's orbit
decides which component each row belongs to, and a flag counts the epochs it exchanged. At
light fractions a factor of several apart, with the amplitudes held, the peaks are not
equivalent: an exchanged row carries the other component's amplitude, and the swap is
decided on a noise draw of the faint component's velocity. On a 95/5 pair the step swapped
19 of 80 epochs and moved the primary's semi-amplitude from 5 to 56 percent off, so the
exchange is skipped, with a flag, wherever the two declared fractions differ by more than a
factor of three. Where it runs and an epoch was exchanged, the table as measured is kept
beside the delivered one as `velocities_unexchanged.rv`, which separates a genuine exchange
from a swap made on noise. The bootstrap's candidate fits follow the same rule on the light
fractions of the library table's own amplitudes. Without it (before D65), the winning
orbits of two field systems whose amplitudes stood a factor 7.1 and 3.25 apart had
re-assigned 3 and 11 of their epochs. Over the 33 blind tables of the third run the rule
skips the exchange on 8 and costs no system its rank 1; under it one Gaia system's true
period moves from rank 20 to 28, another's from 37 to 38, and the field system with the 7%
secondary from absent to rank 25. `bootstrap.exchange_allowed` records the decision.

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
| `velocities.rv`, `velocities.csv` | the epoch velocity table, twice (commented ASCII with the zero-point status in its header; a CSV): one measured velocity per component per epoch by TODCOR against the disentangled components, absolute when the label fit pinned the zero points. The epoch times carry every digit of a float64, so that a period search on the written table reproduces the run's; every other number carries six decimals. The `at_edge` column is merged over the components in both files; the CSV also carries one `at_edge_<component>` column per component, which says which component's velocity was lost at the search edge (a velocity counts wherever its own component was measured). A table whose median R-squared is negative (the templates fit the epochs worse than no template at all), or which has no usable epoch, is marked failed: `# FAILED: <reason>` on the line under the format line, `velocities.status` in `result.json`, a flag, and no orbit fitted to it |
| `velocities_unexchanged.rv` | the table as measured, before the exchange by the orbit, written when an epoch was exchanged |
| `template_velocities.rv`, `template_velocities.csv` | the table measured against library templates at the declared starting labels before the disentangling, on the routes that measure one (the search route's bootstrap, the light measurement, the semi-amplitude start); written as soon as it exists, so it is there when a later stage fails. On the search route its component assignment is the winning orbit's rather than the correlation's, which the header states. Same format as `velocities.rv` and `velocities.csv`, per-component edge flags in the CSV included |
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
