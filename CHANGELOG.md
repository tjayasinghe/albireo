# Changelog

All notable changes to albireo are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and albireo uses
[semantic versioning](https://semver.org/spec/v2.0.0.html), with the caveat that while the
version is below 1.0 the public API may change in any release.

Below 1.0 that caveat covers everything, so it is worth saying which surfaces it is
expected to bear on, because the two are not equally settled.

- **Supported.** The model layer: `data`, `grids`, `preprocess`, `forward`, `likelihood`,
  `priors`, `inference`, `kepler`, `operators`, `scan`, `calibrate`, `simulate`,
  `examples`, `io`, `results` and `plotting`. A change here is recorded below with what to
  do about it.
- **Experimental**, and marked so in each module's own docstring: `facade` (the
  `Disentangler` declaration), `pipeline` together with the `albireo` command line and the
  TOML schema it reads, `todcor`, `rvorbit`, `library`, `match`, `forecast`, `archive` and
  `handoff`. Their vocabulary is still settling and may be renamed or reshaped in any
  release below 1.0. Each is a convenience over the model layer, which reaches the same
  computation directly.
- **Internal**, with no promise at all: `assembly` and `solver`.

Every name above is re-exported at the top level, so the module is what carries the
marker, not the import path.

This file records what changed. The reasons are recorded elsewhere:
[`internal/design.md`](internal/design.md) §2 is the decision ledger, and
[`docs/benchmarks.md`](docs/benchmarks.md) is the validation and performance record.

## [Unreleased]

### Added

- **The table the period search ran on is kept beside the delivered one (D64).** On the
  search route the delivered `template_velocities.rv` carries the winning orbit's component
  assignment rather than the correlation's, because the bootstrap re-assigns the epochs
  where the correlation exchanged two alike spectra. Its header now says so and names the
  count, and the table as measured is written beside it as
  `template_velocities_unexchanged.rv` whenever an epoch moved. Without it the period search
  cannot be reproduced from the written products, which was found by an analysis that tried
  and could not.

- **The template velocity table is a product of every run that measures one (D63).** The
  bootstrap of the search route, the light measurement and the semi-amplitude start each
  measure the epochs against library templates at the declared starting labels before
  anything is disentangled; that table decides the period and seeds the semi-amplitudes,
  and it is now written as soon as it exists (`template_velocities.rv` and `.csv`, with the
  purpose in the header), so it is on disk when a later stage fails and the decision it
  fed can be examined.
- **Real Gaia transit times from GOST (D63).** `gost_transits(ra_deg, dec_deg)` fetches
  the forecast of a sky position's field-of-view crossings from ESA's ObjVisSAP endpoint
  (the nominal scanning law, barycentric MJD in TCB, no CCD row and no scan angle),
  parses the VOTable with the standard library and caches the response under the cache
  directory; `rvs_transit_times_from_gost` keeps the crossings inside the release span,
  draws the RVS rows (four of the seven, the service not publishing which) and thins
  them to the usable fraction of 0.78 (GOST's own reception rate, Katz et al. 2023 and
  the DR3 counts agree on it). `BinarySystem` carries `ra_deg` and `dec_deg`, filled
  from the Gaia query or drawn at the population's ecliptic latitude and rotated to ICRS
  (`ecliptic_to_icrs`, within an arcsecond of astropy's mean ecliptic), and the
  benchmark's `cadence="gost"` (`--cadence gost`) simulates each system at its own
  forecast epochs, with `system_transit_times` to compute them first and the minimum
  transit rule applied to the forecast. The statistical cadence stays the default; where
  a position is known the phase of the scanning law is now reproduced, which the first
  two benchmark runs had stated as an approximation.
- **A BOSZ 2024 box above 7000 K for A and early-F stars (D63).** `bosz2024-hot-rvs` and
  `bosz2024-hot-r20000` cover 7000 to 10,000 K in 250 K steps, log g 3.5 to 5.0 and [M/H]
  -1.0 to +0.5: 364 nodes, every one published (verified against the archive listing on
  2026-09-10, a harvest of which is a test fixture), so the cubic interpolant applies with
  nothing filled. The box crosses the MARCS/ATLAS9 seam between 8000 and 8250 K, and its
  caveats say what that costs, what the Paschen series does to a velocity from the RVS
  band above 8000 K, that A stars rotate faster than R = 11,500 resolves, that the LTE
  error there is unquantified, and that the archive's hot shards predate the hydrogen-line
  recomputation date. 532 MB to fetch once (`fetch_library`), 4 MB cached for the RVS
  slice. The benchmark script's `--library` admits the catalogue systems above 7000 K with
  it. The FGK entries' declared sizes are corrected to the measured ones (621 MB to fetch,
  5 MB and 51 MB cached; they had said 645, 16 and 95).

- **The measured epoch velocities are a product of every run, and of the benchmark (D62).**
  The pipeline has always written one velocity per component per epoch, by TODCOR against
  the disentangled components with the label fit's zero points (`velocities.rv`,
  `velocities.csv`); `result.json` now also carries the disentangling's own Keplerian at
  every epoch and, for a simulation, the injected velocity of every epoch in the order the
  components were compared in. The benchmark gathers them: `collect_velocities` writes
  `velocities.csv` (one row per system, tier, epoch and component, with the error, the
  flags, the injected velocity and the pull), `summarize_velocities` pools the usable
  epochs per tier into an "Epoch velocities" section of the report, and every tier's
  systems are drawn phase-folded against the injected orbit (`figures/rv_curves_<tier>.png`).
- **A declared noise correlation, end to end.** `Disentangler(noise_correlation=...)`
  (a number, a table per instrument, or a `Between`/`Known` to fit one shared value)
  makes the noise model AR(1) along the pixel index, held or fitted through the
  `ar1_phi` site the low-level model already had; `explain()` states it and a declared
  value is listed as an assumption. `todcor(noise_correlation=...)` replaces the
  curvature error by the sandwich through it (math.md §10.4), which
  `Fit.measure_velocities` applies with the declaration's value. The pipeline's
  `noise_correlation` setting carries it to every stage, and the benchmark declares the
  delivered grid's lag-one correlation to every tier as the simulation measured it
  (`noise_model="correlated"`, the default): the pixels of a resampled spectrum are not
  independent, and the archive's errors do not say so.
- **A joint scan over the semi-amplitudes and the phase before the fit.** With a
  semi-amplitude declared as a range, `Disentangler.fit` (`k_scan="auto"`) scans the
  marginal likelihood over a geometric grid of the ranged semi-amplitudes crossed with a
  grid of conjunction phases, refined twice around the best trial, on a copy of the
  declaration with the model grid at twice the pixel, and starts L-BFGS from the best
  trial when it beats the declared start; the record is `fit.k_scan` and the pipeline's
  `k_scan` setting, on by default, reports and flags what it moved. The Gaia DR3
  population had shown a template table of a dozen epochs seeding the range-K routes in
  the wrong basin on half the systems, and a first scan over the semi-amplitudes alone
  showed why the phase must be scanned with them: located at semi-amplitudes a factor of
  three off, the phase can sit a quarter of a period from the truth. The template table's
  orbit now also starts a free eccentricity and argument of periastron
  (`Orbit(ecc=Between(0, hi, start_at=e), omega=w)`) when it detected them at three sigma,
  so the scan runs on the right shape of velocity curve.
- **The one gap in the BOSZ FGK box is filled, and the cubic interpolant returns.** BOSZ
  publishes no model at 5750 K, log g 3.0, [M/H] -0.75, and that single gap cost the whole
  grid its box structure: `library_interpolator` fell back to the barycentric simplex path,
  which is piecewise linear over an arbitrary triangulation of the lattice, and the label
  fit stopped at its kinks. On a perfect spectrum drawn from the grid itself the fit ended
  20 to 150 K from the truth with a chi-square above the truth's; with the box completed
  and the Catmull-Rom cubic in use the same fits end within about 30 K. `ingest_bosz` now
  fills a missing node that two published neighbours bracket along one axis by linear
  interpolation between them (metallicity first), records it in `meta["filled_nodes"]`,
  and the library summary and the label report name it; an unbracketed node is still
  dropped. The registry version of both BOSZ libraries moves to 2, so cached builds are
  remade from the raw shards on first use. This is the temperature bias the first Gaia
  RVS benchmark reported as open (D61).
- **Gaia RVS as a simulator, populations of double-lined binaries, and a benchmark that
  runs them through the pipeline (D60, D61).** `albireo.gaia` reproduces the Gaia RVS
  observation chain of Rowan's reference notebook with every constant verified against a
  primary source: G_RVS from the colour, the S/N per detector pixel from G_RVS and the
  transit count, photon noise on the detector grid, delivery onto the DR3 mean-spectrum or
  the DR4 epoch grid with the variance propagated and the noise correlation recorded, the
  in-flight resolving powers per CCD row, and a transit cadence with the scanning law's
  measured structure. `albireo.population` draws systems from the field's distributions
  (Moe & Di Stefano 2017, a dwarf sequence, the eclipse geometry, tidal synchronisation,
  the RVS light ratio from the library continua) as a magnitude-limited double-lined
  survey sees them, or builds them from DEBCat and the Gaia DR3 double-lined orbits. `albireo.benchmark` simulates each system once, runs it under
  knowledge tiers (`oracle`, `eclipsing`, `orbit`, `blind`) through `run_pipeline`, and
  writes `report.md` with figures, `rows.csv` and `summary.json`; `scripts/gaia_rvs_benchmark.py`
  drives it and resumes. New names: `simulate_rvs_dataset`, `rvs_snr_per_pixel`,
  `predict_grvs`, `rvs_transit_times`, `RVS_DR3_MEAN`, `RVS_DR4_EPOCH`, `RVSProduct`,
  `RVSTruth`, `BinarySystem`, `MainSequence`, `draw_population`, `from_debcat`,
  `from_gaia_sb2`, `query_gaia_sb2`, `read_population`, `write_population`,
  `BenchmarkConfig`, `BenchmarkRun`, `Tier`, `run_benchmark`, `write_report`,
  `t_conj_from_t_peri`. `simulate_dataset` gains photon-counting noise
  (`InstrumentSpec.shot_noise`, `snr_window`) and a per-epoch `epoch_snr`; the truth
  records the noise sigma. `examples/14_gaia_rvs.py` reproduces the notebook's system.
- **Pipeline declarations for what an eclipsing-binary user knows.** `StarConfig` takes
  `t_conj` (a known conjunction instead of the phase scan), `ecc` (held or a range) and
  `omega`; `ComponentConfig(light="measure")` measures the light fractions against library
  templates on any route with a library; the truth block reports element and velocity
  pulls, the recovered spectra's correlation and equivalent widths per window, and the light
  fractions as declared and as measured. `find_period` oversamples the baseline.
  `reassign_by_orbit` exchanges the two components of a velocity table at the epochs
  where the disentangling's orbit says the correlation picked the mirror minimum, which
  alike components at similar light fractions make it do; the pipeline applies it and
  flags the count. `find_period(swap_invariant=True)` searches the magnitude of the
  relative velocity, which the exchange leaves unchanged; the search route's bootstrap
  tries its peaks and their doubles, re-assigns the table by each candidate orbit, and
  falls back to the declared K range when a bootstrap semi-amplitude is degenerate. A
  semi-amplitude declared as a range now starts where a library-template table fitted at
  the declared period puts it (a component the table could not measure starts from its
  nearest measured neighbour, above it for a lighter star and below it for a heavier one,
  and every start is held strictly inside the range; `Between.start` now refuses a start on
  a bound instead of leaving numpyro to say "cannot find valid initial parameters"),
  instead of at evenly spaced points of the range: over a range of
  2 to 250 km/s those points started a 23 km/s primary at 85 km/s, and the fit settled in
  the static-component minimum. The truth block recognises a pair recovered in the other
  order (twins, where the mass-order convention has nothing to work on), flags it, and
  compares in that order.
### Changed

- **The period search is a floating-mean periodogram with twenty candidates, and the
  orbit fit decides among more of them (D63).** `find_period` computes the weighted
  generalized Lomb-Scargle periodogram of Zechmeister and Kurster (2009), with a free
  constant, in numpy, in place of `scipy.signal.lombscargle` on the series scaled by the
  square root of its weights: ten to twenty-five epochs falling into a dozen visibility
  windows hundreds of days apart have a sampling window whose mean is large at most
  frequencies, and a sinusoid with no constant term absorbs it as spurious power. On the
  oracle-tier tables of the second Gaia benchmark the true period ranked 9, 9, 15, 25 and
  103 among the distinct peaks on the five Gaia-like systems the search had lost, and 1, 1,
  1, 10 and 3 with the floating mean. `n_peaks` (default twenty, the best plus nineteen
  aliases) replaces the fixed six, the greedy peak loop walks the local maxima only, and
  `n_harmonics=2` fits a fundamental and its first harmonic for eccentric orbits. The
  pipeline's search route proposes the union of the twenty single-harmonic peaks, the
  twenty two-harmonic peaks, the twenty peaks of the first component's own velocities
  (the source that survives a companion the templates could not follow, whose relative
  velocity is then noise while the primary's curve is intact) and the swap-invariant
  peaks with their doubles, sets aside every candidate orbit above the declared
  semi-amplitude ceiling or eccentricity maximum (a 7-percent secondary's garbage
  velocities had let a wrong period win at 1537 km/s and e = 0.94; a semi-amplitude
  below the floor is what an unmeasurable companion leaves at any period and is not
  tested), fits the
  eccentric Keplerian at every distinct start, merges fits agreeing within 2 percent on
  the fitted period, keeps the lowest chi-square, and names every distinct fitted period
  within 25 in chi-square of the winner as an ambiguity (`bootstrap.ambiguous`,
  `n_candidates`, `harmonic_peak` in the report). On the same 32 tables the recovery within
  2 percent went from 23 to 28, the four left being two tables of ten or eleven epochs whose
  true Keplerian fits worse than an alias (the flag fires on both), one period 2.04 percent
  off and one decision lost by 4.4 in chi-square. The frequency grid (ten per inverse
  baseline) and the 2 percent distinctness rule were tested and kept; a period near a
  third of the baseline is quoted to a few percent by the grid and pinned by the fit.
  On the third Gaia run the table's chi-square still chose absurd orbits on the noisier
  bootstrap tables (twins at 6.10 d sent to 0.248 d with K 233/239 at e 0.73; a 5.36-day
  pair to 1.13 d), inside the declared ranges, so the disentangling itself now decides
  among the best few candidates (`Analysis.period_decision_candidates`, default 8: on the third run's misses the truth ranked 4 to 14 in the table's chi-square): each
  candidate is declared with every semi-amplitude as the same range, so the model grids
  match, its conjunction and semi-amplitudes located by the coarse scans, and the prior-free
  marginal log-likelihood compared; the table is overruled with a flag that names both
  choices and the margin in nats, and `bootstrap.decision` records every candidate. About
  340 coarse solves per candidate.
- **The semi-amplitude scan takes its axes in turn, and judges the move jointly (D63).**
  A measurement on the Gaia system whose 7-percent secondary the D62 scan had lost showed
  where a product grid fails: the companion commands some 25 nats over its whole range
  and prefers its true semi-amplitude only while the primary's is within about ten
  percent of the truth, whereas the primary's own peak falls by 50 nats within twenty
  percent, so on a grid at ratio 1.8 no trial held the primary close enough for the
  companion's evidence to point the right way, and the local refinement could not carry
  the companion out of the basin it was placed in; the guard then refused the correct
  joint move (39 nats) as two single-component holds of 15 nats each. The scan now takes
  the first ranged component over a geometric grid at ratio 1.25 with eight phases, each
  further component over its own grid at what has been located, refines jointly twice,
  takes each component once more over its whole grid at the refined best, and tries the
  exchange-symmetric twin of the best for two ranged components; the start moves when the
  best trial beats it by more than 25 nats jointly, and a component whose own move is
  worth less than 5 nats returns to its start. Fewer trials than before (about 270
  against 446 for two components), and the record's notes say which guard acted.
  Between two passes each star's prior amplitude is profiled at the located orbit
  through the model's light site: the marginal likelihood is exactly invariant under
  scaling a star's light by a factor and its smoothness hyperparameters by the factor
  squared (to 5e-10 nats), so the profile measures the amplitude the data want and not
  the light (it sat at 0.3 to 0.7 of the injected fraction on four systems), and where
  it asks for a factor of two or more the hyperparameter starts are moved and the scan
  repeated (`fit.k_scan.prior_scales`, a note in the record). A secondary declared at
  six times its light, which had sent the scan to a static companion, came back at 4
  percent of its semi-amplitude that way.

### Fixed

- **The conjunction-phase grid holds its own antipodes (D64).** `Disentangler._scan_phase`
  laid 41 trials over one period, an odd count, so the antipode of every trial it sampled
  fell midway between two others and the grid could not choose between the two mirrors of a
  near-equal pair. The count is now 42, which is even and also finer. On one benchmark star
  the antipode the old grid could not reach was better by 672 nats, and the later
  semi-amplitude scan found it and moved the conjunction half a period on, which is how the
  D63 initialisation failure began.
- **A star whose files declare no wavelength medium no longer fails at the semi-amplitude
  start (D64).** Where a semi-amplitude is a range and a library is configured, the pipeline
  seeds the starting values from a template table. Nobody asks for that table, but it raised
  the medium refusal when the files declared no air-or-vacuum scale, which failed the star
  and, among others, the shipped `albireo demo`. The stage is now skipped with a flag saying
  what was lost and how to get it back, which is what the module's policy says an unrunnable
  stage should do. The refusal stands unchanged where the user declared `period = "search"`
  or `light = "measure"`, which are stages that cannot proceed without the medium.
- **Three claims in `docs/math.md` section 5.2 that the package does not back (D64).** The
  physicality floor was described as "available as an optional constraint"; nothing imposes
  it anywhere, and it is now described as what it is, a bound worth reporting beside a
  declared light and measured over 66 components to reach a median 0.61 of the true fraction
  without ever exceeding it. The section also documented a `light_ratio=` argument taking
  `Fixed(values)`, `Free(prior=...)` or `PerEpoch(...)`; none of those names exist, the D46
  façade review having rejected that one in favour of `light=` on the `Star`. What the two
  paths actually offer is now stated.

- **The marginal likelihood rejects an evaluation the arithmetic destroyed (D63).** One
  oracle-tier fit of the second Gaia benchmark diverged: ML-II was raising a component's
  smoothness precision toward its interior optimum when a line search accepted a trial at
  tau of 2e19, where the assembled posterior precision has a Cholesky pivot of 5e9 against
  a data scale of 1e4, the forward substitution loses every digit, and the quadratic form
  came out above the data term, a negative chi-square reported as a 216,000-nat
  improvement; the period then walked 228 prior sigma and the residual z-score rms ended at
  45. `marginal_loglikelihood` now returns minus infinity when the chi-square is negative
  (the form is positive definite by Woodbury, so a negative value is a destroyed
  evaluation; a healthy value and its gradient are bit-identical to before, a rejected
  point has a zero gradient), and `MarginalOrbitModel` carries a `smoothness_bound` factor
  beside its AR(1) and LSF bounds rejecting `log_tau - log_eta` above 30, the ratio at
  which the prior's own factorization is documented to round its pivot to zero (defaults
  start six hyperprior sigma inside it). The same fit now ends at z-score rms 0.97 with the
  period and both semi-amplitudes within 0.1 percent of the truth, the potential falling at
  every step; the sign guard alone does it, the bound never fires there. Traced in
  `internal/research/2026-09-09-gaia-rvs-benchmark/`. Not the AR(1) noise model's doing:
  at correlations of 0.0, 0.1 and 0.5 the same fit converged, only the trajectory differed.
- **The correlation stage no longer writes a velocity it did not measure (D63).** An
  archived oracle-tier star whose disentangling had diverged came out with fifteen rows of
  plausible velocities pinned at one value: the label fit had put the primary's zero point
  on the bound of its frame-offset scan, the shared search window built from the fitted
  Keplerian then could not contain any of that component's true velocities once composed
  with that zero point, the coarse pass sat on the top node of the range in every epoch,
  and the fine pass walked its window out of the range four times and reported a shift
  five pixels beyond the last point at which the chi-square had been evaluated (the update
  of the window start was applied after the last evaluation and added to a position from
  the previous window). `todcor` now reports the position it evaluated, clamps the fine
  window inside the requested range, flags `at_edge` when the fine minimum is still on the
  window boundary after the last attempt, and writes `nan` for the velocity and error of a
  component it could not measure while keeping every diagnostic of the point it did
  evaluate; the summary counts such epochs as "at the search edge, not measured".
  `Fit.measure_velocities` builds one window per template, each offset by that template's
  zero point relative to the median, so every component searches the same interval of
  reported velocity, and refuses by name a zero point that puts a component's window off
  its own fitted velocities (the disowned zero point here was 247 km/s from the other
  component's on a fit where both are one systemic velocity), naming the two ways out.
- **The pipeline stops at a diverged fit, refuses a disowned zero point, and marks a
  failed table (D63).** The same archived star had passed three gates it should not
  have. `Analysis.z_rms_max` (default 10; healthy fits sit near 1, a fit at a wrong period
  on the blind route near 2 to 3, the one divergence seen at 45) now stops a star after
  the disentangling with a flag and a recorded failure, the label and velocity stages not
  run, the template table already on disk. A label fit that beats neither null, or whose
  frame offset sits within one trial step of its scan's bound, or whose components' offsets
  differ by more than the velocity budget on a Keplerian fit (one systemic velocity by
  construction), no longer pins the templates' zero points: the table stays differential
  with one gamma per component, and `velocities.zero_points` records what was adopted or
  refused and why. A table whose median R-squared is negative (the templates fit worse than
  no template) or with no usable epoch is `velocities.status = "failed"`, its orbit stage
  skipped and its `velocities.rv` headed by a `FAILED` line; the "weakly detected" guard
  stays. And a declared velocity table (`velocities = "file"`) with a non-finite entry for a
  matched epoch is refused by component and epoch, the free fit having no site for an
  unmeasured epoch.
  Zero points that leave no search window holding every component (a label fit of a
  broad-lined contact pair put them 95 km/s apart) are dropped with a flag and the
  velocities measured differential, rather than the star failing at the refusal.
- **The velocity table stops losing a component to the exchange step, to shrunk templates
  and to an unlearned zero point (D63).** Three failures of the third run's tables, each
  traced on the archived products with the fit's own templates. The exchange of two
  components by the orbit assumed alike spectra at alike light fractions; on a 95/5 pair it
  swapped 19 of 80 epochs on noise draws of the faint component's velocity and moved the
  primary's semi-amplitude from 5 to 56 percent off, so the exchange now runs only when the
  two fractions are within a factor 3 of each other, and the table as measured is kept
  beside the delivered one (`velocities_unexchanged.rv`) whenever an epoch was exchanged.
  The correlation held the light at the declared fractions on the argument that the
  disentangling recovers each component at that amplitude; once ML-II has raised a
  component's smoothness precision by an order of magnitude the recovered lines are
  shallower (a secondary lost a quarter of its depth as tau went from 400 to 2000 and the
  table lost it at -89 percent; a free amplitude brought it back to -25), so the amplitudes
  are fitted freely when any smoothness precision moved more than a factor 10 from its
  start, with a flag saying the light column is then a template scale and not a fraction.
  And a label fit that learned nothing about a component's frame offset (posterior as wide
  as the prior, the component mostly noise) no longer pins that component's zero point (a
  secondary keeping 11 percent of its equivalent width had every velocity 46 km/s off):
  that component's velocities stay differential with their own systemic velocity, the
  other's remain absolute.
- **The conjunction window is centred on the start the fit carries (D63).** One field
  star failed twice at the optimizer's initialisation: the semi-amplitude scan had moved
  the conjunction half a period on (its phase grid holds the antipode the 41-point phase
  scan cannot sample, and there it was 672 nats better), the second scan pass gained
  exactly nothing so the final phase scan did not run, and the one-period window was
  still centred on the phase scan's best, which put the start on the window's edge,
  where the unconstraining transform is infinite. The window of the fit, and the
  sampler's, are now centred on the conjunction they start from.
- The pipeline's label and bootstrap stages sliced the library in its own wavelength scale
  before converting to the dataset's, which on a vacuum dataset lost about 2.3 A of coverage
  at each edge and failed the label fit; the library is now converted first.
- The BOSZ 2024 citation is A&A 688, A197, not A171.

- **One command from a list of stars to spectra, labels, velocities and orbits:
  `albireo.pipeline` and the `albireo` command line (D58).** `albireo init` writes an
  annotated TOML, `albireo run config.toml --jobs 4` runs every star in it, and `albireo demo`
  runs the same pipeline on two simulated stars with known answers, offline. For each star the
  driver reads the epochs, disentangles them (`Disentangler`), fits Teff / log g / [M/H] /
  *v* sin *i* to the components against a synthetic grid (`match_labels`), measures one velocity
  per component per epoch against those components (`todcor`), fits the Keplerian to the table
  (`fit_rv_orbit`), and writes `summary.txt`, `result.json`, the velocity table as commented
  ASCII and CSV, the component spectra with their bands, the orbit, the labels, `fit.npz`, and
  six diagnostic figures; a batch adds `results.csv` (one row per star), `results.json`,
  `summary.txt` and `failures.txt`. New names: `PipelineConfig`, `StarConfig`,
  `ComponentConfig`, `Analysis`, `StarResult`, `PipelineRun`, `run_pipeline`, `run_star`,
  `load_config`, `config_from_dict`, `write_config_template`, `demo_config`; `plot_phase_scan`;
  `albireo.simulate.synthetic_library` and `library_component` (the toy grid the demo and the
  tests are built on); `albireo.io.read_raw_spectra` and `dataset_from_raw` (the two halves of
  `read_dataset`, so a caller can read the header's resolving power before building the
  epochs). A console script `albireo` (`[project.scripts]`) and `python -m albireo`.
  Zero new dependencies: the configuration is TOML, read with the standard library.

  Three routes into the orbit: a period prior (the Keplerian is inferred from the spectra),
  `period = "search"` (library templates measure a first table, a periodogram finds the period,
  an orbit fitted to the table warm-starts the disentangling through `RVOrbit.to_theta()`), or
  a file of measured velocities (the free per-epoch table). Stars run in a spawn-based process
  pool, measured at 2.0× for four workers and 2.5× for eight on a batch of eight simulated
  stars, sub-linear because one star already keeps several cores busy, with each worker's XLA
  and BLAS threads capped at `cpu_count // jobs` as a precaution (the cap made no measurable
  difference on that benchmark); a failed star is recorded and the batch continues. Every
  caveat a run records is a flag on the report: a noise model that does not describe the
  data, a skipped stage and why, a table whose semi-amplitudes disagree with the disentangling.
  The two rules the stages enforce are enforced unchanged: light fractions are declared,
  never defaulted, and the wavelength medium is declared before a synthetic grid is consulted.

  Two things the driver found. Components must be declared in order of decreasing mass,
  and the fit is started with K₁ < K₂: a symmetric semi-amplitude prior gives the conjunction
  scan two equally deep troughs, the declared assignment and its mirror with the spectra
  swapped and rescaled by the light ratio, and the first demo run landed in the mirror. The
  convention is stated in the template, and the label stage checks it (a fitted light fraction
  far from the declared one is flagged). And `Disentangler`'s `_vector_spec` dropped
  every entry's `start_at` when a semi-amplitude was declared per component, so every star
  started at its midpoint, which is exactly the symmetric configuration; fixed, with a
  regression test.
  Docs: `docs/api/pipeline.md`, `docs/tutorials/pipeline.md`, `examples/13_pipeline.py`,
  `scripts/pipeline_bench.py`, benchmarks.md "D58".

- **Epoch radial velocities for every component by N-dimensional correlation:
  `albireo.todcor` (D56).** The two-dimensional correlation of Zucker & Mazeh (1994),
  generalized to any number of components and to weighted, masked, multi-instrument data by
  writing it as the weighted least-squares fit of shifted templates to the observed pixels,
  which is the model of `docs/math.md` §1.4 with the spectra given rather than marginalized. On
  a uniform grid with uniform weights it is TODCOR: the symmetric expression with the light
  ratio maximized out and the original fixed-ratio one are both pinned to 1e-10 against a NumPy
  transcription. New names: `Template` (from a flux array, a library at given labels, or a
  label match), `todcor`, `todcor_batch`, `todcor_surface`, `VelocityTable`, `TodcorBatch`,
  `TodcorSurface`, plus `plot_todcor_surface` and `plot_velocity_table`. Zero new dependencies.

  What it does that a port would not. The shifted, LSF-convolved templates are projected onto
  each epoch's own pixels (data are never resampled, D4), so masks, chip gaps, cosmic rays,
  per-pixel weights and mixed samplings enter through the weights and change no formula. The
  sub-pixel minimum is computed rather than interpolated: the shift operator is linear in the
  template, so the chi-square is an exact quadratic inside each pixel cell, reconstructed from
  3^N exact evaluations and minimized in closed form. The same fact bounds the operator's
  pixel-locking ripple at ~0.1/σ_px² pixels, which sets the sampling rule (three pixels per LSF
  sigma) and the warning below two. Errors are Zucker's (2003) maximum-likelihood ones, the
  curvature rescaled by the reduced chi-square, beside the trusted-weights version; the
  covariance's off-diagonal flags blended epochs; a per-component Δχ² says whether the epoch
  detects each star; a minimum at the search edge is flagged. Light fractions default
  to `"global"` (free per epoch, then the weighted median over well-detected unblended epochs
  held), with `"free"`, fixed fractions, and the classic free-scale form available.

  The zero point is stated on every table: a synthetic template is absolute, a disentangled
  component's rest frame is not identified (§5.3), and the velocities measured against it
  are reported as differential rather than offset without notice.

- **Orbits from velocity tables: `albireo.rvorbit` (D57), and the loop closed through the
  façade.** `fit_rv_orbit` fits the Keplerian by weighted least squares with the JAX Jacobian,
  using the same Kepler solver and angle conventions as the joint model, with one systemic
  velocity per component whenever a component is differential (a shared γ across two zero
  points is absorbed into both K's, which the test forces and checks); `find_period` runs a
  Lomb-Scargle search on the difference of the first two components and returns the aliases;
  `RVOrbit.to_theta()` hands the elements back as a disentangling warm start. `Fit.templates()`
  turns a fit's stellar components into templates (upsampled to three pixels per LSF sigma,
  intrinsic, zero point declared unknown) and `Fit.measure_velocities()` runs the epochs
  against them with the declared fractions and the declared LSF per instrument. Docs:
  `docs/math.md` §10, `docs/api/todcor.md`, `docs/api/rvorbit.md`, `docs/tutorials/todcor.md`,
  `examples/12_todcor.py`, `scripts/todcor_bench.py`, benchmarks.md "D56".

- **AI Phoenicis validates the label mode on a real star, and corrected it (D55).**
  `scripts/download_aiphe.py` fetches the 36 archival HARPS spectra;
  `scripts/aiphe_labels_bench.py` scores a label fit against them; and
  `docs/tutorials/aiphe-labels.ipynb` is that run, executed, for readers who will not
  download 840 MB to see it. AI Phe is the target because every quantity the mode produces
  has an independent published value, including a radius ratio measured photometrically that
  the fit is never told. The primary comes back at +0.52% in Teff, inside the documented
  2–3% target; the secondary at +4.33% does not, and is recorded as a miss.

  `LabelMatch.radius_ratio` is new, because on an eclipsing system the shared dilution scalar
  is a measurement with an external answer rather than an implementation detail.

- **`match_labels(compare=...)` now defaults to `"native"`, not `"matched"` (D55).** Convolving
  both sides with the LSF correlates the residuals over the kernel width while the likelihood
  stays diagonal, so χ² is over-counted by `1/Σk²`, predicted 4.91 on this dataset and measured
  4.26, and *v* sin *i* absorbs the mis-specification rather than the χ² doing so visibly. On
  AI Phe `matched` pinned both components to the floor of their *v* sin *i* prior where
  `native` returns a physical 2.2 km/s. `matched` remains available and is the right choice
  only alongside a residual-covariance model. The closed-loop test could not have caught this:
  its rows never pass through an LSF or a disentangling.

- **Named synthetic grids, downloaded and cached: `fetch_library` (D52).** The label mode
  shipped able to fit any library the user could construct, and with no way to obtain one.
  Three named grids are now registered: `bosz2024-fgk-r20000` (BOSZ 2024 MARCS, Teff
  4000–7000 K in 250 K steps, log g 3–5, [M/H] −1→+0.5; 455 nodes over 4000–7000 Å),
  `bosz2024-fgk-rvs` (the same nodes in the Gaia RVS window), and `pollux-ob-smc24` (POLLUX
  CMFGEN, SMC metallicity). New names: `fetch_library`, `library_names`, `library_info`,
  `clear_library_cache`, `ingest_bosz`, `ingest_pollux`, `save_library`, `load_library`. Still
  no new dependencies: both grids are plain ASCII, so nothing here needs astropy.

  BOSZ builds automatically because its URLs on MAST are deterministic; the shards download in
  parallel, the raw files are kept so re-cutting another band costs nothing, and the result is
  sliced to the registered band. POLLUX serves its collections through a form that posts to
  `/download/`, so `ingest_pollux` explains the manual step and stops rather than shipping a
  parser written against a file format nobody has inspected.

  Three details of the archive were checked against it rather than read off the documentation,
  and two were not what a careful reading gives: Teff is not zero-padded in a BOSZ filename
  (`t6000`, not `t06000`), and the atmosphere code changes across the grid, being MARCS
  spherical below log g 3.5, plane-parallel at and above it, and ATLAS9 above 8000 K. Both are
  pinned by tests. The third is the medium: a build measures it with `line_core_medium` and
  refuses to reconcile a disagreement with the registry, which on the real grid returns air at
  a ratio of 256 to 1.

  A cached build is verified on every load against a digest taken over the arrays rather than
  the file, so it survives a save/load round trip and is reproducible across machines. The build
  path reads back what it wrote, so a warm cache and a cold one return bit-identical arrays
  instead of differing in the last digits by whether the fluxes had been through float32 yet.

- **Stellar labels for template selection: `albireo.match` and `albireo.library` (D52, D53).**
  A new optional mode that fits Teff, log g, [M/H] and *v* sin *i* to disentangled component
  spectra against published synthetic grids, so a component can be rendered as a template for
  measuring epoch radial velocities elsewhere. New names: `SpectralLibrary`,
  `library_interpolator`, `crossval_library`, `line_core_medium`, `BoxInterpolator`,
  `SimplexInterpolator`, `match_labels`, `refit_draws`, `LabelMatch`, `StarLabels`,
  `RadiusRatio`, `ScalarDilution`, `FixedDilution`, plus `Fit.match_labels(...)` on the façade
  and `rotational_kernel` / `rotational_kernel_traced` / `rotational_radius_for` in
  `albireo.operators`. Zero new dependencies.

  The scope is narrow by design and `internal/roadmap.md`'s non-goal is amended rather than
  widened without notice: this synthesizes no spectrum, carries no line list, solves no
  radiative transfer and fits no abundances; `albireo.handoff` remains the route to GSSP,
  iSpec, Korg.jl and PySME. What it is for is the front-half job: choosing the right template,
  pinning the per-component velocity zero point, and checking an assumed flux ratio.

  Three things it does that a naive version gets wrong. Dilution is fitted jointly, through
  one shared radius ratio with wavelength-dependent light fractions written as a softmax over
  the grids' own continua, so they sum to one at every pixel by construction (GSSP's
  `gssp_binary` parameterization); an assumed light ratio that was wrong comes back as
  dilution instead of as a temperature error, and the spectroscopic light ratio becomes a
  result. The nuisance is additive, because the `k = 0` null space of `docs/math.md` §5.1
  lives in the continuum where a multiplicative polynomial is identically zero; its zeroth term
  is the unconstrained zero point, fitted and reported. Uncertainties are quoted twice, the
  Laplace curvature beside the spread from refitting joint posterior draws, because formal
  errors on correlated residuals run five to ten times optimistic, and `summary()` prints both
  with the ratio.

  Interpolation is in flux, never in model atmospheres (0.031% for a cubic against 0.19% for
  atmospheres on BOSZ's own spacing), with Catmull-Rom on a complete axis product and
  barycentric interpolation over a Delaunay triangulation where physics has cut the grid's
  corners off. Both reproduce a node bit-for-bit. Whether a learned emulator would beat this on
  a given grid is a measurement rather than an argument, and `crossval_library` is the
  measurement.

  `SpectralLibrary.medium` is required and never defaulted: air and vacuum differ by ~83
  km/s, and the upstream documentation is not a safe source, since BOSZ 2017 was vacuum
  throughout and BOSZ 2024 is air above 200 nm, under the same name. `line_core_medium`
  measures the convention from the spectra instead.

  Docs: `docs/math.md` §9 (the forward model, an extension of the §5.4 degeneracy ledger, the
  uncertainty story, and the accuracy target with citations), `docs/api/library.md`,
  `docs/api/match.md`, `docs/tutorials/labels.md`, and `examples/11_labels.py`, an offline
  closed loop that hands the fit deliberately wrong light fractions and asserts they come back
  as dilution.

- **`docs/tutorials/showcase.ipynb`, an executed notebook touring every headline output**
  on the packaged example: `explain()`, the fit summary, the component spectra with their
  uncertainty band, the residual diagnostics, the NUTS posterior (RV-curve draws and a
  corner plot), spectra drawn from the joint posterior, and a sensitivity forecast with
  six planned epochs against the twelve in hand. Rendered into the docs site by
  `mkdocs-jupyter`, which had been a declared docs dependency since the D39 block without
  ever being configured; `execute: false`, so the committed outputs are what the site
  shows and the docs build stays cheap and offline. `scripts/build_showcase_notebook.py`
  regenerates it (a release-time step, ~10 minutes of NUTS), strips kernel-environment
  noise from the outputs, and palette-quantizes the figures: 884 → 373 KiB, under the
  500 kB pre-commit file-size limit.
- **A feature-level comparison against the incumbent's repository** in
  [`docs/benchmarks.md`](docs/benchmarks.md) ("The incumbent's repository, feature for
  feature"): `TomerShenar/Disentangling_Shift_And_Add` examined as software, since the
  three-way table already compares the algorithms, covering the orbit treatment (a χ²
  grid over semi-amplitudes against joint inference), uncertainty (χ² contours on K, and
  nothing on the spectra, against posteriors on both), SB3 support, masking (in the
  published method, so not claimed as an albireo advantage), and
  distribution (unlicensed research scripts against a BSD-3 package). Written from the
  repository's README and GitHub API metadata only; the source was never opened, so the
  clean-room provenance of `scripts/shift_and_add.py` survives the comparison.
### Fixed

- **`docs/quickstart.md` overlaid the injected truth on the wrong wavelength grid.** A
  simulation's truth is stored on the grid it was generated on; the model is solved on a
  grid that is widened by the velocity budget and the LSF radius and takes its sampling from
  the data. On the packaged example those are 663 and 1074 pixels, so the quickstart's final
  plotting call raised from inside matplotlib. The doc now resamples first, as
  `scripts/build_showcase_notebook.py` already did, and `plot_spectra` validates `truth`
  against the grid the way it already validated the mean, so the error names the cause and
  gives the one-line fix instead of reporting a shape mismatch. Had the two grids happened to
  agree in length, the old code would have plotted the truth against the wrong wavelengths
  without any symptom, which is the failure worth guarding against.

- `example_info("sb2_sim")` described the packaged example as a "circular SB2". It is
  eccentric by design, e = 0.15, so that the first thing a new user runs never starts at the
  `(sqrt(e) cos w, sqrt(e) sin w)` singularity, and the generator script says so in a comment
  while the registry description contradicted it. The description now states e = 0.15.

### Changed

- **Gradients are ~2x faster, and the answers are bit-identical.** Two exact changes in
  `albireo.assembly`, both found by re-profiling: D28's recorded attribution ("92% comb
  probing") had been stale since D28 itself removed probing from the hot path. The measured
  split is 82% of a gradient in the assembly, whose backward cost 3.3x its own forward.
  1. `_band_accumulate`, a `custom_vjp` for the per-epoch band update. The forward is
     `band + place(f)`, the identity in `band`, but reverse mode transposes the
     `dynamic_update_slice` and the `dynamic_slice` separately and rebuilds that identity
     out of three passes over the whole band tensor, once per component pair per epoch:
     313 GB of memory traffic to reproduce an input, at the benchmark ladder's first row.
     The closed form (`band_bar = out_bar`, `f_bar = ds(out_bar, idx)`) costs nothing.
  2. The second kernel application in `G = K^T H K` translates columns only, since unlike
     the first it has no row shift, so it is one contraction against a static banded matrix
     instead of `2r+1` read-modify-write passes over the widest image in the assembly.
     18x on that stage, and no extra memory: contracting both applications at once would
     need a 1.9 GB neighbourhood stack, which is the intermediate D29 spent a pass removing.

  Measured at 31,734 model px, SB2, 50 epochs, p = 513: evaluation 2.97 → 2.20 s, gradient
  10.23 → 5.19 s. The log-likelihood and its gradient are bit-identical before and after
  (compared as IEEE-754 hex) for stationary, AR(1) and wavelength-dependent-LSF problems;
  the Hessian moves 4e-13 relative. Four other candidates, a blocked Cholesky, j-factoring
  the T-sandwich, `remat=False`, and the custom-VJP band-to-block packing named on D28's own
  ledger, were measured and rejected; see
  [`docs/benchmarks.md`](docs/benchmarks.md) "D49 speedup pass" for the numbers, including
  why XLA's fp64 `cholesky` at n = 513 running at 13 GFLOP/s against `matmul`'s 249 is not
  fixable by blocking.

  What it cost. `custom_vjp` rejects `jax.jvp`, and `forecast._effective_parameters`
  was the package's only forward-mode site (D47 gets `p_eff` from one directional
  derivative of `log det` in the noise scale). Both `t` and the log-determinant are
  scalars, so it is now a `jax.grad` returning the bit-identical number, 0.283 s to
  0.532 s once per forecast, against 1.8x on a gradient run ~2,600 times per posterior.
  albireo therefore no longer has a forward-mode path through the marginal likelihood;
  D28 had already removed it one stage later at `_solve_stage`, and second derivatives
  remain available through reverse-over-reverse.

- **The three-way comparison was re-run on one machine, and `scripts/fd3_bench.py` now
  times shift-and-add in a fresh process**, because the old convention measured the heap
  rather than the code. Timed in-process after the XLA solve (the convention behind both
  previously recorded walls), the identical call reads 40–80% slower: XLA's allocation storm
  leaves the Windows CRT heap serving shift-and-add's ~35 KB temporaries through microsecond
  free-list walks, with allocating ufuncs ~4× slower and their `out=` twins flat. The re-run
  reproduced all twelve recorded RMS values exactly and re-measured the walls under one
  protocol on a recorded stack: shift-and-add 0.026 s, albireo 0.059 s, fd3 0.064 s
  single-threaded (0.104 s as shipped, its OpenBLAS spinning 32 threads). See
  [`docs/benchmarks.md`](docs/benchmarks.md) "D50 re-run".

### Fixed

- **A detector gap is no longer weighted like data.** `albireo.mask_flux_gaps` zero-weights
  contiguous runs of non-positive flux and warns with the wavelength range; `to_epoch` calls
  it before the spike clip, since a flat run of zeros has no local scatter for a running
  median to catch.
  Found on real HARPS spectra of AI Phoenicis, where the two CCDs leave 32.9 Å of exact
  zeros at 5304.67–5337.61 Å. Nothing marked them: the pixels are finite, there is no
  quality column, and because HARPS ships no error array the inverse variance was estimated
  from the local scatter, which across a flat run of zeros is small. They arrived with
  median ivar 6398 against 6231 for real pixels, and an analysis window that was 33% detector
  gap disentangled to component spectra with negative flux.
  The rule is about runs rather than about any non-positive pixel:
  `RawSpectrum.bad_pixels` declines to treat zero flux as missing and is right to, because one
  zero can be a saturated core or a clipped cosmic ray. Eight in a row cannot be. Same shape
  as D45's zero-errors-read-as-infinite-precision, one level up, and the same principle: the
  reader may decline to answer, but it may not guess.

### Added

- **`albireo.handoff`: the files the atmosphere codes read, and the draws that carry the
  uncertainty into them.** `write_gssp`, `write_ispec` and `export_draws`, with
  [a tutorial](docs/tutorials/downstream.md) and `examples/10_downstream.py`.
  The formats are the hard part and both traps fail without a symptom. iSpec does no unit
  conversion on its text path: its whole internal scale, atomic line lists included, is
  nanometres, so an ångström value lands a factor of ten outside every model grid and still
  fits something. And GSSP infers its synthetic step from the file it is handed ("the step
  width in wavelength that will be used for the calculation of synthetic spectra is computed
  from the observations"), so a log-wavelength grid must be resampled onto an equidistant one
  rather than written directly. Both are regression-tested.
  GSSP has no per-pixel error column at all, with no error path, no S/N entry and no weighting
  entry anywhere in its configuration, so the posterior band cannot reach an effective
  temperature through the file. It can only get there by fitting *N* spectra, which is what
  `export_draws` is for: `draw_spectra` returns `d_hat + L^-T z` on the vector stacked over
  all components, so draws are correlated across wavelength and across the two stars, and
  draw *i* of component A is the same posterior sample as draw *i* of component B.
  That jointness is the reason for the feature, and it is measured rather than asserted.
  Against the established recipe, independent per-pixel noise at the band's amplitude (Kiran
  et al. 2016, §3.5), the joint draws give an equivalent-width spread 1.80× and 3.38× larger
  on the packaged example's two components: white noise understates any integrated quantity,
  and every atmospheric parameter integrates the spectrum. The sharper result was unlooked-for:
  the two components' equivalent widths are correlated at −0.992 across draws, against
  −0.052 under independent noise, which is D47's *k* = 0 exchange mode arriving in a derived
  quantity, so the two stars' difference is far better determined than either alone and
  independent error bars misstate both. See `internal/roadmap.md` Tier 2 item 8 and
  `docs/api/handoff.md`.

- **`albireo.sensitivity_forecast`, which answers an observing question before the
  observation**: "will twelve more epochs at these phases separate the two stars?"
  `albireo.plan_epochs(template, bjd=...)` builds the epochs of a night that has not
  happened, `sensitivity_forecast(grid, design, orbit=..., baseline=...)` returns what
  they would buy, and `albireo.plot_forecast` draws it. This works because the posterior
  covariance of the component spectra, `(Lambda_p + A^T W A)^-1`, contains no flux:
  fluxes reach the marginal likelihood only through terms that move the posterior mean and
  the evidence, never the covariance. The claim is structural rather than asserted: the
  precision is assembled directly and the right-hand side is never formed, and a regression
  test replaces every flux with noise a hundred times the continuum and requires the forecast
  back bit-identical.
  Three summaries, each exact and each quoted against the same quantity under the prior
  alone, so a design that is learning nothing says so: the pointwise band, the
  worst-determined modes of the covariance (the spectral patterns the design cannot
  pin down, by subspace iteration on the banded factor), and `p_eff`, the spectral
  degrees of freedom the data would constrain, from one directional derivative
  of `log det` in the noise scale rather than a stochastic trace estimator. Whole designs
  are ranked by the expected information gain, `0.5 (logdet Lambda_t - logdet Lambda_p)`.
  The modes are taken over the pixels the design actually weights, because a model grid is
  deliberately wider than its data and those margin pixels, prior-only by construction,
  are otherwise the worst-determined direction of every real problem.
  It does not forecast the orbit: the Fisher information for a velocity
  runs through the derivative of the component spectrum, so an error bar on `K_2` needs the
  line depths, which is what has not been measured yet.
  One measured result corrects `docs/math.md` §5.1's own reading: the RMS differential
  velocity is the small-*k* expansion rather than the objective. A cadence aliased to the
  period maximizes it and is the worst of three plans: twelve nights at *P*/2 hold the RMS at
  117.8 km/s, stay blind over 58% of the scale range and are worth 243 nats, while the same
  twelve spread over phase lower the RMS to 99.3 km/s and 33% and are worth 375
  (`examples/08_forecast.py`). See `internal/design.md` D47, `docs/math.md` §5.5 and
  `docs/api/forecast.md`.
- **`Disentangler(velocities=...)`: declare measured velocities when there is no
  orbit to declare.** Exactly one of `orbit=` and `velocities=` is now required. With
  `velocities=` (an `(n_stellar, n_epochs)` km/s table from cross-correlation, shift-and-add,
  or line splitting measured by hand) no orbital sites are sampled at all, and `fit()`
  returns the free per-epoch RV table directly.
  This closes a circle the façade could not previously escape: the table needs a warm start,
  a cold one is 122,000 nats worse (D42), and the only warm start on offer was
  `Fit.free_velocities()`, which needs a Keplerian fit, which needs a period. For an
  unsolved system, the table is what produces the period.
  The declared velocities are a starting point rather than a constraint: the per-component
  zero point stays unidentified, so a systemic offset changes neither the answer nor the
  solver's bandwidth, and the velocity budget is derived from the centred table, itemized in
  `explain()`. The mode's one failure is refused rather than discovered: a declaration whose
  components never separate raises, and one that never resolves them beyond the LSF width
  warns. `scan()` and `detection_limit()` refuse without an orbit.
  Measured: warm-started from velocities carrying 3 km/s of scatter and a 150 km/s systemic
  offset, the recovered table lands at 0.096 / 0.070 km/s, the same as D42's
  Keplerian-warm-started 0.098 / 0.066. See `internal/design.md` D48.
- **A tutorial that takes a BLOeM SB2 from a survey identifier to disentangled spectra**
  (`docs/tutorials/bloem-sb2.md`), the second gap D45 left open. It covers the ordering the
  survey forces and that none of the other tutorials needs: with no published period there
  is nothing to warm-start a Keplerian from, so the free RV table comes first and supplies
  the periodogram. Writing it found an error in `examples/06_bloem.py`: its window was
  documented as sitting "between Hδ and Hγ without either core" and did not, since 4000–4300 Å
  contains Hδ at 4101.7, and `nebular_windows` places a ±300 km/s window at 4099.7–4107.9
  inside it, so the script's stated reason for not modelling the nebula was false. The window
  is now 4120–4300 Å, which contains no nebular line at all.
- **A worked example for the free per-epoch RV table** (`examples/09_rv_table.py`), the one
  gap D42 left open. It is built around the two properties of the mode that are
  counter-intuitive rather than around the API: it shifts one star's velocities by 50 km/s
  and shows the log-likelihood does not move (exactly 0 nats for the relativistic shift
  against 8.7e-6 for the ordinary one, which is only its first-order approximation, and is
  the reason the centering lives in pixel space), and it prints the raw Laplace error bars
  beside the projected ones, 37.947 km/s on every entry because that is `120/√10`, the
  prior, against a measured 0.056–0.065. It also shows the mode's failure rather than
  describing it: a cold start lands 122,000 nats worse than the warm one.
- **`albireo.Disentangler`, a declarative front end**, marked experimental because a
  vocabulary is expensive to change once people depend on it. Declare the system
  (`Star(name, light=...)`, `Telluric()`, `Nebular()`, `Orbit(period=..., k=...)`,
  `LSF.from_resolution(R)`) and it emits the expert path. Twelve lines against
  fifty-nine on the packaged example, recovering the same answer.
  It is a compiler rather than a shortcut: `dis.explain()` prints every derivation and
  `dis.expert()` returns the exact `(model, priors, init)` triple, so the low-level API
  stays the supported surface and dropping down costs three lines.
  Four things are derived rather than typed: the solver's velocity budget (from the
  `k` priors' own support, since it must bound what the prior allows rather than what the
  answer turns out to be), the grid margin, the conjunction phase (by a scan, because the
  likelihood is sharply multimodal in phase), and the smoothness hyperparameters by
  empirical Bayes, reported per component with a flag on any that did not move from its
  start. A fifth is structural: a spec such as `Between(5.5, 6.5)` carries both its prior
  and its starting value, so `priors` and `init` cannot drift apart.
  It refuses to derive five things where a default would be a scientific claim: light
  fractions (required per star, must sum to 1, and repeated in an `Assumed, not measured`
  block on every summary), a period search, an undeclared air/vacuum scale when a nebular
  or telluric component makes it an 83 km/s question, a velocity budget smaller than the
  priors reach, and the `e = 0` singularity, where a free eccentricity never starts at the
  origin and `ecc=Fixed(0.0)` is exact, because those sites are not sampled at all.
  Not in v1, by decision: jitter, AR(1), inferred light fractions and inferred LSF widths.
  Also absent because they could not be delivered reliably, each raising rather than
  approximating: hierarchical triples (`Orbit(outer=...)`), Gauss-Hermite `h3` (it reaches
  the kernel through `build_problem`, not through the model the façade builds), and a
  lower bound on eccentricity (an annulus in the sampled parameterization, not a box).
  See `internal/design.md` D46 and `docs/api/facade.md`.
- **BLOeM targets resolve by name**: `albireo.resolve_bloem("1-002")`,
  `albireo.bloem_catalogue(binary_class="SB2")` (the 59 published double-lined systems),
  and `albireo.bloem_spectra(star)` for that star's epochs, ready for `download`. The
  archive does not know the survey's names: BLOeM spectra sit under
  `obs_collection='GIRAFFE'` with `target_name` set to the Gaia DR3 source id, so the
  cross-match is fetched from VizieR, which uses the same TAP dialect and so needs no new
  dependency.
  Gaia ids are kept as strings because 809 of the 929 do not survive a float64 round trip.
  Defaults to survey programme `112.25R7`: the same 929 stars are also observed by
  `115.28A9` at *R* = 17000 and 23000 in two other windows, and pooling those with LR02
  under one line-spread function would be wrong. See `internal/design.md` D45.
- **The FITS reader dispatches on IVOA utypes (`TUTYPn`), not column names**, so every ESO
  collection reads with the right column, the right unit and the right wavelength scale.
  Names remain a last-resort fallback for non-ESO files. Keying on UCDs instead would have
  been unsafe: UVES gives its sky-background column the same UCD HARPS gives its
  flux column. `RawSpectrum` gained `quality`, `specsys`, `v_bary_source`, `err_source`,
  `columns` and `bad_pixels`, so what the reader chose and why is inspectable.
- **Quality columns are honoured, and so is a zero uncertainty**: a flagged pixel, a
  non-finite flux, or a non-positive error now gets `ivar = 0` rather than full weight. A
  zero error is how these pipelines write "nothing here", not a measurement of infinite
  precision. Flagged pixels are also excluded from the continuum fit. A flag column whose
  convention cannot be read is ignored rather than guessed at: one in which zero never
  appears is not using "zero is good" (UVES_SQUAD's `STATUS` runs `{-5, 1}`, and taken at
  face value it condemns every pixel of all 467 products in that collection), and columns
  named only `MASK`/`FLAG` are not read at all, since those names carry no agreed polarity.
  Losing a mask is recoverable; inverting one keeps exactly the pixels the file rejected.
- `read_dataset(medium=...)`, matching the existing `frame=` override, and a refusal to
  combine files that disagree about air vs vacuum.
  combine files that disagree about air vs vacuum.
- **`albireo.archive`**, an ESO Science Archive client: `spectra_query` builds the
  ADQL, `query` runs it, `download` fetches resumably with a manifest. One query language
  for FEROS, HARPS, UVES, X-shooter, GIRAFFE and ESPRESSO. Stdlib only, so finding data
  costs no dependency. `scripts/download_hr6819.py` is now a thin wrapper over it.
  Two guards exist because the archive is silent where it matters: a query that hits
  `MAXREC` raises (ESO's JSON carries no overflow marker, so a truncated result is
  indistinguishable from a complete one), and a download whose byte count disagrees with
  `Content-Length` is rejected rather than left on disk looking complete.
- **Air vs vacuum is now a declared, validated property**: `EpochData(medium=...)`
  accepts `"air"`, `"vacuum"` or `None` (undeclared, the default and what every epoch
  built before this field existed means), and `Dataset` refuses a mixture rather than
  picking one. The offset is a nearly constant 83 km/s across the optical, the same
  order as the semi-amplitudes albireo measures, and it does not average out. Mixing
  declared with undeclared raises too: "unknown" cannot be checked against "air".
- `albireo.air_to_vacuum` / `albireo.vacuum_to_air`, the IAU-adopted Edlen (1966) /
  Birch & Downs (1994) refractivity, evaluated at the vacuum wavenumber so that converted
  line lists agree with published air values. The round trip closes to float64.
- `LogGrid` gained no new state: the conversions are free functions, because which scale
  a spectrum is on is a property of the observation rather than of the model grid.
- **A free per-epoch radial-velocity table**: theta site `velocity`
  `(n_stellar, n_epochs)`, which replaces the Keplerian rather than supplementing it
  (mixing the two raises). `albireo.relative_velocities` reports the identified table,
  `relative_velocity_errors` its per-epoch bars, and `keplerian_residuals` the
  model check the mode exists for: fit free velocities, then ask whether a Keplerian
  still threads them. The numpyro model records `velocity_rel` as a deterministic.
  Warm-started from a Keplerian 30% wrong in both semi-amplitudes, per-epoch RVs recover
  to 0.098 / 0.066 km/s, 1/60th of a model pixel, and the Wilson mass ratio to 0.4%.
  See `internal/design.md` D42; the mode needs a warm start, and says so.
- `albireo.forward.with_shifts`, the pixel-space core of `with_velocities`, which is now
  a wrapper over it. Pixel shifts are where the model's shift composition is exact, so
  anything that adds or centers shifts has to work there.
- `LogGrid.pixels_to_velocity`, the exact inverse of `velocity_to_pixels`.
- **`albireo.calibrate`**: `detection_limit` turns the K₂ scan's peak into a claim,
  with an empirical null distribution from companion-free trials, a completeness curve from
  a ladder of injected light fractions, and the sentence they add up to
  (`DetectionLimit.summary()`). `false_alarm_probability` never reports below
  `1 / (n_null + 1)`, and the threshold is defined through it, so the realized
  false-alarm rate over the null trials can never exceed the nominal one, where
  an interpolating sample quantile errs the other way. Trials resimulate through the
  observed data's own operators, so 450 full scans (9,450 marginal solves) take 53 s.
- **`MarginalOrbitModel.log_likelihood_sweep`**: a grid of trial θ values as one
  batched `lax.map` instead of a Python loop with a device synchronization per point.
  Measured 2.0-2.8x across problems from 201 to 2,652 model pixels, agreeing with
  the loop to 1e-12 relative or better. `k2_scan` uses it.
- **`k2_scan(k1_sigma=, k1_nodes=)`**: marginalize K₁ over a Gaussian prior with a
  Gauss-Hermite rule, applied to the companion and no-companion models so `D` stays a
  ratio of two marginal likelihoods. `K2ScanResult` gains the `(n_k1, n_k2)` surface,
  the quadrature, and `k1_peak`. `k1_sigma=None` is the previous behavior. This is the
  fix for the failure mode the literature reports: a K₁ 10% high took the recovered
  companion's line pattern from 0.96 correlation with truth to 0.49 while tripling
  `D`, and marginalizing restored it to 0.93. See `internal/design.md` D41.
- **`albireo.forward.with_data`** and **`albireo.simulate.resimulate`**: swap a
  problem's data term, and redraw it from the problem's own forward model. Together they
  are a parametric bootstrap that reuses the rebin operators, pair tables, masks and
  weights, which is what makes the calibration cost scan time rather than build time.
- `albireo.plot_detection_limit`: the null distribution with its calibrated threshold
  beside the completeness curve with its limit. An observed peak that falls off the
  histogram is annotated rather than drawn, since a real companion's `D` can sit orders
  of magnitude above the entire null distribution.
- `examples/05_detection_limit.py`: calibrate, detect, and read the peak against the
  null distribution, with the two-panel figure. Runs in CI.
- **A nebular component** (`build_problem(nebular=True)`, `with_nebular_amplitudes`, θ site
  `log_nebular_amp`): a component at rest in the barycentric frame with a free per-epoch
  amplitude, for the emission lines of the H II region a massive star sits in. Nebular flux
  is added on top of the stellar continuum rather than taken out of it, so the amplitude is
  outside the light-fraction simplex; the amplitude scale is pinned by centering the
  log-amplitudes (`inference.nebular_amplitudes`), and `nebular_v_kms` is a placement
  convention rather than a measurement. Measured in the closed loop: leaving a nebular line
  unmodelled leaves the H-beta core 26% too shallow and costs 11.5% of the equivalent
  width; with the component, 0.14%. The orbit is affected more than the spectra: a
  nebula-blind joint fit returns K_2 59% low, with the eccentricity of a circular orbit
  driven to the solver's clip. See `internal/design.md` D40.
- **Per-pixel prior strengths**: `SmoothnessPrior(tau_profile=..., eta_profile=...)`, with
  the inferred scalars kept separate so the ML-II fit is unchanged. `albireo.window_profile`,
  `albireo.nebular_windows` and `albireo.NEBULAR_LINES` build the profile that confines a
  nebular component to the lines it can have. Uniform profiles are bit-identical to
  the previous prior.
- `albireo.synthetic_nebular_spectrum` and `simulate_dataset(nebular=, nebular_amplitudes=,
  nebular_v_kms=)`, so the closed-loop tests inject through the same operator stack as
  everything else. `SimulationTruth` records what was injected.
- `k2_scan(nebular=...)`. A faint-companion scan is a matched filter for a stationary
  residual, which is what an unmodelled nebular line looks like.
- **`albireo.results`**: persisting and exporting fits. `save_fit` / `load_fit` round-trip
  `MAPResult`, `K2ScanResult` and `MarginalResult` through `.npz` with a JSON header;
  `to_inference_data` converts a NUTS run for arviz; `write_ascii` writes the disentangled
  spectra with no optional dependency at all.
- **`albireo.io.write_spectra`**: the disentangled spectra and their uncertainty band as
  FITS or ECSV, with the light fractions and prior hyperparameters recorded in the header
  (the recovered line depths are only interpretable next to them).
- **`albireo.plotting`**: `plot_rv_curve`, `plot_spectra`, `plot_detection`,
  `plot_residual_zscores`, `plot_phase_fold`, `plot_lsf`, `plot_light_fractions`,
  `plot_corner`. The first three were previously stranded inside `examples/`, which now
  call the module instead. Needs the new `plots` extra (matplotlib, arviz), lazily imported
  exactly as `albireo.io` imports astropy.
- **`albireo.examples`**: `load_example`, `example_info`, `example_names`,
  `clear_example_cache`. A simulated SB2 ships inside the wheel, so the quickstart runs
  offline, in CI, and in a fresh notebook with no download and no astropy. Larger examples
  download to a platform cache directory and are checksum-verified.
- `examples/00_quickstart.py` and `docs/quickstart.md`: load, fit, plot, export in about
  twenty seconds. Runs in CI.
- `data_residual_zscores(..., per_epoch=True)` returns one array per epoch, which is what
  makes the lag-1 autocorrelation diagnostic meaningful (it is only defined within a single
  exposure).
- `internal/roadmap.md`: where albireo is going, in what order, and the non-goals, with the
  ecosystem evidence behind the ordering.
- Rendered API reference (mkdocstrings) and a GitHub Pages deployment workflow. The
  `mkdocstrings` and `mkdocs-jupyter` dependencies were declared but had never been
  configured: `mkdocs.yml` had no `plugins:` block, so neither did anything.
- `py.typed`, so downstream type-checkers see the annotations the package already had.
- `tests/conftest.py` with `slow` / `network` / `gpu` markers and matching `--no-slow`,
  `--no-network`, `--no-gpu` opt-outs. A bare `pytest` still runs everything; CI's fast
  matrix deselects the gates (14 min → 6 min) and a separate job runs them with coverage.
- A `bare-install` CI job that installs with no extras and checks that `import albireo`
  works, which is what makes the optional-dependency guards real rather than intended.
- `CHANGELOG.md`, `docs/citing.md`, `internal/releasing.md`, `codemeta.json`, and a tag-driven
  release workflow using PyPI trusted publishing.

### Changed

- `K2ScanResult` gained `k1_grid`, `k1_log_weights`, `log_likelihood_grid`,
  `log_likelihood_null_grid` and `k1_peak`, all with defaults, and `save_fit` /
  `load_fit` round-trip them. A scan file written before the K₁ marginalization existed
  still loads; its `k1_peak` reads back as NaN.
- `k2_scan` is no longer bit-identical to its pre-vectorization self at fixed K₁:
  batching the trials into one `lax.map` re-associates the linear algebra, which moves
  the log-likelihoods by ~1e-9 out of ~1e5. That is floating point rather than method; the
  quadrature at `k1_sigma=None` is exactly the identity.
- `with_velocities` and `with_light_fractions` now carry the trailing non-stellar columns
  (telluric, nebular) through unchanged instead of rebuilding the telluric one. The result
  is identical, since those velocity laws depend only on the frame and each epoch's
  `v_bary`, both fixed at build time, and it is correct for any number of non-stellar
  components.
- `marginal_loglikelihood` names the components a mismatched prior is missing, and rejects a
  prior whose per-pixel profiles were built for a different grid.
- The package version is now single-sourced from `src/albireo/__init__.py` via
  `[tool.hatch.version]`; `pyproject.toml` declares it dynamic. `tests/test_metadata.py`
  asserts that the installed distribution metadata and `CITATION.cff` agree with it.
- `K2ScanResult.model` may now be None, so a scan result can be read back from disk without
  its `MarginalOrbitModel`. `k2_scan` already constructed the result by keyword, so this is
  a widening rather than a signature change.
- `MarginalResult.precision` may likewise be None, and `MarginalResult.chol` now raises a
  `ValueError` saying so, and saying where the numbers went, rather than failing inside
  the block Cholesky.
- Markdown is excluded from `ruff format`. Recent ruff formats Python code blocks inside
  `.md` files, which would collapse the deliberate comment alignment throughout the docs.

### Fixed

- **`examples/06_bloem.py` used a window it documented as avoiding.** Its region was
  described as sitting "between Hδ (4102) and Hγ (4340) without either core", but
  4000–4300 Å contains Hδ at 4101.7, and `nebular_windows` puts a ±300 km/s window at
  4099.7–4107.9 inside it. Since the script's stated reason for not modelling the nebula
  was that it avoided the Balmer cores, the claim was load-bearing and false. The region
  is now 4120–4300 Å (`nebular_windows` returns nothing there) and the docstring explains
  why the blue edge cannot move rather than asserting the outcome.
- **`EpochData.medium` never reached an epoch read from a file.** `to_epoch` built the
  `EpochData` without it, and `preprocess._replace`, which every trimming and masking
  helper goes through, omitted it, so even a hand-set value was dropped by the first
  `select_region`. D43's guard against combining air with vacuum therefore could not fire
  on the path that matters, and the offset it exists to catch is 83 km/s.
- `ESO TEL TARG ALPHA` / `DELTA` are packed sexagesimal (`HHMMSS.sss`, `±DDMMSS.sss`) and
  were read as degrees. `SkyCoord` accepts any real number as a right ascension, so the
  position wrapped modulo 360 to somewhere plausible and the barycentric corrections were
  wrong by minutes and by km/s, with no symptom. Only reached when `RA`/`DEC` are absent.
- `MJD-OBS + EXPTIME/2` was used as the mid-exposure time for coadded products, where it
  can be a week early. `MJD-END` is now preferred, and the `EXPTIME` fallback is refused
  when `TELAPSE` says the product spans more than one exposure. A product with
  `M_EPOCH=True` warns that it has no single epoch at all.
- A fabricated `v_bary = 0.0` (no keyword, too little header to compute one) now warns.
  Zero is a legitimate value here, which is why inventing it without a warning was wrong.
- An all-NaN `ERR` column, which FEROS and HARPS ship, now warns that the weights have
  become albireo's assumption rather than the archive's, instead of being dropped without
  notice.
- `read_spectrum` no longer requires a time keyword when `bjd=` is supplied, which is what
  the error message for a missing one already told the caller to do. Among other things this
  lets albireo read back the component spectra it writes.
- The spectrum HDU is chosen by utype and `EXTNAME` before column names, so a short
  calibration table earlier in the file can no longer win over the real spectrum.
- Dropped `"R"` from the resolving-power keywords: a one-character card collides with
  anything, and `R = 3.7` implied a 34,000 km/s line-spread function without complaint.
- `read_dataset` on a directory also picks up `.fit` and `.fits.gz`, and no longer passes
  `instrument=`/`frame=` twice when they appear in `read_kwargs`.
- A spectral axis declaring itself `em.freq` or `em.energy` is refused rather than read as
  Angstrom: `SpectralAxis` names those axes too, and the result would be a strictly
  increasing, entirely wrong wavelength grid.
- A table that declares itself the spectrum but has no identifiable flux column now says so,
  instead of falling through to the image reader and returning something else in the same
  file as the science spectrum.
- The all-bad-pixel guard is evaluated on the pixels that survive the final trim rather than
  on the wider slice used for the continuum fit; a dead region beside a live pad previously
  produced a zero-weight epoch that the solver accepts without complaint.
- An error column sharing neither the namespace nor the unit of the chosen flux is rejected
  with a warning: it is the error on the file's other flux column.
- `HELICORR` and `ESO QC VRAD HELICOR` are recognized as heliocentric. Neither spells out
  `HELIO`, so both were being trusted as barycentric without the warning.
- Passing `frame=` suppresses the warnings that exist only to advise passing `frame=`, and a
  `SPECSYS` albireo does not model (`LSRK`, `GEOCENT`) is now reported as such and preserved
  on `RawSpectrum.specsys`, rather than being erased and described as a missing keyword.
- A TAP service's ADQL error is reported with the message from the VOTable body it returns,
  instead of a bare `HTTP Error 400`.
- `pytest` (the console script) failed to collect `tests/test_ar1.py`,
  `tests/test_lsf_h3.py`, and `tests/test_lsf_varying.py`: they import shared helpers via
  `from tests.test_likelihood import ...`, which only resolves when the repository root is
  on `sys.path`. That happened under `python -m pytest` but not under `pytest`, the form
  CI runs. Declaring `pythonpath = ["."]` in the pytest configuration makes the two
  invocations equivalent. Since nothing had ever been pushed, CI had never run and this had
  never been observed.
- `ruff format --check .` did not pass on a clean checkout (`src/albireo/solver.py`), which
  would have failed the lint job on the first push.
- `mypy src/albireo` reported two errors on a clean checkout: `examples.load_example`
  passed a list where `Dataset` is annotated for a tuple, and `results.save_fit`'s
  `**arrays` splat is not expressible against numpy's `savez` stub. Both would likewise
  have failed the lint job on the first push. Same cause as the two defects D39 records:
  nothing has ever been pushed, so CI has never run.
- `MarginalOrbitModel` rebuilt the spectral prior from the sampled `log_tau`/`log_eta` and
  dropped any per-pixel profiles, so a window-confined component was un-confined without
  notice the moment ML-II was switched on. The profiles are structure and the
  scalars are hyperparameters; the merge now respects that. Found before the feature
  shipped, and measured: with the confinement applied, the closed-loop K_2 error
  goes from 2.6% to 0.29%.
