# Changelog

All notable changes to albireo are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and albireo uses
[semantic versioning](https://semver.org/spec/v2.0.0.html), with the caveat that while the
version is below 1.0 the public API may change in any release.

Below 1.0 that caveat applies to every module, but the modules are not equally settled.
The three tiers below state where changes are expected.

- **Supported.** This tier is the model layer: `data`, `grids`, `preprocess`, `forward`,
  `likelihood`, `priors`, `inference`, `kepler`, `operators`, `scan`, `calibrate`,
  `simulate`, `examples`, `io`, `results` and `plotting`. A change here is recorded below
  with the action it requires.
- **Experimental.** These modules are marked as such in their docstrings: `facade` (the
  `Disentangler` declaration), `pipeline` together with the `albireo` command line and the
  TOML schema it reads, `todcor`, `rvorbit`, `library`, `match`, `forecast`, `archive` and
  `handoff`. Their names and interfaces are not settled and may change in any release
  below 1.0. Each is a convenience over the model layer, where the same computation is
  available directly.
- **Internal.** `assembly` and `solver` have no stability guarantee.

Every name above is re-exported at the top level, so the marker is on the module, not on
the import path.

This file records what changed. The reasons are recorded elsewhere:
[`internal/design.md`](internal/design.md) §2 is the decision record, and
[`docs/benchmarks.md`](docs/benchmarks.md) is the validation and performance record.

## [Unreleased]

### Added

- **A step-by-step Gaia RVS notebook.** `docs/tutorials/gaia-rvs-benchmark.ipynb` renders two
  stars from the library, simulates the epochs Gaia would deliver, disentangles them, measures
  the epoch velocities and the labels, draws a population and runs a three-system benchmark,
  with the outputs committed. Its report section tabulates the key numbers of that run from
  `summary.json`, and a closing section summarises the recorded full runs by line separation
  and by tier. `scripts/build_gaia_rvs_notebook.py` regenerates it.
- **Pipeline setting for the label comparison, and epoch-fit results in the report (D65).**
  `Analysis.label_compare` (`compare` under `[labels]`, default `"epochs"`, validated against
  `"epochs"`, `"native"` and `"matched"`) is passed to `Fit.match_labels`, which already had
  that default, and recorded as `labels.compare`. Under `labels`, `result.json` also records
  the light fraction the fit measured and its formal error (`flux_ratio`, `flux_ratio_errors`),
  the declared fraction (`light_declared`), the sites the epoch fit left without a formal error
  and the reason (`at_bounds`), the notes of its restarts (`notes`) and a summary of its
  optimisation (`epoch_fit`). Under `velocities` it records the source of the correlation's
  amplitudes (`light_source`: `"declared"` or `"global re-measure"`). The measured light is not
  used as the correlation's amplitude. The disentangled components are `(w / l0) t`, so `l0` is
  the amplitude that reproduces the epochs with them. On a simulated pair with true light
  fractions of 0.70 and 0.30, declared as 0.45 and 0.55, holding the true fractions instead
  raised the fainter component's rms velocity error from 0.34 to 14.3 km/s. Templates rescaled
  by `l0 / w` gave the declared case's velocities exactly
  (`test_disentangled_templates_are_reproduced_by_the_declared_light_not_the_measured_one`).
- **`albireo.gaia.rvs_delivered_sigma_kms` (D65).** The function returns the Gaussian width of
  a line resampled by `deliver`: `sigma_eff^2 = sigma_R^2 + Delta_det^2 / 6 + (Delta_det^2 -
  Delta_prod^2) / 12`. This holds because a linear interpolation at fraction `t` between
  detector pixels is a two-tap kernel of zero mean and variance `t (1 - t) Delta_det^2`, and
  `t` takes every value across the band on both shipped grids. Each delivered sample also
  retains the detector pixel's box average, whereas an analysis's operator integrates over the
  delivered pixel. At the nominal resolving power of 11,500 the width is 11.60 km/s on the DR4
  grid and 11.83 km/s on the DR3 grid against 11.07, with `Delta_det = 8.56` km/s at the band
  centre. `simulation_dv_kms` adds the simulation's `(5/12) dv^2` (entry under Changed). The
  tests measure the second moment `deliver` adds to 49 lines placed at every phase of the DR4
  grid's beat, which is within 0.1 percent of `Delta_det^2 / 6` on both products. For lines
  integrated over the detector pixels against the same lines integrated over the delivered
  ones, the added moment is 17.312 km^2/s^2 against 17.303 predicted on the DR3 grid and 12.06
  against 11.96 on the DR4 grid.
- `compare="epochs"` for `match_labels`, which compares the template composite with the epoch
  spectra through the disentangling's sufficient statistics (`EpochStatistics`,
  `Fit.epoch_statistics`). The chi-square is exact for the noise model including AR(1) and
  independent of the declared light fractions. It is minimised by a bounded
  Levenberg-Marquardt with rotation, node and joint restarts. Formal errors come from the
  Gauss-Newton matrix, and the restarts are recorded on the result. (D65)
- `SpectralLibrary.resolving_power`, read from `meta["resolving_power"]` or the BOSZ
  `meta["resolution"]`. (D65)
- `LabelMatch.restarts`, `LabelMatch.at_bounds`, `LabelMatch.flux_ratio_errors` and
  `LabelMatch.epoch_fit`. (D65)
- **A detection threshold in the bootstrap's period search (D65).** `Analysis.detection_min`
  (`detection_min` under `[analysis]`, default 100, `0` to turn it off) applies only in the
  period search and the candidate orbit fits of the `period = "search"` bootstrap. There, a
  companion's velocity whose detection statistic is below the threshold is set to `nan` in a
  copy of the table, and the first component of the epoch is kept. Only the components after
  the first, declared in order of decreasing mass, are subject to the threshold. The selected
  orbit is fitted to that copy and gives the disentangling's semi-amplitude, conjunction and
  eccentricity starts. The written table, the light measured from it, the known-period
  route's template table and every velocity measured after the disentangling are left as
  measured. `bootstrap.detection_gate` counts the velocities removed per component. Over the
  33 blind tables of the benchmark's third run, rerun with the code as implemented, the
  threshold changed 9. A field system whose 7% secondary the library templates never detected
  moved from absent in the chi-square ranking to rank 1 (547.4 d against a true 552.5 d). A
  Gaia system moved from rank 20 to 8, and another field system from rank 24 to absent. No
  system lost its rank 1 (21 at rank 1 against 20). A threshold of 25, the table summary's
  value, failed because six epochs with statistics between 25 and 61 kept secondary velocities
  up to 212 km/s in error. With the first component also subject to the threshold, a field
  system whose first template fell below 100 at 5 of 16 epochs moved from rank 2 to absent,
  and the period decision compares the top four.
- **Leave-one-epoch-out candidate periods on short bootstrap tables (D65).** This step runs in
  the bootstrap only (`_period_candidates(..., leave_one_out=True)`). The `velocities = "file"`
  route does not run it. On a table of at most 25 usable epochs as measured
  (`_LEAVE_ONE_OUT_MAX_EPOCHS`, counted before the detection threshold is applied), the
  one-harmonic search is repeated with each of those epochs left out in turn. The three highest
  peaks of each repetition (`_LEAVE_ONE_OUT_PEAKS`) are appended after the round robin wherever
  they lie more than 2% from every start already listed. Rerun with the code as implemented,
  over the 17 blind tables of at most 25 usable epochs this added 18 starts. A Gaia system whose
  twin components were exchanged at one epoch of twelve moved from absent to rank 1, one other
  moved from rank 37 to 38, and no system lost its rank 1 (21 at rank 1 against 20). The
  searches record the added starts under `leave_one_out`, and the bootstrap report counts them.
- **The D65 search changes measured together (D65).** Over the 33 blind tables of the third
  run the chi-square ranking placed the true period first on 22 systems against 20 before,
  gaining the two above and losing none. The other ranks moved among near-degenerate candidates:
  one Gaia system from 39 to 12, one from 23 to 24, one from 20 to 9, one from 3 to 4, one from
  37 to 39, and field systems from 40 to 41, 2 to 6 and 24 to absent.
- **A flag for a bootstrap table on fewer than eight nights (D65).** When the usable epochs of
  the bootstrap table fall on fewer than eight distinct integer parts of the BJD
  (`_FEW_NIGHTS`), a flag states that the velocities cannot be expected to determine the period,
  and `bootstrap.n_nights` records the count. Nothing else changes. Three of the 33 blind tables
  were that sparse, and the search recovered none of them.
- **The table used by the period search is written beside the delivered one (D64).** On the
  search route the delivered `template_velocities.rv` has the selected orbit's component
  assignment rather than the correlation's, because the bootstrap re-assigns the epochs
  where the correlation exchanged two similar spectra. Its header now states this and
  gives the number of re-assigned epochs. Whenever an epoch was re-assigned, the table as
  measured is written beside it as `template_velocities_unexchanged.rv`. Without that
  table the period search cannot be reproduced from the written products.

- **The template velocity table is written by every run that measures one (D63).** The
  bootstrap of the search route, the light measurement and the semi-amplitude start each
  measure the epochs against library templates at the declared starting labels before
  anything is disentangled. That table determines the period and the starting
  semi-amplitudes. It is now written as soon as it exists (`template_velocities.rv` and
  `.csv`, with the purpose in the header), so it is on disk when a later stage fails and
  the decision based on it can be examined.
- **Gaia transit times from GOST (D63).** `gost_transits(ra_deg, dec_deg)` fetches the
  forecast of a sky position's field-of-view crossings from ESA's ObjVisSAP endpoint (the
  nominal scanning law, barycentric MJD in TCB, no CCD row and no scan angle), parses the
  VOTable with the standard library and caches the response under the cache directory.
  `rvs_transit_times_from_gost` keeps the crossings inside the release span, draws the RVS
  rows (four of the seven, since the service does not publish the row) and thins them to
  the usable fraction of 0.78, on which GOST's reception rate, Katz et al. 2023 and the
  DR3 counts agree. `BinarySystem` has `ra_deg` and `dec_deg`, filled from the Gaia
  query or drawn at the population's ecliptic latitude and rotated to ICRS
  (`ecliptic_to_icrs`, within an arcsecond of astropy's mean ecliptic). The benchmark's
  `cadence="gost"` (`--cadence gost`) simulates each system at its own forecast epochs,
  with `system_transit_times` to compute them first and the minimum transit rule applied
  to the forecast. The statistical cadence remains the default. Where a position is known
  the phase of the scanning law is now reproduced. The first two benchmark runs had
  stated it as an approximation.
- **A BOSZ 2024 box above 7000 K for A and early-F stars (D63).** `bosz2024-hot-rvs` and
  `bosz2024-hot-r20000` cover 7000 to 10,000 K in 250 K steps, log g 3.5 to 5.0 and [M/H]
  -1.0 to +0.5. All 364 nodes are published (verified against the archive listing on
  2026-09-10, a harvest of which is a test fixture), so the cubic interpolant applies with
  nothing filled. The box crosses the MARCS/ATLAS9 transition between 8000 and 8250 K. Its
  caveats state the effect of that transition, the effect of the Paschen series on a
  velocity from the RVS band above 8000 K, that A stars rotate faster than R = 11,500
  resolves, that the LTE error there is unquantified, and that the archive's hot shards
  predate the hydrogen-line recomputation date. The one-time download is 532 MB
  (`fetch_library`), and the cached RVS slice is 4 MB. With this box the benchmark
  script's `--library` accepts the catalogue systems above 7000 K. The FGK entries'
  declared sizes are corrected to the measured ones (621 MB to fetch, 5 MB and 51 MB
  cached; previously 645, 16 and 95).

- **The measured epoch velocities are recorded by every run and collected by the benchmark
  (D62).** The pipeline has always written one velocity per component per epoch, by TODCOR
  against the disentangled components with the label fit's zero points (`velocities.rv`,
  `velocities.csv`). `result.json` now also records the disentangling's Keplerian at every
  epoch and, for a simulation, the injected velocity of every epoch in the order in which
  the components were compared. In the benchmark, `collect_velocities` writes
  `velocities.csv` (one row per system, tier, epoch and component, with the error, the
  flags, the injected velocity and the pull), and `summarize_velocities` pools the usable
  epochs per tier into an "Epoch velocities" section of the report. Every tier's systems
  are plotted phase-folded against the injected orbit (`figures/rv_curves_<tier>.png`).
- **A declared noise correlation, applied at every stage.**
  `Disentangler(noise_correlation=...)` (a number, a table per instrument, or a
  `Between`/`Known` to fit one shared value) makes the noise model AR(1) along the pixel
  index, held or fitted through the `ar1_phi` site the low-level model already had.
  `explain()` states it, and a declared value is listed as an assumption.
  `todcor(noise_correlation=...)` replaces the curvature error by the sandwich computed
  with it (math.md §10.4), which `Fit.measure_velocities` applies with the declared value.
  The pipeline's `noise_correlation` setting passes it to every stage. The benchmark
  declares the delivered grid's lag-one correlation to every tier as the simulation
  measured it (`noise_model="correlated"`, the default), because the pixels of a resampled
  spectrum are not independent and the archive's errors do not record this.
- **A joint scan over the semi-amplitudes and the phase before the fit.** With a
  semi-amplitude declared as a range, `Disentangler.fit` (`k_scan="auto"`) scans the
  marginal likelihood over a geometric grid of the ranged semi-amplitudes crossed with a
  grid of conjunction phases, refined twice around the best trial. The scan runs on a
  copy of the declaration with the model grid at twice the pixel size. L-BFGS starts from
  the best trial when its likelihood exceeds that of the declared start. The record is
  `fit.k_scan`, and the pipeline's `k_scan` setting, on by default, reports and flags
  what the scan moved. On the Gaia DR3 population a template table of a dozen epochs had
  started the routes with ranged semi-amplitudes in the wrong basin on half the systems.
  The phase must be scanned with the semi-amplitudes, because a phase located at
  semi-amplitudes a factor of three off can be a quarter of a period from the true value.
  The template table's orbit now also initialises a free eccentricity and argument of
  periastron (`Orbit(ecc=Between(0, hi, start_at=e), omega=w)`) when it detected them at
  three sigma, so the scan uses a velocity curve of the correct shape.
- **The one gap in the BOSZ FGK box is filled, and the cubic interpolant is used again.**
  `ingest_bosz` now fills a missing node that two published neighbours bracket along one
  axis by linear interpolation between them (metallicity first) and records it in
  `meta["filled_nodes"]`. The library summary and the label report name it. An
  unbracketed node is still dropped. The registry version of both BOSZ libraries moves to
  2, so cached builds are remade from the raw shards on first use. BOSZ publishes no model
  at 5750 K, log g 3.0, [M/H] -0.75. With that node missing the grid was not a complete
  box, so `library_interpolator` fell back to the barycentric simplex path, which is
  piecewise linear over an arbitrary triangulation of the lattice, and the label fit
  stopped at its kinks. On a perfect spectrum drawn from the grid itself the fit ended 20
  to 150 K from the true labels with a chi-square above theirs. This was the temperature
  bias that the first Gaia RVS benchmark reported as open (D61). With the box completed
  and the Catmull-Rom cubic in use, the same fits end within about 30 K.
- **A Gaia RVS simulator, populations of double-lined binaries, and a benchmark that runs
  them through the pipeline (D60, D61).** `albireo.gaia` reproduces the Gaia RVS
  observation chain of Rowan's reference notebook, with every constant verified against a
  primary source. The chain is G_RVS from the colour, the S/N per detector pixel from G_RVS
  and the transit count, photon noise on the detector grid, delivery onto the DR3
  mean-spectrum or the DR4 epoch grid with the variance propagated and the noise
  correlation recorded, the in-flight resolving powers per CCD row, and a transit cadence
  with the scanning law's measured structure. `albireo.population` draws systems from the
  field's distributions (Moe & Di Stefano 2017, a dwarf sequence, the eclipse geometry,
  tidal synchronisation, the RVS light ratio from the library continua) as selected by a
  magnitude-limited double-lined survey, or builds them from DEBCat and the Gaia DR3
  double-lined orbits. `albireo.benchmark` simulates each system once, runs it under
  knowledge tiers (`oracle`, `eclipsing`, `orbit`, `blind`) through `run_pipeline`, and
  writes `report.md` with figures, `rows.csv` and `summary.json`.
  `scripts/gaia_rvs_benchmark.py` runs it and resumes an interrupted run. The new names are
  `simulate_rvs_dataset`, `rvs_snr_per_pixel`, `predict_grvs`, `rvs_transit_times`,
  `RVS_DR3_MEAN`, `RVS_DR4_EPOCH`, `RVSProduct`, `RVSTruth`, `BinarySystem`,
  `MainSequence`, `draw_population`, `from_debcat`, `from_gaia_sb2`, `query_gaia_sb2`,
  `read_population`, `write_population`, `BenchmarkConfig`, `BenchmarkRun`, `Tier`,
  `run_benchmark`, `write_report` and `t_conj_from_t_peri`. `simulate_dataset` gains
  photon-counting noise (`InstrumentSpec.shot_noise`, `snr_window`) and a per-epoch
  `epoch_snr`, and the injected truth records the noise sigma. `examples/14_gaia_rvs.py`
  reproduces the notebook's system.
- **Pipeline declarations for the quantities known for an eclipsing binary.** `StarConfig`
  takes `t_conj` (a known conjunction instead of the phase scan), `ecc` (held or a range)
  and `omega`. `ComponentConfig(light="measure")` measures the light fractions against
  library templates on any route with a library. The injected-truth block reports element
  and velocity pulls, the recovered spectra's correlation and equivalent widths per
  window, and the light fractions as declared and as measured. `find_period` oversamples
  the baseline. `reassign_by_orbit` exchanges the two components of a velocity table at
  the epochs where, according to the disentangling's orbit, the correlation selected the
  mirror minimum, which occurs for similar components at similar light fractions. The
  pipeline applies it and flags the count. `find_period(swap_invariant=True)` searches the
  magnitude of the relative velocity, which the exchange leaves unchanged. The search
  route's bootstrap tries its peaks and their doubles, re-assigns the table by each
  candidate orbit, and falls back to the declared K range when a bootstrap semi-amplitude
  is degenerate. A semi-amplitude declared as a range now starts at the value from a
  library-template table fitted at the declared period, instead of at evenly spaced
  points of the range. Over a range of 2 to 250 km/s those points started a 23 km/s
  primary at 85 km/s, and the fit converged to the static-component minimum. A component
  the table could not measure starts from its nearest measured neighbour, above it for a
  lighter star and below it for a heavier one, and every start is held strictly inside the
  range. `Between.start` now rejects a start on a bound, which numpyro reports only as
  "cannot find valid initial parameters". The injected-truth block detects a pair
  recovered in the other order (twins, to which the mass-order convention does not
  apply), flags it, and compares in that order.

### Changed

- **The epoch comparison removes the model grid's smoothing from its operator (D65).**
  `Fit.epoch_statistics(grid_compensation=True)`, the default, and so `Fit.match_labels`, build
  the stellar operator with `sigma_op^2 = sigma_inst^2 - sigma_lib^2 - (7/12) dv^2`. Five discrete
  steps between a library spectrum and the epoch pixels smooth the template by `(7/12) dv^2`:
  the library's box average onto the model grid, the rotation kernel's pixel integration and
  the model pixel in the rebin, each `dv^2 / 12`, and the frame shift and the epoch shifts by
  linear interpolation, each `dv^2 / 6` on average. The epochs do not have this smoothing, and
  the difference was absorbed into `v sin i`. On the closed-loop fixture of
  `tests/test_pipeline.py` (native pixel 4.63 km/s, LSF 5.5 km/s, 11 km/s injected) the mean
  `v sin i` error over five noise draws went from -3.02 to -0.24 km/s.
  `test_the_pipeline_recovers_the_injected_system` had returned 6.91 km/s against the 8.25 its
  25 percent tolerance allows, and now passes with the tolerance unchanged (9.82 km/s in the
  study, with the temperatures, gravities, light fractions, semi-amplitudes and systemic
  velocity unchanged; `d65_grid_smoothing.md`). The compensation assumes epochs with no
  discretisation of their own. Epochs simulated on the model grid itself have all of it but
  the frame shift, and against them the compensation biases `v sin i` high (9 km/s fitted at
  11.1 in `tests/test_match_epochs.py`, whose recovery test now renders its epochs on a 1 km/s
  grid). A width that would fall below half a model pixel (when `dv > 1.10 sigma_q`) is floored
  there. That case and a model grid coarser than `sigma_q` each raise a `UserWarning` that
  names the grid spacing that avoids it. `EpochStatistics` records `grid_variance_kms2`,
  `operator_sigma_kms` and `notes` beside `lsf_sigma_kms`, which now holds the widths before
  the compensation, and prints them in the new `EpochStatistics.summary()`.
  `LabelMatch.assumptions` contains `epoch_quadrature_lsf_sigma_kms` and
  `epoch_grid_variance_kms2`. `match_labels` checks a compensated operator through those fields
  and rejects one whose widths do not follow from the declared instrument width, as produced by
  the resolving-power workaround of the research runs. The pipeline records the operator as
  `labels.epoch_operator` in `result.json` and flags a floored or coarse-grid compensation.
  `grid_compensation=False` reproduces the previous operator exactly.
- **The Gaia benchmark also declares the simulation's discretisation and the detector pixel
  (D65).** `build_star` declares `rvs_delivered_sigma_kms(product, simulation_dv_kms=grid.dv_kms)`,
  11.67 km/s on the DR4 grid (11.90 km/s on the DR3 grid), in place of 11.61. The
  simulation renders its epochs on a 2 km/s model grid through the box average, the rotation
  kernel, the shift interpolation and the rebin, which add `(5/12) dv^2` in variance (measured
  through the shipped chain: 1.29 km^2/s^2 without rotation, 1.52 to 1.58 with it). Each
  delivered sample retains the 0.245 A detector pixel's box average, whereas the operator
  integrates over the delivered pixel. The difference is `(Delta_det^2 - Delta_prod^2) / 12`:
  -0.25 km^2/s^2 on the DR4 grid and +5.09 on the DR3 grid (measured 17.312 against 17.303 with
  the interpolation). The previous declaration left both terms out.
  `gaia.SIMULATION_VARIANCE_FACTOR` is the `5/12`. On three slow-rotator products rebuilt at
  their archived MAP with the grid compensation on, the median `v sin i` error of the six
  components was +0.63 km/s (+0.32 for the primaries) at this declaration. It was +1.09 at
  11.61, +4.22 at the nominal width and +3.06 for the previous, uncompensated fit.
  Temperatures were within 0.9 K, gravities within 0.003 dex and light fractions within
  0.0001 of that fit. Benchmark results before and after are not comparable in rotation.
- **`gaia.simulate_rvs_dataset` takes the components' resolving power from their library
  (D65).** The new `library=` argument supplies `SpectralLibrary.resolving_power` (20,000 for
  the BOSZ entries, `None` for an intrinsic grid) as the default of `library_resolving_power`,
  which no longer defaults to 20,000. One of the two must be given, and they must agree when
  both are. The 20,000 default had broadened an intrinsic library to 9.06 km/s where the RVS
  gives 11.07. `benchmark.simulate_system`, `examples/14_gaia_rvs.py` and the tutorial pass the
  library, and the example and the tutorial now declare the delivered width to the
  disentangler and to TODCOR rather than the nominal 11,500.
- **The Gaia benchmark declares the line-spread width of the delivered epochs (D65).**
  `build_star` declares `rvs_delivered_sigma_kms`, 11.61 km/s at first and 11.67 km/s with the
  terms of the entry above, under both `resolving_power` settings. It previously declared the
  nominal resolving power of 11,500 (11.07 km/s). The manifest records the width as
  `settings.lsf_sigma_kms` and the report states it. Only the delivery's smoothing is added,
  not any model-grid smoothing of the analysis. On the D65 products the converged label fit
  had absorbed the difference into `v sin i`, so benchmark results from before and after this
  change are not comparable in rotation. The disentangled components, the velocity tables and
  the label fits all use the wider operator. `simulate_system` now takes the library's
  resolving power from `SpectralLibrary.resolving_power` instead of
  `meta.get("resolution", 20_000)`. Both give 20,000 for every BOSZ registry entry, but the
  fallback had simulated an intrinsic library (the test suite's toy grid) as one at 20,000,
  so that its delivered epochs had a width of 9.71 km/s against a declared 11.61. With the new
  declaration and the old fallback, the velocity table of one oracle system of the
  benchmark's end-to-end test failed (semi-amplitudes from the table 463 and 695 percent
  high, reduced chi-square 42.5). With both changes the semi-amplitudes are 1.7 and 3.7
  percent low at quoted errors of 2.7 and 2.9 percent.
- **The label stage flags what the epoch fit did not measure, and the light flag tests a ratio
  as well as a difference (D65).** A label in `LabelMatch.at_bounds` is named in its own flag
  with the reason, since a site on a bound or on the rotation plateau is left out of the
  covariance and the posterior-width flag does not detect it. Every note of `epoch_fit.notes`
  (a collapsed faint component) becomes a flag. The light flag is raised when the measured
  fraction differs from the declared one by more than a factor of 1.5 or by more than 0.15. It
  lists every component with its formal error, and it replaces the flag that named only the
  first component past 0.15. With sampling on, an epochs match is not refitted over `d_hat`
  draws, which `refit_draws` rejects. The stage logs this and does not flag a failure.
- **`LabelMatch.summary()` groups the additive-offset pairs of the epochs comparison (D65).**
  The offsets of the components enter the epochs only as their light-weighted sum, so their
  pairs correlate near -0.99 on every fit. They are now one line under "Degenerate pairs"
  rather than one line each, and an offset paired with a label is still listed.
  `flagged_correlations` is unchanged.
- `Fit.match_labels` defaults to `compare="epochs"`, built at the libraries' common resolving
  power. `albireo.match_labels` keeps `"native"` and rejects `statistics` without
  `compare="epochs"`. (D65)
- The warm-start `scan_vsini` default is 1, 5 and 10 km/s and a fifth and a half of the
  prior's upper bound, so slow rotation is tried. (D65)
- `compare="matched"` convolves each template with sqrt(sigma_inst^2 - sigma_lib^2) when its
  library declares a resolving power, and rejects a library at or below the instrument's.
  (D65)
- `match_labels(jitter=...)` defaults to None, which means on for the d_hat comparisons and off
  for "epochs". (D65)
- docs/math.md gains section 9.2a and the measured pull calibration in 9.5. (D65)
- **The candidate periods are merged by rank, and fifty peaks are taken instead of twenty
  (D64).** `_period_candidates` now merges its four periodogram sources round robin by rank,
  which resolves a collision in favour of the higher-ranked peak whichever source proposed it
  and makes a larger count strictly additive. It previously concatenated them in a fixed order
  with the swap-invariant peaks last, then deduplicated greedily at two percent. A low-ranked
  peak of an early source therefore displaced the top peak of a late one, and the proposal was
  not monotone in the peak count: 233 of 1296 starts present at twenty peaks were absent at
  fifty. Over the 33 blind systems of the benchmark's third run, the merge alone brought the
  true period to chi-square rank 1 on one further system, and no system lost its place.
  Raising the count to fifty (`_PERIODOGRAM_PEAKS`) gained one further system again, at twice
  the wall-clock time of the candidate fitting. A hundred peaks was measured and gained
  nothing beyond fifty.
- **`Analysis.period_decision_candidates` defaults to four instead of eight (D64).** Eight was
  set on the assumption that the true period ranked fourth to fourteenth among the candidates
  on the systems the search missed. The whole chi-square ranking, rebuilt offline, shows that
  the true period ranks first on every system recovered and is absent from the candidate list
  on nine of thirty-three. Eight would therefore reach one further system per population, at
  four more coarse scans per blind star. The docstring now records what the stage can and
  cannot do. The stage scores each candidate at a single point in the period and the
  eccentricity, the two quantities a sparse velocity table measures worst. Scanning each
  candidate's whole window instead is biased toward the shortest period, since peaks are
  spaced by about `0.28 P / T` and a short period's window holds ten to thirty times more
  independent trials. It is also unaffordable, at about 62 hours for four candidates against
  116 seconds. The stage has converted no benchmark miss into a hit. Its value is the flag it
  raises. On the Gaia blind tier the flag was raised four times, each on a system that ended
  with a wrong period.
- **The period search uses a floating-mean periodogram with twenty candidates, and the
  orbit fit decides among more of them (D63).** `find_period` computes the weighted
  generalized Lomb-Scargle periodogram of Zechmeister and Kurster (2009), with a free
  constant, in numpy. It replaces `scipy.signal.lombscargle` on the series scaled by the
  square root of its weights. Ten to twenty-five epochs falling into a dozen visibility
  windows hundreds of days apart have a sampling window whose mean is large at most
  frequencies, and a sinusoid with no constant term absorbs it as spurious power. On the
  oracle-tier tables of the second Gaia benchmark the true period ranked 9, 9, 15, 25 and
  103 among the distinct peaks on the five Gaia-like systems the search had missed, and
  1, 1, 1, 10 and 3 with the floating mean. `n_peaks` (default twenty, the best plus
  nineteen aliases) replaces the fixed six. The greedy peak loop visits the local maxima
  only. `n_harmonics=2` fits a fundamental and its first harmonic for eccentric orbits.
  The pipeline's search route proposes the union of the twenty single-harmonic peaks, the
  twenty two-harmonic peaks, the twenty peaks of the first component's own velocities and
  the swap-invariant peaks with their doubles. The first component's velocities are
  unaffected by a companion the templates could not follow, whose relative velocity is
  then noise while the primary's curve is intact. Every candidate orbit above the declared
  semi-amplitude ceiling or eccentricity maximum is set aside, because the spurious
  velocities of a 7-percent secondary had caused a wrong period to be selected at
  1537 km/s and e = 0.94. A semi-amplitude below the floor is not tested, because an
  unmeasurable companion gives one at any period. The route fits the eccentric Keplerian
  at every distinct start, merges fits agreeing within 2 percent on the fitted period and
  keeps the lowest chi-square. It names every distinct fitted period within 25 in
  chi-square of that fit as an ambiguity (`bootstrap.ambiguous`, `n_candidates`,
  `harmonic_peak` in the report). On the same 32 tables the recovery within 2 percent went
  from 23 to 28. The four left are two tables of ten or eleven epochs whose true Keplerian
  fits worse than an alias (the flag is raised on both), one period 2.04 percent off and
  one decision lost by 4.4 in chi-square. The frequency grid (ten per inverse baseline)
  and the 2 percent distinctness rule were tested and kept. A period near a third of the
  baseline is located to a few percent by the grid and determined by the fit. On the third
  Gaia run the table's chi-square still selected implausible orbits inside the declared
  ranges on the noisier bootstrap tables (twins at 6.10 d fitted at 0.248 d with
  K 233/239 at e 0.73; a 5.36-day pair at 1.13 d). The disentangling therefore now decides
  among the best few candidates (`Analysis.period_decision_candidates`, default 8: on the
  third run's misses the true period ranked 4 to 14 in the table's chi-square). Each
  candidate is declared with every semi-amplitude as the same range, so the model grids
  match. Its conjunction and semi-amplitudes are located by the coarse scans, and the
  prior-free marginal log-likelihoods are compared. When the table is overruled, a flag
  names both choices and the margin in nats. `bootstrap.decision` records every candidate.
  The decision takes about 340 coarse solves per candidate.
- **The semi-amplitude scan takes one component at a time and evaluates the move jointly
  (D63).** The scan now takes the first ranged component over a geometric grid at ratio
  1.25 with eight phases, and each further component over its own grid at the values
  already located. It then refines jointly twice, takes each component once more over its
  whole grid at the refined best, and tries the exchange-symmetric twin of the best for
  two ranged components. The start moves when the best trial exceeds it by more than 25
  nats jointly, and a component whose own move gains less than 5 nats returns to its
  start. The scan uses fewer trials than before (about 270 against 446 for two
  components), and the record's notes state which guard acted. The change follows from a
  measurement on the Gaia system whose 7-percent secondary the D62 scan had missed. The
  companion changes the log-likelihood by about 25 nats over its whole range, and its
  maximum is at its true semi-amplitude only while the primary's is within about ten
  percent of the true value. The primary's own peak falls by 50 nats within twenty
  percent. On a product grid at ratio 1.8, no trial held the primary close enough for the
  companion's likelihood to peak at the true value, and the local refinement did not move
  the companion out of the basin in which it started. The guard then rejected the correct
  joint move (39 nats) as two single-component holds of 15 nats each. Between two passes
  each star's prior amplitude is now profiled at the located orbit through the model's
  light site. The marginal likelihood is exactly invariant under scaling a star's light by
  a factor and its smoothness hyperparameters by the factor squared (to 5e-10 nats). The
  profile therefore measures the best-fitting prior amplitude and not the light (its
  maximum was at 0.3 to 0.7 of the injected fraction on four systems). Where the profile
  peaks a factor of two or more from the start, the hyperparameter starts are moved and
  the scan is repeated (`fit.k_scan.prior_scales`, a note in the record). A secondary
  declared at six times its light, for which the scan had returned a static companion,
  was then recovered to within 4 percent of its semi-amplitude.

### Fixed

- **Links from the notebook pages of the docs site.** A relative `.md` or `.ipynb` link in an
  executed notebook was published as written and led to a missing page on the site, although
  it resolved on GitHub. `scripts/mkdocs_notebook_links_hook.py` rewrites these links to the
  built page URLs and warns on a missing target, which fails a strict build.
- **`MAPResult.potential` docstring corrected (D65).** The docstring called it the negative
  log joint. It is numpyro's unconstrained-space potential, which also includes the
  log-Jacobian of every bounded site's transform, so `run_map` returns the mode in the
  unconstrained coordinates. Over the 72 archived benchmark fits that sample a bounded site,
  the difference from the constrained-space MAP on recovered orbits has a median of 0.007
  formal sigma and is at most 0.23, so `run_map` is unchanged.
- **`Template.from_labels` records the width the label template already has (D65).** It
  read `match.config["lsf_sigma_kms"]`, a key the config never has, so a `"matched"` template
  recorded `sigma_kms = 0` and TODCOR applied the instrument profile a second time. A
  `"native"` or `"epochs"` template drawn from a library at its own resolving power had that
  width without recording it, and TODCOR applied the whole instrument profile as well. It now
  records `c / (R 2 sqrt(2 ln 2))` from the match's library, as `Template.from_library` does,
  and for `"matched"` also the width the fit applied, which together make the declared
  instrument width. TODCOR then applies only the quadrature remainder. This changes the
  velocities measured against label templates. The pipeline correlates against the
  disentangled components and is not affected.
- Label templates drawn from a library at its own resolving power are no longer broadened
  twice in the epoch and matched comparisons. (D65)
- **A velocity is used wherever its own component was measured (D65).** `find_period` and
  `fit_rv_orbit` intersected their own finiteness test with `VelocityTable.good`, which
  requires every component to be finite and off the search edge. The search on the first
  component alone, the source for a companion the templates could not follow, therefore lost
  the primary epochs at which that companion was at the edge (one on a benchmark system with
  a 7% secondary, two on another). A velocity is now used where its own component has a
  finite velocity and error off its own search edge at an epoch that is not blended. The
  relative velocity still needs both. Nothing changes where every component is valid.
  `find_period` returns the epochs its series used under `used`. The semi-amplitude start is
  still half the range of a component's velocities, now of its own usable ones. In a joint
  fit a component with no more usable velocities than parameters of its own (one, or two with
  its own systemic velocity) is held (`RVOrbit.held`). Its semi-amplitude stays at 1e-3 km/s
  or the caller's `k` and its own systemic velocity at its start. Neither is fitted or
  counted, both have `nan` errors, and its velocities have no weight. `mass_ratio`,
  `minimum_masses` and `projected_semiaxes` return nothing for it. Left free, such a
  semi-amplitude reached 1.5e7 km/s on a benchmark table whose secondary the detection gate
  removed at every epoch, and the ranking excluded the true period as outside the ranges. A
  start of `sqrt(2)` times the weighted standard deviation was tried and rejected. It is 0.34
  to 0.51 of `K` at `e = 0.9` and moved the ranking on six tables whose components were all
  usable.
- **`fit_rv_orbit` and `RVOrbit.predict` no longer recompile at every call (D65).** The
  residuals and Jacobian were closures over each call's data, and `predict` evaluated the
  Kepler solver eagerly, which compiled its fixed-count Newton loop again at every call. The
  period search's candidate loop calls both once per starting period. Both are now
  module-level functions compiled once per configuration and array shape (`_objective`,
  `_predictor`), with results bit-identical to the base commit on the all-valid reference
  table. On the benchmark harness, three tables through six search configurations took 269 s
  and reached a peak working set of 5061 MB with the objective alone moved, and 26 s and
  446 MB with `predict` moved too. The full 33-table rerun of every configuration peaked at
  697 MB.
- **The velocity CSV has one edge flag per component (D65).** `velocities.csv` and
  `template_velocities.csv` add one `at_edge_<component>` column per component beside the
  merged `at_edge`, which the `.rv` files keep unchanged.
- **A period search can be reproduced from the written velocity table (D65).** `find_period`'s
  default grid was spaced evenly between `1/(2 T)` and `1/(2 dt_min)`, so a shift of 1e-6 d in
  the closest pair of epochs moved every upper frequency. On a Gaia-like table it moved the
  top of the grid by more than a step and changed the order of near-degenerate short-period
  peaks. The velocity tables gave the epoch time to six decimals, and the recorded peaks of
  five benchmark tables could be reproduced only by moving the shortest period by 0.5 to
  4e-6 d. The default grid is now anchored at its low end with a step of `1/(N T)`, `N`
  computed from the span rounded down to two significant figures, and the high end appended.
  Moving an epoch other than the first and last leaves every point below both high ends in
  place unless `N` changes. On a multi-year table (`N = 10`) it cannot, and below that it
  changes only at a rounding boundary. `VelocityTable.write` and the pipeline's CSV write the
  epoch times with every digit of a float64. An explicit `n_frequencies` still gives an
  evenly spaced grid. Over the 33 blind tables of the third run the new grid leaves the
  number of systems whose true period ranks first at 20. It reorders near-degenerate
  candidates elsewhere: one Gaia system from rank 39 to 11, one from 23 to 24, one from 3 to
  4, and field systems from 40 to 41 and from 2 to 6. The last takes that system's true
  period out of the top four that the decision by the disentangling compares, where the start
  nearest the true period now converges at chi-square 587 instead of 460. That decision has
  converted no miss into a hit on either population (D64).
- **The bootstrap's candidate fits follow the exchange rule of the later stage (D65).**
  `_orbit_over_candidates` re-assigned the components at every candidate, whereas the
  velocities measured after the disentangling are exchanged only where the light fractions are
  within a factor 3 (`_exchange_allowed`). The candidate fits now apply the same rule to the
  light fractions of the library table's own amplitudes and flag the skip. The winning orbits
  of two field systems at factors 7.1 and 3.25 had re-assigned 3 and 11 of their epochs. Over
  the 33 blind tables of the third run, the rule skips the exchange on 8 and no system loses
  its rank 1. Under it the true period of one Gaia system moves from rank 20 to 28, of
  another from 37 to 38, and of the field system with the 7% secondary from absent to 25.
- **The conjunction-phase grid contains the antipode of every trial (D64).**
  `Disentangler._scan_phase` placed 41 trials over one period. With an odd count the antipode
  of every trial fell midway between two others, and the grid could not choose between the
  two mirrors of a near-equal pair. The count is now 42, which is even and also finer. On one
  benchmark star the antipode the old grid could not sample was better by 672 nats. The later
  semi-amplitude scan found it and moved the conjunction half a period, which is the origin
  of the D63 initialisation failure.
- **A star whose files declare no wavelength medium no longer fails at the semi-amplitude
  start (D64).** Where a semi-amplitude is a range and a library is configured, the pipeline
  takes the starting values from a template table. That table is not requested, but it
  raised the undeclared-medium error when the files declared no air-or-vacuum scale, which
  failed the star and, among others, the shipped `albireo demo`. The stage is now skipped with
  a flag stating what was skipped and how to enable it, which is the module's policy for a
  stage that cannot run. The error is still raised where the user declared
  `period = "search"` or `light = "measure"`, which are stages that cannot proceed without the
  medium.
- **Three unsupported claims in `docs/math.md` section 5.2 corrected (D64).** The physicality
  floor was described as "available as an optional constraint", but nothing in the package
  imposes it. It is now described as a bound worth reporting beside a declared light. Measured
  over 66 components, it reaches a median 0.61 of the true fraction and never exceeds it. The
  section also documented a `light_ratio=` argument taking `Fixed(values)`, `Free(prior=...)`
  or `PerEpoch(...)`. None of those names exist: the D46 review of the `Disentangler`
  interface rejected that argument in favour of `light=` on the `Star`. The section now states
  what the two paths provide.

- **The marginal likelihood rejects an evaluation with a negative chi-square (D63).** One
  oracle-tier fit of the second Gaia benchmark diverged (traced in
  `internal/research/2026-09-09-gaia-rvs-benchmark/`). ML-II was raising a component's
  smoothness precision toward its interior optimum when a line search accepted a trial at
  tau of 2e19. There the assembled posterior precision has a Cholesky pivot of 5e9 against a
  data scale of 1e4, and the forward substitution loses every digit. The quadratic form
  exceeded the data term, and the negative chi-square was reported as a 216,000-nat
  improvement. The period then moved 228 prior sigma and the residual z-score rms ended at
  45. `marginal_loglikelihood` now returns minus infinity for a negative chi-square. The
  form is positive definite by Woodbury, so a negative value is an invalid evaluation. A
  valid value and its gradient are bit-identical to before, and a rejected point has a zero
  gradient. `MarginalOrbitModel` also has a `smoothness_bound` factor beside its AR(1) and
  LSF bounds, which rejects `log_tau - log_eta` above 30, the ratio at which the prior's own
  factorization is documented to round its pivot to zero (defaults start six hyperprior
  sigma inside it). The same fit now ends at z-score rms 0.97 with the period and both
  semi-amplitudes within 0.1 percent of the true values, the potential decreasing at every
  step. There the sign check alone suffices and the bound is never active. The AR(1) noise
  model is not the cause: at correlations of 0.0, 0.1 and 0.5 the same fit converged and
  only the trajectory differed.
- **The correlation stage no longer writes a velocity it did not measure (D63).** An
  archived oracle-tier star whose disentangling had diverged was written with fifteen rows of
  plausible velocities, all at one value. The label fit had put the primary's zero point on
  the bound of its frame-offset scan. Composed with that zero point, the shared search window
  built from the fitted Keplerian contained none of that component's true velocities. The
  coarse pass returned the top node of the range in every epoch. The fine pass moved its
  window out of the range four times and reported a shift five pixels beyond the last point
  at which the chi-square had been evaluated. The update of the window start was applied
  after the last evaluation and added to a position from the previous window. `todcor` now
  reports the position it evaluated, clamps the fine window inside the requested range, and
  flags `at_edge` when the fine minimum is still on the window boundary after the last
  attempt. For a component it could not measure it writes `nan` for the velocity and error
  and keeps every diagnostic of the point it evaluated. The summary counts such epochs as
  "at the search edge, not measured". `Fit.measure_velocities` builds one window per
  template, each offset by that template's zero point relative to the median, so every
  component is searched over the same interval of reported velocity. It raises an error
  naming a component whose zero point puts its window off its own fitted velocities, with the
  two remedies. Here that zero point was 247 km/s from the other component's, on a fit where
  both share one systemic velocity.
- **The pipeline stops at a diverged fit, rejects an unmeasured zero point, and marks a
  failed table (D63).** The same archived star had passed three checks it should have
  failed. `Analysis.z_rms_max` (default 10) now stops a star after the disentangling, before
  the label and velocity stages, with a flag and a recorded failure. Correct fits are near
  1, a fit at a wrong period on the blind route near 2 to 3, and the one divergence observed
  was at 45. The template table is already on disk. A label fit no longer sets the
  templates' zero points in three cases: it is better than neither null, its frame offset is
  within one trial step of its scan's bound, or its components' offsets differ by more than
  the velocity budget on a Keplerian fit (one systemic velocity by construction). The table
  then stays differential with one gamma per component, and `velocities.zero_points` records
  what was adopted or rejected and why. A table whose median R-squared is negative (the
  templates fit worse than no template) or with no usable epoch has
  `velocities.status = "failed"`, its orbit stage is skipped, and its `velocities.rv` begins
  with a `FAILED` line. The "weakly detected" check is unchanged. A declared velocity table
  (`velocities = "file"`) with a non-finite entry for a matched epoch raises an error naming
  the component and epoch, because the free fit has no site for an unmeasured epoch. Zero
  points that leave no search window containing every component (a label fit of a
  broad-lined contact pair put them 95 km/s apart) are dropped with a flag and the
  velocities measured as differential, instead of failing the star.
- **The velocity table no longer loses a component to the exchange step, to shrunk templates
  or to an unconstrained zero point (D63).** Three failures of the third run's tables were
  traced on the archived products with the fit's own templates. First, the exchange of two
  components by the orbit assumed similar spectra at similar light fractions. On a 95/5 pair
  it exchanged 19 of 80 epochs on noise draws of the faint component's velocity and moved
  the primary's semi-amplitude from 5 to 56 percent off. The exchange now runs only when
  the two fractions are within a factor 3 of each other, and the table as measured is kept
  beside the delivered one (`velocities_unexchanged.rv`) whenever an epoch was exchanged.
  Second, the correlation held the light at the declared fractions, on the argument that the
  disentangling recovers each component at that amplitude. Once ML-II has raised a
  component's smoothness precision by an order of magnitude, the recovered lines are
  shallower. A secondary lost a quarter of its depth as tau went from 400 to 2000, and the
  table lost that secondary, at -89 percent. A free amplitude gave -25. The amplitudes are
  therefore fitted freely when any smoothness precision moved more than a factor 10 from its
  start, with a flag stating that the light column is then a template scale and not a
  fraction. Third, a label fit that did not constrain a component's frame offset (posterior
  as wide as the prior, the component mostly noise) no longer sets that component's zero
  point. A secondary that kept 11 percent of its equivalent width had every velocity 46 km/s
  off. That component's velocities stay differential with their own systemic velocity, and
  the other's remain absolute.
- **The conjunction window is centred on the fit's starting conjunction (D63).** One field
  star failed twice at the optimizer's initialisation. The semi-amplitude scan had moved the
  conjunction half a period, because its phase grid contains the antipode that the 41-point
  phase scan cannot sample, which was 672 nats better. The second scan pass had exactly zero
  gain, so the final phase scan did not run. The one-period window was still centred on the
  phase scan's best trial, which put the start on the window's edge, where the
  unconstraining transform is infinite. The fit's window and the sampler's are now centred
  on the conjunction they start from.
- The pipeline's label and bootstrap stages sliced the library in its own wavelength scale
  before converting to the dataset's. On a vacuum dataset this lost about 2.3 A of coverage
  at each edge, and the label fit failed. The library is now converted first.
- The BOSZ 2024 citation is A&A 688, A197, not A171.

### Added

- **`albireo.pipeline` and the `albireo` command line: one command from a list of stars to
  spectra, labels, velocities and orbits (D58).** `albireo init` writes an annotated TOML,
  `albireo run config.toml --jobs 4` runs every star in it, and `albireo demo` runs the same
  pipeline offline on two simulated stars with known injected values. For each star the
  driver reads the epochs and disentangles them (`Disentangler`). It fits Teff / log g /
  [M/H] / *v* sin *i* to the components against a synthetic grid (`match_labels`), measures
  one velocity per component per epoch against those components (`todcor`), and fits the
  Keplerian to the table (`fit_rv_orbit`). It writes `summary.txt`, `result.json`, the
  velocity table as commented ASCII and CSV, the component spectra with their bands, the
  orbit, the labels, `fit.npz`, and six diagnostic figures. A batch adds `results.csv` (one
  row per star), `results.json`, `summary.txt` and `failures.txt`. New names:
  `PipelineConfig`, `StarConfig`, `ComponentConfig`, `Analysis`, `StarResult`, `PipelineRun`,
  `run_pipeline`, `run_star`, `load_config`, `config_from_dict`, `write_config_template`,
  `demo_config`; `plot_phase_scan`; `albireo.simulate.synthetic_library` and
  `library_component` (the toy grid the demo and the tests are built on);
  `albireo.io.read_raw_spectra` and `dataset_from_raw` (the two halves of `read_dataset`, so
  a caller can read the header's resolving power before building the epochs). A console
  script `albireo` (`[project.scripts]`) and `python -m albireo` are added, with no new
  dependencies: the configuration is TOML, read with the standard library.

  Three routes lead into the orbit: a period prior (the Keplerian is inferred from the
  spectra), `period = "search"` (library templates measure a first table, a periodogram finds
  the period, an orbit fitted to the table warm-starts the disentangling through
  `RVOrbit.to_theta()`), or a file of measured velocities (the free per-epoch table). Stars
  run in a spawn-based process pool, with a measured speed-up of 2.0× for four workers and
  2.5× for eight on a batch of eight simulated stars, sub-linear because one star already
  uses several cores. Each worker's XLA and BLAS threads are capped at `cpu_count // jobs` as
  a precaution (the cap made no measurable difference on that benchmark). A failed star is
  recorded and the batch continues. Every caveat a run records is a flag on the report: a
  noise model that does not describe the data, a skipped stage and why, a table whose
  semi-amplitudes disagree with the disentangling. The two rules of the stages are enforced
  unchanged: light fractions are declared, never defaulted, and the wavelength medium is
  declared before a synthetic grid is consulted.

  Components must be declared in order of decreasing mass, and the fit is started with
  K₁ < K₂. A symmetric semi-amplitude prior gives the conjunction scan two equally deep
  minima, the declared assignment and its mirror with the spectra exchanged and rescaled by
  the light ratio, and the first demo run converged to the mirror. The convention is stated
  in the template, and the label stage checks it (a fitted light fraction far from the
  declared one is flagged). `Disentangler`'s `_vector_spec` also dropped every entry's
  `start_at` when a semi-amplitude was declared per component, so every star started at its
  midpoint, which is the symmetric configuration. This is fixed, with a regression test.
  Docs: `docs/api/pipeline.md`, `docs/tutorials/pipeline.md`, `examples/13_pipeline.py`,
  `scripts/pipeline_bench.py`, `docs/benchmarks.md` ("The pipeline in worker processes").

- **`albireo.todcor`: epoch radial velocities for every component by N-dimensional
  correlation (D56).** The two-dimensional correlation of Zucker & Mazeh (1994) is
  generalized to any number of components and to weighted, masked, multi-instrument data by
  writing it as the weighted least-squares fit of shifted templates to the observed pixels.
  This is the model of `docs/math.md` §1.4 with the spectra given rather than marginalized.
  On a uniform grid with uniform weights it is TODCOR: the symmetric expression with the
  light ratio maximized out and the original fixed-ratio one both agree with a NumPy
  transcription to 1e-10 in the tests. New names: `Template` (from a flux array, a library at
  given labels, or a label match), `todcor`, `todcor_batch`, `todcor_surface`,
  `VelocityTable`, `TodcorBatch`, `TodcorSurface`, plus `plot_todcor_surface` and
  `plot_velocity_table`. No dependency is added.

  The shifted, LSF-convolved templates are projected onto each epoch's own pixels (data are
  never resampled, D4), so masks, chip gaps, cosmic rays, per-pixel weights and mixed
  samplings enter through the weights and change no formula. The sub-pixel minimum is
  computed rather than interpolated. The shift operator is linear in the template, so the
  chi-square is an exact quadratic inside each pixel cell, reconstructed from 3^N exact
  evaluations and minimized in closed form. The same linearity bounds the operator's
  pixel-locking ripple at ~0.1/σ_px² pixels, which sets the sampling rule (three pixels per
  LSF sigma) and the warning below two. Errors are the maximum-likelihood ones of Zucker
  (2003), the curvature rescaled by the reduced chi-square, beside the version without the
  rescaling. The covariance's off-diagonal flags blended epochs, a per-component Δχ²
  indicates whether the epoch detects each star, and a minimum at the search edge is
  flagged. Light fractions default to `"global"` (free per epoch, then held at the weighted
  median over well-detected unblended epochs), with `"free"`, fixed fractions, and the
  classic free-scale form available.

  Every table states its zero point. A synthetic template is absolute. The rest frame of a
  disentangled component is not identified (§5.3), and the velocities measured against it are
  reported as differential.

- **`albireo.rvorbit`: orbits from velocity tables (D57), connected to the disentangling
  through the `Disentangler` interface.** `fit_rv_orbit` fits the Keplerian by weighted least
  squares with the JAX Jacobian, using the same Kepler solver and angle conventions as the
  joint model. It fits one systemic velocity per component whenever a component is
  differential (a shared γ across two zero points is absorbed into both K's, which a test
  constructs and checks). `find_period` runs a Lomb-Scargle search on the difference of the
  first two components and returns the aliases. `RVOrbit.to_theta()` returns the elements as
  a disentangling warm start. `Fit.templates()` converts a fit's stellar components into
  templates (upsampled to three pixels per LSF sigma, intrinsic, zero point declared
  unknown), and `Fit.measure_velocities()` measures the epochs against them with the declared
  fractions and the declared LSF per instrument. Docs: `docs/math.md` §10,
  `docs/api/todcor.md`, `docs/api/rvorbit.md`, `docs/tutorials/todcor.md`,
  `examples/12_todcor.py`, `scripts/todcor_bench.py`, `docs/benchmarks.md` ("Epoch
  velocities by N-dimensional correlation").

- **Validation of the label mode on observed spectra of AI Phoenicis, and a resulting
  correction (D55).** `scripts/download_aiphe.py` fetches the 36 archival HARPS spectra, and
  `scripts/aiphe_labels_bench.py` scores a label fit against them.
  `docs/tutorials/aiphe-labels.ipynb` is that run, executed, for reading without the 840 MB
  download. AI Phe was chosen because every quantity the mode produces has an independent
  published value, including a photometrically measured radius ratio that is not given to
  the fit. The primary is recovered at +0.52% in Teff, inside the documented 2–3% target. The
  secondary, at +4.33%, is outside it and is recorded as a miss.

  `LabelMatch.radius_ratio` is new, because on an eclipsing system the shared dilution scalar
  is a measurement that can be compared with an external value.

- **`match_labels(compare=...)` now defaults to `"native"`, not `"matched"` (D55).** Convolving
  both sides with the LSF correlates the residuals over the kernel width while the likelihood
  stays diagonal, so χ² is over-counted by `1/Σk²` (predicted 4.91 on this dataset, measured
  4.26). *v* sin *i* absorbs the mis-specification, and the χ² does not show it. On AI Phe
  `matched` put both components at the floor of their *v* sin *i* prior, where `native`
  returns a physical 2.2 km/s. `matched` remains available and is the right choice only
  together with a residual-covariance model. The closed-loop test could not detect this,
  because its rows pass through neither an LSF nor a disentangling.

- **`fetch_library`: named synthetic grids, downloaded and cached (D52).** The label mode
  could fit any library the user constructed but provided no way to obtain one. Three named
  grids are now registered: `bosz2024-fgk-r20000` (BOSZ 2024 MARCS, Teff 4000–7000 K in 250 K
  steps, log g 3–5, [M/H] −1→+0.5; 455 nodes over 4000–7000 Å), `bosz2024-fgk-rvs` (the same
  nodes in the Gaia RVS window), and `pollux-ob-smc24` (POLLUX CMFGEN, SMC metallicity). New
  names: `fetch_library`, `library_names`, `library_info`, `clear_library_cache`,
  `ingest_bosz`, `ingest_pollux`, `save_library`, `load_library`. Again no dependency is
  added: both grids are plain ASCII, so astropy is not needed.

  BOSZ is built automatically because its URLs on MAST are deterministic. The shards are
  downloaded in parallel, the raw files are kept so another band can be cut without
  downloading again, and the result is sliced to the registered band. POLLUX serves its
  collections through a form that posts to `/download/`, so `ingest_pollux` explains the
  manual step and stops. No parser is shipped, because the file format has not been
  inspected.

  Three details of the archive were checked against it, and two differ from what a careful
  reading of the documentation gives. Teff is not zero-padded in a BOSZ filename (`t6000`,
  not `t06000`), and the atmosphere code changes across the grid: MARCS spherical below
  log g 3.5, plane-parallel at and above it, and ATLAS9 above 8000 K. Both are asserted by
  tests. The third is the medium. A build measures it with `line_core_medium`, which on the
  real grid returns air at a ratio of 256 to 1, and raises an error if the result disagrees
  with the registry.

  A cached build is verified on every load against a digest taken over the arrays rather
  than the file, so the digest is unchanged by a save/load round trip and reproducible
  across machines. The build path reads back what it wrote, so a warm cache and a cold one
  return bit-identical arrays. Otherwise they would differ in the last digits according to
  whether the fluxes had yet been converted to float32.

- **`albireo.match` and `albireo.library`: stellar labels for template selection (D52, D53).**
  A new optional mode fits Teff, log g, [M/H] and *v* sin *i* to disentangled component
  spectra against published synthetic grids, so a component can be rendered as a template
  for measuring epoch radial velocities elsewhere. New names: `SpectralLibrary`,
  `library_interpolator`, `crossval_library`, `line_core_medium`, `BoxInterpolator`,
  `SimplexInterpolator`, `match_labels`, `refit_draws`, `LabelMatch`, `StarLabels`,
  `RadiusRatio`, `ScalarDilution`, `FixedDilution`, plus `Fit.match_labels(...)` on the
  `Disentangler` interface and `rotational_kernel` / `rotational_kernel_traced` /
  `rotational_radius_for` in `albireo.operators`. No dependency is added.

  The scope is narrow by design, and the non-goal in `internal/roadmap.md` is amended
  explicitly. The mode synthesizes no spectrum, includes no line list, solves no radiative
  transfer and fits no abundances. `albireo.handoff` remains the route to GSSP, iSpec,
  Korg.jl and PySME. The mode covers the first half of the analysis: choosing the right
  template, fixing the per-component velocity zero point, and checking an assumed flux ratio.

  Dilution is fitted jointly, through one shared radius ratio with wavelength-dependent
  light fractions written as a softmax over the grids' own continua, so they sum to one at
  every pixel by construction (GSSP's `gssp_binary` parameterization). A wrong assumed light
  ratio is then recovered as dilution and does not bias the temperature, and the
  spectroscopic light ratio becomes a result. The nuisance is additive, because the `k = 0`
  null space of `docs/math.md` §5.1 is confined to the continuum, where a multiplicative
  polynomial is identically zero. Its zeroth term is the unconstrained zero point, fitted
  and reported. Uncertainties are quoted twice, as the Laplace curvature and the spread from
  refitting joint posterior draws, because formal errors on correlated residuals are too
  small by a factor of five to ten. `summary()` prints both with their ratio.

  Interpolation is in flux, never in model atmospheres (0.031% for a cubic against 0.19% for
  atmospheres on BOSZ's own spacing). It uses Catmull-Rom on a complete axis product and
  barycentric interpolation over a Delaunay triangulation where the grid's corners are
  missing for physical reasons. Both reproduce a node bit-for-bit. Whether a learned emulator
  would be more accurate on a given grid has to be measured, which `crossval_library` does.

  `SpectralLibrary.medium` is required and has no default. Air and vacuum differ by ~83 km/s,
  and the upstream documentation is not a reliable source: BOSZ 2017 was vacuum throughout
  and BOSZ 2024 is air above 200 nm, under the same name. `line_core_medium` measures the
  convention from the spectra.

  Docs: `docs/math.md` §9 (the forward model, an extension of the §5.4 summary of
  degeneracies, the treatment of uncertainties, and the accuracy target with citations),
  `docs/api/library.md`, `docs/api/match.md`, `docs/tutorials/labels.md`, and
  `examples/11_labels.py`, an offline closed-loop test that gives the fit wrong light
  fractions and asserts they are recovered as dilution.

- **`docs/tutorials/showcase.ipynb`, an executed notebook showing every main output** on the
  packaged example: `explain()`, the fit summary, the component spectra with their
  uncertainty band, the residual diagnostics, the NUTS posterior (RV-curve draws and a corner
  plot), spectra drawn from the joint posterior, and a sensitivity forecast with six planned
  epochs against the twelve existing ones. It is rendered into the docs site by
  `mkdocs-jupyter`, which had been a declared docs dependency since the D39 block but had
  never been configured. With `execute: false` the site shows the committed outputs, and the
  docs build stays cheap and offline. `scripts/build_showcase_notebook.py` regenerates it (a
  release-time step, ~10 minutes of NUTS), strips kernel-environment noise from the outputs,
  and palette-quantizes the figures (884 → 373 KiB, under the 500 kB pre-commit file-size
  limit).
- **A feature-level comparison with the shift-and-add repository** in
  [`docs/benchmarks.md`](docs/benchmarks.md) ("Feature comparison with the established
  shift-and-add repository"). `TomerShenar/Disentangling_Shift_And_Add` is examined as
  software, since the three-way table already compares the algorithms. The comparison covers
  the orbit treatment (a χ² grid over semi-amplitudes against joint inference), uncertainty
  (χ² contours on K and nothing on the spectra, against posteriors on both), SB3 support,
  masking (part of the published method, so not claimed as an albireo advantage), and
  distribution (unlicensed research scripts against a BSD-3 package). It was written from the
  repository's README and GitHub API metadata only, without opening the source, so
  `scripts/shift_and_add.py` remains a clean-room implementation.

### Fixed

- **`docs/quickstart.md` overlaid the injected truth on the wrong wavelength grid.** A
  simulation's truth is stored on the grid it was generated on. The model is solved on a
  grid that is widened by the velocity budget and the LSF radius and takes its sampling from
  the data. On the packaged example those are 663 and 1074 pixels, so the quickstart's final
  plotting call raised an error inside matplotlib. The page now resamples first, as
  `scripts/build_showcase_notebook.py` already did. `plot_spectra` now validates `truth`
  against the grid as it already validated the mean, so the error names the cause and gives
  the one-line fix instead of reporting a shape mismatch. With two grids of equal length the
  old code would have plotted the truth against the wrong wavelengths without an error.

- `example_info("sb2_sim")` described the packaged example as a "circular SB2". It is
  eccentric by design, e = 0.15, so that the first example a new user runs does not start at
  the `(sqrt(e) cos w, sqrt(e) sin w)` singularity. A comment in the generator script states
  this, and the registry description contradicted it. The description now states e = 0.15.

### Changed

- **Gradient evaluation is ~2x faster, with bit-identical results.** Two exact changes were
  made in `albireo.assembly` after re-profiling. D28's recorded attribution ("92% comb
  probing") had been stale since D28 removed probing from the hot path. The measured split
  is 82% of a gradient in the assembly, whose backward pass cost 3.3x its forward pass.
  1. `_band_accumulate` is a `custom_vjp` for the per-epoch band update. The forward is
     `band + place(f)`, the identity in `band`, but reverse mode transposes the
     `dynamic_update_slice` and the `dynamic_slice` separately and reconstructs that identity
     from three passes over the whole band tensor, once per component pair per epoch. This is
     313 GB of memory traffic to reproduce an input at the benchmark ladder's first row. The
     closed form (`band_bar = out_bar`, `f_bar = ds(out_bar, idx)`) avoids these passes.
  2. The second kernel application in `G = K^T H K` translates columns only, since unlike
     the first it has no row shift, so it is one contraction against a static banded matrix
     instead of `2r+1` read-modify-write passes over the widest image in the assembly. That
     stage is 18x faster with no extra memory. Contracting both applications at once would
     need a 1.9 GB neighbourhood stack, the intermediate that D29 removed.

  Measured at 31,734 model px, SB2, 50 epochs, p = 513: evaluation 2.97 → 2.20 s, gradient
  10.23 → 5.19 s. The log-likelihood and its gradient are bit-identical before and after
  (compared as IEEE-754 hex) for stationary, AR(1) and wavelength-dependent-LSF problems.
  The Hessian changes by 4e-13 relative. Four other candidates were measured and rejected: a
  blocked Cholesky, j-factoring the T-sandwich, `remat=False`, and the custom-VJP
  band-to-block packing named in D28's own record. See
  [`docs/benchmarks.md`](docs/benchmarks.md) ("Second speedup pass", D49) for the numbers,
  including why XLA's fp64 `cholesky` at n = 513, running at 13 GFLOP/s against `matmul`'s
  249, is not fixable by blocking.

  The cost is that albireo no longer has a forward-mode path through the marginal
  likelihood, because `custom_vjp` rejects `jax.jvp`. `forecast._effective_parameters` was
  the package's only forward-mode site (D47 obtains `p_eff` from one directional derivative
  of `log det` in the noise scale). Both `t` and the log-determinant are scalars, so it now
  uses `jax.grad`, which returns the bit-identical number in 0.532 s instead of 0.283 s once
  per forecast, against 1.8x on a gradient run ~2,600 times per posterior. D28 had already
  removed forward mode one stage later, at `_solve_stage`, and second derivatives remain
  available through reverse-over-reverse.

- **The three-way comparison was re-run on one machine, and `scripts/fd3_bench.py` now
  times shift-and-add in a fresh process.** Under the old convention the timing depended on
  the heap state. Timed in-process after the XLA solve (the convention behind both previously
  recorded timings), the identical call is 40–80% slower. XLA's allocations leave the Windows
  CRT heap serving shift-and-add's ~35 KB temporaries through microsecond free-list
  traversals, with allocating ufuncs ~4× slower and their `out=` equivalents unchanged. The
  re-run reproduced all twelve recorded RMS values exactly and re-measured the timings under
  one protocol on a recorded software stack: shift-and-add 0.026 s, albireo 0.059 s, fd3
  0.064 s single-threaded (0.104 s as shipped, with its OpenBLAS running 32 threads). See
  [`docs/benchmarks.md`](docs/benchmarks.md) ("Re-run of the three-code comparison on one
  machine", D50).

### Fixed

- **A detector gap is no longer weighted like data.** `albireo.mask_flux_gaps` zero-weights
  contiguous runs of non-positive flux and warns with the wavelength range. `to_epoch` calls
  it before the spike clip, because a flat run of zeros has no local scatter for a running
  median to detect.
  The defect appeared on observed HARPS spectra of AI Phoenicis, where the two CCDs leave
  32.9 Å of exact zeros at 5304.67–5337.61 Å. Nothing marked them: the pixels are finite,
  there is no quality column, and because HARPS provides no error array the inverse variance
  was estimated from the local scatter, which is small across a flat run of zeros. They had
  median ivar 6398 against 6231 for real pixels, and an analysis window that was 33% detector
  gap disentangled to component spectra with negative flux.
  The rule applies to runs, not to individual non-positive pixels. `RawSpectrum.bad_pixels`
  correctly does not treat zero flux as missing, because one zero can be a saturated core or
  a clipped cosmic ray. Eight consecutive zeros cannot. The defect has the form of D45's
  (zero errors read as infinite precision), one level up, and the same rule applies: the
  reader does not interpret an ambiguous value.

### Added

- **`albireo.handoff`: input files for the atmosphere codes, and posterior draws that
  propagate the uncertainty.** `write_gssp`, `write_ispec` and `export_draws`, with
  [a tutorial](docs/tutorials/downstream.md) and `examples/10_downstream.py`.
  Each format has a requirement whose violation produces no error. iSpec does no unit
  conversion on its text path. Its whole internal scale, atomic line lists included, is
  nanometres, so an ångström value is a factor of ten outside every model grid and is still
  fitted. GSSP infers its synthetic step from the input file ("the step width in wavelength
  that will be used for the calculation of synthetic spectra is computed from the
  observations"), so a log-wavelength grid must be resampled onto an equidistant one. Both
  are regression-tested.
  GSSP has no per-pixel error column, and its configuration has no error path, no S/N entry
  and no weighting entry, so the posterior band cannot be propagated to an effective
  temperature through the file. That requires fitting *N* spectra, which `export_draws`
  writes. `draw_spectra` returns `d_hat + L^-T z` on the vector stacked over all components,
  so draws are correlated across wavelength and across the two stars, and draw *i* of
  component A is the same posterior sample as draw *i* of component B.
  This joint sampling is the reason for the feature. Compared with the established
  procedure of independent per-pixel noise at the band's amplitude (Kiran et al. 2016,
  §3.5), the joint draws give an equivalent-width spread 1.80× and 3.38× larger on the
  packaged example's two components. White noise understates the spread of any integrated
  quantity, and every atmospheric parameter is one. The two components' equivalent widths
  are correlated at −0.992 across draws, against −0.052 under independent noise. This is
  the *k* = 0 exchange mode of D47 appearing in a derived quantity, so the difference
  between the two stars is far better determined than either alone, and independent error
  bars misstate both. See `internal/roadmap.md` Tier 2 item 8 and `docs/api/handoff.md`.

- **`albireo.sensitivity_forecast`: forecasts for planned observations**, for example
  whether twelve more epochs at given phases would separate the two stars.
  `albireo.plan_epochs(template, bjd=...)` builds the planned epochs,
  `sensitivity_forecast(grid, design, orbit=..., baseline=...)` returns what they would add,
  and `albireo.plot_forecast` plots it. This works because the posterior covariance of the
  component spectra, `(Lambda_p + A^T W A)^-1`, contains no flux. Fluxes enter the marginal
  likelihood only through terms that change the posterior mean and the evidence, not the
  covariance. This holds by construction: the precision is assembled directly and the
  right-hand side is never formed. A regression test replaces every flux with noise a
  hundred times the continuum and requires a bit-identical forecast.
  Three exact summaries are returned, each quoted against the same quantity under the prior
  alone, so a design that adds no information is identified. The first is the pointwise
  band. The second is the set of worst-determined modes of the covariance (the spectral
  patterns the design cannot determine), by subspace iteration on the banded factor. The
  third is `p_eff`, the spectral degrees of freedom the data would constrain, from one
  directional derivative of `log det` in the noise scale rather than a stochastic trace
  estimator. Whole designs are ranked by the expected information gain,
  `0.5 (logdet Lambda_t - logdet Lambda_p)`.
  The modes are taken over the pixels the design weights, because a model grid is wider than
  its data. Its margin pixels are constrained by the prior alone and would otherwise be the
  worst-determined direction of every real problem.
  The orbit is not forecast. The Fisher information for a velocity depends on the derivative
  of the component spectrum, so an error bar on `K_2` needs the line depths, which have not
  yet been measured.
  One measured result corrects the interpretation in `docs/math.md` §5.1: the RMS
  differential velocity is the small-*k* expansion, not the objective. A cadence aliased to
  the period maximizes the RMS and is the worst of three plans. Twelve nights at *P*/2 keep
  the RMS at 117.8 km/s, with a blind fraction of 58% of the scale range and a gain of 243
  nats. The same twelve spread over phase lower the RMS to 99.3 km/s and the blind fraction
  to 33%, and gain 375 nats (`examples/08_forecast.py`). See `internal/design.md` D47,
  `docs/math.md` §5.5 and `docs/api/forecast.md`.
- **`Disentangler(velocities=...)`: measured velocities declared in place of an orbit.**
  Exactly one of `orbit=` and `velocities=` is now required. With `velocities=` (an
  `(n_stellar, n_epochs)` km/s table from cross-correlation, shift-and-add, or line splitting
  measured manually) no orbital sites are sampled, and `fit()` returns the free per-epoch RV
  table.
  This removes a circular dependency in the `Disentangler` interface. The table needs a warm
  start (a cold one is 122,000 nats worse, D42), and the only one available was
  `Fit.free_velocities()`, which needs a Keplerian fit and therefore a period. For an
  unsolved system the period comes from the table.
  The declared velocities are a starting point rather than a constraint. The per-component
  zero point stays unidentified, so a systemic offset changes neither the result nor the
  solver's bandwidth, and the velocity budget is derived from the centred table and itemized
  in `explain()`. The mode's one failure case is checked: a declaration whose components
  never separate raises an error, and one that never separates them by more than the LSF
  width warns. `scan()` and `detection_limit()` raise an error without an orbit.
  Warm-started from velocities with 3 km/s of scatter and a 150 km/s systemic offset, the
  table is recovered to 0.096 / 0.070 km/s, the same as D42's Keplerian-warm-started
  0.098 / 0.066. See `internal/design.md` D48.
- **A tutorial that takes a BLOeM SB2 from a survey identifier to disentangled spectra**
  (`docs/tutorials/bloem-sb2.md`), the second item D45 left open. It covers the order of
  steps the survey requires, which no other tutorial needs. With no published period a
  Keplerian fit has no starting point, so the free RV table is fitted first and is the input
  to the periodogram. The work also corrected an error in `examples/06_bloem.py`. Its window
  was documented as "between Hδ and Hγ without either core", but 4000–4300 Å contains Hδ at
  4101.7, and `nebular_windows` places a ±300 km/s window at 4099.7–4107.9 inside it. The
  script's stated reason for not modelling the nebula was therefore false. The window is now
  4120–4300 Å, which contains no nebular line.
- **A worked example for the free per-epoch RV table** (`examples/09_rv_table.py`), the one
  item D42 left open. It demonstrates the two counter-intuitive properties of the mode.
  Shifting one star's velocities by 50 km/s does not change the log-likelihood. The change
  is exactly 0 nats for the relativistic shift against 8.7e-6 for the ordinary one, which is
  its first-order approximation and the reason the centering is done in pixel space. The raw
  Laplace error bars, printed beside the projected ones, are 37.947 km/s on every entry,
  which is `120/√10` and set by the prior, against a measured 0.056–0.065. The example also
  runs the mode's failure case: a cold start converges 122,000 nats worse than the warm one.
- **`albireo.Disentangler`, a declarative front end**, marked experimental because its
  vocabulary is expensive to change once users depend on it. The user declares the system
  (`Star(name, light=...)`, `Telluric()`, `Nebular()`, `Orbit(period=..., k=...)`,
  `LSF.from_resolution(R)`) and it assembles the low-level fit. The packaged example takes
  twelve lines against fifty-nine and recovers the same result.
  `dis.explain()` prints every derivation and `dis.expert()` returns the exact
  `(model, priors, init)` triple, so the low-level API stays the supported interface and
  moving to it takes three lines.
  Four quantities are derived: the solver's velocity budget, the grid margin, the conjunction
  phase and the smoothness hyperparameters. The budget comes from the support of the `k`
  priors, since it must bound what the prior allows rather than the fitted value. The phase
  is found by a scan, because the likelihood is sharply multimodal in phase. The
  hyperparameters are fitted by empirical Bayes and reported per component, with a flag on
  any that did not move from its start. A fifth is structural: a specification such as
  `Between(5.5, 6.5)` holds both its prior and its starting value, so `priors` and `init`
  cannot diverge.
  It does not derive five things for which a default would be a scientific claim: light
  fractions (required per star, must sum to 1, and repeated in an `Assumed, not measured`
  block on every summary), a period search, an undeclared air/vacuum scale when a nebular
  or telluric component is declared (an 83 km/s difference), a velocity budget smaller than
  the priors allow, and the `e = 0` singularity. There, a free eccentricity never starts at
  the origin, and `ecc=Fixed(0.0)` is exact because those sites are not sampled.
  Jitter, AR(1), inferred light fractions and inferred LSF widths are excluded from v1 by
  decision. Also absent, because they could not be delivered reliably, are hierarchical
  triples (`Orbit(outer=...)`), Gauss-Hermite `h3` (it enters the kernel through
  `build_problem`, not through the model the `Disentangler` interface builds), and a lower
  bound on eccentricity (an annulus in the sampled parameterization, not a box). Each raises
  an error rather than being approximated. See `internal/design.md` D46 and
  `docs/api/facade.md`.
- **BLOeM targets are resolved by name**: `albireo.resolve_bloem("1-002")`,
  `albireo.bloem_catalogue(binary_class="SB2")` (the 59 published double-lined systems),
  and `albireo.bloem_spectra(star)` for that star's epochs, ready for `download`. The
  archive does not use the survey's names. BLOeM spectra are stored under
  `obs_collection='GIRAFFE'` with `target_name` set to the Gaia DR3 source id, so the
  cross-match is fetched from VizieR, which uses the same TAP dialect and so needs no new
  dependency.
  Gaia ids are kept as strings because 809 of the 929 are altered by a float64 round trip.
  The default is survey programme `112.25R7`. The same 929 stars are also observed by
  `115.28A9` at *R* = 17000 and 23000 in two other windows, and pooling those with LR02
  under one line-spread function would be wrong. See `internal/design.md` D45.
- **The FITS reader identifies columns by IVOA utype (`TUTYPn`), not by name**, so every
  ESO collection is read with the correct column, unit and wavelength scale. Names remain a
  last-resort fallback for non-ESO files. UCDs would be unsafe as the key, because UVES
  gives its sky-background column the same UCD that HARPS gives its flux column.
  `RawSpectrum` gained `quality`, `specsys`, `v_bary_source`, `err_source`, `columns` and
  `bad_pixels`, so what the reader chose, and why, can be inspected.
- **Quality columns and zero uncertainties now mask pixels**: a flagged pixel, a
  non-finite flux, or a non-positive error gets `ivar = 0` rather than full weight. In
  these pipelines a zero error marks a pixel with no data, not a measurement of infinite
  precision. Flagged pixels are also excluded from the continuum fit. A flag column whose
  convention cannot be determined is ignored. One in which zero never appears does not use
  zero for good pixels: UVES_SQUAD's `STATUS` takes the values `{-5, 1}`, and read with zero
  as good it rejects every pixel of all 467 products in that collection. Columns named only
  `MASK`/`FLAG` are not read, since those names have no agreed polarity. Ignoring a mask is
  recoverable, whereas inverting one keeps exactly the pixels the file rejected.
- `read_dataset(medium=...)`, matching the existing `frame=` override, and an error when
  the files disagree about air vs vacuum.
- **`albireo.archive`**, an ESO Science Archive client: `spectra_query` builds the
  ADQL, `query` runs it, and `download` fetches resumably with a manifest. One query language
  covers FEROS, HARPS, UVES, X-shooter, GIRAFFE and ESPRESSO. It uses only the standard
  library, so finding data adds no dependency. `scripts/download_hr6819.py` is now a thin
  wrapper over it.
  Two checks cover failures the archive does not report. A query that reaches `MAXREC`
  raises an error, because ESO's JSON has no overflow marker and a truncated result is
  indistinguishable from a complete one. A download whose byte count disagrees with
  `Content-Length` is rejected rather than left on disk as if complete.
- **Air vs vacuum is now a declared, validated property**: `EpochData(medium=...)`
  accepts `"air"`, `"vacuum"` or `None` (undeclared, the default and the value of every
  epoch built before this field existed), and `Dataset` rejects a mixture. The offset is a
  nearly constant 83 km/s across the optical, the same order as the semi-amplitudes albireo
  measures, and it does not average out. Mixing declared with undeclared epochs also raises
  an error, because an undeclared medium cannot be checked against a declared one.
- `albireo.air_to_vacuum` / `albireo.vacuum_to_air`, the IAU-adopted Edlen (1966) /
  Birch & Downs (1994) refractivity, evaluated at the vacuum wavenumber so that converted
  line lists agree with published air values. The round trip closes to float64.
- `LogGrid` gained no new state. The conversions are free functions, because the wavelength
  scale of a spectrum is a property of the observation, not of the model grid.
- **A free per-epoch radial-velocity table**: theta site `velocity`
  `(n_stellar, n_epochs)`, which replaces the Keplerian rather than supplementing it
  (mixing the two raises an error). `albireo.relative_velocities` returns the identified
  table, `relative_velocity_errors` its per-epoch error bars, and `keplerian_residuals` the
  model check the mode exists for: free velocities are fitted and then tested against a
  Keplerian. The numpyro model records `velocity_rel` as a deterministic.
  Warm-started from a Keplerian 30% wrong in both semi-amplitudes, the per-epoch RVs are
  recovered to 0.098 / 0.066 km/s, 1/60th of a model pixel, and the Wilson mass ratio to
  0.4%. The mode needs a warm start, as its documentation states. See
  `internal/design.md` D42.
- `albireo.forward.with_shifts`, the pixel-space core of `with_velocities`, which is now
  a wrapper over it. The model's shift composition is exact in pixel shifts, so shifts must
  be added or centered in pixel space.
- `LogGrid.pixels_to_velocity`, the exact inverse of `velocity_to_pixels`.
- **`albireo.calibrate`**: `detection_limit` calibrates the peak of the K₂ scan with an
  empirical null distribution from companion-free trials, a completeness curve from a ladder
  of injected light fractions, and a statement that summarizes them
  (`DetectionLimit.summary()`). `false_alarm_probability` never reports below
  `1 / (n_null + 1)`, and the threshold is defined through it, so the realized false-alarm
  rate over the null trials cannot exceed the nominal one. An interpolating sample quantile
  errs in the other direction. Trials are resimulated through the observed data's own
  operators, so 450 full scans (9,450 marginal solves) take 53 s.
- **`MarginalOrbitModel.log_likelihood_sweep`**: evaluates a grid of trial θ values as one
  batched `lax.map` instead of a Python loop with a device synchronization per point. It ran
  2.0-2.8x faster across problems from 201 to 2,652 model pixels and agreed with the loop to
  1e-12 relative or better. `k2_scan` uses it.
- **`k2_scan(k1_sigma=, k1_nodes=)`**: marginalizes K₁ over a Gaussian prior with a
  Gauss-Hermite rule, applied to the companion and no-companion models so `D` stays a
  ratio of two marginal likelihoods. `K2ScanResult` gains the `(n_k1, n_k2)` surface,
  the quadrature, and `k1_peak`. `k1_sigma=None` is the previous behavior. This corrects
  the failure mode the literature reports. A K₁ 10% high lowered the correlation of the
  recovered companion's line pattern with the injected one from 0.96 to 0.49 while tripling
  `D`, and marginalizing restored the correlation to 0.93. See `internal/design.md` D41.
- **`albireo.forward.with_data` and `albireo.simulate.resimulate`**: the first replaces a
  problem's data term, and the second redraws it from the problem's own forward model.
  Together they form a parametric bootstrap that reuses the rebin operators, pair tables,
  masks and weights, so the calibration costs scan time rather than build time.
- `albireo.plot_detection_limit`: the null distribution with its calibrated threshold
  beside the completeness curve with its limit. An observed peak beyond the histogram's
  range is annotated rather than drawn, since a real companion's `D` can be orders of
  magnitude above the entire null distribution.
- `examples/05_detection_limit.py`: calibrate, detect, and compare the peak with the
  null distribution, with the two-panel figure. Runs in CI.
- **A nebular component** (`build_problem(nebular=True)`, `with_nebular_amplitudes`, θ site
  `log_nebular_amp`): a component at rest in the barycentric frame with a free per-epoch
  amplitude, for the emission lines of the H II region around a massive star. Nebular flux
  is added to the stellar continuum, not taken from it, so the amplitude is outside the
  light-fraction simplex. The amplitude scale is fixed by centering the log-amplitudes
  (`inference.nebular_amplitudes`), and `nebular_v_kms` is a placement convention, not a
  measurement. In the closed-loop test, an unmodelled nebular line leaves the H-beta core
  26% too shallow and the equivalent width 11.5% too small, against 0.14% with the
  component. The orbit is affected more than the spectra. A joint fit without the component
  returns K_2 59% low and drives the eccentricity of a circular orbit to the solver's clip.
  See `internal/design.md` D40.
- **Per-pixel prior strengths**: `SmoothnessPrior(tau_profile=..., eta_profile=...)`, with
  the inferred scalars kept separate so the ML-II fit is unchanged. `albireo.window_profile`,
  `albireo.nebular_windows` and `albireo.NEBULAR_LINES` build the profile that confines a
  nebular component to the lines it can have. Uniform profiles are bit-identical to
  the previous prior.
- `albireo.synthetic_nebular_spectrum` and `simulate_dataset(nebular=, nebular_amplitudes=,
  nebular_v_kms=)`, so the closed-loop tests inject the nebular component through the same
  operator stack as the others. `SimulationTruth` records what was injected.
- `k2_scan(nebular=...)`. A faint-companion scan is a matched filter for a stationary
  residual, and an unmodelled nebular line leaves such a residual.
- **`albireo.results`**: persisting and exporting fits. `save_fit` / `load_fit` round-trip
  `MAPResult`, `K2ScanResult` and `MarginalResult` through `.npz` with a JSON header.
  `to_inference_data` converts a NUTS run for arviz. `write_ascii` writes the disentangled
  spectra with no optional dependency.
- **`albireo.io.write_spectra`**: the disentangled spectra and their uncertainty band as
  FITS or ECSV, with the light fractions and prior hyperparameters recorded in the header
  (the recovered line depths are interpretable only together with them).
- **`albireo.plotting`**: `plot_rv_curve`, `plot_spectra`, `plot_detection`,
  `plot_residual_zscores`, `plot_phase_fold`, `plot_lsf`, `plot_light_fractions`,
  `plot_corner`. The first three were previously defined inside `examples/`, which now
  call the module. It needs the new `plots` extra (matplotlib, arviz), imported lazily as
  `albireo.io` imports astropy.
- **`albireo.examples`**: `load_example`, `example_info`, `example_names`,
  `clear_example_cache`. A simulated SB2 is included in the wheel, so the quickstart runs
  offline, in CI, and in a fresh notebook with no download and no astropy. Larger examples
  download to a platform cache directory and are checksum-verified.
- `examples/00_quickstart.py` and `docs/quickstart.md`: load, fit, plot, export in about
  twenty seconds. Runs in CI.
- `data_residual_zscores(..., per_epoch=True)` returns one array per epoch. The lag-1
  autocorrelation diagnostic needs this, because it is defined only within a single
  exposure.
- `internal/roadmap.md`: the development plan, its order, and the non-goals, with the
  evidence from the software ecosystem behind the ordering.
- Rendered API reference (mkdocstrings) and a GitHub Pages deployment workflow. The
  `mkdocstrings` and `mkdocs-jupyter` dependencies were declared but had never been
  configured: `mkdocs.yml` had no `plugins:` block, so neither had any effect.
- `py.typed`, so downstream type-checkers use the annotations the package already had.
- `tests/conftest.py` with `slow` / `network` / `gpu` markers and matching `--no-slow`,
  `--no-network`, `--no-gpu` opt-outs. A bare `pytest` still runs every test. CI's fast
  matrix deselects the acceptance tests (14 min → 6 min), and a separate job runs them with
  coverage.
- A `bare-install` CI job that installs with no extras and checks that `import albireo`
  works, so the optional-dependency guards are tested.
- `CHANGELOG.md`, `docs/citing.md`, `internal/releasing.md`, `codemeta.json`, and a tag-driven
  release workflow using PyPI trusted publishing.

### Changed

- `K2ScanResult` gained `k1_grid`, `k1_log_weights`, `log_likelihood_grid`,
  `log_likelihood_null_grid` and `k1_peak`, all with defaults, and `save_fit` /
  `load_fit` round-trip them. A scan file written before the K₁ marginalization existed
  still loads, and its `k1_peak` reads back as NaN.
- `k2_scan` at fixed K₁ is no longer bit-identical to its pre-vectorization version.
  Batching the trials into one `lax.map` re-associates the linear algebra, which changes
  the log-likelihoods by ~1e-9 out of ~1e5. This is a floating-point effect, not a change of
  method: the quadrature at `k1_sigma=None` is exactly the identity.
- `with_velocities` and `with_light_fractions` now pass the trailing non-stellar columns
  (telluric, nebular) through unchanged instead of rebuilding the telluric one. The result
  is identical, since those velocity laws depend only on the frame and each epoch's
  `v_bary`, both fixed at build time, and it is correct for any number of non-stellar
  components.
- `marginal_loglikelihood` names the components a mismatched prior is missing, and rejects a
  prior whose per-pixel profiles were built for a different grid.
- The package version is now single-sourced from `src/albireo/__init__.py` via
  `[tool.hatch.version]`, and `pyproject.toml` declares it dynamic. `tests/test_metadata.py`
  asserts that the installed distribution metadata and `CITATION.cff` agree with it.
- `K2ScanResult.model` may now be None, so a scan result can be read back from disk without
  its `MarginalOrbitModel`. `k2_scan` already constructed the result by keyword, so this
  widens the type without changing the signature.
- `MarginalResult.precision` may likewise be None, and `MarginalResult.chol` then raises a
  `ValueError` stating this and where the stored values are, instead of failing inside the
  block Cholesky.
- Markdown is excluded from `ruff format`. Recent ruff formats Python code blocks inside
  `.md` files, which would remove the comment alignment used throughout the docs.

### Fixed

- **The window of `examples/06_bloem.py` contained the Hδ core it was documented to avoid.**
  Its region was described as "between Hδ (4102) and Hγ (4340) without either core", but
  4000–4300 Å contains Hδ at 4101.7, and `nebular_windows` puts a ±300 km/s window at
  4099.7–4107.9 inside it. The script's stated reason for not modelling the nebula was that
  the region avoided the Balmer cores, so that reason was false. The region is now
  4120–4300 Å (`nebular_windows` returns nothing there), and the docstring explains why the
  blue edge cannot be moved.
- **`EpochData.medium` was not set on an epoch read from a file.** `to_epoch` built the
  `EpochData` without it. `preprocess._replace`, which every trimming and masking helper
  calls, also omitted it, so even a manually set value was dropped by the first
  `select_region`. The D43 check against combining air with vacuum therefore could not
  trigger for epochs read from files, and the offset it detects is 83 km/s.
- `ESO TEL TARG ALPHA` / `DELTA` are packed sexagesimal (`HHMMSS.sss`, `±DDMMSS.sss`) and
  were read as degrees. `SkyCoord` accepts any real number as a right ascension, so the
  position wrapped modulo 360 to a plausible value, and the barycentric corrections were
  wrong by minutes and by km/s with no error or warning. These keywords are read only when
  `RA`/`DEC` are absent.
- `MJD-OBS + EXPTIME/2` was used as the mid-exposure time for coadded products, where it
  can be a week early. `MJD-END` is now preferred, and the `EXPTIME` fallback raises an
  error when `TELAPSE` shows the product spans more than one exposure. A product with
  `M_EPOCH=True` warns that it has no single epoch.
- A placeholder `v_bary = 0.0` (no keyword, too little header to compute one) now warns.
  Zero is a legitimate value here, so setting it without a warning was wrong.
- An all-NaN `ERR` column, which FEROS and HARPS products contain, now warns that the
  weights are albireo's assumption rather than the archive's. Previously it was dropped
  without notice.
- `read_spectrum` no longer requires a time keyword when `bjd=` is supplied, as the error
  message for a missing keyword already instructed. Among other things, this allows albireo
  to read back the component spectra it writes.
- The spectrum HDU is chosen by utype and `EXTNAME` before column names, so a short
  calibration table earlier in the file is no longer selected over the spectrum.
- `"R"` was removed from the resolving-power keywords. A one-character card can match
  unrelated keywords, and `R = 3.7` implied a 34,000 km/s line-spread function without a
  warning.
- `read_dataset` on a directory also reads `.fit` and `.fits.gz` files, and no longer passes
  `instrument=`/`frame=` twice when they appear in `read_kwargs`.
- A spectral axis declared as `em.freq` or `em.energy` is rejected rather than read as
  Angstrom. `SpectralAxis` names those axes too, and the result would be a strictly
  increasing but wrong wavelength grid.
- A table declared as the spectrum but with no identifiable flux column now raises an error
  stating this. Previously the image reader was used instead and returned other data in the
  same file as the science spectrum.
- The all-bad-pixel check is evaluated on the pixels that remain after the final trim, not
  on the wider slice used for the continuum fit. Previously a region of bad pixels beside a
  pad of good ones produced a zero-weight epoch, which the solver accepts without warning.
- An error column sharing neither the namespace nor the unit of the chosen flux is rejected
  with a warning, because it is the error on the file's other flux column.
- `HELICORR` and `ESO QC VRAD HELICOR` are recognized as heliocentric. Neither contains
  `HELIO`, so both were treated as barycentric without the warning.
- Passing `frame=` suppresses the warnings whose only purpose is to advise passing `frame=`.
  A `SPECSYS` that albireo does not model (`LSRK`, `GEOCENT`) is now reported as such and
  kept on `RawSpectrum.specsys`. Previously it was discarded and described as a missing
  keyword.
- A TAP service's ADQL error is reported with the message from the VOTable body it returns,
  instead of a bare `HTTP Error 400`.
- `pytest` (the console script) failed to collect `tests/test_ar1.py`,
  `tests/test_lsf_h3.py`, and `tests/test_lsf_varying.py`. They import shared helpers via
  `from tests.test_likelihood import ...`, which resolves only when the repository root is
  on `sys.path`. That is the case under `python -m pytest` but not under `pytest`, the form
  CI runs. Declaring `pythonpath = ["."]` in the pytest configuration makes the two
  invocations equivalent. Nothing had been pushed, so CI had never run and the failure had
  not been observed.
- `ruff format --check .` did not pass on a clean checkout (`src/albireo/solver.py`), which
  would have failed the lint job on the first push.
- `mypy src/albireo` reported two errors on a clean checkout: `examples.load_example`
  passed a list where `Dataset` is annotated for a tuple, and `results.save_fit`'s
  `**arrays` splat is not expressible against numpy's `savez` stub. Both would likewise
  have failed the lint job on the first push. The cause is that of the two defects D39
  records: nothing has ever been pushed, so CI has never run.
- `MarginalOrbitModel` rebuilt the spectral prior from the sampled `log_tau`/`log_eta` and
  dropped any per-pixel profiles, so a window-confined component lost its confinement
  without notice as soon as ML-II was enabled. The profiles are structure and the scalars
  are hyperparameters, so the merge now replaces only the scalars. The defect was found
  before the feature was released. With the confinement applied, the closed-loop K_2 error
  goes from 2.6% to 0.29%.
