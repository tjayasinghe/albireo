# Changelog

All notable changes to albireo are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and albireo uses
[semantic versioning](https://semver.org/spec/v2.0.0.html) — with the caveat that while the
version is below 1.0 the public API may change in any release.

This file records *what changed*. The reasons live elsewhere and are worth following:
[`docs/design.md`](docs/design.md) §2 is the decision ledger, and
[`docs/benchmarks.md`](docs/benchmarks.md) is the validation and performance record.

## [Unreleased]

### Added

- **`albireo.sensitivity_forecast`, which answers an observing question before the
  observation** — "will twelve more epochs at these phases separate the two stars?"
  `albireo.plan_epochs(template, bjd=...)` builds the epochs of a night that has not
  happened, `sensitivity_forecast(grid, design, orbit=..., baseline=...)` returns what
  they would buy, and `albireo.plot_forecast` draws it. This works because the posterior
  covariance of the component spectra, `(Lambda_p + A^T W A)^-1`, **has no flux in it** —
  fluxes reach the marginal likelihood only through terms that move the posterior mean and
  the evidence, never the covariance. The claim is structural, not asserted: the precision
  is assembled directly and the right-hand side is never formed, and a regression test
  replaces every flux with noise a hundred times the continuum and requires the forecast
  back bit-identical.
  Three summaries, each exact and each quoted against the same quantity under the prior
  alone, so a design that is learning nothing says so: the **pointwise band**, the
  **worst-determined modes** of the covariance (the spectral patterns the design cannot
  pin down, by subspace iteration on the banded factor), and **`p_eff`**, the spectral
  degrees of freedom the data would actually constrain — from one directional derivative
  of `log det` in the noise scale rather than a stochastic trace estimator. Whole designs
  are ranked by the expected information gain, `0.5 (logdet Lambda_t - logdet Lambda_p)`.
  The modes are taken over the pixels the design actually weights, because a model grid is
  deliberately wider than its data and those margin pixels — prior-only by construction —
  are otherwise the worst-determined direction of every real problem.
  It deliberately does **not** forecast the orbit: the Fisher information for a velocity
  runs through the derivative of the component spectrum, so an error bar on `K_2` needs the
  line depths, which is what has not been measured yet.
  One measured result corrects `docs/math.md` §5.1's own reading: the RMS differential
  velocity is the small-*k* expansion, not the objective. A cadence aliased to the period
  maximizes it and is the *worst* of three plans — twelve nights at *P*/2 hold the RMS at
  117.8 km/s, stay blind over 58% of the scale range and are worth 243 nats, while the same
  twelve spread over phase lower the RMS to 99.3 km/s and 33% and are worth 375
  (`examples/08_forecast.py`). See `docs/design.md` D47, `docs/math.md` §5.5 and
  `docs/api/forecast.md`.
- **`Disentangler(velocities=...)` — declare the velocities you measured when there is no
  orbit to declare.** Exactly one of `orbit=` and `velocities=` is now required. With
  `velocities=` (an `(n_stellar, n_epochs)` km/s table from cross-correlation, shift-and-add,
  or line splitting measured by hand) no orbital sites are sampled at all, and `fit()`
  returns the free per-epoch RV table directly.
  This closes a circle the façade could not previously escape: the table needs a warm start,
  a cold one is 122,000 nats worse (D42), and the only warm start on offer was
  `Fit.free_velocities()` — which needs a Keplerian fit, which needs a period. For an
  unsolved system, the table is what *produces* the period.
  The declared velocities are a starting point, not a constraint: the per-component zero
  point stays unidentified, so a systemic offset changes neither the answer nor the solver's
  bandwidth — the velocity budget is derived from the *centred* table, itemized in
  `explain()`. The mode's one failure is refused rather than discovered: a declaration whose
  components never separate raises, and one that never resolves them beyond the LSF width
  warns. `scan()` and `detection_limit()` refuse without an orbit.
  Measured: warm-started from velocities carrying 3 km/s of scatter *and* a 150 km/s systemic
  offset, the recovered table lands at 0.096 / 0.070 km/s — the same as D42's
  Keplerian-warm-started 0.098 / 0.066. See `docs/design.md` D48.
- **A tutorial that takes a BLOeM SB2 from a survey identifier to disentangled spectra**
  (`docs/tutorials/bloem-sb2.md`) — the second gap D45 left open. It covers the ordering the
  survey forces and that none of the other tutorials needs: with no published period there
  is nothing to warm-start a Keplerian from, so the free RV table comes *first* and supplies
  the periodogram. Writing it found a real error in `examples/06_bloem.py`: its window was
  documented as sitting "between Hδ and Hγ without either core" and did not — 4000–4300 Å
  contains Hδ at 4101.7, and `nebular_windows` places a ±300 km/s window at 4099.7–4107.9
  inside it, so the script's stated reason for not modelling the nebula was false. The window
  is now **4120–4300 Å**, which contains no nebular line at all.
- **A worked example for the free per-epoch RV table** (`examples/09_rv_table.py`), the one
  gap D42 left open. It is built around the two properties of the mode that are
  counter-intuitive rather than around the API: it shifts one star's velocities by 50 km/s
  and shows the log-likelihood does not move (exactly 0 nats for the relativistic shift
  against 8.7e-6 for the ordinary one, which is only its first-order approximation — the
  reason the centering lives in pixel space), and it prints the raw Laplace error bars
  beside the projected ones, 37.947 km/s on *every* entry because that is `120/√10`, the
  prior, against a measured 0.056–0.065. It also shows the mode's real failure rather than
  describing it: a cold start lands 122,000 nats worse than the warm one.
- **`albireo.Disentangler`, a declarative front end** — *experimental*, because a
  vocabulary is expensive to change once people depend on it. Declare the system
  (`Star(name, light=...)`, `Telluric()`, `Nebular()`, `Orbit(period=..., k=...)`,
  `LSF.from_resolution(R)`) and it emits the expert path. Twelve lines against
  fifty-nine on the packaged example, recovering the same answer.
  It is a **compiler, not a shortcut**: `dis.explain()` prints every derivation and
  `dis.expert()` returns the exact `(model, priors, init)` triple, so the low-level API
  stays the supported surface and dropping down costs three lines.
  Four things are derived rather than typed — the solver's **velocity budget** (from the
  `k` priors' own support, since it must bound what the *prior* allows, not what the answer
  turns out to be), the **grid margin**, the **conjunction phase** (by a scan, because the
  likelihood is sharply multimodal in phase), and the **smoothness hyperparameters** by
  empirical Bayes, reported per component with a flag on any that did not move from its
  start. A fifth is structural: a spec such as `Between(5.5, 6.5)` carries both its prior
  and its starting value, so `priors` and `init` cannot drift apart.
  It **refuses** to derive five things where a default would be a scientific claim: light
  fractions (required per star, must sum to 1, and repeated in an `Assumed, not measured`
  block on every summary), a period *search*, an undeclared air/vacuum scale when a nebular
  or telluric component makes it an 83 km/s question, a velocity budget smaller than the
  priors reach, and the `e = 0` singularity — a free eccentricity never starts at the
  origin and `ecc=Fixed(0.0)` is exact, because those sites are not sampled at all.
  Not in v1, deliberately: jitter, AR(1), inferred light fractions and inferred LSF widths.
  Also absent because they could not be delivered honestly, each raising rather than
  approximating: hierarchical triples (`Orbit(outer=...)`), Gauss-Hermite `h3` (it reaches
  the kernel through `build_problem`, not through the model the façade builds), and a
  lower bound on eccentricity (an annulus in the sampled parameterization, not a box).
  See `docs/design.md` D46 and `docs/api/facade.md`.
- **BLOeM targets resolve by name** — `albireo.resolve_bloem("1-002")`,
  `albireo.bloem_catalogue(binary_class="SB2")` (the 59 published double-lined systems),
  and `albireo.bloem_spectra(star)` for that star's epochs, ready for `download`. The
  archive does not know the survey's names: BLOeM spectra sit under
  `obs_collection='GIRAFFE'` with `target_name` set to the *Gaia DR3 source id*, so the
  cross-match is fetched from VizieR — the same TAP dialect, hence still no new dependency.
  Gaia ids are kept as strings because 809 of the 929 do not survive a float64 round trip.
  Defaults to survey programme `112.25R7`: the same 929 stars are also observed by
  `115.28A9` at *R* = 17000 and 23000 in two other windows, and pooling those with LR02
  under one line-spread function would be wrong. See `docs/design.md` D45.
- **The FITS reader dispatches on IVOA utypes (`TUTYPn`), not column names** — so every ESO
  collection reads with the right column, the right unit and the right wavelength scale.
  Names remain a last-resort fallback for non-ESO files. Keying on UCDs instead would have
  been actively unsafe: UVES gives its sky-background column the same UCD HARPS gives its
  flux column. `RawSpectrum` gained `quality`, `specsys`, `v_bary_source`, `err_source`,
  `columns` and `bad_pixels`, so what the reader chose and why is inspectable.
- **Quality columns are honoured, and so is a zero uncertainty** — a flagged pixel, a
  non-finite flux, or a non-positive error now gets `ivar = 0` rather than full weight. A
  zero error is how these pipelines write "nothing here", not a measurement of infinite
  precision. Flagged pixels are also excluded from the continuum fit. A flag column whose
  convention cannot be read is **ignored rather than guessed at**: one in which zero never
  appears is not using "zero is good" (UVES_SQUAD's `STATUS` runs `{-5, 1}`, and taken at
  face value it condemns every pixel of all 467 products in that collection), and columns
  named only `MASK`/`FLAG` are not read at all, since those names carry no agreed polarity.
  Losing a mask is recoverable; inverting one keeps exactly the pixels the file rejected.
- `read_dataset(medium=...)`, matching the existing `frame=` override, and a refusal to
  combine files that disagree about air vs vacuum.
- **`albireo.archive`** — an ESO Science Archive client: `spectra_query` builds the
  ADQL, `query` runs it, `download` fetches resumably with a manifest. One query language
  for FEROS, HARPS, UVES, X-shooter, GIRAFFE and ESPRESSO. Stdlib only, so finding data
  costs no dependency. `scripts/download_hr6819.py` is now a thin wrapper over it.
  Two guards exist because the archive is silent where it matters: a query that hits
  `MAXREC` raises (ESO's JSON carries no overflow marker, so a truncated result is
  indistinguishable from a complete one), and a download whose byte count disagrees with
  `Content-Length` is rejected rather than left on disk looking complete.
- **Air vs vacuum is now a declared, validated property** — `EpochData(medium=...)`
  accepts `"air"`, `"vacuum"` or `None` (undeclared, the default and what every epoch
  built before this field existed means), and `Dataset` **refuses a mixture** rather than
  picking one. The offset is a nearly constant 83 km/s across the optical — the same
  order as the semi-amplitudes albireo measures, and it does not average out. Mixing
  declared with undeclared raises too: "unknown" cannot be checked against "air".
- `albireo.air_to_vacuum` / `albireo.vacuum_to_air` — the IAU-adopted Edlen (1966) /
  Birch & Downs (1994) refractivity, evaluated at the vacuum wavenumber so that converted
  line lists agree with published air values. The round trip closes to float64.
- `LogGrid` gained no new state: the conversions are free functions, because which scale
  a *spectrum* is on is a property of the observation, not of the model grid.
- **A free per-epoch radial-velocity table** — theta site `velocity`
  `(n_stellar, n_epochs)`, which *replaces* the Keplerian rather than supplementing it
  (mixing the two raises). `albireo.relative_velocities` reports the identified table,
  `relative_velocity_errors` its honest per-epoch bars, and `keplerian_residuals` the
  model check the mode exists for: fit free velocities, then ask whether a Keplerian
  still threads them. The numpyro model records `velocity_rel` as a deterministic.
  Warm-started from a Keplerian 30% wrong in both semi-amplitudes, per-epoch RVs recover
  to 0.098 / 0.066 km/s — 1/60th of a model pixel — and the Wilson mass ratio to 0.4%.
  See `docs/design.md` D42; the mode needs a warm start, and says so.
- `albireo.forward.with_shifts` — the pixel-space core of `with_velocities`, which is now
  a wrapper over it. Pixel shifts are where the model's shift composition is exact, so
  anything that adds or centers shifts has to work there.
- `LogGrid.pixels_to_velocity` — the exact inverse of `velocity_to_pixels`.
- **`albireo.calibrate`** — `detection_limit` turns the K₂ scan's peak into a claim:
  an empirical null distribution from companion-free trials, a completeness curve from
  a ladder of injected light fractions, and the sentence they add up to
  (`DetectionLimit.summary()`). `false_alarm_probability` never reports below
  `1 / (n_null + 1)`, and the threshold is defined *through* it, so the realized
  false-alarm rate over the null trials can never exceed the nominal one — an
  interpolating sample quantile errs the other way. Trials resimulate through the
  observed data's own operators, so 450 full scans (9,450 marginal solves) take 53 s.
- **`MarginalOrbitModel.log_likelihood_sweep`** — a grid of trial θ values as one
  batched `lax.map` instead of a Python loop with a device synchronization per point.
  Measured **2.0-2.8x** across problems from 201 to 2,652 model pixels, agreeing with
  the loop to 1e-12 relative or better. `k2_scan` uses it.
- **`k2_scan(k1_sigma=, k1_nodes=)`** — marginalize K₁ over a Gaussian prior with a
  Gauss-Hermite rule, applied to the companion *and* no-companion models so `D` stays a
  ratio of two marginal likelihoods. `K2ScanResult` gains the `(n_k1, n_k2)` surface,
  the quadrature, and `k1_peak`. `k1_sigma=None` is the previous behavior. This is the
  fix for the failure mode the literature reports: a K₁ 10% high took the recovered
  companion's line pattern from 0.96 correlation with truth to 0.49 *while tripling*
  `D`, and marginalizing restored it to 0.93. See `docs/design.md` D41.
- **`albireo.forward.with_data`** and **`albireo.simulate.resimulate`** — swap a
  problem's data term, and redraw it from the problem's own forward model. Together they
  are a parametric bootstrap that reuses the rebin operators, pair tables, masks and
  weights, which is what makes the calibration cost scan time rather than build time.
- `albireo.plot_detection_limit` — the null distribution with its calibrated threshold
  beside the completeness curve with its limit. An observed peak that falls off the
  histogram is annotated rather than drawn, since a real companion's `D` can sit orders
  of magnitude above the entire null distribution.
- `examples/05_detection_limit.py` — calibrate, detect, and read the peak against the
  null distribution, with the two-panel figure. Runs in CI.
- **A nebular component** (`build_problem(nebular=True)`, `with_nebular_amplitudes`, θ site
  `log_nebular_amp`) — a component at rest in the *barycentric* frame with a free per-epoch
  amplitude, for the emission lines of the H II region a massive star sits in. Nebular flux
  is added on top of the stellar continuum rather than taken out of it, so the amplitude is
  outside the light-fraction simplex; the amplitude scale is pinned by centering the
  log-amplitudes (`inference.nebular_amplitudes`), and `nebular_v_kms` is a placement
  convention rather than a measurement. Measured in the closed loop: leaving a nebular line
  unmodelled leaves the H-beta core 26% too shallow and costs 11.5% of the equivalent
  width; with the component, 0.14%. The orbit is affected more than the spectra — a
  nebula-blind joint fit returns K_2 59% low, with the eccentricity of a circular orbit
  driven to the solver's clip. See `docs/design.md` D40.
- **Per-pixel prior strengths** — `SmoothnessPrior(tau_profile=..., eta_profile=...)`, with
  the inferred scalars kept separate so the ML-II fit is unchanged. `albireo.window_profile`,
  `albireo.nebular_windows` and `albireo.NEBULAR_LINES` build the profile that confines a
  nebular component to the lines it can actually have. Uniform profiles are bit-identical to
  the previous prior.
- `albireo.synthetic_nebular_spectrum` and `simulate_dataset(nebular=, nebular_amplitudes=,
  nebular_v_kms=)`, so the closed-loop tests inject through the same operator stack as
  everything else. `SimulationTruth` records what was injected.
- `k2_scan(nebular=...)`. A faint-companion scan is a matched filter for a stationary
  residual, which is exactly what an unmodelled nebular line looks like.
- **`albireo.results`** — persisting and exporting fits. `save_fit` / `load_fit` round-trip
  `MAPResult`, `K2ScanResult` and `MarginalResult` through `.npz` with a JSON header;
  `to_inference_data` converts a NUTS run for arviz; `write_ascii` writes the disentangled
  spectra with no optional dependency at all.
- **`albireo.io.write_spectra`** — the disentangled spectra and their uncertainty band as
  FITS or ECSV, with the light fractions and prior hyperparameters recorded in the header
  (the recovered line depths are only interpretable next to them).
- **`albireo.plotting`** — `plot_rv_curve`, `plot_spectra`, `plot_detection`,
  `plot_residual_zscores`, `plot_phase_fold`, `plot_lsf`, `plot_light_fractions`,
  `plot_corner`. The first three were previously stranded inside `examples/`, which now
  call the module instead. Needs the new `plots` extra (matplotlib, arviz), lazily imported
  exactly as `albireo.io` imports astropy.
- **`albireo.examples`** — `load_example`, `example_info`, `example_names`,
  `clear_example_cache`. A simulated SB2 ships inside the wheel, so the quickstart runs
  offline, in CI, and in a fresh notebook with no download and no astropy. Larger examples
  download to a platform cache directory and are checksum-verified.
- `examples/00_quickstart.py` and `docs/quickstart.md` — load, fit, plot, export in about
  twenty seconds. Runs in CI.
- `data_residual_zscores(..., per_epoch=True)` returns one array per epoch, which is what
  makes the lag-1 autocorrelation diagnostic meaningful (it is only defined within a single
  exposure).
- `docs/roadmap.md` — where albireo is going, in what order, and the non-goals, with the
  ecosystem evidence behind the ordering.
- Rendered API reference (mkdocstrings) and a GitHub Pages deployment workflow. The
  `mkdocstrings` and `mkdocs-jupyter` dependencies were declared but had never been
  configured — `mkdocs.yml` had no `plugins:` block, so neither did anything.
- `py.typed`, so downstream type-checkers see the annotations the package already had.
- `tests/conftest.py` with `slow` / `network` / `gpu` markers and matching `--no-slow`,
  `--no-network`, `--no-gpu` opt-outs. A bare `pytest` still runs everything; CI's fast
  matrix deselects the gates (14 min → 6 min) and a separate job runs them with coverage.
- A `bare-install` CI job that installs with no extras and checks that `import albireo`
  works — the job that makes the optional-dependency guards real rather than intended.
- `CHANGELOG.md`, `docs/citing.md`, `docs/releasing.md`, `codemeta.json`, and a tag-driven
  release workflow using PyPI trusted publishing.

### Changed

- `K2ScanResult` gained `k1_grid`, `k1_log_weights`, `log_likelihood_grid`,
  `log_likelihood_null_grid` and `k1_peak`, all with defaults, and `save_fit` /
  `load_fit` round-trip them. A scan file written before the K₁ marginalization existed
  still loads; its `k1_peak` reads back as NaN.
- `k2_scan` is no longer bit-identical to its pre-vectorization self at fixed K₁:
  batching the trials into one `lax.map` re-associates the linear algebra, which moves
  the log-likelihoods by ~1e-9 out of ~1e5. Floating point, not method — the
  quadrature at `k1_sigma=None` is exactly the identity.
- `with_velocities` and `with_light_fractions` now carry the trailing non-stellar columns
  (telluric, nebular) through unchanged instead of rebuilding the telluric one. The result
  is identical — those velocity laws depend only on the frame and each epoch's `v_bary`,
  both fixed at build time — and it is correct for any number of non-stellar components.
- `marginal_loglikelihood` names the components a mismatched prior is missing, and rejects a
  prior whose per-pixel profiles were built for a different grid.
- The package version is now single-sourced from `src/albireo/__init__.py` via
  `[tool.hatch.version]`; `pyproject.toml` declares it dynamic. `tests/test_metadata.py`
  asserts that the installed distribution metadata and `CITATION.cff` agree with it.
- `K2ScanResult.model` may now be None, so a scan result can be read back from disk without
  its `MarginalOrbitModel`. `k2_scan` already constructed the result by keyword, so this is
  a widening rather than a signature change.
- `MarginalResult.precision` may likewise be None, and `MarginalResult.chol` now raises a
  `ValueError` saying so — and saying where the numbers went — rather than failing inside
  the block Cholesky.
- Markdown is excluded from `ruff format`. Recent ruff formats Python code blocks inside
  `.md` files, which would collapse the deliberate comment alignment throughout the docs.

### Fixed

- **`examples/06_bloem.py` used a window it documented as avoiding.** Its region was
  described as sitting "between Hδ (4102) and Hγ (4340) without either core", but
  4000–4300 Å contains Hδ at 4101.7 — and `nebular_windows` puts a ±300 km/s window at
  4099.7–4107.9 inside it. Since the script's stated reason for not modelling the nebula
  was that it avoided the Balmer cores, the claim was load-bearing and false. The region
  is now 4120–4300 Å (`nebular_windows` returns nothing there) and the docstring explains
  why the blue edge cannot move rather than asserting the outcome.
- **`EpochData.medium` never reached an epoch read from a file.** `to_epoch` built the
  `EpochData` without it, and `preprocess._replace` — which every trimming and masking
  helper goes through — omitted it, so even a hand-set value was dropped by the first
  `select_region`. D43's guard against combining air with vacuum therefore could not fire
  on the path that matters, and the offset it exists to catch is 83 km/s.
- `ESO TEL TARG ALPHA` / `DELTA` are packed sexagesimal (`HHMMSS.sss`, `±DDMMSS.sss`) and
  were read as degrees. `SkyCoord` accepts any real number as a right ascension, so the
  position wrapped modulo 360 to somewhere plausible and the barycentric corrections were
  wrong by minutes and by km/s, silently. Only reached when `RA`/`DEC` are absent.
- `MJD-OBS + EXPTIME/2` was used as the mid-exposure time for coadded products, where it
  can be a week early. `MJD-END` is now preferred, and the `EXPTIME` fallback is refused
  when `TELAPSE` says the product spans more than one exposure. A product with
  `M_EPOCH=True` warns that it has no single epoch at all.
- A fabricated `v_bary = 0.0` (no keyword, too little header to compute one) now warns.
  Zero is a legitimate value here, which is exactly why inventing it silently was wrong.
- An all-NaN `ERR` column — FEROS and HARPS ship one — now warns that the weights have
  become albireo's assumption rather than the archive's, instead of being dropped in
  silence.
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
  Angstrom — `SpectralAxis` names those axes too, and the result would be a strictly
  increasing, entirely wrong wavelength grid.
- A table that declares itself the spectrum but has no identifiable flux column now says so,
  instead of falling through to the image reader and returning something else in the same
  file as the science spectrum.
- The all-bad-pixel guard is evaluated on the pixels that survive the final trim rather than
  on the wider slice used for the continuum fit; a dead region beside a live pad previously
  produced a zero-weight epoch that the solver accepts in silence.
- An error column sharing neither the namespace nor the unit of the chosen flux is rejected
  with a warning: it is the error on the file's *other* flux column.
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
  on `sys.path`. That happened under `python -m pytest` but not under `pytest` — the form
  CI runs. Declaring `pythonpath = ["."]` in the pytest configuration makes the two
  invocations equivalent. Since nothing had ever been pushed, CI had never run and this had
  never been observed.
- `ruff format --check .` did not pass on a clean checkout (`src/albireo/solver.py`), which
  would have failed the lint job on the first push.
- `mypy src/albireo` reported two errors on a clean checkout — `examples.load_example`
  passed a list where `Dataset` is annotated for a tuple, and `results.save_fit`'s
  `**arrays` splat is not expressible against numpy's `savez` stub — which would likewise
  have failed the lint job on the first push. Same cause as the two defects D39 records:
  nothing has ever been pushed, so CI has never run.
- `MarginalOrbitModel` rebuilt the spectral prior from the sampled `log_tau`/`log_eta` and
  dropped any per-pixel profiles, so a window-confined component was silently
  un-confined the moment ML-II was switched on. The profiles are structure and the
  scalars are hyperparameters; the merge now respects that. Found before the feature
  shipped, and measured: with the confinement actually applied, the closed-loop K_2 error
  goes from 2.6% to 0.29%.
