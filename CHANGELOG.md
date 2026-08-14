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
