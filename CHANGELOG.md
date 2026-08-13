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
