# AI Phoenicis: two statements measured again under the corrected conventions (2026-10-19)

Two statements about AI Phe in `docs/benchmarks.md` had been measured under conventions that
were corrected on 2026-10-15: the agreement of albireo with the clean-room shift-and-add
(2026-08-15), which compared albireo's deconvolved spectra with shift-and-add's at the
instrumental resolution (D72), and the microturbulence test of the label fit (2026-08-27),
run with `compare="matched"`, whose chi-square is inflated about 4.3 times (D55). Both were
measured again with the code at bfe02bd. The record is in `docs/benchmarks.md`, "Agreement
with shift-and-add at a common resolution (2026-10-19)" and "Microturbulence as the cause of
the secondary offset (negative result)".

## The agreement with shift-and-add

The harness of `scripts/aiphe_bench.py` (its functions, with its module globals switched),
shift-and-add at seven sweeps as in the harness, and at 1, 50, 500 and 5,000 sweeps and the
limit of its iteration (`fd3_bench_checks.ShiftAndAddLimit`, solved directly). albireo's
spectra are brought to the resolution of shift-and-add's output by convolving them with each
epoch's Gaussian (30 at sigma = 1.107 km/s, 6 at 1.591 km/s) and averaging over the epochs.
Core: component 1 below -0.10, a twelfth of the grid left out at either end; mean-aligned RMS,
component 1 / component 2.

| configuration | deconvolved (recorded convention) | instrumental resolution |
|---|---|---|
| 5340-5440 A, HAM width for all 36, P = 24.5924 d (2026-08-15) | 0.0287 / 0.0366 | 0.0260 / 0.0315 |
| 5340-5440 A, declared widths, P = 24.5924 d | 0.0292 / 0.0375 | 0.0261 / 0.0317 |
| 5341-5440 A, declared widths, P = 24.592483 d (aiphe_bench.py now) | 0.0282 / 0.0362 | 0.0248 / 0.0305 |
| 5341-5440 A, declared widths, P = 24.5924 d | 0.0282 / 0.0363 | 0.0249 / 0.0304 |
| 5340-5440 A, declared widths, P = 24.592483 d | 0.0291 / 0.0375 | 0.0260 / 0.0318 |
| 5150-5250 A, declared widths, P = 24.592483 d | 0.0535 / 0.0651 | 0.0517 / 0.0624 |

- The first row reproduces the recorded 0.029 and 0.037 on the current code.
- `scripts/aiphe_bench.py` was changed to print both conventions; its output on 5341-5440 A
  and on its default window is `results/aiphe_bench_5341_log.txt` and
  `results/aiphe_bench_5150_log.txt`, equal to the third and last rows.
- Band split (the bands of D72: below a frequency of 0.00187 per px, a period of 535 px or
  430 km/s, the 36 epochs give the difference mode less than half its information; the
  cutoff of the mean line-spread function is at 0.127 per px): after seven sweeps almost all
  of the difference is in the low band (0.0233 / 0.0267 of 0.0248 / 0.0305 in the current
  configuration); the line band is 0.006 / 0.011-0.012 in every configuration and at every
  number of sweeps, and above the cutoff 0.0017-0.0018 / 0.0026-0.0034.
- The limit of the iteration depends on the window's edges in the low band: from a start at
  5340 A to one at 5341 A, shift-and-add's converged spectra change by 0.018 / 0.021,
  its spectra after seven sweeps by 0.0014 / 0.0015 and albireo's by 0.0023 / 0.0028. The
  period changes neither code in the low band (`results/sa_window_dependence.json`).
- With the mean of the difference kept, the differences after seven sweeps are 0.062 / 0.074
  on the 5340-5440 A and 5341-5440 A windows. The data do not determine that mean (D72).

## The microturbulence test in native mode

Configuration of the label re-run of 2026-10-15 (`scripts/aiphe_labels_bench.py`, each
exposure at its declared width, P = 24.592483 d), log g declared, native comparison. The
160 nodes of the 2026-08-27 test (Teff 4500-6750 K, log g 3.0-4.5, [M/H] -0.50 to +0.25)
were built at xi = 2 and xi = 1 by `albireo.library.ingest_bosz` from the BOSZ files already
in the cache, under a temporary registry entry; nothing was downloaded. The xi = 2 build is
identical to the same nodes of the cache's version-2 build, and the xi = 1 build to the build
of 2026-08-27 (`bosz-vmic1-v1.npz`, content sha256 21b4f15c...) (`results/libraries.json`).
Priors on the 160-node libraries: primary Teff 5400-6750 K, secondary 4500-5900 K, [M/H]
-0.50 to +0.25.

| library | dilution | steps | primary Teff | secondary Teff | [M/H] | chi2 |
|---|---|---|---|---|---|---|
| v2, 455 nodes, script priors | fitted | 400 | 6313.90 | 5224.87 | -0.050 | 419,615.1 |
| v2, 455 nodes, box priors | fitted | 400 | 6315.21 | 5224.56 | -0.050 | 419,623.9 |
| xi = 2, 160 nodes | fitted | 688, converged | 6314.74 | 5225.16 | -0.050 | 419,615.5 |
| xi = 1, 160 nodes | fitted | 990, converged | 6285.77 | 5263.54 | +0.089 | 454,292.2 |
| xi = 2, 160 nodes | frozen | 1495, converged | 6385.01 | 5148.00 | -0.068 | 427,534.6 |
| xi = 1, 160 nodes | frozen | 1768, converged | 6385.89 | 5169.05 | +0.073 | 467,953.4 |

- The first row reproduces the 2026-10-15 result to the last digit.
- The 400-step fits on the 160-node libraries agree with the converged ones to 3.5 K
  (`results/xi_native.json`, keys without `/steps2000`).
- At xi = 1 the warm-start scan found two starting basins within delta chi-square 9
  (`multimodal`); all four starts are optimized and the best is reported.
- Equivalent width at t5250/g3.5, xi = 2 against xi = 1 (`results/ew_check.json`): +11.6%
  over 5150-5250 A at [M/H] 0, +12.6% over 4000-7000 A, +6.4% over 5160-5190 A; the 8.45%
  recorded on 2026-08-27 was not reproduced by any of the definitions tried.

## Documents updated with the native values

- `docs/tutorials/aiphe-labels.ipynb`, the Markdown cell "The result": +0.14 dex in [M/H]
  (was +0.10) and 11.6 per cent in equivalent width at 5250 K and log g 3.5 over the
  notebook's window (was 8.5). The cell was changed in `scripts/build_aiphe_notebook.py` and
  copied into the executed notebook without re-execution by `runs/copy_result_cell.py`,
  which requires every other cell of the builder to equal the notebook's. An nbformat read
  and write of this notebook is byte-identical (`runs/nb_roundtrip_check.py`), so the file
  differs from bfe02bd in three source lines of that cell only.
- `internal/design.md`, D55: a note with the native values after the matched-mode figures.

## Files

- `results/sa_common_resolution.json`, `results/sa_common_resolution_log*.txt`: every
  configuration, both conventions, every number of sweeps, the band split, the line depths.
- `results/sa_window_dependence.json`, `results/sa_window_dependence_log.txt`: the change of
  each code's spectra with the window start and with the period.
- `results/aiphe_bench_5341_log.txt`, `results/aiphe_bench_5150_log.txt`: the changed
  `scripts/aiphe_bench.py`.
- `results/libraries.json`, `results/xi_build_log.txt`: the two 160-node builds and their
  comparisons. `results/ew_check.json`: the equivalent widths.
- `results/xi_native.json`, `results/xi_fit_log*.txt`: the label fits, each with its labels,
  formal errors, light fractions, radius ratio, chi-square and nulls, step count, gradient
  norm and `summary()`.
- `runs/` (not tracked): the harnesses `aiphe_sa_common_resolution.py`,
  `sa_window_dependence.py`, `aiphe_xi_native.py` (`build`, then `fit --only KEY`) and
  `ew_check.py`, the driver `run_xi.sh`, the notebook scripts `nb_roundtrip_check.py` and
  `copy_result_cell.py`, the spectra of each configuration (`sa/*.npz`) and the two 160-node
  libraries (`xi/libraries/*.npz`).
