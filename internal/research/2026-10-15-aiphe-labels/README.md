# AI Phoenicis label fit: re-run at the declared widths and the unrounded period (2026-10-15)

The label fit of `scripts/aiphe_labels_bench.py` on the 36 HARPS spectra of AI Phe was
re-run with the code at 5bc9658. Since the run of 2026-08-27 three inputs had changed: each
exposure is modelled at its declared line-spread width (30 HAM exposures at R = 115,000,
6 EGGS exposures at R = 80,000; D59), the velocities are held at P = 24.592483 d instead of
24.5924 d, and the registry builds `bosz2024-fgk-r20000` as cache version 2 (455 nodes, the
unpublished node at 5750 K, log g 3.0, [M/H] -0.75 filled along [M/H], Catmull-Rom cubic;
D62) instead of version 1 (454 nodes, piecewise-linear simplex interpolation). The record is
in `docs/benchmarks.md`, "Re-run at the declared widths and the unrounded period
(2026-10-15)" and "Effect of the width, the period and the library build (2026-10-15)".

## Results

The script as committed (log g declared, dilution fitted): primary 6313.9 K (+3.9 K,
+0.06%), secondary 5224.9 K (+214.9 K, +4.29%), R2/R1 = 1.5297 (-5.8%), chi-square 419,615
over 15,164 pixels. The run of 2026-08-27 recorded 6342.5 K (+0.52%), 5227.0 K (+4.33%) and
1.5416 (-5.1%).

Log g declared, dilution fitted, current code, the script's v sin i prior (0.5 to 25 km/s):

| library | width | period [d] | primary Teff [K] | secondary Teff [K] | R2/R1 | chi2 |
|---|---|---|---|---|---|---|
| v1 | HAM for all 36 | 24.5924 | 6331.00 | 5269.46 | 1.5155 | 425,695.7 |
| v2 | HAM for all 36 | 24.5924 | 6314.66 | 5225.38 | 1.5302 | 425,202.6 |
| v2 | HAM for all 36 | 24.592483 | 6315.50 | 5224.26 | 1.5304 | 422,402.0 |
| v2 | declared | 24.5924 | 6316.12 | 5224.11 | 1.5314 | 422,439.3 |
| v2 | declared | 24.592483 | 6313.90 | 5224.87 | 1.5297 | 419,615.1 |
| v1 | declared | 24.592483 | 6334.77 | 5268.33 | 1.5171 | 420,079.8 |

- On v2 the width and the period each move the temperatures by less than 2 K, and the two
  effects do not add; every fit stops at 400 L-BFGS steps (gradient norms 3 to 33).
- On v1 the end point depends on the optimizer's path. The code of 2026-08-27 (ee8e46d)
  reproduces the recorded 6342.503 K, 5226.975 K and chi-square 425,869.2 exactly with the
  notebook's v sin i prior (0.1 to 30 km/s) and gives 6383.3 K and 5214.1 K with the
  script's. On v2 the two priors agree to 0.6 K (notebook: 6314.5 K, 5224.7 K).
- The recorded frozen-dilution row of 2026-08-27 (chi-square 1,853,578) is not a native
  fit; native gives 6444.4 K, 5264.3 K, chi-square 435,714 on the code of that date.
- `compare="matched"` is rejected by the current code for this library (R = 20,000 is
  broader than HARPS). With the resolving power undeclared it gives chi-square 4.26 times
  native, as on 2026-08-27.

## Files

- `results/aiphe_labels_bench_log.txt`: output of `scripts/aiphe_labels_bench.py --data
  data/aiphe` at 5bc9658.
- `results/isolation.json`, `results/isolation_log.txt`: the six width, period and library
  combinations, three configurations each, plus `compare="matched"` with the library's
  resolving power undeclared on two of them. Each record holds the labels, formal errors,
  light fractions, radius ratio, chi-square and its nulls, the optimizer's step count and
  gradient norm, label correlations of at least 0.5, and `summary()`.
- `results/aiphe_labels_bench_ee8e46d_log.txt`: the script of 2026-08-27 on the code of
  that date (`git archive ee8e46d src scripts`, run with `PYTHONPATH=<export>/src`) and the
  cached v1 library.
- `results/old_notebook_config_log.txt`: the notebook configuration of 2026-08-27 (v sin i
  prior 0.1 to 30 km/s) on the same code.
- `runs/` (not tracked): the harness `aiphe_labels_isolation.py`, which switches the module
  constants of `aiphe_bench.py` and `aiphe_labels_bench.py` (`P_PUB`, `LSF`,
  `LSF_SIGMA_EGGS`) and reads the v1 build from the cache directly, the driver
  `run_isolation.sh`, the tabulation script, and the script for the old notebook
  configuration.

The v1 build is `bosz2024-fgk-r20000-v1.npz` in the albireo cache (content_sha256
72d9b3c3...), built on 2026-08-27; the current code no longer reads it.
