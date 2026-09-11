# Is the D62 disentangling fit budget-limited or landscape-limited?

One system, `gaia-50868531799454720`, under the `orbit` tier of the D62 Gaia RVS
benchmark, re-run at three L-BFGS step caps. Everything except `max_steps` is the
archived run's: the same population record (index 7 of `gaia4/population.json`), the same
simulation seed (7), the same tier declaration, and the same analysis settings from
`gaia4/manifest.json` (dv 3.0 km/s, k 2 to 250 km/s, e_max 0.9, label_steps 80, noise
model `correlated`, product `dr4`, library `bosz2024-fgk-rvs`, cadence `scanning-law`,
resolving power `nominal`, k-scan on). Truth for this system is P 8.864158 d, K_A 52.962,
K_B 55.301 km/s, e 0.0496, omega -55.264 deg, T_conj 2456880.8793, from 16 epochs at S/N
33.97 per pixel.

Scripts and products: `budget.py`, `collect.py`, `plot.py`, `steps100/`, `steps300/`,
`steps1000/`, `rows.json`, `potential.png`, `run*.log`.

## Reproduction of the archived run

The simulation reproduces the manifest's record to every digit printed there: S/N per
epoch 33.9658148545406, 16 epochs, lag-one 0.2703090115066514, baseline 1786.5091142924502
d. The 100-step re-run then reproduces the archived fit exactly. Potential, gradient norm,
step count, all four orbital elements, both semi-amplitudes, the ML-II hyperparameters,
the residual z-score RMS, the conjunction scan, the semi-amplitude scan and every entry of
the truth block agree to the last recorded digit. The only difference is the wall time:
220.7 s here against 512.1 s in the archive, of which the disentangling stage is 133.5 s
against 366.3 s. The archive ran with `jobs = 5`, which puts each star in a spawn worker
with its XLA and BLAS thread pools capped at 32 // 5 = 6; this re-run used `jobs = 1`,
which runs in process with no cap. The arithmetic is unaffected, the speed is not.

## The three step caps

Disentangling endpoint, and the difference from the injected truth:

| cap | steps | converged | potential | grad norm | K_A | K_B | dK_A | dK_B | e | omega (deg) | T_conj (BJD) | P (d) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| archive, 100 | 100 | False | -35584.928 | 1888.75 | 61.751 | 63.709 | +8.789 | +8.407 | 0.2475 | -45.812 | 2456879.8283 | 8.866013 |
| 100 | 100 | False | -35584.928 | 1888.75 | 61.751 | 63.709 | +8.789 | +8.407 | 0.2475 | -45.812 | 2456879.8283 | 8.866013 |
| 300 | 300 | False | -35683.757 | 294.02 | 56.449 | 58.073 | +3.487 | +2.772 | 0.1296 | -64.353 | 2456880.5800 | 8.864583 |
| 1000 | 1000 | False | -35688.886 | 100.06 | 54.554 | 55.971 | +1.592 | +0.670 | 0.0878 | -59.991 | 2456880.7237 | 8.864441 |

Element differences (result minus truth) for the same three fits: period +0.001855,
+0.000425, +0.000283 d; eccentricity +0.19791, +0.08008, +0.03825; omega +9.452, -9.088,
-4.727 deg; T_conj -1.0510, -0.2993, -0.1557 d. Every one of them shrinks monotonically.

The fit is never flagged converged. The facade sets `tol = max(1e-2, 1e-6 * n_good)`, and
with 15360 good pixels that is 0.01536. The gradient norm at 1000 steps is 100.06, still
6.5e3 times the threshold, so the cap remains the stopping rule at every budget tried.

Downstream products, which is what the benchmark actually reports:

| cap | table K_A | table K_B | dK_A | dK_B | epoch v RMS A/B (km/s) | Keplerian v RMS A/B (km/s) | spectra RMS A/B | label chi2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 54.223 | 56.308 | +1.261 | +1.007 | 1.026 / 0.997 | 1.590 / 1.710 | 0.1114 / 0.1000 | 5360.0 |
| 300 | 53.390 | 54.958 | +0.428 | -0.343 | 0.643 / 0.821 | 0.451 / 0.354 | 0.1019 / 0.0911 | 3440.3 |
| 1000 | 53.527 | 54.966 | +0.565 | -0.336 | 0.587 / 0.763 | 0.366 / 0.321 | 0.0989 / 0.0885 | 3298.9 |

The velocity table's orbit is the exception to the monotonicity: it improves sharply from
100 to 300 steps and then stops, ending marginally worse in K_A at 1000 than at 300. Two
of the pipeline's own flags disappear at 1000 steps, the pair that had recorded a
disagreement between the disentangling's K and the velocity table's K (7.5 sigma at 100
steps, 3.1 at 300, absent at 1000).

## Walls

| cap | disentangle (s) | labels (s) | total (s) |
| --- | --- | --- | --- |
| archive, 100 | 366.3 | 115.0 | 512.1 |
| 100 | 133.5 | 74.0 | 220.7 |
| 300 | 353.6 | 75.3 | 444.3 |
| 1000 | 1808.1 | 60.6 | 1882.5 |

Steps 101 to 300 cost 1.10 s each; steps 301 to 1000 cost 2.08 s each. Part of that rise
is the zoom line search taking more function evaluations per accepted step as the gradient
falls, and part may be machine contention, since other Python processes were resident
during the 1000-step run. Read the 2.08 s as an upper bound. A budget of 300 steps
roughly doubles the wall of a star relative to 100, and 1000 steps costs about eight and a
half times the 100-step run.

## No per-step history exists

`MAPResult` in `src/albireo/inference.py` carries `params`, `unconstrained`, `potential`,
`grad_norm`, `converged` and `num_steps`, all evaluated at the final step. The facade's
`Fit` (`src/albireo/facade.py`) keeps the `Disentangler`, that `MAPResult`, the ML-II
hyperparameters, the phase scan and the semi-amplitude scan, and nothing else. The only
route to a trajectory is `run_map`'s `callback(step, potential, grad_norm, params)`, and
`src/albireo/pipeline.py` line 1398 calls `dis.fit(max_steps=..., k_scan=...)` without a
`progress` argument, so the pipeline records nothing per step. Only the endpoints are
available, and `potential.png` therefore plots three endpoints rather than a trace. The
three points do lie on one path: L-BFGS here is deterministic, the step cap is the only
thing that differs between the runs, and the conjunction scan (T_conj 2456880.951634872)
and the semi-amplitude scan (best 54.3042 and 53.0148 km/s, not used, since the declared
start was better) are identical in all three, as is the 100-step endpoint against the
archive.

## Verdict

The fit is budget-limited, not landscape-limited. The 100-step endpoint at 62 and 64 km/s
is not a stationary point of any kind. It is a point part way down a long valley that the
optimizer is still descending when the cap stops it, and the extra steps walk it back
toward the truth in every parameter at once: the error on K_A falls from 8.79 to 3.49 to
1.59 km/s and on K_B from 8.41 to 2.77 to 0.67 km/s, eccentricity from +0.198 to +0.080 to
+0.038, period from +0.0019 to +0.00028 d, conjunction from -1.05 to -0.16 d, and the
Keplerian velocity residual from 1.6 and 1.7 to 0.37 and 0.32 km/s. The disentangled
spectra improve with them, the label chi-square falls by 38 percent, and the two flags
recording a K disagreement between the disentangling and the velocity table clear. The
gradient norm falls by a factor of 19 across the same range and is still 6.5e3 times the
tolerance at the end, so there is no evidence of a nearby stationary point at 1000 steps
either. The one caution the numbers carry is that the potential is a poor proxy for
progress: 95 percent of its total fall between 100 and 1000 steps has already happened by
step 300, and the remaining 5.1 nats over the next 700 steps buy the largest single
improvement in K_B. The valley is nearly flat along the direction that still matters, so a
step budget cannot be tuned by watching the objective, and a step count reported with
`converged = False` is not by itself a diagnosis. It is also worth recording that the
delivered velocity-table orbit, which is what the benchmark quotes, saturates by 300 steps
while the disentangling's own K keeps improving to 1000, so raising the cap from 100 to
300 recovers most of the accuracy available for roughly double the wall, and the run from
300 to 1000 buys a better disentangling rather than a better published orbit.
