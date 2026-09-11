# Why the coarse semi-amplitude scan lost the faint secondary of gaia-17170287112790656

Experiment of 2026-09-10 (D63) on one simulated Gaia RVS binary from the D62 benchmark,
orbit tier. Marginal log-likelihoods in nats. No fit was run: these are scans of the same
likelihood the scan itself calls. Scripts and raw values: the session scratchpad `wp-e/`
(common.py, run_coarse.py, run_full.py, run_widths.py, run_crossover.py, run_remedy.py,
run_archivelike.py; kb_sweeps.png, ka_and_crossover.png).

## 1. Reproduction

`simulate_system` with the resolved `bosz2024-fgk-rvs` library, the DR4 product, the
scanning-law cadence, the nominal resolving power and seed 1 reproduces the manifest record
exactly: snr_epoch 21.038600162198243, 11 epochs, lag1 0.27023359625423926, baseline
1615.6442610388622 d, grid_n 4613. Injected: P = 0.578633 d, e = 0, K_A = 103.037,
K_B = 192.699 km/s, gamma = 11.426 km/s, conjunction folded into the phase-scan window
2457148.520178. `light_ratio = 0.0722` is F2/F1, so l_A = 0.932655 and l_B = 0.067345: the
secondary carries 6.73 percent of the light.

Declaration (facade level): `Star("A", light=0.932655)`, `Star("B", light=0.067345)`,
`Orbit(period=Known(P, 1e-4 P), k=Between([2, 2], [250, 250], start_at=[84.667, 167.333]),
t_conj="scan", ecc=Fixed(0.0))`, `lsf={"RVS": LSF.from_resolution(11500)}`, dv 3.0 km/s,
`noise_correlation={"RVS": 0.27023}`. Two idealisations against the archived run, both
making the problem easier: the true light fractions (the archived orbit tier measured
A 0.601, B 0.399, wrong for the secondary by a factor of six) and a circular declaration
(the archived run declared `Between(0, 0.9)`; the velocity budget then shrinks from 997.5
to 525 km/s, the grid from 3500 to 3185 px). The starting semi-amplitudes are the archived
run's own.

The harness was checked against the archive by repeating the scan under the archived
declaration (measured light, free eccentricity): it reproduces the archived `k_scan` block
bit for bit (best 84.667 / 22.811, gain 43.097 nats, hold losses A -0.995, B 43.097, 446
trials, the same conjunction 2457148.5110755). What follows is a property of the scan.

## 2. What the D62 scan does on the idealised declaration

Phase scan: best 2457148.5252, contrast 1648 nats, 3.2 s; one grid step from the archived
choice and 0.86 percent of a period from the injected conjunction. The phase is not what
fails.

Semi-amplitude scan (446 trials, 16.6 s): best = start = (84.667, 167.333), gain 0, hold
losses {A: 15.13, B: 16.13}, both components "kept at its start ... below 25". Before the
guards: the bounds rule (2 percent inside the range) gives 6.96 to 245.04 km/s and 7 points,
so level 0 ran at ratio 1.8104 (6.96, 12.60, 22.81, 41.30, 74.76, 135.35, 245.04) crossed
with itself and 8 conjunctions. Level 0 chose (74.76, 22.81); the two refinements moved it
to (105.73, 38.36) at 19487.49, 31.26 nats above the start. The guards then discarded it one
component at a time: holding A at its start costs 15.13 nats (below 25, so A returned),
holding B then costs 16.13 (B returned). On the same coarse model the injected
semi-amplitudes are worth 39.10 nats over the start and 7.83 over the scan's raw best; no
single-component hold reached 25 although the joint move was worth 39.

With the measured light fractions and the free eccentricity the same scan moves K_B to
22.81 with a hold loss of 43.10 nats, above the threshold, which is how the archived run
started L-BFGS at 22.8 and ended at 19.35 km/s with the primary 13 percent high.

## 3. K_B sweeps (geometric grid at ratio 1.10, 2 to 250 km/s)

| curve | model | K_A, conjunction | peak K_B | L(192.7) - L(23.0) | drop at -20%, -10%, +10%, +20% |
|---|---|---|---|---|---|
| a | coarse, dv 6 | 84.667, scan | 22.5 | -14.24 | 0.34, 0.15, 0.09, 0.37 |
| b | coarse, dv 6 | 103.037, truth | 203.4 | +14.73 | 13.79, 7.56, 4.13, 4.85 |
| c | full, dv 3 | 84.667, scan | 22.5 | -9.19 | 0.54, 0.13, 0.18, 0.64 |
| d | full, dv 3 | 103.037, truth | 203.4 | +17.85 | 17.34, 9.98, 9.01, 7.59 |

The secondary's peak exists, sits at 203 km/s (5.5 percent above the injected 192.7) on
both models, and both prefer it over 23 km/s by 15 to 18 nats: the pixel is not what loses
the companion. The peak is shallow in absolute terms: the whole K_B axis spans 23.6 nats
(coarse) and 25.1 (full) with the primary at the truth, against 1635 nats for the
primary's own axis. With K_A wrong by 18 percent the peak inverts (curves a, c): K_B = 23
wins by 14 and 9 nats on a nearly flat curve. The level-0 point 245.04 sits 4.8 (coarse) or
7.4 (full) nats below the peak, inside its +20 percent width; 135.35 sits 14 to 16 nats
below. The ratio-1.81 grid does place a trial inside the secondary's peak; the scan never
evaluated it at a K_A where it would have won.

## 4. The primary

K_A swept the same way on the coarse model with K_B and the conjunction at the truth: peak
at 99.6 km/s (3.3 percent below the injected), drops 38.4, 14.5, 19.8, 50.6 nats at -20,
-10, +10, +20 percent, 1635 nats across the range. The level-0 axis lands 54.0 nats below
the peak at 74.76 and 117.2 below at 135.35.

## 5. How wrong K_A may be before the secondary is lost

Coarse model, L(K_B = 203.4) - L(K_B = 22.5) against K_A / K_A(true):

| K_A / true | at the located conjunction | at the injected conjunction |
|---|---|---|
| 0.726 (grid 74.76) | -27.5 | -19.3 |
| 0.822 (start) | -12.2 | -3.1 |
| 0.879 | -3.6 | +6.2 |
| 0.924 | +2.2 | +11.9 |
| 1.000 | +8.5 | +17.5 |
| 1.072 | +9.3 | +16.8 |
| 1.183 | +0.1 | +4.9 |
| 1.243 | -9.3 | -6.0 |
| 1.314 (grid 135.35) | -23.4 | -21.9 |

The true K_B is preferred only while K_A lies within about 0.92 to 1.18 of its true value
at the located conjunction (0.88 to 1.22 at the injected one), a window a factor 1.28 to
1.39 wide. Both level-0 grid points bracketing the truth fall outside it: there is no point
on the joint scan's level-0 grid at which the secondary's true semi-amplitude is the better
trial.

## 6. Verdict and remedy

Not the model, and not a narrow K_B peak. Two causes act together, both in K_A. The grid is
too coarse for the primary and the axes are scanned jointly: at either bracketing point the
secondary's shallow preference inverts, and the coordinate refinement (a factor 1.41 then
1.19 per level) cannot carry K_B from 22.8 to 193. And the hold guard tests one component at
a time while the evidence is joint: 39 nats of joint move decomposed into two holds of 15
and 16. In the archived run the same guard passed a wrong move because a measured light
fraction of 0.40 for a star carrying 0.07 inflates everything the secondary's row does.

To guarantee a level-0 trial inside the window where the true K_B wins, the K_A axis ratio
must be at most about 1.28; a product grid at that ratio (2312 trials at 1.25, 20808 at
1.1) is unaffordable and the `max_trials` loop would coarsen it straight back. Scanning the
axes one at a time costs 153 trials at ratio 1.25 (136 for K_A with 8 phases, 17 for K_B);
a final full-axis pass per component at the refined best would have moved this star's
secondary from 38 km/s into the right basin for seven extra evaluations (at the scan's own
refined K_A of 105.73 the K_B sweep peaks at 245 on the level-0 axis and at 203 on a
ratio-1.1 axis). Neither helps unless the guard is applied jointly or scaled to what a
component's row can contribute: a 6.7-percent companion commands about 25 nats over the
entire K range, and a fixed 25-nat single-component threshold will always refuse to move it.
These findings are what D63's sequential scan implements.
