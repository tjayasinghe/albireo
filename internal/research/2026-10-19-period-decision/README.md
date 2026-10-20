# The period comparison on a seven-epoch PEPSI binary (D75, 2026-10-19)

On the `period = "search"` route the pipeline compares the best four candidate periods of
the velocity table by the disentangling's coarse marginal likelihood
(`_decide_period_by_disentangling`, D63). On a seven-epoch PEPSI binary from a
collaborator's unpublished programme the comparison chose an eccentric alias at 7.80005 d
over the table's 6.26630 d. This note records why, the rules measured, and the rule
adopted: each candidate is also valued at its own table orbit (`_table_orbit_trial`). The
target, its files and its observation times are not given here.

## Data and recorded outcome

Seven PEPSI blue-arm epochs on seven nights over 702 d (112 cycles), read through the D73
readers; 4790-5410 A with H-beta masked; LSF sigma 2.874 km/s (R = 44,300); `dv_kms = 2.0`,
so the comparison's coarse grid is 4.0 km/s (0.70 px per LSF sigma); light fractions
measured (0.653, 0.347). The bootstrap table (BOSZ r20000 templates, broader than the LSF)
gives 29 distinct orbits. The four compared:

| rank | P [d] | table chi2 (14 velocities, 7 parameters) | scaled difference | table e | recorded value [nats] |
|---|---|---|---|---|---|
| 1 | 6.26630 | 245.0 | 0 | 0.005 +- 0.004 | 81630.8 |
| 2 | 3.13141 | 6895.5 | 190 | 0.477 +- 0.061 | 17582.0 |
| 3 | 5.52617 | 7767.0 | 215 | 0.166 +- 0.034 | 61299.1 |
| 4 | 7.80005 | 8118.6 | 225 | 0.332 +- 0.015 | 93117.7 |

The scaled difference is the chi-square difference in units of the best fit's reduced
chi-square (35.0). The comparison chose rank 4 by 11,487 nats. The fit at 7.80 d ended at
z_rms 1.959 and a final-table reduced chi-square of 1611. A run with the period declared as
[6.2, 6.34] d ended at P = 6.26819 d, K = 62.78 and 68.49 km/s, e = 0.0055, z_rms 1.791. A
dedicated analysis of the same spectra gives P = 6.268198 d, K = 63.09 and 68.84 km/s,
e = 0.015.

## Reproduction

The bootstrap and the comparison, replayed with the pipeline's own functions (the comparison
function captured by monkeypatching), reproduce the four recorded values to 0.1 nat. The
coarse marginal at a given orbit is the same under all four candidates' declarations, which
share one grid and one velocity budget.

## Comparison points and basins

Values on the coarse model, with the smoothness hyperparameters at the comparison's starting
values; in brackets the rms of the orbit's epoch velocities against the declared run's final
table (TODCOR with disentangled templates), per component.

| | candidate 1, 6.26630 d | candidate 4, 7.80005 d |
|---|---|---|
| recorded comparison value | 81630.8 (3.25, 3.50 km/s) | 93117.7 (3.15, 3.40 km/s) |
| the candidate's own table orbit | 153971.0 (0.49, 0.48 km/s) | 124517.6 (1.93, 2.04 km/s) |
| the table orbit made circular | 154723.0 | -230616.4 |
| 30 L-BFGS steps from the table orbit | 155647.6 (0.06, 0.10 km/s) | 128912.9 (1.94, 2.14 km/s) |
| full-model MAP orbit of the basin | 155702.1 (0.04, 0.05 km/s) | 131096.1 (2.01, 2.16 km/s) |

The true basin is 24,606 nats above the alias basin on the coarse model (the two full-model
MAP potentials differ by 25,586, under declarations whose priors differ). The comparison
scored both candidates at points more than one LSF sigma from the measured velocities, 74,071
and 37,978 nats below their basins. Its values were neither converged nor comparable.

## Cause

The comparison never evaluated a candidate's own table orbit. It declares the conjunction as
scanned (a 42-point lattice over one period, then eight conjunctions crossed with the
semi-amplitude lattice and refinements of P/16 and P/32) and holds the period at the table's
value and the eccentricity at its start. Candidate 1's table eccentricity failed the
three-sigma test of `_element_starts`, so its scans ran at the `Disentangler` default,
e = 0.05 at 0.5 rad, a distortion of e K = 3.3 km/s. The conjunction located for that shape
lies 0.155 d (0.025 cycles, 13 times the table's error of 0.012 d) from the table's, and the
semi-amplitude scan then moved K to 59.5 and 66.5 km/s, 5 percent low. Candidate 4's table
eccentricity passed the test, its scans ran on its own table shape, and its conjunction moved
0.0075 cycles. The elements are coupled: at candidate 1's point the period of the basin alone
raises the value by 15,055 nats, a circular orbit alone lowers it by 62,890, and period,
conjunction and eccentricity together raise it by 38,393
(`results/pepsi_seven_epoch.json`). The table's period offset (0.030 percent, 0.034 cycles
over the 112 cycles) is absorbed by the table's own conjunction at the seven epochs.

## Resolution

The 55 decisions recorded by earlier runs and the 53 replayed below are Gaia RVS (LSF sigma
11.1-11.7 km/s, coarse grid 6.0 km/s, 1.9 px per sigma), the regime in which D64 checked the
coarse model's ordering against the full model. On the 212 replayed Gaia candidates the table
orbit lies above the scans' best for 105 (median -0.1 nats, 90th percentile +219, maximum
+8,753). On the PEPSI binary it lies 13,442 to 72,340 nats above for all four candidates. The
coarse model itself orders the PEPSI basins correctly.

## Profile maximum and marginal

The comparison maximizes over the orbital parameters. Their marginal would add an Occam term,
of order the sum over the six parameters of ln(prior width / posterior width), estimated at 6
to 9 nats per parameter and below about 50 nats per candidate, nearly equal for candidates
with the same parametrisation and priors (the period and conjunction windows scale with P and
contribute 2 ln(7.80/6.27) = 0.4 nats). The basins differ by 24,606 nats. The failure is an
incomplete maximization, not the choice between maximum and marginal.

## Rules measured

- **T**: each candidate is also evaluated at its own table orbit (period, conjunction,
  semi-amplitudes inside the declared range, eccentricity), and its value is the larger of
  that and the scans' best. One coarse evaluation per candidate.
- **TC**: T with the table orbit made circular as a third trial.
- **M1**: the scans start from every candidate's own table eccentricity, detected or not.
- **V**: a candidate more than 100 scaled units of table chi-square above the best is not
  compared.
- **V below eight nights**: V on tables whose usable epochs fall on fewer than
  `_FEW_NIGHTS` nights only.
- **Refinement**: L-BFGS from each candidate's point; 3 to 10 s per step on the PEPSI coarse
  model, not pursued.

On the PEPSI binary T keeps 6.26630 d by 29,453 nats, TC by 30,206, M1 by 56,944, and V
without a comparison.

The 55 decisions recorded by earlier runs (gaia5, field5 and both arms of the D71
end-to-end run; `results/recorded_decisions.csv`) can be replayed exactly only for V: any
band from 50 to 200 changes only the PEPSI decision. The largest scaled difference of a
candidate correctly chosen over the table's choice is 40.7, and the incorrect choices lie at
0.4 to 14.6. A band of 25 loses K-12-041.

For the other rules the comparison was replayed with the current code on the 23 systems of
the D71 end-to-end run (all 23 reproduce their recorded decisions) and the other 31 systems
of the blind benchmark, gaia-51574864941234048 stopping in the bootstrap as in the pipeline
(`results/benchmark_replay.csv`). A decision is right when the chosen period is within 2
percent of the injected one.

| | right of 53 Gaia systems | against the table's choice | the PEPSI binary |
|---|---|---|---|
| table's choice alone | 45 | | right |
| the pipeline before D75 | 44 | gains K-12-041; loses gaia-40041022325608704, mixed-0014 | wrong |
| T, TC | 45 | gains K-12-041; loses gaia-40041022325608704 | right |
| M1 | 45 | as T | right |
| V (band 50 to 200) | 45 | as T | right |

The injected period is among the four compared on 46 systems. T and TC also move mixed-0011,
and M1 mixed-0011 and gaia-17170287112790656, between two wrong periods. mixed-0014 (48
epochs, 33 nights) is the PEPSI failure at R = 11,500: the injected 3.94983 d has an
undetected table eccentricity (0.004 +- 0.002) and loses by 73 nats to the half period, whose
detected eccentricity (0.494 +- 0.012) starts its scans, at a scaled table difference of
1,715. Its own table orbit is 1,342 nats above its scans. Of the 46 systems whose injected
period is compared, T widens its margin on 30 and narrows it on 7, by at most 205 nats with
every margin still positive; M1 widens 26 and narrows 5.

## End-to-end run with T

`run_star` on the PEPSI binary with T installed: the bootstrap keeps 6.26630 d by 29,453
nats; the disentangling ends at P = 6.26819 d, K = 62.78 and 68.49 km/s, z_rms 1.791; the
table's orbit is P = 6.268254 +- 0.000089 d, K = 62.83 and 68.51 km/s, e = 0.0026, reduced
chi-square 1.81. This is the declared-period run's orbit to the printed digits; K remains 0.4
to 0.5 percent below the dedicated analysis, as on the declared route. Wall-clock times: the
bootstrap 437 s, the disentangling 1802 s.

## Adopted, and open

T is adopted (D75). It acts at the point the scans omit, at the cost of one evaluation per
candidate, and changes no correct decision on the replayed set. M1 makes the same decisions
but starts every candidate whose eccentricity the table does not detect from that undetected
value. V makes the same decisions with a threshold set by three correct overturns; it is not
adopted, and is available if a decisive table is to be protected independently of the
scoring.

Open: gaia-40041022325608704 (12 epochs, 8 nights), where an eccentric half-period alias at a
scaled table difference of 14.4 wins by 7 nats with T and 55 without. On the replayed set the
comparison adds nothing over the table's choice even with T (45 against 45); whether it
should run by default is a separate question.

## Files

- `results/benchmark_replay.csv`: per compared candidate of the 53 replayed Gaia systems, the
  table's chi-square and scaled difference, the table eccentricity and the scans' start, and
  the coarse marginal for the scans, the table orbit, the circular table orbit and M1.
- `results/recorded_decisions.csv`: the 55 decisions recorded by earlier runs.
- `results/pepsi_seven_epoch.json`: the PEPSI numbers above, with conjunctions as offsets.

The scripts (`bench_replay.py`, `bench_driver.py`, `bench_decide.py`, `bench_margins.py`, the
PEPSI replays and the end-to-end driver) are in session 138754ea's scratchpad and are not
kept here, because the PEPSI scripts address the collaborator's files. A system's replay
takes 2 to 8 minutes on the desktop; the 54 took three hours with two workers.
