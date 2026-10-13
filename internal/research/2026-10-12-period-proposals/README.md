# Period proposals learned from simulated velocity tables (2026-10-12)

The pipeline decides a period for a binary without one in three steps
(`albireo.pipeline`): four periodograms of the velocity table propose candidate periods
(`_period_candidates`), an orbit is fitted from every candidate (`_orbit_over_candidates`,
by `albireo.rvorbit.assign_components`), and the chi-square ranks the orbits. This note
records an experiment on the first step. Proposals learned from simulated tables were
compared with the periodograms on two measures: whether the injected period is in the
list, and whether the orbit ranked first after the fits has it.

The outcome is negative for the learned proposals and is recorded as such: none is added
to the package. A network that scores trial periods at the resolution of the periodogram
lists the injected period more often and higher, and the orbit ranked first does not
change. The experiment did find two properties of the pipeline's own search that cost it
candidates, and one of its candidate fits. They are in the last sections.

The experiment was run by an Opus agent from a design made in the main session. Its
numbers were checked against its result files, and the gate measurement on the blind
tables was added afterwards.

## Data

- **Tables.** The simulated survey of eclipsing binaries (D69,
  `internal/research/2026-10-07-rvs-eclipsing-todcor/`, 7,800 systems), the transits out of
  eclipse under the classified declaration. 6,486 systems have at least 8 usable epochs,
  and the 4,029 of those whose recorded orbit has both semi-amplitudes within 10 percent
  were kept.
- **Split.** 1,200 systems for the test, 150 of them for the fits, 400 for validation and
  2,429 for training. The networks were trained on 7,942 tables: the three declarations of
  the systems outside the test and validation sets, each selected under its own
  declaration.
- **Out of distribution.** The 32 blind tables of the Gaia RVS benchmark, built as
  `scripts/period_decision_bench.py` builds them. The reproduction of the pipeline's
  ranking on them gives the recorded 26 of 32 at rank 1.

## Arms

- **A, the baseline.** `_period_candidates` with the settings of the pipeline
  (`swap_invariant=True`, `detection_min=100`, `leave_one_out=True`) and the candidate
  fits of `_orbit_over_candidates`. The median list has 75 candidates.
- **B, a prior without training.** Masses from the declared temperatures on the main
  sequence give an expected period for an observed amplitude through Kepler's third law
  and an inclination. B(i) ranks the fitted orbits of A by the chi-square less twice the
  logarithm of that prior. B(ii) orders a deeper list of periodogram peaks by peak
  position and prior. An isotropic inclination and an inclination above 70 degrees, as
  the eclipses of the training set imply, were both run.
- **C, a density over the orbital elements.** A permutation-invariant encoder of the
  table (a transformer of three blocks) and an autoregressive spline flow over the seven
  elements, 567,417 parameters. Its proposal is 2,000 samples, each moved to a maximum of
  the periodogram, with the maxima ranked by the number of samples.
- **D, a score per trial period.** For every frequency of the baseline's grid, weighted
  harmonic fits give a phase reference and summary statistics, a small network is applied
  to each epoch of the folded series and pooled, and a softmax over the grid gives the
  score. 42,625 parameters.
- **Controls.** A deeper baseline list (200 peaks per search), a ladder of periods 2
  percent apart around the Kepler estimate with no periodogram and no training, and the
  baseline without its detection gate.

## Measures

- **Coverage.** The share of tables whose list has a period within 2 percent of the
  injected one, at list lengths of 20, 50, 100 and 200 and at the length of the
  baseline's own list, on the 1,200 test tables.
- **Rank 1 after the fits.** The share of tables on which the orbit of lowest chi-square
  is within 2 percent of the injected period, with every arm given as many fits as the
  baseline, on 150 test tables.

Errors are binomial standard errors, and paired comparisons are exact two-sided McNemar
tests.

## Coverage on 1,200 test tables

| arm | at 20 | at 50 | at the baseline's length | first candidate |
|---|---|---|---|---|
| A | 91.7 ± 0.8 | 93.2 ± 0.7 | 93.5 ± 0.7 | 77.2 ± 1.2 |
| A, 200 peaks per search | 91.7 | 93.2 | 93.5 | 77.2 |
| A, no detection gate | 95.4 ± 0.6 | 96.2 | 96.4 ± 0.5 | 78.0 |
| ladder around the Kepler period | 71.3 | 93.4 | 94.9 ± 0.6 | 9.6 |
| B(ii), isotropic | 90.8 ± 0.8 | 92.9 | 93.8 ± 0.7 | 74.7 |
| B(ii), inclination above 70 degrees | 94.3 ± 0.7 | 97.1 | 97.5 ± 0.5 | 82.5 |
| C | 80.4 ± 1.1 | 98.3 | 98.9 ± 0.3 | 7.9 |
| D | 99.0 ± 0.3 | 99.8 | 99.9 ± 0.1 | 95.4 ± 0.6 |

The union of the baseline's list with 20 candidates of another arm covers 95.0 percent
with the deeper list, 96.8 without the gate, 97.9 with the ladder, 98.6 with C and 99.5
with D. Each differs from the baseline at p below 1e-5. The union with C is not separable
from the union with the ladder (17 tables against 9, p = 0.17).

The gain is in one class of tables. On the 163 tables whose secondary is below the
detection gate at half or more of the usable epochs and that have at least one epoch in
the other order of the pair, the baseline covers 61.3 percent, the baseline without the
gate 82.8, B with the high inclination 88.3, C 93.9 and D 99.4. On the other 1,037
tables the baseline covers 96.7 percent or more.

## Rank 1 after the fits on 150 test tables

| arm | rank 1 | rank 1 with both K within 10 percent | lost / gained against A |
|---|---|---|---|
| A | 137 | 115 | |
| B(i) | 137 | 115 | 1 / 1 |
| B(ii), isotropic | 135 | 113 | 3 / 1 |
| B(ii), inclination above 70 degrees | 135 | 114 | 3 / 1 |
| C, nearest maximum | 31 | 25 | 108 / 2 |
| C, highest maximum within 2 percent | 128 | 107 | 11 / 2 |
| D | 138 | 115 | 1 / 2 |
| A with 20 candidates of any other arm | 136 | 115 | 1 / 0 |
| A, ranked by chi-square and the score of D | 139 | 115 | 0 / 2 |
| A with 20 of D, ranked the same way | 140 | 116 | 0 / 3 |

No arm differs from the baseline by more than three tables, and none of the differences
is significant (p of 0.25 or more), except those of C, which are losses.

The first candidate of D, with no orbit fitted, is within 2 percent on 142 of the 150
tables. The baseline reaches 137 after 71 fits (6 tables against 1 in the paired
comparison, p = 0.125).

A sample of C is within 2 percent of the injected period for 5.2 percent of samples, and
moving it to the nearest maximum of the periodogram leaves that share unchanged. The
median error of a sample is 0.089 dex, 23 percent of the period. A peak of the
periodogram of a period of a few days over the span of DR4 is about a thousandth of the
period wide, so the density has the width of the prior it learned and not of a peak. A
fit started within 2 percent of the injected period ends within 0.1 percent for 210 of 436
starts: the start has to be on the right peak, and the samples do not select it.

## The 32 blind tables

| arm | listed at the baseline's length | first candidate | rank 1 | rank 1 with both K within 10 percent |
|---|---|---|---|---|
| A | 27 | 14 | 26 | 18 |
| B(ii), isotropic / inclined | 21 / 24 | 12 / 13 | 20 / 23 | 16 / 17 |
| C, nearest maximum | 22 | 0 | 7 | 4 |
| D | 25 | 18 | 23 | 17 |
| A with 20 of D | 29 | | 27 | 19 |
| A with 20 of the deeper list | 28 | | 27 | 19 |

Out of the distribution it was trained on, D alone is below the baseline, and the one
table its union gains is also gained by a deeper baseline list.

## Cost

Seconds per table in one process: 0.95 for the candidates of A and 7.4 for its 72 fits;
1.75 for the scores of D over a grid of a median 78,770 frequencies; 0.1 for C. Training
on the CPU took 17 minutes for C and 58 for D with its additional hard negatives.

## The detection gate

The pipeline removes a companion's velocity whose detection statistic is below
`Analysis.detection_min` (100) from the period search and from the candidate fits. The
experiment measured both uses on the survey tables, where the secondary of a faint system
is below 100 at most epochs while its velocities are usable.

**The search.** With the gate, a table whose companion is below it at every epoch has no
relative-velocity, two-harmonic or swap-invariant search, and its candidates come from
the first component alone, which for two alike stars is a mixture of both. System 133 of
the survey is an example (P = 1.624 d, K = 104 and 107.5 km/s, G_RVS = 12.8, 17 of its 25
usable epochs in the other order): the injected period is absent with the gate and at
position 6 without it, from the swap-invariant search. 18.2 percent of the test tables
have no relative-velocity search, and they are 78 percent of the baseline's misses.

**The fits.** On the 150 tables, with the candidates of A:

| search, fits | rank 1 | rank 1 with both K within 10 percent |
|---|---|---|
| gate, gate (the pipeline) | 137 | 115 |
| gate, no gate | 138 | 134 |
| no gate, no gate | 140 | 136 |
| candidates of D, no gate | 139 | 135 |

The period decided changes on five tables (two lost, three gained with the baseline's
candidates). The semi-amplitudes of the orbit at rank 1 are within 10 percent on 19 to 21
more tables, with none lost (p below 4e-6). On the 19 tables whose secondary is below the
gate at half or more of the epochs with none exchanged, the gated fit has the period at
rank 1 on all 19 and both semi-amplitudes within 10 percent on 5; without the gate, 17
and 15.

**The blind tables.** The gate was introduced for the tables of the benchmark (D63), so
the same four combinations were run on its 32 blind tables with the pipeline's own
functions (`runs/gate_blind.py`; the table for every system is `results/gate_blind.txt`).

| search, fits | injected period listed | rank 1 | within the first 4 | rank 1 with both K within 10 percent |
|---|---|---|---|---|
| gate, gate (the pipeline) | 27 | 26 | 26 | 18 |
| no gate, gate | 28 | 26 | 26 | 18 |
| gate, no gate | 27 | 23 | 25 | 17 |
| no gate, no gate | 28 | 23 | 25 | 17 |

The two uses of the gate separate. Without it the search lists the injected period on one
more blind table and changes no rank, and on the survey tables it lists 2.9 points more.
Without it the fits lose three of the 26 blind tables at rank 1 (mixed-0010, mixed-0015
and gaia-40041022325608704) and one of the 18 with both semi-amplitudes, where on the
survey tables they gained the semi-amplitudes of 19 in 150. A velocity below the gate is
a wrong minimum on the blind tables, which are measured with library templates at
starting labels, and a noisy measurement of the right minimum on the survey tables, which
are measured with classified templates. One threshold on the detection statistic does not
tell the two apart.

Neither use was changed. The search without the gate, or with the gate only where it
leaves the relative-velocity searches enough epochs, is a candidate for a change, to be
accepted on the end-to-end runs of the benchmark. The gate in the fits needs a criterion
other than a fixed detection statistic before it is changed.

## The shortest period searched

`albireo.rvorbit.find_period` searches, by default, from twice the smallest gap between
two epochs of the series it is given. A table with few usable epochs can have no close
pair. On 13 of the 1,200 test tables the injected period is below that limit for every
search: system 1544 has P = 0.437 d and ten usable epochs of the first component, and
nothing below 15.7 d is searched. The baseline and its deeper list have the period on
none of the 13. C and D, which search from a fixed 0.24 d, have it on all 13.

## Decision

No learned proposal is added.

- C cannot resolve a peak, and with its samples moved to maxima it does what a ladder
  around the Kepler estimate does.
- D has the resolution and a first candidate that is right on 95 percent of tables in
  its training distribution. The orbit decided after the fits is the baseline's on all
  but three of 150 tables, and on the benchmark's own tables it is below the baseline
  alone and adds nothing a deeper list does not.
- The prior of B changes two tables in 150.

The candidate list was not what limited the decision on these tables. What the experiment
measured as limiting on the survey tables is in the pipeline: the gate in the search (2.9
points of coverage at the same list length), the gate in the fits (the semi-amplitudes of
13 percent of the tables), and the default shortest period (1.1 percent of the tables).
Of the three, the gate in the search is the one that the blind tables of the benchmark
also support changing.

## Files

- `results/summary.txt`: every table in full, with the tables on which the arms differ
  and the control without the gate, written by the agent's `s23_summary.py`.
- `results/`: the split, the coverage by class of table, the paired comparisons between
  arms, the first candidates, the census of the fits, the tables below the shortest
  period searched, the timing, and `gate_blind.txt`, the gate on the 32 blind tables.
- `runs/` (not tracked): the agent's working directory with its 49 scripts, the measures
  per arm and split as JSON, the candidate lists, one record per fit, the trained
  parameters and the logs. The scripts read and write beside themselves and run from that
  directory. `runs/gate_blind.py` is the measurement on the blind tables.
