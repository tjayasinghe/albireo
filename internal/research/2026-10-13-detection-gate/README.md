# The detection gate of the period search on faint pairs (2026-10-13)

For a binary without a period the pipeline measures a velocity table against library
templates, proposes candidate periods from its periodograms, fits an orbit from each and
ranks the orbits by chi-square (`albireo.pipeline`, `period = "search"`). In the search
and in the fits a companion's velocity has no weight where its detection statistic is
below `Analysis.detection_min`, 100 (D65). The experiment on period proposals of the day
before (`../2026-10-12-period-proposals/`) recorded that this gate leaves a faint table
without the searches that need the companion, and named the search without the gate as a
candidate for a change. This note records the measurement of that change, which was not
made, and the change of the threshold that was made in its place (D71).

## Summary

- **The search without the gate does not change the period decided.** With the candidate
  fits still gated, the search of the table as measured lists the injected period within
  reach of a fit on 47 further tables of 1,200 and loses it on 3. One table gains the
  first place and four lose it.
- **The tables concerned are faint pairs of alike stars.** Neither star reaches a
  statistic of 100, the gate removes the companion at every epoch, and the correlation
  returns the epochs in either order, so the first component's column that remains holds
  the velocities of both stars. A fit to that column alone cannot confirm the period.
- **The threshold was changed, for the search and the fits alike.** Below `detection_min`
  a companion's velocity is kept where its statistic is at least 25 and at least a tenth
  of the statistic of the epoch as a whole, on a table whose two columns move in antiphase
  (a correlation coefficient below -0.9 over at least eight such epochs).
- **The benchmark is unchanged and the faint tables gain.** No velocity of the 32 blind
  tables of the benchmark is gated differently. On 1,200 held-out survey tables, which
  were selected for a recoverable orbit, the injected period is first on 1,107 against
  1,072 and first with both semi-amplitudes within 10 percent on 1,013 against 947. On
  400 tables drawn without that selection the numbers are 311 against 306 and 184
  against 169. On 122 faint tables built by the pipeline with the templates of the
  benchmark's blind tier the period is first on 64 against 53, and the bootstrap hands
  on both semi-amplitudes within 15 percent for 37 against 20.
- **An independent review changed the rule and its claims.** It moved the cut on the
  antiphase from -0.7 to -0.9, replaced the reference of the second condition by one that
  a badly fitting pair of templates cannot lower, found one way to pass the antiphase test
  with a single star, and showed that the velocities returned are noisier than those the
  gate keeps and less often grossly wrong, not more accurate.
- **End to end the gain is that of the first four candidates.** The whole pipeline was
  run before and after the change on the 21 faint tables on which the candidate fits
  decide differently. The final orbit has the injected period on 20 after the change
  against 10 before it, and both semi-amplitudes within 10 percent as well on 17 against
  8, with none lost. The pipeline compares the best four candidates by the
  disentangling's likelihood, and the injected period is among them on 20 of the 21
  tables against 11. Two benchmark systems give the same final result to every digit.

## Samples

- **Blind tables.** The 32 blind tables of the Gaia RVS benchmark, as the earlier
  experiment stored them from the pipeline's own bootstrap stage.
- **Survey tables.** Tables of the simulated survey of eclipsing binaries (D69), the
  transits out of eclipse under the classified declaration, whose templates have the
  labels and the rotation of a classification with its errors. Three samples: the 1,200
  held-out tables of the earlier experiment and the 150 of them it used for fits, both
  selected (the orbit at the known period has both semi-amplitudes within 10 percent),
  and 400 tables with at least 8 usable epochs drawn without regard to the orbit, outside
  the 1,200. The tables are rebuilt from the stored rows, which have no second minimum,
  so the decision of the epochs flagged for one (D68) is not exercised on them.
- **Faint tables built by the pipeline.** 150 systems of the survey's population with
  G_RVS between 11 and 13 and at least 12 transits, drawn at random, simulated by the
  benchmark's simulator and measured by the pipeline's bootstrap stage with the two
  templates of the benchmark's blind tier (6000 and 5000 K, not rotated), as
  `scripts/period_decision_bench.py` builds the blind tables. The stage stops for 25 of
  them because the templates measure no light, 3 have fewer than four usable epochs, and
  122 have a table and a decision. The templates are far from these stars: the faster
  star of a pair rotates at a median 46 km/s (above 30 km/s on 90 of the 122), and the
  first stars have temperatures of 4,000 to 9,500 K.

## Measures

The candidates and the candidate fits are the pipeline's own functions
(`_period_candidates`, `_orbit_over_candidates`), with a rule in place of
`_detection_mask` or, for the rule adopted, the changed package itself. The pair is
exchanged in the fits where the pipeline's rule on the light fractions allows it. The
earlier experiment exchanged it in every fit, and under that convention the same code
gives its recorded numbers on the 150 tables (137 and 115 with the gate, 140 and 136
without it).

- **Listed** is a candidate within 2 percent of the injected period, and **within reach**
  one within $`1/T`$ of it in frequency, with $`T`$ the span of the usable epochs.
- **Rank 1** is the orbit of lowest chi-square within 2 percent of the injected period,
  and **both K** adds both semi-amplitudes within 10 percent of the injected ones.
- Paired comparisons are exact two-sided sign tests.
- For the faint tables built by the pipeline the outcome is what the bootstrap hands to
  the disentangling: the period with both semi-amplitudes within 15 percent, which become
  a Gaussian prior (**prior**); the period with a semi-amplitude held or outside the
  declared range, which the disentangling then searches (**open**); the period with a
  semi-amplitude more than 30 percent off inside the range, which becomes a wrong prior
  (**off**); or another period (**wrong**).
- **The error of a velocity** is that of its pair in the order that fits better: the
  larger of the two absolute differences from the injected velocities, as measured or
  with the pair exchanged, whichever is smaller. It is compared with 20 km/s and with
  50 km/s, above which it is called gross. A first version of this note counted a
  velocity right within five quoted errors or 10 km/s. The review showed that this
  favours velocities with large errors (half of those the rule returns met it against
  the injected velocities of another epoch), and it is not used here.

## The search without the gate

On the 1,200 tables, with the candidate fits gated as in the pipeline:

| search | listed | within reach | rank 1 | rank 1, both K | rank 1 gained, lost |
|---|---|---|---|---|---|
| gated (the pipeline before the change) | 1,122 | 1,107 | 1,072 | 947 | |
| the table as measured | 1,162 | 1,151 | 1,069 | 947 | 1, 4 |

Two weaker forms of the change (the table as measured only where the gate leaves a
search fewer than four epochs, or fewer than half of them) were measured under the
convention of the earlier experiment and gain no table for three lost
(`results/search_alone_1200.txt`). Of the 43 tables on which only the search of the
table as measured lists the period, the gated fits put it first on 1, at rank 5 or lower
on 25 (a median rank of 52) and nowhere on 17.

The four tables that lose their place show three ways of losing it. On one, the highest
peak of the two-harmonic search of the table as measured lies within 2 percent of the
injected period and $`7/T`$ from it, outside the reach of a fit, and the merge of the
candidates at 2 percent drops the first component's own peak, $`0.02/T`$ from the period,
in its favour. On one, the swap-invariant search of the gated copy has the period and
that of the table as measured does not. On two, both lists have the period first, and a
further candidate fits the first component's column better.

**The reach of a candidate fit.** The 46,879 candidate fits that the earlier experiment
stored for the 150 tables give how far a fit moves from its start
(`results/fit_reach.txt`). Of the starts within $`0.5/T`$ of the injected period, 204 of
210 end on it. Of those between $`0.5/T`$ and $`2/T`$, 5 of 17 do, and of those beyond
$`2/T`$ and still within 2 percent, none of 209. Two percent of the frequency is a median
$`16.5/T`$ on these tables. A period listed within 2 percent is therefore not always one
a fit can reach, and both measures are given above.

**The 43 tables.** They are faint pairs of alike stars (`results/newly_listed.txt`). The
median G_RVS is 12.1, the light ratio is above a third on all of them with a median of
0.80, and a median 40 percent of the usable epochs have the pair in the other order (a
fifth or more on 40 of the 43). The medians of the two statistics are 48 and 40, and the
companion is below 100 at every usable epoch on 28. With the companion removed, the
column that is left holds the velocities of both stars and has no pair to exchange.

## Companion velocities below the threshold

Every usable epoch of the survey tables outside the 1,200 with at least 8 usable epochs
(147,796 epochs of 5,286 tables) was compared with the injected velocities
(`results/velocities_fixed_tolerance.txt`).

| companion's statistic | epochs | within 20 km/s % | above 50 km/s % | median error, km/s |
|---|---|---|---|---|
| 0 to 9 | 16,138 | 9.2 | 76.7 | 110.8 |
| 9 to 25 | 14,068 | 34.7 | 45.0 | 38.8 |
| 25 to 50 | 12,445 | 57.1 | 24.8 | 15.8 |
| 50 to 100 | 13,448 | 66.7 | 21.6 | 11.1 |
| 100 to 300 | 22,099 | 73.9 | 18.4 | 7.5 |
| 300 to 1000 | 23,041 | 80.5 | 12.9 | 4.5 |
| above 1000 | 46,557 | 93.5 | 3.4 | 1.8 |

A statistic between 25 and 100 is not worth the same on every table. By the median
statistic of the first component over the table, the gross errors among the velocities
with a statistic of 25 to 100 are 8 to 14 percent where that median is below 1,000, 29
percent between 1,000 and 5,000, and 43 percent above 5,000. A weak solution beside a
strong first component is often the second template fitted to the residuals of the
first. On a faint pair there is no such residual above the noise, and the velocity is a
noisy measurement. The velocities that the fixed threshold keeps, those above 100, have
gross errors at 9.4 percent of epochs and a median error of 3.2 km/s.

## Rules for the threshold

A rule sets the threshold of an epoch at $`\min(T, \max(F, s/r))`$, with $`T`$ the
declared threshold of 100, $`F`$ a floor of 25 and $`s`$ a reference statistic.

| reference $`s`$ | $`r`$ | velocities kept below 100 | within 20 km/s % | above 50 km/s % |
|---|---|---|---|---|
| none (the floor alone) | | 25,893 | 62.1 | 23.1 |
| the first component at the epoch | 5 | 9,813 | 68.2 | 10.0 |
| the first component, median over the table | 5 | 9,663 | 67.7 | 10.1 |
| the epoch as a whole | 5 | 5,444 | 71.2 | 6.3 |
| the epoch as a whole | 10 | 8,413 | 73.2 | 6.5 |
| the epoch as a whole | 20 | 11,268 | 73.5 | 7.3 |
| the epoch as a whole | 50 | 14,730 | 73.3 | 8.5 |

The statistic of the epoch as a whole is the rise in chi-square when every template is
removed, `chi2_null - chi2` of the table. It is the better reference because it does not
fall where the lines of two bright stars blend, while the statistics of both stars do: a
blended epoch of a bright pair then keeps its threshold, and a faint epoch does not. One
part in ten is the share of a companion with a third of the first component's light
where the lines are separated, since the statistic goes as the square of the light, and
a third is the ratio within which the pipeline treats two components as alike. The floor
of 25 is the statistic under which the table's own summary counts a weak detection.

The table's chi-square is that of the templates at the held light fractions, while the
statistics of the components are those of free amplitudes. The review showed that the
whole can then be far below the first component's own statistic, and negative, where the
held fractions fit badly (template lines twice too deep, or stars six times broader than
the templates, on a strongly detected pair), which would lower the threshold of a bright
pair. The reference adopted is therefore the larger of the whole and the first
component's statistic, and an epoch whose whole is not positive keeps the declared
threshold. On the tables of this note the two references keep the same velocities. Such
epochs are 14 of the 2,734 usable epochs of the pipeline-built tables and 1 of the
190,056 of the survey.

**The antiphase of the two columns.** On the faint tables built by the pipeline, whose
templates are far from the stars, the two conditions above return 833 velocities on 87
tables, of which 50.4 percent are within 20 km/s and 24.7 percent are gross errors. A
template that does not fit the first star leaves its lines to the second template, which
then follows the first star. The velocities of the two stars of a binary lie on a line
of negative slope whatever the period (Wilson 1941), and an epoch in the other order
lies on the line of reciprocal slope, so the correlation coefficient of the two columns,
over the usable epochs that meet the two conditions, separates the two cases
(`results/velocities_fixed_tolerance.txt`). The tables have at least eight such epochs.

| coefficient | survey: tables | velocities returned | within 20 km/s % | above 50 km/s % | pipeline-built: tables | velocities returned | within 20 km/s % | above 50 km/s % |
|---|---|---|---|---|---|---|---|---|
| below -0.95 | 366 | 5,907 | 82.2 | 1.2 | 30 | 392 | 80.1 | 1.5 |
| -0.95 to -0.90 | 49 | 765 | 59.7 | 9.4 | 5 | 64 | 48.4 | 10.9 |
| -0.90 to -0.80 | 16 | 246 | 45.1 | 20.7 | 2 | 23 | 39.1 | 8.7 |
| -0.80 to -0.70 | 7 | 82 | 48.8 | 22.0 | 1 | 18 | 16.7 | 16.7 |
| -0.70 to -0.50 | 7 | 109 | 59.6 | 20.2 | 3 | 32 | 37.5 | 15.6 |
| -0.50 to 0.00 | 10 | 124 | 33.9 | 32.3 | 5 | 82 | 6.1 | 72.0 |
| above 0.00 | 7 | 75 | 8.0 | 72.0 | 10 | 125 | 8.0 | 79.2 |

The condition adopted is a coefficient below -0.9 over at least eight epochs. With it
the rule returns 6,672 velocities on 415 survey tables, of which 79.6 percent are within
20 km/s and 2.2 percent are gross errors, and 456 on 35 pipeline-built tables, with 75.7
and 2.9 percent. Their median error is 11 to 12 km/s, against 3.2 and 10.6 km/s for the
velocities above 100 of the same two samples, whose gross errors are 9.4 and 19.7
percent. The velocities returned are noisier than those the gate keeps and less often
grossly wrong. The band the cut admits last, between -0.95 and -0.90, has the gross
errors of the velocities above 100.

The cut was first fixed at -0.7, after the first 50 of the 150 pipeline-built systems
had been read and before the other 100 and the 1,050 survey tables outside the 150. It
was moved to -0.9 after the review, which measured the rule on the runs of the survey
it had not been measured on (third light, emission in the lines, the resolving power of
each transit, and one catalogue template for both stars in each). Over the eight
combinations of run and declaration the velocities returned are within 20 km/s for 77
to 84 percent where the coefficient is below -0.9 and for 28 to 62 percent between -0.9
and -0.8. On the samples of this note the two cuts give the same counts of decisions.
Unrelated columns pass -0.9 over eight epochs about once in a thousand tables.

The coefficient gives the epochs equal weight. It was first weighted by the quoted
errors, with which one epoch of small errors can carry it, and the two forms pass the
same tables on every sample. A few epochs of large amplitude can carry it at any
weighting.

## The period decided

Rank 1 and rank 1 with both semi-amplitudes within 10 percent, with gained and lost
against the fixed threshold in brackets. The last row is the changed package itself. The
rows of the two conditions are the rule before the review, with the whole alone as the
reference.

| threshold | blind, 32 | survey, 150 | survey, 400 drawn without selection | survey, 1,200 |
|---|---|---|---|---|
| fixed at 100 (before the change) | 26, 18 | 136, 114 | 306, 169 | 1,072, 947 |
| none | 23, 17 | 139 (5, 2), 135 (21, 0) | 299 (9, 16), 202 (41, 8) | not run |
| fixed at 25 | 24, 17 | 140 (5, 1), 136 (22, 0) | 312 (8, 2), 202 (41, 8) | not run |
| 25 and a tenth of the epoch | 26, 18 | 140 (5, 1), 126 (12, 0) | 310 (5, 1), 190 (21, 0) | 1,106 (36, 2), 1,018 (71, 0) |
| the rule adopted | 26, 18 | 141 (5, 0), 126 (12, 0) | 311 (5, 0), 184 (15, 0) | 1,107 (36, 1), 1,013 (66, 0) |

The 150 tables are among the 1,200. On the other 1,050, which were not read before the
cut of -0.7 was fixed, the period is first on 966 against 936 and both
semi-amplitudes are within 10 percent on 887 against 833.

On the blind tables the rule adopted gates no velocity differently from the fixed
threshold (`results/blind_masks.txt`): the largest share of the epoch's statistic that a
companion below 100 has on them is 0.066. Their ranking is unchanged, and
`scripts/period_decision_bench.py` run with the changed package prints its recorded row
(26 first, 26 within the first four, 5 absent, 17 with both semi-amplitudes within 5
percent). No gate, and a threshold of 25, each lose first places on the blind tables, as
D65 recorded. On the survey tables drawn without selection no gate loses 16 first places
for 9 gained, and both of them lose the two semi-amplitudes on 8 tables. The gain of an
ungated fit that the earlier note reported on the selected tables (21 of 150 with both
semi-amplitudes) is in part that selection: those tables were kept because their
velocities, ungated, give the orbit.

On the 927 of the 1,200 tables whose first component has a median statistic above 500
the rule adopted changes no decision. The gains are on the others. Where that median is
below 125 (122 tables) the period is first on 60 against 31, and both semi-amplitudes
are within 10 percent on 53 against 11. Between 125 and 500 (151 tables) the period is
first on 135 against 129, and both semi-amplitudes on 113 against 89. The one table that
loses its first place has the injected orbit at rank 3 with semi-amplitudes 4 percent
off, where the fixed threshold had it first at 11.5 percent. The rule adds candidates
where it acts, since more searches run: a median of 80 against 63 on the 160 tables of
the 1,200.

**Faint tables built by the pipeline.**

| threshold | period first | prior | open | off | wrong |
|---|---|---|---|---|---|
| fixed at 100 | 53 | 20 | 28 | 3 | 69 |
| none | 68 | 43 | 16 | 7 | 54 |
| 25 and a tenth of the epoch | 67 | 40 | 19 | 6 | 55 |
| the rule adopted | 64 | 37 | 22 | 3 | 58 |

The two tables between 15 and 30 percent are left out of the columns. With the rule
adopted 12 tables go from a wrong period to a prior, 5 from an open companion to a
prior, one from between 15 and 30 percent to a prior, one from open to between 15 and 30
percent, and one from a prior to a wrong period, with the injected orbit then at rank 3
behind an alias. The number of wrong priors does not rise, where it doubles with the
first two conditions alone. On the 72 tables read after the first form of the rule was
fixed, 7 go from a wrong period to a prior and 4 from open to a prior, and none loses.

## The review

An Opus agent reviewed the change against the staged code, the tests and the stored
tables, with its own simulations (`runs/review_gate/`). It found no defect in the logic
and confirmed the masks of the blind tables and the counts of the decisions above. What
it changed or added:

- **The cut on the antiphase**, from -0.7 to -0.9 (above).
- **The reference of the second condition**, the larger of the whole and the first
  component's statistic (above).
- **A single star that passes the antiphase test.** Two identical templates narrower
  than the lines of a single star divide its line between them in either order, which is
  antiphase about the star's own velocity. A simulated star at rest whose lines are three
  times as broad as two identical templates gives a coefficient of -0.86 at a S/N of 5,
  which the cut of -0.9 refuses and that of -0.7 did not, and -0.94 at a S/N of 9, where
  the statistics are above 100 and the fixed threshold already returns an orbit of two
  components with semi-amplitudes of 67 and 52 km/s. With an orbital motion of 8 km/s or
  more of the star, or with two templates that are alike and not identical, the
  coefficient is between -0.7 and +0.1. On the stored tables measured with one catalogue
  template for both stars, 12 of 514 tables returned velocities of which fewer than half
  were right under the cut of -0.7. The failure is narrowed by the cut and not removed.
- **The measure of a right velocity** (under Measures).
- **Exchanged epochs of unequal semi-amplitudes** lie on two lines, and the coefficient
  is then $`-2q/(1+q^2)`$ for a ratio $`q`$ with half of the epochs exchanged: -0.88 at
  0.6. The rule does not act on such a table, which is the behaviour before the change.
- **A `detection_min` above 100** raises the threshold of a strongly detected pair only.
  On a weak pair in antiphase the threshold is the rule's.
- **The statistics are not corrected for a declared noise correlation**, as the declared
  threshold is not. For a single star and two templates the companion's statistic stays
  below 18 in white noise and reaches 30 at a lag-one coefficient of 0.27, that of these
  tables. At 0.6 and 0.81 it is above 25 at 10 to 50 percent of epochs, which the
  antiphase condition then leaves to chance.

## The pipeline end to end

Before the change was committed the user asked for a run of the whole pipeline. The blind
tier of the benchmark was run by `pipeline.run_pipeline` with the package before the
change (commit 4248adc) and with the changed one, on the same simulated epochs: the
period search and its candidate fits, the comparison of the best four candidates by the
disentangling's marginal likelihood, the disentangling, the label fit, TODCOR against the
fitted templates and the orbit fit, at the settings of `scripts/gaia_rvs_benchmark.py`.

**Systems.** They were fixed before a result was read. Of the 122 faint tables built by
the pipeline, the changed package gates 35 differently from the fixed threshold, and on
21 of them the outcome of the candidate fits changes: what the bootstrap would hand on,
or the rank of the injected period. Those 21 were run, in a seeded random order. Two of
the 33 systems of the recorded benchmark were run as a check that nothing changes where
the gate does not, the first of the field population and one of the Gaia population. The
other 14 of the 35, and three further benchmark systems, were planned after them and were
not run.

**Measures.** They were fixed before a result was read: the period of the final orbit,
which is fitted to the velocities measured against the fitted templates, within 2 percent
of the injected one; that period with both semi-amplitudes within 10 percent, and within
5 percent, in the order in which the pipeline's own comparison with the injected values
takes the components; and the same for the elements of the disentangling.

**Runs that were stopped.** The disentangling's period is bounded to within 3 percent of
the period the bootstrap hands on (`_bootstrap_spec`), and the final orbit is a local fit
started from the disentangling's period (`_orbit`). A run whose bootstrap handed on a
period more than 10 percent from the injected one therefore cannot recover it. Where the
bootstrap also leaves a semi-amplitude open, the disentangling searches the declared
range with a velocity budget of about 1,000 km/s or more (998 km/s on 14 runs before the
change and up to 2,270, against 479 to 1,198 in the runs after it), and two such runs
had not ended after 77 and 109 minutes. From then on a run was stopped 30 minutes after
such a bootstrap, under either package, and counted as run and not recovered. Six runs
of the package before the change were stopped and none of the changed one. No run
failed.

| the final result, of 21 | before the change | after it | gained | lost |
|---|---|---|---|---|
| period within 2 percent | 10 | 20 | 10 | 0 |
| period, and both semi-amplitudes within 10 percent | 8 | 17 | 9 | 0 |
| period, and both semi-amplitudes within 5 percent | 6 | 13 | 8 | 1 |
| the same from the disentangling's elements, within 10 percent | 8 | 17 | 10 | 1 |

Every period within 2 percent is within 0.1 percent. The two benchmark systems give the
same final period and semi-amplitudes to every digit under the two packages, with
semi-amplitudes within 0.2 and 0.7 percent of the injected ones.

**The first four candidates, not the first.** The pipeline does not take the first orbit
of the chi-square ranking. It compares the best four by the disentangling's likelihood
(`Analysis.period_decision_candidates`). On the 21 tables the injected period is first on
8 with the fixed threshold and on 19 with the rule, and within the first four on 11 and
on 20 (`results/end_to_end.txt`). End to end it is recovered on 10 and on 20, which is
the gain of the first four and not that of the first place. With the rule the
disentangling takes the injected period wherever it is among the four. With the fixed
threshold it does so on 10 of the 11: on G-10-058, where the injected period is fourth,
it prefers the half period by 169 nats. The disentangling chose another candidate than
the first on 9 of the runs before the change, the injected period on two of them, and on
2 of the runs after it, the injected period on one. That one is K-12-041, the table that
loses the first place under the rule, with the injected period third behind its half.
Over all 122 tables the injected period is within the first four on 60 with the fixed
threshold and on 69 with the rule, with none lost, against 53 and 64 for the first
place.

**The one system neither package recovers** is K-13-108, at G_RVS 12.5: the injected
period is absent from the ranking with the fixed threshold and nineteenth with the rule.

**The semi-amplitudes.** On the 10 systems whose period both packages recover, the larger
error of the two semi-amplitudes is smaller with the changed package on 7 and larger on
3, with medians of 2.4 percent before the change and 3.3 after it. These systems do not
show a difference in the semi-amplitudes, and the rule's gain is in the periods. The one
system that leaves the 5 percent row is F-11-042, from 2.1 to 5.8 percent, and the one
that leaves the row of the disentangling's elements is G-11-083, from 3.7 to 28.4
percent, whose final orbit is outside 10 percent with either package (13.2 and 25.7
percent).

**Run time.** The runs before the change took 9.5 hours and those after it 5.8, with the
stopped runs counted to the moment they were stopped. A bootstrap that hands on a wrong
short period with a semi-amplitude left open is the pipeline's slowest path, and the
rule takes it less often.

**Scope.** The 21 tables are those on which the candidate fits decide differently, so the
counts are of that group and not of faint pairs at large: on the other 101 tables of the
122 the candidate fits hand on the same outcome, and on 87 of them the gate is the same
and the two packages run the same computation. The noise is Gaussian, as in every table
of this note.

## The rule adopted and its limits

Below `detection_min` a companion's velocity is kept where its statistic is at least 25
and at least a tenth of the larger of `chi2_null - chi2` and the first component's
statistic, on a two-component table whose columns have a correlation coefficient below
-0.9 over at least eight usable epochs that meet those two conditions. The search and
the candidate fits use the same gated copy, as before.
`bootstrap.detection_gate` of `result.json` gains `kept_below`, the velocities kept, and
`antiphase`, the number of epochs and the coefficient of the test where it was made.

- **A weak companion beside a strong first component stays gated.** Its share of the
  epoch is below a tenth whether it is a detection or not: two blind tables have a
  companion below 100 at every epoch that is right at 55 of 58 epochs, and one has a
  companion that is wrong at all 15. The fixed threshold of 25 shows what is there on the
  survey tables (both semi-amplitudes on 22 further tables of 150, against 12) and what
  it costs on the blind ones (two first places). The line of the two columns could
  decide such epochs one by one, and that is not built.
- **Identical templates narrower than the lines of a single star** can pass the
  antiphase test (the review, above).
- **Fewer than eight epochs do not show the antiphase**, and the table keeps the fixed
  threshold. So does a pair whose errors are more than about a quarter of its
  semi-amplitudes, for which the coefficient does not reach -0.9.
- **At an epoch whose companion is still gated, the first component's velocity of an
  alike pair can be the other star's**, and a single velocity cannot be exchanged
  (`results/first_column.txt`).
- **The noise of every table here is Gaussian**, with a lag-one coefficient of 0.27. A
  floor of 25 has not been measured against cosmic rays or bad pixels, which a threshold
  of 100 tolerates better.
- **The whole pipeline was run on 23 systems.** They are the 21 tables on which the
  candidate fits decide differently and two systems of the benchmark (above). The 14
  tables gated differently with the same outcome in the candidate fits were not run, and
  the 33 systems of the benchmark, whose bootstrap is unchanged, were not run again.
- **Tables of three components** keep the fixed threshold.
- **The merge of the candidates at 2 percent is wider than the reach of a fit.** Two
  percent of the frequency is a median $`16.5/T`$ on the survey tables, and a fit moves
  less than $`2/T`$. Where a peak of one search lies within 2 percent of another search's
  peak and further than that from it, the merge keeps the one it meets first. One table
  above loses its period in this way. The merge was not changed.

## Files

- `results/`: the tables above in full. `search_alone_1200.txt`, `fit_reach.txt`,
  `coverage_within_reach.txt` and `newly_listed.txt` are the search without the gate;
  `velocities_fixed_tolerance.txt` the velocities against the injected ones and the
  coefficient of the two columns; `first_column.txt` the first component at a gated
  epoch; `rules_blind.txt`, `rules_m2.txt`, `rules_open.txt` and `rules_test.txt` the
  period decided under each rule measured before the review;
  `faint_tables_built_by_the_pipeline.txt` the 150 systems measured by the pipeline's own
  stage under those rules; `changed_package.txt` the changed package on every sample;
  `blind_masks.txt` its gate against the fixed threshold on the blind tables;
  `end_to_end.txt` the pipeline run end to end before and after the change, star by
  star, with the runs that were stopped and the ranks in the candidate fits.
  `velocities_survey.txt`, `velocities_blind.txt`, `statistic_scales_survey.txt`,
  `statistic_scales_blind.txt`, `antiphase_survey.txt` and `correlation_forms.txt` are
  the tables of the first version, by the criterion of five quoted errors.
- `runs/` (not tracked): the scripts and their outputs, and the reviewer's under
  `runs/review_gate/`. They read the tables and the helper modules of
  `../2026-10-12-period-proposals/runs/`. The end-to-end runs are `e2e_plan.py` (the
  systems), `e2e_run.py` (one star under one package), `e2e_dispatch.py` (the runs, as
  memory allows), `e2e_watch.py` (the stops) and `e2e_read.py` (the measures), with
  every star's report under `runs/e2e/`.
