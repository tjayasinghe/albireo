# Where the true period sits in the blind tier's chi-square ranking (D64, 2026-09-11)

The blind tier searches for the period: library templates measure a velocity table, four
periodograms propose candidate periods (`albireo.pipeline._period_candidates`), an
eccentric Keplerian is fitted from every candidate and the chi-square ranks them
(`_orbit_over_candidates`), and the best few are then measured against each other by the
disentangling's own marginal likelihood (`_decide_period_by_disentangling`). The third
run compared four; the default has since been raised to eight. The run's record keeps
only the four that were compared, so nobody had seen the rest of the ranking. This note
recovers the full ranking offline for all 33 blind stars of the two archived runs
(`gaia5`, 14 systems; `field5`, 19) and reports where the injected period sits in it.
Velocity tables only: no spectra were read, no `Disentangler` was built, one process.

**The measurement is a re-run of the pipeline's own two functions on the pipeline's own
input.** Each blind directory's `template_velocities.rv` is the bootstrap table the
period search actually ran on. It was read back into a `VelocityTable`, passed through
`_period_candidates(table, swap_invariant=True)` and then through
`_orbit_over_candidates(ctx, table, candidates, circular=False)` with a
`types.SimpleNamespace` standing in for the `_Context` (only `settings` and `flag` are
used, as `tests/test_pipeline.py` establishes), and `record["ranked"]` read off.
`swap_invariant=True` is what `_bootstrap` passes at line 2454 of `pipeline.py`, and it is
the right value here
because every blind table has two components; the branch is what survives components
exchanged between epochs. The settings are the run's own, from the two `manifest.json`
files: `k_min = 2.0`, `k_max = 250.0`, `ecc_max = 0.9`, and `circular` left at its
default `False`, since `albireo.benchmark._analysis` passes no `circular` and the
benchmark script sets no `analysis` overrides. That is 1296 candidate starts fitted over
the 33 tables, a median of 40 candidates proposed per star and 30 distinct fitted orbits
surviving (range 11 to 43, 1020 in total), in 1300 s of wall time in one process, a
median of 38 s per star and 71 s on the worst.

**The written table is the table *after* the exchange, and had to be put back before
anything reproduced.** `_write_template_table` writes `bootstrap["objects"]["table"]`,
which in `_bootstrap` is the table the winning orbit chose, after `reassign_by_orbit` has
exchanged the components at some epochs; `_period_candidates` and the candidate loop both
see the table as measured, before any exchange. Run on the file as written, the
periodogram of `gaia-37608387208382848` peaks at 0.26678 d against the 0.18652 d the run
recorded, and the ranking is unrecognizable. The exchange is exactly recoverable from the
light column: the bootstrap correlation runs with `light = "global"`, one amplitude per
template shared over every epoch, so an unexchanged table has a constant light row, and
`reassign_by_orbit` moves the light with the velocity. An epoch whose light pair is the
other way round is an exchanged epoch. Undoing it puts back 112 epochs over 28 of the 33
stars, and the count agrees with the run's own `bootstrap: N of M epochs had their
components re-assigned` flag on 33 of 33 stars. With the table put back, the same star's
periodogram peaks at 0.1865228 d against the recorded 0.1865218671, a relative difference
of 5e-6 and less than half of one step of a frequency grid whose step in period is

```math
\frac{dP}{P} = \frac{P}{N_{\rm os} T},
```

here 1.2e-5 with $`N_{\rm os} = 10`$ samples per $`1/T`$ and a baseline $`T`$ of 1525 d.
The residual difference is the rounding of the epoch times, which moves the shortest gap
and with it the end of the frequency grid.

**Rank 1 of the recovered ranking is the period the run's own chi-square chose on 30 of
the 33 stars, and the run's whole recorded list reproduces entry for entry on 26.** Two
independent checks are available. On all 14 Gaia stars the run recorded
`bootstrap.decision.candidates`, the top four with their table chi-squares, so rank 1 is
checked against `candidates[0]` and the whole compared list against the first four
entries. On 18 of the 19 field stars there is no `decision` block at all, so
`bootstrap.period` *is* the chi-square winner and checks rank 1 directly. The chi-squares
agree to better than 0.1 percent throughout, for instance 1546.7 against the recorded
1546.6 on `gaia-37608387208382848` and 22126.8 exactly on `gaia-45354171749336960`. Seven
stars' lists differ, and on four of them the difference is a single orbit inserted or
removed with every shared entry in the same order: `gaia-37608387208382848` (an extra
0.17567 d at rank 4), `gaia-53680017390963072` (an extra 4.91253 d at rank 2),
`gaia-7289006178185856` (the run's 0.25993 d missing) and `mixed-0015` (two adjacent
entries, 0.16724 and 0.21379 d, swapped, their chi-squares differing by 1 percent). The
cause is the 6-decimal rounding of the written file: the periodogram's near-degenerate
peaks change order, the greedy 2 percent deduplication then keeps a slightly different
starting period, and one start out of forty converges somewhere else. The three genuine
rank-1 disagreements are `gaia-47620265212420096`, where the run's ranking held an orbit
at 4.84532 d and chi-square 724.8 that no reproduced candidate reached, so every
reproduced entry sits one place lower in the run's own ranking and the correction is
applied below; `gaia-56716765427964800`, where the reproduced list reached a *better*
orbit (0.67875 d at 2423.0) than anything the run found, whose own rank 1 was 1.12943 d at
6167.4, and whose recorded four are the reproduced ranks 2 to 5; and `mixed-0019`, an
11-epoch table with 9 usable epochs whose 42 starts collapse to 11 distinct valid orbits
and on which the run's 0.53104 d
does not appear at all. None of the three moves a bucket: the first stays in ranks 5-8
after its correction, and the truth is absent from the ranking on the other two either
way. The one fact established before this work reproduces: on `gaia-37608387208382848`
the truth is at chi-square rank 3, at chi-square 2027.6 against 1546.7 at rank 1. One
correction to the framing of that fact: the semi-amplitudes of 112.5 and 112.4 km/s in
the record are the coarse semi-amplitude scan's best, not the table's, since that
candidate's `scan_moved` is true; the table's own orbit at 6.11903 d has K of 135.7 and
139.0 km/s against a truth of 68.5 and 68.7, close to twice the truth.

## Per star

`rank` is 1-based over the distinct valid orbits in chi-square order, the first entry
within 2 percent of the injected period; `rank 5%` is the same at the looser tolerance.
`chi2 at truth` and `delta` refer to the 2 percent entry, so they are `n/a` wherever that
entry is absent, `gaia-17170287112790656` included even though it has a 5 percent entry at
rank 16 and chi-square 22741.7. The rank of `gaia-47620265212420096` carries the +1
correction described above.

| population | star | P_true (d) | e | K1+K2 | epochs | usable | proposed | distinct | rank 2% | rank 5% | chi2 rank 1 | chi2 at truth | delta | P chosen (d) | recovered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gaia | gaia-7289006178185856 | 0.5440 | 0.21 | 261 | 10 | 10 | 33 | 24 | absent | absent | 843.3 | n/a | n/a | 0.30892 | no |
| gaia | gaia-17170287112790656 | 0.5786 | 0.00 | 296 | 11 | 11 | 42 | 23 | absent | 16 | 13723.1 | n/a | n/a | 1.87002 | no |
| gaia | gaia-51574864941234048 | 0.7234 | 0.13 | 28 | 14 | 14 | 40 | 32 | 14 | 14 | 182.9 | 507.6 | +324.7 | 0.26094 | no |
| gaia | gaia-40041022325608704 | 2.8086 | 0.01 | 205 | 12 | 12 | 37 | 30 | 1 | 1 | 100.1 | 100.1 | 0.0 | 2.80862 | yes |
| gaia | gaia-45354171749336960 | 3.8268 | 0.00 | 150 | 25 | 25 | 36 | 30 | 1 | 1 | 22126.8 | 22126.8 | 0.0 | 3.82604 | yes |
| gaia | gaia-56716765427964800 | 5.3561 | 0.09 | 128 | 12 | 12 | 37 | 32 | absent | absent | 2423.0 | n/a | n/a | 0.40417 | no |
| gaia | gaia-45788547559850496 | 5.6093 | 0.01 | 152 | 15 | 15 | 42 | 36 | 1 | 1 | 549.7 | 549.7 | 0.0 | 5.60912 | yes |
| gaia | gaia-37608387208382848 | 6.1037 | 0.03 | 137 | 15 | 15 | 35 | 29 | 3 | 3 | 1546.7 | 2027.6 | +480.9 | 0.57230 | no |
| gaia | gaia-56861694804053120 | 6.8142 | 0.42 | 144 | 15 | 14 | 41 | 38 | 1 | 1 | 46.7 | 46.7 | 0.0 | 6.81405 | yes |
| gaia | gaia-47620265212420096 | 7.6752 | 0.00 | 53 | 13 | 13 | 36 | 31 | 6 | 6 | 1187.5 | 1461.6 | +274.2 | 5.10109 | no |
| gaia | gaia-50868531799454720 | 8.8642 | 0.05 | 108 | 16 | 16 | 38 | 26 | 1 | 1 | 18.8 | 18.8 | 0.0 | 8.86435 | yes |
| gaia | gaia-51884824140205824 | 17.7539 | 0.48 | 103 | 24 | 23 | 36 | 30 | 1 | 1 | 66.5 | 66.5 | 0.0 | 17.75372 | yes |
| gaia | gaia-53290511099783296 | 18.8885 | 0.41 | 69 | 16 | 16 | 42 | 41 | absent | absent | 2879.0 | n/a | n/a | 20.34552 | no |
| gaia | gaia-53680017390963072 | 29.6049 | 0.47 | 199 | 17 | 17 | 43 | 30 | 1 | 1 | 2554.1 | 2554.1 | 0.0 | 29.59911 | yes |
| field | mixed-0003 | 1.3226 | 0.00 | 273 | 53 | 53 | 36 | 29 | 24 | 24 | 74255.6 | 250307.4 | +176051.9 | 0.26092 | no |
| field | mixed-0008 | 2.0895 | 0.00 | 72 | 38 | 38 | 43 | 39 | 1 | 1 | 3569.5 | 3569.5 | 0.0 | 2.08947 | yes |
| field | mixed-0002 | 3.5735 | 0.00 | 193 | 28 | 28 | 34 | 29 | 1 | 1 | 272.4 | 272.4 | 0.0 | 3.57351 | yes |
| field | mixed-0014 | 3.9498 | 0.00 | 129 | 48 | 46 | 43 | 27 | 19 | 19 | 112583.9 | 295016.7 | +182432.8 | 0.18312 | no |
| field | mixed-0018 | 3.9864 | 0.00 | 104 | 71 | 71 | 35 | 29 | 1 | 1 | 1345.4 | 1345.4 | 0.0 | 3.98631 | yes |
| field | mixed-0000 | 5.0242 | 0.00 | 137 | 47 | 47 | 33 | 27 | 1 | 1 | 965.1 | 965.1 | 0.0 | 5.02417 | yes |
| field | mixed-0009 | 6.4820 | 0.00 | 134 | 35 | 35 | 36 | 30 | 1 | 1 | 121.5 | 121.5 | 0.0 | 6.48205 | yes |
| field | mixed-0012 | 7.3746 | 0.00 | 119 | 100 | 100 | 37 | 32 | 1 | 1 | 16837.6 | 16837.6 | 0.0 | 7.37479 | yes |
| field | mixed-0006 | 19.1633 | 0.17 | 96 | 53 | 53 | 43 | 33 | 1 | 1 | 3158.5 | 3158.5 | 0.0 | 19.16333 | yes |
| field | mixed-0016 | 23.6586 | 0.64 | 120 | 50 | 50 | 33 | 27 | 1 | 1 | 761.7 | 761.7 | 0.0 | 23.65944 | yes |
| field | mixed-0007 | 45.3942 | 0.50 | 68 | 38 | 36 | 41 | 40 | 1 | 1 | 203.1 | 203.1 | 0.0 | 45.40144 | yes |
| field | mixed-0004 | 116.8901 | 0.72 | 65 | 80 | 56 | 47 | 42 | 1 | 1 | 13897.8 | 13897.8 | 0.0 | 116.36488 | yes |
| field | mixed-0011 | 133.2333 | 0.39 | 63 | 24 | 24 | 42 | 35 | absent | absent | 1816.9 | n/a | n/a | 2.78679 | no |
| field | mixed-0013 | 221.9622 | 0.66 | 60 | 43 | 43 | 34 | 32 | 1 | 1 | 22186.8 | 22186.8 | 0.0 | 221.50599 | yes |
| field | mixed-0017 | 308.6882 | 0.67 | 51 | 22 | 16 | 45 | 43 | 2 | 2 | 357.8 | 460.0 | +102.2 | 2.84147 | no |
| field | mixed-0010 | 310.0558 | 0.70 | 43 | 62 | 62 | 41 | 26 | absent | absent | 1477.3 | n/a | n/a | 0.18239 | no |
| field | mixed-0019 | 330.2300 | 0.58 | 45 | 11 | 9 | 42 | 11 | absent | absent | 7170.0 | n/a | n/a | 0.53104 | no |
| field | mixed-0015 | 552.5478 | 0.48 | 34 | 27 | 26 | 49 | 22 | absent | absent | 121408.2 | n/a | n/a | 0.16724 | no |
| field | mixed-0005 | 844.3146 | 0.62 | 40 | 29 | 29 | 44 | 35 | absent | absent | 668.8 | n/a | n/a | 0.33260 | no |

**The ranking is bimodal: the truth is either first or nowhere near first.** At the 2
percent tolerance the counts are these, and the two populations behave alike in shape
even though the Gaia tables are far sparser.

| population | rank 1 | ranks 2-4 | ranks 5-8 | ranks 9-20 | rank > 20 | absent | n |
|---|---|---|---|---|---|---|---|
| gaia | 7 | 1 | 1 | 1 | 0 | 4 | 14 |
| field | 11 | 1 | 0 | 1 | 1 | 5 | 19 |
| both | 18 | 2 | 1 | 2 | 1 | 9 | 33 |

At the looser 5 percent tolerance exactly one star moves, `gaia-17170287112790656` from
absent to rank 16, giving 18 / 2 / 1 / 3 / 1 / 8 over both populations. Of the 33 stars,
18 have the truth at rank 1 and those same 18 are exactly the 18 the run recovered; 21 of
33 have it anywhere in the top eight and 24 of 33 anywhere in the ranking at all. For the
nine stars where no entry lies within 2 percent of the truth the nearest fitted period is
4.3 percent away on `gaia-17170287112790656` and 5.2 percent on
`gaia-53290511099783296` (whose chosen 20.34552 d is 7.7 percent off and is scored a
miss on that margin alone), but 10 to 100 percent away on the other seven, so on those
seven the truth was never reachable by any decision rule applied to this ranking.

**The gain available from raising the count from four to eight is one star out of fifteen
misses, and that star still has to win the comparison.** Of the 15 stars the run missed,
2 already had the truth among the four offered: `gaia-37608387208382848` at rank 3 and
`mixed-0017` at rank 2. Raising the count to eight adds exactly one more,
`gaia-47620265212420096` at rank 6. The other 12 misses have the truth at rank 14, 19, 24
or absent, beyond any candidate count the comparison could afford, since the comparison
costs one coarse scan per candidate and the 56 recorded scans ran 16 to 51 s each, a
median of 33 s. That is the ceiling: at most one system of 33, and only if the
disentangling's marginal likelihood then prefers it over five aliases. On the one star
where the answer is known it did not:
`gaia-37608387208382848` had the truth in front of it at rank 3 and preferred a wrong
0.5723 d by 251 nats.

**The field population never ran the comparison at all, which matters more than the
count.** `bootstrap.decision` is absent from the report of 18 of the 19 field stars, so
their period is the velocity table's chi-square winner and nothing else; only
`mixed-0015`, evidently re-run later, carries a decision block, and on it the truth is
absent from the whole ranking. The Gaia run compared four candidates on all 14 stars. So
the field row of the bound above is not a statement about four against eight: it says that
enabling the comparison at all would have put the truth in front of it on exactly one of
that population's eight misses (`mixed-0017`, rank 2), and that going from four to eight
adds nothing further there.

**No star the run got right has its truth outside the top eight, so none can be lost by
the truth falling out of the offered set.** All 18 recovered stars have the truth at
chi-square rank 1. The risk of raising the count is of the other shape: the comparison
picks the highest marginal likelihood, not the highest-ranked candidate, so four extra
candidates are four extra chances for an alias to win. What the
chi-square can say about that is that the new entrants are not close rivals. Among the 18
recovered stars the chi-square at rank 5, the first candidate a count of eight would add,
exceeds rank 1's by 758 to 193444 (the two smallest are `mixed-0007` at +758 and
`gaia-40041022325608704` at +865), and the smallest ratio anywhere is 1.42 on
`mixed-0013`. On the seven recovered Gaia stars the comparison as run confirmed the
table's choice every time, by 113 to 11984 nats over the runner-up among the four; the
narrowest, `gaia-40041022325608704` at 113 nats, is the one to watch if the count is
raised.

**The table's chi-square is decisively wrong where it is wrong, not nearly indifferent.**
For the three misses whose truth sits in the top eight the gap from rank 1 to the truth is
+480.9 on `gaia-37608387208382848` (15 usable epochs, 32.1 per epoch), +274.2 on
`gaia-47620265212420096` (13 epochs, 21.1 per epoch) and +102.2 on `mixed-0017` (16 usable
of 22 epochs, 6.4 per epoch). `mixed-0017` is the only star in either population where the
velocity table is close to indifferent between the truth and the alias it prefers, and it
is a P = 309 d, e = 0.67 orbit measured over a 22-epoch table of which 6 epochs are
unusable. Further down, `gaia-51574864941234048` at rank 14 is +324.7 over 14 epochs, and
the two large-rank field cases are hopeless: +176052 on `mixed-0003` and +182433 on
`mixed-0014`, thousands of units per epoch. Where the truth is at rank 1 the gap to rank 2
is 529 at its smallest (`mixed-0007`) and 642 next (`gaia-40041022325608704`), so the
statistic separates the right answer as decisively as it separates the wrong one.

**Where the Gaia comparison moved the answer it never moved it to the truth.** Of the 14
Gaia stars the disentangling confirmed the table on 10, of which 7 were recovered and 3
were wrong already (`gaia-7289006178185856`, `gaia-17170287112790656` and
`gaia-53290511099783296`, all with the truth absent from the ranking at 2 percent), and
overruled it on 4, all four of which ended with a wrong period:
`gaia-37608387208382848` (0.24847 to 0.57230 d, truth 6.1037), `gaia-47620265212420096`
(4.84532 to 5.10109, truth 7.6752), `gaia-51574864941234048` (0.20508 to 0.26094, truth
0.7234) and `gaia-56716765427964800` (1.12943 to 0.40417, truth 5.3561). Only the first of
those four had the truth among the candidates it was choosing from. The comparison has so
far converted no miss into a hit in this run.

## What is left

The measurement bounds the gain and does not test it. Whether `gaia-47620265212420096`'s
truth at rank 6 would actually win a comparison of eight is a question for the marginal
likelihood, which means eight coarse scans on that one star's dataset, about 4 minutes of
work and the only cheap experiment left that could settle the count. Second, the
`mixed-0017` case is the one that makes the comparison look worth having on the field
population, and it is untested for the same reason. Third, and larger than either, 12 of
the 15 misses have the truth at rank 14 or worse or absent altogether, which is a
candidate-generation failure and not a decision failure; the D63 diagnosis reached the
same conclusion on the cleaner oracle tables, and nothing about the comparison's candidate
count addresses it. Fourth, the reproduction is limited by the six decimals of
`template_velocities.rv`: one starting period in forty differs, which moves a single entry
at the top of three stars' rankings and, on `gaia-56716765427964800`, finds an orbit at
0.67875 d and chi-square 2423.0 that the run's own list did not contain, so the run's
ranking on that star is not the best one its own method can reach. Writing the bootstrap
table before the exchange, alongside the exchanged one, would remove the un-exchange step
that this note had to invent; `velocities_unexchanged.rv` already does exactly that for
the post-fit table.

Scripts and the per-star JSON (`ranks.json`, 33 stars with their full rankings) are in the
session scratchpad under `wp-w/`.
