# What merging the candidate sources by rank buys the blind period search (D64, 2026-09-11)

`albireo.pipeline._period_candidates` proposes the starting periods the blind tier's orbit
fit chooses among, from four periodogram sources: the floating-mean generalized
Lomb-Scargle of the relative velocity, its two-harmonic form, the first component's own
velocities, and the swap-invariant search, whose first four peaks each enter with their
double. The companion note [`d64_candidate_breadth.md`](d64_candidate_breadth.md) found
that the function concatenated those four lists in that fixed order before deduplicating
them greedily at 2%, so that a deep peak of an early source could pre-empt the top peak of
a later one, and that the list was therefore not monotone in the number of peaks each
source reports: 233 of the 1296 starts proposed at `n_peaks = 20` were absent at 50 and 403
at 100. This note measures the fix. The four lists are now merged round robin by rank,
every source's highest peak, then every source's second, a source dropping out when it is
exhausted, and the same greedy 2% deduplication is applied to that sequence. Deduplicating
by periodogram power across the sources was considered and is not available, since the four
are different statistics on different series and their powers are not on a common scale.
Velocity tables and Keplerian fits only: no spectra were read, no `Disentangler` was built,
no JAX model was run, one process.

**The measurement is WP-Z's, re-run with the changed function, and the reimplementation is
identical to the changed function on all 33 tables.** Each blind directory's
`template_velocities.rv` was read back into a `VelocityTable` with WP-W's reader, the
exchange `reassign_by_orbit` had applied before the file was written was undone, and the
candidate loop was re-run with the run's own `Analysis` settings from the two
`manifest.json` files ($`k_{\min} = 2`$, $`k_{\max} = 250`$, $`e_{\max} = 0.9`$, `circular`
at its default `False`). Because `_period_candidates` takes no peak count of its own, the
harness rebuilds the four sources from one deep `find_period` call per source and merges
them at the setting under test, exactly as WP-Z did for the concatenation. The check that
this is the same function is direct: on every one of the 33 tables the harness's list at 20
peaks is element for element the list `_period_candidates(table, swap_invariant=True)`
itself returns with the change in place, 33 of 33 with no exceptions. WP-Z's own
reproduction, which agreed with WP-W on every star and every field of the unchanged code,
is what the columns labelled concatenation below rest on, and those columns are read from
its `ranks_breadth.json` rather than recomputed. Keplerian fits from a given start on a
given table are deterministic, so the fits were memoized on the starting period within a
star, which turned 3867 candidate fits into 2566.

## Per star

`rank` is the 1-based position, among the distinct valid orbits in chi-square order, of the
first entry within 2% of the injected period, and `absent` means no entry qualifies. The
columns marked Z are WP-Z's numbers for the unchanged concatenation at the same peak count.

| population | star | P_true (d) | usable | cand 20 | rank 20 | Z rank 20 | chi2 1 at 20 | Z chi2 1 at 20 | cand 50 | rank 50 | Z rank 50 | chi2 1 at 50 | Z chi2 1 at 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gaia | gaia-7289006178185856 | 0.5440 | 10 | 35 | absent | absent | 843.3 | 843.3 | 60 | 39 | absent | 843.3 | 843.3 |
| gaia | gaia-17170287112790656 | 0.5786 | 11 | 41 | absent | absent | 10375.1 | 13723.1 | 76 | 23 | 24 | 9778.8 | 9778.8 |
| gaia | gaia-51574864941234048 | 0.7234 | 14 | 40 | 14 | 14 | 182.9 | 182.9 | 71 | 20 | 17 | 182.9 | 182.9 |
| gaia | gaia-40041022325608704 | 2.8086 | 12 | 38 | 1 | 1 | 100.1 | 100.1 | 69 | 1 | 1 | 100.1 | 100.1 |
| gaia | gaia-45354171749336960 | 3.8268 | 25 | 36 | 1 | 1 | 22126.8 | 22126.8 | 74 | 1 | 1 | 22126.8 | 22126.8 |
| gaia | gaia-56716765427964800 | 5.3561 | 12 | 38 | absent | absent | 739.9 | 2423.0 | 71 | absent | absent | 739.9 | 2423.0 |
| gaia | gaia-45788547559850496 | 5.6093 | 15 | 42 | 1 | 1 | 549.7 | 549.7 | 88 | 1 | 1 | 549.7 | 549.7 |
| gaia | gaia-37608387208382848 | 6.1037 | 15 | 36 | 3 | 3 | 1546.7 | 1546.7 | 69 | 3 | 2 | 1546.7 | 1546.7 |
| gaia | gaia-56861694804053120 | 6.8142 | 14 | 40 | 1 | 1 | 46.7 | 46.7 | 81 | 1 | 1 | 46.7 | 46.7 |
| gaia | gaia-47620265212420096 | 7.6752 | 13 | 36 | 5 | 5 | 1187.5 | 1187.5 | 75 | 6 | 5 | 724.8 | 724.8 |
| gaia | gaia-50868531799454720 | 8.8642 | 16 | 38 | 1 | 1 | 18.8 | 18.8 | 73 | 1 | 1 | 18.8 | 18.8 |
| gaia | gaia-51884824140205824 | 17.7539 | 23 | 36 | 1 | 1 | 66.5 | 66.5 | 72 | 1 | 1 | 66.5 | 66.5 |
| gaia | gaia-53290511099783296 | 18.8885 | 16 | 42 | absent | absent | 2879.0 | 2879.0 | 81 | 37 | 40 | 2879.0 | 2879.0 |
| gaia | gaia-53680017390963072 | 29.6049 | 17 | 43 | 1 | 1 | 2554.1 | 2554.1 | 79 | 1 | 1 | 2554.1 | 2554.1 |
| field | mixed-0003 | 1.3226 | 53 | 35 | 1 | 24 | 67824.7 | 74255.6 | 70 | 1 | 48 | 67824.7 | 74255.6 |
| field | mixed-0008 | 2.0895 | 38 | 44 | 1 | 1 | 3569.5 | 3569.5 | 82 | 1 | 1 | 3569.5 | 3569.5 |
| field | mixed-0002 | 3.5735 | 28 | 34 | 1 | 1 | 272.4 | 272.4 | 77 | 1 | 1 | 272.4 | 272.4 |
| field | mixed-0014 | 3.9498 | 46 | 43 | 18 | 19 | 112583.9 | 112583.9 | 78 | 40 | 43 | 112583.9 | 117297.5 |
| field | mixed-0018 | 3.9864 | 71 | 35 | 1 | 1 | 1345.4 | 1345.4 | 73 | 1 | 1 | 1345.4 | 1345.4 |
| field | mixed-0000 | 5.0242 | 47 | 33 | 1 | 1 | 965.1 | 965.1 | 76 | 1 | 1 | 965.1 | 965.1 |
| field | mixed-0009 | 6.4820 | 35 | 37 | 1 | 1 | 121.5 | 121.5 | 79 | 1 | 1 | 121.5 | 121.5 |
| field | mixed-0012 | 7.3746 | 100 | 37 | 1 | 1 | 16837.6 | 16837.6 | 76 | 1 | 1 | 16837.6 | 16837.6 |
| field | mixed-0006 | 19.1633 | 53 | 43 | 1 | 1 | 3158.5 | 3158.5 | 81 | 1 | 1 | 3158.5 | 3158.5 |
| field | mixed-0016 | 23.6586 | 50 | 33 | 1 | 1 | 761.7 | 761.7 | 72 | 1 | 1 | 761.7 | 761.7 |
| field | mixed-0007 | 45.3942 | 36 | 41 | 1 | 1 | 203.1 | 203.1 | 85 | 1 | 1 | 203.1 | 203.1 |
| field | mixed-0004 | 116.8901 | 56 | 47 | 1 | 1 | 13897.8 | 13897.8 | 94 | 1 | 1 | 13897.8 | 13897.8 |
| field | mixed-0011 | 133.2333 | 24 | 42 | absent | absent | 1816.9 | 1816.9 | 84 | 1 | 1 | 880.4 | 880.4 |
| field | mixed-0013 | 221.9622 | 43 | 34 | 1 | 1 | 22186.8 | 22186.8 | 76 | 1 | 1 | 22186.8 | 22186.8 |
| field | mixed-0017 | 308.6882 | 16 | 45 | 2 | 2 | 357.8 | 357.8 | 92 | 2 | 1 | 357.8 | 460.0 |
| field | mixed-0010 | 310.0558 | 62 | 41 | absent | absent | 1477.3 | 1477.3 | 78 | absent | absent | 1260.4 | 1260.4 |
| field | mixed-0019 | 330.2300 | 9 | 43 | absent | absent | 7170.0 | 7170.0 | 84 | absent | absent | 3941.3 | 1823.4 |
| field | mixed-0015 | 552.5478 | 26 | 50 | absent | absent | 121408.2 | 121408.2 | 89 | absent | absent | 121408.2 | 121408.2 |
| field | mixed-0005 | 844.3146 | 29 | 43 | absent | absent | 668.8 | 668.8 | 81 | 24 | 27 | 668.8 | 668.8 |

**No star that the run recovered loses chi-square rank 1, at either peak count, so the
reordering costs nothing where it matters most.** All 18 stars whose truth is at rank 1
under the concatenation at `n_peaks = 20` are at rank 1 under the round robin at 20 and
again at 50, 18 of 18 with no exceptions and no margin to quote. That is the first thing
the measurement had to establish, because a reordering that recovered a displaced start
while displacing a rank-1 truth would not be worth making whatever else it bought.

**At the current peak count of twenty the reordering moves one system from rank 24 to rank
1, and it is the exact mirror of the loss WP-Z found at a hundred.** `mixed-0003` has a
$`P = 1.3226`$ d orbit over 53 usable epochs. The swap-invariant search's top peak doubled
falls at 1.32254 d, 0.002% from the truth, and the one-harmonic search's sixth peak falls
at 1.31960 d, 0.22% away. Under the concatenation the sixth peak of an early source
pre-empted the first peak of the last source, and the Keplerian started at 1.31960 d
converges to 1.31954 d at chi-square 250307, which is rank 24 of 29 at `n_peaks = 20` and
rank 48 of 57 at 50. Started at 1.32254 d it converges to 1.32271 d at chi-square 67825,
which is rank 1 outright, ahead of the alias at 0.26092 d and 74256 that the concatenation
had ranked first. The same start is the one the swap-invariant branch exists to provide,
and merging by rank is what lets it survive. Over the whole tier this single change takes
the count of truths at rank 1 from 18 to 19 of 33, the top four from 20 to 21 and the top
eight from 21 to 22, at `n_peaks = 20` and at no extra cost at all.

**Four stars rank worse at fifty peaks than the concatenation did, and on none of them is
the truth's own orbit any different.** `gaia-37608387208382848`, the star the fix was meant
to protect, holds rank 3 at 20 and rank 3 at 50, against 3 and 2 for the concatenation, and
never approaches the rank 80 the concatenation reached at a hundred peaks. Its truth is the
orbit at 6.11903 d and chi-square 2027.6 at every setting of both orderings; what changes at
50 is that the swap-invariant search's fourth peak at 0.57231 d, which the round robin keeps
in place of the two-harmonic search's 47th peak at 0.57152 d, reaches an orbit at 0.57230 d
and chi-square 1772.6 and takes second place. The displacing peak that caused the original
loss, at 6.079648 d, is peak 71 of the one-harmonic search and peak 73 of the first
component's, so under the round robin it cannot reach the truth's neighbourhood before the
swap-invariant peak at any count whatever. `gaia-47620265212420096` moves from 5 to 6 and
stays inside the top eight; `gaia-51574864941234048` moves from 17 to 20, outside it under
either ordering; and `mixed-0017` moves from 1 to 2, which is the subject of the next
paragraph. Counting only what the disentangling
comparison can see at its default of eight candidates, nothing leaves the top eight at
either peak count.

**`mixed-0017`'s promotion to rank 1 at fifty peaks was an artefact of the missing start,
and the fix removes it, exactly as WP-Z predicted it would.** Its truth is a
$`P = 308.7`$ d, $`e = 0.67`$ orbit over 16 usable epochs, fitted at chi-square 460.0 at
every setting of both orderings. The orbit that outranks it is at 2.84147 d and chi-square
357.8, started from the swap-invariant search's second peak at 2.84139 d. Under the
concatenation at 50 that peak was pre-empted by the one-harmonic search's 49th peak at
2.82046 d, the 357.8 orbit was never fitted, and the truth became rank 1 by default. Under
the round robin the second peak of the last source survives at both counts, the alias is
fitted again, and the truth is rank 2 at 20 and at 50. The same mechanism repairs
`mixed-0014`, whose rank-1 period of 0.18312 d is the two-harmonic search's fourth peak: the
concatenation lost it at 50 and its rank-1 chi-square degraded from 112584 to 117298, while
the round robin keeps it and the rank-1 chi-square is 112584 at both counts. WP-Z read both
of these as losses of a start rather than discoveries of an orbit, and neither survives the
fix.

**The proposal is now monotone in the peak count, measured and not assumed.** On all 33
stars the list proposed at `n_peaks = 20` is the head of the list proposed at 50, entry for
entry and in order, so no start present at the smaller count is absent at the larger: 0
against the concatenation's 233. The property is exact for any count at or above the eight
entries the swap-invariant source contributes, since below that a round is reached in which
only that source is still supplying peaks. The reordering is not a large change to what is
proposed. At `n_peaks = 20` the two orderings share 1148 starts and differ on 153 against
148, and 152 of those 153 lie within 2% of a start the concatenation proposed instead; at
50 they share 2038 and differ on 528 against 538, all 528 within 2%. What the reordering
changes is almost always which representative of a pair of near-duplicate periods survives,
and `mixed-0003` is the demonstration that this is not cosmetic: 1.32254 d and 1.31960 d are
0.22% apart and their Keplerians land at chi-square 67825 and 250307.

| population | ordering | n_peaks | rank 1 | ranks 2-4 | ranks 5-8 | ranks 9-20 | rank > 20 | absent | top 4 | top 8 | ranked at all | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gaia | concatenated | 20 | 7 | 1 | 1 | 1 | 0 | 4 | 8 | 9 | 10 | 14 |
| gaia | round robin | 20 | 7 | 1 | 1 | 1 | 0 | 4 | 8 | 9 | 10 | 14 |
| gaia | concatenated | 50 | 7 | 1 | 1 | 1 | 2 | 2 | 8 | 9 | 12 | 14 |
| gaia | round robin | 50 | 7 | 1 | 1 | 1 | 3 | 1 | 8 | 9 | 13 | 14 |
| field | concatenated | 20 | 11 | 1 | 0 | 1 | 1 | 5 | 12 | 12 | 14 | 19 |
| field | round robin | 20 | 12 | 1 | 0 | 1 | 0 | 5 | 13 | 13 | 14 | 19 |
| field | concatenated | 50 | 13 | 0 | 0 | 0 | 3 | 3 | 13 | 13 | 16 | 19 |
| field | round robin | 50 | 13 | 1 | 0 | 0 | 2 | 3 | 14 | 14 | 16 | 19 |
| both | concatenated | 20 | 18 | 2 | 1 | 2 | 1 | 9 | 20 | 21 | 24 | 33 |
| both | round robin | 20 | 19 | 2 | 1 | 2 | 0 | 9 | 21 | 22 | 24 | 33 |
| both | concatenated | 50 | 20 | 1 | 1 | 1 | 5 | 5 | 21 | 22 | 28 | 33 |
| both | round robin | 50 | 20 | 2 | 1 | 1 | 5 | 4 | 22 | 23 | 29 | 33 |

**The whole gain from the reordering is in the field population, and on the Gaia population
it is neutral at twenty peaks and slightly positive at fifty.** The Gaia counts of rank 1,
top four and top eight are 7, 8 and 9 at both counts and under both orderings; the only
Gaia movement is that `gaia-7289006178185856` enters the ranking at rank 39 at 50 where the
concatenation had it absent, and that two deep truths move up, `gaia-53290511099783296`
from 40 to 37 and `gaia-17170287112790656` from 24 to 23, against
`gaia-51574864941234048` moving down from 17 to 20. None of that is decision-relevant. The
field counts move from 11, 12 and 12 to 12, 13 and 13 at 20, and from 13, 13 and 13 to 13,
14 and 14 at 50. That is the expected direction for the same reason WP-Z gave: the field
tables are denser, their one-harmonic lists are long and full of cadence aliases, and the
swap-invariant source sitting last was losing its best peaks to them.

**The candidate loop costs twice as much at fifty peaks as at twenty, and the reordering
itself costs nothing.** The round robin proposes 1301 starts over the 33 stars at 20 and
2566 at 50, against the concatenation's 1296 and 2576, and keeps 1013 and 1988 distinct
orbits against 1020 and 2015. The measured fitting cost follows the start count.

| population | ordering | n_peaks | candidate starts | distinct orbits | fit seconds | seconds per star |
|---|---|---|---|---|---|---|
| gaia | round robin | 20 | 541 | 429 | 418 | 30 |
| gaia | round robin | 50 | 1039 | 831 | 814 | 58 |
| field | round robin | 20 | 760 | 584 | 753 | 40 |
| field | round robin | 50 | 1527 | 1157 | 1520 | 80 |
| both | round robin | 20 | 1301 | 1013 | 1172 | 36 |
| both | round robin | 50 | 2566 | 1988 | 2334 | 71 |

The whole two-setting run took 2406 s of wall in one process, 2566 distinct fits with the
memoization against 3867 without, and the box also carried the package's test suite for the
first six minutes of those forty. The absolute seconds are therefore no more comparable with
WP-Z's 1943 s and 3944 s for the same two passes than WP-Z's were with WP-W's 1300 s; the
ratio is the number to price with, and it is 2.0 here as it was there, on a different load
and over a different set of starts. Against the blind tier's real budget this is small: the
comparison runs one coarse disentangling scan per compared candidate, the 56 recorded scans
took 16 to 51 s each, and at a fixed `period_decision_candidates = 8` the comparison's own
cost does not change with the peak count.

**The recommendation is to raise the default peak count from 20 to 50 now that the ordering
is fixed.** With the round robin in place the trade from 20 to 50 is one system moving to
chi-square rank 1, `mixed-0011` from absent, and four more entering the ranking too deep to
matter (`gaia-7289006178185856` at 39, `gaia-53290511099783296` at 37, `mixed-0005` at 24,
`gaia-17170287112790656` at 23), against no star leaving rank 1, no star leaving the top
eight, and a doubling of the candidate loop from 36 to 71 s per star. Three truths sink
within the offered set, `mixed-0014` from 18 to 40, `gaia-51574864941234048` from 14 to 20
and `gaia-47620265212420096` from 5 to 6, and only the last is inside the top eight at
either count. This is a smaller gain than WP-Z measured on the unchanged code, which was two
systems rather than one, and the difference is not a disagreement: `mixed-0017`'s promotion
there was the loss of a start, and the fix restores the start and with it the rank-2 that
the table's chi-square honestly supports. The ordering fix itself is the better of the two
changes, since it delivers `mixed-0003` at the current count for no cost at all, but the two
are independent and both are worth making.

## What is left

The gain is still bounded rather than demonstrated. `mixed-0003` reaches chi-square rank 1
at `n_peaks = 20` and `mixed-0011` at 50, and on the field population that settles the
period, since 18 of those 19 stars recorded no `decision` block and the chi-square winner is
the answer; neither has been run end to end, and a blind re-run of the two is the check that
would turn this bound into a result. The same applies to the 18 stars already at rank 1: the
reordering leaves their rank-1 orbit untouched but changes the field the disentangling
comparison draws from, and this note has not counted how many new entrants the top eight
acquires, which is what WP-Z measured for the peak count and found to be a median of two per
star at 50. Second, `mixed-0017` is now rank 2 at both counts with an alias at 2.84 d
ahead of it by 102 in chi-square, and whether the comparison picks the truth from that
position is a question about `_decide_period_by_disentangling` rather than about the search.
Third, the reproduction inherits WP-W's limitation, the six decimals of
`template_velocities.rv`, which moves about one starting period in forty; that is exactly
the quantity a 2% deduplication is sensitive to, and while the ordering fix removes the
count-dependence it does not remove that, so a re-run against an unrounded bootstrap table
would still be worth having. Fourth, 10 of the 13 remaining misses have the truth at rank 20
or worse or absent at 50, and on two of the four stars that never reach the ranking at all
(`mixed-0010` and `mixed-0015`) a candidate within 2% of the truth is nevertheless proposed
at both counts, so the loss there is at the Keplerian fit or at the 2% merge and no ordering
or peak count addresses it.

Scripts and the per-star JSON (`ranks_ordering.json`, 33 stars with their full rankings at
both settings and both candidate lists) are in the session scratchpad under `wp-ac/`, and
read `wp-z/` and `wp-w/` unchanged.
