# What more periodogram peaks buy the blind period search (D64, 2026-09-11)

`albireo.pipeline._period_candidates` proposes the starting periods the blind tier's
orbit fit chooses among. Three of its four sources report the twenty highest distinct
peaks of a periodogram (`albireo.rvorbit.find_period` with its default `n_peaks = 20`):
the floating-mean generalized Lomb-Scargle of the relative velocity, its two-harmonic
form, and the same one-harmonic search on the first component alone. The fourth, the
swap-invariant search, contributes only its top four peaks and their doubles and does
not grow with `n_peaks` at all. The companion note
[`d64_period_candidate_ranks.md`](d64_period_candidate_ranks.md) recovered the full
chi-square ranking of all 33 blind stars of the third run and found the truth at rank 1
on 18 of them, at ranks 2 to 20 on five, above 20 on one, and absent from the ranking on
nine, which makes candidate generation and not the comparison the binding constraint.
This note measures what raising `n_peaks` to 50 and to 100 does to that ranking, and what
it costs. Velocity tables and Keplerian fits only: no spectra were read, no
`Disentangler` was built, no JAX model was run, one process, nothing under `src/` was
modified.

**The measurement is WP-W's, re-run with one argument changed, and it reproduces WP-W
exactly at `n_peaks = 20`.** Each blind directory's `template_velocities.rv` was read
back into a `VelocityTable`, the exchange `reassign_by_orbit` had applied before the file
was written was undone from the light column, and `_period_candidates` and
`_orbit_over_candidates` were re-run with the run's own `Analysis` settings from the two
`manifest.json` files ($`k_{\min} = 2`$, $`k_{\max} = 250`$, $`e_{\max} = 0.9`$, `circular`
at its default `False`). Both steps use WP-W's reader and its un-exchange unchanged. Two
properties of the library make the three settings cheap to compare and the
reimplementation exact. `_distinct_peaks` walks the local maxima in descending power and
accepts one when it lies more than 2% in period from every peak already accepted,
stopping at `n_peaks`, so the list it returns for `n_peaks = 20` is the first twenty
entries of the list it returns for `n_peaks = 100`: one periodogram per source serves
every setting. And a Keplerian fitted from a given start on a given table is
deterministic, so the fits were memoized on the starting period within a star, which
turned 8193 candidate fits into 4988. Each setting is still priced by the sum of its own
starts' fit times rather than by the shared wall. At `n_peaks = 20` the result agrees with
WP-W's `ranks.json` on every star and every field compared: the number of candidates
proposed, the number fitted, the number of distinct orbits, the rank of the truth at the
2% and the 5% tolerance, and the period and chi-square of rank 1 to nine significant
figures, 33 of 33 with no exceptions. The totals likewise reproduce: 1296 candidate
starts, 1020 distinct orbits.

## Per star

`rank` is the 1-based position, among the distinct valid orbits in chi-square order, of
the first entry within 2% of the injected period, and `absent` means no entry qualifies.
`chi2 1` is the chi-square of rank 1. The 18 stars with rank 1 at `n_peaks = 20` are
exactly the 18 the run recovered.

| population | star | P_true (d) | usable | cand 20 | orbits 20 | rank 20 | chi2 1 at 20 | cand 50 | orbits 50 | rank 50 | chi2 1 at 50 | cand 100 | orbits 100 | rank 100 | chi2 1 at 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gaia | gaia-7289006178185856 | 0.5440 | 10 | 33 | 24 | absent | 843.3 | 59 | 48 | absent | 843.3 | 106 | 77 | absent | 843.3 |
| gaia | gaia-17170287112790656 | 0.5786 | 11 | 42 | 23 | absent | 13723.1 | 76 | 47 | 24 | 9778.8 | 121 | 83 | 39 | 7096.2 |
| gaia | gaia-51574864941234048 | 0.7234 | 14 | 40 | 32 | 14 | 182.9 | 70 | 59 | 17 | 182.9 | 117 | 98 | 19 | 182.9 |
| gaia | gaia-40041022325608704 | 2.8086 | 12 | 37 | 30 | 1 | 100.1 | 69 | 55 | 1 | 100.1 | 123 | 100 | 1 | 100.1 |
| gaia | gaia-45354171749336960 | 3.8268 | 25 | 36 | 30 | 1 | 22126.8 | 73 | 59 | 1 | 22126.8 | 130 | 102 | 1 | 22126.8 |
| gaia | gaia-56716765427964800 | 5.3561 | 12 | 37 | 32 | absent | 2423.0 | 71 | 59 | absent | 2423.0 | 120 | 103 | absent | 777.5 |
| gaia | gaia-45788547559850496 | 5.6093 | 15 | 42 | 36 | 1 | 549.7 | 89 | 80 | 1 | 549.7 | 144 | 134 | 1 | 549.7 |
| gaia | gaia-37608387208382848 | 6.1037 | 15 | 35 | 29 | 3 | 1546.7 | 69 | 58 | 2 | 1546.7 | 113 | 97 | 80 | 1546.7 |
| gaia | gaia-56861694804053120 | 6.8142 | 14 | 41 | 38 | 1 | 46.7 | 80 | 75 | 1 | 46.7 | 135 | 127 | 1 | 46.7 |
| gaia | gaia-47620265212420096 | 7.6752 | 13 | 36 | 31 | 5 | 1187.5 | 76 | 68 | 5 | 724.8 | 135 | 121 | 6 | 724.8 |
| gaia | gaia-50868531799454720 | 8.8642 | 16 | 38 | 26 | 1 | 18.8 | 73 | 49 | 1 | 18.8 | 125 | 75 | 1 | 18.8 |
| gaia | gaia-51884824140205824 | 17.7539 | 23 | 36 | 30 | 1 | 66.5 | 72 | 61 | 1 | 66.5 | 138 | 119 | 1 | 66.5 |
| gaia | gaia-53290511099783296 | 18.8885 | 16 | 42 | 41 | absent | 2879.0 | 79 | 72 | 40 | 2879.0 | 139 | 123 | 52 | 2879.0 |
| gaia | gaia-53680017390963072 | 29.6049 | 17 | 43 | 30 | 1 | 2554.1 | 79 | 53 | 1 | 2554.1 | 139 | 77 | 1 | 2554.1 |
| field | mixed-0003 | 1.3226 | 53 | 36 | 29 | 24 | 74255.6 | 73 | 57 | 48 | 74255.6 | 119 | 96 | 75 | 74255.6 |
| field | mixed-0008 | 2.0895 | 38 | 43 | 39 | 1 | 3569.5 | 80 | 71 | 1 | 3569.5 | 134 | 119 | 1 | 3569.5 |
| field | mixed-0002 | 3.5735 | 28 | 34 | 29 | 1 | 272.4 | 78 | 62 | 1 | 272.4 | 132 | 105 | 1 | 272.4 |
| field | mixed-0014 | 3.9498 | 46 | 43 | 27 | 19 | 112583.9 | 77 | 54 | 43 | 117297.5 | 123 | 94 | 71 | 72147.0 |
| field | mixed-0018 | 3.9864 | 71 | 35 | 29 | 1 | 1345.4 | 75 | 63 | 1 | 1345.4 | 136 | 105 | 1 | 1345.4 |
| field | mixed-0000 | 5.0242 | 47 | 33 | 27 | 1 | 965.1 | 76 | 57 | 1 | 965.1 | 132 | 102 | 1 | 965.1 |
| field | mixed-0009 | 6.4820 | 35 | 36 | 30 | 1 | 121.5 | 80 | 61 | 1 | 121.5 | 134 | 102 | 1 | 121.5 |
| field | mixed-0012 | 7.3746 | 100 | 37 | 32 | 1 | 16837.6 | 75 | 62 | 1 | 16837.6 | 120 | 98 | 1 | 16837.6 |
| field | mixed-0006 | 19.1633 | 53 | 43 | 33 | 1 | 3158.5 | 83 | 60 | 1 | 3158.5 | 133 | 92 | 1 | 3158.5 |
| field | mixed-0016 | 23.6586 | 50 | 33 | 27 | 1 | 761.7 | 74 | 56 | 1 | 761.7 | 128 | 89 | 1 | 761.7 |
| field | mixed-0007 | 45.3942 | 36 | 41 | 40 | 1 | 203.1 | 86 | 82 | 1 | 203.1 | 138 | 126 | 1 | 203.1 |
| field | mixed-0004 | 116.8901 | 56 | 47 | 42 | 1 | 13897.8 | 93 | 81 | 1 | 13897.8 | 140 | 116 | 1 | 13897.8 |
| field | mixed-0011 | 133.2333 | 24 | 42 | 35 | absent | 1816.9 | 84 | 68 | 1 | 880.4 | 135 | 109 | 1 | 687.9 |
| field | mixed-0013 | 221.9622 | 43 | 34 | 32 | 1 | 22186.8 | 76 | 71 | 1 | 22186.8 | 144 | 119 | 1 | 22186.8 |
| field | mixed-0017 | 308.6882 | 16 | 45 | 43 | 2 | 357.8 | 93 | 86 | 1 | 460.0 | 144 | 126 | 1 | 460.0 |
| field | mixed-0010 | 310.0558 | 62 | 41 | 26 | absent | 1477.3 | 79 | 49 | absent | 1260.4 | 125 | 73 | absent | 1260.4 |
| field | mixed-0019 | 330.2300 | 9 | 42 | 11 | absent | 7170.0 | 84 | 20 | absent | 1823.4 | 147 | 37 | absent | 1823.4 |
| field | mixed-0015 | 552.5478 | 26 | 49 | 22 | absent | 121408.2 | 91 | 40 | absent | 121408.2 | 142 | 52 | absent | 121408.2 |
| field | mixed-0005 | 844.3146 | 29 | 44 | 35 | absent | 668.8 | 84 | 72 | 27 | 668.8 | 130 | 114 | 42 | 668.8 |

**Raising `n_peaks` buys one system out of the nine on which the truth was absent, and it
is bought at 50 rather than at 100.** Of those nine, four gain an entry within 2% of the
truth somewhere in the ranking at `n_peaks = 50`: `gaia-17170287112790656` at rank 24,
`gaia-53290511099783296` at rank 40, `mixed-0005` at rank 27 and `mixed-0011` at rank 1.
Only the last is a gain in any operational sense, since the comparison measures the top
`period_decision_candidates` orbits and that default is 8: one of nine appears within the
top eight, and the same one within the top four. Going on to `n_peaks = 100` adds no
further star and moves the three deep ones further down, to ranks 39, 52 and 42. There is
a second gain of a different shape that the absent-to-present count does not capture.
`mixed-0017`, a $`P = 309`$ d, $`e = 0.67`$ orbit whose truth sat at rank 2 at
`n_peaks = 20`, is at rank 1 at both 50 and 100, so the velocity table's own chi-square
now picks it and no comparison is needed; that matters because the field run recorded no
`decision` block on 18 of its 19 stars and so never ran one. The honest reading of that
promotion is given below: it happens because the alias that had been rank 1 loses its
start, not because a better orbit at the truth was found.

| population | n_peaks | rank 1 | ranks 2-4 | ranks 5-8 | ranks 9-20 | rank > 20 | absent | top 4 | top 8 | ranked at all | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gaia | 20 | 7 | 1 | 1 | 1 | 0 | 4 | 8 | 9 | 10 | 14 |
| gaia | 50 | 7 | 1 | 1 | 1 | 2 | 2 | 8 | 9 | 12 | 14 |
| gaia | 100 | 7 | 0 | 1 | 1 | 3 | 2 | 7 | 8 | 12 | 14 |
| field | 20 | 11 | 1 | 0 | 1 | 1 | 5 | 12 | 12 | 14 | 19 |
| field | 50 | 13 | 0 | 0 | 0 | 3 | 3 | 13 | 13 | 16 | 19 |
| field | 100 | 13 | 0 | 0 | 0 | 3 | 3 | 13 | 13 | 16 | 19 |
| both | 20 | 18 | 2 | 1 | 2 | 1 | 9 | 20 | 21 | 24 | 33 |
| both | 50 | 20 | 1 | 1 | 1 | 5 | 5 | 21 | 22 | 28 | 33 |
| both | 100 | 20 | 0 | 1 | 1 | 6 | 5 | 20 | 21 | 28 | 33 |

**The whole gain is in the field population, and the Gaia population gains nothing
decision-relevant at either setting.** The Gaia counts of rank 1, top four and top eight
are 7, 8 and 9 at `n_peaks = 20`, unchanged at 50, and 7, 7 and 8 at 100, so on that
population the change is neutral at 50 and negative at 100. The field counts move from
11, 12 and 12 to 13, 13 and 13 at 50 and stay there at 100. Two stars account for the
whole difference, `mixed-0011` and `mixed-0017`, both long-period and eccentric
($`P = 133`$ d at $`e = 0.39`$ over 24 usable epochs, and $`P = 309`$ d at $`e = 0.67`$
over 16 of 22). That is the expected direction: the field tables are denser, so the
periodogram of a long-period eccentric orbit puts real power at the truth but buries it
under aliases of the observing cadence, and going deeper into the peak list reaches it.
The Gaia tables have 10 to 25 epochs in about eleven visibility windows, and where their
truth is not near the top of a peak list it is not in the list at all.

**No star that the run recovered loses rank 1, at either setting, so the change breaks
nothing.** All 18 stars whose truth is at chi-square rank 1 at `n_peaks = 20` are still at
rank 1 at 50 and at 100. There is no displacement to quote a margin for. The separation
is also essentially untouched: the smallest gap from rank 1 to rank 2 among those 18 is
`mixed-0007` at 529 in chi-square and it is identical at all three settings, and the
second smallest, `gaia-40041022325608704`, goes 641, 641 and 591. Counting from
`n_peaks = 20` to `n_peaks = 100`, nine of the 18 do not move that gap at all, two widen
it (`mixed-0016` from 24650 to 31107 and `gaia-53680017390963072` from 1440 to 2787) and
seven narrow it by between 3.5 and 32 percent, the largest narrowing being `mixed-0008`
from 4658 to 3174 and the next `gaia-45354171749336960` from 39795 to 31558. The real
cost is not at rank 1
but in the density of the field the comparison draws from. Among those 18 stars the top
eight acquires 41 new entrants in total at `n_peaks = 50`, a median of 2 per star and a
range of 1 to 4, and 57 at 100, a median of 3 and a range of 1 to 5. Each is one more
coarse disentangling scan and one more chance for the marginal likelihood to prefer an
alias, which is the risk the docstring of `_period_candidates` already names. On the
table's own statistic they are not close rivals: the nearest new entrant to rank 1
anywhere is 662 above it at `n_peaks = 50` (`mixed-0007`) and 591 at 100
(`gaia-40041022325608704`). The comparison does not use the table's statistic, so that
bounds the risk without eliminating it.

**The one real loss is at `n_peaks = 100`, and it falls on a star the comparison could
otherwise have rescued.** `gaia-37608387208382848` has its truth at rank 3 at
`n_peaks = 20`, improves to rank 2 at 50, and collapses to rank 80 at 100, leaving the
offered set entirely. It is one of only two stars in the whole tier on which the
disentangling comparison ever had the truth in front of it. Three further stars whose
truth was already deep sink deeper, `mixed-0003` from rank 24 to 48 to 75, `mixed-0014`
from 19 to 43 to 71 and `gaia-51574864941234048` from 14 to 17 to 19, and
`gaia-47620265212420096` slips from rank 5 to 6. Counting only what the comparison can
see, the net is one system into the top four and one into the top eight at `n_peaks = 50`
with nothing lost, and one in and one out at `n_peaks = 100`, a net of zero at more than
three times the cost.

**The candidate list is not monotone in `n_peaks`, and that is the mechanism behind both
the loss at 100 and one of the gains.** `_period_candidates` concatenates its four
sources in a fixed order, the one-harmonic search first, then the two-harmonic, then the
first component alone, then the swap-invariant peaks with their doubles, and then walks
the concatenation once keeping a period only when it differs by more than 2% from every
period already kept. A deeper peak of an earlier source can therefore pre-empt a peak of
a later source that the shallower setting had kept, and the source that suffers is the
swap-invariant one, which sits last and does not grow. Measured over the 33 stars, 233 of
the 1296 starts present at `n_peaks = 20` are absent at 50, and 403 are absent at 100; 434
of the 2576 present at 50 are absent at 100. On `gaia-37608387208382848` the consequence is
exact and visible. The start at 6.103977 d, contributed by the swap-invariant search's
doubled top peak and within 0.03% of the true 6.1037 d, is present at `n_peaks = 20` and
50 and gone at 100, deduplicated away by a one-harmonic peak at 6.079648 d that appears
earlier in the concatenation and is 0.4% from it. The Keplerian started at 6.103977 d
reaches chi-square 2027.6; the one started at 6.079648 d reaches 10071.9 and ranks
eightieth. The same mechanism works the other way twice. `mixed-0014` is the only star
whose rank-1 period at `n_peaks = 20` vanishes from the ranking altogether, 0.18312 d
having no survivor nearer than 0.1972 d at 50 and at 100, and its rank-1 chi-square
consequently gets worse at 50, from 112584 to 117298. `mixed-0017` loses its rank-1 orbit
the same way: the start at 2.841394 d is gone at 50, deduplicated away in favour of
2.742112, 2.820458, 2.929259 and 2.999527 d, and the orbit at 2.84147 d and chi-square
357.8 is never fitted again, the best survivor in that neighbourhood being 2.74211 d at
590.8. That is why its truth, at chi-square 460.0 and unchanged between the settings,
becomes rank 1 at 50. Two of the settings-to-settings rank-1 changes are therefore losses
of a start rather than discoveries of an orbit, and that is an argument for fixing the
ordering independently of the peak count.

**Proposing a candidate near the truth and ranking an orbit near the truth are not the
same event, and the gap between them is now the larger of the two failures.** A candidate
within 2% of the injected period is proposed on 26 of 33 stars at `n_peaks = 20`, on 30 at
50 and on 30 at 100, while an orbit within 2% is ranked on 24, 28 and 28. The two counts
move together but the membership does not. `mixed-0010` and `mixed-0015` have a candidate
within 2% of the truth at every setting and never produce a ranked orbit there, and
`gaia-7289006178185856` joins them from `n_peaks = 50`, where a peak at 0.539183 d enters
the list, 0.9% from the true 0.5440 d, and the Keplerian started there converges to
0.55556 d, 2.1% away and just outside the tolerance.
Conversely `mixed-0005` never has a candidate within 2% of its 844.3 d truth at any
setting, yet the truth is ranked 27th at 50 and 42nd at 100, the fit having reached it
from a start further out. On the nine stars absent at `n_peaks = 20`, then, three are at
`n_peaks = 50` failures of the fit or of the 2% merge rather than of the peak count, and
no value of `n_peaks` addresses those three.

**On five stars no setting reaches the truth, and on three of those the periodogram has
no useful power there at all.** The relevant comparison is the power of the highest local
maximum within 2% of the truth against the power of peak 1 of the same periodogram.

| star | P_true (d) | nearest fitted period at 100 | offset | GLS power at truth / at peak 1 | two-harmonic power at truth / at peak 1 | peak rank of the truth |
|---|---|---|---|---|---|---|
| gaia-7289006178185856 | 0.5440 | 0.5556 | +2.1% | 0.858 / 0.976 | 0.976 / 0.998 | 49 single, 37 harmonic |
| gaia-56716765427964800 | 5.3561 | 5.7386 | +7.1% | 0.681 / 0.896 | 0.810 / 0.976 | 108 single, 113 harmonic |
| mixed-0010 | 310.0558 | 529.2012 | +70.7% | no local maximum within 2% / 0.292 | none / 0.347 | none |
| mixed-0015 | 552.5478 | 494.7747 | -10.5% | no local maximum within 2% / 0.530 | none / 0.848 | none |
| mixed-0019 | 330.2300 | 23.8813 | -92.8% | 0.208 / 0.982 | 0.275 / 0.999 | 205 single, 271 harmonic |

On `mixed-0010` and `mixed-0015` neither one-harmonic search has a local maximum within 2%
of the truth anywhere on its grid, so no peak count whatever can propose it from that
source; both are nevertheless proposed by the swap-invariant branch, whose top peak
happens to fall there, and both then fail at the fit. On `mixed-0019`, a nine-epoch table
whose 147 starts collapse to 37 distinct orbits, the truth is peak 205 of the
one-harmonic search at power 0.21 against 0.98 at peak 1 and peak 271 of the two-harmonic
at 0.28 against 0.999, which is a factor of five in power and out of reach of any
practicable list. `gaia-56716765427964800` is peak 108 at 0.68 against 0.90.
`gaia-7289006178185856` is the near miss, peak 37 of the two-harmonic search at 0.976
against 0.998, inside a list of 50 and still lost at the fit. For these five the work
belongs on the velocity table and not on the search.

**The candidate loop costs twice as much at 50 and three and a half times as much at 100,
and the comparison's own cost does not change.** The number of candidates proposed grows
sublinearly, because the three growing sources contribute at most $`3n`$ raw peaks
against the swap-invariant source's eight, and because the 2% deduplication removes more
of them as the lists lengthen:

```math
N_{\rm cand}(n) = f(n)\,(3n + 8), \qquad f(20) = 0.58,\quad f(50) = 0.49,\quad f(100) = 0.43 .
```

All four sources were available on all 33 stars and all returned a full hundred peaks, so
the relation is not truncated anywhere. The measured cost follows.

| population | n_peaks | candidate starts | distinct orbits | fit seconds | seconds per star |
|---|---|---|---|---|---|
| gaia | 20 | 538 | 432 | 555 | 40 |
| gaia | 50 | 1035 | 843 | 1054 | 75 |
| gaia | 100 | 1785 | 1436 | 1895 | 135 |
| field | 20 | 758 | 588 | 1388 | 73 |
| field | 50 | 1541 | 1172 | 2890 | 152 |
| field | 100 | 2536 | 1874 | 4812 | 253 |
| both | 20 | 1296 | 1020 | 1943 | 59 |
| both | 50 | 2576 | 2015 | 3944 | 120 |
| both | 100 | 4321 | 3310 | 6707 | 203 |

The median star proposes 40 candidates and keeps 30 distinct orbits at `n_peaks = 20`, 77
and 60 at 50, and 133 and 102 at 100. The field tables cost about twice as much per
candidate as the Gaia ones, 1.9 s against 1.1 s, because they carry 9 to 100 usable epochs
against 10 to 25. The absolute seconds here are not comparable with WP-W's 1300 s for the
`n_peaks = 20` pass: that pass is reproduced entry for entry but this machine was
carrying other work and took 1943 s of fitting for the same 1296 starts, a factor of 1.5.
The ratios between settings are measured on one machine within one process and are the
numbers to price with: 2.0 and 3.5. The whole three-setting run took 7752 s of wall in one
process, 4988 distinct fits with the memoization against 8193 without. Against the
pipeline's real budget this is small, because the comparison runs one coarse disentangling
scan per compared candidate and the 56 recorded scans took 16 to 51 s each: at a fixed
`period_decision_candidates = 8` the comparison's cost is unchanged by `n_peaks`, and the
candidate loop's extra 60 s per star at 50 is about two scans.

**The recommendation is to raise the default to 50 and not further, and to fix the
ordering of `_period_candidates` before or alongside it.** At 50 the measured trade is two
systems of 33 moving to chi-square rank 1, `mixed-0011` from absent and `mixed-0017` from
rank 2, against no star leaving rank 1, no star leaving the top eight, and a doubling of
the candidate loop's cost, which is a small part of the blind tier's budget. At 100 the
trade is the same two gains against the loss of `gaia-37608387208382848` from the offered
set, at 3.5 times the cost, and that is not worth making. The ordering defect is
independent of the count and arguably the more valuable fix: 403 of 1296 starts present at
`n_peaks = 20` are absent at 100 purely because the greedy 2% deduplication runs over a
fixed concatenation in which the swap-invariant source comes last, and on
`gaia-37608387208382848` the displaced start was the one within 0.03% of the truth.
Proposing the swap-invariant peaks first, or deduplicating by power across sources rather
than by position in the concatenation, would make the candidate list monotone in
`n_peaks` and would remove the only loss this measurement found.

## What is left

The gain is bounded, not demonstrated. `mixed-0011` reaches chi-square rank 1 at
`n_peaks = 50`, which on the field population settles it, since 18 of those 19 stars never
ran a comparison and the chi-square winner is the answer; `mixed-0017` is the same case.
Neither has been run end to end, and a full blind re-run of the two field stars at
`n_peaks = 50` is the check that would convert this bound into a result. Second, the
ordering fix proposed above is untested: whether restoring the 403 displaced starts
changes any rank-1 outcome is one more pass of this same harness and about 20 minutes of
work. Third, and larger than either, 11 of the 15 misses still have the truth at rank 17 or
worse or absent at `n_peaks = 50`, and on three of the five stars that never reach the
ranking at any setting a candidate within 2% of the truth is nevertheless proposed
(`gaia-7289006178185856` from 50, `mixed-0010` and `mixed-0015` at every setting), so the
loss there is at the Keplerian fit or at the 2% merge rather than at the search. That is a
different failure from the one this note tested and no peak count addresses it. Fourth, the reproduction inherits WP-W's limitation, the six
decimals of `template_velocities.rv`, which moves one starting period in forty; that
affects which of two near-degenerate peaks survives deduplication and is exactly the
quantity the ordering defect above is sensitive to, so a re-run against an
unrounded bootstrap table would be worth having before the ordering is changed.

Scripts and the per-star JSON (`peaks.json`, the four periodogram sources and their peak
lists; `ranks_breadth.json`, 33 stars with their full rankings at all three settings) are
in the session scratchpad under `wp-z/`, and read WP-W's `wp-w/` unchanged.
