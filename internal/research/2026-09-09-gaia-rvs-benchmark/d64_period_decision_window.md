# Scoring each candidate over its whole declared window, not at one point (D64, 2026-09-11)

`d64_period_decision_scan.md` established that the blind tier's period comparison scores
each candidate at a single point of the two quantities the velocity table measures worst,
the period and the eccentricity, while scanning the two it measures no better, the
semi-amplitudes and the conjunction. The obvious repair is to score a candidate by the
best coarse marginal likelihood it reaches anywhere inside the window its own declaration
hands the fit, `Between(P - 0.03 P, P + 0.03 P)` at line 2285 of `pipeline.py`, rather
than at the window's centre. This note measures that repair on
`gaia-37608387208382848`: how wide the period peak really is, whether a window score
recovers the true candidate, and what the scan costs. Scripts and raw values: the session
scratchpad `wp-x2/` (shared.py, s1_verify.py, s2_fine_c3.py, s3_scan14.py, s5_ecc_c3.py,
s6_score.py, s7_joint.py, s8_analyse.py), which imports WP-X's `common.py` unchanged for
the re-simulation, the bootstrap table and the comparison declaration. About 127000 coarse
evaluations in seven short processes, one at a time, 113 min of sweeping.

**The reproduction gate is exact in both directions, and the premise this work package
started from was wrong in one respect.** Eleven of the 121 nodes of WP-X's period profile
were recomputed from a fresh declaration at the injected semi-amplitudes and eccentricity
with 41 conjunctions, and every one matched the stored value to 0.000000000 nats; the
declaration behind them is the same 1756-pixel coarse model at $`dv = 6.0`$ km/s and a
velocity budget of 997.5 km/s. All four candidates' archived comparison values were then
reproduced as the centre node of their window grids, to $`-0.000000001`$, $`0.000000000`$,
$`0.000000000`$ and $`-0.000000001`$ nats against the archive's 22260.728745911132,
22538.464468338054, 22287.698097369394 and 21980.84560868964, and candidates 1 and 4, which
WP-X had not replayed, were rerun through the whole comparison and reproduced their
semi-amplitudes as well. The premise was that WP-X had measured its 0.05 percent peak with
the conjunction held. It had not: `step7_period_profile.py` builds its conjunction trials as
`min(bjd) + linspace(0, P, 41, endpoint=False)` at each period node, which is
`Disentangler._scan_phase`'s own grid evaluated at that node's period, so the profile
already had the conjunction free. What the 0.05 percent was is the grid's own resolution:
121 nodes across a window of $`\pm 3`$ percent are spaced 0.0501 percent apart, and the
two nodes found above the winner are one spacing.

**Resolved sixteen times finer, the peak is 0.097 percent wide and its maximum comes within
3 nats of the truth.** The same profile was remeasured over $`\pm 0.25`$ percent of the
injected period at 161 nodes, a spacing of 0.00313 percent, with the 41-point conjunction
scan at every node, 6601 evaluations in 269 s. It peaks at 22720.14 nats at 6.1039165 d,
0.0031 percent from the injected 6.10372573 d, which is 3.01 nats short of the 22723.15 the
coarse model returns at the injected elements themselves. Thirty-two of the 161 nodes
exceed the winner's 22538.46, spanning 6.100483 to 6.106396 d, a width of 0.0969 percent of
the period; the span within 250 nats of the peak is 0.1219 percent and the span within 25
nats is 0.0281 percent. So a free conjunction does widen the usable peak, from the single
grid spacing WP-X could resolve to about a tenth of a percent, but not by the factor the
0.627-cycle drift argument suggested: a shifted conjunction absorbs the drift of the mean
phase, not the change in the rate at which phase accumulates across 249.9 cycles.

**The peak spacing is set by accumulated drift and scales as $`P/T`$, which is what makes
this repair unaffordable.** Counting the separate local maxima of each candidate's profile
gives a mean spacing of 0.12083 percent of the period for candidate 3 at 6.1190 d, 0.01081
for candidate 2 at 0.5723 d, 0.00455 for candidate 4 at 0.2795 d and 0.00406 for candidate 1
at 0.2485 d. Expressed as the phase drift those spacings accumulate over the 1525.3872 d
baseline they are 0.301, 0.288, 0.248 and 0.249 cycles, one number within 20 percent, so

```math
\frac{\Delta P}{P} \simeq 0.28\,\frac{P}{T},
\qquad
N_{\rm indep} \simeq \frac{0.06}{\Delta P / P} \simeq 0.21\,\frac{T}{P}
```

is the width of one peak and the count of independent period trials the declared
$`\pm 3`$ percent window holds. On this star that is 50 trials for the true candidate and
1489, 554 and 1240 for the three short-period rivals. The window that has to be searched is
not the same size for every candidate, and the candidate whose period is right is the one
with least to search.

**Scored over its own window at the sequential scan's semi-amplitudes, the true candidate
wins by 66.5 nats, and wins at a period 1.68 percent from the truth.** Each candidate was
scored as the maximum of the coarse marginal over a period axis whose centre node is the
declared period, 41 conjunctions over one period at every node, and two eccentricities, the
table's own $`(e, \omega)`$ and a circular orbit, with the semi-amplitudes held at what the
sequential scan settled on. The node spacing was fixed at 0.025 percent for candidate 3,
which puts 4.8 nodes across its 0.121 percent peak spacing, and scaled by $`P`$ for the
others, which holds the nodes per peak at 4.0 to 4.6 throughout, so no candidate's grid is
coarser than its peaks. Candidate 3 could therefore be given its whole window at 241 nodes;
the others were given 251 nodes each, which at their own spacings covers 4.23, 9.74 and 4.76
percent of theirs. The scores are 22614.61 for candidate 3, 22548.07 for candidate 2,
22260.73 for candidate 1 and 22043.97 for candidate 4, all four attained at the table's
eccentricity rather than at the circular node. That reverses the run's decision. It
reverses it, however, at 6.0012398 d: candidate 3's window maximum is a noise peak 1.68
percent below its declared period and 1.68 percent below the injected one, where the point
comparison it replaces was 0.25 percent off. At the window node nearest the truth the same
profile reads 22133.76.

**The reversal is an artifact of the rivals' windows being only a twentieth scanned, and
does not survive correction.** Candidate 3's 241 nodes hold 49 separate peaks, mean 22116.9
nats and standard deviation 200.4 over the 36 above the profile median, and its observed
maximum of 22614.61 is the extreme of all 49 the window contains. Candidate 2's 251 nodes
hold 54 peaks of the 554 its window contains, mean 21886.3 and standard deviation 201.8, and
its 22548.07 is the extreme of a tenth of them. Taking the separate peaks as independent
trials of one distribution, which their measured spacing supports, and evaluating
$`\mu + \sigma\sqrt{2\ln N}`$ at each candidate's full count puts candidate 3 first at
22653.2 against candidate 2's 22575.3, a margin of 78 nats; rescaling each candidate's
observed maximum by $`\sqrt{\ln N_{\rm full} / \ln N_{\rm sub}}`$ instead puts candidate 2
first at 22740.5 and candidate 3 second at 22614.6, the truth 126 nats behind. The two
estimators disagree in sign, which is the finding: the margin the repair would turn on is
smaller than the uncertainty in what the unscanned nine tenths of the rivals' windows
contain. A window maximum with no trials penalty rewards the candidate with the most
independent trials, and on this star the rivals have ten to thirty times as many.

**Neither a free conjunction nor a free eccentricity moves the true candidate while the
semi-amplitudes stay where the point scan left them.** At the sequential scan's 112.47 and
112.41 km/s with the table's eccentricity, the $`\pm 0.25`$ percent fine profile peaks at
22287.46 at its top node 6.118985 d, which is the declared period to 0.0008 percent, and not
one of its 161 nodes reaches the winner; circular, it peaks at 21963.63, lower still. A grid
of twelve eccentricities from 0 to 0.80 crossed with four arguments of periastron a quarter
turn apart, each at its own best of 41 conjunctions, reaches only 22167.99 at the injected
period and 22287.70 at the declared one, the latter being the declared start itself and the
archived value to the digit. With the wrong semi-amplitudes the marginal prefers an
eccentric curve, best at $`e \simeq 0.5`$, because an orbit whose semi-amplitude is 64
percent too large spends less of its phase far from the systemic velocity when it is
eccentric; with the injected semi-amplitudes the same grid prefers $`e = 0`$ at 22724.96
against 22137.04 at the table's start, a preference of 588 nats the other way. The
eccentricity axis cannot be scanned usefully at the wrong semi-amplitude, and the
semi-amplitude scan that would fix it ran at the wrong period. That is the same
conditioning failure WP-X found, now shown to close on itself: no two of the four axes are
enough.

**With the semi-amplitudes inside the period loop the profile does peak at the injected
period, and that is the minimum sufficient version.** The shared semi-amplitude axis
($`K_A = K_B`$ over the declaration's own 17-node geometric grid, the level-0 axis of
`_scan_semi_amplitudes`) was crossed with the 41-point conjunction scan at 25 period nodes
of 0.025 percent spacing about the declared period, at a circular orbit: 17425 evaluations
in 899 s. The profile peaks at 22704.80 nats at 6.1037336 d, 0.00013 percent from the
injected period, at 64.454 km/s, with three of the 25 nodes above the winner spanning
0.0501 percent; at the candidate's declared period it reaches 22241.57. A shared axis is a
seventeen-fold reduction of the pipeline's own 17 by 17 product and is favourable to this
star, whose injected semi-amplitudes differ by 0.24 percent, so this is the cheapest form
the sufficient version can take. It is not the affordable one. A free conjunction alone
does nothing; a free conjunction with a free eccentricity does nothing; only the period,
the eccentricity and the semi-amplitudes together, which is to say the whole scan repeated
at every period node, recovers the star.

**The price is between 20 and 125 hours a star against the 116 s the comparison now
spends.** The archived comparison took 27.3, 31.1, 31.0 and 26.4 s for its four candidates.
The four window scans measured here took 77.0 min for 81508 evaluations at 56.7 ms each on
a machine shared with other jobs, and they covered candidate 3's window once and the other
three between a twentieth and a tenth. Extending each to its whole window at the same
spacing needs 241, 2576, 5934 and 5275 period nodes, 1.15 million evaluations, and 20.6 h
for four candidates or 41 h at the current default of eight. Putting the semi-amplitude
scan inside the loop multiplies the 82 evaluations a node costs now by the 283 the
pipeline's own conjunction and semi-amplitude scans use, which is 62.5 h for four
candidates and 125 h for eight; the shared-K form measured above, at 697 evaluations and
36.0 s a node, would be 140 h. Every one of these is three to four orders of magnitude
above the present cost, and the scaling is $`T/P`$, so the shorter the alias the worse it
gets: candidate 1 alone accounts for 26.4 h of the 62.5.

**Verdict: the window scan is not the repair, and the reason is structural rather than
budgetary.** Scoring by the best of a window rather than by one point is not a
like-for-like comparison when the windows hold 50 independent trials for one candidate and
1489 for another, and no amount of compute fixes that; a trials penalty or a marginal
likelihood integrated over the window would, but neither is what was proposed here. What
the measurement does establish is that the true candidate's basin is reachable and sharp,
0.097 percent wide and worth 22720 nats against the winner's 22538, and that reaching it
needs the period, the eccentricity and the semi-amplitudes moved together. Left for a
later work package: whether the same margin can be had at the velocity-table stage instead,
where a period refinement costs milliseconds rather than the 36 s a coarse period node
costs here, so that the comparison is handed a period good to 0.05 percent and can go on
scoring points; whether `_element_starts` should stop accepting a table eccentricity that
passes a three-sigma test while being wrong by a factor 24, which on this star is worth 588
nats once the semi-amplitude is right, and whether `_ecc_omega_specs` should go on fixing
$`\omega`$ at the table's value whenever it does; and whether a trials-penalised or
integrated score would order these four candidates correctly at a cost the comparison can
carry. The other six blind-tier misses of this run never had the true period among their
candidates and remain a different question.
