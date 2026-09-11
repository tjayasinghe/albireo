# Why the period comparison lost the true period on gaia-37608387208382848 (2026-09-11)

The blind tier's period comparison (`_decide_period_by_disentangling`) ranked the true
period third of four on this system, 251 nats behind a candidate at 0.5723 d, although the
true period was among the four compared. The suspicion under test was that the sequential
semi-amplitude scan had measured the true period at the wrong semi-amplitude, the scan
having settled at 112.5 and 112.4 km/s against an injected 68.5 and 68.7 while its rival
settled at 72.06 and 72.06, exactly two nodes lower on the same geometric axis. The
comparison was therefore rebuilt and the full product grid mapped at both periods, on the
coarse scan model and on the full declaration. Scripts and raw values: the session
scratchpad `wp-x/` (common.py, step1_reproduce.py, step2_compare.py, step3_grid.py,
step4_decompose.py, step5_resolution.py, step6_fit.py, step7_period_profile.py).

**The reproduction is exact, and it required one correction for a change made after the
run.** Re-simulating index 2 of `population.json` at seed 2 (the index among the fourteen
systems surviving `min_transits`, which here is also the file's index) reproduces every
entry of the manifest record to every digit: snr_epoch 15.958237467864212, 15 epochs, lag1
0.2702357676791195, pixel_ratio 0.9795918367346935, grid_n 4374, baseline 1525.3871785514057
d. The bootstrap stage then reproduces the measured light fractions 0.537063777123503 and
0.4629362228764971, the 38 fitted candidates, and the four compared candidates' periods and
table chi-squares to their printed precision. Replaying the comparison as the pipeline runs
it gave 22526.07 for candidate 2 against an archived 22538.46, with 284 trials against the
archive's 283. The one extra trial located the discrepancy: `Disentangler._scan_phase` now
uses 42 conjunctions, where the archived run used 41, the change being the D63 fix recorded
in `d63_init_failure.md` and made after this benchmark was written (facade.py carries a
modification time of 08:37 against the archive's 03:25). Repeating the replay with the
41-point grid reproduces both archived values bit for bit: 22538.464468338054 at K 72.06205
and 72.06205 km/s and $`t_{\rm conj}`$ 2456972.6122767 for candidate 2, and
22287.698097369394 at K 112.46958 and 112.40594 km/s and $`t_{\rm conj}`$ 2456977.5104907 for
candidate 3, both at 283 trials, each differing from the archive by 0.000000 nats. The
42-point grid does not change the outcome: candidate 3 then reaches 22244.97 and the margin
widens from 251 to 281 nats.

**The sequential scan left nothing on the product grid: at both periods its answer beats
the best of the full 17 by 17 by 8 sweep.** The shared axis is 17 nodes from 6.96 to 245.04
km/s at ratio 1.249293, crossed with itself and with the eight conjunctions
$`t_{\rm conj} + i P / 8`$, for 2312 evaluations per period, swept in 90 to 108 s at 39 to
47 ms per point against a projection of 97 s from a timed sample of sixteen. At 0.5723043 d
the grid peaks at 22523.39 at node 10 for both components (64.454 km/s) and the first
conjunction, 15.07 nats below the 22538.46 the sequential scan reached; at 6.1190312 d it
peaks at 22268.06 at nodes 12 and 13 (100.596 and 125.674 km/s), 19.64 nats below the
22287.70 the scan reached. Both shortfalls are the scan's two refinement levels, which
leave the lattice: 72.06205 is node 10 times $`\sqrt{1.249293}`$ and 112.46958 is node 12
times the same factor, so the factor 1.5608 between them is indeed exactly two nodes, but
it is a consequence of each scan having refined the node its own period preferred, not
evidence that the axis failed.

**At the truth the coarse model is worth 22723.15 nats, 184.68 above the winning candidate,
and the two declarations agree on that number bit for bit.** Evaluated at the injected
period 6.10372573315804 d, semi-amplitudes 68.53274 and 68.69990 km/s, eccentricity
0.02538927 with $`\omega = 5.479236`$ rad and the conjunction folded to 2456976.6538161, the
coarse marginal is 22723.1471 under candidate 2's declaration and 22723.1471 under candidate
3's. That equality is the check that the comparison's values are commensurable: the two
declarations differ only in their priors and starting values, and the compiled problem
behind them is the same 1756-pixel coarse model at a velocity budget of 997.5 km/s. So the
decision is recoverable in principle. It is not recoverable by the semi-amplitude scan.

**The comparison holds two quantities fixed that both have to be right at once, and neither
is.** Each candidate is measured at one period, the table's fitted central value, and at one
eccentricity, the table's fitted $`e`$, which `_element_starts` accepts whenever it exceeds
three times its own error. On this star the table gave 6.119031 +- 0.000365 d and
$`e = 0.5968 \pm 0.0797`$ against an injected 6.103726 d and 0.025389, so the period is 42
of its own sigmas off and the eccentricity passes the three-sigma guard while being wrong by
a factor 24. The four combinations were mapped on candidate 3's declaration, each following
the pipeline's own order of a 41-point conjunction scan at the declared semi-amplitude start
followed by the product grid over eight conjunctions.

| period | eccentricity | phase-scan max | product-grid best | at the true K, best of 41 conjunctions |
|---|---|---|---|---|
| 6.119031 | 0.5968 | 22160.91 | 22268.06 at (100.60, 125.67) | 22067.03 |
| 6.119031 | 0.02539 | 21124.84 | 22261.35 at (80.52, 80.52) | 22195.85 |
| 6.103726 | 0.5968 | 22126.66 | 22162.94 at (125.67, 125.67) | 22137.04 |
| 6.103726 | 0.02539 | 20769.70 | 22692.38 at (64.45, 64.45) | 22717.02 |

Only the last row clears the winner's 22538.46, and it clears it by 153.92 nats on the
lattice alone, at node 10 of the shared axis, 5.95 percent below the injected
semi-amplitude and 30.77 nats below the truth itself. The K axis is therefore adequate: 17
nodes at ratio 1.25 come within 31 nats of the right answer once the period and the
eccentricity are right. Either error alone destroys it. With the right eccentricity and the
table's period nothing on the grid exceeds 22261.35, and with the right period and the
table's eccentricity nothing exceeds 22162.94, both some 280 to 380 nats below the winner.

**The period peak inside the declared window is 0.05 percent wide and the comparison samples
it once, 0.25 percent away.** Profiling the coarse marginal at the true semi-amplitudes and
the true eccentricity over the full window the declaration hands the fit, 5.935460 to
6.302602 d at 121 nodes with the conjunction relocated by a 41-point scan at each node
(4961 evaluations, 186 s), the profile peaks at 22716.71 at 6.103734 d, one node from the
injected value. Two of the 121 nodes exceed the winner's 22538.46, spanning 6.100674 to
6.103734 d; three lie within 250 nats of the peak, spanning 0.100 percent of the period. At
the candidate's own central period the profile reads 22195.85, which is 520.86 nats below
the peak. The offset is 0.2507 percent, and over a baseline of 1525.39 d, or 249.9 cycles,
that is a drift of 0.627 of a cycle, so the wrong period is not a small perturbation of the
right one at any semi-amplitude.

**The eccentricity start is worth about a thousand nats on its own.** At the true period and
the true semi-amplitudes, with the conjunction relocated by a 41-point scan at each trial,
the marginal falls monotonically from 22724.96 at $`e = 0`$ through 22720.90 at 0.020,
22576.82 at 0.255 and 22099.55 at 0.489 to 21715.12 at 0.606, recovering only slightly above
0.7. The declared start of 0.5968 therefore sits about 1000 nats below the circular value,
and a circular trial alone would have been worth more than the whole margin the comparison
turned on. The circular point, at its own best conjunction, stands 1.81 nats above the truth
evaluated at the injected eccentricity and the injected conjunction, which is within what a
coarse grid and one noise realisation can be expected to move.

**The coarse grid inverts nothing: the full declaration orders every configuration the same
way and widens the truth's lead.** Re-evaluating six configurations on the declaration
itself, 3500 pixels at 3.0 km/s against the coarse copy's 1756 at 6.0, gives differences
against candidate 2's scan best of -7.75 nats for candidate 2's grid best (coarse -15.07),
-255.78 for candidate 3's scan best (coarse -250.77), -263.63 for candidate 3's grid best
(coarse -270.41), +169.72 for the true period with the true eccentricity at the best grid
node (coarse +153.92) and +203.16 for the truth (coarse +184.68). The pixel is not the
question. The coarse model is a faithful ranking of these configurations and is if anything
conservative about the truth.

**A short optimisation does not reorder the basins from the starts the scan supplies, and
does reorder them from the truth.** Running 100 L-BFGS steps on the full declaration with
the façade's own priors and conjunction window, started at each candidate's scan best, the
0.5723 d candidate reaches a potential of -22610.14 at P 0.572312 d, K 101.61 and 102.15
km/s and $`e = 0.786`$, a marginal of 22639.42, while the 6.1190 d candidate reaches
-22589.79 at P 6.121111 d, K 81.41 and 84.67 km/s and $`e = 0.273`$, a marginal of 22606.98:
the wrong period stays ahead by 20.35 nats of potential and 32.44 of marginal, neither fit
converged. Started instead at the truth, inside candidate 3's own declaration, the same 100
steps reach a potential of -22939.42 at P 6.103588 d, K 67.81 and 69.48 km/s and
$`e = 0.0242`$, a marginal of 22960.08, which is 329.28 nats of potential and 320.66 of
marginal ahead of the best the 0.5723 d candidate reached. The right basin is inside the
declaration the comparison would have built; the optimiser cannot walk to it from a start
0.25 percent off in period and a factor 24 high in eccentricity, and a scan that moves
neither cannot see it.

**Verdict: neither the scan nor the coarse grid is at fault, and the comparison is unsound
at the resolution it is run.** The semi-amplitude scan reached the best of its own grid and
better at both periods, the 17-node axis is fine enough to place the truth within 31 nats,
and the full-resolution model confirms every ordering. What loses the period is that a
candidate is scored at a single point of the two quantities the velocity table measures
worst, the period to 0.25 percent where 0.05 percent is needed, and the eccentricity to a
factor 24 where a thousand nats hang on it, while the semi-amplitudes and the conjunction,
which the table measures no better, are scanned. Left for a later work package: whether a
period axis is affordable, since covering the plus or minus 3 percent window at the peak's
own width costs about 120 nodes and, at 41 conjunctions each, 186 s per candidate against
the 30 s the comparison now spends; whether a circular trial beside the table's
eccentricity, at two scans per candidate, recovers this star and does not cost the others;
and whether `_element_starts` should stop accepting a table eccentricity on a three-sigma
test that a five-parameter Keplerian fitted to fifteen velocities passes while being wrong
by a factor 24. The other six blind-tier misses of this run never had the true period among
their candidates and are a different question.
