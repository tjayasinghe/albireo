# What is wrong with the five velocity tables on which the blind period search never finds the period? (D65, 2026-09-16)

The blind tier of the third run (D63) measures a velocity table against library templates
at spread starting labels (6000 K for the first component and 5000 K for the second,
$`\log g = 4`$, $`[{\rm M/H}] = -0.25`$, no rotation), proposes candidate periods from four
periodograms of it (`albireo.pipeline._period_candidates`), fits an eccentric Keplerian
from each and ranks the fits by chi-square (`_orbit_over_candidates`). After the two D64
changes to the proposal (the round robin merge and fifty peaks per source) five systems
still never reach the true period at any peak count, and
[`d64_candidate_breadth.md`](d64_candidate_breadth.md) closed with the statement that for
those five the work belongs on the velocity table and not on the search. This note takes
the five tables apart epoch by epoch against the injected velocities, measures what each
kind of defect costs in periodogram power at the true period and in the chi-square rank of
the true orbit, and tests table-side rules that use no knowledge of the truth. Velocity
tables, periodograms and Keplerian fits only: no spectra were read, no `Disentangler` was
built, one process at a time, nothing under `src/` was modified.

**The tables are WP-W's reconstruction of the pre-exchange bootstrap table, and they
reproduce every power and rank D64 quoted for these five stars.** Each blind directory's
`template_velocities.rv` is written after `reassign_by_orbit`, so the exchange the winning
orbit applied was undone from the light column with WP-W's `undo_exchange` (0, 2, 11, 3 and
2 epochs put back on `gaia-7289006178185856`, `gaia-56716765427964800`, `mixed-0010`,
`mixed-0015` and `mixed-0019`). The four sources were computed exactly as
`_period_candidates` computes them at the current commit (fifty peaks for the
one-harmonic, two-harmonic and first-component searches, the default twenty for the
swap-invariant search of which the first four peaks and their doubles are used), and the
rank walker is WP-Z's. The D64 table is reproduced to the third decimal: 0.858 against
0.976 at peak 49 and 0.976 against 0.998 at peak 37 on `gaia-7289006178185856`, 0.681
against 0.896 at peak 108 on `gaia-56716765427964800`, 0.208 against 0.982 at peak 205 (and
0.275 against 0.999 at peak 271 of the two-harmonic search) on `mixed-0019`, and no local
maximum within 2% in either the one-harmonic or the two-harmonic search of the two other field
stars. The chi-square ranking reproduces
WP-AC's rank-1 chi-square at fifty peaks on all five (843.3, 739.9, 1260.4, 121408.2 and
3941.3). The run's own recorded periodogram peaks are not all reproduced (the top
one-harmonic peak of `gaia-56716765427964800` comes out at 0.27034 d against the recorded
0.17600 d, the two-harmonic top of `gaia-7289006178185856` at 0.15260 d against 0.19469 d),
and the cause was confirmed rather than assumed: moving the shortest period of the grid by
0.5 to 4 $`\times 10^{-6}`$ d, the size of the six-decimal rounding of the epoch times,
reproduces every recorded top peak of both searches on all five stars. Over shifts of
$`\pm 4 \times 10^{-6}`$ d the power at the truth moves by at most 0.002, the power of peak 1
by up to 0.02 and the peak rank of the truth by up to 8 (37 to 45 in the two-harmonic search
of `gaia-7289006178185856`), so the powers below are firm and the ranks are good to a few.

**The injected velocities were checked for frame, zero point and order before any
difference was taken.** The data are declared barycentric with no barycentric correction in
the simulator, the table is barycentric and both library templates are absolute, and the
injected velocities in `truth.epoch_velocities_true` are the stellar velocities including
gamma. They were recomputed from `population.json` with `albireo.kepler.radial_velocity` to
better than 5e-4 km/s, `truth.components_exchanged` is false on all five, and
$`K_2 v_1 + K_1 v_2 = \gamma (K_1 + K_2)`$ holds to 2e-14 km/s at every epoch, so column A
is the injected primary. What remains is the template-mismatch shift of each template, which
was estimated as the median residual over the epochs with the stars at least 1.5 FWHM apart
where one assignment puts both templates within a line width of a star, and removed before
any sigma test: +7.0 and +4.1 km/s on `gaia-7289006178185856` (10 epochs), +0.6 and 0.0 on
`gaia-56716765427964800` (9), -6.3 and -8.6 on `mixed-0010` (7), none on `mixed-0015` (no
such epoch) and -1.5 and -13.0 on `mixed-0019` (2 epochs, and therefore rough). No
periodogram sees these shifts, since every source floats a mean.

## Epoch classes

**Every epoch was given one class by rules that are fixed before any periodogram is
computed.** The width that defines a blend is the one the task set, the RVS line-spread
FWHM $`c / R = 26.07`$ km/s at $`R = 11{,}500`$ (manifest `resolving_power_min` and `max`,
and the declared LSF sigma of 11.070 km/s). A second width $`W`$, the widest line of the
pair (the LSF convolved with the limb-darkened rotation kernel at $`\epsilon = 0.6`$, as
`albireo.gaia` renders the components), is the tolerance within which a template is said to
have measured a star; it is 28 to 49 km/s on four stars and 155 km/s on the
100 km/s rotators of `gaia-7289006178185856`. The classes, first rule that applies:
*unmeasured* (a component at the search edge, velocity nan, dropped by the pipeline);
*undetected* (the secondary template more than $`W`$ from both stars, or within half an FWHM
of the measured primary while the stars are at least $`W`$ apart); *blended* (injected
separation under one FWHM); *exchanged* (the exchanged assignment has the lower chi-square
against the truth and puts both templates within $`W`$ of the other star); *misplaced*
(neither assignment within $`W`$); *outlier* (the relative velocity or the primary more than
5 sigma off after the zero points); *good*. No epoch of the five was misplaced and no
primary was undetected.

| star | P (d) | e | K1, K2 (km/s) | l2 | vsini (km/s) | epochs | usable | good | exchanged | blended | undetected | outlier | unmeasured |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gaia-7289006178185856 | 0.5440 | 0.21 | 128.3, 132.2 | 0.484 | 101, 99 | 10 | 10 | 1 | 6 | 0 | 0 | 3 | 0 |
| gaia-56716765427964800 | 5.3561 | 0.09 | 63.7, 64.6 | 0.495 | 9, 9 | 12 | 12 | 8 | 1 | 2 | 0 | 1 | 0 |
| mixed-0010 | 310.0558 | 0.70 | 20.7, 22.0 | 0.419 | 16, 2 | 62 | 62 | 6 | 1 | 55 | 0 | 0 | 0 |
| mixed-0015 | 552.5478 | 0.48 | 12.0, 21.9 | 0.073 | 20, 1 | 27 | 26 | 0 | 0 | 1 | 24 | 1 | 1 |
| mixed-0019 | 330.2300 | 0.58 | 21.6, 23.7 | 0.424 | 26, 31 | 11 | 9 | 2 | 0 | 1 | 6 | 0 | 2 |

**The table's own blend flag fired on none of the 86 epochs whose stars were less than one
FWHM apart.** `VelocityTable.blended` is raised when the correlation between the two
velocity errors exceeds 0.9 or the curvature is not positive definite; over the five tables
it is false at every epoch, so `table.good` admits every blend. The separation the
correlation reports at a blend is pushed out to about one FWHM (on `mixed-0010` a mean of
0.92 FWHM measured against 0.50 injected over its 55 blends, with the sign of
$`v_A - v_B`$ reversed on 35 of them), which is exactly the configuration in which the two
error terms are no longer strongly correlated. The secondary's detection statistic
separates the same epochs cleanly on the three field stars: it is below 100 at all 55
blends of `mixed-0010` and at or above 100 at all seven of its separated epochs, below 100
at every epoch of `mixed-0015`, and at or above 100 only at the two good epochs of
`mixed-0019`. It does not separate the two Gaia stars, whose blended and exchanged epochs
all carry statistics above 1300 and 400.

## The counterfactuals

**Each defect was costed twice, once by removing its epochs and once by replacing them with
the injected velocities, because on tables of ten to sixty epochs a removal also changes the
cadence and the grid.** The variants are: (a) the table as measured; (b) the same usable
epochs and sigmas with the injected velocities; (c) the exchanged epochs put back by the
truth; (d) the blended epochs removed; (e) the undetected epochs removed; the same classes
replaced by the truth (with the zero points added, so that a replaced epoch agrees with the
measured ones); and the primary alone as a one-component table over every epoch with a
finite primary velocity. Each variant is handed to the four sources with the grid
`find_period` builds from that variant's own usable epochs, to `_period_candidates`, and to
the ranking. The periodogram cells give the power of the highest local maximum within 2% of
the true period against the power of peak 1 of the same source, with the peak rank of the
truth after `r`. On the three long-period systems the grid step at the true period is 1.6 to
3.4% ($`P/T`$ = 0.16, 0.19 and 0.34), and on two of them the one-harmonic peak sits 2 to 4%
from the truth even on the injected velocities, so a 5% figure is given where no local
maximum lies within 2%; the Keplerian fit converges from a start that far out
(`d63_period_search_diagnosis.md`). `cand` is the position of the first proposed start within
2% of the truth in the candidate list (5% in brackets where it differs). `chi2 rank` is the
rank of the first distinct fitted orbit within 2% of the truth among the distinct valid
orbits, with the rank-1 orbit beside it, and `truth start` is the orbit fitted from a start
at the true period with the pipeline's one re-assignment, and the rank it would take. For
(b) the chi-square columns are from the injected velocities without zero points and with one
draw of Gaussian noise at the quoted sigmas, since a shared systemic velocity cannot absorb
two different template shifts; the periodogram columns are the same either way.

### gaia-56716765427964800: one exchanged epoch

**One exchanged epoch of twelve is the whole defect, and it takes the truth from peak 1 of
every one-harmonic source to peak 108.** The pair are twins (6038 and 6020 K, $`K`$ = 63.7
and 64.6 km/s, light 0.505 and 0.495) measured against templates at 6000 and 5000 K, so the
correlation has nothing with which to prefer one assignment. At epoch 7 (BJD 2457374.138)
the stars are 3.49 FWHM apart and both are detected at statistics of 15,641 and 15,577, and
the 6000 K template sits on the secondary and the 5000 K template on the primary, each
90 km/s from its own star. Of the other eleven epochs eight are good to 1.5 km/s, two are
blends at 0.29 and 0.20 FWHM measured 0.52 and 0.48 FWHM apart (one with the sign of the
relative velocity reversed), and one primary is 2.5 km/s (5.3 sigma) off.

| variant | usable | one-harmonic | two-harmonic | first component | swap-invariant at P/2 | cand | chi2 rank, rank 1 | truth start |
|---|---|---|---|---|---|---|---|---|
| (a) measured | 12 | 0.681 / 0.896 r108 | 0.810 / 0.976 r113 | 0.681 / 0.893 r108 | 0.891 / 0.972 r55 | none (none) | absent; 1.0312 d at 739.9 | 5.3561 d at 225.5, rank 1 |
| (b) injected | 12 | 0.990 / 0.990 r1 | 0.999 / 0.999 r1 | 0.990 / 0.990 r1 | 0.872 / 0.975 r63 | 1 | 1; 5.3561 d at 8.2 | rank 1 |
| (c) exchange undone | 12 | 0.984 / 0.984 r1 | 0.997 / 0.998 r2 | 0.984 / 0.984 r1 | 0.891 / 0.972 r55 | 1 | 1; 5.3564 d at 354.7 | rank 1 |
| (d) blends removed | 10 | 0.710 / 0.982 r134 | 0.897 / 0.998 r108 | 0.709 / 0.981 r131 | 0.964 / 0.982 r10 | none (20) | absent; 13.453 d at 199.4 | 5.3561 d at 19.5, rank 1 |
| blends set to truth | 12 | 0.704 / 0.924 r95 | 0.827 / 0.987 r103 | 0.707 / 0.924 r92 | 0.878 / 0.974 r62 | none (none) | | |
| (c+d) | 10 | 0.990 / 0.991 r3 | 0.999 / 0.999 r1 | 0.989 / 0.990 r3 | 0.964 / 0.982 r10 | 2 | | |
| primary alone | 12 | 0.681 / 0.893 r108 | 0.808 / 0.976 r114 | | | none (none) | absent; 0.2802 d at 924.1 | 5.3575 d at 23091, rank 49 |

**The exchange accounts for the entire gap between the measured and the injected
periodograms, the blends for none of it, and the chi-square already prefers the truth by 514
whenever a start near it is offered.** Undoing epoch 7 alone restores 0.984 against a peak 1
of the same value; replacing the two blends with the truth moves the one-harmonic truth from
peak 108 to 95, and removing them makes it worse (peak 134), because two epochs of twelve are
more cadence than they are error. The fit started at the true period on the measured table
converges to 5.3561 d at chi-square 225.5 against 739.9 for the run's rank 1, the one
re-assignment having put epoch 7 back, so the star is lost at the proposal alone. The source
meant for exchanges does not rescue it: the swap-invariant search has the half period at
peak 55 on the measured table and at peak 63 on the injected one, a one-harmonic model of
$`|v_A - v_B|`$ fitting the rectified curve of a near-circular orbit poorly on twelve epochs.
The primary alone inherits the exchanged epoch and fails. **Recoverable, and recovered by a
table-side proposal**: adding the three highest peaks of each of the twelve
leave-one-epoch-out one-harmonic periodograms (seven new starts after the 2% merge) puts the
truth at chi-square rank 1 at 225.5.

### gaia-7289006178185856: six exchanged epochs of 100 km/s rotators

**Six of ten epochs are exchanged, resolving them is the whole difference between peak 37 and
peak 1, and the 18 km/s scatter that remains once they are resolved costs no power at all.**
The stars are twins at 6735 and 6679 K rotating at 101 and 99 km/s, so each line is 155 km/s
wide against the 26 km/s LSF, and both templates are unrotated. The stars are never closer
than 3.3 FWHM (0.56 line widths at epoch 2) and both are detected everywhere (statistics 397
to 643). Exchanged are epochs 0, 1, 2, 4, 7 and 8. Once they are put back, the relative
velocity scatters about the truth by 17.9 km/s rms against a quoted 2.5 km/s, a factor 7; of
the four unexchanged epochs, three are outliers (both components 16 km/s off at epoch 3, the
primary 21 km/s at epoch 6, the secondary 14 km/s at epoch 9, after zero points of +7.0 and
+4.1 km/s) and one is good.

| variant | usable | one-harmonic | two-harmonic | first component | swap-invariant at P/2 | cand | chi2 rank, rank 1 | truth start |
|---|---|---|---|---|---|---|---|---|
| (a) measured | 10 | 0.858 / 0.976 r49 | 0.976 / 0.998 r37 | 0.882 / 0.966 r42 | 0.940 / 0.974 r7 | 47 (31) | 39 of 40; 0.3089 d at 843.3 | 0.5439 d at 63,827, rank 38 |
| (b) injected | 10 | 0.985 / 0.994 r3 | 0.999 / 0.999 r1 | 0.985 / 0.994 r3 | 0.893 / 0.958 r20 | 2 | 1; 0.5440 d at 6.3 | rank 1 |
| (c) exchanges undone | 10 | 0.983 / 0.990 r3 | 0.998 / 0.998 r1 | 0.970 / 0.985 r4 | 0.940 / 0.974 r7 | 2 | 1; 0.5440 d at 564.2 | rank 1 |
| outliers set to truth | 10 | 0.868 / 0.979 r46 | 0.975 / 0.999 r40 | 0.852 / 0.973 r48 | 0.928 / 0.964 r11 | 52 (23) | | |
| every bad epoch set to truth | 10 | 0.984 / 0.993 r3 | 1.000 / 1.000 r1 | 0.982 / 0.992 r3 | 0.900 / 0.957 r15 | 2 | 1; 0.5440 d at 9.3 | rank 1 |
| primary alone | 10 | 0.882 / 0.966 r42 | 0.959 / 0.994 r56 | | | 47 (29) | 34 of 41; 0.3089 d at 196.8 | 0.5439 d at 36,118, rank 40 |

**At the true period the exchanged table fits a compromise orbit that the one re-assignment
cannot undo, and giving the fit the freedom to re-assign lets an alias win instead.** Started
at 0.5440 d on the measured table, the Keplerian converges at the right period with
$`e = 0.61`$ and $`K`$ = 59 and 44 km/s at chi-square 63,827, rank 38; iterating the
re-assignment to convergence (five passes) changes nothing, and neither do the detection and
blend rules below, which have no epoch to act on. With the exchanges undone by the truth the
same fit reaches 564.2, and the nearest alias (0.4895 d) 887.1. Assigning the components
from a rectified circular orbit fitted to $`|v_A - v_B|`$ before the fit (no truth used)
finds that assignment at the true period, 564.2 again, but gives every alias the same
freedom, and an alias at 0.4596 d then fits at 436.6, rank 24 for the truth. In units of the
real scatter (a reduced chi-square of 43 at the truth) the two differ by about 3.
**Not recoverable by any table-side change tested.** The information that would decide the
assignment is a velocity error near the quoted 2.5 km/s rather than 18, which is a template
matched in rotation, a measurement and not a table rule.

### mixed-0015: a secondary that is not in the table

**The 7% secondary is absent from every epoch while the primary is measured to 2.4 km/s at
all 27, and the period is lost at the joint Keplerian fit, not in the periodogram.** The
secondary template is more than a line width from both stars at 24 of 26 usable epochs, 144
to 337 km/s from the injected secondary and spread across the whole search window, with
detection statistics of 1.4 to 61; at one more it is 26.8 km/s off at a statistic of 53, one
epoch is a blend, and at one the secondary is at the search edge. The primary's statistics
are 7,192 to 18,304 and its residuals $`-2.4`$ to $`+1.6`$ km/s at a quoted 0.47. The light stage had put the secondary at 0.124 of the light
against an injected 0.073.

| variant | usable | one-harmonic | two-harmonic | first component | swap-invariant at P | cand | chi2 rank, rank 1 | truth start |
|---|---|---|---|---|---|---|---|---|
| (a) measured | 26 | none (nearest +7.6%) / 0.530 | none (nearest +11.7%) / 0.848 | 0.685 / 0.685 r1 | none (nearest -6.3%) / 0.684 | 3 | absent; 0.1505 d at 121,408 | 511.06 d at 161,194, rank 20 |
| (b) injected | 26 | 0.713 / 0.713 r1 | 0.947 / 0.947 r1 | 0.712 / 0.712 r1 | 0.438 at 5% r49 / 0.631 | 1 | 1; 551.00 d at 31.8 | rank 1 |
| undetected set to truth | 26 | 0.690 / 0.690 r1 | 0.929 / 0.929 r1 | 0.717 / 0.717 r1 | none / 0.548 | 1 | | |
| (e) undetected removed | 2 | too few epochs | | | | | | |
| primary alone | 27 | 0.696 / 0.696 r1 | 0.946 / 0.946 r1 | | | 1 | 1; 547.38 d at 43.4 | rank 1 |
| secondary under detection 100 given no weight | 27 | none (nearest +7.6%) / 0.509 | none (nearest +11.7%) / 0.825 | 0.696 / 0.696 r1 | none (nearest -6.3%) / 0.635 | 3 | 1; 547.38 d at 43.4 | rank 1 |

**The first-component source that D63 added for this case proposes the truth as its own peak
1, and the joint fit then throws it away.** The candidate within 2% of the truth is third in
the list. Fitted jointly, the secondary's noise velocities carry weights of a real
measurement (sigmas near 3 km/s), the start for its semi-amplitude is half the range of its
velocities, 321 km/s and above the declared ceiling of 250, and the fit from the true period
ends at 511 d with $`e = 0.77`$ and $`K`$ = 49 and 66 km/s at chi-square 161,194. The same
table with the secondary given no weight wherever its detection statistic is under 100 (every
epoch, the one with the secondary at the search edge kept for its primary) ranks 547.38 d
first at 43.4, $`-0.9`$% from the truth with $`K_A`$ = 10.4 km/s, the orbit of the primary
alone. The relative-velocity sources gain nothing from it, since a weight removed from every
epoch leaves the relative weights as they were; the rescue is the first-component source
(0.696, peak 1) and the fit. At the statistic's own summary threshold of 25 the rule fails
(rank 1 at 1.27 d, the truth start drifting to 671 d with $`K_B`$ = 216 km/s), because six
epochs between 25 and 61 keep secondary velocities 19 to 212 km/s off.
**Recoverable, and recovered by gating the secondary at a detection statistic of 100.**

### mixed-0010: fifty-five blends with a coin-flip sign

**Fifty-five of 62 epochs are blends whose relative velocity has a random sign and a
separation pushed out to about one FWHM, and they outweigh the seven epochs at which the lines
are apart and measured.** The injected separation is under one FWHM at 55 epochs (mean 0.50
FWHM), where the correlation reports a mean of 0.92 FWHM with the sign reversed at 35, the
primary off by up to 25 km/s, and a secondary statistic of 7 to 65. Of the seven epochs at
1.65 to 2.27 FWHM six are good to 6.4 km/s after zero points of $`-6.3`$ and $`-8.6`$ km/s,
and one (epoch 28) is exchanged.

| variant | usable | one-harmonic | two-harmonic | first component | swap-invariant at P | cand | chi2 rank, rank 1 | truth start |
|---|---|---|---|---|---|---|---|---|
| (a) measured | 62 | none within 5% (nearest -5.4%) / 0.292 | 0.110 at 5% r198 / 0.347 | none within 5% (nearest -6.8%) / 0.261 | 0.316 / 0.316 r1 | 4 | absent; 0.7691 d at 1260.4 | 318.56 d at 2931, rank 44 |
| (b) injected | 62 | 0.561 at 5% r1 / 0.561 | 0.773 at 5% r1 / 0.773 | 0.563 at 5% r1 / 0.563 | 0.438 / 0.438 r1 | (1) | 1; 310.19 d at 111.8 | rank 1 |
| (c) exchange undone | 62 | none within 5% / 0.331 | 0.173 at 5% r150 / 0.395 | none within 5% / 0.298 | 0.316 / 0.316 r1 | 3 | absent; 12.584 d at 1495.9 | 313.30 d at 2825, rank 50 |
| (d) blends removed | 7 | 0.468 / 0.991 r160 | 0.517 / 1.000 r234 | 0.527 / 0.994 r175 | 0.900 at 5% r172 / 0.999 | none | absent; 0.9617 d at 13.5 | 310.14 d at 364.6, rank 32 |
| blends set to truth | 62 | 0.427 at 5% r1 / 0.427 | 0.561 at 5% r1 / 0.561 | 0.502 at 5% r1 / 0.502 | 0.440 / 0.440 r1 | (1) | | |
| primary alone | 62 | none within 5% (nearest -6.8%) / 0.261 | 0.133 at 5% r179 / 0.336 | | | none | absent; 2.2571 d at 1047.0 | 311.22 d at 1342.8, rank 49 |

**The blends are the whole periodogram defect and the one exchange none of it, but the
proposal is not what fails: the swap-invariant search already has its peak 1 at 311.94 d,
0.6% from the truth, fourth in the candidate list.** Replacing the 55 blends with the truth
restores peak 1 in all three one-harmonic and two-harmonic sources; undoing the exchange moves
nothing. The fit from the true period on the measured table ends at 318.6 d with
$`e = 0.82`$ and $`K`$ = 12.6 and 7.2 km/s at chi-square 2931, rank 44 against 1260.4 for a
0.77 d alias, so the loss is at the fit, where a relative velocity whose sign is a coin flip at
55 epochs has no Keplerian that fits it. Removing the blends leaves seven epochs, on which
nothing is decidable. Giving the secondary no weight below a statistic of 100 (exactly the 55
blends) makes the truth start win (312.19 d at 906.4 against 1026.4 for rank 1), but no
proposed start converges there, and adding the measured table's candidates (the start at
311.94 d) reaches only rank 14. What works is to re-sign the relative velocity before the
fit: assigning the components from a rectified circular orbit fitted to $`|v_A - v_B|`$ at
each candidate period re-assigns 20 epochs at the true period and ranks 311.70 d first at
chi-square 1062.0. **Recoverable, and recovered by the seeded assignment, subject to what it
does to the other tables (below).**

### mixed-0019: two measured secondaries in seven visits

**Only two of nine usable epochs measure the secondary, the primary is corrupted at three of
the five epochs where the lines are apart, and even perfectly measured velocities on these
epochs do not decide the period.** Eleven epochs fall in seven visits (two pairs and a triple
within 0.25 d), so the table samples at most seven phases. The secondary is detected
(statistics 169 and 136) only at epochs 4 and 9, both at 2.2 FWHM and both good. At epochs 2
and 8, 2.1 and 2.3 FWHM apart, it is lost (statistics 14 and 5, measured 251 and 368 km/s
from the injected secondary) and the 6000 K template has taken the blend, the primary sitting
18 and 20 km/s toward the secondary; at epoch 3, in the same visit as epoch 2, the secondary
is at the search edge and the primary 16 km/s off. The remaining four undetected epochs, one
blend and one more unmeasured epoch are all at 0.50 to 0.74 FWHM. The light stage had put the secondary at
0.268 of the light against an injected 0.424.

| variant | usable | one-harmonic | two-harmonic | first component | swap-invariant at P/2 | cand | chi2 rank, rank 1 | truth start |
|---|---|---|---|---|---|---|---|---|
| (a) measured | 9 | 0.208 / 0.982 r205 | 0.275 / 0.999 r271 | none within 5% (nearest -19.7%) / 0.966 | 0.395 / 0.980 r189 | none | absent; 0.2192 d at 3941.3 | 338.55 d at 13,091, outside the ranges |
| (b) injected | 9 | 0.998 at 5% r1 / 0.998 | 1.000 / 1.000 r6 | 0.998 at 5% r1 / 0.998 | none / 0.997 | 14 (1) | 3; 1.7976 d at 7.1 | 329.41 d at 9.4, rank 3 |
| undetected set to truth | 9 | 0.919 at 5% r41 / 0.982 | 0.931 / 0.998 r188 | 0.998 at 5% r3 / 0.998 | none / 0.982 | none (11) | | |
| every bad epoch set to truth | 9 | 0.989 at 5% r7 / 0.996 | 0.997 / 0.999 r26 | 0.998 at 5% r1 / 0.998 | none / 0.991 | 54 (3) | | |
| (e) undetected removed | 3 | too few epochs | | | | | | |
| primary alone | 11 | none within 5% (nearest -10.4%) / 0.891 | none within 5% (nearest -5.7%) / 0.994 | | | none | absent; 1.0337 d at 28.3 | 355.16 d at 118.4, rank 32 |

**The six undetected epochs carry the periodogram defect, and nothing on the table can replace
them.** Setting them to the truth takes the first-component source from no local maximum
within 5% to peak 3 at 0.998 and the one-harmonic source from peak 205 to 41; removing them
leaves three epochs. Neither the primary alone (dragged toward the blend at three of five
separated epochs), nor the detection or blend rules, nor the seeded assignment, nor the
leave-one-out proposal brings a fitted orbit within 5% of the truth.

**On these nine epochs even the injected velocities do not decide the period: with noise at
the quoted sigmas the truth ranks first in 2 of 10 draws, and every loss is by a chi-square
difference under 5.** The ten draws put the truth at ranks 4, 1, 5, 1, 5, 2, 3, 4, 5 and 4,
losing to aliases at 5.87 to 66.4 d by 1.4 to 4.7 in chi-square, far inside the 25 at which
the pipeline already calls a period ambiguous. With the two epochs whose secondary was at the
search edge restored (eleven epochs, five draws) the truth is first in 3 of 5, and the two
losses are by 4.0 and 1.0. Noise-free, the injected table ranks the truth first at a
chi-square of zero, so the cadence is not blind, only too sparse to be decisive at these
errors. **Not recoverable by a table-side change; on the nine usable epochs not recoverable
in principle either, and on all eleven only in the sense of an ambiguity that a comparison
would have to settle.**

## Whether the lines of the three long-period systems ever part

| star | line FWHM A, B (km/s) | epochs at >= 1, 1.5, 2 FWHM | epoch separation min / median / max (FWHM) | orbit maximum (FWHM) | fraction of the orbit above 1, 1.5, 2 FWHM | nights, all epochs |
|---|---|---|---|---|---|---|
| mixed-0010 | 32.6, 26.1 | 7, 7, 4 of 62 | 0.01 / 0.64 / 2.27 | 2.45 | 0.133, 0.082, 0.046 | 46 |
| mixed-0015 | 36.2, 26.0 | 4, 4, 0 of 27 | 0.02 / 0.64 / 1.56 | 1.60 | 0.177, 0.058, 0.000 | 19 |
| mixed-0019 | 42.9, 48.9 | 5, 5, 5 of 11 | 0.50 / 0.74 / 2.28 | 2.73 | 0.155, 0.111, 0.075 | 8 |

**The lines part by more than one FWHM for 13 to 18% of each orbit and by more than two for at
most 8%, and at the epochs where they are apart the correlation measures both stars on one
system, one star on another and both at only two of five epochs on the third.** The fractions
are uniform in time over one period, from the injected elements. `mixed-0010` reaches 2.45
FWHM and its cadence samples the 13% in proportion: 7 of 62 epochs lie at 1.65 to 2.27
FWHM, and six of those are measured to 6.4 km/s with the seventh exchanged. `mixed-0015`
never reaches two FWHM, its $`(K_1 + K_2)(1 + e)`$ being 50 km/s or 1.9 FWHM and its injected
maximum 1.60, and at the four epochs at 1.55 and 1.56 FWHM its primary is measured to 1.6 km/s
while its secondary is undetected at three and 26.8 km/s off at the fourth; a 7% secondary is
not measurable at 1.6 FWHM on this S/N whatever the table does. `mixed-0019` reaches 2.73
FWHM, and 5 of its 11 epochs are at 2.1 to 2.3, so its cadence found the separated phases
about as often as the orbit offers them; the correlation measured both stars at two of the
five (epochs 4 and 9), lost the secondary and pulled the primary 18 and 20 km/s toward it at
two (epochs 2 and 8), and lost the secondary at the search edge at one (epoch 3). For all
three the largest separation the orbit ever offers is 1.6 to 2.7 FWHM. Against templates 230
to 1540 K from the stars' temperatures that is enough for the correlation to measure a 42%
companion at some separated epochs and not at others of the same separation, and never enough
for a 7% one; the light stage, reading the same correlation, had put the secondaries at 0.235,
0.124 and 0.268 of the light against injected 0.419, 0.073 and 0.424.

## What the pipeline could do differently

**Two of the five are recovered by a table-side change that costs no star the search already
finds, a third by a change that is not yet safe, and two are not recoverable from their
tables.** Every rule below was measured on all 33 blind tables of the third run, with WP-AC's
ranking at fifty peaks and the round robin (the pipeline as it stands, 20 truths at rank 1) as
the baseline, and run only where it changes the table.

| rule (no truth used) | tables it changes | rank 1 lost | rank 1 gained | other movement |
|---|---|---|---|---|
| secondary given no weight where its detection statistic is under 100, primary kept | 9 | none (4 of 4 kept) | `mixed-0015` | `gaia-51574864941234048` 20 to 8 |
| primary alone when fewer than a quarter of the usable epochs detect the secondary at 100 | 6 | none (2 of 2 kept) | `mixed-0015` | none |
| three peaks of every leave-one-epoch-out one-harmonic periodogram added, tables of at most 25 usable epochs | 17 | none (8 of 8 kept) | `gaia-56716765427964800` | `gaia-53290511099783296` 37 to 38 |
| components assigned from a rectified circular orbit fitted to the magnitude of the relative velocity, before every candidate's fit | 33 | 2 of 20: `gaia-56861694804053120` to 2, `mixed-0011` to 17 | `mixed-0010`, `mixed-0014` (40), `mixed-0017` (2), `gaia-37608387208382848` (3), `gaia-47620265212420096` (6) | `gaia-7289006178185856` 39 to 24 |
| the candidate loop without the re-assignment (light ratio 7.1 and 3.25) | 2 measured | | none | `mixed-0015` absent to 28, `mixed-0010` absent at 2% |

**Gate the secondary on its detection statistic at 100, and search and fit on the primary
where the gate removes it.** On `mixed-0015` this is the whole repair: absent to rank 1 at
547.38 d, and over the nine tables it touches no recovered star moves, including the two it
turns into single-lined fits in effect (`mixed-0007`, no epoch above 100, and `mixed-0008`,
one of 38). The threshold is D63's, and it is the one that works: at the table summary's own
threshold of 25 the same rule fails on `mixed-0015` because six epochs between 25 and 61 keep
secondary velocities up to 212 km/s off. On the field stars 100 is also the blend marker the
table's flag is not (all 55 blends of `mixed-0010` below, all 7 separated epochs above). Doing
this properly needs the two library defects below repaired first: an epoch whose secondary is
gated or unmeasured must keep its primary in the first-component source and in the fit
(the measurement here filled the unmeasured secondary with a zero-weight copy of the primary
to get round it), and the semi-amplitude start must not come from the gated velocities.

**Propose the peaks of the leave-one-epoch-out periodograms on sparse tables.** On
`gaia-56716765427964800` one exchanged epoch of twelve is the whole defect and no statistic on
the table marks it (its detection statistics are 15,641 and 15,577); leaving each epoch out in
turn and adding the three highest one-harmonic peaks of each puts the truth at rank 1. Over
the 17 tables of at most 25 usable epochs the source adds 18 starts in total after the 2%
merge (none on eight tables, seven on this one) and moves no rank 1. It is a proposal change
rather than a table change, but it is aimed at a table defect, one epoch that a periodogram
cannot outvote on ten to twenty-five epochs and a Keplerian with re-assignment can.

**Do not adopt the seeded assignment as it stands, and measure it as a second fit per
candidate inside the comparison instead.** It is the only rule that recovers `mixed-0010`
(rank 1 at 311.70 d), and it also recovers four stars the search had lost to aliases
(`mixed-0014` from rank 40, `mixed-0017` from 2, `gaia-37608387208382848` from 3,
`gaia-47620265212420096` from 6), a net of three systems, but it costs two recovered ones,
`gaia-56861694804053120` to an alias at 3.93 d and `mixed-0011` to one at 4.91 d, because the
freedom to re-sign the relative velocity helps an alias as much as the truth. That trade is
the one `gaia-7289006178185856` shows in miniature: the seeded fit reaches the right
assignment at the true period (chi-square 564.2, exactly the fit with the exchanges undone by
the truth) and an alias at 0.4596 d beats it at 436.6. Whether the disentangling comparison,
which sees the spectra and not the signs, would keep the five gains and undo the two losses is
the measurement that decides it; `gaia-56861694804053120`'s truth is at rank 2 and would be
compared, `mixed-0011`'s at 17 would not at the current default of four
(`period_decision_candidates`) or at eight.

**Flag a table observed on fewer than eight nights as one whose period the velocities cannot
decide, and state the two unrecoverable systems as such.** `mixed-0019` has nine usable epochs
on seven nights and the secondary detected at two; with the injected velocities and noise at
the quoted sigmas the truth ranks first in 2 of 10 draws, every loss inside a chi-square
difference of 5, so no table-side change can do better than an ambiguity and the pipeline's
own flag (rivals within 25) is the honest output. Of the 33 tables three have usable epochs on
fewer than eight nights (`gaia-17170287112790656`, `gaia-7289006178185856`, `mixed-0019`), and
the search recovers none of them; the count is small and the threshold is a description of
these tables rather than a derived limit. `gaia-7289006178185856` is unrecoverable for a
different reason: its cadence decides the period on perfect velocities (rank 1 on the
injected table with noise), but its measured errors are 7 times the quoted ones and with free
assignment an alias fits better, so the repair is a template matched in rotation, which is a
measurement and not a rule on the table.

**The per-star verdicts.**

| star | what is wrong | cost in power at the truth (one-harmonic, or the first component) | recoverable by a table-side change |
|---|---|---|---|
| gaia-56716765427964800 | 1 of 12 epochs exchanged (twins) | 0.681 at peak 108 against 0.984 at peak 1 with it undone | yes: leave-one-out candidates, rank 1 |
| gaia-7289006178185856 | 6 of 10 exchanged; errors 7 times the quoted ones (100 km/s rotators, unrotated templates) | 0.976 at peak 37 (two-harmonic) against 0.998 at peak 1 with them undone | no: with free assignment an alias fits better |
| mixed-0015 | secondary (7% of the light) undetected at 24 of 26 epochs; primary good | none in the first-component source (peak 1); the joint fit loses it | yes: secondary gated at 100 or the primary alone, rank 1 |
| mixed-0010 | 55 of 62 epochs blended, 35 with the sign of the relative velocity reversed | no maximum within 5% against 0.427 at peak 1 with the blends set to the truth | only by the seeded assignment, which is not yet safe |
| mixed-0019 | secondary measured at 2 of 9 epochs, primary dragged at 3 of 5 separated ones; 7 nights | 0.208 at peak 205 against 0.919 at peak 41 (5%) with the undetected epochs set to the truth | no, and on nine epochs not in principle |

## Defects in the library found on the way

**The table's blend flag does not detect blends at this resolution.** `VelocityTable.blended`
(`todcor.py`, the correlation of the two velocity errors above 0.9 or a curvature that is not
positive definite) is false at all 86 epochs of the five tables whose injected separation is
under one FWHM, so `VelocityTable.good` hands every blend to the periodograms and the fit.
The reason is visible in the measured separations: the correlation reports a blend at about
one FWHM, where the two error terms are no longer strongly correlated. On the field stars the
secondary's detection statistic below 100 is the blend marker the flag is not (55 of 55 blends
of `mixed-0010`, none of its 7 separated epochs).

**The first-component source drops the primary wherever the secondary was not measured.**
`find_period(table, components=[name])` and `fit_rv_orbit(..., components=...)` both intersect
their own finiteness test with `table.good`, which requires every component to be finite and
off the search edge, so an epoch whose secondary is at the edge is removed from the search on
the primary alone, the source `_period_candidates` added for a companion the templates could
not follow. It costs one primary epoch on `mixed-0015` and two on `mixed-0019`, and it would
cost every blend if the blend flag worked.

**The semi-amplitude start is set by the least trustworthy velocities.** `fit_rv_orbit` starts
each $`K`$ at half the unweighted range of that component's velocities, so a companion's noise
draws set it: 321 km/s on `mixed-0015`, above the declared ceiling of 250, and the same
whatever weight the velocities carry in the fit.

**The bootstrap re-assigns components that the later stage would not.** `_orbit_over_candidates`
calls `reassign_by_orbit` on every candidate, while the post-disentangling stage skips the
exchange when the light fractions differ by more than a factor 3 (`_exchange_allowed`). The
bootstrap's global light is a factor 7.1 apart on `mixed-0015` and 3.25 on `mixed-0010`, and
the run's winning orbits re-assigned 3 and 11 of their epochs. It is an inconsistency and not
the cause of either loss: ranking the measured tables with no re-assignment at all moves the
truth of `mixed-0015` from absent to rank 28 of 39 (the fit from the truth still ends with
$`K_B`$ = 618 km/s) and leaves `mixed-0010` with no orbit within 2% of it.

**The top of the periodogram at short period is decided by the rounding of one epoch time.**
`find_period`'s default grid ends at $`1/(2\,\Delta t_{\min})`$ and is uniform in frequency, so
a shift of $`10^{-6}`$ d in the closest pair of epochs moves the upper frequencies by more than
one grid step. On all five tables a shift of 0.5 to 4 $`\times 10^{-6}`$ d reorders the top
one-harmonic and two-harmonic peaks and reproduces the run's recorded peaks, which the
six-decimal `template_velocities.rv` cannot. D64 named the rounding; this measures it. The
power at a true period moves by at most 0.002, but the order of near-degenerate aliases, and
with it which of two starts 2% apart survives the merge, is not reproducible from the written
products.

**The swap-invariant source is weak on sparse tables even on perfect velocities.** On the
injected velocities its half-period peak is at rank 63 of `gaia-56716765427964800` and rank 20
of `gaia-7289006178185856`, and only its first four peaks are proposed. A one-harmonic model
of $`|v_A - v_B|`$ fits the rectified curve of a near-circular orbit poorly; this is a limit
of the statistic rather than a coding error, and it is why neither Gaia star's exchanges are
rescued by the source written for them.

## What is left

**The gains are bounds, as D64's were.** No rule was run end to end, and a blind re-run of
`mixed-0015` with the detection gate and of `gaia-56716765427964800` with the leave-one-out
candidates is the check that would turn two rank-1 orbits into two recovered periods, since
the disentangling comparison and everything after it still stand between them.

**The gate was measured with a workaround for the two library defects it depends on.** An
unmeasured secondary was filled with a zero-weight copy of the primary so that `table.good`
kept the epoch, and the semi-amplitude start was left as the library sets it. The repaired
form (the first-component source and the fit taking the primary wherever it is finite, and a
start from the weighted velocities) has not been measured, and it changes the tables of
`mixed-0015` and `mixed-0019` by one and two epochs.

**The seeded assignment is unmeasured in the one place it could be safe.** Entering both the
plain and the seeded fit of each candidate into the disentangling comparison, rather than
replacing one with the other in the chi-square ranking, is what would say whether its five
gains survive without its two losses; that needs the coarse disentangling scans and is outside
a table-only measurement.

**A working blend flag should mark epochs, not remove them.** On `mixed-0010` removing the 55
blends leaves seven epochs on which nothing is decidable, while down-weighting them makes the
chi-square prefer the truth when started there (906.4 against 1026.4 under the gate, 208.5
against 322.2 with both rules), so repairing `VelocityTable.blended` in a way that feeds
`table.good` would make this star worse. What a repaired flag should do downstream is a design
decision this note does not settle.

**The classes rest on two soft quantities, and no rank or power does.** The 5 sigma outlier
test uses the quoted sigmas, which are a factor 7 too small on `gaia-7289006178185856`, and the
zero points of `mixed-0019` come from two epochs. Both change only which name an epoch gets
(outlier or good); every counterfactual that replaces or removes a class was also run with
every non-good epoch replaced (`fix_all` in the tables), and the cadence verdict rests on ten
noise draws for `mixed-0019` and one for each of the other four.

Scripts, per-epoch CSV files (`epochs_<star>.csv`), the classification (`classify.py`,
`classes.json`), the counterfactual periodograms (`counterfactuals.py`,
`counterfactuals.json`), the fits (`fits.py`, `seeded.py`, `cadence_draws.py`,
`noexchange.py` and their JSON), the reproduction checks (`repro_check.py`, `repro_grid.py`,
`repro_power.py` and their logs), the measurement over the 33 tables (`safety.py`,
`safety.json`, `safety_table.py`, `safety.md`) and the table script with its output
(`tables.py`, `tables.md`) are in the session scratchpad under `wp-ae/`, and read `wp-w/`,
`wp-z/` and `wp-ac/` unchanged.
