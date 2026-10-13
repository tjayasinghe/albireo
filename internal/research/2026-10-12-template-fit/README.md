# Templates fitted to the epochs: record of the measurements (2026-10-12 and 13)

`albireo.fit_templates` fits the labels, the rotation and the light fractions of library
templates to a multi-epoch dataset together with the velocities of every epoch (D70). The
method is in `docs/math.md` §10.7, the interface in `docs/api/templatefit.md`, and the
numbers that other pages quote in `docs/benchmarks.md`. This note records how the design
was reached: the prototype, the two reviews, the three forms of the order of the pair that
were tried, and what was measured and not adopted.

The design and the measurements were made in the main session. Two Opus agents reviewed
the code, one before the first scoring run and one after the defect that run found.

## The 48-system sample

Every stage was measured on one sample: the first 48 systems of the D69 population
(`internal/research/2026-10-07-rvs-eclipsing-todcor/runs/population.json`) with
G_RVS below 12.6 and a flux ratio above 0.1, taken in turn from the four temperature
classes. Each system is simulated as the survey simulates it, and an orbit is fitted at
the held ephemeris. The table gives the number of systems with both semi-amplitudes within
10 percent and within 5 percent, and the 68th percentiles of the absolute pulls of the two
semi-amplitudes.

| templates | within 10 % | within 5 % | pull K1 | pull K2 |
|---|---|---|---|---|
| injected | 45 | 41 | 0.92 | 1.01 |
| classified | 37 | 35 | 2.01 | 3.58 |
| one catalogue template | 33 | 26 | 3.38 | 6.58 |
| fitted from the classified ones, prototype | 44 | 40 | 1.30 | 1.17 |
| the same, first package code | 44 | 40 | 1.31 | 1.17 |
| the same, final code, the table it starts from as measured | 44 | 40 | 1.30 | 1.25 |
| the same, the table put in order by the ephemeris, as the survey does | 42 | 39 | 1.29 | 1.20 |
| fitted from the catalogue template, first package code | 39 | 39 | 1.32 | 1.33 |
| the same, final code | 40 | 40 | 1.31 | 1.38 |

The prototype (`runs/tfit_proto.py`) was a scratch implementation of the reduced
objective with finite-difference Jacobians. It gave 44 of 48 where classified templates
give 37 and the injected ones 45, at 2.4 s per system, which decided the build.

## Measured and not adopted

**The template parameters marginalised in the orbit fit.** The fit records the derivative
of every velocity with respect to every fitted parameter (`VelocityTable.sensitivity`),
and the covariance of the velocities of two epochs is
$`\delta_{jk}\mathsf C_j + \mathsf J_j \mathsf C_\phi \mathsf J_k^{\!\top}`$. An orbit fit
can carry the second term by solving for the template parameters beside the orbit, with
$`\mathsf C_\phi`$ as their prior. This was measured in two places.

- *Templates that are not fitted.* The sensitivities are evaluated at the classified
  labels and $`\mathsf C_\phi`$ is the diagonal of the classification's variances. The
  pulls of the semi-amplitudes narrow from 2.01 and 3.58 to 1.18 and 1.87, and 38 systems
  are within 10 percent where 37 were. The errors become nearly honest and the semi-amplitudes
  do not improve: a parameter with a prior as wide as a classification is not determined by
  the velocities.
- *Templates that are fitted.* $`\mathsf C_\phi`$ is the covariance of the fit. The error
  of a velocity from the templates is then 0.056 and 0.052 of its quoted error for the two
  stars in the median system, and 0.15 at the 90th percentile of the systems
  (`runs/template_sigma_48.py`). Marginalising changes the pulls from 1.30 and 1.25 to 1.25
  and 1.27 and leaves 43 systems within 10 percent where 44 were.

The orbit fits of `albireo.rvorbit` therefore use `sigma` alone, and
`VelocityTable.template_sigma` reports the size of the second term per velocity. The
scratch implementation is `mismatch_fit` in `runs/tfit_drive.py`.

**Scans of loosely declared labels.** A label declared with a wide prior was started from
several values. From the catalogue template the fit with the scan recovered 40 of the 48
systems and the fit without it 40, with one restart taken in the sample, at 11.6 s against
5.1 s per fit. It was removed.

## The first review

An Opus agent reviewed the module before the first scoring run and reproduced eight
defects, all corrected before that run:

1. A declared noise correlation was absent from the covariance of the fitted parameters,
   whose errors were then 1.33 times too small at a lag-one coefficient of 0.27. The
   covariance is now the sandwich through the reduced curvature.
2. The passes after the first were fitted to tables in the correlation's own order of the
   pair, which for alike stars is decided by noise at every epoch, and the light fraction
   was biased by about one quoted error.
3. A rotation started below half a pixel of the template grid never moved, because the
   template does not depend on it there. The derivative step now reaches past that range.
4. The refinement of an epoch's shifts did not converge at the faces of the cells of the
   integer shifts, so the objective was not a function of the parameters and the
   sensitivities there were wrong. The minimum of a cell is now the exact box-constrained
   minimum, followed from cell to cell, and a shift left on a face is held.
5. The conversion of a table to shifts compiled one array program per number of epochs.
6. Declarations that could not be fitted were not refused.
7. Several tests passed without the behaviour they named.
8. `template_sigma` was zero, and not undefined, where no parameter had a variance.

Its scripts are in `runs/review/`.

## Errors under correlated noise

Sixteen simulated datasets of twelve epochs at each lag-one coefficient, eight fitted
parameters each, on the toy binary of `tests/test_templatefit.py` (`runs/pulls_ar1.py`):

| coefficient | declared | rms pull of the parameters | rms pull of the velocities |
|---|---|---|---|
| 0 | | 1.10 | 1.08 |
| 0.27 | no | 1.42 | 1.39 |
| 0.27 | yes | 1.09 | 1.09 |
| 0.6 | no | 1.98 | 1.97 |
| 0.6 | yes | 1.06 | 1.11 |

The final code gives the same numbers as the code of the first review.

## The order of the pair: three forms

The correction of item 2 above put a table measured between two passes in the order of
the table before it, wherever the other order of an epoch's pair was nearer the velocities
it had. The first scoring run was made with it.

**The first scoring run** (`runs/_superseded/fitted-order-defect` of the survey's
directory): 1,200 systems, 0 failures, 104 minutes on 6 processes. Fitting raised the
weighted share of systems with both semi-amplitudes within 10 percent from 62 to 74
percent at G_RVS 6 to 9 from classified templates and from 47 to 67 percent from the
catalogue template. It also lost 27 and 22 systems that the declarations it started from
had recovered, several of them bright and with fitted labels close to the injected ones.

**Form 1 and system 30.** System 30 (G_RVS 6.1, light fractions 0.85 and 0.15) had K2 off
by 70 percent after the fit and by 8 percent before it. At nine of its 52 usable transits
the classified templates had left both velocities near the lines of the primary. The
fitted templates measured those transits correctly, the correct pair was nearer the wrong
starting pair in its other order, and the pair was exchanged. The order of two distinct
stars is a measurement, and the table before is no evidence against it.

**Form 2.** The exchange was made only where the chi-square of the other order, at the
minimum followed from the exchanged velocities, was within the threshold of the blend
flag of the one returned. Seven of eight bright lost systems recovered. The 48-system
sample then showed two systems worse than under the code before the first review
(G-06-001: K2 off by 15 percent; K-10-000: by 71). At a blended epoch of two distinct
stars the minimum followed from the exchanged velocities is the minimum returned, the rise
is zero, and the exchange gave the brighter star the velocity of the fainter. A pilot of
120 systems against the first scoring run could not show this, since both had the fault.

**Form 3**, the one kept. The chi-square of the other order is taken at the exchanged
velocities themselves, which is the pair the exchange writes into the table. On the 48
systems no system is lost against the code before the first review.

## The names of the two stars

Two of the bright lost systems (540 and 880 of the run) had fitted labels and light
fractions exchanged between the stars. The classified templates of the two stars were
alike, the correlation had put them on each other's lines at every transit, and the fit
converged to the image of the solution, which has the same chi-square. Three changes
followed.

- `fit_templates` moves to the image where the declared priors are lower there by more
  than 9, and keeps it if its objective is lower (system 540: a sum of priors of about 34
  against 8).
- Where the priors do not tell the stars apart (system 880: 8.2 against 7.8) the ephemeris
  does. Under a fitted declaration the survey fits the orbit in both orders of the table,
  as it did under the catalogue declaration. The classified declaration of D69 does not,
  and is unchanged: 10 of its 1,200 tables are in the other order with a light ratio above
  3 and are not recovered.
- The table a fit starts from is put in order by the ephemeris under both fitted
  declarations (a suggestion of the second review). On 120 systems the fit from the
  classified table as measured recovered 66 and had the stars in the order of their light
  for 103, and from the table put in order 71 and 116. The median error of the fitted
  light fraction fell from 0.015 to 0.010.

The table put in order by `assign_by_ephemeris` also has its flagged transits decided by
the orbit, so the first pass of the fit uses them. A variant that took only the order of
each pair from the ephemeris and left the flags of the correlation was measured beside it
(`runs/pilot_check_e2.py`, `runs/tfit_drive5.py`). Systems with both semi-amplitudes
within 10 percent, from the classified and from the catalogue declaration:

| table the fit starts from | 48 systems | 120 systems |
|---|---|---|
| as measured (in order under the catalogue declaration only) | 44, 40 | 66, 64 |
| in order, flagged transits decided | 42, 40 | 71, 64 |
| in order, flags kept | 43, 40 | 69, 63 |

The two forms of the ordered start differ by one or two systems in either direction, and
the first is the one in the package. One system of the 48 (A-12-000, G_RVS 12.1, two
rotators of 138 and 81 km/s) ends under it with no usable transit in the fitted table and
is lost, where the start as measured recovered it at 8 percent and the start with the
flags kept at 19.

## The second review

An Opus agent reviewed the three changes before the second scoring run (`runs/review2/`).
It confirmed the probe of the order in every configuration asked for, the map of the
exchange of the two stars for every kind of parameter, and the consistency of the result
after an exchange. It found:

1. The alternation could stop as converged in a pass that had changed the order of a
   usable epoch, because the change of the velocities was taken in the order that changes
   least. One fit in 80 with a hand-exchanged start, with the light fraction 2.45 quoted
   errors from the converged value. The change is now taken in the order as written.
2. The threshold of the order ignored a declared noise correlation, which widens the
   scatter of a difference of chi-squares. Two identical stars were left in the other
   order at 3 and 11 of 114 well-separated epochs at coefficients of 0.5 and 0.8. The
   threshold is now multiplied by $`\mathbf d^{\!\top}\mathsf R\,\mathbf d /
   \mathbf d^{\!\top}\mathbf d`$ for the difference $`\mathbf d`$ of the two models, and
   none of the 114 is left at 0, 0.27, 0.5 or 0.8 (`runs/order_phi.py`).
3. The comparison of the objectives does not reject the image of a correct solution whose
   priors were given to the other star: the chi-squares are equal and the priors decide.
   The epochs cannot tell the two cases apart, so the choice is now an argument,
   `exchange_stars`, with the priors naming the stars by default.
4. Five mutations of the code survived every test. Five tests were added from the
   review, two of which failed before the corrections.

It also measured that the form 3 probe can keep a pair that the starting table had in
error where the exchanged pair is within the threshold: 3 of 435 usable epochs at S/N 40
with starting temperatures 300 K off, and none of 442 at S/N 120 with 150 K. This is in
the limits of the interface.

## The second scoring run

1,200 systems under five declarations, 0 failures, 67 minutes on 8 processes
(`runs/fitted` of the survey's directory). The tables are in `docs/benchmarks.md`, and
`results/bench_fitted.txt` is the output of `scripts/template_fit_bench.py`.

| G_RVS | injected | classified | fitted from them | one catalogue template | fitted from it |
|---|---|---|---|---|---|
| 6 to 9 | 86 | 62 | 84 | 47 | 76 |
| 9 to 10 | 78 | 56 | 76 | 42 | 68 |
| 10 to 11 | 67 | 52 | 66 | 42 | 63 |
| 11 to 12 | 59 | 48 | 51 | 42 | 51 |
| 12 to 13 | 18 | 12 | 14 | 13 | 12 |

Against the first scoring run, the fit from the classified templates recovers 761 systems
where it recovered 680 (94 gained and 13 lost, of which 63 and none brighter than
G_RVS = 10), and the fit from the catalogue template 711 where it recovered 643 (68 gained,
none lost). The three declarations that are not fitted are identical to the recorded D69
run in every column compared, on all 1,200 systems.

Three further properties of the run:

- The exchange of the two stars by the declared priors was made in 3 of the 1,200 fits
  from the classified templates and in none from the catalogue template, whose priors are
  the same for both stars. With the starting table in the order of the ephemeris the step
  is rarely needed in the survey.
- The orbit of 9 and 22 recovered systems is the one of the fitted table in its other
  order.
- For 85 systems the table measured with the templates fitted from the classified ones has
  fewer than three usable transits where the classified table had six or more. None of
  them was recovered with the classified templates, their median flux ratio is 0.02, and
  52 percent of the usable transits of their classified tables are more than five quoted
  errors and 3 km/s from the injected velocities (15 percent for the other systems).

**The emission runs** (`runs/activity-fitted`, `runs/activity-emission`; 600 systems each,
0 failures). Of the 282 systems with Ca II emission, 29 percent have both semi-amplitudes
within 10 percent with the classified templates, 34 with the templates fitted and the
emission not in the model, and 50 with its equivalent width fitted beside the labels, where
the same systems without the emission give 48 and 60 percent with the classified and the
fitted templates. Fitting the emission gains 49 systems and loses 3. The fitted widths
have a median error of 0.01 Å brighter than G_RVS = 10, and the stars without emission a
fitted median of zero (`results/bench_all.txt`).

## Files

- `results/`: the 48-system results at each stage, the pull measurements, the pilots and
  the outputs of `scripts/template_fit_bench.py`.
- `runs/` (not tracked): the scratch drivers, the prototype, and the two reviewers'
  scripts and logs. The survey runs themselves are under
  `internal/research/2026-10-07-rvs-eclipsing-todcor/runs/` (`fitted`, `activity-fitted`,
  `activity-emission`, and `_superseded/fitted-order-defect`).
