# What does a converged label fit on the epoch chi-square return? (D65, 2026-09-16)

`d65_label_likelihoods.md` found that $`L_{\rm data}`$, the chi-square of the template composite
against the epoch spectra with the orbit fixed, is the right objective for the labels, and that
every fit it ran stopped near its node-scan warm start, native included, so that no temperature
in that note measured a likelihood. This note builds an estimator that converges on
$`L_{\rm data}`$, runs the native diagonal comparison through the same optimiser, and asks what
each returns. The sample is that note's twelve products and, for each of its eleven systems, the
orbit-tier product of the third benchmark run, rebuilt at that tier's archived MAP and declared
light; `mixed-0008__orbit` belongs to both sets, so there are 22 distinct products. The
re-simulation, the declaration rebuild and the archive check are WP-AB's scripts, unchanged.
Scripts, per-product JSON, logs and tables are in the session scratchpad `wp-af/`; the estimator
itself is `estimator.py`, written so that it could be lifted into `albireo.match`.

**The operator carries only the width that the library does not already have.** The label model
renders its template from BOSZ at the library's declared resolving power, $`R = 20{,}000`$, so the
LSF in $`A`$ was replaced by

```math
\sigma_q = \sqrt{\sigma_{\rm inst}^2 - \sigma_{\rm lib}^2}, \qquad
\sigma_{\rm lib} = \frac{c}{R_{\rm lib}\, 2\sqrt{2\ln 2}} = 6.366\ {\rm km\,s^{-1}},
```

with $`\sigma_{\rm inst} = 11.070`$ km/s the declared RVS width, giving $`\sigma_q = 9.057`$ km/s,
which equals the width the simulation applied to 0.0 on every product. $`h`$, $`z^\top W z`$ and the
band of $`G`$ were built once per product through `forward.with_lsf`, at the archived MAP.

**Both routes minimise a least-squares objective in the constrained space, with no Jacobian
term.** With $`\phi`$ the sixteen parameters (a shared [M/H]; per component $`T_{\rm eff}`$,
$`\log g`$, $`v\sin i`$ and a frame velocity $`v`$; the radius ratio $`r_B`$; three Chebyshev
offsets $`c_{ik}`$ per component) and $`m(\phi)`$ the stacked rows of `albireo.match._model_rows`,
built through the `LabelProblem` that `match_labels` itself constructs,

```math
F_{\rm data}(\phi) = L_{\rm data}\big(m(\phi)\big) + \sum_{i,k} \frac{c_{ik}^2}{s^2}, \qquad
F_{\rm native}(\phi) = \sum_{i,p} w_{ip}\, \frac{\big(m_{ip}(\phi) - \hat d_{ip}\big)^2}{\sigma_{ip}^2} + \sum_{i,k} \frac{c_{ik}^2}{s^2},
```

with $`s = 0.05`$, the offsets' Normal prior kept as its quadratic, $`\sigma_{ip}`$ the archived
posterior standard deviation and the jitter held at one. Every other prior is flat inside its
bounds, and the bounds are enforced by an active set rather than by a transform. The library is a
complete box, so `match_labels` selects the Catmull-Rom interpolator, which is $`C^1`$, and the
hull guard is inactive; the frame velocity enters through `operators.shift_spectrum`, a linear
interpolation whose derivative jumps at whole-pixel shifts, so the objective has kinks in $`v`$. Two prior configurations were run on every product and route: `box`, the
library's range in $`T_{\rm eff}`$ (4000 to 7000 K), $`\log g`$ (3 to 5) and [M/H] (-1 to 0.5),
$`v\sin i`$ from 0 to 150 km/s and $`v`$ within $`\pm 150`$ km/s, which is the orbit and blind tiers'
configuration and the primary result; and `oracle`, the same with $`\pm 300`$ K and $`\pm 0.3`$ dex
centred on the injected values, as archived.

**The optimiser is a bounded Levenberg-Marquardt that does not see the parameter scaling.** The
Jacobian $`J = \partial m / \partial \phi`$, of size $`n_{\rm stack} \times 16`$, comes from
`jax.jacfwd` of the rows, and the gradient and Gauss-Newton matrix are

```math
g_{\rm data} = -2\, J^\top (h - G m), \quad H_{\rm data} = 2\, J^\top G J, \qquad
g_{\rm native} = 2\, J^\top W_d (m - \hat d), \quad H_{\rm native} = 2\, J^\top W_d J,
```

with $`G`$ applied to the sixteen columns through the band matvec, $`W_d = {\rm diag}(w/\sigma^2)`$,
and the offset prior's exact gradient and Hessian added by autodiff. Since $`L_{\rm data}`$ is
exactly quadratic in $`m`$, the only approximation is the neglected curvature of $`m(\phi)`$. A
step solves $`(H_{\mathcal F} + \lambda D_{\mathcal F})\, s_{\mathcal F} = -g_{\mathcal F}`$ on the
free set $`\mathcal F`$, with $`D`$ the running maximum of $`{\rm diag}(H)`$ (Marquardt's scaling,
which makes the step invariant to a diagonal rescaling of $`\phi`$), and is projected onto the
bounds. A parameter on a bound whose gradient points outward is held. The step is accepted when
the actual decrease exceeds $`10^{-4}`$ of the quadratic model's prediction, and $`\lambda`$ follows
Nielsen's gain-ratio update. The fit stops when the undamped Gauss-Newton step predicts a decrease
below $`10^{-3}`$ in chi-square with the active set unchanged from the previous iteration, on a
budget of 150 iterations and 600 evaluations, or when no damping yields a decrease ("stalled").
The formal covariance is $`2 H^{-1}`$ over the parameters neither on a bound nor on the rotation
plateau described below, and the light fraction's error follows by the delta method through the
band median of $`w_i(\lambda)`$ that `LabelMatch.flux_ratio` reports.

**Every fit starts where the pipeline starts, and is then restarted across the directions the
data constrain weakly.** The designed start is the node scan of `match_labels`, captured from the
shipped function by stopping it at its optimiser, so that nothing of it is copied: LM runs from
each of its four candidates and the lowest objective is kept. A restart round then evaluates the
route's own objective at every library $`(T_{\rm eff}, \log g)`$ node the priors admit for the
component with the smaller fitted light (65 nodes under the box priors, at most six under the
oracle ones), every other parameter held at the current optimum, and runs LM from the three best
nodes that lie more than one node step from the optimum and from each other. Rounds repeat, up to
three, while the best objective falls by more than 1. Once per product, route and configuration,
LM also starts from the injected labels, with $`v = \gamma`$ and the ratio and offsets at their
designed start; that solution is a diagnostic and is never selected. The criterion for convergence
is that the restarted best and the truth-started solution agree within 1 in chi-square and within
0.3 formal sigma on every label ($`T_{\rm eff}`$, $`\log g`$, [M/H], $`v\sin i`$ of both components).

**The restart design needed two measured additions, a scan in rotation and a second restart from
the truth start.** The pixel-integrated rotation kernel of `operators.rotational_kernel_traced` is
exactly a delta for $`v\sin i \le 0.5`$ model pixel, 1.5 km/s here, so every objective is exactly
flat there and its gradient in $`v\sin i`$ is zero. On the first product, `mixed-0008__oracle`
under the oracle priors, all four designed candidates stopped on that plateau at
$`v\sin i_B = 0.0`$, 1.67 in chi-square above the truth-started optimum at 11.8 km/s, and the
$`(T_{\rm eff}, \log g)`$ scan had no distinct node to offer inside $`\pm 300`$ K and $`\pm 0.3`$ dex.
Each round therefore also evaluates the objective at twenty values of each component's
$`v\sin i`$ from 0 to 150 km/s and runs LM from the best two that differ from the optimum and from
each other by more than 2.5 km/s or a quarter of the value. The same plateau holds the truth start
itself wherever the injected rotation is below 1.5 km/s and the objective's minimum is not (on
`mixed-0004` B, injected at 0.63 km/s, the data route's optimum is at 6.3 km/s), and there the
criterion reports a restarted best lower than the truth start, which is not a restart failure. The same restart rounds were therefore also run from the truth-started
solution, and the criterion is reported against both. The plateau also exposed a defect in the
first version of the covariance, since fixed: the zero Gauss-Newton row made the matrix singular,
the pseudo-inverse returned a zero variance, and the criterion divided by it on `mixed-0015`. A
$`v\sin i`$ on the plateau is now excluded from the covariance like a parameter on a bound, and a
label that both solutions leave on the plateau, or at the same bound, counts as agreeing.

**Every gate passed on every product, to rounding.** G1: the marginal at the archived MAP
reproduces the archived `spectrum_*.txt` to at most 5.0e-10 on all 22 products, the rounding of the
saved file; $`\hat d = P^{-1} h`$ rebuilt from `forward.rhs` and the marginal's own factor agrees to
4.0e-13; $`z^\top W z - h^\top P^{-1} h`$ equals the chi-square reconstructed from the archived
marginal log-likelihood to 8.9e-13 relative; `forward.with_lsf` at the declared width reproduces the
build kernel to 2.8e-17; and $`\sigma_q`$ from the library's declared resolving power equals the
simulation's applied width to 0.0. G2: $`L_{\rm data}`$ through the band statistics of the
quadrature operator, both as evaluated inside the jitted objective and from the rows, equals WP-AD's
dense brute force, built from `docs/math.md` §1.4a with a per-epoch AR(1) covariance and
`forward.apply_model` and sharing no band assembly with the fast path, at nine points per product
(the injected components and each route's restarted and truth-started optima under both prior
configurations) to at most 8.5e-14 relative. G3: redeclaring $`\ell^0_B \to 0.5\,\ell^0_B`$ and
recomputing the rows through `_model_rows` with the rescaled $`\ell^0`$ and offsets leaves the rows
unchanged to 3.7e-15 in the product $`\ell^0 m`$ and $`L_{\rm data}`$ unchanged to 4.9e-14 relative
at the box data optimum, and to 0.0 at the injected components.

**The estimator converges, in a median of four iterations, on 85 of the 88 product, route and prior
cells.** Over the 1673 LM runs of the sample (designed candidates, restarts, truth starts and the
restarts from them), 1633 stopped on the criterion, one on the iteration budget and 39 stalled, all
of them on the two products whose archives are compromised (18 on `mixed-0017__orbit`, which
exchanged its components, and 21 on `gaia-5157` in either tier, whose orbit is wrong), 82 percent of them
within 1 in chi-square of their cell's best and with a predicted decrease of at most 0.8 when they
stopped. The median run took 4 iterations, the 90th percentile 14. Against the single truth-started
fit the criterion holds in 73 cells; in 12 more the restarted best is lower than the truth start by
2.0 to 584, every one of them a truth start held on the rotation plateau or in a neighbouring basin
(`mixed-0004` in both tiers, routes and prior configurations; `gaia-1717` in both tiers and
`gaia-5157` on the native route; `gaia-5157` on the data route); in one
(`mixed-0017__orbit`, native) the two agree to 0.29 in chi-square with a temperature 0.31 sigma
apart; and in two the truth start is lower. Against the truth start followed by the same restarts,
the criterion holds in 85 of 88 cells, 42 of 44 on the data route and 43 of 44 on the native one.
Counted by route and prior configuration over the 22 products:

| route, priors | single truth start: agree, truth start held higher, flat, truth start lower | truth start and restarts: agree |
| --- | --- | --- |
| data, box | 18, 3, 0, 1 | 21 of 22 |
| data, oracle | 19, 2, 0, 1 | 21 of 22 |
| native, box | 16, 5, 1, 0 | 21 of 22 |
| native, oracle | 20, 2, 0, 0 | 22 of 22 |

The three failures:

| failure | route, priors | $`F_{\rm restarted} - F_{\rm truth+restarts}`$ | worst label |
| --- | --- | --- | --- |
| `gaia-5157__orbit` | data, box | +21.17 | $`v\sin i_B`$, 30.5 sigma |
| `gaia-5157__orbit` | data, oracle | +1.94 | $`v\sin i_B`$, 2.3 sigma |
| `gaia-1717__oracle` | native, box | +3.26 | $`T_{{\rm eff},B}`$, 5.1 sigma |

The two data-route failures are one mechanism, a basin in which the secondary's light has collapsed
(to 0.0016 under the box priors and 0.037 under the oracle ones, against an injected 0.396) and its
labels have gone to their bounds: with no light a
component has no labels, so no one-dimensional restart in its temperature, gravity or rotation can
lower the objective, and the escape needs the radius ratio to move with them. They occur on the one
product whose archived orbit does not describe the data. A joint restart was therefore tried there
(`estimator.joint_scan`: the objective on a grid of six radius ratios, five rotations and every other
temperature node at $`\log g`$ of 3, 4 and 5 for the fainter component, 630 points, and LM from the
best three distinct ones, run in every round and from the truth start alike). Under the oracle priors
it escapes: the restarted best falls by 13.1 to within 0.16 of the truth start's, the secondary's
light rises from 0.037 to 0.274 against an injected 0.396, and the criterion still fails, on the
primary's temperature at 0.47 sigma in a valley flat to 0.2 in chi-square. Under the box priors all
three joint starts return to the collapsed basin, 21 above the solution reached from the truth,
which itself puts the primary on the 7000 K bound. The native failure is the native objective's
multimodality on a secondary with 7 percent of the light, where the restarted best had put the
secondary on two bounds (7000 K, $`\log g = 5`$); the joint restart finds the truth start's solution
(4732 K, $`\log g = 3.83`$) in its first round, lower by 3.26, and the criterion then holds. On the five converged cells of these two products, and on all four of `gaia-5157__oracle`, adding the joint restart changed no optimum by more than 0.01 in chi-square. The restarts that mattered
were almost all in rotation: over the 88 cells a $`(T_{\rm eff}, \log g)`$ restart lowered an optimum
by more than 1 in chi-square twice, both on `gaia-5157__orbit` native (18.7 and 5.1), and a rotation
restart fourteen times, three on the data route (by 1.67, 2.30 and 6.33) and eleven on the native
route (by up to 584). The design without the rotation scan would have left more than 1 in
chi-square on ten cells.

**Converged, the data route halves the native route's temperature error and cuts its gravity,
metallicity, rotation and light errors by factors of 2.5 to 4.** Over the twelve products under the
box priors, as median absolute error with the 16th to 84th percentile range:

| label | set | data | native |
| --- | --- | --- | --- |
| $`T_{\rm eff}`$ (K) | primaries | 47.8 [11.9, 90.6] | 105.4 [55.4, 138.1] |
| | secondaries | 83.9 [17.1, 161.3] | 133.1 [35.3, 867.8] |
| | all | 60.2 [11.8, 112.4] | 118.6 [36.4, 242.5] |
| $`\log g`$ (dex) | primaries | 0.040 [0.011, 0.108] | 0.060 [0.026, 0.217] |
| | secondaries | 0.134 [0.010, 0.418] | 0.692 [0.141, 1.583] |
| | all | 0.062 [0.010, 0.236] | 0.180 [0.035, 1.104] |
| [M/H] (dex) | system | 0.006 [0.004, 0.031] | 0.024 [0.005, 0.045] |
| $`v\sin i`$ (km/s) | primaries | 2.68 [1.64, 4.47] | 6.85 [2.97, 9.42] |
| | secondaries | 4.06 [0.94, 9.01] | 7.77 [2.48, 16.81] |
| | all | 2.69 [1.27, 6.69] | 6.89 [2.40, 14.16] |
| light fraction | secondary | 0.0124 [0.0008, 0.0236] | 0.0350 [0.0068, 0.0780] |

The data route's temperature error is inside the 3 percent that `docs/math.md` §9.6 asks of a
template on 21 of 24 components (inside 2 percent on 20), and its [M/H] is within 0.031 dex on ten
of twelve systems. The native route's worst failures are on the faint secondaries. Converged, it
puts $`\log g_B`$ on the box's lower bound of 3.0 on `mixed-0008` (both tiers), `mixed-0015` and
`gaia-5157`, where the data route returns 4.594 against an injected 4.594 on `mixed-0008` oracle,
4.455 on its orbit tier, and the upper bound 5.0 against 4.580 on `mixed-0015`; and it collapses
the secondary's light on `mixed-0015` to 0.0059 against a true 0.0731, on `mixed-0017` to 0.121
against 0.233 and on `mixed-0008` to 0.085 against 0.112. Neither was traced further. The native
route is also the multimodal one: its best designed candidate was within 1 in chi-square of its
restarted best on 35 of 44 cells over the whole sample, against 41 of 44 for the data route.
Under the oracle priors the picture is the same, and the bounds hold ten native gravities against
three on the data route:

| label | set | data | native |
| --- | --- | --- | --- |
| $`T_{\rm eff}`$ (K) | primaries | 47.5 [11.9, 98.5] | 98.8 [56.9, 139.2] |
| | secondaries | 84.1 [17.1, 167.3] | 122.8 [42.3, 237.3] |
| | all | 60.3 [11.8, 127.9] | 106.0 [43.9, 185.8] |
| $`\log g`$ (dex) | primaries | 0.032 [0.010, 0.108] | 0.057 [0.026, 0.219] |
| | secondaries | 0.134 [0.010, 0.300] | 0.300 [0.141, 0.300] |
| | all | 0.062 [0.009, 0.236] | 0.182 [0.032, 0.300] |
| [M/H] (dex) | system | 0.007 [0.004, 0.031] | 0.022 [0.005, 0.030] |
| $`v\sin i`$ (km/s) | primaries | 2.68 [1.64, 4.46] | 6.84 [2.97, 9.65] |
| | secondaries | 4.06 [0.97, 9.01] | 7.72 [2.48, 14.68] |
| | all | 2.69 [1.29, 6.65] | 6.84 [2.40, 13.61] |
| light fraction | secondary | 0.0124 [0.0008, 0.0237] | 0.0310 [0.0071, 0.0776] |

**The temperatures of `d65_label_likelihoods.md` measured the warm start, and its light
fractions did not measure the likelihood either.** Under the oracle priors of that note, beside
WP-AB's native fits and WP-AD's quadrature-operator data route, both 80-step L-BFGS runs from the
same node scan:

| label | set | native, 80 steps | data, 80 steps | native, converged | data, converged |
| --- | --- | --- | --- | --- | --- |
| $`T_{\rm eff}`$ (K) | primaries | 149.8 | 84.3 | 98.8 | 47.5 |
| | secondaries | 225.6 | 226.0 | 122.8 | 84.1 |
| | all | 185.7 [78.6, 261.5] | 165.6 [38.7, 252.4] | 106.0 [43.9, 185.8] | 60.3 [11.8, 127.9] |
| $`\log g`$ (dex) | all | 0.185 | 0.167 | 0.182 | 0.062 |
| [M/H] (dex) | system | 0.057 | 0.022 | 0.022 | 0.007 |
| $`v\sin i`$ (km/s) | all | 7.65 | 3.57 | 6.84 | 2.69 |
| light fraction | secondary | 0.0277 [0.0070, 0.0736] | 0.0036 [0.0014, 0.0235] | 0.0310 [0.0071, 0.0776] | 0.0124 [0.0008, 0.0237] |

Convergence cuts the data route's temperature error from 165.6 to 60.3 K and its secondaries' from
226.0 to 84.1 K, at an $`L_{\rm data}`$ lower than the 80-step fit's on every product, by 10 to 2689
(median 61). The light fraction goes the other way, from 0.0036 to 0.0124, per product in both
directions: the 80-step value was not the likelihood's answer, and at optima lower in the same
objective on every product the converged figure is the one to quote. The warm start is also poorer
in rotation than either note said: the default `scan_vsini` of `match_labels` is the $`v\sin i`$
prior's midpoint, a quarter and six tenths of its upper bound, which is 75, 37.5 and 90 km/s for the
benchmark's $`v\sin i \le 150`$ (and 150, 75 and 180 km/s at the pipeline's default bound of 300),
so every designed start of every product here began at 37.5 km/s or more, including components
injected at 0.6 to 10 km/s. LM leaves such a start in a few iterations; an 80-step L-BFGS need not.

**The data route's formal errors are close to calibrated for temperature, gravity and metallicity,
and too small by about three for rotation and light; the native route's are too small by three to
thirty-five.** The pull is the error over the formal sigma of $`2H^{-1}`$ at the restarted optimum,
over the twelve products under the box priors, excluding labels on a bound or on the rotation
plateau; a calibrated error has a 68th percentile of $`|{\rm pull}|`$ near 1.

| label | data: median, 68th percentile | native: median, 68th percentile |
| --- | --- | --- |
| $`T_{\rm eff}`$ | 1.11, 1.40 (24 labels) | 3.13, 4.46 (22) |
| $`\log g`$ | 0.77, 0.97 (21) | 2.32, 4.18 (19) |
| [M/H] | 0.80, 1.38 (12) | 2.45, 2.95 (12) |
| $`v\sin i`$ | 2.39, 3.52 (21) | 14.07, 19.10 (22) |
| light fraction | 2.01, 2.72 (12) | 19.92, 35.10 (12) |

The median formal sigmas of the data route are 27 and 81 K for the primaries' and secondaries'
temperatures, 0.066 and 0.098 dex for their gravities, 0.70 and 2.27 km/s for their rotation and
0.0041 for the light fraction; the native route quotes 21 and 43 K, 0.040 and 0.075 dex, 0.29 and
0.89 km/s and 0.0016 while erring by more. On the nine orbit-tier products that did not exchange
their components the data route's 68th percentiles are 1.27, 0.75, 1.12, 4.20 and 2.46 and the
native route's 4.88, 3.55, 2.74, 18.56 and 35.31. The data route's rotation and light pulls are
the two labels that the unmodelled smoothing measured below moves systematically, so they are not
a statement about the noise.

**At the survey S/N the data route's reduced chi-square is one, and the native optimum describes
the epoch spectra worse by a median 147 in chi-square.** Over the twelve products the restarted data
optimum has $`L_{\rm data}/n`$ with a median of 0.9979 (0.9745 to 1.0213), below the injected
components' 1.0004 (0.9757 to 1.1770) on every product; the native optimum evaluated under
$`L_{\rm data}`$ gives 1.0018 (0.9935 to 1.0766), higher than the data optimum by 17 to 5994,
median 147. The native route's own diagonal chi-square against $`\hat d`$, over the pixels used,
is 0.39 (0.16 to 0.83), which read naively says that the archived posterior standard deviations
are too large by a factor of 1.6; the pulls say the opposite, which is what residuals correlated
over many pixels produce. The injected components, placed in the archive's component order, exceed a reduced chi-square of
1.05 only where the archived orbit does not describe the data or the archive exchanged the
components: `gaia-5157` (1.177) in the oracle tier, and in the orbit tier `gaia-1717` (1.376),
`gaia-5157` (1.171), `gaia-4762` (1.080) and the two exchanged archives, `mixed-0013` (1.421) and
`mixed-0017` (1.068).

**In the orbit tier, where the declared light came from the light stage, the data route measures
the light four times better than the declaration it was run with, on ten of eleven products.** The
light fraction of the secondary, box priors, restarted best; the third column is the injected
components' $`L_{\rm data}/n`$ through the archived orbit, a measure of whether that orbit describes
the data, with the data optimum's in parentheses:

| product | S/N | injected (optimum) $`L_{\rm data}/n`$ | declared | injected | data (formal sd) | native (formal sd) | declared error | data error | native error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mixed-0008 | 22.3 | 1.000 (0.994) | 0.1425 | 0.1119 | 0.0987 (0.0040) | 0.0483 (0.0013) | +0.0306 | -0.0132 | -0.0635 |
| mixed-0015 | 40.9 | 1.027 (1.001) | 0.1242 | 0.0731 | 0.0474 (0.0037) | 0.0051 (0.0009) | +0.0512 | -0.0256 | -0.0679 |
| mixed-0003 | 16.7 | 0.995 (0.993) | 0.4505 | 0.4755 | 0.4657 (0.0089) | 0.4655 (0.0075) | -0.0251 | -0.0098 | -0.0100 |
| mixed-0004 | 98.5 | 1.008 (0.998) | 0.0538 | 0.0520 | 0.0515 (0.0006) | 0.0434 (0.0003) | +0.0018 | -0.0005 | -0.0086 |
| mixed-0012 | 29.3 | 1.015 (1.001) | 0.4434 | 0.4860 | 0.4851 (0.0015) | 0.4899 (0.0017) | -0.0426 | -0.0009 | +0.0039 |
| mixed-0013, exchanged | 52.6 | 1.421 (1.002) | 0.3019 | 0.5354 | 0.4760 (0.0039) | 0.4658 (0.0092) | -0.2334 | -0.0593 | -0.0696 |
| mixed-0017, exchanged | 19.6 | 1.068 (0.975) | 0.7490 | 0.7668 | 0.7778 (0.0068) | 0.8782 (0.0019) | -0.0178 | +0.0110 | +0.1115 |
| gaia-1717 | 21.0 | 1.376 (1.018) | 0.3992 | 0.0673 | 0.0694 (0.0125) | 0.0225 (0.0022) | +0.3318 | +0.0021 | -0.0448 |
| gaia-5157 | 40.8 | 1.171 (1.037) | 0.1838 | 0.3960 | 0.0016 (0.0048) | 0.0148 (0.0030) | -0.2122 | -0.3944 | -0.3812 |
| gaia-4762 | 97.1 | 1.080 (1.015) | 0.4279 | 0.4589 | 0.4608 (0.0020) | 0.4562 (0.0015) | -0.0310 | +0.0019 | -0.0027 |
| gaia-5368 | 23.2 | 1.037 (1.000) | 0.3502 | 0.4669 | 0.4458 (0.0118) | 0.4243 (0.0052) | -0.1167 | -0.0211 | -0.0426 |

Two archives exchanged the components: on `mixed-0017` the component declared B is the injected
primary (the fitted temperatures are 1019 K and 981 K from their own stars and within 100 K of the
other's), and on `mixed-0013`, a twin with 115 K between the components, the fitted rotations are 66
and 74 km/s from their own stars and within 4 km/s of the other's. Both rows compare with the swapped
truth. As median absolute error with the 16th to 84th percentile range, the declared light misses by
0.0426 [0.0221, 0.2207], the data route by 0.0110 [0.0015, 0.0391] and the native route by 0.0448
[0.0067, 0.0863]. The data route improves on the declaration on ten of eleven products, including
three of the four the light stage got most wrong: `gaia-1717`, declared 0.399 against 0.067 and measured at
0.069; `mixed-0013`, 0.302 against 0.535 and measured at 0.476; `gaia-5368`, 0.350 against 0.467 and
measured at 0.446. That it does so on `gaia-1717`, whose archived orbit leaves the injected components at a
reduced chi-square of 1.38, and through an exchange on `mixed-0013`, is what G3 predicts:
$`L_{\rm data}`$ does not depend on the declared light at all. The failure is `gaia-5157`,
whose orbit is wrong in both tiers, and where the restarted data fit sits in a basin with the
secondary's light at 0.0016 and its three labels on their bounds (4000 K, $`\log g = 3`$,
150 km/s); it is a restart failure, discussed with the convergence table. Over the nine orbit-tier
products that did not exchange, the labels repeat the twelve products' result: the data route's
median errors are 39.5 K, 0.041 dex, 0.007 dex, 3.29 km/s and 0.0098 against the native route's
111.6 K, 0.093 dex, 0.021 dex, 6.33 km/s and 0.0426.

**The data route's remaining light-fraction error of about 0.01 is carried by its label errors and
by the archived orbit, not by the light estimator.** On the 18 products that neither exchanged nor
sit on the wrong `gaia-5157` orbit, the secondary's light is low on 15, and its error correlates
with the secondary's temperature error at $`r = 0.61`$. Holding the four labels of both components
at the injected values, with the frame velocities, the radius ratio and the offsets still free
(`fixed_labels.py`), cuts the median absolute light error over the twelve products from 0.0124 to
0.0031 [0.0017, 0.0138], and over the 20 products that did not exchange from 0.0108 to 0.0032;
with the operator also widened by 3.05 km/s it is 0.0040 and 0.0029, so the delivery smoothing is
not what moves the light. The errors that survive the true labels are on products whose archived
orbit is off. Replacing the operator's per-epoch shifts by the injected velocities as well
(`floor_orbit.py`), on the six products with the largest surviving errors, where the archived
shifts differ from the injected ones by a median 0.3 to 9.5 km/s per component:

| product | archived shift error, median A / B (km/s) | labels free | labels injected, archived orbit | labels and orbit injected |
| --- | --- | --- | --- | --- |
| mixed-0015 oracle | 0.93 / 0.62 | -0.0262 | -0.0192 | -0.0013 |
| mixed-0015 orbit | 0.95 / 1.71 | -0.0256 | -0.0191 | -0.0013 |
| mixed-0013 oracle | 2.08 / 0.32 | -0.0118 | -0.0121 | -0.0028 |
| mixed-0008 orbit | 0.91 / 7.45 | -0.0132 | -0.0106 | +0.0063 |
| gaia-1717 oracle | 4.46 / 5.30 | -0.0137 | -0.0049 | +0.0017 |
| gaia-5157 oracle | 9.52 / 0.47 | -0.3340 | -0.1717 | -0.0012 |

With both the labels and the orbit right, the light error is at most 0.0063 on all six, and on
`gaia-5157`, whose light no route recovered through the archived orbit, it is 0.3948 against 0.3960.
The native route behaves differently: at the injected labels its light error is 0.0364 over the
twelve against 0.0350 with free labels, so its failure is the comparison's, not the labels'.

**The rotation that the data route over-estimates is the smoothing the archive's resampling adds,
and a width derived from the delivery chain accounts for it to a twentieth of a km/s.** Converged, the
data route still puts $`v\sin i`$ high on 17 of 24 components, by a signed median of +2.14 km/s,
the residual that `d65_label_likelihoods.md` measured at +2.9 km/s through unconverged fits and left
untraced. `gaia.simulate_rvs_dataset`
shifts the components on a 2 km/s grid, convolves the composite with $`\sigma_q`$, integrates it
over detector pixels of $`\Delta_{\rm det} = 0.245`$ A, adds the noise, and `gaia.deliver` then
interpolates each epoch linearly onto the 0.25 A product grid. The operator $`A`$ shifts on its own
3 km/s grid, convolves, and integrates over the 0.25 A product pixels; it has no counterpart of the
delivery interpolation. A linear interpolation at fractional position $`t`$ is a two-tap kernel of
zero mean and variance $`t(1-t)\,\Delta_{\rm det}^2`$, and since the 0.25 and 0.245 A grids beat
with a period of 49 pixels (12 A), $`t`$ is uniform over the band and the mean variance is
$`\Delta_{\rm det}^2/6`$, which at the band centre, where 1 A is 34.9 km/s, is
$`(3.49\ {\rm km\,s^{-1}})^2`$. Against it the operator carries more of its own smoothing: its 3 km/s
grid's pixel representation and shift interpolation exceed the simulator's 2 km/s ones by
$`2(3^2-2^2)/12 = 0.83`$ and $`(3^2-2^2)/6 = 0.83\ {\rm km^2\,s^{-2}}`$, its 0.25 A box exceeds the
0.245 A detector box by $`0.25\ {\rm km^2\,s^{-2}}`$, and the label model's own shift by $`\gamma`$
adds a mean $`3^2/6 = 1.5\ {\rm km^2\,s^{-2}}`$, so that

```math
\sigma_{\rm extra}^2 \simeq \frac{\Delta_{\rm det}^2}{6} - 0.83 - 0.83 - 0.25 - 1.5 = 8.80\ {\rm km^2\,s^{-2}},
\qquad \sigma_{\rm extra} \simeq 2.97\ {\rm km\,s^{-1}} .
```

Measured without noise and without orbit error, by delivering the simulation's own noiseless
detector fluxes exactly as `gaia.deliver` does and scanning the width of an operator that carries the
injected velocities and light fractions (`vsini_width.py`), the best width exceeds $`\sigma_q`$ in
quadrature by 3.24 to 3.34 km/s with $`\gamma`$ in the operator's shifts, and by 3.01 to 3.20 km/s,
median 3.06, with $`\gamma`$ in the rows where the label model puts it, on `mixed-0008`, `mixed-0017`,
`mixed-0013` and `gaia-4762`; at that width the noiseless residual's reduced chi-square falls by a
factor of 3.2 to 5.1, without reaching zero, since a stationary Gaussian cannot represent a
phase-dependent interpolation exactly. The limb-darkened rotation profile with $`\epsilon = 0.6`$ has a variance of $`0.225\,v^2`$,
so a fit that absorbs $`\sigma_{\rm extra} = 3.05`$ km/s into rotation should return
$`v_{\rm fit}^2 = v^2 + \sigma_{\rm extra}^2/0.225`$, that is $`\sqrt{v^2 + 41.3}`$: 7.1 km/s for an
injected 3 km/s and 60.3 km/s for 60. Over the twelve products the fitted rotation exceeds this
prediction by a median of +0.05 km/s (median absolute difference 1.38 km/s) against +2.14 km/s over
the injected value; the native route, which compares with the posterior-smoothed $`\hat d`$,
exceeds it by +4.76 km/s. On the nine orbit-tier products that did not exchange, whose archived
orbits are less accurate, the data route exceeds the prediction by +0.79 km/s. Widening the
operator by the derived width, to $`\sqrt{\sigma_q^2 + 3.05^2} = 9.557`$ km/s, and rerunning the
estimator on all 22 products removes the bias: over the twelve products under the box priors the
signed median rotation error falls from +2.14 to +0.13 km/s (high on 14 of 24 components rather
than 17) and the median absolute error from 2.69 to 1.43 km/s, while the temperature (60.2 against
60.4 K), gravity (0.062 against 0.063 dex), metallicity (0.006 dex both) and light fraction (0.0124
against 0.0111) barely move, and the oracle priors give the same. The injected components' reduced
chi-square falls by 0.009 to 0.010 on the two products at S/N 97 to 99 (`mixed-0004`, 1.0090 to
0.9989; `gaia-4762`, 1.0253 to 1.0162) and on `gaia-5157`, and by at most 0.0012 on the others, so
at high S/N the delivery smoothing is most of what remained of the injected components' excess once
the library width was removed. For the benchmark itself this means that the delivered epochs
carry an effective line-spread width of $`\sqrt{11.07^2 + 3.05^2} = 11.48`$ km/s, $`R \approx 11{,}090`$,
where the disentangling declares 11.07 km/s. To the extent that it is stationary, a width is absorbed by the
free component spectra (`docs/math.md` §1.3), so the archived components carry the extra 3 km/s, and every consumer
that compares them with an unbroadened template, the native label route and the TODCOR templates
built from its labels among them, inherits it; this is reported, not changed.

**At the pipeline's own budget of 500 steps, `run_map` has not converged at S/N 20 to 22, and what
separates it from the likelihood is the budget; the log-Jacobian moves a converged fit by at most 0.4
formal sigma, except at a rotation bound.** On the four products above, box priors, the shipped
`match_labels` was run with its likelihood replaced by $`-\tfrac12 L_{\rm data}`$ on the quadrature
operator, no jitter sites, and 500 L-BFGS steps from each of its four candidates (69 to 72 s), and LM
was run to convergence on

```math
F_J(\phi) = F_{\rm data}(\phi) - 2 \sum_{k\,\in\,\rm bounded} \log \frac{(\phi_k - a_k)(b_k - \phi_k)}{b_k - a_k},
```

twice the potential `run_map` minimises in the unconstrained space. numpyro's potential minus
$`F_J/2`$ is the same constant, 30.684190625, to $`3 \times 10^{-10}`$ at every finite point on all four
products (the one infinite value is the unconstrained image of $`v\sin i = 0`$), so $`F_J`$ is that potential. Label differences in units of the converged data route's
formal sigma:

| product | S/N | `run_map` (500 steps) minus LM, largest | $`F_J`$ above its converged value at `run_map`'s answer | LM on $`F_J`$ minus LM, largest |
| --- | --- | --- | --- | --- |
| mixed-0017 | 19.6 | $`\log g_B`$ -1.01 dex (-5.0), $`T_{{\rm eff},B}`$ -95 K (-1.4), [M/H] (-1.6), $`v\sin i_B`$ +9.8 km/s | 30.5 | $`v\sin i_B`$ 0.0 to 5.7 km/s (on the plateau), light +0.0015 (+0.25) |
| mixed-0008 | 22.3 | $`\log g_B`$ -0.74 dex (-2.6), $`T_{{\rm eff},B}`$ -218 K (-2.1), light (-0.97) | 7.1 | $`v\sin i_B`$ +1.85 km/s (+0.40), $`\log g_B`$ (-0.34) |
| mixed-0013 | 52.6 | [M/H] (-0.10) | 0.05 | $`\log g_B`$ (-0.14) |
| gaia-4762 | 97.1 | $`\log g_A`$ (+0.12) | 0.05 | $`v\sin i_A`$ (+0.09) |

On the two brighter products `run_map` at 500 steps agrees with the constrained optimum to 0.12
sigma. On the two at S/N 20 to 22 it stops where its own potential is still 7.1 and 30.5 above the value
LM reaches on the same potential in 3 to 4 iterations from that point, and the faint secondary's
gravity is 0.74 and 1.01 dex from the converged answer, which is the unconverged fit of the earlier
notes at a larger budget. The converged log-Jacobian pull is at most 0.14 sigma on every label of
the bright pair and 0.40 sigma on `mixed-0008`; its one large effect is the barrier at
$`v\sin i = 0`$, which lifts `mixed-0017` B off the rotation plateau to 5.7 km/s against an injected
2.5. Under the box priors on the data route, then, the prior-centre pull that
`d65_label_likelihoods.md` measured on the hierarchical route is small; it is the step budget, the
rotation plateau and the rotation bound that decide what the shipped optimiser returns.

**A converged label fit costs about a second once the statistics exist, and the statistics cost
six.** Per product, $`h`$, $`z^\top W z`$ and the band of $`G`$ take 6.2 s (4.6 to 7.1), of which the
band assembly is 4.7 s; the objective, gradient and Gauss-Newton matrix together take 0.015 s
(0.010 to 0.028) after a compile of 1.0 to 1.4 s for the first route, and an LM iteration including
its trial evaluations 0.018 s. The production estimator for one route and one prior configuration,
LM from the four designed candidates and the restart rounds, ran a median of 11 LM fits (8 to 25) in
1.1 s (0.6 to 8.3), of which the designed candidates alone took 0.5 s; the truth-started
diagnostics roughly double that. The dominant cost is the part of `match_labels` that was reused
unchanged, the node scan against $`\hat d`$, at 18 s per configuration (15 to 34). A whole product,
with both routes, both configurations, all diagnostics and the gates, took 73 s (57 to 110) and
peaked at 3.3 to 4.9 GiB of working set, on a machine that was also running the user's own jobs.
For comparison, WP-AD's 80-step L-BFGS fits on the data route took 53
to 73 s including the same node scan, so about 35 to 55 s of optimisation against about one here,
and did not converge.

**Recommendation for `match_labels`: give it the epoch-space comparison, and change how it is
optimised.** $`L_{\rm data}`$ converged returns temperatures a factor of two better than the native
comparison converged the same way (60 against 119 K), gravities, metallicities, rotations and light
fractions a factor of 2.5 to 4 better, a reduced chi-square that means something, and formal errors
that are close to right for three of the five labels, where the native route's are too small by 3
to 35. Five changes, none made here: (i) a comparison that accepts the disentangling's statistics,
$`h`$, $`z^\top W z`$ and the band of $`G`$ at the MAP, which `estimator.build_statistics` builds in
6 s from `forward.rhs`, `weighted_data_terms` and `band_block_tridiagonal` with a null prior; (ii) a
declared library resolving power on `StarLabels` or `match_labels`, from which the operator's width
$`\sqrt{\sigma_{\rm inst}^2 - \sigma_{\rm lib}^2}`$ follows, since `SpectralLibrary.meta["resolution"]`
already records it; (iii) the bounded Levenberg-Marquardt in the constrained space in place of
`run_map` for the label fit, which converges in four iterations at 0.02 s each, where `run_map` at the pipeline's 500 steps
had not converged at S/N 20 to 22 and left a faint secondary's gravity 0.7 to 1.0 dex from its optimum; (iv)
restarts in every component's rotation, which is where the local minima are, with the faint
component's $`(T_{\rm eff}, \log g)`$ scan and a coarse joint restart in its radius ratio,
rotation, temperature and gravity, which resolved one of the three remaining failures and closed
the chi-square gap of a second, together with a flag, since no restart here recovers a secondary whose light has collapsed below 1 percent
with its labels on their bounds on a wrong orbit; and (v) a `scan_vsini` default that
reaches slow rotation, since the present one starts every fit at 37.5 km/s or more. The native
comparison should remain only as a diagnostic of the disentangled spectra. The rotation and the
light fraction should be quoted with an error inflated by about three, or with a spread from refits,
until a residual model carries what the pulls show.

**Recommendation for the pipeline's light measurement: report the data route's light fraction,
measured after the disentangling, and keep the correlation stage only to declare a starting light.**
In the orbit tier the label fit on $`L_{\rm data}`$ cuts the median light-fraction error of the
declaration from 0.043 to 0.011, improves on it on ten of eleven products, and by G3 cannot inherit
the declaration's error, because the objective does not depend on the declared light. The
disentangling still needs a declared light to scale its smoothness prior, and a declaration that is
wrong by 0.33 on `gaia-1717` did not prevent the measurement, so a second disentangling pass at the
measured light is not required for the light itself; whether it improves the component spectra or
the orbit was not measured. The benchmark's light metric should be scored on this measurement, and
its formal error inflated by about three, since the formal sigma does not include the orbit and
label errors that carry the remaining 0.01.

**What is left.** First, the light fraction's remaining error of about 0.01 is the label and orbit
errors propagated through a light-temperature degeneracy that the radius ratio does not break; with
both right it falls to a few thousandths, so the estimator that would reach that floor fits the
labels, the light and the orbit together on $`L_{\rm data}`$, which was not built. Second, the formal errors of the
rotation and the light are too small by about three, and neither `refit_draws` nor any
residual-covariance model was run here to supply a realistic one; for $`L_{\rm data}`$ the natural
spread is a refit over draws of the epoch noise rather than over draws of $`\hat d`$. Third, the
light-collapse basin was found on one product, and the joint restart was tried only on the three
products with a failed cell; a sample with more faint secondaries is needed before its grid is
trusted, and before the collapse is turned from a restart into a flag. Fourth, the
labels inherit the archived orbit in the same way, most visibly where the injected components fit
the epochs at a reduced chi-square above 1.05 through it (`gaia-5157` in both tiers, `gaia-1717`
and `gaia-4762` in the orbit tier); the Gauss-Newton structure here extends to the orbit parameters
through `jax.jacfwd` of the operator's shifts, at the cost of rebuilding $`h`$ and $`G`$ as the
shifts move, which the fixed-orbit estimator avoids. Fifth, two of the eleven orbit-tier
archives exchanged their components, and the label fit reports the labels of whichever star the
archive put in each slot; on a twin the rotation, which exchanged cleanly on `mixed-0013`, is the
label that could tell the pipeline so, and nothing checks it now. Sixth, the sample is two-component, on the RVS band at
$`R = 11{,}500`$ with one library, and the delivery smoothing measured here is the simulator's;
whether the DR4 archive interpolates the same way is not known (`gaia.deliver` records that the
DR3 documentation does not state the method).
