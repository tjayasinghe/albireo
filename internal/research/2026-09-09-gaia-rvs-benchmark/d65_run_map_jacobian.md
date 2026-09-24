# Does the log-Jacobian of `run_map` bias the disentangling's own MAP? (D65, 2026-09-16)

`d65_label_likelihoods.md` found that numpyro's change of variables pulls a weakly constrained
stellar label towards the middle of its prior, and `d65_converged_labels.md` bounded that pull at
0.4 formal sigma on converged label fits, except at a rotation bound. This note asks the same
question of the disentangling. `Disentangler.fit` hands the orbital sites and the smoothness
hyperparameters to `albireo.inference.run_map`, which minimises numpyro's potential in the
unconstrained space, so the point it returns is the mode of the density of the unconstrained
coordinates rather than the MAP in the declared ones. The sample is the third benchmark run, 46
products in `gaia5` and 59 in `field5`, each rebuilt at its archived MAP with WP-AB's re-simulation
and declaration rebuild, extended to the priors each tier actually used. Scripts, per-product JSON,
logs and tables are in the session scratchpad `wp-aj/`: `jac.py` is the measurement, `table.py`
and its output `table.txt` the tables, `inventory.py` the census of sampled sites.

**For a bounded site, `run_map`'s optimum satisfies a shifted stationarity condition, and the shift
follows from the Hessian in the declared coordinates.** For a site declared `Between(a, b)`, numpyro
samples $`z`$ with $`x = a + (b - a)\,u`$ and $`u = \sigma(z)`$, and the potential it minimises is

```math
U_z(z) = U_x\big(x(z)\big) - \sum_{i\,\in\,\rm bounded} \Big[\log(b_i - a_i) + \log u_i (1 - u_i)\Big],
```

with $`U_x(x) = -\log p(x \mid \text{data})`$ up to a constant. Differentiating,

```math
\frac{\partial U_z}{\partial z_i} = \frac{\partial U_x}{\partial x_i}\,(b_i - a_i)\,u_i(1 - u_i) - (1 - 2u_i),
```

so at a stationary point of $`U_z`$ the constrained gradient is not zero but

```math
\frac{\partial U_x}{\partial x_i} = g_{J,i} \equiv \frac{1 - 2u_i}{(b_i - a_i)\,u_i(1 - u_i)}
```

on every bounded site and zero elsewhere. Expanding about the constrained MAP $`x^*`$, where the
gradient of $`U_x`$ vanishes, $`H_x (x_J - x^*) \simeq g_J`$, so that

```math
\delta \equiv x_J - x^* \simeq H_x^{-1} g_J, \qquad {\rm pull}_i = \frac{\delta_i}{\sqrt{(H_x^{-1})_{ii}}},
\qquad U_x(x_J) - U_x(x^*) \simeq \tfrac12\, g_J^\top H_x^{-1} g_J ,
```

with $`H_x`$ the Hessian of $`U_x`$ in the declared coordinates. An isolated site below its midpoint
has $`g_J > 0`$ and $`\delta > 0`$: the archived answer sits above the constrained MAP, towards the
middle of the prior, and $`g_J`$ grows as $`1/u`$ near the lower bound. For a semi-amplitude declared
`Between(2, 250)` at 50 km/s, $`g_J = 0.016`$ per km/s; at 4 km/s it is 0.46 per km/s. The pulls
quoted below are $`\delta/\sigma`$ at the archived point, with $`\delta`$ the shift the Jacobian
causes; removing it moves each estimate by $`-\delta`$.

**Only 72 of the 105 products carry a bounded site, and the oracle tier carries none.** The sites
sampled by each tier, read from `fit.npz` (a site whose unconstrained value differs from its
constrained one has a transform) and from `facade._make_specs`, `facade._ecc_sites`,
`benchmark.build_star` and the pipeline's bootstrap route:

| tier | products | bounded sites (sigmoid) | unbounded sites |
| --- | --- | --- | --- |
| oracle | 33 | none | $`K_A, K_B, P, t_{\rm conj}`$ (Normal); $`\log\tau, \log\eta`$ (Normal); $`e, \omega`$ fixed |
| eclipsing | 6 | $`K_A, K_B \in (2, 250)`$; $`\sqrt e\cos\omega, \sqrt e\sin\omega \in \pm 0.949`$ | $`P, t_{\rm conj}`$ (Normal); $`\log\tau, \log\eta`$ |
| orbit | 33 | $`K_A, K_B`$; $`\sqrt e\cos\omega, \sqrt e\sin\omega`$; $`t_{\rm conj}`$ on a window of one period | $`P`$ (Normal); $`\log\tau, \log\eta`$ |
| blind | 33 | $`P`$ within 3 percent of the bootstrap period; $`\sqrt e\cos\omega, \sqrt e\sin\omega`$ ($`K`$ as well on `mixed-0004`) | $`K, t_{\rm conj}`$ (Normal, from the bootstrap); $`\log\tau, \log\eta`$ |

The AR(1) coefficient is declared per epoch and not sampled, and no light, LSF, response or nebular
site is sampled in this run, so the table is complete. On the oracle tier $`U_z`$ and $`U_x`$ differ
by a constant, and the archived MAP is the constrained MAP up to non-convergence. That includes
`gaia-53290511099783296__oracle`, the planned floor case, whose $`K_B = 2.21`$ km/s sits under a
Normal(35.5, 5.3) prior with no floor at all: that collapse owes nothing to the Jacobian, and
removing the Jacobian changes nothing there. The 2 km/s floor exists only where $`K`$ is a range,
and the floor case is taken up below on the orbit tier.

**The rebuild reproduces every archive, and the Jacobian bookkeeping closes to rounding.** On all 72
products the re-simulation reproduces the manifest record, the marginal at the archived MAP
reproduces `spectrum_*.txt` to at most 5.0e-10, and numpyro's transforms map the archived
unconstrained values onto the archived constrained ones exactly, except $`t_{\rm conj}`$ at up to
4.7e-10 d. The orbit tier's conjunction window is not recorded in `result.json`: `fit()` centres a
window one scan period wide on the scanned start. Its lower edge was recovered from the archived
pair $`(z, x)`$ with the width set to the declared period, and on all 33 orbit-tier products the
recovered centre lies on a trial of the scans, an integer or half-integer multiple of $`P/41`$ from
the first epoch, which checks the width. The blind tier's conjunction prior is
Normal($`t_{\rm boot}`$, $`0.05\,P_{\rm boot}`$) with both values read from `result.json`. numpyro's
potential at the archived $`z`$ differs from `disentangling.potential` by -1.63 to +4.4e-4 nats,
median -2.6e-5, because `MAPResult.potential` is evaluated at the convergence check, one accepted
step before `params`, as its docstring says: on fits still descending the archived point is lower,
and on the 32 fits whose last step changed nothing the two agree to 1e-5. $`U_z + \sum \log|dx/dz|`$
equals $`-\log p`$ from `numpyro.infer.util.log_density` at the constrained values to 5.5e-8 nats
(2.6e-12 relative). The chain rule
$`\partial U_x/\partial x_i = (\partial U_z/\partial z_i + \partial_z \log|dx/dz|)/(dx/dz)`$, with both
gradients by autodiff through the two independent paths, holds to below 1e-8 relative on 58 of the 67
products where it was evaluated at $`x(z)`$, and to 2.4e-5 relative at worst, on hyperparameter
components whose gradient is itself below 1e-5; $`\partial_z \log|dx/dz| = 1 - 2u`$ holds to 5.6e-16.
That fixes the sign and the factor of $`g_J`$.

**The potential is only piecewise smooth in the orbital sites, and the Hessian was therefore taken by
central differences of the gradient at a tenth of a sigma.** `laplace_inverse_mass` builds a
reverse-over-reverse Hessian in $`z`$; the same construction on $`-\log p`$ in $`x`$, one VJP of the
gradient per row, was run on 37 products and gated against central differences of the gradient with
a step of 1e-2 of the conditional sigma. On the three `gaia-17170287112790656` products the two agree
to 5e-8 to 4e-6 in the columns of $`K_A`$, $`P`$ and $`\log\tau`$. On `mixed-0015__orbit` the same
differences disagree with autodiff by up to 175 in correlation units, in the columns of $`K_B`$, $`P`$,
the eccentricity sites and $`t_{\rm conj}`$ only, while the hyperparameter columns agree to 2e-5. The
shifts enter the operator through linear interpolation, so the gradient jumps whenever an epoch's
shift crosses a model pixel, a small difference step that straddles a crossing measures the jump, and
autodiff returns the curvature inside one interpolation cell. A step of 0.1 conditional sigma (three
passes, capped at the prior scale or 5 percent of the box) averages over the cells. That Hessian
agrees with autodiff to 1e-3 in pull on 14 of the 22 products where both are positive definite and
to 0.02 on 21, and differs where a faint semi-amplitude is sampled by few pixel crossings: the
$`K_B`$ curvature is 2.4 times the autodiff value on `mixed-0015__orbit`, where it moves the pull from
0.215 to 0.143, and of the opposite sign on `gaia-17170287112790656__orbit`. It also peaks at 2.7 GiB
of working set (3.6 at most) against 3.7 to 10.9 GiB (median 7.5) for the reverse-over-reverse rows,
which on a machine shared with the user's job decided it. Every first-order number below uses it, on
67 products, and the autodiff Hessian on the remaining five, all positive definite with pulls below
0.014.

**To first order the Jacobian moves each site of the disentangling's MAP by a median 0.002 to 0.006
formal sigma, and no site by more than 0.05 sigma on 39 of the 43 products where the shift is
defined.** It is defined at a local
minimum: $`H_x`$ is positive definite on 43 of the 72 products away from the eccentricity disk, 9
fits sit on that disk (below), and 20 more are indefinite: ten near $`e = 0`$ ($`e < 0.05`$), where
$`(\sqrt e\cos\omega, \sqrt e\sin\omega)`$ is singular, nine on fits that lost the orbit (one of them
also near-circular), and the two recovered `gaia-17170287112790656` fits of a secondary with 7 percent
of the light. The marginal
pull over the 43, by site class, with $`e`$ by the delta method:

| site | n | median $`\lvert{\rm pull}\rvert`$ | 90th percentile | largest (product) | above 0.05 | above 0.2 | median signed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $`K`$ of the brighter star | 24 | 0.0052 | 0.038 | 0.494 (`mixed-0019__orbit`) | 2 | 2 | +0.0015 |
| $`K`$ of the fainter star | 24 | 0.0051 | 0.073 | 0.749 (`gaia-53290511099783296__orbit`) | 4 | 1 | +0.0046 |
| $`\sqrt e\cos\omega`$ | 43 | 0.0024 | 0.022 | 0.481 (`gaia-5329__orbit`) | 2 | 2 | -0.0009 |
| $`\sqrt e\sin\omega`$ | 43 | 0.0039 | 0.033 | 0.464 (`gaia-5329__orbit`) | 2 | 1 | +0.0000 |
| $`e`$ | 43 | 0.0063 | 0.033 | 0.462 (`gaia-5329__orbit`) | 2 | 2 | -0.0047 |
| $`t_{\rm conj}`$ (orbit tier) | 22 | 0.0053 | 0.047 | 0.414 (`gaia-5329__orbit`) | 2 | 2 | +0.0014 |
| $`P`$ (blind tier) | 20 | 0.0016 | 0.011 | 0.021 (`gaia-37608387208382848__blind`) | 0 | 0 | -0.0006 |

In physical units the median shift is 0.003 km/s in $`K`$ (90th percentile 0.2 km/s), 3e-5 in $`e`$,
4e-4 d in $`t_{\rm conj}`$ and 2e-7 d in $`P`$. The semi-amplitude shifts point towards the middle of
the prior on 42 of 48 sites; the eccentricity shifts towards $`e = 0`$ on the median. The conditional
pull $`g_{J,i}/\sqrt{H_{ii}}`$, which needs only a positive diagonal and so covers all 72 products, has
a median of 0.004 to 0.006 in $`K`$, 0.0015 to 0.0034 in the eccentricity sites, 8e-5 in
$`t_{\rm conj}`$ and 1e-7 in $`P`$, and exceeds 0.2 on six products. On the 37 positive-definite products whose
orbit was recovered (period within 1 percent and both semi-amplitudes within 30 percent of the
injected values) the largest marginal pull is 0.143, on `mixed-0015__orbit`, and the median of the
per-product largest is 0.007. For scale, the Newton step from each archived point to `run_map`'s own
optimum has a median of 0.09 sigma on the same 43 products and exceeds 1 sigma on six: the archived
fits are further from their own optimum than from the constrained one.

**Every first-order shift above 0.2 sigma is on a product with a semi-amplitude in the lowest 5
percent of its prior range, and every such product has lost the orbit.** Eight semi-amplitudes on
seven orbit-tier products have $`u < 0.05`$, that is, lie below 14.4 km/s; the shifts above 0.2 sigma
on the other sites of `gaia-53290511099783296__orbit` in the table above come through their
correlation with its $`K_B`$:

| product | site | archived $`K`$ (km/s) | injected (km/s) | $`u`$ | formal sigma (km/s) | $`\delta`$ (km/s) | pull | $`H_x`$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mixed-0019__orbit` | $`K_A`$ | 4.15 | 21.6 | 0.0087 | 1.05 | +0.52 | 0.494 | positive definite |
| `mixed-0005__orbit` | $`K_A`$ | 4.11 | 19.4 | 0.0085 | 0.96 | +0.50 | 0.517 | indefinite |
| `mixed-0010__orbit` | $`K_B`$ | 4.64 | 22.0 | 0.0106 | 1.90 | +1.31 | 0.690 | indefinite |
| `gaia-7289006178185856__orbit` | $`K_B`$ | 7.88 | 132.2 | 0.024 | 3.91 | +2.71 | 0.693 | indefinite |
| `gaia-53290511099783296__orbit` | $`K_B`$ | 13.17 | 35.5 | 0.045 | 8.62 | +6.46 | 0.749 | positive definite |
| `mixed-0015__orbit` | $`K_A`$ | 10.13 | 12.0 | 0.033 | 0.18 | +0.006 | 0.033 | positive definite |
| `gaia-51574864941234048__orbit` | $`K_A`$, $`K_B`$ | 2.0008, 2.088 | 13.4, 15.1 | 3e-6, 4e-4 | | | (on the disk) | indefinite |

`mixed-0015__orbit` is the one recovered orbit among them, and its largest pull, 0.143, is below 0.2:
it sits on its fainter star's $`K_B = 18.47`$ km/s ($`u = 0.066`$, injected 21.9) and is the largest
of any recovered product. The others
are the failures of the orbit tier: a semi-amplitude collapsed towards the floor, 2.6 to 32 formal
sigma below the truth.

**Exact refits reproduce the first-order shift wherever the archived fit is converged, and cannot
resolve it where it is not.** The refits run L-BFGS copied from `run_map` (optax `lbfgs` with its zoom
line search, `Disentangler.fit`'s tolerance $`\max(10^{-2}, 10^{-6} n_{\rm good})`$) from the archived
$`z`$, once on $`U_z`$ (the control) and once on $`U_z + \sum\log|dx/dz|`$ (the constrained objective),
with the same rules for both: 300 steps on the first seven pairs, and on the later runs also a stop
when the objective changes by less than 1e-6 nats over 20 steps or the parameters do not change for
10. Two further controls separate the objective from the optimiser: the control restarted from the
archived $`x`$ moved by +0.01 conditional sigma on every site (the noise floor), and on two products
300-step continuations of both objectives from the control's end. The products are the seven with a
semi-amplitude near the floor, plus a twin (`gaia-45354171749336960__orbit`, $`e = 0.002`$), an
eclipsing product (`gaia-53680017390963072__eclipsing`), a near-circular blind product
(`gaia-40041022325608704__blind`, $`e = 0.013`$) and a secondary with 7 percent of the light
(`gaia-17170287112790656__orbit`). Largest difference over the bounded sites, in formal sigma at the
archived point:

| product | Newton step to own optimum | first order | constrained minus control (site) | noise floor | $`U_x`$: constrained minus control (nats) | $`K`$ nearest the floor or most moved: control, constrained (km/s) |
| --- | --- | --- | --- | --- | --- | --- |
| `gaia-53680017390963072__eclipsing` | 7e-4 | 0.013 | 0.012 ($`\sqrt e\cos\omega`$) | 0.001 | -1e-4 | 97.197, 97.186 |
| `mixed-0019__orbit` | 3e-4 | 0.494 | 0.478 ($`K_A`$) | 0.000 | -0.128 | 4.15, 3.65 |
| `mixed-0005__orbit` | 0.006 | 0.517 | 0.604 ($`K_A`$) | 0.000 | -0.172 | 4.12, 3.54 |
| `mixed-0015__orbit` | 0.024 | 0.143 | 0.230 ($`K_B`$) | 0.000 | -0.012 | 18.47, 17.89 |
| `mixed-0010__orbit` | 0.04 | 0.690 | 1.116 ($`K_B`$) | 0.011 | -0.532 | 4.64, 2.52 |
| `gaia-40041022325608704__blind` | 0.23 | 0.009 | 0.031 ($`\sqrt e\sin\omega`$) | 0.007 | +0.001 | |
| `gaia-53290511099783296__orbit` | 8.9 | 0.749 | 0.295 ($`\sqrt e\sin\omega`$); continuation 0.107 ($`K_B`$) | 0.064 | -0.595; continuation -0.879 | 2.94, 2.17; continuation 2.94, 2.02 |
| `gaia-45354171749336960__orbit` | 0.23 | 0.001 | 0.307 ($`\sqrt e\sin\omega`$); continuation 0.054 | 0.401 | -0.070; continuation +0.010 | 75.72, 75.70 |
| `gaia-17170287112790656__orbit` | 17.8 | 0.145 | 2.810 ($`K_B`$) | 2.508 | +0.557 | 228.1, 194.6 |
| `gaia-7289006178185856__orbit` | 0.06 | 0.691 | 4.347 ($`K_A`$) | 1.725 | -6.617 | (192.9, 14.7), (249.9, 2.05) |
| `gaia-51574864941234048__orbit` | 3e3 | 450 (conditional) | 0 (frozen) | | 0 | 2.001, 2.001 |

On the five converged products, all but one with the fit on its own optimum to better than 0.03
sigma and a noise floor below 0.011 sigma, the site with the largest first-order shift moves in the
predicted direction by 0.92 to 1.62 times its size: 0.012 against 0.013 on the eclipsing product,
0.478 against 0.494 on `mixed-0019`, 0.604 against 0.517 on `mixed-0005`, and more than first order
where the semi-amplitude runs towards the floor across several pixel cells (0.230 against 0.143 on
`mixed-0015`, where the autodiff Hessian's 0.215 comes closer, and 1.12 against 0.69 on the indefinite
`mixed-0010`). The other sites follow through their correlations with it and agree with first order
to within 0.55 sigma, not always in sign; on the eclipsing product every site agrees to 1e-3. On the unconverged `gaia5` products
300 steps from a start moved by one hundredth of a sigma scatter by 0.40 sigma on the twin and 2.5
sigma on `gaia-1717`, as large as the pair's difference or larger, and by 1.7 sigma on `gaia-7289` in
a noise run that the later stopping rule ended after 30 steps; on the twin the control itself fell 25
nats below the archive. The continuations from the control's end
give the cleaner answer there: 0.054 sigma on the twin with the constrained run ending 0.010 nats
higher in its own objective than the control, which is noise, and 0.107 sigma on `gaia-5329`, where
the difference is real (0.88 nats) and is the semi-amplitude reaching the floor. The first-order
0.749 on `gaia-5329` was taken at an archived point 8.9 sigma from its own optimum; the control refit
moved $`K_B`$ from 13.2 to 2.94 km/s before the Jacobian mattered.

**Removing the Jacobian leaves the errors against the injected values unchanged to 0.11 sigma in the
median, except on the semi-amplitudes near the floor, which it moves away from the truth every
time.** To first order, over
the 43 positive-definite products, in formal sigma (with the Jacobian, as archived; without it,
archived minus $`\delta`$):

| site | n | median signed error, with | without | median $`\lvert{\rm error}\rvert`$, with | without | moved towards the truth |
| --- | --- | --- | --- | --- | --- | --- |
| $`K`$ brighter | 24 | -0.619 | -0.624 | 1.446 | 1.448 | 12 |
| $`K`$ fainter | 24 | -0.199 | -0.206 | 1.205 | 1.163 | 9 |
| $`e`$ | 43 | +0.307 | +0.410 | 1.204 | 1.192 | 15 |
| $`t_{\rm conj}`$ | 22 | +0.311 | +0.324 | 0.773 | 0.668 | 11 |
| $`P`$ | 20 | -1.058 | -1.060 | 1.441 | 1.442 | 10 |

No median moves by more than 0.11 sigma, and a third to a half of the sites move towards the truth.
The exact refits say the same on the converged, recovered products (the eclipsing product's $`K_B`$
error is -2.15 sigma with and without; the twin's pair changes by at most 0.13 sigma, inside its noise
floor) and the opposite of helpful near the floor. Every semi-amplitude in the lowest 5 percent of its range
moved down, away from an injected value above it: $`K_A`$ on `mixed-0019` from -16.6 to -17.1 sigma
and on `mixed-0005` from -15.9 to -16.5, $`K_B`$ on `mixed-0010` from -9.2 to -10.3, on `gaia-5329`
from -3.8 to -3.9 and on `gaia-7289` from -29.9 to -33.1, and on the recovered `mixed-0015` $`K_B`$
from -1.34 to -1.57 and $`K_A`$ from -10.55 to -10.59. The semi-amplitudes these fits underestimate
are the ones the barrier holds up, so on them the barrier happens to point at the truth. The brighter
star's semi-amplitude on the same failed fits moved away from the truth on two (`gaia-7289`, to the
250 km/s ceiling, and `mixed-0010`) and towards it on `gaia-5329`. The eccentricity errors split 5
closer and 6 further, and $`t_{\rm conj}`$ 4 and 5.

**Without the barrier a collapsed semi-amplitude moves 0.5 to 2 km/s further down, onto the 2 km/s
floor on three of the six, for 0.1 to 0.9 nats where the difference exceeds the noise.** On
`mixed-0019__orbit` and `mixed-0005__orbit` the constrained optimum is interior, at 3.65 and 3.54 km/s
($`u = 0.007`$ and 0.006), 0.13 and 0.17 nats below the control's. On
`gaia-53290511099783296__orbit` the control took $`K_B`$ from 13.2 to 2.94 km/s ($`u = 0.0038`$),
where the barrier held it; the constrained refit put it at 2.17 km/s, and its continuation at 2.019
km/s ($`u = 7.6\times 10^{-5}`$), lower in $`U_x`$ than the control's end by 0.88 nats. On
`gaia-7289006178185856__orbit` the constrained refit put $`K_B`$ at 2.045 km/s, $`K_A`$ at 249.89 km/s
against the 250 km/s ceiling and $`t_{\rm conj}`$ at $`u = 0.9993`$ of its window, 6.6 nats below the
control in $`U_x`$; on `mixed-0010__orbit` it took $`K_B`$ from 4.64 to 2.52 km/s for 0.53 nats. On
`gaia-51574864941234048__orbit`, whose semi-amplitudes are archived at 2.0008 and 2.088 km/s against
an injected 13.4 and 15.1, no refit moves (next paragraph), so $`U_x`$ was evaluated along each
semi-amplitude with every other site held (`jac.py --stage profile`): along $`K_A`$ it falls by 0.0040
nats from the archived value to the floor, so the constrained MAP puts $`K_A`$ on the bound, while
$`U_z`$ rises by 13.6 nats at 1e-9 km/s above it; along $`K_B`$ the minimum is at 2.26 km/s, 0.48 nats
lower, and on a joint grid at (2.000, 5.09) km/s, 7.8 nats lower, so the archived point is not an
optimum of either objective. The floor case, then, is a fit that has already failed, where the
constrained objective differs from `run_map`'s by putting the failed semi-amplitude on or near its
bound rather than 0.5 to 2 km/s above it.

**A fit that reaches the eccentricity disk stops moving, whatever the objective.** Nine of the 72
archived fits have $`e_{\rm raw} = h^2 + s^2`$ within 1e-5 of the declared 0.9: `mixed-0010`,
`mixed-0015`, `mixed-0017` and `mixed-0019` blind, `mixed-0013__orbit`, `gaia-51574864941234048`
blind and orbit, and `gaia-7289006178185856` blind and eclipsing, all of them failed orbits (injected
$`e`$ from 0.13 to 0.70). On all nine numpyro's potential at the archived $`z`$ equals the recorded
one to 2e-8 nats, where fits still moving differ by up to 1.6, and on all nine the descent direction
points out of the disk (its outward component is 82 to 100 percent of the gradient in the
$`(\sqrt e\cos\omega, \sqrt e\sin\omega)`$ plane). On `gaia-51574864941234048__orbit` both refits ended
on the archived unconstrained parameters bit for bit, the constrained one after 300 steps at 9 s a
step with its objective unchanged at every logged step, and the control after the ten unchanged steps
that end a refit: the model's `ecc_disk` factor is $`-\infty`$ outside the disk, and a zoom line search
along an outward descent direction from the edge finds only infinite values and returns a zero step
every time. The fit is frozen wherever it first touched the wall. This is independent of the Jacobian
and is reported, not changed.

**First order costs about a minute a product, and an exact refit one to eleven minutes.** The first
stage (re-simulation, rebuild and gates, the gradient identity, the Hessian, the shifts) took a median
56 s (34 to 129 s) per product in a process that peaked at 2.7 GiB (3.6 at most), of which the
sigma-scale Hessian was 30 s (12 to 93 s); the autodiff Hessian took about the same time at 3.7 to
10.9 GiB. A 300-step refit on `gaia5` took 160 to 660 s (0.5 to 2.2 s a step, the slower ones where
the line search struggles, on `gaia-1717` and `gaia-7289`), at a peak of 2.4 to 2.9 GiB; on `field5`,
whose archives are converged, the stopping rules ended most refits in 6 to 715 s, and the two that ran
300 steps took 16 and 27 minutes (3.2 and 5.5 s a step). A step against the eccentricity wall took
9 s.

**Recommendation: leave `run_map`'s default as it is, and do not add the constrained objective as an
option until the eccentricity wall and the convergence criterion are fixed.** On the fits that
recovered their orbit the Jacobian moves the answer by a median 0.007 formal sigma and by 0.23 sigma
at most (`mixed-0015__orbit`, exact), less than the distance of a typical archived fit from its own
optimum, and in no consistent direction relative to the truth. Where it matters, a semi-amplitude
within a few km/s of its floor, the fit has already failed, and the constrained objective would put
the failed value on its bound, where it moves further from the truth and where
`laplace_inverse_mass` and `Fit.sample`, which convert the constrained answer back to $`z`$, meet an
infinite coordinate. The barrier is acting as a mild regulariser on the one class of site where it
acts at all. What would change the disentangling's answers more is (i) a stall-free treatment of the
eccentricity bound, since nine of 72 fits are frozen at the disk; (ii) convergence, since 300 steps
of L-BFGS on this piecewise-smooth potential leave the positive-definite `gaia5` fits a median 0.13
and up to 8.9 sigma from their own optimum, and scatter by up to 2.5 sigma (0.4 on the twin) from
starts one hundredth of a sigma apart; and (iii) a flag for a semi-amplitude in the lowest 5 percent of its
prior range, which here marks seven products, six of them failed orbits. If an option is added later,
as `objective="constrained"` on `run_map`, it
changes no existing test. As a default it would change the label fit too, since `match_labels` calls
`run_map` with nearly every label bounded, and a survey of the 81 tests that reach `run_map` (read,
not run) finds two sensitive by construction and three possibly sensitive:
`tests/test_lsf_h3.py::test_closed_loop_h3_joint_fit_leaves_orbit_unbiased` asserts
$`|h_3| < 0.15`$ inside a $`\pm 0.2`$ box on sites its docstring says the free spectra absorb, so the
barrier may be what keeps them interior;
`tests/test_lsf_varying.py::test_closed_loop_joint_fit_recovers_orbit_and_ramp_direction` asserts the
three widths strictly inside $`(2, 10.5)`$ with one injected at $`u = 0.88`$ on a direction documented
as about 2 nats degenerate, and that the ramp keeps its direction;
`tests/test_pipeline.py::test_the_pipeline_recovers_the_injected_system` holds the fainter star's
temperature to 5 percent where it misses by about 4, through a label fit with $`\log g_B`$ at
$`u = 0.79`$, the radius ratio near its floor and a bounded jitter; and `tests/test_pipeline.py`
lines 334 and 359 assert `teff_err > 0` and a non-null `teff_pull`, which fail if a label lands on
its bound, where the delta-method factor $`u(1-u)`$ vanishes or `_laplace` returns no covariance.
The other 76 are insensitive: their bounded sites are well constrained or at the prior midpoint, or
all their sites are Normal. `MAPResult.potential`'s docstring, "-log joint, up to constants", would
need the Jacobian named either way, since it includes it now.

**What is left.** First, the first-order shift is defined only at a local minimum, and 29 of the 72
archived fits are not at one (20 indefinite, 9 on the disk); on those the effect is measured only
where a refit was run, and the refits of unconverged fits are limited by the optimiser's own scatter
rather than by the objective. Second, converging the `gaia5` fits would take an optimiser suited to a
potential whose gradient jumps at every pixel crossing of every epoch's shift, or a smoother shift
operator; neither was tried, and the formal errors used as the unit here come from a Hessian that
depends, on faint semi-amplitudes, on the scale at which it is taken (a factor 2.4 in curvature on
`mixed-0015__orbit`). Third, the eccentricity wall: the nine frozen fits' optimum under either
objective is unknown, and a transform that bounds $`e`$ instead of a $`-\infty`$ factor would also
put the eccentricity under this note's question, with its own Jacobian. Fourth, only the MAP was
studied: NUTS samples the posterior correctly in either coordinate system, but the Laplace mass
matrix and `Fit.velocity_errors` are built in $`z`$ at the returned point, and what a site on a bound
does to them was not measured. Fifth, the sample is two-component, on the RVS band, with one run's
priors; a declaration whose bounded sites are weakly constrained at their midpoint, such as a wide
period range in a period search or a sampled LSF width, was not in it.
