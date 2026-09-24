# Can the label fit use the disentangling's own sufficient statistics? (D65, 2026-09-16)

`d64_coupled_label_fit.md` fitted the labels through the coupled operator
$`M = (\Lambda + A^\top W A)^{-1} A^\top W A`$ under a diagonal likelihood, found the
temperatures unmoved and the chi-square inflated by a median factor 2.40, and named the one
version of the idea its result did not close: filter the model and whiten the residual by the
same posterior precision. This note measures that version and two neighbours of it. All three
are built from quantities the disentangling already holds at its archived MAP, so none of them
needs a dense operator. The sample is the same twelve products and twenty-four components,
oracle tier except `mixed-0008__orbit`, with the same seeds, declarations and label settings;
the re-simulation, the declaration rebuild and the archived-MAP check are WP-AB's scripts,
reused. Scripts, per-product JSON and logs are in the session scratchpad `wp-ad/`.

**The three likelihoods are exact rearrangements of the disentangling's normal equations.**
With the orbit, the declared light $`\ell^0`$ and the hyperparameters held at the archived MAP,
write $`z`$ for the epoch data in deviation space, $`W`$ for its noise precision (the AR(1) chain
precision here, $`\phi = 0.27`$ on every product), $`A`$ for the stacked operator, $`\Lambda`$ for
the smoothness prior, and

```math
G = A^\top W A, \qquad h = A^\top W z, \qquad P = \Lambda + G, \qquad \hat d = P^{-1} h .
```

Let $`m = m(\phi)`$ be the stacked label-model rows exactly as `albireo.match._model_rows`
returns them in native mode: intrinsic template deviation, broadened, shifted, scaled by
$`w_i/\ell^0_i`$, plus the additive Chebyshev nuisance. The LSF is not in $`m`$, since $`A`$
applies it, and the declared light cancels in $`A m`$, so all three likelihoods see absolute
line depths and measure the light fraction rather than a ratio to $`\ell^0`$. The first is the
chi-square of the template composite fitted to the epoch spectra with the orbit fixed,

```math
L_{\rm data}(m) = z^\top W z - 2\, m^\top h + m^\top G\, m = \lVert z - A m \rVert_W^2 .
```

The second is the disentangling's own marginal likelihood with the template as the prior mean
of the component instead of zero, $`s_i = m_i + \delta_i`$ with
$`\delta \sim \mathcal{N}(0, \Lambda^{-1})`$. By the Woodbury identity
$`(W^{-1} + A\Lambda^{-1}A^\top)^{-1} = W - W A P^{-1} A^\top W`$, so up to terms independent of
$`m`$

```math
-2 \log p(z \mid \phi) = L_{\rm hier}(m) = L_{\rm data}(m) - (h - G m)^\top P^{-1} (h - G m) .
```

The third is the remainder, which is the route `d64_coupled_label_fit.md` proposed, since
$`\hat d - M m = P^{-1}(h - G m)`$:

```math
L_{W1}(m) = (h - G m)^\top P^{-1} (h - G m) = (\hat d - M m)^\top P\, (\hat d - M m) = L_{\rm data} - L_{\rm hier} .
```

Their curvatures in $`m`$ are $`2G`$, $`2\,G P^{-1} G`$ and
$`2\,(G - G P^{-1} G) = 2\,G P^{-1} \Lambda`$, and the last is bounded in the Loewner order by
both $`2G`$ and $`2\Lambda`$, since $`\Lambda - G P^{-1}\Lambda = \Lambda P^{-1} \Lambda \succeq 0`$.
A template change therefore costs $`L_{\rm hier}`$ no more than the smoothness prior would charge
the deviation $`\delta`$ for absorbing it, and on the modes where the data dominate the prior,
$`L_{W1}`$ and $`L_{\rm data}`$ have the same curvature. Both statements turn out to decide the
result. In each route the label fit's log-likelihood is replaced by $`-\tfrac12 L(m)`$ with the
per-component jitter sites removed, since the noise is exact in data units here, and everything
else in `match_labels` is left as it was: the priors, the hull guard, the `RadiusRatio`
dilution, the offsets, and the node-scan warm start against the archived $`\hat d`$.

**The seam is `albireo.match.label_model`, rebound in-process, and the statistics are applied
through the band matvec and the block solve.** $`h`$ comes from `albireo.forward.rhs`,
$`z^\top W z`$ from `weighted_data_terms`, $`G`$ from `band_block_tridiagonal` with a null prior at
the model's own half-bandwidth and block size, and $`P`$ is the marginal's own assembled
precision, factorised once. The stacked rows are interleaved as $`q\,n_c + j`$ and zero-padded
to the block layout, which is `likelihood._pack`'s convention, and the label rows live on the
disentangling grid. The statistics travel as a second traced model argument, so nothing is
constant-folded; they occupy 0.02 to 0.16 GiB, the null-prior band takes 4.1 to 5.0 s to
assemble and the factor 0.1 s. The effective number of parameters the disentangling spends,
$`p_{\rm eff} = \mathrm{tr}(P^{-1} G)`$ from the Takahashi selected inverse, is 183 to 912 of
5748 to 7000 stacked pixels, 3 to 16 percent, so the prior dominates most stacked modes.

**Every gate passed, most of them to rounding.** G1: the marginal at the archived MAP
returns the archived `spectrum_*.txt` to at most 5.0e-10 on all twelve products, which is the
rounding of the saved file; $`L_{\rm hier}(0)`$ equals the marginal's own chi-square
$`z^\top W z - b^\top P^{-1} b`$, taken from `likelihood._solve_stage` on the same precision, to
at most 2.4e-14 relative, and the chi-square reconstructed from its reported log-likelihood to
8.9e-13; $`\hat d`$ recomputed from $`h`$ and the factor agrees to 2.5e-13. G2: $`L_{\rm data}`$
was compared with a brute force built from the definition in `docs/math.md` section 1.4a and
nothing else, a dense covariance $`\alpha^2 D^{-1/2} R_\phi D^{-1/2}`$ per epoch with
$`(R_\phi)_{pq} = \phi^{|i_p - i_q|}`$ over each run of good pixels whose gaps do not exceed
`ar1_max_gap`, factorised densely and fed the forward model's per-epoch prediction
(`forward.apply_model`), so that no band assembly, adjoint, link table or tridiagonal
precision is shared with the fast path. Over 62 points (the injected components, each route's
optimum and WP-AB's native optimum on every product) the worst relative difference is 7.2e-14,
on `mixed-0004`, and the typical one below 5e-15. G3, on `mixed-0008__oracle`: redeclaring
$`\ell^0_B \to 0.5\,\ell^0_B`$ with $`(\tau_B, \eta_B) \to 0.25\,(\tau_B, \eta_B)`$ doubles
$`\hat d_B`$ to 2.0e-13, the label rows recomputed through `_model_rows` with the rescaled
$`\ell^0`$ and offsets equal the rescaled rows exactly, and at three templates $`L_{\rm data}`$ is
unchanged to 0.0, $`L_{\rm hier}`$ to 6e-16 and $`L_{W1}`$ to 8.7e-15. G4, on the same product at
three templates: $`L_{W1}`$ evaluated as $`(\hat d - Mm)^\top P (\hat d - Mm)`$, with $`Gm`$ applied
matrix-free, equals the brute-force $`L_{\rm data}`$ minus $`L_{\rm hier}`$ taken as the marginal's
own chi-square on the residual data $`z - Am`$ (`forward.with_data`, then
`marginal_loglikelihood`) to 2.5e-13, and $`-2\,\Delta\log p`$ equals
$`L_{\rm hier}(m) - L_{\rm hier}(0)`$ to 1.7e-12, so the constant in the hierarchical likelihood is
exactly the determinant terms. G5: with the likelihood replaced by the native diagonal one,
the patched model's log density at the native MAP is bit-identical to the original's
(20518.26104847191 both), and run end to end the labels, chi-square, potential and every
parameter are identical, differences 0.0, and identical to WP-AB's `native_archive`. That is
tighter than WP-AB's own identity gate, whose extra matrix product changed XLA's fusion.

**The seam carries into the potential and the Laplace covariance, but not into
`LabelMatch.chi2`.** On every product and route the model's own `label_loglike` factor,
evaluated at the fitted parameters, equals $`-\tfrac12 L`$ recomputed from the fit's rows to
0.0 relative. The potential recomputed at the final parameters differs from the reported one
by 1e-9 to 40 nats, because `run_map` reports the value at the step before the last; the 40
nats on `mixed-0004` are a first sign that the fits were still descending. The Laplace site
order carries no jitter rows, so the covariance was built from the patched model.
`LabelMatch.chi2` still reports the diagonal comparison against $`\hat d`$, so $`L`$ at each optimum
was recomputed from `match._rows()`.

**As designed, all three routes report smaller errors than the native route, and none of the
temperature gain survives inspection.** Recovered minus injected $`T_{\rm eff}`$, in K. The first
six columns are the designed comparison, WP-AB's native, coupled and truth-fed columns beside
the three new routes; the last four are the same four likelihoods started at the injected
labels, which the paragraphs below need.

| product | comp | native | coupled | data | hier | whiten | truth-fed | native, truth start | data, truth start | hier, truth start | whiten, truth start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mixed-0008 oracle | A | +99.0 | +100.7 | -27.2 | -66.1 | -39.6 | -167.2 | -6.4 | -44.2 | -66.0 | -51.6 |
| mixed-0008 oracle | B | +183.3 | +183.0 | +169.2 | +141.6 | +160.5 | -64.1 | +3.7 | -4.2 | +131.7 | -11.0 |
| mixed-0015 oracle | A | -188.2 | -187.6 | -186.3 | +25.7 | -136.5 | +104.4 | -22.6 | -29.1 | -5.1 | -34.9 |
| mixed-0015 oracle | B | +232.5 | +233.2 | +232.6 | -74.9 | +229.9 | +233.8 | -7.5 | +1.4 | -100.0 | +1.1 |
| mixed-0003 oracle | A | -80.4 | -79.7 | -72.8 | -37.0 | -72.9 | -85.0 | +2.3 | +1.2 | -38.9 | +1.4 |
| mixed-0003 oracle | B | -237.3 | -237.3 | -236.1 | -6.2 | -235.8 | -234.0 | +0.1 | +1.3 | -6.3 | +1.4 |
| mixed-0008 orbit | A | -343.5 | -400.1 | -227.9 | -305.8 | -410.9 | -301.6 | -69.7 | -120.5 | -314.6 | -125.7 |
| mixed-0008 orbit | B | +2432.1 | +2432.1 | +2432.1 | +2424.9 | +2432.1 | +40.8 | +12.3 | -25.0 | +142.0 | -88.2 |
| mixed-0004 oracle | A | -147.2 | -165.8 | -153.4 | -289.2 | +33.5 | -143.6 | -20.4 | -9.3 | -198.2 | -13.0 |
| mixed-0004 oracle | B | -251.0 | -251.0 | -251.0 | -251.0 | -251.0 | -1.1 | +8.2 | +7.1 | -84.6 | +6.9 |
| mixed-0012 oracle | A | -81.0 | -84.6 | -103.6 | -41.2 | -110.5 | -42.4 | -1.7 | +8.4 | -22.3 | +4.7 |
| mixed-0012 oracle | B | -33.0 | -41.6 | -82.4 | -171.4 | -97.2 | -6.1 | +2.2 | -11.8 | -106.4 | -5.1 |
| mixed-0013 oracle | A | +93.3 | -160.9 | -161.8 | +11.1 | -161.9 | +83.4 | -6.7 | +0.6 | +10.7 | +1.1 |
| mixed-0013 oracle | B | +206.3 | +200.5 | +201.3 | -44.1 | +201.7 | -36.9 | +3.5 | +0.5 | -48.5 | +1.0 |
| mixed-0017 oracle | A | -152.3 | -174.6 | -93.6 | -55.6 | -90.4 | -4.1 | +103.0 | -49.4 | -98.0 | -25.8 |
| mixed-0017 oracle | B | +258.2 | +257.4 | +255.0 | +60.3 | +255.1 | -3.0 | +20.5 | -5.1 | +7.0 | -12.8 |
| gaia-17170287112790656 oracle | A | +278.4 | -217.5 | -13.7 | +1.9 | -40.8 | -224.1 | +12.3 | +5.6 | +0.9 | +7.0 |
| gaia-17170287112790656 oracle | B | +268.5 | +268.7 | +268.1 | -27.8 | +267.7 | -230.2 | -1.8 | +3.8 | -26.8 | +3.8 |
| gaia-51574864941234048 oracle | A | -203.2 | -194.5 | -223.4 | -76.1 | -182.1 | -195.9 | -6.3 | -16.5 | -67.3 | -15.8 |
| gaia-51574864941234048 oracle | B | +218.7 | +221.7 | +216.2 | -73.3 | +213.2 | -278.5 | -12.1 | -33.5 | -203.6 | -53.8 |
| gaia-47620265212420096 oracle | A | -180.0 | -179.5 | -107.1 | -102.3 | -83.3 | -85.8 | -56.4 | -28.0 | -46.3 | -37.6 |
| gaia-47620265212420096 oracle | B | -74.9 | -73.2 | -14.0 | -77.7 | +6.1 | -17.2 | -2.6 | -24.8 | -75.8 | -33.0 |
| gaia-53680017390963072 oracle | A | -38.9 | -36.7 | -40.5 | -73.9 | -40.1 | -53.2 | +6.5 | +7.2 | -84.8 | +10.3 |
| gaia-53680017390963072 oracle | B | +63.0 | +63.7 | +71.2 | -68.2 | +71.0 | +79.6 | -11.5 | -0.1 | -55.2 | -0.1 |

Over the twenty-four components, as median absolute error with the 16th to 84th percentile
range, followed by the number of components improved and worsened against the native route:

| label | native | coupled | data | hier | whiten | truth-fed |
| --- | --- | --- | --- | --- | --- | --- |
| $`T_{\rm eff}`$ (K) | 185.7 [78.6, 261.5] | 185.3 [77.7, 253.1], 12/12 | 165.5 [61.4, 240.9], 15/9 | 70.8 [27.1, 196.9], 19/5 | 148.5 [40.6, 252.3], 17/7 | 84.2 [13.6, 231.4], 18/6 |
| $`\log g`$ (dex) | 0.185 [0.078, 0.262] | 0.185 [0.077, 0.245], 19/5 | 0.184 [0.064, 0.243], 21/3 | 0.034 [0.008, 0.084], 21/3 | 0.160 [0.048, 0.235], 21/3 | 0.179 [0.081, 0.273], 15/9 |
| [M/H] (dex) | 0.057 [0.015, 0.087] | 0.040 [0.011, 0.088], 14/10 | 0.027 [0.004, 0.062], 14/10 | 0.084 [0.040, 0.157], 6/18 | 0.027 [0.005, 0.061], 16/8 | 0.057 [0.018, 0.127], 12/12 |
| $`v\sin i`$ (km/s) | 7.65 [3.70, 28.26] | 3.44 [1.86, 27.58], 19/5 | 3.49 [1.79, 17.35], 16/8 | 1.51 [0.80, 7.41], 20/4 | 3.38 [1.31, 28.66], 16/8 | 1.86 [0.64, 6.12], 22/2 |
| light fraction | 0.0277 [0.0067, 0.0770] | 0.0086 [0.0053, 0.0279], 20/4 | 0.0083 [0.0008, 0.0276], 20/4 | 0.0074 [0.0046, 0.0243], 18/6 | 0.0085 [0.0028, 0.0295], 20/4 | 0.0052 [0.0008, 0.0150], 18/6 |

Separately for the twelve primaries and the twelve secondaries, median absolute error and the
improved/worsened counts:

| set | native | coupled | data | hier | whiten | truth-fed |
| --- | --- | --- | --- | --- | --- | --- |
| primaries, $`T_{\rm eff}`$ (K) | 149.8 | 170.2, 6/6 | 105.3, 7/5 | 60.9, 10/2 | 86.9, 8/4 | 95.1, 9/3 |
| secondaries, $`T_{\rm eff}`$ (K) | 225.6 | 227.5, 6/6 | 224.4, 8/4 | 74.1, 9/3 | 221.5, 9/3 | 52.5, 9/3 |
| primaries, $`v\sin i`$ (km/s) | 6.34 | 2.10, 9/3 | 2.77, 7/5 | 1.12, 9/3 | 2.59, 8/4 | 0.95, 11/1 |
| secondaries, $`v\sin i`$ (km/s) | 15.24 | 10.22, 10/2 | 6.42, 9/3 | 2.16, 11/1 | 8.59, 8/4 | 2.50, 11/1 |

Read at face value, the hierarchical route cuts the temperature error from 185.7 to 70.8 K and
the gravity error by a factor 5.4, to 0.034 dex, and the data and whitened routes cut the
temperature error by 20 and 37 K, most of it on the primaries (149.8 K native against 105.3 and
86.9 K) while the secondaries stay at 224.4 and 221.5 K against 225.6 K. All three recover the
light fraction to a median 0.0074 to 0.0085 against 0.0277 native and a truth-fed 0.0052, which
matches the coupled operator's 0.0086, and all three remove the native route's systematic
over-estimate of rotation: native puts $`v\sin i`$ high on 21 of 24 components with a median
error of +7.65 km/s, and the three routes on 9, 8 and 10, with medians of -1.78, -0.88 and
-1.02 km/s. Three things undercut the temperature reading, and they are measured next.

**The designed fits do not reach the minimum of their own objective, so the warm start and the
80-step budget set their temperatures.** Four measurements say so independently. First, on four
products the data route stops at a higher $`L_{\rm data}`$ than the injected components give,
by 2.8 to 681 (`mixed-0015`, `mixed-0003`, `mixed-0013`, `mixed-0004`); since the injected
components are evaluated through the archived orbit and carry the resolution mismatch described
below, the other eight do not show convergence, only that the template beat an imperfect
reference. Second, the noise-free control. Fed $`z_0 = A\,y_{\rm true}`$ at the archived MAP, for
which every likelihood is zero at the injected components (computed as 3e-11 and below) and the
warm-start scan sees $`M\,y_{\rm true}`$, which is exactly WP-AB's control input, the data route
returns +3.6 and +171.0 K on `mixed-0008` at $`L_{\rm data} = 30.4`$ and -147.5 and +231.9 K on
`mixed-0015` at 209.9, and the whitened route +95.7 and +181.2 K ($`L_{W1} = 22.2`$) and +18.7
and +220.6 K (8.8). WP-AB's coupled fit to the same input returned +117.2 and +182.1 K and
-184.4 and +233.4 K. The secondaries stay within 13 K of the native answer with and without
noise, at a positive value of an objective whose minimum is zero at the truth, so neither the
noise nor the likelihood holds them there. Third, started at the injected labels (one start,
the same 80 steps), the data route reaches a lower potential than its designed fit on all 12
products, by a median 20.1 nats and up to 1125 on `mixed-0004`, and the whitened route on 11 of
12, by a median 13.6; their temperatures then land a median 7.8 K and 10.6 K from the truth, and
their gravities 0.004 and 0.003 dex. Those errors are a best case, since a fit that barely moves
from its start will look accurate when started at the truth; the potential comparison is the
robust part, and it says that on every product a point near the truth is better, by the route's
own objective, than what the designed fit returned. Fourth, five times the steps from the same
scan start moves the data route toward that point: on `mixed-0003` from -72.8 and -236.1 K to
-3.1 and -53.8 K (potential 25285.7 to 25275.3, against 25272.3 from the truth), on `mixed-0015` A
from -186.3 to -8.2 K, on `mixed-0008` B from +169.2 to -36.5 K, and on `gaia-17170287112790656`
B not at all, +268.1 to +254.7 K. The unconstrained problem is badly conditioned for L-BFGS: over
its prior the data likelihood changes by a median 2.1e4 nats in [M/H], 1.9e4 in the primary's
$`v\sin i`$ and 2.3e3 in the radius ratio, but by 145 in the secondary's temperature and 35 in its
gravity, reaching 0.2 and 0.3 on the faintest.

| label | native, truth start | data, truth start | hier, truth start | whiten, truth start |
| --- | --- | --- | --- | --- |
| $`T_{\rm eff}`$ (K) | 7.1 [2.2, 21.2] | 7.8 [1.2, 30.5] | 66.6 [9.5, 135.0] | 10.6 [1.3, 42.1] |
| $`\log g`$ (dex) | 0.004 [0.001, 0.020] | 0.004 [0.001, 0.014] | 0.023 [0.010, 0.083] | 0.003 [0.001, 0.020] |
| [M/H] (dex) | 0.018 [0.012, 0.058] | 0.008 [0.003, 0.030] | 0.060 [0.025, 0.163] | 0.008 [0.001, 0.031] |
| $`v\sin i`$ (km/s) | 4.35 [0.15, 7.40] | 1.10 [0.05, 3.53] | 1.49 [0.65, 6.63] | 1.22 [0.10, 2.83] |
| light fraction | 0.0394 [0.0041, 0.0847] | 0.0075 [0.0022, 0.0196] | 0.0062 [0.0039, 0.0228] | 0.0080 [0.0027, 0.0210] |
| potential below the designed fit | 8 of 12, median -21.3 | 12 of 12, median -20.1 | 12 of 12, median -0.2 | 11 of 12, median -13.6 |

**The same holds for the unmodified native route, which re-reads the D63 and D64 temperature
errors.** Started at the injected labels, the shipped label fit returns a median 7.1 K and
0.004 dex, and reaches a lower potential than its designed fit on 8 of 12 products, by a median of
197 nats over those eight. On the other four (`mixed-0008` oracle, `mixed-0004`, `mixed-0017`,
`gaia-51574864941234048`) the native objective prefers the designed answer, by 53 to 369 nats,
and there the diagonal comparison against the smoothed component is itself biased, which is the
effect D64 set out to measure. On eight products it is not the binding constraint: the warm
start and the step budget are. WP-AB's finding that filtering the model moves the temperature by
a median 0 K was therefore measured through fits that do not leave their warm start, and the
third benchmark run's label fits used `label_steps = 80`, where the pipeline default is 500.
The data likelihood prefers a point near the truth to the designed answer on all twelve
products and the native likelihood on eight, which is the one statement about temperature this
measurement supports.

**The hierarchical route has almost no information about the labels, and in the oracle tier a
flat likelihood returns the truth.** Its curvature is bounded by $`\Lambda`$, and measured as the
range of $`\tfrac12 L`$ along one parameter over its prior, through each route's designed
optimum, it changes by a median 4.3 and 5.5 nats in the two temperatures and 0.66 and 0.60 nats
in the two gravities, against 1030, 145, 196 and 35 nats for the data route, a median 0.9 percent
of the data route's temperature information and 0.4 percent of its gravity information:

| parameter | native | data | whiten | hier |
| --- | --- | --- | --- | --- |
| $`T_{\rm eff,A}`$ | 2590 [360, 3.1e4] | 1030 [62, 5.3e4] | 1080 [62, 2.0e4] | 4.3 [0.89, 280] |
| $`T_{\rm eff,B}`$ | 1150 [0.11, 6300] | 145 [0.18, 1700] | 144 [0.17, 1800] | 5.5 [0.84, 120] |
| $`\log g_A`$ | 1270 [26, 1.4e4] | 196 [14, 6700] | 172 [12, 1.0e4] | 0.66 [0.094, 3.9] |
| $`\log g_B`$ | 141 [0.046, 2000] | 35 [0.34, 510] | 62 [0.047, 520] | 0.60 [0.13, 15] |
| $`v\sin i_A`$ | 5.0e4 [1.8e4, 1.7e5] | 1.9e4 [1300, 1.0e6] | 1.9e4 [1200, 1.1e6] | 174 [83, 4300] |
| $`v\sin i_B`$ | 3300 [2.6, 3.9e4] | 853 [4.2, 3.7e4] | 625 [0.7, 3.7e4] | 117 [6, 680] |
| radius ratio | 5.6e4 [7.4, 1.9e5] | 2300 [5.2, 3.3e4] | 2100 [2.4, 3.3e4] | 164 [50, 940] |

The entries are nats, median and range over the twelve products; the native column is the
diagonal chi-square at its fitted jitter. numpyro optimises in the unconstrained space, where a
`Between(lo, hi)` site carries the log-Jacobian $`\log u(1-u)`$ of its sigmoid, which changes by
3.2 nats over the same range and is largest at the midpoint, and the oracle tier's temperature
and gravity priors are $`\pm 300`$ K and $`\pm 0.3`$ dex centred on the injected values. The
hierarchical likelihood's gravity range is below those 3.2 nats on 11 of 12 products for each
component, and its temperature range on 5 of 12, so the labels it returns are the prior centres,
which here are the truth. Two tests confirm it. Moving both priors by +150 K and +0.15 dex with unchanged widths, on
four products, moves the hierarchical gravities by 0.83 to 0.98 of the shift and its temperatures by
0.40 to 0.89 of it on `mixed-0003` and `gaia-17170287112790656`; on `mixed-0008` and
`mixed-0015`, where the shift also changes the library node the scan picks, its labels move by
-0.41 to 1.85 of it. The data and whitened routes under the same shift move by -0.04 to 3.3 of
it, following the new node rather than the prior centre. And the hierarchical potential from
the truth start is within 0.6 nats of the designed one on 9 of 12 products (median 0.2), where
its temperatures on those nine differ by up to 65 K. The 70.8 K and 0.034 dex
are an artefact of priors centred on the truth and would not survive any other prior. The reason
is the one the derivation gives: the hierarchical model forgives whatever template mismatch the
smooth deviation can absorb at the fitted smoothness, and a change of temperature within the
prior is such a mismatch. It pays for that in the data. At its optimum $`L_{\rm data}/n`$ has a
median of 1.019 against 1.0008 for the data route, and reaches 2.83 on `mixed-0004` and 1.37 on
`gaia-47620265212420096`, the two products at S/N 97 to 99.

**The whitened route is the data route.** On the modes where the data dominate the prior their
curvatures coincide, and the labels live on those modes although most stacked modes are
prior-dominated: the whitened-to-data ratio of the profile ranges above has a median of 0.94 to
1.03 for every parameter but the secondary's rotation (0.86), the designed temperatures agree to
a median 3.1 K (within 5 K on 13 of 24 components and within 15 K on 17), the truth-started ones
to 3.7 K (13 and 21), and the light fractions to a median 0.0018. $`L_{W1}`$ needs a block solve
per evaluation that $`L_{\rm data}`$ does not.

**The data-space routes expose that the label model cannot declare the library's resolving
power.** They put $`v\sin i`$ at the floor of its prior, below 0.5 km/s, on 6 (data) and 7
(whitened) components, where native and coupled put it there on none. The simulation renders
the components from BOSZ at $`R = 20{,}000`$ ($`\sigma_{\rm lib} = 6.37`$ km/s) and broadens them by
the quadrature width 9.06 km/s to reach the RVS 11.07 km/s. The disentangling declares 11.07
km/s, correctly for the data, and $`A`$ applies it. The label model renders its template from the
same library, already at $`\sigma_{\rm lib}`$, so $`A m`$ carries 12.77 km/s where the data carry
11.07, and a slowly rotating component can only lower $`v\sin i`$ to zero. `match_labels` and
`StarLabels` have no parameter for the library's own resolving power, while
`todcor.Template.from_library` does, for exactly this correction. Without noise the effect is
unchanged: fed $`z_0 = A_q\,y_{\rm true}`$, the noise-free counterpart of the simulated data with
$`A_q`$ carrying the applied 9.06 km/s, the data route through the declared operator returns the
primary of `mixed-0008` at $`v\sin i = 0.0`$ against an injected 10.2 km/s, and the injected
components cost $`L_{\rm data} = 106`$ and 154 on the two control products instead of zero.
The native route is likely to hide it, since $`\hat d`$ is a partial deconvolution to the
declared width and the smoothing already inflates rotation there; that was not measured.

**Giving the operator the width the simulation applied removes every floor value and lowers the
light-fraction error below the truth-fed floor, and leaves the temperatures where the warm start
put them.** As a diagnostic, not as a change to the design, the LSF in $`A`$, and so in $`G`$, $`h`$
and $`P`$ with $`(\tau, \eta)`$ unchanged, was replaced by the 9.06 km/s applied above the library,
so that $`A_q m`$ carries 11.07 km/s, which is the correct forward model for a template at the
library's resolving power. `forward.with_lsf` at the declared width reproduces the build
kernel to 2.8e-17, and on all twelve products G2 holds to 6.0e-14 and the factor seam to 0.0.
The injected components' $`L_{\rm data}/n`$ falls from 1.071 to 1.009 on `mixed-0004` and from
1.053 to 1.025 on `gaia-47620265212420096`, so most of the excess chi-square of the truth at
S/N 97 to 99 was this mismatch; on nine of the other ten it changes by at most 0.0032, and on
`gaia-51574864941234048`, whose archived orbit appears wrong, it rises from 1.142 to 1.177. Under
$`A_q`$ the data route puts no component at the rotation floor, and its light-fraction error falls
from a median 0.0083 to 0.0036 [0.0013, 0.0251], improving on 18 of 24 components and passing
below the truth-fed 0.0052: `mixed-0008__orbit` B returns 0.1118 against a true 0.1119,
`mixed-0017` B 0.2317 against 0.2332, `gaia-17170287112790656` B 0.0715 against 0.0673. The
temperatures do not move, a median 165.6 K against 165.5 K with 12 components better and 12
worse, which is what a fit held at its warm start predicts. Rotation now errs high, by a median
+3.57 km/s on 23 of 24 components: by +29 to +82 km/s on the four secondaries whose designed
temperatures sit at the scan node and by +16 and +78 km/s on `gaia-51574864941234048`, and by
-2.1 to +6.3 km/s, median +2.9, on the other eighteen. That residual was not traced; the delivered product's resampling, which the operator does not model beyond the
rebin, and the unconverged fit are both candidates. The hierarchical route under $`A_q`$ keeps its
prior-centred labels (55.1 K, 0.020 dex) and its light fraction worsens to 0.0123, and the
whitened route's improves only to 0.0081, so the equivalence of the whitened and data routes
measured with the declared operator is approximate rather than exact, and did not carry over
to the modified one.

**The light fraction is where a data-space likelihood clearly helps, and it does not need the
temperatures to be right.** Designed, the three routes return a median error of 0.0083, 0.0074
and 0.0085 against the native 0.0277; started at the truth they return 0.0075, 0.0062 and 0.0080,
while the native route started at the truth still misses by 0.0394. So the native error in the
light fraction is the likelihood's, not the optimizer's, and it is removed by comparing in the
data space. The gains are where native lost the secondary: `mixed-0017` B moves from 0.127 to
0.224 under the data route against a true 0.233, `gaia-17170287112790656` B from 0.017 to 0.079
against 0.067, `mixed-0008` B from 0.085 to 0.112 against 0.112, `mixed-0015` B from 0.010 to
0.027 against 0.073. The failure is `gaia-51574864941234048` B, 0.0006 native and 0.0075, 0.079
and 0.002 under the three routes against a true 0.396. There the injected components fit the
epoch data worse than any route's optimum, $`L_{\rm data}/n = 1.142`$ against 1.026, so the
archived orbit itself does not describe the data, and no likelihood on the label side can
repair that.

**At the survey S/N the data route's reduced chi-square is close to one, and it measures
template mismatch only at high S/N.** At the data route's optimum $`L_{\rm data}/n`$ has a median
of 1.0008 and a range of 0.978 to 1.080 over the twelve products; the injected components,
shifted by the systemic velocity and evaluated through the same operator, give a median 1.0019
and a range of 0.979 to 1.142. The native optimum gives 1.0065 (0.994 to 1.115) and the coupled
one 1.0020. Apart from `gaia-51574864941234048`, the two products near or above 1.05 are
`mixed-0004` (1.080 at the optimum, 1.071 for the injected components) and
`gaia-47620265212420096` (1.049 and 1.053), the two at S/N 97 to 99, where the doubled
library width is large enough to show in the chi-square of the truth itself.

**The routes cost what the native fit costs, and less than the coupled one.** The statistics are
built once per star: the null-prior band in 4.1 to 5.0 s, the factor in 0.1 s, the selected
inverse for $`p_{\rm eff}`$ in 0.2 s, 0.02 to 0.16 GiB in all, against the 0.25 to 0.37 GiB dense
operator WP-AB built in 6.6 to 17.4 s. The label fits took 53 to 73 s (median 64) on the data
route, 52 to 88 s (77) on the hierarchical and 49 to 87 s (74) on the whitened, against WP-AB's
65 to 150 s (98) native and 132 to 251 s (150) coupled in an earlier session; measured in one
process on `mixed-0008`, the native fit took 57 s, the identity-gated copy 44 s and the three
routes 53, 52 and 49 s. The machine shared these runs with the WP-AE table diagnosis. The peak working
set per product process was 3.2 to 4.1 GiB, 5.7 GiB for the gate product, which ran two more
fits and the dense gates, and the resident size after a fit 1.7 to 3.9 GiB, against WP-AB's 0.6
to 2.9 GiB by the same measure.

**Recommendation.** For the light fraction, use $`L_{\rm data}`$ now. It is the exact chi-square of
the template composite against the epoch spectra with the orbit fixed, it needs only $`h`$,
$`z^\top W z`$ and the band of $`G`$, which the disentangling already assembles, it costs no more
than the present fit, and it cuts the light-fraction error by a factor 3.3 as designed and 5.3
when both are started at the truth, to within a factor 1.4 to 1.6 of the truth-fed floor, with
a reduced chi-square that is interpretable; with the library's resolving power accounted for in
the operator it reaches 0.0036, below that floor. The whitened route gives the same answers with the
declared operator for the price of a solve, and did not with the quadrature one. The hierarchical route, which recovers the light fraction about as well,
should not be used for anything that also reports labels. For the labels, $`L_{\rm data}`$ is the
right objective, since it prefers a point near the truth to the designed answer on all twelve
products where the native objective does so on eight, but it should replace the native comparison
only after two changes to `match_labels`, neither of which this measurement made: the library's
own resolving power must be declarable, so that the template is broadened only by the quadrature
difference, and the fit must converge in the weakly constrained directions, by a larger step
budget, restarts across the faint component's temperature and gravity, or a better-conditioned
parameterization. Until then no route's temperature, native included, measures its likelihood
rather than its warm start. The hierarchical likelihood should not be used for labels at all:
its information on temperature and gravity is 0.4 to 0.9 percent of the data route's and its
answers are the prior centres.

**What is left.** First, a converged estimator: the truth-started numbers bound what each
likelihood can do near the truth and are not a measurement of an estimator, and the
step-budget test shows that five times the steps is not enough on the faintest secondary. A
restart scan over the faint component's temperature and gravity through $`L_{\rm data}`$, which is
cheap because the statistics are fixed, is the next measurement. Second, the hierarchical
likelihood's curvature is bounded by a $`\Lambda`$ whose $`(\tau, \eta)`$ were fitted for a
zero-mean prior; refitting them with the template as the mean would make the smoothness prior describe template mismatch rather than
the whole spectrum, and whether that restores label information was not tested. Third, every
label-accuracy figure in the oracle tier of the benchmark is exposed to the prior-centre effect
measured here wherever a label is weakly constrained, since those priors are centred on the
truth; the orbit and blind tiers, whose priors are not, were not run with the new likelihoods
beyond `mixed-0008__orbit`. Fourth, where the archived orbit is wrong, as it appears to be on
`gaia-51574864941234048`, no fixed-orbit likelihood can help, and since $`L_{\rm data}`$ is
the epoch chi-square it is also the objective a joint fit of labels and orbit would minimise.
Fifth, under the quadrature operator rotation errs high by a median +2.9 km/s even on the
components whose fits are not stuck, which was not traced. Sixth, the sample is two-component,
on the RVS band at $`R = 11{,}500`$ with one library; at higher resolving power the data-dominated modes widen and the hierarchical
route's flatness may lessen, and neither was measured.
