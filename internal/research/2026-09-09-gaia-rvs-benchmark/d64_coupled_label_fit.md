# Does fitting the labels through the coupled operator recover the temperature? (D64, 2026-09-11)

**[Corrected in part by `d65_label_likelihoods.md` and `d65_converged_labels.md` (D65, 2026-09-16). Every fit in this note ran 80 L-BFGS steps from the node-scan warm start and did not converge, so its temperatures measure that start rather than either likelihood. Started at the injected labels instead, the shipped fit reaches a lower value of its own objective on 8 of the 12 products and lands a median 7.1 K from the truth. Converged by a bounded Levenberg-Marquardt with restarts, the shipped diagonal comparison returns a median 119 K and the exact comparison of the template composite against the epoch spectra returns 60 K. The light-fraction and rotation gains reported here survive, and the epoch comparison improves on both. The recommendation not to rebuild `match_labels` around this operator stands, for a different reason: the epoch comparison is the exact likelihood and a filtered comparison under a diagonal likelihood is not.]**

`d64_posterior_smoothing.md` established that the archived posterior mean is exactly the
injected pair passed through the coupled operator, plus a noise term,

```math
\hat d \;=\; \underbrace{(\Lambda + A^\top W A)^{-1} A^\top W A}_{M}\, y_{\rm true}
\;+\; (\Lambda + A^\top W A)^{-1} A^\top W n ,
```

and closed by proposing that the label fit compare the pair of synthetic models passed
through $`M`$ against the pair of disentangled components jointly, rather than each component
against its own. That reproduces the archive as a *forward* model, which is not the same
claim as recovering the truth when one fits *through* it. This note measures the second
claim on twelve star-and-tier products and twenty-four components. The sample is drawn from
`gaia5` and `field5`, oracle tier wherever a product exists, so that the orbit is right and
only the spectra are in question; it contains the four products `d64_posterior_smoothing.md`
used, so the numbers below sit beside its table, and the remaining eight were chosen to
spread the declared light ratio from 1.06 to 18.2 and the line separation
$`(K_1+K_2)/\mathrm{FWHM}_{\rm LSF}`$ from 1.09 to 11.3. The single non-oracle product is
`mixed-0008__orbit`, carried over from that note. The re-simulation, the declaration rebuild
and the marginal evaluation are its scripts unchanged; scripts, per-star JSON and logs for
this note are in the session scratchpad `wp-ab/`.

| product | epochs | S/N | grid | $`\ell_A/\ell_B`$ | $`(K_1+K_2)/\mathrm{FWHM}`$ | $`\tau_A`$ | $`\tau_B`$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixed-0008 oracle | 38 | 22.3 | 2923 | 7.94 | 2.76 | 3.7e4 | 6.1e4 |
| mixed-0015 oracle | 27 | 40.9 | 2896 | 12.69 | 1.30 | 8.0e4 | 329 |
| mixed-0003 oracle | 53 | 16.7 | 3170 | 1.10 | 10.49 | 1.0e6 | 7.5e5 |
| mixed-0008 orbit | 38 | 22.3 | 3500 | 6.02 | 2.76 | 3.6e4 | 2.0e3 |
| mixed-0004 oracle | 80 | 98.5 | 2972 | 18.23 | 2.49 | 4.4e3 | 1.2e4 |
| mixed-0012 oracle | 100 | 29.3 | 2981 | 1.06 | 4.58 | 1.9e4 | 1.8e4 |
| mixed-0013 oracle | 43 | 52.6 | 2956 | 1.15 | 2.29 | 5.4e6 | 3.2e5 |
| mixed-0017 oracle | 22 | 19.6 | 2939 | 3.29 | 1.95 | 9.2e3 | 1.0e4 |
| gaia-17170287112790656 oracle | 11 | 21.0 | 3197 | 13.85 | 11.34 | 7.3e6 | 249 |
| gaia-51574864941234048 oracle | 14 | 40.8 | 2874 | 1.53 | 1.09 | 5.0e4 | 92.1 |
| gaia-47620265212420096 oracle | 13 | 97.1 | 2899 | 1.18 | 2.02 | 5.6e3 | 5.4e3 |
| gaia-53680017390963072 oracle | 17 | 23.2 | 3193 | 1.14 | 7.64 | 2.1e5 | 2.3e6 |

**The patch is at the right seam, and with the operator replaced by the identity it
reproduces the present route bit for bit.** `albireo.match._model_rows` is the one place
where the stacked $`(n_{\rm star}, n_{\rm pix})`$ candidate model is assembled for the
comparison, and it is reached both by the numpyro model and by `LabelMatch._rows`, so
rebinding it inside this note's own process carries the change into the likelihood, into the
reported chi-square and into the Laplace covariance together. The patched function calls the
original with the additive nuisance held at zero, multiplies the stacked pair by $`M`$ jointly
in the interleaved order $`g = q\,n_c + j`$ that the band assembly uses, and adds the
Chebyshev nuisance afterwards, because what that nuisance absorbs is the disentangling's own
unidentified zero point and therefore lives in the output space. Evaluated eagerly at one
fixed set of MAP arguments, so that no optimiser trajectory stands between the two answers,
the patched rows with $`M = I`$ equal the unpatched rows exactly, `numpy.array_equal` true, on
all three products where the check was run. Run end to end the agreement is no longer exact
but is at the optimiser's own determinism: the largest label difference over a whole fit is
1.4e-5 K on `mixed-0008__oracle`, 6.1e-6 K on `mixed-0012__oracle` and 8.8e-5 K on
`gaia-17170287112790656__oracle`, with potential differences of 1.6e-5, 4.7e-7 and 2.5e-3,
the extra matrix product changing XLA's fusion and so the last bits of an L-BFGS trajectory.

**What was reproduced, and how it was checked.** The marginal evaluated at the archived MAP
returns the archived `spectrum_*.txt` to 5e-10 on every product, as in the preceding note.
The dense $`M`$ built here, column by column from the band assembly of $`A^\top W A`$ with a
null prior and the marginal's own Cholesky factor, agrees with `d64_posterior_smoothing.md`'s
own application of the same operator, which used the band matvec and the block solve directly
and never formed a matrix, to at most 1.9e-14 in normalised flux on the eleven products both
notes cover. The present route on the archived component reproduces the archived
`labels.txt`: over the twenty-four components the median difference in $`T_{\rm eff}`$ is
0.0013 K, twenty are within 0.1 K and twenty-three within 0.5 K, the exception being
`mixed-0008__orbit` A at 9.4 K, which is the multimodal blind-tier fit the benchmark already
flags; $`v\sin i`$ agrees to a median 0.004 km/s and at worst 0.29 km/s. Where
`d64_posterior_smoothing.md` reports the same quantity the two agree exactly, for instance
`mixed-0015__oracle` at -188.2 K and +232.5 K on the archive and +104.4 K and +233.8 K on the
injected component.

**[Corrected by D65: the fits below did not converge; see the note under the title.]** **The answer is no: passing the model through the coupled operator does not recover the
injected temperature.** Recovered minus injected $`T_{\rm eff}`$, in K, under the present
route on the archived component, the patched route on the archived component, and the present
route on the injected component:

| product | comp | native archive | coupled archive | native truth |
| --- | --- | --- | --- | --- |
| mixed-0008 oracle | A | +99.0 | +100.7 | -167.2 |
| mixed-0008 oracle | B | +183.3 | +183.0 | -64.1 |
| mixed-0015 oracle | A | -188.2 | -187.6 | +104.4 |
| mixed-0015 oracle | B | +232.5 | +233.2 | +233.8 |
| mixed-0003 oracle | A | -80.4 | -79.7 | -85.0 |
| mixed-0003 oracle | B | -237.3 | -237.3 | -234.0 |
| mixed-0008 orbit | A | -343.5 | -400.1 | -301.6 |
| mixed-0008 orbit | B | +2432.1 | +2432.1 | +40.8 |
| mixed-0004 oracle | A | -147.2 | -165.8 | -143.6 |
| mixed-0004 oracle | B | -251.0 | -251.0 | -1.1 |
| mixed-0012 oracle | A | -81.0 | -84.6 | -42.4 |
| mixed-0012 oracle | B | -33.0 | -41.6 | -6.1 |
| mixed-0013 oracle | A | +93.3 | -160.9 | +83.4 |
| mixed-0013 oracle | B | +206.3 | +200.5 | -36.9 |
| mixed-0017 oracle | A | -152.3 | -174.6 | -4.1 |
| mixed-0017 oracle | B | +258.2 | +257.4 | -3.0 |
| gaia-17170287112790656 oracle | A | +278.4 | -217.5 | -224.1 |
| gaia-17170287112790656 oracle | B | +268.5 | +268.7 | -230.2 |
| gaia-51574864941234048 oracle | A | -203.2 | -194.5 | -195.9 |
| gaia-51574864941234048 oracle | B | +218.7 | +221.7 | -278.5 |
| gaia-47620265212420096 oracle | A | -180.0 | -179.5 | -85.8 |
| gaia-47620265212420096 oracle | B | -74.9 | -73.2 | -17.2 |
| gaia-53680017390963072 oracle | A | -38.9 | -36.7 | -53.2 |
| gaia-53680017390963072 oracle | B | +63.0 | +63.7 | +79.6 |

The median absolute error moves from 185.7 K to 185.3 K, the mean from 264.3 K to 268.6 K,
and the 16th to 84th percentile range from 78.6 to 261.5 K against 77.7 to 253.1 K. Twelve
components improve and twelve get worse, the median change is +0.0 K and the mean +4.2 K, the
largest gain is 61.0 K on `gaia-17170287112790656` A and the largest loss 67.7 K on
`mixed-0013` A. On sixteen of the twenty-four the change is smaller than 5 K, and the root
mean square change over the sample, 22.9 K, is dominated by those two cases and by
`mixed-0008__orbit` A. Separating the components does not rescue it: the primaries move from
a median 149.8 K to 170.2 K and the secondaries from 225.6 K to 227.5 K, six improving and
six worsening in each half. The third column bounds any gain and the bound is not the
binding constraint here: fed the injected component the same fit misses by a median 84.2 K,
reaching 301.6 K, so a perfect correction would have had a median 100 K to recover, and the
patch recovers none of it.

**The control says why, and it is not the noise.** If the obstacle were the noise term, then
deleting it should let the patched fit find the truth. It does not. Fitting $`M\,m(\phi)`$
against $`M\,y_{\rm true}`$, which is the archived component with its noise term removed
entirely, returns +117.2 K and +182.1 K on `mixed-0008__oracle` where the truth-fed fit
returns -167.2 K and -64.1 K, and -184.4 K and +233.4 K on `mixed-0015__oracle` where it
returns +104.4 K and +233.8 K. Set beside `d64_posterior_smoothing.md`, which fed the same
$`M\,y_{\rm true}`$ to the *unpatched* fit and obtained +113.8, +183.4, -187.3 and +232.2 K,
the difference that filtering the model makes is +3.4, -1.3, +2.9 and +1.2 K. The asymmetry
is complete: putting the operator on the data side moves the recovered temperature by 280 K,
and putting the same operator on the model side as well moves it back by 3 K. The one
exception is `mixed-0008__orbit` A, where the patched noiseless fit returns -95.5 K against
the unpatched -143.3 K, a 48 K difference on the product whose orbit is wrong. This is not
because the operator leaves the model alone. At a fixed MAP, $`M`$ changes the stacked model
rows by a root mean square of 0.027, 0.008 and 0.016 in normalised flux against model root
mean squares of 0.074, 0.060 and 0.061, which is 14 to 37 percent, with maximum excursions of
0.14, 0.055 and 0.10. The operator does a great deal to the spectrum and nothing to the
temperature.

**The likelihood is mis-specified in exactly the way `compare="matched"` was, and that is
visible in the chi-square.** $`M`$ is a smoother, and `docs/math.md` section 9.2 records
that filtering one side of a comparison correlates the residuals over the filter's width
while the label likelihood stays diagonal, at a cost of $`(\sum_p k_p^2)^{-1}`$ in
chi-square, which on AI Phe predicted 4.91 against a measured 4.26 and drove both components
to the $`v\sin i`$ floor. The same statistic for these operators, the median over rows of
$`(\sum_{g'} M_{gg'}^2)^{-1}`$, runs from 6.8 on `gaia-47620265212420096` to 79.2 on
`gaia-51574864941234048`, median 20.4, which is four times the AI Phe kernel's. Measured,
chi-square rises under the patch by a factor 1.61 to 15.46, median 2.40, the bounded jitter
site absorbing the remainder: it is allowed a factor five in scale, hence twenty-five in
variance, and it saturates. A mis-specified likelihood of this size is not a small
correction sitting on top of a correct one, and the temperatures are free to go anywhere
within it. The rows of $`M`$ are also not unit-sum, their median row sum running from 0.002
on `gaia-51574864941234048` to 0.99 on `mixed-0012` and their median diagonal from 0.026 to
0.162, so the operator is a strong low-frequency shrinkage rather than a smoother that
preserves an area. Since an equivalent width is that low frequency, and since the fit holds
a fitted radius ratio that rescales every depth and an additive Chebyshev whose $`m=0`$ term
is a constant, the two nuisances can reabsorb most of what $`M`$ does to a model, and the
measurement says they do.

**Two things the patch does improve, and they are the two the nuisances were absorbing.**
The light fraction the fit reports moves from a median absolute error of 0.0277 to 0.0086,
against a floor of 0.0052 from the truth-fed fit, improving on twenty of the twenty-four
components and worsening on four. The gains are large where the native route had lost the
secondary: on `mixed-0017` B the reported fraction moves from 0.127 to 0.224 against a true
0.233, on `gaia-17170287112790656` B from 0.017 to 0.061 against 0.067, on `mixed-0008` B
from 0.085 to 0.122 against 0.112. It does not repair the worst case, `gaia-51574864941234048`
B, which moves from 0.0006 to 0.043 against a true 0.396. Rotation improves too, the median
absolute $`v\sin i`$ error falling from 7.65 to 3.44 km/s against a floor of 1.86, nineteen
improving and five worsening, and the systematic sign of the error weakening: the native
route over-estimates $`v\sin i`$ on twenty-one of twenty-four components with a median of
+7.65 km/s, and the patched route on fifteen with a median of +1.67 km/s. That is the
expected reading. The present route has no way to represent the smoothing the disentangling
applied, so it absorbs it into extra rotational broadening and into a light ratio that makes
the shallow secondary shallower still; the patched route represents it in the forward model
and gives both back. Metallicity moves a little, from a median 0.057 to 0.040 dex, and
surface gravity not at all, 0.1852 against 0.1854 dex.

**The cost is about half as much again per fit, and a third of a gigabyte.** Assembling
$`A^\top W A`$ and forming $`M`$ as a dense matrix of side 5748 to 7000 takes 6.0 to 9.9 s
per star when measured alone, median 7.2, and 6.6 to 17.4 s inside the fitting runs where the
machine was also carrying the fits; the matrix itself is 0.25 to 0.37 GiB in float64, and the
process peaked at 2.76 GiB against 0.78 GiB for the smallest. The label fit itself runs 65 to
150 s on the present route, median 98, and 132 to 251 s on the patched route, median 150, a
median factor 1.57. The extra is one dense matrix product of side about 6000 per model
evaluation and its transpose in the gradient, plus the compile cost of folding a constant of
that size into the traced step. An operator applied through the band matvec and the block
solve rather than as a dense matrix would be about five times cheaper in arithmetic, and was
not used here because the dense form is what makes the identity gate exact and trivially
differentiable.

**Recommendation.** `match_labels` should not be reshaped to compare through the coupled
operator, on this evidence. The change is well posed, it is cheap enough, and it is the right
forward model; it simply does not move the temperatures, and it cannot, because the two
parameters that would have to stay fixed for the operator to speak about temperature, the
dilution and the broadening, are the two the fit is free to move, and because the residual it
creates is correlated in a way the diagonal likelihood cannot carry. Neither of the two
failure modes anticipated in the work package is the one that occurred: the gain is not real
but swamped by the node grid, since the noise-free control shows the gain is 3 K and not a
swamped 100 K; and the patched fit is not systematically worse, since twelve improve and
twelve worsen. The temperatures are simply insensitive to it. If any part of this is worth
keeping it is the light fraction, where the operator recovers a median factor 3.2 of the
error and lands within a factor 1.6 of the floor, and `d63_light_fractions.md` records that
the flux ratio is what downstream cross-correlation is most sensitive to. That would be
worth revisiting as a light-ratio estimator rather than as a label fit.

**What is left.** The measurement holds the residual covariance diagonal, which is the
mis-specification named above, and the honest version of the proposal is not to filter the
model alone but to filter it and whiten the residual by the same posterior precision, that
is, to minimise $`r^\top (\Lambda + A^\top W A)\, r`$ with $`r = M\,m(\phi) - \hat d`$, which
costs one banded product against a factor the fit already builds. That is a change to the
likelihood rather than to the model rows and was out of scope here; whether it moves the
temperatures is open, and it is the only version of this idea the present result does not
close off. Second, the warm-start scan was left unpatched, so both routes begin from the same
node scan against the same archived component; that keeps the comparison fair but means the
patched fit was never offered a starting point consistent with its own forward model, and on
`mixed-0013` A, where the patched answer moved 67.7 K and chi-square rose by a factor 15.5,
a different basin cannot be ruled out. Third, the sample is twelve two-component products on
the RVS band at $`R = 11{,}500`$, where an operator row spreads a residual over an effective
6.8 to 79 stacked entries; at higher resolving power the components separate over more pixels
and the operator should narrow, and whether the temperature stays insensitive there has not
been measured.
