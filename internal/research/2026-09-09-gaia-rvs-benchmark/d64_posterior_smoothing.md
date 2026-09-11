# Is the disentangled component a smoothed copy of the truth? (D64, 2026-09-11)

Seventeen archived star-and-tier products, fourteen from the third run (`gaia5`, `field5`)
and three from the second (`field3`), were measured against the hypothesis that the
posterior-mean component spectrum is the injected component passed through the stationary
Whittaker-Henderson filter

```math
H_j(f) \;=\; \frac{w_j}{w_j \;+\; \eta_j \;+\; \tau_j\,(2 - 2\cos 2\pi f)^2},
```

with $`f`$ in cycles per model pixel. Each dataset was re-simulated from the population
record at the manifest's own seed (the record reproduced to four significant figures in S/N
per epoch, epoch count, lag-one correlation and baseline), the pipeline's declaration was
rebuilt from the archived `result.json` (grid pixel count, wavelength limits and velocity
budget all reproduced exactly), and the marginal was evaluated at the archived MAP
parameters read from `fit.npz`. That evaluation returns the archived `spectrum_A.txt` and
`spectrum_B.txt` to 5e-10, which is the rounding of the saved file, so nothing below rests
on a re-fit. The sample spans $`\tau_j/w_j`$ from 0.22 to 6.1e4 and carries the three
systems named in `d63_table_losses.md`. Scripts, per-star JSON and the filtered spectra:
the session scratchpad `wp-v/`.

**The data weight is the diagonal of the component's own block of $`A^\top W A`$, it runs
from 0.61 to 2.0e4 across the sample, and it carries the light fraction as $`l_j^2`$.** It
was computed rather than guessed: `assembly.band_block_tridiagonal` was called on the
problem at the archived MAP with `SmoothnessPrior(tau=0, eta=0)`, which returns
$`A^\top W A`$ alone in the interleaved block storage, and $`w_j(q)`$ is entry
$`(qn_c + j,\,qn_c + j)`$ of it, the sum over epochs of
$`l_{je}^2\,\mathrm{diag}\!\left(T^\top K^\top R^\top W'_e R\,K\,T\right)`$ at that epoch's
shift. The light enters exactly as the square: $`w_A/w_B`$ matches
$`(l_A/l_B)^2`$ to within 1.4 percent on every one of the seventeen runs, over ratios from
1.07 to 336. This settles how the light normalisation enters the comparison. Under the
invariance $`(l,\tau,\eta)\to(cl,c^2\tau,c^2\eta)`$ established earlier, $`w_j`$ scales as
$`c^2`$ as well, so $`\tau_j/w_j`$ and $`\eta_j/w_j`$ are invariant and $`H_j`$ is a
well-defined function of the fit however the light was declared. The median used below is
taken over the pixels the data cover; the 16th to 84th percentile spread of $`w_j`$ across
those pixels is about a factor 1.6, from the epoch sampling. The exact banded solve
$`(\mathrm{diag}(w_j) + \tau_j D_2^\top D_2 + \eta_j I)\,x = \mathrm{diag}(w_j)\,y`$, which
carries that variation and makes no stationarity assumption, and the FFT multiply by
$`H_j`$ with a scalar $`w_j`$ differ by at most 0.15 in normalised flux, all of it within a
few pixels of the grid edges where the FFT wraps, and by at most 0.00033 in the line-depth
statistic used below. The stationarity approximation is therefore not what breaks.

**The hypothesis fails, and it fails hardest exactly where it predicts most.** At one
resolution element, $`f_{\rm LSF} = 1/\mathrm{FWHM}_{\rm px} = 0.1151`$ cycles per pixel,
the predicted transfer runs from 0.95 down to 6.5e-5 over the sample. On
`gaia-17170287112790656__oracle` A, where $`\tau_j/w_j`$ is 6.1e4 and $`H_j(f_{\rm LSF})`$
is 6.5e-5, the archived posterior mean has lost no depth at all: it carries 1.049 times the
injected standard deviation over 8552 to 8654 A and 1.142 times the injected equivalent
width. On `mixed-0013__oracle` A, at $`\tau_j/w_j`$ 5620 and a predicted transfer of 7.1e-4,
it carries 1.090 and 2.336 times. Over the whole sample the filtered truth is
shallower than the archived component in the standard deviation of the normalised flux by a
median factor 0.808 (range 0.023 to 0.922), so the filter removes depth the disentangling
did not remove. Measured against the archived component, $`H_j y`$ lowers the residual
root-mean-square by a median of only 2.5 percent relative to the unfiltered truth over the
sixteen components the archive actually recovered (correlation above 0.8), and it is worse
than the unfiltered truth on six of those sixteen. The decisive measurement is the transfer
itself: estimated band by band by least squares from the archived and injected spectra over
the covered pixels, it exceeds unity below 0.01 cycles per pixel on 16 of the 34 components,
which no filter of this family can do, since $`H_j(0) = w_j/(w_j+\eta_j) \le 1`$; and it is
negative on 13 of the 34 between 0.02 and 0.04 cycles per pixel, on 7 between 0.04 and 0.09,
and on 23 above 0.09, reaching -0.92 on `mixed-0012__oracle` A. A non-negative filter cannot
produce a negative transfer either.

**The diagonal of $`A^\top W A`$ is a frequency average, and the frequencies a line
occupies are not near it.** Read as a Toeplitz operator away from the grid edges, the
component's own block has a symbol that at zero frequency is 10.93 to 12.64 times its own
diagonal (median 11.91, near constant because it is set by the line-spread function and the
grid pixel, which are the same on every star), and at $`f_{\rm LSF}`$ is 0.0133 to 0.0155
of it. The symbol crosses its own diagonal at 0.0716 to 0.0736 cycles per pixel across the
whole sample, a period of 13.6 to 14.0 model pixels, whereas the Fourier amplitude of a
resolved line at this sampling is down to half its peak by 0.0508 cycles per pixel, a
period of 19.7 pixels. The line therefore lives almost entirely on the low-frequency side
of the crossing, where the scalar weight understates the true one by up to a factor 12 and
the filter consequently
smooths harder than the disentangling did. Substituting the true symbol for the scalar
moves the half-power point of the filter from a period of 19.3 to 16.1 model pixels on
`mixed-0008__oracle` A and from 77 to 37 on its secondary, and the resulting filter,
applied as an exact banded solve against $`M_{jj}`$, reproduces the archived depth better
than the scalar version does (median ratio 0.885 against 0.808) while still not reproducing
it. At one resolution element the two versions run the other way, the symbol being the
smaller: $`H_j(f_{\rm LSF})`$ falls by a further factor 4.7 to 74 (median 66).

**No per-component filter of any kind can move an equivalent width, and the equivalent
width is what the label fit reads as a temperature.** The equivalent width is the zero
frequency, where the curvature penalty is identically zero, so the transfer there is
$`w_j/(w_j+\eta_j)`$ and the smoothness $`\tau_j`$ does not enter. Measured: the ratio of
the filtered to the injected equivalent width equals $`w_j/(w_j+\eta_j)`$ to within 0.8
percent on all 34 components, over values of $`\eta_j/w_j`$ from 2.4e-4 to 25. Of the 31
components whose archived equivalent width departs from the injected one by more than 0.1 A,
the stationary filter accounts for a median 0.003 of that departure and the exact
per-component banded filter for a median 0.000. The stationary filter accounts for more
than half on seven, and those seven are precisely the seven largest values of
$`\eta_j/w_j`$ in the sample, from 1.6 to 25: the ridge drives the filtered spectrum to
near zero, which happens to be where the archive put a component the fit had already lost.
The whole of the
equivalent-width error therefore lies in the off-diagonal blocks of $`A^\top W A`$, which
couple the two components.

**What the posterior mean actually is, exactly rather than approximately, is the coupled
operator applied to the truth plus a noise term.** With $`\Lambda`$ the stacked prior
precision and $`M = A^\top W A`$ the stacked data term, the posterior mean is
$`(\Lambda+M)^{-1}A^\top W z`$, which splits into

```math
\hat d \;=\; (\Lambda + M)^{-1} M\, y_{\rm true} \;+\; (\Lambda + M)^{-1} A^\top W n ,
```

and the first term was evaluated directly from the assembled band and the marginal's own
Cholesky factor. It accounts for a median 0.952 of the archived equivalent width's
departure from the truth, with a 16th to 84th percentile range of 0.790 to 0.991, and for
more than half of it on 28 of the 31 components, including every sign reversal: on
`mixed-0008__oracle` B the injected equivalent width is +1.807 A, the archive reports
-0.578 A, the coupled operator predicts -0.292 A, and the per-component filters predict
+1.134 A and +1.719 A. It lowers
the residual root-mean-square against the archive on all 34 components, by a median factor
0.711 over the recovered ones. What it leaves is the second term, whose standard deviation
over the metal window is 0.22 to 4.95 times the fit's own reported band; the cases above
one are the same systems the benchmark already flags with a `pull_rms` of 2 to 8, so the
unexplained part is the known under-reporting of the posterior band rather than a second
mechanism.

**Re-read with that split, the line-depth table of D63 is mostly a noise term.** The
standard deviation of the normalised flux over 8552 to 8654 A is the sum in quadrature of
the filtered signal and the propagated noise, and the two terms move differently when
$`\tau`$ moves. Third run over second run:

| case | measured | $`H_j`$ | true $`M_{jj}`$ | coupled (signal term) | noise term | $`\tau`$ |
| --- | --- | --- | --- | --- | --- | --- |
| mixed-0004 A | 1.000 | 1.000 | 1.000 | 0.999 | 1.001 | 4402 -> 4361 |
| mixed-0004 B | 0.905 | 0.944 | 0.935 | 0.956 | 0.898 | 667 -> 12950 |
| mixed-0008 A | 0.797 | 0.916 | 0.952 | 0.956 | 0.572 | 808 -> 35970 |
| mixed-0008 B | 0.753 | 0.660 | 0.885 | 0.875 | 0.766 | 397 -> 1980 |
| mixed-0015 A | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 71230 -> 71090 |
| mixed-0015 B | 0.996 | 0.994 | 0.997 | 0.995 | 0.996 | 293 -> 288 |

The two cases where $`\tau`$ did not move are predicted by every model, which is no
achievement. On the three where it did, the stationary filter recovers between 41 and 62
percent of the departure from unity and gets the sign right, which is why the hypothesis
looked plausible. The split says what the recovered fraction is made of. On `mixed-0008` A
the filtered signal fell only to 0.956 while the noise the posterior mean carries fell to
0.572, so almost the whole of the measured 0.797 is the smoothing removing noise rather
than line; the equivalent width over the same window and the same pair of runs moved by
1.021, which is to say not at all. What `d63_table_losses.md` recorded as a template losing
depth is a template carrying less of the epoch noise, which is why the same note found that
what such a template needs is a larger amplitude than the true light fraction rather than
the true one.

**Run through the label fit, the two pictures give different and sometimes opposite
answers.** The pipeline's label stage was run unchanged (same grid, priors, light
fractions, per-pixel band, dilution model, scan and step budget) against four inputs: the
archived posterior mean, the injected truth, the truth passed through $`H_j`$, and the truth
passed through the coupled operator. Recovered Teff minus injected Teff, in K:

| case | archive | truth | $`H_j y`$ | coupled |
| --- | --- | --- | --- | --- |
| mixed-0008 oracle A | +99.5 | -167.2 | +99.2 | +113.8 |
| mixed-0008 oracle B | +183.2 | -64.1 | +182.1 | +183.4 |
| mixed-0015 oracle A | -188.2 | +104.4 | +105.8 | -187.3 |
| mixed-0015 oracle B | +232.5 | +233.8 | +232.0 | +232.2 |
| mixed-0003 oracle A | -80.4 | -85.0 | -84.0 | -85.4 |
| mixed-0003 oracle B | -237.3 | -234.0 | +24.8 | -234.3 |
| mixed-0008 orbit A | -334.1 | -301.6 | +106.8 | -143.3 |
| mixed-0008 orbit B | +2432.1 | +40.8 | +2432.1 | +2432.1 |

The archived column reproduces the archived labels to the last digit, as it must. The
coupled operator reproduces the archive on seven of the eight, four of them to better than
2 K, and misses only `mixed-0008 orbit` A, where the orbit itself is wrong. The stationary
filter reproduces the archive on five and fails on three, and on two of those three it does
not reproduce the truth either. On `mixed-0015` A it returns the truth's +106 K while the
archive is at -188 K, an error the coupled operator reproduces to 1 K; on `mixed-0003` B,
where the archive is already at the truth's own -234 K, it returns +25 K. Since the fit is
a least-squares comparison of a model against a spectrum, filtering the model by $`H_j`$
shifts the recovered temperature by about as much as filtering the spectrum does and in the
opposite sense, so the correction would remove the 100 to 183 K errors of `mixed-0008
oracle` and carry `mixed-0003` B from -237 K to roughly -496 K. It is not a safe
correction. Five of the eight rows also show that the label fit does not recover the truth
even from the injected component, the errors in the truth column reaching 302 K, so part of
what D63 read as a disentangling failure is the label fit's own floor against a 250 K node
grid.

**Verdict.** The linear-filter picture is exactly right and the closed form is wrong. The
posterior mean is a linear map of the truth plus a noise term, and the map is
$`(\Lambda + A^\top W A)^{-1} A^\top W A`$ on the stacked pair; but its per-component,
stationary, scalar-weight approximation is wrong in two independent ways, each large. It
replaces the banded $`M_{jj}`$ by its diagonal, which is the average of the symbol over
frequency rather than its value anywhere the line sits, and so predicts a transfer of 6.5e-5
at one resolution element on a component that lost no depth at all; and it drops the
off-diagonal blocks,
which is where the whole of the equivalent-width error lives. The correct statement of what
ML-II did between the second and third runs is that it removed noise from the posterior
mean, not line: on the worst case the signal lost 4 percent of its depth and the noise
carried lost 43 percent. Neither tightening the hyperprior on $`\tau`$ nor filtering the
synthetic model through $`H_j`$ addresses the temperatures, because the temperatures move
with a quantity $`\tau`$ cannot touch.

**What is left.** The coupled operator is cheap to apply, one banded matrix-vector product
and one solve against a factor the fit already builds, so it is available as a
forward-model correction for the label fit: compare the pair of synthetic models passed
through $`(\Lambda + A^\top W A)^{-1} A^\top W A`$ against the pair of disentangled
components jointly, rather than each component against its own. That is a different shape
of label fit from the present one, which treats the components as independent apart from
the shared dilution, and it was not tried here. Second, the unexplained residual exceeds the
fit's own reported band by up to a factor 5 on the high-signal systems, which is the same
discrepancy the benchmark's `pull_rms` records and which was not diagnosed here; whether it
is the propagated data noise with an under-reported band, or the difference between the
library's own resolving power and the infinite-resolution spectrum the disentangling solves
for, is open. Third, the sample is fourteen third-run products and three second-run ones,
all two-component and all on the RVS band; whether the off-diagonal term stays dominant at
higher resolving power, where the components separate over more pixels, has not been
measured.
