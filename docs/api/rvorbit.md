# Orbits from velocity tables

The classical radial-velocity orbit: a Keplerian fitted by weighted least squares to a
[velocity table](todcor.md), started from a generalized Lomb-Scargle period search. It uses
the same Kepler solver and the same angle conventions as the joint model, so an orbit fitted
this way and one inferred directly from the spectra can be compared element for element.

Two conventions apply to every result. The systemic velocity is fitted once per component
whenever a component's velocities are differential: a table built from disentangled
templates carries one unidentified zero point per star, and a shared $`\gamma`$ would absorb
two different constants and bias both semi-amplitudes. `RVOrbit.gamma_mode` records which
was done. The quoted errors are the curvature errors rescaled by the reduced chi-square, as
in other orbit codes, because the per-epoch errors of a template fit do not include
template mismatch.

`RVOrbit.to_theta()` returns the elements in the form accepted by `Disentangler(orbit=...)`
and the low-level priors, which supports the reverse direction: velocities are measured
against a library template, the orbit is fitted, and a disentangling is warm-started from
it.

## The period search

`find_period` searches the difference of the first two components' velocities, which is free
of both systemic velocities and both template zero points and has amplitude $`K_1 + K_2`$; a
single-component table is searched as it is. The statistic is the floating-mean, weighted
generalized periodogram of Zechmeister & Kürster (2009), which fits a constant beside the
sinusoid at every frequency. With weights $`w_i = 1/\sigma_i^2`$ normalised to
$`W_i = w_i / \sum_j w_j`$, a weighted mean $`\bar y = \sum_i W_i y_i`$ and
$`YY = \sum_i W_i (y_i - \bar y)^2`$, write at each angular frequency $`\omega`$

```math
C = \sum_i W_i \cos \omega t_i, \qquad S = \sum_i W_i \sin \omega t_i,
```

```math
YC = \sum_i W_i y_i \cos \omega t_i - \bar y \, C, \qquad
YS = \sum_i W_i y_i \sin \omega t_i - \bar y \, S,
```

```math
CC = \sum_i W_i \cos^2 \omega t_i - C^2, \qquad
SS = \sum_i W_i \sin^2 \omega t_i - S^2, \qquad
CS = \sum_i W_i \cos \omega t_i \sin \omega t_i - C S.
```

The reported power is then

```math
p(\omega) = \frac{SS \, YC^2 + CC \, YS^2 - 2 \, CS \, YC \, YS}{YY \, (CC \, SS - CS^2)},
```

the fraction of the weighted variance that the sinusoid and the free constant remove
together. The weights enter through $`W_i`$ alone; the data are never rescaled by them.

The free constant is what makes the search usable on clumped sampling. A classical
periodogram fits $`a \cos \omega t + b \sin \omega t`$ with the offset held at zero, which is
harmless when the epochs are spread evenly enough that the sampling window has no mean at the
frequencies of interest, and is not harmless for a survey cadence: with ten to twenty-five
epochs falling into about eleven visibility windows separated by hundreds of days, the
constant the classical model cannot fit is absorbed into the sinusoid and the spurious power
buries the true period. Over the Gaia-like velocity tables of the
[D62 benchmark](../benchmarks.md) the true period is the highest peak of the classical
periodogram in 5 systems of 13 and of this one in 9 of 13.

The frequency grid is uniform in frequency, by default $`N_\mathrm{os} = 10`$ samples per
$`1/T`$ with $`T`$ the span of the epochs, so the step in period is

```math
\frac{\mathrm{d}P}{P} = \frac{P}{N_\mathrm{os} T},
```

0.1% at a hundredth of the baseline and 4.7% at 0.47 of it. A peak reported near a third of
the baseline is therefore good to a few per cent only, and it is the Keplerian fitted from it,
whose period bounds are $`0.5 P_0`$ to $`2 P_0`$, that pins the period down. Refining the grid
was measured at 10, 20 and 50 samples per $`1/T`$ and changed no outcome for the
single-sinusoid search.

`n_peaks` sets how many distinct peaks are reported, the best and `n_peaks - 1` aliases, two
peaks counting as distinct when their periods differ by more than 2%. With `n_harmonics = 2`
the model becomes $`c + a_1 \cos \omega t + b_1 \sin \omega t + a_2 \cos 2\omega t + b_2 \sin
2\omega t`$, fitted by weighted least squares at every frequency and reported as
$`1 - \chi^2/\chi^2_\mathrm{null}`$ on the same scale. An eccentric orbit's velocity curve is
not a sinusoid, and the second harmonic ranks its period higher: over the same tables it moved
two systems at $`e = 0.41`$ and $`0.47`$ from rank 10 to ranks 2 and 1, and one at
$`e = 0.67`$ from beyond the six hundredth peak to rank 1. It is worse on circular
orbits, where the extra freedom is spent on noise, so it belongs beside the one-harmonic
search as a second source of candidates rather than in place of it. `swap_invariant` searches
the magnitude of the difference, which two alike components exchanged between epochs leave
unchanged; its dominant peak sits at half the period for a circular orbit.

A peak is a starting point and not a period. The Keplerian fitted at a candidate uses the
shape of the curve and every velocity at once, and over the same tables it separated the truth
from the best alias by hundreds in chi-square wherever the truth was reachable at all, while
an alias outranked the truth on the periodogram in 10 systems of 32. Propose many candidates
and let the fit decide, which is what the [pipeline's search route](pipeline.md) does. There
is a floor to this: a table of ten or eleven epochs whose true Keplerian already leaves a
reduced chi-square above about five cannot be searched by any statistic computed on it, since
an alias then fits better than the truth, and the honest output for such a table is a failure
rather than a period.

Background and references: [science overview](../science.md).

::: albireo.rvorbit
