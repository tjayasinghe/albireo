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

The free constant makes the search usable on clumped sampling. A classical periodogram fits
$`a \cos \omega t + b \sin \omega t`$ with the offset held at zero. This is harmless when the
epochs are spread evenly enough that the sampling window has no mean at the frequencies of
interest, but not for a survey cadence: with ten to twenty-five epochs falling into about
eleven visibility windows separated by hundreds of days, the constant the classical model
cannot fit is absorbed into the sinusoid and the spurious power buries the true period. Over the Gaia-like velocity tables of the
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

The default grid is anchored at its low end. Its step is $`1/(N T)`$; its points are
$`1/P_\mathrm{max} + k/(N T)`$ below the highest frequency, and the highest frequency
$`1/P_\mathrm{min}`$ is appended as the last point. $`N`$ is 10, raised where needed to the
smallest integer that gives at least 20,000 frequencies, computed with the frequency span
rounded down to two significant figures. The default shortest period is twice the
smallest gap between two epochs, so a grid spaced evenly between its two ends (the search's
grid before D65) moves every upper frequency whenever one epoch time moves. On a Gaia-like
table, with a smallest gap of 0.074 d over some 1500 d, a shift of $`10^{-6}`$ d in the
closest pair moves the high end by more than one step, and near-degenerate short-period
peaks change order. That is the size of the six-decimal rounding the velocity tables used to
be written with: the recorded peaks of five benchmark tables could only be reproduced by
moving the shortest period by 0.5 to $`4 \times 10^{-6}`$ d. The tables now carry every digit
of the epoch time, and the anchored grid guarantees the following. When an epoch other than
the first and the last moves, $`T`$ and the low end do not, so every point below both the old
and the new highest frequency stays where it was, bit for bit; the last point moves with the
high end, and points are added or removed only between the two, unless $`N`$ changes. $`N`$
is 10 on any table where $`10 T`$ times the span reaches 20,000, which includes every
multi-year survey table, and there does not depend on the high end. Below that, $`N`$ changes
only when the shift carries the span across a two-significant-figure boundary, or carries
$`20000/(T \times \mathrm{span})`$ across an integer. A shift of $`10^{-6}`$ d moves a span of
about 6.8 per day by about $`10^{-4}`$, so the case is rare but possible, and the whole grid is
then respaced. Moving the first or the last epoch changes $`T`$ and respaces every point by a
relative $`\delta T / T`$. An explicit `n_frequencies` keeps the evenly spaced grid it asks
for.

The anchored grid's peaks sit at slightly different frequencies from the evenly spaced
grid's, and a Keplerian started from them can converge elsewhere. Over the 33 blind tables of
the benchmark's third run, searched as the pipeline's bootstrap searches them, it leaves the
number of systems whose true period ranks first at 20 and reshuffles near-degenerate
candidates: one Gaia system from rank 39 to 11, one from 23 to 24, one from 3 to 4, and field
systems from 40 to 41 and from 2 to 6. On the last, the start nearest the true period now
converges at chi-square 587 instead of 460, which takes the truth out of the top four that
the pipeline's decision by the disentangling compares; that decision has converted no miss
into a hit on either population (D64). These moves are the cost of a grid that no longer
moves with the rounding of one epoch time.

`fit_rv_orbit` and `RVOrbit.predict` compile once per configuration and array shape.
Compiled at every call, they made the working set of a harness that fits every candidate
period of a table grow. With the fit's objective compiled once but `predict` evaluating the
Kepler solver's Newton loop eagerly, three tables through six search configurations took
269 s and reached a peak working set of 5061 MB; with both compiled once they take 26 s and
446 MB, and the whole 33-table rerun peaked at 697 MB.

**Which velocities enter.** A velocity enters the search and the fit where its own component
was measured: a finite velocity and error, off that component's search edge, at an epoch that
is not blended. The relative velocity needs both components. A component searched alone
(`components=[name]`), or fitted jointly with the others, keeps every epoch at which it was
measured, whatever happened to the others there. Neither function intersects this test with
`VelocityTable.good`, which requires every component to be measured; before D65 both did, so
the search on the primary alone, which exists for a companion the templates could not
follow, lost exactly the epochs at which that companion sat at the search edge.
`find_period` returns the epochs its series used under `used`.

In a joint fit a component with no more usable velocities than parameters of its own (one, its
semi-amplitude, or two where each component has its own systemic velocity) is held, and
`RVOrbit.held` names it. Its semi-amplitude stays at 1e-3 km/s, or at the caller's `k`, and its
own systemic velocity at its start. Neither is fitted or counted as a parameter, both have
`nan` errors, and its velocities carry no weight, so `residuals` and `rms` are `nan` for it.
`mass_ratio` is `None`, and `minimum_masses` and `projected_semiaxes` leave it out. Left free,
the optimizer does not keep an unconstrained parameter where it started: on a benchmark table
whose secondary was gated at every epoch the semi-amplitude ran to $`1.5 \times 10^7`$ km/s,
and the true period was set aside as outside the declared ranges.

**The semi-amplitude start.** Without `k`, `fit_rv_orbit` starts each component at half the
range of its own usable velocities, which is $`K`$ at any eccentricity once the phases are
covered. Where every component is usable this equals the start before D65, which took the
range over the epochs at which all components were. An undetected companion's draws
still set it (321 km/s on a benchmark system whose 7% secondary the templates never detected,
above the declared ceiling of 250 km/s); removing those velocities is the caller's decision,
and the pipeline's detection gate makes it. $`\sqrt{2}`$ times the weighted standard deviation
was tried in its place and rejected: sampled evenly in time it is 0.34 to 0.51 of $`K`$ at
$`e = 0.9`$, and it moved the chi-square ranking on six benchmark tables whose components were
all usable. The only bound the fit places on a semi-amplitude is zero, and the start is held
at or above 1e-3 km/s.

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
an alias outranked the truth on the periodogram in 10 systems of 32. Many candidates should
therefore be proposed and the fit left to decide, as the
[pipeline's search route](pipeline.md) does. There is a floor: a table of ten or eleven
epochs whose true Keplerian already leaves a reduced chi-square above about five cannot be
searched by any statistic computed on it, since an alias then fits better than the truth,
and the correct output for such a table is a failure rather than a period.

Background and references: [science overview](../science.md).

::: albireo.rvorbit
