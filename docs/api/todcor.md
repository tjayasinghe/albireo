# Epoch radial velocities (TODCOR)

This module measures one velocity per component per epoch by correlating each observed
spectrum against a combination of templates with independent shifts. The method is the
two-dimensional correlation of Zucker & Mazeh (1994), generalized to any number of
components and to weighted, masked, multi-instrument data. The three- and four-component
extensions (Zucker, Torres & Mazeh 1995; Torres, Latham & Stefanik 2007) are the same block
solve with a larger normal matrix.

The velocity table is the input to the rest of the binary-star toolchain, and it is the one
product the joint fit upstream does not produce. The two are complementary. Disentangling
infers the orbit and the spectra when the spectra are unknown. This module measures
velocities epoch by epoch once templates are available, from a library, a label match, or
the disentangling. `Fit.templates()` and `Fit.measure_velocities()` connect the two.

## Differences from the classic implementation

TODCOR is formulated for zero-mean spectra on a uniform log-wavelength grid with uniform
weights, so that the two-dimensional function can be assembled from three one-dimensional
correlations by FFT. albireo evaluates the same estimator as the weighted least-squares fit
of the shifted templates to the observed pixels, with the amplitudes (the light fractions)
either held fixed or solved in closed form at every pair of shifts. On a uniform grid with
uniform weights the two agree to 1e-10 (`tests/test_todcor.py` checks both the symmetric
expression and the fixed-ratio one). The least-squares form has these properties:

- **Masks, gaps, cosmic rays and per-pixel weights** enter through the weights and change
  no formula (`ivar = 0` is the universal mask, as elsewhere in albireo).
- **The data are never resampled.** The shifted templates are projected onto each epoch's
  own pixels, so mixed instruments and mixed samplings need no special treatment.
- **The templates are intrinsic.** Each instrument's LSF is applied to them per epoch, in
  quadrature above any resolution the template already has.
- **The chi-square is exact at fractional shifts**, because the shift operator is linear in
  the template, so the sub-pixel minimum and its curvature are computed rather than
  interpolated.
- **Errors are the maximum-likelihood errors of Zucker (2003)**: the curvature of the
  surface rescaled by the reduced chi-square, so that the noise level is measured from the
  residuals. The version that takes the weights as given is reported alongside.
- **A declared noise correlation widens the errors.** Resampled spectra have correlated
  pixel noise, which does not change the estimator and makes the curvature error too small.
  `noise_correlation` (one lag-one coefficient, or one per instrument) replaces the
  curvature by the sandwich of [§10.4](../math.md#104-uncertainties-and-detection). The
  `Disentangler` interface passes the value it was declared with, and the pipeline passes
  its `noise_correlation` setting.

## The search window

`v_range` is the interval of velocity the correlation searches, and the search is confined
to it: the coarse pass covers it, and the full-resolution windows that refine its minima
are kept inside it. The interval is stated in each template's own rest frame. A
template that has a zero point (`v_zero_kms`, which a label match measures) has that zero
point composed into every velocity reported against it. One shared `(lo, hi)` therefore
searches a different interval of reported velocity for every component whose zero point
differs. The window must then be declared per template. `v_range` accepts one pair per
template as well as one shared pair.

`Fit.measure_velocities()` builds the windows in this way. The common interval is the span
of the fitted velocities widened by 40 km/s at each end. Each template's window is that
interval shifted by the template's zero point relative to the median zero point, so that
every component is searched over the same reported velocities. Where the zero points differ
by more than the span of the fitted velocities, no interval contains every component's own
velocities, and the `Disentangler` interface raises an error that names the component.
This occurs when a label fit's frame-offset scan stopped at its bound. The reported zero
point is then a bound rather than a measurement, and one shared window built around it
does not contain the other component's velocities.

Where the chi-square is still decreasing at the end of the search interval, nothing has
been measured, because the last point evaluated is the edge of the search and not a
minimum. That component is flagged `at_edge`, and its `velocity`, `sigma` and `sigma_ivar`
are `nan`, both in the table and in the file that `write()` produces. The diagnostics of
the point that was evaluated are kept, because they show that the epoch was at an edge
rather than at a peak: `chi2`, `light`, `delta_chi2`, `r_squared`, the pixel count and the
curvature. `good` is false for such an epoch, and `summary()` counts these epochs as "*N*
at the search edge, not measured". The flag is set for a minimum on the coarse grid's first
or last node, and for one still on the boundary of the refinement window after that window
has moved as far as the range allows. The remedy is to widen `v_range`, unless it is
already wide, in which case the templates do not match the line positions in the data.

## Blended epochs

Where the lines of two components overlap, the chi-square has two minima: the solution, and
the pair with the same light-weighted mean velocity and the velocity difference of the
opposite sign ([§10.3](../math.md#103-exact-fractional-shifts-and-pixel-locking)). `todcor`
searches from both and returns the lower, whatever the stride of the coarse pass.

Noise can make the exchanged pair the lower. Both velocities are then wrong by about
their separation, and the quoted errors do not show it, because the curvature at that
minimum is regular. The search keeps every minimum it refines, and the table reports two
quantities from them.

- `margin` is the rise in chi-square from the minimum returned to the lowest other minimum
  at which the velocities differ. It is `inf` where the search found no such minimum.
- `blended` is raised where the margin is below 9 times the reduced chi-square (9 under
  `errors="ivar"`), as well as where the curvature shows a ridge. Such an epoch is not
  `good`, and the orbit fit and the period search leave it out. `second_minimum` marks the
  epochs flagged for this reason.

The velocities of two minima differ where, in every order of the components, some velocity
differs by more than three quoted errors. The same pair of velocities in the other order is
therefore not counted. Two alike stars have that minimum at every separation, about as deep
as the solution, so a flag that counted it would mark every epoch of such a pair. The pair
is measured there. Which velocity belongs to which star is decided by an orbit, with
[`assign_components`](rvorbit.md).

On simulated Gaia RVS epochs (a resolving power of 11,500, light fractions of 0.625 and
0.375, 480 epochs at each S/N) either velocity was more than five quoted errors and 3 km/s
from the injected one at 46, 17, 1 and 0 epochs for a S/N of 15, 40, 100 and 300 per pixel,
all with the lines 10 to 40 km/s apart, where the FWHM of the line-spread function is
27.5 km/s ([TODCOR on Gaia RVS spectra](../tutorials/gaia-rvs-todcor.ipynb)). The minimum
returned was the exchanged pair at 63 of these 64 epochs, and the injected pair was another
refined minimum at all of them. The margin flags 33 of the 64, and 96 epochs that were
measured correctly. The other 31 are the same pair in the two orders to within three quoted
errors, and with the two velocities interchanged 30 of them are within five quoted errors
or 3 km/s of the injected values.

Where the margin raises the flag, the table records the other minimum as well.
`alternative` has its velocities, `alternative_covariance` their covariance on the scale of
that minimum's own chi-square, and `alternative_sigma` the errors. At each of the 33
flagged wrong epochs above the recorded minimum is the injected pair, and at 95 of the 96
flagged correct ones it is a wrong pair. One epoch does not decide between the two, and an
orbit does. [`assign_by_orbit`](rvorbit.md#the-second-minimum-of-a-blended-epoch) takes,
at every such epoch, the minimum that a prediction and the spectrum together fit better,
and [`assign_components`](rvorbit.md) does so while it fits the orbit at a known period.
The epoch then loses the flag. Without an orbit it is left out.

One limit applies. The margin counts the minima the search refined. Where the lines are
closer than their width the two minima merge into one elongated minimum. Such an epoch is
flagged only if the correlation of the two velocities exceeds 0.9, and it has no other
minimum to record.

## Measured light fractions

`light="global"` solves the amplitudes at every epoch, takes the median over the epochs of
each instrument at which every amplitude is positive, the epoch is usable and every
component is detected, and measures again with that median held. Where no epoch
qualifies, the median is over the epochs with positive amplitudes.

An amplitude of the free solution can be negative. Two templates that are both cooler than
the stars reproduce a blend of weaker lines as a difference of the two, and a pair that is
never resolved then has a negative amplitude at every epoch. On one of the 33 systems of
the Gaia RVS benchmark (stars of 6950 and 6530 K whose lines are never more than 32 km/s
apart, against templates of 6000 and 5000 K) the amplitudes are 1.3 to 1.5 and -0.4 to
-0.6 at each of the 14 epochs. Such an epoch does not measure the light. Where no epoch
has positive amplitudes `todcor` raises an error that asks for the light fractions or for
templates closer to the components. `light="free"` returns the amplitudes as solved.

Before D68 the median was then taken over the epochs as they were. The held fractions of
that system were 1.52 and -0.52, and every epoch of its table was on a ridge. The search
before D66 returned at one of the 14 epochs a minimum 2.65 higher in chi-square, with
amplitudes of 0.74 and 0.14 and the second template 27 km/s from the systemic velocity,
and the light fractions of the whole table, 0.84 and 0.16 for an injected 0.60 and 0.40,
were those of that epoch.

## The velocity table

`VelocityTable` contains the diagnostics that qualify each velocity, and `summary()` begins
with them. They show which components are absolute and which have an unidentified zero
point (a disentangled template does;
[§7.6](../math.md#76-free-per-epoch-velocities-the-rv-table)). They show which epochs are
blended (the velocities lie on a ridge, with a covariance correlation above 0.9, or a
second minimum gives other velocities within the margin of the section above) and which
epochs were at the search edge and were therefore not measured. They include the
per-component detection statistic $`\Delta\chi^2`$ (the increase in chi-square when that
component is removed, small for a companion not detected at that epoch) and the Wilson
slope, which equals $`-K_2/K_1`$ and is independent of both zero points.

The theory is in
[§10](../math.md#10-epoch-velocities-by-n-dimensional-correlation).

Background and references: [science overview](../science.md).

::: albireo.todcor
