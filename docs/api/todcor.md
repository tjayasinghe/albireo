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
to it: the coarse pass covers it, and the full-resolution window that refines the coarse
minimum is kept inside it. The interval is stated in each template's own rest frame. A
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

## The velocity table

`VelocityTable` contains the diagnostics that qualify each velocity, and `summary()` begins
with them. They show which components are absolute and which have an unidentified zero
point (a disentangled template does;
[§7.6](../math.md#76-free-per-epoch-velocities-the-rv-table)). They show which epochs are
blended (the velocities lie on a ridge, with a covariance correlation above 0.9) and which
epochs were at the search edge and were therefore not measured. They include the
per-component detection statistic $`\Delta\chi^2`$ (the increase in chi-square when that
component is removed, small for a companion not detected at that epoch) and the Wilson
slope, which equals $`-K_2/K_1`$ and is independent of both zero points.

The theory is in
[§10](../math.md#10-epoch-velocities-by-n-dimensional-correlation).

Background and references: [science overview](../science.md).

::: albireo.todcor
