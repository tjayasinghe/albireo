# Epoch radial velocities (TODCOR)

One velocity per component per epoch, obtained by correlating each observed spectrum
against a combination of templates with independent shifts: the two-dimensional
correlation of Zucker & Mazeh (1994), generalized to any number of components and to
weighted, masked, multi-instrument data. The three- and four-component extensions (Zucker,
Torres & Mazeh 1995; Torres, Latham & Stefanik 2007) are the same block solve with a larger
normal matrix.

The velocity table is the input to the rest of the binary-star toolchain, and it is the one
product the joint fit upstream does not produce. The two are complementary: disentangling
infers the orbit and the spectra when the spectra are unknown; this module measures
velocities epoch by epoch once templates are available, from a library, a label match, or
the disentangling itself. `Fit.templates()` and `Fit.measure_velocities()` connect the two.

## Differences from the classic implementation

TODCOR is formulated for zero-mean spectra on a uniform log-wavelength grid with uniform
weights, so that the two-dimensional function can be assembled from three one-dimensional
correlations by FFT. albireo evaluates the same estimator as the weighted least-squares fit
of the shifted templates to the observed pixels, with the amplitudes (the light fractions)
either held fixed or solved in closed form at every pair of shifts. On a uniform grid with
uniform weights the two agree to 1e-10 (`tests/test_todcor.py` pins both the symmetric
expression and the fixed-ratio one). The least-squares form has these properties:

- **Masks, gaps, cosmic rays and per-pixel weights** enter through the weights and change
  no formula (`ivar = 0` is the universal mask, as elsewhere in albireo).
- **The data are never resampled.** The shifted templates are projected onto each epoch's
  own pixels, so mixed instruments and mixed samplings are handled natively.
- **The templates are intrinsic.** Each instrument's LSF is applied to them per epoch, in
  quadrature above any resolution the template already carries.
- **The chi-square is exact at fractional shifts**, because the shift operator is linear in
  the template, so the sub-pixel minimum and its curvature are computed rather than
  interpolated.
- **Errors are the maximum-likelihood errors of Zucker (2003)**: the curvature of the
  surface rescaled by the reduced chi-square, so that the noise level is measured from the
  residuals, with the trusted-weights version reported alongside.

## The velocity table

`VelocityTable` carries the diagnostics that qualify each velocity, and `summary()` leads
with them: which components are absolute and which carry an unidentified zero point (a
disentangled template does; [§7.6](../math.md#76-free-per-epoch-velocities-the-rv-table));
which epochs are blended (the velocities lie on a ridge, with a covariance correlation
above 0.9); which epochs sat at the search edge; the per-component detection statistic
$\Delta\chi^2$ (the increase in chi-square when that component is removed, small for a
companion the epoch does not detect); and the Wilson slope, which equals $-K_2/K_1$ and is
independent of both zero points.

The theory is in
[§10](../math.md#10-epoch-velocities-by-n-dimensional-correlation-d56d57); the decision
record is D56.

Background and references: [science overview](../science.md).

::: albireo.todcor
