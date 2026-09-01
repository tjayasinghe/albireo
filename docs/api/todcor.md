# Epoch radial velocities (TODCOR)

One velocity per component per epoch, by correlating each observed spectrum against a
combination of templates with independent shifts — the two-dimensional correlation of
Zucker & Mazeh (1994), generalized to any number of components and to weighted, masked,
multi-instrument data.

This is the product the rest of the binary-star toolchain starts from, and the one thing the
joint fit upstream deliberately never makes. The two are complementary rather than
competing: disentangling infers the orbit and the spectra when the spectra are unknown;
this measures velocities epoch by epoch when they are known — from a library, a label match,
or the disentangling itself. `Fit.templates()` and `Fit.measure_velocities()` close that loop.

## What is different from the classic implementation

TODCOR is written for zero-mean spectra on a uniform log-wavelength grid with uniform
weights, so that the two-dimensional function can be assembled from three one-dimensional
correlations by FFT. albireo evaluates the same estimator as the **weighted least-squares fit
of the shifted templates to the observed pixels**, with the amplitudes — the light fractions —
either held or solved in closed form at every pair of shifts. On a uniform grid with uniform
weights the two are identical to 1e-10 (`tests/test_todcor.py` pins both the symmetric
expression and the fixed-ratio one). What the least-squares form buys:

- **masks, gaps, cosmics and per-pixel weights** cost nothing and change no formula
  (`ivar = 0` is the universal mask, as everywhere in albireo);
- **the data are never resampled** — the shifted templates are projected onto each epoch's
  own pixels, so mixed instruments and mixed samplings are handled natively;
- **the templates are intrinsic** and each instrument's LSF is applied to them per epoch,
  in quadrature above any resolution the template already carries;
- **the chi-square is exact at fractional shifts**, because the shift operator is linear in
  the template, so the sub-pixel minimum and its curvature are computed rather than
  interpolated;
- **errors are the maximum-likelihood ones** of Zucker (2003) — the curvature of the surface,
  rescaled by the reduced chi-square so the noise is measured rather than trusted — with
  the trusted-weights version alongside.

## Reading the table

`VelocityTable` carries the diagnostics that qualify a velocity, and `summary()` leads with
them: which components are **absolute** and which carry an unidentified zero point (a
disentangled template does, [§7.6](../math.md#76-free-per-epoch-velocities-the-rv-table));
which epochs are **blended** (the velocities sit on a ridge — a covariance correlation above
0.9); which sat **at the search edge**; the per-component **detection statistic**
$\Delta\chi^2$ (how much worse the fit gets without that component — small for a companion
the epoch does not actually see); and the Wilson slope, which is $-K_2/K_1$ and survives both
zero points.

The theory is [§10](../math.md#10-epoch-velocities-by-n-dimensional-correlation-d56d57);
the decision record is D56.

::: albireo.todcor
