# Orbits from velocity tables

The classical radial-velocity orbit: a Keplerian fitted by weighted least squares to a
[velocity table](todcor.md), started from a Lomb-Scargle period search. It uses the same
Kepler solver and the same angle conventions as the joint model, so an orbit fitted this
way and one inferred directly from the spectra can be compared element for element.

Two conventions apply to every result. The systemic velocity is fitted once per component
whenever a component's velocities are differential: a table built from disentangled
templates carries one unidentified zero point per star, and a shared $\gamma$ would absorb
two different constants and bias both semi-amplitudes. `RVOrbit.gamma_mode` records which
was done. The quoted errors are the curvature errors rescaled by the reduced chi-square, as
in other orbit codes, because the per-epoch errors of a template fit do not include
template mismatch.

`RVOrbit.to_theta()` returns the elements in the form accepted by `Disentangler(orbit=...)`
and the low-level priors, which supports the reverse direction: velocities are measured
against a library template, the orbit is fitted, and a disentangling is warm-started from
it.

Background and references: [science overview](../science.md).

::: albireo.rvorbit
