# Orbits from velocity tables

The classic radial-velocity orbit: a Keplerian fitted by weighted least squares to a
[velocity table](todcor.md), with a Lomb-Scargle period search to start it from. It uses the
same Kepler solver and the same angle conventions as the joint model, so an orbit fitted this
way and one inferred from the spectra directly can be compared element for element.

Two conventions are worth knowing before reading a result. The systemic velocity is fitted
**once per component** whenever a component's velocities are differential — a table built from
disentangled templates carries one unidentified zero point per star, and a shared $\gamma$
would then absorb two different constants and bias both semi-amplitudes; `RVOrbit.gamma_mode`
says which was done. And the quoted errors are the curvature errors rescaled by the reduced
chi-square, as every orbit code does, because the per-epoch errors of a template fit never
include template mismatch.

`RVOrbit.to_theta()` returns the elements in the form `Disentangler(orbit=...)` and the
low-level priors take — the loop closing in the other direction: measure velocities against a
library template, fit the orbit, and warm-start a disentangling from it.

::: albireo.rvorbit
