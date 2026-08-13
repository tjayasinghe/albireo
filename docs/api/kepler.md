# Kepler solver

A differentiable Kepler solver: fixed-iteration Newton wrapped in a `custom_jvp` that
supplies the implicit-function tangent, so the gradient is exact rather than
backpropagated through the iteration. Verified to e = 0.95.

::: albireo.kepler
