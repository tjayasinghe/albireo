# Observing-strategy forecasts

The posterior covariance of the component spectra contains no fluxes — only the epochs,
their phases, the weights, the masks, the line-spread functions, the light fractions and
the prior. So it can be computed for observations that have not been taken, which is what
turns "will twelve more nights at these phases separate the two stars?" into a question
with an exact answer instead of an argument.

`sensitivity_forecast` reports it three ways: the pointwise band each component would come
back with, the worst-determined *modes* of the covariance — the spectral patterns the
design cannot pin down — and the number of spectral degrees of freedom the data would
actually constrain. Each is quoted against the same quantity under the prior alone, so a
design that is learning nothing says so rather than returning a confident-looking number
it inherited from the regularizer.

What it does not do is forecast the orbit. The Fisher information for a velocity runs
through the derivative of the component spectrum, so an error bar on \(K_2\) needs the
line depths — which is exactly what has not been measured yet. The asymmetry is real and
is the reason this page is about spectra.

The theory the numbers are read against is [§5.1](../math.md#51-the-low-frequency-degeneracy-the-undulations-theorem)
and [§5.5](../math.md#55-forecasting-a-design-d47); the honest caveats are in the module
docstring below and are worth reading before quoting a number in a proposal.

::: albireo.forecast
