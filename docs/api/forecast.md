# Observing-strategy forecasts

The posterior covariance of the component spectra contains no fluxes; it depends only on
the epochs, their phases, the weights, the masks, the line-spread functions, the light
fractions and the prior. It can therefore be computed for observations that have not been
taken, so whether a planned set of epochs at given phases will separate two stars is
answerable before the observations are made.

`sensitivity_forecast` reports the result in three forms: the pointwise uncertainty band
each component would have, the worst-determined modes of the covariance (the spectral
patterns the design cannot constrain), and the number of spectral degrees of freedom the
data would constrain. Each is quoted against the same quantity under the prior alone, so a
design that adds no information is reported as such rather than returning a value
inherited from the regularizer.

The function does not forecast the orbit. The Fisher information for a velocity depends on
the derivative of the component spectrum, so an error bar on \(K_2\) requires the line
depths, which are what has not yet been measured. This asymmetry is why the forecast is
restricted to the spectra.

The theory is in [§5.1](../math.md#51-the-low-frequency-degeneracy-the-undulations-theorem)
and [§5.5](../math.md#55-forecasting-a-design-d47). The module docstring below states the
caveats that apply before a forecast is quoted in a proposal.

Background and references: [science overview](../science.md).

::: albireo.forecast
