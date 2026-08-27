# Stellar labels for template selection

Fits Teff, log g, [M/H] and *v* sin *i* to disentangled component spectra, against a published
synthetic grid, so each component can be rendered as a template for measuring epoch radial
velocities in TODCOR, saphires, iSpec or a survey pipeline.

This is template selection, not stellar synthesis, and the distinction is the whole scope. It
synthesizes no spectrum, carries no line list, solves no radiative transfer and fits no
individual abundances. The moment a question needs those, the answer is still
[`albireo.handoff`](handoff.md) and GSSP, iSpec, Korg.jl or PySME.

What it is *for* is the front-half job: choosing the right template, pinning the per-component
velocity zero point, and checking an assumed flux ratio. Framing it that way is not modesty —
it is what the literature supports. A wrong template mostly injects a constant velocity offset
per component rather than degrading precision, and the accuracy needed before a template stops
limiting the RVs is loose: Teff to 2–3%, log g and [M/H] to 0.15 dex, *v* sin *i* to 10%
([§9.6](../math.md#96-the-accuracy-this-has-to-reach)). Labels from this mode are template
coordinates. A label for a paper's abundance table is a different measurement with a different
error budget.

## The three things it gets right on purpose

**Dilution is fitted, not assumed away.** Disentangling returns components scaled by light
fractions that were *assumed*, and the likelihood only ever saw the products, so an error there
rescales every line depth and looks exactly like a temperature error. Both components are
therefore fitted together through one shared radius ratio, with wavelength-dependent light
fractions that sum to one at every pixel by construction — GSSP's binary-mode parameterization.
The spectroscopic light ratio that falls out is a deliverable in its own right; downstream
cross-correlation codes are far more sensitive to a wrong flux ratio than to a wrong
temperature.

**The unconstrained zero point is modelled and reported.** Each component's constant offset is
in the null space of the disentangling problem ([§5.1](../math.md#51-the-low-frequency-degeneracy-the-undulations-theorem))
and is held only by the smoothness ridge. Left alone it lands on the line depths and comes back
as a bogus Teff. The additive Chebyshev nuisance absorbs it, and its zeroth term is that zero
point — fitted, then printed, never silently swallowed.

**Uncertainties are quoted twice.** The Laplace error is the curvature at the optimum, which is
the wrong question on correlated residuals; every code that has checked finds formal errors
optimistic by five to ten times. So `refit_draws` refits the labels once per joint posterior
draw of the component spectra, and `summary()` prints both numbers side by side with the ratio
between them.

## Reading the report

`LabelMatch.summary()` leads with the caveats because the caveats are the finding. Every number
is quoted against a null: the chi-square against a fit with no template at all and against the
best raw grid node, each label's posterior width against its prior width, and the formal error
against the draws spread. A Teff–log g correlation near 0.98 with both free is *expected* and
labelled as such — the remedy is to fix log g from the eclipsing solution, not to distrust the
number.

The theory is [§9](../math.md#9-stellar-labels-from-disentangled-components-d52d55); the
decision record is D53.

::: albireo.match
