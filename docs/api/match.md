# Stellar labels for template selection

Fits Teff, log g, [M/H] and *v* sin *i* to disentangled component spectra against a
published synthetic grid, so that each component can be rendered as a template for
measuring epoch radial velocities in TODCOR, saphires, iSpec or a survey pipeline.

The module performs template selection, not stellar synthesis, and that distinction sets
its scope. It synthesizes no spectrum, carries no line list, solves no radiative transfer
and fits no individual abundances. Questions that require those are addressed through
[`albireo.handoff`](handoff.md) and GSSP, iSpec, Korg.jl or PySME.

Its purpose is the preceding step: choosing the right template, pinning the per-component
velocity zero point, and checking an assumed flux ratio. This scope is what the literature
supports. A wrong template mostly injects a constant velocity offset per component rather
than degrading precision, and the accuracy required before a template stops limiting the
velocities is loose: Teff to 2–3%, log g and [M/H] to 0.15 dex, *v* sin *i* to 10%
([§9.6](../math.md#96-the-accuracy-this-has-to-reach)). Labels from this mode are template
coordinates. A label for an abundance table is a different measurement with a different
error budget.

## Model choices

**Dilution is fitted.** Disentangling returns components scaled by light fractions that
were assumed, and the likelihood saw only the products, so an error in the light fractions
rescales every line depth and is indistinguishable from a temperature error. Both
components are therefore fitted together through one shared radius ratio, with
wavelength-dependent light fractions that sum to one at every pixel by construction: the
parameterization of GSSP's binary mode (Tkachenko 2015). The resulting spectroscopic light
ratio is a product in its own right; downstream cross-correlation codes are far more
sensitive to a wrong flux ratio than to a wrong temperature.

**The unconstrained zero point is modelled and reported.** Each component's constant
offset lies in the null space of the disentangling problem
([§5.1](../math.md#51-the-low-frequency-degeneracy-the-undulations-theorem)) and is
constrained only by the smoothness ridge. Left unmodelled, it shifts the line depths and
returns a biased Teff. The additive Chebyshev nuisance absorbs it, and its zeroth term is
that zero point, which is fitted and then printed.

**Uncertainties are quoted twice.** The Laplace error is the curvature at the optimum,
which is not the relevant quantity on correlated residuals; every code that has checked
finds the formal errors optimistic by five to ten times (Czekala et al. 2015; Gebruers et
al. 2022). `refit_draws` therefore refits the labels once per joint posterior draw of the
component spectra, and `summary()` prints both numbers side by side with the ratio between
them.

## The summary report

`LabelMatch.summary()` leads with the caveats, which are part of the result. Every number is
quoted against a null: the chi-square against a fit with no template and against the best
raw grid node, each label's posterior width against its prior width, and the formal error
against the spread of the draws. A Teff–log g correlation near 0.98 with both free is
expected and is labelled as such; the remedy is to fix log g from the eclipsing solution,
not to discount the number.

The theory is in [§9](../math.md#9-stellar-labels-from-disentangled-components).

Background and references: [science overview](../science.md).

::: albireo.match
