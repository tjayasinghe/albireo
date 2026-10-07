# Downstream handoff

The disentangled spectrum is rarely the final result. It is the input to an atmosphere
code (GSSP, iSpec, Korg.jl, PySME) that derives an effective temperature, a surface gravity
and abundances from it. albireo covers the disentangling step and does not attempt the
atmospheric analysis. The interface between the two is a file that the atmosphere code
reads without manual editing.

`write_gssp` and `write_ispec` write those files, following each code's own documentation.
The two formats differ in almost every detail in which a mistake produces no error. GSSP
requires two columns in ångström on an equidistant grid. iSpec requires three tab-separated
columns in nanometres with an absolute 1σ error and performs no unit conversion. A
log-wavelength grid written directly into a GSSP file produces no error, and GSSP sets its
synthetic step from the first pixel pair.

## Posterior draws

GSSP accepts no per-pixel uncertainty. Appendix B of Tkachenko (2015), which serves as the
manual, specifies a two-column file, and the configuration files for all three of its
modules contain no error path, no signal-to-noise entry and no weighting entry. Its quoted
error bars come from χ² on the fit residuals.

The posterior band therefore cannot be propagated to an effective temperature through the
file. It can be propagated only through repeated fits, which is the purpose of
`export_draws`. *N* draws from the joint posterior are written, all *N* are fitted with
identical settings, and the spread of the resulting parameters is taken as the
disentangling contribution. This is the term the literature currently omits, as its
authors state:

> the uncertainties that could arise from the normalisation procedure are not taken into
> account in the global uncertainties on the presented properties
>
> (Mahy et al. 2020, TMBM III, §3.1)

This procedure is not new: Kiran et al. (2016, §3.5) added Gaussian noise to a
disentangled profile, refitted 500 times and took the scatter. albireo differs in what is
drawn. Its draws are `d_hat + L⁻ᵀz` on the vector stacked over all components, so they are
correlated across wavelength and across the two stars, and draw *i* of component A pairs
with draw *i* of component B. White-noise injection assumes the error is independent from
pixel to pixel. Disentangling error is not, because the problem has a low-frequency null
space, and the low-frequency part of the error shifts the continuum and therefore the
temperature.

## Contributions outside the spread

The atmosphere code's own model error (grid coarseness, LTE, line lists) is outside
albireo's posterior. So is anything albireo conditions on rather than marginalizes. The
light fractions are conditioned on: they are assumed, not inferred, and the marginal
likelihood is flat in them under constant light. This is the systematic that Pavlovski &
Hensberge (2011) identify as the dominant one, and the draw spread contains no information
about it. Finally, iSpec's own `errors['teff']` is a within-draw fit error computed from
the same band, so adding it in quadrature to the spread counts part of the uncertainty
twice.

Background and references: [science overview](../science.md).

::: albireo.handoff
