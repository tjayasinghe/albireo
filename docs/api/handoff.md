# Downstream handoff

The disentangled spectrum is rarely the final result. It is the input to an atmosphere
code (GSSP, iSpec, Korg.jl, PySME) that converts it into an effective temperature, a
surface gravity and abundances. albireo covers the disentangling step and does not attempt
the atmospheric analysis; the interface between the two is a file the atmosphere code reads
without manual editing.

`write_gssp` and `write_ispec` write those files, transcribed from each code's own
documentation. The two formats differ in almost every detail that can go wrong without an
error: GSSP requires two columns in ångström on an equidistant grid, and iSpec requires
three tab-separated columns in nanometres with an absolute 1σ error and performs no unit
conversion. A log-wavelength grid written directly into a GSSP file does not look wrong;
it sets GSSP's synthetic step from the first pixel pair.

## Posterior draws

GSSP accepts no per-pixel uncertainty. Appendix B of Tkachenko (2015), which serves as the
manual, specifies a two-column file, and the configuration files for all three of its
modules contain no error path, no signal-to-noise entry and no weighting entry. Its quoted
error bars come from χ² on the fit residuals.

The posterior band therefore cannot reach an effective temperature through the file. It
can reach one only through repeated fits, which is the purpose of `export_draws`: write *N*
draws from the joint posterior, fit all *N* with identical settings, and take the spread of
the resulting parameters as the disentangling contribution. This is the term the
literature currently omits, as its authors state:

> the uncertainties that could arise from the normalisation procedure are not taken into
> account in the global uncertainties on the presented properties
>
> (Mahy et al. 2020, TMBM III, §3.1)

The loop itself is not new: Kiran et al. (2016, §3.5) added Gaussian noise to a
disentangled profile, refitted 500 times and took the scatter. The difference is in what is
drawn. albireo's draws are `d_hat + L⁻ᵀz` on the vector stacked over all components, so
they are correlated across wavelength and across the two stars, and draw *i* of component A
pairs with draw *i* of component B. White-noise injection assumes the error is independent
from pixel to pixel; disentangling error is not, because the problem has a low-frequency
null space, and the low-frequency part is what moves a continuum and therefore a
temperature.

## Contributions outside the spread

The atmosphere code's own model error (grid coarseness, LTE, line lists) lies outside
albireo's posterior. So does anything albireo conditions on rather than marginalizes, and
the light fractions are conditioned on: they are assumed, not inferred, and the marginal
likelihood is flat in them under constant light. This is the systematic that Pavlovski &
Hensberge (2011) identify as the dominant one, and the draw spread contains no information
about it. Finally, iSpec's own `errors['teff']` is a within-draw fit error computed from
the same band, so adding it in quadrature to the spread counts part of the uncertainty
twice.

Background and references: [science overview](../science.md).

::: albireo.handoff
