# Downstream handoff

The disentangled spectrum is rarely the answer. It is the input to an atmosphere code —
GSSP, iSpec, Korg.jl, PySME — which turns it into an effective temperature, a surface
gravity and abundances. albireo should be the front half of that pipeline and has no
business being the back half; what it owes the back half is a file it reads without the
user hand-editing anything.

`write_gssp` and `write_ispec` are those files, transcribed from each code's own
documentation rather than remembered. The two formats disagree about almost everything that
can be got wrong quietly: GSSP wants two columns in **ångström** on an **equidistant** grid,
iSpec wants three tab-separated columns in **nanometres** with an absolute 1σ error and does
no unit conversion at all. A log-wavelength grid dumped straight into a GSSP file does not
look wrong — it sets GSSP's synthetic step from the first pixel pair.

## Why the draws are the feature

GSSP has nowhere to put a per-pixel uncertainty. Not "it ignores one": Appendix B of
Tkachenko (2015) — which *is* the manual — specifies a two-column file, and the
configuration files for all three of its modules contain no error path, no signal-to-noise
entry and no weighting entry. Its own quoted error bars come from χ² on the fit residuals.

So the posterior band cannot reach an effective temperature through the file. It can only
reach it by fitting many spectra, which is what `export_draws` is for: write *N* draws from
the joint posterior, fit all *N* with identical settings, and the **spread** of the
resulting parameters is the disentangling contribution — the term the literature currently
drops, in the authors' own words:

> the uncertainties that could arise from the normalisation procedure are not taken into
> account in the global uncertainties on the presented properties
>
> — Mahy et al. (2020), TMBM III, §3.1

The loop itself is not new. Kiran et al. (2016, §3.5) added Gaussian noise to a disentangled
profile, refitted 500 times and took the scatter. What is new is what gets drawn: albireo's
draws are `d_hat + L⁻ᵀz` on the vector stacked over *all* components, so they are correlated
across wavelength and across the two stars, and draw *i* of component A pairs with draw *i*
of component B. White-noise injection assumes the error is independent pixel to pixel;
disentangling error is not — it has a genuine low-frequency null space, and the
low-frequency part is the part that moves a continuum and therefore a temperature.

## What the spread does not contain

Worth reading before quoting one. The atmosphere code's own model error — grid coarseness,
LTE, line lists — is outside albireo's posterior entirely. So is anything albireo conditions
on rather than marginalizes, and the **light fractions** are conditioned on: they are
assumed, not inferred, and the marginal likelihood is flat in them under constant light.
That is exactly the systematic Pavlovski & Hensberge (2011) identify as the dominant one,
and the draw spread is silent about it. Finally, iSpec's own `errors['teff']` is a
within-draw fit error computed from the same band, so adding it in quadrature to the spread
counts part of the uncertainty twice.

::: albireo.handoff
