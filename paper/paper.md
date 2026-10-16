---
title: 'albireo: Bayesian spectral disentangling of spectroscopic binaries'
tags:
  - Python
  - astronomy
  - spectroscopy
  - binary stars
  - spectral disentangling
  - Bayesian inference
  - JAX
authors:
  - name: Tharindu Jayasinghe
    orcid: 0000-0002-6244-477X
    corresponding: true
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 15 October 2026
bibliography: paper.bib
---

<!--
Delete this comment before submission (from 2027-02-14), after updating the date, the Gaia
DR4 sentence and the research-use TODO, and re-checking the numbers against docs/benchmarks.md.
-->

# Summary

A spectrograph records an unresolved binary star as a composite spectrum in which the
absorption lines of the two stars are superposed, each Doppler-shifted by its star's orbital
motion. From a time series of such spectra, spectral disentangling recovers the spectrum of
each star and, in most methods, the orbit. The separated spectra give the stars' temperatures
and abundances, and the orbit constrains their masses. The method is applied to massive and
eclipsing binaries and to candidate dormant black holes, in which it has revealed the lines of
a second star [@shenar2020; @bodensteiner2020].

`albireo` is a Python package that treats disentangling as statistical inference. It draws
orbits from their probability distribution given the data and, with them, the spectrum of each
star, so that both carry uncertainties. It is written in JAX [@jax2018].

# Statement of need

Temperatures, abundances and masses derived from disentangled spectra need uncertainties that
include those of the disentangling. Most codes return the component spectra as point estimates,
unregularized or regularized as chosen by the user, and that choice affects the line depths and
broad spectral features. With constant light fractions the light ratio cannot be measured, because
the data determine only the product of each star's light fraction and line depths; an assumed
ratio sets the recovered line depths and the abundances derived from them [@shenar2020].

`albireo` is intended for astronomers who analyse time series of composite spectra of
double-lined binaries, from a dozen epochs of a candidate compact companion to long
multi-instrument series. It fits the regularization weights to the data, and its posterior
draws of the spectra can be propagated into an atmospheric analysis. The user must declare the
light ratio, which is reported as an assumption; per-epoch light fractions can be fitted where
eclipses make them measurable.

# State of the field

Disentangling codes model each composite spectrum as the sum of the component spectra shifted
by the stars' radial velocities. They differ in where the problem is solved and in what is
fitted [@seeburger2024, Table 1], and in the uncertainties they report.

On a grid uniform in $\ln\lambda$ a Doppler shift is a translation, so the problem separates
into one small linear system per Fourier mode, which KOREL [@hadrava1995; @hadrava2009] and
FDBinary/fd3 [@ilijic2004] solve while optimizing the orbital elements. They are fast, but the
spectra must be resampled onto one equidistant grid without per-pixel weights or masks, and the
poorly determined lowest Fourier modes amplify normalization errors into low-frequency
undulations [@ilijic2004; @hensberge2008]; KOREL can leave these modes out of the fit.

In wavelength space the composite spectra are linear in the component spectra for given
velocities [@simon1994]. CRES [@ilijic2004cres] solves this system by singular value
decomposition with per-pixel weights, for velocities supplied by the user, and reports a formal
error per pixel. Spectangular [@sablowski2017; @sablowski2019] solves it without weights but
fits the orbit or per-epoch velocities. @seeburger2024 solve it iteratively with a
second-difference (Tikhonov) penalty whose weight is set by trial and error; their code has not
been released.

Shift-and-add subtracts the current estimate of one component from every epoch, averages the
residuals in the frame of the other, and repeats [@marchenko1998; @gonzalez2006]. In current
use the semi-amplitudes are found by repeating this on a grid of $K_2$ or $(K_1, K_2)$, with
the other elements taken from the primary's orbit [@shenar2020; @shenar2022]. The method is
widely used for massive binaries and candidate compact companions. It has no stopping rule, and
stopping early leaves spurious line wings and wrong line strengths [@quintero2020].

Of these codes only PSOAP [@czekala2017] samples a posterior: it integrates the spectra out
analytically under Gaussian-process priors and samples the orbit and the hyperparameters of the
priors. Its dense covariance matrices over epochs and pixels limited it to segments of about
10\ Å, and it is not maintained. The other codes estimate the uncertainties of the orbit from
$\chi^2$ surfaces or from refits, or report none.

Like PSOAP, `albireo` integrates the spectra out, but its smoothness prior, equivalent to the
curvature penalty of @seeburger2024, has a banded precision matrix, so the cost is linear in
the number of pixels. Unlike PSOAP, it holds the regularization weights at their fitted values
while sampling, which confines the sampler to the orbit but leaves their uncertainty
unpropagated. Extending an existing code was not practical: the design needs a marginal
likelihood and its gradient at every step of a sampler, and of these codes only PSOAP computes
a marginal likelihood, with the dense matrices that limited it. KOREL, fd3 and the public
shift-and-add code [@shenar2023code] also state no licence. On identical simulated data,
shift-and-add at seven iterations is fastest and `albireo` most accurate ([benchmark
record](https://tjayasinghe.github.io/albireo/benchmarks/#the-three-code-comparison-at-a-common-resolution-2026-10-15)).

# Software design

The model of epoch $j$ is

$$
y_j = \mathbf{R}_j \Big[ 1 + \sum_i \ell_{ij} \, \mathbf{B}_j \, \mathbf{T}(v_{ij}) \, d_i \Big] + n_j ,
$$

where $d_i$ is the normalized spectrum of star $i$ minus the continuum, on a grid uniform in
$\ln\lambda$; $\mathbf{T}(v_{ij})$ shifts it to the star's velocity, $\mathbf{B}_j$ convolves
it with the line-spread function, $\ell_{ij}$ is the star's fraction of the continuum light
(constant unless fitted per epoch), $\mathbf{R}_j$ integrates the model over each of the
epoch's own pixels, and $n_j$ is Gaussian noise. Fitting each spectrum on its own pixels,
without resampling, keeps per-pixel weights and masks valid and admits several instruments.
Optional terms add telluric and nebular spectra, and the velocities follow from Keplerian or
nested orbits or are free at each epoch.

For a given orbit the model is linear in the spectra. With $\mathbf{A}$ the design matrix and
$\mathbf{W}$ the inverse variances, the second-difference penalty and a term that pulls each
spectrum toward the continuum are interpreted as a Gaussian prior of precision
$\mathbf{\Lambda}$. The regularized solution $\hat d$ is then the conditional posterior mean of
the spectra, $\mathbf{\Lambda} + \mathbf{A}^\top \mathbf{W} \mathbf{A}$ is their posterior
precision, and the spectra integrate out in closed form:

$$
-2 \ln p(y \mid \theta) = \chi^2(\hat d) + \hat d^\top \mathbf{\Lambda} \hat d
+ \ln\det\big(\mathbf{\Lambda} + \mathbf{A}^\top \mathbf{W} \mathbf{A}\big)
- \ln\det \mathbf{\Lambda} + \mathrm{const},
$$

with $\theta$ the orbital elements. The determinant terms penalize the flexibility of the
spectra and make the regularization weights in $\mathbf{\Lambda}$ estimable by maximizing this
marginal likelihood [type-II maximum likelihood, @mackay1992], whereas $\chi^2$ alone always
prefers the weakest regularization. The sampler moves only in the orbital parameters, whatever
the number of pixels: six for a double-lined binary, excluding the systemic velocity, which is
degenerate with a shift common to both spectra and is measured afterwards against a template.

With the pixels of the stars interleaved, the posterior precision is a single band whose width
is set by the largest velocity difference that the priors allow. It is factorized by a block
Cholesky decomposition at a cost proportional to the number of pixels times the square of the
band width, so that wide velocity ranges cost more. The same factor gives the log-determinant,
the conditional mean and, through a selected-inverse recursion [@erisman1975], the band of the
inverse, which holds the variance of every pixel. The gradient with respect to the precision
matrix follows in closed form from that band, so that reverse-mode differentiation does not
pass through the sequential factorization.

Inference has four stages: scans over the time of conjunction and the semi-amplitudes, which
select among the many local maxima, while the period is optimized within its declared range; a
maximum a posteriori fit of the orbit and the regularization weights; a Laplace approximation
at the optimum; and the No-U-Turn Sampler [@hoffman2014] of NumPyro, with the Laplace
covariance as a fixed mass matrix. Because the posterior widths differ by orders of magnitude,
NUTS from a unit mass matrix had not finished warm-up after 35 minutes on a 12-epoch test
problem, whereas with the Laplace mass matrix warm-up and 250 draws took under two minutes on
one desktop CPU. JAX differentiates the path from the orbital elements to the band and compiles
it for CPUs and GPUs.

# Research impact statement

`albireo` has been public since August 2026, with
[documentation](https://tjayasinghe.github.io/albireo/), tutorials and continuous integration.
Its research uses to date are the author's own, recorded in the [benchmark
record](https://tjayasinghe.github.io/albireo/benchmarks/).

<!-- TODO before submission: add research use beyond the author's own (papers, preprints,
other groups, talks). -->

The 51 public FEROS spectra of HR 6819 [@bodensteiner2020] were disentangled to measure the
stripped star's orbit free of blending with the Be star's lines. The semi-amplitude is 3.7 to
4% above the published value [@klement2025], in the direction that deblending predicts, and a
period difference of 0.04 d remains unexplained. For the eclipsing binary AI Phoenicis,
effective temperatures fitted to the disentangled HARPS spectra are 0.1% and 4.3% above the
published values [@maxted2020].

`albireo` retrieves by survey identifier the public spectra of the 59 double-lined binaries
that the BLOeM survey found among early B-type dwarfs and giants in the Small Magellanic Cloud
[@shenar2024; @villasenor2025], and a
[tutorial](https://tjayasinghe.github.io/albireo/tutorials/bloem-sb2/) sets out the procedure
for one system.

Gaia DR4 will publish an RVS spectrum for each field-of-view transit on 2 December 2026
[@gaiarelease; @gaiadr4]. In a simulation of 7,800 detached eclipsing binaries in that form,
with velocities measured by two-dimensional correlation [@zucker1994] against classified
templates, not by disentangling, and orbits fitted at the photometric ephemeris, both
semi-amplitudes are recovered within 10% for 61% of the systems brighter than
$G_\mathrm{RVS} = 9$ and 15% at 12 to 13; the
[report](https://tjayasinghe.github.io/albireo/reports/gaia-rvs-eclipsing-binaries/) quantifies
the losses from noisier transits, third light and chromospheric emission.

# AI usage disclosure

Generative AI was used throughout development, with Claude Code (Anthropic; versions 2.1.222
to 2.1.295) and the models Claude Opus 5, Claude Opus 5.5, Claude Fable 5 and Claude Fable
5.1; every commit after the initial one names one of these models as co-author. The models
wrote code and tests, drafted the documentation and this paper, searched the literature, and
checked the statements about other codes against the listed
[sources](https://tjayasinghe.github.io/albireo/comparison/#sources). The author made the
primary architectural and design decisions, among them the requirement to declare the
quantities that the data do not determine, the fitting of each spectrum on its own pixels, and
the validation programme, and reviewed, modified and validated all AI-generated code and text.
The tests include a closed-loop test of each inference feature against simulated data, a
check of the marginal likelihood against dense brute-force marginalization to a relative
tolerance of $10^{-10}$, and 24 injections whose posterior intervals are consistent with
nominal coverage.

# Acknowledgements

`albireo` is built on JAX [@jax2018], NumPyro [@phan2019; @bingham2019], Optax [@optax2020],
NumPy [@harris2020] and SciPy [@virtanen2020].

# References
