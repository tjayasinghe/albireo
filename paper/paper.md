---
# NOTE: "GPU-accelerated" deliberately kept out of the title until the M5 GPU-scale
# gate is recorded in docs/benchmarks.md; the body states the architectural claim.
title: 'albireo: differentiable, fully Bayesian spectral disentangling of spectroscopic binaries'
tags:
  - Python
  - astronomy
  - spectroscopy
  - binary stars
  - spectral disentangling
  - Bayesian inference
  - JAX
authors:
  # TODO: add ORCID before submission (JOSS expects one for the corresponding author).
  - name: Tharindu Jayasinghe
    affiliation: 1
affiliations:
  # TODO: no affiliation is recorded in CITATION.cff or pyproject.toml; fill in before submission.
  - name: TODO
    index: 1
date: 11 August 2026
bibliography: paper.bib
---

<!--
TODO (submission gating, not paper content): the package is pre-alpha and the JOSS
checklist is the M5 deliverable in docs/design.md §8 (fd3 benchmark, one published SB2
end-to-end, tutorials in CI). Numbers below are from docs/benchmarks.md through M4 and
should be re-checked against the test suite at submission time.
-->

# Summary

Most binary stars cannot be resolved into two points of light. What a spectrograph records is a
single composite spectrum in which both stars' absorption lines are superposed, each Doppler-shifted
by its own orbital motion and blended differently at every epoch. *Spectral disentangling* is the
inverse problem of recovering, from such a time series, both the orbit and the individual component
spectra — the step that turns an unresolved binary into two stars whose temperatures, abundances,
and masses can be measured separately. It underpins massive-star binary surveys, benchmark
eclipsing-binary mass determinations, and searches for dormant black-hole and neutron-star
companions, where whether a faint second set of lines is present at all can decide how a system is
interpreted.

`albireo` treats disentangling as a single Bayesian inference problem, implemented in JAX. Its
organizing idea is that, conditional on the nonlinear parameters — orbit, light fractions,
line-spread-function (LSF) widths, prior hyperparameters — the component spectra enter the model
*linearly*. Under Gaussian noise and Gaussian smoothness priors those $10^5$–$10^6$ unknowns
integrate out in closed form, leaving a marginal likelihood over 10–200 nonlinear parameters that
Hamiltonian Monte Carlo can explore regardless of how many spectral pixels the data contain. The
component spectra and their covariance are recovered afterwards as a conditional Gaussian at each
posterior draw, so the deliverable is a joint posterior over the orbit *and* the spectra. The model
works in wavelength space on each spectrum's native grid, so masks, chip gaps, per-pixel weights,
and multi-instrument data sets need no resampling. It is written in float64 JAX, differentiable and
jit-compiled end to end, with GPU execution a change of backend rather than of code.

# Statement of need

Existing disentangling codes fall into three families, none of which reports uncertainties on the
recovered spectra. Wavelength-space linear solvers [@simon1994] cast disentangling as a large sparse
least-squares problem — conceptually the right frame, but with no regularization of the
ill-conditioned low-frequency nullspace. Fourier-space codes, above all KOREL [@hadrava1995] and
`fd3`/FDBinary [@ilijic2004], are the incumbents: fast and heavily used, but they require resampling
onto a common equidistant log-wavelength grid, suffer a known low-frequency bias, and return point
estimates only. Iterative shift-and-add grid disentangling [@gonzalez2006; @shenar2020] is the de
facto standard in the massive-star and compact-companion community; it is robust at low
signal-to-noise, but scales poorly with dimension, uses stalled iteration as an uncontrolled
implicit regularizer, and produces no posterior.

Two efforts come closer. PSOAP [@czekala2017] is the direct methodological ancestor: a joint
Bayesian model in wavelength space with the component spectra marginalized analytically. Its dense
Gaussian-process covariances over epochs × pixels cost cubic time and quadratic memory, forcing very
small spectral chunks, and it predates the autodiff/GPU frameworks that would make the approach
scale. @seeburger2024 targets survey scale with a second-derivative Tikhonov penalty and sparse
iterative solves, but is explicitly not Bayesian and returns no posteriors.

The gap `albireo` fills is the conjunction no available code offers: joint Bayesian inference of the
orbit *and* the component spectra with uncertainties on both; scalability to survey-sized data sets;
and modern engineering (autodiff, GPU, tests, packaging). Analytic marginalization of the spectra is
what makes that conjunction possible: it reduces a $10^6$-dimensional inference problem to a
low-dimensional one exactly, rather than approximately.

A second motivation is explicitness about degeneracies. With constant light fractions the likelihood
depends only on the products of light fraction and line depth, so the continuum light ratio is not
measurable from the spectra alone — the point on which the interpretation of published
compact-companion candidates has turned [@shenar2020], and which shift-and-add implementations
settle by fixing light ratios by hand. `albireo` supplies no default: the light-ratio treatment must
be declared, and where per-epoch light variation (eclipses) exists it is *inferred*, breaking the
degeneracy with data rather than by assumption. The same policy covers systemic velocity and
absolute LSF widths, both unidentifiable in a template-free model.

# Functionality

A `Dataset` of per-epoch spectra (wavelength, flux, inverse variance, time, barycentric velocity,
instrument) and a log-wavelength model grid define a marginal-likelihood model. `albireo` builds the
per-epoch forward operator — shift, LSF convolution, rebin to the native grid, response polynomial —
and evaluates $\log p(y \mid \theta)$ with the component spectra integrated out. The resulting
precision matrix is block-tridiagonal and is factorized with a scanned banded Cholesky, giving exact
log-determinants and selected inverses at banded cost.

Inference runs as a three-stage pipeline: MAP with L-BFGS, which also performs ML-II estimation of
the spectral-prior hyperparameters; a Laplace approximation at the MAP supplying the inverse mass
matrix; then NUTS via numpyro [@phan2019; @bingham2019] with mass adaptation disabled. This is not a
convenience. Posterior scales in these problems span some five orders of magnitude, and NUTS started
from a unit mass matrix on the validation problem was still in warmup after 35 minutes, where the
full pipeline converges in about three minutes on a laptop CPU. Beyond the two-star Keplerian case
the package supports telluric components, hierarchical SB3 triples as nested Keplerians, per-epoch
light fractions on the simplex, per-instrument LSF widths, and an SB1 faint-companion mode: a scan
over trial secondary semi-amplitude $K_2$ in which the unknown companion spectrum is marginalized at
every grid point — a matched filter that assumes no template.

Every inference feature ships with a closed-loop test against simulated data with known injected
truth. Validated in the test suite: the marginal likelihood agrees with dense brute-force
marginalization to a relative tolerance of $10^{-10}$ in the presence of masks, mixed instruments,
and response polynomials; on a simulated 12-epoch, SNR-130 SB2 the velocity semi-amplitudes are
recovered to 0.33% and 0.19% with no divergent transitions, and a 24-injection study with truths
drawn from the priors gives central-interval coverage consistent with nominal at that sample size;
and with three eclipse epochs whose light fractions are inferred, the individual component spectra
come back at ~1% RMS in line cores. The running record, including the negative results, is kept in
the repository's benchmark log.

# Acknowledgements

`albireo` is built on JAX [@jax2018], numpyro [@phan2019; @bingham2019], and optax [@optax2020]; the
analytic-marginalization strategy owes an obvious debt to PSOAP [@czekala2017].

# References
