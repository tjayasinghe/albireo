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
checklist is the M5 deliverable in internal/design.md §8 (fd3 benchmark, one published SB2
end-to-end, tutorials in CI). Numbers below are from docs/benchmarks.md through M4 and
should be re-checked against the test suite at submission time.
-->

# Summary

Most binary stars are not spatially resolved. A spectrograph records a single composite
spectrum in which the absorption lines of both stars are superposed, each Doppler-shifted by its
own orbital motion and blended differently at every epoch. *Spectral disentangling* is the inverse
problem of recovering, from such a time series, both the orbit and the individual component
spectra. It is the step that turns an unresolved binary into two stars whose temperatures,
abundances and masses can be measured separately, and it underpins massive-star binary surveys,
benchmark eclipsing-binary mass determinations, and searches for dormant black-hole and
neutron-star companions, where the presence or absence of a faint second set of lines decides how
a system is interpreted.

`albireo` treats disentangling as a single Bayesian inference problem, implemented in JAX.
Conditional on the nonlinear parameters (orbit, light fractions, line-spread-function widths,
prior hyperparameters), the component spectra enter the model linearly. Under Gaussian noise and
Gaussian smoothness priors those $10^5$ to $10^6$ unknowns integrate out in closed form, leaving a
marginal likelihood over 10 to 200 nonlinear parameters that Hamiltonian Monte Carlo can explore
regardless of the number of spectral pixels. The component spectra and their covariance are
recovered afterwards as a conditional Gaussian at each posterior draw, so the result is a joint
posterior over the orbit and the spectra. The model works in wavelength space on each spectrum's
native grid, so masks, chip gaps, per-pixel weights and multi-instrument data sets need no
resampling. It is written in float64 JAX, differentiable and jit-compiled end to end, and runs on
a GPU by a change of backend rather than of code.

# Statement of need

Existing disentangling codes fall into three families, none of which reports uncertainties on the
recovered spectra. Wavelength-space linear solvers [@simon1994] cast disentangling as a large
sparse least-squares problem, which is the appropriate formulation but provides no regularization
of the ill-conditioned low-frequency null space. Fourier-space codes, principally KOREL
[@hadrava1995] and `fd3`/FDBinary [@ilijic2004], are the incumbents: fast and widely used, but
they require resampling onto a common equidistant log-wavelength grid, suffer a known
low-frequency bias, and return point estimates only. Iterative shift-and-add disentangling
[@gonzalez2006; @shenar2020] is the standard method in the massive-star and compact-companion
community; it is robust at low signal-to-noise ratio but scales poorly with the number of
parameters, uses the stopping point of the iteration as an uncontrolled implicit regularizer, and
produces no posterior.

Two efforts are closer to the present work. PSOAP [@czekala2017] is the direct methodological
predecessor: a joint Bayesian model in wavelength space with the component spectra marginalized
analytically. Its dense Gaussian-process covariances over epochs and pixels cost cubic time and
quadratic memory, which restricted it to small spectral chunks, and it predates the automatic
differentiation and GPU frameworks that make the approach scale. @seeburger2024 target survey
scale with a second-derivative Tikhonov penalty and sparse iterative solves, but the method is
not Bayesian and returns no posteriors.

`albireo` provides joint Bayesian inference of the orbit and the component spectra with
uncertainties on both, at a cost that scales linearly with the number of pixels, with automatic
differentiation, GPU execution, tests and packaging. Analytic marginalization of the spectra is
what makes this combination possible: it reduces a $10^6$-dimensional inference problem to a
low-dimensional one exactly rather than approximately.

A second aim is explicitness about degeneracies. With constant light fractions the likelihood
depends only on the products of light fraction and line depth, so the continuum light ratio is not
measurable from the spectra alone. This is the point on which the interpretation of published
compact-companion candidates has turned [@shenar2020], and shift-and-add implementations settle
it by fixing light ratios by hand. `albireo` supplies no default: the light-ratio treatment must
be declared, and where per-epoch light variation from eclipses exists it is inferred, which breaks
the degeneracy with data rather than by assumption. The same policy covers the systemic velocity
and absolute line-spread-function widths, both of which are unidentifiable in a template-free
model.

# Functionality

A `Dataset` of per-epoch spectra (wavelength, flux, inverse variance, time, barycentric velocity,
instrument) and a log-wavelength model grid define a marginal-likelihood model. `albireo` builds
the per-epoch forward operator (shift, LSF convolution, rebinning to the native grid, response
polynomial) and evaluates $\log p(y \mid \theta)$ with the component spectra integrated out. The
posterior precision of the spectra is block-tridiagonal: its band is assembled analytically epoch
by epoch, factorized with a scanned block Cholesky decomposition for exact log-determinants and
selected inverses at banded cost, and differentiated in closed form through the block Takahashi
selected inverse rather than by reverse-mode differentiation through the factorization.

Inference runs in three stages: a maximum a posteriori fit with L-BFGS, which also performs
type-II maximum-likelihood estimation of the spectral-prior hyperparameters; a Laplace
approximation at the optimum supplying the inverse mass matrix; and the No-U-Turn Sampler via
numpyro [@phan2019; @bingham2019] with mass adaptation disabled. The mass matrix is necessary
rather than convenient: posterior scales in these problems span about five orders of magnitude,
and NUTS started from a unit mass matrix on the validation problem was still in warm-up after 35
minutes, whereas the full pipeline converges in about three minutes on a laptop CPU. Beyond the
two-star Keplerian case the package supports telluric components, hierarchical SB3 triples as
nested Keplerians, per-epoch light fractions on the simplex, per-instrument LSF widths, and an SB1
faint-companion mode: a scan over the trial secondary semi-amplitude $K_2$ in which the unknown
companion spectrum is marginalized at every grid point, which is a matched filter that assumes no
template. Further modes fit stellar labels to the disentangled components against published
synthetic grids, measure epoch radial velocities by N-dimensional correlation against templates,
fit Keplerian orbits to velocity tables, forecast the information that planned epochs would add,
and run every stage for a list of stars from one configuration file.

Every inference feature is accompanied by a closed-loop test against simulated data with known
injected truth. The test suite verifies that the marginal likelihood agrees with dense brute-force
marginalization to a relative tolerance of $10^{-10}$ in the presence of masks, mixed instruments
and response polynomials; that on a simulated 12-epoch SB2 at a signal-to-noise ratio of 130 the
velocity semi-amplitudes are recovered to 0.33% and 0.19% with no divergent transitions, and that
a 24-injection study with truths drawn from the priors gives central-interval coverage consistent
with nominal at that sample size; and that with three eclipse epochs whose light fractions are
inferred, the individual component spectra are recovered at about 1% RMS in line cores. The
complete record, including negative results, is kept in the repository's benchmark log.

# Acknowledgements

`albireo` is built on JAX [@jax2018], numpyro [@phan2019; @bingham2019], and optax [@optax2020].
The analytic-marginalization strategy follows PSOAP [@czekala2017].

# References
