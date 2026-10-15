# albireo

**albireo** separates the spectra of the stars in a double- or multiple-lined spectroscopic
binary from a time series of composite spectra, and determines the orbit at the same time.
It works like the wavelength-space disentangling codes: for a trial orbit the component
spectra follow from a linear least-squares fit, here with a smoothing penalty, and the orbit
is adjusted around that fit. Compared with the established codes (KOREL, fd3, Spectangular
and shift-and-add), three things differ:

- Every disentangled spectrum has an uncertainty band. The smoothing penalty is treated as a
  Gaussian prior, so the regularized solution is the posterior mean of the spectra, and the
  matrix solved for it gives their covariance.
- The strength of the smoothing is fitted to the data, by maximizing the marginal
  likelihood, rather than set by hand or by the number of iterations.
- The orbit is sampled by Markov chain Monte Carlo (the No-U-Turn Sampler), not only
  optimized. The spectra are integrated out at every step, so the sampler explores only the
  orbital parameters, however many pixels there are.

PSOAP (Czekala et al. 2017) introduced a probabilistic treatment of this kind for spectral
segments of about 10 Å; albireo applies it to whole spectra, at a cost linear in their length.

The package is written in JAX (float64), is differentiable end to end, and runs on CPU or
GPU. It uses numpyro for sampling and optax for optimization. The name refers to Albireo,
the double star in Cygnus.

!!! warning "Status: pre-alpha"

    albireo is under active development. The API is unstable and may change without
    notice.

## Where to start

- [Quickstart](quickstart.md): load the example dataset that ships with the package, fit
  it, and plot the result. No other data and no network access are needed.
- [Method overview](method.md): what a fit computes, step by step, in the terms of a
  regularized least-squares fit, with figures and a glossary of the statistical terms.
- [Comparison with other codes](comparison.md): what albireo shares with KOREL, fd3,
  Spectangular, shift-and-add, PSOAP and Seeburger et al. (2024), and where it differs.
- [Scientific background](science.md): the disentangling problem, the methods in the
  literature, the degeneracies, and the references for every part of the package.
- Tutorials: [disentangle an SB2 end to end](tutorials/sb2-end-to-end.md) and
  [search for a faint companion with the K₂ scan](tutorials/k2-scan.md), both backed by
  executable scripts in `examples/`; then [read your own spectra](tutorials/real-data.md),
  [disentangle a BLOeM SB2](tutorials/bloem-sb2.md), [fit stellar labels](tutorials/labels.md),
  [measure epoch velocities](tutorials/todcor.md), and
  [run the pipeline](tutorials/pipeline.md).
- [Mathematical foundations](math.md): the forward model, the analytic marginalization of
  the component spectra, the orbital parameterization, and the estimators for stellar
  labels and epoch velocities.
- [Benchmarks](benchmarks.md): the validation and performance record, including the
  closed-loop recovery tests and the degeneracy analyses.
- [Gaia DR4 eclipsing binaries in the RVS](reports/gaia-rvs-eclipsing-binaries.md): a
  simulated survey of 7800 detached eclipsing binaries, with both velocities of every
  transit measured by correlation and the orbits and masses that follow.

## Source

Code and issue tracker: [github.com/tjayasinghe/albireo](https://github.com/tjayasinghe/albireo)

If albireo contributed to a result, please see [Citing albireo](citing.md).
