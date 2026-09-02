# albireo

**albireo** performs spectral disentangling of double- and multiple-lined spectroscopic
binaries. Given a time series of composite spectra, it infers the orbital elements and the
individual component spectra jointly. The component spectra are marginalized analytically,
so only the low-dimensional orbital and instrumental parameters are sampled, with the
No-U-Turn Sampler rather than by iterative least-squares refinement. The package is written
in JAX (float64), is differentiable end to end, and runs on CPU or GPU, with numpyro for
sampling and optax for optimization.

The name refers to Albireo, the double star in Cygnus.

!!! warning "Status: pre-alpha"

    albireo is under active development. The API is unstable and may change without
    notice.

## Where to start

- [Quickstart](quickstart.md): load the example dataset that ships with the package, fit
  it, and plot the result. No data of your own and no network are needed.
- [Scientific background](science.md): the disentangling problem, the methods in the
  literature, the degeneracies, and the references for every part of the package.
- Tutorials: [disentangle an SB2 end to end](tutorials/sb2-end-to-end.md) and
  [search for a faint companion with the K₂ scan](tutorials/k2-scan.md), both backed by
  executable scripts in `examples/`; then [read your own spectra](tutorials/real-data.md),
  [disentangle a BLOeM SB2](tutorials/bloem-sb2.md), [fit stellar labels](tutorials/labels.md),
  [measure epoch velocities](tutorials/todcor.md), and
  [run the pipeline](tutorials/pipeline.md).
- [Design](design.md): architecture, data model, and the decision ledger.
- [Mathematical foundations](math.md): the forward model, the analytic marginalization of
  the component spectra, the orbital parameterization, and the estimators for stellar
  labels and epoch velocities.
- [Benchmarks](benchmarks.md): the validation and performance record, including the
  closed-loop recovery tests and the degeneracy analyses.
- [Roadmap](roadmap.md): planned work and stated non-goals.

## Source

Code and issue tracker: [github.com/tjayasinghe/albireo](https://github.com/tjayasinghe/albireo)

If albireo contributed to a result, please see [Citing albireo](citing.md).
