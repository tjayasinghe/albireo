# albireo

**albireo** performs spectral disentangling of double- and multiple-lined spectroscopic binaries:
given a time series of composite spectra, it jointly infers the orbital elements and the individual
component spectra. The component spectra are marginalized analytically, so only the low-dimensional
orbit needs to be sampled, and this is done fully Bayesianly with NUTS rather than by iterative
least-squares refinement. Everything is written in JAX (float64) and is therefore differentiable and
runs on GPU, with numpyro for posterior sampling and optax for MAP optimization.

The name comes from Albireo, the famous gold-and-blue double star in Cygnus:
*albireo separates the gold from the blue*.

!!! warning "Status: pre-alpha"

    albireo is under active development. The API is unstable and will change without notice.
    It is not yet suitable for production science.

## Where to go next

- Tutorials: [disentangle an SB2 end to end](tutorials/sb2-end-to-end.md) and
  [find a hidden companion with the K₂ scan](tutorials/k2-scan.md) — both backed by
  executable scripts in `examples/` that run in CI.
- [Design](design.md) — architecture, data model, and the shape of the inference problem.
- [Mathematical foundations](math.md) — the disentangling likelihood, the analytic marginalization
  of the component spectra, and the orbital parameterization.
- [Benchmarks](benchmarks.md) — the running correctness/performance record per milestone,
  including the closed-loop recovery gates and the degeneracy analyses.

## Source

Code and issue tracker: [github.com/tjayasinghe/albireo](https://github.com/tjayasinghe/albireo)
