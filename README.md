# albireo

Albireo is the famous gold-and-blue double star in Cygnus: **albireo separates the gold from the
blue** — GPU-accelerated, fully Bayesian spectral disentangling of spectroscopic binaries in JAX.

> [!WARNING]
> **Status: pre-alpha, under active development, API unstable.**
> Expect breaking changes without notice. Not yet suitable for production science.

## Why albireo

- **Analytic marginalization of the component spectra** — the (very high-dimensional) component
  spectra are integrated out in closed form, so NUTS only ever has to explore the low-dimensional
  orbit.
- **Honest uncertainties** on both the orbital elements *and* the recovered component spectra, from
  a single joint posterior rather than from a bootstrap around a point estimate.
- **Works in wavelength space**, so masks, per-pixel weights, spectral gaps, non-uniform sampling,
  and multi-instrument data sets are handled natively — no Fourier-domain padding or interpolation
  onto a common wavelength solution required.
- **GPU-accelerated and differentiable** end to end via JAX (float64), with numpyro for NUTS and
  optax for MAP optimization.
- **SB1 faint-companion detection**, via a K2 scan: profile the marginal likelihood over the
  secondary velocity semi-amplitude to search for companions that never show up as a second set of
  lines.

## Installation

albireo is **not yet on PyPI**. Install from a clone:

```bash
git clone https://github.com/tjayasinghe/albireo.git
cd albireo
pip install -e ".[dev]"
```

Python 3.12+ is required (current jaxlib no longer ships 3.11 wheels). JAX's x64 mode is enabled by the package at import time, so all
computation is done in float64 — you do not need to set `JAX_ENABLE_X64` yourself.

For a GPU build, install the appropriate `jax[cuda]` wheel for your platform following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Current API (M3: joint orbit + spectra inference)

Working today: the simulator, the fixed-orbit marginal solver, and joint NUTS
inference of the orbit with the spectra marginalized (MAP → Laplace mass matrix →
NUTS pipeline; hyperparameters by ML-II). A friendlier `Disentangler` façade with
light-ratio policies is planned; the core below is the supported surface for now.

```python
import albireo as ab
import jax.numpy as jnp
import numpyro.distributions as dist

ds = ab.Dataset([ab.EpochData(wave=w, flux=f, ivar=iv, bjd=t, v_bary=vb, instrument="HERMES"), ...])
model = ab.MarginalOrbitModel(
    ab.LogGrid.from_wavelength_range(4000.0, 6800.0, dv_kms=1.0),
    ds,
    light_fractions=[0.62, 0.38],
    lsf_sigma_v={"HERMES": 4.0},
    v_rel_max_kms=250.0,  # velocity budget; wider priors are guarded, not corrupted
)
priors = {
    "period": dist.Normal(63.1, 0.01), "t_conj": dist.Normal(2457811.5, 0.1),
    "secosw": dist.Uniform(-1, 1), "sesinw": dist.Uniform(-1, 1),
    "k": dist.Uniform(jnp.array([5.0, 5.0]), jnp.array([120.0, 120.0])),
    "log_tau": dist.Normal(jnp.log(300.0) * jnp.ones(2), 3.0),
    "log_eta": dist.Normal(jnp.log(5.0) * jnp.ones(2), 3.0),
}
map_fit = ab.run_map(model.model(priors), init=init_values)          # MAP + ML-II
hyper = {s: map_fit.params[s] for s in ("log_tau", "log_eta")}       # empirical Bayes
nuts_model = model.model({s: d for s, d in priors.items() if s not in hyper}, fixed=hyper)
mcmc = ab.run_nuts(
    nuts_model, rng_key=key, init=map_fit.params,
    inverse_mass_matrix=ab.laplace_inverse_mass(nuts_model, map_fit.params),
)
spectra = ab.posterior_spectra(model, mcmc.get_samples(), key, extra=hyper)
```

## Documentation

- [`docs/design.md`](docs/design.md) — architecture, data model, and the shape of the inference
  problem.
- [`docs/math.md`](docs/math.md) — the disentangling likelihood, analytic marginalization of the
  component spectra, and the orbital parameterization.

Docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and will be
hosted once the package reaches a usable state. To build them locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Citation

A methods paper is in preparation. Until it appears, please cite the repository directly — see
[`CITATION.cff`](CITATION.cff).

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
