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
- **Reads archival spectra directly** — `albireo.io` turns a directory of ESO Phase-3 or
  IRAF-style FITS into a `Dataset`, and `albireo.preprocess` supplies the things reduced
  spectra are missing: a continuum, an inverse variance, masks, and one shared wavelength grid.

## Installation

albireo is **not yet on PyPI**. Install from a clone:

```bash
git clone https://github.com/tjayasinghe/albireo.git
cd albireo
pip install -e ".[dev]"
```

Python 3.12+ is required (current jaxlib no longer ships 3.11 wheels). JAX's x64 mode is enabled by the package at import time, so all
computation is done in float64 — you do not need to set `JAX_ENABLE_X64` yourself.

Reading spectra from FITS needs astropy, which is an optional extra — `pip install -e ".[io]"`.
Nothing else in albireo imports it.

For a GPU build, install the appropriate `jax[cuda]` wheel for your platform following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Current API (M4: joint inference + realism features)

Working today: the simulator, the fixed-orbit marginal solver, joint NUTS inference
of the orbit with the spectra marginalized (MAP → Laplace mass matrix → NUTS
pipeline; hyperparameters by ML-II), and the realism layer — telluric components,
hierarchical SB3 triples (`period_out`/…/`k_out` sites), per-epoch light-fraction
inference (`light` site, Dirichlet priors — the eclipse breaker, inferred),
multi-instrument LSF-width inference (`lsf_sigma` site; anchor one reference
instrument), per-epoch noise rescaling (`log_jitter` site — read
[the benchmarks](docs/benchmarks.md) before trusting one on real data), and the SB1
faint-companion K₂ scan (`ab.k2_scan`). A friendlier
`Disentangler` façade with light-ratio policies is planned; the core below is the
supported surface for now.

### From a directory of FITS files

Reduced spectra are not what the model consumes: they are rarely continuum-normalized, they
often ship no usable error array, and pipelines that apply the barycentric correction before
resampling give every exposure its own wavelength grid. `read_dataset` handles all three, and
warns about anything it had to assume (frame, time system, wavelength unit) instead of guessing
silently.

```python
import albireo as ab

ds = ab.read_dataset(
    "data/hr6819/*.fits",            # ESO Phase-3 or IRAF-style 1-D spectra
    instrument="FEROS",
    region=(4380.0, 4600.0),         # a full echelle order set is far more than you need
    smooth_angstrom=120.0,           # continuum stiffness
)
ds = ab.Dataset(ab.share_wavelength_grid(list(ds)), frame=ds.frame)  # 28 grids -> 1
grid = ab.LogGrid.covering(ds, dv_kms=1.5, v_margin_kms=90.0, lsf_sigma_kms=2.65)
print(ds.summary())
```

`ab.read_spectrum` returns the intermediate `RawSpectrum` if you want to inspect what the
header actually said before any of it is applied. See
[`examples/03_hr6819_real_data.py`](examples/03_hr6819_real_data.py) for a complete run on
real archival data.

### The inference pipeline

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
    "period": dist.Normal(63.1, 0.01),
    "t_conj": dist.Normal(2457811.5, 0.1),
    "secosw": dist.Uniform(-1, 1),
    "sesinw": dist.Uniform(-1, 1),
    "k": dist.Uniform(jnp.array([5.0, 5.0]), jnp.array([120.0, 120.0])),
    "log_tau": dist.Normal(jnp.log(300.0) * jnp.ones(2), 3.0),
    "log_eta": dist.Normal(jnp.log(5.0) * jnp.ones(2), 3.0),
}
map_fit = ab.run_map(model.model(priors), init=init_values)  # MAP + ML-II
hyper = {s: map_fit.params[s] for s in ("log_tau", "log_eta")}  # empirical Bayes
nuts_model = model.model({s: d for s, d in priors.items() if s not in hyper}, fixed=hyper)
mcmc = ab.run_nuts(
    nuts_model,
    rng_key=key,
    init=map_fit.params,
    inverse_mass_matrix=ab.laplace_inverse_mass(nuts_model, map_fit.params),
)
spectra = ab.posterior_spectra(model, mcmc.get_samples(), key, extra=hyper)

# SB1 + faint companion: marginalized K2 detection scan (docs/math.md §6)
scan = ab.k2_scan(
    grid,
    ds,
    orbit=sb1_solution,
    k1=12.0,
    k2_grid=jnp.arange(10.0, 150.0, 2.0),
    light_fractions=(0.95, 0.05),  # explicit — see the light-ratio policy
    lsf_sigma_v={"HERMES": 4.0},
    prior=spec_prior,
    v_rel_max_kms=250.0,
)
scan.k2_peak, scan.detection_peak, scan.companion  # + std, null loglike, ...
```

## Documentation

- [`examples/`](examples/) — executable tutorials (SB2 end-to-end, K₂ scan, and HR 6819 on real
  archival FEROS spectra); the first two run in CI. Narrative versions in
  [`docs/tutorials/`](docs/tutorials/).
- [`docs/design.md`](docs/design.md) — architecture, data model, and the shape of the inference
  problem.
- [`docs/math.md`](docs/math.md) — the disentangling likelihood, analytic marginalization of the
  component spectra, and the orbital parameterization.
- [`docs/benchmarks.md`](docs/benchmarks.md) — the running validation and performance record.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — dev setup and the test philosophy.

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
