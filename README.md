# albireo

[![CI](https://github.com/tjayasinghe/albireo/actions/workflows/ci.yml/badge.svg)](https://github.com/tjayasinghe/albireo/actions/workflows/ci.yml)
[![Docs](https://github.com/tjayasinghe/albireo/actions/workflows/docs.yml/badge.svg)](https://tjayasinghe.github.io/albireo/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

albireo performs Bayesian spectral disentangling of double- and multiple-lined
spectroscopic binaries. Given a time series of composite spectra, it infers the orbital
elements and the individual component spectra jointly, with posterior uncertainties on
both. The component spectra are marginalized analytically, so only the low-dimensional
orbital and instrumental parameters are sampled. The package is written in JAX (float64),
is differentiable end to end, and runs on CPU or GPU. It is named after Albireo, the
double star in Cygnus.

> [!WARNING]
> **Status: pre-alpha.** The API is unstable and may change without notice.

## Quickstart

A simulated double-lined binary with a known injected orbit ships with the package, so
the first fit needs no data and no network:

```python
import albireo as ab

dataset, truth = ab.load_example("sb2_sim", with_truth=True)   # 12 epochs, one instrument

dis = ab.Disentangler(
    dataset,
    components=[ab.Star("primary", light=0.62), ab.Star("secondary", light=0.38)],
    orbit=ab.Orbit(period=ab.Between(5.5, 6.5), k=ab.Between([10.0, 10.0], [90.0, 90.0])),
    lsf={"DEMO": 6.5},                     # Gaussian LSF sigma in km/s
)
fit = dis.fit()                            # phase scan, MAP, empirical-Bayes hyperparameters
post = fit.sample(seed=0)                  # NUTS over the orbit; spectra marginalized
print(post.summary())

ab.plot_spectra(dis.grid, fit.spectra(), std=fit.std())
```

The semi-amplitudes are recovered to better than 0.1% of the injected values, and no
per-epoch radial velocity is measured at any stage: the orbit is inferred from the spectra
directly. The annotated version is [`docs/quickstart.md`](docs/quickstart.md); the
low-level equivalent is [`examples/00_quickstart.py`](examples/00_quickstart.py).

## Capabilities

- **Joint inference of orbit and spectra.** Conditional on the orbit, light fractions,
  line-spread function and prior hyperparameters, the model is linear-Gaussian in the
  component spectra, which are integrated out in closed form. The No-U-Turn Sampler then
  explores 10 to 200 nonlinear parameters regardless of the number of pixels, and the
  spectra with their covariance are recovered at each posterior draw.
- **Wavelength-space model on native pixel grids.** Data are never resampled. Masks,
  chip gaps, cosmic rays, per-pixel weights, non-uniform sampling and multi-instrument
  data sets are handled by the weights alone.
- **Degeneracies made explicit.** The low-frequency separation degeneracy, the light-ratio
  versus line-depth degeneracy, and the systemic-velocity zero point are regularized with
  explicit prior scales and reported in the posterior. There is no default light fraction:
  the treatment must be declared, and per-epoch light fractions can be inferred where
  eclipses exist.
- **Instrumental and astrophysical nuisance components.** Telluric absorption, nebular
  emission with a free amplitude per exposure, per-epoch response polynomials,
  per-instrument and wavelength-dependent line-spread functions with an optional
  Gauss-Hermite asymmetry, per-epoch noise rescaling, and first-order autoregressive
  correlated noise. Hierarchical triples are modelled as nested Keplerians.
- **Faint-companion detection for single-lined systems.** A scan over the companion
  semi-amplitude with the companion spectrum marginalized at every trial, optionally with
  the primary semi-amplitude integrated out. Detection limits and false-alarm probabilities
  are measured by injection and recovery through the observed data's own operators.
- **Observing-strategy forecasts.** The posterior covariance of the spectra contains no
  fluxes, so the uncertainty band, the worst-determined spectral modes and the expected
  information gain of a planned set of epochs are computed before the data are taken.
- **Stellar labels for template selection.** Effective temperature, surface gravity,
  metallicity and projected rotation are fitted to the disentangled components against
  published synthetic grids (BOSZ, POLLUX), with the dilution of both components fitted
  jointly through a shared radius ratio and with two uncertainty estimates: the formal
  covariance and the spread over refits of joint posterior draws.
- **Epoch radial velocities by TODCOR.** One velocity per component per epoch by
  N-dimensional correlation against templates from a library, a label fit, or the
  disentangling itself, evaluated as a weighted least-squares fit so that masks and mixed
  instruments need no special treatment. A Keplerian is fitted to the resulting table with
  the same solver and conventions as the joint model.
- **Archive access and preprocessing.** ESO Phase 3 and IRAF-style FITS spectra are read
  with the wavelength frame, time system and barycentric correction taken from the header,
  the ESO archive is queried and downloaded from with the standard library only, and
  continuum normalization, noise estimation and masking are provided for reduced spectra
  that lack them.
- **A pipeline and command line.** `albireo run config.toml` reads each star's epochs,
  disentangles them, fits labels, measures velocities, fits the orbit, and writes tables,
  spectra with uncertainty bands, a JSON report and diagnostic figures, running stars in
  parallel worker processes and recording failures without stopping the batch.

The scientific background and the literature each method rests on are summarized in
[`docs/science.md`](docs/science.md).

## Command line

```bash
pip install "albireo[io,plots]"
albireo demo                      # two simulated stars with known answers, offline
albireo init                      # writes an annotated albireo.toml
albireo run albireo.toml --jobs 4
```

Each star receives a directory with `summary.txt`, `result.json`, the velocity table, the
disentangled spectra with their uncertainty bands, the orbit, the labels and the figures;
the batch receives `results.csv` with one row per star. Light fractions and the wavelength
medium are required in the configuration file rather than defaulted. See
[`docs/tutorials/pipeline.md`](docs/tutorials/pipeline.md).

## Installation

albireo is not yet on PyPI. Install from a clone:

```bash
git clone https://github.com/tjayasinghe/albireo.git
cd albireo
pip install -e ".[dev]"
```

Python 3.12 or newer is required. JAX 64-bit mode is enabled when the package is imported;
all computation is done in float64.

Two optional extras exist. The core never imports either, so a fit on a headless node
needs neither:

- `pip install -e ".[io]"` installs astropy for reading and writing FITS (`albireo.io`);
- `pip install -e ".[plots]"` installs matplotlib and ArviZ for `albireo.plotting` and
  the posterior diagnostics.

For a GPU build, install the `jax[cuda]` wheel for your platform following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Low-level interface

`Disentangler` is a declarative front end that derives the solver's velocity budget, the
model grid margins, the conjunction phase and the smoothness hyperparameters from the
declaration, and `dis.expert()` returns the `(model, priors, init)` triple it built. The
underlying classes and functions are the supported interface:

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
    v_rel_max_kms=250.0,   # velocity budget of the static solver; exceeding it is guarded
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
map_fit = ab.run_map(model.model(priors), init=init_values)          # MAP and ML-II
hyper = {s: map_fit.params[s] for s in ("log_tau", "log_eta")}
nuts_model = model.model({s: d for s, d in priors.items() if s not in hyper}, fixed=hyper)
mcmc = ab.run_nuts(
    nuts_model,
    rng_key=key,
    init=map_fit.params,
    inverse_mass_matrix=ab.laplace_inverse_mass(nuts_model, map_fit.params),
)
spectra = ab.posterior_spectra(model, mcmc.get_samples(), key, extra=hyper)
```

Reading archival spectra:

```python
ds = ab.read_dataset(
    "data/hr6819/*.fits",            # ESO Phase 3 or IRAF-style 1-D spectra
    instrument="FEROS",
    region=(4380.0, 4600.0),
    smooth_angstrom=120.0,           # continuum stiffness
)
ds = ab.Dataset(ab.share_wavelength_grid(list(ds)), frame=ds.frame)
grid = ab.LogGrid.covering(ds, dv_kms=1.5, v_margin_kms=90.0, lsf_sigma_kms=2.65)
```

The reader warns about every quantity it had to assume (frame, time system, wavelength
unit). See [`examples/03_hr6819_real_data.py`](examples/03_hr6819_real_data.py) for a
complete analysis of archival FEROS spectra.

## Documentation

- [`docs/quickstart.md`](docs/quickstart.md): the first fit, on packaged data.
- [`docs/science.md`](docs/science.md): scientific background and references.
- [`examples/`](examples/): executable tutorials, each ending in assertions against the
  injected truth; narrative versions in [`docs/tutorials/`](docs/tutorials/).
- [`docs/design.md`](docs/design.md): architecture, data model, and the decision ledger.
- [`docs/math.md`](docs/math.md): the forward model, the marginal likelihood, the
  degeneracy analysis, and the estimators for labels and epoch velocities.
- [`docs/benchmarks.md`](docs/benchmarks.md): the validation and performance record.
- [`docs/roadmap.md`](docs/roadmap.md): planned work and stated non-goals.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): development setup and test requirements.

The documentation is built with MkDocs Material:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Citation

A methods paper is in preparation. Until it appears, please cite the repository; see
[`CITATION.cff`](CITATION.cff) and [`docs/citing.md`](docs/citing.md).

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
