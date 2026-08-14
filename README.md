# albireo

[![CI](https://github.com/tjayasinghe/albireo/actions/workflows/ci.yml/badge.svg)](https://github.com/tjayasinghe/albireo/actions/workflows/ci.yml)
[![Docs](https://github.com/tjayasinghe/albireo/actions/workflows/docs.yml/badge.svg)](https://tjayasinghe.github.io/albireo/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

Albireo is the famous gold-and-blue double star in Cygnus: **albireo separates the gold from the
blue** — GPU-accelerated, fully Bayesian spectral disentangling of spectroscopic binaries in JAX.

> [!WARNING]
> **Status: pre-alpha, under active development, API unstable.**
> Expect breaking changes without notice. Not yet suitable for production science.

## Quickstart

An example dataset ships inside the package, so the first thing you run needs no data of
your own and no network:

```python
import albireo as ab

dataset, truth = ab.load_example("sb2_sim", with_truth=True)   # simulated SB2, 12 epochs
grid = ab.LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))

model = ab.MarginalOrbitModel(
    grid, dataset,
    light_fractions=truth["light_fractions"],
    lsf_sigma_v={"DEMO": 6.5},
    v_rel_max_kms=160.0,
)
priors, init = ...              # elided; the runnable version is examples/00_quickstart.py
fit = ab.run_map(model.model(priors), init=init)
marginal = model.marginal(fit.params)

ab.plot_spectra(grid, marginal.d_hat, std=ab.spectra_std(marginal))
```

`K_1` and `K_2` come back to better than 0.05% in about twenty seconds on a laptop, and no
per-epoch radial velocity was measured anywhere: the orbit is inferred from the spectra
directly, with the component spectra marginalized out in closed form. Run
[`examples/00_quickstart.py`](examples/00_quickstart.py) for the whole thing, or read
[`docs/quickstart.md`](docs/quickstart.md) for the annotated walk through it.

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
  lines. K₁ can be integrated out rather than assumed — the fix for the failure mode the
  literature reports, where an error in the primary's semi-amplitude puts spurious features in
  the recovered secondary *and* inflates the detection statistic while doing so.
- **Detection limits with a false-alarm rate attached**, not just a peak. `ab.detection_limit`
  resimulates the observed dataset through its own operators, scans hundreds of companion-free
  draws for the null distribution, and injects a ladder of light fractions for completeness —
  producing the sentence a referee asks for: *any companion contributing more than X% of the
  light would have been detected at 95% confidence*
  ([`examples/05_detection_limit.py`](examples/05_detection_limit.py)).
- **Nebular emission modelled, not masked** — massive stars sit in H II regions, and the emission
  lines that come with them fill the disentangled line cores. albireo fits them as a component at
  rest in the barycentric frame with a free per-epoch amplitude, so the stellar spectra come back
  uncontaminated *and* complete. On a simulated SB2, leaving the line in costs 11.5% of the Hβ
  equivalent width and — because a static line is a component with *K* = 0 — 59% of K₂;
  modelling it costs 0.14% and 0.3%
  ([`examples/04_nebular.py`](examples/04_nebular.py)).
- **Reads archival spectra directly** — `albireo.io` turns a directory of ESO Phase-3 or
  IRAF-style FITS into a `Dataset`, and `albireo.preprocess` supplies the things reduced
  spectra are missing: a continuum, an inverse variance, masks, and one shared wavelength grid.
  Columns are identified by their IVOA utypes rather than their names, because across seven
  instruments no two ESO collections agree on the names, the units, the extension or the
  wavelength scale — and the obvious shortcut is a trap: UVES labels its *sky-background*
  column with the same UCD HARPS puts on its *flux*.
- **Fetches the data too, by name.** `albireo.archive` queries the ESO archive and downloads
  resumably, stdlib-only. For BLOeM — 929 SMC stars, ~25 epochs each, 59 published SB2s whose
  disentangling is still listed as future work — `ab.bloem_catalogue(binary_class="SB2")` gives
  the targets and `ab.bloem_spectra("1-037")` gives one star's epochs, resolving the survey
  identifier through VizieR because the archive files these under their Gaia DR3 source ids
  ([`examples/06_bloem.py`](examples/06_bloem.py)).

## Installation

albireo is **not yet on PyPI**. Install from a clone:

```bash
git clone https://github.com/tjayasinghe/albireo.git
cd albireo
pip install -e ".[dev]"
```

Python 3.12+ is required (current jaxlib no longer ships 3.11 wheels). JAX's x64 mode is enabled by the package at import time, so all
computation is done in float64 — you do not need to set `JAX_ENABLE_X64` yourself.

Two optional extras, both genuinely optional — the core never imports either, so a fit on a
headless node needs neither:

- `pip install -e ".[io]"` — astropy, for reading and writing FITS (`albireo.io`).
- `pip install -e ".[plots]"` — matplotlib and arviz, for `albireo.plotting` and the
  posterior diagnostics.

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
[the benchmarks](docs/benchmarks.md) before trusting one on real data), nebular
emission components (`nebular=True` plus the `log_nebular_amp` site, with per-pixel
prior profiles to confine them to their lines), and the SB1
faint-companion K₂ scan (`ab.k2_scan`).

`ab.Disentangler` is a declarative front end over all of it — **experimental**, because a
vocabulary is expensive to change once people depend on it. It derives the four things the
low-level path makes you get right yourself (the solver's velocity budget, the grid margin,
the conjunction phase, and the smoothness hyperparameters by empirical Bayes), and refuses
to derive the ones where a default would be a scientific claim. It is a compiler rather
than a wall: `dis.explain()` prints every derivation and `dis.expert()` hands back the
`(model, priors, init)` triple, so the core below stays the supported surface.

```python
dis = ab.Disentangler(
    dataset,
    components=[ab.Star("primary", light=0.62), ab.Star("secondary", light=0.38)],
    orbit=ab.Orbit(period=ab.Between(5.5, 6.5), k=ab.Between([10.0, 10.0], [90.0, 90.0])),
    lsf={"DEMO": 6.5},
)
fit = dis.fit()          # phase scan -> MAP + ML-II -> residual check
post = fit.sample(seed=0)  # Laplace mass matrix -> NUTS
```

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
search = dict(
    orbit=sb1_solution,
    k1=12.0,
    k1_sigma=0.4,  # integrate K_1 out rather than condition on it (§6.1)
    k2_grid=jnp.arange(10.0, 150.0, 2.0),
    light_fractions=(0.95, 0.05),  # explicit — see the light-ratio policy
    lsf_sigma_v={"HERMES": 4.0},
    prior=spec_prior,
    v_rel_max_kms=250.0,
)
scan = ab.k2_scan(grid, ds, **search)
scan.k2_peak, scan.detection_peak, scan.companion  # + std, null loglike, ...

# ...and what that peak is worth: a measured false-alarm rate and a limit (§6.2)
limit = ab.detection_limit(
    grid, ds, k2_true=scan.k2_peak, ell2_grid=np.array([0.005, 0.01, 0.02, 0.04]), **search
)
print(limit.summary())
print(limit.false_alarm_probability(scan.detection_peak))
```

## Documentation

- [`docs/quickstart.md`](docs/quickstart.md) — the five-minute version, on packaged data.
- [`examples/`](examples/) — executable tutorials (quickstart, SB2 end-to-end, K₂ scan,
  detection limits, nebular contamination, and HR 6819 on real archival FEROS spectra);
  everything but the
  HR 6819 script runs in CI. Narrative versions in [`docs/tutorials/`](docs/tutorials/).
- [`docs/design.md`](docs/design.md) — architecture, data model, and the shape of the inference
  problem.
- [`docs/math.md`](docs/math.md) — the disentangling likelihood, analytic marginalization of the
  component spectra, and the orbital parameterization.
- [`docs/benchmarks.md`](docs/benchmarks.md) — the running validation and performance record.
- [`docs/roadmap.md`](docs/roadmap.md) — where albireo is going and why, with the non-goals recorded.
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
