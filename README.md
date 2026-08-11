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

## Planned API

> [!NOTE]
> **Design sketch only.** None of this is implemented yet; it is here to show the intended shape of
> the interface and will change.

```python
import albireo as ab

ds = ab.Dataset([ab.EpochData(wave=w, flux=f, ivar=iv, bjd=t, v_bary=vb, instrument="HERMES"), ...])
model = ab.Disentangler(
    dataset=ds,
    grid=ab.LogGrid.from_wavelength_range(4000.0, 6800.0, dv_kms=1.0),
    components=[ab.Star("A"), ab.Star("B")],
    orbit=ab.Keplerian(period=..., t_peri=..., ecc=..., omega=..., k1=..., k2=...),
    light_ratio=ab.FixedLight([0.62, 0.38]),
)
map_fit = model.fit_map()
posterior = model.sample(num_warmup=1000, num_samples=1000)
spectra = model.conditional_spectra(posterior)
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
