# Quickstart

Five minutes, no data of your own, no network.

```bash
pip install albireo
```

Then:

```python
import albireo as ab

dataset, truth = ab.load_example("sb2_sim", with_truth=True)
print(dataset.summary())
```

```
Dataset: 12 epochs, frame='topocentric'
  BJD_TDB 0.00000 to 14.85000 (span 14.85000 d)
  instruments (1):
    DEMO  12 epochs, 4505.00-4554.98 A, 10008 px
  good pixels: 10008 / 10008 (100.0%)
```

That example ships inside the package — it is a simulated double-lined binary with a known
injected truth, so everything below can be checked against the right answer. It needs no
download, no astropy, and no archive account.

## Fit it

```python
import jax.numpy as jnp
import numpyro.distributions as dist

grid = ab.LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))

model = ab.MarginalOrbitModel(
    grid,
    dataset,
    light_fractions=truth["light_fractions"],   # assumed, never inferred here
    lsf_sigma_v={"DEMO": 6.5},                  # Gaussian LSF width [km/s]
    v_rel_max_kms=160.0,                        # velocity budget for the solver structure
)

priors = {
    "period": dist.Uniform(5.5, 6.5),
    "t_conj": dist.Uniform(-1.0, 1.0),
    "secosw": dist.Uniform(-0.8, 0.8),
    "sesinw": dist.Uniform(-0.8, 0.8),
    "k": dist.Uniform(jnp.array([10.0, 10.0]), jnp.array([90.0, 90.0])),
    "log_tau": dist.Normal(jnp.log(300.0) * jnp.ones(2), 3.0),
    "log_eta": dist.Normal(jnp.log(5.0) * jnp.ones(2), 3.0),
}
init = {
    "period": 6.05, "t_conj": 0.1, "secosw": 0.2, "sesinw": 0.2,
    "k": jnp.array([38.0, 58.0]),
    "log_tau": jnp.log(300.0) * jnp.ones(2),
    "log_eta": jnp.log(5.0) * jnp.ones(2),
}

fit = ab.run_map(model.model(priors), init=init)
print(fit.params["k"], truth["k"])     # [41.98 63.01]  vs  [42.0, 63.0]
```

About twenty seconds, most of it JAX compiling the model on its first call. No per-epoch
radial velocity was measured anywhere in that: the orbit is inferred from the spectra
directly, with the component spectra integrated out in closed form.

## Look at what came out

```python
marginal = model.marginal(fit.params)
d_hat = marginal.d_hat                      # (2, n_pix) component deviation spectra
std = ab.spectra_std(marginal)              # pointwise posterior uncertainty

fig, axes = ab.plot_spectra(grid, d_hat, std=std, truth=truth["components"])
```

**Read the band, not the line.** Between the lines, and anywhere the epochs give little
leverage, the recovered spectrum is set by the smoothness prior rather than by the data.
The uncertainty band is what tells you which is which, and producing it honestly is the
reason this package exists. In particular the *k* = 0 (constant offset) mode of each
component spectrum is exactly unconstrained by the data — see
[the degeneracy section](math.md) — so it is the light-weighted sum, not each individual
continuum level, that the data actually measure.

Check the noise model while you are here:

```python
fig, axes = ab.plot_residual_zscores(model.problem_at(fit.params), d_hat, bjd=dataset.bjd)
```

Three panels: the residual distribution against a unit normal, the per-epoch scatter, and
the per-epoch lag-1 autocorrelation. The third is the one that catches the failure the
other two miss — correlated pixels inflate every uncertainty derived from the fit and are
invisible in a histogram.

## Keep it

```python
ab.save_fit(fit, "quickstart_map.npz")
ab.write_ascii("spectra.txt", grid, d_hat, std)          # no dependencies
ab.write_spectra("spectra.fits", grid, d_hat, std)       # needs albireo[io]
```

The FITS and ECSV writers record the light fractions and the prior hyperparameters in the
header, because the recovered line depths are only interpretable next to them.

## Where to go next

- [Disentangle an SB2 end to end](tutorials/sb2-end-to-end.md) — the same problem, but
  sampled with NUTS rather than stopped at MAP. That is where the posterior comes from.
- [Find a hidden companion](tutorials/k2-scan.md) — the SB1 faint-companion scan.
- [Bring your own spectra](tutorials/real-data.md) — FITS files to a `Dataset`.

## Two things that will bite you

**Do not start at exactly zero eccentricity.** The orbit is parameterized by
`secosw` = √e·cos ω and `sesinw` = √e·sin ω, which has no boundary at *e* = 0 and no wrap
in ω — but it is singular at exactly the origin, where ω is undefined and the gradient is
NaN. Initialize slightly off it, as above, even for a binary you believe is circular.
numpyro reports this as `Cannot find valid initial parameters`, which does not obviously
point at the cause.

**There is no default light ratio, and this is deliberate.** With a constant light ratio
the data constrain only the products ℓᵢ·dᵢ, so a light ratio assumed silently would
propagate into every line depth and every abundance derived from it. albireo makes you
choose, or infer it from epochs where the ratio actually varies.
