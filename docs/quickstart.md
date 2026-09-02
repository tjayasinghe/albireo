# Quickstart

This page fits the example dataset that ships with the package. It needs no data of your
own and no network, and takes a few minutes on a laptop.

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

The example is a simulated double-lined binary with a known injected orbit and known
component spectra, so every result below can be compared with the truth. It requires
neither astropy nor an archive account.

## Fit the orbit

```python
dis = ab.Disentangler(
    dataset,
    components=[ab.Star("primary", light=0.62), ab.Star("secondary", light=0.38)],
    orbit=ab.Orbit(period=ab.Between(5.5, 6.5), k=ab.Between([10.0, 10.0], [90.0, 90.0])),
    lsf={"DEMO": 6.5},                       # Gaussian sigma [km/s]; LSF.from_resolution(R) too
)
fit = dis.fit()
print(fit.summary())
```

```
MAP fit: potential -28052.6, |grad| 9.11, 300 steps (stopped at the step cap)
  conjunction scan: t_conj = 0.73171, 1.15e+05 nats between the best and worst phase

  period    6.000167 d
  ecc       0.1512    omega 0.6963 rad
  K(primary)    41.978 km/s   (light 0.62)
  K(secondary)    62.979 km/s   (light 0.38)

ML-II smoothness (empirical Bayes: fitted here, then frozen for sampling)
  primary        tau       300 ->       638 (+0.25 sigma)   eta        5 ->     10.2
  secondary      tau       300 ->  1.15e+03 (+0.45 sigma)   eta        5 ->     3.25

  residual z-score RMS 0.997

Assumed, not measured:
  light fractions    primary=0.62  secondary=0.38
      only l_i * d_i is observable, so every recovered depth scales as 1/l_i.
```

The injected truth is *P* = 6.0 d, *K* = 42.0 and 63.0 km/s, *e* = 0.15, ω = 0.7. The fit
takes about a minute, most of it JAX compilation on the first call. No per-epoch radial
velocity is measured at any stage: the orbit is inferred from the spectra directly, with
the component spectra integrated out in closed form.

Four quantities in that fit were derived from the declaration rather than supplied, and
each is something the low-level interface requires the user to set. `dis.explain()`
prints all of them:

```python
print(dis.explain())
```

- **The velocity budget**, which sets the solver's bandwidth. It must bound the largest
  relative velocity the priors allow, not the value the fit converges to; it is derived
  from the support of the `k` priors, and narrowing those priors reduces the cost of a fit.
- **The model grid**, which is widened by that budget plus the LSF kernel radius so that
  the shifted model does not run off the grid.
- **The conjunction phase**, located by a 41-point scan before optimization. The marginal
  likelihood is strongly multimodal in phase (about 10⁵ nats between the best and worst
  phase here), and an optimizer started in the wrong trough converges to the wrong answer.
- **The smoothness hyperparameters**, fitted by empirical Bayes and then frozen. The
  report flags any hyperparameter that did not move from its starting value, which
  indicates that the hyperprior rather than the data is setting that component's
  smoothness.

## Sample the posterior

```python
post = fit.sample(seed=0)
print(post.summary())
print(post.star("secondary"))   # {'k': 62.99, 'k_std': 0.08, 'k_hdi': (62.84, 63.11), ...}
```

This runs NUTS over the orbital parameters with the component spectra marginalized and
the smoothness hyperparameters held at their empirical-Bayes values. The latter is a
plug-in approximation, and the summary states it: the orbital credible intervals do not
include the uncertainty in the smoothness hyperparameters.

## Inspect the component spectra

```python
import numpy as np

d_hat = fit.spectra()          # (2, n_pix) component deviation spectra
std = fit.std()                # pointwise posterior standard deviation

# The truth was generated on its own grid, which is not the grid the model was solved on:
# `dis.grid` is widened by the velocity budget and the LSF radius and takes its sampling
# from the data. Resample before overlaying.
truth_grid = ab.LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))
truth_on_model = np.stack(
    [
        np.interp(dis.grid.wave, truth_grid.wave, component, left=0.0, right=0.0)
        for component in truth["components"]
    ]
)

fig, axes = ab.plot_spectra(dis.grid, d_hat, std=std, truth=truth_on_model)
```

Between the lines, and wherever the epochs provide little leverage, the recovered
spectrum is set by the smoothness prior rather than by the data. The uncertainty band
identifies those regions. In particular the constant (*k* = 0) mode of each component
spectrum is exactly unconstrained by the data (see the
[degeneracy section](math.md#5-degeneracies-and-identifiability)), so the data determine
the light-weighted sum, `fit.composite()`, rather than each component's continuum level.

The noise model can be checked from the same fit:

```python
fig, axes = ab.plot_residual_zscores(
    dis.model.problem_at(fit.theta), d_hat, bjd=dataset.bjd
)
```

The figure has three panels: the residual distribution against a unit normal, the
per-epoch scatter, and the per-epoch lag-1 autocorrelation. Correlated pixels inflate every
uncertainty derived from the fit and are not visible in a histogram; the third panel is
the diagnostic for them. `fit.z_rms` is the scalar summary printed in every fit summary.

## Save the results

```python
fit.write_spectra("spectra.fits")           # needs albireo[io]
ab.save_fit(fit.result, "quickstart_map.npz")
ab.write_ascii("spectra.txt", dis.grid, d_hat, std)   # no optional dependency
```

The FITS and ECSV writers record the light fractions and the prior hyperparameters in the
header, because the recovered line depths are interpretable only together with them.

## Beyond the declarative interface

`Disentangler` assembles the low-level model and returns it on request:

```python
model, priors, init = dis.expert()
```

That triple is what [`MarginalOrbitModel`](api/inference.md) and `ab.run_map` take, so
features the declarative interface does not expose (per-epoch jitter, AR(1) correlated
noise, inferred light fractions, inferred LSF widths) are added at that level. The
declarative interface is marked experimental while its vocabulary settles; the low-level
interface is the supported one.

## Next steps

- [Disentangle an SB2 end to end](tutorials/sb2-end-to-end.md): the same problem sampled
  with NUTS at the low level.
- [Search for a faint companion](tutorials/k2-scan.md): the SB1 companion scan.
- [Read your own spectra](tutorials/real-data.md): from FITS files to a `Dataset`.
- [Scientific background](science.md): the methods and their references.

## Two conventions to note

**There is no default light fraction.** With constant light fractions the data constrain
only the products ℓᵢ·dᵢ, so a silently assumed value would propagate into every line
depth and every quantity derived from it without any diagnostic in the fit.
`Star(light=...)` is therefore required, the stellar light fractions must sum to 1, and
every summary and FITS header repeats the value under `Assumed, not measured`. Quote the
light fractions alongside any published semi-amplitude.

**Zero eccentricity is a singular point of the parameterization.** The orbit is
parameterized by `secosw` = √e·cos ω and `sesinw` = √e·sin ω, which has no boundary at
*e* = 0 and no wrap in ω but is singular at the origin, where ω is undefined and the
gradient is not finite. numpyro reports this as `Cannot find valid initial parameters`.
The declarative interface never starts a free eccentricity at the origin, and
`ecc=ab.Fixed(0.0)` handles a circular orbit exactly by not sampling those two sites. At
the low level, initialize slightly off the origin even for a binary believed to be
circular.
