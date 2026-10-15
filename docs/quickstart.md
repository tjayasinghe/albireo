# Quickstart

This page fits the example dataset that ships with the package. It needs no other data
and no network access, and takes a few minutes on one CPU.

albireo is not yet on PyPI, so install it from a clone. The figures on this page need the
`plots` extra; nothing else here needs an optional dependency.

```bash
git clone https://github.com/tjayasinghe/albireo.git
cd albireo
pip install -e ".[plots]"
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
component spectra, so every result below can be compared with the injected values. It
requires neither astropy nor an archive account.

## Fit the orbit

```python
dis = ab.Disentangler(
    dataset,
    components=[ab.Star("primary", light=0.62), ab.Star("secondary", light=0.38)],
    orbit=ab.Orbit(period=ab.Between(5.5, 6.5), k=ab.Between([10.0, 10.0], [90.0, 90.0])),
    lsf={"DEMO": 6.5},                       # sigma [km/s]; LSF.from_resolution(R); LSF.per_epoch()
)
fit = dis.fit()
print(fit.summary())
```

```
Best fit (maximum a posteriori) of the orbit, 300 steps (stopped at the step cap)
  -ln posterior -28052.6 (unconstrained coordinates, up to a constant), |gradient| 383
  conjunction scan: t_conj = 0.71429, 1.31e+05 nats between the best and worst phase
  semi-amplitude scan: best of 176 trials at primary 41.5, secondary 62.3 km/s; started from it, +1.57e+04 nats over the declared start (50.0, 50.0)

  period    6.000167 d
  ecc       0.1512    omega 0.6963 rad
  K(primary)    41.978 km/s   (light 0.62)
  K(secondary)    62.979 km/s   (light 0.38)

Regularization weights of the component spectra, fitted by maximizing the marginal
likelihood (ML-II) and held fixed for sampling: curvature weight tau and continuum weight
eta, start -> fitted, with the move in ln tau in hyperprior standard deviations
  primary        tau       300 ->       638 (+0.25 sigma)   eta        5 ->     10.2
  secondary      tau       300 ->  1.15e+03 (+0.45 sigma)   eta        5 ->     3.25

  residual z-score RMS 0.997

Assumed, not measured:
  light fractions    primary=0.62  secondary=0.38
      only l_i * d_i is observable, so every recovered depth scales as 1/l_i.
  smoothness starts  primary=300  secondary=300
      the starting curvature weight tau of each component's regularization. ML-II fits the
      weights, but which optimum it finds depends on where tau starts.
```

The injected values are *P* = 6.0 d, *K* = 42.0 and 63.0 km/s, *e* = 0.15, ω = 0.7. The
fit takes one to three minutes on one CPU, much of it JAX compilation on the first call.
"Stopped at the step cap" means that the optimizer's threshold, an absolute norm of the
gradient, was not reached in 300 steps. It is not a failure here: the best-fit orbit agrees
with the posterior means below to within about a tenth of their standard deviations.
Differences of log likelihood are in nats; a difference of 1 nat corresponds to a
chi-square difference of 2. No per-epoch radial velocity is measured at any stage: the orbit
is inferred from the spectra directly, with the component spectra integrated out in closed
form.

Four quantities in that fit were derived from the declaration. The low-level interface
requires the user to set each of them. `dis.explain()` prints them:

```python
print(dis.explain())
```

- **The velocity budget**, which sets the solver's bandwidth. It must bound the largest
  relative velocity the priors allow, not the value the fit converges to. It is derived
  from the support of the `k` priors, and narrowing those priors reduces the cost of a fit.
- **The model grid**, widened by that budget plus the LSF kernel radius so that the
  shifted model stays within the grid.
- **The conjunction phase**, located by a 42-point scan before optimization. The marginal
  likelihood is strongly multimodal in phase (about 10⁵ nats between the best and worst
  phase here), and an optimizer started in the wrong mode converges to the wrong solution.
- **The smoothness hyperparameters**, the weights of the curvature and continuum penalties
  on each component spectrum. They are fitted to the data by maximizing the marginal
  likelihood (ML-II, or empirical Bayes; see the
  [method overview](method.md#7-the-regularization-strength)) and then held fixed. The
  report flags any weight that did not move from its starting value, which indicates that
  the hyperprior, not the data, sets that component's smoothness.

## Sample the posterior

```python
post = fit.sample(seed=0)
print(post.summary())
print(post.star("secondary"))   # {'k': 62.979, 'k_std': 0.088, 'k_hdi': (62.811, 63.141), ...}
```

```
NUTS posterior: 2 chain(s) x 500 draws of 6 parameters (k, period, secosw, sesinw, t_conj)
  divergences: 0

                                   mean         sd   95% HDI                     R-hat    ESS
  period [d]                   6.000104   0.001110   [5.997982, 6.002377]        0.999   1655
  t_conj [d]                    0.62530    0.00225   [0.62078, 0.62976]          0.999   1982
  ecc                            0.1513     0.0009   [0.1496, 0.1533]            0.999   2339
  omega [rad]                    0.6957     0.0056   [0.6854, 0.7075]            0.999   1881
  K(primary) [km/s]              41.977      0.053   [41.869, 42.076]            0.998   1884
  K(secondary) [km/s]            62.979      0.088   [62.811, 63.141]            1.000   1583

  The regularization weights were held at their ML-II values, so these intervals
  do not include their uncertainty (a plug-in approximation, not a marginalization).

Assumed, not measured:
  light fractions    primary=0.62  secondary=0.38
      only l_i * d_i is observable, so every recovered depth scales as 1/l_i.
  smoothness starts  primary=300  secondary=300
      the starting curvature weight tau of each component's regularization. ML-II fits the
      weights, but which optimum it finds depends on where tau starts.
```

This runs NUTS, a Markov chain Monte Carlo sampler, over the orbital parameters. At every
step the component spectra are solved for exactly and integrated out, so they never enter
the sampler, and the smoothness weights are held at their fitted values. The summary states
this plug-in approximation: the orbital intervals do not include the uncertainty of the
smoothness weights. It also reports two convergence diagnostics per parameter, R-hat and
the effective sample size; the [method overview](method.md#8-uncertainties-on-the-orbit)
describes the stages and the [glossary](method.md#14-glossary) defines the terms.

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

Wherever the epochs constrain it weakly (beyond the edges of the data, in masked pixels, and
for features broader than the stars' velocity separation), the recovered spectrum is set by
the smoothness prior rather than by the data, and the uncertainty band widens there. In
particular the constant (*k* = 0) mode of each component
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
uncertainty derived from the fit and are not visible in a histogram. The third panel is
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
features the declarative interface does not expose (per-epoch jitter, a per-epoch response
polynomial, inferred light fractions, inferred LSF widths) are added at that level. Correlated
(AR(1)) noise is declared directly, with `Disentangler(noise_correlation=...)`. The sites the
declaration holds constant are in `dis.fixed` and are passed as
`model.model(priors, fixed=dis.fixed)`. The declarative interface is marked experimental
because its names may still change. The low-level interface is the supported one.

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
