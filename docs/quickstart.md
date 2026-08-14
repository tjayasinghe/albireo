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

The injected truth is *P* = 6.0 d, *K* = 42.0 and 63.0 km/s, *e* = 0.15, ω = 0.7. About a
minute, most of it JAX compiling the model on its first call. **No per-epoch radial velocity
was measured anywhere in that**: the orbit is inferred from the spectra directly, with the
component spectra integrated out in closed form.

Four things in that fit were derived rather than typed, and each is something the low-level
path makes you get right yourself. `dis.explain()` prints all of them:

```python
print(dis.explain())
```

- **The velocity budget**, which sets the solver's bandwidth. It has to bound the largest
  relative velocity the *priors* allow, not the one the answer turns out to have — too small
  and the sampler stalls against a guard it cannot see. It comes from the `k` priors' own
  support, itemized, and narrowing them is what makes a fit cheaper.
- **The model grid**, wide enough for that budget *plus* the LSF kernel radius. Short of
  that margin the shifted model runs off the end of the grid and the fit quietly loses the
  flux there.
- **The conjunction phase**, located by a 41-point scan before anything is optimized. The
  marginal likelihood is sharply multimodal in phase — 10⁵ nats between the best and worst
  here — and L-BFGS started in the wrong trough converges confidently to the wrong answer.
- **The smoothness hyperparameters**, fitted by empirical Bayes and then frozen. The report
  above flags any that did not move from their starting value, which means the hyperprior
  rather than the data is setting that component's smoothness.

## Sample it

```python
post = fit.sample(seed=0)
print(post.summary())
print(post.star("secondary"))   # {'k': 62.99, 'k_std': 0.08, 'k_hdi': (62.84, 63.11), ...}
```

NUTS over the orbit, with the component spectra still marginalized and the smoothness
hyperparameters held at their ML-II values. That last part is a plug-in approximation and
the summary says so: the orbital credible intervals do not include smoothness uncertainty.

## Look at what came out

```python
d_hat = fit.spectra()          # (2, n_pix) component deviation spectra
std = fit.std()                # pointwise posterior uncertainty
fig, axes = ab.plot_spectra(dis.grid, d_hat, std=std, truth=truth["components"])
```

**Read the band, not the line.** Between the lines, and anywhere the epochs give little
leverage, the recovered spectrum is set by the smoothness prior rather than by the data.
The uncertainty band is what tells you which is which, and producing it honestly is the
reason this package exists. In particular the *k* = 0 (constant offset) mode of each
component spectrum is exactly unconstrained by the data — see
[the degeneracy section](math.md) — so it is the light-weighted sum, `fit.composite()`, not
each individual continuum level, that the data actually measure.

Check the noise model while you are here:

```python
fig, axes = ab.plot_residual_zscores(
    dis.model.problem_at(fit.theta), d_hat, bjd=dataset.bjd
)
```

Three panels: the residual distribution against a unit normal, the per-epoch scatter, and
the per-epoch lag-1 autocorrelation. The third is the one that catches the failure the
other two miss — correlated pixels inflate every uncertainty derived from the fit and are
invisible in a histogram. `fit.z_rms` is the one-number version, printed in every summary.

## Keep it

```python
fit.write_spectra("spectra.fits")           # needs albireo[io]
ab.save_fit(fit.result, "quickstart_map.npz")
ab.write_ascii("spectra.txt", dis.grid, d_hat, std)   # no dependencies
```

The FITS and ECSV writers record the light fractions and the prior hyperparameters in the
header, because the recovered line depths are only interpretable next to them.

## When you need more than the façade

`Disentangler` is a **compiler**, not a wall: it emits the expert path, and hands it to you.

```python
model, priors, init = dis.expert()
```

That triple is exactly what [`MarginalOrbitModel`](api/inference.md) and `ab.run_map` take,
so anything the façade does not name — per-epoch jitter, AR(1) correlated noise, inferred
light fractions, inferred LSF widths — is three lines away rather than a rewrite. The
façade is marked **experimental** while its vocabulary settles; the low-level API below it
is not, and is not going anywhere.

## Where to go next

- [Disentangle an SB2 end to end](tutorials/sb2-end-to-end.md) — the same problem, but
  sampled with NUTS rather than stopped at MAP. That is where the posterior comes from.
- [Find a hidden companion](tutorials/k2-scan.md) — the SB1 faint-companion scan.
- [Bring your own spectra](tutorials/real-data.md) — FITS files to a `Dataset`.

## Two things that will bite you

**There is no default light fraction, and this is deliberate.** With constant light
fractions the data constrain only the products ℓᵢ·dᵢ, so a value assumed silently would
propagate into every line depth and every abundance derived from it — and nothing in the
fit could tell you it was wrong. `Star(light=...)` is required, the star lights must sum to
1, and every summary and FITS header repeats the value under `Assumed, not measured`. Quote
it next to any semi-amplitude you publish.

**Zero eccentricity is a special point, and the façade handles it for you.** The orbit is
parameterized by `secosw` = √e·cos ω and `sesinw` = √e·sin ω, which has no boundary at
*e* = 0 and no wrap in ω — but it is singular at exactly the origin, where ω is undefined
and the gradient is NaN. numpyro reports that as `Cannot find valid initial parameters`,
which does not obviously point at the cause. A free eccentricity is therefore never started
at the origin, and `ecc=ab.Fixed(0.0)` — a genuinely circular orbit — is handled *exactly*,
by not sampling those two sites at all. Working at the low level, initialize slightly off
the origin yourself, even for a binary you believe is circular.
