# From archival FITS files to a `Dataset`

This page covers the steps that prepare reduced archival spectra for albireo, using the 51
public FEROS spectra of HR 6819. The other tutorials start from the simulator, whose spectra
are normalized, have inverse variances, and share a wavelength grid. Reduced archival spectra
have none of these properties.

See the [science overview](../science.md) for background and references.

The worked example is [`examples/03_hr6819_real_data.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/03_hr6819_real_data.py),
which runs on these spectra (ESO programme 073.D-0274(A)):

```bash
python scripts/download_hr6819.py       # 51 files, ~153 MB, no ESO login
python examples/03_hr6819_real_data.py
```

## The short version

```python
import albireo as ab

ds = ab.read_dataset(
    "data/hr6819/*.fits",
    instrument="FEROS",
    region=(4380.0, 4600.0),
    smooth_angstrom=120.0,
)
ds = ab.Dataset(ab.share_wavelength_grid(list(ds)), frame=ds.frame)
grid = ab.LogGrid.covering(ds, dv_kms=1.5, v_margin_kms=90.0, lsf_sigma_kms=2.65)
```

Only reading spectra requires astropy: `pip install -e ".[io]"`.

Each step in those four lines requires a decision.

## 1. What the header must supply

`ab.read_spectrum` returns a `RawSpectrum`, the file as delivered and before any scientific
judgement, so that the assumptions already applied to it can be inspected:

```python
raw = ab.read_spectrum("data/hr6819/ADP.2016-09-20T12-03-37.453.fits")
print(raw.summary())
```

```
ADP...fits: 189628 px 3527.2-9216.1 A (air), no error array, FEROS, R=48000,
barycentric, v_bary=-21.708 km/s, bjd=2453243.51884 (BJD_TDB from TMID)
```

Three header facts determine whether the recovered velocities are correct, and the file
reader warns rather than guessing when a file omits any of them.

**The frame** (`SPECSYS`). `BARYCENT` means the pipeline already applied the correction, so
albireo must not apply it again. It composes the barycentric motion into the telluric
component instead. A wrong frame offsets every velocity by up to 25 km/s, and no error is
raised.

**The applied barycentric velocity.** It is needed even in the barycentric frame, because
telluric lines are at rest topocentrically and so move barycentrically. It is taken from the
pipeline's own keyword (`ESO DRS BARYCORR`, `ESO DRS BERV`, …), since the pipeline's value
defines the frame of the delivered wavelengths, with astropy as the fallback. On the HR 6819
files the two agree to 0.017 km/s.

!!! note "Verifying the sign on new data"
    Cross-correlate a strong telluric band (the O₂ A band at 7580–7720 Å) across epochs with
    widely different corrections. In a barycentric-frame spectrum the band should move by
    the correction, with a positive sign. On the HR 6819 files this gives a slope of 0.9993
    with 0.14 km/s scatter, which confirms the convention in `albireo.data` against
    observation.

**The time.** It is converted to BJD_TDB at mid-exposure. The barycentric light-travel
correction varies by ±8.3 minutes over a year. On a 40-day orbit with K = 60 km/s that is a
0.055 km/s systematic that does not average out, because it is a function of the observing
date.

## 2. Continuum

albireo models `1 + Σ lᵢ dᵢ` around a unit continuum, and its per-epoch response term is
fixed at build time rather than inferred, so the normalization done here is the one the fit
uses. ESO delivers these spectra with `CONTNORM = False`, as raw merged-echelle ADU whose
response falls by a factor of 20 across 3850–4750 Å.

`ab.fit_continuum` fits log(flux) on a knot grid, iterating an asymmetric upper envelope
and then asymmetric sigma clipping. The logarithm is fitted because a continuum is
multiplicative, and in the logarithm a steep exponential response is a straight line, which
is in the nullspace of the curvature penalty and is therefore not penalized. When the flux
itself is fitted, a stiff smoother lags the gradient and the normalized spectrum is 30% wrong
at the blue end.

The normalization therefore depends only weakly on `smooth_angstrom`. On these spectra the
97th percentile of the normalized flux is 1.007–1.011 in every 50 Å bin across the whole 20×
gradient, whether the requested stiffness is 80 Å or 150 Å.

## 3. Inverse variances

The `ERR` column of these files contains only `NaN`, and the header comment states "Error
spectrum not available". `ab.estimate_ivar` measures the noise from the spectrum itself with
the DER_SNR estimator (Stoehr et al. 2008, the same method ESO uses for its own `SNR`
keyword), in wavelength bins, and fits `σ² = s²/continuum` so that the per-pixel weights are
smooth. Noisy weights bias a maximum-likelihood fit, and a smooth `σ(λ)` does not.

!!! warning "Check the noise scale"
    A scale error in the inverse variances propagates into every quoted uncertainty. Measure
    it before correcting it: fit once with no jitter site, then

    ```python
    from albireo.forward import data_residual_zscores
    z = data_residual_zscores(model.problem_at(theta), model.marginal(theta).d_hat)
    print(z.std())   # 1 if the inverse variances are calibrated
    ```

    If it is not 1, fit a `log_jitter` site rather than rescaling `ivar` manually, either one
    shared factor or one per epoch:

    ```python
    import jax.numpy as jnp
    import numpyro.distributions as dist

    priors["log_jitter"] = dist.Normal(0.0, 2.0)                             # shared
    priors["log_jitter"] = dist.Normal(jnp.zeros(len(ds)), 2.0).to_event(1)  # per epoch
    init["log_jitter"] = jnp.zeros(len(ds))   # run_map randomizes any site missing here
    ```

    A fitted jitter is preferable to a manual rescaling because the marginal likelihood's
    log-determinant term supplies the effective-degrees-of-freedom correction. The jitter
    profiles to `α² = χ²/(N − p_eff)`, whereas `z.std()` is the uncorrected `√(χ²/N)`. The
    difference depends on how much of the spectrum the data determine. It is 0.4% on HR 6819
    (an oversampled grid with a stiff fitted prior leaves only ~2900 of 19,876 model pixels
    data-determined) and 4.6% in the weak-prior test fixture. Comparing the two gives
    `p_eff = N[1 − (z.std()/α̂)²]`.

    A fitted jitter does not establish that the noise model is correct (see the next
    warning).

!!! danger "A jitter term can shift the fitted orbit"
    Inflating a diagonal noise model cannot represent a residual that is correlated across
    pixels, and on real spectra it usually is, because of imperfect continua, LSF mismatch
    and, for a Be star, line profiles that change between epochs. Fitting `log_jitter` to
    such a residual drives `data_residual_zscores` to ≈1 by construction, so the diagnostic
    passes while the underlying condition persists.

    On HR 6819 the jitter changes the estimate as well as its interval. The per-epoch
    factors spanned 1.1–3.6, the noisiest exposures were clustered in the first third of the
    135-day baseline, and downweighting them moved the period by 174× the no-jitter formal
    error. Both fits are optima under their own weights. Adding a jitter changes the model,
    and here that change dominates the uncertainty the jitter was meant to express. See
    [benchmarks](../benchmarks.md).

    Inspect the shape of the residuals rather than their scale, per epoch and per pixel, and
    take the error bar from the spread across independent wavelength windows and across
    defensible noise models. On this dataset that spread is 4–18× the formal errors however
    the noise is modelled.

## 4. Region, masks, and the quadratic cost of deleting pixels

A full echelle spectrum has far more pixels than a disentangling run needs. Choose a window
with photospheric lines from both components and no complications. For HR 6819 this is
4380–4600 Å (He I 4388, He I 4471, Mg II 4481, Si III 4552/4568/4575). The window has no
Balmer core, no disc emission (the Be star's disc varies, and albireo assumes one static
spectrum per component), and no telluric band within 1200 Å.

Trim the ends with `select_region`. For anything interior (telluric windows, interstellar
lines, a bad column) use `mask_ranges`, which sets `ivar = 0` and keeps the pixels. albireo
takes bin edges at midpoints between samples, so deleting an interior block extends each of
the two bracketing pixels over half the gap. The rebin row support (a maximum) sets the
solver half-bandwidth, in which the cost is quadratic. Deleting one telluric band was measured
to increase the half-bandwidth from 159 to 3067. `build_problem` warns if it detects the
pattern.

## 5. One wavelength grid

Pipelines that apply the barycentric correction by shifting before resampling give every
exposure its own grid. The 51 HR 6819 spectra share a 0.03 Å step, but their start
wavelengths span 0.78 Å and their lengths differ by tens of pixels, giving 28 distinct grids.

albireo handles this correctly by giving each distinct grid its own rebin operator, but every
group's assembly pre-pass is part of the same compiled graph, so 28 groups are costly.
`ab.share_wavelength_grid` relabels them onto one:

```python
ds = ab.Dataset(ab.share_wavelength_grid(list(ds), atol_kms=0.05), frame=ds.frame)
```

It aligns by index rather than by nearest wavelength (a value search at a window edge can
be off by a whole 1.4 km/s pixel) and trims to the common overlap. It raises an error
quoting the measured deviation if the grids differ by more than `atol_kms`. On these files
the residual is 0.007 km/s, a three-hundredth of a pixel. No flux value is changed. This is
a relabelling rather than a resampling, so the diagonal noise model still holds.

## 6. The model grid, and the margin it needs

```python
grid = ab.LogGrid.covering(ds, dv_kms=1.5, v_margin_kms=90.0, lsf_sigma_kms=2.65)
```

The grid must be wider than the data by the largest component shift plus the LSF kernel
radius. The LSF term is easy to overlook. Inside that margin the shift and convolution
operators zero-fill, so pixels there are modelled with missing flux while still having full
weight. `LogGrid.covering` computes the margin, and `build_problem` warns if weighted pixels
still fall outside it.

Choose `dv_kms` no coarser than the finest native sampling. For a spectrograph with a constant
wavelength step that is the blue end. FEROS's 0.03 Å is 2.25 km/s at 4000 Å but 0.98 km/s at
9200 Å, so a blue window and a red window need different grids. Run them as separate problems
rather than forcing one grid at the red end's resolution.

Convert the resolving power to a Gaussian sigma with `raw.lsf_sigma_kms`
(`c / (R · 2√(2 ln 2))`, which is 2.652 km/s for FEROS at R = 48000). Confusing FWHM with
sigma changes the kernel radius by a factor of 2.35.

## What the data do not constrain

For a non-eclipsing binary with constant light fractions, the likelihood depends only on the
products `lᵢ · dᵢ` ([`math.md` §5.2](../math.md)). The light ratio is therefore an input, and
every recovered line depth scales as `1/lᵢ`. Supply it from an independent measurement, and
record which one. HR 6819's `f = 0.439 ± 0.013` is a K-band interferometric flux ratio and
does not apply unchanged at 4400 Å.

The same section explains why each component's smooth envelope is prior-dominated: the data
measure only the light-weighted sum.
