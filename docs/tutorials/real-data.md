# From archival FITS files to a `Dataset`

The other tutorials start from the simulator, where the spectra are already normalized,
already carry inverse variances, and already share a wavelength grid. Reduced archival
spectra do none of those things. This page covers the steps in between.

See the [science overview](../science.md) for background and references.

The worked example is [`examples/03_hr6819_real_data.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/03_hr6819_real_data.py),
which runs on the 51 public FEROS spectra of HR 6819 (ESO programme 073.D-0274(A)):

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

Reading spectra needs astropy: `pip install "albireo[io]"`. Nothing else in albireo does.

The sections below describe what those four lines do, and why each step is a decision
rather than a default.

## 1. What the header must supply

`ab.read_spectrum` returns a `RawSpectrum`, the file as it stands and before any science
judgement, so that the assumptions already applied to it can be inspected:

```python
raw = ab.read_spectrum("data/hr6819/ADP.2016-09-20T12-03-37.453.fits")
print(raw.summary())
```

```
ADP...fits: 189628 px 3527.2-9216.1 A (air), no error array, FEROS, R=48000,
barycentric, v_bary=-21.708 km/s, bjd=2453243.51884 (BJD_TDB from TMID)
```

Three header facts decide whether the recovered velocities are right, and the reader warns
rather than guessing when a file is silent about any of them.

**The frame** (`SPECSYS`). `BARYCENT` means the pipeline already applied the correction, so
albireo must not apply it again; it composes the barycentric motion into the *telluric*
component instead. An error here offsets every velocity by up to 25 km/s and raises nothing.

**The applied barycentric velocity.** This is needed even in the barycentric frame, because
telluric lines are at rest topocentrically and therefore move barycentrically. It is taken
from the pipeline's own keyword (`ESO DRS BARYCORR`, `ESO DRS BERV`, …), because the
pipeline's value defines the frame of the delivered wavelengths; astropy is only the
fallback. On the HR 6819 files the two agree to 0.017 km/s.

!!! note "Verifying the sign on new data"
    Cross-correlate a strong telluric band (the O₂ A band at 7580–7720 Å) across epochs with
    very different corrections. In a barycentric-frame spectrum the band should move by
    *plus* the correction. On the HR 6819 files this gives a slope of 0.9993 with 0.14 km/s
    scatter, which is how the convention in `albireo.data` was checked against observation
    rather than against itself.

**The time.** Converted to BJD_TDB at mid-exposure. The barycentric light-travel correction
swings by ±8.3 minutes over a year; on a 40-day orbit with K = 60 km/s that is a 0.055 km/s
systematic which never averages out, because it is a function of the observing date.

## 2. Continuum

albireo models `1 + Σ lᵢ dᵢ` around a *unit* continuum, and its per-epoch response term is
fixed at build time rather than inferred, so the normalization done here is the one the fit
uses. ESO delivers these spectra with `CONTNORM = False`: raw merged-echelle ADU whose
response falls by a factor of 20 across 3850–4750 Å.

`ab.fit_continuum` fits **log(flux)** on a knot grid, iterating an asymmetric upper envelope
and then asymmetric sigma clipping. The log matters: a continuum is multiplicative, and in
the log a steep exponential response is a straight line, which lies in the *nullspace* of
the curvature penalty and is therefore free to represent. Fitting the flux itself, a stiff
smoother lags the gradient and the normalized spectrum comes out 30% wrong at the blue end.

The consequence is that the answer barely depends on `smooth_angstrom`. On these spectra the
97th percentile of the normalized flux sits at 1.007–1.011 in every 50 Å bin across the whole
20× gradient, whether the requested stiffness is 80 Å or 150 Å.

## 3. Inverse variances

The `ERR` column of these files is entirely `NaN`, and the header comment says so: *"Error
spectrum not available"*. `ab.estimate_ivar` measures the noise from the spectrum itself with
the DER_SNR estimator (Stoehr et al. 2008, the same recipe ESO uses for its own `SNR`
keyword), in wavelength bins, and fits `σ² = s²/continuum` so that the per-pixel weights are
smooth. Noisy weights bias a maximum-likelihood fit; a smooth `σ(λ)` does not.

!!! warning "Check the noise scale, because nothing else does"
    A scale error in the inverse variances propagates into every quoted uncertainty. Measure
    it before correcting it: fit once with no jitter site, then

    ```python
    from albireo.forward import data_residual_zscores
    z = data_residual_zscores(model.problem_at(theta), model.marginal(theta).d_hat)
    print(z.std())   # 1 if the inverse variances are calibrated
    ```

    If it is not 1, the instrument for that is a `log_jitter` site rather than rescaling
    `ivar` by hand, either one shared factor or one per epoch:

    ```python
    import jax.numpy as jnp
    import numpyro.distributions as dist

    priors["log_jitter"] = dist.Normal(0.0, 2.0)                             # shared
    priors["log_jitter"] = dist.Normal(jnp.zeros(len(ds)), 2.0).to_event(1)  # per epoch
    init["log_jitter"] = jnp.zeros(len(ds))   # run_map randomizes any site missing here
    ```

    A fitted jitter is preferable to a hand rescaling because the marginal likelihood's
    log-determinant term supplies the effective-degrees-of-freedom correction: it profiles
    to `α² = χ²/(N − p_eff)`, whereas `z.std()` is the uncorrected `√(χ²/N)`. How much they
    differ depends on how much of the spectrum the data determine: 0.4% on HR 6819 (an
    oversampled grid with a stiff fitted prior leaves only ~2900 of 19,876 model pixels
    data-determined), 4.6% in the weak-prior test fixture. Comparing the two gives
    `p_eff = N[1 − (z.std()/α̂)²]`.

    Then read the next warning, because a jitter that fits is not the same as a noise model
    that is right.

!!! danger "A jitter widens error bars, and can also move the answer"
    Inflating a diagonal noise model cannot represent a residual that is correlated across
    pixels, and on real spectra it usually is: imperfect continua, LSF mismatch, and, for a
    Be star, line profiles that change between epochs. Fitting `log_jitter` against that
    drives `data_residual_zscores` to ≈1 by construction, so the diagnostic stops warning
    while the condition it was warning about is untouched.

    On HR 6819 the effect is not confined to a wider interval around the same point.
    Per-epoch factors came out spanning 1.1–3.6, the noisiest exposures clustered in the
    first third of the 135-day baseline, and downweighting them moved the period by 174× the
    no-jitter formal error, both fits being genuine optima under their own weights. Adding a
    jitter is a change of model, and here the model change dominates the uncertainty it was
    meant to express. See [benchmarks](../benchmarks.md).

    The practical consequences: inspect the *shape* of the residuals rather than their
    scale, per epoch and per pixel rather than pooled; and take the error bar from the
    spread across independent wavelength windows and across defensible noise models. On this
    dataset that spread is 4–18× the formal errors however the noise is modelled.

## 4. Region, masks, and the quadratic cost of deleting pixels

A full echelle spectrum is far more pixels than a disentangling run needs. Choose a window
with photospheric lines from both components and no complications: for HR 6819, 4380–4600 Å
(He I 4388, He I 4471, Mg II 4481, Si III 4552/4568/4575), which has no Balmer core, no disc
emission (the Be star's disc varies, and albireo assumes one static spectrum per component),
and no telluric band within 1200 Å.

Trim the ends with `select_region`. For anything interior (telluric windows, interstellar
lines, a bad column) use `mask_ranges`, which sets `ivar = 0` and keeps the pixels. This is
not a style preference: albireo takes bin edges at midpoints between samples, so deleting an
interior block makes the two bracketing pixels absorb half the gap each, and the rebin row
support (a maximum) feeds the solver half-bandwidth, whose cost is quadratic. Deleting one
telluric band has been measured to take the half-bandwidth from 159 to 3067. `build_problem`
warns if it sees the pattern.

## 5. One wavelength grid

Pipelines that apply the barycentric correction by shifting *before* resampling give every
exposure its own grid: the 51 HR 6819 spectra share a 0.03 Å step, but their start
wavelengths span 0.78 Å and their lengths differ by tens of pixels, giving 28 distinct grids.

albireo handles that correctly on its own, by giving each distinct grid its own rebin
operator. Every group's assembly pre-pass is live in the same compiled graph, however, so 28
groups is a real cost. `ab.share_wavelength_grid` relabels them onto one:

```python
ds = ab.Dataset(ab.share_wavelength_grid(list(ds), atol_kms=0.05), frame=ds.frame)
```

It aligns by *index* rather than by nearest wavelength (a value search at a window edge can
slip a whole 1.4 km/s pixel), trims to the common overlap, and raises, quoting the measured
deviation, if the grids differ by more than `atol_kms`. On these files the residual is
0.007 km/s, a three-hundredth of a pixel. No flux value is touched: this is a relabelling
rather than a resampling, so the diagonal noise model still holds.

## 6. The model grid, and the margin it needs

```python
grid = ab.LogGrid.covering(ds, dv_kms=1.5, v_margin_kms=90.0, lsf_sigma_kms=2.65)
```

The grid must be wider than the data by the largest component shift **plus** the LSF kernel
radius. The shift part is evident; the LSF part is not. Inside that margin the shift and
convolution operators zero-fill, so pixels there are modelled with missing flux while still
carrying full weight. `LogGrid.covering` computes the margin, and `build_problem` warns if
weighted pixels still fall outside it.

Pick `dv_kms` no coarser than the finest native sampling. For a spectrograph with a constant
*wavelength* step that is the blue end: FEROS's 0.03 Å is 2.25 km/s at 4000 Å but 0.98 km/s
at 9200 Å, so a blue window and a red window want different grids. Run them as separate
problems rather than forcing one grid at the red end's resolution.

Convert the resolving power to a Gaussian sigma with `raw.lsf_sigma_kms`
(`c / (R · 2√(2 ln 2))`; FEROS R = 48000 → 2.652 km/s). Confusing FWHM with sigma is a factor
of 2.35 in the kernel radius.

## What the data do not constrain

For a non-eclipsing binary with constant light fractions, the likelihood only ever sees the
products `lᵢ · dᵢ` ([`math.md` §5.2](../math.md)). The light ratio is therefore an **input**,
and every recovered line depth scales as `1/lᵢ`. Supply it from an independent measurement,
and note which one: HR 6819's `f = 0.439 ± 0.013` is a *K-band* interferometric flux ratio
and does not transfer unchanged to 4400 Å.

The same section explains why each component's smooth envelope is prior-dominated: it is the
light-weighted *sum* that the data measure.
