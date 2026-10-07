# Disentangle a BLOeM SB2 end to end

This page describes how to disentangle a BLOeM double-lined binary, from the archive query to
the component spectra. BLOeM (Binarity at LOw Metallicity) is a VLT/FLAMES-GIRAFFE survey of
929 OBAF stars in the Small Magellanic Cloud, with eight fields, about 25 epochs each, an
intrinsic binary fraction above 70%, and 59 published double-lined systems. The survey team
still lists the disentangling of these systems as future work in its July 2026 review. The
spectra are public, no ESO account is needed, and one star's spectra are about 5 MB.

See the [science overview](../science.md) for background and references.

This makes BLOeM the best available target for the package. It is also the hardest worked
example, because none of the inputs are supplied: the archive does not use the survey's
names, the file layout differs from that of the other tutorials, the spectra are not
normalized, and there is no published orbit. The last changes the analysis, not only the
loading.

The executable starting point is
[`examples/06_bloem.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/06_bloem.py):

```bash
python examples/06_bloem.py 1-037
```

The script stops before the analysis, which §4 onward covers.

!!! warning "Scope of this page"
    The other tutorials run on the simulator, where the injected values are known, and
    quote their results. These systems have no published orbital solutions. Every number
    below was measured and recorded elsewhere in this repository: the archive facts, the
    loader's behaviour, and the closed-loop simulator results that justify each choice.
    Where a step produces a fitted result, the page states what to inspect rather than what
    value to expect.

## The short version

```python
import albireo as ab

star = ab.resolve_bloem("1-037")                       # name -> Gaia DR3 source id
records = ab.bloem_spectra(star, public_only=True)     # ~25 epochs, LR02
ab.download(records, "data/bloem-1-037")

ds = ab.read_dataset(
    "data/bloem-1-037/*.fits",
    instrument="GIRAFFE",
    region=(4120.0, 4300.0),     # strictly between H-delta and H-gamma; see §3
    region_pad_angstrom=40.0,
    smooth_angstrom=60.0,        # these products are not continuum-normalized
)
grid = ab.LogGrid.covering(ds, dv_kms=8.0, v_margin_kms=400.0, lsf_sigma_kms=20.2)
```

`resolve_bloem` and `read_dataset` need the optional extras: `pip install -e ".[io]"`.

## 1. The two steps that are not obvious

**The archive does not use the survey's names.** There is no BLOeM Phase 3 collection. The
spectra are filed under `obs_collection='GIRAFFE'`, and `target_name` is the Gaia DR3 source
id, not `1-037`. `resolve_bloem` fetches the published cross-match from VizieR, which uses
the same TAP dialect as ESO, so the join adds no dependency. Gaia ids are kept as strings
throughout, because 809 of the 929 are altered by a float64 round trip.

**Filter on the survey programme.** A second programme, `115.28A9`, re-observes the same
929 stars at R = 17000 and 23000 in two other windows, adding 1,827 spectra. Pooling those
with LR02 under one line-spread function would be wrong, so `bloem_spectra` defaults to
`112.25R7`. (If `112.25W2` appears in a search result, it is the arXiv v1 typo. In the
archive that identifier belongs to an ERIS programme.)

`public_only=True` is needed because sub-run `.004` releases through 2027-01-15, and the
proprietary rows would otherwise be returned and then fail to download.

## 2. The file layout, and why it need not be specified

GIRAFFE products put the flux in `FLUX_REDUCED`, the errors in `ERR_REDUCED`, the quality
flags in `QUAL_REDUCED`, and the wavelengths in nanometres on an air scale in the
heliocentric frame. The FEROS files of [the real-data tutorial](real-data.md) use
`FLUX`/`ERR`, ångström and the barycentric frame. `ab.read_dataset` reads both without the
layout being specified, because it dispatches on the IVOA utypes rather than on column names.

Keying on UCDs instead is unreliable. UVES gives its sky-background column the same UCD that
HARPS gives its flux column, so a UCD-keyed reader fits the sky without warning. Only the
utype role distinguishes them. The chosen columns can be inspected:

```python
raw = ab.read_spectrum("data/bloem-1-037/<one>.fits")
print(raw.columns, raw.wave_medium, raw.specsys, raw.err_source)
```

Two properties of this collection change the procedure:

**The products are not normalized and not flux-calibrated** (`CONTNORM=F`,
`FLUXCAL='UNCALIBRATED'`). `smooth_angstrom=` is therefore required. It runs
`albireo.preprocess`'s continuum fit. The survey team's own normalized, co-added
reduction is available only with credentials, so assume it is unavailable.

**The epochs do not share a wavelength grid, and must not be forced onto one.** FEROS shifts
before resampling, so its 51 HR 6819 epochs agree to 0.007 km/s and can be relabelled onto a
common grid. GIRAFFE's epochs differ by 5.3 km/s, most of a model pixel, so they are distinct
wavelength solutions, and `share_wavelength_grid` rejects them. albireo gives each its own
rebin operator. The cost is one operator per distinct grid, which is small compared with the
solve. Do not call `share_wavelength_grid` here.

## 3. Three things to decide

**The window.** `examples/06_bloem.py` uses 4120–4300 Å inside LR02's 3960–4571 Å (Si III
4128/4130, He I 4144, He I 4169, He II 4200). The blue edge is the critical value. The
window lies strictly between Hδ (4101.7) and Hγ (4340.5), so that neither Balmer line nor
its nebular window is inside. The margin is smaller than it appears, so check it:

```python
ab.nebular_windows(wave_range=(4120.0, 4300.0), v_kms=150.0)   # (): nothing to model
ab.nebular_windows(wave_range=(4000.0, 4300.0), v_kms=150.0)   # ((4099.7, 4107.9),)
```

A window reaching only 25 Å further blue includes Hδ's ±300 km/s nebular window. See §6
before widening it.

**The line-spread function.** LR02 delivers R ≈ 6200–6300. Take the value from the archive
rows (`em_res_power`, which `bloem_spectra` returns and the example prints) rather than from
a paper, and convert it:

```python
lsf = ab.LSF.from_resolution(6300)     # sigma = c / (R * 2 sqrt(2 ln 2)) = 20.2 km/s
```

`R` refers to the FWHM. Using `c / R` directly changes the kernel radius by a factor of 2.35
and is the most common way to mis-specify an instrument.

**The light fractions.** albireo does not estimate these. The likelihood depends only on the
products `ℓᵢ dᵢ`, so the continuum light ratio is exactly degenerate with the line depths
(`docs/math.md` §5.2) and only external information (photometry, an SED fit, an eclipse
depth) determines it. For a BLOeM SB2 with no light-curve solution, choose a value, state it,
and report how the result changes across a plausible range. Every summary albireo prints
repeats the assumption for this reason.

The grid follows:

```python
grid = ab.LogGrid.covering(ds, dv_kms=8.0, v_margin_kms=400.0, lsf_sigma_kms=lsf.sigma_kms)
```

`v_margin_kms` has to cover the SMC's ~+150 km/s systemic recession plus the orbit, since
the model grid must extend beyond the data by the largest component shift plus the kernel
radius.

## 4. Velocities before an orbit

With no published period there is no starting value for a Keplerian fit, and albireo does
not supply one. The `Disentangler` interface scans the conjunction phase at a single period,
and warns if given a period prior wide enough to constitute a search, because a phase located
for the wrong period is worse than no phase.

Run the free per-epoch RV table first (the mode shown in
[`examples/09_rv_table.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/09_rv_table.py))
and take the period from it, declaring the measured velocities instead of an orbit:

```python
dis = ab.Disentangler(
    ds,
    components=[ab.Star("A", light=0.6), ab.Star("B", light=0.4)],
    velocities=v_measured,          # (2, n_epochs) km/s; no orbit, no period
    lsf={"GIRAFFE": lsf},
    dv_kms=8.0,
)
table = dis.fit()
rv, err = table.velocities(), table.velocity_errors()
```

`v_measured` can be any available estimate: cross-correlation lags, a shift-and-add
pipeline's output, or the He I 4144/4169 splitting read from the two most separated epochs.
It is a starting point, not a constraint. The per-component zero point is unidentified, so
the estimate needs the correct epoch-to-epoch pattern, not the correct level. A systemic
+150 km/s on every entry changes neither the result nor the solver's bandwidth.

!!! danger "A cold start fails"
    `v_measured` may not be a placeholder. With every component at the same velocity at every
    epoch the two stars are indistinguishable, and the fit converges to a solution 122,000
    nats below the warm-started one (measured in the benchmark record). The declaration
    raises an error in that case, and warns if the supplied velocities never separate the
    components by more than the LSF width.

Then run a periodogram on `rv` and use the period in the `Orbit` declaration of §5.
`albireo.find_period` provides one for a `VelocityTable` (see the
[TODCOR tutorial](todcor.md)).

Two properties of the resulting table are counter-intuitive and are described in
[§7.6 of the math](../math.md#76-free-per-epoch-velocities-the-rv-table):

* It has one arbitrary zero point per component, not one in total, so absolute velocities
  and the systemic velocity are not recoverable from it. Each star's variation, the
  epoch-to-epoch differences, and the Wilson slope (the mass ratio) are recoverable, and
  these are what a period search and a mass ratio need.
* **Do not read the raw Laplace diagonal as an error bar.** Each zero point is an exactly
  flat direction, so its posterior width is the prior's and every epoch has that width. On
  the velocity-table fixture the raw bars were `120/√10 = 37.947` km/s on every entry against
  a real 0.059. `table.velocity_errors()` projects the zero points out. At the expert level
  the equivalent is `ab.relative_velocity_errors(cov, fit.unconstrained)`, or posterior
  samples of the `velocity_rel` deterministic.

## 5. The Keplerian, checked against the table

Once a period is known, use the `Disentangler` interface:

```python
dis = ab.Disentangler(
    ds,                                    # no grid: it and the budget are derived
    components=[ab.Star("A", light=0.6), ab.Star("B", light=0.4)],
    orbit=ab.Orbit(
        period=ab.Between(p0 * 0.98, p0 * 1.02),        # narrow: this is not a search
        k=ab.Between([20.0, 20.0], [250.0, 250.0]),     # one bound per star
    ),
    lsf={"GIRAFFE": lsf},
    dv_kms=8.0,                            # match §3; the default is the native sampling
)
print(dis.explain())                       # every derivation, including the phase scan
kep = dis.fit()
print(kep.summary())
```

`dis.grid` is the grid the interface chose, and it is not necessarily the one built manually
in §3. Use `dis.grid` from here on for consistency.

The velocity table provides the model check:

```python
free = kep.free_velocities()
resid = free.keplerian_residuals(kep)      # km/s, both zero points cancel exactly
```

Compare those residuals with the per-epoch uncertainties rather than with zero. Structure,
whether phase-correlated residuals or one outlying epoch, indicates a period that is
slightly wrong, an unmodelled third body, or line-profile variability that the Keplerian has
absorbed into `e`. On the velocity-table fixture a period wrong by 0.5% moved the residuals
from 2.9σ to 49σ.

## 6. The nebular component, before the Balmer lines

BLOeM's targets are in H II regions. Their spectra contain nebular emission that does not
move with either star and varies from night to night with seeing and slit losses, and this
is the main reason the survey's own disentangling is hard.

Unmodelled nebular emission biases the result. On a simulated SB2 whose Hβ absorption
contains a static nebular line
([`examples/04_nebular.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/04_nebular.py)),
ignoring it lowers the equivalent width by 11.5%, which propagates into log *g*. This is
the effect the literature describes. The effect on the orbit is far larger. A static line is
a component with *K* = 0, so a joint fit without the nebular component assigns the emission
to whichever star can be made to move least:

| | injected | without the component | with the component |
|---|---|---|---|
| K₂ [km/s] | 41.0 | **16.77 (−59.1%)** | 40.88 (−0.29%) |
| period [d] | 5.70000 | 5.87115 (**+0.171**) | 5.69986 |
| eccentricity | 0 | **0.950 (the solver's clip)** | 0.0022 |

The contamination therefore affects the masses, not only the atmospheres. Only K₁ remains
accurate, because 70% of the light constrains it.

If the window is widened to Hδ or Hγ, add the component and confine it:

```python
dis = ab.Disentangler(
    ds,
    components=[ab.Star("A", light=0.6), ab.Star("B", light=0.4), ab.Nebular(v_kms=v_neb)],
    orbit=ab.Orbit(period=ab.Between(p0 * 0.98, p0 * 1.02),
                   k=ab.Between([20.0, 20.0], [250.0, 250.0])),
    lsf={"GIRAFFE": lsf},
    dv_kms=8.0,
)
```

The prior confines the component to the nebular line windows. At the expert level that is
`ab.nebular_windows(wave_range=(grid.wave[0], grid.wave[-1]), v_kms=v_neb)` passed to
`ab.window_profile`, and the `Disentangler` interface assembles it. The confinement affects
the orbit. The same fit with the nebular component left free across the whole grid gives K₂
at +2.6% instead of −0.29%, because the extra freedom absorbs stellar signal at wavelengths
where a nebula has no lines. This is the kind of failure the component is meant to prevent.

The component has two conventions that the data do not determine: the amplitude scale (the
geometric mean is fixed to 1) and `v_kms`, which sets the placement of the line windows and
must match the value passed to `nebular_windows`. For the SMC that is the systemic
recession, not zero.

!!! note "The wavelength scale must be declared"
    A nebular line list is a set of absolute wavelengths, and the offset between air and
    vacuum is a nearly constant 83 km/s, so `Disentangler` raises an error rather than
    guessing when the dataset does not declare the scale. GIRAFFE products declare air in
    `TUCD1` and `read_dataset` preserves it, so the check passes on real BLOeM data. A
    `Dataset` assembled manually needs the declaration, as in `EpochData(..., medium="air")`.

## 7. The product

The disentangled spectra are passed to an atmosphere code, and the uncertainty must be passed
with them:

```python
kep.write_spectra("bloem-1-037_spectra.fits")   # mean + band + the assumptions, as FITS
```

Interpret the mean only together with its band. Between the lines, and wherever the epochs
provide little constraint, the recovered spectrum is set by the smoothness prior rather than
by the data, and the band shows where.
`ab.plot_spectra(dis.grid, kep.spectra(), std=kep.std())` draws it.

Make three checks before the result is used:

* `fit.z_rms`, the whitened residual scatter. A value other than 1 means the inverse
  variances are not calibrated, and the error propagates into every downstream uncertainty.
  `ab.plot_residual_zscores` shows the shape, including the lag-1 panel that reveals
  correlated noise a histogram cannot.
* The light-fraction sensitivity. Re-run at the ends of the plausible range and quote the
  spread, which is a non-negligible systematic.
* Whether the available epochs constrain the quantities of interest.
  `ab.sensitivity_forecast(dis.grid, ds, orbit=kep.theta, ...)` needs no fluxes, so it
  applies to requested epochs as well as to those already obtained
  ([the forecast example](https://github.com/tjayasinghe/albireo/blob/main/examples/08_forecast.py)).
  BLOeM's published multiplicity results use only the first nine epochs. The full ~25 are in
  the archive, and a forecast shows what the additional epochs add before a fit is run on
  them.

## What this page does not claim

No orbit here has been validated against a published solution, because none exists. For the
same reason every number above comes from the simulator or from the archive rather than from
a BLOeM fit. A system analysed successfully with this procedure is a new result.

Several limits are known. R ≈ 6300 is low for disentangling, and 4000–4300 Å is a narrow
window. The SB2 classifications are split across five unharmonized VizieR catalogues, with
no single table of all 929 stars. The light ratio, the one free choice in disentangling, is
unconstrained for these targets until photometry is published. All of them belong in the
paper.
