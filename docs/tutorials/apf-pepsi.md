# APF and PEPSI spectra

albireo reads two instrument formats directly: the products of the PEPSI Spectroscopic Data
Systems pipeline (Large Binocular Telescope) and the exposures of the Automated Planet Finder
(Lick Observatory) extracted by the California Planet Search pipeline. Both are recognised by
their headers in [`read_spectrum`][albireo.io.read_spectrum], `read_dataset` and the
pipeline, so a directory of either goes through the same analysis as any other set of
spectra: disentangled components, epoch velocities and an orbit. This page describes what
each reader takes from the files, the one calibration step that APF data need, and a
pipeline configuration for each instrument. Reading requires astropy
(`pip install -e ".[io]"`). Both readers are experimental.

## PEPSI products

A PEPSI product is a binary table with the columns `Arg` (air wavelength, Å), `Fun`
(normalised flux), `Var` (variance) and `Mask`, behind a primary header that holds what the
analysis needs:

| Quantity | Header | Treatment |
|---|---|---|
| Time | `JD-TDB` | BJD_TDB at mid-exposure; agrees with astropy to 0.01 s |
| Barycentric correction | `SSBVEL` (m/s) | `-SSBVEL / 1000` km/s; agrees with astropy to 3 m/s |
| Frame | | barycentric: the pipeline has removed `SSBVEL` from the wavelengths |
| Medium | | air: a vacuum scale would offset the velocities by 83 km/s from the header `RADVEL` |
| Resolving power | `FIBER` | 44,300 for the 300 µm fibre (FWHM 6.77 km/s); other fibres must be declared |
| Arm | file name, `CROSDIS` | `pepsib.*` blue, `pepsir.*` red, and the cross-disperser number |

The resolving power was measured from the widths of 14 weak, isolated telluric O2 lines in
red-arm spectra of 2023-2026, with the lines' own width (1.3 km/s FWHM, an assumption)
removed in quadrature. The products are continuum-normalised by the pipeline and are used
as they are.

```python
import albireo as ab

raw = ab.read_spectrum("data/pepsi/star/pepsib.20240101.001.dxt.bwl")
print(raw.summary())
```

A directory of PEPSI products holds the blue and the red arm of every exposure, which cover
different wavelengths. Passed to `read_dataset` together with a `region`, the files that do
not cover it are left out by `min_coverage`, with a warning that names them:

```python
ds = ab.read_dataset(
    "data/pepsi/star",              # a directory: its pepsib.* and pepsir.* files are read
    region=(4790.0, 5410.0),        # inside cross-disperser 3 of the blue arm
    mask=[(4835.0, 4888.0)],        # H-beta
    min_coverage=0.5,               # the red-arm files do not reach the region
)
```

## APF exposures

The CPS pipeline delivers each exposure as a 79 x 4608 image of counts,
`r<night>.<number>.fits`, without wavelengths. The wavelength solution is a separate file of
the same shape, `apf_wave_2022.fits`, in air Ångström. Row r of both is echelle order
124 - r. The reader needs the solution and does the rest per exposure:

1. **Zero point.** The telluric lines of rows 49-61 (the O2 gamma and B bands and two H2O
   regions near 7200 Å) are correlated with the packaged line list
   ([`albireo.telluric`](../api/telluric.md)), and the wavelengths are corrected by the
   shift found. The internal error is 0.05-0.15 km/s per exposure.
2. **Line-spread function.** The widths of 14 weak, isolated O2 lines, with their own width
   removed in quadrature. Where fewer than three lines can be fitted, the typical value of
   the decker is used (N 3.97, T 5.65, W 3.82 km/s FWHM).
3. **Noise and cosmic rays.** `var = (counts + 359.5) / 0.884` ADU², and positive
   outliers above a running median.
4. **Blaze and merging.** Each order is divided by its upper envelope, and the orders are
   joined at the midpoints of their overlaps without resampling.
5. **Sky and iodine.** Night-sky and street-light emission lines are masked in the
   topocentric frame. Exposures taken through the iodine cell are refused unless
   `iodine="mask"`, which masks 5000-6200 Å.
6. **Time.** BJD_TDB at the photon-weighted midpoint (`THEMIDPT`) and the barycentric
   correction, both with astropy for the APF site and the J2000 header coordinates.

```python
raw = ab.read_spectrum(
    "data/apf/star/rabc.123.fits",
    apf={"wavelengths": "apf_wave_refined.fits"},
)
print(raw.calibration["drift_kms"], raw.calibration["lsf_fwhm_kms"])

ds = ab.read_dataset(
    "data/apf/star/*.fits",
    read_kwargs={"apf": {"wavelengths": "apf_wave_refined.fits"}},
    region=(4450.0, 5650.0),
    mask=[(4835.0, 4888.0)],
)
```

The other keys of `apf=` are the options of [`albireo.apf.extract`][albireo.apf.extract]:
`zero_point`, `lsf`, `blaze`, `rows`, `iodine` and `mask_sky`. `raw.calibration` records
what was measured or assumed for each exposure, and `raw.calibration["notes"]` lists the
steps taken.

The CPS pipeline delivers some exposures twice, as `r<night>.<n>.fits` and as a
sky-subtracted `r<night>.<n>_skysub.fits`. When both are listed, only the first is read: on
the 8 arcsec N decker the subtracted sky contains 7-58 % of the counts, including the star's
own light. A list of only `_skysub` files reads those.

## Refinement of the APF wavelength solution

The 2022 solution does not match exposures of later years. On exposures from 2023-2026 its
zero point is offset by about 1 km/s, which the telluric zero point removes, and its
dispersion within each order differs from these exposures' by up to 1.7 km/s near the blue
end of the order, which no zero point removes. The solution is therefore refitted once to
one exposure of a star with many metal lines (a G or K star, high signal-to-noise, no iodine
cell):

```bash
albireo apf-wavelengths data/apf/kstar/rabc.123.fits --base apf_wave_2022.fits --out apf_wave_refined.fits
```

The command correlates 256-pixel chunks of every order with a synthetic spectrum (BOSZ,
4750 K, log g 4.5, solar metallicity by default; `--teff`, `--logg` and `--mh` match it to
the star) and the chunks of the telluric rows with the telluric line list. It fits
`m * lambda` as a two-dimensional Chebyshev series in the pixel and the order, the form of
the grating equation, to both sets at once with the star's velocity as a free parameter, so
that the telluric lines fix the absolute scale. The library is downloaded once (about
621 MB). The printed summary states the number of chunks used, their scatter about the
fit, the reference star's velocity and the range of the correction. On the exposure used
for the validation below, 1393 stellar chunks scatter by 0.87 km/s rms and 34 telluric
chunks by 0.17 km/s, and the correction reaches -4.5 and +9.0 km/s at the ends of some
orders.

On ten exposures of the reference star taken over 557 days and not used in the fit, the
velocities of 1024-pixel chunks over rows 8-56 scatter by 0.56 km/s (median absolute
deviation) about each exposure's median with the refined solution, against 0.86 km/s with
the 2022 solution, and the blue quarter of the orders agrees with the rest to 0.1 km/s,
against 1.2 km/s. The ten exposures span 557 days, so one refined solution serves
exposures taken well apart in time; the telluric zero point of each exposure absorbs the
drift between them.

## Pipeline configuration

Neither instrument needs an `[instrument]` entry: PEPSI's resolving power follows from the
fibre, and the width of every APF exposure is measured from its own telluric lines and
declared per epoch. An APF configuration gives the solution in `[read.apf]`, with a path
relative to the configuration file:

```toml
[output]
directory = "results"

[read.apf]
wavelengths = "apf_wave_refined.fits"
# iodine = "mask"                 # exposures through the iodine cell: mask 5000-6200 A

[analysis]
region = [4450.0, 5650.0]         # topocentric for APF, barycentric for PEPSI
mask = [[4835.0, 4888.0]]         # H-beta
dv_kms = 2.0

[labels]
library = "bosz2024-fgk-r20000"
mh = [-1.0, 0.5]

[[stars]]
name = "binary-1"
spectra = "data/apf/binary-1/*.fits"
period = "search"

[[stars.components]]
name = "A"
light = "measure"
teff = [5000.0, 7000.0]

[[stars.components]]
name = "B"
light = "measure"
teff = [5000.0, 7000.0]
```

`period = "search"` finds the period from velocities measured against library templates
and needs enough epochs to do so: below eight nights the pipeline flags the period it
chooses as a candidate rather than a measurement. With fewer epochs, declare a range
(`period = [6.2, 6.34]`) and the light fractions.

A PEPSI star is declared in the same way, with `spectra` pointing at the directory. The
pipeline's log names the files that do not reach the region, which it leaves out, and
states the line-spread width it takes for each epoch (2.874 km/s for the 300 µm fibre).

## Validation

The readers were run on two double-lined binaries observed in 2023-2026, whose spectra are
not distributed with albireo, with the configurations of the previous section. The
reference is a dedicated analysis of the same spectra, in which the calibration steps of
the readers were developed; the comparison therefore shows that the readers and the
pipeline reproduce that analysis, not the accuracy of either.

**APF.** Twenty exposures with the T decker, 4450-5650 Å with H-beta masked, the period
searched. The line-spread widths measured from the telluric lines (5.4-6.4 km/s FWHM)
agree with those of the dedicated analysis to 0.10 km/s rms, exposure by exposure, and are
0.16 km/s broader on average because one of the fourteen O2 lines differs between the two
line sets; the wavelength solution makes no difference to them. The bootstrap proposed 35.531 d, and
the disentangling kept it against three other candidate periods, by 51,844 nats over the
next. The velocities remain differential (see the limitations below).

| | Pipeline | Dedicated analysis |
|---|---|---|
| P (d) | 35.5394 ± 0.0017 | 35.5402 ± 0.0007 |
| e | 0.0037 | 0.0038 |
| K_A (km/s) | 38.11 ± 0.05 | 38.11 ± 0.02 |
| K_B (km/s) | 38.02 ± 0.06 | 38.02 ± 0.03 |

The residuals about the orbit are 0.21 and 0.27 km/s rms. The run took 31 minutes on a
32-thread desktop that was running other jobs at the same time.

**PEPSI.** Seven exposures, 4790-5410 Å in the blue arm with H-beta masked; the seven
red-arm files were left out by the coverage rule. With the period searched, the velocity
table preferred 6.266 d (χ² 245, against 8,119 at the period finally chosen), but the
disentangling's marginal likelihood preferred an eccentric alias at 7.800 d by 11,487 nats,
and the orbit fitted there has a reduced χ² of 1,611. The pipeline flags a period searched
on fewer than eight nights as a candidate rather than a measurement; here the candidate is
wrong. With the period declared as a range around the table's (6.20-6.34 d) and the light
fractions of the dedicated analysis:

| | Pipeline | Dedicated analysis |
|---|---|---|
| P (d) | 6.26826 ± 0.00009 | 6.26820 ± 0.00008 |
| e | 0.003 | 0.015 ± 0.002 |
| K_A (km/s) | 62.83 ± 0.04 | 63.09 ± 0.04 |
| K_B (km/s) | 68.52 ± 0.05 | 68.84 ± 0.05 |

The semi-amplitudes are lower by 0.26 and 0.32 km/s (0.4-0.5 %), several times the formal
errors, which do not include the treatment of the continuum. The dedicated analysis
renormalised the PEPSI products on 12 Å scales and removed a residual continuum difference
between the epochs; the pipeline uses the products' own normalisation. The run took
24 minutes.

## Limitations

- **Systemic velocities and labels.** The label stage compares library spectra with the
  epochs at the instrument's resolution, and a library spectrum cannot be sharpened. None of
  the libraries that albireo fetches has a resolving power above APF's or PEPSI's (the BOSZ
  grids are at R = 20,000), so the stage stops with a flag that says so, and the epoch
  velocities remain differential: each component keeps its own unidentified zero point. The
  period, the eccentricity and the semi-amplitudes are measured; the systemic velocity and
  the labels are not. Velocities with an absolute zero point can be measured against library
  templates with [`albireo.fit_templates`][albireo.templatefit.fit_templates], whose
  templates may be broader than the data.
- **Scattered light.** PEPSI spectra are not sky-subtracted, and the APF T decker
  (2 x 3 arcsec) leaves no room for a sky measurement. Scattered moonlight or twilight adds
  a solar spectrum at the topocentric rest velocity, which the model does not contain; such
  exposures are best left out.
- **Low signal-to-noise APF exposures.** The zero point needs telluric lines that correlate
  above 0.3 in at least one region. An exposure without them raises an error that names it,
  and the file must be left out, or read on its own with a declared shift
  (`apf={"zero_point": <km/s>, ...}`).
- **Blaze.** The upper envelope of an order follows any absorption broader than about
  500 pixels (25 Å): the wings of the Balmer lines are partly removed. Mask them, or pass a
  blaze made from exposures of a line-poor star with `albireo.apf.build_blaze`.
- **Other PEPSI fibres** have no default resolving power, and must be declared with
  `resolving_power=` or an `[instrument.PEPSI]` entry.
- **The wavelength solution itself** is the user's: the package contains the telluric line
  list, not the CPS solution file.
