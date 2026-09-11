# Gaia RVS instrument constants for a double-lined-binary simulator

Compiled 2026-09-09. Confidence labels: **verified** = read in the primary source (refereed paper,
ESA/DPAC documentation, or a live Gaia archive query); **likely** = consistent across independent
secondary sources but the primary text was not read directly; **unverified** = could not be sourced.

Live ADQL queries against `https://gea.esac.esa.int/tap-server/tap/sync` were run on 2026-09-09 and
are labelled *TAP*.

## Bottom line

**Every number in the colleague's notebook is correct.** The G_RVS polynomial is Sartoretti et al.
2023 Eqs. (9)-(10) verbatim, and the S/N formula is the ESA science-performance model verbatim —
which I then validated numerically against the published `rv_expected_sig_to_noise` column, where it
agrees to **3-8% across eight magnitudes**.

What needs changing is not the constants but four claims around them:

1. **The rest-frame description is half wrong.** DR3 divides out a per-transit velocity only for
   bright stars (`rv_method_used = 1`); for the 26.8 M faint stars a single combined velocity is
   used, so a faint SB2's mean spectrum is orbitally smeared. And **DR4 epoch spectra are in the
   solar barycentric frame, not the rest frame** — code written against DR3 will silently mis-shift.
2. **Do not generate Paschen wavelengths from the Rydberg formula** — it is +0.05 A (**+1.8 km/s**)
   too red on every line in the band. Use NIST Ritz values.
3. **There is no 5-day loop** in the scanning law, and **RVS gets only 4/7 of astrometric transits**.
   Phase coverage tracks visibility periods (median 13), not transits (median 18).
4. **The DR4 grid claim (961 samples / 0.025 nm / the field names / `normalisation_method` codes) is
   the one thing I could not verify.** It is plausible and internally consistent, but the only source
   is a 2.8 MB zip that needs downloading — see item (5).

---

## (1) G_RVS from G and G_RP — CONFIRMED EXACTLY

**Source: Sartoretti et al. 2023, "Gaia Data Release 3: G_RVS photometry from the RVS spectra",
A&A 674, A6.** Bibcode `2023A&A...674A...6S`, DOI [10.1051/0004-6361/202243615](https://doi.org/10.1051/0004-6361/202243615),
arXiv [2206.05725](https://arxiv.org/abs/2206.05725). The relations are **Eq. (9)** and **Eq. (10)**.

Let `x = G - G_RP`.

| Branch | Equation | Relation | RMS |
|---|---|---|---|
| `-0.15 <= x <= 1.2` | Eq. (9) | `G_RVS - G_RP = -0.0397 - 0.2852 x - 0.0330 x^2 - 0.0867 x^3` | **0.04 mag** |
| `1.2 < x <= 1.7` | Eq. (10) | `G_RVS - G_RP = -4.0618 + 10.0187 x - 9.0532 x^2 + 2.6089 x^3` | **0.09 mag** |

Every coefficient, every sign, the breakpoint at `x = 1.2` and the validity range `-0.15 <= x <= 1.7`
match the notebook exactly. **Confidence: verified.**

Notes worth carrying into the code:

- The paper's symbol is `G_RVS^(G,RP)`, i.e. a *colour-derived estimate*, deliberately distinct from
  the spectroscopic `grvs_mag`. Prefer the catalogue value when present.
- Calibrated on ~3 million sources with well-behaved `G`, `G_BP`, `G_RP` fluxes and `grvs_mag` error
  < 0.05 mag (verified).
- The paper gives **no** relation in `(G_BP - G_RP)`; `G - G_RP` is used exclusively (verified).
- The ESA science-performance page does **not** carry this relation, so Sartoretti et al. 2023 is the
  only primary home for it (verified).
- The two branches are **not continuous in slope** at `x = 1.2`; check your implementation's branch
  boundary is `<=` on the first branch to reproduce the paper.

**Discrepancy: none.**

---

## (2) The S/N formula — CONFIRMED EXACTLY, but know which of the two variants you have

The notebook's formula is, verbatim, ESA's **predictive** S/N model.

**Primary source: ESA Gaia "Science performance" page, section 3 "Spectroscopic performance (updated
in 2022, based on Gaia DR3)",** <https://www.cosmos.esa.int/web/gaia/science-performance>.
Quoted verbatim from that page:

```
S   = 10^((GRVS_ZEROPOINT - grvs_mag)/2.5) * EXPOSURE_TIME * N_SPECTRUM_PER_TRANSIT
      * rv_nb_transits * (PIXEL_WIDTH_AL / BAND_WIDTH)
BCK = MEDIAN_BACKGROUND * N_SPECTRUM_PER_TRANSIT * rv_nb_transits * N_AC_PIXELS
RN  = RON^2 * N_SPECTRUM_PER_TRANSIT * rv_nb_transits * N_AC_SAMPLES
rv_expected_sig_to_noise = S / SQRT(S + BCK + RN)
```

| Constant | ESA value | Notebook | Independent corroboration | Confidence |
|---|---|---|---|---|
| `GRVS_ZEROPOINT` | **21.317 mag** | 21.317 | Sartoretti+2023 §4.2: `G_RVS^spec = -2.5 log(TotFlux) + ZP`, **ZP = 21.317 +/- 0.002 mag**, `TotFlux` in **e-/s** integrated over the whole 846-870 nm band. (Vega spectrum route gives `m_0 = 21.321 +/- 0.016`.) | verified |
| `EXPOSURE_TIME` | **4.4167032 s** | 4.4167032 | Cropper et al. 2018 Table 5: **4.4167 s** per CCD | verified |
| `N_SPECTRUM_PER_TRANSIT` | **3** | 3 | RVS = strips 15-17 x rows 4-7; a source crosses one row, hence 3 CCDs. Cropper Table 5: "41 transits, each over 3 CCDs" | verified |
| `PIXEL_WIDTH_AL` | **0.02453 nm** | 0.02453 | Cropper Table 5 dispersion: 0.0244 nm/pix @847 nm, 0.0246 nm/pix @873 nm (8.51 and 8.58 km/s/pix). 0.02453 is a band-mean | verified |
| `BAND_WIDTH` | **24.0 nm** (870.0 - 846.0) | 24.0 | matches the published spectrum grid | verified |
| `MEDIAN_BACKGROUND` | **4.7 e-/pixel** (per exposure) | 4.7 | see caveat below | verified as ESA's adopted value |
| `N_AC_PIXELS` | **10** | 10 | Sartoretti+2023: "One sample corresponds to 1 AL x 10 AC pixels"; Cropper §7.1: nominal window 10 px AC | verified |
| `RON` | **3.2 e-** | 3.2 | Cropper Table 7: 2.9-3.4 e- across the 12 HR detectors, **mean 3.1 e-** | verified (see discrepancy) |
| `N_AC_SAMPLES` | **10 for grvs_mag <= 7, else 1** | same | Cropper §7.1 window classes: Class 0 = 2D windows, `G_RVS <= 7`; Class 1 = `7 < G_RVS <= 10`; Class 2 = fainter, both 1D (AC-summed on chip) | verified |
| Stated validity | `G_RVS <= 16` (12 for hot stars) | — | ESA page | verified |

The window-class rule is physically right: a 2D window reads 10 AC rows *separately*, so 10
independent read-noise draws; a 1D window sums the 10 AC pixels **on-chip before readout**, so one.
Background is 10 pixels' worth either way, which is why `BCK` always carries `N_AC_PIXELS = 10`
while `RN` carries `N_AC_SAMPLES`. The notebook has this exactly right.

### Numerical validation against the published catalogue

I evaluated the formula at the archive's own values. Selecting DR3 sources within +/-0.1 mag of each
integer `grvs_mag` (*TAP*, `random_index < 5e7`) and feeding the bin's mean `grvs_mag` and mean
`rv_nb_transits` into the formula:

| `grvs_mag` | mean `rv_nb_transits` | archive `AVG(rv_expected_sig_to_noise)` | formula | ratio |
|---|---|---|---|---|
| 6.008 | 24.87 | 640.0 | 661.0 | 1.033 |
| 8.003 | 16.32 | 196.3 | 210.1 | 1.070 |
| 10.002 | 19.60 | 74.21 | 80.44 | 1.084 |
| 12.002 | 19.95 | 19.08 | 20.61 | 1.080 |
| 13.971 | 18.13 | 3.849 | 3.70 | 0.960 |

**The formula reproduces the published column to 3-8% over eight magnitudes.** The residual few
percent is expected: `SNR ∝ sqrt(n)` is concave, so `AVG(SNR) < SNR(AVG(n))` (Jensen), and the
comparison averages over a finite magnitude bin. **Confidence: verified.**

### The one thing to know: there are TWO formulas, and they are not the same object

The DR3 archive data model for `gaia_source.rv_expected_sig_to_noise`
(<https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html>)
gives the **measured** version actually used to populate the catalogue:

```
S   = medianFlux * N_ValidStrips * EXP_TIME * PIXEL_WIDTH_AL / BAND_WIDTH
BCK = medianBackground * N_ValidStrips * N_ACpixels
RN  = RON^2 * N_ValidStrips * N_ACsamples
```

with `medianFlux` the *measured* median integrated flux of the CCD spectra [e-/s],
`N_ValidStrips` the *actual* number of CCD spectra used (not `3 x rv_nb_transits`), and
`medianBackground` the *measured* background. The science-performance page's version is the
**forecasting** substitution: `medianFlux -> 10^((21.317-grvs)/2.5)`, `N_ValidStrips -> 3*n`,
`medianBackground -> 4.7`. For a simulator the predictive form is the right one; just do not describe
it as "the formula the catalogue used". Also note `rv_expected_sig_to_noise` is a **combined** S/N
over all transits, never per-epoch. (verified)

### Discrepancies found in item (2)

1. **Read-out noise 3.2 vs 3.1 e-.** Cropper et al. 2018 Table 7 gives a commissioning mean of
   **3.1 e-** (range 2.9-3.4). ESA's model adopts **3.2**. Immaterial: at `G_RVS = 12` swapping
   3.2 -> 3.1 changes S/N by <0.3% because the background dominates. Keep 3.2 for consistency with
   the published column.
2. **Background 4.7 e-/pixel is not the only number in circulation.** Sartoretti et al. 2023 quotes
   the *nominal* stray-light level as **70 e-/sample** = 7.0 e-/pixel; Sartoretti et al. 2018 §5.2
   quotes a mean scatter-map level of **6.54 e-/pix** (EPSL data, explicitly "not divided by the
   4.4 s exposure time"); Cropper et al. 2018 §9.1 reports **20-30 e-/pixel per exposure** at the
   worst spin phases. ESA's 4.7 e-/pixel is a **mission-median** that post-dates the stray-light
   decrease from December 2015 onward. Since it reproduces the catalogue to <10%, keep it — but a
   simulator that wants per-epoch realism should treat the background as varying by a factor of a
   few with spin phase, not as a constant.
3. **A self-consistency caution.** `S` is the *mean* flux per sample (total band flux / 978 samples),
   not the continuum level. Sartoretti+2023's "S/N ~0.7 per sample on one CCD at `grvs_mag ~14`" is
   higher than this formula's 0.42 for one CCD, because a real spectrum's continuum sits above the
   band mean. Expect the formula to under-predict continuum S/N by a factor ~1.5.

---

## (3) Observed S/N versus G_RVS — sanity check of the formula

There is no published table of S/N vs `G_RVS` in Katz et al. 2023, so I measured it. Live *TAP*
query over `gaiadr3.gaia_source`, bins `|grvs_mag - round(grvs_mag)| < 0.1`, `random_index < 5e7`:

| `grvs_mag` | N sources | mean `rv_nb_transits` | `rv_expected_sig_to_noise` (co-added) | `rvs_spec_sig_to_noise` (measured, mean spectrum) | per single transit = SNR/sqrt(n) |
|---|---|---|---|---|---|
| 5.00 | 30 | 20.1 | 912 | 455 | 203 |
| **6.01** | 102 | 24.9 | **640** | **553** | **128** |
| 7.01 | 336 | 20.8 | 363 | 415 | 80 |
| **8.00** | 829 | 16.3 | **196** | **229** | **49** |
| 9.01 | 2254 | 17.3 | 121 | 140 | 29 |
| **10.00** | 5738 | 19.6 | **74.2** | **83.6** | **16.8** |
| 11.00 | 14356 | 20.5 | 40.4 | 42.3 | 8.9 |
| **12.00** | 31277 | 20.0 | **19.1** | **21.8** | **4.3** |
| 13.00 | 71707 | 19.1 | 8.28 | 15.6 | 1.9 |
| **13.97** | 35475 | 18.1 | **3.85** | (none published) | **0.90** |

Formula predictions for clean reference points (`n` = number of transits):

| `G_RVS` | 1 transit (3 CCDs) | 1 CCD | n = 18 (DR3 median) | n = 35 (DR4 median, est.) |
|---|---|---|---|---|
| 6 | 133 | 76.8 | 564 | 787 |
| 8 | 52.1 | 30.1 | 221 | 308 |
| 10 | 18.2 | 10.5 | 77.2 | 108 |
| 12 | 4.62 | 2.67 | 19.6 | 27.3 |
| 14 | 0.845 | 0.488 | 3.59 | 5.00 |
| 16 | 0.138 | 0.079 | 0.58 | 0.81 |

Important reading notes:

- Because `S`, `BCK` and `RN` are all strictly proportional to `n`, the formula gives
  **`SNR ∝ sqrt(n)` exactly**. Per-transit S/N is therefore exactly `rv_expected_sig_to_noise /
  sqrt(rv_nb_transits)`, with no approximation.
- `rvs_spec_sig_to_noise` is a **different quantity**: the DR3 documentation defines it as the
  *median over bins of `flux / flux_error`* in the published mean spectrum. It runs ~10-15% **above**
  `rv_expected_sig_to_noise` for `G_RVS >= 8`, and far above it at 13. Two reasons: the published
  grid is 0.01 nm while the native sample is 0.02453 nm, so adjacent bins are ~2.45x oversampled and
  correlated (the empirical scatter is optimistic); and mean spectra are only published above a
  `rvs_spec_sig_to_noise >= 15` cut, which truncates the faint bins from below. **Do not use the
  13.00 row.**
- Mean spectra are published only to `grvs_mag ~ 12-13`; there is no observed spectrum S/N at 14.
- Katz et al. 2023 §5.2 gives the whole-sample median `rv_expected_sig_to_noise` = **7.8**
  (verified), consistent with the sample being dominated by `grvs_mag ~ 13`.
- Katz et al. 2023 abstract: median formal RV precision **1.3 km/s at G_RVS = 12** and
  **6.4 km/s at G_RVS = 14** (verified). Measured `AVG(radial_velocity_error)` by bin (*TAP*):
  0.32 km/s at 6, 0.44 at 8, 1.04 at 10, 2.54 at 12, 5.47 at 13, 7.25 at 14 — a useful independent
  anchor if you want to map simulated S/N to an RV error.

**Verdict on the formula: it is the right formula and it is well calibrated.** No correction needed.

---

## (4) The line-spread function

| Quantity | Value | Source | Confidence |
|---|---|---|---|
| Nominal resolving power | **R = 11 500**, defined as **λ/FWHM** (λ/Δλ) | Sartoretti et al. 2018 §5.4; Recio-Blanco et al. 2023: "its medium resolving power is R=λ/Δλ ∼ 11 500" | verified |
| Resolution element | **3 pixels** | Sartoretti et al. 2018 §5.4: "The resolution element is 3 pixels, corresponding to R=11 500" | verified |
| Measured R, per CCD | **10 983 to 12 587**; mean **11 891**, median 11 988 | Cropper et al. 2018 Table 6 (spin periods 680-697, April 2014), measured "from a cross-correlation of Fe lines with a binary mask" | verified |
| Telescope split | **Telescope 1 mean 12 141**, **Telescope 2 mean 11 642** | Cropper Table 6 | verified |

Cropper et al. 2018, Table 6, in full (rows 4-7, strips 15-17):

| Telescope | Row | Strip 15 | Strip 16 | Strip 17 |
|---|---|---|---|---|
| 1 | 4 | 12 587 | 12 361 | 12 240 |
| 1 | 5 | 12 159 | 12 430 | 12 085 |
| 1 | 6 | 12 021 | 12 132 | 12 148 |
| 1 | 7 | 12 117 | 11 885 | 11 525 |
| 2 | 4 | 12 065 | 12 106 | 11 954 |
| 2 | 5 | 11 600 | 11 809 | 11 861 |
| 2 | 6 | 11 523 | 11 447 | 11 901 |
| 2 | 7 | 11 078 | 10 983 | 11 377 |

Implied FWHM at the band centre (858 nm), computed here:

| R | FWHM (nm) | FWHM (AL pixels @0.02453 nm) | FWHM (km/s) |
|---|---|---|---|
| 11 500 (nominal) | 0.0746 | 3.04 | 26.1 |
| 11 891 (measured mean) | 0.0722 | 2.94 | 25.2 |
| 10 983 (worst CCD) | 0.0781 | 3.18 | 27.3 |
| 12 587 (best CCD) | 0.0682 | 2.78 | 23.8 |

**Is the LSF Gaussian?** No — but a Gaussian is the standard, defensible approximation.

- The DPAC model is **not** analytic. DR3 documentation §6.3.4 gives
  `L(u) = H_0(u) + Σ_{n=1..N} h_n H_n(u)`, with `H_0` a fixed mean profile, `H_n` basis functions,
  and **N = 8** basis functions for the LSF-AL, calibrated over a **single waveband**; the LSF-AC uses
  **5** basis functions over **twelve 2-nm wavebands**. In DR2 (Sartoretti et al. 2018 §5.4, Eq. 5)
  the same expansion was built by **PCA**, `LSF = Σ_{n=0..7} h_n H_n`. (verified)
- Neither Cropper et al. 2018 nor Sartoretti et al. 2018 states that the AL profile is Gaussian, and
  neither quantifies non-Gaussian wings. **Confidence that "the LSF is non-Gaussian by construction":
  verified. Confidence on the size of the departure from a Gaussian: unverified** — DPAC has never
  published the AL basis functions or a wing amplitude.
- Practical recommendation: a Gaussian of FWHM = 3.0 AL pixels (R = 11 500) at the band centre is
  what essentially the whole RVS literature convolves with, including GSP-Spec. Carry the
  **10 983-12 587 spread as a systematic**, i.e. FWHM 2.78-3.18 px, roughly +/-7%.

**Wavelength dependence across 846-870 nm: negligible, and DPAC neglects it.** Sartoretti et al.
2018 §5.4 calibrated the on-ground LSF in "15 wavebands, each of 2 nm, covering the range
846-874 nm" and found a "weak dependence of the RVS LSF-AL on wavelength, **which is neglected in the
in-flight model**" (verified). The DR3 LSF-AL likewise uses a single waveband (verified). The only AL
wavelength dependence you must keep is the **dispersion**: 0.0244 nm/pix at 847 nm rising to
0.0246 nm/pix at 873 nm, i.e. **R varies by ~0.8% across the band** (Cropper Table 5, verified).

**Other dependences a simulator may want:** the LSF varies with CCD and field of view (Table 6
above, a ~13% spread in R); with **time** — DR3 splits the 34 months into **ten calibration periods**
whose boundaries are instrument-calibration discontinuities, with `h_n` aggregated over one-hour
Calibration Units (verified, DR3 doc §6.3.4); DR2 needed two separate LSF models split at OBMT 1317,
with ~20% degradation before it (verified). The **across-scan** LSF averages 3.5 px FWHM (Telescope 1)
and 2.8 px (Telescope 2), broadening to 2.25-4.2 px under nominal scanning (Cropper §11.1, verified) —
relevant only to deblending, not to the AL velocity scale.

**Discrepancy: none with the notebook's ~11 500.** But R = 11 500 is the *design/nominal* value; the
*measured* mean is **11 891**, ~3.4% higher, and the per-CCD spread is +/-7%. If the simulator claims
per-transit realism it should draw R per CCD from Table 6 rather than fixing 11 500.

---

## (5) Wavelength grids and conventions

### DR3 `rvs_mean_spectrum`

| Property | Value | Source | Confidence |
|---|---|---|---|
| Grid | **846 to 870 nm, step 0.01 nm, 2401 samples** | DR3 data model §20.12.1: "their wavelength grid ranges from 846 to 870 nm in steps of 0.01 nm (2401 elements)" | verified |
| Endpoints | **inclusive both ends**: 846.00 and 870.00 (`(870-846)/0.01 + 1 = 2401`) | arithmetic | verified |
| Vacuum or air | **VACUUM** | ESA Gaia Image of the Week 2021-07-09: "the wavelength interval between 846 and 870 nm **(in vacuum)**". Corroborated by Recio-Blanco et al. 2023 (GSP-Spec): synthetic spectra "were computed in the air and then converted into vacuum wavelengths thanks to the relation of Birch & Downs 1994" | verified |
| Frame | **rest frame** | DR3 data model §20.12.1 | verified |
| Flux | **normalised, dimensionless**; `flux[2401]`, `flux_error[2401]` | DR3 data model | verified |
| Rows published | **999 645** | *TAP* `COUNT(*) WHERE has_rvs='true'`; matches gaia.aip.de metadata | verified |
| Oversampling | the 0.01 nm grid is **2.45x finer than the 0.02453 nm native sample** — adjacent bins are correlated | arithmetic | verified |

**Correction to the notebook's description of the rest-frame shift.** It is *not* uniformly
"velocity divided out per transit". DR3 documentation §6.4.9 (MTA):

- `rv_method_used = 1` (bright, generally `grvs_mag <= 12`): shifted using the **Gaia-centric
  single-transit radial velocities**, i.e. per transit. 7 009 365 sources (*TAP*).
- `rv_method_used = 2` (faint, `grvs_mag > 12`): shifted using the **multi-transit
  `radial_velocity` with the barycentric correction removed**, i.e. one velocity for all transits.
  26 802 818 sources (*TAP*).

For an SB2 simulator this matters: for faint stars the per-epoch orbital motion is *not* removed
before co-adding, so the DR3 mean spectrum of a faint SB2 is orbitally smeared, whereas a bright one
is shifted to each transit's measured (blended) velocity. (verified)

**Normalisation:** "normalised either to their **pseudo-continuum** or by **scaling with a constant**
(the latter for cool stars or noisy spectra)"; the DR3 doc names M stars with TiO bands and
too-noisy spectra as the constant-scaling cases. The pseudo-continuum fitting procedure itself is
**not documented** (unverified).

**Interpolation method: unverified.** The DR3 documentation says only "interpolated to a common
wavelength array (start λ=846 nm, Δλ=0.01 nm, nbBins=2401)". It never says linear, spline, or
flux-conserving. Do not assert linear.

**Edge artefacts — a real one, worth handling.** The DR3 data model states the mean flux normally
corresponds to `combined_ccds` contributing spectra, "but it may be smaller, **especially in the bins
at the edges of the wavelength range**, which are often not sampled by all the CCD spectra shifted to
the rest frame"; and `flux_error` "may vary within a single spectrum, ... potentially significantly
at spectrum edges" (verified). The rest-frame shift itself depopulates the ends of the grid. Trim a
few nm, or weight by the local contributing count.

### DR4 `rvs_epoch_spectrum`

| Property | Status | Source |
|---|---|---|
| Exists, is a DR4 product | **verified** | ESA "Gaia DR4 content", <https://www.cosmos.esa.int/web/gaia/dr4>, row 17.03 |
| **Per FOV transit, 3 CCDs combined** (not per CCD) | **verified** | Same page, verbatim: "RVS spectra **combined at the FOV-transit level**." Corroborated by the sibling table `rvs_epoch_parameters_single` being described as "FOV-transit-level" |
| **Barycentric frame, normalised** | **verified** | Same page, verbatim: "The spectra are in the **solar barycentric frame** and normalized." (contrast `rvs_mean_spectrum`: "in the rest frame and are normalised") |
| Rows / volume | **6 910 785 949 rows, 49 TB** | Same page |
| 961 samples, 846-870 nm, step 0.025 nm | **UNVERIFIED** | not in any public document I could reach — see below |
| Fields `obs_time_rv, flux, flux_error, normalisation_method, combined_ccds, combined_ccd_in_index, index` | **UNVERIFIED** | same |
| `normalisation_method` codes 0 / 3 / 4 | **UNVERIFIED** | same |

The claimed grid is at least **internally consistent**: `(870 - 846)/0.025 = 960` intervals, so 961
inclusive samples, and 0.025 nm is essentially the native 0.02453 nm sample rather than DR3's 2.45x
oversampling — exactly what you would expect for an epoch product where storage matters (49 TB).
That is a plausibility argument, not a source.

**The only primary source is behind a download.** The ESA DR4 page links
`https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/gaia-dr4-prerelease-draft-data-model_2026-06-26.zip`
(**2 840 832 bytes, Last-Modified 2026-06-26 06:57:22 GMT**, confirmed by HTTP HEAD). I did not
download it. The directory index returns 403, `https://gea.esac.esa.int/archive/documentation/GDR4/`
returns 404, and the archive's `tap_schema` contains **no `gaiadr4` schema** (verified: schemas are
`gaiadr1, gaiadr2, gaiaedr3, gaiadr3, gaiafpr, external, public`, and
`tap_schema.tables WHERE table_name LIKE '%rvs%'` returns zero rows). **If you need the exact DR4
grid and field list, that zip must be fetched — ask the user first.**

Also note the ESA DR4 page's own caveat, verbatim: **"The content of this table is under development
and changes can be expected."** DR4 releases **2 December 2026**; the page was last updated
**28 June 2026**.

### `obs_time_rv` — time origin and scale

**Verified for the DR3 analogue, likely for DR4.** DR3 `vari_epoch_radial_velocity.rv_obs_time`
(<https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_variability_tables/ssec_dm_vari_epoch_radial_velocity.html>)
has unit, verbatim:

> **"Barycentric JD in TCB - 2 455 197.5 (day)"**

and description: "the mean of the observation times of the **three CCDs** used to collect spectra in
the RVS during that transit". So the epoch is **BJD(TCB) - 2455197.5**, i.e. offset from
**J2010.0 = JD 2455197.5 TCB**, which is the standard Gaia time origin used across all Gaia epoch
tables. The notebook's assumption is correct. Applying it to DR4's `obs_time_rv` is **likely** but
not verified — it is the mission-wide convention, but confirm against the draft data model.

Two cautions for a simulator: TCB is **not** TDB (they diverge secularly, ~20 s by 2020 plus a rate
difference of ~1.55e-8), and the "barycentric" here is a light-travel-time correction to the solar
system barycentre, computed **at Gaia**, not at geocentre.

---

## (6) Cadence and transit counts

### Baselines and expected counts

| Release | Data span | Duration | Source |
|---|---|---|---|
| DR3 | 25 Jul 2014 - 28 May 2017 | **34 months** (2.83 yr) | Katz et al. 2023 §3.8, verbatim: "the 34 first months of the nominal mission, that is from 25 July 2014 to 28 May 2017" |
| DR4 | 25 Jul 2014 10:30 UTC - 20 Jan 2020 22:00 UTC | **66 months** (5.5 yr) | ESA DR4 content page |

**The single most important structural fact: RVS gets only 4/7 of the astrometric transits.**
ESA's "End-of-mission Focal-plane Transits" page, verbatim:

> "The spectroscopic pattern follows the astrometric and photometric distribution but is **scaled in
> the number of transits by a factor 4/7** since the spectroscopic instrument has a smaller
> field-of-view which is only served by **4 of the 7** Video Processing Units."

Corroborated by Prusti et al. 2016 §5.2 (`2016A&A...595A...1G`): the sky-average end-of-mission
focal-plane transit count "is around 80 in AF, BP, and RP (**and a factor 4/7 = 0.57 smaller in
RVS**)", and §3.3.7: the RVS is "12 CCDs arranged in three strips of four CCD rows ... 43% (1-4/7)
fewer RVS focal plane transits". The RVS occupies **rows 4-7, strips 15-17** (Cropper et al. 2018).
**A source that transits the astrometric field does NOT automatically get an RVS spectrum.**
(verified)

| Quantity | Value | Source |
|---|---|---|
| Sky-average AF transits, nominal 5 yr | ~80 (~81 at 6% dead time, ~70 at 20%) | Prusti+2016 §5.2; EDR3 doc §1.1.4 Table 1.5 |
| Sky-average **RVS** transits, nominal 5 yr | **~40** (46 before the 20% dead-time allowance) | Cropper et al. 2018 Table 1 ("Average number of transits over mission: 40"), Table 5 ("41, each over 3 CCDs") |
| RVS transit rate | "about **8 transits per star per year**" | Katz et al. 2023 §2 |

| Window | Commanded RVS transits | After ~25% loss | Published median |
|---|---|---|---|
| DR3, 34 mo | ~23 | ~17 | **18** (verified) |
| DR4, 66 mo | ~44 | ~33 | **~35** (estimate: 18 x 66/34) |

### The DR3 `rv_nb_transits` distribution

Katz et al. 2023 §5.2 publishes only min/max/median, verbatim: "The number of transits ranges from
**2** (minimum number for the pipeline to calculate the combined radial velocity) to **227**, with a
**median number of 18**." No percentiles are published.

So I measured them. Live *TAP* histogram of `rv_nb_transits` over a uniform `random_index < 1e7`
subsample (**N = 186 011**), recovering the published median of 18 exactly:

| p1 | p5 | **p10** | p25 | **p50** | p75 | **p90** | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 6 | **8** | 12 | **18** | 24 | **30** | 34 | 46 | **18.69** |

This was **independently reproduced** on a different subsample (`random_index < 2e7`, N = 372 660):
identical p10 = 8, p50 = 18, p90 = 30, mean 18.70. Confidence: **verified**.

### Ecliptic-latitude dependence — measured

Live *TAP*, `AVG(rv_nb_transits)` in 10-degree bins of signed `ecl_lat` (`random_index < 5e6`):

| ecl. lat | -90 | -60 | -50 | -40 | -30 | -20 | -10 | **0** | +10 | +20 | +30 | +40 | +50 | +60 | +90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean `rv_nb_transits` | 27.5 | 18.3 | **24.2** | **24.6** | 17.6 | 15.5 | 13.1 | **12.5** | 13.6 | 15.1 | 18.0 | **25.4** | **26.2** | 19.8 | 24.9 |

The independent `|ecl_lat|`-binned measurement gives medians 12 (0-5 deg), 17 (30-35), 20 (35-40),
**27 (40-45)**, **28 (45-50)**, 22 (50-55), 19 (60-65), 18 (80-90).

**Structure: a ~2.3x peak at |beta| ~ 45 degrees, a minimum on the ecliptic, and a long high tail at
the poles.** The 45-degree rings are the direct consequence of the 45-degree solar aspect angle — the
ESA transits page says of exactly this, "In galactic coordinates, these favoured regions appear as
two rings centred on the ecliptic poles." The pole tail (p90 = 53 at |beta| > 85) comes from the
**first month of Ecliptic Pole Scanning Law** operation (Katz §5.2). Compare the DPAC *astrometric*
prediction, EDR3 Table 1.5: N_obs rises from 61 at beta ~ 0 to 144 at beta ~ 46.4 deg, then falls
back to 75 at the poles — same shape. (verified)

**Caveat (verified, Katz §5.2): transit counts drop sharply in the dense bulge and disc** from
crowding — the window-allocation limit (~35 000 objects/deg^2), onboard memory deletion, and
unrecoverable blending. That is a magnitude/crowding effect, not a scanning-law effect, and no
scanning-law model reproduces it.

### Scanning-law time structure

| Quantity | Value | Source | Confidence |
|---|---|---|---|
| Spin rate | **60 arcsec/s** -> **6.0 h** spin period | Prusti+2016 §5.2: "a fixed spin rate omega_z = 60 arcsec/s" | verified |
| Basic angle | **106.5 deg** | Prusti+2016 §3.3.1 | verified |
| FOV1 -> FOV2 delay | **106.5 min** | EDR3 doc §1.1.4: "phase-shifted by the basic angle multiplied with the spin rate, i.e., by 106.5 minutes of time" | verified |
| Solar aspect angle | **45 deg** | Prusti+2016 §5.2 | verified |
| Precession | **5.8 rev/yr = 4 deg/day**, period **63 d**; constant **S = 4.220745** | Prusti+2016 §5.2, Eqs. (1)-(2) | verified |
| AC image motion | sinusoid, 6 h period, amplitude 173 mas/s | Prusti+2016 §5.2 | verified |
| Distinct epochs | "at least six distinct epochs of observations per year for any object" | Prusti+2016 §5.2 | verified |
| AF CCDs per transit | **9** (AF1-AF9; 8 on row 4, one device is a wave-front sensor) | Prusti+2016 §3.3.5; GDR2 doc §1.1.3 | verified |
| RVS CCDs per transit | **3**, exposure "4.42 s per CCD, or **13.2 s per transit**" | Katz+2023 §2; Cropper Table 5 | verified |
| EPSL -> NSL | EPSL 25 Jul 2014 (OBMT 1078.4) -> 22 Aug 2014 (OBMT 1192.1); "NSL-GAREQ1" from 25 Sep 2014 (OBMT 1326.7) | GDR2 doc §1.3.2 | verified |

**Measured gap ladder** (from a full GOST transit list at RA 45, Dec +20 — 142 transits over 10.5 yr):

| Gap | Count | Interpretation |
|---|---|---|
| **1.78 h** (106.56 min, range 106.56-106.58 across four test positions) | 48 | FOV1 -> FOV2, the basic angle |
| **4.23 h** | 29 | FOV2 -> FOV1 on the next spin (1.78 + 4.23 = 6.01 h) |
| **6.00 h** | 1 | a full spin (missed in one FOV) |
| **7.5 - 30 d** | 64 | between visibility periods |

> **Correction to the brief: there is no ~5-day loop structure in the transit times.** The intra-group
> ladder is 1.78 h / 4.23 h / 6.00 h, and inter-group gaps jump straight to >= 7.5 d — the 1-5 day bin
> is essentially empty. The nearest real "few-day" number is the **4-day gap threshold that defines a
> visibility period** (`gaia_source` data model), a data-model convention rather than a scanning-law
> period. Precession loops have a **63-day** period. Also, "caterpillar"/"Fisheye" could not be
> verified as DPAC terminology; ESA/DPAC describe loops of the spin axis around the solar direction
> with enhanced coverage near |beta| ~ 45 deg. (correction: verified)

**A point that matters more than transit count for orbit fitting.** `rv_visibility_periods_used` has
*TAP*-measured p10/median/p90 = **7 / 13 / 19**, mean 12.66, i.e. only **1.48 transits per visibility
period**, with mean `rv_time_duration` = 869 d. Since transits inside a period sit within ~6 h of
each other, the number of **independent orbital phases** for any binary with P >~ 1 d is closer to
**13** than to 18. Drawing 18 uniformly-random epochs over 34 months would materially overstate
Gaia's phase coverage.

### Tools that return real transit times

**GOST** (<https://gaia.esac.esa.int/gost/>). Manual: GAIA-CU1-UG-ESAC-JFH-005-08, issue 08 rev 1,
2022-08-23, <https://gaia.esac.esa.int/gost/docs/gost_software_user_manual.pdf>.

- **It has a working anonymous programmatic interface**: the IVOA **ObjVisSAP** endpoint,
  `https://gaia.esac.esa.int/gost/ObjVisSAP/gaiaobjvisap?s_ra=<deg>&s_dec=<deg>[&t_min=<MJD>&t_max=<MJD>&MAXREC=<n>&RESPONSEFORMAT=...]`,
  GET or POST, POST accepting a CSV target upload. It returned HTTP 200 with real data when called
  in this session. Columns: `t_validity`, **`t_start` (Barycentric MJD in TCB)**,
  `t_start_gaia_timestamp`, `t_stop`, `t_stop_gaia_timestamp`, `t_visibility` (constant **40.5 s**,
  the 0.675-degree along-scan FOV crossing at 60 arcsec/s). **One row per FOV crossing; no CCD row,
  no FOV label, no scan angle.** ~30 kB per target. Default range
  2014-07-25T10:31:26 -> 2025-07-01 TCB. (verified)
- The older `GostServlet?ra=&dec=` path documented in §4.3/4.4 **returned the HTML query page rather
  than XML** when called anonymously — it now appears to need the web-form session. (verified failure)
- **Web-form CSV columns** (Manual Table 3): `Target, ra[rad], dec[rad], ra[h:m:s], dec[d:m:s],
  ObservationTimeAtGaia[UTC], CcdRow[1-7], normalizedAcPosition[0-1], scanAngle[rad],
  Fov[FovP/FovF], parallaxFactorAlongScan, parallaxFactorAcrossScan,
  ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]`. **Public users get one row per FOV
  transit (AF4); DPAC-authenticated users get all CCDs.** Max **10 000 sources** per submission.
- **Nominal, not as-flown** (verified). Manual §7: "GOST will compute the transit crossing the FoVs
  based on the routine **nominal** scanning law" and "This is not a full guarantee of getting the
  transits observed." The landing page adds the ~80% reception probability and the magnitude gates
  (3 < G < 20; 3 < G_RVS < 16, "lowered to 14 mag" during Galactic Plane Scanning). Decisive proof:
  queries return predicted transits out to **June 2025**, after science operations ended
  15 January 2025.

**`scanninglaw`** (Boubert & Everall, gaiaverse).

| Item | Finding |
|---|---|
| Provides | Predicted scan **times** per FOV, scan counts, gaps/detection-efficiency fractions, and the astrometry spread function |
| Maps | `cogi_2020` (`2020MNRAS.497.1826B`), `cog3_2020` (DR2 **as-flown**, `2021MNRAS.501.2954B`), `dr2_nominal`, `dr3_nominal`, `cogiv_2020` ASF (`2021MNRAS.502.1908E`) |
| **Download sizes (measured)** | `cog_dr2_scanning_law_v1.csv.gz` **486 MB**; v2 CSV **1.10 GB**; `cog_dr2_gaps_and_fractions_v1.h5` **178 MB**; `CommandedScanLaw_001.csv.gz` **570 MB -> 1.55 GB**. `cog3_2020` costs **~1.28 GB** |
| PyPI / maintenance | `scanninglaw` 1.1.1, **last release 2021-06-03**; no `requires_python`, no classifiers; README still says `python setup.py install`. **Treat as unmaintained** |
| Coverage limit | `cog3_2020`, the only genuinely **as-flown** law, covers **DR2 only** (to May 2016) — it does not reach the DR3 baseline, let alone DR4 |

**`gaiaunlimited`** (actively maintained, **0.3.3, released 2026-07-20**, `requires_python >= 3.9`).

| Item | Finding |
|---|---|
| Relevant API | `GaiaScanningLaw(version=..., gaplist="dr3/Spectroscopy").query(ra, dec, count_only=False)` -> arrays of times scanned by the preceding and following FOVs |
| Versions | `dr3_nominal`, `dr2_nominal`, `dr2_cog3`, **`full_operational_mission`** (OBMT 1078.38-17052.625, the whole 10.9 yr) |
| **Gap lists ship inside the wheel and are tiny** | `data/dr3/Spectroscopy.csv` **193 kB / 2830 rows** (`start,end,duration[rev],description,CCD row,origin`, **CCD rows 4-7**), Astrometry 6.9 kB, Photometry 7.8 kB. Origin: <https://www.cosmos.esa.int/web/gaia/dr3-data-gaps> |
| Download sizes | 570 MB (`dr3_nominal`), 1.10 GB (`dr2_cog3`), **2.49 GB** (`full_operational_mission`); ~1 GB cached on disk |
| Dependencies | healpy, xarray, astropy, astroquery, h5py, pandas, scipy, astromet — heavy |
| Bug worth knowing | The `full_operational_mission` entry in `fetch_utils.py` carries `md5sum = d41d8cd98f00b204e9800998ecf8427e`, the MD5 of the **empty string** — a placeholder that will raise `DownloadError` after a 2.5 GB download (likely; read from source, not executed) |
| RVS selection functions | DR3 RVS subsample SF (Castro-Ginard et al. 2023, `2023A&A...677A..37C`) and CoG V RVS SF (`2022MNRAS.509.6205E`) — these give **completeness probabilities, not transit times** |

**Raw ESA products** (sizes measured by HTTP HEAD this session):

| Product | URL | Size | Coverage |
|---|---|---|---|
| EDR3/DR3 commanded scan law | `http://cdn.gea.esac.esa.int/Gaia/gedr3/auxiliary/commanded_scan_law/CommandedScanLaw_001.csv.gz` | **570 MB -> 1.55 GB** | Jul 2014 - May 2017 |
| Full-mission commanded scan law | `https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_General/GaiaScanningLaw/FullGaiaMissionScanningLaw/commanded_scan_law.csv.gz` | **2.49 GB** | whole mission |
| DR2-period (22 months) | `https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_General/GaiaScanningLaw/22months_GaiaDR2period_GaiaScanningLaw.zip` | **446 MB** | Jul 2014 - May 2016 |

Sampling is **10 s**, and ESA warns "this is the **commanded** attitude ... the actual attitude could
deviate from it by up to about **30 arcsec**".

**The overlooked option: query the scan law through TAP, with no download at all.**
`gaiaedr3.commanded_scan_law` is a live TAP table holding **8 967 691** rows (= 34 months / 10 s),
with `jd_time`, `bjd_fov1`, `bjd_fov2`, `obmt_time`, `ra_fov1/dec_fov1/heal_pix_fov1/scan_angle_fov1`
and the FOV2 equivalents. Times are JD in TCB with origin 2455197.5. `heal_pix_fov1/2` are
**HEALPix level 12**, so a cone search on the FOV centre is an indexed query rather than a 570 MB
download. Docs: <https://gea.esac.esa.int/archive/documentation/GEDR3/Gaia_archive/chap_datamodel/sec_dm_auxiliary_tables/ssec_dm_commanded_scan_law.html>
(verified that the table exists and returns rows; the level-12 index trick is *likely*, not tested.)

### Conclusion: cheapest route for a small pure-Python package

**There is no option that is both cheap and offline and gives real transit times.** Ranked:

| Option | Download | Offline? | Pure Python? | Verdict |
|---|---|---|---|---|
| **GOST `ObjVisSAP`** | 0 (~30 kB/target) | no | **yes** (`urllib` + a VOTable regex) | **best when the network is acceptable** — real commanded epochs, whole mission, anonymous, up to 10 000 targets |
| Gaia TAP `gaiaedr3.commanded_scan_law` | 0 | no | yes | good; also gives scan angle, but you do the FOV geometry yourself |
| `gaiaunlimited` | 0.57-2.5 GB | after first fetch | no (healpy, xarray, h5py, astromet) | too heavy |
| `scanninglaw` | 0.49-1.28 GB | after first fetch | no | plus unmaintained since 2021 |
| Reimplement the NSL | 0 | **yes** | **yes** | the only offline route |

**The honest offline approximation, in two separable layers.**

*Layer 1 — analytic NSL, no data at all.* The NSL is fully specified by Prusti et al. 2016
Eqs. (1)-(2):

```
nudot * sin(xi) = lambdadot_sun * sqrt(S^2 - cos^2(nu)) + lambdadot_sun * cos(xi) * sin(nu)
Omegadot        = omega_z - lambdadot_sun * sin(xi) * sin(nu) - nudot * cos(xi)
```

with `xi = 45 deg`, `omega_z = 60 arcsec/s`, **`S = 4.220745`**. RK4 at ~1 h steps for `nu`
(63-day period), `Omega` analytically at 6 h, then test whether the source falls in the FOV — about
150 lines of NumPy, no data files. DPAC does exactly this and Chebyshev-fits the result.

*Layer 2 — draw counts empirically.* Apply the 4/7 RVS factor, then rescale or resample so the
marginal count matches the measured `rv_nb_transits` distribution **conditioned on ecliptic
latitude** — the ~18 numbers in the table above, about 200 bytes hard-coded.

**What that gets right:** the micro-cadence *exactly* (the 106.5-min pair, the 4.23 h complement, the
6.00 h spin all follow from constants, and were verified against GOST to +/-0.02 min); the 63-day
precession envelope and the resulting clustering into ~13 visibility periods with ~1.5 transits each;
the ecliptic-latitude dependence including the 2.3x excess at 45 deg; the count distribution and the
~870-day series span. **In short, the autocorrelation structure that governs period aliasing** —
which is exactly what a disentangling or orbit-fitting test needs.

**What it gets wrong, and must be said out loud:**

- **Absolute phase.** The two free parameters (initial spin phase `Omega_0` and precession phase
  `nu_0`) are not published as numbers anywhere found, and the flown law is a *sequence* of segments
  (EPSL to 22 Aug 2014, then NSL, then "NSL-GAREQ1" from 25 Sep 2014). You get a statistically
  correct but individually wrong epoch list. **You cannot use it to claim "Gaia observed this star on
  these dates."**
- **The first month.** EPSL is a different law entirely and produces the pole excess; a pure NSL model
  misses it.
- **Real gaps.** Only 92.2% of the 34 months were ingested; three decontaminations and two refocus
  events removed specific intervals. If you want these, the DPAC list is **193 kB** and
  redistributable (<https://www.cosmos.esa.int/web/gaia/dr3-data-gaps>) — **the one small data file
  worth vendoring.**
- **Crowding losses** in the bulge/plane, which no scanning-law model reproduces.
- **The 4/7 factor is not an i.i.d. coin flip.** Whether a transit lands on an RVS row depends on the
  across-scan position, which drifts slowly and is correlated *within* a visibility period; treating
  it as independent per transit will get the clumping of RVS epochs wrong. (likely — reasoned from
  the instrument layout, not measured.)
- **Scan angle** is not reproduced unless you carry `Omega(t)`, and then only up to the phase error.

**Recommendation:** Layer 1 + Layer 2 offline as the default (honest, dependency-free, correct in
distribution), plus an optional `fetch_gost()` helper hitting `ObjVisSAP` with `urllib` — ~40 lines,
no new dependency — when a user wants the real epochs for a specific star.

---

## (7) DR3/DR4 double-lined detections

### The "787,312 double-lined RVS transits" figure — SOURCED

**Primary source: ESA "Gaia DR4 content", <https://www.cosmos.esa.int/web/gaia/dr4>, contents table
row 17.01.** Confidence: **verified**.

| Table | Description (verbatim) | Rows | Volume |
|---|---|---|---|
| `rvs_epoch_parameters_double` | "FOV-transit-level information for the radial velocity, spectral line broadening, and Grvs magnitudes for **double-lined** transits." | **787 312** | 128 MB |
| `rvs_epoch_parameters_single` | "...for single-lined transits." | 6 910 423 894 | 323 GB |
| `rvs_epoch_spectrum` | "RVS spectra combined at the FOV-transit level..." | 6 910 785 949 | 49 TB |

**Two caveats that must travel with the number.** (i) The unit is **transits, not stars** — quoting
it as a count of SB2 systems would be wrong. (ii) ESA's own page says "The content of this table is
under development and changes can be expected", and the page is dated 28 June 2026, so it is a
**pre-release planning figure**. Double-lined transits are therefore ~0.011% of all RVS transits.
The DR4 paper that will define this table, "Gaia Data Release 4: Measuring and validating radial
velocities of RVS double lined spectra and the related SB2 orbital solutions", is listed on
<https://www.cosmos.esa.int/web/gaia/dr4-papers> **with no link, i.e. unpublished** (verified).

### DR3 columns (all verified from the DR3 data model, §20.1.1 `gaia_source`)

| Column | Definition |
|---|---|
| `rv_nb_transits` | Number of transits used for `radial_velocity`. "In general, one transit in the RVS field-of-view includes **3 RVS CCD observations**." |
| `rv_expected_sig_to_noise` | Expected S/N **in the combination** of the spectra used — see item (2) |
| `rv_template_teff/logg/fe_h` | Parameters of the **synthetic template**. The data model warns three times: "the purpose of this parameter is to provide information on the synthetic template spectrum used, **not** to provide an estimate of the stellar effective temperature" |
| `rv_method_used` | **1** = bright (generally `grvs_mag <= 12`), `radial_velocity` = **median of epoch RVs**; **2** = faint, the **epoch CCFs are combined** and the RV measured from the combined CCF |
| `rv_visibility_periods_used` | Groups of transits separated by gaps of **>= 4 days** |
| `rv_renormalised_gof` | `sqrt(4.5 nu)(RUWE^(2/3) + 2/(9 nu) - 1)`, `nu = rv_nb_transits - 1`. **Bright stars only** (non-null for 6 993 005, all `grvs_mag <= 12`) |
| `rv_chisq_pvalue` | P-value for RV **constancy**; 1 = strongly constant. Bright only |
| `rv_time_duration` | First-to-last transit span, days |
| `rv_amplitude_robust` | Outlier-clipped `max - min` of the RV series, km/s. Bright only |
| `grvs_mag`, `grvs_mag_error`, `grvs_mag_nb_transits` | Median of epoch `G_RVS`, its error, and the count (**non-blended transits only**) |
| `rvs_spec_sig_to_noise` | **Median over bins of `flux/flux_error`** in the published mean spectrum; only where a mean spectrum exists |

### `rv_assumed_sb2`

- **It does not exist in DR3** (or DR1/DR2/EDR3/FPR): a `tap_schema.columns` query for
  `%sb2%` or `%double%` across every schema returns **zero rows** (verified, *TAP*).
- It is a **DR4** column, defined only in the draft data model zip. Meaning (from a full read of that
  PDF recorded earlier in this project, **not re-verified this session — likely**): set when double
  lines were detected in **>= 10 transits**, with detection running only for
  `external_apparent_grvs <= 12`; setting it **excludes the source from Gaia's multi-transit RV
  solution**; only **about one third** of assumed-SB2 systems are later confirmed as SB2 by NSS
  processing. Table: `all_source_rvs` (likely). Data type: boolean (unverified).
- **DR3 published no per-source double-line flag at all.** `rv_nb_deblended_transits` is **not** one —
  it counts transits deblended from *neighbouring sources*.

### What DR3 published for SB2s (verified, *TAP*, `gaiadr3.nss_two_body_orbit`)

`SB2` = **4 630**, `SB2C` = **746**, total **5 376** double-lined orbits (of 443 205 NSS orbits;
SB1 = 181 327, SB1C = 202). `SB1+SB1C+SB2+SB2C = 186 905` reproduces exactly the "SB1 or SB2" row of
Table 1 of **Gaia Collaboration, Arenou et al. 2023, A&A 674, A34**
(`2023A&A...674A..34G`, [10.1051/0004-6361/202243782](https://doi.org/10.1051/0004-6361/202243782)),
which publishes only the combined figure.

**Katz et al. 2023 (`2023A&A...674A...5K`, [10.1051/0004-6361/202244220](https://doi.org/10.1051/0004-6361/202244220))**, verbatim:

> "Approximately 40 000 sources with 10% or more transits flagged as double-line were considered SB2
> candidates and their combined radial velocities were discarded."

DR3 doc §6.4.8: "the publication of Gaia DR3 `radial_velocity` is **strictly limited to single-lined
spectra**." Detection is by **TODCOR** (Zucker & Mazeh 1994) with an F-test against the single-lined
model: `TodCorLight` (same template both components) screening into `TodCorHeavy` (different
templates per component, 99.9% confidence). Arenou et al. 2023 warns that SB2 sources normally have
**no** `gaia_source.radial_velocity`; use `center_of_mass_velocity`. (all verified)

So DR3: ~40 000 SB2 candidates detected, **0** SB2 velocities in `gaia_source`, **5 376** SB2/SB2C
orbits in NSS — a ~13% conversion.

### The 14 500 K template limit

**Verified exactly.** `MAX(rv_template_teff) = 14500.0` over all 33 812 183 DR3 RV sources (*TAP*),
with 6 525 sources sitting at exactly 14 500 K. The 37 distinct published values:

```
3100..3900 step 100 | 4000..8000 step 250 | 8500..12000 step 500 | 13000..14500 step 500
                                             (12 500 K is ABSENT)
```

The 12 500 K hole is the fingerprint of a missing node in the A-type synthetic library. DR3
documentation §6.2.3 lists the full 6 772-spectrum library — MARCS (4 306, 2500-8000 K), A-type
(304, 8500-14 500 K step 500, **missing 12 500**), OB-type (2 162, 15 000-55 000 K) — with the
decisive sentence that the OB library "has not been used to estimate the radial velocities published
DR3, which are limited to `rv_template_teff` <= 14 500 K". (verified)

The cut is **strictly greater than** 14 500 K; Katz et al. 2023 §7: "the combined radial velocities
of approximately **66 000 stars** with `grvs_mag <= 12` and `rv_template_teff` **> 14 500 K** were
removed." (The DR3 documentation §6.5.2 misstates this as ">=" — the archive settles it.) A second,
harsher cut removed `rv_template_teff >= 7000 K` for `grvs_mag > 12` (~1.7 million sources), and
`< 3100 K` (~243 000).

**Consequence for a hot-binary simulator: DR3 contains no radial velocities for O/early-B stars at
all, and none for any hot star fainter than `grvs_mag ~ 12`.** This is a *template-library* limit,
not a detector limit — the OB grid was computed and shipped, just not used. It is exactly why
HR 6819-class targets are absent from DR3.

Other verified magnitude facts: DR3 RVs span `grvs_mag` **2.758 to 14.0999** (the documented filter
is `> 14.5`, ~21 000 removed); total RVs **33 812 183** (matches Katz's abstract exactly);
`rvs_mean_spectrum` **999 645** rows spanning `grvs_mag` 2.870-13.176, of which 81.8% are
`<= 12`. The actual spectrum-publication criterion was **not** a magnitude cut but
**`rvs_spec_sig_to_noise >= 15`** (observed min 15.000006), plus `combined_transits > 2` and
<= 480 NaN samples. DR3 doc §6.5.2 warns "the spatial distribution of the sources contains noticeable
patterns and gaps".

---

## (8) Everything else a simulator must know

### (8a) Dead time and lost transits

| Quantity | Value | Source | Confidence |
|---|---|---|---|
| GOST's own caveat | "the probability of the data of the target being received on the ground at the indicated time is about **80%**" | <https://gaia.esac.esa.int/gost/> | verified |
| ESA transit budget | "conservatively accounting for **~20% dead time and data losses**"; "a star transits the spectroscopic instrument on average ~40 times, leading to ~120 CCD detector transits" | ESA science-performance page | verified |
| DR3 effective loss | "the effective number of transits used to derive the radial velocities is **reduced by about 25%** by the combination of the dead-time ... and of the processing filters" | Katz et al. 2023 §2 | verified |
| DR3 mission time ingested | "**92.2%** of the 34 months were ingested in the spectroscopic pipeline, while the remaining **7.8%** were either unavailable or considered unfit" | Katz et al. 2023 §3.8 | verified |
| DR2 excluded intervals | "~200 revolutions or **~7.5%** of the total observation time" | Sartoretti et al. 2018 §3.2 | verified |
| RVS transit rate | "on average the RVS records about **8 transits per star per year**" | Katz et al. 2023 §2 | verified |
| Pre-launch dead-time allowance | **0.14** (V=7) to **0.42** (V=16.5-17), magnitude-dependent | Cropper et al. 2018 Table 4 | verified |

**The "~6% dead time" in circulation is unsourced.** The two defensible mission-level numbers are
**7.8%** (DR3 span unavailable/unfit) and **~20%** (ESA's dead time + data losses).

**Recommended value for a simulator: 0.78 of GOST-predicted RVS transits yield a usable spectrum.**
Three independent lines converge: GOST's own ~80%; Katz's ~25% reduction (0.75); and the arithmetic
cross-check — 8 transits/star/yr x 34/12 months = 22.7 predicted vs the published DR3 median of 18,
giving **0.79**. My own version of that check using Cropper's nominal 40 transits per 5 years gives
40 x 34/60 = 22.7 predicted, same answer. This applies to a **bright, isolated** star; for
`G_RVS > 14`, in fields denser than ~35 000 deg^-2, or during Galactic Plane Scanning, the fraction
falls sharply and is dominated by onboard window allocation rather than dead time.

Causes (all verified): decontaminations 2014-09-23, 2015-06-03, 2016-08-22 (each needing ~70
revolutions to re-equilibrate); refocus 2014-10-24, 2015-08-03; micrometeoroid hits at "about **five
hits per month**" causing gaps of "a few minutes" (DR3 doc §4.3.5); stray light, which forced the
RVS limiting magnitude down from the design `G_RVS <= 17` to 16.5, then 16.2, and finally to an
in-flight adaptive **15.3-16.2 depending on instantaneous stray light** (Cropper §§9.1, 10);
orbital maintenance; onboard memory deletion when both FOVs scan the Galactic plane; and
transmission losses.

Deblending is a large effect in DR3: "de-blended spectra represent a little more than **25%** (about
540 million out of 2 billion)" and "about **96%** [of published stars] have at least one de-blended
transit" (Katz §3.5). ~855 million of 2.8 billion spectra were filtered out in the extraction
workflow alone.

### (8b) Lines in the band, vacuum wavelengths

The RVS scale is **vacuum**. Blomme et al. 2023 (A&A 674, A7): "All spectra are at their vacuum rest
wavelength"; ESA IoW 2021-07-09: "the wavelength interval between 846 and 870 nm (in vacuum)".
(verified)

**Ca II infrared triplet.** Source for the air values and their tabulated vacuum counterparts:
**Contursi et al. 2021, A&A 654, A130** (`2021A&A...654A.130C`), VizieR **J/A+A/654/A130** table
`calib`, which carries both `lambdaAir` and `lambdaVac`. I independently reproduced the vacuum column
with the IAU-standard air index to 0.0001 nm.

| Air (nm) | **Vacuum (nm)** | log gf (RVS-calibrated) |
|---|---|---|
| 849.8023 | **850.0358** | -1.44 |
| 854.2091 | **854.4438** | -0.50 |
| 866.2141 | **866.4520** | -0.75 |

**Hydrogen Paschen series — and a trap.** Exactly **five** Paschen lines fall inside 846-870 nm.
Naming follows the usual convention where the number is the **upper** principal quantum number.

| Line | NIST air (nm) | **NIST vacuum (nm)** | Naive Rydberg (nm) | Rydberg error |
|---|---|---|---|---|
| Pa17 (17->3) | 846.7254 | **846.9581** | 846.9630 | +0.050 A = **+1.75 km/s** |
| Pa16 (16->3) | 850.2483 | **850.4819** | 850.4869 | +0.050 A = **+1.77 km/s** |
| Pa15 (15->3) | 854.5383 | **854.7731** | 854.7781 | +0.051 A = **+1.77 km/s** |
| Pa14 (14->3) | 859.8392 | **860.0754** | 860.0805 | +0.051 A = **+1.77 km/s** |
| Pa13 (13->3) | 866.5018 | **866.7398** | 866.7450 | +0.052 A = **+1.80 km/s** |

Pa18 (844.03 nm vac) is blue of the band and blue of the instrument edge; Pa12 (875.29 nm) is red of
it; the Paschen limit is at 820.59 nm. I verified the air wavelengths directly against **NIST ASD**
(H I, 846-871 nm, observed and Ritz, accuracy AAA, +/-0.0006 nm) and converted them myself; my
conversion reproduces Contursi's tabulated vacuum values to 0.0001 nm. **verified**

> **Do not generate Paschen wavelengths from the Rydberg formula.** `R_H = R_inf/(1+m_e/m_p)`
> omits the Dirac/QED corrections to the n=3 level and is **+0.05 A (+1.8 km/s) too red on every
> line in this band** — a systematic larger than most effects an OB-binary study would be trying to
> measure. Use the NIST Ritz values above.

**Ca II / Paschen blending** (computed from the vacuum values above): Ca II 866.452 sits only
**99.6 km/s** blue of Pa13; Ca II 854.444 is **115.5 km/s** from Pa15; Ca II 850.036 is
**157.3 km/s** from Pa16. For SB2s with K of this order the components' Ca II and Paschen features
sweep through each other — worth having in the simulator's test cases.

**N I — the lines that matter for OB stars** (NIST ASD, vacuum; air values computed):

| **Vacuum (nm)** | Air (A) | gA (s^-1) | Multiplet |
|---|---|---|---|
| 857.0089 | 8567.74 | 1.94e7 | 3s 2P - 3p 2P |
| 859.6360 | 8594.00 | 4.18e7 | 3s 2P - 3p 2P |
| 863.1606 | 8629.24 | 1.07e8 | 3s 2P - 3p 2P |
| 865.8256 | 8655.88 | 2.14e7 | 3s 2P - 3p 2P |
| **868.2666** | 8680.28 | **2.02e8** | 3s 4P - 3p 4D |
| **868.5788** | 8683.40 | 1.13e8 | 3s 4P - 3p 4D |
| 868.8535 | 8686.15 | 4.60e7 | 3s 4P - 3p 4D |
| 870.5637 | 8703.25 | 4.32e7 | 3s 4P - 3p 4D (at the red edge) |

N I 8680.28 and 8683.40 A (air) are the standard nitrogen-abundance diagnostics in B stars and are
inside the published band. (verified)

**He I: nothing strong in band.** NIST lists only ten weak high-n He I transitions in 846-870 nm
(gA = 3.5e4 to 2.9e6, two to four orders below the N I lines). The strong He I 8776.7 A air
(**878.89 nm vacuum**) is outside. (verified)

**Cool-star metals:** load **Contursi et al. 2021, VizieR J/A+A/654/A130** table `calib` rather than
hand-typing — 163 calibrated lines with both air and vacuum columns, covering Na I, Mg I, Si I, S I,
Ca I, Ca II, Ti I/II, Cr I, Mn I, Fe I/II, Co I, Ni I, Sm II. Strongest for RV work: Si I 865.0841
(log gf +0.25), S I 869.7014, Si I 855.9127, S I 867.0821, Ca I 863.6306, the Ca II triplet, Fe II
858.7939. Note the only Mg I line in the calibrated list is 861.2092 nm vacuum and it is weak — the
strong Mg I 8806.8 A is outside the band. (verified)

**DIB 862 nm is in band and is the only significant non-stellar absorption.** Gaia DR3 rest
wavelength **8620.86 +/- 0.019 A in air** (Gaia Collaboration/Schultheis et al. 2023, A&A 674, A40)
= **862.3228 nm vacuum**; Saydjari et al. 2023 revise it to 8623.141 +/- 0.030 A vacuum.
(verified for the DR3 value, likely for Saydjari)

**No tellurics** — Gaia observes from L2. This is also why the scale is vacuum. (verified)

### (8c) Spectral edges and response

| Quantity | Value | Source |
|---|---|---|
| Bandpass requirement | 847-874 nm (multilayer filter) | Cropper Table 1, §5.2 |
| **Measured in-flight bandpass** | **845.0-872.5 nm (FWHM); 845.5-872.0 nm (at 90%)** | Cropper Table 5, §11 |
| Total optical throughput | 0.62 (mirrors 0.83 x filter 0.95 x grating 0.80 x prisms/lenses 0.85 x contamination 0.93 x microroughness 0.99) | Cropper Table 2 |
| CCD QE | **0.76 at 847 nm -> 0.65 at 874 nm** | Cropper §8.1 |
| **Total response** | **0.47 at 847 nm -> 0.40 at 874 nm** | Cropper Table 2 |
| Out-of-band rejection | <=10% required, "marginally exceeded at shorter wavelengths at some field points" (700-750 nm leakage) | Cropper §8.1 |
| Sampling | **2.798 +/- 0.009 pixels per optical resolution element** (T1 R4 S15) | Cropper Table 6 |
| Window length | **1260 or 1296 AL pixels** depending on onboard software version | Sartoretti 2018 §3; Cropper §10 |
| Empirical DR3 passband | "better than the pre-launch estimate (by a factor of about **1.23**) and slightly shifted to the blue"; tabulated in `GAIADR3_GRVSFILTER.zip` at <https://www.cosmos.esa.int/web/gaia/dr3-passbands> | Sartoretti 2023 Fig. 14 |

In one sentence: a near-flat filter-defined top-hat with steep multilayer edges at ~845 and ~872 nm,
tilted by the CCD QE so it falls **~15% monotonically from blue to red** across the band.

**The published 846-870 nm range is a TRIM, and here is why** — Sartoretti et al. 2023, verbatim:

> "This is the widest possible (integer) wavelength range properly sampled by all spectra of a given
> source, as the spectra obtained over various transits are not uniformly sampled in the wings."

> "Each RVS observation window is divided into 12 subunits of 108 AL pixels, called macrosamples. To
> limit the processing load on board, all windows in a given CCD are phased at macrosample level and
> can start only at macrosample boundaries. ... the spectra of a given source obtained in different
> observation windows may have their ends cut off by up to **108 AL pixels**."

108 AL pixels x 0.0245 nm = **2.65 nm of potential end truncation per transit** — that is the entire
reason for the 845 -> 846 and 872 -> 870 trim. (verified)

**Edge artefacts in `rvs_mean_spectrum`** (verified): the data model states this "results in typically
**larger `flux_error` at the edges** of the combined spectrum", because edge bins "are often not
sampled by all the CCD spectra"; GSP-Spec notes spectra can have "zero flux values at the beginning
or at the end of their spectral range" as a consequence of the rest-frame shift. **Practical rule:
treat the first and last ~10-30 samples as having inflated, transit-count-dependent errors and
possibly zero flux; do not fit them.**

### (8d) Normalisation

**DR3 `rvs_mean_spectrum`** — DR3 documentation §6.1.1, verbatim (verified):

> "CCD RVS spectra are extracted, cleaned, deblended (if needed), wavelength calibrated and
> **normalised either to their pseudo-continuum or by scaling with a constant (the latter for cool
> stars or noisy spectra)**."

Normalisation is applied **per transit, before averaging**. It is therefore **not** a guaranteed
continuum-at-1.0: GSP-Spec warns explicitly that cool stars "have pseudo-continuum flux values that
can be **much lower than one**", and re-derives the continuum itself (residuals `Res = S/O` against
an interpolated synthetic, iterative linear fit with sigma-clipping, a **third-degree polynomial**
trend, and five iterations of the whole parameter/continuum loop). The pseudo-continuum fitting
procedure used by CU6 itself is **not documented** (unverified).

Note a documentation inconsistency: the archive data model says **2401** elements while the GSP-Spec
papers repeatedly say "2400 wavelength points". **The published arrays are 2401.** GSP-Spec further
rebins to 800 points at 0.03 nm before parameterisation, with no resolution loss.

**DR4 `rvs_epoch_spectrum`**: "The spectra are in the **solar barycentric frame** and normalized"
(verified). **The `normalisation_method` codes 0/3/4 could not be verified** — see item (5). The one
available inference: DR3 used exactly two modes, so a multi-valued DR4 code implies at least one new
mode. That is a guess, not a source.

### (8e) Two access facts

- **DR3 mean spectra are DataLink-only.** There is no `rvs_mean_spectrum` TAP table on ESA's server
  (verified: `tap_schema.tables WHERE table_name LIKE '%rvs%'` returns zero rows). Retrieval is
  `https://gea.esac.esa.int/data-server/datalink/links?ID=Gaia+DR3+<source_id>`, retrieval type
  `RVS`, formats VOTable/FITS/CSV/ECSV, **max 5000 sources per call**, and the **wavelength array is
  returned** (nm, UCD `em.wl`) rather than needing reconstruction from start/step.
- **`grvs_mag` is null for ~1.5 million of the 33.8 million RV sources** (32 276 087 have it,
  Katz §5.2) — it is not estimated when all available transits are deblended or reblended. A
  simulator that keys off `grvs_mag` must handle that, which is exactly where the item (1)
  colour relation earns its place.

---

## Discrepancies found

**Nothing in the notebook is wrong.** Every constant checked out against a primary source. The
findings below are refinements, provenance corrections, and traps.

| # | Item | Finding | Severity |
|---|---|---|---|
| 1 | (1) G_RVS polynomial | **No discrepancy.** All eight coefficients, both signs, the `x = 1.2` breakpoint and the `-0.15 <= x <= 1.7` range match Sartoretti et al. 2023 Eqs. (9) and (10) exactly. Add the quoted RMS (0.04 / 0.09 mag) as an uncertainty | none |
| 2 | (2) whole formula | **No discrepancy.** It is verbatim ESA's science-performance model, and it reproduces the published `rv_expected_sig_to_noise` to **3-8%** over 8 magnitudes | none |
| 3 | (2) read noise | ESA uses **3.2 e-**; Cropper Table 7 measures a mean of **3.1 e-** (range 2.9-3.4). Effect <0.3% at G_RVS=12. Keep 3.2 | cosmetic |
| 4 | (2) background | **4.7 e-/pixel is a mission median, not a constant.** Sartoretti 2023 quotes nominal stray light as 70 e-/sample (=7.0 e-/pix); Sartoretti 2018 §5.2 gives 6.54 e-/pix; Cropper §9.1 gives 20-30 e-/pix at bad spin phases. A per-epoch simulator should vary it | **moderate** |
| 5 | (2) provenance | The notebook's formula is ESA's **predictive** variant. The catalogue column was computed from a **measured** variant using `medianFlux`, `N_ValidStrips` and `medianBackground`. Do not describe the notebook's as "what DR3 used" | documentation |
| 6 | (2) meaning | `rv_expected_sig_to_noise` is a **combined** S/N over all transits, never per-epoch. Because S, BCK and RN are all exactly proportional to n, **per-transit S/N = SNR / sqrt(n) exactly**, with no approximation | useful |
| 7 | (2) continuum | `S` is the **band-mean** flux per sample, not the continuum. Expect the formula to under-predict continuum S/N by ~1.5x | **moderate** |
| 8 | (3) S/N check | Measured, and it passes. But `rvs_spec_sig_to_noise` (median flux/flux_error) runs **10-15% above** `rv_expected_sig_to_noise` because the 0.01 nm grid is 2.45x oversampled and correlated. Don't conflate the two | useful |
| 9 | (4) resolving power | 11 500 is the **nominal/design** value. The **measured** per-CCD mean is **11 891** (+3.4%), with a **10 983-12 587** spread (+/-7%) across the 12 CCDs and a systematic Telescope 1 / Telescope 2 offset (12 141 vs 11 642) | **moderate** |
| 10 | (4) LSF shape | **Not Gaussian by construction** — DPAC models it as `H_0 + sum h_n H_n` with 8 basis functions (PCA in DR2), calibrated in ten time periods. The size of the departure from a Gaussian has never been published | **known unknown** |
| 11 | (4) wavelength dependence | Negligible and **explicitly neglected by DPAC**. The only real AL wavelength dependence is the dispersion, 0.0244 -> 0.0246 nm/pix, i.e. R varies ~0.8% across the band | none |
| 12 | (5) rest frame | **The notebook's "velocity divided out per transit" is only half right.** True for `rv_method_used = 1` (bright, ~7.0M sources); for `rv_method_used = 2` (faint, ~26.8M) a **single** combined velocity is used, so a faint SB2's DR3 mean spectrum is orbitally smeared | **important** |
| 13 | (5) interpolation | "Linearly interpolated onto the fixed grid" is **unverified**. The DR3 docs say only "interpolated". Do not assert linear | flag |
| 14 | (5) DR4 grid | **961 samples / 0.025 nm / the seven field names / `normalisation_method` codes 0,3,4 are all UNVERIFIED.** Internally consistent and plausible, but the only source is a zip that needs downloading | **open** |
| 15 | (5) DR4 frame | **Confirmed and consequential: DR4 epoch spectra are in the SOLAR BARYCENTRIC frame; DR3 mean spectra are in the REST frame.** Code written against DR3 will silently mis-shift DR4 | **important** |
| 16 | (5) `obs_time_rv` | **BJD(TCB) - 2455197.5 d** confirmed for the DR3 analogue and it is the mission-wide convention; **likely** for DR4. TCB is not TDB, and the correction is computed at Gaia, not geocentre | verified/likely |
| 17 | (7) 787 312 | **Sourced**: row count of DR4 `rvs_epoch_parameters_double` on the ESA DR4 content page. But it is **transits, not stars**, and the page says "content ... under development and changes can be expected" | **important** |
| 18 | (7) `rv_assumed_sb2` | **Does not exist in DR3** — zero matching columns across every schema on the Gaia TAP server. It is a DR4 column | flag |
| 19 | (7) 14 500 K | Confirmed exactly (`MAX(rv_template_teff) = 14500.0`). The cut is **strictly `> 14 500`** (Katz); DR3 doc §6.5.2 misstates it as `>=`. The published grid also has a **hole at 12 500 K** | verified |
| 20 | (8) Paschen | **A real bug risk: the Rydberg formula is +0.05 A = +1.8 km/s too red on every Paschen line in the band.** Use NIST Ritz values | **important** |
| 21 | (8) dead time | The "~6%" figure is **unsourced**. Defensible numbers: 7.8% of mission time unavailable/unfit (Katz §3.8), ~20% ESA allowance, ~25% effective reduction (Katz §2). **Use 0.78 usable fraction** | **moderate** |
| 22 | (8) band edges | 846-870 nm is a **trim** of a 845-872.5 nm instrument, forced by 108-pixel macrosample phasing (2.65 nm of possible per-transit truncation). Edge samples have inflated, transit-count-dependent errors | **moderate** |
| 23 | (6) 5-day loops | **The premise is wrong. There is no ~5-day loop in the transit times.** The gap ladder is 1.78 h / 4.23 h / 6.00 h, then straight to >= 7.5 d. The nearest real number is the **4-day threshold defining a visibility period**, a data-model convention. Precession loops are **63 days** | **correction** |
| 24 | (6) 4/7 factor | **RVS gets only 4/7 of astrometric transits** (rows 4-7 of 7, "only served by 4 of the 7 Video Processing Units"). A simulator converting GOST output to RVS epochs must apply this — and it is **not** an i.i.d. per-transit coin flip, since AC position is correlated within a visibility period | **important** |
| 25 | (6) phase coverage | Only **1.48 transits per visibility period**; independent orbital phases track `rv_visibility_periods_used` (median **13**), not `rv_nb_transits` (median 18). Drawing 18 uniform epochs over 34 months **overstates** Gaia's phase coverage | **important** |
| 26 | (6) terminology | "caterpillar"/"Fisheye" could not be verified as DPAC terminology (the physics — rings of enhanced coverage at \|beta\| ~ 45 deg — is verified) | cosmetic |
