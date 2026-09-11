# E — AI Phoenicis: the ~1 % semi-amplitude systematic

Read-only investigation, 2026-09-03. No repository file was modified; the disentangling was
not re-run. Every number below comes either from the repository record, from the FITS
headers and flux arrays in `data/aiphe/`, or from the literature, and the scripts used are
in the sibling scratchpad directory (`phases.py`, `projection.py`, `acf.py`, `naming.py`,
`datacheck.py`).

---

## 0. The one-paragraph answer

The bias is **not** in the data and **not** in the ephemeris. It decomposes exactly into a
**component-antisymmetric** part of −0.550 km/s and a **component-symmetric** part of
−0.162 km/s, and the antisymmetric part alone accounts for the entire mass-ratio offset
(2·0.550/50.14 = 2.20 % predicted, 2.17 % observed). An antisymmetric shift of K₁ and K₂ is
the fingerprint of **line signal being exchanged between the two recovered component
spectra along the low-frequency near-null direction of the disentangling design**
(`docs/math.md` §5.1): d̂₁ carrying +0.33 % of s₂ and d̂₂ carrying −0.85 % of s₁ reproduces
both numbers. Two deterministic drivers of that exchange are identified and not yet
excluded: **0.5–1.0 % of unmodelled third light** (Maxted et al. 2020 measure it; Gallenne
et al. 2019 find the third body that produces it), and **a single R = 115 000 LSF declared
for six epochs that are EGGS-mode at R = 80 000**. The likelihood/prior balance is
independently wrong by 2.64² ≈ 7×, and the smoothness prior is the only thing that pins the
exchange mode, so whatever pushes it is amplified sevenfold.

Everything else on the candidate list is excluded quantitatively, most of it by a single
logical observation: **the published K₁, K₂ come from the same 33 of these 36 spectra**
(Gallenne et al. 2019, broadening functions in RaveSpan), so no property of the data —
barycentric correction, wavelength solution, timing, EGGS zero point, tellurics — can
explain a difference between the two analyses.

---

## 1. What the record says, extracted exactly

Source files: `docs/benchmarks.md` (the section beginning line 537, and the D55 section
beginning line 2517), `scripts/aiphe_bench.py`, `scripts/aiphe_labels_bench.py`,
`scripts/download_aiphe.py`, `docs/tutorials/aiphe-labels.ipynb`.

| item | value | where |
|---|---|---|
| epochs | all 36 `data/aiphe/*.fits`, **no exclusions** (`load()` globs `*.fits`) | `aiphe_bench.py:load` |
| provenance | ESO Phase-3 `SCIENCE.SPECTRUM`, cone search at RA 17.392458°, Dec −46.265583°, r = 0.05°, `INSTRUMENT=HARPS`, `EXPECTED_N=36` | `download_aiphe.py` |
| windows | 5150–5250 Å primary; 5340–5440 Å as the disjoint cross-check; `region_pad_angstrom=3.0`, `smooth_angstrom=25.0` | `aiphe_bench.py` |
| model grid | `dv_kms = 0.8`, `v_margin_kms = 140`, `LogGrid.covering` | `aiphe_bench.py:DV_KMS` |
| LSF | `LSF_SIGMA_V = c/115000/2.3548 = 1.1071 km/s`, applied as `{name: LSF_SIGMA_V for name in ds.instruments}` | `aiphe_bench.py` |
| light fractions | blackbody from R₂/R₁ = 1.6237 and 6310/5010 K → **ℓ₁ = 0.5442, ℓ₂ = 0.4558** at 5200 Å (0.5366/0.4634 at 5390 Å) | `aiphe_bench.py:light_fractions` |
| prior | `SmoothnessPrior(tau = 300, eta = 5)` for both components | `aiphe_bench.py:TAU, ETA` |
| fitted parameters | `period`, `t_conj`, `secosw`, `sesinw`, `k[2]`, `log_tau[2]`, `log_eta[2]`. **No γ** (D14: identically zero) | `aiphe_bench.py:priors` |
| priors on the orbit | P ~ U(24.5824, 24.6024); t_conj ~ U(T₀ ± 0.05 d); secosw, sesinw ~ U(−0.8, 0.8); k ~ U(30, 70) both | ibid. |
| initialisation | P₀ = 24.5924, t_conj₀ = T₀, e-vector at 0.85× published, **k₀ = (47.07, 53.03)** — deliberately swapped relative to the truth | ibid. |
| reference values | K₁ = 51.164 ± 0.007, K₂ = 49.106 ± 0.010, P = 24.5924 d, e = 0.1878 ± 0.0006, ω = 110.30°, T₀ = BJD_TDB 2458362.82847 — **Maxted et al. (2020), MNRAS 498, 332, Tables 2–3** | `aiphe_bench.py` header |
| fitted result | K₁ = 50.452 (−1.39 %), K₂ = 49.495 (+0.79 %), q = 1.0193, e = 0.1879 (+0.05 %); second window 50.440 / 49.479 / 1.0194 | `benchmarks.md` |
| convergence | \|grad\| = 9e−03, same answer at 250 and 444 steps | `benchmarks.md` |
| noise | HARPS ships no error array → ivar from albireo's own local-scatter estimate; **residual z-RMS = 2.64** | `benchmarks.md` |
| **not reported** | the fitted `period`, `t_conj`, and the *signs* of `secosw`/`sesinw` | — |

Two things the record does not report are load-bearing and should be added to it:

* **e is quoted as `sec² + ses²`, which is invariant under (sec, ses) → (−sec, −ses).** The
  reported agreement of e to 0.05 % therefore does *not* by itself exclude a 180° error in
  ω. (§4.2 closes that gap from the data instead.)
* **the fitted period and t_conj.** If either ran to its prior bound, the "not the
  optimizer" claim needs restating.

The D55 label run reuses the same window, grid, LSF and light fractions with the velocities
**fixed** at the published orbit (`aiphe_labels_bench.py:disentangle`), so the label result
inherits every assumption below.

---

## 2. The 36 epochs

All 36 files are ESO Phase-3 1-D products derived from **`s1d`** (`ORIGFILE` =
`HARPS.<date>_s1d_A_DRS_<ver>_ESOSDP.fits`), `SPECSYS = 'BARYCENT'`, `SPEC_BIN` = 0.001 nm
(0.577 km/s at 5200 Å), `M_EPOCH = F`, `NCOMBINE = 1`. `TMID` is present in the SPECTRUM
extension of every file, so albireo takes the mid-exposure time from `TMID` (not
`MJD-OBS + EXPTIME/2`) and converts it to BJD_TDB with astropy.

Ephemeris used for the phases: **T₀ = BJD_TDB 2458362.82850** (Maxted et al. 2020, TESS
primary minimum), **P = 24.592483 d** (Kirkby-Kent et al. 2016), e = 0.1875, ω = 110.34°.
Derived geometry from Maxted run C (r₁ = 0.037724, r₂ = 0.061253, i = 88.357°):

* primary eclipse: full duration **15.1 h = 0.0128 in phase either side of φ = 0**;
  projected separation at mid-eclipse d = 0.023527 against r₂ − r₁ = 0.023529 → **total**
  (Maxted et al.: "the primary eclipse is total").
* secondary eclipse: full duration **20.9 h = 0.0177 in phase either side of φ = 0.45788**;
  partial (annular is not reached).
* T_periastron = BJD_TDB 2458363.7685.

`dBJD` is albireo's BJD_TDB minus the header `ESO DRS BJD`, in seconds. φ_ecl is measured
from primary minimum, φ_per from periastron. v₁, v₂ are the published-orbit velocities
(γ excluded, as albireo's model has none).

| # | file `ADP.` | DATE-OBS | BJD_TDB | dBJD s | BERV km/s | exp s | S/N | programme | DRS | R | φ_ecl | φ_per | v₁ | v₂ | v₁−v₂ | eclipse |
|--:|---|---|---|--:|--:|--:|--:|---|---|--:|--:|--:|--:|--:|--:|---|
| 1 | 2014-09-25T15-33-00.737 | 2011-06-09T07:49:28 | 2455721.83348 | 78.8 | +19.79 | 1200 | 86.4 | 087.C-0012(A) | HARPS_3.5 | 115000 | 0.6097 | 0.5714 | +28.44 | −27.29 | 55.73 | — |
| 2 | 2014-09-25T15-33-43.400 | 2011-06-09T08:45:02 | 2455721.87206 | 78.3 | +19.76 | 1200 | 96.8 | 087.C-0012(A) | HARPS_3.5 | 115000 | 0.6112 | 0.5730 | +28.72 | −27.56 | 56.28 | — |
| 3 | 2014-09-25T15-34-47.070 | 2011-06-10T07:53:45 | 2455722.83652 | 67.1 | +19.80 | 1200 | 111.8 | 087.C-0012(A) | HARPS_3.5 | 115000 | 0.6505 | 0.6122 | +35.29 | −33.87 | 69.16 | — |
| 4 | 2014-09-25T15-36-10.443 | 2011-09-07T03:53:36 | 2455811.67378 | 90.9 | +2.48 | 1200 | 104.1 | 087.C-0012(B) | HARPS_3.5 | 115000 | 0.2628 | 0.2246 | −46.39 | +44.52 | −90.90 | — |
| 5 | 2014-09-25T15-36-30.757 | 2011-09-07T06:44:31 | 2455811.79247 | 90.9 | +2.24 | 1200 | 92.6 | 087.C-0012(B) | HARPS_3.5 | 115000 | 0.2677 | 0.2294 | −45.56 | +43.72 | −89.28 | — |
| 6 | 2014-09-25T15-36-41.123 | 2011-09-07T08:30:26 | 2455811.86602 | 102.3 | +2.09 | 1200 | 102.0 | 087.C-0012(B) | HARPS_3.5 | 115000 | 0.2706 | 0.2324 | −45.03 | +43.22 | −88.25 | — |
| 7 | 2014-09-25T15-36-57.150 | 2011-09-08T03:57:02 | 2455812.67617 | 90.9 | +2.13 | 1200 | 109.4 | 087.C-0012(B) | HARPS_3.5 | 115000 | 0.3036 | 0.2654 | −38.66 | +37.10 | −75.76 | — |
| 8 | 2014-09-25T15-36-57.770 | 2011-09-08T06:11:01 | 2455812.76921 | 90.7 | +1.95 | 1200 | 97.4 | 087.C-0012(B) | HARPS_3.5 | 115000 | 0.3074 | 0.2691 | −37.87 | +36.35 | −74.22 | — |
| 9 | 2014-09-25T15-37-04.203 | 2011-09-08T08:36:34 | 2455812.87029 | 67.0 | +1.75 | 1200 | 116.9 | 087.C-0012(B) | HARPS_3.5 | 115000 | 0.3115 | 0.2733 | −37.00 | +35.52 | −72.52 | — |
| 10 | 2014-09-25T15-37-12.337 | 2011-09-09T04:10:26 | 2455813.68548 | 66.8 | +1.78 | 1200 | 111.8 | 087.C-0012(B) | HARPS_3.5 | 115000 | 0.3446 | 0.3064 | −29.71 | +28.52 | −58.23 | — |
| 11 | 2014-09-25T15-37-21.587 | 2011-09-09T07:25:32 | 2455813.82096 | 78.4 | +1.51 | 1200 | 124.9 | 087.C-0012(B) | HARPS_3.5 | 115000 | 0.3501 | 0.3119 | −28.46 | +27.32 | −55.78 | — |
| 12 | 2014-09-26T16-51-17.683 | 2012-07-29T05:25:12 | 2456137.73457 | 58.5 | +14.03 | 900 | 89.2 | 089.C-0415(A) | HARPS_3.5 | 115000 | 0.5214 | 0.4832 | +10.87 | −10.43 | 21.30 | — |
| 13 | 2014-09-26T16-51-29.857 | 2012-07-29T07:24:24 | 2456137.81805 | 47.1 | +13.91 | 1020 | 88.0 | 089.C-0415(A) | HARPS_3.5 | 115000 | 0.5248 | 0.4865 | +11.60 | −11.13 | 22.72 | — |
| 14 | 2014-09-26T16-52-39.927 | 2012-07-30T05:29:28 | 2456138.73793 | 77.6 | +13.79 | 960 | 91.8 | 089.C-0415(A) | HARPS_3.5 | 115000 | 0.5622 | 0.5240 | +19.38 | −18.60 | 37.97 | — |
| 15 | 2014-09-26T16-53-45.817 | 2012-07-30T08:29:47 | 2456138.86315 | 86.6 | +13.58 | 960 | 76.9 | 089.C-0415(A) | HARPS_3.5 | 115000 | 0.5673 | 0.5290 | +20.40 | −19.58 | 39.97 | — |
| 16 | 2014-09-26T16-54-22.250 | 2012-09-08T06:20:04 | 2456178.77551 | 43.2 | +1.70 | 1200 | **41.3** | 089.C-0415(B) | HARPS_3.5 | 115000 | 0.1902 | 0.1520 | −54.31 | +52.13 | −106.44 | — |
| 17 | 2014-09-26T16-54-32.833 | 2012-09-09T02:44:24 | 2456179.62437 | 77.1 | +1.64 | 960 | 58.5 | 089.C-0415(B) | HARPS_3.5 | 115000 | 0.2247 | 0.1865 | −51.78 | +49.70 | −101.48 | — |
| 18 | 2014-09-26T16-54-45.690 | 2012-09-09T04:59:11 | 2456179.71935 | **7.2** | +1.48 | 1200 | 76.8 | 089.C-0415(B) | HARPS_3.5 | 115000 | 0.2286 | 0.1904 | −51.34 | +49.28 | −100.62 | — |
| 19 | 2014-09-26T16-55-12.347 | 2012-09-09T07:49:16 | 2456179.83746 | 55.3 | +1.23 | 1200 | 106.3 | 089.C-0415(B) | HARPS_3.5 | 115000 | 0.2334 | 0.1952 | −50.76 | +48.72 | −99.48 | — |
| 20 | 2014-10-06T10-05-44.983 | 2013-12-09T00:27:39 | 2456635.52551 | 76.7 | −20.10 | 900 | 72.2 | 092.C-0454(A) | HARPS_3.5 | 115000 | 0.7630 | 0.7247 | +47.23 | −45.33 | 92.56 | — |
| 21 | 2014-10-06T10-07-51.913 | 2013-12-10T00:50:39 | 2456636.54141 | 85.8 | −20.15 | 900 | 101.6 | 092.C-0454(A) | HARPS_3.5 | 115000 | 0.8043 | 0.7661 | +47.65 | −45.74 | 93.39 | — |
| 22 | 2014-10-07T08-34-04.177 | 2014-08-16T07:39:07 | 2456885.82826 | 49.5 | +9.18 | 900 | 77.7 | 093.C-0417(A) | HARPS_3.5 | 115000 | 0.9410 | 0.9028 | +22.02 | −21.13 | 43.15 | 1.15 d before |
| 23 | 2014-10-07T08-35-11.553 | 2014-08-19T05:40:22 | 2456888.74588 | 67.4 | +8.44 | 900 | 98.2 | 093.C-0417(A) | HARPS_3.5 | 115000 | 0.0596 | 0.0214 | −30.28 | +29.06 | −59.34 | 1.15 d after |
| 24 | 2014-12-14T22-55-26.717 | 2014-12-14T04:51:18 | 2457005.70454 | 73.3 | −20.43 | 255 | 60.9 | 094.D-0056(A) | **EGGS_3.5** | **80000** | 0.8155 | 0.7773 | +47.25 | −45.35 | 92.60 | — |
| 25 | 2014-12-15T22-55-22.833 | 2014-12-15T00:14:52 | 2457006.51232 | 74.2 | −20.16 | 220 | 64.1 | 094.D-0056(A) | **EGGS_3.5** | **80000** | 0.8483 | 0.8101 | +44.55 | −42.75 | 87.30 | — |
| 26 | 2015-02-15T23-55-25.697 | 2015-02-15T00:16:15 | 2457068.51386 | 49.6 | −9.21 | 900 | 101.7 | 094.C-0428(A) | HARPS_3.5 | 115000 | 0.3695 | 0.3313 | −24.00 | +23.03 | −47.03 | — |
| 27 | 2015-02-16T23-55-31.017 | 2015-02-16T00:20:29 | 2457069.51678 | 86.0 | −8.89 | 900 | 98.5 | 094.C-0428(A) | HARPS_3.5 | 115000 | 0.4103 | 0.3721 | −14.44 | +13.86 | −28.30 | 0.73 d before sec. |
| 28 | 2015-11-07T02-00-47.457 | 2015-11-06T02:07:03 | 2457332.59662 | 68.8 | −15.82 | 900 | 121.5 | 096.C-0417(A) | HARPS_3.8 | 115000 | 0.1079 | 0.0696 | −46.17 | +44.31 | −90.49 | — |
| 29 | 2016-09-07T01-01-55.859 | 2016-09-06T05:45:28 | 2457637.74977 | 41.5 | +2.42 | 900 | 111.6 | 097.C-0571(B) | HARPS_3.8 | 115000 | 0.5162 | 0.4780 | +9.76 | −9.36 | 19.12 | 1.00 d after sec. |
| 30 | 2016-09-12T07-26-55.192 | 2016-09-09T06:04:03 | 2457640.76269 | 59.2 | +1.37 | 900 | 108.4 | 097.C-0571(B) | HARPS_3.8 | 115000 | 0.6388 | 0.6005 | +33.43 | −32.08 | 65.51 | — |
| 31 | 2016-11-16T01-01-50.126 | 2016-11-15T01:36:01 | 2457707.57452 | 68.6 | −17.70 | 900 | 104.5 | 098.C-0292(A) | HARPS_3.8 | 115000 | 0.3555 | 0.3173 | −27.23 | +26.13 | −53.36 | — |
| 32 | 2016-11-17T01-01-44.673 | 2016-11-16T02:59:04 | 2457708.63213 | 59.3 | −17.98 | 900 | 129.2 | 098.C-0292(A) | HARPS_3.8 | 115000 | 0.3985 | 0.3603 | −17.20 | +16.51 | −33.72 | 1.03 d before sec. |
| 33 | 2017-06-11T01-04-35.219 | 2017-06-10T10:31:13 | 2457914.94133 | 69.7 | +19.67 | 400 | 64.7 | 099.D-0380(A) | **EGGS_3.8** | **80000** | 0.7876 | 0.7494 | +47.82 | −45.90 | 93.72 | — |
| 34 | 2017-06-12T01-03-34.222 | 2017-06-11T10:43:17 | 2457915.94921 | 70.2 | +19.67 | 300 | 70.1 | 099.D-0380(A) | **EGGS_3.8** | **80000** | 0.8286 | 0.7904 | +46.45 | −44.59 | 91.04 | — |
| 35 | 2017-06-14T07-30-18.279 | 2017-06-12T10:32:41 | 2457916.94191 | 70.2 | +19.68 | 300 | **50.0** | 099.D-0380(A) | **EGGS_3.8** | **80000** | 0.8690 | 0.8308 | +41.54 | −39.87 | 81.41 | — |
| 36 | 2017-12-12T01-02-31.499 | 2017-12-11T01:22:50 | 2458098.56024 | 69.5 | −20.23 | 300 | 85.6 | 0100.D-0339(B) | **EGGS_3.8** | **80000** | 0.2541 | 0.2159 | −47.81 | +45.89 | −93.70 | — |

### What the table shows

* **No epoch is in eclipse, and none is within 0.02 in phase of a contact point.** The
  closest approaches are #22 and #23, 1.15 d (0.047 in phase) outside fourth/first contact
  of the primary eclipse, and #27, 0.73 d (0.030 in phase) outside the secondary. The
  observers avoided the eclipses, as one does for radial-velocity work. **Rossiter–McLaughlin
  and the in-eclipse light-fraction change are excluded outright.**
* **Six epochs are HARPS EGGS-mode**, `HIERARCH ESO INS HEFS ST = T`, `SPEC_RES = 80000`,
  `DPR TYPE = 'STAR,DARK,NONE'` (no simultaneous ThAr). All 36 files carry
  `INSTRUME = 'HARPS'`, so `Dataset.instruments` returns one key and the benchmark's single
  R = 115 000 LSF is applied to all of them. The reader *did* read the right value into
  `RawSpectrum.resolving_power`; the benchmark overrides it.
* **Five of those six sit at φ = 0.79–0.87**, the positive extremum of the primary's
  velocity curve (v₁ = +41.5 to +47.8), where only two full-resolution epochs (#20, #21)
  also lie. The high-|v₁| side of the curve is dominated by the mis-declared spectra.
* Timing: albireo's BJD_TDB exceeds the header `ESO DRS BJD` by **+69.3 s on average**
  (7.2–102.3 s, rms 17.8 s). The mean is the TDB−UTC offset (66–69 s over 2011–2017), which
  albireo applies and the HARPS DRS does not, so **albireo's time is the correct one**. The
  ±18 s scatter maps to ≤0.003 km/s of velocity (max \|dv₁/dt\| ≈ 12 km/s d⁻¹). Negligible.
* 20 distinct nights, ~20 distinct phases: #1–2, #4–6, #7–9, #12–13, #14–15, #17–19 are
  same-night groups spanning ≤0.005 in phase. The effective number of independent phases is
  about 20, not 36.
* S/N 41.3–129.2. Only #16 (41.3) and #35 (50.0) are below 58.
* Detector gap: HARPS's inter-CCD gap is 5304.67–5337.61 Å (measured, 3295 zero pixels).
  Window 1 with its 3 Å pad (5147–5253 Å) contains **zero** non-positive pixels.
  Window 2 with its pad starts at 5337.0 Å and therefore clips the last **~33 px per epoch**
  of the gap (1192 non-positive pixels over 36 epochs, longest run 82). `mask_flux_gaps`
  catches runs ≥ 8, so this is handled, but the second window is 0.6 Å inside the gap by
  construction and would be cleaner starting at 5341 Å.

---

## 3. The literature

**Naming convention, identical in every modern source and in the benchmark:** star 1 /
"primary" is the **hotter, smaller, *less* massive F7 V** (Teff 6310 K, M = 1.194 M☉,
r = 0.0377), star 2 / "secondary" is the **cooler, larger, *more* massive K0 IV subgiant**
(5010 K, 1.244 M☉, r = 0.0613). Because star 2 is the more massive, K₁ > K₂ and
q = K₁/K₂ = M₂/M₁ > 1. **At primary minimum star 1 is the one eclipsed** — the K subgiant
passes in front and the eclipse is *total*. So T₀ is the *superior* conjunction of star 1.

| source | instrument / data | RV method | K₁ km/s | K₂ km/s | q = K₁/K₂ | e | ω ° | γ km/s | P d |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| Andersen et al. 1988, A&A 196, 128 | ESO CAT/CES + photographic | photoelectric scanner cross-correlation | *not retrieved* (scanned PDF, no text layer) | | | | | | |
| Hełminiak et al. 2009, MNRAS 400, 969 | échelle | TODCOR | *not retrieved this session*; Gallenne's independent refit of these data agrees with theirs at **<0.7 σ** | | | | | | |
| Sybilski et al. 2018, MNRAS 478, 1942 | échelle | TODCOR + BF | *not retrieved this session*; Gallenne's refit agrees at **<0.5 σ** | | | | | | |
| **Gallenne et al. 2019, A&A 632, A31** | **33 of these same 36 HARPS spectra** (30 HAM + 3 EGGS), 3900–6900 Å, jointly with VLTI/PIONIER astrometry | **Broadening Function, RaveSpan** | **51.166 ± 0.008** | **49.118 ± 0.007** | **1.04169 ± 0.00022** | 0.1872 ± 0.0001 | 110.36 ± 0.03 | −2.111 ± 0.004 | 24.59215 ± 0.00002 |
| **Maxted et al. 2020, MNRAS 498, 332** (the benchmark's reference) | weighted mean of Hełminiak 2009 + Sybilski 2018 + Gallenne 2019; TESS light curve | — | **51.164 ± 0.007** | **49.106 ± 0.010** | **1.04191 ± 0.00026** | 0.1875 ± 0.0009 | 110.34 ± 0.11 | — | 24.5924 (from Kirkby-Kent) |
| Kirkby-Kent et al. 2016, A&A 591, A124 | WASP photometry + published RVs | — | (adopts published) | | | 0.1821 ± 0.0051 | | | **24.592483** |
| Miller, Maxted & Smalley 2020 | Teff paper | — | — | — | — | — | — | — | — |
| **albireo, 5150–5250 Å** | these 36 spectra | joint marginal disentangling | **50.452** | **49.495** | **1.01934** | 0.1879 | (fitted) | ≡ 0 | (fitted) |
| **albireo, 5340–5440 Å** | these 36 spectra | joint marginal disentangling | **50.440** | **49.479** | **1.01942** | — | — | ≡ 0 | — |

**Spread between literature sources.** Gallenne's q and Maxted's adopted q differ by
0.00022, i.e. **0.021 %**, which is exactly 1σ. Maxted et al.: *"The semi-amplitudes of the
spectroscopic orbits derived from spectra obtained with modern échelle spectrographs are
consistent to within 0.1 %"*, and *"the agreement between the parameters derived from these
three independent studies is extraordinarily good."* Cross-check: Maxted's masses give
M₂/M₁ = 1.2438/1.1938 = 1.04188, Gallenne's give 1.2438/1.1941 = 1.04162 — both reproduce
their own q. **albireo's q is offset by 2.17 %, a hundred times the inter-source spread.**

**The decisive structural fact.** Gallenne et al. measured their radial velocities from
**these very spectra**. Maxted's adopted K comes largely from that measurement. Therefore
*any* property of the data — the barycentric correction, the wavelength solution, the
mid-exposure times, the EGGS zero point, telluric absorption, the detector gap — affects
both analyses identically and **cannot** explain a difference between them. And because
Hełminiak's and Sybilski's *different* datasets agree with Gallenne to <0.7σ, an
instrumental error common to the HARPS archive is excluded too.

**Two further literature facts that matter.** Gallenne et al. report that for AI Phe
*"variations in the systemic velocity seem to indicate that there is a possible wider
component in the system"*, and fit a third body with **P ≈ 109 yr, e ≈ 0.8**. Maxted et al.
independently measure **third light of ≈ 0.005–0.010 in the TESS band**. Both are relevant
below.

---

## 4. Diagnosis

### 4.1 The exact decomposition of the bias

albireo's model is v₁ = K₁·u(θ,t), v₂ = −K₂·u(θ,t), with u = cos(ν+ω) + e cos ω, θ = (P,
t_conj, e, ω) free, and a free additive constant per component (γ is degenerate, D14).
Split the observed error into the two irreducible directions:

* a **component-common** velocity error, Δ(t) applied to both stars, moves K₁ and K₂
  *oppositely*: ΔK₁ = +β, ΔK₂ = −β with β = ⟨Δ, ũ⟩/⟨ũ,ũ⟩ after projecting out everything
  the fit can absorb;
* a **component-antisymmetric** velocity error moves them *together*: ΔK₁ = ΔK₂ = s.

From K₁ = 50.452, K₂ = 49.495 against 51.164, 49.106:

```
beta (common)       = (dK1 - dK2)/2 = -0.5505 km/s
s    (antisymmetric)= (dK1 + dK2)/2 = -0.1615 km/s      (K1+K2: -0.32 %)
```

**Only β moves the mass ratio**, and it accounts for all of it:
2β/K̄ = 2(−0.5505)/50.135 = **−2.20 %** against the observed **−2.17 %**. The symmetric part
leaves q untouched.

Re-expressed as cross-talk between the recovered spectra, d̂₁ = s₁ + a·s₂ and d̂₂ = s₂ + b·s₁
give ΔK₁ = (ℓ₂b/ℓ₁)(K₁+K₂) and ΔK₂ = (ℓ₁a/ℓ₂)(K₁+K₂), so the observed pair implies

```
a = +0.0033   (d_hat_1 carries +0.33 % of the secondary's spectrum)
b = -0.0085   (d_hat_2 carries -0.85 % of the primary's spectrum)
```

Opposite signs, weighted by ℓ₁/ℓ₂: **that is the low-frequency exchange (near-null)
direction of the design, `docs/math.md` §5.1, the one the record itself says albireo's
smoothness prior is the only thing pinning.** The record's sentence *"the direction expected
when two similar stars' line signals are partly confused"* is right; what this adds is that
the confusion is *antisymmetric*, not symmetric, that it is 0.3–0.9 % in amplitude, and that
symmetric confusion would have produced a different (and much smaller) signature.

### 4.2 The ephemeris, the conjunction convention and the component naming, checked against the spectra

The autocorrelation of an SB2 spectrum has a side lobe at |v₁ − v₂|, which needs no
template, no light ratio and no disentangling. Measuring it on all 36 epochs in both windows
(`acf.py`):

| window | best-fit scale on (K₁+K₂) | implied K₁+K₂ | rms about the fit |
|---|--:|--:|--:|
| 5150–5250 Å | 0.99870 | 100.140 km/s | 0.88 km/s |
| 5340–5440 Å | 1.00213 | 100.484 km/s | 0.81 km/s |
| | | published 100.270; albireo 99.947 | |

The published u(t) tracks the measured splitting over the whole 19–107 km/s range with no
phase-dependent trend, and the two windows straddle the published K₁+K₂ at ±0.2 %.
**The ephemeris, e, ω and (K₁+K₂) are confirmed directly from the spectra.**

The same test settles the conjunction convention, which the reported e cannot (§1):

| convention | scale | rms | max\|residual\| |
|---|--:|--:|--:|
| ν(t_conj) + ω = +π/2 → component 1 **behind** (eclipsed) at T₀ — what `t_peri_from_t_conj` computes | 0.99870 | **0.88 km/s** | 1.95 |
| ν(t_conj) + ω = −π/2 → component 1 in front at T₀ | 0.90690 | 23.73 km/s | 53.98 |

`aiphe_bench.py` passes ω = 110.30° and T₀ = the *primary* minimum, at which the F7 V
primary is the star eclipsed. That is the convention the code implements and it is the right
one. **Documentation defect, not a bug:** `albireo/kepler.py:119-125` describes
`nu(t_conj) + omega = pi/2` as *"the inferior conjunction of the component whose omega is
given (… the time at which that component passes in front)"*. With
v = K[cos(ν+ω) + e cos ω] and z = r sin(ν+ω) sin i measured away from the observer, ν+ω = π/2
is the *superior* conjunction — the component is at its farthest and is the one **eclipsed**.
The code is correct; the sentence inverts it, and a user following the docstring on an
eclipsing system would put t_conj half an orbit out.

Component naming was checked by median-stacking all 36 spectra in each component's predicted
rest frame (`naming.py`). The component-2 stack shows the deeper Mg I b 5172.64/5183.56 pair
(depths 0.544/0.577 against 0.488 for component 1) and a crowded Fe I forest at 5166–5172
that the component-1 stack does not resolve — the cool-subgiant signature. **albireo's
component 2 is the K0 IV, matching the literature's secondary, which carries K₂.** No swap.

### 4.3 Ranked candidates

Ordered by how much of the −0.5505 km/s common term each can supply. "Amplitude needed" is
the size of the perturbation required to produce the whole bias, computed by regressing each
candidate on u after projecting out {constant, ∂u/∂P, ∂u/∂t_conj, ∂u/∂e, ∂u/∂ω} — everything
the fit is free to absorb (`projection.py`).

---

**1. Exchange of line signal along the disentangling's low-frequency near-null direction,
driven by an unmodelled stationary spectral component: THIRD LIGHT.** *Not excluded; the
leading candidate.*

Maxted et al. measure ℓ₃ ≈ 0.005–0.010 in the TESS band and Gallenne et al. find the third
body (P ≈ 109 yr, e ≈ 0.8) that produces it. albireo's model has two components and a
telluric slot that this run does not use, so a third stellar spectrum at a near-constant
velocity has nowhere to go except into d̂₁ and d̂₂ — and the cheapest place to put it is
exactly the low-k exchange direction, because a stationary component is the k = 0 null mode
seen from the other side (all three codes share that null; `benchmarks.md` says so).

*Testable predictions.* (a) A stationary contaminant of light fraction f pulls both
components' apparent velocities toward it, which is a **symmetric** shrink K_i → (1−f)K_i:
f = 0.32 % is exactly the observed symmetric part, and ℓ₃ ≈ 0.3 % at 5200 Å (a cool
companion, fainter in the blue than in the TESS band) is the right order. That half of the
prediction already matches. (b) The antisymmetric part follows if the contaminant is not
perfectly stationary or if the prior resolves the resulting near-degeneracy asymmetrically.
(c) **The test: refit with a third component held at a fixed velocity** (albireo already
supports a stationary component — that is the telluric slot's geometry) **with ℓ₃ swept over
0–1.5 %, and watch q.** If q climbs toward 1.042 as ℓ₃ approaches the photometric value, it
is confirmed. If q is flat in ℓ₃, it is refuted and the exchange mode is being pushed by
something else. This is the single most informative unrun experiment.

---

**2. One LSF for two instrument modes.** *Not excluded; a concrete, confirmed
misspecification.*

Six epochs are EGGS-mode at R = 80 000 (σ_LSF = 1.591 km/s) and are modelled at R = 115 000
(σ_LSF = 1.107 km/s) — 44 % too narrow — because all 36 files report `INSTRUME = 'HARPS'`
and the benchmark builds `lsf_sigma_v` by instrument key. Five of the six sit at φ =
0.79–0.87, the positive extremum of v₁, where only two correctly-modelled epochs also lie.

*Testable prediction.* A symmetric kernel error produces no first-order velocity shift, so
on its own this should not move K much: modelled as a pure velocity offset ε on those six
epochs, β/ε = 0.256, so ε = −2.15 km/s would be needed and no plausible EGGS zero point is
within two orders of magnitude of that. What it *can* do is push the exchange mode, because
it perturbs the model's line *shape* at a set of epochs concentrated on one side of the
velocity curve. **The test: drop the six EGGS epochs (which also reproduces Gallenne's 30 +
3 selection more closely), or give them their own instrument key and their own σ_LSF, and
re-fit.** A shift in q of ≳0.5 % implicates it; ≲0.1 % clears it. Independently of the
outcome this should be fixed: the reader already knows the right R and the benchmark
discards it. A one-line guard — warn when a dataset mixes resolving powers under one
instrument key — would have caught it.

---

**3. The assumed light fractions, on the specific question of whether the *fitted* K moves
with them.** *Untested by the record.*

The record's exclusion — "sweeping ℓ₂ from 0.38 to 0.53 moves the likelihood difference
between albireo's K and the published K by 9 nats out of 53,306" — is a statement about the
*likelihood at two fixed K's*, not about where the optimum in K goes. There is a stronger
argument for the same conclusion: ℓ_i multiplies d_i, so the product ℓ_i d_i is what the
data determine, and the velocities are set by line *positions*, not amplitudes. ℓ has **no
first-order leverage on K at all**; its only route is second-order, through the prior scale
and through the geometry of the exchange mode.

*Testable prediction.* **Refit K₁, K₂ at ℓ₂ ∈ {0.38, 0.42, 0.46, 0.50, 0.53} and tabulate
q.** If q is flat, ℓ is cleared for a much better reason than the 9 nats, and the record
should say so. If q moves by ~1 % across that sweep, the exchange mode's geometry is what is
carrying the bias and item 1 becomes near-certain.

---

**4. Under-regularisation: the likelihood/prior balance is wrong by 7×.** *Not excluded; an
amplifier rather than a source.*

HARPS ships no error array, so the weights are albireo's own local-scatter estimate, and the
record's residual z-RMS is 2.64 — the assumed inverse variances are 2.64² ≈ 7 times too
confident. The smoothness prior is the only thing pinning the near-null exchange mode, so it
is being under-weighted sevenfold in exactly the direction where the bias lives.

*Testable prediction.* Rescale the ivar by 1/2.64² (or, equivalently, raise the prior
strength by 7×) and re-fit. If q moves toward 1.042, the answer is that the exchange mode
was under-pinned. **Important counter-argument, which is why this is ranked as amplifier and
not source:** the two disjoint windows agree in q to 0.01 %. Under-regularisation lets the
fit follow *noise* along the near-null direction, and the photon noise in two disjoint 100 Å
windows is independent, so noise-following would have produced a scatter of order the bias
itself, not 0.01 %. The driver must be a **deterministic** model error common to both
windows; items 1, 2 and 3 are exactly that, and item 4 multiplies whichever of them is at
work.

---

**5. A systemic-velocity drift from the third body.** *Excluded at the amplitude required;
contributes ≲0.1 %.*

The third body is real (Gallenne: P ≈ 109 yr, e ≈ 0.8), and a γ drift is a component-common
velocity error, so it acts in exactly the right direction. But the projection gives
β = 0.688 per (km/s per year), so **−0.80 km/s per year** would be needed — 5.2 km/s of drift
across the 6.51-year baseline. Gallenne et al. fit a **single** γ = −2.111 ± 0.004 km/s to
these same spectra; a 5 km/s drift would leave ±2.6 km/s residuals, ~650× their quoted error.
A plausible drift for a 109-yr companion away from periastron (≲0.05 km/s yr⁻¹) gives
β ≈ 0.034 km/s, i.e. **0.07 % on K** — a twentieth of what is needed.

*Testable prediction.* Fit a per-epoch-group velocity offset or a linear γ(t) alongside K.
If the recovered slope is ≪0.8 km/s yr⁻¹ (it will be), this is closed.

---

**6. A component-naming or conjunction-convention mismatch.** *Excluded from the data,
§4.2.* The alternative convention fails the template-free splitting test by 27×
(23.7 vs 0.88 km/s rms), and the median stacks identify component 2 as the cool subgiant. A
*documentation* error exists in `kepler.py` but does not affect this run.

---

**7. Ephemeris or period error over the HARPS baseline.** *Excluded for the K fit.*

The baseline is 2376.7 d = 96.6 cycles, and T₀ is 107.4 cycles after the first epoch.
Kirkby-Kent's P = 24.592483 d against the benchmark's rounded `P_PUB = 24.5924` differs by
8.3e−5 d → **0.0089 d (0.00036 in phase) of drift** back to the first epoch → ≤0.107 km/s of
velocity error at the steepest part of the curve. In the K fit `period` and `t_conj` are both
free (±0.01 d, ±0.05 d), so this is absorbed; and §4.2 confirms the shape function directly
from the spectra. **But it is *not* absorbed in the fixed-velocity runs**
(`aiphe_labels_bench.py` and the d̂ used for the label fit and for the shift-and-add
comparison), which use `P_PUB = 24.5924` exactly: those inherit up to 0.1 km/s of
epoch-dependent velocity error. Cheap fix: carry the full 24.592483.
(For scale, using Gallenne's RV-only P = 24.59215 instead would give 0.036 d of drift and
0.43 km/s of error — the photometric period is the one to use.)

---

**8. Pixel locking of the shift operator.** *Excluded quantitatively, by two independent
arguments.*

The model grid is dv = 0.8 km/s and σ_LSF = 1.107 km/s, i.e. **1.38 px per LSF σ**, below the
repo's own ≥3 rule and below the level at which `todcor` warns. Interpolating
`benchmarks.md`'s own measured table (0.0293 px at 1.0 px/σ, 0.0153 at 2.0) gives ≈0.021 px
= **0.017 km/s = 0.033 % of K₁**, forty times too small. Second argument, and the stronger
one: that table's σ is the *LSF* width because it measures correlation against a fixed
narrow template, whereas the operator here shifts d̂, whose lines carry the full stellar
profile. The autocorrelation half-width measured on these spectra is ≥9 km/s, so what is
being translated is ≳5 model pixels wide and the interpolation error is smaller still.
**The grid is not the problem** — though the 1.38 figure should be recorded so that nobody
re-raises it.

---

**9. Cross-talk near conjunction / blending.** *Excluded as a *phase-localised* effect.*

The minimum separation over the 36 epochs is **19.1 km/s** (#29), against an observed line
half-width of ~9 km/s; there are no near-conjunction epochs and no epoch is even close to
the blended regime. The repo's own two-dimensional benchmark shows albireo unbiased down to
0 km/s separation anyway. What remains is not *blending* but the global exchange
near-degeneracy of item 1, which is present at every separation.

---

**10. Telluric contamination.** *Excluded.* The nearest telluric absorption to either window
is the weak H₂O band beginning near 5880 Å; the O₂ γ band starts at 6270 Å. Both windows are
telluric-free, no telluric component was declared in `build_problem`, and in any case
Gallenne used 3900–6900 Å and would have been affected more, not less.

---

**11. Barycentric-correction convention.** *Excluded.* All 36 files are `s1d` products with
`SPECSYS = 'BARYCENT'`; albireo reads that, does **not** re-apply the shift, and keeps
`ESO DRS BERV` only to place the telluric component. Any BERV error would show up in
Gallenne's velocities from the same files. Recorded for completeness: BERV is **amplified**
by this epoch set — regressing BERV on u gives β = 3.34 per unit, so a 1 km/s BERV error
would become a 3.3 km/s K error, and a 16.5 % BERV error would reproduce the whole bias.
That is a useful warning about this design, not a diagnosis. The relativistic composition
term albireo would incur if it did compose the two velocities is
K₁·BERV_max/c = **0.0035 km/s**.

---

**12. Gravitational redshift and convective blueshift.** *Excluded by construction, and this
is worth stating explicitly.* Both are **constant in each star's own frame**, so each adds a
constant to that component's velocity. albireo's model has a free additive constant per
component (the translation of d̂_i that makes γ degenerate, D14). The projection confirms it
numerically: a constant perturbation gives **β = −0.0000**. Their difference between an
F7 V and a K0 IV (of order 0.3–0.6 km/s, and in the right ballpark to be tempting) affects
**γ, and the *difference* between the two components' γ, and nothing else**. It cannot move
K₁ or K₂. Gallenne et al. applied no such correction either, which is why their γ is a
single number.

---

**13. Light-time effects.** *Excluded.* a₁ sin i = 1.70e7 km → a light-time amplitude of
**56.7 s** across the primary's orbit. The velocity distortion it produces is of order
K₁²/c = **0.0087 km/s** (0.017 %), and the transverse-Doppler term is half that. Both are
sixty times too small. Exposure smearing is smaller still: with T = 1200 s and
\|d²v/dt²\| ≲ 10 km/s d⁻², the T²/24 term is ~3e−5 km/s.

---

**14. A systematic in the published values.** *Excluded, with one caveat worth recording.*
Three studies on partly different data agree to <0.7σ; Gallenne's q and Maxted's adopted q
differ by 0.021 %; both reproduce their own masses. Nothing at the 1 % level is available on
that side. **Caveat:** Gallenne's is not an independent *dataset* — it is these spectra —
so what the agreement establishes is that a broadening-function analysis of these photons,
a TODCOR analysis of other photons, and a TODCOR+BF analysis of yet other photons all give
the same answer, and albireo's disentangling of these photons does not.

---

## 5. What to do next, in order

1. **Add a third, stationary component with ℓ₃ swept over 0–1.5 %** and watch q. Predicted
   to be the answer, and the symmetric −0.32 % already matches ℓ₃ ≈ 0.3 %.
2. **Refit K₁, K₂ at five values of ℓ₂** (0.38–0.53). This is the sweep the record thinks it
   already did; it swept the likelihood, not the optimum.
3. **Give the six EGGS epochs their own instrument key and σ_LSF = 1.591 km/s**, and
   separately re-fit without them. Fix the reader/benchmark so a dataset that mixes
   resolving powers under one `INSTRUME` cannot silently take one LSF.
4. **Rescale the weights so z-RMS = 1** and re-fit; this bounds how much of whatever remains
   is under-regularisation.
5. **Report the fitted `period`, `t_conj` and the *signs* of `secosw`, `sesinw`** in the
   benchmark output, and stop quoting e as a bare `sec² + ses²`, which is blind to a 180°
   error in ω.
6. Small corrections: carry P = 24.592483 rather than 24.5924 in the fixed-velocity runs;
   move window 2 to 5341–5440 Å so it clears the detector gap; fix the `t_peri_from_t_conj`
   docstring, which says "passes in front" where the mathematics says "is eclipsed".

## 6. Two things the record should gain regardless of the outcome

* **The decomposition.** ΔK₁ and ΔK₂ should always be reported as (antisymmetric,
  symmetric) = (−0.550, −0.162) km/s, because those two numbers have different causes and
  only the first one moves the mass ratio. The current framing — "a ~1 % systematic biasing
  the mass ratio 2.2 % toward unity" — conflates a −1.10 % effect and a −0.32 % effect that
  have nothing to do with each other.
* **The two-window agreement is stronger evidence than the record uses it for.** Agreement
  to 0.01 km/s across disjoint windows with independent photon noise proves the driver is
  deterministic. That excludes every noise-driven explanation in one line, and it is the
  fact that promotes third light, the LSF and the light fractions above under-regularisation.
