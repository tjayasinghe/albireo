# BOSZ 2024 and Korg.jl: conventions verified against primary sources

Prepared 2026-09-09 for the albireo Gaia RVS simulator. Every fact is marked
**VERIFIED** (read in the primary source, or measured directly from the archive
files), **LIKELY** (inferred, or from a secondary source), or **UNVERIFIED**.

"Measured" below means: files were downloaded from the MAST HLSP archive and
analysed numerically in this session. No file in any git repository was modified.

---

## 0. Corrections to premises in the request

Four premises in the brief are wrong or incomplete. They are load-bearing, so
they come first.

| Premise as stated | Correction | Status |
|---|---|---|
| BOSZ 2024 = A&A **688, A171** | The article number is **A197**. `2024A&A...688A.197M`, DOI [10.1051/0004-6361/202449306](https://doi.org/10.1051/0004-6361/202449306) | **VERIFIED** |
| "MARCS covers ≤ 8000 K and ATLAS9 above" | True at the edges, but **7500–8000 K is a genuine overlap**: `mp`, `ms` *and* `ap` files all exist there. A naive "pick by Teff" reader gets an arbitrary answer. | **VERIFIED** (measured) |
| Known air/vacuum "convention change" 2017 → 2024 | Real, and in the direction people usually guess wrong: **2017 = vacuum throughout; 2024 = air above 200 nm**, vacuum below. | **VERIFIED** (paper + measured) |
| Sartoretti et al. 2023, A&A 674, A6 = DR3 RVS processing | That paper is *"Gaia DR3: G_RVS photometry from the RVS spectra"*. There is **no** refereed DR3 analogue of Sartoretti 2018; DR3 RVS processing lives in the online DR3 documentation Ch. 6 and in Katz et al. 2023 (A&A 674, A5). | **VERIFIED** |

A fifth, which reverses the usual assumption about which code is more capable:

> **Korg cannot synthesise an A or B star at all.** Its MARCS grid stops at
> 8000 K and `synth` throws `Korg.LazyMultilinearInterpError`. BOSZ reaches
> 16 000 K. For an eclipsing-binary population containing A/late-B components,
> BOSZ is the only one of the two routes that works. **VERIFIED** (§B.5, §A.6).

---

## A. BOSZ 2024

Primary sources:
Mészáros et al. 2024, A&A 688, A197 ([arXiv:2407.10872v2](https://arxiv.org/abs/2407.10872));
MAST doc page <https://archive.stsci.edu/hlsp/bosz>;
HLSP README <https://archive.stsci.edu/hlsps/bosz/bosz2024/hlsp_bosz_bosz_sim_all_all_v1_readme.txt>;
HLSP DOI [10.17909/T95G68](https://dx.doi.org/10.17909/T95G68).

Synthesis code: **Synspec** (via the Synple wrapper), LTE, with 23 updated
molecular line lists. Atmospheres: MARCS (Gustafsson et al. 2008) and
ATLAS-APOGEE ATLAS9 (Mészáros et al. 2012). **VERIFIED**

### A.1 Parameter coverage and steps

All of the following was **measured** by parsing the complete directory listing
of `r20000/m+0.00/` (45 724 files), and cross-checked against the README.

**Teff — VERIFIED**

| Range | Step | Atmosphere |
|---|---|---|
| 2800 – 4000 K | 100 K | MARCS (`mp` + `ms`) |
| 4250 – 7250 K | 250 K | MARCS (`mp` + `ms`) |
| **7500 – 8000 K** | 250 K | **MARCS *and* ATLAS9 — both present** |
| 8250 – 12 000 K | 250 K | ATLAS9 (`ap`) |
| 12 500 – 16 000 K | 500 K | ATLAS9 (`ap`) |

Filename atmosphere tokens, exactly as the README defines them — **VERIFIED**:

- `mp` = MARCS **plane-parallel**
- `ms` = MARCS **spherical**
- `ap` = ATLAS9 **plane-parallel** (there is no `as`; ATLAS9 is plane-parallel only)

**log g — VERIFIED (measured).** Step is always 0.5 dex; the range is a function
of Teff (values below are for `m+0.00`; see the caveat on metallicity dependence
in §A.1.1):

| Teff | log g range |
|---|---|
| 2800 – 4000 K | −0.5 … +5.5 |
| 4250 – 4500 K | −0.5 … +5.0 |
| 4750 – 5250 K | 0.0 … +5.0 |
| 5500 – 5750 K | +0.5 … +5.0 |
| 6000 – 7000 K | +1.0 … +5.0 |
| 7250 – 12 000 K | +2.0 … +5.0 |
| 12 500 – 16 000 K | +3.0 … +5.0 |

**Composition and microturbulence — VERIFIED** (README, paper, and measured):

- `[M/H]`: −2.50 … +0.75, step **0.25** → **14** values (14 directories confirmed)
- `[α/M]`: −0.25 … +0.50, step 0.25 → **4** values (`-0.25, +0.00, +0.25, +0.50`)
- `[C/M]`: −0.75 … +0.50, step 0.25 → **6** values
- microturbulence: **0, 1, 2, 4 km/s** (tokens `v0 v1 v2 v4`)

14 × 4 × 6 = **336 unique compositions**, exactly the number the paper quotes.
**VERIFIED arithmetically.**

**Resolutions — VERIFIED.** Eight products per model:
`r500, r1000, r2000, r5000, r10000, r20000, r50000` and `rorig`
(native, R ≈ 200 000 – 600 000 depending on parameters). Note there is **no
R = 11 500 product**, and `r10000` is *already broader* than R = 11 500
(FWHM 29.98 vs 26.07 km/s), so it cannot be sharpened. See §D.

#### A.1.1 The grid is NOT a complete rectangular tensor — measured

This is not documented anywhere and matters directly for an interpolating
library module. **VERIFIED (measured)**:

| metallicity | (atmos, Teff, log g) nodes | files | missing vs full tensor |
|---|---|---|---|
| `m+0.00` | 492 | 45 724 | **3.19 %** |
| `m-2.50` | 460 | 43 316 | **1.91 %** |
| `m+0.75` | 490 | 44 840 | **4.68 %** |

- Only **415 of 492** nodes at `m+0.00` carry all 96 (α, C, v_mic) combinations.
- Holes concentrate in cool MARCS spherical models at low log g (e.g. `ms t3000 g+0.0` has 24/96; `ms t2800 g-0.5` has 12/96).
- **The node set itself is metallicity-dependent** (492 vs 460 vs 490), so the
  grid is not a clean 5-D lattice.
- **Good news for hot stars:** only 6 incomplete nodes exist above 7000 K and
  **all of them are MARCS**. Every ATLAS9 (`ap`) node inspected is complete.
- The paper's "495 Teff–log g pairs" does not match the 492 measured at solar
  metallicity; treat 495 as nominal. **VERIFIED (discrepancy is real)**

### A.2 File naming convention — VERIFIED

```
bosz2024_<atmos>_<teff>_<logg>_<metal>_<alpha>_<carbon>_<micro>_<insbroad>_<prod>.txt.gz
```

Directory layout is `<insbroad>/<metallicity>/`, e.g. `r20000/m+0.00/`.
Note the `+` must be percent-encoded as `%2B` when building URLs.

Real file names, quoted verbatim from the archive listing (**VERIFIED**):

```
bosz2024_mp_t5000_g+4.5_m+0.00_a+0.00_c+0.00_v2_r20000_resam.txt.gz
bosz2024_ap_t10000_g+4.0_m+0.00_a+0.00_c+0.00_v2_r20000_resam.txt.gz     <- hot star
bosz2024_ap_t16000_g+4.0_m+0.00_a+0.00_c+0.00_v2_r20000_resam.txt.gz     <- hottest
bosz2024_ms_t7750_g+2.0_m+0.00_a+0.00_c+0.00_v2_r20000_resam.txt.gz      <- MARCS in the overlap
bosz2024_ap_t7750_g+2.0_m+0.00_a+0.00_c+0.00_v2_r20000_resam.txt.gz      <- ATLAS9, SAME (Teff, logg)
```

The last two illustrate the 7500–8000 K overlap: identical physical parameters,
two different atmospheres, two different files.

`<prod>` is `resam` (resampled, all `r####`), `noresam` (for `rorig`), or
`lineid` (line identifications, `rorig` and v=0 only; lines with W ≥ 0.1 Å).

A URL template for A/B stars:

```
https://archive.stsci.edu/hlsps/bosz/bosz2024/r20000/m%2B0.00/
  bosz2024_ap_t{Teff}_g{+g.g}_m{+m.mm}_a{+a.aa}_c{+c.cc}_v{v}_r20000_resam.txt.gz
```

### A.3 Wavelength grid, coverage, sampling, medium

**The grid is a separate file, not stored per spectrum.** A `resam` file has
**exactly two columns** and one row per grid point (§A.4). **VERIFIED**

Grids live at `wavelength_grids/bosz2024_wave_r{R}.txt`, one per resolution
(`r500 … r50000`; there is none for `rorig`, which carries its own wavelengths).

**Measured from `bosz2024_wave_r20000.txt`** (4 500 000 bytes, **VERIFIED**):

- **Units are Ångström**, not nm: 500.0000000 → 319 993.11 Å (= 50 nm → 32 µm).
- **300 000 points**, exactly `15 × R`.
- **Uniform in ln λ.** Measured Δln λ = 2.1538227×10⁻⁵; the only deviation
  (4×10⁻³ near 1012 Å) is print-rounding of the 8-significant-digit format.

The paper's Eqs. 3–4 reproduce this exactly (**VERIFIED**, and confirmed
numerically to 7×10⁻⁹):

```math
\Delta\lambda_R = \frac{\ln \lambda_{\max} - \ln \lambda_{\min}}{7500 \cdot R/500}
\qquad
\lambda_{R,i} = e^{\,\ln(500) + \Delta\lambda_R\, i}
```

with $`\lambda_{\min}=500`$ Å, $`\lambda_{\max}=320\,000`$ Å, $`i = 0 \ldots 15R-1`$.
Generating the r50000 grid from this formula and re-measuring the Ca II line
reproduced 8542.087 Å against the air reference 8542.09 Å — an independent
confirmation of both the formula and the medium.

A useful closed form: **samples per resolution element = 15/ln(640) = 2.32145**,
independent of R. This is only modestly above Nyquist; see §D.

**Sampling in the 8460–8700 Å band at r20000 — VERIFIED (measured):**

| quantity | value |
|---|---|
| samples in 8460–8700 Å | **1299** |
| Δλ at 8460 Å | 0.1823 Å |
| Δλ at 8700 Å | 0.1873 Å |
| Δv per sample | **6.4570 km/s** (constant) |
| FWHM at 8580 Å, R = 20 000 | 0.4290 Å |
| FWHM at 8580 Å, R = 11 500 | 0.7461 Å |

**Medium: AIR above 200 nm, vacuum below. VERIFIED twice over.**

The paper states verbatim:

> "All spectra span a range from 50 nm to 32 μm using air wavelengths above
> 200 nm and vacuum wavelength below 200 nm."

Independently **measured** by locating line minima in
`bosz2024_mp_t5000_g+4.5_m+0.00_a+0.00_c+0.00_v2_r20000` and, for the UV, in the
t10000 ATLAS9 model:

| line | air ref (Å) | vacuum ref (Å) | BOSZ minimum | Δ vs air | verdict |
|---|---|---|---|---|---|
| Ca II | 8498.02 | 8500.36 | 8498.082 | +0.06 | AIR |
| Ca II | 8542.09 | 8544.44 | 8542.124 | +0.03 | AIR |
| Ca II | 8662.14 | 8664.52 | 8662.180 | +0.04 | AIR |
| Hα | 6562.80 | 6564.61 | 6562.849 | +0.05 | AIR |
| Na D1 | 5895.92 | 5897.56 | 5895.988 | +0.07 | AIR |
| Mg I b2 | 5172.68 | 5174.13 | 5172.646 | −0.03 | AIR |
| Ca II K | 3933.66 | 3934.78 | 3933.654 | −0.01 | AIR |
| Mg II | 2795.53 | 2796.35 | 2795.513 | −0.02 | AIR |

Every residual is under half a sample; the vacuum hypothesis is off by
1.5–2.4 Å. **The change from BOSZ 2017 is real**: the MAST README states the
2017 ASCII files contain *"the wavelengths (measured in vacuum)"*. **VERIFIED**

The formula BOSZ used for the vacuum→air conversion is **not stated** in the
paper or README — **UNVERIFIED**. In practice the candidate formulae (Edlén
1966, Birch & Downs 1994, Ciddor 1996) differ by ≲ 2×10⁻⁸ in n, i.e. ≲ 6 m/s
at 8580 Å, so the choice is immaterial at RVS precision.

### A.4 Columns, units, and the broadening kernel

**Columns — VERIFIED (README + measured).**

- `rorig` / `noresam` files: **3 columns** — wavelength (Å), Eddington first
  moment H (erg s⁻¹ cm⁻² Å⁻¹), continuum.
- `r####` / `resam` files: **2 columns** — flux (H) and continuum, in that
  order, **no wavelength column**. Row *i* pairs with row *i* of the matching
  `bosz2024_wave_r####.txt`. Measured: the r20000 file has exactly 300 000
  rows of 2 fields, matching the grid line-for-line.

**Flux units and the factor-of-4 trap — VERIFIED.** For 2024:

```math
F = 4\pi H, \qquad H \text{ in erg\,s}^{-1}\text{cm}^{-2}\text{\AA}^{-1}
```

but for the **2017** grid the README says to use `F = π·H_BOSZ`, because
*"the BOSZ models are already four times larger"*. The two releases therefore
differ by a **factor of 4 in the stored surface-brightness scale**. Irrelevant
if you rectify by the continuum column, fatal if you mix releases in absolute flux.

**Normalised flux** is simply column 1 / column 2. The continuum column is
smooth (measured: 0.014 % change across one R = 11 500 FWHM), so it does not
matter whether you broaden before or after dividing — the two orders agree to
better than 0.1 %. **VERIFIED (measured)**

**Is the broadening a Gaussian with FWHM = λ/R? — VERIFIED by measurement.**

The paper does **not** state the kernel (**UNVERIFIED** from documentation), so
it was measured by forward modelling: take the r50000 product of the *same*
model, convolve it with a trial Gaussian of constant width in velocity,
resample onto the r20000 grid, and fit the width against the actual r20000 file
over 8400–8800 Å.

| hypothesis | predicted extra FWHM | result |
|---|---|---|
| Gaussian, **FWHM = λ/R** | 13.74 km/s | **best fit 13.35 km/s (2.8 % agreement)**, residual rms 0.00036 |
| Gaussian, σ = λ/R | 32.35 km/s | residual rms 0.0242 — **rejected**, 67× worse |

For scale, the line-depth rms of the spectrum itself is 0.0859. The residual at
the best fit is 0.4 % of that. **Conclusion: BOSZ instrumental broadening is a
Gaussian of constant resolving power, FWHM = λ/R, applied before resampling
onto the log-λ grid.** The residual 2.8 % is consistent with the interpolation
and the 2.32 samples/FWHM of the r20000 grid.

Rotational broadening is **not** applied to any BOSZ product, and is not
mentioned in the paper. **VERIFIED** (2017 filenames carried an `rt0` token;
2024 has no rotation token at all.)

### A.5 Caveats in the 8460–8700 Å band

**The 2025 recomputation did *not* touch this band. VERIFIED.** The README
(updated 2025-09-25) describes both 2024 bugs:

> 1. OH⁺ partition function — *"stronger than expected molecular absorption
>    lines between 300 and 450 nm, and 1 and 8 microns for models with
>    temperatures below 6500K. Higher temperature models are unaffected."*
> 2. Eight-level H atom — *"a lack of higher-level H lines above 5.8 microns …
>    the 50 nm - 5.8 micron region is not affected by this issue."*

Neither window overlaps 846–870 nm. **The RVS band was correct even in the
original 2024 release.** Nonetheless the whole grid was recomputed:

> *"The team have successfully recomputed all models by fixing both issues
> reported on 2024 October 25. … Users are strongly advised to download the
> new models and replace the old ones!"*

**Are the served files the recomputed ones? LIKELY yes.** Measured
`Last-Modified` headers: the two spectra fetched are dated **2025-04-03** and
**2025-05-22**, i.e. after the 2024-10-25 bug discovery, consistent with a
staged recomputation announced complete on 2025-09-25. Notably the
**wavelength grid file is dated 2023-09-17 and was never regenerated** — the
grid is stable across the recomputation, which is convenient for caching.

**Ca II triplet: LTE only, no NLTE anywhere. VERIFIED.** The paper says all
spectra are synthesised *"under the assumption of local thermodynamic
equilibrium (LTE)"* and that models above 16 000 K are excluded *"due to the
lack of the necessary NLTE effects in ATLAS9."* There is **no** discussion of
the Ca II triplet specifically, and none of Gaia RVS.

The size of that omission is quantified elsewhere (§C.3): 3D non-LTE Ca II
triplet **radial-velocity** corrections reach ±0.25 km/s typically and ±1 km/s
in outliers at RVS resolution (Lagae, Amarsi & Lind 2025). This is a floor for
*any* LTE template — BOSZ, Korg, or Gaia's own.

**Hydrogen / Paschen series: present and important. VERIFIED (measured).**
Stark broadening is from Tremblay & Bergeron (2009). The Paschen series
(limit 8204 Å) runs straight through the RVS band and, in the ATLAS9 t10000
model, is deep:

| line | air ref (Å) | measured depth |
|---|---|---|
| P16 | 8502.48 | 0.10 |
| P14 | 8598.39 | 0.21 |
| P12 | 8750.47 | 0.33 |

P15 (8545.4 Å) and P13 (8665.0 Å) are **blended with Ca II 8542 and 8662** — the
minimum-finder locks onto the Ca II core rather than the Paschen core. In an
A star the RVS band is a superposition of broad Paschen wings and the Ca II
triplet: measured Ca II depths are 0.34–0.47 at 10 000 K versus 0.66–0.72 at
5000 K. Mean normalised flux across 8460–8700 Å is 0.914 (hot) and 0.945 (cool).

**Paschen jump caution:** the paper reports BOSZ 2017 → 2024 flux changes of
**10–15 % at the Paschen jump (850–900 nm) above 10 000 K**. That band overlaps
the RVS window directly, so absolute-flux work with hot stars is version-sensitive.
**VERIFIED**

**Molecular opacities:** 23 new molecular line lists were added, mainly to
enable cooler models. No missing-molecule caveat specific to 846–870 nm is
documented. **UNVERIFIED** (absence of a documented caveat, not a positive claim).

### A.6 Do 7000–15 000 K models exist at log g 3.5–4.5?

**Yes — completely, with no gaps. VERIFIED (measured).** For
`m+0.00, a+0.00, c+0.00, v2, r20000`, log g ∈ {3.5, 4.0, 4.5} exists at
**every** Teff node from 7000 K to 16 000 K:

```
T=7000   +1.0 +1.5 +2.0 +2.5 +3.0 +3.5 +4.0 +4.5 +5.0
T=7250   +2.0 +2.5 +3.0 +3.5 +4.0 +4.5 +5.0
T=7500   +2.0 +2.5 +3.0 +3.5 +4.0 +4.5 +5.0
T=8000 … T=12000   +2.0 … +5.0   (step 250 K)
T=12500 … T=16000  +3.0 … +5.0   (step 500 K)
```

For an eclipsing-binary population of A and late-B components this is fully
covered, at 250 K resolution up to 12 000 K and 500 K above. All such nodes are
ATLAS9 (`ap`) and, unlike the cool MARCS corner, **complete**.

---

## B. Korg.jl

Primary sources: source on `main` via `raw.githubusercontent.com`; docs
<https://ajwheeler.github.io/Korg.jl/stable/>;
Wheeler, Casey & Abruzzo 2023, AJ 165, 68 (`2023AJ....165...68W`,
DOI [10.3847/1538-3881/acaaad](https://doi.org/10.3847/1538-3881/acaaad));
Wheeler et al. 2024 ([arXiv:2310.19823](https://arxiv.org/abs/2310.19823)).

Korg is **post-v1.0**; several conventions changed at v1.0 and are flagged below.

### B.1 `Korg.synth` — signature, returns, rectification, medium, step

**VERIFIED** from `src/synth.jl`:

```julia
function synth(; Teff=nothing, logg=nothing, M_H=0.0, alpha_H=M_H,
               linelist=get_VALD_solar_linelist(),
               wavelengths=(5000, 6000), rectify=true, R=Inf, vsini=0, vmic=1.0,
               synthesize_kwargs=Dict(), format_A_X_kwargs=Dict(),
               abundances...)
```

| kwarg | default | note |
|---|---|---|
| `Teff`, `logg` | `nothing` | **required**; `ArgumentError` if omitted (v1.0 change) |
| `M_H` | `0.0` | |
| `alpha_H` | `M_H` | i.e. [α/Fe] = 0 by default |
| `linelist` | `get_VALD_solar_linelist()` | its own docstring: *"intended to be used for quick tests only"* |
| `wavelengths` | `(5000, 6000)` | **Å** |
| `rectify` | **`true`** | |
| `R` | `Inf` | |
| `vsini` | `0` | km/s |
| `vmic` | `1.0` | km/s |

**Returns a 3-tuple** `(wavelengths, flux, cntm)`. **VERIFIED**

**Rectified by default: yes.** `rectify=true` performs `flux ./ cntm`. With
`rectify=false` the units are **erg s⁻¹ cm⁻⁴ Å⁻¹** (v1.0 change from
`erg/s/cm^5`); the FAQ explains the extra cm⁻² as Korg not assuming a stellar
projected area — multiply by $`R_\star^2`$. **VERIFIED**

**Two subtleties worth flagging (VERIFIED from the function body):**

1. **Order of operations is `apply_LSF` *then* `apply_rotation`** — physically
   backwards. Both kernels are constant-width *in velocity*, so on a log-λ grid
   they would commute exactly; Korg works on a **linear** λ grid, where the
   pixel widths vary, so they do not commute exactly. The error is small over a
   240 Å band (±1.4 % kernel-width variation) but it is not zero.
2. **The returned `cntm` is the raw, unbroadened continuum** — not LSF-convolved,
   not rotationally broadened, and returned unchanged regardless of `rectify`.
   So `flux .* cntm` does **not** recover the broadened absolute spectrum.
   Note this also means Korg rectifies *before* broadening, whereas BOSZ ships
   broadened flux and continuum separately. As shown in §A.4 the difference is
   < 0.1 % because the continuum is smooth.

**Wavelength medium: VACUUM, always**, for input and output.
`SynthesisResult.wavelengths` docstring: *"The vector of vacuum wavelengths
(in Å)."* Wheeler+2023: Korg *"uses vacuum wavelengths internally, and will
automatically convert air-wavelength VALD linelists to vacuum."* **VERIFIED**

**`air_wavelengths` is a v1.0 trap. VERIFIED.** The Upgrading page states it
*"is no longer an allowed keyword argument for `Korg.synthesize`"*. It still
exists on the `Korg.Wavelengths` constructor (default `false`), with
`wavelength_conversion_warn_threshold` (default `1e-4` Å); it fits an evenly
spaced *vacuum* range through converted endpoints and throws if the deviation
exceeds the threshold. To use it you must build the `Wavelengths` object
yourself and pass it in.

**Default step: 0.01 Å. VERIFIED** — the optional third tuple element,
`(λstart, λstop)` ≡ `(λstart, λstop, 0.01)`. `wavelengths` also accepts a
**vector of tuples** for disjoint windows, e.g. `[(5000,5500), (6000,6500,0.03)]`.
Units are Å (legacy cm-inference for λ < 1 exists but the docs say do not rely
on it). v1.0 breaking change: wavelength specification is now a **single**
argument, not two positionals.

### B.2 How `R` and `vsini` are applied

**LSF — VERIFIED** from `src/utils.jl`. Gaussian with **FWHM = λ/R**:

```julia
σ = λ0 / R_val / (2sqrt(2log(2)))     # 2√(2 ln 2) = 2.35482
```

- `R` may be a **function of wavelength**: `_resolve_R(R::Function, λ0) = R(λ0*1e8)`
  — the callable receives wavelength **in Å**.
- `window_size` default **4**, in units of **σ** (docstring is explicit:
  *"in units of the LSF width (σ, not HWHM)"*).
- `apply_LSF(flux, wls, R; window_size=4)` short-circuits to `copy(flux)` when `R == Inf`.
- `compute_LSF_matrix(synth_wls, obs_wls, R; window_size=4, verbose=true)` returns a
  sparse matrix and additionally **resamples** onto `obs_wls`.
- **Normalisation and edge handling are identical** between the two: both call
  `_lsf_bounds_and_kernel`, which truncates at the grid edge and then always
  renormalises `ϕ ./ sum(ϕ)`. The only differences are resampling and speed
  (`apply_LSF` is documented as *"a naive, slow implementation"*).
- **Documentation defect (VERIFIED):** `apply_LSF`'s docstring advertises a
  keyword `renormalize_edge=true` that **does not exist in the code**. Passing
  it raises an unknown-keyword error; the behaviour it describes is
  unconditionally on.

This matches BOSZ's measured convention exactly (§A.4). **Both codes use
Gaussian FWHM = λ/R.**

**Rotation — VERIFIED**, `apply_rotation(flux, wls, vsini, ε=0.6)`.

- **Default limb darkening ε = 0.6**, linear law $`I(\mu) = I(1)(1-\varepsilon+\varepsilon\mu)`$,
  docstring citing *"Gray equation 18.14"* — i.e. Gray, *The Observation and
  Analysis of Stellar Photospheres*. Confirms the premise.
- **It is not a sampled convolution.** Korg evaluates the **analytic
  antiderivative** of Gray's kernel at pixel edges and differences them, so it
  is exact even when `vsini` is comparable to or smaller than the pixel size.
  With `Δλrot = λ·vsini/c`, `c1 = 2(1-ε)`, `c2 = πε/2`, `c3 = π(1-ε/3)`, the
  integrand is
  `(0.5*c1*d*sqrt(1-d^2/Δλrot^2) + 0.5*c1*Δλrot*asin(d/Δλrot) + c2*(d - d^3/(3*Δλrot^2))) / c3`.
- `vsini == 0` returns `copy(flux)`. Each contiguous window is broadened independently.

**This is the single most likely place for albireo to differ.** If albireo
samples the Gray kernel at pixel centres rather than integrating it over pixels,
the two agree for fast rotators but diverge once `vsini` approaches the pixel
size — 6.46 km/s on the BOSZ r20000 grid. See §D.

### B.3 `vacuum_to_air` / `air_to_vacuum`

**VERIFIED.** Both docstrings cite *"Formula from Birch and Downs (1994) via the
VALD website."* — [Birch & Downs 1994, Metrologia 31, 315](https://doi.org/10.1088/0026-1394/31/4/006).

With `s = 1e4/λ` (λ in Å):

```julia
# vacuum_to_air:  λ / n
n = 1 + 0.0000834254 + 0.02406147/(130 - s^2) + 0.00015998/(38.9 - s^2)

# air_to_vacuum:  λ * n
n = 1 + 0.00008336624212083 + 0.02408926869968/(130.1065924522 - s^2)
      + 0.0001599740894897/(38.92568793293 - s^2)
```

**These are not exact inverses** — they are the two separately published VALD
fits (the forward Birch & Downs dispersion and Morton's inverted form), so
round-tripping leaves a small residual. **VERIFIED**

This is the same relation GSP-Spec used to put its grid into vacuum (§C.2), so
using Korg's `air_to_vacuum` on BOSZ reproduces Gaia's own convention.

### B.4 Bundled line lists and the RVS blue-end gap

**VERIFIED** from `src/linelist.jl`:

| function | coverage | default kwarg |
|---|---|---|
| `get_GES_linelist(; include_molecules=true)` | 4750–6850 and **8488–8950 Å air** → ≈ 8490.3–8952.5 Å vacuum | `include_molecules=**true**` |
| `get_VALD_solar_linelist()` | not stated — *"quick tests only"* | — |
| `get_APOGEE_DR17_linelist(; include_water=true)` | ≈ 15 000–17 000 Å | `include_water=true` |
| `get_GALAH_DR3_linelist()` | ≈ 4675–7930 Å | — |

GES coverage independently **VERIFIED** from Heiter et al. 2021, A&A 645, A106
([arXiv:2011.02049](https://arxiv.org/abs/2011.02049)), §4.3:
*"the wavelength ranges from 4750 Å to 6850 Å and from 8488 Å to 8950 Å"*, with
*"Wavelength of the transition **in air**"*. Korg converts on load via
`air_to_vacuum.(read(f["wl"]))`, and silently drops CH lines with `log_gf > -1.9`.

> **The gap is real. A Korg synthesis with the GES list over 8460–8700 Å has
> no atomic or molecular lines below ≈ 8490.3 Å (vacuum) — roughly the blue
> 30 Å of the RVS band is pure continuum, silently, with no warning.**
> **VERIFIED**

Aggravating factors:

- **No other bundled list fills it**: GALAH DR3 stops at 7930 Å, APOGEE DR17
  starts at 15 000 Å. A custom VALD extraction is required for the full band.
- **Hydrogen is the exception**: `hydrogen_lines=true` (default) computes H
  lines from Korg's own `Stehle-Hutchson-hydrogen-profiles.h5`, independent of
  the line list. Whether the Paschen lines in 8460–8700 Å (P13–P17) are
  included is **LIKELY but UNVERIFIED** — Stehlé & Hutcheon 1999 tabulate
  *"Lyman, Balmer and Paschen lines"* and Korg's loop is unfiltered by lower
  level, but the shipped level attributes live in HDF5 and could not be read
  from source. Confirm with
  `[(l.lower, l.upper) for l in Korg._hline_stark_profiles]`.
- The Ca II triplet itself (8500.4 / 8544.4 / 8664.5 Å vacuum) is comfortably
  **inside** the covered region.

### B.5 `interpolate_marcs` — and the hard 8000 K ceiling

**VERIFIED** from `src/atmosphere.jl`. The default grid is the **SDSS/APOGEE
MARCS grid** (`SDSS_MARCS_atmospheres_v2` artifact, MARCS_v3_2016,
Gustafsson et al. 2008). Wheeler+2024: *"579,150 model atmospheres … Teff (2500
to 8000 K), log g (−0.5 to 5.5), [metals/H] (−2.5 to 1.0), [α/Fe] (−1 to 1),
and [C/Fe] (−1 to 1)."*

Steps from the [SDSS MARCS_v3_2016 readme](https://dr17.sdss.org/sas/dr17/apogee/spectro/speclib/atmos/marcs/MARCS_v3_2016/Readme_MARCS_v3_2016.txt) — **VERIFIED**:

- Teff: 2500–4000 K step **100 K**; 4000–8000 K step **250 K**
- log g: −0.5 (spherical, poorly converged); 0.0–3.0 step 0.5 (spherical); 3.5–5.0 step 0.5 (plane-parallel)
- [Fe/H], [α/Fe], [C/Fe]: step **0.25 dex**

> **The lower bound is 2500 K, not the 2800 K in the brief.** The upper bound is
> **8000 K**. **VERIFIED**

**Above 8000 K, `synth` fails.** It calls `interpolate_marcs` unconditionally,
which throws `Korg.LazyMultilinearInterpError`:
*"Can't interpolate grid. Teff is out of bounds …"*. `clamp_abundances=true`
clamps **abundance** axes only and will not rescue an out-of-range Teff.
A separate `Korg.AtmosphereInterpolationError` fires for in-bounds but
unconverged corners. **Korg is an FGKM code; there is no path to a hot-star
atmosphere.** **VERIFIED**

Three interpolation branches (**VERIFIED**): standard linear; cool dwarfs
(Teff ≤ 4000 K, log g ≥ 3.5) resampled onto fixed τ₅₀₀₀ with a **cubic spline**;
and low metallicity (−5 ≤ M_H < −2.5) using a fixed-composition grid that
requires `alpha_m = 0.4, C_m = 0`.

`spherical` defaults to `logg < 3.5` → `ShellAtmosphere` (assuming 1 M☉), else
`PlanarAtmosphere`. **VERIFIED**

**Microturbulence trap — VERIFIED.** The MARCS atmospheres were computed with
*"1 km/s for dwarfs and 2 km/s for giants"*. This is **baked into the atmosphere
grid and completely independent of the `vmic` you pass to `synth`** (default
1.0 km/s), which affects only line broadening in the transfer step. Any
`vmic ≠` the grid value is formally inconsistent — **and the same criticism
applies to BOSZ**, whose v0/v1/v2/v4 products are synthesis-side too.

**Abundance handling — VERIFIED.** Passing `A_X` is recommended; passing
`M_H, alpha_m, C_m` directly is labelled *"dangerous!"* and warns unless
suppressed. MARCS assumes **Grevesse+ 2007** solar abundances, while Korg's
v1.0 default for `format_A_X` is **Bergemann et al. 2025** — the atmosphere and
synthesis reference scales differ by design and are reconciled only via `A_X`.

Other `synthesize` defaults (**VERIFIED**): `line_buffer=10.0`, `cntm_step=1.0`,
`hydrogen_lines=true`, `hydrogen_line_window_size=150`, `mu_values=20`,
`line_cutoff_threshold=3e-4`, `return_cntm=true`, `I_scheme="linear_flux_only"`,
`tau_scheme="anchored"`.

---

## C. Comparisons, and the actual Gaia RVS template library

### C.1 Is there a published BOSZ vs Korg comparison?

**No. UNVERIFIED (absence of evidence)**, established negatively at both ends:

- The BOSZ 2024 paper contains **no** comparison to Korg, Turbospectrum, SME,
  MOOG or PHOENIX. Its only comparisons are internal (BOSZ-2024 vs BOSZ-2017,
  and MARCS vs ATLAS9 within BOSZ). **VERIFIED** by reading the paper.
- Neither Korg paper mentions BOSZ. **VERIFIED**

**Wheeler et al. 2023 code comparison — VERIFIED.** Korg vs **MOOG,
Turbospectrum and SME** (no SYNTHE, no ATLAS9, no BOSZ), on solar,
Arcturus-like, HD 122563-like and HD 49933-like stars, in six windows:
3660–3670, 3935–3948, 5160–5175, 6540–6580, 15 000–15 050 Å and a
2000–10 000 Å continuum.

- *"The disagreements between Korg and the other codes are no larger than those between the other codes, although disagreement between codes is substantial."*
- *"fluxes disagreeing at the ten percent level in many cases"*; Hα wings *"at the ∼1% level"*; internal radiative-transfer schemes *"agree at the sub-percent level"*.
- **The Ca II triplet / 846–870 nm region was not examined.** The only refereed
  Korg code-comparison covers nothing in the RVS band. **VERIFIED**

**The one comparison that does cover Ca II: STARDIS** — Shields, Kerzendorf,
Smith et al. 2025, ApJ ([arXiv:2504.17762](https://arxiv.org/abs/2504.17762),
DOI [10.3847/1538-4357/adee11](https://doi.org/10.3847/1538-4357/adee11)),
against **Korg v0.42.0** on MARCS models. **VERIFIED**: solar normalised spectra
agree *"on the level of 3% or better"*, with the **Ca II triplet showing
"strong agreement to better than 1%"**. For HD 122563 (4500/1.5/−2.5), Hα is
~20 % stronger in STARDIS. This ~1 % figure is currently the best published
anchor for Korg's LTE reliability in the RVS band, at solar parameters only.

### C.2 The Gaia RVS template library — it is not PHOENIX

**VERIFIED** from three independent primary sources. The RV pipeline uses an
updated **Sordo et al. (2011)** library with three sub-grids — **MARCS**,
**A-type** and **OB-type** (Sartoretti et al. 2018 §4.4,
[arXiv:1804.09371](https://arxiv.org/abs/1804.09371), A&A 616, A6).

- DR2: **5256** spectra. DR3: **6772** (MARCS 4306, A-type 304, OB-type 2162) —
  Katz et al. 2023, A&A 674, A5 ([arXiv:2206.05902](https://arxiv.org/abs/2206.05902)); [DR3 doc §6.2.3](https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu6spe/sec_cu6spe_input/ssec_cu6spe_auxdata.html).
- Coverage (DR2, verbatim): MARCS Teff 2500–8000 K (100 K below 3900, 250 K
  above), log g −0.5…+5 step 0.5, [Fe/H] −5.0…+1.0; A-type Teff 8500–15 000 K
  step 500 K, log g +0.5…+5; OB-type Teff 15 000–55 000 K.
- **Template preparation:** *"the spectra were convolved with a Gaussian LSF
  with R = 11 500, independent of the CCD and FoV"* and *"A rotational velocity
  = 0.0 km s⁻¹ was assumed"*. Range *"845-872 nm"*. **VERIFIED**
- **The OB grid was not used for DR3.** Blomme et al. 2023 (A&A 674, A7,
  [arXiv:2206.05486](https://arxiv.org/abs/2206.05486)) subsets the same CU6
  library at Teff ≥ 6500 K and tops out at **14 500 K** — consistent with the
  existing `gaia-rvs-facts` note that DR3 has zero stars above 14 500 K. **VERIFIED**
- **Template sampling step: UNVERIFIED — not published** in Sartoretti 2018,
  Katz 2023, or DR3 doc §6.2.3. A real gap if you want to reproduce DR3 templates.
- **Template medium: LIKELY vacuum** (not stated explicitly), inferred from the
  RVS spectra being vacuum, Blomme et al. 2023 Fig. 5 (*"All spectra are at
  their vacuum rest wavelength"*), and GSP-Spec converting its grid to vacuum.

**GSP-Spec (Recio-Blanco et al. 2023, A&A 674, A29,
[arXiv:2206.05541](https://arxiv.org/abs/2206.05541)) — VERIFIED**: MARCS
atmospheres + **Turbospectrum** 19.1.2, Teff 2600–8000 K, log g −0.5…5.5,
[M/H] −5.0…+1.0, no rotation or macroturbulence. Three distinct sampling
steps, and conflating them is the trap:

1. **native synthesis 0.001 nm**; 2. **published RVS spectra 0.01 nm**;
3. **GSP-Spec working grid 0.03 nm** (800 points).

Convolved with *"R = 11 500"*. And the decisive sentence:

> *"The spectra were computed in the air and then converted into vacuum
> wavelengths thanks to the relation of Birch & Downs (1994)."*

**Bandpass and resolving power — VERIFIED.** Cropper et al. 2018
([arXiv:1804.09369](https://arxiv.org/abs/1804.09369), A&A 616, A5) Table 5:
in-flight bandpass **845.0–872.5 nm (FWHM)**; dispersion 8.51 km/s/pix at
847 nm. Processing trims the wings to **846–870 nm**. Design spec
*"average 10 500−12 500"*; **measured per-CCD R runs ≈ 10 983 – 12 587**
(Cropper Table 6, **LIKELY** for individual cells, robust for the bracket) —
i.e. the true resolution sits *above* 11 500 over much of the focal plane. **The
brief's "R ~ 11 000–11 500" understates it**; 11 500 is the nominal value the
pipelines adopt, not the measured centre.

**Published DR3 RVS mean spectra are VACUUM. VERIFIED**, e.g. Sartoretti et al.
2023: *"The Ca II triplet (at 850.035, 854.444, and 866.452 nm; rest-wavelengths
in vacuum)"*. The [`gaiadr3.rvs_mean_spectrum` data model](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_spectroscopic_tables/ssec_dm_rvs_mean_spectrum.html)
states: *"the wavelength grid ranges from 846 to 870 nm in steps of 0.01 nm
(2401 elements)"*, rest-frame and normalised. (Recio-Blanco 2023 says 2400;
846–870 at 0.01 nm is 2400 intervals / 2401 endpoints — trust 2401.)

### C.3 NLTE in the Ca II triplet — the LTE floor, with numbers

**Lagae, Amarsi & Lind 2025, A&A 697, A60**
([arXiv:2503.10378](https://arxiv.org/abs/2503.10378),
DOI [10.1051/0004-6361/202452874](https://doi.org/10.1051/0004-6361/202452874)),
3D non-LTE Ca II, STAGGER vs MARCS. Abstract, **VERIFIED verbatim**:

> *"Abundance corrections for the CaT lines relative to 1D LTE range from +0.1
> to -1.0 dex … **Radial velocity corrections** relative to 1D LTE based on
> cross-correlation of the whole line profile range from **-0.2 km/s to
> +1.5 km/s**, with more severe corrections where the CaT lines are strongest."*

At Gaia RVS resolution the RV corrections are **−0.25 to +0.25 km/s with
outliers to ±1.0 km/s** (**LIKELY** — machine-extracted from the body).

**Osorio et al. 2022, ApJ 928, 173** (`2022ApJ...928..173O`,
DOI [10.3847/1538-4357/ac5a53](https://doi.org/10.3847/1538-4357/ac5a53)):
1D NLTE abundance corrections span ≈ −0.016 to −0.628 dex, exceeding 0.5 dex at
extremely low metallicity. Model atom: Osorio et al. 2019, A&A 623, A103
(`2019A&A...623A.103O`). **VERIFIED**

### C.4 BOSZ already used in this band

**Mészáros et al. 2026**, *"Radial velocity and atmospheric parameter
calculations for the GaiaNIR spectrograph"*
([arXiv:2607.15796](https://arxiv.org/abs/2607.15796)) uses BOSZ 2024 at
R = 5000–20 000 across 800–2300 nm and states *"all calculations are performed
in vacuum wavelengths"* — i.e. **the BOSZ authors themselves convert air→vacuum
for RV work.** The conversion is mandatory, not optional. **VERIFIED**

No paper was found using BOSZ as *Gaia* RVS templates specifically.
**UNVERIFIED (absence of evidence).**

---

## D. Consequences for an albireo simulator

Configuration assumed: **BOSZ r20000 → Gaussian convolution to R = 11 500 →
Gray rotational kernel**, versus a colleague's **Korg.jl** reference.

### D.1 Reproduced identically (same convention, verified on both sides)

1. **The instrumental LSF.** Both are a Gaussian of *constant resolving power*,
   **FWHM = λ/R**. Korg: `σ = λ0/R/(2√(2 ln 2))` in source. BOSZ: measured
   13.35 km/s against 13.74 predicted for the r50000 → r20000 step, with the
   σ = λ/R alternative rejected at 67× worse residual. **The two conventions
   agree exactly.**

2. **Gaussian composition is exact.** Because FWHM = λ/R is constant in
   *velocity*, and the BOSZ grid is uniform in ln λ, reaching R = 11 500 from
   r20000 is an exact Gaussian composition on a uniform grid:

   ```math
   \mathrm{FWHM}_{\rm extra} = c\sqrt{\frac{1}{11500^2}-\frac{1}{20000^2}} = 21.328~\mathrm{km/s}
   ```

   = **3.303 px FWHM (σ = 1.403 px)** on the 6.457 km/s r20000 grid. Comfortably
   resolved; a Gaussian quadrature kernel is the right tool.

3. **Starting from r20000 rather than a finer product costs essentially
   nothing.** Measured, over 8460–8700 Å, against the r50000 product of the same
   model convolved directly to R = 11 500:

   | metric | value |
   |---|---|
   | rms(r20000 route − r50000 route) | 0.000262 = **0.28 % of line-depth rms** |
   | max abs difference | 0.0026 |
   | **RV bias (cross-correlation)** | **−0.7 m/s** |

   For comparison, getting the LSF convention wrong (σ vs FWHM) gives **20.3 %**.
   The final product has **4.04 samples per FWHM** at R = 11 500 — well sampled.

   > **r20000 is the correct choice.** `r10000` is already broader than
   > R = 11 500 (29.98 vs 26.07 km/s) and cannot be sharpened; `r50000` is
   > 2.5× larger for a 0.28 % gain.

4. **Rectification.** BOSZ ships flux and continuum in the same file; the
   continuum is smooth (0.014 % change per R = 11 500 FWHM), so
   `broadened(F)/C` and `broadened(F/C)` — the BOSZ and Korg orders respectively
   — agree to better than 0.1 %.

5. **The Gray rotational kernel with ε = 0.6** — *provided albireo integrates
   the kernel over pixels* as Korg does (§B.2). If albireo samples at pixel
   centres, the two agree for fast rotators and diverge once vsini approaches
   the 6.46 km/s pixel. **This is the most likely silent divergence between the
   two routes; worth matching deliberately.**

6. **The physics class.** Both are 1-D, static, LTE, plane-parallel/spherical
   MARCS below 8000 K. Neither has NLTE Ca II.

### D.2 Cannot be reproduced

1. **Air vs vacuum — the dominant risk, and it is not subtle.**
   BOSZ is **air** above 200 nm; Korg, the Gaia RVS spectra, the RVS templates
   and the GSP-Spec grid are all **vacuum**. At 8580 Å the offset is
   **2.36 Å ≡ 82.4 km/s**. Mixing the media is not a small systematic — it
   swamps every orbit in the RVS band. albireo must apply `air_to_vacuum`
   (Birch & Downs 1994, as Korg and GSP-Spec both do) on read, and the band
   "8460–8700 Å" must be stated as vacuum to match Gaia. Note also that Korg's
   `air_to_vacuum` and `vacuum_to_air` are **not exact inverses**.

   **A residual −4.04 m/s even when both convert correctly. VERIFIED (measured).**
   albireo's `grids.air_to_vacuum` uses *Edlén (1966) in the Birch & Downs (1994)
   parameterisation, evaluated at the vacuum wavenumber* (the Morton 2000
   convention); Korg uses the *VALD-published Birch & Downs fits*; GSP-Spec
   states plain *"Birch & Downs (1994)"*. These are not the same coefficients:

   | air λ | albireo → vac | Korg → vac | difference |
   |---|---|---|---|
   | 8498.02 | 8500.35488 | 8500.35500 | **−4.04 m/s** |
   | 8542.09 | 8544.43680 | 8544.43692 | **−4.04 m/s** |
   | 8662.14 | 8664.51927 | 8664.51939 | **−4.04 m/s** |

   The offset is **constant in velocity across the band**, so it is a pure RV
   zero point, not a distortion — but it is exactly the kind of few-m/s
   systematic that gets misattributed to the disentangling. Worth either
   matching Korg's coefficients for the comparison or subtracting it knowingly.
   (Korg's two non-inverse fits round-trip to < 0.005 m/s here, so that caveat
   is negligible in the RVS band.)

2. **Per-component microturbulence is quantised.** BOSZ offers only
   **v_mic ∈ {0, 1, 2, 4} km/s** as a grid axis; Korg takes a continuous
   `vmic` (default 1.0). Interpolating BOSZ in v_mic across a 0→1→2→4 ladder is
   coarse and non-uniform. *Mitigating*: Korg's own MARCS atmospheres carry a
   **fixed** internal v_mic (1 km/s dwarfs, 2 km/s giants) regardless of the
   `vmic` passed to `synth`, so Korg's continuity is partly cosmetic — the
   atmospheric structure is just as quantised.

3. **Per-component α, and especially calcium.** BOSZ has only **four [α/M]
   values** (−0.25, 0, +0.25, +0.50) and **no individual-element axis at all**.
   Korg accepts arbitrary continuous `alpha_H` *and* per-element overrides
   (`Ca=0.2`). Since the Ca II triplet is the dominant feature of the RVS band
   and Ca is an α element, **albireo cannot vary calcium independently of the
   other α elements, and cannot vary it continuously.** For a study whose
   signal is the Ca II triplet, this is the sharpest scientific limitation of
   the BOSZ route.

4. **Teff limits — in both directions, and the surprise is which way.**
   - **BOSZ reaches 16 000 K; Korg stops at 8000 K.** For A and late-B
     components the Korg reference implementation simply cannot produce a
     spectrum (`LazyMultilinearInterpError`). Any albireo-vs-Korg comparison is
     restricted to Teff ≤ 8000 K for *both* components.
   - **Korg reaches 2500 K; BOSZ stops at 2800 K.** For late-M secondaries
     BOSZ is the one that fails.
   - BOSZ also has **no NLTE above 16 000 K** by design, so early-B and O
     components are out of reach for both.

5. **Line lists differ, and Korg's default has a hole in the band.** BOSZ uses
   Synspec with Kurucz/ExoMol lists continuously across 50 nm–32 µm. Korg with
   `get_GES_linelist()` has **no lines below ≈ 8490.3 Å vacuum** — the blue
   ≈ 30 Å of the RVS band is silently pure continuum (hydrogen excepted). A
   like-for-like comparison must either restrict to λ > 8490 Å or give Korg a
   custom VALD extraction. Korg's default `get_VALD_solar_linelist()` is
   self-described as for *"quick tests only"*.

6. **Continuous vs gridded parameters, with holes.** Korg synthesises at
   arbitrary (Teff, log g, [M/H]) inside MARCS. BOSZ is a lattice
   (250 K / 0.5 dex / 0.25 dex) that is **not a complete tensor**: 1.9–4.7 %
   of nodes are missing depending on metallicity, and the node set itself
   changes with metallicity (§A.1.1). An albireo interpolator must handle holes
   explicitly rather than assume a rectangular grid.

7. **The MARCS/ATLAS9 seam at 7500–8000 K.** Both atmospheres exist for the same
   (Teff, log g) there, and additionally MARCS ships plane-parallel *and*
   spherical variants. The paper quantifies the disagreement as
   *"±0.5% at solar composition between 350 and 1000 nm"*. albireo must make a
   deliberate, documented choice (a `mp`/`ms`/`ap` preference rule); silently
   taking the first match makes the grid discontinuous at 7500 K.

8. **Absolute flux, if ever needed.** BOSZ 2024 is `F = 4πH`; BOSZ **2017** is
   `F = πH` (the stored values are 4× larger). Korg with `rectify=false` is
   `erg s⁻¹ cm⁻⁴ Å⁻¹`, needing `× R★²`. Three different conventions; only
   rectified spectra are safely comparable.

9. **An LTE floor neither route escapes.** 3D non-LTE Ca II triplet **radial
   velocity** corrections reach ±0.25 km/s typically and ±1 km/s in outliers at
   RVS resolution (Lagae et al. 2025). Since the Ca II triplet dominates the
   RVS cross-correlation signal, this bounds the accuracy of *any* LTE
   simulator in this band — worth stating explicitly rather than attributing
   such residuals to the disentangling.

### D.3 Recommended statement of the difference

> albireo reads BOSZ 2024 r20000 (Synspec, LTE, MARCS ≤ 8000 K / ATLAS9
> 7500–16 000 K, **air** wavelengths, log-uniform grid at 6.457 km/s,
> 2.32 samples per R = 20 000 FWHM) and convolves to R = 11 500 with a Gaussian
> of FWHM = λ/R. The colleague's reference synthesises with Korg.jl (LTE,
> SDSS-MARCS 2500–8000 K, **vacuum**, 0.01 Å linear grid) and applies the same
> Gaussian FWHM = λ/R plus Gray's rotational kernel with ε = 0.6.
>
> The LSF and rotational conventions are identical, and the r20000 starting
> point costs 0.28 % in line depth and 0.7 m/s in radial velocity. The routes
> differ irreducibly in (i) wavelength medium — 82.4 km/s if not converted,
> (ii) line lists, with the Korg GES default blind below 8490 Å vacuum,
> (iii) sampling of v_mic and α, with calcium not independently adjustable in
> BOSZ, and (iv) temperature reach — Korg cannot exceed 8000 K, so only BOSZ
> can supply the A/late-B components of the eclipsing-binary population.

### D.4 Cross-check against the in-flight albireo implementation

`src/albireo/gaia.py`, `examples/14_gaia_rvs.py` and `tests/test_gaia.py` were
created in the working tree at 20:57 during this session (not by this research
task, which wrote only to the scratchpad). Read-only inspection shows the
conventions already agree with everything verified above:

| finding | implementation | verdict |
|---|---|---|
| Gaia scale is vacuum | `RVS_BAND` documented "vacuum Angstrom"; `medium="vacuum"` passed throughout | correct |
| LSF is FWHM = λ/R | `rvs_lsf_sigma_kms = C_KMS/R * 1/(2√(2 ln 2))` | correct, matches BOSZ (measured) and Korg (source) |
| library already broadened | `quadrature_sigma_kms` applies `sqrt(σ_RVS² − σ_lib²)`, default `library_resolving_power=20_000.0` | correct — and it **raises** if the library is at or below R = 11 500, which is exactly the `r10000` trap |
| BOSZ medium is version-dependent | `library.py` line 18 already states "BOSZ 2017 was vacuum throughout while BOSZ 2024 is air above 200 nm" and quotes ~83 km/s | correct (measured: 82.4 km/s) |
| do not trust metadata for medium | `line_core_medium()` *measures* the convention from line cores | correct, and the same technique used in §A.3 |

The one item to consider is the −4.04 m/s refractivity-coefficient offset
against Korg documented in §D.2.1.

---

## Appendix: reproduction notes

Files fetched from `https://archive.stsci.edu/hlsps/bosz/bosz2024/` and analysed:

- `hlsp_bosz_bosz_sim_all_all_v1_readme.txt` (9958 B, Last-Modified 2025-09-25)
- `wavelength_grids/bosz2024_wave_r20000.txt` (4 500 000 B, Last-Modified **2023-09-17**)
- `r20000/m+0.00/bosz2024_mp_t5000_g+4.5_m+0.00_a+0.00_c+0.00_v2_r20000_resam.txt.gz` (Last-Modified 2025-05-22)
- `r20000/m+0.00/bosz2024_ap_t10000_g+4.0_m+0.00_a+0.00_c+0.00_v2_r20000_resam.txt.gz` (Last-Modified 2025-04-03)
- `r50000/m+0.00/bosz2024_mp_t5000_g+4.5_m+0.00_a+0.00_c+0.00_v2_r50000_resam.txt.gz`
- Directory listings for `r20000/m+0.00/`, `r20000/m-2.50/`, `r20000/m+0.75/`

Note the `+` in metallicity directories must be percent-encoded (`m%2B0.00`).
The r50000 wavelength grid was **not** downloaded; it was reconstructed from
Eqs. 3–4 and validated by recovering Ca II 8542.09 Å to 0.003 Å.
