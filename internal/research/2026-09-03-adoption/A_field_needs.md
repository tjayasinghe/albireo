# What the binary-star spectroscopy community does and needs, 2023–2026

**For:** the maintainer of `albireo` (Bayesian spectral disentangling in JAX).
**Date:** 2026-09-03. **Coverage:** literature 2023-01 → 2026-09; software checked to 2026-09-02.

---

## 0. How this was built, and how to read it

Four evidence streams, so any claim can be rechecked without trusting a summary.

1. **A machine-built corpus.** The arXiv API was queried for *"spectral disentangling"*, *"spectra disentangling"*, *"disentangled spectra"* and *"disentangling of the spectra"* in title-or-abstract → 169 unique records → filtered to `astro-ph` with binary/eclipsing/companion content → **121 papers, 2004–2026**. The **36 dated 2023-01-01 or later** were downloaded as arXiv HTML full text and machine-scanned for code names and pain-point phrasings. Every quote tagged with an `arXiv:` id below was copied from fetched full text.
2. **Targeted full-text reads** of ~40 further papers (BLOeM series, TMBM, M3W, Gaia processing, LAMOST/APOGEE, BEBOP).
3. **Citation counts** from OpenAlex (`api.openalex.org`). Rankings are reliable; **absolute values are lower bounds versus NASA ADS**, and pre-DOI A&A/A&AS records (Simon & Sturm 1994, Hadrava 1995, Ilijić et al. 2004) could not be resolved there at all.
4. **Repository metadata** from `api.github.com` (`pushed_at`, `license`) and, where the API was rate-limited, from GitHub commit Atom feeds and raw-file probes.

**Verified fact** = read from a fetched page, URL given. **[INFERENCE]** = my reading, flagged inline.

> ### Correction to `docs/science.md` before anything else
> The repo's reference list says *"Bodensteiner, J., Sana, H., Dufton, P. L., et al. 2025, A&A, 698, A40"* and the text says Bodensteiner et al. 2025 covers "the B-type supergiants". **Both are wrong.** Verified from the arXiv metadata API:
> - **A&A 698, A40** (doi:10.1051/0004-6361/202452963, arXiv:2502.12239) is **Britavskiy, N.; Mahy, L.; Lennon, D. J.; Patrick, L. R.; Sana, H.; Villaseñor, J. I.** — *"Multiplicity of early B-type supergiants in the Small Magellanic Cloud"*.
> - **Bodensteiner et al. 2025** is **A&A 698, A38** (doi:10.1051/0004-6361/202452623, arXiv:2502.02641) — *"Multiplicity properties of Oe and Be stars"*, with Shenar, Sana, Britavskiy, Crowther, Langer.
> This matters because the single most useful quote in this whole report (§3.10) is Britavskiy's, and citing it to the wrong first author in a methods paper would be noticed by exactly the people you want to reach.

A caveat for §5: the arXiv abstract search finds papers that *name* disentangling in title or abstract. Papers that disentangle only in a methods section are invisible to it, so §5.1 is a **lower bound**; §5.2's citation counts are the better usage proxy.

---

## 1. Communities

### 1.1 Massive-star surveys — the largest, fastest-growing, and most under-served group

**BLOeM (Binarity at LOw Metallicity).** VLT/FLAMES–GIRAFFE, LR02 setup, 3960–4570 Å, **R = 6200**, 929 SMC massive stars, 25 epochs planned, **first nine analysed**, median S/N ≈ 70–100 per epoch (sample mean 55). Twelve papers to date (arXiv:2407.14593 → arXiv:2607.27395; the ESO review is *The Messenger* 196, 15–19, 2026).

Intrinsic close-binary fractions: **70 % for O-type, ~80 % for early-B dwarfs/giants**, ~40 % for B0–B3 supergiants, ~8 % for A–F supergiants (arXiv:2511.08675). Villaseñor et al. 2025 (A&A 698, A41; arXiv:2503.21936) identify *"153 stars (91 SB1, 59 SB2, 3 SB3)"* in the early-B sample alone.

**No BLOeM paper published so far does spectral disentangling.** The series splits into two RV camps — cross-correlation against a co-added self-template (Shenar 2024, Britavskiy 2025, Bodensteiner 2025, Patrick 2025, Sana 2025), and hierarchical Bayesian multi-line profile fitting in **NumPyro/NUTS** (Villaseñor 2025, Sana 2025 for the 22 visual SB2s). Disentangling appears only as future work or as a thing they say they need (§3.10).

**The BLOeM team's toolkit is JAX + NumPyro.** `MINATO` — https://github.com/jvillasr/MINATO — Python/Jupyter, **MIT**, last push **2026-07-22**, 3 stars. Modules: `span` (simultaneous synthetic-model fitting → Teff, log g, vsini, He/H **and the light ratio**) and `ravel` (SB1/SB2 line-profile RVs with NumPyro, Lomb–Scargle with FAP). Dependencies include `jax` and `numpyro`. **There is no disentangling module and none in the planned features** (planned: population simulations, SED fitting, spectral classification).

**TMBM / VFTS / BBC (30 Doradus).** Shift-and-add grid disentangling, Shenar's implementation. Shenar et al. 2022 (TMBM VI, A&A 665, A148, arXiv:2207.07674): 32 FLAMES epochs, 51 SB1 O/B systems, companions detected to 1–2 % of visual flux, **three-component** version with a static nebular "component C". Deshmukh et al. 2026 (TMBM VII, A&A 706, L17, arXiv:2602.03650): VFTS 812, 26/30 epochs usable, S/N 10.7–49.8; the methodological novelty is **wavelength-dependent error propagation added to shift-and-add**.

**M3W (Milky Way).** Maíz Apellániz et al. 2026, A&A accepted, arXiv:2604.02111 — successor to GOSSS/MONOS/OWN/IACOB, introducing **UNWIND** (§6.2). GLS 11 448: **77 epochs** across HET/HRS (R = 30 000), CARMENES (95 000, to 17 100 Å), HARPS-N (115 000), FIES (25 000), CAFÉ (95 000), HERMES (85 000). First OB SB2 disentangled over 3820–11 000 Å. The LiLiMaRlin archive behind it holds ~26 000 epochs over 515 core stars, median 8 per star.

**Also:** MONOS III (arXiv:2508.04272, UNWIND on 10 SB2E systems); Rauw et al. 2025 (A&A 702, A274); Gosset et al. 2026 (WR 25, arXiv:2607.24390); Deshmukh et al. 2026 (WR 25 triple, arXiv:2607.12836); Nazé et al. 2023 (extreme mass ratios to 0.003, arXiv:2308.02368); Quintero & Eenens 2024 (QER20, MNRAS 532, 2604, arXiv:2407.16816); Rauw 2024 review (arXiv:2411.00793); IACOB CVIII (arXiv:2405.11209, no disentangling). XShootU VIII (arXiv:2406.17678) does **not** disentangle — it merges per-component PoWR models by an assumed luminosity ratio.

### 1.2 Eclipsing and benchmark binaries

Pattern: light curve (JKTEBOP/PHOEBE/WD) → RVs (TODCOR or broadening functions) → disentangling (FDBinary/fd3, or bespoke) → renormalise to individual continua using the photometric light ratio → atmospheres (GSSP, iSpec, SME). Six identifiable groups:

| Group | Disentangling code | Orbit / LC | Data | Epochs |
|---|---|---|---|---|
| **Southworth** (*Rediscussion of eclipsing binaries*, The Observatory; DEBCat) | **none — he does not disentangle and takes no new spectra**; reanalyses published RVs with TESS | `jktebop`, `jktabsdim`, `jktld`, `period04` | TESS + literature RVs, often 1950s–80s plates | — |
| **Pavlovski / Tkachenko / Serebriakova** (Leuven + Zagreb) | **FDBinary** (DFT variant) | `starfit`, GSSP, `iSpecMCMC`; WD, `jktebop`, PHOEBE2, `ellc`; bootstrap (10 000) for orbital errors | HERMES R = 85 000 (workhorse), HARPS 80 000 | 8–10 (survey) to 29–72 |
| **Maxted** (Keele FTEB) | **his own** Simon & Sturm wavelength-space code, LSQR | `jktebop`, `teb`, iSpec CCF | HARPS ~80 000, UVES (**incl. in-eclipse**), NIRPS, SPIRou | 2–3/night, S/N ~70–80 |
| **Torres** (CfA) | **TODCOR / TRICOR**; FDBinary in pure separation mode when the orbit is already fixed | `gssp_single`, `gssp_binary` | TRES R ≈ 44 000, 3800–9100 Å | 48–51, S/N 32–102 |
| **Hełminiak** (Toruń, CRÉME) | own TODCOR + bootstrap; **Rucinski BFs**; **Shenar's `dsaa`**; TRICOR for triples | `v2fit`, `jktebop` (spectroscopic light ratio as constraint), iSpec | CHIRON 28 000, FEROS 48 000, CORALIE 70 000, SALT/HRS 67 000, HARPS 115 000, UVES | 6–34 |
| **Graczyk / Araucaria** | `RaveSpan` — **Rucinski BFs** + **González & Levato shift-and-add** | Wilson–Devinney | Magellan/MIKE, VLT/UVES, HARPS | — |

Corpus entries: Pavlovski et al. 2023 (A&A 671, A139), Tkachenko et al. 2023 (arXiv:2312.14009, the mass-discrepancy programme), Jennings et al. 2023 (KIC 9851944, arXiv:2311.02064), Budding et al. 2023 (VV Ori, arXiv:2311.08247), Moharana et al. 2023/2024/2026, Kemp et al. 2024 (A&A 689, A164), Hełminiak et al. 2024 (A&A 691, A170), Serebriakova et al. 2025 (arXiv:2507.10096), Torres et al. 2025 (27 Tau, arXiv:2507.15933), Dervişoğlu et al. 2026 (Z Vul, arXiv:2608.03348), Takeda 2025 ×2, Yücel et al. 2025/2026, Meng et al. 2026, and **Maxted et al. 2026 on AI Phoenicis** (arXiv:2607.17976, doi:10.1093/mnras/stag1398) — the same system `albireo` validates on. See §4.1.

**[INFERENCE] Three structural facts about this community.** (i) **Southworth does not disentangle at all**, and his entire series is bottlenecked on the *spectroscopic light ratio* he must feed to `jktebop` as a constraint on the radius ratio — which he repeatedly reports as unreliable or unexplained (§3.1). A tool returning a wavelength-resolved light ratio with a defensible error bar slots directly into that slot. (ii) **Hełminiak's 2024 totality paper is an explicit route *around* disentangling**: observe during a total eclipse and you have a genuine single-star spectrum (*"any observations taken at that time are in fact of a single star"*, arXiv:2409.20144). (iii) The whole **FTEB series is built on wavelength-dependent flux ratios measured in multiple bands** — this community needs `l(λ)`, not a scalar.

The community's own roadmap documents ask for R ≥ 50 000–85 000, S/N ≥ 300, flexible phase scheduling, and 0.1 %-level masses and radii as table stakes (Maxted et al. 2026, *"Detached eclipsing binary star science in the 2040s"*, arXiv:2601.17179; Šipková et al. 2025, arXiv:2512.14816).

### 1.3 Dark companions and compact-object searches

The highest-impact strand: eight of the ten most-cited disentangling papers of 2020–2026 (§5.3). Live entries: El-Badry et al. 2025 (ALS 8814, arXiv:2509.01545), TMBM VII (VFTS 812), Müller-Horn et al. 2026 (MWC 656, A&A 708, A187, arXiv:2601.14403), Janssens et al. 2023 (MWC 656, A&A 677, L9), Saracino et al. 2023 (NGC 1850 BH1), Banyard et al. 2023 (NGC 6231, A&A 674, A60), Müller-Horn et al. 2025 (HIP 15429, arXiv:2504.06973), El-Badry et al. 2026 (227 Gaia candidates, 1292 RVs, arXiv:2608.06453), Nagarajan et al. 2026 (arXiv:2602.21289).

Typical data are **small**: 9–31 epochs, S/N often in the teens.

### 1.4 Gaia, and survey-scale SB2 work

- **Gaia DR3 discarded SB2s.** Katz et al. (arXiv:2206.05902, §4.3.1): *"Approximately 40 000 sources with 10% or more transits flagged as double-line were considered SB2 candidates and their combined radial velocities were discarded."*
- **DR3 SB1 chain also rejected them.** Gosset et al. 2025 (arXiv:2410.14372, §2): *"The stars exhibiting a double-line spectrum (also named composite spectrum) were rejected from the process."*
- **DR4 is confirmed for 2026-12-02** and will publish epoch RVS spectra plus FOV-transit information for double-lined transits.
- **No published Gaia-DR4 SB2 tool exists.** The nearest things are Seeburger et al. 2024 (built for R ≈ 2000), MESS (arXiv:2602.05610), and Binnenfeld et al. 2025 (single-exposure deep learning, trained on simulated RVS).
- LAMOST-MRS: Guo et al. 2025 (665 SB2 orbits, arXiv:2504.11954, R ≈ 7500, ≥ 6 epochs); Kovalev et al. 2023 (12 426 SB2 candidates, arXiv:2310.11673). APOGEE: Kounkel's `apogeesb2`; Saad & Ting 2026 (41 466 DR19 SB2 candidates, arXiv:2608.10866).

### 1.5 Pulsators and Be stars in binaries

Kahraman Aliçavuş 2023/2025, Kemp 2024, Moharana 2026 (V446 Cep β Cep), Çakırlı 2025, Harmanec 2026 (o Cas, KOREL), Janssens et al. 2025 (HOney-BeeS II, arXiv:2512.15019). This is the community where disentangling is currently declared **unusable**, not merely hard (§3.8).

### 1.6 Circumbinary planets — adjacent, and a different precision regime

BEBOP (Sairam, Triaud, Baycroft). SOPHIE R = 75 000; BEBOP VI: 244 spectra of six SB2s, median S/N ≈ 85; BEBOP VII: median RV precision **2.3 m s⁻¹**, jitter 6.7 m s⁻¹; BEBOP VIII: 3.4 m s⁻¹, residual RMS 10.3 m s⁻¹. They need 1–10 m s⁻¹; SB2 disentangling has historically delivered 10–15 m s⁻¹ of scatter, which *is* the planet signal.

---

## 2. Codes and data: a measured census

### 2.1 Which code each 2023–2026 paper actually *used*

Built by regex-matching usage constructions (*"we used X"*, *"using X"*, *"X was applied"*) against the 36 downloaded full texts, so it separates **use** from **citation**: Hadrava 1995 is *cited* in 17 of 36 papers, but KOREL is *used* in 2.

| Code | Papers using it (of 36) |
|---|---|
| **shift-and-add** (González & Levato 2006 / Shenar's implementation) | **13** |
| **FDBinary / fd3** (Ilijić et al. 2004) | **8** (6 FDBinary, 2 fd3) |
| **bespoke / own implementation** | **6** |
| TODCOR (Zucker & Mazeh 1994) | 5 |
| iSpec | 5 |
| broadening functions (Rucinski) | 4 |
| KOREL | 2 |
| MINATO / `ravel` | 2 |
| GSSP | 2 |
| UNWIND | 1 |
| LSD | 1 |
| **Spectangular** | **0** |
| **PSOAP** | **0** |

Per-paper (arXiv id → code): 2301.04215 FDBinary · 2301.04409 FDBinary+TODCOR+BF+iSpec+GSSP · 2303.05272 TODCOR+iSpec+Simon&Sturm · 2303.07214 shift-and-add · 2303.07369 shift-and-add · 2307.07766 shift-and-add+ravel · 2308.03255 Simon&Sturm · 2308.08642 shift-and-add · 2312.14009 FDBinary+LSD · 2405.12136 shift-and-add+BF+iSpec · 2405.17772 shift-and-add · 2405.19391 shift-and-add+own · 2406.04131 FDBinary+GSSP · 2410.02573 own · 2507.12363 TODCOR · 2508.14464 FDBinary · 2509.01545 fd3+shift-and-add(+Seeburger) · 2509.12762 shift-and-add+ravel · 2509.18277 own · 2511.13692 TODCOR · 2511.17206 shift-and-add · 2602.03650 shift-and-add+own · 2604.02111 shift-and-add+UNWIND · 2604.20747 KOREL · 2604.25056 FDBinary+TODCOR+BF+Simon&Sturm · 2605.13212 fd3+BF+iSpec · 2605.14018 KOREL+iSpec · 2607.13772 own · 2607.17976 iSpec+own+Simon&Sturm · 2607.24390 shift-and-add.

**Three readings.**

1. **Shift-and-add is the incumbent**, by ~1.6× over fd3/FDBinary and ~6× over everything else. An adoption strategy that does not make a shift-and-add user comfortable is aimed at the wrong population. `albireo`'s clean-room `scripts/shift_and_add.py` and the head-to-head benchmark are the right instrument and are currently buried in `docs/benchmarks.md`.
2. **Six of 36 groups wrote their own disentangling code** in this window (Seeburger, Sairam, Çakırlı, Deshmukh, Xu, Maxted), plus two more full new codes shipped (UNWIND, DERVIS) and one live Python package (BiSpeD). **[INFERENCE]** That is the addressable market: people who already concluded the existing codes did not do what they needed and paid to write one.
3. **Spectangular and PSOAP have zero measured uses in 3.7 years.** Scientifically PSOAP is `albireo`'s predecessor; commercially neither is a competitor for users.

### 2.2 Typical data

| Community | Instruments | R | Epochs | S/N |
|---|---|---|---|---|
| BLOeM | VLT/FLAMES–GIRAFFE LR02 | **6 200** | 9 analysed, 25 planned | mean 55, median 70–100 |
| TMBM/VFTS | FLAMES | ~6 500 / 50 000 (HR IFU) | 26–32 | **10.7–49.8** |
| Galactic massive (M3W/OWN) | HERMES, HARPS-N, CARMENES, FIES, CAFÉ, HET/HRS | 25 000–115 000 | median 8, up to **77** | >30 required, mostly >100 |
| Eclipsing/benchmark | HARPS, UVES, FEROS, HERMES, CHIRON, TRES | 40 000–115 000 | ~10–40 | 50–250 |
| Dark companions | FEROS, TRES, HIRES, MUSE, LAMOST | 7 500–85 000 | **9–31** | teens–200 |
| Post-mass-transfer (Seeburger) | FEROS | ~48 000 | ~10 | 40–150 |
| LAMOST-MRS | LAMOST | ~7 500 | ≥6, mostly <20 | ≥20/pix |
| BEBOP | SOPHIE | 75 000 | 62–244 | 32–85 composite |

**[INFERENCE]** Two things follow. At 9–40 epochs, NUTS over 10–200 nonlinear parameters is entirely affordable, so "too slow" does not bite until someone wants all 929 BLOeM stars at once. And the *actual* dark-companion regime is **9–31 epochs at S/N in the teens** — benchmarks and tutorials built on many-epoch high-S/N data do not resemble what those users have. `albireo`'s packaged example (12 epochs) is well-matched; the HR 6819 FEROS example is not, and a low-S/N few-epoch tutorial would be more persuasive to this audience than a cleaner one.

---

## 3. Pain points, quoted

All quotes copied from fetched full text.

### 3.1 The light ratio is not measurable and is fixed by hand — the universal complaint

> *"In the absence of eclipses, the disentangled spectra can only be retrieved up to a scaling factor that depends on the light contribution of the two components to the total flux"*
> — Shenar et al. 2022, TMBM VI §3.2, arXiv:2207.07674 (verified via ar5iv)

> *"If the spectral subtype of the secondary cannot be determined, we adopt a light ratio of 5% for putative non-degenerate companions and 3% otherwise."*
> — Shenar et al. 2022, §3.3, arXiv:2207.07674

> *"We note that the light ratio of the components cannot be derived from the disentangling procedure."*
> — Saracino et al. 2023, NGC 1850 BH1, arXiv:2303.07369

> *"The light ratio of the two stars cannot be derived from the method without additional assumptions, and only affects the overall scaling of the intensities of the spectral lines."*
> — Villaseñor et al. 2023, VFTS 291, arXiv:2307.07766

> *"Disentangling does not uniquely determine the flux ratio between the two components, which must be constrained by comparison of the disentangled spectra to models."*
> — El-Badry et al. 2025, ALS 8814, arXiv:2509.01545

> *"Since the disentangling procedure requires input light ratio information, we estimated the fractional light contributions … from the BF profiles or from the initial LC solution, and adopted these estimates as essential constraints"*
> — Meng et al. 2026, arXiv:2605.13212

> *"The log g and the light ratio were fixed from the solution of the light curve analysis."*
> — Moharana et al. 2026, V446 Cep, arXiv:2604.25056

And when it is left free:

> *"We repeated this experiment by leaving the light ratio as a free parameter, but it resulted in an unconstrained light ratio and returned almost identical results to the previous fit."*
> — Villaseñor et al. 2023, arXiv:2307.07766

From the eclipsing-binary side, in the same words:

> *"their fractional light contribution cannot be determined without external information"* … *"there is an inherent ambiguity in reconstructing the component spectra"*
> — Torres, Tkachenko, Pavlovski et al. 2025, ApJ, 27 Tau, §IV, arXiv:2507.15933 (verified)

> *"Although the flux ratio affects the overall scaling of the spectral line depths, it cannot be directly derived from the disentangling method"*
> — Müller-Horn et al. 2025, HIP 15429, A&A §6.1.2, arXiv:2504.06973

> *"ambiguity exists in the reconstruction of the components' spectra"*
> — Jennings, Southworth, Pavlovski & Van Reeth 2023, KIC 9851944 §5.1, arXiv:2311.02064 — because no spectra were taken in eclipse, so the light ratio never varies

Worse — the photometric and spectroscopic values *disagree* in real systems, in both communities:

> *"the huge uncertainty about the brightness ratio prevented a quantitative assessment of this effect."*
> *"As for 29 CMa, this value is clearly at odds with the spectroscopic brightness ratio of 2.50 ± 0.43."*
> — Rauw 2024 review, arXiv:2411.00793

> *"our spectroscopic results are quite sensitive to the light fraction, while an excellent fit to the light curve is able to be obtained"*
> — Kemp, Tkachenko, Torres, Pavlovski et al. 2024, KIC 4150611 §5, arXiv:2406.04131 — spectroscopy wants 0.92–0.94, eclipse modelling wants 0.84 ± 0.03

> *"we had to fix the potential of the secondary component in order to obtain an agreement with the spectroscopic light ratio"*
> — Rojas García, Graczyk, Pietrzyński et al. 2024, A&A §5, arXiv:2410.18619

**And Southworth's running complaint, from the one group that does not disentangle but depends entirely on the number:**

> *"spectroscopic light ratios of this system are not reliable due to the chemical peculiarity of both stars"*
> *"These chemical peculiarities appear to have led to erroneous radius measurements in the past, caused by the use of a spectroscopic light ratio"*
> — Southworth 2025, *Rediscussion* XXV (AR Aur), arXiv:2507.21781

> *"The two spectroscopic light ratios are inconsistent with each other and with the photometric values"* … *"The discrepancy remains unexplained."*
> — Southworth 2026, *Rediscussion* 32 (EY Cep), arXiv:2607.10701

> *"This light ratio was obtained at blue wavelengths but was applied directly in our jktebop model of the much redder TESS data"*
> — Southworth 2026, *Rediscussion* 31 (CV Vel), arXiv:2606.04870 — permissible only because the two stars have nearly identical temperatures

### 3.2 The wavelength-dependent light ratio

> *"The current version of the code assumes that the two stars are not too different in Teff, so that the flux fraction does not change much over the wavelength range of interest. For systems composed of e.g. a blue and a red star, a solution is to divide the disentangling process in several wavelength ranges and apply different flux fractions in each."*
> — Maíz Apellániz et al. 2026, UNWIND, arXiv:2604.02111 §3.2 fn. 3

> *"Also, in the current version flux fractions are constant for each component throughout all epochs but variable ones may be implemented in the future for eclipsing binaries."*
> — same, §3.2

> *"This was performed on the wavelength range from 5275 to 5370 Å, as this section contains enough lines to successfully disentangle while still being narrow enough for the assumption of a constant light ratio not to break down."*
> — Seeburger et al. 2024, arXiv:2405.19391

Seeburger et al. 2025 devote §3.4 to *"Finding the light ratio as a function of wavelength"* and report it as a headline product (arXiv:2511.13692). Maxted et al. 2026 grid-search it **per HARPS order**.

The eclipsing-binary community has reached the same conclusion and, independently, `albireo`'s own design:

> *"the wavelength dependent light ratio l_f(λ) between the components must be known"*
> — Dervişoğlu, Adalalı, Güney, … Pavlovski et al. 2026, Z Vul, MNRAS §4.4, arXiv:2608.03348 (verified)

> *"[gssp_single] treats light dilution in the disentangled spectra as a wavelength-independent effect. This implies that the parallel analysis of the binary components' disentangled spectra may result in a total light factor that either exceeds or falls short of unity. The gssp_binary module, on the other hand, couples the components' spectra and computes the light contribution of each star per wavelength bin. In this case, the light factors are replaced by a single free parameter — the squared radius ratio between the two stars"*
> — Torres, Tkachenko, Pavlovski et al. 2025, 27 Tau §V, arXiv:2507.15933 (verified)

**[INFERENCE]** That last passage is independent corroboration of `albireo`'s D52/D53 design decision — that dilution must be fitted *jointly* through a shared radius ratio rather than as two free light factors. Two 2025–2026 papers reach it independently. Cite Tkachenko (2015) Eqs. 3–4 as used by Torres et al. 2025 §V; it is the canonical reference and it makes `albireo.match`'s choice look like community practice rather than an idiosyncrasy.

### 3.3 The k = 0 null space, under four different names

**Name 1 — "featureless continuum can go to either star":**
> *"there are several inherent degeneracies, foremost the fact that any featureless continuum portion of the spectrum can be assigned to either spectral component without consequences in the data match."*
> *"In the end we attribute different portions of the 'featureless continuum' to the two disentangled components so that their equivalent widths are physically plausible."*
> — Seeburger et al. 2024, arXiv:2405.19391 §1, §2.6

**Name 2 — "an arbitrary constant":**
> *"there is an inherent ambiguity in the solution because we can add some arbitrary constant c to the spectrum of one star and subtract the same constant from the spectrum of the other star without changing Mx."*
> — Maxted et al. 2026, AI Phe, arXiv:2607.17976 §3

**Name 3 — "undulations" / "emission wings" / "horns":**
> *"To correct spurious undulations introduced by numerical instability in the disentangling procedure (e.g. Hensberge et al. 2008), we re-normalized the disentangled spectra."*
> — El-Badry et al. 2025, arXiv:2509.01545 §3.3.1 (on the fd3 run)

> *"It is well known that the shift-and-add technique often results in apparent 'emission wings' in the disentangled spectra, which are not of astrophysical origin, but are intrinsic to the method (Quintero et al. 2020). Removing these features typically requires very large number of iterations, making the disentangling procedure inefficient. A workaround is to enforce the disentangled spectra to lie below the continuum for absorption-line spectra"*
> — Shenar et al. 2022, TMBM VI Appendix A, arXiv:2207.07674 (verified via ar5iv)

> *"a wrong choice of initial guesses can introduce spurious structures such as 'horns' (emission peaks on both sides) around absorption lines."*
> — Maíz Apellániz et al. 2026, arXiv:2604.02111 §3.3

And the hand fix, 2026:
> *"Some of segments show a noticeable curvature in the opposite sense between the two stars. This implies a rapid change in flux ratio that is not realistic, so this is likely to be an artifact of the disentangling algorithm. To remove this curvature, we perform a robust fit of a quadratic function to the difference between the disentangled spectra of star A and star B."*
> — Maxted et al. 2026, arXiv:2607.17976 §3

**Name 4 — segment-edge oscillations (Fourier-domain specific):**
> *"The spectral intervals were selected so that their boundaries coincided as closely as possible with the continuum regions; otherwise, when applying the Fourier-transform-based spd, additional sinusoidal-like oscillations could appear in the disentangled spectra."*
> — Dervişoğlu et al. 2026, Z Vul §4.3, arXiv:2608.03348 (verified) — they run a Median+Max continuum finder just to place segment boundaries

> *"The disentangled spectra had trends in the continuum, which were a result of bias in normalisation, light-fraction variation, et cetera"*
> — Moharana, Hełminiak, Marcadon et al. 2024, A&A §3.5, arXiv:2405.12136

> *"we conclude that the effect is only an artefact of the disentangling process"*
> — Harmanec et al. 2026, o Cas §3, arXiv:2604.20747 — a spurious quasi-emission peak in a disentangled Mg II profile, caught by the referee

And the summary judgement, from the group that uses FDBinary most:
> *"the quality of the resultant disentangled spectra suffers from multiple nuances, such as unknown light factors, edge effects, undulations, phase coverage, and so on"*
> — Serebriakova, Tkachenko, Johnston, Pavlovski & Aerts 2025, A&A §2.1, arXiv:2507.10096 (verified)

> *"differences between the two model spectra are of a smaller scale than artefacts in the observed spectra left over from the processes of spectrum normalisation and disentangling"*
> — same, §4.3 — **the artefacts are larger than the Teff signal being measured** (25 590 vs 28 800 K)

**[INFERENCE]** Four communities, four vocabularies, one null space. A documentation page that names all four and shows they are the same object would land better than any amount of equations — and it is the thing `albireo` handles with a prior rather than a post-hoc polynomial. Note the workaround inventory: Maxted modifies the design matrix to admit in-eclipse spectra (and otherwise fits a quadratic to the A−B difference); Moharana scales the disentangled primary against a totality spectrum; Seeburger reaches for Tikhonov; Shenar clamps the spectra below the continuum; Graczyk's group fixes a Wilson–Devinney surface potential by hand; Dervişoğlu picks segment edges with a continuum finder. **Six workarounds for one linear-algebra fact.** A package that exposes the null space as a first-class object — with a documented choice among *constrain it with a light ratio* / *constrain it with an eclipse spectrum* / *regularise it* / *leave it free and report the degeneracy* — consolidates all six. The *"enforce below the continuum"* clamp is the most telling: an ad-hoc regulariser sitting exactly where a principled prior belongs, and Quintero & Eenens's QER20 package (MNRAS 532, 2604, arXiv:2407.16816) exists purely to undo the artefacts it papers over.

### 3.4 No uncertainties on the disentangled spectra — and the visible consequences

**The single most important quote in this report.** From Tkachenko's own group, in 2025, explaining why they had to abandon χ² altogether:

> *"For disentangled spectra, which constitute two out of the four χ² terms, observational uncertainties are not well-defined. The disentangled spectra are derived products that depend on the orbital solution, meaning their quality (and thus their uncertainties) evolves dynamically throughout the optimisation process."*
> — Serebriakova, Tkachenko, Johnston, Pavlovski & Aerts 2025, A&A §2.2.2, arXiv:2507.10096 (verified verbatim)

They replace χ² with a relative residual sum of squares and get systematic uncertainties an order of magnitude larger than the literature (U Oph masses ±0.19 M☉ against published ±0.007–0.01 M☉), justified thus:

> *"our method gives estimation of systematic uncertainty by demanding a consistent solution, while such systematics are pretended to be absent from an iterative approach that treats the light curve and spectra separately."*
> — same, §4.2 (verified)

**[INFERENCE]** This is the gap `albireo` fills, written down as an open problem by the group with the most FDBinary experience in the field, eighteen months ago. `albireo`'s answer — the disentangled spectra are a *conditional Gaussian at each posterior draw*, so their uncertainty genuinely does evolve with the orbital solution and is available at every step — is a direct, formal answer to the exact objection. This quote belongs in the methods paper's introduction, and the sentence after it should be that `albireo` supplies precisely the σᵢ they had to exclude.

The rest of the pattern:

> *"…the best-fit K2 varies with the spectral region being fit, implying that the formal uncertainties on K2 are significantly underestimated."*
> — El-Badry et al. 2025, arXiv:2509.01545

> *"quoted uncertainties are within the scope of these assumptions, and as such can be considered lower bounds on the true uncertainty"*
> — Kemp, Tkachenko, Torres, Pavlovski et al. 2024, KIC 4150611 §3.2, arXiv:2406.04131

> *"the 1σ uncertainties listed for the atmospheric parameters of both stars in Table 4 should be regarded as purely statistical errors"* … *"we cannot rule out potential biases in the atmospheric properties for the secondary that are not reflected in the purely statistical errors"*
> — Torres et al. 2025, 27 Tau §V, §X, arXiv:2507.15933

> *"uncertainties from the covariance matrix of a light curve fit are notoriously underestimated"*
> — Jennings et al. 2023, KIC 9851944 Appendix A, arXiv:2311.02064

> *"Attempts to correct for such line asymmetries during the process of spectrum re-normalisation will inflict additional uncertainty in the determination of atmospheric parameters"*
> — Serebriakova et al. 2025, §4.1, arXiv:2507.10096

> *"All three lines have a best-fit χ²red that is larger than 1, meaning that the two disentangled components cannot fully describe all the variance in the observed spectra."*
> — El-Badry et al. 2025, §3.4

> *"Specifically, the main uncertainty in the disentangling outcome is in the Balmer lines profile, with implications for the surface gravity, rotational velocities, and He/H ratio."*
> — Villaseñor et al. 2023, arXiv:2307.07766

> *"For the accretor, however, the velocity estimates vary substantially and are inconsistent across the different wavelength windows, and their formal uncertainties are considerably larger than those of the donor velocities."*
> — Seeburger et al. 2025, arXiv:2511.13692 §3.1

> *"Given the uncertainty in the continuum correction around the Balmer-line wings in the disentangled tertiary spectrum, we adopted an additional diagnostic to estimate the stellar parameters of this component."*
> — Deshmukh et al. 2026, WR 25, arXiv:2607.12836

> *"The measurement uncertainties σmeas obtained from the methods are underestimated."*
> — Janssens et al. 2025, HOney-BeeS II, arXiv:2512.15019

**The state of the art**, and it is *input* weighting, not *output* uncertainty:
> *"To minimize the effects of these residuals on the disentangling, we implemented wavelength-dependent error propagation in the shift-and-add technique of TS22"*
> — Deshmukh et al. 2026, TMBM VII, arXiv:2602.03650 §4 (verified in v2)

The 2022 baseline they were patching used a **scalar** σ per line:
> *"σᵢ is the S/N in the continuum region in the vicinity of the spectral line, and ν = N(Nepochs − 2) is the number of degrees of freedom, since each pixel in each of the two disentangled spectra is considered a free variable."*
> — Shenar et al. 2022, §3.1, arXiv:2207.07674

And Maxted et al. 2026 quote **one scalar** for the whole product: *"The signal-to-noise ratio of the disentangled spectra is approximately 250 near 650 nm."*

**[INFERENCE]** This is the strongest confirmation of the roadmap's central bet, and it also tells you how to sell it. Nobody *asks* for uncertainty bands, because no tool offers them. They report the symptom instead: K₂ that moves with the window, χ²ᵣₑd > 1, error bars known to be "lower bounds", "purely statistical" or "notoriously underestimated", extra diagnostics bolted on. **The headline should not be "we give you error bars" but "your K₂ stops depending on which window you chose, and here is why."**

### 3.4b How people currently estimate the uncertainty instead

Nobody in 2023–2026 does the Kıran et al. (2016) loop of adding white noise to a disentangled spectrum and refitting. What they do instead, in descending order of rigour:

1. **MCMC over K₁, K₂ propagated into the separated spectra** — the closest thing in the window to a real error band:
> *"This allowed us to quantify how the uncertainties in K₁ and K₂ propagate into the separated component spectra. The resulting variation in the disentangled spectra provides the uncertainties required for the subsequent atmospheric parameter determination, for which reliable error estimates are essential."*
> — Dervişoğlu et al. 2026, Z Vul §4.3, arXiv:2608.03348 (verified), using **FDBeMC** (Barbaros & Dervişoğlu 2023, an `emcee` wrapper around FDBinary)
2. **Bootstrap on the orbital elements only** — Pavlovski et al. 2023 (10 000 resamples), Moharana et al. 2026, Hełminiak's TODCOR errors.
3. **Leave-one-epoch-out bootstrap** — Seeburger et al. 2024 §3.1.
4. **Forward injection of synthetic SB2s through the RV pipeline** to calibrate phase-dependent blending bias — Jennings et al. 2023 §4.
5. **Genetic-algorithm population spread** — Serebriakova et al. 2025 §2.4, who warn that *"the confidence ellipses struggle with double or multiple minima and show large common uncertainty rather than uncertainty per minimum"*.

**[INFERENCE]** Option 1 is the state of the art and it captures only the *orbital* contribution — it holds the light ratio, the normalisation and the null space fixed, and it cannot see the low-frequency modes at all. `albireo`'s posterior contains all of those plus the cross-component correlation (the measured −0.992 EW anti-correlation). A figure comparing an FDBeMC-style K-only band against `albireo`'s full band on the same data would make the difference visible in one panel. Note also that **DERVIS (§6.4) is from this same group** — Dervişoğlu is a co-author on Z Vul and on Meng et al. 2026 — so their trajectory is: FDBinary → emcee wrapper → pure-Python rewrite. They are the group most likely to arrive at `albireo`'s position independently, and the most natural collaborators.

### 3.4c Reproducibility is now an admitted problem, and it is a *software* problem

The sharpest single finding in the whole survey. Two groups, **the same HERMES spectra and the same disentangling code**, materially different K₂ for V446 Cep:

> *"The agreement between our study and Tkachenko et al. (2024) is satisfactory except that the RV semi-amplitude of the secondary component (K₂) differs considerably. It is hard to trace the source of this inconsistency because the same observed spectra and the same disentangling code were used in both studies."*
> — Moharana, Southworth, Pavlovski, Miszuda, Rathour, Hełminiak, Marcadon, Bowman, Pawar & Tkachenko 2026, MNRAS §3.2.1, arXiv:2604.25056 (verified verbatim)

Their own diagnosis names three choices that appear in no published result: which spectra were cut for the Rossiter–McLaughlin effect (*"we only used the 40 out-of-eclipse spectra"*), whether Balmer lines were included (*"the spectral segments avoided prominent Balmer lines because the normalisation routine … was not ready to deal with the broad lines reliably"*), and which segment was used:

> *"The selection of the spectral segment also contributes to the discrepancy, as the line profiles vary substantially for different wavelength ranges"*
> — same, §3.2.1 (verified)

And segment choice is not a free parameter in practice:

> *"After some trials with short spectral segments with a width ∼100 Å, which did not return any stable solutions, a longer segment spanning λ = 4180−5150 Å was adopted"*
> — same, §3.2.1 (verified)

**[INFERENCE] This is the most actionable finding in the report.** The failure is not scientific, it is that the run configuration is not an artefact. `albireo` already has the two pieces needed: a TOML pipeline configuration that *is* the serialisable record, and `sensitivity_forecast` (D47), which is one short step from a "what changes if I move the segment?" report. Shipping a worked V446 Cep or KIC 9851944 example whose output includes a window-sensitivity panel would answer a published, named, unresolved discrepancy between two named groups — the kind of thing that gets a package cited.

### 3.4d Eclipse and Rossiter–McLaughlin spectra are discarded, not modelled

> *"One of the requirements of spd is that there should be no intrinsic line profile variations (LPVs) in the spectra of the components, except for the dilution effect in the course of the orbital cycle. … In our sample of HERMES spectra, 33 were obtained during an eclipse. These spectra were not used in spd, making the sample substantially smaller."*
> — Moharana et al. 2026, V446 Cep §3.2.1, arXiv:2604.25056 (verified) — **33 of 72 spectra thrown away**

> *"including the spectra obtained in the course of ingress or egress of the eclipses in which the line profiles are distorted due to the Rossiter-McLaughlin effect would be a violation"*
> — Dervişoğlu et al. 2026, Z Vul §4.3, arXiv:2608.03348 — 23 of 29 spectra survive

> *"It was found in practice that small line profile distortions due to non-radial modes would not seriously affect the RV curves … but if also radial pulsations are involved, another strategy in spd is needed"*
> — Moharana et al. 2026, §3.2.1 (verified)

**[INFERENCE]** Note the irony against §3.1: the in-eclipse spectra are *exactly* the ones that break the light-ratio degeneracy (Maxted 2026 zeroes the eclipsed star's rows in **M** to exploit them; Hełminiak et al. 2024 builds a whole observing programme around totality), and they are also the ones every other group discards. `albireo`'s per-epoch light fractions plus an RM-aware velocity treatment would let a user *keep* those 33 spectra and get the light ratio from them. That is a concrete, quantifiable pitch — "you threw away 46 % of your data and the discarded half is the half that measures your light ratio" — and it is currently nowhere in the documentation.

### 3.5 Nebular emission — verbatim confirmation of `albireo`'s D40 design

> *"While sky spectra were subtracted to reduce the nebular contamination, strong residuals remain, likely due to the variability of the emission across the Tarantula. This problem worsens by the fact that the nebular residuals are not constant, but vary from epoch to epoch in strength and shape in a non-trivial manner, presumably due to the varying weather and seeing conditions between the different epochs."*
> — Shenar et al. 2022, TMBM VI §3.4, arXiv:2207.07674 (verified via ar5iv)

> *"To partly account for the nebular contamination, we extended the shift-and-add algorithm to three components, where component C represents the nebular lines … Component C is enforced to be static, lie above the continuum, and achieve non-vanishing values only in regions overlapping with the nebular emission."*
> — same, §3.4 — **this is a hand-built version of `albireo`'s windowed nebular component, with the amplitude scaled rather than inferred**

> *"While our method for removing the residual nebular-line contamination works reasonably well … it does not fully remove the impact of nebular lines."*
> *"Generally, not accounting for nebular contamination leads to spurious features that are especially apparent in the amplified spectra of the faint secondaries."*
> — same, §3.4

> *"Nebular emission can lead to spurious features in the disentangled spectrum, and so the profile of the Balmer lines should be taken with caution."*
> *"However, since the secondary component is close to static (see below), the nebular contamination cannot be fully removed."*
> — Villaseñor et al. 2023, arXiv:2307.07766

> *"the spatial as well as temporal variability of the nebular line made it challenging to have no residuals after subtraction, and subsequently to obtain spectra appropriate for spectral disentangling."*
> — Deshmukh et al. 2026, TMBM VII, arXiv:2602.03650

> *"Others retained their SB1 status or were flagged as uncertain due to nebular contamination impeding the disentangling process."*
> — same

> *"the sky-subtracted science spectra are not corrected for nebulosity and therefore retain their nebular component"*
> — Shenar et al. 2024, BLOeM I §3.3, arXiv:2407.14593

And, decisively, from the field's newest code, describing a feature it does **not** have:

> *"In a future version we will implement the possibility of the subtraction of ISM emission lines such as Hα or [O iii] λ5007 being refined on an epoch-by-epoch basis, as different spectrograph apertures and seeing conditions provide different degrees of contamination."*
> — Maíz Apellániz et al. 2026, UNWIND, arXiv:2604.02111 §3.2

That is a verbatim description of `albireo`'s D40 component — static in the barycentric frame, free amplitude per exposure — written in April 2026 as a wish-list item by the authors of the newest disentangling code in the field.

### 3.6 Normalisation and rectification by hand

> *"Rectification is done using wavelength points provided by the user after a careful examination of the spectra. This is done because of the high sensitivity of disentangling to rectification: it may be time consuming but it reduces systematic errors."*
> — Maíz Apellániz et al. 2026, arXiv:2604.02111 §3.2

> *"The disentangled spectra were still in the common continuum of the binary system so needed to be renormalised to the continuum of the individual component stars."*
> — Pavlovski et al. 2023, A&A 671, A139, arXiv:2301.04215

> *"The accurate normalisation of the WR spectra to the continuum is a very difficult step"*
> — Gosset et al. 2026, WR 25, arXiv:2607.24390

> *"These telluric features in the original spectra were either removed in advance by dividing them by the spectrum of a rapid rotator or erased interactively by hand on the screen (if they are weak and not so many)."*
> — Takeda 2025, AR Aur, arXiv:2501.08054

### 3.7 Third light, triples, ISM/DIB contamination

> *"If an unconstrained component is added to the iterative procedure, the most likely outcomes are divergent non-physical solutions. For that reason, UNWIND allows only for fixed third lights"*
> — Maíz Apellániz et al. 2026, arXiv:2604.02111 §3.3

> *"We emphasize that no third-light contribution was included in the fit and we caution that the masses and radii … could be affected at the 10–20% level by unaccounted third light."*
> — MONOS III, arXiv:2508.04272 §4.1.6

> *"Furthermore, we found potential signals of contribution from a third star in the spectra of 15 SB2 systems."*
> — Villaseñor et al. 2025, BLOeM, arXiv:2503.21936

> *"That is one of the reasons why some previous disentangling efforts have been restricted to small wavelength regions instead of to the full spectral region observed."*
> — Maíz Apellániz et al. 2026, §3.3, on DIBs

> *"Since stationary features will bias the results of spectral disentangling and our inference of potential luminous companions, we removed these lines from the data by fitting and subtracting a Gaussian model of each line."*
> — El-Badry et al. 2025, arXiv:2509.01545 §2.1, on DIBs

### 3.8 Time-invariance is violated — the assumption the whole method rests on

> *"Spectral disentangling operates under the ansatz that all the observed spectra are produced by summing together the time-invariant spectra of two components"*
> *"Such variations violate the basic assumption of spectral disentangling – that the component spectra are time-invariant – and thus can be expected to introduce spurious features in the recovered spectrum of the secondary."*
> — El-Badry et al. 2025, arXiv:2509.01545 §3.3, §5.1

> *"While the MRS spectra also contain Hα, obvious changes in the strong emission line profile across epochs make it unsuitable for disentangling."*
> — same, §3.3

> *"Fully mapping the impact of all parameters and line-profile variability on the output of spectral disentangling is beyond the scope of our study, and would require a dedicated study with prespecified models of line variability."*
> — Shenar et al. 2022, TMBM VI, arXiv:2207.07674

> *"In the spectra disentangled with unwind there are strong residuals. This is likely caused by the typical line variability in Oe stars."*
> — MONOS III, arXiv:2508.04272 §4.1.1

> *"the fact that different lines follow different RV curves hampers our possibility to apply spectral disentangling to this system, and thereby prevents us from studying the chemical compositions of the stars."*
> — Rauw 2024, on Plaskett's star, arXiv:2411.00793

> *"Without improvements in our understanding of how disc-companion interactions alter the spectral profiles, spectroscopic analysis on Be stars will remain challenging."*
> — Janssens et al. 2025, arXiv:2512.15019

From the eclipsing/pulsator side, the same assumption failing:

> *"assumes that the line profiles do not change shape as a function of time — an assumption that is violated"*
> — Torres et al. 2025, 27 Tau §X, arXiv:2507.15933 (a He-weak, chemically spotted secondary)

> *"This shows that LPV due to the Rossiter-McLaughlin effect (possibly coupled with the pulsations) could be severe enough to affect the orbital solution obtained using spd."*
> — Moharana et al. 2026, V446 Cep §3.2.1, arXiv:2604.25056 (verified)

> *"Attempts to measure the Hα radial velocity variations through spectral disentangling were unsuccessful"*
> — Rivinius, Klement, Baade et al. 2025, A&A §3.3 (V742 Cas), arXiv:2412.09720

> *"the exact profiles of the Balmer lines in the component spectra may be influenced by emission and should be interpreted with caution"*
> — Müller-Horn et al. 2025, HIP 15429 §5, arXiv:2504.06973 — even Hγ/Hδ, chosen precisely to avoid emission

### 3.9 Too few epochs, low S/N, cost, weak K₂

> *"nine epochs distributed within three months are not efficient in detecting reliable periods with more than 45 days"*
> — Britavskiy et al. 2025, arXiv:2502.12239 §3.3

> *"nine epochs provide sufficient data to robustly estimate the multiplicity fraction and identify interesting systems, they are at the threshold of what is necessary to constrain orbital periods."*
> — Villaseñor et al. 2025, arXiv:2503.21936 §6.2

> *"The major limitation of our data is the S/N, which holds the key to confirm presence or absence of a luminous companion."*
> — Deshmukh et al. 2026, TMBM VII, arXiv:2602.03650 §5

> *"Although disentangling shows rather unambiguously that another luminous source contributes to the observed spectra, it does not produce a robust measurement of K2"*
> — El-Badry et al. 2025, arXiv:2509.01545 §3.4

> *"The fainter the secondary, the less of an impact K2 has on the disentangled spectra. In contrast, even small deviations in K1 can result in spurious features in the disentangled spectrum of the secondary. This typically results in a secondary spectrum that mimics that of the primary, which can lead to an erroneous detection of a non-degenerate companion."*
> — Shenar et al. 2022, TMBM VI Appendix A, arXiv:2207.07674 — **independent confirmation of `albireo`'s own D41 result** (a K₁ 10 % high took the recovered companion from 0.96 to 0.49 correlation with truth while more than tripling the detection statistic)

> *"This poses an especially significant challenge for large surveys such as Gaia or LAMOST, where the number of available spectra per target is often not enough for a proper spectral disentangling."*
> — Binnenfeld et al. 2025, arXiv:2507.12363

> *"Solving the system is typically a computationally demanding problem owing to the need to invert matrix M"*
> — Tkachenko et al. 2023, arXiv:2312.14009

> *"…optimising the model-data match over all possible primary velocities, mass ratios, and disentangled spectra is very time-consuming."*
> — Seeburger et al. 2024, arXiv:2405.19391

### 3.10 BLOeM says in print that it needs disentangling and does not have it

**The single most actionable pair of quotes in this report.**

> *"It is important to mention that in the case of SB2 systems, our RV measurements are potentially unreliable because we used only one stacked template to crosscorrelate the spectra with each other."*
> *"Thus, it is necessary to disentangle the spectra first in order to get true RV estimates of the companions and consequently reliable period estimates."*
> — **Britavskiy** et al. 2025, BLOeM B-supergiants, A&A 698, A40, arXiv:2502.12239 §3.1, §3.3

> *"This method is more robust than individual line-by-line and epoch-per-epoch line profile fitting but not as robust as spectral disentangling."*
> — Sana et al. 2025, BLOeM O stars, arXiv:2509.12488

> *"69 OB stars are excluded from the analysis owing to disk emission or significant contamination by secondaries in SB2 binaries"*
> — Bestenlehner et al. 2025, BLOeM pipeline properties, arXiv:2506.00117 §2 — **the SB2s are thrown out of the parameter pipeline entirely**

> *"for double-lined spectroscopic binaries (SB2), the process of co-adding the data will smear the spectral features of the two components"*
> — Shenar et al. 2024, BLOeM I §3.6, arXiv:2407.14593

> *"The previous line-by-line fitting method was effective for single-lined binaries (SB1s) but struggled with SB2s, particularly when the binary components had similar flux ratios."*
> *"However, challenges remain, particularly in cases with limited epochs, contributions from third objects, or significant profile variability."*
> — Villaseñor et al. 2025, arXiv:2503.21936 §3.1

> *"We applied the SB2 procedure outlined in Sect. 3 to 70 of the 91 SB2 systems, the other systems having a too small velocity separation, too little contrast between components, or an insufficient signal-to-noise ratio."*
> — Lennon et al. 2026, BLOeM vsini, arXiv:2512.12102 §4

### 3.11 Template mismatch

> *"This may lead to rather imperfect template match and in turn may cause systematic problems in TODCOR."*
> *"We have found the determination of the mass ratio to be very sensitive to the templates chosen for both components."*
> — Seeburger et al. 2025, arXiv:2511.13692 §4

> *"Template mismatch errors could produce wrong RVs, most probably the same systematic for all values within a time-series."*
> — Gosset et al. 2025, Gaia DR3 SB1 chain, arXiv:2410.14372 §2

> *"Note that no template spectra are needed for SPD, thus avoiding any biases due to template mismatch"*
> — Pavlovski et al. 2023, arXiv:2301.04215 — i.e. template-freedom is *sold* as a virtue of disentangling

**And the cleanest published measurement of how much worse the light ratio is than the velocity:**

> *"the TODCOR light ratio is more sensitive to the atmospheric parameters of the templates than the derived RVs"*
> — Jennings, Southworth, Pavlovski & Van Reeth 2023, KIC 9851944 §4, arXiv:2311.02064 — the same template perturbation moved K by **~0.1 %** and the light ratio by **~9 %**

> *"the 0.2% and 0.3% increase in the velocity semiamplitudes translates to a 0.6% and 0.9% increase in the derived masses, which is significant"*
> — same, §4

Blending near conjunction is handled by injection-and-correction:

> *"these RVs correspond to phases of conjunction, where line blending effects are most severe"*
> — same, §4 — they inject synthetic SB2 models through TODCOR, subtract the recovered-minus-true difference as a phase-dependent correction, and discard any point needing > 2.5 km s⁻¹

> *"line blending causes an underestimation of the RVs of the stars measured from H and diffuse He lines"*
> — Southworth 2026, *Rediscussion* 31 (CV Vel), arXiv:2606.04870

**[INFERENCE]** The 0.1 %-vs-9 % measurement independently validates `albireo`'s D55 finding that a wrong assumed light ratio becomes a wrong temperature. It is also the strongest available argument for `albireo.todcor`: a code whose light ratio comes from a *fitted, disentangled* component rather than a grid template should be materially less exposed to that 9 %.

### 3.12 The precision-RV regime (a genuinely different problem)

> *"as the two stars orbit one another, they produce a time-varying blending of their weak spectral lines … This makes an accurate measure of radial velocities difficult, producing a typical scatter of 10−15 m s−1. This extra noise prevents the detection of most orbiting circumbinary planets."*
> — Sairam, Triaud et al. 2023/2024, MNRAS 527, 2261, arXiv:2310.07527, abstract

> *"Spectrally disentangling both components of a binary system is hard to do accurately."*
> *"Binary mask cross-correlation methods also handle line blends badly and those regions are usually avoided."*
> — Sairam et al. 2024, BEBOP VI, arXiv:2410.02573

### 3.13 A failure mode no disentangler currently guards against

> *"When the MRS mode operates during bright nights, and the moon phase is nearly full Moon, stars polluted by moonlight may be misidentified as binaries."*
> — Wang, Kovalev, Xiong & Guo 2025, MNRAS 539, 3506 — **126 confirmed false-positive SB2s in LAMOST-MRS that are single stars contaminated by moonlight**

**[INFERENCE]** A near-zero-RV, solar-like second component is exactly what a disentangler will happily fit. A cheap guard — flag a recovered component whose velocity is pinned near zero and whose spectrum matches solar — would be a small feature with an outsized reputational payoff in the LAMOST/SDSS community.

### 3.14 And the sentence that frames the whole faint-companion problem

> *"It is important to realise that the method of disentangling, by nature, always returns a disentangled spectrum for a secondary star. It is then up to the user to decide whether the spectrum is 'flat', or whether it is of stellar origin. Whether or not the companion is classified as 'non-degenerate' is left as a subjective decision, based on a visual inspection"*
> — Shenar et al. 2022, TMBM VI §3.2, arXiv:2207.07674 (verified via ar5iv)

**[INFERENCE]** This is the strongest single argument for `albireo`'s calibrated detection statistic. The most-cited practitioner in the field states in print that the decision is *subjective and made by eye*. `albireo` replaces that with a likelihood ratio, a false-alarm probability measured by injection through the data's own operators, and a completeness limit at a stated confidence. That sentence belongs in the methods-paper introduction.

---

## 4. Case studies worth reading in full

### 4.1 Maxted et al. 2026 on AI Phe (arXiv:2607.17976) — the best available argument for `albireo`

A leading practitioner **wrote a new Simon & Sturm implementation from scratch in 2026** and hand-patched every degeneracy `albireo` handles structurally:

| What they did (verbatim from §3–4) | `albireo` equivalent |
|---|---|
| Own sparse `Mx = b` solved with LSQR | `MarginalOrbitModel`, banded precision |
| Inverse-variance weighting of epochs, added by hand | native |
| **Included in-eclipse UVES spectra and set the eclipsed star's elements of M to 0** to break the flux-ratio ambiguity | per-epoch light fractions (eclipse mode) |
| Grid-searched the flux ratio per HARPS order against in-eclipse residual rms; extrapolated a linear fit for one order with no coverage | wavelength-dependent light fractions |
| Interpolated all spectra onto a common log-λ grid at 4× resolution | **not needed** — native pixel grids |
| Ran per-order, stitched with linear scalings | not needed |
| Patched outliers with reconstructed values | masks / weights |
| Fitted and removed a quadratic from the A−B difference | the k = 0 null space, priored |
| Re-normalised with SUPPNET afterwards | response polynomials in-model |
| Reported one scalar S/N ≈ 250 | full posterior covariance |

**[INFERENCE]** Reproducing this paper's disentangled spectra *with a band* and *without the eight hand steps* is the single most persuasive artefact the project could produce: same system, same archival HARPS+UVES data, a published answer, and an author who publishes his benchmark spectra openly at `https://www.astro.keele.ac.uk/pflm/BenchmarkDEBS/`.

### 4.2 TMBM VII (arXiv:2602.03650) — injection–recovery detection limits, hand-built

Verified in v2:
> *"We performed an F-test in the 4072–4130 Å range to determine whether the model is a better fit to the retrieved spectrum than a featureless flat line, for which we used a 5% significance threshold."*
> *"For every dataset, we ran a companion search using disentangling. The disentangled secondary spectrum was retrieved at the same K₂ value that it was injected at"*

They tabulate secondary mass vs flux ratio (5 M☉ → 1.1 %, 6 → 1.6 %, 8 → 3.1 %, 10 → 5.1 %) and conclude *"We can therefore confidently rule out a ≳6 M⊙ luminous MS companion"*. This is `albireo.calibrate.detection_limit` reinvented one-off, and it is the most statistically careful example in the corpus.

**[INFERENCE] Non-detection limits are reported in four incompatible units across four papers.** Banyard 2023: *"flux ratios of about 1%–1.5%"*. Saracino 2023: *"an upper limit of 10% of a luminous primary source to the observed optical light"*. Deshmukh 2026: *"≳6 M⊙"*. Janssens 2023: *"the spectral signature of a non-degenerate stellar companion with such a low mass cannot be retrieved using our data"* — no number at all. Nothing standardises the conversion. A `detection_limit()` that reports **flux ratio, spectral type and mass together at a stated confidence, from the same injection machinery** would be an unusually legible contribution.

### 4.3 El-Badry et al. 2025 on ALS 8814 (arXiv:2509.01545) — three codes, one dataset, public data

They ran **the Seeburger 2024 code, fd3, and shift-and-add** on the same 26 public LAMOST-MRS spectra (R ≈ 7500, peak S/N ≈ 200) and compared. **[INFERENCE]** This is the cheapest credibility win available: the data are public, the comparison figure is published, and adding a fourth code that returns a posterior is a direct, fair, no-verdict comparison of exactly the kind `internal/roadmap.md` §9 prescribes.

### 4.4 Seeburger et al. 2024/2025 — the survey-scale alternative

Simon & Sturm wavelength space, sparse solve with **LSMR**, Tikhonov regularisation, automatic starting guesses. Its own honest limits:
> *"There is a trade-off: very small [Λ] will lead to very little regularisation … while very large will over-regularise the solution, leading to a loss of the original features."*
> *"We have found a suitable [Λ] for each (simulated) dataset by trial and error; automatisation of this is an endeavour for future work."*
> *"for a mass ratio of unity (twin star spectra) there is a degeneracy between the two components' velocities."*
> *"For the most extreme light ratios (top row), even with 'correct' input parameters, the disentangler struggles to recover the correct spectrum of the secondary."*
> — arXiv:2405.19391 §2.5.2, §3

**[INFERENCE]** "Choose Λ by trial and error" is exactly what `albireo`'s ML-II / empirical-Bayes estimation of `(τ, η)` removes. That is a concrete, quotable, apples-to-apples improvement over the newest survey-scale code, and it is not currently stated anywhere in `albireo`'s docs in those terms.

### 4.5 BEBOP — the niche `albireo` is *not* built for

**[INFERENCE]** DOLBY targets 10–15 m s⁻¹ systematics. `albireo`'s model is not obviously wrong for this, but nothing in the benchmark record demonstrates m s⁻¹ behaviour, and the community's requirements (cm s⁻¹ barycentric corrections, instrumental drift, activity indicators) are a separate engineering programme. **Recommend scoping this out explicitly**, as the roadmap already does for synthesis.

---

## 5. How many papers, and which codes — quantified

### 5.1 Papers naming disentangling in title or abstract (arXiv, astro-ph, binary-related)

| Year | Papers |
|---|---|
| 2022 | 8 |
| **2023** | **9** |
| **2024** | **6** |
| **2025** | **11** |
| **2026 (to 2026-09-03)** | **10** |

Full-year 2026 extrapolates to ~14. Longer series: 2014: 10 · 2015: 6 · 2016: 4 · 2017: 4 · 2018: 9 · 2019: 4 · 2020: 7 · 2021: 4 · 2022: 8.

**This confirms the roadmap's "roughly eight papers a year named in their abstract alone", and shows the rate is now rising**: the 2025+2026 average (10.5/yr) is ~50 % above the 2019–2022 average (5.75/yr). **[INFERENCE]** The rise tracks the compact-object-candidate boom plus the arrival of BLOeM/M3W-scale surveys.

### 5.2 Better proxy — citations per year to the method papers (OpenAlex)

| Method paper | total | 2023 | 2024 | 2025 | 2026* |
|---|---|---|---|---|---|
| Zucker & Mazeh 1994, TODCOR | 477 | 17 | 14 | 13 | 8 |
| González & Levato 2006, shift-and-add | 179 | 12 | 11 | 9 | 5 |
| Rucinski 2002, broadening functions (AJ 124, 1746) | 172 | 5 | 6 | 11 | 5 |
| Czekala et al. 2017, PSOAP | 53 | 5 | 3 | 4 | 1 |
| Sablowski & Weber, Spectangular | 11 | 0 | 1 | 0 | 0 |
| Seeburger et al. 2024 | 7 | — | 1 | 5 | 1 |

*2026 partial. Simon & Sturm 1994, Hadrava 1995 and Ilijić et al. 2004 are unresolvable in OpenAlex (pre-DOI); their true counts exceed the numbers above.

**Reading:** the *method* is used by roughly **20–30 papers a year**; TODCOR-style velocity extraction by ~15/yr; broadening functions ~6–11/yr and rising. Spectangular is dead (≤1 citation/yr).

### 5.3 The ten most-cited disentangling papers, 2020–2026

OpenAlex counts, read 2026-09-03. Ranking reliable; absolute values are lower bounds vs ADS.

| # | Cites | Paper | DOI |
|---|---|---|---|
| 1 | 132 | Shenar et al. 2020, *The "hidden" companion in LB-1 unveiled by spectral disentangling*, A&A 639, L6 | 10.1051/0004-6361/202038275 |
| 2 | 108 | Bodensteiner et al. 2020, *Is HR 6819 a triple system containing a black hole?*, A&A 641, A43 | 10.1051/0004-6361/202038682 |
| 3 | 80 | Mahy et al. 2022, *Identifying quiescent compact objects in massive Galactic single-lined spectroscopic binaries*, A&A | 10.1051/0004-6361/202243147 |
| 4 | 62 | El-Badry et al. 2022, *Unicorns and giraffes in the binary zoo: stripped giants with subgiant companions*, MNRAS | 10.1093/mnras/stac815 |
| 5 | 46 | Zúñiga-Fernández et al. 2020, *Search for associations containing young stars (SACY)*, A&A | 10.1051/0004-6361/202037830 |
| 6 | 27 | Shenar et al. 2021, *The Tarantula Massive Binary Monitoring*, A&A | 10.1051/0004-6361/202140693 |
| 7 | 26 | Sekaran et al. 2020, *Tango of celestial dancers: detached eclipsing binaries with pulsating components*, A&A | 10.1051/0004-6361/202038989 |
| 8 | 25 | Janssens et al. 2023, *MWC 656 is unlikely to contain a black hole*, A&A 677, L9 | 10.1051/0004-6361/202347318 |
| 9 | 25 | Villaseñor et al. 2023, *B-type Binaries Characterisation Programme II: VFTS 291*, MNRAS | 10.1093/mnras/stad2533 |
| 10 | 21 | Pavlovski et al. 2023, *High-mass eclipsing binaries: a testbed for models of interior structure and evolution*, A&A 671, A139 | 10.1051/0004-6361/202244980 |

Near misses: Yuan et al. 2022 (20), Maíz Apellániz et al. 2020 *Lucky spectroscopy* (20), Saracino et al. 2023 (19), Lavail et al. 2020 (16).

**Eight of the top ten are compact-object or massive-star papers.** **[INFERENCE]** Citation impact in disentangling is concentrated almost entirely in "is the dark companion real?". `albireo`'s SB1 K₂ scan with a calibrated false-alarm probability is aimed at the highest-impact question in the field, and the project under-advertises it relative to general disentangling.

---

## 6. Competition and adjacent software

All repository metadata read **2026-09-02/03** from `api.github.com`, commit Atom feeds, or raw-file probes.

### 6.1 The one-line summary

**Of every code audited, only PSOAP (dormant since 2017) has ever advertised posteriors on the component spectra.** fd3, KOREL, Spectangular, UNWIND, BiSpeD, DERVIS, Seeburger's code, MESS and DOLBY all deliver point estimates. TODCOR/BF codes give RV and flux-ratio errors but no spectra. **`albireo` is the only maintained code in the field that returns an uncertainty on the disentangled spectrum.**

### 6.2 BiSpeD — the most direct competitor, and it is alive

| | |
|---|---|
| What | Python library for disentangling and **SB1 faint-companion search**. `spbina` (iterative shift-and-add; user-supplied flux ratio `frat`, iteration count `nit`, sigma clipping), `find2c` (grid over mass ratio *q*, subtract primary, cross-correlate residual against synthetic templates, read *q* and secondary Teff off the CCF maximum), `rvbina`/`setrvs` (FFT cross-correlation RVs, SB1 and SB2), `qfitg` (Gaussian fit to the correlation peak), `onecomp`, `hselect`, `splot` |
| Paper | Martínez & González 2025, *BiSpeD: Binary Spectral Disentangling applied to binary low-mass star searches*, A&A, doi:10.1051/0004-6361/202451496 |
| Repo | https://github.com/israelmarti/BiSpeD |
| Language / licence | Python 3.12 / **MIT** |
| Last activity | **pushed 2026-09-02** (v1.7, multiple commits that day; v1.4 was 2026-03-13) |
| Distribution | `pip install git+https://github.com/israelmarti/bisped@v1.7`. Its PyPI package `bisped` is **stale at 1.1 (2023-03-08)**, so the README's "distributed on PyPI" claim is out of date |
| Adoption | 1 star, 1 fork |
| Uncertainties on spectra | **No** |
| Interface | IRAF-era: `@file` lists of FITS images, `wreg='4000-4320,4360-4850'` strings, `interac=True` |

**[INFERENCE] This is the finding the maintainer most needs.** Same language, same permissive licence, actively developed *yesterday*, published in A&A this year, targeting `albireo`'s highest-impact use case. Its weaknesses are precisely `albireo`'s strengths: hand-set flux ratio, CCF-maximum detection with no null distribution and no injection calibration, no posterior, no uncertainty band, IRAF-era interface. Its authors are the same group whose SB1 method `docs/science.md` already cites (González, Martínez & Alejo 2024, A&A 690, A124). **Cite BiSpeD, benchmark against it on the SB1 scan, and state plainly what a calibrated false-alarm probability adds.** At 1 star, the window is open.

### 6.3 UNWIND — the newest code, and it is not distributed

Maíz Apellániz et al. 2026, M3W I, A&A accepted (v3 2026-07-22), arXiv:2604.02111.

- **Language: IDL**, in the OWN/MONOS lineage.
- **Distribution, verbatim from v3 §3.1:** *"UNWIND is at this point a beta software. We have been able to run it in different computers under Mac OS and Linux but it currently lacks a manual or an installation package, so its dependencies have to be solved one by one. Eventually, we expect to address those issues but currently it can only be shared without support."* (v1 said only *"The long-term idea is to make UNWIND public."*) No repository, no archive, no licence.
- **Algorithm:** shift-and-add of González & Levato (2006), extended to ≥3 components. RVs and flux fractions are **user-supplied, not fitted**; an outer loop is required. Flux fractions constant across epochs. User-supplied rectification points.
- **Genuinely novel, and hard to replicate:** an integrated telluric + standard-ISM + **631-DIB library over 4000–17 100 Å**, built specifically so that disentangling can run over the whole optical–NIR rather than in narrow windows.
- **No uncertainty on the disentangled spectra anywhere in the paper.**

**[INFERENCE]** The strongest *scientific* competitor and the weakest *software* competitor. Its DIB/telluric library is the one capability `albireo` cannot match and should not try to build; the right response is interoperability — accept an externally supplied ISM/DIB template as a static component, which `albireo` already structurally supports, since a DIB spectrum is the telluric component with a different rest frame.

### 6.4 DERVIS — brand new, and a warning sign

- https://github.com/dervisoglu/dervis — *"Disentangling Engine for Radial Velocity Influenced Spectroscopy"*.
- Verified from the commit Atom feed: **initial commit 2026-06-24, last commit 2026-07-22, four commits total.**
- Verified by raw-file probe: **no `LICENSE`, no `setup.py`, no `pyproject.toml`** (all 404) — so it is not installable and is unlicensed.
- README: *"a pure-Python rewrite"* merging **FDBinary (Fourier) and CRES (wavelength domain)** into one library, removing the C dependency, with `emcee` MCMC + Powell wrappers, HDF5 chain backends, chunk-by-chunk SVD reconstruction to bound memory, direct FITS reading, ASCII export of separated components. Paper cited as *"Dervisoglu et al., 2026 (To be submitted)"* — no preprint located.
- **Uncertainties: orbital-parameter posteriors only.** The worked example draws median parameters from the chain, solves once, and writes two-column `(wave, flux)` ASCII.

**[INFERENCE]** DERVIS matters less as a competitor than as evidence, and its authorship is the interesting part. Dervişoğlu is a co-author on Z Vul (arXiv:2608.03348) and Meng et al. 2026 (arXiv:2605.13212), and the author of **FDBeMC** (Barbaros & Dervişoğlu 2023), the `emcee` wrapper that produces the closest thing in the literature to an error band on a disentangled spectrum (§3.4b). Their trajectory is FDBinary → emcee wrapper → pure-Python rewrite. **They are the group most likely to arrive at `albireo`'s position independently, and therefore both the most credible competitor and the most natural collaborator.** DERVIS is also the third new disentangling code in two years: `albireo`'s window as the only maintained option is closing.

### 6.4b Other codes named in the eclipsing-binary literature

- **FDBeMC** (Barbaros & Dervişoğlu 2023) — `emcee` around FDBinary, sampling K₁/K₂ within their RV error bars and propagating the spread into the separated spectra. **The state of the art for "an error bar on a disentangled spectrum", and it captures only the orbital contribution.** No repository located.
- **`starfit`** (Kolbas et al. 2015) — genetic algorithm + MCMC errors, the Zagreb group's atmospheric fitter for disentangled spectra.
- **`RaveSpan`** (Pilecki et al.) — the Araucaria workhorse: Rucinski broadening functions with Coelho templates, **plus a González & Levato shift-and-add disentangler**. Used to measure spectroscopic V-band light ratios from BF peak strengths, explicitly to break the photometric light-ratio ↔ radius-ratio correlation.
- **`v2fit`** (Konacki et al. 2010) — Levenberg–Marquardt double-Keplerian, Hełminiak's orbit fitter.
- **`XTgrid`** (Németh, Vos & Cabezas 2025, CAOSP 55/3, 340) — *template-based* decomposition of composite spectra for non-eclipsing hot-subdwarf binaries. Their framing of the problem: *"These entangled parameters in composite spectra lead to serious degeneracies and systematic errors."*
- **`SUPPNET`** (Różański et al. 2022, github.com/RozanskiT/suppnet) — neural continuum normalisation, used by Maxted to re-normalise disentangled spectra after the fact.

### 6.5 MINATO / `ravel` — the best integration target in the field

- https://github.com/jvillasr/MINATO — Python/Jupyter, **MIT**, last push **2026-07-22**, 3 stars. **Depends on `jax` and `numpyro`.**
- `span`: simultaneous synthetic-model fitting → Teff, log g, vsini, He/H, **and the light ratio**. `ravel`: SB1/SB2 line-profile RVs, Lomb–Scargle with FAP.
- **No disentangling module, none planned.**

**[INFERENCE]** Same stack, same licence class, same community, complementary scope — and its own survey has published that it needs disentangling (§3.10). A worked example taking a BLOeM SB2 from `ravel` velocities into `albireo`'s `Disentangler(velocities=...)` path and back out as TODCOR templates is a two-day job with a named audience.

### 6.6 The incumbents

| Code | URL | Lang | Licence | Last activity | Unc. on spectra |
|---|---|---|---|---|---|
| **shift-and-add** (Shenar) | github.com/TomerShenar/Disentangling_Shift_And_Add | Python | **none** | pushed **2024-03-07**; 10 stars | no |
| **fd3** (Ilijić) | sail.zpf.fer.hr/fdbinary/ (the advertised `/fd3` URL **404s**; HTTP-only, HTTPS refuses) | C + GSL | **none** (no LICENSE/COPYING in the tarball) | tarball `Last-Modified` **2014-07-25**; version string *"fd3 v.3.1 (Florianapolis, 25 July 2014)"* | no |
| **FDBinary** | ascl.net/1705.011 | C | — | **explicitly deprecated**: *"This code has been replaced with the newer fd3"* | no |
| **CRES** | sail.zpf.fer.hr/cres/ | C | — | `Last-Modified` **2006-06-16** | no |
| **KOREL / VO-KOREL** | stel.asu.cas.cz/vo-korel/ (live, HTTP 200) | Fortran | not stated | *"Current version … is KOREL11b - release 9. 9. 2011"*; service page stamp 2019-01-15 | not verified |
| **Spectangular** | github.com/DPSablowski/Spectangular | C++/Qt | **Apache-2.0** | pushed **2023-10-05**; no releases; 8 stars | not verified |
| **PSOAP** | github.com/iancze/PSOAP | Python | **MIT** | pushed **2017-12-06**; 33 stars; not archived; not on PyPI | **yes** (ASCL 1705.013: *"the posteriors of the orbital parameters and the spectra themselves"*) |
| **Seeburger 2024** | — | Python | — | **no public code**: *"available from the corresponding author upon request"*; nothing in the author's GitHub | no |
| **DOLBY** (Sairam/Triaud) | — | not stated | — | **no repository URL in the paper** | no (RV errors only) |
| **GSSP** | fys.kuleuven.be/ster/meetings/binary-2015/gssp-software-package | Fortran 90 | not stated | page states last update **2016-11-29**; tarballs + FTP; **no git repo anywhere** | parameter errors only |
| **iSpec** | github.com/marblestation/iSpec | Fortran/Py/Cython | **AGPL-3.0** | pushed **2026-04-15**; latest release still v2023.08.04; 38 stars | **does not do SB2** |
| **PHOEBE** | github.com/phoebe-project/phoebe2 | Python | **GPL-3.0** | 2.5.4 on **2026-08-28**; 105 stars | forward-models line profiles (`lp`); **no disentangling** |
| **specutils** | github.com/astropy/specutils | Python | BSD-3 | v2.4.0 2026-06-01 | **no SB2 decomposition** |

Third-party wrappers, all unmaintained or unlicensed: `ayushmoharana/fd3_initiator` (Python, no licence, pushed 2025-06-26), `simonkrekels/fd3-helper` (2019), `maurcabezas/pyKorel` (MIT, 2025-05-01), `robert-klement/KOREL_tools` (no licence, 2020), `aplidinio/Spectral-Disentangling` (Jupyter, no licence, 2020-07-14).

### 6.7 The RV / TODCOR neighbourhood, where things *are* moving

| Code | URL / ref | Licence | Last activity | Notes |
|---|---|---|---|---|
| **MESS** | arXiv:2602.05610 (Nachmani, Faigler, Mazeh 2026) | — | 2026-02 | *"MESS extends the two-dimensional TODCOR approach to a global multi-epoch formalism, deriving the radial velocities of both components at each epoch while optimizing the templates jointly across all observations."* Template search over **eight parameters: Teff, log g, vsini per star, common [M/H], and the flux ratio**. Detection floor α ≈ 0.1 at S/N 50, K down to ~10 km s⁻¹. A LAMOST DR11 application is announced. |
| **saphires** | github.com/tofflemire/saphires | MIT | pushed 2025-06-26; PyPI 0.1.17 (2024-11-05) | BF + CCF + **TODCOR** + SB2 flux ratios + vsini. The most alive BF code. |
| **Simchon/TODCOR** | github.com/Simchon/TODCOR | MIT | pushed **2025-10-26** | The only TODCOR repo GitHub search returns. 2025-10-26 commit *"Add RV and alpha error estimates"*. Corrected α-fitting (rejects non-physical negative α) and better edge handling. |
| **SPARTA** | github.com/SPARTA-dev/SPARTA | MIT | commits **2026-07-22/23** | *"todcor,tiravel: add SB2 and template-independent correlation"* + *"Modernize SPARTA: current stack, specutils, RV fix, test suite"*. |
| **agent4binary** | github.com/seratsaad/agent4binary + arXiv:2608.10866 | MIT | created 2026-07-15, pushed **2026-08-30** | Repackages El-Badry+2018 SB2 decomposition as MCP tool servers + a Skill; 41 466 APOGEE DR19 SB2 candidates. Its own abstract: *"applying one to a new data release is limited mostly by operating know-how that is rarely written down"*. Notable as *methodology packaging*, not new math. |
| **apogeesb2** (Kounkel) | github.com/mkounkel/apogeesb2 | MIT | pushed **2023-03-17** | CCF-based SB2 detection; component RVs, not spectra. (The name `apogee_sb2s` does not exist.) |
| **binspec** (El-Badry) | github.com/kareemelbadry/binspec | Jupyter | **none** | pushed **2018-04-16** | `tingyuansen/binspec_plus` (no licence) pushed 2021-03-24. No newer El-Badry-authored tool. |

**[INFERENCE] MESS is the direct competitor to `albireo.todcor` + `albireo.match` combined.** It does multi-epoch TODCOR with joint template optimisation over exactly the labels `albireo.match` fits, plus the flux ratio. Its stated floor (α ≈ 0.1 at S/N 50, K ≈ 10 km s⁻¹) is a concrete benchmark to cite or beat. `albireo`'s advantages over it are the disentangling-derived templates (no synthetic grid needed), the per-component zero point stated rather than absorbed, and the fact that it also returns the spectra.

### 6.8 Machine learning

Binnenfeld et al. 2025, arXiv:2507.12363 (astro-ph.IM), explicitly framed as an **alternative to** disentangling:
> *"The proposed tool uses deep neural networks to extract the stellar parameters of the individual component spectra that comprise the single exposure, without explicitly disentangling them or extracting their radial velocities."*

Motivated by the epoch-count problem in §3.9, trained on simulated Gaia RVS spectra. **[INFERENCE]** The competitive threat in the *survey* segment (few epochs per star), not the *benchmark* segment. It does not compete where the science needs an actual component spectrum for an atmosphere code.

Also worth tracking: **Lux** (Horta, Price-Whelan, Hogg, Ness, Casey, arXiv:2502.01745) — a generative multi-output latent-variable model **built on JAX**, designed to *"properly account for uncertainties and missing data"*. Not binary-specific and not a disentangler, but it is the nearest JAX-native neighbour in the Payne/Cannon lineage.

### 6.9 Confirmed absences

These are as valuable as the presences, and each was checked:

- **No new ASCL disentangling entry since 2017.** `ascl.net/code/search/disentangling` returns exactly four records: fd3, FDBinary, and two unrelated. Of the **250 most recent ASCL entries (207 of them from 2025/2026), not one is a disentangling or SB2 code.** KOREL, BiSpeD and DERVIS have no ASCL entries at all.
- **No JAX or PyTorch spectral disentangling code exists** other than `albireo`. GitHub searches for `disentangling+spectroscopic+binary`, `disentangle+binary+star+spectra`, `spectral+decomposition+binary+star` and `shift+and+add+disentangling` returned 2, 0, 0 and 1 results respectively.
- **No blind-source-separation work in stellar astrophysics**: `abs:"blind source separation" AND cat:astro-ph.SR` since 2024-06 → zero hits. No VAE-based stellar-binary decomposition found.
- **No dedicated Gaia DR4 SB2 preparation tool** has been published.
- **No public code** for Seeburger et al. 2024 or for DOLBY.
- **No GSSP repository** anywhere.

**[INFERENCE] One SEO note.** "Spectral disentangling" is heavily overloaded outside astronomy — hyperspectral imaging, time-series forecasting, representation learning, audio. Keep "spectroscopic binaries" adjacent to "disentangling" in every title, abstract, README heading and PyPI keyword set, or the package will be buried under machine-learning papers.

---

## 7. The twelve capabilities users most plausibly want

Ordered by strength of evidence.

### 1. An uncertainty band on the disentangled spectrum, propagated into the atmospheric parameters

**Evidence.** *"For disentangled spectra … observational uncertainties are not well-defined. The disentangled spectra are derived products that depend on the orbital solution, meaning their quality (and thus their uncertainties) evolves dynamically throughout the optimisation process"* (Serebriakova, Tkachenko, Johnston, Pavlovski & Aerts 2025, arXiv:2507.10096 — an open problem stated by the group with the most FDBinary experience in the field). Plus: *"the best-fit K2 varies with the spectral region being fit"* (El-Badry 2025); *"lower bounds on the true uncertainty"* (Kemp 2024); *"purely statistical errors"* (Torres 2025); *"notoriously underestimated"* (Jennings 2023); *"σmeas … are underestimated"* (Janssens 2025). The state of the art is Deshmukh 2026's *"wavelength-dependent error propagation"* (input weighting, not output uncertainty) and FDBeMC's K₁/K₂-only propagation. **Only PSOAP, dormant since 2017, ever offered a posterior on the spectra.**

**`albireo`: Yes, uniquely among maintained codes.** Posterior covariance, plus `export_draws` correlated across wavelength and across components — and, crucially, an uncertainty that *does* evolve with the orbital solution, which is exactly the property Serebriakova et al. say does not exist. Under-sold: the README phrasing does not connect to any symptom an astronomer recognises.

### 2. Light fractions declared, per-epoch, and wavelength-dependent

**Evidence.** Fifteen separate verbatim quotes in §3.1–3.2, from a dozen groups over 2022–2026, including both new codes (UNWIND assumes constant and Teff-similar; BiSpeD takes `frat` as a float). *"the wavelength dependent light ratio l_f(λ) between the components must be known"* (Dervişoğlu 2026). Photometric and spectroscopic values *disagree* in real systems (Rauw 2024 on 29 CMa; Kemp 2024's 0.92–0.94 vs 0.84 ± 0.03; Southworth's *"The discrepancy remains unexplained"*). Maxted 2026 grid-searches it per HARPS order; Seeburger 2025 makes λ-dependence a headline product. And the same template perturbation moves K by 0.1 % but the TODCOR light ratio by 9 % (Jennings 2023).

**`albireo`: Yes for declared and per-epoch** (no default; per-epoch simplex for eclipses). **Partial for wavelength-dependent** — `albireo.match` fits a shared radius ratio giving λ-dependent light fractions, but the disentangling itself conditions on constant ones. **The clearest remaining gap**, and the one with the widest constituency: it is simultaneously the massive-star complaint, the eclipsing-binary complaint, and Southworth's bottleneck. Note that Torres et al. 2025 §V independently reach `albireo`'s own shared-radius-ratio parameterisation, so the design is already community practice — cite Tkachenko (2015) Eqs. 3–4.

### 3. A nebular / ISM component with a free per-epoch amplitude

**Evidence.** *"the nebular residuals are not constant, but vary from epoch to epoch in strength and shape in a non-trivial manner, presumably due to the varying weather and seeing conditions"* (Shenar 2022, arXiv:2207.07674) — and the corresponding hand-built static "component C". UNWIND's own future-work list, verbatim. TMBM VII flagging SB1s *"as uncertain due to nebular contamination impeding the disentangling process"*.

**`albireo`: Yes (D40).** No competitor has it; the newest competitor has publicly asked for it; the most-cited practitioner built a manual version. **This belongs in the methods-paper abstract.**

### 4. Reliable SB2 radial velocities for survey data, with the zero point stated

**Evidence.** *"in the case of SB2 systems, our RV measurements are potentially unreliable … it is necessary to disentangle the spectra first in order to get true RV estimates of the companions and consequently reliable period estimates"* (Britavskiy 2025, A&A 698, A40). *"not as robust as spectral disentangling"* (Sana 2025). *"69 OB stars are excluded from the analysis"* (Bestenlehner 2025). Gaia discarded ~40 000 SB2 combined RVs (Katz).

**`albireo`: Yes** (D42 free RV table, D56/D57 TODCOR, D58 pipeline), with the per-component zero point stated on every table — exactly the discipline the quoted problem needs. **The highest-leverage unexploited asset in the package.** Competition arriving: MESS (§6.7).

### 5. A calibrated faint-companion detection limit with a false-alarm probability

**Evidence.** Eight of the ten most-cited papers 2020–2026 are compact-object papers. *"the method of disentangling, by nature, always returns a disentangled spectrum for a secondary star … Whether or not the companion is classified as 'non-degenerate' is left as a subjective decision, based on a visual inspection"* (Shenar 2022). TMBM VII hand-builds an F-test injection framework at a 5 % threshold. Limits are reported in four incompatible units across four papers (§4.2).

**`albireo`: Yes (D41)**, including the K₁-marginalisation result — which Shenar independently confirms: *"even small deviations in K1 can result in spurious features in the disentangled spectrum of the secondary. This typically results in a secondary spectrum that mimics that of the primary, which can lead to an erroneous detection of a non-degenerate companion."* **[INFERENCE]** Report the limit simultaneously as flux ratio, spectral type and mass, at a stated confidence, so it is comparable across the four conventions in the literature.

### 6. Not resampling the data; masks, chip gaps and mixed instruments handled natively

**Evidence.** Maxted 2026 interpolates onto a common 8192-px log-λ grid *per HARPS order* and stitches with linear scalings, because *"Accurate recovery … requires that the flux scale of the spectra is consistent across the entire set of spectra. This is difficult to achieve for the full spectral range covered by HARPS"*. UNWIND resamples. M3W combines six spectrographs from R = 25 000 to 115 000 in one analysis. Gaia DR4's RVS fluxes are pre-interpolated upstream.

**`albireo`: Yes.** Native pixel grids, per-instrument LSFs, per-epoch response polynomials. The six-spectrograph M3W case is a demo nothing else handles cleanly.

### 7. Automatic regularisation and continuum handling instead of hand-tuning

**Evidence.** *"We have found a suitable [Λ] for each (simulated) dataset by trial and error; automatisation of this is an endeavour for future work"* (Seeburger 2024). *"Rectification is done using wavelength points provided by the user … it may be time consuming"* (UNWIND). *"A workaround is to enforce the disentangled spectra to lie below the continuum"* (Shenar 2022) — an ad-hoc regulariser where a prior belongs. Maxted's SUPPNET pass; Pavlovski's renormalisation step.

**`albireo`: Yes for regularisation** — ML-II/empirical-Bayes estimation of `(τ, η)` is precisely the automation Seeburger names as future work, and this comparison is not stated anywhere in the docs. **Partial for continuum**: response polynomials and `smooth_angstrom` exist, but the *claim* is not benchmarked ("you need not hand-place rectification points, and here is the measured penalty for a wrong continuum with and without the response model").

### 8. Third light, triples and ≥3 components

**Evidence.** *"If an unconstrained component is added to the iterative procedure, the most likely outcomes are divergent non-physical solutions. For that reason, UNWIND allows only for fixed third lights"* (arXiv:2604.02111). *"masses and radii … could be affected at the 10–20% level by unaccounted third light"* (MONOS III). 15 BLOeM SB2s with third-star signals, plus 3 SB3s. The compact-hierarchical-triple series; WR 25 as a triple.

**`albireo`: Yes** — SB3 and hierarchical triples as nested Keplerians, solved simultaneously rather than by outer iteration. **But the façade *refuses* hierarchical triples (D46)**, so it is expert-path only. Worth closing, and worth saying out loud that the outer loop is unnecessary.

### 9. Time-variable component spectra (discs, pulsations, winds, irradiation)

**Evidence.** *"Such variations violate the basic assumption of spectral disentangling – that the component spectra are time-invariant"* (El-Badry 2025, on a Be+BH candidate). *"Fully mapping the impact of … line-profile variability … is beyond the scope of our study"* (Shenar 2022). *"assumes that the line profiles do not change shape as a function of time — an assumption that is violated"* (Torres 2025, 27 Tau). *"strong residuals … likely caused by the typical line variability in Oe stars"* (MONOS III). *"the fact that different lines follow different RV curves … prevents us from studying the chemical compositions of the stars"* (Rauw 2024, Plaskett's star). *"Attempts to measure the Hα radial velocity variations through spectral disentangling were unsuccessful"* (Rivinius 2025). *"Without improvements in our understanding of how disc-companion interactions alter the spectral profiles, spectroscopic analysis on Be stars will remain challenging"* (Janssens 2025).

A related and cheaper sub-case: **eclipse and Rossiter–McLaughlin spectra are discarded rather than modelled** — 33 of 72 HERMES spectra dropped for V446 Cep (arXiv:2604.25056), 6 of 29 for Z Vul (arXiv:2608.03348) — while those same in-eclipse spectra are exactly what breaks the light-ratio degeneracy (Maxted 2026; Hełminiak et al. 2024).

**`albireo`: No for the general case** — Tier 3. **Partial for the eclipse sub-case**: per-epoch light fractions already model the dilution change, though the RM velocity distortion is not modelled.

**[INFERENCE] Promote this to Tier 2.** The top-cited papers are Be + compact-object systems; it is the named blocker in the newest of them; the nebular component is already rank-one time variation, so the extension is a change of basis inside the linear-Gaussian family rather than a change of method; Shenar's static "component C" is the only thing anyone uses and it was built for nebulae, not discs; and nobody else can do it at all. The eclipse sub-case is cheaper still and comes with a quantifiable pitch: *you discarded 46 % of your spectra, and the discarded half is the half that measures your light ratio.*

### 10. An installable, licensed, documented package whose run configuration is a reproducible artefact

**Evidence, part one — installability.** Six of 36 groups wrote their own code. UNWIND: *"it can only be shared without support"*. Shenar's shift-and-add: no licence, last push 2024-03. fd3: no licence, 2014, advertised URL 404s, HTTPS refuses. Seeburger's code and DOLBY: unreleased. GSSP: no repository. BiSpeD: `@file` lists of FITS images and a stale PyPI package. DERVIS: no LICENSE, no setup.py.

**Evidence, part two — reproducibility, and this is the sharper half.** *"It is hard to trace the source of this inconsistency because the same observed spectra and the same disentangling code were used in both studies"* (Moharana et al. 2026 on V446 Cep, arXiv:2604.25056). The three responsible choices — RM cuts, Balmer inclusion, segment selection — appear in no published result. And `agent4binary`'s abstract makes the same point about a different pipeline: *"applying one to a new data release is limited mostly by operating know-how that is rarely written down"* (arXiv:2608.10866).

**`albireo`: Yes for installability** — BSD-3, CI, docs site, `load_example()`, CLI, `py.typed`, the cleanest in the field. **But it is not on PyPI**, while the README's Command-line section says `pip install "albireo[io,plots]"` and the Installation section says it is not available. **Fix that contradiction first.** **Yes in principle for reproducibility** — the TOML pipeline config *is* the serialisable record — but this is nowhere claimed, and `sensitivity_forecast` (D47) is one short step from the "what changes if I move the segment?" report that would answer the V446 Cep discrepancy directly.

### 11. Batch operation over a survey's worth of stars

**Evidence.** BLOeM: 929 stars, ~150 SB2s, orbital solutions still future work. Gaia DR4 on 2026-12-02. LAMOST-MRS: 12 426 SB2 candidates. APOGEE DR19: 41 466. *"optimising the model-data match over all possible primary velocities, mass ratios, and disentangled spectra is very time-consuming"* (Seeburger 2024).

**`albireo`: Yes** (D58 pipeline + CLI, worker processes, failures recorded; 2.0× at four workers, 2.5× at eight). **[INFERENCE]** The missing number is systems-per-GPU-hour, which the roadmap already flags as the open acceptance gate. Until it exists, the survey segment is a claim rather than a demonstration — and `agent4binary`'s point that *"applying one to a new data release is limited mostly by operating know-how that is rarely written down"* is a direct hit on that gap.

### 12. Interoperability with the atmosphere codes, carrying the uncertainty across

**Evidence.** Pavlovski's renormalise-then-fit relay; Maxted's iSpec → TelFit → own code → SUPPNET → SME chain; UNWIND handing off to FASTWIND; XShootU VIII merging PoWR models by an assumed luminosity ratio; *"no template spectra are needed for SPD, thus avoiding any biases due to template mismatch"* as a selling point.

**`albireo`: Yes** (`albireo.handoff`: `write_gssp`, `write_ispec`, `export_draws`). The measured result that white-noise resampling understates integrated EW uncertainty by 1.8–3.4×, and that the two components' EWs are anti-correlated at −0.992, is a publishable methods result on its own.

### Honourable mentions

- **A DIB / ISM absorption library** so disentangling can run over the full optical–NIR (UNWIND's real innovation). `albireo` has the structural slot; **accept an externally supplied template rather than building the library.**
- **A contamination guard** — flag a recovered component pinned near zero velocity with a solar-like spectrum (126 LAMOST moonlight false positives, §3.13). Cheap, reputationally valuable in the survey community.
- **A "two components are not enough" diagnostic.** El-Badry's χ²ᵣₑd > 1 test is used informally; `albireo`'s marginal likelihood makes it a model-comparison statement rather than an eyeball one.
- **Automatic continuum-aware segmentation, or a method that needs no segments.** Segment placement is a Fourier-specific wart that only one group (Dervişoğlu) automates and that made 100 Å segments simply fail to converge for V446 Cep. **`albireo` is not Fourier-domain and does not need it — say so.**
- **m s⁻¹ SB2 velocities** for circumbinary planets. **Scope out explicitly.**
- **Gaia DR4 SB2 epoch spectra** — roadmap Tier 2 item 7, correctly deferred to 2026-12-02. No competing tool exists.

### Two tribes, incompatible defaults

**[INFERENCE]** The twelve above do not weight equally across the user base, and a single default will not serve both:

- **The eclipsing / benchmark tribe** (Pavlovski, Torres, Maxted, Southworth, Tkachenko) *has* a light curve, wants 0.1 %-level masses and radii, and treats disentangling as preprocessing for abundances and Teff. They need: `l(λ)`, in-eclipse handling, a photometric light ratio accepted as a **hard constraint**, and errors that reach the atmospheric parameters.
- **The Be / stripped-star / dark-companion tribe** (El-Badry, Shenar, Seeburger, Rivinius, Bodensteiner) has **no** light curve, **no** light ratio, few epochs, and variable emission. They need: robustness to ansatz violation, a goodness-of-fit that flags "two components are insufficient", a calibrated non-detection, and **comparison across methods** — El-Badry's ALS 8814 paper runs three codes side by side and reports where they disagree, which is a direct argument for `albireo` shipping the shift-and-add and fd3-comparable modes it already has from the clean-room work.

---

## 8. What I would do next, in order

**[INFERENCE — this whole section.]**

1. **Fix `docs/science.md`'s Britavskiy/Bodensteiner error, then publish to PyPI.** The first is a five-minute correction that prevents an embarrassing citation error in front of exactly the audience you want; the second unblocks everything else, and the README currently contradicts itself.
2. **Write to the BLOeM team.** They have stated in A&A that their SB2 velocities and periods are unreliable without disentangling; a third BLOeM paper throws 69 SB2/emission stars out of the parameter pipeline; their own toolkit is MIT-licensed JAX + NumPyro with no disentangling module; and `albireo` already resolves a BLOeM identifier to its spectra in one line. Offer to run their 59+ SB2s. Highest-conversion action available.
3. **Reproduce Maxted et al. 2026's AI Phe disentangling with a band.** Same archival data, published answer, benchmark spectra openly hosted, and the paper itself enumerates eight hand steps `albireo` removes.
4. **Add a fourth code to the ALS 8814 three-way comparison** (arXiv:2509.01545). Public LAMOST data, published comparison figure, and the only one of the four that returns a posterior.
5. **Benchmark against BiSpeD on the SB1 scan, and cite it.** Live competitor, effectively unadopted, and the comparison — calibrated FAP plus K₁ marginalisation versus a CCF maximum — is favourable and fair.
6. **Reframe the headline** from "posterior uncertainties" to the symptom. The single best framing available is Serebriakova et al. 2025's own sentence — *"For disentangled spectra … observational uncertainties are not well-defined … their quality (and thus their uncertainties) evolves dynamically throughout the optimisation process"* — because `albireo`'s conditional-Gaussian posterior is exactly the object they say does not exist. §3.4 gives six further quotable instances; §3.14 gives the sentence that justifies a calibrated detection statistic.
7. **Answer the V446 Cep discrepancy.** Two groups, same HERMES spectra, same FDBinary, different K₂, and the authors cannot trace why. Run it with a window-sensitivity panel from `sensitivity_forecast` and a serialised config. It is a published, named, unresolved disagreement between two prominent groups — the kind of thing that gets a package cited rather than merely noticed.
8. **Promote time-variable component spectra from Tier 3 to Tier 2**, and close the wavelength-dependent light ratio. These are the named blockers in, respectively, the most-cited strand and the broadest-constituency complaint in the literature, and nothing else in the field can do either. The eclipse sub-case is cheap and comes with a hard number: 33 of 72 spectra discarded for V446 Cep, and they are the ones that measure the light ratio.
9. **Approach Southworth.** He does not disentangle, takes no new spectra, and his entire *Rediscussion* series is bottlenecked on a spectroscopic light ratio he repeatedly calls unreliable, inconsistent, or unexplained. A wavelength-resolved light ratio with a defensible error bar, from archival composite spectra, drops straight into `jktebop`'s constraint slot. Low effort, high visibility, and an unusually well-defined ask.
10. **State the ML-II comparison explicitly.** Seeburger et al. 2024 name automated regularisation as future work (*"We have found a suitable [Λ] … by trial and error; automatisation of this is an endeavour for future work"*); `albireo` already does it by empirical Bayes. A free, quotable, apples-to-apples advantage over the newest survey-scale code, currently invisible in the docs.

### Gaps in this survey, stated plainly

- **OWN Survey I** (2026A&A...708A..98B) could not be fetched: aanda.org returns 403 to automated fetches, ADS returns 405, and it appears not to be on arXiv. Everything about its methods beyond "UNWIND was already used in OWN I" is unverified.
- **"XShootU does not disentangle"** is provisional. XShootU VIII merges per-component PoWR models by an assumed luminosity ratio rather than disentangling, but XShootU IX/X/XI were not exhaustively checked.
- **Klement/Bodensteiner CHARA methods sections** were not obtained; their disentangling appears to be delegated to Shenar's `dsaa`.
- **CAOSP 55/3** (Litomyšl 2024, *"Binary and multiple stars in the era of big sky surveys"*) is a concentrated venue for this material and only two papers from it were sampled. Hadrava has a paper in that volume that was not retrieved. Worth a dedicated pass.
- **Holger Hensberge** has no located 2023–2026 paper. His 2008/2010 methodological work is cited constantly, but treat "Hensberge is an active community member" as unverified.
- **BiSpeD's paper body** could not be read (aanda.org JS-gates and 403s automated fetches); its description here comes from the repository README and API metadata, both of which are authoritative for what the code does but not for what the paper claims.
