# Full report: Gaia archive anchors for a simulated population of eclipsing double-lined binaries in DR4 RVS

Compiled 2026-10-07. "Measured" = computed in this session from an aggregate ADQL query on `https://gea.esac.esa.int/tap-server/tap`; the query text is `<name>.adql` and the result `<name>.csv` in this directory, and every table is generated from the CSV files by `tables.py`. "Quoted" = read in the cited source. G_RVS without qualification is the value predicted from G and G_RP with Sartoretti et al. (2023, A&A 674, A6, bibcode 2023A&A...674A...6S) eqs. (9)-(10).

Sources. [DR4C] ESA "Gaia DR4 content", https://www.cosmos.esa.int/web/gaia/dr4, last update 14 September 2026. [DR4M] draft Gaia DR4 data model, the PDF linked from [DR4C] (`gaia-dr4-prerelease-draft-data-model_2026-06-26.zip`); not downloaded in this session, read from the text extraction that an earlier session of this project saved on 2026-09-03 (`C:\Users\thari\Documents\temp\claude\C--Users-thari-Documents-GitHub-albireo\208f03db-63fc-4474-adc5-1218748183c7\scratchpad\dr4\dm.txt`); section numbers are the draft's; the extraction drops the symbols <= and >=, which are restored here from the context. [K23] Katz et al. 2023, A&A 674, A5 (2023A&A...674A...5K), read as arXiv:2206.05902. [M23] Mowlavi et al. 2023, A&A 674, A16 (2023A&A...674A..16M), read as arXiv:2211.00929; table and equation numbers are those of the arXiv version (the journal pages returned HTTP 403). [DOC3] Gaia DR3 documentation release 1.3, https://gea.esac.esa.int/archive/documentation/GDR3/ (section numbers given). Project notes: RVS-note = `internal/research/2026-09-09-gaia-rvs-benchmark/research_gaia_rvs_instrument.md`, POP-note = `research_binary_populations.md` in the same directory, ACCESS-note = `internal/research/2026-09-03-adoption/B_data_access.md`.

## 1. DR4 scope

**Release date.** Quoted: "Gaia DR4 (based on 66 months of data) 2 December 2026" (https://www.cosmos.esa.int/web/gaia/release); "Coming up: 2 December 2026 at ~12:00 CET" (https://www.cosmos.esa.int/web/gaia/data-release-4). Data span 25 July 2014 10:30 UTC to 20 January 2020 22:00 UTC [DR4C]. [DR4C] says of its table that it "is under development and changes can be expected". Measured: the TAP server had no `gaiadr4` schema on 2026-10-07 (`q00_tap_schemas`).

| Product | Access | Records [DR4C] | Content (quoted or condensed from [DR4C]) |
|---|---|---|---|
| `rvs_epoch_spectrum` | DataLink | 6,910,785,949 (49 TB) | "RVS spectra combined at the FOV-transit level. The spectra are in the solar barycentric frame and normalized." |
| `rvs_epoch_parameters_single` | DataLink | 6,910,423,894 | transit-level radial velocity, line broadening and G_RVS, single-lined transits |
| `rvs_epoch_parameters_double` | DataLink | 787,312 | the same for double-lined transits (transits, not sources) |
| `all_source_rvs` | TAP | 312,247,580 | per-source results of the spectroscopic pipeline |
| `epoch_photometry` | DataLink | 121,845,163,512 | G, BP and RP light curve per field-of-view transit |
| `epoch_photometry_ccd` | DataLink | 121,845,163,512 | G light curve per CCD transit |
| `all_source_photometry` | TAP | 2,793,009,568 | photometry of "nearly all sources" |
| `vari_eclipsing_binary` | TAP | 2,392,500 | "Eclipsing binaries resulting from variability analysis." |
| `nss_two_body_orbit` | TAP | 5,901,111 | two-body solutions, including eclipsing types |

**Sources with `rvs_epoch_spectrum` rows.** Quoted from [DR4M] 2.2, `all_source_flags.has_epoch_rvs` (byte): 0, no spectrum (star not processed by the RVS pipeline, or excluded in the initial steps); 1, "very weak to extremely weak signal (these spectra cannot be used independently)", stars with `external_apparent_grvs` > 14, no epoch radial velocities; 2, weak signal, 12 < `external_apparent_grvs` <= 14, epoch radial velocities computed; 3, "moderate to strong signal", `external_apparent_grvs` <= 12, epoch radial velocities and broadening velocities computed, and "the threshold of external_apparent_grvs = 12 also determines the type of normalisation applied to the CCD spectra before combination". [DR4M] 17.2: "epoch spectra are systematically produced"; 17.3: they "are still produced for detected double-lined, emission-line, and contaminated transits". No signal-to-noise cut is stated; the faint limit is that of the RVS processing ([K23] conclusions: "down to the limiting magnitude G_RVS^on-board = 16.2 mag"). Mean number of epoch spectra per `all_source_rvs` source: 6,910,785,949 / 312,247,580 = 22.1 (arithmetic on [DR4C]). The number of class 3 sources is not published.

**The gating magnitude.** Quoted from [DR4M] 2.5: `external_apparent_grvs` is, "for most sources, ... obtained using the magnitudes measured from the DR3 Gaia data ... (phot_g_mean_mag) and ... (phot_rp_mean_mag), with the transformation described in Sartoretti et al. (2023)", and otherwise the onboard magnitude. The draft does not name the equations; Sartoretti et al. (2023) sect. 5 introduce eqs. (9)-(10) as "this updated estimate of G_RVS^ext" and give no other transformation of their own (the DR3 pipeline used DR2 photometry with eqs. 2-3 of Gaia Collaboration 2018). The predicted G_RVS of this report is therefore, in all likelihood, the DR4 gating magnitude itself. Measured on a 1% random subsample of DR3 scaled by 100 (`q10_all_sources_grvs_pred_random1pct`): 1.23 million sources have predicted G_RVS < 10, 7.50 million < 12, 39.44 million < 14 and 179.69 million < 16. With the DR4 transit count estimated in the RVS-note (about 35 per source) the class 3 population is about 7.5 million sources and 2.6e8 epoch spectra; this is an estimate, not a published figure.

**Per-transit radial velocities.** Published in DR4. Quoted from [DR4M] 17.2: the table "is produced for all stars processed by the pipeline. However, only a very small fraction, specifically, the brightest stars (external_apparent_grvs <= 14 ...), have their epoch radial velocities computed"; none for transits with detected emission lines; velocities are flagged invalid for template temperatures above 14,500 K at `external_apparent_grvs` <= 12 and above 8,500 K when fainter; fainter than 11 only one of the three single-line methods (RvFou) is used. `vbroad` "is computed only for sources with external_apparent_grvs <= 12" and is flagged invalid when > 11, when below 4.5 km/s, and for templates cooler than about 3400 K or hotter than 25,000 K. Each transit carries `radial_velocity_error` ("including a correction to mitigate the original known underestimation"), `raw_radial_velocity_error`, `radial_velocity_mc_error` (interpolated in a table built from 1000 simulated transits per template, rotational velocity and magnitude), `rad_vel_bary_corr`, `rv_systematic_corr` and `expected_sig_to_noise`. Double lines are searched for only at `external_apparent_grvs` <= 12; `rv_assumed_sb2` is set "when double lines are detected in at least 10 RVS transits of the source", and then "no combined radial velocity, line-broadening velocity, mean spectrum, or associated per-source parameters are produced; only the epoch-level results are published"; "only about one third of these sources are subsequently confirmed as SB2 by the NSS pipeline". Quoted from [DR4M] 1.1: the combined velocity is the median of the epoch velocities for `external_apparent_grvs` <= 11 (`grvs_mag` <= 12 in DR3) and comes from combined cross-correlation functions when fainter; "a radial_velocity is provided for about 85 million of stars".

**Epoch photometry and the eclipsing-binary catalogue.** Quoted from [DR4C]: "data will be published for all processed sources, which amount to about 2.8 billion sources in total", with `epoch_photometry` as in the table. Quoted from [DR4M] 2.2: `has_epoch_photometry` flags the sources that have it; "epoch photometry always contains G band integrated photometry ..., together with BP and/or RP integrated photometry when available". The record count is 43.6 field-of-view transits per `all_source_photometry` row (arithmetic on [DR4C]), which is the full transit count; the fraction of sources with the flag set is not stated. DR4 keeps `vari_eclipsing_binary` (2,392,500 rows, 1.095 times the DR3 count); [DR4M] 18.6 keeps the DR3 columns, including `frequency`, `reference_time` (BJD in TCB - 2,455,197.5 d) and `derived_primary_ecl_phase`, so a period and an epoch of primary eclipse can be taken as known for a catalogued system, with the reliability quoted in section 2c. It adds `sample_classification` ("String-based sample classification based on Mowlavi et al. (2023) Table A.1", the samples of section 2d), `frequency_rv_consistency_flag` (the photometric frequency compared with the highest periodogram peak of the single-lined transit velocities: 1 consistent, 2 double, 3 half, 0 inconsistent, -1 too few velocities, -2 no velocities), `bp_usable`, `rp_usable`, `g_num_obs`, `g_signal_to_noise` and `classification_score`. [DR4M] 10.8 lists the `nss_two_body_orbit` solution types `EclipsingBinary`, `EclipsingSpectroSB1` and `EclipsingSpectroSB2`. The two DR4 papers "The second Gaia catalogue of eclipsing binary candidates" (identification and geometric modelling; astrophysical modelling) are listed without links at https://www.cosmos.esa.int/web/gaia/dr4-papers.

**DR3 practice.** Quoted: the spectroscopic pipeline processed stars with external G_RVS <= 14 ([K23] 3.2). Transit velocities were measured for them, but "beyond grvs_mag = 12 mag, the epoch radial velocities are not considered reliable enough to derive the combined radial velocities" ([K23] 3.6.2), so the median of transit velocities (`rv_method_used = 1`) stops at `grvs_mag` = 12; [K23] sect. 1: the signal "allows to derive single epoch radial velocities of G-K type stars down to G_RVS ~ 12-13 mag". Epoch velocities were not published, except for slightly fewer than 2000 Cepheids and RR Lyrae. Double-lined spectra were sought per transit with TodCorLight (one template for both components, low F-test threshold) followed by TodCorHeavy (different templates, single-lined model rejected "only with 99.9% confidence"); the binary model was accepted for 15 < abs(V1 - V2) < 500 km/s, brightness ratio > 0.2 and external G_RVS <= 11 ([DOC3] 6.4.8). Stars with at least 10% of their transits flagged double-lined (about 40,000) had `radial_velocity`, `vbroad`, `grvs_mag` and the mean spectrum removed ([DOC3] 6.5.2; [K23] 4.3.1) and were passed to the non-single-star processing, whose spectroscopic input was `rv_renormalised_gof > 4 & rv_nb_transits >= 10 & 3875 < rv_template_teff < 8125` "or if the source had been detected as SB2 by the spectroscopic processing" ([DOC3] 7.1.2). The documentation states no temperature range and no transit minimum for the SB2 channel itself; measured: every published SB2 orbit has at least 10 good transits (POP-note A.4).

**Processing of a spectrum before delivery.** Per CCD spectrum, condensed from [DOC3] 6.4.5: bias and bias non-uniformity removed; saturated samples flagged (spectra with more than 40 removed); dark current subtracted; straylight background subtracted from a straylight map (spectra with negative total flux removed); spectra with a contaminant brighter than the target + 3 mag that has no window of its own removed; cosmic rays removed; 2-D windows collapsed; overlapping (truncated) windows deblended and badly deblended samples flagged (about 30% of the surviving DR3 spectra are deblended); wavelengths assigned from the trended calibration and cut to 846-870 nm; the ground-measured filter response removed; and, quoted, "the spectra of the bright stars are normalised to their pseudo-continuum using a 2nd-degree polynomial fitting. The stellar lines are iteratively rejected using a sigma-clipping with interval [-3,+10] sigma. For the faint stars (G_RVS^ext > 12) and the very cool stars, presenting the molecular TiO band in their spectrum, the polynomial was replaced by a constant equal to the median of the fluxes." [DR4M] 17.5 keeps these steps as `ccd_status` bits (polynomial, template or constant normalisation; deblended; truncated). Combination into an epoch spectrum, quoted from [DR4M] 17.3: "The fluxes and uncertainties-squared of the CCD spectra are linearly interpolated onto a fixed wavelength grid of 961 spectral elements with a step of 0.025 nm"; the flux is the mean over the n contributing CCDs (n usually 3, fewer where samples "were masked due to bad deblending, cosmic rays, or saturation, or at the edges of the wavelength range", NaN when none is left) and the error is (1/n) sqrt(sum of the n variances), each from "Poisson noise, readout noise, bias non-uniformity noise ..., straylight background, and dark current"; "Before combination, the CCD spectra are normalized, and then linearly interpolated to the wavelength grid. The combined spectrum is then renormalized." `normalisation_method` refers to that second step only: 0, "Not normalised. The combined spectra where double-lines are detected are not normalised"; 3, flux and error divided by the median of the combined spectrum; 4, negative median, not normalised. Read literally, a double-lined epoch spectrum is the mean of CCD spectra that were each normalised and is not renormalised; the ACCESS-note wording "not normalised at all" is stronger than the draft text. Wavelength zero point: [DOC3] 6.3.3 fixes it with standard stars per calibration unit and trends the coefficients; [DOC3] 6.4.8 corrects the transit velocities for offsets that depend on magnitude and on time with polynomials per CCD row, straylight level, trending epoch and field of view (the correction removes the offset in the median of a magnitude bin, not star by star); DR4 publishes the applied correction per transit as `rv_systematic_corr`; [K23] eq. (1) adds a constant 0.113 km/s to the error of the combined velocity "to take into account the wavelength calibration errors and similar sources of uncertainties" (0.112 km/s in [DR4M] 1.1). No per-transit zero-point scatter is published; section 3 measures an upper bound of 0.17 to 0.20 km/s.

## 2. The DR3 eclipsing-binary catalogue

All tables join `gaiadr3.vari_eclipsing_binary` (2,184,477 rows) to `gaiadr3.gaia_source` on `source_id`. The archive ADQL has no CASE; the two branches of the G_RVS relation use `IF_THEN_ELSE`. Totals (measured): 2090678 inside the colour range, 93799 outside, sum 2184477. Quoted from [M23]: the candidates are sources classified as eclipsing with G < 20, at least 16 cleaned field-of-view measurements in G and skewness > -0.2 (sect. 2.1), published when `global_ranking` > 0.4 (sect. 2.3); relative to OGLE4, 28% of the eclipsing binaries brighter than G = 20 are recovered (26% in the Bulge, 48% in the SMC; sect. 4.2). [M23] give no completeness for the bright magnitudes of interest here.

### 2a. Counts by predicted G_RVS

Measured, `q01a_eb_counts_grvs_half_mag`. `radial_velocity` and `rv_nb_transits` are non-null for the same sources in every bin, so one column serves both.

| G_RVS (predicted) | N | N cumulative | % with grvs_mag | % with radial_velocity |
|---|---|---|---|---|
| < 4.0 | 2 | 2 | 100.0 | 100.0 |
| 4.0-4.5 | 1 | 3 | 0.0 | 0.0 |
| 4.5-5.0 | 4 | 7 | 25.0 | 25.0 |
| 5.0-5.5 | 9 | 16 | 44.4 | 44.4 |
| 5.5-6.0 | 22 | 38 | 36.4 | 36.4 |
| 6.0-6.5 | 50 | 88 | 20.0 | 20.0 |
| 6.5-7.0 | 69 | 157 | 24.6 | 24.6 |
| 7.0-7.5 | 129 | 286 | 23.3 | 23.3 |
| 7.5-8.0 | 228 | 514 | 22.8 | 23.7 |
| 8.0-8.5 | 380 | 894 | 27.1 | 27.6 |
| 8.5-9.0 | 603 | 1497 | 27.5 | 28.7 |
| 9.0-9.5 | 924 | 2421 | 32.8 | 34.4 |
| 9.5-10.0 | 1479 | 3900 | 36.6 | 37.5 |
| 10.0-10.5 | 2508 | 6408 | 47.0 | 48.0 |
| 10.5-11.0 | 3951 | 10359 | 71.6 | 74.0 |
| 11.0-11.5 | 6639 | 16998 | 83.3 | 85.9 |
| 11.5-12.0 | 10254 | 27252 | 74.0 | 76.4 |
| 12.0-12.5 | 15209 | 42461 | 54.5 | 56.5 |
| 12.5-13.0 | 22382 | 64843 | 56.6 | 58.2 |
| 13.0-13.5 | 32922 | 97765 | 52.8 | 54.3 |
| 13.5-14.0 | 47100 | 144865 | 36.9 | 38.3 |
| 14.0-14.5 | 65626 | 210491 | 0.1 | 0.1 |
| 14.5-15.0 | 90028 | 300519 | 0.0 | 0.0 |
| 15.0-15.5 | 119273 | 419792 | 0.0 | 0.0 |
| 15.5-16.0 | 156737 | 576529 | 0.0 | 0.0 |
| >= 16.0 | 1514149 | 2090678 | 0.0 | 0.0 |

Sources outside the validity range of the relation (measured, `q01b_eb_counts_colour_out_of_range`):

| Category | N | N with G < 13 | N with G_RP < 12 | N with radial_velocity |
|---|---|---|---|---|
| G - G_RP < -0.15 | 2112 | 15 | 8 | 0 |
| G - G_RP > 1.7 | 32933 | 7 | 16 | 37 |
| G or G_RP missing | 58754 | 33 | - | 238 |

Comparison with all DR3 sources in the same bins (measured, `q10_all_sources_grvs_pred_random1pct`, a 1% random subsample scaled by 100):

| G_RVS (predicted) | N all sources (10^6) | N cumulative (10^6) | % of all with radial_velocity | N EB | EB as % of all | % of EB with radial_velocity |
|---|---|---|---|---|---|---|
| 5-6 | 0.014 | 0.022 | 88.9 | 31 | 0.22 | 38.7 |
| 6-7 | 0.041 | 0.063 | 92.4 | 119 | 0.29 | 22.7 |
| 7-8 | 0.114 | 0.176 | 90.0 | 357 | 0.31 | 23.5 |
| 8-9 | 0.297 | 0.473 | 86.2 | 983 | 0.33 | 28.3 |
| 9-10 | 0.758 | 1.232 | 93.5 | 2403 | 0.32 | 36.3 |
| 10-11 | 1.872 | 3.104 | 95.5 | 6459 | 0.35 | 63.9 |
| 11-12 | 4.394 | 7.498 | 95.4 | 16893 | 0.38 | 80.2 |
| 12-13 | 10.014 | 17.512 | 88.4 | 37591 | 0.38 | 57.5 |
| 13-14 | 21.929 | 39.441 | 80.4 | 80022 | 0.36 | 44.9 |
| 14-15 | 47.170 | 86.611 | 0.2 | 155654 | 0.33 | 0.1 |
| 15-16 | 93.080 | 179.691 | 0.0 | 276010 | 0.30 | 0.0 |

Eclipsing-binary candidates are 0.3 to 0.4% of the sources at every magnitude from 6 to 16. Brighter than G_RVS = 10 only 23 to 39% of them have a DR3 radial velocity, against 86 to 94% of all sources; the fraction rises to 64% at 10-11 and 80% at 11-12. DR3 publishes no per-source reason, so the loss cannot be decomposed; in magnitude it coincides with the double-line search (external G_RVS <= 11) and the removal rule quoted in section 1.

### 2b. Colour and temperature

Measured, `q02_eb_bprp_by_grvs` (0.1 mag bins) and `q03_eb_teff_gspphot_by_grvs` (500 K bins). The class boundaries 0.35 and 0.75 fall inside a bin, which is split in two halves. Colours are not dereddened, and half of the G_RVS <= 12 sample lies at abs(b) < 10 deg (section 2g), so the blue classes are lower limits. Sources with G_RVS <= 12 and no BP-RP: 1. Columns p5 to p95 are percentiles of BP-RP.

| G_RVS | N | % < 0.0 | % 0.0-0.35 | % 0.35-0.75 | % 0.75-1.0 | % 1.0-1.8 | % >= 1.8 | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5-6 | 31 | 9.7 | 45.2 | 27.4 | 11.3 | 6.5 | 0.0 | -0.07 | 0.16 | 0.33 | 0.64 | 1.34 |
| 6-7 | 119 | 12.6 | 29.8 | 34.5 | 11.3 | 10.1 | 1.7 | -0.13 | 0.13 | 0.43 | 0.73 | 1.20 |
| 7-8 | 357 | 6.7 | 29.3 | 40.9 | 12.2 | 10.1 | 0.8 | -0.04 | 0.21 | 0.51 | 0.73 | 1.35 |
| 8-9 | 983 | 2.2 | 23.8 | 45.3 | 15.0 | 13.4 | 0.3 | 0.04 | 0.34 | 0.58 | 0.78 | 1.35 |
| 9-10 | 2403 | 1.2 | 17.1 | 48.0 | 20.7 | 11.9 | 1.0 | 0.13 | 0.43 | 0.64 | 0.83 | 1.31 |
| 10-11 | 6459 | 0.2 | 8.2 | 46.1 | 27.2 | 17.2 | 1.0 | 0.28 | 0.54 | 0.72 | 0.91 | 1.35 |
| 11-12 | 16893 | 0.1 | 4.6 | 38.1 | 32.7 | 23.0 | 1.5 | 0.36 | 0.62 | 0.80 | 1.00 | 1.45 |
| 12-13 | 37591 | 0.0 | 1.8 | 26.5 | 35.2 | 34.2 | 2.2 | 0.46 | 0.72 | 0.90 | 1.11 | 1.58 |
| 13-14 | 80022 | 0.0 | 0.8 | 15.9 | 30.3 | 48.8 | 4.2 | 0.56 | 0.83 | 1.02 | 1.26 | 1.75 |
| <= 10 | 3900 | 2.5 | 20.5 | 46.1 | 18.1 | 12.1 | 0.8 | 0.06 | 0.38 | 0.61 | 0.80 | 1.32 |
| <= 12 | 27252 | 0.5 | 7.7 | 41.2 | 29.3 | 20.1 | 1.3 | 0.27 | 0.57 | 0.75 | 0.96 | 1.41 |

BP-RP histogram for G_RVS <= 12 (lower bin edge:count): -0.3:6 -0.2:28 -0.1:90 +0.0:250 +0.1:450 +0.2:755 +0.3:1299 +0.4:1934 +0.5:2857 +0.6:3736 +0.7:4081 +0.8:3467 +0.9:2481 +1.0:1733 +1.1:1278 +1.2:831 +1.3:578 +1.4:415 +1.5:285 +1.6:207 +1.7:148 +1.8:82 +1.9:64 +2.0:50 +2.1:28 +2.2:35 +2.3:16 +2.4:20 +2.5:15 +2.6:10 +2.7:9 +2.8:6 +2.9:3 +3.0:2 +3.5:1 +3.6:1

| G_RVS | N | % without teff_gspphot | % < 4000 | % 4000-5000 | % 5000-6000 | % 6000-7500 | % 7500-10000 | % >= 10000 | p5 (K) | p50 (K) | p95 (K) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5-6 | 31 | 29.0 | 0.0 | 0.0 | 18.2 | 27.3 | 31.8 | 22.7 | 5638 | 7625 | 15483 |
| 6-7 | 119 | 30.3 | 1.2 | 0.0 | 12.0 | 32.5 | 13.3 | 41.0 | 5572 | 7938 | 19425 |
| 7-8 | 357 | 35.3 | 0.0 | 2.2 | 10.8 | 38.1 | 17.7 | 31.2 | 5514 | 7417 | 21075 |
| 8-9 | 983 | 27.8 | 0.0 | 3.0 | 12.5 | 44.2 | 17.0 | 23.2 | 5403 | 6982 | 17458 |
| 9-10 | 2403 | 27.9 | 0.0 | 1.7 | 16.9 | 46.5 | 16.9 | 18.0 | 5417 | 6893 | 16225 |
| 10-11 | 6459 | 25.8 | 0.1 | 2.7 | 23.0 | 46.4 | 15.3 | 12.4 | 5192 | 6644 | 15092 |
| 11-12 | 16893 | 23.4 | 0.2 | 3.3 | 26.5 | 45.5 | 14.9 | 9.6 | 5122 | 6475 | 12448 |
| 12-13 | 37591 | 19.7 | 0.2 | 4.1 | 32.6 | 43.1 | 13.6 | 6.4 | 5038 | 6334 | 10630 |
| 13-14 | 80022 | 14.6 | 0.4 | 6.0 | 36.4 | 38.6 | 14.8 | 3.8 | 4866 | 6221 | 9713 |
| <= 10 | 3900 | 28.6 | 0.0 | 2.0 | 15.1 | 44.6 | 17.0 | 21.3 | 5433 | 6964 | 17189 |
| <= 12 | 27252 | 24.7 | 0.2 | 2.9 | 24.2 | 45.6 | 15.3 | 11.8 | 5163 | 6580 | 13975 |

`teff_gspphot` histogram for G_RVS <= 12 (lower bin edge in K:count): 3000:7 3500:25 4000:76 4500:529 5000:1193 5500:3764 6000:4220 6500:2757 7000:2376 7500:1313 8000:392 8500:395 9000:513 9500:524 10000:360 10500:206 11000:200 11500:178 12000:171 12500:117 13000:103 13500:68 14000:67 14500:33 15000:295 15500:70 16000:91 16500:53 17000:55 17500:44 18000:47 18500:53 19000:32 19500:38 20000:32 20500:24 21000:15 21500:10 22000:8 22500:9 23000:10 23500:10 24000:2 24500:3 25000:4 25500:1 26000:1 27500:3 28500:2 29500:3 32000:3 32500:2 33500:1 35000:1

The two indicators disagree on the hot fraction at G_RVS <= 12: 0.5% of the sources are bluer than BP-RP = 0.0, while 11.8% of those with a `teff_gspphot` (75% of the sample) are at or above 10,000 K. GSP-Phot fits a single reddened star, so its temperature for a binary in the Galactic plane is tied to the extinction it adopts; the excess at 15,000 K in the histogram is an edge of its model libraries.

### 2c. Orbital period

Quoted: `frequency` is the "orbital frequency of the EB", f_orb = 1/P_orb, in 1/d ([M23] Table 3; the DR3 data model says only "frequency of geometric model of the eclipsing binary light curve"). Measured: the largest frequency is 5.0 1/d, P = 0.2 d (POP-note B.3). Quoted from [M23] 4.1: for cross-matches classified as eclipsing in the literature, more than 85% of the Gaia periods agree with the literature period and 93% are compatible when half and twice the literature period are also accepted; over all cross-matches "the Gaia periods are compatible with literature periods in about 85% of cases", factors of two included. Measured, `q04_eb_logp_by_grvs` (0.1 dex bins of log10 P); the "%" columns are shares in period intervals in days; p5 to p95 are in days.

| G_RVS | N | % < 0.32 | % 0.32-0.5 | % 0.5-1 | % 1-2 | % 2-5 | % 5-10 | % 10-32 | % >= 32 | p5 | p10 | p25 | p50 | p75 | p90 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5-6 | 31 | 3.2 | 0.0 | 9.7 | 38.7 | 29.0 | 12.9 | 6.5 | 0.0 | 0.57 | 0.81 | 1.30 | 1.94 | 3.76 | 7.76 | 11.09 |
| 6-7 | 119 | 3.4 | 6.7 | 18.5 | 20.2 | 29.4 | 11.8 | 4.2 | 5.9 | 0.35 | 0.50 | 0.90 | 2.07 | 4.34 | 10.23 | 40.27 |
| 7-8 | 357 | 1.7 | 9.8 | 12.0 | 23.0 | 28.9 | 10.9 | 7.8 | 5.9 | 0.40 | 0.48 | 1.05 | 2.25 | 4.96 | 14.66 | 43.48 |
| 8-9 | 983 | 3.4 | 9.6 | 17.0 | 24.3 | 26.0 | 10.1 | 6.8 | 2.8 | 0.35 | 0.44 | 0.84 | 1.77 | 3.94 | 9.68 | 18.22 |
| 9-10 | 2403 | 3.5 | 11.8 | 20.0 | 24.3 | 25.6 | 7.7 | 4.7 | 2.4 | 0.34 | 0.41 | 0.73 | 1.52 | 3.16 | 6.97 | 14.49 |
| 10-11 | 6459 | 4.0 | 15.9 | 21.8 | 23.7 | 22.2 | 6.8 | 4.2 | 1.4 | 0.33 | 0.37 | 0.60 | 1.27 | 2.73 | 5.98 | 11.16 |
| 11-12 | 16893 | 5.4 | 18.0 | 21.2 | 21.9 | 21.5 | 6.4 | 4.4 | 1.2 | 0.31 | 0.36 | 0.53 | 1.19 | 2.66 | 5.83 | 10.88 |
| 12-13 | 37591 | 7.2 | 21.6 | 21.8 | 20.4 | 18.3 | 5.6 | 4.1 | 1.0 | 0.29 | 0.33 | 0.46 | 0.98 | 2.33 | 5.35 | 10.20 |
| 13-14 | 80022 | 9.2 | 24.2 | 22.6 | 18.7 | 16.2 | 4.7 | 3.4 | 0.9 | 0.28 | 0.32 | 0.42 | 0.83 | 2.02 | 4.64 | 8.73 |
| <= 10 | 3900 | 3.3 | 10.8 | 18.4 | 24.2 | 26.2 | 8.7 | 5.5 | 2.9 | 0.34 | 0.43 | 0.78 | 1.65 | 3.59 | 8.44 | 17.08 |
| <= 12 | 27252 | 4.8 | 16.5 | 20.9 | 22.7 | 22.4 | 6.8 | 4.5 | 1.5 | 0.32 | 0.37 | 0.57 | 1.27 | 2.80 | 6.14 | 11.69 |
| all (valid colour) | 2090678 | 19.6 | 31.7 | 18.5 | 11.9 | 10.2 | 4.0 | 3.4 | 0.9 | 0.25 | 0.27 | 0.34 | 0.49 | 1.33 | 4.11 | 8.53 |

log10(P/d) histogram, G_RVS <= 12 (lower bin edge:count): -0.7:284 -0.6:1020 -0.5:2185 -0.4:2304 -0.3:1819 -0.2:1896 -0.1:1989 +0.0:2048 +0.1:2029 +0.2:2100 +0.3:1982 +0.4:1688 +0.5:1355 +0.6:1072 +0.7:854 +0.8:554 +0.9:451 +1.0:382 +1.1:317 +1.2:224 +1.3:154 +1.4:147 +1.5:114 +1.6:73 +1.7:60 +1.8:30 +1.9:41 +2.0:23 +2.1:19 +2.2:22 +2.3:7 +2.4:3 +2.5:3 +2.6:1 +2.7:2

### 2d. Light-curve morphology

The table carries no detached, semi-detached or contact label. [M23] Appendix A defines twelve samples from the model type and the model parameters (Table 7 of the arXiv version; "Table A.1" in [DR4M] 18.6, which publishes the label in DR4 as `sample_classification`) and groups them (Table 8) as "wide systems (detached or, under some conditions, semi-detached)" = 2G-A, 2GE-A, 2G-D, 1G (27% of the catalogue), "tight systems" = 2G-B, 2G-C, 2GE-B, 1GE, 0GE (56%) and "to be investigated" = 2G-X, 2G-Y, 2GE-Z (17%). [M23] add that an Algol-type system whose Roche-lobe-filling star is faint can look detached, that 2GE-A systems are "all tight systems with a visible ellipsoidal component" of small amplitude, and that beta Lyr and W UMa fall in 2GE-B. That grouping is applied here, with sp and ss the `geom_model_gaussian*_sigma` of the deeper and of the shallower Gaussian (ranked by `geom_model_gaussian*_depth`; [M23] 2.4: Gaussian 1 is not necessarily the deeper), A_ell = `geom_model_cosine_half_period_amplitude`, and D = abs(abs(`derived_primary_ecl_phase` - `derived_secondary_ecl_phase`) - 0.5) ([M23] eq. 7).

| Sample | `num_model_parameters` | Condition ([M23] Table 7) | N published | N measured (`q05a_eb_samples_whole_catalogue_depth`) |
|---|---|---|---|---|
| 2G-A | 8 | -0.007406 + 0.7 sp + 0.4 (4.3 sp - 0.1)^2 < ss < min(0.0119 + 1.26 sp - 0.6 (2 sp - 0.1)^2, 0.14 - 1.4 sp) | 285,320 | 278806 |
| 2G-B | 8 | max(0.14 - 1.4 sp, -0.013972 + 0.88624 sp) < ss < min(0.012 + 1.26 sp, 0.43 - 1.6 sp) | 834,093 | 834093 |
| 2G-C | 8 | ss > max(0.18 - 0.2 sp, 0.43 - 1.6 sp, 0.204 - 51 (sp - 0.146)) | 24,081 | 24081 |
| 2G-D | 8 | ss > sp, sp < 0.02, not in 2G-A | 111,820 | 111820 |
| 2G-X | 8 | ss > sp, not in 2G-A to 2G-D | 182,280 | 182280 |
| 2G-Y | 8 | ss < sp, not in 2G-A to 2G-D | 150,332 | 156846 |
| 2GE-A | 9 | 2 A_ell < 0.11 mag and D < 0.07 | 162,630 | 162630 |
| 2GE-B | 9 | 2 A_ell >= 0.11 mag and D < 0.07 | 265,276 | 265276 |
| 2GE-Z | 9 | D >= 0.07 | 47,219 | 47219 |
| 1G, 1GE, 0GE | 5, 6, 4 | all | 36,984; 48,215; 36,227 | 36984; 48215; 36227 |

Ten samples are reproduced exactly. 6514 sources (0.3% of the catalogue) have ss < sp, fall below the printed lower boundary of 2G-A and are counted in 2G-Y here but in 2G-A by [M23]; five variants of that boundary did not reproduce the published count (`t05_sample_2ga_boundary_test`, `t06_sample_2ga_boundary_variants`), so 2G-A is 2.3% low and 2G-Y 4.3% high in the tables below.

Shares in per cent by magnitude (measured, `q05b_eb_samples_by_grvs_logp`; last row from `q05a`; t.b.i. = to be investigated):

| G_RVS | N | wide | tight | t.b.i. | 2G-A | 2GE-A | 2G-D | 1G | 2G-B | 2G-C | 2GE-B | 1GE | 0GE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5-6 | 31 | 51.6 | 32.3 | 16.1 | 16.1 | 35.5 | 0.0 | 0.0 | 12.9 | 0.0 | 6.5 | 12.9 | 0.0 |
| 6-7 | 119 | 44.5 | 37.8 | 17.6 | 11.8 | 20.2 | 4.2 | 8.4 | 13.4 | 3.4 | 9.2 | 9.2 | 2.5 |
| 7-8 | 357 | 47.9 | 34.5 | 17.6 | 9.2 | 30.0 | 4.5 | 4.2 | 13.2 | 1.7 | 13.4 | 5.0 | 1.1 |
| 8-9 | 983 | 52.3 | 35.0 | 12.7 | 10.8 | 32.7 | 3.8 | 5.1 | 15.2 | 2.2 | 11.8 | 5.4 | 0.4 |
| 9-10 | 2403 | 48.7 | 38.6 | 12.7 | 8.5 | 30.7 | 3.7 | 5.8 | 15.8 | 2.0 | 13.2 | 6.7 | 0.8 |
| 10-11 | 6459 | 44.4 | 43.4 | 12.2 | 9.3 | 27.4 | 3.3 | 4.5 | 17.4 | 4.1 | 14.7 | 5.7 | 1.4 |
| 11-12 | 16893 | 42.3 | 45.1 | 12.7 | 7.9 | 26.8 | 3.8 | 3.8 | 19.5 | 5.0 | 13.4 | 5.5 | 1.6 |
| 12-13 | 37591 | 40.1 | 48.5 | 11.4 | 8.0 | 24.3 | 3.9 | 3.8 | 21.6 | 4.6 | 15.7 | 5.1 | 1.6 |
| 13-14 | 80022 | 39.7 | 49.5 | 10.8 | 8.9 | 23.0 | 4.0 | 3.8 | 22.6 | 3.5 | 17.5 | 4.4 | 1.4 |
| <= 10 | 3900 | 49.4 | 37.3 | 13.3 | 9.3 | 30.8 | 3.8 | 5.5 | 15.3 | 2.1 | 12.8 | 6.3 | 0.8 |
| <= 12 | 27252 | 43.8 | 43.5 | 12.6 | 8.4 | 27.5 | 3.6 | 4.2 | 18.4 | 4.4 | 13.6 | 5.7 | 1.4 |
| all magnitudes | 2184477 | 27.0 | 55.3 | 17.7 | 12.8 | 7.4 | 5.1 | 1.7 | 38.2 | 1.1 | 12.1 | 2.2 | 1.7 |

Shares in per cent by period for G_RVS <= 12, with the group shares for G_RVS <= 10 in the last three columns (same query, 0.2 dex bins):

| P (d) | N | % wide | % tight | % t.b.i. | % 2G-A+2G-D+1G | % 2GE-A | % 2G-B+2G-C | % 2GE-B | % 1GE+0GE | N (<= 10) | % wide (<= 10) | % tight (<= 10) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < 0.25 | 284 | 37.7 | 29.9 | 32.4 | 31.0 | 6.7 | 12.3 | 14.8 | 2.8 | 31 | 51.6 | 25.8 |
| 0.25-0.40 | 3205 | 11.1 | 83.5 | 5.3 | 2.9 | 8.2 | 38.8 | 42.7 | 2.0 | 285 | 16.1 | 75.8 |
| 0.40-0.63 | 4123 | 15.5 | 77.7 | 6.8 | 4.5 | 11.0 | 43.9 | 30.1 | 3.6 | 429 | 14.2 | 79.7 |
| 0.63-1.00 | 3885 | 36.9 | 54.9 | 8.2 | 8.3 | 28.6 | 34.1 | 13.7 | 7.1 | 522 | 37.0 | 55.6 |
| 1.00-1.58 | 4077 | 55.8 | 33.0 | 11.2 | 14.1 | 41.7 | 18.4 | 5.5 | 9.1 | 624 | 58.0 | 29.8 |
| 1.58-2.51 | 4082 | 66.0 | 21.0 | 13.0 | 21.1 | 44.9 | 9.9 | 2.2 | 8.8 | 653 | 72.0 | 18.5 |
| 2.51-3.98 | 3043 | 66.2 | 18.8 | 15.1 | 26.4 | 39.8 | 7.3 | 1.9 | 9.6 | 478 | 67.4 | 19.2 |
| 3.98-6.31 | 1926 | 62.5 | 19.2 | 18.3 | 33.0 | 29.5 | 7.8 | 2.6 | 8.7 | 372 | 62.1 | 21.8 |
| 6.3-10.0 | 1005 | 53.7 | 21.5 | 24.8 | 36.0 | 17.7 | 9.4 | 2.8 | 9.4 | 177 | 54.8 | 19.2 |
| 10.0-15.8 | 699 | 42.5 | 27.8 | 29.8 | 30.9 | 11.6 | 11.4 | 5.9 | 10.4 | 120 | 48.3 | 23.3 |
| 15.8-25.1 | 378 | 43.4 | 24.1 | 32.5 | 31.7 | 11.6 | 11.6 | 5.0 | 7.4 | 67 | 41.8 | 22.4 |
| 25.1-39.8 | 261 | 38.7 | 22.6 | 38.7 | 32.2 | 6.5 | 12.3 | 4.6 | 5.7 | 53 | 39.6 | 24.5 |
| 39.8-63.1 | 133 | 42.9 | 22.6 | 34.6 | 35.3 | 7.5 | 9.0 | 3.0 | 10.5 | 33 | 30.3 | 24.2 |
| 63.1-100.0 | 71 | 38.0 | 19.7 | 42.3 | 29.6 | 8.5 | 4.2 | 1.4 | 14.1 | 21 | 19.0 | 33.3 |
| >= 100 | 80 | 33.8 | 30.0 | 36.2 | 30.0 | 3.8 | 15.0 | 1.2 | 13.8 | 35 | 25.7 | 34.3 |

At G_RVS <= 12 the wide samples with flat light outside eclipse (2G-A, 2G-D, 1G) hold 16.3% of the candidates, 2GE-A 27.5%, the tight group 43.5% and the unassigned group 12.6%.

Eclipse depths in G (mag) of the derived primary (deepest) and secondary eclipses (measured, `q05d_eb_primary_depth_grvs_le12`, `q05e_eb_secondary_depth_grvs_le12`, 0.01 mag bins). Models with one Gaussian have no secondary and purely ellipsoidal models have neither; the last four columns are shares of the sources that have the depth. [M23] A.1 warn that depths are overestimated where an eclipse is poorly covered, mainly for durations of 0.07 to 0.17 d.

| Eclipse | G_RVS | N | % no depth | p10 | p25 | p50 | p75 | p90 | % >= 0.05 | % >= 0.1 | % >= 0.2 | % >= 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | <= 12 | 27252 | 1.4 | 0.073 | 0.137 | 0.274 | 0.474 | 0.719 | 95.7 | 83.5 | 62.5 | 22.7 |
| primary | <= 10 | 3900 | 0.8 | 0.096 | 0.158 | 0.296 | 0.500 | 0.788 | 98.3 | 89.0 | 66.0 | 25.0 |
| secondary | <= 12 | 27252 | 11.3 | 0.020 | 0.055 | 0.129 | 0.263 | 0.420 | 77.3 | 58.5 | 34.7 | 5.8 |
| secondary | <= 10 | 3900 | 12.7 | 0.023 | 0.065 | 0.138 | 0.273 | 0.414 | 79.9 | 62.5 | 35.7 | 5.3 |

By group for G_RVS <= 12 (measured, `q05c_eb_depth_joint_grvs_le12` in 0.1 mag bins and `q05f_eb_primary_duration_grvs_le12`); depths in mag; durations in phase, defined as 5.6 sigma of the Gaussian with a cap at 0.4 ([M23] 2.2.2); the shares with a secondary depth above a threshold are fractions of the whole group:

| Group | N | median primary depth | % with a secondary eclipse | median secondary depth | % secondary >= 0.1 | % secondary >= 0.2 | primary duration p10 | p50 | p90 | % at the 0.4 cap |
|---|---|---|---|---|---|---|---|---|---|---|
| wide | 11939 | 0.30 | 90.4 | 0.12 | 50.0 | 25.2 | 0.054 | 0.149 | 0.276 | 0.4 |
| 2G-A | 2291 | 0.30 | 100.0 | 0.10 | 50.3 | 30.2 | 0.049 | 0.127 | 0.257 | 0.0 |
| 2GE-A | 7503 | 0.29 | 100.0 | 0.14 | 62.1 | 29.9 | 0.085 | 0.169 | 0.289 | 0.6 |
| tight | 11866 | 0.26 | 83.7 | 0.20 | 61.7 | 42.2 | 0.169 | 0.400 | 0.408 | 50.8 |
| to be investigated | 3447 | 0.23 | 100.0 | 0.07 | 24.5 | 11.1 | 0.060 | 0.188 | 0.405 | 19.8 |

### 2e. Deciles of spectroscopic and photometric quantities

Measured, `q06a_eb_rv_nb_transits_le12`, `q06b_eb_rv_expected_snr_le12`, `q06c_eb_vbroad_le12`, `q06d_eb_rv_amplitude_robust_le12`, `q06e_eb_phot_g_n_obs_le12`; bins of 1 transit, 0.5 in S/N, 1 km/s, 1 km/s and 5 observations, linear interpolation inside a bin; deciles are over the non-null values. `rv_expected_sig_to_noise` is the value combined over all transits (RVS-note 2). `phot_g_n_obs` counts CCD observations, about nine per field-of-view transit.

| Quantity | G_RVS | N | % non-null | p10 | p20 | p30 | p40 | p50 | p60 | p70 | p80 | p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rv_nb_transits | <= 12 | 27252 | 69.5 | 10 | 13 | 16 | 18 | 20 | 22 | 24 | 28 | 33 |
| rv_nb_transits | <= 10 | 3900 | 32.7 | 8 | 10 | 12 | 14 | 17 | 19 | 21 | 24 | 29 |
| rv_expected_sig_to_noise | <= 12 | 27252 | 69.5 | 18.8 | 22.4 | 25.4 | 28.4 | 31.8 | 35.8 | 41.2 | 49.1 | 65.2 |
| rv_expected_sig_to_noise | <= 10 | 3900 | 32.7 | 63.3 | 73.9 | 82.3 | 91.3 | 99.2 | 110.3 | 123.6 | 142.8 | 206.7 |
| vbroad (km/s) | <= 12 | 27252 | 37.9 | 17 | 32 | 48 | 64 | 82 | 102 | 122 | 145 | 176 |
| vbroad (km/s) | <= 10 | 3900 | 22.3 | 12 | 25 | 41 | 61 | 78 | 99 | 118 | 135 | 162 |
| rv_amplitude_robust (km/s) | <= 12 | 27252 | 67.7 | 52 | 79 | 99 | 119 | 139 | 159 | 184 | 217 | 274 |
| rv_amplitude_robust (km/s) | <= 10 | 3900 | 32.3 | 24 | 45 | 63 | 79 | 95 | 111 | 126 | 146 | 168 |
| phot_g_n_obs | <= 12 | 27252 | 100.0 | 219 | 280 | 330 | 363 | 391 | 421 | 455 | 499 | 578 |
| phot_g_n_obs | <= 10 | 3900 | 100.0 | 218 | 276 | 323 | 357 | 385 | 416 | 450 | 493 | 571 |

### 2f. Evolved stars

Measured, `q07a_eb_cmd_grvs_le12_plxsnr10` (bins of 0.1 mag in BP-RP and 0.5 mag in M_G, rebinned here to 0.4 mag and 1 mag; the full grid is in the CSV). M_G = G + 5 log10(parallax/mas) - 10. No extinction or reddening correction is applied. 25658 of the 27,252 sources pass `parallax_over_error > 10` (1 of them without BP-RP).

| M_G \ BP-RP | < 0.0 | 0.0-0.4 | 0.4-0.8 | 0.8-1.2 | 1.2-1.6 | 1.6-2.0 | >= 2.0 | all |
|---|---|---|---|---|---|---|---|---|
| < -2 | 17 | 93 | 63 | 27 | 19 | 6 | 1 | 226 |
| -2 to -1 | 26 | 173 | 152 | 90 | 45 | 16 | 7 | 509 |
| -1 to +0 | 47 | 388 | 332 | 197 | 129 | 49 | 30 | 1172 |
| +0 to +1 | 19 | 773 | 962 | 434 | 287 | 89 | 50 | 2614 |
| +1 to +2 | 1 | 922 | 3156 | 1119 | 373 | 117 | 25 | 5713 |
| +2 to +3 | 0 | 251 | 4932 | 1948 | 331 | 45 | 18 | 7525 |
| +3 to +4 | 0 | 0 | 2268 | 2245 | 236 | 26 | 7 | 4782 |
| +4 to +5 | 0 | 0 | 102 | 1865 | 123 | 19 | 1 | 2110 |
| +5 to +6 | 0 | 1 | 3 | 476 | 243 | 8 | 0 | 731 |
| +6 to +7 | 0 | 0 | 0 | 20 | 132 | 25 | 1 | 178 |
| +7 to +8 | 0 | 0 | 0 | 1 | 7 | 52 | 9 | 69 |
| >= 8 | 0 | 0 | 0 | 0 | 1 | 4 | 23 | 28 |
| all | 110 | 2601 | 11970 | 8422 | 1926 | 456 | 172 | 25657 |

BP-RP >= 0.9: 7718 (30.1%). BP-RP >= 0.9 and M_G < 3.5: 4576 (17.8%). BP-RP >= 0.9 and M_G < 2.0: 2432 (9.5%). The last two sets contain red giants and subgiants and also reddened hotter stars. `logg_gspphot` (measured, `q07b_eb_logg_gspphot_grvs_le12`):

| G_RVS | N | % with logg_gspphot | % logg < 3.5 (of those) | % logg < 3.0 (of those) | % logg < 3.5 (of all) |
|---|---|---|---|---|---|
| <= 12 | 27252 | 75.3 | 12.7 | 3.7 | 9.6 |
| <= 12 and parallax_over_error >= 10 | 25658 | 75.7 | 11.7 | 3.3 | 8.9 |
| <= 10 | 3900 | 71.4 | 17.7 | 5.9 | 12.7 |
| 10-11 | 6459 | 74.2 | 14.4 | 4.2 | 10.7 |
| 11-12 | 16893 | 76.6 | 11.0 | 3.1 | 8.4 |

### 2g. Sky distribution

Measured, `q08_eb_sky_grvs_le12`: 27252 sources with G_RVS <= 12 and 3900 with G_RVS <= 10. Each row is a 10 degree bin of absolute latitude; the ecliptic counts are given for the two hemispheres separately and the percentages for both together; "% of sky" is the area of the bin.

| abs. latitude (deg) | N, ecl_lat < 0 | N, ecl_lat > 0 | % in abs(ecl_lat) bin (<= 12) | same (<= 10) | N in abs(b) bin (<= 12) | % in abs(b) bin (<= 12) | same (<= 10) | % of sky |
|---|---|---|---|---|---|---|---|---|
| 0-10 | 1665 | 1584 | 11.9 | 11.1 | 13623 | 50.0 | 45.1 | 17.4 |
| 10-20 | 1903 | 1479 | 12.4 | 13.4 | 5589 | 20.5 | 20.2 | 16.8 |
| 20-30 | 1980 | 1543 | 12.9 | 12.7 | 3200 | 11.7 | 11.6 | 15.8 |
| 30-40 | 2179 | 1831 | 14.7 | 14.5 | 1893 | 6.9 | 8.2 | 14.3 |
| 40-50 | 2673 | 2241 | 18.0 | 17.4 | 1263 | 4.6 | 6.3 | 12.3 |
| 50-60 | 2444 | 2192 | 17.0 | 17.3 | 781 | 2.9 | 3.8 | 10.0 |
| 60-70 | 1288 | 1098 | 8.8 | 8.3 | 522 | 1.9 | 2.7 | 7.4 |
| 70-80 | 477 | 393 | 3.2 | 4.0 | 284 | 1.0 | 1.5 | 4.5 |
| 80-90 | 132 | 150 | 1.0 | 1.2 | 97 | 0.4 | 0.6 | 1.5 |

## 3. Per-transit velocity precision of single stars in DR3

Quoted, [K23] 3.6.2 eq. (1): eps = [ (sqrt(pi/(2N)) sigma_Vt)^2 + (0.113)^2 ]^0.5, "where N is the number of transits used to derive the median radial velocity, sigma_Vt ... the standard deviation of the epoch radial velocity time series", and "the constant term, 0.113 km/s, is meant to take into account the wavelength calibration errors and similar sources of uncertainties". The DR3 data model prints the constant as 0.11. Measured: the smallest `radial_velocity_error` among the 400,373 `rv_method_used = 1` sources with `grvs_mag` < 9 is 0.11300176 km/s (`q30_rv_error_floor_check`), so 0.113 is the value in the catalogue. Each source is inverted as sigma_Vt = sqrt((error^2 - 0.113^2) 2 N / pi) with N = `rv_nb_transits`.

Selection (measured, `q31_sigma_vt_dwarfs_grvs_5_9_full`, all sources with 5 <= `grvs_mag` < 9, and `q32_sigma_vt_dwarfs_grvs_9_12_random10pct`, sources with `random_index` < 181,170,977): `rv_method_used = 1`, `rv_nb_transits >= 8`, `rv_template_logg >= 3.5`, `vbroad` < 20 km/s or null. Histograms of log10 sigma_Vt in 0.05 dex bins; the median is interpolated inside the bin. Variability cut: a cut on `rv_amplitude_robust` is not usable for this purpose, because the amplitude and the published error are functions of the same scatter (their ratio is about 3 to 4 for a constant star and for a binary alike). The cut applied is the complement of the variability criterion of [K23] 3.7, `rv_chisq_pvalue <= 0.01 & rv_renormalised_gof > 4`, which [K23] state for `rv_nb_transits >= 10` and template temperatures of 3900 to 8000 K and which is applied here to every selected source. `rv_renormalised_gof` exists only for `grvs_mag >= 5.5` and templates below 14,500 K ([K23] App. B), so the first row has no cut. The criterion removes 12 to 33% of the sources between 4500 and 7500 K at `grvs_mag` > 5.5; without it the medians of the 5500-6500 K class are higher by a factor 1.06 to 1.10. The median of a sample standard deviation of 8 to 20 values lies 2 to 5% below the true value.

Median sigma_Vt in km/s (number of sources in parentheses):

| grvs_mag | 3500-4500 K | 4500-5500 K | 5500-6500 K | 6500-7500 K | 7500-10000 K | > 10000 K |
|---|---|---|---|---|---|---|
| 5.0-5.5 | 0.85 (104) | 0.18 (58) | 0.18 (145) | 0.22 (51) | 3.16 (20) | - (6) |
| 5.5-6.0 | 0.57 (73) | 0.17 (58) | 0.19 (267) | 0.23 (66) | 0.23 (15) | - (4) |
| 6.0-6.5 | 0.62 (109) | 0.18 (163) | 0.20 (523) | 0.26 (136) | 0.28 (34) | 0.63 (10) |
| 6.5-7.0 | 0.67 (207) | 0.20 (258) | 0.22 (1056) | 0.28 (313) | 0.31 (63) | 0.94 (26) |
| 7.0-7.5 | 0.71 (467) | 0.23 (540) | 0.26 (1889) | 0.33 (573) | 0.39 (101) | 1.50 (21) |
| 7.5-8.0 | 0.75 (861) | 0.27 (1066) | 0.31 (3629) | 0.40 (913) | 0.49 (215) | 1.40 (47) |
| 8.0-8.5 | 0.75 (1173) | 0.34 (2043) | 0.41 (6341) | 0.55 (1614) | 1.25 (641) | 2.27 (87) |
| 8.5-9.0 | 0.72 (1788) | 0.42 (4463) | 0.52 (12539) | 0.73 (3432) | 3.03 (2074) | 4.47 (230) |
| 9.0-9.5 | 0.74 (285) | 0.53 (918) | 0.68 (2739) | 0.90 (651) | 4.99 (438) | 6.01 (47) |
| 9.5-10.0 | 0.88 (348) | 0.72 (1733) | 0.91 (5049) | 1.17 (1125) | 5.27 (571) | 10.19 (93) |
| 10.0-10.5 | 1.18 (637) | 1.03 (3402) | 1.27 (8586) | 1.54 (1793) | 7.09 (769) | 14.42 (162) |
| 10.5-11.0 | 1.58 (1140) | 1.47 (6100) | 1.82 (14238) | 2.21 (3316) | 13.07 (1463) | 23.12 (306) |
| 11.0-11.5 | 2.39 (1938) | 2.25 (10017) | 2.71 (23859) | 3.35 (4773) | 20.02 (2931) | 38.88 (464) |
| 11.5-12.0 | 3.76 (3081) | 3.69 (15178) | 4.31 (39484) | 6.23 (5829) | 26.66 (4796) | 62.01 (529) |

Two columns need a caveat. The 3500-4500 K column brighter than about 9.5 is not a clean dwarf sample (flat at 0.6 to 0.9 km/s, 28 to 61% flagged variable). In the columns above 6500 K the selection changes near `grvs_mag` = 8 (strongly above 7500 K, mildly at 6500-7500 K): `vbroad` is published for most hot stars brighter than that, so the rows hold slow rotators, and is null for most fainter ones, so the rows hold all rotators. The same statistic resolved by `vbroad`, in 1 mag bins, for sources not flagged variable (measured, `q33_sigma_vt_by_vbroad_grvs_5_9_full`, `q34_sigma_vt_by_vbroad_grvs_9_12_random10pct`; the classes 3500-4500 K and > 10,000 K are in the CSV files):

| rv_template_teff (K) | vbroad (km/s) | 5-6 | 6-7 | 7-8 | 8-9 | 9-10 | 10-11 | 11-12 |
|---|---|---|---|---|---|---|---|---|
| 4500-5500 | < 20 | 0.18 (64) | 0.19 (256) | 0.25 (901) | 0.41 (3449) | 0.68 (1575) | 1.35 (5917) | 2.77 (9763) |
| 4500-5500 | 20-50 | - (3) | - (3) | - (6) | 0.83 (19) | 1.19 (12) | 2.33 (65) | 4.19 (278) |
| 4500-5500 | not published | 0.18 (52) | 0.19 (165) | 0.25 (705) | 0.38 (3057) | 0.62 (1076) | 1.24 (3585) | 3.22 (15432) |
| 5500-6500 | < 20 | 0.19 (215) | 0.23 (722) | 0.31 (2380) | 0.51 (6918) | 0.88 (3587) | 1.67 (9932) | 3.43 (11992) |
| 5500-6500 | 20-50 | - (9) | 0.46 (50) | 0.60 (293) | 0.97 (970) | 1.67 (324) | 3.04 (530) | 5.39 (800) |
| 5500-6500 | 50-100 | - (1) | - (6) | 0.92 (17) | 1.51 (55) | 2.66 (10) | 4.04 (15) | 7.36 (36) |
| 5500-6500 | not published | 0.18 (197) | 0.21 (857) | 0.28 (3138) | 0.46 (11962) | 0.78 (4201) | 1.55 (12892) | 3.69 (51351) |
| 6500-7500 | < 20 | 0.23 (105) | 0.28 (403) | 0.38 (1249) | 0.58 (2308) | 1.02 (1009) | 1.96 (2911) | 4.07 (3023) |
| 6500-7500 | 20-50 | 0.35 (103) | 0.40 (375) | 0.56 (1107) | 0.88 (1710) | 1.58 (650) | 3.04 (1834) | 6.23 (2421) |
| 6500-7500 | 50-100 | 0.52 (54) | 0.70 (209) | 0.97 (741) | 1.46 (1495) | 2.67 (443) | 5.13 (1250) | 10.57 (1868) |
| 6500-7500 | >= 100 | 1.07 (64) | 1.23 (232) | 1.64 (747) | 2.53 (1501) | 4.74 (454) | 8.69 (1111) | 17.57 (1417) |
| 6500-7500 | not published | 0.21 (12) | 0.23 (46) | 0.34 (237) | 0.88 (2738) | 1.22 (767) | 1.98 (2198) | 5.00 (7579) |
| 7500-10000 | < 20 | 0.45 (26) | 0.32 (75) | 0.49 (232) | 0.80 (422) | 1.47 (113) | 3.97 (493) | 12.21 (1092) |
| 7500-10000 | 20-50 | 0.47 (61) | 0.56 (158) | 0.80 (500) | 1.34 (970) | 2.54 (326) | 5.56 (1174) | 12.42 (1706) |
| 7500-10000 | 50-100 | 0.82 (91) | 0.91 (287) | 1.32 (1014) | 2.18 (2005) | 4.09 (632) | 8.20 (1859) | 15.68 (2280) |
| 7500-10000 | >= 100 | 1.47 (234) | 1.75 (755) | 2.54 (2673) | 4.09 (5137) | 7.62 (1321) | 14.01 (3004) | 25.60 (2852) |
| 7500-10000 | not published | - (9) | 0.24 (22) | 0.36 (84) | 3.12 (2293) | 5.83 (896) | 12.25 (1739) | 25.66 (6635) |

Values at integer magnitudes from the first table (geometric mean of the two adjacent bins; the last column is a log-linear extrapolation of the last two bins):

| rv_template_teff | 6.0 | 7.0 | 8.0 | 9.0 | 10.0 | 11.0 | 12.0 (extrapolated) |
|---|---|---|---|---|---|---|---|
| 4500-5500 K | 0.18 | 0.21 | 0.30 | 0.47 | 0.86 | 1.82 | 4.74 |
| 5500-6500 K | 0.19 | 0.24 | 0.36 | 0.59 | 1.07 | 2.22 | 5.44 |
| 6500-7500 K | 0.25 | 0.30 | 0.47 | 0.81 | 1.34 | 2.72 | 8.49 |

The plateau of 0.17 to 0.20 km/s at `grvs_mag` < 6.5 in the 4500-6500 K classes is an upper bound on the per-transit zero-point scatter, since it still contains photon noise, template mismatch and stellar jitter. With the per-transit S/N of the ESA model (52, 18 and 4.6 per pixel at G_RVS = 8, 10 and 12), the 5500-6500 K values correspond to sigma_Vt times S/N of about 16, 19 and 25 km/s at those magnitudes once a plateau of 0.18 km/s is removed in quadrature (derived here, not quoted).

Quoted from [K23] sect. 8, on the formal precision of the combined velocity (median of `radial_velocity_error`): "1.3 km/s at grvs_mag = 12 mag and 6.4 km/s at grvs_mag = 14 mag" for the whole catalogue; for a solar-metallicity red clump star "~125 m/s at grvs_mag = 6 mag, ~230 m/s at grvs_mag = 10 mag and ~5.2 km/s at grvs_mag = 14 mag"; "the median formal precision is weakly dependent on the surface gravity"; it "improves significantly as the effective temperature decreases, in particular between 15 000 and 6000 K ... In hot stars, it is dominated by broad and shallow lines of the Paschen series"; it improves with metallicity; the formal uncertainties are "somewhat under-estimated for grvs_mag < 11-12 mag, and over-estimated for fainter stars". [K23] give no table by temperature; the dwarf samples of their Fig. 9 run from d1 (`rv_template_teff` <= 3750 K) to d7 (>= 10,000 K). Consistency check (derived here): the measured sigma_Vt of the 4500-5500 K dwarf class with N = 18 in eq. (1) gives 0.12 km/s at 6 and 0.28 km/s at 10, against the quoted 0.125 and 0.23 km/s for red clump giants.

## 4. Non-single-star cross-check

Counts in `gaiadr3.nss_two_body_orbit` by solution type and G (measured, `q20_nss_types_by_g`); the totals agree with POP-note A.1.

| G (mag) | EclipsingBinary | EclipsingSpectro | SB2 | SB2C | SB1 |
|---|---|---|---|---|---|
| 3-4 | 0 | 0 | 1 | 0 | 0 |
| 4-5 | 0 | 0 | 6 | 1 | 0 |
| 5-6 | 0 | 0 | 29 | 6 | 9 |
| 6-7 | 11 | 0 | 109 | 6 | 302 |
| 7-8 | 24 | 0 | 321 | 34 | 1740 |
| 8-9 | 74 | 3 | 536 | 98 | 3213 |
| 9-10 | 188 | 15 | 1154 | 191 | 7893 |
| 10-11 | 385 | 39 | 2121 | 352 | 23860 |
| 11-12 | 817 | 62 | 351 | 58 | 58880 |
| 12-13 | 2089 | 36 | 2 | 0 | 74030 |
| 13-14 | 5368 | 0 | 0 | 0 | 11348 |
| >= 14 | 77962 | 0 | 0 | 0 | 52 |
| all | 86918 | 155 | 4630 | 746 | 181327 |

The same solutions restricted to sources that are in `vari_eclipsing_binary`, with the number of candidates per bin for scale (measured, `q21_nss_types_among_eb_candidates_by_g`, `q01c_eb_counts_by_g`):

| G (mag) | < 8 | 8-9 | 9-10 | 10-11 | 11-12 | 12-13 | 13-14 | >= 14 | all |
|---|---|---|---|---|---|---|---|---|---|
| all EB candidates | 311 | 566 | 1421 | 3367 | 8595 | 19888 | 40720 | 2109609 | 2184477 |
| EB candidates with radial_velocity | 70 | 145 | 484 | 1493 | 6676 | 13190 | 23325 | 31441 | 76824 |
| EclipsingBinary | 35 | 74 | 188 | 385 | 817 | 2089 | 5361 | 77903 | 86852 |
| SB1 | 6 | 23 | 118 | 394 | 1029 | 822 | 15 | 0 | 2407 |
| SB1C | 0 | 0 | 0 | 0 | 3 | 5 | 0 | 0 | 8 |
| SB2 | 37 | 45 | 96 | 193 | 36 | 0 | 0 | 0 | 407 |
| SB2C | 5 | 15 | 24 | 68 | 14 | 0 | 0 | 0 | 126 |
| EclipsingSpectro | 0 | 3 | 15 | 39 | 62 | 36 | 0 | 0 | 155 |
| Orbital | 1 | 4 | 21 | 36 | 45 | 115 | 133 | 100 | 455 |
| AstroSpectroSB1 | 0 | 0 | 4 | 2 | 4 | 10 | 1 | 0 | 21 |

Of the 5665 candidates with G < 11, 483 (8.5%) have a DR3 double-lined orbit (`SB2` or `SB2C`); 533 have one at any magnitude, and none is fainter than G = 12.

## 5. What could not be established

1. The number of DR4 sources with `has_epoch_rvs = 3` and the number of `rv_assumed_sb2` sources: neither is in [DR4C] or [DR4M]; the 7.5 million above is a DR3-based estimate.
2. Whether every source has `has_epoch_photometry` set: the record count implies near-complete coverage, the draft gives no fraction.
3. The DR4 content of `vari_eclipsing_binary` beyond the draft column list (selection, period range, completeness): the two DR4 papers are not public, and the draft may change before release.
4. The equations behind `external_apparent_grvs`: the draft cites the paper, not the equation numbers.
5. The DR3 paper on the double-lined processing (cited as Damerdji et al. in [K23] and [DOC3]) was not located, so the template temperature range and the transit minimum of the DR3 SB2 channel rest on the documentation sentences quoted in section 1 and on the measured minimum of 10 good transits.
6. A published per-transit wavelength zero-point scatter: only the 0.113 km/s floor of the combined velocity is documented.
7. Why most bright eclipsing binaries have no DR3 velocity: DR3 has no per-source flag for the reason.
8. The completeness of the DR3 catalogue at G_RVS <= 12 (only faint OGLE4 comparisons are published), and the exact lower boundary of sample 2G-A (6514 sources, section 2d).
9. Dereddened colours and component temperatures: no extinction correction was attempted; `teff_gspphot` is null for 25% of the G_RVS <= 12 sample and is a single-star fit.
10. Journal numbering of the [M23] tables and of the [K23] equation: the arXiv versions were read; [DR4M] refers to the sample table as Table A.1.

## 6. Facts most consequential for the simulated population

1. The DR4 pipeline gates on `external_apparent_grvs`, in all likelihood the same predicted G_RVS as here: epoch velocities to 14; double-line search, `vbroad` and polynomial normalisation to 12; median-based combined velocity and valid `vbroad` to 11. DR3 holds 27252 eclipsing-binary candidates brighter than 12 on that scale, 10359 brighter than 11 and 3900 brighter than 10, about 0.35% of all sources at each magnitude.
2. At G_RVS <= 12 the candidates are mostly F and G in colour (41% and 29%), with 8% A, 0.5% bluer than BP-RP = 0, 20% K and 1.3% M; at G_RVS <= 10 the A share is 21% and the blue share 2.5%. The hot shares are lower limits because of reddening; GSP-Phot puts 12% of its values at or above 10,000 K.
3. Periods are short: median 1.27 d at G_RVS <= 12, 80% between 0.37 and 6.1 d, 6% longer than 10 d.
4. Light-curve morphology at G_RVS <= 12 is 44% wide, 44% tight and 13% unassigned, and depends steeply on period (wide share 11 to 16% from 0.25 to 0.63 d, 56 to 66% from 1 to 6.3 d); only 16% of the candidates are in the samples with flat light outside eclipse. DR4 publishes the sample label (`sample_classification`) and a flag comparing the photometric period with the velocity periodogram.
5. DPAC's own per-transit scatter for 5500-6500 K dwarfs with `vbroad` below 20 km/s or unpublished is 0.19, 0.36, 1.07 and about 5.4 km/s at G_RVS = 6, 8, 10 and 12, and is 2.5 to 5 times larger at `vbroad` of 50 km/s and above; the eclipsing binaries with a published `vbroad` (38%) have a median of 82 km/s.
6. In DR3 only 33% of the candidates brighter than G_RVS = 10 kept a velocity (23 to 39% per magnitude bin), and 8.5% of those with G < 11 received a double-lined orbit; in DR4 such sources are expected under `rv_assumed_sb2`, without combined velocity or mean spectrum but with epoch spectra, and their double-lined epoch spectra carry `normalisation_method = 0`.
