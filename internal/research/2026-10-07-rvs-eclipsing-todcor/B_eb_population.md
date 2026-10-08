# A population of eclipsing double-lined binaries for a Gaia RVS simulation: literature basis

Research date 2026-10-07. Scope: a sourced recipe for drawing detached eclipsing binaries bright enough for RVS epoch spectra, with validation targets. Design decisions are left to the reader.

Verification tags. **[V]** read in the source text during this work, location given (arXiv PDF converted to text; kept in `papers/`). **[A]** abstract only. **[C]** computed during this work, either a server-side aggregate on a public catalogue (TAP, nothing downloaded) or a few-line calculation (`quantiles.py`, `third_light.py`, `third_light_check.py`, `eclipse_numbers.py`, raw numbers in `notes_raw.md`). **[R]** recalled, not checked against the source. Bibcodes in the reference list were checked against the SIMBAD reference table unless marked.

Repo items are not re-derived: the Moe & Di Stefano (2017, MDS17) transcription, the Winn (2010) geometry and the Meibom & Mathieu (2005) circularisation table are in `internal/research/2026-09-09-gaia-rvs-benchmark/research_binary_populations.md` (sections F.2, F.6, F.3).

## 1. Makeup of magnitude-limited eclipsing-binary samples

### 1.1 Class shares

| Sample (limit) | N | Detached | Semi-detached | Contact | Ellipsoidal / other | Tag |
|---|---|---|---|---|---|---|
| ASAS, V < 14, dec < +28 (Paczynski et al. 2006) | 11,099 | 2,758 (24.8%) | 2,957 (26.6%) | 5,384 (48.6%) | | [A] |
| ASAS-SN variable-star database, all magnitudes (VizieR II/366, 2021 version; EA / EB / EW) | 154,530 | 50,095 (32.4%) | 25,932 (16.8%) | 78,503 (50.8%) | 70 ELL | [C] |
| same, V < 13 | 25,065 | 11,042 (44.1%) | 5,909 (23.6%) | 8,114 (32.4%) | | [C] |
| same, V < 12 | 8,392 | 3,957 (47.2%) | 2,087 (24.9%) | 2,348 (28.0%) | | [C] |
| Kepler, manual classes (Prsa et al. 2011, Sect. 3.4) | 1,879 | 52.3% | 7.5% | 25.4% | 7.5% ELV, 7.3% uncertain | [V] |
| Kepler (Slawson et al. 2011, Sect. 6) | 2,165 | 1,261 | 152 | 469 | 137 ELV, 147 uncertain | [V] |
| Kepler final catalogue, morphology c (VizieR J/AJ/151/68; c < 0.5, 0.5-0.7, 0.7-0.8, > 0.8) | 2,702 classified | 1,397 (51.7%) | 426 (15.8%) | 294 (10.9%) | 585 (21.7%) | [C] |
| TESS 2-min targets, sectors 1-26 (VizieR J/ApJS/258/16; same c bins) | 4,581 | 2,206 (48.2%) | 1,393 (30.4%) | 517 (11.3%) | 465 (10.2%) | [C] |
| OGLE bulge (Soszynski et al. 2016, Sect. 4) | 450,598 | 338,633 detached + semi-detached (75.2%) | | 86,560 (19.2%) | 25,405 (5.6%) | [V] |
| OGLE LMC (Wyrzykowski et al. 2003, quoted by Prsa et al. 2011) | 2,768 | 68.0% | 25.9% | 6.1% | | [V, secondary] |
| Gaia DR3 (Mowlavi et al. 2023, Table A.1) | 2,184,477 | Sample 2G-A "well detached": 285,320 (13.1%) | | Sample 2G-B "tight": 834,093 (38.2%); 2GE-B: 265,276 | 0GE: 36,227 | [V] |
| Gaia DR3, G < 13 | 34,148 | Sample 2G-A (my ADQL of Table A.1): 2,757 (8.1%) | | | | [C] |

Notes. The Kepler morphology boundaries are those of Kirk et al. (2016, Sect. 6): c < 0.5 predominantly detached, 0.5-0.7 semi-detached, 0.7-0.8 overcontact, above 0.8 ellipsoidal [V]. The classes are light-curve classes: "EB" and c = 0.5-0.7 contain close detached pairs with ellipsoidal variation as well as true semi-detached systems, and Mowlavi et al. (2023, Sect. 3.1) state that semi-detached systems with a faint lobe-filling star fall in Sample 2G-A [V]. The Gaia sample classes are defined on Gaussian widths and do not map onto EA/EB/EW. My ADQL transcription of the 2G-A definition returns 278,806 sources against the published 285,320 (2.3% low), so the G < 13 count is approximate. ASAS-SN targets brighter than about 11 mag are close to saturation (Rowan et al. 2022, Sect. 2.3 [V]); only 579 of the 11,042 EA systems are brighter than V = 11 [C]. For OBA dwarfs, IJspeert et al. (2021) find 3,155 eclipsing candidates among 91,193 TESS light curves and give no class shares [A].

Counts at bright magnitudes [C]: Gaia DR3 EB candidates 2,298 (G < 10), 5,665 (G < 11), 14,260 (G < 12), 34,148 (G < 13); ASAS-SN EA 3,957 (V < 12) and 11,042 (V < 13); Rowan et al. (2022) detached systems 3,429 (G < 12) and 9,063 (G < 13).

### 1.2 Period distribution of detached systems

Quantiles in days, interpolated from histograms of 0.1 dex (first four rows) or 0.25 dex (last two) [C].

| Sample | N | 5% | 16% | 25% | 50% | 75% | 84% | 95% | P < 1 d | P < 10 d |
|---|---|---|---|---|---|---|---|---|---|---|
| ASAS-SN EA, V < 13 | 11,042 | 0.61 | 1.10 | 1.42 | 2.42 | 3.99 | 5.08 | 9.40 | 13.1% | 95.6% |
| Rowan et al. (2022) detached, G < 13 | 9,063 | 0.83 | 1.25 | 1.55 | 2.50 | 3.95 | 4.98 | 8.31 | 8.5% | 96.8% |
| Gaia DR3 Sample 2G-A (approx.), G < 13 | 2,757 | 0.47 | 1.08 | 1.45 | 2.56 | 4.56 | 6.08 | 14.5 | 14.0% | 92.3% |
| Gaia DR3 all EB candidates, G < 13 | 34,148 | 0.32 | 0.44 | 0.57 | 1.21 | 2.65 | 3.90 | 10.7 | 43.7% | 94.6% |
| Kepler, c < 0.5 | 1,397 | 1.40 | 2.81 | 3.97 | 9.90 | 30.0 | 56.5 | 253 | 1.8% | 50.3% |
| TESS 2-min, c < 0.5 | 2,206 | 1.02 | 1.92 | 2.46 | 4.62 | 9.14 | 12.7 | 23.2 | 4.7% | 78.1% |

The whole Rowan et al. (2022) catalogue (35,464 systems, median V = 13.9) spans 0.35 to 484.95 d with a median of 2.18 d (Sect. 2.2 [V]). Published descriptions: Kepler shows an excess at P about 0.3 d and a broad peak at 2-3 d (Kirk et al. 2016, Sect. 10.1 [V]); TESS shows a narrow peak near 0.25 d and a broad peak near 3 d (Prsa et al. 2022, Sect. 5, read through an automated summary of the ar5iv page); the Gaia wider samples peak near 1 d with a tail beyond 20 d and the tight samples at 0.35 d (Mowlavi et al. 2023, Sect. 3.3 [V]); OGLE peaks at 0.40 d with a cut-off at 0.22 d (Soszynski et al. 2016, Sect. 5 [V]). The three sparsely sampled, magnitude-limited samples agree on a median of 2.4 to 2.6 d. The Kepler and TESS medians are longer because continuous photometry detects narrow eclipses and because c < 0.5 excludes close detached pairs with ellipsoidal variation.

### 1.3 Detached share against period

ASAS-SN, V < 13, share of each class in bins of 0.2 dex [C].

| P [d] | N | EA | EB | EW |
|---|---|---|---|---|
| 0.20-0.32 | 1,330 | 0.10 | 0.03 | 0.87 |
| 0.32-0.50 | 5,492 | 0.04 | 0.13 | 0.82 |
| 0.50-0.79 | 3,812 | 0.14 | 0.46 | 0.40 |
| 0.79-1.26 | 3,182 | 0.41 | 0.39 | 0.19 |
| 1.26-2.00 | 3,156 | 0.69 | 0.25 | 0.06 |
| 2.00-3.16 | 3,232 | 0.82 | 0.17 | 0.01 |
| 3.16-5.01 | 2,505 | 0.87 | 0.12 | 0.01 |
| 5.01-7.94 | 1,254 | 0.86 | 0.14 | 0.01 |
| 7.94-12.6 | 511 | 0.78 | 0.20 | 0.01 |
| 10 and above | 735 | 0.66 | 0.31 | 0.03 |

For P of 1 d and above the shares are EA 75%, EB 21%, EW 4%; below 1 d they are 12%, 26%, 62% [C]. Kepler gives the same shape with its own classes: c < 0.5 holds 0.2% of systems at 0.32-0.56 d, 8% at 0.56-1.0 d, 29% at 1.0-1.8 d, 54% at 1.8-3.2 d, 82% at 3.2-5.6 d and 95% at 5.6-10 d [C]. In Gaia DR3 at G < 13 the 2G-A share rises from 1% at 0.32-0.5 d to 18% at 5-8 d [C].

### 1.4 Other properties of the ASAS-SN detached sample (Rowan et al. 2022)

- Evolutionary state of the photometric primary from MIST isochrones on the Gaia colour-magnitude diagram; the subgiant branch runs from the terminal-age main sequence to R = 1.5 R_TAMS (Sect. 2.6 [V]): 22,392 main sequence, 4,213 subgiant, 649 giant (Sect. 3.3 [V]), that is 82.2%, 15.5%, 2.4% of the 27,254 systems with good parallaxes. At Gaia G < 13 the VizieR table gives 6,963 / 1,122 / 195, that is 84.1%, 13.6%, 2.4% [C]. The authors remark on "a large population of subgiant primaries" [V].
- Eccentricity: 83.0% have e < 0.05; 2,643 systems have e > 0.1, 560 have e > 0.25, 66 have e > 0.5 (Sect. 3.2 [V]). Share with e > 0.1 at G < 13 by period [C]: 4% (1-1.8 d), 6% (1.8-3.2 d), 10% (3.2-5.6 d), 17% (5.6-10 d), 33% (10-18 d). The fitted eccentricities come from survey light curves and have a noise floor (11% have e > 0.05 even at 1-1.8 d).
- Sum of fractional radii at G < 13 [C]: 12%, 34%, 54%, 78% and 95% of systems lie below 0.2, 0.3, 0.4, 0.5 and 0.6; median 0.38.

> **Recommended prescription.** Simulate detached systems only and state the excluded share: contact and semi-detached light-curve classes are 53 to 56% of catalogued eclipsing binaries at V < 12 to 13 (ASAS-SN), 75% in ASAS, and 48% of Kepler; they are 88% of systems below 1 d and 25% of systems at 1 d and above. Restrict the simulated periods to 0.5 d and above and expect the detached share of a real sample to follow Table 1.3. Use Tables 1.2 and 1.4 as targets, choosing the row that matches the assumed discovery channel (ground-based or Gaia photometry: median 2.4-2.6 d; space photometry: 5-10 d).

## 2. From a field binary population to an eclipsing sample

### 2.1 Geometric selection in published forward models

Every model below selects eclipsing systems by geometry from isotropic orientations, which is equivalent to weighting each binary by its eclipse probability.

| Work | Stars | Binary statistics | Detached condition | Eclipse and detection |
|---|---|---|---|---|
| Soderhjelm & Dischler 2005 (Hipparcos, LMC) [V Sects. 2, 3, 7] | Kroupa (2001) IMF, constant star formation over 0-12 Gyr, metallicity mix, BSE evolution (Hurley et al. 2002) | 80% duplicity; f(lg a) Gaussian, mean 1.5, sigma 1.5 (a in au); f(q) = 0.446 [1 + 2 n(q - 0.2, 0.3) + 2 n(q - 1, 0.05)]; e thermal at a > 1000 au, uniform at a < 10 au | BSE mass transfer | P_e(dm) = cos i(dm); counted only if dm > 0.1 mag for at least 5% of the orbit. "Rather ad hoc" but reproduces the Hipparcos period slope; 0.05 mag for 10% and 0.1 mag for 10% fit worse |
| Prsa, Pepper & Stassun 2011 (LSST) [V Sects. 2, 4] | none; light-curve parameters drawn directly | log P uniform on [-1, 4]; T2/T1 normal (1.0, 0.18); rho1 + rho2 uniform up to a limit that declines with log P; e exponential; sin i uniform above grazing | by construction | period recovery and S/N of 10 per point; 24 million EBs, 28% characterised, 25% of those double-lined |
| Sullivan et al. 2015 (TESS) [V Sects. 3.3, 4.2, Table 3] | TRILEGAL, Dartmouth radii for low masses | multiplicity 0.26, 0.34, 0.41, 0.50, 0.75 for 0.1-0.6, 0.6-0.8, 0.8-1.0, 1.0-1.4, > 1.4 Msun; log-normal periods (Duchene & Kraus 2013); dN/dq as q^gamma with gamma 0.3 (0.8-1.4 Msun) and -0.5 (above); e uniform to e_max(P) (Eq. 5) | a above the Roche limit R (3 M1/M2)^(1/3) (Eq. 18) | impact parameters (Eqs. 14-16); depth (R2/R1)^2 F1/(F1 + F2), no limb darkening (Eq. 17); S/N of 7.3. 97,461 EBs with I_C < 12 over 95% of the sky. Model 1.04 against observed 1.85 EBs per square degree at Kp < 12, 0.5 < P < 50 d |
| Kirk et al. 2016 (Kepler) [V Sect. 10.1] | pairs drawn from the 200,000 Kepler targets (KIC Teff, log g), masses and radii from the Torres et al. (2010) relations | period grid, -1 to 3 in log P | none | p = (rho1 + rho2)(1 +/- e sin w)/(1 - e^2); duty cycle 92%, two eclipses; catalogue completeness 89.1 +/- 3.5% |
| Wells & Prsa 2021 (Kepler) [V Sects. 2.3, 2.4, 3] | Galaxia (Besancon); coeval secondary chosen from the synthetic stars | multiplicity against mass after Arenou (2011), Raghavan et al. (2010), Duchene & Kraus (2013); flat q and e | a(1 - e) > 1.5 (R1 + R2) (Eq. 5); e uniform to min[1 - 1.5 (R1 + R2)/a, e_max of MDS17] (Eq. 6) | R1 + R2 > r cos i; "we do not consider the depth"; three eclipses. Result: log P flat above 3.2 d |
| Kochoska et al. 2017 (Gaia) [V Sect. 4] | Kepler EB light curves resampled at Gaia scanning-law times | | | 68% detectable with 87 points in five years; lost: long-period detached systems with narrow eclipses and very shallow eclipses; all-sky mean of 67 points gives less |

Yield predictions for Gaia: about 4 x 10^5 eclipsing binaries discovered, about 10^5 double-lined in RVS (Zwitter et al. 2003, Sect. 1 [V]); 10^6 eclipsing binaries, more than 10^5 at V < 15 with masses to 2%, at least 25% of those double-lined (Wilkinson et al. 2005, Sect. 3 [V]); four million (Eyer et al. 2013), seven million (Zwitter 2002), half a million (Dischler & Soderhjelm 2005), 12% spectroscopic (quoted by Kochoska et al. 2017, Sect. 1 [V, secondary]).

### 2.2 Detection thresholds imposed by surveys

| Survey | Effective threshold | Tag |
|---|---|---|
| Hipparcos | dm > 0.1 mag for at least 5% of the orbit (Soderhjelm & Dischler 2005) | [V] |
| ASAS-SN, V < 13, EA | amplitude distribution: 4.8% below 0.1 mag, 28% in 0.1-0.2 mag, then declining; cut-on near 0.1 mag | [C] |
| Gaia DR3 | G < 20, at least 16 cleaned field-of-view transits, skewness of G above -0.2 (Sect. 2.1); model rejected with phase coverage below 0.6 or fewer than three observations in an eclipse (Sect. 2.2.5); P > 0.2 d and global ranking above 0.4, that is, at least 35% of the variance explained (Sect. 2.3, Eq. 5) | [V] |
| Gaia DR3, G < 13 | derived primary depth: 140, 669, 1,621, 1,813 sources in successive 0.02 mag bins from zero, flat beyond; cut-on at 0.03-0.04 mag. Derived primary duration of Sample 2G-A cuts on at 0.04-0.06 in phase, where 48 transits give three in-eclipse points | [C] |
| TESS | 2-min catalogue "largely complete to SNR of 13" (Prsa et al. 2022, Sect. 5.1); smallest depths 0.5 to 1% in 30-min data (IJspeert et al. 2021, Sect. 5) | [V] |
| Kepler | geometric; 89% complete for two or more eclipses | [V] |

### 2.3 Completeness of the Gaia DR3 catalogue

Mowlavi et al. (2023, Sect. 4.2, Table 5 [V]) give completeness relative to OGLE-IV only: 37% (LMC), 48% (SMC), 26% (bulge) at G < 20 and 42%, 50%, 35% at G < 19. The 72% of OGLE systems that are missed consist of 40% not classified as eclipsing binaries, 5% with fewer than 16 transits and 27% removed by the final filters (percentages of the OGLE sample). They give no completeness against period or depth. Rowan et al. (2023b, Sect. 3) note that "detached eclipsing binaries are less likely to be identified in the sparse, sigma-clipped Gaia light curves" [V].

A direct measurement for bright systems [C]: ASAS-SN classes with V < 13 matched by Gaia source identifier to `vari_eclipsing_binary` (VizieR I/358/veb). Recovered: EA 7,047 of 10,948 (64%), EB 4,545 of 5,863 (78%), EW 5,836 of 8,069 (72%). For EA:

| Period [d] | 0.56-1.0 | 1.0-1.8 | 1.8-3.2 | 3.2-5.6 | 5.6-10 | 10-18 | 18-56 | 56-100 |
|---|---|---|---|---|---|---|---|---|
| Recovered | 0.67 | 0.75 | 0.70 | 0.63 | 0.54 | 0.49 | 0.34 | 0.11 |

| ASAS-SN amplitude [mag] | < 0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-1.0 | > 1.0 |
|---|---|---|---|---|---|---|---|
| Recovered | 0.23 | 0.52 | 0.64 | 0.70 | 0.79 | 0.82 | 0.78 |

These are conditional on ASAS-SN detection. Below 0.56 d the recovery of EA entries falls to 25% (0.32-0.56 d, 305 systems) and 4% (0.18-0.32 d, 130 systems); Gaia DR3 removed periods below 0.2 d and the two catalogues may disagree on period or class there. The number of G transits used for the G < 13 candidates has a median of 48 (16% and 84% quantiles near 31 and 64) [C]; DR4 covers 66 against 34 months, which scales the median to about 93.

> **Recommended prescription.** (1) Draw orientations isotropically and keep eclipsing systems, or weight each binary by p_ecl = (R1 + R2)/a times (1 + |e sin w|)/(1 - e^2); do not draw the inclination inside the eclipsing range without this weight. (2) Compute the primary eclipse depth for limb-darkened spheres and require 0.04 mag in G for a sample defined by Gaia photometry at G < 13, or 0.1 mag for one defined by ground-based surveys. (3) Require three or more photometric points in an eclipse: with N_G transits and an eclipse of duration w in phase (Sect. 5.3), keep with probability P(k >= 3), k Poisson with mean N_G w, for either eclipse; N_G is 48 (DR3) or about 93 (DR4) at the median. Multiply by a plateau efficiency of 0.75 to reproduce the measured DR3 recovery. (4) Alternative single rule: depth above 0.1 mag for at least 5% of the orbit. (5) For a sample defined by TESS, use geometry alone with a depth floor of 1%.

## 3. Close-binary physics that sets the spectra

### 3.1 Synchronisation

- Lurie et al. (2017): 816 Kepler eclipsing binaries with spot modulation; primaries by colour F (122), G (428), K (181), M0-M4 (8). "79% of EBs with orbital periods less than ten days are synchronized" (abstract). Below 2 d, 94% have 0.92 < P_orb/P_rot < 1.2. Between 2 and 10 d, 72% lie in that interval and 15% form a second cluster at P_orb/P_rot of 0.84-0.92 (centre 0.87), attributed to differential rotation of high-latitude spots on equatorially synchronised stars (Sect. 4.4.1). At 10 d there is "a transition from predominantly circular, synchronized EBs to predominantly eccentric, pseudosynchronized EBs" (abstract), with period ratios up to 50% below the prediction of Hut (1981, Eq. 42) (Sect. 4.4.2); beyond about 30 d the amount of synchronisation decreases (abstract). Sixty-one systems below 10 d have P_orb/P_rot < 0.6, of which 22 are probably not eclipsing binaries (Sect. 4.3.1) [V].
- Meibom, Mathieu & Stassun (2006): in M35 and M34 (150 and 250 Myr), of the six binaries below 13 d, two with circular orbits are not synchronised, two with eccentric orbits rotate slower than pseudo-synchronism, and two are circular and synchronised [V abstract]. This paper does not support a field threshold of 15 d; Lurie et al. give 10 d for the field.
- Torres et al. (2010, Sect. 7.1): for 95 well-studied detached systems, "pseudo-synchronisation is in fact an excellent approximation for the great majority of the stars; most of the non-synchronous cases are found below relative radius 0.1"; convective stars show less dispersion; some radiative stars with relative radii above 0.1 are sub-synchronous; larger primaries are usually synchronised while smaller secondaries are still evolving [V].
- Pseudo-synchronous rate (Hut 1981, Eq. 42): Omega_ps/n = (1 + 7.5 e^2 + 5.625 e^4 + 0.3125 e^6) / [(1 + 3 e^2 + 0.375 e^4)(1 - e^2)^1.5] [R; the equation number is from Lurie et al.].

### 3.2 Spin-orbit alignment

Marcussen & Albrecht (2022): 51 close double stars, obliquities for 39 from apsidal motion; 48 of 51 consistent with alignment; a Fisher distribution with concentration 6.7 describes the ensemble, which is consistent with perfect alignment under bootstrap; misaligned: DI Her and AS Cam (eccentric) and CV Vel (circular) [V abstract].

### 3.3 Circularisation and the eccentricity envelope

| Source | Sample | Result | Tag |
|---|---|---|---|
| Torres et al. 2010, Sect. 7.1 | 95 detached systems, 44 eccentric | none eccentric below 1.5 d; all circular for relative radius above 0.25; convective envelopes (Teff < 7000 K) circularise to longer periods | [V] |
| Van Eylen et al. 2016, Sects. 3, 4 | 945 Kepler EBs; hot means Teff > 6250 K | eccentric, defined as abs(e cos w) of 0.02 or more: at 1.5-4 d, 12 of 74 hot-hot, 2 of 156 cool-cool, 2 of 108 hot-cool; at 4-5 d, 2 of 14, 5 of 54, 2 of 29 | [V] |
| Justesen & Albrecht 2021, Table 4 | 809 EBs (TESS 349, Kepler 365, Torres 95) | fitted circularisation distance a/R1 and period by Teff: below 4500 K 29.8, 5.57 d (N = 58); 4500-6250 K 18.0, 9.0 d (193); 6250-8000 K 11.4, 5.5 d (109); 8000-10,000 K 8.4, 5.7 d (20); above 10,000 K 3.8, 2.8 d (29). First eccentric systems at a/R1 of 3.2 or 1.5 d; 17% eccentric at a/R1 < 10, 62% beyond; circular systems out to a/R1 of 50 | [V] |
| Zanazzi 2022, Figs. 2-3 | 524 Kepler and TESS EBs, 4500-7000 K | envelope period 3.2 (+0.6, -0.3) d; mean circularisation period 6.2 (+1.4, -0.8) d; a "cold core" of circular orbits to 10-20 d | [V] |
| IJspeert et al. 2024, Sect. 4, Table 2 | 14,573 TESS O to F EBs | P_env (5th percentile of eccentric systems), P_circ (ratio of the cumulative distributions of eccentric and all systems equal to one half), P_core (95th percentile of circular systems): below 7000 K 2.51 +/- 0.26, 6.65 +/- 0.27, 8.3 +/- 0.7 d; 7000-8000 K 1.98, 5.66, 6.8 d; 8000-10,000 K 1.90, 4.38, 7.7 d; above 10,000 K 1.62, 2.46, 8.2 d. In a/(R1 + R2): 2.84, 5.81, 8.2; 2.54, 5.21, 6.8; 2.47, 4.19, 7.2; 2.37, 3.10, 7.3 | [V] |
| Bashi et al. 2023, Eq. 10, Table 2 | 3,959 Gaia DR3 main-sequence SB1 | cut-off period P_cut = 27.6 (+/- 2.5) - 3.69 (+/- 0.39) Teff/1000 K days: 6.5 d at 5700 K, 2.5 d at 6800 K; flat near 3 d over 6800-7700 K and near 7 d below 5600 K; no dependence on age | [V] |
| Rowan et al. 2022 (Eq. 2); Mowlavi et al. 2023 (Fig. 33) | envelope of Mazeh (2008, Eq. 4.4) | e < 0.98 - 3.25 exp[-(6.3 P)^0.23], zero at 0.35 d, 0.27 at 1 d, 0.44 at 2 d, 0.62 at 5 d | [V, secondary; values C] |
| Schussler, Majumder & Penev 2026 (arXiv:2609.21087) | 105 Sun-like Kepler binaries | log Q' near 6.75 reproduces the overlap of circular and eccentric orbits | [A] |

The repo note (F.3) gives the cluster values of Meibom & Mathieu (2005): 10.3 d for the field. The eclipsing samples show two populations between about 2.5 and 10 d (circular and eccentric), a strong dependence on temperature, and for hot stars circular orbits well beyond the dynamical-tide prediction.

### 3.4 Mass ratio and twins at short period

| Source | Statement | Tag |
|---|---|---|
| MDS17 | solar-type, short period: excess twin fraction 0.29 +/- 0.11, slope -0.5 over 0.3 < q < 1; early-type SB2 at 2-20 d: 0.11 +/- 0.04, slope -0.3 +/- 0.3 | [V Sects. 3.4, 8] |
| Mazeh et al. 2003 | 62 binaries, primaries 0.6-0.85 Msun: "approximately constant over the range q = 1.0 to 0.3" | [A] |
| Tokovinin et al. 2006, Sect. 7.2 | spectroscopic binaries at 1-30 d: "a slight excess of systems with nearly equal-mass components (twins), but otherwise the distributions are almost uniform" | [V] |
| Lucy 2006 | narrow peak at q above 0.95 significant for double-lined orbits of high precision | [A] |
| Simon & Obbie 2009 | twins within 2% of q = 1 prominent below 43 d; about 3% of all spectroscopic binaries | [A] |
| Pinsonneault & Stanek 2006 | 21 detached SMC systems: 55% flat plus 45% with q > 0.95; more than 20-25% twins in general | [A] |
| El-Badry et al. 2019 | twin excess confined to q of 0.95 and above at all separations from 0.01 to 10,000 au, declining with separation | [A] |
| Tokovinin 2014; Raghavan et al. 2010 | nearly uniform q independent of period; twin excess, short periods prefer high q | [A; repo note F.1] |
| Kounkel et al. 2021, Sect. 4.2 | APOGEE SB2: "most of the systems in this sample have q near 1", attributed to the intrinsic distribution and to detection bias; no intrinsic distribution derived | [V] |
| Eker et al. 2018 table (293 detached double-lined systems) | q > 0.95: 30%; q > 0.9: 49%; q > 0.7: 83%; approximate counts per 0.05 bin from 0.70 upward: 20, 24, 23, 32, 57, 88 | [C] |

With the MDS17 solar-type parameters, 61% of systems with q > 0.7 have q > 0.95 before any selection; with an excess of 0.25 (1.5 to 2.5 Msun) the share is 56% [C]. The compilation of well-studied detached systems, which is selected in favour of equal components, shows 36% (88 of 244) and a rise that starts near q = 0.85. Either the excess is weaker than MDS17 for close binaries of 1 to 3 Msun or it is broader than 0.95 to 1. Gaia DR3 SB2 ratios K1/K2 cannot decide this: their distribution is symmetric about 1 (975 in 0.95-1.00, 924 in 1.00-1.05 at P < 10 d), which shows measurement scatter [C].

### 3.5 Radii and temperatures in an eclipsing, magnitude-limited sample

- Eker et al. (2018, Table 5, Sect. 3.2.1): the mass-radius relation below 1.5 Msun has a standard deviation of 0.176 Rsun; above 1.5 Msun the derived relation has 0.787 Rsun; "stellar evolution becomes effective to disperse observed radii towards the high mass end, so that the distribution of stars overflow the one sigma limit if M > 1.15 Msun". Their calibration sample keeps stars between the PARSEC zero-age and terminal-age lines and drops 46 above the terminal-age line (Sect. 2.2) [V]. Torres et al. (2010, Sect. 4): "the large range in radius for a given mass clearly shows the effect of stellar evolution"; the range in radius at fixed Teff reaches a factor of 3 [V].
- Spread measured on the full Eker et al. table (586 stars) [C]. For 1.0 to 2.5 Msun (N = 320), y = log R - 0.8 log M has 5%, 16%, 50%, 84% quantiles at -0.04, 0.01, 0.08, 0.22 dex, and 4% of stars lie above 0.40 (giants). Relative to a lower envelope at -0.05 the 16%, 50% and 84% quantiles are 0.06, 0.13 and 0.27 dex (factors 1.14, 1.35, 1.9). For 0.5 to 1.0 Msun (N = 93), y = log R - 0.9 log M has 48 stars in -0.05 to 0 and 20 in 0 to 0.05, with a tail of 18 stars to 0.2 dex.
- Bias. An eclipsing sample is weighted by (R1 + R2)/a, a magnitude-limited one by L^1.5, and long-period detection needs large radii. Rowan et al. (2022) find 15.5% subgiant and 2.4% giant primaries. Sullivan et al. (2015, Fig. 4) attribute the scatter in radius at absolute I_C brighter than 5 to stellar evolution [V].
- Prescriptions in published work: BSE tracks with constant star formation (Soderhjelm & Dischler 2005); TRILEGAL with Dartmouth radii (Sullivan et al. 2015); Galaxia ages with coeval pairing (Wells & Prsa 2021); observed Teff and log g of the target pool with the Torres relations (Kirk et al. 2016); Yonsei-Yale isochrones (Sybilski et al. 2013); MIST for classification (Rowan et al. 2022). None of the works read here uses a parametric scatter on radius.

### 3.6 Roche-lobe limit

Eggleton (1983): R_L/a = 0.49 q^(2/3) / [0.6 q^(2/3) + ln(1 + q^(1/3))], with q the mass of the star over that of its companion (form checked through Rowan et al. 2022, Eq. 5 [V]; accuracy better than 1% for all q [R]). For q = 1, R_L/a = 0.379, so equal stars fill their lobes at (R1 + R2)/a = 0.758 [C]. Detached: both stars inside their lobes; semi-detached: one fills its lobe; contact: both fill or exceed. Margins in use: fill factor below 70% at periastron (MDS17, Eq. 3 [V]), which is (R1 + R2)/a below 0.53 for twins; a(1 - e) > 1.5 (R1 + R2) (Wells & Prsa 2021 [V]); a(1 - e) > 2 (R1 + R2) (Sybilski et al. 2013, Eq. 1 [V]). The repo limit of 0.6 a corresponds to R/R_L = 0.79 for twins and has no periastron factor.

> **Recommended prescription.**
> Rotation. Stars with Teff below 6250 K: equatorial rotation pseudo-synchronous (Hut 1981, Eq. 42; synchronous for e = 0) for P < 10 d in 90% of systems, the rest drawn from the field distribution; between 10 and 30 d pseudo-synchronous in half; field rotation beyond 30 d. Hotter stars: pseudo-synchronous when R/a > 0.1, field rotation otherwise. Spin axes aligned with the orbit. Alternative: the present rule (synchronous below 15 d), which the cited cluster paper does not support.
> Eccentricity. Circular below P_env(Teff of the primary) = 2.5 d (below 7000 K), 2.0 d (7000-10,000 K), 1.6 d (above). Above it, a share f_ecc(P) = log(P/P_env) / [2 log(P_circ/P_env)], limited to [0, 1], is eccentric and the rest circular, with P_circ = 6.65, 5.66, 4.38, 2.46 d for below 7000, 7000-8000, 8000-10,000 and above 10,000 K (this form is constructed here to pass through the IJspeert et al. numbers). Eccentric systems take the MDS17 distribution under their e_max or the Mazeh envelope. Alternative for cool stars: P_cut = 27.6 - 3.69 Teff/1000 K (Bashi et al.).
> Mass ratio. Keep MDS17 and test the simulated double-lined sample against the Eker et al. shares (30% above 0.95, 49% above 0.9). Alternative: halve the twin excess or spread it over 0.9 to 1.
> Radii and temperatures. Preferred: isochrones (MIST or PARSEC), age uniform on [0, min(main-sequence lifetime, 10 Gyr)], common age and metallicity for both stars, with the magnitude-limit and eclipse weights applied afterwards. Fallback: R = R_env(M) 10^d with log R_env = -0.05 + 0.80 log M for M of 1 Msun and above and d drawn to match the quantiles 0.06, 0.13, 0.27 dex at 16%, 50%, 84%; below 1 Msun a normal scatter of 0.04 dex.
> Roche limit. Keep a system when each radius is below 0.75 R_L (Eggleton) at periastron separation a(1 - e).

## 4. Third light

- Tokovinin et al. (2006): 165 solar-type spectroscopic binaries with P of 1 to 30 d. Tertiary frequency after correction for incompleteness: 63 +/- 5% overall; 0.80 +/- 0.06 for P < 7 d (90 systems, raw 0.66) and 0.40 +/- 0.06 for P > 7 d (75 systems, raw 0.33) (Table 6); 96% for P < 2.9 d and 34% for P > 12 d (Sect. 7.4, Fig. 14, abstract). Tertiary periods run from 2 yr to 10^5 yr with no correlation with the inner period. The distribution of q3 = M3/M1 "does in fact appear rather uniform (Fig. 9)"; 16 tertiaries (17 +/- 4%) have q3 > 1 [V].
- Laos, Stassun & Mathieu (2020): 52 solar-type spectroscopic binaries imaged with Robo-AO at 0.75 micron: 20 binaries, 31 triples, 1 quadruple (62%); 75% for P < 3 d, 90% for 3-6 d, 47% for P > 30 d (Sect. 4). Kepler eclipsing binaries in the same paper: 0.68 +/- 0.47 (3-6 d, N = 3), 0.51 +/- 0.18 (6-30 d, N = 8), 0.34 +/- 0.24 (above 30 d, N = 6) (Table 4) [V].
- Tokovinin (2008): median outer mass ratio in triples 0.39, independent of outer period [A]. Tokovinin (2014): 13 +/- 1% of solar-type systems are hierarchies; component counts 54 : 33 : 8 : 4 : 1 [A]. MDS17 Table 13 (repo note F.2a): triple-or-higher fraction 0.10 for solar-type rising to 0.73 for O stars.
- Contact binaries: lower limit 42 +/- 5% in triples for 151 systems with V < 10, 59 +/- 8% in the better-observed northern part (Pribulla & Rucinski 2006 [A]); spectroscopic tertiaries in 23 of 75 close binaries down to a flux contribution of 0.8% at 5200 A, with M3/M12 between 0.28 and 0.75 (D'Angelo et al. 2006 [A]); companions at 10^3 to 10^4 au for 14.1 +/- 1.0%, 3.1 +/- 0.5 times the field (Hwang et al. 2020 [A]).
- Spectroscopic evidence in detached samples: APOGEE gives 7,273 SB2, 813 SB3 and 19 SB4, so 10% of multi-lined systems show a third star (Kounkel et al. 2021 [V]). Of 17 bright Kepler detached eclipsing binaries observed at high resolution, two are triple-lined and in two to four more the only visible lines belong to a third star (Helminiak et al. 2016, 2017 [A]). Eclipse timing finds third bodies in 222 of 2,600 Kepler binaries, limited to outer periods below about 2,000 d (Borkovits et al. 2016 [A]). ASAS-SN and TESS light curves show 225 candidate triples or doubly eclipsing systems among 35,464 detached systems, a conspicuous subset only (Rowan et al. 2023b, Sect. 2 [V]).
- No homogeneous distribution of the fitted third-light fraction for detached eclipsing binaries was found. The fraction is therefore derived [C] from the Tokovinin statistics: q3 uniform on [0.1, 1.2] (which gives 17% above 1), fluxes in Gaia G_RP from the mean dwarf sequence in `population.py`, l3 = F3/(F1 + F2 + F3).

| Inner pair | P(l3 > 2%) | P(l3 > 5%) | P(l3 > 10%) | P(l3 > 20%) | median l3 |
|---|---|---|---|---|---|
| 1.0 + 1.0 Msun | 0.62 | 0.50 | 0.42 | 0.28 | 0.050 |
| 1.0 + 0.7 Msun | 0.69 | 0.57 | 0.48 | 0.38 | 0.083 |
| 1.3 + 1.3 Msun | 0.63 | 0.54 | 0.44 | 0.34 | 0.067 |
| 1.6 + 1.6 Msun | 0.67 | 0.57 | 0.49 | 0.35 | 0.096 |
| 1.0 + 1.0 Msun, q3 limited to 1 | 0.53 | 0.39 | 0.30 | 0.13 | 0.025 |

These are conditional on a tertiary being present and unresolved. With outer periods log-uniform between 2 yr and 10^5 yr, 0.58, 0.68 and 0.77 of tertiaries lie within 1 arcsec at 150, 300 and 600 pc [C]. Check: the recipe predicts a tertiary above 0.8% of the G-band light within 1 arcsec in about 37% of contact binaries at 100 pc, against 31% (23 of 75) observed by D'Angelo et al. with incomplete detection, so it may be high by a factor of up to about 1.4.

> **Recommended prescription.** Tertiary probability f3(P) = 0.30 + 0.80 exp(-P/7 d), capped at 0.96 (fitted here to the four Tokovinin et al. values: 0.96 below 2.9 d, 0.80 below 7 d, 0.40 above 7 d, 0.34 above 12 d). Tertiary mass M3 = q3 M1 with q3 uniform on [0.1, 1.2]; outer period log-uniform between 2 yr and 10^5 yr with P3 > 5 P1; flux from the same sequence as the binary, same age and metallicity. Count the tertiary as third light when its projected separation at the system distance is inside the RVS window (the window size was not checked here; 1 arcsec is assumed below). Resulting share of close eclipsing binaries at 300 pc with an unresolved third star above 2%, 5%, 10% of the far-red light: about 0.40, 0.33, 0.27 for P < 3 d; 0.32, 0.26, 0.22 for 3-7 d; 0.19, 0.15, 0.13 for 7-30 d, possibly high by a factor of up to 1.4. Alternative: no third light, as now, with the Kounkel et al. 10% triple-lined share as the minimum that a real sample contains.

## 5. Eclipses and spectroscopy

### 5.1 Size of the Rossiter-McLaughlin distortion

The line-centroid anomaly of the eclipsed star is the flux-weighted mean rotational velocity of the hidden surface times f/(1 - f), f being the blocked fraction of that star's light (first-moment form of Ohta et al. 2005; Gaudi & Winn 2007, Eqs. 4-6 [V], who give K_R = v sin i g^2/(1 - g^2) for a small body of radius ratio g). Winn (2010, Eq. 40): maximum about k^2 sqrt(1 - b^2) v sin i [V]. Hirano et al. (2010) show that velocities from cross-correlation differ from the first moment when rotational broadening is appreciable [A]. Values for stellar eclipses, aligned rigid rotation, linear limb darkening 0.5 [C]:

| R2/R1, impact parameter b (units of R1) | max abs(dv)/(v sin i) | light of that star blocked at maximum | maximum blocked | rms over the eclipse |
|---|---|---|---|---|
| 1.0, 0.3 | 0.44 | 0.69 | 0.84 | 0.27 |
| 1.0, 0.6 | 0.27 | 0.49 | 0.65 | 0.18 |
| 1.0, 1.0 | 0.13 | 0.27 | 0.40 | 0.08 |
| 0.7, 0.0 | 0.33 | 0.49 | 0.55 | 0.20 |
| 0.7, 0.5 | 0.18 | 0.33 | 0.45 | 0.12 |
| 0.5, 0.0 | 0.17 | 0.26 | 0.29 | 0.11 |
| 0.5, 0.5 | 0.11 | 0.20 | 0.27 | 0.07 |

The synchronous velocity is 50.6 (R/Rsun)/(P/d) km/s. A 1.5 Rsun star at 2 d (38 km/s) in a partial eclipse that hides half of its light shows an anomaly of about 10 km/s; a 1.2 Rsun star at 5 d (12 km/s) shows about 3 km/s. The anomaly is antisymmetric about mid-eclipse for aligned spins and does not average out over a few transits.

### 5.2 Practice

"Moments of eclipses are avoided. It is a common approach made by observers which makes the RV modelling much easier" (Sybilski et al. 2013, Sect. 2 [V]). Modelling instead of exclusion is done on line profiles or broadening functions, not on velocities (Albrecht et al. 2007 for V1143 Cyg; Albrecht et al. 2009 for DI Her) [V, text of the first; title of the second]. Whether the Gaia DR3 spectroscopic-binary processing excluded in-eclipse transits was not checked.

### 5.3 Fraction of the orbit in eclipse

For a circular orbit the duration from first to last contact is (P/pi) arcsin[sqrt(rho^2 - cos^2 i)/sin i] with rho = (R1 + R2)/a (Winn 2010, Eq. 14 with the sum of radii [V]). Both eclipses together [C]:

| rho | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|---|---|
| edge-on | 0.032 | 0.064 | 0.096 | 0.128 | 0.194 | 0.262 | 0.333 | 0.410 |
| mean over eclipsing inclinations | 0.025 | 0.050 | 0.075 | 0.101 | 0.154 | 0.209 | 0.268 | 0.333 |

The inclination-averaged fraction is rho/2 to within 10% up to rho = 0.5. With the ASAS-SN median rho of 0.38, one fifth of transits fall in an eclipse.

### 5.4 Limb darkening in the RVS band

Claret's tables for Gaia (2019, RNAAS 3, 17; VizieR VI/154) contain G_BP, G and G_RP only; no G_RVS column exists [V at column level]. Claret & Southworth (2022) list Gaia among their bands [A]; their tables were not opened. Linear coefficients u from ATLAS models, least-squares method, log g = 4.5, solar metallicity, microturbulence 2 km/s [V, catalogue values; last row C]:

| Teff [K] | 4000 | 4500 | 5000 | 5500 | 5750 | 6000 | 6500 | 7000 | 7500 | 8000 | 9000 | 10,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cousins I (Claret & Bloemen 2011) | 0.579 | 0.603 | 0.564 | 0.519 | 0.496 | 0.473 | 0.434 | 0.404 | 0.379 | 0.363 | 0.339 | 0.314 |
| Sloan z' (Claret & Bloemen 2011) | 0.520 | 0.554 | 0.519 | 0.476 | 0.455 | 0.434 | 0.396 | 0.366 | 0.341 | 0.322 | 0.290 | 0.268 |
| Gaia G_RP (Claret 2019) | 0.599 | 0.622 | 0.580 | 0.532 | 0.508 | 0.485 | 0.445 | 0.415 | 0.391 | 0.375 | 0.351 | 0.325 |
| RVS, interpolated at 858 nm between I and z' | 0.55 | 0.58 | 0.54 | 0.50 | 0.47 | 0.45 | 0.41 | 0.38 | 0.36 | 0.34 | 0.31 | 0.29 |

These are continuum-band values; the cores of the Ca II and Paschen lines darken differently.

> **Recommended prescription.** Flag as in eclipse every transit within half the eclipse duration of either conjunction (Winn 2010, Eqs. 14-16, sum of radii) and report results with and without them; expect a share rho/2 of transits. If in-eclipse spectra are simulated, compute them by integration over the visible disc, since the velocity error reaches 0.1 to 0.4 v sin i. Linear limb darkening from the last row of Table 5.4 (0.55 at 4000 K falling to 0.29 at 10,000 K).

## 6. Other spectroscopic realities

- Spots. Saar & Donahue (1997, Eq. 1): A_S = 6.5 f_S^0.9 v sin i, with A_S in m/s, f_S the spot coverage in percent and v sin i in km/s, for a dark equatorial spot on a solar-like star, fitted for f_S up to 2% and accurate to 20% for A_S of 5 to 85 m/s; f_S is about 0.4 times the photometric amplitude [V; the unit of the amplitude, taken here as 0.01 mag, was not confirmed]. Extrapolated to close binaries: an amplitude of 0.05 mag (f_S = 2%) gives 12 m/s per km/s of v sin i, that is 0.4 km/s at 30 km/s and 0.7 km/s at 60 km/s; 0.10 mag gives about twice that. The contrast is lower at 860 nm than in V. Rowan et al. (2023b) flag 426 spotted systems among 35,464 from light curves, a lower limit [V].
- Tidal and reflection distortion. Synthetic detached pairs of 0.5 to 1.5 Msun at 3, 5 and 10 d with 2 (R1 + R2) < a(1 - e): the difference between the true and the Keplerian velocity curve has a mean amplitude of 174 m/s, a maximum of 3,800 m/s and a minimum of 0.25 m/s; fitted with a Keplerian model it yields spurious eccentricities, with a safe limit of 0.01 at 100 m/s precision (Sybilski et al. 2013, Sect. 3 [V]). For (R1 + R2)/a below 0.4 this is below the RVS single-transit error except in the closest pairs.
- Ca II triplet emission. RAVE (same band): 53,347 of 456,676 spectra have a candidate emission component; about 14,000 of 44,000 candidate active dwarfs show chromospheric flux at 2 sigma; the excess equivalent width per line runs from -0.2 to 1 A "or even more", the sum over the three lines peaks at 0.1 and 1.5 A, reaches 5 A in the most active stars, and has an uncertainty of 0.16 A; pipeline velocities of 61% of the candidate active spectra needed correction, by 5.25 km/s or more in 5% of the corrected spectra and by up to 10.5 km/s in 99% of them (Zerjal et al. 2013, Sects. 4, 5 [V]). Gaia DR3 activity index (Lanzafame et al. 2023, Sects. 2.3, 3 [V]): excess in the ratio to a template within 0.15 nm of each line core, for 2 x 10^6 stars with G < 13 and Teff of 3000 to 7000 K; values above 0.03 to 0.05 nm indicate accretion or "enhanced activity in close binaries due to tidal interaction"; the highest branch (log R'_IRT above -4.8 to -5.0) holds pre-main-sequence stars and RS CVn systems. Saturation: Rossby number below 0.13 +/- 0.02 with log tau = 1.16 - 1.49 log M - 0.54 log^2 M (tau in days, 0.09 to 1.36 Msun; Wright et al. 2011, Eq. 11 [V]). A synchronised star is therefore saturated below 1.25 d (1.3 Msun), 1.9 d (1.0), 2.6 d (0.8), 3.8 d (0.6 Msun) [C]; X-ray activity falls as Ro^-2.7 beyond. Torres et al. (2010, Sect. 2) excluded RS CVn systems from their sample for their activity [V].

> **Recommended prescription.** Spots: add to each cool synchronised star a velocity term of amplitude 6.5 f_S^0.9 v sin i m/s at the rotation phase, f_S drawn between 1 and 4%, halved for the far red. Proximity: neglect below (R1 + R2)/a = 0.4, or add 0.2 km/s in quadrature. Activity: for stars below 6500 K with Ro < 0.13, fill the three Ca II line cores with an excess equivalent width of 0.3 to 1.0 A per line (largest in 8542 A); scale the excess as (Ro/0.13)^-1 beyond saturation down to zero at Ro = 1 (the scaling is an assumption); test the template mismatch with inactive synthetic templates, since RAVE recorded velocity corrections of up to 10.5 km/s from this cause.

## 7. Requirements

Torres et al. (2010, Sect. 3 [V]): M_1,2 sin^3 i = 1.036149 x 10^-7 (1 - e^2)^(3/2) (K1 + K2)^2 K_2,1 P and a sin i = 1.976682 x 10^-2 (1 - e^2)^(1/2) (K1 + K2) P, in Msun, km/s, days and Rsun. Propagation, with x = K1/(K1 + K2):

- (sigma_M1/M1)^2 = (2x)^2 (sigma_K1/K1)^2 + (3 - 2x)^2 (sigma_K2/K2)^2 + [3e/(1 - e^2)]^2 sigma_e^2 + (3 cot i)^2 sigma_i^2 + (sigma_P/P)^2
- (sigma_M2/M2)^2 = (1 + 2x)^2 (sigma_K1/K1)^2 + (2 - 2x)^2 (sigma_K2/K2)^2 + the same last three terms

For equal masses and equal independent fractional errors s on K1 and K2, sigma_M/M = 2.24 s; for q = 0.5 the factors are 2.43 (primary) and 2.13 (secondary); the total mass has 3 sigma(K1 + K2)/(K1 + K2) [C]. Masses to 1%, 3% and 10% need s of 0.45%, 1.3% and 4.5%, that is 0.4, 1.2 and 4.2 km/s for K near 93 km/s (two solar-mass stars at 3 d). With N transits on a circular orbit of known ephemeris, sigma_K is about sigma_RV sqrt(2/N), so 20 transits allow 1.3, 3.9 and 13 km/s per transit [C]. The inclination term is negligible for eclipsing systems (3 cot i sigma_i is 0.5% at 85 degrees with 1 degree error).

Precision regarded as useful: a nominal 2% in Andersen (1991), as stated by Torres et al.; 3% hard limit, with 3% "too large for firm conclusions in the most demanding tests" (Torres et al. 2010, Sect. 2 [V]); 2% for DEBCat (Southworth 2015 [V]); 3%, 3 to 6% and 6 to 15% as "very accurate", "accurate" and "less accurate", with 15% the limit of use (Eker et al. 2018, Sect. 2.2 [V]); "around 1 per cent in mass and radius is then required" for opacities, convection and rotation, and an expected 2% for Gaia (Wilkinson et al. 2005, Sect. 3 [V]). Achieved with Gaia DR3 SB2 orbits and ASAS-SN light curves: 122 stars with median uncertainties of 7.9% in mass and 6.3% in radius, 4.8% in mass for the non-circular SB2 model, 61 stars below 10% in both; "only 50% of systems have Gaia periods and eccentricities consistent with the ASAS-SN values" (Rowan et al. 2023c [V]). Ground-based follow-up of Kepler systems reaches below 3% for five of eight pairs (Helminiak et al. 2017 [A]).

> **Recommended prescription.** Report the fraction of simulated systems reaching sigma_K/K below 0.45%, 1.3% and 4.5% on both components (masses to 1%, 3%, 10% at q near 1), using the two propagation formulas for other mass ratios. Label 3% as the threshold of the standard compilations, 10% as the level of existing Gaia-based work, 1% as the level needed for model tests.

## 8. What could not be verified

1. Eggleton (1983): the formula was checked only through its reproduction in Rowan et al. (2022); the 1% accuracy is recalled. Hut (1981, Eq. 42): expression recalled. Mazeh (2008): envelope taken from two papers that quote it.
2. No limb-darkening table for G_RVS was found; the text of Claret (2019) and the tables of Claret & Southworth (2022) were not read. The RVS row of Table 5.4 is an interpolation.
3. Mowlavi et al. (2023) publish no completeness against period or depth. The recovery tables in Sect. 2.3 are mine and are conditional on ASAS-SN detection; the 0.04 mag cut-on and the three-point rule as a detection model are inferred from catalogue distributions. The DR4 transit number is a scaling by mission months.
4. The Sample 2G-A selection in ADQL reproduces the published count to 2.3% only.
5. Tokovinin et al. (2006) give the tertiary frequency in the text for the first and last of four period groups; f3(P) is my fit. No published distribution of fitted third light in detached systems was found; Table 4 is derived and may be high by up to 1.4. The RVS window size for resolving tertiaries was not checked.
6. Synchronisation of hot stars rests on Torres et al. (2010) alone; Lurie et al. (2017) cover F to M primaries. Asteroseismic rotation studies of eclipsing binaries were not consulted. The 90% and 50% shares in the rotation prescription are rounded readings of Lurie et al.
7. The f_ecc(P) form and the twin-fraction comparison are constructions on verified numbers, not published results. The Eker et al. compilation is not a magnitude-limited sample.
8. Uniform-age isochrone quantiles of radius were not computed (no model tables were downloaded); the fallback radius offsets are empirical.
9. Spot coverage in close binaries extrapolates Saar & Donahue beyond their fitted range; the unit of their photometric amplitude, the far-red reduction factor and the activity scaling beyond saturation are assumptions. The RAVE resolving power (about 7,500) is recalled.
10. TESS depth thresholds are qualitative statements. Gaia yield predictions of Eyer et al. (2013) and Zwitter (2002) are quoted through Kochoska et al. (2017). Abstract-only items are tagged [A].
11. The Prsa et al. (2022) period statements were read through an automated summary of the ar5iv page; the completeness statement was read in the PDF text.

## 9. Ingredients

| Ingredient | Prescription | Source | Repo now (`draw_population`) |
|---|---|---|---|
| Primary mass | IMF times 10^(-0.6 M_G) with evolved luminosities | Sects. 2.1, 3.5 | Salpeter slope -2.3 times 10^(-0.6 M_G) on the mean dwarf sequence; pair kept by combined-light volume |
| Period | MDS17 companion frequency, then eclipse weight | repo note F.2; Sect. 2 | MDS17 over 0.5-1000 d; no eclipse weight |
| Eclipse selection | isotropic orientation, keep eclipsing (weight p_ecl) | Kirk 2016; Wells & Prsa 2021; Sullivan 2015 | `kind="eclipsing"` draws cos i inside the eclipsing range for every system |
| Detectability | depth 0.04 mag (Gaia) or 0.1 mag (ground); three in-eclipse points; or 0.1 mag for 5% of the orbit | Mowlavi 2023; Soderhjelm & Dischler 2005; Sect. 2.3 | none |
| Mass ratio | MDS17, checked against 30% above 0.95 | Sect. 3.4 | MDS17 with twin excess, q above 0.4 |
| Eccentricity | circular below P_env(Teff); circular share above; MDS17 for the rest | IJspeert 2024; Zanazzi 2022; Bashi 2023 | circular at 2 d and below for all Teff; MDS17 above |
| Radii, Teff | isochrones with uniform age, or empirical offset (median 0.13 dex above envelope at 1-2.5 Msun) | Eker 2018; Rowan 2022; Sect. 3.5 | mean dwarf relation, no scatter, no evolution |
| Roche limit | R below 0.75 R_L (Eggleton) at periastron | Eggleton 1983; MDS17; Wells & Prsa 2021 | R1 + R2 below 0.6 a, no periastron factor |
| Rotation | pseudo-synchronous below 10 d (cool) or for R/a above 0.1 (hot); aligned | Lurie 2017; Torres 2010; Marcussen & Albrecht 2022 | synchronous below 15 d, not pseudo-synchronous; field otherwise |
| Third light | f3(P), q3 uniform on [0.1, 1.2] | Tokovinin 2006; Laos 2020 | none |
| In-eclipse transits | flag; share rho/2 | Winn 2010; Sybilski 2013 | not flagged |
| Limb darkening | linear, 0.55 to 0.29 over 4000-10,000 K | Claret & Bloemen 2011; Claret 2019 | not applicable to the population module |
| Spots, activity | velocity term; Ca II core filling for Ro below 0.13 | Saar & Donahue 1997; Zerjal 2013; Wright 2011 | none |
| Apparent magnitude | from luminosity and a distance draw, limited at G_RVS | Sect. 1.1 counts | normal (9.8, 1.3) in G, independent of the stars |

## 10. Validation targets

1. Class context, V < 13: EA : EB : EW = 44 : 24 : 32; for P of 1 d and above 75 : 21 : 4 (ASAS-SN).
2. Period of detected detached systems, ground-based or Gaia discovery, G or V < 13: 16%, 50%, 84% quantiles 1.1-1.25, 2.4-2.6, 5.0-6.1 d; 8 to 14% below 1 d; 92 to 97% below 10 d. Space photometry: 1.9, 4.6, 12.7 d (TESS 2-min) and 2.8, 9.9, 56 d (Kepler).
3. Sum of fractional radii, ASAS-SN detached at G < 13: 12%, 34%, 54%, 78%, 95% below 0.2, 0.3, 0.4, 0.5, 0.6. A purely geometric sample with flat log P is uniform in this quantity; the observed one lies between uniform and linear.
4. Evolutionary state of primaries at G < 13: 84% main sequence, 14% subgiant, 2.4% giant.
5. Eccentricity: 83% below 0.05 overall; share above 0.1 of 4%, 6%, 10%, 17%, 33% in the bins 1-1.8, 1.8-3.2, 3.2-5.6, 5.6-10, 10-18 d; no eccentric orbit below 1.5 d; hot pairs eccentric in 16% at 1.5-4 d against 1 to 2% for cool pairs.
6. Mass ratio of double-lined detached systems: 30% above 0.95, 49% above 0.9, 83% above 0.7.
7. Radius at fixed mass, 1-2.5 Msun: 0.06, 0.13, 0.27 dex above the lower envelope at 16%, 50%, 84%.
8. Synchronisation: 79% of systems below 10 d; 94% within 0.92 < P_orb/P_rot < 1.2 below 2 d.
9. Tertiaries: 63 +/- 5% for inner periods of 1-30 d; 96% below 2.9 d; 34% above 12 d; at least 10% of multi-lined systems triple-lined.
10. Gaia DR3 recovery of bright detached systems: 64% overall; tables in Sect. 2.3 against period and amplitude.
11. Sky counts: 14,260 (G < 12) and 34,148 (G < 13) Gaia DR3 candidates, of which about 8% in the well-detached class; 3,957 (V < 12) and 11,042 (V < 13) ASAS-SN EA; 1.85 eclipsing binaries per square degree at Kp < 12 and 0.5 < P < 50 d in the Kepler field; 1.3% of Kepler targets and 3.5% of TESS OBA dwarfs with light curves (3,155 of 91,193) are eclipsing.
12. In-eclipse transits: a share of (R1 + R2)/(2a), about 0.19 at the median of target 3.

## References

Bibcodes checked in SIMBAD unless marked (n).
Samples: Paczynski et al. 2006MNRAS.368.1311P; Jayasinghe et al. 2019MNRAS.486.1907J; Rowan et al. 2022MNRAS.517.2190R, 2023MNRAS.520.2386R (2023b), 2023MNRAS.523.2641R (2023c); Prsa et al. 2011AJ....141...83P; Slawson et al. 2011AJ....142..160S; Kirk et al. 2016AJ....151...68K; Prsa et al. 2022ApJS..258...16P; IJspeert et al. 2021A&A...652A.120I; Soszynski et al. 2016AcA....66..405S; Wyrzykowski et al. 2003AcA....53....1W; Mowlavi et al. 2023A&A...674A..16M, 2017A&A...606A..92M.
Yields and selection: Soderhjelm & Dischler 2005A&A...442.1003S; Prsa, Pepper & Stassun 2011AJ....142...52P; Wells et al. 2017PASP..129f5003W; Wells & Prsa 2021ApJS..253...32W; Sullivan et al. 2015ApJ...809...77S; Kochoska et al. 2017A&A...602A.110K; Zwitter et al. 2003A&A...404..333Z; Wilkinson et al. 2005MNRAS.359.1306W; Duchene & Kraus 2013ARA&A..51..269D; Arenou 2011AIPC.1346..107A (n).
Tides: Lurie et al. 2017AJ....154..250L; Meibom et al. 2006ApJ...653..621M; Meibom & Mathieu 2005ApJ...620..970M; Hut 1981A&A....99..126H; Marcussen & Albrecht 2022ApJ...933..227M; Van Eylen et al. 2016ApJ...824...15V; Justesen & Albrecht 2021ApJ...912..123J; Zanazzi 2022ApJ...929L..27Z; IJspeert et al. 2024A&A...691A.242I; Bashi et al. 2023MNRAS.522.1184B; Mazeh 2008EAS....29....1M (n); Schussler et al. 2026, arXiv:2609.21087 (n).
Populations and stars: Moe & Di Stefano 2017ApJS..230...15M; Raghavan et al. 2010ApJS..190....1R; Mazeh et al. 2003ApJ...599.1344M; Lucy 2006A&A...457..629L; Simon & Obbie 2009AJ....137.3442S; Pinsonneault & Stanek 2006ApJ...639L..67P; El-Badry et al. 2019MNRAS.489.5822E; Kounkel et al. 2021AJ....162..184K; Torres et al. 2010A&ARv..18...67T; Eker et al. 2018MNRAS.479.5491E; Southworth 2015ASPC..496..164S; Andersen 1991A&ARv...3...91A (n); Eggleton 1983ApJ...268..368E.
Triples: Tokovinin et al. 2006A&A...450..681T; Tokovinin 2008MNRAS.389..925T, 2014AJ....147...87T; Laos et al. 2020ApJ...902..107L; Pribulla & Rucinski 2006AJ....131.2986P; D'Angelo et al. 2006AJ....132..650D; Hwang et al. 2020MNRAS.497.2250H; Borkovits et al. 2016MNRAS.455.4136B; Helminiak et al. 2016MNRAS.461.2896H, 2017MNRAS.468.1726H.
Eclipses, limb darkening, activity: Winn 2010exop.book...55W (n); Gaudi & Winn 2007ApJ...655..550G; Ohta et al. 2005ApJ...622.1118O; Hirano et al. 2010ApJ...709..458H; Albrecht et al. 2007A&A...474..565A, 2009Natur.461..373A; Sybilski et al. 2013MNRAS.431.2024S; Claret & Bloemen 2011A&A...529A..75C; Claret 2019RNAAS...3...17C; Claret & Southworth 2022A&A...664A.128C; Saar & Donahue 1997ApJ...485..319S; Zerjal et al. 2013ApJ...776..127Z; Lanzafame et al. 2023A&A...674A..30L; Wright et al. 2011ApJ...743...48W.
