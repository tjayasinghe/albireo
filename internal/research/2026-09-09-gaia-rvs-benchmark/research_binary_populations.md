# Catalogues and empirical distributions for a synthetic SB2 benchmark

Research date: **2026-09-09**. Target: draw truths for simulated double-lined spectroscopic
binaries as Gaia RVS would observe them, and verify recovery.

## Verification legend

- **[V] verified** — read live from the catalogue's own service during this research
  (Gaia archive TAP, VizieR TAP `TAP_SCHEMA`, the Gaia DR3 data model pages, or the
  catalogue's own data file). Column names below marked [V] are what your code will
  actually receive.
- **[L] likely** — from the refereed paper, an abstract, or a secondary source, but not
  read from the live service.
- **[U] unverified** — could not be confirmed; listed again in the final section.

### Working access routes (and one that is broken)

`https://cdsarc.cds.unistra.fr/ftp/<cat>/ReadMe` — the usual plain-text VizieR ReadMe route —
is **behind an Anubis anti-bot wall** and returns an HTML "Access Denied" page to
non-browser clients. `vizier.cds.unistra.fr/viz-bin/ReadMe/...` returns HTTP 500, and
`vizier.cfa.harvard.edu` 302-redirects back to the blocked host. Three routes do work:

| Route | URL | Use |
|---|---|---|
| Gaia TAP (sync GET) | `https://gea.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY=...` | Gaia DR3, anonymous, no auth |
| VizieR TAP | `https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?...` | `TAP_SCHEMA.tables`, `TAP_SCHEMA.columns`, real `COUNT(*)` |
| VizieR TSV export | `https://cdsarc.cds.unistra.fr/viz-bin/asu-tsv?-source=<TABLE>&-out.all&-out.max=unlimited` | bulk download |

Two URL-encoding traps that cost real time: in a query string `+` decodes to a space, so
catalogue names containing `+` need `%2B` (`J/A%2BARv/...`); and single quotes need `%27`.
Separately, VizieR stores `table_name` in `TAP_SCHEMA` **with literal double-quote characters
embedded in the value**, so exact matches must be written
`WHERE table_name = '"J/ApJS/258/16/tess-ebs"'` and prefix matches `LIKE '"J/ApJS/258/16%'`.
Substring matches (`LIKE '%258/16%'`) sidestep it.

---

# A. Gaia DR3 spectroscopic-binary orbits — `gaiadr3.nss_two_body_orbit`

Reference: **Gaia Collaboration, Arenou et al. 2023, A&A 674, A34**, bibcode
`2023A&A...674A..34G` [L] (the ESA DR3 papers page still lists the arXiv bibcode
`2022arXiv220605595G` [V]). arXiv:2206.05595.

## A.1 Solution-type counts — live, this date [V]

Queried directly against the archive rather than taken from the paper:

```
nss_solution_type,n
SB1,181327
Orbital,134598
EclipsingBinary,86918
AstroSpectroSB1,33467
SB2,4630
SB2C,746
OrbitalAlternative,619
OrbitalTargetedSearch,345
SB1C,202
OrbitalTargetedSearchValidated,188
EclipsingSpectro,155
OrbitalAlternativeValidated,10
```

**Total rows in `nss_two_body_orbit` = 443,205** [V] (sum of the above). Note the archive
splits `OrbitalAlternative`/`OrbitalAlternativeValidated` and
`OrbitalTargetedSearch`/`OrbitalTargetedSearchValidated` into separate literal values, where
the paper's Table 1 writes them with a bracketed suffix. Total SB2 + SB2C = **5,376** [V].

Other NSS tables (not `two_body_orbit`), from Arenou Table 1 [L]: `Acceleration7` 246,947,
`Acceleration9` 91,268, `FirstDegreeTrendSB1` 24,083, `SecondDegreeTrendSB1` 32,725,
`VIMF` 870. Paper totals: 839,098 solutions over 813,687 unique sources [L].

## A.2 The column-population matrix — the single most important table here [V]

A live `COUNT(<col>)` per solution type. **A column existing in the schema does not mean it is
populated for your solution type.** Zeros below are true NULL columns.

| nss_solution_type | n_tot | period | ecc | K1 | K2 | mass_ratio | gamma | incl | g_lum_ratio | T_ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| SB1 | 181327 | 181327 | 181327 | 181327 | 0 | 0 | 181327 | 0 | 0 | 0 |
| Orbital | 134598 | 134598 | 134598 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| EclipsingBinary | 86918 | 86918 | 86918 | 0 | 0 | 0 | 0 | 86918 | 86918 | 86918 |
| AstroSpectroSB1 | 33467 | 33467 | 33467 | 0 | 0 | 0 | 33467 | 0 | 0 | 0 |
| **SB2** | **4630** | 4630 | 4630 | 4630 | 4630 | **0** | 4630 | **0** | **0** | **0** |
| **SB2C** | **746** | 746 | **0** | 746 | 746 | **0** | 746 | **0** | **0** | **0** |
| SB1C | 202 | 202 | **0** | 202 | 0 | 0 | 202 | 0 | 0 | 0 |
| **EclipsingSpectro** | **155** | 155 | 155 | 155 | **0** | **155** | 155 | **155** | **155** | **155** |
| OrbitalAlternative | 619 | 619 | 619 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| OrbitalTargetedSearch | 345 | 345 | 345 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| OrbitalTargetedSearchValidated | 188 | 188 | 188 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| OrbitalAlternativeValidated | 10 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Four consequences, all verified, all easy to get wrong:

1. **Circular solutions store `eccentricity` as NULL, not 0.** `SB1C` and `SB2C` have
   `n_ecc = 0`. A naive `WHERE eccentricity IS NOT NULL` silently drops all 948 circular
   spectroscopic orbits; a naive `AVG(eccentricity)` is fine but `COALESCE(eccentricity, 0)`
   is what you actually want.
2. **SB2 has no `mass_ratio`.** It has K1 *and* K2, so `q = K1/K2` is derived, not stored.
3. **EclipsingSpectro is the mirror image**: it has `mass_ratio` and K1 but **no K2**
   (`K2 = K1/q`).
4. **`g_luminosity_ratio` is never populated for SB2.** It exists only on the two eclipsing
   types. There is no published Gaia light ratio for a Gaia SB2. See §G.2 for the one
   partial workaround.

## A.3 Full column list of `nss_two_body_orbit` [V]

From the DR3 data model page
(`.../chap_datamodel/sec_dm_non--single_stars_tables/ssec_dm_nss_two_body_orbit.html`).
77 columns. The ones you asked about are in bold.

| # | Column | Unit | Type | Description |
|---|---|---|---|---|
| 1 | `solution_id` | — | long | Solution identifier |
| 2 | `source_id` | — | long | Source identifier |
| 3 | **`nss_solution_type`** | — | string | NSS model adopted |
| 4–13 | `ra`,`ra_error`,`dec`,`dec_error`,`parallax`,`parallax_error`,`pmra`,`pmra_error`,`pmdec`,`pmdec_error` | deg / mas / mas yr⁻¹ | double/float | Barycentric astrometry at reference epoch |
| 14–25 | `a_thiele_innes`…`h_thiele_innes` (+`_error`) | mas / AU | double/float | Thiele-Innes elements A, B, F, G, C, H |
| 26 | **`period`** | day | double | Orbital period |
| 27 | `period_error` | day | float | |
| 28 | **`t_periastron`** | day | double | Periastron epoch |
| 29 | `t_periastron_error` | day | float | |
| 30 | **`eccentricity`** | — | double | Eccentricity of the orbit |
| 31 | `eccentricity_error` | — | float | |
| 32 | **`center_of_mass_velocity`** | km s⁻¹ | double | Radial velocity of centre of mass |
| 33 | `center_of_mass_velocity_error` | km s⁻¹ | float | |
| 34 | **`semi_amplitude_primary`** | km s⁻¹ | double | Semi-amplitude of primary RV curve (K1) |
| 35 | `semi_amplitude_primary_error` | km s⁻¹ | float | |
| 36 | **`semi_amplitude_secondary`** | km s⁻¹ | double | Semi-amplitude of secondary RV curve (K2) |
| 37 | `semi_amplitude_secondary_error` | km s⁻¹ | float | |
| 38 | **`mass_ratio`** | — | double | Mass ratio (SB2/EclipsingSpectro only) |
| 39 | `mass_ratio_error` | — | float | |
| 40–43 | `fill_factor_primary`,`fill_factor_secondary` (+`_error`) | — | double/float | Roche fill factors |
| 44 | **`inclination`** | deg | double | Orbital inclination |
| 45 | `inclination_error` | deg | float | |
| 46 | **`arg_periastron`** | deg | double | Argument of periastron |
| 47 | `arg_periastron_error` | deg | float | |
| 48 | **`temperature_ratio`** | — | double | Ratio of effective temperatures |
| 49 | `temperature_ratio_error` | — | double | |
| 50 | `temperature_ratio_definition` | — | byte | Code defining the T-ratio fitting scenario |
| 51–58 | `astrometric_n_obs_al`,`astrometric_n_good_obs_al`,`rv_n_obs_primary`,`rv_n_good_obs_primary`,`rv_n_obs_secondary`,`rv_n_good_obs_secondary`,`phot_g_n_obs`,`phot_g_n_good_obs` | — | int | Observation counts |
| 59 | `bit_index` | — | long | Boolean mask for the correlation matrix |
| 60 | `corr_vec` | — | float[231] | Upper triangle of the correlation matrix |
| 61–65 | `obj_func`,`goodness_of_fit`,`efficiency`,`significance`,`flags` | — | float/long | Fit quality |
| 66 | `conf_spectro_period` | — | float | Probability the period is not noise |
| 67–69 | `r_pole_sum`,`r_l1_point_sum`,`r_spher_sum` | — | double | Sums of radii in units of the semi-major axis |
| 70–71 | `ecl_time_primary`,`ecl_time_secondary` | JD (day) | double | Eclipse mid-points |
| 72–73 | `ecl_dur_primary`,`ecl_dur_secondary` | day | double | Eclipse durations |
| 74 | **`g_luminosity_ratio`** | — | double | **"Ratio of the G-band luminosity of the secondary over the primary"** |
| 75 | `input_period_error` | day | float | Period error from the variability analysis |
| 76 | `g_rank` | — | double | Rank of the G-band solution |
| 77 | `astrometric_jitter` | mas | double | Uncorrelated astrometric jitter |

**`g_luminosity_ratio` lives in `nss_two_body_orbit` itself**, not in a separate table. [V]

`r_pole_sum`, `r_l1_point_sum`, `r_spher_sum` are the fractional-radius sums (R₁+R₂)/a under
three radius definitions — directly usable as the eclipse-geometry truth for the 86,918
`EclipsingBinary` rows.

## A.4 Parameter ranges of the spectroscopic types [V]

Live aggregates:

| type | P min | P max | P mean | K1 min | K1 max | K1 mean | K2 max | K2 mean | e mean | e max |
|---|---|---|---|---|---|---|---|---|---|---|
| SB2 | 0.2506 | 1022.70 | 20.121 | 4.834 | 352.15 | 64.517 | 364.87 | 64.308 | 0.2323 | 0.9523 |
| SB2C | 0.2518 | 252.99 | 7.528 | 12.734 | 239.97 | 77.123 | 246.65 | 77.926 | (NULL) | (NULL) |
| EclipsingSpectro | 0.3818 | 28.267 | 1.196 | 8.939 | 172.09 | 64.199 | (NULL) | (NULL) | 0.00163 | 0.1066 |

Period distribution of the 5,376 SB2+SB2C [V]: **P < 1 d: 503 (9.4%); P < 3 d: 1,558 (29.0%);
P < 10 d: 3,612 (67.2%)**. Strongly short-period-weighted; the median sits near 5–6 d.

### Epoch counts — how many RVS visits a real Gaia SB2 actually got [V]

`gaia_source.rv_nb_transits` is NULL for every SB2 (§A.6), but
`nss_two_body_orbit.rv_n_good_obs_primary` **is fully populated** and is the correct substitute:

| type | n | `rv_n_good_obs_primary` min | max | mean | `rv_n_good_obs_secondary` mean |
|---|---|---|---|---|---|
| SB1 | 181327 | 10 | 225 | 23.93 | 0.0 |
| SB2 | 4630 | **10** | **79** | **21.52** | **21.52** |
| SB2C | 746 | 10 | 53 | 18.49 | 18.49 |

Three things fall out. The **minimum of exactly 10** empirically confirms the documented
"≥ 10 transits" NSS rule (§E.5). For SB2 the secondary's count *equals* the primary's, i.e.
both components were measured at every good transit. And the realistic epoch budget for a
Gaia RVS SB2 is **~10 to 80 epochs, typically ~20** — which is the number the benchmark's
simulated observing campaign should draw from. `rv_n_obs_primary` (all transits, not just
good ones) is available alongside it.

## A.5 `gaiadr3.binary_masses` [V]

Documented at `.../chap_datamodel/sec_dm_performance_verification/ssec_dm_binary_masses.html`
(**not** under the non-single-stars or astrophysical-parameters chapters — both of those
paths 404). Reference: **Gaia Collaboration, Creevey et al. / Babusiaux et al., "Binary masses
and luminosities with Gaia DR3", A&A 678, A19 (2023)**, bibcode `2023A&A...678A..19C` [L].

| Column | Unit | Type | Description |
|---|---|---|---|
| `source_id` | — | long | Unique identifier from `gaia_source` |
| `m1` | solMass | float | Mass of the primary component |
| `m1_lower` | solMass | float | Lower confidence level (16%) of the primary mass |
| `m1_upper` | solMass | float | Upper confidence level (84%) of the primary mass |
| `m2` | solMass | float | Mass of the secondary component |
| `m2_lower` | solMass | float | Lower confidence level (16%) |
| `m2_upper` | solMass | float | Upper confidence level (84%) |
| `fluxratio` | — | float | **"Ratio of the G-band flux of the secondary over the primary F2/F1"** |
| `fluxratio_lower` | — | float | Lower value of the flux ratio (method-dependent meaning) |
| `fluxratio_upper` | — | float | Upper value of the flux ratio |
| `combination_method` | — | string | Which combination of NSS solutions produced the values |
| `m1_ref` | — | string | Reference for the primary mass (`none`, `IsocLum`, or WD fixed mass) |
| `flag` | — | string | `AMRFClassIII` marks compact-object companions |

Live breakdown [V] — **total 195,315 rows**:

| combination_method | n | with `m2` | with `fluxratio` | with `fluxratio_lower` |
|---|---|---|---|---|
| Orbital+M1 | 111792 | 5454 | 0 | — |
| SB1+M1 | 60271 | 0 | 0 | — |
| AstroSpectroSB1+M1 | 17578 | 17578 | 17578 | 17578 |
| **SB2+M1** | **3856** | **3856** | **0** | **3856** |
| Orbital+SB1+M1 | 1513 | 1513 | 1513 | — |
| Eclipsing+SB1+M1 | 155 | 155 | 155 | — |
| EclipsingSpectro+M1 | 71 | 71 | 71 | — |
| Eclipsing+SB2 | 53 | 53 | 0 | 0 |
| Orbital+SB2 | 23 | 23 | 23 | 23 |
| EclipsingSpectro(SB2) | 3 | 3 | 3 | 0 |

**This is the route to individual masses for Gaia SB2s**: 3,856 of the 5,376 SB2/SB2C sources
get both `m1` and `m2`. But `fluxratio` is NULL for them — only the bracketing
`fluxratio_lower`/`fluxratio_upper` are set, which per the documentation is "the minimum
fluxratio compatible with a main sequence primary", i.e. a bound, not an estimate. [V]

## A.6 Joining to `gaia_source`, and a hard trap [V]

All requested `gaia_source` columns exist with these exact names, units and types [V]:

| Column | Unit | Type |
|---|---|---|
| `phot_g_mean_mag`, `phot_bp_mean_mag`, `phot_rp_mean_mag` | mag | float |
| `grvs_mag`, `grvs_mag_error` | mag | float |
| `rv_nb_transits`, `rv_nb_deblended_transits` | — | short |
| `radial_velocity`, `radial_velocity_error` | km s⁻¹ | float |
| `rv_template_teff` | K | float |
| `teff_gspphot` | K | float |
| `logg_gspphot` | log cgs | float |
| `mh_gspphot` | dex | float |
| `ecl_lat`, `ecl_lon`, `l`, `b` | deg | double |
| `non_single_star` | — | short |
| `ruwe` | — | float |
| `vbroad` | km s⁻¹ | float |

### The trap: Gaia SB2s have **no** `grvs_mag`, no `radial_velocity`, no `rv_nb_transits`

Live join, counting non-NULL values [V]:

| nss_solution_type | n | `grvs_mag` | `radial_velocity` | `rv_nb_transits` | `teff_gspphot` |
|---|---|---|---|---|---|
| SB1 | 181327 | 180182 | 180471 | 180471 | 106162 |
| **SB2** | **4630** | **0** | **0** | **0** | 3663 |
| **SB2C** | **746** | **0** | **0** | **0** | 600 |
| EclipsingSpectro | 155 | 134 | 135 | 135 | 136 |
| EclipsingBinary | 86918 | 8809 | 9036 | 9036 | 74253 |

`vbroad` is likewise 0/5376 for SB2+SB2C [V]. So **you cannot select Gaia SB2s by G_RVS, and
you cannot get their RVS transit count, from `gaia_source`.** Use `phot_g_mean_mag`, which is
fully populated: range **3.5517 to 12.2082, mean 9.7678** over all 5,376 [V].

The documented reason [L], from Katz et al. 2023 (A&A 674, A5) and Sartoretti et al. 2023
(A&A 674, A6): roughly 40,000 sources with ≥10% of transits flagged as double-lined were
treated as SB2 candidates and **their combined radial velocities were discarded**, because a
combined RV is not a meaningful systemic velocity for an SB2; and "all filters applied to
nullify spurious radial velocity measurements were applied to all pipeline products,
**including `grvs_mag`**". The epoch RVs were passed to the NSS group, which is how the
orbits exist at all. Empirically confirmed above.

### Example ADQL (verified to execute)

```sql
SELECT nss.source_id, nss.nss_solution_type,
       nss.period, COALESCE(nss.eccentricity, 0) AS ecc,
       nss.arg_periastron, nss.t_periastron,
       nss.semi_amplitude_primary   AS k1,
       nss.semi_amplitude_secondary AS k2,
       nss.semi_amplitude_primary / nss.semi_amplitude_secondary AS q,
       nss.center_of_mass_velocity  AS gamma,
       bm.m1, bm.m2, bm.fluxratio_lower, bm.fluxratio_upper,
       s.phot_g_mean_mag, s.phot_rp_mean_mag, s.teff_gspphot, s.ecl_lat, s.ruwe
FROM gaiadr3.nss_two_body_orbit AS nss
JOIN gaiadr3.gaia_source        AS s  ON nss.source_id = s.source_id
LEFT JOIN gaiadr3.binary_masses AS bm ON nss.source_id = bm.source_id
WHERE nss.nss_solution_type IN ('SB2', 'SB2C')
```

Run it at `https://gea.esac.esa.int/tap-server/tap` (anonymous), or via
`astroquery.gaia.Gaia.launch_job_async(...)`. For a one-shot GET, remember `%27` for quotes.

**Licence** [V], from `https://www.cosmos.esa.int/web/gaia-users/license`: *"Gaia data are
distributed under the CC BY-NC 3.0 IGO license."* — **Attribution-NonCommercial**, not
ShareAlike. (The CC BY-SA 3.0 IGO licence that is often quoted applies to DPAC's *documentation*,
not to the data.) The **NC clause matters** if the benchmark or any derived dataset is ever
redistributed commercially. Cite Gaia Collaboration, Prusti et al. 2016
(`2016A&A...595A...1P`) plus Vallenari et al. 2023 for DR3 (`2023A&A...674A...1G`) plus
Arenou et al. 2023 for NSS (`2023A&A...674A..34G`), with the standard acknowledgement:

> "This work has made use of data from the European Space Agency (ESA) mission Gaia
> (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis
> Consortium (DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium)."

---

# B. Gaia DR3 eclipsing-binary candidates — `gaiadr3.vari_eclipsing_binary`

Reference: **Mowlavi et al. 2023, "Gaia DR3: The first catalogue of Gaia eclipsing binaries —
candidate identification", A&A 674, A16**, bibcode `2023A&A...674A..16M` [V, from the ESA DR3
papers page].

**Row count: 2,184,477 [V]** (live `COUNT(*)`), confirming the "~2.18 million" figure.

## B.1 Columns [V]

From `.../sec_dm_variability_tables/ssec_dm_vari_eclipsing_binary.html`.

| Column | Unit | Type | Description |
|---|---|---|---|
| `solution_id` | — | long | Solution identifier |
| `source_id` | — | long | Source identifier |
| `global_ranking` | — | float | Quality metric, 0 (worst) to 1 (best) |
| **`reference_time`** | BJD in TCB − 2455197.5 (day) | double | Time reference for the geometric-model phases |
| **`frequency`** | day⁻¹ | double | Frequency of the geometric model |
| `frequency_error` | day⁻¹ | float | |
| `geom_model_reference_level` (+`_error`) | mag | float | Out-of-eclipse magnitude level |
| `geom_model_gaussian1_phase` (+`_error`) | — | float | Gaussian 1 phase, relative to `reference_time` |
| `geom_model_gaussian1_sigma` (+`_error`) | phase | float | Gaussian 1 standard deviation |
| `geom_model_gaussian1_depth` (+`_error`) | mag | float | Gaussian 1 depth |
| `geom_model_gaussian2_phase` (+`_error`) | — | float | Gaussian 2 phase |
| `geom_model_gaussian2_sigma` (+`_error`) | phase | float | Gaussian 2 standard deviation |
| `geom_model_gaussian2_depth` (+`_error`) | mag | float | Gaussian 2 depth |
| `geom_model_cosine_half_period_amplitude` (+`_error`) | mag | float | Ellipsoidal (cosine half-period) amplitude |
| `geom_model_cosine_half_period_phase` (+`_error`) | — | float | Ellipsoidal reference phase |
| **`model_type`** | — | string | Type of geometrical model of the G-band light curve |
| `num_model_parameters` | — | byte | Number of free parameters fitted |
| `reduced_chi2` | — | float | χ²/ν over all observations |
| **`derived_primary_ecl_phase`** (+`_error`) | — | float | Phase of the deepest point |
| **`derived_primary_ecl_duration`** (+`_error`) | phase fraction | float | Primary eclipse duration |
| **`derived_primary_ecl_depth`** (+`_error`) | mag | float | Primary eclipse depth |
| **`derived_secondary_ecl_phase`** (+`_error`) | — | float | Phase of the second-deepest point |
| **`derived_secondary_ecl_duration`** (+`_error`) | phase fraction | float | Secondary eclipse duration |
| **`derived_secondary_ecl_depth`** (+`_error`) | mag | float | Secondary eclipse depth |

**There is no period column** — use `period = 1 / frequency`. **There is no light-ratio
column either**; the two eclipse depths in magnitudes are the closest proxy. For an actual
light ratio you must join to `nss_two_body_orbit` and take `g_luminosity_ratio` from the
`EclipsingBinary` rows (86,918 of them, §A.2).

`reference_time` is offset by **2455197.5**, a different zero point from every other
catalogue in this document.

## B.2 `model_type` values and live counts [V]

```
TWOGAUSSIANS,1587926
TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE1,389725
TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE2,85400
ONEGAUSSIAN_WITH_ELLIPSOIDAL,48215
ONEGAUSSIAN,36984
ELLIPSOIDAL,36227
```
(sums to 2,184,477 [V]). Only the four `TWOGAUSSIANS*`/`ONEGAUSSIAN*` types have a real
eclipse; `ELLIPSOIDAL` (36,227) has none.

## B.3 Population statistics [V]

Live aggregates over all 2,184,477 rows:

- `frequency`: min **0.0014730 d⁻¹** (P ≈ 679 d), max **4.9999 d⁻¹** (P ≈ 0.2000 d — the
  search was hard-capped at 5 d⁻¹), mean 1.9395 d⁻¹.
- `derived_primary_ecl_depth`: mean **0.4450 mag**; `derived_secondary_ecl_depth`: mean
  **0.2396 mag**, present for **2,063,051** sources (94.4%).
- `derived_primary_ecl_duration`: mean **0.2871** in phase fraction — very large, reflecting
  how many contact and near-contact systems are in the sample.

**The RVS-accessible subset is tiny**: only **34,148** of the 2.18 M have
`phot_g_mean_mag < 13` [V]. That is ~1.6%. Plan the benchmark's magnitude distribution
accordingly — the Gaia EB catalogue is overwhelmingly too faint for RVS.

```sql
SELECT v.source_id, 1.0/v.frequency AS period_d, v.reference_time,
       v.derived_primary_ecl_depth, v.derived_secondary_ecl_depth,
       v.derived_primary_ecl_duration, v.model_type, v.global_ranking,
       s.phot_g_mean_mag, s.teff_gspphot
FROM gaiadr3.vari_eclipsing_binary AS v
JOIN gaiadr3.gaia_source AS s ON v.source_id = s.source_id
WHERE s.phot_g_mean_mag < 13 AND v.global_ranking > 0.5
```

---

# C. Well-characterised detached eclipsing binaries (masses, radii, temperatures)

## C.1 DEBCat (Southworth) — `https://www.astro.keele.ac.uk/jkt/debcat/`

- Citation: **Southworth 2015, ASP Conf. Ser. 496, 164**, bibcode `2015ASPC..496..164S`,
  arXiv:1411.1219. The page says verbatim: "Please cite this paper if you use DEBCat in your
  work." [V]
- **A machine-readable file exists**: `https://www.astro.keele.ac.uk/jkt/debcat/debs.dat` [V].
  The page says: "See [here](debs.dat) for a machine-readable ascii table. A few missing
  quantities are indicated using -9.99 in this table." [V]
- **Row count: 370 systems as of March 2026** [L] (360 as of May 2025). An automated line
  count of `debs.dat` returned 846, which I judge to be a miscount by the fetching model
  rather than the true value — **treat the exact count as [L] and count it yourself on
  download.** One row per *system* (both components in the same row).
- HTML table headings [V]: `System | Period (days) | V B-V | Spectral type | Mass (Msun) |
  Radius (Rsun) | Surface gravity (cgs) | log Teff (K) | log (L/Lsun) | [M/H] (dex) |
  References and notes`.

**`debs.dat` header, verbatim [V]** — whitespace-separated, missing value `-9.9900`:

```
# System   SpT1  SpT2  Pday  Vmag  BmV  logM1 logM1e logM2 logM2e  logR1 logR1e logR2 logR2e
  logg1 logg1e logg2 logg2e  logT1 logT1e logT2 logT2e  logL1 logL1e logL2 logL2e  MoH MoHe
```

(28 fields on one line; wrapped here for readability.) Everything is in **log₁₀** —
`logM1` is log(M/M☉), `logR1` is log(R/R☉), `logT1` is log(Teff/K), `logL1` is log(L/L☉).
`Pday` and `Vmag` are linear. A real row [V]:

```
CM_Dra  M4.5_V  M4.5_V  1.268 12.90 1.60 -0.6478 0.0006 -0.6776 0.0006 -0.6003 0.0003
        -0.6243 0.0004 4.9908 0.0008 5.0091 0.0007 3.4960 0.0100 3.4940 0.0140
        -2.2580 0.0380 -2.3130 0.0560 -0.3000 0.1200
```

**This is the single best self-contained truth set in this document**: P, M1, M2, R1, R2,
Teff1, Teff2, logg, [M/H] and V for every row, with per-quantity uncertainties. Its two gaps
are **no eccentricity** and **no G magnitude** — see §C.5 for the cross-match.

```bash
curl -o debs.dat https://www.astro.keele.ac.uk/jkt/debcat/debs.dat
```

## C.2 Torres, Andersen & Giménez 2010 — VizieR **`J/other/A+ARV/18.67`**

**The VizieR identifier is not `J/A+ARv/18/67`.** It is **`J/other/A+ARV/18.67`** [V] —
capital `ARV`, a dot before `67`, and an `other/` prefix. Queries against the commonly-cited
form return nothing.

Reference: A&ARv 18, 67 (2010), bibcode `2010A&ARv..18...67T` [L]. 95 detached systems,
190 stars, all with M and R known to ±3% or better [L].

Four tables [V]:

| Table | Description |
|---|---|
| `J/other/A+ARV/18.67/table1` | Observed parameters and derived quantities for 95 binary systems |
| `J/other/A+ARV/18.67/table2` | Other parameters and references for the 95 systems |
| `J/other/A+ARV/18.67/table3` | Systems with eccentric orbits, and systems with apsidal motion |
| `J/other/A+ARV/18.67/table5` | Parameters for 23 astrometric binary systems |

**`table1` columns [V]** — one row per *component*, 190 rows:

| Label | Unit | Explanation |
|---|---|---|
| `Seq` | — | [1/95] System sequential number — **the join key** |
| `Name` | — | Star name of the component |
| `Per` | d | Period of the component |
| `u_Per` | — | Uncertainty flag on Per |
| `x_Per` | — | **[y] `y` if the period is in YEARS, not days** |
| `Comp` | — | [AB] Component |
| `SpType` | — | MK spectral type |
| `Mass`, `e_Mass` | solMass | Mass of the component |
| `Rad`, `e_Rad` | solRad | Radius of the component |
| `Teff`, `e_Teff` | K | Effective temperature |
| `logg`, `e_logg` | log(cm.s⁻²) | Surface gravity |
| `logL`, `e_logL` | log(solLum) | Luminosity |
| `VMAG`, `e_VMAG` | mag | **Absolute** V magnitude of the component |
| `SimbadName` | — | Simbad name added by CDS |
| `_RA`, `_DE` | deg | J2000 position from SIMBAD (added by CDS) |

**`table2` columns [V]**: `Seq`, `Name`, `n_Name` ([*] pre-main-sequence), `Dist`/`e_Dist`
(pc), `E(B-V)`/`e_E(B-V)` (mag), **`vAsini`/`e_vAsini`, `vBsini`/`e_vBsini` (km/s)**,
`[Fe/H]`/`e_[Fe/H]`/`n_[Fe/H]` (log(Sun)), `logAge`/`u_logAge` (log(yr)),
`Ref1`…`Ref5` (bibcodes), `SimbadName`, `_RA`, `_DE`.

**`table3` columns [V]** — this is where **eccentricity** lives: `Seq`, `Name`, **`Ecc`**,
`e_Ecc`, `dw/dt`, `e_dw/dt` (deg/cycle), `logK2`, `e_logK2` (internal structure constant),
`Flag` ([0/3] peculiarities), `Ref`, `n_Ref`.

**`table5` columns [V]** — 23 astrometric systems, one row per *system*, up to three
components: `Name1`, `Vmag`, `Plx`/`e_Plx` (mas), `Dist`/`e_Dist`, `cp1`, `SpT1`,
`Mass1`/`e_Mass1`, `logL1`/`u_logL1`/`e_logL1`, `Teff1`/`e_Teff1`, `Rad1`/`e_Rad1`, `Ref`,
`Ref2`, `Name2`, `Per`/`x_Per`, `a`/`e_a` (mas), `[Fe/H]`/`e_[Fe/H]`, `cp2`, `SpT2`,
`Mass2`/`e_Mass2`, `logL2`/`u_logL2`/`e_logL2`, `Teff2`/`e_Teff2`, `Rad2`/`e_Rad2`,
`Name3`, `Per3`/`x_Per3`, `a3`/`e_a3`, `cp3`, `SpT3`, `Mass3`/`e_Mass3`,
`logL3`/`u_logL3`/`e_logL3`, `Teff3`/`e_Teff3`, `Rad3`/`e_Rad3`, `SimbadName`, `_RA`, `_DE`.

`table1 ⋈ table3` on `Seq` yields P, M1, M2, R1, R2, Teff1, Teff2 **and** e — a complete
truth set with `vsini` for both components available from `table2`. `table2` is the only
place in this whole document that gives you a **measured rotational velocity per component**,
which matters if you want realistic line broadening rather than an assumed synchronised value.

**Mind `x_Per = 'y'`**: some periods are in years. Filter or convert.

```sql
SELECT a."Seq", a."Name", a."Comp", a."Per", a."x_Per", a."Mass", a."Rad", a."Teff",
       a."logg", a."VMAG", c."Ecc", b."vAsini", b."vBsini", b."[Fe/H]", a."_RA", a."_DE"
FROM "J/other/A+ARV/18.67/table1" AS a
LEFT JOIN "J/other/A+ARV/18.67/table2" AS b ON a."Seq" = b."Seq"
LEFT JOIN "J/other/A+ARV/18.67/table3" AS c ON a."Seq" = c."Seq"
```
against `https://tapvizier.cds.unistra.fr/TAPVizieR/tap`.

## C.3 Eker et al. 2018 — VizieR `J/MNRAS/479/5491`

Reference: **Eker et al. 2018, "Interrelated main-sequence mass–luminosity, mass–radius and
mass–effective temperature relations", MNRAS 479, 5491**, bibcode `2018MNRAS.479.5491E` [L].

One table, `J/MNRAS/479/5491/table1`, **one row per component**, `Seq` range **[1/586]** [V].

| Label | Unit | Explanation |
|---|---|---|
| `Seq` | — | [1/586] Running sequence number |
| `Name` | — | Star name |
| `m_Name` | — | **[ps] Binary component (primary or secondary)** |
| `RAJ2000`, `DEJ2000` | — | J2000 position |
| `Sp` | — | Spectral type |
| `r_Sp` | — | Reference bibcode |
| `Mass`, `e_Mass` | solMass | Stellar mass (**`e_Mass` is a RELATIVE uncertainty**) |
| `Rad`, `e_Rad` | solRad | Stellar radius (**`e_Rad` is a RELATIVE uncertainty**) |
| `logg`, `e_logg` | log(cm.s⁻²) | Surface gravity |
| `r_Rad` | — | Reference bibcode for radius and logg |
| `Teff`, `e_Teff` | K | Published effective temperature |
| `r_Teff` | — | Reference bibcode |
| `Remark` | — | Remarks |
| `SimbadName` | — | Simbad name added by CDS |

**There is no `Per` and no eccentricity column** [V]. Eker 2018 is a stellar-parameter
compilation, not an orbit catalogue. To use it as a benchmark truth set you must supply
periods from elsewhere (DEBCat, Torres, or SB9 by name). Note also that `e_Mass` and `e_Rad`
are *relative* here, unlike Torres where they are absolute — a real unit trap if you write
one loader for both.

`m_Name` is `p`/`s`, so pair rows with `GROUP BY Name`.

Any newer Eker catalogue: I found repeated citation of Eker et al. 2018 as current and no
VizieR table for a 2024/2025 replacement. **[U] — see the unverified list.**

## C.4 ASAS-SN value-added eclipsing binaries — VizieR `J/MNRAS/517/2190`

Not in the original brief, but the closest thing to a large DEB catalogue that is already
keyed to Gaia, so worth flagging.

Reference: **Rowan, Jayasinghe, Stanek, Kochanek, Thompson, Shappee, Holoien, Prieto, Giles
2022, MNRAS 517, 2190** [V, from the VizieR table description]. One table,
`J/MNRAS/517/2190/table`, **35,576 rows** [V] (the `RotFlag` description states
"false for 34977 and true for 599", and `GaiaEDR3` "false for 8222 and true 27354",
both summing to 35,576).

Columns [V], abbreviated to the useful ones:

| Label | Unit | Explanation |
|---|---|---|
| `ID` | — | Internal ASAS-SN identifier |
| `IDJaya` | — | Identifier from the ASAS-SN variable-star catalogue (Cat. II/366) |
| **`(R1+R2)/a`** | — | Sum of fractional radii ρ₁+ρ₂ relative to the semi-major axis |
| **`Teff2/Teff1`** | — | Ratio of effective temperatures |
| **`Per`** | d | Orbital period |
| **`e`** | — | Orbital eccentricity |
| **`Omega`** | deg | Argument of periastron |
| **`i`** | deg | Orbital inclination |
| `t0` | d | Time of superior conjunction, **+2456000 days, UTC** |
| `Chi2` | — | Reduced χ² of the model fit |
| `gmag`, `Vmag` | mag | Median ASAS-SN g and V |
| `Ampg`, `AmpV` | mag | Light-curve amplitudes |
| `RotFlag` | — | Spot-modulation trend flag |
| `ASASSN` | — | ASAS-SN designation JHHMMSS.ss+DDMMSS.s |
| `RAJ2000`, `DEJ2000` | deg | J2000 position |
| `Gmag`, `BP-RP` | mag | Gaia G and BP−RP |
| `ABP`, `ARP`, `AG`, `AV` | mag | mwdust 3D extinction estimates |
| `rpgeo` | pc | Bailer-Jones photogeometric distance (Cat. I/352) |
| `RPlx` | — | Gaia EDR3 parallax over error |
| **`GaiaDR3`** | — | **"Gaia DR3 source ID troncated value in 1e+18 scale"** |
| `BP-RPcor`, `GMag` | mag | Extinction-corrected colour and absolute G |
| `GaiaEDR3` | — | Flag: passed `RPlx>10 and Plx>0` |
| `Heasarc`, `Obs`, `ExpTime`, `Sep`, `r_Heasarc`, `LX`, `f_LX` | — | X-ray cross-match |
| `Class` | — | Evolutionary state |

**Trap: `GaiaDR3` in the VizieR copy is a truncated/rescaled source_id ("in 1e+18 scale"),
not the integer `source_id`.** [V] Do not join it straight to `gaiadr3.gaia_source`. Join on
`RAJ2000`/`DEJ2000` instead, or recover the id from the ASAS-SN source.

This gives period, eccentricity, inclination, ω, (R₁+R₂)/a and T₂/T₁ for 35,576 systems with
Gaia photometry — **excellent for eclipse geometry and dilution, but it has no individual
masses or radii**, only the fractional-radius *sum* and the temperature *ratio*.

A follow-up, **Rowan et al. 2023, MNRAS 523, 2641**, "value-added catalogue of ASAS-SN
eclipsing binaries III: masses and radii of Gaia spectroscopic binaries", derives masses and
radii for >60 binaries by combining these light curves with Gaia DR3 SB2 orbits [L].
**It is not in VizieR** [V] — the TAP query for `J/MNRAS/523/2641` returns no rows.

## C.5 Cross-matching a DEB by name to Gaia — verified route

SIMBAD's TAP service carries Gaia DR3 designations in its `ident` table, so name → `source_id`
is one query, no positional matching required. **Verified live with a real system** [V]:

```sql
SELECT b.main_id, b.ra, b.dec, i2.id
FROM basic AS b
JOIN ident AS i1 ON i1.oidref = b.oid
JOIN ident AS i2 ON i2.oidref = b.oid
WHERE i1.id = 'CM Dra' AND i2.id LIKE 'Gaia DR3%'
```
at `https://simbad.cds.unistra.fr/simbad/sim-tap/sync`, returning:
```
main_id,ra,dec,id
"V* CM Dra",248.5847094419055,57.16232469963777,"Gaia DR3 1431176943768690816"
```

Strip the `Gaia DR3 ` prefix to get the integer id, then query `gaiadr3.gaia_source` for
`phot_g_mean_mag`, `phot_rp_mean_mag` and `grvs_mag`. The alternative,
`I/355/gaiadr3` on VizieR ("Gaia DR3 source catalog (1811709771 sources)" [V]), works for
positional cross-matches but SIMBAD is cleaner when you have names, which is what DEBCat,
Torres and Eker all give you.

---

# D. TESS and Kepler eclipsing-binary catalogues

## D.1 Prša et al. 2022, TESS EBs — VizieR `J/ApJS/258/16`

Bibcode `2022ApJS..258...16P` [V]. "TESS Eclipsing Binary stars. I. Short-cadence
observations of 4584 eclipsing binaries in sectors 1–26."

**One table**, `J/ApJS/258/16/tess-ebs`, **4,584 rows** [V], VizieR last updated
16-Dec-2022 [V]. The count matches the paper, so the VizieR copy is the published version,
not a live one. **Rows are ephemerides, not targets** — `m_TIC` (`signal_id`) enumerates
distinct ephemerides for one TIC, so `TIC` is not unique.

All 28 columns [V]:

| Label | Unit | Explanation |
|---|---|---|
| `TIC` | — | [91961/2046417955] TESS Input Catalog identifier |
| `m_TIC` | — | [1/2] Enumerates distinct ephemerides for a single TIC |
| `Date`, `UpDate` | s | Ingestion and last-modification timestamps (UT) |
| `RAJ2000`, `DEJ2000` | deg | J2000 position |
| `pmRA`, `pmDE` | mas/yr | Proper motion from TIC |
| **`Tmag`** | mag | [2.28/17.66] TESS magnitude from TIC |
| **`BJD0`**, `e_BJD0` | d | [1322/2142] BJD of the first primary eclipse, **BJD0 − 2457000** |
| **`Per`**, `e_Per` | d | [0.048/314.7] Orbital period |
| **`Morph`** | — | **[−1/1.08]** Morphology coefficient |
| `Wp-pf`, `Dp-pf`, `Phip-pf` | phase | Primary eclipse width / depth / phase, **polychain fit** |
| `Ws-pf`, `Ds-pf`, `Phis-pf` | phase | Secondary eclipse width / depth / phase, polychain fit |
| `Wp-2g`, `Dp-2g`, `Phip-2g` | phase | Primary eclipse width / depth / phase, **two-Gaussian fit** |
| `Ws-2g`, `Ds-2g`, `Phis-2g` | phase | Secondary eclipse width / depth / phase, two-Gaussian fit |
| `Sectors` | — | List of sectors (1–26); a **VARCHAR list**, not an integer |

**No Gaia `source_id`, no distance, no Teff** [V] — verified by exhaustive column listing.
A Gaia join needs your own positional or TIC→Gaia cross-match.

Eclipse depths and widths appear **twice**, once per fit method (`-pf` polychain, `-2g`
two-Gaussian). Hyphenated labels must be double-quoted in ADQL: `"Wp-pf"`, `"Phip-2g"`.

The live site `https://tessebs.villanova.edu/` is up and calls itself the live version, but
its default listing implies roughly 2,300 rows against VizieR's 4,584, so the default view is
filtered or grouped. **Treat the live-site count as [U].**

```bash
wget -O tess-ebs.tsv "https://cdsarc.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJS/258/16/tess-ebs&-out.all&-out.max=unlimited"
```
```python
from astroquery.vizier import Vizier
tess = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/ApJS/258/16")[0]
```

## D.2 Kirk et al. 2016, Kepler EBs — VizieR `J/AJ/151/68`

Bibcode `2016AJ....151...68K` [V]. "Kepler eclipsing binary stars. VII."

Ten tables [V]. The main one is `J/AJ/151/68/catalog`, **2,876 rows** live [V] — VizieR's
own prose says 2,878, a two-row discrepancy [V]. The others are small special-population
lists: `table1` heartbeat stars (173), `table2` tidally induced pulsations (24), `table3`
reflection effect (36), `table4` occultation pairs (32), `table5` circumbinary planets (8),
`table6` multiple ephemerides (14), `table7` extraneous events (9), `table8` eclipse depth
variations (43), `table9` no repeating events (32).

`catalog` columns [V]:

| Label | Unit | Explanation |
|---|---|---|
| `V1`, `V2`, `V3b`, `PapIV` | — | Flags for inclusion in earlier catalogue versions |
| **`KIC`** | — | [1026032/12785282] Kepler Input Catalog number |
| `HB` | — | 1 if a heartbeat star |
| `DV` | — | Eclipse depth variation indicator |
| **`Per`**, `e_Per` | d | [0.07/1087.3] Period |
| **`BJD0`**, `e_BJD0` | d | [5487.3/56903.9] Time of eclipse, **BJD − 2400000** |
| **`Morph`** | — | **[0/1]** Morphology value |
| `GLON`, `GLAT` | deg | Galactic coordinates |
| **`Kpmag`** | mag | [−1/19.742] Kepler magnitude |
| **`Teff`** | K | Kepler effective temperature |
| `SC` | — | Short-cadence data available? |
| `LC`, `TS`, `Simbad` | — | Link columns |
| `_RA`, `_DE` | deg | J2000 from SIMBAD (added by CDS) |

**The VizieR copy has no eclipse-depth, width or secondary-separation columns** [V]. There is
no `pdepth`, `sdepth`, `pwidth`, `swidth` or `sep`. This was confirmed two ways: the full
`TAP_SCHEMA.columns` listing, and the raw TSV header from `asu-tsv?-out.all`.

`http://keplerebs.villanova.edu/` **is up** [V], serving the Third Revision, stamped last
updated **Aug 8 2019**, reporting **2,920** results against VizieR's 2,876. Its search form
exposes exactly the columns VizieR lacks, grouped as **"Polyfit: pdepth, sdepth, pwidth,
swidth, separation"** plus three Teff scales (Kepler, Pinsonneault, Casagrande). It offers
comma-, tab- and space-separated export. **If you need depths, widths or the secondary
separation, go to Villanova, not VizieR.** The export hrefs could not be extracted (the page
is JS-driven) — **[U]**.

**Do not treat the two morphology parameters as the same scale**: Kirk's `Morph` is [0/1],
Prša's TESS `Morph` is [−1/1.08] [V].

```bash
wget -O kepler_eb.tsv "https://cdsarc.cds.unistra.fr/viz-bin/asu-tsv?-source=J/AJ/151/68/catalog&-out.all&-out.max=unlimited"
```

---

# E. Non-eclipsing spectroscopic binaries with orbits

## E.1 SB9 — VizieR `B/sb9`, and its successor SBX

Reference: **Pourbaix et al. 2004, A&A 424, 727**, bibcode `2004A&A...424..727P` [V].
The VizieR title carries the version string **"(Version 2013-10-27)"**; VizieR's own
`vizier_date_update` is 01-Jul-2024 [V].

Three TAP-served tables plus a `notes.txt` that is not TAP-served [V]:

| Table | rows | Description |
|---|---|---|
| `B/sb9/main` | **4,079** | Position, magnitudes and spectral types |
| `B/sb9/orbits` | **5,099** | Orbital parameters |
| `B/sb9/alias` | **20,806** | Other designations |

**`B/sb9/main` columns [V]**: `Seq` (System Number, SB8 number when Seq ≤ 1469 — **the join
key**), `N`, `RAJ2000`, `DEJ2000` (deg), `Comp` (component), `mag1`/`n_mag1` (mag + filter
[GJPBVRyrIK]), `mag2`/`n_mag2` (mag + filter [PBVRyrIK]), `Sp1`, `Sp2` (MK types),
`Name` (HIP when it exists).

**`B/sb9/orbits` columns [V]** — independently verified twice, by me and by a second pass:

| Label | Unit | Explanation |
|---|---|---|
| `Seq` | — | System Number, as in `main.dat` |
| `N` | — | (no description in the metadata) |
| `o` | — | Orbit number (a system may carry several orbits) |
| **`Per`** | d | [0.05, 116675] Period |
| `f_Per` | — | **[a] a = assumed (fixed)** |
| `e_Per` | d | Mean error on Per |
| **`T0`** | d | Periastron time, **full JD, not offset** |
| `n_T0` | — | [a] a = assumed (fixed) |
| `e_T0` | d | Mean error on T0 |
| `f_T0` | — | Flag on T0 |
| **`e`** | — | Orbital eccentricity |
| `f_e` | — | [>a] a = assumed (fixed) |
| `e_e` | — | Mean error on eccentricity |
| **`omega`** | deg | [−359, 360] Argument of periastron |
| `f_omega` | — | [a] a = assumed (fixed) |
| `e_omega` | deg | Mean error on omega |
| **`K1`** | km/s | Velocity amplitude of primary |
| `u_K1` | — | Uncertainty flag (`:`) on K1 |
| `f_K1` | — | [>a] a = assumed (fixed) |
| `e_K1` | km/s | Mean error on K1 |
| **`K2`** | km/s | Velocity amplitude of secondary |
| `u_K2`, `f_K2`, `e_K2` | — / km/s | As for K1 |
| **`V0`** | km/s | Systemic velocity ("Systemtic" — the typo is in the catalogue) |
| `u_V0`, `f_V0`, `e_V0` | — / km/s | As above |
| **`rms1`**, **`rms2`** | km/s | RMS residuals of RV primary / secondary |
| `o_K1`, `o_K2` | — | Number of observations of RV1 / RV2 |
| **`Grade`** | — | Grade of orbit (0 = poor, 5 = definitive) |
| `Ref` | — | Bibcode of the orbit publication |
| `Contr` | — | Contributor |

**The `f_*` flags are load-bearing**: `a` means the parameter was *assumed and held fixed*,
not fitted. Filter on them before using an orbit as truth.

Benchmark-relevant live counts [V]:
- Orbits with **both K1 and K2** non-null (true SB2 orbits): **1,674**
- Of those with **Grade ≥ 4**: **513**
- Orbits with a non-null Grade: 2,462, range **0.0 to 5.1** (the max exceeds the documented 0–5)

**Gaia cross-match**: there are no separate `HD`/`HIP` columns, but `alias.Name` is a single
string of the form `"<CATALOG> <designation>"`, and Gaia ids are already present. Sampled
values [V]: `"BD -06 6357"`, `"Flamsteed 33 Psc"`, `"GAIADR2 2444348733778920832"`,
`"GAIADR3 2444348733778920832"`, `"HD 28"`, `"HIP 443"`. **3,530 rows in `B/sb9/alias` match
`Name LIKE 'GAIADR3%'`** [V]. Parse with `SUBSTRING(Name FROM 9)`.

### SB9 is retired; use SBX

`https://sb9.astro.ulb.ac.be/` states: **"Since 2025-06-24, the SB9 catalogue is superseded by
SBX"** [V], pointing at `https://astro.ulb.ac.be/sbx`. **SBX live counts: 4,080 systems,
5,169 orbits** [V] — so the VizieR copy is stale by roughly 70 orbits.

Reference: **Merle, Jorissen, Alexandre et al. 2026, "The SB9 catalogue: status, comparison
with non-single stars from Gaia DR3, and evolution to SBX", MNRAS 547** [V]. The bibcode
rendered on the site as `2026MNRAS.547ag351M`, which is malformed — **bibcode [U], resolve via
ADS.** From that paper [V]: SB9 holds ~2,800 SB1 and ~1,200 SB2; 152 systems in triples,
71 in quadruples, 14 higher order; **3,976 of 4,003 systems have Gaia DR3 counterparts
(99.3%)**, **827 appear in both catalogues**, **655 reliable** (period and eccentricity
agreeing within 10%), and **461 SB2 systems** are in the Gaia NSS catalogue.

**SBX has a TAP service** [V] at `http://astro.ulb.ac.be/sbx/tap/` (tables at
`/sbx/tap/tables`), six tables in schema `public`. It is strictly better than `B/sb9` for
this purpose because it carries **the RV time series**, which VizieR does not:

- **`orbits`**: `sn`, `on`, `period`, `period_error`, `eccentricity`, `eccentricity_error`,
  `periastron_argument`(+`_error`), `periastron_time`(+`_error`), `periastron_time_flag`,
  `K1`, `K1_error`, `K2`, `K2_error`, `systemic_velocity`(+`_error`),
  `apparent_systemic_velocity_1`/`_2` (+errors), `rms_RV1`, `rms_RV2`, `num_RV1`, `num_RV2`,
  `grade`, `grade_legacy`, **`epoch_reference`** (says whether times are JD or MJD, per orbit),
  `bibcode`, `contributor`, `note`, `accessibility_legacy`
- **`systems`**: `sn`, `ra`(+err), `dec`(+err), `pmra`(+err), `pmdec`(+err), `pm_epoch`,
  `pm_source`, `parallax`(+err), `parallax_source`, `position_epoch`, `position_source`,
  `mag1`/`mag1_source`/`band1`, `mag2`/`mag2_source`/`band2`, `st1`/`st1_bibcode`,
  `st2`/`st2_bibcode`, `component`, `coordinates_1900_legacy`, `coordinates_2000_legacy`
- **`velocities`**: `sn`, `on`, `component`, `epoch`, `radial_velocity`,
  `radial_velocity_error`, `weight`, `velocity_source`, `comment_legacy`, `offset_legacy`
- **`alias`**: `sn`, `catalog`, `version`, `identifier` — properly normalised, unlike VizieR's
  single string
- **`configurations`**: `sn`, `family`, `parent`, `child1`, `child2`, `in_triple`,
  `in_quadruple`, `in_high_order`
- **`duplicates`**: `sn`, `sn_duplicate`

SBX's own example ADQL, verbatim from the site [V]:
```sql
SELECT DISTINCT 'SBX', o.sn AS sn, a.catalog, a.version, a.identifier
FROM orbits o LEFT JOIN alias a ON o.sn = a.sn
AND a.catalog = 'Gaia' AND a.version = 'DR3'
WHERE o.K1 IS NOT NULL AND o.K2 IS NOT NULL ORDER BY sn;
```

VizieR equivalent, if you prefer the frozen copy:
```python
import pyvo
tap = pyvo.dal.TAPService("https://tapvizier.cds.unistra.fr/TAPVizieR/tap")
sb2 = tap.search('''SELECT m."Seq", m."RAJ2000", m."DEJ2000", m."Name", m."Sp1", m."Sp2",
                           o."Per", o."T0", o."e", o."omega", o."K1", o."K2", o."V0",
                           o."rms1", o."rms2", o."Grade", o."Ref"
                    FROM "B/sb9/main" AS m JOIN "B/sb9/orbits" AS o ON m."Seq" = o."Seq"
                    WHERE o."K1" IS NOT NULL AND o."K2" IS NOT NULL
                      AND o."Grade" >= 4''').to_table()
```
Note `"e"` and `"o"` need quoting — both are near-reserved words; in SBX, `on` is a column
name and is reserved in some parsers.

## E.2 Kounkel et al. 2021, APOGEE SB2s — VizieR `J/AJ/162/184`

Bibcode `2021AJ....162..184K` [V]. "Double-lined spectroscopic binaries in the APOGEE DR16
and DR17 data." Four tables [V]:

| Table | rows | Description |
|---|---|---|
| `J/AJ/162/184/table1` | **8,105** | Properties of the identified SB2s and higher-order multiples |
| `J/AJ/162/184/table3` | 32,642 | Vetted radial velocities of the individual components |
| `J/AJ/162/184/tablea1` | 31,313 | Parameters of the CCF components extracted by the DR17 pipeline |
| `J/AJ/162/184/dr17samp` | 7,204 | Systems from Table A1 (added by CDS) |

**It has both, but orbits are a minority: only 320 of the 8,105 `table1` rows have a non-null
`Per`** [V]. The rest carry only Wilson-plot mass ratios, RV amplitudes and flags.

`table1` columns [V]: `ID` (APOGEE identifier), `RAJ2000`, `DEJ2000`, `SBn` ([2/4] number of
deconvolved components), `Nepoch` ([1/47]), `DR17` (link), `qW`/`e_qW` (mass ratio from a
Wilson plot), `RVelW`/`e_RVelW` (km/s), **`Per`**/`e_Per` (d, [0.8/348]), **`T0`**/`e_T0`
(d, time of periastron, **full JD**), **`e`**/`e_e`, **`omega`**/`e_omega` (**deg**),
`RVel`/`e_RVel` (km/s, barycentre RV), **`K1`**/`e_K1` (km/s, [7.1/173.1]), **`K2`**/`e_K2`
(km/s, [10.8/169]), `M1sin3i`/`e_M1sin3i`, `M2sin3i`/`e_M2sin3i` (solMass), `asini`/`e_asini`
(km), `MaxdRV` (km/s), `AmpRV1`, `AmpRV2` (km/s), `MaxT` (d), `fLOS`, `fVar`, `PerTESS` (d),
`SimbadName`.

The Wilson-plot uncertainties are unusable as published: `e_qW` ranges [0/65947.1] and
`e_RVelW` [−58300/2200], with negative errors [V]. Filter hard.

`table3` columns [V]: `ID`, `HJD` (d, **HJD − 2400000**, [55804/58933]), `RVel1`/`e_RVel1`,
`RVel2`/`e_RVel2`, `RVel3`/`e_RVel3`, `RVel4`/`e_RVel4` (km/s). `RVel3` has a corrupt range
[−3.3e7/6.8e9]; components 3 and 4 are unreliable.

`tablea1` columns [V]: `ID`, `HJD`, `RAJ2000`, `DEJ2000`, `Amp1..4`/`e_Amp1..4`,
`RVel1..4`/`e_RVel1..4` (km/s), `FWHM1..4`/`e_FWHM1..4`, `Flag1..4`, `N`, `DR16comp`.
**VizieR's declared column order puts `e_RVel4` before `RVel4`** — address columns by name,
never by position [V].

## E.3 GALAH — Traven et al. 2020, `J/A+A/638/A145`

Bibcode `2020A&A...638A.145T` [V]. One table, `J/A+A/638/A145/catalog`, **12,760 rows** [V].

**No orbits at all** [V] — verified by exhaustive column listing: no period, no K1/K2, no
eccentricity, no T0, no omega. This is a **single-epoch spectrum decomposition** keyed by
`spectID` (= `sobject_id`, unique per observation).

What it does give, and why it is still valuable here: every fitted quantity appears as a
five-column block with suffixes `-16`, `-50`, `-84`, `-mean`, `-mode` — `Teff1-*`, `Teff2-*`
(K), `logg1-*`, `logg2-*`, `[Fe/H]-*`, `RV1-*`, `RV2-*` (km/s), **`ratio1-*` … `ratio4-*`
(flux ratio in the four GALAH/HERMES bands)**, `vmic1-*`, `vmic2-*`, `vbroad1-*`, `vbroad2-*`
(km/s), `E(B-V)-*`, `R1-*`, `R2-*` (solRad), then `chi2`, `chi2sp`, `chi2ph`, `RUWE`.
Identifiers and photometry: `2MASS`, **`GaiaDR2`** (BIGINT), `Field`, `RAJ2000`, `DEJ2000`,
`Bmag`, `Vmag`, `GBPmag`, `Gmag`, `GRPmag`, `Jmag`, `Hmag`, `Kmag`, `W1mag`, `W2mag`, `plx`,
`snc1..snc4` (S/N per HERMES channel), `Flag`.

**`ratio1-50` … `ratio4-50` are exactly the wavelength-dependent light-ratio priors a
disentangling benchmark wants**, measured rather than assumed, for 12,760 real FGK SB2s.

Labels contain `[`, `]`, `/`, `(`, `)`, `-` — `"[Fe/H]-50"`, `"E(B-V)-mode"` need quoting.
The Gaia id is **DR2**, not DR3.

## E.4 LAMOST SB2s with orbits

**Guo et al. 2025, ApJS 278, 46 — VizieR `J/ApJS/278/46`** is the best of these. One table,
`J/ApJS/278/46/table1`, **665 rows** [V]; orbits fitted with a modified `thejoker` to 1,119
SB2 systems having ≥6 epochs [L]. Bibcode `2025ApJS..278...46G` [L].

**Full orbits and a Gaia DR3 `source_id`** [V]: `LAMOST` (designation), **`GaiaDR3`**
(BIGINT, a real source_id), `RAJ2000`, `DEJ2000`, `FitFlag` ([1/4]), **`Per`**/`e_Per` (d),
**`e`**/`e_e`, **`K1`**/`e_K1`, **`K2`**/`e_K2` (km/s), **`v0`**/`e_v0` (km/s),
**`omega`**/`e_omega` (**radians**), **`t0`**/`e_t0` (d, JD), `s` (km/s, RV jitter),
`q`/`e_q` (= m2/m1 = K1/K2), `N`, `MaxPhaseGap`, `NMAE`, `M`, `dmag`/`s_dmag` (mag).

**`omega` is in radians**, unlike SB9 and APOGEE. `dmag` is a magnitude difference — a light
ratio in disguise.

**Han, Li & Gao, ApJS 284, 5 — `J/ApJS/284/5`** [V]: `table3` 49,889 SB2 candidates from
LAMOST-MRS DR10; `table4` 33,560 triples; `table5` 294 SB1; `table6` 351 others;
**`table7` 29 orbital solutions for P ≥ 200 d**, with `Per`, `e`, `K1`, `K2`, `v0`,
`omega` (**rad**), **`M0`** (mean anomaly at the reference epoch, rad — this one has no `t0`),
`q`, `Delmag`, and no uncertainties. `table3`'s `HJD` is an **INTEGER**, so epoch resolution
there is one day — a hard limit for any timing use.

Candidate-only LAMOST tables, no orbits [V]: `J/MNRAS/517/356/tablec1` and
`J/MNRAS/517/367/tablec1` (2,460 each), `J/MNRAS/527/521/tableb1` (12,426),
`J/ApJS/266/18/table4|5|6` (2,139 / 36,470 / 728), `J/ApJS/276/11/table4` (2,318).
`J/ApJS/275/40` (Li et al. 2024, 44 LRS orbital solutions) **is not in VizieR** [V].

## E.5 Gaia DR3 SB2 sample statistics and the magnitude limit

- **The SB1 NSS chain used a threshold on internal G_RVS of 12 mag**, quoted verbatim [V]:
  "A threshold on G_RVS^int was selected at 12 mag. Fainter objects are considered to produce
  individual spectra for which the S/N ratio does not enable the derivation of a sound radial
  velocity." (Gaia Collaboration, *Spectroscopic binary-star orbital solutions — the SB1
  processing chain*, arXiv:2410.14372, A&A 2025.)
- **Minimum 10 RVS transits** [V]: "The stellar RVs are not processed by the NSS spectroscopic
  channel if the number of transits (data points) is less than 10."
- Period search extended to **2×ΔT** where ΔT is the 34-month DR3 RVS baseline [V].
- Input filter [V]: `rv_chisq_pvalue <= 0.01 AND rv_renormalised_gof > 4`; period
  significance `conf_spectro_period > 0.95`.
- **An SB2-specific magnitude limit is not stated in that paper** — the SB2 chain
  (Damerdji et al.) is a separate publication that was still "in preparation" when the SB1
  paper appeared. **[U]**. What is verifiable is the *empirical* limit: all 5,376 SB2/SB2C
  sources have **G between 3.55 and 12.21, mean 9.77** [V].
- DR3 RVs generally reach **G_RVS ≈ 14**, not 12 [L] — the 12 mag cut is specific to the NSS
  spectroscopic chain, not to RV publication. Do not conflate them.
- Observed SB2 distributions: see §A.4 (periods, K1, K2, e) and §A.6 (magnitudes).

---

# F. Empirical distributions for a parametric population generator

Method note: the numbers below were read from the papers' own PDFs, with ambiguous equations
**rendered as images and read visually** where the text layer mangles them. The Moe &
Di Stefano model was then **implemented independently and validated against the paper's own
Table 13 — all 20 entries reproduce**, which checks the transcription end to end.

## F.1 Raghavan et al. 2010 — ApJS 190, 1, bibcode `2010ApJS..190....1R`, arXiv:1007.0414

454 stars (F6–K3, within 25 pc), 259 confirmed companion pairs.

| Quantity | Value | Where | Status |
|---|---|---|---|
| **⟨log₁₀ P/d⟩** | **5.03** | §5.3.3, Fig. 13 caption | [V] |
| **σ(log₁₀ P/d)** | **2.28** | §5.3.3, Fig. 13 caption | [V] |
| Mean period | 293 yr | §5.3.3 | [V] |
| Median period | 252 yr | §5.3.3 | [V] |
| Circularisation period | ~12 d | §5.3.4 | [V] |

Your quoted 5.03 / 2.28 is exactly right. Verbatim: *"The period distribution follows a
roughly log-normal Gaussian profile with a mean of log P = 5.03 and σ_log P = 2.28, where P
is in days."* [V]

**Mass ratio** (§5.3.5, Fig. 16) [V]: *"a roughly flat distribution for mass ratios
0.2–0.95"*, with a deficiency below q ≈ 0.2 and a real **twin excess at q > 0.95**
(27 pairs, only 6 with P > ~200 yr). Short periods prefer high q: the fraction with P < 100 d
rises 0% (q<0.2) → 4% (q<0.45) → 8% (q = 0.45–0.9) → **16% (q > 0.9)**.

**Eccentricity** (§5.3.4/5.3.5) [V]: for P > 12 d, *"a roughly flat eccentricity distribution
out to e ≈ 0.6, independent of period"* (117 systems). This **explicitly contradicts
Duquennoy & Mayor 1991** (Gaussian below 1000 d, f(e) = 2e above). P < 12 d are circular,
with one exception (HD 45088 Aa-Ab, P = 7 d, e = 0.147).

**Multiplicity — Table 16** [V], percentages of N = 454: single **56 ± 2**, binary
**33 ± 2**, triple **8 ± 1**, quadruple+ **3 ± 1**. So the **multiplicity fraction is
44 ± 2%**, and 25% of non-single stars are higher-order multiples.

## F.2 Moe & Di Stefano 2017 — ApJS 230, 15, bibcode `2017ApJS..230...15M`, arXiv:1606.05347

### A correction to the brief

**There is no table of f_logP;q>0.3 on an (M1, logP) grid.** That quantity appears only as
**Figure 37 (top)** and as the analytic fit **Eqns. 20–23**. **Table 13** tabulates
**f_logP;q>0.1** — not q>0.3 — at logP = 1, 3, 5, 7 for five spectral bins. Also, §9.1
evaluates at M1 = 1, 3.5, 7, **12**, 28 M☉ — **12, not 12.5**. Both given below.

### F.2a Table 13 verbatim [V]

Bins: solar-type M1 = 0.8–1.2; A/late-B 2–5; mid-B 5–9; early-B 9–16; O-type >16 M☉.

| Statistic | Solar | A/late-B | mid-B | early-B | O-type |
|---|---|---|---|---|---|
| f_mult;q>0.1 | 0.50 ± 0.04 | 0.84 ± 0.11 | 1.3 ± 0.2 | 1.6 ± 0.2 | 2.1 ± 0.3 |
| f_logP<3.7;q>0.1 (close) | 0.15 ± 0.03 | 0.37 ± 0.08 | 0.63 ± 0.13 | 0.8 ± 0.2 | 1.0 ± 0.2 |
| F_n=0 (single) | 0.60 ± 0.04 | 0.41 ± 0.08 | 0.24 ± 0.08 | 0.16 ± 0.09 | 0.06 ± 0.06 |
| F_n=1 (binary) | 0.30 ± 0.04 | 0.37 ± 0.06 | 0.36 ± 0.08 | 0.32 ± 0.10 | 0.21 ± 0.11 |
| F_n≥2 (triple+) | 0.10 ± 0.02 | 0.22 ± 0.07 | 0.40 ± 0.10 | 0.52 ± 0.13 | 0.73 ± 0.16 |
| **f_logP=1;q>0.1** | 0.027 ± 0.009 | 0.07 ± 0.02 | 0.14 ± 0.04 | 0.19 ± 0.06 | 0.29 ± 0.08 |
| **f_logP=3;q>0.1** | 0.057 ± 0.016 | 0.12 ± 0.04 | 0.22 ± 0.07 | 0.26 ± 0.09 | 0.32 ± 0.11 |
| **f_logP=5;q>0.1** | 0.095 ± 0.018 | 0.13 ± 0.03 | 0.20 ± 0.06 | 0.23 ± 0.07 | 0.30 ± 0.09 |
| **f_logP=7;q>0.1** | 0.075 ± 0.015 | 0.09 ± 0.02 | 0.11 ± 0.03 | 0.13 ± 0.04 | 0.18 ± 0.05 |
| F_twin (logP=1) | 0.30 ± 0.09 | 0.22 ± 0.07 | 0.17 ± 0.05 | 0.14 ± 0.04 | 0.08 ± 0.03 |
| F_twin (logP=3) | 0.20 ± 0.06 | 0.10 ± 0.04 | <0.03 | <0.03 | <0.03 |
| F_twin (logP=5) | 0.10 ± 0.03 | <0.03 | <0.03 | <0.03 | <0.03 |
| F_twin (logP=7) | <0.03 | <0.03 | <0.03 | <0.03 | <0.03 |
| γ_largeq (logP=1) | −0.5 ± 0.3 | −0.5 ± 0.3 | −0.5 ± 0.3 | −0.5 ± 0.3 | −0.5 ± 0.3 |
| γ_largeq (logP=3) | −0.5 ± 0.3 | −0.9 ± 0.3 | −1.7 ± 0.3 | −1.7 ± 0.3 | −1.7 ± 0.3 |
| γ_largeq (logP=5) | −0.5 ± 0.3 | −1.4 ± 0.3 | −2.0 ± 0.3 | −2.0 ± 0.3 | −2.0 ± 0.3 |
| γ_largeq (logP=7) | −1.1 ± 0.3 | −2.0 ± 0.3 | −2.0 ± 0.3 | −2.0 ± 0.3 | −2.0 ± 0.3 |
| γ_smallq (logP=1) | 0.3 ± 0.4 | 0.2 ± 0.4 | 0.1 ± 0.4 | 0.1 ± 0.4 | 0.1 ± 0.4 |
| γ_smallq (logP=3) | 0.3 ± 0.6 | 0.1 ± 0.6 | −0.2 ± 0.6 | −0.2 ± 0.6 | −0.2 ± 0.6 |
| γ_smallq (logP=5) | 0.3 ± 0.4 | −0.5 ± 0.4 | −1.2 ± 0.4 | −1.2 ± 0.4 | −1.2 ± 0.4 |
| γ_smallq (logP=7) | 0.3 ± 0.3 | −1.0 ± 0.3 | −1.5 ± 0.3 | −1.5 ± 0.3 | −1.5 ± 0.3 |
| η (logP=2) | 0.1 ± 0.3 | 0.3 ± 0.3 | 0.6 ± 0.3 | 0.7 ± 0.3 | 0.7 ± 0.3 |
| η (logP=4) | 0.4 ± 0.3 | 0.5 ± 0.3 | 0.7 ± 0.3 | 0.8 ± 0.3 | 0.8 ± 0.3 |

### F.2b The f_logP;q>0.3 grid you actually asked for — computed from Eqns 20–23 [V equations, computed table]

```
f_logP<1;q>0.3   (M1) = 0.020 + 0.04 log(M1/M☉) + 0.07 [log(M1/M☉)]²      (Eqn 20)
f_logP=2.7;q>0.3 (M1) = 0.039 + 0.07 log(M1/M☉) + 0.01 [log(M1/M☉)]²      (Eqn 21)
f_logP=5.5;q>0.3 (M1) = 0.078 − 0.05 log(M1/M☉) + 0.04 [log(M1/M☉)]²      (Eqn 22)
```

Eqn 23, with **α = 0.018, ΔlogP = 0.7**, writing a ≡ f_logP<1, b ≡ f_logP=2.7, c ≡ f_logP=5.5:

| logP range | f_logP;q>0.3 |
|---|---|
| 0.2 ≤ logP < 1.0 | a |
| 1.0 ≤ logP < 2.7−ΔlogP | a + (logP−1)/(1.7−ΔlogP) × (b − a − αΔlogP) |
| 2.7−ΔlogP ≤ logP < 2.7+ΔlogP | b + α(logP − 2.7) |
| 2.7+ΔlogP ≤ logP < 5.5 | b + αΔlogP + (logP−2.7−ΔlogP)/(2.8−ΔlogP) × (c − b − αΔlogP) |
| 5.5 ≤ logP < 8.0 | c · exp[−0.3(logP − 5.5)] |

| logP | M1=1 | M1=3.5 | M1=7 | M1=12.5 | M1=28 |
|---|---|---|---|---|---|
| 0.2 | 0.0200 | 0.0625 | 0.1038 | 0.1481 | 0.2245 |
| 0.5 | 0.0200 | 0.0625 | 0.1038 | 0.1481 | 0.2245 |
| **1.0** | **0.0200** | **0.0625** | **0.1038** | **0.1481** | **0.2245** |
| 1.5 | 0.0232 | 0.0650 | 0.0982 | 0.1317 | 0.1866 |
| **2.0** | **0.0264** | **0.0674** | **0.0927** | **0.1152** | **0.1486** |
| 2.5 | 0.0354 | 0.0764 | 0.1017 | 0.1242 | 0.1576 |
| **3.0** | **0.0444** | **0.0854** | **0.1107** | **0.1332** | **0.1666** |
| 3.5 | 0.0529 | 0.0912 | 0.1153 | 0.1371 | 0.1698 |
| **4.0** | **0.0591** | **0.0841** | **0.1026** | **0.1207** | **0.1497** |
| 4.5 | 0.0654 | 0.0769 | 0.0898 | 0.1042 | 0.1296 |
| **5.0** | **0.0717** | **0.0698** | **0.0771** | **0.0877** | **0.1095** |
| 5.5 | 0.0780 | 0.0626 | 0.0643 | 0.0713 | 0.0894 |
| 6.0 | 0.0671 | 0.0539 | 0.0554 | 0.0614 | 0.0770 |
| 6.5 | 0.0578 | 0.0464 | 0.0476 | 0.0528 | 0.0662 |
| 7.0 | 0.0497 | 0.0399 | 0.0410 | 0.0455 | 0.0570 |
| 7.5 | 0.0428 | 0.0344 | 0.0353 | 0.0391 | 0.0491 |
| 8.0 | 0.0368 | 0.0296 | 0.0304 | 0.0337 | 0.0422 |

**Validation**: integrating this over 0.2 < logP < 8.0 gives f_mult;q>0.3 =
0.367 / 0.500 / 0.633 / 0.777 / 1.026, against the paper's published 0.36 (solar),
0.63 ± 0.09 (mid-B), 1.02 ± 0.16 (O-type). Exact match.

**Empirical solar-type counts — Table 11** [V] (Raghavan-based, N_prim = 404, N_comp = 193):

| logP | N_smallq | N_largeq | f_logP;q>0.3 | η |
|---|---|---|---|---|
| 0.5±0.5 | 1 | 7 | 0.017 ± 0.007 | −0.8 ± 0.2 |
| 1.5±0.5 | 3 | 8 | 0.020 ± 0.007 | −0.4 ± 0.3 |
| 2.5±0.5 | 3 | 10 | 0.025 ± 0.008 | 0.2 ± 0.3 |
| 3.5±0.5 | 6 | 18 | 0.045 ± 0.011 | 0.6 ± 0.4 |
| 4.5±0.5 | 7 | 27 | 0.067 ± 0.013 | 0.3 ± 0.3 |
| 5.5±0.5 | 9 | 31 | 0.077 ± 0.014 | — |
| 6.5±0.5 | 9 | 23 | 0.057 ± 0.012 | — |
| 7.5±0.5 | 10 | 21 | 0.052 ± 0.011 | — |

### F.2c Mass-ratio distribution [V]

`p_q ∝ q^γ_smallq` (0.1<q<0.3), `∝ q^γ_largeq` (0.3<q<1.0), plus an **excess** twin fraction
F_twin uniform over 0.95<q<1.00. F_twin is the *excess* over the power law, not the total.

**F_twin — Eqns 5, 6, 7:**
```
F_twin(M1,P) = F_twin;logP<1                                  for logP < 1
             = F_twin;logP<1 × [1 − (logP−1)/(logP_twin−1)]   for 1 ≤ logP < logP_twin
             = 0                                              for logP ≥ logP_twin      (5)

F_twin;logP<1(M1) = 0.30 − 0.15 log(M1/M☉)                                              (6)

log P_twin(M1) = 8.0 − M1/M☉    for M1 ≤ 6.5 M☉
               = 1.5            for M1 > 6.5 M☉                                         (7)
```
σ(F_twin) = max{0.03, 0.3 F_twin} (Eqn 8).

> **Eqn 7 is a transcription trap.** Every PDF text extractor renders it as though there were
> a 1.5 *divisor*. There is not — it is `8.0 − M1/M☉`, and "1.5" is the second branch's
> constant value. The two branches are continuous at 6.5 M☉. Confirmed visually and by
> reproducing all 20 F_twin entries of Table 13. [V]

**γ_largeq — Eqns 9, 10, 11:**

| M1 | logP range | γ_largeq |
|---|---|---|
| 0.8–1.2 M☉ (Eqn 9) | 0.2 ≤ logP < 5.0 | −0.5 |
| | 5.0 ≤ logP < 8.0 | −0.5 − 0.3(logP − 5) |
| 3.5 M☉ (Eqn 10) | 0.2 ≤ logP < 1.0 | −0.5 |
| | 1.0 ≤ logP < 4.5 | −0.5 − 0.2(logP − 1.0) |
| | 4.5 ≤ logP < 6.5 | −1.2 − 0.4(logP − 4.5) |
| | 6.5 ≤ logP < 8.0 | −2.0 |
| >6.0 M☉ (Eqn 11) | 0.0 ≤ logP < 1.0 | −0.5 |
| | 1.0 ≤ logP < 2.0 | −0.5 − 0.9(logP − 1) |
| | 2.0 ≤ logP < 4.0 | −1.4 − 0.3(logP − 2) |
| | 4.0 ≤ logP < 8.0 | −2.0 |

σ(γ_largeq) = 0.3 throughout (Eqn 12).

**γ_smallq — Eqns 13, 14, 15:**

| M1 | logP range | γ_smallq |
|---|---|---|
| 0.8–1.2 M☉ (Eqn 13) | 0.2 < logP < 8.0 | **0.3** (constant) |
| 3.5 M☉ (Eqn 14) | 0.2 ≤ logP < 2.5 | 0.2 |
| | 2.5 ≤ logP < 5.5 | 0.2 − 0.3(logP − 2.5) |
| | 5.5 ≤ logP < 8.0 | −0.7 − 0.2(logP − 5.5) |
| >6.0 M☉ (Eqn 15) | 0.2 ≤ logP < 1.0 | 0.1 |
| | 1.0 ≤ logP < 3.0 | 0.1 − 0.15(logP − 1) |
| | 3.0 ≤ logP < 5.6 | −0.2 − 0.50(logP − 3) |
| | 5.6 ≤ logP < 8.0 | −1.5 |

σ(γ_smallq) (Eqn 16): 0.4 (0.2≤logP<1); 0.4+0.1(logP−1) (1≤logP<3); 0.6−0.1(logP−3)
(3≤logP<6); 0.3 (6≤logP<8).

**Interpolate linearly in M1** (not log M1) between Eqns 9↔10 for M1 = 1.2–3.5 M☉ and 10↔11
for 3.5–6.0 M☉, similarly 13↔14↔15. **That choice is what reproduces Table 13** — the
independent reimplementation matched all 20 entries within 1σ, exactly in the solar column,
and recovered the paper's stated ratios f_q>0.1/f_q>0.3 = 1.25 at short P (paper "≈1.3") and
3.09 at logP = 6.5, M1 ≥ 6 (paper "≈3.1"). [V]

### F.2d Eccentricity [V]

**e_max is theirs, and it is exactly the form in the brief** — **Eqn 3, §2, p. 5**:
```
e_max(P) = 1 − (P / 2 days)^(−2/3)      for P > 2 days                                  (3)
```
*"This relation guarantees the binary components have Roche-lobe fill-factors ≲70% at
periastron."* All binaries with **P ≤ 2 d are assumed circularised**.

**η(P, M1) — Eqns 17, 18**, for p(e) ∝ e^η:
```
η(0.8 < M1/M☉ < 3,  0.5 < log P < 6.0) = 0.6 − 0.7 / (log P − 0.5)                       (17)
η(M1 > 7 M☉,        0.5 < log P < 5.0) = 0.9 − 0.2 / (log P − 0.5)                       (18)
```
Interpolate across M1 = 3–7 M☉. σ(η) = 0.3 for 1 < logP < 5 (Eqn 19). η = 1 is thermal,
η = 0 uniform. The pole sits exactly at the lower domain bound logP = 0.5 (P ≈ 3 d), which is
how the fit drives short-period orbits circular. Solar-type asymptote η ≈ 0.5; early-type
η ≈ 0.9. Zero-age-MS circularisation period, both late- and early-type: **P ≈ 2–6 d**.

Minor inconsistency: Table 13 gives η = 0.6 for mid-B at logP = 2 while Eqn 18 at M1 = 7
gives 0.77 — a 0.17 offset, inside the quoted ±0.3.

### F.2e Multiplicity vs primary mass [V]

f_mult;q>0.1 = **0.50 ± 0.04** (1 M☉) → 0.84 ± 0.11 (2–5) → 1.3 ± 0.2 (5–9) →
1.6 ± 0.2 (9–16) → **2.1 ± 0.3** (>16 M☉). f_mult;q>0.3 = 0.36 ± 0.03 (1 M☉),
1.02 ± 0.16 (28 M☉). Multiplicity counts are consistent with a **Poisson** distribution.
Error-propagation coherence length: l_logP(M1) = 1.0 + 0.7 log(M1/M☉) (Eqn 26).

## F.3 Tidal circularisation

**Meibom & Mathieu 2005, ApJ 620, 970**, bibcode `2005ApJ...620..970M`,
arXiv:astro-ph/0412147. Their **Eqn 1**: e(P) = 0 for P ≤ P′; = α[1 − e^{β(P′−P)}]^γ for
P > P′, with **α = 0.35, β = 0.14, γ = 1.0 all fixed** — P′ is the only free parameter.
**P_circ is the period at which the fitted curve reaches e = 0.01.** Their **Table 3**;
the P_circ-vs-age plot is **Figure 9**.

| Population | Age | P_circ (d) |
|---|---|---|
| PMS binaries | ~3 Myr | 7.1 (+1.2/−1.2) |
| Pleiades | ~100 Myr | 7.2 (+1.8/−1.9) |
| M35 | ~150 Myr | 10.2 (+1.0/−1.5) |
| Hyades/Praesepe | ~630 Myr | 3.2 (+1.2/−1.2) ← see caveat |
| **M67** | **~4 Gyr** | **12.1 (+1.0/−1.5)** |
| NGC 188 | ~6.3 Gyr | 14.5 (+1.4/−2.2) |
| Field | ~8.9 Gyr | 10.3 (+1.5/−3.1) |
| Halo | ~10 Gyr | 15.6 (+2.3/−3.2) |

All [V]. **NGC 6819 is not in this paper.** The Hyades/Praesepe 3.2 d is flagged by the
authors themselves as the exception to the age trend, driven by two single-lined systems
whose secondaries may be white dwarfs; excluding them the same algorithm returns ≈7 d.

Follow-ups [V unless noted]: **Milliman et al. 2014, AJ 148, 38** (`2014AJ....148...38M`),
NGC 6819 at 2.5 Gyr, **P_circ = 6.2 ± 1.1 d**; **Geller et al. 2021**
(`2021AJ....161..190G` [L]), M67, 11.0 (+1.1/−1.0); **Narayan et al. 2026, AJ 171, 102** [L]
(arXiv:2509.12315), NGC 188, 14.4 (+0.14/−0.11) — but that is an MCMC posterior interval and
is **not** methodologically comparable to Meibom & Mathieu's Monte Carlo intervals, so do not
present it as a 10× precision gain.

> **"~10 d for solar-type stars" is a fair one-liner**, but the real spread at a few Gyr is
> **6–15 d**, and NGC 6819 at 6.2 d breaks the monotonic age trend.

**Zahn / Hut — one correction to the brief**: (a/R)^{21/2} is the *radiative-damping dynamical
tide* (early-type), **not** the turbulent-convective equilibrium tide that applies to
solar-type stars. [V]

| Regime | t_circ | t_sync |
|---|---|---|
| Equilibrium tide, turbulent convection (Zahn 1977/1989) — **solar-type** | **∝ (a/R)⁸** | **∝ (a/R)⁶** |
| Dynamical tide, radiative damping (Zahn 1975/1977) — early-type | ∝ (a/R)^{21/2} | ∝ (a/R)^{17/2} |

In period form t_circ ∝ P^{16/3} ≈ P^{5.3}. **Synchronisation always precedes
circularisation**: the (a/R)² ratio is 10²–10³ at a/R ≈ 10–20, and Meibom, Mathieu & Stassun
2006 state directly **t_circ ≃ 10³ t_sync** for solar-type components. Binaries are routinely
synchronised but eccentric; never the reverse. Bibcodes: Zahn 1977 `1977A&A....57..383Z`;
Zahn 1989 `1989A&A...220..112Z`; Zahn & Bouchet 1989 `1989A&A...223..112Z`;
Hut 1981 `1981A&A....99..126H`. Live challenge: **Zanazzi 2022, ApJL 929, L27** argues
cluster P_circ values are biased high and the field value is ~3 d.

## F.4 Pecaut & Mamajek — the live table **works** [V]

`http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt` — fetched
successfully (HTTPS on `www.pas.rochester.edu` also works). **Version 2022.04.16.** Header
verbatim:

```
# "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
# http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
# Eric Mamajek
# Version 2022.04.16
```

Cite **Pecaut & Mamajek 2013, ApJS 208, 9** (`2013ApJS..208....9P`). The live file is
*larger* than the published Table 5, which lacks the absolute magnitudes, luminosities and
the O3–O8 / L / T / Y rows.

### Exact column names, in order — 32 columns [V]

`#SpT` · `Teff` · `logT` · `BCv` · `logL` · `Mbol` · `R_Rsun` · `Mv` · `B-V` · `Bt-Vt` ·
`G-V` · `Bp-Rp` · `G-Rp` · `M_G` · `b-y` · `U-B` · `V-Rc` · `V-Ic` · `V-Ks` · `J-H` ·
`H-Ks` · `M_J` · `M_Ks` · `Ks-W1` · `W1-W2` · `W1-W3` · `W1-W4` · `g-r` · `i-z` · `z-Y` ·
`Msun` · `#SpT`

Against the checklist in the brief:

- **Present**: SpT, Teff, logT, BCv, Mv, logL, B-V, Bt-Vt, G-V, **Bp-Rp**, **G-Rp**, **M_G**,
  b-y, U-B, V-Rc, V-Ic, V-Ks, J-H, H-Ks, Ks-W1, W1-W2, W1-W3, W1-W4, **Msun**, M_J, M_Ks,
  Mbol, i-z, z-Y, **R_Rsun** — plus **`g-r`**, which the brief did not list.
- **ABSENT: `logAge`.** There is no age column at all.
- `b-y` appears **once** (the brief listed it twice); `R_Rsun` is the only radius column
  (there is no separate `Rsun`).

Example rows [V]:

| SpT | Teff | Msun | R_Rsun | M_G | G-Rp | Bp-Rp | Mv | logL |
|---|---|---|---|---|---|---|---|---|
| O5V | 41400 | 43 | 11.45 | … | … | … | −5.35 | 5.54 |
| B0V | 31400 | 17.7 | 7.16 | … | … | … | −3.90 | 4.65 |
| **A0V** | 9700 | 2.18 | 2.193 | 1.00 | −0.020 | −0.037 | 0.99 | 1.58 |
| F0V | 7220 | 1.61 | 1.728 | 2.51 | 0.230 | 0.377 | 2.57 | 0.86 |
| **G2V** | 5770 | 1.00 | 1.012 | 4.635 | 0.459 | 0.823 | 4.80 | 0.01 |
| K0V | 5270 | 0.88 | 0.813 | 5.553 | 0.56 | 0.983 | 5.78 | −0.34 |
| **K5V** | 4440 | 0.70 | 0.701 | 6.83 | 0.74 | 1.43 | 7.28 | −0.76 |
| M0V | 3850 | 0.57 | 0.588 | 8.16 | 0.92 | 1.84 | 8.80 | −1.16 |
| M5V | 3060 | 0.162 | 0.196 | 12.45 | 1.33 | 3.35 | 14.15 | −2.52 |

~220 rows, O3V through Y4V [L — row count not independently confirmed]. `…` marks missing
values; note `b-y` is missing for G2V and K5V in this version.

> **Provenance warning for a population generator**: `R_Rsun` is **not independent**. It is
> back-computed from Mbol and Teff — `R = 10^{(Mbol,⊙ − Mbol)/5} × (Teff,⊙/Teff)²` with
> Mbol,⊙ = 4.74 and Teff,⊙ = 5772 K reproduces A0V (2.193) and K5V (0.701) exactly [V]. So
> logL, Teff and R_Rsun carry only **two** degrees of freedom; treat the table as mass plus
> two, or you will double-count. `Msun` itself comes from a polynomial Mv–mass fit.

## F.5 Eker et al. 2018 — MNRAS 479, 5491, bibcode `2018MNRAS.479.5491E`, arXiv:1807.02568

509 MS stars from detached double-lined EBs. Verified against a rendered image of the
published tables; parenthesised digits are errors on the last decimals.

### Table 4 — six-piece classical MLR, `log L = a log M + b` (solar units) [V]

| Domain | N | Mass range | a (= α) | b | R² | σ |
|---|---|---|---|---|---|---|
| Ultra low-mass | 22 | 0.179 < M/M☉ ≤ 0.45 | **2.028** (135) | **−0.976** (070) | 0.919 | 0.076 |
| Very low-mass | 35 | 0.45 < M/M☉ ≤ 0.72 | **4.572** (319) | **−0.102** (076) | 0.857 | 0.109 |
| Low mass | 53 | 0.72 < M/M☉ ≤ 1.05 | **5.743** (413) | **−0.007** (026) | 0.787 | 0.129 |
| Intermediate | 275 | 1.05 < M/M☉ ≤ 2.40 | **4.329** (087) | **+0.010** (019) | 0.901 | 0.140 |
| High mass | 80 | 2.4 < M/M☉ ≤ 7 | **3.967** (143) | **+0.093** (083) | 0.907 | 0.165 |
| Very high mass | 44 | 7 < M/M☉ ≤ 31 | **2.865** (155) | **+1.105** (176) | 0.888 | 0.152 |

Their §3.1 text says "α = 2.868" for the last row — a typo; the table reads 2.865. [V]

### Table 5 — mass–radius and mass–Teff [V]

```
MRR, 0.179 ≤ M/M☉ ≤ 1.5  (N=233):  R = 0.438(098)·M² + 0.479(180)·M + 0.075(479)
                                    R² = 0.867, σ = 0.176      [LINEAR units, not log]
MRR, 1.5 < M/M☉ ≤ 31     (N=276):  derived from MLR + MTR via L = 4πR²σT⁴, σ = 0.787

MTR, 0.179 ≤ M/M☉ ≤ 1.5  (N=233):  derived from MLR + MRR via L = 4πR²σT⁴, σ = 0.025
MTR, 1.5 < M/M☉ ≤ 31     (N=276):  log Teff = −0.170(026)(log M)² + 0.888(037) log M
                                               + 3.671(010)
                                    R² = 0.961, σ = 0.042
```

The three relations are **deliberately interdependent**: only two are fitted in each mass
range and the third follows from Stefan–Boltzmann. Sanity checks: M = 1 → R = 0.992;
M = 1.5 → Teff = 6637 K; M = 10 → 24,490 K; M = 31 → 41,400 K. Their **Table 6** gives binned
locus points (M, R, Teff, log g, SpT, Mbol, M/L, L/M) in 36 mass bins; **Table 7** gives
SpT ↔ Teff/B−V/U−B/BC/Mv/M/R/log g. Both are directly usable.

### Is there a newer Eker? **No.** [V]

All 22 of Z. Eker's astro-ph.SR arXiv submissions were enumerated. The 2024 paper —
**Eker, Soydugan & Bilir 2024, "Fundamentals of Stars: Critical Looks at Mass-Luminosity
Relations and Beyond", Physics and Astronomy Reports 2, 41** (arXiv:2402.07947) — is a
**review** with no new fitted coefficients, as is **Eker et al. 2025** (arXiv:2503.19965) on
bolometric corrections. The intervening papers (2020 MNRAS 496, 3887; 2021 MNRAS 501/507)
concern bolometric corrections, not the mass relations. **Eker et al. 2018 remains the
current calibration.** Predecessor: Eker et al. 2015, AJ 149, 131 (`2015AJ....149..131E`).

## F.6 Eclipse probability geometry

**Winn 2010, "Transits and Occultations", in *Exoplanets* (ed. Seager)**, arXiv:1001.2010,
bibcode `2010exop.book...55W`. All [V]:

| Eq. | Expression |
|---|---|
| (7) | b_tra = (a cos i / R★)(1 − e²)/(**1 + e sin ω**) |
| (8) | b_occ = (a cos i / R★)(1 − e²)/(**1 − e sin ω**) |
| **(9)** | **p_tra = ((R★ ± R_p)/a) · (1 + e sin ω)/(1 − e²)** |
| **(10)** | **p_occ = ((R★ ± R_p)/a) · (1 − e sin ω)/(1 − e²)** |
| (11) | p = R★/a ≈ 0.005 (R★/R☉)(a/AU)⁻¹ |
| **(12)** | **p_tra = p_occ = ((R★ ± R_p)/a) · 1/(1 − e²)**  ← the ω-averaged form |

**Sign convention**: the **primary eclipse / transit takes (1 + e sin ω)**, the **secondary /
occultation takes (1 − e sin ω)**. Forced by his Eqn (6): f_tra = π/2 − ω ⇒ cos f = sin ω ⇒
r_tra = a(1−e²)/(1+e sin ω), and p ∝ 1/r_conj. The ± is Winn's own: *"the '+' sign allows
grazing eclipses and the '−' sign excludes them."*

**Circular limit P_ecl = (R1+R2)/a is exact**, not a small-angle approximation, because cos i
is uniform on [0,1] for isotropic orbit normals [V]. Three regimes:

| cos i | Outcome | Probability |
|---|---|---|
| ≤ (R1−R2)/a | total / annular | **(R1−R2)/a** |
| (R1−R2)/a < cos i ≤ (R1+R2)/a | partial / grazing | 2R2/a |
| > (R1+R2)/a | none | — |

Total-eclipse probability vanishes for equal-radius twins.

**The ω-averaged eccentric result ⟨p⟩ = (R1+R2)/a · 1/(1−e²)** is verified in four primary
sources. The cleanest explicit marginalisation over ω is **Kipping 2014, MNRAS 444, 2263**
(`2014MNRAS.444.2263K`), whose Eqn (3) states P(ω) = 1/2π, Eqn (5) writes the integral, and
**Eqn (6) gives P(b̂|e,a_R) = (1/a_R)·1/(1−e²)**. **Caveat**: Barnes 2007 (PASP 119, 986)
reaches the same expression by integrating over *true anomaly*, not over ω — cite Kipping or
Winn if you need the literal uniform-ω statement. Burke 2008 (ApJ 679, 1566) Eqn (2) gives
the same (1−e²)⁻¹ enhancement and reports ⟨enhancement⟩ ≈ **1.25** for the observed exoplanet
eccentricity distribution.

**Semi-major axis constant** [V, by direct computation with IAU 2015 nominal constants]:
```
a = 4.208 (M_tot/M☉)^{1/3} (P/day)^{2/3} R☉
```
(the 4th significant figure is convention-bound: 4.206 with the older R☉ = 6.96×10⁸ m).

**Eclipsing fraction vs period.** No paper tabulates a purely geometric eclipse fraction vs
period for solar-type binaries; it follows from R/a as **P_ecl ∝ P^(−2/3)**:

| P (d) | 1 | 3 | 10 | 100 | 1000 |
|---|---|---|---|---|---|
| Twin solar (1+1 M☉, 1+1 R☉): 0.377 P^(−2/3) | 37.7% | 18.1% | **8.13%** | 1.75% | 0.377% |
| Solar + M dwarf (1+0.5 M☉, 1+0.5 R☉): any eclipse | 31.1% | 15.0% | 6.71% | 1.45% | 0.311% |
| … total only | 10.4% | 4.99% | 2.24% | 0.482% | 0.104% |
| Early-B pair (10+5 M☉, 4.0+2.6 R☉) | 63.6% | 30.6% | 13.7% | 2.95% | 0.636% |

> **Published eclipse fractions are detectability-weighted and are much smaller — do not
> confuse them with the geometric numbers.** Söderhjelm & Dischler 2005 (A&A 442, 1003)
> Table A.1, FG bin, Δm ≥ 0.1 mag: 6.6% at 1.02 d, 1.8% at 2.88 d, **0.20% at 8.16 d** —
> 47× below geometric at 8 d, because they require Δm > 0.1 mag for at least 5% of the orbit.
> Moe & Di Stefano 2013 (ApJ 778, 95) state the P^(−2/3) scaling explicitly. Kepler
> survey-integrated EB occurrence: **1.3%** (Kirk et al. 2016), 1.2% (Prša+2011),
> 1.4% (Slawson+2011).

## F.7 Rotation

### The synchronisation formula [V, by explicit arithmetic]

```
v_eq [km/s] = 2π R☉ (R/R☉) / (86400 P_d) = 50.5927 × (R/R☉)/(P/d)
```
with IAU 2015 nominal R☉ = 6.957×10⁸ m. **The brief's 50.6 is right** (50.593; it is 50.615
with the older 6.960×10⁸ m). Inverse: P_rot[d] = 50.593 (R/R☉)/v.

Sun check: R = 1, P = 25.4 d → 1.99 km/s. Synchronised solar-type values: 16.9 km/s at 3 d,
10.1 at 5 d, 5.06 at 10 d, 2.53 at 20 d. This is v_eq; observed vsini carries
⟨sin i⟩ = π/4 ≈ 0.785 for random orientation — **but in an eclipsing or synchronised binary
i ≈ 90°, so vsini ≈ v_eq**, which is exactly the regime this benchmark is in.

### Observed field-FGK vsini

| Source | Bibcode | Numbers | Status |
|---|---|---|---|
| **Nordström et al. 2004, A&A 418, 989** (Geneva-Copenhagen Survey) | `2004A&A...418..989N` | 16,682 F/G dwarfs, vsini from CORAVEL CCF widths. §3.3, Fig. 4: *"the great majority of the programme stars have rotations below 20 km s⁻¹."* Precision: *"vsini … only given to the nearest km s⁻¹, and from 30 km s⁻¹ and upwards only to the nearest 5 or 10 km s⁻¹."* Catalogue **column (36) = vsini**. | [V] |
| | | **They tabulate no median vsini per colour bin** — only the Fig. 4 histogram. | [V] (negative result) |
| **Głębocki & Gnaciński 2005, VizieR III/244** | `2005yCat.3244....0G` | "Catalog of Stellar Rotational Velocities": **28,179 stars**, **39,351 individual vsini measurements** in `catalog.dat`, each with error and method. | [L] (VizieR ReadMe Anubis-blocked; from the ADS/CDS listing) |
| **Valenti & Fischer 2005, ApJS 159, 141** (SPOCS I) | `2005ApJS..159..141V` | 1040 F/G/K dwarfs; **vsini precision 0.5 km/s per spectrum** (abstract). A median of 2.4 km/s, max 54.7, only 44 stars > 11 km/s is widely repeated but **could not be traced to the paper** — do not quote it unchecked. | precision [V]; median **[U]** |
| **McQuillan, Mazeh & Aigrain 2014, ApJS 211, 24** | `2014ApJS..211...24M` | **34,030** Kepler MS rotation periods (25.6% of 133,030 targets), **P = 0.2–70 d**. Bimodal: peaks ~20 d and ~40 d at 3500 K; ~14 d and ~30 d at 4000 K; *"not visible above ~4500 K."* Detection fractions (Table 3): 0.83 (<4000 K), 0.69 (4000–4500), 0.43 (4500–5000), 0.27 (5000–5500), 0.16 (5500–6000), 0.20 (6000–6500). | [V] |

> **Recommendation**: build the rotation model on **periods** (McQuillan) and convert with
> 50.593 R/P, rather than on vsini catalogues directly — the latter are dominated by a
> ~1–3 km/s macroturbulence floor for cool dwarfs, which is exactly the regime where a
> disentangling benchmark's line widths matter most.

**Kraft break** — Kraft 1967, ApJ 150, 551 (`1967ApJ...150..551K`): at ~F5, M ≈ 1.3 M☉,
Teff ≈ 6200 K, the radiative/convective-envelope boundary. Above it vsini ≈ 40–150 km/s;
below it ≲25 km/s, typically ≲10 km/s [L, from secondary sources; Kraft 1967 itself not
fetched]. **Skumanich 1972** (`1972ApJ...171..565S`): v ∝ t^(−1/2) [L].

**Do synchronised binaries actually follow v = 2πR/P?** The observational study is
**Meibom, Mathieu & Stassun 2006, ApJ 653, 621** (`2006ApJ...653..621M`), "Tidal
Synchronization in Solar-Type Binary Stars in M35 and M34": 2 of 4 well-characterised close
systems already synchronous, and **t_circ ≃ 10³ t_sync** [V]. So the synchronisation boundary
sits at substantially longer periods than P_circ ≈ 10 d — **assume synchronism out to at
least P ≈ 10–20 d for solar-type MS binaries**, and expect synchronised-but-eccentric systems
in the 10–20 d range. Note this paper reports **no new P_circ**; it is the synchronisation
paper, not a circularisation one.

## F.8 A tension to resolve before coding

**Raghavan's flat eccentricity distribution disagrees with both Duquennoy & Mayor 1991 and
Moe & Di Stefano's η(P).** MDS17's η ≈ 0.4 over logP = 1.5–5 is a mild rise, not flat. Pick
one and state which — for a benchmark whose truths must be reproducible, the MDS17 η(P)
prescription is the better choice because it is analytic, mass-dependent, and comes with a
stated σ(η) = 0.3.

---

# G. Additional findings

## G.1 GSP-Phot Teff for Gaia SB2s [V]

`teff_gspphot` is present for **4,263 of the 5,376** SB2+SB2C sources (79.3%): 3,663/4,630
for SB2 and 600/746 for SB2C. Mean `teff_gspphot` over the SB2+SB2C sample = **6,504.8 K**.

Treat these with suspicion as *component* temperatures: GSP-Phot fits a **single-star** model
to the BP/RP spectrum of what is by construction a composite source. For an SB2 the result is
a flux-weighted blend, biased toward the primary and pulled by the secondary's contribution.
It is a reasonable prior for the *system*, not a truth for either star. The
`nss_two_body_orbit.temperature_ratio` column is the only Gaia-internal T₂/T₁, and it exists
only for the eclipsing types (§A.2).

## G.2 The light-ratio distribution among Gaia binaries [V]

**There is no light ratio for Gaia SB2s.** `g_luminosity_ratio` is NULL for all 5,376
(§A.2), and `binary_masses.fluxratio` is NULL for all 3,856 `SB2+M1` rows — only the
bracketing `fluxratio_lower`/`fluxratio_upper` are populated there (§A.5).

The measured distribution that *does* exist is over the eclipsing types:

| type | n | min | max | mean | sd |
|---|---|---|---|---|---|
| `EclipsingBinary` | 86,918 | 1.61e−6 | 0.99995 | **0.3181** | **0.2503** |
| `EclipsingSpectro` | 155 | 0.00612 | 0.99262 | 0.1530 | 0.1868 |

So among Gaia EBs the G-band secondary/primary luminosity ratio is broad and centred near
**0.32**, spanning the full range to 1. That is a defensible empirical prior for drawing light
ratios in the benchmark. `binary_masses.fluxratio` where it is populated gives mean 0.5054
(`EclipsingSpectro(SB2)`, n=3), 0.5844 (`Orbital+SB2`, n=23) and 0.0234
(`AstroSpectroSB1+M1`, n=17,578) [V] — the last is an astrometric-selection population of
very unequal pairs and is not representative.

`temperature_ratio` for `EclipsingBinary`: mean 1.409, range 0.0033 to **25.88** [V] — the
upper tail is clearly unphysical, so filter. For `EclipsingSpectro`: mean 1.456, range
0.746–3.105. Note the ratio is **> 1 on average**, so check `temperature_ratio_definition`
(a byte code, §A.3) before assuming it means T₂/T₁.

`inclination` for `EclipsingBinary`: mean 78.8°, **min 34.0°** [V] — the low tail is contact
or near-contact systems where large fractional radii permit eclipses far from edge-on.

## G.3 Overlap between Gaia solution types [V]

Only **67 sources** carry both an `SB2`/`SB2C` and an `EclipsingBinary` solution.
Combined with the 155 `EclipsingSpectro` and the `binary_masses` rows
`Eclipsing+SB2` (53) and `EclipsingSpectro(SB2)` (3), **the fully-specified Gaia truth set —
spectroscopic orbit plus inclination plus light ratio — numbers in the low hundreds, not
thousands.** This is the key sizing fact for the benchmark.

## G.4 Gaia DR4 expectations [V]

From the ESA DR4 content page, last updated **28 June 2026**:

- Release date **2 December 2026** [V].
- Data collected **25 July 2014 (10:30 UTC) to 20 January 2020 (22:00 UTC), spanning
  66 months** [V] (DR3 used 34 months).
- **`nss_two_body_orbit`: 5,901,111 records** [V] — "Non-single-star orbital solution
  parameters for sources compatible with an orbital two-body solution". Against DR3's
  443,205 that is a **13.3× increase**. (The same figure was also reported against the
  astrometric-binary line on that page; if the two are genuinely distinct products one of
  the two attributions is wrong — **[U]**, recheck at release.)
- Spectroscopic-binary trend solutions: **79,653** [V].
- **Eclipsing binaries from variability analysis: 2,392,500** [V] (DR3: 2,184,477).
- Epoch-level RV, `vbroad` and `grvs_mag` products: **787,312** and **6,910,423,894**
  records in two datasets [V].
- **RVS spectra combined at FOV-transit level: 6,910,785,949 records, 49 TB** [V].
- RVs are expected down to the RVS limit **G_RVS ≈ 16** [L], versus ≈14 in DR3.

No DR4-specific SB2 count is published [U]. Note the project's existing
`gaia-rvs-facts` memo already carries the draft `rvs_epoch_spectrum` schema, which is the
table these 6.9 billion records land in.

---

# Recommended draw strategy

## Tier 1 — self-contained truth sets (simulate directly, no cross-match needed)

These give masses, radii, temperatures, period and a magnitude in one file. Use them for the
"parameters matched to real eclipsing binaries" arm of the benchmark.

| Source | n | What you get | What is missing |
|---|---|---|---|
| **DEBCat `debs.dat`** | ~370 | P, M1, M2, R1, R2, Teff1, Teff2, logg1/2, logL1/2, [M/H], V, B−V, SpT1/2, all with errors | **e** (absent), G mag |
| **Torres `table1 ⋈ table2 ⋈ table3`** | 95 systems / 190 stars | P, M, R, Teff, logg, logL, absolute V per component; **e** from table3; **vsini per component** from table2; [Fe/H], age, distance | G mag (but `_RA`/`_DE` are supplied) |

**DEBCat is the primary recommendation.** It is the largest, is actively maintained (370
systems as of March 2026 versus Torres's frozen 95), and carries per-quantity uncertainties.
Its one real gap for an RV simulation is eccentricity — take e from Torres where the systems
overlap, from SB9/SBX by name otherwise, or draw it from the §F prescription conditioned on
the DEBCat period.

Torres remains worth carrying as a second, independent set precisely *because* it supplies
**measured vsini for both components**. Nothing else in this document does. If the benchmark
is to test line broadening honestly rather than assume synchronisation, Torres is the only
empirical anchor.

## Tier 2 — self-contained for orbits, needing a cross-match for stellar parameters

| Source | n | What you get | Cross-match needed |
|---|---|---|---|
| **SBX `orbits ⋈ systems ⋈ alias`** | 5,169 orbits / 4,080 systems; **1,674 with both K1 and K2**, **513 of those Grade ≥ 4** | P, T0, e, ω, K1, K2, V0, rms1/2, grade, **plus the RV time series** | Gaia ids already in `alias` (`catalog='Gaia'`) |
| **Gaia `nss_two_body_orbit` SB2/SB2C** | 5,376 | P, e (NULL⇒0), ω, T_peri, K1, K2, γ, and G from `gaia_source` | `binary_masses` for M1/M2 (3,856 of them) |
| **Guo 2025 LAMOST `J/ApJS/278/46/table1`** | 665 | P, e, K1, K2, v0, ω (rad), t0, q, Δmag, **and a real Gaia DR3 source_id** | Gaia join is trivial |

**Use SBX, not `B/sb9`.** VizieR's copy is stale by ~70 orbits, and SBX additionally carries
the per-epoch radial velocities — which for a disentangling benchmark means you can compare
against the actual observed sampling rather than a synthetic cadence.

## Tier 3 — population shape only (draw distributions, not individual systems)

| Source | n | Use |
|---|---|---|
| **Gaia `vari_eclipsing_binary`** | 2,184,477 | Period distribution (P = 1/`frequency`, capped at 0.2 d), eclipse depths and durations. **Only 34,148 have G < 13** — the RVS-observable tail |
| **ASAS-SN `J/MNRAS/517/2190/table`** | 35,576 | (R₁+R₂)/a, T₂/T₁, i, e, ω, P with Gaia G and BP−RP — the best empirical **eclipse-geometry** distribution |
| **Prša TESS `J/ApJS/258/16/tess-ebs`** | 4,584 | Period / morphology / Tmag distribution |
| **Kirk Kepler `J/AJ/151/68/catalog`** | 2,876 | Period / morphology / Kpmag / Teff distribution |
| **Gaia `EclipsingBinary` NSS rows** | 86,918 | **Light-ratio distribution** (mean 0.318, sd 0.250) and inclination distribution |
| **GALAH `J/A+A/638/A145/catalog`** | 12,760 | **Wavelength-dependent flux ratios** `ratio1-4` per band, plus paired Teff/logg/vbroad |
| **§F parametric prescriptions** | — | P, q, e as functions of M1 when you want to go beyond observed samples |

## Tier 4 — the fully parametric arm (unlimited N, no catalogue at all)

When you want more systems than any catalogue holds, the §F prescriptions compose into a
closed generator with no free choices left:

1. **Draw M1** from an IMF (or fix it to the mass range the benchmark targets).
2. **Draw logP** with weight `f_logP;q>0.3(M1, logP)` from §F.2b (Eqns 20–23), over
   0.2 < logP < 8.0. For RVS-detectable SB2s truncate at the top: the Gaia SB2 sample is
   67% below 10 d (§A.4).
3. **Draw q** from the broken power law of §F.2c — `γ_smallq`, `γ_largeq` as functions of
   (M1, logP) with **linear-in-M1** interpolation, plus the excess twin fraction F_twin from
   Eqns 5–7. Watch Eqn 7: `log P_twin = 8.0 − M1/M☉`, no divisor.
4. **Draw e** from p(e) ∝ e^η with η(P, M1) from Eqns 17–18, truncated at
   `e_max(P) = 1 − (P/2 d)^(−2/3)`, and set e = 0 for P ≤ 2 d.
5. **Map M → Teff, R, logg** with Eker et al. 2018 (§F.5) — it is calibrated *on detached
   eclipsing binaries*, which is exactly this population, so prefer it over Pecaut & Mamajek
   for the mass→radius step. Use Pecaut & Mamajek for the **photometric** step
   (M_G, G−RP, BP−RP), which Eker does not provide. Remember R_Rsun in the Mamajek table is
   back-computed from Mbol and Teff and is not an independent constraint.
6. **Compute a** from `a = 4.208 (M_tot/M☉)^{1/3} (P/d)^{2/3} R☉`, then decide eclipses with
   cos i uniform on [0,1] and the §F.6 thresholds — `(R1+R2)/a · (1 + e sin ω)/(1 − e²)` for
   the primary eclipse specifically, or the ω-averaged `(R1+R2)/a · 1/(1−e²)` if you have
   marginalised over ω.
7. **Set vsini = 50.593 R/P** for P ≲ 10–20 d (synchronised), and draw from a rotation-period
   distribution (McQuillan et al. 2014) above that. In an eclipsing or synchronised binary
   i ≈ 90°, so vsini ≈ v_eq — no ⟨sin i⟩ = π/4 factor.
8. **Light ratio**: convert the component absolute magnitudes to a G-band flux ratio, and
   sanity-check the result against the measured `EclipsingBinary` distribution (mean 0.318,
   sd 0.250, §G.2). If it comes out systematically more extreme than that, the mass-ratio
   draw is probably too bottom-heavy.

This arm is the one that can produce "large numbers" as the brief asks; the catalogue arms
cap out in the hundreds-to-thousands.

## Concrete recommendation

1. **Eclipsing arm**: draw from **DEBCat**, take e from Torres/SBX by name or from §F, get G
   via the **SIMBAD TAP `ident` route** (§C.5, verified). Every simulated system then has
   M, R, Teff, logg, P, e and a real G.
2. **Non-eclipsing arm**: draw from **SBX** filtered to `K1 IS NOT NULL AND K2 IS NOT NULL
   AND grade >= 4` (513 systems), converting K1/K2 to a mass ratio and using `systems.mag1`/
   `mag2` and the Gaia alias for photometry.
3. **Gaia-realism arm**: draw the *observing* parameters — magnitude, transit count, epoch
   sampling — from the Gaia SB2 sample itself, remembering that **`grvs_mag` and
   `rv_nb_transits` are NULL for every Gaia SB2**, so magnitudes must come from
   `phot_g_mean_mag` and the transit count from `nss_two_body_orbit.rv_n_good_obs_primary`
   (which *is* populated) rather than from `gaia_source`.
4. **Light ratios**: do not take them from the Gaia SB2 sample (they do not exist). Use the
   `EclipsingBinary` `g_luminosity_ratio` distribution (mean 0.318, sd 0.250) for a G-band
   scalar, or GALAH's per-band `ratio1-4` if you want wavelength dependence — which for a
   disentangling benchmark you probably do.
5. **Scale expectation**: the number of real systems with a *complete* independently-measured
   truth (orbit + inclination + light ratio + component parameters) is **a few hundred**, not
   thousands. The large catalogues are for distribution shapes; the small ones are for
   end-to-end truth.

---

# Facts I could not verify

1. **DEBCat's exact row count.** The page states no total; an automated count of `debs.dat`
   returned 846, which conflicts with the ~370 systems reported for March 2026. The header
   and format are verified; **count the file yourself after download.**
2. **The SB2-specific magnitude limit in Gaia DR3.** The SB1 chain's G_RVS ≤ 12 is verified
   verbatim; the SB2 chain (Damerdji et al.) was a separate paper still in preparation when
   the SB1 paper appeared, and I could not retrieve it. The empirical G range (3.55–12.21) is
   verified and is the safer thing to use.
3. **The exact ADS bibcode for Merle et al. 2026 (SBX).** The site renders it as
   `2026MNRAS.547ag351M`, which is malformed. A secondary source gave
   `2026MNRAS.547.3351M` — **treat both as unverified** and resolve via ADS.
4. **Whether Gaia DR4's 5,901,111 figure is `nss_two_body_orbit` alone.** The same number was
   reported against both the astrometric-binaries line and the two-body-orbits line on the
   ESA content page. Recheck at release.
5. **Any DR4-specific SB2 count.** Not published.
6. **A newer Eker et al. (2024/2025) compilation.** Eker 2018 is still cited as current and
   no VizieR table for a successor was found. May exist; I could not confirm one.
7. **The Villanova Kepler EB export URLs.** The site is up and offers CSV/TSV/space export,
   but the page is JS-driven and the hrefs could not be extracted programmatically.
8. **The live TESS EB catalogue's true row count.** `tessebs.villanova.edu` implies roughly
   2,300 rows against VizieR's 4,584; the default view is evidently filtered or grouped, and
   no stated total or export link was found.
9. **VizieR note bodies.** The numbered Note blocks referenced by column descriptions
   (TESS notes 1/3/4, Kirk notes 1/2, Kounkel notes) live only in the Anubis-blocked ReadMe
   files. Byte offsets and `Lrecl` are likewise unavailable, so **fixed-width parsing of the
   raw `.dat` files cannot be specified** — use the TSV or VOTable exports.
10. **The Valenti & Fischer 2005 vsini summary statistics** (median 2.4 km/s, max 54.7, only
    44 stars > 11 km/s) are widely repeated but could not be traced to the paper itself. The
    0.5 km/s per-spectrum precision *is* verified from the abstract. Do not quote the median
    unchecked.
11. **The Pecaut & Mamajek row count** (~220 rows, O3V–Y4V). The columns, version and
    example rows are verified; the total was not independently counted.
12. **Kraft 1967's original numbers** (F5 / 1.3 M☉ / 6200 K break; 40–150 km/s above,
    ≲25 below) come from secondary sources; Kraft 1967 itself was not fetched.
13. **Geller et al. 2021 (`2021AJ....161..190G`) and Narayan et al. 2026 (AJ 171, 102)**
    bibcodes are [L].
14. **Głębocki & Gnaciński 2005 (VizieR III/244) row counts** (28,179 stars / 39,351
    measurements) come from the ADS/CDS listing, not the ReadMe, which is Anubis-blocked.

**Corrected** since the brief was written — worth flagging because the brief stated them as
fact: Gaia's licence is **CC BY-NC** 3.0 IGO, not CC BY-SA (verified this session);
Moe & Di Stefano's Table 13 tabulates **f_logP;q>0.1, not q>0.3**, and there is no published
(M1, logP) grid of the q>0.3 quantity; their §9.1 mass points are 1, 3.5, 7, **12**, 28 M☉,
not 12.5; the (a/R)^{21/2} tidal scaling is the **early-type radiative** case, not the
solar-type convective one, which is (a/R)⁸; the Torres VizieR identifier is
**`J/other/A+ARV/18.67`**, not `J/A+ARv/18/67`; and Kirk et al.'s VizieR copy does **not**
carry the eclipse depths the brief assumed.

# Licence and citation summary

**VizieR** [V], from `https://cds.unistra.fr/vizier-org/licences_vizier.html`: there is no
SPDX-style licence. CDS states the data are "free of usage in a scientific context" provided
the original authors and publications are cited; commercial use depends on each dataset's
origin. The metadata system itself is explicitly *not* openly licensed. Required
acknowledgement, verbatim:

> "This research has made use of the VizieR catalogue access tool, CDS, Strasbourg, France
> (DOI : 10.26093/cds/vizier)."

Per-catalogue bibcodes to cite alongside it:

**Gaia** [V]: **CC BY-NC 3.0 IGO** (Attribution-NonCommercial), per
`https://www.cosmos.esa.int/web/gaia-users/license`. Note the **non-commercial** clause.

| Catalogue | Bibcode | Status |
|---|---|---|
| Gaia DR3 mission | `2016A&A...595A...1P`, `2023A&A...674A...1G` | [L] |
| Gaia DR3 NSS (Arenou) | `2023A&A...674A..34G` | [L] |
| Gaia DR3 EB candidates (Mowlavi) | `2023A&A...674A..16M` | [V] |
| Gaia DR3 binary masses (Creevey) | `2023A&A...678A..19C` | [L] |
| Gaia DR3 RVs (Katz) | `2023A&A...674A...5K` | [L] |
| Gaia DR3 G_RVS photometry (Sartoretti) | `2023A&A...674A...6S` | [L] |
| DEBCat (Southworth) | `2015ASPC..496..164S` | [V] |
| Torres, Andersen & Giménez | `2010A&ARv..18...67T` | [L] |
| Eker et al. 2018 | `2018MNRAS.479.5491E` | [L] |
| ASAS-SN VAC (Rowan) | `2022MNRAS.517.2190R` | [L] |
| Prša et al. 2022 TESS EBs | `2022ApJS..258...16P` | [V] |
| Kirk et al. 2016 Kepler EBs | `2016AJ....151...68K` | [V] |
| SB9 (Pourbaix) | `2004A&A...424..727P` | [V] |
| SBX (Merle et al. 2026) | malformed on site | [U] |
| Kounkel et al. 2021 APOGEE SB2 | `2021AJ....162..184K` | [V] |
| Traven et al. 2020 GALAH | `2020A&A...638A.145T` | [V] |
| Guo et al. 2025 LAMOST SB2 | `2025ApJS..278...46G` | [L] |
| Raghavan et al. 2010 | `2010ApJS..190....1R` | [V] |
| Moe & Di Stefano 2017 | `2017ApJS..230...15M` | [V] |
| Meibom & Mathieu 2005 | `2005ApJ...620..970M` | [V] |
| Meibom, Mathieu & Stassun 2006 (synchronisation) | `2006ApJ...653..621M` | [V] |
| Milliman et al. 2014 (NGC 6819) | `2014AJ....148...38M` | [V] |
| Pecaut & Mamajek 2013 | `2013ApJS..208....9P` | [V] |
| Eker et al. 2018 | `2018MNRAS.479.5491E` | [V] |
| Winn 2010 (transits and occultations) | `2010exop.book...55W` | [V] |
| Kipping 2014 (ω-marginalised eclipse probability) | `2014MNRAS.444.2263K` | [V] |
| Moe & Di Stefano 2013 (eclipse fractions) | `2013ApJ...778...95M` | [V] |
| Nordström et al. 2004 (GCS) | `2004A&A...418..989N` | [V] |
| McQuillan, Mazeh & Aigrain 2014 (rotation periods) | `2014ApJS..211...24M` | [V] |
| Głębocki & Gnaciński 2005 (vsini, VizieR III/244) | `2005yCat.3244....0G` | [L] |
| Zahn 1977 / 1989; Hut 1981 (tidal theory) | `1977A&A....57..383Z`, `1989A&A...220..112Z`, `1981A&A....99..126H` | [V] |

SBX is served by ULB, not CDS; its own terms were not stated on the pages read [U].

# Practical notes for whoever writes the loader

1. **Do not `wget` the CDS FTP tree.** `https://cdsarc.cds.unistra.fr/ftp/<cat>/<file>` is
   behind Anubis and returns an HTML block page to non-browser clients. Use `asu-tsv` or TAP.
2. **Bulk-export URL pattern**, confirmed against several tables:
   `https://cdsarc.cds.unistra.fr/viz-bin/asu-tsv?-source=<TABLE>&-out.all&-out.max=unlimited`
3. **astroquery defaults to 50 rows.** Always `Vizier(columns=["**"], row_limit=-1)`;
   `columns=["**"]` is required or you get only the default column subset.
4. **Time zero points differ across every catalogue**: Gaia `vari_eclipsing_binary`
   `reference_time` is BJD(TCB) − 2455197.5; Gaia `nss_two_body_orbit` `t_periastron` is in
   days on the Gaia reference epoch; TESS `BJD0` is BJD − 2457000; Kepler `BJD0` is
   BJD − 2400000; APOGEE `HJD` is HJD − 2400000; ASAS-SN `t0` is +2456000 UTC; SB9 `T0`,
   Kounkel `T0` and Guo `t0` are full JD. Seven conventions.
5. **`omega` units differ**: degrees in Gaia, SB9 and APOGEE; **radians** in both LAMOST
   catalogues.
6. **Error conventions differ**: Torres `e_Mass`/`e_Rad` are absolute; Eker 2018's are
   **relative**.
7. **Quote awkward labels in ADQL**: `"Wp-pf"`, `"Phip-2g"`, `"[Fe/H]-50"`, `"E(B-V)-mean"`,
   `"(R1+R2)/a"`, `"Teff2/Teff1"`, and SB9's `"e"` and `"o"`.
8. **NULL is not zero** in Gaia: `SB1C`/`SB2C` store circular eccentricity as NULL.
9. **Truncated identifiers**: ASAS-SN's `GaiaDR3` column is scaled by 1e18 and is not a
   usable `source_id`.
