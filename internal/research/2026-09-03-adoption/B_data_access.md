# Data access: which archive loaders albireo should build next

Researched 2026-09-03. Every fact below carries the URL it was checked against. Where a claim could
not be reached at a primary source it is marked **unverified**.

Baseline for the Gaia section is `internal/roadmap.md` item 7, written 2026-08-13. That entry is
**confirmed in substance** and corrected in four specifics; the corrections are marked
**[CORRECTION]** and the genuinely new material **[NEW]**.

### The contract every loader has to fill

Effort below is judged against `RawSpectrum` in
`C:\Users\thari\Documents\GitHub\albireo\src\albireo\io.py` (line 178), which is what a new reader
must populate: `wave` (Å), `flux`, `err` (**`None` is legal** — the FEROS precedent), `bjd`,
`v_bary`, `frame` ∈ {`topocentric`, `barycentric`}, `instrument`, `resolving_power`, `wave_medium` ∈
{`air`, `vacuum`, **`unknown`**}, `continuum_normalized`, `quality`, plus the provenance strings
`time_source`, `v_bary_source`, `err_source`.

Three consequences worth keeping in view while reading the rest:

1. `frame="barycentric"` is already a first-class value, so an archive that ships wavelengths already
   shifted to the barycentre (Gaia, HARPS-style `s1d`) needs no new machinery — only a *rest-frame*
   shift is fatal.
2. `err=None` and `wave_medium="unknown"` are already tolerated, so a missing error array downgrades
   an archive rather than disqualifying it.
3. `io.py` already computes BJD_TDB and BERV from observatory location + sky coordinate + MJD
   (`_observatory_location`, `_sky_coord`, `_barycentric_time`, `_barycentric_velocity`). An archive
   that publishes a site and a mid-exposure time but no BERV is therefore **cheap**, not expensive.

And one about the *query* half. `src/albireo/archive.py` already speaks IVOA TAP, and its own comment
records that "VizieR serves the same IVOA TAP dialect as ESO", which is why BLOeM resolves by name
through VizieR with no new transport code. **The single largest cost driver below is therefore
whether an archive exposes TAP/SSAP or only an HTML form**: the former reuses `archive.query`
wholesale, the latter means scraping, which is code albireo does not have and should be reluctant to
acquire.

---

## 1. Gaia DR4

### 1.1 Release date and status: confirmed, unchanged

| | |
|---|---|
| DR4 release | **Wednesday 2 December 2026**, 66 months of data (25 July 2014 – 20 January 2020) |
| Announced | 2026-05-07, "Announcement of Gaia Data Release 4 date" |
| Pre-release | 2026-06-29 (draft data model + 12-source epoch-astrometry sample) |
| Volume | ~400 TB across all products |
| DR5 | "not before the end of 2030" |

- <https://www.cosmos.esa.int/web/gaia/release> — "Gaia DR4 (based on 66 months of data) 2 December 2026"
- <https://www.cosmos.esa.int/web/gaia/news> — the 2026-05-07 and 2026-06-29 items
- <https://www.cosmos.esa.int/web/gaia/dr4> — the full contents table

**Nothing has slipped since June 2026.** The `gaiadr4` schema is not yet on TAP; verified live:

```
GET https://gea.esac.esa.int/tap-server/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv
    &QUERY=SELECT schema_name FROM tap_schema.schemas
-> public, gaiadr3, gaiafpr, gaiadr2, gaiadr1, tap_schema, tap_upload, tap_config,
   job_upload, gaiaedr3, external
```

No `gaiadr4`. A DR4 DataLink call returns HTTP 500 today. **The format cannot be tested against real
data before 2 December** — the only DR4-shaped file in existence is the 12-source epoch-*astrometry*
pre-release, and that is worth reading anyway (§1.6).

86 DR4 papers are listed, including four that bear directly on this work
(<https://www.cosmos.esa.int/web/gaia/dr4-papers>):

- "Gaia Data Release 4: Epoch RVS spectra and radial velocities"
- "Gaia Data Release 4: Measuring and validating radial velocities of RVS double lined spectra and the related SB2 orbital solutions"
- "Gaia Data Release 4: Spectroscopic characterisation of astrometric binaries by line integral analysis of Gaia RVS spectra"
- "Gaia Data Release 4: Derivation and publication of the RVS data products"

### 1.2 `rvs_epoch_spectrum`: the roadmap's table, verified line by line

Source: the draft DR4 data model PDF, downloaded and read in full (54,344 lines of extracted text),
`gaia-dr4-prerelease-draft-data-model_2026-06-26.zip` from
<https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/>, section 17.3.

| Roadmap claim | Status |
|---|---|
| Table `rvs_epoch_spectrum`, DataLink only, not on main TAP | **confirmed** — "Note this table is not available through the main archive TAP interface" |
| Retrieval type `EPOCH_SPECTRUM_RVS` | **confirmed**, and it is in astroquery's DR4 vocabulary (§1.5) |
| 961 elements, 846–870 nm, step 0.025 nm | **confirmed**, with the explicit formula `wave[i] = 846.0 + i * 0.025 nm (i=0..960)` |
| Barycentric frame, normalised | **confirmed** verbatim, but see the normalisation trap below |
| `obs_time_rv`, BJD in TCB − 2 455 197.5 d, Roemer-corrected | **confirmed** |
| `flux_error[961]`, NaN where all CCDs masked | **confirmed** |
| `combined_ccd_in_index` per-pixel coverage | **confirmed but mis-shaped in the roadmap** — see [CORRECTION 1] |
| Row/volume counts 6,910,785,949 / 49 TB | **confirmed** on the contents page |

**[CORRECTION 1] `combined_ccd_in_index` is sparse, not a 961-array.** It is a `short[]` paired with a
second `short[]` column named `index`: "This field contains the indexes of the flux array where the
number of combined CCDs is strictly smaller than `combined_ccds`." Both are null when every bin used
all CCDs. A loader that reads `combined_ccd_in_index` as a per-pixel array of length 961 will be
wrong; it must scatter `combined_ccd_in_index` into `index`.

**[CORRECTION 2] The `has_epoch_rvs` selection flag lives in `all_source_flags`, not
`all_source_rvs`.** The column is *defined* in section 2.2 `all_source_flags [TAP]`. The draft's own
prose is internally inconsistent: §17.3 says "joining on source_id with the table
`all_source_flags`", while §17.1 (`rvs_epoch_parameters_single`) says "the table `all_source_rvs`".
Take the definition site, and re-check on release day.

Both tables are real and they are different sizes, so this is not a cosmetic distinction:
`all_source_flags` has **2,793,009,568** rows (one per processed source) and `all_source_rvs` has
**312,247,580** (one per RVS-processed source). The SB2 selection needs **both** — the flag from one,
the double-lined fraction and brightness from the other:

```sql
SELECT s.source_id
FROM gaiadr4.all_source_flags AS f
JOIN gaiadr4.all_source_rvs   AS s USING (source_id)
WHERE f.has_epoch_rvs = 3              -- moderate/strong; epoch RVs AND vbroad exist
  AND s.rv_assumed_sb2 = 'true'        -- double lines in >= 10 transits
  AND s.external_apparent_grvs <= 12   -- the regime where double-line detection ran at all
```

(Schema prefix and the boolean literal are guesses at DR4 conventions; the column and table names are
from the draft data model. Verify on 2 December.)

**[CORRECTION 3] The `has_epoch_rvs` byte grading is not "0 = none, 1 = very weak" tied to spectrum
S/N.** It is tied to `external_apparent_grvs`:

- 0 — no spectrum (RVS pipeline did not process, or excluded)
- 1 — very weak to extremely weak, `external_apparent_grvs > 14`; **"these spectra cannot be used independently"**, and no epoch RVs were computed
- 2 — weak, `12 < external_apparent_grvs <= 14`; epoch RVs computed
- 3 — moderate to strong, `external_apparent_grvs <= 12`; epoch RVs *and* broadening velocities

Grvs = 12 is the hinge of the whole product: it gates epoch RVs, broadening, double-line detection,
*and* "the type of normalisation applied to the CCD spectra before combination". **albireo's useful
population is `has_epoch_rvs = 3`.**

**[NEW — the biggest single find] Double-lined transits are not normalised at all.** From
`normalisation_method` (byte):

- **0: Not normalised. "The combined spectra where double-lines are detected are not normalised."**
- 3: Degraded — flux *and* flux_error both divided by the median of the combined spectrum
- 4: Median negative flux — not normalised, an indicator of poor quality

So the exact spectra albireo wants arrive **un-normalised**, while the reference population arrives
divided by a median. The loader must branch on `normalisation_method` and must not assume a
pseudo-continuum near 1. On the positive side, method 3 scales flux and error by the *same scalar*,
so the inverse-variance ratio survives; method 0 leaves the data untouched, which is what
disentangling actually prefers.

**[NEW] Other columns worth having:** `combined_ccds` (1, 2 or 3 — the effective exposure of the
transit), `deblended_ccds` (how many contributing CCD spectra had a blend deconvolved — a quality
axis albireo has no analogue for), `transit_id` (decodable to field-of-view, CCD row and AC pixel via
<https://gaia.esac.esa.int/decoder/transitidDecoder.jsp>).

**The D4 conflict stands and is stated correctly in the roadmap.** "The fluxes and squared
uncertainties of the normalized CCD spectra are **linearly interpolated** onto a fixed wavelength
grid" — Gaia resamples upstream, albireo cannot undo it, and the published `flux_error` therefore
understates the correlation. One refinement: the interpolation is over only 1–3 CCD samples per bin,
not a long resampling chain, so the induced correlation is short-range — which is exactly the regime
the AR(1) machinery (D34) was built for.

### 1.3 [NEW] The DR4 epoch grid is undersampled against albireo's own pixel-locking rule

This is the finding with the most direct engineering consequence, and neither the roadmap nor the
memory records it.

The DR4 data model states the grid (0.025 nm) but never states the resolving power, and the abstracts
of the RVS instrument papers do not carry it either. So I measured it from the published spectrum of
DR3 source `249375804190613120` (grid step verified at 10.01 pm, 2398 finite bins). Taking the
narrowest resolved absorption features and their FWHM against a local continuum:

| FWHM | λ | implied R |
|---|---|---|
| 70 pm | 863.16 nm | 12,300 |
| 80 pm | 868.58 nm | 10,900 |
| 90 pm | 868.26 nm | 9,600 |

The narrowest features bracket **R ≈ 11,000–12,300**, which independently confirms the canonical
R ≈ 11,500 the roadmap already assumes. Take FWHM ≈ 0.075 nm at 858 nm, so the LSF Gaussian
σ ≈ 0.032 nm. Then:

| Product | Grid step | **bins per LSF σ** |
|---|---|---|
| DR3 `rvs_mean_spectrum` | 0.010 nm | 3.2 |
| **DR4 `rvs_epoch_spectrum`** | **0.025 nm** | **1.3** |

**DR4 deliberately coarsens the sampling by 2.5× relative to DR3**, landing at roughly three bins per
FWHM. Set that against the rule measured in albireo's own TODCOR work — pixel-locking is controlled
only at **≥ 3 px/σ** — and the DR4 epoch grid sits a factor of ~2.4 *below* the threshold. Two
consequences to plan for rather than discover:

1. **TODCOR on Gaia DR4 epoch spectra should be expected to pixel-lock.** The measured
   pixel-locking table applies directly and predicts it. Either the loader documents this, or
   `albireo.todcor` grows a warning keyed on samples-per-σ.
2. Disentangling itself runs near Nyquist. That is not fatal — it is the regime the fixed-grid
   forward model was built for — but combined with the upstream linear interpolation (§1.2) it means
   the *effective* independent-pixel count is well below 961.

### 1.4 [NEW] Air vs vacuum: settled empirically, because the data model never says

The draft data model does not declare a medium for `rvs_epoch_spectrum`. It does say "Wavelength in
vacuum" for the RVS-band DIB and stacked-ISM products, which is suggestive but not the same product.

I settled it by measuring a real published Gaia RVS spectrum. Downloaded DR3 source
`249375804190613120` (`rvs_spec_sig_to_noise` = 699) as CSV and located the Ca II triplet:

| Line | Deepest bin | vs air | vs vacuum |
|---|---|---|---|
| Ca II 8498 | 850.040 nm | +84 km/s | **+1 km/s** |
| Ca II 8542 | 854.450 nm | +85 km/s | **+2 km/s** |

**Gaia RVS wavelengths are in vacuum**, to within 2 km/s. (The 8662 window is blended by a
neighbouring feature at 866.75 nm and is not a clean test; its absorption centroid still sits far
closer to vacuum than to air.) The loader should declare `vacuum` and not guess.

The same measurement independently **confirms the roadmap's DR3 finding**: the star's Ca II lines sit
at ~+1 km/s, i.e. the radial velocity really has been divided out of the DR3 mean spectrum. DR3
`rvs_mean_spectrum` should be refused, as the roadmap says.

### 1.5 Access: astroquery and stdlib, both verified against the live service

**astroquery** — checked against the source, not the rendered docs
(<https://raw.githubusercontent.com/astropy/astroquery/main/astroquery/gaia/core.py> and
`astroquery/gaia/__init__.py`):

`Gaia.load_data` supports `retrieval_type='EPOCH_SPECTRUM_RVS'`. The DR4 vocabulary in
`conf.VALID_DATALINK_RETRIEVAL_TYPES` also contains `EPOCH_PARAMETERS_RVS_SINGLE`,
`EPOCH_PARAMETERS_RVS_DOUBLE`, `MEAN_SPECTRUM_RVS`, `EPOCH_FLAGS_NSS`. Signature:

```python
load_data(ids, *, data_release=None, data_structure='DATAMODEL_STANDARD',
          retrieval_type="ALL", linking_parameter='SOURCE_ID', valid_data=False,
          format="votable", dump_to_file=False, ...)
```

- `data_structure`: `DATAMODEL_STANDARD` (one file per source_id, all zipped) or **`DATAMODEL_GAIA`**
  ("parameters stored as arrays will remain as such", one row per sourceId). **For epoch spectra
  albireo wants `DATAMODEL_GAIA`** — `DATAMODEL_STANDARD` explodes each spectrum into one row per
  wavelength bin (verified, §1.5).
- `linking_parameter`: `SOURCE_ID`, **`TRANSIT_ID`**, `IMAGE_ID`. Fetching by transit is supported.
- `format`: `votable`, `csv`, `ecsv`, `votable_plain`, `json`, `fits`.
- Renamed in **astroquery 0.4.11 (2025-09-19)**; DR4 types first added in 0.4.9 (2025-01-24).
  <https://astroquery.readthedocs.io/en/latest/changelog.html>
- **0.4.11 is the current PyPI release** (uploaded 2025-09-20; checked
  <https://pypi.org/pypi/astroquery/json>), so `EPOCH_SPECTRUM_RVS` is available in a *released*
  astroquery today — no dependency on an unreleased version. albireo needs neither, since the raw
  HTTP call below is three stdlib modules.

**Standard library only: yes, comfortably.** `load_data` is a thin wrapper that builds a plain
GET/POST against `https://gea.esac.esa.int/data-server/data` with
`ID`, `RELEASE`, `RETRIEVAL_TYPE`, `DATA_STRUCTURE`, `FORMAT`, `VALID_DATA`, `USE_ZIP_ALWAYS=true`,
and optionally `LINKING_PARAMETER`. Verified live and anonymously today with DR3:

```
GET https://gea.esac.esa.int/data-server/data
    ?ID=249375804190613120&RETRIEVAL_TYPE=RVS&RELEASE=Gaia+DR3
    &DATA_STRUCTURE=INDIVIDUAL&FORMAT=csv&USE_ZIP_ALWAYS=true
-> HTTP 200, application/zip, 31 kB, no authentication, no token, no cookie
```

`urllib.request` + `zipfile` + `csv` is sufficient. **This matters more than it looks**: the DR4
pre-release VOTable is serialised as **base64 `BINARY2`**, not `TABLEDATA`, so a stdlib VOTable
parser would have to decode BINARY2 with variable-length arrays. Asking for `FORMAT=csv` or
`FORMAT=fits` sidesteps that entirely. (`FORMAT=json` returns 500.)

Also verified: `DATA_STRUCTURE` vocabulary is release-specific. DR3 accepts `INDIVIDUAL`/`RAW` and
**rejects** `DATAMODEL_STANDARD`/`DATAMODEL_GAIA` with a 500; those are the DR4 words. A loader must
not hard-code one vocabulary for both.

**Limits** (<https://www.cosmos.esa.int/web/gaia-users/archive/faq>,
<https://www.cosmos.esa.int/web/gaia-users/archive/datalink-products>):

| | Anonymous | Registered |
|---|---|---|
| **DataLink sources per call** | **5,000** | 5,000 |
| Max rows per query | 3,000,000 | unlimited |
| Sync / async timeout | 60 s / 120 min | 60 s / 120 min |
| Job retention | 3 days | unlimited |
| Disk quota | none | 20 GB |

No published rate limit or throttle. The 5,000-source cap is per call, not per session; sequential
calls are the documented way past it.

### 1.6 [NEW] What the pre-release file tells us about the DR4 wire format

`gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip` is the only DR4-shaped product that exists.
Reading it yields two things a loader author needs:

**A trap.** Its root element is:

```xml
<RESOURCE type="results" utype="spec:Spectrum">
```

on a file containing **epoch astrometry**. The `xmlns:spec="http://www.ivoa.net/xml/SpectrumModel/v1.01"`
declaration and `utype="spec:Spectrum"` are boilerplate stamped on every Gaia DataLink product
regardless of content. **A loader dispatching on `utype="spec:Spectrum"` will happily accept an
astrometry file.** Dispatch must additionally require the spectral-axis field.

This refines the note in memory that "Gaia products carry no IVOA utypes at all". The
`DATAMODEL_STANDARD`/`INDIVIDUAL` serialisation *does* carry them, and usefully:

```xml
<FIELD name="wavelength" ucd="em.wl" unit="nm" utype="spec:Data.SpectralAxis.Value"/>
<FIELD name="flux"       ucd="phot.flux;em.opt.I"/>
<FIELD name="flux_error" ucd="stat.error;phot.flux;em.opt.I"/>
```

plus `spec:Spectrum.Char.SpectralAxis.{Ucd,Unit,Coverage.Bounds.Start/Stop}` and
`spec:Spectrum.Char.TimeAxis.Coverage.Location.Value` as PARAMs. **The spectral axis is utyped; flux
and flux_error are not** — they carry UCDs only. So albireo's utype dispatch binds to the wavelength
column and must fall through to UCD for the flux pair. That is a precise, testable statement about
where D45's principle holds and where the name/UCD table takes over.

**A gift.** The file carries a machine-readable time frame:

```xml
<TIMESYS ID="time_frame" refposition="BARYCENTER" timeorigin="2455197.5" timescale="TCB"/>
```

The loader can read the 2 455 197.5 offset and the TCB scale out of the file instead of hard-coding
them, and can *verify* the barycentric reference position rather than trusting prose.

Row shape in the RAW/Gaia structure: one row per (`source_id`, `transit_id`), with per-CCD quantities
as `arraysize="*"` columns. Expect `rvs_epoch_spectrum` under `DATAMODEL_GAIA` to be one row per
transit with `flux` and `flux_error` as 961-element arrays — the shape a loader wants. The
`DATAMODEL_STANDARD` CSV, by contrast, is one row per wavelength bin with source metadata repeated on
every row (verified on DR3), and nulls appear as **empty CSV fields**, which the loader must map to
NaN rather than 0.

The pre-release `release` PARAM reads `Gaia DR4_RC3` — release candidate 3.

An ESA Python package accompanies it, `gaia-supdate` (<https://github.com/esa/gaia-supdate>), with
tutorials at <https://github.com/esa/gaia-jupyter-notebooks/tree/main/data-release-4-tutorials>.
That directory currently holds **one** notebook, on epoch astrometry. **There is no ESA epoch-RVS
tutorial yet** — a gap albireo could occupy on release day.

### 1.7 SB2 orbital-solution and epoch-RV tables

All counts from <https://www.cosmos.esa.int/web/gaia/dr4> (contents table), column definitions from
the draft data model.

**Epoch radial velocities — `rvs_epoch_parameters_double` [DataLink, `EPOCH_PARAMETERS_RVS_DOUBLE`],
787,312 transits, 128 MB.** This is the SB2 epoch-RV table and it is extraordinarily well matched to
albireo. Per transit it carries:

- `radial_velocity_primary` / `radial_velocity_secondary` + errors + per-component validity flags
- `is_reordered` — the pipeline assigns primary/secondary **independently per transit**, so roles
  swap; NSS processing re-orders them for consistency and sets this flag. *This is exactly the
  component-ordering hazard that bit the pipeline demo (components must be declared in decreasing-mass
  order).* Gaia has the same problem and publishes its answer.
- **`rad_vel_bary_corr`** — the per-epoch barycentric velocity correction in km/s, computed with the
  Gaia Relativity Model, error ~1 cm/s, "a positive value corresponds to a redshift". This is the
  BERV albireo asks for. Note it lives **here**, not in `rvs_epoch_spectrum`; the spectra are already
  shifted, so a loader that wants to *report* the correction must fetch a second DataLink product.
- `obs_time_rv` and `obs_time_bary_corr` (Roemer correction, ns)
- `template_teff/logg/fe_h` for **primary and secondary separately** — the labels Gaia's own pipeline
  matched. Direct input to albireo's label-matching mode (D52/D53).
- `vbroad_primary` / `vbroad_secondary` — per-component line broadening.
- `grvs_flux`, `grvs_mag`, `grvs_zero_point` for the system.

Selection: `all_source_rvs.fraction_double_lined > 0`.

**Orbital solutions — `nss_two_body_orbit` [TAP], 5,901,111 rows.** Solution types include **`SB2`,
`SB2ScanAngleK1`, `SB2ScanAngleK12`, `AstroSpectroSB2`, `EclipsingSpectroSB2`** alongside the SB1 and
astrometric families. Columns present:

`period`, `period_error`, `t_periastron`, `t_periastron_error`, `eccentricity`, `arg_periastron`,
`inclination`, `pos_ascending_node`, **`center_of_mass_velocity`** (γ), **`semi_amplitude_primary`**
(K1), **`semi_amplitude_secondary`** (K2), `mass_ratio`, `esinw`/`ecosw`, `temperature_primary`,
`temperature_secondary`, `temperature_ratio`, **`g_luminosity_ratio`** (G-band secondary/primary — a
prior on the light ratio, which is the parameter that makes or breaks albireo's joint dilution fit),
`fill_factor_primary/secondary`, `eclipse_time_secondary`, `rv_n_good_obs_primary/secondary`,
`sb2_orth_corr_coeff`, `bic`, `significance`, `flags`, and the packed `correlations` vector with
`bit_index` telling you which parameters were fitted.

Caveat from the data model: "a source may receive independent astrometric, spectroscopic or
photometric orbits, so a query using a given source_id may return **several** solutions."

**`rvs_two_body_orbit` [TAP], 638,966 rows** — SB1-only, from Line Integral Analysis applied to
sources already flagged as *astrometric* binaries, reusing that preliminary solution's `period`,
`t_periastron`, `eccentricity`, `inclination`, `arg_periastron` and fitting only
`center_of_mass_velocity` and `semi_amplitude_primary`. Useful as a target list, not as SB2 truth.

Other counts: `rvs_epoch_parameters_single` 6,910,423,894 transits / 323 GB;
`rvs_transit_description` 9,020,054,046 / 175 GB; `rvs_mean_spectrum` 250,022,127 / 1.8 TB;
`nss_non_linear_spectro` 79,653; `nss_masses` 4,323,300; `nss_multiple_orbits` 16,681.

**How many SB2 targets, really?** The draft data model does not publish the number of `rv_assumed_sb2`
sources. Two published facts bound it:

- A source is flagged only if double lines are detected in **≥ 10 transits**, and there are 787,312
  double-lined transits in total. **Upper bound: ~78,700 flagged sources**, and realistically several
  times fewer, since flagged systems will contribute tens of transits each. *(This is my arithmetic
  on published numbers, not a published figure.)*
- "**Only about one third of these assumed SB2 systems are subsequently confirmed as SB2 by the NSS
  processing.**"

Detection runs only for `external_apparent_grvs ≤ 12`, so the sample is bright.

The roadmap's core insight survives intact and is worth restating in the data model's own words:
epoch spectra "are still produced for detected double-lined, emission-line, and contaminated
transits" — the transits whose RV the pipeline throws away. When `rv_assumed_sb2` is true "the source
is excluded from the multi-transit analysis" and "the corresponding epoch spectra are available in
`rvs_epoch_spectrum`". The spectra exist precisely where Gaia gives up.

### 1.8 Forecasting transits before release: GOST works, anonymously, today

**GOST has a documented and working REST interface**, contrary to what its landing page suggests.
Manual: <https://gaia.esac.esa.int/gost/docs/gost_software_user_manual.pdf> (§4.4, §5).

```
GET https://gaia.esac.esa.int/gost/GostServlet?ra=<deg>&dec=<deg>
-> HTTP 200, application/xhtml+xml, XML; verified live, 1.37 MB for one position, no auth
```

Each `<event>` carries `<kind>` (`AF` or `SM`), `<ccdRow>`, `<fov>`, `<scanAngle>`,
`<parallaxFactorAl/Ac>`, `<eventTcbDate>`, `<eventUtcDate>` and — decisively —
**`<eventTcbBarycentricJulianDateAtBarycentre>`**, which is the same quantity and convention as
`obs_time_rv` (BJD, TCB, barycentre). The forecast window returned today spans 2014-07-25 to
2025-07-01 in TCB, which fully covers DR4's 2014-07-25 → 2020-01-20.

Two caveats:

1. **GOST emits `AF` and `SM` events, not RVS events.** RVS occupies **CCD rows 4–7** (stated in the
   `transit_id` decoder in the DR4 data model: "specifically for RVS in the range 4 to 7"), so RVS
   transits must be inferred by filtering `ccdRow` ∈ 4..7 and requiring the source to be bright
   enough (`Grvs ≲ 14`, and ≲ 12 for anything albireo can use).
2. GOST's own disclaimer: it "does not take into account operational activities preventing nominal
   observations nor matters like, e.g., the gaps between CCDs", with roughly **80 %** probability that
   data was actually taken at a predicted time.

Name resolution is `?name=<target>&service=<code>&resolve` (V/S/N for VizieR/SIMBAD/NED). A bare
`?name=m31` returned HTML in my test, so use coordinates for scripted work.

Two further endpoints: a Swagger REST API at `https://gaia.esac.esa.int/gost/swagger-ui/dist/index.html`
which the manual warns "could be restricted to connections from ESAC", and an IVOA ObjVisSAP service
at `https://gaia.esac.esa.int/gost/ObjVisSAP/gaiaobjvisap` (parameters `s_ra`, `s_dec`, `t_min`,
`t_max`, `MAXREC`, `RESPONSEFORMAT`) which returns *visibility intervals*, not transit times, and so
is the wrong tool here.

**This is actionable now**: albireo can build and test the epoch-forecasting path against GOST before
December, so that on release day only the DataLink read is new.

### 1.9 Workshops and events

No DR4 workshop or hackathon is announced on the Gaia news pages as of 2026-09-03
(<https://www.cosmos.esa.int/web/gaia/news>, <https://www.cosmos.esa.int/web/gaia/news-2026>;
`/web/gaia/events` returns 404). The only DR4-adjacent community event is **"From Gaia to GaiaNIR"**,
Athens, **11–15 January 2027**, announced 2026-05-26 — five weeks after the release, and explicitly
about DR3/DR4 science. That is the natural first venue for a DR4-capable albireo.

### 1.10 Verdict

| | |
|---|---|
| **User base** | Order 10⁴ SB2 candidates with public epoch spectra (≤ ~78,700 hard upper bound; ~⅓ NSS-confirmed), plus 787,312 double-lined transits with published two-component RVs, plus millions of SB1/astrometric orbits. Bright (`Grvs ≤ 12`) and mostly cool. |
| **Effort** | **Low-to-moderate.** Anonymous HTTP, no auth, stdlib-sufficient with `FORMAT=csv`/`fits`. The whole grid is a constant. The work is in quality flags, not parsing. |
| **Quirks** | Upstream linear interpolation onto the fixed grid (violates D4; short-range, AR(1)-shaped). Double-lined spectra **un-normalised** while others are median-scaled. `combined_ccd_in_index` is sparse. `utype="spec:Spectrum"` is stamped on non-spectra. Vacuum. `has_epoch_rvs` lives in `all_source_flags`. Narrow 846–870 nm window with one species for hot stars. |
| **Priority** | **1 by value, but third in build order — the read is blocked until 2 December.** The only dataset here that multiplies albireo's addressable population by orders of magnitude, on a known date, with a ready-made SB2 target list and per-component RVs, templates and vbroad supplied. The two halves that *can* be built now are the GOST forecasting path and a DataLink client tested against DR3. See §6.3. |

---

## 2. SDSS / APOGEE (and BOSS)

**Headline: the only APOGEE product fit for disentangling is the SDSS-IV DR17 `apVisit`.** SDSS-V does
not publish visit spectra in that form. Its visit product, `mwmVisit`, is resampled *and* shifted to
the stellar rest frame, and the SDSS-V `apVisit` tree on the Science Archive Server contains only plots
and logs.

### 2.1 Release status

| Release | Public | Paper |
|---|---|---|
| DR17 (SDSS-IV) | **2021-12-06** | ApJS 259, 35 |
| DR18 | 2023-01-19 | ApJS 267, 44 |
| DR19 (SDSS-V) | **2025-07-10** | ApJS 285, 9; arXiv:2507.07093 |
| **DR20** (SDSS-V) | **2026-07-30** | arXiv:2607.26149 |

<https://www.sdss.org/science/publications/data-release-publications/>,
<https://www.sdss.org/sdss-launches-twentieth-release/>

DR19 is the first release with all three mappers and carries "APO spectra from both APOGEE-N and
BOSS-N through July 5, 2023" (<https://www.sdss.org/dr19/mwm/>). **DR20 is a no-op for APOGEE**: "In
DR20, there are no new APOGEE spectra and no new analysis of previous APOGEE spectra. We have copied
the DR19 data into the DR20 files" (<https://www.sdss.org/dr20/>). No DR21 announcement found
(*unverified* whether one is scheduled).

The traditional products are gone in SDSS-V by design: "All combined spectra (BOSS and APOGEE) are
stored in `mwmStar` files, and **rest-frame** visit spectra are stored in `mwmVisit` files"
(<https://www.sdss.org/dr19/mwm/data/dr17-to-dr19/>). Verified by crawling the SAS: every leaf under
`dr19/spectro/apogee/redux/1.3/visit/…` holds only a `plots/` directory, and `stars/` leaves hold only
`.log`/`.err`/`.sha1sum`. The **DR17 tree is mirrored intact** under DR19 and DR20 and does contain the
FITS.

DR17 scale: 657,000 unique APOGEE targets, **2,660,000 individual visits**, 1.5–1.7 µm, R ≈ 22,500
(<https://www.sdss4.org/dr17/irspec/dr_synopsis/>).

### 2.2 `apVisit`: eleven HDUs, observed frame, BC supplied

<https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/visit/TELESCOPE/FIELD/PLATE_ID/MJD5/apVisit.html>

HDU0 header; **1** flux, **2** error (σ, not ivar), **3** mask, **4** wavelength (float64, Å), 5/6 sky
+ error, 7/8 telluric + error, 9 wavelength coefficients (14×3), **10 LSF coefficients (27×3)**. Every
image HDU is **4096 × 3** — 4096 pixels for each of three detectors.

**The wavelength solution is native and irregular** — "the native wavelength scale is not an evenly
spaced linear or logarithmic scale. The wavelength information is included as a separate wavelength
array" (<https://www.sdss4.org/dr17/irspec/spectra/>). HDU4 is a per-pixel array, so there is no
common-grid rebinning.

**But it is already once-resampled, and SDSS says so.** The extracted `ap1D` frames are 2048 pixels
per chip; `apVisit` is 4096. The doubling is the dither combination, and the documentation is candid
about the consequence:

> "there may be correlated errors between pixels and can occur even in visit spectra because these are
> the combination of two dithered observations" … "any pixel in the combined well-sampled spectrum
> will have contributions from multiple raw pixels" … "**this propagation ignores the correlation of
> uncertainties**" — <https://www.sdss4.org/dr17/irspec/spectra/>

**This is the cleanest documented case in this report of an archive telling you its own error array is
a lie.** The correlation is short-range (~1–2 px), which is again AR(1)-shaped. There is also a stated
uncertainty floor of ~0.5%, capping effective S/N near 200.

**Three chips, with real gaps**: a 1.647–1.696, b 1.585–1.644, c 1.514–1.581 µm, stored as three rows,
in *reverse* wavelength order, with per-fibre variation in the endpoints. The loader must emit three
segments, not one array — the same machinery an echelle order list needs. *Unverified*: which row is
which chip and whether each row runs decreasing; sort empirically.

**Vacuum, with no keyword to say so.** "the wavelength calibration of the APOGEE data is done using
vacuum wavelengths" — it must be hardcoded from documentation, which is precisely the situation
`wave_medium` exists to record honestly.

**Timing and barycentric keywords in HDU0**, with the crucial property that **BC is supplied but not
applied**:

```
BC      = -8.92126133253    / barycentric correction (km/s)
VHELIO  = -4.11189510634    / heliocentric velocity (km/s)
VRAD    =  4.80937          / doppler shift (km/s)
JD-MID  =  2455813.69734    / JD at midpoint of visit
UT-MID  = '2011-09-09T04:44:10.2'
EXPTIME =  3386.35          / Total visit exptime
NPAIRS  = '6'               / Number of dither pairs combined
SNR     =  22.5580          / median S/N, middle chip
```

The wavelength solution stays in the observed frame — exactly what albireo wants. `VRAD`/`VHELIO` are
the pipeline's single-template cross-correlation velocities and are meaningless for an SB2; use them
as a starting guess or ignore them.

`allVisit` indexes it all: `APOGEE_ID`, `FILE`, `FIELD`, `PLATE`, `MJD`, `FIBERID`, `JD`, **`BC`**,
`VHELIO`, `VREL`, `VRELERR`, `SNR`, `STARFLAG`, `DATEOBS` (note: no hyphen)
(<https://data.sdss.org/datamodel/files/APOGEE_ASPCAP/APRED_VERS/ASPCAP_VERS/allVisit.html>).

**Flux is sky-subtracted and telluric-corrected**, both before the dither combination, with the applied
models retained in HDU5–8 — so the correction is inspectable and the affected pixels are bit-flagged.
`APOGEE_PIXMASK` (<https://www.sdss4.org/dr17/algorithms/bitmasks/>): bits 0–5 and 14 (BADPIX, CRPIX,
SATPIX, UNFIXABLE, BADDARK, BADFLAT, NOT_ENOUGH_PSF) should drop; **bits 7, 8, 12, 13** (NOSKY,
LITTROW_GHOST, SIG_SKYLINE, SIG_TELLURIC) should drop or down-weight, because sky and telluric
residuals sit exactly where disentangling manufactures a spurious constant component; 9–11
(persistence) warn. Star-level `APOGEE_STARFLAG` bit 17 `SUSPECT_BROAD_LINES` is a free SB2
pre-filter.

**The LSF is the one expensive piece.** HDU10 holds 27×3 *coefficients*, not an LSF array, and there is
no SDSS DR17 documentation page for the parametrisation (`/dr17/irspec/lsf/` 404s). Full `apLSF`
calibration products are public at <https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/cal/lsf/>
(~56–61 MB per chip). Matching resolution means reverse-engineering APOGEE's LSF code, or ignoring the
LSF and letting vsini absorb it — which is exactly the failure mode D55 already characterised on
AI Phe.

### 2.3 `apStar` and `mwmVisit`: refuse both

`apStar` is **(8575, 2+NVISITS)** on a log-linear grid: `CRVAL1 = 4.179`, `CDELT1 = 6e-6`
(≈4.145 km/s/px, "roughly 3 samples per resolution element"), 15100.8–16999.8 Å, vacuum. The visits
"have been resampled to a common logarithmically-spaced wavelength scale, **with the radial velocity of
each individual visit spectrum removed**", by sinc interpolation
(<https://www.sdss4.org/dr17/irspec/spectral_combination/>). Per-visit `BC1`, `VRAD1`, `JD1`, `HJD1`
record what was divided out, so the shift is nominally reversible — but the sinc interpolation is not,
and for an SB2 the removed velocity is a blend-weighted artefact. **Same verdict as Gaia DR3
`rvs_mean_spectrum`: refuse, do not warn.**

`mwmVisit` (DR19/DR20) is worse: "the `mwmVisit` spectra are stored in vacuum wavelengths **in the
stellar rest-frame**" (<https://www.sdss.org/dr19/mwm/astra/output-files/>), and SDSS-IV stars are
repackaged by copying "the data arrays (flux, inverse variance) … exactly from the SDSS-IV `apStar`
file", inheriting the resampling. Since this is the SDSS-V default, **the loader will meet it
constantly and must explain why it refuses.**

### 2.4 Access: anonymous HTTP, stable paths, no API

No authentication for any public DR (the SDSS-V collaboration tree `data.sdss5.org/sas/sdsswork/`
returns HTTP 401; the public `sas/` listing is open). Verified concrete file:

```
https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/visit/apo25m/000+02/5815/56396/
    apVisit-dr17-5815-56396-001.fits          (495,360 bytes)
```

Pattern `…/visit/{apo25m|lco25m}/{FIELD}/{PLATE}/{MJD}/apVisit-dr17-{PLATE}-{MJD}-{FIBER:03d}.fits`
(`asVisit-` at LCO). Summary file
`…/aspcap/dr17/synspec_rev1/allVisit-dr17-synspec_rev1.fits` (2.8 GB).

**There is no APOGEE support in `astroquery.sdss`** — its API is optical only (`get_spectra`,
`query_specobj`, …). The official tool is `sdss_access`
(<https://sdss-access.readthedocs.io/en/latest/>), and SDSS's own recommendation is plain `wget`. **So
the loader path is: read `allVisit`, filter, build the path from `FIELD`/`PLATE`/`MJD`/`FIBERID` (or
use the `FILE` column), fetch over plain HTTPS.** Three stdlib modules again.

### 2.5 SB2 catalogues

**Kounkel et al. 2021, AJ 162, 184** (arXiv:2107.10860) — "Double-lined Spectroscopic Binaries in the
APOGEE DR16 and DR17 Data": **7,273 candidate SB2s, 813 SB3s, 19 SB4s**, SB2 fraction ~3% among MS
dwarfs, and **325 well-fitted orbital systems** among those with ≥4 epochs. Table 3, "Vetted Radial
Velocities of the Individual Components" (APOGEE ID, heliocentric JD, v₁…v₄), is **a ready-made
per-epoch component-RV truth set** for validating albireo against apVisit spectra. VizieR
**`J/AJ/162/184`** (<https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/162/184>); also an SDSS VAC at
<https://data.sdss.org/sas/dr17/env/APOGEE_SB2/apogee_sb2s-v1_0.fits> (8.6 MB), carried forward into
DR19 and DR20. (Per-table row counts *unverified* — CDS returned Access Denied.)

**El-Badry et al. 2018, MNRAS 476, 528** (arXiv:1711.08793) — APOGEE DR13, ~20,000 MS targets giving
**~2,500 binaries**, ~200 triples, ~700 velocity-variable systems, **64 full orbital solutions**.
Catalogues in the MNRAS Supporting Information; code at <https://github.com/kareemelbadry/binspec>.
VizieR `J/MNRAS/476/528` *unverified* (CDS 500).

**SDSS-V update — Saad & Ting 2026**, arXiv:2608.10866: **41,466 SB2 candidates** from 238,205 DR19
dwarfs, "about fifteen times the 2,645 identified in DR13", with **8,981 orbit-ready systems**. But the
authors are explicit: "The 8.1% control false-positive rate implies that close to 40% are single stars,
so we release the sample as a candidate list." And it is built on `mwmStar`, i.e. rest-frame — **so this
sample is discoverable but its spectra are not reusable in native form.**

Adjacent and useful: **The Joker** VAC gives Keplerian posterior samplings for **358,350 DR17 sources
with ≥3 RV measurements** (<https://data.sdss.org/sas/dr17/env/APOGEE_THEJOKER/v1.2/>).

### 2.6 BOSS epoch spectra: they exist, and they are the wrong resolution

SDSS-V produces daily, epoch, and allepoch coadds, and **`specFull` carries the individual exposures
inside the same file**: HDU0 primary, 1 COADD, 2 SPALL, 3 ZALL, 4 ZLINE, **5…N `MJD_EXP_<MJD>-<NN>`**,
each with `FLUX`, `LOGLAM`, **`IVAR`**, `AND_MASK`, `OR_MASK`, `WDISP`, `SKY`
(<https://www.sdss.org/dr19/bhm/data/spectra/>). Finer epoch granularity than APOGEE offers.

Two disqualifying quirks, though. The grid is log-linear at `Δlog₁₀λ = 0.0001`, i.e. **~69 km/s per
pixel**, and — importantly for double-correction bugs — "**The wavelengths are shifted such that
measured velocities will be relative to the solar system barycentric at the mid-point of each
exposure**", with the value recorded in `HELIO_RV` / `V_RAD`. So BOSS ships already-barycentric and a
loader must not apply BERV twice. At R ≈ 2,000 and 69 km/s/px, BOSS can only disentangle the widest
systems. **Multi-epoch BOSS spectra of SB2s exist as a product but not as a curated sample** — the
DR19/DR20 VAC list has no BOSS SB2 catalogue, only a DA white-dwarf RV-variability binary VAC.

### 2.7 Verdict

| | |
|---|---|
| **User base** | 657,000 stars / **2.66 M public `apVisit` spectra**; 358,350 with ≥3 RV epochs. Curated: **7,273 SB2 + 813 SB3 + 19 SB4** with per-epoch component RVs and 325 orbits, plus ~2,500 El-Badry binaries. A further 41,466 DR19 candidates exist but with rest-frame spectra only. Caveat: H-band APOGEE targets cool giants and dwarfs, **not** the hot massive binaries of the HR 6819 / BLOeM line. |
| **Effort** | **Moderate.** No auth, stable paths, one file per epoch, 11 fixed HDUs. But no IVOA utypes, so it needs a new non-Phase-3 reader path keyed on `TELESCOP`/`INSTRUME` or filename. The genuinely expensive piece is the LSF: coefficients only, undocumented parametrisation. |
| **Quirks** | Dither combination correlates noise **even at visit level**, and SDSS states the propagation ignores it. Three chips with gaps, reverse order, per-fibre endpoints. Vacuum with no keyword. `apStar`/`mwmVisit` rest-frame — refuse. Sky- and telluric-corrected (models retained). 0.5% error floor. BOSS already barycentric-shifted. |
| **Priority** | **3.** The largest anonymous, per-pixel-error, BC-supplied, **observed-frame** epoch archive that exists today, with a 7,273-system SB2 truth set attached. Ship it with a loud quirk report about the dither-induced noise correlation, because that is precisely the assumption the diagonal `ivar` model makes. |

---

## 3. LAMOST MRS

Most of this section was established by **downloading and parsing real DR11 v2.0 MRS files** (obsids
738709169, 773214210, 983705085, 839207123, 845507123, 871707123), not by reading prose. Two questions
that the documentation leaves ambiguous — resampling, and whether the velocity correction is applied —
were settled by measurement, and one of them came out the *opposite* of what the docs imply.

### 3.1 Public access: DR11 v2.0, worldwide, no account

LAMOST publishes a machine-readable version registry that tags every release public or internal:
**<https://www.lamost.org/openapi/dr_versions>**.

| Release | `has_mrs` | Status |
|---|---|---|
| DR1–DR9 | DR6 v1+ onward | **public** |
| DR10 (v0, v1.0, v2.0) | true | **public** (v2.0 public 2024-09-29) |
| **DR11 (v0 … v2.0)** | true | **public** (v2.0 public 2025-09-26) |
| DR12 (v0, v1.0, v1.1) | true | *internal* |
| DR13, DR14 | mixed | *internal* |

Also confirmed at <https://www.lamost.org/lmusers/>. The international announcement
(<https://lamost.org/public/node/488?locale=en>) says "By the end of September 2025, the LAMOST DR11
(v2.0 version) dataset was officially released", so the domestic and worldwide dates coincide for v2.0.

**DR12 exists and is bigger** — v1.1 covers 2023-09-18 to 2024-06-03 with **15,471,948 MRS spectra**
(12.11 M time-domain) — but its doc pages redirect to an OAuth login and `dr_versions` marks every
DR12 sub-version internal. **Nothing past DR11 is reachable as of 2026-09-03.** The data policy
(<http://www.lamost.org/lmusers/cms/article/view?id=1>) releases 1-D spectra 18 months after
collection, so DR12 should open in due course.

**No registration for public data**, stated verbatim in the OpenAPI spec
(<https://www.lamost.org/openapi/openapi.yaml>, docs at <https://www.lamost.org/openapi/docs>):
"Public datasets: Accessible without authentication. Internal datasets: Requires valid authentication
token." Verified empirically from a non-Chinese IP with no account:

```
GET https://www.lamost.org/openapi/dr11/v2.0/mrs/spectrum/fits?obsid=738709169
    -> HTTP 200, 198 kB gzip, valid FITS
GET https://www.lamost.org/dr11/v2.0/tar/mrs-fits/20180407.tar.gz  (Range: bytes=0-2047)
    -> HTTP 206 Partial Content
```

**There is a documented REST API**, which makes discovery nearly free:

```
/openapi/{dr}/{ver}/{lrs|mrs}/spectrum/fits?obsid=      # single spectrum
/openapi/{dr}/{ver}/{lrs|mrs}/voservice/conesearch?ra=&dec=&sr=
/openapi/{dr}/{ver}/{lrs|mrs}/voservice/ssap?pos=&size= # IVOA SSAP
/openapi/{dr}/{ver}/query/{table}                        # POST, JSON constraints
/openapi/{dr}/{ver}/sql
/openapi/dr_versions
```

Bulk mirroring is one `.tar.gz` per observing night at <https://www.lamost.org/dr11/v2.0/tar/mrs-fits/>
(0.6–15 GB each, with MD5 files). A reference Python client documenting the same endpoints is
`pylamost` (<https://github.com/fandongwei/pylamost>).

### 3.2 The prize: 10.4 million public medium-resolution epoch spectra

From <https://www.lamost.org/dr11/>: DR11 v2.0 holds **13,156,050 MRS spectra = 2,737,394
non-time-domain + 10,418,656 time-domain**, and 2,594,070 MRS stellar-parameter sets.

Epoch depth, queried directly from `med_mec` (the MRS Multiple Epoch Catalogue):

| single exposures per source | sources |
|---|---|
| ≥ 30 | **128,716** |
| ≥ 50 | **78,504** |
| ≥ 100 | **19,750** |

Time-domain plates are named `TD*`; 1,074 plate-nights exist, 82 plates have ≥3 nights, and
`TD010142N094445K01` alone was observed on 16 nights from 2018-10-17 to 2021-01-02. **No other public
optical survey offers this many epochs per star at R ≈ 7500 with per-pixel inverse variance.**

### 3.3 Format, verified against real files

One gzipped FITS per obsid (target × plate-night) contains **every epoch**:

- HDU 0 — header-only primary (`EXTNAME = 'Information'`)
- HDU 1, 2 — `COADD_B`, `COADD_R`, present only when `COMBIN_B`/`COMBIN_R` = T (otherwise empty stubs
  with `NAXIS=0`, a case the loader will meet)
- HDU 3…N−1 — **the individual single exposures**, `EXTNAME = 'B-<LMJM>'` / `'R-<LMJM>'`, LMJM being
  the local modified Julian *minute*

Files parsed carried 3, 6, 8, 12 and 16 single-exposure extensions. Naming
`med-MMMMM-PLANID_spXX-FFF.fits` (LMJD, plan, spectrograph 1–16, fibre 1–250).

Columns — coadds: `FLUX`, `IVAR`, `WAVELENGTH`, `ANDMASK`, `ORMASK`, `NORMALIZATION`. Single exposures:
`FLUX`, `IVAR`, `WAVELENGTH`, `PIXMASK`, `NORMALIZATION`. **Per-pixel inverse variance is present
everywhere** — no conversion, no missing-error fallback.

**`VACUUM = T / Wavelengths are in vacuum`** in every file. A declared medium, which most archives never
give.

#### The decisive measurement: coadds are resampled, single exposures are not

| Product | Grid | Pixels |
|---|---|---|
| **Single exposure** | **native, non-uniform per-fibre arc solution** | always exactly **4136** |
| Coadd | uniform log₁₀, **Δlog₁₀λ = 1.000 × 10⁻⁵ (6.90 km/s/px)**, common to all files and both arms | 4129 / 4153 / 4242 (B), 3744 / 3864 / 3869 (R) |

Within one single exposure Δlog₁₀λ ranges 8.3 × 10⁻⁶ to 1.17 × 10⁻⁵, and in the R arm dλ *decreases*
with wavelength — the opposite of log-linear. The grid changes night to night for the same fibre
(median offsets −9 to −94 km/s between epochs of `sp07-123`), i.e. it is the per-exposure arc solution.
That the coadd pixel count differs from the native 4136 is definitional proof of interpolation.

**And the noise consequence was measured**, on 7-pixel-median residuals of a low-S/N file
(obsid 773214210):

| | rms(resid/σ) | lag-1 autocorrelation |
|---|---|---|
| **Single exposures** | 0.94–1.11 | −0.05 to +0.03 → **white; `IVAR` is correct** |
| Coadds | 0.67, 0.70 | **+0.38, +0.41** → correlated; `IVAR` overstates independence |

This is albireo's D4 failure mode, caught in the act — and, uniquely among the archives in this report,
**it is cleanly avoidable**: read the single-exposure extensions that ship in the very same file. The
loader should default to them and refuse or loudly flag the coadds.

#### Correction to the documentation: the heliocentric shift *is* applied

The docs say only that `HELIO_RV` is "the radial velocity used to carry out the heliocentric
correction", which reads as though the wavelengths were left alone. **They were not.** The test: an
RV-constant bright star (`J010609.79+082520.9`, S/N 250–500, catalogue `rv_r0` stable at −4.3 to
−5.9 km/s over 16 epochs), same fibre and spectrograph (`sp07-123`), three epochs whose `HELIO_RV`
spans 31.6 km/s:

| `HELIO_RV` (km/s) | Hα centroid velocity | Mg b 5183 |
|---|---|---|
| −2.84 | −2.04 | +10.79 |
| +6.47 | −5.22 | +10.48 |
| +28.77 | −5.12 | +8.08 |

The lines hold constant to ~2 km/s across a 31.6 km/s swing in `HELIO_RV`. Had the correction not been
applied they would have moved 31.6 km/s ≈ 0.69 Å ≈ 5 pixels. `HELIO = T` in the header confirms it.

**This is the good outcome.** Earth's motion is divided out; the star's own velocity is not. These are
*not* rest-frame spectra, so the orbital signal is intact. Two consequences for the loader: report
`frame` as already-corrected and **do not apply `HELIO_RV` a second time**, and record that the
correction is **heliocentric, not barycentric** (a ≤30 m/s difference, negligible against 6.9 km/s
pixels, but it belongs in `v_bary_source`).

#### What is missing

- **No BJD or HJD keyword.** Only `DATE-OBS` (median UTC, per exposure), `DATE-BEG`, `DATE-END`,
  `LMJM`, `EXPTIME` (typically 1200 s), `SNR`. albireo must compute BJD_TDB itself from `DATE-OBS`
  plus `RA`/`DEC` and the site coordinates, which the header supplies (`LONGITUD = 117.58`,
  `LATITUDE = 40.39`). `io.py` already does exactly this, so it is cheap.
- **No LSF and no per-fibre resolution array.** Only the scalar R = 7500 at 5163 Å and 6593 Å. Label
  matching would need an assumed Gaussian LSF or a fitted-nuisance resolution.
- Coverage is nominally B [4950, 5350] Å and R [6300, 6800] Å; real files run ~4890–5395 and
  ~6257–6851 Å.

Also present and useful: `PCASKYSB`, `NSKIES`, `SKYCHI2`, `FIB_MASK`, `SEEING`, **`MOONPHA`**, `SPID`,
`FIBERID`, and `OBJNAME` carrying the Gaia DR3 source_id for many targets. The `med_catalogue` table
supplies eight pipeline RVs per spectrum (`rv_b0/b1`, `rv_r0/r1`, `rv_br0/br1`, `rv_lasp0/1`); the `*1`
variants are calibrated onto Huang et al. (2018) standards but **correct only spectrograph-to-
spectrograph systematics, not fibre- or exposure-level ones** — i.e. a documented per-fibre RV zero
point that albireo would inherit, and which its per-component zero-point machinery already models.

### 3.4 SB2 catalogues: ~13,000 systems, 665 with orbits

| Reference | Content | Access |
|---|---|---|
| **Kovalev, Zhou, Chen & Han 2024, MNRAS 527, 521** (arXiv:2310.11673) | **12,426 SB2 candidates** (8,105 new) over all MRS spectra, with mass ratios, systemic velocities and Keplerian orbits for the multi-epoch subset | VizieR **`J/MNRAS/527/521`** — the largest MRS SB2 catalogue |
| **Li S.-S. et al. 2025, ApJS 276, 11** (arXiv:2411.14714) | DR9 MRS (29.9 M spectra) → **7,096 SB2 + 1,903 SB3** candidates (70% / 90% new) | VizieR `J/ApJS/276/11` |
| **Guo S. et al. 2025, ApJS 278, 46** (arXiv:2504.11954) | modified `thejoker` on 1,119 SB2s with ≥6 epochs → **665 reliable SB2 orbital solutions** (P, q, e), cross-checked against Kepler/TESS/ZTF | VizieR `J/ApJS/278/46` — **the natural albireo validation set** |
| **Kovalev, Chen & Han 2022, MNRAS 517, 356** (arXiv:2207.06996) | 2,460 SB2 candidates (1,410 new) | VizieR `J/MNRAS/517/356` |
| **Zhang B. et al. 2022, ApJS 258, 26** (arXiv:2112.03818) | CNN on **single-exposure** spectra → 2,198 SB2 candidates; favours FGK MS binaries with q ≥ 0.7, Δv ≥ 50 km/s | VizieR `J/ApJS/258/26` |
| **Li C.-q. et al. 2021, ApJS 256, 31** (arXiv:2109.00751) | 1.3 M DR7 blue-arm spectra → 3,133 SB + 132 triple candidates | no VizieR catalogue found (`J/ApJS/256/31` → 404) |
| **Wang Z. et al. 2025, MNRAS 539, 3506** (arXiv:2601.12824) | **126 published SB2s are false positives** — single stars with *moonlight* contamination in DR10 MRS; drivers are lunar phase, G magnitude, moon angular distance | no VizieR catalogue found |

The union is roughly **13,000–15,000 distinct SB2 candidates** (Kovalev+2024 dominates and absorbs
4,321 previously known), of which **665 have published orbits**. Every one has public epoch spectra
today.

Two of these deserve emphasis. **Guo et al. 2025** is a ready-made validation set of exactly the kind
D55 used AI Phe for, but 665-strong. And **Wang et al. 2025** is a documented, quantified false-positive
channel: moonlight puts a *solar* spectrum into the fibre, which a disentangler will happily report as
a second component. Surfacing `MOONPHA` and the moon angular distance in the loader's metadata is a
cheap defence, and worth doing because albireo's whole selling point is not manufacturing components
that are not there.

Also noted: **Nachmani et al. 2026** (arXiv:2602.05610) presents MESS — multi-epoch global TODCOR with
jointly optimised templates and BIC-based S1/SB1/SB2 model selection — validated on 1,500 *simulated*
LAMOST MRS systems. That is direct methodological competition for `albireo.todcor` on this exact
dataset, and a reason to move rather than wait.

### 3.5 Verdict

| | |
|---|---|
| **User base** | **~13,000–15,000 SB2 candidates**, 665 with published orbits, all with public epoch spectra. **128,716 sources with ≥30 epochs, 19,750 with ≥100.** Population is FGK main-sequence, not hot massive stars. |
| **Effort** | **Low — the lowest in this report.** One gzipped FITS per obsid holding every epoch; plain BINTABLEs with explicitly named columns, so no utype/UCD dispatch needed at all; discovery is one POST to a documented REST API; **no authentication, no captcha, no account**. Real work is: walk the variable HDU list parsing `EXTNAME='{B,R}-<LMJM>'`, compute BJD_TDB (machinery exists), decide the arm policy. |
| **Quirks** | Coadds resampled with measured lag-1 noise autocorrelation ≈ +0.4 — **but avoidable**, single exposures are native and measured white. Heliocentric (not barycentric) correction **already applied**; do not re-apply. No BJD. No LSF. Two disjoint arms with separate RV zero points. Moonlight is a documented SB2 false-positive channel. `VACUUM = T` declared. |
| **Priority** | **2 — the best available-today target, and the one to start now.** It is the only archive combining ~13k known SB2s, 100+ epochs for ~20k stars, per-pixel inverse variance, a declared medium, an applied velocity correction, **unresampled native-grid single exposures**, and zero authentication. |

---

## 4. SOPHIE and ELODIE (OHP)

As with LAMOST, the load-bearing facts here were established by downloading and parsing real products,
because the two archives turn out to differ in a way no documentation states.

**Transport caveat first: `atlas.obs-hp.fr` is HTTP-only.** Port 443 actively refuses connections
(`ECONNREFUSED 193.50.62.66:443`). A loader must not force an HTTPS upgrade — which is exactly what
albireo's `WebFetch`-style helpers and many HTTP libraries do by default.

### 4.1 There is a registered IVOA SSAP endpoint, and it covers both instruments

The archive's own CGI is HTML/plain-text only. <http://atlas.obs-hp.fr/sophie/advanced.html> says
verbatim: "Although this plain-text format is the only machine-oriented format now available for the
catalogue, we may provide a tab-separated format (tsv) or VO_table format in the future." There is no
TAP and no votable/csv output parameter.

**But a RegTAP query against <https://dc.g-vo.org/tap/sync> returns
`ivo://obs-hp.fr/elodie/ssa`** — registered under the ELODIE title, but its own description reads
"ELODIE/SOPHIE spectral archive (SSA access)" and it serves both. Verified live:

```
http://atlas.obs-hp.fr/elodie/E.cgi?a=t&c=ssa&n=ssa&POS=300.1825,22.7108&SIZE=0.02&FORMAT=all
  -> VOTable 1.1, 162 rows: 114 instrume="Sophie", 48 instrume="Elodie"
```

**This is the cheap path**: one SSAP cone search returns a VOTable spanning both instruments, with
`ssa:` utypes, and the SOPHIE rows carry a direct `Access.Reference` FITS URL. Two limits — `mjd` is
declared "Rounded Modified Julian date" and is an **integer** column, and `redshift` is populated
oddly. So SSAP is a **discovery layer only**; timing must come from the FITS header.

The native CGI works as a fallback and, usefully, supports a bulk index: `o=%` returns the whole
archive (22.3 MB / 144,281 rows in ~90 s), so an index can be built in one request without scraping.
**Name canonicalisation is a real trap** (<http://atlas.obs-hp.fr/sophie/designations.html>): "HD 1 is
rewritten to HD000001". Confirmed — `o=HD10700` returns nothing, `o=HD010700` returns rows.

**A documentation bug worth recording**: the FITS URL printed on the archive's own pages returns an
HTML stub. `a=mime:application/fits` is mandatory:

```
http://atlas.obs-hp.fr/sophie/sophie.cgi?c=i&a=mime:application/fits&o=sophie:[s1d,923460]
```

### 4.2 SOPHIE: resampled, barycentric, and with no errors at all

**`e2ds` is public** (listed as "get_e2ds : Download e2ds extracted spectra (39 orders), if released"
at <http://atlas.obs-hp.fr/sophie/intro.html>) and downloads successfully: `E2DS_A` (4077 × 39),
`WAVE_A`, `BLAZE_A`, then the same for fibre B.

**`s1d` is resampled onto a fixed 0.01 Å linear grid, in air.** From a real header (seq 923460):

```
NAXIS1 = 307122   CRVAL1 = 3872.92   CDELT1 = 0.01   CTYPE1 = 'AWAV'   CUNIT1 = 'Angstrom'
```

`CTYPE1='AWAV'` is the FITS WCS code for **air**, corroborated by the SSAP metadata declaring
`spectralucd = "em.wl;obs.atmos"` ("UCD for spectral coord: Air-wavelength"). The archive states the
provenance directly: s1d files "are the result of the resampling in wavelength of the e2ds data, and of
the reconnection of all the orders in a single 1D spectrum"
(<http://atlas.obs-hp.fr/sophie/fits.html>).

**`s1d` is in the barycentric frame — proven twice.** (a) Cross-correlating the s1d against order 20 of
the same observation's `e2ds`/`WAVE_A` gives a best shift of **+27.60 km/s** against a header
`HIERARCH OHP DRS BERV = +27.65 km/s`. (b) Across two HD189733 epochs with ΔBERV = +2.769 km/s, the O₂
B-band tellurics move +2.80 km/s while the stellar bands move +0.40 km/s. **Tellurics track ΔBERV, the
star does not.**

So SOPHIE `s1d` is barycentric while its `e2ds`/`WAVE_A` is **topocentric** — two different frames
inside the same download. Neither is rest-frame shifted, so both are usable; but BERV must not be
applied twice to the s1d.

**No per-pixel errors, at any level.** `s1d` holds `S1D_A` and `S1D_B` only; `e2ds` holds
`E2DS_A/WAVE_A/BLAZE_A` and the fibre-B triple. No variance, sigma or error extension anywhere — the
HARPS situation. For `e2ds` you can reconstruct Poisson errors from `BLAZE_A` (restoring electrons)
plus `HIERARCH OHP DRS CCD SIGDET = 6.` (read noise, e⁻) and `CONAD = 2.85` (e⁻/ADU). **For `s1d` you
cannot**, because the resampling has already correlated the noise.

Verified header keywords: `HIERARCH OHP DRS BJD` (barycentric JD, mid-exposure),
`HIERARCH OHP DRS BERV`, `BERVMX`, `HIERARCH OHP OBS MJD`, `HIERARCH OHP OBS DATE START`/`DATE END`,
`HIERARCH OHP CCD DKTM` (exposure time — **there is no plain `EXPTIME`**),
`HIERARCH OHP DRS CAL EXT SN0…SN38` (per-order S/N), `HIERARCH OHP DRS CCF RV`/`ERR`/`FWHM`/`CONTRAST`/
`SPAN`/`MASK`, `HIERARCH OHP DRS DRIFT RV`, `HIERARCH OHP INS FIBER` (HR/HE), `DIVUL`, `SEQNUM`.

**An asymmetry that costs a second fetch**: the s1d primary header is a **39-card subset**
(`CREATOR = 'Pleinpot 2'`) with no S/N, no CCF and no exposure metadata beyond `OHP CCD DKTM`. The full
1063-card header exists **only in the `e2ds`**.

**Proprietary period and a disentangling-specific hazard.** One year for PIs, but
(<http://atlas.obs-hp.fr/sophie/intro.html>) "some large key programs are granted an extended (5-year)
protection. Spectra benefiting from this extension are released after a year, but in a modified form
where **the exact time of observation is truncated**." And
(<http://atlas.obs-hp.fr/sophie/help.html>): BJD is present "with full precision for the public
observations, and **truncated to a day for the others**."

**A day-truncated BJD cannot phase an orbit.** The loader must reject or hard-flag any epoch whose BJD
lacks sub-day precision rather than silently accepting it — otherwise the truncation appears as
orbital scatter, which is precisely the class of error `time_source` exists to make visible.

Archive size, measured from the full listing today: **144,281 spectra** with valid identifiers, 4,439
(3.1%) marked not public, **139,842 downloadable**, over 8,633 distinct object positions — up 31% since
the archive's stated 2019 figures.

**Resolution measured, not quoted**: R = λ/FWHM over 1,531 ThAr lines in the archive's own
`E2DS_B`/`WAVE_B` gives **median R ≈ 70,000** (16–84%: 64,700–73,300) in HR mode. The commonly quoted
R ≈ 75,000 is *unverified from a primary source* — OHP rebuilt its main site and every instrument
documentation URL the archive links to now soft-404s. Only `atlas.obs-hp.fr` survives.

**Known data defect** (<http://atlas.obs-hp.fr/sophie/news.html>): ~100 spectra from 2006–2007 carry a
wrong `OHP DRS BERV` from a missing decimal point in `OHP TARG DELTA`, and "the s1d reconnected spectra
have also undergone this inexact barycentric correction".

### 4.3 ELODIE: smaller, frozen, has errors, and the opposite frame convention

- **Fully public**: "All data in the Archive (35535 spectra) are public since August 2011"
  (<http://atlas.obs-hp.fr/elodie/intro.html>) — confirmed exactly, the bulk listing returns 35,535
  rows. Data span 1994-07-08 to 2006-08-13. Live and answering in 2026.
- Retrieval:
  `http://atlas.obs-hp.fr/elodie/fE.cgi?n=e500&c=i&z=s1d&a=mime:application/fits&o=elodie:19940826/0009`
- **`s1d`: 4000.00–6799.95 Å, `CDELT1 = 0.05` Å, 56,000 px, `CTYPE1='AWAV'` (air).** Resampled.
- **Per-pixel errors ARE supplied** — three HDUs: `INTENSITY`, `FCANOR` (flux-calibration relation),
  **`NOISE`**. Verified numerically: median flux/noise = 75.3 against a header `SN = 79.9`.
- **`s1d` is TOPOCENTRIC — BERV supplied but not applied.** Two HD190073 epochs with
  ΔBERV = +33.594 km/s: the O₂ γ tellurics move −0.50 km/s (static) while the stellar bands move
  −34.50 and −33.25 km/s, i.e. −ΔBERV.

> **This is the single most dangerous asymmetry in this report. SOPHIE `s1d` is barycentric; ELODIE
> `s1d` is topocentric. A loader that treats "an OHP s1d" uniformly will corrupt one of the two.**
> Dispatch on `INSTRUME`, never on the archive.

- BJD is **split across two keywords**: `JDB1 = 2.44959100E+06` plus `JDB2 = 4.30896282E-01`. Plain
  `BERV`, `BERVMX`, `EXPTIME`, `MJD-OBS`, `SN`, `AIRMASS`, `H_WRESOL` (0.1321 Å FWHM → R ≈ 41,600 at
  5500 Å, matching the nominal 42,000).
- **NaN masking**: "The region of some telluric absorption or of some gliches were masked and replaced
  by IEEE undefined values." Measured 97 NaN pixels in 3 blocks in the red inter-order gaps, consistent
  between `INTENSITY` and `NOISE` — so masking on NaN is safe.
- **`s2d` is awkward**: (1024 × 67) plus `BLZ`, with **no wavelength extension**; the solution lives in
  ~250 Chebyshev coefficient keywords "described in the ELODIE manual", which could not be located
  online (**unverified**).
- Note that VizieR **III/218 "ELODIE archive" is not the archive index** — it is the Prugniel &
  Soubiran 2001 stellar *library*. Description paper: Moultaka et al. 2004, PASP 116, 693.

### 4.4 SB2 target lists — and SB9 has been superseded

**SB9 is retired.** <https://sb9.astro.ulb.ac.be/> states: "Since 2025-06-24, the SB9 catalogue is
superseded by **SBX**". The old server and a mirror remain up.

**SBX** (<https://www.astro.ulb.ac.be/sbx>, Merle, Jorissen, Alexandre et al. 2026, MNRAS 547):

| Systems | Orbits | SB1 | **SB2** | SB1&SB2 |
|---|---|---|---|---|
| 4080 | 5169 | 2858 | **1326** | 129 |

**SBX has a working IVOA TAP service** at `https://astro.ulb.ac.be/sbx/tap/sync`, verified live:
`systems` 4,080; `orbits WHERE k2 IS NOT NULL` **1,699**; **`velocities` 224,148 individual RV
measurements with a `component` column ('a'/'b')**. Tables: `systems`, `orbits`, `velocities`, `alias`,
`configurations`, `duplicates`; `systems` carries Gaia DR3 astrometry, `st1`/`st2` spectral types and
`mag1`/`mag2`.

**This is directly consumable by albireo's existing TAP path and gives both a target list and
per-component RV priors — independent of which spectrograph the spectra come from.** It is the single
most reusable find in this section, and it should be wired up regardless of which loader is built next.
VizieR mirrors the old SB9 at `B/sb9`.

**Cross-match of the 1,333 SB9 SB2 systems against both OHP archives at 20″:**

| | SB2 systems present | ≥3 epochs | ≥5 | ≥10 | total spectra |
|---|---|---|---|---|---|
| **SOPHIE** (public only) | 193 | **133** | 112 | 99 | 2,844 |
| **ELODIE** (all public) | 198 | **101** | 65 | 36 | 1,972 |

Union **318** distinct SB2 systems, 73 in both. Best covered — SOPHIE: HD123999 (12 Boo, 310 spectra),
HD195987 (118), HD178911 (113), HD210027 (ι Peg, 108); ELODIE: DG Leo (275), HD098088 (173),
HD028319 (137).

Published OHP SB2 catalogues: **`J/A+A/587/A64/tableb2`**, "SOPHIE RV of SB2 stars" (Santerne, Moutou,
Tsantaki et al. 2016) — 41 epochs with per-component `RVA`/`RVB`. Also `J/A+A/631/A125` (Kiefer et al.
2019, 54 massive companions, predominantly SB1) and `J/A+A/589/A83/table2` (Gebran et al. 2016, 306 A
stars from PolarBase + SOPHIE + ELODIE). No Halbwachs SOPHIE binary-programme catalogue was found in
VizieR (**unverified**).

### 4.5 Verdict

| | SOPHIE | ELODIE |
|---|---|---|
| **User base** | 133 SB2 systems with ≥3 epochs, 99 with ≥10 | 101 with ≥3 epochs, 36 with ≥10 |
| **Auth** | None, anonymous HTTP (never HTTPS) | None |
| **Discovery** | **Low** — registered SSAP returns VOTable; bulk index in one 90 s request | **Low** — same SSAP; 35,535-row index in one request |
| **Format** | Low-medium. Plain FITS, hierarchical keywords, **but full metadata only in `e2ds`, so two fetches per epoch** | Low for `s1d`; **high** for `s2d` (Chebyshev solution in ~250 keywords, manual offline) |
| **Quirks** | `s1d` resampled to 0.01 Å; **no errors at any level**; `s1d` barycentric while `e2ds` is topocentric; **BJD truncated to a day for 5-year-protected epochs**; ~100 spectra with known-wrong BERV; air | `s1d` resampled to 0.05 Å; **topocentric, the opposite of SOPHIE**; BJD split across `JDB1`+`JDB2`; NaN-masked blocks; air; **`NOISE` extension present** |
| **Priority** | **6** — 3× the SB2 epoch depth and still growing (+31% since 2019), and SSAP makes discovery nearly free; but no errors anywhere is a real cost | **7** — smaller and frozen since 2006, yet the only one of the two shipping per-pixel errors, so the better *validation* target for the diagonal inverse-variance model |

---

## 5. Other archives

Two findings here overturn the assumptions the brief was written on: **HERMES's archive is public even
though the telescope is consortium-only**, and **ExoFOP hosts real reduced spectra, not just
dispositions**.

### 5.1 HERMES / Mercator — public, TAP-served, and it speaks albireo's dialect already

The *telescope* is consortium-only (KU Leuven, ROB, ULB, Geneva, Tautenburg, plus 20% Spanish CAT). The
*archive* is open to anonymous users.

- Portal <https://mercatorvo.ster.kuleuven.be/>, form `/hermes/q/web/form`
- **IVOA TAP at `https://mercatorvo.ster.kuleuven.be/tap`** (DaCHS 2.9.1), one table `hermes.data`
  with full SSA columns. Anonymous ADQL verified.
- **119,650 spectra, 15,843 distinct target names**, MJD 54992.86 (2009-06) to 60299.21 (2023-12-24).
- `embargo` and `owner` are NULL on every row (DaCHS "public"). Anonymous file download returned
  HTTP 200.

**The format is the headline.** `accref` returns an **IVOA SDM VOTable** (`utype="spec:Spectrum"`,
`ssa_model=Spectrum-1.0`) — *the same utype dialect albireo's ESO Phase 3 reader already dispatches
on*. This is the one archive in the report where D45's utype principle pays off directly with no new
dispatch code. The catch: the served VOTable carries only two FIELDs, `spectral` (Å, `ucd=em.wl`) and
`flux`. **No error column.**

The DataLink document (`/hermes/q/sdl/dlmeta?ID=…`) exposes a `#progenitor` link to the native DRS
FITS, which is richer:

```
NAXIS1  = 167782             EXTNAME = 'FLUX'
CTYPE1  = 'log(wavelength)'  CRVAL1 = 8.23303754621335  CDELT1 = 5.20050525665283E-06
BJD     = 2460264.4193128    / Barycentric Julian Date of midpoint
BVCOR   = -18.599829         / [km/s] Barycentric rv correction at midpoint
OBSGEO-X/Y/Z, DATE-AVG present
```

**`BJD` and `BVCOR` are real FITS keywords, not comments — the best timing metadata of any archive
here**, and exactly the two quantities `RawSpectrum` wants.

Quirks: the served product is order-merged onto a uniform **ln-λ grid at 1.559 km/s per pixel**, and
the DRS says so plainly — "the final order merging is only available in logarithmic scale: rebinning is
done with a constant wavelength step in ln-space"
(<http://www.mercator.iac.es/instruments/hermes/drs/>). So it carries the D4 violation. **No error
array is served**: a single HDU, header plus one flux array, no `XTENSION`. The DRS does produce
`_Var_*.fits` variance files and non-resampled order-by-order `_ext.fits` products, but **the archive
serves neither**. Medium: the archive declares `ssa_spectralucd = em.wl`, which in the SSA/DaCHS
convention means vacuum (`em.wl;obs.atmos` would be air) — **declared but not independently verified**.

R = 85,000 (63,000 low-res fibre), 377–900 nm. HERMES has a dedicated binary-star programme literature
("Binary Star Research During the First Six Years of Operation of the HERMES Spectrograph", EAS
Publications Series 71). SB2 user base is **order 10²–10³ systems with public multi-epoch spectra — an
order of magnitude, not a count**. **MELCHIORS** (3,256 spectra of 2,043 stars, A&A 681, A107,
arXiv:2311.02705) is a separate curated release, but at ~1.6 epochs per star it is a *library*, not a
time series; the main archive is the multi-epoch source.

**Priority: 4.** Public, TAP-served, SDM VOTable that the existing reader already understands, real
BJD/BVCOR keywords, and a user base that overlaps albireo's actual hot-star/pulsator community. The only
real cost is documenting the ln-λ resampling and synthesising errors.

### 5.2 CFHT ESPaDOnS via CADC — the best *format* in the report

Proprietary period is "typically 13 months after the end of the semester"
(<https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/cfht/>). Anonymous IVOA TAP verified at
**`https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync`** — note `/argus/`, not `/tap/`, which
404s. Anonymous file download verified at `…/data/pub/CFHT/<file>`.

ESPaDOnS: R ≈ 68,000 (polarimetric / object+sky) or 81,000 (object only), 370–1050 nm. `*i.fits`
(intensity) and `*p.fits` (polarimetric) at `calibrationLevel=2`. Read from a real file (`1095448i.fits`):

- **IMAGE HDU, 213,578 pixels × 12 columns** (28 in Star+Sky mode): COL1–3
  Wavelength / Intensity / **ErrorBar**, *normalised*; COL4–6 the same *unnormalised*; COL7–9
  normalised *without* autowave; COL10–12 unnormalised without autowave.
- **Per-pixel errors are real** (measured ≈0.02 on a continuum of 1.00), and **both normalised and
  unnormalised fluxes ship together** — which is unusual and valuable, since albireo prefers to fit its
  own continuum.
- **Orders are concatenated, not merged, and not resampled**: 36 decreasing steps counted in the
  wavelength array, i.e. it is non-monotonic at order boundaries. **This is the only optical product in
  the report that is both error-bearing and unresampled.**
- Wavelength is in **nanometres** (369.69 → 1048.09), not Å.
- The normalised and unnormalised sets sit on **different wavelength arrays with zero padding**
  (COL1 ≠ COL4; COL5 = 0 where COL2 ≈ 1). A loader must not assume a shared grid.
- The "autowave" telluric wavelength correction shifts the scale by **~0.6 km/s at 605 nm** relative to
  the no-autowave columns (measured); the docs note the no-autowave variant "is used mostly by
  spectroscopists". Expose the choice rather than picking silently.

**The one real cost: barycentric metadata lives in COMMENT cards, not keywords.**

```
COMMENT Heliocentric velocity of observer towards star : -16.312 km/s
COMMENT Geocentric Julian date (UTC) : 2455017.75670
COMMENT Heliocentric Julian date (UTC) : 2455017.76039
COMMENT Heliocentric Julian date (TT)  : 2455017.76114
```

So the spectra **are already in the heliocentric frame**, the correction is *heliocentric* not
barycentric, and the loader **must parse COMMENT text** to recover either quantity. Proper keywords give
only `MJDATE`/`MJD-OBS` (start), `MJDEND`, `REL_DATE`. Air/vacuum **unverified**.

**SPIRou** (near-IR) is also at CADC, `calibrationLevel=2`, products `e/s/t/v.fits`. `t.fits` (telluric
corrected) is a MEF with `FluxAB, WaveAB, BlazeAB, Recon, OHLine, FluxA/B, WaveA/B, BlazeA/B` — 4088 ×
49 orders, **no error extension**. Metadata is excellent: `BERV` (with `BERVSRCE = 'barycorrpy'`),
`BJD`, `BERVMAX`, `MJDMID`, and `WAVE0000…NNNN` polynomial coefficients. Near-IR limits SB2 relevance
mostly to M-dwarf binaries.

**PolarBase** (<http://polarbase.irap.omp.eu/>) is **online in 2026** and actively maintained (rebuilt
as a Vue SPA). Its own text: "Instruments with data available through PolarBase include Narval,
ESPaDOnS and SPIRou. As of 2025, observations of **5,300 stellar objects** … are included." Public by
construction. API endpoints exist (`/api/spectra`, `/download/wget_links`) but take JSON POST with an
**undocumented parameter shape — probes returned HTTP 400, so the API is unverified**; spectrum count
(as opposed to object count) also unverified. Its marginal value over CADC is NARVAL (Pic du Midi).

**Priority: 5.** Public, TAP-served, per-pixel errors, unresampled concatenated orders, and a large
hot-star and pulsator user base. Take PolarBase only if NARVAL is wanted.

### 5.3 CHIRON — reduced spectra exist, but via ExoFOP, not NOIRLab

NOIRLab's Astro Data Archive lists `chiron` among its instruments, but **every CHIRON file is
`proc_type=raw`, `prod_type=image`**; explicit queries for `instcal`, `resampled`, `skysub`, `stacked`
and `mastercal` each returned zero rows. Raw 2-D frames only. The official reduced-data route is a
per-program login issued to PIs — not public. Modes R = 130,000 / 95,000 / 80,000 / 25,000, 440–880 nm;
extracted spectra are "provided as a data cube in a fits file", "in the wavelength range 450-885nm, and
are not sky-subtracted" (<http://www.astro.yale.edu/smarts/1.5m.html>).

**But reduced CHIRON spectra are public in bulk on ExoFOP.** A real file
(`TIC4711S-ct20220324_1154.fits`) reads:

```
NAXIS = 3    NAXIS1 = 2 (0=wavelength, 1=spectrum)   NAXIS2 = 3200 px/order   NAXIS3 = 59 orders
OBSID = 'ct15m.chiron.20220324.070246'   DECKER = 'slicer'   UTSHUT = '2022-03-24T07:02:46.108'
BARYDIR = 'tous/CHIRON/bary/'
```

Per-order, **not merged and not resampled** (good). **No error array** — the cube is wavelength plus
flux only. **No BJD or BCV value keyword**, only `BARYDIR`, a path — so BERV must be computed, which
`io.py` already does. Air/vacuum **unverified**. Effort is low (a 3-D cube, trivially sliced), and it
folds into the ExoFOP route below.

### 5.4 ExoFOP / TESS follow-up — a target list, and unexpectedly a data source

**The spectroscopy table is a log, not data**:
<https://exofop.ipac.caltech.edu/tess/download_spect.php?output=pipe> returns **34,315 rows over 9,144
distinct TIC IDs** with columns `TIC ID | … | Telescope | Instrument | Spectral resolution | … | Obs
date | Notes` — no RVs and no file links. Instrument breakdown: TRES 14,947, HIRES 5,840, **CHIRON
4,723**, MinervaAus 1,969, FIES 1,017, Tull 693, NRES 547, HPF 329, CORALIE 279, PFS 238, NEID 222.
**57 rows mention SB2 or double-lined in free-text Notes**; 19 mention SB1.

**But `download_files.php?id=<TIC>` returns an anonymous tar of that target's uploaded files, and it
contains real spectra.** For TIC 4711 that yielded ~14 `TIC4711S-ct<date>_<n>.fits` files, one of which
is the genuine reduced CHIRON echelle cube quoted above. Convention:
`TIC<id><S|P|I|O>-<observer code><date>`, where **S = spectroscopy (SG2)**. This is the public route to
CHIRON and probably TRES spectra.

Target lists: the TOI table (<https://exofop.ipac.caltech.edu/tess/download_toi.php?output=pipe>) has
**8,148 rows**, with *TESS Disposition* PC 6,416 / **EB 644** / KP 596 / CP 405. Note the *TFOPWG*
disposition column has **no "EB" or "SB2" value** — eclipsing binaries are flagged only in the TESS
Disposition column and otherwise hide inside its 1,295 FPs. Separately, **Prša et al. 2022, ApJS**
(arXiv:2110.13382) gives **4,584 eclipsing binaries from sectors 1–26**, at tessEBs.villanova.edu and
MAST.

Effort is medium — no TAP, but the pipe-delimited bulk endpoints and per-target tars are plain HTTP with
no scraping. Quirks: heterogeneous per-instrument formats inside one tar, no manifest, no uniform
metadata. **Priority: 8 as a data source, but high as a target list** — its 644 EB and 57 SB2-noted
entries are a ready-made albireo demo list.

### 5.5 KOA / Keck HIRES — extracted spectra exist and are unusable for velocity work

**Public TAP with no token at `https://koa.ipac.caltech.edu/TAP`**, verified anonymously (tables
`koa_hires`, `koa_reduced_data`, `koa_kpf`, `koa_esi`, …). "The TAP clients support access to public
data only"; PIs use PyKOA with credentials, and TAP+/astroquery is deprecated in favour of PyKOA and
PyVO. Proprietary period **18 months**, or 12 months for NASA Keck data from 2023A.

HIRES extracted spectra do exist, but not in `koa_reduced_data` (which covers OSIRIS, NIRC2, NIRES,
DEIMOS, KCWI, **KPF with 2.79 M level-2 files**, ESI, MOSFIRE — no HIRES). They are a separate
MAKEE-based browse product: per-order BINTABLEs `KOAID_C_NN_flux.fits` with `wave`, `Flux` and
**`Error` ("One-sigma error of the relative flux")**.

**The disqualifying quirk, verbatim** from
<https://koa.ipac.caltech.edu/UserGuide/HIRES/extracted_products.html>:

> "Wavelength in Angstroms (vacuum, heliocentric, and possibly shifted to match the wavelength
> zero-point of the night-sky lines)."

So the scale is vacuum, the heliocentric correction is already applied **with no keyword recording the
amount**, and there is a **possible unrecorded per-order zero-point shift from night-sky lines**. No
keyword on that page recovers either. KOA's own caveat: the extracted spectra "are intended to be used
as a browse product … may not necessarily be suitable for publishable science".

**For disentangling, an unrecorded per-order velocity zero point is fatal** — it is indistinguishable
from an orbital signal, which is the exact error mode albireo exists to avoid. Add only with a loud
warning, or not at all.

The **California Legacy Survey / Butler et al. 2017** released **60,949 RVs of 1,624 stars**
(arXiv:1702.03571; VizieR `J/AJ/153/208`) — a target list, not spectra.

**Priority: 9, last.** Worth noting for the future: **KPF** is a stabilised high-resolution
spectrograph with 2.79 M level-2 files already in `koa_reduced_data`, and is a far better eventual
target than HIRES.

### 5.6 ESO — confirmed unchanged, with one thing to check in albireo

`http://archive.eso.org/tap_obs/sync` is live and its schema is unchanged: `ivoa.ObsCore`, `dbo.raw`,
`dbo.ssa`, `phase3v2.files`, `phase3v2.product_files`, `phase3v2.provenance`. (Note that
`.../tap_obs/tables` returns only the ASM schema — query `TAP_SCHEMA.tables` instead.)

Phase 3 one-dimensional spectrum counts today:

| GIRAFFE | HARPS | GAIAESO | XSHOOTER | FORS2-SPEC | FEROS | UVES | NIRPS | ESPRESSO | CRIRES+ | XSL | UVES_SQUAD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,182,296 | 352,229 | 194,422 | 188,087 | 102,594 | 99,053 | 87,551 | 34,460 | 27,434 | 4,163 | 2,634 | 467 |

- **Gaia-ESO is complete and static**: last `obs_release_date` 2023-06-28, observations end MJD 58144
  (2018-01). Nothing changed.
- **HARPS is still growing**: data to MJD 61178 (2026-06), releases scheduled to 2028-02. No evidence
  of a re-reduction or DRS change altering the product shape — though DRS version keywords were not
  checked, so "no new HARPS DRS" is **unverified**.
- **One action for albireo**: NIRPS (34,460), ESPRESSO (27,434) and CRIRES+ (4,163) are all in Phase 3
  now. If the utype dispatch and the instrument/LSF tables were written before these existed, confirm
  they fall through correctly. That is a cheap regression test against the existing loader, and it is
  the only ESO-side work this review identifies.

---

## 6. Synthesis

### 6.1 The comparison

| Archive | SB2 targets w/ public epoch spectra | Auth | Discovery | Per-pixel errors | Resampled? | Frame as shipped | Medium declared? | Effort | Priority |
|---|---|---|---|---|---|---|---|---|---|
| **Gaia DR4** `rvs_epoch_spectrum` | ~10⁴ (≤78.7k bound, ⅓ confirmed) + 787k double-lined transits | none | DataLink, 5k/call | **yes** `flux_error` | **yes**, linear interp | barycentric | no (vacuum, measured) | low | **1** (from 2 Dec) |
| **LAMOST MRS** DR11 v2.0 | **~13–15k**, 665 with orbits | **none** | REST API + SSAP | **yes** `IVAR` | **no** (single exp.); yes (coadds) | heliocentric applied | **yes** `VACUUM=T` | **low** | **2** |
| **APOGEE** `apVisit` DR17 | 7,273 SB2 + 813 SB3, 325 orbits | none | `allVisit` + path | yes (σ) | dither-combined | **topocentric**, `BC` given | no (vacuum) | moderate | **3** |
| **HERMES/Mercator** | ~10²–10³ (order of magnitude) | **none** | **TAP + SDM VOTable** | **no** | **yes**, ln-λ 1.56 km/s | barycentric, `BJD`+`BVCOR` | declared, unverified | **lowest** | **4** |
| **CFHT ESPaDOnS** (CADC) | large, unquantified | none, 13 mo | CADC `argus` TAP | **yes** `ErrorBar` | **no**, orders concatenated | heliocentric applied (in COMMENTs) | unverified | medium | **5** |
| **SOPHIE** | 133 with ≥3 epochs | none (HTTP only) | registered SSAP | **no, at any level** | yes, 0.01 Å | barycentric (s1d) | yes, `AWAV` air | low-med | 6 |
| **ELODIE** | 101 with ≥3 epochs | none | same SSAP | **yes** `NOISE` | yes, 0.05 Å | **topocentric** | yes, `AWAV` air | low (s1d) | 7 |
| **ExoFOP / CHIRON** | 644 EB + 57 SB2-noted | none | bulk HTTP | no | no | none recorded | unverified | medium | 8 |
| **KOA / HIRES** | — | none, 18 mo | **public TAP** | yes | no | heliocentric + **unrecorded night-sky shift** | vacuum | low | 9 |
| ESO Phase 3 | already supported | — | `tap_obs` | varies | varies | varies | varies | — | no action |

### 6.2 Three patterns that matter more than any single archive

**Only two products in this entire survey are both unresampled and error-bearing: LAMOST MRS
single-exposure extensions, and CFHT ESPaDOnS.** Everything else violates one half of albireo's core
assumption. Gaia DR4, HERMES, SOPHIE, ELODIE, BOSS and the LAMOST coadds are all resampled; SOPHIE,
HERMES, CHIRON and SPIRou `t.fits` ship no errors at all. This is not a defect in the archives — it is
what the field does — and it means **the resampling-quirk reporting is not an edge case to bolt on, it
is the main event**. Anything less and albireo will silently inherit understated errors from six of the
nine sources here.

Two of those come with a *measured* number rather than a suspicion: LAMOST coadds have lag-1 noise
autocorrelation **+0.4** (single exposures: −0.05 to +0.03), and SDSS states outright that its visit-level
propagation "ignores the correlation of uncertainties". Both are AR(1)-shaped, which is what D34 was
built for. **This survey turns D34 from a speculative capability into the thing that makes five of these
loaders honest.**

**The velocity frame is the dispatch axis, not the archive.** Sorted by what the loader must do:

| Must refuse | Already corrected — do not re-apply | Topocentric — apply BERV |
|---|---|---|
| Gaia DR3 `rvs_mean_spectrum` | Gaia DR4 (barycentric) | **APOGEE `apVisit`** (`BC` supplied) |
| APOGEE `apStar`, `mwmVisit` | LAMOST (heliocentric) | **ELODIE `s1d`** |
| KOA HIRES (unrecorded shift) | SOPHIE `s1d`, BOSS, HERMES, ESPaDOnS | SOPHIE `e2ds` |

Note that **SOPHIE and ELODIE sit in different columns**, as do SOPHIE's own `s1d` and `e2ds`. A loader
keyed on "which archive is this" will corrupt data; it must key on instrument and product. That is a
concrete design constraint, and it was invisible until both were measured.

**Where albireo's utype principle (D45) actually binds.** Exactly one archive here serves an IVOA
Spectrum-model product that the existing reader would understand unmodified: **HERMES**
(`utype="spec:Spectrum"`, `ssa_model=Spectrum-1.0`). Gaia's `DATAMODEL_STANDARD` carries `spec:` utypes
on the *spectral axis only* — flux and flux_error have UCDs but no utypes — and stamps
`utype="spec:Spectrum"` on non-spectra including epoch astrometry, so it is a partial and slightly
treacherous match. LAMOST, APOGEE, SOPHIE, ELODIE, ESPaDOnS and CHIRON carry no utypes at all and are
pure name-table reads. **The name table is the common case; utypes are the exception.** That is worth
recording plainly next to D45, because the memory's existing note ("Gaia products carry no IVOA utypes
at all") is now known to be too strong in one direction and not strong enough in the other.

### 6.3 Recommendation

**Build two loaders before December, then Gaia on release day.**

**First — LAMOST MRS DR11 v2.0.** The largest available-today SB2 population (~13–15k candidates, 665
with published orbits), the deepest epoch coverage anywhere (128,716 stars with ≥30 epochs), zero
authentication, per-pixel `IVAR`, a declared vacuum medium, and **unresampled native-grid single
exposures sitting in the same file as the resampled coadds**. It is also the best possible dress
rehearsal for Gaia DR4: it forces exactly the same three decisions — an already-applied velocity
correction that must not be re-applied, a resampled-versus-native choice, and a missing BJD that
albireo must compute itself. And there is a competitive reason to move: Nachmani et al. 2026 present
multi-epoch global TODCOR validated on simulated LAMOST MRS systems, i.e. someone is already aiming at
this dataset with albireo's method.

**Second — HERMES/Mercator.** The cheapest loader available: public anonymous TAP, an SDM VOTable the
existing utype dispatch already reads, and real `BJD` and `BVCOR` FITS keywords. Crucially it is the
only entry here whose user base is *albireo's own* — the hot-star, massive-binary and pulsator
community that HR 6819 and BLOeM came from. LAMOST and APOGEE are large but cool-star populations;
HERMES buys relevance rather than volume.

**Third — Gaia DR4, wired on 2 December.** Nothing about the read can be tested before then, but two
halves can be built now: the GOST transit-forecasting path (verified working anonymously today) and the
BINARY2/CSV DataLink client, testable against DR3. Budget for the four corrections in §1.2 and for the
fact that the epoch grid samples the LSF at ~1.3 px/σ, below the ≥3 px/σ pixel-locking threshold TODCOR
already measured.

**Then, in order: CFHT ESPaDOnS** (best format in the survey — errors *and* unresampled orders — and a
hot-star user base, costing only a COMMENT parser); **APOGEE `apVisit`** (largest curated SB2 truth set
with per-epoch component RVs, but H-band cool stars and an expensive undocumented LSF); then SOPHIE and
ELODIE. **Do not build KOA/HIRES** — an unrecorded per-order velocity zero point is indistinguishable
from an orbital signal, which is the one error albireo exists to prevent.

**One free win, independent of all of the above: wire up SBX.** SB9 was superseded on 2025-06-24 by SBX
(<https://www.astro.ulb.ac.be/sbx>), which has a working IVOA TAP service at
`https://astro.ulb.ac.be/sbx/tap/sync` carrying 4,080 systems, **1,326 SB2s**, 1,699 orbits with a
measured K₂, and **224,148 individual RV measurements with a per-component label**. albireo's
`archive.py` already speaks this dialect. It is a target list *and* a per-component RV prior *and* a
validation set, it is instrument-independent, and it costs almost nothing.

**One maintenance item on the existing loader:** NIRPS, ESPRESSO and CRIRES+ are now in ESO Phase 3
(34,460 / 27,434 / 4,163 spectra). If the utype dispatch and instrument tables predate them, confirm
they fall through correctly.

### 6.4 What is still unverified

- Whether LAMOST imposes a per-user download quota (none was encountered).
- The air/vacuum medium for HERMES (declared via `ssa_spectralucd`, not independently checked),
  ESPaDOnS, and CHIRON.
- Row counts for the Kounkel and El-Badry VizieR catalogues (CDS returned Access Denied / HTTP 500).
- The PolarBase API parameter shape (probes returned HTTP 400) and its spectrum count.
- Whether `sdss_access` path templates cover `apVisit`.
- Whether a new HARPS DRS re-reduction has changed product shape (DRS version keywords not checked).
- SOPHIE's nominal R ≈ 75,000 — measured R ≈ 70,000 from ThAr lines, but OHP's instrument
  documentation pages are all dead, so the nominal figure has no reachable primary source.
- Which `apVisit` image row is which chip, and each row's wavelength direction.
- Whether an SDSS DR21 is scheduled.
