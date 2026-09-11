# Gaia RVS: simulator, populations, benchmark

Three modules that together answer, with numbers, how well albireo recovers double-lined
binaries from Gaia RVS epoch spectra. `albireo.gaia` is the instrument: the S/N that
follows from G_RVS, the detector and delivered grids, the noise correlation the archive's
resampling introduces, the transit cadence of the scanning law. `albireo.population`
draws the systems, either from the field's distributions or from real catalogues.
`albireo.benchmark` runs every system through the [pipeline](pipeline.md) under
knowledge tiers and writes a report. Nothing outside albireo is used at any stage; a
worked introduction is in the [tutorial](../tutorials/gaia-rvs.md).

```python
import albireo as ab
from albireo.gaia import rvs_components, rvs_model_grid, rvs_transit_times

library = ab.fetch_library("bosz2024-fgk-rvs")
grid = rvs_model_grid(250.0, vsini_max_kms=30.0)
components = rvs_components(
    library,
    [{"teff": 6100.0, "logg": 4.2, "mh": -0.1}, {"teff": 5700.0, "logg": 4.3, "mh": -0.1}],
    grid,
    vsini_kms=[12.0, 11.0],
)
dataset, truth = ab.simulate_rvs_dataset(
    components,
    grid,
    bjd=rvs_transit_times(35),
    light_fractions=(0.625, 0.375),
    orbit=ab.OrbitParams(period=4.0, t_peri=2457000.76, ecc=0.1, omega=0.7, k=(87.7, 95.0)),
    grvs=10.5,
)
```

## What the simulator reproduces, and what it does not

The chain is the one of the reference implementation (Rowan's `SyntheticSB2` with its
`GAIA_RVS` preset): components shifted to a Keplerian orbit and combined at a flux ratio,
observed on the 0.245 Å detector grid, given photon noise there at the S/N that follows
from G_RVS, and delivered on the archive grid by interpolation, so the delivered noise is
correlated. Every constant of the S/N model is a field of `RVSConstants` with its source;
the model reproduces the archive's `rv_expected_sig_to_noise` to 3-8% between G_RVS 6
and 14. Two products are shipped: the DR3 mean-spectrum sampling (0.01 nm, 2401
samples, 2.45 delivered pixels per detector pixel and a lag-one noise correlation above
0.5) and the DR4 epoch grid (0.025 nm, 961 samples), the latter from ESA's draft data
model and to be re-checked against the release.

The component spectra come from a published grid through [`albireo.library`](library.md),
not from a synthesis code: albireo synthesises nothing (the roadmap's standing non-goal).
The consequences are stated rather than hidden. BOSZ 2024 at R = 20,000 is broadened to
the RVS resolving power by a Gaussian of the quadrature width, which costs 0.3% in line
depth against a direct convolution; its microturbulence is fixed; its scale is air and is
converted to vacuum, as Gaia's is; and its FGK box stops at 7000 K, above which
`bosz2024-hot-rvs` continues to 10,000 K on the same 250 K spacing over MARCS and then
ATLAS9 atmospheres. One run uses one library, so `--library bosz2024-hot-rvs` admits the
catalogue systems the FGK box excludes and leaves the cool ones out; what the hot box
costs in exchange, a seam between two model codes, a band carried by the Paschen series
rather than by Ca II, and rotation against a resolving power of 11,500, is set out in
[`albireo.library`](library.md#above-7000-k-the-hot-box). The resolving power can be
drawn per transit from the values measured in flight per CCD row (10,983 to 12,587) while
the analysis declares the nominal 11,500, which is the systematic a real analysis carries.

The cadence model reproduces the measured structure of the scanning law: the count
distribution of `rv_nb_transits` and its dependence on ecliptic latitude, the visibility
periods of about one and a half transits separated by at least four days, and the
106.5-minute field-of-view pairs. It does not reproduce the law's phase, so the epochs
are statistically right and individually wrong; the real times of the next section are
passed as `bjd` in their place.

## Real transit times from GOST

`gost_transits(ra_deg, dec_deg)` asks the Gaia Observation Forecast Tool
(<https://gaia.esac.esa.int/gost/>) when Gaia crossed a position. The call is one GET
against ESA's IVOA Object Visibility endpoint,
`https://gaia.esac.esa.int/gost/ObjVisSAP/gaiaobjvisap?s_ra=&s_dec=`, read with `urllib`
and parsed with `xml.etree`: no new dependency, and about 32 kB per position (142
field-of-view crossings over the whole mission at RA 45, Dec +20). The service returns
`t_start` and `t_stop` of each crossing as barycentric MJD in TCB; `GostTransits.bjd`
is their mid-point plus 2400000.5, on the same TCB scale Gaia's archive uses.

The forecast is the **nominal** scanning law, not the attitude as flown. The manual says
GOST computes the transits "based on the routine nominal scanning law" and that "this is
not a full guarantee of getting the transits observed", and the service still predicts
transits after science operations ended on 15 January 2025. Two corrections stand between
that forecast and an RVS epoch list, and `rvs_transit_times_from_gost` applies both after
cutting to the release's data span:

- **The CCD rows.** The RVS is twelve CCDs in three strips over four of the seven rows of
  the focal plane, so it sees $`4/7`$ of the field-of-view transits: the DR3 documentation
  puts it as the spectroscopic instrument being "only served by 4 of the 7 Video
  Processing Units". The anonymous endpoint does not publish which row a transit crossed
  (the interactive form's CSV does, as `CcdRow[1-7]`, and `GostTransits` carries it when
  given), so the row is drawn uniformly from the seven. That is an approximation stated as
  one: the across-scan position drifts slowly with the scanning law and is correlated
  inside a visibility period, so an independent draw gets the number of RVS epochs right
  and their clumping wrong.
- **The losses.** `GOST_USABLE_FRACTION = 0.78` of the predicted RVS transits yield a
  usable spectrum. GOST's own landing page puts the probability of the data reaching the
  ground at about 80%; Katz et al. (2023) §2 report that dead time and the processing
  filters reduce the effective number of transits by about 25%; and 8 RVS transits per
  star per year over DR3's 34 months predicts 22.7 against the published median of 18, a
  ratio of 0.79. The number describes a bright, isolated star, and neither crowding nor
  the fainter magnitudes are modelled.

Both steps are random and seeded, so the result is one realisation of a plausible epoch
list rather than a claim about which nights Gaia observed a particular star.

The raw response is cached under `cache_dir() / "gost"` in a file named for every
parameter of the query, and the cache is read before the network is touched, so a
benchmark that revisits a position pays for it once and a machine without a network
keeps working. When the request fails and nothing is cached, the `RuntimeError` names the
endpoint and the path the answer would have been written to, so the file can be fetched
elsewhere and copied in.

`draw_population` gives every drawn system a position (an ecliptic longitude uniform over
[0, 360) at the latitude it already drew, rotated to ICRS by `ecliptic_to_icrs`), and
`from_gaia_sb2` takes the catalogue's, so a population is ready for the forecast. The
benchmark uses it through `--cadence gost`, which replaces the drawn `n_transits` with
what the service and the two corrections leave inside the span of the release `--product`
names; a system left under `min_transits` is reported and not simulated, and a system
without a position raises by name.

```python
from albireo.gaia import gost_transits, rvs_transit_times_from_gost

transits = gost_transits(45.0, 20.0)          # 142 crossings, cached after the first call
bjd = rvs_transit_times_from_gost(transits, release="dr4", seed=0)
```

## Populations

`draw_population` composes published prescriptions: a Salpeter mass function, the
period, mass-ratio and eccentricity distributions of Moe & Di Stefano (2017) with the
twin excess defined as they define it, against the companions above $`q = 0.3`$, Pecaut &
Mamajek's (2013) dwarf sequence or the relations of Eker et al. (2018) for the stellar
quantities, the eclipse geometry with its eccentric-orbit factor, tidal synchronisation
below 15 days, and the RVS light ratio from the library's own continua and the radius
ratio. Two selections stand between the field and the sample. The mass function is
weighted as a magnitude-limited survey sees it: a star of absolute magnitude $`M_G`$ is
reached through a volume proportional to $`10^{-0.6 M_G}`$, and a pair through the volume
its combined light reaches, so the F and G dwarfs of Gaia's double-lined sample replace
the field's K dwarfs (`mass_weighting="volume"` turns this off). And `min_separation_kms`
keeps only the orbits whose largest separation, $`(K_1 + K_2)(1 + e)`$, a double-lined
analysis can resolve; the benchmark script uses 40 km/s, one and a half RVS resolution
elements. Under the FGK library box and these selections about half the pairs are twins
with $`q > 0.95`$, as a magnitude-limited double-lined sample is. `from_debcat` and
`from_gaia_sb2` build the same records from real catalogues,
filling only what the catalogue lacks; `query_gaia_sb2` fetches the Gaia DR3 double-lined
orbits with their masses (network). Gaia publishes no G_RVS, no transit count and no
light ratio for its double-lined sources, and the adapters say what they substitute.

One prescription does not extend to the hot box. Above the Kraft break `_draw_vsini` draws
an unsynchronised rotation from a log-normal around 40 km/s, which is an F-star rate; a
field A star rotates at 100 to 200 km/s, four to eight times the RVS resolution element.
A population drawn over `bosz2024-hot-rvs` is therefore optimistic about rotation, and its
recovery rates are an upper bound on what the same systems would give at A-star rates.
Systems below the 15-day synchronisation threshold are unaffected, because their rotation
follows the orbit.

## The benchmark

`run_benchmark` simulates every system once and runs the same epochs under each tier:

| tier | period | conjunction | elements | light fractions | label priors |
|---|---|---|---|---|---|
| `oracle` | known | known | e, ω held; K to 15% | injected | ±300 K, ±0.3 dex |
| `eclipsing` | known | known | free | injected | ±300 K, ±0.3 dex |
| `orbit` | known | scanned | free | measured against library templates | the library box |
| `blind` | searched | scanned | free | measured against library templates | the library box |

The tiers are declared through the pipeline's own vocabulary (`StarConfig`'s `period`,
`t_conj`, `ecc`, `omega`, `k` and `light = "measure"`), so the benchmark runs exactly what
a user runs, and every number in the report is the pipeline's own comparison against the
injected truth: semi-amplitude, element and velocity pulls against the quoted errors,
epoch-velocity residuals, the recovered spectra's correlation and equivalent widths
against the injected ones, label offsets, the measured light fractions. Every tier
declares the delivered grid's lag-one noise correlation to the analysis as the simulation
measured it (`noise_model="correlated"`, the default; `"diagonal"` takes the pixels as
independent), an instrument property declared like the LSF. The measured epoch velocities
are a product of the run and not only a statistic: `collect_velocities` gathers every
star's table with the injected velocity of each epoch into `velocities.csv` (one row per
system, tier, epoch and component), `summarize_velocities` pools the usable epochs per
tier, and the report draws each tier's systems phase-folded against the injected orbit
(`figures/rv_curves_<tier>.png`). The report (`report.md`, `rows.csv`, `velocities.csv`,
`summary.json`, figures) is regenerated by `write_report` at any time, and an interrupted
run resumes.

```bash
python scripts/gaia_rvs_benchmark.py --n 40 --jobs 8 --out bench/rvs
```

Background and references: [science overview](../science.md).

::: albireo.gaia

::: albireo.population

::: albireo.benchmark
