# Gaia RVS double-lined binaries in albireo: simulator, population, benchmark, report

Written 2026-09-09. Status: plan, being executed in the order of §7. Research notes from the
three fact-finding agents are filed beside this document as `research_*.md` once they arrive,
and every constant below that they correct is corrected here too.

## 1. What is being built, in one paragraph

Dom Rowan's `binaryspectra/notebooks/synthetic_rvs_spectra.ipynb` builds Gaia RVS spectra of a
synthetic SB2: two Korg-synthesised components at R = 11,500, Doppler-shifted to a Keplerian
orbit, combined at a flux ratio, resampled onto the 0.245 Å detector grid, given noise at a
per-detector-pixel S/N that follows from G_RVS, and resampled once more onto the delivered
0.1 Å grid so the noise is correlated the way the archive product's is. albireo gets the same
instrument model as a module (`albireo.gaia`), the component spectra from its own library
machinery (BOSZ 2024 in the RVS band) instead of Korg, a population generator that draws systems
matched to real eclipsing and non-eclipsing binaries (`albireo.population`), and a batch harness
that simulates, processes every system through the existing pipeline (disentangle, labels,
TODCOR, orbit), verifies every product against the injected truth, and writes a Markdown report
with figures (`albireo.benchmark`, driven by `scripts/gaia_rvs_benchmark.py`). Nothing outside
albireo is used at any stage.

## 2. Faithful replication: the notebook, step by step, and its albireo counterpart

| Notebook (binaryspectra) | albireo counterpart | Deviation, if any |
|---|---|---|
| `GAIA_RVS = dict(wavelength_range=(8460, 8700), resolution=11500, pixel_scale=0.245, output_pixel_scale=0.1)` | `albireo.gaia.RVS_DR3_MEAN` product preset (detector 0.245 Å, delivered 0.1 Å, 2400 samples) and `RVS_DR4_EPOCH` (detector 0.245 Å, delivered 0.025 nm, 961 samples, the DR4 epoch grid) | DR4 preset added; the notebook only has the DR3-shaped grid |
| `predict_grvs(g, rp)` two cubic polynomials in G−RP | `albireo.gaia.predict_grvs` | coefficients verified against the source paper (research note A) |
| `spectrum_snr_per_pixel(grvs, n_transits)` with the zero point 21.317, 4.4167 s, 3 CCDs, 0.02453 nm px, 24 nm band, 4.7 e−/px background, 10 AC px, 3.2 e− RON, window class at G_RVS = 7 | `albireo.gaia.rvs_snr_per_pixel`, constants in one frozen dataclass `RVSConstants` | none; provenance recorded on each constant |
| Component spectra: `Korg.synth(Teff, logg, M_H, alpha_H, vsini, R=11500, linelist, vmic)`, vacuum→air | `albireo.simulate.library_component` from `fetch_library("bosz2024-fgk-rvs")` (R = 20,000 upstream), Gray rotational kernel for v sin i, then a Gaussian of width sqrt(σ²(11,500) − σ²(20,000)) = 9.05 km/s applied by the simulator so the delivered resolving power is 11,500 exactly | **Synthesis source.** albireo synthesises nothing (roadmap non-goal); the published grid replaces Korg. Consequences: microturbulence fixed at 2 km/s (a second registry entry at 1 km/s is a one-line addition), [α/M] = 0, one line list (BOSZ's), Teff 4000–7000 K until the ATLAS9 shards are registered. The wavelength scale is **vacuum**, as Gaia's is; the notebook works in air |
| `K1`, `K2` from masses, `P`, `e`, `i` | `albireo.population.semi_amplitudes` (same Kepler expression; checked: 87.7 / 95.0 km/s for the notebook's system) | none |
| `init_rv_orbit`: `rv_model` with `M = 2πt/P − phi0`, secondary as `−K2` | `albireo.simulate.OrbitParams(t_peri = phi0 P / 2π, omega, k=(K1, K2))`; the secondary uses `omega + π` | identical velocities (checked to the printed precision at the notebook's epoch of largest separation) |
| `shift_rv`: relativistic factor `sqrt((1−β)/(1+β))` by `np.interp` | log-grid translation by `artanh(β)`, linear 2-tap shift | same Doppler law; interpolation on a log grid rather than a linear one |
| Combine `a s1 + b s2`, `a = 1/(1+fr)`, `b = fr/(1+fr)` | light fractions `(a, b)` in `simulate_dataset` | none |
| Resample onto `np.arange(8460, 8700, 0.245)` by `np.interp` | pixel-integral rebin onto the same detector grid | flux-conserving rebin rather than point interpolation; identical when the model grid is finer than the detector |
| `add_noise_at_wavelength(snr, ref=8580, window=50, shot_noise=True)`: σ_p = (F_ref/snr) sqrt(F_p/F_ref) | `InstrumentSpec(shot_noise=True, snr_window=(8555, 8605))`, new in `simulate_dataset` | none; `ivar` records the σ actually used, as the notebook's `err` column does |
| Resample the noisy detector spectrum onto `np.arange(8460, 8700, 0.1)` by `np.interp(fill=1)` | `albireo.gaia.deliver`: the same linear interpolation, with the variance propagated through the interpolation weights and the lag-one noise correlation recorded in the truth | delivered pixels outside the detector grid (at most one, at the red end) are masked (`ivar = 0`) rather than set to 1 |
| Cross-correlation against the primary template, full band and between the Ca lines; `ccf.model` peaks | `albireo.todcor` with one template is the 1-D CCF; the Ca-free windows are `Analysis.mask` ranges | the notebook's Gaussian-peak model is not reproduced; albireo refines the χ² minimum exactly |
| `spec.todcor(primary, secondary)` and the slices through the 2-D peak | `albireo.todcor` with both templates (Zucker & Mazeh 1994, exact fractional shifts) | none in the estimator; albireo's errors are the Zucker (2003) curvature errors |
| `bt.rv1`, `bt.rv2` as truth | `SimulationTruth.velocities` | none |

Two things the notebook computes that albireo will now also compute: G_RVS from G and G−RP, and
S/N from G_RVS and the transit count. Two things the notebook does not do that the benchmark
needs and albireo will add: a realistic transit cadence, and the DR4 epoch product.

## 3. What is missing from albireo (to be built), and what is deliberately not

Built in this work:

1. `simulate_dataset`: photon-counting noise (`InstrumentSpec.shot_noise`, `snr_window`), and a
   `noise_sigma` record on the truth. Without it the notebook's noise model cannot be reproduced.
2. `albireo.gaia`: the RVS instrument (constants, G_RVS, S/N, LSF), the two product presets, the
   deliver step (detector grid → delivered grid with propagated variances), the cadence model,
   and `simulate_rvs_dataset`, the one-call equivalent of `SyntheticSB2(...).generate_binary_spectra()`.
3. `albireo.population`: main-sequence binaries with masses, radii, temperatures, gravities,
   absolute G and G−RP, orbital elements, inclinations, an eclipse test, tidal circularisation and
   synchronisation, the RVS-band light ratio from the library's own continua and the radii, and
   catalogue adapters (Gaia DR3 `nss_two_body_orbit` SB2 and eclipsing solutions through TAP,
   DEBCat rows from a file) so that a population can be "the real ones".
4. `albireo.benchmark`: build the stars, run them through `run_pipeline`, verify against truth,
   aggregate, and write the report. Resumable.
5. Pipeline extensions the benchmark needs and every user benefits from:
   - `ComponentConfig(light="measure")` on the `period = "search"` route: the light fractions
     are taken from the bootstrap correlation's global amplitudes, recorded and flagged. The rule
     "declared, never defaulted" is kept: a measurement is not a default.
   - the truth block gains component-spectrum fidelity (RMS and correlation of `d_hat − d_true`
     over the line-bearing pixels; equivalent widths of the Ca II triplet and of the metal-line
     windows), per-epoch velocity pulls, `e`, `ω`, `t_conj`, `γ` and `K` pulls against the
     quoted errors, and the light ratio measured by the label fit against the injected one.
   - `find_period`: the frequency grid oversamples the baseline (10 per 1/T) rather than using
     a fixed count, because a 66-month Gaia baseline with 106-minute transit pairs is not
     resolved by 20,000 frequencies.
6. `examples/14_gaia_rvs.py`: the notebook's own system, end to end, with the TODCOR
   comparison the notebook prints (73.93 / −117.32 km/s 1-D, 79.69 / −122.08 2-D, against
   76.05 / −119.88 true).
7. `scripts/gaia_rvs_benchmark.py`: the report generator.

Flagged, not built here (each is a separate decision):

- **Synthesis.** No Korg, no Julia. A user who wants the notebook's exact templates can build a
  `SpectralLibrary` from Korg output offline and pass it as `library=`; the container is
  generic. This is the roadmap's standing non-goal and the benchmark does not reopen it.
- **Hot components.** BOSZ publishes ATLAS9 models above 8000 K; registering an RVS-band A/B-star
  box is a registry entry plus a download, and the population generator already refuses a Teff
  outside the library rather than clamping it. Until then the eclipsing population is the
  FGK part of the catalogues, and the report says what fraction was excluded.
- **Correlated noise in the fit.** The delivered 0.1 Å product has 2.45 delivered pixels per
  detector pixel, so a diagonal `ivar` overcounts the information by that factor. The AR(1)
  chain (D34) is the model for it but the façade and the pipeline do not expose it. The
  benchmark measures the consequence (pull widths, `z_rms`) on both products; exposing
  `noise = "ar1"` in the pipeline is the follow-up if the DR4 product shows it matters.
- **A real scanning law.** The cadence model is an approximation of Gaia's (spin, two fields of
  view, precession, ecliptic-latitude dependence of the transit count); real transit times from
  GOST or the `scanninglaw` package can be passed in as `obs_times`. The research note says which
  is cheapest to integrate later.
- **DR4 reader and DataLink client.** Deferred to the release, as recorded in the roadmap. The
  simulator writes datasets in memory; `write_epochs` puts them on disk as FITS through the
  existing writer, and the future reader is expected to read those files.

## 4. Design

### 4.1 `albireo.gaia` (instrument)

```python
RVS_BAND = (8460.0, 8700.0)          # vacuum Angstrom
RVS_RESOLVING_POWER = 11_500.0
RVS_DETECTOR_STEP = 0.245            # Angstrom per detector pixel (0.02453 nm)

@dataclass(frozen=True)
class RVSConstants: zero_point=21.317, exposure_s=4.4167032, ccds_per_transit=3,
    pixel_nm=0.02453, band_nm=24.0, background_e=4.7, ac_pixels=10, read_noise_e=3.2,
    window_class_limit=7.0     # each with its source in the docstring

@dataclass(frozen=True)
class RVSProduct: name, delivered_step, delivered_n, detector_step, snr_window, ...
RVS_DR3_MEAN  = RVSProduct("dr3-mean",  0.1,   2400, ...)   # np.arange(8460, 8700, 0.1)
RVS_DR4_EPOCH = RVSProduct("dr4-epoch", 0.25,  961, ...)    # 846.0 + 0.025 k nm, k = 0..960

predict_grvs(g_mag, rp_mag) -> array          # NaN outside -0.15 < G-RP < 1.7
rvs_snr_per_pixel(grvs, n_transits=1) -> array
rvs_lsf() -> LSF                              # LSF.from_resolution(11_500)
transit_times(n_transits | grvs, *, start, span_days, ecl_lat_deg, seed) -> bjd array
deliver(dataset_on_detector_grid, product) -> (Dataset, delivery truth)   # interpolation + variance propagation + lag-1 record
simulate_rvs_dataset(components, grid, *, bjd, orbit|velocities, light_fractions,
                     grvs|snr, product=RVS_DR4_EPOCH, library_resolving_power=20_000,
                     shot_noise=True, seed) -> (Dataset, RVSTruth)
```

The dataset comes back declared `frame="barycentric"`, `medium="vacuum"`, instrument `"RVS"`,
`lsf_sigma_kms = 11.07` on every epoch. `RVSTruth` wraps `SimulationTruth` with the detector-grid
epochs, the S/N used, the delivered-grid noise σ, and the lag-one correlation.

Model grid for the simulation: `LogGrid.covering` the band at `dv = 2 km/s` with a margin of
`max|v| + 5σ_LSF + v sin i`. BOSZ r20000 samples at 6.46 km/s; the library is projected onto the
2 km/s grid by the flux-conserving rebin, exactly as `Template.from_library` does for the fit,
so the truth and the templates share one rendering path.

### 4.2 `albireo.population`

```python
@dataclass(frozen=True)
class BinarySystem:   # the truth of one system, JSON-able
    name, m1, m2, r1, r2, teff1, teff2, logg1, logg2, mh, vsini1, vsini2,
    period, ecc, omega, t_peri, incl, gamma, k1, k2,
    eclipsing (bool), light_ratio_rvs, g_mag, g_rp, grvs, n_transits, source ("parametric" | catalogue id)

draw_population(n, *, kind="eclipsing" | "sb2" | "mixed", library, seed, mass_range, period_range,
                grvs_range, ...) -> list[BinarySystem]
from_gaia_nss(rows) / query_gaia_nss(kind, limit) [network] -> list[BinarySystem]
from_debcat(path) -> list[BinarySystem]
semi_amplitudes(m1, m2, period, ecc, incl) -> (k1, k2)
eclipses(r1, r2, a, ecc, omega, incl) -> bool
rvs_light_ratio(library, labels1, labels2, r2_over_r1) -> float   # continua at the RVS band
```

Parametric draws: primary mass from a Kroupa IMF slope over the library's Teff box, mass ratio
and period and eccentricity from Moe & Di Stefano (2017) at the primary's mass, circularised
below the tidal period, synchronised rotation for P < 15 d and a field v sin i distribution above,
isotropic inclination, uniform ω and phase, γ from a Gaussian of 30 km/s. Eclipsing draws are
rejection-sampled on the geometric condition; non-eclipsing draws on its complement. Teff,
radius and photometry come from a main-sequence relation object (Pecaut & Mamajek table or
Eker et al. 2018 power laws, per research note B). G_RVS from the absolute magnitude and a
distance drawn so that the G_RVS distribution matches Gaia's SB2 sample (G_RVS ≤ 12, rising
to the faint end); the transit count from the cadence model.

The catalogue route makes the "matching true binaries" claim literal: rows from Gaia DR3
`nss_two_body_orbit` (SB2 and eclipsing solutions) or DEBCat supply P, e, ω, K1, K2 (or masses
and radii), Teff and G, and the generator fills only what the catalogue lacks.

### 4.3 `albireo.benchmark`

Knowledge tiers (what the analysis is told; the truth is never told):

| Tier | Period | t_conj | e, ω | K | Light fractions | Labels prior |
|---|---|---|---|---|---|---|
| `oracle` | Known(P, tight) | Known | Known | Known(±15%) | injected | ±300 K |
| `eclipsing` | Known(P, 1e-4 P) | Known(primary eclipse) | free | Between | injected (light-curve ratio) | ±300 K |
| `orbit` | Between(0.97 P, 1.03 P) (an NSS-style prior) | scan | free | Between | measured by bootstrap | library box |
| `blind` | "search" | scan | free | Between | measured by bootstrap | library box |

Default assignment: eclipsing systems run `eclipsing` and `blind`; non-eclipsing run `orbit` and
`blind`; `oracle` is the ceiling and runs on a subset.

Per system and tier, the verification records: K1, K2, P, e, ω, t_conj, γ differences and pulls;
per-epoch TODCOR velocity residuals and pulls (both components); orbit-predicted velocity RMS;
component fidelity (RMS, correlation, Ca II triplet and metal-window equivalent widths of
`d_hat` against the injected spectrum, on the fit grid); label offsets (Teff, log g, [M/H],
v sin i) and the fitted light ratio; detection (fraction of epochs with Δχ² > 25 for the
secondary); flags; wall time per stage.

The report (`report.md` + PNG figures + `systems.csv` + `metrics.json`): the population as drawn
(G_RVS, P, K1+K2, q, light ratio, n_transits); recovery of K and q against G_RVS, n_transits,
maximum separation over the LSF width, light ratio and Teff contrast (median and 16–84%
bands, per tier); pull distributions with the fraction inside 1σ and 2σ; per-epoch velocity
precision against per-epoch S/N with the TODCOR formal error overlaid; spectrum fidelity
against total S/N; label recovery; the blind tier's period-recovery rate and its aliases;
failure and flag rates by cause; and the timing table. Every figure is captioned with the
machine and the settings.

### 4.4 Processing settings for RVS (recorded as `gaia.rvs_analysis()`)

`dv_kms = 3.0` (3.7 model pixels per LSF σ; the pipeline's default would take the DR4 native
8.7 km/s, 1.3 px/σ), `k_max` from the population's ceiling (250 km/s), `ecc_max = 0.9`,
`v_zero_range = 150`, `v_range = 400`, `region = None`, `mask = ()` by default with the Ca-free
windows `[(8490, 8506), (8534, 8552), (8654, 8672)]` available as a variant, `plots = True`,
`sample = False` (NUTS is a per-system opt-in through the tier).

## 5. Verification of the replication itself (tests)

- `tests/test_gaia.py`: `predict_grvs` reproduces the notebook's 10.54 for its DR3 source;
  `rvs_snr_per_pixel(10.54) = 13.1`; the delivered grids have 2400 and 961 samples at the
  documented steps; the detector-grid noise has the σ the notebook's `err` column records
  (shot noise scaling, reference window); the delivered noise variance matches the propagated
  one to 1%; the delivered lag-one correlation matches the analytic value; the uncovered pixel
  is masked; the dataset declares vacuum, barycentric, R = 11,500; a closed loop (marked slow)
  recovers K1, K2 of the notebook's system through `Disentangler` to 1% at S/N 40 and the
  TODCOR velocities of the widest-separation epoch to within their errors.
- `tests/test_population.py`: reproducibility; `semi_amplitudes` gives 87.7 / 95.0 for the
  notebook's masses; the eclipse test agrees with the geometric formula on edge cases;
  light ratios lie in (0, 1) and follow the radius ratio; draws respect the library box; the
  catalogue adapters map every column.
- `tests/test_benchmark.py`: a two-system fast run writes every product and the report; the
  verification metrics on a perfect recovery are zero; resumption skips finished systems.
- `tests/test_simulate.py`: the shot-noise option (σ ∝ sqrt(flux), reference window).

## 6. Documentation

`docs/api/gaia.md` (gaia, population, benchmark), `docs/tutorials/gaia-rvs.md` (the notebook's
system, then a ten-system benchmark), a dated section in `docs/benchmarks.md` with the first
report's numbers, `internal/design.md` ledger rows D60 (RVS simulator and shot noise), D61
(population and benchmark harness, and the `light = "measure"` decision), roadmap item 7
amended ("the half that can be built now" is now built), `CHANGELOG.md`.

## 7. Order of work

1. `simulate_dataset` shot noise; `albireo.gaia` constants, S/N, products, deliver, simulate;
   tests; `examples/14_gaia_rvs.py` reproducing the notebook's system and TODCOR numbers.
2. Cadence model (after research note A); `albireo.population` with the parametric generator
   and the Gaia NSS and DEBCat adapters (after research note B); tests.
3. Pipeline extensions (measured light, extended truth block, `find_period` grid);
   `albireo.benchmark` with the report; tests; `scripts/gaia_rvs_benchmark.py`.
4. Run the benchmark on this desktop (both products, all tiers, of order 100 systems), write
   the docs and the ledger from the numbers, and file the report under `docs/benchmarks/`.

## 8. Status at the end of 2026-09-09 (D62 additions of 2026-09-10 at the end)

Built: everything in section 3 items 1 to 7 and section 4, with the tests of section 5 and the
documentation of section 6 except the benchmarks.md section, which waits for the first real run.
Deviations from the plan above, each for a reason found on the way:

- The DR3 product has 2401 samples inclusive (the notebook's arange stops at 869.9 nm); the
  preset follows the archive and the docstring says so.
- The resolving power can be drawn per transit from the in-flight values per CCD row
  (Cropper et al. 2018 Table 6), with the analysis declaring the nominal 11,500. Not in the plan;
  the research pass showed the spread is 7 percent.
- The cadence model follows the measured structure (visibility periods of 1.48 transits on the
  106.5-minute / 4.23-hour ladder, at least four days apart, counts from the archive's
  distribution and its latitude dependence) rather than the "rotations per visit" sketch.
- The orbit tier declares the period as a Gaussian at a Gaia solution's precision, not a
  3 percent range: a range that wide leaves the phase scan meaningless over a 66-month baseline.
- light = "measure" works on any route with a library, not only the search route; the
  correlation stage was factored out of the bootstrap for it.
- Four defects found by the harness and fixed in albireo: the label and bootstrap stages sliced
  the library in air before the vacuum conversion; the correlation's free-amplitude solve raised
  on coincident templates; the bootstrap gave every component the same midpoint template; the
  BOSZ citation said A171 (it is A197).
- The first smoke run (six systems, tiers oracle and blind, fast mode) taught three more things.
  (1) The population: the bare mass function put the primaries at the library's 4000 K floor,
  where the box forced every companion to be a twin (six of six at q > 0.95); the draw is now
  weighted as a magnitude-limited survey sees it (10^(-0.6 M_G), with the pair's combined
  light) and takes a velocity floor on (K1 + K2)(1 + e) for the double-lined selection (40 km/s
  in the script). The twin excess is also now defined as MDS17 define it, against the companions
  above q = 0.3, so a q_min cut does not inflate it. About half the pairs remain twins.
  (2) Alike components: a per-epoch correlation lands in either exchange-symmetric minimum at
  random for a twin pair, which collapsed a K of 44 km/s to 16 in the table (oracle tier) and
  sent the blind tier's bootstrap to K of 900 and 2700 km/s, from which the disentangling grid
  could not be built. `rvorbit.reassign_by_orbit` re-assigns the epochs by an orbit; the
  velocities stage applies it with the disentangling's orbit, and the bootstrap applies it
  with each candidate orbit, adds the peaks of a swap-invariant periodogram (of the magnitude
  of the relative velocity, with their doubles) to the candidates, and where a component's
  semi-amplitude still falls outside the declared range it keeps the bootstrap's period and
  conjunction and searches the range for K, as the known-period route does (a first guard that
  clipped the value and declared it known held a faint secondary's K at the floor). On the rerun
  the twin at P = 5.02 d bootstrapped to P 5.0242 d and K 67.7 and 69.7 km/s with three epochs
  re-assigned.
  (4) A secondary with 5 percent of the light (q 0.54, a K dwarf beside an F star): the
  disentangling recovered its K to 0.6 percent under the oracle tier while the correlation table
  gave zero for it (the secondary's template sits on the primary's lines), so the report now
  tabulates and plots the semi-amplitudes from both sources, and masks the table's argument of
  periastron below e = 0.05, where it is undefined.
  (5) The full run's first pass found the same system failing on every route where K is a
  range (eclipsing, orbit, blind): the evenly spaced starts over 2 to 250 km/s put its 23 km/s
  primary at 85 km/s and the disentangling settled with both semi-amplitudes at the floor. A
  range prior now starts from a template table fitted at the declared period (`_k_starts`), with
  a component the table could not measure started from its nearest measured neighbour (above
  it for a later, lighter star, below it for an earlier, heavier one; the first rule, 1.5 times
  the largest measured one, put a Gaia pair's unmeasured primary on the 250 km/s bound and the
  run failed with numpyro's "cannot find valid initial parameters", so starts are held strictly
  inside the range and `Between.start` refuses a bound); the pass was stopped and the
  eclipsing, orbit and blind tiers rerun.
  (6) Twins at q = 0.98 with measured light fractions came out in the other order on three
  range-K runs (a fit as good as the declared order; the table 178 km/s from the truth as named).
  The truth comparison now tries both orders on the velocities, takes the exchange when it lowers
  the rms threefold, flags it and judges the recovery in that order; the report counts them.
  (3) `find_period` now oversamples the baseline ten times instead of using a fixed 20,000
  frequencies, which a 66-month cadence needs.
  (7) The Gaia DR3 population (14 orbits, 10 to 25 transits each) separated the disentangling
  from its starting values: with the orbit declared (oracle tier) every system's semi-amplitudes
  came back within 4 percent and the medians at 0.3 and 0.2 percent, on 10 to 25 epochs; with a
  semi-amplitude left as a range, the template table's orbit at 10 to 16 epochs is unreliable (a
  twin table collapsed by exchanges gave starts of 37 and 32 km/s for a 102/104 pair, a broad
  secondary gave 228 km/s for 35) and the disentangling then settles in a local minimum (half
  the semi-amplitudes, or one at the floor). Not fixed today; the next step is a coarse scan
  over the semi-amplitudes through the existing conjunction scan before L-BFGS on the range
  routes, and the report records the failure rates as they are.
- Walls: in fast mode (40 disentangling and 60 label steps) a star took 160 to 340 s with three
  workers; at 100 and 80 steps with eight workers 470 to 790 s each, about 80 s of wall per star
  for the batch.

Open, in the order they should be taken: the first real run and its benchmarks.md section; the
ATLAS9 box above 7000 K; the AR(1) model on the delivered product through the facade; the DR4
grid against the release; the scanning law's phase.

### D62 (2026-09-10, second session)

The user asked for the measured epoch velocities as a first-class product and for the open
items to be worked. In order: (a) the per-epoch table was already written per star; the
benchmark now gathers it (`collect_velocities`, `velocities.csv`, the report's "Epoch
velocities" section and the `rv_curves_<tier>` figures) and `result.json` carries the
disentangling's Keplerian and the injected velocity per epoch. (b) The delivered grid's
lag-one correlation is declared to the disentangling (AR(1)) and to TODCOR (the sandwich
covariance); the benchmark declares it to every tier. (c) The range-K failures: a scan over
the semi-amplitudes alone failed at the phase located at wrong semi-amplitudes; the joint
(K, phase) hierarchical scan on a coarser declaration finds the twin basin; two guards
(the static minimum refused, a component moved only when holding it costs `hold_nats`)
followed from the first rerun's regressions on an eccentric contact pair and a 7-percent
secondary; scanning (e, omega) as well was tried on one eccentric system and not adopted;
the template table's orbit now starts a free eccentricity. (d) The temperature bias is the
label fit's, not the disentangling's: the one BOSZ gap forced the simplex interpolant,
whose kinks stop L-BFGS 20 to 150 K from a perfect spectrum; the gap is filled by
interpolation along [M/H] and the cubic applies (fits within about 30 K). Scratchpad
scripts: kscan_timing.py, kscan_check{,2,3}.py, teff_bias.py, label_bias{,2}.py. Reruns:
gaia3 (old library, first scan), then field3 and gaia4 with the final code.
Outcome (reports archived under field/ and gaia/, replacing the first run's): Gaia orbit tier
median K errors 13/25 -> 3.5/4.2 percent, within 5 percent on both 43 -> 64 percent of systems,
exchanged 21 -> 7 percent, epoch-velocity pulls 4.5/7.7 -> 1.5/1.3; eclipsing 16/34 -> 3/14;
oracle epoch-velocity pulls 1.5/1.4 -> 1.1/0.94 (the AR(1) errors); blind still the period
search's (3 of 14). Field: within noise on K (the 28-100 epoch tables were good seeds), blind
secondary 12 -> 5.4 percent, blind pulls 3.8/3.1 -> 1.3/1.6, period search 11 -> 12 of 19.
Label fit: eclipsing Teff 257/205 -> 28/74 K (Gaia); oracle tiers prior-limited (the fit
"learned nothing" in 8 of 14 and 15 of 19 runs). Out-of-memory casualties (31 GB machine,
a test suite run beside the benchmark) were rerun; the test suite must not run beside a
benchmark on this box.

## 9. D63 (2026-09-10 to 11): the open items of the second run, measured first

The user asked to continue on the second run's open items with research and routine
implementation delegated to Opus agents and the design held in the main session. Every item
was measured on the second run's archive before anything changed; the measurements are the
research notes `d63_*.md` beside this plan.

- **The period search** (d63_period_search_diagnosis.md, d63_period_search_acceptance.md):
  lost at candidate generation because the classical Lomb-Scargle has no constant term and
  the clumped Gaia sampling makes the window's mean large; the floating-mean GLS with
  twenty candidates, its two-harmonic form, the first component alone and the range filter
  on the candidate orbits take the archived oracle tables from 23 to 28 of 32.
- **The faint secondary** (d63_faint_secondary_scan.md): lost by the product grid's coarse
  sampling of the primary and a single-component guard; the scan now takes the axes in turn
  with a joint guard.
- **The light fractions** (d63_light_fractions.md, d63_light_after_labels.md,
  d63_light_profile.md): wrong by more than a factor 1.5 on 8 of 33 systems; the marginal
  likelihood cannot measure them (an exact invariance with the smoothness hyperparameters)
  but a profile over the light-equivalent amplitude between two scan passes rescues the
  worst case, and is what the fit now does.
- **The step budget** (d63_step_budget.md): budget-limited; 300 steps for the third run.
- **The diverged oracle fit** (d63_edge_pinned_table.md and the wp-k trace): a numerical
  failure of the marginal at tau of 2e19 accepted by a line search; guarded now, with the
  correlation stage and the pipeline made unable to write the fake table it produced.
- **Real epochs** (d63_gost_endpoint.md): the GOST client and the `gost` cadence.
- **The hot box** (research_bosz_hot_box.md): registered, not downloaded.

Outcome of the third run: the Gaia blind tier finds the period on 7 of 14 against 3 and its semi-amplitude medians fall from 24 to 6 percent; the orbit tier from 3.5 and 4.2 to 1.0 and 2.2 percent with the 7-percent secondary recovered; the oracle tier unchanged and calibrated; the field population unchanged on the orbits and worse on the table (three lost components, traced to the exchange on noise, the ML-II shrinkage and an unlearned zero point, all guarded since); the label fit's temperatures worse under the longer budget, ML-II at its optimum shrinking the spectra. Archived as gaia_d63/ and field_d63/; the D63 subsection of docs/benchmarks.md has the tables.
