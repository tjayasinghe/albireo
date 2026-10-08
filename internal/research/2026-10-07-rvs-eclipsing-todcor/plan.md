# Gaia DR4 eclipsing binaries in the RVS: epoch velocities by TODCOR on a simulated population

Written 2026-10-07. Status: executed on 2026-10-07 after the user approved every
recommendation of section 13; section 15 records what was done and where it departs from
sections 3 to 9. Before that, of sections 3 to 11 only the corrections of section 10.1 were
implemented. The evidence is in three notes beside this page, cited by
letter: **A** the Gaia archive and the scope of DR4, **B** the literature on
eclipsing-binary populations, **C** measurements made on the present code. The numbers of
section 2 are measured (A, C) or quoted from a source that A or B names. Everything from
section 3 on is design.

## 1. Question, scope and products

**Question.** Gaia DR4 (2 December 2026) publishes one RVS spectrum per field-of-view
transit. For the detached eclipsing binaries among those sources, which per-transit
velocities of both components can be measured by correlation against library templates
alone, with what errors, and which orbits and masses follow when the eclipse ephemeris is
taken from the photometry? The answer is wanted as a function of G_RVS (the S/N of a
transit), of the spectral types, and of the quantities that decide whether two sets of lines
are measurable: the flux ratio, the rotation, the line separation and the number of transits.

**Why no disentangling.** The disentangling route of the recorded Gaia RVS benchmarks costs
410 to 790 s per system. The correlation costs about 0.4 s per system (C3), so a population
of 10^4 systems, three template declarations each, is a run of about two hours on this
desktop (section 8).

**In scope.** Detached pairs of main-sequence and subgiant stars; effective temperatures of
3200 to 10,000 K (to 12,000 K as an extension); G_RVS from 6.0 to 13.5; the DR4 epoch
product (961 samples, 846 to 870 nm); velocities by `albireo.todcor` against BOSZ templates;
orbits from the velocity tables at the photometric period.

**Out of scope.** Contact and semi-detached systems (about 55 percent of the catalogued
candidates at these magnitudes, and about 25 percent of those with periods of a day or
more; A 2d, B 1), giants, taken here as stars below log g = 3.0 (2.4 percent of detached
primaries are giants; B 1.4), stars hotter than the library, and the period search, since an
eclipsing binary has a photometric period.

**Products.**

1. A population of simulated systems with the weights that map it onto the Gaia DR3
   eclipsing-binary catalogue, and the figure that compares the two.
2. One table per system and one per epoch, for every template declaration, with the
   injected and the measured values.
3. A report page in the documentation with sixteen figures, each in a light and a dark
   rendering, and the binned tables behind them.
4. A fitted recovery function (probability of a double-lined orbit as a function of a
   predicted S/N of the secondary) and a forecast of the number of DR4 systems that yield
   one.
5. The additions to the package that the experiment needs, with tests (section 10).

## 2. Evidence gathered before planning

### 2.1 What DR4 delivers (A 1)

- 6.9 x 10^9 epoch spectra for the 312 million sources of the spectroscopic pipeline, also
  for double-lined transits. The pipeline's own per-transit velocities are published to a
  gate magnitude of 14, and it searches for double lines to 12 (11 in DR3, with the
  conditions 15 < |V1 - V2| < 500 km/s and a brightness ratio above 0.2). The gate
  magnitude is the G_RVS predicted from G and G_RP, the magnitude used throughout this plan.
- A source with double lines in ten or more transits (`rv_assumed_sb2`) has no combined
  velocity, broadening or mean spectrum in DR4. Its epoch spectra are published, each the
  mean of CCD spectra that were normalised one by one and not renormalised afterwards.
- `vari_eclipsing_binary` has 2,392,500 candidates (2,184,477 in DR3), with the period, the
  reference time, the eclipse phases, the light-curve class of Mowlavi et al. (2023) as a
  column (`sample_classification`), and a flag that compares the photometric period with
  the periodogram of the velocities.

### 2.2 The catalogue at the magnitudes of the RVS (A 2 to 4)

- **Counts.** 3900 candidates brighter than G_RVS = 10, 10,359 brighter than 11, 27,252
  brighter than 12, 64,843 brighter than 13 and 144,865 brighter than 14, which is 0.35
  percent of all sources at every magnitude.
- **Types.** By colour, at G_RVS below 12: 41 percent F, 29 percent G, 20 percent K, 8
  percent A, 1.3 percent M and 0.5 percent bluer than BP - RP = 0. Half of the candidates
  are within 10 degrees of the Galactic plane, so the colours are reddened and the hot
  shares are lower limits: `teff_gspphot`, which three quarters of them have, puts 27
  percent between 7250 and 10,000 K and 10 percent above at G_RVS 10 to 12. The FGK
  library box covers about three fifths of the bright catalogue.
- **Periods.** Median 1.27 d at G_RVS below 12, 80 percent between 0.37 and 6.1 d, 6
  percent above 10 d.
- **Light curves.** 44 percent in the classes that Mowlavi et al. call wide (detached, or
  semi-detached under some conditions), 44 percent tight, 13 percent unassigned. The wide
  share is 11 to 16 percent from 0.25 to 0.63 d and 56 to 66 percent from 1 to 6.3 d.
  Primary eclipse depths: 0.07, 0.27 and 0.72 mag at the 10th, 50th and 90th percentiles.
- **Rotation.** The 38 percent with a published `vbroad` have a median of 82 km/s, 17 to
  176 km/s between the 10th and 90th percentiles.
- **Transits.** `rv_nb_transits` in DR3: 10, 20 and 33 at the 10th, 50th and 90th
  percentiles. DR4 spans 1.94 times as long.
- **What DR3 extracted.** 5376 double-lined orbits, none fainter than G = 12, and 155
  combined eclipsing and spectroscopic solutions. Of the 5665 candidates brighter than
  G = 11, 483 have a double-lined orbit, and two thirds of the candidates brighter than
  G_RVS = 10 have no velocity at all.
- **Gaia's own per-transit scatter**, from the published errors of single dwarfs of 5500 to
  6500 K with `vbroad` below 20 km/s or absent: 0.19, 0.36, 1.07 and about 5.4 km/s at
  G_RVS 6, 8, 10 and 12, and 2.5 to 5 times larger at `vbroad` of 50 km/s and above. The
  values level off at 0.17 to 0.20 km/s at the bright end.

### 2.3 Detached eclipsing binaries in magnitude-limited samples (B)

- Detached systems are 44 to 47 percent of the catalogued eclipsing binaries at V below 12
  to 13 (ASAS-SN), and 75 percent of those with periods of a day or more. Three
  magnitude-limited samples agree on their periods: 1.1 to 1.25, 2.4 to 2.6 and 5.0 to
  6.1 d at the 16th, 50th and 84th percentiles.
- Gaia DR3 lists 64 percent of the ASAS-SN detached systems brighter than V = 13: 75
  percent at 1 to 1.8 d, 54 percent at 5.6 to 10 d and 34 percent at 18 to 56 d.
- Primaries: 84 percent main sequence, 14 percent subgiant, 2.4 percent giant (Rowan et al.
  2022). Every published forward model takes its radii from evolutionary models and
  selects eclipsing systems by geometry.
- 79 percent of systems below 10 d rotate synchronously (Lurie et al. 2017). Eccentric
  orbits begin at 2.5 d for stars cooler than 7000 K and at 1.6 d for stars hotter than
  10,000 K (IJspeert et al. 2024). 96 percent of binaries below 2.9 d and 34 percent above
  12 d have a tertiary (Tokovinin et al. 2006).
- In a partial eclipse the centroid of the eclipsed star's lines moves by 0.13 to 0.44 of
  its v sin i, and about a fifth of the transits of a typical detached system fall in an
  eclipse. Chromospheric emission in the Ca II triplet moved RAVE velocities by up to
  10.5 km/s (Zerjal et al. 2013).
- Masses to 1, 3 and 10 percent need both semi-amplitudes to 0.45, 1.3 and 4.5 percent at a
  mass ratio near one.

### 2.4 The present code (C)

- **Corrected on 2026-10-07 (section 10.1).** The shipped simulation path compiled 59 array
  programs per system and kept them: 1.36 s and 87 MB per system. It now takes 0.13 to
  0.17 s, compiles nothing after the first systems, and its memory is flat (0.74 GB after 20
  systems and after 60). Every delivered spectrum of the 24 systems compared is unchanged
  bit for bit. A pair of templates takes 0.012 s where it took 0.16 s. Outside `jit` the
  Kepler solver compiled its loop on every call, 29 to 40 ms and 1.9 MB each; it is now
  compiled once per array shape, with the same values. `todcor` compiles nothing per system
  either. The orbit fit still compiles for each new number of usable epochs (C7).
- **Cost of the chain.** Correlation with held light fractions 0.38 s per system (11 ms per
  epoch), with measured light fractions 0.99 s, orbit 0.29 s. Four processes started
  together give 3.3 times the throughput of one, and 3.6 times with one thread each.
- **First look**, F and G dwarfs with mass ratios above 0.4 and the injected templates: the
  median epoch error of the primary is 0.3, 1.0, 1.6, 3.7 and 13 km/s at G_RVS 7, 9, 10, 11
  and 12, and both semi-amplitudes are within 5 percent for 19 of 20 systems at G_RVS 10,
  18 of 20 at 11, 10 of 20 at 12, 6 of 20 at 12.5 and 1 of 20 at 13. The transition lies
  between G_RVS 11.5 and 13, where DPAC stops its own search for double lines.
- A linear fit that takes the shape of the velocity curve from the ephemeris recovers the
  same systems at 5 percent and a few more at 10 percent (15 against 8 of 20 at G_RVS 12.5).
- `todcor` raised no error in 380 system runs down to a S/N of 1.3 per pixel. The only
  failures were orbit fits with too few usable velocities.
- The simulation is near Gaia's own scatter before any adjustment. The D66 sweep measured
  26 km/s divided by the S/N for a slowly rotating F star holding 0.625 of the light, which
  is 16 km/s divided by the S/N for the star alone if the error scales with the inverse of
  the light fraction. The DR3 values of section 2.2 correspond to 16, 19 and 25 km/s divided
  by the S/N at G_RVS 8, 10 and 12 (A 3). The single-star run of criterion 5 measures it.

## 3. Design of the experiment

### 3.1 The design sample and its weights

A sample that follows the catalogue's magnitude distribution between G_RVS 6 and 13.5 has
57 percent of its systems in the faintest magnitude, where little is recovered, and 4
percent brighter than 10. The design sample is therefore stratified, and the catalogue
enters as weights.

- **Strata.** Fifteen bins of G_RVS of 0.5 mag from 6.0 to 13.5, by four classes of the
  primary's temperature: K (4000 to 5300 K), G (5300 to 6000 K), F (6000 to 7250 K) and A
  (7250 to 10,000 K). 130 systems per cell, 7800 in all. The extension adds a fifth class,
  late B (10,000 to 12,000 K), for 9750.
- **Inside a stratum** the systems follow the population model of section 4 with no further
  selection, so the conditional recovery in a cell is that of a realistic mixture of mass
  ratios, periods, radii and inclinations. G_RVS is assigned, not derived: the distance is
  free, and the recovery depends on the magnitude only through the S/N.
- **Weights.** Each cell receives the weight N_catalogue / N_simulated, with N_catalogue the
  number of DR3 candidates of the wide light-curve classes in the cell (A 2d, by G_RVS,
  and by temperature class from `teff_gspphot`; one further aggregate query gives the
  three-way table). The simulated period distribution of each cell is compared with that of
  the wide classes and raked to it in bins of 0.2 dex where the two differ by more than
  their sampling errors. DR4 publishes the class per source, so the weights can be taken
  again from the release.
- **Use.** Recovery as a function of magnitude and class is read from the cells without
  weights. Statements about the catalogue (the expected number of orbits, the distribution
  of mass errors of the DR4 sample) use the weights, with bootstrap intervals.

### 3.2 Controls and paired perturbations

| run | systems | content | purpose |
|---|---|---|---|
| design | 7800 | section 3.1 | recovery and errors against every parameter |
| null | 900 | single stars at the same magnitudes and classes, analysed with two templates | rate of false double-lined detections; calibration of the detection threshold |
| third light | 1200 | 20 systems per cell of the design sample, simulated again with the tertiary of B 4 where one is drawn | cost of an unmodelled third star |
| activity | 1200 | the same systems with emission filling the Ca II cores of the components cooler than 6500 K that rotate fast enough to be saturated (B 6) | cost of chromospheric emission that no template has |
| instrument | 1200 | the same systems with a background that varies between transits, a residual scale and a second-order response per transit, analysed with `scale="free"` | cost of the delivery |
| recorded | 33 | the systems of the D66 notebook, same seeds and grids | ties the new driver to recorded numbers |
| pilot | 300 | a subset of the design sample, run first | timing, memory, and the definitions of section 7, which are fixed after it |

Each perturbation is paired: the same systems, epochs and noise seeds as in the design
sample, so that a difference is that of the perturbation and not of the draw.

### 3.3 What is declared to the analysis

Every system is measured under three declarations of the templates, on the same epochs.

| declaration | templates | light fractions | represents |
|---|---|---|---|
| `injected` | the injected labels and rotation | held at the injected values | the estimator and the noise alone |
| `classified` | labels with the errors of a classification, drawn once per system: the larger of 150 K and 3 percent in temperature, 0.2 dex in log g, 0.15 dex in [M/H], 20 percent in v sin i | measured | an analysis that has classified the two stars |
| `catalogue` | one template for both stars: the temperature of the combined colour on a 500 K lattice, log g 4.5, solar metallicity, synchronous rotation for the main-sequence radius at that temperature | measured | a first pass with catalogue information only |

The first two follow the D66 notebook, whose classification error was 150 K at every
temperature. The period and the epoch of primary eclipse are declared exactly in all three:
the photometric ephemeris is far more precise than any quantity measured here. Mowlavi et
al. find 85 to 93 percent of the DR3 periods compatible with the literature, factors of two
included, and the report states that a wrong catalogue period is outside the experiment.

## 4. The population model

The existing `draw_population(kind="eclipsing")` draws the inclination inside the eclipsing
range but does not weight a system by its eclipse probability, uses a mean dwarf sequence
with no evolution and no scatter, keeps both stars inside one library box (which makes half
of the pairs twins), requires a mass ratio above 0.4, and has no eclipse depth, no third
light and no eclipse in the spectra. It stays as it is, because recorded runs depend on it.
A new `draw_eclipsing_population` implements the following. The section of B that supports
each row is named.

| ingredient | prescription | source |
|---|---|---|
| primary mass | initial mass function, drawn within the stratum's temperature class | as now |
| period and mass ratio | companion frequency per decade of period and the broken power law in mass ratio with the twin excess, mass ratios from 0.1 so that single-lined systems are in the denominator | Moe & Di Stefano (2017), as now. Checked afterwards against the detached double-lined compilation: 30 percent above 0.95 and 49 percent above 0.9 (B 3.4) |
| radii and temperatures | both stars from one age, uniform up to the end of the primary's subgiant phase, and radius, temperature and luminosity from a table of tracks (decision 2) | B 3.5: every published model uses evolutionary tracks |
| detached | each radius below 0.75 of its Roche lobe (Eggleton 1983) at periastron | B 3.6 |
| eclipse | orientation isotropic and the system kept if it eclipses, which weights it by its eclipse probability. Kept only if the primary eclipse is 0.04 mag deep in G, and with the probability that three of 93 photometric transits fall in an eclipse | B 2.2 and 2.3 (the onset of the DR3 depths; the rule of Mowlavi et al.; the DR3 transit count scaled to DR4) |
| selection by brightness | a system is kept in proportion to the volume its combined RVS light reaches, relative to the brightest of its stratum | as now, in the RVS band |
| eccentricity | circular below P_env = 2.5, 2.0 and 1.6 d for primaries cooler than 7000 K, of 7000 to 10,000 K and hotter. Above, a share log(P / P_env) / [2 log(P_circ / P_env)] is eccentric, with P_circ from 6.65 to 2.46 d by temperature and the distribution and ceiling of Moe & Di Stefano | B 3.3 (the periods of IJspeert et al. 2024; the form of the share is B's construction through them) |
| rotation | aligned. Pseudo-synchronous (Hut 1981) below 10 d for 90 percent of the stars cooler than 6250 K and for hotter stars with R / a above 0.1; half of the systems from 10 to 30 d; field rotation otherwise | B 3.1 and 3.2 |
| flux ratio | ratio of the library continua in the RVS band times the square of the radius ratio; a star outside every box takes the nearest node and is flagged | as now |
| third light | a tertiary with probability 0.30 + 0.80 exp(-P / 7 d), at most 0.96, of mass uniform between 0.1 and 1.2 of the primary's, counted when unresolved; only in the third-light run | B 4 (Tokovinin et al. 2006) |
| sky and cadence | ecliptic latitude from the catalogue's distribution at G_RVS below 12 (A 2g); transit count and times from `rvs_transit_count` and `rvs_transit_times` over the DR4 span of 2005 d | A, present code |
| metallicity, systemic velocity | [M/H] normal about -0.1 with a dispersion of 0.2 dex; systemic velocity normal with 30 km/s | present code, [M/H] changed from uniform |

**Validation of the population.** Before any spectrum is simulated, the weighted population
is compared with the catalogue's wide classes in G_RVS, temperature class, period and
primary eclipse depth (A 2), and with the targets of B 10: the period quantiles of detached
systems, the shares of subgiant primaries, the eccentric share against period, the sum of
the fractional radii, and the share of mass ratios above 0.95 among the systems recovered
as double-lined. The comparison is the first figure of the report.

## 5. The observation model

| element | treatment | status |
|---|---|---|
| spectra | BOSZ 2024 at R = 20,000, one box per temperature range (cool, FGK, hot), each star rendered from the box that contains it | FGK cached; hot registered and not downloaded; cool to be registered (decision 1) |
| rotation | Gray kernel with the linear limb-darkening coefficient of the far red, 0.55 at 4000 K falling to 0.29 at 10,000 K | present code takes one coefficient; values from B 5.4 |
| velocities | Keplerian, barycentric | present code |
| eclipses | at each transit the visible fraction of each disc from the geometry (two spheres, linear limb darkening, integrated in rings); the light fractions and the S/N of the transit follow from it | new; `simulate_rvs_dataset` already takes per-epoch light fractions and S/N |
| line distortion in eclipse | not modelled in the baseline. A transit in eclipse is flagged from the ephemeris and reported separately, as observers leave such epochs out (B 5.2) | extension (section 13) |
| instrument | S/N per detector pixel from G_RVS, photon noise on the detector grid, one resolving power per transit from the in-flight values, linear interpolation onto the DR4 grid with the noise correlation recorded | present code; the interpolation is that of the DR4 draft (A 1) |
| zero point | one velocity offset per transit, common to both stars, normal with 0.17 km/s | new, one line; an upper bound from the bright end of Gaia's own scatter (A 3) |
| what the analysis declares | the delivered line-spread width (11.67 km/s), the lag-one correlation (0.27), the nominal resolving power | as in D66 |
| background | the mission median of 4.7 electrons per pixel in the baseline; a factor drawn per transit in the instrument run, up to the 20 to 30 electrons per pixel of the worst spin phases (Cropper et al. 2018) | present code takes it through `RVSConstants` |
| normalisation | exact in the baseline; a residual scale and a second-order response per transit in the instrument run. DPAC fits a second-degree polynomial per CCD spectrum at G_RVS below 12 and a constant when fainter (A 1) | `simulate_dataset` has both; a pass-through is needed |

## 6. The analysis

1. **Templates** on one grid of three pixels per line-spread sigma (3.89 km/s), rendered
   from a library resampled once per process.
2. **Correlation.** `todcor` over 450 km/s on either side, with the settings of D66. Under
   the declarations that measure the light fractions, they are measured on the epochs out
   of eclipse and held; epochs in eclipse are measured with free amplitudes.
3. **Order of the components.** For an eclipsing binary the ephemeris fixes which star
   recedes after the primary eclipse, so the pair of velocities of an epoch is assigned by
   the sign of the velocity curve wherever the stars are resolved. This replaces the search
   over assignments that `assign_components` makes for a binary with no ephemeris.
4. **Orbit.** Two fits for every system: `assign_components` at the known period (what the
   package does today), and a fit with the period and the epoch of conjunction held, linear
   in the systemic velocity and the two semi-amplitudes for a circular orbit. The second is
   new (section 10). An orbit is taken as circular where the injected eccentricity is zero,
   which the light curve shows.
5. **Detection of the secondary.** A statistic summed over the epochs along the fitted
   orbit, with its threshold set on the null run at one false detection in a hundred. Two
   candidates (the summed `delta_chi2` of the second template at the orbit's velocities, and
   the significance of K2) are compared on the pilot and one is kept. DPAC's own conditions
   (section 2.1) are drawn in the figures for reference.
6. **Predicted precision.** For every epoch and component, the photon-limited error of an
   isolated line system from the derivative of the template and the noise of the epoch. It
   costs one sum per template and gives the axis on which systems of different temperature,
   rotation, flux ratio and magnitude are compared.

## 7. Quantities reported

Fixed after the pilot and before the full run, and recorded in the run's manifest.

**Per epoch and component.** Error against the injected velocity; quoted error; pull;
whether the epoch is measured, usable, flagged, in eclipse; whether it is wrong (either
velocity more than five quoted errors and 3 km/s from the injected value, the definition of
D66); whether it is right with the two velocities interchanged. Summaries per bin: median
absolute error, robust scatter, median signed error, robust width of the pulls, fractions
beyond three and five quoted errors, and the ratio of the scatter to the predicted
precision.

**Per system.** Number of usable epochs; whether the secondary is detected; errors and
pulls of K1, K2, the mass ratio, the systemic velocity and the two values of M sin^3 i;
whether both semi-amplitudes are within 5 and within 10 percent, and within the 0.45, 1.3
and 4.5 percent that masses to 1, 3 and 10 percent require; error of the measured flux
ratio; the reason where nothing was fitted.

**Per bin.** Fractions with Wilson intervals. Every fraction is given against G_RVS and
temperature class first, then against flux ratio, v sin i, period, largest line separation
in units of the line width, and number of transits.

**For the catalogue.** Weighted counts of systems with a double-lined orbit and with masses
to 3 and to 10 percent, cumulative in G_RVS, with bootstrap intervals; and the fitted
recovery function. The 483 double-lined orbits that DR3 has among its candidates brighter
than G = 11 are the comparison.

## 8. Computation

- **Fixed array lengths.** One model grid, one template grid, epochs in blocks. The
  simulation, the templates and `todcor` are in this state since section 10.1 and compile
  nothing after the first ten systems. The orbit fit still compiles two to four programs
  per system, one set per number of usable epochs, and a process grows by 8 MB per system
  over its first 80 (C7). Its results are pinned bit for bit by a regression test, and
  padding its epochs changes the order of its sums, so the experiment uses the linear fit
  of section 6 for circular orbits and replaces its workers at intervals.
- **Workers.** A process pool with the spawn context, one thread per process, chunks of 50
  systems, each worker replaced after a fixed number of chunks. A chunk writes one `.npz`
  when it is complete, so a run resumes. No chunk is started while less than 3 GB of memory
  is free, because the machine is shared.
- **Seeds.** One seed per system from `SeedSequence(run seed).spawn`, by index, so that the
  tables do not depend on the number of workers or the chunking.
- **Tables.** `systems` (one row per system and declaration) and `epochs` (one row per
  epoch and declaration), float32, written by a `collect` step with a manifest: commit,
  package versions, machine, configuration, seeds, definitions, checksums.
- **Wall-clock time.** With the stage costs of section 2.4, three declarations cost 3.4 s
  per system on one process. The 12,300 simulated systems of section 3.2 are 12 CPU-hours,
  under two hours on eight workers. The pilot is about three minutes.

## 9. The report

**Where.** A page of the documentation site, `docs/reports/gaia-rvs-eclipsing-binaries.md`,
under a new navigation entry, with its figures and binned tables in a directory beside it.
`docs/benchmarks.md` receives a short section with the key numbers and a link. The page and
every number in it are written by `scripts/rvs_eb_report.py` from the collected tables, so
the text cannot differ from the tables, and a test checks that it does not.

**Figures.** Static, built with matplotlib from one style module. Each is rendered for the
light and for the dark theme of the site and shown with the theme. The method is that of
the palette the repository already uses: the form is chosen from what the reader has to do,
magnitude is one hue from light to dark, a signed quantity is a diverging pair about a grey
midpoint, the temperature classes take fixed hues in a fixed order in every figure, no panel
has two different vertical scales, a panel that would need more than three hues among
unordered marks is split into small multiples, and every figure has its table. The
categorical hues are checked with the validator for both themes before they are used.

| | figure | form | what it answers |
|---|---|---|---|
| 1 | The sample against the DR3 catalogue | histograms in small multiples, simulation over catalogue in grey | is the population that of the catalogue |
| 2 | Epoch spectra | grid of spectra, temperature class by G_RVS 8, 10 and 12, with the two components | what the data are |
| 3 | Recovery of both semi-amplitudes | heat map, G_RVS by temperature class, contours at 50 and 90 percent | the headline |
| 4 | Epoch error against G_RVS | lines with 16 to 84 percent bands, one panel per class, primary and secondary, predicted precision and Gaia's own scatter in grey | the error of a transit |
| 5 | Calibration of the quoted errors | pull distribution against a unit normal; robust width against S/N | whether the errors can be used |
| 6 | Fate of an epoch | stacked bars per magnitude: right, right when interchanged, flagged, wrong, not measured | where epochs are lost |
| 7 | Line separation | wrong fraction and error against separation in line widths, by S/N | when two sets of lines are resolved |
| 8 | The secondary | heat map of detection, flux ratio by G_RVS, with the 50 percent contour per class and DPAC's ratio of 0.2 | the faintest companion measured |
| 9 | Rotation | scatter over predicted precision against v sin i, per class | what synchronous rotation costs |
| 10 | Orbits and masses | cumulative distributions of the errors of K and of the masses, by magnitude | what the orbits are good for |
| 11 | The ephemeris | paired points per magnitude: period known against ephemeris known | what the eclipses add |
| 12 | The templates | paired points per class for the three declarations | what template knowledge costs |
| 13 | Effects outside the baseline | signed bars with intervals for third light, activity, instrument, epochs in eclipse | the error budget |
| 14 | The recovery function | binned recovery against the predicted S/N of the secondary, with the fit | a rule that can be applied to a catalogue |
| 15 | The DR4 catalogue | cumulative counts against limiting G_RVS: candidates, double-lined orbits, masses to 10 and to 3 percent, and the DR3 orbits | the yield |
| 16 | Failures | six phased velocity curves with the injected curves | what a failure looks like |

**Kept in the repository.** The page, the figures (PNG at twice the display size, about
4 MB in all), the binned tables behind every figure (CSV), the manifest, and the
population file. The row-level tables (about 4 x 10^5 epochs by three declarations) are
reproducible from the manifest and are attached to a release, not committed; the repository
is 8 MB today.

## 10. Changes to the code

### 10.1 Made on 2026-10-07 (not committed)

The simulation path compiled array programs per system and kept them (C2). Four causes were
removed, with the delivered epochs, the injected velocities, the templates and the
resampled libraries of the comparison unchanged bit for bit.

| module | change |
|---|---|
| `albireo.kepler` | the Newton step of `solve_kepler` is a function of the module with the eccentricity and the mean anomaly in the loop state. As a closure it had the loop compiled on every call outside `jit` |
| `albireo.library` | `resampled_to` is a sparse matrix product in SciPy and no longer compiles a program per call. For a library with read-only arrays the result and its interpolator are kept, up to 96 MB per library. `load_library` returns read-only arrays |
| `albireo.gaia`, `albireo.benchmark` | `rvs_model_grid(length_multiple=)`; `simulate_system` rounds the length of its grid up to a multiple of 512 where the library extends that far |
| `albireo.grids`, `albireo.simulate`, `albireo.todcor` | the per-epoch velocities and pixel shifts of the simulation, and the barycentric velocities of a dataset in `todcor`, are evaluated in blocks of 32 epochs |
| `albireo.operators`, `albireo.todcor` | the rotation and Gaussian kernels of the NumPy callers are no longer converted to JAX and back |

### 10.2 To be made

| module | change | kind |
|---|---|---|
| `albireo.library` | registry entry for a cool BOSZ box in the RVS band (3200 to 4000 K, log g 4.0 to 5.5) | small |
| `albireo.population` | `draw_eclipsing_population`; Roche radius; eclipse geometry (visible fractions, depths, durations); the evolutionary table and its reader; pseudo-synchronous rotation; third light; design strata and weights | the main addition |
| `albireo.gaia` | pass-through of the response and scale of `simulate_dataset`; a background factor and a zero-point offset per transit | small |
| `albireo.rvorbit` | a fit with the period and the conjunction held (linear for a circular orbit), with the assignment by the sign of the curve | new public function |
| `albireo.benchmark` or a new experimental module | the driver: configuration, the per-system chain, the chunked runner, `collect`, the metrics | new |
| `scripts/` | `rvs_eb_run.py` (draw, pilot, run, collect) and `rvs_eb_report.py` | new |
| `tests/` | geometry against closed forms; determinism across worker counts; the linear fit against `fit_rv_orbit`; a 12-system end-to-end run; the report's numbers against its tables | new |
| `docs/`, `internal/design.md`, `CHANGELOG.md` | API pages, the report, the benchmarks section, decision record D69 | |

`_DR4_SPAN_DAYS` stays 1998 d until the fourth benchmark run, as decided on 2026-10-06. The
experiment passes the span of 2005 d explicitly.

## 11. Acceptance criteria

1. Met on 2026-10-07: the epochs of the corrected simulation equal those of the code
   before it for the same seeds on 24 systems, and the templates equal those of
   `Template.from_library`.
2. The resident memory of a worker is flat to within 1 MB per system after its first 100
   systems, or the workers are replaced often enough to stay below 2 GB. Met for the
   simulation, the templates and `todcor`; open for the orbit fit.
3. The tables of a 200-system run are identical with one and with eight workers.
4. The 33 recorded systems, simulated on their recorded grids, give the counts of the D66
   notebook under `injected` (93.4 percent of epochs usable as measured; both
   semi-amplitudes within 5 percent for 21 of 22 systems above 100 km/s and 9 of 11 below).
5. For single stars the scatter of the velocity is within 10 percent of the predicted
   precision at S/N above 10, and the robust width of the pulls is between 0.9 and 1.2. The
   scatter against magnitude, temperature and rotation is compared with Gaia's own (A 3):
   a comparison that is reported, not a criterion.
6. The weighted population reproduces the catalogue's counts in G_RVS and temperature class
   by construction, and the targets of B 10 to within their stated ranges or with the
   departure reported.
7. Every number on the report page is read from the tables by the builder.

## 12. Order of work

| stage | content | ends when | effort |
|---|---|---|---|
| 0 | decisions of section 13; downloads | libraries built and checked | an hour, mostly transfer |
| 1 | population model and its validation figure | criterion 6 | one session |
| 2 | eclipses, the held-ephemeris fit, the driver with replaced workers, tests | criteria 2 to 4 | one to two sessions |
| 3 | pilot of 300 systems; definitions fixed; an independent review of the metric code | criterion 5; manifest written | half a session |
| 4 | the runs of section 3.2 | tables collected | about two hours of machine time |
| 5 | report, documentation, records | criterion 7; `mkdocs build --strict` passes | one session |

Stages 1 to 3 need no download for the F, G and K classes and can start at once.

## 13. Decisions requested

1. **Temperature range, which is a question of downloads from the BOSZ archive at STScI.**
   (a) Recommended: 3200 to 10,000 K. The hot box (registered, 532 MB, measured) and a cool
   box for the secondaries (140 nodes, about 190 MB, to be confirmed against the archive
   listing). The built caches are 4 MB each and the downloaded files can then be deleted.
   (b) Extension to 12,000 K, late B: 331 MB more. The archive's files for the hot boxes
   may predate the recomputation of the hydrogen lines, and everything above 8000 K is a
   Paschen-line velocity in LTE; the report would state both. (c) F, G and K only, with no
   download, which covers about three fifths of the bright catalogue.
2. **Evolution of the stars.** (a) Recommended: a table reduced from MIST tracks at solar
   metallicity (one download of the order of 100 MB, to be confirmed; a reduced table of
   about 100 kB kept in the package with its citation). (b) No download: radii drawn above
   a lower envelope, log R = -0.05 + 0.80 log M, by 0.06, 0.13 and 0.27 dex at the 16th,
   50th and 84th percentiles for 1 to 2.5 solar masses (measured by B on the compilation of
   Eker et al. 2018); the two stars of a pair then have no common age, and no published
   model has done it this way.
3. **Size.** Recommended: the runs of section 3.2 (12,300 simulated systems). A tenth of it
   answers the first-order questions in ten minutes and cannot support the maps against a
   third variable.
4. **Where the report is kept.** Recommended: section 9, with the row-level tables on a
   release.
5. **Where the driver is kept.** Recommended: an experimental module of the package with
   two thin scripts, as `albireo.benchmark` is, because worker processes import it.
6. **Extensions, none of which the baseline needs.** (a) A search over (K1, K2, systemic
   velocity) on all epochs at once, along the ephemeris. It is not an epoch velocity, and it
   is the only route below G_RVS of about 12.5, where single transits fail (C5, C6); the
   catalogue has 2.2 times more candidates per magnitude there. (b) The distortion of the
   lines of the eclipsed star, 0.13 to 0.44 of v sin i in the centroid (B 5.1). (c) A fourth
   declaration in which the templates are chosen by the correlation itself over a lattice
   of the library. (d) An interactive companion to the report.

## 14. What the experiment cannot show

- The data and the templates come from one library. The `classified` and `catalogue`
  declarations and the activity run introduce mismatch, but not the difference between
  BOSZ and a star.
- The line-spread function is Gaussian here and is not in the instrument, and its departure
  has not been published.
- The DR4 grid, the normalisation codes and the selection flags are from the draft data
  model and are to be checked against the release.
- The stars are spheres on Keplerian orbits. Ellipsoidal distortion, reflection and spots
  are absent. For detached pairs of 3 to 10 d the departure from a Keplerian curve averages
  0.17 km/s and reaches 3.8 km/s (Sybilski et al. 2013), and spots add 0.4 to 0.7 km/s at
  30 to 60 km/s of rotation (B 6).
- The catalogue weights are those of DR3 candidates of the wide classes, of which the
  detached systems are a part, and DR3 lists about two thirds of the detached systems that
  a ground survey has at these magnitudes (B 2.3).

## 15. Execution record (2026-10-07)

The user approved every recommendation of section 13 on 2026-10-07, and the plan was
executed the same day. The results are in `docs/reports/gaia-rvs-eclipsing-binaries.md`,
which `scripts/rvs_eb_report.py` writes from the runs of `scripts/rvs_eb_run.py`. The runs
are kept under `runs/` beside this page and are not in the repository. Decision record: D69.

**What was built.** `albireo.eclipsing` (section 4), `albireo.survey` (sections 5 to 8),
`rvorbit.fit_rv_ephemeris` and `assign_by_ephemeris` (section 6), the registry entry
`bosz2024-cool-rvs`, the packaged table of MIST tracks, the two scripts, and the report with
sixteen figures in two renderings.

**Runs.** Design 7,800 systems (109 min on 8 processes), null 900 (17 min), third light,
emission and instrument 1200 each (18, 18 and 21 min), recorded 33. The stages of one system
sum to 6.4 s under three declarations with the eight processes running, of which 5.7 s are
the correlations, and to 4.8 s in one process alone. Section 8 estimated 3.4 s.

**Acceptance criteria (section 11).**

1. Met before the plan was approved.
2. Met: a worker holds 1.0 GB and does not grow, after the fit with a free conjunction was
   evaluated on epochs padded to whole blocks (`rvorbit._epoch_blocks`). Workers are still
   replaced after four chunks.
3. Met on 96 systems: identical tables with one process and with six.
4. Met exactly: 93.4 percent usable epochs, 21 of 22 and 9 of 11.
5. Met for the pulls and not for the scatter. On the primaries of the null run, measured
   with their own template, the pulls have a robust width of 0.96 (criterion: 0.9 to 1.2).
   The scatter is 0.87 times the predicted precision at S/N above 10 (criterion: within 10
   percent), on the side of smaller errors: the prediction takes the noise of a transit as
   uniform at its median, and the quoted errors, which do not, are calibrated. The
   comparison with Gaia's own scatter was to be reported and not a criterion. It shows a
   difference that the page states as its first limit: the scatter that the errors of DR3
   imply for one transit of a slowly rotating G dwarf is 1.2, 1.5 and 1.6 times the
   simulated one at G_RVS = 8, 10 and 12, the equivalent of 0.5, 0.7 and 0.6 mag.
6. Met for the magnitudes and classes by construction. Departures from the targets of B 10
   are reported on the page: the model has a longer tail of periods than the DR3 catalogue
   and is more detached than the ASAS-SN class.
7. Met: `tests/test_survey_report.py` renders the page again from `numbers.json` and
   compares, and checks the numbers of two tables.

**Departures from the plan, each made on a measurement.**

- *Twin excess (section 4).* The published excess of Moe & Di Stefano gave 53 percent of the
  sample above q = 0.95, and 67 percent of its systems with a flux ratio above 0.1, where
  the compilation of Eker et al. has 30 percent and the DR3 detached classes at most 28
  percent of equal eclipse depths. The draw takes 0.3 of the published excess, rising
  linearly from q = 0.85, and then has 31, 50 and 86 percent above 0.95, 0.9 and 0.7 (30, 49
  and 83 in the compilation).
- *Eccentricity.* The construction through the envelope and circularisation periods, with
  the eccentricity distribution of Moe & Di Stefano, gave 3 and 7 percent of orbits above e
  = 0.1 at 3.2 to 5.6 d and 5.6 to 10 d, where target 5 of B 10 has 10 and 17. The share
  above 0.1 is now taken from the eclipsing samples directly, by period and temperature,
  with eccentricities uniform up to the envelope of Mazeh (2008).
- *Transit count.* Drawn from the `rv_nb_transits` of the catalogue's own candidates (10, 20
  and 33 at the 10th, 50th and 90th percentiles in DR3), scaled to DR4, and not from the
  all-sky distribution at the system's ecliptic latitude.
- *Evolution.* Primaries from the zero-age main sequence to log g = 3.5, the floor of the
  hot box, and not 3.0. The cool box covers 3200 to 4000 K.
- *Weights.* The weights by cell are the baseline. The weights raked to the catalogue's
  periods are bounded to a factor of four and to the period bins that the model populates
  with 20 systems or more, and serve as a sensitivity: 9 percent of the candidates lie in
  other bins, mostly at periods below those of any detached pair of their class.
- *Transits in eclipse* are measured with free amplitudes under every declaration, the
  `injected` one included.
- *Detection of the secondary.* The statistic is the semi-amplitude of the secondary over
  its error from the fit at the held ephemeris, with its threshold from the null run. No
  threshold per epoch was added. An undetected secondary still returns a velocity where the
  light fractions are measured, so the page conditions on the detection of the system.
- *Single-lined systems.* A measurement with the first template alone was added to the
  chain, since a correlation with two templates flags the epochs whose second velocity is
  not determined and leaves few for the primary of a pair with a faint companion.
- *Paired runs.* The baseline of the three perturbation runs is the design run itself, which
  has the same systems and seeds.
- *Figures.* No set of four hues passes the palette validator on both page surfaces, so the
  four classes are small multiples and colour separates at most three series.
- *Defects found on the way.* `todcor` raised on two identical templates fitted with free
  amplitudes (the `catalogue` declaration in eclipse); it now flags the epoch. Two reviews
  by Opus agents that were given the code and not the design found four defects and three
  minor items in the chain and sixteen issues in the report's statistics. The largest was in
  `assign_by_ephemeris`: with the eccentricity fitted it ordered the pairs of alike stars by
  the sign of a circular curve, which exchanged correctly measured epochs of eccentric
  orbits. The design run had completed 42 of its 156 chunks; every run was stopped and
  repeated with the corrected code. The raked weights gave weight to cells that the
  catalogue does not list, and were recomputed. In the report, shares that pool cells are
  weighted to the catalogue, the intervals of the yield resample within cells, and the
  calibration of the errors is tested against the velocity put into the spectrum.

**What the runs did not confirm.** A prototype on the pilot suggested that the fit at the
held ephemeris recovers more faint systems than the fit with the period alone. In the design
run the two shares agree within 2 percentage points in every half magnitude, once the
semi-amplitudes of the fit with the period alone are compared in the order that fits. The
ephemeris names the two stars and needs no starting value. It does not add recovered orbits
at a known period.

**Not done.** The row-level tables are not published: a release is an outward-facing action
and waits for the user. The extensions of item 6 of section 13 were not recommendations and
were not built. Section 14 stands.

## 16. References

Bibcodes for the sources quoted through A and B are in those notes.

Cropper, M., Katz, D., Sartoretti, P., et al. 2018, A&A, 616, A5 (2018A&A...616A...5C)
Eggleton, P. P. 1983, ApJ, 268, 368 (1983ApJ...268..368E)
Eker, Z., Bakis, V., Bilir, S., et al. 2018, MNRAS, 479, 5491 (2018MNRAS.479.5491E)
Hut, P. 1981, A&A, 99, 126 (1981A&A....99..126H)
IJspeert, L. W., et al. 2024, A&A, 691, A242 (2024A&A...691A.242I)
Katz, D., Sartoretti, P., Guerrier, A., et al. 2023, A&A, 674, A5 (2023A&A...674A...5K)
Lurie, J. C., et al. 2017, AJ, 154, 250 (2017AJ....154..250L)
Meszaros, Sz., Bohlin, R., Allende Prieto, C., et al. 2024, A&A, 688, A197 (2024A&A...688A.197M)
Moe, M. & Di Stefano, R. 2017, ApJS, 230, 15 (2017ApJS..230...15M)
Mowlavi, N., Holl, B., Lecoeur-Taibi, I., et al. 2023, A&A, 674, A16 (2023A&A...674A..16M)
Rowan, D. M., Jayasinghe, T., Stanek, K. Z., et al. 2022, MNRAS, 517, 2190 (2022MNRAS.517.2190R)
Sybilski, P., et al. 2013, MNRAS, 431, 2024 (2013MNRAS.431.2024S)
Tokovinin, A., Thomas, S., Sterzik, M. & Udry, S. 2006, A&A, 450, 681 (2006A&A...450..681T)
Zerjal, M., Zwitter, T., Matijevic, G., et al. 2013, ApJ, 776, 127 (2013ApJ...776..127Z)
Zucker, S. 2003, MNRAS, 342, 1291 (2003MNRAS.342.1291Z)
Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806 (1994ApJ...420..806Z)
