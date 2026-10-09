# Gaia DR4 eclipsing binaries in the RVS: epoch velocities by correlation

Gaia DR4 publishes one RVS spectrum per field-of-view transit. This page reports a
simulation of 7,800 detached eclipsing binaries as that product will show them, in
which both velocities are measured at every transit by correlation against library
templates ([TODCOR](../tutorials/todcor.md)) and an orbit is fitted at the photometric
ephemeris. Nothing is disentangled. The page, its figures and every number on it are
written by `scripts/rvs_eb_report.py` from the tables of the runs (albireo 77a3a32 with uncommitted changes,
2026-10-08 UTC).

## Summary

Five findings on the extraction of radial velocities from RVS epoch spectra. Each is
developed in a section below.

1. **Precision of a single transit.** With the injected templates the scatter of a
   transit's velocity about the velocity in its spectrum is 0.93
   times the photon-limited prediction, and the quoted errors are calibrated: the robust
   width of the pulls is 0.98 for the primaries and 1.01 for the
   secondaries. With the zero point of 0.17 km/s that every transit carries, the scatter of
   a G-type primary about its orbit is 0.69, 1.75 and 8.27
   km/s in the half magnitudes that begin at G_RVS = 8, 10 and 12.
2. **Cost of imperfect templates.** At the bright end the result is limited by the
   templates. Brighter than G_RVS = 9, both semi-amplitudes are within 10 percent of the
   injected ones for 85 percent of the systems with the injected
   templates, 61 percent with templates from a classification and
   44 percent with one template from the colour of the pair. A-type
   primaries lose the most: among the pairs with a flux ratio above 0.2 and G_RVS below 10
   the three declarations recover 95,
   69 and 46 percent of them.
   The quoted errors do not contain the mismatch. For the recovered systems the pulls of
   the two semi-amplitudes have robust widths of 1.62 and
   2.40 with classified templates.
3. **Faint limit and the scale of the noise.** With classified templates, among the
   systems whose secondary has more than a fifth of the primary's flux, both
   semi-amplitudes are within 10 percent for 80 percent brighter
   than G_RVS = 9, 81 percent at 10 to 11,
   70 percent at 11 to 12 and 25 percent
   at 12 to 13. Half of them are recovered at G_RVS = 12.0. Over all
   systems in the catalogue's
   mixture of classes, which include every mass ratio from 0.1, the shares are
   61, 53, 43 and
   15 percent. A simulated transit is at the photon limit of the
   S/N that DPAC expects for it. The scatter of one transit that the errors of Gaia DR3
   imply for single dwarfs is 1.2, 1.5 and
   1.6 times that of the simulated single stars at G_RVS = 8, 10 and 12,
   the equivalent of 0.5, 0.7 and 0.6 mag.
   If the transits of DR4 are as DR3 measured them, a recovery quoted here at a magnitude
   holds that much brighter ([Limits](#limits)).
4. **Blended lines and faint companions.** For the pairs with a flux ratio above 0.2, at
   S/N above 30 with classified templates, 100 percent of the transits whose
   lines are separated by more than 1.5 line widths are usable, and 1.7 percent
   of those have a velocity more than five quoted errors and 3 km/s from the injected one.
   Below 0.6 widths the numbers are 90 and 34.9 percent. Where the
   secondary has less than a tenth of the primary's flux, the primary's semi-amplitude
   from its template alone is within 10 percent for 97 percent of the
   systems brighter than G_RVS = 9, 94 percent at 11 to 12 and
   79 percent at 12 to 13.
5. **Third light and chromospheric emission.** In the model of the population
   49 percent of the systems have an unresolved third star. Left out of
   the analysis, it lowers the share recovered among them from 51 to
   33 percent, and to 12 percent where it gives more
   than a tenth of the light. Its lines stand at the systemic velocity and draw the
   semi-amplitudes low: in that case the median errors of the two are -6.7
   and -29.2 percent among the pairs with a flux ratio above 0.2 and G_RVS
   below 11. Emission in the Ca II cores at the strength of saturated activity, in the
   44 percent of systems with a star that qualifies, lowers the share from
   45 to 28 percent, with median errors of +2.1
   and +3.6 percent. A background that varies between transits changes the share
   by +2.0 points. These are shares of a sample with equal numbers at
   every magnitude and class.

Weighted to the 36,131 Gaia DR3 candidates of the detached light-curve
classes with G_RVS between 6 and 13.5, the baseline gives 4,789 double-lined orbits
brighter than G_RVS = 12 and 7,123 brighter than 13.5, with both masses to 10
percent for 4,880 systems and to 3 percent for 1,552. The baseline has
no third light, no emission and photon-limited transits. One system takes
6.4 s for three template declarations with 8 processes
running: 0.2 s for the simulation, 5.7 s for the correlations and
0.6 s for the orbit fits. The 7,800 systems, 313,125 transits,
took 109 minutes on a desktop.

## Question and scope

For the detached eclipsing binaries among the sources of the RVS, which per-transit
velocities of both components can be measured by correlation alone, with what errors, and
which orbits and masses follow when the eclipse ephemeris is taken from the photometry?
The answer is given against G_RVS, which sets the S/N of a transit, and against the
spectral types, the flux ratio, the rotation and the separation of the lines.

The simulation covers detached pairs of main-sequence and subgiant stars with primaries of
4000 to 10,000 K, G_RVS from 6.0 to 13.5, and the DR4 epoch product: 961 samples from 846
to 870 nm at a resolving power near 11,500. Contact and semi-detached systems, giants and
stars hotter than 10,000 K are outside it. So is the period search: an eclipsing binary
has a photometric period.

## The simulated sample

The sample is stratified. A sample that followed the catalogue's magnitudes would have
more than half of its systems in the faintest magnitude, where little is recovered, so
every cell of fifteen bins of G_RVS by four classes of the primary's temperature holds the
same number of systems, 130, and the catalogue enters as weights. G_RVS is
assigned and not derived from a distance, since the S/N of a transit depends on nothing
else.

Within a class the systems follow a population model
([`albireo.eclipsing.draw_eclipsing_population`](../api/eclipsing.md)):

| ingredient | prescription |
|---|---|
| primary | initial mass function and a constant star-formation rate over 10 Gyr; kept while its MIST track is in the class, from the zero-age main sequence to log g = 3.5 |
| companion | period and mass ratio of Moe & Di Stefano (2017) from q = 0.1, with their excess of twins reduced to 0.3 of the published fraction and spread from q = 0.85 (see below); the same age as the primary |
| detached | each radius below 0.75 of its Roche lobe at periastron |
| eclipses | isotropic orientations, kept where the system eclipses; the deeper eclipse at least 0.04 mag in G, and three of 93 photometric transits in an eclipse |
| brightness | kept in proportion to the volume that the combined light in the band reaches |
| eccentricity | the share of orbits above e = 0.1 of the eclipsing samples, by period and temperature |
| rotation | aligned; pseudo-synchronous below 10 d for cool stars and for hot stars with R/a above 0.1; that of a field star otherwise |
| spectra | BOSZ 2024 at R = 20,000 in three boxes, 3200 to 10,000 K, each star from the box that contains it |

**The twin excess.** With the published excess of twins, 53 percent of
such a sample has q above 0.95, since a magnitude-limited eclipsing sample favours equal
stars twice over. Two samples of eclipsing binaries disagree with it: 30 percent of the
293 detached double-lined systems of Eker et al. (2018) have q above 0.95, and 14 percent
of the Gaia DR3 candidates of the detached classes with a primary eclipse of 0.2 mag have
the two depths in the same bin of 0.1 mag (28 percent if every single-eclipse model is a
twin at half its period). The sample drawn here has 31, 50 and 86 percent
above 0.95, 0.9 and 0.7 among the systems with a flux ratio above 0.1, against 30, 49 and
83 in the compilation, and 32 percent of equal depths.

**Weights.** Each system carries the number of DR3 candidates of the light-curve classes
that Mowlavi et al. (2023) call wide (2G-A, 2G-D, 2GE-A, 1G) in its cell, divided by the
number of simulated systems there. The classes hold 40,179 candidates between
G_RVS 6.0 and 13.5, of which 10 percent have a temperature
outside the four classes; the temperatures are `teff_gspphot`, and the candidates without
one are spread over the classes of their magnitude in proportion. The weighted sample has
the magnitudes and classes of the catalogue and the periods of the model. A second set of
weights is also raked to the catalogue's periods in the bins that the model populates;
9 percent of the candidates lie in other bins, mostly at
periods below those of any detached pair of their class.

[![Figure 1](gaia-rvs-eclipsing-binaries/figures/fig01-sample-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig01-sample-light.png)
[![Figure 1](gaia-rvs-eclipsing-binaries/figures/fig01-sample-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig01-sample-dark.png)

*Figure 1.* The simulated sample against the Gaia DR3 candidates of the detached light-curve classes. Top: periods by class of the primary, with the two sets of weights. Bottom: the flat design against the catalogue's counts; the depth of the deeper eclipse at G_RVS below 12; the mass ratio; the sum of the fractional radii. Table: [`fig01-sample-periods.csv`](gaia-rvs-eclipsing-binaries/tables/fig01-sample-periods.csv).


The weighted sample has a median period of 2.67 d (1.08 and
8.02 d at the 16th and 84th percentiles), 9 percent of primaries past
the main sequence, 86 percent of orbits with e below 0.05, a median sum of
fractional radii of 0.28, a median depth of the deeper eclipse of
0.21 mag, a median projected rotation of the primary of 44 km/s,
and a median of 39 transits. The model has more systems beyond 6 d than
the DR3 catalogue, whose candidates were found with half as many photometric transits as
DR4 will have, and fewer below 0.6 d, where the catalogue's classes contain pairs closer
to contact than the model allows.

## The simulated observations

Each star is rendered from the library with its rotation and limb darkening. At every
transit the light fractions and the S/N follow from the eclipse geometry of two
limb-darkened spheres, a velocity offset of 0.17 km/s common to both stars is added, and
the transit takes the resolving power of its CCD row. Photon noise is added on the
detector pixels and the spectrum is interpolated onto the DR4 grid, as in the
[Gaia RVS simulator](../tutorials/gaia-rvs.md). A share of 16 percent of the
transits falls in an eclipse. The distortion of the eclipsed star's lines is not modelled,
so those transits are measured but kept out of the orbits, as observers keep them out.

[![Figure 2](gaia-rvs-eclipsing-binaries/figures/fig02-spectra-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig02-spectra-light.png)
[![Figure 2](gaia-rvs-eclipsing-binaries/figures/fig02-spectra-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig02-spectra-dark.png)

*Figure 2.* One transit near quadrature of a typical pair of each class at three magnitudes (grey), with the two components at the resolving power of the instrument, each scaled by its light fraction. Table: [`fig02-spectra.csv`](gaia-rvs-eclipsing-binaries/tables/fig02-spectra.csv).


## The analysis

Every system is measured under three declarations of the templates, on the same epochs.

| declaration | templates | light fractions |
|---|---|---|
| `injected` | the injected labels and rotation | held at the injected values |
| `classified` | labels with the errors of a classification, drawn once per system: the larger of 150 K and 3 percent in temperature, 0.2 dex in log g, 0.15 dex in [M/H], 20 percent in v sin i | measured |
| `catalogue` | one template for both stars: the temperature of the pair's colour on a lattice of 500 K, log g = 4.5, solar metallicity, synchronous rotation | measured |

The period and the time of the primary's eclipse are declared exactly. The chain is
[`albireo.survey.run_system`](../api/survey.md): templates on a grid of three pixels per
line-spread sigma; `todcor` over 450 km/s on either side with the delivered line-spread
width (11.67 km/s) and the lag-one noise correlation of the product declared; measured
light fractions taken on the transits out of eclipse and held; and
[`assign_by_ephemeris`](../api/rvorbit.md), which fits the two semi-amplitudes and the
systemic velocity with the ephemeris held and decides the order of the two velocities of a
transit by it. The eccentricity is fitted where the injected orbit is eccentric.

The definitions were fixed on a pilot of 300 systems.

- A transit is *usable* where it is out of eclipse, both velocities are measured and the
  table does not flag it after the assignment.
- A velocity is *wrong* where it is more than five quoted errors and 3 km/s from the
  injected one.
- A system is *recovered* where both semi-amplitudes are within 10 percent of the injected
  ones. Masses to 10, 3 and 1 percent need them within 4.5, 1.3 and 0.45 percent.
- The secondary is *detected* where its semi-amplitude divided by its error exceeds the
  value that one in a hundred of 900 null systems exceeds. A null system is the
  primary alone, at the system's magnitude, analysed with the two templates of the pair.
  The thresholds are 3.0, 6.4 and
  7.7 for the three declarations.
- An *error* is taken against the velocity of the orbit, and so contains the zero point of
  the transit. Where the estimator itself is tested (the scatter over the predicted
  error, the pulls), it is taken against the velocity that was put into the spectrum.
- A share quoted for a range of magnitudes is *weighted* to the catalogue's numbers in the
  cells it pools. The maps by cell, and the curves against S/N, line separation, rotation
  and predicted S/N, are not weighted.

## Recovery of both semi-amplitudes

[![Figure 3](gaia-rvs-eclipsing-binaries/figures/fig03-recovery-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig03-recovery-light.png)
[![Figure 3](gaia-rvs-eclipsing-binaries/figures/fig03-recovery-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig03-recovery-dark.png)

*Figure 3.* Share of systems with both semi-amplitudes within 10 percent, with classified templates. Cells with fewer than five systems are empty. Table: [`fig03-recovery.csv`](gaia-rvs-eclipsing-binaries/tables/fig03-recovery.csv).


| G_RVS | injected | classified | catalogue | classified, flux ratio above 0.2 | K1 from one template, flux ratio below 0.1 |
|---|---|---|---|---|---|
| 6 to 9 | 85 | 61 | 44 | 80 | 97 |
| 9 to 10 | 80 | 60 | 47 | 81 | 97 |
| 10 to 11 | 67 | 53 | 41 | 81 | 97 |
| 11 to 12 | 52 | 43 | 39 | 70 | 94 |
| 12 to 13 | 22 | 15 | 16 | 25 | 79 |
| 13 to 13.5 | 4 | 3 | 4 | 6 | 57 |

Percent of systems with both semi-amplitudes within 10 percent, the four classes weighted
to the catalogue. The last column is for the systems whose secondary has less than a
tenth of the primary's flux: the share whose primary's semi-amplitude is within 10 percent
when the transits are measured with the primary's classified template alone. Over all
systems that share is 53 percent at G_RVS 6 to 9 and
55 percent at 11 to 12, since one template measures the blend of two
sets of lines.

A stratum contains every mass ratio from 0.1: 43 percent of the weighted
sample has a secondary with less than a fifth of the primary's flux, and 32
percent less than a tenth. The recovery of the whole stratum is therefore below that of
its double-lined systems at the bright end. With the injected templates half of the
systems with a flux ratio above 0.2 are recovered at G_RVS = 12.2, with
classified templates at 12.0 and with one catalogue template at
12.0.

## Velocities of single transits

[![Figure 4](gaia-rvs-eclipsing-binaries/figures/fig04-epoch-error-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig04-epoch-error-light.png)
[![Figure 4](gaia-rvs-eclipsing-binaries/figures/fig04-epoch-error-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig04-epoch-error-dark.png)

*Figure 4.* Robust scatter of the velocity of one transit about its orbit, with the injected templates. It contains the zero point of 0.17 km/s per transit. Bands: 16th to 84th percentiles of the absolute error. The secondary is that of systems with a flux ratio above 0.2. Squares: the scatter of one transit that the errors of Gaia DR3 imply for single dwarfs of 5500 to 6500 K, the same in every panel. Table: [`fig04-epoch-error.csv`](gaia-rvs-eclipsing-binaries/tables/fig04-epoch-error.csv).


With the injected templates the robust scatter of the primary's velocity about the
velocity in its spectrum, over all usable transits with S/N above 10, is
0.93 times the photon-limited prediction for an isolated line system,
and 0.94 times at S/N above 100. Against the orbit, which adds the zero point of
the transit, the two numbers are 0.99 and
1.11. The prediction takes the noise of a transit as uniform at its
median, and the photon noise is lower in the cores of the lines, so a ratio slightly below
one is expected. The figure shows the scatter against the orbit. For G-type
primaries it is 0.69, 1.75 and 8.27 km/s in the half
magnitudes that begin at G_RVS = 8, 10 and 12, and for A-type primaries 1.78,
5.92 and 24.23 km/s. The secondary of a system with a flux ratio above
0.2 has 1.1, 3.2 and 15.4 km/s in the G class. The
squares are the scatter of one transit that the errors of Gaia DR3 imply for single dwarfs
of 5500 to 6500 K: 0.19, 0.36, 1.07 and 5.4 km/s at G_RVS = 6, 8, 10 and 12, the last an
extrapolation from 11.75. The single stars of this simulation are compared with it under
[Checks](#checks-of-the-experiment), and the difference is the first of the
[Limits](#limits).

[![Figure 5](gaia-rvs-eclipsing-binaries/figures/fig05-pulls-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig05-pulls-light.png)
[![Figure 5](gaia-rvs-eclipsing-binaries/figures/fig05-pulls-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig05-pulls-dark.png)

*Figure 5.* Calibration of the quoted errors, against the velocity in the spectrum. Left: pulls with the injected templates; the secondaries are those of systems with a flux ratio above 0.2. Right: robust width of the primaries' pulls against the S/N of the transit. Table: [`fig05-pulls.csv`](gaia-rvs-eclipsing-binaries/tables/fig05-pulls.csv).


The pulls are taken against the velocity in the spectrum, and those of the secondaries
over the systems with a flux ratio above 0.2. With the injected templates they have a
robust width of 0.98 (primaries) and 1.01 (secondaries);
1.3 and 2.5 percent lie beyond three quoted errors and
0.4 and 0.9 percent beyond five. At S/N above 100 the width
of the primaries' pulls is 0.98, and 1.15 against the orbit: the
quoted errors do not contain the zero point of a transit. With classified templates the
widths are 1.23 and 1.46, and with the catalogue template
2.11 and 2.11: a template that does not match moves the
velocity by more than the quoted error where the S/N is high. Among the systems with a
flux ratio above 0.2 the share of usable transits with a wrong velocity is
1.1, 6.5 and 20.1 percent.

[![Figure 6](gaia-rvs-eclipsing-binaries/figures/fig06-fate-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig06-fate-light.png)
[![Figure 6](gaia-rvs-eclipsing-binaries/figures/fig06-fate-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig06-fate-dark.png)

*Figure 6.* Transits out of eclipse by what the table gives for them, in the catalogue's mixture of classes. Table: [`fig06-fate.csv`](gaia-rvs-eclipsing-binaries/tables/fig06-fate.csv).


With classified templates, 74 percent of the transits out of eclipse are
usable with both velocities right in the half magnitude that begins at G_RVS = 8,
72 percent at 10, 62 percent at 12 and 42
percent at 13. The primary's velocity is usable and right at 94,
95, 88 and 58 percent. A transit
that the table flags counts against both.

[![Figure 7](gaia-rvs-eclipsing-binaries/figures/fig07-separation-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig07-separation-light.png)
[![Figure 7](gaia-rvs-eclipsing-binaries/figures/fig07-separation-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig07-separation-dark.png)

*Figure 7.* Transits of systems with a flux ratio above 0.2, with classified templates, against the separation of the lines in units of their width. Table: [`fig07-separation.csv`](gaia-rvs-eclipsing-binaries/tables/fig07-separation.csv).


The lines of the two stars are resolved where their separation exceeds their width, taken
here as the quadrature sum of the line-spread FWHM (27.5 km/s) and 1.6 times the larger
v sin i. At S/N above 30, 100 percent of the transits beyond 1.5 widths are
usable and 1.7 percent of those have a wrong velocity; below 0.6 widths the
numbers are 90 and 34.9 percent.

## Detection of the secondary

[![Figure 8](gaia-rvs-eclipsing-binaries/figures/fig08-secondary-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig08-secondary-light.png)
[![Figure 8](gaia-rvs-eclipsing-binaries/figures/fig08-secondary-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig08-secondary-dark.png)

*Figure 8.* Share of systems whose secondary is detected, with classified templates. Dashed: the flux ratio of 0.2 that the DPAC search for double lines requires. Cells with fewer than five systems are empty. Table: [`fig08-secondary.csv`](gaia-rvs-eclipsing-binaries/tables/fig08-secondary.csv).


With classified templates the secondary is detected in 32 percent of
the weighted sample, of which more than half lies in the faintest magnitude. Of the
systems detected, 62 percent are recovered to 10 percent, and of the
systems recovered, 98 percent are detected. The
DPAC search for double lines requires a flux ratio above 0.2 and reaches G_RVS = 12.

## Rotation

[![Figure 9](gaia-rvs-eclipsing-binaries/figures/fig09-rotation-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig09-rotation-light.png)
[![Figure 9](gaia-rvs-eclipsing-binaries/figures/fig09-rotation-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig09-rotation-dark.png)

*Figure 9.* Scatter of the primary's velocity about the velocity in its spectrum, over the photon-limited prediction, at S/N above 10. Table: [`fig09-rotation.csv`](gaia-rvs-eclipsing-binaries/tables/fig09-rotation.csv).


The figure shows the scatter of the primary's velocity about the velocity in its
spectrum, over its photon-limited prediction, against v sin i at S/N above 10. The
prediction contains the loss of information to rotation, so a ratio of one at every
rotation means that the estimator loses nothing more.

## Orbits and masses

[![Figure 10](gaia-rvs-eclipsing-binaries/figures/fig10-orbits-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig10-orbits-light.png)
[![Figure 10](gaia-rvs-eclipsing-binaries/figures/fig10-orbits-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig10-orbits-dark.png)

*Figure 10.* Cumulative shares of all systems, weighted to the catalogue, with classified templates. A system without an orbit never enters. Table: [`fig10-orbits.csv`](gaia-rvs-eclipsing-binaries/tables/fig10-orbits.csv).


With classified templates, both minimum masses are within 10 percent for
52 percent of the systems brighter than G_RVS = 9, 45
percent at 9 to 11 and 23 percent at 11 to 12.5. They are within 3
percent for 25, 20 and 6 percent, and
within 1 percent for 8, 4 and 1
percent. For the recovered systems the pulls of the two semi-amplitudes have robust widths
of 0.95 and 0.99 with the injected templates and of
1.62 and 2.40 with classified ones, and the systemic
velocity a scatter of 0.19 and 0.44 km/s. The
quoted errors of an orbit contain the photon noise of its transits and not the mismatch
of its templates.

## What the eclipse ephemeris adds

[![Figure 11](gaia-rvs-eclipsing-binaries/figures/fig11-ephemeris-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig11-ephemeris-light.png)
[![Figure 11](gaia-rvs-eclipsing-binaries/figures/fig11-ephemeris-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig11-ephemeris-dark.png)

*Figure 11.* Recovery with the ephemeris held and with the period alone, weighted to the catalogue's classes. Bars: Wilson intervals at one sigma for the effective number of systems. Table: [`fig11-ephemeris.csv`](gaia-rvs-eclipsing-binaries/tables/fig11-ephemeris.csv).


A fit that holds the period and the time of conjunction has two parameters fewer than one
that knows the period alone, and the ephemeris also states which star recedes after the
primary eclipse. The fit with the period alone cannot name the two stars, so its
semi-amplitudes are compared with the injected ones in the order that fits, which
20 percent of its tables needed. That comparison gives it the naming for
nothing. With it, the largest difference between the two fits in any half magnitude is
2 percentage points: at a known period the ephemeris adds the names of
the stars, and not recovered orbits. The two fits recover 56 and 55
percent of the systems brighter than G_RVS = 11, 33 and 33
percent at 11 to 12.5, and 6 and 7 percent beyond.

## What the templates cost

[![Figure 12](gaia-rvs-eclipsing-binaries/figures/fig12-templates-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig12-templates-light.png)
[![Figure 12](gaia-rvs-eclipsing-binaries/figures/fig12-templates-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig12-templates-dark.png)

*Figure 12.* Recovery of the systems with a flux ratio above 0.2 under the three declarations. Bands: Wilson intervals at one sigma. Table: [`fig12-templates.csv`](gaia-rvs-eclipsing-binaries/tables/fig12-templates.csv).


Among the systems with a flux ratio above 0.2 and G_RVS below 10, where photons do not
limit, the injected templates recover 99,
100, 99 and
95 percent of the K, G, F and A classes, classified templates
98, 98,
85 and 69 percent, and one
catalogue template 89, 92,
77 and 46 percent.

## Effects outside the baseline

Twenty systems of every cell were simulated again with one effect added, with the same
transit times and seeds, and compared with their baseline in the design run. The shares
are those of that subsample, which has equal numbers at every magnitude and class. The
scatter is that of the primary's velocity in km/s over the transits usable in both runs.
The third star is that of the population model: present in a share of the systems that
falls with the period from 96 percent, of 0.1 to 1.2 times the primary's mass and of its
age, at the systemic velocity of the pair to within a few km/s, and unresolved in 68
percent of cases (Tokovinin et al. 2006). The analysis is not told of it.

The emission is that of saturated activity. Every star cooler than 6500 K with a Rossby
number below 0.13 (Wright et al. 2011) has an excess equivalent width between 0.3 and
1.0 Angstrom in the core of Ca II 8542, and 0.6 and 0.8 of it in the two other lines,
which no template has. RAVE measured excesses per line from -0.2 to 1 Angstrom or more in
this band (Zerjal et al. 2013), so the run is at the strong end of observed activity: at
the middle of its range the core of the 8542 line of a G dwarf is filled to about the
continuum. The instrument run draws the background of every transit log-uniformly
between 1/5.5 and 5.5 times the mission median and leaves in the normalisation of a
transit a residual scale, slope and curvature, each with a standard deviation of 2
percent, and the analysis fits the scale of every transit.

[![Figure 13](gaia-rvs-eclipsing-binaries/figures/fig13-perturbations-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig13-perturbations-light.png)
[![Figure 13](gaia-rvs-eclipsing-binaries/figures/fig13-perturbations-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig13-perturbations-dark.png)

*Figure 13.* Paired changes against the design run, with classified templates. Bars: 16th to 84th percentiles of a bootstrap over systems. Table: [`fig13-perturbations.csv`](gaia-rvs-eclipsing-binaries/tables/fig13-perturbations.csv).


| effect | systems affected | recovered before [%] | recovered with it [%] | change of the primary's scatter [%] |
|---|---|---|---|---|
| an unmodelled third star | 590 | 51 | 33 | +28 |
| a third star above 10 percent of the light | 259 | 51 | 12 | +165 |
| Ca II emission in the line cores | 530 | 45 | 28 | +79 |
| background and normalisation of a transit | 1200 | 50 | 52 | +8 |
| the same, systems fainter than G_RVS = 11 | 400 | 25 | 26 | +4 |

Third light and emission are the two effects that matter, and they act in opposite
directions. Among the affected systems with a flux ratio above 0.2 and G_RVS below 11,
the median errors of the two semi-amplitudes are -6.7 and
-29.2 percent under a third light above a tenth: the lines of the third
star stand at the systemic velocity and draw both measured velocities towards it. Under
the emission they are +2.1 and +3.6 percent. A varying background
leaves them at +0.1 and -0.1 percent, since the
transits with less background than the median gain what those with more lose.

The transits in eclipse were measured with free amplitudes. With the injected templates the
primary's velocity is within three predicted errors of the injected one at
78 percent of them, against 94 percent of the transits
out of eclipse, before any distortion of the lines.

## A recovery function

[![Figure 14](gaia-rvs-eclipsing-binaries/figures/fig14-function-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig14-function-light.png)
[![Figure 14](gaia-rvs-eclipsing-binaries/figures/fig14-function-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig14-function-dark.png)

*Figure 14.* Recovery against the predicted S/N of the secondary's semi-amplitude, with the fitted functions. Table: [`fig14-function.csv`](gaia-rvs-eclipsing-binaries/tables/fig14-function.csv).


The predicted S/N of the secondary's semi-amplitude is its injected value times the square
root of the summed inverse variances of the photon-limited errors of its transits, each
weighted by the square of the velocity curve. It needs the labels, the magnitude, the
period and the number of transits, and no simulation. The share recovered to 10 percent
follows `top / (1 + exp(-(log10 x - log10 x_half) / w))` with

| declaration | top | x_half | w |
|---|---|---|---|
| injected | 1.00 | 22 | 0.15 |
| classified | 0.99 | 47 | 0.25 |
| catalogue | 0.92 | 69 | 0.32 |

## Expected yield for the catalogue

[![Figure 15](gaia-rvs-eclipsing-binaries/figures/fig15-yield-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig15-yield-light.png)
[![Figure 15](gaia-rvs-eclipsing-binaries/figures/fig15-yield-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig15-yield-dark.png)

*Figure 15.* Cumulative numbers weighted to the DR3 catalogue. Bands: 16th to 84th percentiles of a bootstrap over the simulated systems of each cell. Table: [`fig15-yield.csv`](gaia-rvs-eclipsing-binaries/tables/fig15-yield.csv).


| limiting G_RVS | candidates | double-lined orbits | masses to 10 percent | masses to 3 percent |
|---|---|---|---|---|
| 10 | 1,417 | 849 (829 to 868) | 708 | 350 |
| 11 | 3,838 | 2,108 (2,067 to 2,159) | 1,758 | 777 |
| 12 | 10,082 | 4,789 (4,674 to 4,912) | 3,750 | 1,419 |
| 13 | 23,823 | 6,784 (6,576 to 6,968) | 4,796 | 1,552 |
| 13.5 | 36,131 | 7,123 (6,897 to 7,323) | 4,880 | 1,552 |

Numbers of systems with classified templates, weighted to the DR3 candidates of the
detached classes in the four temperature classes. A double-lined orbit is one whose
secondary is detected and whose semi-amplitudes are both within 10 percent. The ranges are
the 16th to 84th percentiles of a bootstrap over the simulated systems of each cell.
Brighter than 13.5, 19.7 percent of the candidates have a double-lined orbit
and 13.5 percent masses to 10 percent. With the weights raked to the
catalogue's periods, which cover 91 percent of the candidates, the shares
are 20.5 and 14.0 percent. Gaia DR3 has
5376 double-lined orbits, none fainter than G = 12, and 483 of them among its
eclipsing-binary candidates brighter than G = 11.

## Examples

[![Figure 16](gaia-rvs-eclipsing-binaries/figures/fig16-cases-light.png#only-light)](gaia-rvs-eclipsing-binaries/figures/fig16-cases-light.png)
[![Figure 16](gaia-rvs-eclipsing-binaries/figures/fig16-cases-dark.png#only-dark)](gaia-rvs-eclipsing-binaries/figures/fig16-cases-dark.png)

*Figure 16.* Six systems with classified templates. Lines: the injected curves. Filled points: usable transits as assigned by the ephemeris. Open points: flagged transits as measured. Table: [`fig16-cases.csv`](gaia-rvs-eclipsing-binaries/tables/fig16-cases.csv).


## Checks of the experiment

- **Recorded systems.** The 33 systems of the recorded disentangling benchmarks,
  simulated from their recorded seeds and measured by the same code with the injected
  templates, have 93.4 percent of their epochs usable as measured, and both
  semi-amplitudes within 5 percent for 21 of 22 systems whose lines separate by more
  than 100 km/s and 9 of 11 of the others. The
  [TODCOR notebook](../tutorials/gaia-rvs-todcor.ipynb) has 93.4 percent, 21 of 22 and 9
  of 11.
- **Single stars.** The primaries of the null run, measured with their own template
  alone, have a scatter about the velocity in the spectrum of 0.87 times the
  photon-limited prediction at S/N above 10, and pulls of robust width 0.96.
  For those of 5500 to 6500 K that rotate below 20 km/s the scatter of a transit about the
  orbit is 0.20 km/s at G_RVS 6.0 to 6.5, 0.30 km/s at 7.5 to
  8.5, 0.73 km/s at 9.5 to 10.5 and 2.63 km/s at 11.5 to
  12.0. The errors that Gaia DR3 publishes for such stars give 0.20,
  0.36, 1.07 and 4.31 km/s in the same ranges:
  1.0, 1.2, 1.5 and 1.6
  times the simulated scatter
  ([`check-single-stars.csv`](gaia-rvs-eclipsing-binaries/tables/check-single-stars.csv)).
- **Workers.** The tables of 96 systems are identical when they are run by one process and
  by six.
- **Failures.** 0 of the 7,800 systems raised an error.

## Limits

- The transits of the simulation are at the photon limit. Their S/N is that of the
  science-performance model, which reproduces the S/N that DPAC expects for a source
  (`rv_expected_sig_to_noise`), with the mission's median background at every transit, and
  the velocities reach the precision that this S/N allows. The scatter that the errors of
  Gaia DR3 imply for one transit of a slowly rotating dwarf of 5500 to 6500 K is larger:
  1.2, 1.5 and 1.6 times that of the
  simulated single stars at G_RVS = 8, 10 and 12. With the zero point taken from both, a
  transit of DR3 has the scatter of a simulated transit 0.5,
  0.7 and 0.6 mag fainter. The published number is a
  standard deviation over transits whose background differs by a factor of a few, and it
  contains whatever the DR3 pipeline lost; this page does not separate the two. The
  background accounts for a small part: one that is up to 5.5 times the median or below
  it, as in the instrument run, raises the standard deviation of the noise of a transit
  by 6 percent at G_RVS = 10 and 15 percent at 12 in the S/N
  model, and changes the robust scatter of the primary's velocity by
  +4 percent for the systems fainter than 11. If the
  transits of DR4 are as DR3 measured them, the magnitude at which a given share of
  systems is recovered is brighter than quoted here by about those amounts, and the
  yields are lower.
- The data and the templates come from one library. The `classified` and `catalogue`
  declarations and the emission run introduce mismatch, but not the difference between a
  model atmosphere and a star.
- The line-spread function is Gaussian here. The DR4 grid, the normalisation and the
  selection flags are those of the draft data model.
- The stars are spheres on Keplerian orbits: no ellipsoidal distortion, reflection or
  spots, and no distortion of the lines in eclipse.
- The tracks are of solar composition, and the metallicity enters the spectra only.
- The weights are those of DR3 candidates of light-curve classes, of which the detached
  systems are a part, and the yield assumes classified templates for every system, no
  third light and no emission.

## Reproduction

```bash
python scripts/build_mist_tracks.py --check
python scripts/rvs_eb_run.py draw
python scripts/rvs_eb_run.py run design
python scripts/rvs_eb_run.py run null
python scripts/rvs_eb_run.py run third-light
python scripts/rvs_eb_run.py run activity
python scripts/rvs_eb_run.py run instrument
python scripts/rvs_eb_run.py run recorded
python scripts/rvs_eb_report.py
```

The three BOSZ boxes must be in the albireo cache (`albireo.fetch_library`). The manifests
of the runs are in
`gaia-rvs-eclipsing-binaries/manifests`, beginning with
[`design.json`](gaia-rvs-eclipsing-binaries/manifests/design.json), and the population in
[`tables/population.csv.gz`](gaia-rvs-eclipsing-binaries/tables/population.csv.gz).

## References

- Choi, J., Dotter, A., Conroy, C., et al. 2016, ApJ, 823, 102
- Eker, Z., Bakis, V., Bilir, S., et al. 2018, MNRAS, 479, 5491
- Katz, D., Sartoretti, P., Guerrier, A., et al. 2023, A&A, 674, A5
- Meszaros, Sz., Bohlin, R., Allende Prieto, C., et al. 2024, A&A, 688, A197
- Moe, M. & Di Stefano, R. 2017, ApJS, 230, 15
- Mowlavi, N., Holl, B., Lecoeur-Taibi, I., et al. 2023, A&A, 674, A16
- Tokovinin, A., Thomas, S., Sterzik, M., & Udry, S. 2006, A&A, 450, 681
- Wright, N. J., Drake, J. J., Mamajek, E. E., & Henry, G. W. 2011, ApJ, 743, 48
- Zerjal, M., Zwitter, T., Matijevic, G., et al. 2013, ApJ, 776, 127
- Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806
- Zucker, S. 2003, MNRAS, 342, 1291
