# Correlation survey of a simulated population

!!! warning "Experimental"

    The configuration, the columns of the tables and the layout of a run may change.

The [benchmark](gaia.md) disentangles every system and costs minutes per system.
`albireo.survey` measures a population by correlation alone. Every system is simulated as
Gaia RVS epoch spectra with its eclipses, both velocities are measured at every transit
against library templates ([`todcor`](todcor.md)), and an orbit is fitted at the photometric
ephemeris ([`assign_by_ephemeris`](rvorbit.md)). One system takes a few seconds, so a
population of 10^4 systems is a run of two to three hours on a desktop. The experiment
made with it is reported in [Gaia DR4 eclipsing binaries in the
RVS](../reports/gaia-rvs-eclipsing-binaries.md).

```python
from albireo.eclipsing import draw_eclipsing_population
from albireo.survey import SurveyConfig, collect, read_tables, run_survey, run_system

systems = draw_eclipsing_population(10, seed=1)
result = run_system(systems[0], SurveyConfig())            # one system, three declarations
run_survey(systems, SurveyConfig(), "runs/first", workers=6)
manifest = collect("runs/first")
table, epochs = read_tables("runs/first")
```

## Declarations and variants

A system is measured under several declarations of what the analysis knows about the
templates, on the same epochs.

| declaration | templates | light fractions |
|---|---|---|
| `injected` | the injected labels and rotation | held at the injected values |
| `classified` | labels with the errors of a classification, drawn once per system | measured |
| `catalogue` | one template for both stars, from the colour of the pair | measured |

A run simulates one variant.

| variant | what is simulated |
|---|---|
| `baseline` | the pair with its eclipses |
| `null` | the primary alone at the system's magnitude, analysed with the two templates of the pair |
| `third-light` | the tertiary of `meta["tertiary"]` added, unknown to the analysis |
| `activity` | emission in the cores of the Ca II triplet of the stars with `meta["activity_ew"]` above zero |
| `instrument` | a background that varies between transits and a residual scale, slope and curvature of the normalisation, analysed with `scale="free"` |

Every system is simulated from its own seed, `meta["seed"]`, so a variant has the transit
times and the noise seeds of the baseline, and the tables do not depend on the number of
worker processes or on the size of the chunks.

A run is kept in one directory, one file per chunk, and is continued by the same call. The
directory records what its chunks were made from (`run.json`: a digest of the systems, the
configuration and the chunk size), and a call with other systems, another configuration or
another chunk size is refused. The systems are compared by their content, since the names
of a drawn population are positional. The manifest that `collect` writes gives the commit
and whether the package or the scripts had changes that the commit does not hold.

## The chain for one system

1. **Simulation.** The transit times follow the scanning law over the span of DR4. Each
   star is rendered from its library box with its rotation and limb darkening. The light
   fractions and the S/N of every transit follow from the eclipse geometry
   ([`eclipse_light`](eclipsing.md)). A velocity offset common to both stars is added per
   transit, and each transit takes the resolving power of its CCD row.
2. **Correlation.** Transits out of eclipse are measured with the declared light
   fractions, held or measured on those transits. Transits in eclipse, which the ephemeris
   identifies, are measured with the amplitudes of each epoch and do not enter the orbit.
3. **Orbit.** The semi-amplitudes and the systemic velocity are fitted with the period and
   the time of the primary's eclipse held, and the eccentricity fitted where the injected
   orbit is eccentric. The table is also fitted with the period alone
   ([`assign_components`](rvorbit.md)).
4. **Single-lined.** The transits out of eclipse are also measured with the first template
   alone, and one semi-amplitude is fitted at the ephemeris.

With identical templates the correlation does not name the stars and the ephemeris does:
the table is fitted in both orders of its components and the order of the lower chi-square
is kept.

## The tables

`collect` writes two tables. `systems.npz` has one row per system and declaration, and
`epochs.npz` one row per epoch and declaration. `system` is the index of the system in
`population.json`, `decl` the index of the declaration, and `epoch` the index of the
transit.

**Epochs, injected.**

| column | content |
|---|---|
| `bjd`, `phase` | time of the transit, and its phase from the primary's eclipse |
| `snr`, `resolving_power` | S/N per detector pixel and resolving power applied |
| `in_eclipse`, `flux`, `separation` | whether either star is partly hidden; the light relative to that out of eclipse; the projected separation in units of the sum of the radii |
| `l1_true`, `l2_true` | light fractions of the two stars at the transit |
| `v1_true`, `v2_true`, `zero_point` | the orbit's velocities, and the offset added to both in the simulation |
| `pred1`, `pred2` | photon-limited error of each velocity for the star alone in the spectrum |

**Epochs, measured.**

| column | content |
|---|---|
| `v1`, `v2`, `s1`, `s2`, `rho` | velocities and quoted errors as measured, and their correlation |
| `l1`, `l2`, `d1`, `d2` | amplitudes of the templates and the detection statistic of each |
| `blended`, `second`, `edge`, `margin`, `a1`, `a2` | the flags of the [velocity table](todcor.md), and the velocities of the other minimum |
| `chi2`, `chi2_null`, `n_pix` | the fit of the transit |
| `ve1`, `ve2`, `se1`, `se2`, `ge`, `xe`, `oe` | velocities and errors after the assignment by the ephemeris; whether the transit is usable; whether it was exchanged; whether the other minimum was taken |
| `vp1`, `vp2`, `sp1`, `sp2`, `gp`, `xp`, `op` | the same after the assignment with the period alone, in the order of the components that fits the injected velocities |
| `vs`, `ss`, `gs` | velocity, error and flag of the first template alone |

**Systems.**

| column | content |
|---|---|
| `n_epochs`, `n_in_eclipse`, `n_out`, `n_good` | transits; those in eclipse; those out of eclipse; those usable as measured |
| `k1_true`, `k2_true`, `gamma_true`, `light1_true`, `light2_true`, `light3_true` | injected values |
| `quality1`, `quality2`, `snr_k1`, `snr_k2` | velocity information of each spectrum per unit S/N, and the predicted S/N of each semi-amplitude |
| `light1`, `light2`, `fallback` | light fractions as measured, and whether no transit gave both a positive amplitude |
| `e_k1`, `e_k2`, `e_k1_err`, `e_k2_err`, `e_gamma`, `e_gamma_err`, `e_ecc`, `e_omega`, `e_chi2`, `e_npts`, `e_npar` | the fit at the held ephemeris |
| `e_ok`, `e_nx`, `e_no`, `e_held`, `e_light1`, `e_light2`, `swapped` | whether it was made; transits exchanged relative to the table as measured; other minima taken; components held; light fractions of the two stars; under identical templates, whether the fit kept is that of the table with its two components interchanged as a whole (with alike light fractions both orders reach the same assignment and the first is recorded) |
| `p_*` | the same for the fit with the period alone; `p_swapped` marks a table compared in the other order |
| `s_k1`, `s_k1_err`, `s_gamma`, `s_gamma_err`, `s_chi2`, `s_npts`, `s_light`, `s_ok` | the fit of the first template alone |
| `sum_d1`, `sum_d2`, `med_d2` | detection statistics summed over the usable transits |
| `t_sim`, `t_todcor`, `t_orbit` | wall-clock seconds of the stages |

## Cost and memory

The design run of the report, 7,800 systems and 313,125 transits under
three declarations, took 109 minutes on 8 processes of a 32-thread
desktop that was shared with other work: 1.2 systems per second, at
1.0 GB per process. The stages of one system sum to 6.4 s with the
eight processes running (0.2 s for the simulation, 5.7 s for the
correlations and 0.6 s for the orbit fits) and to 4.8 s in one
process alone.

The orbit fit with a free conjunction compiles one set of array programs for every number
of usable epochs, and a process that fitted the tables of a population grew by 16 MB per
system. The survey evaluates that fit on epochs padded to whole blocks, which removes the
growth and changes the fitted values by at most 1.4e-10 of their size.

::: albireo.survey
