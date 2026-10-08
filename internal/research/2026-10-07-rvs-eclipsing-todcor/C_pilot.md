# C. Pilot measurements for the plan (2026-10-07)

Measured on the development desktop (Ryzen 9 9950X3D, 16 cores and 32 threads, 31 GB, CPU
only, float64) at commit `77a3a32`, while the machine was shared with other jobs (6 to 12 GB
free). The scripts were throw-away and are not kept; each measurement is described well
enough to be repeated, and the driver of the experiment repeats all of them in its own pilot
stage. Wall-clock times are from one process unless stated, and are indicative only: the runs
were not made alone.

The systems are `draw_population(kind="eclipsing", library="bosz2024-fgk-rvs",
period_range=(0.5, 20), q_min=0.4)`, simulated as DR4 epoch spectra with the scanning-law
cadence (9 to 56 epochs per system in C1, median 29) and measured with the injected templates
on a grid of three pixels per line-spread sigma, a search range of 350 km/s on either side,
the delivered line-spread width and the lag-one noise correlation declared.

## C1. Cost per stage with the shipped calls

Eight systems, each at the G_RVS it was drawn with (7.3 to 10.4).

| stage | call | median per system [s] | per epoch [ms] |
|---|---|---|---|
| simulation | `benchmark.simulate_system` | 1.42 | 47 |
| two templates | `Template.from_library` | 0.13 | 4.3 |
| correlation, light fractions held | `todcor(light=(l1, l2))` | 0.32 | 9.2 |
| correlation, light fractions measured | `todcor(light="global")` | 0.94 | 30 |
| orbit at the known period | `rvorbit.assign_components` | 0.38 | 12 |

The resident memory of the process rose from 0.46 to 1.47 GB over the eight systems.

## C2. The cause of the simulation cost and of the memory growth

A profile of `simulate_system` over eight systems: 11.6 s in total, of which 7.5 s in
`jax._src.compiler.backend_compile_and_load`, 471 compilations (59 per system). The model
grid of `rvs_model_grid` depends on each system's velocity amplitude and rotation, so every
system has arrays of a new length and every un-jitted JAX primitive applied to them is
compiled again. Each compilation is retained.

| variant | per system [s] | memory growth over 10 systems [GB] |
|---|---|---|
| (a) shipped: one model grid per system | 1.43 | 0.98 |
| (b) one model grid shared by all systems (4977 pixels of 2 km/s) | 0.27 | 0.11 |
| (c) as (b), the library resampled onto the grid once and the components rendered from it | 0.15 (components 0.014, simulation and delivery 0.135) | 0.02 |
| (d) as (c), one resolving power per transit | 0.13 | 0.04 |

The components and the templates rendered from the library resampled once are equal to
those of `rvs_components` and `Template.from_library` to the last bit (largest difference
0.0 on both). With the library resampled once onto the template grid (2400 pixels of
3.89 km/s), two templates take 0.013 s.

In (c) and (d) the ten systems were those of (a) and (b), so every epoch count had been
compiled already. On 60 new systems the simulation took 0.49 s per system: the Keplerian
velocities and the per-epoch loop are evaluated by un-jitted JAX on arrays whose length is
the number of epochs, and each new epoch count compiles again.

## C3. The whole chain on 60 systems, steady state

One shared model grid, one shared template grid, the library resampled once on each, one
resolving power per transit, the systems placed on a ladder of G_RVS (10 per magnitude).

| stage | median per system [s] | per epoch [ms] |
|---|---|---|
| simulation | 0.49 | 10.7 |
| two templates | 0.013 | 0.4 |
| correlation, light fractions held | 0.38 | 11.2 |
| correlation, light fractions measured | 0.99 | 32.6 |
| orbit at the known period | 0.29 | 8.1 |
| all | 2.34 | |

The resident memory rose from 0.56 to 2.54 GB over the 60 systems, about 33 MB per system,
and from 0.32 to 3.13 GB over 120 systems in the run of C6, which has no second correlation.
The growth slows as epoch counts repeat (57 MB per system over the first 24 systems of C6,
the first compilations included, and 10 MB per system over the last 24) and is attributed to
compilations per array length in the simulation and in the orbit fits. It was not traced
stage by stage.

No call of `todcor` raised an error on the 60 systems, at any magnitude. Five orbit fits
raised `ValueError` for too few usable velocities, all at G_RVS 12 and 13.

## C4. Processes in parallel

Four processes of 24 systems each, started together, against one process (2.4 s per system
over its first 20 systems).

| threading | per system in each process [s] | wall-clock time of the four [s] | throughput against one process |
|---|---|---|---|
| default | 2.80 to 3.04 | 79 | 3.3 |
| one thread per process (`XLA_FLAGS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) | 2.54 to 2.74 | 71 | 3.6 |

## C5. First look at the dependence on magnitude

The 60 systems of C3 (F and G dwarfs, mass ratios above 0.4, periods of 0.5 to 20 d,
injected templates, light fractions held). The epoch statistics are medians over the ten
systems of a magnitude. The orbit is that of `assign_components` at the known period, and
the semi-amplitudes are compared in the order that fits. The epoch errors are in the order
of the table as measured, so they include the epochs of alike pairs returned in the other
order.

| G_RVS | S/N per pixel | usable epochs | median error, primary and secondary [km/s] | systems with K1, with K2 within 5 percent |
|---|---|---|---|---|
| 7 | 82 | 0.95 | 0.28, 0.43 | 10, 10 of 10 |
| 9 | 32 | 0.96 | 1.00, 1.48 | 10, 10 of 10 |
| 10 | 18 | 0.96 | 1.64, 3.54 | 10, 10 of 10 |
| 11 | 9.7 | 0.96 | 3.73, 5.36 | 10, 9 of 10 |
| 12 | 4.6 | 0.89 | 13.2, 13.6 | 8, 8 of 10 |
| 13 | 2.0 | 0.21 | 28.6, 51.2 | 1, 1 of 10 |

One system with a mass ratio of 0.47 and a flux ratio of 0.03 at G_RVS 7.3 (S/N 72) gave
K2 = 175.6 km/s for an injected 177 km/s.

## C6. The orbit with the shape of the velocity curve known

120 systems, 20 per magnitude. The table of each (injected templates, light fractions held)
was fitted in three ways: by `assign_components` at the known period; by a linear fit of
the systemic velocity and the two semi-amplitudes with the shape of the velocity curve taken
as known, on the usable epochs; and by the same fit on every measured epoch, flagged or not.
The linear fit orders the two velocities of an epoch by the sign of the curve and then by its
own solution, and leaves out residuals beyond four robust standard deviations. Counts are of
systems with both semi-amplitudes within 5 percent and within 10 percent of the injected
values.

| G_RVS | S/N per pixel | period known: 5 %, 10 % | shape known, usable epochs | shape known, all measured epochs |
|---|---|---|---|---|
| 10.0 | 18.2 | 19, 19 of 20 | 18, 19 | 20, 20 |
| 11.0 | 9.7 | 18, 19 | 18, 19 | 18, 19 |
| 12.0 | 4.6 | 10, 14 | 12, 15 | 12, 16 |
| 12.5 | 3.1 | 6, 8 | 7, 15 | 6, 12 |
| 13.0 | 2.0 | 1, 2 | 2, 3 | 0, 7 |
| 13.5 | 1.3 | 0, 0 | 0, 0 | 1, 3 |

With 20 systems per row the counts at 5 percent do not distinguish the three. At 10 percent
the known shape adds systems at G_RVS 12.5 and 13. The prototype took the shape from the
injected orbit, eccentricity and argument of periastron included, which is more than a light
curve gives for an eccentric orbit.

## C7. After the correction of the simulation path (2026-10-07)

The causes found in C2 and C3 were removed the same day (plan section 10.1). The
measurements below are from the shipped calls, `benchmark.simulate_system` and
`Template.from_library`, on 24 and on 100 drawn systems, half of them with one resolving
power per transit. The machine was shared, so the times are indicative.

| quantity | before | after |
|---|---|---|
| distinct lengths of the model grid over 24 systems | 21 (4380 to 4570 pixels) | 1 (4608) |
| XLA compilations per system, after the first systems | 59 | 0 |
| simulation per system [s] | 1.36 | 0.13 to 0.17 |
| resident memory after 24 systems [GB] | 2.40 | 0.77 |
| growth of the resident memory | 87 MB per system | none from the 20th to the 60th system (0.74 GB); 0.80 GB once a second grid length had occurred |
| two templates on a shared grid [s] | 0.16 | 0.012 |
| `radial_velocity` outside `jit`, per call, 1500 calls on one shape | 29 to 40 ms, one compilation, 1.9 MB retained | 0.47 ms, none, none |
| a new number of epochs, 60 counts from 2 to 120 | 17 compilations, 0.38 s, 28 MB | 0.9 compilations, 0.02 s, 1.6 MB (blocks of 32 epochs) |

**Equality.** The 24 systems were simulated before and after with the same seeds. The
delivered flux and inverse variance, the noiseless detector spectra, the injected
velocities, the S/N and the lag-one correlation of every epoch, 16 templates and a
resampled library are equal bit for bit (186 arrays). The 24 arrays of component spectra
differ in the last 3 to 47 pixels of the unrounded grid, by up to 0.014 of the continuum,
where the rotation kernel of the unrounded grid met its zero-filled end. Those pixels lie in
the margin and do not reach the detector, as the equal detector spectra show. The Kepler
solver in its new form equals the old one bit for bit in 34 comparisons: outside `jit` for
1 to 200 anomalies and scalar and array eccentricities, under `jit`, under `vmap`, and
through `grad` and `jacfwd`. Padding the epochs to a block changed no bit for 2 to 140
epochs; for a single epoch it changed the last bit of the velocity-to-pixel conversion, so a
single epoch is not padded.

**What still compiles.** In the whole chain on 80 systems the simulation and the templates
compile nothing after the first ten systems. `todcor` compiled about one program per system
(5 to 22 per ten systems), all of them in the conversion of the dataset's barycentric
velocities to pixels, an array of the length of the dataset (`todcor.py`, `_run`); it was
moved to the same blocks of epochs, and `todcor` then compiles nothing after the first ten
systems. `assign_components` compiles two to four programs per system (9 to 45 per ten
systems, falling as the numbers of usable epochs repeat; 44 distinct counts in 80
systems): the compiled objective, its Jacobian and the predictor of `fit_rv_orbit` take
arrays of the length of the usable epochs (`rvorbit.py` lines 894, 908, 933, 984 and 585
at `77a3a32`). The resident memory rose from 1.12 GB after 10 systems
to 1.68 GB after 80, 8 MB per system, against 33 MB per system in C3. The orbit fit was
left as it is: padding its epochs changes the order of its sums, and its results are
pinned bit for bit by a regression test.

Stage costs in that run, per system: simulation 0.12 to 0.15 s, two templates 0.011 to
0.012 s, `todcor` with held light fractions 0.31 to 0.43 s, `assign_components` 0.09 to
0.30 s.
