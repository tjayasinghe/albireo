# Does the model grid's own smoothing bias v sin i in the epoch comparison, and should the operator give it back? (D65, 2026-09-16)

After the epoch comparison became the default of `Fit.match_labels`, the slow closed loop
`tests/test_pipeline.py::test_the_pipeline_recovers_the_injected_system` returned
$`v\sin i_A = 6.9`$ km/s for an injected 11.0, outside its 25% tolerance. Its fixture has native
pixels of 0.08 A (4.63 km/s at 5185 A), a Gaussian LSF of $`\sigma = 5.5`$ km/s and a simulator
grid of 2 km/s, and the facade's default model grid takes the native pixel. With the orbit fixed
at the truth, the epoch comparison had returned 7.4 km/s on that grid and 10.1 km/s on a 2 km/s
one. This note measures the discretisation smoothing that the hypothesis blames, the bias it
causes on a grid of model spacings and rotations, a compensation through the operator's LSF
width, and the same compensation on three Gaia RVS benchmark products. Scripts, per-case JSON,
logs and tables are in the session scratchpad `wp-ak/`: `mechanism.py` (the smoothing),
`labels.py` (the fixture at the true orbit), `pipeline_check.py` (the failing test end to end),
`rvs_check.py` (the benchmark products) and `table.py`, whose output is `tables.md`. Nothing
under `src/` or `tests/` was edited; the compensation was applied by passing the shipped
`Fit.epoch_statistics` a resolving power whose width is the one to be removed.

## The smoothing

**Every discrete step between a template and the epoch pixels adds a variance proportional to
$`\Delta v^2`$, and with uniform phases the steps add up to $`7\,\Delta v^2/12`$.** The label rows
(`match._component_model`) rebin the library onto the model grid, convolve with the
pixel-integrated rotation kernel and shift by the frame velocity $`v`$ with
`operators.shift_spectrum`. The operator $`A`$ (`forward._epoch_model`) shifts each epoch with the
same function, convolves with the sampled Gaussian LSF and rebins onto the native pixels, reading
each model pixel as constant over its width. A shift by a fraction $`f`$ of a pixel is the
two-tap kernel with weights $`1-f`$ at $`-f\,\Delta v`$ and $`f`$ at $`(1-f)\,\Delta v`$, of
mean zero and variance $`f(1-f)\,\Delta v^2`$, which averages to $`\Delta v^2/6`$ and peaks at
$`\Delta v^2/4`$; a box of width $`\Delta v`$ has variance $`\Delta v^2/12`$. Hence

```math
\sigma_{\rm grid}^2 = \Delta v^2 \Big[
\underbrace{\tfrac{1}{12}}_{\rm library\ bin} + \underbrace{\tfrac{1}{12}}_{\rm rotation\ kernel}
+ \underbrace{f_v(1-f_v)}_{{\rm shift\ by}\ v}
+ \underbrace{\langle f_e(1-f_e)\rangle}_{\rm epoch\ shifts}
+ \underbrace{\tfrac{1}{12}}_{\rm model\ pixel\ in\ the\ rebin} \Big]
\;\longrightarrow\; \tfrac{7}{12}\,\Delta v^2 ,
```

of which $`\Delta v^2/4`$ is in $`A`$ and $`\Delta v^2/3`$ in the rows. The sampled Gaussian LSF adds
nothing measurable: its variance is short of $`\sigma^2`$ by at most 0.023 km$`^2`$ s$`^{-2}`$ for
$`\sigma \geq 0.85`$ pixel, by 0.07 to 0.09 at 0.65 pixel and by 35% of $`\sigma^2`$ at 0.43 pixel.

**Measured through NumPy replicas of the four operators, which match the JAX originals to
$`4\times10^{-16}`$, the terms are additive to 0.02 km$`^2`$ s$`^{-2}`$ and independent of the line
width.** A Gaussian line of 14.5 km/s (the fixture library's width) or 2 km/s was rendered on a
0.116 km/s grid, taken through the chain at 128 positions around 5185 A, and the central second
moment of the native-pixel profile was compared with that of the continuous line convolved with
the LSF and averaged over the same native bins. Excess variance in km$`^2`$ s$`^{-2}`$, with
$`f(1-f)\,\Delta v^2`$ in parentheses:

| $`\Delta v`$ (km/s) | model pixel in the rebin, at 5185 A | $`\Delta v^2/12`$ | library bin | epoch shift, $`f=0.25`$ | epoch shift, $`f=0.5`$ | $`f_v=f_e=0.5`$ | rotation kernel, $`v\sin i`$ = 5 / 11 / 25 | $`7\Delta v^2/12`$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5 | 0.170 | 0.188 | 0.189 | 0.426 (0.422) | 0.567 (0.562) | 1.136 (1.125) | 0.21 / 0.20 / 0.17 | 1.31 |
| 2.0 | 0.320 | 0.333 | 0.334 | 0.752 (0.750) | 1.002 (1.000) | 2.004 (2.000) | 0.23 / 0.28 / 0.30 | 2.33 |
| 3.0 | 0.755 | 0.750 | 0.751 | 1.680 (1.688) | 2.241 (2.250) | 4.483 (4.500) | 0.42 / 0.55 / 0.81 | 5.25 |
| 4.63 | 0.815 | 1.783 | 1.784 | 4.008 (4.012) | 5.345 (5.349) | 10.685 (10.698) | 2.81 / 1.91 / 1.81 | 12.48 |
| 6.0 | 3.016 | 3.000 | 3.001 | 6.743 (6.750) | 8.993 (9.000) | 17.999 (18.000) | 3.24 / 2.82 / 3.67 | 21.00 |

The 2 km/s line agreed with the 14.5 km/s line to 0.033 on the 1 to 4.63 km/s grids, and to 0.19 on
the 6 km/s grid, where its position-sampling error is 0.15. Two terms depend on more
than $`\Delta v`$. On the default grid the model pixel and the native pixel have equal widths at
5185 A, so their phase drifts by only 1.8 pixels over the band and the rebin term there is 0.815
rather than its band average of 1.783; the smoothing is not stationary across the spectrum. The
rotation kernel's pixel integration averages $`\Delta v^2/12`$ but oscillates when
$`v\sin i`$ is comparable with $`\Delta v`$, reaching 2.81 at 5 km/s on the 4.63 km/s grid.

**On the fixture the frame shift costs nothing on three of the four grids, because
$`\gamma = 12`$ km/s is a whole number of 3, 2 and 1.5 km/s pixels.** On the native grid
$`\gamma`$ is 2.594 pixels, so $`f_v(1-f_v) = 0.241`$ and the term is 5.16 km$`^2`$ s$`^{-2}`$.
The mean $`f_e(1-f_e)`$ over the ten epochs is 0.166 and 0.129 for A and B on the native grid and
0.152 to 0.197 on the others. The simulator runs the same chain on its 2 km/s grid without a frame
shift, so its epochs already carry $`\Delta v_{\rm sim}^2\,(3/12 + 0.186) = 1.7`$ km$`^2`$
s$`^{-2}`$ for A, and on the 2 km/s model grid the two chains are identical.

**A second-moment budget predicts the grid dependence of the fitted rotation to 0.1 km/s for A and 0.25 km/s for B.**
The limb-darkened rotation profile with $`\epsilon = 0.6`$ has variance $`0.225\,(v\sin i)^2`$,
so a model carrying an excess $`\Delta\sigma^2`$ over the data fits

```math
v_{\rm fit}^2 \simeq v^2 - \frac{\Delta\sigma^2}{0.225}, \qquad
v_{\rm fit}^2 \simeq v^2 - 2.59\,\Delta v^2 \quad {\rm for}\ \Delta\sigma^2 = \tfrac{7}{12}\,\Delta v^2 .
```

With the fixture's phases and the tabulated rotation kernels, the budget predicts
$`v\sin i_A`$ = 8.20, 10.67, 11.00 and 11.14 km/s on the 4.63, 3, 2 and 1.5 km/s grids, and the
shipped comparison returned 7.37, 9.69, 10.08 and 10.28. Relative to the 2 km/s grid the
prediction is -2.80, -0.33 and +0.14 km/s and the measurement -2.71, -0.39 and +0.20; for B
(27 km/s) it is -0.95, -0.19 and +0.04 against -0.70, -0.16 and +0.03. The bias in $`v^2`$ does not
depend on the line width or the LSF, so the fractional bias in $`v\sin i`$ scales as
$`(\Delta v / v\sin i)^2`$; the LSF enters only through how much width there is to give back.

## The bias

**On the test's fixture at the true orbit, the shipped epoch comparison underestimates
$`v\sin i_A`$ on every grid, by 3.6 km/s at 11 km/s and to the 1 km/s bound at 5 km/s on the
default grid, while Teff, log g and the light fraction do not move.** `labels.py` copies the
test's `_simulate` with $`v\sin i_A`$ set to 5, 11 or 25 km/s ($`v\sin i_B = 27`$), declares the
orbit `Fixed` at the truth, wraps the declaration in a `Fit` at its start as
`d65-verify/diag_pipeline_vsini.py` did, and calls the shipped `Fit.match_labels` with that
script's options; the 11 km/s case on the 4.63 and 2 km/s grids reproduces its 7.37, 9.03, 10.08
and 10.54 exactly. $`v\sin i_A`$ in km/s:

| $`v\sin i_A`$ | grid (km/s) | epochs | native | epochs, compensated | $`\Delta L_{\rm data}`$ at the true $`v\sin i_A`$, shipped / compensated |
| --- | --- | --- | --- | --- | --- |
| 5 | 4.63 | 1.00 (bound) | 4.91 | 3.55 | +31.2 / +3.9 |
| 5 | 3 | 1.83 | 5.02 | 4.93 | +6.3 / +0.0 |
| 5 | 2 | 2.36 | 4.93 | 4.05 | +3.6 / +0.7 |
| 5 | 1.5 | 3.18 | 4.93 | 4.06 | +2.3 / +0.9 |
| 11 | 4.63 | 7.37 | 9.03 | 10.08 | +44.4 / +2.9 |
| 11 | 3 | 9.69 | 10.96 | 10.87 | +6.0 / +0.1 |
| 11 | 2 | 10.08 | 10.54 | 10.56 | +3.4 / +0.8 |
| 11 | 1.5 | 10.28 | 11.01 | 10.54 | +2.2 / +0.9 |
| 25 | 4.63 | 23.62 | 24.16 | 24.56 | +19.2 / +2.1 |
| 25 | 3 | 24.42 | 24.90 | 24.80 | +3.8 / +0.5 |
| 25 | 2 | 24.55 | 25.09 | 24.72 | +2.2 / +0.9 |
| 25 | 1.5 | 24.62 | 24.86 | 24.72 | +1.6 / +0.9 |

$`\Delta L_{\rm data}`$ moves only $`v\sin i_A`$ to the truth and holds everything else. Over the
twelve cases the signed median error in $`v\sin i_A`$ is -1.34 km/s for the shipped epoch
comparison and -0.08 for the native one. The epoch comparison's Teff stays within 25 to 33 K of
the truth for A and 29 to 45 K for B on every grid, its log g within 0.01 dex and its light
fraction within 0.0055 of 0.62, the largest departures being the native-grid case whose rotation
sits on its bound (Teff$`_B`$ -15 K and light +0.005 against the other grids). The native
comparison's rotation is closer, but on the 4.63 and 3 km/s grids it puts Teff$`_B`$ at 4259 to
4267 K, 193 to 201 K cold, and the light fraction 0.012 to 0.016 high, against 4421 to 4463 K
and 0.005 to 0.009 on the 2 and 1.5 km/s grids.

**The bias is not the noise draw: over five noise and epoch seeds the shipped comparison on the
default grid returns 3.0 km/s low for an injected 11 km/s, and 2.72 km/s lower than on a 2 km/s
grid with a scatter of 0.19.** These runs simulate on a 0.5 km/s grid (`fx05`), whose own
smoothing is 0.15 km$`^2`$ s$`^{-2}`$, so that the data are close to continuous. The test's seed
is a low draw, 0.90 to 1.26 km/s below the five-seed mean in each of the four rows, and the formal
errors of the compensated fits, 0.53 to 0.73 km/s, are close to the seed scatter of 0.71 to 0.88.

| grid (km/s) | comparison | mean $`v\sin i_A`$, five seeds | scatter | mean minus 11 |
| --- | --- | --- | --- | --- |
| 4.63 | epochs | 7.98 | 0.88 | -3.02 |
| 4.63 | epochs, compensated | 10.76 | 0.78 | -0.24 |
| 2 | epochs | 10.70 | 0.74 | -0.30 |
| 2 | epochs, compensated | 11.14 | 0.71 | +0.14 |

Against the 2 km/s simulation the same ladder sits 0.3 km/s higher (10.08 against 9.75 on the 2 km/s
grid with seed 7), which is the 1.7 km$`^2`$ s$`^{-2}`$ that simulation carries.

## The compensation

**Reducing the operator's width by the grid variance,
$`\sigma_{\rm op}^2 = \sigma_{\rm inst}^2 - \sigma_{\rm lib}^2 - 7\Delta v^2/12`$, removes the
grid dependence to 0.39 km/s and leaves Teff, log g, [M/H] and the light fraction unchanged.** The
reduced width was put through the shipped `Fit.epoch_statistics` by declaring the resolving power
whose $`\sigma_{\rm lib}`$ equals $`\sigma_{\rm grid}`$, so that `forward.with_lsf` applied it, and
the statistics' recorded resolving power was then reset to the library's. On the test's LSF the
operator width becomes 4.215, 5.000, 5.284 and 5.379 km/s on the four grids (0.91 pixel at the
coarsest). Over the twelve cases above the signed median error in $`v\sin i_A`$ falls from -1.34 to
-0.44 km/s, the largest light error from 0.0055 to 0.0009, and Teff and log g move by at most 3 K
and 0.01 dex outside the bound case, where Teff$`_B`$ recovers the 15 K. On the five seeds the
native-grid-minus-2 km/s difference falls from -2.72 (standard error 0.08) to -0.39 km/s (0.04),
and the compensation raises $`v\sin i_A`$ by 2.78 km/s on the default grid and 0.45 km/s on the 2
km/s grid. The remaining -0.39 km/s is what the average formula leaves out: the frame shift on the
default grid costs $`(0.241 - 1/6)\,\Delta v^2 = 1.6`$ km$`^2`$ s$`^{-2}`$ more than
$`\Delta v^2/6`$, which is -0.32 km/s at 11 km/s, while on the 3, 2 and 1.5 km/s grids the fitted
$`v = 11.8`$ km/s puts $`f_v(1-f_v)`$ at 0.05 to 0.10 and the formula removes slightly too much.

**Computing the epoch-shift term from the operator's own phases changes nothing.** Replacing
$`\Delta v^2/6`$ by the mean $`f_e(1-f_e)`$ of the stars' shifts in the problem (0.148 on the default
grid) moved $`v\sin i_A`$ by at most 0.12 km/s on the twelve cases, and by 0.02 km/s on the RVS
products below. The term the operator cannot know is the frame shift, whose phase is a fitted
parameter.

**Where the grid variance exceeds the available width, a floor recovers only part of the bias,
and a finer grid is the remedy.** With the LSF narrowed to 3.0 km/s (`nl3`, simulated on 0.5 km/s),
the default grid needs 12.48 km$`^2`$ s$`^{-2}`$ out of a width of 9.00. Floored at half a pixel
(2.31 km/s, whose sampled kernel realises 4.60 km$`^2`$ s$`^{-2}`$, so 4.40 were removed),
$`v\sin i_A`$ went from 7.16 to 8.23 km/s with $`\Delta L_{\rm data}`$ at the truth still +34.2; on a
2 km/s grid, where the compensated width is 2.58 km/s, it went from 9.84 to 10.32. The floor is
reached when $`\sqrt{\sigma_q^2 - 7\Delta v^2/12} < \Delta v/2`$, that is when
$`\Delta v > 1.10\,\sigma_q`$ with $`\sigma_q^2 = \sigma_{\rm inst}^2 - \sigma_{\rm lib}^2`$.

**End to end, the compensation alone makes the failing test pass with its tolerance unchanged.**
`pipeline_check.py` runs the test's body through `run_star` (orbit and period fitted) with
`Fit.epoch_statistics` wrapped in-process:

| variant | model grid (km/s) | assertions | $`v\sin i`$ A / B | Teff A / B | log g A / B | K A / B | $`\gamma`$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shipped | 4.63 | fails on $`v\sin i_A`$ | 6.91 / 26.76 | 5206 / 4434 | 4.05 / 4.55 | 29.94 / 54.54 | 11.88 |
| compensated | 4.63 | all pass | 9.82 / 27.53 | 5207 / 4435 | 4.05 / 4.55 | 29.94 / 54.54 | 11.88 |
| shipped, `dv_kms=2` | 2.00 | all pass | 9.78 / 27.43 | 5208 / 4434 | 4.04 / 4.55 | 29.94 / 54.57 | 11.89 |
| compensated, `dv_kms=2` | 2.00 | all pass | 10.27 / 27.58 | 5208 / 4434 | 4.04 / 4.55 | 29.94 / 54.57 | 11.89 |

The compensated run clears the 8.25 km/s limit by 1.57 km/s, twice the five-seed scatter, on a
seed that is itself a low draw. The four runs shared one process in the order listed; the 2 km/s
grid took 73 to 85 s against 38 to 57 s for the default grid.

## The benchmark products

**On three Gaia RVS products the compensation leaves Teff, log g, [M/H] and the light fraction
unchanged and raises every fitted rotation, which on these slow rotators makes the benchmark's
known rotation excess larger until the archive's resampling is declared as well.** `rvs_check.py`
rebuilds `mixed-0004`, `mixed-0008` and `mixed-0012` (oracle tier) at their archived MAP with
WP-AD's `setup`, which reproduces the archive to $`5\times10^{-10}`$, and runs the shipped label fit
under box priors on five operator widths: the shipped $`\sigma_q = 9.057`$ km/s; the compensated
$`\sqrt{\sigma_q^2 - 7\Delta v^2/12} = 8.763`$ km/s on the 3 km/s grid; the same from the operator's
phases; $`\sqrt{\sigma_q^2 + 3.05^2} = 9.557`$ km/s, the width `d65_converged_labels.md` measured
for the delivered epochs; and $`\sqrt{\sigma_q^2 + \Delta_{\rm det}^2/6 - 7\Delta v^2/12} = 9.434`$
km/s, which declares the archive's linear resampling from its analytic variance
($`\Delta_{\rm det} = 0.245`$ A, 12.21 km$`^2`$ s$`^{-2}`$ at the band centre) and compensates the grid.
Errors in $`v\sin i`$ (A, B per product, in the order 0004, 0008, 0012):

| operator width (km/s) | $`v\sin i`$ errors (km/s) | median | median, primaries | largest change from shipped, Teff / log g / light |
| --- | --- | --- | --- | --- |
| 9.057, shipped | +3.91, +5.70, +1.72, +5.75, +2.22, +2.07 | +3.06 | +2.22 | 0 / 0 / 0 |
| 8.763, compensated | +5.21, +7.50, +2.52, +6.59, +3.24, +3.16 | +4.22 | +3.24 | 1.3 K / 0.003 dex / 0.0001 |
| 9.557, measured delivery | +0.08, +0.66, -0.02, +3.89, +0.10, -0.06 | +0.09 | +0.08 | 1.3 K / 0.002 dex / 0.0001 |
| 9.434, analytic delivery, compensated | +1.54, +2.15, +0.53, +4.38, +0.64, +0.51 | +1.09 | +0.64 | 1.5 K / 0.002 dex / 0.0001 |

The compensation is correct here too; what it exposes is smoothing in the data that no operator
declares. The analytic delivery with compensation still leaves the primaries 0.53 to 1.54 km/s high,
which is 2.4 to 3.0 km$`^2`$ s$`^{-2}`$ of data smoothing, the size of the benchmark simulator's own
2 km/s chain less the 0.25 km$`^2`$ s$`^{-2}`$ by which the product pixel exceeds the detector
pixel. Declared completely, the widths agree:
$`\sigma_q^2 + 12.21 + 2.33 - 0.25 - 5.25 = (9.54\ {\rm km\,s^{-1}})^2`$ against the measured
9.557 km/s, whose 3.05 km/s already netted the operator's grid against the simulator's.

## What should change

**`Fit.epoch_statistics` should apply the compensation by default, with the analytic
$`\sigma_{\rm grid}^2 = 7\Delta v^2/12`$ subtracted after the library's width.** The analytic form is
as good as the operator-computed one on both samples (0.12 and 0.02 km/s), it needs no phases, and
it is a property of the model rather than of the data, so it belongs where the library width is
already removed. The statistics should record it in a field of their own, since
`_check_statistics` compares the library resolving power exactly and a compensated operator can
otherwise only be passed by declaring a false resolving power, as was done here. When
$`\Delta v > \sigma_q`$ the compensated kernel falls below 0.65 pixel, and beyond $`1.10\,\sigma_q`$
below half a pixel, where the floor removed 4.4 of 12.5 km$`^2`$ s$`^{-2}`$; there the call should warn
and name the grid spacing that would be needed rather than floor silently. On the Gaia RVS products
the compensation should ship together with a declared width for the archive's resampling, since on
its own it raised the median rotation error on those slow rotators from +3.06 to +4.22 km/s.

**The pipeline should keep the native-pixel default grid and warn when $`\Delta v > \sigma_q`$; a
finer grid is not a substitute for the compensation.** Uncompensated, a 1.5 km/s grid still returned
3.18 km/s for 5 and 10.28 km/s for 11 on the test's fixture, and on the 0.5 km/s simulation a 2 km/s
grid returned 1.21 km/s for 5; compensated, the default grid returned 3.55, 10.08 and 24.56 against
the 2 km/s grid's 4.05, 10.56 and 24.72, and on the test the 2 km/s grid took 85 s against 57 s
for the default grid, each including its compilation.

**The failing test should keep its 25% tolerance and pass through the compensation.** With it the
test returns 9.82 km/s and every assertion passes; moving it to a 2 km/s grid passes at 9.78 km/s
but is slower and hides a bias of the default configuration that users get, and widening the
tolerance would do the same. Its fixture's $`\gamma = 12`$ km/s is a whole number of pixels on the
3, 2 and 1.5 km/s grids, so a variant of the test on a finer grid would not exercise the frame-shift
term.

**Defects and statements found on the way.** `SpectralLibrary.resampled_to` states that its box
average is negligible and "identical on both sides of the comparison, since the data are convolved by
the same instrument profile"; it is $`\Delta v^2/12`$ (1.78 km$`^2`$ s$`^{-2}`$ on the default grid,
about 8 km$`^2`$ s$`^{-2}`$ in $`(v\sin i)^2`$), and the data of the epoch comparison are never
box-averaged on the model grid. `docs/math.md` §9.5 puts the grid smoothing at "order
$`\Delta v^2/2`$", which leaves out the rotation kernel's pixel integration (the total is
$`7\Delta v^2/12`$), and advises a model grid finer than the native pixel, which the table above shows
is not enough for slow rotators. The native comparison is grid-sensitive in temperature (Teff$`_B`$
about 195 K cold on the 4.63 and 3 km/s grids, within 39 K on 2 and 1.5 km/s), which was not traced.

**What is left.** First, the frame-shift term is averaged, and on the default grid of the test it is
0.241 rather than 1/6 of $`\Delta v^2`$, which leaves -0.39 km/s at 11 km/s; applying $`v`$ in the rows
without interpolation, for example through the library resampling, or re-evaluating the term at the
fitted $`v`$ once, would remove it, and neither was built. Second, the rotation kernel's pixel term
oscillates for $`v\sin i \lesssim \Delta v`$ (2.81 rather than 1.78 km$`^2`$ s$`^{-2}`$ at 5 km/s on the
default grid), which the formula does not follow. Third, the default grid's rebin term is not
stationary along the band (0.815 at 5185 A against a mean of 1.783), so a single width compensates the
mean only. Fourth, the compensation was tested on a scalar LSF; its form for per-anchor or per-epoch
widths is the same subtraction per width, but it was not run, nor with AR(1) noise. Fifth, the RVS
check covers three slow-rotator products; the rerun that matters for the benchmark is all 22 products
with the delivery width declared and the grid compensated. Sixth, the seeds were run at 11 km/s only;
the 5 and 25 km/s rows rest on one draw.
