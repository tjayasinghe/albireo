# Measure epoch velocities with TODCOR

See the [science overview](../science.md) for background and references.

The rest of this site infers the orbit from the composite spectra directly and never writes
down a velocity for a single night. That is the right approach when the component spectra are
unknown. It is not what an eclipsing-binary analysis, a survey pipeline or an existing orbit
code consumes: those need a table with one velocity per component per epoch. This page produces
that table.

The method is TODCOR, the two-dimensional correlation of Zucker & Mazeh (1994). Instead of
correlating a spectrum against one template and reading two peaks off the result, it correlates
the spectrum against a combination of two templates, each with its own shift, and reads both
velocities off the location of the single maximum. Because the second star is in the model, the
two peaks stop pulling each other as they approach, and a companion much fainter than the
primary can be measured from one spectrum. albireo's [`todcor`](../api/todcor.md) is that
estimator written as the weighted least-squares fit it is, so that masks, gaps, cosmics,
per-pixel weights and mixed instruments enter through the weights, generalized to any number of
components, with the maximum-likelihood errors of Zucker (2003). On a uniform grid with uniform
weights it reproduces the published formulae to 1e-10, which the test suite checks.

The runnable version of this page is
[`examples/12_todcor.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/12_todcor.py);
the numbers below are its output.

## Where the templates come from

A correlation needs one template per component, and the choice determines what the velocities
mean. There are three routes:

| Route | Zero point | When |
|---|---|---|
| `fit.templates()`, the disentangled components themselves | differential: each component's rest frame is not identified ([§5.3](../math.md#53-systemic-velocity-zero-point)) | the system has been disentangled and the per-epoch table it implies is wanted |
| `Template.from_labels(match, name)`, the label fit's model spectrum | absolute, because the label fit measured the disentangled frame's offset | after [turning a component into an RV template](labels.md) |
| `Template.from_library(...)`, a published grid at assumed labels | absolute | a survey of similar stars, or a star that has not been disentangled |

Every table records which it got (`table.absolute`), and `summary()` states it in the first
lines. A differential velocity is not a worse velocity, since the semi-amplitudes, the
eccentricity and the mass ratio are all untouched by a constant, but a systemic velocity read
off one is meaningless, and the orbit fit below gives each such component its own.

## The minimum call

```python
import albireo as ab
import numpy as np

dataset, truth = ab.load_example("sb2_sim", with_truth=True)
coarse = ab.LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))
# Three pixels per LSF sigma for a correlation template (see "The grid" below).
grid = ab.LogGrid(x0=coarse.x0, dx=coarse.dx / 3, n=(coarse.n - 1) * 3 + 1)

templates = [
    ab.Template(name, grid, np.interp(grid.x, coarse.x, d), v_zero_kms=0.0)
    for name, d in zip(("primary", "secondary"), truth["components"])
]
table = ab.todcor(
    dataset, templates,
    v_range=(-120.0, 120.0),          # barycentric km/s, per component or shared
    light=truth["light_fractions"],   # held; or "global" (default), or "free"
    lsf_sigma_v={"DEMO": 6.5},        # per instrument, as build_problem takes it
)
print(table.summary())
```

```
TODCOR velocities: 2 components x 12 epochs, 12 usable (data topocentric; velocities barycentric)
  primary: -37.015 to +46.516 km/s, median sigma 0.1232 km/s, light 0.620 (fixed); absolute
  secondary: -69.648 to +55.626 km/s, median sigma 0.1928 km/s, light 0.380 (fixed); absolute
  reduced chi-square: median 1.051 (range 1.009-1.170); R^2 median 0.975
  Wilson slope secondary vs primary: -1.5009 (= -K_secondary/K_primary)
```

Against the injected velocities the primary comes back with an rms error of 0.140 km/s on a
quoted 0.124, the secondary 0.135 on 0.194: a pull rms of 1.13 and 0.70, which is what
calibrated errors look like on twelve epochs.

Three arguments in that call need a note each.

**The grid.** Every template must live on one `LogGrid`, it must extend beyond the data by the
velocity range being searched (`LogGrid.covering(dataset, dv_kms=..., v_margin_kms=...)` builds
one), and it should sample the narrowest LSF with three or more pixels per sigma. The shift
operator interpolates linearly, which carries a pixel-locking ripple of order
$`0.1/\sigma_{\rm px}^2`$ pixels
([§10.3](../math.md#103-fractional-shifts-are-exact-and-the-pixel-locking-bound)): a few
thousandths of a pixel at three per sigma, a few hundredths at one. `todcor` warns below two;
`fit.templates()` upsamples automatically; a library template's grid is built by the caller and
should be built fine. The packaged example's truth grid samples the DEMO LSF at one pixel per
sigma, which is why the snippet upsamples it by three.

**The light fractions.** `light="global"` (the default) fits them freely in every epoch, takes
the weighted median over the well-detected, unblended epochs of each instrument, and holds
that. This is standard practice, because a per-epoch ratio is noisy and a ratio fitted at a
blended phase is not a measurement. Pass a sequence to hold declared values instead. If the
templates are the disentangled components, hold the fractions the disentangling assumed: those
spectra were solved against them, and no other amplitude is consistent with what they are
([§9.1](../math.md#91-what-a-disentangled-component-actually-is)).
`fit.measure_velocities()` does that.

**The LSF.** The templates are intrinsic, and each instrument's LSF is applied to them in
quadrature above whatever resolution the template already carries (`Template.sigma_kms`). A
template rendered from an $`R = 20{,}000`$ grid is therefore not broadened twice, and a template
broader than the instrument is used as it is, with a warning.

## Reading the diagnostics

`VelocityTable` carries more than velocities, and the extra columns are what make a batch
checkable:

- `sigma` is the curvature of the chi-square surface at its minimum, rescaled by the reduced
  chi-square so that the noise level is measured from the residuals rather than taken from
  `ivar`; this is Zucker's (2003) estimator. `sigma_ivar` uses the weights as given.
- `blended` marks epochs where the two velocities were measured along a ridge (a covariance
  correlation above 0.9). It fires for twin spectra at the same velocity; it does not fire for
  two different spectra at the same velocity, because two different line lists remain
  separable, which is the basis of the method.
- `delta_chi2` is how much worse the fit becomes when each component is removed and the rest
  refitted. A small value means the epoch does not detect that star, which is what a
  faint-companion search must notice epoch by epoch.
- `at_edge` marks a minimum at the boundary of `v_range`: widen it.
- `light` is the amplitude assigned to each template. With `scale="free"` its column sum is the
  composite's fitted scale, and a value far from one indicates the normalization is off.

## Why two dimensions

Running the same spectra against the primary's template alone, the one-dimensional CCF, and
comparing the primary's velocity error against the two-dimensional result epoch by epoch, in
order of the separation between the two stars' lines:

```
|v1 - v2| [km/s]   1-D error   2-D error   (km/s)
     4.3            -0.134     +0.078
    10.5            -0.622     -0.131
    39.8            -0.925     +0.188
    66.4            -0.095     -0.198
    ...
    92.4            +2.439     +0.103
   116.1            +1.629     +0.077
rms over the four most blended epochs: 1-D 0.563, 2-D 0.157 km/s
```

The one-dimensional error is not confined to the blended epochs. A secondary contributing 38%
of the light contaminates the primary's peak at every separation, in a direction that depends
on which of its lines lie near the primary's, and reaches 2.4 km/s, twenty times the quoted
error, at 92 km/s. The two-dimensional fit removes the bias because the contaminant is in the
model.

## The orbit from the table

```python
search = ab.find_period(table, period_range=(2.0, 20.0))   # Lomb-Scargle on v1 - v2
orbit = ab.fit_rv_orbit(table, period=search["period"])
print(orbit.summary())
```

```
Keplerian fit to 24 velocities of 2 component(s): chi2 15.65 for 17 dof (errors rescaled by sqrt(0.921))
  P      = 6.000532 +- 0.001062 d
  T_conj = 0.62427 +- 0.00213
  e      = 0.1515 +- 0.0009
  omega  = 39.89 +- 0.32 deg
  K_primary = 41.9900 +- 0.0490 km/s   gamma_primary = +0.0400 +- 0.0290 km/s   rms 0.1259 km/s
  K_secondary = 63.0223 +- 0.0765 km/s   gamma_secondary = +0.0400 +- 0.0290 km/s   rms 0.1014 km/s
  systemic velocity: shared
  q = K_primary/K_secondary = 0.6663;  M_primary sin^3 i = 0.4173 Msun, M_secondary sin^3 i = 0.2780 Msun
```

against an injected $`P = 6`$, $`e = 0.15`$, $`\omega = 40.1^\circ`$, $`K = 42, 63`$. The fit uses the
same Kepler solver and angle conventions as the joint model, so `orbit.to_theta()` is what
`Disentangler(orbit=...)` takes as a warm start: measure against a library template, fit the
orbit, disentangle from it. The period search returns its aliases as well (6.083, 5.843,
6.205 d here); a sparsely sampled table's periodogram is rarely unambiguous, and the aliases
should be inspected.

## Closing the loop

```python
fit = ab.Disentangler(dataset, components=[...], orbit=..., lsf={"DEMO": 6.5}).fit()
own = fit.measure_velocities()       # against the fit's own components
own_orbit = ab.fit_rv_orbit(own, period=orbit.period)
```

The disentangled components are the best templates available for a system, with the right
lines, depths and rotation, measured from these very epochs. The velocities measured against
them recover the injected ones to 0.13 and 0.10 km/s rms once each component's zero point is
removed. They are differential: `own.absolute` is `(False, False)`, `summary()` reports
"template zero point unknown", and `fit_rv_orbit` fits one $`\gamma`$ per component rather than
forcing a shared one onto two different constants, which would corrupt both $`K`$ values (a test
covers that case). The semi-amplitudes come back at 41.983 ± 0.046 and 62.991 ± 0.072 km/s.

Absolute velocities from this loop are what the label mode provides: fit labels to the
components ([the previous tutorial](labels.md)), and `Template.from_labels(match, name)`
carries the fitted frame offset into the template's zero point.

## A batch

```python
batch = ab.todcor_batch(
    {"HD 1": ds1, "HD 2": ds2, ...},     # {star: Dataset}
    templates,                            # shared, or {star: [templates]}
    v_range=(-300.0, 300.0), light="global", lsf_sigma_v={"HARPS": 1.1},
)
print(batch.summary())
batch.write("velocities/")               # one ASCII table per star, plus failures.txt
```

A failing star is recorded in `batch.failures` with its message and does not stop the run
(`on_error="raise"` reverses that). Each table's `to_dict()` is one step from a
`pandas.DataFrame`, and `write()` produces a commented ASCII file whose header states the
frame, the zero-point status of every component, and how the light fractions were set. On the
example the twelve epochs of one star take well under a second once the kernels are compiled.

## What this does not do

It does not replace the joint fit. A per-epoch table discards the phase coherence that lets
disentangling separate stars whose lines never resolve, and its accuracy is bounded by how well
the templates match the stars; template mismatch mostly costs a constant offset per component,
which lies outside the quoted error. Where the components are unknown, disentangle first. Where
they are known, this is faster, simpler, and works on a single spectrum, which is the split
Zucker drew between the two methods. It also synthesizes nothing: the templates come from the
disentangling, from a label match, or from the published grids of `albireo.library`.
