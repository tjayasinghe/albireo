# Run everything in one command

See the [science overview](../science.md) for background and references.

The pipeline applies every stage to each star in a file: read the epochs, disentangle, fit
labels to the components, measure one velocity per component per epoch, fit the orbit to the
table, and write the products out. Failures are recorded, and the results go into a table a
spreadsheet can read.

```bash
pip install -e ".[io,plots]"
albireo demo
```

That runs the whole pipeline on two simulated stars whose answers are known and writes
`albireo_demo/`. Read `albireo_demo/summary.txt` first, then a star's `summary.txt` and its
figures. Nothing downloads, and the run takes a minute or two.

## Declaring your own stars

```bash
albireo init
```

writes an annotated `albireo.toml`. The entries that must be filled in are the ones the data
cannot supply:

```toml
[instrument.HARPS]
resolving_power = 115000          # or sigma_kms = 1.1

[[stars]]
name = "AI Phe"
spectra = "data/aiphe/*.fits"     # a glob, a directory, or a list of files
period = [24.5, 24.7]             # [lo, hi] uniform; a value to hold; or "search"

[[stars.components]]              # in order of DECREASING MASS: the brighter star first
name = "primary"
light = 0.55                      # required: an assumption the data cannot check
teff = [5500.0, 7000.0]           # label priors for the template stage (optional)
logg = 4.0

[[stars.components]]
name = "secondary"
light = 0.45
teff = [4300.0, 5900.0]
logg = 3.6
```

then

```bash
albireo run albireo.toml --jobs 4
```

The pipeline refuses to guess three declarations.

**The light fractions are required.** With constant light fractions the likelihood sees only
$`\ell_i d_i`$ ([§5.2](../math.md#52-light-ratio-line-depth)), so every recovered line depth
scales as $`1/\ell_i`$ and no part of the fit can report that the assumed value was wrong. Quote
them beside every result, as every report does under *Assumed, not measured*.

**The components are declared in order of decreasing mass.** The likelihood cannot tell which
spectrum belongs to which star: with a symmetric semi-amplitude prior the conjunction scan sees
the declared assignment and its mirror as equally good. The pipeline therefore starts the fit
with $`K_1 < K_2`$ and lets the label stage check it afterwards, flagging a fitted light fraction
far from the declared one as the signature of an order declared the wrong way round. With a
library configured the starting values themselves come from a template table fitted at the
declared period, so the convention only breaks the ties the data leave; a component the table
cannot measure starts from its nearest measured neighbour, above it for a lighter star and
below it for a heavier one.

**The wavelength scale is declared before a synthetic grid is consulted.** ESO files declare
it and the reader takes it from them; a file that does not gets `medium = "air"` (or
`"vacuum"`) on its star after the convention has been checked. Until then the label stage is
skipped with a flag rather than run on an 83 km/s guess.

## The label stage

```toml
[labels]
library = "bosz2024-fgk-r20000"   # albireo.library_names(); ~621 MB once, then cached
mh = [-1.0, 0.5]
# compare = "epochs"              # the default; "native" or "matched" compare with the components
```

With a library declared, each disentangled component is fitted for Teff, log *g*, [M/H] and
*v* sin *i* against the grid ([the previous tutorial](labels.md)) and, for the velocities, for
the offset of its rest frame. By default the templates are compared with the epoch spectra
themselves, through the disentangling's sufficient statistics, rather than with the
disentangled components, and the light fraction the fit measures does not depend on the one
declared; `result.json` records it beside the declaration under `labels`.

A disentangled component's zero point is not identified
([§5.3](../math.md#53-systemic-velocity-zero-point)); the label fit measures it, and the
pipeline applies it to the templates so that the epoch velocities come out absolute. If the
fit disowned that offset (it beat neither of its nulls, it was pinned on the bound of its own
scan, or the two components disagree by more than one systemic velocity can hold), the offset
is refused with a flag and the table stays differential. Without a
library the velocities are differential: semi-amplitudes, eccentricity and mass ratio exact,
systemic velocity meaningless, and the orbit fit gives each component its own $`\gamma`$. Every
report states which it got, in the first lines of the velocity table's summary and in
`result.json["velocities"]["absolute"]`.

The demo shows the difference. Its second star has a systemic velocity of +12 km/s that the
disentangling alone cannot see; with the toy library declared, the label fit measures each
component's frame at about +12, the velocities come out absolute, and the orbit fitted to them
returns $`\gamma = 11.96 \pm 0.15`$ km/s.

## Two routes when the period is unknown

```toml
period = "search"
```

renders library templates at the starting labels, measures a first velocity table against them,
proposes candidate periods from three periodograms, fits an orbit to the table from each,
keeps the one with the lowest chi-square, and warm-starts the disentangling from it. Template
mismatch costs a constant per component here, to which the period and the semi-amplitudes are
insensitive. This route needs the library.

```toml
velocities = "aiphe_rv.txt"        # columns: [bjd] v_primary v_secondary
```

declares velocities measured elsewhere (cross-correlation lags, line splitting) and fits the
free per-epoch table instead of a Keplerian
([the façade's `velocities=`](../api/facade.md)); the period comes from the table afterwards.

## Reading a report

`summary.txt` collects every stage's own report: the dataset, the derivations
`Disentangler.explain()` prints, the fit, the labels against their nulls, the velocity table
with its zero-point status, and the orbit with its errors. It ends with the flags, every caveat
the run recorded; read those first. Recurring flags:

- *residual z-score rms 1.4: the noise model does not describe these data*: the inverse
  variances are off; read [the benchmarks](../benchmarks.md) before reaching for a jitter.
- *velocities are differential*: no library, or no medium; see above.
- *labels learned nothing about teff_B*: the component carried no information on that label,
  and the prior came back as the result.
- *K_secondary from the velocity table disagrees with the disentangling*: the templates or the
  light fractions need inspection.
- *the label fit measures light fractions of primary 0.380 +- 0.004, secondary 0.620 +- 0.004
  against the declared 0.620, 0.380*: the components are probably declared in the wrong order,
  or the declared fractions are wrong. The flag fires above a factor of 1.5 or a difference of
  0.15.
- *labels with no formal error: vsini_B (rotation plateau)*: the epoch comparison left that
  label out of its covariance, because it sits on a bound or below half a model pixel of
  rotation, where the chi-square is flat; the value is a limit.

`result.json` carries the same content in a shape a script can read, and the batch's
`results.csv` has one row per star with the period, eccentricity, semi-amplitudes and systemic
velocities with their errors, the labels, and the flags.

## Batches and workers

```bash
albireo run survey.toml --jobs auto       # cpu_count // 4 workers
```

Stars are independent, so they run in worker processes. On the development desktop four
workers finish a batch of eight simulated stars 2.0× faster than one process, and eight workers
2.5× faster; the scaling is sub-linear because one star already keeps several cores busy. Each
worker's XLA and BLAS threads are capped at `cpu_count // jobs` as a precaution against
oversubscription; on that benchmark the cap made no measurable difference. One failing star (a
missing file, or a declaration the façade refuses) lands in `failures.txt` with its message and
does not stop the others. The numbers are in
[the benchmark record](../benchmarks.md#the-pipeline-in-worker-processes-2026-09-01).

From Python the same run is

```python
import albireo as ab

run = ab.run_pipeline("survey.toml", jobs="auto")
print(run.summary())
row = run.results["AI Phe"].report["orbit"]      # period, ecc, k, gamma, errors, ...
```

and with `jobs=1` each `StarResult.live` also holds the objects themselves, the `Fit`, the
velocity table, the label match and the orbit, for anything the written products do not cover.

## BLOeM by name

```bash
albireo fetch 1-037 1-002 --out data
```

downloads each star's public GIRAFFE epochs and prints the `[[stars]]` entries that would
analyse them. The SMC's +150 km/s systemic velocity is why the label stage's default frame
offset range is ±300 km/s, and why an OB grid is needed for those stars
([`library_info("pollux-ob-smc24")`](../api/library.md)).

## What this page does not do

It does not make the assumptions on the user's behalf; the flags record them, and a report is
worth reading in preference to a table. A pipeline that completed every star of a survey has
turned every light fraction it was given into a line depth, and each report repeats that
because no downstream number escapes it.
