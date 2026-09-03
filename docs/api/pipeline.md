# Pipeline and command line

One declaration in, structured products out. For every star in a list the pipeline reads
the epochs, disentangles them, fits atmospheric labels to the components against a
synthetic grid, measures one velocity per component per epoch against those components,
fits a Keplerian to the table, and writes the products (tables, spectra with their
uncertainty bands, a JSON report, diagnostic figures) into one directory per star, with a
batch-level table and a record of every failure.

```bash
albireo init            # writes an annotated albireo.toml
albireo run albireo.toml --jobs 4
albireo demo            # two simulated stars with known answers, offline
```

The same from Python:

```python
import albireo as ab

run = ab.run_pipeline("albireo.toml", jobs=4)
run.results["AI Phe"].report["orbit"]["k"]     # {"primary": ..., "secondary": ...}
```

## Scope

The pipeline is a driver. Every scientific decision is made by the stage that owns it
([`Disentangler`](facade.md), [`match_labels`](match.md), [`todcor`](todcor.md),
[`fit_rv_orbit`](rvorbit.md)), and the two rules those stages enforce apply unchanged: light
fractions are declared, never defaulted, and the wavelength medium is declared before a
synthetic grid is consulted. Where the pipeline cannot honour a request, it records a flag
on the star's report and continues rather than guessing or stopping the batch.

Three routes to the orbit:

| Declaration | What runs |
|---|---|
| `period = [lo, hi]` (or a value, or `{value, sigma}`) | the Keplerian is inferred from the spectra; the epochs are measured back against the components |
| `period = "search"` | library templates at the declared starting labels measure a first table, a periodogram finds the period, an orbit fitted to the table warm-starts the disentangling |
| `velocities = "file"` | the free per-epoch table is fitted from measured velocities; the period comes from the table afterwards |

**Component order.** Components must be declared in order of decreasing mass, which for a
main-sequence pair is the brighter star first. The likelihood cannot tell which spectrum
belongs to which star: with a symmetric semi-amplitude prior the conjunction scan sees two
equally deep troughs, the declared assignment and its mirror with the spectra swapped and
rescaled by the light ratio. The pipeline starts the fit with $`K_1 < K_2 < \dots`$ (the
first star moves least), which is a convention rather than a constraint, and the label
stage checks it: a fitted light fraction far from the declared one is flagged as the
signature of a reversed order.

**Zero points.** Velocities measured against a disentangled component are differential,
because its rest frame is not identified
([§5.3](../math.md#53-systemic-velocity-zero-point)), unless the label fit measured the
frame's offset, in which case the pipeline applies it to the templates and the velocities
are absolute. `result.json` records which case applies (`velocities.absolute`), and so does
the orbit fit, which uses one systemic velocity per component whenever a component is
differential.

## Batches

Stars are independent, so `jobs > 1` runs them in a spawn-based process pool. On the
development desktop, four workers finish a batch of eight simulated stars 2.0× faster than
one process and eight workers 2.5× faster. The scaling is sub-linear because a single star
already occupies several cores, so the workers overlap only each star's serial part. Each
worker's XLA and BLAS thread counts are capped at `cpu_count // jobs` as a precaution
against oversubscription; on that benchmark the cap made no measurable difference. A
worker returns a plain-data report; the live objects (the `Fit`, the velocity table, the
label match) are retained only on an in-process run (`jobs=1`), since they carry compiled
JAX programs that cannot cross a pipe. A star that fails is recorded in `failures.txt` and
does not stop the others. The measurements are in
[the benchmark record](../benchmarks.md#the-pipeline-in-worker-processes-2026-09-01).

## Products

Per star, `<output>/<star>/`:

| File | What |
|---|---|
| `summary.txt` | every stage's own report, the assumptions block, the flags |
| `result.json` | the machine-readable report: dataset, declaration, orbit from the spectra, labels with both error bars, the velocity table, the orbit from the table with errors, timings, flags, files |
| `velocities.rv`, `velocities.csv` | the epoch velocity table, twice (commented ASCII with the zero-point status in its header; a CSV) |
| `spectrum_<component>.txt`, `spectra.fits` | the disentangled components with their uncertainty band |
| `orbit.txt`, `labels.txt`, `template_<component>.txt` | the Keplerian from the table; the label report; the label fit's model spectra |
| `fit.npz`, `posterior.npz` | the MAP result for `load_fit`; the NUTS draws when sampling was asked for |
| `spectra.png`, `residuals.png`, `velocities.png`, `phase_scan.png`, `todcor_surface.png`, `rv_curve.png` | the figures |
| `log.txt` | the progress lines with timings, and every warning |

Per batch: `results.json` (every report), `results.csv` (one row per star: period,
eccentricity, semi-amplitudes and systemic velocities with errors, labels, flags),
`summary.txt`, `failures.txt`, and `run.json` (the declaration as run).

Background and references: [science overview](../science.md).

::: albireo.pipeline

::: albireo.cli
