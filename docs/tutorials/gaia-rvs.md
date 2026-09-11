# Simulate Gaia RVS binaries, and measure how well they come back

See the [science overview](../science.md) for background and references.

Gaia DR4 (2 December 2026) publishes an epoch RVS spectrum for every transit, in the
barycentric frame, for the double-lined transits the Gaia pipeline itself rejects. This
page builds such spectra for a synthetic binary, checks that albireo recovers what was
put in, and then runs a population of systems through the pipeline to measure recovery
as a function of brightness, transit count, separation and light ratio. The instrument
model follows the reference implementation of Rowan's `synthetic_rvs_spectra` notebook;
the component spectra come from the BOSZ 2024 grid in the RVS band rather than from a
synthesis code, which is the one deliberate difference, stated in the
[API page](../api/gaia.md).

```bash
pip install -e ".[io,plots]"
python -c "import albireo; albireo.fetch_library('bosz2024-fgk-rvs')"   # 621 MB, 5 MB cached
python examples/14_gaia_rvs.py     # the notebook's own system, end to end
```

## One binary

The notebook's system: a 6100 K and a 5700 K dwarf at a flux ratio of 0.60, 1.30 and
1.20 solar masses in a 4-day orbit of eccentricity 0.1 seen at 87 degrees, systemic
velocity −18 km/s, ten epochs evenly spaced over one period at a S/N of 40 per detector
pixel.

```python
import numpy as np
import albireo as ab
from albireo.gaia import (
    RVS_DR3_MEAN, rvs_components, rvs_model_grid, simulate_rvs_dataset, uniform_phase_times,
)
from albireo.population import semi_amplitudes

k1, k2 = semi_amplitudes(1.30, 1.20, 4.0, 0.10, np.radians(87.0))   # 87.7, 95.0 km/s
orbit = ab.OrbitParams(period=4.0, t_peri=1.2 * 4.0 / (2 * np.pi), ecc=0.10, omega=0.7,
                       k=(k1, k2), gamma=-18.0)

library = ab.fetch_library("bosz2024-fgk-rvs")
grid = rvs_model_grid(250.0, vsini_max_kms=12.0)
components = rvs_components(
    library,
    [{"teff": 6100.0, "logg": 4.2, "mh": -0.1}, {"teff": 5700.0, "logg": 4.3, "mh": -0.1}],
    grid,
    vsini_kms=[12.0, 11.0],
)
dataset, truth = simulate_rvs_dataset(
    components, grid,
    bjd=uniform_phase_times(4.0, 10, start=2457000.0),
    light_fractions=(1 / 1.6, 0.6 / 1.6),
    orbit=orbit,
    snr=40.0,
    product=RVS_DR3_MEAN,
    library_resolving_power=float(library.meta["resolution"]),
)
print(dataset.summary())
print(truth.delivery.pixel_ratio, np.mean(truth.delivery.lag1))   # 2.45, 0.8
```

The last line is the point of the delivery step: the archive's 0.01 nm grid has 2.45
samples per detector pixel, and the noise on it is correlated with a lag-one coefficient
of 0.8, which the archive's per-pixel `flux_error` does not express. On the DR4 epoch
grid (`RVS_DR4_EPOCH`, the default) the sampling is 0.025 nm and the correlation is
weaker and periodic.

The dataset is declared barycentric and vacuum with the RVS width on every epoch, so the
disentangler needs nothing but the light fractions and the orbit prior:

```python
dis = ab.Disentangler(
    dataset,
    components=[ab.Star("primary", 1 / 1.6), ab.Star("secondary", 0.6 / 1.6)],
    orbit=ab.Orbit(period=ab.Known(4.0, 0.004),
                   k=ab.Between([20.0, 20.0], [160.0, 160.0], start_at=[70.0, 110.0]),
                   ecc=ab.Between(0.0, 0.5)),
    lsf={"RVS": ab.LSF.from_resolution(11_500)},
    dv_kms=3.0,
    noise_correlation={"RVS": float(np.mean(truth.delivery.lag1))},   # the delivered grid's 0.8
)
fit = dis.fit()
print(fit.star("primary")["k"], fit.star("secondary")["k"])   # within a percent of 87.7 and 95.0
```

The last argument declares what the delivery did to the noise: the disentangling then runs
the AR(1) noise model along the pixel index, and the velocity table measured afterwards
carries the same correlation in its errors. Left out, the pixels are taken as independent,
which is what the archive's errors say and what the delivered spectra are not.

`examples/14_gaia_rvs.py` runs this and adds the epoch-velocity measurement by TODCOR
against library templates, which the notebook also prints.

The epochs above are evenly spaced over one period, which no real star is observed at.
`rvs_transit_times` draws the scanning law's structure without its phase, and
`gost_transits(ra, dec)` fetches the transits the Gaia Observation Forecast Tool actually
predicts for a position, which `rvs_transit_times_from_gost` cuts to a release, to the
four of seven CCD rows the RVS covers and to the 0.78 that reach the ground; the
[API page](../api/gaia.md#real-transit-times-from-gost) sets out what each step rests on
and what it gets wrong.

## A population

`draw_population` draws systems from the field's distributions (Moe & Di Stefano 2017
for the orbits and mass ratios, a dwarf sequence for the stars, the eclipse geometry for
the inclinations, the scanning law for the transit counts) as a magnitude-limited survey
sees them, and `from_debcat` or `from_gaia_sb2` build the same records from real
eclipsing and double-lined binaries. The velocity floor below stands for the double-lined
selection: an orbit whose largest separation is under 40 km/s, one and a half RVS
resolution elements, is not in Gaia's double-lined sample either.

A catalogue row can also name a star the library cannot render. Gaia masses of 1.9 solar
put a primary at 8600 K, above the FGK box's 7000 K ceiling, and the benchmark leaves such
a system out and names it rather than clamping its temperature. `--library
bosz2024-hot-rvs` selects the box that runs 7000 to 10,000 K instead, which admits those
systems and leaves the cool ones out, since one run uses one library. What the hot box
states about a velocity measured above 8000 K, where the Paschen series has taken the band
over from Ca II, and about the rotation of A stars against a resolving power of 11,500, is
in [`albireo.library`](../api/library.md#above-7000-k-the-hot-box), and it should be read
before its numbers are quoted.

```python
from albireo.population import draw_population, population_summary

systems = draw_population(
    40, kind="mixed", seed=1, library=library, min_separation_kms=40.0
)
print(population_summary(systems))
```

## The benchmark

Every system is simulated once and the same epochs are run through the pipeline under
each knowledge tier: `oracle` (the ceiling), `eclipsing` (an ephemeris and a light-curve
light ratio), `orbit` (a Gaia period and nothing else) and `blind` (a period search).

```python
from albireo.benchmark import BenchmarkConfig, run_benchmark

config = BenchmarkConfig(output="bench/rvs", systems=systems, jobs=8)
run = run_benchmark(config)
print(run.report)          # bench/rvs/report.md
```

or, from the command line, with the same options and a resumable output directory:

```bash
python scripts/gaia_rvs_benchmark.py --n 40 --jobs 8 --out bench/rvs
python scripts/gaia_rvs_benchmark.py --n 40 --cadence gost --out bench/rvs-gost
python scripts/gaia_rvs_benchmark.py --report-only --out bench/rvs
```

`--cadence gost` replaces the drawn transit count with the forecast for each system's own
position, so the epochs fall at the times the nominal scanning law puts them rather than
at drawn ones; it needs the network once per position and caches the answers.

The report tabulates, per tier, the median and 16th to 84th percentile of the
semi-amplitude errors from the disentangling and from the orbit fitted to the velocity
table (the two fail differently: a secondary of a few percent of the light is recovered
by the disentangling and lost by the correlation), the pulls of the latter, the element
errors, the epoch-velocity residuals
and their pulls, the recovered spectra's correlation and equivalent-width ratios against
the injected ones, the label offsets and the measured light fractions, plus the period
search's recovery rate and the failures and flags, with figures of each against G_RVS,
the transit count and the separation. The epoch velocities themselves, one per component
per epoch from TODCOR against the disentangled components, are gathered with the injected
velocity of each epoch into `velocities.csv`, pooled per tier in an "Epoch velocities"
section of the report, and drawn phase-folded against the injected orbit for every system
of every tier. The numbers of the runs on the 32-thread desktop are recorded in the
[benchmarks](../benchmarks.md).

## Reading the results honestly

Three things the report cannot tell you, because the simulation cannot. The templates
and the truth come from the same grid, so template mismatch is absent: a real star's
lines are not BOSZ's, and the label offsets here are a floor. The line-spread function
is a Gaussian at the nominal resolving power unless `resolving_power="per-transit"` is
requested, and even then it is a Gaussian, which the DPAC model is not. And the epochs
follow the scanning law's statistics, not its phase, so a real star's period aliases are
its own. What the report does tell you is how the recovery scales with the observables,
and where the quoted errors are calibrated.
