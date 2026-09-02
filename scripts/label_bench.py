"""Measure what the label-matching mode costs and what it recovers (D52, D53).

Produces the numbers ``docs/benchmarks.md`` records: interpolation error against the
published reference values, closed-loop recovery at a range of signal-to-noise, the
wall-clock split across the stages of a fit, and the ratio between the formal error and the
spread over posterior draws.

Nothing here needs the network. The grid is a toy built in this file, at a node density
matched to the BOSZ subset the mode is designed for (250 K in Teff, 0.5 dex in log g,
0.25 dex in [M/H]), so the interpolation numbers are comparable with the literature's.

    python scripts/label_bench.py [--quick]
"""

from __future__ import annotations

import argparse
import platform
import time

import numpy as np

import albireo as ab

# BOSZ's own spacing, so the interpolation error is comparable with Meszaros & Allende
# Prieto (2013): 0.051% linear and 0.031% cubic, against ~0.1% for a Payne-style network.
TEFF_AXIS = np.arange(4000.0, 7001.0, 250.0)
LOGG_AXIS = np.arange(3.0, 5.01, 0.5)
MH_AXIS = np.arange(-1.0, 0.51, 0.25)

WAVE = np.linspace(5150.0, 5250.0, 2000)
TRUTH = {
    "A": {"teff": 5180.0, "logg": 4.05, "mh": -0.15, "vsini": 11.0},
    "B": {"teff": 4460.0, "logg": 4.55, "mh": -0.15, "vsini": 27.0},
}
TRUE_LIGHT = np.array([0.62, 0.38])
ASSUMED_LIGHT = np.array([0.72, 0.28])
LSF_SIGMA_KMS = 5.5


def toy_spectrum(teff, logg, mh, wave):
    """Each label drives its own lines, so the label-to-spectrum map is invertible."""
    t, g = (teff - 4800.0) / 600.0, logg - 4.0
    lines = (
        (5167.3, 0.30 + 0.13 * np.tanh(t)),
        (5172.7, 0.22 - 0.11 * np.tanh(0.8 * t)),
        (5183.6, 0.26 + 0.09 * g + 0.02 * g**2),
        (5195.4, 0.17 - 0.07 * g),
        (5205.9, 0.21 + 0.16 * mh + 0.04 * mh**2),
        (5227.2, 0.19 + 0.10 * mh - 0.03 * np.tanh(t)),
    )
    flux = np.ones_like(wave)
    for center, depth in lines:
        flux = flux - depth * np.exp(-0.5 * ((wave - center) / 0.25) ** 2)
    return flux, 30.0 + 4.0 * np.log(teff / 5000.0) - 0.025 * (wave - wave[0]) / 100.0


def build_library(wave=WAVE):
    nodes, normalized, continua = [], [], []
    for teff in TEFF_AXIS:
        for logg in LOGG_AXIS:
            for mh in MH_AXIS:
                flux, log_continuum = toy_spectrum(teff, logg, mh, wave)
                nodes.append((teff, logg, mh))
                normalized.append(flux)
                continua.append(log_continuum)
    return ab.SpectralLibrary(
        label_names=("teff", "logg", "mh"),
        nodes=np.asarray(nodes),
        normalized=np.asarray(normalized),
        log_continuum=np.asarray(continua),
        wave=wave,
        medium="air",
        meta={"grid": "toy at BOSZ node density"},
    )


def bench_interpolation(library):
    print("\n--- interpolation error (leave-out at doubled node spacing) ---")
    print(f"{'method':<10} {'rms':>10} {'median':>10} {'p95':>10} {'max':>10}  n")
    for method in ("linear", "cubic"):
        report = ab.crossval_library(library, method=method)
        print(
            f"{method:<10} {report['rms']:10.3e} {report['median']:10.3e} "
            f"{report['p95']:10.3e} {report['max']:10.3e}  {report['n_tested']}"
        )
    print("  reference (Meszaros & Allende Prieto 2013, same spacing, real ATLAS9 grid):")
    print("    linear 5.1e-04    cubic-Bezier 3.1e-04    Payne-style network ~1e-03")

    interpolator = ab.library_interpolator(library)
    exact = all(
        np.array_equal(np.asarray(interpolator(library.nodes[i])[0]), library.normalized[i])
        for i in range(0, library.n_nodes, 37)
    )
    print(f"  node reproduction exact on this box grid (bit-for-bit): {exact}")


def inject(library, grid, noise, seed=20260827):
    interpolator = ab.library_interpolator(library.resampled_to(grid, medium="air"))
    rows = []
    for i, labels in enumerate(TRUTH.values()):
        deviation = (
            np.asarray(interpolator(np.array([labels["teff"], labels["logg"], labels["mh"]]))[0])
            - 1.0
        )
        kernel = np.asarray(ab.rotational_kernel(labels["vsini"] / grid.dv_kms))
        rows.append(np.convolve(deviation, kernel, mode="same") * TRUE_LIGHT[i] / ASSUMED_LIGHT[i])
    rows = np.stack(rows)
    return rows + np.random.default_rng(seed).normal(0.0, noise, rows.shape)


def declare(library):
    return {
        "A": ab.StarLabels(
            library=library,
            teff=ab.Between(4200.0, 6200.0),
            logg=ab.Between(3.2, 4.9),
            vsini=ab.Between(1.0, 60.0),
            v_kms=ab.Fixed(0.0),
        ),
        "B": ab.StarLabels(
            library=library,
            teff=ab.Between(4100.0, 5600.0),
            logg=ab.Between(3.2, 4.9),
            vsini=ab.Between(1.0, 60.0),
            v_kms=ab.Fixed(0.0),
        ),
    }


def bench_recovery(library, grid, noises, steps):
    print("\n--- closed-loop recovery (off-node injection, wrong assumed light ratio) ---")
    print(
        f"{'S/N':>6} {'star':>5} {'dTeff/K':>9} {'dlogg':>8} {'dmh':>8} "
        f"{'dvsini':>8} {'formal K':>9} {'ell':>7}"
    )
    fits = {}
    for noise in noises:
        rows = inject(library, grid, noise)
        match = ab.match_labels(
            grid,
            rows,
            stars=declare(library),
            medium="air",
            light_fractions=ASSUMED_LIGHT,
            lsf_sigma_kms=LSF_SIGMA_KMS,
            std=np.full_like(rows, noise),
            mh=ab.Between(-0.9, 0.4),
            dilution=ab.RadiusRatio(),
            max_steps=steps,
        )
        fits[noise] = match
        formal = match.errors("laplace")
        for name, truth in TRUTH.items():
            got = match.labels[name]
            print(
                f"{1 / noise:6.0f} {name:>5} {got['teff'] - truth['teff']:+9.1f} "
                f"{got['logg'] - truth['logg']:+8.4f} {got['mh'] - truth['mh']:+8.4f} "
                f"{got['vsini'] - truth['vsini']:+8.3f} {formal[name]['teff']:9.2f} "
                f"{match.flux_ratio[name]:7.3f}"
            )
        print(
            f"       chi2 {match.chi2:.1f} / {match.n_pixels_used} px; nulls: "
            f"node {match.chi2_nearest_node:.4g}, continuum {match.chi2_continuum:.4g}"
        )
    print(f"  true light fractions {list(TRUE_LIGHT)}, assumed {list(ASSUMED_LIGHT)}")
    return fits


def bench_stages(library, grid, steps):
    print("\n--- wall clock by stage ---")
    rows = inject(library, grid, 0.004)
    timings = {}

    t0 = time.perf_counter()
    projected = library.resampled_to(grid, medium="air")
    timings["resample library onto the model grid"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ab.library_interpolator(projected)
    timings["build interpolator"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    match = ab.match_labels(
        grid,
        rows,
        stars=declare(library),
        medium="air",
        light_fractions=ASSUMED_LIGHT,
        lsf_sigma_kms=LSF_SIGMA_KMS,
        std=np.full_like(rows, 0.004),
        mh=ab.Between(-0.9, 0.4),
        dilution=ab.RadiusRatio(),
        max_steps=steps,
    )
    timings["full match_labels (scan + 4 x L-BFGS + Laplace)"] = time.perf_counter() - t0

    draws = rows[None, :, :] + np.random.default_rng(11).normal(0.0, 0.004, (8, *rows.shape))
    t0 = time.perf_counter()
    propagated = ab.refit_draws(match, draws, max_steps=40)
    timings["refit 8 posterior draws"] = time.perf_counter() - t0

    for label, seconds in timings.items():
        print(f"  {label:<52} {seconds:7.2f} s")
    print(f"  library: {library.n_nodes} nodes x {library.n_pix} px -> {grid.n} model px")

    print("\n--- formal error against the honest one ---")
    formal, spread = propagated.errors("laplace"), propagated.errors("draws")
    for name in propagated.labels:
        for label in ("teff", "logg", "vsini"):
            f, s = formal[name].get(label), spread[name].get(label)
            if f and s:
                print(f"  {name} {label:<6} formal {f:8.3f}   draws {s:8.3f}   x{s / f:.1f}")
    print("  literature: 5-10x on real disentangled spectra (Gebruers+ 2022, Czekala+ 2015)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="fewer steps and one S/N tier")
    args = parser.parse_args()

    steps = 200 if args.quick else 600
    noises = [0.01] if args.quick else [0.01, 0.004, 0.002]

    print(f"machine: {platform.processor() or platform.machine()} / {platform.system()}")
    print(f"python {platform.python_version()}")
    t_start = time.perf_counter()

    library = build_library()
    grid = ab.LogGrid.from_wavelength_range(5165.0, 5235.0, dv_kms=4.0)
    print(library.summary())

    bench_interpolation(library)
    bench_recovery(library, grid, noises, steps)
    bench_stages(library, grid, steps)

    print(f"\ntotal wall {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
