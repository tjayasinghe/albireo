"""What the TODCOR mode delivers, and what it costs (D56).

Produces the numbers ``docs/benchmarks.md`` records for ``albireo.todcor``:

1. precision, bias and error calibration against signal-to-noise: the Zucker (2003)
   test, ``(v - v_true) / sigma`` over many noise realizations;
2. the pixel-locking bias of the linear shift operator against the template sampling,
   measured on noiseless data simulated on a finer grid than the templates (so the
   template interpolation error is real rather than an inverse crime);
3. two-dimensional against one-dimensional correlation as the components blend, which
   is the case the method addresses;
4. wall clock per epoch against pixel count and search range, plus the compile;
5. three components.

Everything is offline and generated here.

    python scripts/todcor_bench.py [--quick]
"""

from __future__ import annotations

import argparse
import os
import platform
import time
import warnings

import numpy as np

import albireo as ab
from albireo.operators import rebin_operator
from albireo.todcor import Template, todcor

LIGHT = (0.6, 0.4)
LSF = 5.0  # km/s
ORBIT = ab.OrbitParams(period=6.31, t_peri=2.0, ecc=0.15, omega=0.7, k=(30.0, 55.0), gamma=12.0)
RUN = {"v_range": (-150.0, 150.0), "light": LIGHT, "lsf_sigma_v": {"a": LSF}}


def machine() -> str:
    cpu = platform.processor() or platform.machine()
    return f"{platform.system()} {platform.release()}, {cpu}, {os.cpu_count()} threads"


def components(grid, seeds=(21, 22), **kwargs):
    options = {"sigma_v_range": (4.0, 12.0), "margin": 0.12}
    options.update(kwargs)
    return [ab.synthetic_deviation_spectrum(grid, seed=s, **options) for s in seeds]


def simulate(grid, comps, *, snr, seed, n_epochs=8, step=0.05, velocities=None, light=LIGHT):
    rng = np.random.default_rng(3)
    bjd = np.sort(rng.uniform(0.0, 21.0, size=n_epochs))
    inst = {"a": ab.InstrumentSpec(wave=np.arange(5008.0, 5052.0, step), sigma_v_lsf=LSF, snr=snr)}
    options = {"instruments": inst, "light_fractions": light, "seed": seed}
    if velocities is None:
        options["orbit"] = ORBIT
    else:
        options["velocities"] = velocities
    return ab.simulate_dataset(grid, comps, bjd=bjd, **options)


def table(header: str, rows: list[list[str]]) -> None:
    columns = header.split("|")
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in rows:
        print("| " + " | ".join(row) + " |")


# ---------------------------------------------------------------------------
# 1. precision and calibration against S/N
# ---------------------------------------------------------------------------


def bench_snr(n_real: int):
    grid = ab.LogGrid.from_wavelength_range(5000.0, 5060.0, dv_kms=1.0)
    comps = components(grid)
    templates = [Template(n, grid, c, v_zero_kms=0.0) for n, c in zip("AB", comps, strict=True)]
    print(
        f"\n## Precision, bias and calibration against S/N "
        f"(fixed light, 8 epochs x {n_real} realizations)\n"
    )
    rows = []
    for snr in (30.0, 100.0, 300.0):
        errors, sigmas, sigmas_ivar = [], [], []
        for seed in range(n_real):
            dataset, truth = simulate(grid, comps, snr=snr, seed=100 + seed)
            result = todcor(dataset, templates, **RUN)
            errors.append(result.velocity - truth.velocities)
            sigmas.append(result.sigma)
            sigmas_ivar.append(result.sigma_ivar)
        err = np.stack(errors)  # (n_real, 2, n_ep)
        sig = np.stack(sigmas)
        sig_i = np.stack(sigmas_ivar)
        for i, name in enumerate("AB"):
            e, s, si = err[:, i].ravel(), sig[:, i].ravel(), sig_i[:, i].ravel()
            rows.append(
                [
                    f"{snr:.0f}",
                    name,
                    f"{e.mean():+.4f} +- {e.std() / np.sqrt(e.size):.4f}",
                    f"{e.std():.4f}",
                    f"{s.mean():.4f}",
                    f"{np.sqrt(np.mean((e / s) ** 2)):.3f}",
                    f"{np.sqrt(np.mean((e / si) ** 2)):.3f}",
                ]
            )
    table(
        "S/N|star|bias [km/s]|scatter [km/s]|mean quoted sigma [km/s]|pull rms|"
        "pull rms (ivar errors)",
        rows,
    )


# ---------------------------------------------------------------------------
# 2. pixel locking against template sampling
# ---------------------------------------------------------------------------


def bench_pixel_locking():
    fine = ab.LogGrid.from_wavelength_range(5000.0, 5060.0, dv_kms=0.25)
    comps = components(fine)
    # Fractional-pixel velocities on the coarsest template grid, spanning a whole pixel.
    velocities = np.array(
        [[10.0 + 0.125 * 5.0 * k for k in range(8)], [-20.0 - 0.125 * 5.0 * k for k in range(8)]]
    )
    dataset, truth = simulate(fine, comps, snr=1e5, seed=1, velocities=velocities)
    print(
        "\n## Pixel locking of the shift operator against template sampling "
        "(noiseless data simulated at 0.25 km/s)\n"
    )
    rows = []
    for dv in (0.5, 1.0, 2.5, 5.0):
        grid = ab.LogGrid.from_wavelength_range(5000.0, 5060.0, dv_kms=dv)
        op = rebin_operator(x_in=fine.wave, x_out=grid.wave)
        coarse = [np.asarray(op(c)) for c in comps]
        templates = [
            Template(n, grid, c, v_zero_kms=0.0) for n, c in zip("AB", coarse, strict=True)
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the coarse grids are intentional here
            result = todcor(dataset, templates, **RUN)
        err = result.velocity - truth.velocities
        sigma_px = LSF / dv
        rows.append(
            [
                f"{dv:.2f}",
                f"{sigma_px:.1f}",
                f"{0.1 / sigma_px**2:.4f}",
                f"{np.max(np.abs(err)) / dv:.4f}",
                f"{np.max(np.abs(err)):.4f}",
                f"{np.sqrt(np.mean(err**2)):.4f}",
            ]
        )
    table(
        "template dv [km/s]|LSF sigma [px]|estimate 0.1/sigma_px^2 [px]|max abs error [px]|"
        "max abs error [km/s]|rms error [km/s]",
        rows,
    )


# ---------------------------------------------------------------------------
# 3. two dimensions against one as the lines blend
# ---------------------------------------------------------------------------


def bench_blending():
    grid = ab.LogGrid.from_wavelength_range(5000.0, 5060.0, dv_kms=1.0)
    comps = components(grid)
    templates = [Template(n, grid, c, v_zero_kms=0.0) for n, c in zip("AB", comps, strict=True)]
    seps = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 160.0])
    velocities = np.stack([10.0 + 0.5 * seps, 10.0 - 0.5 * seps])
    dataset, truth = simulate(grid, comps, snr=200.0, seed=7, velocities=velocities)
    two = todcor(dataset, templates, **RUN)
    one = todcor(
        dataset, templates[:1], v_range=(-150.0, 150.0), light=[1.0], lsf_sigma_v={"a": LSF}
    )
    print(
        "\n## Two-dimensional against one-dimensional correlation as the components blend "
        "(S/N 200, light 0.6/0.4)\n"
    )
    rows = []
    for j, sep in enumerate(seps):
        rows.append(
            [
                f"{sep:.0f}",
                f"{one.velocity[0, j] - truth.velocities[0, j]:+.3f}",
                f"{two.velocity[0, j] - truth.velocities[0, j]:+.3f}",
                f"{two.velocity[1, j] - truth.velocities[1, j]:+.3f}",
                f"{two.sigma[0, j]:.3f}",
                "yes" if two.blended[j] else "no",
            ]
        )
    table(
        "separation [km/s]|1-D primary error [km/s]|2-D primary error [km/s]|"
        "2-D secondary error [km/s]|2-D sigma A|blended flag",
        rows,
    )


# ---------------------------------------------------------------------------
# 4. wall clock
# ---------------------------------------------------------------------------


def bench_timing():
    print("\n## Wall clock per epoch (fixed light, one instrument; min of 3 after a warm-up)\n")
    cases = (
        (5000.0, 5060.0, 0.2, 40),
        (5000.0, 5060.0, 0.05, 40),
        (5000.0, 5060.0, 0.0125, 40),
        (4500.0, 6500.0, 0.05, 1200),  # a whole optical range, spanning many echelle orders
    )
    rows = []
    for lo, hi, step, n_lines in cases:
        grid = ab.LogGrid.from_wavelength_range(lo, hi, dv_kms=1.0)
        comps = components(grid, n_lines=n_lines)
        templates = [Template(n, grid, c, v_zero_kms=0.0) for n, c in zip("AB", comps, strict=True)]
        for span in (100.0, 300.0):
            rng = np.random.default_rng(3)
            bjd = np.sort(rng.uniform(0.0, 21.0, size=4))
            wave = np.arange(lo + 8.0, hi - 8.0, step)
            inst = {"a": ab.InstrumentSpec(wave=wave, sigma_v_lsf=LSF, snr=100.0)}
            dataset, _ = ab.simulate_dataset(
                grid, comps, bjd=bjd, instruments=inst, light_fractions=LIGHT, orbit=ORBIT, seed=5
            )
            one_epoch = ab.Dataset([dataset[0]], frame=dataset.frame)
            options = {"v_range": (-span, span), "light": LIGHT, "lsf_sigma_v": {"a": LSF}}
            t0 = time.perf_counter()
            todcor(one_epoch, templates, **options)
            first = time.perf_counter() - t0
            best = np.inf
            for _ in range(3):
                t0 = time.perf_counter()
                result = todcor(dataset, templates, **options)
                best = min(best, (time.perf_counter() - t0) / dataset.n_epochs)
            rows.append(
                [
                    f"{hi - lo:.0f}",
                    f"{dataset[0].n_pixels}",
                    f"+-{span:.0f}",
                    f"{result.settings['coarse_step']}",
                    f"{first:.2f}",
                    f"{best:.3f}",
                ]
            )
    table(
        "window [A]|native pixels|search range [km/s]|coarse step [px]|"
        "compile + first epoch [s]|per epoch [s]",
        rows,
    )


# ---------------------------------------------------------------------------
# 5. three components
# ---------------------------------------------------------------------------


def bench_three():
    grid = ab.LogGrid.from_wavelength_range(5000.0, 5060.0, dv_kms=1.0)
    comps = components(grid, seeds=(21, 22, 23))
    velocities = np.array(
        [[-30.0, 20.0, 45.0, 5.0], [40.0, -35.0, -60.0, 12.0], [5.0, 8.0, 3.0, -50.0]]
    )
    light = (0.5, 0.3, 0.2)
    dataset, truth = simulate(
        grid, comps, snr=200.0, seed=3, n_epochs=4, velocities=velocities, light=light
    )
    templates = [Template(n, grid, c, v_zero_kms=0.0) for n, c in zip("ABC", comps, strict=True)]
    t0 = time.perf_counter()
    result = todcor(dataset, templates, v_range=(-80.0, 80.0), light="free", lsf_sigma_v={"a": LSF})
    wall = time.perf_counter() - t0
    err = result.velocity - truth.velocities
    print("\n## Three components (light 0.5/0.3/0.2, S/N 200, four epochs, +-80 km/s)\n")
    rows = []
    for i, name in enumerate("ABC"):
        rows.append(
            [
                name,
                f"{np.sqrt(np.mean(err[i] ** 2)):.4f}",
                f"{result.sigma[i].mean():.4f}",
                f"{np.sqrt(np.mean((err[i] / result.sigma[i]) ** 2)):.3f}",
                f"{np.median(result.light[i]):.3f}",
            ]
        )
    table("star|rms error [km/s]|mean quoted sigma [km/s]|pull rms|recovered light", rows)
    print(f"\nWall: {wall:.1f} s for four epochs including the compile.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_real = 4 if args.quick else 16
    print(f"# TODCOR benchmark - {machine()}, jax CPU float64")
    t_start = time.perf_counter()
    bench_snr(n_real)
    bench_pixel_locking()
    bench_blending()
    bench_timing()
    bench_three()
    print(f"\nTotal wall: {time.perf_counter() - t_start:.0f} s")


if __name__ == "__main__":
    main()
