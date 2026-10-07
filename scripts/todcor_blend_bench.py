"""The TODCOR search at blended epochs, against the lowest chi-square of a lattice of shifts.

    python scripts/todcor_blend_bench.py
    python scripts/todcor_blend_bench.py --reference path/to/an/older/todcor.py
    python scripts/todcor_blend_bench.py --margin [--thresholds 4 9 16 25]

Where the lines of two components overlap, the chi-square of ``albireo.todcor`` has two
minima: the solution, and the pair exchanged about its light-weighted mean velocity
(``docs/math.md`` §10.3). This script measures whether the search returns the lower one.
The epochs are those of step 3 of ``docs/tutorials/gaia-rvs-todcor.ipynb``: a 6100 K and a
5700 K dwarf at light fractions of 0.625 and 0.375, simulated as Gaia RVS epochs on the
DR4 grid at separations of 0 to 160 km/s, forty epochs at each, at four values of the S/N.
For every epoch with the lines at most 60 km/s apart, the exact chi-square is evaluated on
a lattice of an eighth of a template pixel over 28 pixels on either side of the injected
pair, and the lowest lattice value is refined below the lattice step. The search is counted
as above the minimum where its chi-square exceeds that value by more than 0.01.

The lattice uses the inner products of ``albireo.todcor`` at the integer shifts and its own
transcription of the interpolation identity, so it tests the search and not the operators,
which ``tests/test_todcor.py`` checks against direct summation. With ``--reference`` the
same epochs are also measured by another copy of the module, for instance the one before
the search was changed (``git show d5cb014:src/albireo/todcor.py``).

``--margin`` adds a second table, of the epochs that are wrong and of what the blend flag
does with them. An epoch is wrong where either velocity is more than five quoted errors and
3 km/s from the injected one. The search keeps every minimum it refines, and
``VelocityTable.margin`` is the rise in chi-square from the one returned to the lowest
other one at which the velocities differ in every order of the components. Below 9 times
the reduced chi-square it raises ``VelocityTable.blended``. The table gives, for the wrong
epochs: how many are at the exchanged pair of the injected one, how many have the injected
pair among the refined minima and at what rise, how many the margin flags, and how many of
the others are right once the two velocities are interchanged. It also gives the number of
correct epochs the margin flags. ``--thresholds`` repeats the counts for other thresholds
in place of 9, with the search keeping every minimum within the largest.

The script needs the RVS box of the BOSZ grid (``albireo.fetch_library``). The results are
recorded in ``docs/benchmarks.md``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time

import jax.numpy as jnp
import numpy as np

import albireo as ab
from albireo.gaia import (
    RVS_DR4_EPOCH,
    rvs_components,
    rvs_delivered_sigma_kms,
    rvs_model_grid,
    simulate_rvs_dataset,
)

PRIMARY = {"teff": 6100.0, "logg": 4.2, "mh": -0.1}
SECONDARY = {"teff": 5700.0, "logg": 4.3, "mh": -0.1}
VSINI = (12.0, 11.0)
LIGHT = (0.625, 0.375)
SEPARATIONS = np.array([0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 120, 160], dtype=float)
PER_SEPARATION = 40
V_SEARCH = 250.0
LATTICE_HALF_WIDTH = 28  # template pixels on either side of the injected pair
LATTICE_STEPS = 8  # lattice points per template pixel


def separation_sweep(components, grid, library, snr, seed=7):
    """Epochs of the pair at fixed separations: the sweep of the notebook's step 3."""
    rng = np.random.default_rng(seed)
    n = SEPARATIONS.size * PER_SEPARATION
    sep = np.repeat(SEPARATIONS, PER_SEPARATION) * rng.choice([-1.0, 1.0], n)
    mean = rng.uniform(-40.0, 40.0, n)
    velocities = np.stack([mean - LIGHT[1] * sep, mean + LIGHT[0] * sep])
    epochs, injected = simulate_rvs_dataset(
        components,
        grid,
        bjd=2457000.0 + np.arange(n),
        light_fractions=LIGHT,
        velocities=velocities,
        snr=snr,
        product=RVS_DR4_EPOCH,
        library=library,
        seed=seed,
    )
    return epochs, np.asarray(injected.velocities), np.abs(sep)


def lattice_chi2(terms, amps, position):
    """The chi-square of two templates at every pair of fractional window positions.

    The inner products at a fractional shift are the two-tap interpolation of those at the
    integer shifts on either side (``docs/math.md`` §10.3), applied here to the whole
    lattice at once with the amplitudes held and the nuisance profiled.
    """
    n = np.clip(np.floor(position).astype(int), 0, terms.n_shift - 2)
    f = position - n

    def linear(values):  # along the last axis
        return (1.0 - f) * values[..., n] + f * values[..., n + 1]

    def diagonal(gram):
        lo, hi = gram[n, n], gram[n + 1, n + 1]
        return (1.0 - f) ** 2 * lo + 2.0 * f * (1.0 - f) * gram[n, n + 1] + f**2 * hi

    b1, b2 = linear(terms.b[0]), linear(terms.b[1])
    cross = terms.gram[0, 1]
    g12 = (
        np.outer(1.0 - f, 1.0 - f) * cross[np.ix_(n, n)]
        + np.outer(1.0 - f, f) * cross[np.ix_(n, n + 1)]
        + np.outer(f, 1.0 - f) * cross[np.ix_(n + 1, n)]
        + np.outer(f, f) * cross[np.ix_(n + 1, n + 1)]
    )
    a1, a2 = amps
    chi2 = (
        terms.zwz
        - 2.0 * (a1 * b1[:, None] + a2 * b2[None, :])
        + a1**2 * diagonal(terms.gram[0, 0])[:, None]
        + a2**2 * diagonal(terms.gram[1, 1])[None, :]
        + 2.0 * a1 * a2 * g12
    )
    if terms.m:
        p1, p2 = linear(terms.pwa[0]), linear(terms.pwa[1])
        resid = terms.pwz[:, None, None] - a1 * p1[:, :, None] - a2 * p2[:, None, :]
        chi2 = chi2 - np.einsum("mij,mn,nij->ij", resid, np.linalg.inv(terms.pwp), resid)
    return chi2


def lattice_minimum(dataset, j, templates, sigma_kms, v_true):
    """The lowest chi-square of epoch ``j`` on the lattice, refined below the lattice step."""
    from albireo import todcor as module  # the function shadows the module on the package

    module = sys.modules[module.__module__]
    grid = templates[0].grid
    epoch = dataset[j]
    stack, _ = module._convolved_templates(
        templates, grid, epoch.instrument, {epoch.instrument: sigma_kms}, None
    )
    work = module._prepare_epoch(j, epoch, grid, 0)
    centre = int(np.rint(np.asarray(grid.velocity_to_pixels(np.mean(v_true)))))
    n = 2 * LATTICE_HALF_WIDTH + 1
    start = centre - LATTICE_HALF_WIDTH
    padded = module._round_up(n, module._SHIFT_CHUNK)
    deltas = np.stack([start + np.arange(padded)] * 2).astype(np.int32)
    out = module._epoch_terms(
        jnp.asarray(stack),
        work.rows,
        work.cols,
        work.vals,
        work.z,
        work.w,
        jnp.asarray(deltas),
        work.basis,
        chunk=module._SHIFT_CHUNK,
    )
    terms = module._terms_numpy(out, n)
    position = np.arange(0, (n - 1) * LATTICE_STEPS + 1) / LATTICE_STEPS
    surface = lattice_chi2(terms, np.asarray(LIGHT), position)
    i1, i2 = np.unravel_index(int(np.argmin(surface)), surface.shape)
    lowest = float(surface[i1, i2])
    nearest = np.clip(np.rint([position[i1], position[i2]]).astype(int), 1, n - 2)
    refined = module._refine(terms, nearest, np.asarray(LIGHT), "fixed")[0]
    return min(lowest, float(refined))


def wrong(table, v_true):
    """Epochs with either velocity more than five quoted errors and 3 km/s off."""
    error = np.abs(table.velocity - v_true)
    return np.any((error > 5.0 * table.sigma) & (error > 3.0), axis=0)


def wrong_in_either_order(table, v_true):
    """Epochs that are wrong as labelled and with the two velocities interchanged."""
    error = np.abs(table.velocity[::-1] - v_true)
    interchanged = np.any((error > 5.0 * table.sigma[::-1]) & (error > 3.0), axis=0)
    return wrong(table, v_true) & interchanged


def ridge(table):
    """Epochs the curvature flags: a covariance correlation above 0.9, or no error."""
    c = table.covariance
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.abs(c[:, 0, 1]) / np.sqrt(c[:, 0, 0] * c[:, 1, 1])
    return ~np.all(np.isfinite(table.sigma), axis=0) | (correlation > 0.9)


def exchanged_pair(v_true):
    """The pair with the light-weighted mean of ``v_true`` and the opposite difference."""
    mean = LIGHT[0] * v_true[0] + LIGHT[1] * v_true[1]
    difference = v_true[0] - v_true[1]
    return np.stack([mean - LIGHT[1] * difference, mean + LIGHT[0] * difference])


def near(velocity, target, sigma):
    """Both velocities within three quoted errors, or 4 km/s if larger, of ``target``."""
    return np.all(np.abs(velocity - target) < np.maximum(3.0 * sigma, 4.0), axis=0)


def search_with_minima(dataset, templates, options):
    """The velocity table of ``albireo.todcor`` and the minima it refined in every epoch.

    The minima are recorded by wrapping the module's fine pass, which returns those it
    refined and is called once per starting point, and its epoch preparation, which is
    called once per epoch. Each is the rise in chi-square above the minimum returned and
    the velocities.
    """
    from albireo import todcor as function

    module = sys.modules[function.__module__]
    refine_at, prepare_epoch = module._refine_at, module._prepare_epoch
    grid = templates[0].grid
    epochs: list[tuple[float, list]] = []

    def recording_refine_at(*args):
        solution, found = refine_at(*args)
        epochs[-1][1].extend(found)
        return solution, found

    def recording_prepare_epoch(*args):
        work = prepare_epoch(*args)
        epochs.append((work.bary_pix, []))
        return work

    module._refine_at, module._prepare_epoch = recording_refine_at, recording_prepare_epoch
    try:
        table = function(dataset, templates, **options)
    finally:
        module._refine_at, module._prepare_epoch = refine_at, prepare_epoch
    minima = [
        [
            (
                minimum[0] - table.chi2[j],
                module._reported_velocities(
                    grid, templates, minimum[4] + minimum[1], dataset.frame, bary_pix
                ),
            )
            for minimum in found
        ]
        for j, (bary_pix, found) in enumerate(epochs)
    ]
    return table, minima


def margin_rows(snr, table, minima, v_true, thresholds):
    """The rows of the margin table for one S/N: the flag as built, and other thresholds."""
    from albireo import todcor as function

    module = sys.modules[function.__module__]
    missed = wrong(table, v_true)
    either = wrong_in_either_order(table, v_true)
    at_exchanged = near(table.velocity, exchanged_pair(v_true), table.sigma)
    rise = np.full(table.n_epochs, np.inf)  # to the refined minimum at the injected pair
    for j in np.flatnonzero(missed):
        at_injected = [
            c for c, v in minima[j] if near(v[:, None], v_true[:, [j]], table.sigma[:, [j]])[0]
        ]
        rise[j] = min(at_injected, default=np.inf)
    with_injected = missed & np.isfinite(rise)
    largest = f"{rise[with_injected].max():.1f}" if with_injected.any() else "-"
    rows = []
    for threshold in thresholds:
        margin = np.array(
            [
                module._margin(0.0, table.velocity[:, j], table.sigma[:, j], minima[j])[0]
                if np.all(np.isfinite(table.sigma[:, j]))
                else np.nan
                for j in range(table.n_epochs)
            ]
        )
        with np.errstate(invalid="ignore"):
            flagged = (margin < threshold * table.reduced_chi2) & ~ridge(table)
        rows.append(
            f"| {snr:.0f} | {threshold:g} | {int(missed.sum())} "
            f"| {int((missed & at_exchanged).sum())} "
            f"| {int(with_injected.sum())} ({largest}) | {int((missed & flagged).sum())} "
            f"| {int((missed & ~flagged & ~either).sum())} | {int((either & ~flagged).sum())} "
            f"| {int((~missed & flagged).sum())} | {int(ridge(table).sum())} |"
        )
    return rows


def load_module(path):
    spec = importlib.util.spec_from_file_location("todcor_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["todcor_reference"] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", help="another copy of albireo/todcor.py to measure as well")
    parser.add_argument("--snr", nargs="+", type=float, default=[15.0, 40.0, 100.0, 300.0])
    parser.add_argument(
        "--margin",
        action="store_true",
        help="also tabulate the wrong epochs and what the margin to a second minimum flags",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help="thresholds on the margin to tabulate in place of the one albireo.todcor applies",
    )
    args = parser.parse_args(argv)
    module = sys.modules[ab.todcor.__module__]
    thresholds = args.thresholds or [module._BLEND_CHI2]
    if args.margin:
        # The search keeps the minima within its threshold, so the largest one is set.
        module._BLEND_CHI2 = max(thresholds)

    library = ab.fetch_library("bosz2024-fgk-rvs")
    grid = rvs_model_grid(V_SEARCH, vsini_max_kms=max(VSINI))
    components = rvs_components(library, [PRIMARY, SECONDARY], grid, vsini_kms=VSINI)
    sigma = rvs_delivered_sigma_kms(RVS_DR4_EPOCH, simulation_dv_kms=grid.dv_kms)
    searches = {"albireo.todcor": ab.todcor}
    if args.reference:
        searches = {"reference": load_module(args.reference).todcor, **searches}

    print(
        "| S/N | search | above the lattice minimum | largest excess | wrong epochs "
        "| ms per epoch |"
    )
    print("|---|---|---|---|---|---|")
    margins = []
    for snr in args.snr:
        dataset, v_true, sep = separation_sweep(components, grid, library, snr)
        template_grid = ab.LogGrid.covering(
            dataset, sigma / 3.0, v_margin_kms=V_SEARCH + 60.0, lsf_sigma_kms=sigma
        )
        templates = [
            ab.Template.from_library(
                name,
                library,
                labels,
                grid=template_grid,
                medium="vacuum",
                vsini_kms=v,
                resolving_power=library.resolving_power,
            )
            for name, labels, v in zip(
                ("primary", "secondary"), (PRIMARY, SECONDARY), VSINI, strict=True
            )
        ]
        blended = np.flatnonzero(sep <= 60.0)
        lowest = np.array(
            [lattice_minimum(dataset, j, templates, sigma, v_true[:, j]) for j in blended]
        )
        options = {
            "v_range": (-V_SEARCH, V_SEARCH),
            "light": LIGHT,
            "lsf_sigma_v": {"RVS": sigma},
            "noise_correlation": {"RVS": 0.27},
        }
        for name, search in searches.items():
            search(ab.Dataset([dataset[0]], frame="barycentric"), templates, **options)  # compile
            t0 = time.perf_counter()
            table = search(dataset, templates, **options)
            per_epoch = 1e3 * (time.perf_counter() - t0) / dataset.n_epochs
            excess = table.chi2[blended] - lowest
            print(
                f"| {snr:.0f} | {name} | {int(np.sum(excess > 0.01))} of {blended.size} "
                f"| {excess.max():.2f} | {int(wrong(table, v_true).sum())} of "
                f"{dataset.n_epochs} | {per_epoch:.1f} |"
            )
        if args.margin:
            table, minima = search_with_minima(dataset, templates, options)
            margins += margin_rows(snr, table, minima, v_true, thresholds)
    if margins:
        print(
            "\n| S/N | threshold | wrong epochs | at the exchanged pair | with the injected pair "
            "a refined minimum (largest rise) | flagged by the margin | not flagged, right when "
            "interchanged | not flagged, wrong in either order | correct epochs flagged "
            "| flagged by the curvature |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        print("\n".join(margins))
    return 0


if __name__ == "__main__":
    sys.exit(main())
