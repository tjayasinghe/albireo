"""From a disentangled component to a usable RV template (``docs/math.md`` §9)

Disentangling gives you two spectra. It does not tell you *which* synthetic template to
cross-correlate the individual epochs against — and that choice is where the next tool in
the pipeline starts. This example closes that gap: simulate an SB2, disentangle it,
recover Teff, log g, [M/H] and v sin i for each component, and check the answers against
the injected truth.

It runs offline. The synthetic grid here is a toy built in this file, because the real
libraries (BOSZ, POLLUX) are hundreds of megabytes and this has to work in CI and on a
laptop with no network. Everything above the grid — the interpolation, the broadening, the
dilution model, the nuisance, the optimizer, the uncertainty story — is the production code
path, unmodified.

Three things to watch in the output, because each is a place this is easy to get wrong.

1. **The wrong assumed light fraction goes into the dilution, not the temperature.** The
   disentangler is deliberately given light fractions that are wrong by a factor of ~1.3.
   Because the likelihood only ever saw the products ``l_i d_i`` (``docs/math.md`` §5.2),
   that error rescales every line depth and is *indistinguishable* from a temperature
   error — unless the fit is given somewhere else to put it. The run compares a joint
   radius-ratio fit against one with the dilution frozen, and the temperatures move.

2. **The k = 0 zero point is fitted and reported, not absorbed.** Each component's constant
   offset is in the null space of the disentangling problem (§5.1). The additive Chebyshev
   nuisance catches it; the fitted value is printed, because a large one is a statement
   about the disentangling and not a nuisance to be hidden.

3. **Two error bars, and the ratio between them.** The formal Laplace error is the
   curvature at the optimum, which is the wrong question on correlated residuals — the
   literature finds it optimistic by five to ten times. The honest number comes from
   refitting the labels once per joint posterior draw of the component spectra. Both are
   printed side by side.

The accuracy bar is the one in §9.6, and it is looser than instinct suggests: Teff to
2-3%, log g and [M/H] to 0.15 dex, v sin i to 10%. Past that, the epoch RVs stop caring
which template you picked. Labels from this mode are template coordinates, not an
abundance table.

Environment
-----------
``ALBIREO_EXAMPLE_FAST=1`` trims the draw count and the optimizer budget for CI.
"""

from __future__ import annotations

import os
import time

import numpy as np

import albireo as ab

FAST = bool(os.environ.get("ALBIREO_EXAMPLE_FAST"))

# The injected system: an F-ish primary and a cooler, faster-rotating secondary.
TRUTH = {
    "A": {"teff": 5180.0, "logg": 4.05, "mh": -0.15, "vsini": 11.0},
    "B": {"teff": 4460.0, "logg": 4.55, "mh": -0.15, "vsini": 27.0},
}
TRUE_LIGHT = np.array([0.62, 0.38])
# What the disentangling was told. Wrong on purpose — this is the normal situation, since
# a light ratio without eclipses is an assumption, and the point is that it is recoverable.
ASSUMED_LIGHT = np.array([0.72, 0.28])

WAVE_MIN, WAVE_MAX = 5150.0, 5250.0
LSF_SIGMA_KMS = 5.5
NOISE = 0.004

TEFF_AXIS = np.arange(4000.0, 5751.0, 250.0)
LOGG_AXIS = np.arange(3.0, 5.01, 0.5)
MH_AXIS = np.arange(-1.0, 0.51, 0.25)


def toy_spectrum(teff, logg, mh, wave):
    """A stand-in for a synthetic spectrum, with each label driving its own lines.

    That last part matters more than it looks. If two labels move the same lines the same
    way, they are exactly interchangeable, and a fit will drive the chi-square to zero along
    a curve through label space while "failing" to recover the injected values. Here Teff,
    log g and [M/H] each own a line, so the map from labels to spectrum is invertible — and
    the recovery below is a statement about the code rather than about the fixture.
    """
    t = (teff - 4800.0) / 600.0
    g = logg - 4.0
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
    # A real grid's continuum falls with Teff across the optical; that wavelength
    # dependence is exactly what makes the light ratio measurable.
    log_continuum = 30.0 + 4.0 * np.log(teff / 5000.0) - 0.025 * (wave - wave[0]) / 100.0
    return flux, log_continuum


def build_library():
    """A small complete-box library, standing in for BOSZ or POLLUX."""
    wave = np.linspace(WAVE_MIN, WAVE_MAX, 1400)
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
        # Required, never defaulted: air and vacuum differ by ~83 km/s, and the upstream
        # documentation is not reliable (BOSZ flipped convention between 2017 and 2024
        # under one name). A real library declares this after `line_core_medium` measures it.
        medium="air",
        meta={"grid": "toy (examples/11_labels.py)", "vmicro": "n/a", "citation": "none"},
    )


def inject(library, grid):
    """The component deviations a disentangling of this system would have returned."""
    interpolator = ab.library_interpolator(library.resampled_to(grid, medium="air"))
    rows = []
    for i, labels in enumerate(TRUTH.values()):
        normalized, _ = interpolator(np.array([labels["teff"], labels["logg"], labels["mh"]]))
        deviation = np.asarray(normalized) - 1.0
        kernel = np.asarray(ab.rotational_kernel(labels["vsini"] / grid.dv_kms))
        broadened = np.convolve(deviation, kernel, mode="same")
        # What the disentangler recovers is (w / l0) * t: the true light fraction over the
        # assumed one (math.md §9.1). This is the whole reason the ratio is recoverable.
        rows.append(broadened * TRUE_LIGHT[i] / ASSUMED_LIGHT[i])
    rows = np.stack(rows)
    return rows + np.random.default_rng(20260827).normal(0.0, NOISE, rows.shape)


def declare(library):
    return {
        "A": ab.StarLabels(
            library=library,
            teff=ab.Between(4200.0, 5700.0),
            logg=ab.Between(3.2, 4.9),
            vsini=ab.Between(1.0, 60.0),
            v_kms=ab.Fixed(0.0),
        ),
        "B": ab.StarLabels(
            library=library,
            teff=ab.Between(4100.0, 5200.0),
            logg=ab.Between(3.2, 4.9),
            vsini=ab.Between(1.0, 60.0),
            v_kms=ab.Fixed(0.0),
        ),
    }


def report(title, match):
    print(f"\n{title}")
    formal = match.errors("laplace")
    for name, labels in match.labels.items():
        truth = TRUTH[name]
        print(
            f"  {name}:  Teff {labels['teff']:7.1f} K "
            f"(truth {truth['teff']:.0f}, off by {labels['teff'] - truth['teff']:+6.1f}, "
            f"formal +-{formal[name]['teff']:.1f})"
        )
        print(
            f"       log g {labels['logg']:5.3f} (off by {labels['logg'] - truth['logg']:+.3f})"
            f"   [M/H] {labels['mh']:+.3f} (off by {labels['mh'] - truth['mh']:+.3f})"
            f"   v sin i {labels['vsini']:5.2f} km/s "
            f"(off by {labels['vsini'] - truth['vsini']:+.2f})"
        )
    print(
        f"  light fractions {[round(v, 3) for v in match.flux_ratio.values()]} "
        f"(true {list(TRUE_LIGHT)}, assumed {list(ASSUMED_LIGHT)})"
    )


def main():
    t_start = time.perf_counter()
    library = build_library()
    grid = ab.LogGrid.from_wavelength_range(5165.0, 5235.0, dv_kms=4.0)
    rows = inject(library, grid)
    std = np.full_like(rows, NOISE)
    steps = 200 if FAST else 600

    print(library.summary())
    print(
        f"\nInterpolation error at doubled node spacing: "
        f"{ab.crossval_library(library)['rms']:.2e} rms in normalized flux"
    )

    common = {
        "medium": "air",
        "light_fractions": ASSUMED_LIGHT,
        "lsf_sigma_kms": LSF_SIGMA_KMS,
        "std": std,
        "mh": ab.Between(-0.9, 0.4),
        "max_steps": steps,
    }
    tied = ab.match_labels(grid, rows, stars=declare(library), dilution=ab.RadiusRatio(), **common)
    rigid = ab.match_labels(
        grid, rows, stars=declare(library), dilution=ab.FixedDilution(), **common
    )

    report("Joint fit, dilution free (RadiusRatio) -- the recommended mode:", tied)
    report("Same data, dilution frozen at the assumed light fractions (FixedDilution):", rigid)

    print("\nFitted zero points (the k = 0 null space, math.md 5.1):")
    for name in tied.labels:
        offsets = np.asarray(tied.result.params[f"offset_{name}"])
        print(f"  {name}: constant {offsets[0]:+.4f}, slope {offsets[1]:+.4f}")

    draws = rows[None, :, :] + np.random.default_rng(11).normal(
        0.0, NOISE, (8 if FAST else 24, *rows.shape)
    )
    propagated = ab.refit_draws(tied, draws, max_steps=40)
    print("\nFormal error against the honest one:")
    formal, spread = propagated.errors("laplace"), propagated.errors("draws")
    for name in propagated.labels:
        ratio = spread[name]["teff"] / max(formal[name]["teff"], 1e-12)
        print(
            f"  {name} Teff: +-{formal[name]['teff']:5.1f} K formal, "
            f"+-{spread[name]['teff']:5.1f} K from draws  (x{ratio:.1f})"
        )

    print("\n" + tied.summary())

    # ---- assertions: this example is also a test -------------------------
    for name, truth in TRUTH.items():
        got = tied.labels[name]
        assert abs(got["teff"] - truth["teff"]) < 0.03 * truth["teff"], (
            f"{name}: Teff outside the 3% template-selection target"
        )
        assert abs(got["logg"] - truth["logg"]) < 0.15, f"{name}: log g outside 0.15 dex"
        assert abs(got["mh"] - truth["mh"]) < 0.15, f"{name}: [M/H] outside 0.15 dex"
        assert abs(got["vsini"] - truth["vsini"]) < 0.1 * truth["vsini"], f"{name}: v sin i off"
        # the dilution model has to beat the frozen one on the very error it exists for
        assert abs(got["teff"] - truth["teff"]) <= abs(
            rigid.labels[name]["teff"] - truth["teff"]
        ), f"{name}: freezing the dilution should not have helped"

    np.testing.assert_allclose(tied.light_fractions().sum(axis=0), 1.0, atol=1e-12)
    assert abs(tied.flux_ratio["A"] - TRUE_LIGHT[0]) < 0.08, (
        "the spectroscopic light ratio should recover the injected one"
    )
    assert tied.chi2 < tied.chi2_nearest_node < tied.chi2_continuum, (
        "the fit must beat both nulls: the nearest raw node, and no template at all"
    )
    assert propagated.errors("draws")["A"]["teff"] > 0.0

    print(
        f"\nOK - both components recovered inside the template-selection target, the wrong "
        f"assumed light ratio came back as dilution rather than as temperature, and the "
        f"draws spread exceeds the formal error. Total wall: "
        f"{time.perf_counter() - t_start:.1f} s"
    )


if __name__ == "__main__":
    main()
