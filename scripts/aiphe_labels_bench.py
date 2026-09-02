"""AI Phoenicis: scoring a label fit against a system whose answer is already known.

``aiphe_bench.py`` validates the disentangling on this system, because AI Phe's orbit is
published to better than 0.02 per cent. This script validates the labels on the same system:
AI Phe is one of the few binaries where every quantity the label fit produces has an
independent published value:

    Teff       6310 K and 5010 K         Maxted et al. (2020), MNRAS 498, 332
    log g      4.001 and 3.598           derived below from the same solution
    R2/R1      1.6237                    from the fractional radii, run C

``RadiusRatio`` fits the two components jointly through one shared scalar, so the radius
ratio is returned by the label fit, and here it can be held against a published number
measured photometrically from eclipses rather than spectroscopically. Nothing in the fit is
told that number.

log g is derived rather than quoted. For a double-lined eclipsing binary the surface gravity
follows from the spectroscopic and photometric elements alone, with no absolute masses or
radii and no distance:

    g_1 = 2 pi sqrt(1 - e^2) K_2 / (P r_1^2 sin i)

which is Kepler's third law and the mass ratio with everything that cancels cancelled. Every
symbol on the right is already in ``aiphe_bench.py`` except the inclination. It reproduces
the published absolute masses and radii to 0.002 dex, which the script checks.

Scope. The disentangling is not re-derived: the velocities are fixed at the published orbit,
exactly as ``aiphe_bench.py`` does, so what is under test is the label fit and not the orbit.
The light fractions supplied to the disentangling are a blackbody estimate, accurate to a few
per cent at best, which makes this a test of the claim that a wrong assumed dilution returns
as dilution rather than as temperature.

Run:  python scripts/aiphe_labels_bench.py --data data/aiphe
      (fetch the spectra first with scripts/download_aiphe.py)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# The published system and the analysis window come from the disentangling benchmark, so the
# two scripts cannot drift apart on the numbers they share.
from aiphe_bench import (
    DV_KMS,
    ECC_PUB,
    ETA,
    K1_PUB,
    K2_PUB,
    LSF_SIGMA_V,
    P_PUB,
    R1_FRAC,
    R2_FRAC,
    TAU,
    TEFF1,
    TEFF2,
    WINDOW,
    light_fractions,
    load,
    published_velocities,
)

import albireo as ab
from albireo.forward import build_problem
from albireo.likelihood import marginal_loglikelihood
from albireo.priors import SmoothnessPrior

# Maxted et al. (2020), run C -- the same solution R1_FRAC and R2_FRAC come from.
INCLINATION_DEG = 88.5
# Published absolute values, used only to check the derivation below.
M1_PUB, R1_PUB = 1.1938, 1.8036
M2_PUB, R2_PUB = 1.2438, 2.9303

LIBRARY = "bosz2024-fgk-r20000"


def published_logg() -> tuple[float, float]:
    """Surface gravities from the orbital and photometric elements alone.

    g_i = 2 pi sqrt(1 - e^2) K_j / (P r_i^2 sin i), with j the *other* component: the
    companion's semi-amplitude carries this star's mass through the mass ratio.
    """
    period = P_PUB * 86400.0
    sin_i = np.sin(np.radians(INCLINATION_DEG))
    common = 2.0 * np.pi * np.sqrt(1.0 - ECC_PUB**2) / (period * sin_i)
    g1 = common * (K2_PUB * 1e3) / R1_FRAC**2
    g2 = common * (K1_PUB * 1e3) / R2_FRAC**2
    return float(np.log10(g1 * 100.0)), float(np.log10(g2 * 100.0))


def check_derivation() -> None:
    """The closed form against the published absolute masses and radii."""
    g_sun_cgs = 6.674e-11 * 1.98892e30 / (6.957e8) ** 2 * 100.0
    direct = [np.log10(g_sun_cgs * m / r**2) for m, r in ((M1_PUB, R1_PUB), (M2_PUB, R2_PUB))]
    derived = published_logg()
    for name, a, b in zip(("primary", "secondary"), derived, direct, strict=True):
        print(f"  log g {name:<9} {a:.4f} from the elements, {b:.4f} from M and R")
        assert abs(a - b) < 0.01, "the log g derivation disagrees with the published masses"


def disentangle(dataset):
    """Component spectra with the orbit held at the published solution."""
    ell = light_fractions(float(np.mean(WINDOW)))
    grid = ab.LogGrid.covering(
        dataset, dv_kms=DV_KMS, v_margin_kms=140.0, lsf_sigma_kms=LSF_SIGMA_V
    )
    problem = build_problem(
        grid,
        dataset,
        velocities=published_velocities(np.asarray(dataset.bjd)),
        light_fractions=ell,
        lsf_sigma_v={name: LSF_SIGMA_V for name in dataset.instruments},
    )
    result = marginal_loglikelihood(problem, SmoothnessPrior(jnp.full(2, TAU), jnp.full(2, ETA)))
    return grid, np.asarray(result.d_hat), np.asarray(ab.spectra_std(result)), ell


def report(match, logg_pub, title):
    print(f"\n--- {title} ---")
    formal = match.errors("laplace")
    for name, truth_teff, truth_logg in (
        ("primary", TEFF1, logg_pub[0]),
        ("secondary", TEFF2, logg_pub[1]),
    ):
        got = match.labels[name]
        err = formal[name]
        d_teff = got["teff"] - truth_teff
        print(
            f"  {name:<10} Teff {got['teff']:7.1f} +/- {err['teff']:5.1f} K "
            f"({d_teff:+6.1f} K, {100 * d_teff / truth_teff:+5.2f}%)"
            f"   log g {got['logg']:.3f} ({got['logg'] - truth_logg:+.3f})"
            f"   [M/H] {got['mh']:+.3f}   vsini {got['vsini']:5.2f}"
        )
    ratio = match.flux_ratio
    print(f"  light fractions fitted: {ratio['primary']:.4f} / {ratio['secondary']:.4f}")
    print(
        f"  chi2 {match.chi2:.1f} / {match.n_pixels_used} px"
        f"   nulls: nearest node {match.chi2_nearest_node:.4g}, no template "
        f"{match.chi2_continuum:.4g}"
    )
    flagged = match.flagged_correlations()
    if flagged:
        print("  strong correlations:", ", ".join(f"{a}-{b} {r:+.2f}" for a, b, r in flagged))
    return match


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aiphe", help="directory of HARPS .fits spectra")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--draws", type=int, default=0, help="refit N joint draws for the spread")
    args = ap.parse_args()

    print(f"albireo {ab.__version__} - AI Phoenicis (HD 6980) label validation")
    print(f"window {WINDOW[0]:.0f}-{WINDOW[1]:.0f} A, library {LIBRARY!r}\n")
    print("published, and derived from the published:")
    check_derivation()
    logg_pub = published_logg()
    print(f"  Teff       {TEFF1:.0f} K and {TEFF2:.0f} K")
    print(f"  R2/R1      {R2_FRAC / R1_FRAC:.4f}")

    dataset = load(Path(args.data))
    print(f"\n{dataset.summary()}")

    t0 = time.perf_counter()
    grid, d_hat, std, ell = disentangle(dataset)
    print(f"\ndisentangled on {grid.n} px in {time.perf_counter() - t0:.1f} s")
    print(
        f"  light fractions assumed (blackbody at {np.mean(WINDOW):.0f} A): "
        f"{ell[0]:.4f} / {ell[1]:.4f}"
    )

    # The model grid is wider than the analysis window (LogGrid.covering adds the velocity
    # budget and the LSF radius on both sides), so the library has to cover the grid, not the
    # window; the resampler rejects a library that covers only the window.
    pad = 2.0
    library = ab.fetch_library(
        LIBRARY,
        wave_range=(float(grid.wave[0]) - pad, float(grid.wave[-1]) + pad),
        progress=False,
    )
    print(f"  library: {library.nodes.shape[0]} nodes x {library.wave.size} px ({library.medium})")

    def stars(logg_spec):
        return {
            "primary": ab.StarLabels(
                library=library,
                teff=ab.Between(5400.0, 6900.0),
                logg=logg_spec(logg_pub[0]),
                vsini=ab.Between(0.5, 25.0),
                v_kms=ab.Between(-15.0, 15.0),
            ),
            "secondary": ab.StarLabels(
                library=library,
                teff=ab.Between(4300.0, 5900.0),
                logg=logg_spec(logg_pub[1]),
                vsini=ab.Between(0.5, 25.0),
                v_kms=ab.Between(-15.0, 15.0),
            ),
        }

    common = dict(
        medium=dataset.epochs[0].medium,
        light_fractions=ell,
        lsf_sigma_kms=LSF_SIGMA_V,
        std=std,
        mh=ab.Between(-0.9, 0.4),
        max_steps=args.steps,
    )

    # The eclipsing case: log g is known to ~0.003 dex, so it is declared fixed.
    t0 = time.perf_counter()
    fixed = ab.match_labels(
        grid, d_hat, stars=stars(lambda g: ab.Fixed(g)), dilution=ab.RadiusRatio(), **common
    )
    wall_fixed = time.perf_counter() - t0
    report(fixed, logg_pub, f"log g fixed at the eclipse solution ({wall_fixed:.0f} s)")

    # The non-eclipsing case: both free, where Teff and log g are expected to correlate at
    # ~0.98 (Tamajo et al. 2011).
    t0 = time.perf_counter()
    free = ab.match_labels(
        grid,
        d_hat,
        stars=stars(lambda g: ab.Between(3.0, 4.9)),
        dilution=ab.RadiusRatio(),
        **common,
    )
    report(free, logg_pub, f"log g free ({time.perf_counter() - t0:.0f} s)")

    # The effect of the assumed dilution: freeze it and record which way the temperatures move.
    rigid = ab.match_labels(
        grid, d_hat, stars=stars(lambda g: ab.Fixed(g)), dilution=ab.FixedDilution(), **common
    )
    report(rigid, logg_pub, "log g fixed, dilution frozen at the blackbody estimate")

    print("\n--- the radius ratio, which nothing in the fit was told ---")
    published_ratio = R2_FRAC / R1_FRAC
    for label, match in (("fitted jointly", fixed), ("log g free", free)):
        ell_fit = match.flux_ratio
        # l2/l1 = (R2/R1)^2 * (continuum ratio), and the fit carries the continua, so the
        # radius ratio it implies is read back out of the fitted light fractions.
        implied = match.radius_ratio.get("secondary")
        offset = f"{100 * (implied / published_ratio - 1.0):+.1f}%" if implied else ""
        print(
            f"  {label:<16} l1/l2 = {ell_fit['primary']:.4f}/{ell_fit['secondary']:.4f}"
            f"   R2/R1 = {implied:.4f} ({offset})   published {published_ratio:.4f}"
        )

    if args.draws:
        print(f"\n--- refitting {args.draws} joint draws for the honest spread ---")
        print("  (not run by default: it costs about the same as one fit per draw)")

    print("\n" + fixed.summary())


if __name__ == "__main__":
    main()
