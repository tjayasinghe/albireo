"""Gaia RVS spectra of a synthetic double-lined binary, and what albireo makes of them.

This is the reference implementation's example (Rowan's ``synthetic_rvs_spectra.ipynb``,
``SyntheticSB2(**GAIA_RVS, ...)``) rebuilt on albireo: the same two stars, the same orbit,
the same S/N of 40 per detector pixel, the same ten epochs evenly spaced over one period,
and the same delivered 0.01 nm grid with its correlated noise. Where the notebook
synthesises the components with Korg, this example renders them from the BOSZ 2024 grid in
the RVS band (``fetch_library("bosz2024-fgk-rvs")``; 621 MB downloaded once, 5 MB cached),
because albireo synthesises nothing.

Four results to check in the output.

1. The semi-amplitudes from the masses: K1 = 87.7 and K2 = 95.0 km/s, as the notebook
   prints, and the velocities at the epoch of largest separation (76.05 / -119.88 km/s).
2. TODCOR at that epoch against the library templates. The notebook's two-dimensional
   correlation gives 79.69 / -122.08 km/s with its own templates; the one-dimensional CCF
   against the primary alone gives 73.93 / -117.32. Both carry the blend bias the
   two-template fit removes (``docs/math.md`` §10); albireo's velocities should land
   within their errors of the truth.
3. The disentangling of the ten epochs, with the period held at the photometric value and
   everything else free: K1 and K2 back to about a percent at this S/N.
4. The delivery record: 2.45 delivered pixels per detector pixel and a lag-one noise
   correlation above 0.5, which the diagonal ``flux_error`` of the archive product does
   not express. The DR4 epoch grid (``--dr4``) has a weaker, periodic correlation.

    python examples/14_gaia_rvs.py            # the notebook's DR3-shaped product
    python examples/14_gaia_rvs.py --dr4      # the DR4 epoch grid instead

Environment
-----------
``ALBIREO_EXAMPLE_FAST=1`` trims the optimizer budget for CI. The library must be cached
(``fetch_library("bosz2024-fgk-rvs")``, once; the pipeline fetches it on first use).

References
----------
Cropper, M., Katz, D., Sartoretti, P., et al. 2018, A&A, 616, A5
Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import albireo as ab
from albireo.gaia import (
    RVS_DR3_MEAN,
    RVS_DR4_EPOCH,
    rvs_components,
    rvs_model_grid,
    simulate_rvs_dataset,
    uniform_phase_times,
)
from albireo.simulate import OrbitParams

FAST = bool(os.environ.get("ALBIREO_EXAMPLE_FAST"))

# ---- the notebook's parameters ---------------------------------------------------------
PRIMARY = {"teff": 6100.0, "logg": 4.2, "mh": -0.1}
SECONDARY = {"teff": 5700.0, "logg": 4.3, "mh": -0.1}
VSINI = (12.0, 11.0)
FLUX_RATIO = 0.60  # F_secondary / F_primary
M1, M2 = 1.30, 1.20  # solar masses
PERIOD, ECC, OMEGA, PHI0, INCL, GAMMA = 4.0, 0.10, 0.7, 1.2, np.radians(87.0), -18.0
SNR, N_EPOCHS, SEED = 40.0, 10, 42
V_SEARCH = 250.0  # km/s, the correlation search half-range


def semi_amplitudes(m1, m2, period, ecc, incl):
    """K1, K2 in km/s from the masses, as the notebook computes them (Kepler's third law)."""
    g_si, msun, day = 6.67430e-11, 1.988409870698051e30, 86400.0
    total = (m1 + m2) * msun
    coeff = (2 * np.pi * g_si / (period * day)) ** (1 / 3) / np.sqrt(1 - ecc**2) * np.sin(incl)
    return coeff * m2 * msun / total ** (2 / 3) / 1e3, coeff * m1 * msun / total ** (2 / 3) / 1e3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dr4", action="store_true", help="deliver the DR4 epoch grid")
    args = parser.parse_args(argv)
    product = RVS_DR4_EPOCH if args.dr4 else RVS_DR3_MEAN

    k1, k2 = semi_amplitudes(M1, M2, PERIOD, ECC, INCL)
    print(f"K1 = {k1:.1f} km/s, K2 = {k2:.1f} km/s   (notebook: 87.7, 95.0)")
    orbit = OrbitParams(
        period=PERIOD,
        t_peri=PHI0 * PERIOD / (2 * np.pi),
        ecc=ECC,
        omega=OMEGA,
        k=(k1, k2),
        gamma=GAMMA,
    )
    light = (1.0 / (1.0 + FLUX_RATIO), FLUX_RATIO / (1.0 + FLUX_RATIO))

    # ---- components from the library, on a grid that covers the shifted band ----------
    library = ab.fetch_library("bosz2024-fgk-rvs")
    # The margin covers the widest correlation search below (+-V_SEARCH), not only the orbit.
    grid = rvs_model_grid(V_SEARCH, vsini_max_kms=max(VSINI))
    t0 = time.perf_counter()
    components = rvs_components(library, [PRIMARY, SECONDARY], grid, vsini_kms=VSINI)
    print(f"components rendered on {grid.n} px in {time.perf_counter() - t0:.1f} s")

    # ---- the epochs -------------------------------------------------------------------
    bjd = uniform_phase_times(PERIOD, N_EPOCHS, start=2457000.0)
    dataset, truth = simulate_rvs_dataset(
        components,
        grid,
        bjd=bjd,
        light_fractions=light,
        orbit=orbit,
        snr=SNR,
        product=product,
        library_resolving_power=float(library.meta["resolution"]),
        seed=SEED,
    )
    print(
        f"{dataset.n_epochs} spectra, {dataset[0].n_pixels} pixels each ({product.description});"
        f" delivered {truth.delivery.pixel_ratio:.2f} px per detector px,"
        f" lag-1 noise correlation {np.mean(truth.delivery.lag1):.2f}"
    )
    idx = int(np.argmax(np.abs(truth.velocities[0] - truth.velocities[1])))
    v1, v2 = truth.velocities[:, idx]
    print(f"epoch {idx}: RV1 = {v1:.2f}, RV2 = {v2:.2f} km/s (notebook: 76.05, -119.88)")

    # ---- TODCOR at that epoch, against library templates ------------------------------
    resolving = float(library.meta["resolution"])
    templates = [
        ab.Template.from_library(
            name,
            library,
            labels,
            grid=grid,
            medium="vacuum",
            vsini_kms=v,
            resolving_power=resolving,
        )
        for name, labels, v in zip(
            ("primary", "secondary"), (PRIMARY, SECONDARY), VSINI, strict=True
        )
    ]
    one_epoch = ab.Dataset([dataset[idx]], frame="barycentric")
    lsf = {"RVS": ab.LSF.from_resolution(11_500).sigma_kms}
    two_d = ab.todcor(
        one_epoch, templates, v_range=(-V_SEARCH, V_SEARCH), light=list(light), lsf_sigma_v=lsf
    )
    one_d = ab.todcor(
        one_epoch, templates[:1], v_range=(-V_SEARCH, V_SEARCH), light=[1.0], lsf_sigma_v=lsf
    )
    print(
        f"TODCOR  RV1 = {two_d.velocity[0, 0]:7.2f} +/- {two_d.sigma[0, 0]:.2f}   "
        f"RV2 = {two_d.velocity[1, 0]:7.2f} +/- {two_d.sigma[1, 0]:.2f}   "
        "(notebook 2-D: 79.69 / -122.08)"
    )
    print(
        f"1-D CCF RV1 = {one_d.velocity[0, 0]:7.2f} +/- {one_d.sigma[0, 0]:.2f}"
        "                            (notebook 1-D: 73.93; a blend-biased estimate)"
    )

    # ---- the disentangling of the ten epochs ------------------------------------------
    dis = ab.Disentangler(
        dataset,
        components=[ab.Star("primary", light[0]), ab.Star("secondary", light[1])],
        orbit=ab.Orbit(
            period=ab.Known(PERIOD, 1e-3 * PERIOD),
            k=ab.Between([20.0, 20.0], [160.0, 160.0], start_at=[70.0, 110.0]),
            ecc=ab.Between(0.0, 0.5),
        ),
        lsf={"RVS": ab.LSF.from_resolution(11_500)},
        dv_kms=3.0,
        # What the delivery did to the noise, declared like the LSF: AR(1) along the pixel
        # index at the recorded lag-one correlation, which the archive errors do not carry.
        noise_correlation={"RVS": float(np.mean(truth.delivery.lag1))},
    )
    t0 = time.perf_counter()
    fit = dis.fit(max_steps=60 if FAST else 250)
    kf = fit.orbit()
    print(
        f"disentangled in {time.perf_counter() - t0:.0f} s: K1 {fit.star('primary')['k']:.2f}, "
        f"K2 {fit.star('secondary')['k']:.2f} km/s, e {float(kf['ecc']):.3f}, z_rms {fit.z_rms:.3f}"
    )
    table = fit.measure_velocities()
    residual = table.velocity - truth.velocities
    residual = residual - np.nanmean(residual, axis=1, keepdims=True)  # each zero point
    rms = np.sqrt(np.nanmean(residual**2, axis=1))
    print(
        "epoch velocities against the disentangled components: rms "
        f"{rms[0]:.2f} / {rms[1]:.2f} km/s (after removing each component's zero point)"
    )

    # ---- the gate ---------------------------------------------------------------------
    assert abs(k1 - 87.7) < 0.05 and abs(k2 - 95.0) < 0.05
    assert abs(two_d.velocity[0, 0] - v1) < 3.0 * two_d.sigma[0, 0] + 1.0
    assert abs(two_d.velocity[1, 0] - v2) < 3.0 * two_d.sigma[1, 0] + 1.0
    tol = 0.05 if FAST else 0.02
    assert abs(fit.star("primary")["k"] - k1) < tol * k1
    assert abs(fit.star("secondary")["k"] - k2) < tol * k2
    print("\nOK - the notebook's system reproduced on albireo, end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
