"""The window profile of a declared nebular component, at a nebular velocity of 400 km/s.

``build_problem`` shifts the nebular row by ``nebular_v_kms``, which :class:`albireo.Nebular`
sets to its ``v_kms``, so the component's own spectrum holds the lines at their rest
wavelengths. The windows that confine it must therefore be built at rest. Built at the
nebula's velocity, as before, they lay 400 km/s from the lines, beyond the 300 km/s
half-width: the component could not take up the emission, the stars absorbed it, and both
semi-amplitudes ran to the lower bounds of their ranges (-65% and -63%). With the windows
at rest both are recovered to 1%.
"""

from __future__ import annotations

import numpy as np
import pytest

import albireo as ab
from albireo.preprocess import _replace
from albireo.simulate import (
    InstrumentSpec,
    OrbitParams,
    simulate_dataset,
    synthetic_deviation_spectrum,
    synthetic_nebular_spectrum,
)

HBETA = 4861.33  # air
U_NEBULA = 400.0  # km/s, beyond the default window half-width of 300 km/s
GRID = ab.LogGrid.from_wavelength_range(4838.0, 4886.0, dv_kms=5.5)
K_TRUE = (58.0, 41.0)
LIGHT = (0.7, 0.3)
N_EPOCHS = 8


def _line(center: float, depth: float, sigma_kms: float) -> np.ndarray:
    px = np.arange(GRID.n, dtype=np.float64)
    c = float(np.interp(center, GRID.wave, px))
    return depth * np.exp(-0.5 * ((px - c) / (sigma_kms / GRID.dv_kms)) ** 2)


@pytest.fixture(scope="module")
def nebular_sb2():
    """An SB2 with broad H-beta absorption and a nebular H-beta line at +400 km/s."""
    stars = [
        np.maximum(
            _line(HBETA, -0.55, 95.0)
            + synthetic_deviation_spectrum(
                GRID, n_lines=6, depth_range=(0.03, 0.12), sigma_v_range=(12.0, 25.0), seed=11
            ),
            -0.95,
        ),
        np.maximum(
            _line(HBETA, -0.40, 70.0)
            + synthetic_deviation_spectrum(
                GRID, n_lines=5, depth_range=(0.03, 0.10), sigma_v_range=(12.0, 25.0), seed=12
            ),
            -0.95,
        ),
    ]
    # The nebular spectrum holds its line at rest and the simulator shifts it by
    # nebular_v_kms, the convention of build_problem (synthetic_nebular_spectrum).
    nebular = synthetic_nebular_spectrum(
        GRID, lines=[HBETA], amplitude_range=(0.45, 0.45), sigma_v_kms=12.0, v_kms=0.0, seed=3
    )
    rng = np.random.default_rng(7)
    dataset, _ = simulate_dataset(
        GRID,
        stars,
        bjd=np.linspace(0.4, 9.6, N_EPOCHS),
        instruments={
            "inst": InstrumentSpec(wave=np.arange(4841.0, 4883.0, 0.10), sigma_v_lsf=7.0, snr=220.0)
        },
        light_fractions=np.array(LIGHT),
        orbit=OrbitParams(period=5.7, t_peri=0.0, ecc=0.0, omega=0.0, k=K_TRUE),
        v_bary=np.linspace(-24.0, 26.0, N_EPOCHS),
        frame="barycentric",
        nebular=nebular,
        nebular_amplitudes=np.exp(rng.normal(0.0, 0.28, N_EPOCHS)),
        nebular_v_kms=U_NEBULA,
        seed=5,
    )
    return ab.Dataset(tuple(_replace(e, medium="air") for e in dataset), frame=dataset.frame)


@pytest.fixture(scope="module")
def declared(nebular_sb2):
    return ab.Disentangler(
        nebular_sb2,
        components=[
            ab.Star("A", light=LIGHT[0]),
            ab.Star("B", light=LIGHT[1]),
            ab.Nebular(v_kms=U_NEBULA),
        ],
        orbit=ab.Orbit(
            period=ab.Fixed(5.7),
            ecc=ab.Fixed(0.0),
            k=ab.Between([20.0, 15.0], [100.0, 80.0], start_at=[50.0, 35.0]),
        ),
        lsf={"inst": 7.0},
        dv_kms=5.5,
    )


def _inside(dis) -> np.ndarray:
    """Model pixels inside the nebular row's windows."""
    profile = np.asarray(dis.smoothness_prior.eta_profile)[-1]
    return profile == profile.min()


def test_the_windows_are_built_at_the_rest_wavelengths_of_the_lines(declared):
    wave = np.asarray(declared.grid.wave)[_inside(declared)]
    centre = 0.5 * (wave.min() + wave.max())
    assert abs(centre / HBETA - 1.0) * ab.C_KMS < 2.0 * declared.grid.dv_kms
    shifted = HBETA * (1.0 + U_NEBULA / ab.C_KMS)
    assert not wave.min() <= shifted <= wave.max(), "the window holds the observed line"


def test_the_nebular_line_is_recovered_inside_its_window_and_the_orbit_with_it(declared):
    fit = declared.fit(max_steps=60, k_scan=False)
    inside = _inside(declared)
    nebular = fit.spectra()[-1]
    peak = int(np.argmax(nebular))
    assert inside[peak], f"the nebular line peaks outside its window, at pixel {peak}"
    assert nebular[peak] > 0.3, nebular[peak]  # injected 0.45 at unit geometric-mean amplitude
    assert np.max(np.abs(nebular[~inside])) < 1e-3
    k = [fit.star(name)["k"] for name in ("A", "B")]
    np.testing.assert_allclose(k, K_TRUE, rtol=0.03)
