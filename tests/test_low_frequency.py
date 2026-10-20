"""Continuum errors that differ between epochs, and the broad structure of the components.

The data determine the light-weighted sum of the components at every scale, but the
division of its broad structure between the stars only through the small Doppler shifts
between epochs. The continuum weight ``eta`` of each component is the only term of the
prior that holds that division. When the continuum normalization of the epochs differs
from epoch to epoch, as the residual blaze of an echelle spectrograph does, ML-II lowers
``eta`` by orders of magnitude and the components take up the errors as drifts of a
continuum unit in opposite directions (``docs/method.md``, section 10). This was found on
PEPSI spectra of a double-lined binary, where ``eta`` fell from 5 to 0.015 and 0.0014.

The binary below is simulated with a ripple of 2.5% rms in every epoch, a Chebyshev
response of order 12 over 200 A (structure of 40-80 A), which the model does not contain.
The tests check that the fit reports the fall of ``eta``, that a declared bound holds, and
that renormalizing the epochs against the model restores a fit like one to clean data.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

import albireo as ab
from albireo.grids import LogGrid
from albireo.simulate import (
    InstrumentSpec,
    OrbitParams,
    simulate_dataset,
    synthetic_deviation_spectrum,
)

LO, HI = 4560.0, 4760.0
LIGHT = (0.75, 0.25)
K_TRUE = (42.0, 63.0)
FELL = "eta fell more than one hyperprior standard deviation"


@pytest.fixture(scope="module")
def rippled():
    grid = LogGrid.from_wavelength_range(LO, HI, dv_kms=6.0)
    components = [
        synthetic_deviation_spectrum(
            grid, n_lines=52, seed=seed, margin=0.03, depth_range=(0.05, 0.6)
        )
        for seed in (21, 22)
    ]
    orbit = OrbitParams(period=6.0, t_peri=0.0, ecc=0.15, omega=0.7, k=K_TRUE)
    dataset, truth = simulate_dataset(
        grid,
        components,
        bjd=np.linspace(0.0, 2.7 * 6.0, 12, endpoint=False),
        instruments={
            "DEMO": InstrumentSpec(
                wave=np.arange(LO + 10.0, HI - 10.0, 0.08), sigma_v_lsf=6.5, snr=100.0
            )
        },
        light_fractions=LIGHT,
        orbit=orbit,
        response_order=12,
        response_amplitude=0.008,
        seed=7,
    )
    return dataset, truth


def _declare(dataset, smoothness=None):
    smoothness = smoothness or ab.Smoothness()
    return ab.Disentangler(
        dataset,
        components=[
            ab.Star("A", light=LIGHT[0], smoothness=smoothness),
            ab.Star("B", light=LIGHT[1], smoothness=smoothness),
        ],
        orbit=ab.Orbit(period=ab.Between(5.95, 6.05), k=ab.Between([20.0, 30.0], [70.0, 95.0])),
        lsf={"DEMO": 6.5},
    )


def _drifts(fit, truth):
    """RMS of the components' errors on scales above 10 A, and of their weighted sum."""
    wave = np.asarray(fit.dis.grid.wave)
    injected = np.array([np.interp(wave, np.asarray(truth.grid.wave), c) for c in truth.components])
    width = 10.0 / float(np.median(np.diff(wave)))
    inside = (wave > LO + 15.0) & (wave < HI - 15.0)
    error = fit.spectra()[:2] - injected
    low = np.array([gaussian_filter1d(e, width) for e in error])
    composite = LIGHT[0] * low[0] + LIGHT[1] * low[1]
    rms = np.sqrt(np.mean(low[:, inside] ** 2, axis=1))
    return rms, float(np.sqrt(np.mean(composite[inside] ** 2)))


@pytest.fixture(scope="module")
def ml2_fit(rippled):
    dataset, _ = rippled
    return _declare(dataset).fit(max_steps=150)


@pytest.mark.slow
def test_continuum_errors_lower_eta_and_the_summary_reports_it(rippled, ml2_fit):
    _, truth = rippled
    summary = ml2_fit.summary()
    assert summary.count(FELL) == 2, summary
    assert all(ml2_fit.hyper[name]["eta"] < 0.5 for name in "AB")
    drifts, composite = _drifts(ml2_fit, truth)
    # The fainter component drifts by most of a continuum unit; the sum does not.
    assert drifts[1] > 0.5 and composite < 0.02


@pytest.mark.slow
def test_renormalized_epochs_give_the_fit_of_clean_data(rippled, ml2_fit):
    _, truth = rippled
    renorm = ml2_fit.renormalize_epochs()
    assert renorm.rms.shape == (3, 12)
    first, last = np.median(renorm.rms[0]), np.median(renorm.rms[-1])
    assert first > 0.01 and last < 0.15 * first
    assert "pass 3" in renorm.summary()
    # Only the flux and the inverse variance change.
    original, corrected = ml2_fit.dis.dataset[3], renorm.dataset[3]
    np.testing.assert_array_equal(corrected.wave, original.wave)
    np.testing.assert_allclose(corrected.flux * renorm.corrections[3], original.flux)
    np.testing.assert_allclose(corrected.ivar, original.ivar * renorm.corrections[3] ** 2)

    refit = _declare(renorm.dataset).fit(max_steps=150)
    summary = refit.summary()
    assert FELL not in summary, summary
    assert all(refit.hyper[name]["eta"] > 1.0 for name in "AB")
    drifts, _ = _drifts(refit, truth)
    assert np.all(drifts < 0.06), drifts
    assert 0.9 < refit.z_rms < 1.1
    np.testing.assert_allclose(np.asarray(refit.orbit()["k"]), K_TRUE, atol=0.2)


@pytest.mark.slow
def test_a_bound_on_eta_holds_and_reduces_the_drifts(rippled, ml2_fit):
    dataset, truth = rippled
    fit = _declare(dataset, ab.Smoothness(eta_min=5.0)).fit(max_steps=150)
    assert all(fit.hyper[name]["eta"] >= 5.0 for name in "AB")
    summary = fit.summary()
    assert "eta is at its declared bound 5" in summary and FELL not in summary
    bounded, _ = _drifts(fit, truth)
    free, _ = _drifts(ml2_fit, truth)
    assert bounded[1] < 0.5 * free[1]
