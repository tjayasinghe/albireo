"""Tests for log-wavelength grids and the Doppler log-shift mapping."""

import numpy as np
import pytest

import albireo as ab


def test_x64_enabled_on_import():
    import jax.numpy as jnp

    assert jnp.zeros(3).dtype == jnp.float64


def test_relativistic_shift_is_exactly_antisymmetric():
    v = np.array([1.0, 37.3, 250.0, 613.2])
    plus = np.asarray(ab.log_doppler_shift(v))
    minus = np.asarray(ab.log_doppler_shift(-v))
    np.testing.assert_allclose(minus, -plus, rtol=1e-15, atol=0.0)


def test_classical_shift_is_not_antisymmetric():
    # ln(1+b) + ln(1-b) = ln(1-b^2) != 0: classical shifts do not compose/invert exactly.
    v = 600.0
    resid = float(
        ab.log_doppler_shift(v, relativistic=False) + ab.log_doppler_shift(-v, relativistic=False)
    )
    assert abs(resid) > 1e-7


def test_relativistic_matches_classical_at_low_velocity():
    v = 10.0  # km/s: difference is O(beta^2/2) ~ 6e-10 in xi
    rel = float(ab.log_doppler_shift(v))
    cla = float(ab.log_doppler_shift(v, relativistic=False))
    assert abs(rel - cla) < 1e-9
    assert rel == pytest.approx(v / ab.C_KMS, rel=1e-4)


def test_grid_construction_covers_range():
    grid = ab.LogGrid.from_wavelength_range(4000.0, 5000.0, dv_kms=1.5)
    wave = grid.wave
    assert wave[0] == pytest.approx(4000.0, rel=1e-14)
    assert wave[-1] >= 5000.0
    assert wave[-2] < 5000.0
    # uniform in log; the exp/log round-trip floor is ~|x|*eps/dx ~ 1e-10 relative
    np.testing.assert_allclose(np.diff(np.log(wave)), grid.dx, rtol=1e-9)
    np.testing.assert_allclose(np.log(wave), grid.x, rtol=1e-15)


def test_grid_pixel_velocity_roundtrip():
    grid = ab.LogGrid.from_wavelength_range(4000.0, 5000.0, dv_kms=1.5)
    assert grid.dv_kms == pytest.approx(1.5, rel=1e-12)
    # one pixel corresponds to exactly one pixel-velocity
    assert float(grid.velocity_to_pixels(grid.dv_kms)) == pytest.approx(1.0, rel=1e-12)
    # shifts compose exactly in pixel space (relativistic mapping)
    d1 = float(grid.velocity_to_pixels(123.4))
    d2 = float(grid.velocity_to_pixels(-123.4))
    assert d1 + d2 == pytest.approx(0.0, abs=1e-15)


def test_grid_validation():
    with pytest.raises(ValueError):
        ab.LogGrid.from_wavelength_range(5000.0, 4000.0, dv_kms=1.0)
    with pytest.raises(ValueError):
        ab.LogGrid.from_wavelength_range(4000.0, 5000.0, dv_kms=-1.0)
