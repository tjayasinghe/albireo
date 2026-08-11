"""Closed-loop tests for the simulator (M1).

The simulator is the oracle for all downstream inference tests, so these tests pin down
its physics: noise consistent with ivar, masks/gaps/cosmics behaving as advertised,
telluric frame behavior, end-to-end velocity sign conventions, response application, and
mixed-instrument support.
"""

import numpy as np
import pytest

import albireo as ab
from albireo.kepler import radial_velocity
from albireo.simulate import (
    InstrumentSpec,
    OrbitParams,
    chebyshev_response,
    simulate_dataset,
    synthetic_deviation_spectrum,
    synthetic_telluric_spectrum,
)

GRID = ab.LogGrid.from_wavelength_range(4500.0, 4600.0, dv_kms=2.0)

ORBIT = OrbitParams(period=7.7, t_peri=1.2, ecc=0.3, omega=0.8, k=(60.0, 95.0), gamma=0.0)
BJD = np.linspace(0.0, 7.0, 6)


def make_instrument(dlam=0.03, lo=4510.0, hi=4590.0, sigma_v=4.0, snr=100.0):
    return InstrumentSpec(wave=np.arange(lo, hi, dlam), sigma_v_lsf=sigma_v, snr=snr)


def two_components():
    return [
        synthetic_deviation_spectrum(GRID, seed=1),
        synthetic_deviation_spectrum(GRID, seed=2),
    ]


def single_line_component(grid, center_wave, depth=0.6, sigma_v=8.0):
    center_px = (np.log(center_wave) - grid.x0) / grid.dx
    px = np.arange(grid.n, dtype=np.float64)
    return -depth * np.exp(-0.5 * ((px - center_px) / (sigma_v / grid.dv_kms)) ** 2)


def measure_line_velocity(wave, flux, lam0):
    """Line-center velocity from a 3-point parabolic fit around the flux minimum."""
    i = int(np.argmin(flux))
    x = np.log(wave[i - 1 : i + 2])
    y = flux[i - 1 : i + 2]
    h = 0.5 * (x[2] - x[0])
    xc = x[1] + 0.5 * h * (y[0] - y[2]) / (y[0] - 2 * y[1] + y[2])
    return ab.C_KMS * np.tanh(xc - np.log(lam0))


def simulate_basic(**kwargs):
    defaults = dict(
        bjd=BJD,
        orbit=ORBIT,
        light_fractions=[0.65, 0.35],
        instruments={"A": make_instrument()},
        v_bary=np.zeros(len(BJD)),
        seed=7,
    )
    defaults.update(kwargs)
    return simulate_dataset(GRID, two_components(), **defaults)


def test_reproducible_and_seed_sensitive():
    ds1, _ = simulate_basic()
    ds2, _ = simulate_basic()
    ds3, _ = simulate_basic(seed=8)
    np.testing.assert_array_equal(ds1[0].flux, ds2[0].flux)
    np.testing.assert_array_equal(ds1[0].ivar, ds2[0].ivar)
    assert np.max(np.abs(ds1[0].flux - ds3[0].flux)) > 0


def test_noise_is_consistent_with_ivar():
    ds, truth = simulate_basic()
    z = np.concatenate(
        [(ep.flux - nl) * np.sqrt(ep.ivar) for ep, nl in zip(ds, truth.noiseless_flux, strict=True)]
    )
    n = z.size
    assert abs(z.mean()) < 5 / np.sqrt(n)
    assert abs(z.var() - 1.0) < 5 * np.sqrt(2.0 / n)


def test_gaps_and_cosmics_are_masked_and_corrupted():
    ds, truth = simulate_basic(gap_fraction=0.08, cosmic_fraction=0.005)
    for ep, noiseless in zip(ds, truth.noiseless_flux, strict=True):
        n = ep.n_pixels
        bad = ~ep.good
        assert bad.sum() >= int(0.08 * n)
        # one contiguous gap of at least ~the requested width
        runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], bad.view(np.int8), [0]]))))
        assert runs.max() >= int(0.9 * 0.08 * n)
        # corrupted pixels carry garbage, so downstream code must honor the mask
        assert np.max(np.abs(ep.flux[bad] - noiseless[bad])) > 1.0
        # effective_ivar folds the mask in
        assert np.all(ep.effective_ivar[bad] == 0.0)


def test_telluric_static_in_topocentric_frame():
    tell = synthetic_telluric_spectrum(GRID, seed=5)
    kwargs = dict(
        bjd=BJD[:4],
        velocities=np.zeros((1, 4)),
        light_fractions=[1.0],
        instruments={"A": make_instrument(snr=1e9)},
        v_bary=np.linspace(-30.0, 30.0, 4),
        telluric=tell,
        seed=3,
    )
    ds_topo, _ = simulate_dataset(GRID, [np.zeros(GRID.n)], frame="topocentric", **kwargs)
    ref = ds_topo[0].flux
    for ep in ds_topo:
        np.testing.assert_allclose(ep.flux, ref, atol=1e-8)

    ds_bary, _ = simulate_dataset(GRID, [np.zeros(GRID.n)], frame="barycentric", **kwargs)
    # in the barycentric frame the tellurics move epoch-to-epoch
    assert np.max(np.abs(ds_bary[0].flux - ds_bary[-1].flux)) > 0.01


def test_stellar_line_velocity_end_to_end():
    lam0 = 4550.0
    comp = single_line_component(GRID, lam0)
    v_star = np.array([50.0, -80.0, 0.0])
    v_bary = np.array([10.0, -25.0, 0.0])
    kwargs = dict(
        bjd=np.arange(3.0),
        velocities=v_star[None, :],
        light_fractions=[1.0],
        instruments={"A": make_instrument(sigma_v=2.0, snr=1e8)},
        v_bary=v_bary,
        seed=4,
    )
    ds_topo, _ = simulate_dataset(GRID, [comp], frame="topocentric", **kwargs)
    beta_s, beta_b = v_star / ab.C_KMS, v_bary / ab.C_KMS
    expected_topo = ab.C_KMS * np.tanh(np.arctanh(beta_s) - np.arctanh(beta_b))
    for ep, v_exp in zip(ds_topo, expected_topo, strict=True):
        assert abs(measure_line_velocity(ep.wave, ep.flux, lam0) - v_exp) < 0.5

    ds_bary, _ = simulate_dataset(GRID, [comp], frame="barycentric", **kwargs)
    for ep, v_exp in zip(ds_bary, v_star, strict=True):
        assert abs(measure_line_velocity(ep.wave, ep.flux, lam0) - v_exp) < 0.5


def test_truth_velocities_match_kepler():
    _, truth = simulate_basic()
    import jax.numpy as jnp

    v1 = np.asarray(
        radial_velocity(
            jnp.asarray(BJD),
            period=ORBIT.period,
            t_peri=ORBIT.t_peri,
            ecc=ORBIT.ecc,
            omega=ORBIT.omega,
            k=ORBIT.k[0],
            gamma=ORBIT.gamma,
        )
    )
    np.testing.assert_allclose(truth.velocities[0], v1, rtol=1e-12)
    # SB2 anti-phase: (v1 - gamma) K2 = -(v2 - gamma) K1
    np.testing.assert_allclose(
        (truth.velocities[0] - ORBIT.gamma) * ORBIT.k[1],
        -(truth.velocities[1] - ORBIT.gamma) * ORBIT.k[0],
        rtol=1e-10,
    )


def test_response_polynomial_applied():
    ds, truth = simulate_dataset(
        GRID,
        [np.zeros(GRID.n)],
        bjd=np.arange(3.0),
        velocities=np.zeros((1, 3)),
        light_fractions=[1.0],
        instruments={"A": make_instrument(snr=1e9)},
        v_bary=np.zeros(3),
        response_order=2,
        response_amplitude=0.05,
        seed=11,
    )
    for ep, coeffs in zip(ds, truth.response_coeffs, strict=True):
        assert coeffs.shape == (3,)
        np.testing.assert_allclose(ep.flux, chebyshev_response(ep.wave, coeffs), atol=1e-8)


def test_mixed_instruments_and_resolutions():
    lam0 = 4550.0
    comp = single_line_component(GRID, lam0, sigma_v=6.0)
    instruments = {
        "HI": make_instrument(dlam=0.03, sigma_v=3.0, snr=1e8),
        "LO": make_instrument(dlam=0.09, sigma_v=15.0, snr=1e8),
    }
    ds, _ = simulate_dataset(
        GRID,
        [comp],
        bjd=np.arange(2.0),
        velocities=np.zeros((1, 2)),
        light_fractions=[1.0],
        instruments=instruments,
        epoch_instruments=["HI", "LO"],
        v_bary=np.zeros(2),
        seed=6,
    )
    assert ds[0].n_pixels != ds[1].n_pixels
    assert ds.instruments == ("HI", "LO")
    # the low-resolution epoch's line is shallower (LSF-broadened)
    depth_hi = 1.0 - ds[0].flux.min()
    depth_lo = 1.0 - ds[1].flux.min()
    assert depth_hi > depth_lo * 1.3


def test_per_epoch_light_fractions():
    ell = np.column_stack([[0.65, 0.35]] * 5 + [[1.0, 0.0]]).astype(float)
    ds_ecl, _ = simulate_basic(light_fractions=ell)
    ds_const, _ = simulate_basic()
    # non-eclipse epochs identical, eclipse epoch differs
    np.testing.assert_array_equal(ds_ecl[0].flux, ds_const[0].flux)
    assert np.max(np.abs(ds_ecl[5].flux - ds_const[5].flux)) > 0.01


def test_validation_errors():
    with pytest.raises(ValueError, match="sum to 1"):
        simulate_basic(light_fractions=[0.7, 0.7])
    with pytest.raises(ValueError, match="exactly one"):
        simulate_basic(velocities=np.zeros((2, len(BJD))))
    with pytest.raises(ValueError, match="exactly one"):
        simulate_basic(orbit=None)
    with pytest.raises(ValueError, match="beyond the model grid"):
        simulate_basic(instruments={"A": make_instrument(lo=4450.0)})
    with pytest.raises(ValueError, match="semi-amplitudes"):
        simulate_dataset(
            GRID,
            [np.zeros(GRID.n)],
            bjd=BJD,
            orbit=ORBIT,
            light_fractions=[1.0],
            instruments={"A": make_instrument()},
        )
