"""Closed-loop tests for the SB1 + faint-companion K2 scan (M4).

The detection statistic D(K2) = 2 [log p(y | K2) - log p(y | null)] is calibrated
empirically (docs/math.md §6) — these tests assert its *behavior*: a sharp peak at
the injected K2 with the companion spectrum recovered, and (crucially) D < 0
everywhere when no companion exists, because the marginal likelihood's Occam term
charges for the extra marginalized component and no coherent signal pays for it.

The companion's smooth envelope (mean blanketing) is prior-dominated at small
ell_2 — the ell_1/ell_2-amplified low-frequency degeneracy (math.md §5.1-5.2) —
so spectrum asserts are offset-removed; the line *pattern* is what the scan
recovers.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import albireo as ab

GRID = ab.LogGrid.from_wavelength_range(5000.0, 5045.0, dv_kms=5.5)
P, ECC, OMEGA, K1, K2_TRUE = 6.31, 0.15, 0.7, 12.0, 38.0
ELL = (0.9, 0.1)
K2_GRID = np.arange(10.0, 70.0, 4.0)  # K2_TRUE = 38.0 lands exactly on the grid
PRIOR = ab.SmoothnessPrior(jnp.asarray([300.0, 30.0]), jnp.asarray([5.0, 5.0]))


def _sb1_orbit():
    nu_c = 0.5 * np.pi - OMEGA
    e_c = 2.0 * np.arctan2(
        np.sqrt(1.0 - ECC) * np.sin(0.5 * nu_c), np.sqrt(1.0 + ECC) * np.cos(0.5 * nu_c)
    )
    t_conj = 2.0 + (e_c - ECC * np.sin(e_c)) * P / (2.0 * np.pi)
    return {
        "period": P,
        "t_conj": t_conj,
        "secosw": np.sqrt(ECC) * np.cos(OMEGA),
        "sesinw": np.sqrt(ECC) * np.sin(OMEGA),
    }


def _simulate(with_companion: bool):
    rng = np.random.default_rng(7)
    bjd = np.sort(rng.uniform(0.0, 21.0, size=10))
    primary = ab.synthetic_deviation_spectrum(GRID, seed=21)
    companion = ab.synthetic_deviation_spectrum(GRID, seed=22)
    inst = {
        "a": ab.InstrumentSpec(wave=np.arange(5003.0, 5042.0, 0.11), sigma_v_lsf=7.0, snr=150.0)
    }
    tperi = 2.0
    if with_companion:
        orbit = ab.OrbitParams(period=P, t_peri=tperi, ecc=ECC, omega=OMEGA, k=(K1, K2_TRUE))
        comps, ell = [primary, companion], ELL
    else:
        orbit = ab.OrbitParams(period=P, t_peri=tperi, ecc=ECC, omega=OMEGA, k=(K1,))
        comps, ell = [primary], (1.0,)
    ds, _ = ab.simulate_dataset(
        GRID, comps, bjd=bjd, instruments=inst, light_fractions=ell, orbit=orbit, seed=5
    )
    return ds, companion


def _scan(ds):
    return ab.k2_scan(
        GRID,
        ds,
        orbit=_sb1_orbit(),
        k1=K1,
        k2_grid=K2_GRID,
        light_fractions=ELL,
        lsf_sigma_v={"a": 7.0},
        prior=PRIOR,
        v_rel_max_kms=105.0,
    )


@pytest.fixture(scope="module")
def injected_scan():
    ds, companion = _simulate(with_companion=True)
    return _scan(ds), companion


def test_detection_peaks_at_injected_k2(injected_scan):
    result, _ = injected_scan
    assert result.k2_peak == K2_TRUE
    # decisive contrast against the scan edges (absolute D is calibration-dependent)
    edge = max(result.detection[0], result.detection[-1])
    assert result.detection_peak - edge > 1e3
    assert result.log_likelihood.shape == K2_GRID.shape
    np.testing.assert_allclose(
        result.detection, 2.0 * (result.log_likelihood - result.log_likelihood_null)
    )


def test_companion_spectrum_recovered_at_peak(injected_scan):
    result, companion = injected_scan
    edge = int(0.05 * GRID.n)
    mask = companion < -0.05
    mask[:edge] = False
    mask[-edge:] = False
    err = result.companion[mask] - companion[mask]
    offset = err.mean()
    # The smooth envelope is prior-set at ell_2 = 0.1 (ell_1/ell_2-amplified k~0
    # degeneracy); the line pattern is the recoverable content.
    assert abs(offset) < 0.3
    assert np.sqrt(np.mean((err - offset) ** 2)) < 0.1
    assert np.corrcoef(result.companion[mask], companion[mask])[0, 1] > 0.96


def test_null_dataset_yields_negative_detection():
    ds, _ = _simulate(with_companion=False)
    result = _scan(ds)
    # No companion: the Occam term makes every trial K2 lose to the null model.
    assert result.detection.max() < 0.0


def test_scan_validation_errors():
    ds, _ = _simulate(with_companion=False)
    orbit = _sb1_orbit()
    kwargs = dict(
        k1=K1,
        k2_grid=K2_GRID,
        light_fractions=ELL,
        lsf_sigma_v={"a": 7.0},
        prior=PRIOR,
        v_rel_max_kms=105.0,
    )
    with pytest.raises(ValueError, match="missing"):
        ab.k2_scan(GRID, ds, orbit={k: v for k, v in orbit.items() if k != "period"}, **kwargs)
    with pytest.raises(ValueError, match="k1 is a separate"):
        ab.k2_scan(GRID, ds, orbit={**orbit, "k": 12.0}, **kwargs)
    with pytest.raises(ValueError, match="positive"):
        ab.k2_scan(GRID, ds, orbit=orbit, **{**kwargs, "k2_grid": np.array([-5.0, 10.0])})
    with pytest.raises(ValueError, match="prior must have 2 components"):
        ab.k2_scan(
            GRID,
            ds,
            orbit=orbit,
            **{
                **kwargs,
                "prior": ab.SmoothnessPrior(jnp.asarray([300.0]), jnp.asarray([5.0])),
            },
        )
