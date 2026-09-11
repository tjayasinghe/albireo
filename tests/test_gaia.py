"""Tests for the Gaia RVS simulator (``albireo.gaia``).

Three kinds of claim. The photometric and signal-to-noise relations reproduce the reference
implementation's numbers (Rowan's ``synthetic_rvs_spectra.ipynb``). The delivery step does
to the noise exactly what it says: the propagated variance and the lag-one correlation are
what the delivered pixels show. And the one-call simulator declares what Gaia delivers
(vacuum, barycentric, the RVS width) and closes the loop through the disentangler.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path

import numpy as np
import pytest

import albireo as ab
from albireo import gaia as gaia_module
from albireo.gaia import (
    GOST_ENDPOINT,
    GOST_USABLE_FRACTION,
    RVS_CCD_ROWS,
    RVS_DR3_MEAN,
    RVS_DR4_EPOCH,
    RVS_SPANS,
    GostTransits,
    RVSProduct,
    deliver,
    gost_transits,
    predict_grvs,
    quadrature_sigma_kms,
    random_times,
    rvs_detector_grid,
    rvs_lsf_sigma_kms,
    rvs_model_grid,
    rvs_snr_per_pixel,
    rvs_transit_count,
    rvs_transit_resolving_powers,
    rvs_transit_times,
    rvs_transit_times_from_gost,
    simulate_rvs_dataset,
    uniform_phase_times,
)
from albireo.simulate import OrbitParams, synthetic_deviation_spectrum

# The notebook's system: masses 1.30 / 1.20 Msun, P = 4 d, e = 0.1, i = 87 deg give
# K1 = 87.7, K2 = 95.0 km/s; phi0 = 1.2 rad is t_peri = phi0 P / 2 pi.
NOTEBOOK_ORBIT = OrbitParams(
    period=4.0, t_peri=1.2 * 4.0 / (2 * np.pi), ecc=0.10, omega=0.7, k=(87.70, 95.00), gamma=-18.0
)
NOTEBOOK_LIGHT = (1.0 / 1.6, 0.6 / 1.6)  # flux ratio 0.60


# ---------------------------------------------------------------------------
# photometry and S/N
# ---------------------------------------------------------------------------


def test_grvs_prediction_reproduces_the_notebook():
    # Gaia DR3 1191216433749408128: G = 11.156782, G_RP = 10.719377 -> G_RVS = 10.54
    assert np.round(predict_grvs(11.156782, 10.719377), 2) == 10.54
    assert np.isnan(predict_grvs(10.0, 10.2))  # G - RP = -0.2, outside the calibration
    assert np.isnan(predict_grvs(12.0, 10.0))  # 2.0, outside
    out = predict_grvs([11.156782, 12.0], [10.719377, 11.0])
    assert out.shape == (2,) and np.isfinite(out).all()


def test_snr_reproduces_the_notebook_and_its_window_classes():
    assert np.round(rvs_snr_per_pixel(10.54), 1) == 13.1
    assert np.round(rvs_snr_per_pixel(10.54, 30), 1) == 71.6
    # Brighter than the window-class limit, every across-scan pixel pays read noise; the
    # S/N therefore jumps *up* just past G_RVS = 7 (fewer samples, less read noise).
    assert rvs_snr_per_pixel(7.0) < rvs_snr_per_pixel(7.01)
    # Monotonic in magnitude elsewhere, and vectorised.
    mags = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
    snr = rvs_snr_per_pixel(mags)
    assert snr.shape == (5,) and np.all(np.diff(snr) < 0.0)
    # Co-adding n transits helps by sqrt(n) at the bright end, where the source dominates.
    assert rvs_snr_per_pixel(8.0, 4) / rvs_snr_per_pixel(8.0) == pytest.approx(2.0, rel=0.05)


def test_transit_resolving_powers_span_the_measured_rows():
    r = rvs_transit_resolving_powers(200, seed=3)
    assert r.shape == (200,)
    assert r.min() >= 11_146.0 - 1.0 and r.max() <= 12_396.0 + 1.0
    assert np.unique(r).size == 8
    np.testing.assert_array_equal(r, rvs_transit_resolving_powers(200, seed=3))


def test_lsf_width_and_quadrature():
    sigma = rvs_lsf_sigma_kms()
    assert sigma == pytest.approx(299792.458 / (11_500 * 2.3548), rel=1e-4)
    assert quadrature_sigma_kms(None) == sigma
    q = quadrature_sigma_kms(20_000.0)
    assert q**2 + rvs_lsf_sigma_kms(20_000.0) ** 2 == pytest.approx(sigma**2)
    with pytest.raises(ValueError, match="cannot be broadened"):
        quadrature_sigma_kms(11_000.0)


# ---------------------------------------------------------------------------
# grids and products
# ---------------------------------------------------------------------------


def test_the_products_have_the_documented_sampling():
    assert RVS_DR3_MEAN.n_pixels == 2401
    assert RVS_DR3_MEAN.wave[0] == 8460.0 and RVS_DR3_MEAN.wave[-1] == pytest.approx(8700.0)
    assert RVS_DR3_MEAN.step == pytest.approx(0.1)
    assert RVS_DR4_EPOCH.n_pixels == 961
    assert RVS_DR4_EPOCH.wave[0] == 8460.0 and RVS_DR4_EPOCH.wave[-1] == 8700.0
    assert RVS_DR4_EPOCH.step == pytest.approx(0.25)
    detector = rvs_detector_grid()
    assert detector.size == 980 and detector[0] == 8460.0 and detector[-1] < 8700.0
    assert np.allclose(np.diff(detector), 0.245)
    with pytest.raises(ValueError):
        RVSProduct(name="bad", wave=[8460.0, 8459.0])


def test_the_model_grid_covers_the_shifted_band():
    grid = rvs_model_grid(300.0, vsini_max_kms=50.0)
    assert grid.wave[0] < 8460.0 * (1 - 350.0 / 299792.458)
    assert grid.wave[-1] > 8700.0 * (1 + 350.0 / 299792.458)
    assert grid.dv_kms == pytest.approx(2.0, rel=1e-6)


def test_epoch_time_helpers_follow_the_reference_implementation():
    t = uniform_phase_times(4.0, 10)
    np.testing.assert_allclose(t, np.linspace(0.0, 4.0, 10))
    r = random_times(10, 12.0, seed=42)
    assert r.shape == (10,) and np.all(np.diff(r) > 0) and r.min() >= 0.0 and r.max() <= 12.0
    np.testing.assert_array_equal(r, random_times(10, 12.0, seed=42))


def test_transit_times_come_in_visibility_periods_on_the_ladder():
    t = rvs_transit_times(40, seed=1)
    assert t.shape == (40,) and np.all(np.diff(t) >= 0)
    assert t.min() >= 2456863.94 and t.max() <= 2456863.94 + 1998.0 + 1.0
    gaps = np.diff(t)
    pair = 106.5 / 360.0 * 0.25
    within = gaps < 4.0
    # Inside a visibility period the only gaps are the 106.5-minute pair and the 4.23-hour
    # complement of a spin; between periods the gaps are at least four days.
    assert np.all(np.isclose(gaps[within], pair) | np.isclose(gaps[within], 0.25 - pair))
    assert np.all(gaps[~within] >= 4.0)
    n_periods = int(np.sum(~within)) + 1
    assert 20 <= n_periods <= 35  # 40 transits at about 1.5 per period
    np.testing.assert_array_equal(t, rvs_transit_times(40, seed=1))
    assert not np.array_equal(t, rvs_transit_times(40, seed=2))
    assert rvs_transit_times(1, seed=0).shape == (1,)
    assert rvs_transit_times(7, seed=0, transits_per_period=1.0).shape == (7,)


def test_transit_counts_follow_the_archive_distribution():
    dr3 = rvs_transit_count(20000, release="dr3", seed=0)
    assert dr3.min() >= 2
    assert abs(np.median(dr3) - 18.0) <= 1.0
    assert abs(np.percentile(dr3, 10) - 8.0) <= 1.5
    assert abs(np.percentile(dr3, 90) - 30.0) <= 2.0
    dr4 = rvs_transit_count(20000, release="dr4", seed=0)
    assert abs(np.median(dr4) / np.median(dr3) - 1998.0 / 1038.0) < 0.05
    ecliptic = rvs_transit_count(5000, ecl_lat_deg=0.0, release="dr3", seed=1)
    ring = rvs_transit_count(5000, ecl_lat_deg=45.0, release="dr3", seed=1)
    assert np.mean(ring) / np.mean(ecliptic) > 1.8
    with pytest.raises(ValueError, match="release"):
        rvs_transit_count(3, release="dr5")


# ---------------------------------------------------------------------------
# real transit times: GOST
# ---------------------------------------------------------------------------

# The recorded answer of https://gaia.esac.esa.int/gost/ObjVisSAP/gaiaobjvisap for
# s_ra = 45, s_dec = +20, t_min = 56800, t_max = 60800, MAXREC = 10000, fetched on
# 2026-09-10: 142 field-of-view crossings over the whole mission, 32 kB. Every test below
# reads it instead of the network.
GOST_FIXTURE = Path(__file__).resolve().parent / "data" / "gost_ra45_dec+20.xml"
GOST_QUERY = dict(t_min_mjd=56800.0, t_max_mjd=60800.0, max_records=10000)


@pytest.fixture
def offline_gost(monkeypatch, tmp_path):
    """The fixture in place of the network, and a cache directory of its own.

    Returns the list the calls are recorded in, so a test can assert that the cache
    spared the second one.
    """
    calls = []

    def fake_fetch(url, *, timeout):
        calls.append(url)
        return GOST_FIXTURE.read_bytes()

    monkeypatch.setattr(gaia_module, "_fetch_gost", fake_fetch)
    monkeypatch.setattr(gaia_module, "cache_dir", lambda: tmp_path)
    return calls


def test_the_gost_votable_parses_into_transits(offline_gost):
    transits = gost_transits(45.0, 20.0, **GOST_QUERY)
    assert len(transits) == transits.n == 142
    assert transits.bjd.dtype == np.float64 and np.all(np.diff(transits.bjd) >= 0)
    # The first row of the file: t_start 56903.930801604874, t_stop 56903.93125121109,
    # whose mid-point plus 2400000.5 is the transit's BJD.
    assert transits.bjd[0] == pytest.approx(2456904.431026408, abs=1e-6)
    assert transits.bjd[1] == pytest.approx(2457023.797094317, abs=1e-6)
    assert transits.bjd[-1] == pytest.approx(2460754.299468420, abs=1e-6)
    # The service publishes neither the field of view, nor the CCD row, nor the scan angle.
    assert transits.fov is None and transits.ccd_row is None
    assert transits.scan_angle_deg is None
    assert transits.ra_deg == 45.0 and transits.dec_deg == 20.0
    assert transits.source.startswith(GOST_ENDPOINT) and "s_ra=45" in transits.source
    # The gap ladder of the scanning law, unmodelled and simply there: the 106.5-minute
    # field-of-view pair, the 4.23-hour complement of a spin, and the spin itself.
    gaps = np.diff(transits.bjd) * 24.0
    close = gaps[gaps < 7.0]
    assert close.size > 60
    assert np.all(
        np.isclose(close, 1.776, atol=0.01)
        | np.isclose(close, 4.228, atol=0.01)
        | np.isclose(close, 6.003, atol=0.01)
    )


def test_gost_reads_its_cache_before_the_network_and_says_where_it_looked(
    offline_gost, tmp_path, monkeypatch
):
    first = gost_transits(45.0, 20.0, **GOST_QUERY)
    assert len(offline_gost) == 1
    cached = tmp_path / "gost" / "objvissap_ra45_dec+20_t56800-60800_n10000.vot"
    assert cached.is_file() and cached.read_bytes() == GOST_FIXTURE.read_bytes()

    second = gost_transits(45.0, 20.0, **GOST_QUERY)
    np.testing.assert_array_equal(second.bjd, first.bjd)
    assert len(offline_gost) == 1  # the cache answered

    # A different query is a different file, and cache=False stores nothing.
    gost_transits(45.0, 20.0, **{**GOST_QUERY, "max_records": 50})
    assert len(offline_gost) == 2
    assert (tmp_path / "gost" / "objvissap_ra45_dec+20_t56800-60800_n50.vot").is_file()
    gost_transits(45.0, -20.0, cache=False, **GOST_QUERY)
    assert not (tmp_path / "gost" / "objvissap_ra45_dec-20_t56800-60800_n10000.vot").exists()

    def refuse(url, *, timeout):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(gaia_module, "_fetch_gost", refuse)
    with pytest.raises(RuntimeError) as error:
        gost_transits(10.0, -30.0, **GOST_QUERY)
    assert GOST_ENDPOINT in str(error.value)
    assert "objvissap_ra10_dec-30_t56800-60800_n10000.vot" in str(error.value)


def test_gost_transits_validates_what_it_is_given():
    with pytest.raises(ValueError, match="sorted"):
        GostTransits(bjd=[2.0, 1.0], ra_deg=0.0, dec_deg=0.0, source="x")
    with pytest.raises(ValueError, match="one entry per transit"):
        GostTransits(bjd=[1.0, 2.0], ra_deg=0.0, dec_deg=0.0, source="x", ccd_row=[4])
    empty = GostTransits(bjd=[], ra_deg=0.0, dec_deg=0.0, source="x")
    assert len(empty) == 0
    assert rvs_transit_times_from_gost(empty).size == 0


def test_rvs_times_from_gost_cut_the_span_the_rows_and_the_losses(offline_gost):
    transits = gost_transits(45.0, 20.0, **GOST_QUERY)
    # 1. The release span. GOST forecasts the whole mission, DR3 covers 34 months of it
    #    and DR4 66, so the two cuts nest and both are smaller than the forecast.
    for release, expected in (("dr3", 25), ("dr4", 89)):
        start, span = RVS_SPANS[release]
        inside = transits.bjd[(transits.bjd >= start) & (transits.bjd <= start + span)]
        assert inside.size == expected
        kept = rvs_transit_times_from_gost(transits, release=release, usable_fraction=1.0)
        assert kept.size <= inside.size
        assert np.all((kept >= start) & (kept <= start + span))
        assert np.all(np.isin(kept, transits.bjd))
    # 2. The CCD rows, when the table carries them: the RVS occupies four of the seven.
    start, _ = RVS_SPANS["dr4"]
    rows = np.arange(14) % 7 + 1
    labelled = GostTransits(
        bjd=start + np.arange(14, dtype=float),
        ra_deg=45.0,
        dec_deg=20.0,
        source="test",
        ccd_row=rows,
    )
    kept = rvs_transit_times_from_gost(labelled, usable_fraction=1.0)
    np.testing.assert_array_equal(kept, start + np.flatnonzero(np.isin(rows, RVS_CCD_ROWS)))
    # Without a row column the 4/7 is drawn, and averages over seeds to 4/7 of 0.78.
    fractions = [
        rvs_transit_times_from_gost(transits, release="dr4", seed=s).size / 89.0 for s in range(200)
    ]
    assert abs(float(np.mean(fractions)) - 4.0 / 7.0 * GOST_USABLE_FRACTION) < 0.02
    # 3. Determinism, and the arguments that are refused.
    once = rvs_transit_times_from_gost(transits, seed=3)
    np.testing.assert_array_equal(once, rvs_transit_times_from_gost(transits, seed=3))
    assert not np.array_equal(once, rvs_transit_times_from_gost(transits, seed=4))
    assert np.all(np.diff(once) >= 0)
    with pytest.raises(ValueError, match="release"):
        rvs_transit_times_from_gost(transits, release="dr5")
    with pytest.raises(ValueError, match="usable_fraction"):
        rvs_transit_times_from_gost(transits, usable_fraction=0.0)


# ---------------------------------------------------------------------------
# delivery
# ---------------------------------------------------------------------------


def _flat_detector_dataset(n_epochs: int, sigma: float, seed: int = 0):
    """Flat unit spectra on the detector grid with white noise of a known sigma."""
    rng = np.random.default_rng(seed)
    wave = rvs_detector_grid()
    epochs = [
        ab.EpochData(
            wave=wave,
            flux=1.0 + rng.normal(0.0, sigma, size=wave.size),
            ivar=np.full(wave.size, 1.0 / sigma**2),
            bjd=2457000.0 + j,
            instrument="RVS-detector",
        )
        for j in range(n_epochs)
    ]
    return ab.Dataset(epochs, frame="barycentric")


@pytest.mark.parametrize("product", [RVS_DR3_MEAN, RVS_DR4_EPOCH])
def test_delivery_propagates_the_variance_and_records_the_correlation(product):
    n_epochs, sigma = 400, 0.05
    detector = _flat_detector_dataset(n_epochs, sigma)
    delivered, record = deliver(detector, product)
    assert delivered.n_epochs == n_epochs and delivered.frame == "barycentric"
    ep = delivered[0]
    assert ep.instrument == "RVS" and ep.medium == "vacuum"
    assert ep.lsf_sigma_kms == pytest.approx(rvs_lsf_sigma_kms())
    assert ep.wave.size == product.n_pixels
    # The last delivered pixel lies beyond the last detector pixel (8699.855 A) and is
    # masked rather than filled.
    # Delivered samples beyond the last detector pixel (8699.855 A) are masked rather
    # than filled: two on the DR3 grid (869.9 and 870.0 nm), one on the DR4 grid.
    outside = product.wave > rvs_detector_grid()[-1]
    assert outside.sum() == (2 if product is RVS_DR3_MEAN else 1)
    assert np.all(ep.ivar[outside] == 0.0) and np.all(ep.flux[outside] == 1.0)
    assert not record.covered[0][outside].any()
    assert record.covered[0][~outside].all()

    # Sample variance across epochs at every delivered pixel against the propagated one.
    flux = np.stack([e.flux for e in delivered])
    predicted = record.sigma[0] ** 2
    ok = record.covered[0]
    measured = flux[:, ok].var(axis=0, ddof=1)
    ratio = measured / predicted[ok]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.05)
    assert np.all(ratio > 0.6) and np.all(ratio < 1.6)

    # Lag-one correlation across epochs against the recorded coefficient.
    resid = flux[:, ok] - 1.0
    lag1 = np.mean(
        [np.corrcoef(resid[:, k], resid[:, k + 1])[0, 1] for k in range(0, resid.shape[1] - 1)]
    )
    assert lag1 == pytest.approx(record.lag1[0], abs=0.04)
    if product is RVS_DR3_MEAN:
        assert record.lag1[0] > 0.5  # 2.45 delivered pixels per detector pixel
        assert record.pixel_ratio == pytest.approx(2.45, rel=0.02)
    else:
        assert 0.2 < record.lag1[0] < 0.5  # the 0.245 -> 0.25 beat
        assert record.pixel_ratio == pytest.approx(0.98, rel=0.02)


def test_delivery_masks_what_reads_a_masked_detector_pixel():
    detector = _flat_detector_dataset(1, 0.05)
    ivar = np.array(detector[0].ivar)
    ivar[500] = 0.0
    masked = ab.Dataset(
        [
            ab.EpochData(
                wave=detector[0].wave, flux=detector[0].flux, ivar=ivar, bjd=detector[0].bjd
            )
        ],
        frame="barycentric",
    )
    delivered, _record = deliver(masked, RVS_DR3_MEAN)
    wave_bad = detector[0].wave[500]
    reads_it = np.abs(delivered[0].wave - wave_bad) < 0.245
    assert np.all(delivered[0].ivar[reads_it] == 0.0)
    inside = delivered[0].wave <= detector[0].wave[-1]
    assert np.all(delivered[0].ivar[~reads_it & inside] > 0.0)


# ---------------------------------------------------------------------------
# the one-call simulator
# ---------------------------------------------------------------------------


def _toy_components(grid):
    return [
        synthetic_deviation_spectrum(grid, n_lines=30, seed=11, sigma_v_range=(6.0, 15.0)),
        synthetic_deviation_spectrum(grid, n_lines=30, seed=12, sigma_v_range=(6.0, 15.0)),
    ]


def test_simulate_rvs_dataset_declares_what_gaia_delivers():
    grid = rvs_model_grid(150.0, dv_kms=3.0)
    components = _toy_components(grid)
    bjd = uniform_phase_times(4.0, 6, start=2457000.0)
    dataset, truth = simulate_rvs_dataset(
        components,
        grid,
        bjd=bjd,
        light_fractions=NOTEBOOK_LIGHT,
        orbit=NOTEBOOK_ORBIT,
        grvs=10.54,
        product=RVS_DR3_MEAN,
        library_resolving_power=None,
        seed=3,
    )
    assert dataset.frame == "barycentric" and dataset.n_epochs == 6
    assert dataset.instruments == ("RVS",)
    assert all(e.medium == "vacuum" for e in dataset)
    assert all(e.n_pixels == 2401 for e in dataset)
    np.testing.assert_allclose(dataset.lsf_sigma_kms, rvs_lsf_sigma_kms())
    np.testing.assert_allclose(truth.snr, rvs_snr_per_pixel(10.54))
    assert truth.grvs == 10.54
    np.testing.assert_allclose(truth.quadrature_sigma_kms, rvs_lsf_sigma_kms())
    np.testing.assert_allclose(truth.resolving_power, 11_500.0)
    assert truth.velocities.shape == (2, 6)
    assert truth.detector.n_epochs == 6 and truth.detector[0].n_pixels == 980
    assert truth.delivery.product is RVS_DR3_MEAN
    # Photon noise: the detector-grid sigma is smaller where the flux is lower.
    sig = truth.simulation.noise_sigma[0]
    noiseless = truth.simulation.noiseless_flux[0]
    core, cont = np.argmin(noiseless), np.argmax(noiseless)
    assert sig[core] < sig[cont]
    assert sig[core] / sig[cont] == pytest.approx(np.sqrt(noiseless[core] / noiseless[cont]))
    # Reproducible, and seed-sensitive.
    again, _ = simulate_rvs_dataset(
        components,
        grid,
        bjd=bjd,
        light_fractions=NOTEBOOK_LIGHT,
        orbit=NOTEBOOK_ORBIT,
        grvs=10.54,
        product=RVS_DR3_MEAN,
        library_resolving_power=None,
        seed=3,
    )
    np.testing.assert_array_equal(again[0].flux, dataset[0].flux)


def test_simulate_rvs_dataset_refuses_ambiguous_or_uncovered_input():
    grid = rvs_model_grid(150.0, dv_kms=3.0)
    components = _toy_components(grid)
    bjd = uniform_phase_times(4.0, 3)
    with pytest.raises(ValueError, match="exactly one of snr= and grvs="):
        simulate_rvs_dataset(
            components, grid, bjd=bjd, light_fractions=NOTEBOOK_LIGHT, orbit=NOTEBOOK_ORBIT
        )
    narrow = ab.LogGrid.from_wavelength_range(8470.0, 8690.0, dv_kms=3.0)
    with pytest.raises(ValueError, match="does not cover the detector grid"):
        simulate_rvs_dataset(
            _toy_components(narrow),
            narrow,
            bjd=bjd,
            light_fractions=NOTEBOOK_LIGHT,
            orbit=NOTEBOOK_ORBIT,
            snr=40.0,
            library_resolving_power=None,
        )


def test_per_transit_resolving_powers_are_applied_but_the_nominal_one_is_declared():
    grid = rvs_model_grid(150.0, dv_kms=3.0)
    components = _toy_components(grid)
    bjd = uniform_phase_times(4.0, 4)
    r = rvs_transit_resolving_powers(4, seed=1)
    dataset, truth = simulate_rvs_dataset(
        components,
        grid,
        bjd=bjd,
        light_fractions=NOTEBOOK_LIGHT,
        orbit=NOTEBOOK_ORBIT,
        snr=50.0,
        resolving_power=r,
        library_resolving_power=None,
        seed=2,
    )
    np.testing.assert_allclose(truth.resolving_power, r)
    np.testing.assert_allclose(truth.lsf_sigma_kms, [rvs_lsf_sigma_kms(x) for x in r])
    np.testing.assert_allclose(dataset.lsf_sigma_kms, rvs_lsf_sigma_kms())
    assert len(truth.detector.instruments) == np.unique(r).size
    assert dataset.instruments == ("RVS",)
    declared, _ = simulate_rvs_dataset(
        components,
        grid,
        bjd=bjd,
        light_fractions=NOTEBOOK_LIGHT,
        orbit=NOTEBOOK_ORBIT,
        snr=50.0,
        resolving_power=r,
        declare_lsf="truth",
        library_resolving_power=None,
        seed=2,
    )
    np.testing.assert_allclose(declared.lsf_sigma_kms, truth.lsf_sigma_kms)
    # The same seed and the same resolving powers give the same noise draw.
    np.testing.assert_array_equal(declared[0].flux, dataset[0].flux)


def test_per_epoch_snr_is_honoured():
    grid = rvs_model_grid(150.0, dv_kms=3.0)
    components = _toy_components(grid)
    bjd = uniform_phase_times(4.0, 3)
    dataset, truth = simulate_rvs_dataset(
        components,
        grid,
        bjd=bjd,
        light_fractions=NOTEBOOK_LIGHT,
        orbit=NOTEBOOK_ORBIT,
        snr=[10.0, 40.0, 160.0],
        product=RVS_DR4_EPOCH,
        library_resolving_power=None,
        seed=5,
    )
    np.testing.assert_allclose(truth.snr, [10.0, 40.0, 160.0])
    med = [np.median(np.sqrt(1.0 / e.ivar[e.ivar > 0])) for e in dataset]
    assert med[0] > med[1] > med[2]
    assert med[0] / med[2] == pytest.approx(16.0, rel=0.15)


@pytest.mark.slow
def test_the_notebook_system_closes_the_loop_through_the_disentangler():
    """The notebook's orbit at its S/N of 40, ten epochs over one period: K to 1%."""
    grid = rvs_model_grid(150.0, dv_kms=3.0)
    components = _toy_components(grid)
    bjd = uniform_phase_times(4.0, 10, start=2457000.0)
    dataset, _truth = simulate_rvs_dataset(
        components,
        grid,
        bjd=bjd,
        light_fractions=NOTEBOOK_LIGHT,
        orbit=NOTEBOOK_ORBIT,
        snr=40.0,
        product=RVS_DR3_MEAN,
        library_resolving_power=None,
        seed=1,
    )
    dis = ab.Disentangler(
        dataset,
        components=[ab.Star("primary", NOTEBOOK_LIGHT[0]), ab.Star("secondary", NOTEBOOK_LIGHT[1])],
        orbit=ab.Orbit(
            period=ab.Known(4.0, 0.002),
            k=ab.Between([20.0, 20.0], [150.0, 150.0], start_at=[70.0, 110.0]),
            ecc=ab.Between(0.0, 0.5),
        ),
        lsf={"RVS": ab.LSF.from_resolution(11_500)},
        dv_kms=3.0,
    )
    fit = dis.fit(max_steps=200)
    k = [fit.star("primary")["k"], fit.star("secondary")["k"]]
    assert abs(k[0] - 87.70) < 0.01 * 87.70, k
    assert abs(k[1] - 95.00) < 0.01 * 95.00, k
