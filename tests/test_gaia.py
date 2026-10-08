"""Tests for the Gaia RVS simulator (``albireo.gaia``).

Three kinds of claim are tested. The photometric and signal-to-noise relations reproduce
the reference implementation's numbers (Rowan's ``synthetic_rvs_spectra.ipynb``). The
delivery step treats the noise as declared: the propagated variance and the lag-one
correlation equal those measured in the delivered pixels. The one-call simulator declares
what Gaia delivers (vacuum, barycentric, the RVS width) and is tested in a closed loop
through the disentangler.
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
    rvs_components,
    rvs_delivered_sigma_kms,
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
    # Brighter than the window-class limit, every across-scan pixel adds read noise. The
    # S/N therefore increases just past G_RVS = 7 (fewer samples, less read noise).
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


def test_the_model_grid_length_can_be_rounded_up_at_the_red_end():
    # Systems of different velocity amplitude have grids of different lengths, and JAX
    # compiles an array program once per length. A rounded length is shared.
    grid = rvs_model_grid(137.0, vsini_max_kms=21.0)
    rounded = rvs_model_grid(137.0, vsini_max_kms=21.0, length_multiple=512)
    assert rounded.n % 512 == 0 and 0 < rounded.n - grid.n < 512
    assert (rounded.x0, rounded.dx, rounded.relativistic) == (grid.x0, grid.dx, grid.relativistic)
    np.testing.assert_array_equal(rounded.wave[: grid.n], grid.wave)
    lengths = {rvs_model_grid(v, vsini_max_kms=s).n for v in (60.0, 137.0, 240.0) for s in (5, 60)}
    shared = {
        rvs_model_grid(v, vsini_max_kms=s, length_multiple=512).n
        for v in (60.0, 137.0, 240.0)
        for s in (5, 60)
    }
    assert len(lengths) == 6 and len(shared) <= 2
    assert rvs_model_grid(137.0, vsini_max_kms=21.0, length_multiple=grid.n).n == grid.n
    for bad in (0, -4, 2.5):
        with pytest.raises(ValueError, match="length_multiple"):
            rvs_model_grid(137.0, length_multiple=bad)


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

# The recorded response of https://gaia.esac.esa.int/gost/ObjVisSAP/gaiaobjvisap for
# s_ra = 45, s_dec = +20, t_min = 56800, t_max = 60800, MAXREC = 10000, fetched on
# 2026-09-10: 142 field-of-view crossings over the whole mission, 32 kB. Every test below
# reads it instead of the network.
GOST_FIXTURE = Path(__file__).resolve().parent / "data" / "gost_ra45_dec+20.xml"
GOST_QUERY = dict(t_min_mjd=56800.0, t_max_mjd=60800.0, max_records=10000)


@pytest.fixture
def offline_gost(monkeypatch, tmp_path):
    """The fixture in place of the network, and a separate cache directory.

    Returns the list in which the calls are recorded, so a test can assert that the
    second call was served from the cache.
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
    # The gap ladder of the scanning law is present without being modelled: the 106.5-minute
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
    assert len(offline_gost) == 1  # served from the cache

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
    # 2. The CCD rows, when the table has them: the RVS occupies four of the seven.
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
    # 3. Determinism, and the arguments that are rejected.
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


@pytest.mark.parametrize("product", [RVS_DR3_MEAN, RVS_DR4_EPOCH])
def test_the_delivered_width_is_the_second_moment_the_interpolation_adds(product):
    """``sigma_eff^2 - sigma_R^2`` against the second moment ``deliver`` adds to narrow lines.

    Gaussian lines of 0.35 A sigma are sampled at the detector pixel centres and delivered.
    The same lines sampled directly on the product grid are the reference, so that the
    discrete sampling and the window cancel and only the interpolation remains. The 49
    lines are 4.25 A apart, 17/49 of the DR4 grid's 12.25 A beat period, so their centres
    fall at every phase of the beat once. Each line's excess is converted to km/s at its own
    wavelength. On the DR3 grid every line gains ``Delta^2 / 6`` to within a few percent. On
    the DR4 grid the gain of a line varies with its phase from a quarter of that to about
    one and a half times it (``Delta^2 / 4`` at ``t = 1/2``), and only the mean is the
    stationary width.
    """
    sigma_line = 0.35
    centres = 8474.0 + 4.25 * np.arange(49)
    detector_wave = rvs_detector_grid()
    flux = 1.0 - sum(0.5 * np.exp(-0.5 * ((detector_wave - c) / sigma_line) ** 2) for c in centres)
    detector = ab.Dataset(
        [
            ab.EpochData(
                wave=detector_wave,
                flux=flux,
                ivar=np.full(detector_wave.size, 1.0e4),
                bjd=2457000.0,
                instrument="RVS-detector",
            )
        ],
        frame="barycentric",
    )
    delivered, _record = deliver(detector, product)
    wave = delivered[0].wave
    exact = 1.0 - sum(0.5 * np.exp(-0.5 * ((wave - c) / sigma_line) ** 2) for c in centres)

    excess = []
    for c in centres:
        window = np.abs(wave - c) < 5.0 * sigma_line
        x = wave[window] - c

        def moment(depth, x=x):
            return float(np.sum(x**2 * depth) / np.sum(depth))

        added = moment(1.0 - delivered[0].flux[window]) - moment(1.0 - exact[window])
        excess.append(added * (ab.C_KMS / c) ** 2)
    excess = np.asarray(excess)

    step_kms = ab.C_KMS * 0.245 / 8580.0
    predicted = step_kms**2 / 6.0
    # The declared width also counts the detector pixel's width in place of the delivered
    # one. Point samples on either grid are insensitive to it, and the next test measures it.
    pixels = (step_kms**2 - (ab.C_KMS * product.step / 8580.0) ** 2) / 12.0
    assert rvs_delivered_sigma_kms(product) ** 2 - rvs_lsf_sigma_kms() ** 2 == pytest.approx(
        predicted + pixels, rel=1e-12
    )
    expected = {"dr3-mean": 11.826, "dr4-epoch": 11.598}[product.name]
    assert rvs_delivered_sigma_kms(product) == pytest.approx(expected, abs=5e-4)
    assert np.mean(excess) == pytest.approx(predicted, rel=0.01)
    if product is RVS_DR3_MEAN:
        assert np.all(np.abs(excess / predicted - 1.0) < 0.05)
    else:
        # Between 0.24 and 1.57 of the mean; Delta^2 / 4 at t = 1/2 is 1.5 times the mean.
        assert excess.min() < 0.3 * predicted and excess.max() > 1.3 * predicted
        assert excess.max() < 1.65 * predicted
    # The declared resolving power enters in quadrature, and nothing else does.
    wider = rvs_delivered_sigma_kms(product, resolving_power=10_000.0)
    assert wider**2 - rvs_lsf_sigma_kms(10_000.0) ** 2 == pytest.approx(
        predicted + pixels, rel=1e-12
    )


# The line positions of the two moment tests below: 26 lines 7.9 A (276 km/s) apart, so that
# a line's +-110 km/s moment window contains no wing of its neighbour.
_LINE_CENTRES = 8478.0 + 7.9 * np.arange(26)
_MOMENT_HALF_KMS = 110.0


def _excess_moments(u, depths, edges_u, centres_u, sigma):
    """Central second moment of each line minus that of the continuous line on the same bins."""
    from scipy.special import ndtr

    out = []
    for depth, centres in zip(depths, centres_u, strict=True):
        for centre in centres:
            window = np.abs(u - centre) < _MOMENT_HALF_KMS

            def moment(w, x=u[window]):
                mean = np.sum(w * x) / np.sum(w)
                return np.sum(w * (x - mean) ** 2) / np.sum(w)

            ideal = (
                ndtr((edges_u[1:] - centre) / sigma) - ndtr((edges_u[:-1] - centre) / sigma)
            ) / np.diff(edges_u)
            out.append(moment(depth[window]) - moment(ideal[window]))
    return np.asarray(out)


@pytest.mark.parametrize("product", [RVS_DR3_MEAN, RVS_DR4_EPOCH])
def test_the_delivered_width_counts_the_detector_pixel_it_was_read_from(product):
    """Lines integrated over detector pixels, delivered, against lines integrated over the
    delivered pixels: the excess is ``Delta_det^2 / 6 + (Delta_det^2 - Delta_prod^2) / 12``.

    The second term is the detector pixel's box, which every delivered sample inherits, less
    the delivered pixel's box, over which an analysis's operator integrates its model:
    +5.09 km^2/s^2 on the DR3 grid and -0.25 on the DR4 grid.
    """
    from albireo.operators import bin_edges_from_centers

    sigma = 11.07
    detector_wave = rvs_detector_grid()
    det_u = ab.C_KMS * np.log(bin_edges_from_centers(detector_wave))
    shifts = 0.0137 * np.arange(8)  # A, eight phases against both grids
    epochs, centres = [], []
    for j, shift in enumerate(shifts):
        u_c = ab.C_KMS * np.log(_LINE_CENTRES + shift)
        depth = np.zeros(detector_wave.size)
        for c in u_c:
            depth += (
                0.4
                * (_excess_ndtr(det_u[1:], c, sigma) - _excess_ndtr(det_u[:-1], c, sigma))
                / np.diff(det_u)
            )
        epochs.append(
            ab.EpochData(
                wave=detector_wave,
                flux=1.0 - depth,
                ivar=np.full(detector_wave.size, 1.0e4),
                bjd=2457000.0 + j,
                instrument="RVS-detector",
            )
        )
        centres.append(u_c)
    delivered, _ = deliver(ab.Dataset(epochs, frame="barycentric"), product)
    u = ab.C_KMS * np.log(product.wave)
    edges = ab.C_KMS * np.log(bin_edges_from_centers(product.wave))
    excess = _excess_moments(
        u, [1.0 - np.asarray(e.flux) for e in delivered], edges, centres, sigma
    )
    det = ab.C_KMS * product.detector_step / 8580.0
    prod = ab.C_KMS * product.step / 8580.0
    predicted = det**2 / 6.0 + (det**2 - prod**2) / 12.0
    assert rvs_delivered_sigma_kms(product) ** 2 - rvs_lsf_sigma_kms() ** 2 == pytest.approx(
        predicted, rel=1e-12
    )
    # 208 lines: 17.312 against 17.303 on the DR3 grid, and 12.06 against 11.96 on the DR4
    # grid, whose interpolation term varies with the 12.25 A beat phase.
    tolerance = 0.005 if product is RVS_DR3_MEAN else 0.02
    assert np.mean(excess) == pytest.approx(predicted, rel=tolerance)
    if product is RVS_DR3_MEAN:
        assert np.mean(excess) > det**2 / 6.0 + 4.5  # the pixel term, not the interpolation


def _excess_ndtr(x, centre, sigma):
    from scipy.special import ndtr

    return ndtr((x - centre) / sigma)


def _line_library(sigma_line_kms=2.0):
    """An intrinsic library of narrow Gaussian lines on a 0.005 A grid, the same at every node."""
    from albireo.library import SpectralLibrary

    wave = np.arange(8350.0, 8850.0, 0.005)
    u = ab.C_KMS * np.log(wave)
    flux = np.ones_like(wave)
    for c in _LINE_CENTRES:
        flux -= 0.4 * np.exp(-0.5 * ((u - ab.C_KMS * np.log(c)) / sigma_line_kms) ** 2)
    nodes = np.array([[t, g, m] for t in (5000.0, 5500.0) for g in (4.0, 4.5) for m in (0.0, 0.5)])
    return SpectralLibrary(
        label_names=("teff", "logg", "mh"),
        nodes=nodes,
        normalized=np.tile(flux, (nodes.shape[0], 1)),
        log_continuum=np.zeros((nodes.shape[0], wave.size)),
        wave=wave,
        medium="vacuum",
        meta={},
    )


def test_the_simulation_adds_five_twelfths_of_its_model_pixel_squared():
    """The simulator's discretisation, measured through the shipped chain (D65).

    Narrow lines rendered by ``rvs_components`` and ``simulate_rvs_dataset`` on the 2 km/s
    model grid, at 26 positions and 32 epoch velocities, are compared with the continuous
    line convolved with the applied width and integrated over the detector pixels. Without
    rotation three steps contribute (the box average onto the grid, the shift interpolation
    and the model pixel in the rebin), ``(4/12) dv^2`` in total, less the 0.050 km^2/s^2 by
    which the Gaussian kernel truncated at four sigma falls short of ``sigma^2``. The
    rotation kernel's pixel integration adds up to ``dv^2 / 12`` more (0.28 at 11 km/s),
    and the declaration uses the ``(5/12) dv^2`` of a rotation that the grid resolves.
    """
    from albireo.operators import bin_edges_from_centers, gaussian_kernel

    library = _line_library()
    detector_wave = rvs_detector_grid()
    u = ab.C_KMS * np.log(detector_wave)
    edges = ab.C_KMS * np.log(bin_edges_from_centers(detector_wave))
    n_ep = 32
    velocities = -40.0 + 80.0 * np.mod(np.arange(n_ep) * 0.6180339887498949 + 0.123, 1.0)
    results = {}
    for vsini in (0.0, 11.0):
        grid = rvs_model_grid(60.0, dv_kms=2.0, vsini_max_kms=max(vsini, 1.0))
        components = rvs_components(
            library, [{"teff": 5000.0, "logg": 4.0, "mh": 0.0}], grid, vsini_kms=[vsini]
        )
        _, truth = simulate_rvs_dataset(
            components,
            grid,
            bjd=2457000.0 + np.arange(n_ep, dtype=float),
            light_fractions=np.array([1.0]),
            velocities=velocities[None, :],
            snr=1.0e6,
            library=library,
        )
        assert truth.library_resolving_power is None  # an intrinsic library gets the full width
        sigma = float(truth.quadrature_sigma_kms[0])
        assert sigma == pytest.approx(rvs_lsf_sigma_kms())
        rotation = 0.225 * vsini**2  # the limb-darkened profile at epsilon 0.6
        centres = [
            ab.C_KMS * (np.log(_LINE_CENTRES) + float(ab.log_doppler_shift(v))) for v in velocities
        ]
        depths = [1.0 - np.asarray(f) for f in truth.simulation.noiseless_flux]
        results[vsini] = _excess_moments(
            u, depths, edges, centres, float(np.sqrt(2.0**2 + sigma**2 + rotation))
        ).mean()
        dv = float(grid.dv_kms)
    kernel = np.asarray(gaussian_kernel(rvs_lsf_sigma_kms() / dv))
    radius = (kernel.size - 1) // 2
    deficit = (
        np.sum(kernel * np.arange(-radius, radius + 1) ** 2) * dv**2 - rvs_lsf_sigma_kms() ** 2
    )
    assert -0.06 < deficit < -0.04
    assert results[0.0] == pytest.approx((4.0 / 12.0) * dv**2 + deficit, rel=0.03)
    declared = gaia_module.SIMULATION_VARIANCE_FACTOR * dv**2
    assert declared == pytest.approx(5.0 / 12.0 * 4.0)
    assert results[11.0] == pytest.approx(declared, rel=0.10)
    assert 0.0 < results[11.0] - results[0.0] < dv**2 / 12.0
    simulated = rvs_delivered_sigma_kms(RVS_DR4_EPOCH, simulation_dv_kms=dv)
    assert simulated**2 - rvs_delivered_sigma_kms(RVS_DR4_EPOCH) ** 2 == pytest.approx(
        declared, rel=1e-12
    )
    assert simulated == pytest.approx(11.670, abs=5e-4)
    with pytest.raises(ValueError, match="simulation_dv_kms must be positive"):
        rvs_delivered_sigma_kms(RVS_DR4_EPOCH, simulation_dv_kms=0.0)


def test_the_simulator_reads_the_resolving_power_from_the_library():
    """The components' own broadening comes from the library, not from a BOSZ default (D65)."""
    grid = rvs_model_grid(150.0, dv_kms=3.0)
    components = _toy_components(grid)
    bjd = uniform_phase_times(4.0, 3, start=2457000.0)
    common = dict(bjd=bjd, light_fractions=NOTEBOOK_LIGHT, orbit=NOTEBOOK_ORBIT, snr=40.0, seed=1)
    with pytest.raises(ValueError, match="declare the resolving power the components"):
        simulate_rvs_dataset(components, grid, **common)
    intrinsic = _line_library()
    _, truth = simulate_rvs_dataset(components, grid, library=intrinsic, **common)
    assert truth.library_resolving_power is None
    np.testing.assert_allclose(truth.quadrature_sigma_kms, rvs_lsf_sigma_kms())
    published = intrinsic.replace(meta={"resolution": 20000})
    _, truth = simulate_rvs_dataset(components, grid, library=published, **common)
    assert truth.library_resolving_power == 20000.0
    np.testing.assert_allclose(truth.quadrature_sigma_kms, quadrature_sigma_kms(20_000.0))
    # an explicit declaration works without a library, and must agree with one that is given
    _, truth = simulate_rvs_dataset(components, grid, library_resolving_power=20_000.0, **common)
    assert truth.library_resolving_power == 20000.0
    with pytest.raises(ValueError, match="contradicts the library"):
        simulate_rvs_dataset(
            components, grid, library=intrinsic, library_resolving_power=20_000.0, **common
        )


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


def test_a_normalisation_residual_is_passed_to_the_detector_epochs():
    """`response_order` and `response_amplitude` give each transit its own polynomial, and
    the default leaves the normalisation exact."""
    from albireo.gaia import rvs_components, rvs_model_grid, simulate_rvs_dataset
    from albireo.simulate import synthetic_library

    library = synthetic_library((8440.0, 8720.0), n_pix=700, medium="vacuum")
    grid = rvs_model_grid(60.0, vsini_max_kms=10.0)
    labels = [{"teff": 5500.0, "logg": 4.5, "mh": 0.0}, {"teff": 5000.0, "logg": 4.5, "mh": 0.0}]
    components = rvs_components(library, labels, grid, vsini_kms=[8.0, 6.0])
    common = {
        "bjd": np.linspace(0.0, 9.0, 6),
        "light_fractions": (0.7, 0.3),
        "velocities": np.array(
            [[20.0, 5.0, -15.0, -20.0, 0.0, 18.0], [-30.0, -8.0, 22.0, 30.0, 0.0, -27.0]]
        ),
        "snr": 200.0,
        "library": library,
        "seed": 4,
    }
    exact, truth = simulate_rvs_dataset(components, grid, **common)
    assert all(coefficients.size == 0 for coefficients in truth.simulation.response_coeffs)
    tilted, record = simulate_rvs_dataset(
        components, grid, response_order=2, response_amplitude=0.03, **common
    )
    coefficients = np.array(record.simulation.response_coeffs)
    assert coefficients.shape == (6, 3) and 0.005 < coefficients.std() < 0.1
    # The first coefficient is a scale: the mean flux of an epoch moves by it.
    for a, b, c in zip(exact, tilted, coefficients, strict=True):
        ratio = np.mean(b.flux[b.ivar > 0]) / np.mean(a.flux[a.ivar > 0])
        assert ratio == pytest.approx(1.0 + c[0], abs=0.03)
