"""Labels from epochs sharper than the library (D74).

A library spectrum cannot be sharpened, so where the library is at or below an
instrument's resolving power the epoch comparison smooths the epochs to the library's
resolution instead (:func:`albireo.smooth_epoch`, :meth:`albireo.Fit.smoothing_widths`,
:meth:`albireo.Fit.smoothed`, ``Fit.match_labels(..., smoothing=)``). The convolution is
checked against analytic profiles, on uniform and non-uniform pixels and around masked
pixels. The noise is checked against the reported ``sum(k^2)``. The interface is checked
for the widths it chooses, what it records and what it rejects, and a closed loop recovers
the labels and the frame offsets from epochs sharper than the library. All data are
simulated in the tests. Nothing is downloaded.
"""

import warnings

import numpy as np
import pytest
from test_library import build_library
from test_match_epochs import DV_KMS, LSF_KMS, TRUTH, _stars, _system

import albireo as ab
from albireo.data import EpochData
from albireo.preprocess import _smoothed, smooth_epoch

C = ab.C_KMS


def _sigma_of(resolving_power):
    return C / resolving_power / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _epoch(u, flux, *, ivar=None, sigma=3.0, mask=None):
    """An epoch on the velocity grid ``u`` (km/s), wavelengths 5000 A exp(u / c)."""
    return EpochData(
        wave=5000.0 * np.exp(np.asarray(u) / C),
        flux=flux,
        ivar=np.ones(np.size(u)) if ivar is None else ivar,
        bjd=0.0,
        mask=mask,
        medium="air",
        lsf_sigma_kms=sigma,
    )


def _line(u, center, sigma, depth=0.5):
    return 1.0 - depth * np.exp(-0.5 * ((u - center) / sigma) ** 2)


def _moments(u, flux, keep):
    """Equivalent width (km/s), centroid and variance of the absorption ``1 - flux``."""
    a = np.where(keep, 1.0 - flux, 0.0)
    w = np.gradient(u)
    ew = float(np.sum(a * w))
    centre = float(np.sum(a * w * u) / ew)
    variance = float(np.sum(a * w * (u - centre) ** 2) / ew)
    return ew, centre, variance


# ---------------------------------------------------------------------------
# the convolution
# ---------------------------------------------------------------------------


def test_a_gaussian_line_is_broadened_to_the_target_width():
    """A line of sigma 3 km/s becomes one of sigma 5 km/s with its area and centre kept."""
    u = np.arange(0.0, 2000.0, 1.0)
    epoch = _epoch(u, _line(u, 1000.37, 3.0), sigma=3.0)
    broad = smooth_epoch(epoch, 5.0)
    assert broad.lsf_sigma_kms == 5.0
    keep = broad.ivar > 0
    ew0, c0, v0 = _moments(u, epoch.flux, np.ones(u.size, bool))
    ew, c, v = _moments(u, broad.flux, keep)
    assert abs(ew / ew0 - 1.0) < 1e-6
    assert abs(c - c0) < 1e-6
    # 9 + 16 km^2/s^2; the truncation at five sigma removes 2e-5 of the kernel's variance
    assert v == pytest.approx(25.0, abs=1e-3)
    assert abs(v0 - 9.0) < 1e-6
    np.testing.assert_array_equal(broad.ivar[keep], epoch.ivar[keep])
    # the profile is the analytic convolution: a Gaussian of sigma 5 and depth 0.5 * 3 / 5
    expected = _line(u, 1000.37, 5.0, depth=0.5 * 3.0 / 5.0)
    assert np.max(np.abs(broad.flux[keep] - expected[keep])) < 3e-4


def test_a_non_uniform_sampling_is_integrated_by_pixel_width():
    """Merged orders: a line across a change of pixel size keeps its centre.

    Each pixel is weighted by its width in velocity. Without that the half with twice as
    many pixels would carry twice the weight and pull the line by about a kilometre per
    second.
    """
    u = np.concatenate([np.arange(0.0, 1000.0, 1.0), np.arange(1000.0, 1600.0, 0.5)])
    epoch = _epoch(u, _line(u, 1000.0, 3.0), sigma=3.0)
    broad = smooth_epoch(epoch, 6.0)
    keep = broad.ivar > 0
    _, c0, _ = _moments(u, epoch.flux, np.ones(u.size, bool))
    ew0, _, _ = _moments(u, epoch.flux, np.ones(u.size, bool))
    ew, c, v = _moments(u, broad.flux, keep)
    assert abs(c - c0) < 0.02
    assert abs(ew / ew0 - 1.0) < 1e-3
    assert v == pytest.approx(36.0, abs=0.2)


def test_masked_pixels_never_make_a_kernel_asymmetric():
    """A short masked run is filled; near a long one, and near the ends, pixels are masked."""
    u = np.arange(0.0, 3000.0, 1.0)
    flux = _line(u, 900.0, 3.0) * _line(u, 2000.0, 3.0)
    clean = smooth_epoch(_epoch(u, flux), 7.0)
    sigma_s = np.sqrt(7.0**2 - 3.0**2)
    reach = 5.0 * sigma_s

    ivar = np.ones(u.size)
    ivar[898] = 0.0  # a cosmic-ray hit in the wing of the first line
    ivar[1500:1510] = 0.0  # a long masked run in the continuum
    hit = smooth_epoch(_epoch(u, np.where(ivar > 0, flux, 7.0), ivar=ivar), 7.0)
    assert hit.ivar[898] == 0.0, "the filled pixel stays masked"
    near_run = np.abs(u - 1504.5) <= 4.5 + reach  # within the reach of a pixel of the run
    assert np.all(hit.ivar[near_run] == 0.0)
    beyond = (np.abs(u - 1504.5) > 4.5 + reach + 1.0) & (clean.ivar > 0) & (u != 898.0)
    assert np.all(hit.ivar[beyond] > 0.0)
    # every kept pixel away from the hit has the clean value: its kernel never met a gap
    far = (hit.ivar > 0) & (np.abs(u - 898.0) > reach + 1.0)
    np.testing.assert_allclose(hit.flux[far], clean.flux[far], rtol=0.0, atol=1e-12)
    # near the hit the linear fill changes the flux by far less than the line depth
    close = (hit.ivar > 0) & (np.abs(u - 898.0) <= reach + 1.0)
    assert np.max(np.abs(hit.flux[close] - clean.flux[close])) < 2e-3
    # the ends: no pixel within the reach of either end is kept
    assert np.all(clean.ivar[u < reach] == 0.0) and np.all(clean.ivar[u > u[-1] - reach] == 0.0)


def test_the_input_variances_are_kept_and_the_noise_falls_by_sum_k2():
    """The smoothed noise has the variance ``sum(k^2)`` the diagnostics report."""
    rng = np.random.default_rng(3)
    u = np.arange(0.0, 120_000.0, 1.0)
    noise = rng.normal(0.0, 1.0, u.size)
    epoch = _epoch(u, 1.0 + noise, ivar=np.ones(u.size), sigma=2.0)
    out, info = _smoothed(epoch, 6.0, 2.0, 3)
    keep = out.ivar > 0
    assert info["kernel_sigma_kms"] == pytest.approx(np.sqrt(32.0))
    # sum k^2 of a Gaussian of s pixels is 1 / (2 sqrt(pi) s)
    assert info["sum_k2"] == pytest.approx(1.0 / (2.0 * np.sqrt(np.pi) * np.sqrt(32.0)), rel=1e-3)
    assert np.var(out.flux[keep] - 1.0) == pytest.approx(info["sum_k2"], rel=0.04)
    np.testing.assert_array_equal(out.ivar[keep], 1.0)
    assert info["kept"] == int(keep.sum()) and info["good"] == u.size


def test_a_narrower_target_or_an_undeclared_width_is_refused():
    u = np.arange(0.0, 500.0, 1.0)
    epoch = _epoch(u, _line(u, 250.0, 3.0), sigma=3.0)
    with pytest.raises(ValueError, match="cannot be sharpened"):
        smooth_epoch(epoch, 2.0)
    unknown = _epoch(u, epoch.flux, sigma=None)
    with pytest.raises(ValueError, match="declares no line-spread width"):
        smooth_epoch(unknown, 5.0)
    assert smooth_epoch(unknown, 5.0, from_sigma_kms=3.0).lsf_sigma_kms == 5.0
    # the epoch's own width: nothing to add, the flux is returned as it is
    same = smooth_epoch(epoch, 3.0)
    np.testing.assert_array_equal(same.flux, epoch.flux)
    with pytest.raises(ValueError, match="no pixel of this epoch keeps a whole"):
        smooth_epoch(_epoch(u[:40], epoch.flux[:40]), 30.0)


# ---------------------------------------------------------------------------
# the Disentangler interface
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def library():
    return build_library()


@pytest.fixture(scope="module")
def plain(library):
    return _system(library, snr=150.0)


def test_the_widths_list_only_what_the_library_does_not_resolve(plain):
    """R = 15,000 (sigma 8.49 km/s) is broader than the 7 km/s epochs; R = 30,000 is not."""
    fit, _ = plain
    assert fit.smoothing_widths(None) == {}
    assert fit.smoothing_widths(30_000.0) == {}
    lib = _sigma_of(15_000.0)
    # "auto" is max(3 km/s, 1.2 dv), and the model pixel is 5 km/s here
    assert fit.smoothing_widths(15_000.0) == {"S": pytest.approx(np.hypot(lib, 1.2 * DV_KMS))}
    assert fit.smoothing_widths(15_000.0, 4.0) == {"S": pytest.approx(np.hypot(lib, 4.0))}
    for bad in (0.0, -1.0, float("nan"), "wide"):
        with pytest.raises(ValueError, match="sigma_q must be"):
            fit.smoothing_widths(15_000.0, bad)


def test_a_smoothed_fit_keeps_the_orbit_and_records_the_smoothing(plain):
    fit, _ = plain
    target = fit.smoothing_widths(15_000.0)["S"]
    smoothed = fit.smoothed({"S": target})
    assert fit.smoothing is None and fit.smoothed({}) is fit
    assert all(e.lsf_sigma_kms == target for e in smoothed.dis.dataset)
    assert float(smoothed.dis.lsf["S"].sigma_kms) == target
    np.testing.assert_array_equal(smoothed.velocities(), fit.velocities())
    (entry,) = smoothed.smoothing
    assert entry["instrument"] == "S" and entry["n_epochs"] == fit.dis.dataset.n_epochs
    assert entry["from_sigma_kms"] == (LSF_KMS, LSF_KMS)
    assert entry["kernel_sigma_kms"][0] == pytest.approx(np.sqrt(target**2 - LSF_KMS**2))
    assert 0.9 < entry["kept_fraction"] <= 1.0 and 0.0 < entry["sum_k2"] < 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the smoothing is recorded, not warned
        stats = smoothed.epoch_statistics(resolving_power=15_000.0)
    assert stats.smoothed_lsf_sigma_kms == (("S", target),)
    assert stats.lsf_sigma_kms[0][1][0] == pytest.approx(1.2 * DV_KMS)  # the quadrature width
    (note,) = stats.notes
    assert f"smoothed to sigma {target:.3f} km/s" in note and "upper bounds" in note
    assert "epochs smoothed before the comparison: S to" in stats.summary()


def test_the_epoch_comparison_names_the_fallback_and_rejects_misuse(plain, library):
    fit, _ = plain
    coarse = build_library(resolving_power=15_000.0)
    with pytest.raises(ValueError, match=r"at or below the instrument.*smoothing='auto'"):
        fit.match_labels(_stars(coarse), mh=ab.Between(-1.0, 0.5))
    with pytest.raises(ValueError, match="smoothing= applies to the epoch comparison"):
        fit.match_labels(_stars(coarse), compare="native", smoothing="auto")
    stats = fit.epoch_statistics()
    with pytest.raises(ValueError, match="either smoothing= or statistics="):
        fit.match_labels(_stars(library), smoothing="auto", statistics=stats)
    with pytest.raises(ValueError, match="no epochs of instrument"):
        fit.smoothed({"T": 9.0})
    with pytest.raises(ValueError, match="cannot be sharpened"):
        fit.smoothed({"S": 5.0})


def test_a_declared_noise_correlation_is_not_smoothed(library):
    """A convolution changes the correlation that an AR(1) declaration describes."""
    fit, _ = _system(library, ar1=0.3)
    with pytest.raises(ValueError, match="declares a noise correlation"):
        fit.smoothed({"S": 9.0})


def test_the_pipeline_scans_frame_offsets_at_the_smoothed_width(plain):
    """The scan over frame offsets steps by twice the narrowest width of the comparison."""
    from types import SimpleNamespace

    from albireo.pipeline import Analysis, _smoothed_widths, _v_zero_scan

    fit, _ = plain
    settings = Analysis()
    reach, step = _v_zero_scan(fit, settings)
    assert (reach, step) == (300.0, 2.0 * LSF_KMS)
    target = fit.smoothing_widths(15_000.0)["S"]
    assert _v_zero_scan(fit, settings, {"S": target}) == (300.0, 2.0 * target)
    match = SimpleNamespace(statistics=SimpleNamespace(smoothed_lsf_sigma_kms=(("S", target),)))
    assert _smoothed_widths(match) == {"S": target}
    assert _smoothed_widths(SimpleNamespace()) == {}


# ---------------------------------------------------------------------------
# the closed loop
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_labels_and_frame_offsets_are_recovered_from_epochs_sharper_than_the_library(library):
    """Epochs at 7 km/s, a library at R = 15,000 (8.49 km/s): the comparison runs at 10.4.

    The epochs are rendered on a 1 km/s grid from the intrinsic toy library, as the slow
    test of the native comparison does, and the frame offset injected is zero. The library
    of the fit is the same rule broadened analytically to R = 15,000, so only the smoothing
    separates this fit from the native one with the intrinsic library, which is run
    beside it. The two agree within a formal error, and both recover the injected values
    (measured: Teff within 11 K, log g and [M/H] within 0.02 dex, v sin i within 0.9 km/s,
    frame offsets within 0.11 km/s, light within 0.001).
    """
    fit, weights = _system(library, snr=150.0, render_dv=1.0)
    coarse = build_library(resolving_power=15_000.0)
    options = {"mh": ab.Between(-1.0, 0.5), "scan_vsini": [20.0], "top_k": 1, "max_steps": 150}
    got = fit.match_labels(_stars(coarse), smoothing="auto", **options)
    native = fit.match_labels(_stars(library), **options)
    target = float(np.hypot(_sigma_of(15_000.0), 1.2 * DV_KMS))
    assert got.statistics.smoothed_lsf_sigma_kms == (("S", pytest.approx(target)),)
    assert got.assumptions["epoch_smoothed_lsf_sigma_kms"] == {"S": pytest.approx(target)}
    assert native.statistics.smoothed_lsf_sigma_kms == ()
    assert got.epoch_fit.status == "converged"
    for i, (name, (teff, logg, mh, vsini)) in enumerate(TRUTH.items()):
        labels, errors = got.labels[name], got.errors("laplace")[name]
        assert abs(labels["teff"] - teff) < 0.01 * teff, (name, labels)
        assert abs(labels["logg"] - logg) < 0.05, (name, labels)
        assert abs(labels["mh"] - mh) < 0.03, (name, labels)
        assert abs(labels["vsini"] - vsini) < 0.15 * vsini, (name, labels)
        assert abs(labels["v_kms"]) < 0.25, (name, labels)
        assert abs(got.flux_ratio[name] - float(np.median(weights[i]))) < 0.005
        for label in ("teff", "logg", "mh", "v_kms"):
            gap = abs(labels[label] - native.labels[name][label])
            assert gap < errors[label], (name, label, gap, errors[label])
            # the input variances make the formal errors upper bounds: never below native's
            assert errors[label] >= 0.95 * native.errors("laplace")[name][label]
    # with the input variances on correlated pixels the chi-square per pixel is about
    # sum(k^2), not one
    (entry,) = fit.smoothed({"S": target}).smoothing
    assert got.chi2 / got.n_pixels_used == pytest.approx(entry["sum_k2"], rel=0.15)
    assert got.chi2 < got.chi2_nearest_node < got.chi2_continuum
