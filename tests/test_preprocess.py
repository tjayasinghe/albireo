"""Tests for :mod:`albireo.preprocess` — the archival-spectrum entry path.

The recurring theme is that these functions exist to survive things the simulator never
produces: a response that varies by an order of magnitude across the window, no error
array at all, deep one-sided lines, cosmic rays, and per-exposure wavelength grids that
differ by a hundredth of a pixel. Each test below pins one of those.
"""

from __future__ import annotations

import numpy as np
import pytest

from albireo.data import EpochData
from albireo.preprocess import (
    TELLURIC_BANDS,
    der_snr_sigma,
    estimate_ivar,
    fit_continuum,
    mask_ranges,
    mask_spikes,
    mask_tellurics,
    normalize,
    select_region,
    share_wavelength_grid,
)

RNG = np.random.default_rng(20240811)


def _spectrum(n=4000, lo=4000.0, hi=4600.0, decay=0.0, noise=0.0, lines=((4300.0, 1.0, 0.5),)):
    """A synthetic spectrum: ``continuum * (1 - sum of Gaussian lines) + noise``."""
    wave = np.linspace(lo, hi, n)
    continuum = 2.0 * np.exp(-decay * (wave - lo))
    absorption = np.ones_like(wave)
    for center, width, depth in lines:
        absorption -= depth * np.exp(-0.5 * ((wave - center) / width) ** 2)
    flux = continuum * absorption
    if noise:
        flux = flux + noise * continuum * RNG.standard_normal(n)
    return wave, flux, continuum


# --------------------------------------------------------------------------- continuum


def test_fit_continuum_recovers_a_flat_continuum_through_lines():
    wave, flux, truth = _spectrum(lines=((4200.0, 1.0, 0.6), (4400.0, 2.0, 0.4)))
    cont = fit_continuum(wave, flux, smooth_angstrom=150.0)
    assert np.max(np.abs(cont / truth - 1.0)) < 0.005


@pytest.mark.parametrize("decay", [0.002, 0.005, 0.01])
def test_fit_continuum_follows_a_steep_multiplicative_response(decay):
    """The reason the fit is done in the log.

    A merged echelle spectrum's response falls by an order of magnitude across a few
    hundred Angstrom (measured: 20x over 3850-4750 A in FEROS). A curvature penalty
    applied to the flux lags that badly; applied to its logarithm it is exact for a pure
    exponential, since straight lines are in the penalty's nullspace.
    """
    wave, flux, truth = _spectrum(decay=decay, lines=((4300.0, 1.5, 0.5),))
    contrast = truth[0] / truth[-1]
    assert contrast > 3.0, "test would not be probing anything"
    cont = fit_continuum(wave, flux, smooth_angstrom=150.0)
    assert np.max(np.abs(cont / truth - 1.0)) < 0.01


def test_fit_continuum_is_insensitive_to_the_smoothing_scale():
    """A well-posed continuum should not depend much on a knob the user has to guess."""
    wave, flux, truth = _spectrum(decay=0.004, noise=0.002)
    fits = [fit_continuum(wave, flux, smooth_angstrom=s) for s in (60.0, 120.0, 240.0)]
    for cont in fits:
        assert np.max(np.abs(cont / truth - 1.0)) < 0.02
    assert np.max(np.abs(fits[0] / fits[-1] - 1.0)) < 0.02


def test_fit_continuum_is_not_dragged_down_by_a_broad_line():
    """Balmer wings are 40 A wide; the rejection stage, not stiffness, must handle them."""
    wave, flux, truth = _spectrum(lines=((4300.0, 15.0, 0.5),))
    cont = fit_continuum(wave, flux, smooth_angstrom=150.0)
    core = np.abs(wave - 4300.0) < 5.0
    assert np.max(np.abs(cont[core] / truth[core] - 1.0)) < 0.05


def test_fit_continuum_survives_a_very_long_smoothing_scale():
    """Regression: a per-pixel Whittaker smoother needs lam ~ (L/2pi)^4.

    At L = 5000 pixels that is 4e12, the weight term is lost to rounding, and the
    factorization fails outright ("leading minor not positive definite"). The knot basis
    keeps the penalty at O(1) whatever the requested scale.
    """
    wave, flux, truth = _spectrum(n=20000, decay=0.003)
    cont = fit_continuum(wave, flux, smooth_angstrom=300.0)
    assert np.all(np.isfinite(cont))
    assert np.max(np.abs(cont / truth - 1.0)) < 0.02


def test_fit_continuum_ignores_non_finite_and_non_positive_samples():
    wave, flux, truth = _spectrum(decay=0.003)
    flux = flux.copy()
    flux[100:150] = np.nan
    flux[1000:1010] = -5.0
    cont = fit_continuum(wave, flux, smooth_angstrom=150.0)
    assert np.all(np.isfinite(cont)) and np.all(cont > 0)
    assert np.max(np.abs(cont / truth - 1.0)) < 0.02


def test_fit_continuum_rejects_impossible_inputs():
    wave, flux, _ = _spectrum(n=100)
    with pytest.raises(ValueError, match="equal length"):
        fit_continuum(wave, flux[:-1])
    with pytest.raises(ValueError, match="no wider than the spectrum"):
        fit_continuum(wave, flux, smooth_angstrom=10_000.0)
    with pytest.raises(ValueError, match="pixels wide"):
        fit_continuum(wave, flux, smooth_angstrom=1e-3)
    with pytest.raises(ValueError, match="positive flux"):
        fit_continuum(wave, -np.abs(flux), smooth_angstrom=100.0)
    with pytest.raises(ValueError, match="asymmetry"):
        fit_continuum(wave, flux, smooth_angstrom=100.0, asymmetry=0.2)


def test_normalize_guards_a_collapsing_continuum():
    wave = np.linspace(4000.0, 4600.0, 4000)
    flux = np.exp(-0.05 * (wave - 4000.0)) + 1e-9  # falls through nine orders of magnitude
    flux_norm, ivar, cont = normalize(wave, flux, smooth_angstrom=100.0)
    assert ivar is None
    bad = ~np.isfinite(flux_norm)
    assert bad.any(), "the guard should have fired somewhere"
    assert np.all(cont[~bad] > 0)
    assert np.nanmax(np.abs(flux_norm[~bad] - 1.0)) < 0.5


def test_normalize_propagates_supplied_errors():
    wave, flux, truth = _spectrum(decay=0.002, noise=0.0)
    err = 0.01 * truth
    _, ivar, _ = normalize(wave, flux, err=err, smooth_angstrom=150.0)
    assert ivar is not None
    # ivar = (continuum / err)^2, and continuum ~ truth, so sigma_norm ~ 0.01 everywhere.
    sigma = 1.0 / np.sqrt(ivar[ivar > 0])
    assert np.allclose(sigma, 0.01, rtol=0.05)


# ------------------------------------------------------------------------------- noise


def test_der_snr_matches_a_known_gaussian_noise_level():
    flux = 1.0 + 0.01 * RNG.standard_normal(20000)
    assert der_snr_sigma(flux) == pytest.approx(0.01, rel=0.05)


def test_der_snr_is_not_fooled_by_lines():
    """The lag-2 stencil annihilates locally linear signal; the median handles the rest."""
    wave = np.linspace(4000.0, 4600.0, 20000)
    clean = 1.0 - 0.7 * np.exp(-0.5 * ((wave - 4300.0) / 2.0) ** 2)
    noisy = clean + 0.01 * RNG.standard_normal(wave.size)
    assert der_snr_sigma(noisy) == pytest.approx(0.01, rel=0.10)


def test_der_snr_returns_nan_for_too_few_samples():
    assert np.isnan(der_snr_sigma([1.0, 2.0, 3.0]))


def test_estimate_ivar_poisson_scaling_tracks_the_continuum():
    """Photon noise in a normalized spectrum scales as 1/sqrt(counts)."""
    wave = np.linspace(4000.0, 4600.0, 40000)
    continuum = 1000.0 * np.exp(-0.005 * (wave - 4000.0))  # 20x fall in throughput
    sigma_true = 1.0 / np.sqrt(continuum)
    flux_norm = 1.0 + sigma_true * RNG.standard_normal(wave.size)
    ivar = estimate_ivar(wave, flux_norm, continuum=continuum, scaling="poisson")
    sigma = 1.0 / np.sqrt(ivar)
    assert np.allclose(sigma, sigma_true, rtol=0.08)


def test_estimate_ivar_falls_back_when_the_continuum_is_not_positive():
    wave = np.linspace(4000.0, 4600.0, 5000)
    continuum = np.linspace(-1.0, 100.0, 5000)
    flux = 1.0 + 0.01 * RNG.standard_normal(5000)
    with pytest.warns(RuntimeWarning, match="strictly positive continuum"):
        ivar = estimate_ivar(wave, flux, continuum=continuum, scaling="poisson")
    assert np.all(ivar > 0)


def test_estimate_ivar_zeroes_non_finite_pixels_and_needs_a_continuum():
    wave = np.linspace(4000.0, 4600.0, 5000)
    flux = 1.0 + 0.01 * RNG.standard_normal(5000)
    flux[10:20] = np.nan
    ivar = estimate_ivar(wave, flux, scaling="constant")
    assert np.all(ivar[10:20] == 0.0)
    assert np.all(ivar[100:200] > 0.0)
    with pytest.raises(ValueError, match="needs continuum"):
        estimate_ivar(wave, flux, scaling="poisson")
    with pytest.raises(ValueError, match="scaling must be"):
        estimate_ivar(wave, flux, scaling="nonsense")


# ------------------------------------------------------------------- region and masking


def _epoch(n=500, lo=4000.0, hi=4600.0, instrument="x", bjd=2453000.0, offset=0.0):
    wave = np.linspace(lo, hi, n) + offset
    return EpochData(
        wave=wave,
        flux=1.0 + 0.001 * RNG.standard_normal(n),
        ivar=np.full(n, 1e6),
        bjd=bjd,
        instrument=instrument,
    )


def test_select_region_slices_and_validates():
    ep = _epoch()
    cut = select_region(ep, 4200.0, 4300.0)
    assert cut.wave[0] >= 4200.0 and cut.wave[-1] <= 4300.0
    assert cut.n_pixels < ep.n_pixels and cut.bjd == ep.bjd
    with pytest.raises(ValueError, match="wave_min < wave_max"):
        select_region(ep, 4300.0, 4200.0)
    with pytest.raises(ValueError, match="contains"):
        select_region(ep, 9000.0, 9100.0)


def test_mask_ranges_zeroes_weight_without_deleting_pixels():
    """The distinction that costs a quadratic factor in solver bandwidth if reversed."""
    ep = _epoch()
    masked = mask_ranges(ep, [(4200.0, 4250.0)])
    assert masked.n_pixels == ep.n_pixels
    assert np.array_equal(masked.wave, ep.wave)
    inside = (ep.wave >= 4200.0) & (ep.wave <= 4250.0)
    assert np.all(masked.ivar[inside] == 0.0)
    assert np.all(masked.ivar[~inside] > 0.0)


def test_mask_ranges_rejects_malformed_and_total_masks():
    ep = _epoch()
    with pytest.raises(ValueError, match="min < max"):
        mask_ranges(ep, [(4300.0, 4200.0)])
    with pytest.raises(ValueError, match="every pixel"):
        mask_ranges(ep, [(0.0, 1e6)])


def test_mask_tellurics_is_a_no_op_in_the_blue_and_bites_in_the_red():
    blue = _epoch(lo=4000.0, hi=4600.0)
    assert mask_tellurics(blue) is blue  # no band below 5870 A
    red = _epoch(lo=7500.0, hi=7800.0)
    masked = mask_tellurics(red)
    assert (masked.ivar == 0.0).sum() > 0
    # The O2 A band sits at 7580-7720; padding widens it but must not swallow 7800.
    assert masked.ivar[-1] > 0.0


def test_mask_spikes_removes_cosmics_and_spares_line_cores():
    ep = _epoch(n=4000)
    flux = ep.flux.copy()
    flux[1000] += 0.5  # cosmic ray
    flux[2000:2005] -= 0.5  # a narrow absorption core
    ep = EpochData(wave=ep.wave, flux=flux, ivar=ep.ivar, bjd=ep.bjd)
    cleaned = mask_spikes(ep, threshold=6.0)
    assert cleaned.ivar[1000] == 0.0
    assert np.all(cleaned.ivar[2000:2005] > 0.0)
    both = mask_spikes(ep, threshold=6.0, both_sides=True)
    assert np.all(both.ivar[2000:2005] == 0.0)


# --------------------------------------------------------------- shared wavelength grid


def test_share_wavelength_grid_aligns_sub_pixel_offsets():
    step = 0.03
    base = np.arange(4000.0, 4060.0, step)
    epochs = [
        EpochData(
            wave=base[k:] + 1e-5 * k,
            flux=np.ones(base.size - k),
            ivar=np.ones(base.size - k),
            bjd=2453000.0 + k,
        )
        for k in range(4)
    ]
    aligned = share_wavelength_grid(epochs)
    assert all(e.wave is aligned[0].wave for e in aligned)
    assert len({e.n_pixels for e in aligned}) == 1
    # The flux samples must not have moved: epoch k lost its first k samples only.
    for k, e in enumerate(aligned):
        assert e.flux.size == aligned[0].flux.size
        assert e.bjd == 2453000.0 + k


def test_share_wavelength_grid_preserves_flux_sample_identity():
    """A relabelling, not a resampling: every retained flux value is untouched."""
    step = 0.03
    n = 500
    flux = RNG.standard_normal(n) * 0.01 + 1.0
    a = EpochData(wave=np.arange(n) * step + 4000.0, flux=flux, ivar=np.ones(n), bjd=2453000.0)
    b = EpochData(
        wave=np.arange(n) * step + 4000.0 + 3 * step + 2e-5,
        flux=flux,
        ivar=np.ones(n),
        bjd=2453001.0,
    )
    aligned = share_wavelength_grid([a, b])
    assert np.array_equal(aligned[0].flux, a.flux[3:])
    assert np.array_equal(aligned[1].flux, b.flux[: b.n_pixels - 3])


def test_share_wavelength_grid_refuses_genuinely_different_grids():
    a = _epoch(n=500, lo=4000.0, hi=4600.0)
    b = _epoch(n=500, lo=4000.5, hi=4600.5)  # half an Angstrom = many pixels of mismatch
    with pytest.raises(ValueError, match="genuinely different wavelength solutions"):
        share_wavelength_grid([a, b], atol_kms=0.05)


def test_share_wavelength_grid_keeps_instruments_independent():
    a = _epoch(n=500, instrument="feros")
    b = _epoch(n=500, instrument="feros", offset=1e-5)
    c = _epoch(n=400, lo=6000.0, hi=6200.0, instrument="harps")
    out = share_wavelength_grid([a, b, c])
    assert out[0].wave is out[1].wave
    assert out[2].wave is not out[0].wave
    assert out[2].n_pixels == 400


def test_share_wavelength_grid_rejects_non_overlapping_epochs():
    a = _epoch(n=500, lo=4000.0, hi=4600.0)
    b = _epoch(n=500, lo=5000.0, hi=5600.0)
    with pytest.raises(ValueError, match="not the same spectral window"):
        share_wavelength_grid([a, b])


def test_telluric_bands_are_sane():
    for lo, hi in TELLURIC_BANDS:
        assert 3000.0 < lo < hi < 12000.0
    starts = [lo for lo, _ in TELLURIC_BANDS]
    assert starts == sorted(starts)
