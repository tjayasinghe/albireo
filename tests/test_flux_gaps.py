"""Detector gaps are not measurements of zero flux.

Found on real HARPS spectra of AI Phoenicis. The instrument's two CCDs leave a 32.9 A hole
at 5304.67-5337.61 A which the pipeline fills with exact zeros. Nothing marks them: the
pixels are finite, the quality column is absent, and because HARPS ships no error array the
inverse variance is estimated from the local scatter — which across a flat run of zeros is
*small*, so the gap arrived weighted like good data. A 100 A analysis window placed across
it was 33% zeros at full weight, and disentangling it produced component spectra with
negative flux.

The rule is deliberately about *runs*, not about any non-positive pixel:
`RawSpectrum.bad_pixels` declines to treat zero flux as missing, and that is right, because
a single zero can be a saturated core or a clipped cosmic ray. Eight in a row cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from albireo.data import EpochData
from albireo.preprocess import mask_flux_gaps


def _epoch(flux, wave=None):
    flux = np.asarray(flux, dtype=float)
    n = flux.size
    wave = np.linspace(5300.0, 5400.0, n) if wave is None else np.asarray(wave, dtype=float)
    return EpochData(wave=wave, flux=flux, ivar=np.ones(n), bjd=2458000.0, instrument="X")


def test_a_run_of_zeros_is_zero_weighted():
    flux = np.ones(200)
    flux[80:120] = 0.0
    out = mask_flux_gaps(_epoch(flux), warn=False)
    ivar = np.asarray(out.ivar)
    assert np.all(ivar[80:120] == 0.0)
    assert np.all(ivar[:80] > 0.0) and np.all(ivar[120:] > 0.0)


def test_an_isolated_zero_is_left_alone():
    """One zero can be a saturated core or a clipped cosmic ray. Only runs are gaps."""
    flux = np.ones(200)
    flux[57] = 0.0
    out = mask_flux_gaps(_epoch(flux), warn=False)
    np.testing.assert_array_equal(np.asarray(out.ivar), np.ones(200))


def test_the_run_length_threshold_is_honoured():
    flux = np.ones(200)
    flux[10:14] = 0.0  # 4 pixels
    assert np.all(np.asarray(mask_flux_gaps(_epoch(flux), warn=False).ivar) == 1.0)
    out = mask_flux_gaps(_epoch(flux), min_run=4, warn=False)
    assert np.all(np.asarray(out.ivar)[10:14] == 0.0)


def test_negative_flux_counts_too():
    """A pipeline that oversubtracts a background writes negatives, not zeros."""
    flux = np.ones(120)
    flux[40:60] = -0.02
    out = mask_flux_gaps(_epoch(flux), warn=False)
    assert np.all(np.asarray(out.ivar)[40:60] == 0.0)


def test_several_gaps_are_all_caught():
    flux = np.ones(300)
    flux[20:40] = 0.0
    flux[100:130] = 0.0
    flux[250:280] = 0.0
    ivar = np.asarray(mask_flux_gaps(_epoch(flux), warn=False).ivar)
    for lo, hi in ((20, 40), (100, 130), (250, 280)):
        assert np.all(ivar[lo:hi] == 0.0), (lo, hi)
    assert ivar.sum() == 300 - 20 - 30 - 30


def test_it_warns_and_names_the_wavelengths():
    """Losing a third of a window must never be silent."""
    flux = np.ones(200)
    flux[100:150] = 0.0
    wave = np.linspace(5300.0, 5400.0, 200)
    with pytest.warns(RuntimeWarning, match="non-positive flux"):
        mask_flux_gaps(_epoch(flux, wave))
    with pytest.warns(RuntimeWarning, match=r"5350\.\d+-5374\.\d+ A"):
        mask_flux_gaps(_epoch(flux, wave))


def test_a_clean_spectrum_is_returned_untouched():
    flux = 1.0 - 0.3 * np.exp(-0.5 * ((np.arange(200) - 100) / 4.0) ** 2)
    ep = _epoch(flux)
    assert mask_flux_gaps(ep, warn=False) is ep


def test_an_all_gap_epoch_raises_rather_than_returning_nothing():
    with pytest.raises(ValueError, match="every pixel"):
        mask_flux_gaps(_epoch(np.zeros(50)), warn=False)


def test_min_run_below_two_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        mask_flux_gaps(_epoch(np.ones(10)), min_run=1)


def test_already_masked_pixels_stay_masked():
    flux = np.ones(100)
    flux[50:70] = 0.0
    ep = _epoch(flux)
    ivar = np.asarray(ep.ivar).copy()
    ivar[:10] = 0.0
    ep = EpochData(wave=ep.wave, flux=ep.flux, ivar=ivar, bjd=ep.bjd, instrument=ep.instrument)
    out = np.asarray(mask_flux_gaps(ep, warn=False).ivar)
    assert np.all(out[:10] == 0.0) and np.all(out[50:70] == 0.0)
