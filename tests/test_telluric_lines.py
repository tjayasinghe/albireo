"""Tests for :mod:`albireo.telluric`: the packaged telluric line list and the zero point and
line-spread width measured from it.

The spectra are built from the line list itself, displaced by a known velocity and with
the lines as wide as a known line-spread function makes them, so the measurements have
exact answers. Three things are checked: the size and sign of the shift (the correction
that puts the lines back at rest), the width, and the errors raised for a spectrum that
does not reach the telluric bands.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from albireo.telluric import (
    INTRINSIC_FWHM_KMS,
    TELLURIC_REGIONS,
    TelluricLines,
    telluric_lines,
    telluric_lsf,
    telluric_model,
    telluric_shift,
)

C_KMS = 299_792.458
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def lines_at(lsf_fwhm_kms: float) -> TelluricLines:
    """The packaged list with each line as wide as an instrument of this LSF shows it."""
    lines = telluric_lines()
    return TelluricLines(
        wave=lines.wave,
        depth=lines.depth,
        fwhm_kms=np.full(lines.wave.size, math.hypot(lsf_fwhm_kms, INTRINSIC_FWHM_KMS)),
        centre_err_kms=lines.centre_err_kms,
        isolated=lines.isolated,
        band=lines.band,
    )


def spectrum(shift_kms=0.0, lines=None, *, step_kms=1.0, lo=6250.0, hi=7350.0, noise=0.0):
    wave = np.exp(np.arange(math.log(lo), math.log(hi), step_kms / C_KMS))
    flux = 1.0 + telluric_model(wave, lines, shift_kms=shift_kms)
    if noise:
        flux = flux + noise * np.random.default_rng(5).standard_normal(wave.size)
    return [(wave, flux, np.ones(wave.size, dtype=bool))]


def test_the_packaged_list():
    lines = telluric_lines()
    assert len(lines) == 183
    assert int((lines.band == "O2").sum()) == 72
    assert set(lines.band) == {"O2", "H2O"}
    assert np.all(np.diff(lines.wave) > 0)
    assert 6276.0 < lines.wave[0] and lines.wave[-1] < 7411.0
    assert len(lines.lsf_lines()) == 14
    assert "telluric_lines_air.csv" in lines.source
    assert telluric_lines() is lines, "read once"
    for lo, hi in TELLURIC_REGIONS.values():
        assert ((lines.wave > lo) & (lines.wave < hi)).sum() >= 5


def test_the_model_is_a_sum_of_lines():
    lines = telluric_lines()
    assert np.all(telluric_model(np.linspace(5000.0, 6200.0, 1000)) == 0.0)
    wave = np.exp(np.arange(math.log(6860.0), math.log(6960.0), 0.25 / C_KMS))
    model = telluric_model(wave)
    assert model.min() >= -0.98
    deepest = int(np.argmax(lines.depth))
    near = np.abs(wave / lines.wave[deepest] - 1.0) * C_KMS < 3.0
    assert model[near].min() == pytest.approx(-min(lines.depth[deepest], 0.98), abs=2e-3)
    moved = telluric_model(wave, shift_kms=10.0)
    lag = np.argmin(moved) - np.argmin(model)
    assert lag * 0.25 == pytest.approx(10.0, abs=0.5)


@pytest.mark.parametrize("displacement", [2.5, -4.0, 31.0])
def test_the_shift_puts_the_lines_back_at_rest(displacement):
    found = telluric_shift(spectrum(displacement), lsf_sigma_kms=7.3 / FWHM_PER_SIGMA)
    assert found.v_kms == pytest.approx(-displacement, abs=0.02)
    assert found.n_regions == len(TELLURIC_REGIONS)
    for v, r in found.regions.values():
        assert v == pytest.approx(-displacement, abs=0.05) and r > 0.9


def test_the_shift_of_a_narrower_noisy_spectrum():
    lsf = 4.2
    segments = spectrum(1.7, lines_at(lsf), step_kms=0.9, noise=0.01)
    found = telluric_shift(segments, lsf_sigma_kms=lsf / FWHM_PER_SIGMA)
    assert found.v_kms == pytest.approx(-1.7, abs=0.05)
    assert found.err_kms < 0.05


def test_the_line_spread_width():
    for lsf in (3.9, 5.6, 8.0):
        segments = spectrum(-1.0, lines_at(lsf), step_kms=0.9)
        fwhm, n = telluric_lsf(segments, shift_kms=1.0)
        assert fwhm == pytest.approx(lsf, abs=0.05)
        assert n == 14
    assert telluric_lsf(spectrum(0.0, lo=7000.0, hi=7300.0)) is None, "no O2 line inside"


def test_a_spectrum_without_telluric_bands_is_refused():
    wave = np.linspace(5000.0, 5500.0, 20_000)
    segments = [(wave, np.ones(wave.size), np.ones(wave.size, dtype=bool))]
    with pytest.raises(ValueError, match="no telluric region"):
        telluric_shift(segments, lsf_sigma_kms=2.0)
    flat = spectrum(0.0)
    flat[0] = (flat[0][0], np.ones(flat[0][0].size), flat[0][2])
    with pytest.raises(ValueError, match="correlation peak"):
        telluric_shift(flat, lsf_sigma_kms=2.0)
