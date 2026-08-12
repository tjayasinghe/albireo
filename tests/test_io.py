"""Tests for :mod:`albireo.io` — FITS in, :class:`~albireo.data.Dataset` out.

The fixtures below reproduce the two container layouts the reader claims to handle, and
in particular the exact shape of an ESO Phase-3 FEROS spectrum: a one-row binary table
with array columns, ``TMID`` in the *extension* header rather than the primary,
``SPECSYS='BARYCENT'``, ``CONTNORM=False``, and an ``ERR`` column that is entirely
``NaN``. Every one of those was a place the first version of this reader could have got
a velocity wrong without saying so.
"""

from __future__ import annotations

import numpy as np
import pytest

from albireo.data import Dataset

pytest.importorskip("astropy")
from astropy.io import fits

from albireo.io import RawSpectrum, read_dataset, read_spectrum, to_epoch

# La Silla, as the FEROS headers give it.
GEOLON, GEOLAT, GEOELEV = -70.7346, -29.2543, 2335.0
# HR 6819.
RA, DEC = 274.246199, -56.02876
RNG = np.random.default_rng(7)


def _flux(wave, decay=0.002):
    """A sloping continuum with a couple of absorption lines, in arbitrary units."""
    continuum = 500.0 * np.exp(-decay * (wave - wave[0]))
    lines = 1.0
    for center, width, depth in ((4400.0, 1.2, 0.4), (4471.0, 2.0, 0.3)):
        lines = lines - depth * np.exp(-0.5 * ((wave - center) / width) ** 2)
    return continuum * lines


def write_eso_phase3(
    path,
    *,
    mjd_obs=53243.01478576,
    tmid=53243.01565381,
    barycorr=-21.7078,
    specsys="BARYCENT",
    n=6000,
    lo=4350.0,
    step=0.03,
    err="nan",
    unit="angstrom",
    with_coords=True,
):
    """Write a file shaped like an ESO Phase-3 1-D spectrum."""
    wave = lo + step * np.arange(n)
    flux = _flux(wave) + 5.0 * RNG.standard_normal(n)
    scale = {"angstrom": 1.0, "nm": 0.1}[unit]

    primary = fits.PrimaryHDU()
    hdr = primary.header
    hdr["INSTRUME"] = "FEROS"
    hdr["MJD-OBS"] = mjd_obs
    hdr["EXPTIME"] = 149.9999
    hdr["SPEC_RES"] = 48000.0
    hdr["CONTNORM"] = False
    if specsys is not None:
        hdr["SPECSYS"] = specsys
    if barycorr is not None:
        hdr["HIERARCH ESO DRS BARYCORR"] = barycorr
    if with_coords:
        hdr["RA"] = RA
        hdr["DEC"] = DEC
        hdr["HIERARCH ESO TEL GEOLON"] = GEOLON
        hdr["HIERARCH ESO TEL GEOLAT"] = GEOLAT
        hdr["HIERARCH ESO TEL GEOELEV"] = GEOELEV

    if err == "nan":
        err_col = np.full(n, np.nan, dtype=np.float32)
    elif err is None:
        err_col = None
    else:
        err_col = np.full(n, float(err), dtype=np.float32)

    columns = [
        fits.Column(name="WAVE", format=f"{n}D", unit=unit, array=[wave * scale]),
        fits.Column(name="FLUX", format=f"{n}E", unit="adu", array=[flux]),
    ]
    if err_col is not None:
        columns.append(fits.Column(name="ERR", format=f"{n}E", unit="adu", array=[err_col]))
    table = fits.BinTableHDU.from_columns(columns, name="SPECTRUM")
    table.header["TMID"] = tmid  # the extension header, as ESO writes it
    table.header["TUCD1"] = "em.wl;obs.atmos"
    fits.HDUList([primary, table]).writeto(path, overwrite=True)
    return path


def write_wcs_image(path, *, n=4000, crval=4350.0, cdelt=0.05):
    """Write a classic IRAF-style 1-D image spectrum."""
    wave = crval + cdelt * np.arange(n)
    hdu = fits.PrimaryHDU(data=_flux(wave).astype(np.float32))
    hdu.header["CRVAL1"] = crval
    hdu.header["CDELT1"] = cdelt
    hdu.header["CRPIX1"] = 1.0
    hdu.header["CTYPE1"] = "AWAV"
    hdu.header["CUNIT1"] = "Angstrom"
    hdu.header["INSTRUME"] = "SOMETHING"
    hdu.header["MJD-OBS"] = 53243.0
    hdu.header["EXPTIME"] = 600.0
    hdu.header["SPECSYS"] = "TOPOCENT"
    hdu.header["RA"] = RA
    hdu.header["DEC"] = DEC
    hdu.header["HIERARCH ESO TEL GEOLON"] = GEOLON
    hdu.header["HIERARCH ESO TEL GEOLAT"] = GEOLAT
    hdu.header["HIERARCH ESO TEL GEOELEV"] = GEOELEV
    hdu.writeto(path, overwrite=True)
    return path


# ----------------------------------------------------------------------------- reading


def test_reads_an_eso_phase3_spectrum(tmp_path):
    raw = read_spectrum(write_eso_phase3(tmp_path / "a.fits"))
    assert isinstance(raw, RawSpectrum)
    assert raw.instrument == "FEROS"
    assert raw.frame == "barycentric"
    assert raw.v_bary == pytest.approx(-21.7078)
    assert raw.resolving_power == 48000.0
    assert raw.wave_medium == "air"
    assert raw.continuum_normalized is False
    assert raw.n_pixels == 6000
    assert raw.wave[0] == pytest.approx(4350.0)
    assert raw.err is None, "an all-NaN ERR column must be reported as absent"
    assert "BJD_TDB" in raw.time_source and "TMID" in raw.time_source
    assert "FEROS" in raw.summary()


def test_lsf_sigma_from_resolving_power():
    """R quotes a FWHM; sigma is smaller by 2 sqrt(2 ln 2). Getting it backwards is 2.35x."""
    raw = RawSpectrum(
        wave=np.array([1.0, 2.0]),
        flux=np.array([1.0, 1.0]),
        err=None,
        bjd=0.0,
        v_bary=0.0,
        frame="barycentric",
        instrument="x",
        resolving_power=48000.0,
    )
    assert raw.lsf_sigma_kms == pytest.approx(2.6523, abs=1e-3)
    assert (
        RawSpectrum(
            wave=np.array([1.0, 2.0]),
            flux=np.array([1.0, 1.0]),
            err=None,
            bjd=0.0,
            v_bary=0.0,
            frame="barycentric",
            instrument="x",
        ).lsf_sigma_kms
        is None
    )


def test_time_is_bjd_tdb_at_mid_exposure(tmp_path):
    """The Roemer delay is up to 8.3 min and is a function of date, so it cannot average out."""
    raw = read_spectrum(write_eso_phase3(tmp_path / "a.fits", tmid=53243.01565381))
    naive_jd = 53243.01565381 + 2400000.5
    offset_seconds = (raw.bjd - naive_jd) * 86400.0
    assert abs(offset_seconds) > 60.0, "no barycentric correction appears to have been applied"
    assert abs(offset_seconds) < 600.0, "correction exceeds the maximum possible light-travel time"


def test_mid_exposure_is_used_when_tmid_is_absent(tmp_path):
    path = write_eso_phase3(tmp_path / "a.fits")
    with fits.open(path, mode="update") as hdul:
        del hdul[1].header["TMID"]
    raw = read_spectrum(path)
    assert "MJD-OBS + EXPTIME/2" in raw.time_source


def test_computed_barycentric_velocity_agrees_with_the_pipeline_keyword(tmp_path):
    """astropy's correction and ESO's BARYCORR must agree in sign as well as size."""
    with_key = read_spectrum(write_eso_phase3(tmp_path / "a.fits", barycorr=-21.7078))
    without = read_spectrum(write_eso_phase3(tmp_path / "b.fits", barycorr=None))
    assert without.v_bary == pytest.approx(with_key.v_bary, abs=0.1)


def test_missing_specsys_warns_and_assumes_topocentric(tmp_path):
    path = write_eso_phase3(tmp_path / "a.fits", specsys=None)
    with pytest.warns(RuntimeWarning, match="no SPECSYS"):
        raw = read_spectrum(path)
    assert raw.frame == "topocentric"


def test_heliocentric_specsys_warns_but_is_accepted(tmp_path):
    path = write_eso_phase3(tmp_path / "a.fits", specsys="HELIOCEN")
    with pytest.warns(RuntimeWarning, match="heliocentric"):
        raw = read_spectrum(path)
    assert raw.frame == "barycentric"


def test_missing_coordinates_warns_and_falls_back_to_jd_utc(tmp_path):
    path = write_eso_phase3(tmp_path / "a.fits", with_coords=False)
    with pytest.warns(RuntimeWarning, match="cannot be put on the barycentre"):
        raw = read_spectrum(path)
    assert "no barycentric correction" in raw.time_source


def test_nanometre_wavelengths_are_converted(tmp_path):
    raw = read_spectrum(write_eso_phase3(tmp_path / "a.fits", unit="nm"))
    assert raw.wave[0] == pytest.approx(4350.0, rel=1e-9)


def test_reads_a_wcs_image_spectrum(tmp_path):
    raw = read_spectrum(write_wcs_image(tmp_path / "img.fits"))
    assert raw.n_pixels == 4000
    assert raw.wave[0] == pytest.approx(4350.0)
    assert raw.frame == "topocentric"
    assert raw.instrument == "SOMETHING"


def test_unreadable_file_raises_a_useful_error(tmp_path):
    path = tmp_path / "empty.fits"
    fits.PrimaryHDU().writeto(path)
    with pytest.raises(ValueError, match="no 1-D spectrum found"):
        read_spectrum(path)


def test_overrides_win_over_the_header(tmp_path):
    path = write_eso_phase3(tmp_path / "a.fits")
    raw = read_spectrum(
        path, instrument="mine", frame="topocentric", bjd=1.0, v_bary=2.0, resolving_power=3.0
    )
    assert (raw.instrument, raw.frame, raw.bjd, raw.v_bary, raw.resolving_power) == (
        "mine",
        "topocentric",
        1.0,
        2.0,
        3.0,
    )


# ------------------------------------------------------------------------ to_epoch


def test_to_epoch_normalizes_and_estimates_ivar(tmp_path):
    raw = read_spectrum(write_eso_phase3(tmp_path / "a.fits"))
    epoch = to_epoch(raw, region=(4400.0, 4500.0), smooth_angstrom=60.0)
    assert 4400.0 <= epoch.wave[0] and epoch.wave[-1] <= 4500.0
    good = epoch.good
    assert good.sum() > 0.9 * epoch.n_pixels
    assert np.median(epoch.flux[good]) == pytest.approx(1.0, abs=0.05)
    assert np.all(epoch.ivar[good] > 0)
    assert epoch.v_bary == pytest.approx(raw.v_bary)


def test_to_epoch_uses_a_real_error_column_when_there_is_one(tmp_path):
    raw = read_spectrum(write_eso_phase3(tmp_path / "a.fits", err=5.0))
    assert raw.err is not None
    epoch = to_epoch(raw, region=(4400.0, 4500.0), smooth_angstrom=60.0)
    # sigma_norm = err / continuum, and the continuum is ~400 ADU here.
    sigma = 1.0 / np.sqrt(epoch.ivar[epoch.good])
    assert 0.005 < np.median(sigma) < 0.05


def test_to_epoch_region_outside_the_spectrum_raises(tmp_path):
    raw = read_spectrum(write_eso_phase3(tmp_path / "a.fits"))
    with pytest.raises(ValueError, match="contains"):
        to_epoch(raw, region=(9000.0, 9100.0))


def test_to_epoch_masks_extra_ranges_and_tellurics(tmp_path):
    raw = read_spectrum(write_eso_phase3(tmp_path / "a.fits"))
    epoch = to_epoch(raw, region=(4400.0, 4500.0), smooth_angstrom=60.0, mask=[(4440.0, 4460.0)])
    inside = (epoch.wave >= 4440.0) & (epoch.wave <= 4460.0)
    assert np.all(epoch.ivar[inside] == 0.0)


# ------------------------------------------------------------------------ read_dataset


def test_read_dataset_builds_a_sorted_dataset(tmp_path):
    times = [53243.5, 53200.5, 53280.5]
    for k, t in enumerate(times):
        write_eso_phase3(tmp_path / f"s{k}.fits", mjd_obs=t, tmid=t + 0.001)
    ds = read_dataset(
        str(tmp_path / "*.fits"),
        instrument="FEROS",
        region=(4400.0, 4500.0),
        smooth_angstrom=60.0,
    )
    assert isinstance(ds, Dataset)
    assert len(ds) == 3
    assert ds.frame == "barycentric"
    assert list(ds.bjd) == sorted(ds.bjd)
    assert ds.instruments == ("FEROS",)


def test_read_dataset_accepts_a_directory(tmp_path):
    write_eso_phase3(tmp_path / "s0.fits")
    ds = read_dataset(str(tmp_path), region=(4400.0, 4500.0), smooth_angstrom=60.0)
    assert len(ds) == 1


def test_read_dataset_rejects_mixed_frames(tmp_path):
    write_eso_phase3(tmp_path / "a.fits", specsys="BARYCENT")
    write_eso_phase3(tmp_path / "b.fits", specsys="TOPOCENT")
    with pytest.raises(ValueError, match="different wavelength frames"):
        read_dataset(str(tmp_path / "*.fits"), region=(4400.0, 4500.0), smooth_angstrom=60.0)


def test_read_dataset_with_no_matches_raises(tmp_path):
    with pytest.raises(ValueError, match="no files matched"):
        read_dataset(str(tmp_path / "nothing*.fits"))
