"""Header conventions of older and non-ESO spectra that :func:`albireo.io.read_spectrum` reads.

Each case below is a header that a reader keyed on the ESO Phase 3 layout misreads without an
error: a logarithmic wavelength axis in one of its three conventions, an IRAF axis with no
dispersion solution, a ``DATE-OBS`` that holds the date but not the time, and site and
target coordinates written as sexagesimal strings.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits

from albireo.io import read_spectrum

# La Silla and HR 6819, as the FEROS headers give them.
GEOLON, GEOLAT, GEOELEV = -70.7346, -29.2543, 2335.0
RA, DEC = 274.246199, -56.02876


def write_image(path, *, n=400, crval=4350.0, cdelt=0.05, ctype="AWAV", **cards):
    """A 1-D image spectrum with a full set of time, site and target keywords."""
    hdu = fits.PrimaryHDU(data=np.linspace(1.0, 0.9, n).astype(np.float32))
    header = hdu.header
    header["CRVAL1"] = crval
    header["CDELT1"] = cdelt
    header["CRPIX1"] = 1.0
    header["CTYPE1"] = ctype
    header["INSTRUME"] = "SOMETHING"
    header["MJD-OBS"] = 53243.0
    header["EXPTIME"] = 600.0
    header["SPECSYS"] = "TOPOCENT"
    header["HIERARCH ESO DRS BARYCORR"] = -21.7
    header["RA"] = RA
    header["DEC"] = DEC
    header["HIERARCH ESO TEL GEOLON"] = GEOLON
    header["HIERARCH ESO TEL GEOLAT"] = GEOLAT
    header["HIERARCH ESO TEL GEOELEV"] = GEOELEV
    for key, value in cards.items():
        key = {"DC_FLAG": "DC-FLAG", "DATE_OBS": "DATE-OBS", "TIME_OBS": "TIME-OBS"}.get(key, key)
        key = {"MJD_OBS": "MJD-OBS", "UT_START": "UT-START"}.get(key, key)
        if value is None:
            header.remove(key, ignore_missing=True)
        else:
            header[key] = value
    hdu.writeto(path, overwrite=True)
    return str(path)


# -- the wavelength axis ------------------------------------------------------


def test_a_linear_axis_reads_as_before(tmp_path):
    raw = read_spectrum(write_image(tmp_path / "lin.fits"))
    assert np.array_equal(raw.wave, 4350.0 + 0.05 * (np.arange(400) + 1.0 - 1.0))


def test_an_iraf_log_linear_axis_reads_as_before(tmp_path):
    """DC-FLAG = 1: CRVAL1 and CDELT1 are in log10 of the wavelength."""
    crval, cdelt = math.log10(4350.0), 5e-6
    raw = read_spectrum(
        write_image(tmp_path / "dc.fits", crval=crval, cdelt=cdelt, ctype="LINEAR", DC_FLAG=1)
    )
    assert np.array_equal(raw.wave, np.power(10.0, crval + cdelt * np.arange(400.0)))


def test_a_fits_log_axis_has_its_reference_value_in_wavelength(tmp_path):
    """FITS Paper III -LOG: lambda = CRVAL1 exp(CDELT1 (p - CRPIX1) / CRVAL1); 10**4350 is inf."""
    raw = read_spectrum(
        write_image(tmp_path / "wl.fits", crval=4350.0, cdelt=0.05, ctype="WAVE-LOG")
    )
    expected = 4350.0 * np.exp(0.05 * np.arange(400.0) / 4350.0)
    np.testing.assert_allclose(raw.wave, expected, rtol=1e-14)
    assert raw.wave_medium == "vacuum"


def test_a_natural_log_axis_is_not_raised_to_the_power_ten(tmp_path):
    """Mercator HERMES writes CTYPE1 = 'log(wavelength)' with natural logarithms."""
    path = write_image(
        tmp_path / "ln.fits", crval=math.log(4350.0), cdelt=5e-6, ctype="log(wavelength)"
    )
    with pytest.warns(RuntimeWarning, match="natural logarithm"):
        raw = read_spectrum(path)
    np.testing.assert_allclose(raw.wave, np.exp(math.log(4350.0) + 5e-6 * np.arange(400.0)))


def test_an_axis_with_no_dispersion_solution_is_refused(tmp_path):
    """IRAF's DC-FLAG = -1 means the axis is in pixels, not wavelength."""
    path = write_image(tmp_path / "px.fits", crval=1.0, cdelt=1.0, ctype="PIXEL", DC_FLAG=-1)
    with pytest.raises(ValueError, match="no dispersion solution"):
        read_spectrum(path)


def test_an_axis_in_an_unstated_unit_is_refused(tmp_path):
    """Metres read as Angstrom give 4e-7 A, which nothing downstream would question."""
    path = write_image(tmp_path / "m.fits", crval=4.35e-7, cdelt=5e-12, ctype="WAVE")
    with pytest.raises(ValueError, match="gives no wavelengths between 100 and 1e"):
        read_spectrum(path)
    assert read_spectrum(path, wave_scale=1e10).wave[0] == pytest.approx(4350.0)


# -- the time -----------------------------------------------------------------


def test_a_date_without_a_time_takes_the_time_from_time_obs(tmp_path):
    """DATE-OBS = 'YYYY-MM-DD' is midnight to astropy; TIME-OBS holds the start time."""
    split = read_spectrum(
        write_image(
            tmp_path / "split.fits", MJD_OBS=None, DATE_OBS="2004-08-28", TIME_OBS="06:00:00"
        )
    )
    joined = read_spectrum(
        write_image(tmp_path / "joined.fits", MJD_OBS=None, DATE_OBS="2004-08-28T06:00:00")
    )
    assert split.bjd == joined.bjd
    assert "DATE-OBS + TIME-OBS + EXPTIME/2" in split.time_source
    assert "DATE-OBS + EXPTIME/2" in joined.time_source


def test_the_old_fits_date_form_is_read_as_the_twentieth_century(tmp_path):
    old = read_spectrum(
        write_image(tmp_path / "old.fits", MJD_OBS=None, DATE_OBS="28/08/98", UT="6:00:00")
    )
    iso = read_spectrum(
        write_image(tmp_path / "iso.fits", MJD_OBS=None, DATE_OBS="1998-08-28T06:00:00")
    )
    assert old.bjd == iso.bjd
    assert "UT" in old.time_source


def test_a_date_without_any_time_is_refused(tmp_path):
    """Read as midnight, the epoch would be up to a day early with no warning."""
    path = write_image(tmp_path / "d.fits", MJD_OBS=None, DATE_OBS="2004-08-28")
    with pytest.raises(ValueError, match="not the time"):
        read_spectrum(path)
    assert read_spectrum(path, bjd=2453245.75).bjd == 2453245.75


# -- the site and the target ----------------------------------------------------


def test_a_sexagesimal_site_is_read(tmp_path):
    """LONGITUD = '-70:44:04.56' is the NOAO/IRAF form; float() of it raises."""
    cards = {
        "HIERARCH ESO TEL GEOLON": None,
        "HIERARCH ESO TEL GEOLAT": None,
        "HIERARCH ESO DRS BARYCORR": None,
    }
    numeric = read_spectrum(
        write_image(tmp_path / "n.fits", LONGITUD=GEOLON, LATITUDE=GEOLAT, **cards)
    )
    text = read_spectrum(
        write_image(tmp_path / "s.fits", LONGITUD="-70:44:04.56", LATITUDE="-29:15:15.48", **cards)
    )
    assert text.v_bary_source == "computed with astropy"
    assert abs(text.bjd - numeric.bjd) * 86400.0 < 1e-3
    assert text.v_bary == pytest.approx(numeric.v_bary, abs=1e-6)


def test_sexagesimal_target_coordinates_are_read_in_hours_and_degrees(tmp_path):
    """RA = '18:16:59.09' is in hours, as NOAO and Lick headers write it."""
    numeric = read_spectrum(write_image(tmp_path / "n.fits"))
    text = read_spectrum(write_image(tmp_path / "s.fits", RA="18:16:59.09", DEC="-56:01:43.5"))
    assert text.time_source.startswith("BJD_TDB")
    assert abs(text.bjd - numeric.bjd) * 86400.0 < 0.5


def test_unreadable_target_keywords_fall_back_to_the_telescope_ones(tmp_path):
    numeric = read_spectrum(write_image(tmp_path / "n.fits"))
    packed = read_spectrum(
        write_image(
            tmp_path / "p.fits",
            RA="unknown",
            DEC="unknown",
            **{"HIERARCH ESO TEL TARG ALPHA": 181659.09, "HIERARCH ESO TEL TARG DELTA": -560143.5},
        )
    )
    assert packed.time_source.startswith("BJD_TDB")
    assert abs(packed.bjd - numeric.bjd) * 86400.0 < 0.5
