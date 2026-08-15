"""Reading ESO Science Data Products by what they declare, not what they are called (D45).

Thirteen real Phase 3 spectra across seven instruments were dumped column by column to
write this file, and the headline is that **no two collections agree on anything except
the utypes**. The flux column is ``FLUX`` on HARPS, ``FLUX_REDUCED`` on GIRAFFE, and both
at once on X-shooter; the extension is ``SPECTRUM`` except in the Gaia-ESO release, where
it is ``phase3spectrum``; wavelengths are angstrom, Angstrom or nm; and ESO misspells
``Accuracy`` as ``Accurancy`` in ESPRESSO and GIRAFFE products but not in X-shooter.

Four things have to hold, and each one is a way a name-keyed reader gets a confident wrong
answer rather than an error:

**The sky is not the star.** UVES ships ``BGFLUX_REDUCED`` whose UCD is byte-identical to
the UCD on the HARPS *flux* column. Only the utype role separates them, which is why the
role is the key and the UCD is a tie-breaker.

**The error must belong to the flux it weights.** X-shooter carries a calibrated flux in
erg/cm2/s/A beside a raw error in adu. Both are finite and positive, so pairing them across
namespaces is wrong by the whole flux calibration and complains about nothing.

**Air is not vacuum.** The wavelength scale is declared only in the spectral axis's own
UCD, and it is worth 83 km/s — the same order as the semi-amplitudes being measured.

**A flagged pixel is not a measurement.** Nor is a zero uncertainty, which is how these
pipelines write "nothing here" rather than "known exactly".
"""

import warnings

import numpy as np
import pytest

from albireo.data import EpochData
from albireo.preprocess import select_region

pytest.importorskip("astropy")
from astropy.io import fits

from albireo.io import read_dataset, read_spectrum, to_epoch, write_spectra

# La Silla and HR 6819, as the FEROS headers give them.
GEOLON, GEOLAT, GEOELEV = -70.7346, -29.2543, 2335.0
RA, DEC = 274.246199, -56.02876
RNG = np.random.default_rng(45)

# The UCDs and units below are verbatim from real files; naming them keeps the layout
# tables readable as the transcriptions they are.
AIR = "em.wl;obs.atmos"
VACUUM = "em.wl"
RAW_FLUX = "phot.flux.density;em.wl;stat.uncalib"
NET_FLUX = "phot.flux.density;em.wl;src.net;stat.uncalib"
CAL_FLUX = "phot.flux.density;em.wl;src.net;meta.main"
RAW_ERR = "stat.error;phot.flux.density"
CAL_ERR = "stat.error;phot.flux.density;meta.main"
CGS = "erg cm**(-2) s**(-1) angstrom**(-1)"

WAVE_V1 = "Spectrum.Data.SpectralAxis.Value"
FLUX_V1 = "Spectrum.Data.FluxAxis.Value"
ERR_V1 = "Spectrum.Data.FluxAxis.Accuracy.StatError"
WAVE_V2 = "spec:Data.SpectralAxis.Value"
FLUX_V2 = "spec:Data.FluxAxis.Value"
ERR_V2 = "spec:Data.FluxAxis.Accuracy.StatError"
BACKGROUND = "spec:Data.BackgroundModel.Value"
# ESO's own misspelling, exactly as it appears in GIRAFFE and ESPRESSO products.
QUAL_V2 = "spec:Data.FluxAxis.Accurancy.QualityStatus"
# ESO's reduced columns sit in their own namespace beside the calibrated ones.
FLUX_ESO = "eso:Data.FluxAxis.Value"
ERR_ESO = "eso:Data.FluxAxis.Accuracy.StatError"


def _flux(wave):
    """A sloping continuum with two absorption lines, in arbitrary units."""
    continuum = 500.0 * np.exp(-0.002 * (wave - wave[0]))
    lines = 1.0
    for center, width, depth in ((4400.0, 1.2, 0.4), (4471.0, 2.0, 0.3)):
        lines = lines - depth * np.exp(-0.5 * ((wave - center) / width) ** 2)
    return continuum * lines


def _primary(**cards):
    """The primary header every ESO Phase 3 product carries, before the overrides."""
    hdu = fits.PrimaryHDU()
    header = hdu.header
    header["INSTRUME"] = "TESTSPEC"
    header["MJD-OBS"] = 53243.01478576
    header["EXPTIME"] = 149.9999
    header["SPEC_RES"] = 48000.0
    header["CONTNORM"] = False
    header["SPECSYS"] = "BARYCENT"
    header["HIERARCH ESO DRS BARYCORR"] = -21.7078
    header["RA"] = RA
    header["DEC"] = DEC
    header["HIERARCH ESO TEL GEOLON"] = GEOLON
    header["HIERARCH ESO TEL GEOLAT"] = GEOLAT
    header["HIERARCH ESO TEL GEOELEV"] = GEOELEV
    # Only the hyphenated FITS keywords need translating from a Python identifier; mapping
    # every underscore would silently rename SPEC_RES and defeat the test that uses it.
    hyphenated = {"MJD_OBS": "MJD-OBS", "MJD_END": "MJD-END", "DATE_OBS": "DATE-OBS"}
    for key, value in cards.items():
        key = hyphenated.get(key, key)
        if value is None:
            header.remove(key, ignore_missing=True)
        else:
            header[key] = value
    return hdu


def write_sdp(
    path,
    columns,
    *,
    extname="SPECTRUM",
    tmid=53243.01565381,
    n=3000,
    lo=4350.0,
    step=0.05,
    decoy=False,
    n_rows=1,
    **primary_cards,
):
    """Write a Phase 3 spectrum with an arbitrary column layout.

    ``columns`` is a list of ``(name, tutyp, tucd, unit, values)``. ``values`` may be an
    array, or one of the strings this module uses as shorthand for the ways a real product
    fills a column: ``"wave"``, ``"flux"``, ``"err"``, ``"nan"``, ``"zeros"``, ``"good"``.
    """
    wave = lo + step * np.arange(n)
    flux = _flux(wave) + 5.0 * RNG.standard_normal(n)
    shorthand = {
        "wave": wave,
        "flux": flux,
        "err": np.full(n, 5.0),
        "nan": np.full(n, np.nan),
        "zeros": np.zeros(n),
        "good": np.zeros(n, dtype=np.int32),
    }

    hdus = [_primary(**primary_cards)]
    if decoy:
        # A short calibration table that a name-keyed reader would happily read instead.
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="WAVE", format="5D", array=[np.linspace(4000.0, 4100.0, 5)]),
                    fits.Column(name="FLUX", format="5D", array=[np.ones(5)]),
                ],
                name="RESPONSE",
            )
        )

    fits_columns = []
    for name, _tutyp, _tucd, unit, values in columns:
        array = shorthand[values] if isinstance(values, str) else np.asarray(values)
        integer = np.issubdtype(array.dtype, np.integer)
        code = "J" if integer else "D"
        if n_rows > 1:
            fits_columns.append(fits.Column(name=name, format=code, unit=unit, array=array))
        else:
            fits_columns.append(
                fits.Column(name=name, format=f"{array.size}{code}", unit=unit, array=[array])
            )
    table = fits.BinTableHDU.from_columns(fits_columns, name=extname)
    if tmid is not None:
        table.header["TMID"] = tmid
    for index, (_name, tutyp, tucd, _unit, _values) in enumerate(columns, start=1):
        if tutyp is not None:
            table.header[f"TUTYP{index}"] = tutyp
        if tucd is not None:
            table.header[f"TUCD{index}"] = tucd
    hdus.append(table)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return str(path)


# The layouts below are transcriptions of real files, not inventions.
HARPS = [
    ("WAVE", WAVE_V1, AIR, "Angstrom", "wave"),
    ("FLUX", FLUX_V1, RAW_FLUX, "adu", "flux"),
    ("ERR", ERR_V1, RAW_ERR, "adu", "nan"),
]

UVES = [
    ("WAVE", WAVE_V2, AIR, "angstrom", "wave"),
    ("FLUX_REDUCED", FLUX_V2, NET_FLUX, "adu", "flux"),
    ("ERR_REDUCED", ERR_V2, RAW_ERR, "adu", "err"),
    # The trap: the same UCD the HARPS *flux* column carries, on the sky background.
    ("BGFLUX_REDUCED", BACKGROUND, RAW_FLUX, "adu", "zeros"),
]

BLOEM = [
    ("WAVE", WAVE_V2, AIR, "nm", "wave"),
    ("FLUX_REDUCED", FLUX_V2, RAW_FLUX, "adu", "flux"),
    ("ERR_REDUCED", ERR_V2, RAW_ERR, "adu", "err"),
    ("QUAL_REDUCED", QUAL_V2, "meta.code.qual", "", "good"),
]


# -- choosing the column ------------------------------------------------------


def test_the_sky_background_is_never_mistaken_for_the_flux(tmp_path):
    """UVES gives its background column the UCD HARPS gives its flux column."""
    raw = read_spectrum(write_sdp(tmp_path / "uves.fits", UVES))
    assert raw.columns["flux"] == "FLUX_REDUCED", (
        "the background column carries a flux UCD, so only the utype role can exclude it; "
        f"the reader chose {raw.columns['flux']!r}"
    )
    assert np.any(raw.flux != 0.0), "the background column is all zeros; this is it"


def test_the_calibrated_flux_wins_when_a_file_carries_two(tmp_path):
    """X-shooter ships a calibrated FLUX beside a raw FLUX_REDUCED in the eso: namespace."""
    xshooter = [
        ("WAVE", WAVE_V2, AIR, "nm", "wave"),
        ("FLUX", FLUX_V2, CAL_FLUX, CGS, "flux"),
        ("ERR", ERR_V2, CAL_ERR, CGS, "err"),
        ("FLUX_REDUCED", FLUX_ESO, NET_FLUX, "adu", "zeros"),
        ("ERR_REDUCED", ERR_ESO, RAW_ERR, "adu", "err"),
    ]
    raw = read_spectrum(write_sdp(tmp_path / "xshooter.fits", xshooter))
    assert raw.columns["flux"] == "FLUX", "meta.main marks the calibrated column"


def test_the_error_column_is_matched_to_the_flux_it_weights(tmp_path):
    """Pairing a cgs flux with an adu error is wrong by the flux calibration, silently."""
    xshooter = [
        ("WAVE", WAVE_V2, AIR, "nm", "wave"),
        ("FLUX", FLUX_V2, CAL_FLUX, CGS, "flux"),
        # The adu error comes first, so index order alone would pick the wrong one.
        ("ERR_REDUCED", ERR_ESO, RAW_ERR, "adu", "err"),
        ("ERR", ERR_V2, CAL_ERR, CGS, "err"),
    ]
    raw = read_spectrum(write_sdp(tmp_path / "x.fits", xshooter))
    assert raw.columns["err"] == "ERR", (
        "the error must come from the same namespace as the flux, not from whichever "
        "error-shaped column appears first"
    )


def test_a_flux_error_column_is_found_by_name_when_no_utype_says_so(tmp_path):
    """FLUX_ERROR spelled out, with no utype to dispatch on — the Gaia RVS shape.

    Products outside the ESO Phase 3 world carry no IVOA utypes at all, so the whole
    dispatch falls through to the name table, and there `FLUX_ERROR` is a different string
    from `FLUX_ERR`. Missing it is not a missing feature: the reader would report no error
    column and weight the epoch by the scatter it estimates itself, which looks exactly
    like a spectrum whose archive supplied no uncertainties.
    """
    nameless = [
        ("WAVE", "", "", "angstrom", "wave"),
        ("FLUX", "", "", "", "flux"),
        ("FLUX_ERROR", "", "", "", "err"),
    ]
    raw = read_spectrum(write_sdp(tmp_path / "gaia_like.fits", nameless))
    assert raw.columns["err"] == "FLUX_ERROR", (
        "a spelled-out FLUX_ERROR must be recognized; falling through to scatter-estimated "
        "weights would silently replace the archive's uncertainties with an assumption"
    )
    assert raw.err is not None


def test_esos_misspelled_utype_still_names_the_quality_column(tmp_path):
    """ESPRESSO and GIRAFFE write Accurancy where X-shooter writes Accuracy."""
    raw = read_spectrum(write_sdp(tmp_path / "bloem.fits", BLOEM))
    assert raw.columns["quality"] == "QUAL_REDUCED"
    assert raw.quality is not None


def test_the_gaia_eso_extension_name_is_recognized(tmp_path):
    """Everything writes EXTNAME='SPECTRUM' except the Gaia-ESO release."""
    raw = read_spectrum(write_sdp(tmp_path / "ge.fits", HARPS, extname="phase3spectrum"))
    assert raw.n_pixels == 3000


def test_a_calibration_table_does_not_win_over_the_real_spectrum(tmp_path):
    """The first table with columns called WAVE and FLUX is not necessarily the spectrum."""
    raw = read_spectrum(write_sdp(tmp_path / "decoy.fits", HARPS, decoy=True))
    assert raw.n_pixels == 3000, (
        f"read a {raw.n_pixels}-pixel table; the five-row RESPONSE decoy won on column names"
    )


def test_a_file_with_no_utypes_at_all_still_reads(tmp_path):
    """The name tables survive as a fallback, because not every product is an ESO one."""
    plain = [
        ("WAVE", None, None, "angstrom", "wave"),
        ("FLUX", None, None, "adu", "flux"),
        ("ERR", None, None, "adu", "err"),
    ]
    raw = read_spectrum(write_sdp(tmp_path / "plain.fits", plain))
    assert raw.n_pixels == 3000
    assert raw.err is not None
    assert raw.wave_medium == "unknown", "with no UCD the file does not say, and neither may we"


# -- the wavelength scale -----------------------------------------------------


def test_air_and_vacuum_come_from_the_wave_columns_own_ucd(tmp_path):
    """Reading TUCD1 by index is right only while the wave column happens to be first."""
    swapped = [
        ("FLUX", FLUX_V2, "phot.flux.density", "adu", "flux"),
        ("WAVE", WAVE_V2, AIR, "angstrom", "wave"),
    ]
    assert read_spectrum(write_sdp(tmp_path / "s.fits", swapped)).wave_medium == "air", (
        "TUCD1 here describes the flux; a fixed index would report vacuum for an air "
        "spectrum, which is an 83 km/s error that nothing downstream can detect"
    )


def test_a_vacuum_spectrum_is_declared_vacuum(tmp_path):
    espresso = [
        ("WAVE", WAVE_V2, "em.wl;meta.main", "angstrom", "wave"),
        ("FLUX", FLUX_V2, "phot.flux.density;meta.main", "adu", "flux"),
    ]
    assert read_spectrum(write_sdp(tmp_path / "e.fits", espresso)).wave_medium == "vacuum"


def test_nanometre_wavelengths_are_converted_to_angstrom(tmp_path):
    """GIRAFFE, X-shooter and the Gaia-ESO release all deliver nm."""
    raw = read_spectrum(write_sdp(tmp_path / "nm.fits", BLOEM, lo=435.0, step=0.005))
    assert 4340.0 < raw.wave[0] < 4360.0, f"wavelengths came out at {raw.wave[0]:.1f}"


def test_the_wavelength_scale_reaches_the_epoch(tmp_path):
    """The whole point of reading it: EpochData.medium is what Dataset validates."""
    epoch = to_epoch(read_spectrum(write_sdp(tmp_path / "a.fits", UVES)), smooth_angstrom=60.0)
    assert epoch.medium == "air"


def test_trimming_an_epoch_keeps_its_wavelength_scale(tmp_path):
    """Every masking and slicing helper goes through one rebuild, and it must carry it."""
    epoch = EpochData(
        wave=np.linspace(4400.0, 4500.0, 200),
        flux=np.ones(200),
        ivar=np.full(200, 100.0),
        bjd=2453243.5,
        medium="air",
    )
    assert select_region(epoch, 4420.0, 4480.0).medium == "air"


def test_read_dataset_refuses_a_mixture_of_air_and_vacuum(tmp_path):
    write_sdp(tmp_path / "air.fits", UVES)
    vac = [
        ("WAVE", "spec:Data.SpectralAxis.Value", VACUUM, "angstrom", "wave"),
        ("FLUX", "spec:Data.FluxAxis.Value", "phot.flux.density", "adu", "flux"),
    ]
    write_sdp(tmp_path / "vac.fits", vac)
    with pytest.raises(ValueError, match="different wavelength scales"):
        read_dataset(str(tmp_path), region=(4400.0, 4500.0), smooth_angstrom=60.0)


def test_one_warning_per_complaint_not_per_file(tmp_path):
    """51 copies of the same sentence is how a user learns to ignore warnings."""
    for index in range(6):
        write_sdp(tmp_path / f"s{index}.fits", HARPS, tmid=53243.0 + index)
    with pytest.warns(RuntimeWarning) as caught:
        read_dataset(str(tmp_path), region=(4400.0, 4500.0), smooth_angstrom=60.0)
    err_warnings = [w for w in caught if "no finite positive value" in str(w.message)]
    assert len(err_warnings) == 1, f"got {len(err_warnings)} copies of one complaint"
    assert "5 more of the 6 files" in str(err_warnings[0].message)


def test_a_declared_medium_overrides_the_files(tmp_path):
    """After converting by hand, the user must be able to say so."""
    write_sdp(tmp_path / "air.fits", UVES)
    ds = read_dataset(str(tmp_path), medium="vacuum", region=(4400.0, 4500.0), smooth_angstrom=60.0)
    assert ds.epochs[0].medium == "vacuum"


# -- weights and bad pixels ---------------------------------------------------


def test_a_flagged_pixel_gets_no_weight(tmp_path):
    """X-shooter flags 17% of its pixels, and their errors look perfectly healthy."""
    quality = np.zeros(3000, dtype=np.int32)
    quality[100:150] = 1
    layout = [*BLOEM[:3], ("QUAL_REDUCED", QUAL_V2, "meta.code.qual", "", quality)]
    epoch = to_epoch(
        read_spectrum(write_sdp(tmp_path / "q.fits", layout, lo=435.0, step=0.005)),
        smooth_angstrom=60.0,
    )
    flagged = (epoch.wave >= 435.5 * 10.0) & (epoch.wave <= 435.745 * 10.0)
    assert flagged.sum() > 0
    assert np.all(epoch.ivar[flagged] == 0.0), (
        "a nonzero quality flag means the pipeline recorded no measurement there"
    )


def test_a_zero_uncertainty_is_not_infinite_precision(tmp_path):
    """It is how these pipelines write 'nothing here'; weighting it as exact is fatal."""
    err = np.full(3000, 5.0)
    err[200:220] = 0.0
    layout = [BLOEM[0], BLOEM[1], (BLOEM[2][0], BLOEM[2][1], BLOEM[2][2], "adu", err)]
    raw = read_spectrum(write_sdp(tmp_path / "z.fits", layout, lo=435.0, step=0.005))
    assert raw.bad_pixels[200:220].all()
    assert not raw.bad_pixels[300:400].any()


def test_an_all_nan_error_column_is_reported_not_swallowed(tmp_path):
    """FEROS and HARPS ship one, and the weights become albireo's assumption instead."""
    with pytest.warns(RuntimeWarning, match="no finite positive value"):
        raw = read_spectrum(write_sdp(tmp_path / "n.fits", HARPS))
    assert raw.err is None
    assert "ERR" in raw.err_source


# -- the header facts ---------------------------------------------------------


def test_packed_sexagesimal_telescope_coordinates_are_decoded(tmp_path):
    """ESO TEL TARG ALPHA = 181703.0 is 18h 17m 03s, not 181703 degrees."""
    path = write_sdp(
        tmp_path / "sx.fits",
        UVES,
        RA=None,
        DEC=None,
        **{"HIERARCH ESO TEL TARG ALPHA": 181703.000, "HIERARCH ESO TEL TARG DELTA": -560143.0},
    )
    packed = read_spectrum(path)
    plain = read_spectrum(write_sdp(tmp_path / "pl.fits", UVES, RA=274.2625, DEC=-56.02861))
    assert abs(packed.bjd - plain.bjd) * 86400.0 < 1.0, (
        "read as degrees the position wraps to somewhere else on the sky, and the "
        "light-travel correction is minutes wrong without any complaint"
    )


def test_a_coadd_refuses_to_invent_a_mid_exposure_time(tmp_path):
    """MJD-OBS + EXPTIME/2 is the middle of one exposure, not of a month-long stack."""
    with pytest.raises(ValueError, match="TELAPSE"):
        read_spectrum(
            write_sdp(tmp_path / "c.fits", UVES, tmid=None, TELAPSE=2.59e6, EXPTIME=900.0)
        )


def test_mjd_end_beats_the_exposure_time_fallback(tmp_path):
    path = write_sdp(tmp_path / "e.fits", UVES, tmid=None, MJD_END=53243.02478576)
    raw = read_spectrum(path)
    assert "MJD-END" in raw.time_source


def test_a_fabricated_barycentric_velocity_says_so(tmp_path):
    """Zero is a real number here, and it is wrong unless the pipeline applied nothing."""
    path = write_sdp(
        tmp_path / "b.fits", UVES, RA=None, DEC=None, **{"HIERARCH ESO DRS BARYCORR": None}
    )
    with pytest.warns(RuntimeWarning, match="fabricated value"):
        raw = read_spectrum(path)
    assert raw.v_bary == 0.0
    assert "assumed 0" in raw.v_bary_source


def test_a_one_character_R_keyword_is_not_a_resolving_power(tmp_path):
    """R = 3.7 would imply a 34,000 km/s line-spread function."""
    path = write_sdp(tmp_path / "r.fits", UVES, SPEC_RES=None, R=3.7)
    assert read_spectrum(path).resolving_power is None


def test_the_raw_specsys_survives_even_when_the_frame_is_aliased(tmp_path):
    """Heliocentric is reported as barycentric; the file's own word is still recorded."""
    with pytest.warns(RuntimeWarning, match="heliocentric"):
        raw = read_spectrum(write_sdp(tmp_path / "h.fits", UVES, SPECSYS="HELIOCEN"))
    assert raw.frame == "barycentric"
    assert raw.specsys == "HELIOCEN"


# -- what the reader refuses to guess -----------------------------------------


def test_a_quality_column_that_never_says_zero_is_ignored_not_inverted(tmp_path):
    """UVES_SQUAD's STATUS runs {-5, 1}; read as 'nonzero is bad' it condemns every pixel."""
    status = np.where(np.arange(3000) % 3 == 0, -5, 1).astype(np.int32)
    layout = [*BLOEM[:3], ("QUAL_REDUCED", QUAL_V2, "meta.code.qual", "", status)]
    with pytest.warns(RuntimeWarning, match="never takes the value 0"):
        raw = read_spectrum(write_sdp(tmp_path / "squad.fits", layout, lo=435.0, step=0.005))
    assert raw.quality is None
    assert "quality" not in raw.columns
    assert not raw.bad_pixels.all(), "467 published UVES_SQUAD products look exactly like this"


def test_a_polarity_free_mask_column_is_not_guessed_at(tmp_path):
    """MASK/FLAG carry no agreed convention, and albireo's own EpochData.mask is True=GOOD."""
    good = np.ones(3000, dtype=np.int32)
    good[100:150] = 0
    layout = [*BLOEM[:3], ("MASK", None, None, "", good)]
    raw = read_spectrum(write_sdp(tmp_path / "m.fits", layout, lo=435.0, step=0.005))
    assert "quality" not in raw.columns, (
        "reading this as nonzero-is-bad keeps exactly the 50 pixels the file rejected "
        "and discards the 2950 it vouched for"
    )


def test_a_frequency_axis_is_refused_rather_than_read_as_angstrom(tmp_path):
    radio = [
        ("FREQ", WAVE_V2, "em.freq", "Hz", "wave"),
        ("FLUX", FLUX_V2, "phot.flux.density", "Jy", "flux"),
    ]
    with pytest.raises(ValueError, match="not a wavelength"):
        read_spectrum(write_sdp(tmp_path / "radio.fits", radio))


def test_a_spectrum_table_with_no_flux_says_so(tmp_path):
    """Falling through to the image reader would return something else in the same file."""
    headless = [("WAVE", WAVE_V2, AIR, "angstrom", "wave")]
    with pytest.raises(ValueError, match="no flux column"):
        read_spectrum(write_sdp(tmp_path / "h.fits", headless))


def test_an_error_column_belonging_to_the_other_flux_is_rejected(tmp_path):
    """A cgs flux weighted by an adu error is wrong by the whole flux calibration."""
    mixed = [
        ("WAVE", WAVE_V2, AIR, "nm", "wave"),
        ("FLUX", FLUX_V2, CAL_FLUX, CGS, "flux"),
        ("ERR_REDUCED", ERR_ESO, RAW_ERR, "adu", "err"),
    ]
    with pytest.warns(RuntimeWarning, match="different flux column"):
        raw = read_spectrum(write_sdp(tmp_path / "mix.fits", mixed))
    assert raw.err is None


def test_a_dead_region_beside_a_live_pad_is_still_refused(tmp_path):
    """The guard must ask about the pixels that survive the trim, not the padded slice."""
    quality = np.zeros(3000, dtype=np.int32)
    wave = 435.0 + 0.005 * np.arange(3000)
    quality[(wave >= 436.0) & (wave <= 437.0)] = 1
    layout = [*BLOEM[:3], ("QUAL_REDUCED", QUAL_V2, "meta.code.qual", "", quality)]
    path = write_sdp(tmp_path / "dead.fits", layout, lo=435.0, step=0.005)
    with pytest.raises(ValueError, match="every pixel in the selected range"):
        to_epoch(read_spectrum(path), region=(4360.0, 4370.0), region_pad_angstrom=8.0)


def test_only_bjd_is_enough_when_the_file_has_no_time(tmp_path):
    """The error for a missing time says to pass bjd=, so passing bjd= has to work."""
    path = write_sdp(tmp_path / "t.fits", UVES, tmid=None, MJD_OBS=None, EXPTIME=None)
    raw = read_spectrum(path, bjd=2453243.5)
    assert raw.bjd == 2453243.5


def test_an_unmodelled_specsys_keeps_the_word_the_file_used(tmp_path):
    with pytest.warns(RuntimeWarning, match="does not model"):
        raw = read_spectrum(write_sdp(tmp_path / "lsr.fits", UVES, SPECSYS="LSRK"))
    assert raw.specsys == "LSRK"
    assert raw.frame == "topocentric"


def test_declaring_the_frame_silences_the_advice_to_declare_the_frame(tmp_path):
    path = write_sdp(tmp_path / "q.fits", UVES, SPECSYS=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        read_spectrum(path, frame="barycentric", v_bary=0.0)


# -- the other table layout ---------------------------------------------------


def test_a_row_per_pixel_table_reads_too(tmp_path):
    """The IVOA layout is one row of arrays; the ordinary one is N rows of scalars."""
    raw = read_spectrum(write_sdp(tmp_path / "rows.fits", UVES, n_rows=2))
    assert raw.n_pixels == 3000


def test_albireo_can_read_what_albireo_wrote(tmp_path):
    """Exported component spectra are a table of scalar columns, and must round-trip."""
    from albireo.grids import LogGrid

    grid = LogGrid.from_wavelength_range(4400.0, 4500.0, 3.0)
    d_hat = np.zeros((1, grid.n))
    d_hat[0, grid.n // 2] = -0.3
    path = write_spectra(tmp_path / "comp.fits", grid, d_hat)
    raw = read_spectrum(str(path), bjd=2453243.5, v_bary=0.0, frame="barycentric")
    assert raw.n_pixels == grid.n
    assert np.isclose(raw.flux.min(), 0.7)


# -- the live archive ---------------------------------------------------------


@pytest.mark.network
def test_a_bloem_identifier_resolves_to_a_gaia_source_id():
    from albireo.archive import resolve_bloem

    star = resolve_bloem("BLOeM 1-001")
    assert star.bloem_id == "1-001"
    assert star.gaia_dr3 == "4690503998385774848", (
        "the Gaia id must survive as an exact decimal string; 809 of the 929 do not "
        "survive a float64 round trip"
    )
    assert star.spectral_type == "B9 Iab"


@pytest.mark.network
def test_the_published_sb2_sample_is_what_comes_back():
    from albireo.archive import bloem_catalogue

    assert len(bloem_catalogue(binary_class="SB2")) == 59


@pytest.mark.network
def test_a_bloem_star_has_its_published_epochs():
    from albireo.archive import bloem_spectra

    records = bloem_spectra("1-002")
    assert 20 <= len(records) <= 40, f"got {len(records)} epochs"
    assert {r.row["obs_collection"] for r in records} == {"GIRAFFE"}
    assert all(str(r.row["proposal_id"]).startswith("112.25R7") for r in records)


@pytest.mark.network
def test_the_follow_up_programme_is_not_swept_in_by_default():
    """115.28A9 observes the same stars at R = 17000 and 23000 in other windows."""
    from albireo.archive import bloem_spectra, resolve_bloem

    star = resolve_bloem("1-002")
    survey = bloem_spectra(star)
    everything = bloem_spectra(star, programme=None)
    assert len(everything) > len(survey)
    assert {r.row["em_res_power"] for r in survey} == {6300.0}
    assert len({r.row["em_res_power"] for r in everything}) > 1
