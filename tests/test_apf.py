"""Tests for :mod:`albireo.apf`, the reader of APF exposures extracted by the CPS pipeline.

The exposures are synthetic. A 79 x 4608 image of counts is built from a wavelength
solution that follows the grating equation (``m * lambda`` a smooth function of the pixel),
a blaze, the packaged telluric line list displaced by a known drift and broadened to a
known line-spread function, optional stellar lines, and the reader's own noise model. The
tests check that the reader recovers the drift and the line-spread width it was given,
merges the orders without resampling, masks the iodine band and the sky lines, and that
the refinement of a distorted solution recovers the solution the exposure was made with.
"""

from __future__ import annotations

import math
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("astropy")
pytest.importorskip("scipy")
from astropy.io import fits

from albireo import apf
from albireo.io import read_raw_spectra, read_spectrum
from albireo.telluric import INTRINSIC_FWHM_KMS, TelluricLines, telluric_lines, telluric_model

C_KMS = 299_792.458
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
N_ROWS, N_PIX = apf.APF_SHAPE

# m * lambda = K + D t + E t^2 with t the pixel scaled to [-1, 1]: row 0 (order 124) starts
# at 3730 A and row 78 (order 46) ends at 10200 A, as on the instrument.
K_GRATING, D_GRATING, E_GRATING = 465_860.0, 3_340.0, -40.0

# The header cards that time an exposure, for Vega (J2000) on a June night. The
# photon-weighted midpoint is 30 s after the middle of the 300 s exposure.
HEADER_TIME = {
    "THEMIDPT": "2025-06-15T08:00:00.000",
    "DATE-OBS": "2025-06-15T07:57:00.00",
    "EXPTIME": 300,
    "RA": "18:36:56.3",
    "DEC": "+38:47:01",
}
# BJD_TDB and the barycentric correction (km/s) that astropy gives for them at the APF site.
# Earth's orbital velocity projected on Vega's direction is about +5.1 km/s in mid-June.
EXPECTED_BJD = 2460841.8367367
EXPECTED_V_BARY = 5.18320


def grating_solution(k=K_GRATING, d=D_GRATING, e=E_GRATING) -> np.ndarray:
    t = (np.arange(N_PIX, dtype=float) - (N_PIX - 1) / 2.0) / ((N_PIX - 1) / 2.0)
    orders = 124.0 - np.arange(N_ROWS, dtype=float)
    return (k + d * t[None, :] + e * t[None, :] ** 2) / orders[:, None]


SOLUTION = grating_solution()


def telluric_lines_at(lsf_fwhm_kms: float) -> TelluricLines:
    """The packaged list with every line as wide as an instrument of this LSF shows it."""
    lines = telluric_lines()
    width = math.hypot(lsf_fwhm_kms, INTRINSIC_FWHM_KMS)
    return TelluricLines(
        wave=lines.wave,
        depth=lines.depth,
        fwhm_kms=np.full(lines.wave.size, width),
        centre_err_kms=lines.centre_err_kms,
        isolated=lines.isolated,
        band=lines.band,
        source="test",
    )


def stellar_lines(lo=5500.0, hi=7600.0, n=4000, seed=11):
    """A rest-frame spectrum with ``n`` random absorption lines, on a 0.25 km/s grid."""
    rng = np.random.default_rng(seed)
    wave = np.exp(np.arange(math.log(lo), math.log(hi), 0.25 / C_KMS))
    deviation = np.zeros(wave.size)
    centres = rng.uniform(lo + 5.0, hi - 5.0, n)
    depths = rng.uniform(0.05, 0.6, n)
    sigma_kms = 3.0
    for centre, depth in zip(centres, depths, strict=True):
        i0, i1 = np.searchsorted(wave, centre * (1.0 + np.array([-6.0, 6.0]) * sigma_kms / C_KMS))
        v = (wave[i0:i1] / centre - 1.0) * C_KMS
        deviation[i0:i1] -= depth * np.exp(-0.5 * (v / sigma_kms) ** 2)
    return wave, np.maximum(deviation, -0.95)


def write_exposure(
    path,
    true_wave: np.ndarray,
    *,
    lsf_fwhm_kms: float = 4.2,
    star=None,
    v_star_kms: float = 0.0,
    decker: str = "N (0.50:8.0)",
    iodine: bool = False,
    peak_counts: float = 30_000.0,
    telluric: bool = True,
    seed: int = 3,
):
    """Write a synthetic CPS exposure whose pixels have the wavelengths ``true_wave``."""
    rng = np.random.default_rng(seed)
    x = np.arange(N_PIX, dtype=float)
    blaze = peak_counts * np.exp(-0.5 * ((x - 2200.0) / 1500.0) ** 2) + 1500.0
    transmission = 1.0 + telluric_model(true_wave, telluric_lines_at(lsf_fwhm_kms)) * telluric
    spectrum = np.ones_like(true_wave)
    if star is not None:
        star_wave, star_dev = star
        rest = true_wave / (1.0 + v_star_kms / C_KMS)
        spectrum = 1.0 + np.interp(rest, star_wave, star_dev, left=0.0, right=0.0)
    counts = blaze[None, :] * transmission * spectrum
    counts = counts + rng.standard_normal(counts.shape) * np.sqrt((counts + 359.5) / 0.884)
    hdu = fits.PrimaryHDU(counts.astype(np.float32))
    header = hdu.header
    header["VERSION"] = "apf"
    header["OBSNUM"] = 10420
    header["DECKRNAM"] = decker
    header["ICELNAM"] = "In" if iodine else "Out"
    header["TOBJECT"] = "SYNTH"
    header["OBJECT"] = "SYNTH"
    for key, value in HEADER_TIME.items():
        header[key] = value
    hdu.writeto(path, overwrite=True)
    return path


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """A solution file and three exposures: drifted, through iodine, and a line-rich star."""
    root = tmp_path_factory.mktemp("apf")
    solution_path = root / "apf_wave_test.fits"
    apf.ApfWavelengths(SOLUTION, source="synthetic grating").save(solution_path)
    drifted = write_exposure(root / "rsyn.101.fits", SOLUTION * (1.0 + 1.7 / C_KMS))
    iodine = write_exposure(root / "rsyn.102.fits", SOLUTION, iodine=True, seed=4)
    return {"root": root, "solution": solution_path, "drifted": drifted, "iodine": iodine}


@pytest.fixture(scope="module")
def drifted_spectrum(files):
    exposure = apf.read_apf_exposure(files["drifted"])
    return exposure, apf.extract(exposure, files["solution"])


# ---------------------------------------------------------------------------
# 1. solutions and exposures
# ---------------------------------------------------------------------------


def test_a_solution_is_validated_and_round_trips(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        apf.ApfWavelengths(SOLUTION[:-1])
    with pytest.raises(ValueError, match="increasing"):
        apf.ApfWavelengths(SOLUTION[:, ::-1])
    with pytest.raises(ValueError, match="Angstrom"):
        apf.ApfWavelengths(SOLUTION / 10.0)
    solution = apf.ApfWavelengths(SOLUTION, source="synthetic grating")
    path = solution.save(tmp_path / "solution.fits")
    loaded = apf.ApfWavelengths.load(path)
    np.testing.assert_array_equal(loaded.wave, SOLUTION)
    assert loaded.source == "synthetic grating"
    assert fits.getheader(path)["MEDIUM"] == "air"
    np.testing.assert_allclose(solution.shifted(3.0).wave / SOLUTION - 1.0, 3.0 / C_KMS)
    assert apf.ApfWavelengths.order(0) == 124 and apf.ApfWavelengths.order(78) == 46
    with pytest.raises(OSError):
        solution.save(path)


def test_an_exposure_is_recognised_by_its_shape_and_header(files, tmp_path):
    exposure = apf.read_apf_exposure(files["drifted"])
    assert exposure.counts.shape == apf.APF_SHAPE
    assert exposure.decker == "N"
    assert not exposure.iodine_in
    assert exposure.target == "SYNTH"
    assert apf.is_apf_exposure({"DECKRNAM": "T"}, apf.APF_SHAPE)
    assert not apf.is_apf_exposure({"VERSION": "apf"}, (79, 4096))
    assert not apf.is_apf_exposure({}, apf.APF_SHAPE)
    other = tmp_path / "image.fits"
    fits.PrimaryHDU(np.zeros(apf.APF_SHAPE, dtype=np.float32)).writeto(other)
    with pytest.raises(ValueError, match="not a CPS-extracted APF exposure"):
        apf.read_apf_exposure(other)


# ---------------------------------------------------------------------------
# 2. extraction
# ---------------------------------------------------------------------------


def test_the_telluric_lines_give_the_drift_and_the_line_spread_width(drifted_spectrum):
    _, spectrum = drifted_spectrum
    assert spectrum.drift_kms == pytest.approx(1.7, abs=0.05)
    assert spectrum.zero_point is not None and spectrum.zero_point.n_regions >= 4
    assert spectrum.zero_point.err_kms < 0.1
    assert spectrum.lsf_fwhm_kms == pytest.approx(4.2, abs=0.15)
    assert "telluric O2 widths" in spectrum.lsf_source


def test_the_orders_are_merged_without_resampling(drifted_spectrum):
    _, spectrum = drifted_spectrum
    wave = spectrum.wave
    assert np.all(np.diff(wave) > 0)
    # every merged pixel is a pixel of the solution, moved by the measured drift
    pixels = (SOLUTION[3:61] * (1.0 + spectrum.drift_kms / C_KMS)).ravel()
    assert np.isin(wave, pixels).all()
    assert wave[0] == pytest.approx(SOLUTION[3, 0], rel=1e-4)
    good = ~spectrum.bad
    assert np.all(np.isfinite(spectrum.err[good]))
    clean = good & (wave > 5000.0) & (wave < 6200.0)
    assert np.median(spectrum.flux[clean]) == pytest.approx(1.0, abs=0.02)
    # night-sky [O I] 5577 is masked in the topocentric frame
    near_sky = np.abs(wave / 5577.34 - 1.0) * C_KMS < 10.0
    assert near_sky.any() and spectrum.bad[near_sky].all()


def test_the_zero_point_and_the_width_can_be_declared(files):
    exposure = apf.read_apf_exposure(files["drifted"])
    rows = range(49, 62)
    fixed = apf.extract(exposure, files["solution"], zero_point=0.5, lsf=5.0, rows=rows)
    assert fixed.drift_kms == 0.5 and fixed.zero_point is None
    assert fixed.lsf_fwhm_kms == 5.0 and fixed.lsf_source == "supplied"
    none = apf.extract(exposure, files["solution"], zero_point="none", lsf="decker", rows=rows)
    assert none.drift_kms == 0.0
    assert none.lsf_fwhm_kms == apf.DECKER_LSF_FWHM_KMS["N"]
    unmasked = apf.extract(exposure, files["solution"], lsf=5.0, rows=rows, mask_sky=False)
    assert not unmasked.bad[np.abs(unmasked.wave / 6300.30 - 1.0) * C_KMS < 5.0].any()
    with pytest.raises(ValueError, match="zero_point"):
        apf.extract(exposure, files["solution"], zero_point="tellurics")
    with pytest.raises(ValueError, match="lsf"):
        apf.extract(exposure, files["solution"], zero_point="none", lsf="header")


def test_the_rows_and_a_declared_width_are_checked(files):
    exposure = apf.read_apf_exposure(files["drifted"])
    declared = {"zero_point": "none", "lsf": 5.0}
    forward = apf.extract(exposure, SOLUTION, rows=(50, 51, 52), **declared)
    # Rows in echelle-order sequence, or repeated, are merged as rows 50-52. In an earlier
    # version (52, 51, 50) returned only part of the spectrum, and repeated rows gave a wrong
    # spectrum or raised LinAlgError.
    for rows in ((52, 51, 50), (50, 51, 51, 52, 50)):
        other = apf.extract(exposure, SOLUTION, rows=rows, **declared)
        np.testing.assert_array_equal(other.wave, forward.wave)
        np.testing.assert_array_equal(other.flux, forward.flux)
    for rows in ((), (-1,), (30, 79)):
        with pytest.raises(ValueError, match="rows"):
            apf.extract(exposure, SOLUTION, rows=rows, **declared)
    with pytest.raises(ValueError, match="rows"):
        apf.build_blaze([exposure], SOLUTION, rows=(78, 79))
    with pytest.raises(ValueError, match="rows"):
        apf.refine_wavelengths(
            SOLUTION,
            exposure,
            template_wave=np.linspace(5000.0, 6000.0, 1000),
            template_flux=np.ones(1000),
            rows=(),
        )
    for fwhm in (0.0, -2.0, float("nan")):
        with pytest.raises(ValueError, match="positive FWHM"):
            apf.extract(exposure, SOLUTION, zero_point="none", lsf=fwhm, rows=(50,))
    with pytest.raises(ValueError, match="finite shift"):
        apf.extract(exposure, SOLUTION, zero_point=float("nan"), lsf=5.0, rows=(50,))


def test_an_exposure_through_the_iodine_cell_is_refused_or_masked(files):
    exposure = apf.read_apf_exposure(files["iodine"])
    assert exposure.iodine_in
    with pytest.raises(ValueError, match="iodine"):
        apf.extract(exposure, files["solution"])
    with pytest.raises(ValueError, match="iodine must be"):
        apf.extract(exposure, files["solution"], iodine="drop")
    masked = apf.extract(exposure, files["solution"], iodine="mask", rows=range(30, 50))
    band = (masked.wave > 5000.0) & (masked.wave < 6200.0)
    assert band.any() and masked.bad[band].all()
    assert not masked.bad[masked.wave > 6250.0].all()


def test_an_exposure_without_telluric_lines_names_itself(tmp_path):
    path = write_exposure(tmp_path / "rsyn.501.fits", SOLUTION, telluric=False)
    exposure = apf.read_apf_exposure(path)
    with pytest.raises(ValueError, match=r"rsyn\.501\.fits: the telluric zero point"):
        apf.extract(exposure, SOLUTION, rows=range(49, 62))
    declared = apf.extract(exposure, SOLUTION, zero_point=0.0, lsf="decker", rows=range(49, 62))
    assert declared.drift_kms == 0.0


def test_time_and_barycentric_correction():
    bjd, v_bary, source = apf.apf_barycentric(HEADER_TIME)
    assert "THEMIDPT" in source
    assert bjd == pytest.approx(EXPECTED_BJD, abs=2e-6)
    assert v_bary == pytest.approx(EXPECTED_V_BARY, abs=0.002)
    fallback = {k: v for k, v in HEADER_TIME.items() if k != "THEMIDPT"}
    bjd_mid, v_mid, source_mid = apf.apf_barycentric(fallback)
    assert "DATE-OBS" in source_mid
    assert (bjd - bjd_mid) * 86400.0 == pytest.approx(30.0, abs=0.01)
    assert v_bary == pytest.approx(v_mid, abs=0.01)
    with pytest.raises(ValueError, match="RA/DEC"):
        apf.apf_barycentric({"THEMIDPT": HEADER_TIME["THEMIDPT"]})
    with pytest.raises(ValueError, match="THEMIDPT nor DATE-OBS"):
        apf.apf_barycentric({"RA": "1:00:00", "DEC": "10:00:00"})


# ---------------------------------------------------------------------------
# 3. the io route
# ---------------------------------------------------------------------------


def test_read_spectrum_reads_an_exposure_with_its_solution(files):
    with pytest.raises(ValueError, match="no wavelengths"):
        read_spectrum(str(files["drifted"]))
    raw = read_spectrum(str(files["drifted"]), apf={"wavelengths": str(files["solution"])})
    assert raw.instrument == "APF"
    assert raw.frame == "topocentric"
    assert raw.wave_medium == "air"
    assert not raw.continuum_normalized
    assert raw.calibration["reader"] == "albireo.apf"
    assert raw.calibration["drift_kms"] == pytest.approx(1.7, abs=0.05)
    fwhm = raw.calibration["lsf_fwhm_kms"]
    assert raw.resolving_power == pytest.approx(C_KMS / fwhm)
    assert raw.bjd == pytest.approx(apf.apf_barycentric(HEADER_TIME)[0])
    supplied = read_spectrum(
        str(files["drifted"]),
        apf={"wavelengths": str(files["solution"]), "rows": range(49, 62)},
        bjd=2460000.5,
        v_bary=1.25,
    )
    assert supplied.bjd == 2460000.5 and supplied.v_bary == 1.25
    assert supplied.time_source == "supplied by the caller"


def test_a_sky_subtracted_copy_is_not_read_as_a_second_epoch(files, tmp_path):
    directory = tmp_path / "night"
    directory.mkdir()
    for name in ("rsyn.201.fits", "rsyn.201_skysub.fits", "rsyn.202.fits"):
        shutil.copyfile(files["drifted"], directory / name)
    options = {"apf": {"wavelengths": str(files["solution"]), "rows": range(49, 62)}}
    with pytest.warns(RuntimeWarning, match="sky-subtracted copies"):
        raws = read_raw_spectra(str(directory), read_kwargs=options)
    assert [os.path.basename(r.path) for r in raws] == ["rsyn.201.fits", "rsyn.202.fits"]
    alone = read_raw_spectra([str(directory / "rsyn.201_skysub.fits")], read_kwargs=options)
    assert len(alone) == 1


# ---------------------------------------------------------------------------
# 4. refining a solution
# ---------------------------------------------------------------------------


def test_the_refinement_recovers_a_distorted_solution(tmp_path):
    star = stellar_lines(n=1500)
    true_wave = SOLUTION
    t = (np.arange(N_PIX, dtype=float) - (N_PIX - 1) / 2.0) / ((N_PIX - 1) / 2.0)
    distortion_kms = 1.2 + 1.5 * t - 1.0 * t**2
    base = apf.ApfWavelengths(true_wave / (1.0 + distortion_kms[None, :] / C_KMS))
    path = write_exposure(
        tmp_path / "rsyn.301.fits", true_wave, star=star, v_star_kms=-37.0, seed=9
    )
    exposure = apf.read_apf_exposure(path)
    result = apf.refine_wavelengths(
        base,
        exposure,
        template_wave=star[0],
        template_flux=1.0 + star[1],
        template_resolving_power=None,
        search_kms=100.0,
    )
    assert result.v_star_kms == pytest.approx(-37.0, abs=0.1)
    rows = list(result.rows)
    assert rows[0] <= 49 and rows[-1] >= 61
    error_kms = (result.wavelengths.wave[rows] / true_wave[rows] - 1.0) * C_KMS
    assert np.max(np.abs(error_kms[:, 100:-100])) < 0.3
    assert np.median(np.abs(error_kms)) < 0.1
    lo, hi = result.correction_range_kms
    assert lo == pytest.approx(distortion_kms.min(), abs=0.3)
    assert hi == pytest.approx(distortion_kms.max(), abs=0.3)
    assert "refined" in result.summary()
    # the exposure the solution was refined on has no telluric drift left
    zero = apf.extract(exposure, result.wavelengths, lsf=4.2, rows=range(49, 62))
    assert zero.drift_kms == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("n_lines", [500, 4000])
def test_the_refinement_survives_sparse_and_crowded_spectra(tmp_path, n_lines):
    """Two failure modes of an earlier version, kept as regressions.

    With 500 lines a few spurious chunk lags pulled the first fit in the red corner of the
    image, sigma clipping then rejected the good chunks there, and the solution was 5.6 km/s
    wrong at the end of the reddest order. With 4000 lines the stellar lines correlated with
    the telluric template in the telluric chunks, and the absolute scale was 3 km/s wrong.
    """
    star = stellar_lines(n=n_lines)
    t = (np.arange(N_PIX, dtype=float) - (N_PIX - 1) / 2.0) / ((N_PIX - 1) / 2.0)
    base = apf.ApfWavelengths(SOLUTION / (1.0 + (0.8 - 1.2 * t)[None, :] / C_KMS))
    path = write_exposure(tmp_path / "rsyn.401.fits", SOLUTION, star=star, v_star_kms=-37.0)
    result = apf.refine_wavelengths(
        base,
        apf.read_apf_exposure(path),
        template_wave=star[0],
        template_flux=1.0 + star[1],
        template_resolving_power=None,
        search_kms=100.0,
    )
    assert result.v_star_kms == pytest.approx(-37.0, abs=0.25)
    rows = list(result.rows)
    error_kms = (result.wavelengths.wave[rows] / SOLUTION[rows] - 1.0) * C_KMS
    assert np.max(np.abs(error_kms[:, 100:-100])) < 0.4


# ---------------------------------------------------------------------------
# 5. the configuration and the command
# ---------------------------------------------------------------------------


def test_a_configuration_resolves_the_solution_beside_it(tmp_path):
    from albireo.pipeline import load_config, write_config_template

    path = write_config_template(tmp_path / "albireo.toml")
    text = path.read_text(encoding="utf-8")
    extra = '\n[read.apf]\nwavelengths = "calib/apf_wave_2022.fits"\niodine = "mask"\n'
    path.write_text(text + extra, encoding="utf-8")
    options = load_config(path).read_kwargs["apf"]
    expected = tmp_path / "calib" / "apf_wave_2022.fits"
    assert os.path.normcase(options["wavelengths"]) == os.path.normcase(str(expected))
    assert options["iodine"] == "mask"


def test_the_command_reports_a_missing_exposure_and_keeps_an_existing_file(files, tmp_path, capsys):
    from albireo import cli

    out = tmp_path / "refined.fits"
    argv = ["apf-wavelengths", str(tmp_path / "missing.fits"), "--base", str(files["solution"])]
    assert cli.main([*argv, "--out", str(out)]) == 1
    assert "missing.fits" in capsys.readouterr().err
    out.write_bytes(b"")
    assert cli.main([*argv, "--out", str(out)]) == 1
    assert "exists" in capsys.readouterr().err
