"""Tests for :mod:`albireo.pepsi`, the reader of PEPSI products of the SDS pipeline.

The products are written here as the pipeline writes them: an empty primary HDU whose header
carries ``INSTRUME = 'PEPSI'``, ``JD-TDB``, ``SSBVEL`` (m/s), ``FIBER``, ``ARM`` and
``CROSDIS``, followed by a binary table with the double-precision columns ``Arg``, ``Fun``
and ``Var`` and the logical column ``Mask``. The file names end in the product kind
(``pepsib.<date>.<number>.dxt.bwl``) rather than ``.fits``. A directory holds the blue- and
the red-arm files of each exposure, which cover different wavelengths.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits

from albireo import pepsi
from albireo.io import read_dataset, read_raw_spectra, read_spectrum, region_coverage

C_KMS = 299_792.458
BLUE = (4800.0, 5441.0, "Blue", "3: 4800 - 5441")
RED = (6278.0, 7419.0, "Red", "5: 6278 - 7419")


def write_product(
    path,
    *,
    arm=BLUE,
    jd_tdb=2460500.75,
    ssbvel=12_345.6,
    fiber="300",
    n=6000,
    mask=None,
    var=None,
    descending=False,
    instrume="PEPSI",
    seed=0,
):
    """Write a PEPSI product with a few absorption lines on a normalised continuum."""
    lo, hi, arm_name, crosdis = arm
    rng = np.random.default_rng(seed)
    wave = np.exp(np.linspace(math.log(lo), math.log(hi), n))
    flux = np.ones(n)
    for centre in np.linspace(lo + 20.0, hi - 20.0, 12):
        flux -= 0.4 * np.exp(-0.5 * ((wave - centre) / 0.08) ** 2)
    flux += 0.005 * rng.standard_normal(n)
    var = np.full(n, 0.005**2) if var is None else var
    mask = np.zeros(n, dtype=bool) if mask is None else mask
    if descending:
        wave, flux, var, mask = wave[::-1], flux[::-1], var[::-1], mask[::-1]
    primary = fits.PrimaryHDU()
    header = primary.header
    header["INSTRUME"] = instrume
    header["TELESCOP"] = "LBT"
    header["OBJECT"] = "SYNTH"
    header["ARM"] = arm_name
    header["CROSDIS"] = crosdis
    header["FIBER"] = fiber
    if jd_tdb is not None:
        header["JD-OBS"] = jd_tdb - 0.003
        header["JD-TDB"] = jd_tdb
    if ssbvel is not None:
        header["SSBVEL"] = ssbvel
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="Arg", format="1D", array=wave),
            fits.Column(name="Fun", format="1D", array=flux),
            fits.Column(name="Var", format="1D", array=var),
            fits.Column(name="Mask", format="1L", array=mask),
        ]
    )
    fits.HDUList([primary, table]).writeto(path)
    return path


def test_a_product_is_read_with_its_time_velocity_and_resolution(tmp_path):
    path = write_product(tmp_path / "pepsib.20240101.001.dxt.bwl")
    product = pepsi.read_pepsi(path)
    assert np.all(np.diff(product.wave) > 0)
    assert product.wave[0] == pytest.approx(4800.0)
    np.testing.assert_allclose(product.err, 0.005)
    assert product.bjd == 2460500.75
    assert product.v_bary == pytest.approx(-12.3456), "SSBVEL in m/s, the sign reversed"
    assert product.arm == "blue" and product.cross_disperser == 3
    assert product.fiber == "300" and product.resolving_power == 44_300.0
    assert product.lsf_sigma_kms == pytest.approx(C_KMS / 44_300.0 / 2.3548, rel=1e-4)
    assert product.kind == "bwl" and product.target == "SYNTH"
    assert not product.bad.any()


def test_read_spectrum_recognises_a_product(tmp_path):
    path = str(write_product(tmp_path / "pepsir.20240101.001.dxt.bwl", arm=RED))
    raw = read_spectrum(path)
    assert raw.instrument == "PEPSI"
    assert raw.frame == "barycentric"
    assert raw.wave_medium == "air"
    assert raw.continuum_normalized
    assert raw.resolving_power == 44_300.0
    assert raw.bjd == 2460500.75 and raw.v_bary == pytest.approx(-12.3456)
    assert raw.calibration["arm"] == "red" and raw.calibration["cross_disperser"] == 5
    assert raw.columns["wave"] == "Arg"
    declared = read_spectrum(path, instrument="PEPSI-300", bjd=2460000.5, v_bary=3.0)
    assert declared.instrument == "PEPSI-300"
    assert declared.bjd == 2460000.5 and declared.v_bary == 3.0
    assert declared.time_source == "supplied by the caller"


def test_flagged_and_invalid_pixels_are_bad_and_the_order_is_restored(tmp_path):
    n = 6000
    mask = np.zeros(n, dtype=bool)
    mask[100:110] = True
    var = np.full(n, 0.005**2)
    var[200] = 0.0
    var[201] = np.nan
    path = write_product(tmp_path / "pepsib.1.dxt.bwl", mask=mask, var=var, descending=True)
    product = pepsi.read_pepsi(path)
    assert np.all(np.diff(product.wave) > 0), "sorted by wavelength"
    assert product.bad.sum() == 12
    assert np.all(np.isnan(product.flux[product.bad]))
    assert np.all(np.isinf(product.err[product.bad]))
    assert not product.bad[:100].any() and product.bad[100:110].all()


def test_a_mask_that_flags_most_pixels_is_not_applied(tmp_path):
    mask = np.ones(6000, dtype=bool)
    mask[:10] = False
    path = write_product(tmp_path / "pepsib.2.dxt.bwl", mask=mask)
    with pytest.warns(RuntimeWarning, match="Mask column flags"):
        product = pepsi.read_pepsi(path)
    assert not product.bad.any()


def test_an_unknown_fibre_needs_a_declared_resolution(tmp_path):
    path = write_product(tmp_path / "pepsib.3.dxt.bwl", fiber="200")
    with pytest.warns(RuntimeWarning, match="no default resolving power"):
        product = pepsi.read_pepsi(path)
    assert product.resolving_power is None and product.lsf_sigma_kms is None
    declared = pepsi.read_pepsi(path, resolving_power=60_000.0)
    assert declared.resolving_power == 60_000.0


def test_other_files_and_incomplete_headers_are_refused(tmp_path):
    other = write_product(tmp_path / "other.fits", instrume="HARPS")
    with pytest.raises(ValueError, match="not a PEPSI product"):
        pepsi.read_pepsi(other)
    no_time = write_product(tmp_path / "pepsib.4.dxt.bwl", jd_tdb=None)
    with pytest.raises(ValueError, match="JD-TDB"):
        pepsi.read_pepsi(no_time)
    no_velocity = write_product(tmp_path / "pepsib.5.dxt.bwl", ssbvel=None)
    with pytest.raises(ValueError, match="SSBVEL"):
        pepsi.read_pepsi(no_velocity)
    empty = tmp_path / "pepsib.6.dxt.bwl"
    header = fits.Header()
    header["INSTRUME"] = "PEPSI"
    fits.PrimaryHDU(header=header).writeto(empty)
    with pytest.raises(ValueError, match="Arg, Fun and Var"):
        pepsi.read_pepsi(empty)


def test_a_product_covering_both_arms_is_labelled_so(tmp_path):
    both = (4800.0, 7419.0, "Blue", "3: 4800 - 5441")
    path = write_product(tmp_path / "pepsib.20240101.001.dxt.all", arm=both, n=20_000)
    assert pepsi.read_pepsi(path).arm == "both"


def test_a_pixel_without_a_wavelength_is_bad_and_does_not_change_the_arm(tmp_path):
    """A non-finite wavelength sorts to the end; the arm follows the finite ones."""
    both = (4800.0, 7419.0, "Blue", "3: 4800 - 5441")
    path = tmp_path / "pepsib.20240101.002.dxt.all"
    write_product(path, arm=both, n=20_000)
    with fits.open(path, mode="update") as hdulist:
        hdulist[1].data["Arg"][-1] = np.nan
    product = pepsi.read_pepsi(path)
    assert product.arm == "both"
    assert np.isnan(product.wave[-1]) and product.bad[-1] and not product.bad[:-1].any()
    assert np.all(np.diff(product.wave[:-1]) > 0)


def test_a_repeated_wavelength_is_refused(tmp_path):
    """Overlapping samples would reach the epoch with two values at one wavelength."""
    path = tmp_path / "pepsib.20240101.003.dxt.bwl"
    write_product(path)
    with fits.open(path, mode="update") as hdulist:
        wave = hdulist[1].data["Arg"]
        wave[3001] = wave[3000]
    with pytest.raises(ValueError, match="not strictly increasing"):
        pepsi.read_pepsi(path)


@pytest.fixture
def two_arms(tmp_path):
    """Three exposures, each as a blue-arm and a red-arm file, in one directory."""
    directory = tmp_path / "star"
    directory.mkdir()
    for k in range(3):
        jd = 2460500.75 + 3.1 * k
        write_product(directory / f"pepsib.2024010{k + 1}.00{k + 1}.dxt.bwl", jd_tdb=jd, seed=k)
        write_product(
            directory / f"pepsir.2024010{k + 1}.00{k + 1}.dxt.bwl", arm=RED, jd_tdb=jd, seed=k
        )
    (directory / "notes.txt").write_text("not a spectrum", encoding="utf-8")
    return directory


def test_a_directory_of_both_arms_is_read_whole(two_arms):
    raws = read_raw_spectra(str(two_arms))
    assert len(raws) == 6
    assert sorted(r.calibration["arm"] for r in raws) == ["blue"] * 3 + ["red"] * 3
    blue = next(r for r in raws if r.calibration["arm"] == "blue")
    assert region_coverage(blue, (5000.0, 5300.0)) == pytest.approx(1.0, abs=1e-3)
    assert region_coverage(blue, (5300.0, 5600.0)) == pytest.approx(141.0 / 300.0, abs=1e-3)
    assert region_coverage(blue, (6500.0, 6600.0)) == 0.0


def test_the_arm_that_misses_the_region_is_left_out(two_arms):
    with pytest.raises(ValueError, match="region"):
        read_dataset(str(two_arms), region=(5000.0, 5300.0))
    with pytest.warns(RuntimeWarning, match="3 of 6 spectra cover less than"):
        dataset = read_dataset(str(two_arms), region=(5000.0, 5300.0), min_coverage=0.5)
    assert len(dataset) == 3
    assert all(5000.0 <= e.wave[0] and e.wave[-1] <= 5300.0 for e in dataset)
    with pytest.warns(RuntimeWarning, match="pepsib"):
        red = read_dataset(str(two_arms), region=(6500.0, 6600.0), min_coverage=0.5)
    assert len(red) == 3
    with warnings.catch_warnings(), pytest.raises(ValueError, match="no spectrum covers"):
        warnings.simplefilter("ignore", RuntimeWarning)
        read_dataset(str(two_arms), region=(5600.0, 5700.0), min_coverage=0.5)
