"""Tests for the ``albireo`` command: the novice's route through the pipeline.

The file path is what a first-time user actually takes -- FITS files on disk, a TOML
they edited, one command -- so the ``run`` test writes a simulated SB2 as IRAF-style
FITS images with the resolving power in the header, and checks that the command reads
them, takes the LSF from the header, and writes the products. ``fetch`` is tested against
a stubbed archive, since the real one is the network.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import albireo as ab
from albireo import cli
from albireo.pipeline import _with_medium, load_config
from albireo.simulate import InstrumentSpec, OrbitParams, simulate_dataset


def test_init_writes_a_template_the_loader_accepts(tmp_path, capsys):
    path = tmp_path / "albireo.toml"
    assert cli.main(["init", str(path)]) == 0
    assert "wrote" in capsys.readouterr().out
    config = load_config(path)
    assert config.stars[0].name == "AI Phe"
    assert cli.main(["init", str(path)]) == 1, "an existing file is not overwritten silently"
    assert cli.main(["init", str(path), "--force"]) == 0


def test_run_reports_a_missing_or_broken_configuration(tmp_path, capsys):
    assert cli.main(["run", str(tmp_path / "missing.toml")]) == 1
    assert "no such configuration" in capsys.readouterr().err
    broken = tmp_path / "broken.toml"
    broken.write_text('[[stars]]\nname = "x"\nspectra = "a/*.fits"\nperiod = [1, 2]\n', "utf-8")
    assert cli.main(["run", str(broken)]) == 1
    assert "at least one component" in capsys.readouterr().err


def _write_fits_epochs(dataset, directory):
    """Each epoch as an IRAF-style image with the facts the reader needs in its header."""
    fits = pytest.importorskip("astropy.io.fits")
    directory.mkdir(parents=True, exist_ok=True)
    for j, epoch in enumerate(dataset):
        step = float(np.median(np.diff(epoch.wave)))
        hdu = fits.PrimaryHDU(data=np.asarray(epoch.flux, dtype=np.float32))
        hdr = hdu.header
        hdr["CRVAL1"] = float(epoch.wave[0])
        hdr["CDELT1"] = step
        hdr["CRPIX1"] = 1.0
        hdr["CTYPE1"] = "AWAV"
        hdr["CUNIT1"] = "Angstrom"
        hdr["INSTRUME"] = "SIM"
        hdr["SPEC_RES"] = 23_150.0  # FWHM 12.95 km/s -> a Gaussian sigma of 5.5 km/s
        hdr["CONTNORM"] = True
        hdr["SPECSYS"] = "BARYCENT"
        hdr["BARYCORR"] = float(epoch.v_bary)
        # No coordinates: the reader warns and takes the time as it is, which is what a
        # simulated BJD needs. MJD-OBS + EXPTIME/2 lands on the epoch's own bjd.
        hdr["MJD-OBS"] = float(epoch.bjd) - 2400000.5 - 0.5 * 60.0 / 86400.0
        hdr["EXPTIME"] = 60.0
        hdu.writeto(directory / f"epoch_{j:02d}.fits", overwrite=True)
    return directory


@pytest.fixture(scope="module")
def fits_star(tmp_path_factory):
    """A simulated SB2 on disk: 8 barycentric-frame epochs, continuum-normalized."""
    grid = ab.LogGrid.from_wavelength_range(5000.0, 5040.0, dv_kms=2.0)
    components = [
        ab.synthetic_deviation_spectrum(grid, n_lines=14, seed=s, margin=0.1) for s in (1, 2)
    ]
    bjd = 2459000.0 + np.sort(np.random.default_rng(5).uniform(0.0, 21.0, size=8))
    dataset, truth = simulate_dataset(
        grid,
        components,
        bjd=bjd,
        instruments={
            "SIM": InstrumentSpec(wave=np.arange(5005.0, 5035.0, 0.05), sigma_v_lsf=5.5, snr=150.0)
        },
        light_fractions=(0.6, 0.4),
        orbit=OrbitParams(period=6.31, t_peri=2.0, ecc=0.1, omega=0.7, k=(30.0, 50.0)),
        frame="barycentric",
        seed=11,
    )
    directory = _write_fits_epochs(_with_medium(dataset, "air"), tmp_path_factory.mktemp("fits"))
    return directory, dataset, truth


def test_run_takes_fits_files_from_a_toml_to_the_products(fits_star, tmp_path, capsys):
    directory, dataset, _truth = fits_star
    config = tmp_path / "albireo.toml"
    config.write_text(
        "\n".join(
            [
                "[output]",
                'directory = "out"',
                "plots = false",
                "",
                "[analysis]",
                "max_steps = 3",
                "k_max = 90.0",
                "",
                "[[stars]]",
                'name = "SIM binary"',
                f'spectra = "{(directory / "*.fits").as_posix()}"',
                "period = [6.0, 6.6]",
                'medium = "air"',
                "",
                "[[stars.components]]",
                'name = "primary"',
                "light = 0.6",
                "",
                "[[stars.components]]",
                'name = "secondary"',
                "light = 0.4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    status = cli.main(["run", str(config), "--fast", "--quiet"])
    out = capsys.readouterr().out
    assert status == 0, out
    assert "1 of 1 star(s) completed" in out
    report = json.loads((tmp_path / "out" / "SIM_binary" / "result.json").read_text("utf-8"))
    assert report["dataset"]["n_epochs"] == 8
    assert report["dataset"]["frame"] == "barycentric"
    assert report["dataset"]["medium"] == "air"
    assert report["dataset"]["lsf_sigma_kms"]["SIM"] == pytest.approx(5.5, rel=0.02), (
        "the LSF came from the files' own SPEC_RES header"
    )
    np.testing.assert_allclose(report["dataset"]["bjd"], np.asarray(dataset.bjd), atol=1e-4)
    assert (tmp_path / "out" / "results.csv").is_file()
    assert (tmp_path / "out" / "SIM_binary" / "velocities.csv").is_file()


def test_run_without_an_lsf_anywhere_names_the_fix(fits_star, tmp_path, capsys):
    directory, _, _ = fits_star
    config = tmp_path / "albireo.toml"
    config.write_text(
        "\n".join(
            [
                "[output]",
                'directory = "out"',
                "[read]",
                "resolving_power = 0.0",  # the header's value, overridden to nothing
                "[[stars]]",
                'name = "x"',
                f'spectra = "{(directory / "*.fits").as_posix()}"',
                "period = [6.0, 6.6]",
                "[[stars.components]]",
                'name = "a"',
                "light = 0.5",
                "[[stars.components]]",
                'name = "b"',
                "light = 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    status = cli.main(["run", str(config), "--fast", "--quiet", "--no-plots"])
    assert status == 2, "a failed star is a non-zero exit"
    text = (tmp_path / "out" / "failures.txt").read_text(encoding="utf-8")
    assert "no line-spread function" in text and "[instrument.SIM]" in text


def test_fetch_prints_a_stars_entry_from_a_stubbed_archive(tmp_path, monkeypatch, capsys):
    from albireo import archive

    target = archive.BloemTarget(
        bloem_id="1-037",
        gaia_dr3="4685929099908473856",
        ra_deg=12.0,
        dec_deg=-73.0,
        spectral_type="B0 V",
        binary_class="SB2",
    )
    monkeypatch.setattr(archive, "resolve_bloem", lambda name, **kw: target)
    monkeypatch.setattr(archive, "bloem_spectra", lambda star, **kw: ["r1", "r2"])
    monkeypatch.setattr(archive, "download", lambda records, outdir, **kw: ["OK", "OK"])
    assert cli.main(["fetch", "1-037", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "2 of 2 epochs" in out
    assert "[[stars]]" in out and 'name = "BLOeM 1-037"' in out and 'period = "search"' in out


def test_jobs_must_be_an_integer_or_auto(tmp_path):
    config = tmp_path / "albireo.toml"
    config.write_text(
        '[[stars]]\nname = "x"\nspectra = "a/*.fits"\nperiod = [1, 2]\n'
        '[[stars.components]]\nname = "a"\nlight = 1.0\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="--jobs"):
        cli.main(["run", str(config), "--jobs", "many"])
