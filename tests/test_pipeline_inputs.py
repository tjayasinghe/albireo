"""What the pipeline reads and records before the disentangling.

The noise model of a declaration is recorded as declared: a ``Fixed`` correlation is held,
not fitted. A BLOeM star is resolved without the multiplicity table, which the pipeline
does not read. With a region, each spectrum is cut to it on reading, and the epochs are
the same bit for bit.
"""

from __future__ import annotations

import dataclasses
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from test_io import write_eso_phase3

import albireo as ab
from albireo import pipeline

# One of the files has no usable error column, on purpose: its weights are estimated from
# the scatter, which the cut must leave unchanged as well.
pytestmark = pytest.mark.filterwarnings("ignore:.*holds no finite positive value:RuntimeWarning")


@pytest.fixture(scope="module")
def dataset():
    return ab.load_example("sb2_sim")


def _dis(dataset, noise_correlation):
    return ab.Disentangler(
        dataset,
        components=[ab.Star("primary", light=0.62), ab.Star("secondary", light=0.38)],
        orbit=ab.Orbit(period=ab.Between(5.5, 6.5), k=ab.Between([10.0, 10.0], [90.0, 90.0])),
        lsf={"DEMO": 6.5},
        noise_correlation=noise_correlation,
    )


@pytest.mark.parametrize("declared", [0.3, {"DEMO": 0.3}, ab.Fixed(0.3)], ids=str)
def test_a_held_noise_correlation_is_recorded_by_its_value(dataset, declared):
    dis = _dis(dataset, declared)
    assert pipeline._describe_noise(dis) == {"kind": "ar1", "correlation": {"DEMO": 0.3}}
    assert pipeline._noise_words(dis) == "AR(1), lag-one DEMO 0.30"


def test_a_fitted_noise_correlation_is_recorded_with_its_prior(dataset):
    dis = _dis(dataset, ab.Between(-0.9, 0.9, start_at=0.0))
    described = pipeline._describe_noise(dis)
    assert described == {"kind": "ar1", "correlation": "fitted", "prior": {"lo": -0.9, "hi": 0.9}}
    assert pipeline._noise_words(dis) == "AR(1), correlation fitted"
    assert pipeline._describe_noise(_dis(dataset, None)) == {"kind": "diagonal"}


def test_a_bloem_star_is_resolved_without_the_multiplicity_table(tmp_path, monkeypatch):
    """The class table is not read, so an empty or renamed one must not stop the star."""
    from albireo import archive

    calls = []
    target = archive.BloemTarget(
        bloem_id="1-037",
        gaia_dr3="4685929099908473856",
        ra_deg=12.0,
        dec_deg=-73.0,
        spectral_type="B0 V",
        binary_class=None,
    )

    def resolve(name, *, with_classification=True, timeout=180.0):
        calls.append(with_classification)
        if with_classification:
            raise RuntimeError("VizieR returned no rows for the classification table")
        return target

    monkeypatch.setattr(archive, "resolve_bloem", resolve)
    monkeypatch.setattr(archive, "bloem_spectra", lambda star, **kw: ["r1", "r2"])
    monkeypatch.setattr(archive, "download", lambda records, outdir, **kw: ["OK", "OK"])
    logs: list[str] = []
    ctx = SimpleNamespace(
        star=SimpleNamespace(bloem="1-037"), directory=tmp_path, log=logs.append, flag=logs.append
    )
    pattern = pipeline._fetch_bloem(ctx)
    assert calls == [False]
    assert pattern.endswith("*.fits") and "Gaia DR3 4685929099908473856: 2 epochs" in logs[-1]


# -- the cut on reading ------------------------------------------------------------------


def _files(directory):
    """Five full-band files, one of which does not reach the region."""
    for i, (lo, err) in enumerate(
        [(4350.0, "nan"), (4352.5, 3.0), (4340.0, None), (4420.0, 2.0), (4600.0, 2.0)]
    ):
        write_eso_phase3(directory / f"s{i}.fits", lo=lo, err=err, mjd_obs=53243.0 + i)
    return str(directory / "*.fits")


def _context(spectra, region, smooth_angstrom, logs):
    return SimpleNamespace(
        star=SimpleNamespace(
            spectra=spectra, instrument=None, bloem=None, dataset=None, medium=None
        ),
        settings=SimpleNamespace(
            region=region, smooth_angstrom=smooth_angstrom, mask=((4410.0, 4412.0),)
        ),
        log=logs.append,
        flag=logs.append,
        config=SimpleNamespace(read_kwargs=None),
    )


@pytest.mark.parametrize("smooth_angstrom", [None, 2.0])
def test_each_file_is_cut_to_the_region_on_reading_and_the_epochs_are_unchanged(
    tmp_path, monkeypatch, smooth_angstrom
):
    from albireo import io

    spectra = _files(tmp_path)
    region = (4380.0, 4480.0)
    seen = []
    read = io.read_raw_spectra

    def recording(*args, **kwargs):
        seen.append(kwargs.get("epoch_kwargs"))
        return read(*args, **kwargs)

    monkeypatch.setattr(io, "read_raw_spectra", recording)
    logs: list[str] = []
    dataset, _ = pipeline._load_dataset(_context(spectra, region, smooth_angstrom, logs))
    expected = {"region": region}
    if smooth_angstrom is not None:
        expected["smooth_angstrom"] = smooth_angstrom
    assert seen == [expected]
    (line,) = [entry for entry in logs if entry.startswith("read ")]
    assert "cut on reading to the region" in line, line
    assert any("left out 1 of 5 files" in entry for entry in logs), logs

    # The read without the cut, as before: the same epochs, bit for bit.
    options = {"region": region, "mask": [(4410.0, 4412.0)], **expected}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*cover less than .* of the region")
        whole = io.dataset_from_raw(read(spectra), min_coverage=pipeline._MIN_COVERAGE, **options)
    whole = pipeline._try_share_grid(whole, lambda *a: None)
    assert dataset.n_epochs == whole.n_epochs == 4
    for cut, full in zip(dataset, whole, strict=True):
        for f in dataclasses.fields(cut):
            a, b = getattr(cut, f.name), getattr(full, f.name)
            if isinstance(a, np.ndarray):
                assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True), f.name
            else:
                assert a == b or (a != a and b != b), f.name


def test_without_a_region_the_files_are_read_whole(tmp_path, monkeypatch):
    from albireo import io

    spectra = _files(tmp_path)
    seen = []
    read = io.read_raw_spectra
    monkeypatch.setattr(
        io,
        "read_raw_spectra",
        lambda *a, **kw: seen.append(kw.get("epoch_kwargs")) or read(*a, **kw),
    )
    logs: list[str] = []
    pipeline._load_dataset(_context(spectra, None, 2.0, logs))
    assert seen == [None]
    (line,) = [entry for entry in logs if entry.startswith("read ")]
    assert "cut on reading" not in line and "6000 px" in line, line
