"""Air vs vacuum wavelengths as a declared, validated property.

The whole point is that this is worth **83 km/s** — nearly constant across the optical,
and the same order as the orbital semi-amplitudes albireo exists to measure. It does not
average out, and it is not a rounding error. Before this field existed there was nowhere
to say which scale an epoch was on, so combining an ESPRESSO epoch (vacuum) with a FEROS
epoch (air) put the same physical line at two different model pixels and silently biased
every velocity that depended on the offending epochs.

`medium=None` stays legal because it is what every epoch built before the field existed
means. What is *not* legal is a mixture — including a mixture of declared and undeclared,
since "unknown" cannot be checked against "air".
"""

import numpy as np
import pytest

import albireo as ab
from albireo.data import Dataset, EpochData


def _edlen_reference(wave_vacuum):
    """The IAU-adopted Edlen (1966) / Birch & Downs (1994) refractivity, transcribed here.

    An independent plain-NumPy transcription straight from the published coefficients,

        (n - 1) x 1e8 = 8342.13 + 2406030/(130 - s^2) + 15997/(38.9 - s^2),  s = 1e4/lambda_vac

    used to check albireo's implementation. Anchoring on the formula rather than on
    remembered air/vacuum line pairs is deliberate: published line values come from
    different sources with different conventions and are not reliably self-consistent to
    the milli-Angstrom, whereas the refractivity is exactly what the standard defines.
    What this cross-check catches is the wiring — the direction of the division, the
    wavenumber convention (vacuum, not air), and the micron/Angstrom unit factor.
    """
    wave = np.asarray(wave_vacuum, dtype=float)
    sigma2 = (1e4 / wave) ** 2
    return 1e-8 * (8342.13 + 2406030.0 / (130.0 - sigma2) + 15997.0 / (38.9 - sigma2))


def _epoch(**kw):
    return EpochData(
        wave=np.linspace(4000.0, 4010.0, 20),
        flux=np.ones(20),
        ivar=np.full(20, 100.0),
        bjd=2459000.5,
        **kw,
    )


# -- the conversion ----------------------------------------------------------


def test_conversion_applies_the_edlen_refractivity():
    """The ratio the conversion applies must be the IAU refractivity itself."""
    wave = np.geomspace(3000.0, 10000.0, 501)
    air = np.asarray(ab.vacuum_to_air(wave))
    assert np.allclose(wave / air - 1.0, _edlen_reference(wave), rtol=1e-12)
    # And the refractivity is in the right regime: air's index of refraction is a few
    # parts in 1e4, falling with wavelength.
    n_minus_one = _edlen_reference(wave)
    assert 2.7e-4 < n_minus_one.min() < n_minus_one.max() < 3.0e-4
    assert np.all(np.diff(n_minus_one) < 0)


def test_halpha_lands_where_optical_line_lists_put_it():
    """One end-to-end sanity anchor on a line everybody knows.

    H-alpha is 6564.61 A in vacuum and 6562.80 A in air. Loose tolerance on purpose: the
    exact tabulated value differs by a few mA between sources (and the line is a blended
    multiplet), so this checks the conversion is right to well under a pixel, not that it
    reproduces one particular table.
    """
    assert float(np.asarray(ab.vacuum_to_air(6564.614))) == pytest.approx(6562.80, abs=0.01)
    assert float(np.asarray(ab.air_to_vacuum(6562.80))) == pytest.approx(6564.61, abs=0.01)


def test_air_to_vacuum_is_the_exact_inverse():
    """Two fixed-point iterations, and the round trip has to close to float64."""
    wave = np.geomspace(3000.0, 10000.0, 2001)
    back = np.asarray(ab.air_to_vacuum(ab.vacuum_to_air(wave)))
    assert np.max(np.abs(back - wave)) < 1e-10
    forth = np.asarray(ab.vacuum_to_air(ab.air_to_vacuum(wave)))
    assert np.max(np.abs(forth - wave)) < 1e-10


def test_the_offset_is_the_velocity_that_makes_it_matter():
    """~83 km/s across the optical — the claim the docstrings and the guard rest on."""
    wave = np.array([3000.0, 5000.0, 6562.8, 10000.0])
    air = np.asarray(ab.vacuum_to_air(wave))
    v = ab.C_KMS * (wave - air) / wave
    assert np.all(v > 82.0) and np.all(v < 88.0), v
    # And in Angstrom it grows with wavelength, which is why a constant shift is wrong.
    offsets = wave - air
    assert np.all(np.diff(offsets) > 0)
    assert offsets[0] == pytest.approx(0.874, abs=0.01)
    assert offsets[-1] == pytest.approx(2.741, abs=0.01)


def test_conversion_preserves_shape_and_monotonicity():
    wave = np.linspace(4000.0, 5000.0, 101)
    air = np.asarray(ab.vacuum_to_air(wave))
    assert air.shape == wave.shape
    assert np.all(np.diff(air) > 0)
    assert np.all(air < wave)  # air wavelengths are always the shorter ones


# -- the declaration ---------------------------------------------------------


def test_medium_defaults_to_undeclared_and_accepts_the_two_values():
    assert _epoch().medium is None
    assert _epoch(medium="air").medium == "air"
    assert _epoch(medium="vacuum").medium == "vacuum"


def test_medium_rejects_anything_else():
    for bad in ("Air", "AIR", "vac", "topocentric", ""):
        with pytest.raises(ValueError, match="medium must be one of"):
            _epoch(medium=bad)


def test_a_dataset_of_one_medium_is_fine():
    for medium in (None, "air", "vacuum"):
        ds = Dataset([_epoch(medium=medium), _epoch(medium=medium)])
        assert {e.medium for e in ds.epochs} == {medium}


def test_mixing_air_and_vacuum_raises_rather_than_picking_one():
    """The exception this field exists to produce."""
    with pytest.raises(ValueError, match="disagree about their wavelength scale") as exc:
        Dataset([_epoch(medium="air"), _epoch(medium="vacuum")])
    message = str(exc.value)
    assert "83 km/s" in message
    assert "air_to_vacuum" in message
    assert "air: epochs [0]" in message
    assert "vacuum: epochs [1]" in message


def test_mixing_declared_with_undeclared_also_raises():
    """'Unknown' is not a value that can be checked against 'air'."""
    with pytest.raises(ValueError, match="disagree about their wavelength scale") as exc:
        Dataset([_epoch(medium="air"), _epoch()])
    assert "undeclared" in str(exc.value)


def test_the_error_names_the_offending_epochs_and_truncates_long_lists():
    with pytest.raises(ValueError) as exc:
        Dataset([*[_epoch(medium="air") for _ in range(6)], _epoch(medium="vacuum")])
    message = str(exc.value)
    assert "'...'" in message, "a long epoch list should be truncated, not dumped"
    assert "vacuum: epochs [6]" in message


def test_converting_makes_the_mixture_legal():
    """The documented fix has to actually work end to end."""
    wave = np.linspace(4000.0, 4010.0, 20)
    air_epoch = EpochData(
        wave=wave, flux=np.ones(20), ivar=np.full(20, 100.0), bjd=1.0, medium="air"
    )
    vacuum_epoch = EpochData(
        wave=np.asarray(ab.air_to_vacuum(wave)),
        flux=np.ones(20),
        ivar=np.full(20, 100.0),
        bjd=2.0,
        medium="vacuum",
    )
    with pytest.raises(ValueError, match="disagree"):
        Dataset([air_epoch, vacuum_epoch])

    harmonized = EpochData(
        wave=np.asarray(ab.vacuum_to_air(vacuum_epoch.wave)),
        flux=vacuum_epoch.flux,
        ivar=vacuum_epoch.ivar,
        bjd=vacuum_epoch.bjd,
        medium="air",
    )
    ds = Dataset([air_epoch, harmonized])
    assert len(ds) == 2
    # Converting there and back must land on the original grid.
    assert np.max(np.abs(harmonized.wave - air_epoch.wave)) < 1e-10


def test_medium_survives_the_existing_epoch_machinery():
    """The field is carried, not dropped, by whatever already copies epochs."""
    ds = Dataset([_epoch(medium="vacuum"), _epoch(medium="vacuum")], frame="barycentric")
    assert ds[0].medium == "vacuum"
    assert all(e.medium == "vacuum" for e in ds)
