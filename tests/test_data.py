"""Tests for the EpochData/Dataset containers, their coercion and their validation."""

import numpy as np
import pytest

from albireo.data import Dataset, EpochData


def make_epoch(n: int = 8, **kwargs) -> EpochData:
    """A valid epoch of ``n`` pixels; keyword arguments override the defaults."""
    defaults = {
        "wave": 4000.0 + 0.1 * np.arange(n),
        "flux": np.ones(n),
        "ivar": np.full(n, 100.0),
        "bjd": 2459000.5,
    }
    return EpochData(**{**defaults, **kwargs})


# --------------------------------------------------------------------------- construction


def test_construction_coerces_to_float64_arrays():
    epoch = EpochData(
        wave=[4000.0, 4000.1, 4000.2],
        flux=[1, 0, 1],  # ints on purpose
        ivar=[100, 100, 100],
        bjd=2459000,
        v_bary=-13,
        instrument="HERMES",
    )
    for name in ("wave", "flux", "ivar"):
        arr = getattr(epoch, name)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.shape == (3,)
    assert isinstance(epoch.bjd, float)
    assert isinstance(epoch.v_bary, float)
    assert epoch.bjd == 2459000.0
    assert epoch.v_bary == -13.0
    assert epoch.instrument == "HERMES"
    assert epoch.mask is None
    assert epoch.n_pixels == 3


def test_construction_defaults():
    epoch = make_epoch()
    assert epoch.v_bary == 0.0
    assert epoch.instrument == "default"
    assert epoch.mask is None


def test_frozen_dataclass_rejects_assignment():
    epoch = make_epoch()
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raises FrozenInstanceError
        epoch.bjd = 1.0


# ----------------------------------------------------------------------------- validation


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        EpochData(wave=[1.0, 2.0, 3.0], flux=[1.0, 1.0], ivar=[1.0, 1.0, 1.0], bjd=0.0)


def test_too_few_pixels_raises():
    with pytest.raises(ValueError, match="at least 2 pixels"):
        EpochData(wave=[4000.0], flux=[1.0], ivar=[1.0], bjd=0.0)


def test_non_1d_raises():
    with pytest.raises(ValueError, match="1-D"):
        EpochData(
            wave=np.ones((2, 3)),
            flux=np.ones((2, 3)),
            ivar=np.ones((2, 3)),
            bjd=0.0,
        )


@pytest.mark.parametrize(
    "wave",
    [
        [4000.0, 4000.2, 4000.1],  # decreasing step
        [4000.0, 4000.1, 4000.1],  # duplicate wavelength
    ],
)
def test_non_increasing_wave_raises(wave):
    with pytest.raises(ValueError, match="strictly increasing"):
        make_epoch(3, wave=wave)


def test_nonpositive_or_nonfinite_wave_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        make_epoch(3, wave=[-1.0, 0.5, 4000.0])
    with pytest.raises(ValueError, match="wave must be finite"):
        make_epoch(3, wave=[4000.0, np.nan, 4000.2])


def test_negative_ivar_raises():
    with pytest.raises(ValueError, match="non-negative"):
        make_epoch(3, ivar=[100.0, -1.0, 100.0])


def test_non_finite_ivar_raises():
    with pytest.raises(ValueError, match="ivar must be finite"):
        make_epoch(3, ivar=[100.0, np.inf, 100.0])


def test_non_finite_flux_at_weighted_pixel_raises():
    with pytest.raises(ValueError, match="finite wherever ivar > 0"):
        make_epoch(3, flux=[1.0, np.nan, 1.0], ivar=[100.0, 100.0, 100.0])
    with pytest.raises(ValueError, match="finite wherever ivar > 0"):
        make_epoch(3, flux=[1.0, np.inf, 1.0], ivar=[100.0, 100.0, 100.0])


def test_garbage_flux_at_zero_ivar_pixels_is_accepted():
    # Deliberate: a masked pixel's flux is never read, so cosmic-ray spikes and pipeline
    # NaNs are allowed to sit there rather than forcing the user to scrub their arrays.
    epoch = make_epoch(4, flux=[1.0, np.nan, 1e30, 1.0], ivar=[100.0, 0.0, 0.0, 100.0])
    assert epoch.good.tolist() == [True, False, False, True]
    assert np.all(np.isfinite(epoch.effective_ivar))


def test_mask_false_does_not_license_non_finite_flux():
    # Resolution of a spec ambiguity: the finite-flux rule is keyed on ivar > 0 alone, so
    # `mask` (a convenience layer folded in only by effective_ivar) cannot excuse a NaN.
    with pytest.raises(ValueError, match="finite wherever ivar > 0"):
        make_epoch(
            3,
            flux=[1.0, np.nan, 1.0],
            ivar=[100.0, 100.0, 100.0],
            mask=[True, False, True],
        )


@pytest.mark.parametrize("field", ["bjd", "v_bary"])
def test_non_finite_times_and_velocities_raise(field):
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        make_epoch(3, **{field: np.nan})


def test_bad_mask_raises():
    with pytest.raises(ValueError, match="same length as wave"):
        make_epoch(3, mask=[True, False])
    with pytest.raises(ValueError, match="1-D"):
        make_epoch(4, mask=np.ones((2, 2), dtype=bool))


# ------------------------------------------------------------------- good / effective_ivar


def test_good_and_effective_ivar_without_mask():
    epoch = make_epoch(4, ivar=[100.0, 0.0, 25.0, 0.0])
    assert epoch.mask is None
    np.testing.assert_array_equal(epoch.good, [True, False, True, False])
    np.testing.assert_allclose(epoch.effective_ivar, [100.0, 0.0, 25.0, 0.0])


def test_good_and_effective_ivar_with_mask():
    epoch = make_epoch(
        4,
        ivar=[100.0, 0.0, 25.0, 49.0],
        mask=[True, True, False, True],  # True = GOOD; pixel 2 flagged out
    )
    assert epoch.mask.dtype == np.bool_
    np.testing.assert_array_equal(epoch.good, [True, False, False, True])
    np.testing.assert_allclose(epoch.effective_ivar, [100.0, 0.0, 0.0, 49.0])
    # the mask is folded in *only* by effective_ivar: raw ivar is untouched
    np.testing.assert_allclose(epoch.ivar, [100.0, 0.0, 25.0, 49.0])


def test_effective_ivar_returns_a_fresh_array():
    epoch = make_epoch(3)
    first = epoch.effective_ivar
    first[:] = -1.0
    np.testing.assert_allclose(epoch.effective_ivar, 100.0)


# -------------------------------------------------------------------------------- Dataset


def test_dataset_len_iter_and_indexing():
    epochs = [make_epoch(3, bjd=2459000.0 + k) for k in range(3)]
    ds = Dataset(epochs)
    assert len(ds) == 3
    assert ds.n_epochs == 3
    assert ds[0] is epochs[0]
    assert ds[-1] is epochs[-1]
    assert list(ds) == epochs
    assert isinstance(ds.epochs, tuple)  # list coerced to tuple
    assert ds.frame == "topocentric"


def test_dataset_array_properties_preserve_input_order():
    ds = Dataset(
        [
            make_epoch(3, bjd=2459005.0, v_bary=-13.5),
            make_epoch(3, bjd=2459000.0, v_bary=7.25),
        ]
    )
    assert ds.bjd.dtype == np.float64
    np.testing.assert_allclose(ds.bjd, [2459005.0, 2459000.0])
    np.testing.assert_allclose(ds.v_bary, [-13.5, 7.25])
    assert ds.bjd.shape == (ds.n_epochs,)


def test_dataset_instruments_are_unique_and_sorted():
    ds = Dataset(
        [
            make_epoch(3, instrument="UVES"),
            make_epoch(3, instrument="HERMES"),
            make_epoch(3, instrument="UVES"),
            make_epoch(3, instrument="APOGEE"),
        ]
    )
    assert ds.instruments == ("APOGEE", "HERMES", "UVES")


def test_dataset_validation():
    with pytest.raises(ValueError, match="at least one epoch"):
        Dataset([])
    with pytest.raises(ValueError, match="must be an EpochData"):
        Dataset([make_epoch(3), "not an epoch"])
    with pytest.raises(ValueError, match="not a single EpochData"):
        Dataset(make_epoch(3))
    with pytest.raises(ValueError, match="frame must be one of"):
        Dataset([make_epoch(3)], frame="heliocentric")


def test_dataset_accepts_both_frames():
    for frame in ("topocentric", "barycentric"):
        assert Dataset([make_epoch(3)], frame=frame).frame == frame


def test_summary_reports_epochs_instruments_and_good_fraction():
    ds = Dataset(
        [
            make_epoch(4, bjd=2459000.0, instrument="HERMES", ivar=[100.0, 100.0, 0.0, 0.0]),
            make_epoch(4, bjd=2459010.0, instrument="HERMES"),
            make_epoch(4, bjd=2459005.0, instrument="UVES", wave=4200.0 + 0.1 * np.arange(4)),
        ],
        frame="barycentric",
    )
    text = ds.summary()
    assert isinstance(text, str)
    assert "3 epochs" in text
    assert "HERMES" in text and "UVES" in text
    assert "2 epochs" in text  # the HERMES breakdown line
    assert "1 epoch," in text  # the UVES breakdown line, not pluralized
    assert "barycentric" in text
    assert "10.00000 d" in text  # BJD span
    assert "10 / 12" in text  # good pixels
    assert len(text.splitlines()) == 6
