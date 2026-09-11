"""The line-spread width belongs to the exposure, not to the instrument name.

HARPS observes at R = 115,000 in its high-accuracy mode and at R = 80,000 in its
high-efficiency mode, and both say ``INSTRUME = 'HARPS'``. On AI Phe six of 36 archival
spectra were the second kind and were modelled at the first width for a month, because
the reader read the right resolving power into ``RawSpectrum.resolving_power`` and
``to_epoch`` then dropped it. These tests pin the repair: the width travels with the epoch,
an instrument may be declared ``PER_EPOCH`` and is then modelled at each epoch's own width,
and pooling two widths under one number is reported rather than silent.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import albireo as ab
from albireo.data import Dataset, EpochData
from albireo.forward import PER_EPOCH, build_problem, declared_lsf_widths, with_lsf
from albireo.likelihood import marginal_loglikelihood
from albireo.preprocess import select_region
from albireo.priors import SmoothnessPrior
from albireo.simulate import InstrumentSpec, simulate_dataset, synthetic_deviation_spectrum

GRID = ab.LogGrid.from_wavelength_range(4500.0, 4530.0, dv_kms=3.0)
VEL = np.array([[20.0, -35.0, 5.0, 40.0], [-30.0, 50.0, -8.0, -55.0]])
LIGHT = [0.6, 0.4]
WAVE = np.arange(4504.0, 4526.0, 0.05)


def two_modes(*, keyed: bool):
    """Four epochs of one spectrograph in two modes: widths 5 and 9 km/s.

    ``keyed=True`` files the modes under two instrument keys, which is the explicit
    declaration; ``keyed=False`` files all four under one key, as an archive does, with
    each epoch carrying its own width, as the reader records it.
    """
    comps = [synthetic_deviation_spectrum(GRID, seed=s, margin=0.1) for s in (1, 2)]
    ds, _ = simulate_dataset(
        GRID,
        comps,
        bjd=np.arange(4.0),
        velocities=VEL,
        light_fractions=LIGHT,
        instruments={
            "HAM": InstrumentSpec(wave=WAVE, sigma_v_lsf=5.0, snr=80.0),
            "EGGS": InstrumentSpec(wave=WAVE, sigma_v_lsf=9.0, snr=80.0),
        },
        epoch_instruments=["HAM", "EGGS", "HAM", "EGGS"],
        v_bary=np.zeros(4),
        seed=2,
    )
    if keyed:
        return ds
    return Dataset(tuple(dataclasses.replace(e, instrument="HARPS") for e in ds), frame=ds.frame)


def test_the_epoch_carries_its_width_and_validates_it():
    ep = EpochData(wave=[4000.0, 4000.1], flux=[1.0, 1.0], ivar=[1.0, 1.0], bjd=0.0)
    assert ep.lsf_sigma_kms is None
    ep = EpochData(
        wave=[4000.0, 4000.1], flux=[1.0, 1.0], ivar=[1.0, 1.0], bjd=0.0, lsf_sigma_kms=2.5
    )
    assert ep.lsf_sigma_kms == 2.5
    with pytest.raises(ValueError, match="lsf_sigma_kms must be positive"):
        EpochData(
            wave=[4000.0, 4000.1], flux=[1.0, 1.0], ivar=[1.0, 1.0], bjd=0.0, lsf_sigma_kms=0.0
        )
    with pytest.raises(ValueError, match="lsf_sigma_kms"):
        EpochData(
            wave=[4000.0, 4000.1], flux=[1.0, 1.0], ivar=[1.0, 1.0], bjd=0.0, lsf_sigma_kms=np.nan
        )


def test_trimming_and_the_simulator_keep_the_width():
    ds = two_modes(keyed=False)
    np.testing.assert_allclose(ds.lsf_sigma_kms, [5.0, 9.0, 5.0, 9.0])
    trimmed = select_region(ds[1], 4510.0, 4520.0)
    assert trimmed.lsf_sigma_kms == 9.0, "a derived epoch must not lose its width"
    assert declared_lsf_widths(ds, "HARPS") == {5.0: [0, 2], 9.0: [1, 3]}


def test_summary_lists_the_widths_under_one_key():
    text = two_modes(keyed=False).summary()
    assert "5.000 km/s x2" in text and "9.000 km/s x2" in text


def test_per_epoch_matches_two_explicit_keys_exactly():
    prior = SmoothnessPrior(jnp.full(2, 50.0), jnp.full(2, 2.0))
    keyed = build_problem(
        GRID,
        two_modes(keyed=True),
        velocities=VEL,
        light_fractions=LIGHT,
        lsf_sigma_v={"HAM": 5.0, "EGGS": 9.0},
    )
    pooled = build_problem(
        GRID,
        two_modes(keyed=False),
        velocities=VEL,
        light_fractions=LIGHT,
        lsf_sigma_v={"HARPS": PER_EPOCH},
    )
    assert len(pooled.groups) == 2
    assert {g.instrument for g in pooled.groups} == {"HARPS"}
    assert sorted(g.kernel.shape[-1] for g in pooled.groups) == sorted(
        g.kernel.shape[-1] for g in keyed.groups
    )
    a = marginal_loglikelihood(keyed, prior)
    b = marginal_loglikelihood(pooled, prior)
    np.testing.assert_allclose(float(a.log_likelihood), float(b.log_likelihood), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(a.d_hat), np.asarray(b.d_hat), atol=1e-12)


def test_pooling_two_widths_under_one_number_warns_and_names_the_epochs():
    ds = two_modes(keyed=False)
    with pytest.warns(RuntimeWarning, match=r"2 different LSF widths.*\[1, 3\].*PER_EPOCH"):
        build_problem(GRID, ds, velocities=VEL, light_fractions=LIGHT, lsf_sigma_v={"HARPS": 5.0})


def test_one_declared_width_under_one_key_is_silent(recwarn):
    ds = two_modes(keyed=True)
    build_problem(
        GRID, ds, velocities=VEL, light_fractions=LIGHT, lsf_sigma_v={"HAM": 5.0, "EGGS": 9.0}
    )
    assert not [w for w in recwarn if "LSF widths" in str(w.message)]


def test_per_epoch_refuses_an_epoch_without_a_width():
    ds = two_modes(keyed=False)
    epochs = list(ds)
    epochs[2] = dataclasses.replace(epochs[2], lsf_sigma_kms=None)
    ds = Dataset(tuple(epochs), frame=ds.frame)
    with pytest.raises(ValueError, match=r"1 epoch\(s\) declare none \(indices \[2\]\)"):
        build_problem(
            GRID, ds, velocities=VEL, light_fractions=LIGHT, lsf_sigma_v={"HARPS": PER_EPOCH}
        )


def test_per_epoch_refuses_anchors_and_traced_swaps():
    ds = two_modes(keyed=False)
    with pytest.raises(ValueError, match="cannot carry lsf_anchors_angstrom"):
        build_problem(
            GRID,
            ds,
            velocities=VEL,
            light_fractions=LIGHT,
            lsf_sigma_v={"HARPS": PER_EPOCH},
            lsf_anchors_angstrom={"HARPS": [4505.0, 4525.0]},
        )
    problem = build_problem(
        GRID, ds, velocities=VEL, light_fractions=LIGHT, lsf_sigma_v={"HARPS": PER_EPOCH}
    )
    with pytest.raises(ValueError, match="not a parameter"):
        with_lsf(problem, {"HARPS": PER_EPOCH})


def test_the_model_takes_per_epoch_and_refuses_an_lsf_site():
    ds = two_modes(keyed=False)
    model = ab.MarginalOrbitModel(
        GRID, ds, light_fractions=LIGHT, lsf_sigma_v={"HARPS": PER_EPOCH}, v_rel_max_kms=120.0
    )
    assert (
        model.problem.kernel_radius
        == build_problem(
            GRID,
            two_modes(keyed=True),
            velocities=VEL,
            light_fractions=LIGHT,
            lsf_sigma_v={"HAM": 5.0, "EGGS": 9.0},
        ).kernel_radius
    ), "the widest declared width fixes the kernel radius"
    theta = {"velocity": jnp.asarray(VEL), "lsf_sigma": jnp.array([5.0])}
    with pytest.raises(ValueError, match="declared PER_EPOCH"):
        model.problem_at(theta)


def test_the_facade_accepts_per_epoch_and_reports_it():
    ds = two_modes(keyed=False)
    dis = ab.Disentangler(
        ds,
        components=[ab.Star("A", light=0.6), ab.Star("B", light=0.4)],
        velocities=VEL,
        lsf={"HARPS": ab.LSF.per_epoch()},
    )
    assert dis._widest_lsf() == 9.0 and dis._narrowest_lsf() == 5.0
    assert len(dis.model.problem.groups) == 2
    text = dis.explain()
    assert "per epoch, from the files (5.000 km/s x2, 9.000 km/s x2)" in text
    # The string form is the TOML spelling and means the same thing.
    same = ab.Disentangler(
        ds,
        components=[ab.Star("A", light=0.6), ab.Star("B", light=0.4)],
        velocities=VEL,
        lsf={"HARPS": "per-epoch"},
    )
    assert same._widest_lsf() == 9.0


def test_the_facade_refuses_per_epoch_when_an_epoch_declares_nothing():
    ds = two_modes(keyed=False)
    epochs = list(ds)
    epochs[0] = dataclasses.replace(epochs[0], lsf_sigma_kms=None)
    ds = Dataset(tuple(epochs), frame=ds.frame)
    with pytest.raises(ValueError, match=r"declare none \(indices \[0\]\)"):
        ab.Disentangler(
            ds,
            components=[ab.Star("A", light=0.6), ab.Star("B", light=0.4)],
            velocities=VEL,
            lsf={"HARPS": ab.LSF.per_epoch()},
        )


def test_todcor_convolves_per_epoch_like_two_keys():
    from albireo.todcor import Template, todcor

    comps = [synthetic_deviation_spectrum(GRID, seed=s, margin=0.1) for s in (1, 2)]
    templates = [
        Template(name=n, grid=GRID, deviation=np.asarray(c), sigma_kms=0.0)
        for n, c in zip(("A", "B"), comps, strict=True)
    ]
    keyed = todcor(
        two_modes(keyed=True),
        templates,
        v_range=(-80.0, 80.0),
        light=LIGHT,
        lsf_sigma_v={"HAM": 5.0, "EGGS": 9.0},
    )
    pooled = todcor(
        two_modes(keyed=False),
        templates,
        v_range=(-80.0, 80.0),
        light=LIGHT,
        lsf_sigma_v={"HARPS": PER_EPOCH},
    )
    np.testing.assert_allclose(pooled.velocity, keyed.velocity, atol=1e-9)
    np.testing.assert_allclose(pooled.sigma, keyed.sigma, atol=1e-9)
