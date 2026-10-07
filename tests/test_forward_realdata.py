"""Regression tests of the forward model on configurations from archival spectra.

Every configuration here occurs in ESO Phase-3 FEROS data (``internal/design.md`` D30):
per-exposure wavelength grids, non-finite flux at masked pixels, and epochs that extend
past the model grid. The simulator produces none of them, so tests on simulated data do
not cover them.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import albireo as ab
from albireo.forward import build_problem

RNG = np.random.default_rng(4114)

GRID = ab.LogGrid.from_wavelength_range(4500.0, 4530.0, dv_kms=3.0)
LIGHT = [0.6, 0.4]
PRIOR = ab.SmoothnessPrior(tau=jnp.array([300.0, 300.0]), eta=jnp.array([5.0, 5.0]))


def _dataset(offsets, lengths=None, flux=None, ivar=None, wave0=4505.0, n_default=400):
    """Epochs sharing an instrument, each with its own wavelength array."""
    lengths = lengths or [n_default] * len(offsets)
    epochs = []
    for j, (off, n) in enumerate(zip(offsets, lengths, strict=True)):
        wave = wave0 + off + 0.05 * np.arange(n)
        epochs.append(
            ab.EpochData(
                wave=wave,
                flux=(1.0 + 0.01 * RNG.standard_normal(n)) if flux is None else flux,
                ivar=np.full(n, 1e4) if ivar is None else ivar,
                bjd=2453000.0 + j,
                instrument="A",
            )
        )
    return ab.Dataset(epochs, frame="barycentric")


def _loglike(dataset):
    problem = build_problem(
        GRID,
        dataset,
        velocities=np.zeros((2, dataset.n_epochs)),
        light_fractions=LIGHT,
        lsf_sigma_v={"A": 5.0},
    )
    return float(ab.marginal_loglikelihood(problem, PRIOR).log_likelihood)


# ----------------------------------------------------------------- per-exposure grids


def test_epochs_with_different_grids_are_grouped_not_rejected():
    """A pipeline that shifts before resampling gives one grid per exposure.

    ``build_problem`` must not require bit-identical ``wave`` arrays within an instrument.
    Archival data does not satisfy that: 51 ESO FEROS spectra of one target are on 28
    distinct grids. The documented workaround, relabelling epochs as separate
    instruments, would also split the LSF and response tables.
    """
    ds = _dataset([0.0, 1e-4, 0.02], lengths=[400, 398, 401])
    problem = build_problem(
        GRID,
        ds,
        velocities=np.zeros((2, 3)),
        light_fractions=LIGHT,
        lsf_sigma_v={"A": 5.0},  # one width for the instrument, not one per grid
    )
    assert len(problem.groups) == 3
    assert {g.instrument for g in problem.groups} == {"A"}
    assert problem.n_epochs == 3
    assert np.isfinite(float(ab.marginal_loglikelihood(problem, PRIOR).log_likelihood))


def test_epochs_sharing_a_grid_stay_in_one_group():
    """The common case must not regress into one group per epoch."""
    ds = _dataset([0.0, 0.0, 0.0])
    problem = build_problem(
        GRID, ds, velocities=np.zeros((2, 3)), light_fractions=LIGHT, lsf_sigma_v={"A": 5.0}
    )
    assert len(problem.groups) == 1
    assert problem.groups[0].epoch_indices == (0, 1, 2)


def test_grouping_is_exact_not_approximate():
    """Two grids that differ by a whole pixel must not be merged by the hasher."""
    ds = _dataset([0.0, 0.05])
    problem = build_problem(
        GRID, ds, velocities=np.zeros((2, 2)), light_fractions=LIGHT, lsf_sigma_v={"A": 5.0}
    )
    assert len(problem.groups) == 2


def test_split_groups_give_the_same_answer_as_one_group():
    """Grouping is a performance decision; it must not change the likelihood."""
    shared = _dataset([0.0, 0.0])
    # Same data, but the second epoch's wave array is a distinct (equal-valued) object
    # perturbed far below any tolerance, forcing a second group.
    epochs = list(shared)
    nudged = ab.EpochData(
        wave=epochs[1].wave + 1e-9,
        flux=epochs[1].flux,
        ivar=epochs[1].ivar,
        bjd=epochs[1].bjd,
        instrument=epochs[1].instrument,
    )
    split = ab.Dataset([epochs[0], nudged], frame="barycentric")
    problem_split = build_problem(
        GRID, split, velocities=np.zeros((2, 2)), light_fractions=LIGHT, lsf_sigma_v={"A": 5.0}
    )
    assert len(problem_split.groups) == 2
    assert _loglike(split) == pytest.approx(_loglike(shared), rel=1e-10)


def test_marginal_orbit_model_deduplicates_instruments_across_groups():
    """The ``lsf_sigma`` site is per instrument; one entry per group would be wrong."""
    ds = _dataset([0.0, 1e-4, 0.02])
    model = ab.MarginalOrbitModel(
        GRID,
        ds,
        light_fractions=LIGHT,
        lsf_sigma_v={"A": 5.0},
        v_rel_max_kms=60.0,
        prior=PRIOR,
    )
    assert len(model.problem.groups) == 3
    assert model.instruments == ("A",), "one entry per instrument, not per operator group"


# -------------------------------------------------- invalid flux at zero-weight pixels


@pytest.mark.parametrize("garbage", [np.nan, np.inf, -np.inf, 1e300])
def test_non_finite_flux_at_zero_weight_pixels_is_ignored(garbage):
    """A masked pixel's flux must never be read, as ``data.py`` documents.

    Every consumer multiplies ``z = flux - r * base`` by the weight, but ``0 * nan`` is
    ``nan``. If ``z`` were formed unmasked, one such pixel would make the entire marginal
    log-likelihood ``nan``. :func:`albireo.preprocess.normalize` writes this input
    wherever the fitted continuum falls to a small fraction of its median, which is
    routine on a merged echelle spectrum.
    """
    n = 400
    flux = 1.0 + 0.01 * RNG.standard_normal(n)
    ivar = np.full(n, 1e4)
    ivar[100:140] = 0.0
    poisoned = flux.copy()
    poisoned[100:140] = garbage

    reference = _loglike(_dataset([0.0, 0.0], flux=flux, ivar=ivar))
    assert np.isfinite(reference)
    assert _loglike(_dataset([0.0, 0.0], flux=poisoned, ivar=ivar)) == pytest.approx(
        reference, rel=0, abs=0
    )


def test_a_fully_masked_epoch_does_not_poison_the_likelihood():
    """The limiting case of the above: an exposure with no usable pixel."""
    n = 400
    flux = np.full(n, np.nan)
    ivar = np.zeros(n)
    good = _dataset([0.0], flux=1.0 + 0.01 * RNG.standard_normal(n))
    both = ab.Dataset(
        [
            good[0],
            ab.EpochData(wave=good[0].wave, flux=flux, ivar=ivar, bjd=2453099.0, instrument="A"),
        ],
        frame="barycentric",
    )
    assert np.isfinite(_loglike(both))


# ------------------------------------------------------------------ grid-margin guards


def test_row_support_warning_is_not_triggered_by_a_narrow_model_grid():
    """The support statistic must exclude rows that have no entries in the operator.

    Native pixels lying entirely outside the model grid have no rebin entries, and
    including them gives a support of ``0 - int64_max + 1``. With more than half the epoch
    outside, the reported median is then the int64 minimum and every number in the
    warning is wrong. False warnings lead users to ignore valid ones.
    """
    n = 2000
    wave = 4470.0 + 0.05 * np.arange(n)  # 4470-4570 A: most of it outside GRID
    ds = ab.Dataset(
        [
            ab.EpochData(
                wave=wave,
                flux=np.ones(n),
                ivar=np.where((wave > 4506.0) & (wave < 4524.0), 1e4, 0.0),
                bjd=2453000.0,
                instrument="A",
            )
        ],
        frame="barycentric",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        build_problem(
            GRID, ds, velocities=np.zeros((2, 1)), light_fractions=LIGHT, lsf_sigma_v={"A": 5.0}
        )


def test_a_region_disjoint_from_the_model_grid_says_so():
    """A window disjoint from the grid must be reported as such, not as 'empty rebin operator'."""
    n = 500
    wave = 4300.0 + 0.05 * np.arange(n)  # 4300-4325 A, disjoint from GRID's 4500-4530
    ds = ab.Dataset(
        [
            ab.EpochData(
                wave=wave, flux=np.ones(n), ivar=np.full(n, 1e4), bjd=2453000.0, instrument="A"
            )
        ],
        frame="barycentric",
    )
    with pytest.raises(ValueError, match="does not overlap"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        build_problem(
            GRID, ds, velocities=np.zeros((2, 1)), light_fractions=LIGHT, lsf_sigma_v={"A": 5.0}
        )


def test_weighted_pixels_outside_the_model_grid_warn():
    """The guard reports weighted data that would otherwise be dropped silently."""
    n = 1000
    wave = 4490.0 + 0.05 * np.arange(n)  # starts 10 A blueward of the grid
    ds = ab.Dataset(
        [
            ab.EpochData(
                wave=wave, flux=np.ones(n), ivar=np.full(n, 1e4), bjd=2453000.0, instrument="A"
            )
        ],
        frame="barycentric",
    )
    with pytest.warns(RuntimeWarning, match="outside the model grid"):
        build_problem(
            GRID, ds, velocities=np.zeros((2, 1)), light_fractions=LIGHT, lsf_sigma_v={"A": 5.0}
        )


def test_log_grid_covering_leaves_the_shift_and_lsf_margin():
    """``LogGrid.covering`` must clear the data by more than shift + kernel radius."""
    ds = _dataset([0.0], lengths=[2000])
    v_margin, sigma = 90.0, 2.652
    grid = ab.LogGrid.covering(ds, dv_kms=1.5, v_margin_kms=v_margin, lsf_sigma_kms=sigma)
    data_lo, data_hi = ds[0].wave[0], ds[0].wave[-1]
    assert grid.wave[0] < data_lo and grid.wave[-1] > data_hi
    # Margin in model pixels must exceed the shift plus the kernel radius.
    margin_px = np.log(data_lo / grid.wave[0]) / grid.dx
    needed = abs(grid.velocity_to_pixels(v_margin)) + np.ceil(4.0 * sigma / grid.dv_kms)
    assert margin_px >= needed
    # A grid built that way must not trigger the coverage warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        build_problem(
            grid, ds, velocities=np.zeros((2, 1)), light_fractions=LIGHT, lsf_sigma_v={"A": sigma}
        )


# ------------------------------------------------------------------- prior conditioning


@pytest.mark.parametrize("log_eta", [-30.0, -40.0, -60.0])
def test_prior_logdet_stays_finite_at_tiny_eta(log_eta):
    """ML-II has no lower bound on ``log_eta`` and will reach the rounding floor.

    The pentadiagonal Cholesky pivot is a difference of like-sized terms. Below
    ``eta/tau ~ 1e-13`` it rounds non-positive, where ``sqrt`` would return ``nan`` and
    make the whole likelihood ``nan``.
    """
    from albireo.assembly import prior_logdet

    prior = ab.SmoothnessPrior(tau=jnp.array([1.0e3]), eta=jnp.exp(jnp.array([log_eta])))
    assert np.isfinite(float(prior_logdet(prior, 500)))
