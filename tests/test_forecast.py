"""Tests for :mod:`albireo.forecast` — the observing-strategy forecast (D47).

Two things carry the module and both are pinned here against independent oracles.

The **linear algebra** is checked against dense NumPy on a small problem: the
log-determinants, the pointwise band, the effective parameter count (which the module
gets from one directional derivative of ``log det`` rather than a trace estimator), and
the worst-determined modes (which it gets from subspace iteration on the banded factor).
Each has a two-line dense equivalent, so there is no reason for any of them to be taken
on trust.

The **claim that makes the feature possible** — that the posterior covariance of the
component spectra does not depend on the observed fluxes — is pinned by overwriting every
flux with garbage and requiring the forecast back bit-identical. That is the whole
license for forecasting epochs that have not been taken, so it is asserted with ``==``
rather than a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

import albireo as ab
from albireo.assembly import band_block_tridiagonal
from albireo.data import Dataset, EpochData
from albireo.forecast import _default_scales, _separation_diagnostics, sensitivity_forecast
from albireo.forward import build_problem
from albireo.grids import LogGrid
from albireo.priors import SmoothnessPrior
from albireo.simulate import InstrumentSpec, OrbitParams, simulate_dataset
from albireo.simulate import synthetic_deviation_spectrum as synth
from albireo.solver import dense_from_block_tridiagonal

ORBIT = OrbitParams(period=6.0, t_peri=0.0, ecc=0.0, omega=0.0, k=(40.0, 60.0))
LIGHT = (0.6, 0.4)
LSF = {"A": 8.25}


@pytest.fixture(scope="module")
def grid():
    return LogGrid.from_wavelength_range(4500.0, 4512.0, dv_kms=6.0)


@pytest.fixture(scope="module")
def dataset(grid):
    ds, _ = simulate_dataset(
        grid,
        [synth(grid, n_lines=6, seed=s, margin=0.1) for s in (1, 2)],
        bjd=np.array([0.7, 2.2, 4.9, 6.0]),
        instruments={
            "A": InstrumentSpec(wave=np.arange(4502.0, 4510.0, 0.08), sigma_v_lsf=8.25, snr=60.0)
        },
        light_fractions=LIGHT,
        orbit=ORBIT,
        seed=4,
    )
    return ds


@pytest.fixture(scope="module")
def prior():
    return SmoothnessPrior(tau=np.array([1e3, 1e3]), eta=np.array([1e-2, 1e-2]))


def forecast(grid, dataset, prior, **kwargs):
    kwargs.setdefault("orbit", ORBIT)
    kwargs.setdefault("light_fractions", LIGHT)
    kwargs.setdefault("lsf_sigma_v", LSF)
    return sensitivity_forecast(grid, dataset, prior=prior, **kwargs)


def dense_posterior(grid, dataset, prior):
    """``(Sigma, Lambda_p)`` densely, in the solver's interleaved (pixel-major) layout."""
    problem = build_problem(
        grid,
        dataset,
        velocities=ORBIT.component_velocities(dataset.bjd),
        light_fractions=LIGHT,
        lsf_sigma_v=LSF,
    )
    b_nat = max(problem.natural_half_bandwidth, prior.half_bandwidth)
    lam = dense_from_block_tridiagonal(band_block_tridiagonal(problem, prior, b_nat, None))
    n_comp, n_pix = 2, grid.n
    order = np.arange(n_comp * n_pix).reshape(n_comp, n_pix).T.reshape(-1)
    return np.linalg.inv(lam), prior.dense(n_pix)[np.ix_(order, order)], lam


# ---------------------------------------------------------------------------
# The load-bearing claim
# ---------------------------------------------------------------------------


def test_forecast_does_not_depend_on_the_fluxes(grid, dataset, prior, rng):
    """The posterior covariance has no flux in it, so the forecast must be identical.

    Not "close": identical. ``Lambda_p + A^T W A`` is assembled from weights, response,
    shifts, light fractions and kernels, and the right-hand side ``A^T W z`` is never
    formed — so replacing every flux with noise a hundred times the continuum may not
    move a single bit. This is what licenses forecasting an observation that has not
    happened.
    """
    garbage = Dataset(
        [
            EpochData(
                wave=e.wave,
                flux=rng.normal(0.0, 100.0, e.wave.size),
                ivar=e.ivar,
                bjd=e.bjd,
                v_bary=e.v_bary,
                instrument=e.instrument,
            )
            for e in dataset
        ],
        frame=dataset.frame,
    )
    a = forecast(grid, dataset, prior, n_modes=3)
    b = forecast(grid, garbage, prior, n_modes=3)
    assert a.logdet_posterior == b.logdet_posterior
    assert a.p_eff == b.p_eff
    assert np.array_equal(a.component_std, b.component_std)
    assert np.array_equal(a.mode_std, b.mode_std)
    assert np.array_equal(a.region, b.region)


# ---------------------------------------------------------------------------
# Against dense linear algebra
# ---------------------------------------------------------------------------


def test_logdets_match_dense(grid, dataset, prior):
    fc = forecast(grid, dataset, prior, n_modes=0)
    _, lam_p, lam = dense_posterior(grid, dataset, prior)
    assert fc.logdet_posterior == pytest.approx(np.linalg.slogdet(lam)[1], rel=1e-10)
    assert fc.logdet_prior == pytest.approx(np.linalg.slogdet(lam_p)[1], rel=1e-10)
    # The expected KL from prior to posterior, which is what information_nats means.
    assert fc.information_nats == pytest.approx(
        0.5 * (np.linalg.slogdet(lam)[1] - np.linalg.slogdet(lam_p)[1]), rel=1e-10
    )


def test_component_std_matches_dense(grid, dataset, prior):
    fc = forecast(grid, dataset, prior, n_modes=0)
    sigma, lam_p, _ = dense_posterior(grid, dataset, prior)
    want = np.sqrt(np.diag(sigma)).reshape(grid.n, 2).T
    assert np.allclose(fc.component_std, want, rtol=1e-8)
    prior_want = np.sqrt(np.diag(np.linalg.inv(lam_p))).reshape(grid.n, 2).T
    assert np.allclose(fc.prior_std, prior_want, rtol=1e-8)


def test_p_eff_matches_dense_trace(grid, dataset, prior):
    """``p_eff`` comes from d(log det)/d(log noise scale); the oracle is the trace itself."""
    fc = forecast(grid, dataset, prior, n_modes=0)
    sigma, lam_p, lam = dense_posterior(grid, dataset, prior)
    assert fc.p_eff == pytest.approx(float(np.trace(sigma @ (lam - lam_p))), rel=1e-8)
    assert 0.0 < fc.p_eff < fc.n_linear


def test_modes_match_dense_submatrix(grid, dataset, prior):
    """Subspace iteration reaches the eigenvalues of ``Sigma`` restricted to the region."""
    fc = forecast(grid, dataset, prior, n_modes=3)
    sigma, _, _ = dense_posterior(grid, dataset, prior)
    keep = fc.region.T.reshape(-1)
    want = np.sqrt(np.linalg.eigvalsh(sigma[np.ix_(keep, keep)])[::-1][:3])
    assert np.allclose(fc.mode_std, want, rtol=1e-6)
    assert fc.mode_residual < 1e-6
    # Unit-norm modes, zero outside the region, and no all-zero row.
    flat = fc.mode_vectors.transpose(0, 2, 1).reshape(3, -1)
    assert np.allclose(np.linalg.norm(flat, axis=1), 1.0)
    assert np.all(flat[:, ~keep] == 0.0)


def test_leading_mode_is_the_component_exchange(grid, dataset, prior):
    """The worst direction is a delocalized see-saw across components, not one pixel.

    This is ``docs/math.md`` §5.1's ``k = 0`` mode, and it is what the region
    restriction exists to expose: without it the answer is a grid-margin pixel, which
    reports how much margin the grid was given rather than anything about the epochs.
    """
    fc = forecast(grid, dataset, prior, n_modes=1)
    mode = fc.mode_vectors[0]
    power = (mode**2).sum(axis=1)
    assert power.min() > 0.05, "the leading mode should involve both components"
    # Inverse participation ratio: 1 for a single-pixel mode, ~n for a delocalized one.
    participation = 1.0 / float((mode**4).sum())
    assert participation > 0.1 * fc.n_covered, "the mode should be spread, not one edge pixel"
    # And it should be barely better determined than the prior, per §5.1.
    assert 0.8 < fc.worst_mode_gain < 3.0


# ---------------------------------------------------------------------------
# Behaviour a forecast has to have
# ---------------------------------------------------------------------------


def test_more_epochs_never_hurt(grid, dataset, prior):
    """Adding data can only add information — every summary must move the right way.

    Run through the baseline mechanism, so the modes of the two designs are eigenvalues
    of submatrices of the *same* coordinates: extra epochs add a positive-semidefinite
    term to the precision, which orders the covariances, which orders the eigenvalues of
    any common submatrix. Compared over two independently derived regions there would be
    no such guarantee, and the assertion would be a coincidence rather than a theorem.
    """
    design = Dataset(
        [*dataset, *ab.plan_epochs(dataset[0], bjd=[1.4, 3.6, 5.2])], frame=dataset.frame
    )
    many = forecast(grid, design, prior, n_modes=3, baseline=range(dataset.n_epochs))
    few = many.baseline
    assert many.logdet_posterior > few.logdet_posterior
    assert many.information_nats > few.information_nats
    assert many.p_eff > few.p_eff
    assert np.all(many.component_std <= few.component_std * (1 + 1e-9))
    assert np.all(many.mode_std <= few.mode_std * (1 + 1e-6))


def test_prior_is_the_ceiling(grid, dataset, prior):
    """A design with essentially no weight recovers the prior and says so."""
    faint = Dataset(
        [
            EpochData(
                wave=e.wave,
                flux=e.flux,
                ivar=e.ivar * 1e-12,
                bjd=e.bjd,
                v_bary=e.v_bary,
                instrument=e.instrument,
            )
            for e in dataset
        ],
        frame=dataset.frame,
    )
    fc = forecast(grid, faint, prior, n_modes=2)
    assert np.allclose(fc.component_std, fc.prior_std, rtol=1e-4)
    assert fc.p_eff < 1e-3
    assert fc.information_nats < 1e-3
    assert fc.worst_mode_gain == pytest.approx(1.0, rel=1e-4)


def test_jitter_removes_information(grid, dataset, prior):
    fc = forecast(grid, dataset, prior, n_modes=0)
    inflated = forecast(grid, dataset, prior, n_modes=0, jitter=3.0)
    assert inflated.p_eff < fc.p_eff
    assert inflated.information_nats < fc.information_nats
    assert np.all(inflated.component_std >= fc.component_std * (1 - 1e-9))


def _t_conj_of(orbit: OrbitParams) -> float:
    """The ``t_conj`` this orbit's ``t_peri`` corresponds to, without assuming a convention.

    ``t_peri_from_t_conj`` is ``t_conj - m_conj(e, omega) * P / 2pi``, an exact shift, so
    evaluating it at zero recovers the shift and inverting is one subtraction.
    """
    from albireo.kepler import t_peri_from_t_conj

    shift = float(t_peri_from_t_conj(0.0, period=orbit.period, ecc=orbit.ecc, omega=orbit.omega))
    return orbit.t_peri - shift


def test_orbit_forms_agree(grid, dataset, prior):
    """An ``OrbitParams`` and the equivalent theta mapping are the same design."""
    theta = {
        "period": ORBIT.period,
        "t_conj": _t_conj_of(ORBIT),
        "secosw": np.sqrt(ORBIT.ecc) * np.cos(ORBIT.omega),
        "sesinw": np.sqrt(ORBIT.ecc) * np.sin(ORBIT.omega),
        "k": np.asarray(ORBIT.k),
    }
    a = forecast(grid, dataset, prior, n_modes=0)
    b = forecast(grid, dataset, prior, n_modes=0, orbit=theta)
    assert b.logdet_posterior == pytest.approx(a.logdet_posterior, rel=1e-10)
    assert np.allclose(a.component_std, b.component_std, rtol=1e-8)


def test_explicit_velocities(grid, dataset, prior):
    vel = ORBIT.component_velocities(dataset.bjd)
    a = forecast(grid, dataset, prior, n_modes=0)
    b = forecast(grid, dataset, prior, n_modes=0, orbit=None, velocities=vel)
    assert b.logdet_posterior == pytest.approx(a.logdet_posterior, rel=1e-12)


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


def test_baseline_shares_the_region_and_reports_the_difference(grid, dataset, prior):
    design = Dataset(
        [*dataset, *ab.plan_epochs(dataset[0], bjd=[1.4, 3.6, 5.2])], frame=dataset.frame
    )
    fc = forecast(grid, design, prior, n_modes=3, baseline=range(dataset.n_epochs))
    assert fc.baseline is not None
    assert fc.baseline.baseline is None
    assert fc.n_planned == 3
    assert fc.baseline.n_epochs == dataset.n_epochs
    # The comparison is only meaningful on one subspace, and the design derives it.
    assert np.array_equal(fc.region, fc.baseline.region)
    assert fc.gain_nats == pytest.approx(fc.information_nats - fc.baseline.information_nats)
    assert fc.gain_nats > 0.0
    # And the baseline must equal the standalone forecast of the same epochs, apart from
    # the region it is summarized over.
    alone = forecast(grid, dataset, prior, n_modes=0)
    assert fc.baseline.logdet_posterior == pytest.approx(alone.logdet_posterior, rel=1e-10)


def test_a_spread_cadence_beats_one_aliased_to_the_period(prior):
    """The result the module exists to produce, and the one Var(Delta) alone gets wrong.

    Epochs at half-period intervals visit the two extreme differential velocities over
    and over: the *largest* RMS differential velocity of any design, and a poor one.
    Spreading the same number over phase lowers that RMS and is nevertheless worth far
    more, which is why ``blind_fraction`` and the exact numbers are reported rather than
    ``rms_delta_kms`` alone.
    """
    grid = LogGrid.from_wavelength_range(4490.0, 4530.0, dv_kms=5.0)
    inst = InstrumentSpec(wave=np.arange(4498.0, 4522.0, 0.06), sigma_v_lsf=6.0, snr=70.0)
    period = 6.0
    have = np.array([0.0, 3.0, 6.0, 9.0])
    ds, _ = simulate_dataset(
        grid,
        [synth(grid, n_lines=10, seed=s, margin=0.08) for s in (1, 2)],
        bjd=have,
        instruments={"A": inst},
        light_fractions=LIGHT,
        orbit=OrbitParams(period=period, t_peri=0.0, ecc=0.0, omega=0.0, k=(40.0, 60.0)),
        v_bary=np.zeros(have.size),
        frame="barycentric",
        seed=1,
    )
    orbit = OrbitParams(period=period, t_peri=0.0, ecc=0.0, omega=0.0, k=(40.0, 60.0))
    kw = {
        "light_fractions": LIGHT,
        "lsf_sigma_v": {"A": 6.0},
        "prior": prior,
        "orbit": orbit,
        "n_modes": 3,
    }
    plans = {
        "aliased": have[-1] + period * np.arange(1, 9) / 2.0,
        "spread": have[-1] + period * (np.arange(8) + 0.5) / 8.0,
    }
    out = {}
    for name, t_new in plans.items():
        design = Dataset([*ds, *ab.plan_epochs(ds[0], t_new)], frame=ds.frame)
        out[name] = sensitivity_forecast(grid, design, baseline=range(ds.n_epochs), **kw)

    assert out["aliased"].rms_delta_kms > out["spread"].rms_delta_kms, (
        "the aliased cadence should win on the naive Var(Delta) statistic"
    )
    assert out["spread"].gain_nats > out["aliased"].gain_nats
    assert out["spread"].mode_std[1] < out["aliased"].mode_std[1]
    assert out["spread"].blind_fraction < out["aliased"].blind_fraction


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------


def test_region_excludes_the_grid_margin(grid, dataset, prior):
    fc = forecast(grid, dataset, prior, n_modes=0)
    assert fc.region.shape == (2, grid.n)
    assert 0 < fc.n_covered < fc.region.size
    covered = grid.wave[fc.region[0]]
    # The data span 4502-4510; the region may spill by the shift-plus-kernel reach but
    # must not run to the ends of a grid built out to 4500 and 4512.
    assert covered.min() < 4502.5 and covered.max() > 4509.5
    assert covered.min() > grid.wave[0] and covered.max() < grid.wave[-1]


def test_region_window_narrows_the_summary(grid, dataset, prior):
    wide = forecast(grid, dataset, prior, n_modes=2)
    narrow = forecast(grid, dataset, prior, n_modes=2, region=(4504.0, 4506.0))
    assert narrow.n_covered < wide.n_covered
    assert np.all(grid.wave[narrow.region[0]] >= 4504.0)
    assert np.all(grid.wave[narrow.region[0]] <= 4506.0)
    # A boolean mask spelling the same window is the same region.
    mask = (grid.wave >= 4504.0) & (grid.wave <= 4506.0)
    by_mask = forecast(grid, dataset, prior, n_modes=0, region=mask)
    assert np.array_equal(narrow.region, by_mask.region)


def test_region_floor_zero_keeps_every_touched_pixel(grid, dataset, prior):
    cut = forecast(grid, dataset, prior, n_modes=0)
    everything = forecast(grid, dataset, prior, n_modes=0, region_floor=0.0)
    assert everything.n_covered > cut.n_covered


def test_region_disjoint_from_the_data_raises(grid, dataset, prior):
    with pytest.raises(ValueError, match="region is empty"):
        forecast(grid, dataset, prior, region=(4500.0, 4500.5))


# ---------------------------------------------------------------------------
# plan_epochs
# ---------------------------------------------------------------------------


def test_plan_epochs_from_an_epoch(dataset):
    plan = ab.plan_epochs(dataset[0], bjd=[10.0, 11.5], v_bary=[3.0, -4.0])
    assert len(plan) == 2
    for ep in plan:
        assert np.array_equal(ep.wave, dataset[0].wave)
        assert np.array_equal(ep.ivar, dataset[0].ivar)
        assert np.all(ep.flux == 1.0), "the placeholder flux is the continuum"
        assert ep.instrument == dataset[0].instrument
    assert [ep.bjd for ep in plan] == [10.0, 11.5]
    assert [ep.v_bary for ep in plan] == [3.0, -4.0]


def test_plan_epochs_snr_override_preserves_the_mask(dataset):
    template = EpochData(
        wave=dataset[0].wave,
        flux=dataset[0].flux,
        ivar=np.where(np.arange(dataset[0].n_pixels) < 10, 0.0, 100.0),
        bjd=0.0,
    )
    (ep,) = ab.plan_epochs(template, bjd=[5.0], snr=20.0)
    assert np.all(ep.ivar[:10] == 0.0), "a masked pixel of the template stays masked"
    assert np.all(ep.ivar[10:] == 400.0)


def test_plan_epochs_from_an_instrument_spec():
    spec = InstrumentSpec(wave=np.linspace(4500.0, 4510.0, 128), sigma_v_lsf=5.0, snr=50.0)
    plan = ab.plan_epochs(spec, bjd=[0.0, 1.0], instrument="UVES", medium="air")
    assert len(plan) == 2
    assert plan[0].instrument == "UVES" and plan[0].medium == "air"
    assert np.all(plan[0].ivar == 2500.0)
    (custom,) = ab.plan_epochs(spec, bjd=[0.0], snr=10.0)
    assert np.all(custom.ivar == 100.0)


def test_plan_epochs_rejects_nonsense(dataset):
    with pytest.raises(ValueError, match="template must be"):
        ab.plan_epochs(object(), bjd=[1.0])
    with pytest.raises(ValueError, match="snr must be positive"):
        ab.plan_epochs(dataset[0], bjd=[1.0], snr=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        ab.plan_epochs(dataset[0], bjd=[])


def test_planned_epochs_combine_with_real_ones(grid, dataset, prior):
    """The whole workflow: a Dataset of observed plus planned epochs forecasts fine."""
    design = Dataset([*dataset, *ab.plan_epochs(dataset[0], bjd=[8.0, 9.3])], frame=dataset.frame)
    fc = forecast(grid, design, prior, n_modes=2, baseline=range(dataset.n_epochs))
    assert fc.n_epochs == dataset.n_epochs + 2
    assert "planned" in fc.summary()


# ---------------------------------------------------------------------------
# Extra components, and validation
# ---------------------------------------------------------------------------


def test_non_stellar_components_are_labelled_and_priced(grid, dataset):
    prior3 = SmoothnessPrior(tau=np.full(3, 1e3), eta=np.full(3, 1e-2))
    fc = forecast(grid, dataset, prior3, n_modes=1, telluric=True)
    assert fc.component_labels == ("star 1", "star 2", "telluric")
    assert fc.component_std.shape == (3, grid.n)
    assert fc.region.shape == (3, grid.n)


def test_more_than_two_stars_skips_the_idealized_diagnostic(grid, dataset):
    prior2 = SmoothnessPrior(tau=np.full(2, 1e3), eta=np.full(2, 1e-2))
    fc = forecast(grid, dataset, prior2, n_modes=0)
    assert np.isfinite(fc.rms_delta_kms)
    prior3 = SmoothnessPrior(tau=np.full(3, 1e3), eta=np.full(3, 1e-2))
    vel = np.vstack([ORBIT.component_velocities(dataset.bjd), np.zeros(dataset.n_epochs)])
    fc3 = sensitivity_forecast(
        grid,
        dataset,
        light_fractions=(0.4, 0.35, 0.25),
        lsf_sigma_v=LSF,
        prior=prior3,
        orbit=None,
        velocities=vel,
        n_modes=0,
    )
    assert np.isnan(fc3.rms_delta_kms)
    assert np.isnan(fc3.blind_fraction)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"orbit": None}, "exactly one of orbit"),
        ({"velocities": np.zeros((2, 4))}, "exactly one of orbit"),
        ({"light_fractions": (0.3, 0.3, 0.4)}, "light_fractions must be"),
        ({"region_floor": 1.0}, "region_floor"),
        ({"penalty_threshold": 0.5}, "penalty_threshold"),
        ({"baseline": []}, "baseline is empty"),
        ({"baseline": [0, 0, 1]}, "repeated epoch indices"),
        ({"baseline": [0, 9]}, r"baseline indices must lie"),
        ({"baseline": [0, 1, 2, 3]}, "nothing planned"),
        ({"feature_scales_kms": [50.0, 10.0]}, "strictly ascending"),
    ],
)
def test_validation(grid, dataset, prior, kwargs, match):
    with pytest.raises(ValueError, match=match):
        forecast(grid, dataset, prior, n_modes=0, **kwargs)


def test_arguments_are_checked_before_any_work_is_done(grid, dataset, prior):
    """A survey-scale forecast is minutes; a bad epoch index must not cost them.

    Both arguments here are wrong. If the baseline check ran after the design was built,
    the LSF failure from ``build_problem`` would surface first — so the message tells us
    the ordering, not just that something was rejected.
    """
    with pytest.raises(ValueError, match="baseline indices must lie"):
        forecast(grid, dataset, prior, n_modes=0, baseline=[0, 99], lsf_sigma_v={})


def test_prior_length_is_checked(grid, dataset):
    wrong = SmoothnessPrior(tau=np.full(3, 1e3), eta=np.full(3, 1e-2))
    with pytest.raises(ValueError, match="prior has 3 components"):
        forecast(grid, dataset, wrong, n_modes=0)


def test_prior_profile_grid_is_checked(grid, dataset):
    wrong = SmoothnessPrior(
        tau=np.full(2, 1e3), eta=np.full(2, 1e-2), eta_profile=np.ones((2, grid.n + 5))
    )
    with pytest.raises(ValueError, match="prior profiles cover"):
        forecast(grid, dataset, wrong, n_modes=0)


# ---------------------------------------------------------------------------
# The idealized diagnostic on its own
# ---------------------------------------------------------------------------


def test_separation_penalty_is_the_closed_form(grid):
    """``sqrt[(J + |g|) / (J - |g|)]`` with ``g = sum_j exp(i k Delta_j)``, term by term."""
    delta = np.array([-3.0, -1.0, 0.5, 2.0, 4.5])
    scales = _default_scales(grid, 16)
    k, penalty, blind = _separation_diagnostics(delta, grid, scales, 2.0)
    g = np.abs(np.exp(1j * k[:, None] * delta[None, :]).sum(axis=1))
    assert np.allclose(penalty, np.sqrt((delta.size + g) / (delta.size - g)))
    assert blind == pytest.approx(float(np.mean(penalty > 2.0)))
    # The broad end is always blind: that is the k -> 0 singularity of §5.1.
    assert penalty[-1] > penalty[0]


def test_two_valued_delta_combs_the_penalty(grid):
    """A cadence aliased to the period leaves |g| recurring to J, so poles repeat."""
    aliased = np.repeat([-5.0, 5.0], 8)
    spread = np.linspace(-5.0, 5.0, 16)
    scales = _default_scales(grid, 96)
    _, pen_a, blind_a = _separation_diagnostics(aliased, grid, scales, 2.0)
    _, _, blind_s = _separation_diagnostics(spread, grid, scales, 2.0)
    assert np.std(aliased) > np.std(spread), "the aliased design has the larger Var(Delta)"
    assert blind_a > blind_s, "and is nevertheless blind over more of the range"
    assert pen_a.max() > 10.0  # a genuine pole, not a slow rise


def test_summary_is_readable(grid, dataset, prior):
    text = forecast(grid, dataset, prior, n_modes=2).summary()
    assert "no fluxes were used" in text
    assert "constrained spectral degrees of freedom" in text
    assert "worst-determined modes" in text
