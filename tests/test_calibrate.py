"""Calibrated faint-companion detection: the vectorized scan, K1 marginalization, limits.

Three layers, in the order they depend on each other.

1. **Exactness.** The vectorized sweep must agree with the Python loop it replaces, and a
   fixed-K1 scan must agree with the pre-marginalization one. Neither is bit-identical —
   batching trials into one ``lax.map`` re-associates the linear algebra — so the bar is
   float64 round-off on a log-likelihood of order 1e5, not equality.
2. **The bootstrap.** ``resimulate`` draws from the problem's own forward model, so its
   mean must converge to the noiseless model and its scatter to ``1/sqrt(w)``. If that is
   wrong, every calibrated threshold downstream is wrong by the same factor and nothing
   else in the suite would notice.
3. **The statistic.** A null distribution that a real detection sits far above, a
   completeness curve that rises with the injected light fraction, and a false-alarm
   probability that never claims more resolution than the trial count supports.
"""

import time

import jax.numpy as jnp
import numpy as np
import pytest

import albireo as ab
from albireo.calibrate import _interpolate_limit
from albireo.forward import apply_model
from albireo.inference import _sweep_batch_default
from albireo.scan import _k1_quadrature, _logsumexp

GRID = ab.LogGrid.from_wavelength_range(5000.0, 5040.0, dv_kms=6.0)
P, ECC, OMEGA, K1, K2_TRUE = 6.31, 0.15, 0.7, 12.0, 38.0
ELL = (0.9, 0.1)
K2_GRID = np.arange(14.0, 62.0, 6.0)  # K2_TRUE = 38.0 lands on the grid
PRIOR = ab.SmoothnessPrior(jnp.asarray([300.0, 30.0]), jnp.asarray([5.0, 5.0]))
V_REL_MAX = 105.0
LSF = {"a": 7.0}


def _orbit():
    nu_c = 0.5 * np.pi - OMEGA
    e_c = 2.0 * np.arctan2(
        np.sqrt(1.0 - ECC) * np.sin(0.5 * nu_c), np.sqrt(1.0 + ECC) * np.cos(0.5 * nu_c)
    )
    t_conj = 2.0 + (e_c - ECC * np.sin(e_c)) * P / (2.0 * np.pi)
    return {
        "period": P,
        "t_conj": t_conj,
        "secosw": np.sqrt(ECC) * np.cos(OMEGA),
        "sesinw": np.sqrt(ECC) * np.sin(OMEGA),
    }


def _simulate(*, with_companion: bool, seed: int = 5, gaps: float = 0.0):
    rng = np.random.default_rng(7)
    bjd = np.sort(rng.uniform(0.0, 21.0, size=9))
    primary = ab.synthetic_deviation_spectrum(GRID, seed=21)
    companion = ab.synthetic_deviation_spectrum(GRID, seed=22)
    inst = {
        "a": ab.InstrumentSpec(wave=np.arange(5003.0, 5037.0, 0.12), sigma_v_lsf=7.0, snr=150.0)
    }
    if with_companion:
        orbit = ab.OrbitParams(period=P, t_peri=2.0, ecc=ECC, omega=OMEGA, k=(K1, K2_TRUE))
        comps, ell = [primary, companion], ELL
    else:
        orbit = ab.OrbitParams(period=P, t_peri=2.0, ecc=ECC, omega=OMEGA, k=(K1,))
        comps, ell = [primary], (1.0,)
    ds, truth = ab.simulate_dataset(
        GRID,
        comps,
        bjd=bjd,
        instruments=inst,
        light_fractions=ell,
        orbit=orbit,
        gap_fraction=gaps,
        seed=seed,
    )
    return ds, truth, primary, companion


def _model(ds, ell=ELL, prior=PRIOR):
    return ab.MarginalOrbitModel(
        GRID,
        ds,
        light_fractions=np.asarray(ell),
        lsf_sigma_v=LSF,
        v_rel_max_kms=V_REL_MAX,
        prior=prior,
    )


@pytest.fixture(scope="module")
def sb2():
    return _simulate(with_companion=True)


@pytest.fixture(scope="module")
def sb1():
    return _simulate(with_companion=False)


# -- 1. the sweep is the loop ------------------------------------------------


def test_sweep_matches_the_python_loop_it_replaces(sb2):
    """One lax.map over the grid, against one jitted call per point."""
    ds = sb2[0]
    model = _model(ds)
    theta = {name: jnp.asarray(v) for name, v in _orbit().items()}

    loop = np.array(
        [float(model.log_likelihood({**theta, "k": jnp.asarray([K1, k2])})) for k2 in K2_GRID]
    )
    pairs = jnp.asarray(np.stack([np.full(K2_GRID.size, K1), K2_GRID], axis=1))
    swept = np.asarray(model.log_likelihood_sweep(theta, {"k": pairs}))

    assert swept.shape == loop.shape
    # float64 round-off on |ll| ~ 1e5, not equality: batching re-associates the sums.
    assert np.max(np.abs(swept - loop)) < 1e-6
    assert np.max(np.abs(swept - loop)) / max(np.max(np.abs(loop)), 1.0) < 1e-11


def test_sweep_batching_does_not_change_the_answer(sb2):
    """batch_size is a memory-vs-speed knob, not a numerical one."""
    model = _model(sb2[0])
    theta = {name: jnp.asarray(v) for name, v in _orbit().items()}
    pairs = jnp.asarray(np.stack([np.full(K2_GRID.size, K1), K2_GRID], axis=1))
    ref = np.asarray(model.log_likelihood_sweep(theta, {"k": pairs}, batch_size=1))
    for batch in (2, 3, K2_GRID.size):
        got = np.asarray(model.log_likelihood_sweep(theta, {"k": pairs}, batch_size=batch))
        assert np.max(np.abs(got - ref)) < 1e-6


def test_sweep_can_vary_more_than_one_site(sb2):
    """The trial axis is a pytree axis, so several sites move together."""
    model = _model(sb2[0])
    theta = {name: jnp.asarray(v) for name, v in _orbit().items()}
    n = 4
    pairs = jnp.asarray(np.stack([np.full(n, K1), np.linspace(30.0, 45.0, n)], axis=1))
    taus = jnp.asarray(np.stack([np.full(n, 300.0), np.linspace(20.0, 60.0, n)], axis=1))
    one_by_one = np.array(
        [
            float(
                model.log_likelihood(
                    {
                        **theta,
                        "k": pairs[i],
                        "log_tau": jnp.log(taus[i]),
                        "log_eta": jnp.log(PRIOR.eta),
                    }
                )
            )
            for i in range(n)
        ]
    )
    # log_eta has to be supplied alongside log_tau; the sweep inherits it from theta,
    # so pass it explicitly there too and compare.
    swept = np.asarray(
        model.log_likelihood_sweep(
            {**theta, "log_eta": jnp.log(PRIOR.eta)},
            {"k": pairs, "log_tau": jnp.log(taus)},
        )
    )
    assert np.max(np.abs(swept - one_by_one)) < 1e-6


def test_sweep_rejects_malformed_input(sb2):
    model = _model(sb2[0])
    theta = {name: jnp.asarray(v) for name, v in _orbit().items()}
    with pytest.raises(ValueError, match="sweep is empty"):
        model.log_likelihood_sweep(theta, {})
    with pytest.raises(ValueError, match="unknown sites in sweep"):
        model.log_likelihood_sweep(theta, {"kk": jnp.zeros((3, 2))})
    with pytest.raises(ValueError, match="leading trial axis"):
        model.log_likelihood_sweep(theta, {"k": jnp.asarray(3.0)})
    with pytest.raises(ValueError, match="disagree on the trial-axis length"):
        model.log_likelihood_sweep(theta, {"k": jnp.zeros((3, 2)), "log_tau": jnp.zeros((4, 2))})
    with pytest.raises(ValueError, match="no trials"):
        model.log_likelihood_sweep(theta, {"k": jnp.zeros((0, 2))})


def test_sweep_batch_policy_is_two_regime():
    """Whole sweep in one batch while it is cheap; capped, never zero, when it is not."""
    assert _sweep_batch_default(16, 4_000, 40) == 16  # small: hoisted
    big = _sweep_batch_default(10_000, 200_000, 400)
    assert 1 <= big < 10_000
    assert _sweep_batch_default(10_000, 10**7, 10**4) == 1  # never zero


# -- 2. the parametric bootstrap ---------------------------------------------


def test_with_data_swaps_only_the_data_term(sb2):
    ds = sb2[0]
    model = _model(ds)
    problem = model.problem
    z_new = [np.full(g.z.shape, 0.25) for g in problem.groups]
    swapped = ab.with_data(problem, z_new)

    for g_old, g_new in zip(problem.groups, swapped.groups, strict=True):
        w = np.asarray(g_old.w)
        assert np.allclose(np.asarray(g_new.z)[w > 0], 0.25)
        assert np.all(np.asarray(g_new.z)[w == 0] == 0.0)  # masked pixels forced to zero
        for field in ("w", "r", "base", "kernel", "shifts", "light", "bary_pix"):
            assert np.array_equal(
                np.asarray(getattr(g_old, field)), np.asarray(getattr(g_new, field))
            )

    with pytest.raises(ValueError, match="expected 1 z arrays"):
        ab.with_data(problem, z_new * 2)
    with pytest.raises(ValueError, match="must have shape"):
        ab.with_data(problem, [np.zeros((3, 3))])


def test_with_data_zeroes_nan_under_the_mask():
    """A masked pixel may hold anything; 0 * nan is nan, so it must not survive."""
    ds = _simulate(with_companion=False, gaps=0.05)[0]
    model = _model(
        ds, ell=(1.0,), prior=ab.SmoothnessPrior(jnp.asarray([300.0]), jnp.asarray([5.0]))
    )
    z_new = [np.full(g.z.shape, np.nan) for g in model.problem.groups]
    swapped = ab.with_data(model.problem, z_new)
    good = np.asarray(model.problem.groups[0].w) > 0
    assert np.all(np.isfinite(np.asarray(swapped.groups[0].z)[~good]))


def test_resimulate_draws_from_the_problems_own_forward_model(sb1):
    """Mean over draws -> the noiseless model; scatter -> 1/sqrt(w)."""
    ds, _, primary, _ = sb1
    prior1 = ab.SmoothnessPrior(jnp.asarray([300.0]), jnp.asarray([5.0]))
    model = _model(ds, ell=(1.0,), prior=prior1)
    theta = {**{n: jnp.asarray(v) for n, v in _orbit().items()}, "k": jnp.asarray([K1])}
    at_truth = model.problem_at(theta)
    d_true = np.asarray(primary)[None, :]

    truth_z = np.asarray(apply_model(at_truth, d_true)[0]) * np.asarray(at_truth.groups[0].r)
    n_draw = 300
    draws = np.stack(
        [np.asarray(ab.resimulate(at_truth, d_true, seed=s).groups[0].z) for s in range(n_draw)]
    )
    w = np.asarray(at_truth.groups[0].w)
    good = w > 0
    sigma = 1.0 / np.sqrt(w[good])

    mean_err = (draws.mean(axis=0)[good] - truth_z[good]) / (sigma / np.sqrt(n_draw))
    assert np.abs(mean_err).max() < 5.0  # every pixel's mean within 5 standard errors
    ratio = draws.std(axis=0, ddof=1)[good] / sigma
    assert 0.9 < ratio.mean() < 1.1
    assert np.all(draws[:, ~good] == 0.0)


def test_resimulate_is_reproducible_and_checked(sb1):
    ds, _, primary, _ = sb1
    prior1 = ab.SmoothnessPrior(jnp.asarray([300.0]), jnp.asarray([5.0]))
    model = _model(ds, ell=(1.0,), prior=prior1)
    d_true = np.asarray(primary)[None, :]
    a = ab.resimulate(model.problem, d_true, seed=11).groups[0].z
    b = ab.resimulate(model.problem, d_true, seed=11).groups[0].z
    c = ab.resimulate(model.problem, d_true, seed=12).groups[0].z
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))
    with pytest.raises(ValueError, match="d_stack must have shape"):
        ab.resimulate(model.problem, np.zeros((2, GRID.n)), seed=0)


def test_resimulated_data_recovers_the_injected_companion(sb1):
    """The loop closes through the bootstrap: inject, redraw, scan, find it."""
    ds, _, primary, companion = sb1
    model = _model(ds)
    orbit = {n: jnp.asarray(v) for n, v in _orbit().items()}
    at_truth = model.problem_at({**orbit, "k": jnp.asarray([K1, K2_TRUE])})
    drawn = ab.resimulate(at_truth, np.stack([primary, companion]), seed=99)

    swapped = ab.with_data(model.problem, [g.z for g in drawn.groups])
    pairs = jnp.asarray(np.stack([np.full(K2_GRID.size, K1), K2_GRID], axis=1))
    ll = np.asarray(model.log_likelihood_sweep(orbit, {"k": pairs}, problem=swapped))
    assert K2_GRID[int(np.argmax(ll))] == K2_TRUE


# -- 3. the K1 quadrature ----------------------------------------------------


def test_k1_quadrature_is_a_normalized_gaussian_rule():
    nodes, log_w = _k1_quadrature(50.0, None, 7)
    assert nodes.shape == (1,) and nodes[0] == 50.0
    assert log_w[0] == 0.0  # a delta: every log-sum-exp downstream is the identity

    nodes, log_w = _k1_quadrature(50.0, 3.0, 9)
    w = np.exp(log_w)
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(np.sum(w * nodes), 50.0)  # mean
    assert np.isclose(np.sum(w * (nodes - 50.0) ** 2), 9.0)  # variance
    assert np.isclose(np.sum(w * (nodes - 50.0) ** 4), 3.0 * 81.0)  # kurtosis

    with pytest.raises(ValueError, match="non-positive semi-amplitude"):
        _k1_quadrature(5.0, 4.0, 7)
    with pytest.raises(ValueError, match="k1_nodes must be at least 2"):
        _k1_quadrature(50.0, 1.0, 1)
    with pytest.raises(ValueError, match="k1_sigma must be non-negative"):
        _k1_quadrature(50.0, -1.0, 7)


def test_logsumexp_matches_the_naive_form():
    a = np.array([[1.0, -2.0, 3.0], [0.5, 0.25, -1.0]])
    assert np.allclose(_logsumexp(a, axis=0), np.log(np.exp(a).sum(axis=0)))
    assert np.allclose(_logsumexp(a, axis=1), np.log(np.exp(a).sum(axis=1)))
    big = a + 900.0  # would overflow without the shift
    assert np.allclose(_logsumexp(big, axis=0), _logsumexp(a, axis=0) + 900.0)


def _scan(ds, **kw):
    return ab.k2_scan(
        GRID,
        ds,
        orbit=_orbit(),
        k1=kw.pop("k1", K1),
        k2_grid=K2_GRID,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        prior=PRIOR,
        v_rel_max_kms=V_REL_MAX,
        **kw,
    )


@pytest.fixture(scope="module")
def fixed_scan(sb2):
    return _scan(sb2[0])


def test_fixed_k1_scan_is_the_one_node_rule(sb2, fixed_scan):
    """Marginalizing over a delta must reproduce holding K1 fixed."""
    assert fixed_scan.log_likelihood_grid.shape == (1, K2_GRID.size)
    assert fixed_scan.log_likelihood_null_grid.shape == (1,)
    assert fixed_scan.k1_peak == K1
    assert not np.isnan(fixed_scan.k1_peak)
    assert np.allclose(fixed_scan.log_likelihood, fixed_scan.log_likelihood_grid[0])

    narrow = _scan(sb2[0], k1_sigma=1e-9, k1_nodes=5)
    assert narrow.log_likelihood_grid.shape == (5, K2_GRID.size)
    assert np.max(np.abs(narrow.detection - fixed_scan.detection)) < 1e-6
    assert narrow.k2_peak == fixed_scan.k2_peak


def test_marginal_scan_finds_the_companion(sb2):
    result = _scan(sb2[0], k1_sigma=2.0, k1_nodes=7)
    assert result.k2_peak == K2_TRUE
    assert result.detection_peak > 0
    assert np.all(np.isfinite(result.detection))
    assert result.k1_grid.size == 7
    assert np.isclose(float(np.exp(_logsumexp(result.k1_log_weights, axis=0))), 1.0)
    assert result.k1_peak in result.k1_grid
    # The marginal is bounded by the grid it averages: a weighted mean of the rows sits
    # between the smallest and the largest of them.
    assert np.all(result.log_likelihood <= result.log_likelihood_grid.max(axis=0) + 1e-8)
    assert np.all(result.log_likelihood >= result.log_likelihood_grid.min(axis=0) - 1e-8)


def test_marginalizing_k1_beats_assuming_a_wrong_one(sb2):
    """The failure mode the literature reports, and the fix.

    With K1 mis-set, the companion's recovered spectrum picks up structure that is not
    the companion. Integrating K1 out lets the scan profile over it instead, and the
    recovered line pattern goes back to matching the injected one. The comparison is on
    the *offset-removed* correlation: the smooth envelope is prior-dominated at
    ell_2 = 0.1 (math.md 5.1-5.2), so the line pattern is what carries the information.
    """
    ds, _, _, companion = sb2
    truth = np.asarray(companion) - np.mean(companion)

    def pattern_match(res):
        got = res.companion - np.mean(res.companion)
        return float(np.dot(got, truth) / np.sqrt(np.dot(got, got) * np.dot(truth, truth)))

    wrong_k1 = K1 * 1.35
    fixed_wrong = pattern_match(_scan(ds, k1=wrong_k1))
    marginalized = pattern_match(_scan(ds, k1=wrong_k1, k1_sigma=0.2 * K1, k1_nodes=9))
    right = pattern_match(_scan(ds, k1=K1))

    assert right > 0.9, f"sanity: the correct K1 should recover the companion ({right:.3f})"
    assert marginalized > fixed_wrong, (
        f"marginalizing K1 should beat assuming a wrong one: "
        f"{marginalized:.3f} vs {fixed_wrong:.3f}"
    )


def test_scan_result_round_trips_with_the_k1_grid(sb2, tmp_path):
    result = _scan(sb2[0], k1_sigma=2.0, k1_nodes=5)
    path = ab.save_fit(result, tmp_path / "scan.npz")
    back = ab.load_fit(path)
    assert back.model is None
    assert np.allclose(back.log_likelihood_grid, result.log_likelihood_grid)
    assert np.allclose(back.k1_grid, result.k1_grid)
    assert np.allclose(back.k1_log_weights, result.k1_log_weights)
    assert back.k1_peak == result.k1_peak
    assert back.detection_peak == pytest.approx(result.detection_peak)


# -- 4. the calibrated limit -------------------------------------------------


@pytest.fixture(scope="module")
def limit(sb1):
    """One calibration, reused: it is the most expensive fixture in the file."""
    return ab.detection_limit(
        GRID,
        sb1[0],
        orbit=_orbit(),
        k1=K1,
        k2_true=K2_TRUE,
        k2_grid=K2_GRID,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        prior=PRIOR,
        v_rel_max_kms=V_REL_MAX,
        ell2_grid=np.array([0.005, 0.01, 0.02, 0.04]),
        n_null=24,
        n_trials=12,
        false_alarm=0.05,
        seed=3,
    )


def test_detection_limit_shapes_and_bookkeeping(limit):
    assert limit.null_peaks.shape == (24,)
    assert limit.signal_peaks.shape == (4, 12)
    assert limit.n_null == 24
    assert limit.fap_floor == pytest.approx(1.0 / 25.0)
    assert np.all(np.isfinite(limit.null_peaks))
    assert np.all(np.isfinite(limit.signal_peaks))
    assert limit.k2_true == K2_TRUE
    assert limit.k1_marginalized is False


def test_the_threshold_never_exceeds_the_requested_false_alarm_rate(limit):
    """Conservative by construction, in both of the two ways that must agree.

    An interpolating sample quantile satisfies neither: it lands between order
    statistics and can leave more of the null distribution above the threshold than the
    budget allows, which would sell an 8% false-alarm rate as 5%.
    """
    assert np.mean(limit.null_peaks > limit.threshold) <= limit.false_alarm + 1e-12
    # Anything the calibration calls a detection reports a FAP within budget.
    eps = 1e-9 * max(abs(limit.threshold), 1.0)
    assert limit.false_alarm_probability(limit.threshold + eps) <= limit.false_alarm + 1e-12
    assert limit.threshold <= limit.null_peaks.max()


def test_threshold_degrades_to_beating_every_trial_below_the_resolution_floor():
    """Asking for a FAP finer than the trials resolve must not invent precision."""
    from albireo.calibrate import _threshold_at

    peaks = np.arange(20.0)  # 20 trials: floor = 1/21 = 0.048
    assert _threshold_at(peaks, 0.01) == peaks.max()
    assert _threshold_at(peaks, 0.048) == peaks.max()
    # With enough trials it steps down through the order statistics, conservatively.
    peaks = np.arange(100.0)
    for fa in (0.05, 0.1, 0.2):
        thr = _threshold_at(peaks, fa)
        assert np.mean(peaks > thr) <= fa + 1e-12
        assert (1 + np.count_nonzero(peaks >= thr + 1e-9)) / 101 <= fa + 1e-12


def test_detection_grows_with_the_injected_light_fraction(limit):
    """More companion, larger statistic — the completeness curve's whole premise."""
    medians = np.median(limit.signal_peaks, axis=1)
    assert np.all(np.diff(medians) > 0)
    assert np.all(np.diff(limit.completeness) >= 0)
    assert limit.completeness[-1] == 1.0
    assert limit.completeness[0] < 1.0, "the faintest rung should not be trivially found"


def test_the_quoted_limit_is_bracketed_by_the_ladder(limit):
    assert np.isfinite(limit.ell2_limit)
    assert limit.limit_is_bracketed, "this fixture's ladder should straddle the crossing"
    assert limit.ell2_grid[0] <= limit.ell2_limit <= limit.ell2_grid[-1]
    # The limit is where completeness crosses `confidence`: rungs below it must be
    # incomplete, rungs above it complete.
    for e, c in zip(limit.ell2_grid, limit.completeness, strict=True):
        if e < limit.ell2_limit:
            assert c < limit.confidence
        if e > limit.ell2_limit:
            assert c >= limit.confidence


def test_false_alarm_probability_never_claims_zero(limit):
    huge = float(limit.null_peaks.max()) + 1e6
    assert limit.false_alarm_probability(huge) == pytest.approx(limit.fap_floor)
    assert limit.false_alarm_probability(float(limit.null_peaks.min()) - 1.0) == 1.0
    mid = float(np.median(limit.null_peaks))
    assert 0.4 < limit.false_alarm_probability(mid) < 0.6


def test_summary_states_the_limit_and_its_caveats(limit):
    text = limit.summary()
    assert f"{100.0 * limit.ell2_limit:.2f}%" in text
    assert "95% confidence" in text
    assert "false-alarm" in text
    assert "resolution floor" in text
    assert "held fixed" in text


def test_a_real_companion_lands_far_above_the_null(limit, sb2):
    """The calibration's point: the observed peak has to be read against the null."""
    observed = _scan(sb2[0]).detection_peak
    assert observed > limit.null_peaks.max()
    assert limit.false_alarm_probability(observed) == pytest.approx(limit.fap_floor)


def test_null_trials_carry_the_occam_penalty(limit):
    """D < 0 with no companion: the marginal likelihood charges for the free spectrum.

    Documented in tests/test_scan.py and worth pinning here too, because it is why the
    threshold is calibrated rather than set at zero — 'D > 0' would be a *conservative*
    test here, and on another dataset it might not be.
    """
    assert np.all(limit.null_peaks < 0.0)
    assert limit.threshold < 0.0


def test_detection_limit_rejects_malformed_input(sb1):
    kwargs = dict(
        orbit=_orbit(),
        k1=K1,
        k2_true=K2_TRUE,
        k2_grid=K2_GRID,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        prior=PRIOR,
        v_rel_max_kms=V_REL_MAX,
        ell2_grid=np.array([0.01, 0.02]),
        n_null=1,
        n_trials=1,
    )
    ds = sb1[0]
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        ab.detection_limit(GRID, ds, **{**kwargs, "ell2_grid": np.array([0.0, 0.2])})
    with pytest.raises(ValueError, match="strictly ascending"):
        ab.detection_limit(GRID, ds, **{**kwargs, "ell2_grid": np.array([0.2, 0.1])})
    with pytest.raises(ValueError, match="false_alarm must lie"):
        ab.detection_limit(GRID, ds, **{**kwargs, "false_alarm": 1.5})
    with pytest.raises(ValueError, match="confidence must lie"):
        ab.detection_limit(GRID, ds, **{**kwargs, "confidence": 0.0})
    with pytest.raises(ValueError, match="at least 1"):
        ab.detection_limit(GRID, ds, **{**kwargs, "n_null": 0})
    with pytest.raises(ValueError, match="two components"):
        ab.detection_limit(GRID, ds, **{**kwargs, "light_fractions": (1.0,)})
    with pytest.raises(ValueError, match="prior must have 2 components"):
        ab.detection_limit(
            GRID,
            ds,
            **{**kwargs, "prior": ab.SmoothnessPrior(jnp.asarray([300.0]), jnp.asarray([5.0]))},
        )
    with pytest.raises(ValueError, match="primary_template must have shape"):
        ab.detection_limit(GRID, ds, **{**kwargs, "primary_template": np.zeros(3)})
    with pytest.raises(ValueError, match="companion_template must have shape"):
        ab.detection_limit(GRID, ds, **{**kwargs, "companion_template": np.zeros(3)})
    # A calibration and the scan it calibrates take the same arguments, so they must
    # reject the same mistakes in the same words (scan._check_search).
    with pytest.raises(ValueError, match="orbit is missing sites"):
        ab.detection_limit(
            GRID, ds, **{**kwargs, "orbit": {k: v for k, v in _orbit().items() if k != "period"}}
        )
    with pytest.raises(ValueError, match="k1 is a separate"):
        ab.detection_limit(GRID, ds, **{**kwargs, "orbit": {**_orbit(), "k": 12.0}})
    with pytest.raises(ValueError, match="k2_grid must be positive"):
        ab.detection_limit(GRID, ds, **{**kwargs, "k2_grid": np.array([-5.0, 20.0])})


def test_progress_callback_counts_every_trial(sb1):
    seen = []
    ab.detection_limit(
        GRID,
        sb1[0],
        orbit=_orbit(),
        k1=K1,
        k2_true=K2_TRUE,
        k2_grid=K2_GRID[:3],
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        prior=PRIOR,
        v_rel_max_kms=V_REL_MAX,
        ell2_grid=np.array([0.02]),
        n_null=2,
        n_trials=2,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_interpolated_limit_takes_the_first_upcrossing():
    ell2 = np.array([0.01, 0.02, 0.03, 0.04])
    # Reaches 0.95 between rungs 1 and 2, then dips: noise must not push the limit out.
    got, bracketed = _interpolate_limit(ell2, np.array([0.5, 0.9, 1.0, 0.93]), 0.95)
    assert 0.02 < got < 0.03
    assert bracketed

    missed, bracketed = _interpolate_limit(ell2, np.array([0.1, 0.2, 0.3, 0.4]), 0.95)
    assert np.isnan(missed)
    assert not bracketed


def test_a_ladder_that_never_straddles_the_crossing_is_flagged():
    """The faintest rung already complete bounds the limit; it does not measure it."""
    ell2 = np.array([0.01, 0.02, 0.03])
    got, bracketed = _interpolate_limit(ell2, np.array([1.0, 1.0, 1.0]), 0.95)
    assert got == 0.01
    assert not bracketed

    saturated = ab.calibrate.DetectionLimit(
        ell2_grid=ell2,
        null_peaks=np.array([-1.0, -2.0]),
        signal_peaks=np.ones((3, 2)),
        threshold=-1.0,
        false_alarm=0.05,
        fap_floor=1 / 3,
        completeness=np.array([1.0, 1.0, 1.0]),
        confidence=0.95,
        ell2_limit=got,
        k2_true=40.0,
        k2_grid=np.array([40.0]),
        k1_marginalized=False,
        limit_is_bracketed=bracketed,
    )
    text = saturated.summary()
    assert "at most 1.00%" in text
    assert "Extend ell2_grid downward" in text


# -- 5. the speedup the vectorization exists for -----------------------------


@pytest.mark.slow
def test_the_sweep_is_much_faster_than_the_loop(sb2):
    """The scan's cost is why K1 marginalization and calibration are affordable at all.

    Timing assertions are usually a bad idea; this one is worth its flakiness risk
    because the whole point of the change is the constant factor, and a regression that
    silently reinstates the per-point device synchronization would otherwise show up
    only as a calibration that takes an hour.
    """
    model = _model(sb2[0])
    theta = {name: jnp.asarray(v) for name, v in _orbit().items()}
    grid = np.linspace(14.0, 60.0, 32)
    pairs = jnp.asarray(np.stack([np.full(grid.size, K1), grid], axis=1))

    model.log_likelihood({**theta, "k": jnp.asarray([K1, grid[0]])})  # warm both
    np.asarray(model.log_likelihood_sweep(theta, {"k": pairs}))

    def best_of(fn, repeats=3):
        """Minimum, not mean: this machine is shared, and interference only ever adds."""
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    loop = best_of(
        lambda: [float(model.log_likelihood({**theta, "k": jnp.asarray([K1, k2])})) for k2 in grid]
    )
    swept = best_of(lambda: np.asarray(model.log_likelihood_sweep(theta, {"k": pairs})))

    # Measured 2.0-2.8x across 201 to 2,652 model pixels (docs/benchmarks.md D41). The
    # bar sits below that range on purpose: the point is to catch a reinstated per-point
    # device synchronization, which would put the ratio at ~1, not to police the factor.
    assert swept < loop / 1.5, f"sweep {swept * 1e3:.0f} ms vs loop {loop * 1e3:.0f} ms"
