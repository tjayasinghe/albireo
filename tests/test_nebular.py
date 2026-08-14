"""The nebular component and the per-pixel prior profiles it needs (D40).

Four groups of assertions:

1. *Structure* — the component is appended last, its shift law is the mirror of the
   telluric one (static in the **barycentric** frame), and the θ-path swaps reproduce a
   fresh :func:`albireo.forward.build_problem` exactly.
2. *Exactness* — the forward model reproduces the simulator's injection to float
   precision, and the D28 band assembly still equals the matrix-free operator with a
   nebular column present.
3. *The per-pixel prior* — ``tau_profile`` / ``eta_profile`` against a dense
   construction, including the determinant recursion the marginal likelihood uses.
4. *The point of the exercise* — an unmodelled nebular line leaks into the stellar
   components as a spurious core-fill, and the component removes it. That test is the
   reason this file exists; everything above it is the scaffolding that makes it
   trustworthy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import albireo as ab
from albireo.assembly import prior_logdet
from albireo.forward import (
    apply_model,
    build_problem,
    with_light_fractions,
    with_nebular_amplitudes,
    with_velocities,
)
from albireo.inference import MarginalOrbitModel, nebular_amplitudes, run_map
from albireo.likelihood import marginal_loglikelihood
from albireo.priors import (
    NEBULAR_LINES,
    SmoothnessPrior,
    nebular_windows,
    window_profile,
)
from albireo.simulate import (
    InstrumentSpec,
    OrbitParams,
    simulate_dataset,
    synthetic_deviation_spectrum,
    synthetic_nebular_spectrum,
    synthetic_telluric_spectrum,
)

# H-beta, because it is where the problem actually bites: a nebular emission line sitting
# in the core of the broad Balmer absorption every massive star has.
HBETA = 4861.33
GRID = ab.LogGrid.from_wavelength_range(4838.0, 4886.0, dv_kms=5.5)
BJD = np.array([0.4, 1.3, 2.1, 3.0, 3.8, 4.6, 5.5, 6.2, 7.1, 8.0, 8.8, 9.6])
N_EP = BJD.size
ELL = np.array([0.7, 0.3])
K_TRUE = (58.0, 41.0)
P_TRUE, TPERI_TRUE = 5.7, 0.0
LSF_KMS = 7.0
# (K_1 + K_2) plus the nebula at rest plus barycentric motion, with headroom.
V_REL_MAX = 170.0


def _instrument(snr: float = 220.0) -> InstrumentSpec:
    return InstrumentSpec(wave=np.arange(4841.0, 4883.0, 0.10), sigma_v_lsf=LSF_KMS, snr=snr)


def _gaussian_line(center_angstrom: float, depth: float, sigma_kms: float) -> np.ndarray:
    """A single Gaussian feature on the model grid (negative depth = absorption)."""
    px = np.arange(GRID.n, dtype=np.float64)
    center = float(np.interp(center_angstrom, GRID.wave, px))
    return depth * np.exp(-0.5 * ((px - center) / (sigma_kms / GRID.dv_kms)) ** 2)


def _stellar_components() -> list[np.ndarray]:
    """Two stars, each with a broad H-beta absorption plus some metal lines.

    Broad and deep on purpose: the whole question is whether a narrow emission line in
    the core of this profile ends up in the star or in the nebula.
    """
    primary = _gaussian_line(HBETA, -0.55, 95.0) + synthetic_deviation_spectrum(
        GRID, n_lines=6, depth_range=(0.03, 0.12), sigma_v_range=(12.0, 25.0), seed=11
    )
    secondary = _gaussian_line(HBETA, -0.40, 70.0) + synthetic_deviation_spectrum(
        GRID, n_lines=5, depth_range=(0.03, 0.10), sigma_v_range=(12.0, 25.0), seed=12
    )
    return [np.maximum(primary, -0.95), np.maximum(secondary, -0.95)]


def _nebular_spectrum() -> np.ndarray:
    return synthetic_nebular_spectrum(
        GRID, lines=[HBETA], amplitude_range=(0.45, 0.45), sigma_v_kms=12.0, seed=3
    )


def _amplitudes() -> np.ndarray:
    """Seeing/slit-loss variation: a factor of ~2 between the best and worst night."""
    rng = np.random.default_rng(7)
    return np.exp(rng.normal(0.0, 0.28, N_EP))


def _orbit() -> OrbitParams:
    return OrbitParams(period=P_TRUE, t_peri=TPERI_TRUE, ecc=0.0, omega=0.0, k=K_TRUE)


def _simulate(*, frame: str = "barycentric", snr: float = 220.0, **kwargs):
    defaults = dict(
        bjd=BJD,
        instruments={"inst": _instrument(snr)},
        light_fractions=ELL,
        orbit=_orbit(),
        v_bary=np.linspace(-24.0, 26.0, N_EP),
        frame=frame,
        nebular=_nebular_spectrum(),
        nebular_amplitudes=_amplitudes(),
        seed=5,
    )
    defaults.update(kwargs)
    return simulate_dataset(GRID, _stellar_components(), **defaults)


NEB_WINDOW = nebular_windows(lines=[HBETA], halfwidth_kms=500.0)


def _prior(n_comp: int, *, confine: bool = False) -> SmoothnessPrior:
    """Stellar entries, plus a softer window-confined one when the nebular component is on.

    The *stellar* entries are identical either way, so every with/without comparison
    below differs in the model and in nothing else.
    """
    tau = np.full(n_comp, 200.0)
    eta = np.full(n_comp, 2.0)
    eta_profile = None
    if n_comp > 2:  # the trailing entry is the nebular one
        tau[-1] = 8.0  # its line is narrow; do not smooth it away
    if confine:
        eta_profile = np.ones((n_comp, GRID.n))
        eta_profile[-1] = window_profile(GRID.wave, NEB_WINDOW)
    return SmoothnessPrior(tau, eta, None, eta_profile)


# ---------------------------------------------------------------------------
# 1. Structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame", ["barycentric", "topocentric"])
def test_component_is_appended_last_with_the_barycentric_velocity_law(frame):
    """Order is stellar, telluric, nebular — and the nebular law mirrors the telluric one."""
    ds, _ = _simulate(frame=frame, telluric=synthetic_telluric_spectrum(GRID, seed=9))
    problem = build_problem(
        GRID,
        ds,
        velocities=_orbit().component_velocities(BJD),
        light_fractions=ELL,
        lsf_sigma_v={"inst": LSF_KMS},
        telluric=True,
        nebular=True,
        nebular_amplitudes=_amplitudes(),
    )
    assert problem.n_components == 4
    assert problem.n_stellar == 2
    (group,) = problem.groups
    shifts = np.asarray(group.shifts)
    bary = np.asarray(group.bary_pix)
    if frame == "barycentric":
        np.testing.assert_allclose(shifts[:, 2], bary)  # telluric moves
        np.testing.assert_allclose(shifts[:, 3], 0.0, atol=0)  # nebular is at rest
    else:
        np.testing.assert_allclose(shifts[:, 2], 0.0, atol=0)  # telluric is at rest
        np.testing.assert_allclose(shifts[:, 3], -bary)  # nebular moves
    np.testing.assert_allclose(np.asarray(group.light)[:, 2], 1.0)  # telluric light = 1
    np.testing.assert_allclose(np.asarray(group.light)[:, 3], _amplitudes())


def test_nebular_velocity_shifts_the_component_on_the_model_grid():
    """``nebular_v_kms`` moves where the component's lines land, in either frame."""
    ds, _ = _simulate()
    common = dict(
        velocities=_orbit().component_velocities(BJD),
        light_fractions=ELL,
        lsf_sigma_v={"inst": LSF_KMS},
        nebular=True,
    )
    at_rest = build_problem(GRID, ds, **common, nebular_v_kms=0.0)
    moved = build_problem(GRID, ds, **common, nebular_v_kms=150.0)
    delta = np.asarray(moved.groups[0].shifts)[:, 2] - np.asarray(at_rest.groups[0].shifts)[:, 2]
    np.testing.assert_allclose(delta, float(GRID.velocity_to_pixels(150.0)))


@pytest.mark.parametrize("frame", ["barycentric", "topocentric"])
def test_theta_swaps_reproduce_a_fresh_build(frame):
    """``with_velocities`` / ``with_light_fractions`` leave the nebular column alone."""
    ds, _ = _simulate(frame=frame)
    amps = _amplitudes()
    vel = _orbit().component_velocities(BJD)
    reference = build_problem(
        GRID,
        ds,
        velocities=vel,
        light_fractions=ELL,
        lsf_sigma_v={"inst": LSF_KMS},
        nebular=True,
        nebular_amplitudes=amps,
    )
    swapped = with_light_fractions(with_velocities(reference, vel), ELL)
    for g_ref, g_new in zip(reference.groups, swapped.groups, strict=True):
        np.testing.assert_allclose(np.asarray(g_new.shifts), np.asarray(g_ref.shifts), rtol=1e-14)
        np.testing.assert_allclose(np.asarray(g_new.light), np.asarray(g_ref.light), rtol=1e-14)


def test_with_nebular_amplitudes_equals_a_fresh_build():
    ds, _ = _simulate()
    amps = _amplitudes()
    common = dict(
        velocities=_orbit().component_velocities(BJD),
        light_fractions=ELL,
        lsf_sigma_v={"inst": LSF_KMS},
        nebular=True,
    )
    reference = build_problem(GRID, ds, **common, nebular_amplitudes=amps)
    swapped = with_nebular_amplitudes(build_problem(GRID, ds, **common), amps)
    for g_ref, g_new in zip(reference.groups, swapped.groups, strict=True):
        np.testing.assert_array_equal(np.asarray(g_new.light), np.asarray(g_ref.light))
        np.testing.assert_array_equal(np.asarray(g_new.shifts), np.asarray(g_ref.shifts))


def test_swaps_reject_bad_input():
    ds, _ = _simulate()
    common = dict(
        velocities=_orbit().component_velocities(BJD),
        light_fractions=ELL,
        lsf_sigma_v={"inst": LSF_KMS},
    )
    plain = build_problem(GRID, ds, **common)
    with pytest.raises(ValueError, match="no nebular component"):
        with_nebular_amplitudes(plain, np.ones(N_EP))
    nebular = build_problem(GRID, ds, **common, nebular=True)
    with pytest.raises(ValueError, match="shape"):
        with_nebular_amplitudes(nebular, np.ones(N_EP + 1))
    with pytest.raises(ValueError, match="positive"):
        build_problem(GRID, ds, **common, nebular=True, nebular_amplitudes=-np.ones(N_EP))
    with pytest.raises(ValueError, match="shape"):
        build_problem(GRID, ds, **common, nebular=True, nebular_amplitudes=np.ones(N_EP + 1))


def test_simulator_rejects_amplitudes_without_a_spectrum():
    with pytest.raises(ValueError, match="without a nebular spectrum"):
        _simulate(nebular=None, nebular_amplitudes=np.ones(N_EP))


# ---------------------------------------------------------------------------
# 2. Exactness against the simulator and across assembly paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame", ["barycentric", "topocentric"])
def test_forward_model_reproduces_the_injection(frame):
    """The model's noiseless prediction equals the simulator's, to float precision."""
    ds, truth = _simulate(frame=frame, snr=1e9)
    problem = build_problem(
        GRID,
        ds,
        velocities=truth.velocities,
        light_fractions=ELL,
        lsf_sigma_v={"inst": LSF_KMS},
        nebular=True,
        nebular_amplitudes=truth.nebular_amplitudes,
    )
    d_stack = jnp.stack([jnp.asarray(c) for c in truth.components] + [jnp.asarray(truth.nebular)])
    (model_dev,) = apply_model(problem, d_stack)
    (group,) = problem.groups
    predicted = np.asarray(group.base)[None, :] + np.asarray(model_dev)
    injected = np.stack([np.asarray(f) for f in truth.noiseless_flux])
    np.testing.assert_allclose(predicted, injected, atol=1e-12, rtol=0)


def test_band_assembly_matches_the_matrix_free_operator():
    """``validate=True`` and band == probe, with the nebular column and a window profile."""
    ds, truth = _simulate()
    problem = build_problem(
        GRID,
        ds,
        velocities=truth.velocities,
        light_fractions=ELL,
        lsf_sigma_v={"inst": LSF_KMS},
        nebular=True,
        nebular_amplitudes=truth.nebular_amplitudes,
    )
    prior = _prior(3, confine=True)
    hb = problem.half_bandwidth_bound(V_REL_MAX)
    band = marginal_loglikelihood(problem, prior, half_bandwidth=hb, validate=True)
    probe = marginal_loglikelihood(problem, prior, half_bandwidth=hb, assembly="probe")
    np.testing.assert_allclose(float(band.log_likelihood), float(probe.log_likelihood), rtol=1e-11)
    np.testing.assert_allclose(
        np.asarray(band.d_hat), np.asarray(probe.d_hat), atol=1e-9, rtol=1e-7
    )


def test_likelihood_rejects_a_prior_of_the_wrong_length_or_grid():
    ds, truth = _simulate()
    problem = build_problem(
        GRID,
        ds,
        velocities=truth.velocities,
        light_fractions=ELL,
        lsf_sigma_v={"inst": LSF_KMS},
        nebular=True,
    )
    hb = problem.half_bandwidth_bound(V_REL_MAX)
    with pytest.raises(ValueError, match="nebular"):
        marginal_loglikelihood(problem, _prior(2), half_bandwidth=hb)
    wrong_grid = SmoothnessPrior(np.full(3, 100.0), np.full(3, 1.0), None, np.ones((3, GRID.n + 5)))
    with pytest.raises(ValueError, match="pixels"):
        marginal_loglikelihood(problem, wrong_grid, half_bandwidth=hb)


# ---------------------------------------------------------------------------
# 3. The per-pixel prior
# ---------------------------------------------------------------------------


def _profiled_prior(n_pix: int, seed: int = 0) -> SmoothnessPrior:
    rng = np.random.default_rng(seed)
    return SmoothnessPrior(
        np.array([3.0, 0.7]),
        np.array([0.5, 2.0]),
        rng.uniform(0.2, 40.0, (2, n_pix)),
        rng.uniform(0.1, 1e4, (2, n_pix)),
    )


def test_profiled_prior_apply_matches_dense():
    n = 40
    prior = _profiled_prior(n)
    d = np.random.default_rng(1).normal(size=(2, n))
    got = np.asarray(prior.apply(jnp.asarray(d)))
    want = (prior.dense(n) @ d.ravel()).reshape(2, n)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_profiled_prior_is_symmetric_positive_definite():
    n = 40
    dense = _profiled_prior(n).dense(n)
    np.testing.assert_allclose(dense, dense.T, rtol=1e-13, atol=1e-13)
    assert np.min(np.linalg.eigvalsh(dense)) > 0


def test_profiled_prior_logdet_matches_dense():
    n = 40
    prior = _profiled_prior(n)
    sign, want = np.linalg.slogdet(prior.dense(n))
    assert sign > 0
    np.testing.assert_allclose(float(prior_logdet(prior, n)), want, rtol=1e-10)


def test_uniform_profile_is_the_unprofiled_prior():
    """A profile of ones must be bit-for-bit the v1 prior, not merely close."""
    n = 30
    tau, eta = np.array([2.0, 5.0]), np.array([0.3, 1.7])
    plain = SmoothnessPrior(tau, eta)
    profiled = SmoothnessPrior(tau, eta, np.ones((2, n)), np.ones((2, n)))
    d = jnp.asarray(np.random.default_rng(2).normal(size=(2, n)))
    np.testing.assert_allclose(
        np.asarray(profiled.apply(d)), np.asarray(plain.apply(d)), rtol=1e-14
    )
    np.testing.assert_allclose(profiled.dense(n), plain.dense(n), rtol=1e-14)
    np.testing.assert_allclose(
        float(prior_logdet(profiled, n)), float(prior_logdet(plain, n)), rtol=1e-13
    )


def test_profile_broadcasts_from_one_row_and_reports_its_grid():
    n = 25
    prior = SmoothnessPrior(np.ones(3), np.ones(3), None, np.linspace(1.0, 2.0, n))
    assert prior.eta_profile.shape == (3, n)
    assert prior.n_pixels == n
    with pytest.raises(ValueError, match="rebuild it"):
        prior.ridge_weights(n + 1)
    with pytest.raises(ValueError, match="same pixel count"):
        SmoothnessPrior(np.ones(2), np.ones(2), np.ones((2, n)), np.ones((2, n + 1)))


def test_window_helpers():
    """Windows merge, filter to a grid, and turn into a two-valued multiplier."""
    windows = nebular_windows(lines=[4861.33], halfwidth_kms=300.0)
    assert len(windows) == 1
    lo, hi = windows[0]
    assert lo < 4861.33 < hi
    np.testing.assert_allclose(hi - lo, 2 * 4861.33 * 300.0 / ab.C_KMS, rtol=1e-12)

    # [S II] 6716/6731 are 14 A apart: at 900 km/s half-width they merge into one.
    assert len(nebular_windows(lines=[6716.44, 6730.82], halfwidth_kms=900.0)) == 1
    assert len(nebular_windows(lines=[6716.44, 6730.82], halfwidth_kms=50.0)) == 2

    # v_kms moves the windows with the nebula.
    (shifted,) = nebular_windows(lines=[4861.33], halfwidth_kms=300.0, v_kms=150.0)
    np.testing.assert_allclose(np.mean(shifted), 4861.33 * (1 + 150.0 / ab.C_KMS), rtol=1e-12)

    visible = nebular_windows(wave_range=(GRID.wave[0], GRID.wave[-1]))
    assert len(visible) == 1  # only H-beta is inside this grid
    assert set(NEBULAR_LINES) > {"H-alpha", "H-beta", "[O III] 5007"}

    profile = window_profile(GRID.wave, visible, inside=1.0, outside=1e6)
    inside = profile == 1.0
    assert 0 < inside.sum() < GRID.n
    assert np.all(profile[~inside] == 1e6)
    np.testing.assert_allclose(GRID.wave[inside].mean(), HBETA, atol=0.6)

    with pytest.raises(ValueError, match="no windows"):
        window_profile(GRID.wave, [])
    with pytest.raises(ValueError, match="overlap the grid"):
        window_profile(GRID.wave, [(1000.0, 1001.0)])
    with pytest.raises(ValueError, match="positive"):
        window_profile(GRID.wave, visible, outside=0.0)


def test_window_profile_confines_the_recovered_component():
    """With a windowed ridge the nebular spectrum is pinned to the continuum outside it."""
    ds, truth = _simulate()
    problem = build_problem(
        GRID,
        ds,
        velocities=truth.velocities,
        light_fractions=ELL,
        lsf_sigma_v={"inst": LSF_KMS},
        nebular=True,
        nebular_amplitudes=truth.nebular_amplitudes,
    )
    hb = problem.half_bandwidth_bound(V_REL_MAX)
    free = marginal_loglikelihood(problem, _prior(3), half_bandwidth=hb)
    confined = marginal_loglikelihood(problem, _prior(3, confine=True), half_bandwidth=hb)
    window = window_profile(GRID.wave, nebular_windows(lines=[HBETA], halfwidth_kms=500.0)) == 1.0
    outside_free = np.max(np.abs(np.asarray(free.d_hat)[2][~window]))
    outside_confined = np.max(np.abs(np.asarray(confined.d_hat)[2][~window]))
    assert outside_confined < 1e-3, f"nebular leaks outside its window: {outside_confined:.2e}"
    assert outside_confined < 0.05 * outside_free
    # Inside the window the line survives the confinement essentially untouched.
    peak = np.max(np.asarray(confined.d_hat)[2][window])
    assert peak > 0.3, f"the confined component lost its line (peak {peak:.3f})"


# ---------------------------------------------------------------------------
# 4. The amplitude parameterization and its gradient
# ---------------------------------------------------------------------------


def test_amplitudes_are_centered_and_shift_invariant():
    u = np.array([0.3, -1.1, 0.7, 0.2])
    a = np.asarray(nebular_amplitudes({"log_nebular_amp": u}))
    np.testing.assert_allclose(np.exp(np.mean(np.log(a))), 1.0, rtol=1e-14)
    shifted = np.asarray(nebular_amplitudes({"log_nebular_amp": u + 4.2}))
    np.testing.assert_allclose(shifted, a, rtol=1e-13)
    np.testing.assert_allclose(a / a[0], np.exp(u - u[0]), rtol=1e-13)


def _model(dataset, *, nebular: bool, confine: bool = False) -> MarginalOrbitModel:
    return MarginalOrbitModel(
        GRID,
        dataset,
        light_fractions=ELL,
        lsf_sigma_v={"inst": LSF_KMS},
        v_rel_max_kms=V_REL_MAX,
        nebular=nebular,
        prior=_prior(3 if nebular else 2, confine=confine),
    )


def _conjunction_time() -> float:
    """t_conj for the circular test orbit (nu + omega = pi/2 at t_peri + P/4)."""
    return TPERI_TRUE + 0.25 * P_TRUE


def _theta(*, nebular: bool) -> dict:
    """θ at the injected orbit. ``secosw = sesinw = 1e-3``, never 0 — the parameterization
    is singular at exactly the origin (design.md D39)."""
    theta = {
        "period": jnp.asarray(P_TRUE),
        "t_conj": jnp.asarray(_conjunction_time()),
        "secosw": jnp.asarray(1e-3),
        "sesinw": jnp.asarray(1e-3),
        "k": jnp.asarray(K_TRUE),
    }
    if nebular:
        theta["log_nebular_amp"] = jnp.zeros(N_EP)
    return theta


def test_gradient_in_the_amplitudes_matches_finite_differences():
    ds, _ = _simulate()
    model = _model(ds, nebular=True, confine=True)
    log_amp = np.log(_amplitudes())

    def f(u):
        return model.log_likelihood({**_theta(nebular=True), "log_nebular_amp": u})

    grad = np.asarray(jax.grad(f)(jnp.asarray(log_amp)))
    step = 1e-5
    for j in (0, 5, N_EP - 1):
        bump = np.zeros(N_EP)
        bump[j] = step
        fd = (float(f(jnp.asarray(log_amp + bump))) - float(f(jnp.asarray(log_amp - bump)))) / (
            2 * step
        )
        assert abs(grad[j] - fd) < 1e-4 * max(1.0, abs(fd)), f"epoch {j}: {grad[j]} vs {fd}"
    # Centering makes the gradient sum to zero: a common shift changes nothing.
    assert abs(float(np.sum(grad))) < 1e-6 * max(1.0, float(np.max(np.abs(grad))))


def test_site_requires_the_component():
    ds, _ = _simulate()
    model = _model(ds, nebular=False)
    with pytest.raises(ValueError, match="without nebular=True"):
        model.log_likelihood(_theta(nebular=True))


def test_inferred_hyperparameters_keep_the_window_profile():
    """ML-II replaces the *scalars*; the profile is structure and must survive.

    Regression test: dropping the profile here would silently un-confine a windowed
    component the moment ``log_tau``/``log_eta`` were sampled — the confinement would
    still be configured, still be documented, and simply not happen.
    """
    ds, _ = _simulate()
    model = _model(ds, nebular=True, confine=True)
    theta = {
        **_theta(nebular=True),
        "log_tau": jnp.log(jnp.asarray([200.0, 200.0, 8.0])),
        "log_eta": jnp.log(jnp.full(3, 2.0)),
    }
    prior = model._prior(theta)
    assert prior.eta_profile is not None
    np.testing.assert_allclose(
        np.asarray(prior.eta_profile), np.asarray(model.fixed_prior.eta_profile)
    )
    # And it reaches the answer: the confined component stays at the continuum outside
    # its window, which an un-profiled prior would not do.
    window = window_profile(GRID.wave, NEB_WINDOW) == 1.0
    d_hat = np.asarray(model.marginal(theta).d_hat)
    assert np.max(np.abs(d_hat[2][~window])) < 1e-3


# ---------------------------------------------------------------------------
# 5. Why the component exists: the core-fill it removes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def core_fill():
    """Disentangle the same data with and without the nebular component, orbit fixed.

    Holding the orbit at truth isolates the claim being tested — this is a statement
    about the *spectra*, not about whether the orbit survives (which
    :func:`test_joint_map_recovers_orbit_and_amplitudes` covers separately).
    """
    ds, truth = _simulate()
    vel = truth.velocities
    common = dict(light_fractions=ELL, lsf_sigma_v={"inst": LSF_KMS}, velocities=vel)
    without = build_problem(GRID, ds, **common)
    with_neb = build_problem(
        GRID, ds, **common, nebular=True, nebular_amplitudes=truth.nebular_amplitudes
    )
    hb = with_neb.half_bandwidth_bound(V_REL_MAX)
    res_without = marginal_loglikelihood(without, _prior(2), half_bandwidth=hb)
    res_with = marginal_loglikelihood(with_neb, _prior(3, confine=True), half_bandwidth=hb)
    return truth, res_without, res_with


def _combination_error(d_hat, truth) -> np.ndarray:
    """Light-weighted stellar combination minus truth, with the low-frequency offset removed.

    The combination is what constant light fractions leave observable (math.md §5.1),
    and its overall level is set by the ridge rather than by the data, so comparing the
    *shape* is the only honest comparison.
    """
    truth_comb = ELL @ np.stack([np.asarray(c) for c in truth.components])
    err = (ELL @ np.asarray(d_hat)[:2]) - truth_comb
    interior = slice(int(0.06 * GRID.n), GRID.n - int(0.06 * GRID.n))
    return err - np.mean(err[interior])


def _line_core() -> np.ndarray:
    """Pixels within ±2 nebular line widths of H-beta."""
    return np.abs(GRID.wave - HBETA) < HBETA * 30.0 / ab.C_KMS


def test_unmodelled_nebular_emission_fills_the_stellar_line_core(core_fill):
    """The whole point: without the component the emission is absorbed by the stars."""
    truth, res_without, res_with = core_fill
    core = _line_core()
    assert core.sum() > 3, "line-core mask is degenerate"

    err_without = _combination_error(res_without.d_hat, truth)
    err_with = _combination_error(res_with.d_hat, truth)

    fill_without = float(np.mean(err_without[core]))
    fill_with = float(np.mean(err_with[core]))
    # Unmodelled: a *positive* (emission-like) bias in the core of an absorption line —
    # exactly the artificial narrowing the literature reports.
    assert fill_without > 0.05, f"expected a visible core-fill, got {fill_without:+.4f}"
    assert abs(fill_with) < 0.2 * fill_without, (
        f"the component did not remove the core-fill: {fill_with:+.4f} vs {fill_without:+.4f}"
    )

    rms_without = float(np.sqrt(np.mean(err_without[core] ** 2)))
    rms_with = float(np.sqrt(np.mean(err_with[core] ** 2)))
    assert rms_with < 0.25 * rms_without, f"core RMS {rms_with:.4f} vs {rms_without:.4f}"


def test_the_core_fill_costs_equivalent_width(core_fill):
    """The quantity that actually propagates: the H-beta equivalent width.

    The disentangled spectra are fed to an atmosphere code, so a filled core is not an
    aesthetic problem — it is a systematically understated line strength, and gravity
    from a Balmer profile is about as sensitive to that as a measurement gets.
    """
    truth, res_without, res_with = core_fill
    truth_comb = ELL @ np.stack([np.asarray(c) for c in truth.components])
    profile = np.abs(GRID.wave - HBETA) < 8.0
    d_lambda = np.gradient(GRID.wave)

    def ew(comb):
        return float(-np.sum(comb[profile] * d_lambda[profile]))

    ew_true = ew(truth_comb)
    err_without = abs(ew(ELL @ np.asarray(res_without.d_hat)[:2]) - ew_true) / ew_true
    err_with = abs(ew(ELL @ np.asarray(res_with.d_hat)[:2]) - ew_true) / ew_true
    assert err_without > 0.05, f"expected a visible EW deficit, got {100 * err_without:.1f}%"
    assert err_with < 0.01, f"EW still off by {100 * err_with:.1f}% with the component"


def test_the_data_prefer_the_nebular_model_by_a_wide_margin(core_fill):
    """Both are marginal likelihoods, so the difference is a Bayes factor."""
    _, res_without, res_with = core_fill
    gain = float(res_with.log_likelihood) - float(res_without.log_likelihood)
    assert gain > 100.0, f"the nebular model gains only {gain:.1f} nats"


def test_recovered_nebular_spectrum_matches_the_injection(core_fill):
    """Up to the amplitude convention, the third component is the injected line."""
    truth, _, res_with = core_fill
    injected = np.asarray(truth.nebular) * float(np.exp(np.mean(np.log(truth.nebular_amplitudes))))
    recovered = np.asarray(res_with.d_hat)[2]
    core = _line_core()
    np.testing.assert_allclose(recovered[core], injected[core], atol=0.06)
    assert float(np.corrcoef(recovered[core], injected[core])[0, 1]) > 0.99


def _map_fit(model, *, nebular: bool, max_steps: int):
    """ML-II MAP over the orbit, the hyperparameters and (if present) the amplitudes.

    Identical priors and starting point either way, apart from the sites the extra
    component owns — so the two fits differ in the model and in nothing else.
    """
    n_comp = 3 if nebular else 2
    tau0 = np.array([200.0, 200.0, 8.0])[:n_comp]
    priors = {
        "period": dist.Normal(P_TRUE + 0.002, 0.01),
        "t_conj": dist.Normal(_conjunction_time() + 0.01, 0.05),
        "secosw": dist.Uniform(-1.0, 1.0),
        "sesinw": dist.Uniform(-1.0, 1.0),
        # Wide enough that a failing fit lands at an optimum rather than on a rail:
        # a nebular-blind fit drives K_2 down hard, and a bound it hits would be
        # reporting the prior instead of the failure.
        "k": dist.Uniform(np.array([10.0, 5.0]), np.array([90.0, 70.0])),
        "log_tau": dist.Normal(np.log(tau0), 3.0),
        "log_eta": dist.Normal(np.full(n_comp, np.log(2.0)), 3.0),
    }
    init = {
        "period": P_TRUE + 0.002,
        "t_conj": _conjunction_time() + 0.01,
        "secosw": 0.05,
        "sesinw": 0.05,
        "k": np.array([50.0, 48.0]),
        "log_tau": np.log(tau0),
        "log_eta": np.full(n_comp, np.log(2.0)),
    }
    if nebular:
        priors["log_nebular_amp"] = dist.Normal(np.zeros(N_EP), 0.5)
        init["log_nebular_amp"] = np.zeros(N_EP)
    return run_map(model.model(priors), init=init, max_steps=max_steps)


@pytest.fixture(scope="module")
def joint_fits():
    """The same contaminated data fitted twice: with the component, and without it."""
    ds, truth = _simulate()
    with_neb = _map_fit(_model(ds, nebular=True, confine=True), nebular=True, max_steps=300)
    without = _map_fit(_model(ds, nebular=False), nebular=False, max_steps=300)
    return truth, with_neb, without


def test_k2_scan_carries_the_component_through_both_models():
    """The faint-companion scan is a matched filter for exactly this contaminant.

    A static emission line is what a companion at ``K_2 = 0`` looks like, so a scan with
    nowhere to put it will happily report one. The plumbing under test is the prior
    slicing: the null model drops the *companion* entry and keeps the nebular one, and
    it is one index off from the pre-D40 version.
    """
    k1, k2_true = 14.0, 44.0
    ell = np.array([0.9, 0.1])
    orbit = OrbitParams(period=P_TRUE, t_peri=TPERI_TRUE, ecc=0.0, omega=0.0, k=(k1, k2_true))
    # Line-rich components, not this module's broad-Balmer pair: a matched filter is
    # localized by sharp features, and a broad line gives a peak too wide to test.
    components = [
        synthetic_deviation_spectrum(GRID, n_lines=30, seed=seed, margin=0.08) for seed in (21, 22)
    ]
    ds, _ = simulate_dataset(
        GRID,
        components,
        bjd=BJD,
        instruments={"inst": _instrument()},
        light_fractions=ell,
        orbit=orbit,
        v_bary=np.linspace(-24.0, 26.0, N_EP),
        frame="barycentric",
        nebular=_nebular_spectrum(),
        nebular_amplitudes=_amplitudes(),
        seed=5,
    )
    tau = np.array([200.0, 60.0, 8.0])
    eta_profile = np.ones((3, GRID.n))
    eta_profile[2] = window_profile(GRID.wave, NEB_WINDOW)
    prior = SmoothnessPrior(tau, np.full(3, 2.0), None, eta_profile)

    common = dict(
        orbit={
            "period": P_TRUE,
            "t_conj": _conjunction_time(),
            "secosw": 1e-3,
            "sesinw": 1e-3,
        },
        k1=k1,
        light_fractions=ell,
        lsf_sigma_v={"inst": LSF_KMS},
        v_rel_max_kms=V_REL_MAX,
        nebular=True,
    )
    result = ab.k2_scan(GRID, ds, k2_grid=np.arange(20.0, 70.0, 6.0), prior=prior, **common)
    assert result.k2_peak == k2_true
    assert result.detection_peak > 0
    assert np.all(np.isfinite(result.detection))

    with pytest.raises(ValueError, match="nebular"):
        ab.k2_scan(
            GRID,
            ds,
            k2_grid=np.array([k2_true]),
            prior=SmoothnessPrior(tau[:2], np.full(2, 2.0)),
            **common,
        )


@pytest.mark.slow
def test_joint_map_recovers_orbit_and_amplitudes(joint_fits):
    """Orbit, hyperparameters and per-epoch amplitudes together, from a cold start."""
    truth, fit, _ = joint_fits
    assert np.isfinite(fit.potential)

    k_map = np.asarray(fit.params["k"])
    for i, bound in enumerate((0.01, 0.015)):
        rel = abs(k_map[i] - K_TRUE[i]) / K_TRUE[i]
        assert rel < bound, f"K_{i + 1} off by {100 * rel:.2f}% (target < {100 * bound:.1f}%)"
    # Measured 0.15% and 0.29%. The secondary gets the looser bound because it carries
    # 30% of the light in a 48 A window; both are set about 5x off the measurement.
    assert abs(float(fit.params["period"]) - P_TRUE) < 2e-3
    assert float(fit.params["ecc"]) < 0.02
    # ML-II finds on its own that the nebular component should be less smooth than the
    # stellar ones — the prior discovering a shape it was told nothing about.
    assert float(fit.params["log_tau"][0]) > float(fit.params["log_tau"][2])

    # The amplitudes are identified only up to a common factor, so compare the
    # centered logs — which is exactly what the model applies.
    got = np.log(np.asarray(nebular_amplitudes(fit.params)))
    want = np.log(truth.nebular_amplitudes)
    want = want - want.mean()
    assert float(np.corrcoef(got, want)[0, 1]) > 0.99, (
        f"amplitude correlation too low\n{got}\n{want}"
    )
    assert float(np.sqrt(np.mean((got - want) ** 2))) < 0.05


@pytest.mark.slow
def test_a_nebular_blind_fit_gets_the_orbit_badly_wrong(joint_fits):
    """The sharpest statement in this file: the contamination is an *orbit* error too.

    A nebular line does not move, so a model with nowhere else to put it represents it
    with whichever stellar component can be made to move least — which drags the
    secondary's semi-amplitude down and pulls the period and the eccentricity after it.
    This is not a subtle bias; it is the fit going somewhere else entirely, and it does
    not even settle (the blind fit is still at |grad| ~ 3e4 where the modelled one is at
    2). Only the primary survives, because 70% of the light pins it.

    The assertions are chosen to sit far from their thresholds rather than to be
    comprehensive: K_2 and the period miss by factors, not percentages, and the
    eccentricity of a circular orbit runs all the way to the solver's clip.
    """
    _, fit, blind = joint_fits
    err = np.abs(np.asarray(fit.params["k"]) - K_TRUE) / np.asarray(K_TRUE)
    err_blind = np.abs(np.asarray(blind.params["k"]) - K_TRUE) / np.asarray(K_TRUE)
    assert err_blind[1] > 0.2, (
        f"expected the blind fit to lose K_2; it was off by {err_blind[1]:.3f}"
    )
    assert err[1] < 0.25 * err_blind[1], f"modelling did not help K_2: {err[1]} vs {err_blind[1]}"
    assert abs(float(blind.params["period"]) - P_TRUE) > 10 * abs(
        float(fit.params["period"]) - P_TRUE
    )
    assert float(blind.params["ecc"]) > 0.5, "the blind fit did not run the eccentricity up"
