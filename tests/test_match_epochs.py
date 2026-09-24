"""The epoch comparison of the label fit (D65): statistics, invariance, optimiser, façade.

``compare="epochs"`` replaces the diagonal comparison against the disentangled components
with the chi-square of the template composite against the epoch spectra,
``L_data(m) = z^T W z - 2 m^T h + m^T G m``, evaluated through the disentangling's own
sufficient statistics at its MAP. Three properties carry the design and are tested here
against constructions that share nothing with the fast path:

- the identity is exact, AR(1) noise and a telluric row held at its posterior mean
  included, and with the stellar operator reduced to the quadrature width a library at its
  own resolving power still needs; the oracle is a dense per-epoch covariance built from
  ``docs/math.md`` §1.4a and the forward model's own prediction, with no band assembly,
  adjoint or link table in it;
- the declared light fractions cancel, so a redeclared light with its smoothness scales
  leaves ``L_data`` unchanged to rounding;
- the bounded Levenberg-Marquardt reaches the labels and the light fraction from a poor
  start, and the rotation restart leaves the plateau below half a model pixel, where the
  objective is exactly flat and no gradient method can move.

Everything is simulated in-test; nothing downloads.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
from test_library import build_library

import albireo as ab
from albireo.forward import apply_model, with_lsf
from albireo.inference import MAPResult
from albireo.kepler import t_conj_from_t_peri
from albireo.library import library_interpolator
from albireo.match import (
    EpochStatistics,
    StarLabels,
    _check_statistics,
    _default_scan_vsini,
    _degenerate_lines,
    _grid_compensated_widths,
    _quadrature_width,
    match_labels,
)
from albireo.preprocess import _replace

TRUTH = {"A": (4830.0, 4.10, -0.20, 14.0), "B": (4390.0, 4.60, -0.20, 9.0)}
RADIUS_RATIO = 0.8
ELL0 = np.array([0.6, 0.4])
LSF_KMS = 7.0
DV_KMS = 5.0
PERIOD, K = 6.0, (40.0, 60.0)
AR1_MAX_GAP = 4  # build_problem's default, which the façade does not override


def _sigma_of(resolving_power):
    return ab.C_KMS / resolving_power / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@pytest.fixture(scope="module")
def library():
    return build_library()


def _components(library, grid):
    """Stellar rows ``(w_i / l0_i) t_i`` and the true light fractions ``w_i(lambda)``."""
    interpolator = library_interpolator(library.resampled_to(grid, medium="air"))
    deviations, log_continua = [], []
    for teff, logg, mh, vsini in TRUTH.values():
        normalized, log_continuum = interpolator(jnp.asarray([teff, logg, mh]))
        kernel = np.asarray(ab.rotational_kernel(vsini / grid.dv_kms))
        deviations.append(np.convolve(np.asarray(normalized) - 1.0, kernel, mode="same"))
        log_continua.append(np.asarray(log_continuum))
    stacked = np.stack(log_continua) + np.log([1.0, RADIUS_RATIO**2])[:, None]
    weights = np.exp(stacked - np.logaddexp(stacked[0], stacked[1])[None, :])
    return (weights / ELL0[:, None]) * np.stack(deviations), weights


def _system(
    library,
    *,
    telluric=False,
    ar1=0.0,
    snr=150.0,
    lights=ELL0,
    scale=(1.0, 1.0),
    render_dv=None,
):
    """A simulated SB2 with a known orbit, and a Fit placed at its declared start.

    No MAP is run: the orbit is declared Fixed at the truth, and the statistics are exact at
    any parameters, so the fit's starting theta serves. ``lights`` and ``scale`` redeclare
    the stars' light fractions with their smoothness scales multiplied by ``scale**2``.
    ``render_dv`` renders the epochs on a finer grid than the 5 km/s model grid; by default
    they are rendered on the model grid itself, and then carry its discretisation.
    """
    # The epochs must cover every line of the toy library, the temperature lines at 5167.3
    # and 5172.7 A included: without them Teff enters only through the continua, and a fit
    # 700 K from the truth has the lower chi-square (a degenerate fixture, not a bad fitter).
    grid = ab.LogGrid.from_wavelength_range(5155.0, 5245.0, dv_kms=DV_KMS)
    render = (
        grid if render_dv is None else ab.LogGrid.from_wavelength_range(5155.0, 5245.0, render_dv)
    )
    rows, weights = _components(library, render)
    extra = {}
    if telluric:
        wave = np.asarray(render.wave)
        extra["telluric"] = -0.08 * np.exp(-0.5 * ((wave - 5201.0) / 0.12) ** 2)
    dataset, _ = ab.simulate_dataset(
        render,
        list(rows),
        bjd=np.linspace(0.3, 5.7, 10),
        instruments={
            "S": ab.InstrumentSpec(
                wave=np.arange(5162.0, 5238.0, 0.06), sigma_v_lsf=LSF_KMS, snr=snr
            )
        },
        light_fractions=ELL0,
        orbit=ab.OrbitParams(period=PERIOD, t_peri=0.0, ecc=0.0, omega=0.0, k=K),
        ar1_phi=ar1,
        seed=5,
        **extra,
    )
    dataset = ab.Dataset(tuple(_replace(e, medium="air") for e in dataset), frame=dataset.frame)
    t_conj = float(t_conj_from_t_peri(0.0, period=PERIOD, ecc=0.0, omega=0.0))
    base = ab.Smoothness()
    components = [
        ab.Star(
            name,
            light=float(light),
            smoothness=ab.Smoothness(tau0=base.tau0 * c**2, eta0=base.eta0 * c**2),
        )
        for name, light, c in zip(("A", "B"), lights, scale, strict=True)
    ]
    if telluric:
        components.append(ab.Telluric())
    dis = ab.Disentangler(
        dataset,
        components=components,
        orbit=ab.Orbit(
            period=ab.Fixed(PERIOD),
            k=(ab.Fixed(K[0]), ab.Fixed(K[1])),
            t_conj=ab.Fixed(t_conj),
            ecc=ab.Fixed(0.0),
        ),
        lsf={"S": LSF_KMS},
        dv_kms=DV_KMS,
        noise_correlation={"S": ar1} if ar1 else None,
    )
    start = MAPResult(
        params=dict(dis.init),
        unconstrained={},
        potential=0.0,
        grad_norm=0.0,
        converged=False,
        num_steps=0,
    )
    return ab.Fit(dis=dis, result=start, hyper={}), weights


def _operator_width(resolving_power, grid_compensation):
    """The stellar operator's width, written out from ``docs/math.md`` §9.2a."""
    variance = LSF_KMS**2
    if resolving_power is not None:
        variance -= _sigma_of(resolving_power) ** 2
    if grid_compensation:
        variance -= 7.0 / 12.0 * DV_KMS**2
    return float(np.sqrt(variance))


def _brute_force_chi2(fit, stellar_rows, resolving_power, grid_compensation=True):
    """``||z - A_e d_e - A_s m||_W^2`` epoch by epoch, from a dense AR(1) covariance.

    ``C_e = alpha^2 D^-1/2 R_phi D^-1/2`` over each run of good pixels whose index gaps do
    not exceed the chain's maximum gap, factorised densely, fed the forward model's
    per-epoch prediction: the non-stellar rows through the declared LSF, the stellar rows
    through the quadrature width less the grid's own ``(7/12) dv^2``.
    """
    model = fit.dis.model
    problem = model.problem_at(fit.theta)
    n_stellar = fit.dis.n_stellar
    d_hat = np.asarray(model.marginal(fit.theta).d_hat)
    others = np.zeros_like(d_hat)
    others[n_stellar:] = d_hat[n_stellar:]
    stars = np.zeros_like(d_hat)
    stars[:n_stellar] = stellar_rows
    operator = problem
    if resolving_power is not None or grid_compensation:
        width = _operator_width(resolving_power, grid_compensation)
        operator = with_lsf(problem, {"S": width})
    predicted_others = apply_model(problem, others)
    predicted_stars = apply_model(operator, stars)
    total = 0.0
    for gi, g in enumerate(problem.groups):
        z, w, r = np.asarray(g.z), np.asarray(g.w), np.asarray(g.r)
        alpha = np.asarray(g.jitter)
        phi = np.asarray(g.ar_phi) if problem.correlated else np.zeros(z.shape[0])
        model_e = np.asarray(predicted_others[gi]) + np.asarray(predicted_stars[gi])
        for e in range(z.shape[0]):
            good = np.flatnonzero(w[e] > 0)
            for run in np.split(good, np.flatnonzero(np.diff(good) > AR1_MAX_GAP) + 1):
                distance = np.abs(run[:, None] - run[None, :]).astype(float)
                corr = np.power(float(phi[e]), distance) if phi[e] != 0 else np.eye(run.size)
                factor = sla.cho_factor(corr, lower=True)
                res = np.sqrt(w[e, run]) / alpha[e] * (z[e, run] - r[e, run] * model_e[e, run])
                total += float(res @ sla.cho_solve(factor, res))
    return total


def _stars(library, **overrides):
    labels = {
        "teff": ab.Between(4000.0, 5500.0),
        "logg": ab.Between(3.0, 5.0),
        "vsini": ab.Between(0.0, 60.0),
        "v_kms": ab.Between(-20.0, 20.0),
        **overrides,
    }
    return {name: StarLabels(library=library, **labels) for name in ("A", "B")}


@pytest.fixture(scope="module")
def correlated_telluric(library):
    return _system(library, telluric=True, ar1=0.3)


@pytest.fixture(scope="module")
def plain(library):
    return _system(library, snr=150.0)


# ---------------------------------------------------------------------------
# the statistics: exact, and blind to the declared light
# ---------------------------------------------------------------------------


def test_epoch_chi2_equals_a_dense_brute_force(correlated_telluric, library):
    """AR(1) noise, a telluric row at its posterior mean, and the quadrature operator.

    With and without the grid compensation; at R = 30,000 ``sigma_q`` = 5.57 km/s stays
    above the 5 km/s model pixel, so no warning is raised.
    """
    fit, _ = correlated_telluric
    rng = np.random.default_rng(1)
    rows, _ = _components(library, fit.dis.grid)
    for resolving_power in (None, 30_000.0):
        for compensate in (True, False):
            stats = fit.epoch_statistics(
                resolving_power=resolving_power, grid_compensation=compensate
            )
            assert stats.conditioned == ("telluric",)
            assert fit.dis.model.problem_at(fit.theta).correlated
            for trial in (np.zeros_like(rows), rows, rows + rng.normal(0.0, 0.01, rows.shape)):
                fast = float(stats.chi2(trial))
                brute = _brute_force_chi2(fit, trial, resolving_power, compensate)
                assert abs(fast - brute) <= 1e-10 * abs(brute), (resolving_power, fast, brute)


def test_the_declared_light_cancels(library):
    """Redeclaring l0 with (tau, eta) scaled by c^2 leaves L_data unchanged to rounding."""
    first, _ = _system(library, telluric=True, ar1=0.3)
    lights = np.array([0.75, 0.25])
    scale = lights / ELL0
    second, _ = _system(library, telluric=True, ar1=0.3, lights=lights, scale=scale)
    a = first.epoch_statistics(resolving_power=30_000.0)
    b = second.epoch_statistics(resolving_power=30_000.0)
    rows, _ = _components(library, first.dis.grid)
    # the same physical template: the label rows carry w / l0, so they scale by l0 / l0'
    redeclared = rows * (ELL0 / lights)[:, None]
    for trial, trial_b in ((rows, redeclared), (0.5 * rows, 0.5 * redeclared)):
        assert abs(float(a.chi2(trial)) - float(b.chi2(trial_b))) <= 1e-10 * float(a.chi2(trial))


def test_resolving_power_sets_the_operator_width_and_is_refused_when_too_coarse(plain):
    fit, _ = plain
    intrinsic = fit.epoch_statistics(grid_compensation=False)
    assert intrinsic.library_resolving_power is None
    assert intrinsic.lsf_sigma_kms == (("S", (LSF_KMS,)),)
    reduced = fit.epoch_statistics(resolving_power=20_000.0, grid_compensation=False)
    expected = float(np.sqrt(LSF_KMS**2 - _sigma_of(20_000.0) ** 2))
    assert reduced.lsf_sigma_kms[0][1][0] == pytest.approx(expected, rel=1e-12)
    assert reduced.operator_sigma_kms == reduced.lsf_sigma_kms
    assert reduced.grid_variance_kms2 == 0.0 and reduced.notes == ()
    # the band is the one assembled from the declared operator with that width
    model = fit.dis.model
    manual = EpochStatistics.from_problem(
        with_lsf(model.problem_at(fit.theta), {"S": expected}),
        names=("A", "B"),
        medium="air",
        light_fractions=ELL0,
        half_bandwidth=model.half_bandwidth,
        block_size=model.block_size,
    )
    np.testing.assert_allclose(np.asarray(reduced.band.diag), np.asarray(manual.band.diag))
    np.testing.assert_allclose(np.asarray(reduced.h), np.asarray(manual.h))
    with pytest.raises(ValueError, match="at or below the instrument"):
        fit.epoch_statistics(resolving_power=15_000.0)  # sigma_lib 8.49 km/s > 7 km/s


def test_the_grid_compensation_removes_seven_twelfths_of_a_model_pixel_squared(plain):
    """``sigma_op^2 = sigma_inst^2 - sigma_lib^2 - (7/12) dv^2``, recorded in its own fields.

    ``grid_compensation=False`` builds exactly the operator of the declared (or quadrature)
    width, band and ``h`` included, as before the compensation existed.
    """
    fit, _ = plain
    model = fit.dis.model
    dv = fit.dis.grid.dv_kms
    assert dv == pytest.approx(DV_KMS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # dv = 5 < sigma_q = 7: nothing to warn about
        stats = fit.epoch_statistics()
    width = float(np.sqrt(LSF_KMS**2 - 7.0 / 12.0 * dv**2))
    assert width == pytest.approx(_operator_width(None, True), rel=1e-6)
    assert stats.grid_variance_kms2 == pytest.approx(7.0 / 12.0 * dv**2, rel=1e-14)
    assert stats.lsf_sigma_kms == (("S", (LSF_KMS,)),)
    assert stats.operator_sigma_kms[0][0] == "S"
    assert stats.operator_sigma_kms[0][1][0] == pytest.approx(width, rel=1e-14)
    assert stats.notes == ()
    manual = EpochStatistics.from_problem(
        with_lsf(model.problem_at(fit.theta), {"S": width}),
        names=("A", "B"),
        medium="air",
        light_fractions=ELL0,
        half_bandwidth=model.half_bandwidth,
        block_size=model.block_size,
    )
    np.testing.assert_allclose(
        np.asarray(stats.band.diag), np.asarray(manual.band.diag), rtol=1e-12
    )
    np.testing.assert_allclose(np.asarray(stats.h), np.asarray(manual.h), rtol=1e-12)
    summary = stats.summary()
    assert "(7/12) dv^2 = 14.583 km^2/s^2" in summary and "7.000 -> 5.867" in summary

    off = fit.epoch_statistics(grid_compensation=False)
    assert off.grid_variance_kms2 == 0.0 and off.operator_sigma_kms == off.lsf_sigma_kms
    declared = EpochStatistics.from_problem(
        model.problem_at(fit.theta),
        names=("A", "B"),
        medium="air",
        light_fractions=ELL0,
        half_bandwidth=model.half_bandwidth,
        block_size=model.block_size,
    )
    np.testing.assert_array_equal(np.asarray(off.band.diag), np.asarray(declared.band.diag))
    np.testing.assert_array_equal(np.asarray(off.band.lower), np.asarray(declared.band.lower))
    np.testing.assert_array_equal(np.asarray(off.h), np.asarray(declared.h))
    assert float(off.zwz) == float(declared.zwz)
    assert "grid compensation: none" in off.summary()


def test_a_compensation_below_half_a_model_pixel_is_floored_and_warned(plain):
    """R = 20,000 leaves sigma_q = 2.91 km/s against a 5 km/s pixel: (7/12) dv^2 = 14.6
    km^2/s^2 exceeds sigma_q^2 = 8.5, so the width is floored at half a pixel and named."""
    fit, _ = plain
    sigma_q = float(np.sqrt(LSF_KMS**2 - _sigma_of(20_000.0) ** 2))
    with pytest.warns(UserWarning, match="grid compensation floored") as record:
        stats = fit.epoch_statistics(resolving_power=20_000.0)
    messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
    assert len(messages) == 1, messages
    message = messages[0]
    assert "dv = 5.000 km/s" in message
    assert f"sigma_q = {sigma_q:.3f} km/s" in message
    assert f"dv < sigma_q / 1.095 = {sigma_q / np.sqrt(1.2):.3f} km/s" in message
    assert stats.operator_sigma_kms[0][0] == "S"
    assert stats.operator_sigma_kms[0][1] == pytest.approx((0.5 * fit.dis.grid.dv_kms,), rel=1e-14)
    assert stats.notes == (message,)
    assert f"! {message}" in stats.summary()
    # A width already narrower than half a pixel is never widened by the floor.
    widths, floored = _grid_compensated_widths([1.5, 2.9, 7.0], DV_KMS)
    np.testing.assert_allclose(widths, [1.5, 2.5, np.sqrt(49.0 - 7.0 / 12.0 * 25.0)])
    np.testing.assert_array_equal(floored, [True, True, False])


def test_a_model_grid_coarser_than_sigma_q_is_warned_without_a_floor(plain):
    """R = 25,000: sigma_q = 4.80 km/s < dv = 5 km/s, and the compensated 2.91 km/s is above
    half a pixel, so it is applied in full and the warning says the kernel is under a pixel."""
    fit, _ = plain
    sigma_q = float(np.sqrt(LSF_KMS**2 - _sigma_of(25_000.0) ** 2))
    width = float(np.sqrt(sigma_q**2 - 7.0 / 12.0 * DV_KMS**2))
    assert 0.5 * DV_KMS < width < sigma_q < DV_KMS
    with pytest.warns(UserWarning, match="below a model pixel") as record:
        stats = fit.epoch_statistics(resolving_power=25_000.0)
    messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
    assert len(messages) == 1 and "floored" not in messages[0], messages
    assert f"sigma_q = {sigma_q:.3f} km/s" in messages[0] and "dv = 5.000 km/s" in messages[0]
    assert stats.operator_sigma_kms[0][1][0] == pytest.approx(width, rel=1e-12)
    assert stats.notes == tuple(messages)


def test_the_consistency_check_reads_the_compensation_from_its_fields(plain, library):
    """A compensated operator passes through its fields; one narrowed by a false resolving
    power, or with its recorded widths or variance edited, is refused."""
    from dataclasses import replace

    fit, _ = plain
    grid, names = fit.dis.grid, ("A", "B")
    resolving = {"A": 30_000.0, "B": 30_000.0}
    compensated = fit.epoch_statistics(resolving_power=30_000.0)
    uncompensated = fit.epoch_statistics(resolving_power=30_000.0, grid_compensation=False)
    for stats in (compensated, uncompensated):
        _check_statistics(stats, names, grid, "air", ELL0, resolving, LSF_KMS)

    # The route before the compensation had fields: declare the resolving power whose width
    # is the one to remove, then restore the library's. The widths are self-consistent, so
    # only the instrument width they imply gives it away.
    sigma_fake = float(np.sqrt(_sigma_of(30_000.0) ** 2 + 7.0 / 12.0 * DV_KMS**2))
    r_fake = ab.C_KMS / sigma_fake / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    faked = replace(
        fit.epoch_statistics(resolving_power=r_fake, grid_compensation=False),
        library_resolving_power=30_000.0,
    )
    np.testing.assert_allclose(
        faked.operator_sigma_kms[0][1], compensated.operator_sigma_kms[0][1], rtol=1e-9
    )
    with pytest.raises(ValueError, match="cannot be told apart from a wrong instrument width"):
        _check_statistics(faked, names, grid, "air", ELL0, resolving, LSF_KMS)
    with pytest.raises(ValueError, match="cannot be told apart"):
        _check_statistics(compensated, names, grid, "air", ELL0, resolving, 6.5)
    edited_variance = replace(compensated, grid_variance_kms2=1.0)
    with pytest.raises(ValueError, match=r"\(7/12\) dv\^2 for this call's grid"):
        _check_statistics(edited_variance, names, grid, "air", ELL0, resolving, LSF_KMS)
    for edited in (
        replace(compensated, operator_sigma_kms=(("S", (6.0,)),)),
        replace(uncompensated, grid_variance_kms2=compensated.grid_variance_kms2),
    ):
        with pytest.raises(ValueError, match="are not the quadrature widths"):
            _check_statistics(edited, names, grid, "air", ELL0, resolving, LSF_KMS)
    # and end to end, before any scan runs
    resolved = library.replace(meta={**library.meta, "resolving_power": 30_000.0})
    with pytest.raises(ValueError, match="cannot be told apart"):
        match_labels(
            grid,
            fit.spectra()[:2],
            stars=_stars(resolved),
            medium="air",
            light_fractions=ELL0,
            lsf_sigma_kms=LSF_KMS,
            compare="epochs",
            statistics=faked,
        )


def test_quadrature_width_and_its_refusal():
    np.testing.assert_allclose(_quadrature_width(11.07, None, "x"), [11.07])
    width = _quadrature_width(11.07, 20_000.0, "x")
    assert width[0] == pytest.approx(np.sqrt(11.07**2 - _sigma_of(20_000.0) ** 2))
    with pytest.raises(ValueError, match="at or below"):
        _quadrature_width(6.0, 20_000.0, "x")


def test_the_offset_pairs_of_the_epoch_comparison_are_grouped_in_the_summary():
    """On every epochs fit the offsets of the two components correlate near -0.99.

    Only their light-weighted sum reaches the epoch spectra, so the pairs say nothing about
    the labels, and listed one by one they hid the pairs that do. An offset paired with a
    label stays listed, and the d_hat comparisons, where no such degeneracy exists, list
    every pair as before.
    """
    flagged = [
        ("offset_A[0]", "offset_B[0]", -0.994),
        ("teff_B", "logg_B", 0.981),
        ("offset_A[1]", "offset_B[1]", -0.991),
        ("offset_B[0]", "ratio_B", 0.962),
        ("offset_A[2]", "offset_B[2]", -0.989),
    ]
    lines = _degenerate_lines(flagged, epochs=True)
    assert lines[:2] == ["", "  Degenerate pairs:"]
    assert "    teff_B / logg_B: +0.981  <- expected when both are free" in lines[2]
    assert lines[3] == "    offset_B[0] / ratio_B: +0.962"
    assert len(lines) == 5
    assert lines[4].startswith(
        "    3 additive-offset pair(s), up to -0.994 (offset_A[0] / offset_B"
    )
    assert "only the light-weighted sum of the components' offsets reaches the epochs" in lines[4]

    native = _degenerate_lines(flagged, epochs=False)
    assert len(native) == 2 + len(flagged) and "additive-offset" not in "".join(native)
    assert _degenerate_lines([], epochs=True) == []
    only_offsets = _degenerate_lines(flagged[:1], epochs=True)
    assert only_offsets[1] == "  Degenerate pairs:" and len(only_offsets) == 3


def test_the_default_rotation_scan_reaches_slow_rotation():
    """The old default started every fit at 37.5 km/s or more on a 0-150 km/s prior."""
    trials = _default_scan_vsini(0.0, 150.0)
    assert min(trials) < 2.0 and 5.0 in trials and 10.0 in trials
    assert max(trials) <= 150.0 and len(trials) <= 5
    narrow = _default_scan_vsini(8.0, 35.0)
    assert all(8.0 <= v <= 35.0 for v in narrow)


# ---------------------------------------------------------------------------
# refusals: statistics are never silently ignored
# ---------------------------------------------------------------------------


def test_statistics_are_refused_by_the_d_hat_comparisons(plain, library):
    fit, _ = plain
    stats = fit.epoch_statistics()
    d_hat = fit.spectra()[:2]
    common = {
        "stars": _stars(library),
        "medium": "air",
        "light_fractions": ELL0,
        "lsf_sigma_kms": LSF_KMS,
    }
    with pytest.raises(ValueError, match="compare is unset"):
        match_labels(fit.dis.grid, d_hat, statistics=stats, **common)
    with pytest.raises(ValueError, match="used only by compare='epochs'"):
        match_labels(fit.dis.grid, d_hat, statistics=stats, compare="native", **common)
    with pytest.raises(ValueError, match="used only by compare='epochs'"):
        fit.match_labels(_stars(library), compare="matched", statistics=stats)
    with pytest.raises(ValueError, match="needs the disentangling's epoch statistics"):
        match_labels(fit.dis.grid, d_hat, compare="epochs", **common)
    with pytest.raises(ValueError, match="light fractions"):
        match_labels(
            fit.dis.grid,
            d_hat,
            compare="epochs",
            statistics=stats,
            **{**common, "light_fractions": [0.5, 0.5]},
        )
    with pytest.raises(ValueError, match="no jitter sites"):
        match_labels(fit.dis.grid, d_hat, compare="epochs", statistics=stats, jitter=True, **common)
    resolved = library.replace(meta={**library.meta, "resolving_power": 20_000.0})
    with pytest.raises(ValueError, match="reduced for R = None"):
        match_labels(
            fit.dis.grid,
            d_hat,
            compare="epochs",
            statistics=stats,
            **{**common, "stars": _stars(resolved)},
        )


def test_matched_mode_convolves_the_template_with_the_quadrature_width(plain, library, monkeypatch):
    """The data side keeps the instrument profile; the template gets only what it lacks."""
    import albireo.match as match_module

    fit, _ = plain
    seen = {}

    class Stop(Exception):
        pass

    def capture(model, **kwargs):
        seen["problem"] = model.model_args[0]
        raise Stop

    monkeypatch.setattr(match_module, "run_map", capture)
    resolved = library.replace(meta={**library.meta, "resolving_power": 20_000.0})
    with pytest.raises(Stop):
        fit.match_labels(_stars(resolved), compare="matched", mh=ab.Between(-1.0, 0.5), top_k=1)
    problem = seen["problem"]
    dv = fit.dis.grid.dv_kms
    expected = ab.gaussian_kernel(np.sqrt(LSF_KMS**2 - _sigma_of(20_000.0) ** 2) / dv)
    for kernel in problem.model_lsf_kernels:
        np.testing.assert_allclose(np.asarray(kernel), np.asarray(expected))
    np.testing.assert_allclose(
        np.asarray(problem.lsf_kernel), np.asarray(ab.gaussian_kernel(LSF_KMS / dv))
    )
    coarse = library.replace(meta={**library.meta, "resolving_power": 15_000.0})
    with pytest.raises(ValueError, match="at or below the instrument"):
        fit.match_labels(_stars(coarse), compare="matched", mh=ab.Between(-1.0, 0.5))


# ---------------------------------------------------------------------------
# the measured light and the velocity table
# ---------------------------------------------------------------------------


def test_disentangled_templates_are_reproduced_by_the_declared_light_not_the_measured_one(
    library,
):
    """Why the pipeline's velocity table holds the declared fractions after an epochs fit.

    The epoch comparison measures the light without reference to the declaration, and on
    the D65 benchmark products it measured it better than the declaration. A disentangled
    component is ``(w / l0) t``, though, so the amplitude that reproduces the epochs with it
    as a TODCOR template is ``l0``: holding the true fraction ``w`` instead scales each
    template's contribution by ``w / l0``. Measured here with the true median light and a
    declaration off by 0.25: the velocities degrade by an order of magnitude or more, while
    templates rescaled by ``l0 / w`` give back the declared case to rounding.
    """
    from dataclasses import replace

    lights = np.array([0.45, 0.55])
    fit, weights = _system(library, lights=lights, snr=150.0)
    true_light = np.median(weights, axis=1)
    true_light = true_light / true_light.sum()
    truth = np.asarray(fit.velocities())
    templates = fit.templates()

    def rms(table):
        v = np.asarray(table.velocity)
        assert np.all(table.good), "every epoch measured"
        return np.sqrt(np.mean((v - truth) ** 2, axis=1))

    declared = fit.measure_velocities(templates=templates, light=list(lights))
    measured = fit.measure_velocities(templates=templates, light=list(true_light))
    rescaled = fit.measure_velocities(
        templates=[
            replace(t, deviation=t.deviation * (lights[i] / true_light[i]))
            for i, t in enumerate(templates)
        ],
        light=list(true_light),
    )
    assert np.all(rms(declared) < 0.5), rms(declared)
    assert np.all(rms(measured) > 3.0 * rms(declared)), (rms(measured), rms(declared))
    np.testing.assert_allclose(rescaled.velocity, declared.velocity, atol=1e-6)


# ---------------------------------------------------------------------------
# the optimiser
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered_finely(library):
    """Epochs rendered on a 1 km/s grid rather than on the 5 km/s model grid.

    The façade's operator removes the model grid's own smoothing, (7/12) dv^2 = 14.6
    km^2/s^2 here, on the premise that the epochs carry none of it. Epochs rendered on the
    model grid itself carry all of it but the frame shift, and against them the compensated
    fit put v sin i of the 9 km/s component at 11.1 km/s. Rendered at a fifth of the pixel
    they carry 0.4 km^2/s^2, as spectra from a telescope carry none.
    """
    return _system(library, snr=150.0, render_dv=1.0)


@pytest.mark.slow
def test_the_epoch_fit_recovers_labels_and_light_from_a_poor_start(rendered_finely, library):
    """Warm-started at 45 km/s in rotation, from one scan candidate; the façade default."""
    fit, weights = rendered_finely
    got = fit.match_labels(
        _stars(library), mh=ab.Between(-1.0, 0.5), scan_vsini=[45.0], top_k=1, max_steps=150
    )
    assert got.compare == "epochs" and got.epoch_fit.status == "converged"
    for i, (name, (teff, logg, mh, vsini)) in enumerate(TRUTH.items()):
        labels = got.labels[name]
        assert abs(labels["teff"] - teff) < 0.02 * teff
        assert abs(labels["logg"] - logg) < 0.15
        assert abs(labels["mh"] - mh) < 0.1
        assert abs(labels["vsini"] - vsini) < 0.2 * vsini
        assert abs(got.flux_ratio[name] - float(np.median(weights[i]))) < 0.02
    # the reported chi-square is L_data at the reported parameters
    assert got.chi2 == pytest.approx(float(got.statistics.chi2(got._rows())), rel=1e-9)
    # the façade's statistics carry the grid compensation, and the assumptions record it
    assert got.assumptions["epoch_grid_variance_kms2"] == pytest.approx(7.0 / 12.0 * DV_KMS**2)
    assert got.assumptions["epoch_quadrature_lsf_sigma_kms"] == {"S": [LSF_KMS]}
    assert got.assumptions["epoch_operator_lsf_sigma_kms"]["S"][0] == pytest.approx(
        _operator_width(None, True)
    )
    assert 0.9 < got.chi2 / got.n_pixels_used < 1.1
    assert got.chi2 < got.chi2_nearest_node < got.chi2_continuum
    # bounds respected, and the formal errors finite and positive where measured
    for name, spec_prefix in (("A", "A"), ("B", "B")):
        assert 4000.0 <= got.labels[name]["teff"] <= 5500.0
        assert 0.0 <= got.labels[name]["vsini"] <= 60.0
        errors = got.errors("laplace")[spec_prefix]
        for label in ("teff", "logg", "vsini", "v_kms"):
            assert np.isfinite(errors[label]) and errors[label] > 0.0
    assert np.isfinite(got.flux_ratio_errors["B"]) and got.flux_ratio_errors["B"] > 0.0
    summary = got.summary()
    assert "epochs comparison" in summary and "restart round" in summary
    # The offsets of the two components reach the epochs only as their light-weighted sum,
    # so their pairs are grouped into one line rather than listed as degeneracies.
    offset_pairs = [
        (a, b)
        for a, b, _ in got.flagged_correlations()
        if a.startswith("offset_") and b.startswith("offset_")
    ]
    listed = [line for line in summary.splitlines() if " / " in line and ": " in line]
    assert not any(line.strip().startswith("offset_") for line in listed), listed
    assert ("additive-offset pair(s)" in summary) is bool(offset_pairs)


@pytest.mark.slow
def test_the_rotation_restart_leaves_the_plateau(plain, library):
    """Started at v sin i = 0 the objective is exactly flat in rotation; the scan escapes."""
    fit, _ = plain
    stats = fit.epoch_statistics()
    options = {"mh": ab.Between(-1.0, 0.5), "scan_vsini": [0.0], "top_k": 1, "statistics": stats}
    stuck = fit.match_labels(_stars(library), restart_rounds=0, **options)
    assert stuck.labels["B"]["vsini"] < 0.5 * DV_KMS
    assert stuck.at_bounds.get("vsini_B") == "rotation plateau"
    assert "vsini" not in stuck.errors("laplace")["B"]
    freed = fit.match_labels(_stars(library), **options)
    assert abs(freed.labels["B"]["vsini"] - TRUTH["B"][3]) < 3.0
    assert any(kind.startswith("vsini") for kind in freed.epoch_fit.moved_by)
    assert freed.chi2 < stuck.chi2 - 1.0


@pytest.mark.slow
def test_the_d_hat_comparisons_still_run_through_the_facade(plain, library):
    fit, _ = plain
    for compare in ("native", "matched"):
        got = fit.match_labels(
            _stars(library), compare=compare, mh=ab.Between(-1.0, 0.5), top_k=1, max_steps=60
        )
        assert got.compare == compare and got.epoch_fit is None
        assert np.isfinite(got.chi2) and got.n_pixels_used == 2 * fit.dis.grid.n


@pytest.mark.slow
def test_draw_refits_are_refused_for_the_epoch_comparison(plain, library):
    fit, _ = plain
    got = fit.match_labels(_stars(library), mh=ab.Between(-1.0, 0.5), top_k=1, restart_rounds=0)
    with pytest.raises(ValueError, match="compare='epochs'"):
        ab.refit_draws(got, np.zeros((2, 2, fit.dis.grid.n)))
