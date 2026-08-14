"""Smoke tests for the figure helpers.

These assert structure — that a figure is produced, that it has the panels and labels it
claims, that the data plotted are the data passed in — not pixels. Image comparison would
be brittle across matplotlib versions for no real gain; what actually breaks in plotting
code is an API drift or a shape bug, and both show up here.

The one piece of real numerics in this module, :func:`albireo.plotting._lag1`, is tested
against a known answer rather than smoke-tested: it is the statistic the AR(1) diagnostic
turns on, so it has to be right rather than merely present.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import albireo as ab  # noqa: E402
from albireo import plotting  # noqa: E402
from albireo.forward import build_problem  # noqa: E402
from albireo.likelihood import marginal_loglikelihood  # noqa: E402
from albireo.priors import SmoothnessPrior  # noqa: E402
from albireo.scan import K2ScanResult  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def posterior_samples():
    rng = np.random.default_rng(3)
    n = 40
    return {
        "period": 6.0 + 0.01 * rng.normal(size=n),
        "t_conj": 0.02 * rng.normal(size=n),
        "secosw": 0.01 * rng.normal(size=n),
        "sesinw": 0.01 * rng.normal(size=n),
        "k": np.stack([40.0 + rng.normal(size=n), 60.0 + rng.normal(size=n)], axis=-1),
    }


@pytest.fixture
def fitted_problem(small_grid, small_dataset):
    """A solved small problem: the Problem and its posterior-mean spectra."""
    problem = build_problem(
        small_grid,
        small_dataset,
        velocities=np.zeros((2, len(list(small_dataset)))),
        light_fractions=(0.6, 0.4),
        lsf_sigma_v={"A": 8.25},
    )
    prior = SmoothnessPrior(tau=np.array([300.0, 300.0]), eta=np.array([5.0, 5.0]))
    return problem, marginal_loglikelihood(problem, prior).d_hat


# ---------------------------------------------------------------------------
# orbit
# ---------------------------------------------------------------------------


def test_plot_rv_curve_draws_both_components_and_the_epochs(posterior_samples):
    bjd = np.array([0.7, 2.2, 4.9, 6.0])
    fig, ax = plotting.plot_rv_curve(posterior_samples, bjd, n_draws=5)

    assert ax.get_ylabel() == "radial velocity [km/s]"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "posterior draws, component 1" in labels
    assert "posterior draws, component 2" in labels
    assert "epochs" in labels
    assert fig is ax.figure


def test_plot_phase_fold_wraps_into_the_unit_interval():
    bjd = np.array([0.0, 3.0, 6.0, 9.0])
    _, ax = plotting.plot_phase_fold(bjd, np.arange(4.0), period=6.0, t_conj=0.0)

    assert ax.get_xlim() == (0.0, 1.0)
    np.testing.assert_allclose(plotting.phase_of(bjd, 6.0, 0.0), [0.0, 0.5, 0.0, 0.5])


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------


def test_plot_spectra_accepts_draws_and_makes_one_panel_per_component(small_grid):
    rng = np.random.default_rng(0)
    draws = rng.normal(scale=0.01, size=(20, 2, small_grid.n))

    _, axes = plotting.plot_spectra(small_grid, draws)

    assert len(axes) == 2
    assert axes[0].get_ylabel() == "$d_{1}$"
    assert axes[-1].get_xlabel() == "wavelength [Å]"


def test_plot_spectra_accepts_a_mean_with_explicit_std(small_grid):
    mean = np.zeros((2, small_grid.n))
    std = np.full_like(mean, 0.01)

    _, axes = plotting.plot_spectra(small_grid, mean, std=std)

    # The plotted mean line is the mean passed in, not a transform of it.
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), mean[0])


def test_plot_spectra_can_show_normalized_flux(small_grid):
    mean = np.full((1, small_grid.n), -0.3)
    _, axes = plotting.plot_spectra(small_grid, mean, flux=True)

    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), 0.7)
    assert axes[0].get_ylabel() == "$1 + d_{1}$"


def test_plot_spectra_rejects_a_grid_mismatch(small_grid):
    with pytest.raises(ValueError, match="pixels but the grid has"):
        plotting.plot_spectra(small_grid, np.zeros((2, small_grid.n + 3)))


def test_plot_spectra_rejects_a_bad_rank(small_grid):
    with pytest.raises(ValueError, match="must have shape"):
        plotting.plot_spectra(small_grid, np.zeros(small_grid.n))


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def test_lag1_recovers_a_known_autocorrelation():
    # An AR(1) series with a large phi must show a clearly positive lag-1 coefficient,
    # and white noise must not. This is the discriminator the AR(1) work turns on.
    rng = np.random.default_rng(11)
    white = rng.normal(size=4000)
    correlated = np.empty_like(white)
    correlated[0] = white[0]
    for i in range(1, white.size):
        correlated[i] = 0.7 * correlated[i - 1] + white[i]

    assert plotting._lag1(correlated) == pytest.approx(0.7, abs=0.05)
    assert abs(plotting._lag1(white)) < 0.05
    assert np.isnan(plotting._lag1(np.array([1.0, 2.0])))
    assert np.isnan(plotting._lag1(np.zeros(10)))


def test_plot_residual_zscores_has_three_panels(fitted_problem, small_dataset):
    problem, d_hat = fitted_problem

    _, axes = plotting.plot_residual_zscores(problem, d_hat, bjd=small_dataset.bjd)

    assert len(axes) == 3
    assert axes[0].get_xlabel() == "whitened residual"
    assert axes[2].get_ylabel() == "lag-1 autocorrelation"
    # One point per epoch in the per-epoch panels.
    assert len(axes[1].lines[0].get_xdata()) == len(list(small_dataset))


def test_data_residual_zscores_per_epoch_partitions_the_flat_array(fitted_problem, small_dataset):
    from albireo.forward import data_residual_zscores

    problem, d_hat = fitted_problem

    flat = data_residual_zscores(problem, d_hat)
    per_epoch = data_residual_zscores(problem, d_hat, per_epoch=True)

    assert len(per_epoch) == len(list(small_dataset))
    assert sum(r.size for r in per_epoch) == flat.size
    # Same pixels, just grouped — so the multisets agree.
    np.testing.assert_allclose(np.sort(np.concatenate(per_epoch)), np.sort(flat))


def test_plot_lsf_draws_the_build_time_bound():
    wave = np.array([4000.0, 4500.0, 5000.0])
    _, axes = plotting.plot_lsf(wave, np.array([4.0, 4.2, 4.4]), sigma_max=6.0)

    assert len(axes) == 1
    bound = [line for line in axes[0].lines if line.get_linestyle() == "--"]
    assert bound and bound[0].get_ydata()[0] == pytest.approx(6.0)


def test_plot_lsf_adds_an_h3_panel_and_accepts_samples():
    wave = np.array([4000.0, 4500.0])
    rng = np.random.default_rng(0)
    sigma = 4.0 + rng.normal(scale=0.1, size=(30, 2))
    h3 = rng.normal(scale=0.01, size=(30, 2))

    _, axes = plotting.plot_lsf(wave, sigma, h3=h3)

    assert len(axes) == 2
    assert axes[1].get_ylabel() == "$h_3$ (skewness)"


def test_plot_light_fractions_handles_per_epoch_and_constant():
    rng = np.random.default_rng(5)
    per_epoch = rng.dirichlet([8.0, 5.0], size=(25, 4))
    _, ax = plotting.plot_light_fractions({"light": per_epoch}, bjd=np.arange(4.0))
    assert ax.get_ylabel() == "light fraction $\\ell$"
    assert ax.get_xlabel() == "BJD"

    constant = rng.dirichlet([8.0, 5.0], size=25)
    _, ax2 = plotting.plot_light_fractions({"light": constant})
    assert ax2.get_xlabel() == "epoch"


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def test_plot_detection_marks_the_injection_and_the_threshold():
    grid = np.arange(10.0, 60.0, 5.0)
    result = K2ScanResult(
        k2_grid=grid,
        log_likelihood=np.zeros(grid.size),
        log_likelihood_null=-1.0,
        detection=np.linspace(-2.0, 20.0, grid.size),
        k2_peak=55.0,
        primary=np.zeros(4),
        primary_std=np.zeros(4),
        companion=np.zeros(4),
        companion_std=np.zeros(4),
    )

    _, ax = plotting.plot_detection(result, injected_k2=38.0, threshold=12.0)

    assert ax.get_ylabel() == "$D(K_2)$"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("injected" in label for label in labels)
    assert "calibrated threshold" in labels


def _limit(*, completeness, bracketed=True, ell2_limit=0.02):
    from albireo.calibrate import DetectionLimit

    completeness = np.asarray(completeness, dtype=float)
    return DetectionLimit(
        ell2_grid=np.array([0.01, 0.02, 0.04]),
        null_peaks=np.linspace(-600.0, -500.0, 40),
        signal_peaks=np.zeros((3, 12)),
        threshold=-510.0,
        false_alarm=0.05,
        fap_floor=1.0 / 41.0,
        completeness=completeness,
        confidence=0.95,
        ell2_limit=ell2_limit,
        k2_true=38.0,
        k2_grid=np.arange(10.0, 60.0, 5.0),
        k1_marginalized=False,
        limit_is_bracketed=bracketed,
    )


def test_plot_detection_limit_draws_the_null_and_the_completeness_curve():
    limit = _limit(completeness=[0.3, 0.97, 1.0])
    _, (ax_null, ax_comp) = plotting.plot_detection_limit(limit, observed=-550.0)

    assert "null distribution" in ax_null.get_title()
    labels = [t.get_text() for t in ax_null.get_legend().get_texts()]
    assert any("threshold" in label for label in labels)
    # An observed peak *inside* the null range is drawn, not annotated.
    assert any("observed" in label for label in labels)
    assert ax_comp.get_ylim()[1] > 1.0
    assert "completeness" in ax_comp.get_title()


def test_plot_detection_limit_annotates_an_off_scale_detection():
    """A real companion sits orders of magnitude above the null; the axis must not chase it."""
    limit = _limit(completeness=[0.3, 0.97, 1.0])
    _, (ax_null, _) = plotting.plot_detection_limit(limit, observed=4.0e4)

    labels = [t.get_text() for t in ax_null.get_legend().get_texts()]
    assert not any("observed" in label for label in labels)
    texts = [t.get_text() for t in ax_null.texts]
    assert any("observed peak" in t and "trial-count floor" in t for t in texts)
    assert ax_null.get_xlim()[1] < 1.0e4


def test_plot_detection_limit_says_when_the_limit_is_not_bracketed():
    limit = _limit(completeness=[1.0, 1.0, 1.0], bracketed=False, ell2_limit=0.01)
    _, (_, ax_comp) = plotting.plot_detection_limit(limit)
    assert any("not bracketed" in t.get_text() for t in ax_comp.texts)


# ---------------------------------------------------------------------------
# posterior
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_idata():
    """A tiny posterior carrying orbital sites plus a nuisance site."""
    pytest.importorskip("arviz")
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def model():
        numpyro.sample("period", dist.Normal(40.0, 1.0))
        numpyro.sample("k", dist.Normal(jnp.array([12.0, 60.0]), 1.0))
        numpyro.sample("log_tau", dist.Normal(jnp.zeros(2), 1.0))

    mcmc = MCMC(NUTS(model), num_warmup=40, num_samples=60, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(1))
    return ab.to_inference_data(mcmc)


def test_default_corner_vars_selects_the_orbit_and_drops_nuisances(toy_idata):
    # This is albireo's logic rather than arviz's, so it is tested directly: it survives
    # the arviz 0.x -> 1.x change that made the plot's return type version-dependent.
    assert plotting._default_corner_vars(toy_idata) == ["period", "k"]


def test_default_corner_vars_falls_back_when_nothing_matches():
    class OnlyNuisances:
        def __init__(self):
            self.posterior = {"log_tau": None, "log_eta": None}

    # None means "let arviz decide" — better than plotting an empty figure.
    assert plotting._default_corner_vars(OnlyNuisances()) is None
    assert plotting._default_corner_vars(object()) is None


def test_plot_corner_runs(toy_idata):
    # A smoke test only: arviz 1.x returns its own PlotMatrix rather than an axes array,
    # so there is no version-stable structure to assert on. What this catches is the
    # failure that actually happened — passing styling kwargs that a newer arviz rejects.
    assert plotting.plot_corner(toy_idata) is not None


# ---------------------------------------------------------------------------
# lazy export
# ---------------------------------------------------------------------------


def test_plotting_names_are_exported_lazily():
    # Reached through the package's __getattr__, exactly as albireo.read_dataset is.
    assert ab.plot_spectra is plotting.plot_spectra
    assert "plot_spectra" in dir(ab)
    with pytest.raises(AttributeError):
        _ = ab.plot_nothing_at_all
