"""Tests for the label-matching mode.

The main test is a closed loop: build a toy library, inject two components at off-node
labels, add noise at a declared level, and fit. It is an end-to-end test, because the
injected labels are recovered only if the library, the interpolation, the rotation kernel,
the dilution model, the nuisance and the optimizer are all correct.

Two invariants are tested directly:

- adding a constant to a component leaves the labels unchanged when the additive nuisance
  is fitted, and shifts them when it is not (the k = 0 null space of ``docs/math.md`` §5.1);
- a wrong assumed light fraction is recovered as a fitted radius ratio and does not bias
  the temperature.

All data are generated in the tests. Nothing is downloaded.
"""

import dataclasses
import gc
import warnings
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_library import build_library

import albireo as ab
from albireo import match as match_mod
from albireo.library import SpectralLibrary, library_interpolator
from albireo.match import (
    FixedDilution,
    LabelProblem,
    RadiusRatio,
    ScalarDilution,
    StarLabels,
    match_labels,
    refit_draws,
)

TRUTH = {"A": (4830.0, 4.10, -0.20, 12.0), "B": (4390.0, 4.60, -0.20, 30.0)}
ELL0 = np.array([0.65, 0.35])
NOISE = 0.003


@pytest.fixture(scope="module")
def library():
    return build_library()


@pytest.fixture(scope="module")
def grid():
    return ab.LogGrid.from_wavelength_range(5165.0, 5235.0, dv_kms=5.0)


def clean_rows(library, grid, truth=TRUTH, weights=None):
    """Inject components through the same forward chain the fit inverts."""
    interpolator = library_interpolator(library.resampled_to(grid, medium="air"))
    rows = []
    for i, (teff, logg, mh, vsini) in enumerate(truth.values()):
        deviation = np.asarray(interpolator(jnp.asarray([teff, logg, mh]))[0]) - 1.0
        kernel = np.asarray(ab.rotational_kernel(vsini / grid.dv_kms))
        row = np.convolve(deviation, kernel, mode="same")
        rows.append(row if weights is None else row * weights[i])
    return np.stack(rows)


def noisy_rows(library, grid, seed=7, noise=NOISE, **kwargs):
    rows = clean_rows(library, grid, **kwargs)
    return rows + np.random.default_rng(seed).normal(0.0, noise, rows.shape)


def fit(library, grid, rows, *, noise=NOISE, **kwargs):
    options = {
        "medium": "air",
        "light_fractions": ELL0,
        "lsf_sigma_kms": 6.0,
        "std": np.full_like(rows, noise),
        "mh": ab.Between(-0.9, 0.4),
        "dilution": FixedDilution(),
        "compare": "native",
        "max_steps": 500,
    }
    options.update(kwargs)
    stars = options.pop(
        "stars",
        {
            "A": StarLabels(
                library=library,
                teff=ab.Between(4200.0, 5400.0),
                logg=ab.Between(3.2, 4.9),
                vsini=ab.Between(1.0, 60.0),
                v_kms=ab.Fixed(0.0),
            ),
            "B": StarLabels(
                library=library,
                teff=ab.Between(4100.0, 5000.0),
                logg=ab.Between(3.2, 4.9),
                vsini=ab.Between(1.0, 60.0),
                v_kms=ab.Fixed(0.0),
            ),
        },
    )
    return match_labels(grid, rows, stars=stars, **options)


@pytest.fixture(scope="module")
def baseline(library, grid):
    return fit(library, grid, noisy_rows(library, grid))


# ---------------------------------------------------------------------------
# the closed loop
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_closed_loop_recovers_injected_labels(baseline):
    """Labels injected off-node at declared noise are recovered well inside the accuracy target.

    The tolerances are the target accuracy the module documents: Teff to 2-3%, log g and
    [M/H] to 0.15 dex, v sin i to 10%. At that accuracy a template stops limiting the
    radial velocities, which are the purpose of this mode.
    """
    for name, (teff, logg, mh, vsini) in TRUTH.items():
        got = baseline.labels[name]
        assert abs(got["teff"] - teff) < 0.02 * teff
        assert abs(got["logg"] - logg) < 0.15
        assert abs(got["mh"] - mh) < 0.15
        assert abs(got["vsini"] - vsini) < 0.1 * vsini


@pytest.mark.slow
def test_fit_is_better_than_both_nulls(baseline):
    """The fit is compared with two nulls: no template, and the nearest raw node."""
    assert baseline.chi2 < baseline.chi2_nearest_node < baseline.chi2_continuum
    # noise was injected at the declared level, so the reduced chi-square is near one
    assert 0.7 < baseline.chi2 / baseline.n_pixels_used < 1.4


@pytest.mark.slow
def test_labels_and_errors_are_reported_for_every_component(baseline):
    assert set(baseline.labels) == set(TRUTH)
    formal = baseline.errors("laplace")
    for name in TRUTH:
        assert formal[name]["teff"] > 0.0
        assert baseline.labels[name]["v_kms"] == 0.0  # declared Fixed
        assert "v_kms" in baseline.fixed[name]
    assert "chi2" in baseline.summary()
    assert "Assumed, not measured" in baseline.summary()


@pytest.mark.slow
def test_shared_metallicity_is_one_number(baseline):
    assert baseline.labels["A"]["mh"] == baseline.labels["B"]["mh"]
    assert "mh" in baseline.result.params


@pytest.mark.slow
def test_metallicity_can_be_freed_per_component(library, grid):
    got = fit(
        library,
        grid,
        noisy_rows(library, grid),
        mh={"A": ab.Between(-0.9, 0.4), "B": ab.Between(-0.9, 0.4)},
    )
    assert "mh_A" in got.result.params and "mh_B" in got.result.params
    assert abs(got.labels["A"]["mh"] + 0.2) < 0.2


# ---------------------------------------------------------------------------
# the two invariants tested directly
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_additive_nuisance_absorbs_the_unconstrained_zero_point(library, grid):
    """The k = 0 mode must not become a temperature error.

    Each component's constant offset is in the null space of disentangling and is
    constrained only by the smoothness ridge, so a real ``d_hat`` has an arbitrary one.
    With the additive nuisance the labels must be unchanged. Without it they must shift,
    which shows that the nuisance is needed.
    """
    rows = noisy_rows(library, grid)
    shifted = rows + np.array([0.02, -0.015])[:, None]

    protected = fit(library, grid, shifted)
    exposed = fit(library, grid, shifted, offset_order=None)
    reference = fit(library, grid, rows)

    for name, (teff, *_rest) in TRUTH.items():
        with_nuisance = abs(protected.labels[name]["teff"] - reference.labels[name]["teff"])
        without = abs(exposed.labels[name]["teff"] - teff)
        assert with_nuisance < 0.01 * teff, f"{name}: the nuisance failed to absorb the offset"
        assert without > with_nuisance, f"{name}: removing the nuisance should have hurt"
    # and the offset appears in the fitted zero point
    assert np.max(np.abs(protected.result.params["offset_A"])) > 1e-3


@pytest.mark.slow
def test_wrong_assumed_light_fraction_is_recovered_as_a_radius_ratio(library, grid):
    """A wrong ``l0`` must appear in the dilution, not in the temperatures.

    The data are built with light fractions that differ from the ones declared at
    disentangling time. A fit with no dilution freedom can absorb the difference only in
    the line depths, i.e. Teff. A joint radius-ratio fit absorbs it in the dilution.
    """
    true_weights = np.array([0.50, 0.50])
    rows = clean_rows(library, grid, weights=true_weights / ELL0)
    rows = rows + np.random.default_rng(11).normal(0.0, NOISE, rows.shape)

    tied = fit(library, grid, rows, dilution=RadiusRatio())
    rigid = fit(library, grid, rows, dilution=FixedDilution())

    for name, (teff, *_rest) in TRUTH.items():
        assert abs(tied.labels[name]["teff"] - teff) < abs(rigid.labels[name]["teff"] - teff)
        assert abs(tied.labels[name]["teff"] - teff) < 0.03 * teff
    # the light fractions still sum to one at every pixel, by construction
    np.testing.assert_allclose(tied.light_fractions().sum(axis=0), 1.0, atol=1e-12)
    assert abs(tied.flux_ratio["A"] - 0.5) < 0.1


# ---------------------------------------------------------------------------
# dilution models
# ---------------------------------------------------------------------------


def test_fixed_dilution_returns_the_assumed_fractions(baseline):
    np.testing.assert_allclose(baseline.light_fractions()[:, 0], ELL0)
    assert baseline.assumptions["dilution"] == "fixed"


@pytest.mark.slow
def test_scalar_dilution_is_offered_and_flagged_as_weaker(library, grid):
    got = fit(library, grid, noisy_rows(library, grid), dilution=ScalarDilution())
    assert "scale_A" in got.result.params
    assert "weaker than a joint radius-ratio fit" in got.summary()


def test_radius_ratio_needs_two_components(library, grid):
    rows = noisy_rows(library, grid)[:1]
    with pytest.raises(ValueError, match="at least two components"):
        fit(
            library,
            grid,
            rows,
            light_fractions=[1.0],
            dilution=RadiusRatio(),
            stars={
                "A": StarLabels(
                    library=library,
                    teff=ab.Between(4200.0, 5400.0),
                    logg=ab.Between(3.2, 4.9),
                    vsini=ab.Between(1.0, 60.0),
                    v_kms=ab.Fixed(0.0),
                )
            },
        )


# ---------------------------------------------------------------------------
# declarations applied exactly
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_fixed_label_never_moves(library, grid):
    """A label declared Fixed is returned bit-for-bit."""
    stars = {
        "A": StarLabels(
            library=library,
            teff=ab.Fixed(4750.0),
            logg=ab.Between(3.2, 4.9),
            vsini=ab.Between(1.0, 60.0),
            v_kms=ab.Fixed(0.0),
        ),
        "B": StarLabels(
            library=library,
            teff=ab.Between(4100.0, 5000.0),
            logg=ab.Fixed(4.5),
            vsini=ab.Between(1.0, 60.0),
            v_kms=ab.Fixed(0.0),
        ),
    }
    got = fit(library, grid, noisy_rows(library, grid), stars=stars)
    assert got.labels["A"]["teff"] == 4750.0
    assert got.labels["B"]["logg"] == 4.5
    assert "teff" in got.fixed["A"] and "logg" in got.fixed["B"]
    assert "teff_A" not in got.result.params  # no site was created


def test_specs_reject_ambiguous_declarations(library, grid):
    rows = noisy_rows(library, grid)
    with pytest.raises(TypeError, match="use Between"):
        fit(
            library,
            grid,
            rows,
            stars={
                "A": StarLabels(library=library, teff=(4200.0, 5400.0), logg=4.0, vsini=10.0),
                "B": StarLabels(library=library, teff=4400.0, logg=4.5, vsini=30.0),
            },
        )
    with pytest.raises(TypeError, match="Scanned"):
        fit(
            library,
            grid,
            rows,
            stars={
                "A": StarLabels(library=library, teff=ab.Scanned([4200.0]), logg=4.0, vsini=10.0),
                "B": StarLabels(library=library, teff=4400.0, logg=4.5, vsini=30.0),
            },
        )


def test_priors_narrower_than_the_grid_are_honoured_not_crashed(library, grid):
    """A prior tighter than the node spacing must not raise numpyro's opaque error."""
    got = fit(
        library,
        grid,
        noisy_rows(library, grid),
        stars={
            "A": StarLabels(
                library=library,
                teff=ab.Between(4700.0, 4900.0),
                logg=ab.Between(3.9, 4.3),
                vsini=ab.Between(8.0, 16.0),
                v_kms=ab.Fixed(0.0),
            ),
            "B": StarLabels(
                library=library,
                teff=ab.Between(4300.0, 4500.0),
                logg=ab.Between(4.4, 4.8),
                vsini=ab.Between(25.0, 35.0),
                v_kms=ab.Fixed(0.0),
            ),
        },
        max_steps=60,
    )
    assert 4700.0 < got.labels["A"]["teff"] < 4900.0


def test_a_prior_disjoint_from_the_grid_says_so(library, grid):
    with pytest.raises(ValueError, match="no library node"):
        fit(
            library,
            grid,
            noisy_rows(library, grid),
            stars={
                "A": StarLabels(
                    library=library,
                    teff=ab.Between(9000.0, 9500.0),
                    logg=ab.Between(3.2, 4.9),
                    vsini=ab.Between(1.0, 60.0),
                ),
                "B": StarLabels(
                    library=library,
                    teff=ab.Between(4100.0, 5000.0),
                    logg=ab.Between(3.2, 4.9),
                    vsini=ab.Between(1.0, 60.0),
                ),
            },
        )


def test_a_library_without_a_metallicity_axis_requires_fixed_mh(library, grid):
    single = library.replace(
        label_names=("teff", "logg"),
        nodes=library.nodes[library.nodes[:, 2] == 0.0][:, :2],
        normalized=library.normalized[library.nodes[:, 2] == 0.0],
        log_continuum=library.log_continuum[library.nodes[:, 2] == 0.0],
    )
    with pytest.raises(ValueError, match="no metallicity axis"):
        fit(
            library,
            grid,
            noisy_rows(library, grid),
            stars={
                "A": StarLabels(
                    library=single,
                    teff=ab.Between(4200.0, 5400.0),
                    logg=ab.Between(3.2, 4.9),
                    vsini=ab.Between(1.0, 60.0),
                ),
                "B": StarLabels(
                    library=single,
                    teff=ab.Between(4100.0, 5000.0),
                    logg=ab.Between(3.2, 4.9),
                    vsini=ab.Between(1.0, 60.0),
                ),
            },
        )


# ---------------------------------------------------------------------------
# input contracts
# ---------------------------------------------------------------------------


def test_medium_is_required_and_checked(library, grid):
    with pytest.raises(ValueError, match="medium must be one of"):
        fit(library, grid, noisy_rows(library, grid), medium="Air")


def test_shapes_and_scales_are_validated(library, grid):
    rows = noisy_rows(library, grid)
    with pytest.raises(ValueError, match="Drop telluric and nebular rows"):
        fit(library, grid, np.vstack([rows, rows[:1]]))
    with pytest.raises(ValueError, match="one positive value per star"):
        fit(library, grid, rows, light_fractions=[0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="lsf_sigma_kms must be positive"):
        fit(library, grid, rows, lsf_sigma_kms=0.0)
    with pytest.raises(ValueError, match="std must match"):
        fit(library, grid, rows, std=np.ones(3))
    with pytest.raises(ValueError, match="compare must be"):
        fit(library, grid, rows, compare="deconvolved")
    with pytest.raises(ValueError, match="removed every pixel"):
        fit(library, grid, rows, exclude_angstrom=[(0.0, 1e6)])


def test_excluded_ranges_drop_pixels(library, grid):
    got = fit(
        library,
        grid,
        noisy_rows(library, grid),
        exclude_angstrom=[(5180.0, 5190.0)],
        max_steps=40,
    )
    assert got.n_pixels_used < 2 * grid.n


# ---------------------------------------------------------------------------
# uncertainties
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_draws_refit_gives_a_wider_spread_than_the_curvature(library, grid, baseline):
    """The spread over refitted draws exceeds the formal error, so both are quoted.

    The residuals of a disentangling are correlated over several pixels, and the formal
    error assumes them white. The draws here are correlated over a Gaussian of 3 pixels at
    the per-pixel noise level. With white draws the spread equals the formal error, which
    the test then asserted by a margin of 4 percent once the refits started at the optimum.
    """
    rng = np.random.default_rng(3)
    base = np.asarray(baseline.problem.data)
    kernel = np.exp(-0.5 * (np.arange(-9.0, 10.0) / 3.0) ** 2)
    kernel /= np.sqrt(np.sum(kernel**2))
    white = rng.normal(0.0, NOISE, (8, *base.shape))
    correlated = np.apply_along_axis(np.convolve, -1, white, kernel, mode="same")
    got = refit_draws(baseline, base[None, :, :] + correlated, max_steps=40)
    spread = got.errors("draws")
    formal = got.errors("laplace")
    assert spread["A"]["teff"] > 0.0
    assert spread["A"]["teff"] > formal["A"]["teff"]
    assert "from draws" in got.summary()


def test_draw_refits_start_from_the_optimum(baseline, monkeypatch):
    """Each draw is refitted from the optimum, not from the warm-start node.

    From the node, 40 to 60 steps did not bring a draw equal to the data back to the optimum
    (10 and 32 K short in Teff on this problem), so the spread of the draws included the
    optimizer's distance from convergence and doubled the spread of teff_B.
    """
    real = match_mod.run_map
    seen = []

    def spy(model, **kwargs):
        result = real(model, **kwargs)
        seen.append((kwargs["init"], result))
        return result

    monkeypatch.setattr(match_mod, "run_map", spy)
    data = np.asarray(baseline.problem.data)
    refit_draws(baseline, np.stack([data, data]), max_steps=60)
    assert len(seen) == 2
    for init, result in seen:
        assert set(init) == set(baseline.result.params)
        for key, value in baseline.result.params.items():
            np.testing.assert_allclose(init[key], value, rtol=1e-9)
        for name in ("A", "B"):
            assert abs(result.params[f"teff_{name}"] - baseline.labels[name]["teff"]) < 0.5


def test_errors_refuse_to_invent_a_spread(baseline):
    with pytest.raises(ValueError, match="no draws have been refitted"):
        baseline.errors("draws")
    with pytest.raises(ValueError, match="must be 'laplace' or 'draws'"):
        baseline.errors("mcmc")


def test_refit_draws_validates_its_input(baseline):
    with pytest.raises(ValueError, match=r"draws must be \(n_draws"):
        refit_draws(baseline, np.zeros((4, 3, 10)))


@pytest.mark.slow
def test_identifiability_is_reported_not_hidden(library, grid):
    """With Teff and log g both free the pair is expected to correlate, and the fit reports it."""
    got = fit(library, grid, noisy_rows(library, grid))
    report = got.correlation
    assert report["matrix"].shape == (len(report["sites"]),) * 2
    np.testing.assert_allclose(np.diag(report["matrix"]), 1.0, atol=1e-9)
    assert set(got.posterior_over_prior) <= set(report["sites"])
    for _a, _b, rho in got.flagged_correlations():
        assert abs(rho) >= 0.95


def test_covariance_rows_line_up_with_their_sites(baseline):
    """With vector sites a covariance row is not a site, so the labels must expand with them."""
    assert baseline.covariance.shape[0] == len(baseline.site_order)
    assert "offset_A[0]" in baseline.site_order and "offset_A[2]" in baseline.site_order
    assert "teff_A" in baseline.site_order
    # a misaligned diagonal would give an error on the wrong scale, so the scale is checked
    assert 0.05 < baseline.errors("laplace")["A"]["teff"] < 500.0


# ---------------------------------------------------------------------------
# comparison mode, tracing, derived products
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_matched_and_native_comparison_both_recover_the_labels(library, grid):
    rows = noisy_rows(library, grid)
    for compare in ("matched", "native"):
        got = fit(library, grid, rows, compare=compare)
        assert abs(got.labels["A"]["teff"] - TRUTH["A"][0]) < 0.03 * TRUTH["A"][0]
        assert got.assumptions["compare"] == compare


def test_the_default_comparison_is_native(library, grid):
    """The default is `native` (D55).

    `matched` convolves both sides with the LSF, which appears to be the more careful
    choice. Convolving the residuals correlates them while the likelihood stays diagonal,
    so chi-square is over-counted by ~1/sum(k^2) and v sin i absorbs the mis-specification.
    On the HARPS spectra of AI Phe both components went to the lower bound of their
    v sin i prior. The closed loop here does not detect this, because its rows never pass
    through an LSF or a disentangling. This test asserts the default so that it is not
    changed back unnoticed.
    """
    got = fit(library, grid, noisy_rows(library, grid))
    assert got.assumptions["compare"] == "native"


def test_problem_is_a_traceable_pytree(baseline):
    """It must pass a jit boundary as an argument, not be folded in as a constant."""
    leaves, structure = jax.tree.flatten(baseline.problem)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    rebuilt = jax.tree.unflatten(structure, leaves)
    assert isinstance(rebuilt, LabelProblem)
    assert rebuilt.n_pix == baseline.problem.n_pix
    assert baseline.problem.data.dtype == jnp.float64


@pytest.mark.slow
def test_template_and_nearest_node_are_usable_downstream(baseline):
    template = baseline.template("A")
    assert template.shape == (baseline.problem.n_pix,)
    assert np.all(np.isfinite(template))
    assert 0.0 < template.min() < 1.0 and abs(template.max() - 1.0) < 0.1
    node = baseline.nearest_node("A")
    assert set(node) == {"teff", "logg", "mh"}
    assert node["teff"] in np.unique(baseline.node_tables[0][:, 0])
    with pytest.raises(ValueError, match="unknown component"):
        baseline.template("C")


# ---------------------------------------------------------------------------
# memory of the label stage
# ---------------------------------------------------------------------------

# XLA temporaries of the compiled scan programs, in rows of the model grid. On the 455-node toy
# box below, the program that interpolates and broadens a block of 32 nodes needs 2,560 rows (80
# per node), and the chi-square of the block at one velocity 160. The single program that did
# both over a block of 32 needed 2,592 rows, 5,632 with the earlier slab gather of
# `BoxInterpolator`, and 80,082 for the 455 nodes in one program (176 per node), the
# configuration that needed 11.2 GB at 18,739 model pixels. The counts do not depend on the
# number of pixels (the same at 809 and 4,037).
SCAN_TEMPORARY_ROWS = 4000


class _ScanReached(Exception):
    """Raised in place of the warm-start scan, once ``match_labels`` has built its problem."""


@pytest.fixture(scope="module")
def scan_inputs(grid):
    """The problem and node table that ``match_labels`` passes to the warm-start scan.

    The toy box has the 13 x 5 x 7 = 455 nodes of the BOSZ FGK R = 20,000 library, and the
    labels are declared over the whole box, so that every node is scanned.
    """
    box = build_library(
        teff=np.linspace(4000.0, 7000.0, 13),
        logg=np.linspace(3.0, 5.0, 5),
        mh=np.linspace(-1.0, 0.5, 7),
    )
    captured = {}

    def capture(problem, index, nodes, vsini_trials, v_trials):
        captured.update(problem=problem, nodes=np.asarray(nodes))
        raise _ScanReached

    stars = {
        name: StarLabels(library=box, vsini=ab.Between(1.0, 60.0), v_kms=ab.Fixed(0.0))
        for name in ("A", "B")
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(match_mod, "_scan_component", capture)
        with pytest.raises(_ScanReached):
            fit(box, grid, noisy_rows(box, grid), stars=stars, mh=None)
    assert captured["nodes"].shape == (455, 3)
    return captured["problem"], captured["nodes"]


def test_the_node_scan_runs_in_blocks_and_matches_one_program(scan_inputs, monkeypatch):
    """Every block the scan broadens has ``_NODE_CHUNK`` nodes, and every node is returned.

    The reference is one program over all 455 nodes, and for three nodes the profiled
    chi-square of the full forward model, :func:`_component_model`. The chi-square agrees to
    rounding, and the trial rotation and velocity selected for each node are the same.
    """
    problem, nodes = scan_inputs
    build = match_mod._broadened_nodes
    shapes = set()

    def recording(problem, index, block, vsini):
        shapes.add(block.shape)
        return build(problem, index, block, vsini)

    monkeypatch.setattr(match_mod, "_broadened_nodes", recording)
    vsini_trials, v_trials = (5.0, 20.0), (-30.0, 0.0)
    best = match_mod._scan_component(problem, 0, nodes, vsini_trials, v_trials)
    assert shapes == {(match_mod._NODE_CHUNK, nodes.shape[1])}

    chi2_at = match_mod._scan_chi2_program(problem, 0)
    trials = [(vsini, v) for vsini in vsini_trials for v in v_trials]
    chi2 = np.stack(
        [np.asarray(chi2_at(build(problem, 0, jnp.asarray(nodes), s), v)) for s, v in trials]
    )
    np.testing.assert_allclose(best[:, 0], chi2.min(axis=0), rtol=1e-12)
    np.testing.assert_array_equal(best[:, 1:], np.asarray(trials)[np.argmin(chi2, axis=0)])

    weight = problem.weight[0] / problem.sigma[0] ** 2
    gram_inv = match_mod._nuisance_gram_inv(weight, problem.basis)
    for k in (0, 200, 454):
        for t, (vsini, v) in enumerate(trials):
            row, _ = match_mod._component_model(problem, 0, jnp.asarray(nodes[k]), vsini, v)
            direct = match_mod._profiled_chi2(row, problem.data[0], weight, problem.basis, gram_inv)
            np.testing.assert_allclose(chi2[t, k], float(direct), rtol=1e-12)


def test_the_node_scan_program_has_bounded_temporaries(scan_inputs):
    """XLA's buffer assignment of the scan programs, compiled for the blocks the scan uses.

    One program over the 455 nodes of the BOSZ FGK box needed 11.2 GB of temporaries at
    18,739 model pixels, and about twice that in commit, because jaxlib releases a program's
    temporaries after the next call of the scan has allocated its own. The test fails if the
    nodes are evaluated in larger blocks or if the box interpolator gathers more than its
    stencil. The library slab must be an argument of the broadening program: compiled in as
    a constant, it was copied into the executable (59 MB of program text for a 14.7 MB slab
    at 4,037 pixels) and added to the peak of the scan.
    """
    problem, nodes = scan_inputs
    blocks = match_mod._node_blocks(nodes)
    np.testing.assert_array_equal(np.concatenate(blocks)[: nodes.shape[0]], nodes)
    program = match_mod._broadening_program(problem)
    broaden = program.lower(problem, 0, blocks[0], 5.0).compile()
    analysis = broaden.memory_analysis()
    if analysis is None:
        pytest.skip("this backend reports no buffer assignment")
    assert analysis.temp_size_in_bytes / (8 * problem.n_pix) < SCAN_TEMPORARY_ROWS
    assert analysis.argument_size_in_bytes >= problem.interpolators[0].normalized.nbytes
    deviations = match_mod._broadened_nodes(problem, 0, blocks[0], 5.0)
    chi2_at = match_mod._scan_chi2_program(problem, 0)
    analysis = chi2_at.lower(deviations, 0.0).compile().memory_analysis()
    assert analysis.temp_size_in_bytes / (8 * problem.n_pix) < SCAN_TEMPORARY_ROWS


def test_a_repeated_scan_compiles_nothing(scan_inputs, compilations):
    """The scan's programs are kept: the broadening by the problem's layout, the chi-square
    by the content of its constants. A repeat of the same fit compiled one program per
    component, 0.3 to 0.7 s each."""
    problem, nodes = scan_inputs

    def scan():
        match_mod._scan_component(problem, 0, nodes[:40], (5.0, 20.0), (-10.0, 0.0))

    scan()
    assert compilations(scan) == 0


def test_the_scan_releases_its_programs_on_another_problem_signature(scan_inputs):
    """The scan's programs are kept for one problem signature only.

    A jitted function at module level keeps an executable for every model grid it meets, so
    a process that fitted star after star kept the programs of every grid. A problem of
    another signature, here another grid spacing, releases the programs of the previous
    one.
    """
    problem, _ = scan_inputs
    broaden = match_mod._broadening_program(problem)
    chi2_at = match_mod._scan_chi2_program(problem, 0)
    assert match_mod._broadening_program(problem) is broaden
    assert match_mod._scan_chi2_program(problem, 0) is chi2_at
    released = [weakref.ref(broaden), weakref.ref(chi2_at)]
    del broaden, chi2_at
    other = dataclasses.replace(problem, dx=problem.dx * (1.0 + 1e-6))
    match_mod._broadening_program(other)
    match_mod._scan_chi2_program(other, 0)
    gc.collect()
    assert [ref() for ref in released] == [None, None]


def test_kept_programs_are_bounded_per_signature():
    """At most ``kept`` functions per signature, the oldest released first."""
    programs = match_mod._Programs(kept=2)
    made = [programs.get("a", key, object) for key in range(3)]
    assert list(programs.functions.values()) == made[1:]
    assert programs.get("a", 2, object) is made[2]
    programs.get("b", 0, object)
    assert len(programs.functions) == 1 and programs.signature == "b"


def test_stars_that_share_a_library_share_its_projection(library, grid, monkeypatch):
    """One projection and one interpolator per library object, however many stars use it.

    The pipeline declares one library for every star. Where the library's own cache does
    not apply (writable arrays, as here, or a projection above 96 MB), each star made and
    held its own copy: 0.25 GB per copy at 36,085 model pixels.
    """
    rows = noisy_rows(library, grid)
    projected = []
    resample = SpectralLibrary.resampled_to

    def counted(self, *args, **kwargs):
        projected.append(self)
        return resample(self, *args, **kwargs)

    captured = {}

    def capture(problem, index, nodes, vsini_trials, v_trials):
        captured["problem"] = problem
        raise _ScanReached

    monkeypatch.setattr(SpectralLibrary, "resampled_to", counted)
    monkeypatch.setattr(match_mod, "_scan_component", capture)
    with pytest.raises(_ScanReached):
        fit(library, grid, rows)
    assert len(projected) == 1 and projected[0] is library
    first, second = captured["problem"].interpolators
    assert first is second


# ---------------------------------------------------------------------------
# the hook in the Disentangler interface
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def blue_library():
    """A library covering the shared ``small_grid`` fixture's band, not the module's own."""
    wave = np.linspace(4490.0, 4525.0, 700)
    centers = np.array([4497.0, 4502.5, 4508.0, 4513.5, 4518.0, 4522.0])
    nodes, normalized, continua = [], [], []
    for teff in (4000.0, 4500.0, 5000.0, 5500.0):
        for logg in (3.0, 3.5, 4.0, 4.5, 5.0):
            t, g = (teff - 4800.0) / 600.0, logg - 4.0
            depths = np.array(
                [
                    0.30 + 0.13 * np.tanh(t),
                    0.22 - 0.11 * np.tanh(0.8 * t),
                    0.26 + 0.09 * g + 0.02 * g**2,
                    0.17 - 0.07 * g,
                    0.21,
                    0.19 - 0.03 * np.tanh(t),
                ]
            )
            flux = np.ones_like(wave)
            for center, depth in zip(centers, depths, strict=True):
                flux = flux - depth * np.exp(-0.5 * ((wave - center) / 0.30) ** 2)
            nodes.append((teff, logg))
            normalized.append(flux)
            continua.append(30.0 + 4.0 * np.log(teff / 5000.0) - 0.02 * (wave - wave[0]) / 100.0)
    return ab.SpectralLibrary(
        label_names=("teff", "logg"),  # single composition: mh must be Fixed
        nodes=np.asarray(nodes),
        normalized=np.asarray(normalized),
        log_continuum=np.asarray(continua),
        wave=wave,
        medium="air",
        meta={"grid": "toy-blue"},
    )


def _declaration(library, dataset):
    return ab.Disentangler(
        dataset=dataset,
        components=[
            ab.Star("A", light=0.6, smoothness=ab.Smoothness(tau0=300.0, eta0=5.0)),
            ab.Star("B", light=0.4, smoothness=ab.Smoothness(tau0=300.0, eta0=5.0)),
        ],
        orbit=ab.Orbit(
            period=6.0, k=(ab.Fixed(40.0), ab.Fixed(60.0)), t_conj=ab.Fixed(0.0), ecc=ab.Fixed(0.0)
        ),
        lsf={"A": ab.LSF(sigma_kms=8.25)},
    )


def _labelled(library):
    return {
        "A": ab.StarLabels(
            library=library,
            teff=ab.Between(4100.0, 5400.0),
            logg=ab.Between(3.1, 4.9),
            vsini=ab.Between(1.0, 60.0),
        ),
        "B": ab.StarLabels(
            library=library,
            teff=ab.Between(4100.0, 5400.0),
            logg=ab.Between(3.1, 4.9),
            vsini=ab.Between(1.0, 60.0),
        ),
    }


def _redeclared(dataset, medium):
    """The same epochs with their wavelength scale declared (or not)."""
    return ab.Dataset(
        tuple(
            ab.EpochData(
                wave=e.wave,
                flux=e.flux,
                ivar=e.ivar,
                bjd=e.bjd,
                v_bary=e.v_bary,
                instrument=e.instrument,
                medium=medium,
            )
            for e in dataset
        ),
        frame=dataset.frame,
    )


@pytest.fixture(scope="module")
def facade_fit(small_dataset, blue_library):
    dis = _declaration(blue_library, _redeclared(small_dataset, "air"))
    return dis.fit(max_steps=25)


@pytest.mark.slow
def test_facade_hook_fills_everything_in_from_the_fit(facade_fit, blue_library):
    """`Fit.match_labels` must not need the grid, spectra, light fractions, LSF or medium again."""
    got = facade_fit.match_labels(_labelled(blue_library), max_steps=30, mh=ab.Fixed(0.0))
    assert set(got.labels) == {"A", "B"}
    # every derived input came from the fit rather than from the caller
    assert got.assumptions["light_fractions"] == [0.6, 0.4]
    assert got.assumptions["lsf_sigma_kms"] == pytest.approx(8.25)
    assert got.assumptions["medium"] == "air"
    assert got.problem.n_pix == facade_fit.dis.grid.n


@pytest.mark.slow
def test_facade_hook_refuses_a_partial_or_unknown_declaration(facade_fit, blue_library):
    with pytest.raises(ValueError, match="unknown star"):
        facade_fit.match_labels({**_labelled(blue_library), "C": _labelled(blue_library)["A"]})
    with pytest.raises(ValueError, match="every star"):
        facade_fit.match_labels({"A": _labelled(blue_library)["A"]})


@pytest.mark.slow
def test_facade_hook_refuses_an_undeclared_wavelength_scale(small_dataset, blue_library):
    """An undeclared medium is an 83 km/s ambiguity, so it is rejected rather than guessed."""
    dis = _declaration(blue_library, _redeclared(small_dataset, None))
    fit = dis.fit(max_steps=10)
    with pytest.raises(ValueError, match="air or vacuum"):
        fit.match_labels(_labelled(blue_library))


# ---------------------------------------------------------------------------
# declarations taken as declared
# ---------------------------------------------------------------------------


def _one_star(library, **labels):
    declared = {
        "teff": ab.Between(4200.0, 5400.0),
        "logg": ab.Between(3.2, 4.9),
        "vsini": ab.Between(1.0, 60.0),
        "v_kms": ab.Fixed(0.0),
        **labels,
    }
    return {"A": StarLabels(library=library, **declared)}


def test_a_single_star_is_fitted_with_a_free_scalar_dilution(library, grid):
    """One star always has light 1, the upper end of a light fraction.

    The default dilution of one star, ``ScalarDilution``, put the start of its factor on the
    upper bound of its prior, which no fit can start from, so every single-star fit raised.
    The prior now extends above 1 for a declared light of 1.
    """
    rows = noisy_rows(library, grid)[:1]
    got = fit(
        library,
        grid,
        rows,
        light_fractions=[1.0],
        dilution=None,
        stars=_one_star(library),
        top_k=1,
        max_steps=40,
    )
    assert got.assumptions["dilution"] == "scalar"
    lo, hi = match_mod._spec_bounds(got.specs["scale_A"])
    assert lo < 1.0 < hi
    assert lo < got.result.params["scale_A"] < hi
    assert np.isfinite(got.chi2)


def test_a_prior_between_two_nodes_starts_from_the_bracketing_nodes(library, grid):
    """A prior inside the grid that contains no node value is not disjoint from the grid.

    The toy grid has a Teff node every 250 K, so (4800, 4950) lies between the nodes at
    4750 and 5000. Only a prior outside the grid's range is refused.
    """
    stars = {
        "A": StarLabels(
            library=library,
            teff=ab.Between(4800.0, 4950.0),
            logg=ab.Between(3.2, 4.9),
            vsini=ab.Between(1.0, 60.0),
            v_kms=ab.Fixed(0.0),
        ),
        "B": StarLabels(
            library=library,
            teff=ab.Between(4100.0, 5000.0),
            logg=ab.Between(3.2, 4.9),
            vsini=ab.Between(1.0, 60.0),
            v_kms=ab.Fixed(0.0),
        ),
    }
    got = fit(library, grid, noisy_rows(library, grid), stars=stars, top_k=1, max_steps=20)
    assert 4800.0 < got.labels["A"]["teff"] < 4950.0
    scanned = np.isfinite(got.node_scan[0][:, 0])
    assert set(np.unique(got.node_tables[0][scanned, 0])) == {4750.0, 5000.0}


class _TrialsSeen(Exception):
    """Raised once the warm-start scan has been asked for every component."""


def _scan_trials(library, grid, stars, monkeypatch, **kwargs):
    seen = {}

    def capture(problem, index, nodes, vsini_trials, v_trials):
        seen[index] = ([float(v) for v in vsini_trials], [float(v) for v in v_trials])
        if index == len(stars) - 1:
            raise _TrialsSeen
        return np.zeros((np.asarray(nodes).shape[0], 3))

    monkeypatch.setattr(match_mod, "_scan_component", capture)
    with pytest.raises(_TrialsSeen):
        fit(library, grid, noisy_rows(library, grid), stars=stars, **kwargs)
    return seen


def test_the_scan_trials_follow_each_components_own_prior(library, grid, monkeypatch):
    """The default trials of a component lie inside its own prior and include a fixed value.

    They were the same for every component: velocities symmetric about zero out to the
    largest bound of any star, and rotations from the smallest lower to the largest upper
    bound. A prior of (-80, 10) km/s was scanned over (-10, 10), a Known(-60, 2) over
    (-70, 70), and a fixed v sin i of 12 km/s at 1, 5, 10, 12 and 30 km/s.
    """
    stars = {
        "A": StarLabels(
            library=library,
            teff=ab.Between(4200.0, 5400.0),
            logg=ab.Between(3.2, 4.9),
            vsini=ab.Fixed(12.0),
            v_kms=ab.Between(-80.0, 10.0),
        ),
        "B": StarLabels(
            library=library,
            teff=ab.Between(4100.0, 5000.0),
            logg=ab.Between(3.2, 4.9),
            vsini=ab.Between(25.0, 35.0),
            v_kms=ab.Known(-60.0, 2.0),
        ),
    }
    seen = _scan_trials(library, grid, stars, monkeypatch)
    assert seen[0] == ([12.0], list(np.linspace(-80.0, 10.0, 5)))
    assert seen[1][0] == match_mod._default_scan_vsini(25.0, 35.0)
    np.testing.assert_allclose(seen[1][1], [-64.0, -62.0, -60.0, -58.0, -56.0])


def test_identical_declarations_keep_the_shared_scan_trials(library, grid, monkeypatch):
    """With the same prior for every star the trials are those the scan always used."""
    stars = {
        name: StarLabels(
            library=library,
            teff=ab.Between(4100.0, 5400.0),
            logg=ab.Between(3.2, 4.9),
            vsini=ab.Between(1.0, 60.0),
            v_kms=ab.Between(-50.0, 50.0),
        )
        for name in ("A", "B")
    }
    seen = _scan_trials(library, grid, stars, monkeypatch)
    expected = (match_mod._default_scan_vsini(1.0, 60.0), list(np.linspace(-50.0, 50.0, 5)))
    assert seen[0] == expected and seen[1] == expected
    explicit = _scan_trials(
        library, grid, stars, monkeypatch, scan_vsini=[3.0], scan_velocities=[-5.0, 5.0]
    )
    assert explicit[0] == explicit[1] == ([3.0], [-5.0, 5.0])


def test_a_prior_beyond_a_box_library_is_narrowed_to_it(library, grid):
    """A box library is extrapolated outside its nodes, so a fit there measures nothing.

    The toy box ends at [M/H] = -1.0. A declared (-1.3, 0.4) is narrowed to (-1.0, 0.4),
    with a warning and a record in the assumptions. On an APF blind-test target the fit
    otherwise returned [M/H] = -1.24 against a box ending at -1.0 (D74).
    """
    with pytest.warns(UserWarning, match="narrowed to the library"):
        got = fit(
            library, grid, noisy_rows(library, grid), mh=ab.Between(-1.3, 0.4), top_k=1,
            max_steps=5,
        )  # fmt: skip
    assert match_mod._spec_bounds(got.specs["mh"]) == (-1.0, 0.4)
    assert got.assumptions["label_priors_narrowed_to_library"] == {"mh": [-1.0, 0.4]}
    assert -1.0 <= got.labels["A"]["mh"] <= 0.4


def test_an_unbounded_label_prior_is_held_inside_the_box(library, grid, monkeypatch):
    """A Known label on a box library is bounded by the box in the epoch fit and warned of
    in the d_hat comparisons, whose optimizer has no bounds."""
    captured = {}

    def capture(problem, index, nodes, vsini_trials, v_trials):
        captured["problem"] = problem
        raise _ScanReached

    monkeypatch.setattr(match_mod, "_scan_component", capture)
    stars = {
        "A": StarLabels(
            library=library,
            teff=ab.Known(4800.0, 1000.0),
            logg=ab.Between(3.2, 4.9),
            vsini=ab.Between(1.0, 60.0),
            v_kms=ab.Fixed(0.0),
        ),
        "B": StarLabels(
            library=library,
            teff=ab.Between(4100.0, 5000.0),
            logg=ab.Between(3.2, 4.9),
            vsini=ab.Between(1.0, 60.0),
            v_kms=ab.Fixed(0.0),
        ),
    }
    with pytest.warns(UserWarning, match="teff_A"), pytest.raises(_ScanReached):
        fit(library, grid, noisy_rows(library, grid), stars=stars)
    problem = captured["problem"]
    specs = {
        "mh": ab.Between(-0.9, 0.4),
        "teff_A": ab.Known(4800.0, 1000.0),
        "logg_A": ab.Between(3.2, 4.9),
        "vsini_A": ab.Between(1.0, 60.0),
        "v_A": ab.Fixed(0.0),
        "teff_B": ab.Between(4100.0, 5000.0),
        "logg_B": ab.Between(3.2, 4.9),
        "vsini_B": ab.Between(1.0, 60.0),
        "v_B": ab.Fixed(0.0),
    }
    config = {"names": ("A", "B"), "shared_mh": True, "hull_guard": False, "offsets": False}
    layout, lo, hi, _ = match_mod._epoch_layout(specs, config, problem)
    j = layout.index("teff_A")
    assert (lo[j], hi[j]) == (4000.0, 5500.0)
    k = layout.index("teff_B")
    assert (lo[k], hi[k]) == (4100.0, 5000.0)


def test_a_normal_prior_inside_the_box_is_not_warned(library, grid, monkeypatch):
    """A Known label is warned of only when three standard deviations reach past the box.

    Every Known label on a box library was warned of as reaching beyond it, including
    Known(4800, 50) on the 4000-5500 K box, so routine declarations hid the real case.
    """

    def capture(problem, index, nodes, vsini_trials, v_trials):
        raise _ScanReached

    monkeypatch.setattr(match_mod, "_scan_component", capture)
    common = {"logg": ab.Between(3.2, 4.9), "vsini": ab.Between(1.0, 60.0), "v_kms": ab.Fixed(0.0)}
    stars = {
        "A": StarLabels(library=library, teff=ab.Known(4800.0, 50.0), **common),
        "B": StarLabels(library=library, teff=ab.Known(4400.0, 400.0), **common),
    }
    with warnings.catch_warnings(record=True) as caught, pytest.raises(_ScanReached):
        warnings.simplefilter("always")
        fit(library, grid, noisy_rows(library, grid), stars=stars)
    beyond = [str(w.message) for w in caught if "reach beyond" in str(w.message)]
    assert len(beyond) == 1
    assert "teff_B" in beyond[0] and "teff_A" not in beyond[0]


def test_non_finite_spectra_are_refused_by_name(library, grid):
    """One NaN pixel made every scan chi-square NaN, and the error blamed the priors."""
    rows = noisy_rows(library, grid)
    bad = rows.copy()
    bad[0, 100] = np.nan
    with pytest.raises(ValueError, match="d_hat has 1 non-finite pixel"):
        fit(library, grid, bad)
    std = np.full_like(rows, NOISE)
    std[1, 5:7] = np.nan
    std[0, 9] = -NOISE
    with pytest.raises(ValueError, match="std has 3 NaN or negative value"):
        fit(library, grid, rows, std=std)
    std = np.full_like(rows, NOISE)
    std[1] = np.inf
    with pytest.raises(ValueError, match="std is infinite at every pixel of the row"):
        fit(library, grid, rows, std=std)


@pytest.mark.parametrize("compare", ["native", "matched"])
def test_an_infinite_std_removes_a_pixel_from_the_comparison(library, grid, monkeypatch, compare):
    """A pixel with an infinite std has zero weight, whatever ``d_hat`` holds there.

    In the warm-start scan an infinite std gave ``1/std^2 = 0``, so it served as a mask; in
    the likelihood of the d_hat comparisons its ``log(std)`` was infinite. The pixel now
    enters with zero weight and a finite std. In ``"matched"`` mode the pixels within the
    kernel's half-width go with it, as for ``exclude_angstrom``.
    """
    captured = []
    scan = match_mod._scan_component

    def capture(problem, index, nodes, vsini_trials, v_trials):
        captured.append((problem, np.asarray(nodes)))
        raise _ScanReached

    monkeypatch.setattr(match_mod, "_scan_component", capture)
    rows = noisy_rows(library, grid)
    spike = rows.copy()
    spike[0, 300:305] += 0.5
    std = np.full_like(rows, NOISE)
    std[0, 300:305] = np.inf
    for data in (rows, spike):
        with pytest.raises(_ScanReached):
            fit(library, grid, data, std=std, compare=compare)
    (clean, nodes), (spiked, _) = captured
    half = len(np.asarray(clean.lsf_kernel)) // 2 if compare == "matched" else 0
    weight = np.asarray(clean.weight)
    assert np.all(np.isfinite(np.asarray(clean.sigma)))
    assert np.flatnonzero(weight[0] == 0.0).tolist() == list(range(300 - half, 305 + half))
    assert np.all(weight[1] == 1.0)
    np.testing.assert_array_equal(np.asarray(spiked.weight), weight)
    trials = ((5.0, 20.0), (-10.0, 0.0))
    np.testing.assert_array_equal(
        scan(spiked, 0, nodes[:8], *trials), scan(clean, 0, nodes[:8], *trials)
    )


def test_matched_mode_keeps_excluded_flux_out_of_the_comparison(library, grid, monkeypatch):
    """The LSF spreads flux inside an excluded range into its neighbours, which are dropped.

    ``"matched"`` convolves ``d_hat`` with the instrument profile before the comparison. An
    emission line inside an excluded range then changed the data of the weighted pixels
    within the kernel's half-width of the range (by up to 134 sigma for a line of 0.5).
    """
    captured = []

    def capture(problem, index, nodes, vsini_trials, v_trials):
        captured.append(problem)
        raise _ScanReached

    monkeypatch.setattr(match_mod, "_scan_component", capture)
    rows = noisy_rows(library, grid)
    line = rows.copy()
    wave = np.asarray(grid.wave)
    inside = (wave >= 5195.0) & (wave <= 5200.0)
    line[0, inside] += 0.5 * np.exp(-0.5 * ((wave[inside] - 5197.5) / 0.3) ** 2)
    for data in (rows, line):
        with pytest.raises(_ScanReached):
            fit(
                library,
                grid,
                data,
                compare="matched",
                dilution=RadiusRatio(),
                exclude_angstrom=[(5195.0, 5200.0)],
            )
    clean, emitting = captured
    weighted = np.asarray(clean.weight) > 0
    np.testing.assert_array_equal(np.asarray(emitting.weight), np.asarray(clean.weight))
    np.testing.assert_array_equal(
        np.asarray(emitting.data)[weighted], np.asarray(clean.data)[weighted]
    )
    half = len(np.asarray(clean.lsf_kernel)) // 2
    assert int((~weighted[0]).sum()) == int(inside.sum()) + 2 * half


def test_the_nearest_node_null_keeps_one_metallicity(library, grid):
    """With a shared [M/H] the null is the best pair on one metallicity slice.

    The per-star minima here lie on different slices (A at 0.0, B at -0.5), a pair the
    model cannot represent. Their sum was quoted as the null, below every pair the fit can
    reach, so the pipeline's test that a fit beats both nulls failed spuriously.
    """
    truth = {"A": (4830.0, 4.10, 0.0, 12.0), "B": (4390.0, 4.60, -0.5, 30.0)}
    rows = noisy_rows(library, grid, truth=truth)
    got = fit(library, grid, rows, top_k=1, max_steps=5)
    mh_column = 2
    best = [int(np.argmin(scan[:, 0])) for scan in got.node_scan]
    slices = [got.node_tables[i][best[i], mh_column] for i in range(2)]
    assert slices[0] != slices[1]
    independent = sum(float(np.min(scan[:, 0])) for scan in got.node_scan)
    assert got.chi2_nearest_node == got.candidates[0][0]
    assert got.chi2_nearest_node > independent


def test_a_failed_laplace_approximation_is_reported(library, grid, monkeypatch):
    """The fit is kept, and the summary says why it has no formal errors."""

    def fails(*args, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(match_mod, "laplace_inverse_mass", fails)
    got = fit(library, grid, noisy_rows(library, grid), top_k=1, max_steps=5)
    assert got.covariance is None
    assert got.assumptions["laplace_failed"] == "RuntimeError: out of memory"
    summary = got.summary()
    assert "Laplace approximation failed (RuntimeError: out of memory)" in summary
    assert "laplace_failed" not in summary
