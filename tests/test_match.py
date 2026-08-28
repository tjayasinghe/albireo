"""Tests for the label-matching mode.

The centrepiece is a closed loop: build a toy library, inject two components at *off-node*
labels, add noise at a declared level, and fit. It is a real end-to-end exercise because the
library, the interpolation, the rotation kernel, the dilution model, the nuisance and the
optimizer all have to be right together for the injected labels to come back.

Two invariants carry most of the scientific weight and are tested directly rather than
inferred from a good chi-square:

- adding a constant to a component leaves the labels alone *because* the additive nuisance
  is there, and moves them when it is not — the k = 0 null space of ``docs/math.md`` §5.1;
- a deliberately wrong assumed light fraction is recovered as a fitted radius ratio rather
  than laundered into a wrong temperature.

Everything here is offline and generated in-test; nothing downloads.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_library import build_library

import albireo as ab
from albireo.library import library_interpolator
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
    """Off-node injection, declared noise, labels back well inside the accuracy target.

    The tolerances are the ones the module documents as the point of the exercise: Teff to
    2-3%, log g and [M/H] to 0.15 dex, v sin i to 10%. Past those, a template stops
    limiting the radial velocities, which is what this mode exists to serve.
    """
    for name, (teff, logg, mh, vsini) in TRUTH.items():
        got = baseline.labels[name]
        assert abs(got["teff"] - teff) < 0.02 * teff
        assert abs(got["logg"] - logg) < 0.15
        assert abs(got["mh"] - mh) < 0.15
        assert abs(got["vsini"] - vsini) < 0.1 * vsini


@pytest.mark.slow
def test_fit_is_better_than_both_nulls(baseline):
    """Every number quoted against a null: no template at all, and the nearest raw node."""
    assert baseline.chi2 < baseline.chi2_nearest_node < baseline.chi2_continuum
    # noise was injected at the declared level, so the reduced chi-square lands near one
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
# the two invariants that carry the science
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_additive_nuisance_absorbs_the_unconstrained_zero_point(library, grid):
    """The k = 0 mode must not become a temperature error.

    Each component's constant offset is in the null space of disentangling and is held
    only by the smoothness ridge, so a real ``d_hat`` carries an arbitrary one. With the
    additive nuisance the labels must not care; without it they must visibly move, which
    is what makes the nuisance load-bearing rather than decorative.
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
    # and the fitted zero point reports the offset instead of hiding it
    assert np.max(np.abs(protected.result.params["offset_A"])) > 1e-3


@pytest.mark.slow
def test_wrong_assumed_light_fraction_is_recovered_as_a_radius_ratio(library, grid):
    """A wrong ``l0`` must show up in the dilution, not in the temperatures.

    The data are built with light fractions that differ from the ones declared at
    disentangling time. A fit with no dilution freedom has nowhere to put that but the line
    depths, i.e. Teff; a joint radius-ratio fit puts it where it belongs.
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
# declarations honoured exactly
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_fixed_label_never_moves(library, grid):
    """Exact invariant: what was declared Fixed comes back bit-for-bit."""
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
    assert "teff_A" not in got.result.params  # no site was created at all


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
    """A prior tighter than the node spacing used to surface as numpyro's opaque error."""
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
    """The honest error against the formal one — and the point of quoting both."""
    rng = np.random.default_rng(3)
    base = np.asarray(baseline.problem.data)
    draws = base[None, :, :] + rng.normal(0.0, NOISE, (8, *base.shape))
    got = refit_draws(baseline, draws, max_steps=40)
    spread = got.errors("draws")
    formal = got.errors("laplace")
    assert spread["A"]["teff"] > 0.0
    assert spread["A"]["teff"] > formal["A"]["teff"]
    assert "from draws" in got.summary()


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
    """With Teff and log g both free the pair is expected to correlate, and to say so."""
    got = fit(library, grid, noisy_rows(library, grid))
    report = got.correlation
    assert report["matrix"].shape == (len(report["sites"]),) * 2
    np.testing.assert_allclose(np.diag(report["matrix"]), 1.0, atol=1e-9)
    assert set(got.posterior_over_prior) <= set(report["sites"])
    for _a, _b, rho in got.flagged_correlations():
        assert abs(rho) >= 0.95


def test_covariance_rows_line_up_with_their_sites(baseline):
    """Vector sites make a covariance row not a site; the labels must expand with them."""
    assert baseline.covariance.shape[0] == len(baseline.site_order)
    assert "offset_A[0]" in baseline.site_order and "offset_A[2]" in baseline.site_order
    assert "teff_A" in baseline.site_order
    # a misaligned diagonal is exactly how this went wrong: sanity-check the scale
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
    """The default is `native`, and D55 is why.

    `matched` convolves both sides with the LSF, which reads like the careful choice and was
    the default until AI Phe was fitted. Convolving the residuals correlates them while the
    likelihood stays diagonal, so chi-square is over-counted by ~1/sum(k^2) and v sin i
    absorbs the mis-specification -- on real HARPS data both components went to the floor of
    their prior. The closed loop here cannot see any of that, because its rows never pass
    through an LSF or a disentangling; this test pins the default so that the decision has to
    be taken deliberately rather than drifting back.
    """
    got = fit(library, grid, noisy_rows(library, grid))
    assert got.assumptions["compare"] == "native"


def test_problem_is_a_traceable_pytree(baseline):
    """It must survive a jit boundary as an argument, not be folded in as a constant."""
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
# the facade hook
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
    """`Fit.match_labels` must not need the grid, spectra, lights, LSF or medium again."""
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
    """An undeclared medium is an 83 km/s question, so it is refused rather than guessed."""
    dis = _declaration(blue_library, _redeclared(small_dataset, None))
    fit = dis.fit(max_steps=10)
    with pytest.raises(ValueError, match="air or vacuum"):
        fit.match_labels(_labelled(blue_library))
