"""The `Disentangler` façade: a declaration compiles to the expert path (D46).

The façade's value is not that it is shorter — it is that four things it derives are
things users get wrong, and that the things it refuses to derive are the ones where a
default would be a scientific claim rather than a convenience. Both halves are tested
here, and the refusals get as much attention as the derivations.

**The velocity budget is derived, not defaulted.** It bounds the largest relative velocity
any two components can reach under the *priors*, which is where the information already
was. Too small and the sampler stalls against a guard it cannot see.

**The `priors` and `init` dicts cannot disagree**, because a spec carries both. In the
low-level path they are two dicts a user writes twice and an assert checks.

**A circular orbit is exact, not approximate.** The `(sqrt(e)cos w, sqrt(e)sin w)`
parameterization is singular at exactly `e = 0` — NaN gradient, and numpyro reports only
"Cannot find valid initial parameters" — so a declared circular orbit does not sample
those sites at all, and a free eccentricity never starts at the origin.

**Light fractions are required.** With constant light fractions the likelihood sees only
`l_i * d_i`, so the value is an assumption the data cannot contradict.
"""

import numpy as np
import pytest

import albireo as ab
from albireo.facade import _ecc_sites, _vector_spec


@pytest.fixture(scope="module")
def dataset():
    return ab.load_example("sb2_sim")


@pytest.fixture(scope="module")
def truth():
    return ab.load_example("sb2_sim", with_truth=True)[1]


def _stars():
    return [ab.Star("primary", light=0.62), ab.Star("secondary", light=0.38)]


def _orbit(**kwargs):
    defaults = {
        "period": ab.Between(5.5, 6.5),
        "k": ab.Between([10.0, 10.0], [90.0, 90.0]),
    }
    return ab.Orbit(**{**defaults, **kwargs})


def _dis(dataset, **kwargs):
    options = {"components": _stars(), "orbit": _orbit(), "lsf": {"DEMO": 6.5}}
    return ab.Disentangler(dataset, **{**options, **kwargs})


# -- what it refuses to guess -------------------------------------------------


def test_light_fractions_are_required_and_must_sum_to_one(dataset):
    """Only l_i * d_i is observable, so this is an assumption, not a default."""
    with pytest.raises(TypeError):
        ab.Star("primary")  # no light=
    with pytest.raises(ValueError, match="sum to 1"):
        _dis(dataset, components=[ab.Star("a", light=0.6), ab.Star("b", light=0.6)])


def test_every_instrument_needs_a_declared_lsf(dataset):
    with pytest.raises(ValueError, match="no LSF declared"):
        _dis(dataset, lsf={"NOT_THE_INSTRUMENT": 6.5})


def test_a_nebular_component_refuses_an_undeclared_wavelength_scale(dataset):
    """Nebular and telluric components are keyed to absolute line positions: 83 km/s."""
    with pytest.raises(ValueError, match="air or vacuum"):
        _dis(dataset, components=[*_stars(), ab.Nebular(v_kms=10.0)])


def test_star_names_must_be_unique(dataset):
    with pytest.raises(ValueError, match="unique"):
        _dis(dataset, components=[ab.Star("a", light=0.5), ab.Star("a", light=0.5)])


def test_a_bare_tuple_is_not_accepted_as_a_range(dataset):
    """(5.5, 6.5) reads as either a range or a two-component vector."""
    with pytest.raises(TypeError, match="deliberately not accepted"):
        assert _dis(dataset, orbit=_orbit(period=(5.5, 6.5))).priors


# -- what it derives ----------------------------------------------------------


def test_the_velocity_budget_bounds_what_the_priors_allow(dataset):
    dis = _dis(dataset)
    budget = dis.velocity_budget
    # Both semi-amplitude priors reach 90, at up to e = 0.95, on topocentric data.
    assert budget.total > (90.0 + 90.0) * (1.0 + 0.95)
    assert any("barycentric" in name for name, _ in budget.terms), (
        "these data are topocentric, so the components move with the barycentric term too"
    )
    assert str(budget).startswith("velocity budget")


def test_narrowing_the_priors_narrows_the_budget(dataset):
    wide = _dis(dataset).velocity_budget.total
    narrow = _dis(dataset, orbit=_orbit(k=ab.Between([30.0, 50.0], [55.0, 75.0]))).velocity_budget
    assert narrow.total < wide, "the budget has to follow the declaration, or it is a constant"


def test_a_budget_override_may_not_shrink_below_what_the_priors_reach(dataset):
    """Sampling stalls against that guard rather than failing, which is worse."""
    with pytest.raises(ValueError, match="smaller than"):
        assert _dis(dataset, velocity_budget_kms=50.0).velocity_budget


def test_a_period_prior_wide_enough_to_be_a_search_warns(dataset):
    """The scan resolves phase at one period — the prior's midpoint — not a period."""
    with pytest.warns(RuntimeWarning, match="not a period search"):
        _dis(dataset, orbit=_orbit(period=ab.Between(2.0, 12.0))).fit(max_steps=1)


def test_a_normal_period_prior_does_not_warn(dataset):
    """The quickstart's own prior must stay quiet, or the warning trains people to ignore it."""
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        _dis(dataset)._warn_if_the_period_prior_is_a_search()


def test_the_grid_covers_the_data_plus_the_budget_and_the_kernel(dataset):
    dis = _dis(dataset)
    grid = dis.grid
    lo = min(float(np.min(epoch.wave)) for epoch in dataset)
    hi = max(float(np.max(epoch.wave)) for epoch in dataset)
    assert grid.wave[0] < lo and grid.wave[-1] > hi


def test_priors_and_init_cannot_disagree(dataset):
    dis = _dis(dataset)
    assert set(dis.priors) == set(dis.init), (
        "a spec carries its own starting value, so the two dicts are built together"
    )


def test_smoothness_is_always_fitted_by_empirical_bayes(dataset):
    dis = _dis(dataset)
    assert "log_tau" in dis.priors and "log_eta" in dis.priors


def test_per_component_smoothness_reaches_the_prior(dataset):
    """A rotationally broad star wants a tau five orders of magnitude from a sharp one."""
    dis = _dis(
        dataset,
        components=[
            ab.Star("sharp", light=0.62, smoothness=ab.Smoothness(tau0=1.0e3)),
            ab.Star("broad", light=0.38, smoothness=ab.Smoothness(tau0=1.0e8)),
        ],
    )
    assert np.allclose(np.asarray(dis.smoothness_prior.tau), [1.0e3, 1.0e8])


def test_expert_hands_back_the_low_level_triple(dataset):
    model, priors, init = _dis(dataset).expert()
    assert isinstance(model, ab.MarginalOrbitModel)
    assert set(priors) == set(init)


def test_explain_names_every_derivation_and_every_assumption(dataset):
    text = _dis(dataset).explain()
    for expected in ("velocity budget", "model grid", "sampled sites", "Assumed, not measured"):
        assert expected in text, f"explain() should mention {expected!r}"
    assert "primary=0.62" in text


# -- the eccentricity singularity ---------------------------------------------


def test_a_circular_orbit_does_not_sample_the_singular_sites(dataset):
    """At exactly e = 0 the gradient is NaN; not sampling means never visiting it."""
    dis = _dis(dataset, orbit=_orbit(ecc=ab.Fixed(0.0)))
    assert "secosw" not in dis.priors and "sesinw" not in dis.priors
    assert float(np.asarray(dis.fixed["secosw"])) == 0.0


def test_a_free_eccentricity_never_starts_at_the_origin(dataset):
    dis = _dis(dataset)
    start = np.hypot(float(np.asarray(dis.init["secosw"])), float(np.asarray(dis.init["sesinw"])))
    assert start > 1e-3, (
        "starting at secosw = sesinw = 0 gives a NaN gradient, which numpyro reports only "
        "as 'Cannot find valid initial parameters'"
    )


def test_a_fixed_nonzero_eccentricity_needs_omega():
    orbit = ab.Orbit(period=ab.Fixed(6.0), k=ab.Fixed([40.0, 60.0]), ecc=ab.Fixed(0.3))
    with pytest.raises(ValueError, match="needs omega"):
        _ecc_sites(orbit, 0.95)


def test_a_fixed_eccentricity_round_trips_through_the_parameterization():
    orbit = ab.Orbit(
        period=ab.Fixed(6.0), k=ab.Fixed([40.0, 60.0]), ecc=ab.Fixed(0.36), omega=ab.Fixed(0.7)
    )
    sites = dict(_ecc_sites(orbit, 0.95))
    secosw = float(np.asarray(sites["secosw"].value))
    sesinw = float(np.asarray(sites["sesinw"].value))
    assert np.isclose(secosw**2 + sesinw**2, 0.36)
    assert np.isclose(np.arctan2(sesinw, secosw), 0.7)


def test_an_eccentricity_above_the_solver_clip_is_refused():
    orbit = ab.Orbit(period=ab.Fixed(6.0), k=ab.Fixed([40.0, 60.0]), ecc=ab.Between(0.0, 0.99))
    with pytest.raises(ValueError, match="exceeds ecc_max"):
        _ecc_sites(orbit, 0.95)


# -- things an adversarial review of this module caught -----------------------


def _air(dataset):
    from albireo.preprocess import _replace

    return ab.Dataset(tuple(_replace(e, medium="air") for e in dataset), frame=dataset.frame)


def test_component_order_follows_the_model_not_the_declaration(dataset):
    """The model orders rows stars-telluric-nebular whatever order they were declared in.

    Assembling the smoothness rows in declaration order instead is silent: the vectors are
    still the right length, so the only guard downstream (a length check) passes, and a
    rotationally broadened star gets a sharp-lined star's curvature penalty.
    """
    dis = ab.Disentangler(
        _air(dataset),
        components=[
            ab.Telluric(smoothness=ab.Smoothness(tau0=50.0)),
            ab.Star("broad", light=0.6, smoothness=ab.Smoothness(tau0=1.0e8)),
            ab.Star("sharp", light=0.4, smoothness=ab.Smoothness(tau0=1.0e3)),
        ],
        orbit=_orbit(),
        lsf={"DEMO": 6.5},
    )
    assert dis.component_names == ("broad", "sharp", "telluric")
    assert np.allclose(np.asarray(dis.smoothness_prior.tau), [1.0e8, 1.0e3, 50.0])
    assert np.allclose(np.exp(np.asarray(dis.priors["log_tau"].loc)), [1.0e8, 1.0e3, 50.0])


def test_a_declared_eccentricity_bound_is_actually_enforced(dataset):
    """The two sites are bounded independently, so their box corner reaches e = 2 * hi."""
    dis = _dis(dataset, orbit=_orbit(ecc=ab.Between(0.0, 0.08)))
    assert dis.effective_ecc_max == pytest.approx(0.08), (
        "the model's disk factor is what enforces the bound; without it the declaration "
        "is decoration and the fit returns an eccentricity above what was declared"
    )


def test_an_eccentricity_lower_bound_is_refused_rather_than_dropped(dataset):
    """A lower bound on e is an annulus in (sqrt(e)cos w, sqrt(e)sin w), not a box."""
    with pytest.raises(ValueError, match="lower bound must be 0"):
        assert _dis(dataset, orbit=_orbit(ecc=ab.Between(0.3, 0.6))).priors


def test_a_sampled_semi_amplitude_must_declare_its_own_reach(dataset):
    """The starting value is not a bound, and the velocity budget is derived from bounds."""
    import numpyro.distributions as dist

    loose = ab.Sampled(dist.Normal(40.0, 30.0), start_at=[40.0, 40.0])
    with pytest.raises(ValueError, match="upper_bound"):
        assert _dis(dataset, orbit=_orbit(k=loose)).velocity_budget
    bounded = ab.Sampled(dist.Normal(40.0, 30.0), start_at=[40.0, 40.0], upper_bound=150.0)
    assert _dis(dataset, orbit=_orbit(k=bounded)).velocity_budget.total > 150.0


def test_the_conjunction_prior_is_anchored_on_the_data(dataset):
    """Real epochs sit near BJD 2.46e6; a prior centred on the origin excludes them all."""
    dis = _dis(dataset)
    prior = dis.priors["t_conj"]
    first = float(np.min(np.asarray(dataset.bjd)))
    assert float(prior.low) <= first <= float(prior.high)


def test_the_model_grid_follows_the_finest_epoch_not_the_median(dataset):
    """A grid coarser than a contributing epoch throws that epoch's resolution away."""
    dis = _dis(dataset)
    finest = min(
        float(np.median(np.diff(np.asarray(e.wave)) / np.asarray(e.wave)[:-1]) * ab.C_KMS)
        for e in dataset
    )
    assert dis.grid.dv_kms <= finest * 1.001


def test_replacing_a_declaration_does_not_inherit_the_old_model(dataset):
    """dataclasses.replace copies declared fields, and a derivation cache is not one."""
    import dataclasses

    wide = _dis(dataset)
    _ = wide.grid
    narrow = dataclasses.replace(wide, orbit=_orbit(k=ab.Between([10.0, 10.0], [40.0, 40.0])))
    assert narrow.velocity_budget.total < wide.velocity_budget.total


def test_mutating_the_components_list_afterwards_cannot_desynchronize_it(dataset):
    components = _stars()
    dis = _dis(dataset, components=components)
    components.append(ab.Star("late", light=0.5))
    assert dis.component_names == ("primary", "secondary")


def test_a_light_fraction_of_zero_is_not_a_component(dataset):
    with pytest.raises(ValueError, match="finite and positive"):
        _dis(dataset, components=[ab.Star("a", light=1.0), ab.Star("b", light=0.0)])


def test_a_hierarchical_triple_says_it_is_not_in_v1(dataset):
    """The model supports it; the façade's vocabulary does not, and does not pretend to."""
    outer = ab.Orbit(period=ab.Fixed(400.0), k=ab.Fixed([5.0, 20.0]), t_conj=ab.Fixed(0.0))
    with pytest.raises(NotImplementedError, match="expert"):
        _dis(dataset, orbit=_orbit(outer=outer))


def test_a_scan_declaration_will_not_pretend_to_be_a_fit(dataset):
    """A Scanned semi-amplitude is a grid axis, not a site with a posterior."""
    dis = _dis(dataset, orbit=_orbit(k=[ab.Fixed(40.0), ab.Scanned(np.arange(5.0, 60.0, 5.0))]))
    with pytest.raises(ValueError, match=r"dis\.scan"):
        dis.fit()
    assert "scan declaration" in dis.explain()


def test_a_fixed_period_still_gets_its_conjunction_scanned(dataset):
    """The period lives in `fixed`, not `init`, and the scan has to look in both."""
    dis = _dis(dataset, orbit=_orbit(period=ab.Fixed(6.0)))
    assert "period" in dis.fixed
    fit = dis.fit(max_steps=1)
    assert fit.phase_scan is not None


def test_an_anchored_lsf_is_refused_by_the_scan_rather_than_truncated(dataset):
    dis = ab.Disentangler(
        dataset,
        components=[ab.Star("a", light=0.97), ab.Star("b", light=0.03)],
        orbit=ab.Orbit(
            period=ab.Fixed(6.0),
            t_conj=ab.Fixed(0.5),
            ecc=ab.Fixed(0.0),
            k=[ab.Fixed(40.0), ab.Scanned(np.arange(20.0, 60.0, 10.0))],
        ),
        lsf={"DEMO": ab.LSF([6.0, 7.0], anchors_angstrom=[4500.0, 4560.0])},
    )
    with pytest.raises(ValueError, match="one line-spread width per instrument"):
        dis.scan()


def test_nebular_windows_move_onto_a_vacuum_grid(dataset):
    """NEBULAR_LINES are air wavelengths; unconverted they sit 83 km/s off a vacuum grid."""
    from albireo.preprocess import _replace

    centres = {}
    for medium in ("air", "vacuum"):
        ds = ab.Dataset(tuple(_replace(e, medium=medium) for e in dataset), frame=dataset.frame)
        # A line inside this dataset's own 4505-4555 A window, declared in air.
        dis = _dis(ds, components=[*_stars(), ab.Nebular(v_kms=0.0, lines=[4530.0])])
        profile = np.asarray(dis.smoothness_prior.eta_profile)[-1]
        inside = np.asarray(dis.grid.wave)[profile < profile.max()]
        assert inside.size, f"{medium}: the nebular row came out unconfined"
        centres[medium] = float(np.mean(inside))
    shift = (centres["vacuum"] - centres["air"]) / centres["air"] * ab.C_KMS
    assert 70.0 < shift < 95.0, (
        f"the vacuum window should sit ~83 km/s redward of the air one; got {shift:.1f}"
    )


def test_a_nebular_component_with_no_lines_on_the_grid_is_refused(dataset):
    """A component pinned to the continuum everywhere is one that can do nothing."""
    from albireo.preprocess import _replace

    ds = ab.Dataset(tuple(_replace(e, medium="air") for e in dataset), frame=dataset.frame)
    with pytest.raises(ValueError, match="fall outside the model grid"):
        assert _dis(ds, components=[*_stars(), ab.Nebular(v_kms=0.0)]).smoothness_prior


# -- the spec vocabulary ------------------------------------------------------


def test_a_vector_site_may_be_one_spec_or_one_spec_per_star():
    combined = _vector_spec(ab.Between([1.0, 2.0], [3.0, 4.0]), 2, "k")
    split = _vector_spec([ab.Between(1.0, 3.0), ab.Between(2.0, 4.0)], 2, "k")
    assert np.allclose(np.asarray(combined.hi), np.asarray(split.hi))


def test_mixing_spec_kinds_in_one_vector_site_is_refused():
    with pytest.raises(TypeError, match="one distribution family"):
        _vector_spec([ab.Between(1.0, 3.0), ab.Known(2.0, 0.1)], 2, "k")


def test_the_wrong_number_of_entries_is_caught():
    with pytest.raises(ValueError, match="3 entries"):
        _vector_spec([ab.Fixed(1.0)] * 3, 2, "orbit.k")


def test_resolution_is_converted_as_a_fwhm():
    """c / R is the FWHM; using it as sigma is a factor of 2.35 in the kernel radius."""
    assert np.isclose(ab.LSF.from_resolution(48_000).sigma_kms, 2.6528, atol=1e-3)
    with pytest.raises(ValueError, match="must be positive"):
        ab.LSF.from_resolution(0.0)


# -- the SB1 workflow ---------------------------------------------------------


def test_a_scan_needs_a_fixed_orbit(dataset):
    dis = _dis(dataset, orbit=_orbit(k=[ab.Fixed(40.0), ab.Scanned(np.arange(5.0, 60.0, 5.0))]))
    with pytest.raises(ValueError, match="must be declared Fixed"):
        dis.scan()


def test_a_scan_needs_a_scanned_companion(dataset):
    dis = _dis(
        dataset,
        orbit=ab.Orbit(
            period=ab.Fixed(6.0),
            t_conj=ab.Fixed(0.5),
            ecc=ab.Fixed(0.0),
            k=ab.Fixed([40.0, 60.0]),
        ),
    )
    with pytest.raises(ValueError, match=r"ab\.Scanned"):
        dis.scan()


@pytest.mark.slow
def test_a_scan_finds_the_companion_it_was_pointed_at(dataset, truth):
    """The same declaration drives the scan, so its arguments cannot drift out of step."""
    dis = ab.Disentangler(
        dataset,
        components=[ab.Star("primary", light=0.97), ab.Star("companion", light=0.03)],
        orbit=ab.Orbit(
            period=ab.Fixed(float(truth["period"])),
            t_conj=ab.Fixed(0.73171),
            ecc=ab.Fixed(0.0),
            # Known, not Fixed: K1 is marginalized, which is the only thing that catches a
            # wrong K1 inflating the detection statistic rather than blurring it.
            k=[ab.Known(float(truth["k"][0]), 1.5), ab.Scanned(np.arange(20.0, 90.0, 10.0))],
        ),
        lsf={"DEMO": 6.5},
        dv_kms=6.0,
    )
    result = dis.scan()
    peak = float(result.k2_grid[int(np.argmax(result.detection))])
    assert abs(peak - truth["k"][1]) <= 10.0, f"scan peaked at K2 = {peak}, truth {truth['k'][1]}"


# -- the closed loop ----------------------------------------------------------


@pytest.mark.slow
def test_the_facade_recovers_the_injected_orbit(dataset, truth):
    """Twelve lines to the same answer the fifty-nine-line expert path gives."""
    dis = _dis(dataset)
    fit = dis.fit(max_steps=150)

    assert np.isclose(float(fit.orbit()["period"]), truth["period"], atol=1e-3)
    for name, k_true in zip(("primary", "secondary"), truth["k"], strict=True):
        assert np.isclose(fit.star(name)["k"], k_true, atol=0.2), name
    assert np.isclose(float(fit.orbit()["ecc"]), truth["ecc"], atol=0.02)
    # The phase scan is what makes a cold start work at all.
    assert fit.phase_scan is not None and fit.phase_scan.contrast > 1e3
    assert 0.8 < fit.z_rms < 1.2, f"noise model mismatch: z RMS {fit.z_rms}"


@pytest.mark.slow
def test_the_hyperparameters_come_back_keyed_by_component_name(dataset):
    fit = _dis(dataset).fit(max_steps=60)
    assert set(fit.hyper) == {"primary", "secondary"}
    assert all({"tau", "eta"} == set(v) for v in fit.hyper.values())
    assert "ML-II" in fit.summary()


@pytest.mark.slow
def test_the_free_velocity_table_threads_the_keplerian(dataset):
    """Fit free velocities, then ask whether a Keplerian still goes through them."""
    keplerian = _dis(dataset).fit(max_steps=150)
    table = keplerian.free_velocities(max_steps=60)
    assert table.mode == "velocity"
    assert table.velocities().shape == (2, dataset.n_epochs)
    residual = table.keplerian_residuals(keplerian)
    assert float(np.sqrt(np.mean(residual**2))) < 1.0
    with pytest.raises(ValueError, match="no orbital elements"):
        table.orbit()


@pytest.mark.slow
def test_sampling_freezes_the_hyperparameters_and_says_so(dataset, truth):
    fit = _dis(dataset).fit(max_steps=150)
    post = fit.sample(seed=0, num_warmup=40, num_samples=40, num_chains=1)
    assert "log_tau" not in post.samples, "the ML-II values are fixed for sampling"
    assert np.isclose(post.star("secondary")["k"], truth["k"][1], atol=0.5)
    assert "plug-in approximation" in post.summary()
    assert "Assumed, not measured" in post.summary()
