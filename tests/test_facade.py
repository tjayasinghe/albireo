"""The `Disentangler` interface: a declaration compiles to the expert path (D46).

The interface derives four quantities that users get wrong. It does not derive the
quantities for which a default would be a scientific claim rather than a convenience.
Both are tested here, in equal detail.

**The velocity budget is derived.** It bounds the largest relative velocity any two
components can reach under the priors, which already contain that information. If the
budget is too small, the sampler stalls at a guard that raises no error.

**The `priors` and `init` dicts cannot disagree**, because a spec defines both. In the
low-level path they are two dicts that a user writes separately and an assert checks.

**A circular orbit is exact, not approximate.** The `(sqrt(e)cos w, sqrt(e)sin w)`
parameterization is singular at `e = 0`: the gradient is NaN, and numpyro reports only
"Cannot find valid initial parameters". A declared circular orbit therefore does not
sample those sites, and a free eccentricity never starts at the origin.

**Light fractions are required.** With constant light fractions the likelihood depends
only on `l_i * d_i`, so the value is an assumption the data cannot contradict.
"""

import importlib
import warnings
from types import SimpleNamespace

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


# -- what it does not derive --------------------------------------------------


def test_light_fractions_are_required_and_must_sum_to_one(dataset):
    """Only l_i * d_i is observable, so the light fraction is an assumption with no default."""
    with pytest.raises(TypeError):
        ab.Star("primary")  # no light=
    with pytest.raises(ValueError, match="sum to 1"):
        _dis(dataset, components=[ab.Star("a", light=0.6), ab.Star("b", light=0.6)])


def test_every_instrument_needs_a_declared_lsf(dataset):
    with pytest.raises(ValueError, match="no LSF declared"):
        _dis(dataset, lsf={"NOT_THE_INSTRUMENT": 6.5})


def test_a_nebular_component_refuses_an_undeclared_wavelength_scale(dataset):
    """Nebular and telluric components are keyed to absolute line positions, and air and
    vacuum wavelengths differ by 83 km/s."""
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
    """Sampling stalls at the guard rather than failing, which is worse."""
    with pytest.raises(ValueError, match="smaller than"):
        assert _dis(dataset, velocity_budget_kms=50.0).velocity_budget


def test_a_period_prior_wide_enough_to_be_a_search_warns(dataset):
    """The scan resolves phase at one period (the prior's midpoint), not a period."""
    with pytest.warns(RuntimeWarning, match="not a period search"):
        _dis(dataset, orbit=_orbit(period=ab.Between(2.0, 12.0))).fit(max_steps=1, k_scan=False)


def test_a_normal_period_prior_does_not_warn(dataset):
    """The quickstart's own prior must not warn, or users learn to ignore the warning."""
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
    """A rotationally broad star needs a tau five orders of magnitude from a sharp one."""
    dis = _dis(
        dataset,
        components=[
            ab.Star("sharp", light=0.62, smoothness=ab.Smoothness(tau0=1.0e3)),
            ab.Star("broad", light=0.38, smoothness=ab.Smoothness(tau0=1.0e8)),
        ],
    )
    assert np.allclose(np.asarray(dis.smoothness_prior.tau), [1.0e3, 1.0e8])


def test_expert_hands_back_the_low_level_path(dataset):
    model, priors, init, fixed = _dis(dataset).expert()
    assert isinstance(model, ab.MarginalOrbitModel)
    assert set(priors) == set(init)
    assert fixed == {}


def test_expert_hands_back_the_sites_held_constant(dataset):
    """A circular orbit fixes secosw and sesinw, and a model rebuilt without them is incomplete."""
    dis = _dis(dataset, orbit=_orbit(ecc=ab.Fixed(0.0)))
    model, priors, init, fixed = dis.expert()
    assert set(fixed) == {"secosw", "sesinw"}
    assert not set(fixed) & (set(priors) | set(init))
    with pytest.raises(ValueError, match="missing orbital sites"):
        model.model(priors)
    assert callable(model.model(priors, fixed=fixed))
    fixed.clear()
    assert set(dis.fixed) == {"secosw", "sesinw"}, "expert() must return copies"


def test_explain_names_every_derivation_and_every_assumption(dataset):
    text = _dis(dataset).explain()
    for expected in ("velocity budget", "model grid", "sampled sites", "Assumed, not measured"):
        assert expected in text, f"explain() should mention {expected!r}"
    assert "primary=0.62" in text


# -- the eccentricity singularity ---------------------------------------------


def test_a_circular_orbit_does_not_sample_the_singular_sites(dataset):
    """At exactly e = 0 the gradient is NaN; with the sites not sampled it is never evaluated."""
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


def test_a_free_eccentricity_may_start_where_a_table_put_it():
    """A start_at on the range and an omega give the start; the default is 0.05 at 0.5 rad."""
    sites = dict(_ecc_sites(ab.Orbit(period=6.0, k=ab.Between(10.0, 90.0)), 0.95))
    e0 = float(sites["secosw"].start()) ** 2 + float(sites["sesinw"].start()) ** 2
    assert e0 == pytest.approx(0.05)
    started = dict(
        _ecc_sites(
            ab.Orbit(
                period=6.0,
                k=ab.Between(10.0, 90.0),
                ecc=ab.Between(0.0, 0.9, start_at=0.6),
                omega=1.2,
            ),
            0.9,
        )
    )
    e1 = float(started["secosw"].start()) ** 2 + float(started["sesinw"].start()) ** 2
    w1 = np.arctan2(float(started["sesinw"].start()), float(started["secosw"].start()))
    assert e1 == pytest.approx(0.6) and w1 == pytest.approx(1.2)
    # A start on the bound is moved inside it, and one at the origin off it.
    edge = dict(
        _ecc_sites(
            ab.Orbit(period=6.0, k=ab.Between(10.0, 90.0), ecc=ab.Between(0.0, 0.5, start_at=0.5)),
            0.9,
        )
    )
    assert float(edge["secosw"].start()) ** 2 + float(edge["sesinw"].start()) ** 2 < 0.5


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


# -- defects found by a review of this module ---------------------------------


def _air(dataset):
    from albireo.preprocess import _replace

    return ab.Dataset(tuple(_replace(e, medium="air") for e in dataset), frame=dataset.frame)


def test_component_order_follows_the_model_not_the_declaration(dataset):
    """The model orders rows stars-telluric-nebular whatever order they were declared in.

    Assembling the smoothness rows in declaration order raises no error: the vectors are
    still the right length, so the only downstream check (a length check) passes, and a
    rotationally broadened star is given a sharp-lined star's curvature penalty.
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
    """Real epochs are near BJD 2.46e6; a prior centred on the origin excludes them all."""
    dis = _dis(dataset)
    prior = dis.priors["t_conj"]
    first = float(np.min(np.asarray(dataset.bjd)))
    assert float(prior.low) <= first <= float(prior.high)


def test_the_model_grid_follows_the_finest_epoch_not_the_median(dataset):
    """A grid coarser than a contributing epoch discards that epoch's resolution."""
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
    """The model supports it; the `Disentangler` interface does not and raises an error."""
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
    """The period is in `fixed`, not `init`, and the scan must read both."""
    dis = _dis(dataset, orbit=_orbit(period=ab.Fixed(6.0)))
    assert "period" in dis.fixed
    fit = dis.fit(max_steps=1, k_scan=False)  # the joint scan is covered by the slow test
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
    """NEBULAR_LINES are air wavelengths; unconverted they are 83 km/s off a vacuum grid."""
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
    """A component confined to the continuum everywhere has no effect."""
    from albireo.preprocess import _replace

    ds = ab.Dataset(tuple(_replace(e, medium="air") for e in dataset), frame=dataset.frame)
    with pytest.raises(ValueError, match="fall outside the model grid"):
        assert _dis(ds, components=[*_stars(), ab.Nebular(v_kms=0.0)]).smoothness_prior


# -- the spec vocabulary ------------------------------------------------------


def test_a_vector_site_may_be_one_spec_or_one_spec_per_star():
    combined = _vector_spec(ab.Between([1.0, 2.0], [3.0, 4.0]), 2, "k")
    split = _vector_spec([ab.Between(1.0, 3.0), ab.Between(2.0, 4.0)], 2, "k")
    assert np.allclose(np.asarray(combined.hi), np.asarray(split.hi))


def test_a_vector_site_keeps_each_entry_starting_value():
    """If start_at is dropped, every component starts at its midpoint without an error.
    Equal starts are the symmetric configuration in which the conjunction scan cannot
    distinguish the declared component assignment from its mirror."""
    spec = _vector_spec([ab.Between(10.0, 90.0, 30.0), ab.Between(10.0, 90.0, 60.0)], 2, "k")
    np.testing.assert_allclose(np.asarray(spec.start()), [30.0, 60.0])
    plain = _vector_spec([ab.Between(10.0, 90.0), ab.Between(10.0, 90.0)], 2, "k")
    np.testing.assert_allclose(np.asarray(plain.start()), [50.0, 50.0])


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
    """The same declaration is used for the scan, so its arguments cannot become inconsistent."""
    dis = ab.Disentangler(
        dataset,
        components=[ab.Star("primary", light=0.97), ab.Star("companion", light=0.03)],
        orbit=ab.Orbit(
            period=ab.Fixed(float(truth["period"])),
            t_conj=ab.Fixed(0.73171),
            ecc=ab.Fixed(0.0),
            # Known, not Fixed: K1 is marginalized, which is the only protection against a
            # wrong K1 inflating the detection statistic rather than blurring it.
            k=[ab.Known(float(truth["k"][0]), 1.5), ab.Scanned(np.arange(20.0, 90.0, 10.0))],
        ),
        lsf={"DEMO": 6.5},
        dv_kms=6.0,
    )
    result = dis.scan()
    peak = float(result.k2_grid[int(np.argmax(result.detection))])
    assert abs(peak - truth["k"][1]) <= 10.0, f"scan peaked at K2 = {peak}, truth {truth['k'][1]}"


# -- the noise model -----------------------------------------------------------


def test_a_declared_noise_correlation_builds_the_correlated_model(dataset):
    """A lag-one correlation per instrument is held as a fixed per-epoch site."""
    dis = _dis(dataset, noise_correlation={"DEMO": 0.3})
    assert dis.model.ar1 is True
    assert np.asarray(dis.fixed["ar1_phi"]).shape == (dataset.n_epochs,)
    assert np.all(np.asarray(dis.fixed["ar1_phi"]) == 0.3)
    assert "AR(1)" in dis.explain() and "noise correlation" in dis.assumptions()
    assert "ar1_phi" not in dis.priors
    fitted = _dis(dataset, noise_correlation=ab.Between(-0.9, 0.9))
    assert fitted.model.ar1 is True
    assert "ar1_phi" in fitted.priors and "ar1_phi" not in fitted.fixed
    assert "fitted" in fitted.explain()
    plain = _dis(dataset)
    assert plain.model.ar1 is False and "ar1_phi" not in plain.fixed
    assert "diagonal" in plain.explain()


def test_a_noise_correlation_is_checked_before_anything_is_built(dataset):
    with pytest.raises(ValueError, match="no entry for instrument"):
        _dis(dataset, noise_correlation={"OTHER": 0.3})
    with pytest.raises(ValueError, match=r"lie in \(-1, 1\)"):
        _dis(dataset, noise_correlation=1.0)
    with pytest.raises(ValueError, match="one value shared"):
        _dis(dataset, noise_correlation=ab.Fixed([0.1, 0.2]))
    with pytest.raises(TypeError, match="not a scan"):
        _dis(dataset, noise_correlation=ab.Scanned(np.array([0.1, 0.2])))


def test_a_zero_noise_correlation_is_the_diagonal_model(dataset):
    """with_ar1 at phi = 0 is the diagonal model to floating-point ordering."""
    plain = _dis(dataset)
    zero = _dis(dataset, noise_correlation=0.0)
    theta = {**plain.init, **plain.fixed}
    theta_zero = {**zero.init, **zero.fixed}
    assert "ar1_phi" in theta_zero
    ll_plain = float(plain.model.log_likelihood(theta))
    ll_zero = float(zero.model.log_likelihood(theta_zero))
    assert np.isclose(ll_zero, ll_plain, rtol=1e-8, atol=1e-6), (ll_zero, ll_plain)


# -- the conjunction-phase scan ---------------------------------------------------


def test_the_conjunction_grid_contains_the_antipode_of_every_trial(dataset):
    """For a near-equal pair the phase likelihood is near-mirror-symmetric under a shift
    of half a period, so the scan can only choose between the two mirrors when both are
    on its grid. An odd trial count leaves every antipode midway between two trials, and
    on one benchmark star the antipode that was off the grid was better by 672 nats (D63).
    """
    scanner = _dis(dataset)._scan_declaration()
    scan = scanner._scan_phase(dict(scanner.init))
    trials, period = scan.trials, scan.period
    assert len(trials) % 2 == 0, "an odd count cannot hold its own antipodes"

    offsets = trials - trials[0]
    antipodes = np.mod(offsets + 0.5 * period, period)
    gap = np.abs(antipodes[:, None] - offsets[None, :]).min(axis=1)
    tol = 64.0 * float(np.spacing(float(np.max(np.abs(trials)))))
    assert gap.max() < tol, (gap.max(), tol)
    # The tolerance must be far below the grid step, or the assertion tests nothing.
    assert tol < 1e-6 * period / len(trials)


# -- the semi-amplitude scan -----------------------------------------------------


def test_the_scan_record_reads_its_own_numbers():
    from albireo.facade import SemiAmplitudeScan

    scan = SemiAmplitudeScan(
        k=np.array([[10.0, 20.0], [20.0, 10.0], [30.0, 60.0]]),
        t_conj=np.array([1.0, 1.0, 1.0]),
        values=np.array([-5.0, -4.0, 3.0]),
        best=np.array([30.0, 60.0]),
        best_t_conj=1.0,
        best_value=3.0,
        start=np.array([10.0, 20.0]),
        start_value=-5.0,
    )
    assert scan.n_trials == 3 and scan.gain == 8.0 and scan.contrast == 8.0
    assert scan.hold_losses == {} and scan.notes == ()


def test_the_scan_is_skipped_when_no_semi_amplitude_is_a_range(dataset):
    dis = _dis(dataset, orbit=_orbit(k=[ab.Known(30.0, 3.0), ab.Known(55.0, 3.0)]))
    assert not dis._has_ranged_k() and dis._scan_semi_amplitudes(dict(dis.init)) is None
    assert _dis(dataset)._has_ranged_k()  # the vector Between([10, 10], [90, 90]) is a range
    from albireo.facade import _k_specs, _range_bounds

    vector = _k_specs(_orbit(k=ab.Between([10.0, 5.0], [90.0, 70.0])), 2)
    assert [_range_bounds(s) for s in vector] == [(10.0, 90.0), (5.0, 70.0)]
    assert _range_bounds(ab.Known(30.0, 3.0)) is None
    with pytest.raises(ValueError, match="k_scan must be"):
        _dis(dataset).fit(max_steps=1, k_scan="always")


@pytest.mark.slow
def test_the_semi_amplitude_scan_finds_the_basin_the_start_missed(dataset, truth):
    """Started at a fifth of the injected values, the scan ends within a grid step of them."""
    starts = [ab.Between(5.0, 120.0, start_at=6.0), ab.Between(5.0, 120.0, start_at=11.0)]
    dis = _dis(dataset, orbit=_orbit(k=starts))
    scanner = dis._scan_declaration()
    assert scanner.grid.dv_kms == pytest.approx(2.0 * dis.grid.dv_kms)
    init = dict(dis.init)
    phase = scanner._scan_phase(init)
    init["t_conj"] = phase.best
    scan = scanner._scan_semi_amplitudes(init)
    assert scan is not None and scan.gain > 0.0 and scan.dv_kms == scanner.grid.dv_kms
    # The sweeps of the scan all have one trial count, so one compiled program serves them.
    assert scanner.model._sweep_jit._cache_size() == 1
    for best, k_true in zip(scan.best, truth["k"], strict=True):
        assert max(best / k_true, k_true / best) < 1.3, (scan.best, truth["k"])
    fit = dis.fit(max_steps=150)
    assert fit.k_scan is not None and fit.k_scan.gain > 0.0
    assert "semi-amplitude scan" in fit.summary()
    for name, k_true in zip(("primary", "secondary"), truth["k"], strict=True):
        assert np.isclose(fit.star(name)["k"], k_true, atol=0.5), name

    # -- the closed loop ----------------------------------------------------------
    # The conjunction window passed to L-BFGS is one period wide and centred on the start
    # the fit used, so the fitted conjunction lies strictly inside it. A start on the
    # window's edge has an infinite unconstrained coordinate (D63).
    window = fit.priors_used["t_conj"]
    t_fit = float(fit.orbit()["t_conj"])
    assert float(window.low) < t_fit < float(window.high)
    assert abs(float(window.high) - float(window.low) - fit.phase_scan.period) < 1e-9


@pytest.mark.slow
def test_the_facade_recovers_the_injected_orbit(dataset, truth):
    """Twelve lines give the same result as the fifty-nine-line expert path."""
    dis = _dis(dataset)
    fit = dis.fit(max_steps=150)

    assert np.isclose(float(fit.orbit()["period"]), truth["period"], atol=1e-3)
    for name, k_true in zip(("primary", "secondary"), truth["k"], strict=True):
        assert np.isclose(fit.star(name)["k"], k_true, atol=0.2), name
    assert np.isclose(float(fit.orbit()["ecc"]), truth["ecc"], atol=0.02)
    # The phase scan is what makes a cold start work.
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
    """Fit free velocities, then check that a Keplerian still goes through them."""
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
    widening = post.plug_in_widening
    assert widening is not None, post.plug_in_note
    assert set(widening) == {"k[0]", "k[1]", "period", "secosw", "sesinw", "t_conj"}
    # Conditioning on the weights cannot widen a Gaussian, and on this example the orbit and
    # the weights are nearly uncorrelated (docs/benchmarks.md, 2026-10-20).
    assert min(widening.values()) >= 1.0 - 1e-9 and max(widening.values()) < 1.01
    assert "freeing the weights would widen" in post.summary()


# -- the continuum weight and the stationarity of the best fit -----------------
#
# The flags of `Fit.summary` are tested on fits assembled from given values, so that each
# case is reached without an optimization. The fits that produce them are in
# tests/test_low_frequency.py.


def _hyper(eta_primary, eta_secondary):
    return {
        "primary": {"tau": 600.0, "eta": eta_primary},
        "secondary": {"tau": 1100.0, "eta": eta_secondary},
    }


def test_a_bound_on_eta_truncates_its_hyperprior_and_moves_the_start(dataset):
    import jax
    import numpyro.distributions as dist
    from numpyro.distributions.transforms import biject_to

    with pytest.raises(ValueError, match="eta_min"):
        ab.Smoothness(eta_min=0.0)
    # Without a bound the hyperprior is the normal it always was.
    assert type(_dis(dataset).priors["log_eta"]) is dist.Normal

    stars = [
        ab.Star("primary", light=0.62, smoothness=ab.Smoothness(eta_min=20.0)),
        ab.Star("secondary", light=0.38),
    ]
    dis = _dis(dataset, components=stars)
    prior = dis.priors["log_eta"]
    start = np.asarray(dis.init["log_eta"])
    assert start[0] > np.log(20.0) and start[1] == pytest.approx(np.log(5.0))
    normal = dist.Normal(np.log(5.0), 3.0)
    value = np.array([np.log(30.0), np.log(0.01)])
    density = np.asarray(prior.log_prob(value))
    assert density[1] == pytest.approx(float(normal.log_prob(value[1])))
    assert density[0] > float(normal.log_prob(value[0]))  # renormalized above the bound
    assert np.isneginf(np.asarray(prior.log_prob(np.array([np.log(10.0), 0.0])))[0])
    # The bounded entry is unconstrained through ln(ln eta - ln eta_min); the other keeps
    # the identity it has under the normal hyperprior.
    transform = biject_to(prior.support)
    z = np.array([0.3, -1.7])
    y = np.asarray(transform(z))
    assert y[0] == pytest.approx(np.log(20.0) + np.exp(0.3)) and y[1] == -1.7
    np.testing.assert_allclose(np.asarray(transform.inv(y)), z)
    draws = np.asarray(prior.sample(jax.random.PRNGKey(0), (2000,)))
    assert draws[:, 0].min() > np.log(20.0)


def test_a_fallen_continuum_weight_is_flagged(dataset):
    dis = _dis(dataset)
    report = ab.Fit(dis=dis, result=None, hyper=_hyper(0.0146, 0.00135))._hyper_report()
    assert report.count("eta fell more than one hyperprior standard deviation") == 2
    assert "Fit.renormalize_epochs" in report and "Smoothness(eta_min=...)" in report
    drift = (np.log(0.00135) - np.log(5.0)) / 3.0
    assert f"({drift:+.2f} sigma)" in report
    # 1.07 is half a hyperprior standard deviation below the centre: not flagged.
    assert "fell" not in ab.Fit(dis=dis, result=None, hyper=_hyper(1.07, 8.3))._hyper_report()
    # With one star the data determine its broad structure, and nothing is flagged.
    single = _dis(
        dataset, components=[ab.Star("only", light=1.0)], orbit=_orbit(k=ab.Between(10.0, 90.0))
    )
    hyper = {"only": {"tau": 300.0, "eta": 0.001}}
    assert "fell" not in ab.Fit(dis=single, result=None, hyper=hyper)._hyper_report()


def test_an_eta_held_at_its_bound_is_reported(dataset):
    bounded = ab.Smoothness(eta_min=5.0)
    stars = [
        ab.Star("primary", light=0.62, smoothness=bounded),
        ab.Star("secondary", light=0.38, smoothness=bounded),
    ]
    dis = _dis(dataset, components=stars)
    report = ab.Fit(dis=dis, result=None, hyper=_hyper(5.02, 9.0))._hyper_report()
    assert "'primary' eta is at its declared bound 5" in report
    assert "'secondary' eta is at its declared bound" not in report
    assert "bound 5)" in report and "fell" not in report


def _injected_t_conj(truth):
    return float(
        ab.t_conj_from_t_peri(
            truth["t_peri"], period=truth["period"], ecc=truth["ecc"], omega=truth["omega"]
        )
    )


def _fit_at_injected(dis, truth, trace, *, converged=False, drop=(), t_conj=None, **fields):
    """A fit at the injected orbit whose optimizer record is the given potential trace.

    ``drop`` removes sites from the optimizer's parameters (a site the declaration fixes is
    not among them), and ``fields`` are passed to :class:`Fit`.
    """
    import jax.numpy as jnp

    from albireo.inference import MAPResult

    t_conj = _injected_t_conj(truth) if t_conj is None else t_conj
    root = np.sqrt(truth["ecc"])
    params = {
        "period": jnp.asarray(truth["period"]),
        "t_conj": jnp.asarray(t_conj),
        "secosw": jnp.asarray(root * np.cos(truth["omega"])),
        "sesinw": jnp.asarray(root * np.sin(truth["omega"])),
        "k": jnp.asarray(truth["k"]),
        "log_tau": jnp.log(jnp.array([600.0, 1100.0])),
        "log_eta": jnp.log(jnp.array([10.0, 3.0])),
    }
    params = {site: value for site, value in params.items() if site not in drop}
    trace = tuple(float(v) for v in trace)
    result = MAPResult(
        params=params,
        unconstrained={},
        potential=trace[-1],
        grad_norm=50.0,
        converged=converged,
        num_steps=len(trace),
        potential_trace=trace,
    )
    return ab.Fit(dis=dis, result=result, hyper=_hyper(10.0, 3.0), **fields)


@pytest.mark.slow
def test_the_laplace_check_places_the_orbit_at_its_maximum_or_away_from_it(dataset):
    from dataclasses import replace

    import jax.numpy as jnp

    fit = _dis(dataset).fit(max_steps=150)
    check = fit.laplace_check()
    assert check.valid, check.problems()
    assert check.labels == ("k[0]", "k[1]", "period", "secosw", "sesinw", "t_conj")
    # Half a km/s in K(primary) is about ten standard deviations.
    params = dict(fit.result.params)
    params["k"] = jnp.asarray(params["k"]) + jnp.array([0.5, 0.0])
    moved = replace(fit, result=replace(fit.result, params=params))
    away = moved.laplace_check()
    assert not away.valid and not away.stationary and away.step_sigma[0] > 3.0
    assert away.drop > 10.0 and check.quadratic
    with pytest.warns(RuntimeWarning, match="not the maximum"):
        moved.sample(num_warmup=2, num_samples=2, num_chains=1)


def test_the_summary_reports_how_far_the_potential_fell_at_the_step_cap(dataset, truth):
    dis = _dis(dataset)
    falling = _fit_at_injected(dis, truth, np.linspace(100.0, 40.0, 30)).summary()
    assert "! the potential fell by 20.7 nats over the last 10 steps" in falling
    assert "laplace_check()" in falling
    assert "30 steps (stopped at the step cap)" in falling
    flat = _fit_at_injected(dis, truth, [100.0] * 20 + [99.999] * 10).summary()
    assert "the potential changed by 0.001 nats over the last 10 steps" in flat
    assert "! the potential" not in flat
    done = _fit_at_injected(dis, truth, np.linspace(100.0, 40.0, 30), converged=True).summary()
    assert "the potential" not in done
    # A fit stopped by its progress callback before the cap is not said to be at the cap.
    trace = np.linspace(100.0, 40.0, 30)
    capped = _fit_at_injected(dis, truth, trace, max_steps=30).summary()
    assert "30 steps (stopped at the step cap)" in capped
    stopped = _fit_at_injected(dis, truth, trace, max_steps=300).summary()
    assert "30 steps (stopped by the progress callback)" in stopped


# -- the posterior report -----------------------------------------------------
#
# The report is tested on stand-in draws, so that its intervals and diagnostics are
# checked against known distributions without a sampling run.


class _Draws:
    """A stand-in for numpyro's ``MCMC``: grouped draws and the divergence flags."""

    def __init__(self, grouped, diverging=None):
        self.grouped = {name: np.asarray(values) for name, values in grouped.items()}
        self.diverging = diverging

    def get_samples(self, group_by_chain=False):
        if group_by_chain:
            return self.grouped
        return {name: v.reshape(-1, *v.shape[2:]) for name, v in self.grouped.items()}

    def get_extra_fields(self):
        return {} if self.diverging is None else {"diverging": self.diverging}


def _orbit_draws(rng, *, n_chains=2, n_draws=200, omega=0.7, ecc_sampled=True):
    """Independent draws around the sb2_sim orbit, with the deterministic sites recorded."""
    shape = (n_chains, n_draws)
    draws = {
        "period": 6.0 + 1e-4 * rng.normal(size=shape),
        "t_conj": 0.73 + 1e-3 * rng.normal(size=shape),
        "k": np.stack(
            [42.0 + 0.05 * rng.normal(size=shape), 63.0 + 0.08 * rng.normal(size=shape)],
            axis=-1,
        ),
    }
    if ecc_sampled:
        root_e = np.sqrt(0.15) + 0.002 * rng.normal(size=shape)
        angle = omega + 0.01 * rng.normal(size=shape)
        draws["secosw"], draws["sesinw"] = root_e * np.cos(angle), root_e * np.sin(angle)
        draws["ecc"] = draws["secosw"] ** 2 + draws["sesinw"] ** 2
        draws["omega"] = np.arctan2(draws["sesinw"], draws["secosw"])
    else:
        draws["ecc"], draws["omega"] = np.zeros(shape), np.zeros(shape)
    return draws


def _row(text, label):
    """The numbers of one row of the posterior report, after its label."""
    line = next(line for line in text.splitlines() if line.strip().startswith(label))
    return line.strip()[len(label) :].split()


def test_k_hdi_is_the_narrowest_interval_holding_95_percent_of_the_draws(dataset):
    """On skewed draws the highest-density interval is narrower than the percentile one."""
    rng = np.random.default_rng(0)
    shape = (2, 2000)
    k = np.stack([40.0 + rng.lognormal(sigma=0.8, size=shape), 60.0 + rng.normal(size=shape)], -1)
    post = ab.Posterior(dis=_dis(dataset), mcmc=_Draws({"k": k}), map=None)
    low, high = post.star("primary")["k_hdi"]
    assert isinstance(low, float) and isinstance(high, float)
    draws = k[..., 0].ravel()
    assert np.mean((draws >= low) & (draws <= high)) >= 0.949
    p_low, p_high = np.percentile(draws, [2.5, 97.5])
    assert high - low < 0.9 * (p_high - p_low)


def test_the_posterior_report_gives_intervals_and_convergence_diagnostics(dataset):
    rng = np.random.default_rng(1)
    draws = {**_orbit_draws(rng), "ar1_phi": 0.3 + 0.01 * rng.normal(size=(2, 200))}
    diverging = np.zeros((2, 200), dtype=bool)
    text = ab.Posterior(dis=_dis(dataset), mcmc=_Draws(draws, diverging), map=None).summary()
    # ecc and omega are derived, not sampled: 1 + 1 + 2 + 2 + 1 parameters.
    assert "2 chain(s) x 200 draws of 7 parameters" in text
    assert "divergences: 0" in text
    for label in ("95% HDI", "R-hat", "ESS", "period [d]", "t_conj [d]", "omega [rad]"):
        assert label in text
    for label, centre in (("K(primary) [km/s]", 42.0), ("K(secondary) [km/s]", 63.0)):
        mean, _sd, low, high, rhat, ess = _row(text, label)
        assert abs(float(mean) - centre) < 0.05
        assert float(low.strip("[,")) < centre < float(high.strip("]"))
        assert abs(float(rhat) - 1.0) < 0.05 and float(ess) > 100.0
    assert abs(float(_row(text, "ecc")[0]) - 0.15) < 0.01
    assert "other sampled sites (ar1_phi): largest R-hat" in text
    assert "nan" not in text
    assert "plug-in approximation" in text and "Assumed, not measured" in text


def test_omega_is_summarized_about_its_circular_mean(dataset):
    """Draws of omega that straddle +-pi are one cluster, not two a full turn apart."""
    draws = _orbit_draws(np.random.default_rng(2), omega=np.pi)
    text = ab.Posterior(dis=_dis(dataset), mcmc=_Draws(draws), map=None).summary()
    mean, sd, *_ = _row(text, "omega [rad]")
    assert abs(abs(float(mean)) - np.pi) < 0.01
    assert float(sd) < 0.05


def test_a_held_eccentricity_is_reported_as_held_without_warnings(dataset):
    draws = _orbit_draws(np.random.default_rng(3), ecc_sampled=False)
    post = ab.Posterior(dis=_dis(dataset), mcmc=_Draws(draws), map=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = post.summary()
    assert "draws of 4 parameters (period, t_conj, k)" in text
    assert "held at 0" in text and "nan" not in text


def test_too_few_draws_for_the_diagnostics_are_reported_as_undefined(dataset):
    """Split R-hat needs four draws per chain; fewer must not raise."""
    draws = _orbit_draws(np.random.default_rng(4), n_chains=1, n_draws=3)
    text = ab.Posterior(dis=_dis(dataset), mcmc=_Draws(draws), map=None).summary()
    assert _row(text, "K(primary) [km/s]")[-2:] == ["n/a", "n/a"]


def test_the_report_states_how_much_freeing_the_weights_would_widen_it(dataset):
    """The largest widening is printed with its coordinate, or why it was not computed."""
    draws = _orbit_draws(np.random.default_rng(5))
    widening = {"k[0]": 1.00009, "k[1]": 1.031, "period": 1.00003, "t_conj": 1.0001}
    text = ab.Posterior(
        dis=_dis(dataset), mcmc=_Draws(draws), map=None, plug_in_widening=widening
    ).summary()
    assert "freeing the weights would widen the standard deviations of the sampled parameters" in (
        " ".join(text.split())
    )
    assert "by at most 3.1% (K(secondary))." in " ".join(text.split())
    noted = ab.Posterior(
        dis=_dis(dataset),
        mcmc=_Draws(draws),
        map=None,
        plug_in_note="the Hessian with the weights free is not positive definite",
    ).summary()
    assert (
        "The widening from freeing the weights was not computed: the Hessian with the weights "
        "free is not positive definite." in " ".join(noted.split())
    )
    bare = ab.Posterior(dis=_dis(dataset), mcmc=_Draws(draws), map=None).summary()
    assert "freeing the weights" not in bare


def test_widening_percentages_keep_two_significant_digits():
    from albireo.facade import _percent

    assert _percent(0.000110) == "0.011%"
    assert _percent(0.031) == "3.1%"
    assert _percent(0.5) == "50%"
    assert _percent(1.2) == "120%"
    assert _percent(-1e-12) == "less than 0.001%"


def test_no_widening_comes_from_a_held_hessian_that_is_not_positive_definite(dataset, truth):
    """The free Hessian is not computed when the held one already fails."""
    from albireo.inference import LaplaceCheck

    fit = _fit_at_injected(_dis(dataset), truth, [100.0] * 5)
    n = 6
    held = LaplaceCheck(
        covariance=np.eye(n),
        labels=("k[0]", "k[1]", "period", "secosw", "sesinw", "t_conj"),
        gradient=np.zeros(n),
        newton_step=np.zeros(n),
        step_sigma=np.zeros(n),
        eigenvalues=np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        eigenvectors=np.eye(n),
        floor=1e-10,
    )
    factors, note = fit._plug_in_widening(held)
    assert factors is None
    assert note == "the Hessian with the weights held is not positive definite"


# -- declaring measured velocities instead of an orbit ------------------------
#
# In an unsolved system the free table is what produces the period, so reaching the table
# cannot require a period. These tests cover the declaration, the budget it derives from
# a source with no `k` priors in it, and the cases that raise an error.


def _measured(truth, *, scatter=0.0, systemic=0.0, seed=3):
    """Velocities as an external pipeline would report them: noisy, with an arbitrary zero point."""
    v = np.asarray(truth["velocities"], dtype=float)
    if scatter:
        v = v + np.random.default_rng(seed).normal(0.0, scatter, v.shape)
    return v + systemic


def test_exactly_one_of_orbit_and_velocities(dataset, truth):
    with pytest.raises(ValueError, match="exactly one of orbit"):
        ab.Disentangler(dataset, components=_stars(), lsf={"DEMO": 6.5})
    with pytest.raises(ValueError, match="exactly one of orbit"):
        _dis(dataset, velocities=_measured(truth))


def test_declared_velocities_are_checked_for_shape_and_finiteness(dataset, truth):
    with pytest.raises(ValueError, match=r"velocities must have shape \(2, 12\)"):
        _dis(dataset, orbit=None, velocities=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="must all be finite"):
        _dis(dataset, orbit=None, velocities=np.full((2, dataset.n_epochs), np.nan))


def test_a_cold_declaration_is_refused_rather_than_discovered(dataset):
    """Equal velocities at every epoch are the cold start, which D42 measured to fail."""
    with pytest.raises(ValueError, match="never separate the components"):
        _dis(dataset, orbit=None, velocities=np.zeros((2, dataset.n_epochs)))


def test_velocities_inside_the_lsf_width_warn(dataset):
    """A warm start in the unresolved regime is close to the cold one."""
    n = dataset.n_epochs
    barely = np.stack([np.zeros(n), np.linspace(0.0, 1.0, n)])
    with pytest.warns(RuntimeWarning, match="below the widest LSF sigma"):
        _dis(dataset, orbit=None, velocities=barely)


def test_the_budget_comes_from_the_velocities_when_there_is_no_orbit(dataset, truth):
    """There are no `k` priors, so the bound is the declared table, centred and with headroom."""
    v = _measured(truth, systemic=150.0)
    dis = _dis(dataset, orbit=None, velocities=v)
    centred = v - v.mean(axis=1, keepdims=True)
    reach = float(np.sum(np.max(np.abs(centred), axis=1)))

    names = [name for name, _ in dis.velocity_budget.terms]
    assert any("declared per-star" in name for name in names)
    assert not any("|K|" in name for name in names)
    assert dis.velocity_budget.terms[0][1] == pytest.approx(reach)
    assert dis.velocity_budget.total > reach

    # The systemic offset is unidentified and removed before it enters the model, so it
    # must not increase the bandwidth: +150 km/s on every velocity changes nothing.
    plain = _dis(dataset, orbit=None, velocities=_measured(truth))
    assert dis.velocity_budget.total == pytest.approx(plain.velocity_budget.total)


def test_a_velocity_declaration_samples_no_orbital_sites(dataset, truth):
    dis = _dis(dataset, orbit=None, velocities=_measured(truth))
    assert set(dis.priors) == {"velocity", "log_tau", "log_eta"}
    assert dis.fixed == {}
    assert np.allclose(np.asarray(dis.init["velocity"]), _measured(truth))
    assert "velocity" in dis.explain()
    assert "declared velocities" in dis.assumptions()


def test_a_velocity_declaration_still_carries_the_nebular_amplitudes(dataset, truth):
    """Dropping them would silently hold the component static at amplitude 1."""
    epochs = [
        ab.EpochData(
            wave=e.wave,
            flux=e.flux,
            ivar=e.ivar,
            bjd=e.bjd,
            v_bary=e.v_bary,
            instrument=e.instrument,
            medium="air",
        )
        for e in dataset
    ]
    declared = ab.Dataset(tuple(epochs), frame=dataset.frame)
    dis = ab.Disentangler(
        declared,
        components=[*_stars(), ab.Nebular(v_kms=0.0)],
        velocities=_measured(truth),
        lsf={"DEMO": 6.5},
    )
    assert "log_nebular_amp" in dis.priors
    assert dis.priors["log_tau"].batch_shape[-1] == 3


def test_a_scan_needs_an_orbit_and_says_which(dataset, truth):
    dis = _dis(dataset, orbit=None, velocities=_measured(truth))
    with pytest.raises(ValueError, match="needs orbit="):
        dis.scan()
    with pytest.raises(ValueError, match="needs orbit="):
        dis.detection_limit()


@pytest.mark.slow
def test_measured_velocities_warm_start_the_table_as_well_as_a_keplerian(dataset, truth):
    """No period is needed, and the result is not worse without one.

    The declared velocities have 3 km/s of scatter and a 150 km/s systemic offset that the
    fit does not identify, which is what an external pipeline provides. The recovered table
    must still match the injected one, and its zero point must not enter the result.
    """
    dis = _dis(dataset, orbit=None, velocities=_measured(truth, scatter=3.0, systemic=150.0))
    fit = dis.fit(max_steps=120)
    assert fit.mode == "velocity"

    got = fit.velocities()
    want = np.asarray(ab.relative_velocities(np.asarray(truth["velocities"]), dis.grid))
    assert got.shape == want.shape
    rms = np.sqrt(np.mean((got - want) ** 2, axis=1))
    assert np.all(rms < 1.0), f"per-epoch RV rms {rms} km/s"
    # The systemic offset must not enter the result: the spans are what is identified.
    assert np.allclose(np.ptp(got, axis=1), np.ptp(want, axis=1), rtol=0.05)
    assert fit.velocity_errors().shape == got.shape
    with pytest.raises(ValueError, match="no orbital elements"):
        fit.orbit()


# -- the correlation search window --------------------------------------------
#
# `measure_velocities` passes `todcor` a search window derived from the fitted velocities.
# `todcor` searches each template's own rest frame and reports the shift composed with
# that template's zero point, so one shared window searches a different interval of
# reported velocity for every component whose zero point differs. The window rule is
# tested here without running either the optimizer or the correlation, because the
# rule needs only the fitted velocities and the templates' zero points.


def _velocity_mode_fit(dataset, velocities):
    """A `Fit` in velocity mode whose parameters are the declared table."""
    dis = _dis(dataset, orbit=None, velocities=velocities)
    result = SimpleNamespace(params={"velocity": np.asarray(velocities)})
    return ab.Fit(dis=dis, result=result, hyper={}, mode="velocity")


def _flat_templates(dis, zero_points):
    """One template per star at the given zero points; the deviation is never used."""
    from albireo.todcor import Template

    return [
        Template(star.name, dis.grid, np.zeros(dis.grid.n), v_zero_kms=zero)
        for star, zero in zip(dis.stars, zero_points, strict=True)
    ]


@pytest.fixture
def captured_todcor(monkeypatch):
    """Capture what `measure_velocities` passes to `todcor` instead of correlating."""
    # `albireo.todcor` names the function on the package, so the module is fetched by name.
    todcor_module = importlib.import_module("albireo.todcor")
    seen = {}

    def spy(dataset, templates, **kwargs):
        seen.clear()
        seen.update(kwargs)
        seen["templates"] = templates
        return "table"

    monkeypatch.setattr(todcor_module, "todcor", spy)
    return seen


def test_the_search_window_follows_each_template_zero_point(dataset, truth, captured_todcor):
    """Windows offset by the zero points cover one common interval of reported velocity."""
    fit = _velocity_mode_fit(dataset, _measured(truth))
    assert fit.measure_velocities(templates=_flat_templates(fit.dis, (10.0, -50.0))) == "table"
    ranges = np.asarray(captured_todcor["v_range"], dtype=float)
    assert ranges.shape == (2, 2)
    np.testing.assert_allclose(ranges[1] - ranges[0], 60.0)  # the zero points, 60 km/s apart

    fitted = np.asarray(fit.velocities())
    common = np.array([fitted.min() - 40.0, fitted.max() + 40.0])
    np.testing.assert_allclose(ranges[0], common - 30.0)  # zero point above the median
    np.testing.assert_allclose(ranges[1], common + 30.0)
    # Templates with no zero point are all in the fit's own frame: one window.
    fit.measure_velocities(templates=_flat_templates(fit.dis, (None, None)))
    np.testing.assert_allclose(np.asarray(captured_todcor["v_range"]), [common, common])


def test_zero_points_that_move_a_window_off_its_own_velocities_are_refused(
    dataset, truth, captured_todcor
):
    """The failure guarded against is a table of velocities at the edge of the search.

    A label fit whose frame-offset scan stopped at its bound returns a zero point that
    differs from the other component's by more than the fitted velocities span. No
    interval of reported velocity then contains both components, and a search of one
    measures nothing while returning a full table.
    """
    fit = _velocity_mode_fit(dataset, _measured(truth))
    templates = _flat_templates(fit.dis, (-150.0, 25.0))
    with pytest.raises(ValueError, match="component 'primary'") as raised:
        fit.measure_velocities(templates=templates)
    message = str(raised.value)
    assert "zero points disagree" in message
    assert "v_range=" in message and "v_zero_kms=None" in message
    assert not captured_todcor  # nothing was searched
    # A declared window is used as given, whatever the zero points are.
    fit.measure_velocities(templates=templates, v_range=(-300.0, 300.0))
    assert captured_todcor["v_range"] == (-300.0, 300.0)


# -- defects found by the 2026-10-20 review ------------------------------------
#
# Each test below failed on the code before the review. The fits are stubbed where only
# the bookkeeping around the scans is under test, so that nothing is compiled.


class _ReachedRunMap(Exception):
    """Raised in place of the optimizer by the stubbed fits below."""


def _stub_the_scans(monkeypatch, *, gains, profile):
    """Replace the scans and the optimizer of `fit` with records of how they were called.

    The n-th semi-amplitude scan gains ``gains[n]`` nats, and a scan that gains moves every
    semi-amplitude by a factor of 1.5.
    """
    facade = importlib.import_module("albireo.facade")
    calls = {"phase": [], "amplitude": [], "init": None}

    def phase(self, init):
        calls["phase"].append(np.array(init["k"], dtype=float))
        first = float(np.min(np.asarray(self.dataset.bjd)))
        return facade.PhaseScan(
            period=6.0, trials=np.array([first]), values=np.array([0.0]), best=first + 1.0
        )

    def amplitude(self, init):
        start = np.array(init["k"], dtype=float)
        gain = gains[len(calls["amplitude"])]
        calls["amplitude"].append(start)
        best = 1.5 * start if gain > 0.0 else start
        first = float(np.min(np.asarray(self.dataset.bjd)))
        return facade.SemiAmplitudeScan(
            k=best[None, :],
            t_conj=np.array([first + 2.0]),
            values=np.array([gain]),
            best=best,
            best_t_conj=first + 2.0,
            best_value=gain,
            start=start,
            start_value=0.0,
        )

    def run_map(model, *, init, **kwargs):
        calls["init"] = {key: np.asarray(value) for key, value in init.items()}
        raise _ReachedRunMap

    monkeypatch.setattr(facade.Disentangler, "_scan_phase", phase)
    monkeypatch.setattr(facade.Disentangler, "_scan_semi_amplitudes", amplitude)
    monkeypatch.setattr(
        facade.Disentangler, "_profile_prior_amplitudes", lambda self, init, **kw: dict(profile)
    )
    monkeypatch.setattr(facade, "run_map", run_map)
    return calls


def test_an_adapted_prior_amplitude_moves_only_the_star_rows(dataset, monkeypatch):
    """The adaptation reached a telluric row by name, and a Telluric has no name."""
    calls = _stub_the_scans(monkeypatch, gains=[40.0, 40.0], profile={0: 1.0, 1: 4.0})
    dis = _dis(_air(dataset), components=[*_stars(), ab.Telluric()])
    with pytest.raises(_ReachedRunMap):
        dis.fit(max_steps=1)
    shift = calls["init"]["log_tau"] - np.asarray(dis.init["log_tau"])
    np.testing.assert_allclose(shift, [0.0, -2.0 * np.log(4.0), 0.0])
    shift = calls["init"]["log_eta"] - np.asarray(dis.init["log_eta"])
    np.testing.assert_allclose(shift, [0.0, -2.0 * np.log(4.0), 0.0])


def test_the_phase_is_rescanned_when_only_the_first_scan_pass_moved(dataset, monkeypatch):
    """The second pass is measured from the first pass's best, so its gain alone does not
    say whether the start moved. With the first pass moving the semi-amplitudes and the
    adapted pass holding them, the phase must still be rescanned at the new values."""
    calls = _stub_the_scans(monkeypatch, gains=[40.0, 0.0], profile={0: 1.0, 1: 4.0})
    dis = _dis(dataset)
    declared = np.array(dis.init["k"], dtype=float)
    with pytest.raises(_ReachedRunMap):
        dis.fit(max_steps=1)
    assert len(calls["amplitude"]) == 2
    assert len(calls["phase"]) == 2, "the phase scan at the chosen semi-amplitudes did not run"
    np.testing.assert_allclose(calls["phase"][0], declared)
    np.testing.assert_allclose(calls["phase"][1], 1.5 * declared)
    np.testing.assert_allclose(calls["init"]["k"], 1.5 * declared)


def test_a_constant_offset_between_the_stars_is_still_a_cold_start(dataset):
    """The model removes one zero point per star, so per-star constants separate nothing."""
    n = dataset.n_epochs
    with pytest.raises(ValueError, match="never separate the components"):
        _dis(dataset, orbit=None, velocities=np.stack([np.full(n, 20.0), np.full(n, -20.0)]))
    # An unresolved pattern is not hidden by large per-star offsets.
    barely = np.stack([np.zeros(n) + 40.0, np.linspace(0.0, 1.0, n) - 40.0])
    with pytest.warns(RuntimeWarning, match="below the widest LSF sigma"):
        _dis(dataset, orbit=None, velocities=barely)


def test_a_single_valued_semi_amplitude_for_two_stars_is_refused(dataset):
    """A scalar spec built a scalar k site, which failed at the first likelihood evaluation."""
    with pytest.raises(ValueError, match="one entry per star"):
        _dis(dataset, orbit=_orbit(k=ab.Between(10.0, 90.0)))
    with pytest.raises(ValueError, match="one entry per star"):
        _dis(dataset, orbit=_orbit(k=40.0))
    # A bare number is Fixed, as everywhere else, and is valid for a single star.
    single = _dis(dataset, components=[ab.Star("only", light=1.0)], orbit=_orbit(k=40.0))
    assert float(np.asarray(single.fixed["k"])) == 40.0 and "k" not in single.priors
    assert single.velocity_budget.terms[0][1] == pytest.approx(40.0)


def test_one_vector_range_is_scanned_per_star_whichever_field_is_the_vector():
    """Between([10, 5], 90) has two entries although its upper bound is a scalar."""
    from albireo.facade import _k_specs, _range_bounds

    specs = _k_specs(_orbit(k=ab.Between([10.0, 5.0], 90.0)), 2)
    assert [_range_bounds(s) for s in specs] == [(10.0, 90.0), (5.0, 90.0)]
    assert sum(s.upper() for s in specs) == pytest.approx(180.0)


def _k2_declaration(dataset, **kwargs):
    orbit = ab.Orbit(
        period=ab.Fixed(6.0),
        t_conj=ab.Fixed(0.5),
        ecc=ab.Fixed(0.0),
        k=[ab.Fixed(40.0), ab.Scanned(np.arange(20.0, 60.0, 10.0))],
    )
    options = {
        "components": [ab.Star("a", light=0.97), ab.Star("b", light=0.03)],
        "orbit": orbit,
        "lsf": {"DEMO": 6.5},
    }
    return ab.Disentangler(dataset, **{**options, **kwargs})


def test_a_scan_refuses_a_correlated_noise_model_it_cannot_evaluate(dataset):
    """k2_scan has no AR(1) noise model; the declared correlation was dropped silently."""
    dis = _k2_declaration(dataset, noise_correlation={"DEMO": 0.3})
    with pytest.raises(ValueError, match="no AR\\(1\\) noise model"):
        dis.scan()
    with pytest.raises(ValueError, match="no AR\\(1\\) noise model"):
        dis.detection_limit(k2_true=40.0, ell2_grid=[0.1])
    fitted = _k2_declaration(dataset, noise_correlation=ab.Between(-0.9, 0.9))
    with pytest.raises(ValueError, match="no AR\\(1\\) noise model"):
        fitted.scan()
    # A zero correlation is the diagonal model, which the scan does evaluate.
    _k2_declaration(dataset, noise_correlation=0.0)._refuse_correlated_noise("scan")


def test_the_callers_block_size_reaches_the_calibration(dataset, monkeypatch):
    """The declaration's block size was inserted before the caller's, which then collided."""
    calibrate = importlib.import_module("albireo.calibrate")
    seen = {}

    def spy(grid, dataset, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return "limit"

    monkeypatch.setattr(calibrate, "detection_limit", spy)
    dis = _k2_declaration(dataset, block_size=64)
    assert dis.detection_limit(block_size=32, k2_true=40.0, ell2_grid=[0.1]) == "limit"
    assert seen["block_size"] == 32
    dis.detection_limit(k2_true=40.0, ell2_grid=[0.1])
    assert seen["block_size"] == 64
    assert seen["k1"] == 40.0 and seen["k1_sigma"] is None
    with pytest.raises(TypeError, match="from the declaration"):
        dis.detection_limit(k1=41.0, k2_true=40.0, ell2_grid=[0.1])


def test_a_velocity_declaration_without_an_lsf_names_the_instrument(dataset, truth):
    """The velocity check read the LSF before the missing-LSF check had run."""
    with pytest.raises(ValueError, match="no LSF declared"):
        _dis(dataset, orbit=None, velocities=_measured(truth), lsf={"OTHER": 6.5})


def test_anchors_given_as_an_array_are_refused_by_the_scan(dataset):
    """The truth value of an array of anchors is ambiguous, which raised the wrong error."""
    lsf = ab.LSF([6.0, 7.0], anchors_angstrom=np.array([4500.0, 4560.0]))
    dis = _k2_declaration(dataset, lsf={"DEMO": lsf})
    with pytest.raises(ValueError, match="one line-spread width per instrument"):
        dis.scan()


def test_a_fixed_noise_correlation_is_reported_as_held(dataset):
    """Fixed(phi) holds the site, as a bare number does, and is an assumption."""
    dis = _dis(dataset, noise_correlation=ab.Fixed(0.3))
    assert float(np.asarray(dis.fixed["ar1_phi"])) == pytest.approx(0.3)
    assert "ar1_phi" not in dis.priors
    np.testing.assert_allclose(dis.noise_correlation_per_epoch(), 0.3)
    assert "(declared)" in dis._noise_line() and "fitted" not in dis._noise_line()
    assert "noise correlation  DEMO=0.300" in dis.assumptions()
    fitted = _dis(dataset, noise_correlation=ab.Between(-0.9, 0.9))
    assert fitted.noise_correlation_per_epoch() is None and "fitted" in fitted._noise_line()


def test_declared_velocities_are_copied(dataset, truth):
    """The budget, grid and model are built once, from the table as it was declared."""
    v = _measured(truth)
    dis = _dis(dataset, orbit=None, velocities=v)
    budget = dis.velocity_budget.total
    v[:] = 0.0
    assert np.ptp(dis.velocities) > 0.0
    assert dis.velocity_budget.total == budget
    with pytest.raises(ValueError, match="read-only"):
        dis.velocities[0, 0] = 1.0


def test_smoothness_starts_must_be_positive(dataset):
    """ln(0) is -inf and a negative width is not a normal distribution."""
    for bad in ({"tau0": 0.0}, {"eta0": -1.0}, {"sigma": 0.0}, {"tau0": float("nan")}):
        with pytest.raises(ValueError, match="finite positive"):
            ab.Smoothness(**bad)
    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        _dis(dataset, noise_correlation=ab.Between(-1.5, 0.9))
    assert "ar1_phi" in _dis(dataset, noise_correlation=ab.Between(-1.0, 1.0)).priors


# -- the compiled programs of the scans ------------------------------------------------


def test_the_scan_batch_is_sixteen_unless_a_batch_would_exceed_its_memory_budget(dataset):
    """52 bytes per band entry and component, measured from XLA's buffer assignment."""
    from albireo.facade import Disentangler

    assert _dis(dataset)._scan_declaration()._scan_batch() == 16

    def batch(n_components, n_pixels, half_bandwidth):
        problem = SimpleNamespace(n_components=n_components, grid=SimpleNamespace(n=n_pixels))
        model = SimpleNamespace(problem=problem, half_bandwidth=half_bandwidth)
        return Disentangler._scan_batch(SimpleNamespace(model=model))

    # A coarse echelle grid of 4,309 pixels at half-bandwidth 156 needs 140 MB per trial,
    # so 7 trials fit in 1 GiB; the batch is a divisor of 16, so that the padding of a
    # sweep's last chunk never exceeds what a batch of 16 evaluates.
    assert batch(2, 4309, 156) == 4
    assert batch(2, 2509, 156) == 8
    assert batch(3, 4309, 156) == 2
    assert batch(2, 1000, 50) == 16
    assert batch(2, 200_000, 400) == 1


def test_every_sweep_of_the_scan_has_one_size(dataset, monkeypatch):
    """A sweep is compiled per trial count; chunks of one size share one program."""
    dis = _dis(dataset)
    scanner = dis._scan_declaration()
    model = scanner.model
    sizes = []

    def sweep(theta, trials, *, batch_size=None, problem=None):
        k = np.asarray(trials["k"], dtype=float)
        sizes.append((k.shape[0], batch_size))
        # A value that identifies each trial, so the reassembly can be checked.
        return -np.abs(k[:, 0] - 40.0) - np.abs(k[:, 1] - 60.0) - np.asarray(trials["t_conj"])

    def single(theta):
        k = np.asarray(theta["k"], dtype=float)
        return -abs(k[0] - 40.0) - abs(k[1] - 60.0) - float(np.asarray(theta["t_conj"]))

    monkeypatch.setattr(model, "log_likelihood_sweep", sweep)
    monkeypatch.setattr(model, "log_likelihood", single)
    scan = scanner._scan_semi_amplitudes(dict(dis.init))
    assert scan is not None and len(sizes) > 1
    assert {size for size, _ in sizes} == {16} and {b for _, b in sizes} == {16}
    # Each recorded value is that of its own trial: the padding was trimmed in order.
    expected = -np.abs(scan.k[:, 0] - 40.0) - np.abs(scan.k[:, 1] - 60.0) - np.asarray(scan.t_conj)
    np.testing.assert_array_equal(scan.values, expected)


def test_the_amplitude_profile_keeps_the_sites_of_the_scans(dataset, monkeypatch):
    """Scaling the light added a site, and the marginal was compiled again for it. The
    invariance L(c l, tau, eta) = L(l, tau / c^2, eta / c^2) gives the same profile with
    the hyperparameters of the star moved instead."""
    dis = _dis(dataset)
    scanner = dis._scan_declaration()
    seen = []
    target = {0: 0.25, 1: 2.0}  # the profile maximum of each star, in c

    def log_likelihood(theta):
        seen.append(theta)
        shift = np.asarray(theta["log_tau"]) - np.asarray(dis.init["log_tau"])
        c = np.exp(-0.5 * shift)
        return -sum(abs(np.log(c[j] / target[j])) for j in target)

    monkeypatch.setattr(scanner.model, "log_likelihood", log_likelihood)
    profile = scanner._profile_prior_amplitudes(dict(dis.init))
    assert profile == pytest.approx(target)
    keys = set(scanner._theta_at(dict(dis.init)))
    assert all(set(theta) == keys for theta in seen)
    for theta in seen:
        np.testing.assert_allclose(
            np.asarray(theta["log_eta"]) - np.asarray(dis.init["log_eta"]),
            np.asarray(theta["log_tau"]) - np.asarray(dis.init["log_tau"]),
        )
    # Trials keep the scaled light at or below 0.98: five at 0.62, six at 0.38.
    assert len(seen) == 5 + 6


# -- the accessors of a fit and the free velocity table -------------------------------


def test_the_accessors_of_a_fit_share_one_solve(dataset, truth, monkeypatch):
    """The pipeline read the solve at the MAP about ten times per star, each one a solve.

    The values are those of one solve each, bit for bit, and a copy made with
    ``dataclasses.replace`` solves again.
    """
    from dataclasses import replace

    import jax.numpy as jnp

    from albireo.forward import data_residual_zscores
    from albireo.likelihood import spectra_std

    fit = _fit_at_injected(_dis(dataset), truth, [100.0] * 5)
    marginal = fit.marginal()
    d_hat, std = np.asarray(marginal.d_hat), np.asarray(spectra_std(marginal))
    z = np.asarray(data_residual_zscores(fit.dis.model.problem_at(fit.theta), marginal.d_hat))
    z_rms = float(np.sqrt(np.mean(np.square(z[np.isfinite(z)]))))
    solves = []
    solve = ab.Fit.marginal

    def counted(self):
        solves.append(self)
        return solve(self)

    monkeypatch.setattr(ab.Fit, "marginal", counted)
    shared = fit._spectra_and_std()
    assert fit.z_rms == z_rms and fit.z_rms == z_rms
    fit.composite(), fit.templates(), fit.summary()
    assert len(solves) == 1
    np.testing.assert_array_equal(shared[0], d_hat)
    np.testing.assert_array_equal(shared[1], std)
    np.testing.assert_array_equal(fit.spectra(), d_hat)
    np.testing.assert_array_equal(fit.std(), std)
    np.testing.assert_array_equal(fit.residual_zscores(), z)
    assert not fit.spectra().flags.writeable and not fit.std().flags.writeable
    # A copy may change the declaration or the parameters.
    moved = dict(fit.result.params, k=jnp.asarray(fit.result.params["k"]) + 1.0)
    copy = replace(fit, result=replace(fit.result, params=moved))
    assert not np.array_equal(copy.spectra(), d_hat)
    assert len(solves) == 2


def _stub_the_optimizer(monkeypatch):
    """Replace `run_map` with one that returns its start and records its arguments."""
    from albireo.inference import MAPResult

    facade = importlib.import_module("albireo.facade")
    seen = {}

    def run_map(model, *, init, max_steps, tol=1e-2, **kwargs):
        seen.update(max_steps=max_steps, tol=tol)
        return MAPResult(
            params=dict(init),
            unconstrained={},
            potential=0.0,
            grad_norm=1.0,
            converged=False,
            num_steps=max_steps,
        )

    monkeypatch.setattr(facade, "run_map", run_map)
    return seen


def test_a_free_velocity_table_of_a_circular_orbit_can_be_solved(dataset, truth, monkeypatch):
    """The table's theta took the declaration's fixed (secosw, sesinw) back in, and the
    model rejects Keplerian sites alongside a velocity table, so every solve raised."""
    _stub_the_optimizer(monkeypatch)
    dis = _dis(dataset, orbit=_orbit(ecc=ab.Fixed(0.0)))
    assert {"secosw", "sesinw"} <= set(dis.fixed)
    table = _fit_at_injected(dis, truth, [100.0] * 5).free_velocities(max_steps=2)
    assert table.mode == "velocity" and not {"secosw", "sesinw"} & set(table.theta)
    assert table.spectra().shape == (2, dis.grid.n)
    assert "free velocity table" in table.summary()
    with pytest.raises(ValueError, match="already a free velocity table"):
        table.free_velocities(max_steps=1)


def test_a_fitted_noise_correlation_is_kept_by_the_free_velocity_table(dataset, truth, monkeypatch):
    """The table copied only the smoothness and nebular sites, and was fitted under the
    diagonal noise model without notice."""
    _stub_the_optimizer(monkeypatch)
    dis = _dis(dataset, noise_correlation=ab.Between(-0.9, 0.9))
    table = _fit_at_injected(dis, truth, [100.0] * 5).free_velocities(max_steps=2)
    assert table.priors_used["ar1_phi"] is dis.priors["ar1_phi"]
    assert set(table.priors_used) == {"velocity", "log_tau", "log_eta", "ar1_phi"}
    correlation = table.noise_correlation()
    assert set(correlation) == {"DEMO"} and np.isfinite(correlation["DEMO"])


@pytest.mark.slow
def test_a_free_velocity_table_is_fitted_with_fixed_orbital_sites_and_a_fitted_correlation(
    dataset, truth
):
    """The two defects above, through the optimizer and the correlated model."""
    dis = _dis(dataset, orbit=_orbit(ecc=ab.Fixed(0.0)), noise_correlation=ab.Between(-0.9, 0.9))
    table = _fit_at_injected(dis, truth, [100.0] * 5).free_velocities(max_steps=2)
    assert "ar1_phi" in table.params and np.isfinite(table.noise_correlation()["DEMO"])
    assert table.spectra().shape == (2, dis.grid.n)
    assert "free velocity table" in table.summary()


def test_the_free_velocity_table_keeps_the_smoothing_and_the_step_cap(dataset, truth, monkeypatch):
    seen = _stub_the_optimizer(monkeypatch)
    smoothed = _fit_at_injected(_dis(dataset), truth, [100.0] * 5).smoothed({"DEMO": 9.0})
    table = smoothed.free_velocities(max_steps=7, tol=0.5)
    assert seen == {"max_steps": 7, "tol": 0.5}
    assert table.smoothing == smoothed.smoothing and table.max_steps == 7


def test_a_fixed_semi_amplitude_is_reported_by_the_fit_and_the_posterior(dataset, truth):
    """A fixed k is a constant of the model, which neither the optimizer nor the sampler
    returns, and star() read it from them."""
    dis = _dis(dataset, orbit=_orbit(k=ab.Fixed([42.0, 63.0])))
    fit = _fit_at_injected(dis, truth, [100.0] * 5, drop=("k",))
    assert "k" not in fit.params
    assert fit.star("secondary")["k"] == 63.0
    assert "K(primary)    42.000 km/s" in fit.summary()
    draws = {k: v for k, v in _orbit_draws(np.random.default_rng(7)).items() if k != "k"}
    post = ab.Posterior(dis=dis, mcmc=_Draws(draws), map=fit)
    row = post.star("primary")
    assert row["k"] == 42.0 and row["k_std"] == 0.0 and row["k_hdi"] == (42.0, 42.0)
    assert "K(primary)" not in post.summary()


def test_the_posterior_of_one_star_is_reported(dataset):
    """One star's k is a scalar site, and its draws have no component axis."""
    single = _dis(
        dataset, components=[ab.Star("only", light=1.0)], orbit=_orbit(k=ab.Between(10.0, 90.0))
    )
    draws = _orbit_draws(np.random.default_rng(8))
    draws["k"] = draws["k"][..., 0]
    post = ab.Posterior(dis=single, mcmc=_Draws(draws), map=None)
    row = post.star("only")
    assert abs(row["k"] - 42.0) < 0.05 and 0.0 < row["k_std"] < 0.1
    mean, _sd, low, high, rhat, ess = _row(post.summary(), "K(only) [km/s]")
    assert abs(float(mean) - 42.0) < 0.05 and abs(float(rhat) - 1.0) < 0.05
    assert float(low.strip("[,")) < 42.0 < float(high.strip("]")) and float(ess) > 100.0


def test_a_smoothed_fit_keeps_the_conjunction_window_it_was_fitted_under(
    dataset, truth, monkeypatch
):
    """A smoothed fit has no scan record, and its Laplace and NUTS window fell back to the
    declaration's, which starts at the first epoch: a fitted conjunction before it had no
    unconstrained coordinate."""
    import numpyro.distributions as dist

    from albireo.facade import PhaseScan

    dis = _dis(dataset)
    period = float(truth["period"])
    first = float(np.min(dataset.bjd))
    t_conj = _injected_t_conj(truth)
    t_conj += np.floor((first - t_conj) / period) * period  # the last conjunction before
    assert t_conj < first
    scan = PhaseScan(period=period, trials=np.array([t_conj]), values=np.array([0.0]), best=t_conj)
    window = dist.Uniform(t_conj - 0.5 * period, t_conj + 0.5 * period)
    fit = _fit_at_injected(
        dis,
        truth,
        [100.0] * 5,
        t_conj=t_conj,
        phase_scan=scan,
        priors_used={**dis.priors, "t_conj": window},
    )
    seen = []

    def model(self, priors, *, fixed=None):
        seen.append(priors["t_conj"])

    monkeypatch.setattr(ab.MarginalOrbitModel, "model", model)
    fit._orbit_model()
    smoothed = fit.smoothed({"DEMO": 9.0})
    assert smoothed.phase_scan is None
    smoothed._orbit_model()
    held, kept = seen
    assert float(kept.low) < t_conj < float(kept.high)
    np.testing.assert_allclose(
        [float(kept.low), float(kept.high)], [float(held.low), float(held.high)], atol=1e-9
    )


def test_a_scan_with_a_failed_trial_keeps_a_finite_contrast():
    """A failed solve is -inf, and one such trial made the phase contrast infinite."""
    from dataclasses import replace

    from albireo.facade import PhaseScan, SemiAmplitudeScan

    values = np.array([-10.0, -3.0, -np.inf, -7.0])
    phase = PhaseScan(period=6.0, trials=np.arange(4.0), values=values, best=1.0)
    assert phase.contrast == 7.0
    assert np.isnan(replace(phase, values=np.full(4, -np.inf)).contrast)
    amplitude = SemiAmplitudeScan(
        k=np.ones((2, 2)),
        t_conj=np.ones(2),
        values=np.full(2, -np.inf),
        best=np.ones(2),
        best_t_conj=1.0,
        best_value=-np.inf,
        start=np.ones(2),
        start_value=-np.inf,
    )
    assert np.isnan(amplitude.contrast)


def test_the_scan_declaration_is_built_once(dataset):
    """A repeated fit reuses the coarse declaration, its model and its compiled programs."""
    from dataclasses import replace

    dis = _dis(dataset)
    scanner = dis._scan_declaration()
    assert dis._scan_declaration() is scanner
    assert scanner.grid.dv_kms == pytest.approx(2.0 * dis.grid.dv_kms)
    assert replace(dis)._scan_declaration() is not scanner
