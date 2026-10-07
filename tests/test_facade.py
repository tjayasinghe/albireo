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
