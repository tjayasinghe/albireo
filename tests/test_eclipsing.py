"""Tests for the eclipsing-binary population (``albireo.eclipsing``).

The geometry is tested against closed forms: the area of the intersection of two discs,
the central transit of a limb-darkened disc, and the duration of a circular eclipse. The
tracks reader is tested on the packaged table, the library set on two toy boxes that share
a node, and the draw for the properties its docstring states: every system is detached,
eclipses, lies in its cell, and is reproduced by its seed.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import albireo.eclipsing as ec
from albireo.population import (
    BinarySystem,
    _mass_ratio_pdf,
    read_population,
    semi_amplitudes,
    semimajor_axis_rsun,
    write_population,
)
from albireo.simulate import synthetic_library

MH = (-1.0, -0.5, 0.0, 0.5)


@pytest.fixture(scope="module")
def boxes():
    """Two toy boxes in the RVS band that share the node at 5500 K."""
    common = {"wave_range": (8440.0, 8720.0), "n_pix": 700, "mh": MH, "medium": "vacuum"}
    cool = synthetic_library(teff=(4000.0, 4500.0, 5000.0, 5500.0), logg=(4.0, 4.5, 5.0), **common)
    hot = synthetic_library(teff=(5500.0, 6000.0, 6500.0, 7000.0), logg=(3.5, 4.0, 4.5), **common)
    return ec.LibrarySet([hot, cool])


def _lens(d, rb, rf):
    """Area of the intersection of two discs over the area of the first."""
    if d >= rb + rf:
        return 0.0
    if d <= abs(rb - rf):
        return min(rb, rf) ** 2 / rb**2
    a1 = rb**2 * math.acos((d**2 + rb**2 - rf**2) / (2 * d * rb))
    a2 = rf**2 * math.acos((d**2 + rf**2 - rb**2) / (2 * d * rf))
    a3 = 0.5 * math.sqrt((-d + rb + rf) * (d + rb - rf) * (d - rb + rf) * (d + rb + rf))
    return (a1 + a2 - a3) / (math.pi * rb**2)


def _system(**changes) -> BinarySystem:
    """A circular, edge-on pair of 1.2 and 0.9 solar masses at 3 d."""
    m1, m2, period, incl = 1.2, 0.9, 3.0, math.radians(89.5)
    k1, k2 = semi_amplitudes(m1, m2, period, 0.0, incl)
    base = BinarySystem(
        name="toy",
        m1=m1,
        m2=m2,
        r1=1.3,
        r2=0.85,
        teff1=6200.0,
        teff2=5300.0,
        logg1=4.3,
        logg2=4.5,
        mh=0.0,
        vsini1=22.0,
        vsini2=14.0,
        period=period,
        ecc=0.0,
        omega=0.0,
        t_peri=2457000.0,
        incl=incl,
        gamma=5.0,
        k1=float(k1),
        k2=float(k2),
        eclipsing=True,
        light_ratio=0.25,
        g_mag=9.0,
        g_rp=0.4,
        grvs=8.5,
        ecl_lat_deg=30.0,
        n_transits=30,
    )
    return dataclasses.replace(base, **changes)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_the_hidden_fraction_of_a_uniform_disc_is_the_area_of_the_intersection():
    rng = np.random.default_rng(1)
    rb, rf = rng.uniform(0.2, 3.0, size=(2, 2000))
    d = rng.uniform(0.0, 1.05, size=2000) * (rb + rf)
    got = ec.hidden_fraction(d, rb, rf, 0.0)
    expected = np.array([_lens(*args) for args in zip(d, rb, rf, strict=True)])
    np.testing.assert_allclose(got, expected, atol=1e-9)
    assert got.min() >= 0.0 and got.max() <= 1.0


@pytest.mark.parametrize("u", [0.0, 0.3, 0.6])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_a_central_transit_of_a_limb_darkened_disc_has_its_closed_form(u, p):
    # The light inside the fractional radius p of I(mu) = 1 - u (1 - mu), over the total.
    expected = ((1 - u) * p**2 + (2 * u / 3) * (1 - (1 - p**2) ** 1.5)) / (1 - u / 3)
    assert float(ec.hidden_fraction(0.0, 1.0, p, u)) == pytest.approx(expected, abs=1e-9)


def test_a_total_eclipse_hides_all_the_light_and_no_overlap_hides_none():
    assert float(ec.hidden_fraction(0.3, 1.0, 1.5, 0.5)) == 1.0
    assert float(ec.hidden_fraction(2.6, 1.0, 1.5, 0.5)) == 0.0
    # Limb darkening puts more of the light at the centre: a central transit hides more.
    assert ec.hidden_fraction(0.0, 1.0, 0.4, 0.6) > ec.hidden_fraction(0.0, 1.0, 0.4, 0.0)
    # Vectorised over every argument.
    assert ec.hidden_fraction(np.zeros((3, 2)), 1.0, np.full(2, 0.5), 0.2).shape == (3, 2)


def test_the_roche_radius_and_the_pseudo_synchronous_rate():
    assert float(ec.roche_radius(1.0)) == pytest.approx(0.3789, abs=2e-4)
    assert float(ec.roche_radius(0.5)) == pytest.approx(0.3208, abs=2e-4)
    # The two lobes of a pair lie inside the separation.
    assert ec.roche_radius(0.3) + ec.roche_radius(1 / 0.3) < 1.0
    assert float(ec.pseudo_synchronous_ratio(0.0)) == 1.0
    assert float(ec.pseudo_synchronous_ratio(0.4)) == pytest.approx(2.045, abs=2e-3)
    assert np.all(np.diff(ec.pseudo_synchronous_ratio(np.linspace(0.0, 0.8, 9))) > 0.0)


def test_the_limb_darkening_table():
    assert float(ec.limb_darkening(4000.0)) == 0.55
    assert float(ec.limb_darkening(10000.0)) == 0.29
    assert float(ec.limb_darkening(5750.0, "rp")) == 0.508
    # Held constant outside the table, and weaker in the far red than in G_RP.
    assert float(ec.limb_darkening(3000.0)) == 0.55 and float(ec.limb_darkening(2e4)) == 0.29
    teff = np.linspace(4000.0, 10000.0, 25)
    assert np.all(ec.limb_darkening(teff, "rvs") < ec.limb_darkening(teff, "rp"))
    with pytest.raises(ValueError, match="band"):
        ec.limb_darkening(5000.0, "g")


def test_the_light_fractions_through_an_orbit():
    system = _system()
    t_conj = system.t_conj
    times = t_conj + system.period * np.array([0.0, 0.25, 0.5, 0.75])
    light = ec.eclipse_light(system, times)
    np.testing.assert_allclose(light["fractions"].sum(axis=0), 1.0, atol=1e-12)
    # At its conjunction the primary is the farther star and loses light. Half a period
    # later the smaller secondary is behind the primary and is hidden whole.
    assert light["primary_behind"][0] and not light["primary_behind"][2]
    assert light["visible"][0, 0] < 1.0 and light["visible"][1, 0] == 1.0
    assert light["visible"][1, 2] == 0.0 and light["fractions"][1, 2] == 0.0
    assert light["flux"][2] == pytest.approx(1.0 / 1.25)
    # At the quadratures nothing is hidden and the fractions are those of the record.
    assert not light["in_eclipse"][1] and not light["in_eclipse"][3]
    np.testing.assert_allclose(light["fractions"][:, 1], system.light_fractions)
    assert np.array_equal(light["in_eclipse"], [True, False, True, False])
    # The primary's eclipse is an annular one: the hidden fraction of the closed form.
    u1 = float(ec.limb_darkening(system.teff1))
    separation = system.a_rsun * math.cos(system.incl)
    hidden = float(ec.hidden_fraction(separation, system.r1, system.r2, u1))
    assert light["visible"][0, 0] == pytest.approx(1.0 - hidden)


def test_a_third_light_dilutes_the_pair_and_is_not_eclipsed():
    system = _system()
    times = system.t_conj + system.period * np.array([0.25, 0.5])
    light = ec.eclipse_light(system, times, third_light=0.2)
    assert light["fractions"].shape == (3, 2)
    np.testing.assert_allclose(light["fractions"].sum(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(light["fractions"][:, 0], [0.64, 0.16, 0.2])
    # With the secondary hidden, the third light is a larger share of what remains.
    assert light["fractions"][2, 1] == pytest.approx(0.2 / 0.84)
    with pytest.raises(ValueError, match="third_light"):
        ec.eclipse_light(system, times, third_light=1.0)


@pytest.mark.parametrize("ecc,omega", [(0.0, 0.0), (0.3, 0.8), (0.3, 4.0)])
def test_the_eclipse_duration_against_the_sampled_orbit(ecc, omega):
    system = _system(ecc=ecc, omega=omega, incl=math.radians(87.0), period=6.0)
    times = system.t_peri + np.linspace(0.0, system.period, 200_001)[:-1]
    light = ec.eclipse_light(system, times)
    rho = (system.r1 + system.r2) / system.a_rsun
    cos_i, e_sin = math.cos(system.incl), ecc * math.sin(omega)
    for behind, sign in ((True, 1.0), (False, -1.0)):
        sampled = np.mean(light["in_eclipse"] & (light["primary_behind"] == behind))
        duration = float(ec._eclipse_duration(system.period, rho, cos_i, ecc, sign * e_sin))
        # The closed form holds the separation and the angular rate at their values at
        # the conjunction, which is exact for a circular orbit.
        assert duration / system.period == pytest.approx(sampled, rel=1e-3 if ecc == 0 else 0.05)


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


def test_the_packaged_tracks_give_a_solar_model_and_its_evolution():
    tracks = ec.StellarTracks.load()
    assert tracks is ec.StellarTracks.load()  # read once
    assert tracks.mass[0] == pytest.approx(0.10) and tracks.mass[-1] == pytest.approx(6.0)
    assert tracks.zams == 202 and tracks.tams == 454
    sun = tracks.at(1.0, 4.57e9)
    assert float(sun["radius"]) == pytest.approx(1.0, abs=0.05)
    assert float(sun["teff"]) == pytest.approx(5772.0, abs=120.0)
    assert float(sun["logg"]) == pytest.approx(4.44, abs=0.04)
    assert tracks.zams < float(sun["eep"]) < tracks.tams
    # A star grows along its main sequence and has left it by the terminal age.
    ages = np.array([0.5e9, 2e9, 5e9, 9e9])
    radius = tracks.at(1.0, ages)["radius"]
    assert np.all(np.diff(radius) > 0.0)
    assert float(tracks.age_at(1.0, tracks.zams)) < float(tracks.age_at(1.0, tracks.tams))
    assert float(tracks.at(1.0, 1.02 * tracks.age_at(1.0, tracks.tams))["eep"]) > tracks.tams
    # More massive stars are hotter on the main sequence and live for less time.
    young = tracks.at(np.array([0.7, 1.0, 1.5, 2.5]), 2e8)
    assert np.all(np.diff(young["teff"]) > 0.0)
    assert np.all(np.diff(tracks.age_at(np.array([1.0, 1.5, 2.5]), tracks.tams)) < 0.0)


def test_the_tracks_return_nan_outside_the_table():
    tracks = ec.StellarTracks.load()
    out = tracks.at(np.array([0.05, 1.0, 7.0, 3.0]), np.array([1e9, 1e9, 1e8, 5e9]))
    # Below and above the tabulated masses, and beyond the last point of a 3 Msun track.
    assert np.isfinite(out["radius"]).tolist() == [False, True, False, False]
    assert np.isnan(tracks.age_at(0.05, tracks.zams))
    # Broadcasting, and a mass between two tracks lies between them.
    grid = tracks.at(np.array([[1.0], [1.01], [1.02]]), np.array([1e9, 3e9]))
    assert grid["teff"].shape == (3, 2)
    assert np.all(np.diff(grid["teff"], axis=0) > 0.0)


# ---------------------------------------------------------------------------
# Library boxes
# ---------------------------------------------------------------------------


def test_a_library_set_renders_each_star_from_the_box_that_contains_it(boxes):
    assert boxes.teff_range == (4000.0, 7000.0)
    assert boxes.box([4200.0, 5499.0, 5500.0, 6900.0, 9000.0, 3000.0]).tolist() == [
        0,
        0,
        1,
        1,
        1,
        0,
    ]
    library, labels, moved = boxes.labels(4800.0, 4.6, 0.1)
    assert library is boxes.libraries[0] and not moved
    assert labels == {"teff": 4800.0, "logg": 4.6, "mh": 0.1}
    # Outside its box a label is moved to the edge and the star is flagged.
    library, labels, moved = boxes.labels(3000.0, 5.4, 0.1)
    assert library is boxes.libraries[0] and moved
    assert labels == {"teff": 4000.0, "logg": 5.0, "mh": 0.1}
    _, labels, moved = boxes.labels(6200.0, 3.0, 0.9)
    assert moved and labels == {"teff": 6200.0, "logg": 3.5, "mh": 0.5}


def test_the_continuum_is_on_one_scale_across_the_boxes(boxes):
    # Just below and just above the shared node, in the two boxes.
    below = boxes.log_continuum(5500.0 - 1e-6, 4.3, 0.0)
    above = boxes.log_continuum(5500.0, 4.3, 0.0)
    assert float(below) == pytest.approx(float(above), abs=1e-6)
    # The toy continuum is 4 ln(T / 5000) plus a constant: the flux ratio of two stars in
    # different boxes is (T2 / T1)^4 (R2 / R1)^2 at the nodes.
    ratio = boxes.flux_ratio(
        {"teff": 6500.0, "logg": 4.0, "mh": 0.0}, {"teff": 4500.0, "logg": 4.5, "mh": 0.0}, 0.7
    )
    assert ratio == pytest.approx((4500.0 / 6500.0) ** 4 * 0.49, rel=1e-9)
    assert boxes.log_continuum(np.array([4500.0, 6500.0]), 4.5, 0.0).shape == (2,)
    with pytest.raises(ValueError, match="at least one"):
        ec.LibrarySet([])


# ---------------------------------------------------------------------------
# Mass ratios
# ---------------------------------------------------------------------------


def test_the_mass_ratio_density_is_the_published_one_by_default():
    q = np.linspace(0.1, 1.0, 901)
    for m1, logp in ((1.0, 0.3), (2.0, 0.6), (1.0, 1.5)):
        density = ec._mass_ratio_density(m1, logp, q)
        above = q >= 0.3
        assert float(np.trapezoid(density[above], q[above])) == pytest.approx(1.0, rel=1e-9)
        published = _mass_ratio_pdf(m1, logp, q)
        ours = density / np.trapezoid(density, q)
        # The two differ only where each places the step of the twin excess on its grid.
        away = np.abs(q - 0.95) > 0.005
        np.testing.assert_allclose(ours[away], published[away], rtol=0.01)


def test_a_scaled_and_widened_twin_excess():
    q = np.linspace(0.1, 1.0, 901)
    above = q >= 0.3
    published = ec._mass_ratio_density(1.0, 0.3, q)
    none = ec._mass_ratio_density(1.0, 0.3, q, (0.0, 0.95, "uniform"))
    ramp = ec._mass_ratio_density(1.0, 0.3, q, (0.3, 0.85, "ramp"))
    for density in (none, ramp):
        assert float(np.trapezoid(density[above], q[above])) == pytest.approx(1.0, rel=1e-9)

    def excess(density):
        # The density of the power law alone is that of `none` times one minus the excess.
        share = 1.0 - np.median(density[(q > 0.3) & (q < 0.8)] / none[(q > 0.3) & (q < 0.8)])
        return float(share)

    assert excess(published) == pytest.approx(0.30, abs=1e-6)
    assert excess(ramp) == pytest.approx(0.09, abs=1e-6)
    # The ramp starts from zero at 0.85 and rises to one.
    at = lambda x: float(np.interp(x, q, ramp / none))  # noqa: E731
    assert at(0.84) == pytest.approx(at(0.5)) and at(0.9) < at(0.95) < at(1.0)


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def drawn(boxes):
    classes = {"G": (5300.0, 6000.0), "F": (6000.0, 6900.0)}
    return ec.draw_eclipsing_population(
        4, seed=5, libraries=boxes, classes=classes, grvs_edges=(8.0, 9.0, 10.0, 11.0)
    )


def test_the_draw_fills_every_cell(drawn):
    assert len(drawn) == 2 * 3 * 4
    assert [s.name for s in drawn[:4]] == ["G-00-000", "G-01-000", "G-02-000", "G-00-001"]
    cells = ec.design_cells({"G": (5300.0, 6000.0), "F": (6000.0, 6900.0)}, (8.0, 9.0, 10.0, 11.0))
    assert len(cells) == 6 and cells[0] == ("G", 0, 8.0, 9.0)
    for system in drawn:
        lo, hi = {"G": (5300.0, 6000.0), "F": (6000.0, 6900.0)}[system.meta["class"]]
        assert lo <= system.teff1 < hi
        cell = system.meta["grvs_bin"]
        assert 8.0 + cell <= system.grvs <= 9.0 + cell
        assert system.m2 <= system.m1 and system.m2 >= 0.1
        assert system.logg1 >= 3.5 and system.meta["eep"][0] >= 202.0
        assert system.eclipsing and system.source == "eclipsing-design"
        assert system.n_transits >= 2 and 0.2 <= system.period <= 500.0


def test_every_drawn_system_is_detached_and_eclipses(drawn):
    for system in drawn:
        a = float(semimajor_axis_rsun(system.m1 + system.m2, system.period))
        periastron = a * (1.0 - system.ecc)
        q = system.m2 / system.m1
        assert system.r1 < 0.75 * float(ec.roche_radius(1.0 / q)) * periastron
        assert system.r2 < 0.75 * float(ec.roche_radius(q)) * periastron
        # The deeper eclipse is at least 0.04 mag in G, and the light is lost at one of
        # the two conjunctions.
        assert max(system.meta["depth_mag"]) >= 0.04
        conjunctions = system.t_conj + system.period * np.array([0.0, 0.5])
        if system.ecc == 0.0:
            assert ec.eclipse_light(system, conjunctions)["in_eclipse"].any()
        k1, k2 = semi_amplitudes(system.m1, system.m2, system.period, system.ecc, system.incl)
        assert system.k1 == pytest.approx(float(k1)) and system.k2 == pytest.approx(float(k2))
        assert system.meta["rho"] == pytest.approx((system.r1 + system.r2) / a)


def test_the_draw_is_reproduced_by_its_seed_and_survives_a_file(drawn, boxes, tmp_path):
    classes = {"G": (5300.0, 6000.0), "F": (6000.0, 6900.0)}
    again = ec.draw_eclipsing_population(
        4, seed=5, libraries=boxes, classes=classes, grvs_edges=(8.0, 9.0, 10.0, 11.0)
    )
    assert [s.to_dict() for s in again] == [s.to_dict() for s in drawn]
    other = ec.draw_eclipsing_population(
        4, seed=6, libraries=boxes, classes=classes, grvs_edges=(8.0, 9.0, 10.0, 11.0)
    )
    assert [s.period for s in other] != [s.period for s in drawn]
    # Every system has its own simulation seed.
    assert len({s.meta["seed"] for s in drawn}) == len(drawn)
    path = write_population(tmp_path / "population.json", drawn)
    assert [s.to_dict() for s in read_population(path)] == [s.to_dict() for s in drawn]
    columns = ec.population_columns(drawn)
    assert columns["klass"].tolist().count("G") == 12
    assert np.allclose(columns["q"], [s.m2 / s.m1 for s in drawn])
    assert np.all(columns["weight"] == 1.0)


def test_the_draw_validates_its_arguments(boxes):
    with pytest.raises(ValueError, match="n_per_cell"):
        ec.draw_eclipsing_population(0, libraries=boxes)
    with pytest.raises(ValueError, match="q_min"):
        ec.draw_eclipsing_population(1, libraries=boxes, q_min=0.05)
    with pytest.raises(ValueError, match="twin_shape"):
        ec.draw_eclipsing_population(1, libraries=boxes, twin_shape="step")


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def _weighted(systems, **kwargs):
    counts = {}
    for cell in range(3):
        for name, total in (("G", 300.0), ("F", 600.0)):
            # A catalogue with twice as many systems below 2 d as above, in two period bins.
            counts[(cell, name, 0)] = (cell + 1) * total * 2 / 3
            counts[(cell, name, 2)] = (cell + 1) * total / 3
    return ec.catalogue_weights(systems, counts, **kwargs), counts


def test_the_cell_weights_reproduce_the_catalogue_counts(drawn):
    (weighted, summary), counts = _weighted(drawn, min_systems=1)
    columns = ec.population_columns(weighted)
    assert summary["catalogue_total"] == pytest.approx(sum(counts.values()))
    for cell in range(3):
        for name in ("G", "F"):
            sel = (columns["klass"] == name) & (columns["grvs_bin"] == cell)
            total = counts[(cell, name, 0)] + counts[(cell, name, 2)]
            assert columns["weight_cell"][sel].sum() == pytest.approx(total)
            assert np.ptp(columns["weight_cell"][sel]) == pytest.approx(0.0, abs=1e-9)
    assert summary["effective_size_cell"] <= len(drawn)


def test_the_raked_weights_follow_the_supported_periods(drawn):
    (weighted, summary), counts = _weighted(drawn, min_systems=1, max_factor=1e6)
    columns = ec.population_columns(weighted)
    pbin = np.floor(np.log10(columns["period"]) / 0.2).astype(int)
    # A system in a period bin that the catalogue lacks has no weight.
    in_support = np.isin(pbin, (0, 2))
    assert np.all(columns["weight"][~in_support] == 0.0)
    assert np.all(columns["weight"][in_support] > 0.0)
    for name in ("G", "F"):
        in_class = columns["klass"] == name
        # The share of the class's catalogue entries in the period bins the sample has.
        have = [p for p in (0, 2) if np.any(in_class & (pbin == p))]
        share = sum(2 / 3 if p == 0 else 1 / 3 for p in have)
        for cell in range(3):
            sel = in_class & (columns["grvs_bin"] == cell)
            if not np.any(sel & in_support):
                continue
            total = counts[(cell, name, 0)] + counts[(cell, name, 2)]
            assert columns["weight"][sel].sum() == pytest.approx(total * share, rel=1e-6)
    assert 0.0 <= summary["share_outside_period_support"] <= 1.0
    # With a bound of one the raking leaves the ratios of the weights inside a cell alone.
    (bounded, _), _ = _weighted(drawn, min_systems=1, max_factor=1.0)
    ratio = ec.population_columns(bounded)
    for name in ("G", "F"):
        for cell in range(3):
            sel = (ratio["klass"] == name) & (ratio["grvs_bin"] == cell) & (ratio["weight"] > 0.0)
            if sel.any():
                assert np.ptp(ratio["weight"][sel]) == pytest.approx(0.0, abs=1e-9)


def test_a_cell_that_the_catalogue_does_not_list_has_no_weight(drawn):
    """Counts built from a grouped query have no key for an empty cell."""
    (_, _), counts = _weighted(drawn, min_systems=1)
    partial = {key: value for key, value in counts.items() if not (key[0] == 0 and key[1] == "G")}
    weighted, _ = ec.catalogue_weights(drawn, partial, min_systems=1)
    columns = ec.population_columns(weighted)
    unlisted = (columns["klass"] == "G") & (columns["grvs_bin"] == 0)
    assert unlisted.sum() == 4
    assert np.all(columns["weight_cell"][unlisted] == 0.0)
    assert np.all(columns["weight"][unlisted] == 0.0)
    assert columns["weight"][~unlisted].sum() > 0.0


def test_the_bound_of_the_raking_is_about_the_supported_share(drawn):
    """Where part of a class lies in period bins the sample lacks, a system that needs no
    period factor has its weight by cell times the supported share, inside any bound."""
    (weighted, _), counts = _weighted(drawn, min_systems=1, max_factor=1.0)
    columns = ec.population_columns(weighted)
    pbin = np.floor(np.log10(columns["period"]) / 0.2).astype(int)
    for name in ("G", "F"):
        in_class = columns["klass"] == name
        have = [p for p in (0, 2) if np.any(in_class & (pbin == p))]
        share = sum(2 / 3 if p == 0 else 1 / 3 for p in have)
        inside = in_class & np.isin(pbin, have)
        if not inside.any():
            continue
        for cell in range(3):
            sel = inside & (columns["grvs_bin"] == cell)
            if sel.any():
                total = counts[(cell, name, 0)] + counts[(cell, name, 2)]
                assert columns["weight"][sel].sum() == pytest.approx(total * share, rel=1e-9)
