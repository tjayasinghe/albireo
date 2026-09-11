"""Tests for the binary populations (``albireo.population``).

Three kinds of claim: the physics helpers reproduce known numbers (the notebook's
semi-amplitudes, the solar main sequence, the eclipse geometry); the parametric draw is
reproducible, respects its declared ranges and produces systems whose semi-amplitudes,
light ratios and inclinations are consistent with each other; and the catalogue adapters
map every column they are given and fill the rest.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from albireo.population import (
    BinarySystem,
    MainSequence,
    draw_population,
    eclipse_probability,
    ecliptic_to_icrs,
    from_debcat,
    from_gaia_sb2,
    population_summary,
    read_population,
    rvs_light_ratio,
    semi_amplitudes,
    semimajor_axis_rsun,
    write_population,
)
from albireo.simulate import synthetic_library


def test_semi_amplitudes_reproduce_the_notebook():
    k1, k2 = semi_amplitudes(1.30, 1.20, 4.0, 0.10, math.radians(87.0))
    assert round(float(k1), 1) == 87.7 and round(float(k2), 1) == 95.0
    # Vectorised, and the sum scales as (M sin i) / (P^(1/3) sqrt(1 - e^2)).
    k1v, k2v = semi_amplitudes([1.0, 1.0], [1.0, 0.5], [10.0, 10.0], [0.0, 0.0], math.pi / 2)
    assert k1v.shape == (2,) and k1v[0] == pytest.approx(k2v[0])
    assert k1v[1] < k1v[0] and k2v[1] > k1v[1]


def test_keplers_law_and_the_eclipse_geometry():
    assert float(semimajor_axis_rsun(1.0, 365.25)) == pytest.approx(215.0, rel=0.01)
    # Twin solar pair at 10 d: (R1 + R2) / a = 0.0813 (research note F.6).
    a = float(semimajor_axis_rsun(2.0, 10.0))
    assert float(eclipse_probability(1.0, 1.0, a)) == pytest.approx(0.0813, abs=0.002)
    # Eccentricity raises the probability by (1 + e |sin w|) / (1 - e^2), capped at one.
    circular = float(eclipse_probability(1.0, 1.0, a))
    eccentric = float(eclipse_probability(1.0, 1.0, a, ecc=0.5, omega=math.pi / 2))
    assert eccentric == pytest.approx(circular * 1.5 / 0.75)
    assert float(eclipse_probability(5.0, 5.0, 8.0)) == 1.0


def test_the_main_sequence_passes_through_the_sun():
    seq = MainSequence()
    assert float(seq.teff(1.0)) == pytest.approx(5770.0, rel=0.01)
    assert float(seq.radius(1.0)) == pytest.approx(1.012, rel=0.01)
    assert float(seq.logg(1.0)) == pytest.approx(4.43, abs=0.02)
    assert float(seq.absolute_g(1.0)) == pytest.approx(4.635, abs=0.02)
    assert float(seq.g_minus_rp(1.0)) == pytest.approx(0.459, abs=0.01)
    assert float(seq.mass_from_teff(5770.0)) == pytest.approx(1.0, rel=0.01)
    assert np.all(np.diff(seq.teff(np.linspace(0.2, 2.5, 50))) > 0)
    eker = MainSequence("eker")
    assert float(eker.radius(1.0)) == pytest.approx(0.992, rel=0.01)
    assert float(eker.teff(1.5)) == pytest.approx(6637.0, rel=0.01)
    assert float(eker.teff(10.0)) == pytest.approx(24_490.0, rel=0.01)
    assert float(eker.absolute_g(1.0)) == pytest.approx(4.6, abs=0.1)
    with pytest.raises(ValueError):
        MainSequence("other")


@pytest.fixture(scope="module")
def library():
    # A toy library over the RVS band with a continuum that depends on Teff.
    return synthetic_library(
        (8400.0, 8760.0),
        n_pix=1400,
        teff=(4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0),
        logg=(3.5, 4.0, 4.5, 5.0),
        mh=(-0.5, 0.0, 0.5),
    )


def test_the_light_ratio_follows_the_radius_ratio_and_the_continua(library):
    hot = {"teff": 6000.0, "logg": 4.3, "mh": 0.0}
    cool = {"teff": 4800.0, "logg": 4.5, "mh": 0.0}
    equal = rvs_light_ratio(library, hot, hot, 1.0)
    assert equal == pytest.approx(1.0, rel=1e-6)
    assert rvs_light_ratio(library, hot, hot, 0.5) == pytest.approx(0.25, rel=1e-6)
    assert rvs_light_ratio(library, hot, cool, 1.0) < 1.0  # the toy continuum falls with Teff


def test_the_twin_excess_is_the_fraction_of_the_wide_companions():
    """MDS17 define F_twin against the companions with q > 0.3; a cut at 0.4 must not inflate it."""
    from albireo.population import _f_twin, _mass_ratio_pdf

    m1, logp = 1.0, 0.6
    grid = np.linspace(0.3, 1.0, 7001)
    pdf = _mass_ratio_pdf(m1, logp, grid)
    twins = grid >= 0.95
    # The power law continues under the twin box at the amplitude it has at q = 0.9 (the
    # density is tabulated, and its last cell before the box carries the edge's ramp);
    # what stands above it is the excess, a fraction of the unit mass on 0.3 < q < 1.
    at = int(np.argmin(np.abs(grid - 0.9)))
    amplitude = pdf[at] / (grid[at] / 0.3) ** -0.5
    power = amplitude * (grid / 0.3) ** -0.5
    excess = np.trapezoid((pdf - power)[twins], grid[twins])
    assert excess == pytest.approx(_f_twin(m1, logp), abs=0.01)
    # A grid cut at 0.4 sees the conditional density: the same excess, renormalised.
    cut = np.linspace(0.4, 1.0, 6001)
    pdf_cut = _mass_ratio_pdf(m1, logp, cut)
    frac_full = np.trapezoid(pdf[grid >= 0.4], grid[grid >= 0.4])
    assert np.trapezoid(pdf_cut[cut >= 0.95], cut[cut >= 0.95]) == pytest.approx(
        np.trapezoid(pdf[twins], grid[twins]) / frac_full, rel=0.02
    )


def test_the_magnitude_limited_draw_favours_the_bright_and_the_separated():
    """A magnitude-limited draw reaches further for the luminous, and keeps the resolvable."""
    common = dict(seed=5, mass_range=(0.6, 1.5), period_range=(0.8, 200.0))
    field = draw_population(150, mass_weighting="volume", **common)
    bright = draw_population(150, mass_weighting="magnitude", min_separation_kms=40.0, **common)
    assert np.median([s.m1 for s in bright]) > np.median([s.m1 for s in field])
    assert all(s.max_separation_kms >= 40.0 for s in bright)
    assert any(s.max_separation_kms < 40.0 for s in field)
    with pytest.raises(ValueError, match="mass_weighting"):
        draw_population(2, mass_weighting="flux")


@pytest.mark.parametrize("kind", ["eclipsing", "spectroscopic", "mixed"])
def test_the_parametric_draw_is_consistent(kind, library):
    systems = draw_population(30, kind=kind, seed=3, library=library, period_range=(0.8, 200.0))
    assert len(systems) == 30
    assert len({s.name for s in systems}) == 30
    for s in systems:
        assert s.m1 >= s.m2 > 0 and 0.4 <= s.q <= 1.0
        assert 0.8 <= s.period <= 200.0 and 0.0 <= s.ecc < 1.0
        assert s.ecc == 0.0 or s.period > 2.0
        assert 0.0 < s.incl <= math.pi / 2 and 0.0 <= s.omega < 2 * math.pi
        assert 4000.0 <= s.teff2 <= s.teff1 <= 7000.0
        assert 3.5 <= s.logg1 <= 5.0 and 3.5 <= s.logg2 <= 5.0
        assert -0.5 <= s.mh <= 0.3
        assert 0.0 < s.light_ratio <= 1.5
        assert 5.0 <= s.g_mag <= 12.2 and np.isfinite(s.grvs) and s.grvs < s.g_mag
        assert -90.0 <= s.ecl_lat_deg <= 90.0 and s.n_transits >= 2
        k1, k2 = semi_amplitudes(s.m1, s.m2, s.period, s.ecc, s.incl)
        assert s.k1 == pytest.approx(float(k1)) and s.k2 == pytest.approx(float(k2))
        assert s.k1 * s.m1 == pytest.approx(s.k2 * s.m2, rel=1e-6)
        p_ecl = float(eclipse_probability(s.r1, s.r2, s.a_rsun, s.ecc, s.omega))
        assert s.eclipsing == (math.cos(s.incl) <= p_ecl)
        if s.period < 15.0:
            assert s.vsini1 == pytest.approx(50.593 * s.r1 / s.period * math.sin(s.incl), rel=1e-3)
        assert s.light_fractions[0] + s.light_fractions[1] == pytest.approx(1.0)
        assert s.orbit().k == (s.k1, s.k2)
        assert s.t_peri <= s.t_conj < s.t_peri + s.period
    if kind == "eclipsing":
        assert all(s.eclipsing for s in systems)
    if kind == "spectroscopic":
        assert not any(s.eclipsing for s in systems)
    again = draw_population(30, kind=kind, seed=3, library=library, period_range=(0.8, 200.0))
    assert [s.to_dict() for s in again] == [s.to_dict() for s in systems]
    other = draw_population(30, kind=kind, seed=4, library=library, period_range=(0.8, 200.0))
    assert other[0].period != systems[0].period


def test_the_draw_without_a_library_uses_the_sequence_photometry():
    systems = draw_population(10, seed=1, mass_range=(0.8, 1.4))
    for s in systems:
        assert 0.8 <= s.m2 <= s.m1 <= 1.4
        assert s.light_ratio == pytest.approx(
            10
            ** (
                -0.4
                * (float(MainSequence().absolute_g(s.m2)) - float(MainSequence().absolute_g(s.m1)))
            ),
            rel=1e-6,
        )
    with pytest.raises(ValueError, match="kind"):
        draw_population(2, kind="all")
    with pytest.raises(ValueError, match="mass_range"):
        draw_population(2, mass_range=(2.0, 1.0))


def test_populations_round_trip_through_json(tmp_path, library):
    systems = draw_population(5, seed=2, library=library)
    path = write_population(tmp_path / "pop.json", systems)
    back = read_population(path)
    assert [s.to_dict() for s in back] == [s.to_dict() for s in systems]
    assert all(s.ra_deg is not None and s.dec_deg is not None for s in back)
    text = population_summary(systems)
    assert "5 systems" in text and "K1 + K2" in text
    assert population_summary([]) == "empty population"


def test_the_ecliptic_rotation_agrees_with_astropy():
    """The hand-written rotation against the reference implementation of the frames.

    astropy is not a dependency of albireo; this test runs where it happens to be
    installed. ``BarycentricTrueEcliptic`` adds the nutation of the equinox and of the
    obliquity to the mean J2000 frame the rotation uses, and ``GeocentricTrueEcliptic``
    places the origin at the Earth rather than the barycentre; both differences stay well
    inside an arcminute, and a position enters this module only through the ecliptic
    latitude that sets the transit count and through the position a transit forecast is
    requested for.
    """
    astropy_coordinates = pytest.importorskip("astropy.coordinates")
    units = pytest.importorskip("astropy.units")

    rng = np.random.default_rng(0)
    lon = rng.uniform(0.0, 360.0, 200)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 200)))
    ra, dec = ecliptic_to_icrs(lon, lat)
    assert np.all((ra >= 0.0) & (ra < 360.0)) and np.all(np.abs(dec) <= 90.0)
    mine = astropy_coordinates.SkyCoord(ra=ra * units.deg, dec=dec * units.deg)
    for frame, tolerance in (
        (astropy_coordinates.BarycentricTrueEcliptic, 60.0),
        (astropy_coordinates.GeocentricTrueEcliptic, 60.0),
        (astropy_coordinates.BarycentricMeanEcliptic, 1.0),
    ):
        reference = astropy_coordinates.SkyCoord(
            lon=lon * units.deg, lat=lat * units.deg, frame=frame(equinox="J2000")
        ).icrs
        assert np.max(mine.separation(reference).arcsec) < tolerance
    # The poles and the solstices, which the formula must place exactly.
    assert ecliptic_to_icrs(0.0, 90.0)[0] == pytest.approx(270.0)
    assert ecliptic_to_icrs(0.0, 90.0)[1] == pytest.approx(90.0 - 23.4392911)
    assert ecliptic_to_icrs(90.0, 0.0)[0] == pytest.approx(90.0)
    assert ecliptic_to_icrs(90.0, 0.0)[1] == pytest.approx(23.4392911)
    assert ecliptic_to_icrs(0.0, 0.0)[0] == pytest.approx(0.0)


def test_drawn_systems_carry_the_position_their_latitude_belongs_to():
    systems = draw_population(40, seed=7, mass_range=(0.7, 1.4))
    for s in systems:
        assert 0.0 <= s.ra_deg < 360.0 and -90.0 <= s.dec_deg <= 90.0
        # The declination that an ecliptic latitude allows: |beta| - eps to |beta| + eps.
        assert abs(s.dec_deg) <= abs(s.ecl_lat_deg) + 23.4392911 + 1e-9
        ra, dec = ecliptic_to_icrs(s.meta["ecl_lon"], s.ecl_lat_deg)
        assert s.ra_deg == pytest.approx(float(ra)) and s.dec_deg == pytest.approx(float(dec))
    # Uniform in longitude means uniform in right ascension: no quadrant is left empty.
    counts = np.histogram([s.ra_deg for s in systems], bins=4, range=(0.0, 360.0))[0]
    assert counts.min() > 0
    # A record written before there were positions is still a record: they default to None.
    older = {k: v for k, v in systems[0].to_dict().items() if k not in ("ra_deg", "dec_deg")}
    bare = BinarySystem.from_dict(older)
    assert bare.ra_deg is None and bare.dec_deg is None


def test_debcat_rows_are_completed(tmp_path, library):
    # Two systems in DEBCat's format: CM Dra (real row) and a solar-type pair with a missing
    # metallicity and a missing V magnitude.
    lines = [
        "# System SpT1 SpT2 Pday Vmag BmV logM1 logM1e logM2 logM2e logR1 logR1e logR2 logR2e",
        "# logg1 logg1e logg2 logg2e logT1 logT1e logT2 logT2e logL1 logL1e logL2 logL2e MoH MoHe",
        "CM_Dra M4.5_V M4.5_V 1.268 12.90 1.60 -0.6478 0.0006 -0.6776 0.0006 -0.6003 0.0003 "
        "-0.6243 0.0004 4.9908 0.0008 5.0091 0.0007 3.4960 0.0100 3.4940 0.0140 "
        "-2.2580 0.0380 -2.3130 0.0560 -0.3000 0.1200",
        "Sun_Twin G2_V G5_V 4.000 -9.99 -9.99 0.0000 0.01 -0.0200 0.01 0.0052 0.005 "
        "-0.0100 0.005 4.438 0.01 4.45 0.01 3.7612 0.005 3.7500 0.005 0.0100 0.02 "
        "-0.0500 0.02 -9.99 -9.99",
    ]
    path = tmp_path / "debs.dat"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    systems = from_debcat(path, seed=0, library=library)
    assert [s.name for s in systems] == ["debcat-CM_Dra", "debcat-Sun_Twin"]
    cm = systems[0]
    assert cm.eclipsing and cm.period == 1.268 and cm.mh == pytest.approx(-0.3)
    assert cm.m1 == pytest.approx(10**-0.6478) and cm.teff1 == pytest.approx(10**3.4960)
    assert cm.ecc == 0.0  # circular below two days
    assert cm.g_mag == pytest.approx(12.90 + (12.04 - 13.58), abs=0.5)  # V + (M_G - M_V) at M4.5V
    assert cm.source == "DEBCat" and cm.meta["spt1"] == "M4.5_V"
    sun = systems[1]
    assert sun.mh == 0.0 and 5.0 <= sun.g_mag <= 12.2  # missing values filled
    assert sun.ecc >= 0.0 and sun.eclipsing
    assert math.cos(sun.incl) <= float(
        eclipse_probability(sun.r1, sun.r2, sun.a_rsun, sun.ecc, sun.omega)
    )
    assert 0.0 < sun.light_ratio < 1.5
    subset = from_debcat(path, names=["CM_Dra"])
    assert len(subset) == 1


def test_gaia_rows_are_completed(library):
    rows = [
        {
            "source_id": 123,
            "nss_solution_type": "SB2",
            "period": 6.31,
            "eccentricity": None,
            "arg_periastron": 40.0,
            "k1": 30.0,
            "k2": 55.0,
            "gamma": 12.0,
            "m1": 1.1,
            "m2": 0.6,
            "phot_g_mean_mag": 9.5,
            "phot_rp_mean_mag": 9.0,
            "teff_gspphot": 5900.0,
            "ecl_lat": 33.0,
            "rv_n_good_obs_primary": 21,
        },
        {"source_id": 124, "period": None, "k1": 30.0, "k2": 40.0},
        {
            "source_id": 125,
            "nss_solution_type": "SB2C",
            "period": 2.5,
            "eccentricity": None,
            "arg_periastron": None,
            "k1": 80.0,
            "k2": 90.0,
            "gamma": -5.0,
            "m1": None,
            "m2": None,
            "phot_g_mean_mag": 11.0,
            "phot_rp_mean_mag": 10.4,
            "teff_gspphot": 6200.0,
            "ecl_lat": -60.0,
            "rv_n_good_obs_primary": 14,
        },
    ]
    systems = from_gaia_sb2(rows, seed=0, library=library)
    assert [s.name for s in systems] == ["gaia-123", "gaia-125"]
    first = systems[0]
    assert first.ecc == 0.0 and first.omega == pytest.approx(math.radians(40.0))
    assert first.k1 == 30.0 and first.k2 == 55.0 and first.gamma == 12.0
    assert first.m1 == 1.1 and first.m2 == 0.6 and first.n_transits == 21
    assert first.g_mag == 9.5 and first.g_rp == pytest.approx(0.5)
    assert first.ecl_lat_deg == 33.0 and np.isfinite(first.grvs)
    # The inclination reconciles the catalogue's K with the masses.
    k1, _ = semi_amplitudes(first.m1, first.m2, first.period, first.ecc, first.incl)
    assert float(k1) == pytest.approx(30.0, rel=1e-6)
    second = systems[1]
    assert second.m1 >= second.m2 and second.q == pytest.approx(80.0 / 90.0, rel=0.05)
    assert second.source.startswith("Gaia")


def test_binary_system_from_dict_ignores_unknown_keys():
    data = draw_population(1, seed=5)[0].to_dict()
    data["extra"] = 1
    system = BinarySystem.from_dict(data)
    assert system.name == data["name"]
