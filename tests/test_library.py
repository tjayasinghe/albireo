"""Tests for synthetic spectral libraries and their differentiable interpolators.

Interpolators are checked against scipy's implementations (``RegularGridInterpolator`` for
the box, ``LinearNDInterpolator`` for the scattered case), so the oracle is independent
code. Node reproduction is the invariant that allows the warm-start node scan and the
continuous fit to be compared on equal terms. The box interpolators return the stored
spectrum bit-for-bit, and the simplex interpolator returns it to rounding, for the reason
recorded at ``NODE_TOL_EPS`` below.

Nothing here needs the network. Libraries are generated in the tests from an analytic rule,
and the air/vacuum measurement is exercised on spectra built with a known convention.
"""

import dataclasses
import gzip
import io
import urllib.error
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

import albireo as ab
from albireo import library as lib_mod
from albireo.library import (
    BoxInterpolator,
    SimplexInterpolator,
    SpectralLibrary,
    crossval_library,
    library_interpolator,
    line_core_medium,
)

RNG = np.random.default_rng(20260827)

TEFF = np.array([4000.0, 4250.0, 4500.0, 4750.0, 5000.0, 5250.0, 5500.0])
LOGG = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
MH = np.array([-1.0, -0.5, 0.0, 0.5])
WAVE = np.linspace(5150.0, 5250.0, 1200)
INTRINSIC_WIDTH = 0.25  # Angstrom; v sin i broadening is applied by a kernel, never here


def spectrum_rule(teff, logg, mh, wave=WAVE):
    """A smooth analytic stand-in for a synthetic spectrum.

    The tests that use it depend on two properties.

    Each label controls its own lines. If Teff and [M/H] both scaled a single depth they
    would be exactly interchangeable. A fit would then reach zero chi-square along a curve
    in label space without recovering the injected values, and such a degenerate fixture
    would leave real errors undetected. Here line 1 responds only to Teff, line 3 only to
    log g and line 5 only to [M/H], so the map from labels to spectrum is injective.

    The rule is nonlinear in every label, so multilinear interpolation is not exact and the
    cubic-versus-linear comparison is meaningful.
    """
    t = (teff - 4800.0) / 600.0
    g = logg - 4.0
    lines = (
        (5167.3, 0.30 + 0.13 * np.tanh(t)),  # temperature
        (5172.7, 0.22 - 0.11 * np.tanh(0.8 * t)),  # temperature, opposite sense
        (5183.6, 0.26 + 0.09 * g + 0.02 * g**2),  # gravity
        (5195.4, 0.17 - 0.07 * g),  # gravity, opposite sense
        (5205.9, 0.21 + 0.16 * mh + 0.04 * mh**2),  # metallicity
        (5227.2, 0.19 + 0.10 * mh - 0.03 * np.tanh(t)),  # mildly mixed
    )
    flux = np.ones_like(wave)
    for center, depth in lines:
        flux = flux - depth * np.exp(-0.5 * ((wave - center) / INTRINSIC_WIDTH) ** 2)
    log_continuum = 30.0 + 4.0 * np.log(teff / 5000.0) - 0.02 * (wave - wave[0]) / 100.0
    return flux, log_continuum


def build_library(teff=TEFF, logg=LOGG, mh=MH, medium="air", drop=()):
    nodes, normalized, continua = [], [], []
    for t in teff:
        for g in logg:
            for m in mh:
                if (t, g, m) in drop:
                    continue
                flux, log_continuum = spectrum_rule(t, g, m)
                nodes.append((t, g, m))
                normalized.append(flux)
                continua.append(log_continuum)
    return SpectralLibrary(
        label_names=("teff", "logg", "mh"),
        nodes=np.asarray(nodes),
        normalized=np.asarray(normalized),
        log_continuum=np.asarray(continua),
        wave=WAVE,
        medium=medium,
        meta={"grid": "toy", "citation": "generated in tests"},
    )


@pytest.fixture(scope="module")
def library():
    return build_library()


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------


def test_library_reports_its_geometry(library):
    assert library.n_nodes == TEFF.size * LOGG.size * MH.size
    assert library.n_pix == WAVE.size
    assert library.bounds["teff"] == (4000.0, 5500.0)
    axes = library.axes()
    assert axes is not None
    np.testing.assert_allclose(axes[0], TEFF)
    assert "complete box" in library.summary()


def test_resolving_power_is_read_from_the_metadata(library):
    """A published grid is already broadened, and the container must report to what (D65).

    ``resolving_power`` takes precedence over the BOSZ ingest's ``resolution``, and a
    library with neither is intrinsic. A value that is not a positive number is rejected,
    because reading it as intrinsic would broaden every template twice without warning.
    """
    assert library.resolving_power is None
    assert "intrinsic" in library.summary()
    bosz_like = library.replace(meta={**library.meta, "resolution": 20000})
    assert bosz_like.resolving_power == 20000.0
    assert "R = 20000" in bosz_like.summary()
    declared = library.replace(meta={"resolving_power": 115000.0, "resolution": 20000})
    assert declared.resolving_power == 115000.0
    # every transform keeps the metadata, so slicing and projection preserve the value
    assert bosz_like.sliced(5160.0, 5240.0).resolving_power == 20000.0
    assert bosz_like.in_medium("vacuum").resolving_power == 20000.0
    for bad in ("orig", -1.0, 0.0, float("nan")):
        with pytest.raises(ValueError, match="finite positive resolving power"):
            _ = library.replace(meta={"resolution": bad}).resolving_power
    # the registry fixes every BOSZ entry at R = 20,000, which the ingest records
    for name in lib_mod.library_names():
        info = lib_mod.library_info(name)
        if info["source"] == "bosz2024":
            assert info["fixed"]["resolution"] == 20000
    from albireo.simulate import synthetic_library

    assert synthetic_library(n_pix=64).resolving_power is None


def test_library_detects_irregular_coverage():
    punched = build_library(drop={(5500.0, 3.0, -1.0), (4000.0, 5.0, 0.5)})
    assert punched.axes() is None
    assert "irregular coverage" in punched.summary()


def test_medium_is_required_and_validated():
    with pytest.raises(ValueError, match="medium must be one of"):
        build_library(medium="Air")
    with pytest.raises(ValueError, match="medium must be one of"):
        build_library(medium=None)


def test_library_rejects_malformed_input(library):
    with pytest.raises(ValueError, match="strictly increasing"):
        library.replace(wave=WAVE[::-1])
    with pytest.raises(ValueError, match="duplicate"):
        library.replace(nodes=np.repeat(library.nodes[:1], library.n_nodes, axis=0))
    with pytest.raises(ValueError, match="finite"):
        library.replace(normalized=np.full_like(library.normalized, np.nan))
    with pytest.raises(ValueError, match="same shape"):
        library.replace(normalized=library.normalized[:, :10])
    with pytest.raises(ValueError, match="to match label_names"):
        library.replace(nodes=library.nodes[:, :2])


def test_in_medium_round_trips_against_the_grids_oracle(library):
    vacuum = library.in_medium("vacuum")
    assert vacuum.medium == "vacuum"
    np.testing.assert_allclose(vacuum.wave, np.asarray(ab.air_to_vacuum(WAVE)), rtol=0, atol=0)
    # ~1.4 A in the green, the class of error this conversion prevents
    assert 1.2 < float(np.mean(vacuum.wave - WAVE)) < 1.7
    np.testing.assert_allclose(vacuum.in_medium("air").wave, WAVE, atol=1e-9)
    assert library.in_medium("air") is library  # free to call unconditionally
    np.testing.assert_array_equal(vacuum.normalized, library.normalized)  # fluxes untouched


def test_sliced_keeps_a_margin(library):
    window = library.sliced(5180.0, 5200.0)
    assert window.wave[0] <= 5180.0 and window.wave[-1] >= 5200.0
    assert window.n_pix < library.n_pix
    with pytest.raises(ValueError, match="does not overlap"):
        library.sliced(6000.0, 6100.0)


def test_resampled_to_projects_onto_the_model_grid(library):
    grid = ab.LogGrid.from_wavelength_range(5170.0, 5230.0, dv_kms=8.0)
    projected = library.resampled_to(grid, medium="air")
    assert projected.n_pix == grid.n
    np.testing.assert_allclose(projected.wave, grid.wave)
    # a box average leaves a line-free stretch of continuum at exactly its own level
    continuum = projected.normalized[0][(grid.wave > 5187.5) & (grid.wave < 5192.5)]
    assert continuum.size > 3
    np.testing.assert_allclose(continuum, 1.0, atol=1e-6)
    with pytest.raises(ValueError, match="cannot cover"):
        library.resampled_to(ab.LogGrid.from_wavelength_range(5000.0, 5400.0, 8.0), medium="air")


def test_resampled_to_converts_medium_before_rebinning(library):
    grid = ab.LogGrid.from_wavelength_range(5175.0, 5225.0, dv_kms=6.0)
    vacuum = library.resampled_to(grid, medium="vacuum")
    assert vacuum.medium == "vacuum"
    # the same feature must be ~1.4 A apart on the two scales
    air = library.resampled_to(grid, medium="air")
    offset = (
        grid.wave[int(np.argmin(vacuum.normalized[0]))]
        - grid.wave[int(np.argmin(air.normalized[0]))]
    )
    assert 1.0 < offset < 1.9


# ---------------------------------------------------------------------------
# box interpolation
# ---------------------------------------------------------------------------


def test_box_linear_matches_scipy_regular_grid(library):
    interpolator = library_interpolator(library, method="linear")
    assert isinstance(interpolator, BoxInterpolator)
    values = np.stack([spectrum_rule(t, g, m)[0] for t in TEFF for g in LOGG for m in MH]).reshape(
        TEFF.size, LOGG.size, MH.size, WAVE.size
    )
    oracle = RegularGridInterpolator((TEFF, LOGG, MH), values)
    points = np.column_stack(
        [
            RNG.uniform(4000.0, 5500.0, 25),
            RNG.uniform(3.0, 5.0, 25),
            RNG.uniform(-1.0, 0.5, 25),
        ]
    )
    got = np.asarray(jax.jit(jax.vmap(interpolator))(points)[0])
    np.testing.assert_allclose(got, oracle(points), rtol=1e-12, atol=1e-13)


def test_box_cubic_matches_an_independent_catmull_rom(library):
    interpolator = library_interpolator(library, method="cubic")

    def catmull_rom_1d(axis, values, x):
        """Catmull-Rom with the phantom end nodes extrapolated linearly, in plain NumPy."""
        n = axis.size
        i = min(max(int(np.searchsorted(axis, x, side="right") - 1), 0), n - 2)
        t = (x - axis[i]) / (axis[i + 1] - axis[i])
        w = np.array(
            [
                -0.5 * t**3 + t**2 - 0.5 * t,
                1.5 * t**3 - 2.5 * t**2 + 1.0,
                -1.5 * t**3 + 2 * t**2 + 0.5 * t,
                0.5 * t**3 - 0.5 * t**2,
            ]
        )
        if i == 0:  # f(-1) := 2 f(0) - f(1)
            w[1] += 2 * w[0]
            w[2] -= w[0]
            w[0] = 0.0
        if i == n - 2:  # f(n) := 2 f(n-1) - f(n-2)
            w[2] += 2 * w[3]
            w[1] -= w[3]
            w[3] = 0.0
        idx = [max(i - 1, 0), i, i + 1, min(i + 2, n - 1)]
        return np.tensordot(w, values[idx], axes=(0, 0))

    values = np.stack([spectrum_rule(t, g, m)[0] for t in TEFF for g in LOGG for m in MH]).reshape(
        TEFF.size, LOGG.size, MH.size, WAVE.size
    )
    for point in ((4380.0, 3.7, -0.2), (5111.0, 4.9, 0.31), (4020.0, 3.05, -0.98)):
        step = catmull_rom_1d(TEFF, values, point[0])
        step = catmull_rom_1d(LOGG, step, point[1])
        expected = catmull_rom_1d(MH, step, point[2])
        got = np.asarray(interpolator(jnp.asarray(point))[0])
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-13)


# The simplex path reproduces a node to rounding, not bit-for-bit. Its weights come from
# Qhull's affine transforms, so at a vertex they are 1 - eps and eps rather than 1 and 0, and
# the error is eps times the spread of the rows across the simplex. The bound is written in
# units of eps * max(|y|, 1), which is between one and two true ulp of y. Over 200
# triangulations of each test grid (node order permuted; 105,600 node evaluations), a quarter
# were inexact and the worst was 1.075 in these units. Four leaves a 3.7x margin on grids
# this smooth. A rougher library would need more, which would be a property of that library.
NODE_TOL_EPS = 4


def _assert_reproduced_to_rounding(got, want):
    tol = NODE_TOL_EPS * np.finfo(np.float64).eps * np.maximum(np.abs(want), 1.0)
    assert np.all(np.abs(np.asarray(got) - want) <= tol)


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_box_interpolation_at_a_node_is_bit_exact(library, method):
    """At a node the weights are exactly 1 and 0, so the node is returned bit-for-bit."""
    interpolator = library_interpolator(library, method=method)
    for index in (0, 37, library.n_nodes - 1):
        normalized, log_continuum = interpolator(jnp.asarray(library.nodes[index]))
        assert np.array_equal(np.asarray(normalized), library.normalized[index])
        assert np.array_equal(np.asarray(log_continuum), library.log_continuum[index])


def test_simplex_interpolation_at_a_node_is_exact_to_rounding(library):
    """The scattered path is accurate to a few ulp, not bit-exact; see ``NODE_TOL_EPS``."""
    interpolator = library_interpolator(library, method="simplex")
    for index in (0, 37, library.n_nodes - 1):
        normalized, log_continuum = interpolator(jnp.asarray(library.nodes[index]))
        _assert_reproduced_to_rounding(normalized, library.normalized[index])
        _assert_reproduced_to_rounding(log_continuum, library.log_continuum[index])


@pytest.mark.parametrize("fixture", ["library", "punched_library"])
def test_simplex_node_reproduction_does_not_depend_on_the_triangulation(request, fixture):
    """Every node is reproduced to rounding under three triangulations of each grid.

    Which nodes are returned bit-exact is a property of the triangulation Qhull chose, and
    that choice differs between scipy builds: a Linux CI runner and a Windows desktop
    differed. Permuting the node order changes the triangulation the same way, so this
    exercises the guaranteed invariant on every platform. The punched grid is the one that
    reaches the simplex path through ``method="auto"``.
    """
    base = request.getfixturevalue(fixture)
    rng = np.random.default_rng(20260902)
    for _ in range(3):
        perm = rng.permutation(base.n_nodes)
        permuted = base.replace(
            nodes=base.nodes[perm],
            normalized=base.normalized[perm],
            log_continuum=base.log_continuum[perm],
        )
        interpolator = library_interpolator(permuted, method="simplex")
        got, got_lc = jax.jit(jax.vmap(interpolator))(jnp.asarray(permuted.nodes))
        _assert_reproduced_to_rounding(got, permuted.normalized)
        _assert_reproduced_to_rounding(got_lc, permuted.log_continuum)


def test_cubic_beats_linear_on_the_toy_grid(library):
    """The cubic has the smaller error here, which justifies 4^k taps instead of 2^k."""
    coarse = build_library(teff=TEFF[::2], logg=LOGG, mh=MH)
    errors = {}
    for method in ("linear", "cubic"):
        interpolator = library_interpolator(coarse, method=method)
        probes = np.column_stack(
            [RNG.uniform(4000.0, 5500.0, 40), RNG.uniform(3.0, 5.0, 40), RNG.uniform(-1.0, 0.5, 40)]
        )
        got = np.asarray(jax.jit(jax.vmap(interpolator))(probes)[0])
        truth = np.stack([spectrum_rule(*p)[0] for p in probes])
        errors[method] = float(np.sqrt(np.mean((got - truth) ** 2)))
    assert errors["cubic"] < errors["linear"]


def test_box_interpolator_gradient_matches_finite_differences(library):
    interpolator = library_interpolator(library)
    point = jnp.asarray([4712.0, 3.83, -0.17])

    def scalar(labels):
        return jnp.sum(interpolator(labels)[0] ** 2)

    analytic = np.asarray(jax.jit(jax.grad(scalar))(point))
    scale = float(np.max(np.abs(analytic)))
    for axis, step in enumerate((1e-3, 1e-6, 1e-6)):
        bump = jnp.asarray(np.eye(3)[axis] * step)
        numeric = (float(scalar(point + bump)) - float(scalar(point - bump))) / (2 * step)
        # atol against the largest component: a near-zero partial cannot be tested to a
        # relative tolerance by finite differences.
        np.testing.assert_allclose(analytic[axis], numeric, rtol=1e-5, atol=1e-6 * scale)


def test_box_hull_margin_signs(library):
    interpolator = library_interpolator(library)
    assert float(interpolator.hull_margin(jnp.asarray([4700.0, 4.0, -0.2]))) > 0
    assert float(interpolator.hull_margin(jnp.asarray([6000.0, 4.0, -0.2]))) < 0
    assert float(interpolator.hull_margin(jnp.asarray([4000.0, 4.0, -0.2]))) == 0.0


# ---------------------------------------------------------------------------
# scattered interpolation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def punched_library():
    """A grid with its corners removed, as they are in real OB libraries for physical reasons."""
    drop = {
        (t, g, m)
        for t in TEFF
        for g in LOGG
        for m in MH
        if (t >= 5250.0 and g <= 3.0) or (t <= 4250.0 and g >= 5.0)
    }
    return build_library(drop=drop)


def test_simplex_matches_scipy_linear_nd(punched_library):
    interpolator = library_interpolator(punched_library)
    assert isinstance(interpolator, SimplexInterpolator)
    origin = punched_library.nodes.min(axis=0)
    span = punched_library.nodes.max(axis=0) - origin
    oracle = LinearNDInterpolator(
        (punched_library.nodes - origin) / span, punched_library.normalized
    )
    probes = np.column_stack(
        [RNG.uniform(4300.0, 5200.0, 30), RNG.uniform(3.6, 4.4, 30), RNG.uniform(-0.9, 0.4, 30)]
    )
    expected = oracle((probes - origin) / span)
    assert np.all(np.isfinite(expected))  # probes chosen inside the hull
    got = np.asarray(jax.jit(jax.vmap(interpolator))(probes)[0])
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-11)


def test_simplex_hull_margin_flags_the_punched_corner(punched_library):
    interpolator = library_interpolator(punched_library)
    # strictly inside a simplex
    assert float(interpolator.hull_margin(jnp.asarray([4712.0, 3.87, -0.23]))) > 0
    # exactly on a shared facet: inside the hull, but with no margin. The contract is
    # ">= 0 inside", and a node coordinate puts the point on a lattice hyperplane. The margin
    # is zero to rounding rather than exactly. The barycentric coordinates are Qhull's affine
    # transform applied to the point, and at a vertex they are in error by ~1e-16 of either
    # sign (measured -2.2e-16 at worst over 400 triangulations of the test grids). The
    # tolerance below is the one ``crossval_library`` uses for the same decision.
    margin = float(interpolator.hull_margin(jnp.asarray([4700.0, 4.0, -0.2])))
    assert abs(margin) <= lib_mod.HULL_MARGIN_TOL
    assert float(interpolator.hull_margin(jnp.asarray([5500.0, 3.0, -1.0]))) < 0
    # outside the hull it extrapolates flat rather than diverging
    outside = np.asarray(interpolator(jnp.asarray([5500.0, 3.0, -1.0]))[0])
    assert np.all(np.isfinite(outside)) and outside.max() < 1.5


def test_simplex_gradient_is_finite(punched_library):
    interpolator = library_interpolator(punched_library)

    def scalar(labels):
        return jnp.sum(interpolator(labels)[0] ** 2)

    grad = np.asarray(jax.jit(jax.grad(scalar))(jnp.asarray([4700.0, 4.0, -0.2])))
    assert np.all(np.isfinite(grad)) and np.any(grad != 0.0)


def test_forcing_a_box_method_on_scattered_nodes_is_refused(punched_library):
    with pytest.raises(ValueError, match="complete axis-product grid"):
        library_interpolator(punched_library, method="cubic")
    with pytest.raises(ValueError, match="auto, linear, cubic or simplex"):
        library_interpolator(punched_library, method="quadratic")


# ---------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------


def test_crossval_reports_error_at_doubled_spacing(library):
    report = crossval_library(library)
    assert report["spacing"] == "doubled"
    assert report["n_tested"] > 0
    assert 0.0 < report["rms"] < 0.05
    assert report["max"] >= report["p95"] >= report["median"]


def test_crossval_prefers_the_cubic(library):
    linear = crossval_library(library, method="linear")
    cubic = crossval_library(library, method="cubic")
    assert cubic["rms"] < linear["rms"]


def test_crossval_handles_scattered_coverage(punched_library):
    report = crossval_library(punched_library, seed=3)
    assert report["spacing"] == "scattered"
    assert report["n_tested"] > 0
    assert np.isfinite(report["rms"])


def test_crossval_refuses_a_grid_too_small_to_split():
    tiny = build_library(teff=TEFF[:2], logg=LOGG[:2], mh=MH[:1])
    with pytest.raises(ValueError, match=r"too few nodes|nothing to measure"):
        crossval_library(tiny)


# ---------------------------------------------------------------------------
# wavelength medium, measured
# ---------------------------------------------------------------------------


def medium_test_spectrum(medium, n=6000):
    """A spectrum with absorption at the reference lines, on a declared scale."""
    wave = np.linspace(4050.0, 6650.0, n)
    flux = np.ones_like(wave)
    for _, vac in ab.library._MEDIUM_LINES:
        center = vac if medium == "vacuum" else float(ab.vacuum_to_air(vac))
        flux -= 0.6 * np.exp(-0.5 * ((wave - center) / 0.35) ** 2)
    return wave, flux


@pytest.mark.parametrize("medium", ["air", "vacuum"])
def test_line_core_medium_recovers_a_known_convention(medium):
    """The same code returns the correct medium for spectra built on either scale."""
    verdict = line_core_medium(*medium_test_spectrum(medium))
    assert verdict["medium"] == medium
    assert verdict["n_lines"] >= 5
    assert verdict["ratio"] > 20.0


def test_line_core_medium_refuses_when_it_cannot_tell():
    wave = np.linspace(4050.0, 6650.0, 4000)
    with pytest.raises(ValueError, match="not decisive"):
        line_core_medium(wave, np.ones_like(wave))
    # too narrow a span to contain two reference lines
    narrow = np.linspace(5180.0, 5190.0, 200)
    with pytest.raises(ValueError, match="at least two reference lines"):
        line_core_medium(narrow, np.ones_like(narrow))


def test_line_core_medium_validates_input_shape():
    with pytest.raises(ValueError, match="same length"):
        line_core_medium(np.linspace(4000.0, 6000.0, 10), np.ones(9))


# ---------------------------------------------------------------------------
# tracing contract
# ---------------------------------------------------------------------------


def test_interpolators_survive_a_jit_boundary_as_arguments(library, punched_library):
    """They must pass as traced model arguments, not be embedded as constants (D27)."""

    @jax.jit
    def evaluate(interpolator, labels):
        return interpolator(labels)[0]

    for lib in (library, punched_library):
        interpolator = library_interpolator(lib)
        out = evaluate(interpolator, jnp.asarray([4700.0, 4.0, -0.2]))
        assert out.shape == (lib.n_pix,)
        assert out.dtype == jnp.float64  # the round trip must preserve x64


# ---------------------------------------------------------------------------
# the registry: naming, caching, and downloads that never touch the network
# ---------------------------------------------------------------------------
#
# Everything checkable offline is checked offline, as in tests/test_archive.py: with a fake
# transport, not a recorded cassette. The BOSZ URL facts were confirmed against the live
# archive on 2026-08-27 and are asserted here so that a silent change upstream appears as
# a failing test rather than as a wrong spectrum.


def _fake_bosz_shard(n_pix, seed=0):
    """Two whitespace columns, flux and continuum, gzipped, which is the real BOSZ layout."""
    rng = np.random.default_rng(seed)
    continuum = 1e6 * np.exp(-0.3 * np.linspace(0.0, 1.0, n_pix))
    flux = continuum * (1.0 - 0.4 * rng.random(n_pix))
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as handle:
        np.savetxt(handle, np.column_stack([flux, continuum]), fmt="%.6e")
    return buffer.getvalue()


@pytest.fixture
def offline_registry(monkeypatch, tmp_path):
    """A tiny BOSZ-shaped library whose downloads are served from memory."""
    monkeypatch.setenv("ALBIREO_DATA_DIR", str(tmp_path))
    wave = np.linspace(4000.0, 7000.0, 400)
    entry = dataclasses.replace(
        lib_mod._LIBRARIES["bosz2024-fgk-r20000"],
        name="test-grid",
        axes={"teff": [5000.0, 5250.0], "logg": [4.0, 4.5], "mh": [0.0, 0.25]},
        download_mb=1.0,
    )
    monkeypatch.setitem(lib_mod._LIBRARIES, "test-grid", entry)

    calls: list[str] = []

    def fake_download(url, destination, attempts=4):
        calls.append(url)
        if url.endswith(".txt"):  # the shared wavelength grid
            destination.write_text("\n".join(f"{w:.7e}" for w in wave))
        else:
            destination.write_bytes(_fake_bosz_shard(wave.size, seed=len(calls)))

    monkeypatch.setattr(lib_mod, "_download_with_retries", fake_download)
    # The medium check needs real line cores. A random spectrum has none, so the check
    # is not decisive, which is the branch this fixture exercises.
    return entry, calls


@pytest.fixture
def gapped_registry(monkeypatch, tmp_path):
    """A BOSZ-shaped library with one interior node unpublished (the archive returns 404)."""
    monkeypatch.setenv("ALBIREO_DATA_DIR", str(tmp_path))
    wave = np.linspace(4000.0, 7000.0, 400)
    entry = dataclasses.replace(
        lib_mod._LIBRARIES["bosz2024-fgk-r20000"],
        name="gapped-grid",
        axes={"teff": [5000.0, 5250.0], "logg": [4.0, 4.5], "mh": [0.0, 0.25, 0.5]},
        download_mb=1.0,
    )
    monkeypatch.setitem(lib_mod._LIBRARIES, "gapped-grid", entry)
    fixed = entry.fixed
    gap = lib_mod._bosz_url(
        5000.0,
        4.0,
        0.25,
        alpha=fixed["alpha"],
        carbon=fixed["carbon"],
        vmicro=fixed["vmicro"],
        resolution=int(fixed["resolution"]),
    )
    corner = lib_mod._bosz_url(
        5250.0,
        4.5,
        0.5,
        alpha=fixed["alpha"],
        carbon=fixed["carbon"],
        vmicro=fixed["vmicro"],
        resolution=int(fixed["resolution"]),
    )
    calls: list[str] = []

    def fake_download(url, destination, attempts=4):
        calls.append(url)
        if url in (gap, corner):
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        if url.endswith(".txt"):
            destination.write_text("\n".join(f"{w:.7e}" for w in wave))
        else:
            destination.write_bytes(_fake_bosz_shard(wave.size, seed=len(calls)))

    monkeypatch.setattr(lib_mod, "_download_with_retries", fake_download)
    return entry, calls


def test_a_bracketed_gap_is_filled_and_an_unbracketed_one_dropped(gapped_registry):
    """The interior gap is filled, keeping the box complete. The unbracketed corner is dropped."""
    library = ab.fetch_library("gapped-grid", progress=False)
    # Twelve nodes are requested and two are unpublished. The interior one is filled and
    # the corner is dropped.
    assert library.n_nodes == 11
    assert library.meta["n_missing"] == 2
    assert library.meta["n_filled"] == 1 and library.meta["n_dropped"] == 1
    (filled,) = library.meta["filled_nodes"]
    assert (filled["teff"], filled["logg"], filled["mh"]) == (5000.0, 4.0, 0.25)
    assert filled["axis"] == "mh" and filled["weight"] == pytest.approx(0.5)
    assert filled["between"] == [[5000.0, 4.0, 0.0], [5000.0, 4.0, 0.5]]
    assert "filled_note" in library.meta and "missing_note" in library.meta
    assert "filled" in library.summary() and "1 node(s) by interpolation" in library.summary()

    def row(teff, logg, mh):
        hits = np.flatnonzero(
            (library.nodes[:, 0] == teff)
            & (library.nodes[:, 1] == logg)
            & (library.nodes[:, 2] == mh)
        )
        assert hits.size == 1
        return int(hits[0])

    filled_row = row(5000.0, 4.0, 0.25)
    lo, hi = row(5000.0, 4.0, 0.0), row(5000.0, 4.0, 0.5)
    np.testing.assert_allclose(
        library.normalized[filled_row], 0.5 * (library.normalized[lo] + library.normalized[hi])
    )
    np.testing.assert_allclose(
        library.log_continuum[filled_row],
        0.5 * (library.log_continuum[lo] + library.log_continuum[hi]),
    )
    # The dropped corner leaves the box irregular here. On the real grid, whose only gap is
    # interior, the fill restores the cubic.
    assert library.axes() is None
    cached = ab.fetch_library("gapped-grid", progress=False)
    assert cached.meta["filled_nodes"] == library.meta["filled_nodes"]


def test_bracketing_prefers_metallicity_then_gravity_then_temperature():
    nodes = [
        (5000.0, 4.0, 0.0),
        (5000.0, 4.0, 0.5),
        (5000.0, 3.5, 0.25),
        (5000.0, 4.5, 0.25),
        (4750.0, 4.0, 0.25),
        (5250.0, 4.0, 0.25),
        (5000.0, 4.0, 0.25),  # the gap, index 6
    ]
    names = ("teff", "logg", "mh")
    present = {lib_mod._node_key(n): i for i, n in enumerate(nodes) if i != 6}
    assert lib_mod._bracketing_neighbours(nodes, 6, present, names) == (0, 1, 0.5, "mh")
    without_mh = {k: v for k, v in present.items() if v not in (0, 1)}
    assert lib_mod._bracketing_neighbours(nodes, 6, without_mh, names) == (2, 3, 0.5, "logg")
    only_teff = {k: v for k, v in present.items() if v in (4, 5)}
    assert lib_mod._bracketing_neighbours(nodes, 6, only_teff, names) == (4, 5, 0.5, "teff")
    assert lib_mod._bracketing_neighbours(nodes, 6, {}, names) is None
    # An edge node has no bracket along an axis it bounds.
    edge = [*nodes, (5000.0, 4.0, -0.5)]
    assert lib_mod._bracketing_neighbours(edge, 7, present, names) is None


def test_the_registry_lists_what_it_can_build():
    names = ab.library_names()
    assert "bosz2024-fgk-r20000" in names
    assert {"bosz2024-hot-r20000", "bosz2024-hot-rvs"} <= set(names)
    assert names == sorted(names)


@pytest.mark.parametrize("name", ["bosz2024-hot-rvs", "bosz2024-hot-r20000"])
def test_the_hot_box_is_registered_as_the_archive_publishes_it(name):
    """Both hot entries have the same 364 nodes and the same shards, in two bands.

    The numbers are the archive's, measured on its index on 2026-09-10, so this test detects
    a silent change to the registered box.
    """
    entry = lib_mod._LIBRARIES[name]
    assert isinstance(entry, lib_mod._Library)
    assert entry.name == name
    assert entry.source == "bosz2024" and entry.medium == "air"
    assert entry.axes is lib_mod._BOSZ_HOT_AXES
    assert entry.fixed is lib_mod._BOSZ_FIXED
    assert set(entry.axes) == set(entry.label_names)

    info = ab.library_info(name)
    assert info["n_nodes"] == 364
    assert (info["axes"]["teff"][0], info["axes"]["teff"][-1]) == (7000.0, 10000.0)
    assert info["axes"]["logg"] == [3.5, 4.0, 4.5, 5.0]
    assert info["axes"]["mh"] == [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
    # Every node is published, so nothing is filled and nothing is dropped. The box is
    # complete and the cubic applies with no caveat about an interpolated node.
    assert info["known_gaps"] == []
    assert info["download_mb"] == 532.0
    assert info["wave_range"] == ((8350.0, 8850.0) if name.endswith("rvs") else (4000.0, 7000.0))
    assert any("MARCS/ATLAS9 boundary" in c for c in info["caveats"])
    assert any("Paschen" in c for c in info["caveats"])
    # Only the RVS entry is used with a vacuum-scale instrument.
    assert any("vacuum" in c for c in info["caveats"]) is name.endswith("rvs")


@pytest.mark.parametrize("name", ab.library_names())
def test_every_entry_carries_its_provenance(name):
    """A library without a licence and a citation is not redistributable or publishable."""
    info = ab.library_info(name)
    for key in ("description", "licence", "citation", "medium", "wave_range", "upstream_note"):
        assert info[key], f"{name} is missing {key}"
    assert info["wave_range"][0] < info["wave_range"][1]
    assert info["caveats"], f"{name} records no caveats, which is never true of a real grid"


def test_an_unknown_name_lists_the_known_ones():
    with pytest.raises(KeyError, match="bosz2024-fgk-r20000"):
        ab.library_info("no-such-grid")


def test_bosz_urls_match_the_archive():
    """The URL was confirmed against files fetched from MAST on 2026-08-27.

    Two of its parts are not what a careful reading of the documentation would give: Teff
    is not zero-padded, and the atmosphere code changes across the grid. An error in either
    gives a 404.
    """
    url = lib_mod._bosz_url(6000.0, 4.0, 0.0, alpha=0.0, carbon=0.0, vmicro=2, resolution=20000)
    assert url == (
        "https://archive.stsci.edu/hlsps/bosz/bosz2024/r20000/m+0.00/"
        "bosz2024_mp_t6000_g+4.0_m+0.00_a+0.00_c+0.00_v2_r20000_resam.txt.gz"
    )
    assert "_t6000_" in url and "_t06000_" not in url


@pytest.mark.parametrize(
    ("teff", "logg", "code"),
    [
        (4000.0, 3.0, "ms"),  # MARCS spherical below log g 3.5
        (4000.0, 3.5, "mp"),  # plane-parallel at and above it
        (7000.0, 4.5, "mp"),
        (7750.0, 4.0, "mp"),  # inside the 7500-8000 overlap, where both families exist
        (8000.0, 4.0, "mp"),  # MARCS is used in the 7500-8000 overlap, so one family throughout
        (8250.0, 4.0, "ap"),  # the first ATLAS9 node of the hot box
        (9000.0, 4.0, "ap"),  # ATLAS9 above it
    ],
)
def test_the_atmosphere_code_follows_the_archive(teff, logg, code):
    assert lib_mod._bosz_atmosphere(teff, logg) == code


def test_the_hot_box_crosses_the_atmosphere_seam_where_its_caveat_says():
    """There is one crossing, between 8000 and 8250 K, and the caveat quotes both numbers.

    The hot box is the first registered library to span both model-atmosphere families, so
    the location of the change of code is part of what the entry documents. It is stated in
    prose in the caveats and computed here.
    """
    axis = lib_mod._BOSZ_HOT_AXES["teff"]
    families = [lib_mod._bosz_atmosphere(teff, 4.0) for teff in axis]
    assert families == ["mp"] * 5 + ["ap"] * 8
    first_atlas = families.index("ap")
    assert (axis[first_atlas - 1], axis[first_atlas]) == (8000.0, 8250.0)
    seam = next(c for c in lib_mod._BOSZ_HOT_CAVEATS if "MARCS/ATLAS9 boundary" in c)
    assert "between 8000 and 8250 K" in seam


def test_every_registered_bosz_box_has_uniformly_spaced_axes():
    """Catmull-Rom assumes equal spacing, so a registered box must supply it.

    ``_axis_weights`` uses the spline in its uniform-parameter form: the tangent at a node
    is estimated from the two neighbours as if they were equally spaced. Across a change of
    step that estimate is wrong, and the interpolant loses linear reproduction there. BOSZ's
    temperature axis changes step twice, 100 K below 4000 K and 500 K above 12,000 K. A
    registered box is therefore in part a choice of a range over which the step is constant,
    and this test asserts that choice.
    """
    boxes = [entry for entry in lib_mod._LIBRARIES.values() if entry.source == "bosz2024"]
    assert len(boxes) >= 4
    for entry in boxes:
        for label, values in entry.axes.items():
            steps = np.diff(np.asarray(values, dtype=float))
            assert steps.size >= 1, f"{entry.name}: the {label} axis has a single node"
            np.testing.assert_allclose(
                steps,
                steps[0],
                rtol=0.0,
                atol=1e-9,
                err_msg=f"{entry.name}: the {label} axis is not uniformly spaced",
            )


HOT_SHARDS = Path(__file__).resolve().parent / "data" / "bosz2024_hot_shards.txt"


def test_every_node_the_hot_box_requests_is_published():
    """The 364 names the box would fetch are in the archive index harvested 2026-09-10.

    The fixture is the harvested listing rather than a regeneration of the axes. It holds
    every published metallicity over the box's temperature and gravity range, fourteen
    against the box's seven, including the hole at (7750 K, log g +4.0, [M/H] -1.25) that
    the box's [M/H] axis avoids. A change to the axes, to the atmosphere dispatch or to
    the file-name convention therefore appears here as a name that is absent from the
    archive, rather than as a 404 partway through a 532 MB build.
    """
    lines = HOT_SHARDS.read_text(encoding="utf-8").splitlines()
    published = {line for line in lines if line and not line.startswith("#")}
    assert len(published) == 895

    axes = lib_mod._BOSZ_HOT_AXES
    requested = {
        lib_mod._bosz_filename(teff, logg, mh, **lib_mod._BOSZ_FIXED)
        for teff in axes["teff"]
        for logg in axes["logg"]
        for mh in axes["mh"]
    }
    assert len(requested) == 364
    assert requested <= published, sorted(requested - published)[:5]

    # This pair shows that the listing is a harvest and not a product of the axes.
    stem = "bosz2024_mp_t7750_g+4.0_m{}_a+0.00_c+0.00_v2_r20000_resam.txt.gz"
    assert stem.format("-1.25") not in published
    assert stem.format("-1.00") in published


def test_a_build_downloads_once_and_is_cached(offline_registry):
    _, calls = offline_registry
    first = ab.fetch_library("test-grid", progress=False)
    assert first.nodes.shape == (8, 3)
    assert len(calls) == 9  # eight nodes and one shared wavelength grid

    second = ab.fetch_library("test-grid", progress=False)
    assert len(calls) == 9, "a cache hit must not touch the network"
    # The results are bit-identical. The build path reads back what it wrote, so a warm
    # cache and a cold one cannot return different precision.
    assert np.array_equal(first.normalized, second.normalized)
    assert np.array_equal(first.log_continuum, second.log_continuum)


def test_the_build_is_reproducible_across_machines(offline_registry):
    """The digest is over the arrays, not the file, so npz framing cannot change it."""
    built = ab.fetch_library("test-grid", progress=False)
    digest = built.meta["content_sha256"]
    assert digest == lib_mod._content_digest(built)

    path = lib_mod._library_cache_path(lib_mod._LIBRARIES["test-grid"])
    reloaded = lib_mod.load_library(path)
    assert lib_mod._content_digest(reloaded) == digest


def test_a_corrupted_cache_is_caught_and_named(offline_registry):
    ab.fetch_library("test-grid", progress=False)
    path = lib_mod._library_cache_path(lib_mod._LIBRARIES["test-grid"])
    with np.load(path, allow_pickle=True) as handle:
        contents = {key: handle[key] for key in handle}
    contents["normalized"] = contents["normalized"] * 1.01
    np.savez_compressed(path, **contents)

    with pytest.raises(RuntimeError, match="clear_library_cache"):
        ab.fetch_library("test-grid", progress=False)


def test_an_interrupted_download_never_becomes_a_cache_entry(offline_registry, monkeypatch):
    def explode(url, destination, attempts=4):
        destination.write_bytes(b"half a fi")
        raise RuntimeError("connection reset")

    monkeypatch.setattr(lib_mod, "_download_with_retries", explode)
    with pytest.raises(RuntimeError, match="connection reset"):
        ab.fetch_library("test-grid", progress=False)

    path = lib_mod._library_cache_path(lib_mod._LIBRARIES["test-grid"])
    assert not path.is_file()
    raw = lib_mod._raw_dir()
    assert not list(raw.glob("*.gz")), "a partial transfer was left where a build would use it"


def test_a_band_can_be_narrowed_but_not_widened(offline_registry):
    inside = ab.fetch_library("test-grid", wave_range=(5000.0, 6000.0), progress=False)
    assert inside.wave.min() >= 5000.0 and inside.wave.max() <= 6000.0

    with pytest.raises(ValueError, match="was built over"):
        ab.fetch_library("test-grid", wave_range=(3000.0, 9000.0), progress=False)


def test_clearing_the_cache_keeps_the_raw_shards(offline_registry):
    ab.fetch_library("test-grid", progress=False)
    raw_before = sorted(p.name for p in lib_mod._raw_dir().glob("*"))
    assert raw_before

    removed = ab.clear_library_cache("test-grid")
    assert removed and not lib_mod._library_cache_path(lib_mod._LIBRARIES["test-grid"]).is_file()
    # Another band can be sliced from the raw shards without downloading them again.
    assert sorted(p.name for p in lib_mod._raw_dir().glob("*")) == raw_before

    assert ab.clear_library_cache("_raw")
    assert not list(lib_mod._raw_dir().glob("*"))


def test_a_declared_medium_is_checked_against_the_spectra(offline_registry, monkeypatch):
    """BOSZ changed convention between 2017 and 2024 under one name. The build therefore
    measures the medium and raises an error if it disagrees with the declared one."""
    entry = lib_mod._LIBRARIES["test-grid"]
    monkeypatch.setitem(
        lib_mod._LIBRARIES, "test-grid", dataclasses.replace(entry, medium="vacuum")
    )
    monkeypatch.setattr(
        lib_mod,
        "line_core_medium",
        lambda *a, **k: {"medium": "air", "ratio": 120.0, "n_lines": 6, "residuals": {}},
    )
    with pytest.raises(ValueError, match="upstream convention has moved"):
        ab.fetch_library("test-grid", progress=False)


def test_a_two_column_file_is_required(tmp_path):
    """The reader states the format it expects rather than silently taking column 0."""
    path = tmp_path / "wrong.txt.gz"
    with gzip.open(path, "wt") as handle:
        np.savetxt(handle, np.ones((10, 3)))
    with pytest.raises(ValueError, match="two columns"):
        lib_mod._read_bosz_shard(path, np.arange(10))


def test_pollux_refuses_rather_than_guessing_a_format():
    """No parser is shipped for a file format that has not been examined."""
    with pytest.raises(NotImplementedError, match="pollux"):
        ab.ingest_pollux(None)


def test_a_single_valued_axis_is_constant_rather_than_nan():
    """A library sliced to one metallicity keeps its column.

    Unless this case is handled, the degenerate cell has lo == hi and the resulting 0/0
    gives a NaN at every pixel. No error is raised, and the only symptom is a fit that
    fails to converge.
    """
    wave = np.linspace(5000.0, 5010.0, 40)
    nodes = [(t, g, 0.0) for t in (6000.0, 6250.0) for g in (4.0, 4.5)]
    lib = SpectralLibrary(
        label_names=("teff", "logg", "mh"),
        nodes=np.asarray(nodes),
        normalized=np.full((4, wave.size), 0.9),
        log_continuum=np.zeros((4, wave.size)),
        wave=wave,
        medium="air",
    )
    out = np.asarray(library_interpolator(lib)(jnp.asarray([6100.0, 4.2, 0.0]))[0])
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, 0.9)


def test_saving_and_loading_round_trips(tmp_path):
    wave = np.linspace(4500.0, 4600.0, 64)
    rng = np.random.default_rng(3)
    lib = SpectralLibrary(
        label_names=("teff", "logg"),
        nodes=np.asarray([(6000.0, 4.0), (6250.0, 4.0), (6000.0, 4.5), (6250.0, 4.5)]),
        normalized=rng.random((4, wave.size)),
        log_continuum=rng.random((4, wave.size)),
        wave=wave,
        medium="vacuum",
        meta={"grid": "unit-test", "citation": "nobody 2026"},
    )
    path = lib_mod.save_library(lib, tmp_path / "round-trip.npz")
    back = lib_mod.load_library(path)

    assert back.label_names == lib.label_names
    assert back.medium == "vacuum"
    assert back.meta["citation"] == "nobody 2026"
    assert back.normalized.dtype == np.float64  # stored as float32, restored to x64
    np.testing.assert_array_equal(back.nodes, lib.nodes)
    np.testing.assert_allclose(back.normalized, lib.normalized, rtol=1e-6)


@pytest.mark.network
def test_the_archive_still_serves_what_the_registry_expects():
    """One live request checks the two facts that a silent upstream change would break.

    The request is small: it fetches headers for a single shard and no spectrum.
    """
    import urllib.request

    url = lib_mod._bosz_url(6000.0, 4.0, 0.0, alpha=0.0, carbon=0.0, vmicro=2, resolution=20000)
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "albireo test"})
    with urllib.request.urlopen(request, timeout=60) as response:
        assert response.status == 200
        assert int(response.headers["Content-Length"]) > 100_000
