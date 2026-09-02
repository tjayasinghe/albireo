"""Tests for synthetic spectral libraries and their differentiable interpolators.

Interpolators are checked against scipy's own implementations — ``RegularGridInterpolator``
for the box, ``LinearNDInterpolator`` for the scattered case — so the oracle is independent
code rather than a second copy of the same arithmetic. Node reproduction is the invariant
that lets the warm-start node scan and the continuous fit be compared on equal terms: the
box interpolators return the stored spectrum bit-for-bit, and the simplex interpolator
returns it to rounding, for the reason recorded at ``NODE_TOL_EPS`` below.

Nothing here needs the network. Libraries are generated in-test from an analytic rule, and
the air/vacuum measurement is exercised on spectra built with a known convention.
"""

import dataclasses
import gzip
import io

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

    Two properties are deliberate and load-bearing for the tests that use it.

    *Each label drives its own lines.* An earlier version let Teff and [M/H] both scale a
    single depth, which makes them exactly interchangeable — a fit then drives the
    chi-square to zero on a curve through label space and "fails" to recover the injected
    values while fitting perfectly. That is a degenerate fixture, not a broken fitter, and
    it hides real errors. Here line 1 responds only to Teff, line 3 only to log g and line
    5 only to [M/H], so the map from labels to spectrum is injective.

    *Nonlinear in every label*, so multilinear interpolation is not accidentally exact and
    the cubic-versus-linear comparison means something.
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
    # ~1.4 A in the green: the error class this exists to prevent
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
    # the same feature must land ~1.4 A apart on the two scales, not on top of itself
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


# The simplex path reproduces a node to rounding, not bit-for-bit: its weights come from
# Qhull's affine transforms, so at a vertex they are 1 - eps and eps rather than 1 and 0, and
# the error is eps times the spread of the rows across the simplex. The bound is written in
# units of eps * max(|y|, 1), which is between one and two true ulp of y. Measured over 200
# triangulations of each test grid (node order permuted; 105,600 node evaluations): a quarter
# inexact, the worst 1.075 in these units. Four leaves 3.7x headroom on grids this smooth;
# a rougher library would need more, which is a statement about that library.
NODE_TOL_EPS = 4


def _assert_reproduced_to_rounding(got, want):
    tol = NODE_TOL_EPS * np.finfo(np.float64).eps * np.maximum(np.abs(want), 1.0)
    assert np.all(np.abs(np.asarray(got) - want) <= tol)


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_box_interpolation_at_a_node_is_bit_exact(library, method):
    """A node lands on weights that are exactly 1 and 0, so it comes back bit-for-bit."""
    interpolator = library_interpolator(library, method=method)
    for index in (0, 37, library.n_nodes - 1):
        normalized, log_continuum = interpolator(jnp.asarray(library.nodes[index]))
        assert np.array_equal(np.asarray(normalized), library.normalized[index])
        assert np.array_equal(np.asarray(log_continuum), library.log_continuum[index])


def test_simplex_interpolation_at_a_node_is_exact_to_rounding(library):
    """The scattered path promises a few ulp, not bit-exactness; see ``NODE_TOL_EPS``."""
    interpolator = library_interpolator(library, method="simplex")
    for index in (0, 37, library.n_nodes - 1):
        normalized, log_continuum = interpolator(jnp.asarray(library.nodes[index]))
        _assert_reproduced_to_rounding(normalized, library.normalized[index])
        _assert_reproduced_to_rounding(log_continuum, library.log_continuum[index])


@pytest.mark.parametrize("fixture", ["library", "punched_library"])
def test_simplex_node_reproduction_does_not_depend_on_the_triangulation(request, fixture):
    """Every node, under three triangulations of each grid, to the promised bound.

    Which nodes happen to come back bit-exact is a property of the triangulation Qhull
    chose, and that choice differs between scipy builds: a Linux CI runner and a Windows
    desktop disagreed. Permuting the node order changes the triangulation the same way, so
    this exercises the invariant that is actually promised, on every platform. The punched
    grid is the one that reaches the simplex path through ``method="auto"``.
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
    """The claim that justifies paying 4^k taps instead of 2^k."""
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
        # atol against the largest component: a near-zero partial cannot be pinned to a
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
    """A grid with its corners cut away, as physics does to real OB libraries."""
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
    # exactly on a shared facet: inside the hull, but with no margin — the contract is
    # ">= 0 inside", and a node coordinate puts the point on a lattice hyperplane. Zero to
    # rounding rather than exactly: the barycentric coordinates are Qhull's affine transform
    # applied to the point, and at a vertex they carry ~1e-16 of either sign (measured
    # -2.2e-16 at worst over 400 triangulations of the test grids), which is the slack
    # ``crossval_library`` uses for the same decision.
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
    """Same code, opposite spectra, opposite answers — the method validates itself."""
    verdict = line_core_medium(*medium_test_spectrum(medium))
    assert verdict["medium"] == medium
    assert verdict["n_lines"] >= 5
    assert verdict["ratio"] > 20.0


def test_line_core_medium_refuses_when_it_cannot_tell():
    wave = np.linspace(4050.0, 6650.0, 4000)
    with pytest.raises(ValueError, match="not decisive"):
        line_core_medium(wave, np.ones_like(wave))
    # too narrow a span to carry two reference lines
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
    """They must pass as traced model arguments, not be baked in as constants (D27)."""

    @jax.jit
    def evaluate(interpolator, labels):
        return interpolator(labels)[0]

    for lib in (library, punched_library):
        interpolator = library_interpolator(lib)
        out = evaluate(interpolator, jnp.asarray([4700.0, 4.0, -0.2]))
        assert out.shape == (lib.n_pix,)
        assert out.dtype == jnp.float64  # x64 must survive the round trip


# ---------------------------------------------------------------------------
# the registry: naming, caching, and downloads that never touch the network
# ---------------------------------------------------------------------------
#
# Everything checkable offline is checked offline, in the shape tests/test_archive.py uses:
# a fake transport, not a recorded cassette. The BOSZ URL facts were confirmed against the
# live archive on 2026-08-27 and are pinned here so a silent change upstream shows up as a
# failing test rather than as a wrong spectrum.


def _fake_bosz_shard(n_pix, seed=0):
    """Two whitespace columns, flux and continuum, gzipped -- the real BOSZ layout."""
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
    # The medium check needs real line cores; a random spectrum has none, so it declines to
    # answer rather than guessing -- which is the branch this fixture exercises.
    return entry, calls


def test_the_registry_lists_what_it_can_build():
    names = ab.library_names()
    assert "bosz2024-fgk-r20000" in names
    assert names == sorted(names)


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
    """Pinned against files fetched from MAST on 2026-08-27.

    Two of these are not what a careful reading of the documentation would give: Teff is
    not zero-padded, and the atmosphere code changes across the grid. Both were wrong in
    the first draft of the builder and cost a 404 each.
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
        (8000.0, 4.0, "mp"),  # MARCS wins the 7500-8000 overlap, so one family throughout
        (9000.0, 4.0, "ap"),  # ATLAS9 above it
    ],
)
def test_the_atmosphere_code_follows_the_archive(teff, logg, code):
    assert lib_mod._bosz_atmosphere(teff, logg) == code


def test_a_build_downloads_once_and_is_cached(offline_registry):
    _, calls = offline_registry
    first = ab.fetch_library("test-grid", progress=False)
    assert first.nodes.shape == (8, 3)
    assert len(calls) == 9  # eight nodes and one shared wavelength grid

    second = ab.fetch_library("test-grid", progress=False)
    assert len(calls) == 9, "a cache hit must not touch the network"
    # Bit-identical, not merely close: the build path reads back what it wrote, so a warm
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
    # Re-slicing another band out of the raw shards is free; re-downloading them is not.
    assert sorted(p.name for p in lib_mod._raw_dir().glob("*")) == raw_before

    assert ab.clear_library_cache("_raw")
    assert not list(lib_mod._raw_dir().glob("*"))


def test_a_declared_medium_is_checked_against_the_spectra(offline_registry, monkeypatch):
    """BOSZ flipped convention between 2017 and 2024 under one name, so this is not
    hypothetical: the build measures the medium and refuses to reconcile a disagreement."""
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
    """No parser is shipped for a file format nobody here has seen."""
    with pytest.raises(NotImplementedError, match="pollux"):
        ab.ingest_pollux(None)


def test_a_single_valued_axis_is_constant_rather_than_nan():
    """A library sliced to one metallicity keeps its column.

    Before this was handled the degenerate cell gave lo == hi and a 0/0 that reached every
    pixel as a NaN -- silent, and only visible once a fit failed to converge.
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
    """One live request, pinning the two facts a silent upstream change would break.

    Deliberately small: it fetches headers for a single shard rather than any spectrum.
    """
    import urllib.request

    url = lib_mod._bosz_url(6000.0, 4.0, 0.0, alpha=0.0, carbon=0.0, vmicro=2, resolution=20000)
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "albireo test"})
    with urllib.request.urlopen(request, timeout=60) as response:
        assert response.status == 200
        assert int(response.headers["Content-Length"]) > 100_000
