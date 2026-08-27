"""Tests for synthetic spectral libraries and their differentiable interpolators.

Interpolators are checked against scipy's own implementations — ``RegularGridInterpolator``
for the box, ``LinearNDInterpolator`` for the scattered case — so the oracle is independent
code rather than a second copy of the same arithmetic. The one exact invariant is node
reproduction: interpolating *at* a node must return the stored spectrum bit-for-bit, which
is what lets the warm-start node scan and the continuous fit be compared on equal terms.

Nothing here needs the network. Libraries are generated in-test from an analytic rule, and
the air/vacuum measurement is exercised on spectra built with a known convention.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

import albireo as ab
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


@pytest.mark.parametrize("method", ["linear", "cubic", "simplex"])
def test_interpolation_at_a_node_is_exact(library, method):
    """The exact invariant: a node comes back bit-for-bit, not merely to tolerance."""
    interpolator = library_interpolator(library, method=method)
    for index in (0, 37, library.n_nodes - 1):
        normalized, log_continuum = interpolator(jnp.asarray(library.nodes[index]))
        assert np.array_equal(np.asarray(normalized), library.normalized[index])
        assert np.array_equal(np.asarray(log_continuum), library.log_continuum[index])


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
    # ">= 0 inside", and a node coordinate puts the point on a lattice hyperplane
    assert float(interpolator.hull_margin(jnp.asarray([4700.0, 4.0, -0.2]))) == 0.0
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
