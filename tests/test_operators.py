"""Tests for the shift/interp/rebin linear operators.

Every linear operator is checked for: exact adjoint (inner-product identity and agreement
with ``jax.linear_transpose``), correctness against analytic references, flux/sum
conservation, and gradient correctness versus central finite differences.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

import albireo as ab

RNG = np.random.default_rng(20260811)
N = 512

SHIFTS = [7.3, -4.6, 0.0, 12.37, -0.25]


def gaussian(x, center, sigma):
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


# ---------------------------------------------------------------------------
# Doppler shift operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta", SHIFTS)
def test_shift_adjoint_inner_product(delta):
    u = jnp.asarray(RNG.standard_normal(N))
    w = jnp.asarray(RNG.standard_normal(N))
    lhs = jnp.vdot(ab.shift_spectrum(u, delta), w)
    rhs = jnp.vdot(u, ab.shift_spectrum_adjoint(w, delta))
    np.testing.assert_allclose(float(lhs), float(rhs), rtol=1e-13)


@pytest.mark.parametrize("delta", SHIFTS)
def test_shift_adjoint_matches_linear_transpose(delta):
    u = jnp.asarray(RNG.standard_normal(N))
    w = jnp.asarray(RNG.standard_normal(N))
    (ct,) = jax.linear_transpose(lambda f: ab.shift_spectrum(f, delta), u)(w)
    np.testing.assert_allclose(
        np.asarray(ct), np.asarray(ab.shift_spectrum_adjoint(w, delta)), rtol=1e-14, atol=1e-14
    )


def test_shift_is_linear_in_flux():
    u = jnp.asarray(RNG.standard_normal(N))
    w = jnp.asarray(RNG.standard_normal(N))
    delta = 3.7
    lhs = ab.shift_spectrum(2.5 * u - 1.3 * w, delta)
    rhs = 2.5 * ab.shift_spectrum(u, delta) - 1.3 * ab.shift_spectrum(w, delta)
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-13, atol=1e-13)


def test_shift_zero_is_identity():
    u = jnp.asarray(RNG.standard_normal(N))
    np.testing.assert_array_equal(np.asarray(ab.shift_spectrum(u, 0.0)), np.asarray(u))


@pytest.mark.parametrize("delta", [12.37, -7.81])
def test_shift_matches_analytic_gaussian(delta):
    # Linear interpolation error bound: |err| <= h^2/8 * max|f''| with h = 1 px and
    # max|f''| = 1/sigma^2 for a unit-amplitude Gaussian.
    sigma = 20.0
    x = np.arange(N, dtype=np.float64)
    f = gaussian(x, N / 2, sigma)
    shifted = np.asarray(ab.shift_spectrum(jnp.asarray(f), delta))
    analytic = gaussian(x - delta, N / 2, sigma)
    bound = 1.1 * (1.0 / 8.0) / sigma**2
    assert np.max(np.abs(shifted - analytic)) < bound


def test_shift_roundtrip():
    sigma = 25.0
    x = np.arange(N, dtype=np.float64)
    f = gaussian(x, N / 2, sigma)
    delta = 33.37
    back = ab.shift_spectrum(ab.shift_spectrum(jnp.asarray(f), delta), -delta)
    bound = 2.2 * (1.0 / 8.0) / sigma**2
    assert np.max(np.abs(np.asarray(back) - f)) < bound


def test_shift_constant_is_constant_interior():
    delta = 3.5
    out = np.asarray(ab.shift_spectrum(jnp.ones(N), delta))
    # pixels whose stencil is fully inside keep the constant exactly (weights sum to 1)
    np.testing.assert_allclose(out[5:], 1.0, rtol=1e-14)
    # pixels reaching entirely outside the domain are zero-filled
    np.testing.assert_array_equal(out[:3], 0.0)


def test_shift_conserves_sum_for_interior_support():
    # Each source pixel distributes total weight (1-f) + f = 1 as long as its image
    # stays inside the grid, so the pixel sum is conserved.
    f = np.zeros(N)
    f[100:400] = RNG.standard_normal(300)
    for delta in (7.3, -55.25):
        out = np.asarray(ab.shift_spectrum(jnp.asarray(f), delta))
        np.testing.assert_allclose(out.sum(), f.sum(), rtol=1e-13)


def test_shift_gradient_wrt_shift_matches_fd():
    f = jnp.asarray(gaussian(np.arange(N, dtype=np.float64), N / 2, 15.0))
    w = jnp.asarray(RNG.standard_normal(N))

    def loss(delta):
        return jnp.vdot(ab.shift_spectrum(f, delta), w)

    delta0 = 7.37  # generic non-integer shift: no interpolation-kink crossing under FD
    h = 1e-4
    grad_ad = float(jax.grad(loss)(delta0))
    grad_fd = float((loss(delta0 + h) - loss(delta0 - h)) / (2 * h))
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-7)


def test_gradient_through_velocity_chain_matches_fd():
    # End-to-end chain: v [km/s] -> log-shift -> pixels -> shifted spectrum -> scalar loss.
    grid = ab.LogGrid.from_wavelength_range(4000.0, 4100.0, dv_kms=1.0)
    x = np.arange(grid.n, dtype=np.float64)
    f = jnp.asarray(gaussian(x, grid.n / 2, 12.0))
    target = jnp.asarray(gaussian(x, grid.n / 2 + 20.3, 12.0))

    def loss(v_kms):
        shifted = ab.shift_spectrum(f, grid.velocity_to_pixels(v_kms))
        return jnp.sum((shifted - target) ** 2)

    v0 = 21.7
    h = 1e-3
    grad_ad = float(jax.grad(loss)(v0))
    grad_fd = float((loss(v0 + h) - loss(v0 - h)) / (2 * h))
    np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-6)


def test_shift_output_dtype_is_float64():
    out = ab.shift_spectrum(jnp.ones(8), 0.5)
    assert out.dtype == jnp.float64


# ---------------------------------------------------------------------------
# Point interpolation between grids
# ---------------------------------------------------------------------------


def _random_monotone_grid(n, lo, hi):
    steps = RNG.uniform(0.5, 1.5, size=n)
    x = np.cumsum(steps)
    x = lo + (hi - lo) * (x - x[0]) / (x[-1] - x[0])
    return x


def test_interp_matches_numpy_interp_inside():
    x_in = _random_monotone_grid(200, 4000.0, 4100.0)
    f = RNG.standard_normal(200)
    x_out = np.linspace(4005.0, 4095.0, 137)
    op = ab.interp_operator(x_in, x_out)
    np.testing.assert_allclose(
        np.asarray(op(jnp.asarray(f))), np.interp(x_out, x_in, f), rtol=1e-12, atol=1e-12
    )


def test_interp_zero_fills_outside():
    x_in = np.linspace(4000.0, 4100.0, 50)
    op = ab.interp_operator(x_in, np.array([3990.0, 4050.0, 4110.0]))
    out = np.asarray(op(jnp.ones(50)))
    np.testing.assert_allclose(out, [0.0, 1.0, 0.0], atol=1e-15)


def test_interp_adjoint():
    x_in = _random_monotone_grid(150, 0.0, 10.0)
    x_out = np.linspace(1.0, 9.0, 90)
    op = ab.interp_operator(x_in, x_out)
    u = jnp.asarray(RNG.standard_normal(150))
    w = jnp.asarray(RNG.standard_normal(90))
    lhs = jnp.vdot(op(u), w)
    rhs = jnp.vdot(u, op.adjoint(w))
    np.testing.assert_allclose(float(lhs), float(rhs), rtol=1e-13)
    (ct,) = jax.linear_transpose(op, u)(w)
    np.testing.assert_allclose(np.asarray(ct), np.asarray(op.adjoint(w)), rtol=1e-14, atol=1e-14)


def test_interp_validates_input():
    with pytest.raises(ValueError):
        ab.interp_operator(np.array([1.0, 1.0, 2.0]), np.array([1.5]))


# ---------------------------------------------------------------------------
# Flux-conserving rebinning
# ---------------------------------------------------------------------------


def test_rebin_exact_block_average():
    # 10 unit input bins -> 2 output bins of width 5: exact block means.
    edges_in = np.arange(11, dtype=np.float64)
    edges_out = np.array([0.0, 5.0, 10.0])
    f = RNG.standard_normal(10)
    op = ab.rebin_operator(edges_in=edges_in, edges_out=edges_out)
    expected = np.array([f[:5].mean(), f[5:].mean()])
    np.testing.assert_allclose(np.asarray(op(jnp.asarray(f))), expected, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(op.coverage), 1.0, rtol=1e-14)


def test_rebin_conserves_integrated_flux():
    # Non-uniform input grid; output grid strictly inside; flux supported strictly
    # inside the output coverage -> total integral must be preserved exactly.
    x_in = _random_monotone_grid(300, 4000.0, 4100.0)
    edges_in = ab.bin_edges_from_centers(x_in)
    grid_out = ab.LogGrid.from_wavelength_range(4020.0, 4080.0, dv_kms=8.0)
    edges_out = ab.bin_edges_from_centers(grid_out.wave)

    f = np.zeros(300)
    support = (x_in > 4030.0) & (x_in < 4070.0)
    f[support] = 1.0 + 0.5 * RNG.standard_normal(support.sum())

    op = ab.rebin_operator(edges_in=edges_in, edges_out=edges_out)
    out = np.asarray(op(jnp.asarray(f)))
    integral_in = np.sum(f * np.diff(edges_in))
    integral_out = np.sum(out * np.diff(edges_out))
    np.testing.assert_allclose(integral_out, integral_in, rtol=1e-12)


def test_rebin_constant_preserved_where_covered():
    x_in = np.linspace(4000.0, 4100.0, 500)
    x_out = np.linspace(3995.0, 4105.0, 60)  # extends beyond the input span
    op = ab.rebin_operator(x_in=x_in, x_out=x_out)
    out = np.asarray(op(jnp.ones(500)))
    coverage = np.asarray(op.coverage)
    full = coverage > 1.0 - 1e-12
    assert full.sum() > 40  # sanity: most bins are fully covered
    np.testing.assert_allclose(out[full], 1.0, rtol=1e-13)
    # partially/un-covered bins report coverage < 1 so callers can mask them
    assert (coverage[~full] < 1.0 - 1e-12).all()


def test_rebin_adjoint():
    x_in = _random_monotone_grid(120, 0.0, 50.0)
    x_out = np.linspace(5.0, 45.0, 33)
    op = ab.rebin_operator(x_in=x_in, x_out=x_out)
    u = jnp.asarray(RNG.standard_normal(120))
    w = jnp.asarray(RNG.standard_normal(33))
    lhs = jnp.vdot(op(u), w)
    rhs = jnp.vdot(u, op.adjoint(w))
    np.testing.assert_allclose(float(lhs), float(rhs), rtol=1e-13)
    (ct,) = jax.linear_transpose(op, u)(w)
    np.testing.assert_allclose(np.asarray(ct), np.asarray(op.adjoint(w)), rtol=1e-14, atol=1e-14)


def test_rebin_operators_are_jit_compatible():
    # Static sizes live in the pytree aux data, so operators can cross jit boundaries.
    x_in = np.linspace(0.0, 10.0, 100)
    x_out = np.linspace(1.0, 9.0, 40)
    op = ab.rebin_operator(x_in=x_in, x_out=x_out)
    f = jnp.asarray(RNG.standard_normal(100))

    @jax.jit
    def apply(op, f):
        return op(f)

    np.testing.assert_allclose(np.asarray(apply(op, f)), np.asarray(op(f)), rtol=1e-15)


# ---------------------------------------------------------------------------
# Rotational (v sin i) broadening
# ---------------------------------------------------------------------------


def rotation_profile(x, epsilon):
    """Gray (2005) eq. 18.14 in half-width units — the oracle's integrand."""
    if abs(x) >= 1.0:
        return 0.0
    return (
        2.0 * (1.0 - epsilon) * np.sqrt(1.0 - x * x) + 0.5 * np.pi * epsilon * (1.0 - x * x)
    ) / (np.pi * (1.0 - epsilon / 3.0))


def rotation_kernel_oracle(vsini_px, radius, epsilon=0.6):
    """Independent kernel: adaptive quadrature of the profile over each pixel.

    ``scipy.integrate.quad`` with the support edges declared as break points. Fixed-rule
    quadrature (Simpson, Gauss-Legendre) is *not* adequate here: the profile has a
    square-root edge, so a rule that straddles it converges at O(h^1.5) and disagrees in
    the 4th decimal — which is a statement about the rule, not about the kernel.
    """
    taps = []
    for offset in range(-radius, radius + 1):
        lo, hi = offset - 0.5, offset + 0.5
        breaks = [p for p in (-vsini_px, vsini_px) if lo < p < hi]
        taps.append(
            integrate.quad(
                lambda u: rotation_profile(u / vsini_px, epsilon) / vsini_px,
                lo,
                hi,
                points=breaks or None,
                limit=200,
            )[0]
        )
    taps = np.asarray(taps)
    return taps / taps.sum()


@pytest.mark.parametrize("vsini_px", [0.7, 2.0, 5.5, 17.3])
def test_rotational_kernel_matches_quadrature_oracle(vsini_px):
    k = np.asarray(ab.rotational_kernel(vsini_px))
    radius = (k.size - 1) // 2
    np.testing.assert_allclose(k, rotation_kernel_oracle(vsini_px, radius), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("epsilon", [0.0, 0.3, 0.6, 1.0])
def test_rotational_kernel_normalized_and_exactly_symmetric(epsilon):
    k = np.asarray(ab.rotational_kernel(6.25, epsilon=epsilon))
    assert k.size % 2 == 1
    np.testing.assert_allclose(k.sum(), 1.0, rtol=1e-14)
    # Exact invariant: the profile is even, so the taps mirror bit-for-bit.
    assert np.array_equal(k, k[::-1])


def test_rotational_kernel_rejects_bad_arguments():
    with pytest.raises(ValueError):
        ab.rotational_kernel(0.0)
    with pytest.raises(ValueError):
        ab.rotational_kernel(3.0, epsilon=1.5)
    with pytest.raises(ValueError):
        ab.rotational_kernel_traced(3.0, 0)


def test_rotational_kernel_traced_matches_static():
    for vsini_px in (1.5, 4.0, 9.0):
        radius = (np.asarray(ab.rotational_kernel(vsini_px)).size - 1) // 2
        static = np.asarray(ab.rotational_kernel(vsini_px))
        traced = np.asarray(
            jax.jit(ab.rotational_kernel_traced, static_argnums=1)(vsini_px, radius)
        )
        np.testing.assert_allclose(traced, static, rtol=1e-14, atol=1e-16)


def _kernel_second_moment(vsini_px, radius):
    """A functional of the whole kernel, so every tap carries gradient."""
    k = ab.rotational_kernel_traced(vsini_px, radius)
    return jnp.sum(k * jnp.arange(-radius, radius + 1, dtype=jnp.float64) ** 2)


@pytest.mark.parametrize("vsini_px", [0.9, 3.0, 6.3, 12.0, 25.0, 41.7])
def test_rotational_kernel_gradient_matches_finite_differences(vsini_px):
    """The differentiability the optimizer relies on, measured rather than asserted."""
    radius = ab.rotational_radius_for(300.0, 6.0)
    analytic = float(jax.jit(jax.grad(_kernel_second_moment), static_argnums=1)(vsini_px, radius))
    eps = 1e-6 * vsini_px
    numeric = (
        float(_kernel_second_moment(vsini_px + eps, radius))
        - float(_kernel_second_moment(vsini_px - eps, radius))
    ) / (2 * eps)
    assert np.isfinite(analytic)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6)


@pytest.mark.parametrize("vsini_px", [7.5, 49.5])
def test_rotational_kernel_is_c1_where_the_profile_edge_lands_on_a_pixel_edge(vsini_px):
    """C^1 but not C^2 at half-integer widths — and C^1 is what L-BFGS and NUTS need.

    When ``vsini_px`` is a half-integer the profile's support boundary coincides exactly
    with a pixel boundary, and that tap picks up a ``|delta|^{3/2}`` term: the first
    derivative exists (both one-sided limits agree with the analytic gradient) while the
    second does not. Central differences therefore converge only as ``sqrt(eps)`` here,
    which is a property of the profile's square-root edge, not an error — so the test
    asserts *convergence toward* the analytic value instead of a fixed tolerance.
    """
    radius = ab.rotational_radius_for(300.0, 6.0)
    analytic = float(jax.jit(jax.grad(_kernel_second_moment), static_argnums=1)(vsini_px, radius))
    assert np.isfinite(analytic)

    errors = []
    for eps in (1e-4 * vsini_px, 1e-5 * vsini_px, 1e-6 * vsini_px):
        one_sided = [
            (
                float(_kernel_second_moment(vsini_px + s * eps, radius))
                - float(_kernel_second_moment(vsini_px, radius))
            )
            / (s * eps)
            for s in (+1.0, -1.0)
        ]
        # both one-sided slopes approach the same value: the derivative exists
        assert abs(one_sided[0] - one_sided[1]) < 0.02 * abs(analytic)
        errors.append(max(abs(d - analytic) for d in one_sided))
    assert errors[-1] < errors[0] / 5.0  # converging, at the sqrt(eps) rate
    assert errors[-1] < 1e-3 * abs(analytic)


def test_rotational_kernel_degenerates_to_delta():
    radius = 6
    k = np.asarray(ab.rotational_kernel_traced(1e-9, radius))
    assert k[radius] == pytest.approx(1.0)
    np.testing.assert_allclose(np.delete(k, radius), 0.0, atol=1e-15)
    # the floor keeps the vsini -> 0 limit differentiable rather than NaN
    g = jax.grad(lambda v: ab.rotational_kernel_traced(v, radius)[radius])(0.0)
    assert np.isfinite(float(g))


def test_rotational_broadening_conserves_equivalent_width():
    """Rotation redistributes flux within a line; it must not create or destroy any."""
    x = np.arange(N, dtype=np.float64)
    line = -0.4 * gaussian(x, N / 2, 3.0)
    out = np.asarray(ab.convolve_spectrum(jnp.asarray(line), ab.rotational_kernel(8.0)))
    np.testing.assert_allclose(out.sum(), line.sum(), rtol=1e-12)
    assert out.min() > line.min()  # shallower


def test_rotational_radius_for_covers_the_bound():
    radius = ab.rotational_radius_for(120.0, 8.0)
    assert radius >= 120.0 / 8.0
    # the bound's own kernel fits inside the radius it asks for
    assert (np.asarray(ab.rotational_kernel(120.0 / 8.0)).size - 1) // 2 <= radius
    with pytest.raises(ValueError):
        ab.rotational_radius_for(0.0, 8.0)


# ---------------------------------------------------------------------------
# Stationary (LSF) convolution
# ---------------------------------------------------------------------------


def test_gaussian_kernel_properties():
    k = np.asarray(ab.gaussian_kernel(2.5))
    assert k.size % 2 == 1
    np.testing.assert_allclose(k.sum(), 1.0, rtol=1e-14)
    np.testing.assert_allclose(k, k[::-1], rtol=1e-14)  # symmetric
    with pytest.raises(ValueError):
        ab.gaussian_kernel(0.0)


def test_convolve_symmetric_kernel_is_self_adjoint():
    kernel = ab.gaussian_kernel(3.0)
    u = jnp.asarray(RNG.standard_normal(N))
    w = jnp.asarray(RNG.standard_normal(N))
    lhs = jnp.vdot(ab.convolve_spectrum(u, kernel), w)
    rhs = jnp.vdot(u, ab.convolve_spectrum(w, kernel))
    np.testing.assert_allclose(float(lhs), float(rhs), rtol=1e-13)


def test_convolve_adjoint_is_reversed_kernel():
    # generic asymmetric kernel: adjoint of zero-padded 'same' convolution is
    # convolution with the reversed kernel
    kernel = jnp.asarray(RNG.standard_normal(9))
    u = jnp.asarray(RNG.standard_normal(N))
    w = jnp.asarray(RNG.standard_normal(N))
    (ct,) = jax.linear_transpose(lambda f: ab.convolve_spectrum(f, kernel), u)(w)
    np.testing.assert_allclose(
        np.asarray(ct), np.asarray(ab.convolve_spectrum(w, kernel[::-1])), rtol=1e-13, atol=1e-13
    )


def test_convolve_preserves_constant_interior():
    kernel = ab.gaussian_kernel(4.0)
    r = (len(np.asarray(kernel)) - 1) // 2
    out = np.asarray(ab.convolve_spectrum(jnp.ones(N), kernel))
    np.testing.assert_allclose(out[r:-r], 1.0, rtol=1e-14)


def test_edges_from_centers():
    centers = np.array([1.0, 2.0, 4.0])
    edges = ab.bin_edges_from_centers(centers)
    np.testing.assert_allclose(edges, [0.5, 1.5, 3.0, 5.0])
    with pytest.raises(ValueError):
        ab.bin_edges_from_centers(np.array([1.0]))
