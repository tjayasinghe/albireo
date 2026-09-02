"""Stellar labels for disentangled components: Teff, log g, [M/H] and v sin i.

The mode fits atmospheric labels to the component spectra a disentangling returns, against a
published synthetic grid (:mod:`albireo.library`), so that a component can serve as a
radial-velocity template for TODCOR (:mod:`albireo.todcor`), saphires, iSpec or a survey
pipeline. Fitting the labels here retains the per-pixel uncertainties of the disentangling
posterior and the fact that the components were measured jointly. The equations are in
``docs/math.md`` §9.

There is no synthesis in this module: no line list, no model atmosphere, no radiative
transfer and no individual abundances. Those remain the province of GSSP, iSpec, Korg.jl and
PySME, reached through :mod:`albireo.handoff`. The accuracy targeted is the accuracy at
which the template stops limiting the velocities: roughly 2-3% in Teff, 0.15 dex in log g
and [M/H], and 10% in v sin i (``docs/math.md`` §9.6). A label from this mode is a template
coordinate, not an entry in an abundance table.

Three model choices follow from the structure of the problem. First, dilution is fitted
jointly. Disentangling returns component spectra in the common continuum, scaled by assumed
light fractions, and the likelihood constrains only the products ``l_i d_i``, so an error in
the assumed ``l`` rescales every line depth and is degenerate with Teff. The components are
therefore fitted together with one shared scalar per companion (a radius ratio), and the
wavelength dependence of the light fractions is taken from the grids' own continua, which
makes them sum to one at every wavelength by construction. This is the binary mode of GSSP
(Tkachenko 2015), where a wavelength-independent dilution was measured to shift a
secondary's Teff by 275 K.

Second, the zero point is modelled explicitly. Each component's constant offset, and very
nearly its slope, lies in the null space of the disentangling problem and is held only by
the smoothness prior's ridge (``docs/math.md`` §5.1); left unmodelled it lands on the line
depths and returns as a Teff error. Each component therefore carries a low-order additive
Chebyshev nuisance whose zeroth term is that zero point, which is fitted and reported
(``docs/math.md`` §9.1). The nuisance is additive rather than multiplicative because the
null space lives in the continuum, where the deviation spectrum is zero and a multiplicative
term has no effect.

Third, two uncertainties are reported. Residuals from disentangling are correlated rather
than white, and formal errors on this problem run five to ten times optimistic: Gebruers
et al. (2022) report 70 K formal against 425 K realistic for B stars at S/N 150. The Laplace
covariance is therefore quoted beside the spread obtained by refitting joint posterior draws
of the component spectra (:func:`refit_draws`), and :meth:`LabelMatch.summary` prints both
(``docs/math.md`` §9.5).

References
----------
Tkachenko, A. 2015, A&A, 581, A129
Gebruers, S., Tkachenko, A., Bowman, D. M., et al. 2022, A&A, 665, A36
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from albireo.grids import LogGrid, log_doppler_shift
from albireo.inference import laplace_inverse_mass, run_map
from albireo.library import SUPPORTED_MEDIA, SpectralLibrary, library_interpolator
from albireo.operators import (
    gaussian_kernel,
    rotational_kernel_traced,
    rotational_radius_for,
    shift_spectrum,
)

__all__ = [
    "FixedDilution",
    "LabelMatch",
    "LabelProblem",
    "RadiusRatio",
    "ScalarDilution",
    "StarLabels",
    "match_labels",
    "refit_draws",
]

_LABEL_ORDER = ("teff", "logg", "mh")


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StarLabels:
    """The declaration for one component: what is fitted, and what is assumed.

    Each label accepts the declaration vocabulary of the façade
    (:class:`~albireo.facade.Fixed`, :class:`~albireo.facade.Known`,
    :class:`~albireo.facade.Between`, :class:`~albireo.facade.Sampled`) or a bare float,
    which is treated as fixed. The specs are duck-typed rather than imported, so this module
    does not depend on the façade and can be driven from another code's output.

    ``logg`` requires a decision before the fit. Teff and log g correlate at about 0.98 when
    both are free (Tamajo et al. 2011). For an eclipsing binary the light curve and the orbit
    give log g to 0.01 dex, and fixing it there is what makes the analysis well posed. A
    non-eclipsing SB2 has no such anchor: run the fit free, run it fixed, and report the
    spread as the uncertainty.

    ``macro_kms`` is a fixed Gaussian macroturbulence folded into the intrinsic broadening.
    It is not fitted because it is not separable from ``vsini`` at survey resolution. With
    the default of zero, a fitted ``vsini`` measures all broadening beyond the instrument
    profile, which is what a template requires.

    Leaving ``teff``, ``logg`` or ``vsini`` as ``None`` adopts the library's own range (and
    0-300 km/s for ``vsini``), which is a starting point rather than a considered prior.

    References
    ----------
    Tamajo, E., Pavlovski, K. & Southworth, J. 2011, A&A, 526, A76
    """

    library: SpectralLibrary
    teff: Any = None
    logg: Any = None
    vsini: Any = None
    v_kms: Any = None
    macro_kms: float = 0.0


@dataclass(frozen=True)
class RadiusRatio:
    """Wavelength-dependent dilution from one shared scalar per companion (the default).

    The light fractions are ``w_i(lambda) = A_i C_i(lambda) / sum_j A_j C_j(lambda)`` with
    ``A_1 = 1`` and ``A_i = r_i^2``, where ``C`` is each grid's own continuum and ``r_i`` the
    radius ratio relative to the first star. The fractions sum to one at every wavelength by
    construction, so no constraint site or penalty is needed and none can drift, and their
    wavelength dependence comes from the model atmospheres rather than from a fitted
    polynomial.

    This is the ``gssp_binary`` parameterization (Tkachenko 2015), and it is what makes a
    spectroscopic light ratio measurable. Published spectroscopic ratios agree with
    light-curve ratios to a few percent and are competitive with them where the photometric
    solution is degenerate.

    References
    ----------
    Tkachenko, A. 2015, A&A, 581, A129
    """

    ratio: Any = None


@dataclass(frozen=True)
class ScalarDilution:
    """One free wavelength-independent factor per component: the single-star fallback.

    The ``gssp_single`` parameterization (Tkachenko 2015). It applies when there is only one
    component to fit, or when two grids' continua cannot be trusted on a common scale. It is
    strictly weaker than :class:`RadiusRatio`: nothing ties the components together, nothing
    enforces the sum to one, and the wavelength dependence that carries the light-ratio
    information is discarded. :meth:`LabelMatch.summary` records that a fit used it.

    References
    ----------
    Tkachenko, A. 2015, A&A, 581, A129
    """

    factor: Any = None


@dataclass(frozen=True)
class FixedDilution:
    """Hold the light fractions at their assumed values: ``w_i == l0_i``.

    A diagnostic rather than a recommended configuration. It gives the labels that follow
    with no dilution freedom at all, so the shift between this and a :class:`RadiusRatio` fit
    measures how far the assumed light fractions were moving the answer.
    """


# ---------------------------------------------------------------------------
# Spec handling (duck-typed against the façade vocabulary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FixedValue:
    """A bare float presented through the spec interface."""

    value: Any

    def distribution(self):
        return None

    def start(self):
        return jnp.asarray(self.value, dtype=jnp.float64)

    def upper(self) -> float:
        return float(np.max(np.abs(np.atleast_1d(np.asarray(self.value, dtype=float)))))


def _as_spec(value, name: str, default=None):
    if value is None:
        if default is None:
            raise ValueError(f"{name} must be declared (a float, or Fixed/Between/Known/Sampled)")
        return default
    if isinstance(value, (int, float, np.floating)):
        return _FixedValue(float(value))
    if isinstance(value, tuple):
        raise TypeError(
            f"{name} was given as a tuple; use Between{value} instead, so that the prior and "
            "the starting value stay in one object"
        )
    if type(value).__name__ == "Scanned":
        raise TypeError(
            f"{name}=Scanned(...) is not supported here. The warm start already evaluates "
            "every library node; declare a range with Between(lo, hi) instead."
        )
    for method in ("distribution", "start", "upper"):
        if not callable(getattr(value, method, None)):
            raise TypeError(
                f"{name} must be a float or a spec with .distribution()/.start()/.upper() "
                f"(Fixed, Known, Between, Sampled); got {type(value).__name__}"
            )
    return value


def _sample(name: str, spec):
    """One numpyro site, or a constant when the spec is fixed."""
    distribution = spec.distribution()
    if distribution is None:
        return jnp.asarray(spec.start(), dtype=jnp.float64)
    return numpyro.sample(name, distribution)


def _is_fixed(spec) -> bool:
    return spec.distribution() is None


# ---------------------------------------------------------------------------
# The traced problem
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LabelProblem:
    """Everything the label likelihood needs, as one traced pytree argument.

    Passed to the numpyro model through ``model_args`` rather than captured in a closure:
    the interpolated grids run to tens of megabytes, and XLA constant-folds closure constants
    into the compiled executable (``docs/design.md`` D27).
    """

    interpolators: tuple
    data: jax.Array
    sigma: jax.Array
    weight: jax.Array
    basis: jax.Array
    ell0: jax.Array
    lsf_kernel: jax.Array
    macro_kernels: tuple
    dx: float
    dv_kms: float
    rot_radius: int
    relativistic: bool
    matched: bool
    dilution: str
    label_axes: tuple

    @property
    def n_star(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_pix(self) -> int:
        return int(self.data.shape[1])

    def tree_flatten(self):
        children = (
            self.interpolators,
            self.data,
            self.sigma,
            self.weight,
            self.basis,
            self.ell0,
            self.lsf_kernel,
            self.macro_kernels,
        )
        aux = (
            self.dx,
            self.dv_kms,
            self.rot_radius,
            self.relativistic,
            self.matched,
            self.dilution,
            self.label_axes,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


def _broadening_kernel(problem: LabelProblem, index: int, vsini):
    """Rotation, any fixed macroturbulence, and the instrument profile when matched.

    Convolved in ``full`` mode so that no wing is truncated. Every length is static, since
    only the kernel values depend on the traced ``v sin i``. The rotational profile is the
    limb-darkened profile of Gray (2005), built by
    :func:`albireo.operators.rotational_kernel_traced`.

    References
    ----------
    Gray, D. F. 2005, The Observation and Analysis of Stellar Photospheres, 3rd ed.
    (Cambridge: Cambridge University Press)
    """
    kernel = rotational_kernel_traced(vsini / problem.dv_kms, problem.rot_radius)
    macro = problem.macro_kernels[index]
    if macro.shape[0] > 1:
        kernel = jnp.convolve(kernel, macro, mode="full")
    if problem.matched:
        kernel = jnp.convolve(kernel, problem.lsf_kernel, mode="full")
    return kernel / jnp.sum(kernel)


def _component_model(problem: LabelProblem, index: int, labels, vsini, v_kms):
    """One component's broadened, shifted deviation spectrum, and its continuum.

    The chain of ``docs/math.md`` §9.1: interpolate, subtract the continuum, broaden, then
    Doppler shift. Every operator is stationary on the uniform log grid, so they commute and
    the order affects readability only.
    """
    normalized, log_continuum = problem.interpolators[index](labels)
    deviation = jnp.convolve(
        normalized - 1.0, _broadening_kernel(problem, index, vsini), mode="same"
    )
    shift = log_doppler_shift(v_kms, relativistic=problem.relativistic) / problem.dx
    return shift_spectrum(deviation, shift), log_continuum


def _light_fractions(log_continua, amplitudes):
    """Light fractions that sum to one at every pixel, evaluated in the log.

    ``w_i = A_i C_i / sum_j A_j C_j`` as a softmax over ``log C_i + log A_i``. The continua
    span decades across a Teff range, and this form cannot overflow.
    """
    stacked = jnp.stack(log_continua) + jnp.log(amplitudes)[:, None]
    return jnp.exp(stacked - jax.scipy.special.logsumexp(stacked, axis=0, keepdims=True))


def _model_rows(problem: LabelProblem, labels, vsini, v_kms, amplitudes, offsets):
    """The full ``(n_star, n_pix)`` model, in the deviation space the data live in."""
    parts = [
        _component_model(problem, i, labels[i], vsini[i], v_kms[i]) for i in range(problem.n_star)
    ]
    deviations = jnp.stack([p[0] for p in parts])

    if problem.dilution == "radius_ratio":
        weights = _light_fractions([p[1] for p in parts], amplitudes)
    elif problem.dilution == "scalar":
        weights = amplitudes[:, None] * jnp.ones((1, problem.n_pix))
    else:
        weights = problem.ell0[:, None] * jnp.ones((1, problem.n_pix))

    # The disentangler solved for `d` against an assumed l0, so what it recovered is the
    # true light fraction divided by the assumed one (math.md §9.1). A wrong l0 therefore
    # appears here as a fitted ratio rather than as a wrong temperature.
    scaled = (weights / problem.ell0[:, None]) * deviations
    return scaled + offsets @ problem.basis.T


def label_model(problem: LabelProblem, specs: dict, config: dict):
    """Build the numpyro model for a label fit.

    ``specs`` holds the declared priors, which are small enough to be closure constants;
    the arrays travel in ``problem`` as a traced model argument.
    """

    names = config["names"]

    def _model(problem):
        mh_shared = _sample("mh", specs["mh"]) if config["shared_mh"] else None

        labels, vsini, v_kms = [], [], []
        for i, name in enumerate(names):
            values = {
                "teff": _sample(f"teff_{name}", specs[f"teff_{name}"]),
                "logg": _sample(f"logg_{name}", specs[f"logg_{name}"]),
                "mh": mh_shared
                if config["shared_mh"]
                else _sample(f"mh_{name}", specs[f"mh_{name}"]),
            }
            labels.append(jnp.stack([values[axis] for axis in problem.label_axes[i]]))
            vsini.append(_sample(f"vsini_{name}", specs[f"vsini_{name}"]))
            v_kms.append(_sample(f"v_{name}", specs[f"v_{name}"]))

        if problem.dilution == "radius_ratio":
            ratios = [jnp.asarray(1.0)] + [
                _sample(f"ratio_{name}", specs[f"ratio_{name}"]) for name in names[1:]
            ]
            amplitudes = jnp.stack(ratios) ** 2
        elif problem.dilution == "scalar":
            amplitudes = jnp.stack(
                [_sample(f"scale_{name}", specs[f"scale_{name}"]) for name in names]
            )
        else:
            amplitudes = problem.ell0

        offsets = (
            jnp.stack([_sample(f"offset_{name}", specs[f"offset_{name}"]) for name in names])
            if config["offsets"]
            else jnp.zeros((len(names), 0))
        )
        log_jitter = (
            jnp.stack(
                [_sample(f"log_jitter_{name}", specs[f"log_jitter_{name}"]) for name in names]
            )
            if config["jitter"]
            else jnp.zeros(len(names))
        )

        rows = _model_rows(problem, labels, vsini, v_kms, amplitudes, offsets)
        scale = problem.sigma * jnp.exp(log_jitter)[:, None]
        residual = (rows - problem.data) / scale
        loglike = -0.5 * jnp.sum(problem.weight * residual**2) - jnp.sum(
            problem.weight * jnp.log(scale)
        )

        if config["hull_guard"]:
            # A soft barrier rather than a rejection: outside the hull the simplex
            # interpolator extrapolates flat, which is finite but meaningless, so the
            # potential must slope back inside rather than sit on a plateau.
            margin = jnp.stack(
                [problem.interpolators[i].hull_margin(labels[i]) for i in range(len(names))]
            )
            loglike = loglike - 1e3 * jnp.sum(jax.nn.softplus(-50.0 * margin) ** 2)

        numpyro.factor("label_loglike", loglike)

    _model.model_args = (problem,)
    return _model


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------


def _profiled_chi2(model_row, data, weight, basis, gram_inv):
    """Chi-square after the additive nuisance is solved for in closed form.

    The Chebyshev offsets enter linearly, so at each trial they are profiled out with one
    small solve instead of being searched. This makes scanning every library node affordable
    and leaves the scan's chi-square comparable with the fitted one.
    """
    residual = data - model_row
    coeffs = gram_inv @ (basis.T @ (weight * residual))
    return jnp.sum(weight * (residual - basis @ coeffs) ** 2)


def _scan_component(problem: LabelProblem, index: int, nodes, vsini_trials, v_trials):
    """Profiled chi-square at every library node, for one component.

    Returns ``(chi2, vsini, v)`` per node, each the best over the trial broadenings and
    velocities. The components decouple here because the dilution is held at its assumed
    value, which is the ``FixedDilution`` model. The scan is a coarse start whose purpose is
    to place the optimizer in the right basin.
    """
    data = problem.data[index]
    weight = problem.weight[index] / problem.sigma[index] ** 2
    basis = problem.basis
    gram = basis.T @ (weight[:, None] * basis)
    gram_inv = jnp.linalg.inv(gram + 1e-10 * jnp.trace(gram) * jnp.eye(basis.shape[1]))

    @jax.jit
    def chi2_at(node, vsini, v):
        row, _ = _component_model(problem, index, node, vsini, v)
        return _profiled_chi2(row, data, weight, basis, gram_inv)

    scan = jax.vmap(chi2_at, in_axes=(0, None, None))
    best = np.full((nodes.shape[0], 3), np.inf)
    for vsini in vsini_trials:
        for v in v_trials:
            chi2 = np.asarray(scan(nodes, float(vsini), float(v)))
            better = chi2 < best[:, 0]
            best[better] = np.column_stack(
                [chi2, np.full_like(chi2, vsini), np.full_like(chi2, v)]
            )[better]
    return best


def _spec_bounds(spec):
    """``(lo, hi)`` for a bounded prior, else ``None``."""
    distribution = spec.distribution()
    if distribution is None:
        return None
    low, high = getattr(distribution, "low", None), getattr(distribution, "high", None)
    if low is None or high is None:
        return None
    return float(np.asarray(low)), float(np.asarray(high))


def _allowed_nodes(table, axes, specs, name, shared_mh):
    """Which library nodes the declared priors permit.

    Without this filter the warm start can hand the optimizer a node the prior excludes, and
    numpyro then rejects the fit with "cannot find valid initial parameters", an opaque
    message for a grid wider than its prior. A fixed label is narrowed to the nearest
    available node value rather than to nothing, since a fixed Teff rarely lands exactly on a
    grid point.
    """
    allowed = np.ones(table.shape[0], dtype=bool)
    for column, label in enumerate(axes):
        key = "mh" if (label == "mh" and shared_mh) else f"{label}_{name}"
        spec = specs.get(key)
        if spec is None:
            continue
        bounds = _spec_bounds(spec)
        if bounds is not None:
            allowed &= (table[:, column] >= bounds[0]) & (table[:, column] <= bounds[1])
        elif _is_fixed(spec):
            values = np.unique(table[:, column])
            nearest = values[int(np.argmin(np.abs(values - float(np.asarray(spec.start())))))]
            allowed &= table[:, column] == nearest
    return allowed


def _combine_scans(scans, node_tables, label_axes, shared_mh, top_k):
    """Rank joint starting points, honouring a shared metallicity.

    With one [M/H] for the system the components cannot be optimized independently: the
    best pair is the best slice of the shared axis, not the pair of individual bests.
    Scanning each star separately and then combining on the shared axis costs ``n_nodes`` per
    star instead of ``n_nodes`` raised to the number of stars, and reaches the same
    candidates.
    """
    mh_columns = [axes.index("mh") if "mh" in axes else None for axes in label_axes]
    if shared_mh and any(column is not None for column in mh_columns):
        values = sorted(
            {
                float(v)
                for table, column in zip(node_tables, mh_columns, strict=True)
                if column is not None
                for v in np.unique(table[:, column])
            }
        )
        candidates = []
        for mh in values:
            picks, total = [], 0.0
            for scan, table, column in zip(scans, node_tables, mh_columns, strict=True):
                allowed = (
                    np.flatnonzero(table[:, column] == mh)
                    if column is not None
                    else np.arange(table.shape[0])
                )
                if allowed.size == 0:
                    picks = None
                    break
                best = allowed[int(np.argmin(scan[allowed, 0]))]
                picks.append(best)
                total += float(scan[best, 0])
            if picks is not None and np.isfinite(total):
                candidates.append((total, tuple(picks)))
    else:
        # Independent metallicities (or none): each star is ranked on its own.
        orders = [np.argsort(scan[:, 0]) for scan in scans]
        candidates = [
            (
                sum(float(scan[order[rank], 0]) for scan, order in zip(scans, orders, strict=True)),
                tuple(int(order[rank]) for order in orders),
            )
            for rank in range(min(top_k, min(len(order) for order in orders)))
        ]
        candidates = [item for item in candidates if np.isfinite(item[0])]
    if not candidates:
        raise ValueError(
            "the warm-start scan found no library node consistent with the declared priors"
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[: max(1, top_k)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _chebyshev_basis(n_pix: int, order: int | None) -> np.ndarray:
    """``T_m`` evaluated on the grid, mapped to ``[-1, 1]``, shape ``(n_pix, order + 1)``.

    ``order=None`` returns a zero-column basis, that is, no additive nuisance at all. That
    is the control case against which the nuisance is judged, not a recommended
    configuration; see ``docs/math.md`` §5.1.
    """
    if order is None:
        return np.zeros((n_pix, 0))
    if order < 0:
        raise ValueError("offset_order must be >= 0, or None for no additive nuisance")
    x = np.linspace(-1.0, 1.0, n_pix)
    basis = np.empty((n_pix, order + 1))
    basis[:, 0] = 1.0
    if order >= 1:
        basis[:, 1] = x
    for m in range(2, order + 1):
        basis[:, m] = 2.0 * x * basis[:, m - 1] - basis[:, m - 2]
    return basis


def match_labels(
    grid: LogGrid,
    d_hat,
    *,
    stars,
    medium: str,
    light_fractions,
    lsf_sigma_kms: float,
    std=None,
    mh=None,
    dilution=None,
    compare: str = "native",
    offset_order: int = 2,
    offset_scale: float = 0.05,
    jitter: bool = True,
    exclude_angstrom=(),
    interpolation: str = "auto",
    scan_vsini=None,
    scan_velocities=None,
    top_k: int = 4,
    max_steps: int = 500,
    tol: float = 1e-2,
    seed: int = 0,
    progress=None,
) -> LabelMatch:
    """Fit atmospheric labels to disentangled component spectra.

    Parameters
    ----------
    grid
        The model grid the components are defined on (``fit.dis.grid`` for a façade fit).
    d_hat
        Component *deviation* spectra, shape ``(n_star, n_pix)``: what ``Fit.spectra()``
        returns for the stellar rows, in units of the common continuum. Telluric and
        nebular rows must be dropped before calling.
    stars
        Mapping of component name to :class:`StarLabels`. Order sets the reference star
        for :class:`RadiusRatio` (the first is ``r = 1``).
    medium
        ``"air"`` or ``"vacuum"``: the scale the data are on, taken from the dataset's own
        declaration. Required, because an incorrect choice is an 83 km/s error and no default
        can be safe.
    light_fractions
        The light fractions assumed at disentangling time, one per star. These are what
        ``d_hat`` was scaled against, not a measurement.
    lsf_sigma_kms
        The declared instrumental Gaussian width, in km/s. Fixed, never fitted: a
        stationary LSF is exactly absorbed by the free component spectra, so disentangling
        cannot identify it (``docs/math.md`` §1.3), and fitting it here would relocate that
        degeneracy rather than resolve it.
    std
        Per-pixel posterior standard deviations, ``Fit.std()``. Defaults to a flat scale,
        in which case the jitter site carries all of the weighting. Supplying real per-pixel
        uncertainties is preferable.
    mh
        Metallicity spec, shared across components by default (one binary, one composition).
        Pass a mapping of name to spec to free them independently. Defaults to the range
        the libraries cover, or to ``Fixed`` when a library has no metallicity axis.
    dilution
        :class:`RadiusRatio` (default for two or more stars), :class:`ScalarDilution`
        (default for one), or :class:`FixedDilution`.
    compare
        ``"native"`` (default) compares the intrinsic model against ``d_hat`` directly.
        ``"matched"`` convolves both sides with the declared LSF first. The latter appears
        the more careful choice, since ``d_hat`` is a regularized partial deconvolution and
        is therefore shrunk toward zero at fine scales by the smoothness prior, but
        measurement contradicts it and the default was changed accordingly (D55).

        Convolving the residuals correlates them over the kernel width while the likelihood
        stays diagonal, so the mis-specification costs a factor of ``1 / sum(k^2)`` in
        chi-square and ``v sin i`` absorbs it. On AI Phe (HARPS, R = 115,000) matched drove
        both components to the ``v sin i`` floor and inflated chi-square by 4.26x against
        native, where the kernel predicts 4.91x, which accounts for the whole gap. Use
        ``"matched"`` only with a residual-covariance model that can carry the correlation.
    offset_order
        Degree of the additive Chebyshev nuisance per component. The ``m = 0`` term is the
        unconstrained zero point of ``docs/math.md`` §5.1; the default of 2 also absorbs
        the slope and curvature that the low-``k`` exchange modes leave behind.
    exclude_angstrom
        Wavelength ranges to drop: nebular cores, detector gaps, and any region the
        disentangling could not model.
    scan_vsini, scan_velocities
        Trial values for the warm-start scan. Default to three widths spanning the ``vsini``
        prior and five velocities spanning the ``v_kms`` prior.
    top_k
        How many of the scan's best starting points to run L-BFGS from. More than one,
        because a label surface with two basins is common; the report states when the
        runners-up were close.
    max_steps
        L-BFGS steps per start. ``tol`` is an absolute gradient-norm threshold on a
        potential whose scale grows with the pixel count, so it is unreachable on real data
        and ``converged`` reads ``False`` however good the fit is; :func:`albireo.run_map`
        documents the same caveat. Judge convergence from the chi-square against its nulls,
        which :meth:`LabelMatch.summary` prints.

    Returns
    -------
    LabelMatch
        Labels, uncertainties, the light ratio the fit measured, the scan surface, and the
        nulls against which each should be read.

    References
    ----------
    Tkachenko, A. 2015, A&A, 581, A129
    Gebruers, S., Tkachenko, A., Bowman, D. M., et al. 2022, A&A, 665, A36
    """
    names = tuple(stars)
    if not names:
        raise ValueError("stars is empty; declare at least one component")
    if medium not in SUPPORTED_MEDIA:
        raise ValueError(f"medium must be one of {SUPPORTED_MEDIA}, got {medium!r}")
    if compare not in ("matched", "native"):
        raise ValueError(f"compare must be 'matched' or 'native', got {compare!r}")

    data = np.atleast_2d(np.asarray(d_hat, dtype=np.float64))
    if data.shape != (len(names), grid.n):
        raise ValueError(
            f"d_hat must be (n_star, n_pix) = ({len(names)}, {grid.n}) to match `stars` and "
            f"`grid`; got {data.shape}. Drop telluric and nebular rows before calling."
        )
    ell0 = np.asarray(light_fractions, dtype=np.float64).reshape(-1)
    if ell0.size != len(names) or np.any(ell0 <= 0):
        raise ValueError(f"light_fractions must give one positive value per star ({len(names)})")
    if lsf_sigma_kms <= 0:
        raise ValueError("lsf_sigma_kms must be positive")

    declared = {name: stars[name] for name in names}
    dilution = (
        dilution
        if dilution is not None
        else (RadiusRatio() if len(names) > 1 else ScalarDilution())
    )
    if isinstance(dilution, RadiusRatio) and len(names) < 2:
        raise ValueError(
            "RadiusRatio needs at least two components to tie together; a single "
            "disentangled component carries no light-ratio information, so use "
            "ScalarDilution() and read its factor as the weaker measurement it is."
        )
    dilution_kind = {
        RadiusRatio: "radius_ratio",
        ScalarDilution: "scalar",
        FixedDilution: "fixed",
    }.get(type(dilution))
    if dilution_kind is None:
        raise TypeError("dilution must be RadiusRatio(), ScalarDilution() or FixedDilution()")

    # -- libraries onto the model grid ------------------------------------
    projected, interpolators, label_axes = [], [], []
    for name in names:
        library = declared[name].library.resampled_to(grid, medium=medium)
        projected.append(library)
        interpolators.append(library_interpolator(library, method=interpolation))
        label_axes.append(tuple(library.label_names))
    hull_guard = any(not hasattr(i, "axes") for i in interpolators)

    # -- data side ---------------------------------------------------------
    sigma = (
        np.full_like(data, float(np.std(data)) or 1.0)
        if std is None
        else np.atleast_2d(np.asarray(std, dtype=np.float64))
    )
    if sigma.shape != data.shape:
        raise ValueError(f"std must match d_hat's shape {data.shape}; got {sigma.shape}")

    lsf_kernel = np.asarray(gaussian_kernel(lsf_sigma_kms / grid.dv_kms))
    matched = compare == "matched"
    if matched:
        # Convolve both sides, so the comparison happens in the space the data
        # constrained.
        data = np.stack([np.convolve(row, lsf_kernel, mode="same") for row in data])
        # Convolution correlates the noise; this is the scale of the smoothed residual,
        # and the jitter site carries whatever the approximation misses.
        sigma = np.sqrt(
            np.stack([np.convolve(row**2, lsf_kernel**2, mode="same") for row in sigma])
        )
    sigma = np.maximum(sigma, 1e-12 * float(np.max(np.abs(data)) or 1.0))

    weight = np.ones_like(data)
    for lo, hi in exclude_angstrom:
        weight[:, (grid.wave >= lo) & (grid.wave <= hi)] = 0.0
    if not np.any(weight):
        raise ValueError("exclude_angstrom removed every pixel")

    # -- specs -------------------------------------------------------------
    from albireo.facade import Between, Fixed  # local: the façade imports this module lazily

    specs: dict[str, Any] = {}
    shared_mh = not isinstance(mh, dict)
    mh_bounds = [library.bounds["mh"] for library in projected if "mh" in library.label_names]
    if shared_mh:
        default_mh = (
            Between(max(b[0] for b in mh_bounds), min(b[1] for b in mh_bounds))
            if mh_bounds
            else Fixed(0.0)
        )
        specs["mh"] = _as_spec(mh, "mh", default_mh)
    for i, name in enumerate(names):
        star, library = declared[name], projected[i]
        specs[f"teff_{name}"] = _as_spec(
            star.teff, f"teff for {name}", Between(*library.bounds["teff"])
        )
        specs[f"logg_{name}"] = _as_spec(
            star.logg, f"logg for {name}", Between(*library.bounds["logg"])
        )
        specs[f"vsini_{name}"] = _as_spec(star.vsini, f"vsini for {name}", Between(0.0, 300.0))
        specs[f"v_{name}"] = _as_spec(star.v_kms, f"v_kms for {name}", Between(-50.0, 50.0))
        if offset_order is not None:
            specs[f"offset_{name}"] = _normal_spec(offset_order + 1, offset_scale)
        # Bounded rather than a wide normal. The jitter's maximum-likelihood point is the
        # RMS residual, so as a fit approaches perfect the scale runs to zero and the
        # log-determinant term grows without limit; with a normal prior the likelihood wins
        # by a factor of the pixel count, the site diverges, the gradient norm reaches 1e6
        # and L-BFGS stalls. The bound states that the quoted per-pixel errors are wrong by
        # at most a factor of five.
        specs[f"log_jitter_{name}"] = Between(float(np.log(0.2)), float(np.log(5.0)), start_at=0.0)
        if not shared_mh:
            specs[f"mh_{name}"] = _as_spec(
                mh.get(name),
                f"mh for {name}",
                Between(*library.bounds["mh"]) if "mh" in library.label_names else Fixed(0.0),
            )
        if "mh" not in library.label_names:
            key = "mh" if shared_mh else f"mh_{name}"
            if not _is_fixed(specs[key]):
                raise ValueError(
                    f"the library for {name!r} has no metallicity axis (it was computed at a "
                    f"single composition, {library.meta.get('mh', 'see its metadata')}), so "
                    f"{key} must be Fixed. Declare mh=Fixed(<the grid's value>)."
                )

    if dilution_kind == "radius_ratio":
        for i, name in enumerate(names[1:], start=1):
            start = float(np.sqrt(ell0[i] / ell0[0]))
            specs[f"ratio_{name}"] = _as_spec(
                dilution.ratio, f"ratio for {name}", Between(0.02, 50.0, start_at=start)
            )
    elif dilution_kind == "scalar":
        for i, name in enumerate(names):
            specs[f"scale_{name}"] = _as_spec(
                dilution.factor, f"factor for {name}", Between(1e-3, 1.0, start_at=float(ell0[i]))
            )

    # -- the traced problem ------------------------------------------------
    vsini_upper = max(specs[f"vsini_{name}"].upper() for name in names)
    problem = LabelProblem(
        interpolators=tuple(interpolators),
        data=jnp.asarray(data),
        sigma=jnp.asarray(sigma),
        weight=jnp.asarray(weight),
        basis=jnp.asarray(_chebyshev_basis(grid.n, offset_order)),
        ell0=jnp.asarray(ell0),
        lsf_kernel=jnp.asarray(lsf_kernel),
        macro_kernels=tuple(
            jnp.asarray(
                gaussian_kernel(declared[name].macro_kms / grid.dv_kms)
                if declared[name].macro_kms > 0
                else np.ones(1)
            )
            for name in names
        ),
        dx=float(grid.dx),
        dv_kms=float(grid.dv_kms),
        rot_radius=rotational_radius_for(max(vsini_upper, grid.dv_kms), grid.dv_kms),
        relativistic=bool(grid.relativistic),
        matched=matched,
        dilution=dilution_kind,
        label_axes=tuple(label_axes),
    )
    config = {
        "names": names,
        "shared_mh": shared_mh,
        "jitter": jitter,
        "hull_guard": hull_guard,
        "offsets": offset_order is not None,
    }

    # -- warm start --------------------------------------------------------
    if scan_vsini is None:
        scan_vsini = [
            float(specs[f"vsini_{names[0]}"].start()),
            0.25 * vsini_upper,
            0.6 * vsini_upper,
        ]
    if scan_velocities is None:
        reach = max(specs[f"v_{name}"].upper() for name in names)
        scan_velocities = np.linspace(-reach, reach, 5) if reach > 0 else [0.0]

    node_tables = [np.asarray(library.nodes) for library in projected]
    scans = []
    for i, name in enumerate(names):
        scan = _scan_component(problem, i, jnp.asarray(node_tables[i]), scan_vsini, scan_velocities)
        allowed = _allowed_nodes(node_tables[i], label_axes[i], specs, name, shared_mh)
        if not allowed.any():
            raise ValueError(
                f"no library node for {name!r} satisfies the declared priors. The grid covers "
                f"{projected[i].bounds}; widen the priors, or check that they are in the same "
                "units as the grid."
            )
        scan[~allowed, 0] = np.inf
        scans.append(scan)
    candidates = _combine_scans(scans, node_tables, label_axes, shared_mh, top_k)

    # -- MAP from each candidate ------------------------------------------
    model = label_model(problem, specs, config)
    best_result, best_start = None, None
    for _, picks in candidates:
        init = _init_from_scan(specs, config, names, label_axes, node_tables, scans, picks)
        try:
            result = run_map(
                model,
                init=init,
                max_steps=max_steps,
                tol=tol,
                rng_key=jax.random.PRNGKey(seed),
                callback=progress,
            )
        except FloatingPointError:
            continue
        if best_result is None or result.potential < best_result.potential:
            best_result, best_start = result, init
    if best_result is None:
        raise RuntimeError(
            "every starting point produced a non-finite gradient. Check that d_hat, std and "
            "light_fractions are finite and that the libraries cover the declared ranges."
        )

    covariance, site_order = _laplace(model, best_result, seed)
    return LabelMatch(
        names=names,
        result=best_result,
        problem=problem,
        specs=specs,
        config=config,
        libraries=tuple(projected),
        covariance=covariance,
        site_order=site_order,
        node_scan=tuple(scans),
        node_tables=tuple(node_tables),
        candidates=tuple(candidates),
        wave=np.asarray(grid.wave),
        start=best_start,
        assumptions={
            "light_fractions": ell0.tolist(),
            "lsf_sigma_kms": float(lsf_sigma_kms),
            "medium": medium,
            "compare": compare,
            "macro_kms": {name: declared[name].macro_kms for name in names},
            "offset_order": offset_order,
            "dilution": dilution_kind,
            "vmicro": {
                name: projected[i].meta.get("vmicro", "unrecorded") for i, name in enumerate(names)
            },
            "grids": {
                name: projected[i].meta.get("grid", "unnamed") for i, name in enumerate(names)
            },
        },
    )


def _normal_spec(size, scale):
    """A zero-mean normal spec for a nuisance: vector when ``size`` is given, else scalar."""
    if size is None:
        return _NormalSpec(dist.Normal(0.0, scale), 0.0)
    return _NormalSpec(dist.Normal(jnp.zeros(size), scale), jnp.zeros(size))


@dataclass(frozen=True)
class _NormalSpec:
    distribution_: Any
    start_at: Any

    def distribution(self):
        return self.distribution_

    def start(self):
        return self.start_at

    def upper(self) -> float:
        return float(np.max(np.abs(np.atleast_1d(np.asarray(self.start_at, dtype=float)))) + 1.0)


def _clip_to_support(spec, value: float) -> float:
    """Nudge a starting value strictly inside a bounded prior.

    A start exactly on a Uniform's boundary maps to an infinite unconstrained coordinate,
    which numpyro reports only as "cannot find valid initial parameters".
    """
    bounds = _spec_bounds(spec)
    if bounds is None:
        return value
    lo, hi = bounds
    pad = 1e-6 * (hi - lo)
    return float(np.clip(value, lo + pad, hi - pad))


def _init_from_scan(specs, config, names, label_axes, node_tables, scans, picks):
    """Constrained starting values for every sampled site, from one scan candidate."""
    init = {}
    for i, name in enumerate(names):
        node = node_tables[i][picks[i]]
        axes = label_axes[i]
        for label in _LABEL_ORDER:
            if label not in axes:
                continue
            value = float(node[axes.index(label)])
            key = label if (label == "mh" and config["shared_mh"]) else f"{label}_{name}"
            if not _is_fixed(specs[key]):
                init[key] = _clip_to_support(specs[key], value)
        for key, value in (
            (f"vsini_{name}", float(scans[i][picks[i], 1])),
            (f"v_{name}", float(scans[i][picks[i], 2])),
        ):
            if not _is_fixed(specs[key]):
                init[key] = _clip_to_support(specs[key], value)
        if config["offsets"]:
            init[f"offset_{name}"] = np.zeros(np.shape(specs[f"offset_{name}"].start()))
        if config["jitter"]:
            init[f"log_jitter_{name}"] = 0.0
    for key, spec in specs.items():
        if key.startswith(("ratio_", "scale_")) and not _is_fixed(spec):
            init[key] = _clip_to_support(spec, float(np.asarray(spec.start())))
    return init


def _laplace(model, result, seed):
    """Laplace covariance in the unconstrained space, and the labels of its rows.

    The row labels are built from the sites' shapes rather than from their names: the
    Chebyshev offsets are vectors, so a covariance row is not a site. Zipping sorted site
    names against the diagonal shifts every entry after the first vector site, which presents
    as an implausibly small uncertainty rather than as an error.
    """
    flat = {key: np.atleast_1d(np.asarray(value)) for key, value in result.unconstrained.items()}
    labels: list[str] = []
    for key in sorted(flat):  # jax ravels dict pytrees in sorted-key order
        size = flat[key].size
        labels.extend([key] if size == 1 else [f"{key}[{i}]" for i in range(size)])
    try:
        inverse_mass = np.asarray(
            laplace_inverse_mass(model, result.params, rng_key=jax.random.PRNGKey(seed + 1))
        )
    except Exception:  # pragma: no cover - a singular Hessian is reported, not raised
        return None, tuple(labels)
    if inverse_mass.shape[0] != len(labels):  # pragma: no cover - shape contract changed
        return None, tuple(labels)
    return inverse_mass, tuple(labels)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelMatch:
    """Labels for each component, with the nulls against which they should be read.

    Each number is reported against a reference. The chi-square is quoted against a fit with
    no template at all and against the best raw grid node, so that the fitted value can be
    compared with what a null achieves. Each label's posterior width is quoted against its
    prior width, since a parameter returned at its prior width was not measured. The Laplace
    error is quoted against the spread from refitting the disentangling posterior's own
    draws, because on correlated residuals the formal error runs five to ten times optimistic
    (Gebruers et al. 2022).

    References
    ----------
    Gebruers, S., Tkachenko, A., Bowman, D. M., et al. 2022, A&A, 665, A36
    """

    names: tuple
    result: Any
    problem: LabelProblem
    specs: dict = field(repr=False)
    config: dict = field(repr=False)
    libraries: tuple = field(repr=False)
    covariance: Any
    site_order: tuple
    node_scan: tuple = field(repr=False)
    node_tables: tuple = field(repr=False)
    candidates: tuple = field(repr=False)
    wave: np.ndarray = field(repr=False)
    start: dict = field(repr=False)
    assumptions: dict = field(default_factory=dict)
    draws: Any = field(default=None, repr=False)

    # -- labels ------------------------------------------------------------

    def _value(self, key):
        if key in self.result.params:
            return float(np.asarray(self.result.params[key]))
        return float(np.asarray(self.specs[key].start()))

    @property
    def labels(self) -> dict[str, dict[str, float]]:
        """MAP labels per component, including the ones that were held fixed."""
        out = {}
        for i, name in enumerate(self.names):
            axes = self.problem.label_axes[i]
            entry = {}
            for label in _LABEL_ORDER:
                if label not in axes:
                    continue
                key = label if (label == "mh" and self.config["shared_mh"]) else f"{label}_{name}"
                entry[label] = self._value(key)
            entry["vsini"] = self._value(f"vsini_{name}")
            entry["v_kms"] = self._value(f"v_{name}")
            out[name] = entry
        return out

    @property
    def fixed(self) -> dict[str, list[str]]:
        """Which labels were held fixed for each component."""
        return {
            name: sorted(
                label
                for label in ("teff", "logg", "mh", "vsini", "v_kms")
                for key in [
                    "mh"
                    if (label == "mh" and self.config["shared_mh"])
                    else f"{'v' if label == 'v_kms' else label}_{name}"
                ]
                if key in self.specs and _is_fixed(self.specs[key])
            )
            for name in self.names
        }

    @property
    def radius_ratio(self) -> dict[str, float]:
        """Fitted radius ratios R_i / R_first, where the dilution model carries them.

        The shared scalar that converts the grids' own continua into light fractions is a
        radius ratio, so on a system whose radii are known from eclipses there is an external
        number to compare it against. Empty for the other dilution models, which carry no
        such scalar.
        """
        if self.problem.dilution != "radius_ratio":
            return {}
        return {self.names[0]: 1.0} | {
            name: self._value(f"ratio_{name}") for name in self.names[1:]
        }

    def errors(self, method: str = "laplace") -> dict[str, dict[str, float]]:
        """Label uncertainties.

        ``"laplace"`` is the curvature at the MAP, projected to the constrained
        parameterization by the delta method. ``"draws"`` is the spread over refits of the
        component-spectrum posterior draws (:func:`refit_draws`), which is typically several
        times wider. The difference is the part of the error budget that formal curvature
        cannot see (``docs/math.md`` §9.5).
        """
        if method == "draws":
            if self.draws is None:
                raise ValueError(
                    "no draws have been refitted yet. Call refit_draws(match, draws) with "
                    "joint draws of the component spectra (Posterior.spectra() or "
                    "albireo.draw_spectra) and read .errors('draws') on what it returns."
                )
            return self.draws
        if method != "laplace":
            raise ValueError("method must be 'laplace' or 'draws'")
        if self.covariance is None:
            return {name: {} for name in self.names}
        sigma = dict(
            zip(
                self.site_order,
                np.sqrt(np.clip(np.diag(self.covariance), 0.0, None)),
                strict=True,
            )
        )
        out: dict[str, dict[str, float]] = {}
        for name in self.names:
            entry = {}
            for label in (*_LABEL_ORDER, "vsini", "v_kms"):
                key = self._site_key(label, name)
                if key is None or key not in sigma:
                    continue
                entry[label] = float(sigma[key] * self._jacobian(key))
            out[name] = entry
        return out

    def _site_key(self, label: str, name: str) -> str | None:
        if label == "mh" and self.config["shared_mh"]:
            key = "mh"
        elif label == "v_kms":
            key = f"v_{name}"
        else:
            key = f"{label}_{name}"
        if key not in self.specs or _is_fixed(self.specs[key]):
            return None
        return key

    def _jacobian(self, key: str) -> float:
        """d(constrained)/d(unconstrained) at the MAP, for the delta method.

        numpyro optimizes bounded sites through a logistic transform, so the unconstrained
        standard deviation must be pushed back through it rather than reported as if it were
        the parameter's own.
        """
        distribution = self.specs[key].distribution()
        low = getattr(distribution, "low", None)
        high = getattr(distribution, "high", None)
        if low is None or high is None:
            return 1.0
        span = float(np.asarray(high) - np.asarray(low))
        value = self._value(key)
        u = np.clip((value - float(np.asarray(low))) / span, 1e-12, 1 - 1e-12)
        return span * u * (1.0 - u)

    # -- identifiability ---------------------------------------------------

    @property
    def correlation(self) -> dict:
        """Laplace correlation matrix, as ``{"sites": [...], "matrix": array}``."""
        if self.covariance is None:
            return {"sites": list(self.site_order), "matrix": None}
        sigma = np.sqrt(np.clip(np.diag(self.covariance), 1e-300, None))
        return {
            "sites": list(self.site_order),
            "matrix": self.covariance / np.outer(sigma, sigma),
        }

    def flagged_correlations(self, threshold: float = 0.95) -> list[tuple[str, str, float]]:
        """Site pairs the fit could not separate, worst first.

        A Teff / log g pair near 0.98 with both free is the published behaviour of this
        problem (Tamajo et al. 2011) rather than a defect; the remedy is to fix log g from
        the eclipsing solution.

        References
        ----------
        Tamajo, E., Pavlovski, K. & Southworth, J. 2011, A&A, 526, A76
        """
        report = self.correlation
        if report["matrix"] is None:
            return []
        sites, matrix = report["sites"], report["matrix"]
        found = [
            (sites[a], sites[b], float(matrix[a, b]))
            for a in range(len(sites))
            for b in range(a + 1, len(sites))
            if abs(matrix[a, b]) >= threshold
        ]
        return sorted(found, key=lambda item: -abs(item[2]))

    @property
    def prior_width(self) -> dict[str, float]:
        """Prior standard deviation per free site, the denominator of the width ratio."""
        out = {}
        for key, spec in self.specs.items():
            distribution = spec.distribution()
            if distribution is None:
                continue
            low, high = getattr(distribution, "low", None), getattr(distribution, "high", None)
            if low is not None and high is not None:
                out[key] = float(np.asarray(high) - np.asarray(low)) / np.sqrt(12.0)
            elif hasattr(distribution, "scale"):
                out[key] = float(np.max(np.asarray(distribution.scale)))
        return out

    @property
    def posterior_over_prior(self) -> dict[str, float]:
        """Posterior width divided by prior width, per free site.

        A ratio near 1 means the data constrained that parameter negligibly and the number
        reported is the prior. The sensitivity forecast quotes the same null.
        """
        if self.covariance is None:
            return {}
        widths = self.prior_width
        sigma = dict(
            zip(
                self.site_order,
                np.sqrt(np.clip(np.diag(self.covariance), 0.0, None)),
                strict=True,
            )
        )
        return {
            key: float(sigma[key] * self._jacobian(key) / widths[key])
            for key in widths
            if key in sigma and widths[key] > 0
        }

    @property
    def hit_step_cap(self) -> bool:
        """Whether L-BFGS stopped on its step budget rather than on the gradient."""
        return not bool(self.result.converged)

    @property
    def multimodal(self) -> bool:
        """Whether a runner-up starting basin came within ``delta chi-square < 9``."""
        if len(self.candidates) < 2:
            return False
        best, second = self.candidates[0][0], self.candidates[1][0]
        return bool(second - best < 9.0 and self.candidates[0][1] != self.candidates[1][1])

    # -- goodness, against nulls -------------------------------------------

    def _rows(self):
        params = dict(self.result.params)
        labels, vsini, v_kms, offsets = [], [], [], []
        for i, name in enumerate(self.names):
            axes = self.problem.label_axes[i]
            values = {}
            for label in _LABEL_ORDER:
                key = "mh" if (label == "mh" and self.config["shared_mh"]) else f"{label}_{name}"
                if key in self.specs:
                    values[label] = jnp.asarray(
                        params[key] if key in params else self.specs[key].start()
                    )
            labels.append(jnp.stack([values[axis] for axis in axes]))
            vsini.append(jnp.asarray(self._value(f"vsini_{name}")))
            v_kms.append(jnp.asarray(self._value(f"v_{name}")))
            offsets.append(
                jnp.asarray(params[f"offset_{name}"]) if self.config["offsets"] else jnp.zeros(0)
            )
        if self.problem.dilution == "radius_ratio":
            amplitudes = (
                jnp.stack(
                    [jnp.asarray(1.0)]
                    + [jnp.asarray(self._value(f"ratio_{name}")) for name in self.names[1:]]
                )
                ** 2
            )
        elif self.problem.dilution == "scalar":
            amplitudes = jnp.stack(
                [jnp.asarray(self._value(f"scale_{name}")) for name in self.names]
            )
        else:
            amplitudes = self.problem.ell0
        return _model_rows(self.problem, labels, vsini, v_kms, amplitudes, jnp.stack(offsets))

    @property
    def chi2(self) -> float:
        """Chi-square at the MAP, without the jitter rescaling, so the nulls compare."""
        residual = (self._rows() - self.problem.data) / self.problem.sigma
        return float(jnp.sum(self.problem.weight * residual**2))

    @property
    def chi2_continuum(self) -> float:
        """The null with no template at all: the additive nuisance alone, profiled.

        A fitted chi-square that is not well below this indicates that the spectrum carried
        no label information, and that the reported labels are the priors.
        """
        total = 0.0
        for i in range(self.problem.n_star):
            weight = self.problem.weight[i] / self.problem.sigma[i] ** 2
            basis = self.problem.basis
            gram = basis.T @ (weight[:, None] * basis)
            gram_inv = jnp.linalg.inv(gram + 1e-10 * jnp.trace(gram) * jnp.eye(basis.shape[1]))
            total += float(
                _profiled_chi2(
                    jnp.zeros(self.problem.n_pix), self.problem.data[i], weight, basis, gram_inv
                )
            )
        return total

    @property
    def chi2_nearest_node(self) -> float:
        """The null of snapping to the best raw grid node, the practical alternative.

        The gap between this and :attr:`chi2` measures what continuous interpolation, fitted
        broadening and fitted dilution contribute over selecting the nearest node.
        """
        return float(sum(np.min(scan[:, 0]) for scan in self.node_scan))

    @property
    def n_pixels_used(self) -> int:
        """Pixels contributing to the likelihood, after exclusions."""
        return int(np.asarray(self.problem.weight).sum())

    # -- derived astrophysics ---------------------------------------------

    def light_fractions(self, wave=None):
        """Fitted light fractions per component, shape ``(n_star, n_pix)``.

        The spectroscopic light ratio is a deliverable in its own right: published values
        match light-curve ratios to a few percent, and cross-correlation codes downstream are
        more sensitive to an incorrect flux ratio than to an incorrect temperature. Returns
        the assumed fractions unchanged for a :class:`FixedDilution` fit.
        """
        rows = self._light_fraction_rows()
        if wave is None:
            return np.asarray(rows)
        return np.stack([np.interp(np.asarray(wave), self.wave, np.asarray(row)) for row in rows])

    def _light_fraction_rows(self):
        if self.problem.dilution == "fixed":
            return jnp.broadcast_to(
                self.problem.ell0[:, None], (self.problem.n_star, self.problem.n_pix)
            )
        params = dict(self.result.params)
        if self.problem.dilution == "scalar":
            values = jnp.stack([jnp.asarray(self._value(f"scale_{n}")) for n in self.names])
            return jnp.broadcast_to(values[:, None], (self.problem.n_star, self.problem.n_pix))
        amplitudes = (
            jnp.stack(
                [jnp.asarray(1.0)]
                + [jnp.asarray(self._value(f"ratio_{n}")) for n in self.names[1:]]
            )
            ** 2
        )
        continua = []
        for i, name in enumerate(self.names):
            axes = self.problem.label_axes[i]
            values = {}
            for label in _LABEL_ORDER:
                key = "mh" if (label == "mh" and self.config["shared_mh"]) else f"{label}_{name}"
                if key in self.specs:
                    values[label] = jnp.asarray(
                        params[key] if key in params else self.specs[key].start()
                    )
            continua.append(self.problem.interpolators[i](jnp.stack([values[a] for a in axes]))[1])
        return _light_fractions(continua, amplitudes)

    @property
    def flux_ratio(self) -> dict[str, float]:
        """Band-median light fraction per component, the value pipelines request."""
        rows = np.asarray(self._light_fraction_rows())
        return {name: float(np.median(rows[i])) for i, name in enumerate(self.names)}

    def template(self, name: str) -> np.ndarray:
        """The MAP model spectrum for one component, as flux on the fit's grid.

        Broadened and shifted as fitted, and undiluted: a template is the star, not the
        star's share of the system's light. Writing it to a file is the task of
        :mod:`albireo.handoff`.
        """
        if name not in self.names:
            raise ValueError(f"unknown component {name!r}; declared: {', '.join(self.names)}")
        index = self.names.index(name)
        params = dict(self.result.params)
        axes = self.problem.label_axes[index]
        values = {}
        for label in _LABEL_ORDER:
            key = "mh" if (label == "mh" and self.config["shared_mh"]) else f"{label}_{name}"
            if key in self.specs:
                values[label] = jnp.asarray(
                    params[key] if key in params else self.specs[key].start()
                )
        deviation, _ = _component_model(
            self.problem,
            index,
            jnp.stack([values[a] for a in axes]),
            jnp.asarray(self._value(f"vsini_{name}")),
            jnp.asarray(self._value(f"v_{name}")),
        )
        return np.asarray(1.0 + deviation)

    def nearest_node(self, name: str) -> dict[str, float]:
        """The closest library node to the fitted labels.

        Pipelines that take a menu choice rather than arbitrary labels (HERMES's fixed
        masks, Gaia's ``rv_template_*`` grid) require the answer snapped to a node they hold,
        and the grid's own step is the appropriate granularity.
        """
        index = self.names.index(name)
        table = self.node_tables[index]
        axes = self.problem.label_axes[index]
        fitted = np.array([self.labels[name][axis] for axis in axes])
        span = np.maximum(table.max(axis=0) - table.min(axis=0), 1e-30)
        nearest = table[int(np.argmin(np.sum(((table - fitted) / span) ** 2, axis=1)))]
        return dict(zip(axes, (float(v) for v in nearest), strict=True))

    # -- reporting ---------------------------------------------------------

    def summary(self) -> str:
        """Formatted report of the labels, their nulls, and the assumptions behind them."""
        laplace = self.errors("laplace")
        drawn = self.draws
        lines = [
            f"LabelMatch: {len(self.names)} component(s), {self.n_pixels_used} pixels, "
            f"chi2 = {self.chi2:.1f}",
            f"  null (no template)   chi2 = {self.chi2_continuum:.1f}",
            f"  null (nearest node)  chi2 = {self.chi2_nearest_node:.1f}",
            "",
        ]
        for name in self.names:
            lines.append(f"  {name}:")
            fixed = set(self.fixed[name])
            for label, value in self.labels[name].items():
                unit = {"teff": " K", "vsini": " km/s", "v_kms": " km/s"}.get(label, " dex")
                formal = laplace.get(name, {}).get(label)
                text = f"    {label:<6} {value:10.3f}{unit}"
                if label in fixed:
                    text += "   (fixed)"
                elif formal is not None:
                    text += f" +- {formal:.3f} formal"
                    if drawn and label in drawn.get(name, {}):
                        spread = drawn[name][label]
                        text += f", +- {spread:.3f} from draws (x{spread / max(formal, 1e-12):.1f})"
                lines.append(text)
            lines.append(f"    light fraction {self.flux_ratio[name]:.4f} (median over the band)")

        if drawn is None:
            lines += [
                "",
                "  Formal errors only. On disentangled spectra these run 5-10x optimistic,",
                "  because the residuals are correlated rather than white (Gebruers+ 2022:",
                "  70 K formal against 425 K realistic). Refit the spectral posterior draws",
                "  with refit_draws(match, draws) before quoting these anywhere.",
            ]
        weak = {k: v for k, v in self.posterior_over_prior.items() if v > 0.8}
        if weak:
            lines += [
                "",
                "  Learned nothing here (posterior width >= 80% of the prior):",
                "    " + ", ".join(f"{k} ({v:.0%})" for k, v in sorted(weak.items())),
            ]
        flagged = self.flagged_correlations()
        if flagged:
            lines += ["", "  Degenerate pairs:"]
            for a, b, rho in flagged:
                note = ""
                if {a.split("_")[0], b.split("_")[0]} == {"teff", "logg"}:
                    note = "  <- expected when both are free; fix log g from the orbit"
                lines.append(f"    {a} / {b}: {rho:+.3f}{note}")
        if self.multimodal:
            lines += [
                "",
                "  The scan found a second basin within delta chi2 < 9. The reported labels are",
                "  the better of them, not the only ones consistent with the data.",
            ]
        if self.hit_step_cap:
            lines += [
                "",
                f"  L-BFGS used all {self.result.num_steps} steps it was given "
                f"(final gradient norm {self.result.grad_norm:.3g}). That is normal: the",
                "  tolerance is absolute and unreachable at this scale, but if the fitted",
                "  chi-square is not well below the nearest-node null, re-run with more.",
            ]
        lines += ["", "  Assumed, not measured:"]
        for key, value in self.assumptions.items():
            lines.append(f"    {key}: {value}")
        if self.assumptions.get("dilution") == "scalar":
            lines.append(
                "    (a wavelength-independent dilution: weaker than a joint radius-ratio fit)"
            )
        return "\n".join(lines)


def refit_draws(match: LabelMatch, draws, *, max_steps: int = 60, seed: int = 0) -> LabelMatch:
    """Refit the labels once per posterior draw of the component spectra.

    The Laplace covariance measures the curvature of the likelihood at the optimum, which
    on correlated residuals understates the uncertainty. This function instead measures how
    far the labels move when the component spectra move as the disentangling posterior
    permits, including the exchange modes that trade flux between components
    (``docs/math.md`` §9.5). It performs internally the loop that
    :func:`albireo.handoff.export_draws` documents for an external code.

    Parameters
    ----------
    match
        A completed fit, whose priors, data weighting and nuisances are reused unchanged.
    draws
        Joint draws of the stellar component spectra, shape ``(n_draws, n_star, n_pix)``,
        from ``Posterior.spectra()`` or :func:`albireo.draw_spectra`, sliced to the stellar
        rows. They must be joint: independent per-component draws would omit the correlation
        this function propagates.
    max_steps
        L-BFGS steps per draw, starting from the MAP, which is normally very close.

    Returns
    -------
    LabelMatch
        The same fit with ``.draws`` populated, so ``.errors("draws")`` and
        ``.summary()`` report the spread alongside the formal error.
    """
    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[1:] != (match.problem.n_star, match.problem.n_pix):
        raise ValueError(
            f"draws must be (n_draws, n_star, n_pix) = (*, {match.problem.n_star}, "
            f"{match.problem.n_pix}); got {draws.shape}"
        )
    kernel = np.asarray(match.problem.lsf_kernel)
    collected: dict[str, dict[str, list[float]]] = {n: {} for n in match.names}
    for index, draw in enumerate(draws):
        rows = (
            np.stack([np.convolve(row, kernel, mode="same") for row in draw])
            if match.problem.matched
            else draw
        )
        problem = LabelProblem(
            **{
                **{
                    f.name: getattr(match.problem, f.name)
                    for f in match.problem.__dataclass_fields__.values()
                },
                "data": jnp.asarray(rows),
            }
        )
        model = label_model(problem, match.specs, match.config)
        try:
            result = run_map(
                model,
                init=match.start,
                max_steps=max_steps,
                rng_key=jax.random.PRNGKey(seed + index),
            )
        except FloatingPointError:  # pragma: no cover - a pathological draw is skipped
            continue
        refit = LabelMatch(
            **{
                **{f.name: getattr(match, f.name) for f in match.__dataclass_fields__.values()},
                "result": result,
                "problem": problem,
            }
        )
        for name, entry in refit.labels.items():
            for label, value in entry.items():
                collected[name].setdefault(label, []).append(value)

    spread = {
        name: {
            label: float(np.std(values, ddof=1))
            for label, values in entry.items()
            if len(values) > 1
        }
        for name, entry in collected.items()
    }
    if not any(spread.values()):
        raise RuntimeError("no draw refitted successfully; nothing to take a spread over")
    return LabelMatch(
        **{
            **{f.name: getattr(match, f.name) for f in match.__dataclass_fields__.values()},
            "draws": spread,
        }
    )
