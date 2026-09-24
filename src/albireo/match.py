"""Stellar labels for disentangled components: Teff, log g, [M/H] and v sin i.

**Experimental.** The label vocabulary and the return shapes may change.

The mode fits atmospheric labels to the component spectra of a disentangling, against a
published synthetic grid (:mod:`albireo.library`), so that a component can serve as a
radial-velocity template for TODCOR (:mod:`albireo.todcor`), saphires, iSpec or a survey
pipeline. Fitting here retains the per-pixel uncertainties of the disentangling posterior
and the joint measurement of the components. The equations are in ``docs/math.md`` §9.

The module does no synthesis: no line list, no model atmosphere, no radiative transfer and
no individual abundances. Those remain with GSSP, iSpec, Korg.jl and PySME, reached through
:mod:`albireo.handoff`. The target accuracy is that at which the template stops limiting the
velocities: roughly 2-3% in Teff, 0.15 dex in log g and [M/H], and 10% in v sin i
(``docs/math.md`` §9.6). A label from this mode is a template coordinate, not an entry in an
abundance table.

Three model choices follow. First, dilution is fitted jointly. Disentangling returns
component spectra in the common continuum, scaled by assumed light fractions, and the
likelihood constrains only the products ``l_i d_i``, so an error in the assumed ``l``
rescales every line depth and is degenerate with Teff. The components are fitted together
with one shared scalar per companion (a radius ratio), and the wavelength dependence of the
light fractions is taken from the grids' own continua, so they sum to one at every
wavelength by construction. This is the binary mode of GSSP (Tkachenko 2015), where a
wavelength-independent dilution was measured to shift a secondary's Teff by 275 K.

Second, the zero point is modelled explicitly. Each component's constant offset, and very
nearly its slope, lies in the null space of the disentangling problem and is held only by
the smoothness prior's ridge (``docs/math.md`` §5.1); left unmodelled it lands on the line
depths and returns as a Teff error. Each component carries a low-order additive Chebyshev
nuisance whose zeroth term is that zero point, fitted and reported (``docs/math.md`` §9.1).
The nuisance is additive because the null space lives in the continuum, where the deviation
spectrum is zero and a multiplicative term has no effect.

Third, two uncertainties are reported. Disentangling residuals are correlated, and formal
errors on this problem run five to ten times optimistic: Gebruers et al. (2022) report 70 K
formal against 425 K realistic for B stars at S/N 150. The Laplace covariance is quoted
beside the spread from refitting joint posterior draws of the component spectra
(:func:`refit_draws`), and :meth:`LabelMatch.summary` prints both (``docs/math.md`` §9.5).

Three comparisons are offered. ``"native"`` and ``"matched"`` compare the template with the
disentangled component ``d_hat`` under a diagonal likelihood, which is mis-specified, since
``d_hat`` is a smoothed partial deconvolution whose errors are correlated across pixels.
``"epochs"`` compares the template composite with the epoch spectra themselves, through the
disentangling's own sufficient statistics at its MAP (:class:`EpochStatistics`,
``docs/math.md`` §9.2a). Its chi-square is exact for the declared noise model and does not
depend on the declared light fractions; it is minimised by a bounded Levenberg-Marquardt
with restarts rather than by L-BFGS. It is the default of :meth:`albireo.Fit.match_labels`.
On simulated Gaia RVS binaries it halved the temperature error of the native comparison
converged the same way, and cut the light-fraction error by a factor of three
(``docs/benchmarks.md``, Gaia RVS, after the third run).

References
----------
Tkachenko, A. 2015, A&A, 581, A129
Gebruers, S., Tkachenko, A., Bowman, D. M., et al. 2022, A&A, 665, A36
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from albireo.grids import C_KMS, LogGrid, log_doppler_shift
from albireo.inference import MAPResult, laplace_inverse_mass, run_map
from albireo.library import SUPPORTED_MEDIA, SpectralLibrary, library_interpolator
from albireo.operators import (
    gaussian_kernel,
    rotational_kernel_traced,
    rotational_radius_for,
    shift_spectrum,
)

__all__ = [
    "EpochStatistics",
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
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _resolution_sigma_kms(resolving_power: float) -> float:
    """Gaussian sigma in km/s of a profile whose FWHM is ``lambda / R``."""
    return C_KMS / float(resolving_power) * _FWHM_TO_SIGMA


def _quadrature_width(sigma_kms, library_resolving_power, where: str) -> np.ndarray:
    """The width a template at the library's resolving power still needs: elementwise
    ``sqrt(sigma^2 - sigma_lib^2)``, or ``sigma`` itself for an intrinsic library (``None``).

    A published grid already carries a Gaussian of ``sigma_lib``, so convolving it with the
    whole instrument profile broadens it twice. A library at or below the instrument's
    resolving power cannot be brought to it by any convolution, and is refused.
    """
    sigma = np.atleast_1d(np.asarray(sigma_kms, dtype=np.float64))
    if library_resolving_power is None:
        return sigma
    have = _resolution_sigma_kms(library_resolving_power)
    if np.any(sigma <= have):
        raise ValueError(
            f"{where}: the library is broadened to R = {float(library_resolving_power):g} "
            f"(sigma {have:.3f} km/s), which is at or below the instrument's own width "
            f"({float(np.min(sigma)):.3f} km/s). A template cannot be sharpened by a "
            "convolution; use a library at a higher resolving power than the data."
        )
    return np.sqrt(sigma**2 - have**2)


_GRID_VARIANCE_FACTOR = 7.0 / 12.0
"""The discretisation variance of the epoch comparison per squared model pixel.

Five discrete steps lie between a library spectrum and the epoch pixels, and each adds a
variance proportional to ``dv^2`` (``d65_grid_smoothing.md``): the library's box average onto
the model grid (``dv^2 / 12``), the pixel integration of the rotation kernel (``dv^2 / 12``),
the label rows' shift by the frame velocity and the operator's shift of each epoch (each a
two-tap linear interpolation of variance ``f (1 - f) dv^2``, ``dv^2 / 6`` on average over the
fractional shift ``f``), and the model pixel held constant across the rebin onto the native
pixels (``dv^2 / 12``). The sum is ``7 dv^2 / 12``."""

_OPERATOR_FLOOR_PIXELS = 0.5
"""The narrowest compensated operator width, in model pixels. Below it a sampled Gaussian
no longer realises its variance (short by 35 percent of ``sigma^2`` at 0.43 pixel)."""


def _grid_variance(dv_kms: float) -> float:
    """``(7/12) dv^2``, the mean discretisation variance of the epoch comparison, km^2/s^2."""
    return _GRID_VARIANCE_FACTOR * float(dv_kms) ** 2


def _grid_compensated_widths(sigma_q_kms, dv_kms: float) -> tuple[np.ndarray, np.ndarray]:
    """Operator widths with the grid's own smoothing removed, and which of them were floored.

    Elementwise ``sqrt(sigma_q^2 - (7/12) dv^2)``, floored at half a model pixel, or at
    ``sigma_q`` itself when that is narrower, so that the compensation never widens the
    operator. See :meth:`albireo.Fit.epoch_statistics` for the derivation.
    """
    sigma_q = np.atleast_1d(np.asarray(sigma_q_kms, dtype=np.float64))
    floor = np.minimum(_OPERATOR_FLOOR_PIXELS * float(dv_kms), sigma_q)
    wanted = sigma_q**2 - _grid_variance(dv_kms)
    floored = wanted < floor**2
    return np.sqrt(np.where(floored, floor**2, wanted)), floored


def _grid_compensation_notes(floored: dict, narrow: dict, dv_kms: float) -> list[str]:
    """The warnings of a grid compensation that was floored, or that ran below a pixel.

    ``floored`` maps an instrument to the narrowest ``sigma_q`` among its floored widths;
    ``narrow`` maps an instrument to ``(sigma_q, width)`` for the narrowest width that was
    compensated in full with ``dv > sigma_q``.
    """
    dv = float(dv_kms)
    variance = _grid_variance(dv)
    notes = []
    if floored:
        sigma_q = min(floored.values())
        floor = min(_OPERATOR_FLOOR_PIXELS * dv, sigma_q)
        removed = sigma_q**2 - floor**2
        notes.append(
            f"grid compensation floored for {', '.join(sorted(floored))}: the model grid's "
            f"discretisation variance (7/12) dv^2 = {variance:.2f} km^2/s^2 at dv = {dv:.3f} "
            f"km/s leaves {np.sqrt(max(sigma_q**2 - variance, 0.0)):.3f} km/s of sigma_q = "
            f"{sigma_q:.3f} km/s, below half a model pixel, so the operator width was floored "
            f"at {floor:.3f} km/s and removes {removed:.2f} of the {variance:.2f} km^2/s^2; "
            "v sin i stays biased low. A model grid with dv < sigma_q / 1.095 = "
            f"{sigma_q / np.sqrt(1.2):.3f} km/s avoids the floor (docs/math.md 9.2a)"
        )
    if narrow:
        instrument = min(narrow, key=lambda key: narrow[key][1])
        sigma_q, width = narrow[instrument]
        notes.append(
            f"grid compensation below a model pixel for {', '.join(sorted(narrow))}: dv = "
            f"{dv:.3f} km/s exceeds sigma_q = {sigma_q:.3f} km/s, so the compensated operator "
            f"width {width:.3f} km/s is {width / dv:.2f} of a model pixel, where a sampled "
            "Gaussian no longer realises its variance and the compensation is approximate. A "
            f"model grid with dv < sigma_q = {sigma_q:.3f} km/s avoids it (docs/math.md 9.2a)"
        )
    return notes


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StarLabels:
    """The declaration for one component: what is fitted, and what is assumed.

    Each label accepts the declaration vocabulary of the façade
    (:class:`~albireo.facade.Fixed`, :class:`~albireo.facade.Known`,
    :class:`~albireo.facade.Between`, :class:`~albireo.facade.Sampled`) or a bare float,
    which is treated as fixed. The specs are duck-typed, so this module does not depend on
    the façade and can be driven from another code's output.

    ``logg`` requires a decision before the fit. Teff and log g correlate at about 0.98 when
    both are free (Tamajo et al. 2011). For an eclipsing binary the light curve and the orbit
    give log g to 0.01 dex, and fixing it there makes the analysis well posed. A
    non-eclipsing SB2 has no such anchor: the fit is run free and fixed, and the spread is
    reported as the uncertainty.

    ``macro_kms`` is a fixed Gaussian macroturbulence folded into the intrinsic broadening.
    It is not fitted because it is not separable from ``vsini`` at survey resolution. With
    the default of zero, a fitted ``vsini`` measures all broadening beyond the instrument
    profile, which is what a template requires.

    ``None`` for ``teff``, ``logg`` or ``vsini`` adopts the library's own range (0-300 km/s
    for ``vsini``), a starting point rather than a considered prior.

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
    construction, so no constraint site or penalty is needed and none can drift. Their
    wavelength dependence comes from the model atmospheres, not from a fitted polynomial.

    This is the ``gssp_binary`` parameterization (Tkachenko 2015), which makes a
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

    The ``gssp_single`` parameterization (Tkachenko 2015), for a single component, or when
    two grids' continua cannot be trusted on a common scale. It is strictly weaker than
    :class:`RadiusRatio`: nothing ties the components together, nothing enforces the sum to
    one, and the wavelength dependence that carries the light-ratio information is
    discarded. :meth:`LabelMatch.summary` records that a fit used it.

    References
    ----------
    Tkachenko, A. 2015, A&A, 581, A129
    """

    factor: Any = None


@dataclass(frozen=True)
class FixedDilution:
    """Hold the light fractions at their assumed values: ``w_i == l0_i``.

    A diagnostic, not a recommended configuration. It gives the labels with no dilution
    freedom, so the shift between this and a :class:`RadiusRatio` fit measures how far the
    assumed light fractions move the answer.
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
    into the compiled executable.

    ``lsf_kernel`` is the declared instrument profile, which ``"matched"`` applies to
    ``d_hat``. ``model_lsf_kernels`` holds, per star, the profile applied to that star's
    template in ``"matched"`` mode: the quadrature width
    ``sqrt(sigma_inst^2 - sigma_lib^2)`` when its library declares a resolving power, and the
    instrument profile otherwise. The other modes carry no LSF in the template rows.
    """

    interpolators: tuple
    data: jax.Array
    sigma: jax.Array
    weight: jax.Array
    basis: jax.Array
    ell0: jax.Array
    lsf_kernel: jax.Array
    macro_kernels: tuple
    model_lsf_kernels: tuple
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
            self.model_lsf_kernels,
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
    :func:`albireo.operators.rotational_kernel_traced`. In matched mode the instrument
    profile is the star's entry of ``model_lsf_kernels``, which carries only the width its
    library does not already have.

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
        kernel = jnp.convolve(kernel, problem.model_lsf_kernels[index], mode="full")
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
# The epoch comparison: the disentangling's sufficient statistics
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EpochStatistics:
    """What the epoch comparison needs from a disentangling, fixed at its MAP.

    With the orbit, the declared light fractions ``l0``, the LSF and the noise model held at
    the disentangling's MAP, write ``z`` for the stacked epoch data in deviation space,
    ``W`` for its noise precision (diagonal, or the AR(1) chain precision of
    ``docs/math.md`` §1.4a, jitter included) and ``A`` for the stacked operator of §1.4,
    which shifts, weights by ``l0``, convolves with the LSF, rebins and multiplies by the
    response. For stacked component rows ``m`` the epoch chi-square is

    ``L_data(m) = ||z - A m||_W^2 = z^T W z - 2 m^T h + m^T G m``,
    ``h = A^T W z``, ``G = A^T W A``,

    an identity of the Gaussian quadratic form, exact for any noise model whose precision the
    disentangling applies, AR(1) included, with no approximation of ``W``. It is the
    chi-square of a template composite fitted to the epoch spectra with the orbit fixed.
    Since the disentangling's posterior mean satisfies ``(Lambda + G) d_hat = h``, it differs
    from the exact likelihood of ``d_hat`` under its own posterior only by the prior term and
    a constant (``docs/math.md`` §9.2a).

    The declared light cancels. The label rows scale each template by ``w_i / l0_i``
    (§9.1) and ``A`` multiplies row ``i`` by ``l0_i``, so ``A m`` carries the absolute
    ``w_i t_i``; redeclaring ``l0_i -> c l0_i`` (with the smoothness scales that go with it)
    leaves ``L_data`` unchanged to rounding. The fit measures the light fraction, not a
    ratio to the declaration.

    Components that are not stars (a telluric or nebular row) are conditioned on at their
    posterior mean: their predicted contribution ``A_e d_hat_e`` is subtracted from ``z``
    before ``h`` and ``z^T W z`` are formed: ``L_data(m) = ||z - A_e d_hat_e -
    A_s m||_W^2``. With one operator for all rows this is ``h_s - G_se d_hat_e`` and the
    corresponding correction of ``z^T W z``.

    A synthetic library is already broadened to its own resolving power, so the stellar
    operator carries only the quadrature width ``sigma_q = sqrt(sigma_inst^2 - sigma_lib^2)``
    (:attr:`library_resolving_power`), while the conditioned rows keep the declared
    profile under which their posterior mean was solved. The template's path to the epoch
    pixels also runs through five discrete steps on the model grid, which together smooth it
    by ``(7/12) dv^2`` in variance, so by default the stellar operator carries
    ``sqrt(sigma_q^2 - (7/12) dv^2)`` instead (:attr:`grid_variance_kms2`,
    :attr:`operator_sigma_kms`; derivation in :meth:`albireo.Fit.epoch_statistics`).

    Built by :meth:`albireo.Fit.epoch_statistics`, or by :meth:`from_problem` from a
    :class:`~albireo.forward.Problem` that already carries the operator and data. Passed to
    :func:`match_labels` with ``compare="epochs"``.

    Attributes
    ----------
    h
        ``A_s^T W z``, shape ``(n_star, n_pix)``: row ``i`` is the star ``names[i]``, pixel
        ``p`` the model-grid pixel.
    zwz
        ``z^T W z``, a scalar.
    band
        :class:`~albireo.solver.BlockTridiagonal` storage of ``G = A^T W A`` over all
        ``n_components`` rows of the problem, interleaved as pixel ``q``, component ``j`` at
        ``q n_components + j``. :meth:`matvec` applies its stellar block.
    names
        The stellar components, in row order.
    n_components
        Rows of the problem the band was assembled for, stars first.
    n_good
        Epoch pixels with non-zero weight, the number of terms in ``L_data``.
    grid
        The model grid.
    medium
        The wavelength scale of the epochs.
    light_fractions
        The declared ``l0`` of the stars, which ``A`` carries.
    lsf_sigma_kms
        ``((instrument, widths), ...)``, one entry per epoch group: the declared Gaussian
        widths in km/s reduced for the library's resolving power, ``sigma_q``, before any
        grid compensation (one per LSF anchor where anchors are declared).
    library_resolving_power
        The library resolving power the widths were reduced against, or ``None`` when the
        full declared profile was kept (an intrinsic library).
    conditioned
        Names of the non-stellar rows held at their posterior mean.
    grid_variance_kms2
        The discretisation variance removed from every width, ``(7/12) dv^2`` in km^2/s^2,
        or ``0.0`` when the operator was built without the grid compensation.
    operator_sigma_kms
        ``((instrument, widths), ...)`` in the order of :attr:`lsf_sigma_kms`: the widths
        the stellar operator applies, ``sqrt(sigma_q^2 - grid_variance_kms2)`` floored at
        half a model pixel. Equal to :attr:`lsf_sigma_kms` without the compensation.
    notes
        What the compensation could not do: a width floored at half a model pixel, or a
        model grid coarser than ``sigma_q``. Each was also raised as a ``UserWarning``.
    """

    h: jax.Array
    zwz: jax.Array
    band: Any
    names: tuple
    n_components: int
    n_good: int
    grid: LogGrid
    medium: str
    light_fractions: tuple
    lsf_sigma_kms: tuple = ()
    library_resolving_power: float | None = None
    conditioned: tuple = ()
    grid_variance_kms2: float = 0.0
    operator_sigma_kms: tuple = ()
    notes: tuple = ()

    @property
    def n_star(self) -> int:
        return len(self.names)

    @property
    def n_pix(self) -> int:
        return int(self.grid.n)

    def tree_flatten(self):
        children = (self.h, self.zwz, self.band)
        aux = (
            self.names,
            self.n_components,
            self.n_good,
            self.grid,
            self.medium,
            self.light_fractions,
            self.lsf_sigma_kms,
            self.library_resolving_power,
            self.conditioned,
            self.grid_variance_kms2,
            self.operator_sigma_kms,
            self.notes,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)

    def summary(self) -> str:
        """The operator the statistics were built for: stars, widths, and the compensation."""
        dv = float(self.grid.dv_kms)
        resolving = self.library_resolving_power
        lines = [
            f"EpochStatistics: {', '.join(self.names)} on a {self.n_pix}-pixel model grid "
            f"(dv = {dv:.3f} km/s, {self.medium}), {self.n_good} epoch pixels",
            "  declared light fractions "
            + ", ".join(
                f"{n} {x:.4f}" for n, x in zip(self.names, self.light_fractions, strict=True)
            ),
            "  library "
            + (f"at R = {resolving:g}" if resolving is not None else "intrinsic (full LSF kept)"),
        ]
        if self.grid_variance_kms2 > 0.0:
            lines.append(
                f"  grid compensation: (7/12) dv^2 = {self.grid_variance_kms2:.3f} km^2/s^2 "
                "removed from every width"
            )
        else:
            lines.append("  grid compensation: none")
        operator = self.operator_sigma_kms or self.lsf_sigma_kms
        if self.lsf_sigma_kms:
            lines.append("  stellar operator width (km/s), sigma_q -> applied:")
            for (instrument, sigma_q), (_, applied) in zip(
                self.lsf_sigma_kms, operator, strict=True
            ):
                lines.append(
                    f"    {instrument}: "
                    + ", ".join(
                        f"{q:.3f} -> {a:.3f}" for q, a in zip(sigma_q, applied, strict=True)
                    )
                )
        if self.conditioned:
            lines.append(f"  held at their posterior mean: {', '.join(self.conditioned)}")
        lines.extend(f"  ! {note}" for note in self.notes)
        return "\n".join(lines)

    def matvec(self, rows):
        """``G_ss m`` for stellar rows ``m`` of shape ``(n_star, n_pix)``, same shape out.

        The non-stellar rows are held at zero, so the stellar entries of the full band
        product are exactly the stellar block applied to ``m``.
        """
        rows = jnp.asarray(rows)
        n_pix = rows.shape[1]
        if self.n_components > rows.shape[0]:
            rows = jnp.concatenate(
                [rows, jnp.zeros((self.n_components - rows.shape[0], n_pix), dtype=rows.dtype)]
            )
        stacked = rows.T.reshape(-1)
        n_pad = self.band.num_blocks * self.band.block_size
        stacked = jnp.pad(stacked, (0, n_pad - stacked.shape[0]))
        out = self.band.matvec(stacked)[: self.n_components * n_pix]
        return out.reshape(n_pix, self.n_components).T[: self.n_star]

    def chi2(self, rows):
        """``L_data(m) = z^T W z - 2 m^T h + m^T G m`` for stellar rows ``m`` (traceable)."""
        rows = jnp.asarray(rows)
        return self.zwz - 2.0 * jnp.sum(rows * self.h) + jnp.sum(rows * self.matvec(rows))

    @classmethod
    def from_problem(
        cls,
        problem,
        *,
        names,
        medium: str,
        light_fractions,
        half_bandwidth: int,
        block_size: int | None = None,
        lsf_sigma_kms=(),
        library_resolving_power: float | None = None,
        conditioned=(),
        grid_variance_kms2: float = 0.0,
        operator_sigma_kms=None,
        notes=(),
    ) -> EpochStatistics:
        """Assemble ``h``, ``z^T W z`` and the band of ``G`` from a fixed problem.

        ``problem`` must already carry the operator and the data the comparison is to use:
        the stellar LSF reduced for the library's resolving power and for the grid's own
        smoothing, and the data with any non-stellar contribution subtracted.
        :meth:`albireo.Fit.epoch_statistics` prepares both. ``G`` is assembled by
        :func:`albireo.assembly.band_block_tridiagonal` with a null smoothness prior, at the
        half-bandwidth and block size of the disentangling.

        Parameters
        ----------
        problem
            A :class:`~albireo.forward.Problem` whose first ``len(names)`` rows are the stars.
        names
            The stellar components, in row order.
        medium
            ``"air"`` or ``"vacuum"``, the scale of the epochs.
        light_fractions
            The ``l0`` the problem's operator carries, one per star.
        half_bandwidth, block_size
            As for the disentangling's own band assembly (``MarginalOrbitModel``).
        lsf_sigma_kms, library_resolving_power, conditioned, grid_variance_kms2, notes
            Recorded on the result; see the class attributes.
        operator_sigma_kms
            Recorded on the result; ``None`` records ``lsf_sigma_kms``, which is right for
            an operator built without the grid compensation.
        """
        from albireo.assembly import band_block_tridiagonal
        from albireo.forward import rhs, weighted_data_terms
        from albireo.priors import SmoothnessPrior

        names = tuple(str(n) for n in names)
        nc = int(problem.n_components)
        if not 0 < len(names) <= nc:
            raise ValueError(
                f"names gives {len(names)} stars for a problem with {nc} component rows"
            )
        if medium not in SUPPORTED_MEDIA:
            raise ValueError(f"medium must be one of {SUPPORTED_MEDIA}, got {medium!r}")
        lights = tuple(float(x) for x in np.atleast_1d(np.asarray(light_fractions, float)))
        if len(lights) != len(names):
            raise ValueError(f"light_fractions must give one value per star ({len(names)})")
        null = SmoothnessPrior(tau=jnp.zeros(nc), eta=jnp.zeros(nc))
        band = band_block_tridiagonal(problem, null, int(half_bandwidth), block_size)
        h = jnp.asarray(rhs(problem))[: len(names)]
        zwz, _, n_good = weighted_data_terms(problem)

        def widths(entries):
            return tuple((str(k), tuple(float(x) for x in np.atleast_1d(v))) for k, v in entries)

        quadrature = widths(lsf_sigma_kms)
        return cls(
            h=h,
            zwz=jnp.asarray(zwz, dtype=jnp.float64),
            band=band,
            names=names,
            n_components=nc,
            n_good=int(n_good),
            grid=problem.grid,
            medium=medium,
            light_fractions=lights,
            lsf_sigma_kms=quadrature,
            library_resolving_power=(
                None if library_resolving_power is None else float(library_resolving_power)
            ),
            conditioned=tuple(str(n) for n in conditioned),
            grid_variance_kms2=float(grid_variance_kms2),
            operator_sigma_kms=quadrature
            if operator_sigma_kms is None
            else widths(operator_sigma_kms),
            notes=tuple(str(n) for n in notes),
        )


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------


def _profiled_chi2(model_row, data, weight, basis, gram_inv):
    """Chi-square after the additive nuisance is solved for in closed form.

    The Chebyshev offsets enter linearly, so each trial profiles them out with one small
    solve. Scanning every library node is then affordable, and the scan's chi-square stays
    comparable with the fitted one.
    """
    residual = data - model_row
    coeffs = gram_inv @ (basis.T @ (weight * residual))
    return jnp.sum(weight * (residual - basis @ coeffs) ** 2)


def _scan_component(problem: LabelProblem, index: int, nodes, vsini_trials, v_trials):
    """Profiled chi-square at every library node, for one component.

    Returns ``(chi2, vsini, v)`` per node, each the best over the trial broadenings and
    velocities. The components decouple because the dilution is held at its assumed value
    (the ``FixedDilution`` model). The scan is a coarse start that places the optimizer in
    the right basin.
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


def _default_scan_vsini(lower: float, upper: float) -> list[float]:
    """The warm-start scan's trial rotations: slow, moderate, and a fraction of the bound.

    Late-type binary components mostly rotate at a few km/s; a start at the prior's midpoint
    with a quarter and six tenths of its upper bound (37.5 km/s or more on a 0-150 km/s
    prior) began every D65 benchmark fit there, including components injected at 0.6 to
    10 km/s (``d65_converged_labels.md``). The set is 1 km/s, standing for every rotation the
    model grid cannot resolve (the pixel-integrated kernel is an exact delta below half a
    model pixel, so all such values give the same template); 5 and 10 km/s for slow rotators
    a survey resolution can separate from the instrument profile; and a fifth and a half of
    the upper bound for fast ones. Five values keep the scan within 5/3 of the three-value
    cost. Each is clipped into ``[lower, upper]`` and duplicates are dropped.
    """
    upper = float(upper)
    lower = min(max(float(lower), 0.0), upper)
    trials = (1.0, 5.0, 10.0, 0.2 * upper, 0.5 * upper)
    return sorted({float(np.clip(v, lower, upper)) for v in trials})


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

    Without it the warm start can hand the optimizer a node the prior excludes, and numpyro
    rejects the fit with the opaque "cannot find valid initial parameters". A fixed label is
    narrowed to the nearest node value, since a fixed Teff rarely lands on a grid point.
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
    Scanning each star separately and combining on the shared axis costs ``n_nodes`` per star
    instead of ``n_nodes`` to the power of the number of stars, and reaches the same
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

    ``order=None`` returns a zero-column basis (no additive nuisance): the control case
    against which the nuisance is judged, not a recommended configuration; see
    ``docs/math.md`` §5.1.
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
    compare: str | None = None,
    statistics: EpochStatistics | None = None,
    offset_order: int = 2,
    offset_scale: float = 0.05,
    jitter: bool | None = None,
    exclude_angstrom=(),
    interpolation: str = "auto",
    scan_vsini=None,
    scan_velocities=None,
    top_k: int = 4,
    max_steps: int = 500,
    tol: float = 1e-2,
    restart_rounds: int = 8,
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
        nebular rows must be dropped before calling. In ``"epochs"`` mode they serve only
        the warm-start scan.
    stars
        Mapping of component name to :class:`StarLabels`. Order sets the reference star
        for :class:`RadiusRatio` (the first is ``r = 1``).
    medium
        ``"air"`` or ``"vacuum"``: the scale the data are on, taken from the dataset's own
        declaration. Required: an incorrect choice is an 83 km/s error, so no default is
        safe.
    light_fractions
        The light fractions assumed at disentangling time, one per star: what ``d_hat`` was
        scaled against, not a measurement.
    lsf_sigma_kms
        The declared instrumental Gaussian width, in km/s. Fixed, never fitted: a
        stationary LSF is exactly absorbed by the free component spectra, so disentangling
        cannot identify it (``docs/math.md`` §1.3), and fitting it here would relocate that
        degeneracy rather than resolve it.
    std
        Per-pixel posterior standard deviations, ``Fit.std()``. Defaults to a flat scale,
        with the jitter site carrying all of the weighting; real per-pixel uncertainties
        are preferable.
    mh
        Metallicity spec, shared across components by default (one binary, one composition).
        A mapping of name to spec frees them independently. Defaults to the range the
        libraries cover, or to ``Fixed`` when a library has no metallicity axis.
    dilution
        :class:`RadiusRatio` (default for two or more stars), :class:`ScalarDilution`
        (default for one), or :class:`FixedDilution`.
    compare
        ``"native"`` (the default here) compares the intrinsic model against ``d_hat``
        directly. ``"matched"`` convolves both sides with the declared LSF first; when a
        star's library declares a resolving power
        (:attr:`albireo.SpectralLibrary.resolving_power`) its template is convolved only
        with the quadrature width ``sqrt(sigma_inst^2 - sigma_lib^2)``, since the library
        already carries the rest, and a library at or below the instrument's resolving
        power is refused. ``"epochs"`` compares the template composite with the epoch
        spectra through ``statistics`` (``docs/math.md`` §9.2a) and is the default of
        :meth:`albireo.Fit.match_labels`, which builds them.

        In ``"matched"``, convolving the residuals correlates them over the kernel width
        while the likelihood stays diagonal, so the mis-specification costs a factor of
        ``1 / sum(k^2)`` in chi-square and ``v sin i`` absorbs it. On AI Phe (HARPS,
        R = 115,000) matched drove both components to the ``v sin i`` floor and inflated
        chi-square by 4.26x against native, where the kernel predicts 4.91x. Neither
        ``d_hat`` comparison is correctly specified: converged the same way on simulated
        Gaia RVS binaries, native returned temperatures twice as far from the truth as
        ``"epochs"`` and formal errors too small by a factor of three to thirty-five
        (``docs/math.md`` §9.2a).
    statistics
        The disentangling's :class:`EpochStatistics`, required by ``compare="epochs"`` and
        refused by the other modes, which would ignore them. Their stars, grid, medium,
        declared light fractions and library resolving power must match this call, and
        their operator widths must be the declared instrument width ``lsf_sigma_kms`` (the
        widest, where there are several) reduced for that resolving power and, when
        :attr:`EpochStatistics.grid_variance_kms2` is set, for the grid's own smoothing. An
        operator narrowed by any other route, such as a resolving power that is not the
        library's, is refused.
    offset_order
        Degree of the additive Chebyshev nuisance per component. The ``m = 0`` term is the
        unconstrained zero point of ``docs/math.md`` §5.1; the default of 2 also absorbs
        the slope and curvature that the low-``k`` exchange modes leave behind.
    jitter
        Whether each component carries a bounded noise-scale site. Defaults to ``True``
        for the ``d_hat`` comparisons and ``False`` for ``"epochs"``, whose chi-square is
        exact in data units and which refuses one.
    exclude_angstrom
        Wavelength ranges to drop: nebular cores, detector gaps, and any region the
        disentangling could not model. Not available in ``"epochs"`` mode, where the
        comparison runs over epoch pixels: mask them in the epochs before disentangling.
    scan_vsini, scan_velocities
        Trial values for the warm-start scan. ``scan_vsini`` defaults to 1, 5 and 10 km/s
        and a fifth and a half of the ``vsini`` prior's upper bound, clipped to the priors,
        so that slow rotation is tried (reasoning in ``_default_scan_vsini``);
        ``scan_velocities`` to five velocities spanning the ``v_kms`` prior.
    top_k
        How many of the scan's best starting points to optimise from. More than one, since
        a label surface with two basins is common; the report states when the runners-up
        were close.
    max_steps
        L-BFGS steps per start for ``"native"`` and ``"matched"``, and the iteration cap of
        each Levenberg-Marquardt run for ``"epochs"``, which converges in a median of four.
        ``tol`` is an absolute gradient-norm threshold on a potential whose scale grows
        with the pixel count, so it is unreachable on real data and ``converged`` reads
        ``False`` however good an L-BFGS fit is (the same caveat as :func:`albireo.run_map`).
        Convergence is judged from the chi-square against its nulls, which
        :meth:`LabelMatch.summary` prints. ``"epochs"`` ignores ``tol`` and stops when the
        undamped Gauss-Newton step predicts a decrease below 0.001 in chi-square.
    restart_rounds
        ``"epochs"`` only: the most restart rounds run after the warm-started fits. Each
        round scans every component's ``v sin i``, then the faintest component's
        ``(Teff, log g)`` library nodes, and, when that component's light has collapsed,
        a coarse joint grid in its dilution, rotation, temperature and gravity; it refits
        from the best distinct points. The rounds stop once one gains less than 1 in
        chi-square. ``0`` disables them.

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
    if compare is None:
        if statistics is not None:
            raise ValueError(
                "statistics were passed but compare is unset, and the default comparison "
                "('native', against d_hat) would ignore them. Pass compare='epochs' to "
                "compare with the epoch spectra through them (Fit.match_labels does this by "
                "default), or drop statistics."
            )
        compare = "native"
    if compare not in ("epochs", "matched", "native"):
        raise ValueError(f"compare must be 'epochs', 'matched' or 'native', got {compare!r}")
    epochs = compare == "epochs"
    if statistics is not None and not epochs:
        raise ValueError(
            f"statistics are used only by compare='epochs'; compare={compare!r} compares "
            "against d_hat and would ignore them. Pass compare='epochs', or drop statistics."
        )
    if epochs:
        if not isinstance(statistics, EpochStatistics):
            raise ValueError(
                "compare='epochs' needs the disentangling's epoch statistics: pass "
                "statistics=fit.epoch_statistics(resolving_power=...), or call "
                "Fit.match_labels, which builds them from the libraries' resolving power"
            )
        if jitter:
            raise ValueError(
                "compare='epochs' has no jitter sites: its chi-square is exact in data units "
                "under the disentangling's own noise model. Leave jitter unset."
            )
        if len(tuple(exclude_angstrom)) > 0:
            raise ValueError(
                "exclude_angstrom drops component-spectrum pixels, and compare='epochs' "
                "compares epoch pixels. Mask the range in the epochs (ivar = 0) before "
                "disentangling instead."
            )
        if int(restart_rounds) < 0:
            raise ValueError("restart_rounds must be >= 0")
    if jitter is None:
        jitter = not epochs

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
    resolving = {name: projected[i].resolving_power for i, name in enumerate(names)}
    if epochs:
        _check_statistics(statistics, names, grid, medium, ell0, resolving, lsf_sigma_kms)

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
    model_kernels = tuple(jnp.asarray(lsf_kernel) for _ in names)
    if matched:
        # The data side is d_hat, a partial deconvolution to intrinsic resolution, so it
        # takes the whole instrument profile; a template drawn from a library at its own
        # resolving power takes only the width the library does not already carry.
        model_kernels = tuple(
            jnp.asarray(
                gaussian_kernel(
                    float(_quadrature_width(lsf_sigma_kms, resolving[name], f"star {name!r}")[0])
                    / grid.dv_kms
                )
            )
            for name in names
        )
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
        model_lsf_kernels=model_kernels,
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
        lowers = [
            _spec_bounds(specs[f"vsini_{name}"])[0]
            if _spec_bounds(specs[f"vsini_{name}"]) is not None
            else 0.0
            for name in names
        ]
        scan_vsini = _default_scan_vsini(min(lowers), vsini_upper)
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

    assumptions = {
        "light_fractions": ell0.tolist(),
        "lsf_sigma_kms": float(lsf_sigma_kms),
        "medium": medium,
        "compare": compare,
        "library_resolving_power": dict(resolving),
        "macro_kms": {name: declared[name].macro_kms for name in names},
        "offset_order": offset_order,
        "dilution": dilution_kind,
        **(
            {
                "library_filled_nodes": {
                    name: _filled_words(projected[i])
                    for i, name in enumerate(names)
                    if projected[i].meta.get("filled_nodes")
                }
            }
            if any(p.meta.get("filled_nodes") for p in projected)
            else {}
        ),
        "vmicro": {
            name: projected[i].meta.get("vmicro", "unrecorded") for i, name in enumerate(names)
        },
        "grids": {name: projected[i].meta.get("grid", "unnamed") for i, name in enumerate(names)},
    }

    if epochs:
        assumptions["epoch_operator_lsf_sigma_kms"] = {
            instrument: list(widths)
            for instrument, widths in (statistics.operator_sigma_kms or statistics.lsf_sigma_kms)
        }
        assumptions["epoch_quadrature_lsf_sigma_kms"] = {
            instrument: list(widths) for instrument, widths in statistics.lsf_sigma_kms
        }
        assumptions["epoch_grid_variance_kms2"] = float(statistics.grid_variance_kms2)
        if statistics.conditioned:
            assumptions["conditioned_on_posterior_mean"] = list(statistics.conditioned)
        return _fit_epochs(
            problem,
            specs,
            config,
            statistics,
            libraries=tuple(projected),
            node_tables=node_tables,
            scans=scans,
            candidates=candidates,
            wave=np.asarray(grid.wave),
            assumptions=assumptions,
            max_steps=int(max_steps),
            restart_rounds=int(restart_rounds),
            progress=progress,
        )

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
        assumptions=assumptions,
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

    The row labels are built from the sites' shapes, not their names: the Chebyshev offsets
    are vectors, so a covariance row is not a site. Zipping sorted site names against the
    diagonal shifts every entry after the first vector site, which presents as an
    implausibly small uncertainty rather than as an error.
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
# The epoch comparison: a bounded Levenberg-Marquardt with restarts
# ---------------------------------------------------------------------------

# Stop when the undamped Gauss-Newton step predicts less than this decrease in chi-square
# with the active set unchanged (d65_converged_labels.md).
_LM_PRED_TOL = 1e-3
# A restart round that lowers the objective by less than this ends the rounds.
_RESTART_GAIN = 1.0
# Batch size of the restart scans: one compiled shape for every scan.
_SCAN_CHUNK = 32
# Rotation trials of the per-component scan; clipped to each prior, which D65 ran at 0-150.
_VSINI_TRIALS = (
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0,
    60.0, 80.0, 100.0, 125.0, 150.0, 200.0, 250.0, 300.0,
)  # fmt: skip
# The joint restart's grid in the faint component's relative radius and rotation.
_JOINT_RATIOS = (0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
_JOINT_VSINI = (2.0, 8.0, 20.0, 40.0, 80.0)
# A light fraction below this counts as collapsed.
_COLLAPSED_LIGHT = 0.01


def _close(a, b, rel: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= rel * max(abs(float(a)), abs(float(b)), 1.0)


def _check_statistics(statistics, names, grid, medium, ell0, resolving, lsf_sigma_kms) -> None:
    """Refuse statistics built for different stars, grid, medium, light or operator.

    The operator is checked through its recorded widths: the grid variance must be none or
    ``(7/12) dv^2`` of this grid, each applied width must follow from its quadrature width
    and that variance, and the widest quadrature width with the library's own width restored
    must be the declared instrument width. The last test refuses an operator narrowed by
    declaring a resolving power that is not the library's.
    """
    if tuple(statistics.names) != tuple(names):
        raise ValueError(
            f"the statistics were built for the stars {list(statistics.names)} and this call "
            f"declares {list(names)}; the rows must be the same stars in the same order"
        )
    other = statistics.grid
    if (
        int(other.n) != int(grid.n)
        or not _close(other.dx, grid.dx, 1e-12)
        or not _close(other.x0, grid.x0, 1e-12)
    ):
        raise ValueError(
            f"the statistics were built on a {other.n}-pixel grid starting at "
            f"{float(np.exp(other.x0)):.4f} A and this call's grid has {grid.n} pixels starting "
            f"at {float(np.exp(grid.x0)):.4f} A; build both from the same fit"
        )
    if statistics.medium != medium:
        raise ValueError(
            f"the statistics were built from epochs on the {statistics.medium!r} scale and this "
            f"call declares {medium!r}"
        )
    declared = np.asarray(statistics.light_fractions, dtype=np.float64)
    if declared.shape != ell0.shape or not np.allclose(declared, ell0, rtol=1e-9, atol=0.0):
        raise ValueError(
            f"the statistics' operator carries the light fractions {declared.tolist()} and this "
            f"call declares {ell0.tolist()}. The label rows are scaled by w / l0 and the "
            "operator by l0, so the two must be the same declaration for l0 to cancel."
        )
    for name, value in resolving.items():
        built = statistics.library_resolving_power
        same = (value is None and built is None) or (
            value is not None and built is not None and _close(value, built, 1e-9)
        )
        if not same:
            raise ValueError(
                f"star {name!r}'s library is at R = {value} and the statistics' operator was "
                f"reduced for R = {built}. The operator must carry exactly the width the library "
                "lacks, or every template is broadened wrongly and v sin i absorbs it. Rebuild "
                "with fit.epoch_statistics(resolving_power=library.resolving_power)."
            )

    rebuild = (
        "Rebuild with fit.epoch_statistics(resolving_power=library.resolving_power), which "
        "applies the grid compensation itself (grid_compensation=False leaves it out)."
    )
    dv = float(grid.dv_kms)
    variance = float(statistics.grid_variance_kms2)
    if variance != 0.0 and not _close(variance, _grid_variance(dv), 1e-9):
        raise ValueError(
            f"the statistics remove a grid variance of {variance:.6g} km^2/s^2, and (7/12) dv^2 "
            f"for this call's grid (dv = {dv:.6g} km/s) is {_grid_variance(dv):.6g}. {rebuild}"
        )
    quadrature = tuple(statistics.lsf_sigma_kms)
    operator = tuple(statistics.operator_sigma_kms) or quadrature
    if len(operator) != len(quadrature):
        raise ValueError(
            f"the statistics record {len(quadrature)} quadrature width entries and "
            f"{len(operator)} operator width entries. {rebuild}"
        )
    for (instrument, q), (other, applied) in zip(quadrature, operator, strict=True):
        q, applied = np.asarray(q, dtype=np.float64), np.asarray(applied, dtype=np.float64)
        expected = q if variance == 0.0 else _grid_compensated_widths(q, dv)[0]
        if (
            other != instrument
            or applied.shape != q.shape
            or not np.allclose(applied, expected, rtol=1e-9, atol=0.0)
        ):
            raise ValueError(
                f"the statistics' operator widths {applied.tolist()} km/s for {other!r} are not "
                f"the quadrature widths {q.tolist()} km/s of {instrument!r} reduced by the "
                f"recorded grid variance of {variance:.6g} km^2/s^2. {rebuild}"
            )
    if quadrature:
        built = statistics.library_resolving_power
        have = 0.0 if built is None else _resolution_sigma_kms(built)
        widest_q = max(float(np.max(q)) for _, q in quadrature)
        restored = float(np.sqrt(widest_q**2 + have**2))
        if not _close(restored, float(lsf_sigma_kms), 1e-6):
            raise ValueError(
                f"the statistics' widest quadrature width {widest_q:.6g} km/s with the "
                f"library's own {have:.6g} km/s restored is an instrument width of "
                f"{restored:.6g} km/s, and this call declares lsf_sigma_kms = "
                f"{float(lsf_sigma_kms):.6g} km/s. An operator narrowed by declaring a "
                "resolving power that is not the library's cannot be told apart from a wrong "
                f"instrument width, so both are refused. {rebuild}"
            )


@dataclass(frozen=True)
class _Layout:
    """The flat parameter vector of the epoch fit: every free site in a fixed order."""

    names: tuple
    keys: tuple
    sizes: tuple
    starts: tuple
    label_axes: tuple
    shared_mh: bool
    dilution: str
    offsets: bool
    hull_guard: bool
    fixed: tuple  # ((key, value), ...)

    @property
    def n(self) -> int:
        return int(sum(self.sizes))

    def slot(self, key: str) -> slice:
        i = self.keys.index(key)
        return slice(self.starts[i], self.starts[i] + self.sizes[i])

    def index(self, key: str) -> int | None:
        """The flat position of a scalar free site, or ``None`` when it is not free."""
        return self.starts[self.keys.index(key)] if key in self.keys else None

    def flat_labels(self) -> tuple[str, ...]:
        out = []
        for key, size in zip(self.keys, self.sizes, strict=True):
            out.extend([key] if size == 1 else [f"{key}[{j}]" for j in range(size)])
        return tuple(out)

    def to_flat(self, values: dict) -> np.ndarray:
        out = np.empty(self.n)
        for key in self.keys:
            out[self.slot(key)] = np.atleast_1d(np.asarray(values[key], dtype=np.float64))
        return out

    def to_params(self, phi) -> dict:
        phi = np.asarray(phi, dtype=np.float64)
        out = {}
        for key, size in zip(self.keys, self.sizes, strict=True):
            part = phi[self.slot(key)]
            out[key] = float(part[0]) if size == 1 else part.copy()
        return out

    def site(self, phi, key: str):
        if key in self.keys:
            i = self.keys.index(key)
            start, size = self.starts[i], self.sizes[i]
            return phi[start] if size == 1 else phi[start : start + size]
        return jnp.asarray(dict(self.fixed)[key], dtype=jnp.float64)

    def label_key(self, label: str, name: str) -> str:
        if label == "mh" and self.shared_mh:
            return "mh"
        if label == "v_kms":
            return f"v_{name}"
        return f"{label}_{name}"


def _epoch_layout(specs, config, problem):
    """``(layout, lo, hi, prior_terms)`` for the epoch fit.

    Uniform priors become bounds and contribute nothing inside them. A Normal prior
    contributes its quadratic ``((x - mu) / s)^2``, and any other prior ``-2 log p``, with
    bounds from its support where it has them.
    """
    names = tuple(config["names"])
    order = ["mh"] if config["shared_mh"] else []
    for name in names:
        order += [f"teff_{name}", f"logg_{name}"]
        if not config["shared_mh"]:
            order.append(f"mh_{name}")
        order += [f"vsini_{name}", f"v_{name}"]
    if problem.dilution == "radius_ratio":
        order += [f"ratio_{name}" for name in names[1:]]
    elif problem.dilution == "scalar":
        order += [f"scale_{name}" for name in names]
    if config["offsets"]:
        order += [f"offset_{name}" for name in names]

    keys, sizes, starts, fixed, lo, hi, terms = [], [], [], [], [], [], []
    position = 0
    for key in order:
        spec = specs[key]
        distribution = spec.distribution()
        if distribution is None:
            fixed.append((key, float(np.asarray(spec.start(), dtype=np.float64))))
            continue
        size = int(np.asarray(spec.start()).size)
        if isinstance(distribution, dist.Uniform):
            low, high = distribution.low, distribution.high
        else:
            support = distribution.support
            low = getattr(support, "lower_bound", -np.inf)
            high = getattr(support, "upper_bound", np.inf)
            terms.append((position, size, distribution))
        lo.append(np.broadcast_to(np.asarray(low, dtype=np.float64), (size,)))
        hi.append(np.broadcast_to(np.asarray(high, dtype=np.float64), (size,)))
        keys.append(key)
        sizes.append(size)
        starts.append(position)
        position += size
    layout = _Layout(
        names=names,
        keys=tuple(keys),
        sizes=tuple(sizes),
        starts=tuple(starts),
        label_axes=tuple(tuple(a) for a in problem.label_axes),
        shared_mh=bool(config["shared_mh"]),
        dilution=str(problem.dilution),
        offsets=bool(config["offsets"]),
        hull_guard=bool(config["hull_guard"]),
        fixed=tuple(fixed),
    )
    lo = np.concatenate(lo) if lo else np.zeros(0)
    hi = np.concatenate(hi) if hi else np.zeros(0)
    return layout, lo, hi, tuple(terms)


def _layout_labels(layout: _Layout, phi, problem):
    labels = []
    for i, name in enumerate(layout.names):
        axes = problem.label_axes[i]
        values = {lab: layout.site(phi, layout.label_key(lab, name)) for lab in axes}
        labels.append(jnp.stack([values[axis] for axis in axes]))
    return labels


def _layout_amplitudes(layout: _Layout, phi, problem):
    if layout.dilution == "radius_ratio":
        ratios = [jnp.asarray(1.0)] + [layout.site(phi, f"ratio_{n}") for n in layout.names[1:]]
        return jnp.stack(ratios) ** 2
    if layout.dilution == "scalar":
        return jnp.stack([layout.site(phi, f"scale_{n}") for n in layout.names])
    return problem.ell0


def _layout_rows(layout: _Layout, phi, problem):
    """The stacked label rows of ``_model_rows`` at a flat parameter vector."""
    vsini = [layout.site(phi, f"vsini_{n}") for n in layout.names]
    v_kms = [layout.site(phi, f"v_{n}") for n in layout.names]
    offsets = (
        jnp.stack([layout.site(phi, f"offset_{n}") for n in layout.names])
        if layout.offsets
        else jnp.zeros((len(layout.names), 0))
    )
    return _model_rows(
        problem,
        _layout_labels(layout, phi, problem),
        vsini,
        v_kms,
        _layout_amplitudes(layout, phi, problem),
        offsets,
    )


def _layout_light(layout: _Layout, phi, problem):
    """Band-median light fraction per component, as :attr:`LabelMatch.flux_ratio`."""
    if layout.dilution == "fixed":
        return problem.ell0
    if layout.dilution == "scalar":
        return _layout_amplitudes(layout, phi, problem)
    labels = _layout_labels(layout, phi, problem)
    continua = [problem.interpolators[i](labels[i])[1] for i in range(len(layout.names))]
    rows = _light_fractions(continua, _layout_amplitudes(layout, phi, problem))
    return jnp.median(rows, axis=1)


class _EpochObjective:
    """``F(phi) = L_data(m(phi)) - 2 log prior``, its gradient and Gauss-Newton matrix.

    In the constrained space and in chi-square units, with no Jacobian term: bounded
    parameters are held by the optimiser's active set rather than by a transform. With
    ``J = dm/dphi`` from forward-mode differentiation of the rows, the gradient of
    ``L_data`` is ``-2 J^T (h - G m)`` and its Gauss-Newton matrix ``2 J^T G J``, exact up
    to the neglected curvature of ``m(phi)`` because ``L_data`` is quadratic in ``m``. The
    prior and the hull guard add their exact gradient and Hessian by autodiff; each prior is
    evaluated on its own slice of ``phi``, so no masked branch can put ``inf * 0`` into the
    Hessian.
    """

    def __init__(self, layout: _Layout, terms):
        def prior(phi, problem):
            total = jnp.asarray(0.0)
            for start, size, distribution in terms:
                value = phi[start] if size == 1 else phi[start : start + size]
                if isinstance(distribution, dist.Normal):
                    # the quadratic itself, without the normalising constant, so that the
                    # objective reads as L_data plus the offsets' chi-square
                    z = (value - distribution.loc) / distribution.scale
                    total = total + jnp.sum(z**2)
                else:
                    total = total - 2.0 * jnp.sum(distribution.log_prob(value))
            if layout.hull_guard:
                labels = _layout_labels(layout, phi, problem)
                margin = jnp.stack(
                    [
                        problem.interpolators[i].hull_margin(labels[i])
                        for i in range(len(layout.names))
                    ]
                )
                total = total + 2.0 * 1e3 * jnp.sum(jax.nn.softplus(-50.0 * margin) ** 2)
            return total

        def rows(phi, problem):
            return _layout_rows(layout, phi, problem)

        def value(phi, problem, stats):
            loss = stats.chi2(rows(phi, problem))
            return loss + prior(phi, problem), loss

        def everything(phi, problem, stats):
            m = rows(phi, problem)
            jac = jax.jacfwd(rows)(phi, problem)  # (n_star, n_pix, P)
            gm = stats.matvec(m)
            loss = stats.zwz - 2.0 * jnp.sum(m * stats.h) + jnp.sum(m * gm)
            gjac = jax.vmap(stats.matvec, in_axes=-1, out_axes=-1)(jac)
            grad = -2.0 * jnp.einsum("spk,sp->k", jac, stats.h - gm)
            gn = 2.0 * jnp.einsum("spk,spl->kl", jac, gjac)
            hess = jax.hessian(prior)(phi, problem)
            return (
                loss + prior(phi, problem),
                loss,
                grad + jax.grad(prior)(phi, problem),
                0.5 * (gn + gn.T) + hess,
            )

        def light(phi, problem):
            return _layout_light(layout, phi, problem)

        self.layout = layout
        self._value = jax.jit(value)
        self._everything = jax.jit(everything)
        self._batch = jax.jit(jax.vmap(value, in_axes=(0, None, None)))
        self._rows = jax.jit(rows)
        self._light = jax.jit(light)
        self._light_jac = jax.jit(jax.jacfwd(light))

    def value(self, phi, problem, stats):
        f, loss = self._value(jnp.asarray(phi), problem, stats)
        return float(f), float(loss)

    def everything(self, phi, problem, stats):
        f, loss, g, h = self._everything(jnp.asarray(phi), problem, stats)
        return float(f), float(loss), np.asarray(g, np.float64), np.asarray(h, np.float64)

    def batch(self, phis, problem, stats) -> np.ndarray:
        phis = np.asarray(phis, dtype=np.float64)
        out = []
        for k in range(0, phis.shape[0], _SCAN_CHUNK):
            block = phis[k : k + _SCAN_CHUNK]
            n = block.shape[0]
            if n < _SCAN_CHUNK:  # pad to one compiled shape
                block = np.concatenate([block, np.repeat(block[-1:], _SCAN_CHUNK - n, axis=0)])
            f, _ = self._batch(jnp.asarray(block), problem, stats)
            out.append(np.asarray(f)[:n])
        return np.concatenate(out) if out else np.zeros(0)

    def rows(self, phi, problem) -> np.ndarray:
        return np.asarray(self._rows(jnp.asarray(phi), problem))

    def light(self, phi, problem) -> np.ndarray:
        return np.asarray(self._light(jnp.asarray(phi), problem), dtype=np.float64)

    def light_jac(self, phi, problem) -> np.ndarray:
        return np.asarray(self._light_jac(jnp.asarray(phi), problem), dtype=np.float64)


@dataclass
class _LMRun:
    phi: np.ndarray
    objective: float
    chi2: float
    grad: np.ndarray
    gn: np.ndarray
    status: str
    iterations: int
    evaluations: int
    kind: str


def _bound_tol(lo, hi):
    width = hi - lo
    finite = np.isfinite(width)
    return np.where(finite, 1e-10 * np.where(finite, width, 0.0), 0.0)


def _damped_step(gn, grad, free, lam, dscale):
    """Solve ``(H_F + lam D_F) s_F = -g_F`` with Marquardt's diagonal scaling ``D``.

    Scaling by the running maximum of the Gauss-Newton diagonal makes the step invariant
    to a rescaling of the parameters: over its prior, [M/H] moves the chi-square by 1e4
    while a faint component's log g moves it by a few tens.
    """
    idx = np.flatnonzero(free)
    step = np.zeros_like(grad)
    if idx.size == 0:
        return step
    inv = 1.0 / np.sqrt(np.maximum(dscale[idx], 1e-300))
    a = inv[:, None] * gn[np.ix_(idx, idx)] * inv[None, :] + lam * np.eye(idx.size)
    rhs = -inv * grad[idx]
    try:
        y = np.linalg.solve(a, rhs)
        if not np.all(np.isfinite(y)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(a, rhs, rcond=None)[0]
    step[idx] = inv * y
    return step


def _levenberg_marquardt(
    objective, problem, stats, phi0, lo, hi, *, kind, max_iter, max_eval, progress=None
) -> _LMRun:
    """Bounded, damped Gauss-Newton in the constrained space (``d65_converged_labels.md``).

    A parameter on a bound whose gradient points outward is held (the active set), the
    trial step is projected onto the bounds, a step is accepted when the actual decrease
    exceeds 1e-4 of the quadratic model's prediction, and the damping follows Nielsen's
    gain-ratio update. The run stops when the undamped step predicts a decrease below
    ``_LM_PRED_TOL`` with the active set unchanged ("converged"), when no damping yields a
    decrease ("stalled"), or on the iteration or evaluation budget.
    """
    tol = _bound_tol(lo, hi)
    phi = np.clip(np.asarray(phi0, dtype=np.float64), lo, hi)
    f, loss, grad, gn = objective.everything(phi, problem, stats)
    evaluations, iterations = 1, 0
    if not (np.isfinite(f) and np.all(np.isfinite(grad)) and np.all(np.isfinite(gn))):
        return _LMRun(phi, float("inf"), float("inf"), grad, gn, "nonfinite_start", 0, 1, kind)
    dscale = np.maximum(np.diag(gn).copy(), 0.0)
    lam, nu = 1e-3, 2.0
    previous = None
    status = "iteration_budget"
    while iterations < max_iter:
        active = ((phi <= lo + tol) & (grad > 0)) | ((phi >= hi - tol) & (grad < 0))
        dscale = np.maximum(dscale, np.diag(gn))
        free = ~active
        s_gn = np.clip(phi + _damped_step(gn, grad, free, 0.0, dscale), lo, hi) - phi
        predicted = float(max(-(grad @ s_gn + 0.5 * s_gn @ gn @ s_gn), 0.0))
        if predicted < _LM_PRED_TOL and previous is not None and np.array_equal(active, previous):
            status = "converged"
            break
        previous = active
        accepted = False
        while evaluations < max_eval:
            trial = np.clip(phi + _damped_step(gn, grad, free, lam, dscale), lo, hi)
            step = trial - phi
            gain = -(grad @ step + 0.5 * step @ gn @ step)
            if gain > 0 and np.any(step != 0):
                f_try, _ = objective.value(trial, problem, stats)
                evaluations += 1
                if np.isfinite(f_try) and (f - f_try) / gain > 1e-4:
                    rho = (f - f_try) / gain
                    phi = trial
                    lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                    nu = 2.0
                    accepted = True
                    break
            lam *= nu
            nu *= 2.0
            if lam > 1e16:
                break
        if not accepted:
            status = "stalled" if evaluations < max_eval else "evaluation_budget"
            break
        f, loss, grad, gn = objective.everything(phi, problem, stats)
        evaluations += 1
        iterations += 1
        if progress is not None:
            progress(iterations, 0.5 * f, float(np.linalg.norm(grad[free])), phi)
    return _LMRun(phi, f, loss, grad, gn, status, iterations, evaluations, kind)


def _plateau_mask(layout: _Layout, phi, dv_kms: float) -> np.ndarray:
    """``v sin i`` sites below half a model pixel, where the rotation kernel is a delta.

    :func:`albireo.operators.rotational_kernel_traced` integrates the profile over pixels, so
    below half a pixel it is exactly one tap: the objective is exactly flat in ``v sin i``
    there, its Gauss-Newton row is zero, and the label is unresolved rather than measured.
    """
    mask = np.zeros(layout.n, dtype=bool)
    for name in layout.names:
        j = layout.index(f"vsini_{name}")
        if j is not None:
            mask[j] = float(phi[j]) <= 0.5 * float(dv_kms)
    return mask


def _formal_covariance(run: _LMRun, lo, hi, flat) -> tuple[np.ndarray, np.ndarray]:
    """``(cov, excluded)``: ``2 H^-1`` over the parameters neither at a bound nor flat.

    Excluded parameters are conditioned on and their rows are NaN. A matrix that cannot be
    inverted leaves NaN rather than a pseudo-inverse, which reports a zero variance for a
    parameter on the rotation plateau.
    """
    tol = _bound_tol(lo, hi)
    excluded = (run.phi <= lo + tol) | (run.phi >= hi - tol) | (np.diag(run.gn) <= 0.0)
    excluded = excluded | np.asarray(flat, dtype=bool)
    free = np.flatnonzero(~excluded)
    cov = np.full((run.phi.size, run.phi.size), np.nan)
    if free.size:
        try:
            inv = 2.0 * np.linalg.inv(run.gn[np.ix_(free, free)])
        except np.linalg.LinAlgError:
            return cov, excluded
        good = np.isfinite(np.diag(inv)) & (np.diag(inv) > 0.0)
        inv[~good, :] = np.nan
        inv[:, ~good] = np.nan
        cov[np.ix_(free, free)] = inv
    return cov, excluded


def _distinct(values, current, kept, near) -> bool:
    return not near(values, current) and not any(near(values, other) for other in kept)


def _vsini_starts(objective, problem, stats, layout, lo, hi, phi, index, n_best=2):
    """Starts along one component's ``v sin i``, every other parameter held.

    The objective is exactly flat below half a model pixel, so a gradient method that starts
    or lands there cannot leave, even when a lower minimum lies at a few km/s (on the first
    D65 product all four warm-started fits stopped there, 1.67 in chi-square above the
    optimum at 11.8 km/s).
    """
    name = layout.names[index]
    j = layout.index(f"vsini_{name}")
    if j is None:
        return []
    low = float(lo[j]) if np.isfinite(lo[j]) else 0.0
    high = float(hi[j]) if np.isfinite(hi[j]) else _VSINI_TRIALS[-1]
    trials = [v for v in _VSINI_TRIALS if low <= v <= high]
    values = np.unique(np.asarray([*trials, low, high], dtype=np.float64))
    if values.size < 8:
        values = np.linspace(low, high, 12)
    batch = np.repeat(phi[None, :], values.size, axis=0)
    batch[:, j] = values
    scores = objective.batch(batch, problem, stats)

    def near(a, b):
        return abs(a - b) <= max(2.5, 0.25 * max(abs(a), abs(b)))

    kept = []
    for k in np.argsort(scores):
        if np.isfinite(scores[k]) and _distinct(values[k], float(phi[j]), kept, near):
            kept.append(float(values[k]))
        if len(kept) == n_best:
            break
    starts = []
    for value in kept:
        start = phi.copy()
        start[j] = value
        starts.append((f"vsini {name}", start))
    return starts


def _node_starts(objective, problem, stats, layout, lo, hi, phi, index, node_tables, n_best=3):
    """Starts at the faint component's best ``(Teff, log g)`` library nodes, others held."""
    name = layout.names[index]
    axes = layout.label_axes[index]
    free = [
        (label, layout.index(f"{label}_{name}"))
        for label in ("teff", "logg")
        if label in axes and layout.index(f"{label}_{name}") is not None
    ]
    if not free:
        return []
    table = np.asarray(node_tables[index])
    columns = [axes.index(label) for label, _ in free]
    points = np.unique(table[:, columns], axis=0)
    keep = np.ones(points.shape[0], dtype=bool)
    for c, (_, j) in enumerate(free):
        keep &= (points[:, c] >= lo[j]) & (points[:, c] <= hi[j])
    points = points[keep]
    if points.shape[0] == 0:
        return []
    steps = []
    for c in range(len(free)):
        unique = np.unique(points[:, c])
        steps.append(float(np.min(np.diff(unique))) if unique.size > 1 else 0.0)
    batch = np.repeat(phi[None, :], points.shape[0], axis=0)
    for c, (_, j) in enumerate(free):
        batch[:, j] = points[:, c]
    scores = objective.batch(batch, problem, stats)
    current = np.asarray([phi[j] for _, j in free])

    def near(a, b):
        return all(abs(a[c] - b[c]) <= steps[c] + 1e-9 for c in range(len(free)))

    kept = []
    for k in np.argsort(scores):
        if np.isfinite(scores[k]) and _distinct(points[k], current, kept, near):
            kept.append(points[k])
        if len(kept) == n_best:
            break
    starts = []
    for point in kept:
        start = phi.copy()
        for c, (_, j) in enumerate(free):
            start[j] = point[c]
        starts.append((f"nodes {name}", start))
    return starts


def _dilution_trials(layout: _Layout, lo, hi, phi, index):
    """``[(multiplier label, start vector)]`` moving one component's share of the light.

    For a radius-ratio dilution the faint component's relative radius takes the values of
    ``_JOINT_RATIOS``; when the faint component is the reference star, every other ratio is
    divided by them instead. For a scalar dilution its factor takes their squares.
    """
    name = layout.names[index]
    out = []
    for x in _JOINT_RATIOS:
        start = phi.copy()
        if layout.dilution == "radius_ratio":
            if index > 0:
                j = layout.index(f"ratio_{name}")
                if j is None:
                    return [(None, phi.copy())]
                start[j] = x
            else:
                moved = False
                for other in layout.names[1:]:
                    j = layout.index(f"ratio_{other}")
                    if j is not None:
                        start[j] = phi[j] / x
                        moved = True
                if not moved:
                    return [(None, phi.copy())]
        elif layout.dilution == "scalar":
            j = layout.index(f"scale_{name}")
            if j is None:
                return [(None, phi.copy())]
            start[j] = x * x
        else:
            return [(None, phi.copy())]
        out.append((x, np.clip(start, lo, hi)))
    return out


def _joint_starts(objective, problem, stats, layout, lo, hi, phi, index, node_tables, n_best=3):
    """Starts from a coarse joint grid in one component's light, rotation and labels.

    A component whose light has collapsed has no labels, so no single-direction move lowers
    the objective: on the one D65 product where this happened every such restart returned
    to the collapsed basin, while a joint grid of six radius ratios, five rotations and
    every other temperature node at three gravities escaped it under one prior
    configuration (``d65_converged_labels.md``).
    """
    name = layout.names[index]
    axes = layout.label_axes[index]
    j_v = layout.index(f"vsini_{name}")
    j_t = layout.index(f"teff_{name}") if "teff" in axes else None
    j_g = layout.index(f"logg_{name}") if "logg" in axes else None
    table = np.asarray(node_tables[index])
    teffs = [None]
    if j_t is not None:
        values = np.unique(table[:, axes.index("teff")])
        values = values[(values >= lo[j_t]) & (values <= hi[j_t])]
        teffs = list(values[::2] if values.size > 6 else values) or [None]
    loggs = [None]
    if j_g is not None:
        values = np.unique(table[:, axes.index("logg")])
        values = values[(values >= lo[j_g]) & (values <= hi[j_g])]
        if values.size:
            loggs = sorted({float(values[0]), float(values[values.size // 2]), float(values[-1])})
    rotations = [None]
    if j_v is not None:
        rotations = sorted({float(np.clip(v, lo[j_v], hi[j_v])) for v in _JOINT_VSINI})
    grid, labels = [], []
    for x, base in _dilution_trials(layout, lo, hi, phi, index):
        for v in rotations:
            for t in teffs:
                for g in loggs:
                    start = base.copy()
                    if v is not None:
                        start[j_v] = v
                    if t is not None:
                        start[j_t] = t
                    if g is not None:
                        start[j_g] = g
                    grid.append(np.clip(start, lo, hi))
                    labels.append((x, v, t, g))
    if not grid:
        return []
    scores = objective.batch(np.asarray(grid), problem, stats)

    def near(a, b):
        same_x = a[0] is None or b[0] is None or abs(np.log(a[0] / b[0])) < 0.3
        same_v = a[1] is None or b[1] is None or abs(a[1] - b[1]) < 10.0
        same_t = a[2] is None or b[2] is None or abs(a[2] - b[2]) <= 500.0
        return same_x and same_v and same_t

    kept = []
    for k in np.argsort(scores):
        if np.isfinite(scores[k]) and not any(near(labels[k], labels[s]) for s in kept):
            kept.append(int(k))
        if len(kept) == n_best:
            break
    return [(f"joint {name}", grid[k]) for k in kept]


def _collapse(layout: _Layout, lo, hi, phi, light, index) -> str | None:
    """Why one component's fit looks collapsed, or ``None``.

    Collapsed means its light fraction is below one percent, or its dilution scalar sits at
    the bottom of its prior, or at least two of its free labels are on their bounds.
    """
    name = layout.names[index]
    reasons = []
    if float(light[index]) < _COLLAPSED_LIGHT:
        reasons.append(f"its light fraction is {float(light[index]):.4f}")
    key = {"radius_ratio": f"ratio_{name}", "scalar": f"scale_{name}"}.get(layout.dilution)
    j = layout.index(key) if key is not None else None
    if j is not None and np.isfinite(lo[j]) and phi[j] <= lo[j] * 1.5 + 1e-12:
        reasons.append(f"{key} is at the bottom of its prior ({phi[j]:.4g})")
    tol = _bound_tol(lo, hi)
    on_bounds = []
    for label in ("teff", "logg", "mh", "vsini"):
        site = layout.label_key(label, name)
        if label == "mh" and layout.shared_mh:
            continue
        k = layout.index(site)
        if k is not None and (phi[k] <= lo[k] + tol[k] or phi[k] >= hi[k] - tol[k]):
            on_bounds.append(site)
    if len(on_bounds) >= 2:
        reasons.append(f"{', '.join(on_bounds)} are on their bounds")
    return "; ".join(reasons) if reasons else None


def _offset_design(problem, n_star: int) -> np.ndarray:
    """The additive nuisance as columns: ``(n_star, n_pix, n_star * (order + 1))``."""
    basis = np.asarray(problem.basis)
    per = basis.shape[1]
    design = np.zeros((n_star, basis.shape[0], n_star * per))
    for s in range(n_star):
        design[s, :, s * per : (s + 1) * per] = basis
    return design


def _profiled_null(stats, rows, design, gram) -> float:
    """``min_c L_data(rows + X c)``: the chi-square with the additive nuisance profiled out."""
    rows = jnp.asarray(rows)
    loss = float(stats.chi2(rows))
    if design.shape[-1] == 0:
        return loss
    residual = np.asarray(stats.h - stats.matvec(rows))
    r = np.einsum("spk,sp->k", design, residual)
    ridge = 1e-10 * max(float(np.trace(gram)), 1e-300) * np.eye(gram.shape[0])
    return float(loss - r @ np.linalg.solve(gram + ridge, r))


@dataclass(frozen=True)
class EpochFit:
    """How an ``"epochs"`` label fit was optimised, and what it could not measure.

    Attributes
    ----------
    chi2
        ``L_data`` at the optimum, the epoch chi-square without the priors.
    objective
        ``L_data`` plus the non-uniform priors in chi-square units (for the offsets' Normal,
        the sum of their squared ratios to the prior scale).
    n_good
        Epoch pixels in the chi-square.
    status, iterations
        How the Levenberg-Marquardt run that reached the optimum stopped, and after how many
        iterations.
    lm_runs
        Levenberg-Marquardt runs in total: warm starts and restarts.
    designed
        ``({"candidate", "objective", "status"}, ...)`` for the warm-started runs.
    rounds
        One record per restart round: the component treated as faint, whether a collapse
        was detected and the joint restart run, the number and kinds of starts, the
        objective before, the gain, and which restart moved the optimum.
    moved_by
        The restart kinds that lowered the optimum, in order.
    at_bounds
        Free sites excluded from the covariance, with the reason: a bound, or the rotation
        plateau below half a model pixel.
    light_sigma
        Formal error of each component's band-median light fraction, by the delta method.
    chi2_continuum, chi2_nearest_node
        The nulls in the same chi-square: the additive nuisance alone, and the best scan
        candidate with the nuisance profiled out.
    multimodal
        Whether a run from another start stopped within 9 in chi-square of the optimum with
        some free label more than three formal sigma away.
    notes
        Anything the restarts could not resolve, such as a collapse that survived the joint
        restart.
    seconds
        Wall time of the optimisation, restarts included.
    """

    chi2: float
    objective: float
    n_good: int
    status: str
    iterations: int
    lm_runs: int
    designed: tuple
    rounds: tuple
    moved_by: tuple
    at_bounds: dict
    light_sigma: dict
    chi2_continuum: float
    chi2_nearest_node: float
    multimodal: bool
    notes: tuple
    seconds: float


def _fit_epochs(
    problem,
    specs,
    config,
    statistics,
    *,
    libraries,
    node_tables,
    scans,
    candidates,
    wave,
    assumptions,
    max_steps,
    restart_rounds,
    progress,
) -> LabelMatch:
    """The ``"epochs"`` comparison: warm-started LM, restart rounds, formal errors, nulls."""
    t0 = time.perf_counter()
    names = tuple(config["names"])
    layout, lo, hi, terms = _epoch_layout(specs, config, problem)
    if layout.n == 0:
        raise ValueError("every label is declared Fixed; there is nothing to fit")
    objective = _EpochObjective(layout, terms)
    budget = {"max_iter": max(int(max_steps), 1), "max_eval": 4 * max(int(max_steps), 1)}
    report = None
    if progress is not None:

        def report(step, potential, grad_norm, phi):
            progress(step, potential, grad_norm, layout.to_params(phi))

    def lm(start, kind):
        return _levenberg_marquardt(
            objective, problem, statistics, start, lo, hi, kind=kind, progress=report, **budget
        )

    label_axes = [tuple(a) for a in problem.label_axes]
    runs, designed, inits = [], [], []
    for index, (_, picks) in enumerate(candidates):
        init = _init_from_scan(specs, config, names, label_axes, node_tables, scans, picks)
        run = lm(np.clip(layout.to_flat(init), lo, hi), f"scan candidate {index}")
        runs.append(run)
        inits.append(init)
        designed.append({"candidate": index, "objective": run.objective, "status": run.status})
    finite = [i for i, run in enumerate(runs) if np.isfinite(run.objective)]
    if not finite:
        raise RuntimeError(
            "every warm start gave a non-finite epoch chi-square. Check that the statistics "
            "and the libraries are finite and that the priors lie inside the grids."
        )
    first = min(finite, key=lambda i: runs[i].objective)
    best, best_init = runs[first], inits[first]

    def restart_plan(phi, joint: bool):
        light = objective.light(phi, problem)
        faint = int(np.argmin(light))
        plan = []
        for index in range(len(names)):
            plan += _vsini_starts(objective, problem, statistics, layout, lo, hi, phi, index)
        plan += _node_starts(
            objective, problem, statistics, layout, lo, hi, phi, faint, node_tables
        )
        collapsed = _collapse(layout, lo, hi, phi, light, faint)
        if collapsed is not None or joint:
            plan += _joint_starts(
                objective, problem, statistics, layout, lo, hi, phi, faint, node_tables
            )
        return faint, collapsed, plan

    rounds, moved_by = [], []
    joint_on = None  # the optimum the joint restart last ran from
    for number in range(restart_rounds):
        faint, collapsed, plan = restart_plan(best.phi, joint=False)
        if collapsed is not None:
            joint_on = best.phi.copy()
        tried = [lm(start, kind) for kind, start in plan]
        runs += tried
        before = best.objective
        winner = min(tried, key=lambda r: r.objective) if tried else None
        gain = before - winner.objective if winner is not None else 0.0
        record = {
            "round": number,
            "faint": names[faint],
            "collapse": collapsed,
            "starts": len(plan),
            "kinds": sorted({kind.split(" ")[0] for kind, _ in plan}),
            "objective_before": before,
            "gain": max(gain, 0.0),
            "moved_by": winner.kind if gain > 0.0 else None,
        }
        rounds.append(record)
        if gain > 0.0:
            best = winner
            moved_by.append(winner.kind)
        if gain < _RESTART_GAIN:
            break

    notes = []
    light = objective.light(best.phi, problem)
    faint = int(np.argmin(light))
    collapsed = _collapse(layout, lo, hi, best.phi, light, faint)
    if collapsed is not None and restart_rounds > 0:
        if joint_on is None or not np.array_equal(joint_on, best.phi):
            # The collapse appeared after the last joint restart, or none ran: run it once
            # from the final optimum before recording anything.
            _, _, plan = restart_plan(best.phi, joint=True)
            plan = [(kind, start) for kind, start in plan if kind.startswith("joint")]
            tried = [lm(start, kind) for kind, start in plan]
            runs += tried
            winner = min(tried, key=lambda r: r.objective) if tried else None
            gain = best.objective - winner.objective if winner is not None else 0.0
            rounds.append(
                {
                    "round": len(rounds),
                    "faint": names[faint],
                    "collapse": collapsed,
                    "starts": len(plan),
                    "kinds": ["joint"],
                    "objective_before": best.objective,
                    "gain": max(gain, 0.0),
                    "moved_by": winner.kind if gain > 0.0 else None,
                }
            )
            if gain > 0.0:
                best = winner
                moved_by.append(winner.kind)
            light = objective.light(best.phi, problem)
            faint = int(np.argmin(light))
            collapsed = _collapse(layout, lo, hi, best.phi, light, faint)
        if collapsed is not None:
            notes.append(
                f"{names[faint]!r} looks collapsed after the joint restart ({collapsed}). A "
                "component with no light has no labels; its labels and light fraction are "
                "not measurements. On D65 this happened where the archived orbit did not "
                "describe the data."
            )
    elif collapsed is not None:
        notes.append(f"{names[faint]!r} looks collapsed ({collapsed}) and restarts were off.")
    if best.status != "converged":
        notes.append(
            f"the run that reached the optimum stopped as {best.status!r} after "
            f"{best.iterations} iterations"
        )

    # -- formal errors --------------------------------------------------------
    plateau = _plateau_mask(layout, best.phi, problem.dv_kms)
    cov, excluded = _formal_covariance(best, lo, hi, plateau)
    flat_labels = layout.flat_labels()
    tol = _bound_tol(lo, hi)
    at_bounds = {}
    for k, label in enumerate(flat_labels):
        if not excluded[k]:
            continue
        if plateau[k]:
            at_bounds[label] = "rotation plateau"
        elif best.phi[k] <= lo[k] + tol[k]:
            at_bounds[label] = "lower bound"
        elif best.phi[k] >= hi[k] - tol[k]:
            at_bounds[label] = "upper bound"
        else:
            at_bounds[label] = "no curvature"
    jac = objective.light_jac(best.phi, problem)
    ok = np.isfinite(np.diag(cov))
    sub = np.nan_to_num(cov[np.ix_(ok, ok)])
    variance = np.einsum("ip,pq,iq->i", jac[:, ok], sub, jac[:, ok])
    light_sigma = {n: float(np.sqrt(max(variance[i], 0.0))) for i, n in enumerate(names)}

    # -- nulls in the same chi-square ----------------------------------------------
    design = _offset_design(problem, len(names))
    gram = (
        np.einsum(
            "spk,spl->kl",
            design,
            np.asarray(jax.vmap(statistics.matvec, in_axes=-1, out_axes=-1)(jnp.asarray(design))),
        )
        if design.shape[-1]
        else np.zeros((0, 0))
    )
    zero_rows = np.zeros((len(names), problem.n_pix))
    chi2_continuum = _profiled_null(statistics, zero_rows, design, gram)
    nearest = []
    for init in inits:
        start = np.clip(layout.to_flat(init), lo, hi)
        if layout.offsets:
            for name in names:
                start[layout.slot(f"offset_{name}")] = 0.0
        nearest.append(_profiled_null(statistics, objective.rows(start, problem), design, gram))
    chi2_nearest = float(min(nearest))

    # -- a second basin ----------------------------------------------------------------
    sigma = np.sqrt(np.diag(cov))
    labels_at = [
        layout.index(layout.label_key(label, name))
        for name in names
        for label in ("teff", "logg", "mh", "vsini")
    ]
    labels_at = sorted({j for j in labels_at if j is not None and np.isfinite(sigma[j])})
    multimodal = any(
        run is not best
        and np.isfinite(run.objective)
        and run.objective - best.objective < 9.0
        and any(abs(run.phi[j] - best.phi[j]) > 3.0 * sigma[j] for j in labels_at)
        for run in runs
    )

    free_grad = np.where(excluded, 0.0, best.grad)
    result = MAPResult(
        params=layout.to_params(best.phi),
        unconstrained={},
        potential=0.5 * best.objective,
        grad_norm=float(np.linalg.norm(free_grad)),
        converged=best.status == "converged",
        num_steps=int(best.iterations),
    )
    record = EpochFit(
        chi2=float(best.chi2),
        objective=float(best.objective),
        n_good=int(statistics.n_good),
        status=best.status,
        iterations=int(best.iterations),
        lm_runs=len(runs),
        designed=tuple(designed),
        rounds=tuple(rounds),
        moved_by=tuple(moved_by),
        at_bounds=at_bounds,
        light_sigma=light_sigma,
        chi2_continuum=float(chi2_continuum),
        chi2_nearest_node=chi2_nearest,
        multimodal=bool(multimodal),
        notes=tuple(notes),
        seconds=time.perf_counter() - t0,
    )
    return LabelMatch(
        names=names,
        result=result,
        problem=problem,
        specs=specs,
        config=config,
        libraries=libraries,
        covariance=cov,
        site_order=flat_labels,
        node_scan=tuple(scans),
        node_tables=tuple(node_tables),
        candidates=tuple(candidates),
        wave=wave,
        start=best_init,
        assumptions=assumptions,
        statistics=statistics,
        epoch_fit=record,
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelMatch:
    """Labels for each component, with the nulls against which they should be read.

    Each number is reported against a reference. The chi-square is quoted against a fit with
    no template at all and against the best raw grid node. Each label's posterior width is
    quoted against its prior width, since a parameter returned at its prior width was not
    measured. The Laplace error is quoted against the spread from refitting the
    disentangling posterior's own draws, because on correlated residuals the formal error
    runs five to ten times optimistic (Gebruers et al. 2022).

    In ``"epochs"`` mode (``assumptions["compare"]``) the quantities keep their names and
    change their space. :attr:`chi2` and both nulls are epoch chi-squares ``L_data`` over
    :attr:`n_pixels_used` epoch pixels; ``result`` is a :class:`~albireo.MAPResult` whose
    ``potential`` is half the objective and whose ``num_steps`` are Levenberg-Marquardt
    iterations; ``covariance`` is ``2 H^-1`` from the Gauss-Newton matrix, in the
    constrained parameters, with NaN rows for the sites in :attr:`at_bounds`; and
    ``epoch_fit`` (:class:`EpochFit`) records the restarts and any unresolved collapse.
    The formal errors of this mode were measured close to calibrated for Teff, log g and
    [M/H] and too small by about three for ``v sin i`` and the light fraction
    (``docs/math.md`` §9.5); :func:`refit_draws` does not apply to it.

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
    statistics: Any = field(default=None, repr=False)
    epoch_fit: Any = field(default=None, repr=False)

    @property
    def compare(self) -> str:
        """The comparison this fit ran: ``"native"``, ``"matched"`` or ``"epochs"``."""
        return str(self.assumptions.get("compare", "native"))

    @property
    def restarts(self) -> tuple:
        """The ``"epochs"`` restart rounds (:attr:`EpochFit.rounds`); empty otherwise."""
        return () if self.epoch_fit is None else tuple(self.epoch_fit.rounds)

    @property
    def at_bounds(self) -> dict[str, str]:
        """``"epochs"`` sites without a formal error, and why: a bound or the plateau."""
        return {} if self.epoch_fit is None else dict(self.epoch_fit.at_bounds)

    @property
    def flux_ratio_errors(self) -> dict[str, float]:
        """Formal errors of :attr:`flux_ratio` in ``"epochs"`` mode; empty otherwise.

        Measured too small by about three on simulated Gaia RVS binaries, where the
        remaining light error is carried by the label and orbit errors that the formal
        covariance at a fixed orbit does not include (``docs/math.md`` §9.5).
        """
        return {} if self.epoch_fit is None else dict(self.epoch_fit.light_sigma)

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
        radius ratio, so radii known from eclipses give an external check. Empty for the
        other dilution models, which carry no such scalar.
        """
        if self.problem.dilution != "radius_ratio":
            return {}
        return {self.names[0]: 1.0} | {
            name: self._value(f"ratio_{name}") for name in self.names[1:]
        }

    def errors(self, method: str = "laplace") -> dict[str, dict[str, float]]:
        """Label uncertainties.

        ``"laplace"`` is the curvature at the MAP, projected to the constrained
        parameterization by the delta method; in ``"epochs"`` mode it is ``2 H^-1`` from the
        Gauss-Newton matrix, already in the constrained parameters, and a label in
        :attr:`at_bounds` has no entry. ``"draws"`` is the spread over refits of the
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
                if key is None or key not in sigma or not np.isfinite(sigma[key]):
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
        standard deviation must be pushed back through it. The ``"epochs"`` covariance is
        already in the constrained parameters.
        """
        if self.epoch_fit is not None:
            return 1.0
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
            if key in sigma and widths[key] > 0 and np.isfinite(sigma[key])
        }

    @property
    def hit_step_cap(self) -> bool:
        """Whether the optimizer stopped on its budget rather than on its criterion.

        For L-BFGS the criterion is an absolute gradient norm that real data rarely reach;
        for the ``"epochs"`` Levenberg-Marquardt it is a predicted decrease below 0.001 in
        chi-square, which a converged fit does reach.
        """
        if self.epoch_fit is not None:
            return self.epoch_fit.status in ("iteration_budget", "evaluation_budget")
        return not bool(self.result.converged)

    @property
    def multimodal(self) -> bool:
        """Whether a runner-up starting basin came within ``delta chi-square < 9``.

        In ``"epochs"`` mode, whether any optimisation from another start stopped within 9
        in chi-square with a free label more than three formal sigma from the optimum, which
        a single quadratic basin cannot produce.
        """
        if self.epoch_fit is not None:
            return bool(self.epoch_fit.multimodal)
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
        """Chi-square at the MAP, without the jitter rescaling, so the nulls compare.

        In ``"epochs"`` mode, the epoch chi-square ``L_data`` at the optimum.
        """
        if self.epoch_fit is not None:
            return float(self.epoch_fit.chi2)
        residual = (self._rows() - self.problem.data) / self.problem.sigma
        return float(jnp.sum(self.problem.weight * residual**2))

    @property
    def chi2_continuum(self) -> float:
        """The null with no template at all: the additive nuisance alone, profiled.

        A fitted chi-square not well below this means the spectrum carried no label
        information and the reported labels are the priors. In ``"epochs"``
        mode it is the epoch chi-square of the nuisance alone.
        """
        if self.epoch_fit is not None:
            return float(self.epoch_fit.chi2_continuum)
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
        broadening and fitted dilution contribute over selecting the nearest node. In
        ``"epochs"`` mode it is the epoch chi-square of the best warm-start candidate (its
        nodes, scanned rotations and velocities, and starting dilution) with the additive
        nuisance profiled out.
        """
        if self.epoch_fit is not None:
            return float(self.epoch_fit.chi2_nearest_node)
        return float(sum(np.min(scan[:, 0]) for scan in self.node_scan))

    @property
    def n_pixels_used(self) -> int:
        """Pixels contributing to the likelihood, after exclusions (epoch pixels in
        ``"epochs"`` mode)."""
        if self.epoch_fit is not None:
            return int(self.epoch_fit.n_good)
        return int(np.asarray(self.problem.weight).sum())

    # -- derived astrophysics ---------------------------------------------

    def light_fractions(self, wave=None):
        """Fitted light fractions per component, shape ``(n_star, n_pix)``.

        The spectroscopic light ratio is a deliverable in its own right: published values
        match light-curve ratios to a few percent, and downstream cross-correlation codes are
        more sensitive to an incorrect flux ratio than to an incorrect temperature. A
        :class:`FixedDilution` fit returns the assumed fractions unchanged.
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
        star's share of the system's light. :mod:`albireo.handoff` writes it to a file.
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
        masks, Gaia's ``rv_template_*`` grid) need the answer snapped to a node they hold;
        the grid's own step is the appropriate granularity.
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
        epochs = self.epoch_fit
        space = "epoch pixels" if epochs is not None else "pixels"
        lines = [
            f"LabelMatch ({self.compare} comparison): {len(self.names)} component(s), "
            f"{self.n_pixels_used} {space}, chi2 = {self.chi2:.1f}"
            + (
                f" (reduced {self.chi2 / max(self.n_pixels_used, 1):.4f})"
                if epochs is not None
                else ""
            ),
            f"  null (no template)   chi2 = {self.chi2_continuum:.1f}",
            f"  null (nearest node)  chi2 = {self.chi2_nearest_node:.1f}",
            "",
        ]
        held = self.at_bounds
        for name in self.names:
            lines.append(f"  {name}:")
            fixed = set(self.fixed[name])
            for label, value in self.labels[name].items():
                unit = {"teff": " K", "vsini": " km/s", "v_kms": " km/s"}.get(label, " dex")
                formal = laplace.get(name, {}).get(label)
                site = self._site_key(label, name)
                text = f"    {label:<6} {value:10.3f}{unit}"
                if label in fixed:
                    text += "   (fixed)"
                elif site is not None and site in held:
                    text += f"   (no formal error: {held[site]})"
                elif formal is not None:
                    text += f" +- {formal:.3f} formal"
                    if drawn and label in drawn.get(name, {}):
                        spread = drawn[name][label]
                        text += f", +- {spread:.3f} from draws (x{spread / max(formal, 1e-12):.1f})"
                lines.append(text)
            light_error = self.flux_ratio_errors.get(name)
            lines.append(
                f"    light fraction {self.flux_ratio[name]:.4f}"
                + (f" +- {light_error:.4f} formal" if light_error is not None else "")
                + " (median over the band)"
            )

        if epochs is not None:
            lines += [
                "",
                f"  Levenberg-Marquardt: {epochs.lm_runs} runs; the optimum's run {epochs.status} "
                f"after {epochs.iterations} iterations ({epochs.seconds:.1f} s)",
            ]
            for entry in epochs.rounds:
                moved = f"moved by {entry['moved_by']}" if entry["moved_by"] else "no move"
                lines.append(
                    f"    restart round {entry['round']}: {entry['starts']} starts "
                    f"({', '.join(entry['kinds']) or 'none'}), faint {entry['faint']!r}, gain "
                    f"{entry['gain']:.2f}, {moved}"
                    + (f"; collapse: {entry['collapse']}" if entry["collapse"] else "")
                )
            if not epochs.rounds:
                lines.append("    no restart rounds were run")
            for note in epochs.notes:
                lines.append(f"  ! {note}")
            for note in () if self.statistics is None else tuple(self.statistics.notes):
                lines.append(f"  ! {note}")
            lines += [
                "",
                "  Formal errors from the Gauss-Newton matrix at a fixed orbit. On simulated",
                "  Gaia RVS binaries they were close to calibrated for Teff, log g and [M/H]",
                "  (68th-percentile pulls 1.4, 1.0, 1.4) and too small by about three for",
                "  v sin i and the light fraction (3.5 and 2.7); see docs/math.md 9.5.",
            ]
        elif drawn is None:
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
        lines += _degenerate_lines(self.flagged_correlations(), epochs=epochs is not None)
        if self.multimodal:
            lines += [
                "",
                "  The scan found a second basin within delta chi2 < 9. The reported labels are",
                "  the better of them, not the only ones consistent with the data.",
            ]
        if self.hit_step_cap and epochs is None:
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


def _degenerate_lines(flagged, *, epochs: bool) -> list[str]:
    """The "Degenerate pairs" block of :meth:`LabelMatch.summary`.

    In the ``"epochs"`` comparison the additive offsets of different components are
    degenerate by construction: the operator multiplies each component's row by its declared
    light fraction, and a low-order polynomial is barely changed by the shifts, so only the
    light-weighted sum of the offsets of each order reaches the epoch spectra, and their
    correlations run near -0.99 on every fit. Listed one by one they hide the real pairs, so
    they are grouped into one line; a pair of an offset with a label stays listed.
    """
    if not flagged:
        return []
    offsets, listed = [], []
    for a, b, rho in flagged:
        if epochs and a.startswith("offset_") and b.startswith("offset_"):
            offsets.append((a, b, rho))
        else:
            listed.append((a, b, rho))
    lines = ["", "  Degenerate pairs:"]
    for a, b, rho in listed:
        note = ""
        if {a.split("_")[0], b.split("_")[0]} == {"teff", "logg"}:
            note = "  <- expected when both are free; fix log g from the orbit"
        lines.append(f"    {a} / {b}: {rho:+.3f}{note}")
    if offsets:
        worst = max(offsets, key=lambda item: abs(item[2]))
        lines.append(
            f"    {len(offsets)} additive-offset pair(s), up to {worst[2]:+.3f} "
            f"({worst[0]} / {worst[1]}): expected, since only the light-weighted sum of the "
            "components' offsets reaches the epochs, and not a degeneracy of the labels"
        )
    return lines


def _filled_words(library) -> str:
    """The library's filled nodes in one phrase, for the assumptions block."""
    filled = library.meta.get("filled_nodes") or []
    listed = "; ".join(
        ", ".join(f"{k} {f[k]:g}" for k in library.label_names if k in f)
        + f" (along {f.get('axis', '?')})"
        for f in filled
    )
    return f"{len(filled)} node(s) filled by interpolation, not published: {listed}"


def refit_draws(match: LabelMatch, draws, *, max_steps: int = 60, seed: int = 0) -> LabelMatch:
    """Refit the labels once per posterior draw of the component spectra.

    The Laplace covariance measures the curvature of the likelihood at the optimum, which
    on correlated residuals understates the uncertainty. This function measures how far the
    labels move when the component spectra move as the disentangling posterior permits,
    including the exchange modes that trade flux between components (``docs/math.md``
    §9.5). It is the loop that :func:`albireo.handoff.export_draws` documents for an
    external code, run internally.

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
    if match.epoch_fit is not None:
        raise ValueError(
            "refit_draws refits d_hat draws against the d_hat comparison, and this match used "
            "compare='epochs', which compares with the epoch spectra. Its natural spread is a "
            "refit over draws of the epoch noise, which is not implemented; the formal errors "
            "of this mode were measured close to calibrated for Teff, log g and [M/H] and too "
            "small by about three for v sin i and the light fraction (docs/math.md 9.5)."
        )
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
