"""Templates fitted to the epochs: labels, rotation and light fractions with the velocities.

**Experimental.** The names and return shapes defined here may change.

:func:`albireo.todcor` measures velocities against templates that are given, and its
accuracy is bounded by the agreement between templates and stars. This module fits the
templates to the same epochs. The model of one epoch is that of ``docs/math.md`` §10.1,

    y_j = 1 + sum_i a_i R_j B_j T(delta_ij) t_i(theta) + (nuisance) + noise,

with the templates ``t_i`` rendered from a synthetic library at labels ``theta``: the
effective temperature, surface gravity and rotation of every star and a metallicity, shared
by default. The shifts ``delta_ij`` and the nuisance are parameters of one epoch. The
labels and the light fractions ``a_i`` are parameters of the whole dataset. The sum of the
chi-squares of the epochs is minimised over both sets (``docs/math.md`` §10.7).

The global parameters are found by a bounded Levenberg-Marquardt on the reduced objective,
in which every epoch's shifts and nuisance are at their own minimum for the trial labels
(variable projection, Golub & Pereyra 1973). The epochs' shifts are then measured again by
:func:`albireo.todcor` with the fitted templates, over the whole velocity range, and the
two steps alternate until the velocities stop changing. The second step is needed because
the first follows each epoch's minimum locally, and a template far from the star can leave
an epoch in the wrong minimum of its surface.

The curvature of the reduced objective gives the covariance of the global parameters, with
the velocities of every epoch marginalised. The same matrices give the derivative of every
epoch's velocities with respect to every global parameter, which is recorded on the
velocity table (``VelocityTable.sensitivity``). An error of a template moves the velocities
of all epochs together, by an amount that depends on how far apart the lines are, so the
part of the velocity error that comes from the templates is correlated between epochs.

A star can carry components that no library spectrum has, as vectors added to its template
before rotation with fitted amplitudes (:class:`TemplateDeviation`): emission in line
cores, for instance. They move with the star and are fitted with its labels.

The joint optimisation of templates and epoch velocities over a multi-epoch dataset is the
approach of Nachmani, Faigler & Mazeh (2026). No spectrum is synthesised here: the templates
are interpolated in a published grid (:mod:`albireo.library`).

References
----------
Golub, G. H. & Pereyra, V. 1973, SIAM Journal on Numerical Analysis, 10, 413
Nachmani, G., Faigler, S. & Mazeh, T. 2026, arXiv:2602.05610
Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806
"""

from __future__ import annotations

import itertools
import math
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from albireo.data import Dataset
from albireo.grids import C_KMS, LogGrid, _in_epoch_blocks
from albireo.library import SUPPORTED_MEDIA, SpectralLibrary, library_interpolator
from albireo.match import (
    _as_spec,
    _is_fixed,
    _levenberg_marquardt,
)
from albireo.operators import _gaussian_kernel_numpy, _rotational_kernel_numpy
from albireo.todcor import (
    _BLEND_CHI2,
    Template,
    VelocityTable,
    _ar1_apply,
    _blend_slack,
    _blend_threshold,
    _convolved_templates,
    _dv_dpix,
    _epoch_columns,
    _LightNotMeasured,
    _lsf_key,
    _minimize_box_quadratic,
    _prepare_epoch,
    _resolve_correlation,
    _resolve_light,
    _resolve_ranges,
    _run,
    todcor,
)

__all__ = ["TemplateDeviation", "TemplateFit", "fit_templates", "line_deviation"]

_LABELS = ("teff", "logg", "mh")
# Half-steps of the central differences of a template with respect to its labels. The
# interpolant is a cubic in each label between two nodes, so a step of a fiftieth of the
# node spacing of the published grids leaves an error of order 1e-4 of the cubic term.
_STEPS = {"teff": 5.0, "logg": 0.01, "mh": 0.01, "vsini": 0.1}
_VSINI_MAX_KMS = 400.0
_LIGHT_BOUNDS = (1e-4, 1.0 - 1e-4)
_STACK_MULTIPLE = 4
_MIN_EPOCHS = 2
# The largest number of cells of the integer shifts an epoch's minimum is followed across
# in one evaluation, and of alternations of that walk with the epoch's free scale.
_MAX_CELL_MOVES = 64
_MAX_SCALE_ROUNDS = 20
# A shift of a table within this many pixels of an integer is on the face of a cell.
_FACE_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateDeviation:
    """A component of one star's spectrum that its library does not have.

    The vector is added to the star's interpolated spectrum, in units of its continuum,
    before rotation, and its amplitude is fitted with the labels. It moves with the star.

    Parameters
    ----------
    name
        The name of the amplitude in the fit, combined with the star's as
        ``"<name>_<star>"``.
    vector
        The component on the template grid, in the star's rest frame, at unit amplitude.
    amplitude
        A float (held), or a specification with the vocabulary of :mod:`albireo.facade`:
        :class:`~albireo.facade.Between` bounds it, and :class:`~albireo.facade.Known` gives
        it a Gaussian prior and no bound. The default is a free amplitude started at zero.
    """

    name: str
    vector: Any
    amplitude: Any = None


def line_deviation(
    grid: LogGrid, centres_angstrom, sigma_angstrom: float, strengths=None
) -> np.ndarray:
    """Gaussian profiles at given wavelengths, as the vector of a :class:`TemplateDeviation`.

    Each profile has unit area in Angstrom times its strength, so the fitted amplitude is
    the equivalent width, in Angstrom, of a line of unit strength: positive for emission
    that fills an absorption core.

    Parameters
    ----------
    grid
        The template grid.
    centres_angstrom
        The line centres, on the wavelength scale of ``grid``.
    sigma_angstrom
        The Gaussian width of every profile before the star's rotation.
    strengths
        The relative strengths of the lines. Default: one for every line.
    """
    centres = np.atleast_1d(np.asarray(centres_angstrom, dtype=np.float64))
    strengths = (
        np.ones(centres.size)
        if strengths is None
        else np.atleast_1d(np.asarray(strengths, dtype=np.float64))
    )
    if strengths.shape != centres.shape:
        raise ValueError("strengths must give one value per line centre")
    if not float(sigma_angstrom) > 0.0:
        raise ValueError("sigma_angstrom must be positive")
    wave = np.asarray(grid.wave, dtype=np.float64)
    norm = float(sigma_angstrom) * math.sqrt(2.0 * math.pi)
    out = np.zeros(wave.size)
    for centre, strength in zip(centres, strengths, strict=True):
        out += strength * np.exp(-0.5 * ((wave - centre) / float(sigma_angstrom)) ** 2) / norm
    return out


# ---------------------------------------------------------------------------
# One star: its label space and how a template is rendered
# ---------------------------------------------------------------------------


@jax.jit
def _normalized(interpolator, point):
    """The normalized spectrum of a library at one label point, compiled once per library."""
    return interpolator(point)[0]


class _LabelSpace:
    """One or more library boxes that adjoin in temperature, as one label space.

    A star is rendered from the box that contains its temperature, with the other labels
    moved into that box. The interpolant is continuous across a shared node where the
    boxes agree there, as the BOSZ boxes of the registry do.
    """

    def __init__(self, libraries, grid: LogGrid, medium: str, where: str) -> None:
        libraries = tuple(libraries)
        if not libraries:
            raise ValueError(f"{where}: no library")
        for lib in libraries:
            if not isinstance(lib, SpectralLibrary):
                raise TypeError(
                    f"{where}: a library box must be a SpectralLibrary; got {type(lib).__name__}"
                )
            if "teff" not in lib.label_names:
                raise ValueError(f"{where}: a library without a teff axis cannot be fitted")
            unknown = [axis for axis in lib.label_names if axis not in _LABELS]
            if unknown:
                raise ValueError(
                    f"{where}: the library has the axes {list(lib.label_names)}; the fit "
                    f"knows {list(_LABELS)}"
                )
        libraries = tuple(sorted(libraries, key=lambda lib: lib.bounds["teff"][0]))
        powers = {lib.resolving_power for lib in libraries}
        if len(powers) != 1:
            raise ValueError(
                f"{where}: the library boxes declare different resolving powers "
                f"({sorted(str(p) for p in powers)}); one template has one width"
            )
        self.resolving_power = next(iter(powers))
        self.libraries = libraries
        self.boxes: list[dict[str, Any]] = []
        for lib in libraries:
            resampled = lib.resampled_to(grid, medium=medium)
            self.boxes.append(
                {
                    "interpolator": library_interpolator(resampled),
                    "axes": tuple(resampled.label_names),
                    "bounds": {axis: tuple(lib.bounds[axis]) for axis in lib.label_names},
                }
            )
        self._upper = np.array([box["bounds"]["teff"][1] for box in self.boxes[:-1]])
        for below, above in zip(self.boxes[:-1], self.boxes[1:], strict=True):
            if above["bounds"]["teff"][0] > below["bounds"]["teff"][1]:
                raise ValueError(
                    f"{where}: the library boxes leave no template between "
                    f"{below['bounds']['teff'][1]:g} and {above['bounds']['teff'][0]:g} K; "
                    "boxes used as one label space must adjoin in temperature"
                )

    @property
    def sigma_kms(self) -> float:
        power = self.resolving_power
        if power is None:
            return 0.0
        return C_KMS / (float(power) * 2.0 * math.sqrt(2.0 * math.log(2.0)))

    def box(self, teff: float) -> dict:
        return self.boxes[int(np.searchsorted(self._upper, float(teff), side="right"))]

    def range(self, label: str, teff_lo: float, teff_hi: float) -> tuple[float, float] | None:
        """The range of a label: every box for the temperature, and for the others the
        whole range of the boxes between two temperatures. ``None`` where no box has it.

        A label outside the range of the box its star is in is moved to that box's edge
        when the template is rendered (:meth:`clamped`), so the objective is flat beyond
        it. The range common to the boxes would instead exclude a subgiant whose
        temperature prior reaches a box of dwarfs.
        """
        if label == "teff":
            return self.boxes[0]["bounds"]["teff"][0], self.boxes[-1]["bounds"]["teff"][1]
        spanned = [
            box
            for box in self.boxes
            if box["bounds"]["teff"][1] >= teff_lo and box["bounds"]["teff"][0] <= teff_hi
        ]
        having = [box["bounds"][label] for box in spanned if label in box["bounds"]]
        if not having:
            return None
        return min(lo for lo, _ in having), max(hi for _, hi in having)

    def clamped(self, values: Mapping[str, float], box: dict | None = None) -> dict[str, float]:
        """The labels as rendered: each moved into the box of the temperature, or ``box``."""
        box = self.box(values["teff"]) if box is None else box
        out = {key: float(value) for key, value in values.items()}
        for axis in box["axes"]:
            out[axis] = float(np.clip(values[axis], *box["bounds"][axis]))
        return out

    def deviation(self, values: Mapping[str, float], box: dict | None = None) -> np.ndarray:
        """``flux - 1`` at the labels, each moved into the box of the temperature, or ``box``."""
        box = self.box(values["teff"]) if box is None else box
        inside = self.clamped(values, box)
        point = [inside[axis] for axis in box["axes"]]
        normalized = _normalized(box["interpolator"], jnp.asarray(point))
        return np.asarray(normalized, dtype=np.float64) - 1.0


class _Star:
    """One star of the fit: label space, broadening, and the vectors of its deviations."""

    def __init__(self, name, space: _LabelSpace, grid: LogGrid, epsilon, macro_kms, vectors):
        self.name = str(name)
        self.space = space
        self.grid = grid
        self.epsilon = float(epsilon)
        self.macro_kms = float(macro_kms)
        self.vectors = [np.asarray(v, dtype=np.float64) for v in vectors]
        self._macro = (
            _gaussian_kernel_numpy(self.macro_kms / grid.dv_kms) if self.macro_kms > 0.0 else None
        )

    def broaden(self, spectrum: np.ndarray, vsini: float) -> np.ndarray:
        if vsini > 0.0:
            kernel = _rotational_kernel_numpy(vsini / self.grid.dv_kms, epsilon=self.epsilon)
            spectrum = np.convolve(spectrum, kernel, mode="same")
        if self._macro is not None:
            spectrum = np.convolve(spectrum, self._macro, mode="same")
        return spectrum

    def intrinsic(self, values: Mapping[str, float], amplitudes) -> np.ndarray:
        spectrum = self.space.deviation(values)
        for amplitude, vector in zip(amplitudes, self.vectors, strict=True):
            spectrum = spectrum + float(amplitude) * vector
        return spectrum

    def render(self, values: Mapping[str, float], amplitudes) -> np.ndarray:
        return self.broaden(self.intrinsic(values, amplitudes), float(values["vsini"]))

    def template(self, deviation: np.ndarray, name: str | None = None, meta=None) -> Template:
        return Template(
            name=self.name if name is None else name,
            grid=self.grid,
            deviation=deviation,
            sigma_kms=self.space.sigma_kms,
            v_zero_kms=0.0,
            meta={} if meta is None else meta,
        )


# ---------------------------------------------------------------------------
# The parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Parameter:
    key: str
    kind: str  # "label", "light" or "deviation"
    stars: tuple  # the stars whose template it changes, or the star of a light fraction
    label: str
    group: int = -1  # instrument of a light fraction
    index: int = -1  # position of a deviation among its star's


@dataclass
class _State:
    """The values of everything the templates and amplitudes are built from."""

    labels: list  # per star: {"teff", "logg", "mh", "vsini"}
    amplitudes: list  # per star: the amplitudes of its deviations
    light: np.ndarray  # (n_instruments, n_stars)

    def copy(self) -> _State:
        return _State(
            labels=[dict(v) for v in self.labels],
            amplitudes=[np.array(a, dtype=np.float64) for a in self.amplitudes],
            light=np.array(self.light, dtype=np.float64),
        )


def _declared_spec(value, where: str, default=None):
    """The specification of a declared parameter, checked where it is declared.

    A declaration that the specification itself rejects, such as ``Known`` with a zero
    width or ``Between`` with ``lo >= hi``, raises its ``ValueError`` prefixed with
    ``where``, so that the message names the parameter as well as the declaration.
    """
    spec = _as_spec(value, where, default)
    try:
        spec.distribution()
        spec.start()
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    return spec


def _prior_of(spec, where: str):
    """``(start, lower, upper, (mean, sigma) or None)`` of a free specification."""
    distribution = spec.distribution()
    start = float(np.asarray(spec.start(), dtype=np.float64))
    if isinstance(distribution, dist.Uniform):
        return (
            start,
            float(np.asarray(distribution.low)),
            float(np.asarray(distribution.high)),
            None,
        )
    if isinstance(distribution, dist.Normal):
        mean = float(np.asarray(distribution.loc))
        sigma = float(np.asarray(distribution.scale))
        # Known checks its own width. A Normal declared through Sampled is not checked by
        # numpyro (its arguments are not validated by default) and reaches this test.
        if not sigma > 0.0:
            raise ValueError(f"{where}: the prior width must be positive")
        return start, -np.inf, np.inf, (mean, sigma)
    raise TypeError(
        f"{where}: the template fit takes a held value, a range (Between) or a Gaussian "
        f"prior (Known); got a {type(distribution).__name__} prior"
    )


class _Layout:
    """The flat vector of the fitted parameters, with bounds, priors and the held values."""

    def __init__(self) -> None:
        self.parameters: list[_Parameter] = []
        self.start: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.priors: dict[str, tuple[float, float]] = {}

    def add(self, parameter: _Parameter, start, lower, upper, prior) -> None:
        if parameter.key in self.keys:
            raise ValueError(
                f"two fitted parameters are named {parameter.key!r}; the deviations of a "
                "star need names of their own, other than those of the labels"
            )
        if not lower < upper:
            raise ValueError(f"{parameter.key}: empty range [{lower:g}, {upper:g}]")
        self.parameters.append(parameter)
        self.start.append(float(np.clip(start, lower, upper)))
        self.lower.append(float(lower))
        self.upper.append(float(upper))
        if prior is not None:
            self.priors[parameter.key] = prior

    @property
    def n(self) -> int:
        return len(self.parameters)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(p.key for p in self.parameters)

    def unpack(self, phi, held: _State) -> _State:
        state = held.copy()
        fitted_light = False
        for value, par in zip(np.asarray(phi, dtype=np.float64), self.parameters, strict=True):
            if par.kind == "label":
                for star in par.stars:
                    state.labels[star][par.label] = float(value)
            elif par.kind == "deviation":
                state.amplitudes[par.stars[0]][par.index] = float(value)
            else:
                state.light[par.group, par.stars[0]] = float(value)
                fitted_light = True
        if fitted_light:
            state.light[:, 0] = 1.0 - state.light[:, 1:].sum(axis=1)
        return state


def _star_exchange(layout: _Layout, stars: Sequence[_Star]):
    """The exchange of two stars as a map of the fitted parameters, or ``None``.

    The chi-square of every epoch is unchanged where the labels, the amplitudes and the
    light fractions of two stars are exchanged together with their rows of the velocity
    table, provided the two are rendered in the same way. It changes by what the two
    differ in beside their fitted parameters: a limb darkening, a macroturbulence, or a
    label or a light fraction held to its star. The priors are those of the stars as
    declared, so a solution and its image differ in the objective, and the image of a
    solution reached with the stars in the other order can be the lower one.

    The exchange is defined where the fitted parameters of the two stars pair up: the
    same library boxes and deviation vectors, and the same parameters fitted for both.
    Returns ``(partner, complement)``: for every parameter the index of the one whose
    value it takes, and which parameters are light fractions, which take one minus
    their value. ``None`` where it is not defined.
    """
    if len(stars) != 2:
        return None
    first, second = stars
    alike = (
        len(first.space.libraries) == len(second.space.libraries)
        and all(a is b for a, b in zip(first.space.libraries, second.space.libraries, strict=True))
        and len(first.vectors) == len(second.vectors)
        and all(np.array_equal(a, b) for a, b in zip(first.vectors, second.vectors, strict=True))
    )
    if not alike:
        return None
    index = {
        (par.kind, par.label, par.index, par.group, par.stars): k
        for k, par in enumerate(layout.parameters)
    }
    partner = np.arange(layout.n)
    complement = np.zeros(layout.n, dtype=bool)
    for k, par in enumerate(layout.parameters):
        if par.kind == "light":
            complement[k] = True
            continue
        other = tuple(sorted(1 - star for star in par.stars))
        match = index.get((par.kind, par.label, par.index, par.group, other))
        if match is None:
            return None
        partner[k] = match
    return partner, complement


# ---------------------------------------------------------------------------
# The reduced objective
# ---------------------------------------------------------------------------


def _prepared_epoch(index: int, epoch, grid: LogGrid, nuisance_order) -> tuple:
    """One epoch's work (:func:`albireo.todcor._prepare_epoch`) and its weighted arrays.

    The weighted arrays are the square roots of the weights, the weighted data and nuisance
    basis, and the inverse of the basis's normal matrix (``None`` without a nuisance).
    """
    work = _prepare_epoch(index, epoch, grid, nuisance_order)
    weight = np.sqrt(np.asarray(work.w, dtype=np.float64))
    basis = np.asarray(work.basis, dtype=np.float64) * weight[:, None]
    inverse = np.linalg.inv(basis.T @ basis) if basis.shape[1] else None
    return work, (weight, np.asarray(work.z, dtype=np.float64) * weight, basis, inverse)


class _Objective:
    """``F(phi) = sum_j min over the epoch's own parameters of chi2_j + priors``.

    The epoch's own parameters are its shifts, its nuisance coefficients and, with a free
    scale, the scale of its composite. They are eliminated from the Gauss-Newton equations
    by a Schur complement, which is exact for a least-squares problem whose parameters
    separate in this way (``docs/math.md`` §10.7). The object has the interface of the
    bounded Levenberg-Marquardt of :mod:`albireo.match`.
    """

    def __init__(
        self,
        dataset: Dataset,
        stars: Sequence[_Star],
        layout: _Layout,
        held: _State,
        *,
        groups: Sequence[str],
        shifts,
        scales,
        correlation: Mapping[str, float],
        lsf_sigma_v,
        lsf_anchors_angstrom,
        nuisance_order,
        free_scale: bool,
        move: bool = True,
        prepared: Sequence[tuple] | None = None,
    ) -> None:
        self.stars = list(stars)
        self.layout = layout
        self.held = held
        self.grid = self.stars[0].grid
        self.lsf_sigma_v = lsf_sigma_v
        self.lsf_anchors = lsf_anchors_angstrom
        self.free_scale = bool(free_scale)
        self.move = bool(move)
        # ``prepared``: the (work, static) of every epoch of ``dataset`` from
        # :func:`_prepared_epoch`, which depend on the epoch, the grid and the nuisance
        # order only and can be shared between objectives.
        if prepared is None:
            prepared = [
                _prepared_epoch(j, epoch, self.grid, nuisance_order)
                for j, epoch in enumerate(dataset)
            ]
        self.works = [work for work, _ in prepared]
        self.static = [static for _, static in prepared]
        self.keys = [_lsf_key(epoch, lsf_sigma_v) for epoch in dataset]
        self.group = [list(groups).index(epoch.instrument) for epoch in dataset]
        self.correlation = [float(correlation.get(epoch.instrument, 0.0)) for epoch in dataset]
        self.shift_ref = np.array(shifts, dtype=np.float64).T.copy()  # (n_epochs, n_stars)
        self.scale_ref = np.array(scales, dtype=np.float64).copy()
        self.lower = np.asarray(layout.lower, dtype=np.float64)
        self.upper = np.asarray(layout.upper, dtype=np.float64)
        self.evaluations = 0
        # The epochs of the last evaluation without the Jacobian, with the point and the
        # starting shifts and scales they were found from.
        self._epochs_at: tuple | None = None

    # -- templates and their derivatives -------------------------------------

    def _vectors(self, phi, state: _State, jacobian: bool):
        """Every star's template, then the derivative of a template for each parameter.

        Returns the vectors, the star each belongs to, and the row of the derivative of
        star ``s`` with respect to parameter ``p`` under ``(p, s)``.
        """
        n_star = len(self.stars)
        intrinsic = [
            star.intrinsic(state.labels[i], state.amplitudes[i])
            for i, star in enumerate(self.stars)
        ]
        vectors = [
            star.broaden(intrinsic[i], float(state.labels[i]["vsini"]))
            for i, star in enumerate(self.stars)
        ]
        owners = list(range(n_star))
        slots: dict[tuple[int, int], int] = {}
        if not jacobian:
            return vectors, np.array(owners), slots
        for p, par in enumerate(self.layout.parameters):
            if par.kind == "light":
                continue
            for s in par.stars:
                star, labels = self.stars[s], state.labels[s]
                vsini = float(labels["vsini"])
                if par.kind == "deviation":
                    vector = star.broaden(star.vectors[par.index], vsini)
                else:
                    step = _STEPS[par.label]
                    up = min(float(phi[p]) + step, self.upper[p])
                    down = max(float(phi[p]) - step, self.lower[p])
                    # Both sides of a difference are rendered from the box of the star's
                    # temperature. Two boxes agree at their shared temperature only for the
                    # other labels both hold, and a temperature on the upper edge of a box
                    # that is not the last would select the next box.
                    box = star.space.box(labels["teff"])
                    if par.label == "vsini":
                        # Below half a pixel the rotation kernel is one pixel and the
                        # template does not depend on the rotation: the step reaches past
                        # that range, so that a rotation started inside it can leave.
                        flat = 0.5 * self.grid.dv_kms
                        up = min(max(up, flat + 2.0 * step), self.upper[p])
                    elif par.label == "teff":
                        inside = box["bounds"]["teff"]
                        up, down = min(up, inside[1]), max(down, inside[0])
                    if not up > down:
                        continue
                    if par.label == "vsini":
                        vector = (
                            star.broaden(intrinsic[s], up) - star.broaden(intrinsic[s], down)
                        ) / (up - down)
                    else:
                        high = star.space.deviation({**labels, par.label: up}, box)
                        low = star.space.deviation({**labels, par.label: down}, box)
                        vector = star.broaden((high - low) / (up - down), vsini)
                slots[(p, s)] = len(vectors)
                vectors.append(vector)
                owners.append(s)
        # A fixed number of rows per block of parameters, so that the compiled gather is
        # shared between fits that hold different parameters.
        while len(vectors) % _STACK_MULTIPLE:
            vectors.append(np.zeros_like(vectors[0]))
            owners.append(0)
        return vectors, np.array(owners), slots

    def _convolved(self, vectors, owners) -> dict:
        templates = [
            self.stars[int(owner)].template(vector, name=f"row{k}")
            for k, (vector, owner) in enumerate(zip(vectors, owners, strict=True))
        ]
        out = {}
        for key in dict.fromkeys(self.keys):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # raised once by the measurement
                stack, _ = _convolved_templates(
                    templates,
                    self.grid,
                    key[0],
                    self.lsf_sigma_v,
                    self.lsf_anchors,
                    epoch_sigma_kms=key[1],
                )
            out[key] = jnp.asarray(stack)
        return out

    # -- one epoch -----------------------------------------------------------

    @staticmethod
    def _gather(stack, work, weight, first, n_shifts: int):
        """Projected, weighted vectors at ``n_shifts`` integer shifts starting at ``first``.

        ``first`` has one integer per row of ``stack``. Returns ``(rows, pixels, n_shifts)``.
        """
        deltas = (first[:, None] + np.arange(n_shifts)[None, :]).astype(np.int32)
        columns = np.asarray(
            _epoch_columns(stack, work.rows, work.cols, work.vals, work.w, jnp.asarray(deltas))
        )
        return columns * weight[None, :, None]

    @staticmethod
    def _without_nuisance(x, basis, inverse):
        """``x`` (a vector, or one vector per row) with its projection on the nuisance removed."""
        if inverse is None:
            return x
        return x - ((x @ basis) @ inverse) @ basis.T

    def _cell(self, window, base, corner, light, scale, static):
        """The minimum of the chi-square over one cell of the integer shifts.

        ``window`` holds the templates at the shifts ``base - 1`` to ``base + 2``. Inside the
        cell whose lowest corner is ``corner`` every template is the linear interpolation
        of its two integer neighbours (``docs/math.md`` §10.3), so with the nuisance solved
        for, the chi-square is a quadratic in the fractional shifts. Its minimum over the
        cell is found in closed form, on a face where the unconstrained one lies outside.
        Returns the chi-square and the fractional shifts.
        """
        _, data, basis, inverse = static
        rows = np.arange(len(self.stars))
        k = corner - base + 1
        low = window[rows, :, k]
        slope = window[rows, :, k + 1] - low
        residual = self._without_nuisance(data - scale * (light @ low), basis, inverse)
        design = self._without_nuisance(scale * light[:, None] * slope, basis, inverse)
        fraction, value = _minimize_box_quadratic(
            -2.0 * (design @ residual), 2.0 * (design @ design.T)
        )
        return float(residual @ residual + value), np.asarray(fraction, dtype=np.float64)

    def _walk(self, stack, work, static, shift, light, scale):
        """The minimum of the epoch's chi-square over the shifts nearest ``shift``.

        The cell that holds the position is minimised exactly. Where the minimum is on a
        face of the cell, every cell that touches it is minimised as well and the position
        moves to the lowest, until no cell touching the position is lower: the procedure
        of the correlation's own refinement (:func:`albireo.todcor._refine`). A minimum
        left on a face is a kink of the surface, the pixel-locking of the linear shift
        operator, and the shift there does not respond to a small change of the templates.
        Returns the lowest corner and the fractional shifts of the cell, and the window of
        templates and its base, which hold the cell.
        """
        n_star = len(self.stars)
        corner = np.floor(shift).astype(int)
        base = corner.copy()
        window = self._gather(stack, work, static[0], base - 1, 4)
        chi2, fraction = self._cell(window, base, corner, light, scale, static)
        for _ in range(_MAX_CELL_MOVES):
            choices = []
            for i in range(n_star):
                here = [int(corner[i])]
                if fraction[i] <= 0.0:
                    here.append(int(corner[i]) - 1)
                elif fraction[i] >= 1.0:
                    here.append(int(corner[i]) + 1)
                choices.append(here)
            best = None
            for candidate in itertools.product(*choices):
                candidate = np.array(candidate)
                if np.array_equal(candidate, corner):
                    continue
                if np.any(candidate < base - 1) or np.any(candidate > base + 1):
                    base = corner.copy()
                    window = self._gather(stack, work, static[0], base - 1, 4)
                value, inside = self._cell(window, base, candidate, light, scale, static)
                if value < chi2 - 1e-10 * max(chi2, 1.0) and (best is None or value < best[0]):
                    best = (value, candidate, inside)
            if best is None:
                break
            chi2, corner, fraction = best
        return corner, fraction, window, base

    def _pair(self, stack, work, static, corner, window, base):
        """The weighted templates at the shifts ``corner`` and ``corner + 1``.

        They are read from the window of a walk (:meth:`_walk`) where it holds them, which
        it does for the cell the walk ends in, and gathered otherwise.
        """
        if window is not None:
            k = corner - base + 1
            if np.all((k >= 0) & (k <= 2)):
                rows = np.arange(len(self.stars))
                return np.stack([window[rows, :, k], window[rows, :, k + 1]], axis=-1)
        return self._gather(stack, work, static[0], corner, 2)

    def _epoch(self, stack, work, static, shift, scale: float, light):
        """One epoch at its own minimum: residual, local design, and where the minimum is.

        Returns the shifts, the scale, the weighted residual, the weighted design of the
        epoch's own free parameters (the shifts that are not on a face, the scale where it
        is free, the nuisance), the weighted templates at the shifts, the lowest corner
        and the fractional shifts of the cell, and which shifts are free.
        """
        _, data, basis, inverse = static
        n_star = len(self.stars)
        rows = np.arange(n_star)
        shift = np.array(shift, dtype=np.float64)
        scale = float(scale)
        window = base = None
        if self.move:
            corner, fraction, window, base = self._walk(stack, work, static, shift, light, scale)
        else:
            # At the shifts as given, which a correlation left on a face where they are
            # within rounding of an integer.
            nearest = np.rint(shift)
            on_face = np.abs(shift - nearest) < _FACE_TOLERANCE
            corner = np.where(on_face, nearest, np.floor(shift)).astype(int)
            fraction = np.where(on_face, 0.0, shift - corner)
        for round_ in range(_MAX_SCALE_ROUNDS):
            pair = self._pair(stack, work, static, corner, window, base)
            slope = pair[:, :, 1] - pair[:, :, 0]
            value = pair[:, :, 0] + fraction[:, None] * slope
            composite = light @ value
            if not self.free_scale:
                break
            # The scale and the nuisance are linear at given shifts.
            columns = np.column_stack([composite, basis])
            solved = np.linalg.lstsq(columns, data, rcond=None)[0]
            change, scale = abs(float(solved[0]) - scale), float(solved[0])
            # The last round ends with the scale solved at the shifts returned, so that the
            # residual and the design below are those of the shifts.
            if change < 1e-10 or not self.move or round_ == _MAX_SCALE_ROUNDS - 1:
                break
            corner, fraction, window, base = self._walk(
                stack, work, static, corner + fraction, light, scale
            )
        residual = self._without_nuisance(data - scale * composite, basis, inverse)
        free = (fraction > 0.0) & (fraction < 1.0)
        columns = [scale * light[i] * slope[i] for i in rows[free]]
        if self.free_scale:
            columns.append(composite)
        local = np.column_stack([*columns, basis]) if columns or basis.shape[1] else None
        return corner + fraction, scale, residual, local, value, corner, fraction, free

    # -- the sum over epochs -------------------------------------------------

    def _prior(self, phi):
        total = 0.0
        gradient = np.zeros(self.layout.n)
        hessian = np.zeros((self.layout.n, self.layout.n))
        for p, par in enumerate(self.layout.parameters):
            prior = self.layout.priors.get(par.key)
            if prior is None:
                continue
            mean, sigma = prior
            total += ((phi[p] - mean) / sigma) ** 2
            gradient[p] = 2.0 * (phi[p] - mean) / sigma**2
            hessian[p, p] = 2.0 / sigma**2
        return total, gradient, hessian

    def evaluate(self, phi, jacobian: bool, residuals: bool = False) -> dict:
        phi = np.asarray(phi, dtype=np.float64)
        self.evaluations += 1
        state = self.layout.unpack(phi, self.held)
        vectors, owners, slots = self._vectors(phi, state, jacobian)
        stacks = self._convolved(vectors, owners)
        n_star, n_par = len(self.stars), self.layout.n
        chi2 = 0.0
        n_pixels = 0
        n_local = 0
        schur = np.zeros((n_par, n_par))
        meat = np.zeros((n_par, n_par))
        gradient = np.zeros(n_par)
        shifts = np.empty_like(self.shift_ref)
        scales = np.empty_like(self.scale_ref)
        sensitivity = np.zeros((len(self.works), n_star, n_par))
        chi2_epochs = np.zeros(len(self.works))
        dof_epochs = np.zeros(len(self.works), dtype=int)
        kept = []
        # The Levenberg-Marquardt evaluates a trial point without the Jacobian and then,
        # where it accepts it, with. An epoch depends on the point and on the shifts and
        # scale it starts from, and the rows of the templates are rendered and convolved one
        # by one, so the epochs of the first evaluation are those of the second, bit for bit.
        found = None
        if jacobian and self._epochs_at is not None:
            point, shift_ref, scale_ref, epochs = self._epochs_at
            if (
                point == phi.tobytes()
                and shift_ref is self.shift_ref
                and scale_ref is self.scale_ref
            ):
                found = epochs
        epochs_here = []
        for j, work in enumerate(self.works):
            stack = stacks[self.keys[j]]
            static = self.static[j]
            light = state.light[self.group[j]]
            if found is not None:
                epoch = found[j]
            else:
                epoch = self._epoch(
                    stack[:n_star], work, static, self.shift_ref[j], self.scale_ref[j], light
                )
            epochs_here.append(epoch)
            shift, scale, residual, local, value, corner, fraction, free = epoch
            shifts[j], scales[j] = shift, scale
            chi2_epochs[j] = float(residual @ residual)
            if residuals:
                kept.append(residual)
            dof_epochs[j] = int(work.n_good) - (0 if local is None else int(local.shape[1]))
            chi2 += chi2_epochs[j]
            n_pixels += int(work.n_good)
            n_local += 0 if local is None else int(local.shape[1])
            if not jacobian:
                continue
            pair = self._gather(stack, work, static[0], corner[owners], 2)
            every = pair[:, :, 0] + fraction[owners][:, None] * (pair[:, :, 1] - pair[:, :, 0])
            glob = np.zeros((residual.size, n_par))
            for p, par in enumerate(self.layout.parameters):
                if par.kind == "light":
                    if par.group == self.group[j]:
                        glob[:, p] = value[par.stars[0]] - value[0]
                    continue
                for s in par.stars:
                    row = slots.get((p, s))
                    if row is not None:
                        glob[:, p] += light[s] * every[row]
            glob *= scale
            if local is None:
                reduced = glob
            else:
                solved = np.linalg.lstsq(local.T @ local, local.T @ glob, rcond=None)[0]
                reduced = glob - local @ solved
                sensitivity[j, free] = -solved[: int(free.sum())]
            schur += reduced.T @ reduced
            gradient += -2.0 * (glob.T @ residual)
            # With noise correlated along the pixel index the scatter of the gradient is
            # that of the correlated noise, and the covariance is the sandwich through
            # the curvature (docs/math.md 10.4 and 10.7).
            correlation = self.correlation[j]
            meat += reduced.T @ (_ar1_apply(reduced, correlation) if correlation else reduced)
        if not jacobian:
            self._epochs_at = (phi.tobytes(), self.shift_ref, self.scale_ref, epochs_here)
        prior, prior_gradient, prior_hessian = self._prior(phi)
        out = {
            "objective": chi2 + prior,
            "chi2": chi2,
            "chi2_epochs": chi2_epochs,
            "dof_epochs": dof_epochs,
            "residuals": kept,
            "n_pixels": n_pixels,
            "n_local": n_local,
            "shifts": shifts,
            "scales": scales,
            "state": state,
        }
        if jacobian:
            out["gradient"] = gradient + prior_gradient
            out["hessian"] = 2.0 * schur + prior_hessian
            out["curvature"] = schur
            out["scatter"] = meat
            out["prior_precision"] = 0.5 * prior_hessian
            out["sensitivity"] = sensitivity
        return out

    # The interface of albireo.match._levenberg_marquardt.

    def value(self, phi, problem=None, stats=None):
        out = self.evaluate(phi, jacobian=False)
        return out["objective"], out["chi2"]

    def everything(self, phi, problem=None, stats=None):
        out = self.evaluate(phi, jacobian=True)
        # An accepted point: the next trial starts each epoch from its minimum here.
        self.shift_ref, self.scale_ref = out["shifts"], out["scales"]
        return out["objective"], out["chi2"], out["gradient"], out["hessian"]


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateFit:
    """Templates fitted to a dataset, with the velocities measured against them.

    Attributes
    ----------
    names
        The stars, in the order of the templates and of the table's components.
    labels
        Per star: ``teff``, ``logg``, ``mh`` and ``vsini`` as rendered: fitted or held, and
        moved to the edge of the library box of the star's temperature where they lay
        outside it (``values`` holds the fitted parameters as the optimiser left them).
        ``nan`` for a label that is not an axis of the star's library.
    errors
        Per star and label: the standard error, ``nan`` for a label that was held, that
        ended on a bound, or that the data do not constrain. With ``errors="profiled"``
        the curvature errors are rescaled by the reduced chi-square.
    light, light_errors
        Per instrument: the light fractions, in the order of ``names``, and their errors
        (zero where the fractions were held).
    amplitudes, amplitude_errors
        Per star: the amplitudes of its deviations, in the order declared.
    parameter_names, values, covariance
        The fitted parameters, their values and their covariance, with the velocities of
        every epoch marginalised. A shared metallicity is the single parameter ``"mh"``.
    priors, bounds
        The Gaussian priors ``(mean, sigma)`` and the bounds ``(lower, upper)`` of the
        fitted parameters, by name.
    at_bounds
        The parameters that ended on a bound.
    prior_fraction
        Per parameter with a Gaussian prior: the ratio of its posterior to its prior
        variance. A value near one marks a label the epochs did not constrain.
    chi2, n_pixels, n_parameters
        The chi-square of the epochs that entered the fit at the fitted parameters, the
        number of their weighted pixels, and the number of parameters, those of the
        epochs included.
    passes
        One record per alternation: the status and counts of the optimiser, the
        chi-square, the number of epochs used, the largest change of a velocity, in
        units of its error, at the measurement that followed, and whether the pass
        continued from the solution with the two stars exchanged (``stars_exchanged``).
        The counts of such a pass are those of both of its fits.
    converged
        Whether the velocities of the last measurement agreed with those of the one
        before to within ``tolerance``.
    templates
        The fitted :class:`~albireo.todcor.Template` of every star.
    table
        The velocities measured with the fitted templates, with the sensitivities
        recorded (``VelocityTable.sensitivity``). For two components the pair of an
        epoch is in the order of the starting table where the epoch does not decide the
        order, and in the order of the lower chi-square where it does or where the
        starting table has no pair. After an exchange of the two stars, which ``notes``
        records, the rows are exchanged against the starting table.
    start
        The velocity table the fit started from.
    notes
        Conditions worth reading before the result is used.
    seconds
        Wall-clock time of the fit.
    """

    names: tuple[str, ...]
    labels: dict
    errors: dict
    light: dict
    light_errors: dict
    amplitudes: dict
    amplitude_errors: dict
    parameter_names: tuple[str, ...]
    values: np.ndarray
    covariance: np.ndarray
    priors: dict
    bounds: dict
    at_bounds: tuple[str, ...]
    prior_fraction: dict
    chi2: float
    n_pixels: int
    n_parameters: int
    passes: tuple
    converged: bool
    templates: tuple
    table: VelocityTable
    start: VelocityTable = field(repr=False)
    notes: tuple[str, ...] = ()
    seconds: float = 0.0

    @property
    def reduced_chi2(self) -> float:
        """``chi2 / (n_pixels - n_parameters)`` of the epochs that entered the fit."""
        return float(self.chi2 / max(self.n_pixels - self.n_parameters, 1))

    def template(self, name: str) -> Template:
        """The fitted template of one star."""
        if name not in self.names:
            raise KeyError(f"no star {name!r}; the fit has {list(self.names)}")
        return self.templates[self.names.index(name)]

    def summary(self) -> str:
        """A text report: labels with errors, light fractions, and the alternation."""
        lines = [
            f"Templates fitted to the epochs: {len(self.names)} star(s), "
            f"{len(self.parameter_names)} parameters, reduced chi-square "
            f"{self.reduced_chi2:.3f} over {self.n_pixels} pixels; "
            f"{len(self.passes)} pass(es), "
            + ("velocities converged" if self.converged else "velocities NOT converged")
        ]
        units = {"teff": "K", "logg": "dex", "mh": "dex", "vsini": "km/s"}
        for name in self.names:
            parts = []
            for label in (*_LABELS, "vsini"):
                value, error = self.labels[name][label], self.errors[name][label]
                if not np.isfinite(value):
                    continue  # not an axis of this star's library
                text = f"{label} {value:.4g}"
                text += f" +- {error:.2g}" if np.isfinite(error) else " (held or unconstrained)"
                parts.append(f"{text} {units[label]}")
            lines.append(f"  {name}: " + ", ".join(parts))
            for k, amplitude in enumerate(self.amplitudes.get(name, ())):
                error = self.amplitude_errors[name][k]
                lines.append(f"    deviation {k}: {amplitude:.4g} +- {error:.2g}")
        for instrument, fractions in self.light.items():
            errors = self.light_errors[instrument]
            lines.append(
                f"  light fractions ({instrument}): "
                + ", ".join(
                    f"{name} {value:.4f} +- {error:.4f}"
                    for name, value, error in zip(self.names, fractions, errors, strict=True)
                )
            )
        if self.at_bounds:
            lines.append("  on a bound: " + ", ".join(self.at_bounds))
        weak = [key for key, ratio in self.prior_fraction.items() if ratio > 0.8]
        if weak:
            lines.append("  set by the prior, not by the epochs: " + ", ".join(weak))
        lines.extend(f"  ! {note}" for note in self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def _held(spec, available: tuple[float, float], where: str) -> float:
    """The value of a held label, which must lie in the range its library covers."""
    value = float(np.asarray(spec.start(), dtype=np.float64))
    if not available[0] <= value <= available[1]:
        raise ValueError(
            f"{where} is held at {value:g}, outside the range the library covers "
            f"({available[0]:g} to {available[1]:g}); a template cannot be rendered there"
        )
    return value


def _libraries_of(declared) -> tuple:
    """The library boxes a star declares: one library, a sequence, or a set of boxes."""
    boxes = getattr(declared, "libraries", None)
    if boxes is not None:
        return tuple(boxes)
    if isinstance(declared, (list, tuple)):
        return tuple(declared)
    return (declared,)


def _table_shifts(table: VelocityTable, grid: LogGrid, dataset: Dataset) -> np.ndarray:
    """The shifts of a table of absolute templates in template pixels, ``(n_stars, n_epochs)``."""
    velocity = np.asarray(table.velocity, dtype=np.float64)
    finite = np.where(np.isfinite(velocity), velocity, 0.0)
    # In blocks of epochs: outside jit the conversion is compiled once per array length.
    pixels = np.stack([_in_epoch_blocks(grid.velocity_to_pixels, row) for row in finite])
    if dataset.frame == "topocentric":
        pixels = pixels - _bary_pixels(grid, dataset)[None, :]
    return pixels


def _bary_pixels(grid: LogGrid, dataset: Dataset) -> np.ndarray:
    """The barycentric velocity of every epoch as a shift in template pixels."""
    return np.asarray(_in_epoch_blocks(grid.velocity_to_pixels, dataset.v_bary), dtype=np.float64)


def _nearer_exchanged(table: VelocityTable, reference: VelocityTable) -> np.ndarray:
    """Per epoch: the pair of ``table`` is nearer the velocities of ``reference`` in its
    other order than in the one it has.

    The distance of an order is the largest difference of a velocity in units of its error.
    False at every epoch for a table of one component or of more than two.
    """
    if table.n_components != 2:
        return np.zeros(table.n_epochs, dtype=bool)
    new, old, sigma = table.velocity, reference.velocity, table.sigma
    with np.errstate(invalid="ignore", divide="ignore"):
        direct = np.max(np.abs(new - old) / sigma, axis=0)
        other = np.max(np.abs(new[::-1] - old) / sigma[::-1], axis=0)
        return np.isfinite(direct) & np.isfinite(other) & (other < direct)


def _covariance(final: dict, rescale: float, excluded: np.ndarray) -> np.ndarray:
    """The covariance of the fitted parameters, ``nan`` for the excluded ones.

    With ``S`` the reduced curvature of the data, ``M`` the scatter of its gradient
    (``S`` itself for white noise, and ``sum_j G_j^T R_j G_j`` over the reduced Jacobians
    with the correlation ``R_j`` of the noise otherwise), ``L`` the precision of the
    priors and ``r`` the reduced chi-square, the covariance is the sandwich
    ``H^-1 (M / r + L) H^-1`` with ``H = S / r + L`` (``docs/math.md`` §10.7).
    """
    n = excluded.size
    covariance = np.full((n, n), np.nan)
    free = np.flatnonzero(~excluded)
    if free.size == 0:
        return covariance
    pick = np.ix_(free, free)
    curvature = final["curvature"][pick] / rescale + final["prior_precision"][pick]
    scatter = final["scatter"][pick] / rescale + final["prior_precision"][pick]
    try:
        inverse = np.linalg.inv(curvature)
    except np.linalg.LinAlgError:
        return covariance
    block = inverse @ scatter @ inverse
    good = np.isfinite(np.diag(block)) & (np.diag(block) > 0.0)
    block[~good, :] = np.nan
    block[:, ~good] = np.nan
    covariance[pick] = 0.5 * (block + block.T)
    return covariance


def _velocity_change(before: VelocityTable, after: VelocityTable) -> float:
    """The largest change of a velocity between two tables, in units of its error.

    Taken over the epochs usable in both. For two components it is taken in the order of
    the components as written. The table measured after a pass has every pair that its
    epoch leaves open in the order of the table before (``in_order`` in
    :func:`fit_templates`), so a pair in another order is one the chi-square decided
    against the order the pass was fitted with, and the alternation has not converged.
    For another number of components, whose order nothing keeps, each epoch is taken in
    the order that changes least. Infinite where no epoch is usable in both.
    """
    both = before.good & after.good
    if not both.any():
        return float("inf")
    old, new, sigma = before.velocity[:, both], after.velocity[:, both], after.sigma[:, both]
    orders = (
        [tuple(range(old.shape[0]))]
        if old.shape[0] == 2
        else itertools.permutations(range(old.shape[0]))
    )
    worst = np.full(old.shape[1], np.inf)
    for order in orders:
        with np.errstate(invalid="ignore", divide="ignore"):
            moved = np.max(np.abs(new[list(order)] - old) / sigma[list(order)], axis=0)
        worst = np.fmin(worst, moved)
    return float(np.max(worst))


def _scatter_ratio(difference: np.ndarray, phi: float) -> float:
    """``d^T R d / d^T d`` for the AR(1) correlation ``R`` of lag-one coefficient ``phi``.

    The difference of the chi-squares of two models scatters with the variance of
    ``2 d . n``, with ``d`` the difference of the two weighted models and ``n`` the
    standardized noise. This is the ratio of that variance under the correlated noise to
    the one under white noise: one at ``phi = 0``, and up to ``(1 + phi) / (1 - phi)``
    for a difference that varies slowly along the pixels.
    """
    norm = float(difference @ difference)
    if not phi or not norm > 0.0:
        return 1.0
    return float(difference @ _ar1_apply(difference, float(phi))) / norm


def _light_key(name: str, group: str, groups: Sequence[str]) -> str:
    """The name of the fitted light fraction of a star, per instrument where there are several."""
    return f"light_{name}" if len(groups) == 1 else f"light_{name}[{group}]"


def _measured(table: VelocityTable) -> np.ndarray:
    """Per epoch: every velocity of the table is finite and none is at the edge of its range."""
    return np.all(np.isfinite(table.velocity), axis=0) & ~np.any(table.at_edge, axis=0)


def _instrument_light(table: VelocityTable, groups: Sequence[str]) -> np.ndarray:
    """Starting light fractions per instrument from a table: the medians of its amplitudes.

    The amplitudes are read from the table's rows, which an orbit has exchanged with the
    velocities where it exchanged the pair, and not from the fractions the correlation
    recorded in its settings, which are in the order of the templates.
    """
    n_star = table.n_components
    out = np.full((len(groups), n_star), 1.0 / n_star)
    light = np.asarray(table.light, dtype=np.float64)
    for g, instrument in enumerate(groups):
        here = np.array([name == instrument for name in table.instrument])
        with np.errstate(invalid="ignore"):
            usable = here & np.all(np.isfinite(light) & (light > 0.0), axis=0)
        if not usable.any():
            continue
        fractions = np.median(light[:, usable] / light[:, usable].sum(axis=0), axis=1)
        if np.all(np.isfinite(fractions)) and fractions.sum() > 0.0:
            out[g] = np.clip(fractions / fractions.sum(), 0.02, 0.98)
            out[g] /= out[g].sum()
    return out


def fit_templates(
    dataset: Dataset,
    stars: Mapping[str, Any],
    *,
    grid: LogGrid,
    medium: str,
    mh=None,
    light="global",
    deviations: Mapping[str, Sequence[TemplateDeviation]] | None = None,
    epsilon: float | Mapping[str, float] = 0.6,
    start: VelocityTable | None = None,
    exchange_stars: bool = True,
    v_range=(-300.0, 300.0),
    lsf_sigma_v: Mapping[str, Any] | None = None,
    lsf_anchors_angstrom: Mapping[str, Sequence[float]] | None = None,
    nuisance_order: int | None = 0,
    errors: str = "profiled",
    scale: str = "fixed",
    noise_correlation: float | Mapping[str, float] | None = None,
    max_passes: int = 4,
    max_iterations: int = 30,
    tolerance: float = 0.1,
    progress: bool = False,
) -> TemplateFit:
    """Fit library templates to a dataset together with the velocities of its epochs.

    Parameters
    ----------
    dataset
        The epochs, as :func:`albireo.todcor` takes them. Epochs at which the light
        fractions differ from the rest, such as those in eclipse, are left out by the
        caller and measured afterwards with the fitted templates.
    stars
        Mapping of component name to :class:`albireo.StarLabels`, in the order of the
        components. ``library`` is one :class:`~albireo.SpectralLibrary`, a sequence of
        library boxes that adjoin in temperature, or an object with a ``libraries``
        attribute (:class:`albireo.eclipsing.LibrarySet`). ``teff``, ``logg`` and
        ``vsini`` take a float (held), :class:`~albireo.facade.Between` (a range) or
        :class:`~albireo.facade.Known` (a Gaussian prior, the classification of the star
        with its error). The starting template is rendered at the starting values, so
        they should be those of a classification: ``None`` adopts the library's range
        and starts at its middle. ``v_kms`` must be unset, since the velocity of every
        epoch is fitted.
    grid, medium
        The template grid (:meth:`LogGrid.covering`, with the margin of ``v_range``) and
        the wavelength scale of the data, ``"air"`` or ``"vacuum"``.
    mh
        The metallicity: one specification shared by the stars (the default, over the
        range the libraries cover), or a mapping of name to specification.
    light
        ``"global"`` (default) fits one set of light fractions per instrument, started
        at the fractions the correlation measures. A sequence or mapping of fractions
        summing to one holds them.
    deviations
        Mapping of component name to :class:`TemplateDeviation` instances.
    epsilon
        The linear limb-darkening coefficient of the rotation profile, one value or one
        per star.
    start
        A velocity table of these epochs to start from, with its components in the
        order of ``stars``. Default: :func:`albireo.todcor` with the starting templates.
        Two stars declared alike give a table in which the order of the pair at each
        epoch is arbitrary. Such a table is put in order first
        (:func:`albireo.rvorbit.assign_components`) and passed here.
    exchange_stars
        What names the two stars where the starting table and the declared priors
        disagree. A table with the stars in the other order at every epoch leads to the
        solution with their labels and light fractions exchanged, which has the same
        chi-square as its image under the exchange of the two stars: the epochs do not
        tell rows in the other order from priors given to the other star. With ``True``
        (default) the priors name the stars: where their sum is lower at the image by
        more than 9, the image is fitted, with the rows of the table exchanged, and kept
        if its objective is the lower, which it is unless a label or a light fraction
        held to its star says otherwise. With ``False`` the starting table names them,
        and ``notes`` reports a label that ended far from its prior. The exchange needs
        two stars with the same parameters fitted, from the same library boxes.
    v_range, lsf_sigma_v, lsf_anchors_angstrom, nuisance_order, errors, scale, noise_correlation
        As :func:`albireo.todcor`. With ``scale="free"`` the scale of every epoch's
        composite is one of that epoch's own parameters. A declared noise correlation
        widens the errors of the fitted parameters as it does those of the velocities:
        their covariance is then the sandwich through the reduced curvature.
    max_passes
        The largest number of alternations of the two steps.
    max_iterations
        The iteration cap of the Levenberg-Marquardt of each pass.
    tolerance
        The alternation stops when no velocity changed by more than this fraction of its
        error between two measurements, over the epochs usable in both. A pair measured
        in the other order than the pass was fitted with is such a change.
    progress
        Print one line per pass.

    Returns
    -------
    TemplateFit

    Raises
    ------
    ValueError
        If fewer than two epochs have a measured pair of velocities to start from.

    Notes
    -----
    The epochs that enter the fit are the usable ones of the current table
    (``VelocityTable.good``), or every epoch with measured velocities where fewer than
    three are usable. The light fractions and the line strengths are separately
    measurable because the data are normalized: the fractions are held to sum to one, so
    a change of a fraction cannot be absorbed by the metallicity of both stars at once.

    The errors are the curvature errors of the reduced objective. They include the
    uncertainty of every epoch's velocities and exclude the error of the library itself.
    The minimum of an epoch is the box-constrained minimum of a cell of the template
    grid's pixels, as in the correlation (``docs/math.md`` §10.3). Where it lies on the
    face of a cell the chi-square has a kink, the pixel-locking of the linear shift
    operator: the shift does not respond to a small change of the templates, its
    sensitivity is zero, and the reduced objective is continuous there with a gradient
    that is not. The optimiser can therefore end with the status ``"stalled"`` at a point
    that is a minimum to within that ripple.

    A rotation below half a pixel of the template grid does not change the template. Such
    a rotation has no error, and a fit started there leaves only if the data ask for a
    rotation above that limit.

    References
    ----------
    Golub, G. H. & Pereyra, V. 1973, SIAM Journal on Numerical Analysis, 10, 413
    Nachmani, G., Faigler, S. & Mazeh, T. 2026, arXiv:2602.05610
    """
    from albireo.facade import Between
    from albireo.rvorbit import _exchanged

    t_start = time.perf_counter()
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be an albireo Dataset")
    if medium not in SUPPORTED_MEDIA:
        raise ValueError(f"medium must be one of {SUPPORTED_MEDIA}, got {medium!r}")
    if errors not in ("profiled", "ivar"):
        raise ValueError(f"errors must be 'profiled' or 'ivar'; got {errors!r}")
    if scale not in ("fixed", "free"):
        raise ValueError(f"scale must be 'fixed' or 'free'; got {scale!r}")
    names = tuple(str(name) for name in stars)
    if not names:
        raise ValueError("stars is empty; declare at least one component")
    n_star = len(names)
    groups = tuple(dict.fromkeys(epoch.instrument for epoch in dataset))
    deviations = dict(deviations or {})
    unknown = sorted(set(deviations) - set(names))
    if unknown:
        raise ValueError(f"deviations name star(s) {unknown} that are not declared")
    shared_mh = not isinstance(mh, Mapping)
    if not shared_mh:
        unknown = sorted(set(mh) - set(names))
        if unknown:
            raise ValueError(f"mh names star(s) {unknown} that are not declared")
    if isinstance(epsilon, Mapping):
        missing = [name for name in names if name not in epsilon]
        if missing:
            raise ValueError(f"epsilon has no entry for {missing}")
    reserved = {*_LABELS, "vsini", "light"}
    for name, declared_deviations in deviations.items():
        taken = [d.name for d in declared_deviations if d.name in reserved]
        if taken:
            raise ValueError(
                f"deviation(s) of {name!r} are named {taken}, which are names of labels"
            )

    # -- the stars, their label spaces and the layout of the parameters --------
    layout = _Layout()
    held = _State(
        labels=[{} for _ in names],
        amplitudes=[np.zeros(len(deviations.get(name, ()))) for name in names],
        light=np.full((len(groups), n_star), 1.0 / n_star),
    )
    models: list[_Star] = []
    spans = []
    # Stars that declare the same library boxes share one label space, and so one
    # projection and one interpolator of each box. The kept copies of a library
    # (SpectralLibrary.resampled_to) skip a writable library and an optical grid, and each
    # copy of the FGK box on a grid of 20,000 pixels is 146 MB.
    spaces: dict[tuple[int, ...], _LabelSpace] = {}
    for i, name in enumerate(names):
        declared = stars[name]
        if getattr(declared, "v_kms", None) is not None:
            raise ValueError(
                f"star {name!r}: v_kms is a label of a disentangled component's frame; the "
                "template fit measures a velocity at every epoch and takes none"
            )
        boxes = _libraries_of(declared.library)
        shared = tuple(sorted(id(box) for box in boxes))
        space = spaces.get(shared)
        if space is None:
            space = spaces[shared] = _LabelSpace(boxes, grid, medium, f"star {name!r}")
        teff_range = space.range("teff", 0.0, 0.0)
        teff_spec = _declared_spec(declared.teff, f"teff of {name!r}", Between(*teff_range))
        if _is_fixed(teff_spec):
            value = _held(teff_spec, teff_range, f"teff of {name!r}")
            span = (value, value)
        else:
            value, lower, upper, prior = _prior_of(teff_spec, f"teff of {name!r}")
            lower, upper = max(lower, teff_range[0]), min(upper, teff_range[1])
            if prior is not None:
                lower = max(lower, prior[0] - 5.0 * prior[1])
                upper = min(upper, prior[0] + 5.0 * prior[1])
            span = (lower, upper)
            layout.add(
                _Parameter(f"teff_{name}", "label", (i,), "teff"), value, lower, upper, prior
            )
        held.labels[i]["teff"] = float(np.clip(value, *teff_range))
        spans.append(span)

        logg_range = space.range("logg", *span)
        if logg_range is None:
            if declared.logg is not None:
                raise ValueError(
                    f"logg of {name!r} is declared, and the library of that star has no logg axis"
                )
            held.labels[i]["logg"] = float("nan")
        else:
            spec = _declared_spec(declared.logg, f"logg of {name!r}", Between(*logg_range))
            if _is_fixed(spec):
                held.labels[i]["logg"] = _held(spec, logg_range, f"logg of {name!r}")
            else:
                value, lower, upper, prior = _prior_of(spec, f"logg of {name!r}")
                lower, upper = max(lower, logg_range[0]), min(upper, logg_range[1])
                layout.add(
                    _Parameter(f"logg_{name}", "label", (i,), "logg"), value, lower, upper, prior
                )
                held.labels[i]["logg"] = float(np.clip(value, lower, upper))

        spec = _declared_spec(declared.vsini, f"vsini of {name!r}", Between(0.0, _VSINI_MAX_KMS))
        if _is_fixed(spec):
            held.labels[i]["vsini"] = float(np.asarray(spec.start()))
        else:
            value, lower, upper, prior = _prior_of(spec, f"vsini of {name!r}")
            lower, upper = max(lower, 0.0), min(upper, _VSINI_MAX_KMS)
            layout.add(
                _Parameter(f"vsini_{name}", "label", (i,), "vsini"), value, lower, upper, prior
            )
            held.labels[i]["vsini"] = float(np.clip(value, lower, upper))
        if held.labels[i]["vsini"] < 0.0:
            raise ValueError(f"vsini of {name!r} must be non-negative")

        vectors = []
        for k, deviation in enumerate(deviations.get(name, ())):
            vector = np.asarray(deviation.vector, dtype=np.float64)
            if vector.shape != (grid.n,):
                raise ValueError(
                    f"deviation {deviation.name!r} of {name!r} has shape {vector.shape}; the "
                    f"template grid has {grid.n} pixels"
                )
            vectors.append(vector)
            key = f"{deviation.name}_{name}"
            if deviation.amplitude is None:
                layout.add(
                    _Parameter(key, "deviation", (i,), deviation.name, index=k),
                    0.0,
                    -np.inf,
                    np.inf,
                    None,
                )
                continue
            spec = _declared_spec(deviation.amplitude, f"amplitude {key!r}")
            if _is_fixed(spec):
                held.amplitudes[i][k] = float(np.asarray(spec.start()))
            else:
                value, lower, upper, prior = _prior_of(spec, f"amplitude {key!r}")
                layout.add(
                    _Parameter(key, "deviation", (i,), deviation.name, index=k),
                    value,
                    lower,
                    upper,
                    prior,
                )
                held.amplitudes[i][k] = value
        eps = epsilon[name] if isinstance(epsilon, Mapping) else epsilon
        models.append(
            _Star(name, space, grid, eps, float(getattr(declared, "macro_kms", 0.0)), vectors)
        )

    # The metallicity: one parameter for every star, or one each. A star whose library has
    # no mh axis takes no part in it, and its mh is nan.
    mh_ranges = [models[i].space.range("mh", *spans[i]) for i in range(n_star)]
    blocks = [tuple(range(n_star))] if shared_mh else [(i,) for i in range(n_star)]
    mh_stars: set[int] = set()
    for block in blocks:
        key = "mh" if shared_mh else f"mh_{names[block[0]]}"
        declared_mh = mh if shared_mh else mh.get(names[block[0]])
        for i in block:
            held.labels[i]["mh"] = float("nan")
        having = tuple(i for i in block if mh_ranges[i] is not None)
        if not having:
            if declared_mh is not None:
                raise ValueError(f"{key} is declared, and no library of its star(s) has an mh axis")
            continue
        covered = [r for r in (mh_ranges[i] for i in having) if r is not None]
        available = (max(lo for lo, _ in covered), min(hi for _, hi in covered))
        spec = _declared_spec(declared_mh, key, Between(*available))
        if _is_fixed(spec):
            for i in having:
                held.labels[i]["mh"] = _held(spec, available, key)
            continue
        value, lower, upper, prior = _prior_of(spec, key)
        lower, upper = max(lower, available[0]), min(upper, available[1])
        layout.add(_Parameter(key, "label", having, "mh"), value, lower, upper, prior)
        mh_stars.update(having)
        for i in having:
            held.labels[i]["mh"] = float(np.clip(value, lower, upper))

    fit_light = isinstance(light, str)
    if fit_light and light != "global":
        raise ValueError(
            f"light must be 'global' or fractions summing to one; got {light!r}. Amplitudes "
            "free at every epoch are measured afterwards with the fitted templates "
            "(todcor(..., light='free'))"
        )
    if fit_light and n_star > 2:
        raise ValueError(
            "light='global' fits the fractions of one or two stars. With three or more, "
            "bounds on each fraction do not keep the first, which is one minus the others, "
            "positive: hold the fractions (light=<fractions>)"
        )
    if not fit_light:
        _, fixed_light = _resolve_light(light, names)
        held.light[:] = fixed_light[None, :]

    ranges = _resolve_ranges(v_range, n_star)
    phi_of = _resolve_correlation(noise_correlation, dataset)
    options = {
        "v_range": v_range,
        "lsf_sigma_v": lsf_sigma_v,
        "lsf_anchors_angstrom": lsf_anchors_angstrom,
        "nuisance_order": nuisance_order,
        "errors": errors,
        "scale": scale,
        "noise_correlation": noise_correlation,
    }
    notes: list[str] = []

    def templates_of(state: _State) -> list[Template]:
        out = []
        for i, model in enumerate(models):
            meta = {
                "source": "template fit",
                "labels": {
                    label: value
                    for label, value in model.space.clamped(state.labels[i]).items()
                    if label in _LABELS
                },
                "vsini_kms": float(state.labels[i]["vsini"]),
                "medium": medium,
            }
            out.append(
                model.template(model.render(state.labels[i], state.amplitudes[i]), meta=meta)
            )
        return out

    def measure(state: _State) -> VelocityTable:
        """The velocities of every epoch against the templates of ``state``."""
        if n_star == 1 or not fit_light:
            fractions = tuple(float(x) for x in state.light[0])
            return todcor(dataset, templates_of(state), light=fractions, **options)
        amplitudes = {group: state.light[g] for g, group in enumerate(groups)}
        table = _run(
            dataset,
            templates_of(state),
            grid,
            ranges,
            "global",
            amplitudes,
            lsf_sigma_v,
            lsf_anchors_angstrom,
            nuisance_order,
            None,
            errors,
            scale,
            phi_of,
            False,
        )
        return replace(table, light_mode="fitted with the templates")

    # -- the starting table -----------------------------------------------------
    state = layout.unpack(layout.start, held)
    if start is None:
        if fit_light and n_star > 1:
            try:
                start = todcor(dataset, templates_of(state), light="global", **options)
            except _LightNotMeasured:
                start = todcor(dataset, templates_of(state), light="free", **options)
                notes.append(
                    "the starting templates gave no epoch with positive amplitudes; the fit "
                    "started from equal light fractions"
                )
        else:
            start = measure(state)
    elif tuple(start.names) != names or start.n_epochs != dataset.n_epochs:
        raise ValueError(
            f"start has the components {list(start.names)} over {start.n_epochs} epochs; the "
            f"fit has {list(names)} over {dataset.n_epochs}"
        )
    if fit_light:
        held.light[:] = _instrument_light(start, groups)
        for g, group in enumerate(groups):
            for i in range(1, n_star):
                layout.add(
                    _Parameter(
                        _light_key(names[i], group, groups), "light", (i,), "light", group=g
                    ),
                    held.light[g, i],
                    *_LIGHT_BOUNDS,
                    None,
                )
    lower = np.asarray(layout.lower, dtype=np.float64)
    upper = np.asarray(layout.upper, dtype=np.float64)
    phi = np.asarray(layout.start, dtype=np.float64)
    if layout.n == 0:
        raise ValueError("every parameter is held; there is nothing to fit")

    # The work of every epoch, prepared once and shared by the objectives of the passes,
    # of the exchanged image, of the order probes and of the final sensitivities.
    prepared: dict[int, tuple] = {}

    def prepared_for(epochs: np.ndarray) -> list[tuple]:
        for j in (int(j) for j in epochs):
            if j not in prepared:
                prepared[j] = _prepared_epoch(j, dataset.epochs[j], grid, nuisance_order)
        return [prepared[int(j)] for j in epochs]

    def objective_for(table: VelocityTable, epochs: np.ndarray, *, move: bool) -> _Objective:
        subset = Dataset(epochs=tuple(dataset.epochs[j] for j in epochs), frame=dataset.frame)
        shifts = _table_shifts(table, grid, dataset)[:, epochs]
        scales = np.ones(epochs.size)
        if scale == "free":
            total = np.nansum(np.asarray(table.light, dtype=np.float64)[:, epochs], axis=0)
            scales = np.where(np.isfinite(total) & (total > 0.0), total, 1.0)
        return _Objective(
            subset,
            models,
            layout,
            held,
            groups=groups,
            shifts=shifts,
            scales=scales,
            correlation=phi_of,
            lsf_sigma_v=lsf_sigma_v,
            lsf_anchors_angstrom=lsf_anchors_angstrom,
            nuisance_order=nuisance_order,
            free_scale=scale == "free",
            move=move,
            prepared=prepared_for(epochs),
        )

    def minimise(objective: _Objective, phi0):
        return _levenberg_marquardt(
            objective,
            None,
            None,
            phi0,
            lower,
            upper,
            kind="templates",
            max_iter=int(max_iterations),
            max_eval=6 * int(max_iterations),
        )

    def in_order(measured: VelocityTable, before: VelocityTable, phi0) -> VelocityTable:
        """``measured`` with the pair in the order of ``before`` where an epoch leaves it open.

        The correlation returns each pair in the order of the lower chi-square. For two
        distinct stars that order is the measurement. For two alike stars the two orders
        are equally deep and the noise decides between them, epoch by epoch, while the
        fit of the next pass reads each star's shift from its own row. An epoch is
        therefore exchanged where its other order is nearer the velocities of ``before``
        and the chi-square with the two velocities exchanged between the templates of
        ``phi0`` is above the one returned by less than the threshold of the blend flag
        (:func:`albireo.todcor._blend_threshold`): 9 times the reduced chi-square, or 9
        with the weights as given. Under a declared noise correlation the threshold is
        multiplied by the factor by which the correlation widens the scatter of that
        difference of chi-squares (:func:`_scatter_ratio`), without which the noise
        decides the order of two identical stars at one epoch in ten at a lag-one
        coefficient of 0.8.

        The chi-square of the other order is taken at the exchanged velocities
        themselves, which is the pair the exchange writes into the table. A minimum
        followed from there is another pair: at a blended epoch of two distinct stars it
        is the minimum returned, within the threshold of itself, and the exchange would
        give the brighter star the velocity of the fainter. The chi-square recorded for
        an exchanged epoch stays that of the pair returned. An epoch that ``before``
        does not have a pair for is left as returned.
        """
        nearer = np.flatnonzero(_nearer_exchanged(measured, before))
        if nearer.size == 0:
            return measured
        probe = objective_for(measured, nearer, move=False)
        returned = probe.evaluate(phi0, jacobian=False, residuals=True)
        probe.shift_ref = probe.shift_ref[:, ::-1].copy()
        other = probe.evaluate(phi0, jacobian=False, residuals=True)
        limit = np.array(
            [
                _blend_slack(float(chi2), _blend_threshold(max(int(dof), 1), errors))
                * _scatter_ratio(there - here, probe.correlation[j])
                for j, (chi2, dof, here, there) in enumerate(
                    zip(
                        returned["chi2_epochs"],
                        returned["dof_epochs"],
                        returned["residuals"],
                        other["residuals"],
                        strict=True,
                    )
                )
            ]
        )
        open_ = np.zeros(measured.n_epochs, dtype=bool)
        open_[nearer] = other["chi2_epochs"] - returned["chi2_epochs"] < limit
        if not open_.any():
            return measured
        settings = {**dict(measured.settings), "exchanged_to_keep_the_order": int(open_.sum())}
        return replace(_exchanged(measured, open_), settings=settings)

    symmetry = _star_exchange(layout, models) if exchange_stars else None

    def image_of(phi0, objective: _Objective, pairs):
        """``phi0`` with the two stars exchanged, where the priors decide for it.

        The image is moved into the declared ranges. ``None`` where the exchange is not
        defined (``pairs`` of :func:`_star_exchange`) or not asked for, and where the sum
        of the priors is not lower at the image by more than 9, the rise in chi-square
        that decides between two minima of an epoch (``albireo.todcor._BLEND_CHI2``).
        Two stars declared alike keep the order of the starting table.
        """
        if pairs is None:
            return None
        partner, complement = pairs
        image = np.array(phi0, dtype=np.float64)[partner]
        image[complement] = 1.0 - image[complement]
        image = np.clip(image, lower, upper)
        here, there = objective._prior(phi0)[0], objective._prior(image)[0]
        return image if here - there > _BLEND_CHI2 else None

    # -- the alternation ----------------------------------------------------------
    table = start
    passes = []
    converged = False
    run = None
    objective = None
    for number in range(max(int(max_passes), 1)):
        measured = _measured(table)
        use = np.flatnonzero(table.good)
        if use.size < 3:
            use = np.flatnonzero(measured)
        if use.size < _MIN_EPOCHS:
            raise ValueError(
                f"{int(measured.sum())} epoch(s) have a measured pair of velocities; the "
                f"template fit needs at least {_MIN_EPOCHS}. The starting templates do not "
                "describe the components: declare labels closer to them"
            )
        objective = objective_for(table, use, move=True)
        run = minimise(objective, phi)
        iterations, evaluations = int(run.iterations), int(objective.evaluations)
        # A table that has the two stars in the other order at its epochs, as a correlation
        # with alike templates returns it, leads to the solution with the labels and the
        # light fractions of the two exchanged. Its image under the exchange has the same
        # chi-square, to within what the stars differ in beside their fitted parameters,
        # and the priors of the stars as declared. Where those decide for the image it is
        # fitted, and the fit continues there if its objective is the lower. For two stars
        # with nothing held to either the chi-squares are equal and the priors alone
        # decide: the epochs cannot tell rows in the other order from priors given to the
        # other star. An image that fits worse differs in something held, which no later
        # pass changes, so it is not tried again.
        image = image_of(run.phi, objective, symmetry)
        stars_exchanged = False
        if image is not None:
            other_table = _exchanged(table, np.ones(table.n_epochs, dtype=bool))
            other_objective = objective_for(other_table, use, move=True)
            other_run = minimise(other_objective, image)
            iterations += int(other_run.iterations)
            evaluations += int(other_objective.evaluations)
            if other_run.objective < run.objective:
                table, objective, run = other_table, other_objective, other_run
                stars_exchanged = True
                notes.append(
                    f"pass {number + 1} ended at the solution with the two stars exchanged "
                    "against the declared priors; the fit continued from its image, with the "
                    "rows of the velocity table exchanged against the starting table"
                )
            else:
                symmetry = None
        phi = run.phi
        state = layout.unpack(phi, held)
        new_table = in_order(measure(state), table, phi)
        change = _velocity_change(table, new_table)
        passes.append(
            {
                "status": run.status,
                "iterations": iterations,
                "evaluations": evaluations,
                "chi2": float(run.chi2),
                "n_epochs": int(use.size),
                "velocity_change": change,
                "stars_exchanged": stars_exchanged,
            }
        )
        if progress:
            print(
                f"  pass {number + 1}: {run.status}, {iterations} iterations, chi2 "
                f"{run.chi2:.1f} over {use.size} epochs; largest change of a velocity "
                f"{change:.3g} sigma" + ("; the two stars exchanged" if stars_exchanged else "")
            )
        table = new_table
        if change < float(tolerance):
            converged = True
            break
    if not converged:
        notes.append(
            f"the velocities still changed by {passes[-1]['velocity_change']:.2g} of their "
            f"errors at the last of {len(passes)} passes"
        )

    # -- covariance at the fitted parameters ---------------------------------------
    final = objective.evaluate(phi, jacobian=True)
    n_parameters = int(final["n_local"]) + layout.n
    # With errors="profiled" the noise level is estimated from the residuals, which scales
    # the curvature of the data and leaves the priors as declared.
    rescale = (
        final["chi2"] / max(final["n_pixels"] - n_parameters, 1) if errors == "profiled" else 1.0
    )
    tol = 1e-9 * np.where(np.isfinite(upper - lower), upper - lower, 1.0)
    on_bound = (phi <= lower + tol) | (phi >= upper - tol)
    at_bounds = tuple(
        par.key for par, there in zip(layout.parameters, on_bound, strict=True) if there
    )
    # A parameter is conditioned on, and has no error, where it is on a bound, where the
    # epochs and its prior give it no curvature, and where it is a rotation below half a
    # pixel of the template grid, which the template does not depend on.
    total = np.diag(final["curvature"]) / rescale + np.diag(final["prior_precision"])
    unresolved = np.array(
        [
            par.label == "vsini" and value <= 0.5 * grid.dv_kms
            for value, par in zip(phi, layout.parameters, strict=True)
        ]
    )
    covariance = _covariance(final, rescale, on_bound | ~(total > 0.0) | unresolved)
    sigma = np.sqrt(np.where(np.diag(covariance) > 0.0, np.diag(covariance), np.nan))
    prior_fraction = {}
    for p, par in enumerate(layout.parameters):
        prior = layout.priors.get(par.key)
        if prior is not None and np.isfinite(sigma[p]):
            prior_fraction[par.key] = float(covariance[p, p] / prior[1] ** 2)
        if prior is not None and abs(phi[p] - prior[0]) > 3.0 * prior[1]:
            notes.append(
                f"{par.key} ended {abs(phi[p] - prior[0]) / prior[1]:.1f} prior widths from "
                f"its declared value ({prior[0]:g}): the declaration, or the minimum the fit "
                "reached, is in doubt"
            )
    if at_bounds:
        notes.append(
            "parameter(s) on a bound are held there for the covariance: " + ", ".join(at_bounds)
        )

    # -- sensitivities of the velocities of the final table --------------------------
    measured = np.flatnonzero(_measured(table))
    sensitivity = np.full((n_star, table.n_epochs, layout.n), np.nan)
    if measured.size:
        at_table = objective_for(table, measured, move=False).evaluate(phi, jacobian=True)
        pixels = at_table["shifts"]
        if dataset.frame == "topocentric":
            pixels = pixels + _bary_pixels(grid, dataset)[measured, None]
        for row, j in enumerate(measured):
            sensitivity[:, j, :] = (
                _dv_dpix(grid, pixels[row])[:, None] * at_table["sensitivity"][row]
            )
    table = replace(
        table,
        sensitivity=sensitivity,
        parameter_names=layout.keys,
        parameter_covariance=covariance,
    )

    # -- the record ---------------------------------------------------------------
    index = {key: p for p, key in enumerate(layout.keys)}

    def error_of(key: str) -> float:
        return float(sigma[index[key]]) if key in index else float("nan")

    labels_out, errors_out = {}, {}
    for i, name in enumerate(names):
        # As rendered: a label beyond the range of the box its star is in sits on the edge.
        rendered = models[i].space.clamped(state.labels[i])
        labels_out[name] = {label: float(rendered[label]) for label in (*_LABELS, "vsini")}
        errors_out[name] = {
            "teff": error_of(f"teff_{name}"),
            "logg": error_of(f"logg_{name}"),
            "mh": error_of("mh" if shared_mh else f"mh_{name}") if i in mh_stars else math.nan,
            "vsini": error_of(f"vsini_{name}"),
        }
    light_out, light_errors = {}, {}
    for g, group in enumerate(groups):
        light_out[group] = np.array(state.light[g])
        variance = np.zeros(n_star)
        if fit_light and n_star > 1:
            rows = [index[_light_key(names[i], group, groups)] for i in range(1, n_star)]
            block = covariance[np.ix_(rows, rows)]
            variance[1:] = np.diag(block)
            variance[0] = float(np.sum(block))  # the first fraction is one minus the others
        light_errors[group] = np.sqrt(np.where(variance >= 0.0, variance, np.nan))
    amplitudes_out, amplitude_errors = {}, {}
    for i, name in enumerate(names):
        if deviations.get(name):
            amplitudes_out[name] = np.array(state.amplitudes[i])
            amplitude_errors[name] = np.array(
                [error_of(f"{d.name}_{name}") for d in deviations[name]]
            )
    return TemplateFit(
        names=names,
        labels=labels_out,
        errors=errors_out,
        light=light_out,
        light_errors=light_errors,
        amplitudes=amplitudes_out,
        amplitude_errors=amplitude_errors,
        parameter_names=layout.keys,
        values=np.array(phi),
        covariance=covariance,
        priors=dict(layout.priors),
        bounds={
            key: (float(lo), float(hi))
            for key, lo, hi in zip(layout.keys, lower, upper, strict=True)
        },
        at_bounds=at_bounds,
        prior_fraction=prior_fraction,
        chi2=float(final["chi2"]),
        n_pixels=int(final["n_pixels"]),
        n_parameters=int(n_parameters),
        passes=tuple(passes),
        converged=converged,
        templates=tuple(templates_of(state)),
        table=table,
        start=start,
        notes=tuple(notes),
        seconds=time.perf_counter() - t_start,
    )
