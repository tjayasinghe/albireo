"""Run every stage of the analysis for a list of stars, from one declaration.

**Experimental.** The TOML schema read here and the ``albireo`` command line built
on it may be renamed or reshaped. They are a convenience over the stage APIs, each
of which does the same work when called directly.

For each star the driver reads the epochs (:mod:`albireo.io`), disentangles them
(:class:`albireo.Disentangler`), and fits atmospheric labels to the components so that
each can serve as a radial-velocity template (:mod:`albireo.match`). It then measures one
velocity per component per epoch by TODCOR against those templates
(:mod:`albireo.todcor`), fits a Keplerian to the resulting table (:mod:`albireo.rvorbit`),
and writes the products with figures. A stage that cannot be run (a label fit on data
whose wavelength medium is undeclared, an orbit from too few usable epochs) is recorded
as a flag on the star's report. A failure in one star does not stop the batch. The
command line is ``albireo run config.toml``.

Two rules of the underlying stages are enforced unchanged. Light fractions must be
declared and have no default: with constant light the likelihood constrains only
``l_i * d_i``, so the data cannot detect a wrong value (``docs/math.md`` §5.2). The
wavelength medium must be declared before a synthetic grid is used, because air and
vacuum wavelengths differ by a nearly constant 83 km/s.

Components are declared in order of decreasing mass (for a main-sequence pair, the
brighter star first). The likelihood is symmetric under swapping the components with their
spectra rescaled by the light ratio, so a symmetric semi-amplitude prior gives the
conjunction scan two equally deep minima. The fit is therefore started with
``K_1 < K_2 < ...`` (the first star moves least). This is a starting convention, not a
constraint, and the label stage checks the outcome: a fitted light fraction far from the
declared one is flagged as the signature of a reversed order.

There are three routes into the orbit:

- ``period = [lo, hi]`` (or a value): the Keplerian is inferred from the spectra directly,
  and the epochs are then measured against the disentangled components
  (:meth:`albireo.Fit.measure_velocities`).
- ``period = "search"``: no period is known but a synthetic library is declared. Library
  templates at the declared starting labels measure a first velocity table, a
  Lomb-Scargle search proposes the period, an orbit is fitted to the table, and the
  disentangling is warm-started from it (:meth:`albireo.RVOrbit.to_theta`). Template
  mismatch shifts each component's velocities by a constant, which leaves the period and
  the semi-amplitudes unchanged.
- ``velocities = "file"``: per-epoch velocities measured elsewhere (cross-correlation,
  line splitting). The free per-epoch table is fitted instead of a Keplerian
  (``Disentangler(velocities=...)``), and the period is found from the table afterwards.

A disentangled component's rest frame is not identified (``docs/math.md`` §5.3), so
velocities measured against it are differential: the semi-amplitudes, eccentricity and
mass ratio are exact, and the systemic velocity is not defined. When the label fit has
measured the frame offset, the pipeline applies it to the templates and the velocities
are absolute. Every velocity table and ``result.json`` state which case applies.

Stars are independent, so a batch run with ``jobs > 1`` distributes them over worker
processes; see :func:`run_pipeline`.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import datetime as _dt
import gc
import importlib.util
import json
import math
import multiprocessing
import os
import re
import sys
import time
import traceback
import warnings
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from albireo.data import Dataset, EpochData
from albireo.facade import (
    LSF,
    Between,
    Disentangler,
    Fit,
    Fixed,
    Known,
    Nebular,
    Orbit,
    Smoothness,
    Spec,
    Star,
    Telluric,
    _smoothness_of,
)
from albireo.forward import PER_EPOCH, declared_lsf_widths
from albireo.grids import LogGrid

__all__ = [
    "Analysis",
    "ComponentConfig",
    "PipelineConfig",
    "PipelineRun",
    "StarConfig",
    "StarResult",
    "config_from_dict",
    "config_template",
    "demo_config",
    "load_config",
    "run_pipeline",
    "run_star",
    "write_config_template",
]

_DEFAULT_OUTPUT = "albireo_results"
_LIBRARY_PAD_ANGSTROM = 2.0
_MAX_NAME_LENGTH = 80
_LABEL_COMPARISONS = ("epochs", "native", "matched")


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def _spec(value: Any, what: str) -> Spec | None:
    """Coerce a config value into a prior spec of :mod:`albireo.facade`.

    ``None`` stays ``None`` (the stage's default applies). A number is ``Fixed``, a
    two-element list is ``Between``, and a mapping with ``value`` and ``sigma`` is
    ``Known``. A :class:`~albireo.facade.Spec` passes through, so the Python API accepts
    specs too.
    """
    if value is None or isinstance(value, Spec):
        return value
    if isinstance(value, bool):
        raise TypeError(f"{what}: a boolean is not a prior")
    if isinstance(value, int | float | np.floating):
        return Fixed(float(value))
    if isinstance(value, Mapping):
        if "value" in value and "sigma" in value:
            return Known(float(value["value"]), float(value["sigma"]))
        if "lo" in value and "hi" in value:
            return Between(float(value["lo"]), float(value["hi"]))
        raise ValueError(f"{what}: a table must carry value/sigma (Known) or lo/hi (Between)")
    if isinstance(value, list | tuple | np.ndarray) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        if not hi > lo:
            raise ValueError(f"{what}: a range needs lo < hi; got [{lo}, {hi}]")
        return Between(lo, hi)
    raise TypeError(
        f"{what}: expected a number (fixed), a two-element range [lo, hi], a table with "
        f"value and sigma, or a façade spec; got {type(value).__name__}"
    )


def _lsf(value: Any, key: str) -> LSF:
    """Coerce a per-instrument LSF declaration."""
    if isinstance(value, LSF):
        return value
    if isinstance(value, bool):
        raise TypeError(f"lsf[{key!r}]: a boolean is not a line-spread function")
    if isinstance(value, int | float | np.floating):
        return LSF(sigma_kms=float(value))
    if isinstance(value, str) and value == PER_EPOCH:
        return LSF.per_epoch()
    if isinstance(value, Mapping):
        if "resolving_power" in value:
            return LSF.from_resolution(float(value["resolving_power"]))
        if "sigma_kms" in value:
            anchors = value.get("anchors_angstrom")
            return LSF(
                sigma_kms=value["sigma_kms"],
                anchors_angstrom=None if anchors is None else tuple(float(a) for a in anchors),
            )
        raise ValueError(f"lsf[{key!r}]: give resolving_power or sigma_kms")
    raise TypeError(f"lsf[{key!r}]: expected a sigma in km/s, a table, {PER_EPOCH!r}, or an LSF")


@dataclass(frozen=True)
class ComponentConfig:
    """One stellar component of a star, as declared to the pipeline.

    Parameters
    ----------
    name
        The component's name, used wherever a row index would otherwise be.
    light
        Its light fraction. It is required and has no default because it is an assumption:
        the light fractions of a star must sum to one, and no part of the fit can detect a
        wrong value (``docs/math.md`` §5.2). It should be quoted beside every result
        derived from the spectra. The string ``"measure"`` (for every component of the
        star) instead takes the fractions from a correlation against library templates,
        as the amplitudes fitted to the templates in the well-detected epochs
        (``light="global"`` of :func:`albireo.todcor`). This measurement is recorded in
        the report and flagged. It needs a library, and on the ``period = "search"`` route
        it reuses the bootstrap's table.
    teff, logg, vsini
        Label priors for the template-identification stage, in K, cgs dex and km/s: a
        number to hold, a ``[lo, hi]`` range, or ``None`` for the library's own range
        (``0-vsini_max`` for ``vsini``). ``logg`` is the label to fix when an eclipse
        solution provides it.
    k
        Optional semi-amplitude prior for this component in km/s, overriding the star's
        ``k_min``/``k_max``. Every component must use the same kind (all ranges, or all
        values).
    smoothness_tau0
        Starting value of this component's smoothness hyperparameter. A rotationally
        broadened star needs a much larger value than a sharp-lined one.
    """

    name: str
    light: float
    teff: Any = None
    logg: Any = None
    vsini: Any = None
    k: Any = None
    smoothness_tau0: float | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("every component needs a name")
        if isinstance(self.light, str):
            if self.light.strip().lower() != _MEASURE:
                raise ValueError(
                    f"component {self.name!r}: light must be a fraction or the string "
                    f"{_MEASURE!r}; got {self.light!r}"
                )
            object.__setattr__(self, "light", _MEASURE)
        else:
            light = float(self.light)
            if not (math.isfinite(light) and light > 0.0):
                raise ValueError(f"component {self.name!r}: light must be finite and positive")
            object.__setattr__(self, "light", light)
        object.__setattr__(self, "name", str(self.name))
        for label in ("teff", "logg", "vsini", "k"):
            _spec(getattr(self, label), f"component {self.name!r}: {label}")

    def labels_start(self) -> dict[str, float]:
        """The midpoint of each declared label prior, for a library template."""
        out = {}
        for label in ("teff", "logg", "vsini"):
            spec = _spec(getattr(self, label), label)
            if spec is not None:
                out[label] = float(np.asarray(spec.start()))
        return out


@dataclass(frozen=True)
class Analysis:
    """Settings shared by every star, each overridable per star.

    Parameters
    ----------
    region
        Wavelength window to analyse, in Angstrom. Recommended for echelle data, since the
        solve cost grows with the number of pixels.
    smooth_angstrom
        Continuum smoothing scale in Angstrom, for :func:`albireo.io.to_epoch`.
    mask
        Extra wavelength ranges to zero-weight, in Angstrom.
    k_min, k_max
        The semi-amplitude prior ``Between(k_min, k_max)`` in km/s, for every component
        that does not declare its own. ``k_max`` sets the solver's velocity budget, so a
        generous value is slower but no less correct.
    ecc_max
        Upper bound of the eccentricity prior; ``circular`` holds ``e = 0`` exactly.
    max_steps
        L-BFGS cap for the disentangling.
    z_rms_max
        Ceiling on the residual z-score rms of the disentangling
        (:attr:`albireo.Fit.z_rms`). Above it the fit is taken as diverged and the star
        stops with an error before the label and velocity stages, which would otherwise
        measure velocities against components that do not fit the data. The rms is near 1
        for a good fit and 2 to 3 at the wrong period on the blind route. The one
        divergence in the D62 benchmark had 45, with the semi-amplitudes constrained only
        by their priors.
    dv_kms
        Model-grid pixel size in km/s; the default is the finest sampling in the data.
    v_range
        Search half-range in km/s for the library-template velocity table of the
        ``period = "search"`` route.
    period_decision_candidates
        How many of the best candidate periods the disentangling decides among on the
        ``period = "search"`` route (:func:`_decide_period_by_disentangling`). The
        velocity table's chi-square ranks the candidates. The top few are compared by the
        disentangling's marginal likelihood, which uses every pixel of every epoch and has
        no per-epoch freedom, and the fit starts from the best of them. ``0`` or ``1``
        leaves the decision to the table. Each candidate requires one coarse scan.

        Four was set by measurement (D64). Eight would include the true period in the
        comparison on one further benchmark system, at four more coarse scans per blind
        star. The comparison scores each candidate at a single point in period and
        eccentricity, the two quantities a sparse table measures worst, so its flag
        indicates that the period is unreliable, not that the selected period is better.
        On the Gaia RVS benchmark the stage corrected no wrong period, and every system on
        which it overruled the table ended on a wrong period. The measurements are in
        ``docs/benchmarks.md`` (Gaia RVS, the third run).
    detection_min
        The detection statistic (:attr:`albireo.todcor.VelocityTable.delta_chi2`, the rise
        in chi-square when the component is removed) below which a companion's velocity at
        an epoch has no weight in the ``period = "search"`` bootstrap's period search
        and candidate orbit fits (:func:`_detection_gate`). Only the components after the
        first (declared in order of decreasing mass) are gated. The first component's
        velocities are kept even at a gated epoch. ``0`` turns the gate off. The orbit
        selected on the search route is also fitted to the gated copy and gives the
        disentangling its starting semi-amplitude, conjunction and eccentricity. The gate
        is not applied to the written ``template_velocities.rv``, the light measured from
        that table, the template table of the known-period route (``_table_orbit``), or
        the velocities measured after the disentangling.

        A hundred was set by measurement (D65). Over the 33 blind tables of the Gaia RVS
        benchmark's third run it recovered one period whose faint companion the library
        templates never detected, and no system lost its rank 1. The table summary's own
        threshold of 25 fails on that system. The first component is not gated because
        gating it as well worsened the result for one system. The measurements are in
        ``docs/benchmarks.md`` (Gaia RVS, after the third run).
    vsini_max, v_zero_range
        Default ceiling of the ``vsini`` prior, and the half-range of the per-component
        frame offset the label fit may measure, both in km/s. The disentangled frame is
        at the systemic velocity, so the range must cover it; 300 km/s covers the
        Magellanic Clouds.
    label_steps
        Optimizer cap for the label fit: the iteration cap of each Levenberg-Marquardt run
        in the ``"epochs"`` comparison, and the L-BFGS steps per start in the other two.
    label_compare
        What the label fit compares the templates with (:func:`albireo.match_labels`).
        ``"epochs"`` (default) compares the template composite against the epoch spectra
        through the disentangling's sufficient statistics, which does not depend on the
        declared light fractions. ``"native"`` and ``"matched"`` compare the templates
        against the disentangled components. On simulated Gaia RVS binaries the epoch
        comparison returned temperatures twice as close to the injected values as the
        native one and light fractions three times closer (``docs/math.md`` §9.2a). The
        report records it as ``labels.compare``.
    dilution
        ``"radius_ratio"`` (joint, the default), ``"scalar"`` or ``"fixed"``.
    sample, num_warmup, num_samples, num_chains
        Whether to run NUTS after the MAP, and how much.
    telluric, nebular, nebular_v_kms
        Extra components, as in the ``Disentangler`` interface.
    noise_correlation
        Lag-one correlation of each epoch's noise along its pixel index, the signature of
        a pipeline that resampled the spectra onto a common step (Gaia's RVS grids have
        0.27 and 0.81). ``None`` (default) takes the pixels as independent. A number
        declares it for every instrument, a table ``{instrument = value}`` per instrument
        (instruments left out are taken as independent), and ``"fit"`` fits one shared
        value. A declared value makes the disentangling's noise model AR(1) along the
        pixel index and widens the velocity table's errors by the sandwich the
        correlation implies (:func:`albireo.todcor`). The routes that correlate library
        templates before the disentangling use a declared value too.
    k_scan
        Scan the marginal likelihood over a coarse grid of every semi-amplitude declared
        as a range before the disentangling's L-BFGS, and start from the best trial when
        it is better than the declared start (:meth:`albireo.Disentangler.fit`). The
        default is on, because the likelihood is multimodal in the semi-amplitudes and a
        start from a dozen-epoch template table can be in the wrong basin.
    plots
        Write the diagnostic figures (needs matplotlib).
    fast
        Trim every optimizer budget for a smoke run. The qualitative result is unchanged;
        the numbers are less precise.
    """

    region: tuple[float, float] | None = None
    smooth_angstrom: float | None = None
    mask: tuple[tuple[float, float], ...] = ()
    k_min: float = 1.0
    k_max: float = 120.0
    ecc_max: float = 0.5
    circular: bool = False
    max_steps: int = 300
    z_rms_max: float = 10.0
    dv_kms: float | None = None
    v_range: float = 300.0
    period_decision_candidates: int = 4
    detection_min: float = 100.0
    vsini_max: float = 300.0
    v_zero_range: float = 300.0
    label_steps: int = 500
    label_compare: str = "epochs"
    dilution: str = "radius_ratio"
    sample: bool = False
    num_warmup: int = 300
    num_samples: int = 300
    num_chains: int = 2
    telluric: bool = False
    nebular: bool = False
    nebular_v_kms: float = 0.0
    noise_correlation: Any = None
    k_scan: bool = True
    plots: bool = True
    fast: bool = False

    def __post_init__(self) -> None:
        if self.region is not None:
            lo, hi = (float(v) for v in self.region)
            if not hi > lo:
                raise ValueError(f"region needs lo < hi; got {self.region}")
            object.__setattr__(self, "region", (lo, hi))
        object.__setattr__(
            self, "mask", tuple((float(lo), float(hi)) for lo, hi in (self.mask or ()))
        )
        if not 0.0 < self.k_min < self.k_max:
            raise ValueError(f"need 0 < k_min < k_max; got {self.k_min}, {self.k_max}")
        if not 0.0 < self.ecc_max <= 0.95:
            raise ValueError(f"ecc_max must be in (0, 0.95]; got {self.ecc_max}")
        if self.dilution not in ("radius_ratio", "scalar", "fixed"):
            raise ValueError("dilution must be 'radius_ratio', 'scalar' or 'fixed'")
        if self.label_compare not in _LABEL_COMPARISONS:
            raise ValueError(
                f"label_compare must be one of {', '.join(repr(c) for c in _LABEL_COMPARISONS)}; "
                f"got {self.label_compare!r}"
            )
        if self.max_steps < 1 or self.label_steps < 1:
            raise ValueError("max_steps and label_steps must be positive")
        if not float(self.z_rms_max) > 0.0:
            raise ValueError(f"z_rms_max must be positive; got {self.z_rms_max}")
        if isinstance(self.period_decision_candidates, bool) or self.period_decision_candidates < 0:
            raise ValueError(
                "period_decision_candidates must be a non-negative count of candidate "
                f"periods; got {self.period_decision_candidates!r}"
            )
        if isinstance(self.detection_min, bool) or not float(self.detection_min) >= 0.0:
            raise ValueError(
                "detection_min must be a non-negative detection statistic (0 turns the gate "
                f"off); got {self.detection_min!r}"
            )
        declared = self.noise_correlation
        if isinstance(declared, str):
            if declared != "fit":
                raise ValueError(
                    f"noise_correlation must be a number, a table per instrument or 'fit'; "
                    f"got {declared!r}"
                )
        elif declared is not None:
            values = list(declared.values()) if isinstance(declared, Mapping) else [declared]
            for value in values:
                if isinstance(value, bool) or not -1.0 < float(value) < 1.0:
                    raise ValueError(
                        f"noise_correlation must lie in (-1, 1); got {declared!r}. It is the "
                        "lag-one correlation of the pixel noise, not a variance."
                    )

    def effective(self) -> Analysis:
        """These settings with the ``fast`` trims applied."""
        if not self.fast:
            return self
        return replace(
            self,
            max_steps=min(self.max_steps, 40),
            label_steps=min(self.label_steps, 60),
            num_warmup=min(self.num_warmup, 60),
            num_samples=min(self.num_samples, 60),
            num_chains=1,
        )


_ANALYSIS_KEYS = frozenset(f.name for f in dataclasses.fields(Analysis))
_MEASURE = "measure"


@dataclass(frozen=True)
class StarConfig:
    """One star: where its spectra are, what its components are, what is known.

    Parameters
    ----------
    name
        Used for the output directory and every report line. Must be unique in a batch.
    spectra
        A glob, a directory or a list of FITS paths. One of ``spectra``, ``dataset`` and
        ``bloem`` is required.
    dataset
        An in-memory :class:`~albireo.Dataset` instead of files (the route of the Python
        API).
    bloem
        A BLOeM survey identifier; the epochs are fetched from the ESO archive into the
        star's output directory first (network).
    period
        The orbital period in days: ``[lo, hi]`` for a uniform prior, a number to hold it,
        ``{value, sigma}`` for a Gaussian, or ``"search"`` to bootstrap from library
        templates (which needs a library).
    t_conj
        The time of conjunction (the primary eclipse of an eclipsing binary, in the
        epochs' time system): ``{value, sigma}`` for a Gaussian, ``[lo, hi]`` for a
        range, or a number to hold it. Default ``None``: the conjunction phase is located
        by a scan over one period before the fit.
    ecc
        The eccentricity: a number to hold it (with ``omega``), or ``[0, hi]`` for a
        range. Default ``None``: ``[0, ecc_max]`` from the settings, or held at zero when
        ``circular`` is set.
    omega
        The argument of periastron of the first component in radians, needed only with
        a held non-zero ``ecc``.
    velocities
        Alternative to ``period``: a text file of measured per-epoch velocities in km/s,
        one column per component (optionally preceded by a BJD column, matched to the
        epochs), or an ``(n_components, n_epochs)`` array. The free per-epoch table is
        then fitted instead of a Keplerian.
    components
        One :class:`ComponentConfig` per star. Lights must sum to one.
    instrument
        Override the instrument key every file resolves to.
    medium
        Declare the wavelength scale (``"air"`` or ``"vacuum"``) when the files do not.
    lsf
        Per-instrument line-spread declarations for this star, overriding the batch's.
    labels
        Set ``False`` to skip the label stage for this star even when a library is given.
    truth
        For simulated stars only: injected values to compare against. Keys: ``k`` (one
        per component, km/s), ``period`` (d), ``ecc``, ``omega`` (rad), ``t_conj``,
        ``gamma`` (km/s), ``velocities`` (``(n_components, n_epochs)``, barycentric),
        ``light_fractions``, ``labels`` (per component name), ``components`` (deviation
        spectra on ``grid``), and ``windows`` (a mapping of window name to a list of
        ``(lo, hi)`` in Angstrom over which the component spectra are compared by
        equivalent width). Every key is optional; the report compares what is given.
    overrides
        Per-star values for any :class:`Analysis` field.
    """

    name: str
    spectra: Any = None
    dataset: Dataset | None = None
    bloem: str | None = None
    period: Any = None
    velocities: Any = None
    components: Sequence[ComponentConfig] = ()
    t_conj: Any = None
    ecc: Any = None
    omega: Any = None
    instrument: str | None = None
    medium: str | None = None
    lsf: Mapping[str, Any] = field(default_factory=dict)
    labels: bool = True
    truth: Mapping[str, Any] | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("every star needs a name")
        if len(name) > _MAX_NAME_LENGTH:
            raise ValueError(f"star name {name!r} is longer than {_MAX_NAME_LENGTH} characters")
        object.__setattr__(self, "name", name)
        sources = [s for s in (self.spectra, self.dataset, self.bloem) if s is not None]
        if len(sources) != 1:
            raise ValueError(
                f"star {name!r}: give exactly one of spectra=, dataset= and bloem=; "
                f"got {len(sources)}"
            )
        components = tuple(
            c if isinstance(c, ComponentConfig) else ComponentConfig(**dict(c))
            for c in self.components
        )
        if not components:
            raise ValueError(f"star {name!r}: declare at least one component")
        names = [c.name for c in components]
        if len(set(names)) != len(names):
            raise ValueError(f"star {name!r}: component names must be unique; got {names}")
        measured = [c.light == _MEASURE for c in components]
        if any(measured):
            if not all(measured):
                raise ValueError(
                    f"star {name!r}: light = 'measure' applies to every component or to "
                    "none; the fractions are measured together"
                )
        else:
            total = sum(c.light for c in components)
            if abs(total - 1.0) > 1e-6:
                listed = ", ".join(f"{c.name}={c.light:g}" for c in components)
                raise ValueError(
                    f"star {name!r}: the light fractions must sum to 1; {listed} sums to "
                    f"{total:g}. This is an assumption the data cannot check, which is why "
                    "it has no default."
                )
        object.__setattr__(self, "components", components)
        t_conj = _spec(self.t_conj, f"star {name!r}: t_conj")
        ecc = _spec(self.ecc, f"star {name!r}: ecc")
        if isinstance(ecc, Known):
            raise ValueError(
                f"star {name!r}: ecc takes a number to hold or a [0, hi] range; a Gaussian "
                "cannot be expressed in the (sqrt(e) cos w, sqrt(e) sin w) parameterization"
            )
        if isinstance(ecc, Between) and float(np.asarray(ecc.lo)) != 0.0:
            raise ValueError(f"star {name!r}: an ecc range must start at 0")
        omega = _spec(self.omega, f"star {name!r}: omega")
        if isinstance(ecc, Fixed) and float(np.asarray(ecc.value)) > 0.0 and omega is None:
            raise ValueError(f"star {name!r}: a held non-zero ecc needs omega (radians) as well")
        del t_conj
        if (self.period is None) == (self.velocities is None):
            raise ValueError(
                f"star {name!r}: declare exactly one of period= and velocities=. A period "
                "([lo, hi], a value, or 'search') fits a Keplerian; a velocity table fits "
                "the free per-epoch table instead."
            )
        if self.period is not None and not (
            isinstance(self.period, str) and self.period.lower() == "search"
        ):
            _spec(self.period, f"star {name!r}: period")
        if self.medium is not None and self.medium not in ("air", "vacuum"):
            raise ValueError(f"star {name!r}: medium must be 'air' or 'vacuum'")
        unknown = sorted(set(self.overrides) - _ANALYSIS_KEYS)
        if unknown:
            raise ValueError(
                f"star {name!r}: unknown setting(s) {unknown}; the per-star settings are "
                f"{sorted(_ANALYSIS_KEYS)}"
            )
        object.__setattr__(self, "lsf", dict(self.lsf))
        object.__setattr__(self, "overrides", dict(self.overrides))
        for key, value in self.lsf.items():
            _lsf(value, key)

    @property
    def searching(self) -> bool:
        """Whether the orbit is to be bootstrapped from library templates."""
        return isinstance(self.period, str) and self.period.lower() == "search"

    @property
    def measures_light(self) -> bool:
        """Whether the light fractions are to be measured by the bootstrap correlation."""
        return any(c.light == _MEASURE for c in self.components)

    def settings(self, base: Analysis) -> Analysis:
        """The batch settings with this star's overrides and the ``fast`` trims applied."""
        return replace(base, **self.overrides).effective()


@dataclass(frozen=True)
class PipelineConfig:
    """A batch: the stars, the shared instrument facts, and the shared settings.

    Parameters
    ----------
    stars
        The :class:`StarConfig` declarations, with unique names.
    output
        Directory for the results; one sub-directory per star is created inside it.
    lsf
        Per-instrument line-spread functions shared by every star, keyed by the
        instrument name the files resolve to: a sigma in km/s, ``{"resolving_power": R}``,
        ``{"sigma_kms": ...}``, ``"per-epoch"`` or an :class:`~albireo.LSF`. An instrument
        with no entry anywhere takes ``R`` from each file's own header when every file
        has one (the per-epoch declaration).
    library
        The synthetic grid for the label stage: a registry name
        (:func:`albireo.library_names`), a path to a saved library, or a
        :class:`~albireo.SpectralLibrary`. ``None`` skips the stage.
    mh
        The shared metallicity prior for the label fit, in dex (a number, a range, or
        ``None`` for the library's own range).
    analysis
        The shared :class:`Analysis` settings.
    read_kwargs
        Extra keyword arguments for :func:`albireo.io.read_spectrum`.
    """

    stars: Sequence[StarConfig]
    output: str | os.PathLike = _DEFAULT_OUTPUT
    lsf: Mapping[str, Any] = field(default_factory=dict)
    library: Any = None
    mh: Any = None
    analysis: Analysis = field(default_factory=Analysis)
    read_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stars = tuple(s if isinstance(s, StarConfig) else StarConfig(**dict(s)) for s in self.stars)
        if not stars:
            raise ValueError("a pipeline needs at least one star")
        names = [s.name for s in stars]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"star names must be unique; duplicated: {duplicates}")
        object.__setattr__(self, "stars", stars)
        object.__setattr__(self, "lsf", dict(self.lsf))
        object.__setattr__(self, "read_kwargs", dict(self.read_kwargs))
        for key, value in self.lsf.items():
            _lsf(value, key)
        _spec(self.mh, "mh")
        if isinstance(self.analysis, Mapping):
            object.__setattr__(self, "analysis", Analysis(**dict(self.analysis)))
        for star in stars:
            if star.searching and self.library is None:
                raise ValueError(
                    f"star {star.name!r} declares period = 'search', which bootstraps the "
                    "orbit from library templates, so a library is required."
                )
            if star.measures_light and self.library is None:
                raise ValueError(
                    f"star {star.name!r} declares light = 'measure', which correlates the "
                    "epochs against library templates, so a library is required."
                )

    def star(self, name: str) -> StarConfig:
        """The declaration of one star, by name."""
        for star in self.stars:
            if star.name == name:
                return star
        raise KeyError(f"no star called {name!r}; this batch has {[s.name for s in self.stars]}")

    def without_stars(self) -> PipelineConfig:
        """The shared part alone, as sent to each worker process at initialization."""
        return replace(self, stars=(self.stars[0],))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-able echo of the declaration, for the run manifest."""
        return _jsonable(
            {
                "output": os.fspath(self.output),
                "lsf": {k: _describe_lsf(_lsf(v, k)) for k, v in self.lsf.items()},
                "library": _describe_library(self.library),
                "mh": _describe_spec(_spec(self.mh, "mh")),
                "analysis": dataclasses.asdict(self.analysis),
                "stars": [_describe_star(s) for s in self.stars],
            }
        )


def _describe_spec(spec: Spec | None) -> Any:
    if spec is None:
        return None
    if isinstance(spec, Fixed):
        return {"fixed": np.asarray(spec.value).tolist()}
    if isinstance(spec, Between):
        return {"lo": np.asarray(spec.lo).tolist(), "hi": np.asarray(spec.hi).tolist()}
    if isinstance(spec, Known):
        return {"value": np.asarray(spec.value).tolist(), "sigma": np.asarray(spec.sigma).tolist()}
    return {"spec": type(spec).__name__}


def _describe_lsf(lsf: LSF) -> dict[str, Any]:
    out: dict[str, Any] = {"sigma_kms": np.asarray(lsf.sigma_kms).tolist()}
    if lsf.anchors_angstrom is not None:
        out["anchors_angstrom"] = list(lsf.anchors_angstrom)
    return out


def _describe_library(library: Any) -> Any:
    if library is None:
        return None
    if isinstance(library, str | os.PathLike):
        return os.fspath(library)
    meta = dict(getattr(library, "meta", {}))
    return {"in_memory": True, "grid": meta.get("grid", "unnamed"), "n_nodes": library.n_nodes}


def _describe_star(star: StarConfig) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": star.name,
        "spectra": None if star.spectra is None else _paths_as_list(star.spectra),
        "dataset": None if star.dataset is None else f"in memory ({star.dataset.n_epochs} epochs)",
        "bloem": star.bloem,
        "period": "search"
        if star.searching
        else _describe_spec(_spec(star.period, "period") if star.period is not None else None),
        "velocities": None
        if star.velocities is None
        else (star.velocities if isinstance(star.velocities, str) else "array"),
        "t_conj": "scan" if star.t_conj is None else _describe_spec(_spec(star.t_conj, "t_conj")),
        "ecc": _describe_spec(_spec(star.ecc, "ecc")),
        "omega": _describe_spec(_spec(star.omega, "omega")),
        "components": [
            {
                "name": c.name,
                "light": c.light,
                "teff": _describe_spec(_spec(c.teff, "teff")),
                "logg": _describe_spec(_spec(c.logg, "logg")),
                "vsini": _describe_spec(_spec(c.vsini, "vsini")),
                "k": _describe_spec(_spec(c.k, "k")),
            }
            for c in star.components
        ],
        "instrument": star.instrument,
        "medium": star.medium,
        "lsf": {k: _describe_lsf(_lsf(v, k)) for k, v in star.lsf.items()},
        "labels": star.labels,
        "overrides": dict(star.overrides),
    }
    return out


def _paths_as_list(paths: Any) -> list[str]:
    if isinstance(paths, str | os.PathLike):
        return [os.fspath(paths)]
    return [os.fspath(p) for p in paths]


# ---------------------------------------------------------------------------
# TOML
# ---------------------------------------------------------------------------

_TEMPLATE = """# albireo pipeline configuration.
#
#     albireo run albireo.toml            # every star below, one after the other
#     albireo run albireo.toml --jobs 4   # four stars at a time in worker processes
#
# Two values are required and have no default, because they are assumptions the data
# cannot check: the light fraction of each component, and the wavelength scale wherever a
# synthetic grid is consulted. Every other setting has a default, which the reports state.

[output]
directory = "albireo_results"   # one sub-directory per star is written inside it
plots = true                    # diagnostic figures (needs matplotlib)

# Line-spread function per instrument. The key is the instrument name the FITS headers
# resolve to (INSTRUME), or the `instrument =` override on a star. An instrument with no
# entry takes its resolving power from each file's own header (SPEC_RES) when every file
# has one, so exposures at two resolving powers under one name (HARPS HAM and EGGS) are
# each modelled at their own; `"per-epoch"` says so explicitly.
# [instrument]
# UVES = "per-epoch"           # each file's own SPEC_RES, stated rather than defaulted
[instrument.HARPS]
resolving_power = 115000
# [instrument.FEROS]
# sigma_kms = 2.65             # a Gaussian sigma in km/s, instead of a resolving power

[analysis]
region = [5000.0, 5300.0]       # Angstrom. Recommended: the solve cost grows with the pixel count
smooth_angstrom = 120.0         # continuum smoothing scale for unnormalized spectra
k_min = 1.0                     # semi-amplitude prior, km/s, for components without their own
k_max = 120.0                   # sets the solver's velocity budget; a large value costs time only
ecc_max = 0.5                   # eccentricity prior ceiling; `circular = true` holds e = 0
max_steps = 300                 # L-BFGS cap for the disentangling
z_rms_max = 10.0                # stop a star whose disentangling diverged: the residual
#                               # z-score rms is near 1 for a healthy fit
# noise_correlation = 0.3       # lag-one correlation of the pixel noise, from the resampling
#                               # that produced the spectra: a number, a table per
#                               # instrument ({HARPS = 0.3}), or "fit"
# k_scan = true                 # scan the semi-amplitudes declared as ranges before the fit
# period_decision_candidates = 4  # candidate periods the disentangling itself decides among
#                               # on the search route; 0 leaves the decision to the table
# detection_min = 100.0         # on the search route, a companion's velocity whose detection
#                               # statistic is below this carries no weight in the period
#                               # search and the candidate fits; 0 turns the gate off
# sample = true                 # NUTS after the MAP: posterior widths, slow

# Template identification: fit Teff, log g, [M/H] and v sin i to the disentangled
# components against a published grid, and use the fitted frame offset to make the epoch
# velocities absolute. Delete this table to skip the stage (velocities stay differential).
[labels]
library = "bosz2024-fgk-r20000"   # albireo.library_names(); downloads ~621 MB once
mh = [-1.0, 0.5]                  # shared metallicity range, or a number to hold it
v_zero_range = 300.0              # how far the disentangled frame may sit from rest, km/s
# compare = "epochs"              # the templates against the epoch spectra (default), or
#                                 # "native"/"matched" against the disentangled components

[[stars]]
name = "AI Phe"
spectra = "data/aiphe/*.fits"   # a glob, a directory, or a list of files
period = [24.5, 24.7]           # days: [lo, hi] uniform, a value to hold, or "search"
# velocities = "aiphe_rv.txt"   # instead of period: measured per-epoch velocities
# t_conj = {value = 2455000.12, sigma = 0.01}   # a known conjunction (primary eclipse)
#                                                 instead of the phase scan
# ecc = 0.19                    # hold the eccentricity (then give omega, radians), or
# ecc = [0.0, 0.5]              # a range; default: [0, ecc_max] from [analysis]

# Components in order of decreasing mass (the brighter star first for a main-sequence
# pair): the fit is started with K_1 < K_2, which is the convention that assigns the
# spectra to the stars.
[[stars.components]]
name = "primary"
light = 0.55                    # required: the light fractions must sum to 1
teff = [5500.0, 7000.0]         # label priors: a range, a value to hold, or omit
logg = 4.0

[[stars.components]]
name = "secondary"
light = 0.45
teff = [4300.0, 5900.0]
logg = 3.6

# [[stars]]
# name = "BLOeM 1-037"
# bloem = "1-037"               # fetched from the ESO archive into the output directory
# period = "search"             # bootstrapped from library templates (needs [labels])
# region = [4120.0, 4300.0]     # any [analysis] key can be overridden per star
# medium = "air"
# [[stars.components]]
# name = "A"
# light = "measure"             # on the "search" route only: the fractions the library
#                               # templates receive in the bootstrap correlation
"""


def config_template() -> str:
    """The annotated ``albireo.toml`` that ``albireo init`` writes."""
    return _TEMPLATE


def write_config_template(path: str | os.PathLike = "albireo.toml", *, overwrite=False) -> Path:
    """Write the annotated template to ``path`` and return it."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    path.write_text(_TEMPLATE, encoding="utf-8")
    return path


def load_config(path: str | os.PathLike) -> PipelineConfig:
    """Read a TOML configuration into a :class:`PipelineConfig`.

    Relative paths inside the file (the spectra globs, the output directory, a library
    path) are resolved against the file's own directory, so a configuration can be run
    from any working directory.
    """
    import tomllib

    path = Path(path)
    with path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{path}: not valid TOML: {exc}") from None
    return config_from_dict(data, base_dir=path.parent)


def _resolve_path(value: Any, base_dir: Path | None) -> Any:
    if base_dir is None or not isinstance(value, str):
        return value
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
        return value
    candidate = Path(value)
    return value if candidate.is_absolute() else os.fspath(base_dir / candidate)


def config_from_dict(data: Mapping[str, Any], *, base_dir: str | os.PathLike | None = None):
    """Build a :class:`PipelineConfig` from the dictionary form of the TOML schema."""
    base = None if base_dir is None else Path(base_dir)
    data = dict(data)
    known = {"output", "instrument", "analysis", "labels", "sampling", "stars", "read"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown top-level table(s) {unknown}; expected {sorted(known)}")

    output_table = dict(data.get("output", {}))
    output = _resolve_path(output_table.pop("directory", _DEFAULT_OUTPUT), base)
    analysis_values: dict[str, Any] = {}
    if "plots" in output_table:
        analysis_values["plots"] = bool(output_table.pop("plots"))
    if output_table:
        raise ValueError(f"unknown [output] key(s) {sorted(output_table)}")

    analysis_values.update(dict(data.get("analysis", {})))
    labels_table = dict(data.get("labels", {}))
    library = labels_table.pop("library", None)
    if isinstance(library, str):
        from albireo.library import library_names

        # A registered library name is not a path; any other string is resolved as a file.
        if library not in library_names():
            library = _resolve_path(library, base)
    mh = labels_table.pop("mh", None)
    for key in ("vsini_max", "v_zero_range", "dilution"):
        if key in labels_table:
            analysis_values[key] = labels_table.pop(key)
    if "steps" in labels_table:
        analysis_values["label_steps"] = int(labels_table.pop("steps"))
    if "compare" in labels_table:
        analysis_values["label_compare"] = str(labels_table.pop("compare"))
    if labels_table:
        raise ValueError(f"unknown [labels] key(s) {sorted(labels_table)}")

    sampling = dict(data.get("sampling", {}))
    if sampling:
        analysis_values["sample"] = bool(sampling.pop("enabled", True))
        for key in ("num_warmup", "num_samples", "num_chains"):
            if key in sampling:
                analysis_values[key] = int(sampling.pop(key))
        if sampling:
            raise ValueError(f"unknown [sampling] key(s) {sorted(sampling)}")

    unknown_analysis = sorted(set(analysis_values) - _ANALYSIS_KEYS)
    if unknown_analysis:
        raise ValueError(
            f"unknown [analysis] key(s) {unknown_analysis}; the settings are "
            f"{sorted(_ANALYSIS_KEYS)}"
        )
    analysis = Analysis(**analysis_values)

    stars = []
    for entry in data.get("stars", []):
        entry = dict(entry)
        components = [dict(c) for c in entry.pop("components", [])]
        star_keys = {
            "name",
            "spectra",
            "bloem",
            "period",
            "velocities",
            "t_conj",
            "ecc",
            "omega",
            "instrument",
            "medium",
            "lsf",
            "labels",
        }
        overrides = {k: entry.pop(k) for k in list(entry) if k in _ANALYSIS_KEYS}
        unknown_star = sorted(set(entry) - star_keys)
        if unknown_star:
            raise ValueError(
                f"star {entry.get('name', '?')!r}: unknown key(s) {unknown_star}; expected "
                f"{sorted(star_keys)} or a per-star setting from {sorted(_ANALYSIS_KEYS)}"
            )
        spectra = entry.get("spectra")
        if isinstance(spectra, list):
            spectra = [_resolve_path(p, base) for p in spectra]
        else:
            spectra = _resolve_path(spectra, base)
        velocities = entry.get("velocities")
        if isinstance(velocities, str):
            velocities = _resolve_path(velocities, base)
        stars.append(
            StarConfig(
                name=entry.get("name", ""),
                spectra=spectra,
                bloem=entry.get("bloem"),
                period=entry.get("period"),
                velocities=velocities,
                components=[ComponentConfig(**c) for c in components],
                t_conj=entry.get("t_conj"),
                ecc=entry.get("ecc"),
                omega=entry.get("omega"),
                instrument=entry.get("instrument"),
                medium=entry.get("medium"),
                lsf=dict(entry.get("lsf", {})),
                labels=bool(entry.get("labels", True)),
                overrides=overrides,
            )
        )
    return PipelineConfig(
        stars=stars,
        output=output,
        lsf=dict(data.get("instrument", {})),
        library=library,
        mh=mh,
        analysis=analysis,
        read_kwargs=dict(data.get("read", {})),
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class StarResult:
    """The products of one star: plain data that can be pickled, plus file paths.

    Attributes
    ----------
    name, status, directory
        The star, ``"ok"`` or ``"failed"``, and the directory holding its files.
    seconds
        Wall time per stage, and ``total``.
    flags
        Every caveat the run recorded: a noise model that does not describe the data, a
        skipped stage and the reason, an orbit that disagrees with the disentangling. They
        qualify the numbers in ``report``.
    warnings
        Every distinct warning the stages raised.
    files
        Paths of the written products, keyed by kind.
    report
        The JSON-able report (the contents of ``result.json``).
    summary
        The text report (the contents of ``summary.txt``).
    error, traceback
        On failure, the exception and its traceback.
    live
        On an in-process run only: the :class:`~albireo.Fit`, the velocity table, the
        label match, the orbit and the posterior, keyed by name. ``None`` from a worker.
    """

    name: str
    status: str
    directory: str
    seconds: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    error: str | None = None
    traceback: str | None = None
    live: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        """Everything but the live objects."""
        # dataclasses.asdict is avoided: it would deep-copy the live Fit and its compiled
        # model, which the dictionary omits.
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "live"}
        return _jsonable(out)


@dataclass
class PipelineRun:
    """A finished batch: one :class:`StarResult` per star, and the failures by name."""

    results: dict[str, StarResult]
    directory: Path
    seconds: float
    jobs: int

    @property
    def failures(self) -> dict[str, str]:
        return {n: r.error or "failed" for n, r in self.results.items() if not r.ok}

    @property
    def succeeded(self) -> dict[str, StarResult]:
        return {n: r for n, r in self.results.items() if r.ok}

    def rows(self) -> list[dict[str, Any]]:
        """One flat row per star, the columns of ``results.csv``."""
        return [_flat_row(result) for result in self.results.values()]

    def summary(self) -> str:
        lines = [
            f"albireo pipeline: {len(self.succeeded)} of {len(self.results)} star(s) "
            f"completed in {self.seconds:.1f} s with {self.jobs} worker(s); "
            f"results in {self.directory}"
        ]
        for name, result in self.results.items():
            if not result.ok:
                lines.append(f"  {name}: FAILED - {result.error}")
                continue
            orbit = result.report.get("orbit") or {}
            k = orbit.get("k") or {}
            k_text = ", ".join(f"K_{n} {v:.2f}" for n, v in k.items()) if k else "no orbit"
            period = orbit.get("period")
            period_text = f"P {period:.5f} d, " if period else ""
            flags = f", {len(result.flags)} flag(s)" if result.flags else ""
            lines.append(
                f"  {name}: {period_text}{k_text} km/s"
                f" ({result.seconds.get('total', 0.0):.0f} s{flags})"
            )
        return "\n".join(lines)

    def write(self) -> dict[str, Path]:
        """Write ``results.json``, ``results.csv``, ``summary.txt`` and ``failures.txt``."""
        self.directory.mkdir(parents=True, exist_ok=True)
        written = {}
        payload = {
            "albireo": _version(),
            "created": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "seconds": self.seconds,
            "jobs": self.jobs,
            "stars": {name: r.to_dict() for name, r in self.results.items()},
        }
        written["results"] = self.directory / "results.json"
        written["results"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written["table"] = self.directory / "results.csv"
        _write_csv(written["table"], self.rows())
        written["summary"] = self.directory / "summary.txt"
        written["summary"].write_text(self.summary() + "\n", encoding="utf-8")
        failures = self.failures
        path = self.directory / "failures.txt"
        if failures:
            path.write_text(
                "\n".join(f"{name}: {why}" for name, why in failures.items()) + "\n",
                encoding="utf-8",
            )
            written["failures"] = path
        elif path.exists():
            path.unlink()
        return written


def _flat_row(result: StarResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "star": result.name,
        "status": result.status,
        "seconds": round(result.seconds.get("total", 0.0), 1),
    }
    report = result.report
    dataset = report.get("dataset") or {}
    row["n_epochs"] = dataset.get("n_epochs")
    velocities = report.get("velocities") or {}
    row["n_usable"] = velocities.get("n_usable")
    row["absolute"] = velocities.get("absolute_all")
    orbit = report.get("orbit") or {}
    for key in ("period", "period_err", "ecc", "ecc_err", "q"):
        row[key] = orbit.get(key)
    names = list((report.get("declaration") or {}).get("component_names", []))
    for i, name in enumerate(names, start=1):
        k = (orbit.get("k") or {}).get(name)
        row[f"K_{i}"] = k
        row[f"K_{i}_err"] = (orbit.get("k_err") or {}).get(name)
        row[f"gamma_{i}"] = (orbit.get("gamma") or {}).get(name)
        labels = ((report.get("labels") or {}).get("components") or {}).get(name) or {}
        for label in ("teff", "logg", "mh", "vsini"):
            row[f"{label}_{i}"] = labels.get(label)
    disentangling = report.get("disentangling") or {}
    for i, name in enumerate(names, start=1):
        row[f"K_{i}_disentangling"] = (disentangling.get("k") or {}).get(name)
    row["z_rms"] = disentangling.get("z_rms")
    row["flags"] = "; ".join(result.flags)
    row["error"] = result.error or ""
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def _version() -> str:
    from albireo import __version__

    return __version__


def _jsonable(value: Any) -> Any:
    """Numpy and JAX values as plain Python, recursively, for ``json.dumps``."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__array__") and not isinstance(value, str):
        array = np.asarray(value)
        if array.ndim == 0:
            return array.item()
        return array.tolist()
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# One star
# ---------------------------------------------------------------------------


class _Log:
    """Per-star progress lines: printed as they happen and kept in ``log.txt``."""

    def __init__(self, name: str, path: Path, progress: bool):
        self.name = name
        self.path = path
        self.progress = progress
        self.lines: list[str] = []
        self.warnings: list[str] = []
        self._started = time.perf_counter()

    def __call__(self, text: str) -> None:
        stamp = time.perf_counter() - self._started
        line = f"[{self.name} {stamp:7.1f}s] {text}"
        self.lines.append(line)
        if self.progress:
            print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            self("warning: " + message.replace("\n", " "))


@dataclass
class _Context:
    star: StarConfig
    config: PipelineConfig
    settings: Analysis
    directory: Path
    log: _Log
    flags: list[str] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    zero_points: dict[str, Any] | None = None

    def flag(self, text: str) -> None:
        self.flags.append(text)
        self.log("flag: " + text)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - t0


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", name.strip()).strip("._")
    return cleaned or "star"


def run_star(
    star: StarConfig | Mapping[str, Any],
    config: PipelineConfig | None = None,
    *,
    directory: str | os.PathLike | None = None,
    progress: bool = True,
) -> StarResult:
    """Run every stage for one star and write its products.

    This is the single-star entry point of the Python API. It raises on a failure, unlike
    the batch driver, which records it. Use :func:`run_pipeline` for many stars.

    Parameters
    ----------
    star
        A :class:`StarConfig`, or its dictionary form.
    config
        The shared :class:`PipelineConfig` (instrument facts, library, settings). Default:
        one built from the star alone.
    directory
        Where to write. Default: ``<config.output>/<star name>``.
    progress
        Print one line per stage.
    """
    if not isinstance(star, StarConfig):
        star = StarConfig(**dict(star))
    if config is None:
        config = PipelineConfig(stars=[star])
    where = (
        Path(directory) if directory is not None else Path(config.output) / _safe_name(star.name)
    )
    result = _run_star_guarded(star, config, where, progress=progress, keep_live=True)
    if not result.ok:
        raise RuntimeError(f"{star.name}: {result.error}\n{result.traceback or ''}")
    return result


def _run_star_guarded(
    star: StarConfig, config: PipelineConfig, directory: Path, *, progress: bool, keep_live: bool
) -> StarResult:
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "log.txt"
    if log_path.exists():
        log_path.unlink()
    log = _Log(star.name, log_path, progress)
    settings = star.settings(config.analysis)
    ctx = _Context(star=star, config=config, settings=settings, directory=directory, log=log)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = lambda message, category, *_args, **_kw: log.warn(
            f"{category.__name__}: {message}"
        )
        try:
            report, summary, live = _run_stages(ctx)
        except Exception as exc:
            ctx.seconds["total"] = time.perf_counter() - t0
            text = traceback.format_exc()
            log(f"FAILED: {type(exc).__name__}: {exc}")
            (directory / "error.txt").write_text(text, encoding="utf-8")
            return StarResult(
                name=star.name,
                status="failed",
                directory=os.fspath(directory),
                seconds=dict(ctx.seconds),
                flags=list(ctx.flags),
                warnings=list(log.warnings),
                files={"log": os.fspath(log_path), "error": os.fspath(directory / "error.txt")},
                error=f"{type(exc).__name__}: {exc}",
                traceback=text,
            )
    ctx.seconds["total"] = time.perf_counter() - t0
    report["seconds"] = dict(ctx.seconds)
    report["flags"] = list(ctx.flags)
    report["warnings"] = list(log.warnings)
    ctx.files["log"] = os.fspath(log_path)
    ctx.files["report"] = os.fspath(directory / "result.json")
    ctx.files["summary"] = os.fspath(directory / "summary.txt")
    report["files"] = dict(ctx.files)
    (directory / "result.json").write_text(json.dumps(_jsonable(report), indent=2), "utf-8")
    summary = summary + "\n\n" + _files_block(ctx.files)
    (directory / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    log(f"done in {ctx.seconds['total']:.1f} s")
    return StarResult(
        name=star.name,
        status="ok",
        directory=os.fspath(directory),
        seconds=dict(ctx.seconds),
        flags=list(ctx.flags),
        warnings=list(log.warnings),
        files=dict(ctx.files),
        report=_jsonable(report),
        summary=summary,
        live=live if keep_live else None,
    )


def _run_stages(ctx: _Context) -> tuple[dict[str, Any], str, dict[str, Any]]:
    star, settings, log = ctx.star, ctx.settings, ctx.log
    sections: list[str] = [
        f"albireo {_version()} pipeline report for {star.name}",
        f"  written {_dt.datetime.now(_dt.UTC).isoformat(timespec='seconds')} into {ctx.directory}",
    ]
    report: dict[str, Any] = {"star": star.name, "albireo": _version(), "status": "ok"}
    live: dict[str, Any] = {}

    # 1. the epochs
    with ctx.stage("read"):
        dataset, header_lsf = _load_dataset(ctx)
        lsf = _resolve_lsf(ctx, dataset, header_lsf)
    log(
        f"{dataset.n_epochs} epochs from {len(dataset.instruments)} instrument(s), "
        f"{dataset.frame}, {dataset[0].wave[0]:.1f}-{dataset[0].wave[-1]:.1f} A"
    )
    sections.append(dataset.summary())
    report["dataset"] = _describe_dataset(dataset, lsf)
    live["dataset"] = dataset

    # 2. the library, if any
    library = None
    if ctx.config.library is not None and (
        star.labels
        or star.searching
        or star.measures_light
        or any(c.k is None for c in star.components)
    ):
        with ctx.stage("library"):
            library = _resolve_library(ctx.config.library, log)

    # 3. the orbit declaration, bootstrapped if requested
    bootstrap = None
    declared_velocities = None
    if star.velocities is not None:
        declared_velocities = _read_velocities(star, dataset)
        log("free per-epoch table declared from measured velocities")
    elif star.searching:
        with ctx.stage("bootstrap"):
            orbit_spec, bootstrap = _bootstrap(ctx, dataset, lsf, library)
            _write_template_table(
                ctx,
                bootstrap["objects"]["table"],
                "bootstrap",
                unexchanged=bootstrap["objects"]["table_unexchanged"],
            )
        star = ctx.star  # the bootstrap may have filled in measured light fractions
        sections.append(bootstrap["text"])
        report["bootstrap"] = bootstrap["report"]
        live["bootstrap"] = bootstrap["objects"]
    else:
        starts = None
        elements = None
        if star.measures_light:
            with ctx.stage("light"):
                measured = _measure_light(ctx, dataset, lsf, library)
                _write_template_table(ctx, measured["objects"]["table"], "light")
            star = ctx.star  # the measured fractions replaced the declaration
            sections.append(measured["text"])
            report["light"] = measured["report"]
            live["light"] = measured["objects"]
            table_orbit = _table_orbit(ctx, measured["objects"]["table"], star, settings)
            starts = _k_starts(ctx, table_orbit, star, settings)
            elements = _element_starts(ctx, table_orbit, star, settings)
        elif library is not None and any(c.k is None for c in star.components):
            # A semi-amplitude is a range and a library is available, so a template table
            # at the declared period provides the starting values, at a cost of a few
            # seconds. The table was not requested, so where it cannot be rendered the star
            # continues without it. The declared ranges are then started at their evenly
            # spaced points, as when no library is declared.
            if dataset[0].medium is None:
                ctx.flag(
                    "semi-amplitude starts skipped: the files do not declare whether their "
                    "wavelengths are air or vacuum (an 83 km/s question), so no library "
                    "template can be rendered; the range's evenly spaced starts are used, and "
                    "setting medium = 'air' or 'vacuum' on the star once you have checked "
                    "would seed them from a template table instead"
                )
            else:
                with ctx.stage("k-start"):
                    try:
                        table, _ = _library_table(ctx, dataset, lsf, library, "semi-amplitude")
                    except ValueError as error:
                        ctx.flag(
                            f"semi-amplitude starts skipped ({error}); the range's evenly "
                            "spaced starts are used"
                        )
                    else:
                        _write_template_table(ctx, table, "semi-amplitude start")
                        table_orbit = _table_orbit(ctx, table, star, settings)
                        starts = _k_starts(ctx, table_orbit, star, settings)
                        elements = _element_starts(ctx, table_orbit, star, settings)
        orbit_spec = _declared_orbit(star, settings, starts=starts, elements=elements)

    # 4. disentangle
    with ctx.stage("disentangle"):
        dis = _declare(
            ctx,
            dataset,
            lsf,
            orbit_spec if declared_velocities is None else None,
            declared_velocities,
        )
        log(
            f"disentangling: {dis.n_stellar} stars, grid {dis.grid.n} px, "
            f"budget {dis.velocity_budget.total:.0f} km/s, half-bandwidth "
            f"{dis.model.half_bandwidth}, {settings.max_steps} steps, noise " + _noise_words(dis)
        )
        fit = dis.fit(max_steps=settings.max_steps, k_scan=settings.k_scan)
    live["fit"] = fit
    sections.append(dis.explain())
    sections.append(fit.summary())
    report["declaration"] = _describe_declaration(dis)
    report["disentangling"] = _describe_fit(fit)
    _assess_fit(ctx, fit)
    _refuse_diverged(ctx, fit)
    if fit.k_scan is not None:
        scan = fit.k_scan
        chosen = ", ".join(f"{s.name} {k:.1f}" for s, k in zip(dis.stars, scan.best, strict=True))
        log(
            f"semi-amplitude scan: {scan.n_trials} trials, best {chosen} km/s, "
            + (
                f"{scan.gain:+.0f} nats over the start; started there"
                if scan.gain > 0.0
                else "the start was better and was kept"
            )
        )
    if fit.mode == "keplerian":
        k_text = ", ".join(f"K_{s.name} {fit.star(s.name)['k']:.2f}" for s in dis.stars)
        log(f"disentangled: P {float(fit.orbit()['period']):.5f} d, {k_text} km/s")
    else:
        log("disentangled: free per-epoch table")

    # 5. labels
    match = None
    if library is not None and star.labels:
        with ctx.stage("labels"):
            match = _labels(ctx, fit, library)
        if match is not None:
            sections.append(match.summary())
            report["labels"] = _describe_match(match)
            live["labels"] = match
            log(
                "labels: "
                + ", ".join(
                    f"{n} Teff {v['teff']:.0f} K, v_zero {v['v_kms']:+.2f} km/s"
                    for n, v in match.labels.items()
                )
            )
    elif ctx.config.library is None:
        ctx.flag(
            "labels skipped: no library declared, so the epoch velocities are differential "
            "(each component carries its own unidentified zero point)"
        )
    elif not star.labels:
        ctx.flag("labels skipped: disabled for this star, so the velocities are differential")

    # 6. epoch velocities
    with ctx.stage("velocities"):
        templates = _templates(ctx, fit, match)
        # The declared fractions are used even where the label fit's measured ones are
        # closer to the true values. The disentangled components are (w / l0) t, so l0 is
        # the amplitude that reproduces the epochs with these templates (docs/math.md §9.1).
        lights = [s.light for s in dis.stars]
        amplitudes = _template_light(ctx, fit, lights)
        light_source = "global re-measure" if isinstance(amplitudes, str) else "declared"
        table, templates = _measure_epoch_velocities(ctx, fit, templates, amplitudes)
        if isinstance(amplitudes, str) and table.light_mode == "free per epoch":
            light_source = "free per epoch"  # no epoch gave positive amplitudes to hold
        if fit.mode == "keplerian" and table.n_components == 2:
            # The orbit decides what one epoch leaves open: the order of two alike
            # components, where the light fractions allow the exchange, and the minimum at
            # the epochs flagged for a second one.
            from albireo.rvorbit import assign_by_orbit

            unexchanged = table
            table, decided = assign_by_orbit(
                table, np.asarray(fit.velocities()), exchange=_exchange_allowed(ctx, lights)
            )
            if decided.exchanged.any():
                ctx.flag(
                    f"{int(decided.exchanged.sum())} of {table.n_epochs} epochs had their "
                    "components re-assigned by the disentangling's orbit: the two spectra are "
                    "alike enough that the correlation alone could not tell them apart there"
                )
            if decided.resolved.any():
                ctx.flag(
                    f"{int(decided.resolved.sum())} of {table.n_epochs} epochs had a second "
                    "minimum of the correlation as deep as the first, and the disentangling's "
                    f"orbit decided between the two: it took the other minimum at "
                    f"{int(decided.alternative.sum())}"
                )
            if decided.changes:
                # The table as measured distinguishes a real exchange from a swap made on
                # noise, so it is kept beside the delivered one.
                ctx.directory.mkdir(parents=True, exist_ok=True)
                ctx.files["velocities_unexchanged"] = os.fspath(
                    unexchanged.write(
                        ctx.directory / "velocities_unexchanged.rv",
                        header=f"star: {ctx.star.name}\nas measured, before the orbit's decisions",
                    )
                )
    live["templates"] = templates
    live["velocities"] = table
    sections.append(table.summary())
    report["velocities"] = _describe_table(table)
    report["velocities"]["zero_points"] = ctx.zero_points
    report["velocities"]["light_source"] = light_source
    _assess_table(ctx, table)
    log(
        f"velocities: {int(table.good.sum())}/{table.n_epochs} usable epochs, median sigma "
        + ", ".join(f"{n} {np.nanmedian(table.sigma[i]):.3f}" for i, n in enumerate(table.names))
        + " km/s, "
        + ("absolute" if all(table.absolute) else "differential")
        + f", amplitudes {light_source}"
    )

    # 7. the orbit from the table
    with ctx.stage("orbit"):
        rv_orbit, period_source = _orbit(ctx, fit, table)
    if rv_orbit is not None:
        live["orbit"] = rv_orbit
        sections.append(f"Orbit from the velocity table (period from {period_source}):")
        sections.append(rv_orbit.summary())
        report["orbit"] = _describe_orbit(rv_orbit, period_source)
        _assess_orbit(ctx, fit, rv_orbit)
        log(
            f"orbit from the table: P {rv_orbit.period:.5f} d, "
            + ", ".join(
                f"K_{n} {k:.2f}+-{e:.2f}"
                for n, k, e in zip(rv_orbit.names, rv_orbit.k, rv_orbit.errors["k"], strict=True)
            )
            + " km/s"
        )
    else:
        report["orbit"] = None

    # 8. sampling, optionally
    posterior = None
    if settings.sample:
        with ctx.stage("sample"):
            posterior, match = _sample(ctx, fit, match)
        if posterior is not None:
            live["posterior"] = posterior
            sections.append(posterior.summary())
            report["posterior"] = _describe_posterior(posterior)
            if match is not None and match.draws is not None:
                sections.append("Label errors after refitting the posterior draws:")
                sections.append(match.summary())
                report["labels"] = _describe_match(match)
                live["labels"] = match

    # 9. truth, for simulations
    if star.truth:
        block, comparison = _compare_truth(ctx, fit, table, rv_orbit, match)
        sections.append(block)
        report["truth"] = comparison

    # 10. products and figures
    with ctx.stage("write"):
        _write_products(ctx, fit, table, rv_orbit, match, posterior)
    if settings.plots:
        with ctx.stage("plots"):
            _write_plots(ctx, fit, table, rv_orbit, templates, posterior)

    if ctx.flags:
        sections.append("Flags:\n" + "\n".join(f"  - {f}" for f in ctx.flags))
    if log.warnings:
        sections.append("Warnings:\n" + "\n".join(f"  - {w}" for w in log.warnings))
    return report, "\n\n".join(sections), live


# -- stages ---------------------------------------------------------------------


def _load_dataset(ctx: _Context) -> tuple[Dataset, dict[str, float]]:
    star, settings, log = ctx.star, ctx.settings, ctx.log
    header_lsf: dict[str, float] = {}
    if star.dataset is not None:
        dataset = star.dataset
    else:
        from albireo.io import dataset_from_raw, read_raw_spectra

        paths = star.spectra
        if star.bloem is not None:
            paths = _fetch_bloem(ctx)
        raws = read_raw_spectra(
            paths, instrument=star.instrument, read_kwargs=ctx.config.read_kwargs
        )
        # The widest header width per instrument is kept for the log line. The model
        # uses each epoch's own width (a PER_EPOCH declaration), not this number.
        for raw in raws:
            sigma = raw.lsf_sigma_kms
            if sigma is not None:
                header_lsf[raw.instrument] = max(float(sigma), header_lsf.get(raw.instrument, 0.0))
        log(f"read {len(raws)} files; first: {raws[0].summary()}")
        options: dict[str, Any] = {}
        if settings.region is not None:
            options["region"] = settings.region
        if settings.smooth_angstrom is not None:
            options["smooth_angstrom"] = settings.smooth_angstrom
        if settings.mask:
            options["mask"] = list(settings.mask)
        dataset = dataset_from_raw(raws, medium=star.medium, **options)
        if settings.region is None:
            n_pix = max(e.n_pixels for e in dataset)
            if n_pix > 20_000:
                ctx.flag(
                    f"no region declared: every epoch is fitted in full ({n_pix} pixels); "
                    "declare region = [lo, hi] to fit a window"
                )
    if star.medium is not None and dataset[0].medium != star.medium:
        dataset = _with_medium(dataset, star.medium)
    if star.dataset is None:
        dataset = _try_share_grid(dataset, log)
    return dataset, header_lsf


def _fetch_bloem(ctx: _Context) -> str:
    from albireo import archive

    target = ctx.star.bloem
    directory = ctx.directory / "spectra"
    ctx.log(f"resolving BLOeM {target} and fetching its public epochs into {directory}")
    star = archive.resolve_bloem(str(target))
    records = archive.bloem_spectra(star, public_only=True)
    if not records:
        raise ValueError(f"no public BLOeM spectra for {target}")
    statuses = archive.download(records, directory)
    failed = [s for s in statuses if s.startswith("FAIL")]
    if failed:
        ctx.flag(f"{len(failed)} of {len(statuses)} BLOeM downloads failed")
    ctx.log(f"BLOeM {star.bloem_id} = Gaia DR3 {star.gaia_dr3}: {len(records)} epochs")
    return os.fspath(directory / "*.fits")


def _with_medium(dataset: Dataset, medium: str) -> Dataset:
    epochs = tuple(
        EpochData(
            wave=e.wave,
            flux=e.flux,
            ivar=e.ivar,
            bjd=e.bjd,
            v_bary=e.v_bary,
            instrument=e.instrument,
            mask=e.mask,
            medium=medium,
        )
        for e in dataset
    )
    return Dataset(epochs, frame=dataset.frame)


def _try_share_grid(dataset: Dataset, log: _Log) -> Dataset:
    """Collapse sub-pixel per-exposure grids onto one, when that is exact."""
    from albireo.preprocess import share_wavelength_grid

    try:
        shared = share_wavelength_grid(list(dataset))
    except ValueError:
        return dataset
    before = len({e.wave.tobytes() for e in dataset})
    after = len({e.wave.tobytes() for e in shared})
    if after < before:
        log(f"relabelled {before} per-exposure wavelength grids onto {after} (sub-pixel, exact)")
    return Dataset(shared, frame=dataset.frame)


def _resolve_lsf(ctx: _Context, dataset: Dataset, header_lsf: Mapping[str, float]):
    out: dict[str, LSF] = {}
    for key in dataset.instruments:
        if key in ctx.star.lsf:
            out[key] = _lsf(ctx.star.lsf[key], key)
        elif key in ctx.config.lsf:
            out[key] = _lsf(ctx.config.lsf[key], key)
        elif (
            key in header_lsf
            and not np.isnan(dataset.lsf_sigma_kms[[e.instrument == key for e in dataset]]).any()
        ):
            out[key] = LSF.per_epoch()
            widths = declared_lsf_widths(dataset, key)
            detail = ", ".join(f"{s:.3f} km/s x{len(idx)}" for s, idx in widths.items())
            ctx.log(
                f"instrument {key!r}: LSF width taken per epoch from the files' own "
                f"SPEC_RES header ({detail})"
            )
        else:
            raise ValueError(
                f"no line-spread function for instrument {key!r} and its files do not all "
                f"declare a resolving power. Add [instrument.{key}] with resolving_power or "
                "sigma_kms to the configuration (or lsf={...} on the star)."
            )
    return out


_LIBRARY_CACHE: dict[str, Any] = {}


def _resolve_library(spec: Any, log: _Log):
    from albireo.library import SpectralLibrary, fetch_library, library_names, load_library

    if isinstance(spec, SpectralLibrary):
        return spec
    key = os.fspath(spec)
    if key in _LIBRARY_CACHE:
        return _LIBRARY_CACHE[key]
    if key in library_names():
        log(f"loading library {key!r} (downloaded and cached on first use)")
        library = fetch_library(key, progress=True)
    elif Path(key).is_file():
        log(f"loading library from {key}")
        library = load_library(key)
    else:
        raise ValueError(
            f"library {key!r} is neither a registered name ({library_names()}) nor a file"
        )
    _LIBRARY_CACHE[key] = library
    return library


def _ecc_omega_specs(
    star: StarConfig, settings: Analysis, elements: tuple[float, float] | None = None
) -> tuple[Spec, Spec | None]:
    """The eccentricity and omega declarations, or the settings' default range.

    With ``elements`` (an eccentricity and an argument of periastron from a velocity
    table's orbit) a free eccentricity is started there rather than at the
    ``Disentangler`` default of 0.05, so that the scans that precede the fit use a
    velocity curve of the right shape.
    """
    ecc = _spec(star.ecc, f"star {star.name!r}: ecc")
    omega = _spec(star.omega, f"star {star.name!r}: omega")
    if ecc is None:
        if settings.circular:
            ecc = Fixed(0.0)
        elif elements is not None:
            ecc = Between(0.0, settings.ecc_max, start_at=elements[0])
            omega = Fixed(elements[1]) if omega is None else omega
        else:
            ecc = Between(0.0, settings.ecc_max)
    return ecc, omega


def _declared_orbit(
    star: StarConfig,
    settings: Analysis,
    starts: Sequence[float | None] | None = None,
    elements: tuple[float, float] | None = None,
) -> Orbit:
    period = _spec(star.period, "period")
    if isinstance(period, Fixed) and not float(np.asarray(period.value)) > 0.0:
        raise ValueError(f"star {star.name!r}: the period must be positive")
    ecc, omega = _ecc_omega_specs(star, settings, elements)
    t_conj = "scan" if star.t_conj is None else _spec(star.t_conj, "t_conj")
    return Orbit(
        period=period,
        k=_k_prior(star, settings, starts=starts),
        t_conj=t_conj,
        ecc=ecc,
        omega=omega,
    )


def _table_orbit(ctx: _Context, table, star: StarConfig, settings: Analysis):
    """The orbit fitted to a template table at the declared period, or ``None``.

    The orbit is fitted at the period's central value, with exchanged epochs re-assigned
    as on the search route. ``None`` is returned when every semi-amplitude is declared
    (there is nothing to start), the period has no central value, or the table gave no
    orbit.
    """
    if all(c.k is not None for c in star.components):
        return None
    central = float(np.asarray(_spec(star.period, "period").start(), dtype=float))
    if not central > 0.0:
        return None
    try:
        orbit, _, _ = _orbit_over_candidates(ctx, table, [central], circular=settings.circular)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        ctx.log(
            "starting values: the template table gave no orbit at the declared period "
            f"({type(exc).__name__}); the range's evenly spaced starts are used"
        )
        return None
    return orbit


def _element_starts(
    ctx: _Context, orbit, star: StarConfig, settings: Analysis
) -> tuple[float, float] | None:
    """An eccentricity and argument of periastron to start a free eccentricity from.

    They are taken from the table's orbit when the star declares no eccentricity, the
    orbit is not held circular, and the fitted eccentricity is usable. A usable value is
    finite, at least 0.02 (below which the default start is as good and the argument is
    undefined), inside the prior by a margin, and detected at more than three times its
    error. A table of a dozen epochs fits an eccentricity of 0.2 to a circular pair as
    readily as not, and a scan started on that shape of curve misses a faint companion.
    The default start of 0.05 is therefore used unless the table detects the
    eccentricity. ``None`` is returned otherwise.
    """
    if orbit is None or star.ecc is not None or settings.circular:
        return None
    e = float(orbit.ecc)
    omega = float(orbit.omega)
    error = orbit.errors.get("ecc") if isinstance(orbit.errors, Mapping) else None
    error = float(error) if error is not None else float("nan")
    if not (np.isfinite(e) and np.isfinite(omega)) or e < 0.02 or e > 0.9 * settings.ecc_max:
        return None
    if not (np.isfinite(error) and error > 0.0 and e > 3.0 * error):
        ctx.log(
            f"eccentricity from the template table's orbit not used: e {e:.3f} +- "
            f"{error:.3f} is not detected at three sigma; the default start applies"
        )
        return None
    ctx.log(
        f"eccentricity started from the template table's orbit: e {e:.3f} +- {error:.3f}, "
        f"omega {omega:.3f} rad"
    )
    return e, omega


def _k_starts(
    ctx: _Context, orbit, star: StarConfig, settings: Analysis
) -> list[float | None] | None:
    """Semi-amplitude starting values from a table's orbit (:func:`_table_orbit`).

    A component whose fitted semi-amplitude falls outside the declared range is reported
    as ``None`` and :func:`_k_prior` starts it from the others. ``None`` is returned when
    there is no orbit.
    """
    if orbit is None:
        return None
    starts: list[float | None] = []
    for k in np.asarray(orbit.k, dtype=float):
        usable = bool(np.isfinite(k)) and settings.k_min <= float(k) <= settings.k_max
        starts.append(float(k) if usable else None)
    ctx.log(
        "semi-amplitudes started from the template table at the declared period: "
        + ", ".join(
            f"{c.name} " + ("unusable" if s is None else f"{s:.1f} km/s")
            for c, s in zip(star.components, starts, strict=True)
        )
    )
    return starts


def _k_prior(
    star: StarConfig, settings: Analysis, starts: Sequence[float | None] | None = None
) -> list[Spec]:
    """One semi-amplitude prior per component, started in the declared order.

    A symmetric prior does not assign the spectra to the stars. With every component
    started at the same semi-amplitude the conjunction scan has two equally deep minima
    (the declared assignment and its mirror, with the spectra swapped and rescaled by the
    light ratio), and L-BFGS converges to whichever it started in. The data cannot
    distinguish the two, since only ``l_i * d_i`` is observable, so a convention is used.
    Components are declared in order of decreasing mass, the first star moves least, and
    the fit is started with ``K_1 < K_2 < ...`` at evenly spaced points of the shared
    range, which the scan then discriminates on. The ordering is a starting point, not a
    constraint (the bounds are the same for every component), and the label stage checks
    the outcome: a fitted light fraction far from the declared one is the signature of a
    reversed order.

    With ``starts`` (one entry per component, ``None`` where no table could measure the
    star) each undeclared component starts at its measured value: the semi-amplitudes of
    an orbit fitted to a library-template table at the declared period, or a bootstrap's.
    A component without a usable entry starts from the nearest measured one, in the
    direction the declared order implies: a later-declared (lighter) star at 1.5 times its
    neighbour's semi-amplitude, an earlier-declared (heavier) one at two thirds of it.
    With nothing measured, it starts at its evenly spaced point. Every start is held
    strictly inside the range, since a start on a bound has no valid initial parameters.
    A Gaia pair whose synchronised primary the table could not measure started at the
    250 km/s bound, and the run failed there. The evenly spaced points alone are not
    enough. Over a range of 2 to 250 km/s they start a 23 km/s primary at 85 km/s. From
    that start a Gaia RVS simulation converged to the static-component minimum, with both
    semi-amplitudes at the lower bound, on a pair whose secondary has 5 percent of the
    light.
    """
    n = len(star.components)
    declared = [_spec(c.k, f"component {c.name!r}: k") for c in star.components]
    seeds = list(starts) if starts is not None else [None] * n
    if len(seeds) != n:
        raise ValueError(f"{len(seeds)} semi-amplitude starts for {n} components")
    lo, hi = float(settings.k_min), float(settings.k_max)
    inside = (lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo))

    def usable(value) -> bool:
        return value is not None and bool(np.isfinite(value)) and lo <= float(value) <= hi

    measured = {j: float(s) for j, s in enumerate(seeds) if usable(s)}
    out: list[Spec] = []
    for i, spec in enumerate(declared):
        if spec is not None:
            out.append(spec)
            continue
        if i in measured:
            start = measured[i]
        elif measured:
            # The nearest measured component sets the scale, and the declared order the
            # direction: a later (lighter) star moves more, an earlier (heavier) one less.
            j = min(measured, key=lambda j: (abs(j - i), j))
            start = measured[j] * 1.5 if j < i else measured[j] / 1.5
        else:
            start = lo + (hi - lo) * (i + 1) / (n + 1)
        start = float(min(max(start, inside[0]), inside[1]))
        out.append(Between(lo, hi, start_at=start))
    return out


def _components(star: StarConfig, settings: Analysis) -> list:
    components: list[Any] = []
    for c in star.components:
        smooth = Smoothness() if c.smoothness_tau0 is None else Smoothness(tau0=c.smoothness_tau0)
        components.append(Star(c.name, light=c.light, smoothness=smooth))
    if settings.telluric:
        components.append(Telluric())
    if settings.nebular:
        components.append(Nebular(v_kms=settings.nebular_v_kms))
    return components


def _declare(ctx: _Context, dataset: Dataset, lsf, orbit: Orbit | None, velocities):
    settings = ctx.settings
    return Disentangler(
        dataset,
        components=_components(ctx.star, settings),
        orbit=orbit,
        velocities=velocities,
        lsf=lsf,
        dv_kms=settings.dv_kms,
        ecc_max=max(settings.ecc_max, 0.05) if not settings.circular else 0.95,
        noise_correlation=_noise_declaration(settings, dataset),
    )


def _noise_declaration(settings: Analysis, dataset: Dataset):
    """The ``noise_correlation`` argument from the settings: values per instrument, or a site."""
    declared = settings.noise_correlation
    if declared is None:
        return None
    if isinstance(declared, str):
        return Between(-0.9, 0.9, start_at=0.0)
    if isinstance(declared, Mapping):
        return {name: float(declared.get(name, 0.0)) for name in dataset.instruments}
    return float(declared)


def _noise_values(settings: Analysis, dataset: Dataset) -> dict[str, float] | None:
    """The declared correlation per instrument for a template table, or ``None`` when fitted."""
    declared = _noise_declaration(settings, dataset)
    if declared is None or isinstance(declared, Spec):
        return None
    if isinstance(declared, Mapping):
        return dict(declared)
    return dict.fromkeys(dataset.instruments, float(declared))


def _read_velocities(star: StarConfig, dataset: Dataset) -> np.ndarray:
    """A declared velocity table, ``(n_stellar, n_epochs)``, matched to the epochs."""
    n = len(star.components)
    if isinstance(star.velocities, str | os.PathLike):
        table = np.loadtxt(star.velocities, ndmin=2, comments="#")
    else:
        table = np.asarray(star.velocities, dtype=float)
        if table.shape == (n, dataset.n_epochs):
            return _checked_velocities(star, dataset, table)
        if table.shape == (dataset.n_epochs, n):
            return _checked_velocities(star, dataset, table.T)
    if table.ndim != 2:
        raise ValueError(f"star {star.name!r}: the velocity table must be two-dimensional")
    if table.shape[1] == n:
        if table.shape[0] != dataset.n_epochs:
            raise ValueError(
                f"star {star.name!r}: {table.shape[0]} velocity rows for "
                f"{dataset.n_epochs} epochs; add a leading BJD column to match by time"
            )
        return _checked_velocities(star, dataset, table.T)
    if table.shape[1] == n + 1:
        bjd = table[:, 0]
        out = np.empty((n, dataset.n_epochs))
        for j, t in enumerate(dataset.bjd):
            k = int(np.argmin(np.abs(bjd - t)))
            if abs(bjd[k] - t) > 0.02:
                raise ValueError(
                    f"star {star.name!r}: no declared velocity within 0.02 d of epoch "
                    f"{j} (BJD {t:.5f}); nearest is {bjd[k]:.5f}"
                )
            out[:, j] = table[k, 1:]
        return _checked_velocities(star, dataset, out)
    raise ValueError(
        f"star {star.name!r}: the velocity table has {table.shape[1]} columns; expected one "
        f"per component ({n}) or BJD plus one per component ({n + 1})"
    )


def _checked_velocities(star: StarConfig, dataset: Dataset, values: np.ndarray) -> np.ndarray:
    """Reject a declared table that leaves an epoch unmeasured, naming it.

    The free-velocity fit has one velocity site per component per epoch and no site for
    an epoch that was not measured, so a non-finite entry cannot be passed to it. The
    correlation stage writes ``nan`` where it measured nothing, and this is the route
    that reads such a file back.
    """
    values = np.asarray(values, dtype=float)
    bad = np.argwhere(~np.isfinite(values))
    if bad.size:
        i, j = (int(k) for k in bad[0])
        others = (
            ""
            if len(bad) == 1
            else f", and {len(bad) - 1} further entr{'y' if len(bad) == 2 else 'ies'}"
        )
        raise ValueError(
            f"star {star.name!r}: the declared velocity of component "
            f"{star.components[i].name!r} at epoch {j} (BJD {float(dataset.bjd[j]):.5f}) is "
            f"not a number{others}. The free-velocity fit has one site per component per "
            "epoch and no site for an unmeasured one: remove that epoch from the dataset, "
            "or measure it."
        )
    return values


def _library_table(ctx: _Context, dataset: Dataset, lsf, library, purpose: str):
    """A velocity table from library templates at the declared starting labels.

    The ``period = "search"`` bootstrap and the ``light = "measure"`` stage share this
    function. The templates are rendered on a grid covering the search, and the epochs
    are correlated against them with free-then-held light fractions.

    Where no epoch gives a positive amplitude to every template, the correlation measures
    neither the light fractions nor the companion, and an error says so. Two templates
    cooler than both stars fit a pair that is never resolved as a difference of the two
    at every epoch (one of the 33 benchmark systems with the templates of the blind tier,
    ``docs/benchmarks.md``).
    """
    from albireo.todcor import Template, _LightNotMeasured, todcor

    star, settings, log = ctx.star, ctx.settings, ctx.log
    medium = dataset[0].medium
    if medium is None:
        raise ValueError(
            f"star {star.name!r}: {purpose} renders library templates, which needs "
            "the wavelength medium; the files did not declare one, so set medium = 'air' "
            "(or 'vacuum') on the star once you have checked which it is"
        )
    resolved = [v.widths(dataset, k) for k, v in lsf.items() if k in dataset.instruments]
    narrowest = min(float(np.min(w)) for w in resolved)
    widest = max(float(np.max(w)) for w in resolved)
    grid = LogGrid.covering(
        dataset,
        max(narrowest / 3.0, 0.2),
        v_margin_kms=settings.v_range + 60.0,
        lsf_sigma_kms=widest,
    )
    # The library is converted to the data's scale before slicing. Air and vacuum differ
    # by about 2.3 A here, more than the pad, and a slice taken in the library's own scale
    # would not cover the grid once the templates are rendered on the data's.
    lib = library.in_medium(medium).sliced(
        grid.wave[0] - _LIBRARY_PAD_ANGSTROM, grid.wave[-1] + _LIBRARY_PAD_ANGSTROM
    )
    resolving = lib.meta.get("resolution")
    starts = [c.labels_start() for c in star.components]
    label_sets = []
    for start in starts:
        labels = {}
        for axis in lib.label_names:
            if axis in start:
                labels[axis] = start[axis]
            elif axis == "mh":
                mh = _spec(ctx.config.mh, "mh")
                labels[axis] = (
                    float(np.asarray(mh.start()))
                    if mh is not None
                    else float(np.mean(lib.bounds["mh"]))
                )
            else:
                labels[axis] = float(np.mean(lib.bounds[axis]))
        label_sets.append(labels)
    if (
        len(label_sets) > 1
        and "teff" in lib.label_names
        and not any("teff" in start for start in starts)
    ):
        # With no temperature prior every component would get the same template, the box
        # midpoint, and identical templates coincide at equal shifts, where the correlation
        # cannot distinguish the components. The components are declared in order of
        # decreasing mass, so the starts are spread across the box, hotter first.
        lo, hi = lib.bounds["teff"]
        n = len(label_sets)
        for i, labels in enumerate(label_sets):
            labels["teff"] = float(lo + (hi - lo) * (n - i) / (n + 1))
        log("no temperature priors: template starts spread across the library box, hotter first")
    templates = []
    for c, start, labels in zip(star.components, starts, label_sets, strict=True):
        vsini = start.get("vsini", 0.0)
        templates.append(
            Template.from_library(
                c.name,
                lib,
                labels,
                grid=grid,
                medium=medium,
                vsini_kms=vsini,
                resolving_power=float(resolving) if resolving else None,
            )
        )
        log(
            f"bootstrap template {c.name}: "
            + ", ".join(f"{k} {v:g}" for k, v in labels.items())
            + f", vsini {vsini:g} km/s"
        )
    lsf_sigma = {k: v.sigma_kms for k, v in lsf.items()}
    anchors = {k: v.anchors_angstrom for k, v in lsf.items() if v.anchors_angstrom is not None}
    try:
        table = todcor(
            dataset,
            templates,
            v_range=(-settings.v_range, settings.v_range),
            light="global",
            lsf_sigma_v=lsf_sigma,
            lsf_anchors_angstrom=anchors or None,
            noise_correlation=_noise_values(settings, dataset),
        )
    except _LightNotMeasured as error:
        raise ValueError(
            f"star {star.name!r}: {purpose}: the correlation against library templates at "
            "the starting labels gives no epoch with a positive amplitude for every "
            "component, so it measures neither the light fractions nor every component's "
            "velocity. The templates do not describe the components: declare starting "
            "temperatures closer to the stars. A star with a declared period and declared "
            "light fractions needs no template table"
        ) from error
    log(
        f"{purpose} velocities: {int(table.good.sum())}/{table.n_epochs} usable epochs "
        f"(library templates, absolute)"
    )
    return table, templates


def _table_light_fractions(table) -> np.ndarray | None:
    """The light fractions a table's amplitudes imply, or ``None`` with no usable epoch.

    Each is the median over the usable epochs of the component's share of the summed
    amplitudes, renormalized to sum to one. With the bootstrap's ``light="global"`` every
    epoch has the same amplitudes, so the median is that one pair.
    """
    light = np.asarray(table.light, dtype=float)
    usable = table.good & np.all(np.isfinite(light), axis=0)
    if not usable.any():
        return None
    fractions = light[:, usable] / light[:, usable].sum(axis=0, keepdims=True)
    measured = np.median(fractions, axis=1)
    return measured / measured.sum()


def _apply_measured_light(ctx: _Context, table) -> np.ndarray:
    """Replace the star's ``"measure"`` fractions by the table's held amplitudes."""
    star = ctx.star
    measured = _table_light_fractions(table)
    if measured is None:
        raise ValueError(f"star {star.name!r}: no usable epoch to measure the light fractions from")
    if np.any(measured <= 0.0):
        raise ValueError(
            f"star {star.name!r}: the correlation measured a non-positive light fraction "
            f"({np.round(measured, 3).tolist()}); declare the fractions instead"
        )
    ctx.star = replace(
        star,
        components=[
            replace(c, light=float(l)) for c, l in zip(star.components, measured, strict=True)
        ],
    )
    ctx.flag(
        "light fractions measured by correlation against library templates rather than "
        "declared: "
        + ", ".join(f"{c.name} {l:.3f}" for c, l in zip(star.components, measured, strict=True))
        + " (the amplitudes the templates received; a template mismatch moves them)"
    )
    return measured


def _measure_light(ctx: _Context, dataset: Dataset, lsf, library) -> dict[str, Any]:
    """The ``light = "measure"`` stage on a route with a declared period."""
    table, templates = _library_table(ctx, dataset, lsf, library, "light = 'measure'")
    measured = _apply_measured_light(ctx, table)
    text = (
        "Light fractions measured by correlation against library templates:\n"
        + table.summary()
        + "\n  -> declared for the disentangling: "
        + ", ".join(f"{c.name} {l:.3f}" for c, l in zip(ctx.star.components, measured, strict=True))
    )
    return {
        "text": text,
        "report": {
            "light_measured": {
                c.name: float(l) for c, l in zip(ctx.star.components, measured, strict=True)
            },
            "n_usable": int(table.good.sum()),
            "templates": [t.meta for t in templates],
        },
        "objects": {"table": table, "templates": templates},
    }


def _bootstrap_spec(
    ctx: _Context,
    orbit,
    *,
    comparison: bool = False,
    k_starts: Sequence[float] | None = None,
) -> tuple[Orbit, str]:
    """The disentangling's orbit declaration built from one bootstrap orbit.

    The two paths of :func:`_bootstrap` share this function: the declaration made from
    the chosen orbit (``comparison=False``), and the declaration each candidate period is
    compared on (``comparison=True``). In both, the period is the fitted one to within 3
    percent, and the eccentricity and argument of periastron are started from the table's
    orbit wherever it measured them at three sigma (:func:`_element_starts`).

    The two differ in the semi-amplitudes and the conjunction. Both differences make the
    candidates comparable with each other. The comparison declares every semi-amplitude
    as the settings' range started at the table's value, where the final declaration
    gives a usable table semi-amplitude a Gaussian prior centred on it. The velocity
    budget is the sum of the semi-amplitude priors' upper bounds and fixes the model
    grid's extent, so Gaussians centred on 233 km/s and on 40 km/s would put two
    candidates on grids of different length, on which marginal likelihoods are not
    comparable. The comparison also scans the conjunction rather than holding it at the
    table's, which is as uncertain as the table's period, unless the star declares one (a
    measurement that is valid at every candidate period).

    A semi-amplitude the table left outside the declared range is not a measurement, so
    the final declaration uses the range there too (the flag records this once).
    ``k_starts``, the semi-amplitudes selected by the chosen candidate's scan, then
    replace the table's as the starting values. They are ignored where the table's
    semi-amplitudes are kept, since the Gaussian around them is a prior and not a start.

    Returns the declaration and the phrase describing its semi-amplitudes.
    """
    star, settings = ctx.star, ctx.settings
    k_boot = np.asarray(orbit.k, dtype=float)
    degenerate = ~np.isfinite(k_boot) | (k_boot > settings.k_max) | (k_boot < settings.k_min)
    starts: list[float | None] = [
        None if d else float(k) for k, d in zip(k_boot, degenerate, strict=True)
    ]
    if comparison:
        k_spec = _k_prior(star, settings, starts=starts)
        k_text = f"K within {settings.k_min:g}-{settings.k_max:g}, started from the table"
    elif degenerate.any():
        # A component the correlation could not follow (a faint secondary, an exchanged
        # twin) gives a semi-amplitude of zero or of thousands of km/s, and a Gaussian
        # prior would hold the disentangling to it. The period and the conjunction are kept
        # and the semi-amplitudes revert to the declared range, as on the known-period route.
        names = [c.name for c, d in zip(star.components, degenerate, strict=True) if d]
        ctx.flag(
            f"bootstrap semi-amplitudes {np.round(k_boot, 1).tolist()} km/s fall outside the "
            f"declared range {settings.k_min:g}-{settings.k_max:g} for {names}: the "
            "disentangling searches the range instead, from the bootstrap's period"
        )
        if k_starts is not None:
            starts = [float(k) for k in k_starts]
        k_spec = _k_prior(star, settings, starts=starts)
        k_text = (
            f"K within {settings.k_min:g}-{settings.k_max:g}, started from the bootstrap "
            "where it measured a semi-amplitude"
        )
    else:
        k_sigma = np.maximum(3.0 * np.asarray(orbit.errors["k"]), 0.15 * k_boot)
        k_sigma = np.where(np.isfinite(k_sigma), k_sigma, 0.3 * k_boot)
        k_sigma = np.minimum(k_sigma, 0.3 * settings.k_max)
        k_spec = Known(k_boot, np.asarray(k_sigma, dtype=float))
        k_text = f"K Gaussian with sigma {np.round(k_sigma, 2).tolist()} km/s"
    if star.t_conj is not None:
        t_conj: Any = _spec(star.t_conj, "t_conj")
    elif comparison:
        t_conj = "scan"
    else:
        t_conj = Known(float(orbit.t_conj), 0.05 * orbit.period)
    ecc_spec, omega_spec = _ecc_omega_specs(
        star, settings, _element_starts(ctx, orbit, star, settings)
    )
    width = 0.03 * orbit.period
    spec = Orbit(
        period=Between(orbit.period - width, orbit.period + width),
        k=k_spec,
        t_conj=t_conj,
        ecc=ecc_spec,
        omega=omega_spec,
    )
    return spec, k_text


def _coarse_marginal(scanner: Disentangler, init: Mapping[str, Any]) -> float:
    """The coarse declaration's marginal log-likelihood at ``init``, in nats.

    This is the fallback for a candidate whose declaration has nothing to scan (a declared
    conjunction and a declared semi-amplitude for every component). Like the scans, it
    includes no prior: :meth:`albireo.inference.MarginalOrbitModel.log_likelihood` is the
    marginal of the data over the spectra alone, and the priors over the orbital
    parameters enter only through the numpyro model the optimizer runs.
    """
    theta = {**dict(init), **dict(scanner.fixed)}
    return float(scanner.model.log_likelihood(theta))


def _decide_period_by_disentangling(ctx: _Context, dataset: Dataset, lsf, ranked) -> dict | None:
    """Decide among the best few candidate periods by the disentangling's likelihood.

    The velocity table's chi-square cannot separate the aliases of a poor table. On two
    blind Gaia systems of the D62 population, an eccentric Keplerian at 0.2485 d with
    semi-amplitudes of 233 and 239 km/s at e = 0.73 fitted a fifteen-epoch table better
    than the true 6.104 d, and one at 1.1293 d with 182 and 185 km/s at e = 0.72 fitted a
    twelve-epoch table better than the true 5.356 d. Both orbits lie inside the declared
    ranges, so the range filter does not exclude them. Both were passed to the
    disentangling as a period known to 3 percent, which is not recoverable. A
    five-parameter Keplerian fitted to a dozen noisy velocities has that freedom; the
    disentangling does not. Its marginal likelihood uses every pixel of every epoch, the
    spectra are shared across the epochs, and a wrong period has to explain the whole
    dataset with one pair of component spectra.

    The top ``period_decision_candidates`` orbits of the chi-square ranking are therefore
    compared here. Each is declared as :func:`_bootstrap_spec` declares it for a
    comparison, on the same coarse grid that the ``Disentangler`` scans use
    (:meth:`albireo.Disentangler._scan_declaration`, twice the pixel, a quarter to an
    eighth of the full model's cost). Its value is the best marginal log-likelihood the
    coarse declaration reaches: the conjunction-phase scan over one period, then, where a
    semi-amplitude is a range, the coarse semi-amplitude scan, whose best trial is taken
    when it is better than the start. The values are comparable because every candidate
    is evaluated on the same data with the same noise model, the same declaration-wide
    semi-amplitude bounds and therefore the same model grid. The marginal log-likelihood
    also includes no prior (:func:`_coarse_marginal`), so a period is not favoured for
    agreeing with a prior.

    The prior-amplitude profile and the second scan pass of :meth:`albireo.Disentangler.fit`
    are not run. They refine a fit within a basin, and this comparison only selects the
    basin. Each candidate's declaration, which holds a compiled model, is released once
    its value is known.

    ``None`` is returned when the comparison does not apply: fewer than two distinct
    candidates, or ``period_decision_candidates`` below two.
    """
    settings, log = ctx.settings, ctx.log
    n = min(int(settings.period_decision_candidates), len(ranked))
    if settings.period_decision_candidates < 2 or n < 2:
        return None
    log(
        f"deciding among the best {n} of {len(ranked)} candidate periods by the "
        "disentangling's marginal likelihood on the coarse grid"
    )
    entries: list[dict[str, Any]] = []
    for rank, (orbit, _) in enumerate(ranked[:n]):
        started = time.perf_counter()
        log(
            f"  candidate {rank + 1}/{n}: P {orbit.period:.5f} d, table chi2 "
            f"{float(orbit.chi2):.1f}, table K {np.round(orbit.k, 1).tolist()} km/s"
        )
        spec, _ = _bootstrap_spec(ctx, orbit, comparison=True)
        dis = _declare(ctx, dataset, lsf, spec, None)
        scanner = dis._scan_declaration()
        init = dict(dis.init)
        scan = None
        if spec.t_conj == "scan":
            scan = scanner._scan_phase(init)
            init["t_conj"] = scan.best
        amp = scanner._scan_semi_amplitudes(init) if dis._has_ranged_k() else None
        if amp is not None and float(amp.gain) > 0.0:
            value = float(amp.best_value)
            k_best = [float(v) for v in np.atleast_1d(np.asarray(amp.best, dtype=float))]
            t_conj = float(amp.best_t_conj)
            moved = True
        else:
            # This is the value at the best phase for the starting semi-amplitudes, the
            # number the semi-amplitude scan reports as its start value and the only one
            # available when there is no semi-amplitude to scan.
            value = (
                float(np.max(np.asarray(scan.values, dtype=float)))
                if scan is not None
                else _coarse_marginal(scanner, init)
            )
            k_best = [float(v) for v in np.atleast_1d(np.asarray(init["k"], dtype=float))]
            declared_t = init.get("t_conj", dis.fixed.get("t_conj", np.nan))
            t_conj = float(np.asarray(declared_t, dtype=float))
            moved = False
        n_trials = (0 if scan is None else int(np.size(scan.values))) + (
            0 if amp is None else int(amp.n_trials)
        )
        seconds = time.perf_counter() - started
        entries.append(
            {
                "period": float(orbit.period),
                "table_chi2": float(orbit.chi2),
                "value_nats": value,
                "k": [float(k) for k in k_best],
                "t_conj": t_conj,
                "scan_moved": moved,
                "n_trials": n_trials,
                "seconds": float(seconds),
            }
        )
        log(
            f"    -> marginal {value:.1f} nats at K {[round(k, 1) for k in k_best]} km/s, "
            f"t_conj {t_conj:.4f} ({n_trials} trials, {seconds:.0f} s"
            + ("; the scan moved the start)" if moved else "; the start was kept)")
        )
        del scanner, dis, amp, scan
        gc.collect()
    order = sorted(range(len(entries)), key=lambda i: entries[i]["value_nats"], reverse=True)
    chosen, runner_up = order[0], order[1]
    margin = entries[chosen]["value_nats"] - entries[runner_up]["value_nats"]
    over_table = entries[chosen]["value_nats"] - entries[0]["value_nats"]
    log(
        f"period decided by the disentangling: P {entries[chosen]['period']:.5f} d, "
        f"{margin:.0f} nats over the runner-up P {entries[runner_up]['period']:.5f} d "
        + ("(the table's choice stands)" if chosen == 0 else "(the table had preferred another)")
    )
    if chosen != 0:
        ctx.flag(
            f"the table preferred P {entries[0]['period']:.5f} d at chi2 "
            f"{entries[0]['table_chi2']:.1f}; the disentangling prefers P "
            f"{entries[chosen]['period']:.5f} d by {over_table:.0f} nats of marginal "
            "likelihood, and that period is the one the fit is started from. The table's "
            "chi-square is fitted to a dozen velocities with five free parameters; the "
            "marginal likelihood uses every pixel of every epoch"
        )
    return {
        "index": chosen,
        "entries": entries,
        "margin_nats": float(margin),
        "over_table_nats": float(over_table),
    }


def _describe_period_decision(decision: dict | None, *, period: float, reason: str) -> dict:
    """The ``bootstrap.decision`` report block: how the period was chosen, and from what.

    ``by`` is ``"disentangling"`` when the comparison overruled the velocity table's
    chi-square and ``"table"`` when the table's choice was kept, either because the
    disentangling confirmed it or because the comparison did not run, which ``reason``
    then states. ``margin_nats`` is the chosen candidate's margin over the second-best in
    nats of marginal log-likelihood, and ``over_table_nats`` its margin over the table's
    choice, which is zero when the two agree.
    """
    if decision is None:
        return {
            "by": "table",
            "candidates": [],
            "chosen_period": float(period),
            "margin_nats": None,
            "over_table_nats": None,
            "table_period": float(period),
            "reason": reason,
        }
    entries, chosen = decision["entries"], decision["index"]
    return {
        "by": "table" if chosen == 0 else "disentangling",
        "candidates": [{**entry, "chosen": i == chosen} for i, entry in enumerate(entries)],
        "chosen_period": float(entries[chosen]["period"]),
        "margin_nats": float(decision["margin_nats"]),
        "over_table_nats": float(decision["over_table_nats"]),
        "table_period": float(entries[0]["period"]),
        "reason": reason,
    }


def _bootstrap(ctx: _Context, dataset: Dataset, lsf, library):
    """Bootstrap the orbit from library templates for the ``period = "search"`` route.

    Templates at the declared starting labels measure a velocity table, four periodograms
    propose candidate periods (:func:`_period_candidates`), an orbit is fitted from each,
    and the chi-square ranks them. The best few are then compared by the disentangling
    (:func:`_decide_period_by_disentangling`), because a dozen-epoch velocity table does
    not decide between a period and its aliases. The chosen orbit becomes the warm
    Keplerian prior for the fit. The report block records the peak of each search, how
    many candidates were fitted, every other fitted period the chi-square cannot separate
    from the chosen one, and, under ``decision``, the value of each compared candidate.
    The returned objects include the table that goes with the chosen orbit and, under
    ``table_unexchanged``, the table the period search ran on. The two differ at every
    epoch the chosen orbit re-assigned.

    Three rules of D65 act on the search and not on the table. A companion's velocity whose
    detection statistic is below ``detection_min`` has no weight in the period search or
    the candidate fits (:func:`_detection_gate`), and the report counts those velocities per
    component under ``detection_gate``. The candidate fits exchange the two components only
    where the light fractions of the table's amplitudes allow it, by the rule the velocities
    measured after the disentangling follow (:func:`_exchange_allowed`). A table whose
    usable epochs fall on fewer than ``_FEW_NIGHTS`` nights is flagged
    (:func:`_usable_nights`), since its velocities cannot be expected to decide the period.
    """
    star, settings, log = ctx.star, ctx.settings, ctx.log
    table, templates = _library_table(ctx, dataset, lsf, library, "bootstrap")
    unexchanged = table  # the table the period search runs on, before any re-assignment
    detection_min = float(settings.detection_min)
    gated = _detection_mask(table, detection_min)
    if detection_min > 0.0:
        log(
            f"detection gate at {detection_min:g}: "
            + ", ".join(
                f"{name} {int(n)} of {int(np.isfinite(v).sum())} measured velocities"
                for name, n, v in zip(table.names, gated.sum(axis=1), table.velocity, strict=True)
            )
            + " carry no weight in the period search and the candidate fits"
        )
    n_usable = int(np.sum(table.good))
    nights = _usable_nights(table)
    if nights < _FEW_NIGHTS:
        ctx.flag(
            f"the bootstrap table's {n_usable} usable epochs fall on {nights} night(s), fewer "
            f"than {_FEW_NIGHTS}: a table this sparse in phase cannot be expected to decide the "
            "period, and the one the search chose is a candidate rather than a measurement"
        )
    lights = _table_light_fractions(table)
    exchange = (
        table.n_components != 2
        or lights is None
        or _exchange_allowed(ctx, lights, where="in the bootstrap's candidate fits")
    )
    candidates, searches = _period_candidates(
        table, swap_invariant=True, detection_min=detection_min, leave_one_out=True
    )
    search = searches["single"]
    harmonic = searches["harmonic"]
    orbit, table, record = _orbit_over_candidates(
        ctx,
        table,
        candidates,
        circular=settings.circular,
        detection_min=detection_min,
        exchange=exchange,
    )
    ranked = record["ranked"]
    # The light fractions, when they are measured rather than declared, must be available
    # before any declaration can be built. They are measured from the table the chi-square
    # chose, and again from the chosen one when the comparison selects another.
    measured_light = _apply_measured_light(ctx, table) if star.measures_light else None
    reason = (
        "period_decision_candidates disabled"
        if settings.period_decision_candidates < 2
        else f"{len(ranked)} distinct candidate orbit(s), fewer than the two a comparison needs"
    )
    decision = _decide_period_by_disentangling(ctx, dataset, lsf, ranked)
    k_starts = None
    if decision is not None:
        reason = (
            f"top {len(decision['entries'])} of {len(ranked)} candidates compared on the "
            "coarse declaration"
        )
        chosen = decision["index"]
        orbit, chosen_table = ranked[chosen]
        if decision["entries"][chosen]["scan_moved"]:
            k_starts = decision["entries"][chosen]["k"]
        if chosen_table is not table:
            table = chosen_table
            if star.measures_light:
                measured_light = _apply_measured_light(ctx, table)
    swapped = int(table.settings.get("reassigned_by_orbit", 0))
    if swapped:
        ctx.flag(
            f"bootstrap: {swapped} of {table.n_epochs} epochs had their components "
            "re-assigned by the orbit fitted to the table; alike components are exchanged "
            "at random by a per-epoch correlation"
        )
    resolved = int(table.settings.get("resolved_by_orbit", 0))
    if resolved:
        ctx.flag(
            f"bootstrap: {resolved} of {table.n_epochs} epochs had a second minimum of the "
            "correlation as deep as the first, and the orbit fitted to the table decided "
            f"between the two: it took the other minimum at "
            f"{int(table.settings.get('second_minimum_by_orbit', 0))}"
        )
    if search is None:
        # Every relative velocity was gated or unmeasured below four epochs. The first
        # component's own search and the candidates it proposed remain.
        peak_text = "no relative-velocity search: too few epochs measured both components"
        search_text = "none on the relative velocity (too few epochs measured both components)"
    else:
        peak_text = (
            f"periodogram peak {search['period']:.4f}, "
            f"aliases {[round(p, 3) for p in search['aliases'][:3]]}"
        )
        search_text = (
            f"{search['period']:.5f} d, aliases {[round(p, 4) for p in search['aliases'][:4]]}"
        )
    n_leave_one_out = len(searches.get("leave_one_out") or [])
    log(
        f"bootstrap orbit: P {orbit.period:.5f} d ({peak_text}; "
        f"{record['n_candidates']} candidates, "
        + ("the disentangling decided" if decision is not None else "the orbit fit decided")
        + f"), K {np.round(orbit.k, 2).tolist()} km/s"
    )
    spec, k_text = _bootstrap_spec(ctx, orbit, k_starts=k_starts)
    ambiguous = record["ambiguous"]
    if ambiguous:
        ambiguity = "; ".join(f"{a['period']:.4f} d at +{a['delta_chi2']:.1f}" for a in ambiguous)
    else:
        ambiguity = "none within delta chi2 25"
    described = _describe_period_decision(decision, period=float(orbit.period), reason=reason)
    if decision is None:
        decided = "  the lowest chi-square candidate is the period; no comparison ran\n"
    else:
        decided = (
            f"  {len(decision['entries'])} of them compared by the disentangling's marginal "
            f"likelihood on the coarse grid: {described['chosen_period']:.5f} d, "
            f"{described['margin_nats']:.0f} nats over the runner-up"
            + (
                "; the table's chi-square had preferred "
                f"{described['table_period']:.5f} d, by {described['over_table_nats']:.0f} "
                "nats the worse of the two\n"
                if described["by"] == "disentangling"
                else " (the table's choice confirmed)\n"
            )
        )
    text = (
        "Bootstrap from library templates (period = 'search'):\n"
        f"  templates at the declared starting labels; period search {search_text}"
        + (f"; two-harmonic search {harmonic['period']:.5f} d" if harmonic is not None else "")
        + (
            f"; {n_leave_one_out} further starts from the leave-one-epoch-out searches"
            if n_leave_one_out
            else ""
        )
        + (
            f"\n  detection gate at {detection_min:g}: "
            + ", ".join(
                f"{name} {int(n)}"
                for name, n in zip(unexchanged.names, gated.sum(axis=1), strict=True)
            )
            + " velocities without weight in the search"
            if detection_min > 0.0
            else ""
        )
        + f"\n  {record['n_candidates']} candidate periods fitted; "
        f"other fitted periods within reach: {ambiguity}\n"
        + decided
        + table.summary()
        + "\n"
        + orbit.summary()
        + "\n  -> disentangling prior: period within +-3%, "
        + k_text
        + ", t_conj Gaussian"
    )
    return spec, {
        "text": text,
        "report": {
            "period": float(orbit.period),
            "periodogram_peak": None if search is None else float(search["period"]),
            "harmonic_peak": None if harmonic is None else float(harmonic["period"]),
            "aliases": [] if search is None else [float(p) for p in search["aliases"]],
            "n_leave_one_out": n_leave_one_out,
            "detection_gate": {
                "threshold": detection_min,
                "gated": {
                    name: int(n)
                    for name, n in zip(unexchanged.names, gated.sum(axis=1), strict=True)
                },
            },
            "n_nights": nights,
            "exchange_allowed": bool(exchange),
            "n_candidates": int(record["n_candidates"]),
            "ambiguous": ambiguous,
            "decision": described,
            "k": {n: float(k) for n, k in zip(orbit.names, orbit.k, strict=True)},
            "t_conj": float(orbit.t_conj),
            "ecc": float(orbit.ecc),
            "n_usable": int(table.good.sum()),
            "templates": [t.meta for t in templates],
            "light_measured": None
            if measured_light is None
            else {c.name: float(l) for c, l in zip(star.components, measured_light, strict=True)},
        },
        "objects": {
            "table": table,
            "table_unexchanged": unexchanged,
            "orbit": orbit,
            "templates": templates,
        },
    }


# The number of peaks taken from each periodogram whose peak list grows with it: fifty rather
# than the `find_period` default of twenty, on a measurement (D64). Over the 33 blind systems of
# the third run, fifty puts the true period at the top of the chi-square ranking on one further
# system and no system loses its place. The candidate fitting takes twice as long (30 to 58
# seconds a star on the Gaia population, 40 to 80 on the field, against some 670 seconds a star
# overall). A hundred was measured too and gains nothing beyond fifty. Raising the count is safe
# only because the merge below is round robin by rank. Under the concatenation this function
# used before D64 the proposal was not monotone in the count, and 233 of 1296 starts present at
# twenty were absent at fifty.
_PERIODOGRAM_PEAKS = 50

# The candidate fits of the bootstrap make their assignments over the periods within this
# many 1/T of each candidate, T being the time span of the usable epochs
# (`albireo.rvorbit.assign_components`, D68). A candidate is a periodogram peak, and for two
# alike components, exchanged at random between the epochs, the peak is displaced from the
# period by a fraction of 1/T, which is enough for the assignment decided at the candidate to
# be wrong. Over the 32 blind tables of the third run that the correlation measures, the
# injected period is first in the chi-square ranking on 26 with the window and on 25 without
# it, and no table loses its place (`scripts/period_decision_bench.py`).
_CANDIDATE_PERIOD_WINDOW = 1.0

# The leave-one-epoch-out source of the bootstrap (D65). On a table of at most
# `_LEAVE_ONE_OUT_MAX_EPOCHS` usable epochs as measured, the `_LEAVE_ONE_OUT_PEAKS` highest peaks
# of the one-harmonic search with each epoch left out in turn are appended to the candidates.
# Over the 33 blind tables of the third run, with the code as implemented, the 17 tables of at
# most 25 usable epochs gain 18 starts in total after the 2% merge (none on eight of them, seven
# on one). One system whose twin components the correlation exchanged at one epoch of twelve
# goes from absent in the chi-square ranking to rank 1, one other moves from rank 37 to 38, and
# no system loses its rank 1 (21 at rank 1 against 20). The limits are those measured and were
# not tuned. A periodogram does not tolerate one wrong epoch in ten to twenty-five and a
# Keplerian fit with re-assignment does, and nothing was measured above twenty-five.
_LEAVE_ONE_OUT_MAX_EPOCHS = 25
_LEAVE_ONE_OUT_PEAKS = 3

# A bootstrap table whose usable epochs fall on fewer nights than this is flagged (D65). A night
# is an integer part of the epoch's BJD. Over the 33 blind tables of the third run three have
# usable epochs on fewer than eight nights, and the search recovers the period of none of them.
# On one, with nine usable epochs on seven nights, the injected velocities with noise at the
# quoted errors put the true period first in 2 of 10 draws, with every loss inside a chi-square
# difference of 5. The count is small and the threshold describes those tables rather than a
# derived limit, so it raises a flag and changes nothing.
_FEW_NIGHTS = 8


def _usable_nights(table) -> int:
    """The number of distinct nights (integer parts of the BJD) among the usable epochs."""
    bjd = np.asarray(table.bjd, dtype=float)[np.asarray(table.good, dtype=bool)]
    return int(np.unique(np.floor(bjd)).size)


def _detection_mask(table, detection_min: float) -> np.ndarray:
    """``(n_comp, n_epochs)``: the gated velocities, measured and below ``detection_min``.

    Only the components after the first, which are declared in order of decreasing mass,
    can be gated (see ``Analysis.detection_min``).
    """
    statistic = np.asarray(table.delta_chi2, dtype=float)
    velocity = np.asarray(table.velocity, dtype=float)
    gated = np.isfinite(velocity) & (np.nan_to_num(statistic, nan=np.inf) < float(detection_min))
    gated[:1] = False
    return gated


def _detection_gate(table, detection_min: float):
    """The table with every weakly detected companion velocity removed, and the mask.

    A velocity of a component after the first is removed (its velocity and errors set to
    ``nan``) where that component's detection statistic, ``table.delta_chi2``, is below
    ``detection_min``. The first component's velocities are never removed. The other
    components of the epoch are unchanged, and the search and the fits keep them
    (:func:`albireo.rvorbit.find_period` and :func:`albireo.rvorbit.fit_rv_orbit` take each
    velocity where its own component was measured). Returns the table itself, and an
    all-false mask, when nothing falls below the threshold or the threshold is zero.
    """
    gated = _detection_mask(table, detection_min)
    if not gated.any():
        return table, gated

    def cleared(array):
        out = np.array(array, dtype=float)
        out[gated] = np.nan
        return out

    return (
        replace(
            table,
            velocity=cleared(table.velocity),
            sigma=cleared(table.sigma),
            sigma_ivar=cleared(table.sigma_ivar),
        ),
        gated,
    )


def _period_candidates(
    table,
    *,
    swap_invariant: bool,
    detection_min: float | None = None,
    leave_one_out: bool = False,
):
    """The starting periods the orbit fit chooses among, and the searches that proposed them.

    Four periodograms of the table are searched (:func:`albireo.rvorbit.find_period`).
    The floating-mean generalized Lomb-Scargle of the relative velocity contributes its
    ``_PERIODOGRAM_PEAKS`` highest peaks. Its two-harmonic form, which ranks an eccentric
    orbit's period higher, contributes the same number. For two components, the
    swap-invariant search contributes its first four peaks with their doubles, and is the
    only source unaffected by components exchanged between epochs. For two or more
    components, the first component's own velocities contribute ``_PERIODOGRAM_PEAKS``
    peaks, and are unaffected by a companion the templates could not follow (its relative
    velocity is then noise while the primary's curve is intact). The lists are merged
    round robin by rank: every source's highest peak, then every source's second, and so
    on, a source dropping out when its list is exhausted. The merged sequence is
    deduplicated greedily at the same 2% the fit loop uses. On the D62 oracle tables this
    left a median of 37 starting periods out of 48 before the fourth source was added.

    The merge is by rank because the greedy deduplication resolves a collision in favour
    of whichever period it reaches first. Concatenating the sources (the behaviour before
    D64) let a low-ranked peak of an early source displace the top peak of a later one.
    The source affected was the swap-invariant one, which comes last and does not grow
    with the peak count. Concatenation also made the proposal non-monotone in that count.
    Over the 33 blind-tier stars of D64, 233 of the 1296 starts proposed at twenty peaks
    were absent at fifty and 403 at a hundred. On one Gaia system the start displaced at a
    hundred was the one within 0.03% of the true period, whose orbit fits at chi-square
    2028 against 10072 for the peak that displaced it. Merging by rank corrects both. The
    highest-ranked peak of each source is kept in its collisions, and a larger peak count
    only appends entries beyond the old limit, so the list at the larger count contains
    the list at the smaller. This is exact for any count at or above the eight entries
    the swap-invariant source contributes. The sources cannot be ranked against each
    other by periodogram power, because they are different statistics on different series
    (a relative velocity, its two-harmonic form, one component's own velocities and the
    magnitude of a difference), whose powers are not on a common scale.

    Measured end to end over those 32 tables, this recovers the period of 28, against 23
    for the six peaks of the classical periodogram used before. The floating mean accounts
    for about three systems and the longer list for about two. Halves and doubles of the
    ordinary peaks are not added. They made one further true period reachable and changed
    no decision, and every extra candidate is one more chance for an alias to have the
    lowest chi-square.

    A search with too few epochs for its model contributes nothing, and the others are
    used: a table too short for the five parameters of the two-harmonic fit, and, where
    velocities are gated or unmeasured, a relative velocity defined at fewer than four
    epochs while the first component was measured at more. The one-harmonic search's
    error is raised only when every search fails.

    With ``detection_min`` (the bootstrap passes ``Analysis.detection_min``) every search runs
    on :func:`_detection_gate`'s copy of the table, in which a companion velocity whose
    detection statistic is below the threshold is not a measurement. The first component
    is never gated, so its source keeps every epoch at which it was measured. For a
    companion detected nowhere it is the only source left.

    With ``leave_one_out`` (passed only by the bootstrap) and a table of at most
    ``_LEAVE_ONE_OUT_MAX_EPOCHS`` usable epochs as measured (``table.good`` before any gate),
    a fifth source follows the merge. The one-harmonic search, on the gated copy where
    there is one, is repeated with each of those epochs left out in turn, and the
    ``_LEAVE_ONE_OUT_PEAKS`` highest peaks of each, in epoch order, are appended wherever
    they lie more than 2% from every candidate already listed. The source targets one
    wrong epoch among ten to twenty-five, which a periodogram does not tolerate and a
    Keplerian fit with re-assignment does. On the benchmark system that motivated it, the
    correlation exchanged twin components at one epoch of twelve, with detection
    statistics of 15,641 and 15,577 that mark nothing. This put the true period at peak
    108 of the one-harmonic search, and the leave-one-out peaks put it at rank 1 of the
    chi-square ranking (the measurement is beside ``_LEAVE_ONE_OUT_MAX_EPOCHS``). The block
    comes after the round robin and is deduplicated against it, as measured. The round
    robin still only appends as the peak count grows, and the block never removes a
    periodogram start. A longer periodogram list can, however, replace a leave-one-out
    start with a periodogram start within 2% of it, so the whole list is monotone in the
    peak count only up to that substitution.

    The searches are returned under their names, with ``"leave_one_out"`` the list of the
    starts the fifth source added (``None`` where it did not run).
    """
    from albireo.rvorbit import find_period

    # The leave-one-out limit and the epochs it leaves out are the table's usable epochs as
    # measured, before any gate, because the source was measured in that configuration.
    measured_good = np.asarray(table.good, dtype=bool) if leave_one_out else None
    if detection_min:
        table, _ = _detection_gate(table, detection_min)

    def search(**kwargs):
        try:
            return find_period(table, **kwargs), None
        except ValueError as exc:
            return None, exc

    single, refusal = search(n_peaks=_PERIODOGRAM_PEAKS)
    harmonic, _ = search(n_harmonics=2, n_peaks=_PERIODOGRAM_PEAKS)
    sources: list[list[float]] = []
    if single is not None:
        sources.append([single["period"], *single["aliases"]])
    if harmonic is not None:
        sources.append([harmonic["period"], *harmonic["aliases"]])
    first = None
    if table.n_components >= 2:
        # The first component alone: a faint companion the templates could not follow
        # leaves the relative velocity as noise, while the primary's own curve is intact.
        first, _ = search(components=[table.names[0]], n_peaks=_PERIODOGRAM_PEAKS)
        if first is not None:
            sources.append([first["period"], *first["aliases"]])
    invariant = None
    if swap_invariant and table.n_components == 2:
        invariant, _ = search(swap_invariant=True)
        if invariant is not None:
            doubled: list[float] = []
            for peak in [invariant["period"], *invariant["aliases"][:3]]:
                doubled.extend([peak, 2.0 * peak])
            sources.append(doubled)
    if not sources:
        raise refusal if refusal is not None else ValueError("no period search could run")
    # Round robin by rank, so that the deduplication below resolves a collision in favour of
    # the higher-ranked peak whichever source proposed it.
    proposed = [
        peaks[rank]
        for rank in range(max(len(entries) for entries in sources))
        for peaks in sources
        if rank < len(peaks)
    ]
    candidates: list[float] = []
    for period in proposed:
        period = float(period)
        if period > 0.0 and all(abs(period / other - 1.0) > 0.02 for other in candidates):
            candidates.append(period)
    added = None
    if leave_one_out and int(np.sum(measured_good)) <= _LEAVE_ONE_OUT_MAX_EPOCHS:
        added = []
        for j in np.flatnonzero(measured_good):
            left_out = {}
            for key in ("velocity", "sigma", "sigma_ivar"):
                values = np.array(getattr(table, key), dtype=float)
                values[:, j] = np.nan
                left_out[key] = values
            try:
                found = find_period(replace(table, **left_out), n_peaks=_LEAVE_ONE_OUT_PEAKS)
            except ValueError:
                continue
            for period in [found["period"], *found["aliases"]]:
                period = float(period)
                if period > 0.0 and all(abs(period / other - 1.0) > 0.02 for other in candidates):
                    candidates.append(period)
                    added.append(period)
    return candidates, {
        "single": single,
        "harmonic": harmonic,
        "invariant": invariant,
        "first": first,
        "leave_one_out": added,
    }


def _orbit_over_candidates(
    ctx: _Context,
    table,
    periods,
    *,
    circular: bool,
    detection_min: float | None = None,
    exchange: bool = True,
):
    """Fit an orbit from every candidate period and keep the lowest chi-square.

    The periodogram of a sparsely sampled table is rarely unambiguous. On the ten-epoch
    test fixture the highest peak was a 2.25 d alias whose orbit fits at chi-square 73,
    against 16 at the true period, to which every other peak converged. The periodogram
    peaks are starting points, and the orbit fit decides. Over the D62 oracle tables,
    wherever a candidate reached the true period the eccentric chi-square preferred it in
    27 of 29 systems, usually by hundreds, while an alias outranked the true period on the
    periodogram in 10 of 32. Many candidates are therefore fitted, at a fixed model order.
    A circular fit and a BIC choice between the two orders were both measured and neither
    helped.

    Distinct starting periods are all fitted, and the fits are merged on the period they
    converged to, since starts a few per cent apart reach the same optimum and the same
    period twice is not an ambiguity. Every remaining fitted period within a chi-square
    difference of 25 of the best is named in one flag, because a bootstrap gives the
    disentangling a period to within 3%, which is unrecoverable if it is the wrong one.

    Each candidate's orbit is fitted by :func:`albireo.rvorbit.assign_components`, which
    decides with the orbit what one epoch of a two-component table leaves open: the order
    of the pair where the correlation exchanged two alike components, and the minimum at
    the epochs the table flags for a second one. The fit and the decisions are repeated
    until nothing changes, from the table as measured and from assignments made from the
    period alone: the candidate period and the periods within
    ``_CANDIDATE_PERIOD_WINDOW`` of it, in units of the reciprocal of the time span of the
    usable epochs. Every flagged epoch is decided at every candidate, so the orbits of all
    candidates are fitted to the same velocities and their chi-squares are comparable. The
    table that goes with the best orbit is returned beside it. A third value holds the
    number of candidates fitted, the ambiguous periods, and under ``"ranked"`` every
    distinct valid orbit with its own table in chi-square order, the best first, of which
    :func:`_decide_period_by_disentangling` takes the top few.

    Before D68 each candidate had one exchange by its first fit, and the flagged epochs
    had no weight. ``scripts/period_decision_bench.py`` measures both procedures on the
    blind tables of the benchmark (``docs/benchmarks.md``).

    The bootstrap passes two further settings. With ``detection_min`` the orbits are fitted
    to :func:`_detection_gate`'s copy of the table, so that a velocity below the detection
    threshold has no weight in any fit. The decisions are made on that copy and applied to
    the table as measured, which is the table returned. With ``exchange`` false no pair
    is interchanged, as :func:`_exchange_allowed` decides for light fractions too far
    apart for the correlation's two peaks to be equivalent solutions. The second minima
    are still decided.
    """
    import albireo.rvorbit as rvorbit

    fit_table = table
    if detection_min:
        fit_table, _ = _detection_gate(table, detection_min)
    seen: list[float] = []
    fitted = []
    for period in periods:
        period = float(period)
        if not period > 0.0 or any(abs(period / p - 1.0) < 0.02 for p in seen):
            continue
        seen.append(period)
        try:
            # The pipeline's own rule has decided whether the pair may be interchanged.
            assigned, orbit, decided = rvorbit.assign_components(
                fit_table,
                period=period,
                circular=circular,
                exchange=exchange,
                light_ratio_max=None,
                period_window=_CANDIDATE_PERIOD_WINDOW,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        fitted.append((orbit, assigned if fit_table is table else decided.apply(table)))
    if not fitted:
        raise ValueError("no candidate period gave an orbit fit")
    fitted.sort(key=lambda pair: pair[0].chi2)
    n_fitted = len(fitted)
    # An orbit above the declared ranges is not a solution the run could accept, so it is
    # set aside. A companion the templates could not follow gives velocities that a wrong
    # period fits with an implausible semi-amplitude and eccentricity at a lower chi-square
    # than the true period (1537 km/s at e = 0.94 on a 7-percent secondary), and the table's
    # chi-square cannot distinguish the two. A semi-amplitude below the lower bound is not
    # tested. An unmeasurable companion gives one at the true period as readily as at a
    # wrong one, and the degenerate-K logic downstream handles it. The best orbit set aside
    # is named when it was the lowest, and every fit is kept when none remains.
    settings = ctx.settings
    valid = [
        pair
        for pair in fitted
        if np.all(np.asarray(pair[0].k) <= settings.k_max)
        and float(pair[0].ecc) <= settings.ecc_max
    ]
    if valid and valid[0] is not fitted[0]:
        outside = fitted[0][0]
        ctx.flag(
            f"the lowest chi-square candidate orbit (P {outside.period:.4f} d, K "
            f"{np.round(outside.k, 1).tolist()} km/s, e {outside.ecc:.2f}) lies outside the "
            f"declared ranges (K up to {settings.k_max:g}, e up to {settings.ecc_max:g}) and "
            "was set aside"
        )
    if valid:
        fitted = valid
    else:
        ctx.flag(
            "no candidate orbit lies inside the declared ranges; the lowest chi-square one "
            "is taken as it stands"
        )
    distinct = []
    for orbit, candidate_table in fitted:
        if all(abs(orbit.period / other.period - 1.0) > 0.02 for other, _ in distinct):
            distinct.append((orbit, candidate_table))
    best, best_table = distinct[0]
    ambiguous = [
        {"period": float(other.period), "delta_chi2": float(other.chi2 - best.chi2)}
        for other, _ in distinct[1:]
        if other.chi2 - best.chi2 < 25.0
    ]
    if ambiguous:
        ctx.flag(
            f"period ambiguous: {len(ambiguous)} other fitted period(s) lie within delta chi2 "
            f"25 of the chosen {best.period:.4f} d: "
            + ", ".join(f"{a['period']:.4f} d at +{a['delta_chi2']:.1f}" for a in ambiguous)
        )
    return (
        best,
        best_table,
        {"n_candidates": n_fitted, "ambiguous": ambiguous, "ranked": list(distinct)},
    )


def _labels(ctx: _Context, fit: Fit, library):
    from albireo.match import FixedDilution, RadiusRatio, ScalarDilution, StarLabels

    star, settings, log = ctx.star, ctx.settings, ctx.log
    medium = fit.dis.dataset[0].medium
    if medium is None:
        ctx.flag(
            "labels skipped: the files do not declare whether their wavelengths are air or "
            "vacuum (an 83 km/s question); set medium = 'air' or 'vacuum' on the star once "
            "you have checked, and the velocities will come out absolute"
        )
        return None
    grid = fit.dis.grid
    try:
        lib = library.in_medium(medium).sliced(
            grid.wave[0] - _LIBRARY_PAD_ANGSTROM, grid.wave[-1] + _LIBRARY_PAD_ANGSTROM
        )
        if lib.wave[0] > grid.wave[0] or lib.wave[-1] < grid.wave[-1]:
            raise ValueError(
                f"the library spans {library.wave[0]:.1f}-{library.wave[-1]:.1f} A and the "
                f"model grid needs {grid.wave[0]:.1f}-{grid.wave[-1]:.1f} A"
            )
    except ValueError as exc:
        ctx.flag(f"labels skipped: {exc}")
        return None
    if settings.dilution == "radius_ratio" and fit.dis.n_stellar > 1:
        dilution: Any = RadiusRatio()
    elif settings.dilution == "fixed":
        dilution = FixedDilution()
    else:
        dilution = ScalarDilution()
    reach, step = _v_zero_scan(fit, settings)
    scan_velocities = np.arange(-reach, reach + 0.5 * step, step)
    stars = {}
    for c in star.components:
        stars[c.name] = StarLabels(
            library=lib,
            teff=_spec(c.teff, "teff"),
            logg=_spec(c.logg, "logg"),
            vsini=_spec(c.vsini, "vsini") or Between(0.0, settings.vsini_max),
            v_kms=Between(-reach, reach),
        )
    options: dict[str, Any] = {
        "dilution": dilution,
        "compare": settings.label_compare,
        "max_steps": settings.label_steps,
        "scan_velocities": scan_velocities,
    }
    if ctx.config.mh is not None:
        options["mh"] = _spec(ctx.config.mh, "mh")
    budget = (
        f"{settings.label_steps} Levenberg-Marquardt iterations per run"
        if settings.label_compare == "epochs"
        else f"{settings.label_steps} steps"
    )
    log(
        f"label fit ({settings.label_compare} comparison) against {lib.n_nodes} nodes, "
        f"{len(scan_velocities)} trial frame offsets over +-{reach:g} km/s, {budget}"
    )
    try:
        match = fit.match_labels(stars, **options)
    except (ValueError, RuntimeError) as exc:
        ctx.flag(f"labels failed: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        return None
    _assess_labels(ctx, match, {c.name: c.light for c in star.components})
    return match


_LIGHT_FACTOR = 1.5
"""Ratio between the label fit's light fraction and the declared one above which it is
flagged. On the orbit tier of the D65 benchmark products the epoch comparison measured the
light to a median absolute error of 0.011. The correlation stage that declared it had an
error of 0.043, with errors of 0.33, 0.23, 0.21 and 0.12 among eleven products
(``d65_converged_labels.md``)."""

_LIGHT_DIFFERENCE = 0.15
"""Absolute difference in light fraction above which it is flagged whatever the ratio."""


def _assess_labels(ctx: _Context, match, declared_light: Mapping[str, float]) -> None:
    """The flags a finished label fit raises: what it did not measure, and what it contradicts.

    A site whose posterior is as wide as its prior was not measured. In the ``"epochs"``
    comparison a site on a bound or on the rotation plateau is left out of the formal
    covariance (:attr:`albireo.LabelMatch.at_bounds`), so it has no posterior width to
    compare and that test does not detect it. Separate flags name such a site, anything the
    optimiser's restarts could not resolve (``epoch_fit.notes``, such as a faint component
    whose light collapsed), and a grid compensation of the epoch operator that was floored
    at half a model pixel or ran on a model grid coarser than the width it compensates
    (``statistics.notes``). A light fraction that differs from the declared one by
    more than a factor :data:`_LIGHT_FACTOR`, or by more than :data:`_LIGHT_DIFFERENCE`, is
    the signature of a wrong declaration or of a reversed component order. A second basin
    close in chi-square is flagged, and a fit that is not better than both nulls measured
    nothing.
    """
    weak = {
        k: v
        for k, v in match.posterior_over_prior.items()
        if v > 0.8 and not k.startswith(("log_jitter", "offset"))
    }
    if weak:
        ctx.flag(
            "labels learned nothing about "
            + ", ".join(sorted(weak))
            + " (posterior width >= 80% of the prior)"
        )
    held = {
        k: why
        for k, why in (getattr(match, "at_bounds", None) or {}).items()
        if not k.startswith(("log_jitter", "offset"))
    }
    if held:
        ctx.flag(
            "labels with no formal error: "
            + ", ".join(f"{k} ({why})" for k, why in sorted(held.items()))
            + ". A site on a bound, or a v sin i on the plateau below half a model pixel where "
            "the chi-square is flat, is left out of the covariance, so the posterior-width "
            "test cannot flag it; the value is a limit, not a measurement"
        )
    epoch_fit = getattr(match, "epoch_fit", None)
    for note in () if epoch_fit is None else tuple(epoch_fit.notes):
        ctx.flag(f"labels: {note}")
    # The grid compensation of the epoch operator: a width floored at half a model pixel, or
    # a model grid coarser than the width it compensates (Fit.epoch_statistics).
    statistics = getattr(match, "statistics", None)
    for note in () if statistics is None else tuple(statistics.notes):
        ctx.flag(f"labels: {note}")

    errors = getattr(match, "flux_ratio_errors", None) or {}
    measured = dict(match.flux_ratio)
    off = {}
    for name, fitted in measured.items():
        declared = float(declared_light[name])
        ratio = max(fitted, 1e-12) / declared
        factor = max(ratio, 1.0 / ratio)
        if factor > _LIGHT_FACTOR or abs(fitted - declared) > _LIGHT_DIFFERENCE:
            off[name] = factor
    if off:
        ctx.flag(
            "the label fit measures light fractions of "
            + ", ".join(
                f"{n} {measured[n]:.3f}"
                + (f" +- {errors[n]:.3f}" if errors.get(n) is not None else "")
                for n in measured
            )
            + " against the declared "
            + ", ".join(f"{float(declared_light[n]):.3f}" for n in measured)
            + " ("
            + ", ".join(f"{n} off by a factor {f:.2f}" for n, f in off.items())
            + f"; flagged above a factor {_LIGHT_FACTOR:g} or a difference of "
            f"{_LIGHT_DIFFERENCE:g}): either the declared light fractions are wrong, or the "
            "components were declared in the wrong order (the pipeline assumes decreasing "
            "mass: the first star moves least)"
            + (
                ". The epochs comparison measures the light without reference to the "
                "declaration, which the disentangled components and the velocity table "
                "still carry"
                if getattr(match, "compare", None) == "epochs"
                else ""
            )
        )
    if match.multimodal:
        ctx.flag("labels: the scan found a second basin within delta chi2 < 9")
    if not _beats_both_nulls(match):
        ctx.flag(
            "labels: the fit does not beat both nulls (chi2 "
            f"{match.chi2:.1f}, nearest node {match.chi2_nearest_node:.1f}, no template "
            f"{match.chi2_continuum:.1f}); treat the labels and the zero points as unmeasured"
        )


def _v_zero_scan(fit: Fit, settings: Analysis) -> tuple[float, float]:
    """Half-range and trial step of the label fit's scan over each component's frame offset."""
    reach = float(settings.v_zero_range)
    step = max(2.0 * fit.dis._narrowest_lsf(), 5.0, 2.0 * reach / 120.0)
    return reach, step


def _beats_both_nulls(match) -> bool:
    """Whether the label fit beat the nearest-node and the no-template nulls.

    The label stage raises a flag on this test, and it is the first of the three on which
    the zero points are rejected. A fit that is worse than no template has measured no
    frame offset, whatever value its ``v_kms`` site has.
    """
    return bool(match.chi2 < match.chi2_nearest_node < match.chi2_continuum)


def _disowned_zero_points(
    ctx: _Context, fit: Fit, match, offsets: Mapping[str, float]
) -> list[str]:
    """The reasons the label fit's frame offsets cannot serve as template zero points.

    In each case the fit reports a value it did not measure. The offset enters every
    reported velocity as a constant, so an unmeasured one shifts the whole table without
    changing any diagnostic in it.
    """
    reasons = []
    if not _beats_both_nulls(match):
        reasons.append(
            f"the label fit beats neither null (chi2 {match.chi2:.1f}, nearest node "
            f"{match.chi2_nearest_node:.1f}, no template {match.chi2_continuum:.1f})"
        )
    reach, step = _v_zero_scan(fit, ctx.settings)
    for name, offset in offsets.items():
        gap = reach - abs(offset)
        if gap <= step:
            reasons.append(
                f"the frame offset of {name!r} is pinned at {offset:+.4f} km/s, {gap:.4f} "
                f"km/s from the {reach:g} km/s bound of its scan and inside one {step:.1f} "
                "km/s trial step of it (widen v_zero_range if the frame really sits there)"
            )
    if fit.mode == "keplerian" and offsets:
        budget = float(fit.dis.velocity_budget.total)
        spread = max(offsets.values()) - min(offsets.values())
        if spread > budget:
            reasons.append(
                f"the frame offsets differ by {spread:.1f} km/s, more than the "
                f"{budget:.1f} km/s velocity budget of the declaration, while a Keplerian "
                "fit gives every component the same systemic velocity by construction"
            )
    return reasons


def _templates(ctx: _Context, fit: Fit, match) -> list:
    """The disentangled components as correlation templates, with their zero points.

    A disentangled component's rest frame is not identified (``docs/math.md`` §5.3), so
    the templates leave ``v_zero_kms`` at ``None`` and the velocities measured against
    them are differential. A label fit that measured each component's frame offset sets
    the zero points, and the velocities are then absolute. If that offset is rejected
    (:func:`_disowned_zero_points`), the table stays differential and the orbit fit has
    one systemic velocity per component, as on the no-library route.
    """
    templates = fit.templates()
    if match is None:
        ctx.zero_points = {"source": None, "adopted": False, "v_zero_kms": {}, "refused": []}
        return templates
    offsets = {t.name: float(match.labels[t.name]["v_kms"]) for t in templates}
    refused = _disowned_zero_points(ctx, fit, match, offsets)
    ctx.zero_points = {
        "source": "label match",
        "adopted": not refused,
        "v_zero_kms": offsets,
        "refused": refused,
    }
    if refused:
        ctx.flag(
            "template zero points refused, so the velocities stay differential (one gamma "
            "per component in the orbit fit): " + "; ".join(refused)
        )
        return templates
    # A frame offset the label fit did not measure (its posterior as wide as the prior, the
    # component being mostly noise) is rejected for that component alone. A secondary of a
    # third benchmark run kept 11 percent of its equivalent width, its fitted offset was
    # 46 km/s from the systemic velocity, and every one of its velocities included it.
    widths = getattr(match, "posterior_over_prior", {}) or {}
    unlearned = {
        t.name: float(widths[f"v_{t.name}"])
        for t in templates
        if float(widths.get(f"v_{t.name}", 0.0)) > 0.8
    }
    if unlearned:
        ctx.flag(
            "zero point refused for "
            + ", ".join(
                f"{n} (frame offset posterior {w:.2f} of the prior)" for n, w in unlearned.items()
            )
            + ": the label fit learned nothing about it, so that component's velocities stay "
            "differential with its own systemic velocity in the orbit fit"
        )
        ctx.zero_points["refused"] = [
            f"{n}: the label fit learned nothing about the frame offset" for n in unlearned
        ]
        ctx.zero_points["adopted"] = len(unlearned) < len(templates)
    pinned = []
    for t in templates:
        if t.name in unlearned:
            pinned.append(t)
            continue
        offset = offsets[t.name]
        pinned.append(
            replace(
                t,
                v_zero_kms=offset,
                meta={**t.meta, "zero_point": "label match", "v_zero_kms": offset},
            )
        )
    ctx.log(
        "template zero points from the label fit: "
        + ", ".join(
            f"{t.name} {t.v_zero_kms:+.2f} km/s" if t.v_zero_kms is not None else f"{t.name} none"
            for t in pinned
        )
    )
    return pinned


_SMOOTHNESS_MOVED = 10.0
"""Factor by which a fitted smoothness precision must differ from its start before the
correlation stops holding the light at the declared fractions."""

_EXCHANGE_LIGHT_RATIO = 3.0
"""Largest ratio of two light fractions at which the correlation's two peaks are still
treated as equivalent solutions that the orbit may exchange."""


def _template_light(ctx: _Context, fit: Fit, lights: list):
    """The declared fractions, or free amplitudes when ML-II moved the smoothness far.

    The disentangling recovers each component as ``(w / l0) t`` at the declared fraction
    ``l0``, so ``l0`` is the amplitude consistent with the template only while the posterior
    mean is not shrunk. When ML-II raises a component's smoothness precision by an order of
    magnitude or more the recovered lines are shallower than the true ones, and the
    consistent amplitude is larger than the fraction. On a pair whose primary's ``tau``
    went from 800 to 36,000 and secondary's from 400 to 2000, the secondary lost a quarter
    of its depth. Under the declared fraction the table then lost the secondary, at -89
    percent, and a free amplitude gave -25. The correlation then fits the amplitudes
    freely (``light="global"``). The table's light column is that scale, not a light
    fraction, and the flag states this.
    """
    moved = {}
    for star in fit.dis.stars:
        tau0 = float(_smoothness_of(star).tau0)
        tau = float(fit.hyper[star.name]["tau"])
        if tau / tau0 > _SMOOTHNESS_MOVED or tau0 / tau > _SMOOTHNESS_MOVED:
            moved[star.name] = (tau0, tau)
    if not moved:
        return lights
    ctx.flag(
        "the correlation fitted the template amplitudes freely rather than holding the "
        "declared light fractions: the smoothness precision moved from its start by more "
        f"than a factor {_SMOOTHNESS_MOVED:g} ("
        + ", ".join(f"{n} {t0:g} -> {t:.3g}" for n, (t0, t) in moved.items())
        + "), so the disentangled components are shrunk and the declared fractions are not "
        "their scale; the table's light column is that scale, not a light fraction"
    )
    return "global"


def _exchange_allowed(ctx: _Context, lights, *, where: str = "by the orbit") -> bool:
    """Whether the orbit may exchange the two components between epochs.

    The exchange assumes two alike spectra at alike light fractions, where the
    correlation's two peaks are equivalent solutions. Under held amplitudes at fractions
    a factor of several apart they are not. An exchanged row has the other component's
    amplitude, and a swap decided on a noise draw of the faint component's velocity moves
    a well-measured primary velocity into the wrong column. A 95/5 pair went from 5 to 56
    percent off in the primary in this way, with nineteen of eighty epochs swapped.

    The same rule applies to the bootstrap's candidate fits (D65), with the fractions of the
    library table's amplitudes (:func:`_table_light_fractions`). Without it, the
    chosen orbits of two benchmark systems whose amplitudes differed by factors of 7.1 and
    3.25 had re-assigned 3 and 11 of their epochs. Over the 33 blind tables of the third
    run the rule skips the exchange on 8, and no system loses its rank 1 in the chi-square
    ranking. Under it one Gaia system's true period moves from rank 20 to 28, another's
    from 37 to 38, and the field system with the 7% secondary from absent to rank 25.
    ``where`` names the step in the flag.
    """
    fractions = np.asarray(lights, dtype=float)
    if fractions.size != 2 or not np.all(np.isfinite(fractions)) or np.any(fractions <= 0):
        return True
    ratio = float(fractions.max() / fractions.min())
    if ratio <= _EXCHANGE_LIGHT_RATIO:
        return True
    ctx.flag(
        f"the exchange of the two components {where} was skipped: the light fractions "
        f"{np.round(fractions, 3).tolist()} differ by a factor {ratio:.1f}, above "
        f"{_EXCHANGE_LIGHT_RATIO:g}, and under held amplitudes the two correlation peaks "
        "are not equivalent solutions"
    )
    return False


def _measure_epoch_velocities(ctx: _Context, fit: Fit, templates: list, lights):
    """Correlate the epochs against the disentangled templates, dropping zero points if needed.

    The default search window of a template is offset from the fitted velocities by its
    own zero point, and zero points that disagree by more than those velocities span leave
    no window that contains every component. The label fit of a broad-lined pair put them
    95 km/s apart, on velocities spanning 60. The zero points are then dropped rather than
    the star. The velocities stay differential, as when the label fit's offsets are
    rejected (:func:`_disowned_zero_points`), and the orbit fit has one systemic velocity
    per component. A failure with no zero point to drop is re-raised.

    Where the amplitudes are fitted freely (``lights`` is ``"global"``) and no epoch gives
    positive ones, there is no amplitude to hold. The table is then measured with the
    amplitudes of each epoch (``light="free"``) and a flag says so. The products of the
    disentangling are kept, and :func:`_assess_table` marks the table failed where it has
    no usable epoch.

    Returns the table and the templates it was measured against, which the report and the
    figures must show.
    """
    from albireo.todcor import _LightNotMeasured

    def measure(pair):
        try:
            return fit.measure_velocities(templates=pair, light=lights)
        except _LightNotMeasured:
            ctx.flag(
                "the freely fitted template amplitudes are not positive at any epoch, so none "
                "is held: the table has the amplitudes of each epoch, which are not light "
                "fractions, and the templates do not describe the components"
            )
            return fit.measure_velocities(templates=pair, light="free")

    try:
        return measure(templates), templates
    except ValueError as exc:
        if not any(t.v_zero_kms is not None for t in templates):
            raise
        ctx.flag(
            "template zero points dropped, so the velocities stay differential (one gamma "
            f"per component in the orbit fit): {exc}"
        )
        previous = ctx.zero_points or {}
        ctx.zero_points = {
            **previous,
            "adopted": False,
            "refused": [*previous.get("refused", []), str(exc)],
        }
        templates = [
            replace(t, v_zero_kms=None, meta={**t.meta, "zero_point": "dropped"}) for t in templates
        ]
        return measure(templates), templates


def _orbit(ctx: _Context, fit: Fit, table):
    from albireo.rvorbit import fit_rv_orbit

    settings = ctx.settings
    failure = _table_failure(table)
    if failure is not None:
        ctx.flag(f"orbit from the table skipped: the table failed ({failure})")
        return None, None
    n_par = 2 + (0 if settings.circular else 2) + 2 * table.n_components
    if int(table.good.sum()) * table.n_components <= n_par:
        ctx.flag(
            f"orbit from the table skipped: {int(table.good.sum())} usable epochs x "
            f"{table.n_components} components cannot constrain {n_par} parameters"
        )
        return None, None
    try:
        if fit.mode == "keplerian":
            # For every Keplerian fit, including a bootstrapped one, the disentangling's
            # period is the starting point.
            period = float(fit.orbit()["period"])
            source = "the disentangling"
            orbit = fit_rv_orbit(table, period=period, circular=settings.circular)
        else:
            # The detection gate and the leave-one-epoch-out starts are not used here. Both
            # were measured on the bootstrap's library-template tables only (D65).
            candidates, searches = _period_candidates(table, swap_invariant=True)
            search = searches["single"]
            if search is None:
                peak = "no relative-velocity search: too few epochs measured both components"
            else:
                peak = (
                    f"peak {search['period']:.4f} d, aliases "
                    f"{[round(p, 4) for p in search['aliases'][:3]]}"
                )
            source = (
                f"a periodogram ({peak}; {len(candidates)} candidates from the periodogram "
                "searches, the orbit fit decided)"
            )
            orbit, _, _ = _orbit_over_candidates(ctx, table, candidates, circular=settings.circular)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        ctx.flag(f"orbit from the table failed: {type(exc).__name__}: {exc}")
        return None, None
    return orbit, source


def _sample(ctx: _Context, fit: Fit, match):
    settings, log = ctx.settings, ctx.log
    if fit.mode != "keplerian":
        ctx.flag("sampling skipped: a free-velocity fit has no orbital sites to sample")
        return None, match
    log(
        f"NUTS: {settings.num_chains} chain(s) x {settings.num_warmup} warmup + "
        f"{settings.num_samples} samples"
    )
    posterior = fit.sample(
        num_warmup=settings.num_warmup,
        num_samples=settings.num_samples,
        num_chains=settings.num_chains,
    )
    extra = getattr(posterior.mcmc, "get_extra_fields", dict)()
    if "diverging" in extra:
        n_div = int(np.sum(np.asarray(extra["diverging"])))
        if n_div:
            ctx.flag(f"sampling: {n_div} divergent transitions")
    if match is not None and getattr(match, "epoch_fit", None) is not None:
        # refit_draws refits d_hat draws, which the epochs comparison does not use; its
        # summary states how its formal errors were calibrated instead.
        log("label draws skipped: the epochs comparison is not refitted over d_hat draws")
    elif match is not None:
        from albireo.match import refit_draws

        n_draws = 8 if settings.fast else 16
        draws = posterior.spectra(num_draws=n_draws)
        try:
            match = refit_draws(match, draws[:, : fit.dis.n_stellar], max_steps=40)
            log(f"label errors from {n_draws} joint posterior draws")
        except (ValueError, RuntimeError) as exc:
            ctx.flag(f"label draws failed: {type(exc).__name__}: {exc}")
    return posterior, match


# -- assessment -------------------------------------------------------------------


def _assess_fit(ctx: _Context, fit: Fit) -> None:
    z = fit.z_rms
    if abs(z - 1.0) > 0.2:
        ctx.flag(
            f"residual z-score rms {z:.3f}: the noise model does not describe these data "
            "(read docs/benchmarks.md before adding a jitter term)"
        )
    for name, component in zip(fit.dis.component_names, fit.dis.ordered_components, strict=True):
        smooth = component.smoothness
        fitted = fit.hyper[name]["tau"]
        drift = (math.log(fitted) - math.log(smooth.tau0)) / smooth.sigma
        if abs(drift) < 0.02:
            ctx.flag(
                f"smoothness of {name!r} did not move from its start: the hyperprior, not "
                "the data, is setting it"
            )
    scan = fit.phase_scan
    if scan is not None:
        values = np.sort(np.asarray(scan.values))[::-1]
        if values.size > 1 and values[0] - values[1] < 1.0:
            ctx.flag(
                "conjunction scan: the best two phases are within 1 nat; the orbit may be "
                "the component-swapped mirror"
            )
    amplitude_scan = fit.k_scan
    if amplitude_scan is not None and amplitude_scan.gain > 0.0:
        moved = []
        for star, start, best in zip(
            fit.dis.stars, amplitude_scan.start, amplitude_scan.best, strict=True
        ):
            if start > 0.0 and best > 0.0 and max(best / start, start / best) > 1.5:
                moved.append(f"K_{star.name} from {start:.1f} to {best:.1f} km/s")
        if moved:
            ctx.flag(
                "the semi-amplitude scan moved "
                + ", ".join(moved)
                + f" ({amplitude_scan.gain:+.0f} nats): the starting value sat in another "
                "basin of the likelihood, which L-BFGS alone would not have left"
            )


def _refuse_diverged(ctx: _Context, fit: Fit) -> None:
    """Stop the star when the disentangling diverged, before anything is measured from it.

    The residual z-score rms separates a fit from a failure of the optimizer. It is near 1
    wherever the noise model describes the data and near 2 to 3 at a wrong period on the
    blind route. It was 45 in the one divergence of the D62 benchmark, where the
    component spectra correlated with the injected ones at 0.07 and 0.02 and the
    semi-amplitudes were constrained by their priors alone. Everything downstream (labels,
    templates, epoch velocities) is measured against those spectra, so every later product
    would be computed from a failed fit and reported as a measurement. The ceiling is
    ``z_rms_max``.
    """
    z = float(fit.z_rms)
    if not z > float(ctx.settings.z_rms_max):
        return
    message = (
        f"the disentangling diverged: residual z-score rms {z:.1f}, above the z_rms_max "
        f"ceiling of {float(ctx.settings.z_rms_max):g} (a healthy fit sits near 1). The fit "
        "is not usable, so the label and velocity stages were not run: velocities measured "
        "against these components would carry no measurement. Check the noise model first "
        "(the declared uncertainties, and noise_correlation where the spectra were "
        "resampled onto a common grid), then the priors (the period, the semi-amplitudes "
        "and the conjunction), which is where a fit of this kind is usually started wrong."
    )
    ctx.flag(message)
    raise RuntimeError(message)


def _table_failure(table) -> str | None:
    """Why the velocity table as a whole contains no measurement, or ``None``.

    A median R-squared below zero means the templates fit the epochs worse than no template,
    and no usable epoch means nothing was measured. In both cases the rows are the output
    of a failed correlation, not velocities.
    """
    reasons = []
    median = _finite_median(table.r_squared)
    if median < 0.0:
        reasons.append(
            f"median R-squared {median:.2f} (the templates fit the epochs worse than no "
            "template at all)"
        )
    if not int(table.good.sum()):
        reasons.append(f"no usable epoch of {table.n_epochs}")
    return "; ".join(reasons) or None


def _finite_median(values) -> float:
    """Median of the finite entries, ``nan`` when there are none (as in a failed column)."""
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float("nan")


def _assess_table(ctx: _Context, table) -> None:
    failure = _table_failure(table)
    if failure is not None:
        ctx.flag(
            f"the velocity table failed: {failure}. It is written marked FAILED, and no "
            "orbit is fitted to it; the rows are not measurements"
        )
    n_bad = int((~table.good).sum())
    if n_bad:
        ctx.flag(
            f"{n_bad} of {table.n_epochs} epochs unusable in the velocity table "
            f"({int(table.blended.sum())} blended, "
            f"{int(np.any(table.at_edge, axis=0).sum())} at the search edge)"
        )
    for i, name in enumerate(table.names):
        weak = int((table.delta_chi2[i] < 25.0).sum())
        if weak:
            ctx.flag(f"{name} is weakly detected (delta chi2 < 25) in {weak} epoch(s)")
    if not all(table.absolute):
        ctx.flag(
            "velocities are differential: each component carries its own unidentified zero "
            "point, so the systemic velocities below are meaningless and the orbit fit uses "
            "one gamma per component"
        )


def _assess_orbit(ctx: _Context, fit: Fit, orbit) -> None:
    if fit.mode != "keplerian":
        return
    k_dis = np.asarray(fit.orbit()["k"], dtype=float)
    k_tab = np.asarray(orbit.k, dtype=float)
    err = np.asarray(orbit.errors["k"], dtype=float)
    for i, name in enumerate(orbit.names):
        gap = abs(k_tab[i] - k_dis[i])
        tolerance = max(0.05 * abs(k_dis[i]), 3.0 * (err[i] if np.isfinite(err[i]) else 0.0))
        if gap > tolerance:
            ctx.flag(
                f"K_{name} from the velocity table ({k_tab[i]:.2f} km/s) disagrees with the "
                f"disentangling ({k_dis[i]:.2f}) by {gap:.2f}: the templates or the light "
                "fractions deserve a look"
            )
    dof = orbit.n_points - orbit.n_parameters
    if dof > 0 and orbit.chi2 / dof > 5.0:
        ctx.flag(
            f"orbit from the table: reduced chi-square {orbit.chi2 / dof:.1f}; the scatter "
            "about the Keplerian is far above the per-epoch errors"
        )


# -- describing ------------------------------------------------------------------


def _applied_widths(spec: LSF, dataset: Dataset, key: str):
    if spec.is_per_epoch:
        widths = spec.widths(dataset, key).tolist() if key in dataset.instruments else []
        return widths[0] if len(widths) == 1 else widths
    return np.asarray(spec.sigma_kms).tolist()


def _describe_dataset(dataset: Dataset, lsf) -> dict[str, Any]:
    n_pixels = sum(e.n_pixels for e in dataset)
    n_good = sum(int(e.good.sum()) for e in dataset)
    return {
        "n_epochs": dataset.n_epochs,
        "frame": dataset.frame,
        "medium": dataset[0].medium,
        "instruments": list(dataset.instruments),
        # The widths as applied: a per-epoch declaration is reported as the distinct
        # widths the epochs declared (one number when they agree), never as the sentinel.
        "lsf_sigma_kms": {k: _applied_widths(v, dataset, k) for k, v in lsf.items()},
        "lsf_per_epoch": [k for k, v in lsf.items() if v.is_per_epoch],
        "bjd": dataset.bjd.tolist(),
        "wavelength_angstrom": [
            float(min(e.wave[0] for e in dataset)),
            float(max(e.wave[-1] for e in dataset)),
        ],
        "n_pixels": int(n_pixels),
        "good_pixel_fraction": float(n_good / n_pixels) if n_pixels else None,
    }


def _describe_declaration(dis: Disentangler) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mode": "keplerian" if dis.orbit is not None else "velocity",
        "component_names": [s.name for s in dis.stars],
        "components": [{"name": s.name, "light": s.light} for s in dis.stars],
        "extra_components": [
            n for n in dis.component_names if n not in {s.name for s in dis.stars}
        ],
        "velocity_budget_kms": dis.velocity_budget.total,
        "grid": {
            "n": dis.grid.n,
            "dv_kms": dis.grid.dv_kms,
            "wavelength_angstrom": [float(dis.grid.wave[0]), float(dis.grid.wave[-1])],
        },
    }
    if dis.orbit is not None:
        out["orbit_prior"] = {
            "period": _describe_spec(_spec(dis.orbit.period, "period")),
            "ecc": _describe_spec(dis.orbit.ecc),
            "k": [_describe_spec(_spec(k, "k")) for k in dis.orbit.k]
            if not isinstance(dis.orbit.k, Spec)
            else _describe_spec(dis.orbit.k),
        }
    out["noise_model"] = _describe_noise(dis)
    return out


def _describe_noise(dis: Disentangler) -> dict[str, Any]:
    declared = dis.noise_correlation
    if declared is None:
        return {"kind": "diagonal"}
    if isinstance(declared, Spec):
        return {"kind": "ar1", "correlation": "fitted", "prior": _describe_spec(declared)}
    values = dis.noise_correlation_per_epoch()
    by_instrument: dict[str, float] = {}
    for epoch, phi in zip(dis.dataset, values, strict=True):
        by_instrument.setdefault(epoch.instrument, float(phi))
    return {"kind": "ar1", "correlation": by_instrument}


def _noise_words(dis: Disentangler) -> str:
    described = _describe_noise(dis)
    if described["kind"] == "diagonal":
        return "diagonal"
    if described["correlation"] == "fitted":
        return "AR(1), correlation fitted"
    return "AR(1), lag-one " + ", ".join(
        f"{k} {v:.2f}" for k, v in described["correlation"].items()
    )


def _describe_fit(fit: Fit) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mode": fit.mode,
        "potential": float(fit.result.potential),
        "grad_norm": float(fit.result.grad_norm),
        "num_steps": int(fit.result.num_steps),
        "converged": bool(fit.result.converged),
        "z_rms": float(fit.z_rms),
        "hyper": {n: dict(v) for n, v in fit.hyper.items()},
    }
    if fit.phase_scan is not None:
        out["phase_scan"] = {
            "t_conj": float(fit.phase_scan.best),
            "contrast_nats": float(fit.phase_scan.contrast),
        }
    if fit.k_scan is not None:
        names = [s.name for s in fit.dis.stars]
        out["k_scan"] = {
            "n_trials": int(fit.k_scan.n_trials),
            "best": {n: float(k) for n, k in zip(names, fit.k_scan.best, strict=True)},
            "start": {n: float(k) for n, k in zip(names, fit.k_scan.start, strict=True)},
            "gain_nats": float(fit.k_scan.gain),
            "contrast_nats": float(fit.k_scan.contrast),
            "used": bool(fit.k_scan.gain > 0.0),
            "hold_losses_nats": {
                names[i]: float(v) for i, v in fit.k_scan.hold_losses.items() if i < len(names)
            },
            "notes": list(fit.k_scan.notes),
            "prior_scales": {k: float(v) for k, v in fit.k_scan.prior_scales.items()},
        }
    noise = fit.noise_correlation()
    if noise is not None:
        out["noise_correlation"] = noise
    # The per-epoch velocities of the joint fit: the Keplerian at the epochs, without a
    # systemic velocity, which the disentangling does not determine (the table's are
    # absolute when the label fit set the zero points).
    velocities = np.asarray(fit.velocities())
    out["velocities"] = {
        n: velocities[i].tolist() for i, n in enumerate(s.name for s in fit.dis.stars)
    }
    if fit.mode == "keplerian":
        params = fit.orbit()
        out.update(
            {
                "period": float(params["period"]),
                "t_conj": float(params["t_conj"]),
                "ecc": float(params["ecc"]),
                "omega_rad": float(params["omega"]),
                "k": {s.name: float(fit.star(s.name)["k"]) for s in fit.dis.stars},
            }
        )
    velocities = np.asarray(fit.velocities())
    out["velocities"] = {s.name: velocities[i].tolist() for i, s in enumerate(fit.dis.stars)}
    return out


def _describe_match(match) -> dict[str, Any]:
    laplace = match.errors("laplace")
    drawn = match.errors("draws") if match.draws is not None else {}
    components = {}
    for name, labels in match.labels.items():
        entry = dict(labels)
        for label, value in laplace.get(name, {}).items():
            entry[f"{label}_err"] = value
        for label, value in drawn.get(name, {}).items():
            entry[f"{label}_err_draws"] = value
        entry["fixed"] = list(match.fixed.get(name, []))
        entry["nearest_node"] = match.nearest_node(name)
        components[name] = entry
    epoch_fit = getattr(match, "epoch_fit", None)
    statistics = getattr(match, "statistics", None)
    return {
        "compare": match.compare,
        "epoch_operator": None if statistics is None else _describe_operator(statistics),
        "components": components,
        "flux_ratio": dict(match.flux_ratio),
        "flux_ratio_errors": dict(match.flux_ratio_errors),
        "light_declared": dict(
            zip(match.names, match.assumptions.get("light_fractions", []), strict=False)
        ),
        "at_bounds": dict(match.at_bounds),
        "notes": [] if epoch_fit is None else list(epoch_fit.notes),
        "epoch_fit": None
        if epoch_fit is None
        else {
            "status": epoch_fit.status,
            "iterations": int(epoch_fit.iterations),
            "lm_runs": int(epoch_fit.lm_runs),
            "restart_rounds": len(epoch_fit.rounds),
            "moved_by": list(epoch_fit.moved_by),
            "seconds": float(epoch_fit.seconds),
        },
        "radius_ratio": dict(match.radius_ratio),
        "chi2": float(match.chi2),
        "chi2_nearest_node": float(match.chi2_nearest_node),
        "chi2_continuum": float(match.chi2_continuum),
        "n_pixels_used": int(match.n_pixels_used),
        "multimodal": bool(match.multimodal),
        "learned_nothing": sorted(k for k, v in match.posterior_over_prior.items() if v > 0.8),
        "correlations_flagged": [list(t) for t in match.flagged_correlations()],
        "assumptions": dict(match.assumptions),
        "errors_from_draws": match.draws is not None,
    }


def _describe_operator(statistics) -> dict[str, Any]:
    """The stellar operator the epoch comparison ran with, for ``labels`` in result.json.

    ``quadrature_sigma_kms`` is the declared width reduced for the library's resolving power
    and ``operator_sigma_kms`` what the operator applied once the model grid's own smoothing,
    ``grid_variance_kms2``, was removed from it (:meth:`albireo.Fit.epoch_statistics`).
    There is one entry per epoch group and one width per LSF anchor. ``notes`` records a
    floored width or a model grid coarser than the width, each also flagged.
    """
    operator = statistics.operator_sigma_kms or statistics.lsf_sigma_kms
    return {
        "dv_kms": float(statistics.grid.dv_kms),
        "library_resolving_power": statistics.library_resolving_power,
        "grid_variance_kms2": float(statistics.grid_variance_kms2),
        "quadrature_sigma_kms": [[k, list(v)] for k, v in statistics.lsf_sigma_kms],
        "operator_sigma_kms": [[k, list(v)] for k, v in operator],
        "notes": list(statistics.notes),
    }


def _describe_table(table) -> dict[str, Any]:
    failure = _table_failure(table)
    return {
        "status": "ok" if failure is None else "failed",
        "failure": failure,
        "names": list(table.names),
        "bjd": table.bjd.tolist(),
        "instrument": list(table.instrument),
        "velocity": {n: table.velocity[i].tolist() for i, n in enumerate(table.names)},
        "sigma": {n: table.sigma[i].tolist() for i, n in enumerate(table.names)},
        "light": {n: table.light[i].tolist() for i, n in enumerate(table.names)},
        "light_mode": table.light_mode,
        "delta_chi2": {n: table.delta_chi2[i].tolist() for i, n in enumerate(table.names)},
        "good": table.good.tolist(),
        "n_usable": int(table.good.sum()),
        "blended": table.blended.tolist(),
        "absolute": {n: bool(table.absolute[i]) for i, n in enumerate(table.names)},
        "absolute_all": bool(all(table.absolute)),
        "reduced_chi2_median": float(np.nanmedian(table.reduced_chi2)),
        "r_squared_median": _finite_median(table.r_squared),
        "wilson_slope": None if table.wilson() is None else float(table.wilson()[0]),
        "noise_correlation": table.settings.get("noise_correlation"),
    }


def _describe_orbit(orbit, period_source) -> dict[str, Any]:
    e = orbit.errors
    dof = orbit.n_points - orbit.n_parameters
    return {
        "period_source": period_source,
        "period": float(orbit.period),
        "period_err": float(e["period"]),
        "t_conj": float(orbit.t_conj),
        "t_conj_err": float(e["t_conj"]),
        "ecc": float(orbit.ecc),
        "ecc_err": float(e["ecc"]),
        "omega_deg": float(math.degrees(orbit.omega)),
        "omega_err_deg": float(math.degrees(e["omega"])),
        "k": {n: float(k) for n, k in zip(orbit.names, orbit.k, strict=True)},
        "k_err": {n: float(k) for n, k in zip(orbit.names, e["k"], strict=True)},
        "gamma": {n: float(g) for n, g in zip(orbit.names, orbit.gamma, strict=True)},
        "gamma_err": {n: float(g) for n, g in zip(orbit.names, e["gamma"], strict=True)},
        "gamma_mode": orbit.gamma_mode,
        "k_held": list(orbit.held),
        "q": orbit.mass_ratio,
        "m_sin3i_msun": orbit.minimum_masses(),
        "a_sini_rsun": orbit.projected_semiaxes(),
        "rms_kms": {n: float(r) for n, r in zip(orbit.names, orbit.rms, strict=True)},
        "chi2": float(orbit.chi2),
        "dof": int(dof),
        "n_points": int(orbit.n_points),
    }


def _describe_posterior(posterior) -> dict[str, Any]:
    samples = posterior.samples
    out: dict[str, Any] = {"n_draws": int(np.asarray(samples["period"]).shape[0])}
    for site in ("period", "t_conj"):
        values = np.asarray(samples[site]).ravel()
        out[site] = {"mean": float(values.mean()), "std": float(values.std())}
    out["k"] = {}
    for star in posterior.dis.stars:
        row = posterior.star(star.name)
        out["k"][star.name] = {"mean": row["k"], "std": row["k_std"], "hdi95": list(row["k_hdi"])}
    extra = getattr(posterior.mcmc, "get_extra_fields", dict)()
    if "diverging" in extra:
        out["divergences"] = int(np.sum(np.asarray(extra["diverging"])))
    return out


def _wrap_period(diff: float, period: float) -> float:
    """``diff`` reduced to ``(-period/2, period/2]``: a conjunction is defined modulo P."""
    return float((diff + 0.5 * period) % period - 0.5 * period)


def _wrap_angle(diff: float) -> float:
    """``diff`` in radians reduced to ``(-pi, pi]``."""
    return float((diff + math.pi) % (2.0 * math.pi) - math.pi)


def _pull(diff: float, err: float | None) -> float | None:
    if err is None or not np.isfinite(err) or err <= 0.0:
        return None
    return float(diff / err)


def _truth_spectra(truth: Mapping[str, Any], fit: Fit) -> np.ndarray | None:
    """The injected stellar deviation spectra interpolated onto the fit's grid."""
    if "components" not in truth or "grid" not in truth:
        return None
    grid, source = fit.dis.grid, truth["grid"]
    rows = [
        np.interp(grid.wave, np.asarray(source.wave), np.asarray(c), left=0.0, right=0.0)
        for c in truth["components"]
    ]
    return np.stack(rows)


def _compare_spectra(fit: Fit, truth_spectra: np.ndarray, windows) -> dict[str, Any]:
    """Fidelity of the disentangled components against the injected ones.

    The comparison is restricted to the pixels the data cover (the grid has margins the
    data never constrain) and, per component, to what the star contributes. It gives the
    root-mean-square difference, the correlation, the standardized difference against the
    reported band (``pull_rms``, near 1 when the band is calibrated), and the
    equivalent-width ratio in each window.
    """
    grid = fit.dis.grid
    wave = grid.wave
    data_lo = min(float(e.wave[0]) for e in fit.dis.dataset)
    data_hi = max(float(e.wave[-1]) for e in fit.dis.dataset)
    covered = (wave >= data_lo) & (wave <= data_hi)
    names = [s.name for s in fit.dis.stars]
    rows = [fit.dis.component_names.index(n) for n in names]
    d_hat, std = fit.spectra()[rows], fit.std()[rows]
    d_true = truth_spectra[: len(names)]
    step = np.gradient(wave)
    if not windows:
        windows = {"band": [(data_lo, data_hi)]}
    out: dict[str, Any] = {}
    for i, name in enumerate(names):
        diff = d_hat[i] - d_true[i]
        sel = covered
        entry: dict[str, Any] = {
            "rms": float(np.sqrt(np.mean(diff[sel] ** 2))),
            "rms_truth": float(np.sqrt(np.mean(d_true[i][sel] ** 2))),
            "corr": float(np.corrcoef(d_hat[i][sel], d_true[i][sel])[0, 1])
            if np.std(d_true[i][sel]) > 0 and np.std(d_hat[i][sel]) > 0
            else float("nan"),
            "pull_rms": float(np.sqrt(np.mean((diff[sel] / np.maximum(std[i][sel], 1e-12)) ** 2))),
            "depth_ratio": float(np.min(d_hat[i][sel]) / np.min(d_true[i][sel]))
            if np.min(d_true[i][sel]) < 0
            else float("nan"),
        }
        ew: dict[str, Any] = {}
        for wname, ranges in windows.items():
            mask = np.zeros(wave.size, dtype=bool)
            for lo, hi in ranges:
                mask |= (wave >= float(lo)) & (wave <= float(hi))
            mask &= covered
            if not mask.any():
                continue
            ew_true = float(np.sum(-d_true[i][mask] * step[mask]))
            ew_hat = float(np.sum(-d_hat[i][mask] * step[mask]))
            ew[wname] = {
                "ew_true_angstrom": ew_true,
                "ew_fit_angstrom": ew_hat,
                "ratio": float(ew_hat / ew_true) if ew_true != 0.0 else float("nan"),
                "rms": float(np.sqrt(np.mean(diff[mask] ** 2))),
            }
        entry["windows"] = ew
        out[name] = entry
    return out


def _exchange_rms(v_fit, v_true, good, absolute) -> tuple[float, float]:
    """The rms velocity error of a two-component fit as named and with the pair exchanged."""
    v_fit = np.asarray(v_fit, dtype=float)
    v_true = np.asarray(v_true, dtype=float)
    good = np.asarray(good, dtype=bool)

    def rms(order) -> float:
        total = 0.0
        for i in range(2):
            diff = v_fit[i] - v_true[order[i]]
            if not absolute[i]:
                diff = diff - np.mean(diff[good])
            total += float(np.mean(diff[good] ** 2))
        return math.sqrt(total / 2.0)

    return rms((0, 1)), rms((1, 0))


def _exchanged_truth(truth: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """The truth block with its two components in the other order."""
    out = dict(truth)
    for key in ("k", "light_fractions"):
        if key in out:
            out[key] = list(np.asarray(out[key]).reshape(-1)[::-1])
    for key in ("velocities", "components"):
        if key in out:
            out[key] = np.asarray(out[key])[::-1]
    if isinstance(out.get("labels"), Mapping) and all(n in out["labels"] for n in names):
        out["labels"] = {names[0]: out["labels"][names[1]], names[1]: out["labels"][names[0]]}
    return out


def _compare_truth(ctx: _Context, fit: Fit, table, orbit, match) -> tuple[str, dict[str, Any]]:
    truth = dict(ctx.star.truth or {})
    names = [s.name for s in fit.dis.stars]
    lines = ["Against the injected truth:"]
    out: dict[str, Any] = {
        "note": "differences (result minus injected) and pulls (difference over the quoted "
        "error), except the rms entries",
        "components_exchanged": False,
    }
    # A pair too alike for the mass-order convention to fix can be recovered in the other
    # order. The recovery is then compared with the injected values in that order, and the
    # exchange is flagged, since it is a limit of the convention and not of the fit.
    v_true = np.asarray(truth["velocities"], dtype=float) if "velocities" in truth else None
    if len(names) == 2 and v_true is not None and v_true.shape[0] == 2:
        as_named = exchanged = None
        if v_true.shape == table.velocity.shape:
            good = table.good & np.all(np.isfinite(table.velocity), axis=0)
            if int(good.sum()) >= 3:
                as_named, exchanged = _exchange_rms(table.velocity, v_true, good, table.absolute)
        if as_named is None and fit.mode == "keplerian":
            v_fit = np.asarray(fit.velocities())
            if v_fit.shape == v_true.shape:
                as_named, exchanged = _exchange_rms(
                    v_fit, v_true, np.ones(v_true.shape[1], dtype=bool), (False, False)
                )
        if as_named is not None and exchanged * 3.0 < as_named:
            truth = _exchanged_truth(truth, names)
            out["components_exchanged"] = True
            ctx.flag(
                f"components recovered in the other order: {names[0]} moves like the injected "
                f"{names[1]} (velocity rms {as_named:.1f} km/s as named, {exchanged:.1f} "
                "exchanged); the truth is compared in that order, the pair being too alike "
                "for the mass-order convention to fix"
            )
            lines.append(
                f"  the components came out in the other order ({names[0]} is the injected "
                f"{names[1]}); compared in that order"
            )
    period_true = float(truth["period"]) if "period" in truth else None
    if "k" in truth:
        k_true = np.asarray(truth["k"], dtype=float)
        if fit.mode == "keplerian":
            k_fit = np.array([fit.star(n)["k"] for n in names])
            out["k_disentangling"] = {n: float(k_fit[i] - k_true[i]) for i, n in enumerate(names)}
            lines.append(
                "  K from the disentangling: "
                + ", ".join(
                    f"{n} {k_fit[i]:.3f} (truth {k_true[i]:g}, {k_fit[i] - k_true[i]:+.3f})"
                    for i, n in enumerate(names)
                )
            )
        if orbit is not None:
            out["k_table"] = {n: float(orbit.k[i] - k_true[i]) for i, n in enumerate(names)}
            out["k_table_pull"] = {
                n: _pull(orbit.k[i] - k_true[i], orbit.errors["k"][i]) for i, n in enumerate(names)
            }
            lines.append(
                "  K from the velocity table: "
                + ", ".join(
                    f"{n} {orbit.k[i]:.3f}+-{orbit.errors['k'][i]:.3f} "
                    f"(truth {k_true[i]:g}, {orbit.k[i] - k_true[i]:+.3f})"
                    for i, n in enumerate(names)
                )
            )
    if fit.mode == "keplerian":
        params = fit.orbit()
        elements: dict[str, Any] = {}
        if period_true is not None:
            elements["period"] = float(params["period"]) - period_true
        if "ecc" in truth:
            elements["ecc"] = float(params["ecc"]) - float(truth["ecc"])
        if "omega" in truth and "ecc" in truth and float(truth["ecc"]) > 0.0:
            elements["omega_deg"] = math.degrees(
                _wrap_angle(float(params["omega"]) - float(truth["omega"]))
            )
        if "t_conj" in truth:
            p = period_true if period_true is not None else float(params["period"])
            elements["t_conj"] = _wrap_period(float(params["t_conj"]) - float(truth["t_conj"]), p)
        if elements:
            out["elements_disentangling"] = elements
            lines.append(
                "  elements from the disentangling (result minus truth): "
                + ", ".join(f"{k} {v:+.5g}" for k, v in elements.items())
            )
    if orbit is not None:
        e = orbit.errors
        elements = {}
        pulls = {}
        if period_true is not None:
            elements["period"] = float(orbit.period - period_true)
            pulls["period"] = _pull(elements["period"], e.get("period"))
            out["period"] = elements["period"]
            lines.append(
                f"  period from the table {orbit.period:.5f} d (truth {period_true:g}, "
                f"{elements['period']:+.5f})"
            )
        if "ecc" in truth:
            elements["ecc"] = float(orbit.ecc - float(truth["ecc"]))
            pulls["ecc"] = _pull(elements["ecc"], e.get("ecc"))
        if "omega" in truth and "ecc" in truth and float(truth["ecc"]) > 0.0:
            d_omega = _wrap_angle(float(orbit.omega) - float(truth["omega"]))
            elements["omega_deg"] = math.degrees(d_omega)
            pulls["omega_deg"] = _pull(d_omega, e.get("omega"))
        if "t_conj" in truth:
            p = period_true if period_true is not None else float(orbit.period)
            elements["t_conj"] = _wrap_period(float(orbit.t_conj) - float(truth["t_conj"]), p)
            pulls["t_conj"] = _pull(elements["t_conj"], e.get("t_conj"))
        if elements:
            out["elements_table"] = elements
            out["elements_table_pull"] = pulls
            lines.append(
                "  elements from the table (result minus truth): "
                + ", ".join(f"{k} {v:+.5g}" for k, v in elements.items())
            )
        if "gamma" in truth:
            gamma_true = float(truth["gamma"])
            out["gamma"] = {n: float(orbit.gamma[i] - gamma_true) for i, n in enumerate(names)}
            out["gamma_pull"] = {
                n: _pull(orbit.gamma[i] - gamma_true, e["gamma"][i]) if table.absolute[i] else None
                for i, n in enumerate(names)
            }
            note = "" if all(table.absolute) else "   (differential: not comparable)"
            lines.append(
                "  systemic velocity: "
                + ", ".join(f"{n} {orbit.gamma[i]:+.3f}" for i, n in enumerate(names))
                + f" (truth {gamma_true:+g}){note}"
            )
    if "velocities" in truth:
        v_true = np.asarray(truth["velocities"], dtype=float)
        if v_true.shape == table.velocity.shape:
            # The injected velocity of every epoch, in the order the components were
            # compared in, so that a reader of result.json can draw the table against it.
            out["epoch_velocities_true"] = {n: v_true[i].tolist() for i, n in enumerate(names)}
            rows, pulls, counts = {}, {}, {}
            for i, n in enumerate(names):
                good = table.good & np.isfinite(table.velocity[i]) & np.isfinite(table.sigma[i])
                diff = table.velocity[i] - v_true[i]
                if not table.absolute[i]:
                    diff = diff - np.nanmean(diff[good]) if good.any() else diff
                rows[n] = float(np.sqrt(np.mean(diff[good] ** 2))) if good.any() else float("nan")
                pulls[n] = (
                    float(np.sqrt(np.mean((diff[good] / table.sigma[i][good]) ** 2)))
                    if good.any()
                    else float("nan")
                )
                counts[n] = int(good.sum())
            out["velocity_rms"] = rows
            out["velocity_pull_rms"] = pulls
            out["velocity_n_used"] = counts
            lines.append(
                "  epoch velocities rms error: "
                + ", ".join(f"{n} {r:.3f} km/s (pull rms {pulls[n]:.2f})" for n, r in rows.items())
                + ("" if all(table.absolute) else " (after removing each zero point)")
            )
            if fit.mode == "keplerian":
                v_fit = np.asarray(fit.velocities())
                if v_fit.shape == v_true.shape:
                    kep = {}
                    for i, n in enumerate(names):
                        diff = v_fit[i] - v_true[i]
                        diff = diff - np.mean(diff)  # the joint fit has no systemic velocity
                        kep[n] = float(np.sqrt(np.mean(diff**2)))
                    out["velocity_rms_keplerian"] = kep
                    lines.append(
                        "  Keplerian velocities of the disentangling, rms error after removing "
                        "the mean: " + ", ".join(f"{n} {r:.3f} km/s" for n, r in kep.items())
                    )
    if "labels" in truth and match is not None:
        rows = {}
        errors = match.errors("laplace")
        for n, labels in truth["labels"].items():
            if n not in match.labels:
                continue
            got = match.labels[n]
            rows[n] = {k: float(got[k] - v) for k, v in labels.items() if k in got}
            rows[n].update(
                {
                    f"{k}_pull": _pull(got[k] - v, errors.get(n, {}).get(k))
                    for k, v in labels.items()
                    if k in got
                }
            )
            lines.append(
                f"  labels {n}: "
                + ", ".join(
                    f"{k} {got[k]:.3g} (truth {v:g}, {got[k] - v:+.3g})"
                    for k, v in labels.items()
                    if k in got
                )
            )
        out["labels"] = rows
    if "light_fractions" in truth:
        ell_true = np.asarray(truth["light_fractions"], dtype=float).reshape(-1)
        declared = {s.name: float(s.light) for s in fit.dis.stars}
        out["light_declared"] = {n: declared[n] - ell_true[i] for i, n in enumerate(names)}
        lines.append(
            "  light fractions used by the fit (minus truth): "
            + ", ".join(
                f"{n} {declared[n]:.3f} ({declared[n] - ell_true[i]:+.3f})"
                for i, n in enumerate(names)
            )
        )
        if match is not None:
            fitted = dict(match.flux_ratio)
            out["light_label_fit"] = {
                n: float(fitted[n] - ell_true[i]) for i, n in enumerate(names) if n in fitted
            }
            lines.append(
                "  light fractions measured by the label fit (minus truth): "
                + ", ".join(
                    f"{n} {fitted[n]:.3f} ({fitted[n] - ell_true[i]:+.3f})"
                    for i, n in enumerate(names)
                    if n in fitted
                )
            )
    truth_spectra = _truth_spectra(truth, fit)
    if truth_spectra is not None:
        out["spectra"] = _compare_spectra(fit, truth_spectra, truth.get("windows"))
        lines.append(
            "  component spectra (rms of fit minus truth over the covered band, correlation, "
            "pull rms against the band): "
            + ", ".join(
                f"{n} {v['rms']:.4f} / {v['corr']:.3f} / {v['pull_rms']:.2f}"
                for n, v in out["spectra"].items()
            )
        )
    return "\n".join(lines), out


# -- products --------------------------------------------------------------------


def _files_block(files: Mapping[str, str]) -> str:
    return "Files:\n" + "\n".join(f"  {k:<14s} {v}" for k, v in sorted(files.items()))


def _write_products(ctx: _Context, fit: Fit, table, orbit, match, posterior) -> None:
    from albireo.results import save_fit, write_ascii

    directory, files = ctx.directory, ctx.files
    header = f"star: {ctx.star.name}"
    files["velocities"] = os.fspath(
        table.write(directory / "velocities.rv", header=_velocity_header(ctx, table))
    )
    files["velocities_csv"] = os.fspath(_write_velocity_csv(directory / "velocities.csv", table))
    spectra, std = fit.spectra(), fit.std()
    for i, name in enumerate(fit.dis.component_names):
        path = write_ascii(
            directory / f"spectrum_{_safe_name(name)}.txt",
            fit.dis.grid,
            spectra,
            std,
            component=i,
            header=f"{header}; component {name}",
        )
        files[f"spectrum_{name}"] = os.fspath(path)
    if importlib.util.find_spec("astropy") is not None:
        try:
            files["spectra_fits"] = os.fspath(fit.write_spectra(directory / "spectra.fits"))
        except Exception as exc:
            ctx.flag(f"spectra.fits not written: {type(exc).__name__}: {exc}")
    files["fit"] = os.fspath(save_fit(fit.result, directory / "fit.npz"))
    if orbit is not None:
        (directory / "orbit.txt").write_text(orbit.summary() + "\n", encoding="utf-8")
        files["orbit"] = os.fspath(directory / "orbit.txt")
    if match is not None:
        (directory / "labels.txt").write_text(match.summary() + "\n", encoding="utf-8")
        files["labels"] = os.fspath(directory / "labels.txt")
        for name in match.names:
            path = directory / f"template_{_safe_name(name)}.txt"
            wave = np.asarray(match.wave)
            flux = np.asarray(match.template(name))
            np.savetxt(
                path,
                np.column_stack([wave, flux]),
                header=f"{header}; label-fit model spectrum of {name}, flux on the fit grid",
                fmt="%.6f",
            )
            files[f"template_{name}"] = os.fspath(path)
    if posterior is not None:
        samples = {k: np.asarray(v) for k, v in posterior.samples.items()}
        np.savez_compressed(directory / "posterior.npz", **samples)
        files["posterior"] = os.fspath(directory / "posterior.npz")


def _velocity_header(ctx: _Context, table) -> str:
    """The header of ``velocities.rv``: the star, marked ``FAILED`` where the table is.

    The ``FAILED`` line is under the format line, before any row, which is the only place
    a reader who takes the columns and not the flags will look.
    """
    header = f"star: {ctx.star.name}"
    failure = _table_failure(table)
    return header if failure is None else f"FAILED: {failure}\n{header}"


def _write_template_table(ctx: _Context, table, purpose: str, *, unexchanged=None) -> None:
    """Write the library-template velocity table a stage measured, as a separate product.

    The bootstrap, the light measurement and the semi-amplitude start each measure the
    epochs against library templates at the declared starting labels before anything is
    disentangled. That table determines the period on the search route and gives the
    starting semi-amplitudes on the others, so it is written as soon as it exists
    (``template_velocities.rv`` and ``.csv``), whether or not the stages after it succeed.

    On the search route the delivered table is the chosen orbit's. The bootstrap re-assigns
    the components at the epochs where the correlation exchanged two alike spectra
    (:func:`_orbit_over_candidates`), so some rows have an assignment the correlation did
    not make. The header states this, and ``unexchanged``, the table the period search ran
    on, is written beside it as ``template_velocities_unexchanged.rv`` whenever an epoch
    was re-assigned, so that the period search can be reproduced from the written products.
    """
    directory = ctx.directory
    directory.mkdir(parents=True, exist_ok=True)
    header = (
        f"star: {ctx.star.name}\n"
        f"purpose: {purpose}; velocities against library templates at the declared "
        "starting labels, before the disentangling"
    )
    swapped = int(table.settings.get("reassigned_by_orbit", 0))
    resolved = int(table.settings.get("resolved_by_orbit", 0))
    if swapped:
        header += (
            "\ncomponent assignment: the winning orbit's, not the correlation's; "
            f"{swapped} of {table.n_epochs} epochs re-assigned"
        )
    if resolved:
        header += (
            f"\nsecond minima: decided by the winning orbit at {resolved} of {table.n_epochs} "
            f"epochs, the other minimum taken at "
            f"{int(table.settings.get('second_minimum_by_orbit', 0))}"
        )
    ctx.files["template_velocities"] = os.fspath(
        table.write(directory / "template_velocities.rv", header=header)
    )
    ctx.files["template_velocities_csv"] = os.fspath(
        _write_velocity_csv(directory / "template_velocities.csv", table)
    )
    if (swapped or resolved) and unexchanged is not None:
        # The period search ran on the table before the orbit's decisions, so that table
        # reproduces the periodogram and is kept beside the delivered one.
        ctx.files["template_velocities_unexchanged"] = os.fspath(
            unexchanged.write(
                directory / "template_velocities_unexchanged.rv",
                header=(
                    f"star: {ctx.star.name}\n"
                    f"purpose: {purpose}; velocities against library templates at the declared "
                    "starting labels, before the disentangling\n"
                    "as measured, before the exchange: the table the period search ran on"
                ),
            )
        )


def _write_velocity_csv(path: Path, table) -> Path:
    """The velocity table as CSV: the columns of ``VelocityTable.to_dict`` and more.

    ``at_edge`` is merged over the components, as in the ``.rv`` file. The CSV adds one
    ``at_edge_<component>`` column per component, since a velocity is used wherever its
    own component was measured (D65) and the merged flag does not identify which one was
    lost.
    """
    columns = dict(table.to_dict())
    at_edge = np.broadcast_to(
        np.asarray(table.at_edge, dtype=bool), (table.n_components, table.n_epochs)
    )
    for i, name in enumerate(table.names):
        columns[f"at_edge_{name}"] = at_edge[i]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(columns))
        for j in range(table.n_epochs):
            row = []
            for key, col in columns.items():
                value = col[j]
                if key == "instrument":
                    row.append(str(value))
                elif key == "bjd":
                    # All digits of the float64 are written, so that the period search is
                    # reproducible from the written table (D65).
                    row.append(repr(float(value)))
                elif isinstance(value, bool | np.bool_):
                    row.append(int(value))
                elif key == "n_pix":
                    row.append(int(value))
                else:
                    row.append(f"{float(value):.6f}")
            writer.writerow(row)
    return path


def _write_plots(ctx: _Context, fit: Fit, table, orbit, templates, posterior) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        ctx.flag("plots skipped: matplotlib is not installed (pip install 'albireo[plots]')")
        return
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from albireo import plotting

    directory, files = ctx.directory, ctx.files
    truth = dict(ctx.star.truth or {})

    def save(name: str, make) -> None:
        try:
            fig = make()
            if fig is None:  # nothing to draw (an unmeasured table, for instance)
                return
            path = directory / f"{name}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            files[f"plot_{name}"] = os.fspath(path)
        except Exception as exc:
            ctx.flag(f"figure {name} not written: {type(exc).__name__}: {exc}")

    grid = fit.dis.grid
    truth_spectra = None
    if "components" in truth and "grid" in truth:
        source = truth["grid"]
        truth_spectra = np.stack(
            [
                np.interp(grid.wave, np.asarray(source.wave), np.asarray(c), left=0.0, right=0.0)
                for c in truth["components"]
            ]
        )
        n_extra = len(fit.dis.component_names) - truth_spectra.shape[0]
        if n_extra > 0:
            truth_spectra = np.vstack([truth_spectra, np.zeros((n_extra, grid.n))])

    def spectra_figure():
        fig, _ = plotting.plot_spectra(
            grid,
            fit.spectra(),
            std=fit.std(),
            truth=truth_spectra,
            labels=list(fit.dis.component_names),
            flux=True,
        )
        fig.suptitle(f"{ctx.star.name}: disentangled components (band = +-2 sigma)")
        return fig

    def residual_figure():
        fig, _ = plotting.plot_residual_zscores(
            fit.dis.model.problem_at(fit.theta), fit.marginal().d_hat, bjd=fit.dis.dataset.bjd
        )
        fig.suptitle(f"{ctx.star.name}: whitened residuals")
        return fig

    def velocity_figure():
        v_truth = truth.get("velocities")
        fig, _ = plotting.plot_velocity_table(
            table,
            orbit=orbit,
            truth=None if v_truth is None else np.asarray(v_truth),
        )
        fig.suptitle(f"{ctx.star.name}: epoch velocities" + (" and orbit" if orbit else ""))
        return fig

    def scan_figure():
        fig, _ = plotting.plot_phase_scan(fit.phase_scan)
        fig.suptitle(ctx.star.name)
        return fig

    def surface_figure():
        from albireo.todcor import todcor_surface

        separation = np.abs(table.velocity[0] - table.velocity[1])
        j = int(np.nanargmin(np.where(np.isfinite(separation), separation, np.inf)))
        finite = np.abs(table.velocity)[np.isfinite(table.velocity)]
        if finite.size == 0:
            return None
        span = float(np.max(finite)) + 40.0
        lsf_sigma = {k: v.sigma_kms for k, v in fit.dis.lsf.items() if isinstance(v, LSF)}
        for k, v in fit.dis.lsf.items():
            if not isinstance(v, LSF):
                lsf_sigma[k] = float(v)
        anchors = {
            k: v.anchors_angstrom
            for k, v in fit.dis.lsf.items()
            if isinstance(v, LSF) and v.anchors_angstrom is not None
        }
        surface = todcor_surface(
            fit.dis.dataset,
            j,
            templates[:2],
            v_range=(-span, span),
            light=[s.light for s in fit.dis.stars],
            lsf_sigma_v=lsf_sigma,
            lsf_anchors_angstrom=anchors or None,
            step=2,
        )
        v_truth = truth.get("velocities")
        fig, _ = plotting.plot_todcor_surface(
            surface, truth=None if v_truth is None else np.asarray(v_truth)[:, j]
        )
        fig.suptitle(f"{ctx.star.name}: TODCOR surface of the most blended epoch ({j})")
        return fig

    def rv_curve_figure():
        fig, _ = plotting.plot_rv_curve(posterior.samples, fit.dis.dataset.bjd)
        fig.suptitle(f"{ctx.star.name}: posterior orbit draws")
        return fig

    save("spectra", spectra_figure)
    save("residuals", residual_figure)
    save("velocities", velocity_figure)
    if fit.phase_scan is not None:
        save("phase_scan", scan_figure)
    if len(templates) == 2:
        save("todcor_surface", surface_figure)
    if posterior is not None:
        save("rv_curve", rv_curve_figure)


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------

_WORKER_CONFIG: PipelineConfig | None = None


def _thread_environment(n_jobs: int) -> dict[str, str]:
    """Environment variables capping each worker's threads at ``cpu_count // n_jobs``.

    XLA's CPU backend and every BLAS size their thread pools to the whole machine, so N
    workers would run N x cores threads between them. The cap is a precaution against
    that oversubscription, not a measured gain: on the recorded benchmark eight capped and
    eight uncapped workers finished the same batch in 54.1 and 54.7 s
    (``docs/benchmarks.md``, "The pipeline in worker processes"; D58). It has no measured
    cost there, and oversubscription has been observed in BLAS-heavy stages (the 32-thread
    OpenBLAS of the D50 record).
    """
    cores = os.cpu_count() or 1
    threads = max(1, cores // max(1, n_jobs))
    env = {
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "MPLBACKEND": "Agg",
    }
    flags = os.environ.get("XLA_FLAGS", "")
    if "intra_op_parallelism_threads" not in flags:
        env["XLA_FLAGS"] = (
            f"{flags} --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={threads}"
        ).strip()
    return env


@contextlib.contextmanager
def _environment(values: Mapping[str, str]) -> Iterator[None]:
    """Set environment variables for the duration of a block, restoring them after."""
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _worker_init(config: PipelineConfig, env: Mapping[str, str]) -> None:
    global _WORKER_CONFIG
    os.environ.update(env)
    _WORKER_CONFIG = config


def _star_task(star: StarConfig, directory: str, progress: bool) -> StarResult:
    assert _WORKER_CONFIG is not None, "the worker was not initialized"
    return _run_star_guarded(
        star, _WORKER_CONFIG, Path(directory), progress=progress, keep_live=False
    )


def _resolve_jobs(jobs: int | str | None, n_stars: int) -> int:
    if jobs is None or jobs == 1:
        return 1
    if isinstance(jobs, str):
        if jobs.lower() != "auto":
            raise ValueError("jobs must be an integer or 'auto'")
        jobs = 0
    jobs = int(jobs)
    if jobs <= 0:
        cores = os.cpu_count() or 1
        jobs = max(1, cores // 4)
    return max(1, min(jobs, n_stars))


def run_pipeline(
    config: PipelineConfig | str | os.PathLike | Mapping[str, Any],
    *,
    jobs: int | str | None = 1,
    stars: Sequence[str] | None = None,
    progress: bool = True,
) -> PipelineRun:
    """Run every star of a configuration and write the batch products.

    Parameters
    ----------
    config
        A :class:`PipelineConfig`, the path of a TOML file, or the dictionary form.
    jobs
        Worker processes. ``1`` (default) runs in this process and keeps the live objects
        on each :class:`StarResult`. ``"auto"`` or ``0`` uses ``cpu_count // 4``. Any
        larger number runs that many stars at a time, each with its threads capped so the
        workers do not oversubscribe the machine.
    stars
        Run only these names.
    progress
        Print one line per stage per star.

    Returns
    -------
    PipelineRun
        With ``results.json``, ``results.csv``, ``summary.txt`` and, when needed,
        ``failures.txt`` already written into the output directory.

    Notes
    -----
    Stars are independent, so with ``jobs > 1`` they run in a process pool started with
    the ``spawn`` method on every platform. A script that calls this with ``jobs > 1``
    must do so from under ``if __name__ == "__main__":``, because the workers import the
    script as a module and an unguarded call would start the batch again in each of them.
    The ``albireo`` command is guarded.

    The scaling is sub-linear because a single star already occupies several cores, so
    the workers overlap only the serial part of each star (compilation, the Python-side
    scans, the orbit fit, the writing). On the development desktop (16 cores, eight
    simulated stars) four workers finished the batch 2.0x faster than one process and
    eight 2.5x. Capping each worker's XLA and BLAS threads at ``cpu_count // jobs`` made
    no measurable difference on that benchmark (``docs/benchmarks.md``). A worker
    returns a plain-data :class:`StarResult`. The live objects (the :class:`~albireo.Fit`,
    the velocity table, the label match) are kept only on an in-process run, since they
    hold compiled JAX programs that cannot be pickled.
    """
    if isinstance(config, str | os.PathLike):
        config = load_config(config)
    elif isinstance(config, Mapping):
        config = config_from_dict(config)
    selected = list(config.stars)
    if stars is not None:
        wanted = list(stars)
        unknown = sorted(set(wanted) - {s.name for s in selected})
        if unknown:
            raise KeyError(f"unknown star(s) {unknown}; the batch has {[s.name for s in selected]}")
        selected = [s for s in selected if s.name in wanted]
    directory = Path(config.output)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "albireo": _version(),
        "created": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "stars": [s.name for s in selected],
    }
    (directory / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_jobs = _resolve_jobs(jobs, len(selected))
    subdirs = _star_directories(directory, selected)
    t0 = time.perf_counter()
    results: dict[str, StarResult] = {}
    if progress:
        print(
            f"albireo pipeline: {len(selected)} star(s), {n_jobs} worker(s), output {directory}",
            flush=True,
        )
    if n_jobs == 1:
        for star in selected:
            results[star.name] = _run_star_guarded(
                star, config, subdirs[star.name], progress=progress, keep_live=True
            )
    else:
        env = _thread_environment(n_jobs)
        shared = config.without_stars()
        context = multiprocessing.get_context("spawn")
        with (
            _environment(env),
            ProcessPoolExecutor(
                max_workers=n_jobs,
                mp_context=context,
                initializer=_worker_init,
                initargs=(shared, env),
            ) as pool,
        ):
            futures = {
                pool.submit(_star_task, star, os.fspath(subdirs[star.name]), progress): star.name
                for star in selected
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    results[name] = StarResult(
                        name=name,
                        status="failed",
                        directory=os.fspath(subdirs[name]),
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                    )
        results = {star.name: results[star.name] for star in selected}
    run = PipelineRun(
        results=results, directory=directory, seconds=time.perf_counter() - t0, jobs=n_jobs
    )
    run.write()
    if progress:
        print(run.summary(), flush=True)
    return run


def _star_directories(directory: Path, stars: Sequence[StarConfig]) -> dict[str, Path]:
    taken: dict[str, str] = {}
    out = {}
    for star in stars:
        base = _safe_name(star.name)
        candidate, k = base, 1
        while candidate in taken.values():
            k += 1
            candidate = f"{base}_{k}"
        taken[star.name] = candidate
        out[star.name] = directory / candidate
    return out


# ---------------------------------------------------------------------------
# The demo
# ---------------------------------------------------------------------------


def demo_config(
    directory: str | os.PathLike = "albireo_demo", *, fast: bool = False, sample: bool = False
) -> PipelineConfig:
    """The batch that ``albireo demo`` runs: two simulated stars with known injected values.

    The first star is the packaged example (:func:`albireo.load_example`), disentangled
    and measured against its own components. Its velocities are differential, because
    its files declare no wavelength medium and no library is used for it. The second
    star's components are drawn from a toy synthetic library
    (:func:`albireo.simulate.synthetic_library`) at known labels, so the label stage
    recovers them and sets the zero point. Its systemic velocity of +12 km/s is not
    identifiable by the disentangling alone, and the orbit fitted to the absolute
    velocities recovers it. Both reports include an "against the injected truth" block.
    Nothing is downloaded.
    """
    from albireo.examples import load_example
    from albireo.simulate import (
        InstrumentSpec,
        OrbitParams,
        library_component,
        simulate_dataset,
        synthetic_library,
    )

    dataset, truth = load_example("sb2_sim", with_truth=True)
    grid = LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))
    packaged = StarConfig(
        name="sb2_sim",
        dataset=dataset,
        period=(5.5, 6.5),
        components=[
            ComponentConfig("primary", float(truth["light_fractions"][0])),
            ComponentConfig("secondary", float(truth["light_fractions"][1])),
        ],
        lsf={"DEMO": 6.5},
        labels=False,
        truth={
            "k": [float(v) for v in truth["k"]],
            "period": float(truth["period"]),
            "velocities": np.asarray(truth["velocities"]),
            "components": np.asarray(truth["components"]),
            "grid": grid,
        },
        overrides={"k_max": 90.0},
    )

    library = synthetic_library((5140.0, 5260.0))
    labels = {
        "A": {"teff": 5180.0, "logg": 4.05, "mh": -0.15, "vsini": 11.0},
        "B": {"teff": 4460.0, "logg": 4.55, "mh": -0.15, "vsini": 27.0},
    }
    toy_grid = LogGrid.from_wavelength_range(5150.0, 5250.0, dv_kms=1.5)
    components = [
        library_component(
            library,
            {k: v for k, v in values.items() if k != "vsini"},
            toy_grid,
            medium="air",
            vsini_kms=values["vsini"],
        )
        for values in labels.values()
    ]
    orbit = OrbitParams(period=6.31, t_peri=2.0, ecc=0.15, omega=0.7, k=(30.0, 55.0), gamma=12.0)
    rng = np.random.default_rng(2026)
    bjd = np.sort(rng.uniform(0.0, 21.0, size=12))
    toy_dataset, toy_truth = simulate_dataset(
        toy_grid,
        components,
        bjd=bjd,
        instruments={
            "TOY": InstrumentSpec(wave=np.arange(5156.0, 5244.0, 0.06), sigma_v_lsf=5.5, snr=150.0)
        },
        light_fractions=(0.62, 0.38),
        orbit=orbit,
        seed=7,
    )
    toy_dataset = _with_medium(toy_dataset, "air")
    toy = StarConfig(
        name="toy_library_sb2",
        dataset=toy_dataset,
        period=(6.0, 6.6),
        components=[
            ComponentConfig("A", 0.62, teff=(4200.0, 5700.0), logg=(3.2, 4.9), vsini=(1.0, 60.0)),
            ComponentConfig("B", 0.38, teff=(4100.0, 5200.0), logg=(3.2, 4.9), vsini=(1.0, 60.0)),
        ],
        lsf={"TOY": 5.5},
        truth={
            "k": [30.0, 55.0],
            "period": 6.31,
            "gamma": 12.0,
            "velocities": np.asarray(toy_truth.velocities),
            "components": np.asarray(toy_truth.components),
            "grid": toy_grid,
            "labels": labels,
        },
        overrides={"k_max": 90.0},
    )
    return PipelineConfig(
        stars=[packaged, toy],
        output=directory,
        library=library,
        mh=(-0.9, 0.4),
        analysis=Analysis(sample=sample, fast=fast, v_zero_range=60.0),
    )
