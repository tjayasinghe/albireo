"""Run every stage of the analysis for a list of stars, from one declaration.

For each star the driver reads the epochs (:mod:`albireo.io`), disentangles them
(:class:`albireo.Disentangler`), fits atmospheric labels to the components so that each
can serve as a radial-velocity template (:mod:`albireo.match`), measures one velocity per
component per epoch by TODCOR against those templates (:mod:`albireo.todcor`), fits a
Keplerian to the resulting table (:mod:`albireo.rvorbit`), and writes the products with
figures. A stage that cannot be run (a label fit on data whose wavelength medium is
undeclared, an orbit from too few usable epochs) is recorded as a flag on the star's
report, and a failure in one star does not stop the batch. ``albireo run config.toml`` is
the command line.

Two rules of the underlying stages are enforced unchanged. Light fractions must be
declared and have no default: with constant light the likelihood constrains only
``l_i * d_i``, so the data cannot detect a wrong value (``docs/math.md`` §5.2). The
wavelength medium must be declared before a synthetic grid is consulted, because air and
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
)
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


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def _spec(value: Any, what: str) -> Spec | None:
    """Coerce a config value into the façade's prior vocabulary.

    ``None`` stays ``None`` (the stage's own default applies); a number is ``Fixed``; a
    two-element list is ``Between``; a mapping with ``value`` and ``sigma`` is ``Known``.
    A :class:`~albireo.facade.Spec` passes through, so the Python API accepts the façade's
    own specs.
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
    raise TypeError(f"lsf[{key!r}]: expected a sigma in km/s, a table, or an LSF")


@dataclass(frozen=True)
class ComponentConfig:
    """One stellar component of a star, as declared to the pipeline.

    Parameters
    ----------
    name
        The component's name, used wherever a row index would otherwise be.
    light
        Its light fraction. Required, with no default, because it is an assumption: the
        light fractions of a star must sum to one, and no part of the fit can detect a
        wrong value (``docs/math.md`` §5.2). It should be quoted beside every result
        derived from the spectra.
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
        light = float(self.light)
        if not (math.isfinite(light) and light > 0.0):
            raise ValueError(f"component {self.name!r}: light must be finite and positive")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "light", light)
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
        generous value costs time, not correctness.
    ecc_max
        Upper bound of the eccentricity prior; ``circular`` holds ``e = 0`` exactly.
    max_steps
        L-BFGS cap for the disentangling.
    dv_kms
        Model-grid pixel size in km/s; the default is the finest sampling in the data.
    v_range
        Search half-range in km/s for the library-template velocity table of the
        ``period = "search"`` route.
    vsini_max, v_zero_range
        Default ceiling of the ``vsini`` prior, and the half-range of the per-component
        frame offset the label fit may measure, both in km/s. The disentangled frame sits
        at the systemic velocity, so the range must cover it; 300 km/s reaches the
        Magellanic Clouds.
    label_steps
        L-BFGS cap for the label fit.
    dilution
        ``"radius_ratio"`` (joint, the default), ``"scalar"`` or ``"fixed"``.
    sample, num_warmup, num_samples, num_chains
        Whether to run NUTS after the MAP, and how much.
    telluric, nebular, nebular_v_kms
        Extra components, as the façade declares them.
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
    dv_kms: float | None = None
    v_range: float = 300.0
    vsini_max: float = 300.0
    v_zero_range: float = 300.0
    label_steps: int = 500
    dilution: str = "radius_ratio"
    sample: bool = False
    num_warmup: int = 300
    num_samples: int = 300
    num_chains: int = 2
    telluric: bool = False
    nebular: bool = False
    nebular_v_kms: float = 0.0
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
        if self.max_steps < 1 or self.label_steps < 1:
            raise ValueError("max_steps and label_steps must be positive")

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
        For simulated stars only: injected values to compare against
        (``k``, ``period``, ``ecc``, ``gamma``, ``velocities``, ``labels``, and
        ``components`` on ``grid``).
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
        total = sum(c.light for c in components)
        if abs(total - 1.0) > 1e-6:
            listed = ", ".join(f"{c.name}={c.light:g}" for c in components)
            raise ValueError(
                f"star {name!r}: the light fractions must sum to 1; {listed} sums to "
                f"{total:g}. This is an assumption the data cannot check, which is why it "
                "has no default."
            )
        object.__setattr__(self, "components", components)
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
        ``{"sigma_kms": ...}`` or an :class:`~albireo.LSF`. An instrument with no entry
        anywhere takes ``R`` from its own FITS header when the header carries one.
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
# entry takes its resolving power from its own header (SPEC_RES) when there is one.
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
# sample = true                 # NUTS after the MAP: posterior widths, slow

# Template identification: fit Teff, log g, [M/H] and v sin i to the disentangled
# components against a published grid, and use the fitted frame offset to make the epoch
# velocities absolute. Delete this table to skip the stage (velocities stay differential).
[labels]
library = "bosz2024-fgk-r20000"   # albireo.library_names(); downloads ~645 MB once
mh = [-1.0, 0.5]                  # shared metallicity range, or a number to hold it
v_zero_range = 300.0              # how far the disentangled frame may sit from rest, km/s

[[stars]]
name = "AI Phe"
spectra = "data/aiphe/*.fits"   # a glob, a directory, or a list of files
period = [24.5, 24.7]           # days: [lo, hi] uniform, a value to hold, or "search"
# velocities = "aiphe_rv.txt"   # instead of period: measured per-epoch velocities

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

    The single-star entry point of the Python API. Raises on a failure, unlike the batch
    driver, which records it; use :func:`run_pipeline` for many stars.

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
    if ctx.config.library is not None and (star.labels or star.searching):
        with ctx.stage("library"):
            library = _resolve_library(ctx.config.library, log)

    # 3. the orbit declaration, bootstrapped if asked
    bootstrap = None
    declared_velocities = None
    if star.velocities is not None:
        declared_velocities = _read_velocities(star, dataset)
        log("free per-epoch table declared from measured velocities")
    elif star.searching:
        with ctx.stage("bootstrap"):
            orbit_spec, bootstrap = _bootstrap(ctx, dataset, lsf, library)
        sections.append(bootstrap["text"])
        report["bootstrap"] = bootstrap["report"]
        live["bootstrap"] = bootstrap["objects"]
    else:
        orbit_spec = _declared_orbit(star, settings)

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
            f"{dis.model.half_bandwidth}, {settings.max_steps} steps"
        )
        fit = dis.fit(max_steps=settings.max_steps)
    live["fit"] = fit
    sections.append(dis.explain())
    sections.append(fit.summary())
    report["declaration"] = _describe_declaration(dis)
    report["disentangling"] = _describe_fit(fit)
    _assess_fit(ctx, fit)
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
        table = fit.measure_velocities(templates=templates, light=[s.light for s in dis.stars])
    live["templates"] = templates
    live["velocities"] = table
    sections.append(table.summary())
    report["velocities"] = _describe_table(table)
    _assess_table(ctx, table)
    log(
        f"velocities: {int(table.good.sum())}/{table.n_epochs} usable epochs, median sigma "
        + ", ".join(f"{n} {np.nanmedian(table.sigma[i]):.3f}" for i, n in enumerate(table.names))
        + " km/s, "
        + ("absolute" if all(table.absolute) else "differential")
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
        for raw in raws:
            sigma = raw.lsf_sigma_kms
            if sigma is not None and raw.instrument not in header_lsf:
                header_lsf[raw.instrument] = float(sigma)
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
        elif key in header_lsf:
            out[key] = LSF(sigma_kms=header_lsf[key])
            ctx.log(
                f"instrument {key!r}: LSF sigma {header_lsf[key]:.3f} km/s taken from the "
                "files' own SPEC_RES header"
            )
        else:
            raise ValueError(
                f"no line-spread function for instrument {key!r} and its files declare no "
                f"resolving power. Add [instrument.{key}] with resolving_power or sigma_kms "
                "to the configuration (or lsf={...} on the star)."
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


def _declared_orbit(star: StarConfig, settings: Analysis) -> Orbit:
    period = _spec(star.period, "period")
    if isinstance(period, Fixed) and not float(np.asarray(period.value)) > 0.0:
        raise ValueError(f"star {star.name!r}: the period must be positive")
    return Orbit(
        period=period,
        k=_k_prior(star, settings),
        ecc=Fixed(0.0) if settings.circular else Between(0.0, settings.ecc_max),
    )


def _k_prior(star: StarConfig, settings: Analysis) -> list[Spec]:
    """One semi-amplitude prior per component, started in the declared order.

    A symmetric prior cannot assign the spectra to the stars: with every component
    started at the same semi-amplitude the conjunction scan sees two equally deep minima
    (the declared assignment and its mirror, with the spectra swapped and rescaled by the
    light ratio), and L-BFGS converges to whichever it started in. The data cannot break
    the tie, since only ``l_i * d_i`` is observable, so a convention does: components are
    declared in order of decreasing mass, the first star moves least, and the fit is
    started with ``K_1 < K_2 < ...`` at evenly spaced points of the shared range, which
    the scan then discriminates on. The ordering is a starting point, not a constraint
    (the bounds are the same for every component), and the label stage checks the
    outcome: a fitted light fraction far from the declared one is the signature of a
    reversed order.
    """
    n = len(star.components)
    declared = [_spec(c.k, f"component {c.name!r}: k") for c in star.components]
    out: list[Spec] = []
    for i, spec in enumerate(declared):
        if spec is not None:
            out.append(spec)
            continue
        start = settings.k_min + (settings.k_max - settings.k_min) * (i + 1) / (n + 1)
        out.append(Between(settings.k_min, settings.k_max, start_at=start))
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
    )


def _read_velocities(star: StarConfig, dataset: Dataset) -> np.ndarray:
    """A declared velocity table, ``(n_stellar, n_epochs)``, matched to the epochs."""
    n = len(star.components)
    if isinstance(star.velocities, str | os.PathLike):
        table = np.loadtxt(star.velocities, ndmin=2, comments="#")
    else:
        table = np.asarray(star.velocities, dtype=float)
        if table.shape == (n, dataset.n_epochs):
            return table
        if table.shape == (dataset.n_epochs, n):
            return table.T
    if table.ndim != 2:
        raise ValueError(f"star {star.name!r}: the velocity table must be two-dimensional")
    if table.shape[1] == n:
        if table.shape[0] != dataset.n_epochs:
            raise ValueError(
                f"star {star.name!r}: {table.shape[0]} velocity rows for "
                f"{dataset.n_epochs} epochs; add a leading BJD column to match by time"
            )
        return table.T
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
        return out
    raise ValueError(
        f"star {star.name!r}: the velocity table has {table.shape[1]} columns; expected one "
        f"per component ({n}) or BJD plus one per component ({n + 1})"
    )


def _bootstrap(ctx: _Context, dataset: Dataset, lsf, library):
    """Bootstrap the orbit from library templates for the ``period = "search"`` route.

    Templates at the declared starting labels measure a velocity table, a period search
    proposes candidates, an orbit is fitted from each, and the best orbit becomes a warm
    Keplerian prior for the disentangling.
    """
    from albireo.rvorbit import find_period
    from albireo.todcor import Template, todcor

    star, settings, log = ctx.star, ctx.settings, ctx.log
    medium = dataset[0].medium
    if medium is None:
        raise ValueError(
            f"star {star.name!r}: period = 'search' renders library templates, which needs "
            "the wavelength medium; the files did not declare one, so set medium = 'air' "
            "(or 'vacuum') on the star once you have checked which it is"
        )
    sigmas = [
        float(np.min(np.atleast_1d(np.asarray(v.sigma_kms, dtype=float)))) for v in lsf.values()
    ]
    narrowest, widest = (
        min(sigmas),
        max(
            float(np.max(np.atleast_1d(np.asarray(v.sigma_kms, dtype=float)))) for v in lsf.values()
        ),
    )
    grid = LogGrid.covering(
        dataset,
        max(narrowest / 3.0, 0.2),
        v_margin_kms=settings.v_range + 60.0,
        lsf_sigma_kms=widest,
    )
    lib = library.sliced(
        grid.wave[0] - _LIBRARY_PAD_ANGSTROM, grid.wave[-1] + _LIBRARY_PAD_ANGSTROM
    )
    resolving = lib.meta.get("resolution")
    templates = []
    for c in star.components:
        start = c.labels_start()
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
    table = todcor(
        dataset,
        templates,
        v_range=(-settings.v_range, settings.v_range),
        light="global",
        lsf_sigma_v=lsf_sigma,
        lsf_anchors_angstrom=anchors or None,
    )
    log(
        f"bootstrap velocities: {int(table.good.sum())}/{table.n_epochs} usable epochs "
        f"(library templates, absolute)"
    )
    search = find_period(table)
    orbit = _orbit_over_candidates(
        ctx, table, [search["period"], *search["aliases"]], circular=settings.circular
    )
    log(
        f"bootstrap orbit: P {orbit.period:.5f} d (periodogram peak {search['period']:.4f}, "
        f"aliases {[round(p, 3) for p in search['aliases'][:3]]}; the orbit fit decided), K "
        f"{np.round(orbit.k, 2).tolist()} km/s"
    )
    period_width = 0.03 * orbit.period
    k_sigma = np.maximum(3.0 * np.asarray(orbit.errors["k"]), 0.15 * np.asarray(orbit.k))
    k_sigma = np.where(np.isfinite(k_sigma), k_sigma, 0.3 * np.asarray(orbit.k))
    spec = Orbit(
        period=Between(orbit.period - period_width, orbit.period + period_width),
        k=Known(np.asarray(orbit.k, dtype=float), np.asarray(k_sigma, dtype=float)),
        t_conj=Known(float(orbit.t_conj), 0.05 * orbit.period),
        ecc=Fixed(0.0) if settings.circular else Between(0.0, settings.ecc_max),
    )
    text = (
        "Bootstrap from library templates (period = 'search'):\n"
        f"  templates at the declared starting labels; period search "
        f"{search['period']:.5f} d, aliases {[round(p, 4) for p in search['aliases'][:4]]}\n"
        + table.summary()
        + "\n"
        + orbit.summary()
        + "\n  -> disentangling prior: period within +-3%, K Gaussian with sigma "
        + f"{np.round(k_sigma, 2).tolist()} km/s, t_conj Gaussian"
    )
    return spec, {
        "text": text,
        "report": {
            "period": float(orbit.period),
            "periodogram_peak": float(search["period"]),
            "aliases": [float(p) for p in search["aliases"]],
            "k": {n: float(k) for n, k in zip(orbit.names, orbit.k, strict=True)},
            "t_conj": float(orbit.t_conj),
            "ecc": float(orbit.ecc),
            "n_usable": int(table.good.sum()),
            "templates": [t.meta for t in templates],
        },
        "objects": {"table": table, "orbit": orbit, "templates": templates},
    }


def _orbit_over_candidates(ctx: _Context, table, periods, *, circular: bool):
    """Fit an orbit from every candidate period and keep the lowest chi-square.

    The periodogram of a sparsely sampled table is rarely unambiguous: on the ten-epoch
    test fixture the highest peak was a 2.25 d alias whose orbit fits at chi-square 73,
    against 16 at the true period, to which every other peak converged. The periodogram
    peaks are therefore starting points, and the orbit fit decides. A runner-up at a
    different period (more than 2% from the best) within a chi-square difference of 9 is
    flagged as an ambiguity.
    """
    from albireo.rvorbit import fit_rv_orbit

    fitted = []
    for period in periods:
        try:
            orbit = fit_rv_orbit(table, period=float(period), circular=circular)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        fitted.append(orbit)
    if not fitted:
        raise ValueError("no candidate period gave an orbit fit")
    fitted.sort(key=lambda o: o.chi2)
    best = fitted[0]
    for other in fitted[1:]:
        if abs(other.period / best.period - 1.0) > 0.02:
            if other.chi2 - best.chi2 < 9.0:
                ctx.flag(
                    f"period ambiguous: an orbit at {other.period:.4f} d fits within "
                    f"delta chi2 {other.chi2 - best.chi2:.1f} of the chosen {best.period:.4f} d"
                )
            break
    return best


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
        lib = library.sliced(
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
    reach = float(settings.v_zero_range)
    narrowest = min(
        float(np.min(np.atleast_1d(np.asarray(v.sigma_kms, dtype=float))))
        for v in fit.dis.lsf.values()
        if isinstance(v, LSF)
    )
    step = max(2.0 * narrowest, 5.0, 2.0 * reach / 120.0)
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
        "max_steps": settings.label_steps,
        "scan_velocities": scan_velocities,
    }
    if ctx.config.mh is not None:
        options["mh"] = _spec(ctx.config.mh, "mh")
    log(
        f"label fit against {lib.n_nodes} nodes, {len(scan_velocities)} trial frame offsets "
        f"over +-{reach:g} km/s, {settings.label_steps} steps"
    )
    try:
        match = fit.match_labels(stars, **options)
    except (ValueError, RuntimeError) as exc:
        ctx.flag(f"labels failed: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        return None
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
    declared_light = {c.name: c.light for c in star.components}
    for name, fitted in match.flux_ratio.items():
        if abs(fitted - declared_light[name]) > 0.15:
            ctx.flag(
                f"the label fit measures a light fraction of {fitted:.2f} for {name!r} "
                f"against the declared {declared_light[name]:.2f}: either the declared "
                "light fractions are wrong, or the components were declared in the wrong "
                "order (the pipeline assumes decreasing mass: the first star moves least)"
            )
            break
    if match.multimodal:
        ctx.flag("labels: the scan found a second basin within delta chi2 < 9")
    if not match.chi2 < match.chi2_nearest_node < match.chi2_continuum:
        ctx.flag(
            "labels: the fit does not beat both nulls (chi2 "
            f"{match.chi2:.1f}, nearest node {match.chi2_nearest_node:.1f}, no template "
            f"{match.chi2_continuum:.1f}); treat the labels and the zero points as unmeasured"
        )
    return match


def _templates(ctx: _Context, fit: Fit, match) -> list:
    templates = fit.templates()
    if match is None:
        return templates
    pinned = []
    for t in templates:
        offset = float(match.labels[t.name]["v_kms"])
        pinned.append(
            replace(
                t,
                v_zero_kms=offset,
                meta={**t.meta, "zero_point": "label match", "v_zero_kms": offset},
            )
        )
    ctx.log(
        "template zero points from the label fit: "
        + ", ".join(f"{t.name} {t.v_zero_kms:+.2f} km/s" for t in pinned)
    )
    return pinned


def _orbit(ctx: _Context, fit: Fit, table):
    from albireo.rvorbit import find_period, fit_rv_orbit

    settings = ctx.settings
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
            search = find_period(table)
            source = (
                f"a periodogram (peak {search['period']:.4f} d, aliases "
                f"{[round(p, 4) for p in search['aliases'][:3]]}; the orbit fit decided)"
            )
            orbit = _orbit_over_candidates(
                ctx, table, [search["period"], *search["aliases"]], circular=settings.circular
            )
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
    if match is not None:
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


def _assess_table(ctx: _Context, table) -> None:
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


def _describe_dataset(dataset: Dataset, lsf) -> dict[str, Any]:
    n_pixels = sum(e.n_pixels for e in dataset)
    n_good = sum(int(e.good.sum()) for e in dataset)
    return {
        "n_epochs": dataset.n_epochs,
        "frame": dataset.frame,
        "medium": dataset[0].medium,
        "instruments": list(dataset.instruments),
        "lsf_sigma_kms": {k: np.asarray(v.sigma_kms).tolist() for k, v in lsf.items()},
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
    return out


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
    return {
        "components": components,
        "flux_ratio": dict(match.flux_ratio),
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


def _describe_table(table) -> dict[str, Any]:
    return {
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
        "wilson_slope": None if table.wilson() is None else float(table.wilson()[0]),
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


def _compare_truth(ctx: _Context, fit: Fit, table, orbit, match) -> tuple[str, dict[str, Any]]:
    truth = dict(ctx.star.truth or {})
    names = [s.name for s in fit.dis.stars]
    lines = ["Against the injected truth:"]
    out: dict[str, Any] = {"note": "differences (result minus injected), except velocity_rms"}
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
            lines.append(
                "  K from the velocity table: "
                + ", ".join(
                    f"{n} {orbit.k[i]:.3f}+-{orbit.errors['k'][i]:.3f} "
                    f"(truth {k_true[i]:g}, {orbit.k[i] - k_true[i]:+.3f})"
                    for i, n in enumerate(names)
                )
            )
    if "period" in truth and orbit is not None:
        out["period"] = float(orbit.period - float(truth["period"]))
        lines.append(
            f"  period from the table {orbit.period:.5f} d (truth {float(truth['period']):g}, "
            f"{out['period']:+.5f})"
        )
    if "gamma" in truth and orbit is not None:
        gamma_true = float(truth["gamma"])
        out["gamma"] = {n: float(orbit.gamma[i] - gamma_true) for i, n in enumerate(names)}
        note = "" if all(table.absolute) else "   (differential: not comparable)"
        lines.append(
            "  systemic velocity: "
            + ", ".join(f"{n} {orbit.gamma[i]:+.3f}" for i, n in enumerate(names))
            + f" (truth {gamma_true:+g}){note}"
        )
    if "velocities" in truth:
        v_true = np.asarray(truth["velocities"], dtype=float)
        if v_true.shape == table.velocity.shape:
            rows = {}
            for i, n in enumerate(names):
                diff = table.velocity[i] - v_true[i]
                if not table.absolute[i]:
                    diff = diff - np.nanmean(diff)
                rows[n] = float(np.sqrt(np.nanmean(diff**2)))
            out["velocity_rms"] = rows
            lines.append(
                "  epoch velocities rms error: "
                + ", ".join(f"{n} {r:.3f} km/s" for n, r in rows.items())
                + ("" if all(table.absolute) else " (after removing each zero point)")
            )
    if "labels" in truth and match is not None:
        rows = {}
        for n, labels in truth["labels"].items():
            if n not in match.labels:
                continue
            got = match.labels[n]
            rows[n] = {k: float(got[k] - v) for k, v in labels.items() if k in got}
            lines.append(
                f"  labels {n}: "
                + ", ".join(
                    f"{k} {got[k]:.3g} (truth {v:g}, {got[k] - v:+.3g})"
                    for k, v in labels.items()
                    if k in got
                )
            )
        out["labels"] = rows
    return "\n".join(lines), out


# -- products --------------------------------------------------------------------


def _files_block(files: Mapping[str, str]) -> str:
    return "Files:\n" + "\n".join(f"  {k:<14s} {v}" for k, v in sorted(files.items()))


def _write_products(ctx: _Context, fit: Fit, table, orbit, match, posterior) -> None:
    from albireo.results import save_fit, write_ascii

    directory, files = ctx.directory, ctx.files
    header = f"star: {ctx.star.name}"
    files["velocities"] = os.fspath(table.write(directory / "velocities.rv", header=header))
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


def _write_velocity_csv(path: Path, table) -> Path:
    columns = table.to_dict()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(columns))
        for j in range(table.n_epochs):
            row = []
            for key, col in columns.items():
                value = col[j]
                if key == "instrument":
                    row.append(str(value))
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
        span = float(np.nanmax(np.abs(table.velocity))) + 40.0
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
    that oversubscription rather than a measured gain: on the recorded benchmark eight
    capped workers and eight uncapped ones finished the same batch in 54.1 and 54.7 s
    (``docs/benchmarks.md``, D58). It is kept because it has no measured cost there and
    because oversubscription has been observed in BLAS-heavy stages (the 32-thread
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
        on each :class:`StarResult`; ``"auto"`` or ``0`` uses ``cpu_count // 4``; any
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
    must therefore do so from under ``if __name__ == "__main__":``: the workers import
    the script as a module, and an unguarded call would start the batch again in each of
    them. The ``albireo`` command is guarded.

    The scaling is sub-linear because a single star already occupies several cores, so
    the workers overlap only the serial part of each star (compilation, the Python-side
    scans, the orbit fit, the writing). On the development desktop (16 cores, eight
    simulated stars) four workers finished the batch 2.0x faster than one process and
    eight 2.5x; capping each worker's XLA and BLAS threads at ``cpu_count // jobs`` made
    no measurable difference on that benchmark (``docs/benchmarks.md``). A worker
    returns a plain-data :class:`StarResult`; the live objects (the :class:`~albireo.Fit`,
    the velocity table, the label match) are kept only on an in-process run, since they
    carry compiled JAX programs that cannot be pickled.
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
    """The batch that ``albireo demo`` runs: two simulated stars with known answers.

    The first star is the packaged example (:func:`albireo.load_example`), disentangled
    and measured against its own components. Its velocities are differential, because
    its files declare no wavelength medium and no library is consulted for it. The second
    is a star whose components are drawn from a toy synthetic library
    (:func:`albireo.simulate.synthetic_library`) at known labels, so the label stage
    recovers them and pins the zero point: its systemic velocity of +12 km/s is not
    identifiable by the disentangling alone, and the orbit fitted to the absolute
    velocities recovers it. Both reports carry an "against the injected truth" block.
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
