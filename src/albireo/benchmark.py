"""Simulate, process and verify populations of Gaia RVS double-lined binaries; report.

**Experimental.** The tiers, the metrics and the report layout may change.

The harness answers one question with numbers: how well does albireo recover the orbit,
the epoch velocities, the component spectra and the atmospheric labels of a double-lined
binary from Gaia RVS epoch spectra, as a function of what the system is (brightness,
transits, separation, light ratio, temperatures) and of what the analysis is told. Every
system of a population (:mod:`albireo.population`) is simulated once
(:mod:`albireo.gaia`), the same epochs are run through the pipeline
(:mod:`albireo.pipeline`) under one or more knowledge tiers, the pipeline's own truth
block is collected, and a Markdown report with figures is written. Nothing outside
albireo is used at any stage.

The knowledge tiers state what the analysis is told; the truth is never told:

- ``oracle``: period, conjunction, eccentricity, argument of periastron and the
  semi-amplitudes known (the latter to 15%), light fractions and narrow label priors
  from the truth. The ceiling: it measures the disentangling and the velocity
  extraction alone.
- ``eclipsing``: the ephemeris (period and time of primary eclipse) and the light ratio
  known, as a light curve gives them, with narrow label priors; elements free.
- ``orbit``: the period known to a Gaia non-single-star solution's precision, everything
  else free, light fractions measured by correlation against library templates, labels
  from the library box. This is a Gaia double-lined orbit as published.
- ``blind``: nothing known: the period is searched, the light fractions are measured, the
  labels are the library box.

The tiers are declared to the pipeline through its own vocabulary, so the benchmark
runs exactly what a user runs.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from albireo.data import Dataset
from albireo.gaia import (
    RVS_DR3_MEAN,
    RVS_DR4_EPOCH,
    RVSProduct,
    RVSTruth,
    gost_transits,
    rvs_components,
    rvs_lsf_sigma_kms,
    rvs_model_grid,
    rvs_snr_per_pixel,
    rvs_transit_resolving_powers,
    rvs_transit_times,
    rvs_transit_times_from_gost,
    simulate_rvs_dataset,
    uniform_phase_times,
)
from albireo.grids import LogGrid
from albireo.kepler import radial_velocity
from albireo.pipeline import Analysis, ComponentConfig, PipelineConfig, StarConfig, run_pipeline
from albireo.population import BinarySystem, population_summary, read_population, write_population

__all__ = [
    "CADENCES",
    "CA_TRIPLET_WINDOWS",
    "DEFAULT_WINDOWS",
    "TIERS",
    "BenchmarkConfig",
    "BenchmarkRun",
    "Tier",
    "build_star",
    "build_stars",
    "collect",
    "collect_velocities",
    "resolve_tier",
    "run_benchmark",
    "simulate_system",
    "summarize",
    "system_transit_times",
    "write_report",
]

CA_TRIPLET_WINDOWS: tuple[tuple[float, float], ...] = (
    (8490.0, 8506.0),
    (8534.0, 8552.0),
    (8654.0, 8672.0),
)
"""The three Ca II triplet lines, 16-18 Angstrom wide each, in vacuum Angstrom."""

DEFAULT_WINDOWS: dict[str, tuple[tuple[float, float], ...]] = {
    "ca_triplet": CA_TRIPLET_WINDOWS,
    "metal": ((8460.0, 8490.0), (8506.0, 8534.0), (8552.0, 8654.0), (8672.0, 8700.0)),
}
"""The windows the component spectra are compared over: the triplet, and the rest."""

PRODUCTS: dict[str, RVSProduct] = {"dr3": RVS_DR3_MEAN, "dr4": RVS_DR4_EPOCH}

CADENCES: tuple[str, ...] = ("scanning-law", "uniform-phase", "gost")
"""The epoch-time models: the scanning law's structure without its phase, evenly spaced
phases over one period, and the real forecast of the Gaia Observation Forecast Tool."""


@dataclasses.dataclass(frozen=True)
class Tier:
    """What the analysis is told about a system.

    Parameters
    ----------
    name
        The tier's name; it suffixes every star name.
    period
        ``"known"`` (a Gaussian of relative width ``period_sigma``), ``"range"`` (a
        uniform prior of relative half-width ``period_sigma``) or ``"search"`` (the
        pipeline's period search from library templates).
    period_sigma
        Relative width of the period prior.
    t_conj
        ``"known"`` (a Gaussian of width 1% of the period about the true primary
        conjunction, as an eclipse ephemeris gives it) or ``"scan"``.
    elements
        ``"known"``: eccentricity and argument of periastron held at the truth, the
        semi-amplitudes given as Gaussians of 15%; ``"free"``: fitted within the settings'
        ranges.
    light
        ``"truth"``: the injected fractions declared; ``"measure"``: measured by the
        pipeline against library templates.
    labels
        ``"narrow"``: Teff within 300 K and log g within 0.3 dex of the truth (what a
        photometric classification gives); ``"wide"``: the library box.
    eclipsing_only
        Whether the tier applies to eclipsing systems only.
    sample
        Run NUTS after the MAP fit.
    """

    name: str
    period: str = "known"
    period_sigma: float = 1e-4
    t_conj: str = "scan"
    elements: str = "free"
    light: str = "truth"
    labels: str = "wide"
    eclipsing_only: bool = False
    sample: bool = False

    def __post_init__(self) -> None:
        if self.period not in ("known", "range", "search"):
            raise ValueError(f"tier {self.name!r}: period must be known, range or search")
        if self.t_conj not in ("known", "scan"):
            raise ValueError(f"tier {self.name!r}: t_conj must be known or scan")
        if self.elements not in ("known", "free"):
            raise ValueError(f"tier {self.name!r}: elements must be known or free")
        if self.light not in ("truth", "measure"):
            raise ValueError(f"tier {self.name!r}: light must be truth or measure")
        if self.labels not in ("narrow", "wide"):
            raise ValueError(f"tier {self.name!r}: labels must be narrow or wide")
        if self.period == "search" and self.t_conj == "known":
            raise ValueError(f"tier {self.name!r}: a searched period has no known conjunction")


TIERS: dict[str, Tier] = {
    "oracle": Tier("oracle", period="known", t_conj="known", elements="known", labels="narrow"),
    "eclipsing": Tier(
        "eclipsing",
        period="known",
        t_conj="known",
        elements="free",
        light="truth",
        labels="narrow",
        eclipsing_only=True,
    ),
    "orbit": Tier("orbit", period="known", t_conj="scan", elements="free", light="measure"),
    "blind": Tier("blind", period="search", t_conj="scan", elements="free", light="measure"),
}
"""The four standard tiers, from the ceiling to the blind run."""


def resolve_tier(tier: str | Tier) -> Tier:
    """A :class:`Tier` from its name or itself."""
    if isinstance(tier, Tier):
        return tier
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; the standard tiers are {sorted(TIERS)}")
    return TIERS[tier]


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    """A benchmark: the population, the tiers, the instrument, and the analysis settings.

    Parameters
    ----------
    output
        Directory for the products; the pipeline writes one sub-directory per star.
    systems
        The population (:func:`albireo.population.draw_population` or an adapter).
    tiers
        Tier names or :class:`Tier` objects; an ``eclipsing_only`` tier is skipped for
        systems that do not eclipse.
    product
        ``"dr4"`` (default), ``"dr3"``, or an :class:`~albireo.gaia.RVSProduct`.
    library
        The library both the simulation and the analysis use: a registry name or a
        :class:`~albireo.library.SpectralLibrary` covering the RVS band.
    cadence
        ``"scanning-law"`` (default, :func:`albireo.gaia.rvs_transit_times`),
        ``"uniform-phase"`` (the notebook's evenly spaced epochs over one period), or
        ``"gost"`` (the transits the Gaia Observation Forecast Tool predicts for the
        system's own position, cut to the release span, to the RVS CCD rows and to the
        fraction that reaches the ground; the system must carry ``ra_deg`` and
        ``dec_deg``, and the number of epochs is then the service's, not the record's
        ``n_transits``).
    resolving_power
        ``"nominal"`` observes every transit at R = 11,500; ``"per-transit"`` draws each
        transit's resolving power from the in-flight measurements while the analysis
        still declares the nominal one.
    dv_kms
        Model-grid pixel of the disentangling in km/s.
    k_min, k_max, ecc_max
        The semi-amplitude and eccentricity priors of the free tiers, km/s.
    max_steps, label_steps
        Optimizer budgets.
    mask_ca
        Zero-weight the Ca II triplet windows in the disentangling.
    noise_model
        ``"correlated"`` (default) declares the delivered grid's lag-one noise correlation
        to the analysis, as the simulation measured it, so that the disentangling's noise
        model is AR(1) along the pixel index and the velocity table's errors carry the
        correlation; ``"diagonal"`` takes the pixels as independent, which is what the
        archive's errors say.
    min_transits
        Systems with fewer transits are left out (Gaia's own double-lined chain needs
        ten).
    windows
        Wavelength windows the component spectra are compared over.
    jobs
        Worker processes for the pipeline (``"auto"`` for one per core).
    fast
        Trim every optimizer budget; a smoke run.
    plots
        Write the pipeline's per-star figures.
    resume
        Skip stars that already have a result in ``output``.
    seed
        Seed for the simulation noise.
    analysis
        Extra :class:`~albireo.pipeline.Analysis` overrides.
    title
        Title of the report.
    """

    output: str | os.PathLike
    systems: Sequence[BinarySystem]
    tiers: Sequence[str | Tier] = ("oracle", "eclipsing", "orbit", "blind")
    product: str | RVSProduct = "dr4"
    library: Any = "bosz2024-fgk-rvs"
    cadence: str = "scanning-law"
    resolving_power: str = "nominal"
    dv_kms: float = 3.0
    k_min: float = 2.0
    k_max: float = 250.0
    ecc_max: float = 0.9
    max_steps: int = 300
    label_steps: int = 300
    mask_ca: bool = False
    noise_model: str = "correlated"
    min_transits: int = 10
    windows: Mapping[str, Sequence[tuple[float, float]]] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_WINDOWS)
    )
    jobs: int | str = 1
    fast: bool = False
    plots: bool = True
    resume: bool = True
    seed: int = 0
    analysis: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    title: str = "albireo on Gaia RVS double-lined binaries"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tiers", tuple(resolve_tier(t) for t in self.tiers))
        if isinstance(self.product, str):
            if self.product not in PRODUCTS:
                raise ValueError(f"product must be one of {sorted(PRODUCTS)}; got {self.product!r}")
            object.__setattr__(self, "product", PRODUCTS[self.product])
        if self.cadence not in CADENCES:
            raise ValueError(f"cadence must be one of {', '.join(repr(c) for c in CADENCES)}")
        if self.resolving_power not in ("nominal", "per-transit"):
            raise ValueError("resolving_power must be 'nominal' or 'per-transit'")
        if self.noise_model not in ("correlated", "diagonal"):
            raise ValueError("noise_model must be 'correlated' or 'diagonal'")
        if not self.systems:
            raise ValueError("a benchmark needs at least one system")
        names = [s.name for s in self.systems]
        if len(set(names)) != len(names):
            raise ValueError("system names must be unique")
        object.__setattr__(self, "windows", {k: tuple(v) for k, v in self.windows.items()})
        object.__setattr__(self, "analysis", dict(self.analysis))

    @property
    def product_name(self) -> str:
        return self.product.name

    @property
    def release(self) -> str:
        """The release the product belongs to, which is the span a GOST cadence cuts to.

        ``"dr3"`` for the mean-spectrum product, whose data span is 34 months, and
        ``"dr4"`` for the epoch product and for any other, whose span is 66
        (:data:`albireo.gaia.RVS_SPANS`).
        """
        return "dr3" if str(self.product.name).startswith("dr3") else "dr4"


# ---------------------------------------------------------------------------
# Simulation and declaration
# ---------------------------------------------------------------------------


def star_name(system: BinarySystem, tier: Tier) -> str:
    """The pipeline name of a system under a tier."""
    return f"{system.name}__{tier.name}"


def _resolve_library(library):
    if isinstance(library, str):
        from albireo.library import fetch_library

        return fetch_library(library, progress=False)
    return library


def system_transit_times(
    system: BinarySystem, *, cadence: str = "scanning-law", seed: int = 0, release: str = "dr4"
) -> np.ndarray:
    """The epoch times of one system under a cadence model, sorted BJD.

    ``"scanning-law"`` and ``"uniform-phase"`` place exactly the system's own
    ``n_transits`` epochs. ``"gost"`` asks the Gaia Observation Forecast Tool what it
    predicts for the system's position (:func:`albireo.gaia.gost_transits`, cached under
    ``cache_dir()``) and cuts the answer to the release span, to the CCD rows the RVS
    covers and to the fraction that reaches the ground
    (:func:`albireo.gaia.rvs_transit_times_from_gost`), so the number of epochs is the
    service's rather than the record's.

    Raises
    ------
    ValueError
        Under ``"gost"``, if the system carries no position.
    """
    if cadence not in CADENCES:
        raise ValueError(f"cadence must be one of {', '.join(repr(c) for c in CADENCES)}")
    if cadence == "uniform-phase":
        return uniform_phase_times(system.period, system.n_transits, start=system.t_peri)
    if cadence == "gost":
        if system.ra_deg is None or system.dec_deg is None:
            raise ValueError(
                f"system {system.name!r} has no ra_deg/dec_deg, so the GOST cadence has no "
                "position to forecast for; draw the population with draw_population (which "
                "gives every system one), take it from a catalogue that has coordinates, or "
                "use cadence='scanning-law'"
            )
        transits = gost_transits(float(system.ra_deg), float(system.dec_deg))
        return rvs_transit_times_from_gost(transits, release=release, seed=seed)
    return rvs_transit_times(system.n_transits, seed=seed)


def simulate_system(
    system: BinarySystem,
    *,
    library,
    product: RVSProduct = RVS_DR4_EPOCH,
    cadence: str = "scanning-law",
    resolving_power: str = "nominal",
    seed: int = 0,
    grid: LogGrid | None = None,
    bjd=None,
) -> tuple[Dataset, RVSTruth, LogGrid, list[np.ndarray]]:
    """Simulate one system's RVS epochs from the library.

    The components are rendered from the library at the system's labels and rotation,
    the epochs are placed by the cadence model (:func:`system_transit_times`, or the
    ``bjd`` given), the S/N follows from G_RVS with one transit per epoch, and the
    delivered product is the one requested.

    Returns
    -------
    (Dataset, RVSTruth, LogGrid, list of arrays)
        The delivered dataset, the truth record, the model grid and the two component
        deviation spectra on it.
    """
    library = _resolve_library(library)
    labels1, labels2 = system.labels
    if grid is None:
        v_max = abs(system.gamma) + max(system.k1, system.k2) * (1.0 + system.ecc) + 30.0
        grid = rvs_model_grid(v_max, vsini_max_kms=max(system.vsini1, system.vsini2, 1.0))
    components = rvs_components(
        library, [labels1, labels2], grid, vsini_kms=[system.vsini1, system.vsini2]
    )
    if bjd is None:
        bjd = system_transit_times(system, cadence=cadence, seed=seed)
    bjd = np.asarray(bjd, dtype=np.float64)
    r_ep = (
        rvs_transit_resolving_powers(bjd.size, seed=seed)
        if resolving_power == "per-transit"
        else None
    )
    dataset, truth = simulate_rvs_dataset(
        components,
        grid,
        bjd=bjd,
        light_fractions=system.light_fractions,
        orbit=system.orbit(),
        grvs=system.grvs,
        product=product,
        library_resolving_power=float(library.meta.get("resolution", 20_000.0)),
        resolving_power=r_ep,
        declare_lsf="nominal",
        seed=seed,
    )
    return dataset, truth, grid, components


def _bounds(library) -> dict[str, tuple[float, float]]:
    library = _resolve_library(library)
    return dict(library.bounds)


def _label_prior(value: float, half_width: float, bounds: tuple[float, float]):
    lo = max(value - half_width, bounds[0])
    hi = min(value + half_width, bounds[1])
    if not hi > lo:
        return None
    return (float(lo), float(hi))


def build_star(
    system: BinarySystem,
    tier: Tier,
    *,
    dataset: Dataset,
    truth: RVSTruth,
    grid: LogGrid,
    components: Sequence[np.ndarray],
    config: BenchmarkConfig,
    bounds: Mapping[str, tuple[float, float]],
) -> StarConfig:
    """The pipeline declaration of one system under one tier, with its truth block."""
    names = ("A", "B")
    lights = system.light_fractions
    ks = (system.k1, system.k2)
    teffs = (system.teff1, system.teff2)
    loggs = (system.logg1, system.logg2)
    components_cfg = []
    for i, name in enumerate(names):
        kwargs: dict[str, Any] = {}
        if tier.labels == "narrow":
            kwargs["teff"] = _label_prior(teffs[i], 300.0, bounds["teff"])
            kwargs["logg"] = _label_prior(loggs[i], 0.3, bounds["logg"])
        if tier.elements == "known":
            kwargs["k"] = {"value": float(ks[i]), "sigma": float(0.15 * ks[i])}
        light: Any = "measure" if tier.light == "measure" else float(lights[i])
        components_cfg.append(ComponentConfig(name, light, **kwargs))

    if tier.period == "known":
        period: Any = {
            "value": float(system.period),
            "sigma": float(tier.period_sigma * system.period),
        }
    elif tier.period == "range":
        period = [
            float(system.period * (1.0 - tier.period_sigma)),
            float(system.period * (1.0 + tier.period_sigma)),
        ]
    else:
        period = "search"
    t_conj: Any = None
    if tier.t_conj == "known":
        t_conj = {"value": float(system.t_conj), "sigma": float(0.01 * system.period)}
    ecc: Any = None
    omega: Any = None
    if tier.elements == "known":
        ecc = float(system.ecc)
        omega = float(system.omega) if system.ecc > 0.0 else None

    truth_block: dict[str, Any] = {
        "k": [float(system.k1), float(system.k2)],
        "period": float(system.period),
        "ecc": float(system.ecc),
        "omega": float(system.omega),
        "t_conj": float(system.t_conj),
        "gamma": float(system.gamma),
        "light_fractions": [float(lights[0]), float(lights[1])],
        "velocities": np.asarray(truth.velocities),
        "components": np.asarray(components),
        "grid": grid,
        "labels": {
            "A": {
                "teff": system.teff1,
                "logg": system.logg1,
                "mh": system.mh,
                "vsini": system.vsini1,
            },
            "B": {
                "teff": system.teff2,
                "logg": system.logg2,
                "mh": system.mh,
                "vsini": system.vsini2,
            },
        },
        "windows": {k: [tuple(w) for w in v] for k, v in config.windows.items()},
    }
    overrides: dict[str, Any] = {"sample": bool(tier.sample)}
    if config.noise_model == "correlated":
        # An instrument property, declared like the LSF: the correlation the delivery
        # onto the archive grid gives the noise, which the archive's errors do not carry.
        overrides["noise_correlation"] = {"RVS": float(np.nanmean(truth.delivery.lag1))}
    lsf: dict[str, Any] = {"RVS": {"resolving_power": 11_500.0}}
    return StarConfig(
        name=star_name(system, tier),
        dataset=dataset,
        period=period,
        t_conj=t_conj,
        ecc=ecc,
        omega=omega,
        components=components_cfg,
        lsf=lsf,
        truth=truth_block,
        overrides=overrides,
    )


def _analysis(config: BenchmarkConfig) -> Analysis:
    values: dict[str, Any] = {
        "dv_kms": config.dv_kms,
        "k_min": config.k_min,
        "k_max": config.k_max,
        "ecc_max": config.ecc_max,
        "max_steps": config.max_steps,
        "label_steps": config.label_steps,
        "v_range": config.k_max + 100.0,
        "vsini_max": 150.0,
        "v_zero_range": 150.0,
        "plots": config.plots,
        "fast": config.fast,
    }
    if config.mask_ca:
        values["mask"] = tuple(CA_TRIPLET_WINDOWS)
    values.update(config.analysis)
    return Analysis(**values)


def _planned(config: BenchmarkConfig) -> list[tuple[BinarySystem, Tier]]:
    """The (system, tier) pairs to run, before the epochs are known.

    The minimum-transit rule is applied here on the record's ``n_transits``, which is the
    number of epochs the scanning-law and uniform-phase cadences place. Under the GOST
    cadence the count is the service's, so the rule is applied in :func:`build_stars`
    instead, once the epochs exist.
    """
    plan = []
    for system in config.systems:
        if config.cadence != "gost" and system.n_transits < config.min_transits:
            continue
        for tier in config.tiers:
            if tier.eclipsing_only and not system.eclipsing:
                continue
            plan.append((system, tier))
    return plan


def build_stars(
    config: BenchmarkConfig, *, progress: bool = True
) -> tuple[list[StarConfig], dict[str, dict[str, Any]]]:
    """Simulate every system once and declare it under every applicable tier.

    Returns the star declarations and, per system, a record of the simulation (S/N per
    epoch, delivered noise correlation, the resolving powers applied). A system whose
    cadence yields fewer epochs than ``min_transits`` is simulated no further and is
    recorded with ``skipped`` set; under the GOST cadence, where the count is the
    service's, that is where the minimum-transit rule takes effect.
    """
    library = _resolve_library(config.library)
    bounds = _bounds(library)
    stars: list[StarConfig] = []
    records: dict[str, dict[str, Any]] = {}
    plan = _planned(config)
    systems = []
    for system, _ in plan:
        if system not in systems:
            systems.append(system)
    t0 = time.perf_counter()
    for index, system in enumerate(systems):
        seed = config.seed * 100_003 + index
        bjd = system_transit_times(
            system, cadence=config.cadence, seed=seed, release=config.release
        )
        if bjd.size < config.min_transits:
            records[system.name] = {
                "seed": seed,
                "n_epochs": int(bjd.size),
                "skipped": f"{bjd.size} epochs, below min_transits = {config.min_transits}",
            }
            if progress:
                print(
                    f"skipped {system.name}: {bjd.size} epochs from the {config.cadence} "
                    f"cadence, below min_transits = {config.min_transits}",
                    flush=True,
                )
            continue
        dataset, truth, grid, components = simulate_system(
            system,
            library=library,
            product=config.product,
            cadence=config.cadence,
            resolving_power=config.resolving_power,
            seed=seed,
            bjd=bjd,
        )
        records[system.name] = {
            "seed": seed,
            "snr_epoch": float(np.median(truth.snr)),
            "n_epochs": int(dataset.n_epochs),
            "lag1": float(np.nanmean(truth.delivery.lag1)),
            "pixel_ratio": float(truth.delivery.pixel_ratio),
            "resolving_power_min": float(np.min(truth.resolving_power)),
            "resolving_power_max": float(np.max(truth.resolving_power)),
            "grid_n": int(grid.n),
            "baseline_days": float(dataset.bjd.max() - dataset.bjd.min()),
        }
        for tier in config.tiers:
            if tier.eclipsing_only and not system.eclipsing:
                continue
            stars.append(
                build_star(
                    system,
                    tier,
                    dataset=dataset,
                    truth=truth,
                    grid=grid,
                    components=components,
                    config=config,
                    bounds=bounds,
                )
            )
        if progress:
            print(
                f"simulated {index + 1}/{len(systems)} {system.name}: {dataset.n_epochs} epochs, "
                f"S/N {records[system.name]['snr_epoch']:.1f}/px, "
                f"{time.perf_counter() - t0:.0f} s",
                flush=True,
            )
    return stars, records


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BenchmarkRun:
    """A finished benchmark: where it is, its rows, and the report path."""

    directory: Path
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    report: Path | None
    seconds: float


def _machine() -> str:
    cpu = platform.processor() or platform.machine()
    return f"{platform.system()} {platform.release()}, {cpu}, {os.cpu_count()} threads"


def _manifest(config: BenchmarkConfig, stars, records) -> dict[str, Any]:
    from albireo import __version__

    skipped = [
        system.name
        for system in config.systems
        if (config.cadence != "gost" and system.n_transits < config.min_transits)
        or records.get(system.name, {}).get("skipped")
    ]
    return {
        "albireo": __version__,
        "created": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "machine": _machine(),
        "title": config.title,
        "product": config.product.name,
        "product_description": config.product.description,
        "library": config.library if isinstance(config.library, str) else "in memory",
        "cadence": config.cadence,
        "resolving_power": config.resolving_power,
        "tiers": [dataclasses.asdict(t) for t in config.tiers],
        "settings": {
            "dv_kms": config.dv_kms,
            "k_min": config.k_min,
            "k_max": config.k_max,
            "ecc_max": config.ecc_max,
            "max_steps": config.max_steps,
            "label_steps": config.label_steps,
            "mask_ca": config.mask_ca,
            "noise_model": config.noise_model,
            "min_transits": config.min_transits,
            "fast": config.fast,
            "seed": config.seed,
            "jobs": config.jobs,
        },
        "windows": {k: [list(w) for w in v] for k, v in config.windows.items()},
        "n_systems": len(config.systems),
        "n_planned": len(stars),
        "stars": [star.name for star in stars],
        "skipped_transits": skipped,
        "simulation": records,
    }


def run_benchmark(config: BenchmarkConfig, *, progress: bool = True) -> BenchmarkRun:
    """Simulate, run the pipeline under every tier, collect the truth blocks, and report.

    With ``resume`` (the default) stars whose ``result.json`` already exists in
    ``output`` are not run again, so an interrupted benchmark continues where it stopped
    and a report can be regenerated at any time with :func:`write_report`.
    """
    t0 = time.perf_counter()
    directory = Path(config.output)
    directory.mkdir(parents=True, exist_ok=True)
    write_population(directory / "population.json", list(config.systems))
    stars, records = build_stars(config, progress=progress)
    manifest = _manifest(config, stars, records)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    todo = stars
    if config.resume:
        todo = [s for s in stars if not _result_path(directory, s.name).is_file()]
        if progress and len(todo) < len(stars):
            print(f"resuming: {len(stars) - len(todo)} of {len(stars)} stars already have results")
    if todo:
        library = config.library
        bounds = _bounds(library)
        pipeline = PipelineConfig(
            stars=todo,
            output=directory,
            library=library,
            mh=tuple(bounds["mh"]),
            analysis=_analysis(config),
        )
        run_pipeline(pipeline, jobs=config.jobs, progress=progress)
    rows = collect(directory)
    summary = summarize(rows)
    report = write_report(directory, title=config.title) if rows else None
    return BenchmarkRun(
        directory=directory,
        rows=rows,
        summary=summary,
        report=report,
        seconds=time.perf_counter() - t0,
    )


def _safe_name(name: str) -> str:
    import re

    cleaned = re.sub(r"[^\w.-]+", "_", name.strip()).strip("._")
    return cleaned or "star"


def _result_path(directory: Path, name: str) -> Path:
    return directory / _safe_name(name) / "result.json"


# ---------------------------------------------------------------------------
# Collecting and summarizing
# ---------------------------------------------------------------------------


def _get(mapping, *keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def collect(directory) -> list[dict[str, Any]]:
    """One row per planned star: the truth, what the pipeline recovered, and the metrics.

    Reads ``population.json``, ``manifest.json`` and every star's ``result.json`` in
    ``directory``. Stars without a result are rows with ``status = "missing"``.
    """
    directory = Path(directory)
    systems = {s.name: s for s in read_population(directory / "population.json")}
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    records = manifest.get("simulation", {})
    rows = []
    for name in manifest["stars"]:
        system_name, tier_name = name.rsplit("__", 1)
        system = systems[system_name]
        record = records.get(system_name, {})
        row: dict[str, Any] = {
            "star": name,
            "system": system_name,
            "tier": tier_name,
            "source": system.source,
            "eclipsing": bool(system.eclipsing),
            "period": system.period,
            "ecc": system.ecc,
            "k1": system.k1,
            "k2": system.k2,
            "q": system.q,
            "gamma": system.gamma,
            "light_ratio": system.light_ratio,
            "teff1": system.teff1,
            "teff2": system.teff2,
            "vsini1": system.vsini1,
            "vsini2": system.vsini2,
            "grvs": system.grvs,
            "n_transits": system.n_transits,
            "snr_epoch": record.get("snr_epoch", float(rvs_snr_per_pixel(system.grvs))),
            "snr_total": record.get("snr_epoch", float(rvs_snr_per_pixel(system.grvs)))
            * math.sqrt(system.n_transits),
            "max_separation_kms": system.max_separation_kms,
            "separation_over_fwhm": system.max_separation_kms / (2.3548 * rvs_lsf_sigma_kms()),
            "baseline_days": record.get("baseline_days"),
            "lag1": record.get("lag1"),
        }
        path = _result_path(directory, name)
        if not path.is_file():
            error = path.with_name("error.txt")
            if error.is_file():
                # The pipeline writes the traceback and no result for a failed star.
                tail = [line for line in error.read_text(encoding="utf-8").splitlines() if line]
                row["status"] = "failed"
                row["error"] = tail[-1] if tail else "failed"
            else:
                row["status"] = "missing"
            rows.append(row)
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        row["status"] = report.get("status", "ok")
        row["error"] = report.get("error")
        row["seconds"] = _get(report, "seconds", "total")
        row["flags"] = list(report.get("flags", []))
        row["n_flags"] = len(row["flags"])
        row["z_rms"] = _get(report, "disentangling", "z_rms")
        row["k_scan_used"] = _get(report, "disentangling", "k_scan", "used")
        row["k_scan_gain"] = _get(report, "disentangling", "k_scan", "gain_nats")
        row["noise_model"] = _get(report, "declaration", "noise_model", "kind")
        row["exchanged"] = bool(_get(report, "truth", "components_exchanged") or False)
        row["n_usable"] = _get(report, "velocities", "n_usable")
        row["absolute"] = _get(report, "velocities", "absolute_all")
        truth = report.get("truth") or {}
        k_true = {"A": system.k1, "B": system.k2}
        for comp in ("A", "B"):
            d = _get(truth, "k_table", comp)
            row[f"k_{comp}_rel"] = None if d is None else d / k_true[comp]
            row[f"k_{comp}_pull"] = _get(truth, "k_table_pull", comp)
            d = _get(truth, "k_disentangling", comp)
            row[f"k_{comp}_rel_dis"] = None if d is None else d / k_true[comp]
            row[f"v_rms_{comp}"] = _get(truth, "velocity_rms", comp)
            row[f"v_pull_{comp}"] = _get(truth, "velocity_pull_rms", comp)
            row[f"v_rms_kep_{comp}"] = _get(truth, "velocity_rms_keplerian", comp)
            row[f"gamma_err_{comp}"] = _get(truth, "gamma", comp)
            row[f"gamma_pull_{comp}"] = _get(truth, "gamma_pull", comp)
            row[f"spec_corr_{comp}"] = _get(truth, "spectra", comp, "corr")
            row[f"spec_rms_{comp}"] = _get(truth, "spectra", comp, "rms")
            row[f"spec_pull_{comp}"] = _get(truth, "spectra", comp, "pull_rms")
            for window in ("ca_triplet", "metal"):
                row[f"ew_{window}_{comp}"] = _get(
                    truth, "spectra", comp, "windows", window, "ratio"
                )
            for label in ("teff", "logg", "mh", "vsini"):
                row[f"{label}_err_{comp}"] = _get(truth, "labels", comp, label)
                row[f"{label}_pull_{comp}"] = _get(truth, "labels", comp, f"{label}_pull")
            row[f"light_err_{comp}"] = _get(truth, "light_label_fit", comp)
            row[f"light_declared_err_{comp}"] = _get(truth, "light_declared", comp)
        period_diff = _get(truth, "period")
        row["period_rel"] = None if period_diff is None else 1.0 + period_diff / system.period
        row["period_pull"] = _get(truth, "elements_table_pull", "period")
        row["ecc_err"] = _get(truth, "elements_table", "ecc")
        row["ecc_pull"] = _get(truth, "elements_table_pull", "ecc")
        # The argument of periastron is undefined for a circular orbit; below e = 0.05 the
        # comparison would measure the noise's choice of direction.
        eccentric = float(system.ecc) >= 0.05
        row["omega_err_deg"] = _get(truth, "elements_table", "omega_deg") if eccentric else None
        row["omega_pull"] = _get(truth, "elements_table_pull", "omega_deg") if eccentric else None
        t_conj = _get(truth, "elements_table", "t_conj")
        row["t_conj_err_phase"] = None if t_conj is None else t_conj / system.period
        row["t_conj_pull"] = _get(truth, "elements_table_pull", "t_conj")
        row["ecc_err_dis"] = _get(truth, "elements_disentangling", "ecc")
        boot = _get(report, "bootstrap", "period")
        row["bootstrap_period_rel"] = None if boot is None else boot / system.period - 1.0
        row["period_recovered"] = (
            None if boot is None else bool(abs(row["bootstrap_period_rel"]) < 0.02)
        )
        dchi2 = _get(report, "velocities", "delta_chi2", "B")
        row["detected_B"] = (
            None if not dchi2 else float(np.mean(np.asarray(dchi2, dtype=float) > 25.0))
        )
        rows.append(row)
    return rows


def collect_velocities(directory) -> list[dict[str, Any]]:
    """Every measured epoch velocity of the run, one row per star, epoch and component.

    Reads each star's ``result.json``: the velocity table (TODCOR against the
    disentangled components, with the label fit's zero points), the injected velocity of
    the epoch in the order the components were compared in, and the disentangling's own
    Keplerian at the epoch (differential, without a systemic velocity). The phase is
    folded on the injected ephemeris.
    """
    directory = Path(directory)
    systems = {s.name: s for s in read_population(directory / "population.json")}
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for name in manifest["stars"]:
        path = _result_path(directory, name)
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status", "ok") != "ok" or not report.get("velocities"):
            continue
        system_name, tier_name = name.rsplit("__", 1)
        system = systems[system_name]
        table = report["velocities"]
        if not table.get("bjd") or not table.get("names"):
            continue  # a report written without the per-epoch columns
        truth = report.get("truth") or {}
        v_true = truth.get("epoch_velocities_true") or {}
        keplerian = _get(report, "disentangling", "velocities") or {}
        exchanged = bool(truth.get("components_exchanged") or False)
        bjd = np.asarray(table["bjd"], dtype=float)
        phase = ((bjd - system.t_conj) / system.period) % 1.0
        for j in range(bjd.size):
            for comp in table["names"]:
                v = table["velocity"][comp][j]
                sigma = table["sigma"][comp][j]
                injected = None if comp not in v_true else v_true[comp][j]
                residual = None if injected is None or v is None else float(v) - float(injected)
                pull = (
                    None
                    if residual is None or sigma is None or not float(sigma) > 0.0
                    else residual / float(sigma)
                )
                rows.append(
                    {
                        "star": name,
                        "system": system_name,
                        "tier": tier_name,
                        "component": comp,
                        "epoch": j,
                        "bjd": float(bjd[j]),
                        "phase": float(phase[j]),
                        "v_kms": v,
                        "sigma_kms": sigma,
                        "good": bool(table["good"][j]),
                        "blended": bool(table["blended"][j]),
                        "delta_chi2": table["delta_chi2"][comp][j],
                        "light": table["light"][comp][j],
                        "absolute": bool(table["absolute"][comp]),
                        "v_true_kms": injected,
                        "residual_kms": residual,
                        "pull": pull,
                        "v_keplerian_dis_kms": None
                        if comp not in keplerian
                        else keplerian[comp][j],
                        "exchanged": exchanged,
                    }
                )
    return rows


def summarize_velocities(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-tier statistics of the epoch velocities: counts, residuals and pulls per component."""
    tiers = sorted(
        {r["tier"] for r in rows}, key=lambda t: list(TIERS).index(t) if t in TIERS else 99
    )
    out: dict[str, Any] = {}
    for tier in tiers:
        sub = [r for r in rows if r["tier"] == tier]
        entry: dict[str, Any] = {
            "n_epochs": len({(r["star"], r["epoch"]) for r in sub}),
            "n_systems": len({r["system"] for r in sub}),
            "usable_fraction": _fraction(
                [1.0 if r["good"] else 0.0 for r in sub if r["component"] == "A"],
                lambda a: a > 0.5,
            ),
        }
        for comp in ("A", "B"):
            good = [r for r in sub if r["component"] == comp and r["good"]]
            residuals = _finite([r.get("residual_kms") for r in good])
            pulls = _finite([r.get("pull") for r in good])
            entry[f"rms_{comp}"] = float(np.sqrt(np.mean(residuals**2))) if residuals.size else None
            entry[f"median_abs_{comp}"] = (
                float(np.median(np.abs(residuals))) if residuals.size else None
            )
            entry[f"pull_rms_{comp}"] = float(np.sqrt(np.mean(pulls**2))) if pulls.size else None
            entry[f"within_3sigma_{comp}"] = _fraction(pulls, lambda a: np.abs(a) < 3.0)
            entry[f"median_sigma_{comp}"] = (_stats([r.get("sigma_kms") for r in good]) or {}).get(
                "median"
            )
        out[tier] = entry
    return out


def _finite(values) -> np.ndarray:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _stats(values) -> dict[str, float | None] | None:
    arr = _finite(values)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
        "mean": float(np.mean(arr)),
    }


def _abs_stats(values):
    return _stats([None if v is None else abs(float(v)) for v in values])


def _fraction(values, predicate) -> float | None:
    arr = _finite(values)
    if arr.size == 0:
        return None
    return float(np.mean(predicate(arr)))


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-tier summary statistics of the collected rows."""
    tiers = sorted(
        {r["tier"] for r in rows}, key=lambda t: list(TIERS).index(t) if t in TIERS else 99
    )
    out: dict[str, Any] = {"tiers": {}, "n_rows": len(rows)}
    for tier in tiers:
        sub = [r for r in rows if r["tier"] == tier]
        ok = [r for r in sub if r.get("status") == "ok"]
        entry: dict[str, Any] = {
            "n": len(sub),
            "n_ok": len(ok),
            "n_failed": sum(r.get("status") == "failed" for r in sub),
            "n_missing": sum(r.get("status") == "missing" for r in sub),
            "seconds_median": _stats([r.get("seconds") for r in ok]),
        }
        for comp in ("A", "B"):
            entry[f"k_{comp}_abs_rel"] = _abs_stats([r.get(f"k_{comp}_rel") for r in ok])
            entry[f"k_{comp}_within_5pct"] = _fraction(
                [r.get(f"k_{comp}_rel") for r in ok], lambda a: np.abs(a) < 0.05
            )
            entry[f"k_{comp}_within_1pct"] = _fraction(
                [r.get(f"k_{comp}_rel") for r in ok], lambda a: np.abs(a) < 0.01
            )
            entry[f"k_{comp}_abs_rel_dis"] = _abs_stats([r.get(f"k_{comp}_rel_dis") for r in ok])
            entry[f"k_{comp}_within_5pct_dis"] = _fraction(
                [r.get(f"k_{comp}_rel_dis") for r in ok], lambda a: np.abs(a) < 0.05
            )
            entry[f"k_{comp}_within_1pct_dis"] = _fraction(
                [r.get(f"k_{comp}_rel_dis") for r in ok], lambda a: np.abs(a) < 0.01
            )
            pulls = _finite([r.get(f"k_{comp}_pull") for r in ok])
            entry[f"k_{comp}_pull_rms"] = float(np.sqrt(np.mean(pulls**2))) if pulls.size else None
            entry[f"k_{comp}_pull_within_1"] = _fraction(pulls, lambda a: np.abs(a) < 1.0)
            entry[f"k_{comp}_pull_within_2"] = _fraction(pulls, lambda a: np.abs(a) < 2.0)
            entry[f"v_rms_{comp}"] = _stats([r.get(f"v_rms_{comp}") for r in ok])
            entry[f"v_pull_{comp}"] = _stats([r.get(f"v_pull_{comp}") for r in ok])
            entry[f"spec_corr_{comp}"] = _stats([r.get(f"spec_corr_{comp}") for r in ok])
            entry[f"spec_pull_{comp}"] = _stats([r.get(f"spec_pull_{comp}") for r in ok])
            entry[f"ew_metal_{comp}"] = _stats([r.get(f"ew_metal_{comp}") for r in ok])
            entry[f"ew_ca_triplet_{comp}"] = _stats([r.get(f"ew_ca_triplet_{comp}") for r in ok])
            entry[f"teff_abs_err_{comp}"] = _abs_stats([r.get(f"teff_err_{comp}") for r in ok])
            entry[f"logg_abs_err_{comp}"] = _abs_stats([r.get(f"logg_err_{comp}") for r in ok])
            entry[f"vsini_abs_err_{comp}"] = _abs_stats([r.get(f"vsini_err_{comp}") for r in ok])
            entry[f"light_abs_err_{comp}"] = _abs_stats([r.get(f"light_err_{comp}") for r in ok])
            entry[f"light_declared_abs_err_{comp}"] = _abs_stats(
                [r.get(f"light_declared_err_{comp}") for r in ok]
            )
            entry[f"gamma_abs_err_{comp}"] = _abs_stats([r.get(f"gamma_err_{comp}") for r in ok])
        entry["period_abs_rel"] = _abs_stats(
            [None if r.get("period_rel") is None else r["period_rel"] - 1.0 for r in ok]
        )
        entry["ecc_abs_err"] = _abs_stats([r.get("ecc_err") for r in ok])
        entry["exchanged_fraction"] = _fraction(
            [1.0 if r.get("exchanged") else 0.0 for r in ok], lambda a: a > 0.5
        )
        entry["omega_abs_err_deg"] = _abs_stats([r.get("omega_err_deg") for r in ok])
        entry["t_conj_abs_err_phase"] = _abs_stats([r.get("t_conj_err_phase") for r in ok])
        entry["z_rms"] = _stats([r.get("z_rms") for r in ok])
        entry["detected_B"] = _stats([r.get("detected_B") for r in ok])
        recovered = [r.get("period_recovered") for r in ok if r.get("period_recovered") is not None]
        entry["period_recovered_fraction"] = (
            None if not recovered else float(np.mean([bool(v) for v in recovered]))
        )
        entry["n_period_search"] = len(recovered)
        flags: dict[str, int] = {}
        for r in ok:
            for flag in r.get("flags", []):
                key = flag.split(":")[0].strip()
                flags[key] = flags.get(key, 0) + 1
        entry["flags"] = dict(sorted(flags.items(), key=lambda kv: -kv[1])[:12])
        out["tiers"][tier] = entry
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

# The categorical slots of the validated reference palette, in their fixed order; the
# report never uses more than three on one panel, and facets by tier beyond that.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
_INK = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e6e5e1"
_TIER_ORDER = ("oracle", "eclipsing", "orbit", "blind")


def _tier_colour(tier: str, tiers: Sequence[str]) -> str:
    return _SERIES[list(tiers).index(tier) % len(_SERIES)]


def _fmt(value, digits: int = 3, scale: float = 1.0, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, Mapping):
        if value.get("median") is None:
            return "n/a"
        return (
            f"{value['median'] * scale:.{digits}g} "
            f"({value['p16'] * scale:.{digits}g} to {value['p84'] * scale:.{digits}g}){suffix}"
        )
    return f"{float(value) * scale:.{digits}g}{suffix}"


def _pct(value) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.0f}%"


def _style_axes(ax) -> None:
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.xaxis.label.set_color(_INK)
    ax.yaxis.label.set_color(_INK)
    ax.title.set_color(_INK)


def _write_figures(
    directory: Path, rows: Sequence[Mapping[str, Any]], tiers: Sequence[str]
) -> dict[str, Path]:
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: dict[str, Path] = {}
    ok = [r for r in rows if r.get("status") == "ok"]
    fig_dir = directory / "figures"
    fig_dir.mkdir(exist_ok=True)

    def save(name: str, fig) -> None:
        path = fig_dir / f"{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures[name] = path

    def column(subset, key):
        return np.array([np.nan if r.get(key) is None else float(r[key]) for r in subset])

    # 1. The population as drawn (one row per system; the first tier's rows suffice).
    first = tiers[0] if tiers else None
    systems = [r for r in rows if r["tier"] == first] if first else rows
    if systems:
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
        ecl = [r for r in systems if r["eclipsing"]]
        non = [r for r in systems if not r["eclipsing"]]
        ax = axes[0]
        for subset, colour, label in (
            (non, _SERIES[0], "not eclipsing"),
            (ecl, _SERIES[1], "eclipsing"),
        ):
            if subset:
                ax.scatter(
                    column(subset, "period"),
                    column(subset, "grvs"),
                    s=14,
                    color=colour,
                    label=label,
                    alpha=0.85,
                    edgecolors="none",
                )
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_xlabel("period [d]")
        ax.set_ylabel("G_RVS")
        ax.set_title("the population", fontsize=10)
        if ecl and non:
            ax.legend(fontsize=8, frameon=False)
        for ax, key, label in (
            (axes[1], "max_separation_kms", "largest separation [km/s]"),
            (axes[2], "n_transits", "RVS transits"),
            (axes[3], "light_ratio", "light ratio F2/F1"),
        ):
            values = column(systems, key)
            values = values[np.isfinite(values)]
            if values.size:
                ax.hist(values, bins=20, color=_SERIES[0], edgecolor="white", linewidth=0.8)
            ax.set_xlabel(label)
            ax.set_ylabel("systems")
        for ax in axes:
            _style_axes(ax)
        fig.tight_layout()
        save("population", fig)

    # 2. Semi-amplitude recovery against G_RVS, faceted by tier: the disentangling's
    #    Keplerian above, the orbit fitted to the velocity table below.
    if ok:
        n_t = len(tiers)
        fig, axes = plt.subplots(2, n_t, figsize=(3.6 * n_t, 6.4), squeeze=False)
        for j, tier in enumerate(tiers):
            sub = [r for r in ok if r["tier"] == tier]
            for i, (suffix, source) in enumerate((("_dis", "disentangling"), ("", "table"))):
                ax = axes[i][j]
                for comp, colour, marker in (("A", _SERIES[0], "o"), ("B", _SERIES[1], "s")):
                    rel = np.abs(column(sub, f"k_{comp}_rel{suffix}"))
                    ax.scatter(
                        column(sub, "grvs"),
                        np.maximum(rel, 1e-5),
                        s=14,
                        color=colour,
                        marker=marker,
                        label=f"K_{comp}",
                        alpha=0.85,
                        edgecolors="none",
                    )
                ax.axhline(0.01, color=_MUTED, linewidth=0.8, linestyle=":")
                ax.axhline(0.05, color=_MUTED, linewidth=0.8, linestyle="--")
                ax.set_yscale("log")
                ax.set_xlabel("G_RVS")
                ax.set_ylabel(f"|dK / K|, {source}")
                if i == 0:
                    ax.set_title(f"tier: {tier}", fontsize=10)
                    ax.legend(fontsize=8, frameon=False, loc="upper left")
                _style_axes(ax)
        fig.tight_layout()
        save("k_recovery", fig)

        # 3. Against the number of transits and the separation.
        fig, axes = plt.subplots(2, n_t, figsize=(3.6 * n_t, 6.4), squeeze=False)
        for j, tier in enumerate(tiers):
            sub = [r for r in ok if r["tier"] == tier]
            for i, (key, label) in enumerate(
                (
                    ("n_transits", "RVS transits"),
                    ("separation_over_fwhm", "largest separation / LSF FWHM"),
                )
            ):
                ax = axes[i][j]
                for comp, colour, marker in (("A", _SERIES[0], "o"), ("B", _SERIES[1], "s")):
                    rel = np.abs(column(sub, f"k_{comp}_rel_dis"))
                    ax.scatter(
                        column(sub, key),
                        np.maximum(rel, 1e-5),
                        s=14,
                        color=colour,
                        marker=marker,
                        label=f"K_{comp}",
                        alpha=0.85,
                        edgecolors="none",
                    )
                ax.axhline(0.01, color=_MUTED, linewidth=0.8, linestyle=":")
                ax.set_yscale("log")
                ax.set_xlabel(label)
                ax.set_ylabel("|dK / K|")
                if i == 0:
                    ax.set_title(f"tier: {tier}", fontsize=10)
                    ax.legend(fontsize=8, frameon=False, loc="upper right")
                _style_axes(ax)
        fig.tight_layout()
        save("k_vs_transits_separation", fig)

        # 4. Pulls: the quoted errors against the truth.
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        grid = np.linspace(-5, 5, 200)
        for ax, keys, title in (
            (axes[0], ("k_A_pull", "k_B_pull"), "semi-amplitude pulls"),
            (axes[1], ("v_pull_A", "v_pull_B"), "epoch-velocity pull rms"),
        ):
            for tier in tiers:
                sub = [r for r in ok if r["tier"] == tier]
                values = np.concatenate([column(sub, k) for k in keys])
                values = values[np.isfinite(values)]
                if values.size:
                    if ax is axes[0]:
                        ax.hist(
                            np.clip(values, -5, 5),
                            bins=30,
                            range=(-5, 5),
                            histtype="step",
                            linewidth=1.6,
                            color=_tier_colour(tier, tiers),
                            label=f"{tier} (rms {np.sqrt(np.mean(values**2)):.2f})",
                            density=True,
                        )
                    else:
                        ax.hist(
                            np.clip(values, 0, 6),
                            bins=30,
                            range=(0, 6),
                            histtype="step",
                            linewidth=1.6,
                            color=_tier_colour(tier, tiers),
                            label=f"{tier} (median {np.median(values):.2f})",
                            density=True,
                        )
            if ax is axes[0]:
                ax.plot(
                    grid,
                    np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi),
                    color=_MUTED,
                    linewidth=1.0,
                    linestyle="--",
                    label="N(0, 1)",
                )
                ax.set_xlabel("(K fitted - K true) / quoted error")
            else:
                ax.axvline(1.0, color=_MUTED, linewidth=1.0, linestyle="--")
                ax.set_xlabel("rms of (v - v_true) / sigma per system")
            ax.set_ylabel("density")
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8, frameon=False)
            _style_axes(ax)
        fig.tight_layout()
        save("pulls", fig)

        # 5. Epoch velocities and spectra against the signal.
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
        for tier in tiers:
            sub = [r for r in ok if r["tier"] == tier]
            colour = _tier_colour(tier, tiers)
            axes[0].scatter(
                column(sub, "snr_epoch"),
                column(sub, "v_rms_A"),
                s=14,
                color=colour,
                label=tier,
                alpha=0.85,
                edgecolors="none",
            )
            axes[1].scatter(
                column(sub, "snr_total"),
                column(sub, "spec_corr_A"),
                s=14,
                color=colour,
                label=tier,
                alpha=0.85,
                edgecolors="none",
            )
            axes[2].scatter(
                column(sub, "teff1"),
                column(sub, "teff_err_A"),
                s=14,
                color=colour,
                label=tier,
                alpha=0.85,
                edgecolors="none",
            )
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("S/N per detector pixel per epoch")
        axes[0].set_ylabel("rms epoch-velocity error, primary [km/s]")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("S/N per pixel, all epochs combined")
        axes[1].set_ylabel("correlation of recovered and true primary spectrum")
        axes[1].set_ylim(0.0, 1.02)
        axes[2].axhline(0.0, color=_MUTED, linewidth=0.8)
        axes[2].set_xlabel("true Teff of the primary [K]")
        axes[2].set_ylabel("fitted - true Teff, primary [K]")
        for ax in axes:
            ax.legend(fontsize=8, frameon=False)
            _style_axes(ax)
        fig.tight_layout()
        save("velocities_spectra_labels", fig)

        # 6. The period search, where it ran.
        searched = [r for r in ok if r.get("bootstrap_period_rel") is not None]
        if searched:
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            found = [r for r in searched if r.get("period_recovered")]
            lost = [r for r in searched if not r.get("period_recovered")]
            for subset, colour, label in (
                (found, _SERIES[0], "recovered within 2%"),
                (lost, _SERIES[1], "an alias or a miss"),
            ):
                if subset:
                    ax.scatter(
                        column(subset, "period"),
                        1.0 + column(subset, "bootstrap_period_rel"),
                        s=16,
                        color=colour,
                        label=f"{label} ({len(subset)})",
                        alpha=0.85,
                        edgecolors="none",
                    )
            ax.axhline(1.0, color=_MUTED, linewidth=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("true period [d]")
            ax.set_ylabel("searched period / true period")
            ax.set_title("the period search of the blind tier", fontsize=10)
            ax.legend(fontsize=8, frameon=False)
            _style_axes(ax)
            fig.tight_layout()
            save("period_search", fig)
    return figures


def _write_rv_figures(
    directory: Path,
    velocity_rows: Sequence[Mapping[str, Any]],
    systems: Mapping[str, BinarySystem],
    tiers: Sequence[str],
) -> dict[str, Path]:
    """Phase-folded epoch velocities against the injected orbit, one panel per system."""
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None or not velocity_rows:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: dict[str, Path] = {}
    fig_dir = directory / "figures"
    fig_dir.mkdir(exist_ok=True)
    dense = np.linspace(0.0, 1.0, 300)
    for tier in tiers:
        sub = [r for r in velocity_rows if r["tier"] == tier]
        names = sorted({r["system"] for r in sub}, key=lambda n: systems[n].period)
        if not names:
            continue
        n_cols = min(4, len(names))
        n_rows = math.ceil(len(names) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.4 * n_cols, 2.7 * n_rows), squeeze=False
        )
        for ax, name in zip(axes.ravel(), names, strict=False):
            system = systems[name]
            points = [r for r in sub if r["system"] == name]
            exchanged = any(r["exchanged"] for r in points)
            t = system.t_conj + dense * system.period
            for i, comp in enumerate(("A", "B")):
                injected = i if not exchanged else 1 - i
                k = (system.k1, system.k2)[injected]
                omega = system.omega + (0.0 if injected == 0 else math.pi)
                curve = radial_velocity(
                    t,
                    period=system.period,
                    t_peri=system.t_peri,
                    ecc=system.ecc,
                    omega=omega,
                    k=k,
                    gamma=system.gamma,
                )
                ax.plot(dense, np.asarray(curve), color=_SERIES[i], lw=0.9, alpha=0.55)
                mine = [r for r in points if r["component"] == comp and r["v_kms"] is not None]
                for flagged, filled in ((False, True), (True, False)):
                    pts = [r for r in mine if (not r["good"]) == flagged]
                    if not pts:
                        continue
                    ax.errorbar(
                        [r["phase"] for r in pts],
                        [r["v_kms"] for r in pts],
                        yerr=[0.0 if r["sigma_kms"] is None else r["sigma_kms"] for r in pts],
                        fmt="o" if comp == "A" else "s",
                        ms=3.2,
                        capsize=0,
                        color=_SERIES[i],
                        mfc=_SERIES[i] if filled else "white",
                        mew=0.8,
                        elinewidth=0.7,
                        alpha=0.9 if filled else 0.7,
                        label=f"{comp}" if filled else None,
                    )
            ax.set_title(
                f"{name}  P {system.period:.3g} d  G_RVS {system.grvs:.1f}  "
                f"n {system.n_transits}" + ("  (exchanged)" if exchanged else ""),
                fontsize=7,
            )
            ax.set_xlim(0.0, 1.0)
            ax.tick_params(labelsize=7)
            _style_axes(ax)
        for ax in axes.ravel()[len(names) :]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("phase from the injected conjunction", fontsize=8)
        for row in axes:
            row[0].set_ylabel("velocity [km/s]", fontsize=8)
        axes[0][0].legend(fontsize=7, frameon=False, loc="upper right")
        fig.suptitle(
            f"tier: {tier}. Measured epoch velocities (filled: usable; open: flagged) "
            "against the injected orbit",
            fontsize=9,
        )
        fig.tight_layout()
        path = fig_dir / f"rv_curves_{tier}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        figures[f"rv_curves_{tier}"] = path
    return figures


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("; ".join(v) if isinstance(v, list) else v) for k, v in r.items()})


def write_report(directory, *, title: str | None = None) -> Path:
    """Collect the results in ``directory`` and write ``report.md`` with its figures.

    Also writes ``rows.csv`` (one line per star with the truth and every metric) and
    ``summary.json`` (the per-tier statistics). Can be re-run on a finished or partial
    benchmark at any time.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    systems = read_population(directory / "population.json")
    rows = collect(directory)
    summary = summarize(rows)
    tiers = [t for t in _TIER_ORDER if t in summary["tiers"]] + [
        t for t in summary["tiers"] if t not in _TIER_ORDER
    ]
    _write_csv(directory / "rows.csv", rows)
    velocity_rows = collect_velocities(directory)
    summary["epoch_velocities"] = summarize_velocities(velocity_rows)
    _write_csv(directory / "velocities.csv", velocity_rows)
    (directory / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    figures = _write_figures(directory, rows, tiers)
    figures.update(_write_rv_figures(directory, velocity_rows, {s.name: s for s in systems}, tiers))
    title = title or manifest.get("title") or "albireo on Gaia RVS double-lined binaries"
    settings = manifest.get("settings", {})

    lines: list[str] = [f"# {title}", ""]
    lines.append(
        f"Generated {_dt.datetime.now(_dt.UTC).isoformat(timespec='seconds')} by albireo "
        f"{manifest.get('albireo')} on {manifest.get('machine')}. Product "
        f"`{manifest.get('product')}` ({manifest.get('product_description')}); library "
        f"`{manifest.get('library')}`; cadence `{manifest.get('cadence')}`; resolving power "
        f"`{manifest.get('resolving_power')}`; model grid {settings.get('dv_kms')} km/s; "
        f"K prior {settings.get('k_min')} to {settings.get('k_max')} km/s; eccentricity to "
        f"{settings.get('ecc_max')}; {settings.get('max_steps')} disentangling and "
        f"{settings.get('label_steps')} label steps"
        + ("; fast mode" if settings.get("fast") else "")
        + ("; Ca II triplet masked" if settings.get("mask_ca") else "")
        + (
            "; noise model AR(1) with the delivered grid's lag-one correlation"
            if settings.get("noise_model", "diagonal") == "correlated"
            else "; noise model diagonal"
        )
        + f"; seed {settings.get('seed')}."
    )
    lines.append("")
    lines.append(
        "Every system was simulated once with albireo's Gaia RVS model (photon noise per "
        "detector pixel at the S/N that follows from G_RVS, one transit per epoch, delivered "
        "on the archive grid with its correlated noise) and run through the pipeline under "
        "each tier; the numbers below come from the pipeline's own comparison against the "
        "injected truth. A pull is a difference over the quoted error; a calibrated error "
        "gives a pull rms near one."
    )
    lines.append("")
    lines.append("## The population")
    lines.append("")
    lines.append("```")
    lines.append(population_summary(systems))
    lines.append("```")
    lines.append("")
    skipped = manifest.get("skipped_transits", [])
    lines.append(
        f"{manifest.get('n_systems')} systems drawn, {manifest.get('n_planned')} star runs "
        f"planned over the tiers {', '.join(tiers)}"
        + (
            f"; {len(skipped)} {'systems' if len(skipped) != 1 else 'system'} left out for "
            "having fewer than "
            f"{settings.get('min_transits')} transits"
            if skipped
            else ""
        )
        + "."
    )
    if "population" in figures:
        lines.append("")
        lines.append("![the population](figures/population.png)")
    lines.append("")
    lines.append("## Tiers")
    lines.append("")
    lines.append("| tier | period | conjunction | elements | light fractions | label priors |")
    lines.append("|---|---|---|---|---|---|")
    for t in manifest.get("tiers", []):
        lines.append(
            f"| {t['name']} | {t['period']}"
            + (f" (width {t['period_sigma']:g})" if t["period"] != "search" else "")
            + f" | {t['t_conj']} | {t['elements']} | {t['light']} | {t['labels']} |"
        )
    lines.append("")
    lines.append("## Recovery by tier")
    lines.append("")
    lines.append("Median and the 16th to 84th percentile range over the systems that completed.")
    lines.append("")
    header = "| quantity | " + " | ".join(tiers) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(tiers))

    def row(label, key, **kw):
        cells = [_fmt(summary["tiers"][t].get(key), **kw) for t in tiers]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    def row_pct(label, key):
        cells = [_pct(summary["tiers"][t].get(key)) for t in tiers]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append(
        "| completed / planned | "
        + " | ".join(f"{summary['tiers'][t]['n_ok']} / {summary['tiers'][t]['n']}" for t in tiers)
        + " |"
    )
    row("abs(dK1 / K1) [%], disentangling", "k_A_abs_rel_dis", scale=100.0)
    row("abs(dK2 / K2) [%], disentangling", "k_B_abs_rel_dis", scale=100.0)
    row_pct("K1 within 1%, disentangling", "k_A_within_1pct_dis")
    row_pct("K2 within 1%, disentangling", "k_B_within_1pct_dis")
    row_pct("K1 within 5%, disentangling", "k_A_within_5pct_dis")
    row_pct("K2 within 5%, disentangling", "k_B_within_5pct_dis")
    row("abs(dK1 / K1) [%], velocity table", "k_A_abs_rel", scale=100.0)
    row("abs(dK2 / K2) [%], velocity table", "k_B_abs_rel", scale=100.0)
    row_pct("K1 within 1%, velocity table", "k_A_within_1pct")
    row_pct("K1 within 5%, velocity table", "k_A_within_5pct")
    row_pct("K2 within 5%, velocity table", "k_B_within_5pct")
    row("K1 pull rms, velocity table", "k_A_pull_rms")
    row("K2 pull rms, velocity table", "k_B_pull_rms")
    row_pct("K1 pull within 1, velocity table", "k_A_pull_within_1")
    row_pct("K2 pull within 2, velocity table", "k_B_pull_within_2")
    row_pct("components recovered in the other order", "exchanged_fraction")
    row("abs(dP / P) from the table", "period_abs_rel")
    row("abs(de)", "ecc_abs_err")
    row("abs(d omega) [deg]", "omega_abs_err_deg")
    row("abs(d t_conj) [phase]", "t_conj_abs_err_phase")
    row("abs(d gamma) A [km/s]", "gamma_abs_err_A")
    row("epoch velocity rms, A [km/s]", "v_rms_A")
    row("epoch velocity rms, B [km/s]", "v_rms_B")
    row("epoch velocity pull rms, A", "v_pull_A")
    row("epoch velocity pull rms, B", "v_pull_B")
    row("secondary detected per epoch (dchi2 > 25)", "detected_B")
    row("spectrum correlation, A", "spec_corr_A")
    row("spectrum correlation, B", "spec_corr_B")
    row("spectrum pull rms, A", "spec_pull_A")
    row("EW ratio, metal windows, A", "ew_metal_A")
    row("EW ratio, Ca II triplet, A", "ew_ca_triplet_A")
    row("EW ratio, metal windows, B", "ew_metal_B")
    row("abs(dTeff) A [K]", "teff_abs_err_A", digits=3)
    row("abs(dTeff) B [K]", "teff_abs_err_B", digits=3)
    row("abs(dlog g) A", "logg_abs_err_A")
    row("abs(dvsini) A [km/s]", "vsini_abs_err_A")
    row("abs(dlight) from the label fit, A", "light_abs_err_A")
    row("abs(dlight) as declared, A", "light_declared_abs_err_A")
    row("residual z rms", "z_rms")
    lines.append(
        "| period search recovered within 2% | "
        + " | ".join(
            (
                f"{_pct(summary['tiers'][t]['period_recovered_fraction'])} "
                f"of {summary['tiers'][t]['n_period_search']}"
                if summary["tiers"][t]["n_period_search"]
                else "n/a"
            )
            for t in tiers
        )
        + " |"
    )
    row("wall per star [s]", "seconds_median", digits=3)
    lines.append("")
    for name, caption in (
        (
            "k_recovery",
            "Relative error of the semi-amplitudes against G_RVS, from the Keplerian of the "
            "disentangling (above) and from the orbit fitted to the velocity table (below); "
            "the dotted and dashed lines mark 1% and 5%.",
        ),
        (
            "k_vs_transits_separation",
            "The semi-amplitudes of the disentangling against the number of transits and "
            "against the largest velocity separation in units of the line-spread FWHM "
            "(26 km/s).",
        ),
        (
            "pulls",
            "Calibration: the semi-amplitude pulls against a unit normal, and the "
            "per-system rms of the epoch-velocity pulls against one.",
        ),
        (
            "velocities_spectra_labels",
            "Epoch-velocity precision against the per-epoch S/N, the recovered primary "
            "spectrum's correlation with the truth against the combined S/N, and the "
            "primary's temperature error.",
        ),
        ("period_search", "The blind tier's period search."),
    ):
        if name in figures:
            lines.append(f"![{name}](figures/{name}.png)")
            lines.append("")
            lines.append(caption)
            lines.append("")
    lines.append("## Epoch velocities")
    lines.append("")
    lines.append(
        "One velocity per component per epoch, measured by TODCOR against the disentangled "
        "components with the label fit's zero points, is a product of every star run "
        "(`velocities.rv` and `velocities.csv` in its directory). `velocities.csv` in this "
        "directory gathers them all with the injected velocity of each epoch, one row per "
        "system, tier, epoch and component; the statistics below pool the usable epochs of "
        "every system in the tier."
    )
    lines.append("")
    velocity_tiers = [t for t in tiers if t in summary.get("epoch_velocities", {})]
    if velocity_tiers:
        lines.append("| quantity | " + " | ".join(velocity_tiers) + " |")
        lines.append("|---|" + "---|" * len(velocity_tiers))
        entries = summary["epoch_velocities"]

        def vrow(label, key, fmt):
            lines.append(
                f"| {label} | "
                + " | ".join(fmt(entries[t].get(key)) for t in velocity_tiers)
                + " |"
            )

        def num(digits):
            return lambda v: "n/a" if v is None else f"{float(v):.{digits}g}"

        vrow("systems, epochs measured", "n_epochs", lambda v: "n/a" if v is None else str(v))
        lines[-1] = (
            "| systems, epochs measured | "
            + " | ".join(
                f"{entries[t]['n_systems']}, {entries[t]['n_epochs']}" for t in velocity_tiers
            )
            + " |"
        )
        vrow("epochs usable", "usable_fraction", _pct)
        vrow("rms residual, A [km/s]", "rms_A", num(3))
        vrow("rms residual, B [km/s]", "rms_B", num(3))
        vrow("median abs residual, A [km/s]", "median_abs_A", num(3))
        vrow("median abs residual, B [km/s]", "median_abs_B", num(3))
        vrow("median quoted error, A [km/s]", "median_sigma_A", num(3))
        vrow("median quoted error, B [km/s]", "median_sigma_B", num(3))
        vrow("pull rms, A", "pull_rms_A", num(3))
        vrow("pull rms, B", "pull_rms_B", num(3))
        vrow("within 3 sigma, A", "within_3sigma_A", _pct)
        vrow("within 3 sigma, B", "within_3sigma_B", _pct)
        lines.append("")
    for t in tiers:
        if f"rv_curves_{t}" in figures:
            lines.append(f"![rv_curves_{t}](figures/rv_curves_{t}.png)")
            lines.append("")
    if any(f"rv_curves_{t}" in figures for t in tiers):
        lines.append(
            "Phase-folded measured velocities (circles the primary, squares the secondary; "
            "filled where the table calls the epoch usable, open where it flags it) against "
            "the injected orbit, one panel per system, per tier. A pair recovered in the "
            "other order is drawn against the exchanged truth and marked."
        )
        lines.append("")
    lines.append("## Failures and flags")
    lines.append("")
    for t in tiers:
        entry = summary["tiers"][t]
        lines.append(f"- **{t}**: {entry['n_failed']} failed, {entry['n_missing']} not run.")
        for flag, count in entry["flags"].items():
            lines.append(f"    - {count}x {flag}")
    failed = [r for r in rows if r.get("status") == "failed"]
    if failed:
        lines.append("")
        lines.append("Failed stars:")
        lines.append("")
        for r in failed[:30]:
            lines.append(f"- `{r['star']}`: {r.get('error')}")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `rows.csv`: one line per star run with the truth and every metric.")
    lines.append(
        "- `velocities.csv`: every measured epoch velocity with its error, flags, the "
        "injected velocity and the pull."
    )
    lines.append("- `summary.json`: the per-tier statistics tabulated above.")
    lines.append("- `population.json`, `manifest.json`: the systems and the settings.")
    lines.append(
        "- one directory per star with the pipeline's `summary.txt`, `result.json`, "
        "tables, spectra and figures."
    )
    lines.append("")
    path = directory / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
