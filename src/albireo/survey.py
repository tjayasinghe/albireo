"""Epoch velocities by correlation for a simulated survey of eclipsing binaries.

**Experimental.** The configuration, the columns of the tables and the file layout may
change.

The disentangling benchmark of :mod:`albireo.benchmark` costs minutes per system. This
module measures a population of 10^4 systems by correlation alone: every system is
simulated as Gaia RVS epoch spectra (:mod:`albireo.gaia`) with its eclipses
(:mod:`albireo.eclipsing`), both velocities are measured at every transit against library
templates (:func:`albireo.todcor`), and an orbit is fitted to the table at the photometric
ephemeris (:mod:`albireo.rvorbit`). Nothing is disentangled, and one system takes a few
seconds.

A system is measured under several *declarations* of what the analysis knows about the
templates, on the same epochs:

``injected``
    The injected labels and rotation, with the light fractions held at the injected
    values. It measures the estimator and the noise alone.
``classified``
    Labels with the errors of a classification, drawn once per system, with the light
    fractions measured.
``catalogue``
    One template for both stars, from the colour of the pair, with the light fractions
    measured.

and it can be simulated under a *variant* that adds one effect to the baseline:
``"null"`` (the primary alone, to count false detections of a secondary), ``"third-light"``,
``"activity"`` and ``"instrument"``.

:func:`run_system` is the chain for one system, :func:`run_survey` runs a population in
worker processes and writes one file per chunk, and :func:`collect` joins the chunks into
two tables, one row per system and declaration and one per epoch and declaration.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import functools
import hashlib
import json
import math
import os
import platform
import time
import traceback
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from albireo.data import Dataset
from albireo.eclipsing import (
    DEFAULT_LIBRARIES,
    LibrarySet,
    _teff_from_colour,
    eclipse_light,
    limb_darkening,
)
from albireo.gaia import (
    RVS_BAND,
    RVS_CONSTANTS,
    RVS_DR4_EPOCH,
    quadrature_sigma_kms,
    rvs_delivered_sigma_kms,
    rvs_detector_grid,
    rvs_model_grid,
    rvs_snr_per_pixel,
    rvs_transit_resolving_powers,
    rvs_transit_times,
    simulate_rvs_dataset,
)
from albireo.grids import C_KMS, LogGrid
from albireo.population import (
    _SYNCHRONOUS_KMS,
    BinarySystem,
    MainSequence,
    write_population,
)

__all__ = [
    "DECLARATIONS",
    "VARIANTS",
    "SurveyConfig",
    "collect",
    "declare",
    "read_tables",
    "run_survey",
    "run_system",
]

DECLARATIONS = ("injected", "classified", "catalogue")
"""The declarations of the templates, in the order of the ``decl`` column."""

VARIANTS = ("baseline", "null", "third-light", "activity", "instrument")
"""The variants of the simulation."""

# Vacuum wavelengths of the Ca II triplet in Angstrom, and the excess equivalent width of
# the emission in each line relative to that at 8542 (air).
_CA_TRIPLET = ((8500.35, 0.6), (8544.44, 1.0), (8664.52, 0.8))
_CA_EMISSION_SIGMA = 0.25  # Angstrom, before rotation


@dataclasses.dataclass(frozen=True)
class SurveyConfig:
    """What is simulated and how it is measured.

    Attributes
    ----------
    declarations
        The declarations measured, from :data:`DECLARATIONS`.
    variant
        One of :data:`VARIANTS`. ``"baseline"``: the pair with its eclipses. ``"null"``: the
        primary alone at the system's G_RVS, analysed with the two templates of the pair.
        ``"third-light"``: the tertiary recorded in ``meta["tertiary"]`` added where the
        system has one, unknown to the analysis. ``"activity"``: emission in the cores of
        the Ca II triplet of the stars with ``meta["activity_ew"]`` above zero, which no
        template has. ``"instrument"``: a background that varies between transits and a
        residual scale, slope and curvature of the normalisation per transit, analysed
        with ``scale="free"``.
    simulation
        ``"survey"``: one model grid for every system, eclipses, a zero-point offset and a
        resolving power per transit. ``"recorded"``: the simulation of
        :func:`albireo.benchmark.simulate_system`, with no eclipse, for systems of recorded
        runs.
    period_fit
        The declarations under which the orbit is also fitted with the period known and
        the conjunction free (:func:`albireo.rvorbit.assign_components`).
    v_search_kms
        Half-width of the velocity range searched for each component.
    v_max_kms, vsini_max_kms
        Largest velocity and rotation the model grid leaves room for.
    span_days
        Time span of the transits.
    zero_point_kms
        Standard deviation of the velocity offset of a transit, common to both stars.
    libraries
        Registry names of the library boxes (:class:`albireo.eclipsing.LibrarySet`).
    classification
        Errors of the ``classified`` declaration: the temperature error in K and as a
        fraction (the larger applies), the errors of ``log g`` and of ``[M/H]`` in dex, and
        the fractional error of ``v sin i``.
    background_factor
        ``"instrument"``: the background of a transit is the mission median times a
        factor drawn log-uniformly between the reciprocal of this number and it.
    response_order, response_amplitude
        ``"instrument"``: the polynomial of the normalisation residual
        (:func:`albireo.gaia.simulate_rvs_dataset`).
    """

    declarations: tuple[str, ...] = DECLARATIONS
    variant: str = "baseline"
    simulation: str = "survey"
    period_fit: tuple[str, ...] = DECLARATIONS
    v_search_kms: float = 450.0
    v_max_kms: float = 700.0
    vsini_max_kms: float = 320.0
    span_days: float = 2005.0
    zero_point_kms: float = 0.17
    libraries: tuple[str, ...] = DEFAULT_LIBRARIES
    classification: tuple[float, float, float, float, float] = (150.0, 0.03, 0.2, 0.15, 0.2)
    background_factor: float = 5.5
    response_order: int = 2
    response_amplitude: float = 0.02

    def __post_init__(self) -> None:
        for name in self.declarations:
            if name not in DECLARATIONS:
                raise ValueError(f"unknown declaration {name!r}; choose from {DECLARATIONS}")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}; got {self.variant!r}")
        if self.simulation not in ("survey", "recorded"):
            raise ValueError("simulation must be 'survey' or 'recorded'")
        if not self.v_search_kms > 0.0:
            raise ValueError("v_search_kms must be positive")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self)))

    @classmethod
    def from_dict(cls, data) -> SurveyConfig:
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(
            **{k: (tuple(v) if isinstance(v, list) else v) for k, v in data.items() if k in fields}
        )


# ---------------------------------------------------------------------------
# What a process keeps between systems
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Context:
    libraries: LibrarySet
    grid: LogGrid
    delivered_kms: float
    detector_wave: np.ndarray
    template_grids: dict[float, LogGrid] = dataclasses.field(default_factory=dict)


_LIBRARY_SETS: dict[tuple[str, ...], LibrarySet] = {}
"""Library sets by the names a configuration gives them. A set that is not here is fetched
from the registry by those names (:meth:`albireo.eclipsing.LibrarySet.fetch`). A set built
in memory is registered under a name of its own, in every process that uses it."""


@functools.cache
def _context(libraries: tuple[str, ...], v_max: float, vsini_max: float) -> _Context:
    grid = rvs_model_grid(v_max, vsini_max_kms=vsini_max, length_multiple=512)
    return _Context(
        libraries=_LIBRARY_SETS.get(libraries) or LibrarySet.fetch(libraries),
        grid=grid,
        delivered_kms=rvs_delivered_sigma_kms(RVS_DR4_EPOCH, simulation_dv_kms=grid.dv_kms),
        detector_wave=rvs_detector_grid(RVS_DR4_EPOCH.detector_step),
    )


def _template_grid(ctx: _Context, dataset: Dataset, v_search: float) -> LogGrid:
    """The grid of the templates: three pixels per delivered line-spread sigma, with the
    margin of the velocity range searched. One grid serves every system of a process."""
    if v_search not in ctx.template_grids:
        ctx.template_grids[v_search] = LogGrid.covering(
            dataset,
            ctx.delivered_kms / 3.0,
            v_margin_kms=v_search + 60.0,
            lsf_sigma_kms=ctx.delivered_kms,
        )
    return ctx.template_grids[v_search]


def _seed(system: BinarySystem) -> int:
    return int(system.meta.get("seed", zlib.crc32(system.name.encode())))


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


def declare(system: BinarySystem, declaration: str, libraries: LibrarySet, config=None):
    """The templates and the light fractions an analysis declares for a system.

    Parameters
    ----------
    system
        The system.
    declaration
        One of :data:`DECLARATIONS`.
    libraries
        The library boxes; every label is moved into its box.
    config
        A :class:`SurveyConfig`, for the classification errors. Default: the defaults.

    Returns
    -------
    dict
        ``labels``: two label mappings. ``vsini``: two rotations in km/s. ``epsilon``: two
        limb-darkening coefficients. ``light``: the ``light`` option of
        :func:`albireo.todcor`, the injected fractions or ``"global"``. ``identical``:
        whether the two templates are the same spectrum, in which case the correlation
        does not tell the two stars apart.

    Notes
    -----
    ``classified`` draws its errors from a generator seeded by the system's seed, so the
    declared labels are the same in every variant: a temperature error of the larger of
    150 K and 3 percent, 0.2 dex in ``log g``, 0.15 dex in ``[M/H]`` common to both stars,
    and a log-normal factor of 20 percent on ``v sin i``. ``catalogue`` takes the
    temperature of the dwarf sequence at the pair's ``G - G_RP`` on a lattice of 500 K,
    ``log g = 4.5``, solar metallicity, and the rotation of a synchronised star of the
    main-sequence radius at that temperature.
    """
    config = SurveyConfig() if config is None else config
    if declaration not in DECLARATIONS:
        raise ValueError(f"unknown declaration {declaration!r}")
    stars = system.labels
    vsini = [float(system.vsini1), float(system.vsini2)]
    light: Any = "global"
    identical = False
    if declaration == "injected":
        total = float(sum(system.light_fractions))
        light = tuple(float(x) / total for x in system.light_fractions)
    elif declaration == "classified":
        d_teff, f_teff, d_logg, d_mh, f_vsini = config.classification
        rng = np.random.default_rng([_seed(system), 2])
        mh_error = rng.normal(0.0, d_mh)
        drawn = []
        for star in stars:
            sigma = max(d_teff, f_teff * star["teff"])
            drawn.append(
                {
                    "teff": star["teff"] + rng.normal(0.0, sigma),
                    "logg": star["logg"] + rng.normal(0.0, d_logg),
                    "mh": star["mh"] + mh_error,
                }
            )
        stars = drawn
        vsini = [float(v * np.exp(rng.normal(0.0, f_vsini))) for v in vsini]
    else:
        lo, hi = libraries.teff_range
        teff = 500.0 * round(float(_teff_from_colour(system.g_rp)) / 500.0)
        teff = float(np.clip(teff, 500.0 * math.ceil(lo / 500.0), 500.0 * math.floor(hi / 500.0)))
        sequence = MainSequence()
        radius = float(sequence.radius(sequence.mass_from_teff(teff)))
        one = {"teff": teff, "logg": 4.5, "mh": 0.0}
        stars = [one, dict(one)]
        vsini = [_SYNCHRONOUS_KMS * radius / system.period] * 2
        identical = True
    labels = []
    for star in stars:
        _, clamped, _ = libraries.labels(star["teff"], star["logg"], star["mh"])
        labels.append(clamped)
    return {
        "labels": labels,
        "vsini": vsini,
        "epsilon": [float(limb_darkening(star["teff"], "rvs")) for star in labels],
        "light": light,
        "identical": identical,
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _component(ctx: _Context, teff, logg, mh, vsini, epsilon, emission_ew: float = 0.0):
    """One star as a deviation spectrum on the model grid, from the box that contains it."""
    from albireo.operators import _rotational_kernel_numpy
    from albireo.simulate import library_component

    library, labels, _ = ctx.libraries.labels(teff, logg, mh)
    if emission_ew <= 0.0:
        return library_component(
            library, labels, ctx.grid, medium="vacuum", vsini_kms=float(vsini), epsilon=epsilon
        )
    # Chromospheric emission fills the line cores before the star's rotation broadens them.
    deviation = library_component(library, labels, ctx.grid, medium="vacuum", vsini_kms=0.0)
    wave = np.asarray(ctx.grid.wave)
    for centre, ratio in _CA_TRIPLET:
        profile = np.exp(-0.5 * ((wave - centre) / _CA_EMISSION_SIGMA) ** 2)
        deviation = deviation + emission_ew * ratio * profile / (
            _CA_EMISSION_SIGMA * math.sqrt(2.0 * math.pi)
        )
    if vsini > 0.0:
        kernel = _rotational_kernel_numpy(float(vsini) / ctx.grid.dv_kms, epsilon=epsilon)
        deviation = np.convolve(deviation, kernel, mode="same")
    return deviation


def _quality(ctx: _Context, deviation: np.ndarray, grid: LogGrid | None = None) -> float:
    """Velocity information of a spectrum per unit S/N: ``sqrt(sum (ds/dv)^2)`` in s/km.

    The spectrum is broadened to the nominal resolving power and sampled on the detector
    pixels of the band. The photon-limited error of its velocity, alone in the spectrum at
    a noise ``sigma`` per pixel, is ``sigma / quality`` (Connes 1985; Bouchy et al. 2001).
    """
    from albireo.operators import _gaussian_kernel_numpy

    grid = ctx.grid if grid is None else grid
    power = ctx.libraries.libraries[0].resolving_power
    kernel = _gaussian_kernel_numpy(quadrature_sigma_kms(power) / grid.dv_kms)
    smooth = np.convolve(deviation, kernel, mode="same")
    wave = ctx.detector_wave
    keep = (wave >= RVS_BAND[0]) & (wave <= RVS_BAND[1])
    flux = np.interp(wave, np.asarray(grid.wave), smooth)
    slope = np.gradient(flux, np.log(wave)) / C_KMS
    return float(np.sqrt(np.sum(slope[keep] ** 2)))


def _simulate(system: BinarySystem, config: SurveyConfig, ctx: _Context):
    """The epochs of one system under the configuration, and what was injected."""
    seed = _seed(system)
    if config.simulation == "recorded":
        from albireo.benchmark import simulate_system

        dataset, truth, own_grid, components = simulate_system(
            system, library=ctx.libraries.libraries[0], seed=seed
        )
        n = dataset.n_epochs
        bjd = np.array([epoch.bjd for epoch in dataset])
        pair = np.asarray(system.light_fractions, dtype=np.float64)
        injected = {
            "bjd": bjd,
            "fractions": np.repeat(pair[:, None], n, axis=1),
            "flux": np.ones(n),
            "in_eclipse": np.zeros(n, dtype=bool),
            "separation": np.full(n, np.nan),
            "zero_point": np.zeros(n),
            "velocity": np.asarray(truth.velocities),
            "quality": [_quality(ctx, c, own_grid) for c in components],
            "third": 0.0,
        }
        return dataset, truth, injected

    aux = np.random.default_rng([seed, 1])
    bjd = rvs_transit_times(system.n_transits, span_days=config.span_days, seed=seed)
    n = bjd.size
    zero = aux.normal(0.0, config.zero_point_kms, size=n)
    tertiary = system.meta.get("tertiary") if config.variant == "third-light" else None
    third = float(tertiary["light"]) if tertiary else 0.0
    geometry = eclipse_light(system, bjd, third_light=third)
    velocity = system.orbit().component_velocities(bjd)
    epsilon = system.meta.get("limb_darkening") or [
        float(limb_darkening(system.teff1, "rvs")),
        float(limb_darkening(system.teff2, "rvs")),
    ]
    emission = [0.0, 0.0]
    if config.variant == "activity":
        emission = [float(x) for x in system.meta.get("activity_ew", emission)]
    components = [
        _component(
            ctx, system.teff1, system.logg1, system.mh, system.vsini1, epsilon[0], emission[0]
        )
    ]
    quality = [_quality(ctx, components[0]), np.nan]
    if config.variant == "null":
        fractions = np.ones((1, n))
        flux = np.ones(n)
        shifts = velocity[:1] + zero
    else:
        components.append(
            _component(
                ctx, system.teff2, system.logg2, system.mh, system.vsini2, epsilon[1], emission[1]
            )
        )
        quality[1] = _quality(ctx, components[1])
        fractions, flux = geometry["fractions"], geometry["flux"]
        shifts = velocity + zero
        if tertiary:
            eps3 = float(limb_darkening(tertiary["teff"], "rvs"))
            components.append(
                _component(
                    ctx, tertiary["teff"], tertiary["logg"], system.mh, tertiary["vsini"], eps3
                )
            )
            shifts = np.vstack([shifts, system.gamma + float(tertiary["dv"]) + zero])
    grvs = system.grvs - 2.5 * np.log10(flux)
    response = {}
    if config.variant == "instrument":
        extra = np.random.default_rng([seed, 3])
        limit = math.log(config.background_factor)
        factor = np.exp(extra.uniform(-limit, limit, size=n))
        snr = np.array(
            [
                rvs_snr_per_pixel(
                    grvs[j],
                    constants=dataclasses.replace(
                        RVS_CONSTANTS, background_e=RVS_CONSTANTS.background_e * factor[j]
                    ),
                )
                for j in range(n)
            ]
        )
        response = {
            "response_order": config.response_order,
            "response_amplitude": config.response_amplitude,
        }
    else:
        snr = np.asarray(rvs_snr_per_pixel(grvs), dtype=np.float64)
    dataset, truth = simulate_rvs_dataset(
        components,
        ctx.grid,
        bjd=bjd,
        light_fractions=fractions,
        velocities=shifts,
        snr=snr,
        product=RVS_DR4_EPOCH,
        library_resolving_power=ctx.libraries.libraries[0].resolving_power,
        resolving_power=rvs_transit_resolving_powers(n, seed=seed),
        declare_lsf="nominal",
        seed=seed,
        **response,
    )
    pair = np.asarray(geometry["fractions"][:2], dtype=np.float64)
    if config.variant == "null":
        pair = np.vstack([np.ones(n), np.zeros(n)])
    injected = {
        "bjd": bjd,
        "fractions": pair,
        "flux": flux,
        "in_eclipse": np.asarray(geometry["in_eclipse"], dtype=bool),
        "separation": geometry["separation"],
        "zero_point": zero,
        "velocity": np.asarray(velocity),
        "quality": quality,
        "third": third,
    }
    return dataset, truth, injected


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

_EPOCH_SHARED = (
    "bjd phase snr in_eclipse flux l1_true l2_true v1_true v2_true zero_point resolving_power "
    "separation pred1 pred2"
).split()
_EPOCH_MEASURED = (
    "v1 v2 s1 s2 rho l1 l2 d1 d2 blended edge margin a1 a2 chi2 chi2_null n_pix second"
).split()
_EPOCH_ASSIGNED = "ve1 ve2 se1 se2 ge xe oe vp1 vp2 sp1 sp2 gp xp op vs ss gs".split()
_SYSTEM_SHARED = (
    "n_epochs n_in_eclipse snr_median k1_true k2_true gamma_true light1_true light2_true "
    "light3_true quality1 quality2 snr_k1 snr_k2 t_sim"
).split()
_SYSTEM_DECLARED = (
    "n_out n_good light1 light2 fallback swapped sum_d1 sum_d2 med_d2 t_todcor t_orbit "
    "e_ok e_k1 e_k2 e_k1_err e_k2_err e_gamma e_gamma_err e_ecc e_omega e_chi2 e_npts e_npar "
    "e_nx e_no e_held e_light1 e_light2 "
    "p_ok p_k1 p_k2 p_k1_err p_k2_err p_gamma p_gamma_err p_ecc p_period p_chi2 p_npts p_npar "
    "p_nx p_no p_held p_swapped "
    "s_ok s_k1 s_k1_err s_gamma s_gamma_err s_chi2 s_npts s_light"
).split()


def _correlate(dataset: Dataset, pair, light, config: SurveyConfig, ctx: _Context, lag1: float):
    """One call of the correlation with the settings of the survey; ``(table, fallback)``.

    Measured light fractions need an epoch at which both amplitudes are positive. Where
    there is none the epochs are measured with the amplitudes of each epoch, and the
    second value returned is true.
    """
    from albireo.todcor import todcor

    key = dataset.epochs[0].instrument
    options = {
        "v_range": (-config.v_search_kms, config.v_search_kms),
        "lsf_sigma_v": {key: ctx.delivered_kms},
        "noise_correlation": {key: lag1},
    }
    if config.variant == "instrument":
        options["scale"] = "free"
    try:
        return todcor(dataset, pair, light=light, **options), False
    except ValueError:
        if light != "global":
            raise
        return todcor(dataset, pair, light="free", **options), True


def _fill(columns: dict[str, np.ndarray], row: int, where: np.ndarray, table) -> None:
    """Write the columns of a velocity table into the epochs ``where`` of row ``row``."""
    variance = np.diagonal(table.covariance, axis1=1, axis2=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = table.covariance[:, 0, 1] / np.sqrt(variance[:, 0] * variance[:, 1])
    values = {
        "v1": table.velocity[0],
        "v2": table.velocity[1],
        "s1": table.sigma[0],
        "s2": table.sigma[1],
        "rho": rho,
        "l1": table.light[0],
        "l2": table.light[1],
        "d1": table.delta_chi2[0],
        "d2": table.delta_chi2[1],
        "blended": table.blended,
        "edge": np.any(table.at_edge, axis=0),
        "margin": table.margin,
        "a1": table.alternative[0],
        "a2": table.alternative[1],
        "chi2": table.chi2,
        "chi2_null": table.chi2_null,
        "n_pix": table.n_pixels,
        "second": table.second_minimum,
    }
    for name, value in values.items():
        columns[name][row, where] = value


def _exchange_all(table):
    from albireo.rvorbit import _exchanged

    return _exchanged(table, np.ones(table.n_epochs, dtype=bool))


def _better_order(table, v_true: np.ndarray) -> bool:
    """Whether the table matches the injected velocities better with its components
    interchanged as a whole, over its usable epochs."""
    use = table.good
    if not use.any():
        return False
    direct = np.mean((table.velocity[:, use] - v_true[:, use]) ** 2)
    other = np.mean((table.velocity[::-1, use] - v_true[:, use]) ** 2)
    return bool(other < direct)


def _orbit_record(prefix: str, orbit, decided, out: dict[str, float]) -> None:
    out[prefix + "ok"] = 1.0
    out[prefix + "k1"], out[prefix + "k2"] = float(orbit.k[0]), float(orbit.k[1])
    out[prefix + "k1_err"] = float(orbit.errors["k"][0])
    out[prefix + "k2_err"] = float(orbit.errors["k"][1])
    out[prefix + "gamma"] = float(orbit.gamma[0])
    out[prefix + "gamma_err"] = float(orbit.errors["gamma"][0])
    out[prefix + "ecc"] = float(orbit.ecc)
    out[prefix + "chi2"] = float(orbit.chi2)
    out[prefix + "npts"] = float(orbit.n_points)
    out[prefix + "npar"] = float(orbit.n_parameters)
    out[prefix + "nx"] = float(decided.exchanged.sum())
    out[prefix + "no"] = float(decided.alternative.sum())
    out[prefix + "held"] = float(len(orbit.held))


def run_system(system: BinarySystem, config: SurveyConfig | None = None) -> dict[str, Any]:
    """Simulate one system and measure it under every declaration of the configuration.

    The chain is:

    1. *Simulation.* The transit times follow the scanning law over ``span_days``. Each
       star is rendered from its library box with its rotation and limb darkening, the
       light fractions and the S/N of every transit follow from the eclipse geometry
       (:func:`albireo.eclipsing.eclipse_light`), a velocity offset common to both stars is
       added per transit, and each transit takes the resolving power of its CCD row.
    2. *Correlation.* For each declaration the templates are rendered on one grid and
       :func:`albireo.todcor` measures both velocities at every transit. Transits out of
       eclipse are measured with the declared light fractions, held or measured on those
       transits. Transits in eclipse, which the ephemeris identifies, are measured with
       the amplitudes of each epoch and do not enter the orbit.
    3. *Orbit.* :func:`albireo.rvorbit.assign_by_ephemeris` fits the semi-amplitudes and the
       systemic velocity with the period and the time of the primary's eclipse held, and
       the eccentricity fitted where the injected orbit is eccentric. Under the
       declarations of ``config.period_fit`` the table is also fitted by
       :func:`albireo.rvorbit.assign_components`, which knows the period alone.
    4. *Single-lined.* The transits out of eclipse are also measured with the first
       template alone, and one semi-amplitude is fitted at the ephemeris. A correlation
       with two templates flags an epoch whose second velocity is not determined, and
       with a companion too faint to measure it leaves few epochs for the primary.

    With identical templates (``catalogue``) the correlation does not name the stars. The
    ephemeris does: the table is fitted in both orders of its components and the order of
    the lower chi-square is kept. The fit with the period alone cannot tell the two orders
    apart, and its semi-amplitudes are compared with the injected ones in the order that
    fits (``p_swapped``), as for any pair of alike stars.

    Parameters
    ----------
    system
        The system. ``meta["seed"]`` seeds its simulation; without one the seed is a hash
        of the name.
    config
        The configuration; default :class:`SurveyConfig`.

    Returns
    -------
    dict
        ``name``; ``epochs``, a mapping of column name to an array of ``(n_epochs,)`` for
        the injected quantities and ``(n_declarations, n_epochs)`` for the measured ones;
        and ``system``, a mapping of column name to a scalar or an array of
        ``(n_declarations,)``. The columns are described in ``docs/api/survey.md``.
    """
    from albireo.rvorbit import (
        Assignment,
        _epoch_blocks,
        assign_by_ephemeris,
        assign_components,
        fit_rv_ephemeris,
    )
    from albireo.todcor import Template

    config = SurveyConfig() if config is None else config
    ctx = _context(tuple(config.libraries), float(config.v_max_kms), float(config.vsini_max_kms))
    recorded = config.simulation == "recorded"
    t0 = time.perf_counter()
    dataset, truth, injected = _simulate(system, config, ctx)
    t_sim = time.perf_counter() - t0
    n = dataset.n_epochs
    n_decl = len(config.declarations)
    bjd = injected["bjd"]
    v_true = injected["velocity"]
    in_eclipse = injected["in_eclipse"]
    out_of_eclipse = np.flatnonzero(~in_eclipse)
    lag1 = float(np.mean(truth.delivery.lag1))
    t_conj = system.t_conj
    grid = _template_grid(ctx, dataset, config.v_search_kms)
    library_power = ctx.libraries.libraries[0].resolving_power

    noise = np.array([float(np.median(sigma)) for sigma in truth.simulation.noise_sigma])
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = noise[None, :] / (
            injected["fractions"] * np.asarray(injected["quality"], dtype=np.float64)[:, None]
        )
    curve = (v_true[0] - system.gamma) / system.k1 if system.k1 > 0.0 else np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(in_eclipse[None, :], 0.0, (curve[None, :] / predicted) ** 2)
        snr_k = np.array([system.k1, system.k2]) * np.sqrt(np.nansum(weight, axis=1))

    epochs: dict[str, np.ndarray] = {
        "bjd": bjd,
        "phase": ((bjd - t_conj) / system.period) % 1.0,
        "snr": np.asarray(truth.snr),
        "in_eclipse": in_eclipse,
        "flux": injected["flux"],
        "l1_true": injected["fractions"][0],
        "l2_true": injected["fractions"][1],
        "v1_true": v_true[0],
        "v2_true": v_true[1],
        "zero_point": injected["zero_point"],
        "resolving_power": np.asarray(truth.resolving_power),
        "separation": injected["separation"],
        "pred1": predicted[0],
        "pred2": predicted[1],
    }
    for name in _EPOCH_MEASURED + _EPOCH_ASSIGNED:
        epochs[name] = np.full((n_decl, n), np.nan)
    record: dict[str, Any] = {
        "n_epochs": float(n),
        "n_in_eclipse": float(in_eclipse.sum()),
        "snr_median": float(np.median(truth.snr)),
        "k1_true": system.k1,
        "k2_true": system.k2,
        "gamma_true": system.gamma,
        "light1_true": float(np.median(injected["fractions"][0][~in_eclipse]))
        if out_of_eclipse.size
        else np.nan,
        "light2_true": float(np.median(injected["fractions"][1][~in_eclipse]))
        if out_of_eclipse.size
        else np.nan,
        "light3_true": float(injected["third"]),
        "quality1": float(injected["quality"][0]),
        "quality2": float(injected["quality"][1]),
        "snr_k1": float(snr_k[0]),
        "snr_k2": float(snr_k[1]),
        "t_sim": t_sim,
    }
    for name in _SYSTEM_DECLARED:
        record[name] = np.full(n_decl, np.nan)

    subsets = []
    for index in (out_of_eclipse, np.flatnonzero(in_eclipse)):
        subset = None
        if index.size:
            subset = Dataset(epochs=tuple(dataset.epochs[j] for j in index), frame=dataset.frame)
        subsets.append((index, subset))
    (index_out, data_out), (index_in, data_in) = subsets

    for row, declaration in enumerate(config.declarations):
        declared = declare(system, declaration, ctx.libraries, config)
        if recorded:
            declared["epsilon"] = [0.6, 0.6]
        pair = []
        for name, labels, vsini, epsilon in zip(
            ("A", "B"), declared["labels"], declared["vsini"], declared["epsilon"], strict=True
        ):
            library, labels, _ = ctx.libraries.labels(labels["teff"], labels["logg"], labels["mh"])
            pair.append(
                Template.from_library(
                    name,
                    library,
                    labels,
                    grid=grid,
                    medium="vacuum",
                    vsini_kms=float(vsini),
                    epsilon=epsilon,
                    resolving_power=library_power,
                )
            )
        t0 = time.perf_counter()
        table = None
        fallback = False
        if data_out is not None:
            table, fallback = _correlate(data_out, pair, declared["light"], config, ctx, lag1)
            _fill(epochs, row, index_out, table)
        if data_in is not None:
            eclipsed, _ = _correlate(data_in, pair, "free", config, ctx, lag1)
            _fill(epochs, row, index_in, eclipsed)
        # The first template alone, as an analysis that takes the system as single-lined.
        single = None
        if data_out is not None:
            single, _ = _correlate(data_out, pair[:1], "global", config, ctx, lag1)
            epochs["vs"][row, index_out] = single.velocity[0]
            epochs["ss"][row, index_out] = single.sigma[0]
            epochs["gs"][row, index_out] = single.good
        record["t_todcor"][row] = time.perf_counter() - t0
        record["n_out"][row] = float(index_out.size)
        record["fallback"][row] = float(fallback)
        if table is None:
            continue
        record["n_good"][row] = float(table.good.sum())
        with np.errstate(invalid="ignore"), contextlib.suppress(ValueError):
            record["light1"][row] = float(np.nanmedian(table.light[0]))
            record["light2"][row] = float(np.nanmedian(table.light[1]))
        use = table.good
        if use.any():
            record["sum_d1"][row] = float(np.sum(table.delta_chi2[0, use]))
            record["sum_d2"][row] = float(np.sum(table.delta_chi2[1, use]))
            record["med_d2"][row] = float(np.median(table.delta_chi2[1, use]))

        t0 = time.perf_counter()
        truth_out = v_true[:, index_out]
        record["s_ok"][row] = 0.0
        with contextlib.suppress(ValueError, np.linalg.LinAlgError):
            orbit = fit_rv_ephemeris(
                single,
                period=system.period,
                t_conj=t_conj,
                fit_eccentricity=system.ecc > 0.0 or recorded,
            )
            record["s_ok"][row] = 1.0
            record["s_k1"][row] = float(orbit.k[0])
            record["s_k1_err"][row] = float(orbit.errors["k"][0])
            record["s_gamma"][row] = float(orbit.gamma[0])
            record["s_gamma_err"][row] = float(orbit.errors["gamma"][0])
            record["s_chi2"][row] = float(orbit.chi2)
            record["s_npts"][row] = float(orbit.n_points)
            with np.errstate(invalid="ignore"):
                record["s_light"][row] = float(np.nanmedian(single.light[0]))
        # The orbit at the held ephemeris.
        candidates = [(False, table)]
        if declared["identical"]:
            candidates.append((True, _exchange_all(table)))
        best = None
        for swapped, candidate in candidates:
            try:
                fitted = assign_by_ephemeris(
                    candidate,
                    period=system.period,
                    t_conj=t_conj,
                    fit_eccentricity=system.ecc > 0.0 or recorded,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if best is None or fitted[1].chi2 < best[1][1].chi2:
                best = (swapped, fitted)
        record["e_ok"][row] = 0.0
        if best is not None:
            swapped, (assigned, orbit, decided) = best
            values: dict[str, float] = {}
            _orbit_record("e_", orbit, decided, values)
            values["e_omega"] = float(orbit.omega)
            for key, value in values.items():
                record[key][row] = value
            record["swapped"][row] = float(swapped)
            record["e_nx"][row] = float(np.sum(decided.exchanged ^ swapped))
            with np.errstate(invalid="ignore"), contextlib.suppress(ValueError):
                record["e_light1"][row] = float(np.nanmedian(assigned.light[0]))
                record["e_light2"][row] = float(np.nanmedian(assigned.light[1]))
            epochs["ve1"][row, index_out] = assigned.velocity[0]
            epochs["ve2"][row, index_out] = assigned.velocity[1]
            epochs["se1"][row, index_out] = assigned.sigma[0]
            epochs["se2"][row, index_out] = assigned.sigma[1]
            epochs["ge"][row, index_out] = assigned.good
            epochs["xe"][row, index_out] = decided.exchanged ^ swapped
            epochs["oe"][row, index_out] = decided.alternative
        # The orbit with the period known and the conjunction free.
        if declaration in config.period_fit:
            record["p_ok"][row] = 0.0
            try:
                # In blocks of epochs, or a process compiles a set of array programs for
                # every number of usable epochs. The recorded systems are fitted as their
                # runs fitted them.
                with _epoch_blocks() if not recorded else contextlib.nullcontext():
                    assigned, orbit, decided = assign_components(
                        table,
                        period=system.period,
                        circular=system.ecc == 0.0 and not recorded,
                    )
            except (ValueError, np.linalg.LinAlgError):
                assigned, orbit, decided = table, None, Assignment.none(table.n_epochs)
            other = _better_order(assigned, truth_out)
            if other:
                assigned = _exchange_all(assigned)
            if orbit is not None:
                values = {}
                _orbit_record("p_", orbit, decided, values)
                if other:
                    values["p_k1"], values["p_k2"] = values["p_k2"], values["p_k1"]
                    values["p_k1_err"], values["p_k2_err"] = values["p_k2_err"], values["p_k1_err"]
                values["p_period"] = float(orbit.period)
                for key, value in values.items():
                    record[key][row] = value
            record["p_swapped"][row] = float(other)
            epochs["vp1"][row, index_out] = assigned.velocity[0]
            epochs["vp2"][row, index_out] = assigned.velocity[1]
            epochs["sp1"][row, index_out] = assigned.sigma[0]
            epochs["sp2"][row, index_out] = assigned.sigma[1]
            epochs["gp"][row, index_out] = assigned.good
            epochs["xp"][row, index_out] = decided.exchanged ^ other
            epochs["op"][row, index_out] = decided.alternative
        record["t_orbit"][row] = time.perf_counter() - t0
    return {"name": system.name, "epochs": epochs, "system": record}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _tables(results: Sequence[dict[str, Any]], indices: Sequence[int], n_decl: int):
    """The results of a chunk as flat tables: one row per system and declaration, and one
    per epoch and declaration."""
    systems: dict[str, list] = {"system": [], "decl": []}
    epochs: dict[str, list] = {"system": [], "decl": [], "epoch": []}
    for index, result in zip(indices, results, strict=True):
        record, columns = result["system"], result["epochs"]
        n = int(record["n_epochs"])
        systems["system"].append(np.full(n_decl, index))
        systems["decl"].append(np.arange(n_decl))
        for name in _SYSTEM_SHARED:
            systems.setdefault(name, []).append(np.full(n_decl, record[name], dtype=np.float64))
        for name in _SYSTEM_DECLARED:
            systems.setdefault(name, []).append(np.asarray(record[name], dtype=np.float64))
        epochs["system"].append(np.full(n_decl * n, index))
        epochs["decl"].append(np.repeat(np.arange(n_decl), n))
        epochs["epoch"].append(np.tile(np.arange(n), n_decl))
        for name in _EPOCH_SHARED:
            epochs.setdefault(name, []).append(
                np.tile(np.asarray(columns[name], np.float64), n_decl)
            )
        for name in _EPOCH_MEASURED + _EPOCH_ASSIGNED:
            epochs.setdefault(name, []).append(np.asarray(columns[name], np.float64).reshape(-1))

    def join(table):
        out = {}
        for name, parts in table.items():
            array = np.concatenate(parts) if parts else np.zeros(0)
            if name in ("system", "decl", "epoch"):
                out[name] = array.astype(np.int32)
            elif name == "bjd":
                out[name] = array.astype(np.float64)
            else:
                out[name] = array.astype(np.float32)
        return out

    return join(systems), join(epochs)


def _write_chunk(path: Path, systems, epochs, failures) -> None:
    payload = {f"s_{key}": value for key, value in systems.items()}
    payload.update({f"e_{key}": value for key, value in epochs.items()})
    payload["failures"] = np.asarray(json.dumps(failures))
    partial = path.with_name(path.name + ".part")
    with open(partial, "wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(partial, path)


def _run_chunk(records: list[dict], indices: list[int], config: dict, path: str) -> dict[str, Any]:
    """Run the systems of one chunk in this process and write their tables."""
    cfg = SurveyConfig.from_dict(config)
    t0 = time.perf_counter()
    results, kept, failures = [], [], []
    for index, record in zip(indices, records, strict=True):
        system = BinarySystem.from_dict(record)
        try:
            results.append(run_system(system, cfg))
            kept.append(index)
        except Exception:  # one system must not lose the chunk
            failures.append({"system": index, "name": system.name, "error": traceback.format_exc()})
    systems, epochs = _tables(results, kept, len(cfg.declarations))
    _write_chunk(Path(path), systems, epochs, failures)
    return {
        "path": path,
        "n": len(kept),
        "failures": len(failures),
        "seconds": time.perf_counter() - t0,
        "rss_gb": _resident_gb(),
    }


def _resident_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return float("nan")


def _free_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:
        return float("inf")


def run_survey(
    systems: Sequence[BinarySystem],
    config: SurveyConfig,
    directory,
    *,
    workers: int = 8,
    chunk_size: int = 50,
    chunks_per_worker: int = 3,
    min_free_gb: float = 3.0,
    progress: bool = True,
) -> Path:
    """Run a population in worker processes, one file per chunk of systems.

    The population and the configuration are written to ``directory`` first. The systems
    are then divided into chunks in their order, and each chunk is run by one worker
    process, which writes ``chunks/chunk_<number>.npz`` when it is complete. A chunk whose
    file exists is not run again, so an interrupted run is continued by calling the
    function again with the same arguments.

    Every system is simulated from its own seed (``meta["seed"]``), so the tables do not
    depend on the number of workers or on the size of the chunks.

    Parameters
    ----------
    systems
        The population.
    config
        The configuration.
    directory
        Where the run is kept.
    workers
        Number of worker processes, each restricted to one thread. ``0`` runs the chunks
        in this process.
    chunk_size
        Systems per chunk.
    chunks_per_worker
        A worker is replaced after this many chunks. The orbit fit with a free conjunction
        compiles one array program for every number of usable epochs, and a process that
        runs a few hundred systems holds a few hundred of them.
    min_free_gb
        No chunk is started while less memory than this is available.
    progress
        Print one line per chunk.

    Returns
    -------
    pathlib.Path
        ``directory``.
    """
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from multiprocessing import get_context

    directory = Path(directory)
    (directory / "chunks").mkdir(parents=True, exist_ok=True)
    systems = list(systems)
    records = [s.to_dict() for s in systems]
    # What the chunks of this directory were made from. The names of a drawn population
    # are positional, so the systems are compared by their content.
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    identity = {
        "population_sha256": digest,
        "n_systems": len(systems),
        "chunk_size": int(chunk_size),
        "config": config.to_dict(),
    }
    record = directory / "run.json"
    if record.exists():
        kept = json.loads(record.read_text(encoding="utf-8"))
        for key, what in (
            ("population_sha256", "another population"),
            ("config", "another configuration"),
            ("chunk_size", "another chunk size"),
        ):
            if kept.get(key) != identity[key]:
                raise ValueError(
                    f"{directory} holds a run of {what}; continue it with the same "
                    "arguments or use another directory"
                )
    else:
        record.write_text(json.dumps(identity, indent=1), encoding="utf-8")
    population = directory / "population.json"
    if not population.exists():
        write_population(population, systems)
    (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=1), encoding="utf-8")
    chunks = []
    for number, start in enumerate(range(0, len(systems), int(chunk_size))):
        path = directory / "chunks" / f"chunk_{number:05d}.npz"
        if not path.exists():
            chunks.append((list(range(start, min(start + chunk_size, len(systems)))), path))
    done = 0
    t0 = time.perf_counter()

    def report(summary) -> None:
        nonlocal done
        done += 1
        if progress:
            print(
                f"[{time.perf_counter() - t0:7.0f} s] chunk {done}/{len(chunks)} "
                f"{Path(summary['path']).name}: {summary['n']} systems, "
                f"{summary['failures']} failed, {summary['seconds']:.0f} s, "
                f"{summary['rss_gb']:.2f} GB resident, {_free_gb():.1f} GB free",
                flush=True,
            )

    if workers <= 0:
        for indices, path in chunks:
            report(_run_chunk([records[i] for i in indices], indices, config.to_dict(), str(path)))
        return directory

    single = {
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    saved = {key: os.environ.get(key) for key in single}
    os.environ.update(single)
    try:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=get_context("spawn"),
            max_tasks_per_child=int(chunks_per_worker),
        ) as pool:
            pending = set()
            queue = list(chunks)
            while queue or pending:
                while queue and len(pending) < workers:
                    if pending and _free_gb() < min_free_gb:
                        break
                    indices, path = queue.pop(0)
                    pending.add(
                        pool.submit(
                            _run_chunk,
                            [records[i] for i in indices],
                            indices,
                            config.to_dict(),
                            str(path),
                        )
                    )
                finished, pending = wait(pending, timeout=30.0, return_when=FIRST_COMPLETED)
                for future in finished:
                    report(future.result())
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return directory


def _machine() -> str:
    cpu = platform.processor() or platform.machine()
    return f"{platform.system()} {platform.release()}, {cpu}, {os.cpu_count()} threads"


def collect(directory) -> dict[str, Any]:
    """Join the chunks of a run into its two tables and write its manifest.

    Writes ``systems.npz`` and ``epochs.npz`` (the columns as arrays, sorted by system,
    declaration and epoch) and ``manifest.json`` (the package version and commit, the
    machine, the configuration, the number of systems and epochs, the failures and the
    SHA-256 of the two tables) into ``directory``.

    Returns
    -------
    dict
        The manifest.
    """
    from albireo import __version__

    directory = Path(directory)
    paths = sorted((directory / "chunks").glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no chunk in {directory / 'chunks'}")
    tables: dict[str, dict[str, list]] = {"s": {}, "e": {}}
    failures = []
    for path in paths:
        with np.load(path) as data:
            failures += json.loads(str(data["failures"]))
            for key in data.files:
                if key[:2] in ("s_", "e_"):
                    tables[key[0]].setdefault(key[2:], []).append(data[key])
    joined = {
        kind: {name: np.concatenate(parts) for name, parts in table.items()}
        for kind, table in tables.items()
    }
    order = np.lexsort((joined["s"]["decl"], joined["s"]["system"]))
    systems = {name: value[order] for name, value in joined["s"].items()}
    order = np.lexsort((joined["e"]["epoch"], joined["e"]["decl"], joined["e"]["system"]))
    epochs = {name: value[order] for name, value in joined["e"].items()}
    checksums = {}
    for name, table in (("systems", systems), ("epochs", epochs)):
        target = directory / f"{name}.npz"
        np.savez_compressed(target, **table)
        digest = hashlib.sha256()
        for key in sorted(table):
            if key.startswith("t_"):
                continue  # wall-clock times differ between runs
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(table[key]).tobytes())
        checksums[name] = digest.hexdigest()
    commit, uncommitted = "unknown", None
    with contextlib.suppress(Exception):
        import subprocess

        def git(*arguments: str) -> str:
            return subprocess.run(
                ["git", *arguments],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).parent,
            ).stdout.strip()

        commit = git("rev-parse", "HEAD")
        # Changes to the package or the scripts that the commit does not hold.
        uncommitted = bool(git("status", "--porcelain", "--", "../../src", "../../scripts"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    manifest = {
        "albireo": __version__,
        "commit": commit,
        "uncommitted_changes": uncommitted,
        "created": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "machine": _machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "config": config,
        "declarations": list(config["declarations"]),
        "n_systems": int(np.unique(systems["system"]).size),
        "n_system_rows": int(systems["system"].size),
        "n_epoch_rows": int(epochs["system"].size),
        "failures": failures,
        "content_sha256": checksums,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def read_tables(directory) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The two tables of a collected run: ``(systems, epochs)``, each a mapping of column
    name to array."""
    directory = Path(directory)
    out = []
    for name in ("systems", "epochs"):
        with np.load(directory / f"{name}.npz") as data:
            out.append({key: data[key] for key in data.files})
    return out[0], out[1]
