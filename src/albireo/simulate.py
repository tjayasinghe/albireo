"""Synthetic spectroscopic-binary datasets, the test harness for the inference code.

Milestone M1 (``internal/design.md`` §8). :func:`simulate_dataset` generates composite epochs
through the same operator stack the inference code uses: shift, LSF convolution, rebin to
the native grid, then multiplicative response. Closed-loop tests therefore exercise the
forward model itself, under the pathologies the model claims to handle: chip gaps, cosmic
hits, mixed instruments and resolutions, tellurics, nebular emission with a per-epoch
amplitude, barycentric frames, and per-epoch light fractions.

Component spectra are deviation spectra ``d = s - 1`` on the model
:class:`~albireo.grids.LogGrid`, zero in the continuum and negative in absorption. Frame
conventions follow ``docs/math.md`` §1.2: with ``frame="topocentric"`` the stellar
log-shift is ``xi(v_star) - xi(v_bary)`` and tellurics are static; with
``frame="barycentric"`` the stellar shift is ``xi(v_star)`` and tellurics move by
``+xi(v_bary)``.

:func:`resimulate` redraws the data of an existing :class:`~albireo.forward.Problem` from
its own forward model, which is the parametric bootstrap :mod:`albireo.calibrate` runs.
:func:`synthetic_library` builds a small spectral library standing in for a published
synthetic grid, and :func:`library_component` renders one of its nodes as a component
spectrum, so the label-fitting and template modes can be exercised offline.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from albireo.data import Dataset, EpochData
from albireo.grids import LogGrid
from albireo.kepler import radial_velocity
from albireo.operators import (
    convolve_spectrum,
    convolve_varying,
    gaussian_kernel,
    gaussian_lsf_profiles,
    rebin_operator,
    shift_spectrum,
)

if TYPE_CHECKING:  # albireo.forward imports chebyshev_response from here, so the
    from albireo.forward import Problem  # runtime import in resimulate must be local

__all__ = [
    "InstrumentSpec",
    "OrbitParams",
    "SimulationTruth",
    "chebyshev_response",
    "library_component",
    "resimulate",
    "simulate_dataset",
    "synthetic_deviation_spectrum",
    "synthetic_library",
    "synthetic_nebular_spectrum",
    "synthetic_telluric_spectrum",
]


@dataclasses.dataclass(frozen=True)
class InstrumentSpec:
    """A simulated instrument: native wavelength grid, Gaussian LSF width, and SNR.

    Attributes
    ----------
    wave
        Native wavelength grid (Å), strictly increasing; must lie inside the model grid.
    sigma_v_lsf
        Gaussian LSF width in km/s: a scalar for a stationary LSF, or, together with
        ``lsf_anchors_angstrom``, one width per anchor for a wavelength-dependent LSF
        (D37), linearly interpolated across the grid exactly as the forward model
        realizes it (:func:`albireo.operators.gaussian_lsf_profiles`).
    snr
        Per-pixel continuum signal-to-noise (noise sigma = 1/snr on normalized flux).
    lsf_anchors_angstrom
        Optional anchor wavelengths (strictly increasing, >= 2) for a
        wavelength-dependent LSF; None keeps the stationary one.
    lsf_h3
        Optional Gauss-Hermite skewness (D38): a scalar or one value per anchor,
        anchored instruments only; None keeps pure Gaussian profiles.
    """

    wave: np.ndarray
    sigma_v_lsf: float | Sequence[float]
    snr: float
    lsf_anchors_angstrom: tuple[float, ...] | None = None
    lsf_h3: float | Sequence[float] | None = None


@dataclasses.dataclass(frozen=True)
class OrbitParams:
    """Keplerian orbit for the simulator; components 2, 4, ... use ``omega + pi``.

    Attributes
    ----------
    period
        Orbital period [d].
    t_peri
        Time of periastron passage [d].
    ecc
        Eccentricity.
    omega
        Argument of periastron of component 1 [rad].
    k
        One radial-velocity semi-amplitude [km/s] per stellar component.
    gamma
        Systemic velocity [km/s].
    """

    period: float
    t_peri: float
    ecc: float
    omega: float  # argument of periastron of component 1 [rad]
    k: tuple[float, ...]  # (K_1, K_2, ...) one semi-amplitude per stellar component
    gamma: float = 0.0

    def component_velocities(self, bjd: np.ndarray) -> np.ndarray:
        """Radial velocities, shape ``(n_components, n_epochs)``, barycentric frame."""
        rows = []
        for i, k_i in enumerate(self.k):
            omega_i = self.omega + (i % 2) * np.pi  # secondary opposes primary
            rows.append(
                np.asarray(
                    radial_velocity(
                        jnp.asarray(bjd),
                        period=self.period,
                        t_peri=self.t_peri,
                        ecc=self.ecc,
                        omega=omega_i,
                        k=k_i,
                        gamma=self.gamma,
                    )
                )
            )
        return np.stack(rows)


@dataclasses.dataclass(frozen=True)
class SimulationTruth:
    """Everything :func:`simulate_dataset` injected: the reference for closed-loop tests.

    Component, telluric and nebular spectra are deviation spectra on the model grid, and
    the stellar velocities are recorded in the barycentric frame whatever frame the
    returned :class:`~albireo.data.Dataset` declares.
    """

    grid: LogGrid
    components: tuple[np.ndarray, ...]  # deviation spectra on the model grid
    telluric: np.ndarray | None
    orbit: OrbitParams | None
    velocities: np.ndarray  # (n_comp, n_ep) stellar RVs, barycentric frame
    v_bary: np.ndarray  # (n_ep,)
    light_fractions: np.ndarray  # (n_comp, n_ep)
    response_coeffs: tuple[np.ndarray, ...]  # per-epoch Chebyshev coefficients
    noiseless_flux: tuple[np.ndarray, ...]  # per-epoch native-grid flux, pre-noise
    epoch_instruments: tuple[str, ...]
    ar1_phi: float = 0.0  # AR(1) correlation of the injected noise (0 = white)
    nebular: np.ndarray | None = None  # injected nebular deviation spectrum
    nebular_amplitudes: np.ndarray | None = None  # (n_ep,) as injected, un-normalized
    nebular_v_kms: float = 0.0


def synthetic_deviation_spectrum(
    grid: LogGrid,
    *,
    n_lines: int = 40,
    depth_range: tuple[float, float] = (0.05, 0.7),
    sigma_v_range: tuple[float, float] = (4.0, 20.0),
    margin: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Random absorption-line deviation spectrum (Gaussian lines in velocity space).

    Line centers are kept ``margin`` (fraction of the grid) away from the edges so that
    shifted/convolved spectra stay clear of the zero-padded boundary. The result is
    clipped at -0.95 so the corresponding flux stays positive.
    """
    rng = np.random.default_rng(seed)
    px = np.arange(grid.n, dtype=np.float64)
    centers = rng.uniform(margin * grid.n, (1.0 - margin) * grid.n, size=n_lines)
    depths = rng.uniform(*depth_range, size=n_lines)
    sigmas = rng.uniform(*sigma_v_range, size=n_lines) / grid.dv_kms
    d = np.zeros(grid.n)
    for c, a, s in zip(centers, depths, sigmas, strict=True):
        d -= a * np.exp(-0.5 * ((px - c) / s) ** 2)
    return np.maximum(d, -0.95)


def synthetic_telluric_spectrum(
    grid: LogGrid,
    *,
    n_bands: int = 3,
    lines_per_band: int = 15,
    depth_range: tuple[float, float] = (0.02, 0.85),
    sigma_v_range: tuple[float, float] = (1.5, 4.0),
    margin: float = 0.05,
    seed: int = 1,
) -> np.ndarray:
    """Telluric-like deviation spectrum: narrow lines clustered in a few bands."""
    rng = np.random.default_rng(seed)
    px = np.arange(grid.n, dtype=np.float64)
    lo, hi = margin * grid.n, (1.0 - margin) * grid.n
    band_centers = rng.uniform(lo, hi, size=n_bands)
    band_width = 0.05 * grid.n
    d = np.zeros(grid.n)
    for bc in band_centers:
        centers = np.clip(rng.normal(bc, band_width, size=lines_per_band), lo, hi)
        depths = rng.uniform(*depth_range, size=lines_per_band)
        sigmas = rng.uniform(*sigma_v_range, size=lines_per_band) / grid.dv_kms
        for c, a, s in zip(centers, depths, sigmas, strict=True):
            d -= a * np.exp(-0.5 * ((px - c) / s) ** 2)
    return np.maximum(d, -0.95)


def synthetic_nebular_spectrum(
    grid: LogGrid,
    *,
    lines: Sequence[float] | None = None,
    amplitude_range: tuple[float, float] = (0.1, 0.8),
    sigma_v_kms: float = 12.0,
    v_kms: float = 0.0,
    margin: float = 0.02,
    seed: int = 2,
) -> np.ndarray:
    """Nebular emission deviation spectrum: narrow positive lines at fixed wavelengths.

    The line positions are physical rather than random, unlike those of the stellar and
    telluric generators. A nebular component contaminates the stellar features its lines
    coincide with (Balmer, He I), so randomly placed lines would not reproduce the effect
    the component exists to describe.

    Parameters
    ----------
    grid
        Model grid; lines outside it (or within ``margin`` of an edge) are dropped.
    lines
        Rest wavelengths in air angstrom; default :data:`albireo.priors.NEBULAR_LINES`.
    amplitude_range
        Uniform range for each line's peak deviation (positive = emission).
    sigma_v_kms
        Intrinsic Gaussian width [km/s]. Nebular lines are thermally narrow, about
        10 km/s at 10⁴ K including turbulence; the observed width is set by the
        instrument, which the simulator applies downstream.
    v_kms
        Velocity of the nebula in the model grid's frame; must match the
        ``nebular_v_kms`` the fit is built with.
    margin
        Fraction of the grid kept clear at each edge.
    seed
        Seed for the line amplitudes.

    Returns
    -------
    numpy.ndarray
        ``(grid.n,)`` non-negative deviation spectrum.

    Raises
    ------
    ValueError
        If no line falls inside the grid with the requested edge margin.
    """
    from albireo.grids import C_KMS
    from albireo.priors import NEBULAR_LINES

    if lines is None:
        lines = list(NEBULAR_LINES.values())
    rng = np.random.default_rng(seed)
    px = np.arange(grid.n, dtype=np.float64)
    lo, hi = margin * grid.n, (1.0 - margin) * grid.n
    wave = np.asarray(grid.wave, dtype=np.float64)
    sigma_px = float(sigma_v_kms) / grid.dv_kms
    d = np.zeros(grid.n)
    used = 0
    for lam in sorted(float(x) for x in lines):
        lam_obs = lam * (1.0 + float(v_kms) / C_KMS)
        if not (wave[0] <= lam_obs <= wave[-1]):
            continue
        center = float(np.interp(lam_obs, wave, px))
        if not (lo <= center <= hi):
            continue
        d += rng.uniform(*amplitude_range) * np.exp(-0.5 * ((px - center) / sigma_px) ** 2)
        used += 1
    if used == 0:
        raise ValueError(
            f"none of the {len(list(lines))} nebular lines fall inside the model grid "
            f"({wave[0]:.2f}-{wave[-1]:.2f} A) with a {margin:.0%} edge margin"
        )
    return d


def chebyshev_response(wave: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Multiplicative response ``r = 1 + sum_m c_m T_m(x)``, ``x`` scaled to [-1, 1]."""
    wave = np.asarray(wave, dtype=np.float64)
    coeffs = np.atleast_1d(np.asarray(coeffs, dtype=np.float64))
    if coeffs.size == 0:
        return np.ones_like(wave)
    x = 2.0 * (wave - wave[0]) / (wave[-1] - wave[0]) - 1.0
    return 1.0 + np.polynomial.chebyshev.chebval(x, coeffs)


def _light_fraction_matrix(light_fractions, n_comp: int, n_ep: int) -> np.ndarray:
    ell = np.asarray(light_fractions, dtype=np.float64)
    if ell.ndim == 1:
        ell = np.repeat(ell[:, None], n_ep, axis=1)
    if ell.shape != (n_comp, n_ep):
        raise ValueError(
            f"light_fractions must have shape ({n_comp},) or ({n_comp}, {n_ep}); got {ell.shape}"
        )
    if np.any(ell < 0):
        raise ValueError("light fractions must be non-negative")
    if not np.allclose(ell.sum(axis=0), 1.0, atol=1e-10):
        raise ValueError("light fractions must sum to 1 at every epoch")
    return ell


def simulate_dataset(
    grid: LogGrid,
    components: Sequence[np.ndarray],
    *,
    bjd: np.ndarray,
    instruments: Mapping[str, InstrumentSpec],
    light_fractions,
    orbit: OrbitParams | None = None,
    velocities: np.ndarray | None = None,
    epoch_instruments: Sequence[str] | None = None,
    v_bary: np.ndarray | None = None,
    frame: str = "topocentric",
    telluric: np.ndarray | None = None,
    nebular: np.ndarray | None = None,
    nebular_amplitudes=None,
    nebular_v_kms: float = 0.0,
    response_order: int = 0,
    response_amplitude: float = 0.0,
    ar1_phi: float = 0.0,
    gap_fraction: float = 0.0,
    cosmic_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[Dataset, SimulationTruth]:
    """Generate a synthetic multi-epoch dataset plus the injected truth.

    Each epoch is built through the operator stack of ``docs/math.md`` §1: the component
    deviation spectra are shifted to their velocities and summed with the light fractions
    of that epoch, the telluric and nebular spectra are added at their own shifts, the sum
    is convolved with the instrument LSF, rebinned onto the instrument's native
    wavelength grid, multiplied by the epoch's Chebyshev response, and finally given
    white or AR(1) noise. Chip gaps and cosmic hits are then applied by zeroing ``ivar``.

    Parameters
    ----------
    grid
        Model log-wavelength grid.
    components
        Stellar deviation spectra on ``grid`` (one array of shape ``(grid.n,)`` per
        component).
    bjd
        Epoch times [d], shape ``(n_ep,)``.
    instruments
        Named instrument specs; each epoch is observed with one of them.
    light_fractions
        ``(n_comp,)`` constant or ``(n_comp, n_ep)`` per-epoch; must sum to 1 per epoch.
    orbit, velocities
        Either a Keplerian :class:`OrbitParams` (with ``len(orbit.k) == n_comp``) or an
        explicit ``(n_comp, n_ep)`` velocity array [km/s] (free-velocity mode). Exactly
        one must be given.
    epoch_instruments
        Instrument name per epoch; defaults to the first instrument for all epochs.
    v_bary
        Barycentric correction velocities per epoch [km/s]; default: uniform random in
        ±30 km/s.
    frame
        ``"topocentric"`` (default) or ``"barycentric"``: the frame of the emitted
        wavelength grids, with the shift conventions of ``docs/math.md`` §1.2.
    telluric
        Optional telluric deviation spectrum on ``grid`` (light fraction 1, additive).
    nebular
        Optional nebular deviation spectrum on ``grid``
        (:func:`synthetic_nebular_spectrum`): additive, static in the barycentric frame,
        and scaled per epoch by ``nebular_amplitudes``. This is the D40 component.
    nebular_amplitudes
        Per-epoch amplitude of the nebular component, ``(n_ep,)`` or a scalar
        (default 1), representing the seeing and slit-loss variation the component
        absorbs. Recovered amplitudes match these only up to one overall scale (see
        :func:`albireo.forward.with_nebular_amplitudes`).
    nebular_v_kms
        Velocity of the nebula [km/s] in the model grid's frame, matching
        :func:`albireo.forward.build_problem`.
    response_order, response_amplitude
        Per-epoch multiplicative Chebyshev response ``1 + sum_{m<=order} c_m T_m`` with
        ``c_m ~ N(0, response_amplitude^2)``. Amplitude 0 disables it (r = 1).
    ar1_phi
        AR(1) correlation of the pixel noise (``|phi| < 1``): the noise is a stationary
        AR(1) process over the native pixel index with marginal standard deviation
        ``1/snr``. This is the model of :func:`albireo.forward.with_ar1`, so closed-loop
        tests can inject and recover it. The process runs over all pixels, masked ones
        included, which is what makes the observed subset carry ``phi**gap`` correlations
        across masked gaps. 0 = white noise (default).
    gap_fraction
        Fraction of each epoch's pixels lost to one contiguous chip gap. The gap is given
        ivar = 0 and its flux is overwritten with unusable values, so downstream code must
        honor the mask.
    cosmic_fraction
        Fraction of pixels hit by cosmics (large positive spikes, ivar = 0).
    seed
        Seed for all randomness (noise, v_bary, response, gaps, cosmics).

    Returns
    -------
    (Dataset, SimulationTruth)
        The simulated dataset, tagged with ``frame``, and the injected truth.

    Raises
    ------
    ValueError
        If the shapes, the light fractions, the frame or the instrument grids are
        inconsistent, or if neither or both of ``orbit`` and ``velocities`` are given.
    """
    rng = np.random.default_rng(seed)
    components = tuple(np.asarray(c, dtype=np.float64) for c in components)
    n_comp = len(components)
    if n_comp == 0:
        raise ValueError("need at least one component")
    for c in components:
        if c.shape != (grid.n,):
            raise ValueError(f"each component must have shape ({grid.n},); got {c.shape}")
    bjd = np.asarray(bjd, dtype=np.float64)
    n_ep = bjd.size

    if (orbit is None) == (velocities is None):
        raise ValueError("provide exactly one of orbit= or velocities=")
    if orbit is not None:
        if len(orbit.k) != n_comp:
            raise ValueError(f"orbit has {len(orbit.k)} semi-amplitudes for {n_comp} components")
        vel = orbit.component_velocities(bjd)
    else:
        vel = np.asarray(velocities, dtype=np.float64)
        if vel.shape != (n_comp, n_ep):
            raise ValueError(f"velocities must have shape ({n_comp}, {n_ep}); got {vel.shape}")

    ell = _light_fraction_matrix(light_fractions, n_comp, n_ep)

    if frame not in ("topocentric", "barycentric"):
        raise ValueError(f"unknown frame {frame!r}")
    if v_bary is None:
        v_bary = rng.uniform(-30.0, 30.0, size=n_ep)
    v_bary = np.asarray(v_bary, dtype=np.float64)
    if v_bary.shape != (n_ep,):
        raise ValueError(f"v_bary must have shape ({n_ep},); got {v_bary.shape}")

    if epoch_instruments is None:
        epoch_instruments = [next(iter(instruments))] * n_ep
    epoch_instruments = tuple(epoch_instruments)
    if len(epoch_instruments) != n_ep:
        raise ValueError("epoch_instruments must have one entry per epoch")

    # Per-instrument static operators, built once.
    rebin_ops, kernels = {}, {}
    for name, spec in instruments.items():
        op = rebin_operator(x_in=grid.wave, x_out=np.asarray(spec.wave, dtype=np.float64))
        if np.any(np.asarray(op.coverage) < 1.0 - 1e-10):
            raise ValueError(f"instrument {name!r} wavelength grid extends beyond the model grid")
        rebin_ops[name] = op
        if spec.lsf_anchors_angstrom is not None:
            sig = np.atleast_1d(np.asarray(spec.sigma_v_lsf, dtype=np.float64))
            if sig.size == 1:
                sig = np.full(len(spec.lsf_anchors_angstrom), sig[0])
            h3_arr = None
            if spec.lsf_h3 is not None:
                h3_arr = np.atleast_1d(np.asarray(spec.lsf_h3, dtype=np.float64))
                if h3_arr.size == 1:
                    h3_arr = np.full(len(spec.lsf_anchors_angstrom), h3_arr[0])
            kernels[name] = jnp.asarray(
                gaussian_lsf_profiles(
                    sig / grid.dv_kms, spec.lsf_anchors_angstrom, grid.wave, h3=h3_arr
                )
            )
        else:
            if not isinstance(spec.sigma_v_lsf, (int, float)):
                raise ValueError(
                    f"instrument {name!r}: per-anchor LSF widths need lsf_anchors_angstrom"
                )
            if spec.lsf_h3 is not None:
                raise ValueError(f"instrument {name!r}: lsf_h3 needs lsf_anchors_angstrom")
            kernels[name] = gaussian_kernel(spec.sigma_v_lsf / grid.dv_kms)

    if telluric is not None:
        telluric = np.asarray(telluric, dtype=np.float64)
        if telluric.shape != (grid.n,):
            raise ValueError(f"telluric must have shape ({grid.n},); got {telluric.shape}")

    neb_amp = None
    if nebular is not None:
        nebular = np.asarray(nebular, dtype=np.float64)
        if nebular.shape != (grid.n,):
            raise ValueError(f"nebular must have shape ({grid.n},); got {nebular.shape}")
        if nebular_amplitudes is None:
            neb_amp = np.ones(n_ep)
        else:
            neb_amp = np.atleast_1d(np.asarray(nebular_amplitudes, dtype=np.float64))
            if neb_amp.size == 1:
                neb_amp = np.full(n_ep, float(neb_amp[0]))
        if neb_amp.shape != (n_ep,):
            raise ValueError(
                f"nebular_amplitudes must be a scalar or have shape ({n_ep},); got {neb_amp.shape}"
            )
    elif nebular_amplitudes is not None:
        raise ValueError("nebular_amplitudes given without a nebular spectrum")

    bary_pix = np.asarray(grid.velocity_to_pixels(v_bary))
    star_pix = np.asarray(grid.velocity_to_pixels(vel))  # (n_comp, n_ep)
    neb_pix = np.full(n_ep, float(np.asarray(grid.velocity_to_pixels(float(nebular_v_kms)))))
    if frame == "topocentric":
        star_pix = star_pix - bary_pix[None, :]
        tell_pix = np.zeros(n_ep)
        neb_pix = neb_pix - bary_pix  # static in the barycentric frame, so it moves here
    else:
        tell_pix = bary_pix

    epochs = []
    response_coeffs = []
    noiseless_fluxes = []
    for j in range(n_ep):
        spec = instruments[epoch_instruments[j]]
        wave_native = np.asarray(spec.wave, dtype=np.float64)
        n_native = wave_native.size

        d_total = jnp.zeros(grid.n)
        for i in range(n_comp):
            d_total = d_total + ell[i, j] * shift_spectrum(components[i], star_pix[i, j])
        if telluric is not None:
            d_total = d_total + shift_spectrum(telluric, tell_pix[j])
        if nebular is not None and neb_amp is not None:
            d_total = d_total + neb_amp[j] * shift_spectrum(nebular, neb_pix[j])
        kern = kernels[epoch_instruments[j]]
        d_total = (
            convolve_varying(d_total, kern) if kern.ndim == 2 else convolve_spectrum(d_total, kern)
        )
        flux_native = np.asarray(rebin_ops[epoch_instruments[j]](1.0 + d_total))

        if response_amplitude > 0:
            coeffs = rng.normal(0.0, response_amplitude, size=response_order + 1)
        else:
            coeffs = np.zeros(0)
        response_coeffs.append(coeffs)
        noiseless = chebyshev_response(wave_native, coeffs) * flux_native
        noiseless_fluxes.append(noiseless)

        sigma = 1.0 / spec.snr
        if ar1_phi != 0.0:
            if not -1.0 < ar1_phi < 1.0:
                raise ValueError(f"ar1_phi must lie in (-1, 1); got {ar1_phi}")
            eps = rng.normal(0.0, 1.0, size=n_native)
            noise = np.empty(n_native)
            noise[0] = eps[0]  # stationary start: marginal sd 1 everywhere
            innov = np.sqrt(1.0 - ar1_phi**2)
            for i in range(1, n_native):
                noise[i] = ar1_phi * noise[i - 1] + innov * eps[i]
            flux = noiseless + sigma * noise
        else:
            flux = noiseless + rng.normal(0.0, sigma, size=n_native)
        ivar = np.full(n_native, spec.snr**2, dtype=np.float64)

        if gap_fraction > 0:
            width = max(1, round(gap_fraction * n_native))
            start = rng.integers(0, n_native - width + 1)
            gap = slice(start, start + width)
            ivar[gap] = 0.0
            flux[gap] = rng.normal(0.0, 10.0, size=width)  # unusable: mask must be honored
        if cosmic_fraction > 0:
            n_hit = max(1, round(cosmic_fraction * n_native))
            hits = rng.choice(n_native, size=n_hit, replace=False)
            ivar[hits] = 0.0
            flux[hits] += rng.uniform(5.0, 50.0, size=n_hit)

        epochs.append(
            EpochData(
                wave=wave_native,
                flux=flux,
                ivar=ivar,
                bjd=float(bjd[j]),
                v_bary=float(v_bary[j]),
                instrument=epoch_instruments[j],
            )
        )

    truth = SimulationTruth(
        grid=grid,
        components=components,
        telluric=telluric,
        orbit=orbit,
        velocities=vel,
        v_bary=v_bary,
        light_fractions=ell,
        response_coeffs=tuple(response_coeffs),
        noiseless_flux=tuple(noiseless_fluxes),
        epoch_instruments=epoch_instruments,
        ar1_phi=float(ar1_phi),
        nebular=nebular,
        nebular_amplitudes=neb_amp,
        nebular_v_kms=float(nebular_v_kms),
    )
    return Dataset(epochs=tuple(epochs), frame=frame), truth


@functools.cache
def _model_applier():
    """Jitted :func:`albireo.forward.apply_model`, built once and reused.

    :func:`resimulate` is called once per bootstrap trial with the same problem structure,
    so the forward apply is one compiled graph rather than a few hundred eagerly
    dispatched operations per call. The cache is at module level because a fresh
    ``jax.jit`` wrapper per call would recompile on every trial. The import is local; see
    the ``TYPE_CHECKING`` note at the top of this module.
    """
    from albireo.forward import apply_model

    return jax.jit(apply_model)


def _ar1_noise(rng, n: int, phi: float) -> np.ndarray:
    """Stationary AR(1) process with unit marginal standard deviation."""
    eps = rng.normal(0.0, 1.0, size=n)
    if phi == 0.0:
        return eps
    out = np.empty(n)
    out[0] = eps[0]
    innov = np.sqrt(1.0 - phi**2)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + innov * eps[i]
    return out


def resimulate(problem: Problem, d_stack, *, seed: int = 0) -> Problem:
    """Redraw ``problem``'s data from its own forward model: a parametric bootstrap.

    The observed dataset fixes structure that a from-scratch simulation would have to
    assume: which epochs exist and when, each one's barycentric velocity and
    signal-to-noise, where the chip gaps and cosmics fell, the native wavelength
    solutions, and the response. All of that already lives in ``problem``, so a matched
    trial dataset is one forward apply plus a noise draw:

        ``z' = r (R B sum_i l_ij T(delta_ij) d_i) + n,   n ~ N(0, W^-1)``

    with the same weights, masks and operators; only the noise and the injected spectra
    differ. :mod:`albireo.calibrate` runs this in its inner loop, which is why an
    injection-recovery calibration on real data costs scan time rather than build time
    (:func:`albireo.forward.with_data`).

    The velocities, light fractions, LSF and response are read from ``problem`` as passed,
    so injecting at the truth requires a problem already evaluated there
    (:meth:`albireo.inference.MarginalOrbitModel.problem_at`). The returned problem keeps
    those velocities; moving the data onto whatever base problem the analysis then uses is
    left to the caller.

    Noise follows the problem's own model: standard deviation ``1/sqrt(w/alpha^2)`` per
    good pixel, so a fitted jitter is honored, and, where ``ar_phi`` is nonzero, an AR(1)
    process on the standardized noise. That is the model of
    :func:`albireo.forward.with_ar1`, run over all native pixels so that the observed
    subset carries ``phi**gap`` correlations across masked gaps. Masked pixels come back
    exactly zero.

    Parameters
    ----------
    problem
        The problem to draw from. Only its data term is replaced.
    d_stack
        Injected deviation spectra, ``(n_components, grid.n)``: the layout
        :attr:`albireo.likelihood.MarginalResult.d_hat` returns, so a fit can be
        bootstrapped from its own solution. A zeroed row leaves that component out.
    seed
        Seed for the noise draw.

    Returns
    -------
    Problem
        ``problem`` with the redrawn data.

    Raises
    ------
    ValueError
        If ``d_stack`` does not have shape ``(problem.n_components, problem.grid.n)``.

    Examples
    --------
    >>> base = model.problem_at({**orbit, "k": k_true})  # doctest: +SKIP
    >>> trial = resimulate(base, d_true, seed=7)  # doctest: +SKIP
    """
    from albireo.forward import with_data  # local: see TYPE_CHECKING above

    rng = np.random.default_rng(seed)
    d_stack = jnp.asarray(d_stack)
    if d_stack.shape != (problem.n_components, problem.grid.n):
        raise ValueError(
            f"d_stack must have shape ({problem.n_components}, {problem.grid.n}); "
            f"got {tuple(d_stack.shape)}"
        )
    z_new = []
    for g, model_dev in zip(problem.groups, _model_applier()(problem, d_stack), strict=True):
        w = np.asarray(g.effective_w)
        sigma = np.where(w > 0.0, 1.0 / np.sqrt(np.where(w > 0.0, w, 1.0)), 0.0)
        phi = np.asarray(g.ar_phi)
        noise = np.stack([_ar1_noise(rng, w.shape[1], float(phi[e])) for e in range(w.shape[0])])
        z_new.append(np.asarray(g.r) * np.asarray(model_dev) + sigma * noise)
    return with_data(problem, z_new)


# ---------------------------------------------------------------------------
# A small spectral library: the simulator's stand-in for BOSZ or POLLUX
# ---------------------------------------------------------------------------

_LIBRARY_DEFAULT_TEFF = tuple(float(t) for t in range(4000, 5751, 250))
_LIBRARY_DEFAULT_LOGG = (3.0, 3.5, 4.0, 4.5, 5.0)
_LIBRARY_DEFAULT_MH = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5)


def _library_line_depths(teff, logg, mh, n_lines: int) -> list[float]:
    """Line depths as functions of the labels, each label driving lines of its own.

    Two labels that moved the same lines in the same way would be interchangeable, and a
    fit would then drive the chi-square to zero along a curve through label space without
    recovering the injected values (the degenerate fixture recorded under D53). Teff,
    log g and [M/H] each drive lines the others do not, so the map from labels to spectrum
    is invertible and a recovery test measures the code rather than the fixture.
    """
    t = (teff - 4800.0) / 600.0
    g = logg - 4.0
    rules = (
        0.30 + 0.13 * np.tanh(t),
        0.22 - 0.11 * np.tanh(0.8 * t),
        0.26 + 0.09 * g + 0.02 * g**2,
        0.17 - 0.07 * g,
        0.21 + 0.16 * mh + 0.04 * mh**2,
        0.19 + 0.10 * mh - 0.03 * np.tanh(t),
    )
    depths = []
    for i in range(n_lines):
        base = rules[i % len(rules)]
        # Repeats past the sixth line are scaled so no two lines are identical copies.
        depths.append(float(base * (1.0 - 0.15 * (i // len(rules)))))
    return depths


def synthetic_library(
    wave_range: tuple[float, float] = (5150.0, 5250.0),
    *,
    n_pix: int = 1400,
    n_lines: int = 6,
    teff: Sequence[float] = _LIBRARY_DEFAULT_TEFF,
    logg: Sequence[float] = _LIBRARY_DEFAULT_LOGG,
    mh: Sequence[float] = _LIBRARY_DEFAULT_MH,
    medium: str = "air",
    line_width_angstrom: float = 0.25,
    seed: int = 0,
):
    """A small, complete-box spectral library standing in for BOSZ or POLLUX.

    The stand-in exercises every part of the synthetic-grid machinery above the grid
    itself: interpolation, broadening, the dilution model, the additive nuisance, the
    label optimizer, and the templates :mod:`albireo.todcor` correlates against. The label
    mode and the pipeline can therefore be tested and demonstrated offline; the published
    grids run to hundreds of megabytes (:func:`albireo.fetch_library`).

    Each node is a set of Gaussian absorption lines at fixed wavelengths whose depths
    depend on the labels, with each label driving lines of its own so that the map from
    labels to spectrum is invertible (see :func:`_library_line_depths`), plus a continuum
    that falls with Teff across the window. That wavelength dependence is what makes a
    light ratio measurable.

    Parameters
    ----------
    wave_range
        The window, in Angstrom.
    n_pix
        Samples across the window (uniform in wavelength).
    n_lines
        Lines per spectrum, spread across the window with an 8% margin at each edge.
    teff, logg, mh
        The node axes: effective temperature [K], surface gravity [dex] and metallicity
        [dex]. The default box matches the BOSZ FGK spacing (250 K, 0.5 dex).
    medium
        The wavelength scale to declare, ``"air"`` or ``"vacuum"``. Required by the
        container; the published grids carry no default either.
    line_width_angstrom
        Intrinsic Gaussian sigma of every line [Å]. Rotational broadening is applied by a
        kernel afterwards, never here.
    seed
        Seeds the jitter of the line positions.

    Returns
    -------
    albireo.library.SpectralLibrary
        A library on a complete Teff-log g-[M/H] box, with normalized flux and a
        log continuum at every node.

    Raises
    ------
    ValueError
        If ``wave_range`` is not increasing.

    References
    ----------
    Bohlin, R. C., Mészáros, Sz., Fleming, S. W., et al. 2017, AJ, 153, 234

    Palacios, A., Gebran, M., Josselin, E., et al. 2010, A&A, 516, A13
    """
    from albireo.library import SpectralLibrary

    lo, hi = float(wave_range[0]), float(wave_range[1])
    if not hi > lo:
        raise ValueError("wave_range must satisfy lo < hi")
    wave = np.linspace(lo, hi, int(n_pix))
    rng = np.random.default_rng(seed)
    margin = 0.08 * (hi - lo)
    centers = np.linspace(lo + margin, hi - margin, int(n_lines))
    spacing = (hi - lo - 2 * margin) / max(n_lines - 1, 1)
    centers = centers + rng.uniform(-0.15, 0.15, size=centers.size) * spacing

    nodes, normalized, continua = [], [], []
    for t in teff:
        for g in logg:
            for m in mh:
                depths = _library_line_depths(float(t), float(g), float(m), int(n_lines))
                flux = np.ones_like(wave)
                for center, depth in zip(centers, depths, strict=True):
                    flux = flux - depth * np.exp(
                        -0.5 * ((wave - center) / line_width_angstrom) ** 2
                    )
                log_continuum = (
                    30.0 + 4.0 * np.log(float(t) / 5000.0) - 0.025 * (wave - wave[0]) / 100.0
                )
                nodes.append((float(t), float(g), float(m)))
                normalized.append(flux)
                continua.append(log_continuum)
    return SpectralLibrary(
        label_names=("teff", "logg", "mh"),
        nodes=np.asarray(nodes),
        normalized=np.asarray(normalized),
        log_continuum=np.asarray(continua),
        wave=wave,
        medium=medium,
        meta={
            "grid": "albireo.simulate.synthetic_library (toy)",
            "vmicro": "n/a",
            "citation": "none: a synthetic stand-in generated by albireo",
        },
    )


def library_component(
    library,
    labels: Mapping[str, float],
    grid: LogGrid,
    *,
    medium: str,
    vsini_kms: float = 0.0,
    epsilon: float = 0.6,
) -> np.ndarray:
    """A library spectrum at ``labels`` as a deviation on ``grid``, rotationally broadened.

    This is the component a simulated star is built from when its labels are to be
    recovered. The library is resampled onto ``grid``, interpolated at ``labels`` with the
    same interpolator the label fit uses, reduced to a deviation by subtracting the unit
    continuum, and broadened with the pixel-integrated limb-darkened rotation kernel of
    Gray (2005). The result is passed to :func:`simulate_dataset` as one of its
    ``components``.

    Parameters
    ----------
    library
        A :class:`~albireo.library.SpectralLibrary`, e.g. :func:`synthetic_library`.
    labels
        ``{"teff": ..., "logg": ..., "mh": ...}``: every axis the library has.
    grid
        The model grid to render on.
    medium
        The wavelength scale of ``grid``; the library is converted onto it.
    vsini_kms, epsilon
        Projected rotational velocity [km/s] and the linear limb-darkening coefficient.
        A ``vsini_kms`` of 0 leaves the spectrum unbroadened.

    Returns
    -------
    numpy.ndarray
        ``(grid.n,)`` deviation spectrum ``d = s - 1``.

    Raises
    ------
    ValueError
        If ``labels`` omits a library axis, or if ``vsini_kms`` is negative.

    References
    ----------
    Gray, D. F. 2005, The Observation and Analysis of Stellar Photospheres, 3rd ed.
    (Cambridge: Cambridge University Press)
    """
    from albireo.library import library_interpolator
    from albireo.operators import rotational_kernel

    resampled = library.resampled_to(grid, medium=medium)
    missing = [axis for axis in resampled.label_names if axis not in labels]
    if missing:
        raise ValueError(f"labels are missing the library axes {missing}")
    interpolator = library_interpolator(resampled)
    point = jnp.asarray([float(labels[axis]) for axis in resampled.label_names])
    normalized, _ = interpolator(point)
    deviation = np.asarray(normalized, dtype=np.float64) - 1.0
    if vsini_kms < 0.0:
        raise ValueError("vsini_kms must be non-negative")
    if vsini_kms > 0.0:
        kernel = np.asarray(rotational_kernel(vsini_kms / grid.dv_kms, epsilon=epsilon))
        deviation = np.convolve(deviation, kernel, mode="same")
    return deviation
