"""Synthetic spectroscopic-binary datasets — the test harness for everything downstream.

Milestone M1 (``docs/design.md`` §8). Composite epochs are generated through the same
operator stack the inference code uses — shift → LSF convolution → rebin-to-native →
multiplicative response — so closed-loop tests exercise the real pipeline, including
every advertised pathology: chip gaps, cosmic hits, mixed instruments/resolutions,
tellurics, barycentric frames, and per-epoch light fractions.

Component spectra are *deviation* spectra ``d = s - 1`` on the model :class:`~albireo.grids.LogGrid`
(zero in the continuum, negative dips for absorption). Frame conventions follow
``docs/math.md`` §1.2: with ``frame="topocentric"`` the stellar log-shift is
``xi(v_star) - xi(v_bary)`` and tellurics are static; with ``frame="barycentric"`` the
stellar shift is ``xi(v_star)`` and tellurics move by ``+xi(v_bary)``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

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

__all__ = [
    "InstrumentSpec",
    "OrbitParams",
    "SimulationTruth",
    "chebyshev_response",
    "simulate_dataset",
    "synthetic_deviation_spectrum",
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
        Gaussian LSF width in km/s: a scalar for a stationary LSF, or — with
        ``lsf_anchors_angstrom`` — one width per anchor for a wavelength-dependent
        LSF (D37), linearly interpolated across the grid exactly as the forward
        model realizes it (:func:`albireo.operators.gaussian_lsf_profiles`).
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
    """Keplerian SB2 orbit for the simulator (component 2 uses ``omega + pi``)."""

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
    """Everything that was injected — the oracle for closed-loop tests."""

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
    response_order: int = 0,
    response_amplitude: float = 0.0,
    ar1_phi: float = 0.0,
    gap_fraction: float = 0.0,
    cosmic_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[Dataset, SimulationTruth]:
    """Generate a synthetic multi-epoch dataset plus the injected truth.

    Parameters
    ----------
    grid
        Model log-wavelength grid.
    components
        Stellar deviation spectra on ``grid`` (one array of shape ``(grid.n,)`` per
        component).
    bjd
        Epoch times, shape ``(n_ep,)``.
    instruments
        Named instrument specs; each epoch is observed with one of them.
    light_fractions
        ``(n_comp,)`` constant or ``(n_comp, n_ep)`` per-epoch; must sum to 1 per epoch.
    orbit, velocities
        Either a Keplerian :class:`OrbitParams` (with ``len(orbit.k) == n_comp``) or an
        explicit ``(n_comp, n_ep)`` velocity array (free-velocity mode). Exactly one
        must be given.
    epoch_instruments
        Instrument name per epoch; defaults to the first instrument for all epochs.
    v_bary
        Barycentric correction velocities per epoch [km/s]; default: uniform random in
        ±30 km/s.
    frame
        ``"topocentric"`` (default) or ``"barycentric"`` — the frame of the emitted
        wavelength grids, with shift conventions from ``docs/math.md`` §1.2.
    telluric
        Optional telluric deviation spectrum on ``grid`` (light fraction 1, additive).
    response_order, response_amplitude
        Per-epoch multiplicative Chebyshev response ``1 + sum_{m<=order} c_m T_m`` with
        ``c_m ~ N(0, response_amplitude^2)``. Amplitude 0 disables it (r = 1).
    ar1_phi
        AR(1) correlation of the pixel noise (``|phi| < 1``): the noise is a
        stationary AR(1) process over the native pixel index with marginal standard
        deviation ``1/snr`` — the model of :func:`albireo.forward.with_ar1`, so
        closed-loop tests can inject and recover it. The process runs over *all*
        pixels (masked ones included), which is what makes the observed subset carry
        ``phi**gap`` correlations across masked gaps. 0 = white noise (default).
    gap_fraction
        Fraction of each epoch's pixels lost to one contiguous chip gap (ivar = 0 and
        flux overwritten with garbage, so downstream code MUST honor the mask).
    cosmic_fraction
        Fraction of pixels hit by cosmics (large positive spikes, ivar = 0).
    seed
        Seed for all randomness (noise, v_bary, response, gaps, cosmics).

    Returns
    -------
    (Dataset, SimulationTruth)
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

    bary_pix = np.asarray(grid.velocity_to_pixels(v_bary))
    star_pix = np.asarray(grid.velocity_to_pixels(vel))  # (n_comp, n_ep)
    if frame == "topocentric":
        star_pix = star_pix - bary_pix[None, :]
        tell_pix = np.zeros(n_ep)
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
            flux[gap] = rng.normal(0.0, 10.0, size=width)  # garbage: mask must be honored
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
    )
    return Dataset(epochs=tuple(epochs), frame=frame), truth
