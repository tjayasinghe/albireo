"""Log-wavelength grids and the Doppler shift ↔ log-shift mapping.

The model grid is uniform in ``x = ln(lambda)``, so that a Doppler shift is a pure
translation in ``x`` (see ``docs/math.md`` §1.1). All velocity/pixel conversions go
through :func:`log_doppler_shift`, which is the single place the Doppler convention lives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

__all__ = ["C_KMS", "LogGrid", "log_doppler_shift"]

C_KMS: float = 299_792.458
"""Speed of light in km/s."""


def log_doppler_shift(v_kms, *, relativistic: bool = True):
    """Log-wavelength shift ``xi(v) = ln(1 + z)`` for radial velocity ``v``.

    Positive ``v`` means the source is receding (redshift): observed wavelengths satisfy
    ``ln(lambda_obs) = ln(lambda_emit) + xi(v)``.

    Parameters
    ----------
    v_kms
        Radial velocity in km/s. May be a scalar or array; differentiable under JAX.
    relativistic
        If True (default), use the exact radial-Doppler mapping ``xi = artanh(v/c)``,
        which is exactly antisymmetric — shifts compose and invert exactly, and
        barycentric corrections are exact additive compositions in ``x``. If False,
        use the classical ``xi = ln(1 + v/c)`` (wrong by ~0.6 km/s at 600 km/s).

    Returns
    -------
    jax.Array
        The log-wavelength shift ``xi(v)`` (dimensionless).
    """
    beta = jnp.asarray(v_kms) / C_KMS
    if relativistic:
        return jnp.arctanh(beta)
    return jnp.log1p(beta)


@dataclass(frozen=True)
class LogGrid:
    """A uniform grid in ``x = ln(lambda)``: ``x_p = x0 + p * dx``.

    Attributes
    ----------
    x0
        Log-wavelength of the first pixel, ``ln(lambda_0)`` with wavelength in Angstrom.
    dx
        Pixel spacing in log-wavelength.
    n
        Number of pixels.
    relativistic
        Doppler convention used by :meth:`velocity_to_pixels` (must match the convention
        used when building the grid; see :meth:`from_wavelength_range`).
    """

    x0: float
    dx: float
    n: int
    relativistic: bool = True

    @classmethod
    def from_wavelength_range(
        cls,
        wave_min: float,
        wave_max: float,
        dv_kms: float,
        *,
        relativistic: bool = True,
    ) -> LogGrid:
        """Build a grid covering ``[wave_min, wave_max]`` with pixel width ``dv_kms``.

        The first pixel sits exactly at ``wave_min``; the last pixel is at or just above
        ``wave_max``.
        """
        if not (wave_min > 0 and wave_max > wave_min):
            raise ValueError("need 0 < wave_min < wave_max")
        if not dv_kms > 0:
            raise ValueError("dv_kms must be positive")
        dx = float(log_doppler_shift(dv_kms, relativistic=relativistic))
        span = math.log(wave_max) - math.log(wave_min)
        n = math.ceil(span / dx) + 1
        return cls(x0=math.log(wave_min), dx=dx, n=n, relativistic=relativistic)

    @property
    def x(self) -> np.ndarray:
        """Log-wavelengths of the pixel centers."""
        return self.x0 + self.dx * np.arange(self.n)

    @property
    def wave(self) -> np.ndarray:
        """Wavelengths of the pixel centers (same unit as the grid was built with)."""
        return np.exp(self.x)

    @property
    def dv_kms(self) -> float:
        """Pixel width expressed as a velocity, inverting the Doppler mapping."""
        if self.relativistic:
            return C_KMS * math.tanh(self.dx)
        return C_KMS * math.expm1(self.dx)

    def velocity_to_pixels(self, v_kms):
        """Shift in *pixels* corresponding to radial velocity ``v_kms``.

        Differentiable under JAX; this is what feeds the shift operators.
        """
        return log_doppler_shift(v_kms, relativistic=self.relativistic) / self.dx
