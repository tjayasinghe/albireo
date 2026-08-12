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

    @classmethod
    def covering(
        cls,
        dataset,
        dv_kms: float,
        *,
        v_margin_kms: float,
        lsf_sigma_kms: float = 0.0,
        lsf_truncate: float = 4.0,
        extra_pixels: int = 4,
        relativistic: bool = True,
    ) -> LogGrid:
        """Build a model grid that covers a dataset *with the margin the solver needs*.

        The model grid must be wider than the data, and by a specific amount. Two effects
        set it:

        - **Velocity.** A component shifted by ``v`` maps model pixel ``q`` onto data at
          ``q + xi(v)/dx``, so the grid has to extend beyond the data by the largest shift
          any component will ever take — the orbital semi-amplitudes, plus the barycentric
          motion when the data are topocentric or a telluric component is in play.
          Without it the shifted model runs off the end of the grid and the fit quietly
          loses the flux there.
        - **The LSF.** Convolution mixes a further ``truncate * sigma`` pixels in from each
          side. This one is sharper than it sounds: a margin smaller than the kernel
          radius was the trigger for a real defect in albireo's band assembly, in which
          the convolution wrote entries at columns off the model grid and the shift
          sandwich read them back (fixed, and now guarded — but the margin is what keeps
          the situation from arising).

        Together with a few pixels of slack, that is the whole rule, and this method
        applies it so callers do not have to rediscover it.

        Parameters
        ----------
        dataset : Dataset
            Epochs to cover; only the wavelength extremes are used.
        dv_kms : float
            Grid pixel width as a velocity. Make it no coarser than the finest native
            sampling in the data — for a spectrograph delivering a constant *wavelength*
            step, that is at the blue end.
        v_margin_kms : float
            Largest velocity by which any component will be shifted relative to the data
            frame. For an SB2 in barycentric-frame data: ``max(K_i) * (1 + e)``, with
            headroom. Add ~30 km/s if the data are topocentric or a telluric component is
            enabled.
        lsf_sigma_kms : float, optional
            Widest Gaussian LSF sigma among the instruments, km/s. Default 0.
        lsf_truncate : float, optional
            Kernel truncation in sigmas; must match
            :func:`albireo.operators.gaussian_kernel` (default 4).
        extra_pixels : int, optional
            Slack added on each side. Default 4.
        relativistic : bool, optional
            Doppler convention, as in :meth:`from_wavelength_range`.

        Returns
        -------
        LogGrid
            A grid spanning the data plus the margin, on both sides.

        Examples
        --------
        >>> import numpy as np
        >>> from albireo.data import Dataset, EpochData
        >>> ep = EpochData(wave=np.linspace(4000.0, 4600.0, 100), flux=np.ones(100),
        ...                ivar=np.ones(100), bjd=2453000.0)
        >>> grid = LogGrid.covering(Dataset([ep]), dv_kms=2.0, v_margin_kms=80.0,
        ...                         lsf_sigma_kms=2.7)
        >>> bool(grid.wave[0] < 4000.0 and grid.wave[-1] > 4600.0)
        True
        """
        if not dv_kms > 0:
            raise ValueError("dv_kms must be positive")
        if v_margin_kms < 0 or lsf_sigma_kms < 0:
            raise ValueError("v_margin_kms and lsf_sigma_kms must be non-negative")
        wave_min = min(float(epoch.wave[0]) for epoch in dataset)
        wave_max = max(float(epoch.wave[-1]) for epoch in dataset)

        dx = float(log_doppler_shift(dv_kms, relativistic=relativistic))
        kernel_radius = max(1, math.ceil(lsf_truncate * lsf_sigma_kms / dv_kms))
        margin_dx = (
            abs(float(log_doppler_shift(v_margin_kms, relativistic=relativistic)))
            + (kernel_radius + int(extra_pixels)) * dx
        )
        lo = math.exp(math.log(wave_min) - margin_dx)
        hi = math.exp(math.log(wave_max) + margin_dx)
        return cls.from_wavelength_range(lo, hi, dv_kms, relativistic=relativistic)

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
