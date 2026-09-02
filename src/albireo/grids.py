"""Log-wavelength grids and the mapping between radial velocity and log-wavelength shift.

The model grid is uniform in ``x = ln(lambda)``, so a Doppler shift is a pure translation in
``x`` (``docs/math.md`` §1.1). All velocity and pixel conversions go through
:func:`log_doppler_shift`, which is the single place the Doppler convention is defined.

The module also provides the IAU-adopted air/vacuum conversions, :func:`vacuum_to_air` and
:func:`air_to_vacuum`.

References
----------
Edlén, B. 1966, Metrologia, 2, 71
Birch, K. P. & Downs, M. J. 1994, Metrologia, 31, 315
Morton, D. C. 2000, ApJS, 130, 403
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

__all__ = ["C_KMS", "LogGrid", "air_to_vacuum", "log_doppler_shift", "vacuum_to_air"]

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
        If True (default), use the exact radial-Doppler mapping ``xi = artanh(v/c)``. It is
        antisymmetric, so shifts compose and invert exactly and barycentric corrections are
        exact additive compositions in ``x``. If False, use the classical
        ``xi = ln(1 + v/c)`` (wrong by ~0.6 km/s at 600 km/s).

    Returns
    -------
    jax.Array
        The log-wavelength shift ``xi(v)`` (dimensionless).
    """
    beta = jnp.asarray(v_kms) / C_KMS
    if relativistic:
        return jnp.arctanh(beta)
    return jnp.log1p(beta)


def _iau_n_minus_one(sigma2):
    """Edlén (1966) and Birch & Downs (1994) refractivity, as adopted by the IAU.

    ``sigma2`` is the squared vacuum wavenumber in um^-2. This is the form Morton (2000)
    tabulates and the one SDSS, VALD and the NIST tables use, so line lists converted with it
    agree to well below 1 m/s.
    """
    return 1e-8 * (8342.13 + 2406030.0 / (130.0 - sigma2) + 15997.0 / (38.9 - sigma2))


def vacuum_to_air(wave_vacuum):
    """Convert vacuum wavelengths [Angstrom] to standard air.

    Air wavelengths are what most optical spectrographs report and what most optical line
    lists are tabulated in; vacuum is used in the UV and the IR and by ESPRESSO and Gaia RVS.
    The difference is 0.87 Angstrom at 3000 A, rising to 2.74 A at 10000 A. Expressed as a
    velocity it is nearly constant at 83 km/s across the optical (87.4 at 3000 A, 82.8 at
    Halpha, 82.2 at 10000 A), the same order as the orbital semi-amplitudes albireo measures,
    and it does not average out over epochs.

    The conversion uses the IAU-adopted Edlén (1966) refractivity in the Birch & Downs (1994)
    parameterization, evaluated at the vacuum wavenumber. That is the convention Morton (2000)
    tabulates, and therefore the one published air line lists agree with.

    Parameters
    ----------
    wave_vacuum
        Vacuum wavelengths in Angstrom. Any shape; differentiable under JAX.

    Returns
    -------
    jax.Array
        Air wavelengths, same shape.

    See Also
    --------
    air_to_vacuum : the inverse.
    albireo.data.EpochData : declares which scale an epoch is on.

    References
    ----------
    Edlén, B. 1966, Metrologia, 2, 71
    Birch, K. P. & Downs, M. J. 1994, Metrologia, 31, 315
    Morton, D. C. 2000, ApJS, 130, 403
    """
    wave = jnp.asarray(wave_vacuum, dtype=float)
    sigma2 = (1e4 / wave) ** 2
    return wave / (1.0 + _iau_n_minus_one(sigma2))


def air_to_vacuum(wave_air):
    """Convert standard-air wavelengths [Angstrom] to vacuum, the inverse of :func:`vacuum_to_air`.

    The refractivity is defined at the vacuum wavenumber, so the inverse has no closed form.
    Two fixed-point iterations are used. The refractivity changes by ~1e-8 over the 0.03% by
    which the wavelength moves, so the first iteration is already correct to ~1e-11 Angstrom
    and the second makes the round trip exact to float64. The test suite verifies the round
    trip to 1e-10 Angstrom over 3000-10000 Angstrom.

    Parameters
    ----------
    wave_air
        Standard-air wavelengths in Angstrom. Any shape; differentiable under JAX.

    Returns
    -------
    jax.Array
        Vacuum wavelengths, same shape.

    See Also
    --------
    vacuum_to_air : the forward conversion.

    References
    ----------
    Edlén, B. 1966, Metrologia, 2, 71
    Birch, K. P. & Downs, M. J. 1994, Metrologia, 31, 315
    Morton, D. C. 2000, ApJS, 130, 403
    """
    wave = jnp.asarray(wave_air, dtype=float)
    vac = wave * (1.0 + _iau_n_minus_one((1e4 / wave) ** 2))
    for _ in range(2):
        vac = wave * (1.0 + _iau_n_minus_one((1e4 / vac) ** 2))
    return vac


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
        """Build a model grid covering a dataset, with the margin the solver requires.

        The model grid must be wider than the data by an amount set by two effects, plus a
        few pixels of slack:

        - Velocity. A component shifted by ``v`` maps model pixel ``q`` onto data at
          ``q + xi(v)/dx``, so the grid must extend beyond the data by the largest shift any
          component takes: the orbital semi-amplitudes, plus the barycentric motion when the
          data are topocentric or a telluric component is present. Otherwise the shifted
          model runs off the end of the grid and the flux there is lost without a warning.
        - The LSF. Convolution mixes a further ``truncate * sigma`` pixels in from each side.
          A margin smaller than the kernel radius lets the convolution address columns off
          the model grid, which the shift operators would then read back.

        Parameters
        ----------
        dataset : Dataset
            Epochs to cover; only the wavelength extremes are used.
        dv_kms : float
            Grid pixel width as a velocity. It should be no coarser than the finest native
            sampling in the data, which for a spectrograph delivering a constant wavelength
            step is at the blue end.
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
        """Shift in pixels corresponding to radial velocity ``v_kms``.

        This is the quantity consumed by the shift operators. Differentiable under JAX.
        """
        return log_doppler_shift(v_kms, relativistic=self.relativistic) / self.dx

    def pixels_to_velocity(self, pixels):
        """Radial velocity [km/s] corresponding to a shift of ``pixels``, the exact inverse.

        With the default relativistic mapping ``xi = artanh(v/c)`` the inverse is
        ``v = c tanh(xi)``. Because ``xi`` turns relativistic velocity addition into ordinary
        addition, differences of pixel shifts map to exact relative velocities rather than to
        approximations of them. A per-epoch velocity table is therefore exactly identified
        (:func:`albireo.inference.relative_velocities`): the arbitrary zero point is removed
        by subtraction in pixel space, and this method maps the remainder back to km/s.

        Differentiable under JAX.
        """
        xi = jnp.asarray(pixels) * self.dx
        if self.relativistic:
            return C_KMS * jnp.tanh(xi)
        return C_KMS * jnp.expm1(xi)
