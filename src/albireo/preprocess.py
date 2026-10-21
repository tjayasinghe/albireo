"""Turning a reduced spectrum into an :class:`~albireo.data.EpochData`.

The disentangling model is ``1 + sum_i l_i d_i`` with ``d_i`` a deviation from a unit
continuum (``docs/math.md`` §1.4), and every weight is an inverse variance. A
pipeline-reduced archival spectrum therefore requires three operations before it can enter
a :class:`~albireo.data.Dataset`.

1. Continuum normalization (:func:`fit_continuum`, :func:`normalize`). Merged echelle spectra
   retain the blaze and the instrument response and are routinely delivered unnormalized; ESO
   Phase 3 marks this with ``CONTNORM = False``. albireo's response term ``r_j`` is fixed at
   build time rather than inferred, so the normalization done here is the normalization the
   model uses.
2. An inverse variance (:func:`estimate_ivar`). Many archival products have no usable error
   array: the ESO Phase 3 FEROS spectra, for instance, have an ``ERR`` column that is
   entirely ``NaN``. The noise must then be estimated from the spectrum, in a way that
   deep absorption lines do not bias.
3. Region selection and masking (:func:`select_region`, :func:`mask_ranges`,
   :func:`mask_tellurics`, :func:`mask_spikes`). Masking is always ``ivar = 0`` rather than
   deletion; :func:`mask_ranges` gives the reason.

Everything here is pure NumPy, as in :mod:`albireo.data`, so the preprocessing is inspectable
without JAX. Nothing here reads a file; see :mod:`albireo.io` for that.

The functions compose, but each is usable on its own: a normalized spectrum with reliable
inverse variances needs only :func:`select_region` and the masking helpers.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.linalg import solveh_banded

from albireo.data import EpochData

__all__ = [
    "TELLURIC_BANDS",
    "continuum_correction",
    "der_snr_sigma",
    "estimate_ivar",
    "fit_continuum",
    "mask_flux_gaps",
    "mask_ranges",
    "mask_spikes",
    "mask_tellurics",
    "normalize",
    "select_region",
    "share_wavelength_grid",
    "smooth_epoch",
]

_C_KMS = 299_792.458


# Main telluric absorption complexes in the optical, as (lambda_min, lambda_max) in
# Angstrom (air). The O2 gamma, B and A bands are sharp and deep. The H2O complexes are
# forests of weaker lines whose strength tracks the precipitable water vapour, so they
# vary between exposures in a way that a static telluric component cannot model. The
# ranges are wide: masking loses one pixel per pixel masked, while a telluric line left
# in the data produces a spurious component. The list is not exhaustive below 5800 A,
# where telluric absorption is weak enough to ignore at typical SNR.
TELLURIC_BANDS: tuple[tuple[float, float], ...] = (
    (5870.0, 6000.0),  # H2O
    (6270.0, 6330.0),  # O2 gamma
    (6450.0, 6620.0),  # H2O
    (6860.0, 6970.0),  # O2 B
    (7150.0, 7400.0),  # H2O
    (7580.0, 7720.0),  # O2 A
    (8100.0, 8400.0),  # H2O
    (8900.0, 9900.0),  # H2O
)
"""Default telluric windows used by :func:`mask_tellurics` (Angstrom, air)."""


def _finite_weights(flux: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Starting weights: the caller's, zeroed wherever the flux is not finite."""
    if weights is None:
        w = np.ones(flux.size, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != flux.shape:
            raise ValueError(f"weights must have shape {flux.shape}, got {w.shape}")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
        w = np.where(np.isfinite(w), w, 0.0)
    return np.where(np.isfinite(flux), w, 0.0)


# Knots per half-power smoothing length. Eight is well above the two a piecewise-linear
# basis needs to represent a feature of that width, and it fixes the penalty weight at
# (8 / 2pi)^4 ~ 2.6 (see _KnotBasis). A per-pixel smoother would instead need
# lam ~ (L_pixels / 2pi)^4, which is ~1e12 at the 5000-pixel smoothing lengths a merged
# echelle spectrum requires. There the weight term is lost to rounding and the nominally
# positive-definite matrix factorizes as singular.
_KNOTS_PER_LENGTH = 8.0


class _KnotBasis:
    """Piecewise-linear basis on uniform knots, with a curvature-penalized fit.

    Fits ``z(p)``, linear between ``m`` uniformly spaced knots, by minimizing

        ``sum_p w_p (y_p - z(p))^2 + lam * sum_k (D2 z_knots)_k^2``

    The normal matrix is ``B^T W B + lam D2^T D2``. ``B`` has two nonzeros per row at adjacent
    knots, so ``B^T W B`` is tridiagonal and the sum is pentadiagonal, solved by one
    ``solveh_banded`` call in ``O(m)``. Reducing the unknowns from pixels to knots keeps the
    system well conditioned at long smoothing lengths (see :data:`_KNOTS_PER_LENGTH`) and the
    iteration cheap. A continuum with a 150 A scale does not require 20000 free parameters.
    """

    def __init__(self, n: int, smooth_pixels: float):
        n_knots = round(n * _KNOTS_PER_LENGTH / smooth_pixels) + 1
        self.m = m = int(np.clip(n_knots, 5, n))
        u = np.linspace(0.0, m - 1.0, n)
        self.i0 = np.clip(np.floor(u).astype(np.intp), 0, m - 2)
        self.frac = u - self.i0
        # Half-power scale in knots; ~_KNOTS_PER_LENGTH except where m was clamped.
        self.lam = float(((smooth_pixels * (m - 1) / max(n - 1, 1)) / (2.0 * math.pi)) ** 4)

    def evaluate(self, z_knots: np.ndarray) -> np.ndarray:
        """Sample the piecewise-linear function at every pixel."""
        return (1.0 - self.frac) * z_knots[self.i0] + self.frac * z_knots[self.i0 + 1]

    def fit(self, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Knot values of the penalized weighted least-squares fit."""
        m, i0, frac, lam = self.m, self.i0, self.frac, self.lam
        b0, b1 = 1.0 - frac, frac
        # B^T W B (tridiagonal) and B^T W y, by scatter-add over pixels.
        main = np.bincount(i0, weights=w * b0 * b0, minlength=m)
        main += np.bincount(i0 + 1, weights=w * b1 * b1, minlength=m)
        off = np.bincount(i0, weights=w * b0 * b1, minlength=m)[: m - 1]
        rhs = np.bincount(i0, weights=w * b0 * y, minlength=m)
        rhs += np.bincount(i0 + 1, weights=w * b1 * y, minlength=m)

        # lam * D2^T D2: pentadiagonal Toeplitz (6, -4, 1) with boundary corrections, the
        # same operator albireo.assembly._prior_diagonals builds for the spectral prior.
        d0 = np.full(m, 6.0)
        d0[0] -= 5.0
        d0[1] -= 1.0
        d0[-1] -= 5.0
        d0[-2] -= 1.0
        d1 = np.full(m - 1, -4.0)
        d1[0] += 2.0
        d1[-1] += 2.0

        ab = np.zeros((3, m), dtype=np.float64)
        ab[2] = main + lam * d0
        ab[1, 1:] = off + lam * d1
        ab[0, 2:] = lam
        return solveh_banded(ab, rhs, lower=False, check_finite=False)


def fit_continuum(
    wave,
    flux,
    *,
    smooth_angstrom: float | None = None,
    weights=None,
    asymmetry: float = 0.97,
    low_reject: float = 1.5,
    high_reject: float = 3.0,
    envelope_iterations: int = 12,
    clip_iterations: int = 6,
) -> np.ndarray:
    """Fit a smooth continuum through the upper envelope of a spectrum.

    The fit is performed on ``log(flux)``. A continuum is a multiplicative factor (blaze,
    throughput, extinction) that varies strongly across a merged echelle spectrum: over
    3850-4750 A a FEROS exposure of a B star falls by a factor of 20. A curvature penalty
    applied to the flux cannot follow that, since a stiff smoother lags a steep exponential
    and the normalized spectrum is wrong by 30%. In the log the same fall is nearly a straight
    line, and straight lines lie in the null space of the second-difference penalty, so they
    are not penalized. Absorption depths are additive in the log, so the rejection thresholds
    below are fractional.

    The fit proceeds in two stages, both penalized least-squares fits on a knot grid
    (:class:`_KnotBasis`).

    Stage 1, asymmetric envelope. The fit is iteratively reweighted with weight ``asymmetry``
    above the current curve and ``1 - asymmetry`` below it, which raises the curve above
    the absorption lines. This stage alone leaves the curve about one noise sigma high,
    because it fits the upper envelope of the noise as well as of the lines.

    Stage 2, asymmetric sigma clipping. Pixels more than ``low_reject`` sigma below or
    ``high_reject`` sigma above the curve are given zero weight and the smoother is refit on
    the remaining pixels, with sigma re-estimated robustly at each pass. The thresholds are
    asymmetric because absorption lines are deep and one-sided while cosmic rays are high and
    rare. The remaining pixels are then continuum, so the curve is centred on them instead of
    on their upper envelope.

    Parameters
    ----------
    wave : array_like
        Wavelengths, shape ``(n,)``, strictly increasing. Used only to convert
        ``smooth_angstrom`` into pixels (via the median spacing). The fit runs on the pixel
        index, so on a non-uniform grid the smoothing scale is correct only on average.
    flux : array_like
        Observed flux, shape ``(n,)``. Non-finite and non-positive samples are given zero
        weight, since the log is undefined there. They need not be removed beforehand; the
        smoother interpolates across them, so the returned continuum is strictly positive
        everywhere.
    smooth_angstrom : float, optional
        Half-power smoothing scale. Structure much broader than this is treated as continuum
        and structure much narrower as signal. Default: one eighth of the wavelength span.
        It must be wider than the widest line to be preserved, or that line is fitted as
        continuum. Broad lines are handled mainly by the rejection stage rather than by
        stiffness, so a scale that follows the blaze is preferable.
    weights : array_like, optional
        Prior per-pixel weights, shape ``(n,)``, non-negative. A 0/1 mask excludes known-bad
        pixels from the fit. Default: all ones. These weight the log residuals, so passing
        ``ivar`` (which weights flux residuals) is not statistically exact; use it as a mask.
    asymmetry : float, optional
        Stage-1 weight given to pixels above the curve; ``1 - asymmetry`` below.
        Must lie in ``(0.5, 1)``. Default 0.97.
    low_reject, high_reject : float, optional
        Stage-2 clipping thresholds in robust sigma, below and above the curve.
        Defaults 1.5 and 3.0.
    envelope_iterations, clip_iterations : int, optional
        Iteration counts for the two stages; both stop early once the fit stops changing.
        Defaults 12 and 6.

    Returns
    -------
    numpy.ndarray
        The continuum, shape ``(n,)``, float64, strictly positive.

    Raises
    ------
    ValueError
        If the arrays differ in shape, fewer than 5 samples are supplied, no pixel has
        both positive flux and positive weight, or a parameter is out of range.

    See Also
    --------
    normalize : divides by this fit and returns an :class:`~albireo.data.EpochData`-ready flux.

    Examples
    --------
    >>> import numpy as np
    >>> wave = np.linspace(4000.0, 4600.0, 2000)
    >>> truth = 2.0 * np.exp(-0.004 * (wave - 4000.0))   # a steep multiplicative response
    >>> flux = truth * (1.0 - 0.5 * np.exp(-0.5 * ((wave - 4300.0) / 1.0) ** 2))  # + a line
    >>> cont = fit_continuum(wave, flux, smooth_angstrom=150.0)
    >>> bool(np.all(np.abs(cont / truth - 1.0) < 0.01))
    True
    """
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if wave.shape != flux.shape or wave.ndim != 1:
        raise ValueError(
            f"wave and flux must be 1-D of equal length; got {wave.shape}, {flux.shape}"
        )
    if not 0.5 < asymmetry < 1.0:
        raise ValueError(f"asymmetry must lie in (0.5, 1); got {asymmetry}")
    if low_reject <= 0 or high_reject <= 0:
        raise ValueError("low_reject and high_reject must be positive")

    w0 = _finite_weights(flux, weights)
    w0 = np.where(flux > 0.0, w0, 0.0)  # the fit is in the log
    if not np.any(w0 > 0):
        raise ValueError(
            "no pixel has both positive flux and positive weight; nothing to fit a "
            "continuum to. A spectrum that is everywhere <= 0 cannot be normalized."
        )
    y = np.log(np.where(w0 > 0, flux, 1.0))

    span = float(wave[-1] - wave[0])
    if smooth_angstrom is None:
        smooth_angstrom = span / 8.0
    if not 0.0 < smooth_angstrom <= span:
        raise ValueError(
            f"smooth_angstrom must be positive and no wider than the spectrum ({span:g} A); "
            f"got {smooth_angstrom:g}"
        )
    step = float(np.median(np.diff(wave)))
    if not step > 0:
        raise ValueError("wave must be strictly increasing")
    smooth_pixels = float(smooth_angstrom) / step
    if smooth_pixels < 4.0:
        raise ValueError(
            f"smooth_angstrom={smooth_angstrom:g} is only {smooth_pixels:.1f} pixels wide; "
            "a continuum that flexible will follow the lines. Use at least a few tens of pixels."
        )
    basis = _KnotBasis(wave.size, smooth_pixels)

    # Stage 1: asymmetric upper envelope.
    good = w0 > 0
    z = basis.evaluate(basis.fit(y, w0))
    for _ in range(int(envelope_iterations)):
        w = np.where(y > z, asymmetry, 1.0 - asymmetry) * w0
        z_new = basis.evaluate(basis.fit(y, w))
        converged = np.allclose(z_new, z, rtol=0.0, atol=1e-8)
        z = z_new
        if converged:
            break

    # Stage 2: asymmetric sigma clipping about the envelope.
    keep = good.copy()
    for _ in range(int(clip_iterations)):
        resid = y - z
        sigma = _robust_sigma(resid[keep]) if np.any(keep) else 0.0
        if not sigma > 0:
            break
        keep_new = good & (resid > -low_reject * sigma) & (resid < high_reject * sigma)
        if keep_new.sum() < basis.m or np.array_equal(keep_new, keep):
            break
        keep = keep_new
        z = basis.evaluate(basis.fit(y, np.where(keep, w0, 0.0)))
    return np.exp(z)


def _robust_sigma(values: np.ndarray) -> float:
    """Median-absolute-deviation sigma of ``values`` (0.0 for an empty input)."""
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - med)))


def der_snr_sigma(flux) -> float:
    """Robust noise estimate from lag-2 differences (Stoehr et al. 2008, "DER_SNR").

    ``sigma = 1.482602 / sqrt(6) * median(|2 f_i - f_{i-2} - f_{i+2}|)``

    The lag-2 stencil annihilates any locally linear signal, so a spectral line contributes
    only through its curvature. The median is robust to that contribution, so the estimate
    reflects the noise. Skipping the immediate neighbours also limits the bias when a
    pipeline has resampled the spectrum and correlated adjacent pixels, as a merged,
    rebinned echelle product has.

    This is the estimator ESO uses for the ``SNR`` keyword of its Phase 3 spectra, so an
    estimate made here is directly comparable with the archive's value.

    Parameters
    ----------
    flux : array_like
        Flux samples, shape ``(n,)`` with ``n >= 5``. Non-finite samples are dropped.

    Returns
    -------
    float
        The noise standard deviation in the units of ``flux``; ``nan`` if fewer than
        five samples are finite.

    References
    ----------
    Stoehr, F. et al. 2008, in ASP Conf. Ser. 394, Astronomical Data Analysis Software and
    Systems XVII, 505
    """
    f = np.asarray(flux, dtype=np.float64)
    f = f[np.isfinite(f)]
    if f.size < 5:
        return float("nan")
    d = np.abs(2.0 * f[2:-2] - f[:-4] - f[4:])
    return float(1.482602 / math.sqrt(6.0) * np.median(d))


def estimate_ivar(
    wave,
    flux,
    *,
    continuum=None,
    n_bins: int = 32,
    scaling: str = "poisson",
    mask=None,
) -> np.ndarray:
    """Estimate per-pixel inverse variances from the spectrum itself.

    Intended for archival products with no usable error array. The noise is measured with
    :func:`der_snr_sigma` (Stoehr et al. 2008) in ``n_bins`` wavelength bins, and the binned
    estimates are converted to a per-pixel ``sigma(lambda)`` by one of the ``scaling`` rules
    below. The estimate is binned because a per-pixel noise estimate is itself noisy, and
    noisy weights bias a maximum-likelihood fit, whereas a smooth ``sigma(lambda)`` does not.

    Parameters
    ----------
    wave, flux : array_like
        Wavelengths and the normalized flux (continuum near 1), shape ``(n,)``.
    continuum : array_like, optional
        The continuum in the original flux units, shape ``(n,)``, as returned by
        :func:`fit_continuum` before the division. Required for ``scaling="poisson"``,
        unused otherwise.
    n_bins : int, optional
        Number of wavelength bins in which the noise is measured. Default 32. Each bin
        needs at least ~50 pixels to give a stable median; bins are merged automatically
        if the spectrum is too short.
    scaling : {"poisson", "interpolate", "constant"}, optional
        How the binned sigmas become a per-pixel sigma:

        - ``"poisson"`` (default): fit ``sigma^2 = s^2 / continuum`` for a single constant
          ``s``. In a photon-limited spectrum the normalized flux has variance inversely
          proportional to the collected counts, so this one-parameter law has the correct
          shape and is determined by all the bins. Falls back to ``"interpolate"``, with a
          warning, if the continuum is not everywhere positive.
        - ``"interpolate"``: linearly interpolate the binned sigmas over wavelength. This
          makes no assumption about the origin of the noise; it applies when the throughput
          and the noise are not simply related (sky-subtraction residuals, co-added
          exposures of unequal depth).
        - ``"constant"``: one sigma for the whole spectrum.
    mask : array_like of bool, optional
        ``True`` marks pixels to exclude from the noise measurement; they still receive an
        inverse variance. Intended for known cosmic rays or emission lines.

    Returns
    -------
    numpy.ndarray
        Inverse variances, shape ``(n,)``, float64, strictly positive and finite.
        Pixels whose flux is not finite get ``ivar = 0``, matching
        :class:`~albireo.data.EpochData`'s masking convention.

    Raises
    ------
    ValueError
        On shape mismatches, an unknown ``scaling``, a missing ``continuum`` when one is
        required, or if no bin yields a finite noise estimate.

    Notes
    -----
    The result estimates the noise of the delivered product and includes the effects of the
    reduction pipeline. Resampling in particular correlates neighbouring pixels, so the
    diagonal inverse-variance model albireo uses is an approximation for any archival product
    that has been rebinned onto a common wavelength grid. ``docs/math.md`` §1.1 gives the
    same reason for not resampling in albireo, but an archive may already have done it. The
    lag-2 stencil keeps the amplitude approximately right; the neglected correlation makes
    formal uncertainties slightly too small.

    References
    ----------
    Stoehr, F. et al. 2008, in ASP Conf. Ser. 394, Astronomical Data Analysis Software and
    Systems XVII, 505
    """
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if wave.shape != flux.shape or wave.ndim != 1:
        raise ValueError(
            f"wave and flux must be 1-D of equal length; got {wave.shape}, {flux.shape}"
        )
    if scaling not in ("poisson", "interpolate", "constant"):
        raise ValueError(f"scaling must be 'poisson', 'interpolate' or 'constant'; got {scaling!r}")
    n = wave.size
    usable = np.isfinite(flux)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != flux.shape:
            raise ValueError(f"mask must have shape {flux.shape}, got {m.shape}")
        usable &= ~m
    if usable.sum() < 5:
        raise ValueError("fewer than 5 usable pixels; cannot estimate a noise level")

    cont = None
    if scaling == "poisson":
        if continuum is None:
            raise ValueError("scaling='poisson' needs continuum= (the pre-division continuum)")
        cont = np.asarray(continuum, dtype=np.float64)
        if cont.shape != flux.shape:
            raise ValueError(f"continuum must have shape {flux.shape}, got {cont.shape}")
        if not (np.all(np.isfinite(cont)) and np.all(cont > 0)):
            warnings.warn(
                "scaling='poisson' needs a strictly positive continuum, but this one is not "
                "(non-positive or non-finite values: typically the far blue end of a merged "
                "echelle spectrum). Falling back to scaling='interpolate'.",
                RuntimeWarning,
                stacklevel=2,
            )
            scaling, cont = "interpolate", None

    n_bins = max(1, min(int(n_bins), int(usable.sum()) // 50 or 1))
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    bin_centers: list[float] = []
    bin_sigmas: list[float] = []
    bin_continua: list[float] = []
    for lo, hi in itertools.pairwise(edges):
        sel = usable[lo:hi]
        if sel.sum() < 5:
            continue
        s = der_snr_sigma(flux[lo:hi][sel])
        if not (np.isfinite(s) and s > 0):
            continue
        bin_centers.append(float(np.mean(wave[lo:hi])))
        bin_sigmas.append(s)
        if cont is not None:
            bin_continua.append(float(np.median(cont[lo:hi][sel])))
    if not bin_sigmas:
        raise ValueError(
            "no wavelength bin yielded a finite noise estimate: the spectrum is either "
            "too short, entirely masked, or constant"
        )
    centers = np.asarray(bin_centers, dtype=np.float64)
    sigmas = np.asarray(bin_sigmas, dtype=np.float64)

    if scaling == "poisson":
        assert cont is not None  # set together with scaling == "poisson" above
        # sigma^2 = s^2 / C for one constant s: average s^2 = sigma_bin^2 * C_bin over bins.
        s2 = float(np.median(sigmas**2 * np.asarray(bin_continua, dtype=np.float64)))
        sigma_pix = np.sqrt(s2 / cont)
    elif scaling == "interpolate":
        sigma_pix = np.interp(wave, centers, sigmas) if centers.size > 1 else np.full(n, sigmas[0])
    else:
        sigma_pix = np.full(n, float(np.median(sigmas)))

    ivar = np.where(sigma_pix > 0, 1.0 / np.maximum(sigma_pix, 1e-300) ** 2, 0.0)
    return np.where(np.isfinite(flux) & np.isfinite(ivar), ivar, 0.0)


def normalize(
    wave,
    flux,
    *,
    err=None,
    smooth_angstrom: float | None = None,
    weights=None,
    min_continuum_fraction: float = 0.05,
    **continuum_kwargs,
):
    """Divide out a fitted continuum; return ``(flux_norm, ivar_or_None, continuum)``.

    Composes :func:`fit_continuum` with the division and adds one guard. The ratio diverges
    where the fitted continuum falls to a small fraction of its median (the far blue end of a
    merged echelle spectrum, a dead order, the edge of a chip). Those pixels are marked bad
    (``nan`` flux, ``ivar = 0``) instead of entering the model as large spurious deviations.

    Parameters
    ----------
    wave, flux : array_like
        Wavelengths and the observed (unnormalized) flux, shape ``(n,)``.
    err : array_like, optional
        Observed flux uncertainties in the same units, shape ``(n,)``. When supplied, the
        returned inverse variance is ``(continuum / err) ** 2``, the pipeline's error
        propagated through the division. When ``None`` the second return value is ``None``
        and :func:`estimate_ivar` should be called on the result.
    smooth_angstrom, weights, **continuum_kwargs
        Passed through to :func:`fit_continuum`.
    min_continuum_fraction : float, optional
        Pixels where the continuum is below this fraction of its median are marked bad.
        Default 0.05. Set to 0 to disable the guard.

    Returns
    -------
    flux_norm : numpy.ndarray
        Normalized flux, ``nan`` at guarded pixels.
    ivar : numpy.ndarray or None
        Inverse variance of ``flux_norm``, or ``None`` if ``err`` was not supplied.
    continuum : numpy.ndarray
        The fitted continuum in the input flux units, as required by
        :func:`estimate_ivar` with ``scaling="poisson"``.
    """
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    continuum = fit_continuum(
        wave, flux, smooth_angstrom=smooth_angstrom, weights=weights, **continuum_kwargs
    )
    finite_cont = continuum[np.isfinite(continuum) & (continuum > 0)]
    scale = float(np.median(finite_cont)) if finite_cont.size else 0.0
    bad = ~np.isfinite(continuum) | (continuum <= min_continuum_fraction * scale)

    with np.errstate(divide="ignore", invalid="ignore"):
        flux_norm = np.where(bad, np.nan, flux / continuum)
    flux_norm = np.where(np.isfinite(flux_norm), flux_norm, np.nan)

    ivar = None
    if err is not None:
        e = np.asarray(err, dtype=np.float64)
        if e.shape != flux.shape:
            raise ValueError(f"err must have shape {flux.shape}, got {e.shape}")
        with np.errstate(divide="ignore", invalid="ignore"):
            ivar = (continuum / e) ** 2
        ivar = np.where(np.isfinite(ivar) & (e > 0) & ~bad & np.isfinite(flux_norm), ivar, 0.0)
    return flux_norm, ivar, continuum


# Pixels where the model flux is below this are left out of `continuum_correction`: the
# ratio of data to model is noisy in deep line cores, and a model that reaches zero there
# would make it diverge.
_MIN_MODEL_FLUX = 0.2


def continuum_correction(
    wave,
    flux,
    ivar,
    model,
    *,
    knot_spacing_angstrom: float = 12.0,
    clip: float = 3.0,
    exclude: Iterable[Sequence[float]] = (),
    max_iterations: int = 5,
) -> np.ndarray:
    """A smooth multiplicative correction that brings an epoch's continuum to a model's.

    Fits a cubic least-squares spline to ``flux / model`` with weights
    ``sqrt(ivar) * model`` (the inverse errors of the ratio), with interior knots about
    every ``knot_spacing_angstrom`` of covered wavelength, placed at quantiles of the
    wavelengths used, so that no interval between knots falls in a masked gap. Pixels more
    than ``clip`` robust standard deviations from the spline are rejected and the spline
    refitted, until the set of pixels stops changing. Dividing the epoch by the result
    removes structure that its ratio to the model has on scales above the knot spacing:
    errors of the continuum normalization, such as the residual blaze of an echelle
    spectrograph, which differ from epoch to epoch.

    The knot spacing must be well below the scale of that structure. On PEPSI spectra whose
    normalization left a ripple on scales of 40-80 A (the length of the echelle orders),
    knots every 30 A left most of it in the epochs and knots every 12 A removed it. It must
    also be well above the width of the lines, or the correction follows a mismatch of the
    model in a line.

    Parameters
    ----------
    wave, flux, ivar, model : array_like
        Wavelengths [A], flux, inverse variance and the model flux of the epoch, shape
        ``(n,)``. Pixels with ``ivar = 0``, a non-finite value, or a model flux below 0.2
        (deep line cores) are not used.
    knot_spacing_angstrom : float, optional
        Spacing of the interior knots in angstroms of covered wavelength, in which a gap
        between used pixels counts as at most 1 A. Default 12.
    clip : float, optional
        Rejection threshold in robust standard deviations of the weighted residuals.
        Default 3.
    exclude : iterable of (float, float), optional
        Wavelength ranges ``(lo, hi)`` with ``lo < hi`` left out of the fit (lines whose
        shape the model does not follow, telluric bands); the spline is interpolated
        across them.
    max_iterations : int, optional
        Largest number of rejection passes. Default 5.

    Returns
    -------
    numpy.ndarray
        The correction, shape ``(n,)``, positive. It is held at its edge value outside the
        range of the pixels used.

    Raises
    ------
    ValueError
        If the arrays differ in shape, a parameter or an excluded range is out of range,
        or too few pixels remain to fit a spline with at least one interior knot.
    """
    from scipy.interpolate import LSQUnivariateSpline

    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    ivar = np.asarray(ivar, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    if not (wave.shape == flux.shape == ivar.shape == model.shape) or wave.ndim != 1:
        raise ValueError(
            "wave, flux, ivar and model must be 1-D of equal length; got "
            f"{wave.shape}, {flux.shape}, {ivar.shape}, {model.shape}"
        )
    if not (np.isfinite(knot_spacing_angstrom) and knot_spacing_angstrom > 0.0):
        raise ValueError(f"knot_spacing_angstrom must be positive; got {knot_spacing_angstrom}")
    if not clip > 0.0:
        raise ValueError(f"clip must be positive; got {clip}")
    use = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & np.isfinite(ivar)
        & (ivar > 0.0)
        & np.isfinite(model)
        & (model > _MIN_MODEL_FLUX)
    )
    for lo, hi in exclude:
        if not float(hi) > float(lo):
            raise ValueError(f"each excluded range must be (lo, hi) with lo < hi; got {(lo, hi)!r}")
        use &= ~((wave >= float(lo)) & (wave <= float(hi)))
    x = wave[use]
    y = flux[use] / model[use]
    weight = np.sqrt(ivar[use]) * model[use]
    keep = np.ones(x.size, dtype=bool)
    spline = None
    for _ in range(max(int(max_iterations), 1)):
        kept = x[keep]
        # Covered wavelength: the steps between used pixels, with a gap counted as 1 A.
        covered = float(np.sum(np.minimum(np.diff(kept), 1.0))) if kept.size > 1 else 0.0
        n_knots = int(covered / float(knot_spacing_angstrom))
        if n_knots < 1 or kept.size < 4 * (n_knots + 4):
            raise ValueError(
                f"{kept.size} usable pixels over {covered:.1f} A of covered wavelength are "
                f"too few for a spline with knots every {knot_spacing_angstrom:g} A"
            )
        knots = np.quantile(kept, np.linspace(0.0, 1.0, n_knots + 2)[1:-1])
        try:
            spline = LSQUnivariateSpline(kept, y[keep], knots, w=weight[keep], k=3, ext=3)
        except ValueError as exc:
            raise ValueError(f"the correction spline could not be fitted ({exc})") from exc
        residual = (y - spline(x)) * weight
        scale = 1.4826 * float(np.median(np.abs(residual[keep])))
        if not scale > 0.0:
            break
        new = np.abs(residual) < clip * scale
        if np.array_equal(new, keep):
            break
        keep = new
    assert spline is not None  # the first pass fits it or raises
    correction = np.asarray(spline(wave), dtype=np.float64)
    if not np.all(np.isfinite(correction) & (correction > 0.0)):
        raise ValueError("the correction spline is not positive and finite everywhere")
    return correction


def _replace(epoch: EpochData, **changes) -> EpochData:
    """Rebuild an epoch with some fields replaced, re-running its validation.

    ``dataclasses.replace`` would do the same, but calling the constructor explicitly
    revalidates every derived epoch, so a slicing or masking error raises a ``ValueError``
    naming the field instead of distorting a fit.
    """
    return EpochData(
        wave=changes.get("wave", epoch.wave),
        flux=changes.get("flux", epoch.flux),
        ivar=changes.get("ivar", epoch.ivar),
        bjd=changes.get("bjd", epoch.bjd),
        v_bary=changes.get("v_bary", epoch.v_bary),
        instrument=changes.get("instrument", epoch.instrument),
        mask=changes.get("mask", epoch.mask),
        # Every field of EpochData is listed explicitly. A field added there and not added
        # here is dropped when an epoch is trimmed or masked, with no error raised: `medium`
        # must be copied so that trimming does not discard the wavelength scale.
        medium=changes.get("medium", epoch.medium),
        lsf_sigma_kms=changes.get("lsf_sigma_kms", epoch.lsf_sigma_kms),
    )


def select_region(epoch: EpochData, wave_min: float, wave_max: float) -> EpochData:
    """Return the contiguous slice of ``epoch`` inside ``[wave_min, wave_max]``.

    Reduces a full echelle spectrum to the region to be modelled. Cutting the ends off a
    spectrum is safe. Removing interior pixels distorts the bin edges of the remaining
    pixels, so interior pixels are masked with :func:`mask_ranges`.

    Parameters
    ----------
    epoch : EpochData
        The epoch to cut.
    wave_min, wave_max : float
        Inclusive bounds in the epoch's wavelength units.

    Returns
    -------
    EpochData
        A new epoch holding only the pixels inside the bounds.

    Raises
    ------
    ValueError
        If the bounds are out of order, or fewer than 2 pixels fall inside them.
    """
    if not wave_max > wave_min:
        raise ValueError(f"need wave_min < wave_max; got {wave_min} and {wave_max}")
    sel = (epoch.wave >= wave_min) & (epoch.wave <= wave_max)
    if sel.sum() < 2:
        raise ValueError(
            f"[{wave_min}, {wave_max}] contains {int(sel.sum())} pixels of this epoch, which "
            f"spans {epoch.wave[0]:.2f} to {epoch.wave[-1]:.2f}"
        )
    return _replace(
        epoch,
        wave=epoch.wave[sel],
        flux=epoch.flux[sel],
        ivar=epoch.ivar[sel],
        mask=None if epoch.mask is None else epoch.mask[sel],
    )


def mask_ranges(epoch: EpochData, ranges: Iterable[Sequence[float]]) -> EpochData:
    """Zero the inverse variance inside each ``(lambda_min, lambda_max)`` range.

    The pixels remain in the array. albireo takes bin edges at the midpoints between
    neighbouring samples (:func:`albireo.operators.bin_edges_from_centers`), so deleting an
    interior block widens the two pixels beside the gap by half the gap each. The rebin row
    support is a maximum over pixels and it sets the solver half-bandwidth, whose cost is
    quadratic, so a few deleted telluric windows can multiply the runtime of the whole fit.
    Setting ``ivar = 0`` leaves the sampling regular and the bandwidth unchanged;
    :func:`albireo.forward.build_problem` warns if it detects a deleted block.

    Parameters
    ----------
    epoch : EpochData
        The epoch to mask.
    ranges : iterable of (float, float)
        Inclusive wavelength ranges to mask. Ranges may overlap, and ranges entirely
        outside the epoch are ignored.

    Returns
    -------
    EpochData
        A new epoch with ``ivar = 0`` inside the ranges.

    Raises
    ------
    ValueError
        If a range is malformed, or if the masking would leave no usable pixel.
    """
    ivar = epoch.ivar.copy()
    for r in ranges:
        lo, hi = (float(v) for v in r)
        if not hi > lo:
            raise ValueError(
                f"each range must be (lambda_min, lambda_max) with min < max; got {r!r}"
            )
        ivar[(epoch.wave >= lo) & (epoch.wave <= hi)] = 0.0
    if not np.any(ivar > 0):
        raise ValueError("masking removed every pixel of this epoch")
    return _replace(epoch, ivar=ivar)


def mask_tellurics(
    epoch: EpochData,
    bands: Iterable[Sequence[float]] | None = None,
    *,
    velocity_pad_kms: float = 40.0,
) -> EpochData:
    """Zero the inverse variance in the telluric absorption bands.

    Convenience wrapper over :func:`mask_ranges` with :data:`TELLURIC_BANDS` as the
    default band list, widened by ``velocity_pad_kms`` on each side.

    The bands are quoted in the topocentric frame, where telluric lines are static, while
    the spectrum is usually delivered in the barycentric frame, where they move by up to
    ~30 km/s over a year. Padding by more than that keeps the mask valid at every epoch
    without shifting it per exposure.

    Masking discards the information. albireo can instead model the tellurics as an extra
    component (``telluric=True`` in :func:`albireo.forward.build_problem`), which is
    preferable when the required region is telluric-contaminated. Masking is appropriate when
    the contamination is variable (water vapour) rather than static.

    Parameters
    ----------
    epoch : EpochData
        The epoch to mask.
    bands : iterable of (float, float), optional
        Wavelength ranges to mask. Default :data:`TELLURIC_BANDS`.
    velocity_pad_kms : float, optional
        Extra width added to each side of every band, expressed as a velocity.
        Default 40 km/s, which covers the full barycentric velocity range with margin.

    Returns
    -------
    EpochData
        A new epoch with the telluric windows zero-weighted. Returned unchanged if no
        band overlaps this epoch.
    """
    bands = TELLURIC_BANDS if bands is None else bands
    pad = float(velocity_pad_kms) / _C_KMS
    padded = [(lo * (1.0 - pad), hi * (1.0 + pad)) for lo, hi in (tuple(b) for b in bands)]
    overlapping = [(lo, hi) for lo, hi in padded if hi >= epoch.wave[0] and lo <= epoch.wave[-1]]
    if not overlapping:
        return epoch
    return mask_ranges(epoch, overlapping)


def mask_spikes(
    epoch: EpochData, *, threshold: float = 6.0, window: int = 21, both_sides: bool = False
) -> EpochData:
    """Zero the inverse variance at cosmic-ray spikes and other isolated outliers.

    Each pixel is compared with a running median and rejected if it lies more than
    ``threshold`` robust sigma above it. Rejection is one-sided by default: in an
    absorption-line spectrum a narrow feature above the local continuum is almost always a
    detector artefact, whereas a narrow feature below it is usually a line, so a symmetric
    clip removes sharp line cores.

    Parameters
    ----------
    epoch : EpochData
        The epoch to clean. Already-masked pixels stay masked.
    threshold : float, optional
        Rejection threshold in robust sigma. Default 6.0.
    window : int, optional
        Running-median width in pixels; forced odd. Default 21. It should be comfortably
        wider than a spike and narrower than a line.
    both_sides : bool, optional
        Also reject downward outliers. Default False. Safe only for a spectrum whose lines
        are all much broader than ``window``; otherwise line cores are clipped.

    Returns
    -------
    EpochData
        A new epoch with the outliers zero-weighted.

    Notes
    -----
    Cosmic rays are better removed before extraction, where their spatial profile identifies
    them. This function is the fallback for archival 1-D spectra, where that is no longer
    possible.
    """
    window = int(window) | 1
    if window < 3:
        raise ValueError(f"window must be at least 3 pixels; got {window}")
    good = epoch.good
    if good.sum() < window:
        return epoch
    flux = np.where(good, epoch.flux, np.nan)
    half = window // 2
    padded = np.pad(flux, half, mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows inside big gaps
        smooth = np.nanmedian(strided, axis=-1)
    resid = flux - smooth
    sigma = _robust_sigma(resid[np.isfinite(resid)])
    if not sigma > 0:
        return epoch
    bad = np.isfinite(resid) & (resid > threshold * sigma)
    if both_sides:
        bad |= np.isfinite(resid) & (resid < -threshold * sigma)
    if not bad.any():
        return epoch
    ivar = np.where(bad, 0.0, epoch.ivar)
    if not np.any(ivar > 0):
        raise ValueError("spike rejection removed every pixel of this epoch")
    return _replace(epoch, ivar=ivar)


def _masked_runs(bad: np.ndarray) -> list[tuple[int, int]]:
    """``(start, stop)`` of each run of True values, ``stop`` exclusive."""
    edges = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def mask_flux_gaps(epoch: EpochData, *, min_run: int = 8, warn: bool = True) -> EpochData:
    """Zero the inverse variance across contiguous runs of non-positive flux.

    A detector gap is not a measurement of zero flux, but nothing in a Phase 3 product states
    that. The pixels are finite and unflagged, and for the many pipelines that provide no
    error array the inverse variance estimated from the local scatter is small across a flat
    run of zeros, so the gap is weighted like good data. HARPS spectra are one case: the two
    CCDs leave a 32.9 A gap at 5304.7-5337.6 A, so a 100 A window placed across it is 33%
    zeros at full weight, which yields disentangled component spectra with negative flux.

    Only contiguous runs are treated as gaps. :attr:`RawSpectrum.bad_pixels` does not treat
    zero flux as missing: a single zero may be a saturated core, a clipped cosmic ray or a
    valid measurement, and no general rule separates those from a gap. A measured spectrum
    is not exactly zero for eight consecutive pixels. Isolated non-positive pixels are left
    unmasked, as in the readers' quality-flag policy: an ambiguous value is not interpreted.

    Parameters
    ----------
    epoch : EpochData
        The epoch to clean. Already-masked pixels stay masked.
    min_run : int, optional
        Shortest run of non-positive flux treated as a gap. Default 8.
    warn : bool, optional
        Emit a :class:`RuntimeWarning` naming the wavelength ranges removed. Default True, so
        that the loss of a large fraction of a window is reported.

    Returns
    -------
    EpochData
        A new epoch with the gaps zero-weighted, or ``epoch`` unchanged if there are none.
    """
    if int(min_run) < 2:
        raise ValueError(f"min_run must be at least 2 pixels; got {min_run}")
    flux = np.asarray(epoch.flux)
    nonpositive = np.isfinite(flux) & (flux <= 0.0)
    if not nonpositive.any():
        return epoch

    runs = [(a, b) for a, b in _masked_runs(nonpositive) if b - a >= int(min_run)]
    if not runs:
        return epoch

    wave = np.asarray(epoch.wave)
    gap = np.zeros_like(nonpositive)
    ranges = []
    for start, stop in runs:
        gap[start:stop] = True
        ranges.append((float(wave[start]), float(wave[stop - 1])))

    ivar = np.where(gap, 0.0, epoch.ivar)
    if not np.any(ivar > 0):
        raise ValueError("every pixel of this epoch lies inside a zero-flux gap")
    if warn:
        where = ", ".join(f"{lo:.2f}-{hi:.2f} A" for lo, hi in ranges[:4])
        if len(ranges) > 4:
            where += f", and {len(ranges) - 4} more"
        warnings.warn(
            f"{gap.sum()} pixels ({gap.mean():.1%} of this epoch) lie in contiguous runs of "
            f"non-positive flux and have been zero-weighted: {where}. These are detector or "
            "order gaps, not measurements of zero flux; left in they are weighted like data.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _replace(epoch, ivar=ivar)


def _lattice_offset(wave: np.ndarray, reference: np.ndarray, step: float | None) -> int:
    """The index on ``reference``'s lattice of ``wave[0]``, rounded to a whole pixel.

    With ``step`` given this is ``round((wave[0] - reference[0]) / step)``. Otherwise it is
    the fractional index of ``wave[0]`` among the reference wavelengths, or minus that of
    ``reference[0]`` among ``wave`` when the epoch starts first, which follows a lattice
    whose step changes along it (a log-uniform grid). An epoch that does not reach the
    reference gets an offset that leaves no overlap.
    """
    if step is not None:
        return round((wave[0] - reference[0]) / step)
    if wave[0] >= reference[0]:
        if wave[0] > reference[-1]:
            return int(reference.size)
        return round(float(np.interp(wave[0], reference, np.arange(reference.size))))
    if reference[0] > wave[-1]:
        return -int(wave.size)
    return -round(float(np.interp(reference[0], wave, np.arange(wave.size))))


def share_wavelength_grid(
    epochs: Sequence[EpochData], *, atol_kms: float = 0.05, step: float | None = None
) -> list[EpochData]:
    """Put epochs that differ only by sub-pixel offsets onto one shared wavelength array.

    Some pipelines apply the barycentric correction by shifting before resampling onto a fixed
    step, so each exposure has its own grid. The ESO Phase 3 FEROS spectra of one target have
    the same 0.03 A step but start wavelengths spread over 0.78 A and lengths differing by
    tens of pixels. albireo handles that correctly by giving each distinct grid its own rebin
    operator (:func:`albireo.forward._epoch_groups`), but every group's assembly pre-pass is
    part of the same compiled graph. One group per exposure was measured at several times
    the memory of one shared grid, with a much larger program to compile.

    When the grids agree to well within a pixel, this function replaces them with one. The
    operation is a relabelling, not a resampling: no flux value is modified and the ``ivar``
    model stays diagonal. Only the wavelength assigned to each sample changes, by at most
    ``atol_kms``. Epochs are trimmed to their common overlap, so the shared array is exact
    for all of them.

    Alignment is by index, not by value comparison: the first sample of each epoch is placed
    on the reference epoch's lattice (the first epoch of its instrument) at its fractional
    pixel index, rounded, and every later sample follows by its own index. With ``step``
    given the index is ``round((wave[0] - reference[0]) / step)``. A search for the nearest
    wavelength at a window edge can be off by one native pixel, and one FEROS pixel is
    1.4 km/s, a radial-velocity error far larger than the sub-pixel mismatch being
    corrected. The fractional index follows a lattice whose step changes along it, such as
    a log-uniform grid, on which a single median step misplaces an epoch that starts
    several pixels later.

    Parameters
    ----------
    epochs : sequence of EpochData
        Epochs to align. Those with different ``instrument`` keys are aligned within
        their own key and left independent of each other.
    atol_kms : float, optional
        Largest wavelength relabelling tolerated, as a velocity. Default 0.05 km/s
        (about 1/50 of a FEROS pixel and 1/50 of its LSF sigma). Raises if exceeded.
    step : float, optional
        Wavelength step of the common grid, for a lattice of constant step. Default: the
        reference epoch's own wavelengths. Only used for the index alignment.

    Returns
    -------
    list of EpochData
        Aligned epochs, in the input order. Each shares one ``wave`` array object with the
        others of its instrument, which makes them a single operator group.

    Raises
    ------
    ValueError
        If an instrument's epochs do not share a common step, if the residual mismatch
        exceeds ``atol_kms`` (the message quotes the measured value), or if the common
        overlap is shorter than 2 pixels.

    Examples
    --------
    >>> import numpy as np
    >>> from albireo.data import EpochData
    >>> a = EpochData(wave=np.arange(4000.0, 4010.0, 0.03), flux=np.ones(334),
    ...               ivar=np.ones(334), bjd=2453000.0)
    >>> b = EpochData(wave=np.arange(4000.0, 4010.0, 0.03) + 1e-5, flux=np.ones(334),
    ...               ivar=np.ones(334), bjd=2453040.0)
    >>> aligned = share_wavelength_grid([a, b])
    >>> aligned[0].wave is aligned[1].wave
    True
    """
    if len(epochs) == 0:
        raise ValueError("share_wavelength_grid needs at least one epoch")
    tol_factor = float(atol_kms) / _C_KMS
    out: list[EpochData | None] = [None] * len(epochs)

    by_instrument: dict[str, list[int]] = {}
    for k, epoch in enumerate(epochs):
        by_instrument.setdefault(epoch.instrument, []).append(k)

    for instrument, members in by_instrument.items():
        ref = epochs[members[0]]
        dlam = None if step is None else float(step)
        if dlam is not None and not dlam > 0:
            raise ValueError(f"instrument {instrument!r}: non-positive wavelength step {dlam}")

        # Offset of each epoch's first sample from the reference lattice, in whole pixels.
        offsets = [_lattice_offset(epochs[k].wave, ref.wave, dlam) for k in members]
        # Common index window on the reference lattice: [start, stop).
        start = max(offsets)
        stop = min(off + epochs[k].n_pixels for off, k in zip(offsets, members, strict=True))
        if stop - start < 2:
            raise ValueError(
                f"instrument {instrument!r}: the epochs overlap in {max(stop - start, 0)} "
                "pixel(s) once aligned; they are not the same spectral window"
            )

        shared = ref.wave[start - offsets[0] : stop - offsets[0]].copy()
        worst, worst_epoch = 0.0, 0
        for off, k in zip(offsets, members, strict=True):
            lo, hi = start - off, stop - off
            deviation = float(np.max(np.abs(epochs[k].wave[lo:hi] - shared)))
            if deviation > worst:
                worst, worst_epoch = deviation, k
        if worst > tol_factor * shared[0]:
            raise ValueError(
                f"instrument {instrument!r}: epoch {worst_epoch} differs from the shared grid "
                f"by up to {worst:.3e} A ({worst / shared[0] * _C_KMS:.4f} km/s), more "
                f"than atol_kms={atol_kms}. These are genuinely different wavelength "
                "solutions, not sub-pixel offsets: leave them unaligned (albireo gives each "
                "its own rebin operator) rather than relabelling real differences away."
            )
        for off, k in zip(offsets, members, strict=True):
            lo, hi = start - off, stop - off
            epoch = epochs[k]
            out[k] = _replace(
                epoch,
                wave=shared,
                flux=epoch.flux[lo:hi],
                ivar=epoch.ivar[lo:hi],
                mask=None if epoch.mask is None else epoch.mask[lo:hi],
            )
    return [e for e in out if e is not None]


_SMOOTHING_REACH = 5.0
"""Truncation of the smoothing kernel, in kernel sigmas. A pixel closer than this to a long
masked run or to an end of the epoch is masked in the result, so every kept pixel has the
whole kernel. The Gaussian weight beyond five sigma is 6e-7 of the total. At four sigma the
weight at the edge, 3e-5 of the total, entered or left a row's kernel with the rounding of
the pixel velocities wherever the edge fell on a pixel, and the equivalent width of a line
changed by 2e-6."""


def _smoothed(
    epoch: EpochData, sigma_kms: float, from_sigma_kms: float, fill_max: int
) -> tuple[EpochData, dict]:
    """:func:`smooth_epoch` with its diagnostics: the kernel width in km/s, the median of
    ``sum(k^2)`` over the kept pixels, and the numbers of good, kept and filled pixels."""
    sigma_to = float(sigma_kms)
    sigma_from = float(from_sigma_kms)
    if not (np.isfinite(sigma_to) and sigma_to > 0.0):
        raise ValueError(f"the target width must be a positive number of km/s; got {sigma_kms}")
    if not (np.isfinite(sigma_from) and sigma_from > 0.0):
        raise ValueError(f"the epoch's own width must be positive; got {from_sigma_kms}")
    if sigma_to < sigma_from * (1.0 - 1e-9):
        raise ValueError(
            f"the target width {sigma_to:.4f} km/s is narrower than the epoch's own "
            f"{sigma_from:.4f} km/s; a spectrum cannot be sharpened by a convolution"
        )
    if int(fill_max) < 0:
        raise ValueError(f"fill_max must be a non-negative number of pixels; got {fill_max}")
    good = epoch.good
    n_good = int(good.sum())
    if n_good == 0:
        raise ValueError("the epoch has no unmasked pixel to smooth")
    variance = sigma_to**2 - sigma_from**2
    if variance <= (1e-6 * sigma_to) ** 2:
        info = {"kernel_sigma_kms": 0.0, "sum_k2": 1.0, "good": n_good, "kept": n_good}
        return _replace(epoch, lsf_sigma_kms=sigma_to), {**info, "filled": 0}
    sigma_s = float(np.sqrt(variance))

    u = _C_KMS * np.log(np.asarray(epoch.wave, dtype=np.float64))
    n = u.size
    flux = np.where(good, epoch.flux, 0.0)
    long_bad = ~good
    filled = np.zeros(n, dtype=bool)
    for start, stop in _masked_runs(~good):
        if stop - start <= int(fill_max) and start > 0 and stop < n:
            filled[start:stop] = True
            long_bad[start:stop] = False
    if filled.any():
        flux[filled] = np.interp(u[filled], u[good], np.asarray(epoch.flux)[good])

    # The kernel of every pixel, truncated at the reach, weighted by the pixel width in u and
    # normalised over the pixels it covers.
    reach = _SMOOTHING_REACH * sigma_s
    width = np.gradient(u)
    # The windows are found with a margin, and the support is then cut on the difference
    # itself: u_i - u_j is exactly antisymmetric in floating point, so pixel j is in the
    # kernel of pixel i exactly when i is in the kernel of j.
    lo = np.searchsorted(u, u - reach * (1.0 + 1e-9), side="left")
    hi = np.searchsorted(u, u + reach * (1.0 + 1e-9), side="right")
    counts = hi - lo
    rows = np.repeat(np.arange(n), counts)
    cols = np.repeat(lo, counts) + (
        np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    )
    offset = u[rows] - u[cols]
    weights = np.where(
        np.abs(offset) <= reach, np.exp(-0.5 * (offset / sigma_s) ** 2) * width[cols], 0.0
    )
    weights /= np.bincount(rows, weights=weights, minlength=n)[rows]
    smoothed = np.bincount(rows, weights=weights * flux[cols], minlength=n)
    sum_k2 = np.bincount(rows, weights=weights**2, minlength=n)

    # A pixel keeps its value only where its whole kernel lies on usable pixels.
    keep = good & (u - u[0] >= reach) & (u[-1] - u >= reach)
    bad_u = u[long_bad]
    if bad_u.size:
        j = np.searchsorted(bad_u, u)
        left = np.where(j > 0, u - bad_u[np.maximum(j - 1, 0)], np.inf)
        right = np.where(j < bad_u.size, bad_u[np.minimum(j, bad_u.size - 1)] - u, np.inf)
        keep &= np.minimum(left, right) > reach
    if not keep.any():
        raise ValueError(
            f"no pixel of this epoch keeps a whole smoothing kernel of {sigma_s:.2f} km/s "
            f"(truncated at {_SMOOTHING_REACH:g} sigma): the epoch is shorter than the kernel, "
            "or masked too densely"
        )
    out = _replace(
        epoch,
        flux=np.where(keep, smoothed, 1.0),
        ivar=np.where(keep, epoch.ivar, 0.0),
        mask=None,
        lsf_sigma_kms=sigma_to,
    )
    info = {
        "kernel_sigma_kms": sigma_s,
        "sum_k2": float(np.median(sum_k2[keep])),
        "good": n_good,
        "kept": int(keep.sum()),
        "filled": int(filled.sum()),
    }
    return out, info


def smooth_epoch(
    epoch: EpochData,
    sigma_kms: float,
    *,
    from_sigma_kms: float | None = None,
    fill_max: int = 3,
) -> EpochData:
    """Convolve an epoch to a broader Gaussian line-spread function.

    The label fit compares the epochs with library spectra that are already broadened to
    the library's resolving power, and a library spectrum cannot be sharpened. Where the
    library is broader than an instrument, the epochs are broadened instead
    (:meth:`albireo.Fit.match_labels` with ``smoothing=``). The flux is convolved with a
    Gaussian of ``sigma_s = sqrt(sigma_kms^2 - from_sigma_kms^2)``, so that the epoch's
    line-spread function becomes a Gaussian of ``sigma_kms``.

    The kernel is evaluated in velocity, ``u = c ln(lambda)``, on the native pixels, and
    each pixel is weighted by its width in ``u``, so a non-uniform sampling such as merged
    echelle orders is integrated correctly. The kernel is truncated at five ``sigma_s`` and
    normalised over the pixels it covers. Masked pixels are treated so that the kernel of
    every pixel kept in the result is whole and symmetric, since an asymmetric kernel
    shifts the lines it contains. A run of at most ``fill_max`` masked pixels (a cosmic-ray
    hit) is filled by linear interpolation in velocity before the convolution and stays
    masked in the result. A pixel whose kernel reaches a longer masked run or an end of the
    epoch is masked in the result.

    The result keeps the input inverse variances, with zeros where it is masked. The
    convolution correlates the noise over the kernel width and reduces its variance per
    pixel by ``sum(k^2)``. A diagonal likelihood with the reduced variances counts the
    information of a signal that is smooth on the kernel scale ``1 / sum(k^2)`` times, the
    mis-specification found in the ``"matched"`` comparison (D55). With the input variances
    such a signal is counted once and the information above the kernel's cutoff less than
    once, so curvature errors computed from them are upper bounds on the noise part of the
    error. The chi-square of a perfect model is then about ``sum(k^2)`` per pixel, not one
    (``docs/math.md`` §9.2b).

    Parameters
    ----------
    epoch : EpochData
        The epoch to broaden.
    sigma_kms : float
        Gaussian sigma of the result's line-spread function, in km/s.
    from_sigma_kms : float, optional
        Gaussian sigma of the epoch's own line-spread function, in km/s. Default:
        :attr:`~albireo.data.EpochData.lsf_sigma_kms`.
    fill_max : int, optional
        Longest run of masked pixels filled before the convolution. Default 3.

    Returns
    -------
    EpochData
        The broadened epoch on the same pixels, declaring ``lsf_sigma_kms = sigma_kms``.

    Raises
    ------
    ValueError
        If the epoch declares no width and none is given, if ``sigma_kms`` is narrower than
        the epoch's own width, or if no pixel keeps a whole kernel.

    Examples
    --------
    >>> import numpy as np
    >>> from albireo.data import EpochData
    >>> wave = 5000.0 * np.exp(np.arange(2000) * 1.0 / 299_792.458)  # 1 km/s pixels
    >>> line = 1.0 - 0.5 * np.exp(-0.5 * (np.arange(2000) - 1000.0) ** 2 / 3.0**2)
    >>> epoch = EpochData(wave=wave, flux=line, ivar=np.ones(2000), bjd=0.0, lsf_sigma_kms=3.0)
    >>> broad = smooth_epoch(epoch, 5.0)
    >>> broad.lsf_sigma_kms, round(float(broad.flux.min()), 3)
    (5.0, 0.7)
    """
    sigma_from = epoch.lsf_sigma_kms if from_sigma_kms is None else from_sigma_kms
    if sigma_from is None:
        raise ValueError(
            "the epoch declares no line-spread width (EpochData.lsf_sigma_kms); pass "
            "from_sigma_kms, the width the epoch was taken at"
        )
    return _smoothed(epoch, sigma_kms, sigma_from, fill_max)[0]
