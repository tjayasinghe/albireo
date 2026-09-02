"""Turning a reduced spectrum into an :class:`~albireo.data.EpochData`.

Pipeline-reduced spectra are not what the disentangling model consumes. The model is
``1 + sum_i l_i d_i`` with ``d_i`` a deviation from a unit continuum (``docs/math.md`` §1.4),
and every weight is an inverse variance, so an archival spectrum requires three operations
before it can enter a :class:`~albireo.data.Dataset`.

1. Continuum normalization (:func:`fit_continuum`, :func:`normalize`). Merged echelle spectra
   carry the blaze and the instrument response and are routinely delivered unnormalized; ESO
   Phase 3 marks this with ``CONTNORM = False``. albireo's response term ``r_j`` is fixed at
   build time rather than inferred, so the normalization done here is the normalization the
   model uses.
2. An inverse variance (:func:`estimate_ivar`). Many archival products ship no usable error
   array: the ESO Phase 3 FEROS spectra, for instance, carry an ``ERR`` column that is
   entirely ``NaN``. The noise must then be estimated from the spectrum itself, in a way that
   deep absorption lines cannot bias.
3. Region selection and masking (:func:`select_region`, :func:`mask_ranges`,
   :func:`mask_tellurics`, :func:`mask_spikes`). Masking is always ``ivar = 0`` rather than
   deletion; :func:`mask_ranges` records the reason.

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
]


# Main telluric absorption complexes in the optical, as (lambda_min, lambda_max) in
# Angstrom (air). The O2 gamma, B and A bands are sharp and deep; the H2O complexes are
# forests of weaker lines whose strength tracks the precipitable water vapour, so they
# vary between exposures in a way no static telluric component can absorb. The ranges are
# generous: masking loses one pixel per pixel masked, while a telluric line left in the
# data produces a spurious component. Not exhaustive below 5800 A, where telluric
# absorption is weak enough to ignore at typical SNR.
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


# Knots per half-power smoothing length. Eight is comfortably more than the two a
# piecewise-linear basis needs to represent a feature of that width, and it fixes the
# penalty weight at (8 / 2pi)^4 ~ 2.6 (see _KnotBasis). A per-pixel smoother would instead
# need lam ~ (L_pixels / 2pi)^4, which is ~1e12 at the 5000-pixel smoothing lengths a merged
# echelle spectrum calls for; there the weight term is lost to rounding and the nominally
# positive-definite matrix factorizes as singular.
_KNOTS_PER_LENGTH = 8.0


class _KnotBasis:
    """Piecewise-linear basis on uniform knots, with a curvature-penalized fit.

    Fits ``z(p)``, linear between ``m`` uniformly spaced knots, by minimizing

        ``sum_p w_p (y_p - z(p))^2 + lam * sum_k (D2 z_knots)_k^2``

    The normal matrix is ``B^T W B + lam D2^T D2``. ``B`` has two nonzeros per row at adjacent
    knots, so ``B^T W B`` is tridiagonal and the sum is pentadiagonal: one ``solveh_banded``,
    ``O(m)``. Reducing the unknowns from pixels to knots keeps the system well conditioned at
    long smoothing lengths (see :data:`_KNOTS_PER_LENGTH`) and keeps the iteration cheap; a
    continuum with a 150 A scale does not require 20000 free parameters.
    """

    def __init__(self, n: int, smooth_pixels: float):
        n_knots = round(n * _KNOTS_PER_LENGTH / smooth_pixels) + 1
        self.m = m = int(np.clip(n_knots, 5, n))
        u = np.linspace(0.0, m - 1.0, n)
        self.i0 = np.clip(np.floor(u).astype(np.intp), 0, m - 2)
        self.frac = u - self.i0
        # Half-power scale in knots; ~_KNOTS_PER_LENGTH except where m hit a clamp.
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
    throughput, extinction) and on a merged echelle spectrum it is large: over 3850-4750 A a
    FEROS exposure of a B star falls by a factor of 20. A curvature penalty applied to the
    flux itself cannot track that, since a stiff smoother lags a steep exponential and the
    normalized spectrum comes out 30% wrong. In the log the same fall is nearly a straight
    line, and straight lines lie in the null space of the second-difference penalty, so they
    cost nothing to represent. Absorption depths are additive in the log, so the rejection
    thresholds below are fractional.

    The fit proceeds in two stages, both penalized least-squares fits on a knot grid
    (:class:`_KnotBasis`).

    Stage 1, asymmetric envelope. The fit is iteratively reweighted with weight ``asymmetry``
    above the current curve and ``1 - asymmetry`` below it, which moves the curve up out of
    the absorption lines. On its own it sits about one noise sigma high, because it fits the
    upper envelope of the noise as well as of the lines.

    Stage 2, asymmetric sigma clipping. Pixels more than ``low_reject`` sigma below or
    ``high_reject`` sigma above the curve are given zero weight and the smoother is refit on
    the survivors, with sigma re-estimated robustly at each pass. The thresholds are
    asymmetric because absorption lines are deep and one-sided while cosmic rays are high and
    rare. The surviving pixels are then continuum, so the curve centres on them instead of on
    their upper envelope.

    Parameters
    ----------
    wave : array_like
        Wavelengths, shape ``(n,)``, strictly increasing. Used only to convert
        ``smooth_angstrom`` into pixels (via the median spacing); the fit itself runs on the
        pixel index, so on a non-uniform grid the smoothing scale is correct on average
        rather than everywhere.
    flux : array_like
        Observed flux, shape ``(n,)``. Non-finite and non-positive samples are given zero
        weight, since the log is undefined there. They need not be removed beforehand, and
        the smoother interpolates across them. The returned continuum is therefore strictly
        positive everywhere.
    smooth_angstrom : float, optional
        Half-power smoothing scale. Structure much broader than this is continuum; structure
        much narrower is signal. Default: one eighth of the wavelength span. It must be
        wider than the widest line to be preserved, or the continuum absorbs that line.
        Broad lines are handled mainly by the rejection stage rather than by stiffness, so a
        scale that follows the blaze is preferable.
    weights : array_like, optional
        Prior per-pixel weights, shape ``(n,)``, non-negative. A 0/1 mask keeps known-bad
        pixels out of the fit entirely. Default: all ones. These weight the log residuals,
        so passing ``ivar`` (which weights flux residuals) is not statistically exact; use
        it as a mask.
    asymmetry : float, optional
        Stage-1 weight given to pixels above the curve; ``1 - asymmetry`` below.
        Must lie in ``(0.5, 1)``. Default 0.97.
    low_reject, high_reject : float, optional
        Stage-2 clipping thresholds in robust sigma, below and above the curve.
        Defaults 1.5 and 3.0.
    envelope_iterations, clip_iterations : int, optional
        Iteration counts for the two stages; both stop early once the fit stops moving.
        Defaults 12 and 6.

    Returns
    -------
    numpy.ndarray
        The continuum, shape ``(n,)``, float64, strictly positive.

    Raises
    ------
    ValueError
        If the arrays disagree in shape, fewer than 5 samples are supplied, no pixel has
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
    only through its curvature, and the median makes even that contribution robust: the
    estimate reflects the noise rather than the lines. Skipping the immediate neighbours also
    limits the bias when a pipeline has resampled the spectrum and correlated adjacent pixels,
    as a merged, rebinned echelle product has.

    This is the recipe ESO uses for the ``SNR`` keyword of its Phase 3 spectra, so an estimate
    made here is directly comparable with the archive's own.

    Parameters
    ----------
    flux : array_like
        Flux samples, shape ``(n,)`` with ``n >= 5``. Non-finite samples are dropped.

    Returns
    -------
    float
        The noise standard deviation in the units of ``flux``; ``nan`` if fewer than
        five finite samples survive.

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

    For archival products with no usable error array. The noise scale is measured directly
    from the data with :func:`der_snr_sigma` (Stoehr et al. 2008) in ``n_bins`` wavelength
    bins, and those binned estimates are then turned into a per-pixel ``sigma(lambda)`` by one
    of the ``scaling`` rules below. The estimate is binned before it is smoothed because a
    per-pixel noise estimate is itself noisy, and noisy weights bias a maximum-likelihood fit,
    whereas a smooth ``sigma(lambda)`` does not.

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
          shape and averages every bin into it. Falls back to ``"interpolate"``, with a
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
    The result estimates the noise of the delivered product and inherits whatever the
    reduction pipeline did to it. Resampling in particular correlates neighbouring pixels, so
    the diagonal inverse-variance model albireo uses is an approximation for any archival
    product that has been rebinned onto a common wavelength grid (``internal/design.md`` D4 gives
    the same reason for not resampling in albireo, but an archive may already have done it).
    The lag-2 stencil keeps the amplitude approximately right; the neglected correlation makes
    formal uncertainties mildly optimistic.

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

    A composition of :func:`fit_continuum` with the division, plus one guard: where the fitted
    continuum falls to a small fraction of its own median (the far blue end of a merged
    echelle spectrum, a dead order, the edge of a chip) the ratio diverges, so those pixels are
    marked bad (``nan`` flux, ``ivar = 0``) instead of entering the model as large spurious
    deviations.

    Parameters
    ----------
    wave, flux : array_like
        Wavelengths and the observed (unnormalized) flux, shape ``(n,)``.
    err : array_like, optional
        Observed flux uncertainties in the same units, shape ``(n,)``. When supplied, the
        returned inverse variance is ``(continuum / err) ** 2``, the pipeline's own error
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
        The fitted continuum in the input flux units. It is the array required by
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


def _replace(epoch: EpochData, **changes) -> EpochData:
    """Rebuild an epoch with some fields replaced, re-running its validation.

    ``dataclasses.replace`` would do the same, but going through the constructor explicitly
    revalidates every derived epoch, so a slicing or masking error surfaces as a
    ``ValueError`` naming the field rather than as a distorted fit.
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
        # here is dropped when an epoch is trimmed or masked, without any error: `medium`
        # must be carried through so that trimming does not discard the wavelength scale.
        medium=changes.get("medium", epoch.medium),
    )


def select_region(epoch: EpochData, wave_min: float, wave_max: float) -> EpochData:
    """Return the contiguous slice of ``epoch`` inside ``[wave_min, wave_max]``.

    Cutting the ends off a spectrum is safe, unlike removing pixels from its interior, which
    distorts the bin edges of the surviving pixels (see :func:`mask_ranges`). This function
    reduces a full echelle spectrum to the region to be modelled; interior removals go
    through :func:`mask_ranges`.

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
    interior block makes the two pixels bracketing the hole absorb half the gap each. The
    rebin row support is a maximum over pixels and it sets the solver half-bandwidth, whose
    cost is quadratic, so a few deleted telluric windows can multiply the runtime of the whole
    fit. Setting ``ivar = 0`` leaves the sampling regular and the bandwidth unchanged;
    :func:`albireo.forward.build_problem` warns if it detects the other pattern.

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

    The padding is required because the bands are quoted in the topocentric frame, where
    telluric lines are static, while the spectrum is usually delivered in the barycentric
    frame, where they move by up to ~30 km/s over a year. Padding by more than that keeps the
    mask valid at every epoch without shifting it per exposure.

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
        Default 40 km/s, which covers the full barycentric swing with margin.

    Returns
    -------
    EpochData
        A new epoch with the telluric windows zero-weighted. Returned unchanged if no
        band overlaps this epoch.
    """
    bands = TELLURIC_BANDS if bands is None else bands
    pad = float(velocity_pad_kms) / 299_792.458
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


def mask_flux_gaps(epoch: EpochData, *, min_run: int = 8, warn: bool = True) -> EpochData:
    """Zero the inverse variance across contiguous runs of non-positive flux.

    A detector gap is not a measurement of zero flux, but nothing in a Phase 3 product states
    that. The pixels arrive finite and unflagged, and for the many pipelines that ship no
    error array the inverse variance estimated from the local scatter is small across a flat
    run of zeros, so the gap is weighted like good data. HARPS spectra are one case: the two
    CCDs leave a 32.9 A hole at 5304.7-5337.6 A, and a 100 A window placed across it arrives
    33% zeros at full weight, which yields disentangled component spectra with negative flux.

    Only contiguous runs are treated as gaps. :attr:`RawSpectrum.bad_pixels` does not treat
    zero flux as missing, since a single zero may be a saturated core, a clipped cosmic ray or
    a genuine measurement, and no generic rule separates those from a gap. A contiguous run is
    different in kind: real spectra do not hold exactly zero for eight consecutive pixels.
    Isolated non-positive pixels are therefore left alone, following the quality-flag policy
    of D45, under which the reader may decline to answer but may not guess.

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

    # Contiguous-run labelling without scipy: the cumulative count of "run starts".
    starts = np.flatnonzero(nonpositive & ~np.r_[False, nonpositive[:-1]])
    ends = np.flatnonzero(nonpositive & ~np.r_[nonpositive[1:], False])
    keep = (ends - starts + 1) >= int(min_run)
    if not keep.any():
        return epoch

    wave = np.asarray(epoch.wave)
    gap = np.zeros_like(nonpositive)
    ranges = []
    for lo, hi in zip(starts[keep], ends[keep], strict=True):
        gap[lo : hi + 1] = True
        ranges.append((float(wave[lo]), float(wave[hi])))

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


def share_wavelength_grid(
    epochs: Sequence[EpochData], *, atol_kms: float = 0.05, step: float | None = None
) -> list[EpochData]:
    """Put epochs that differ only by sub-pixel offsets onto one shared wavelength array.

    Some pipelines apply the barycentric correction by shifting before resampling onto a fixed
    step, so each exposure lands on its own grid: the ESO Phase 3 FEROS spectra of one target
    have the same 0.03 A step but start wavelengths spread over 0.78 A and lengths differing
    by tens of pixels. albireo handles that correctly by giving each distinct grid its own
    rebin operator (:func:`albireo.forward._epoch_groups`), but every group's assembly pre-pass
    is live in the same compiled graph, so one group per exposure has been measured at several
    times the memory of one shared grid, in addition to a much larger program to compile.

    When the grids agree to well within a pixel, this function collapses them back to one. The
    operation is a relabelling, not a resampling: no flux value is modified and the ``ivar``
    model stays diagonal (``internal/design.md`` D4). What changes is the wavelength assigned to
    each sample, by at most ``atol_kms``. Epochs are trimmed to their common overlap, so the
    shared array is exact for all of them.

    Alignment is by index, ``round((wave[0] - reference[0]) / step)``, never by value
    comparison: a search for the nearest wavelength at a window edge can land one native pixel
    out, and one FEROS pixel is 1.4 km/s, a radial-velocity error far larger than the
    sub-pixel mismatch being repaired.

    Parameters
    ----------
    epochs : sequence of EpochData
        Epochs to align. Those with different ``instrument`` keys are aligned within
        their own key and left independent of each other.
    atol_kms : float, optional
        Largest wavelength relabelling tolerated, as a velocity. Default 0.05 km/s
        (about 1/50 of a FEROS pixel and 1/50 of its LSF sigma). Raises if exceeded.
    step : float, optional
        Wavelength step of the common grid. Default: the median step of the first epoch
        of each instrument. Only used for the index alignment.

    Returns
    -------
    list of EpochData
        Aligned epochs, in the input order. Each shares one ``wave`` array object with the
        others of its instrument, which is what makes them a single operator group.

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
    tol_factor = float(atol_kms) / 299_792.458
    out: list[EpochData | None] = [None] * len(epochs)

    by_instrument: dict[str, list[int]] = {}
    for k, epoch in enumerate(epochs):
        by_instrument.setdefault(epoch.instrument, []).append(k)

    for instrument, members in by_instrument.items():
        ref = epochs[members[0]]
        dlam = float(step) if step is not None else float(np.median(np.diff(ref.wave)))
        if not dlam > 0:
            raise ValueError(f"instrument {instrument!r}: non-positive wavelength step {dlam}")

        # Offset of each epoch's first sample from the reference lattice, in whole pixels.
        offsets = [round((epochs[k].wave[0] - ref.wave[0]) / dlam) for k in members]
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
                f"by up to {worst:.3e} A ({worst / shared[0] * 299_792.458:.4f} km/s), more "
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
