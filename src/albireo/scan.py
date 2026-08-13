"""SB1 + faint-companion detection: the marginalized K2 scan (``docs/math.md`` §6).

Given a fixed SB1 orbital solution (period, conjunction time, eccentricity vector,
and the primary semi-amplitude ``K_1``), scan a grid of trial companion
semi-amplitudes ``K_2``. At each trial the companion's deviation spectrum is a
*linear* component and marginalizes analytically, so the detection statistic

    D(K_2) = 2 [ log p(y | K_2) - log p(y | no companion) ]

costs one linear solve per grid point and is the optimal matched filter
marginalized over the unknown companion spectrum — no template grid. The recovered
companion spectrum and its pointwise uncertainty at the peak come for free from the
conditional Gaussian.

Calibration: ``D`` depends on the companion's prior scale ``(tau_2, eta_2)``, so its
null distribution is **estimated empirically** by injection-recovery on simulated
datasets (``albireo.simulate``) matched to the data — it is *not* asymptotically
chi-squared, and albireo deliberately makes no such claim. The light fraction of
the putative companion must be chosen explicitly (design decision D13): the
observable is ``ell_2 * d_2``, so ``ell_2`` trades exactly against the companion's
line depths, and only external information (photometry, eclipse depths) sets it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from albireo.data import Dataset
from albireo.grids import LogGrid
from albireo.inference import MarginalOrbitModel
from albireo.likelihood import spectra_std
from albireo.priors import SmoothnessPrior

__all__ = ["K2ScanResult", "k2_scan"]


@dataclass(frozen=True)
class K2ScanResult:
    """Result of :func:`k2_scan`.

    ``detection`` is ``D(K_2)`` on ``k2_grid``; spectra rows (``primary``,
    ``companion``, and their pointwise standard deviations) are the conditional
    posterior at the *peak* trial ``K_2``, on the model grid. ``model`` is the
    two-component :class:`MarginalOrbitModel`, reusable for follow-up (e.g. a joint
    NUTS run seeded at the peak); it is None on a result read back from disk by
    :func:`albireo.results.load_fit`, which stores the numbers and not the machinery.
    """

    k2_grid: np.ndarray
    log_likelihood: np.ndarray  # log p(y | K_2), same length as k2_grid
    log_likelihood_null: float  # log p(y | no companion)
    detection: np.ndarray  # D(K_2) = 2 * (log_likelihood - log_likelihood_null)
    k2_peak: float
    primary: np.ndarray
    primary_std: np.ndarray
    companion: np.ndarray
    companion_std: np.ndarray
    model: MarginalOrbitModel | None = None

    @property
    def peak_index(self) -> int:
        return int(np.argmax(self.detection))

    @property
    def detection_peak(self) -> float:
        return float(self.detection[self.peak_index])


def k2_scan(
    grid: LogGrid,
    dataset: Dataset,
    *,
    orbit: Mapping,
    k1: float,
    k2_grid,
    light_fractions,
    lsf_sigma_v: Mapping[str, float],
    prior: SmoothnessPrior,
    v_rel_max_kms: float,
    telluric: bool = False,
    response_coeffs=None,
    block_size: int | None = None,
) -> K2ScanResult:
    """Scan trial companion semi-amplitudes against the no-companion model.

    Parameters
    ----------
    grid, dataset, lsf_sigma_v, telluric, response_coeffs, block_size
        As in :class:`albireo.inference.MarginalOrbitModel`.
    orbit
        The fixed SB1 solution: mapping with ``period``, ``t_conj``, ``secosw``,
        ``sesinw`` (gamma = 0 as everywhere; D14). The companion moves with
        ``omega + pi`` at each trial ``K_2``.
    k1
        Primary semi-amplitude [km/s], held fixed.
    k2_grid
        Trial companion semi-amplitudes [km/s]; ``(K_1 + max(k2_grid))(1 + e)``
        (plus barycentric motion if ``telluric``) must fit in ``v_rel_max_kms``.
    light_fractions
        Explicit ``(ell_1, ell_2)`` (or per-epoch ``(2, n_epochs)``) — mandatory,
        see the module docstring on the ``ell_2`` ↔ line-depth trade.
    prior
        Spectral prior for the two-component model, one ``(tau, eta)`` pair per
        component in order (primary, companion[, telluric]). The null model reuses
        the primary (and telluric) entries.
    v_rel_max_kms
        Static bandwidth budget for the *two-component* model (the null model
        inherits it; it only needs less).

    Returns
    -------
    K2ScanResult
    """
    orbit = {name: jnp.asarray(v) for name, v in dict(orbit).items()}
    expected = ("period", "t_conj", "secosw", "sesinw")
    missing = [s for s in expected if s not in orbit]
    if missing:
        raise ValueError(f"orbit is missing sites {missing}")
    extra = [s for s in orbit if s not in expected]
    if extra:
        raise ValueError(f"unexpected orbit sites {extra} (k1 is a separate argument)")
    k2_grid = np.atleast_1d(np.asarray(k2_grid, dtype=np.float64))
    if np.any(k2_grid <= 0):
        raise ValueError("k2_grid must be positive")

    ell = np.asarray(light_fractions, dtype=np.float64)
    if ell.shape[0] != 2:
        raise ValueError("light_fractions must have two components (primary, companion)")
    n_expected = 3 if telluric else 2
    if prior.n_components != n_expected:
        raise ValueError(
            f"prior must have {n_expected} components (primary, companion"
            f"{', telluric' if telluric else ''}); got {prior.n_components}"
        )

    model = MarginalOrbitModel(
        grid,
        dataset,
        light_fractions=ell,
        lsf_sigma_v=lsf_sigma_v,
        v_rel_max_kms=v_rel_max_kms,
        response_coeffs=response_coeffs,
        telluric=telluric,
        prior=prior,
        block_size=block_size,
    )
    null_idx = np.asarray([0, 2] if telluric else [0])
    null_prior = SmoothnessPrior(prior.tau[null_idx], prior.eta[null_idx])
    null_model = MarginalOrbitModel(
        grid,
        dataset,
        light_fractions=np.ones((1,)) if ell.ndim == 1 else np.ones((1, ell.shape[1])),
        lsf_sigma_v=lsf_sigma_v,
        v_rel_max_kms=v_rel_max_kms,
        response_coeffs=response_coeffs,
        telluric=telluric,
        prior=null_prior,
        block_size=block_size,
    )

    ll_null = float(null_model.log_likelihood({**orbit, "k": jnp.asarray([k1])}))
    ll = np.array(
        [float(model.log_likelihood({**orbit, "k": jnp.asarray([k1, k2])})) for k2 in k2_grid]
    )
    detection = 2.0 * (ll - ll_null)

    peak = int(np.argmax(detection))
    result = model.marginal({**orbit, "k": jnp.asarray([k1, k2_grid[peak]])})
    std = np.asarray(spectra_std(result))
    d_hat = np.asarray(result.d_hat)
    return K2ScanResult(
        k2_grid=k2_grid,
        log_likelihood=ll,
        log_likelihood_null=ll_null,
        detection=detection,
        k2_peak=float(k2_grid[peak]),
        primary=d_hat[0],
        primary_std=std[0],
        companion=d_hat[1],
        companion_std=std[1],
        model=model,
    )
