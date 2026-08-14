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

``K_1`` may be marginalized rather than fixed (``k1_sigma=``). The published state of
the art fixes it at a literature value, and the papers are explicit about the cost:
small deviations in the assumed primary semi-amplitude put spurious features in the
recovered secondary spectrum. Integrating it out removes that failure mode instead of
mitigating it — over a Gauss-Hermite rule on a Gaussian ``K_1`` prior, applied to the
*same* prior in both the companion and the no-companion model so the statistic stays a
ratio of two marginal likelihoods.

Calibration: ``D`` depends on the companion's prior scale ``(tau_2, eta_2)``, so its
null distribution is **estimated empirically** by injection-recovery on simulated
datasets (:mod:`albireo.calibrate`) matched to the data — it is *not* asymptotically
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


def _check_search(orbit: Mapping, k2_grid):
    """Validate the fixed SB1 solution and the trial grid; return them normalized.

    Shared by :func:`k2_scan` and :func:`albireo.calibrate.detection_limit` so that a
    calibration and the scan it calibrates reject the same mistakes with the same words —
    they take the same arguments and must mean the same thing by them.
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
    return orbit, k2_grid


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)), axis=axis)


def _k1_quadrature(k1: float, k1_sigma: float | None, k1_nodes: int):
    """Nodes and normalized log-weights for the ``K_1`` prior.

    ``k1_sigma=None`` gives the single-node rule — a delta at ``k1``, whose log-sum-exp
    is exactly the identity, so a fixed-``K_1`` scan costs nothing for the machinery it
    does not use. (It is not bit-identical to the pre-vectorization scan: batching the
    trials into one ``lax.map`` re-associates the linear algebra, which moves the
    log-likelihoods by ~1e-9 — floating point, not method.) Otherwise Gauss-Hermite on
    ``N(k1, k1_sigma^2)``: ``int f(K) N(K) dK = sum_a (w_a / sqrt(pi)) f(k1 + sqrt(2)
    sigma x_a)``, which is exact for polynomials up to degree ``2n - 1`` and needs far
    fewer likelihood evaluations than a uniform grid of the same accuracy. The same rule
    already carries the LSF's Gauss-Hermite skewness (D38).
    """
    if k1_sigma is None or k1_sigma == 0.0:
        return np.array([float(k1)]), np.zeros(1)
    if k1_sigma < 0.0:
        raise ValueError(f"k1_sigma must be non-negative; got {k1_sigma}")
    if k1_nodes < 2:
        raise ValueError(f"k1_nodes must be at least 2; got {k1_nodes}")
    x, w = np.polynomial.hermite.hermgauss(int(k1_nodes))
    nodes = float(k1) + np.sqrt(2.0) * float(k1_sigma) * x
    if np.any(nodes <= 0.0):
        raise ValueError(
            f"the K_1 quadrature reaches {nodes.min():.3f} km/s — a non-positive "
            f"semi-amplitude, which is not a physical trial and flips the companion's "
            f"phase. Narrow k1_sigma (currently {k1_sigma}) or lower k1_nodes "
            f"(currently {k1_nodes}); the rule spans about "
            f"+/-{np.sqrt(2.0) * k1_sigma * x.max():.1f} km/s around k1={k1}."
        )
    return nodes, np.log(w) - 0.5 * np.log(np.pi)


@dataclass(frozen=True)
class K2ScanResult:
    """Result of :func:`k2_scan`.

    ``detection`` is ``D(K_2)`` on ``k2_grid``; spectra rows (``primary``,
    ``companion``, and their pointwise standard deviations) are the conditional
    posterior at the *peak* trial ``K_2``, on the model grid. ``model`` is the
    two-component :class:`MarginalOrbitModel`, reusable for follow-up (e.g. a joint
    NUTS run seeded at the peak); it is None on a result read back from disk by
    :func:`albireo.results.load_fit`, which stores the numbers and not the machinery.

    With ``K_1`` marginalized (``k1_sigma=``), ``log_likelihood`` is the *marginal*
    over the quadrature and ``log_likelihood_grid`` is the ``(n_k1, n_k2)`` surface
    behind it — worth looking at, because the shape of the ridge in that plane is the
    ``K_1``-``K_2`` covariance the fixed-``K_1`` scan silently assumed away. The peak
    spectra are conditional on ``k1_peak``, the best node at ``k2_peak`` (a profile,
    not a marginal: there is no closed form for the K_1-marginalized spectrum, and
    quoting a mixture of the nodes' spectra would blur the lines rather than widen
    their error bars). With ``K_1`` fixed, the grid has one row and ``k1_peak`` is it.
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
    k1_grid: np.ndarray | None = None  # quadrature nodes [km/s]
    k1_log_weights: np.ndarray | None = None  # normalized: logsumexp(.) == 0
    log_likelihood_grid: np.ndarray | None = None  # (n_k1, n_k2)
    log_likelihood_null_grid: np.ndarray | None = None  # (n_k1,)
    k1_peak: float = float("nan")

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
    k1_sigma: float | None = None,
    k1_nodes: int = 7,
    telluric: bool = False,
    nebular: bool = False,
    nebular_v_kms: float = 0.0,
    response_coeffs=None,
    block_size: int | None = None,
    sweep_batch: int | None = None,
) -> K2ScanResult:
    """Scan trial companion semi-amplitudes against the no-companion model.

    Parameters
    ----------
    grid, dataset, lsf_sigma_v, telluric, nebular, nebular_v_kms, response_coeffs, block_size
        As in :class:`albireo.inference.MarginalOrbitModel`. A nebular component
        (D40) is worth its extra component here specifically: an unmodelled static
        emission line is a *stationary* residual, and a faint-companion scan is a
        matched filter for exactly that — it will happily report the nebula at
        whatever ``K_2`` puts the companion nearest to rest.
    orbit
        The fixed SB1 solution: mapping with ``period``, ``t_conj``, ``secosw``,
        ``sesinw`` (gamma = 0 as everywhere; D14). The companion moves with
        ``omega + pi`` at each trial ``K_2``.
    k1
        Primary semi-amplitude [km/s]. Held fixed unless ``k1_sigma`` is given, in
        which case it is the *mean* of the Gaussian prior integrated over.
    k2_grid
        Trial companion semi-amplitudes [km/s]; ``(max(k1_grid) + max(k2_grid))(1 + e)``
        (plus barycentric motion if ``telluric``, and ``|nebular_v_kms|`` if
        ``nebular``) must fit in ``v_rel_max_kms``. Budget for the quadrature's reach,
        not for ``k1`` alone — a 7-node rule spans about ``+/-3.8 k1_sigma``.
    k1_sigma
        Standard deviation of the Gaussian prior on ``K_1``, or None (default) to hold
        it fixed and reproduce the fixed-``K_1`` scan exactly. Set it to the published
        uncertainty on the primary's semi-amplitude: the detection statistic then
        compares two models that are *both* honest about not knowing ``K_1``, rather
        than conditioning on a value whose error leaks into the companion's spectrum.
        Costs a factor ``k1_nodes`` in likelihood evaluations, which is what the
        vectorized sweep is for.
    k1_nodes
        Gauss-Hermite nodes for that integral (default 7). Ignored when
        ``k1_sigma`` is None.
    light_fractions
        Explicit ``(ell_1, ell_2)`` (or per-epoch ``(2, n_epochs)``) — mandatory,
        see the module docstring on the ``ell_2`` ↔ line-depth trade.
    prior
        Spectral prior for the two-component model, one ``(tau, eta)`` pair per
        component, ordered primary, companion, then the telluric and nebular entries
        for whichever of those is enabled. The null model reuses every entry but the
        companion's, per-pixel profiles included.
    v_rel_max_kms
        Static bandwidth budget for the *two-component* model (the null model
        inherits it; it only needs less).
    sweep_batch
        Trials per vmapped batch of the scan, as in
        :meth:`albireo.inference.MarginalOrbitModel.log_likelihood_sweep`. None
        (default) lets the size-adaptive policy choose.

    Returns
    -------
    K2ScanResult
    """
    orbit, k2_grid = _check_search(orbit, k2_grid)

    ell = np.asarray(light_fractions, dtype=np.float64)
    if ell.shape[0] != 2:
        raise ValueError("light_fractions must have two components (primary, companion)")
    extra = [name for name, on in (("telluric", telluric), ("nebular", nebular)) if on]
    n_expected = 2 + len(extra)
    if prior.n_components != n_expected:
        raise ValueError(
            f"prior must have {n_expected} components (primary, companion"
            + "".join(f", {name}" for name in extra)
            + f"); got {prior.n_components}"
        )

    common = {
        "lsf_sigma_v": lsf_sigma_v,
        "v_rel_max_kms": v_rel_max_kms,
        "response_coeffs": response_coeffs,
        "telluric": telluric,
        "nebular": nebular,
        "nebular_v_kms": nebular_v_kms,
        "block_size": block_size,
    }
    model = MarginalOrbitModel(grid, dataset, light_fractions=ell, prior=prior, **common)
    # The null model drops the companion; every other component keeps its own prior
    # entry, so the indices are 0 (primary) plus the trailing non-stellar ones.
    null_idx = np.asarray([0, *range(2, n_expected)])
    null_prior = SmoothnessPrior(
        prior.tau[null_idx],
        prior.eta[null_idx],
        None if prior.tau_profile is None else prior.tau_profile[null_idx],
        None if prior.eta_profile is None else prior.eta_profile[null_idx],
    )
    null_model = MarginalOrbitModel(
        grid,
        dataset,
        light_fractions=np.ones((1,)) if ell.ndim == 1 else np.ones((1, ell.shape[1])),
        prior=null_prior,
        **common,
    )

    k1_grid, k1_log_w = _k1_quadrature(k1, k1_sigma, k1_nodes)
    ll_grid, ll_null_grid = _scan_grids(
        model, null_model, orbit, k1_grid, k2_grid, sweep_batch=sweep_batch
    )
    # Marginalize both models over the *same* K_1 prior, so D stays a ratio of two
    # marginal likelihoods rather than a comparison of differently-conditioned ones.
    ll = _logsumexp(k1_log_w[:, None] + ll_grid, axis=0)
    ll_null = float(_logsumexp(k1_log_w + ll_null_grid, axis=0))
    detection = 2.0 * (ll - ll_null)

    peak = int(np.argmax(detection))
    # Spectra are conditional on the best node at the peak K_2 — a profile; see
    # K2ScanResult on why the mixture is not quoted instead.
    peak_k1 = int(np.argmax(ll_grid[:, peak]))
    result = model.marginal({**orbit, "k": jnp.asarray([k1_grid[peak_k1], k2_grid[peak]])})
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
        k1_grid=k1_grid,
        k1_log_weights=k1_log_w,
        log_likelihood_grid=ll_grid,
        log_likelihood_null_grid=ll_null_grid,
        k1_peak=float(k1_grid[peak_k1]),
    )


def _scan_grids(
    model: MarginalOrbitModel,
    null_model: MarginalOrbitModel,
    orbit: Mapping,
    k1_grid: np.ndarray,
    k2_grid: np.ndarray,
    *,
    sweep_batch: int | None = None,
    problem=None,
    null_problem=None,
):
    """The two log-likelihood surfaces a scan is built from, as ``(n_k1, n_k2)`` and ``(n_k1,)``.

    Split out from :func:`k2_scan` because :mod:`albireo.calibrate` runs exactly this,
    thousands of times, against resimulated data on the *same* operators — the
    ``problem`` / ``null_problem`` overrides are how the redrawn data get in without a
    rebuild (:func:`albireo.forward.with_data`).
    """
    n1, n2 = k1_grid.size, k2_grid.size
    pairs = np.stack(
        [np.repeat(k1_grid, n2), np.tile(np.asarray(k2_grid, dtype=np.float64), n1)], axis=1
    )
    ll = model.log_likelihood_sweep(
        orbit, {"k": jnp.asarray(pairs)}, batch_size=sweep_batch, problem=problem
    )
    ll_null = null_model.log_likelihood_sweep(
        orbit,
        {"k": jnp.asarray(k1_grid[:, None])},
        batch_size=sweep_batch,
        problem=null_problem,
    )
    return np.asarray(ll).reshape(n1, n2), np.asarray(ll_null)
