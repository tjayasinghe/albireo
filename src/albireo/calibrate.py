"""Injection-recovery calibration of the faint-companion detection statistic.

:func:`albireo.scan.k2_scan` reports the detection statistic ``D`` and the semi-amplitude
at its peak. Interpreting the peak requires the null distribution of ``D`` (how often noise
alone produces a peak of that size) and the completeness (how often a companion of a given
brightness would have been detected). Neither has a closed form. ``D`` is not
asymptotically chi-squared: the companion is a boundary hypothesis whose nuisance parameter
is a prior-regularized function, so Wilks' theorem does not apply, and ``D`` depends on the
companion's prior scale ``(tau_2, eta_2)``, on the epoch sampling, on the masks and on the
per-pixel weights (``docs/math.md`` section 6.2).

Both distributions are therefore measured. Companion-free datasets are drawn from the
fitted model and scanned exactly as the observed data were scanned, which gives the null
distribution of the peak ``D``; a companion injected at a ladder of light fractions gives
the completeness curve. Together they give a completeness limit at a stated confidence: the
light fraction above which a companion would have been detected at, for example, 95%
confidence. Each trial is drawn through the observed dataset's own operators
(:func:`albireo.simulate.resimulate`), so the epoch times, barycentric velocities, chip
gaps, cosmics, native wavelength solutions, response and per-pixel weights are those of the
data. Only the noise and the injected spectra change, and the replacement is a data-term
swap (:func:`albireo.forward.with_data`) that reuses the rebin operators and pair tables,
so thousands of trials cost scan time rather than build time.

The detection threshold is the smallest ``D`` whose estimated false-alarm probability
``(1 + #{null >= D}) / (N + 1)`` is within budget, so the realized null exceedance never
exceeds the nominal rate, and no false-alarm probability below ``1 / (N + 1)`` is reported.

The calibration is conditional on the assumed ``K_1``, orbit, light fractions and
companion spectrum. The null trials are drawn at the same ``K_1``, orbit and light
fractions the scan assumes, so the threshold is self-consistent with those assumptions and
cannot detect that any of them is wrong. ``K_1`` matters most, because an error in it
inflates ``D``: unremoved primary signal is coherent across epochs, the companion's free
spectrum absorbs it, and the peak grows. Measured (``docs/benchmarks.md``): a ``K_1``
10% high tripled the detection statistic and reduced the recovered companion's line
pattern from 0.96 correlation with the truth to 0.49. A calibrated threshold does not flag
this; marginalizing ``K_1`` (``k1_sigma=``) addresses it, and the two are complementary.
The observable is ``ell_2 * d_2``, so a companion with no lines is invisible at any light
fraction and one with deeper lines than assumed is found below the quoted limit. The
default template is the primary's own recovered spectrum, the usual assumption in the
literature; the assumption should be quoted with the limit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from albireo.data import Dataset
from albireo.forward import with_data, with_light_fractions
from albireo.grids import LogGrid
from albireo.inference import MarginalOrbitModel
from albireo.priors import SmoothnessPrior
from albireo.scan import _check_search, _k1_quadrature, _logsumexp, _scan_grids
from albireo.simulate import resimulate

__all__ = ["DetectionLimit", "detection_limit"]


@dataclass(frozen=True)
class DetectionLimit:
    """Result of :func:`detection_limit`: a null distribution and a completeness curve.

    ``null_peaks`` is the largest ``D`` reached anywhere on ``k2_grid`` in each
    companion-free trial. That maximum, not ``D`` at one trial ``K_2``, is the statistic
    a search reports, and calibrating the pointwise statistic instead understates the
    false-alarm rate. ``signal_peaks`` is the same quantity with a companion injected,
    one row per rung of ``ell2_grid``.
    """

    ell2_grid: np.ndarray  # injected companion light fractions
    null_peaks: np.ndarray  # (n_null,) max D per companion-free trial
    signal_peaks: np.ndarray  # (n_ell, n_trials) max D per injected trial
    threshold: float  # D at the requested false-alarm probability
    false_alarm: float  # the requested false-alarm probability
    fap_floor: float  # 1 / (n_null + 1): the smallest FAP this many trials can resolve
    completeness: np.ndarray  # (n_ell,) fraction of injected trials above threshold
    confidence: float
    ell2_limit: float  # smallest ell_2 detected at `confidence` (nan if never reached)
    k2_true: float
    k2_grid: np.ndarray
    k1_marginalized: bool
    limit_is_bracketed: bool = True  # False when the ladder never straddles `confidence`

    @property
    def n_null(self) -> int:
        return int(self.null_peaks.size)

    def false_alarm_probability(self, d: float) -> float:
        """Probability that a companion-free dataset yields a peak at least as large as ``d``.

        The estimator is ``(1 + #{null >= d}) / (n_null + 1)``, which never returns
        zero: with a finite number of trials, the absence of a null trial above ``d`` is
        evidence for a small false-alarm rate, not for none. A value equal to
        :attr:`fap_floor` means the rate is below the resolution of this calibration;
        more null trials are needed to say anything sharper.
        """
        return float((1 + np.count_nonzero(self.null_peaks >= d)) / (self.n_null + 1))

    def summary(self) -> str:
        """One-paragraph statement of the result."""
        pct = 100.0 * self.confidence
        if np.isfinite(self.ell2_limit) and self.limit_is_bracketed:
            limit = (
                f"Any companion contributing more than {100.0 * self.ell2_limit:.2f}% of "
                f"the light would have been detected at {pct:.0f}% confidence"
            )
        elif np.isfinite(self.ell2_limit):
            # The faintest rung was already recovered at the requested confidence: the
            # ladder bounds the limit but does not measure it.
            limit = (
                f"The limit is at most {100.0 * self.ell2_limit:.2f}% of the light: the "
                f"faintest rung tested, and it was recovered {100.0 * self.completeness[0]:.0f}% "
                f"of the time, so the search is more sensitive than this ladder resolves. "
                f"Extend ell2_grid downward to measure the {pct:.0f}% crossing"
            )
        else:
            limit = (
                f"No rung of the injected ladder (up to "
                f"{100.0 * self.ell2_grid[-1]:.2f}% of the light) reached {pct:.0f}% "
                f"completeness, so the search sets no limit at this confidence"
            )
        return (
            f"{limit}, against a detection threshold D > {self.threshold:.1f} set at a "
            f"{100.0 * self.false_alarm:.2g}% false-alarm probability from "
            f"{self.n_null} companion-free trials (resolution floor "
            f"{100.0 * self.fap_floor:.2g}%). Companion injected at "
            f"K_2 = {self.k2_true:.1f} km/s; K_1 "
            f"{'marginalized' if self.k1_marginalized else 'held fixed'}."
        )


def _threshold_at(null_peaks: np.ndarray, false_alarm: float) -> float:
    """The smallest ``D`` whose estimated false-alarm probability is at most ``false_alarm``.

    Defined through :meth:`DetectionLimit.false_alarm_probability` rather than as a
    sample quantile, so the two agree by construction: every detection the calibration
    reports carries a false-alarm probability within budget (``docs/math.md`` section
    6.2).

    An interpolating quantile lacks that property and errs in the anti-conservative
    direction. ``np.quantile(null, 0.95)`` falls between order statistics, so with 24
    trials it can leave two of them above the threshold, an 8% empirical false-alarm
    rate reported as 5%. Here the threshold is the ``(c+1)``-th largest null peak with
    ``c = floor(fa (n+1)) - 1``, which bounds the strict exceedance count by ``c`` and
    hence gives ``FAP <= fa``. When ``fa`` is below the resolution floor ``1/(n+1)`` the
    rule reduces to exceeding every null trial.
    """
    c = max(0, int(np.floor(false_alarm * (null_peaks.size + 1))) - 1)
    ordered = np.sort(null_peaks)[::-1]
    return float(ordered[min(c, ordered.size - 1)])


def _interpolate_limit(
    ell2: np.ndarray, completeness: np.ndarray, confidence: float
) -> tuple[float, bool]:
    """Smallest ``ell_2`` whose completeness reaches ``confidence``, linear between rungs.

    Takes the first upcrossing rather than the global one: completeness is monotone in
    principle but is estimated from a finite number of trials, so a later dip below the
    line is noise and does not move the limit outward.

    Returns the limit and whether the ladder brackets it. It does not when the faintest
    rung is already complete; the search is then more sensitive than any rung tested,
    and the value is an upper bound on the limit rather than a measurement of it.
    Reporting the first rung as a measured crossing would understate the sensitivity
    and would be indistinguishable from a real crossing.
    """
    hits = np.nonzero(completeness >= confidence)[0]
    if hits.size == 0:
        return float("nan"), False
    i = int(hits[0])
    if i == 0:
        return float(ell2[0]), False
    c0, c1 = completeness[i - 1], completeness[i]
    if c1 <= c0:
        return float(ell2[i]), True
    frac = (confidence - c0) / (c1 - c0)
    return float(ell2[i - 1] + frac * (ell2[i] - ell2[i - 1])), True


def detection_limit(
    grid: LogGrid,
    dataset: Dataset,
    *,
    orbit: Mapping,
    k1: float,
    k2_true: float,
    k2_grid,
    light_fractions,
    lsf_sigma_v: Mapping[str, float],
    prior: SmoothnessPrior,
    v_rel_max_kms: float,
    ell2_grid,
    primary_template=None,
    companion_template=None,
    extra_templates=None,
    k1_sigma: float | None = None,
    k1_nodes: int = 7,
    n_null: int = 100,
    n_trials: int = 50,
    false_alarm: float = 0.01,
    confidence: float = 0.95,
    seed: int = 0,
    telluric: bool = False,
    nebular: bool = False,
    nebular_v_kms: float = 0.0,
    response_coeffs=None,
    block_size: int | None = None,
    sweep_batch: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> DetectionLimit:
    """Calibrate the K2 scan by injection and recovery through the observed data's operators.

    Runs ``n_null`` companion-free trials to obtain the null distribution of the scan's
    peak ``D``, then ``n_trials`` trials at each rung of ``ell2_grid`` to obtain the
    completeness. Every trial is a full :func:`albireo.scan.k2_scan` over ``k2_grid``
    with the same grid, the same prior and the same ``K_1`` treatment as the real
    analysis; a threshold calibrated for a different search does not apply to this one.
    The output is a completeness limit at a stated confidence (``docs/math.md`` section
    6.2 and the module docstring).

    Parameters
    ----------
    grid, dataset, lsf_sigma_v, telluric, nebular, nebular_v_kms, response_coeffs,
    block_size, sweep_batch
        As in :func:`albireo.scan.k2_scan`.
    orbit, k1, k2_grid, light_fractions, prior, v_rel_max_kms, k1_sigma, k1_nodes
        As in :func:`albireo.scan.k2_scan`; they must match the real scan.
        ``light_fractions`` is the assumed pair the analysis scans with. The
        injected amplitude is set by ``ell2_grid`` and varies independently: the
        assumption belongs to the analysis, and the limit is a statement about the truth.
    k2_true
        Semi-amplitude [km/s] at which the companion is injected. In an SB2 the
        components move in antiphase, so their relative velocity never drops below
        roughly ``K_1``, and the limit is nearly flat in ``K_2`` whenever ``K_1`` is
        large: measured 0.292 / 0.296 / 0.297% at ``K_2`` = 20 / 40 / 65 km/s with
        ``K_1`` = 55 km/s (``docs/benchmarks.md``). A real dependence is expected
        only when ``K_1`` is small enough that the pair is barely resolved at any phase;
        there, calibrate at several ``K_2`` and quote the worst.
    ell2_grid
        Ladder of injected companion light fractions, ascending, in ``(0, 1)``. Each rung
        costs ``n_trials`` scans.
    primary_template
        Deviation spectrum injected for the primary, ``(grid.n,)``. Default: the
        no-companion fit to the observed data, which makes the calibration a parametric
        bootstrap of the real analysis.
    companion_template
        Deviation spectrum injected for the companion, ``(grid.n,)``. Default: the
        primary template, i.e. a companion with the same line pattern as the primary.
        The limit is conditional on this choice (see the module docstring), and the
        assumption should be quoted with the limit.
    extra_templates
        Deviation spectra injected for the non-stellar components, ``(n_extra, grid.n)``
        in the order telluric, nebular, for whichever are enabled. Default: their rows of
        the same no-companion fit, so the trials carry the sky at the strength the data
        show.
    n_null, n_trials
        Number of companion-free trials, and of trials per rung. ``n_null`` sets the
        resolution floor on the false-alarm probability at ``1 / (n_null + 1)``: 100
        trials cannot substantiate a claim below about 1%.
    false_alarm
        False-alarm probability at which the detection threshold is placed (default
        0.01). The threshold is conservative by construction (:func:`_threshold_at`):
        the realized rate over the null trials is at most this value.
    confidence
        Completeness at which the reported limit is quoted (default 0.95).
    seed
        Base seed. Every trial receives its own derived seed, so the calibration is
        reproducible and no two trials share a noise draw.
    progress
        Optional ``callback(done, total)``, called after each trial. A full calibration
        is thousands of linear solves and produces no other feedback.

    Returns
    -------
    DetectionLimit

    Examples
    --------
    >>> lim = detection_limit(grid, ds, ell2_grid=np.linspace(0.01, 0.1, 5), ...)  # doctest: +SKIP
    >>> print(lim.summary())  # doctest: +SKIP
    """
    ell = np.asarray(light_fractions, dtype=np.float64)
    if ell.shape[0] != 2:
        raise ValueError("light_fractions must have two components (primary, companion)")
    ell2_grid = np.atleast_1d(np.asarray(ell2_grid, dtype=np.float64))
    if np.any(ell2_grid <= 0.0) or np.any(ell2_grid >= 1.0):
        raise ValueError("ell2_grid entries must lie strictly between 0 and 1")
    if np.any(np.diff(ell2_grid) <= 0.0):
        raise ValueError("ell2_grid must be strictly ascending")
    if not 0.0 < false_alarm < 1.0:
        raise ValueError(f"false_alarm must lie in (0, 1); got {false_alarm}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1); got {confidence}")
    if n_null < 1 or n_trials < 1:
        raise ValueError("n_null and n_trials must be at least 1")

    orbit, k2_grid = _check_search(orbit, k2_grid)
    k1_grid, k1_log_w = _k1_quadrature(k1, k1_sigma, k1_nodes)

    common = {
        "lsf_sigma_v": lsf_sigma_v,
        "v_rel_max_kms": v_rel_max_kms,
        "response_coeffs": response_coeffs,
        "telluric": telluric,
        "nebular": nebular,
        "nebular_v_kms": nebular_v_kms,
        "block_size": block_size,
    }
    extra = [name for name, on in (("telluric", telluric), ("nebular", nebular)) if on]
    n_expected = 2 + len(extra)
    if prior.n_components != n_expected:
        raise ValueError(
            f"prior must have {n_expected} components (primary, companion"
            + "".join(f", {name}" for name in extra)
            + f"); got {prior.n_components}"
        )
    model = MarginalOrbitModel(grid, dataset, light_fractions=ell, prior=prior, **common)
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

    # Everything injected other than the companion comes from the no-companion fit to
    # the observed data, so the trials are a parametric bootstrap of the analysis, and a
    # telluric or nebular component, if the scan models one, is present in the trials
    # at the strength the data show. Injecting zero for those would calibrate against a
    # cleaner sky than the one observed and understate the false-alarm rate: an
    # unmodelled static residual is exactly what the scan mistakes for a companion.
    n_extra = model.problem.n_components - 2
    fit = None
    if primary_template is None or (n_extra and extra_templates is None):
        fit = null_model.marginal({**orbit, "k": jnp.asarray([k1])})
    if primary_template is None:
        primary_template = np.asarray(fit.d_hat[0])  # type: ignore[union-attr]
    primary_template = np.asarray(primary_template, dtype=np.float64)
    if primary_template.shape != (grid.n,):
        raise ValueError(f"primary_template must have shape ({grid.n},)")
    if companion_template is None:
        companion_template = primary_template
    companion_template = np.asarray(companion_template, dtype=np.float64)
    if companion_template.shape != (grid.n,):
        raise ValueError(f"companion_template must have shape ({grid.n},)")
    if n_extra:
        if extra_templates is None:
            extras = np.asarray(fit.d_hat[1:])  # type: ignore[union-attr]
        else:
            extras = np.atleast_2d(np.asarray(extra_templates, dtype=np.float64))
        if extras.shape != (n_extra, grid.n):
            raise ValueError(
                f"extra_templates must have shape ({n_extra}, {grid.n}); one row per "
                f"non-stellar component, in the order " + ", ".join(extra) + f"; got {extras.shape}"
            )
    else:
        extras = np.zeros((0, grid.n))

    # Injection problem: the two-component model at the true orbit. Only its light
    # fractions change between rungs; the companion's amplitude is the ladder variable.
    d_inject = np.concatenate([primary_template[None], companion_template[None], extras], axis=0)
    inject_at = model.problem_at({**orbit, "k": jnp.asarray([k1, float(k2_true)])})

    total = n_null + n_trials * ell2_grid.size
    done = 0

    def run(ell2: float, trial_seed: int) -> float:
        """One trial: inject at ell2, redraw the noise, scan, return the peak D."""
        # Stellar columns only; with_light_fractions carries the telluric and nebular
        # columns through unchanged (fraction 1, amplitude 1).
        stellar = jnp.asarray([1.0 - ell2, ell2])
        drawn = resimulate(with_light_fractions(inject_at, stellar), d_inject, seed=trial_seed)
        z = [g.z for g in drawn.groups]
        ll_grid, ll_null_grid = _scan_grids(
            model,
            null_model,
            orbit,
            k1_grid,
            k2_grid,
            sweep_batch=sweep_batch,
            problem=with_data(model.problem, z),
            null_problem=with_data(null_model.problem, z),
        )
        ll = _logsumexp(k1_log_w[:, None] + ll_grid, axis=0)
        ll_null = float(_logsumexp(k1_log_w + ll_null_grid, axis=0))
        return float(np.max(2.0 * (ll - ll_null)))

    # Distinct seed streams per rung, so adding a rung never perturbs the others.
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=(ell2_grid.size + 1, max(n_null, n_trials)))

    null_peaks = np.empty(n_null)
    for t in range(n_null):
        null_peaks[t] = run(0.0, int(seeds[0, t]))
        done += 1
        if progress is not None:
            progress(done, total)

    signal_peaks = np.empty((ell2_grid.size, n_trials))
    for i, ell2 in enumerate(ell2_grid):
        for t in range(n_trials):
            signal_peaks[i, t] = run(float(ell2), int(seeds[i + 1, t]))
            done += 1
            if progress is not None:
                progress(done, total)

    threshold = _threshold_at(null_peaks, float(false_alarm))
    completeness = np.mean(signal_peaks > threshold, axis=1)
    ell2_limit, bracketed = _interpolate_limit(ell2_grid, completeness, float(confidence))
    return DetectionLimit(
        ell2_grid=ell2_grid,
        null_peaks=null_peaks,
        signal_peaks=signal_peaks,
        threshold=threshold,
        false_alarm=float(false_alarm),
        fap_floor=1.0 / (n_null + 1),
        completeness=completeness,
        confidence=float(confidence),
        ell2_limit=ell2_limit,
        k2_true=float(k2_true),
        k2_grid=k2_grid,
        k1_marginalized=k1_sigma is not None and k1_sigma > 0.0,
        limit_is_bracketed=bracketed,
    )
