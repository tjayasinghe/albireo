"""Diagnostic and result figures.

matplotlib is an optional dependency (``pip install "albireo[plots]"``) and this module is
never imported at package import — the pattern is the same as :mod:`albireo.io` with
astropy. ``albireo.plot_spectra`` stays discoverable and raises an actionable error if
matplotlib is missing, rather than an ``ImportError`` from somewhere deep inside a figure.

Every function takes arrays or result objects, returns ``(fig, axes)``, and does not call
``plt.show()`` or touch the global style. That is what lets them be composed into a larger
figure, restyled for a paper, or called from a script that writes PNGs — the caller owns
the output.

The set is chosen to cover the questions that actually come up after a fit: did the orbit
converge somewhere sensible (:func:`plot_rv_curve`, :func:`plot_corner`), do the recovered
spectra mean anything where they look interesting (:func:`plot_spectra` — read the
uncertainty band, not the mean), does the noise model fit (:func:`plot_residual_zscores`),
was a companion detected and what its peak is worth (:func:`plot_detection`,
:func:`plot_detection_limit`), and are the nuisance parameters doing something surprising
(:func:`plot_lsf`, :func:`plot_light_fractions`). One of them comes up *before* a fit
rather than after: :func:`plot_forecast`, which is what a proposal for more nights argues
from.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "plot_corner",
    "plot_detection",
    "plot_detection_limit",
    "plot_forecast",
    "plot_light_fractions",
    "plot_lsf",
    "plot_phase_fold",
    "plot_residual_zscores",
    "plot_rv_curve",
    "plot_spectra",
    "plot_todcor_surface",
    "plot_velocity_table",
]

_COMPONENT_COLORS = ("C0", "C3", "C2", "C4", "C5", "C6")


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the bare install
        raise ModuleNotFoundError(
            "albireo.plotting needs matplotlib, which is an optional dependency. "
            'Install it with `pip install "albireo[plots]"` (or `pip install '
            "matplotlib`). Nothing else in albireo imports it."
        ) from exc
    return plt


def _require_arviz():
    try:
        import arviz
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the bare install
        raise ModuleNotFoundError(
            "albireo.plotting.plot_corner needs arviz, which is an optional dependency. "
            'Install it with `pip install "albireo[plots]"` (or `pip install arviz`).'
        ) from exc
    return arviz


def _color(i: int) -> str:
    return _COMPONENT_COLORS[i % len(_COMPONENT_COLORS)]


def phase_of(bjd, period: float, t_conj: float) -> np.ndarray:
    """Orbital phase in ``[0, 1)`` measured from conjunction."""
    return ((np.asarray(bjd, dtype=float) - t_conj) / period) % 1.0


# ---------------------------------------------------------------------------
# orbit
# ---------------------------------------------------------------------------


def plot_rv_curve(samples, bjd, *, truth=None, n_draws: int = 60, ax=None):
    """Posterior orbit draws, phase-folded, with the epochs marked.

    albireo never measures a per-epoch radial velocity — the orbit is inferred from the
    spectra directly — so this is the posterior *curve*, not a fit through RV points. The
    epoch ticks along the bottom are there because they are the thing that actually
    determines how well the orbit is constrained: their phase coverage.

    Parameters
    ----------
    samples
        Posterior samples, e.g. ``mcmc.get_samples()``. Needs the sites ``period``,
        ``t_conj``, ``secosw``, ``sesinw`` and ``k``.
    bjd
        Epoch times, for the phase ticks.
    truth
        Optional :class:`~albireo.simulate.SimulationTruth` (or anything with an ``orbit``
        exposing ``component_velocities``) to overplot — for simulated data only.
    n_draws
        How many posterior draws to plot.
    ax
        Existing axis to draw into; a new figure is made if omitted.

    Returns
    -------
    (Figure, Axes)
    """
    import jax.numpy as jnp

    from albireo.inference import orbit_velocities

    plt = _plt()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.2, 4.4))

    period = float(np.asarray(samples["period"]).mean())
    t_conj = float(np.asarray(samples["t_conj"]).mean())
    phase = np.linspace(0.0, 1.0, 400)
    t_dense = t_conj + phase * period

    sites = ("period", "t_conj", "secosw", "sesinw", "k")
    n_post = int(np.asarray(samples["period"]).shape[0])
    indices = np.linspace(0, n_post - 1, min(n_draws, n_post)).astype(int)

    n_comp = 0
    for i in indices:
        theta = {s: jnp.asarray(np.asarray(samples[s])[i]) for s in sites}
        velocity = np.asarray(orbit_velocities(theta, t_dense))
        n_comp = velocity.shape[0]
        for c in range(n_comp):
            ax.plot(phase, velocity[c], color=_color(c), alpha=0.10, lw=0.9)
    for c in range(n_comp):
        ax.plot([], [], color=_color(c), lw=2, label=f"posterior draws, component {c + 1}")

    y_floor = ax.get_ylim()[0]
    if truth is not None:
        truth_velocity = np.asarray(truth.orbit.component_velocities(t_dense))
        for c in range(truth_velocity.shape[0]):
            ax.plot(
                phase,
                truth_velocity[c],
                color="k",
                ls="--" if c == 0 else ":",
                lw=1.3,
                label=f"truth, component {c + 1}",
            )
        y_floor = float(min(truth_velocity.min(), y_floor))

    epoch_phase = phase_of(bjd, period, t_conj)
    ax.plot(
        epoch_phase, np.full_like(epoch_phase, y_floor), "|", color="0.3", ms=12, label="epochs"
    )
    ax.set_xlabel("phase from conjunction of component 1")
    ax.set_ylabel("radial velocity [km/s]")
    ax.legend(fontsize=8, loc="best")
    return fig, ax


def plot_phase_fold(bjd, values, period: float, t_conj: float, *, yerr=None, ax=None, **kwargs):
    """Fold any per-epoch quantity on the orbital period.

    Useful for anything that should *not* depend on phase — jitter, light fractions,
    residual scatter — where a phase-dependent pattern means the model is absorbing
    something it should be describing.
    """
    plt = _plt()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.0, 3.6))

    phase = phase_of(bjd, period, t_conj)
    style = {"fmt": "o", "ms": 4, "capsize": 2, **kwargs}
    ax.errorbar(phase, np.asarray(values, dtype=float), yerr=yerr, **style)
    ax.set_xlabel("phase from conjunction")
    ax.set_xlim(0.0, 1.0)
    return fig, ax


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------


def plot_spectra(grid, spectra, *, std=None, truth=None, labels=None, axes=None, flux=False):
    """Disentangled component spectra with their uncertainty band.

    The band is the point of the figure. Between the lines, and anywhere the epochs give
    little leverage, the recovered spectrum is set by the smoothness prior rather than by
    the data, and the band is what says so — a mean line on its own invites exactly the
    over-reading this package exists to prevent (``docs/math.md`` §5.1).

    Parameters
    ----------
    grid
        The :class:`~albireo.grids.LogGrid` the spectra live on.
    spectra
        Either posterior draws, shape ``(n_draws, n_comp, n_pix)`` — the band is then the
        draw scatter — or a mean, shape ``(n_comp, n_pix)``, in which case pass ``std``.
    std
        Pointwise standard deviations matching a ``(n_comp, n_pix)`` mean, e.g. from
        :func:`albireo.likelihood.spectra_std`.
    truth
        Optional injected truth, shape ``(n_comp, n_pix)``, for simulated data. It must
        already be on ``grid``. A simulation stores its truth on the grid it was generated
        on, which is *not* the grid the model was solved on, so resample it first --
        ``np.interp(grid.wave, truth_grid.wave, component, left=0.0, right=0.0)``.
    labels
        Component names for the y-axis; defaults to ``d_1``, ``d_2``, ...
    axes
        An array of existing axes, one per component.
    flux
        Plot ``1 + d`` (normalized flux) instead of the deviation ``d``.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    plt = _plt()
    spectra = np.asarray(spectra)
    if spectra.ndim == 3:
        mean, band = spectra.mean(axis=0), spectra.std(axis=0)
    elif spectra.ndim == 2:
        mean = spectra
        band = None if std is None else np.asarray(std)
    else:
        raise ValueError(
            f"spectra must have shape (n_draws, n_comp, n_pix) or (n_comp, n_pix), "
            f"got {spectra.shape}"
        )

    wave = np.asarray(grid.wave)
    if mean.shape[-1] != wave.size:
        raise ValueError(
            f"spectra have {mean.shape[-1]} pixels but the grid has {wave.size}; "
            "they must come from the same fit."
        )

    if truth is not None:
        truth = np.asarray(truth)
        if truth.shape[-1] != wave.size:
            raise ValueError(
                f"truth has {truth.shape[-1]} pixels but the grid has {wave.size}. "
                "A simulation's truth is stored on the grid it was generated on, which is "
                "not the grid the model was solved on -- the model grid is widened by the "
                "velocity budget and the LSF radius, and its sampling comes from the data. "
                "Resample it first: np.interp(grid.wave, truth_grid.wave, component, "
                "left=0.0, right=0.0) per component."
            )

    n_comp = mean.shape[0]
    if axes is None:
        fig, axes = plt.subplots(n_comp, 1, figsize=(9.0, 3.0 * n_comp), sharex=True, squeeze=False)
        axes = axes.ravel()
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].figure

    offset = 1.0 if flux else 0.0
    for i in range(n_comp):
        ax, color = axes[i], _color(i)
        if band is not None:
            ax.fill_between(
                wave,
                offset + mean[i] - 2 * band[i],
                offset + mean[i] + 2 * band[i],
                color=color,
                alpha=0.35,
                lw=0,
                label="posterior $\\pm 2\\sigma$",
            )
        ax.plot(wave, offset + mean[i], color=color, lw=1.0, label="posterior mean")
        if truth is not None:
            ax.plot(wave, offset + truth[i], "k--", lw=0.8, label="truth")
        default = f"$1 + d_{{{i + 1}}}$" if flux else f"$d_{{{i + 1}}}$"
        ax.set_ylabel(labels[i] if labels is not None else default)
        ax.legend(fontsize=8, loc="lower right", ncol=3)
    axes[-1].set_xlabel("wavelength [Å]")
    return fig, axes


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def plot_residual_zscores(problem, d_stack, *, bjd=None, axes=None):
    """Three views of the whitened residuals: distribution, per epoch, and lag-1.

    Under a correct noise model the whitened residuals are standard normal and
    independent. The three panels test different failure modes, and the third is the one
    that earns its place: heavy lag-1 autocorrelation means neighbouring pixels are
    correlated, which inflates every uncertainty derived from the fit and is invisible in
    a histogram. It is the statistic the AR(1) work turned on
    (``docs/benchmarks.md``, D34).

    The lag-1 coefficient is computed *within* each epoch, since consecutive pixels of
    different exposures have nothing to do with each other.

    Parameters
    ----------
    problem
        The :class:`~albireo.forward.Problem` the residuals are taken against.
    d_stack
        Component deviation spectra, shape ``(n_comp, n_pix)``.
    bjd
        Optional epoch times; the middle panel uses them for the x-axis instead of the
        epoch index.
    axes
        Three existing axes.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    from albireo.forward import data_residual_zscores

    plt = _plt()
    per_epoch = data_residual_zscores(problem, d_stack, per_epoch=True)
    flat = np.concatenate(per_epoch)

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].figure

    axes[0].hist(flat, bins=60, density=True, color="0.6", edgecolor="none")
    x = np.linspace(-4.0, 4.0, 200)
    axes[0].plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), "k--", lw=1.2, label="$N(0,1)$")
    axes[0].set_xlabel("whitened residual")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"all pixels: sd = {flat.std():.3f}")
    axes[0].legend(fontsize=8)

    scatter = np.array([r.std() for r in per_epoch])
    x_epoch = np.arange(len(per_epoch)) if bjd is None else np.asarray(bjd, dtype=float)
    axes[1].plot(x_epoch, scatter, "o", ms=4, color="C0")
    axes[1].axhline(1.0, color="k", ls=":", lw=1.0)
    axes[1].set_xlabel("epoch" if bjd is None else "BJD")
    axes[1].set_ylabel("residual sd")
    axes[1].set_title("per epoch (1.0 = noise model matches)")

    lag1 = np.array([_lag1(r) for r in per_epoch])
    axes[2].plot(x_epoch, lag1, "o", ms=4, color="C3")
    axes[2].axhline(0.0, color="k", ls=":", lw=1.0)
    axes[2].set_xlabel("epoch" if bjd is None else "BJD")
    axes[2].set_ylabel("lag-1 autocorrelation")
    axes[2].set_title(f"lag-1 (median {np.median(lag1):+.3f})")
    return fig, axes


def _lag1(residuals) -> float:
    """Lag-1 autocorrelation of one epoch's residuals; NaN if too short."""
    r = np.asarray(residuals, dtype=float)
    if r.size < 3:
        return np.nan
    r = r - r.mean()
    denominator = float(r @ r)
    if denominator <= 0.0:
        return np.nan
    return float(r[:-1] @ r[1:] / denominator)


def plot_lsf(anchor_wave, sigma, *, h3=None, sigma_max=None, axes=None):
    """Inferred line-spread-function width — and skewness — against wavelength.

    Read this as a diagnostic, not as a measurement of the instrument. The LSF parameters
    are the most degenerate part of the model: they trade against the intrinsic line
    widths of the components and against the smoothness prior, so a width that drifts with
    wavelength may be describing the spectrograph or may be absorbing something else
    entirely (``docs/benchmarks.md``, D37 and D38). What the figure is genuinely good for
    is spotting a width that has run into its build-time bound, which is drawn when
    ``sigma_max`` is given — the kernel radius is fixed at build time, so a fit pressed
    against that bound is a fit whose structure was too small.

    Parameters
    ----------
    anchor_wave
        Anchor wavelengths [Å], as passed to the model.
    sigma
        Inferred widths [km/s] at each anchor — a posterior mean, or samples of shape
        ``(n_draws, n_anchor)``, in which case a band is drawn.
    h3
        Optional Gauss-Hermite skewness at each anchor, same shape convention.
    sigma_max
        The build-time upper bound on the width, drawn as a limit line.
    axes
        One axis (no ``h3``) or two.

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    plt = _plt()
    n_panels = 1 if h3 is None else 2
    if axes is None:
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(7.0, 3.2 * n_panels), sharex=True, squeeze=False
        )
        axes = axes.ravel()
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].figure

    wave = np.asarray(anchor_wave, dtype=float)
    _plot_anchored(axes[0], wave, sigma, "C0")
    axes[0].set_ylabel(r"$\sigma_{\rm LSF}$ [km/s]")
    if sigma_max is not None:
        axes[0].axhline(float(sigma_max), color="k", ls="--", lw=1.0, label="build-time bound")
        axes[0].legend(fontsize=8)
    if h3 is not None:
        _plot_anchored(axes[1], wave, h3, "C3")
        axes[1].axhline(0.0, color="k", ls=":", lw=1.0)
        axes[1].set_ylabel("$h_3$ (skewness)")
    axes[-1].set_xlabel("wavelength [Å]")
    return fig, axes


def _plot_anchored(ax, wave, values, color) -> None:
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        mean, sd = values.mean(axis=0), values.std(axis=0)
        ax.fill_between(wave, mean - 2 * sd, mean + 2 * sd, color=color, alpha=0.3, lw=0)
        ax.plot(wave, mean, "o-", color=color, ms=4)
    else:
        ax.plot(wave, values, "o-", color=color, ms=4)


def plot_light_fractions(samples, *, bjd=None, period=None, t_conj=None, ax=None):
    """Per-epoch light fractions, against phase where an orbit is available.

    The light ratio is the one quantity spectroscopy alone cannot pin down for a constant
    ratio, which is why albireo refuses to assume one (``docs/design.md`` §5). Where the
    ratio *does* vary — eclipses, most obviously — that variation is what breaks the
    degeneracy, and this figure is where you check that the inferred variation looks like
    the eclipse rather than like noise.

    Parameters
    ----------
    samples
        Posterior samples containing the ``light`` site, shape
        ``(n_draws, n_epochs, n_comp)`` or ``(n_draws, n_comp)`` for a constant ratio.
    bjd
        Epoch times. With ``period`` and ``t_conj`` the x-axis becomes phase.
    period, t_conj
        Orbital elements for the phase fold.
    ax
        Existing axis.

    Returns
    -------
    (Figure, Axes)
    """
    plt = _plt()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.2, 4.0))

    light = np.asarray(samples["light"] if isinstance(samples, dict) else samples, dtype=float)
    if light.ndim == 2:  # constant light ratio: (n_draws, n_comp)
        light = light[:, None, :]
    mean, sd = light.mean(axis=0), light.std(axis=0)
    n_epochs, n_comp = mean.shape

    if bjd is not None and period is not None and t_conj is not None:
        x = phase_of(bjd, float(period), float(t_conj))
        ax.set_xlabel("phase from conjunction")
        ax.set_xlim(0.0, 1.0)
    elif bjd is not None:
        x = np.asarray(bjd, dtype=float)
        ax.set_xlabel("BJD")
    else:
        x = np.arange(n_epochs)
        ax.set_xlabel("epoch")

    for c in range(n_comp):
        ax.errorbar(
            x,
            mean[:, c],
            yerr=sd[:, c],
            fmt="o",
            ms=4,
            capsize=2,
            color=_color(c),
            label=f"component {c + 1}",
        )
    ax.set_ylabel("light fraction $\\ell$")
    ax.legend(fontsize=8)
    return fig, ax


# ---------------------------------------------------------------------------
# detection and posterior
# ---------------------------------------------------------------------------


def plot_detection(result, *, injected_k2=None, threshold=None, ax=None, label=None):
    """The K₂ detection statistic across the scanned grid.

    ``D`` is twice the log-likelihood ratio against the no-companion model. It is *not* a
    chi-squared and no p-value is implied: the Occam term keeps ``D`` below zero when there
    is nothing to find, and turning a peak into a false-alarm probability requires an
    injection-recovery calibration. Where a calibrated threshold exists, pass it as
    ``threshold`` and it is drawn — that line, not the peak height, is what makes a
    detection claim.

    Parameters
    ----------
    result
        A :class:`~albireo.scan.K2ScanResult`.
    injected_k2
        Known or injected K₂ to mark — simulations and recovery tests.
    threshold
        A calibrated detection threshold on ``D``.
    ax
        Existing axis.
    label
        Legend label for the curve.

    Returns
    -------
    (Figure, Axes)
    """
    plt = _plt()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.2, 4.0))

    ax.plot(
        np.asarray(result.k2_grid),
        np.asarray(result.detection),
        "o-",
        color="C0",
        ms=3,
        label=label,
    )
    ax.axhline(0.0, color="k", ls=":", lw=1.0)
    if injected_k2 is not None:
        ax.axvline(
            float(injected_k2),
            color="k",
            ls="--",
            lw=1.0,
            label=f"injected $K_2$ = {float(injected_k2):g} km/s",
        )
    if threshold is not None:
        ax.axhline(float(threshold), color="C3", ls="-.", lw=1.2, label="calibrated threshold")
    ax.set_xlabel("trial $K_2$ [km/s]")
    ax.set_ylabel("$D(K_2)$")
    if label is not None or injected_k2 is not None or threshold is not None:
        ax.legend(fontsize=8)
    return fig, ax


def plot_detection_limit(limit, *, observed=None, axes=None):
    """The two halves of a calibrated search: the null distribution, and completeness.

    Left, how large the detection statistic gets when there is nothing to find, with the
    calibrated threshold on it; right, how often an injected companion of a given light
    fraction clears that threshold, with the quoted limit marked. Read together they are
    the claim — a peak means nothing without the left panel, and a non-detection means
    nothing without the right one.

    The two panels are usually on wildly different scales: a real companion's ``D`` can sit
    orders of magnitude above the entire null distribution. Rather than rescale the axis
    into uselessness, ``observed`` is annotated rather than drawn when it falls off the
    histogram.

    Parameters
    ----------
    limit
        A :class:`~albireo.calibrate.DetectionLimit`.
    observed
        The peak ``D`` measured on the real data
        (:attr:`~albireo.scan.K2ScanResult.detection_peak`), marked against the null.
    axes
        A pair of existing axes, in the order (null, completeness).

    Returns
    -------
    (Figure, (Axes, Axes))
    """
    plt = _plt()
    if axes is None:
        fig, (ax_null, ax_comp) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    else:
        ax_null, ax_comp = axes
        fig = ax_null.figure

    null = np.asarray(limit.null_peaks)
    ax_null.hist(null, bins=18, color="0.75", edgecolor="0.4", label="no companion")
    ax_null.axvline(
        float(limit.threshold),
        color="C3",
        lw=2,
        label=f"threshold (FAP {limit.false_alarm:g})",
    )
    ax_null.set_xlabel("peak $D$ over the $K_2$ grid")
    ax_null.set_ylabel("trials")
    ax_null.set_title(f"null distribution ({limit.n_null} trials)")
    if observed is not None:
        observed = float(observed)
        fap = limit.false_alarm_probability(observed)
        floor = fap <= limit.fap_floor
        if null.min() <= observed <= null.max():
            ax_null.axvline(observed, color="C2", lw=2, label=f"observed ($D$ = {observed:.0f})")
        else:
            ax_null.annotate(
                f"observed peak: $D$ = {observed:.0f}\n"
                + (
                    f"(FAP $\\leq$ {limit.fap_floor:.3g}, the trial-count floor)"
                    if floor
                    else f"(FAP = {fap:.3g})"
                ),
                xy=(0.97, 0.74),
                xycoords="axes fraction",
                ha="right",
                fontsize=8,
                color="C2",
            )
    ax_null.legend(fontsize=8)

    ell2 = 100.0 * np.asarray(limit.ell2_grid)
    ax_comp.plot(ell2, np.asarray(limit.completeness), "o-", color="C0")
    ax_comp.axhline(limit.confidence, color="0.5", ls="--", lw=1)
    if np.isfinite(limit.ell2_limit):
        pos = 100.0 * limit.ell2_limit
        ax_comp.axvline(pos, color="C3", lw=2)
        ax_comp.annotate(
            (f"{pos:.2f}%" if limit.limit_is_bracketed else f"$\\leq$ {pos:.2f}% (not bracketed)"),
            xy=(pos, 0.32),
            xytext=(4, 0),
            textcoords="offset points",
            color="C3",
            fontsize=9,
        )
    ax_comp.set_xlabel(r"injected companion light fraction $\ell_2$ [%]")
    ax_comp.set_ylabel(f"fraction detected ($D$ > {limit.threshold:.0f})")
    ax_comp.set_ylim(-0.03, 1.05)
    ax_comp.set_title(f"completeness, {limit.signal_peaks.shape[1]} trials per rung")
    fig.tight_layout()
    return fig, (ax_null, ax_comp)


def plot_forecast(forecast, *, axes=None):
    """What an observing design buys: the band, the worst mode, and the mode ladder.

    Three panels, answering the three forms of "will more epochs help".

    Left, the **pointwise band** each component would come back with, against the prior
    alone. The prior line is the one to read first: wherever the forecast band touches
    it, the design is learning nothing there and no exposure time changes that — it is
    the D42 lesson drawn as a figure. The shaded span marks the region the summaries are
    taken over; outside it the band climbs to the prior because the model grid is wider
    than the data, which is expected rather than alarming.

    Middle, the **worst-determined mode** — the spectral pattern the design cannot pin
    down. For two components it is the ``k = 0`` exchange of ``docs/math.md`` §5.1:
    equal and opposite low-frequency structure, degenerate for every design. Seeing it
    is worth more than its eigenvalue, because it says which *shape* of error the
    disentangled spectra will carry.

    Right, the **mode ladder** on a log axis. This is the panel that ranks designs: the
    leading mode barely moves, and what a good cadence does is drag the rest of the
    ladder down. With a baseline present both ladders are drawn, so the gap between them
    is the value of the planned epochs.

    Parameters
    ----------
    forecast
        A :class:`~albireo.forecast.SensitivityForecast`.
    axes
        Three existing axes, in the order (band, mode, ladder).

    Returns
    -------
    (Figure, ndarray of Axes)
    """
    plt = _plt()
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].figure
    ax_band, ax_mode, ax_ladder = axes[0], axes[1], axes[2]

    wave = np.asarray(forecast.grid.wave)
    base = forecast.baseline
    for i, name in enumerate(forecast.component_labels):
        color = _color(i)
        if base is not None:
            ax_band.plot(
                wave, base.component_std[i], color=color, lw=0.9, ls=":", label=f"{name}, in hand"
            )
        ax_band.plot(wave, forecast.component_std[i], color=color, lw=1.2, label=name)
        ax_band.plot(wave, forecast.prior_std[i], color=color, lw=0.8, alpha=0.4)
    inside = np.flatnonzero(forecast.region.any(axis=0))
    if inside.size:
        ax_band.axvspan(wave[inside[0]], wave[inside[-1]], color="0.85", alpha=0.4, zorder=0)
    ax_band.set_yscale("log")
    ax_band.set_xlabel("wavelength [Å]")
    ax_band.set_ylabel(r"forecast $\sigma$ on $d_i$")
    ax_band.set_title("pointwise band (faint = prior alone)")
    ax_band.legend(fontsize=7, ncol=2)

    if forecast.mode_std.size:
        mode = forecast.mode_vectors[0]
        for i, name in enumerate(forecast.component_labels):
            ax_mode.plot(wave, mode[i], color=_color(i), lw=1.0, label=name)
        ax_mode.axhline(0.0, color="0.6", lw=0.8)
        ax_mode.set_xlabel("wavelength [Å]")
        ax_mode.set_ylabel("mode amplitude")
        ax_mode.set_title(rf"worst-determined mode, $\sigma$ = {forecast.worst_mode_std:.3g}")
        ax_mode.legend(fontsize=8)

        rungs = np.arange(1, forecast.mode_std.size + 1)
        if base is not None:
            ax_ladder.plot(rungs, base.mode_std, "o:", color="0.45", label="in hand")
            ax_ladder.plot(
                rungs, forecast.mode_std, "o-", color="C0", label=f"+{forecast.n_planned} planned"
            )
        else:
            ax_ladder.plot(rungs, forecast.mode_std, "o-", color="C0", label="this design")
        ax_ladder.plot(rungs, forecast.prior_mode_std, "s--", color="C3", label="prior alone")
        ax_ladder.set_yscale("log")
        ax_ladder.set_xticks(rungs)
        ax_ladder.set_xlabel("mode (worst first)")
        ax_ladder.set_ylabel(r"$\sigma$ of the mode")
        ax_ladder.set_title("what the epochs move")
        ax_ladder.legend(fontsize=8)
    fig.tight_layout()
    return fig, axes


_ORBIT_SITES = ("period", "t_conj", "secosw", "sesinw", "k", "ecc", "omega")


def _default_corner_vars(idata):
    """The orbital sites present in ``idata``, or None to let arviz choose.

    Defaulting to the orbital block rather than every site matters here: with the component
    spectra marginalized out the sampled space is small, but the smoothness
    hyperparameters, the per-epoch light fractions, and the LSF anchors would still crowd
    a corner plot past the point of being readable.
    """
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return None
    available = set(posterior)
    return [name for name in _ORBIT_SITES if name in available] or None


def plot_corner(idata, *, var_names=None, **kwargs):
    """Pairwise posterior for the orbital parameters, via arviz.

    Parameters
    ----------
    idata
        Whatever :func:`albireo.results.to_inference_data` returned, or anything else
        ``arviz.plot_pair`` accepts.
    var_names
        Sites to include. Defaults to the orbital sites present in the object.
    **kwargs
        Passed straight to ``arviz.plot_pair``.

    Returns
    -------
    object
        Whatever ``arviz.plot_pair`` returns — **the one function in this module that does
        not return** ``(fig, axes)``. arviz 0.x returned an array of matplotlib axes; arviz
        1.x returns its own ``PlotMatrix``. Adapting between them would mean depending on
        arviz internals that are in the middle of changing, so this is a thin passthrough
        and the caller works with whatever their arviz gives them. Nothing is passed on the
        caller's behalf beyond ``var_names``, for the same reason: styling arguments that
        were valid in arviz 0.x (``kind``, ``marginals``) raise in 1.x.
    """
    az = _require_arviz()
    _plt()
    if var_names is None:
        var_names = _default_corner_vars(idata)
    return az.plot_pair(idata, var_names=var_names, **kwargs)


# ---------------------------------------------------------------------------
# TODCOR: the surface and the table
# ---------------------------------------------------------------------------


def plot_todcor_surface(surface, *, truth=None, ax=None, levels: int = 30):
    """The two-dimensional correlation surface of one epoch, with its maximum marked.

    Parameters
    ----------
    surface
        A :class:`albireo.todcor.TodcorSurface` from :func:`albireo.todcor_surface`.
    truth
        Optional ``(v1, v2)`` to mark, for simulated data.
    ax
        Existing axis to draw into.
    levels
        Number of filled contour levels of ``R^2``.

    Returns
    -------
    (Figure, Axes)
    """
    plt = _plt()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5.6, 5.0))
    r2 = np.asarray(surface.r_squared)
    finite = r2[np.isfinite(r2)]
    lo = float(np.percentile(finite, 5.0)) if finite.size else 0.0
    hi = float(np.nanmax(r2)) if finite.size else 1.0
    contour = ax.contourf(
        surface.v2, surface.v1, np.clip(r2, lo, hi), levels=levels, cmap="viridis"
    )
    fig.colorbar(contour, ax=ax, label="$R^2 = 1 - \\chi^2/\\chi^2_{\\rm null}$")
    peak = surface.peak
    ax.plot(peak[1], peak[0], "w+", ms=12, mew=1.5, label="maximum")
    if truth is not None:
        ax.plot(truth[1], truth[0], "rx", ms=9, mew=1.5, label="truth")
    ax.plot(surface.v2, surface.v2, color="w", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel(f"{surface.names[1]} velocity [km/s]")
    ax.set_ylabel(f"{surface.names[0]} velocity [km/s]")
    ax.legend(fontsize=8, loc="upper left")
    return fig, ax


def plot_velocity_table(table, *, period=None, t_conj=None, orbit=None, truth=None, axes=None):
    """A velocity table against time or phase, with its errors and flags.

    Parameters
    ----------
    table
        A :class:`albireo.todcor.VelocityTable`.
    period, t_conj
        Fold on this ephemeris; otherwise plot against BJD. Taken from ``orbit`` when one
        is given.
    orbit
        An :class:`albireo.rvorbit.RVOrbit` to overplot, and to fold on.
    truth
        Optional ``(n_comp, n_epochs)`` injected velocities, for simulated data.
    axes
        A pair of axes ``(curve, residuals)``; residuals are drawn only with ``orbit``.

    Returns
    -------
    (Figure, Axes)
    """
    plt = _plt()
    if orbit is not None:
        period, t_conj = orbit.period, orbit.t_conj
    if axes is None:
        if orbit is not None:
            fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True, height_ratios=[3, 1])
        else:
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            axes = (ax,)
    else:
        fig = axes[0].figure
    ax = axes[0]
    folded = period is not None and t_conj is not None
    x = phase_of(table.bjd, period, t_conj) if folded else table.bjd
    good = table.good
    for i, name in enumerate(table.names):
        color = _color(i)
        ax.errorbar(
            x[good],
            table.velocity[i, good],
            yerr=table.sigma[i, good],
            fmt="o",
            ms=4,
            capsize=2,
            color=color,
            label=name,
        )
        if (~good).any():
            ax.errorbar(
                x[~good],
                table.velocity[i, ~good],
                yerr=table.sigma[i, ~good],
                fmt="o",
                ms=4,
                mfc="none",
                capsize=2,
                color=color,
                alpha=0.6,
            )
        if truth is not None:
            ax.plot(x, np.asarray(truth)[i], "_", color="k", ms=10, alpha=0.7)
    if orbit is not None:
        if folded:
            dense_phase = np.linspace(0.0, 1.0, 400)
            dense_t = t_conj + dense_phase * period
            model = orbit.predict(dense_t)
            for i in range(model.shape[0]):
                ax.plot(dense_phase, model[i], color=_color(i), lw=1.0, alpha=0.7)
        else:
            dense_t = np.linspace(table.bjd.min(), table.bjd.max(), 1500)
            model = orbit.predict(dense_t)
            for i in range(model.shape[0]):
                ax.plot(dense_t, model[i], color=_color(i), lw=1.0, alpha=0.7)
        if len(axes) > 1:
            res = axes[1]
            for i in range(orbit.residuals.shape[0]):
                res.errorbar(
                    x,
                    orbit.residuals[i],
                    yerr=table.sigma[i],
                    fmt="o",
                    ms=3,
                    capsize=2,
                    color=_color(i),
                )
            res.axhline(0.0, color="0.6", lw=0.8)
            res.set_ylabel("O - C [km/s]")
            res.set_xlabel("phase from conjunction" if folded else "BJD")
    if len(axes) == 1 or orbit is None:
        ax.set_xlabel("phase from conjunction" if folded else "BJD")
    if folded:
        ax.set_xlim(0.0, 1.0)
    ax.set_ylabel("radial velocity [km/s]")
    ax.legend(fontsize=8)
    return fig, axes
