"""Tests for what the pipeline driver reports about a fit: truth comparisons, flags, figures.

Each test covers one defect found in a review of :mod:`albireo.pipeline` and fails on the
code before its correction:

1. **An exchanged pair is compared element by element.** Where the components are
   recovered in the other order, the argument of periastron and the conjunction of the
   fit's first component are those of the injected secondary, and the truth is moved to
   them.
2. **The flags state what was found.** The mirror-orbit flag of the conjunction scan
   compares the best trial with the best one away from it, not with its neighbour. A
   z-score rms that is not finite stops the star. The ambiguity flag of the candidate fits
   names its reference period by its chi-square, since on the search route the
   disentangling decides after it.
3. **A failed figure is closed**, so that an in-process batch does not accumulate open
   figures.

No test here runs a disentangling.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import albireo as ab
from albireo.kepler import _t_peri_from_t_conj_numpy, true_anomaly_numpy
from albireo.pipeline import (
    Analysis,
    ComponentConfig,
    PipelineConfig,
    StarConfig,
    _Context,
    _Log,
)


def _files_star(name="x", **kwargs):
    options = {
        "spectra": ["a.fits"],
        "period": (6.0, 6.6),
        "components": [ComponentConfig("A", 0.6), ComponentConfig("B", 0.4)],
    }
    options.update(kwargs)
    return StarConfig(name=name, **options)


def _ctx(directory, *, name="x", settings=None):
    star = _files_star(name)
    directory.mkdir(parents=True, exist_ok=True)
    return _Context(
        star=star,
        config=PipelineConfig(stars=[star], output=directory),
        settings=settings or Analysis(),
        directory=directory,
        log=_Log(star.name, directory / "log.txt", progress=False),
    )


def _keplerian(t, *, period, t_conj, ecc, omega, k):
    """Radial velocity of the component whose ``omega`` and conjunction are given."""
    t_peri = _t_peri_from_t_conj_numpy(t_conj, period=period, ecc=ecc, omega=omega)
    nu = true_anomaly_numpy(t, period=period, t_peri=t_peri, ecc=ecc)
    return k * (np.cos(nu + omega) + ecc * np.cos(omega))


# ---------------------------------------------------------------------------
# 1. an exchanged pair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ecc", [0.0, 0.15, 0.6])
def test_an_exchanged_truth_moves_omega_and_the_conjunction_to_the_secondary(ecc):
    """The elements of the exchanged truth describe the injected secondary's curve.

    A fit whose first component is the injected secondary converges to ``omega + pi`` and
    to that component's own conjunction. The exchanged truth must carry those, or every
    exchanged system is reported 180 degrees off in omega and up to half a period off in
    the conjunction.
    """
    from albireo.pipeline import _exchanged_truth, _wrap_angle, _wrap_period

    period, omega, k = 6.31, 0.7, (30.0, 55.0)
    t_conj = 2455001.3
    truth = {
        "k": list(k),
        "period": period,
        "ecc": ecc,
        "omega": omega,
        "t_conj": t_conj,
        "velocities": np.zeros((2, 3)),
    }
    swapped = _exchanged_truth(truth, ["A", "B"])
    assert swapped["k"] == [55.0, 30.0]
    assert truth["omega"] == omega and truth["t_conj"] == t_conj, "the input is not changed"

    t = t_conj + np.linspace(0.0, 2.0 * period, 61)
    primary = _keplerian(t, period=period, t_conj=t_conj, ecc=ecc, omega=omega, k=k[0])
    secondary = -k[1] / k[0] * primary  # the companion moves opposite, by the mass ratio
    as_first = _keplerian(
        t,
        period=period,
        t_conj=swapped["t_conj"],
        ecc=ecc,
        omega=swapped["omega"],
        k=swapped["k"][0],
    )
    np.testing.assert_allclose(as_first, secondary, atol=1e-9)
    assert abs(_wrap_angle(swapped["omega"] - omega)) == pytest.approx(math.pi, abs=1e-12)
    if ecc == 0.0:
        # On a circular orbit the two conjunctions are half a period apart.
        shift = abs(_wrap_period(swapped["t_conj"] - t_conj, period))
        assert shift == pytest.approx(0.5 * period)


def test_an_exchanged_truth_without_omega_keeps_an_eccentric_conjunction():
    """On an eccentric orbit the other conjunction needs omega; without it nothing is moved."""
    from albireo.pipeline import _exchanged_truth

    truth = {"k": [30.0, 55.0], "period": 6.31, "ecc": 0.3, "t_conj": 2455001.3}
    assert _exchanged_truth(truth, ["A", "B"])["t_conj"] == truth["t_conj"]
    circular = {"k": [30.0, 55.0], "period": 6.31, "t_conj": 2455001.3}
    moved = _exchanged_truth(circular, ["A", "B"])["t_conj"]
    assert moved == pytest.approx(2455001.3 - 0.5 * 6.31)
    assert "omega" not in _exchanged_truth(circular, ["A", "B"])


# ---------------------------------------------------------------------------
# 2. the flags
# ---------------------------------------------------------------------------


def _scan_fit(values, *, z_rms=1.0):
    """Enough of a fit for ``_assess_fit``: a phase scan and nothing else to flag."""
    return SimpleNamespace(
        z_rms=z_rms,
        dis=SimpleNamespace(component_names=(), ordered_components=()),
        hyper={},
        phase_scan=SimpleNamespace(values=np.asarray(values, dtype=float)),
        k_scan=None,
    )


def test_the_mirror_flag_compares_the_best_phase_with_the_best_away_from_it(tmp_path):
    """A single broad peak is not a mirror; two peaks half a period apart are.

    The 42 trials of the conjunction scan are 1/42 of a period apart, so near a smooth
    peak the second-best trial is the neighbour of the best and within 1 nat of it
    whenever the peak is broad. The mirror orbit lies half a period away.
    """
    from albireo.pipeline import _assess_fit

    phase = np.arange(42) / 42.0
    for centre in (0.0, 0.31, 0.77):
        single = 50.0 * np.cos(2.0 * np.pi * (phase - centre))
        ctx = _ctx(tmp_path / f"single{centre}")
        _assess_fit(ctx, _scan_fit(single))
        assert not any("mirror" in f for f in ctx.flags), ctx.flags

    from albireo.pipeline import _mirror_within

    assert not _mirror_within(50.0 * np.cos(2.0 * np.pi * phase), 1.0)
    # Two equal modes half a period apart, and an eccentric mirror 0.4 of a period away.
    for offset in (0.5, 0.4):
        twin = np.maximum(
            300.0 * np.cos(2.0 * np.pi * (phase - 0.2)),
            300.0 * np.cos(2.0 * np.pi * (phase - 0.2 - offset)) - 0.5,
        )
        assert _mirror_within(twin, 1.0)
        ctx = _ctx(tmp_path / f"twin{offset}")
        _assess_fit(ctx, _scan_fit(twin))
        assert sum("component-swapped mirror" in f for f in ctx.flags) == 1, ctx.flags
    # A mode 2 nats below the best, and trials whose value is not finite, are not mirrors.
    clear = np.maximum(
        300.0 * np.cos(2.0 * np.pi * (phase - 0.2)),
        300.0 * np.cos(2.0 * np.pi * (phase - 0.7)) - 2.0,
    )
    assert not _mirror_within(clear, 1.0)
    clear[5] = np.nan
    assert not _mirror_within(clear, 1.0)
    assert not _mirror_within([np.nan, np.nan], 1.0) and not _mirror_within([3.0], 1.0)
    # The rival lies at least a quarter period away: of 42 trials, 11 (0.262 P), not 10.
    flat = np.full(42, -100.0)
    flat[0] = 0.0
    flat[10] = -0.5
    assert not _mirror_within(flat, 1.0)
    flat[11] = -0.5
    assert _mirror_within(flat, 1.0)


def test_a_z_score_rms_that_is_not_finite_stops_the_star(tmp_path):
    """No finite residual z-score means the fit cannot be checked, so it is not used."""
    from albireo.pipeline import _assess_fit, _refuse_diverged

    ctx = _ctx(tmp_path / "nan")
    fit = _scan_fit(np.zeros(42) + np.arange(42.0) * 10.0, z_rms=float("nan"))
    _assess_fit(ctx, fit)
    assert not any("noise model" in f for f in ctx.flags), "the refusal names it instead"
    with pytest.raises(RuntimeError, match="no finite residual z-score") as excinfo:
        _refuse_diverged(ctx, fit)
    assert "were not run" in str(excinfo.value)
    assert ctx.flags[-1].startswith("the disentangling diverged")
    # A finite value at or below the ceiling passes, and one above it is refused as before.
    _refuse_diverged(ctx, fit, z_rms=10.0)
    with pytest.raises(RuntimeError, match=r"residual z-score rms 45\.0, above the z_rms_max"):
        _refuse_diverged(ctx, fit, z_rms=45.0)


def test_the_ambiguity_flag_names_its_reference_by_its_chi_square(monkeypatch, tmp_path):
    """On the search route the comparison by the disentangling may choose another period."""
    import albireo.rvorbit as rvorbit_module
    from albireo.pipeline import _orbit_over_candidates

    outcome = {6.0: (6.30, 90.0), 12.6: (12.62, 110.0)}

    def assign(table, *, period, **kwargs):
        fitted, chi2 = outcome[round(float(period), 4)]
        orbit = SimpleNamespace(period=fitted, chi2=chi2, k=np.array([40.0, 50.0]), ecc=0.1)
        return table, orbit, None

    monkeypatch.setattr(rvorbit_module, "assign_components", assign)
    ctx = _ctx(tmp_path / "ambiguous")
    table = SimpleNamespace(n_components=2, bjd=np.arange(10.0))
    _orbit_over_candidates(ctx, table, [6.0, 12.6], circular=False)
    (flag,) = ctx.flags
    assert "of the lowest chi-square one, 6.3000 d" in flag and "chosen" not in flag
    assert "12.6200 d at +20.0" in flag


# ---------------------------------------------------------------------------
# 3. figures
# ---------------------------------------------------------------------------


def test_a_figure_that_fails_is_closed(tmp_path, monkeypatch):
    """A figure whose drawing fails, or a plot function that fails after opening one.

    Each is flagged and the other figures are written. Under an in-process batch the
    figures left open by failures accumulate, with the memory they hold.
    """
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from albireo import plotting
    from albireo.pipeline import _write_plots

    def broken_savefig(*args, **kwargs):
        raise ValueError("the figure could not be drawn")

    def spectra(*args, **kwargs):
        fig, ax = plt.subplots()
        fig.savefig = broken_savefig
        return fig, ax

    def fails_after_opening(*args, **kwargs):
        plt.subplots()
        plt.subplots()
        raise RuntimeError("the plot failed after opening its figures")

    def drawn(*args, **kwargs):
        return plt.subplots()

    monkeypatch.setattr(plotting, "plot_spectra", spectra)
    monkeypatch.setattr(plotting, "plot_residual_zscores", fails_after_opening)
    monkeypatch.setattr(plotting, "plot_velocity_table", drawn)
    grid = ab.LogGrid.from_wavelength_range(5150.0, 5160.0, dv_kms=5.0)
    fit = SimpleNamespace(
        dis=SimpleNamespace(
            grid=grid,
            component_names=("A", "B"),
            dataset=SimpleNamespace(bjd=np.arange(3.0)),
            model=SimpleNamespace(problem_at=lambda theta: None),
        ),
        theta={},
        phase_scan=None,
        marginal=lambda: SimpleNamespace(d_hat=np.zeros((2, grid.n))),
        _spectra_and_std=lambda: (np.zeros((2, grid.n)), np.ones((2, grid.n))),
    )
    ctx = _ctx(tmp_path / "figures")
    opened = set(plt.get_fignums())
    _write_plots(ctx, fit, SimpleNamespace(), None, [], None)
    assert set(plt.get_fignums()) == opened, "every figure the stage opened is closed"
    assert [f.split(":")[0] for f in ctx.flags] == [
        "figure spectra not written",
        "figure residuals not written",
    ]
    assert (ctx.directory / "velocities.png").is_file()
    assert "plot_velocities" in ctx.files and "plot_spectra" not in ctx.files
