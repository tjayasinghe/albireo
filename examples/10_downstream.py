"""Propagating a disentangling posterior into a derived quantity (roadmap Tier 2 item 8).

The disentangled spectrum is not the answer; it is the input to an atmosphere code, and the
number that comes out of *that* is what ends up in a table. The uncertainty on the
disentangled spectrum is dropped at exactly that joint, and the literature says so in its
own words — Mahy et al. (2020), TMBM III §3.1: "the uncertainties that could arise from the
normalisation procedure are not taken into account in the global uncertainties on the
presented properties".

albireo can close that gap because it has a posterior, not just a best fit. The recipe is
old (Kiran et al. 2016 refitted a noise-perturbed disentangled profile 500 times); what is
new is *what gets drawn*.

**The point of this script is the difference between those two things, and it is measured
rather than argued.** Both give you N spectra and a spread. But:

* Kiran's loop adds **independent** noise per pixel, at the amplitude of the error bar.
* :func:`albireo.draw_spectra` returns draws from the **joint** posterior — correlated
  across wavelength, and correlated across the two stars, because the draw is
  ``d_hat + L^-T z`` on the vector stacked over all components.

Disentangling error has a genuine low-frequency null space, and low-frequency error is
precisely what moves a continuum and therefore a temperature. So for any quantity that
*integrates* the spectrum — an equivalent width, and through it log g — the two recipes do
not merely differ in polish, they differ in answer. This script measures by how much.

Equivalent width is the right stand-in for the atmosphere code here, and not only because it
is cheap: D40 established that EW is the quantity that reaches the atmosphere code, and
quantified an 11.5% EW error as a systematic in log g. A propagated EW uncertainty is an
honest miniature of a propagated log g uncertainty, and it runs in seconds instead of
requiring a Fortran grid search that is not installed on this machine.

What this script does NOT do is run GSSP or iSpec. It writes the files they read, verifies
the two things about those files that fail silently, and prints the fitting loop you would
run next. See ``docs/tutorials/downstream.md`` for that half.

Environment
-----------
ALBIREO_EXAMPLE_FAST=1
    CI-sized run: fewer draws and fewer L-BFGS steps. The conclusions are unchanged; the
    Monte-Carlo spread is quoted with a wider error on itself.

Usage
-----
    python examples/10_downstream.py
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import albireo as ab

FAST = bool(os.environ.get("ALBIREO_EXAMPLE_FAST"))
N_DRAWS = 24 if FAST else 200
MAX_STEPS = 120 if FAST else 400

# A window around the deepest line in the packaged example's first component. An
# equivalent width needs a line and a stretch of continuum either side of it.
EW_HALF_WIDTH_A = 1.2


def equivalent_width(wave, flux, center, half_width):
    """Integral of ``1 - flux`` over a window, in angstrom. Positive for absorption."""
    inside = np.abs(wave - center) <= half_width
    return float(np.trapezoid(1.0 - flux[inside], wave[inside]))


def main() -> None:
    print(f"albireo {ab.__version__}   ({'fast' if FAST else 'full'} run, N = {N_DRAWS} draws)")

    dataset, truth = ab.load_example("sb2_sim", with_truth=True)
    grid = ab.LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))
    wave = np.asarray(grid.wave)

    model = ab.MarginalOrbitModel(
        grid,
        dataset,
        light_fractions=truth["light_fractions"],
        lsf_sigma_v={"DEMO": 6.5},
        v_rel_max_kms=160.0,
    )
    priors = {
        "period": dist.Uniform(5.5, 6.5),
        "t_conj": dist.Uniform(-1.0, 1.0),
        "secosw": dist.Uniform(-0.8, 0.8),
        "sesinw": dist.Uniform(-0.8, 0.8),
        "k": dist.Uniform(jnp.array([10.0, 10.0]), jnp.array([90.0, 90.0])),
        "log_tau": dist.Normal(jnp.log(300.0) * jnp.ones(2), 3.0),
        "log_eta": dist.Normal(jnp.log(5.0) * jnp.ones(2), 3.0),
    }
    init = {
        "period": 6.05,
        "t_conj": 0.1,
        "secosw": 0.2,
        "sesinw": 0.2,
        "k": jnp.array([38.0, 58.0]),
        "log_tau": jnp.log(300.0) * jnp.ones(2),
        "log_eta": jnp.log(5.0) * jnp.ones(2),
    }

    print("\nfitting...")
    start = time.perf_counter()
    fit = ab.run_map(model.model(priors), init=init, max_steps=MAX_STEPS)
    print(f"  {fit.num_steps} L-BFGS steps in {time.perf_counter() - start:.1f} s")

    marginal = model.marginal(fit.params)
    d_hat = np.asarray(marginal.d_hat)
    std = np.asarray(ab.spectra_std(marginal))

    # ---------------------------------------------------------------- the export
    print("\n--- 1. the files the atmosphere codes read ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gssp = ab.write_gssp(tmp / "component.dat", grid, d_hat)
        ispec = ab.write_ispec(tmp / "component.txt", grid, d_hat, std)

        g_wave = np.loadtxt(gssp[0])[:, 0]
        g_steps = np.diff(g_wave)
        i_first = ispec[0].read_text().splitlines()[1].split("\t")
        src_drift = np.ptp(np.diff(wave)) / np.mean(np.diff(wave))

        print(f"  GSSP  {gssp[0].name}: 2 columns, angstrom, first lambda = {g_wave[0]:.3f}")
        print(
            f"        equidistant to {np.ptp(g_steps):.1e} A, from a source grid that "
            f"drifts {src_drift:.2%} across the window"
        )
        print(f"  iSpec {ispec[0].name}: 3 tab-separated columns, first row = {i_first[0]} nm")
        print(f"        (the same pixel; {float(i_first[0]) * 10:.3f} A -- a factor of ten apart)")
        # The unit trap, asserted rather than trusted: iSpec does no conversion on the text
        # path, so an angstrom value here would sit ten times outside every model grid.
        assert abs(float(i_first[0]) * 10.0 - g_wave[0]) < 1e-3, "iSpec column must be nm"

    # ---------------------------------------------------------------- the draws
    print(f"\n--- 2. {N_DRAWS} joint posterior draws ---")
    start = time.perf_counter()
    draws = np.asarray(ab.draw_spectra(marginal, jax.random.key(0), N_DRAWS))
    print(f"  drew {draws.shape} in {time.perf_counter() - start:.2f} s")

    # The line to integrate: deepest feature of component 1, away from the grid edges.
    interior = slice(grid.n // 10, -grid.n // 10)
    center = float(wave[interior][np.argmin(d_hat[0][interior])])
    print(f"  measuring the equivalent width of the line at {center:.2f} A")

    ew_joint = np.array(
        [
            [
                equivalent_width(wave, 1.0 + draws[k, c], center, EW_HALF_WIDTH_A)
                for k in range(N_DRAWS)
            ]
            for c in range(d_hat.shape[0])
        ]
    )
    ew_hat = [equivalent_width(wave, 1.0 + d_hat[c], center, EW_HALF_WIDTH_A) for c in range(2)]

    # The comparison: the same N spectra, but perturbed the way the literature does it --
    # independent Gaussian noise per pixel at the pointwise band's amplitude.
    rng = np.random.default_rng(11)
    white = d_hat[None, :, :] + rng.normal(0.0, 1.0, (N_DRAWS, *d_hat.shape)) * std[None, :, :]
    ew_white = np.array(
        [
            [
                equivalent_width(wave, 1.0 + white[k, c], center, EW_HALF_WIDTH_A)
                for k in range(N_DRAWS)
            ]
            for c in range(d_hat.shape[0])
        ]
    )

    print("\n--- 3. the result: joint draws vs independent per-pixel noise ---")
    print(f"  {'':10s} {'EW (A)':>10s} {'joint sd':>12s} {'white sd':>12s} {'ratio':>8s}")
    ratios = []
    for c in range(2):
        sj, sw = ew_joint[c].std(ddof=1), ew_white[c].std(ddof=1)
        ratios.append(sj / sw)
        print(f"  component {c + 1} {ew_hat[c]:10.4f} {sj:12.5f} {sw:12.5f} {sj / sw:8.2f}x")

    corr = np.corrcoef(ew_joint[0], ew_joint[1])[0, 1]
    corr_white = np.corrcoef(ew_white[0], ew_white[1])[0, 1]
    print(f"\n  correlation between the two stars' EW across draws: {corr:+.3f}")
    print(f"  the same statistic under independent per-pixel noise:  {corr_white:+.3f}")
    print("  This is the k=0 exchange mode of D47 arriving in a derived quantity: the two")
    print("  components trade line depth almost exactly, so their *difference* is far")
    print("  better determined than either one alone. Fitting the two stars separately and")
    print("  quoting independent error bars therefore misstates the answer in both")
    print("  directions at once. Pooling the draws per component throws it away, which is")
    print("  why export_draws keeps the draw index in the filename.")

    rel_err = 1.0 / np.sqrt(2.0 * (N_DRAWS - 1))
    print(f"\n  the spread itself carries {rel_err:.1%} relative error at N = {N_DRAWS}")

    # ---------------------------------------------------------------- what next
    print("\n--- 4. the loop you would run next ---")
    with tempfile.TemporaryDirectory() as tmp:
        paths = ab.export_draws(tmp, grid, draws[:4], format="gssp")
        print(f"  ab.export_draws(outdir, grid, draws, format='gssp')  ->  {len(paths)} x 2 files")
        print(f"     {paths[0][0].name}, {paths[0][1].name}, {paths[1][0].name}, ...")
    print("  then fit every file with the same grid and settings, and take the spread of")
    print("  the resulting Teff / log g. See docs/tutorials/downstream.md.")

    # The claim this script exists to make: for an integrated quantity the two recipes
    # disagree, because disentangling error is not white. Assert it rather than print it.
    assert max(ratios) > 1.5, (
        f"joint-vs-white EW spread ratios {ratios} -- expected the joint draws to give a "
        "materially larger integrated uncertainty, which is the whole argument for drawing "
        "from the posterior rather than adding noise at the band's amplitude"
    )
    assert np.all(ew_joint.std(axis=1, ddof=1) > 0.0)
    # And the second finding, which is the sharper one: the components exchange line depth.
    # Independent per-pixel noise cannot produce this at all -- it has no cross-component
    # structure to produce it with -- so the two numbers printed above are a direct measure
    # of what the white-noise recipe is blind to.
    assert corr < -0.5, f"expected a strong anti-correlation between components, got {corr:+.3f}"
    assert abs(corr_white) < 0.5, (
        f"independent per-pixel noise should carry no cross-component structure, "
        f"got {corr_white:+.3f}"
    )
    print("\nok")


if __name__ == "__main__":
    main()
