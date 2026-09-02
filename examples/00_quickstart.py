"""Quickstart: load the packaged example, fit the orbit at MAP, and plot the spectra.

This is the shortest complete analysis in albireo. It loads the example dataset that
ships inside the package, fits the orbit with the component spectra marginalized, and
writes two figures. It runs in about a minute on a laptop, most of which is JAX
compiling the model on its first call.

The script stops at the maximum a posteriori fit. The posterior over the orbit, which is
the purpose of the package, is sampled in ``examples/01_sb2_end_to_end.py``.

Usage
-----
    python examples/00_quickstart.py
"""

from __future__ import annotations

import importlib.util
import time

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import albireo as ab


def main() -> None:
    print(f"albireo {ab.__version__}")

    # 1. Data. The example is packaged with albireo, so this works offline. With
    #    `with_truth=True` the injected orbit and component spectra are returned as well,
    #    which lets the script check its own result at the end.
    dataset, truth = ab.load_example("sb2_sim", with_truth=True)
    print(dataset.summary())

    # 2. The model grid. The example records the grid it was generated on. For observed
    #    data, `ab.LogGrid.covering(dataset, ...)` builds one with margins sized from the
    #    velocity budget and the LSF width.
    grid = ab.LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))

    # 3. The model. Two quantities are not inferred and must be supplied: the light
    #    fractions (with a constant light ratio the data constrain only the products
    #    l_i * d_i; docs/math.md section 5.2) and the LSF width. `v_rel_max_kms` is the
    #    velocity budget for which the static solver structure is built; orbits that
    #    exceed it are rejected with a non-finite log density rather than mis-solved.
    model = ab.MarginalOrbitModel(
        grid,
        dataset,
        light_fractions=truth["light_fractions"],
        lsf_sigma_v={"DEMO": 6.5},
        v_rel_max_kms=160.0,
    )

    # 4. Priors. These are broad, as befits a demonstration. `secosw` and `sesinw` are
    #    sqrt(e) cos(omega) and sqrt(e) sin(omega), which sample better than (e, omega)
    #    because the pair has no boundary at e = 0 and no wrap in omega.
    #
    #    The parameterization is singular at exactly e = 0, where omega is undefined and
    #    the gradient is not finite. Never initialize at secosw = sesinw = 0; start
    #    slightly off the origin, as here, even for a binary believed to be circular.
    #    numpyro reports the singular start as "Cannot find valid initial parameters".
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

    print("\nfitting (the first evaluation compiles the model)...")
    start = time.perf_counter()
    fit = ab.run_map(model.model(priors), init=init)
    print(f"  {fit.num_steps} L-BFGS steps in {time.perf_counter() - start:.1f} s")

    k_fit = np.asarray(fit.params["k"])
    k_true = np.asarray(truth["k"], dtype=float)
    print(f"\n  period    {float(fit.params['period']):8.4f} d   (truth {truth['period']:.4f})")
    print(f"  K_1       {k_fit[0]:8.3f} km/s (truth {k_true[0]:.3f})")
    print(f"  K_2       {k_fit[1]:8.3f} km/s (truth {k_true[1]:.3f})")

    # 5. The component spectra conditional on the fitted orbit, with their pointwise
    #    uncertainties. Where the epochs give no leverage the smoothness prior sets the
    #    answer, and the uncertainty band identifies those regions.
    marginal = model.marginal(fit.params)
    d_hat = np.asarray(marginal.d_hat)
    std = np.asarray(ab.spectra_std(marginal))
    print(f"\n  recovered spectra: {d_hat.shape[0]} components x {d_hat.shape[1]} pixels")
    print(f"  median uncertainty: {np.median(std):.4f} in normalized flux")

    # 6. Save the fit and export the spectra.
    ab.save_fit(fit, "quickstart_map.npz")
    ab.write_ascii("quickstart_spectra.txt", grid, d_hat, std)
    print("\n  wrote quickstart_map.npz, quickstart_spectra_1.txt, quickstart_spectra_2.txt")

    if importlib.util.find_spec("matplotlib") is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, _ = ab.plot_spectra(grid, d_hat, std=std, truth=truth["components"])
        fig.set_layout_engine("constrained")
        fig.savefig("quickstart_spectra.png", dpi=150)
        plt.close(fig)

        fig, _ = ab.plot_residual_zscores(
            model.problem_at(fit.params), marginal.d_hat, bjd=dataset.bjd
        )
        fig.set_layout_engine("constrained")
        fig.savefig("quickstart_residuals.png", dpi=150)
        plt.close(fig)
        print("  wrote quickstart_spectra.png, quickstart_residuals.png")
    else:
        print('  (install "albireo[plots]" for the figures)')

    # The tolerances are loose because this is a MAP fit on 12 epochs of a small problem.
    assert abs(k_fit[0] - k_true[0]) / k_true[0] < 0.05, "K_1 off by more than 5%"
    assert abs(k_fit[1] - k_true[1]) / k_true[1] < 0.05, "K_2 off by more than 5%"
    assert abs(float(fit.params["period"]) - truth["period"]) < 0.05, "period off by > 0.05 d"
    print("\nok")


if __name__ == "__main__":
    main()
