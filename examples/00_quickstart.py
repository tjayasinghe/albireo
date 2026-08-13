"""The five-minute quickstart: load, fit, plot — no data of your own, no network.

This is the shortest thing in albireo that does something real. It loads the example
dataset that ships inside the package, fits the orbit with the component spectra
marginalized out, and writes two figures. On a laptop it takes about a minute, most of
which is JAX compiling the model the first time it is called.

It deliberately stops at MAP rather than running NUTS. MAP is enough to show the machinery
working and to get a picture; the posterior — which is the actual reason to use this
package — is what ``examples/01_sb2_end_to_end.py`` goes on to do.

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

    # 1. Data. Packaged with albireo, so this works offline and in a fresh notebook.
    #    `with_truth=True` also hands back what was injected, which is what lets the
    #    script check itself at the end.
    dataset, truth = ab.load_example("sb2_sim", with_truth=True)
    print(dataset.summary())

    # 2. The model grid. The example records the grid it was generated on; for your own
    #    data, build one with `ab.LogGrid.covering(dataset, ...)`, which sizes the margins
    #    from the velocity budget and the LSF width.
    grid = ab.LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))

    # 3. The model. Two things are not inferred here and have to be supplied: the light
    #    fractions (with a constant light ratio the data only ever constrain the products
    #    l_i * d_i — docs/math.md 5.2) and the LSF width. `v_rel_max_kms` is the velocity
    #    budget the static solver structure is built for; exceeding it is guarded, not
    #    silently wrong.
    model = ab.MarginalOrbitModel(
        grid,
        dataset,
        light_fractions=truth["light_fractions"],
        lsf_sigma_v={"DEMO": 6.5},
        v_rel_max_kms=160.0,
    )

    # 4. Priors. Deliberately broad — this is a demonstration, not a well-motivated fit.
    #    `secosw`/`sesinw` are sqrt(e)*cos(w) and sqrt(e)*sin(w), which samples far better
    #    than (e, omega) because it has no boundary at e = 0 and no wrap in omega.
    #
    #    One trap worth knowing: that parameterization is singular at *exactly* e = 0,
    #    where omega is undefined and the gradient is NaN. Never initialize at
    #    secosw = sesinw = 0 — even for a binary you believe is circular, start slightly
    #    off the origin, as here. numpyro reports this as "Cannot find valid initial
    #    parameters", which does not obviously point at the cause.
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

    print("\nfitting (the first evaluation compiles the model; that is the slow part)...")
    start = time.perf_counter()
    fit = ab.run_map(model.model(priors), init=init)
    print(f"  {fit.num_steps} L-BFGS steps in {time.perf_counter() - start:.1f} s")

    k_fit = np.asarray(fit.params["k"])
    k_true = np.asarray(truth["k"], dtype=float)
    print(f"\n  period    {float(fit.params['period']):8.4f} d   (truth {truth['period']:.4f})")
    print(f"  K_1       {k_fit[0]:8.3f} km/s (truth {k_true[0]:.3f})")
    print(f"  K_2       {k_fit[1]:8.3f} km/s (truth {k_true[1]:.3f})")

    # 5. The component spectra, conditional on the fitted orbit, with uncertainties. Read
    #    the band, not the mean: where the epochs give no leverage the smoothness prior
    #    sets the answer, and the band is what says so.
    marginal = model.marginal(fit.params)
    d_hat = np.asarray(marginal.d_hat)
    std = np.asarray(ab.spectra_std(marginal))
    print(f"\n  recovered spectra: {d_hat.shape[0]} components x {d_hat.shape[1]} pixels")
    print(f"  median uncertainty: {np.median(std):.4f} in normalized flux")

    # 6. Save the fit and export the spectra. This is the part that turns a run into
    #    something a co-author can open.
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

    # The point of the example is that this passes. Tolerances are loose because the fit
    # is MAP on 12 epochs of a deliberately small problem.
    assert abs(k_fit[0] - k_true[0]) / k_true[0] < 0.05, "K_1 off by more than 5%"
    assert abs(k_fit[1] - k_true[1]) / k_true[1] < 0.05, "K_2 off by more than 5%"
    assert abs(float(fit.params["period"]) - truth["period"]) < 0.05, "period off by > 0.05 d"
    print("\nok")


if __name__ == "__main__":
    main()
