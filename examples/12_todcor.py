"""Epoch radial velocities for both stars by TODCOR, and the orbit from them (``docs/math.md`` §10)

Everything before this example infers the orbit from the spectra directly. This one makes
the artifact the rest of the binary-star toolchain expects instead — one velocity per
component per epoch — by correlating each spectrum against a combination of two templates
with independent shifts (Zucker & Mazeh 1994), then fits a Keplerian to the table, then
closes the loop by using the disentangled components themselves as the templates.

Four things to watch in the output.

1. **Both velocities from every single spectrum**, with errors that mean what they say. The
   templates here are the injected component spectra — the best case — and the pull
   ``(v - v_true) / sigma`` comes out near unit variance.
2. **Why two dimensions.** The same spectra correlated against the primary alone — the
   one-dimensional CCF — carry a bias that grows as the two stars' lines approach each
   other; the two-dimensional fit does not, because the second template is in the model.
3. **The orbit from the table**, with the period found by a periodogram, against the
   injected elements — and ``to_theta()`` handing it back to the disentangler as a warm
   start.
4. **The loop closed**: a quick MAP disentangling of the same epochs, its two components
   turned into templates, and the epochs measured against *them*. Those velocities are
   differential — a disentangled component's rest frame is not identified — so the orbit is
   fitted with one systemic velocity per component, and the semi-amplitudes still come back.

Then a batch of two stars, because a survey does not call this once.

Environment
-----------
``ALBIREO_EXAMPLE_FAST=1`` trims the disentangling's optimizer budget for CI.
"""

from __future__ import annotations

import importlib.util
import os
import time

import numpy as np

import albireo as ab

FAST = bool(os.environ.get("ALBIREO_EXAMPLE_FAST"))
LSF = {"DEMO": 6.5}


def perfect_templates(grid, truth, factor: int = 3):
    """The injected component spectra as templates, upsampled for the correlation.

    The packaged truth grid samples the DEMO instrument's LSF at about one pixel per sigma,
    which is fine for the solver and coarse for a correlation template: the linear shift
    operator's pixel-locking ripple is of order ``0.1 / sigma_px^2`` pixels
    (``docs/math.md`` §10.3). Linear upsampling by an integer factor costs nothing and puts
    three pixels under the sigma — the same thing ``Fit.templates()`` does.
    """
    fine = ab.LogGrid(x0=grid.x0, dx=grid.dx / factor, n=(grid.n - 1) * factor + 1)
    return [
        ab.Template(name, fine, np.interp(fine.x, grid.x, deviation), v_zero_kms=0.0)
        for name, deviation in zip(("primary", "secondary"), truth["components"], strict=True)
    ]


def main() -> None:
    t_start = time.perf_counter()
    dataset, truth = ab.load_example("sb2_sim", with_truth=True)
    grid = ab.LogGrid(x0=truth["grid_x0"], dx=truth["grid_dx"], n=int(truth["grid_n"]))
    light = np.asarray(truth["light_fractions"], dtype=float)
    v_true = np.asarray(truth["velocities"], dtype=float)
    print(dataset.summary())

    # 1. Both velocities from every spectrum, against the true component spectra ---------
    templates = perfect_templates(grid, truth)
    table = ab.todcor(dataset, templates, v_range=(-120.0, 120.0), light=light, lsf_sigma_v=LSF)
    error = table.velocity - v_true
    pull = error / table.sigma
    print("\n1. TODCOR against the injected component spectra")
    print(table.summary())
    for i, name in enumerate(table.names):
        print(
            f"   {name}: rms error {np.sqrt(np.mean(error[i] ** 2)):.4f} km/s, "
            f"mean quoted sigma {table.sigma[i].mean():.4f} km/s, "
            f"pull rms {np.sqrt(np.mean(pull[i] ** 2)):.2f}"
        )

    # 2. Why two dimensions: the one-dimensional CCF of the same spectra ------------------
    one_d = ab.todcor(dataset, templates[:1], v_range=(-120.0, 120.0), light=[1.0], lsf_sigma_v=LSF)
    separation = np.abs(v_true[0] - v_true[1])
    order = np.argsort(separation)
    print("\n2. The primary's velocity error, one template against two (epochs by separation)")
    print("   |v1 - v2| [km/s]   1-D error   2-D error   (km/s)")
    for j in order:
        print(
            f"   {separation[j]:8.1f}          {one_d.velocity[0, j] - v_true[0, j]:+8.3f}   "
            f"{table.velocity[0, j] - v_true[0, j]:+8.3f}"
        )
    close = order[:4]
    bias_1d = np.sqrt(np.mean((one_d.velocity[0, close] - v_true[0, close]) ** 2))
    bias_2d = np.sqrt(np.mean((table.velocity[0, close] - v_true[0, close]) ** 2))
    print(f"   rms over the four most blended epochs: 1-D {bias_1d:.3f}, 2-D {bias_2d:.3f} km/s")

    # 3. The orbit from the table ---------------------------------------------------------
    search = ab.find_period(table, period_range=(2.0, 20.0))
    orbit = ab.fit_rv_orbit(table, period=search["period"])
    print(
        f"\n3. Period search: {search['period']:.4f} d (truth {truth['period']:.4f}); aliases "
        f"{[round(p, 3) for p in search['aliases'][:3]]}"
    )
    print(orbit.summary())
    k_true = np.asarray(truth["k"], dtype=float)
    theta = orbit.to_theta()
    print(
        f"   to_theta(): period {float(theta['period']):.4f}, k {np.asarray(theta['k']).round(3)} "
        "- the warm start Disentangler(orbit=...) would take"
    )

    # 4. The loop closed: disentangle, then measure against the components ----------------
    dis = ab.Disentangler(
        dataset,
        components=[
            ab.Star("primary", light=float(light[0])),
            ab.Star("secondary", light=float(light[1])),
        ],
        orbit=ab.Orbit(period=ab.Between(5.5, 6.5), k=ab.Between([10.0, 10.0], [90.0, 90.0])),
        lsf=LSF,
    )
    t0 = time.perf_counter()
    fit = dis.fit(max_steps=60 if FAST else 150)
    k_fit = np.asarray(fit.params["k"]).round(3)
    print(f"\n4. Disentangled in {time.perf_counter() - t0:.1f} s; K = {k_fit}")
    own = fit.measure_velocities()
    print(own.summary())
    own_orbit = ab.fit_rv_orbit(own, period=orbit.period)
    print(own_orbit.summary())
    zero_points = [float(np.mean(own.velocity[i] - v_true[i])) for i in range(2)]
    centred = own.velocity - np.asarray(zero_points)[:, None]
    print(
        f"   per-component zero points absorbed: {np.round(zero_points, 3)} km/s; "
        f"rms after removing them {np.sqrt(np.mean((centred - v_true) ** 2, axis=1)).round(4)} km/s"
    )

    # 5. A batch: two stars, one call, one table each -------------------------------------
    rng = np.random.default_rng(1)
    twin_flux = [
        ab.EpochData(
            wave=e.wave,
            flux=e.flux + rng.normal(0.0, 0.02, e.flux.shape),
            ivar=1.0 / (1.0 / e.ivar + 0.02**2),
            bjd=e.bjd,
            v_bary=e.v_bary,
            instrument=e.instrument,
        )
        for e in dataset
    ]
    twin = ab.Dataset(twin_flux, frame=dataset.frame)
    batch = ab.todcor_batch(
        {"sb2_sim": dataset, "sb2_sim_noisier": twin},
        templates,
        progress=False,
        v_range=(-120.0, 120.0),
        light="global",
        lsf_sigma_v=LSF,
    )
    print("\n5. Batch")
    print(batch.summary())
    written = batch.write("todcor_tables")
    print(f"   wrote {[p.name for p in written]} to todcor_tables/")

    # 6. Figures, only if matplotlib happens to be installed ------------------------------
    if importlib.util.find_spec("matplotlib") is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, _ = ab.plot_velocity_table(table, orbit=orbit, truth=v_true)
        fig.savefig("todcor_orbit.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        surface = ab.todcor_surface(
            dataset,
            int(order[0]),
            templates,
            v_range=(-120.0, 120.0),
            light=light,
            lsf_sigma_v=LSF,
            step=2,
        )
        fig, _ = ab.plot_todcor_surface(surface, truth=v_true[:, order[0]])
        fig.savefig("todcor_surface.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("\nwrote todcor_orbit.png, todcor_surface.png")
    else:
        print("\nmatplotlib not installed - skipping the figures (it is not a dependency)")

    # 7. The gate ---------------------------------------------------------------------------
    assert np.sqrt(np.mean(error**2)) < 0.2, f"perfect-template rms {np.sqrt(np.mean(error**2))}"
    assert 0.5 < np.sqrt(np.mean(pull**2)) < 2.0, f"pull rms {np.sqrt(np.mean(pull**2))}"
    assert table.good.all(), "every epoch of the perfect-template run should be usable"
    assert bias_1d > 2.0 * bias_2d, (
        f"the 1-D CCF should be visibly worse when blended ({bias_1d} vs {bias_2d})"
    )
    assert abs(search["period"] / truth["period"] - 1.0) < 0.01, "the period search missed"
    assert np.all(np.abs(orbit.k - k_true) / k_true < 0.01), f"K {orbit.k} vs {k_true}"
    assert np.all(np.abs(own_orbit.k - k_true) / k_true < 0.03), f"self-consistent K {own_orbit.k}"
    assert own.absolute == (False, False) and own_orbit.gamma_mode == "one per component"
    assert set(batch.tables) == {"sb2_sim", "sb2_sim_noisier"} and not batch.failures
    print(
        f"\nOK - both velocities from every spectrum with calibrated errors, the 1-D bias "
        f"removed, K to <1% from the table and <3% through the disentangling loop. "
        f"Total wall: {time.perf_counter() - t_start:.1f} s"
    )


if __name__ == "__main__":
    main()
