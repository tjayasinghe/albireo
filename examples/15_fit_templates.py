"""Templates fitted to the epochs with the velocities (``docs/math.md`` §10.7).

``examples/12_todcor.py`` measures velocities against templates that are given. A survey
gives a classification instead: a temperature, a gravity and a rotation for each star, each
with an error. This example measures a simulated double-lined binary with templates
rendered at such a classification, then fits the templates to the same epochs
(:func:`albireo.fit_templates`) and measures again.

Three results should be checked in the output.

1. The fit recovers the labels, the rotation and the light fractions of both stars from
   the epochs, with errors, starting from a classification that is 150 K, 0.2 dex and 30
   percent in error. The prior fractions it prints are small: the epochs, not the
   classification, set the labels.
2. The velocities measured with the fitted templates are closer to the injected ones than
   those measured with the classified templates, and their errors describe them.
3. The orbit fitted to the second table has semi-amplitudes consistent with the injected
   ones, with errors about half those of the first table's, whose scatter about its orbit
   includes the mismatch of its templates.

The library is the toy grid of :func:`albireo.simulate.synthetic_library`, so the example
needs no download. A published grid (:func:`albireo.fetch_library`) is used the same way.

References
----------
Nachmani, G., Faigler, S. & Mazeh, T. 2026, arXiv:2602.05610
Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806
"""

from __future__ import annotations

import numpy as np

import albireo as ab
from albireo.simulate import library_component, synthetic_library

LSF_KMS = 6.0
PERIOD = 5.3
INJECTED = {
    "A": {"teff": 5500.0, "logg": 4.2, "mh": -0.1, "vsini": 28.0},
    "B": {"teff": 4900.0, "logg": 4.5, "mh": -0.1, "vsini": 14.0},
}
LIGHT = (0.62, 0.38)
K = (70.0, 95.0)
V_RANGE = (-200.0, 200.0)


def simulate(library, n_epochs=16, snr=80.0, seed=4):
    """A binary rendered from the library at the injected labels, on a circular orbit."""
    grid = ab.LogGrid.from_wavelength_range(5140.0, 5260.0, dv_kms=1.0)
    rng = np.random.default_rng(seed)
    bjd = 2459000.0 + np.sort(rng.uniform(0.0, 40.0, n_epochs))
    curve = np.sin(2.0 * np.pi * (bjd - bjd[0]) / PERIOD)
    velocity = np.stack([12.0 + K[0] * curve, 12.0 - K[1] * curve])
    components = [
        library_component(
            library,
            {axis: labels[axis] for axis in ("teff", "logg", "mh")},
            grid,
            medium="vacuum",
            vsini_kms=labels["vsini"],
        )
        for labels in INJECTED.values()
    ]
    spectrograph = ab.InstrumentSpec(
        wave=np.arange(5156.0, 5244.0, 0.07), sigma_v_lsf=LSF_KMS, snr=snr
    )
    dataset, _ = ab.simulate_dataset(
        grid,
        components,
        bjd=bjd,
        instruments={"spec": spectrograph},
        light_fractions=LIGHT,
        velocities=velocity,
        frame="barycentric",
        seed=seed,
    )
    return dataset, velocity


def rms(table, velocity):
    use = table.good
    return np.sqrt(np.mean((table.velocity[:, use] - velocity[:, use]) ** 2, axis=1))


def main() -> None:
    library = synthetic_library(
        (5100.0, 5300.0),
        n_pix=4000,
        n_lines=40,
        teff=tuple(float(t) for t in range(4500, 6751, 250)),
        logg=(3.5, 4.0, 4.5, 5.0),
        mh=(-0.5, -0.25, 0.0, 0.25),
        medium="vacuum",
    )
    dataset, velocity = simulate(library)
    grid = ab.LogGrid.covering(dataset, 2.0, v_margin_kms=260.0, lsf_sigma_kms=LSF_KMS)

    # The classification: every label in error, with the error it quotes as the prior.
    stars = {}
    for sign, (name, truth) in zip((1.0, -1.0), INJECTED.items(), strict=True):
        stars[name] = ab.StarLabels(
            library=library,
            teff=ab.Known(truth["teff"] + sign * 150.0, 200.0),
            logg=ab.Known(truth["logg"] + sign * 0.2, 0.3),
            vsini=ab.Known(1.3 * truth["vsini"], 0.4 * truth["vsini"]),
        )
    fit = ab.fit_templates(
        dataset,
        stars,
        grid=grid,
        medium="vacuum",
        mh=ab.Known(0.05, 0.2),
        v_range=V_RANGE,
        lsf_sigma_v={"spec": LSF_KMS},
    )
    print(fit.summary())
    print("\ninjected:")
    for name, truth in INJECTED.items():
        print(f"  {name}: " + ", ".join(f"{label} {value:g}" for label, value in truth.items()))
    print(f"  light fractions: {LIGHT[0]:.2f}, {LIGHT[1]:.2f}")
    print(
        "\nposterior over prior variance: "
        + ", ".join(f"{key} {ratio:.3f}" for key, ratio in fit.prior_fraction.items())
    )

    print("\nrms error of the velocities [km/s], primary and secondary")
    for label, table in (("classified templates", fit.start), ("fitted templates", fit.table)):
        error = rms(table, velocity)
        print(f"  {label:21s} {error[0]:6.3f} {error[1]:6.3f}   ({int(table.good.sum())} epochs)")
    use = fit.table.good
    pull = (fit.table.velocity[:, use] - velocity[:, use]) / fit.table.sigma[:, use]
    print(f"  pull rms with the fitted templates: {np.sqrt(np.mean(pull**2)):.2f}")
    share = np.nanmedian(fit.table.template_sigma[:, use] / fit.table.sigma[:, use])
    print(f"  error from the templates over the quoted error, median: {share:.2f}")

    print(f"\nsemi-amplitudes [km/s], injected {K[0]:g} and {K[1]:g}")
    for label, table in (("classified templates", fit.start), ("fitted templates", fit.table)):
        orbit = ab.fit_rv_orbit(table, period=PERIOD, circular=True)
        print(
            f"  {label:21s} {orbit.k[0]:7.3f} +- {orbit.errors['k'][0]:.3f}   "
            f"{orbit.k[1]:7.3f} +- {orbit.errors['k'][1]:.3f}"
        )


if __name__ == "__main__":
    main()
