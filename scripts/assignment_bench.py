"""The assignment of the velocities of two alike stars to the stars, at a known period.

    python scripts/assignment_bench.py [--tables 60]

Two alike spectra at alike light fractions give a correlation surface that is nearly
symmetric under the exchange of the two shifts, and each epoch returns the pair of
velocities in one of the two orders. This script measures how often the orbit and the
assignment are recovered from such a table when the period is known.

The tables are built directly, without spectra. The orbit has semi-amplitudes of 61 and
64 km/s and a period of 6.31 d, with the argument and the time of periastron drawn per
table. It is sampled at random times over 40 d with Gaussian errors of 0.5 km/s, and the two
velocities of every epoch are put in a random order. A table counts as recovered where both
semi-amplitudes are within 2 percent of the injected ones, in either order of the two
stars. Three procedures are compared at every eccentricity and number of epochs:

- one exchange: a Keplerian fit, one call of ``reassign_by_orbit`` and a second fit, as the
  pipeline's candidate fits made it before D68;
- the circular scan: ``assign_components`` with one starting assignment beside the table
  as measured, taken from the circular curves alone, which is the procedure of
  ``docs/tutorials/gaia-rvs-todcor.ipynb`` before it was moved into ``albireo.rvorbit``;
- ``albireo.rvorbit.assign_components`` as it is.

A second table gives the period to ``assign_components`` with an error, stated in units of
``1 / T`` in frequency, ``T`` being the time span of the epochs, and compares the function
at the period given with the function under ``period_window=1``, which also makes the
assignments at the periods within ``1 / T`` of the one given.

The results are recorded in ``docs/benchmarks.md``.
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

import albireo as ab
from albireo import rvorbit
from albireo.todcor import VelocityTable

PERIOD = 6.31
K = (61.0, 64.0)
GAMMA = 12.0
SIGMA = 0.5
SPAN = 40.0
ECCENTRICITIES = (0.0, 0.3, 0.6, 0.8)
EPOCHS = (10, 12, 20, 40)
OFFSETS = (0.1, 0.25, 0.5, 1.0, 1.5)  # of the period given, in units of 1/T in frequency


def twin_table(seed: int, n_epochs: int, ecc: float) -> VelocityTable:
    """A table of the orbit at ``n_epochs`` random times, each epoch in a random order."""
    rng = np.random.default_rng(seed)
    bjd = np.sort(rng.uniform(0.0, SPAN, size=n_epochs))
    orbit = ab.OrbitParams(
        period=PERIOD,
        t_peri=rng.uniform(0.0, PERIOD),
        ecc=ecc,
        omega=rng.uniform(0.0, 2.0 * np.pi),
        k=K,
        gamma=GAMMA,
    )
    velocity = orbit.component_velocities(bjd)
    velocity = velocity + rng.normal(0.0, SIGMA, velocity.shape)
    exchanged = rng.uniform(size=n_epochs) < 0.5
    velocity[:, exchanged] = velocity[::-1, exchanged]
    sigma = np.full(velocity.shape, SIGMA)
    return VelocityTable(
        names=("A", "B"),
        bjd=bjd,
        instrument=("a",) * n_epochs,
        velocity=velocity,
        sigma=sigma,
        sigma_ivar=sigma,
        covariance=np.repeat(np.diag([SIGMA**2, SIGMA**2])[None], n_epochs, axis=0),
        light=np.full(velocity.shape, 0.5),
        light_mode="fixed",
        chi2=np.full(n_epochs, 1000.0),
        chi2_null=np.full(n_epochs, 1e5),
        n_pixels=np.full(n_epochs, 1000),
        delta_chi2=np.full(velocity.shape, 1e4),
        blended=np.zeros(n_epochs, dtype=bool),
        at_edge=np.zeros(velocity.shape, dtype=bool),
        refined=np.ones(n_epochs, dtype=bool),
        absolute=(True, True),
        frame="barycentric",
    )


def recovered(orbit) -> bool:
    """Both semi-amplitudes within 2 percent of the injected ones, in either order."""
    k = np.asarray(orbit.k)
    errors = [np.max(np.abs(order / np.array(K) - 1.0)) for order in (k, k[::-1])]
    return bool(min(errors) < 0.02)


def one_exchange(table):
    fitted = rvorbit.fit_rv_orbit(table, period=PERIOD)
    exchanged, moved = rvorbit.reassign_by_orbit(table, fitted.predict(table.bjd))
    return rvorbit.fit_rv_orbit(exchanged, period=PERIOD) if moved.any() else fitted


def circular_scan(table):
    """``assign_components`` with one further start, from the circular curves alone."""
    grid = (rvorbit._ASSIGNMENT_ECCENTRICITIES, rvorbit._ASSIGNMENT_STARTS)
    rvorbit._ASSIGNMENT_ECCENTRICITIES, rvorbit._ASSIGNMENT_STARTS = (), 1
    try:
        return rvorbit.assign_components(table, period=PERIOD)[1]
    finally:
        rvorbit._ASSIGNMENT_ECCENTRICITIES, rvorbit._ASSIGNMENT_STARTS = grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=int, default=60, help="tables per configuration")
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore")  # the fits of wrongly assigned tables hit their bounds

    print(f"tables recovered, of {args.tables}:")
    print("| eccentricity | epochs | one exchange | circular scan | assign_components |")
    print("|---|---|---|---|---|")
    for ecc in ECCENTRICITIES:
        for n_epochs in EPOCHS:
            counts = np.zeros(3, dtype=int)
            for seed in range(args.tables):
                table = twin_table(seed, n_epochs, ecc)
                orbits = (
                    one_exchange(table),
                    circular_scan(table),
                    rvorbit.assign_components(table, period=PERIOD)[1],
                )
                counts += np.array([recovered(orbit) for orbit in orbits])
            print(f"| {ecc:.1f} | {n_epochs} | {counts[0]} | {counts[1]} | {counts[2]} |")

    print(f"\ntwelve epochs, the period given off by x / T; tables recovered, of {args.tables}:")
    print("| eccentricity | x | at the period given | with a window of 1 |")
    print("|---|---|---|---|")
    for ecc in ECCENTRICITIES[:3]:
        for offset in OFFSETS:
            counts = np.zeros(2, dtype=int)
            for seed in range(args.tables):
                table = twin_table(seed, 12, ecc)
                given = 1.0 / (1.0 / PERIOD + offset / float(np.ptp(table.bjd)))
                orbits = (
                    rvorbit.assign_components(table, period=given)[1],
                    rvorbit.assign_components(table, period=given, period_window=1.0)[1],
                )
                counts += np.array([recovered(orbit) for orbit in orbits])
            print(f"| {ecc:.1f} | {offset:g} | {counts[0]} | {counts[1]} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
