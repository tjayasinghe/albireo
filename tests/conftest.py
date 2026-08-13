"""Shared pytest configuration.

Two jobs. First, the opt-in gates: tests marked ``slow`` (real sampling, multi-minute
closed-loop fits), ``network`` (public archive queries and example-data downloads), and
``gpu`` run by default so that a bare ``pytest`` still runs the acceptance gates — the
suite is the oracle and skipping it silently is worse than waiting. CI deselects them
explicitly with ``-m "not slow"`` on the fast matrix and runs them in a separate job.
``--no-slow`` / ``--no-network`` are the local shorthands for the same thing.

Second, a small number of shared fixtures for *new* test modules. The existing modules
build their own simulated datasets and are deliberately left alone: they encode specific
scales (bandwidths, epoch counts, SNRs) that the closed-loop tolerances are tuned to, and
rewriting them onto shared fixtures would put those tolerances at risk for no gain.
"""

from __future__ import annotations

import numpy as np
import pytest

_OPT_OUT = {
    "slow": ("--no-slow", "deselected by --no-slow"),
    "network": ("--no-network", "deselected by --no-network"),
    "gpu": ("--no-gpu", "deselected by --no-gpu"),
}


def pytest_addoption(parser):
    parser.addoption("--no-slow", action="store_true", help="skip tests marked slow")
    parser.addoption("--no-network", action="store_true", help="skip tests needing the internet")
    parser.addoption("--no-gpu", action="store_true", help="skip tests needing a GPU backend")


def pytest_collection_modifyitems(config, items):
    skips = {
        marker: pytest.mark.skip(reason=reason)
        for marker, (flag, reason) in _OPT_OUT.items()
        if config.getoption(flag)
    }
    if not skips:
        return
    for item in items:
        for marker, skip in skips.items():
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def rng():
    """A seeded NumPy generator, so a test that uses it is reproducible by construction."""
    return np.random.default_rng(20260813)


@pytest.fixture(scope="session")
def small_grid():
    """A small log-wavelength grid — big enough to carry lines, small enough to be instant."""
    from albireo.grids import LogGrid

    return LogGrid.from_wavelength_range(4500.0, 4512.0, dv_kms=6.0)


@pytest.fixture(scope="session")
def small_simulation(small_grid):
    """A four-epoch SB2 with known injected truth: ``(dataset, truth)``.

    Session-scoped and therefore **read-only** — a test that needs to mutate epochs should
    build its own. The scale matches the existing fast closed-loop tests (a few hundred
    pixels, circular orbit, one instrument) so that it stays instant.
    """
    from albireo.simulate import InstrumentSpec, OrbitParams, simulate_dataset
    from albireo.simulate import synthetic_deviation_spectrum as synth

    components = [synth(small_grid, n_lines=6, seed=seed, margin=0.1) for seed in (1, 2)]
    orbit = OrbitParams(period=6.0, t_peri=0.0, ecc=0.0, omega=0.0, k=(40.0, 60.0))
    return simulate_dataset(
        small_grid,
        components,
        bjd=np.array([0.7, 2.2, 4.9, 6.0]),
        instruments={
            "A": InstrumentSpec(wave=np.arange(4502.0, 4510.0, 0.08), sigma_v_lsf=8.25, snr=60.0)
        },
        light_fractions=[0.6, 0.4],
        orbit=orbit,
        seed=4,
    )


@pytest.fixture(scope="session")
def small_dataset(small_simulation):
    """The :class:`~albireo.data.Dataset` half of :func:`small_simulation`."""
    dataset, _ = small_simulation
    return dataset
