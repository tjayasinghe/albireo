"""Regenerate the example datasets that :mod:`albireo.examples` serves.

Run this when the simulator changes in a way that alters what it produces, or to rebuild
the packaged file from scratch:

    python scripts/build_example_datasets.py

Everything is seeded, so a rebuild on any machine produces a byte-identical file. The
script prints the SHA-256 of what it wrote; for downloaded (non-packaged) examples that
value goes into the ``_EXAMPLES`` registry in ``src/albireo/examples.py``, so that a
corrupted download is detected.

The packaged example is small: it ships inside the wheel, so it has to stay well under the
500 kB pre-commit file-size limit and has to be quick to fit. It provides a first offline
run rather than a realistic science case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from albireo.grids import LogGrid  # noqa: E402
from albireo.simulate import (  # noqa: E402
    InstrumentSpec,
    OrbitParams,
    simulate_dataset,
    synthetic_deviation_spectrum,
)

PACKAGED_DIR = REPO_ROOT / "src" / "albireo" / "data_files"

# A short window at modest resolution: enough pixels to carry a dozen lines and show the
# disentangling working, few enough that a MAP fit takes seconds rather than minutes.
GRID = LogGrid.from_wavelength_range(4500.0, 4560.0, dv_kms=6.0)
PERIOD = 6.0
T_PERI = 0.0
K1, K2 = 42.0, 63.0
LIGHT = (0.62, 0.38)
N_EPOCHS = 12
# Modestly eccentric rather than circular. The orbit is parameterized by
# (sqrt(e) cos w, sqrt(e) sin w), which is singular at e = 0: the argument of periastron is
# undefined there, e*cos(w) behaves like |x|, and the gradient is NaN at exactly the origin.
# A circular example would put both the truth and the natural starting guess on that point,
# so the first example run would fail to initialize.
ECC = 0.15
OMEGA = 0.7


def build_sb2_sim() -> dict:
    """A 12-epoch eccentric SB2 on one instrument, with its injected truth."""
    components = [
        synthetic_deviation_spectrum(GRID, n_lines=14, seed=seed, margin=0.08) for seed in (11, 12)
    ]
    orbit = OrbitParams(period=PERIOD, t_peri=T_PERI, ecc=ECC, omega=OMEGA, k=(K1, K2))
    # Epochs spread over an interval that is not a whole number of periods, so the phase
    # coverage is even rather than clumped at a few phases.
    bjd = np.linspace(0.0, 2.7 * PERIOD, N_EPOCHS, endpoint=False)

    dataset, truth = simulate_dataset(
        GRID,
        components,
        bjd=bjd,
        instruments={
            "DEMO": InstrumentSpec(wave=np.arange(4505.0, 4555.0, 0.06), sigma_v_lsf=6.5, snr=80.0)
        },
        light_fractions=LIGHT,
        orbit=orbit,
        seed=2026,
    )
    return {
        "dataset": dataset,
        "truth": {
            "components": np.asarray(truth.components),
            "light_fractions": np.asarray(LIGHT, dtype=float),
            "velocities": np.asarray(orbit.component_velocities(bjd)),
            "scalars": {
                "period": PERIOD,
                "t_peri": T_PERI,
                "ecc": ECC,
                "omega": OMEGA,
                "k": [K1, K2],
                "grid_x0": GRID.x0,
                "grid_dx": GRID.dx,
                "grid_n": GRID.n,
            },
        },
        "description": (
            f"Simulated SB2: {N_EPOCHS} epochs, P = {PERIOD} d, e = {ECC}, "
            f"K = ({K1}, {K2}) km/s, light fractions {LIGHT}, S/N 80."
        ),
    }


def write_npz(path: pathlib.Path, built: dict) -> pathlib.Path:
    dataset = built["dataset"]
    epochs = list(dataset)
    arrays: dict[str, np.ndarray] = {}
    meta = []
    for i, epoch in enumerate(epochs):
        arrays[f"wave/{i}"] = np.asarray(epoch.wave, dtype=np.float64)
        arrays[f"flux/{i}"] = np.asarray(epoch.flux, dtype=np.float64)
        arrays[f"ivar/{i}"] = np.asarray(epoch.ivar, dtype=np.float64)
        meta.append(
            {
                "bjd": float(epoch.bjd),
                "v_bary": float(epoch.v_bary),
                "instrument": str(epoch.instrument),
            }
        )

    header = {
        "n_epochs": len(epochs),
        "frame": dataset.frame,
        "epochs": meta,
        "description": built["description"],
        "has_truth": "truth" in built,
    }
    if "truth" in built:
        truth = built["truth"]
        arrays["truth/components"] = truth["components"]
        arrays["truth/light_fractions"] = truth["light_fractions"]
        arrays["truth/velocities"] = truth["velocities"]
        header["truth_scalars"] = truth["scalars"]

    arrays["__albireo_example__"] = np.array(json.dumps(header, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=PACKAGED_DIR,
        help="where to write (default: the packaged data directory)",
    )
    args = parser.parse_args(argv)

    built = build_sb2_sim()
    path = write_npz(args.outdir / "sb2_sim.npz", built)
    size = path.stat().st_size
    print(f"wrote {path}")
    print(f"  {built['description']}")
    print(f"  {size / 1024:.1f} kB   sha256 {sha256(path)}")
    if size > 500 * 1024:
        print(
            "  WARNING: over the 500 kB pre-commit limit for a packaged file — "
            "shorten the grid or drop epochs.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
