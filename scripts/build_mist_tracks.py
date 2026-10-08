"""Reduce the MIST evolutionary tracks to the table that `albireo.eclipsing` reads.

The eclipsing-binary population of ``albireo.eclipsing`` gives both stars of a pair one
age, and takes their radii, temperatures and luminosities from evolutionary tracks. This
script reduces the MIST v1.2 tracks of solar composition to the small table kept in the
package, ``src/albireo/data_files/mist_v1.2_solar_tracks.npz``:

- the tracks of 0.10 to 6.0 solar masses (128 of the 196 published);
- from the start of the pre-main-sequence contraction (equivalent evolutionary point 1)
  to the tip of the red giant branch (605): every eighth point up to the zero-age main
  sequence (202), every fourth up to the terminal-age main sequence (454) and every fifth
  beyond it. A track that ends at the terminal-age main sequence, as those below 0.6
  solar masses do, has ``nan`` beyond it;
- the age and the logarithms of the luminosity, the effective temperature and the radius.

Equivalent evolutionary points are the same stage of evolution on every track, so a
quantity is interpolated between two masses at a fixed point (Dotter 2016).

The source is one file of 112 MB, fetched once into the albireo cache
(``albireo.examples.cache_dir() / "tracks" / "_raw"``) and not kept in the repository.

Usage::

    python scripts/build_mist_tracks.py            # download if needed, and reduce
    python scripts/build_mist_tracks.py --check    # compare the packaged table with a rebuild

References
----------
Choi, J., Dotter, A., Conroy, C., et al. 2016, ApJ, 823, 102
Dotter, A. 2016, ApJS, 222, 8
Paxton, B., Bildsten, L., Dotter, A., et al. 2011, ApJS, 192, 3
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

from albireo.examples import _download_with_retries, _sha256, cache_dir

URL = (
    "https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/"
    "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS.txz"
)
SHA256 = "64644c1be477ae72121bd822b5f0eb67538e7ad8e5374109f52bec8733f32fcb"
"""Digest of the file as fetched on 2026-10-07 (112.0 MB, Last-Modified 2025-11-18)."""

MASS_RANGE = (0.10, 6.0)
ZAMS, TAMS, RGB_TIP = 202, 454, 605
EEPS = np.unique(
    np.concatenate(
        [
            np.arange(1, ZAMS + 1, 8),
            np.arange(ZAMS, TAMS + 1, 4),
            np.arange(TAMS, RGB_TIP + 1, 5),
        ]
    )
)
COLUMNS = ("star_age", "log_L", "log_Teff", "log_R")
OUT = Path(__file__).resolve().parent.parent / "src" / "albireo" / "data_files"
NAME = "mist_v1.2_solar_tracks.npz"
CITATION = (
    "MIST v1.2 (MESA revision 7503), [Fe/H] = 0, v/vcrit = 0.4: Choi et al. 2016, ApJ, 823, "
    "102; Dotter 2016, ApJS, 222, 8; Paxton et al. 2011, ApJS, 192, 3; 2013, ApJS, 208, 4; "
    "2015, ApJS, 220, 15"
)


def source_file() -> Path:
    """The MIST archive in the albireo cache, downloaded on first use."""
    raw = cache_dir() / "tracks" / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / URL.rsplit("/", 1)[1]
    if not target.is_file():
        print(f"downloading {URL} (112 MB) to {target}")
        part = target.with_name(target.name + ".part")
        _download_with_retries(URL, part)
        part.replace(target)
    digest = _sha256(target)
    if digest != SHA256:
        raise RuntimeError(
            f"{target} hashes to {digest}, not to the recorded {SHA256}: the upstream file "
            "has changed, and the table built from it would differ from the packaged one"
        )
    return target


def reduce(path: Path) -> dict[str, np.ndarray]:
    """The reduced table: masses, points, and one ``(n_mass, n_point)`` array per quantity."""
    masses, rows = [], []
    with tarfile.open(path, "r:xz") as tar:
        members = sorted(
            (m for m in tar.getmembers() if m.name.endswith(".track.eep")), key=lambda m: m.name
        )
        for member in members:
            mass = int(member.name.rsplit("/", 1)[1][:5]) / 100.0
            if not MASS_RANGE[0] <= mass <= MASS_RANGE[1]:
                continue
            lines = tar.extractfile(member).read().decode().split("\n")
            header = [line for line in lines if line.startswith("#")]
            names = header[-1].lstrip("#").split()
            use = [names.index(column) for column in COLUMNS]
            data = np.loadtxt(lines[len(header) :], usecols=use)
            table = np.full((EEPS.size, len(COLUMNS)), np.nan)
            have = EEPS <= data.shape[0]
            table[have] = data[EEPS[have] - 1]
            masses.append(mass)
            rows.append(table)
            print(f"  {mass:5.2f} Msun: {data.shape[0]} points, {int(have.sum())} kept", flush=True)
    stack = np.stack(rows)
    return {
        "mass": np.asarray(masses, dtype=np.float64),
        "eep": EEPS.astype(np.int32),
        "log_age": np.log10(stack[:, :, 0]).astype(np.float32),
        "log_l": stack[:, :, 1].astype(np.float32),
        "log_teff": stack[:, :, 2].astype(np.float32),
        "log_r": stack[:, :, 3].astype(np.float32),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--check", action="store_true", help="compare with the packaged table")
    args = parser.parse_args(argv)

    table = reduce(source_file())
    target = OUT / NAME
    if args.check:
        with np.load(target) as packaged:
            same = all(
                np.array_equal(packaged[key], value, equal_nan=True) for key, value in table.items()
            )
        print("the packaged table equals a rebuild" if same else "the packaged table DIFFERS")
        return 0 if same else 1
    meta = {
        "source": URL,
        "sha256": SHA256,
        "citation": CITATION,
        "composition": "[Fe/H] = 0.00, [alpha/Fe] = 0.00, Yinit = 0.2703, Zinit = 0.0142857",
        "rotation": "v/vcrit = 0.40",
        "points": {"zams": ZAMS, "tams": TAMS, "rgb_tip": RGB_TIP},
        "units": "mass [Msun]; log_age [log10 yr]; log_l [log10 Lsun]; log_teff [log10 K]; "
        "log_r [log10 Rsun]",
        "built": datetime.date.today().isoformat(),
    }
    np.savez_compressed(target, meta=np.asarray(json.dumps(meta)), **table)
    print(
        f"wrote {target} ({target.stat().st_size / 1e3:.0f} kB): {table['mass'].size} tracks "
        f"of {table['eep'].size} points"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
