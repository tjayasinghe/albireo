"""BLOeM by name: from a survey identifier to a model-ready Dataset.

BLOeM (Binarity at LOw Metallicity) is a VLT/FLAMES-GIRAFFE survey of 929 OBAF stars in the
Small Magellanic Cloud, with about 25 epochs each, an intrinsic binary fraction above 70%,
and 59 published double-lined systems whose disentangling the survey team lists as future
work. It is the largest public dataset albireo was built for, and none of it requires an ESO
account.

    python examples/06_bloem.py            # defaults to BLOeM 1-037, an SB2
    python examples/06_bloem.py 1-002      # any identifier; 'BLOeM_1-2' also works

What this adds to examples 01-05
--------------------------------
The other examples either simulate their data or read a directory that has already been
filled. This one starts from an identifier printed in a paper, and two of the steps in
between are not obvious:

* The archive does not use the survey's names. BLOeM spectra are filed under
  ``obs_collection='GIRAFFE'`` (there is no BLOeM Phase 3 collection) and ``target_name``
  is the Gaia DR3 source id, not ``1-037``. :func:`albireo.resolve_bloem` fetches the
  published cross-match from VizieR, which speaks the same TAP dialect as ESO, so the join
  adds no dependency.
* The file layout differs from the FEROS one of example 03. These are GIRAFFE products: the
  flux is in ``FLUX_REDUCED``, the errors in ``ERR_REDUCED``, the quality flags in
  ``QUAL_REDUCED``, and the wavelengths are in nanometres on an air scale in the
  heliocentric frame. Example 03's FEROS files use ``FLUX``/``ERR``, angstrom, barycentric.
  :func:`albireo.read_dataset` reads both without being told which is which, because it
  dispatches on the IVOA utypes rather than on column names (``internal/design.md`` D45).

Two things this script does not do
----------------------------------
It does not fit an orbit. These systems have no published orbital solutions, so there is no
literature value to score against and no informative prior to start from. The script builds
the problem and evaluates the marginal likelihood once, which is where a loading example
ends.

It does not model the nebula. BLOeM's targets sit in H II regions, so a real analysis of the
Balmer lines needs the nebular component of example 04. The window below avoids the Balmer
cores for that reason.

The window
----------
4120-4300 A, inside LR02's 3960-4571 A: Si III 4128/4130, He I 4144, He I 4169,
He II 4200. It lies strictly between H-delta (4101.7) and H-gamma (4340.5), with neither
line nor its wings inside, and well away from the order edges.

That bound is the reason for the window, and the margin is smaller than it appears. A wider
blue edge would reach H-delta, which in an H II region carries nebular emission that this
script does not model; ``albireo.nebular_windows`` places a +/-300 km/s window around it, so
anything below ~4096 A is contaminated. Widen this region only together with the nebular
component of example 04: an unmodelled static emission line is a component with K = 0, and
D40 measured K_2 59% low under one.

Environment
-----------
ALBIREO_BLOEM_DATA
    Directory for the downloaded FITS files. Default ``data/bloem-<id>``.

Usage
-----
    python examples/06_bloem.py [BLOeM-ID]
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np

import albireo as ab

DEFAULT_TARGET = "1-037"
# Strictly between H-delta and H-gamma; see "The window" above before widening it.
REGION = (4120.0, 4300.0)
# LR02 delivers R = 6300, so the resolution element is c/R = 47.6 km/s FWHM.
LSF_SIGMA = 47.6 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
DV_KMS = 8.0
V_REL_MAX = 400.0
# The SMC recedes at about +150 km/s, and these are SB2s on top of that.
LIGHT_FRACTIONS = (0.6, 0.4)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    target = argv[0] if argv else DEFAULT_TARGET

    # --- 1. Name -> Gaia DR3 source id --------------------------------------------
    t0 = time.time()
    star = ab.resolve_bloem(target)
    print(
        f"[{time.time() - t0:5.1f}s] BLOeM {star.bloem_id} = Gaia DR3 {star.gaia_dr3}"
        f"  {star.spectral_type or '(no type)'}"
        f"  {star.binary_class or 'unclassified in the B-star table'}"
    )
    print(f"           RA {star.ra_deg:.6f}  Dec {star.dec_deg:+.6f}")

    # --- 2. Fetch the epochs ------------------------------------------------------
    # public_only because sub-run .004 releases through 2027-01-15; the proprietary rows
    # would otherwise be returned and then fail to download.
    t0 = time.time()
    records = ab.bloem_spectra(star, public_only=True)
    if not records:
        print(f"no public spectra for BLOeM {star.bloem_id}", file=sys.stderr)
        return 1
    resolving_powers = sorted({r.row["em_res_power"] for r in records})
    print(
        f"[{time.time() - t0:5.1f}s] {len(records)} public epochs, "
        f"R = {', '.join(f'{p:.0f}' for p in resolving_powers)}, "
        f"{sum(r.size_bytes or 0 for r in records) / 1e6:.1f} MB"
    )

    data_dir = pathlib.Path(os.environ.get("ALBIREO_BLOEM_DATA", f"data/bloem-{star.bloem_id}"))
    t0 = time.time()
    statuses = ab.download(records, data_dir)
    failed = [s for s in statuses if s.startswith("FAIL")]
    print(f"[{time.time() - t0:5.1f}s] downloaded into {data_dir}")
    if failed:
        print(f"  {len(failed)} failed:", *failed[:3], sep="\n  ", file=sys.stderr)
        if len(failed) == len(statuses):
            return 1

    # --- 3. FITS -> Dataset -------------------------------------------------------
    # Nothing here names GIRAFFE's column layout, its nanometres or its air scale: the
    # reader takes all three from the file. Compare example 03, which passes the same
    # arguments to FEROS files that share none of those three conventions.
    t0 = time.time()
    dataset = ab.read_dataset(
        str(data_dir / "*.fits"),
        instrument="GIRAFFE",
        region=REGION,
        region_pad_angstrom=40.0,
        smooth_angstrom=60.0,
    )
    # No share_wavelength_grid() here, in contrast with example 03. FEROS shifts before
    # resampling, so its 51 epochs sit on grids that agree to 0.007 km/s and can be
    # relabelled onto one. GIRAFFE's differ by 5.3 km/s, most of a model pixel, so those
    # are distinct wavelength solutions rather than sub-pixel bookkeeping, and
    # share_wavelength_grid refuses them. albireo gives each its own rebin operator.
    print(f"[{time.time() - t0:5.1f}s] ingest")
    print(dataset.summary())

    first = ab.read_spectrum(sorted(data_dir.glob("*.fits"))[0])
    print(
        f"  columns chosen by utype: {first.columns}\n"
        f"  wavelength scale {first.wave_medium!r} (SPECSYS {first.specsys!r}), "
        f"uncertainties from {first.err_source}"
    )
    zeroed = sum(int((epoch.ivar == 0).sum()) for epoch in dataset)
    print(
        f"  {zeroed} of {sum(e.n_pixels for e in dataset)} pixels carry no weight "
        "(quality flags, non-finite flux, non-positive error)"
    )

    # --- 4. Build the problem -----------------------------------------------------
    grid = ab.LogGrid.covering(
        dataset, dv_kms=DV_KMS, v_margin_kms=V_REL_MAX, lsf_sigma_kms=LSF_SIGMA
    )
    model = ab.MarginalOrbitModel(
        grid,
        dataset,
        light_fractions=LIGHT_FRACTIONS,
        lsf_sigma_v={"GIRAFFE": LSF_SIGMA},
        v_rel_max_kms=V_REL_MAX,
    )
    print(
        f"model grid {grid.n} px ({grid.wave[0]:.1f}-{grid.wave[-1]:.1f} A, "
        f"dv={grid.dv_kms:.2f} km/s), {len(model.problem.groups)} operator group(s), "
        f"half-bandwidth {model.half_bandwidth}"
    )

    # One evaluation, confirming that the path is wired end to end. The velocities are a
    # placeholder: this star has no published orbit.
    theta = {
        "velocity": np.zeros((2, len(dataset))),
        "log_tau": np.log(np.array([1.0e4, 1.0e4])),
        "log_eta": np.log(np.array([1.0e2, 1.0e2])),
    }
    t0 = time.time()
    value = float(model.log_likelihood(theta))
    print(f"[{time.time() - t0:5.1f}s] marginal log-likelihood at zero shifts: {value:.6g}")

    print(
        "\nNext: this is where an analysis begins, not ends. With no published orbit the\n"
        "first step is the free per-epoch RV table (docs/math.md 7.6) rather than a\n"
        "Keplerian, and the Balmer lines need the nebular component of example 04 before\n"
        "they can be used."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
