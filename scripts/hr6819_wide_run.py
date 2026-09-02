"""HR 6819, one wide window: 4120-4600 A with the Hgamma core masked (D36).

The last recorded lever on this dataset. D30 fitted two separated windows
(A 4380-4600, B 4120-4330) bracketing Hgamma, whose core carries the Be disc's
variable emission, a time-variable feature that violates the
one-static-spectrum-per-component assumption. D34 measured the windows' AR(1)
orbits 0.0016 d apart in period. This run joins the two regions into one fit,
2.3x window A's pixels with every line constraining one orbit, and masks the
Balmer core (``preprocess.mask_ranges``: ivar = 0, pixels stay, so the AR(1) chain
restarts across the hole and the solver bandwidth is unchanged). The broad Hgamma
absorption wings are static stellar features and stay in.

Noise model and procedure are the D34 configuration exactly (per-epoch jitters and
shared ar1_phi, conjunction scan, literature init), so the A/B AR(1) records are
the comparison baselines. The run uses the D35 band assembly, and is affordable
where the probe path was not: that path's gradient alone peaked at 23.7 GiB at half
this size.

Run:  python scripts/hr6819_wide_run.py [--max-steps 200]
      (expects the FEROS FITS under data/hr6819 or $ALBIREO_HR6819_DATA)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
from hr6819_ar1_run import run_ar1
from hr6819_response_run import (
    DV_KMS,
    LIGHT_FRACTIONS,
    LSF_SIGMA,
    V_REL_MAX,
    conjunction_scan,
)

import albireo as ab
from albireo.preprocess import mask_ranges, share_wavelength_grid

WINDOW = (4120.0, 4600.0)
# Hgamma 4340.47: disc emission and the deep, RV-blended core. +-15 A ~ +-1000 km/s
# covers the emission and core; the far wings stay, and they are static.
HGAMMA_MASK = (4325.0, 4355.0)


def load_wide():
    data_dir = pathlib.Path(
        os.environ.get("ALBIREO_HR6819_DATA")
        or pathlib.Path(__file__).resolve().parent.parent / "data" / "hr6819"
    )
    dataset = ab.read_dataset(
        str(data_dir / "*.fits"),
        instrument="FEROS",
        region=WINDOW,
        region_pad_angstrom=60.0,
        smooth_angstrom=120.0,
    )
    epochs = [mask_ranges(ep, [HGAMMA_MASK]) for ep in dataset]
    dataset = ab.Dataset(share_wavelength_grid(epochs), frame=dataset.frame)
    grid = ab.LogGrid.covering(
        dataset, dv_kms=DV_KMS, v_margin_kms=V_REL_MAX, lsf_sigma_kms=LSF_SIGMA
    )
    model = ab.MarginalOrbitModel(
        grid,
        dataset,
        light_fractions=LIGHT_FRACTIONS,
        lsf_sigma_v={"FEROS": LSF_SIGMA},
        v_rel_max_kms=V_REL_MAX,
        ar1=True,
    )
    return dataset, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    print(f"\n=== wide window: {WINDOW[0]:.0f}-{WINDOW[1]:.0f} A, Hgamma masked (ar1) ===")
    t0 = time.time()
    dataset, model = load_wide()
    masked = sum(
        int(np.sum((ep.wave >= HGAMMA_MASK[0]) & (ep.wave <= HGAMMA_MASK[1]))) for ep in dataset
    )
    print(
        f"  ingest {time.time() - t0:.1f}s: {dataset.n_epochs} epochs, "
        f"{model.problem.grid.n} model px, half-bandwidth {model.half_bandwidth} "
        f"(ar extra {model.problem.ar_bandwidth_extra}); "
        f"{masked} native px masked for Hgamma",
        flush=True,
    )
    t0 = time.time()
    t_conj0 = conjunction_scan(dataset, model)
    print(f"  conjunction scan {time.time() - t0:.1f}s: t_conj0 = {t_conj0:.4f}", flush=True)
    r = run_ar1(dataset, model, t_conj0, max_steps=args.max_steps)

    print(
        "\n\n=== summary (D34 AR(1) baselines: A P 40.37115 K 63.242/2.446 e 0.0273 "
        "phi +0.801; B P 40.36956 K 63.518/3.756 e 0.0228 phi +0.694; "
        "lit P 40.3261 K 61.15/3.90 e 0.0289) ===",
        flush=True,
    )
    print(
        f"  wide/ar1: P {r['period']:.5f}  K {r['k_presd']:.3f}/{r['k_be']:.3f}  "
        f"e {r['ecc']:.4f}  phi {r['phi']:+.3f}  "
        f"alpha {r['alpha_min']:.2f}-{r['alpha_max']:.2f} (med {r['alpha_med']:.2f})  "
        f"chain sd/lag1 {r['resid_sd']:.3f}/{r['resid_lag1']:+.3f}  "
        f"diag lag1 {r['diag_lag1']:+.3f}  loglike {r['loglike']:.1f}  "
        f"({r['seconds'] / max(r['steps'], 1):.1f} s/step)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
