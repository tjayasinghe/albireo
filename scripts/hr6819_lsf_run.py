"""HR 6819, wide window, wavelength-dependent LSF fitted jointly (D37).

The 0.041 d literature period offset survived four configurations (D30-D36); the
recorded suspects are the Gaussian LSF stand-in, disc variability, and the published
CCF blending. This run opens D8's tabulated-LSF seam on the first of them. FEROS's
resolution is not constant across the merged echelle spectrum, varying along and
between orders, so the LSF width becomes a per-anchor θ-site (one Gaussian width
every ~40 A, the order scale, linearly interpolated across the grid) fitted jointly
with the D36 configuration (wide window, Hgamma masked, per-epoch jitters and shared
ar1_phi).

What is and is not identified: a stationary width change commutes with the component
shifts on the log grid, so the free spectra absorb it, and on this data with ML-II
hyperparameters the absolute width level is close to degenerate by construction. The
identified content is the anchor-to-anchor variation, which breaks the commutation
through the epoch-dependent shifts, and the orbit's response. If the fitted width
profile moves P toward the literature value, the LSF-width hypothesis stands; if the
orbit does not move, a wavelength-dependent width joins the continuum (D33) and the
noise model (D34) on the exonerated list, and the surviving LSF suspect narrows to
profile asymmetry (a tabulated non-Gaussian bank, which the operator already accepts;
only the θ-parameterization would be new).

Build widths are 3.5 km/s at every anchor, the strict upper bound that fixes the
kernel radius (nominal FEROS sigma is 2.652; +32% headroom, +2 px half-bandwidth).

Run:  python scripts/hr6819_lsf_run.py [--max-steps 200]
      (expects the FEROS FITS under data/hr6819 or $ALBIREO_HR6819_DATA)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from hr6819_response_run import (
    DV_KMS,
    LIGHT_FRACTIONS,
    V_REL_MAX,
    conjunction_scan,
    priors_and_init,
)
from hr6819_wide_run import HGAMMA_MASK, WINDOW

import albireo as ab
from albireo.forward import data_residual_zscores
from albireo.preprocess import mask_ranges, share_wavelength_grid

LSF_SIGMA_MAX = 3.5  # build width = per-anchor upper bound (nominal 2.652)
LSF_SIGMA_MIN = 1.5
LSF_SIGMA_INIT = 2.652
N_ANCHORS = 13  # 4120-4600 every 40 A ~ the FEROS order scale


def load_wide_lsf():
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
        dataset, dv_kms=DV_KMS, v_margin_kms=V_REL_MAX, lsf_sigma_kms=LSF_SIGMA_MAX
    )
    anchors = tuple(np.linspace(WINDOW[0], WINDOW[1], N_ANCHORS))
    model = ab.MarginalOrbitModel(
        grid,
        dataset,
        light_fractions=LIGHT_FRACTIONS,
        lsf_sigma_v={"FEROS": LSF_SIGMA_MAX},
        lsf_anchors_angstrom={"FEROS": anchors},
        v_rel_max_kms=V_REL_MAX,
        ar1=True,
    )
    return dataset, model, anchors


def run_lsf(dataset, model, t_conj0, n_anchors, *, max_steps: int):
    priors, init = priors_and_init(t_conj0, dataset.n_epochs, response=False)
    priors["log_jitter"] = dist.Normal(jnp.zeros(dataset.n_epochs), 2.0).to_event(1)
    priors["ar1_phi"] = dist.Uniform(-0.9, 0.9)
    priors["lsf_sigma"] = dist.Uniform(
        jnp.full(n_anchors, LSF_SIGMA_MIN), jnp.full(n_anchors, LSF_SIGMA_MAX)
    ).to_event(1)
    init["log_jitter"] = jnp.zeros(dataset.n_epochs)
    init["ar1_phi"] = jnp.asarray(0.3)
    init["lsf_sigma"] = jnp.full(n_anchors, LSF_SIGMA_INIT)

    t0 = time.time()

    def progress(step, potential, grad_norm, params):
        if step % 10 == 0:
            sig = np.asarray(params["lsf_sigma"])
            print(
                f"    [lsf] step {step:4d}  potential {potential:16.3f}  "
                f"|grad| {grad_norm:10.3g}  P {float(params['period']):.5f}  "
                f"K {np.asarray(params['k']).round(3)}  phi {float(params['ar1_phi']):+.3f}  "
                f"sig {sig.min():.2f}-{sig.max():.2f}  ({time.time() - t0:7.1f}s)",
                flush=True,
            )
        return False

    fit = ab.run_map(
        model.model(priors), init=init, max_steps=max_steps, tol=1.0, callback=progress
    )
    elapsed = time.time() - t0

    theta_sites = (
        "period",
        "t_conj",
        "secosw",
        "sesinw",
        "k",
        "log_tau",
        "log_eta",
        "log_jitter",
        "ar1_phi",
        "lsf_sigma",
    )
    theta_map = {s: jnp.asarray(fit.params[s]) for s in theta_sites}
    result = model.marginal(theta_map)
    z = data_residual_zscores(model.problem_at(theta_map), result.d_hat)
    theta_diag = dict(theta_map)
    theta_diag.pop("ar1_phi")
    z_diag = data_residual_zscores(model.problem_at(theta_diag), result.d_hat)

    def lag1(x):
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    alpha = np.exp(np.asarray(fit.params["log_jitter"]))
    k_map = np.asarray(fit.params["k"])
    sig = np.asarray(fit.params["lsf_sigma"])
    out = {
        "steps": fit.num_steps,
        "seconds": elapsed,
        "loglike": float(result.log_likelihood),
        "period": float(fit.params["period"]),
        "ecc": float(fit.params["ecc"]),
        "k_presd": float(k_map[0]),
        "k_be": float(k_map[1]),
        "phi": float(fit.params["ar1_phi"]),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "alpha_med": float(np.median(alpha)),
        "sigma": sig,
        "resid_sd": float(z.std()),
        "resid_lag1": lag1(z),
        "diag_sd": float(z_diag.std()),
        "diag_lag1": lag1(z_diag),
    }
    print(
        f"    [lsf] done: {fit.num_steps} steps, {elapsed:.0f}s, "
        f"loglike {out['loglike']:.1f}, phi {out['phi']:+.3f}, "
        f"chain sd/lag1 {out['resid_sd']:.3f}/{out['resid_lag1']:+.3f}, "
        f"diag sd/lag1 {out['diag_sd']:.3f}/{out['diag_lag1']:+.3f}",
        flush=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    print(
        f"\n=== wide window {WINDOW[0]:.0f}-{WINDOW[1]:.0f} A, Hgamma masked, "
        f"fitted LSF x{N_ANCHORS} (ar1) ===",
        flush=True,
    )
    t0 = time.time()
    dataset, model, anchors = load_wide_lsf()
    print(
        f"  ingest {time.time() - t0:.1f}s: {dataset.n_epochs} epochs, "
        f"{model.problem.grid.n} model px, half-bandwidth {model.half_bandwidth} "
        f"(ar extra {model.problem.ar_bandwidth_extra}, kernel radius "
        f"{model.problem.kernel_radius}); anchors every "
        f"{anchors[1] - anchors[0]:.0f} A",
        flush=True,
    )
    t0 = time.time()
    t_conj0 = conjunction_scan(dataset, model)
    print(f"  conjunction scan {time.time() - t0:.1f}s: t_conj0 = {t_conj0:.4f}", flush=True)
    r = run_lsf(dataset, model, t_conj0, N_ANCHORS, max_steps=args.max_steps)

    print(
        "\n\n=== summary (D36 baseline: P 40.36750 K 63.396/3.482 e 0.0261 phi +0.737 "
        "loglike 3388604.2; lit P 40.3261 K 61.15/3.90 e 0.0289) ===",
        flush=True,
    )
    print(
        f"  wide/lsf: P {r['period']:.5f}  K {r['k_presd']:.3f}/{r['k_be']:.3f}  "
        f"e {r['ecc']:.4f}  phi {r['phi']:+.3f}  "
        f"alpha {r['alpha_min']:.2f}-{r['alpha_max']:.2f} (med {r['alpha_med']:.2f})  "
        f"chain sd/lag1 {r['resid_sd']:.3f}/{r['resid_lag1']:+.3f}  "
        f"diag lag1 {r['diag_lag1']:+.3f}  loglike {r['loglike']:.1f}  "
        f"({r['seconds'] / max(r['steps'], 1):.1f} s/step)"
    )
    sig = r["sigma"]
    with np.printoptions(precision=3, suppress=True):
        print(f"  sigma(lambda) at anchors [km/s]: {sig}")
    print(
        f"  sigma spread {sig.min():.3f}-{sig.max():.3f} "
        f"(nominal 2.652; bounds {LSF_SIGMA_MIN}-{LSF_SIGMA_MAX})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
