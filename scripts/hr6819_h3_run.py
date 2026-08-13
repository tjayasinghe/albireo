"""HR 6819, wide window, wavelength-dependent LSF *asymmetry* fitted jointly (D38).

The last instrumental suspect. D37 exonerated LSF width variation (+90.5 nats,
orbit unmoved); the surviving LSF channel is profile asymmetry — the first-order
centroid effect a symmetric kernel cannot produce. This run frees per-anchor
Gauss-Hermite ``h3`` alongside the D37 widths (13 + 13 anchors) on the D36
configuration (wide window, Hgamma masked, per-epoch jitters + shared AR(1) phi).

What the closed loop already measured (tests/test_lsf_h3.py): the free spectra
absorb even a wavelength-varying asymmetry outright — an injected h3 ramp came back
flat with the orbit unharmed — because a static centroid-warp field is representable
by the spectra; the data-identified remainder is the epoch-coupled sampling term
~ c'(lambda) * lambda * (v - v_b)/c, tens of m/s here (math.md §1.3). Fitted h3
profiles are therefore diagnostics, and the orbit's response is the readout: if the
period does not move, every LSF channel this model can express is exonerated, and
the surviving suspects for the 0.041 d literature offset reduce to the Be disc's
variability and the published CCF blending itself.

300 steps (not 200): D37 measured the width directions flattening the already-flat
K_Be axis; the h3 directions add 13 more near-degenerate parameters.

Run:  python scripts/hr6819_h3_run.py [--max-steps 300]
      (expects the FEROS FITS under data/hr6819 or $ALBIREO_HR6819_DATA)
"""

from __future__ import annotations

import argparse
import sys
import time

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from hr6819_lsf_run import (
    LSF_SIGMA_INIT,
    LSF_SIGMA_MAX,
    LSF_SIGMA_MIN,
    N_ANCHORS,
    load_wide_lsf,
)
from hr6819_response_run import conjunction_scan, priors_and_init
from hr6819_wide_run import WINDOW

import albireo as ab
from albireo.forward import data_residual_zscores

H3_MAX = 0.2


def run_h3(dataset, model, t_conj0, n_anchors, *, max_steps: int):
    priors, init = priors_and_init(t_conj0, dataset.n_epochs, response=False)
    priors["log_jitter"] = dist.Normal(jnp.zeros(dataset.n_epochs), 2.0).to_event(1)
    priors["ar1_phi"] = dist.Uniform(-0.9, 0.9)
    priors["lsf_sigma"] = dist.Uniform(
        jnp.full(n_anchors, LSF_SIGMA_MIN), jnp.full(n_anchors, LSF_SIGMA_MAX)
    ).to_event(1)
    priors["lsf_h3"] = dist.Uniform(
        jnp.full(n_anchors, -H3_MAX), jnp.full(n_anchors, H3_MAX)
    ).to_event(1)
    init["log_jitter"] = jnp.zeros(dataset.n_epochs)
    init["ar1_phi"] = jnp.asarray(0.3)
    init["lsf_sigma"] = jnp.full(n_anchors, LSF_SIGMA_INIT)
    init["lsf_h3"] = jnp.zeros(n_anchors)

    t0 = time.time()

    def progress(step, potential, grad_norm, params):
        if step % 10 == 0:
            sig = np.asarray(params["lsf_sigma"])
            h3 = np.asarray(params["lsf_h3"])
            print(
                f"    [h3] step {step:4d}  potential {potential:16.3f}  "
                f"|grad| {grad_norm:10.3g}  P {float(params['period']):.5f}  "
                f"K {np.asarray(params['k']).round(3)}  phi {float(params['ar1_phi']):+.3f}  "
                f"sig {sig.min():.2f}-{sig.max():.2f}  "
                f"h3 {h3.min():+.3f}..{h3.max():+.3f}  ({time.time() - t0:7.1f}s)",
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
        "lsf_h3",
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
        "sigma": np.asarray(fit.params["lsf_sigma"]),
        "h3": np.asarray(fit.params["lsf_h3"]),
        "resid_sd": float(z.std()),
        "resid_lag1": lag1(z),
        "diag_sd": float(z_diag.std()),
        "diag_lag1": lag1(z_diag),
    }
    print(
        f"    [h3] done: {fit.num_steps} steps, {elapsed:.0f}s, "
        f"loglike {out['loglike']:.1f}, phi {out['phi']:+.3f}, "
        f"chain sd/lag1 {out['resid_sd']:.3f}/{out['resid_lag1']:+.3f}, "
        f"diag sd/lag1 {out['diag_sd']:.3f}/{out['diag_lag1']:+.3f}",
        flush=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    print(
        f"\n=== wide window {WINDOW[0]:.0f}-{WINDOW[1]:.0f} A, Hgamma masked, "
        f"fitted (sigma, h3) x{N_ANCHORS} (ar1) ===",
        flush=True,
    )
    t0 = time.time()
    dataset, model, _anchors = load_wide_lsf()
    print(
        f"  ingest {time.time() - t0:.1f}s: {dataset.n_epochs} epochs, "
        f"{model.problem.grid.n} model px, half-bandwidth {model.half_bandwidth} "
        f"(ar extra {model.problem.ar_bandwidth_extra}, kernel radius "
        f"{model.problem.kernel_radius})",
        flush=True,
    )
    t0 = time.time()
    t_conj0 = conjunction_scan(dataset, model)
    print(f"  conjunction scan {time.time() - t0:.1f}s: t_conj0 = {t_conj0:.4f}", flush=True)
    r = run_h3(dataset, model, t_conj0, N_ANCHORS, max_steps=args.max_steps)

    print(
        "\n\n=== summary (D36: P 40.36750 K 63.396/3.482 e 0.0261 phi +0.737 loglike "
        "3388604.2; D37: P 40.36769 K1 63.395 e 0.0262 phi +0.737 loglike 3388694.7; "
        "lit P 40.3261 K 61.15/3.90 e 0.0289) ===",
        flush=True,
    )
    print(
        f"  wide/h3: P {r['period']:.5f}  K {r['k_presd']:.3f}/{r['k_be']:.3f}  "
        f"e {r['ecc']:.4f}  phi {r['phi']:+.3f}  "
        f"alpha {r['alpha_min']:.2f}-{r['alpha_max']:.2f} (med {r['alpha_med']:.2f})  "
        f"chain sd/lag1 {r['resid_sd']:.3f}/{r['resid_lag1']:+.3f}  "
        f"diag lag1 {r['diag_lag1']:+.3f}  loglike {r['loglike']:.1f}  "
        f"({r['seconds'] / max(r['steps'], 1):.1f} s/step)"
    )
    sig, h3 = r["sigma"], r["h3"]
    cshift = np.sqrt(3.0) * h3 * sig  # implied per-anchor centroid shift [km/s]
    with np.printoptions(precision=3, suppress=True):
        print(f"  sigma(lambda) at anchors [km/s]: {sig}")
        print(f"  h3(lambda) at anchors:           {h3}")
        print(f"  implied centroid shift [km/s]:   {cshift}")
    print(
        f"  h3 spread {h3.min():+.3f}..{h3.max():+.3f} (bound +-{H3_MAX}); "
        f"centroid-shift spread {cshift.max() - cshift.min():.3f} km/s across the window"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
