"""HR 6819 with the AR(1) correlated-noise model (D34): whiten AND keep the orbit?

D31 measured that per-epoch noise *rescaling* whitens the residual scale and relocates
the period by 174 formal sigmas; D33 measured that the continuum is not the culprit.
This is the remaining recorded check: model the correlation itself — scalar AR(1)
``phi`` shared across epochs (a property of the pipeline's resampling, not of one
exposure) alongside the D31 per-epoch jitters — and see where the orbit goes, what
``phi`` comes out, and whether the chain whitener removes the lag-1 autocorrelation.
Same uniform procedure as scripts/hr6819_response_run.py (conjunction scan, literature
init, 150 L-BFGS steps), so the D33 table is the comparison baseline. The D34 record
ran on the probe assembly path at ~15x the D33 per-step cost; since D35 the correlated
marginal runs the band assembly (link pair tables), so a rerun pays near-D33 cost.

Run:  python scripts/hr6819_ar1_run.py [--windows A B] [--max-steps 150]
"""

from __future__ import annotations

import argparse
import sys
import time

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from hr6819_response_run import (
    WINDOWS,
    conjunction_scan,
    load_window,
    priors_and_init,
)

import albireo as ab
from albireo.forward import data_residual_zscores


def run_ar1(dataset, model, t_conj0, *, max_steps: int):
    priors, init = priors_and_init(t_conj0, dataset.n_epochs, response=False)
    priors["log_jitter"] = dist.Normal(jnp.zeros(dataset.n_epochs), 2.0).to_event(1)
    priors["ar1_phi"] = dist.Uniform(-0.9, 0.9)
    init["log_jitter"] = jnp.zeros(dataset.n_epochs)
    init["ar1_phi"] = jnp.asarray(0.3)

    t0 = time.time()

    def progress(step, potential, grad_norm, params):
        if step % 10 == 0:
            print(
                f"    [ar1] step {step:4d}  potential {potential:16.3f}  "
                f"|grad| {grad_norm:10.3g}  P {float(params['period']):.5f}  "
                f"K {np.asarray(params['k']).round(3)}  phi {float(params['ar1_phi']):+.3f}  "
                f"({time.time() - t0:7.1f}s)",
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
    )
    theta_map = {s: jnp.asarray(fit.params[s]) for s in theta_sites}
    result = model.marginal(theta_map)
    z = data_residual_zscores(model.problem_at(theta_map), result.d_hat)
    # The same residuals read through the *diagonal* whitener (phi and alpha kept):
    # what the chain absorbed shows up here as surviving lag-1 autocorrelation.
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
        "resid_sd": float(z.std()),
        "resid_lag1": lag1(z),
        "diag_sd": float(z_diag.std()),
        "diag_lag1": lag1(z_diag),
    }
    print(
        f"    [ar1] done: {fit.num_steps} steps, {elapsed:.0f}s, "
        f"loglike {out['loglike']:.1f}, phi {out['phi']:+.3f}, "
        f"chain sd/lag1 {out['resid_sd']:.3f}/{out['resid_lag1']:+.3f}, "
        f"diag sd/lag1 {out['diag_sd']:.3f}/{out['diag_lag1']:+.3f}",
        flush=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", default=["A", "B"], choices=list(WINDOWS))
    ap.add_argument("--max-steps", type=int, default=150)
    args = ap.parse_args()

    results = {}
    for name in args.windows:
        lo, hi = WINDOWS[name]
        print(f"\n=== window {name}: {lo:.0f}-{hi:.0f} A (ar1) ===", flush=True)
        t0 = time.time()
        dataset, model = load_window(name, ar1=True)
        print(
            f"  ingest {time.time() - t0:.1f}s: {dataset.n_epochs} epochs, "
            f"{model.problem.grid.n} model px, half-bandwidth {model.half_bandwidth} "
            f"(ar extra {model.problem.ar_bandwidth_extra})",
            flush=True,
        )
        t0 = time.time()
        t_conj0 = conjunction_scan(dataset, model)
        print(f"  conjunction scan {time.time() - t0:.1f}s: t_conj0 = {t_conj0:.4f}", flush=True)
        results[name] = run_ar1(dataset, model, t_conj0, max_steps=args.max_steps)

    print(
        "\n\n=== summary (D33 baselines: A P 40.36566 K 63.308/1.928 e 0.0302 sd 1.674; "
        "B P 40.36091 K 63.575/2.946 e 0.0240 sd 1.401; lit P 40.3261) ===",
        flush=True,
    )
    for name, r in results.items():
        print(
            f"  {name}/ar1: P {r['period']:.5f}  K {r['k_presd']:.3f}/{r['k_be']:.3f}  "
            f"e {r['ecc']:.4f}  phi {r['phi']:+.3f}  "
            f"alpha {r['alpha_min']:.2f}-{r['alpha_max']:.2f} (med {r['alpha_med']:.2f})  "
            f"chain sd/lag1 {r['resid_sd']:.3f}/{r['resid_lag1']:+.3f}  "
            f"diag lag1 {r['diag_lag1']:+.3f}  loglike {r['loglike']:.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
