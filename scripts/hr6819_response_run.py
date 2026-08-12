"""HR 6819 with the per-epoch response site (D33): does the period offset survive?

The D30 record (docs/benchmarks.md, "HR 6819") closes by naming this run: what the
fit needs is "an honest noise model, a wider window, and a check of whether the period
offset survives a per-epoch continuum treatment". D31 built the noise-scale site and
measured it relocating the period by 174 formal sigmas. This script is the continuum
check: both windows, MAP with and without an order-2 per-epoch multiplicative response
site (``albireo.forward.with_response``, D33), everything else held at the D30
configuration so the comparison is against the recorded numbers.

Uncertainties are *conditional-orbit Laplace*: curvature over the orbit sites only,
hyperparameters (and response coefficients, where fitted) held at their MAP values,
pushed to constrained space by sampling. Statistical only — the D30/D31 record is
explicit that window-to-window and noise-model-to-noise-model spread is the honest
error bar on this dataset.

Run:  python scripts/hr6819_response_run.py [--windows A B] [--max-steps 150]
      (expects the FEROS FITS under data/hr6819 or $ALBIREO_HR6819_DATA;
       scripts/download_hr6819.py fetches them)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree
from numpyro.distributions.transforms import biject_to
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model

import albireo as ab
from albireo.forward import data_residual_zscores
from albireo.preprocess import share_wavelength_grid

WINDOWS = {"A": (4380.0, 4600.0), "B": (4120.0, 4330.0)}
DV_KMS = 1.5
LSF_SIGMA = ab.C_KMS / (48_000.0 * 2.0 * np.sqrt(2.0 * np.log(2.0)))  # FEROS, 2.652 km/s
V_REL_MAX = 90.0
LIGHT_FRACTIONS = (0.45, 0.55)  # optical, Bodensteiner et al. 2020 — an input (D13)
RESPONSE_ORDER = 2
RESPONSE_PRIOR_SIGMA = 0.02  # preprocess.normalize should be good to ~1%; allow 2%
P_LIT, K_LIT = 40.3261, (61.15, 3.90)  # Klement et al. 2025, Table 3
ORBIT_SITES = ("period", "t_conj", "secosw", "sesinw", "k")

LOG_TAU0 = np.log(np.array([1.0e3, 1.0e8]))
LOG_ETA0 = np.log(np.array([1.0e2, 1.0e2]))


def load_window(name: str):
    data_dir = pathlib.Path(
        os.environ.get("ALBIREO_HR6819_DATA")
        or pathlib.Path(__file__).resolve().parent.parent / "data" / "hr6819"
    )
    dataset = ab.read_dataset(
        str(data_dir / "*.fits"),
        instrument="FEROS",
        region=WINDOWS[name],
        region_pad_angstrom=60.0,
        smooth_angstrom=120.0,
    )
    dataset = ab.Dataset(share_wavelength_grid(list(dataset)), frame=dataset.frame)
    grid = ab.LogGrid.covering(
        dataset, dv_kms=DV_KMS, v_margin_kms=V_REL_MAX, lsf_sigma_kms=LSF_SIGMA
    )
    model = ab.MarginalOrbitModel(
        grid,
        dataset,
        light_fractions=LIGHT_FRACTIONS,
        lsf_sigma_v={"FEROS": LSF_SIGMA},
        v_rel_max_kms=V_REL_MAX,
    )
    return dataset, model


def conjunction_scan(dataset, model) -> float:
    def theta(t):
        return {
            "period": jnp.asarray(P_LIT),
            "t_conj": jnp.asarray(t),
            "secosw": jnp.asarray(1e-3),
            "sesinw": jnp.asarray(1e-3),
            "k": jnp.asarray(K_LIT),
            "log_tau": jnp.asarray(LOG_TAU0),
            "log_eta": jnp.asarray(LOG_ETA0),
        }

    trial = np.min(dataset.bjd) + np.linspace(0.0, P_LIT, 41, endpoint=False)
    scan = np.array([float(model.log_likelihood(theta(t))) for t in trial])
    return float(trial[int(np.argmax(scan))])


def priors_and_init(t_conj0: float, n_epochs: int, *, response: bool):
    priors = {
        "period": dist.Normal(P_LIT, 0.5),
        "t_conj": dist.Normal(t_conj0, 2.0),
        "secosw": dist.Uniform(-1.0, 1.0),
        "sesinw": dist.Uniform(-1.0, 1.0),
        "k": dist.Uniform(jnp.array([20.0, 0.1]), jnp.array([120.0, 40.0])),
        "log_tau": dist.Normal(jnp.asarray(LOG_TAU0), 6.0),
        "log_eta": dist.Normal(jnp.asarray(LOG_ETA0), 6.0),
    }
    init = {
        "period": P_LIT,
        "t_conj": t_conj0,
        "secosw": 0.05,
        "sesinw": 0.05,
        "k": jnp.array(K_LIT),
        "log_tau": jnp.asarray(LOG_TAU0),
        "log_eta": jnp.asarray(LOG_ETA0),
    }
    if response:
        priors["response"] = dist.Normal(
            jnp.zeros((n_epochs, RESPONSE_ORDER + 1)), RESPONSE_PRIOR_SIGMA
        )
        init["response"] = jnp.zeros((n_epochs, RESPONSE_ORDER + 1))
    return priors, init


def conditional_orbit_sigmas(model, priors, fit_params, *, seed=0, n_draws=4096):
    """Laplace curvature over the orbit sites, nuisances fixed at MAP, constrained std.

    The unconstrained covariance is pushed to constrained space by sampling and
    applying the sites' bijections directly (the k sites live behind a sigmoid, so the
    delta method would need its jacobian anyway) — NOT via numpyro's postprocess,
    which replays the whole model (marginal likelihood included) per draw.
    Deterministic given the seed.
    """
    fixed = {s: fit_params[s] for s in fit_params if s in ("log_tau", "log_eta", "response")}
    orbit_priors = {s: d for s, d in priors.items() if s in ORBIT_SITES}
    cond_model = model.model(orbit_priors, fixed=fixed)
    model_info = initialize_model(
        jax.random.PRNGKey(seed),
        cond_model,
        init_strategy=init_to_value(values={s: fit_params[s] for s in ORBIT_SITES}),
        dynamic_args=True,
        model_args=cond_model.model_args,
    )
    potential = model_info.potential_fn(*cond_model.model_args)
    z = jax.tree.map(jnp.asarray, model_info.param_info.z)
    flat, unravel = ravel_pytree(z)
    hess = jax.jacrev(jax.jacrev(lambda zf: potential(unravel(zf))))(flat)
    hess = 0.5 * (hess + hess.T)
    eigval, eigvec = jnp.linalg.eigh(hess)
    eigval = jnp.maximum(eigval, 1e-10 * jnp.max(eigval))
    cov = eigvec @ jnp.diag(1.0 / eigval) @ eigvec.T
    chol = np.linalg.cholesky(np.asarray(cov))
    draws = (
        np.asarray(flat)[None, :]
        + np.random.default_rng(seed).normal(size=(n_draws, flat.size)) @ chol.T
    )
    zs = jax.vmap(unravel)(jnp.asarray(draws))  # dict of (n_draws, ...) unconstrained
    cons = {s: np.asarray(biject_to(orbit_priors[s].support)(zs[s])) for s in zs}
    cons["ecc"] = cons["secosw"] ** 2 + cons["sesinw"] ** 2
    return {s: np.std(v, axis=0, ddof=1) for s, v in cons.items()}


def run_config(dataset, model, t_conj0, *, response: bool, max_steps: int, warm=None):
    label = "response" if response else "baseline"
    priors, init = priors_and_init(t_conj0, dataset.n_epochs, response=response)
    if warm is not None:  # warm-start the orbit/hypers from the baseline MAP
        for s in ("period", "t_conj", "secosw", "sesinw", "k", "log_tau", "log_eta"):
            init[s] = jnp.asarray(warm[s])
    t0 = time.time()
    last = {"t": t0}

    def progress(step, potential, grad_norm, params):
        if step % 10 == 0 or time.time() - last["t"] > 120.0:
            last["t"] = time.time()
            print(
                f"    [{label}] step {step:4d}  potential {potential:16.3f}  "
                f"|grad| {grad_norm:10.3g}  P {float(params['period']):.5f}  "
                f"K {np.asarray(params['k']).round(3)}  ({time.time() - t0:6.1f}s)",
                flush=True,
            )
        return False

    fit = ab.run_map(
        model.model(priors), init=init, max_steps=max_steps, tol=1.0, callback=progress
    )
    elapsed = time.time() - t0

    theta_sites = [
        s
        for s in ("period", "t_conj", "secosw", "sesinw", "k", "log_tau", "log_eta", "response")
        if s in fit.params
    ]
    theta_map = {s: jnp.asarray(fit.params[s]) for s in theta_sites}
    result = model.marginal(theta_map)
    zsc = data_residual_zscores(model.problem_at(theta_map), result.d_hat)
    sig = conditional_orbit_sigmas(model, priors, fit.params)

    k_map = np.asarray(fit.params["k"])
    out = {
        "label": label,
        "steps": fit.num_steps,
        "seconds": elapsed,
        "loglike": float(result.log_likelihood),
        "period": float(fit.params["period"]),
        "period_sig": float(sig["period"]),
        "ecc": float(fit.params["ecc"]),
        "ecc_sig": float(sig["ecc"]),
        "k_presd": float(k_map[0]),
        "k_presd_sig": float(sig["k"][0]),
        "k_be": float(k_map[1]),
        "k_be_sig": float(sig["k"][1]),
        "resid_sd": float(zsc.std()),
        "params": fit.params,
    }
    if response:
        c = np.asarray(fit.params["response"])
        out["response_rms"] = float(np.sqrt(np.mean(c**2)))
        out["response_diff_rms"] = float(np.sqrt(np.mean((c - c.mean(axis=0)) ** 2)))
        out["response_max"] = float(np.max(np.abs(c)))
    print(
        f"    [{label}] done: {fit.num_steps} steps, {elapsed:.0f}s, "
        f"loglike {out['loglike']:.1f}, resid sd {out['resid_sd']:.3f}",
        flush=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", default=["A", "B"], choices=list(WINDOWS))
    ap.add_argument("--max-steps", type=int, default=150)
    args = ap.parse_args()

    results: dict[str, dict[str, dict]] = {}
    for name in args.windows:
        lo, hi = WINDOWS[name]
        print(f"\n=== window {name}: {lo:.0f}-{hi:.0f} A ===", flush=True)
        t0 = time.time()
        dataset, model = load_window(name)
        print(
            f"  ingest {time.time() - t0:.1f}s: {dataset.n_epochs} epochs, "
            f"{model.problem.grid.n} model px, {len(model.problem.groups)} group(s)",
            flush=True,
        )
        t0 = time.time()
        t_conj0 = conjunction_scan(dataset, model)
        print(f"  conjunction scan {time.time() - t0:.1f}s: t_conj0 = {t_conj0:.4f}", flush=True)

        base = run_config(dataset, model, t_conj0, response=False, max_steps=args.max_steps)
        resp = run_config(
            dataset, model, t_conj0, response=True, max_steps=args.max_steps, warm=base["params"]
        )
        results[name] = {"baseline": base, "response": resp}

    print(
        "\n\n=== summary (literature: P 40.3261+-0.0013, "
        "K 61.15+-0.88 / 3.90+-0.27, e 0.0289+-0.0058) ==="
    )
    print(
        f"{'window/config':>22} {'period [d]':>20} {'K_presd':>16} "
        f"{'K_Be':>14} {'ecc':>14} {'resid sd':>9} {'loglike':>14}"
    )
    for name, cfgs in results.items():
        for label, r in cfgs.items():
            print(
                f"{name + '/' + label:>22} "
                f"{r['period']:12.5f}+-{r['period_sig']:.5f} "
                f"{r['k_presd']:9.3f}+-{r['k_presd_sig']:.3f} "
                f"{r['k_be']:8.3f}+-{r['k_be_sig']:.3f} "
                f"{r['ecc']:8.4f}+-{r['ecc_sig']:.4f} "
                f"{r['resid_sd']:9.3f} {r['loglike']:14.1f}"
            )
            if "response_rms" in r:
                print(
                    f"{'':>22} response rms {r['response_rms']:.4f} (difference mode "
                    f"{r['response_diff_rms']:.4f}, max |c| {r['response_max']:.4f})"
                )
    if len(results) == 2:
        for label in ("baseline", "response"):
            a, b = (results[w][label] for w in ("A", "B"))
            dp = abs(a["period"] - b["period"])
            sp = np.hypot(a["period_sig"], b["period_sig"])
            dk = abs(a["k_presd"] - b["k_presd"])
            sk = np.hypot(a["k_presd_sig"], b["k_presd_sig"])
            print(
                f"  window A-B [{label}]: dP = {dp:.5f} d ({dp / sp:.1f} sigma), "
                f"dK_presd = {dk:.3f} km/s ({dk / sk:.1f} sigma)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
