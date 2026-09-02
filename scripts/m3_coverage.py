"""M3 injection-coverage study: calibration of the marginal orbital posterior.

For each injection, a truth θ* is drawn from the *sampling priors* (including the
disk constraint and the bandwidth-guard truncation, replicated exactly), a dataset is
simulated at θ*, hyperparameters are refit by ML-II (MAP), and NUTS samples the
orbital posterior. Recorded per site: the SBC rank of the truth among the posterior
draws, the z-score of the posterior mean, and central-interval coverage hits.

This is not strict SBC over the full joint model: the injected spectra are random line
lists rather than draws from the smoothness prior, so the spectral prior is
realistically misspecified and (tau, eta) are refit per injection. Rank uniformity for
the orbital sites remains the calibration target; a plug-in (empirical Bayes) optimism
of a few percent in coverage is the documented trade (math.md §7.3).

Usage:  python scripts/m3_coverage.py --n-inj 24 --out coverage.json
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import albireo as ab
from albireo.forward import with_velocities
from albireo.inference import (
    MarginalOrbitModel,
    _max_relative_shift,
    laplace_inverse_mass,
    orbit_velocities,
    run_map,
    run_nuts,
)
from albireo.kepler import t_peri_from_t_conj
from albireo.simulate import InstrumentSpec, OrbitParams, simulate_dataset
from albireo.simulate import synthetic_deviation_spectrum as synth_spectrum

# Observing setup (matches the gate test in tests/test_inference.py)
GRID = ab.LogGrid.from_wavelength_range(5000.0, 5045.0, dv_kms=5.5)
P0, T0 = 6.31, 2.05
ELL = np.array([0.62, 0.38])
LSF = {"inst": 7.0}
N_EP = 12
V_REL_MAX = 52.0 * 1.2 * 1.35
K_LO, K_HI = np.array([10.0, 5.0]), np.array([45.0, 40.0])
ECC_MAX = 0.95

PRIORS = {
    "period": dist.Normal(P0, 0.003),
    "t_conj": dist.Normal(T0, 0.02),
    "secosw": dist.Uniform(-1.0, 1.0),
    "sesinw": dist.Uniform(-1.0, 1.0),
    "k": dist.Uniform(jnp.asarray(K_LO), jnp.asarray(K_HI)),
    "log_tau": dist.Normal(jnp.full(2, np.log(300.0)), 3.0),
    "log_eta": dist.Normal(jnp.full(2, np.log(5.0)), 3.0),
}


def draw_truth(rng: np.random.Generator, model: MarginalOrbitModel, bjd) -> dict:
    """Draw θ* from the sampling prior, replicating disk + bandwidth-guard truncation."""
    while True:
        theta = {
            "period": rng.normal(P0, 0.003),
            "t_conj": rng.normal(T0, 0.02),
            "secosw": rng.uniform(-1.0, 1.0),
            "sesinw": rng.uniform(-1.0, 1.0),
            "k": rng.uniform(K_LO, K_HI),
        }
        if theta["secosw"] ** 2 + theta["sesinw"] ** 2 > ECC_MAX:
            continue
        vel = orbit_velocities(theta, bjd)
        rel = float(_max_relative_shift(with_velocities(model.problem, vel)))
        if rel <= model._shift_bound:
            return theta


def one_injection(i: int, seed: int, warmup: int, samples: int) -> dict:
    rng = np.random.default_rng(seed)
    bjd = np.sort(rng.uniform(0.0, 2.4 * P0, N_EP))
    v_bary = rng.uniform(-25.0, 25.0, N_EP)
    comps = [
        synth_spectrum(
            GRID, n_lines=30, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=seed * 7 + 1
        ),
        synth_spectrum(
            GRID, n_lines=25, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=seed * 7 + 2
        ),
    ]
    spec = InstrumentSpec(wave=np.arange(5003.0, 5042.0, 0.11), sigma_v_lsf=7.0, snr=130.0)

    # A scaffold model on a placeholder dataset supplies the static geometry for
    # truth rejection; the real model is rebuilt on the simulated data below.
    ds0, _ = simulate_dataset(
        GRID,
        comps,
        bjd=bjd,
        instruments={"inst": spec},
        light_fractions=ELL,
        velocities=np.zeros((2, N_EP)),
        v_bary=v_bary,
        frame="topocentric",
        seed=seed,
    )
    scaffold = MarginalOrbitModel(
        GRID, ds0, light_fractions=ELL, lsf_sigma_v=LSF, v_rel_max_kms=V_REL_MAX
    )
    theta_true = draw_truth(rng, scaffold, bjd)

    ecc = theta_true["secosw"] ** 2 + theta_true["sesinw"] ** 2
    omega = float(np.arctan2(theta_true["sesinw"], theta_true["secosw"]))
    tperi = float(
        t_peri_from_t_conj(theta_true["t_conj"], period=theta_true["period"], ecc=ecc, omega=omega)
    )
    orbit = OrbitParams(
        period=float(theta_true["period"]),
        t_peri=tperi,
        ecc=float(ecc),
        omega=omega,
        k=tuple(theta_true["k"]),
    )
    ds, _ = simulate_dataset(
        GRID,
        comps,
        bjd=bjd,
        instruments={"inst": spec},
        light_fractions=ELL,
        orbit=orbit,
        v_bary=v_bary,
        frame="topocentric",
        gap_fraction=0.01,
        cosmic_fraction=0.002,
        seed=seed + 1,
    )
    model = MarginalOrbitModel(
        GRID, ds, light_fractions=ELL, lsf_sigma_v=LSF, v_rel_max_kms=V_REL_MAX
    )

    init = {
        **theta_true,
        "k": jnp.asarray(theta_true["k"]),
        "log_tau": jnp.full(2, np.log(300.0)),
        "log_eta": jnp.full(2, np.log(5.0)),
    }
    t0 = time.perf_counter()
    map_res = run_map(model.model(PRIORS), init=init)
    t_map = time.perf_counter() - t0

    hyper = {"log_tau": map_res.params["log_tau"], "log_eta": map_res.params["log_eta"]}
    orbit_priors = {k: v for k, v in PRIORS.items() if k not in hyper}
    nuts_model = model.model(orbit_priors, fixed=hyper)
    t0 = time.perf_counter()
    mcmc = run_nuts(
        nuts_model,
        rng_key=jax.random.PRNGKey(seed),
        init=map_res.params,
        inverse_mass_matrix=laplace_inverse_mass(nuts_model, map_res.params),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=1,
    )
    post = mcmc.get_samples()
    jax.block_until_ready(post["k"])  # mcmc.run dispatches asynchronously
    t_nuts = time.perf_counter() - t0
    extra = mcmc.get_extra_fields()
    truth_flat = {
        "period": float(theta_true["period"]),
        "t_conj": float(theta_true["t_conj"]),
        "secosw": float(theta_true["secosw"]),
        "sesinw": float(theta_true["sesinw"]),
        "k1": float(theta_true["k"][0]),
        "k2": float(theta_true["k"][1]),
    }
    site_draws = {
        "period": np.asarray(post["period"]),
        "t_conj": np.asarray(post["t_conj"]),
        "secosw": np.asarray(post["secosw"]),
        "sesinw": np.asarray(post["sesinw"]),
        "k1": np.asarray(post["k"])[:, 0],
        "k2": np.asarray(post["k"])[:, 1],
    }
    rec: dict = {
        "injection": i,
        "seed": seed,
        "truth": truth_flat,
        "divergences": int(np.sum(np.asarray(extra["diverging"]))),
        "mean_leapfrogs": float(np.mean(np.asarray(extra["num_steps"]))),
        "map_converged": bool(map_res.converged),
        "map_steps": int(map_res.num_steps),
        "t_map_s": round(t_map, 2),
        "t_nuts_s": round(t_nuts, 2),
        "sites": {},
    }
    for name, draws in site_draws.items():
        tr = truth_flat[name]
        rec["sites"][name] = {
            "rank": int(np.sum(draws < tr)),
            "n_draws": int(draws.size),
            "z": float((draws.mean() - tr) / draws.std()),
            "in68": bool(np.percentile(draws, 16) < tr < np.percentile(draws, 84)),
            "in90": bool(np.percentile(draws, 5) < tr < np.percentile(draws, 95)),
            "rel_err_mean": float(abs(draws.mean() - tr) / max(abs(tr), 1e-12)),
        }
    return rec


def summarize(records: list[dict]) -> None:
    n = len(records)
    print(f"\n=== coverage summary over {n} injections ===")
    print(f"total divergences: {sum(r['divergences'] for r in records)}")
    print(f"MAP converged: {sum(r['map_converged'] for r in records)}/{n}")
    print(f"{'site':>8} {'cov68':>6} {'cov90':>6} {'mean|z|':>8} {'max rank-KS':>12}")
    for name in ("period", "t_conj", "secosw", "sesinw", "k1", "k2"):
        c68 = np.mean([r["sites"][name]["in68"] for r in records])
        c90 = np.mean([r["sites"][name]["in90"] for r in records])
        zs = np.array([r["sites"][name]["z"] for r in records])
        # KS distance of normalized ranks against uniform
        u = np.sort(
            [(r["sites"][name]["rank"] + 0.5) / (r["sites"][name]["n_draws"] + 1) for r in records]
        )
        ks = np.max(np.abs(u - (np.arange(1, n + 1) - 0.5) / n))
        print(f"{name:>8} {c68:6.2f} {c90:6.2f} {np.mean(np.abs(zs)):8.2f} {ks:12.3f}")
    k_err = [
        max(r["sites"]["k1"]["rel_err_mean"], r["sites"]["k2"]["rel_err_mean"]) for r in records
    ]
    print(
        f"worst per-injection K error: median {100 * np.median(k_err):.2f}%  "
        f"max {100 * np.max(k_err):.2f}%"
    )
    binom_sd = np.sqrt(0.9 * 0.1 / n)
    print(f"(binomial 1-sigma on cov90 at n={n}: +/-{binom_sd:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-inj", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--samples", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--out", type=str, default="m3_coverage.json")
    args = ap.parse_args()

    records = []
    t_start = time.perf_counter()
    for i in range(args.n_inj):
        rec = one_injection(i, seed=args.seed + 1000 * i, warmup=args.warmup, samples=args.samples)
        records.append(rec)
        worst = max(rec["sites"]["k1"]["rel_err_mean"], rec["sites"]["k2"]["rel_err_mean"])
        print(
            f"[{i + 1:2d}/{args.n_inj}] div={rec['divergences']:3d} "
            f"worst K err {100 * worst:5.2f}%  map {rec['t_map_s']:6.1f}s "
            f"nuts {rec['t_nuts_s']:6.1f}s",
            flush=True,
        )
        with open(args.out, "w") as fh:
            json.dump(records, fh, indent=1)
    summarize(records)
    print(f"total wall: {(time.perf_counter() - t_start) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
