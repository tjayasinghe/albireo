"""AI Phoenicis: the tests that decompose the semi-amplitude offset.

The K1 and K2 that ``aiphe_bench.py`` recovers from 36 archival HARPS spectra are 1.4 per
cent low and 0.8 per cent high against Maxted et al. (2020), the same in two disjoint
windows. In the model's own coordinates that is a component-common velocity error of
-0.55 km/s, which accounts for the whole mass-ratio offset, and a component-antisymmetric
one of -0.16 km/s, which does not. The two windows agree to 0.01 per cent, so the cause is
deterministic and the candidates are model errors shared by every window. This script runs
the tests that rank them, each a MAP fit of the orbit under one change to the model:

    eggs_keyed     the six HARPS EGGS exposures (R = 80,000, modelled at R = 115,000 by
                   ``aiphe_bench.py`` because every file has ``INSTRUME = 'HARPS'``) get
                   their own instrument key and their own line-spread width
    eggs_dropped   the same six exposures removed, which is closer to the 33 spectra
                   Gallenne et al. (2019) measured their velocities from
    third_f*       a stationary third component at light fraction f, with the stars'
                   hyperparameters held at their ML-II values; Maxted et al. measure a
                   third light of 0.5 to 1.0 per cent in the TESS band
    third_ml2      the same third component with its own hyperparameters free
    ell2_*         the orbit refitted at five values of the secondary's light fraction,
                   which tests the fitted K rather than the likelihood at two fixed orbits
    ivar_scaled    the inverse variances divided by the measured residual z-RMS squared
    jitter         a shared noise-inflation site fitted alongside the orbit

Every fit reports the fitted period and time of conjunction, and the argument of periastron
from the signs of ``secosw`` and ``sesinw``, which the eccentricity alone does not determine.

``--nuts CONFIG`` samples the orbit of one configuration with NUTS, the hyperparameters
held at their MAP values, and writes the posterior summary.

Run:  python scripts/aiphe_offset_tests.py --data data/aiphe --out results.json
      python scripts/aiphe_offset_tests.py --data data/aiphe --nuts eggs_keyed
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import albireo as ab
from albireo.forward import data_residual_zscores
from albireo.inference import laplace_inverse_mass, run_nuts
from albireo.io import dataset_from_raw, read_raw_spectra
from albireo.preprocess import _replace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiphe_bench import (
    DV_KMS,
    ECC_PUB,
    ETA,
    K1_PUB,
    K2_PUB,
    OMEGA_PUB_DEG,
    P_PUB,
    T0_PUB,
    TAU,
    WINDOW,
    light_fractions,
)

C_KMS = 299792.458
SIGMA_HAM = C_KMS / 115000.0 / 2.3548  # HARPS high-accuracy mode
SIGMA_EGGS = C_KMS / 80000.0 / 2.3548  # HARPS high-efficiency mode
EGGS_KEY = "HARPS_EGGS"
SITES = ("period", "t_conj", "secosw", "sesinw", "k", "log_tau", "log_eta", "log_jitter")


# --- data variants -----------------------------------------------------------


def load_raws(data_dir: Path):
    return read_raw_spectra(str(data_dir / "*.fits"))


def is_eggs(raw) -> bool:
    return raw.resolving_power is not None and raw.resolving_power < 100000


def make_dataset(raws, *, key_eggs: bool = False, drop_eggs: bool = False, window=WINDOW):
    """The benchmark's dataset, optionally with the EGGS epochs keyed or removed."""
    out = []
    for raw in raws:
        if is_eggs(raw):
            if drop_eggs:
                continue
            if key_eggs:
                raw = dataclasses.replace(raw, instrument=EGGS_KEY)
        out.append(raw)
    return dataset_from_raw(out, region=window, region_pad_angstrom=3.0, smooth_angstrom=25.0)


def lsf_for(ds) -> dict[str, float]:
    return {name: (SIGMA_EGGS if name == EGGS_KEY else SIGMA_HAM) for name in ds.instruments}


def scale_ivar(ds, factor: float):
    epochs = tuple(_replace(ep, ivar=ep.ivar * factor) for ep in ds)
    return ab.Dataset(epochs, frame=ds.frame)


# --- one MAP fit -------------------------------------------------------------


def orbit_priors(n_stellar: int, *, k3_halfwidth: float = 0.05) -> dict:
    lo = [30.0, 30.0] + [-k3_halfwidth] * (n_stellar - 2)
    hi = [70.0, 70.0] + [k3_halfwidth] * (n_stellar - 2)
    return {
        "period": dist.Uniform(P_PUB - 0.01, P_PUB + 0.01),
        "t_conj": dist.Uniform(T0_PUB - 0.05, T0_PUB + 0.05),
        "secosw": dist.Uniform(-0.8, 0.8),
        "sesinw": dist.Uniform(-0.8, 0.8),
        "k": dist.Uniform(jnp.array(lo), jnp.array(hi)),
    }


def orbit_init(n_stellar: int) -> dict:
    omega = np.radians(OMEGA_PUB_DEG)
    return {
        "period": P_PUB,
        "t_conj": T0_PUB,
        # Away from the published solution, so that agreement is not due to the init.
        "secosw": float(np.sqrt(ECC_PUB) * np.cos(omega)) * 0.85,
        "sesinw": float(np.sqrt(ECC_PUB) * np.sin(omega)) * 0.85,
        "k": jnp.array([K1_PUB * 0.92, K2_PUB * 1.08] + [0.0] * (n_stellar - 2)),
    }


def theta_of(params: dict, fixed: dict | None) -> dict:
    theta = {s: jnp.asarray(params[s]) for s in SITES if s in params}
    for name, value in (fixed or {}).items():
        theta[name] = jnp.asarray(value)
    return theta


def fit_orbit(
    label: str,
    ds,
    *,
    ell,
    lsf,
    fixed: dict | None = None,
    hyper_free: bool = True,
    jitter: bool = False,
    max_steps: int = 400,
    verbose: bool = True,
) -> dict:
    """MAP fit of the orbit; returns the record the tables below are built from."""
    ell = np.asarray(ell, dtype=np.float64)
    n_stellar = ell.shape[0]
    grid = ab.LogGrid.covering(
        ds, dv_kms=DV_KMS, v_margin_kms=140.0, lsf_sigma_kms=max(lsf.values())
    )
    model = ab.MarginalOrbitModel(
        grid, ds, light_fractions=ell, lsf_sigma_v=lsf, v_rel_max_kms=140.0
    )
    priors = orbit_priors(n_stellar)
    init = orbit_init(n_stellar)
    for name in fixed or {}:
        priors.pop(name, None)
        init.pop(name, None)
    if hyper_free:
        priors["log_tau"] = dist.Normal(jnp.log(TAU) * jnp.ones(n_stellar), 3.0)
        priors["log_eta"] = dist.Normal(jnp.log(ETA) * jnp.ones(n_stellar), 3.0)
        init["log_tau"] = jnp.log(TAU) * jnp.ones(n_stellar)
        init["log_eta"] = jnp.log(ETA) * jnp.ones(n_stellar)
    if jitter:
        priors["log_jitter"] = dist.Normal(0.0, 2.0)
        init["log_jitter"] = jnp.log(2.6)

    def progress(step, potential, grad_norm, params):
        if verbose and step % 50 == 0:
            k = np.asarray(params["k"])
            print(
                f"    [{label}] step {step:4d}  U={potential:.1f}  |g|={grad_norm:.2e}  "
                f"K1={k[0]:.3f} K2={k[1]:.3f}",
                flush=True,
            )

    t0 = time.perf_counter()
    fit = ab.run_map(
        model.model(priors, fixed=fixed), init=init, max_steps=max_steps, callback=progress
    )
    wall = time.perf_counter() - t0
    theta = theta_of(fit.params, fixed)
    result = model.marginal(theta)
    z = np.asarray(data_residual_zscores(model.problem_at(theta), result.d_hat))
    d_hat = np.asarray(result.d_hat)

    k = np.asarray(fit.params["k"], dtype=float)
    omega = float(np.degrees(np.arctan2(fit.params["sesinw"], fit.params["secosw"]))) % 360.0
    rec = {
        "label": label,
        "n_epochs": int(ds.n_epochs),
        "instruments": {name: float(lsf[name]) for name in ds.instruments},
        "light_fractions": [float(x) for x in ell],
        "K1": float(k[0]),
        "K2": float(k[1]),
        "K3": float(k[2]) if k.size > 2 else None,
        "q": float(k[0] / k[1]),
        "dK1_pct": float((k[0] - K1_PUB) / K1_PUB * 100),
        "dK2_pct": float((k[1] - K2_PUB) / K2_PUB * 100),
        "dq_pct": float((k[0] / k[1] - K1_PUB / K2_PUB) / (K1_PUB / K2_PUB) * 100),
        "common_kms": float(((k[0] - K1_PUB) - (k[1] - K2_PUB)) / 2),
        "antisym_kms": float(((k[0] - K1_PUB) + (k[1] - K2_PUB)) / 2),
        "ecc": float(fit.params["ecc"]),
        "omega_deg": omega,
        "secosw": float(fit.params["secosw"]),
        "sesinw": float(fit.params["sesinw"]),
        "period": float(fit.params["period"]),
        "t_conj": float(fit.params["t_conj"]),
        "log_tau": [float(x) for x in np.atleast_1d(theta["log_tau"])],
        "log_eta": [float(x) for x in np.atleast_1d(theta["log_eta"])],
        "jitter": float(np.exp(fit.params["log_jitter"])) if jitter else None,
        "z_rms": float(np.sqrt(np.mean(z**2))),
        "log_likelihood": float(result.log_likelihood),
        "potential": float(fit.potential),
        "grad_norm": float(fit.grad_norm),
        "converged": bool(fit.converged),
        "steps": int(fit.num_steps),
        "wall_s": wall,
        "d_hat_rms": [float(np.sqrt(np.mean(row**2))) for row in d_hat],
        "d_hat_min": [float(row.min()) for row in d_hat],
    }
    print(
        f"  {label:14s} K1 {k[0]:.3f} ({rec['dK1_pct']:+.2f}%)  K2 {k[1]:.3f} "
        f"({rec['dK2_pct']:+.2f}%)  q {rec['q']:.4f} ({rec['dq_pct']:+.2f}%)  "
        f"common {rec['common_kms']:+.3f}  anti {rec['antisym_kms']:+.3f}  "
        f"e {rec['ecc']:.4f}  omega {omega:.2f}  P {rec['period']:.6f}  "
        f"z-RMS {rec['z_rms']:.3f}  |g| {rec['grad_norm']:.1e} "
        f"{'conv' if rec['converged'] else 'CAP'} {rec['steps']} steps {wall:.0f} s",
        flush=True,
    )
    return rec


# --- the campaign ------------------------------------------------------------


def run_tests(raws, out: Path, *, steps: int, only: list[str] | None) -> None:
    results: dict[str, dict] = {}
    if out.exists():
        results = json.loads(out.read_text())
        print(f"resuming: {len(results)} results already in {out}")

    def save():
        out.write_text(json.dumps(results, indent=2))

    def wanted(label: str) -> bool:
        if label in results:
            print(f"  {label}: already done")
            return False
        return only is None or any(label.startswith(o) for o in only)

    ell = light_fractions(float(np.mean(WINDOW)))
    ds_base = make_dataset(raws)
    lsf_base = lsf_for(ds_base)

    print("\n--- baseline: the aiphe_bench.py configuration (36 epochs, one HARPS key) ---")
    if wanted("baseline"):
        results["baseline"] = fit_orbit("baseline", ds_base, ell=ell, lsf=lsf_base, max_steps=steps)
        save()

    print("\n--- test 1: the six EGGS epochs keyed at R = 80,000, then dropped ---")
    if wanted("eggs_keyed"):
        ds = make_dataset(raws, key_eggs=True)
        print(f"  instruments {ds.instruments}; widths {lsf_for(ds)}")
        results["eggs_keyed"] = fit_orbit(
            "eggs_keyed", ds, ell=ell, lsf=lsf_for(ds), max_steps=steps
        )
        save()
    if wanted("eggs_dropped"):
        ds = make_dataset(raws, drop_eggs=True)
        results["eggs_dropped"] = fit_orbit(
            "eggs_dropped", ds, ell=ell, lsf=lsf_for(ds), max_steps=steps
        )
        save()

    print("\n--- test 2: a stationary third component, light fraction swept ---")
    base = results.get("baseline")
    if base is None:
        print("  needs the baseline's ML-II hyperparameters; skipped")
    else:
        lt, le = base["log_tau"], base["log_eta"]
        fixed2 = {"log_tau": jnp.array(lt), "log_eta": jnp.array(le)}
        fixed3 = {
            "log_tau": jnp.array([*lt, lt[1]]),
            "log_eta": jnp.array([*le, le[1]]),
        }
        if wanted("third_f0.00pct"):
            results["third_f0.00pct"] = fit_orbit(
                "third_f0.00pct",
                ds_base,
                ell=ell,
                lsf=lsf_base,
                fixed=fixed2,
                hyper_free=False,
                max_steps=steps,
            )
            save()
        for f in (0.0025, 0.005, 0.010, 0.015):
            label = f"third_f{100 * f:.2f}pct"
            if wanted(label):
                ell3 = np.array([ell[0] * (1 - f), ell[1] * (1 - f), f])
                results[label] = fit_orbit(
                    label,
                    ds_base,
                    ell=ell3,
                    lsf=lsf_base,
                    fixed=fixed3,
                    hyper_free=False,
                    max_steps=steps,
                )
                save()
        if wanted("third_ml2"):
            f = 0.005
            ell3 = np.array([ell[0] * (1 - f), ell[1] * (1 - f), f])
            results["third_ml2"] = fit_orbit(
                "third_ml2", ds_base, ell=ell3, lsf=lsf_base, max_steps=steps
            )
            save()

    print("\n--- test 3: the orbit refitted at five secondary light fractions ---")
    for l2 in (0.38, 0.42, 0.46, 0.50, 0.53):
        label = f"ell2_{l2:.2f}"
        if wanted(label):
            results[label] = fit_orbit(
                label, ds_base, ell=np.array([1 - l2, l2]), lsf=lsf_base, max_steps=steps
            )
            save()

    print("\n--- period held at the photometric value (Kirkby-Kent et al. 2016) ---")
    if wanted("period_fixed"):
        results["period_fixed"] = fit_orbit(
            "period_fixed", ds_base, ell=ell, lsf=lsf_base, fixed={"period": P_PUB}, max_steps=steps
        )
        save()
    if wanted("period_fixed_eggs"):
        ds = make_dataset(raws, key_eggs=True)
        results["period_fixed_eggs"] = fit_orbit(
            "period_fixed_eggs",
            ds,
            ell=ell,
            lsf=lsf_for(ds),
            fixed={"period": P_PUB},
            max_steps=steps,
        )
        save()

    print("\n--- test 4: the weights rescaled to the measured z-RMS, and a jitter site ---")
    if wanted("ivar_scaled"):
        z = results["baseline"]["z_rms"] if "baseline" in results else 2.64
        results["ivar_scaled"] = fit_orbit(
            "ivar_scaled", scale_ivar(ds_base, 1.0 / z**2), ell=ell, lsf=lsf_base, max_steps=steps
        )
        results["ivar_scaled"]["ivar_factor"] = 1.0 / z**2
        save()
    if wanted("jitter"):
        results["jitter"] = fit_orbit(
            "jitter", ds_base, ell=ell, lsf=lsf_base, jitter=True, max_steps=steps
        )
        save()

    print("\n--- combined: EGGS keyed and the weights rescaled ---")
    if wanted("eggs_keyed_scaled"):
        ds = make_dataset(raws, key_eggs=True)
        z = results["eggs_keyed"]["z_rms"] if "eggs_keyed" in results else 2.64
        results["eggs_keyed_scaled"] = fit_orbit(
            "eggs_keyed_scaled",
            scale_ivar(ds, 1.0 / z**2),
            ell=ell,
            lsf=lsf_for(ds),
            max_steps=steps,
        )
        save()

    print("\n=== summary ===")
    print(
        f"{'config':18s} {'K1':>8s} {'K2':>8s} {'q':>8s} {'dq%':>7s} {'common':>8s} "
        f"{'anti':>8s} {'omega':>7s} {'P':>10s} {'zRMS':>6s}"
    )
    for label, r in results.items():
        print(
            f"{label:18s} {r['K1']:8.3f} {r['K2']:8.3f} {r['q']:8.4f} {r['dq_pct']:+7.2f} "
            f"{r['common_kms']:+8.3f} {r['antisym_kms']:+8.3f} {r['omega_deg']:7.2f} "
            f"{r['period']:10.6f} {r['z_rms']:6.3f}"
        )
    print(f"published          {K1_PUB:8.3f} {K2_PUB:8.3f} {K1_PUB / K2_PUB:8.4f}")


# --- NUTS on one configuration ----------------------------------------------


def run_posterior(
    raws, config: str, out: Path, *, warmup: int, samples: int, chains: int, steps: int
):
    ell = light_fractions(float(np.mean(WINDOW)))
    if config == "baseline":
        ds = make_dataset(raws)
    elif config == "eggs_keyed":
        ds = make_dataset(raws, key_eggs=True)
    elif config == "eggs_dropped":
        ds = make_dataset(raws, drop_eggs=True)
    else:
        raise SystemExit(f"unknown configuration {config!r}")
    lsf = lsf_for(ds)
    jitter = True  # the posterior must use the measured noise scale, not the estimate
    print(f"MAP first ({config}, {ds.n_epochs} epochs, jitter site on) ...")
    rec = fit_orbit(f"{config}+jitter", ds, ell=ell, lsf=lsf, jitter=jitter, max_steps=steps)

    grid = ab.LogGrid.covering(
        ds, dv_kms=DV_KMS, v_margin_kms=140.0, lsf_sigma_kms=max(lsf.values())
    )
    model = ab.MarginalOrbitModel(
        grid, ds, light_fractions=ell, lsf_sigma_v=lsf, v_rel_max_kms=140.0
    )
    fixed = {
        "log_tau": jnp.array(rec["log_tau"]),
        "log_eta": jnp.array(rec["log_eta"]),
        "log_jitter": jnp.log(rec["jitter"]),
    }
    priors = orbit_priors(2)
    numpyro_model = model.model(priors, fixed=fixed)
    init = {
        "period": rec["period"],
        "t_conj": rec["t_conj"],
        "secosw": rec["secosw"],
        "sesinw": rec["sesinw"],
        "k": jnp.array([rec["K1"], rec["K2"]]),
    }
    t0 = time.perf_counter()
    mass = laplace_inverse_mass(numpyro_model, init)
    print(f"Laplace mass matrix in {time.perf_counter() - t0:.0f} s; sampling ...", flush=True)
    t0 = time.perf_counter()
    mcmc = run_nuts(
        numpyro_model,
        rng_key=jax.random.PRNGKey(1),
        init=init,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        inverse_mass_matrix=mass,
        progress_bar=True,
    )
    wall = time.perf_counter() - t0
    mcmc.print_summary()
    post = mcmc.get_samples(group_by_chain=True)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    k = np.asarray(post["k"])  # (chains, samples, 2)
    q = k[..., 0] / k[..., 1]
    from numpyro.diagnostics import effective_sample_size, gelman_rubin

    summary = {
        "config": config,
        "n_epochs": int(ds.n_epochs),
        "map": rec,
        "num_warmup": warmup,
        "num_samples": samples,
        "num_chains": chains,
        "wall_s": wall,
        "divergences": int(np.sum(np.asarray(extra["diverging"]))),
        "mean_tree_steps": float(np.mean(np.asarray(extra["num_steps"]))),
        "K1": {"mean": float(k[..., 0].mean()), "std": float(k[..., 0].std())},
        "K2": {"mean": float(k[..., 1].mean()), "std": float(k[..., 1].std())},
        "q": {"mean": float(q.mean()), "std": float(q.std())},
        "ecc": {
            "mean": float(np.asarray(post["ecc"]).mean()),
            "std": float(np.asarray(post["ecc"]).std()),
        },
        "omega_deg": {
            "mean": float(np.degrees(np.asarray(post["omega"])).mean()),
            "std": float(np.degrees(np.asarray(post["omega"])).std()),
        },
        "period": {
            "mean": float(np.asarray(post["period"]).mean()),
            "std": float(np.asarray(post["period"]).std()),
        },
        "t_conj": {
            "mean": float(np.asarray(post["t_conj"]).mean()),
            "std": float(np.asarray(post["t_conj"]).std()),
        },
        "rhat": {
            name: [float(x) for x in np.atleast_1d(gelman_rubin(np.asarray(post[name])))]
            for name in ("period", "t_conj", "secosw", "sesinw", "k")
        },
        "ess": {
            name: [float(x) for x in np.atleast_1d(effective_sample_size(np.asarray(post[name])))]
            for name in ("period", "t_conj", "secosw", "sesinw", "k")
        },
        "sigma_from_published": {
            "K1": float((k[..., 0].mean() - K1_PUB) / k[..., 0].std()),
            "K2": float((k[..., 1].mean() - K2_PUB) / k[..., 1].std()),
        },
    }
    out.write_text(json.dumps(summary, indent=2))
    np.savez(
        out.with_suffix(".npz"),
        **{name: np.asarray(v) for name, v in post.items()},
        diverging=np.asarray(extra["diverging"]),
        num_steps=np.asarray(extra["num_steps"]),
    )
    s1, s2, sq = summary["K1"], summary["K2"], summary["q"]
    off = summary["sigma_from_published"]
    print(
        f"\nposterior ({config}): K1 {s1['mean']:.3f} +/- {s1['std']:.3f}, "
        f"K2 {s2['mean']:.3f} +/- {s2['std']:.3f}, q {sq['mean']:.4f} +/- {sq['std']:.4f}; "
        f"published offset {off['K1']:+.1f} and {off['K2']:+.1f} sigma; "
        f"{summary['divergences']} divergences; {wall:.0f} s"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="aiphe_offset_tests.json")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--only", nargs="*", default=None, help="label prefixes to run")
    ap.add_argument("--nuts", default=None, metavar="CONFIG")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--chains", type=int, default=2)
    args = ap.parse_args()

    raws = load_raws(Path(args.data))
    n_eggs = sum(is_eggs(r) for r in raws)
    print(f"albireo {ab.__version__}: {len(raws)} spectra, {n_eggs} EGGS-mode at R = 80,000")
    print(f"window {WINDOW}, sigma_HAM {SIGMA_HAM:.3f} km/s, sigma_EGGS {SIGMA_EGGS:.3f} km/s")
    if args.nuts:
        run_posterior(
            raws,
            args.nuts,
            Path(args.out),
            warmup=args.warmup,
            samples=args.samples,
            chains=args.chains,
            steps=args.steps,
        )
    else:
        run_tests(raws, Path(args.out), steps=args.steps, only=args.only)


if __name__ == "__main__":
    main()
