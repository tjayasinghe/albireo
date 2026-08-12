"""Design-target scale ladder: jitted marginal eval + gradient timings (M5 gate).

Reproduces the benchmark ladder of ``docs/benchmarks.md`` (SB2, 50 epochs, fixed
per-component half-bandwidth 256 => stacked p = 513): four problem sizes up to the
design target of ~2x10^5 model pixels. The problem is passed to ``jax.jit`` as a
pytree *argument* (never closed over — see design ledger D27), and the marginal is
assembled by the direct band path (D28).

Run:  python scripts/m5_scale_bench.py [--rows 0 3] [--reps 2]
On GPU, run as-is; JAX picks up the accelerator. x64 stays on (the solver contract).
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import albireo as ab
from albireo.forward import build_problem, with_velocities
from albireo.likelihood import marginal_loglikelihood
from albireo.priors import SmoothnessPrior

B_NAT = 256
N_EP = 50
PRIOR = SmoothnessPrior(jnp.asarray([300.0, 300.0]), jnp.asarray([5.0, 5.0]))

# (wave_lo, wave_hi, native_step): ~31.7k, 74.3k, 135k, 203k model px at dv = 3 km/s.
ROWS = [
    (4000.0, 5495.0, 0.114),
    (4000.0, 8415.0, 0.1331),
    (4000.0, 15452.0, 0.1722),
    (3600.0, 27570.0, 0.2244),
]


def make_problem(wave_lo, wave_hi, native_step):
    grid = ab.LogGrid.from_wavelength_range(wave_lo, wave_hi, dv_kms=3.0)
    rng = np.random.default_rng(7)
    bjd = np.sort(rng.uniform(0.0, 30.0, N_EP))
    comps = [
        ab.synthetic_deviation_spectrum(
            grid, n_lines=120, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=s
        )
        for s in (1, 2)
    ]
    orbit = ab.OrbitParams(period=11.3, t_peri=2.0, ecc=0.3, omega=0.8, k=(45.0, 70.0))
    wave_native = np.arange(wave_lo + 3.0, wave_hi - 3.0, native_step)
    spec = ab.InstrumentSpec(wave=wave_native, sigma_v_lsf=7.0, snr=100.0)
    ds, truth = ab.simulate_dataset(
        grid,
        comps,
        bjd=bjd,
        instruments={"inst": spec},
        light_fractions=np.array([0.6, 0.4]),
        orbit=orbit,
        frame="barycentric",
        seed=3,
    )
    problem = build_problem(
        grid,
        ds,
        velocities=truth.velocities,
        light_fractions=np.array([0.6, 0.4]),
        lsf_sigma_v={"inst": 7.0},
    )
    return grid, problem, jnp.asarray(truth.velocities), wave_native.size


def timeit(fn, *args, reps):
    out = fn(*args)
    jax.block_until_ready(out)  # compile + warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
        jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs=2, default=(0, 3), help="first/last ladder row")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    print(f"backend: {jax.default_backend()}  |  {N_EP} epochs, SB2, b_nat={B_NAT} (p=513)")
    print(f"{'n (model px)':>14} {'native px/ep':>14} {'eval':>10} {'grad':>10}")

    @jax.jit
    def loss(pb, v):
        return marginal_loglikelihood(
            with_velocities(pb, v), PRIOR, half_bandwidth=B_NAT
        ).log_likelihood

    grad = jax.jit(jax.grad(loss, argnums=1))

    for row in ROWS[args.rows[0] : args.rows[1] + 1]:
        grid, problem, vel, n_native = make_problem(*row)
        t_eval = timeit(loss, problem, vel, reps=args.reps)
        t_grad = timeit(grad, problem, vel, reps=max(1, args.reps - 1))
        print(f"{grid.n:>14,} {n_native:>14,} {t_eval:>9.2f}s {t_grad:>9.2f}s")


if __name__ == "__main__":
    main()
