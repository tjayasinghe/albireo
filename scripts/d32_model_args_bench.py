"""Closure-captured vs. argument-passed Problem through the numpyro potential (D32).

``MarginalOrbitModel.model()`` formerly closed over its ``Problem``; the jitted numpyro
potential then baked the problem's arrays into the HLO as constants, and XLA's
compile-time folding of every θ-independent subgraph (chiefly the velocity-independent
kernel-sandwich pre-pass, which D29 measured at ~9 GB un-batched at the design target)
ran at compile time, against compile-time memory. D27 fixed this for
``MarginalOrbitModel.marginal`` by passing the problem as a jit argument; D32 threads
the same contract through ``run_map`` / ``laplace_inverse_mass`` / ``run_nuts``.

This script measures one ``value_and_grad(potential)``, the graph L-BFGS and NUTS
compile, in both regimes, on the m5 ladder's synthetic SB2 (50 epochs, p = 513). One
mode per invocation: the peak-working-set counter is per-process and monotone, so the
two regimes must not share a process.

Run:  python scripts/d32_model_args_bench.py --mode arg --row 0
      python scripts/d32_model_args_bench.py --mode closure --row 0
"""

import argparse
import ctypes
import sys
import time

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model

import albireo as ab
from albireo.inference import MarginalOrbitModel

N_EP = 50
ORBIT = ab.OrbitParams(period=11.3, t_peri=2.0, ecc=0.3, omega=0.8, k=(45.0, 70.0))
ELL = np.array([0.6, 0.4])

# (wave_lo, wave_hi, native_step): the scripts/m5_scale_bench.py ladder,
# ~31.7k, 74.3k, 135k, 203k model px at dv = 3 km/s.
ROWS = [
    (4000.0, 5495.0, 0.114),
    (4000.0, 8415.0, 0.1331),
    (4000.0, 15452.0, 0.1722),
    (3600.0, 27570.0, 0.2244),
]


def peak_working_set_bytes() -> int:
    """Process-lifetime peak working set (Windows) / peak RSS (POSIX), in bytes."""
    if sys.platform == "win32":

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(pmc.PeakWorkingSetSize)
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru) * (1 if sys.platform == "darwin" else 1024)


def make_model(wave_lo, wave_hi, native_step) -> MarginalOrbitModel:
    grid = ab.LogGrid.from_wavelength_range(wave_lo, wave_hi, dv_kms=3.0)
    rng = np.random.default_rng(7)
    bjd = np.sort(rng.uniform(0.0, 30.0, N_EP))
    comps = [
        ab.synthetic_deviation_spectrum(
            grid, n_lines=120, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=s
        )
        for s in (1, 2)
    ]
    wave_native = np.arange(wave_lo + 3.0, wave_hi - 3.0, native_step)
    spec = ab.InstrumentSpec(wave=wave_native, sigma_v_lsf=7.0, snr=100.0)
    ds, _ = ab.simulate_dataset(
        grid,
        comps,
        bjd=bjd,
        instruments={"inst": spec},
        light_fractions=ELL,
        orbit=ORBIT,
        frame="barycentric",
        seed=3,
    )
    return MarginalOrbitModel(
        grid,
        ds,
        light_fractions=ELL,
        lsf_sigma_v={"inst": 7.0},
        v_rel_max_kms=float(sum(ORBIT.k)) * (1 + ORBIT.ecc) * 1.35,
    )


PRIORS = {
    "period": dist.Normal(ORBIT.period, 0.1),
    "t_conj": dist.Normal(2.0, 1.0),
    "secosw": dist.Uniform(-1.0, 1.0),
    "sesinw": dist.Uniform(-1.0, 1.0),
    "k": dist.Uniform(jnp.array([10.0, 20.0]), jnp.array([80.0, 110.0])),
    "log_tau": dist.Normal(jnp.full(2, np.log(300.0)), 3.0),
    "log_eta": dist.Normal(jnp.full(2, np.log(5.0)), 3.0),
}
INIT = {
    "period": ORBIT.period,
    "t_conj": 2.0,
    "secosw": np.sqrt(ORBIT.ecc) * np.cos(ORBIT.omega),
    "sesinw": np.sqrt(ORBIT.ecc) * np.sin(ORBIT.omega),
    "k": jnp.asarray(ORBIT.k),
    "log_tau": jnp.full(2, np.log(300.0)),
    "log_eta": jnp.full(2, np.log(5.0)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("arg", "closure"), required=True)
    ap.add_argument("--row", type=int, default=0, help="m5 ladder row (0-3)")
    args = ap.parse_args()

    model = make_model(*ROWS[args.row])
    print(
        f"backend: {jax.default_backend()}  |  mode: {args.mode}  |  "
        f"{model.problem.grid.n:,} model px, {N_EP} epochs, SB2"
    )
    info = initialize_model(
        jax.random.PRNGKey(0),
        model.model(PRIORS),
        init_strategy=init_to_value(values=INIT),
        dynamic_args=True,
        model_args=(model.problem,),
        validate_grad=False,
    )
    z = jax.tree.map(jnp.asarray, info.param_info.z)
    pot_gen = info.potential_fn
    ws_before = peak_working_set_bytes()

    if args.mode == "arg":
        fn = jax.jit(lambda zz, pb: jax.value_and_grad(pot_gen(pb))(zz))
        run_args = (z, model.problem)
    else:
        fn = jax.jit(lambda zz: jax.value_and_grad(pot_gen(model.problem))(zz))
        run_args = (z,)

    t0 = time.perf_counter()
    lowered = fn.lower(*run_args)
    t_lower = time.perf_counter() - t0
    t0 = time.perf_counter()
    compiled = lowered.compile()
    t_compile = time.perf_counter() - t0
    ws_compiled = peak_working_set_bytes()

    t0 = time.perf_counter()
    value, grad = compiled(*run_args)
    jax.block_until_ready(grad)
    t_run = time.perf_counter() - t0
    ws_run = peak_working_set_bytes()

    gib = 1024.0**3
    print(f"lower: {t_lower:8.1f} s   compile: {t_compile:8.1f} s   value+grad: {t_run:8.1f} s")
    flat_grad, _ = jax.flatten_util.ravel_pytree(grad)
    print(
        f"potential at init: {float(value):.3f}   |grad|: {float(jnp.linalg.norm(flat_grad)):.3g}"
    )
    print(
        f"peak working set  before lower: {ws_before / gib:6.2f} GiB   "
        f"after compile: {ws_compiled / gib:6.2f} GiB   after run: {ws_run / gib:6.2f} GiB"
    )
    mem = compiled.memory_analysis()
    for field in (
        "temp_size_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "generated_code_size_in_bytes",
        "peak_memory_in_bytes",
    ):
        size = getattr(mem, field, None)
        if size is not None:
            print(f"xla {field.replace('_in_bytes', ''):>24}: {size / gib:8.3f} GiB")


if __name__ == "__main__":
    main()
