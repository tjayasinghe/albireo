"""Probe vs band assembly for the correlated (AR(1)) marginal (D35).

D34 ran HR 6819's AR(1) fits on the probe path (the D28 band sandwich assumed
diagonal weights) at ~15x the per-step cost of the D33 fits, with plain reverse-mode
gradients through 2p + 1 comb-probe operator applications. D35 extends the band
assembly to the tridiagonal chain precision via static link pair tables, restoring
the D28/D29 fast path (closed-form solve VJP included) for correlated problems.

This script measures ``value`` and ``value+grad`` of the correlated marginal, the
work of one L-BFGS step with gradients taken in velocities, phi and the jitter, at
HR-window scale (51 epochs, ~9.8k model px, ~374k native px, the HR runs'
half-bandwidth) on one assembly path per invocation: the peak-working-set counter is
per-process and monotone, so the two paths must not share a process.

Run:  python scripts/d35_ar1_band_bench.py --assembly probe
      python scripts/d35_ar1_band_bench.py --assembly band
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import albireo as ab
from albireo.forward import build_problem, with_ar1, with_jitter, with_velocities
from albireo.likelihood import marginal_loglikelihood
from albireo.priors import SmoothnessPrior

# HR 6819 window A's shape: 51 epochs, 4380-4600 A, dv = 1.5 km/s (~9.8k model px),
# native step ~2 km/s (~7.3k px/epoch), FEROS-like LSF, |v_rel| <= 90 km/s.
N_EP = 51
WAVE_LO, WAVE_HI = 4380.0, 4600.0
DV_KMS = 1.5
NATIVE_STEP = 0.03
LSF_SIGMA_V = 2.652
V_REL_MAX = 90.0
PHI0, ALPHA0 = 0.5, 1.4
ELL = np.array([0.55, 0.45])


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


def make_problem():
    grid = ab.LogGrid.from_wavelength_range(WAVE_LO, WAVE_HI, dv_kms=DV_KMS)
    rng = np.random.default_rng(7)
    comps = [
        ab.synthetic_deviation_spectrum(
            grid, n_lines=60, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=s
        )
        for s in (1, 2)
    ]
    vel = np.stack([rng.uniform(-0.45 * V_REL_MAX, 0.45 * V_REL_MAX, N_EP) for _ in range(2)])
    wave_native = np.arange(WAVE_LO + 2.0, WAVE_HI - 2.0, NATIVE_STEP)
    spec = ab.InstrumentSpec(wave=wave_native, sigma_v_lsf=LSF_SIGMA_V, snr=100.0)
    ds, _ = ab.simulate_dataset(
        grid,
        comps,
        bjd=np.arange(float(N_EP)),
        velocities=vel,
        light_fractions=ELL,
        instruments={"inst": spec},
        epoch_instruments=["inst"] * N_EP,
        v_bary=np.zeros(N_EP),
        cosmic_fraction=0.005,  # realized multi-pixel gap links, as in real data
        ar1_phi=PHI0,
        seed=11,
    )
    problem = build_problem(
        grid, ds, velocities=vel, light_fractions=ELL, lsf_sigma_v={"inst": LSF_SIGMA_V}
    )
    return problem, jnp.asarray(vel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembly", choices=("probe", "band"), required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    problem, vel = make_problem()
    t_build = time.perf_counter() - t0
    hb = problem.half_bandwidth_bound(V_REL_MAX) + problem.ar_bandwidth_extra
    n_native = sum(int(np.asarray(g.w).size) for g in problem.groups)
    print(
        f"backend: {jax.default_backend()}  |  assembly: {args.assembly}  |  "
        f"{problem.grid.n:,} model px, {N_EP} epochs, {n_native:,} native px, "
        f"half-bandwidth {hb} (ar extra {problem.ar_bandwidth_extra})  |  "
        f"build {t_build:.1f} s"
    )
    prior = SmoothnessPrior(tau=[300.0, 300.0], eta=[5.0, 5.0])

    def loglike(pb, v, phi, log_alpha):
        q = with_ar1(with_jitter(with_velocities(pb, v), jnp.exp(log_alpha)), phi)
        return marginal_loglikelihood(
            q, prior, half_bandwidth=hb, assembly=args.assembly
        ).log_likelihood

    theta = (vel, jnp.asarray(PHI0), jnp.asarray(np.log(ALPHA0)))
    f_eval = jax.jit(loglike)
    f_grad = jax.jit(
        lambda pb, v, p, a: jax.value_and_grad(loglike, argnums=(1, 2, 3))(pb, v, p, a)
    )
    gib = 1024.0**3

    t0 = time.perf_counter()
    value = f_eval(problem, *theta)
    jax.block_until_ready(value)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    value = f_eval(problem, *theta)
    jax.block_until_ready(value)
    t_eval = time.perf_counter() - t0
    ws_eval = peak_working_set_bytes()
    print(
        f"eval:      first {t_first:7.1f} s   steady {t_eval:7.2f} s   "
        f"loglike {float(value):.3f}   peak WS {ws_eval / gib:6.2f} GiB"
    )

    t0 = time.perf_counter()
    _, grads = f_grad(problem, *theta)
    jax.block_until_ready(grads)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    value, grads = f_grad(problem, *theta)
    jax.block_until_ready(grads)
    t_grad = time.perf_counter() - t0
    ws_grad = peak_working_set_bytes()
    g_vel, g_phi, g_alpha = grads
    print(
        f"val+grad:  first {t_first:7.1f} s   steady {t_grad:7.2f} s   "
        f"d/dphi {float(g_phi):+.4g}   d/dlog_alpha {float(g_alpha):+.4g}   "
        f"|d/dvel| {float(jnp.linalg.norm(g_vel)):.4g}   peak WS {ws_grad / gib:6.2f} GiB"
    )


if __name__ == "__main__":
    main()
