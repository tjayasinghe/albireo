"""Disentangle real archival spectra: HR 6819, from ESO Phase-3 FITS to an orbit.

The end-to-end path on *observed* data, as opposed to the simulator of example 01. The
target is HR 6819 (HD 167128, QV Tel), a Be star with a stripped, bloated pre-subdwarf
companion on a 40.3-day orbit — famous for having been proposed as a black-hole host
before interferometry resolved the pair (Rivinius et al. 2020; Bodensteiner et al. 2020;
Frost et al. 2022; Klement et al. 2025). The data are the 51 public FEROS spectra of ESO
programme 073.D-0274(A), the same ones every published analysis of this system used.

    python scripts/download_hr6819.py      # 51 files, ~153 MB, no ESO login needed
    python examples/03_hr6819_real_data.py

What this demonstrates that the simulator cannot
------------------------------------------------
Archival spectra arrive in a state the model does not accept, and the interesting part is
the four decisions in between (all in :mod:`albireo.io` and :mod:`albireo.preprocess`):

* **Continuum.** ESO delivers these with ``CONTNORM = False``: raw merged-echelle ADU,
  whose response falls by a factor of 20 across 3850-4750 A. albireo's model is
  ``1 + sum_i l_i d_i`` around a *unit* continuum, and its response term is fixed at
  build time rather than inferred, so the normalization done here is the one the fit gets.
* **Noise.** The ``ERR`` column of these files is entirely ``NaN`` — the header says so
  outright. Inverse variances are estimated from the spectra themselves.
* **Frame and time.** ``SPECSYS = 'BARYCENT'`` with the applied correction in
  ``ESO DRS BARYCORR``, and a mid-exposure time that has to be put on the barycentre.
* **Per-exposure wavelength grids.** The pipeline shifts before resampling, so the 51
  spectra sit on 28 distinct grids. :func:`albireo.preprocess.share_wavelength_grid`
  relabels them onto one, which is a hundredth of a pixel of relabelling and the
  difference between 28 operator groups and 1.

The window
----------
4380-4600 A: He I 4388, He I 4471, Mg II 4481, Si III 4552/4568/4575. Photospheric lines
present in both a sharp-lined stripped star and a broad-lined Be star; no Balmer core, no
disc emission (the Be star's disc emission is variable, and albireo assumes each component
has *one* spectrum), and no telluric band within 1200 A.

What to compare against, and what not to
----------------------------------------
Klement et al. (2025), Table 3, combined eccentric solution: P = 40.3261 +/- 0.0013 d,
e = 0.0289 +/- 0.0058, K_pre-sd = 61.15 +/- 0.88 km/s, K_Be = 3.90 +/- 0.27 km/s.
Those are the numbers this script scores itself against.

The light ratio is **not** one of them. With constant light fractions the likelihood sees
only the products ``l_i * d_i`` (``docs/math.md`` §5.2), so for a non-eclipsing system it
is an input, not a result: every recovered line depth scales as ``1 / l_i``. The value
used below is the optical estimate from Bodensteiner et al. (2020); GRAVITY's
f = 0.439 +/- 0.013 is a *K-band* flux ratio and does not transfer to 4400 A unchanged.

Environment
-----------
ALBIREO_HR6819_DATA
    Directory holding the FITS files. Default ``data/hr6819`` beside the repository root.
ALBIREO_HR6819_NUTS
    Set to ``1`` to run the (slow) Laplace + NUTS stage after the MAP.

Usage
-----
    python examples/03_hr6819_real_data.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

import albireo as ab
from albireo.forward import data_residual_zscores
from albireo.preprocess import share_wavelength_grid

# --- configuration ----------------------------------------------------------------
REGION = (4380.0, 4600.0)
DV_KMS = 1.5
RESOLVING_POWER = 48_000.0  # FEROS
LSF_SIGMA = ab.C_KMS / (RESOLVING_POWER * 2.0 * np.sqrt(2.0 * np.log(2.0)))  # 2.652 km/s
V_REL_MAX = 90.0  # bounds (K_1 + K_2)(1 + e) with headroom; sets the solver bandwidth

# Component 0 = the stripped pre-subdwarf (sharp-lined, K ~ 61 km/s).
# Component 1 = the Be star (rotationally broadened, K ~ 4 km/s).
# The ordering is load-bearing: `k`, `light_fractions` and the recovered spectra are all
# indexed by it, and nothing downstream can detect a swap.
LIGHT_FRACTIONS = (0.45, 0.55)  # optical, Bodensteiner et al. 2020 — an input (see above)

# Klement et al. 2025, A&A, Table 3 (combined eccentric solution, astrometry + both RVs).
LITERATURE = {
    "period": (40.3261, 0.0013),
    "ecc": (0.0289, 0.0058),
    "k_presd": (61.15, 0.88),
    "k_be": (3.90, 0.27),
}


def main() -> int:
    data_dir = pathlib.Path(
        os.environ.get("ALBIREO_HR6819_DATA")
        or pathlib.Path(__file__).resolve().parent.parent / "data" / "hr6819"
    )
    if not sorted(data_dir.glob("*.fits")):
        print(
            f"No FITS files in {data_dir}.\n"
            "Fetch them first (51 files, ~153 MB, public — no ESO account needed):\n"
            "    python scripts/download_hr6819.py",
            file=sys.stderr,
        )
        return 0  # not a failure: the data are deliberately not in the repository

    # --- 1. FITS -> Dataset -------------------------------------------------------
    t0 = time.time()
    dataset = ab.read_dataset(
        str(data_dir / "*.fits"),
        instrument="FEROS",
        region=REGION,
        region_pad_angstrom=60.0,  # fit the continuum on a wider slice, then trim
        smooth_angstrom=120.0,
    )
    # One rebin operator instead of 28. Exact to ~0.007 km/s; raises if that is untrue.
    dataset = ab.Dataset(share_wavelength_grid(list(dataset)), frame=dataset.frame)
    print(f"[{time.time() - t0:5.1f}s] ingest")
    print(dataset.summary())

    # --- 2. Model grid ------------------------------------------------------------
    # Wider than the data by the largest shift *plus* the LSF kernel radius: inside that
    # margin the shift and convolution operators zero-fill, and the pixels there would be
    # modelled with missing flux at full weight.
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
    print(
        f"model grid {grid.n} px ({grid.wave[0]:.1f}-{grid.wave[-1]:.1f} A, "
        f"dv={grid.dv_kms:.2f} km/s), {len(model.problem.groups)} operator group(s), "
        f"half-bandwidth {model.half_bandwidth}"
    )

    # --- 3. Locate the orbit by scanning, before optimizing anything --------------
    # The marginal likelihood is sharply multimodal in conjunction phase; L-BFGS started
    # in the wrong trough converges confidently to the wrong answer. A 1-D scan over one
    # period costs a minute and removes the question.
    log_tau0 = np.log(np.array([1.0e3, 1.0e8]))  # sharp component / rotationally broad one
    log_eta0 = np.log(np.array([1.0e2, 1.0e2]))

    def theta(period, t_conj, k):
        return {
            "period": jnp.asarray(period),
            "t_conj": jnp.asarray(t_conj),
            "secosw": jnp.asarray(1e-3),
            "sesinw": jnp.asarray(1e-3),
            "k": jnp.asarray(k),
            "log_tau": jnp.asarray(log_tau0),
            "log_eta": jnp.asarray(log_eta0),
        }

    p_lit = LITERATURE["period"][0]
    k_lit = [LITERATURE["k_presd"][0], LITERATURE["k_be"][0]]
    t0 = time.time()
    trial = np.min(dataset.bjd) + np.linspace(0.0, p_lit, 41, endpoint=False)
    scan = np.array([float(model.log_likelihood(theta(p_lit, t, k_lit))) for t in trial])
    t_conj0 = float(trial[int(np.argmax(scan))])
    print(
        f"[{time.time() - t0:5.1f}s] conjunction scan: t_conj = {t_conj0:.4f}, "
        f"{scan.max() - scan.min():.3g} nats between best and worst phase"
    )

    # --- 4. MAP over the orbit and the spectral hyperparameters (ML-II) ----------
    priors = {
        "period": dist.Normal(p_lit, 0.5),
        "t_conj": dist.Normal(t_conj0, 2.0),
        "secosw": dist.Uniform(-1.0, 1.0),
        "sesinw": dist.Uniform(-1.0, 1.0),
        "k": dist.Uniform(jnp.array([20.0, 0.1]), jnp.array([120.0, 40.0])),
        "log_tau": dist.Normal(jnp.asarray(log_tau0), 6.0),
        "log_eta": dist.Normal(jnp.asarray(log_eta0), 6.0),
    }
    init = {
        "period": p_lit,
        "t_conj": t_conj0,
        "secosw": 0.05,
        "sesinw": 0.05,
        "k": jnp.array(k_lit),
        "log_tau": jnp.asarray(log_tau0),
        "log_eta": jnp.asarray(log_eta0),
    }
    assert set(init) == set(priors), "run_map randomizes any site missing from init"

    t0 = time.time()
    fit = ab.run_map(model.model(priors), init=init, max_steps=200, tol=1.0)
    print(f"[{time.time() - t0:5.1f}s] MAP: {fit.num_steps} steps, |grad| = {fit.grad_norm:.3g}")

    k_map = np.asarray(fit.params["k"])
    assert k_map[0] > k_map[1], "component ordering swapped — see LIGHT_FRACTIONS above"
    print("\n  parameter        albireo        Klement et al. 2025")
    for name, key, value in (
        ("period [d]", "period", float(fit.params["period"])),
        ("ecc", "ecc", float(fit.params["ecc"])),
        ("K_pre-sd [km/s]", "k_presd", float(k_map[0])),
        ("K_Be [km/s]", "k_be", float(k_map[1])),
    ):
        ref, err = LITERATURE[key]
        print(
            f"  {name:16s} {value:11.4f}    {ref:9.4f} +/- {err:.4f}"
            f"   ({(value - ref) / err:+.1f} sigma)"
        )

    # --- 5. Is the estimated noise calibrated? ------------------------------------
    # The inverse variances were estimated from the spectra, so this check is not
    # optional: whitened residuals must have unit scatter or every quoted uncertainty is
    # wrong by the same factor. Measure it *here*, with no jitter site in theta, because
    # a fitted jitter drives this number to 1 by construction and then it tells you
    # nothing.
    theta_map = {
        k: jnp.asarray(v)
        for k, v in fit.params.items()
        if k in ("period", "t_conj", "secosw", "sesinw", "k", "log_tau", "log_eta")
    }
    result = model.marginal(theta_map)
    z = data_residual_zscores(model.problem_at(theta_map), result.d_hat)
    print(f"\n  whitened residuals: mean {z.mean():+.3f}, sd {z.std():.3f} (target 1.000)")

    # The D15/D31 answer to sd != 1 is a `log_jitter` site: add
    #     "log_jitter": dist.Normal(jnp.zeros(len(ds)), 2.0).to_event(1)
    # to `priors` and refit. Do read docs/benchmarks.md before believing the result on
    # *this* dataset. Fitting one here does what it says — the per-epoch factors come out
    # spanning 1.1 to 3.1, and the residuals whiten — but it also moves the period by
    # ~175x the no-jitter formal error, because downweighting the noisiest exposures
    # changes which of them carry the period leverage. A noise model that is honest about
    # its scale is still a diagonal noise model, and these residuals are correlated.
    #
    # The complementary D33 handle is a per-epoch continuum: add
    #     "response": dist.Normal(jnp.zeros((len(dataset), 3)), 0.02)
    # to `priors` (init at zeros) and refit. Measured on this dataset
    # (scripts/hr6819_response_run.py): it absorbs ~4,000 nats of real epoch-structured
    # signal with coefficients of a few per mil — and moves nothing else (period by
    # ~0.4 formal sigma, K by <0.001 km/s, residual sd from 1.674 to 1.668). Whatever
    # biases this orbit, it is not the continuum.
    #
    # D34 closes the loop on that correlation: build the model with
    #     MarginalOrbitModel(..., ar1=True)
    # and add "ar1_phi": dist.Uniform(-0.9, 0.9) alongside the jitters to model the
    # correlation itself — an AR(1) chain per epoch, on the probe assembly path at ~15x
    # the per-step cost (the fast band assembly assumes diagonal noise). Measured on this
    # dataset (scripts/hr6819_ar1_run.py): phi comes out at 0.7-0.8, the residuals
    # whiten in scale *and* lag-1 autocorrelation, the jitters collapse to a near-uniform
    # 1.3-1.9, the D31 period relocation does not recur, and the two analysis windows
    # land 3x closer to each other in period. The ~0.044 d offset from the published
    # period survives every noise model tried; see docs/benchmarks.md for the ledger.

    if os.environ.get("ALBIREO_HR6819_NUTS") != "1":
        print("\nSet ALBIREO_HR6819_NUTS=1 to continue into Laplace + NUTS.")
        return 0

    # --- 6. Laplace mass matrix, then NUTS ----------------------------------------
    hyper = {s: fit.params[s] for s in ("log_tau", "log_eta")}
    nuts_model = model.model({s: d for s, d in priors.items() if s not in hyper}, fixed=hyper)
    t0 = time.time()
    mass = ab.laplace_inverse_mass(nuts_model, fit.params)
    print(f"[{time.time() - t0:5.1f}s] Laplace mass matrix")
    t0 = time.time()
    mcmc = ab.run_nuts(
        nuts_model,
        rng_key=jax.random.PRNGKey(0),
        init=fit.params,
        inverse_mass_matrix=mass,
        num_warmup=300,
        num_samples=300,
        num_chains=2,
    )
    print(f"[{time.time() - t0:5.1f}s] NUTS")
    mcmc.print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
