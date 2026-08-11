# Benchmarks & validation record

Running record of correctness and performance results at each milestone. Numbers are
from the referenced tests, which reproduce them deterministically (fixed seeds); no
claim here is asserted without a test.

## M2 — fixed-orbit linear solver (2026-08-11)

Machine: Windows 11 laptop, CPU only, float64, un-jitted (JAX 0.11). Performance work
is deliberately deferred (jit, custom band assembly, GPU batching — M3/M5); M2 is a
correctness milestone.

### Exactness

| Check | Result | Test |
|---|---|---|
| marginal log-likelihood vs. dense brute-force marginalization (mixed instruments, masks, response) | rtol 1e-10 | `test_marginal_matches_dense_brute_force` |
| posterior mean & pointwise variance vs. dense | rtol 1e-7 | same |
| block-tridiagonal Cholesky / logdet / solves / Takahashi selected inverse vs. LAPACK dense | rtol 1e-10 | `test_solver.py` |
| comb-probe assembly reproduces the matrix-free operator | exact (validated per run) | `test_probe_assembly_is_exact`, `validate=True` |
| garbage values at masked pixels change nothing | rtol 1e-12 | `test_masked_pixel_values_do_not_affect_anything` |

### Closed-loop recovery (the M2 acceptance gate)

Simulated SB2: SNR 100, 30 epochs (4 in partial eclipse), K = (45, 70) km/s, e = 0.2,
Gaussian LSF 4 km/s, model grid dv = 2.5 km/s (n = 2114 px), 5% chip gaps + cosmics,
truth solved with true nonlinear parameters (`test_closed_loop_recovery_snr100_30_epochs`):

| Metric | Component 1 | Component 2 | Gate |
|---|---|---|---|
| line-region RMS error | 0.52% | 0.73% | < 1% ✓ |
| whitened data residual variance | 0.975 (pooled) | | ~1 ✓ |
| posterior-draw std vs. Takahashi std (median ratio) | ~1.0 | | consistency ✓ |

Wall time for one full marginal-likelihood evaluation (assembly + factorization +
solve + logdet) at this scale: **1.6 s** (CPU, un-jitted, probe assembly = 141
operator applications). The design-target scale (2×10⁵ px, 50 epochs) on GPU is the
M5 gate.

### Nullspace / degeneracy analysis (empirical)

The low-frequency separation theory of [math.md §5.1](math.md) is verified
numerically: the posterior variance of the per-mode "difference" direction matches
`1 / (w l² (J − |g(k)|) + prior(k))` with `g(k) = Σ_j e^{ik ΔΔ_j}` within a factor
~1.5 over a decade of spatial frequency, with the predicted ~5× variance inflation
toward low k (`test_low_frequency_degeneracy_matches_theory`, Hann-windowed modes).

Two lessons from the closed loop, both now baked into tests and docs:

1. **The k = 0 additive indeterminacy is real and quantitative.** Each component's
   mean absorption depression differs; with constant light ratios the data see only
   the light-weighted sum, and the invisible difference (the `ℓ₁Δd₁ + ℓ₂Δd₂ = 0`
   direction) is set entirely by the prior — a ~1.5–2% systematic offset between
   components in our configuration, *for any method*. Four eclipse epochs (per-epoch
   light fractions) make it observable and remove it — the documented breaker
   (math.md §5.2) demonstrated end-to-end.
2. **Sub-LSF scales are honestly unrecoverable.** With a weak smoothness prior the
   posterior std is dominated by a flat ~5–7% contribution from deconvolution modes
   below the instrument resolution. The prior curvature scale must encode the true
   spectral smoothness (here set by hand; ML-II optimization of the marginal
   likelihood is the M3 mechanism). This is reported variance, not hidden error —
   the point of the method.

## M3 — joint NUTS inference (2026-08-11)

Machine: same Windows 11 laptop, CPU only, float64 — now **jit-compiled** through the
whole θ → velocities → shifts → probed marginal likelihood path (math.md §7.1), with
reverse-mode gradients through the comb probing and the scan-based block Cholesky.

### θ-path exactness and gradients

| Check | Result | Test |
|---|---|---|
| jitted `MarginalOrbitModel.log_likelihood(θ)` vs. the M2 `build_problem` route (different but sufficient bandwidths) | rtol 1e-12 | `test_model_loglike_matches_m2_path` |
| `orbit_velocities(θ)` vs. the simulator's Kepler conventions | atol 1e-10 | `test_orbit_velocities_match_simulator` |
| ∂ log p/∂θ (K, t_conj, √e cos ω, log τ) vs. central finite differences | rtol 1e-4 (measured ~1e-7) | `test_gradient_matches_finite_differences` |
| bandwidth guard: epoch-realized shift excess ⇒ non-finite log-density | pass | `test_bandwidth_guard_rejects_out_of_bound_orbits` |

Jitted timings at n = 1191 px × 2 components, 14 epochs, half-bandwidth bound 45
(probe stride 187): marginal evaluation **53 ms**, value+gradient **222 ms**
(vs. 1.6 s un-jitted at the larger M2 config — jit alone buys ~an order of magnitude
on CPU; GPU batching remains M5).

### The MAP → Laplace → NUTS pipeline (the load-bearing engineering result)

On the gate problem (below), NUTS with numpyro's default warmup — unit-scale initial
mass matrix — spends its early transitions at the tree-depth cap (2⁸ leapfrogs ×
~30 ms each), because the posterior scales span ~5 orders of magnitude
(σ_P ~ 9×10⁻⁴ d vs. σ_K ~ 0.06 km/s): **>35 min and still warming up**. The shipped
pipeline instead:

1. **MAP + ML-II** (L-BFGS on numpyro's unconstrained potential, hyperparameters
   included): 40 s, K's to 0.3% before any sampling.
2. **Laplace inverse mass matrix** at the MAP (6-dim dense Hessian, eigenvalue-floored):
   23 s.
3. **NUTS with mass adaptation off** (the supplied matrix would otherwise be
   overwritten by a poor few-sample estimate in the first adaptation window):
   150 warmup + 250 samples in **102 s**, **0 divergences, mean 6.5 leapfrogs** per
   transition.

Total: ~3 min for a converged orbital posterior on a laptop CPU.

### Closed-loop NUTS gate (the M3 acceptance gate)

Simulated SB2: SNR 130, 12 epochs, K = (30, 22) km/s, e = 0.2, ω = 0.7, LSF 7 km/s,
grid dv = 5.5 km/s (n = 490 px), topocentric frame with |v_bary| ≤ 25 km/s, gaps +
cosmics; priors: photometric-quality Normal on P and T_conj, Uniform(−1, 1)² on the
√e-pair (disk-constrained), Uniform on K's spanning ±50%; hyperparameters ML-II
(`test_nuts_gate_k_within_one_percent`):

| Parameter | Posterior mean − truth | Posterior sd | Gate |
|---|---|---|---|
| K₁ | −0.100 km/s (**0.33%**) | 0.058 km/s | < 1% ✓ |
| K₂ | +0.042 km/s (**0.19%**) | 0.091 km/s | < 1% ✓ |
| P | −7.1×10⁻⁵ d | 9.0×10⁻⁴ d | truth in 95% CI ✓ |
| t_conj | +0.0021 d | 0.0023 d | ✓ |
| e | −0.0017 | 0.0011 | ✓ |
| ω | −0.0009 | 0.0072 | ✓ |

0 divergences; truth inside the central 95% interval for every parameter. ML-II
selected τ ≈ (330, 390) and η ≈ (0.5, 2.4) — a weaker continuum anchor than the
hand-tuned M2 value, which is honest: without eclipse epochs the k = 0 anchor *is*
prior-dominated (see below).

### Posterior spectra under θ uncertainty

`posterior_spectra` mixes conditional Gaussian draws over posterior θ samples. In the
gate configuration (constant light fractions — deliberately *no* eclipse breaker) the
light-weighted observable combination ℓ₁d₁ + ℓ₂d₂ is recovered to < 2% RMS in line
cores, while individual components carry the expected k = 0 invisible-direction
scatter (M2 lesson 1, made larger here because ML-II honestly refuses to fake a tight
continuum anchor the data don't constrain). The test asserts exactly this split
(`test_posterior_spectra_from_samples`).

### Injection–coverage study

`scripts/m3_coverage.py` (fixed seed, reproducible): 24 injections with truths drawn
from the sampling priors (disk + bandwidth-guard truncation replicated exactly),
independent random line lists per injection — i.e. the spectral prior is
*deliberately misspecified* and (τ, η) are refit by ML-II each time — NUTS 150+250
per injection via the MAP → Laplace → NUTS pipeline. Total: 101 min on the laptop
CPU (~4.2 min/injection).

| Site | cov68 | cov90 | mean \|z\| | rank-KS |
|---|---|---|---|---|
| period | 0.62 | 0.92 | 0.89 | 0.182 |
| t_conj | 0.62 | 0.96 | 0.82 | 0.151 |
| √e cos ω | 0.54 | 0.92 | 0.98 | 0.187 |
| √e sin ω | 0.75 | 0.92 | 0.87 | 0.166 |
| K₁ | 0.58 | 0.92 | 0.84 | 0.168 |
| K₂ | 0.67 | 0.96 | 0.82 | 0.109 |

Binomial 1σ at n = 24: ±0.095 (cov68), ±0.06 (cov90); N(0,1) expects mean |z| = 0.80;
the KS critical value (α = 0.05) is 0.278. **Every site is consistent with a
calibrated posterior**: central-interval coverage within 1.5σ of nominal (erring
slightly *over*-covered at 90%), |z| never above 2.05 across 144 site-checks, and
truth-rank distributions consistent with uniform. Empirical-Bayes plug-in optimism,
the known trade of fixing hyperparameters at ML-II (math.md §7.3), is not detectable
at this sample size.

Point recovery across the prior: median worst-of-(K₁, K₂) error 0.74%, max 5.0% —
the tail cases are honest wide-posterior draws (K₂ ≈ 5–7 km/s at e ≈ 0.8, where 5%
is ~1.5 posterior sd), not misestimates. Divergences: 24 of 6000 post-warmup
transitions (0.4%), concentrated in the 3 injections with truths against the hard
constraint walls (e = 0.75–0.95 at K₁+K₂ ≈ 70–76 km/s — the e_max and
bandwidth-guard boundaries), where reflecting trajectories are expected to diverge;
interior-of-prior injections show zero. The Laplace mass matrix holds up across the
whole prior: median 6.6 leapfrogs per transition, max 36. The strict MAP `converged`
flag (grad-norm < 10⁻²) fired on only 5/24 — the tolerance is conservative; MAP point
quality (K's to <1% typical) is unaffected.
