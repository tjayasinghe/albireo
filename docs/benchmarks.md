# Benchmarks & validation record

Running record of correctness and performance results at each milestone. Numbers are
from the referenced tests, which reproduce them deterministically (fixed seeds); no
claim here is asserted without a test.

## The machine

Every table in this file was measured on one computer: an AMD Ryzen 9 9950X3D desktop,
16 cores / 32 threads, 32 GB, Windows 11 Pro build 26200, CPU only, float64. Tables
written before the D50 re-run label it a "Windows 11 laptop"; it is the same machine.

Two statements in later sections assumed two machines and do not hold:

* The 1.46× "residual clean-machine gap" between shift-and-add's recorded 0.018 s and the
  0.0263 s measured under the D50 protocol is not a hardware difference. Its cause cannot
  be recovered: the earlier number's software stack, warmup and dose of heap contamination
  are unrecorded, and that contamination slows a run, so it does not explain a number below
  the clean one. 0.018 s is an unreproduced measurement.
* "fd3 moved 12% across the hardware change" describes no hardware change. D50 attributed
  part of that motion to OpenBLAS oversubscription; the remainder is unexplained.

The accuracy record does not depend on this: those results are deterministic and were
reproduced exactly in the re-run, so accuracy is the durable comparison here and the walls
are not.

## The fixed-orbit linear solver (2026-08-11)

Machine: the desktop above, CPU only, float64, un-jitted (JAX 0.11). This is a correctness
milestone; performance work (jit, custom band assembly, GPU batching) comes later.

### Exactness

| Check | Result | Test |
|---|---|---|
| marginal log-likelihood vs. dense brute-force marginalization (mixed instruments, masks, response) | rtol 1e-10 | `test_marginal_matches_dense_brute_force` |
| posterior mean & pointwise variance vs. dense | rtol 1e-7 | same |
| block-tridiagonal Cholesky / logdet / solves / Takahashi selected inverse vs. LAPACK dense | rtol 1e-10 | `test_solver.py` |
| comb-probe assembly reproduces the matrix-free operator | exact (validated per run) | `test_probe_assembly_is_exact`, `validate=True` |
| garbage values at masked pixels change nothing | rtol 1e-12 | `test_masked_pixel_values_do_not_affect_anything` |

### Closed-loop recovery (the acceptance gate)

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
scale gate below.

### Nullspace / degeneracy analysis (empirical)

The low-frequency separation theory of [math.md §5.1](math.md) is verified
numerically: the posterior variance of the per-mode "difference" direction matches
`1 / (w l² (J − |g(k)|) + prior(k))` with `g(k) = Σ_j e^{ik ΔΔ_j}` within a factor
~1.5 over a decade of spatial frequency, with the predicted ~5× variance inflation
toward low k (`test_low_frequency_degeneracy_matches_theory`, Hann-windowed modes).

Two lessons from the closed loop, both covered by tests and documentation:

1. **The k = 0 additive indeterminacy is real and quantitative.** Each component's
   mean absorption depression differs; with constant light ratios the data see only
   the light-weighted sum, and the unobservable difference (the `ℓ₁Δd₁ + ℓ₂Δd₂ = 0`
   direction) is set entirely by the prior: a ~1.5–2% systematic offset between
   components in this configuration, for any method. Four eclipse epochs (per-epoch
   light fractions) make it observable and remove it, demonstrating the breaker of
   math.md §5.2 end to end.
2. **Sub-LSF scales are unrecoverable.** With a weak smoothness prior the posterior
   std is dominated by a flat ~5–7% contribution from deconvolution modes below the
   instrument resolution. The prior curvature scale must encode the true spectral
   smoothness (here set by hand; ML-II arrives with joint inference, below). This
   contribution appears as reported variance, not as unreported error.

## Joint NUTS inference (2026-08-11)

Machine: the same desktop, CPU only, float64, now **jit-compiled** through the
whole θ → velocities → shifts → probed marginal likelihood path (math.md §7.1), with
reverse-mode gradients through the comb probing and the scan-based block Cholesky.

### θ-path exactness and gradients

| Check | Result | Test |
|---|---|---|
| jitted `MarginalOrbitModel.log_likelihood(θ)` vs. the fixed-orbit `build_problem` route (different but sufficient bandwidths) | rtol 1e-12 | `test_model_loglike_matches_m2_path` |
| `orbit_velocities(θ)` vs. the simulator's Kepler conventions | atol 1e-10 | `test_orbit_velocities_match_simulator` |
| ∂ log p/∂θ (K, t_conj, √e cos ω, log τ) vs. central finite differences | rtol 1e-4 (measured ~1e-7) | `test_gradient_matches_finite_differences` |
| bandwidth guard: epoch-realized shift excess ⇒ non-finite log-density | pass | `test_bandwidth_guard_rejects_out_of_bound_orbits` |
| chi-square sign guard (D63): a negative quadratic form ⇒ log-likelihood −∞ with a zero gradient; a healthy value and its gradient bit-identical | pass | `test_negative_chi_square_is_rejected_with_a_finite_gradient`, `test_sign_guard_leaves_a_healthy_evaluation_and_its_gradient_untouched` |
| smoothness bound (D63): log τ − log η above 30 ⇒ non-finite log-density | pass | `test_smoothness_bound_rejects_extreme_stiffness_ratios` |

Jitted timings at n = 1191 px × 2 components, 14 epochs, half-bandwidth bound 45
(probe stride 187): marginal evaluation **53 ms**, value+gradient **222 ms** (vs. 1.6
s un-jitted at the larger fixed-orbit config; jit alone gains about an order of magnitude
on CPU).

### The MAP → Laplace → NUTS pipeline (the load-bearing engineering result)

On the gate problem (below), NUTS with numpyro's default warmup (unit-scale initial
mass matrix) spends its early transitions at the tree-depth cap (2⁸ leapfrogs × ~30 ms
each), because the posterior scales span ~5 orders of magnitude (σ_P ~ 9×10⁻⁴ d vs.
σ_K ~ 0.06 km/s): **>35 min** and still in warmup. The shipped pipeline instead:

1. **MAP + ML-II** (L-BFGS on numpyro's unconstrained potential, hyperparameters
   included): 40 s, K's to 0.3% before any sampling.
2. **Laplace inverse mass matrix** at the MAP (6-dim dense Hessian, eigenvalue-floored):
   23 s.
3. **NUTS with mass adaptation off** (the supplied matrix would otherwise be
   overwritten by a poor few-sample estimate in the first adaptation window):
   150 warmup + 250 samples in **102 s**, **0 divergences, mean 6.5 leapfrogs** per
   transition.

Total: ~3 min for a converged orbital posterior on one desktop CPU.

### Closed-loop NUTS gate (the acceptance gate)

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
selected τ ≈ (330, 390) and η ≈ (0.5, 2.4), a weaker continuum anchor than the
hand-tuned fixed-orbit value, as expected: without eclipse epochs the k = 0 anchor is
prior-dominated (see below).

### Posterior spectra under θ uncertainty

`posterior_spectra` mixes conditional Gaussian draws over posterior θ samples. In the
gate configuration (constant light fractions, with no eclipse breaker) the
light-weighted observable combination ℓ₁d₁ + ℓ₂d₂ is recovered to < 2% RMS in line
cores, while individual components carry the expected k = 0 unobservable-direction
scatter (the fixed-orbit stage's first lesson, larger here because ML-II does not impose a
tight continuum anchor that the data do not constrain). The test asserts this split
(`test_posterior_spectra_from_samples`).

### Injection–coverage study

`scripts/m3_coverage.py` (fixed seed, reproducible): 24 injections with truths drawn
from the sampling priors (disk + bandwidth-guard truncation replicated exactly),
independent random line lists per injection (so the spectral prior is misspecified by
construction and (τ, η) are refit by ML-II each time), and NUTS 150+250 per injection
via the MAP → Laplace → NUTS pipeline. Total: 101 min on the desktop CPU (~4.2
min/injection).

| Site | cov68 | cov90 | mean \|z\| | rank-KS |
|---|---|---|---|---|
| period | 0.62 | 0.92 | 0.89 | 0.182 |
| t_conj | 0.62 | 0.96 | 0.82 | 0.151 |
| √e cos ω | 0.54 | 0.92 | 0.98 | 0.187 |
| √e sin ω | 0.75 | 0.92 | 0.87 | 0.166 |
| K₁ | 0.58 | 0.92 | 0.84 | 0.168 |
| K₂ | 0.67 | 0.96 | 0.82 | 0.109 |

Binomial 1σ at n = 24: ±0.095 (cov68), ±0.06 (cov90); N(0,1) expects mean |z| = 0.80;
the KS critical value (α = 0.05) is 0.278. Every site is consistent with a calibrated
posterior: central-interval coverage within 1.5σ of nominal (slightly over-covered at
90%), |z| never above 2.05 across 144 site-checks, and truth-rank distributions
consistent with uniform. Empirical-Bayes plug-in optimism, the known trade of fixing
hyperparameters at ML-II (math.md §7.3), is not detectable at this sample size.

Point recovery across the prior: median worst-of-(K₁, K₂) error 0.74%, max 5.0%; the
tail cases are wide-posterior draws (K₂ ≈ 5–7 km/s at e ≈ 0.8, where 5% is ~1.5
posterior sd), not misestimates. Divergences: 24 of 6000 post-warmup transitions
(0.4%), concentrated in the 3 injections with truths against the hard constraint walls
(e = 0.75–0.95 at K₁+K₂ ≈ 70–76 km/s, the e_max and bandwidth-guard boundaries), where
reflecting trajectories are expected to diverge; interior-of-prior injections show
zero. The Laplace mass matrix holds across the whole prior: median 6.6 leapfrogs per
transition, max 36. The strict MAP `converged` flag (grad-norm < 10⁻²) fired on only
5/24; the tolerance is conservative, and MAP point quality (K's to <1% typical) is
unaffected.

## Realism: tellurics, SB3, per-epoch light, LSF widths, K₂ scan (2026-08-11)

Machine: the same desktop, CPU only, float64, jitted θ-path throughout. Every
number below is asserted (usually with margin) by a deterministic closed-loop test.

### θ-path exactness (new sites)

| Check | Result | Test |
|---|---|---|
| `with_light_fractions` vs. fresh `build_problem` (± telluric, constant & per-epoch) | rtol 1e-14 | `test_forward.py`, `test_realism.py` |
| `with_lsf` vs. fresh build at matched kernel radius | identical kernels; loglike rtol 1e-12 | same |
| `with_lsf` at narrower width vs. fresh (smaller-radius) build | truncation-tail level (~1e-5) | `test_with_lsf_narrower_width_agrees_to_truncation` |
| SB3 `orbit_velocities` vs. hand-composed nested Keplerians | atol 1e-12 | `test_sb3_velocities_match_hand_composed` |
| ∂ log p / ∂(light, lsf_sigma) vs. finite differences | rtol 1e-4 | `test_gradients_light_lsf_match_finite_differences` |
| LSF-bound / outer-disk guards reject | non-finite log-density | `test_guards_reject_wide_lsf_and_outer_disk` |

### Closed loops (the acceptance gate — one per feature)

All at the joint-NUTS gate scale (n = 490 px, 10–14 epochs, SNR 110–150, topocentric,
gaps + cosmics), MAP/ML-II point recovery:

| Feature | Recovery | Wall time |
|---|---|---|
| **Tellurics** (3rd component, topocentric) | K₁ 0.07%, K₂ 0.41%; telluric spectrum RMS 0.047, corr 0.989 (135 core px) | 57 s |
| **SB3** (hierarchical, 14 epochs over 2.2 P_out) | K₁ 0.10%, K₂ −0.08%, K_AB 0.39%, K_C 0.50%; e_in ±0.004, e_out ±0.008, P_out rel 9e-5; observable combination RMS 0.007 | 76 s |
| **Per-epoch light** (12 epochs, 3 in partial eclipse, ℓ inferred with flat Dirichlet) | ℓ₁ per-epoch rms 0.0028 (eclipse epochs ±0.004); K's −0.77% / −0.14%; **each component individually recovered** (core RMS 0.010 / 0.012) | 48 s |
| **LSF widths** (2 instruments, reference pinned) | σ_B −0.67% (asserted <3%); K's <0.2% | 36 s |
| **K₂ scan** (ℓ₂ = 0.1 companion, 15-point grid) | peak exactly at injected K₂ = 38; contrast > 4000 in D over the scan edges; companion line pattern corr 0.977, offset-removed RMS 0.05 | ~3 s/scan |

In the per-epoch-light row, the k = 0 additive indeterminacy that capped constant-light
component recovery at the ~0.1 level is broken by eclipse epochs whose light fractions are
inferred rather than supplied: component spectra are recovered individually at the 0.01
level, with ℓ(t) to 0.003.

### Negative results worth as much as the positive ones

1. **Absolute LSF widths are unidentifiable in a template-free model.** ML-II with
   both instruments free inflates σ by +35% / +13% (trading against intrinsic line
   widths; K's unaffected at <0.2%). With one reference instrument pinned, the other
   width recovers to <1%. Policy: pin one instrument, the same declared-anchor convention
   as the light ratio.
2. **The K₂-scan null is negative.** On a companion-free dataset, D(K₂) ∈ [−544, −465]
   over the whole grid: the marginal likelihood's Occam term penalizes the extra
   marginalized component and no coherent signal offsets the penalty. Detection
   thresholds remain empirically calibrated (math.md §6), but the baseline is
   repulsive, not neutral.
3. **The faint companion's envelope is prior-dominated.** At ℓ₂ = 0.1 the §5.1
   low-frequency degeneracy is amplified by ℓ₁/ℓ₂ = 9: the recovered companion carries
   a ~+0.19 constant offset (its mean blanketing is absorbed by the primary) while the
   line pattern is intact. Recorded in math.md §6; line-pattern quantities are the
   deliverable of the scan.
4. **A second exact k = 0 mode appears with tellurics** (telluric constant vs.
   common stellar constant, since Σℓ = 1): measured offsets +0.030 / −0.029,
   cancelling to 0.001 in the light-weighted sum. Ledger row added (§5.4).
5. **Injected tellurics must be representable on the model grid**: sub-pixel telluric
   lines behind a 7 km/s LSF are resolution-limited (recovery ceiling RMS ≈ 0.2
   against the raw truth at any SNR or epoch count). This is a simulator-configuration
   constraint, not a solver limitation.

## Scale, benchmarks, release readiness (2026-08-11)

Machine: the same desktop (32 GB RAM), CPU, float64. The scale gate ("2×10⁵
px / 50 epochs samples in minutes on one GPU") was projected from these CPU
measurements. On a real GPU (below, 2026-08-14) the CUDA path works and scales as
predicted, but the gate does not close on the consumer card available here: 16 GB of
device memory against a gradient that needs 18.24 GB, and fp64 at 1/50 of fp32. The gate
stays open with a specific hardware requirement.

### Three scale pathologies, found and fixed

The first attempt to evaluate the marginal likelihood at survey scale
(n = 31,750 px × 2 components, 50 epochs, half-bandwidth 513) failed with an
**82 GB** allocation. Three distinct causes, each now fixed and regression-guarded
by the exactness suite (identical log-likelihoods to 12 digits before/after):

1. **Closure-captured data arrays.** `jax.jit` of a method closing over the problem
   embedded every data array as an XLA constant, and constant folding of the
   θ-independent graph exploded. Fix: `Problem`/`EpochGroup` are registered pytrees,
   and the jitted marginal takes the problem as an argument (runtime parameter). At
   the failing size this fix alone left the planned temporaries unchanged.
2. **Unrolled probe batches.** Comb probing applied 2p+1 = 1027 matvecs in 16 unrolled
   `vmap` chunks with no data dependence between them; XLA scheduled them with
   overlapping live ranges, and its own buffer analysis planned **79.6 GB** of
   temporaries. Fix: probing is a sequential `lax.scan` over batches (combs generated
   from offsets inside the body), which forces buffer reuse: planned temporaries
   dropped to **2.1 GB** (38×) and compile time fell ~5× (one scan body instead of 16
   unrolled copies). The scatter-based block assembly (O(n·p) index maps, ~8 GB at
   design scale) was likewise replaced by per-block gathers under `lax.map` (O(B²)
   transients).
3. **Scan stores the backward pass.** Reverse-mode through the probe scan saved every
   batch's forward intermediates: gradient memory grew back to the unrolled total,
   **469 GB requested** at the design target. Fix: `jax.checkpoint` on the batch body;
   the backward sweep recomputes each batch (~1.5-2× backward probing cost) and
   gradient memory stays at the outputs array plus one batch.

Unconditional remat + serialized small batches cost up to 2× NUTS wall time at small scale
(tutorial run 72 → 141 s; gate test 156 s vs ~102 s at the joint-NUTS baseline;
bit-identical posteriors). XLA's parallel execution of independent unrolled probe batches,
which overlapped 80 GB of live buffers at scale, is a multi-core speedup at small scale.
Batch size and remat are therefore size-adaptive on the probe-output footprint (64 MB
threshold): small problems run all probes as one parallel batch without remat (gate test
back to 112 s), large problems get the sequential remat scan. The prior factor has its own
small block size instead of the posterior's (its bandwidth is 2 per component; factorizing
it at block 513 doubled Cholesky cost for a determinant of negligible cost).

### Design-target ladder (CPU, jitted, fixed bandwidth p = 513, 50 epochs, SB2)

| n (model px) | native px/epoch | eval | ∇ eval |
|---|---|---|---|
| 31,750 | 13,134 | 24.7 s | 102.7 s |
| 74,331 | 33,134 | 52.5 s | 249.9 s |
| 135,063 | 66,467 | 96.9 s | 466.0 s |
| **203,497 (design target)** | 106,467 | **149.7 s** | **729.9 s** |

Both scale linearly in n at fixed bandwidth (~0.75 ms/px eval, ~3.4 ms/px gradient),
as the O(n·p²) flop count predicts; log-likelihood values are bit-identical across all
three solver revisions. Peak memory stays within the machine's 32 GB at every size. A
single design-target marginal evaluation (disentangled spectra at a given orbit) takes
**2.5 min on one desktop CPU**; posterior sampling at this scale is deferred to the GPU
(below).

### GPU projection (stated as projection, not measurement)

At the design target a NUTS run needs ~2,600 gradient evaluations (150+250 transitions
× 6.5 mean leapfrogs, the pipeline numbers measured above). On this CPU that is 2600 ×
12.2 min ≈ three weeks, hence the GPU target. The dominant costs (batched probe
matvecs; 800-step scanned Cholesky of 513² blocks) are dense, batched, and fp64; on a
single A100-class device the same
graph is expected to run the gradient in ~1-3 s (probe batches become large GEMM-like
work, the block Cholesky ~0.1 s of batched `potrf`/`trsm`), putting a converged
posterior at 1-2 hours for the widest-bandwidth massive-star config, and tens of
minutes at moderate bandwidths (p ~ 200: flops drop ~6×). The `probe_chunk` knob
(raise on GPU) and `remat=False` (80 GB HBM fits the stored backward) are the tuning
levers. These projections remain open until a run on such hardware.

### First real GPU run (2026-08-14): the path works, the gate does not close

Hardware: NVIDIA GeForce RTX 5070 Ti, 16 GB, driver 595.97, under WSL2 Ubuntu with
`jax 0.11.0` on the `cuda12` backend (`jax.default_backend()` → `gpu`,
`[CudaDevice(id=0)]`). Blackwell needs no special handling. Same
`scripts/m5_scale_bench.py` graph as the CPU ladder above.

**What works.** The CUDA path runs and is linear in *n*, as the flop count predicts:

| n (model px) | native px/epoch | eval | ∇ eval |
|---|---|---|---|
| 4,877 | 1,702 | 0.232 s | 0.236 s |
| 9,526 | 3,457 | 0.447 s | 0.461 s |
| 18,221 | 6,965 | 0.915 s | 0.871 s |
| 31,734 | 13,062 | **out of memory** | — |

**Why the gate stays open.** Two independent hardware properties, neither a bug.

*Memory.* The run fails at 31,734 model px (one sixth of the design target) on a
single 7.42 GiB request, with 13.8 GiB free and preallocation disabled, consistent with
the memory-pass table below: the gradient needs **18.24 GB** at the design target, more
than any 16 GB card has. The GPU requests ~2.5× the CPU peak at the same size in one
contiguous buffer, so the CPU figures are a floor for GPU sizing, not an estimate of it.

*Arithmetic.* A GeForce card runs double precision at a fraction of its single-precision
rate, and albireo's solver contract is float64. Measured on a 4096³ matmul:

| | GFLOP/s |
|---|---|
| float32 | 39,023 |
| float64 | **783** |

**A 50× penalty.** 783 GFLOP/s of fp64 is the rate of a good desktop CPU, not of an
accelerator, so the eval times above beat this desktop's CPU by only about 2× rather than
by the order of magnitude projected. The projection assumed A100-class fp64, which
consumer silicon does not provide.

The hardware requirement is therefore ≥ 24–40 GB of device memory and a 1:2 fp64 ratio,
i.e. A100 or H100 class, not "one GPU". On that hardware both blockers lift: 80 GB clears
the 18.24 GB gradient with room to switch `remat=False`, and ~10–20 TFLOP/s of vector fp64
is 12–25× the card measured here. The 1–2 hour projection above stands untested: nothing
measured here contradicts or confirms it.

The run does settle portability, independently of the throughput gate: albireo's graph
compiles and runs correctly under CUDA with no code change, on a consumer card, on Windows
via WSL2.

### The hand-set light-ratio systematic, quantified (`scripts/m5_light_ratio_demo.py`)

The LB-1/HR 6819-type failure mode, measured on a seeded simulation (the paper
asset behind the planned HR 6819 headline case):

1. Disentangling at a hand-set wrong ℓ rescales the recovered line depths by exactly
   ℓ_true/ℓ_assumed: measured affine slopes 1.44 / 0.97 / 0.59 against predictions
   1.50 / 1.00 / 0.60 (assumed ℓ₂ = 0.2 / 0.3 / 0.5, truth 0.3), with the separate
   additive k≈0 envelope offset isolated by the fit. Line depths feed log g /
   luminosity-class diagnostics: this is the mechanism behind the debate.
2. The marginal likelihood profiled over ℓ₁ with hyperparameters refit by ML-II at
   every trial (like for like) is flat to <0.5 log-units across ℓ₁ ∈ [0.50, 0.85]
   under constant light: the data carry no light-ratio information. With fixed
   hyperparameters the profile shows O(10–100) spurious curvature that is entirely
   prior-mediated (a wrong ℓ forces rescaled spectra, which a fixed prior scale
   penalizes); this failure mode is documented. With three partial-eclipse epochs the
   same profile peaks at the true ℓ₁ = 0.70 with Δlog L = −145 at ±0.05.

### fd3 comparison harness (`scripts/fd3_bench.py`)

The fd3 v3.1 input format was reverse-engineered from the official example files and
the C source (documented in the script header): ln-λ master matrix with a
`# ncols X nrows` header, a comment-free stdin token stream for the control file, ω in
degrees, per-epoch σ only (no per-pixel weights, no masks; the benchmark therefore
runs gap-free so that neither code is handicapped), component B's RV sign applied
internally (identical to albireo's ω+π convention), and fd3's internal c = 299,800
km/s. The harness simulates an SB2 on the common log grid fd3 requires (no resampling
for either code), writes fd3 separation- and fit-mode inputs, runs the albireo side,
and compares component spectra (raw and mean-aligned, since both codes carry a k≈0
freedom) and wall time. The fd3 side needs the binary (~1.9 MB source tarball, GSL,
builds on Linux/WSL; no license is stated on the fd3 page, v2 was GPL and v3's GPL
statement was removed, so the author should be contacted before any redistribution).

### fd3, head to head (2026-08-14)

The tarball's prebuilt binary is 32-bit i386 and does not run on a modern x86-64 host, so
fd3 was rebuilt from source against conda-forge GCC and GSL under WSL2 Ubuntu. It is not
vendored into this repository: the distribution states no license.

The build was validated against the author's shipped `.mod` / `.res` / `.rvs` outputs for
four worked examples, which tests the rebuild across a different compiler, architecture
and GSL:

| example | fd3 wall | max abs. difference from the shipped `.mod` |
|---|---|---|
| `art_single` | 0.14 s | **0** (exact) |
| `art_double` | 2.52 s | 1.3 × 10⁻⁶ (the files are written to ~6 dp) |
| `art_triple` | 3.58 s | 1.0 × 10⁻⁹ |
| **`V453_Cyg`** (1344 px, a real published system) | 62.6 s | **0** (exact) |

The comparison, on the harness's seeded SB2 (20 epochs, SNR 100, a common ln-λ grid so
neither code resamples, no gaps and no masks so neither is handicapped, and the orbit fixed
at truth for both):

| | comp 1 RMS | comp 2 RMS | steady-state wall |
|---|---|---|---|
| **albireo** | 0.0118 | 0.0165 | 0.182 s |
| **fd3** | 0.1767 | 0.2597 | **0.111 s** |
| albireo, mean-aligned | **0.0093** | **0.0116** | |
| fd3, mean-aligned | 0.0198 | 0.0223 | |

fd3 is faster: 1.64× in steady state, 5.7× from cold (0.630 s including JAX compilation).
It is a small C program that starts, solves and exits, and a 1200-pixel two-component
separation is the regime in which a compiled direct method is expected to win; albireo is
in the same class, not an order of magnitude behind. (The harness's earlier "3.93 s" figure
was its un-jitted single-solve path and overstated the gap by 20×.) Timings are the minimum
of five repeats, both codes on CPU.

fd3's raw error is ~15× larger, and about nine tenths of it is a constant: mean-aligning
collapses comp 1 from 0.1767 to 0.0198. That is the *k* = 0 freedom both codes carry and
neither can determine from constant-light data, the null space of
[§5.1](math.md#51-the-low-frequency-degeneracy-the-undulations-theorem), and the reason the
literature's workflow includes a hand renormalization against an external light ratio.
albireo's smoothness prior pins the offset; fd3 leaves it to the user.

On shape, once that offset is removed, albireo is about 2× more accurate (0.0093 / 0.0116
against 0.0198 / 0.0223), from the prior constraining the low-*k* modes.

fd3 returns a point estimate with no uncertainty on the component spectra. [The handoff
tutorial](tutorials/downstream.md) turns albireo's uncertainty into an error bar on log *g*.

### Shift-and-add, clean room (2026-08-15)

The third code is the most widely used in the field. `scripts/shift_and_add.py` is a
clean-room implementation written from González & Levato (2006) §2.1 Eqs. (1)–(2) and §2.3,
with the identical recurrence restated independently by Quintero et al. (2020), and from no
source code. The widely used existing implementation, the one behind the LB-1 and HR 6819
companion identifications, carries no license file and was never opened.

Two details of the published method are easy to get wrong. The iteration is Gauss–Seidel:
the *B* update consumes the *A* produced in the same sweep; the Jacobi variant is a
different algorithm with a different convergence rate. The initialization is B = 0 with A
not seeded, so the first primary estimate is the plain rest-frame co-add.

The implementation is validated against the paper's theory. §2.3 derives that the residual
is not annihilated but diffused: each sweep convolves it with
`f(x) = n⁻² Σ δ(x − dᵢ + d_k)`, so after *m* sweeps it has been smeared by a Gaussian of
`σ = √(2m)·σ_d`. Seeding a delta-function error and measuring the returned width
reproduces that law (`tests/test_shift_and_add.py`), evidence that the coded recurrence is
the paper's.

Sweeps: the paper says "rarely more than 5–7". That holds on the benchmark's data, and the
comparison does not depend on the choice:

| sweeps | comp 1 aligned | comp 2 aligned | wall |
|---|---|---|---|
| 1 | 0.0431 | 0.0458 | 0.003 s |
| **7** (published figure) | **0.0248** | **0.0302** | **0.018 s** |
| 50 | 0.0221 | 0.0239 | 0.128 s |

Seven sweeps is close to converged; fifty improves the result by 11% for seven times the
cost and is still worse than albireo by a factor of 2.4, so stopping early does not
handicap the method.

### All three, on identical data

Same seeded SB2: 20 epochs, SNR 100, a common ln-λ grid so nothing resamples, no gaps or
masks, and the orbit fixed at truth for every code. Shift-and-add uses albireo's own linear
shift operator, so the comparison is between algorithms rather than between two
interpolators.

| | comp 1 raw | comp 1 aligned | comp 2 raw | comp 2 aligned | wall | uncertainty? |
|---|---|---|---|---|---|---|
| **albireo** | **0.0118** | **0.0093** | **0.0165** | **0.0116** | 0.182 s | **yes, a posterior** |
| fd3 | 0.1767 | 0.0198 | 0.2597 | 0.0223 | 0.111 s | no |
| shift-and-add | 0.0317 | 0.0248 | 0.0849 | 0.0302 | **0.018 s** | no |

On speed, albireo loses to both. Shift-and-add is 10× faster than albireo's steady state and
6× faster than fd3: it is a small number of array shifts and means, the expected ordering
for a 1200-pixel two-component separation.

These walls describe this measurement convention rather than the codes. The re-run section
at the end of this file took all three on the same desktop under one protocol: every
accuracy value reproduced exactly; the shift-and-add wall had been contaminated by the
harness's own in-process timing (in both recorded measurements) and fd3 inflated by its
BLAS spinning 32 threads; the clean ranking is shift-and-add 0.026 s, albireo 0.059 s, fd3
0.064 s pinned.

On accuracy albireo is about 2× better, and the margin is not a stopping artifact (see the
sweep table). The aligned ordering is albireo 0.0093 / 0.0116, fd3 0.0198 / 0.0223,
shift-and-add 0.0248 / 0.0302.

On raw error the two incumbents fail differently, through the same degeneracy. fd3's raw
RMS is 15× albireo's and nine tenths of it is a constant. Shift-and-add's raw error is much
smaller than fd3's but grows on the fainter component (0.0849 raw against 0.0302 aligned)
because `B = 0` leaves the secondary's continuum level set by the initialization rather
than by the data. Both are the *k* = 0 null space: the per-mode convergence factor of the
shift-and-add recursion has modulus exactly 1 at zero frequency, so that mode is a fixed
point no number of sweeps can move. Three independent methods reach the degeneracy of
[§5.1](math.md#51-the-low-frequency-degeneracy-the-undulations-theorem). albireo's
smoothness prior pins it; the other two leave it to the user, which is the role of the
literature's hand renormalization against an external light ratio.

No handicap equalizes the last column, the presence of an uncertainty.

A comparison on an SB2 with a nebular line, a method that masks the contaminated pixels
against methods that cannot, would be unfair to shift-and-add: González & Levato explicitly
permit "any combination algorithm... weights or some rejection algorithm", so masking is
inside the published method. `tests/test_shift_and_add.py` exercises it: zeroing an epoch's
weight removes a ruined epoch. What the method cannot do is produce an uncertainty.

### AI Phoenicis: real spectra, and an orbit known better than any code can measure it
(2026-08-15)

AI Phe has 36 archival HARPS spectra (ESO, *R* = 115,000, 3782–6913 Å, SNR 41–129, all ten
phase bins filled, fetched with `albireo.archive`). There is no truth spectrum, but the
orbit is published to a precision no disentangling code approaches:

> K₁ = 51.164 ± 0.007 km/s,  K₂ = 49.106 ± 0.010 km/s,  P = 24.5924 d,
> e = 0.1878 ± 0.0006,  ω = 110.30 ± 0.06°,  T₀ = BJD_TDB 2458362.82847
> — Maxted et al. (2020), MNRAS 498, 332

That is 0.014% and 0.020%, from several independent studies agreeing to 0.1%, so the
ground truth is the orbit. `scripts/aiphe_bench.py` runs it.

The eccentricity is recovered from the spectra alone, starting the optimizer 15% off the
published eccentricity vector and 8% off both semi-amplitudes:

| | albireo | published | |
|---|---|---|---|
| e | **0.1879** | 0.1878 ± 0.0006 | +0.0001 |

A TESS light curve and 36 HARPS spectra agree to 0.05% by independent routes.

The semi-amplitudes carry a reproducible ~1% systematic. Two disjoint windows, chosen to
share no lines:

| | 5150–5250 Å | 5340–5440 Å | published |
|---|---|---|---|
| K₁ | 50.452 (−1.39%) | 50.440 (−1.42%) | 51.164 ± 0.007 |
| K₂ | 49.495 (+0.79%) | 49.479 (+0.76%) | 49.106 ± 0.010 |
| *q* = K₁/K₂ | 1.0193 | 1.0194 | 1.0419 |

Both runs used the period rounded to 24.5924 d and a second window starting at 5340 Å.
`aiphe_bench.py` now carries the unrounded photometric period, 24.592483 d (Kirkby-Kent
et al. 2016), and starts the cross-check window at 5341 Å so that its 3 Å pad clears
HARPS's inter-CCD gap, 5304.67–5337.61 Å; at 5340 Å the pad reached 0.6 Å into the gap
and clipped about 33 pixels per epoch, which `mask_flux_gaps` removed. Neither changes
this table: `period` and `t_conj` are free in the K fit, and the clipped pixels carry no
flux. Both matter for the runs below that hold the velocities fixed.

The two windows agree with each other to 0.02% in K₁ and 0.01% in the mass ratio, and both
sit the same distance from the published values. Three explanations are excluded:

* Not the optimizer. The fit converged, `|grad| = 9e-03`, and the answer is unchanged
  between a 250-step run that hit its cap and 444 steps that did not.
* Not line selection or the window. Two disjoint windows agree to 0.02%.
* Not the light ratio. Sweeping the assumed ℓ₂ from 0.38 to 0.53 moves the likelihood
  difference between albireo's K and the published K by 9 nats out of 53,306.

What remains is a ~1% systematic in the disentangling of this system, biasing the mass ratio
2.2% toward unity, the direction expected when two similar stars' line signals are partly
confused. It is an open lead, not a measurement and not a correction to the literature: a
value good to 0.02% from several independent cross-correlation studies of full échelle
spectra is the better number.

The formal statistics cannot arbitrate. albireo's optimum sits 53,306 nats above the
published K, which is not decisive: 358,265 high-SNR pixels make a one-pixel systematic
velocity offset worth that much. When formal precision is far finer than the systematic,
likelihood ratios stop being informative. The residual z-RMS is 2.64 rather than 1: HARPS
ships no error array, so the weights are albireo's own scatter estimate, and any formal
error bar from this fit is ~2.6× too tight until they are rescaled.

On the spectra, albireo and the clean-room shift-and-add agree on real data: mean-aligned
RMS 0.029 and 0.037 in the line cores of the 5340–5440 Å window, against line depths near
0.8. This is the available consistency check with no truth spectrum. Shift-and-add again
took 0.07 s against albireo's 11 s.

### The bug real data found, which no simulation would have

A second window at 5300–5400 Å straddles the gap between HARPS's two CCDs
(5304.67–5337.61 Å, 32.9 Å of exact zeros), and those pixels arrived weighted like data:
finite, with no quality column, and, because HARPS ships no error array, with an inverse
variance estimated from the local scatter, which across a flat run of zeros is small.
Median ivar was 6398 across the gap against 6231 for real pixels, with `mask` empty. A
window that was 33% detector gap disentangled to component spectra with negative flux.

`albireo.mask_flux_gaps` zero-weights contiguous runs of non-positive flux and warns with
the wavelength range; `to_epoch` calls it before the spike clip, since a flat run has no
local scatter for a running median to catch. The rule is about runs: `RawSpectrum.bad_pixels`
does not treat a single zero flux value as missing, because one zero can be a saturated core
or a clipped cosmic ray, whereas eight in a row cannot. It parallels the reader's
zero-error-means-infinite-precision rule: the reader may decline to answer, but it may not
guess.

### A correction to why this system was chosen

AI Phe eclipses, but the eclipse does not give the spectroscopic light ratio directly. It
pins the fractional radii, the inclination and the surface-brightness ratio in the
photometric band, TESS, centred near 7860 Å. The light ratio at an optical spectroscopic
window is a different number, computed from the radii and the two temperatures, and
strongly wavelength dependent: for AI Phe's 6310 K + 5010 K pair at R₂/R₁ = 1.624, a
blackbody estimate gives ℓ₂ = 0.375 at 4000 Å and 0.510 at 6500 Å. It is far better
constrained than for a non-eclipsing system, but using the TESS-band value at 5200 Å would
be a ~10% error in the quantity every recovered line depth scales by.

### Tutorials, examples, CI

Two executable tutorials (`examples/01_sb2_end_to_end.py`, `02_k2_scan.py`) run
the real pipeline with asserts and back the narrative docs pages; a dedicated CI
smoke job runs both with `ALBIREO_EXAMPLE_FAST=1` (~3.6 min + 7 s measured).
The SB2 example's NUTS posterior at tutorial scale: P to 0.013%, K₁ 0.015%,
K₂ 0.21%, zero divergences.

### Release readiness (JOSS) and the real-data decision

`paper/paper.md` + `paper.bib` drafted (claims cross-checked against this file;
"GPU-accelerated" kept out of the title until the GPU gate closes),
`CONTRIBUTING.md` added; remaining JOSS blockers are maintainer-only (affiliation,
ORCID, archive DOI at acceptance, PyPI registration). For the real published SB2
end-to-end, the research recommendation is **HR 6819** (51 public FEROS spectra,
one ESO program, ~153 MB, no login; hand-set light ratio at the heart of the
2020 black-hole debate, with an *interferometric* ground truth
f = 0.439 ± 0.013 from GRAVITY to score the posterior against) with
**AI Phoenicis** as the precision backup (60 public epochs; K's known to 0.02%).
Data download awaits maintainer approval.

## Speedup pass: direct band assembly and a closed-form gradient (2026-08-11)

Same machine, same ladder configuration, same seeds; log-likelihoods agree with
the scale-stage record to machine precision (several rows bit-identical, the rest at
~1e-15 relative, since the assembly changes only the floating-point summation
order).

### Where the time actually went

A stage-split profile at the 31.7k ladder row attributed 22.3 s of the 24.3 s
evaluation (92%) to comb probing; the block Cholesky was 0.86 s, and the
remainder was noise. Probing pays 2p+1 = 1027 matrix-free operator applications,
the union of all epochs' band offsets, although each epoch contributes only a
~50-pixel-wide band at a velocity-determined offset. That redundancy, not the
factorization, was the scale problem.

### What replaced it

1. **Direct per-epoch band assembly** (`albireo/assembly.py`, math.md §4.5): each
   epoch's (i,j) block is ℓᵢℓⱼ·T(δᵢ)ᵀ·G·T(δⱼ) with G = KᵀRᵀW′RK a narrow band,
   assembled by static rebin pair tables (one `segment_sum` per epoch), two
   unrolled kernel-shift passes, and a four-term tent-weighted combination of
   row-translated copies, accumulated into a global band tensor by
   `dynamic_update_slice` (no scatters anywhere on the hot path). O(band width)
   work per epoch instead of O(bandwidth) matvecs: ~12× on the assembly stage.
   Probing survives as `assembly="probe"` (reference) and as the `validate=True`
   oracle, which now cross-checks the band assembly against the matrix-free
   operator directly.
2. **Closed-form solve-stage gradient** (custom VJP): cotangents of
   {log det, quadratic form, d̂} against the precision come from the block-
   Takahashi banded selected inverse and d̂-outer products, so reverse mode
   never walks the Cholesky/solve scans; assembly-side reverse work shrank further
   by hoisting the velocity-independent G stage out of the rematerialized scan
   and by expressing row translation as clip-safe `dynamic_slice` (its transpose
   is a contiguous copy, where a gather's transpose is a scatter). Verified
   against plain autodiff at 1e-13 relative and by finite differences.

### Ladder, before → after (CPU, same desktop, jitted, p = 513, 50 epochs, SB2)

| n (model px) | eval before | eval after | ∇ before | ∇ after |
|---|---|---|---|---|
| 31,734 | 24.7 s | **3.0 s** (8.2×) | 102.7 s | **10.6 s** (9.7×) |
| 74,322 | 52.5 s | **8.0 s** (6.6×) | 249.9 s | **24.9 s** (10.0×) |
| 135,052 | 96.9 s | **14.5 s** (6.7×) | 466.0 s | **56.5 s** (8.2×) |
| **203,440 (design target)** | 149.7 s | **26.0 s** (5.8×) | 729.9 s | **111.3 s** (6.6×) |

(`scripts/m5_scale_bench.py`, single sequential run, no external load.) A
design-target marginal evaluation (disentangled spectra at a fixed orbit) is now ~26 s
on one desktop CPU (was 2.5 min), and a gradient under 2 min (was 12 min). The gate-scale
NUTS test takes ~65 s wall (Laplace + warmup + 250 samples) against ~102 s at the
joint-NUTS baseline and 112 s in the scale-stage record, so the small-problem regression
that the probe-era size-adaptive policy prevented no longer occurs and the policy is no
longer needed. The ratio erosion at the top row is because the design-target gradient's
working set exceeded this machine's 32 GB (measured 34.0 GB; the memory pass below brought
it to 18.2 GB and the top row to 22.2 s / 87.4 s). At realistic single-star bandwidths
(HR 6819-like: p ≈ 160 rather than the ladder's conservative 513) the same operations take
seconds per gradient, putting full NUTS posteriors for real SB2 problems within reach of
one desktop CPU; the GPU budget becomes headroom rather than a requirement.

### Found in passing: the Laplace mass matrix was built from a defective Hessian

Two second-order facts. First, the initial custom rule was first-order exact but
second-order wrong (8e-3 relative): its forward rule called the custom function itself,
so Hessians re-entered the custom boundary and lost the chol-mediated terms through the
dropped cotangent. Inlining the primal in the forward rule fixes it; `jacrev(jacrev(...))`
then agrees with plain autodiff to 1e-15. Second, independent of the scale work:
`jax.hessian` (forward-over-reverse) produces an asymmetric Hessian on this stack even for
the plain-autodiff path (off-diagonal 0.566 vs 0.855 on the diagnostic problem), while
reverse-over-reverse matches central finite differences of the gradient to 8 digits at
three step sizes. `laplace_inverse_mass`, the only forward-mode consumer in the package,
had been symmetrizing a slightly wrong matrix since joint inference was added; it now uses
reverse-over-reverse (math.md §4.5). The mass matrix is a preconditioner, so posteriors
were never biased; warmup was tuned from a mildly wrong curvature estimate.

### GPU consequences

The hot path is now vmapped dense convolution-like passes, contiguous dynamic
slices, and one 50-step scan with large per-step work, all GPU-native shapes;
the probe-era `probe_chunk`/`remat` tuning is gone along with probing. The
remaining GPU-specific bottleneck is chain latency in the sequential block
Cholesky/Takahashi scans (~800 steps of 513² work at the design target). Remaining
levers: the associative-scan (parallel-prefix) factorization that removes it, a
custom-VJP band→block packing, a fully analytic assembly VJP, and opt-in mixed precision.
Projection, not measurement: design-target gradient ~0.5–1.5 s on one A100-class device,
a converged design-target posterior in tens of minutes; the earlier 1–2 h projection
stands as the conservative bound.

---

## Memory pass, and a boundary bug it turned up (2026-08-12)

After the speedup pass the design target was fast but not runnable: XLA
`memory_analysis()` of the compiled executables (buffer assignment without allocation)
put the design-target gradient at 34.0 GB against 32 GB of RAM.

### Where the bytes were (design target: 203,440 model px, SB2, 50 epochs, p = 513)

| stage | GB | fate |
|---|---|---|
| `G` pre-pass, `vmap`ped over all 50 epochs | ~9 | `vmap` batches *every intermediate* of the chain (H, the two kernel stages), not just the 4.5 GB result |
| band tensor + its cotangent | 6.2 | unchanged (next lever) |
| `bt` + Cholesky factor | 6.2 | unchanged (both genuinely live) |
| selected inverse (`s_diag`, `s_sub`) | 3.1 | **removed** — fused into the cotangent |
| `_pack_band` gather indices/masks, stacked over K blocks | ~5 | **removed** — hoisted + rematerialized |
| prior block-tridiagonal factor | 0.8 | **removed** — scalar recursion |

### Four exact reductions

1. **Fuse the Takahashi sweep into the cotangent** (`solver.selected_inverse_cotangent`).
   Each Σ block is contracted against `d`, `u` at the step that produces it, so
   the `2K − 1` selected-inverse blocks and the outer-product temporaries never
   exist. `selected_inverse_blocks` stays as the test oracle.
2. **Batch the `G` pre-pass over epochs** (`epoch_chunk`). Because `G` is
   velocity-independent it is computed once per epoch either way, so any
   batching costs exactly one extra `G` pass in the rematerialized backward, and
   the size matters only for `vmap` width. The default hoists the whole pre-pass
   below ~1 GB (small and gate-scale problems are unaffected) and otherwise batches
   to ~0.5 GB.
3. **Prior determinant by scalar pentadiagonal recursion**
   (`assembly.prior_logdet`) instead of factorizing a bandwidth-2 matrix as
   6,358 dense 64×64 blocks.
4. **Hoist `_pack_band`'s gather indices** out of the `lax.map` body (the band
   coordinates depend only on the within-block (row, col), never on the block
   counter) and rematerialize the body, so reverse mode stops stacking (B, B)
   index and mask arrays over all K iterations.

| n (model px) | eval before → after | ∇ before → after |
|---|---|---|
| 31,734 | 2.93 → 2.94 GB | 5.02 → **4.00 GB** |
| 74,322 | 7.04 → **4.86 GB** | 11.92 → **11.47 GB** |
| 135,052 | 12.0 → **7.83 GB** | 16.64 → **14.37 GB** |
| **203,440 (design target)** | 20.64 → **11.06 GB** | 34.00 → **18.24 GB** |

Log-likelihoods, gradients and Hessians are unchanged; the equivalences are
regression-tested against the routes they replaced (blocked prior determinant,
unfused selected inverse, unbatched pre-pass).

Wall clock, same desktop, same seeds. At the design target the pass also reduced time:

| n (model px) | eval, before → after | ∇, before → after |
|---|---|---|
| 31,734 | 3.0 → **2.82 s** | 10.6 → 11.22 s |
| 74,322 | 8.0 → **6.95 s** | 24.9 → 30.67 s |
| 135,052 | 14.5 → 14.90 s | 56.5 → 59.53 s |
| **203,440 (design target)** | 26.0 → **22.16 s** | 111.3 → **87.43 s** |

The middle rows' gradients get slower by 5–23%: the batched pre-pass's extra backward
`G` pass, paid where memory was not the binding constraint. At the design target, which
previously did not fit, halving the working set wins outright because the gradient was
thrashing against RAM. Raising `epoch_chunk` to the epoch count on a machine with memory
to spare (or on GPU) trades back.

### The bug the memory work uncovered

Re-deriving the band layout exposed a correctness defect in the new assembly.
`G = Kᵀ(RᵀW′R)K` is built as a band image;
`H` is exactly zero outside the model grid, but the LSF convolution smears
in-grid mass outward, writing band entries at column indices that correspond to
grid pixels that do not exist. The T-sandwich reads those entries whenever an
epoch's shift places a component's support against a grid edge, where `T(δ)` has
no row and the contribution should be zero.

The trigger is the data-coverage margin, in model pixels, being smaller than the
LSF kernel radius. Measured band-vs-probing, dense and entrywise (probe is the
symmetric ground truth; only the column side leaked, so asymmetry is the
discriminator):

| coverage margin (model px) | kernel radius | max relative entry error | Δ log L |
|---|---|---|---|
| ~24 | 6 | 6.4e-15 | 2.7e-15 |
| ~8 | 6 | 6.4e-15 | 0 |
| ~4 | 6 | 6.4e-04 | 1.8e-07 |
| **0** | **6** | **6.8e-02** (asymmetry 6.8e-02) | **−57 nats** |
| 19 | 34 | 6.9e-04 | 0.16 nats |

In the last row, at a fixed grid, fitting a wider LSF is enough to reach the defect. A
margin of zero is a usual configuration: it follows from choosing a model grid narrower
than the observed range to fit a sub-region, the documented pattern. Every pre-existing
fixture left a margin exceeding the kernel radius, so the weights vanished where the defect
lives, which is why 174 tests passed over it. `scripts/m5_scale_bench.py`'s largest row
sat about one pixel from the boundary.

The fix masks `G`'s out-of-grid columns at the source (one static boolean per
group); the margin-0 case then agrees at 5.7e-15 with asymmetry 2.9e-16. Two
fixtures pin it, `edge_covered` in the equivalence set and a dedicated
model-grid-inside-the-data case, and the equivalence check gained an entrywise
dense comparison plus an asymmetry assertion: the log-determinant and solve it
previously relied on average a boundary defect away, which at the 0.3 Å margin put it
under a 1e-11 scalar threshold.

### Two more silent-wrongness fixes

- **Gradients through the Cholesky factor were identically zero.** `_solve_stage`
  returned the factor, but its closed-form reverse rule cannot carry a cotangent
  on it (propagating one is the reverse pass through the factorization that the
  rule exists to avoid), so anything differentiating `spectra_std` or
  `draw_spectra` received a silent zero. `MarginalResult` now stores
  the precision and rebuilds the factor outside the custom boundary, where plain
  autodiff applies; the cost is one extra block Cholesky, paid only by callers
  that ask for the factor, which the sampling hot path never does.
- **A few wide native pixels silently set the solver bandwidth.** `row_support`
  is a max over native rows, and it drives a cost quadratic in the block size.
  In real spectra a wide row usually means samples were deleted (telluric
  window, order or chip gap): edges sit at midpoints, so removing samples makes
  the two bracketing pixels absorb half the gap each.
  `build_problem` now warns, names the offending pixel, and states the remedy
  (mask with `ivar = 0`, or split into separate instrument labels).

### What is left

The band tensor and its cotangent (6.2 GB) still store both triangles of a
symmetric matrix, and `bt` and its factor (6.2 GB) are both live across the
Cholesky. Folding the band to one triangle, and assembling directly into block
storage so the band tensor never exists, are the next levers. Neither is needed to
run the design target, which has ~13 GB of headroom on this machine.

## HR 6819 — the first observed dataset (2026-08-12)

The first observed dataset is the 51 public FEROS exposures of HR 6819
(ESO 073.D-0274(A), PI Rivinius, 153 MB, anonymous), the data behind every published
analysis of the system.

### What the data actually are, versus what the model wanted

| | expected | delivered |
|---|---|---|
| continuum | flux ~ 1 | `CONTNORM = False`; raw merged-echelle ADU, response falling **20×** over 3850–4750 Å, negative below ~3830 Å |
| uncertainties | `ivar > 0` | `ERR` column **entirely NaN** ("Error spectrum not available") |
| wavelength grid | one per instrument | 0.03 Å step shared, but start wavelengths spread over 0.78 Å and lengths 189621–189653 → **28 distinct grids** |
| frame | declared | `SPECSYS = BARYCENT`, correction in `ESO DRS BARYCORR` (−25.13 to +16.12 km/s) |
| time | BJD_TDB mid-exposure | `TMID` = MJD(UTC), and in the **extension** header, not the primary |

### Five defects, each found by a property of the data

1. **NaN at a zero-weight pixel took the whole likelihood to `nan`.** `data.py`
   documents that a masked pixel's flux "is never read"; `build_problem` formed
   `z = flux − r·base` unmasked and every consumer multiplies by `w`, so
   `0 · nan = nan`. `normalize` writes exactly that wherever the fitted continuum
   collapses. Measured: identical log-likelihood before/after the fix when the
   masked block holds `nan`, `±inf`, or `1e300`.
2. **Per-exposure grids were rejected outright.** Grouping is now by
   (instrument, grid); the instrument key still keys the LSF, so `lsf_sigma_v`
   stays one entry per instrument rather than 28.
3. **`prior_logdet` returned `nan` below `eta/tau ≈ 1e-13`**: an unguarded
   `sqrt` of a difference of like-sized terms, reachable by unbounded ML-II.
4. **The row-support warning quoted `int64` minimum** as its median whenever more
   than half an epoch lay outside the model grid: rows the rebin operator never
   touched carried a sentinel instead of being excluded.
5. **A region disjoint from the model grid failed as "empty rebin operator"**,
   with no indication of which of the two ranges was wrong.

### Continuum: why the fit is done in the log

A curvature penalty applied to the flux cannot track a multiplicative response.
Measured on one exposure over 3850–4750 Å, normalizing with a 150 Å linear-space
smoother left the 97th percentile of the normalized flux at 0.55–0.63 in the
worst 50 Å bins, a 40% error. Fitting `log(flux)` instead (a pure exponential is
a straight line, and straight lines are in the penalty's nullspace):

| smoothing scale | p97 of normalized flux, per 50 Å bin, 3850–4750 Å |
|---|---|
| 80 Å | 1.006 – 1.010 |
| 150 Å | 1.007 – 1.011 |

The result is flat across the whole 20× gradient and insensitive to the
smoothing scale.

The second half of the fix is the knot basis. A per-pixel Whittaker smoother needs
`λ ≈ (L_px/2π)^4`; at the 5000–10000 pixel smoothing lengths a merged echelle
spectrum calls for, that is `λ ~ 10¹²`, the weight term is lost to rounding, and
`solveh_banded` fails with *"leading minor not positive definite"*. Eight knots per
smoothing length holds `λ ≈ 2.6` at any requested scale.

### Barycentric sign, checked against the sky rather than against ourselves

No test pinned albireo's `v_bary` convention to an external standard. Cross-correlating
the telluric O₂ A band (7595–7660 Å) across epochs spanning 55 km/s of correction:

| | rms residual |
|---|---|
| telluric shift = **+BARYCORR** | **0.138 km/s** |
| telluric shift = −BARYCORR | 59.3 km/s |

The slope is 0.9993, so `frame="barycentric"` with `v_bary = ESO DRS BARYCORR` is
correct, and the 0.14 km/s floor is CCF precision plus real water-vapour variability.
Independently, astropy's own correction agrees with the pipeline's keyword to
**0.017 km/s**.

### Configuration for the science run

4380–4600 Å (He I 4388/4471, Mg II 4481, Si III 4552/4568/4575: photospheric in
both components, no Balmer core, no variable disc emission, nearest telluric band
1200 Å away). After `share_wavelength_grid` (residual relabelling **0.007 km/s**,
1/300 of a pixel) the 28 groups collapse to **1**.

| | |
|---|---|
| epochs / native pixels | 51 / 373,983 (100.0% good) |
| model grid | 9,938 px, 4378.45–4601.65 Å, dv = 1.50 km/s |
| operator groups | 1 |
| row support / kernel radius | 3 / 8 |
| half-bandwidth `b_nat`, `p` | 81, 163 |
| marginal log-likelihood eval | 0.5 s (after a 1.0 s compile) |

### The fit, and what it says about systematics

MAP + ML-II from a conjunction-phase scan: 120 L-BFGS steps, 2820 s, peak RSS
2.6 GB. The gradient norm plateaus around 4×10² and oscillates, because
`run_map`'s absolute `tol` is unreachable at 3.7×10⁵ good pixels; a `callback`
lets convergence be judged from the parameters rather than from the flag.

Profiling the marginal likelihood around the MAP, in **two independent windows**:

| | A: 4380–4600 Å | B: 4120–4330 Å | Klement et al. 2025 |
|---|---|---|---|
| lines | He I 4388/4471, Mg II 4481, Si III 4552/68/75 | He I 4144/4169, Si II, Fe II | — |
| period [d] | 40.36583 ± 0.00045 | 40.37022 ± 0.00065 | 40.3261 ± 0.0013 |
| K<sub>pre-sd</sub> [km/s] | 63.314 ± 0.013 | 63.724 ± 0.019 | 61.15 ± 0.88 |
| K<sub>Be</sub> [km/s] | 1.985 ± 0.151 | 3.022 ± 0.153 | 3.90 ± 0.27 |
| eccentricity | 0.0302 | — | 0.0289 ± 0.0058 |
| whitened residual sd | 1.674 | 1.403 | 1 if calibrated |

This is a useful result but not a publishable orbit:

* Eccentricity lands at 0.2σ: 0.0302 against 0.0289 ± 0.0058, on a nearly circular orbit,
  from spectra alone.
* The quoted ± are statistical only, and they are wrong by at least an order of magnitude.
  The two windows differ by 0.0044 d in period (5.6σ of their combined internal errors) and
  0.41 km/s in K<sub>pre-sd</sub> (17.8σ). Window-to-window scatter is the lower bound on
  the error bar; the likelihood curvature is not.
* The residual scatter, 1.4–1.7× the assumed noise, indicates real unmodelled
  structure. Candidates, in rough order of suspicion: the pipeline resampled these spectra
  onto a common step, so the diagonal `ivar` model is optimistic by construction; per-epoch
  continuum residuals; a Gaussian LSF standing in for FEROS's real one; and the Be star's
  disc emission violating the one-static-spectrum-per-component assumption over a 135-day
  baseline.
* K<sub>pre-sd</sub> is 3.5–4% above the literature, consistently in both windows. The sign
  is the physically expected one: the published values come from cross-correlation and
  Gaussian line fits, which blend the sharp lines with the Be star's broad, nearly
  stationary ones and are therefore biased toward the systemic velocity. Deblending should
  raise K. This run does not test that hypothesis.
* K<sub>Be</sub> is detected but not measured. The profile is a single clean peak, 83 nats
  above K = 0 in window A, yet the two windows give 1.99 and 3.02 against a literature
  3.90. At v sin i ≈ 200 km/s the Be star's reflex motion is ~1/50 of a line width, the
  unattributable regime of `math.md` §5.1, and its recovered spectrum is prior-dominated.

NUTS would return the same optimistic width around the same systematics-limited point.
The run first needs a noise model with a fitted inflation factor, a wider window, and a
check of whether the period offset survives a per-epoch continuum treatment. The next
section covers the first.

---

## The jitter site, and what it did to HR 6819 (2026-08-12)

The run above took its estimated `ivar` at face value and reported a residual scatter of
1.4–1.7 as a caveat. `forward.with_jitter` and the `log_jitter` θ site supply a per-epoch
noise-inflation factor.

### First: the marginal counts resolution elements, not pixels

Profiling a single shared α at the previous MAP, against the standard deviation of the
whitened residuals (the naive estimator):

| | window A | window B |
|---|---|---|
| weighted pixels `N` | 373,813 | 356,928 |
| residual sd (the naive estimator) | 1.6743 | 1.4030 |
| profiled `α̂` | **1.6807** | **1.4088** |
| implied `p_eff = N[1 − (sd/α̂)²]` | **2,843** | **2,930** |
| model pixels `n_comp · n_pix` | 19,876 | 20,160 |
| resolution elements (FWHM = 4.16 px) | 4,779 | 4,846 |

The correction is in the predicted direction (`math.md` §3.2a: `α̂² = χ²/(N − p_eff)`,
not `χ²/N`), but here it is only 0.4%, because `p_eff` is not the model pixel count. At
`dv = 1.5` km/s the grid oversamples FEROS by 4.2×, and the ML-II smoothness prior is
stiffer still, so of ~20,000 nominal spectral parameters only ~2,900 are data-determined,
roughly 60% of the resolution-element count. Inverting the two estimators is an
inexpensive route to `p_eff`, otherwise awkward to obtain; in the `tests/test_jitter.py`
fixture, built with a weak prior, the same inversion gives `p_eff/N = 0.09` and the naive
estimator is 4.6% low.

Per-epoch factors gain much more than one shared factor: +19,763 nats (A) and
+10,020 nats (B) over the best shared α. The exposures are not equally good: α runs
1.11–3.61 in window A, so the worst exposure carries 1/13 of the weight the `ERR`-free
DER_SNR estimate would have given it.

### Then: it whitens the residuals and makes the orbit worse

Joint MAP over orbit + hyperparameters + 51 per-epoch jitters, 70 L-BFGS steps, ~380 s per
window, both windows fitted independently:

| | A, no jitter | A, jitter | B, no jitter | B, jitter | Klement et al. 2025 |
|---|---|---|---|---|---|
| period [d] | 40.36583 ± 0.00045 | **40.44429 ± 0.00067** | 40.37022 ± 0.00065 | **40.42979 ± 0.00088** | 40.3261 ± 0.0013 |
| K<sub>pre-sd</sub> [km/s] | 63.314 ± 0.013 | **63.074 ± 0.022** | 63.724 ± 0.019 | **63.400 ± 0.024** | 61.15 ± 0.88 |
| K<sub>Be</sub> [km/s] | 1.985 ± 0.151 | **1.450 ± 0.209** | 3.022 ± 0.153 | **2.658 ± 0.248** | 3.90 ± 0.27 |
| eccentricity | 0.0302 | **0.0241** | — | **0.0213** | 0.0289 ± 0.0058 |
| whitened residual sd | 1.674 | **0.997** | 1.403 | **0.997** | 1 if calibrated |

The noise model is now self-consistent, with residual sd 0.997 in both windows. Everything
else got worse:

| | no jitter | with jitter |
|---|---|---|
| window A vs B, period | 5.6 × combined formal σ | **13.1 ×** |
| window A vs B, K<sub>pre-sd</sub> | 17.8 × | 10.0 × |
| window A vs B, K<sub>Be</sub> | 4.8 × | 3.7 × |
| period vs literature (A) | 28.9 σ | **80.7 σ** |
| eccentricity vs literature (A) | 0.2 σ | 0.8 σ |

The error bars grew by about α, as expected (1.3–1.7×). The central values moved much
further than that: the period shifted by 0.078 d, which is 174× the no-jitter formal error
on the same window, and away from the published value.

### Why — and a check that it is not a bug

Evaluating both parameter vectors under both weightings separates "the reweighting moved
the optimum" from "the optimizer walked somewhere the objective would not go":

| | weights α = 1 | weights α = MAP |
|---|---|---|
| θ from the no-jitter fit | **1,350,406** | 1,514,430 |
| θ from the jittered fit | 1,339,784 | **1,518,265** |

Each wins under its own weights, by 10,622 and 3,835 nats respectively: both fits are
correct and answer different questions. Scanning the period under the jittered weights
shows structure far wider than the surface's curvature: from the no-jitter θ the
conditional optimum is 40.3600 ± 0.0007, from the jittered θ it is 40.4442 ± 0.0007, two
optima 125 σ apart, with the second higher.

The mechanism is in which exposures got downweighted. Within window A,
`corr(α, phase along the baseline) = −0.25`: the noisiest exposures concentrate early, four
of the worst five below phase 0.35 of the 134.7-day baseline. Downweighting one end of the
baseline gives up period leverage there, and a period fit pivots about the weighted centre
of its data, which the reweighting moved from phase 0.581 to 0.605. The two solutions
behave like a pivot: they agree on a conjunction at BJD 2453221.7 (phase 0.615, within
1.4 d of that weighted centroid) and diverge on either side of it.

### What to take from this

* The jitter site is exact (jitter α is bit-equivalent to being handed `ivar/α²`), it
  profiles to the dof-corrected estimate, and it turns a residual excess of 1.67× from a
  caveat in prose into a fitted parameter.
* It is not a repair for correlated residuals: a rescaled diagonal noise model is still
  diagonal. Here it whitened the residual scale while leaving whatever generates the
  structure untouched, and relocated the answer by 174 formal σ.
* The noise model selects the optimum. Two defensible noise models, fitted to the same data
  in the same window, disagree by far more than either one's stated error. The spread
  across independent windows and across defensible noise models should be quoted alongside
  any formal error.
* The eccentricity survives, less well: 0.0241 and 0.0213 against 0.0289 ± 0.0058, i.e.
  0.8σ and 1.3σ, where the no-jitter run gave 0.2σ. Still consistent; the 0.2σ was
  fortuitous.

## The response site, and the exoneration of the continuum (2026-08-12)

"Per-epoch continuum residuals" was near the top of the HR 6819 suspect list for the
1.4–1.7× residual excess. Here the multiplicative response coefficients, previously fixed
at build time, become a θ site.

### The swap

The response enters the targets ``z = y − r(R·1)`` and the sandwich weights ``w r²``, not
only the forward operator. The swap is exact and inexpensive because ``R·1`` (the rebinned
unit continuum, stored per group) is
response-independent: ``z_new = z_old + (r_old − r_new)·R·1`` rebuilds the target with no
raw fluxes carried, re-masked so the ``0·nan`` trap found on HR 6819 cannot resurface, and the
``Σ log w`` term is untouched because the noise lives on the data (math.md §7.5). The
traced Clenshaw matches ``np.polynomial.chebyshev.chebval`` operation-for-operation, so
`with_response` equals a fresh ``build_problem`` with ``r`` bitwise identical and the
marginal to rtol 1e-12 (`tests/test_response.py`).

### Closed loop (10 epochs, order 2, injected c ~ N(0, 0.03), SNR 130)

Joint MAP over orbit + hypers + 30 response coefficients: injected coefficient rms 0.0343,
difference-mode error rms 0.0020, K errors +0.21% / −0.10%, and the fitted response exceeds
the unit response by 29,938 nats at the same orbit. The epoch-shared mode comes out at its
zero-centered prior rather than at truth (c₀ error −0.069 against a 0.05 prior σ), which is
§5's response-to-broad-features degeneracy. The test asserts the common mode at prior
scale; a tighter assertion would test the prior rather than the data.

### The answer on HR 6819: the offsets are not the continuum's fault

`scripts/hr6819_response_run.py`: both windows, 150 L-BFGS steps per config, response
fits warm-started from the baseline MAP, order 2 per epoch (153 coefficients), prior
N(0, 0.02²). Uncertainties are conditional-orbit Laplace (nuisances at MAP), computed
identically for all four fits; they are ~2× the formal errors recorded above, which came by
a different route, and this does not affect the comparisons below.

| | A: baseline | A: response | B: baseline | B: response |
|---|---|---|---|---|
| period [d] | 40.36566 ± 0.00099 | 40.36606 ± 0.00099 | 40.36091 ± 0.00140 | 40.36069 ± 0.00140 |
| K<sub>pre-sd</sub> [km/s] | 63.308 ± 0.015 | 63.308 ± 0.015 | 63.575 ± 0.021 | 63.575 ± 0.021 |
| K<sub>Be</sub> [km/s] | 1.928 ± 0.189 | 1.928 ± 0.193 | 2.946 ± 0.175 | 2.947 ± 0.178 |
| eccentricity | 0.0302 | 0.0301 | 0.0240 | 0.0240 |
| whitened residual sd | 1.674 | **1.668** | 1.401 | **1.394** |
| Δ log-likelihood | — | **+4,100** | — | **+3,926** |
| fitted response rms | — | 0.0044 | — | 0.0011 |
| … difference mode | — | 0.0005 | — | 0.0005 |

* The site works and the continuum was already good. Thousands of nats of real,
  epoch-structured signal are absorbed by coefficients of a few per mil, with the
  epoch-to-epoch differences at 5×10⁻⁴ rms in both windows. `preprocess.normalize`'s
  log-space knot fit left half a per mil of per-epoch continuum error unmodelled.
* Nothing else moves. The period shifts by +0.0004 / −0.0002 d (0.4σ / 0.2σ of the formal
  error, against the jitter site's 174σ relocation), K by < 0.001 km/s, the eccentricity by
  ≤ 0.0001, and the residual sd by 0.4–0.5%, so the excess scatter is not continuum-shaped.
  Window-to-window disagreement is unchanged: ΔP 2.8σ → 3.1σ,
  ΔK<sub>pre-sd</sub> 10.5σ → 10.4σ.
* The continuum is crossed off the suspect list by measurement. The surviving suspects for
  the correlated residual are the pipeline's resampling (the diagonal ivar is optimistic by
  construction), the Gaussian stand-in for FEROS's real LSF, and the Be star's variable
  disc emission. The next steps are a correlated-noise model and a wider window; the
  continuum treatment remains a hygiene term, not a fix.

Window A's baseline here reproduces the earlier record to 0.0002 d, but window B's
uniform-procedure MAP lands at P = 40.36091, 0.0093 d below the 40.37022 recorded there and
6.6× the combined formal errors, with both runs converged (parameters stationary to
~0.0002 d over the final 20 steps). With the jitter relocation (174σ) and the two-optima
period scan (125σ apart), this is the third demonstration that this surface holds optima
far outside their curvature widths, selected by the optimizer's path. Every comparison in
the table above is therefore between fits sharing one procedure.

## The AR(1) chain: whiten the residuals *and* keep the orbit (2026-08-12)

With the continuum crossed off, the pipeline's resampling correlations head the suspect
list. They are modelled as an AR(1) chain per epoch,
`C = α²·D^(−1/2)·R_φ·D^(−1/2)` (math.md §1.4a), with φ shared across epochs on the θ site
(a property of the resampling, not of one exposure) alongside the per-epoch jitters.

### Closed forms, and the trap they avoid

The correlation matrix `R_φ` has unit diagonal by construction, so heteroscedastic pixels
keep their supplied variances and the residual-sd diagnostic is blind to φ. The
discriminator is the lag-1 autocorrelation of the whitened residuals: ~φ under diagonal
whitening, ~0 under the chain whitener.

Masked pixels are treated exactly: a subset of a Markov chain is Markov, so a gap becomes
a single link with `ρ = φ^gap` (capped at build time,
`ar1_max_gap=4`; beyond it the chain restarts, which at |φ| ≤ 0.9 discards ρ < 0.66⁴ ≈ 0.2
in the worst case and ~1e-3 at the fitted values below). The precision stays tridiagonal
with closed-form `log det`, and the whitener is the innovation transform
`(ε_i − ρ_i·ε_prev)/√(1 − ρ_i²)`. All of it is pinned against a dense reference that shares
nothing with the closed forms, the chain correlation matrix built from link products and
inverted with LAPACK: marginal log-likelihood to rtol 1e-10 with gaps and jitter composed,
`φ = 0` reproducing the diagonal model to rtol 1e-12, and gradients against finite
differences including exactly at φ = 0 (`jnp.power`'s nan-gradient at a zero base; the
gap-1 links use φ directly).

Two structural costs. The direct band assembly assumes diagonal weights, so a correlated
problem selects the probe path: 2p + 1 = 347 operator applications per evaluation with
plain reverse-mode gradients, ~15× the response-site per-step cost as measured below (the
tridiagonal band-sandwich extension was deferred until AR(1) earned a permanent place; it
is built in the band-assembly section below). The chain also couples pixels across masked
gaps, so the solver bandwidth grows by a statically reserved `ar_bandwidth_extra` (the
declared-bandwidth convention; 5 and 6 model pixels on the HR 6819 windows), behind an
explicit `MarginalOrbitModel(ar1=True)`.

### Closed loop: φ and α jointly, neither poisons the orbit

Gate scale (10 epochs, SNR 130), injecting both miscalibrations at once: φ = 0.45
correlation and supplied `ivar` overstated by α² = 1.5²:

| | injected | recovered |
|---|---|---|
| φ | 0.45 | **0.4493** |
| α | 1.5 | **1.4868** |
| K errors | — | −0.05% / −0.31% |
| chain-whitened residual sd, lag-1 | 1, 0 | 0.944, −0.077 |
| same residuals, diagonal whitener: lag-1 | φ ≈ 0.45 | **+0.395** |

The 0.944 is not a miscalibration: residuals about the fitted spectra read low by
`√(1 − p_eff/N)` (math.md §3.2a, the dof effect of the jitter section; p_eff/N ≈ 0.11 at
gate scale), and the marginal's own α̂ is dof-corrected. In the last two rows, on identical
residual vectors, the diagonal whitener sees the injected correlation and the chain
removes it.

### HR 6819: the noise model closes, the orbit stays

Same uniform procedure as the response-site fits (conjunction scan, literature init,
150 L-BFGS steps), θ = orbit + hypers + 51 per-epoch jitters + shared φ; ~53–56 s/step on
the probe path (8,369 / 7,948 s per window) against ~3.6–4.2 s/step for those fits.

| | A: baseline | A: jitter | A: AR(1) | B: baseline | B: jitter | B: AR(1) |
|---|---|---|---|---|---|---|
| period [d] | 40.36566 | 40.44429 | **40.37115** | 40.36091 | 40.42979 | **40.36956** |
| K<sub>pre-sd</sub> [km/s] | 63.308 | 63.074 | **63.242** | 63.575 | 63.400 | **63.518** |
| K<sub>Be</sub> [km/s] | 1.928 | 1.450 | **2.446** | 2.946 | 2.658 | **3.756** |
| eccentricity | 0.0302 | 0.0241 | **0.0273** | 0.0240 | 0.0213 | **0.0228** |
| φ̂ | — | — | **+0.801** | — | — | **+0.694** |
| α̂ range (median) | — | 1.11–3.61 | 1.55–1.93 (1.66) | — | — | 1.27–1.58 (1.40) |
| whitened residual sd | 1.674 | 0.997 | **0.997** | 1.401 | 0.997 | **0.997** |
| whitened residual lag-1 | — | — | **+0.041** | — | — | **+0.012** |
| … diagonal whitener, lag-1 | — | — | +0.797 | — | — | +0.688 |

(Klement et al. 2025: P 40.3261 ± 0.0013, K 61.15 ± 0.88, K_Be 3.90 ± 0.27,
e 0.0289 ± 0.0058.)

* The noise model closes. Both windows whiten in both moments (sd 0.997 with lag-1 +0.041
  and +0.012), the first fits in this campaign to do so (the jitter fixed the scale and
  left the structure; the response site fixed neither). Self-consistency: the same
  residuals read through the diagonal whitener show lag-1 +0.797 and +0.688, which is φ̂
  (+0.801, +0.694) to two decimals. In ML-II terms, window A's chain model sits ~1.7×10⁵
  nats above the diagonal-jitter model (1,691,463 vs the 1,518,265 recorded at that model's
  own MAP): the correlation term accounts for most of the miscalibration.
* The jitter relocation does not recur. Under the chain the per-epoch jitters collapse from
  1.11–3.61 to 1.55–1.93, and their medians (1.66 and 1.40) reproduce the earlier residual
  sds (1.674, 1.403) closely. The diagonal jitters were fitting correlated structure,
  epoch by epoch, as per-exposure scale, and the epochs where they over-fitted it were the
  ones whose downweighting moved the period. Modelled as correlation, the excess is
  roughly uniform across epochs, a pipeline property as the resampling hypothesis
  predicts, and the period lands within 0.006/0.009 d of the diagonal baselines instead of
  0.08 d away.
* The two windows move toward each other. ΔP across the independent windows is 0.00475 d
  at the baseline and 0.00159 d under the chain, 3.0× closer, where the jitter model widened
  it to 0.0145 d. The K<sub>pre-sd</sub> spread is nearly unchanged (0.267 → 0.276 km/s).
  This is the first noise model to make the windows agree more rather than less, the
  available internal evidence that the chain is closer to the truth of these data.
* K<sub>Be</sub> moves toward the literature and remains unmeasured: 1.93 → 2.45 (A) and
  2.95 → 3.76 (B), against 3.90 ± 0.27, so window B is now within 1σ of Klement et al.
  The A–B spread grew (1.02 → 1.31 km/s) and A's value was still drifting at step 150.
  The ~1/50-linewidth reflex of the Be star (math.md §5.1's unattributable regime) does
  not yield to a noise model.
* The literature period offset survives its third noise model: 40.3712 and 40.3696 against
  40.3261, a difference of 0.044 d, marginally further than the baseline. It is measured
  not to be the continuum, the noise scale, or the pixel correlation. The surviving
  candidates are the Gaussian stand-in for FEROS's LSF, the Be disc's variability, and the
  published CCF analysis itself, which blends the components this model separates; the
  last cannot be adjudicated from here. φ̂ = 0.7–0.8 sets the scale of the first fit's
  optimism: at these correlations a diagonal model overcounts low-frequency information by
  (1+φ)/(1−φ) ≈ 5.5–9×, which is why formal errors of ±0.0005 d coexisted with window
  disagreements of 0.005 d.

## The numpyro path stops baking the problem into the graph (2026-08-12)

`MarginalOrbitModel.marginal` passes the `Problem` pytree to `jax.jit` as an argument,
because closure-captured arrays are embedded in the graph as constants, and XLA then
evaluates every θ-independent subgraph over them at compile time, against compile-time
memory (~80 GB at the design target when first measured). The numpyro path lacked that
contract: the model closure from `MarginalOrbitModel.model()` captured `self.problem`, so
`run_map`'s jitted L-BFGS step and the NUTS sample loop both compiled with the problem
baked in. The fix uses numpyro's own machinery for this case:

* the model takes the base problem as an optional argument and advertises it via a
  `model_args` attribute on the returned closure;
* `run_map` and `laplace_inverse_mass` build the potential with
  `initialize_model(..., dynamic_args=True, model_args=...)`;
* `run_nuts` runs `MCMC(..., jit_model_args=True)`, which regenerates the potential from
  the *traced* arguments inside the jitted sample loop (`hmc.py`'s `_potential_fn_gen`).

Existing call sites benefit without change (the runners resolve `model_args` as explicit
argument > model attribute > none). Calling the model with no argument, as plain numpyro
utilities such as `log_density` do, falls back to the closure, which is correct but not
compile-safe at scale; `model_args=()` selects that path explicitly.

### Measured: value+grad of the numpyro potential, the graph L-BFGS and NUTS compile

`scripts/d32_model_args_bench.py`, m5-ladder SB2 (50 epochs, p = 513), CPU, one process
per cell (the peak-working-set counter is monotone), single run per cell:

| | 31,734 px, arg | 31,734 px, closure | 74,322 px, arg | 74,322 px, closure |
|---|---|---|---|---|
| compile | **0.8 s** | 20.9 s | **0.8 s** | 10.8 s |
| constants baked into the executable | 0.02 GiB | **1.99 GiB** | 0.04 GiB | 0.04 GiB |
| process peak during compile | +0.00 GiB | **+1.35 GiB** | +0.00 GiB | +0.00 GiB |
| value+grad runtime | 6.2 s | 5.9 s | 20.6 s | 19.0 s |
| potential, gradient | identical to all printed digits | ← | identical | ← |

XLA's slow-operation alarm names the mechanism during the closure builds, on the predicted
instructions: *"Constant folding an instruction is taking
> 1s: %scatter-add.342 = f64[50,126936] scatter(...)"*, which are θ-independent
weight/response subgraphs being evaluated at compile time, at 50 × native-pixels scale.

Two qualifications:

* The memory cost is heuristic-gated. At 31.7k px XLA folded ~2 GiB of derived constants
  into the executable; at 74.3k px its folding guards declined the largest folds, so the
  memory cost did not materialize while the compile-time cost (13×) remained. Whether the
  blow-up recurs at a given scale depends on XLA's internal thresholds and version; the
  argument-passing contract removes the exposure rather than relying on the guard.
* Folded constants are marginally faster at runtime (5.9 vs 6.2 s at row 0), the trade XLA
  is designed to make. At survey scale it is the wrong one: the same mechanism costs tens
  of GB against the design target, and a NUTS warmup recompiling per mass-matrix window
  would pay the folding repeatedly.

### Regression tests (`tests/test_inference.py`)

* `test_potential_with_model_args_embeds_no_problem_constants` asserts on the jaxpr
  consts in both directions: nothing problem-sized with the problem as an argument, and
  the closure build must show the leak, confirming the probe can see it.
* `test_run_map_closure_and_argument_paths_agree` (float tolerance, different graphs) and
  `test_laplace_closure_and_argument_paths_agree` (exact, same eager ops).
* The NUTS acceptance gate now runs through `jit_model_args=True` as its default
  path, so the gate itself regression-tests the traced sample loop.

## The band assembly learns the chain (2026-08-12)

The AR(1) fits above ran on the probe path at ~15× the response-site per-step cost. AR(1)
is the only noise model that whitens both moments and the first to bring the two windows
closer together, so the tridiagonal band-sandwich extension is built; the next step (a
wider window) also cannot run on the probe path, whose gradient peaked at 23.7 GiB at the
current window scale on a 32 GB machine.

### The extension

The only stage of the direct band assembly that assumed diagonal noise is the innermost
sandwich `H = RᵀW′R`. The chain adds one symmetric cross-row term per link,
`−c·√(wₙw_p)·rₙr_p·(RₙᵀR_p + R_pᵀRₙ)`, and re-weights the diagonal by the chain
diagonal `1 + Σ a` (math.md §4.5a). Both enter through the machinery that already
carried the diagonal: a second set of static pair tables
(`operators.rebin_link_pair_tables`), built at `build_problem` time over the union
of links realized in any epoch, consumed by one extra `segment_sum` per epoch whose
traced weights carry φ, α and r; a per-epoch gap test selects each epoch's own links
against the shared tables, because masks differ by epoch. `H` widens by the group's
static `ar_step`, which is the `ar_bandwidth_extra` the bandwidth reservation already
declared, and everything downstream is unchanged: the LSF convolutions, the
T-sandwich, the band accumulation, the chunking, and the custom-VJP solve see
only a slightly wider velocity-independent band image. The likelihood's auto-selection
becomes `"band"` unconditionally; probing remains the reference implementation and
the `validate` oracle.

Exactness carries over: the dense-LAPACK gold test (gaps + jitter composed) runs the band
path at the same rtol 1e-10, band = probe at rtol 1e-12 with ∂/∂φ agreeing to 1e-9, and
the `epoch_chunk` batching is invariant. The AR weight tuple (diagonal, link, gap table)
pads and slices together, pinned by a test because a batched run that dropped link terms
would fail silently.

### Measured: the correlated marginal at HR-window scale

`scripts/d35_ar1_band_bench.py`, 51 epochs, 9,796 model px, 367,200 native px,
half-bandwidth 85, CPU, one assembly path per process (the peak-working-set counter
is monotone). Gradients in velocities, φ and the jitter, i.e. one L-BFGS step's work:

| | probe path | band path | ratio |
|---|---|---|---|
| eval, steady | 5.10 s | **0.71 s** | 7.2× |
| value+grad, steady | 58.83 s | **3.07 s** | **19.2×** |
| eval peak working set | 11.34 GiB | **1.14 GiB** | 9.9× |
| grad peak working set | 23.69 GiB | **1.85 GiB** | **12.8×** |
| log-likelihood, gradients | 1165077.330 | identical to all printed digits | — |

The gradient gap exceeds the eval gap as in the first speedup pass: the band path's solve
stage carries the closed-form custom VJP, while the probe path pays plain reverse mode
through 2p + 1 = 347 operator applications. At 1.85 GiB the wider window fits where
23.7 GiB was already pressing against the machine.

### The proof on real data: window A, refit unchanged

`scripts/hr6819_ar1_run.py --windows A`, rerun with no changes beyond the assembly:
868 s where the probe path took 8,369 (9.6× end-to-end, 5.8 vs 55.8 s/step), converging
to the same optimum: log-likelihood 1,691,463.5 to the printed digit, φ̂ +0.801,
P 40.37113 vs 40.37115 (0.02 formal σ), α range 1.55–1.93 (median 1.66) and residual
diagnostics (chain sd/lag-1 0.997/+0.041, diagonal lag-1 +0.797) identical.
K<sub>Be</sub> reads 2.420 vs 2.446, the direction both runs were still sliding along at
step 150: the flattest axis of the surface, not a path discrepancy.

## One wide window: 4120–4600 Å, Hγ masked (2026-08-12)

Runnable only because of the band assembly, `scripts/hr6819_wide_run.py` joins windows A
and B into a single fit of 22,169 model px and ~765k good native pixels, 2.26× window
A, including the 25 Å strip 4355–4380 not previously fitted. Hγ's core
(4325–4355 Å) is masked by `preprocess.mask_ranges`: ivar = 0 keeps the sampling
regular, the AR(1) chain restarts across the hole (a masked gap beyond
`ar1_max_gap`), the bandwidth is unchanged, and the broad absorption wings, which are
static stellar features, stay in. Noise model and procedure are the AR(1) configuration.
200 L-BFGS steps, 2,447 s at 12.2 s/step, linear in pixels from the window A refit
(5.8 s/step at 0.44× the size); the probe path would have cost roughly 7 hours against
a ~50 GB gradient, which this machine does not have.

| | A: AR(1) | B: AR(1) | **wide: AR(1)** | Klement et al. 2025 |
|---|---|---|---|---|
| period [d] | 40.37115 | 40.36956 | **40.36750** | 40.3261 ± 0.0013 |
| K<sub>pre-sd</sub> [km/s] | 63.242 | 63.518 | **63.396** | 61.15 ± 0.88 |
| K<sub>Be</sub> [km/s] | 2.446 | 3.756 | **3.482** | 3.90 ± 0.27 |
| eccentricity | 0.0273 | 0.0228 | **0.0261** | 0.0289 ± 0.0058 |
| φ̂ | +0.801 | +0.694 | **+0.737** | — |
| α̂ range (median) | 1.55–1.93 (1.66) | 1.27–1.58 (1.40) | 1.46–1.86 (1.61) | — |
| whitened residual sd, lag-1 | 0.997, +0.041 | 0.997, +0.012 | **0.997, +0.019** | 1, 0 |
| … diagonal whitener, lag-1 | +0.797 | +0.688 | +0.731 | — |

* The noise model closes at 2.3× the data: sd 0.997 with lag-1 +0.019, and the
  self-consistency check holds, the diagonal whitener reading +0.731 against
  φ̂ = +0.737. φ̂ and the α̂ range sit between the two single-window values, as expected of
  a pipeline property that varies mildly with wavelength.
* K<sub>Be</sub> moves toward the literature: 3.48 against 3.90 ± 0.27 (1.5σ), where
  window A alone gave 2.42. The ~1/50-linewidth reflex is the data-starved direction
  (math.md §5.1), and it responds to more lines in one joint constraint.
* The eccentricity is unchanged at 0.0261, 0.5σ from the published 0.0289 ± 0.0058.
* The period lands below both single-window optima, 40.3675 against 40.3711/40.3696,
  outside their interval. A joint fit is not an average of window MAPs (the
  cross-window coupling and the new pixels carry information), and the standing
  multimodality observation applies to any single optimum. The value moves toward the
  literature and remains 0.041 d away.

The literature period offset survives its fourth configuration. Across two independent
windows, three noise models, and one joint wide fit, P ∈ [40.3675, 40.3712], internally
consistent to 0.004 d, against a published 40.3261 ± 0.0013, with K<sub>pre-sd</sub> at
63.2–63.5, consistently 3.7–4% above the CCF value, in the direction deblending predicts.
It is measured not to be the continuum, the noise scale, the pixel correlation, or the
window choice. The surviving candidates are the Gaussian stand-in for FEROS's real LSF (a
tabulated LSF is a banded-operator swap, reserved by design), the Be disc's variability,
and the published CCF analysis itself, which blends the components this model separates.

## The tabulated-LSF seam opened: fitted σ(λ), and the orbit's answer (2026-08-12)

The design reserved this seam ("tabulated LSF is v2 — a banded matrix, no structural
change"). The kernel slot becomes a per-pixel profile bank realized
from per-anchor kernels through static log-λ interpolation tables
(`operators.convolve_varying`, exact adjoint pair, arbitrary asymmetric banks
accepted); the band assembly keeps its structure with scalar taps replaced by
row-shifted profile columns and the second sandwich application run against the
band-transpose of the first (G = Kᵀ(KᵀH)ᵀ, by G's symmetry, since only left
applications broadcast on a row-major band image). Band == probe == dense at
rtol 1e-12/1e-10 under diagonal and AR(1) noise, per-anchor width gradients to
1e-9, random asymmetric banks pinned against a hidden kernel flip. 320 tests.

The closed loop (gate scale, injected σ ramp 5.0→9.5 km/s): the joint fit leaves the
orbit unbiased (K to 0.3%) and recovers the ramp's direction, but the marginal does not
prefer the injected truth. A flat width exceeded it by ~3 nats, and the ML profile
exceeded the truth by ~8 while sitting ~3 km/s off one anchor. A stationary kernel change
commutes with the shifts, so the free spectra absorb it (deconvolution), and the width
preference is dominated by the smoothness prior rather than by the instrument. Only the
anchor-to-anchor variation is data-identified, through the epoch-dependent shifts. Fitted
anchor widths are therefore diagnostics, not measurements; the readout is the orbit's
response, not σ̂(λ) itself.

HR 6819 (`scripts/hr6819_lsf_run.py`): the wide-window configuration (4120–4600 Å, Hγ
core masked, per-epoch jitters + shared AR(1) φ) plus 13 Gaussian width anchors every
40 Å (the FEROS order scale), bounds 1.5–3.5 km/s around the nominal 2.652 (radius from
the bound: half-bandwidth 91 vs 87). 200 L-BFGS steps, 5,650 s at 28.3 s/step (2.3× the
wide-window fit: larger radius, varying-kernel band stages, 13-anchor VJP).

| | fixed σ = 2.652 | **fitted σ(λ) ×13** | Klement et al. 2025 |
|---|---|---|---|
| period [d] | 40.36750 | **40.36769** | 40.3261 ± 0.0013 |
| K<sub>pre-sd</sub> [km/s] | 63.396 | **63.395** | 61.15 ± 0.88 |
| eccentricity | 0.0261 | **0.0262** | 0.0289 ± 0.0058 |
| φ̂ | +0.737 | **+0.737** | — |
| α̂ range (median) | 1.46–1.86 (1.61) | 1.46–1.86 (1.61) | — |
| whitened residual sd, lag-1 | 0.997, +0.019 | 0.998, +0.019 | 1, 0 |
| log-likelihood | 3,388,604.2 | **3,388,694.7** | — |

σ̂(λ) at the anchors [km/s]: 2.15 at 4120 Å, then 3.1–3.4 across the rest of the
window, close to the 3.5 bound. As in the closed loop, the marginal gains smoother implied
spectra with broader kernels, so the absolute level is bound-limited and diagnostic only.
K<sub>Be</sub> is omitted from the table because it did not converge in 200 steps: it
oscillated 1.30–4.16 across the last 100 steps (a band that spans the fixed-σ value 3.48),
ending at 1.30 with |grad| 59 where that fit ended at 3.85. The 13 near-flat width
directions slow the already-flattest axis; every tabulated quantity above was pinned over
the same trajectory (P within ±0.001, K₁ within ±0.02, φ̂ to three digits).

The fitted width profile absorbs +90.5 nats, so the effective width has wavelength
structure, and the orbit does not respond: P +0.0002 d (0.5% of the offset, within the
trajectory wobble), K₁ −0.001 km/s, e +0.0001, φ̂ and every residual moment unchanged, the
response site's pattern (+4,100 nats absorbed, nothing moved) repeated for the LSF. The
literature period offset survives its fifth configuration, and σ(λ) joins the exonerated
list. The surviving LSF candidate is profile asymmetry, the first-order centroid channel,
whose epoch-coupled part enters as an apparent velocity perturbation ∝ λc′(λ)v(t)/c
(math.md §1.3); the operator already accepts arbitrary banks, so only a
θ-parameterization (e.g. per-anchor Gauss–Hermite h₃) would be new. Beyond the LSF, the
candidates are disc variability and the published CCF blending itself.

## The asymmetry lever, and the LSF exonerated in full (2026-08-13)

Profile asymmetry, the first-order centroid effect a symmetric kernel cannot produce, is
parameterized by a per-anchor Gauss–Hermite h₃ (`operators.gauss_hermite_kernel_traced`,
|h₃| ≤ 0.2, h₃ = 0 bit-identical to the Gaussian machinery) behind an `lsf_h3` site.

The closed-loop identifiability is sharper than in the width case: an injected h₃ ramp of
∓0.12 came back flat (fitted |h₃| ≤ 0.03) with the orbit recovered to 1%. A free spectrum
represents any static centroid-warp field c(λ) ≈ √3·h₃(λ)·σ outright, so the
data-identified remainder is only the epoch-coupled sampling of the warp's gradient,
Δc ≈ c′(λ)·λ·(v − v_bary)/c ≈ 30 m/s at this configuration (math.md §1.3), two
orders below the ~4 km/s of accumulated RV signature the 0.041 d offset represents; this
estimate bounds out the instrument-frame per-epoch kernel realization. The fixed-spectra
data term does prefer the injected profile, so the injection is real and detected;
band == probe == dense with h₃ anchors under diagonal and AR(1) noise; gradients in h₃ to
1e-9. 329 tests.

HR 6819 (`scripts/hr6819_h3_run.py`): the fitted-σ(λ) configuration plus 13 free h₃
anchors, 26 LSF parameters joint with the orbit and the AR(1) noise model. 300
L-BFGS steps (|grad| 44 at the end, better converged than that 200-step run),
10,789 s at 36.0 s/step.

| | fixed LSF | fitted σ(λ) | **fitted σ, h₃** | Klement et al. 2025 |
|---|---|---|---|---|
| period [d] | 40.36750 | 40.36769 | **40.36719** | 40.3261 ± 0.0013 |
| K<sub>pre-sd</sub> [km/s] | 63.396 | 63.395 | **63.391** | 61.15 ± 0.88 |
| eccentricity | 0.0261 | 0.0262 | **0.0261** | 0.0289 ± 0.0058 |
| φ̂ | +0.737 | +0.737 | **+0.737** | — |
| whitened residual sd, lag-1 | 0.997, +0.019 | 0.998, +0.019 | **0.998, +0.020** | 1, 0 |
| log-likelihood | 3,388,604.2 | 3,388,694.7 | **3,388,721.3** | — |

ĥ₃(λ) at the anchors: interior anchors at the |h₃| ≤ 0.02 level, the largest
values 0.042–0.052 at three anchors including the data-starved blue edge, which
imply centroid shifts of −0.13 to +0.31 km/s, a 0.53 km/s spread, all diagnostics by the
closed-loop measurement. +26.6 nats over the width fit for 13 parameters, an order below
the widths' +90.5: asymmetry has far less to absorb once the spectra are free.
K<sub>Be</sub> again did not settle on its flat axis (2.73 at |grad| 44, inside the
1.3–4.2 band the width run wandered); every tabulated quantity above was pinned.

The LSF is exonerated in full and the offset survives its sixth configuration. P moved
−0.0005 d from the width fit, inside its own trajectory wobble, and K₁, e, φ̂, α̂, and both
residual moments are unchanged to the last digit. Every instrumental channel this model
can express has been given a θ-site and measured against the orbit: the continuum
(+4.1k nats), the noise scale, the pixel correlation (+1.7e5 nats), the window choice, the
LSF width (+90.5 nats), and the LSF asymmetry (+26.6 nats). None moved the period. Across
all six configurations P ∈ [40.3672, 40.3712], internally consistent to 0.004 d, against a
published 40.3261 ± 0.0013. The surviving candidates are not instrumental: the Be disc's
variability (a time-variable component this static-spectrum model cannot express, and a
systematic of the published analysis as well), and the published CCF blending itself,
which measures velocities on composite line profiles this model separates. The
instrumental-systematics campaign on this dataset is complete.


---

## The nebular component, and per-pixel prior strengths (2026-08-13)

From `tests/test_nebular.py` and `examples/04_nebular.py`: one SB2 in an H II region, 12
epochs, SNR 220, 540 model pixels over 4838-4886 A, K = (58, 41) km/s, light fractions
(0.7, 0.3), both stars carrying a broad Hbeta absorption (true composite depth -0.506,
EW 1.911 A), and a static nebular Hbeta emission line of peak 0.45 whose amplitude
varies +-30% per epoch with a factor of ~2 between the best and worst night.

### Exactness

| Check | Result |
|---|---|
| forward model vs. the simulator's injection, barycentric **and** topocentric | atol **1e-12** |
| `with_velocities` + `with_light_fractions` vs. a fresh `build_problem` | rtol **1e-14** (the nebular and telluric columns are carried, not rebuilt) |
| `with_nebular_amplitudes` vs. a fresh `build_problem` | bit-identical |
| band assembly vs. the matrix-free operator, nebular column + window profile | `validate=True`, rel err < 1e-10 |
| band vs. probe assembly, same configuration | log-likelihood rtol **1e-11**, spectra atol 1e-9 |
| per-pixel prior `apply` / `dense` / `prior_logdet` vs. dense NumPy | rtol **1e-12 / 1e-10** |
| uniform profile vs. the unprofiled prior | rtol 1e-14 (`apply`, `dense`, `prior_logdet`) |
| d(log L)/d(log_nebular_amp) vs. central differences | < 1e-4 relative; the gradient sums to zero, as centering requires |

The determinant recursion is the component most exposed to a silent error:
`prior_logdet` is an O(P) scalar Cholesky over the pentadiagonal prior, and per-pixel
`tau` and `eta` change all three of its diagonals. It is checked against `slogdet` of the
dense construction with random profiles spanning 0.2-40 in curvature and 0.1-1e4 in ridge,
not against a uniform special case.

### What the contamination costs the spectra (orbit held at truth)

Two disentanglings of the same data with identical stellar priors, differing only in
whether the nebular component exists, with the orbit fixed at the injected values.

| | truth | no nebular component | **with the component** |
|---|---|---|---|
| Hbeta core depth (light-weighted composite) | -0.506 | -0.375 (**26% shallower**) | **-0.508** |
| mean core error | 0 | **+0.154** | **+0.0015** |
| core RMS error | 0 | 0.155 | **0.0057** |
| Hbeta equivalent width [A] | 1.911 | 1.690 (**-11.5%**) | **1.908 (-0.14%)** |
| marginal log-likelihood | — | reference | **+81,424 nats** |

Both log-likelihoods are marginal, with the component spectra integrated out and the
Occam terms included, so their difference is a Bayes factor rather than a fit-quality
score: the extra component costs likelihood unless coherent signal pays for it, and
here it is paid 8.1e4 times over.

Equivalent width is the quantity that reaches the atmosphere code. An 11.5% error in a
Balmer EW is a large, systematic error in log g, and nothing in the current literature
propagates it: the disentangled spectra arrive at the next stage without an uncertainty.

`examples/04_nebular.py` adds the third treatment used in the literature, masking the
contaminated pixels (`ivar = 0` over +-150 km/s). It is defensible but has a cost: with
the core deleted there is nothing behind those pixels but the prior, so the composite comes
back flat there and the product is incomplete at exactly the wavelengths where a Balmer
gravity diagnostic is read. The three-way comparison takes 9 seconds to produce.

### What the contamination costs the *orbit* (joint MAP, cold start)

Same data, same priors, same starting point, 300 L-BFGS steps of ML-II MAP over the
orbit and the hyperparameters (plus the 12 log-amplitudes when the component exists).

| | truth | nebular-blind fit | **with the component** |
|---|---|---|---|
| K<sub>1</sub> [km/s] | 58.0 | 57.38 (-1.1%) | **57.91 (-0.15%)** |
| K<sub>2</sub> [km/s] | 41.0 | **16.77 (-59.1%)** | **40.88 (-0.29%)** |
| period [d] | 5.70000 | **5.87115 (+0.171)** | **5.69986 (-0.00014)** |
| eccentricity | 0 | **0.950 — the solver's clip** | **0.0022** |
| potential at the end | — | +30,629 | **-19,220** |
| gradient norm at the end | — | 2.9e4 (still wandering) | 8.3 (settled) |
| wall time | — | 177 s | 49 s |

A static line is a component with K = 0, so a model with nowhere else to put it uses
whichever stellar component can be made to move least: the secondary's semi-amplitude
collapses by 59%, and the period and eccentricity follow, giving a circular orbit
reported at *e* = 0.95, which is the eccentricity clip rather than a fit. Only
K<sub>1</sub> survives, because 70% of the light pins it. The blind fit is still
wandering at 300 steps where the modelled one has settled, and takes 3.6x the wall time.
(Neither sets `MAPResult.converged`: that flag tests an absolute gradient-norm tolerance
that is unreachable at these pixel counts, as on HR 6819. The three orders of magnitude
between the two gradient norms is the readable statement.)

The per-epoch amplitudes come back with correlation 0.99930 against the injected ones
and 0.0066 rms in log, against an injected spread of 0.78x to 1.50x. They are compared
after centering, because only `a_j * d_neb` is observable and the geometric mean is a
convention (math.md §1.3). ML-II independently keeps the nebular component less smooth
than the stellar ones (log tau 7.9 against 11.4), recovering a shape the prior was given
no information about.

The window profile is not cosmetic. The same joint fit with the nebular component free
across the whole grid lands K<sub>2</sub> at +2.6% instead of -0.29%, with the potential
250 nats worse: the freedom the profile removes was spent absorbing stellar signal at
wavelengths where a nebula has no lines. Measuring it also surfaced a defect:
`MarginalOrbitModel` rebuilt the prior from the sampled `log_tau`/`log_eta` and dropped
the profiles, so a windowed component was silently un-confined as soon as ML-II was
switched on. The profiles are structure, the scalars are hyperparameters, and the merge
now respects that (math.md §2).

### Readings

The failure mode is worse than the literature describes, and the fix is inexpensive.
The published concern is line-profile narrowing and biased atmospheric parameters, which
is real (-11.5% in EW). The contamination also propagates into the dynamical answer, the
masses, through a 59% error in K<sub>2</sub>. Both are removed by one extra component and
twelve extra parameters, at 49 s against the blind fit's 177 s.

Nothing downstream had to change. The nebular column is one more column of A with a
different velocity law and a free amplitude, so the band assembly, the AR(1) link tables,
the chunking policy, the custom-VJP solve, and the bandwidth contract are unchanged; the
per-pixel prior generalizes three diagonals and keeps the same O(P) determinant recursion.
By the same property of the linear-Gaussian family, a general time-variable component, of
which this is the rank-one case, can be a change of basis rather than a change of method.

Two degeneracies are closed by convention, not by data (math.md §5.4): the amplitude
scale, pinned by centering the log-amplitudes, and the nebular velocity, which decides
where the component's lines land on the model grid and is not a measurement.



---

## Calibrated faint-companion detection (2026-08-13)

Three pieces: vectorize the scan, marginalize K₁, and calibrate the statistic by injection
and recovery. From `tests/test_calibrate.py` and `examples/05_detection_limit.py`: one
SB1/SB2 pair (14 epochs, SNR 200, 717 model pixels over 5000-5060 A, 520 native pixels,
K = (55, 40) km/s, light fractions (0.93, 0.07), P = 7.3 d, e = 0.12) scanned on a
20-point K₂ grid from 14 to 71 km/s.

### The vectorized sweep

One batched `lax.map` over the trial grid, against the Python loop it replaces (one
jitted call and one device synchronization per point). Best of three, shared machine.

| model pixels | native | epochs | half-bandwidth | loop / point | sweep / point | speedup | relative agreement |
|---|---|---|---|---|---|---|---|
| 201 | 100 | 8 | 46 | 3.71 ms | 1.33 ms | **2.8x** | 1.1e-12 |
| 717 | 520 | 14 | 55 | 17.34 ms | 8.66 ms | **2.0x** | 1.4e-13 |
| 2,652 | 2,150 | 20 | 66 | 97.41 ms | 43.91 ms | **2.2x** | 1.8e-16 |

The factor is near-flat in problem size, so the gain comes from the batching rather than
from removing per-point dispatch (a dispatch-overhead explanation would predict the
opposite); the acceptance gate asserts only 1.5x. It is not bit-identical to the loop:
batching re-associates the linear algebra, and the log-likelihoods move in the last few
digits. It enables the two features built on top: a 7x20 (K₁, K₂) grid costs 0.88 s where
the loop would need ~1.41 s, and the 450-scan calibration below, 9,450 marginal solves,
costs 53 s.

### Marginalizing K₁ against assuming a wrong one

The literature reports that a small error in the assumed primary semi-amplitude puts
spurious features in the recovered secondary spectrum. The unreported, more consequential
effect is on the detection statistic.

| K₁ treatment | K₂ peak [km/s] | companion line-pattern correlation | D at the peak |
|---|---|---|---|
| correct, fixed | 41 | **0.961** | 40,609 |
| 5% high (57.75), fixed | 38 — one grid step low | 0.720 | **66,837** |
| 5% high, marginalized (σ = 5%) | **41** | **0.955** | 40,455 |
| 10% high (60.5), fixed | 41 | **0.486** | **135,410** |
| 10% high, marginalized (σ = 10%) | **41** | **0.931** | 41,310 |

A wrong K₁ makes the detection look stronger while the answer gets worse. Unremoved
primary signal is coherent across epochs, and the companion's free spectrum is the only
thing that can absorb it, so D more than triples while the recovered spectrum correlates
0.49 with truth. Marginalizing over a Gauss-Hermite rule on N(μ₁, σ₁²), applied to the
companion model and the no-companion model so that D stays a ratio of two marginal
likelihoods, recovers 0.93-0.96 and puts D back where the correct K₁ has it. Correlations
are offset-removed: at ℓ₂ = 0.07 the companion's smooth envelope is prior-dominated
(math.md §5.1-5.2), so the line pattern carries the information.

The 7-node rule costs about 30% more wall time than the fixed scan (2.2 s against 1.7 s),
not 7x, because the trials share one compiled graph.

### The calibrated limit

200 companion-free trials for the null distribution, 50 trials at each of six injected
light fractions for completeness, all resimulated through the observed dataset's own
operators. 450 full scans in **71 s**.

| | |
|---|---|
| null peak D | min -776.6, median -730.1, max **-676.7** |
| threshold at 1% false alarm | **-692.2** |
| realized null exceedance | 0.0050 (budget 0.01) |
| resolution floor, 1/(N+1) | 0.0050 |
| the real SB2's peak | D = 40,609 at K₂ = 41, FAP at the floor |

| injected ℓ₂ | 0.05% | 0.10% | 0.15% | 0.20% | 0.30% | 0.50% |
|---|---|---|---|---|---|---|
| detected | 0.00 | 0.10 | 0.18 | 0.36 | 0.96 | 1.00 |
| median D | -730.8 | -723.1 | -714.9 | -705.5 | -653.1 | -508.7 |

> Any companion contributing more than **0.30%** of the light would have been detected at
> 95% confidence, against a detection threshold D > -692.2 set at a 1% false-alarm
> probability from 200 companion-free trials.

The null peaks are strictly negative: the marginal likelihood charges an Occam term for
the companion's free spectrum, and with nothing to find, nothing pays for it. "D > 0"
would have been a conservative test on this dataset; on another it need not be, hence the
measured threshold.

Two properties are enforced by construction. The threshold is defined through the
false-alarm estimator (1 + #{null >= D})/(N+1) rather than as a sample quantile:
`np.quantile` interpolates between order statistics and was measured leaving 8.3% of the
null above a nominal 5% threshold on a 24-trial run (caught by a test), anti-conservative
in the direction that matters for a detection claim. No FAP below 1/(N+1) is reported;
below that the rule degrades to "must exceed every null trial".

### One expected dependence that is not there

| injected K₂ [km/s] | 20 | 40 | 65 |
|---|---|---|---|
| limit on ℓ₂ | 0.292% | 0.296% | 0.297% |

The limit is flat in K₂. In an SB2 the components move in antiphase, so their relative
velocity never falls below about K₁, and at K₁ = 55 km/s the pair is well separated at
every trial K₂. A K₂ dependence should appear only when K₁ is small enough that the pair
is barely resolved at any phase.

None of this checks the model. The null trials are drawn at the same K₁, orbit and light
fractions the scan assumes, so the threshold is self-consistent with those assumptions and
insensitive to their being wrong; the K₁ table above is that failure, and no calibrated
threshold would have flagged it. The limit is also conditional on the assumed companion
template: the observable is ℓ₂·d₂, and a featureless companion is invisible at any light
fraction. Both conditions are stated wherever the numbers are.

---

## The free per-epoch radial-velocity table (2026-08-13)

No Keplerian: every epoch's velocity is its own parameter. From
`tests/test_velocity_table.py`. One SB2: 10 epochs, SNR 200, 400 model pixels at
dv = 6.00 km/s over 5000-5040 A (284 native pixels), K = (30, 55) km/s, light fractions
(0.6, 0.4), P = 6.31 d, e = 0.15.

### The zero point, and why it is removed

A free table has one arbitrary zero point per component: with no orbit tying the
stars together, each free spectrum absorbs a constant added to its own shifts. The
equality `T(d + D) x = T(d) [T(D) x]` is exact only for whole-pixel `D`, because the
model shifts by linear interpolation and a fractional shift blurs as well as translates.

| common shift applied to one component | change in log-likelihood |
|---|---|
| 1.00 model pixel | **4e-9 relative** (boundary effects only) |
| 0.10 model pixel | -7.3 nats |
| 0.01 model pixel | -0.11 nats |

An uncentered table's absolute level is therefore pinned by interpolation error rather
than by data: it would resemble a systemic velocity, move when the grid is resampled, and
carry no information. albireo centers the pixel shifts per component, which makes the
likelihood exactly invariant:

| offset added to one component (relativistic addition) | change in log-likelihood |
|---|---|
| 5 / 50 / 200 km/s | **0.000e+00**, exactly |
| 0.5 km/s | -9.3e-10 (relative 9.8e-14, float64 round-off) |

Centering in velocity space instead is correct only to `O(v^2/c^2)` and leaves a residual
four to six orders of magnitude larger (-9.9e-8 at 0.5 km/s, +8.7e-6 at 50 km/s), measured
in the suite.

### Recovery, by starting point

250 L-BFGS steps of ML-II MAP over the 20 velocities and the four hyperparameters.

| start | per-epoch RV rms [km/s] | Wilson slope | potential |
|---|---|---|---|
| the Keplerian truth | 0.098 / 0.066 | -1.8255 | -9627.1 |
| K wrong by 10% | 0.106 / 0.063 | -1.8300 | -9627.6 |
| **K wrong by 30%** | **0.098 / 0.066** | **-1.8255** | **-9627.1** |
| truth + 15 km/s noise | 0.098 / 0.066 | -1.8255 | -9627.1 |
| cold start (every epoch at 0) | **12.94 / 28.70** | **+0.5916** | **+112,692** |

Truth is -1.8333 = -K2/K1, so the recovered mass ratio is 0.4% off. An rms of 0.098 km/s
is 1/60th of a model pixel. Every warm start reaches the same optimum to four decimals,
including one 30% wrong in both semi-amplitudes.

The cold start fails: with every epoch at one velocity the two components are
indistinguishable, and the mode is documented as needing a warm start. The failure is
visible: 122,000 nats worse, with a Wilson slope of the wrong sign.

### Uncertainties — and the trap in reading them

| | mean sigma [km/s] | rms error [km/s] | error / sigma |
|---|---|---|---|
| raw Laplace diagonal | **37.95** | 0.098 / 0.066 | 0.002-0.26 |
| zero points projected out | **0.059** | 0.098 / 0.066 | 1.44 |

The raw number is `120/sqrt(10)`, the `Normal(0, 120)` prior divided by the epoch count,
identical to four digits across both components and all ten epochs: the signature of a
flat direction, whose posterior width is the prior's and which every epoch's marginal
variance inherits. It is 640x too large, and would take the same value on a dataset that
constrained nothing. `relative_velocity_errors` projects each component's mean out; the
projected block then has exactly 2 zero eigenvalues, one per component, confirming the
identifiability claim numerically.

The projected bars run ~1.4x optimistic against the realized errors, as expected of a
Laplace approximation with the hyperparameters pinned at their MAP values. Posterior
samples of the `velocity_rel` deterministic need no projection and no Gaussian assumption
and are the recommended route; this is the fast estimate.

Per-epoch precision of 0.059 km/s is 1/102 of a model pixel.

### The model check

`keplerian_residuals` centers both tables the same way and differences them in pixel
space, so the two arbitrary zero points cancel exactly (verified: offsetting the recovered
table by +77 and -31 km/s moves the residuals by < 1e-9).

| Keplerian tested against the recovered table | max abs residual | in units of the per-epoch sigma |
|---|---|---|
| the orbit that generated the data | 0.164 km/s | 2.8 |
| period wrong by 0.5% | 2.979 km/s | 50 |
| K_2 wrong by 5% | 2.581 km/s | 44 |

A Keplerian is a strong constraint; a table fitted without one indicates whether the data
support it.

---

## Second speedup pass: the assembly's reverse pass, and four dead ends (2026-08-15)

Same harness as the scale and speedup ladders above, on [the machine](#the-machine),
measured at 66.8 GB/s streaming (triad) and 1188 GFLOP/s fp64 `dgemm` at n = 2000. The
earlier tables ran under an unrecorded software stack, so absolute numbers should not be
read across that boundary; every before/after pair below was measured back to back and is
valid.

### The old attribution had expired

The first speedup pass's 92%-probing attribution no longer applies, since that pass
removed probing from the hot path. Re-measured at the ladder's first row (31,734 model px,
SB2, 50 epochs, p = 513), jitted, stage by stage:

| stage | s | % |
|---|---|---|
| band assembly (incl. band to block packing) | 1.94 | 65.2 |
| block Cholesky | 0.93 | 31.4 |
| solves + quadratic form | 0.075 | 2.5 |
| `A^T W z` | 0.025 | 0.8 |
| log-determinants, weighted data terms | 0.003 | 0.1 |
| **whole marginal (jitted)** | **2.97** | |
| **gradient in the velocities** | **10.23** | 3.45x eval |

Inside the assembly, 88% is the epoch band scan, and that splits almost exactly in
half: 0.80 s for the velocity-independent `G = K^T H K` pre-pass and 0.76 s for
the T-sandwich accumulation.

### The gradient is a different problem from the evaluation

NUTS runs the gradient ~2,600 times per posterior and the evaluation once, so the
gradient is what sets wall time. Splitting it:

| stage | value | gradient | backward only | ratio |
|---|---|---|---|---|
| band assembly | 1.91 | **8.27** | 6.37 | 4.34x |
| solve stage (custom VJP) | 0.98 | 1.68 | 0.70 | 1.71x |
| full marginal | 3.01 | 10.15 | 7.15 | 3.38x |

82% of the gradient is assembly, whose backward costs 3.3x its own forward, while the
solve stage, which the custom VJP was built for, is 16%. Of the 6.37 s, 5.94 s is in the
epoch band scan and 0.20 s in the band-to-block packing.

### Change 1: the band accumulate is the identity, and reverse mode did not know

Each epoch adds its (i, j) block into the global band tensor as

    band_out = dus(band, ds(band, idx) + f, idx)

which is `band + place(f, idx)`, the identity in `band`. Reverse mode transposes
the `dynamic_update_slice` and the `dynamic_slice` separately and rebuilds that
identity as

    band_bar = dus(out_bar, 0, idx) + dus(zeros_like(band), ds(out_bar, idx), idx)

which is algebraically `out_bar`, but three passes over the whole 522 MB band tensor,
once per (i, j) block per epoch. That is 4 x 50 x 3 x 522 MB = 313 GB of memory
traffic to reproduce an input, and it scales with the band tensor rather than with
the slice touched. `assembly._band_accumulate` is a `custom_vjp` whose reverse
rule is the closed form: the operand cotangent is the output cotangent, and `f`'s is
the slice of it. In an isolated harness at this scale:

| | forward | gradient | backward |
|---|---|---|---|
| nested dynamic slices | 0.390 s | 5.615 s | 5.224 s |
| closed-form `custom_vjp` | 0.370 s | **0.840 s** | **0.470 s** |

with values and gradients bit-identical.

A forward-only profile would have passed over this line, because the ablation inverts:
deleting the band accumulate entirely makes the forward 24% slower (0.98 s against
0.79 s) while making the backward 2.7x faster.

### Change 2: `G`'s second kernel application translates columns only

`G = K^T H K` was two unrolled accumulations over the 2r+1 kernel taps. The first
shifts rows, so it stays a loop. The second does not: it adds `kernel[s] * u` at
column offset `2r - s`, which makes it a contraction against a static `(w_u, w_g)`
banded matrix, `Q[k2, k] = kernel[k2 + 2r - k]`. The loop form re-read and rewrote the
widest image in the assembly once per tap:

| stage | s |
|---|---|
| first (row-shifting) loop | 0.327 |
| second loop, as written | 0.548 |
| second stage as one contraction | **0.030** |

18x on that stage, 2.45x on the whole `G` pre-pass, and no extra memory. The two
applications also compose into a single `((2r+1) w_h, w_g)` map, so both can be one
contraction, but that needs a 37 MB neighbourhood stack per epoch, 1.9 GB across a
hoisted 50-epoch pre-pass (the kind of vmapped intermediate the memory pass removed), and
measured 0.32 s against 0.36 s. Ten percent for 1.9 GB was declined.

### What the two changes bought

The full design-target ladder (`scripts/m5_scale_bench.py`, same seeds, same machine,
committed code stashed and re-run for the "before" column, not carried over from the
earlier tables):

| n (model px) | eval before | eval after | | ∇ before | ∇ after | |
|---|---|---|---|---|---|---|
| 31,734 | 2.71 s | **2.16 s** | 1.25x | 10.07 s | **5.52 s** | **1.82x** |
| 74,322 | 6.45 s | **5.24 s** | 1.23x | 27.98 s | **15.25 s** | **1.83x** |
| 135,052 | 12.64 s | **10.15 s** | 1.25x | 53.02 s | **29.09 s** | **1.82x** |
| **203,440 (design target)** | 19.11 s | **15.67 s** | 1.22x | 80.23 s | **45.88 s** | **1.75x** |

The ratios are flat in problem size, 1.22–1.25x on an evaluation and 1.75–1.83x on a
gradient, as both changes predict: each removes a fixed multiple of the per-epoch band
traffic, which is linear in `n`. A design-target gradient lands at 46 s, and the
gradient/evaluation ratio falls from 4.2x to 2.9x.

Stage by stage at row 0:

| | before | after | |
|---|---|---|---|
| band assembly | 1.94 s | 1.38 s | 1.41x |
| block Cholesky | 0.93 s | 0.72 s | — (unchanged; run-to-run spread) |
| whole marginal evaluation | 2.97 s | 2.20 s | 1.35x |
| gradient in the velocities | 10.23 s | 5.19 s | 1.97x |

Repeat runs of the whole-marginal figure vary by about 10% on this machine, so the ladder
above is the number to quote; the gradient ratio is stable.

### Exactness

The log-likelihood and its gradient are bit-identical before and after, compared as raw
IEEE-754 hex rather than to a tolerance, in all three model variants: stationary LSF,
AR(1) correlated noise, and wavelength-dependent LSF. The Hessian moves by 4e-13
relative, which is reassociation in the second-order path.

Bit-identity is what was measured, not what is guaranteed. The contraction of change 2
promises equality only up to summation order, like the rest of the band assembly:
increasing `k2` is increasing `s`, so the two ideal orders coincide, but XLA is free to
block a GEMM's accumulation as it chooses. Against a random kernel it differs from the
loop by 0.5 ulp. The code comments make the weaker claim.

The earlier second-order re-entry defect recurred and its regression test caught it: a
`custom_vjp` whose forward rule calls the custom function itself is first-order exact but
returns the transpose of the true Hessian, and
`test_second_order_reverse_matches_plain_autodiff` failed on the first attempt. The fix is
the same: inline the primal in the forward rule.

### The three-way head-to-head is now hardware-bound, and needs re-running

On the fd3 benchmark's configuration (4,444 px, 20 epochs, `b_nat` = 63) this pass takes
albireo's steady state from 0.071 s to 0.059 s, 1.20x, in line with the ladder. That is
the only part of the comparison measured with one change.

The recorded comparison no longer reads the same way, mostly not because of this pass:

| | recorded earlier | this machine |
|---|---|---|
| albireo, committed code | 0.182 s | 0.071 s |
| albireo, with this pass | — | **0.059 s** |
| fd3 (rebuilt binary, WSL, min of five) | 0.111 s | 0.099 s |
| shift-and-add, 7 sweeps | 0.018 s | 0.049 s — **does not reproduce** |

fd3 moved 12% and albireo 2.6x between the two measurements, with no hardware change (see
[The machine](#the-machine)). fd3 is a single-threaded C program and albireo's XLA uses all
32 threads. In this measurement albireo is ~1.7x faster than fd3, where the record says
1.64x slower.

The three-way table above was not updated from this partial recheck: the shift-and-add
figure did not reproduce (0.049 s against 0.018 s recorded, while everything else got
faster or stayed flat), and a fair three-way needs all three codes re-run end to end under
one protocol. Accuracy is unaffected: the recovered spectra reproduce the recorded RMS
exactly (0.0093 / 0.0116 mean-aligned).

The re-run at the end of this file did that. It reproduced all twelve accuracy values
exactly, explained the 0.049 s (the harness's in-process convention, timing shift-and-add
on a heap the XLA solve had just worked over, shared by both recorded numbers), caught
fd3's OpenBLAS spinning 32 threads (pinned to one, fd3 is 1.7× faster), and moved the
harness to fresh-process timing.

### Memory: unchanged, which was the requirement

A speedup must not undo the memory pass. Peak buffer-assignment bytes (XLA
`memory_analysis()`, the same instrument):

| n (model px) | eval, then | eval, now | ∇, then | ∇, now |
|---|---|---|---|---|
| 31,734 | 2.94 GB | 2.96 GB | 4.00 GB | 4.02 GB |
| 74,322 | 4.86 GB | 4.76 GB | 11.47 GB | **10.26 GB** |
| 135,052 | 7.83 GB | 7.81 GB | 14.37 GB | **12.84 GB** |
| **203,440** | 11.06 GB | 11.14 GB | 18.24 GB | 18.33 GB |

Flat at the design target (+0.5%, the row that has to fit in 32 GB) and 8-11% better in
the middle rows, where the closed-form reverse rule stops materializing whole-tensor
temporaries.

### What it cost: the package no longer has a forward-mode path

`custom_vjp` rejects `jax.jvp` outright, and `forecast._effective_parameters` was the
only place in albireo that used forward mode: the forecast gets `p_eff = tr[Sigma A^T W A]`
from one directional derivative of `log det` in the noise scale, because `with_jitter` is
already that one-parameter family. The full suite caught it: eleven `test_forecast.py`
failures and two in `test_plotting.py`, all `TypeError: can't apply forward-mode autodiff
(jvp) to a custom_vjp function`.

Both `t` and the log-determinant are scalars, so forward and reverse mode compute the
same single number; it is now a `jax.grad`:

| | committed code | with this pass |
|---|---|---|
| forward (`jax.jvp`) | 0.283 s | **rejected** |
| reverse (`jax.grad`) | 0.773 s | 0.532 s |
| `p_eff` | 3979.897960 | 3979.897960 |

Bit-identical (absolute difference exactly 0), and `test_p_eff_matches_dense_trace` pins
it against a dense trace oracle at rel 1e-8 on either route. The cost is 0.283 s → 0.532 s
on a call made once per forecast, in exchange for 1.8x on a gradient evaluated ~2,600
times per posterior.

This closes forward mode through the marginal likelihood entirely; the first speedup pass
had already closed it one stage later (`_solve_stage` is a `custom_vjp`, which is why
`laplace_inverse_mass` uses reverse-over-reverse). Second derivatives remain available
(and tested) through reverse-over-reverse.

### Four candidates killed by measurement

Each looked correct on paper. They bound what remains.

1. **A blocked Cholesky.** The block factorization is 31% of an evaluation, and XLA's
   fp64 dense `cholesky` is far slower per flop than its `matmul` at the block size the
   solver uses:

   | n | `matmul` | `cholesky` | `solve_triangular` |
   |---|---|---|---|
   | 256 | 183 GF/s | 7 GF/s | 41 GF/s |
   | **513** | **249** | **13** | **103** |
   | 1026 | 390 | 26 | 164 |
   | 2048 | 921 | 69 | 281 |

   A recursive blocked factorization built out of trsm + gemm was expected to win by a
   large factor. It gained 1.36x (2.39 ms against 3.26 ms at n = 513, best inner block
   128), because neither trsm nor the leaf factorizations parallelize at that size either.
   Larger blocks are worse: the cost is `O(n B^2)`, so doubling `B` pays 4x the flops to
   buy about 2x the rate. The block Cholesky is at its practical ceiling on this stack,
   and is now the largest single item in an evaluation.
2. **j-factoring the T-sandwich.** `f_ij = sum_ab w_i[a] w_j[b] S[i][a][1+b-a]` equals
   `A_i + frac_j * B_i`, which builds the tent slices once per component instead of
   once per (i, j) pair, 16 slab operations down to 10. Measured slower (0.81 s
   against 0.79 s; results agree to 5e-16). XLA already fuses the four terms into one
   pass, so the restructure only adds a materialized intermediate.
3. **`remat=False`.** The memory pass chose rematerialization of the epoch body, which is
   also faster: 7.64 s against 9.81 s for the epoch scan's gradient. The memory and time
   choices coincide.
4. **A custom-VJP band-to-block packing**, from the earlier list of remaining levers.
   The effect is real (`_pack_band`'s gather does transpose to a scatter), but it costs
   0.20 s of a 10 s gradient. It is unimplemented by decision.

Rejected by arithmetic before implementing: re-laying-out the band tensor as
`(nc, nc, n_pix, n_k)` so each epoch's slice is contiguous rather than strided by `nc`.
The band read-modify-write is only ~0.15 s of the 0.76 s T-sandwich, and ablating it
entirely made the forward slower: the traffic is in building `f`, not in storing it.

## Re-run: all three codes, one machine, and the wall that would not reproduce (2026-08-16)

All three codes under one protocol on one machine. Both walls that misbehaved are
properties of the environment, not of the codes.

Stack (the original tables record none, which left 0.018 s unfalsifiable): AMD Ryzen 9
9950X3D (16 cores / 32 threads), 31.1 GiB, Windows 11 Pro build 26200, WSL2 kernel
6.18.33.2 for fd3; Python 3.13.9, jax 0.11.0, numpy 2.5.2.

### Accuracy first: twelve values, twelve exact reproductions

Each code ran once before any timing. All twelve recorded RMS values reproduce to the
printed precision (albireo 0.0118 / 0.0093 and 0.0165 / 0.0116, fd3 0.1767 / 0.0198 and
0.2597 / 0.0223, shift-and-add 0.0317 / 0.0248 and 0.0849 / 0.0302), with the exported fd3
inputs checksum-identical to the prior session's and fd3's deterministic output `.mod`
byte-identical. The computations are fixed points; what changed is the measurement.

### The walls, one protocol

Two to three warmups, then nine recorded repeats per code, strictly sequential, nothing else
running:

| | recorded earlier | min | median | convention |
|---|---|---|---|---|
| shift-and-add, 7 sweeps | 0.018 s | **0.0263 s** | 0.0267 s | fresh process, jax never imported |
| albireo | 0.182 s | **0.0591 s** | 0.0625 s | jitted steady state; cold compile + first call 0.495 s |
| fd3, `OMP_NUM_THREADS=1` | — | **0.0636 s** | 0.0640 s | full WSL process; see below |
| fd3, environment as found | 0.111 s | 0.1042 s | 0.1113 s | full WSL process |

Ranking: shift-and-add first, then albireo and single-threaded fd3 at parity (mins 0.0591
against 0.0636, medians 0.0625 against 0.0640), then fd3 as it ships. The earlier "albireo
loses to both" described one measurement. The durable statements are that shift-and-add is
fastest everywhere, as a small number of array shifts and means should be, and that
albireo's 32-thread XLA graph recovers fd3's single-thread advantage on current hardware.
The speedup pass's partial recheck reproduces: its 0.059 s is this table's 0.0591, its fd3
0.099 s sits inside a later control series (minima 0.092–0.104), and its irreproducible
0.049 s sits inside the contaminated band below.

### Where the missing milliseconds went: the harness heated the heap

The committed harness timed shift-and-add in the same process, after the albireo solve,
the convention behind both recorded numbers. Dose–response, one process,
minimum wall per stage: numpy-only 0.0265 s → `import jax` 0.0267 → backend init 0.0267 → a
tiny jit 0.0275 → after the big jitted solve 0.0632–0.0736 s, and it does not recover:
`clear_caches()` plus gc reads 0.0729, three seconds of idle 0.0787. The inner `_shift`
goes 45.9 → 125.8 µs. It is neither threads nor the CPU: the code is pure NumPy, thread
pinning is flat, and at every stage user time ≈ wall with sys = 0 and zero page faults, so
the same thread runs the same instructions ~2.7× slower, but only when it allocates.

The discriminator: allocating ufuncs slow ~4× (`np.floor(a)` 0.82 → 3.46 µs, `a + b`
0.88 → 3.42 µs) while their `out=` twins are flat (0.62 → 0.64, 0.76 → 0.77 µs). The
penalty has the size structure of the Windows CRT heap: 8 KB requests (low-fragmentation
heap) flat at 0.17 → 0.15 µs; 34.6 KB, the size of shift-and-add's full-grid temporaries,
0.23 → 2.14 µs; 128 KB 0.23 → 5.54; 800 KB 0.24 → 9.96. XLA's allocation storm leaves
requests above 16 KB walking a user-mode free list for microseconds (the same address comes
back, so the walk is the cost), and `disentangle` makes ~10⁴ such allocations per call:
+20–45 ms, which is the observed band. Reproduced directly: the committed convention gives
0.037–0.043 s here, and after fully jitted runs 0.049–0.084 s, which contains that 0.049.

The earlier 0.018 s was taken through the same convention, on this same machine, with an
unknown dose, so the 1.46× residual (0.0263 here against 0.018 there) does not decompose
further (see [The machine](#the-machine)): serial small-array NumPy throughput on an
unrecorded stack, plus contamination of a size that cannot be reconstructed, and
contamination only slows a run. 0.018 s is unreproduced. `scripts/fd3_bench.py` now times
shift-and-add in a fresh interpreter (warmup, then min of five), the convention this table
uses; the row above is its first number under a recorded stack.

### The control that would not sit still: fd3's BLAS is not single-threaded

fd3, single-threaded C, was the control, expected to be flat under thread pinning. It was
not. The binary as built links its GSL against OpenBLAS, and in the default environment it
shows 1938% CPU, 1.95 s of user time inside a 0.10 s wall with 32 threads spinning, while
`OMP_NUM_THREADS=1` gives 93% CPU and 0.0636 s, 1.7× faster. On a many-core machine the
library's spin pool costs fd3 60%. This accounts for part of the "fd3 moved 12%" (see
[The machine](#the-machine)). The pinned figure is the one to use for fd3 as an algorithm;
the as-found row stays in the table because that is how it is normally run.

### What stands

The earlier tables stand as a record of this same desktop, software stack and timing dose
unrecorded. The accuracy result is confirmed a second time by exact replication: ~2× on
shape, the same *k* = 0 null space in all three codes, and a posterior from one of them.

## The incumbent's repository, feature for feature (2026-08-27)

The clean-room `scripts/shift_and_add.py` implements González & Levato's recurrence and
nothing else, so the three-way table compares methods, not codebases. This section compares
the repository most widely used in the field,
[`TomerShenar/Disentangling_Shift_And_Add`](https://github.com/TomerShenar/Disentangling_Shift_And_Add),
as software: what it provides, how it is distributed, and which differences from albireo
are differences in kind rather than in degree.

Provenance: everything below comes from the repository's README and the GitHub API. The
source files were never opened: the repository has no license (`"license": null` from the
API, checked 2026-08-27), so reading it would contaminate the one implementation of this
algorithm that albireo can legally maintain. Anyone who does open it should not afterwards
edit `scripts/shift_and_add.py`. Where the rule limits what this page can claim, the limit
is stated.

The repository is a set of Python research scripts, configured by editing
`Input_disentangle.py` and run as `python disentangle_shift_and_add.py`, with the core in
`Disentangling/disentangle_functions.py`. Around the recurrence it provides what a working
research tool needs: a χ² grid over the semi-amplitudes (K₁ and K₂, and K₃, since it
supports triples) given (P, T0, e, ω) from elsewhere; a negativity constraint against
spurious emission features; multi-instrument input in ASCII or FITS; mock-data generators for
SB2s and SB3s; and plotting utilities. V2.0 is dated September 2023, the last commit
2024-03-07; at the check date it had 10 stars, 2 forks and 1 open issue. It cites González &
Levato (2006) for the algorithm and asks users to cite Shenar et al. 2020 (A&A 639, A6) and
2022 (A&A 665, A148), both already in `paper/paper.bib`. It is the code behind the LB-1
and HR 6819 companion identifications, hence "the incumbent".

The numbers above cover the shared core only. The three-way table measured the published
recurrence under the paper's own stopping rule with the orbit fixed at truth, so its RMS and
wall values carry over to the incumbent's algorithm, not its code: the negativity constraint
and the χ² grid sit on top of that core, were never run here, and under the provenance rule
cannot be reimplemented from the source. A head-to-head against the repository as it ships
would be legitimate (running unlicensed code is permitted; deriving from it is not) and
remains undone.

| | `Disentangling_Shift_And_Add` | albireo |
|---|---|---|
| orbit | χ² grid over (K₁, K₂[, K₃]); P, T0, e, ω supplied | joint inference of every orbital site, gradient-based, with NUTS |
| uncertainty on K | χ² map over the grid | posterior draws |
| uncertainty on spectra | none | pointwise band, and draws from the joint posterior |
| the *k* = 0 null space | suppressed by the negativity constraint | pinned by the smoothness prior; reported as the leading mode by the forecast |
| contaminated pixels | masking/weights (in the published method) | the same masking, plus telluric and nebular *components* when masking would discard the science |
| triples | yes, K₃ in the grid | yes at the expert level (hierarchical outer orbit); the façade refuses it in v1 |
| validation | mock-data generators | closed-loop gates in CI against packaged truth, plus calibrated detection |
| distribution | scripts + a config file; no license, no package, no tests | BSD-3-Clause package on PyPI, CI, docs, tutorials |

The differences in degree are the measured ones above: about 2× on aligned shape, and the
fastest wall in the comparison belongs to the incumbent's algorithm, as a small number of
shifts and means should. The difference in kind is the column no handicap equalizes in the
three-way table: an uncertainty on the disentangled spectra, which no code in
[the survey of methods](science.md#2-methods-of-spectral-disentangling) produces. The
difference in practice is the license line: the incumbent cannot be vendored, forked, or
legally built upon, which is why the clean room exists and is a barrier for anyone
extending the method. This is not a criticism of its authors, for whom disentangling is a
means to an end and the software a by-product of the science.

As noted above, weights and rejection are inside the published method, so masking is not
an albireo advantage. The difference arises when masking would discard the pixels the
science needs, such as a nebular line sitting in Hβ, where albireo models the contaminant
instead (unmodelled, it moves K₂ by −59% and reports a circular orbit at e = 0.95, so the
choice reaches the masses and not only the atmospheres).

BLOeM, the survey behind [the BLOeM tutorial](tutorials/bloem-sb2.md), is led by the
incumbent's author, and its 59 published SB2s have no orbital solutions, the case
`Disentangler(velocities=...)` was built for. The two codes are stages rather than rivals:
the incumbent's shift-and-add found several of those systems, and albireo addresses what
comes after: the orbit, the spectra, and the error bars on both.


## Stellar labels from disentangled components (2026-08-27)

Machine: [the desktop](#the-machine), CPU only, float64; comparable with the second
speedup pass and the re-run, not with the earlier tables (unrecorded stack). Harness:
`scripts/label_bench.py`, offline and reproducible; the grid is a toy at BOSZ's own node
density (250 K in Teff, 0.5 dex in log g, 0.25 dex in [M/H]; 455 nodes x 2000 px) so that
the interpolation numbers can be read against the published ones.

### Interpolation, and the emulator question settled by measurement

Leave-out error at doubled node spacing, a pessimistic proxy since the real fit interpolates
on the full grid, in fractional normalized flux:

| method | rms | p95 | max | n tested |
|---|---|---|---|---|
| multilinear | 3.88e-04 | 1.74e-06 | 8.16e-03 | 371 |
| Catmull-Rom cubic | **1.84e-04** | 2.00e-07 | 4.51e-03 | 371 |

Against the literature for the same spacing on a real ATLAS9 grid (Meszaros & Allende Prieto
2013): linear 5.1e-04, cubic-Bezier 3.1e-04, and a Payne-style network about 1e-03.

1. The cubic is worth its 4^k taps: 2.1x better than multilinear here, 1.6x in the
   published comparison.
2. On a grid at this density a learned emulator would be worse, by roughly a factor of
   five, so for FGK none is built. This says nothing about the coarse, strongly
   non-linear OB grids, where the same measurement has to be repeated before an emulator
   is built or dismissed. `crossval_library` is that measurement, and it ships.

Node reproduction on this box grid is exact bit-for-bit (`==`, not a tolerance), which lets
the warm-start node scan and the continuous fit be compared on one footing. The simplex
path used for punched grids reproduces a node to rounding instead (its weights at a vertex
are 1 − ε and ε), and which nodes come back exact depends on the triangulation Qhull chose;
the test suite asserts the rounding bound under permuted node orders so that the comparison
holds on every scipy build.

### Closed-loop recovery

Two components injected at off-node labels, given to the fit with deliberately wrong light
fractions (assumed 0.72/0.28 against a true 0.62/0.38), noise at the declared level:

| S/N | star | dTeff [K] | dlog g | d[M/H] | dv sin i [km/s] | formal sigma(Teff) [K] | fitted light fraction |
|---|---|---|---|---|---|---|---|
| 100 | A | +17.9 | +0.0010 | +0.0080 | -0.887 | 11.05 | 0.621 |
| 100 | B | +12.6 | -0.0125 | +0.0080 | +0.195 | 7.68 | 0.379 |
| 250 | A | +7.2 | +0.0003 | +0.0033 | -0.333 | 4.36 | 0.620 |
| 250 | B | +5.0 | -0.0048 | +0.0033 | +0.077 | 3.09 | 0.380 |
| 500 | A | +3.7 | +0.0004 | +0.0017 | -0.169 | 2.13 | 0.620 |
| 500 | B | +2.3 | -0.0016 | +0.0017 | +0.033 | 1.57 | 0.380 |

The worst row is 0.35% in Teff against a target of 2-3% (math.md 9.6), 0.013 dex in log g
against 0.15, and 8% in v sin i against 10%. For scale, GSSP's own simulation recovery at
S/N 150 is +-40 K and +-0.06 dex, so this is inside the bar the mode has to clear, on a toy
grid (the caveat recorded below).

The light ratio is the main result. It comes back as 0.621/0.379 against a truth of
0.62/0.38, from an assumption of 0.72/0.28: the joint radius-ratio fit placed a 16% error in
the assumed dilution in the dilution parameter rather than in the temperatures. It is
quoted against `FixedDilution` on the same data: the example asserts that the frozen fit is
never closer to the truth.

Chi-square is 1835.8 of 2020 pixels at all three signal-to-noise levels, as it should be:
the quoted sigma matches the injected noise and the seed is fixed, so residual/sigma is the
same array and the reduced chi-square is scale-invariant at 0.909. The nulls move as
expected: the nearest-node null runs 7.6e3 / 3.8e4 / 1.5e5 and the no-template
null 1.7e5 / 1.1e6 / 4.2e6, both in units of the shrinking sigma.

### Wall clock

| stage | seconds |
|---|---|
| resample the library onto the model grid | 0.05 |
| build the interpolator | 0.00 |
| `match_labels`: node scan + 4 x L-BFGS + Laplace | 27.8 |
| refit 8 posterior draws | 41.1 |

455 nodes x 2000 px projected onto 1010 model pixels. The draws refit is the expensive half,
scales linearly in the draw count, and is opt-in.

### The formal error against the honest one — and what this run does *not* show

| star | label | formal | draws | ratio |
|---|---|---|---|---|
| A | Teff | 4.36 | 17.60 | 4.0x |
| A | v sin i | 0.182 | 0.339 | 1.9x |
| B | Teff | 3.09 | 4.33 | 1.4x |
| B | v sin i | 0.087 | 0.556 | 6.4x |
| A | log g | 0.007 | 0.007 | 0.9x |
| B | log g | 0.005 | 0.003 | 0.5x |

This table demonstrates the machinery, not the physics. This harness has no disentangling
behind it, so the draws are the data plus fresh white noise and carry none of the
correlated structure that makes the formal error optimistic. The spread measures
label-space non-linearity alone, and the scatter across rows (0.5x to 6.4x) is partly the
sampling error of a standard deviation over eight draws, about 27% on its own. The
literature's 5-10x (Gebruers et al. 2022: 70 K formal against 425 K realistic; Czekala et
al. 2015) is for joint posterior draws of real disentangled spectra, which carry the low-k
exchange modes; reproducing it is a task for the AI Phe validation run. This run
establishes that the propagation path works end to end and that the two numbers are
reported side by side with their ratio.

### Scope of these numbers

Every figure above is on a toy grid whose spectra are analytic Gaussian lines with each label
driving its own set, so that the label-to-spectrum map is invertible (a grid where Teff and
[M/H] both scale one depth drove chi-square to 1e-26 while failing to recover the injected
labels). Real grids have blends, saturated cores and a continuum that is not a smooth
exponential in Teff, so the recovery figures here are an upper bound. The real-data gate is
AI Phe against Maxted et al. (2020), not run here.


## AI Phoenicis: the label fit against a star (2026-08-27)

Machine: the same desktop. Harness: `scripts/aiphe_labels_bench.py` over 36 archival HARPS
spectra (R = 115,000, `scripts/download_aiphe.py`), disentangled on 5150-5250 A with the
velocities held at the published orbit, so that the label fit is under test and not the
orbit. Library: `bosz2024-fgk-r20000`, 454 nodes. The notebook of the same run is
`docs/tutorials/aiphe-labels.ipynb`.

The velocities were held at the rounded period, 24.5924 d. `aiphe_bench.py` now carries
24.592483 d, which moves the fixed velocities by up to 0.107 km/s at the steepest phases;
the figures in this section predate that change and are due a re-run.

Every quantity the mode produces has an independent published value for AI Phe: Teff
6310 K and 5010 K, log g 4.001 and 3.598, R2/R1 = 1.6237 (Maxted et al. 2020, run C). The
log g values are derived from the spectroscopic and photometric elements rather than
quoted, via `g_i = 2 pi sqrt(1-e^2) K_j / (P r_i^2 sin i)`, which needs no absolute
masses and reproduces the published ones to 0.002 dex; the script asserts this.

### Result

| configuration | primary Teff | secondary Teff | R2/R1 | chi2 |
|---|---|---|---|---|
| log g declared, dilution fitted | **6342.5 K (+0.52%)** | **5227.0 K (+4.33%)** | 1.5416 (-5.1%) | 425,869 |
| log g free | 6019.7 K (-4.60%) | 4900.3 K (-2.19%) | 1.5459 | 412,882 |
| log g declared, dilution frozen | 6449.4 K (+2.21%) | 5179.8 K (+3.39%) | n/a | 1,853,578 |

The primary is recovered to 0.52 per cent, inside the 2-3 per cent that math.md 9.6 says is
enough for template selection. The secondary is not: +217 K, or 4.3 per cent, a miss. The
radius ratio, which the fit is not told, comes back 5 per cent low from spectroscopy alone
against a photometric measurement.

Freeing log g reproduces on real data the failure the tutorial warns about: log g runs to the
bottom of its prior (3.000, the grid edge) and drags both temperatures down with it, while
chi-square improves. The correlation report comes back empty, because a parameter pinned
against a bound stops varying and the curvature at the optimum no longer shows the degeneracy
that produced the answer. A flagged correlation is evidence; an empty report is not evidence
of absence.

### The comparison mode was wrong, and this is what found it

`compare="matched"` convolves both the model and the data with the declared LSF before
comparing, on the argument that `d_hat` is a regularized partial deconvolution. It was the
original default. On AI Phe it drove both components to the floor of their `v sin i` prior
(0.14 and 0.46 km/s) and inflated chi-square against `native`:

| mode | primary Teff | secondary Teff | v sin i | chi2 |
|---|---|---|---|---|
| `matched` | 6319.5 K (+0.15%) | 5280.1 K (+5.39%) | 0.14 / 0.46 km/s | 1,813,881 |
| `native` | 6342.5 K (+0.52%) | 5227.0 K (+4.33%) | 2.23 / 2.21 km/s | 425,869 |

Convolving the residuals correlates them over the kernel width while the likelihood stays
diagonal, so chi-square is over-counted by the effective-sample-size factor `1/sum(k^2)`.
At sigma_LSF = 1.38 px that predicts 4.91 and the fit measured 4.26, the whole gap between
the two modes. The mis-specified likelihood does not only inflate chi-square; `v sin i`
absorbs it. The default is now `native`; `matched` is appropriate only once a
residual-covariance model can carry the correlation it creates.

The closed-loop test could not have found this. Its injected rows never pass through an LSF
or a disentangling, so they are intrinsic spectra and both modes recover them (matched is
better at S/N 1000: -0.1 K against -4.6 K). Only real data with a real deconvolution behind
it separates the two. A toy fixture validates the arithmetic it contains, not the
assumption it was built on.

### Microturbulence: a hypothesis, tested and refuted

The first suspect for the secondary was the library's pinned microturbulence. BOSZ offers
xi in {0, 1, 2, 4} km/s and the registry pins 2; a K subgiant requires nearer 1.3, and too
much microturbulence makes the model's metal lines too strong, which a fit can answer by
raising Teff. The direction is right: at the t5250/g3.5 node, xi = 2 gives 8.45 per cent more
equivalent width than xi = 1, so a 160-node library was rebuilt at xi = 1.

The secondary got worse (+292 K against +273 K) and chi-square with it (1.97e6 against
1.81e6). [M/H] moved instead, by +0.10 dex, the documented [M/H]-xi degeneracy absorbing
the change where math.md 9.2 says it goes. Recorded so that the test is not repeated.

What remains unexplained is most of the secondary's offset. Candidates, untested: a 100 A
window carrying far more temperature leverage for an F star than for a K subgiant; the
published 5010 K being a photometric/SED temperature rather than a spectroscopic one; and the
assumed light fractions, which the fitted radius ratio only partly absorbs.

### Caveats on these numbers

Formal errors here are sub-kelvin and carry no information: they are the curvature of an
optimum on correlated residuals, and `summary()` states this whenever it prints them.
`refit_draws` is the number to quote and was not run for this record. HARPS ships no error
array, so the weights are albireo's own estimate (the z-RMS 2.64 noted in the D-numbered AI
Phe disentangling entry applies here too). One system, one window, one library: this is a
validation, not a survey.

## Epoch velocities by N-dimensional correlation (2026-09-01)

Machine: the same desktop, CPU only, float64. Harness: `scripts/todcor_bench.py`, offline
and reproducible. The fixture is a simulated SB2 through the real operator stack (LSF sigma
5 km/s, rebin onto a 0.05 Å native grid, light 0.6/0.4, barycentric motion), with the
injected component spectra as templates on a 1 km/s grid (five pixels per LSF sigma). Every
number here is about the estimator, not template mismatch, which a real star adds on top
(discussed at the end).

### The estimator is TODCOR, exactly

On a uniform grid with uniform weights and the data on the model grid, the
weighted-least-squares surface albireo evaluates reproduces Zucker & Mazeh's (1994) symmetric
two-dimensional correlation (light ratio maximized out), their original fixed-ratio
expression, and the pinned least squares, each to 1e-10 against an independent NumPy
transcription of the published formulae (`tests/test_todcor.py`, the three identity tests).
The rest of this section covers data the published form cannot take.

### Precision, bias and calibration against S/N

Sixteen noise realizations of eight epochs each, fixed light fractions, errors profiled
(Zucker 2003) and, in the last column, with the declared weights trusted:

| S/N | star | bias [km/s] | scatter [km/s] | mean quoted sigma [km/s] | pull rms | pull rms (ivar errors) |
|---|---|---|---|---|---|---|
| 30 | A | -0.0233 +- 0.0142 | 0.1604 | 0.1702 | 0.957 | 0.953 |
| 30 | B | +0.0208 +- 0.0168 | 0.1901 | 0.1923 | 0.999 | 0.993 |
| 100 | A | -0.0070 +- 0.0043 | 0.0484 | 0.0510 | 0.961 | 0.958 |
| 100 | B | +0.0065 +- 0.0051 | 0.0573 | 0.0577 | 1.005 | 0.999 |
| 300 | A | -0.0023 +- 0.0014 | 0.0161 | 0.0170 | 0.960 | 0.957 |
| 300 | B | +0.0021 +- 0.0017 | 0.0192 | 0.0192 | 1.007 | 1.001 |

The quoted errors are calibrated: the pull rms sits between 0.96 and 1.01 at every S/N,
reproducing Zucker's (2003) Figure 4 for the weighted, projected, sub-pixel version. The
profiled and trusted errors agree because the noise was injected at the declared level; the
rescaling matters only on real data. The scatter scales as 1/(S/N) from 0.16 to 0.016 km/s,
a sixtieth of a pixel at S/N 300. A bias of a few thousandths of a km/s, of opposite sign in
the two components and independent of S/N (−0.002 / +0.002 km/s at S/N 300, about 1.5 sigma
each), is the shift-interpolation systematic of the next table, at the 0.002–0.006 px level
this grid sampling predicts, not a property of the noise.

### Pixel locking of the shift operator against template sampling

The linear shift operator blurs a template at half-pixel shifts, which adds a
one-pixel-periodic ripple to the chi-square and pulls the minimum toward integer shifts
(math.md §10.3). Measured on **noiseless** data simulated at 0.25 km/s, with the templates
rebinned onto coarser grids so the template interpolation error is real rather than an inverse
crime, and the injected velocities spanning a whole pixel of the coarsest grid in eighths:

| template dv [km/s] | LSF sigma [px] | estimate 0.1/sigma_px^2 [px] | max abs error [px] | max abs error [km/s] | rms error [km/s] |
|---|---|---|---|---|---|
| 0.50 | 10.0 | 0.0010 | 0.0022 | 0.0011 | 0.0007 |
| 1.00 | 5.0 | 0.0040 | 0.0058 | 0.0058 | 0.0033 |
| 2.50 | 2.0 | 0.0250 | 0.0153 | 0.0382 | 0.0173 |
| 5.00 | 1.0 | 0.1000 | 0.0293 | 0.1463 | 0.0967 |

The order-of-magnitude estimate is correct within a factor of three either way; the docs
quote it as an estimate and this table as the measurement. Three or more pixels per LSF
sigma keep the systematic below a hundredth of a pixel, while at one pixel per sigma it is
a tenth of a km/s, the size of a good epoch error. `Fit.templates()` therefore upsamples to
three per sigma, and `todcor` warns below two.

### Two dimensions against one as the components blend

The same spectra correlated against the primary's template alone (the one-dimensional CCF,
`todcor` with one template) and with the two-dimensional fit, as the injected separation of
the two stars' lines closes, at S/N 200:

| separation [km/s] | 1-D primary error [km/s] | 2-D primary error [km/s] | 2-D secondary error [km/s] | 2-D sigma A | blended flag |
|---|---|---|---|---|---|
| 0 | -1.924 | +0.010 | -0.001 | 0.024 | no |
| 5 | -2.369 | -0.023 | +0.056 | 0.026 | no |
| 10 | -2.743 | +0.019 | +0.022 | 0.025 | no |
| 20 | -0.990 | +0.026 | +0.047 | 0.026 | no |
| 40 | -0.267 | +0.003 | -0.018 | 0.025 | no |
| 80 | -0.148 | +0.028 | -0.007 | 0.026 | no |
| 120 | +0.621 | -0.019 | +0.012 | 0.026 | no |
| 160 | -0.316 | +0.001 | +0.065 | 0.025 | no |

The one-dimensional error reaches 2.7 km/s at 10 km/s separation, a hundred times the
two-dimensional quoted error, and it does not vanish when the lines separate: a secondary
carrying 40% of the light contaminates the primary's peak at every separation, in a direction
set by which of its lines fall near the primary's (+0.6 km/s at 120 km/s). The
two-dimensional fit is unbiased throughout because the contaminant is in the model, and its
error is flat in separation. The blending flag correctly never fires here: two different
spectra at the same velocity remain separable because their line lists differ, and the flag
is a statement about the covariance rather than about the velocity difference. It fires for
twin spectra at one velocity (`test_twin_stars_at_the_same_velocity_are_flagged_blended...`),
the degenerate case.

### Wall clock

Fixed light, one instrument, the coarse search striding five template pixels (the LSF sigma);
min of three after a warm-up, and the compile:

| window [A] | native pixels | search range [km/s] | coarse step [px] | compile + first epoch [s] | per epoch [s] |
|---|---|---|---|---|---|
| 60 | 220 | +-100 | 5 | 0.23 | 0.007 |
| 60 | 220 | +-300 | 5 | 0.11 | 0.007 |
| 60 | 880 | +-100 | 5 | 0.01 | 0.006 |
| 60 | 880 | +-300 | 5 | 0.10 | 0.006 |
| 60 | 3520 | +-100 | 5 | 0.22 | 0.011 |
| 60 | 3520 | +-300 | 5 | 0.11 | 0.012 |
| 2000 | 39680 | +-100 | 5 | 0.39 | 0.081 |
| 2000 | 39680 | +-300 | 5 | 0.25 | 0.125 |

A whole optical range (2000 Å at 0.05 Å, forty thousand pixels, an echelle's worth of
orders) over ±300 km/s costs 0.13 s per epoch on the CPU, and the compile a quarter of a
second per distinct (instrument, pixel count) shape. A BLOeM-sized survey (929 stars × 25
epochs of ~2000 pixels) is minutes rather than hours; the cost is dominated by the pair Gram
matrix, one matrix product of size pixels × shifts², an operation suited to a GPU when the
window and the search range both grow. The example's twelve epochs of ten
thousand pixels take 0.6 s including everything but the compile.

### Three components

| star | rms error [km/s] | mean quoted sigma [km/s] | pull rms | recovered light |
|---|---|---|---|---|
| A | 0.0251 | 0.0313 | 0.816 | 0.500 |
| B | 0.0413 | 0.0389 | 1.098 | 0.300 |
| C | 0.0645 | 0.0649 | 0.962 | 0.200 |

Light 0.5/0.3/0.2 fitted freely, S/N 200, ±80 km/s, four epochs in 1.3 s including the
compile. The 20% component, a TRICOR tertiary, comes back to 0.06 km/s with a calibrated
error and its light fraction to three decimals. The search grid grows as the cube of the
shift count, so the range was kept narrow; beyond three components the docstring recommends
narrowing `v_range` and coarsening `coarse_step`.

### Scope of these numbers

They measure the estimator against its own templates. A real star adds template mismatch
(the wrong temperature, gravity or rotation, or a disentangled component's own noise and
null-space contamination, §5.1), which is outside the quoted error and, per the literature
cited in `docs/tutorials/labels.md` (Posbic et al. 2012), mostly costs a constant zero point
per component rather than precision. The one real-data-shaped check here is the
self-consistent loop in `examples/12_todcor.py`: against the components a MAP disentangling
of the packaged example recovered, the velocities come back to 0.13 and 0.10 km/s rms once
each component's zero point is removed, and the Keplerian fitted to them returns *K* to
0.05%, with the reduced chi-square of the correlation itself at 1.005. AI Phoenicis, where
every element has a published value and the templates would be the label fits above, is
the next gate and has not been run.

## The pipeline in worker processes (2026-09-01)

Machine: the same desktop, CPU only, float64. Harness: `scripts/pipeline_bench.py`,
offline. The batch is eight simulated stars (the pipeline's own toy star, a two-component
SB2 drawn from `albireo.simulate.synthetic_library` at known labels, 8 epochs of 725 native
pixels each, S/N 120) with the label stage and the figures off, so the timing covers what
every star pays for: the conjunction scan and 60 L-BFGS steps of disentangling on an
893-pixel grid, the velocity table against the fit's own components, the orbit fit, and the
products. Every star is identical up to its noise seed. One warm-up star runs in-process
first so that the compile is paid before the in-process timing, as every worker pays it
once too. Another project's test suite was running during the first four rows and had
finished before the last two; the capped eight-worker point was measured in both states
and agreed to 0.1 s.

| workers | threads per worker | batch wall [s] | mean per star [s] | speedup |
|---|---|---|---|---|
| 1 (in-process) | 32 | 132.2 | 16.5 | 1.00× |
| 2 | 16 | 101.4 | 24.0 | 1.30× |
| 4 | 8 | 66.7 | 29.9 | 1.98× |
| 8 | 4 | 54.0 | 43.5 | 2.45× |
| 8 | 32 (cap off) | 54.7 | 43.0 | 2.42× |

Workers help, sub-linearly. Four workers finish the batch twice as fast as one process
and eight 2.5× as fast, while the mean wall per star climbs from 16.5 to 43.5 s as the
workers share the machine. A single in-process star is not a serial program: XLA's CPU
backend already spreads the banded solve and the operator assembly over the cores, so the
extra processes overlap only the serial remainder of each star (compilation, the 41-point
conjunction scan's Python loop, the orbit fit, the writing), which bounds the gain.

The thread cap made no measurable difference. The pipeline caps each worker's XLA and
BLAS threads at `cpu_count // jobs`, but eight workers each free to use all 32 threads
finished the same batch in 54.7 s against 54.1 s capped: on this workload the operating
system's scheduler absorbs the oversubscription. The cap stays as a precaution with no
measurable cost here, because BLAS-heavy stages are where oversubscription has been
observed on this machine (the re-run's 32-thread OpenBLAS); it is not the source of the
speedup.

`jobs="auto"` is `cpu_count // 4`, eight on this machine, the last capped row. Going beyond
the core count gains nothing on a CPU: the batch is compute-bound once the serial remainder
is overlapped, and each worker holds its own XLA runtime (a few hundred megabytes) and its
own copy of the shared configuration.

Not measured here: the label stage, which is the expensive part of a full run
(~50 s against ~30 s of disentangling on this star in fast mode) and scales the same way,
being one more in-process JAX program per star; and a GPU, on which the in-process star
would be faster and the workers' overlap smaller.

## Gaia RVS: disentangling and velocities on simulated double-lined binaries (2026-09-10)

Machine: the same desktop, CPU only, float64. Harness: `scripts/gaia_rvs_benchmark.py`,
which drives `albireo.benchmark.run_benchmark` and writes the full report with its figures,
`rows.csv` and `summary.json`; the numbers below are copied from it. Every star is run as a
user runs it, through `run_pipeline` from a `StarConfig`, and every metric is the
pipeline's own truth block, so the harness adds nothing the pipeline does not report.
Research notes cited by file name in this section are in the repository under
`internal/research/2026-09-09-gaia-rvs-benchmark/`.

**What is simulated.** Each system's two components are BOSZ 2024 spectra
(`bosz2024-fgk-rvs`, R = 20,000) at the labels the dwarf sequence gives for the masses,
rotationally broadened, and broadened to the RVS resolving power of 11,500 by the
quadrature width; the two are shifted on the relativistic log-wavelength grid, combined at
the RVS light fractions (the library's own continua times the radius ratio), and observed
at the epochs of the cadence model with photon noise per 0.245 A detector pixel at the S/N
the ESA predictive formula gives for the system's G_RVS and one transit per epoch. The
detector spectrum is then delivered on the DR4 epoch grid (0.025 nm, 961 samples from
846.0 to 870.0 nm) by linear interpolation with the variance propagated, so the delivered
noise is correlated as the archive's will be (lag-1 coefficient 0.27 on this grid, 0.81 on
the DR3 mean-spectrum grid), and the pipeline is told nothing of it. Wavelengths are vacuum
and barycentric.

**Two populations.** The first is the field as a magnitude-limited double-lined survey
sees it: 20 systems drawn (`--n 20 --seed 3`) from the Moe & Di Stefano (2017) period,
mass-ratio and eccentricity distributions, with the primary mass function weighted by the
survey volume, the pair kept in proportion to the volume its combined light reaches, and a
floor of 40 km/s on the largest velocity separation; G from the Gaia double-lined sample's
distribution, the transit count from the scanning-law model over the DR4 span. One system
with four transits was left out (the harness asks for ten). Quartiles of the 20 drawn:
P 4.8 / 34 / 267 d, e 0 / 0.43 / 0.65, K1 + K2 57 / 70 / 122 km/s, q 0.87 / 0.94 / 0.98,
F2/F1 0.62 / 0.81 / 0.89, Teff1 6100 / 6400 / 6700 K, G_RVS 8.5 / 9.2 / 10.1, transits
28 / 40 / 53; two eclipse. Half the pairs are twins with q > 0.95, as a magnitude-limited
double-lined sample is; one pair has a secondary of 5 percent of the light (q 0.54, P 117 d,
e 0.72), and one a period of 844 days over the 1998-day span.

The second population is real: the first 16 double-lined orbits (`SB2` and `SB2C`
solutions) the Gaia archive returns from `nss_two_body_orbit`, with their `binary_masses`
where the catalogue has them (else masses from the sequence at the Gaia temperature and the
catalogue's semi-amplitude ratio), the catalogue's period, eccentricity, argument of
periastron, semi-amplitudes and systemic velocity, the inclination that reconciles the masses
with the semi-amplitudes, the DR3 number of good transits, and G_RVS from G and G_RP. Systems
the library cannot render (a primary above 7000 K) are left out by name. These are observed
on the DR4 epoch grid with their DR3 transit counts: the sample DR4 will contain, at a
fraction of the epochs it will have. Fourteen systems ran (46 runs; one failed on the first pass at a starting value on the range's bound and completed on the rerun that followed the guard). Quartiles: P 3.1 / 5.9 / 8.6 d, e 0.013 / 0.068 / 0.36, K1 + K2 105 / 141 / 187 km/s, q 0.93 / 0.96 / 0.98, F2/F1 0.76 / 0.85 / 0.96, Teff1 5800 / 6200 / 6700 K, G_RVS 7.4 / 8.5 / 9.7, transits 12 / 15 / 16; four eclipse.

**Tiers.** Every system is simulated once and analysed under four declarations, each a
`StarConfig` in the pipeline's own vocabulary:

| tier | period | conjunction | elements | light fractions | label priors |
|---|---|---|---|---|---|
| oracle | known (Gaussian, width 1e-4 P) | known | known (e, omega held) | truth | narrow (300 K, 0.3 dex) |
| eclipsing | known | known (an eclipse ephemeris) | free | truth | narrow |
| orbit | known to a Gaia solution's precision | scanned | free | measured by correlation | wide |
| blind | searched (bootstrap from library templates) | scanned | free | measured by correlation | wide |

The eclipsing tier runs only on the eclipsing systems. The oracle tier measures the
disentangling and the velocities alone; the blind tier measures what the pipeline delivers
from the spectra alone. The semi-amplitudes are a range of 2 to 250 km/s wherever they are
not declared, the same for every system.

**The field population: 19 systems, 59 runs, none failed.** Median and the 16th to 84th
percentile over the systems of each tier, from `summary.json`; the semi-amplitudes come
from two places, the Keplerian of the disentangling and the orbit fitted afterwards to the
velocity table, because they fail differently.

| quantity | oracle (19) | eclipsing (2) | orbit (19) | blind (19) |
|---|---|---|---|---|
| abs(dK1 / K1) [%], disentangling | 1.69 (0.15 to 14.8) | 1.69 (1.65 to 1.73) | 1.78 (0.45 to 56.7) | 2.11 (0.31 to 13.6) |
| abs(dK2 / K2) [%], disentangling | 0.72 (0.12 to 8.8) | 0.67 (0.61 to 0.74) | 2.61 (0.28 to 25.2) | 12.2 (0.45 to 100) |
| K1 and K2 within 5%, disentangling | 63% and 79% | 100% and 100% | 74% and 58% | 63% and 42% |
| abs(dK1 / K1) [%], velocity table | 1.79 (0.28 to 14.7) | 1.67 (1.55 to 1.79) | 1.84 (0.33 to 32.6) | 1.78 (0.27 to 87.6) |
| abs(dK2 / K2) [%], velocity table | 2.5 (0.16 to 23.7) | 0.31 (0.26 to 0.36) | 2.87 (0.22 to 29.5) | 19.4 (0.47 to 121) |
| K1 and K2 pull rms, velocity table | 3.3 and 6.3 | 4.2 and 0.7 | 5.9 and 13.7 | 2.0 and 6.3 |
| components recovered in the other order | 0% | 0% | 21% | 5% |
| abs(dP / P) from the table | 3.1e-5 | 4.1e-6 | 2.5e-5 | 4.2e-4 |
| abs(de) | 0.004 | 0.008 | 0.004 | 0.017 |
| abs(d t_conj) [phase] | 0.0012 | 0.0027 | 0.0075 | 0.084 |
| abs(d gamma) [km/s] | 0.10 | 0.11 | 0.21 | 0.27 |
| epoch velocity rms, A and B [km/s] | 1.35 and 1.95 | 2.23 and 2.21 | 1.42 and 2.25 | 3.5 and 14.2 |
| epoch velocity pull rms, A and B | 1.49 and 1.23 | 1.33 and 1.21 | 1.44 and 1.61 | 3.8 and 3.1 |
| spectrum correlation, A and B | 0.92 and 0.80 | 0.90 and 0.91 | 0.89 and 0.78 | 0.92 and 0.74 |
| abs(dTeff), A and B [K] | 130 and 179 | 143 and 262 | 255 and 561 | 302 and 453 |
| abs(dlog g), A | 0.20 | 0.28 | 0.29 | 0.33 |
| period search recovered within 2% | | | | 58% (11 of 19) |
| wall per star [s], 8 workers | 476 (296 to 641) | 533 | 742 (579 to 978) | 647 (460 to 1650) |

The separation decides, more than the magnitude or the transit count. Ten of the nineteen
systems have a largest velocity separation above four line-spread widths (about 100 km/s):
nine with P < 25 d and the eccentric 117-day pair. Under the oracle tier all ten have both
semi-amplitudes within 1.7 percent, seven of them within 0.7; under the orbit tier within
2.3 percent but for the faint companion of the 117-day pair (4.2); under the blind tier
within 2.1 percent for seven of the ten, the exceptions a twin whose period was found at a
third of its value (P = 3.95 d, both semi-amplitudes then 10 percent off) and two secondaries
at 3.6 and 6 percent. Their recovered spectra correlate with the injected ones at 0.85 to
0.99. The nine below four widths (K1 + K2 of 34 to 72 km/s, seven of them with P > 130 d and
e of 0.4 to 0.7, 11 to 62 transits, S/N of 7 to 11 per pixel) have oracle-tier primary
errors of 3 to 32 percent and secondary errors of 0.1 to 76, larger under the other tiers,
and spectra at 0.4 to 0.7. The lines of such a pair never part by more than two resolution
elements, and no cadence repairs that. G_RVS and the transit count order the residuals only
within the separated group.

The blind period search recovers 11 of 19 within 2 percent: every system with P < 50 d but
one (the 3.95-day twin above, found at 0.30 of its period), and the 117-day pair, whose
primary alone carried the periodogram. All seven with P > 130 d fail, to aliases at 0.0003
to 0.09 of the true period: the scanning law's day-scale structure, sampled 11 to 62 times
over the 1998-day span, aliases a slow orbit whose semi-amplitudes are one resolution
element.

Two sources of K fail differently. For the four systems with a faint secondary (F2/F1 of
0.05 to 0.17) the disentangling recovers K2 to 0.1 to 8 percent under the oracle tier, and
to 0.02 km/s for the 5 percent companion under the blind tier once the semi-amplitudes were
started from the table; the correlation table gives zero for that companion in every tier
(the template of a 5 percent secondary sits on the primary's lines, and the pipeline flags
the disagreement with the disentangling). The table's formal errors are optimistic: the
semi-amplitude pull rms is 3 to 6 in the tiers where the orbit is known, the scatter about
the Keplerian being above the per-epoch errors for the poorly separated systems, while the
epoch-velocity pulls sit at 1.2 to 1.5 (the delivered grid's noise correlation of 0.27 is
not in the noise model, which takes the per-epoch errors as white).

Twins came out in the other order in four of the nineteen orbit-tier runs and one blind
run: at q of 0.98 with the light fractions measured rather than declared, the mass-order
convention had nothing to work on, and the fit converged with K_A above K_B by the 2
percent the data allow. The truth comparison recognises the exchange and judges the
recovery in that order; judged so, these runs are as good as the rest (0.1 to 2.3 percent).
The eclipsing tier, where the light fractions are declared, kept the order.

The label fit's temperatures are 130 K (primary) and 180 K (secondary) from the truth
under the oracle tier, 250 to 300 K and 450 to 560 K under the wide priors of the orbit
and blind tiers; the primary's error is negative in fifteen of nineteen oracle runs, a bias
worth its own study. Surface gravity is recovered to 0.2 to 0.3 dex, and the light fraction
the label fit measures agrees with the declared one to 0.008 under the oracle tier and to
0.03 to 0.05 where the fractions were themselves measured by correlation.

Before the semi-amplitudes were started from a template table (a change this run forced,
in the ledger), the orbit tier of the same population had a median secondary error
of 44 percent over its 13 completed runs, with both semi-amplitudes of the faint-secondary
system at the floor of the range; the oracle tier, whose semi-amplitudes are declared, is
untouched by the change.

Walls: at 100 disentangling and 80 label steps with eight workers, 476 s per oracle star
(the velocity budget follows the declared K), 742 s per orbit-tier star and 647 s per blind
star (a range of 2 to 250 km/s sets a 998 km/s budget and a 3500-pixel grid), 110 minutes
for the 59 runs.

**The Gaia DR3 orbits: 14 systems, 46 runs.** The archive's first sixteen double-lined
orbits, two left out for a primary above the library's 7000 K (`--gaia-sb2 16`); the
fourteen have P of 0.54 to 30 d, K1 + K2 of 29 to 296 km/s, q of 0.53 to 1.00, F2/F1 of
0.07 to 0.98, G_RVS of 6.7 to 10.4, and 10 to 25 transits, the DR3 counts. Four eclipse by
the inclination that reconciles their masses with their semi-amplitudes; two are contact-close
pairs at 0.54 and 0.58 d with synchronised rotation of 100 to 120 km/s.

| quantity | oracle (14) | eclipsing (4) | orbit (14) | blind (14) |
|---|---|---|---|---|
| abs(dK1 / K1) [%], disentangling | 0.27 (0.04 to 1.5) | 16.4 (8.1 to 38.6) | 12.8 (0.42 to 79) | 56 (2.9 to 88) |
| abs(dK2 / K2) [%], disentangling | 0.21 (0.05 to 0.60) | 34.5 (9.6 to 76.4) | 25 (0.47 to 62) | 57 (3.3 to 110) |
| K1 and K2 within 5%, disentangling | 100% and 100% | 0% and 25% | 43% and 43% | 21% and 21% |
| abs(dK1 / K1) [%], velocity table | 1.1 (0.18 to 4.3) | 37.6 (9.1 to 66.1) | 10.5 (0.63 to 87) | 40 (3.6 to 79) |
| abs(dK2 / K2) [%], velocity table | 0.72 (0.22 to 6.9) | 34.3 (7.7 to 73.7) | 20 (0.22 to 91) | 32 (3.4 to 109) |
| K1 and K2 pull rms, velocity table | 1.6 and 1.8 | 12.0 and 7.8 | 3.8 and 7.2 | 9.6 and 9.4 |
| components recovered in the other order | 0% | 0% | 21% | 7% |
| abs(de) | 0.005 | 0.068 | 0.046 | 0.12 |
| abs(d t_conj) [phase] | 0.0018 | 0.023 | 0.076 | 0.21 |
| epoch velocity rms, A and B [km/s] | 1.1 and 1.9 | 60 and 64 | 5.2 and 14.1 | 43 and 53 |
| epoch velocity pull rms, A and B | 1.53 and 1.39 | 38 and 40 | 4.5 and 7.7 | 41 and 33 |
| spectrum correlation, A and B | 0.89 and 0.85 | 0.76 and 0.50 | 0.81 and 0.80 | 0.79 and 0.57 |
| abs(dTeff), A and B [K] | 192 and 209 | 257 and 205 | 210 and 326 | 388 and 473 |
| period search recovered within 2% | | | | 36% (5 of 14) |
| wall per star [s], 8 workers | 369 (277 to 435) | 400 (273 to 466) | 462 (394 to 552) | 495 (378 to 811) |

This population separates the disentangling from its starting values. With the orbit
declared, the oracle tier recovers every one of the fourteen systems within 4 percent in
both semi-amplitudes, ten within 1 percent in the primary and thirteen in the secondary, at
medians of 0.27 and 0.21 percent, on 10 to 25 epochs; the epoch velocities are good to 1 to
2 km/s with pulls of 1.4 to 1.5, and the worst system is the 0.72-day pair whose
semi-amplitudes sum to 29 km/s, one line-spread width (4.1 percent on K1). That is what
Gaia's DR3 epoch count supports when the period, the conjunction and the elements are known.

With a semi-amplitude left as a range the same systems fail on about half the runs: 43
percent of the orbit-tier runs within 5 percent, 21 percent of the blind ones. The failures
are of the starting value, not of the fit that follows. A template table of 10 to 16
epochs is not a reliable seed: on the 2.81-day twins the eclipsing tier's table came back
collapsed by exchanged epochs and started the fit at 37 and 32 km/s for a pair at 102 and
104, from where L-BFGS settled at 47 and 49 (half the answer), while the orbit tier's table
of the same star, drawn with measured light fractions, started at 130 and 127 and converged
to 102.2 and 103.9; on an 18.9-day pair with a secondary rotating at 87 km/s the table
started the secondary at 228 km/s for 35 and the fit stayed near it; on a 5.36-day twin the
table measured neither star and the evenly spaced starts ended with the primary at the floor.
The blind tier adds the period search, which recovers 5 of 14 within 2 percent on 10 to 25
epochs (the field population's 11 of 19 had 28 to 100), the rest going to aliases at 0.04 to
2.3 of the period. The next step, not in this run, is a coarse scan over the
semi-amplitudes through the existing conjunction scan before L-BFGS on every route where
they are a range.

Twins came out in the other order in 21 percent of the orbit-tier runs and 7 percent of
the blind, as in the field population, and are judged in that order. The label fit gives
temperatures 190 to 210 K from the truth under the oracle tier on these 10 to 25 epochs,
against 130 to 180 K on the field population's 28 to 100.

**What the run changed in albireo, and what it left open.** Every defect the harness
found is in the ledger (D61); listed here is what a user meets. The bootstrap of the
search route re-assigns exchanged epochs by each candidate orbit and tries the peaks of a
swap-invariant periodogram, and a degenerate bootstrap semi-amplitude sends the fit to the
range instead of pinning it. A semi-amplitude declared as a range now starts where a
template table fitted at the declared period puts it, every start held strictly inside the
range. A pair recovered in the other order is recognised in the truth comparison and
flagged. The report tabulates the semi-amplitudes of the disentangling beside the table's.
Left open: the scan over the semi-amplitudes, the noise correlation in the noise model and
the temperature bias (all taken by the second run below; the bias turned out to be the
library's, not the disentangling's); the scanning law's phase and the ATLAS9 box above
7000 K, which would admit the two Gaia systems left out here, remain.

### The second run (D62, 2026-09-10): the epoch velocities as a product, the noise model, the scan, the library

Both populations were run again with the same simulated epochs (the same seeds) and the
same optimizer budgets, with five or six workers on the same desktop, which was also running
unrelated jobs for much of the time, so the walls below are not comparable with the first
run's.

**What changed.** (1) The measured epoch velocities, one per component per epoch from
TODCOR against the disentangled components with the label fit's zero points, were a
product of every star run (`velocities.rv`) but only two statistics of the report; the
benchmark now gathers them all with the injected velocity of each epoch into
`velocities.csv`, pools the usable epochs per tier, and draws every system phase-folded
against the injected orbit. (2) The delivered grid's lag-one noise correlation (0.27 on the
DR4 grid) is declared to every tier: the disentangling runs the AR(1) noise model along the
pixel index and the velocity table's errors carry the correlation through the sandwich of
math.md §10.4. (3) A template table of a dozen epochs had seeded the range-K routes in the
wrong basin on half the Gaia population. A scan over the semi-amplitudes alone, at the
phase the conjunction scan had
located at the wrong semi-amplitudes, found nothing on three test systems (a phase located
at semi-amplitudes a factor of three off sat a quarter of a period from the truth), so the
façade now scans the marginal likelihood jointly over a geometric grid of every ranged
semi-amplitude and eight conjunction phases, refined twice around the best trial, on a copy
of the declaration with the model grid at twice the pixel (a quarter of the cost at a dozen
epochs, two thirds at eighty), and starts L-BFGS from the best trial when it beats the seed.
On the 2.81-day Gaia twins seeded at 37 and 32 km/s the joint scan lands at 105.7 and 105.7
for a 102/103 pair, with the phase right; holding either component at its seed costs 350 to
400 nats. Two guards followed from a first rerun: a best trial with every ranged
semi-amplitude at the floor of its grid is the static-component minimum and is refused (an
eccentric contact pair scanned near-circular preferred it), and a component is moved from
its seed only when holding it there costs more than 25 nats (holding a 7-percent secondary
at its seed cost 11). The template table's orbit starts a free eccentricity where it
detected one at three sigma; a dozen epochs fit an eccentricity of 0.2 to a circular pair as
readily as not, and a scan on that shape of curve lost the same secondary. (4) The
temperature bias was not the disentangling's. A label fit on a perfect spectrum drawn from
the grid itself ended 20 to 150 K from the truth with a chi-square above the truth's,
because BOSZ publishes no model at 5750 K, log g 3.0, [M/H] −0.75 and that one gap had made
the whole grid fall back from the Catmull-Rom cubic to the barycentric simplex
interpolant, piecewise linear over an arbitrary triangulation of the lattice, whose kinks
stop L-BFGS. The gap is now filled by linear interpolation along [M/H] between its
published neighbours, the box is complete, the cubic applies, and the same fits end within
about 30 K; the first run's formal label errors, meaningless at a kink (a pull rms of 2300
in the oracle tier), become curvatures of a smooth surface.

**The field population again: 19 systems, 59 runs, none failed.** The same nineteen
systems and epochs as above. Median and the 16th to 84th percentile over the systems of
each tier; the first run's value follows in brackets where it differs by more than the
spread.

| quantity | oracle (19) | eclipsing (2) | orbit (19) | blind (19) |
|---|---|---|---|---|
| abs(dK1 / K1) [%], disentangling | 0.77 (0.14 to 20) [1.7] | 2.0 | 1.9 (0.35 to 66) | 3.4 (0.48 to 84) |
| abs(dK2 / K2) [%], disentangling | 0.79 (0.22 to 4.1) | 1.2 | 1.8 (0.76 to 110) | 5.4 (0.6 to 49) [12] |
| K1 and K2 within 5%, disentangling | 58% and 84% | 100% and 100% | 63% and 58% [74 and 58] | 53% and 47% [63 and 42] |
| abs(dK1 / K1) [%], velocity table | 2.1 (0.27 to 47) | 1.6 | 1.2 (0.22 to 12) | 3.6 (0.41 to 74) |
| abs(dK2 / K2) [%], velocity table | 1.0 (0.17 to 42) [2.5] | 0.89 | 1.8 (0.47 to 62) | 14 (0.51 to 81) |
| components recovered in the other order | 0% | 0% | 26% | 16% [5] |
| abs(de) | 0.008 | 0.010 | 0.006 | 0.015 |
| abs(d t_conj) [phase] | 0.0015 | 0.0007 | 0.018 | 0.024 [0.084] |
| epoch velocity rms, A and B [km/s] | 2.2 and 2.7 [1.3 and 1.9] | 2.6 and 2.6 | 1.9 and 2.7 | 4.0 and 4.5 [3.5 and 14] |
| epoch velocity pull rms, A and B | 1.4 and 1.1 [1.5 and 1.2] | 1.3 and 1.2 | 1.4 and 1.3 | 1.3 and 1.6 [3.8 and 3.1] |
| epochs, pooled: median abs residual A and B [km/s] | 0.91 and 1.3 | 2.6 and 1.9 | 0.93 and 1.4 | 1.2 and 1.9 |
| epochs, pooled: within 3 sigma, A and B | 90% and 91% | 99% and 100% | 86% and 84% | 80% and 80% |
| spectrum correlation, A and B | 0.94 and 0.78 | 0.91 and 0.91 | 0.89 and 0.78 | 0.89 and 0.72 |
| abs(dTeff), A and B [K] | 123 and 186 | 114 and 104 [143 and 262] | 128 and 424 [255 and 561] | 191 and 488 [302 and 453] |
| period search recovered within 2% | | | | 63% (12 of 19) [58%] |
| wall per star [s], 5 workers, shared machine | 421 | 477 | 614 | 580 |

This population's range-K routes were seeded from tables of 28 to 100 epochs, which were
good seeds, so the scan changes little here and mostly agrees with them: the
orbit-tier medians move within their spread (1.9 and 1.8 percent against 1.8 and 2.6), the
blind tier's secondary improves from 12 to 5.4 percent and its conjunction error from
0.084 of a period to 0.024, the period search finds 12 of 19 against 11, and the
epoch-velocity pulls of the blind tier fall from 3.8 and 3.1 to 1.3 and 1.6. The oracle
tier's primary improves from 1.7 to 0.8 percent, and its epoch velocities, 859 pooled
epochs of which 98 percent are usable, sit within three sigma of the injected value 90
percent of the time at pulls of 1.4 and 1.1, with a median absolute residual of 0.9 and
1.3 km/s against a median quoted error of 1.0 and 1.2. The exchanged runs rise from four to
five in the orbit tier and from one to three in the blind, the same twins at q of 0.98,
judged in the order they came out. The separation still decides: the seven systems with P
above 130 days and K1 + K2 below 65 km/s are the worst of every tier, the oracle included, where their
primaries come back 6 to 59 percent off (three of them 54 to 59 percent low) and no other
tier does better, because their lines never part by more than two resolution elements;
the scan finds nothing there because there is nothing to find. The label fit's temperatures
improve where the disentangled spectra are good, the eclipsing tier's secondary from 262
to 104 K and the orbit tier's primary from 255 to 128, and stay at 120 to 190 K under the
oracle tier's 300 K priors, where the reports again say the fit learned nothing about the
primary's temperature in fifteen of the nineteen runs.

**The Gaia DR3 orbits again: 14 systems, 46 runs, none failed.** The same fourteen
systems and epochs as above. Median and the 16th to 84th percentile over the systems of
each tier; the first run's value follows in brackets where it differs by more than the
spread.

| quantity | oracle (14) | eclipsing (4) | orbit (14) | blind (14) |
|---|---|---|---|---|
| abs(dK1 / K1) [%], disentangling | 0.45 (0.23 to 3.0) | 3.0 (1.4 to 6.4) [16] | 3.5 (0.32 to 34) [13] | 24 (3.8 to 110) [56] |
| abs(dK2 / K2) [%], disentangling | 0.36 (0.03 to 1.1) | 14 (5.2 to 22) [34] | 4.2 (0.57 to 88) [25] | 24 (3.5 to 150) [57] |
| K1 and K2 within 5%, disentangling | 93% and 100% | 75% and 25% [0 and 25] | 64% and 50% [43 and 43] | 21% and 21% |
| abs(dK1 / K1) [%], velocity table | 0.62 (0.31 to 7.3) | 1.9 (1.0 to 5.4) [38] | 0.91 (0.18 to 25) [10] | 27 (1.5 to 86) |
| abs(dK2 / K2) [%], velocity table | 0.77 (0.12 to 1.9) | 11 (4.4 to 19) [34] | 1.9 (0.09 to 82) [20] | 25 (2.2 to 280) |
| components recovered in the other order | 0% | 0% | 7% [21] | 14% |
| abs(de) | 0.005 | 0.011 [0.068] | 0.011 [0.046] | 0.20 |
| abs(d t_conj) [phase] | 0.0022 | 0.0065 [0.023] | 0.0046 [0.076] | 0.29 |
| epoch velocity rms, A and B [km/s] | 1.2 and 2.9 | 3.5 and 14 [60 and 64] | 1.9 and 3.0 [5.2 and 14] | 20 and 31 [43 and 52] |
| epoch velocity pull rms, A and B | 1.1 and 0.94 [1.5 and 1.4] | 2.4 and 2.8 [38 and 40] | 1.5 and 1.3 [4.5 and 7.7] | 16 and 17 [41 and 33] |
| epochs, pooled: median abs residual A and B [km/s] | 0.61 and 0.78 | 1.6 and 8.7 | 0.65 and 0.93 | 1.9 and 1.8 |
| epochs, pooled: within 3 sigma, A and B | 93% and 95% | 78% and 56% | 81% and 79% | 59% and 57% |
| spectrum correlation, A and B | 0.92 and 0.86 | 0.82 and 0.81 [0.76 and 0.50] | 0.86 and 0.63 | 0.91 and 0.61 |
| abs(dTeff), A and B [K] | 196 and 122 [192 and 209] | 28 and 74 [257 and 205] | 166 and 260 [210 and 326] | 155 and 432 [388 and 473] |
| period search recovered within 2% | | | | 21% (3 of 14) [36%] |
| wall per star [s], 6 then 5 workers, shared machine | 413 | 609 | 581 | 551 |

The range-K tiers are where the run changed. With the period known and the conjunction
scanned, nine of the fourteen systems now come back within 5 percent in both
semi-amplitudes, against six before, the median errors fall from 13 and 25 percent to
3.5 and 4.2, the table's from 10 and 20 to 0.9 and 1.9, the conjunction error from 0.076 of
a period to 0.005, the exchanged runs from three to one, and the epoch-velocity pulls from
4.5 and 7.7 to 1.5 and 1.3. The 2.81-day twins, whose eclipsing tier had settled at half
the semi-amplitudes from a table seed of 37 and 32 km/s, are recovered to 3 percent in
that tier and 5 in the orbit tier; the table itself seeded them well this time, at 98 and
101 km/s, the cubic interpolant having improved its templates, and the scan agreed with the
seed. The four eclipsing runs go from 16 and 34 percent to 3 and 14. What still fails does
so for reasons the scan does not address: the 0.54-day contact pair at e = 0.21 with both
stars rotating at 100 km/s, whose lines are four resolution elements wide (the scan at a
near-circular start preferred the static minimum and was refused, and the fit from the
seed stayed 51 and 68 percent low); the 7-percent secondary of the 0.58-day pair, whose
semi-amplitude of 193 km/s the coarse scan put at 23 with a hold cost of 43 nats, above
the guard's 25, while the primary came back to 13 percent; the 0.72-day pair with
K1 + K2 = 28 km/s, one line-spread width, which no start repairs; and the 18.9-day pair at
e = 0.41 whose secondary rotates at 87 km/s, where a static broad secondary fits the
scan's resolution as well as the moving one. The 8.86-day pair started the orbit tier at the
table's 54 and 53 km/s (the truth 53 and 55), the scan found nothing better, and the fit
ended at 62 and 64 after its 100 L-BFGS steps: a fit that walked away from a good start,
decided by the step budget or the likelihood's own shape, not the scan.

The oracle tier is unchanged within its spread (medians 0.45 and 0.36 percent against
0.27 and 0.21), and under the AR(1) noise model the epoch-velocity pull rms falls from 1.5
and 1.4 to 1.1 and 0.94, and 93 to 95 percent of the 215 pooled epochs sit within three sigma of the injected velocity, with a median
absolute residual of 0.6 and 0.8 km/s against a median quoted error of 0.7 and 0.9. The
pooled rms residuals (18 km/s in this tier) are dominated by the two pairs whose lines
never separate, the 13/15 km/s pair and the contact pair, so the report quotes them beside
the medians. The blind tier remains limited by the period search: 3 of 14
found within 2 percent (5 before, the difference within the noise of a bootstrap whose
templates changed), and every failure downstream is an alias's.

The label fit's temperatures move where the disentangled spectra let them: the eclipsing
tier's errors fall from 257 and 205 K to 28 and 74, the orbit tier's from 210 and 326 to
166 and 260, the blind tier's primary from 388 to 155 K. The oracle tier stays at 196 K
for the primary: in eight of its fourteen runs the fit reports it "learned nothing about
teff_A" (a posterior width above 80 percent of the prior), the ten to
twenty-five epochs at S/N 14 to 100 per pixel carrying little temperature information in
this band once the prior is 300 K wide, so the residual is the prior's and not the
interpolant's.

**What the second run settles, and what it leaves.** The epoch velocities are a product
the report can be read from directly, and their errors are calibrated where the noise model
is the declared one: under the oracle and orbit tiers the pull rms of the usable epochs
sits at 0.9 to 1.5 and more than 80 percent of them lie within three sigma of the injected
velocity, which the first run's diagonal errors did not achieve. The range-K routes recover
the systems whose lines separate by more than about three resolution elements, from a
template table of a dozen epochs or from nothing; the remaining failures are of four kinds
a scan cannot fix: lines that never part (a combined semi-amplitude of one resolution
element), companions of a few percent of the light whose semi-amplitude the coarse
likelihood cannot weigh, rotation at a hundred kilometres per second in an eccentric orbit,
and a period the blind search did not find. The temperature scatter of the label fit was
the interpolant's, and the eclipsing and orbit tiers now measure temperatures to a few tens
and a hundred-odd kelvin; where the prior is 300 K wide and the epochs are few, the fit
correctly says it learned nothing. Taken up by the third run below: the period search (on
a floating mean with the disentangling deciding), the step budget (measured, 300), the
scanning law's phase (real epochs from GOST are now a cadence) and the ATLAS9 box above
7000 K (registered, not yet fetched); the lines that never part remain. The DR4 grid still
rests on the draft data model, and no run here was made on the released product.

### The third run (D63, 2026-09-11): the period search, the scan taken in turn, the step budget, and real epochs

Each of the second run's open items was first measured on its archived products and then
changed, and both populations were run again with the same simulated epochs (the same
seeds) under the changed code and a step budget of 300. The walls are again not
comparable: the desktop was shared with other jobs, and the workers were fewer.

**Measured on the second run's archive before any change.** (1) The blind tier's period
search loses the period
at candidate generation, not at the decision. On the oracle tier's velocity tables of the
same epochs (near-truth velocities with realistic errors, 32 usable systems), the true
period ranked 9, 9, 15, 25 and 103 among the distinct peaks of the classical Lomb-Scargle
periodogram on the five Gaia-like systems the search had lost, and 1, 1, 1, 10 and 3 on a
floating-mean periodogram of the same series: `scipy.signal.lombscargle` fits a sinusoid
with no constant term, and ten to twenty-five epochs falling into a dozen visibility windows
hundreds of days apart have a sampling window whose mean is large at most frequencies, so
the constant the model cannot fit is absorbed into the sinusoid and the spurious power buries
the truth. Six candidates were also too few (twenty is right), while the point-wise scaling
of the series by the square root of its weights (wrong, but small here) and the frequency
grid's density (ten per inverse baseline; refining it to fifty changed no outcome) were not
the cause. The orbit's chi-square, by contrast, decides well: where the truth was among the
candidates it won in 27 of 29 systems, usually by hundreds. Twenty candidates from the
floating-mean periodogram and twenty from its two-harmonic form, with the swap-invariant
peaks kept, take the recovery on those tables from 23 to 28 of 32; the four left are two
tables of ten or eleven epochs whose true Keplerian fits worse than an alias, a period 2.04
percent off, and one genuine decision loss by 4.4 in chi-square. (2) The 7-percent secondary
the scan had lost was not lost to the coarse model: at the pixel doubled the marginal
likelihood peaks at 203 km/s for an injected 193 with the primary at the truth, the same as
the full model, and prefers it over the static value by 15 to 18 nats. It was lost because
the companion's evidence points the right way only while the primary's semi-amplitude is
within about ten percent of the truth, whereas the primary's own peak falls by 50 nats
within twenty percent: on the product grid at ratio 1.8 no trial held the primary close
enough, the local refinement could not carry the companion out of the basin it had been
placed in, and the guard then refused the correct joint move of 39 nats as two
single-component holds of 15 and 16. (3) The fit is budget-limited, not landscape-limited.
The 8.86-day pair's orbit-tier fit, reproduced to the last digit at 100 steps, ends 8.8 and
8.4 km/s from the truth in the semi-amplitudes; at 300 steps 3.5 and 2.8, at 1000 steps 1.6
and 0.7, the eccentricity from +0.20 to +0.04 and the conjunction from 1.05 to 0.16 d, with
the gradient norm still 6500 times the tolerance at the end and the potential a poor proxy
for the progress (95 percent of its fall by step 300, the largest gain in K_B after it). The
velocity table's orbit, which is what the report quotes, saturates by 300 steps; steps 101 to
300 cost about 1.1 s each on 16 epochs. (4) The light fractions the orbit and blind tiers
measure by correlation against generic library templates were off by more than a factor 1.5
on 8 of 33 systems, in two modes with nothing between them: secondaries below 8 percent of
the light measured too bright (a 6.7-percent star at 0.40) and near-equal pairs measured too
faint (0.42 to 0.54 at 0.18 to 0.30). The sign of the error sets the sign of the
semi-amplitude error, a too-bright secondary collapsing K_B and a too-faint one inflating
it; the label fit inherits the failure rather than correcting it, and the second correlation
pass holds the light fixed, so nothing downstream re-measures it. The velocity table's
detection statistic for the secondary gates it (below 100 catches 11 of the 16 runs and 4
of 50 good ones). (5) One oracle-tier table of the second run had every epoch at the search
edge with no error: its disentangling had diverged (a single line-search acceptance at a
smoothness precision of 2e19, where the Cholesky solve loses every digit and the
quadratic form came out above the data term, a negative chi-square read as a 216,000-nat
improvement; at correlations 0.0, 0.1 and 0.5 the same fit converges), the label fit had
disowned its result and pinned a zero point on its scan bound, and the correlation stage
had searched a window that could not contain the truth while a fine-pass defect reported
a position five pixels from any it evaluated.

**What changed.** In the order the measurements above put them: (1) The period
search: `find_period` is the weighted floating-mean generalized Lomb-Scargle of Zechmeister
and Kurster (2009), with twenty peaks and a two-harmonic form; the pipeline proposes the union
of those two lists, the peaks of the first component alone (the source that survives a
companion the templates could not follow) and the swap-invariant peaks with their doubles,
fits the eccentric Keplerian at every distinct start, sets aside an orbit above the declared
semi-amplitude ceiling or eccentricity maximum (a lost companion's velocities had let a wrong
period win at 1537 km/s and e = 0.94), merges fits on the fitted period and names every rival
within 25 in chi-square; and the disentangling itself decides among the best few candidates
(four, which a later measurement kept it at; it ran on the Gaia population and on one
field system, the stage having been finished after the field blind tier had already run),
each declared with every semi-amplitude as the same range so the coarse grids match, located by
the coarse scans and compared on the prior-free marginal log-likelihood, because on the noisier
bootstrap tables the chi-square still chose absurd orbits inside the declared ranges (twins at
6.10 d sent to 0.248 d at K 233/239 and e 0.73). On the 32 archived tables the candidate list
and the chi-square alone recovered 28 against 23, the flag firing on three of the four misses
and on no recovery. (2) The semi-amplitude scan takes its axes in
turn, the first ranged component at a factor 1.25 with eight phases and each further one at
what has been located, refines jointly, takes each component once more over its whole grid,
tries the exchange-symmetric twin of the best, and moves the start only when the best beats
it by 25 nats jointly (a component whose own move is worth under 5 nats returns to its
start): about 270 trials against 446. (3) Between two scan passes each star's prior amplitude
is profiled at the located orbit through the model's light site and the hyperparameter
starts moved where the profile asks for a factor of two or more; the profile is exactly a
profile over the prior amplitude and not a light measurement (the marginal likelihood is
invariant under scaling a star's light by a factor and its smoothness hyperparameters by
the factor squared, to 5e-10 nats; as a light estimate it sat at 0.3 to 0.7 of the truth),
but it rejected the declared amplitude of the 7-percent secondary by 150 nats at the very
orbit the first scan had located, and the re-scan brought its semi-amplitude to 4 percent
where the second run had a static companion. (4) The step budget is 300. (5) The marginal
likelihood returns minus infinity on a negative chi-square and the model rejects a smoothness
ratio above e^30; the diverged fit now ends at z-score rms 0.97 within 0.1 percent of the
truth. (6) The correlation stage reports the position it evaluated, clamps its window,
writes nan for an unmeasured component and searches one window per template offset by its
zero point; the pipeline refuses a disowned zero point, marks a failed table in its file and
report, stops a star whose residual z-score rms exceeds 10, and refuses a declared table with
an unmeasured epoch. (7) Real transit times from GOST (`cadence="gost"`), not used in this run
so that its epochs stay those of the first two. (8) A BOSZ box from 7000 to 10,000 K,
registered and not downloaded, so the two hot Gaia systems stay out of this run as before.
(9) The library-template velocity table is a product of every run. (10) Three guards on
the velocity table, from the third run's own products and after its tables were measured
(the tables below predate them): the exchange of the two components by the orbit runs only
when their light fractions are within a factor 3 (on a 95/5 pair it had swapped 19 of 80
epochs on noise and moved the primary from 5 to 56 percent off), the table as measured is
kept beside the delivered one, the amplitudes are fitted freely when any smoothness precision
moved more than a factor 10 from its start (a secondary whose lines ML-II had shallowed by a
quarter was lost at the declared fraction and half recovered free), and a frame offset the
label fit did not learn no longer pins that component's zero point (a mostly-noise secondary
had carried a 46 km/s offset into every velocity).

**The Gaia DR3 orbits again: 14 systems, 46 runs, none failed.** The same fourteen
systems and epochs as the two runs above, at 300 steps. Median and the 16th to 84th
percentile over the systems of each tier; the second run's value follows in brackets where
it lies outside the spread. The blind tier was run last, after the decision by the
disentangling had been added; the eclipsing contact pair had failed at the correlation
stage's new window refusal and was run again once the zero points could be dropped instead.

| quantity | oracle (14) | eclipsing (4) | orbit (14) | blind (14) |
|---|---|---|---|---|
| abs(dK1 / K1) [%], disentangling | 0.59 (0.2 to 3) | 5.9 (2.3 to 51.5) | 1 (0.36 to 17.3) | 5.7 (0.44 to 44.0) [24] |
| abs(dK2 / K2) [%], disentangling | 0.49 (0.32 to 3.6) | 9.1 (2.8 to 39.2) | 2.2 (0.23 to 58.2) | 6.3 (1.1 to 56.0) [24] |
| K1 and K2 within 5%, disentangling | 93% and 93% [100] | 50% [75] and 50% [25] | 71% [64] and 79% [50] | 50% [21] and 50% [21] |
| abs(dK1 / K1) [%], velocity table | 0.53 (0.23 to 2.8) | 4.8 (1.1 to 31.0) | 0.59 (0.33 to 6.6) | 1.4 (0.26 to 32.4) [27] |
| abs(dK2 / K2) [%], velocity table | 1 (0.21 to 2.8) | 7.8 (2.4 to 56.2) | 0.9 (0.29 to 21.9) | 2.2 (0.2 to 58.8) [25] |
| components recovered in the other order | 0% | 0% | 7% | 14% |
| abs(de) | 0.0034 | 0.043 | 0.009 | 0.074 [0.20] |
| abs(d t_conj) [phase] | 0.0027 | 0.011 | 0.0029 | 0.17 [0.29] |
| epoch velocity rms, A and B [km/s] | 1.1 and 2 | 4.6 and 14.1 | 1.1 and 2.2 | 4.3 and 9.2 [20 and 31] |
| epoch velocity pull rms, A and B | 1.1 and 0.97 | 0.99 [2.4] and 1.2 | 1.3 and 1.1 | 3.7 and 3.6 [16 and 17] |
| epochs, pooled: median abs residual A and B [km/s] | 0.64 and 0.79 | 3.1 and 7.1 [1.6 and 8.7] | 0.59 and 0.83 | 0.96 and 1.5 [2.2 and 2.2] |
| epochs, pooled: within 3 sigma, A and B | 92% and 91% | 100% and 82% [78 and 56] | 86% and 83% [81 and 79] | 70% and 69% [59 and 57] |
| spectrum correlation, A and B | 0.89 and 0.89 | 0.86 and 0.86 | 0.86 and 0.86 | 0.9 and 0.8 |
| abs(dTeff), A and B [K] | 132 and 108 | 258 [27.5] and 166 | 136 and 328 | 186 and 320 |
| period search recovered within 2% | | | | 50% (7 of 14) [21%] |
| wall per star [s], 3 workers, shared machine | 403 | 580 | 654 | 671 |

The blind tier is where the run changed. With nothing declared but the spectra, the
period is found within 2 percent on seven of the fourteen systems against three, the median
semi-amplitude errors fall from 24 percent to 6, the table's from 27 and 25 to 1.4 and 2.2,
the epoch-velocity pulls from 16 and 17 to 3.7 and 3.6, and the eccentricity error from 0.20
to 0.07. The decision by the disentangling overruled the table's chi-square on four of the
seven misses and chose another alias each time. Rebuilding the whole chi-square ranking
offline from the archived bootstrap tables, undoing the component exchange those tables
carry, reproduces the run's own choice on 30 of the 33 blind systems and supersedes the
ranks first quoted here (4, 6, 6 and 14). It puts the truth first on every system the run
recovered, third on one miss, in the fifth to eighth places on one, in the ninth to
twentieth on one, and absent from the candidate list altogether on four
(d64_period_candidate_ranks.md). Raising the number compared from four to eight would
reach one further system in each population, so the default stays at four.

**The decision by the disentangling warns; it does not correct (D64, measured on the
archive).** The comparison scans the semi-amplitudes and the conjunction but scores each
candidate at a single point in period and eccentricity, the two quantities a sparse table
measures worst. On the one Gaia system where the truth was among the four compared it lost
by 251 nats, while the same model at the true period, eccentricity and conjunction beats the
winner by 185 (d64_period_decision_scan.md). Scoring each candidate over its whole declared
window does not work and cannot be afforded. The window maximum favours the shortest
period: peaks are spaced by about $`0.28\,P/T`$, so a three percent window holds some 50
independent trials for a candidate near the truth against 550 to 1500 for its shorter
rivals, and correcting that bias by extreme-value extrapolation leaves two estimators
disagreeing over the winner. The truth is found only when the semi-amplitudes are scanned
inside the period loop, which costs about 62 hours for four candidates against the 116
seconds of the point comparison, scaling as $`T/P`$. The stage has converted no miss into a
hit on either population and has cost none. Its four overrules on the Gaia blind tier all
landed on systems that ended on a wrong period, and three further misses drew no overrule,
so its flag reads as "this period is not to be trusted", not as "this period is better".

The orbit tier, with the period known and the conjunction scanned, comes back
within 5 percent in both semi-amplitudes on ten of the fourteen against nine before, its
medians from 3.5 and 4.2 percent to 1.0 and 2.2, and the 7-percent secondary, static in the
second run, is now 7.5 and 2.8 percent off: the prior-amplitude pass rejected its declared
amplitude and the sequential scan found it. What still fails: the 0.54-day contact pair
with both stars at 100 km/s (46 and 94 percent off, in every tier but the oracle), the 0.72-day pair whose lines never part, and the 18.9-day pair
at e = 0.41 whose secondary rotates at 87 km/s, where the marginal likelihood now prefers a
static broad secondary even under the oracle tier: at 300 steps the fit reaches a minimum
12 nats deeper than the second run's with K_B at 2 km/s against 35, a preference of the
likelihood at this resolution and not of the optimizer.

The oracle tier is unchanged within its spread, and its epoch velocities keep their
calibration (92 and 91 percent of the 215 pooled epochs within three sigma, median absolute
residuals 0.64 and 0.79 km/s). The eclipsing tier's medians are the contact pair's: the other
three systems sit at 2 to 6 percent, and the tier's epoch-velocity pulls fall from 2.4 and
2.8 to 1.0 and 1.2 with the correlation stage's window per template. The label fit's
temperatures moved the other way: the eclipsing tier's primary error rises from 28 K to 258
and the orbit tier's secondary from 260 to 328, with the fit reporting that it learned
nothing about the temperature on two more stars. The budget changed this through ML-II: at
100 steps the smoothness precision had barely left its start (300 to 400 on the twins); at
300 it reaches its optimum, 1e4 to 1e6 on most stars, and the disentangled components,
better correlated with the truth (0.82 to 0.93 in the eclipsing tier), carry less standard
deviation in their normalised flux. That fall is not a shrunken posterior mean whose lost
depth the label fit takes for dilution (the reading first given here; corrected in
d64_posterior_smoothing.md). The curvature penalty is identically zero at zero frequency
and an equivalent width is the zero frequency, so a component's own smoothness cannot move
its equivalent width: the transfer there is the data weight over the data weight plus the
ridge, which holds to 0.8 percent over 34 components, and the smoothness precision does not
enter it. Most of the lost standard deviation is noise the smoothing removed rather than
line, the equivalent width over the same window moving by 1.021 on mixed-0008's primary,
whose standard deviation fell to 0.797. What does move an equivalent width is the coupling
between the components, the off-diagonal blocks of the accumulated normal matrix, which
account for a median 0.952 of the archived departure against 0.003 for any per-component
filter. The orbits improved and the labels did not; both follow from the same change.

**The field population again: 19 systems, 59 runs, none failed once the window was centred.** The same nineteen
systems and epochs as above, at 300 steps. One orbit-tier run of the 3.99-day twins had ended
in the optimizer's initialisation, twice: the semi-amplitude scan had moved the conjunction
half a period on (its phase grid holds the antipode the 41-point phase scan cannot sample,
672 nats better there), the second scan pass gained exactly nothing so the final phase scan
did not run, and the one-period window was still centred on the phase scan's best, which put
the start on the window's edge where the unconstraining transform is infinite; with the
window centred on the start the fit carries, the star converges to 0.2 and 0.1 km/s of the
truth. The blind run of the 553-day pair had failed at the window refusal and was run again
with the zero points dropped.

| quantity | oracle (19) | eclipsing (2) | orbit (19) | blind (19) |
|---|---|---|---|---|
| abs(dK1 / K1) [%], disentangling | 1.5 (0.12 to 20.1) | 0.92 (0.8 to 1) [2] | 1.5 (0.29 to 73.4) | 4.9 (0.3 to 51.3) |
| abs(dK2 / K2) [%], disentangling | 0.98 (0.25 to 6) | 0.2 (0.17 to 0.23) [1.2] | 1.7 (0.21 to 36.1) | 9.9 (0.42 to 78.7) |
| K1 and K2 within 5%, disentangling | 58% and 74% [84] | 100% and 100% | 58% [63] and 58% | 53% and 42% [47] |
| abs(dK1 / K1) [%], velocity table | 2.7 (0.26 to 56.1) | 0.85 (0.51 to 1.2) [1.6] | 4.6 (0.39 to 29.0) | 5.8 (0.36 to 74.2) |
| abs(dK2 / K2) [%], velocity table | 1.1 (0.18 to 19.3) | 0.093 (0.062 to 0.12) [0.89] | 6.9 (0.2 to 78.6) | 9.2 (0.23 to 72.2) |
| components recovered in the other order | 0% | 0% | 16% [26] | 16% |
| abs(de) | 0.0034 | 0.012 [0.01] | 0.0041 | 0.044 |
| abs(d t_conj) [phase] | 0.0035 | 0.00085 | 0.0092 | 0.095 |
| epoch velocity rms, A and B [km/s] | 2.2 and 2.7 | 2.6 and 2.7 | 3 and 2.7 | 4.6 and 6.2 |
| epoch velocity pull rms, A and B | 1.4 and 1.1 | 1 [1.3] and 0.98 [1.2] | 1.2 and 1.3 | 1.9 and 1.8 |
| epochs, pooled: median abs residual A and B [km/s] | 0.91 and 1.2 | 1.8 and 2 [2.6 and 1.9] | 0.95 and 1.3 | 1.2 and 2 |
| epochs, pooled: within 3 sigma, A and B | 91% and 92% | 100% and 100% | 85% and 86% | 76% and 77% |
| spectrum correlation, A and B | 0.94 and 0.79 | 0.95 [0.91] and 0.94 [0.91] | 0.92 and 0.85 | 0.91 and 0.79 |
| abs(dTeff), A and B [K] | 124 and 187 | 117 and 215 [104] | 150 and 412 | 256 and 430 |
| period search recovered within 2% |  |  |  | 58% (11 of 19) [63%] |
| wall per star [s], 3 workers, shared machine | 506 | 779 | 908 | 757 |

This population did not improve, for the reason the two runs above gave: the seven systems
with periods above 130 days and K1 + K2 below 65 km/s, whose lines never
part by more than two resolution elements, are the worst of every tier and swing from run to
run (the oracle tier's primaries 6 to 59 percent off in both), and the twelve short-period
systems were already within a few percent at 100 steps, where they stay. Two things did
change, both in the velocity table. The orbit tier's table errors rise from 1.2 and 1.8
percent to 4.6 and 6.9 because three tables lost a component, and each was traced on the
archived products to a different cause: on the 95/5 pair the exchange step swapped 19 of 80
epochs on noise draws of the faint component's velocity and moved a primary measured to 5
percent to 56; on the 2.09-day pair the disentangled secondary, shallower by a quarter as
its smoothness precision went from 400 to 2000, was lost under the declared light fraction
and half recovered under a free amplitude; on the 553-day pair the label fit had placed a
mostly-noise secondary's rest frame 46 km/s from the systemic velocity. The three guards
that followed (an exchange only between alike light fractions, free amplitudes once the
smoothness has moved, no zero point from an unlearned frame offset) came after these tables
were measured. And the blind tier's period search found eleven against twelve, the
difference one long-period system whose 2 percent window is the grid's own resolution.
Those eleven are the candidate list and the chi-square alone: the decision by the
disentangling did not run on this population, the stage having been finished after its
blind tier had, and the one field system it ran on is the 553-day pair, rerun afterwards,
where it overruled the table and chose another alias.

**What the third run settles, and what it leaves.** On Gaia's own epochs, ten to
twenty-five per star over five years, the blind route now finds the period on half the
double-lined orbits and recovers both semi-amplitudes within 5 percent on half, from a
fifth; with the period known, ten of fourteen come back within 5 percent on both, and the
faint secondary among them. The epoch velocities, which are the product, keep their
calibration where the orbit is right (pulls of 1.0 to 1.3 in the oracle, eclipsing and
orbit tiers) and read as failed where it is not, rather than as numbers. Three things the
run measured: the fit is budget-limited, and the budget cannot be read off the potential;
the divergence the second run had recorded was the arithmetic of the marginal likelihood at
a smoothness precision of 2e19, not the noise model, and is guarded; and the disentangled
components change with the budget in a way the label fit is not told about, which its
temperatures pay for. The mechanism is not ML-II shrinking the lines: no per-component
smoothing can move an equivalent width, and the coupling between the components accounts
for a median 0.952 of the departure (d64_posterior_smoothing.md). Open, in the order they
should be taken:
the label fit compared through the operator the disentangling actually applies, which is
the coupled one over both components and not a per-component smoothing; the candidate
periods themselves, since the decision among them scores each candidate at a single point
in period and in eccentricity, the two quantities a sparse table measures worst, and was
afterwards measured to have converted no miss into a hit in fourteen systems; a light measurement,
which cannot come from the disentangling at all because the likelihood sees only the
products of light and line depth; and the lines that never part, which no route here
reaches. The phase grid that samples its own antipode was taken up and is done. The label
fit, the light measurement and the velocity-table failures were taken up afterwards, on this
run's archive, in the D65 subsection below.

### After the third run (D65, 2026-09-17): measured on its archive, not rerun

Everything in this subsection was measured offline on the third run's archived products, and
the code it describes postdates the tables above: no benchmark has been rerun with it.

**The third run's label temperatures measure the warm start, not the likelihood.** Its label
fits stopped near the node the warm-start scan chose, at the 80 L-BFGS steps allowed.
Started at the injected labels instead, the same fit
reaches a lower value of its own objective on 8 of 12 products and lands a median 7 K from
the truth (d65_label_likelihoods.md). The oracle tier's label priors are also centred on the
truth, and the optimiser's unconstrained-space Jacobian pulls a weakly constrained label to
the centre of its prior, so a label the data barely constrain looks accurate there.

**Compared against the epochs rather than against the disentangled components, and converged,
the label fit halves its temperature error.** The disentangling's own statistics give the
chi-square of a template pair against the epoch spectra in closed form, exact for the
correlated noise and independent of the declared light. Minimised by a bounded
Levenberg-Marquardt with restarts over 22 products, it returns a median temperature error of
60 K against 119 K for the previous comparison converged the same way, surface gravity 0.040
and 0.134 dex against 0.060 and 0.692 for primaries and secondaries, metallicity 0.006
against 0.024 dex, and formal errors close to calibrated for temperature, gravity and
metallicity (d65_converged_labels.md). It is now the default of `Fit.match_labels` and of the
pipeline's label stage.

**The same comparison measures the light fraction, which the disentangling cannot.** On the
orbit tier its light fraction misses the injected one by a median 0.011, against 0.043 for
the fraction the light stage measured by correlation before the disentangling, and it is
closer on 10 of 11 products; the exception is the product whose archived orbit is wrong. The
pipeline reports it beside the declared value and flags a disagreement. It does not replace
the declared value in the velocity measurement, because the velocity templates are the
disentangled components, whose line depths only the declared light reproduces.

**Rotation was biased by smoothing that nothing declared, in the data and in the model, and
both are now accounted for.** The delivered epochs carry the delivery interpolation, the
detector pixel and the simulator's own grid on top of the nominal resolving power,
11.670 km/s on the DR4 product and 11.896 on DR3 against 11.07; the benchmark now declares that width. Separately, the
disentangling's own shifts and pixel boxes on the model grid add seven twelfths of the
squared grid spacing, which the label fit read as slower rotation (6.9 against 11.0 km/s on
the pipeline's closed-loop test at its default grid); the epoch comparison now removes it by
default (d65_grid_smoothing.md). On three products the median rotation error falls from
+3.06 to +0.63 km/s, with some 0.9 square km/s of width still unaccounted for.

**Two of the five blind systems the period search never reached are recovered by changes to the
velocity table, and two cannot be.** Epoch by epoch (d65_velocity_table_failures.md), one twin
at 5.36 days had a single exchanged epoch of twelve, recovered by proposing the peaks of the
leave-one-epoch-out periodograms on tables of at most 25 epochs; a 7 percent companion at 553
days was undetected at 24 of 26 epochs and wrecked the joint fit, recovered by giving a
companion's velocity no weight where its detection statistic is under 100. Over the 33 blind
tables the truth reaches rank 1 on 22 against 20, none lost. A pair of 100 km/s rotators at
0.54 days measured with unrotated templates, and a nine-epoch table on seven nights, are not
recoverable from their tables; the pipeline now flags a table on fewer than eight nights. A
near-twin at 310 days whose lines blend at 55 of 62 epochs is recovered only by an assignment
rule that costs two other systems, and is left open.

**The detection gate's threshold of 100.** Measured with the code as implemented over the 33
blind tables, the gate changes 9, those with a companion velocity below 100. The 7 percent
secondary at 553 days was detected by the library templates at no epoch; its velocities
spread across the whole search window at statistics of 1.4 to 61 and, fitted jointly, had
carried the orbit from the true period to 511 d. The gate takes that system from absent in
the chi-square ranking to rank 1, at 547.4 d against a true 552.5 d. It moves one Gaia
system from rank 20 to rank 8, takes one field system from rank 24 to absent, and costs no
system its rank 1: 21 systems rank the true period first against 20, and the
leave-one-epoch-out candidates add the 22nd. At 25, the table summary's own threshold for a
weak detection, the rule fails on the 553-day system, because six epochs with statistics
between 25 and 61 keep secondary velocities up to 212 km/s off. On the field population 100
is also where blends part from measured epochs, which the table's blend flag does not mark at
this resolution: on one system all 55 epochs whose lines are less than one line
width apart lie below it and all 7 separated epochs above. The first component is never
gated. Gating every component was measured too and cost a system: a field system whose
first template fell below 100 at 5 of 16 epochs had its true period go from rank 2 of the
chi-square ranking to absent, and the top four of that ranking are what the period decision
compares.

**The optimiser's change of variables does not bias the orbits it recovers, and a hard
eccentricity bound freezes some that it does not.** Removing the log-Jacobian that
`run_map` minimises along with the posterior moves the archived orbits by a median 0.007
formal sigma and at most 0.23; the pull is larger only where a semi-amplitude has collapsed
into the bottom 5 percent of its range, which marked 7 products of which 6 were failed
orbits (d65_run_map_jacobian.md). Nine archived fits, all failed orbits, sit on the
eccentricity limit of 0.9, where the model's bound is an infinite barrier that stops the
line search, so those fits never left it.

Open, in the order they should be taken: a fourth run with these changes, which is the only
way to turn the offline numbers above into benchmark numbers; a blend flag that fires at this
resolving power (the present one fired on none of 86 blended epochs); an eccentricity
bound that does not freeze the fit; the assignment rule
measured inside the period decision; the remaining width on the RVS products; the lines that
never part; a GOST-cadence run; the hot box; and the DR4 grid against the release.
