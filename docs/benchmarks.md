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

## M4 — realism: tellurics, SB3, per-epoch light, LSF widths, K₂ scan (2026-08-11)

Machine: same Windows 11 laptop, CPU only, float64, jitted θ-path throughout. Every
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

### Closed loops (the M4 acceptance gate — one per feature)

All at the gate scale of M3 (n = 490 px, 10–14 epochs, SNR 110–150, topocentric,
gaps + cosmics), MAP/ML-II point recovery:

| Feature | Recovery | Wall time |
|---|---|---|
| **Tellurics** (3rd component, topocentric) | K₁ 0.07%, K₂ 0.41%; telluric spectrum RMS 0.047, corr 0.989 (135 core px) | 57 s |
| **SB3** (hierarchical, 14 epochs over 2.2 P_out) | K₁ 0.10%, K₂ −0.08%, K_AB 0.39%, K_C 0.50%; e_in ±0.004, e_out ±0.008, P_out rel 9e-5; observable combination RMS 0.007 | 76 s |
| **Per-epoch light** (12 epochs, 3 in partial eclipse, ℓ inferred with flat Dirichlet) | ℓ₁ per-epoch rms 0.0028 (eclipse epochs ±0.004); K's −0.77% / −0.14%; **each component individually recovered** (core RMS 0.010 / 0.012) | 48 s |
| **LSF widths** (2 instruments, reference pinned) | σ_B −0.67% (asserted <3%); K's <0.2% | 36 s |
| **K₂ scan** (ℓ₂ = 0.1 companion, 15-point grid) | peak exactly at injected K₂ = 38; contrast > 4000 in D over the scan edges; companion line pattern corr 0.977, offset-removed RMS 0.05 | ~3 s/scan |

The per-epoch-light row closes the M2 story: the k = 0 additive indeterminacy that
capped constant-light component recovery at the ~0.1 level is broken by eclipse
epochs whose light fractions are *inferred*, not supplied — component spectra come
back individually at the 0.01 level with ℓ(t) recovered to 0.003.

### Negative results worth as much as the positive ones

1. **Absolute LSF widths are unidentifiable in a template-free model.** ML-II with
   both instruments free inflates σ by +35% / +13% (trading against intrinsic line
   widths; K's unaffected at <0.2%). With one *reference instrument* pinned, the
   other width recovers to <1%. Policy recorded as D25; same honest-anchor
   philosophy as the light ratio (D13).
2. **The K₂-scan null is negative.** On a companion-free dataset, D(K₂) ∈
   [−544, −465] over the whole grid: the marginal likelihood's Occam term charges
   for the extra marginalized component and nothing pays for it. Detection
   thresholds remain empirically calibrated (math.md §6), but the baseline is
   *repulsive*, not neutral.
3. **The faint companion's envelope is prior-dominated.** At ℓ₂ = 0.1 the §5.1
   low-frequency degeneracy is amplified by ℓ₁/ℓ₂ = 9: the recovered companion
   carries a ~+0.19 constant offset (its mean blanketing is absorbed by the
   primary) while the line pattern is intact. Reported honestly in math.md §6;
   line-pattern quantities are the deliverable of the scan.
4. **A second exact k = 0 mode appears with tellurics** (telluric constant vs.
   common stellar constant, since Σℓ = 1): measured offsets +0.030 / −0.029,
   cancelling to 0.001 in the light-weighted sum. Ledger row added (§5.4).
5. **Injected tellurics must be representable on the model grid**: sub-pixel
   telluric lines behind a 7 km/s LSF are resolution-limited (recovery ceiling
   RMS ≈ 0.2 against the raw truth no matter the SNR or epoch count) — a
   simulator-configuration lesson, not a solver limitation.

## M5 — scale, benchmarks, release readiness (2026-08-11)

Machine: same Windows 11 laptop (32 GB RAM), CPU, float64. The M5 scale gate
("2×10⁵ px / 50 epochs samples in minutes on one GPU") was *projected* from these CPU
measurements. It has since been run on a real GPU (below, 2026-08-14): the CUDA path
works and scales as predicted, but the gate does **not** close on the consumer card
available here, for two measured reasons — 16 GB against a gradient that needs 18.24 GB,
and fp64 at 1/50 of fp32. The gate stays open, now with a specific hardware requirement
rather than an open-ended one.

### Three scale pathologies, found and fixed (D27)

The first attempt to evaluate the marginal likelihood at survey scale
(n = 31,750 px × 2 components, 50 epochs, half-bandwidth 513) failed with an
**82 GB** allocation. Three distinct causes, each now fixed and regression-guarded
by the exactness suite (identical log-likelihoods to 12 digits before/after):

1. **Closure-captured data arrays.** `jax.jit` of a method closing over the
   problem embedded every data array as an XLA constant, and constant folding of
   the θ-independent graph exploded. Fix: `Problem`/`EpochGroup` are registered
   pytrees, and the jitted marginal takes the problem as an *argument* (runtime
   parameter). Planned temporaries at the failing size: unchanged symptom until…
2. **Unrolled probe batches.** Comb probing applied 2p+1 = 1027 matvecs in 16
   unrolled `vmap` chunks with no data dependence between them — XLA scheduled
   them with overlapping live ranges, and its own buffer analysis planned
   **79.6 GB** of temporaries. Fix: probing is a *sequential* `lax.scan` over
   batches (combs generated from offsets inside the body), which forces buffer
   reuse: planned temporaries dropped to **2.1 GB** (38×) and compile time fell
   ~5× (one scan body instead of 16 unrolled copies). The scatter-based block
   assembly (O(n·p) index maps, ~8 GB at design scale) was likewise replaced by
   per-block gathers under `lax.map` (O(B²) transients).
3. **Scan stores the backward pass.** Reverse-mode through the probe scan saved
   every batch's forward intermediates: gradient memory grew back to the unrolled
   total — **469 GB requested** at the design target. Fix: `jax.checkpoint` on the
   batch body; the backward sweep recomputes each batch (~1.5-2× backward probing
   cost) and gradient memory stays at the outputs array plus one batch.

One trade landed and was corrected the same day: unconditional remat + serialized
small batches **cost up to 2× NUTS wall time at small scale** (tutorial run
72 → 141 s; gate test 156 s vs ~102 s M3 baseline — bit-identical posteriors,
caught by the tutorial smoke runs). The mechanism is instructive: XLA's parallel
execution of independent unrolled probe batches — the very thing that overlapped
80 GB of live buffers at scale — is a multi-core *speedup* at small scale. Batch
size and remat are now size-adaptive on the probe-output footprint (64 MB
threshold): small problems run all probes as one parallel batch without remat
(gate test back to 112 s), large problems get the sequential remat scan. The
prior factor also gets its own small block size instead of the posterior's (its
bandwidth is 2 per component; factorizing it at block 513 doubled Cholesky cost
for a determinant that is nearly free).

### Design-target ladder (CPU, jitted, fixed bandwidth p = 513, 50 epochs, SB2)

| n (model px) | native px/epoch | eval | ∇ eval |
|---|---|---|---|
| 31,750 | 13,134 | 24.7 s | 102.7 s |
| 74,331 | 33,134 | 52.5 s | 249.9 s |
| 135,063 | 66,467 | 96.9 s | 466.0 s |
| **203,497 (design target)** | 106,467 | **149.7 s** | **729.9 s** |

Both scale linearly in n at fixed bandwidth (~0.75 ms/px eval, ~3.4 ms/px
gradient), as the O(n·p²) flop count predicts; log-likelihood values are
bit-identical across all three solver revisions. Peak memory stays within the
laptop's 32 GB at every size. A single design-target marginal evaluation — the
"give me disentangled spectra at this orbit" operation — is thus **2.5 min on a
laptop CPU**; posterior sampling at this scale is the GPU's job (below).

### GPU projection (stated as projection, not measurement)

At the design target a NUTS run needs ~2,600 gradient evaluations (150+250
transitions × 6.5 mean leapfrogs, the measured M3/M4 pipeline numbers). On this
CPU that is 2600 × 12.2 min ≈ three weeks — out of reach, which is exactly why
the design brief targets GPU. The dominant costs (batched probe matvecs; 800-step scanned Cholesky
of 513² blocks) are dense, batched, and fp64; on a single A100-class device the
same graph is expected to run the gradient in ~1-3 s (probe batches become large
GEMM-like work, the block Cholesky ~0.1 s of batched `potrf`/`trsm`), putting a
converged posterior at **1-2 hours for the widest-bandwidth massive-star config,
and tens of minutes at moderate bandwidths** (p ~ 200: flops drop ~6×). The
`probe_chunk` knob (raise on GPU) and `remat=False` (80 GB HBM fits the stored
backward) are the tuning levers. These projections close only with a real GPU
run — deliberately left open in this record.

### First real GPU run (2026-08-14): the path works, the gate does not close

Hardware: NVIDIA GeForce RTX 5070 Ti, 16 GB, driver 595.97, under WSL2 Ubuntu with
`jax 0.11.0` on the `cuda12` backend (`jax.default_backend()` → `gpu`,
`[CudaDevice(id=0)]`). Blackwell needs no special handling. Same
`scripts/m5_scale_bench.py` graph as the CPU ladder above.

**What works.** The CUDA path runs and is linear in *n*, exactly as the flop count says:

| n (model px) | native px/epoch | eval | ∇ eval |
|---|---|---|---|
| 4,877 | 1,702 | 0.232 s | 0.236 s |
| 9,526 | 3,457 | 0.447 s | 0.461 s |
| 18,221 | 6,965 | 0.915 s | 0.871 s |
| 31,734 | 13,062 | **out of memory** | — |

**Why the gate stays open, in two independent numbers.** Neither is a bug, and neither is
fixed by waiting.

*Memory.* The run dies at 31,734 model px — **one sixth of the design target** — on a
single 7.42 GiB request, with 13.8 GiB free and preallocation disabled. That is not a
surprise in hindsight: the D29 table below already measured the gradient needing
**18.24 GB** at the design target, which no 16 GB card has. Worth recording that the GPU
asks for ~2.5× the *CPU* peak at the same size in one contiguous buffer, so the CPU
figures are a floor for GPU sizing, not an estimate of it.

*Arithmetic.* A GeForce card runs double precision at a fraction of its single-precision
rate, and albireo's solver contract is float64. Measured here on a 4096³ matmul:

| | GFLOP/s |
|---|---|
| float32 | 39,023 |
| float64 | **783** |

**A 50× penalty.** 783 GFLOP/s of fp64 is a good desktop CPU, not an accelerator — which
is why the eval times above beat this laptop's CPU by only about 2×, rather than the order
of magnitude the projection assumed. The projection was not wrong about the *graph*; it
assumed A100-class fp64, and consumer silicon does not have it.

**So the acceptance gate is now open for a stated reason rather than for want of
hardware, and the requirement is specific.** "One GPU" is not the spec. The spec is
**≥ 24–40 GB of device memory and a 1:2 fp64 ratio** — A100 or H100 class. On that
hardware both blockers lift at once: 80 GB clears the 18.24 GB gradient with room to
switch `remat=False`, and ~10–20 TFLOP/s of vector fp64 is 12–25× the card measured here.
The 1–2 hour projection above is therefore still the projection to beat; nothing measured
today contradicts it, and nothing measured today confirms it either.

What this does close is the *portability* question, which was never separated out before:
albireo's graph compiles and runs correctly under CUDA with no code change, on a consumer
card, on Windows via WSL2. That is worth knowing independently of the throughput gate.

### The hand-set light-ratio systematic, quantified (`scripts/m5_light_ratio_demo.py`)

The LB-1/HR 6819-type failure mode, measured on a seeded simulation (the paper
asset behind the planned HR 6819 headline case):

1. Disentangling at a hand-set wrong ℓ rescales the recovered line depths by
   exactly ℓ_true/ℓ_assumed — measured affine slopes 1.44 / 0.97 / 0.59 against
   predictions 1.50 / 1.00 / 0.60 (assumed ℓ₂ = 0.2 / 0.3 / 0.5, truth 0.3),
   with the separate additive k≈0 envelope offset isolated by the fit. Line
   depths feed log g / luminosity-class diagnostics: this is the debate's engine,
   reproduced.
2. The marginal likelihood profiled over ℓ₁ **with hyperparameters refit by
   ML-II at every trial** (like for like) is flat to <0.5 log-units across
   ℓ₁ ∈ [0.50, 0.85] under constant light — the data carry no light-ratio
   information, and a first attempt with *fixed* hypers showed O(10–100)
   spurious curvature that was purely prior-mediated (wrong ℓ forces rescaled
   spectra, which a fixed prior scale penalizes — a subtle way for an analysis
   to fool itself, now documented). With three partial-eclipse epochs the same
   profile peaks at the true ℓ₁ = 0.70 with Δlog L = −145 at ±0.05.

### fd3 comparison harness (`scripts/fd3_bench.py`)

The fd3 v3.1 input format was reverse-engineered from the official example files
and the C source (documented in the script header): ln-λ master matrix with a
`# ncols X nrows` header, a comment-free stdin token stream for the control file,
ω in degrees, per-epoch σ only (no per-pixel weights, no masks — the benchmark
therefore runs gap-free so neither code is handicapped), component B's RV sign
applied internally (identical to albireo's ω+π convention), and fd3's internal
c = 299,800 km/s. The harness simulates an SB2 *on the common log grid fd3
requires* (no resampling for either code), writes fd3 separation- and fit-mode
inputs, runs the albireo side, and compares component spectra (raw and
mean-aligned — both codes carry a k≈0 freedom) and wall time. The fd3 side needs
the binary (~1.9 MB source tarball, GSL, builds on Linux/WSL; **no license is
stated** on the fd3 page — v2 was GPL, v3's GPL statement was removed — so the
author should be contacted before any redistribution).

### fd3, head to head (2026-08-14)

fd3 is now built and the numbers are measured. The tarball ships a prebuilt binary that is
**32-bit i386** and will not run on a modern x86-64 host, so it was rebuilt from source
against conda-forge GCC and GSL under WSL2 Ubuntu. It is **not** vendored into this
repository, and should not be: the distribution states no license.

**The build was validated against the author's own shipped outputs before it was used for
anything.** The tarball includes `.mod` / `.res` / `.rvs` for four worked examples, so
reproducing them is a real regression test across a different compiler, a different
architecture and a different GSL:

| example | fd3 wall | max abs. difference from the shipped `.mod` |
|---|---|---|
| `art_single` | 0.14 s | **0** (exact) |
| `art_double` | 2.52 s | 1.3 × 10⁻⁶ (the files are written to ~6 dp) |
| `art_triple` | 3.58 s | 1.0 × 10⁻⁹ |
| **`V453_Cyg`** (1344 px, a real published system) | 62.6 s | **0** (exact) |

**The comparison**, on the harness's seeded SB2 — 20 epochs, SNR 100, a common ln-λ grid so
neither code resamples, no gaps and no masks so neither is handicapped, and the orbit fixed
at truth for both:

| | comp 1 RMS | comp 2 RMS | steady-state wall |
|---|---|---|---|
| **albireo** | 0.0118 | 0.0165 | 0.182 s |
| **fd3** | 0.1767 | 0.2597 | **0.111 s** |
| albireo, mean-aligned | **0.0093** | **0.0116** | |
| fd3, mean-aligned | 0.0198 | 0.0223 | |

Three things, and the first is not in albireo's favour.

**fd3 is faster: 1.64× in steady state, 5.7× from cold** (0.630 s including JAX
compilation). It is a small C program that starts, solves and exits, and against a
1200-pixel two-component separation that is exactly the regime where a compiled direct
method should win. What is worth saying is that albireo is in the same class rather than an
order of magnitude behind — the harness's original "3.93 s" figure was its un-jitted
single-solve path, and quoting it would have overstated the gap by 20×. Timings are min of
five repeats, both codes on CPU.

**fd3's raw error is ~15× larger, and about nine tenths of that is a constant.**
Mean-aligning collapses comp 1 from 0.1767 to 0.0198. That is the *k* = 0 freedom both codes
carry and neither can determine from constant-light data — the same null space that
[§5.1](math.md#51-the-low-frequency-degeneracy-the-undulations-theorem) is about, and the
reason the literature's workflow includes a hand renormalization against an external light
ratio. albireo's smoothness prior pins the offset to something usable; fd3 leaves it to the
user. Neither is wrong. The difference is where the assumption is written down.

**On shape, once that offset is removed, albireo is about 2× more accurate** (0.0093 /
0.0116 against 0.0198 / 0.0223) — from the prior doing real work at low *k*, which is the
whole design.

And the difference that no handicap can equalize: fd3 returns a point estimate. It has no
uncertainty on the component spectra at all, which is the gap
[the roadmap](roadmap.md) exists to close and what
[the handoff tutorial](tutorials/downstream.md) turns into an error bar on log *g*.

Still outstanding for a complete benchmark page: a clean-room shift-and-add implementation
from the published algorithm (the existing code carries no license either), and a run on
AI Phoenicis, where the eclipse makes the light ratio externally known and removes the one
genuinely free choice in disentangling.

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

## M5 speedup pass — direct band assembly + closed-form gradient (2026-08-11, D28)

Same machine, same ladder configuration, same seeds; log-likelihoods agree with
the M5 record to machine precision (several rows bit-identical, the rest at
~1e-15 relative — the assembly changes only the floating-point summation order).

### Where the time actually went

A stage-split profile at the 31.7k ladder row attributed **22.3 s of the 24.3 s
evaluation (92%) to comb probing**; the block Cholesky was 0.86 s, everything
else noise. Probing pays 2p+1 = 1027 matrix-free operator applications — the
*union* of all epochs' band offsets — even though each epoch only contributes a
~50-pixel-wide band at a velocity-determined offset. That redundancy, not the
factorization, was the entire scale problem.

### What replaced it

1. **Direct per-epoch band assembly** (`albireo/assembly.py`, math.md §4.5): each
   epoch's (i,j) block is ℓᵢℓⱼ·T(δᵢ)ᵀ·G·T(δⱼ) with G = KᵀRᵀW′RK a narrow band —
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
   Takahashi banded selected inverse and d̂-outer products — reverse mode never
   walks the Cholesky/solve scans, and assembly-side reverse work shrank further
   by hoisting the velocity-independent G stage out of the rematerialized scan
   and by expressing row translation as clip-safe `dynamic_slice` (its transpose
   is a contiguous copy, where a gather's transpose is a scatter). Verified
   against plain autodiff at 1e-13 relative and by finite differences.

### Ladder, before → after (CPU, same laptop, jitted, p = 513, 50 epochs, SB2)

| n (model px) | eval before | eval after | ∇ before | ∇ after |
|---|---|---|---|---|
| 31,734 | 24.7 s | **3.0 s** (8.2×) | 102.7 s | **10.6 s** (9.7×) |
| 74,322 | 52.5 s | **8.0 s** (6.6×) | 249.9 s | **24.9 s** (10.0×) |
| 135,052 | 96.9 s | **14.5 s** (6.7×) | 466.0 s | **56.5 s** (8.2×) |
| **203,440 (design target)** | 149.7 s | **26.0 s** (5.8×) | 729.9 s | **111.3 s** (6.6×) |

(`scripts/m5_scale_bench.py`, single sequential run, no external load.) A
design-target marginal evaluation — "give me disentangled spectra at this
orbit" — is now **~26 s on a laptop CPU** (was 2.5 min), and a gradient **under
2 min** (was 12 min). The gate-scale NUTS test rides along: ~65 s wall
(Laplace + warmup + 250 samples) vs ~102 s at the M3 baseline and 112 s in the
M5 record — the small-problem regression the probe-era size-adaptive policy
existed to prevent is simply gone, along with the policy's reason to exist. The
design-target gradient's working set turned out to *exceed* this machine's
32 GB (measured 34.0 GB — see the D29 pass below, which brought it to 18.2 GB
and, with it, the top row to 22.2 s / 87.4 s), which is where the ratio erosion
at the top row comes from. At realistic single-star bandwidths (HR 6819-like: p ≈ 160 rather
than the ladder's conservative 513) the same operations land at seconds per
gradient, which puts **full NUTS posteriors for real SB2 problems within reach
of a laptop CPU** — the GPU budget becomes headroom rather than a requirement.

### Found in passing: the Laplace mass matrix was built from a defective Hessian

Wiring the custom VJP into the suite surfaced two second-order facts. First, the
initial custom rule was first-order exact but second-order wrong (8e-3 relative):
its forward rule called the custom function itself, so Hessians re-entered the
custom boundary and lost the chol-mediated terms through the dropped cotangent —
fixed by inlining the primal in the forward rule, after which
`jacrev(jacrev(...))` agrees with plain autodiff to 1e-15. Second, and
independent of all M5 work: `jax.hessian` (forward-over-reverse) produces an
**asymmetric** Hessian on this stack even for the plain-autodiff path
(off-diagonal 0.566 vs 0.855 on the diagnostic problem), while
reverse-over-reverse matches central finite differences of the gradient to
8 digits at three step sizes. `laplace_inverse_mass` — the only forward-mode
consumer in the package — had therefore been symmetrizing a slightly wrong
matrix since M3. It now uses reverse-over-reverse (math.md §4.5). The mass
matrix is a preconditioner, so posteriors were never biased; warmup was simply
tuned from a mildly wrong curvature estimate.

### GPU consequences

The hot path is now vmapped dense convolution-like passes, contiguous dynamic
slices, and one 50-step scan with large per-step work — all GPU-native shapes;
the probe-era `probe_chunk`/`remat` tuning story is gone along with probing. The
remaining GPU-specific bottleneck is chain latency in the sequential block
Cholesky/Takahashi scans (~800 steps of 513² work at the design target);
the associative-scan (parallel-prefix) factorization that removes it is on the
ledger (D28), as are a custom-VJP band→block packing, a fully analytic
assembly VJP, and opt-in mixed precision. Projection (still projection, not
measurement): design-target gradient ~0.5–1.5 s on one A100-class device, a
converged design-target posterior in **tens of minutes**; the earlier 1–2 h
projection stands as the conservative bound.

---

## D29 memory pass — and a boundary bug it turned up (2026-08-12)

The D28 speedup left the design target *fast* but not *runnable*: measuring the
compiled executables rather than estimating them (XLA
`memory_analysis()`, which reports buffer assignment without allocating it)
put the design-target gradient at **34.0 GB against 32 GB of RAM**.

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
   velocity-independent it is computed once per epoch either way, so *any*
   batching costs exactly one extra `G` pass in the rematerialized backward —
   the size matters only for `vmap` width. Hence a two-regime default: hoist the
   whole pre-pass below ~1 GB (small and gate-scale problems pay nothing),
   otherwise batch to ~0.5 GB.
3. **Prior determinant by scalar pentadiagonal recursion**
   (`assembly.prior_logdet`) instead of factorizing a bandwidth-2 matrix as
   6,358 dense 64×64 blocks.
4. **Hoist `_pack_band`'s gather indices** out of the `lax.map` body — the band
   coordinates depend only on the within-block (row, col), never on the block
   counter — and rematerialize the body, so reverse mode stops stacking (B, B)
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

Wall clock, same laptop, same seeds — the pass was aimed at memory, but at the
design target it bought time as well:

| n (model px) | eval D28 → D29 | ∇ D28 → D29 |
|---|---|---|
| 31,734 | 3.0 → **2.82 s** | 10.6 → 11.22 s |
| 74,322 | 8.0 → **6.95 s** | 24.9 → 30.67 s |
| 135,052 | 14.5 → 14.90 s | 56.5 → 59.53 s |
| **203,440 (design target)** | 26.0 → **22.16 s** | 111.3 → **87.43 s** |

The middle rows' gradients get slower by 5–23%: that is the batched pre-pass's
extra backward `G` pass, paid where memory was not the binding constraint. At
the design target — the row that previously did not fit — halving the working
set wins outright, because the gradient was thrashing against RAM. Raise
`epoch_chunk` to the epoch count on a machine with memory to spare (or on GPU)
to trade back.

### The bug the memory work uncovered

Hunting allocation hot spots meant re-deriving the band layout from scratch,
which exposed a **correctness** defect in D28's assembly. `G = Kᵀ(RᵀW′R)K` is
built as a band image; `H` is exactly zero outside the model grid, but the LSF
convolution smears in-grid mass *outward*, writing band entries at column
indices that correspond to grid pixels that do not exist. The T-sandwich reads
those entries whenever an epoch's shift places a component's support against a
grid edge — `T(δ)` has no row there, so the contribution should be zero.

The trigger is precise: **the data-coverage margin, in model pixels, being
smaller than the LSF kernel radius.** Measured band-vs-probing, dense and
entrywise (probe is the symmetric ground truth; only the column side leaked, so
asymmetry is the sharp discriminator):

| coverage margin (model px) | kernel radius | max relative entry error | Δ log L |
|---|---|---|---|
| ~24 | 6 | 6.4e-15 | 2.7e-15 |
| ~8 | 6 | 6.4e-15 | 0 |
| ~4 | 6 | 6.4e-04 | 1.8e-07 |
| **0** | **6** | **6.8e-02** (asymmetry 6.8e-02) | **−57 nats** |
| 19 | 34 | 6.9e-04 | 0.16 nats |

The last row is the one to worry about: at a *fixed* grid, simply fitting a
wider LSF walks into it. And a margin of zero is not exotic — it is what you get
by choosing a model grid narrower than the observed range to fit a sub-region,
which is the documented pattern. Every pre-existing fixture happened to leave a
margin exceeding the kernel radius, so the weights vanish where the defect lives
and it is multiplied by zero; that is why 174 tests passed over it. Worth noting
that `scripts/m5_scale_bench.py`'s largest row sat about one pixel from the
cliff.

The fix masks `G`'s out-of-grid columns at the source (one static boolean per
group); the margin-0 case then agrees at 5.7e-15 with asymmetry 2.9e-16. Two
fixtures now pin it — `edge_covered` in the equivalence set, and a dedicated
model-grid-inside-the-data case — and the equivalence check gained an
**entrywise** dense comparison plus an asymmetry assertion: the log-determinant
and solve it previously relied on average a boundary defect away, and at the
0.3 Å margin that put it under a 1e-11 scalar threshold.

### Two more silent-wrongness fixes

- **Gradients through the Cholesky factor were identically zero.** `_solve_stage`
  returned the factor, but its closed-form reverse rule cannot carry a cotangent
  on it (propagating one is precisely the reverse pass through the factorization
  the rule exists to avoid). Anything differentiating `spectra_std` or
  `draw_spectra` therefore got a silent zero. `MarginalResult` now stores the
  precision and rebuilds the factor outside the custom boundary, where plain
  autodiff applies; the cost is one extra block Cholesky, paid only by callers
  that ask for the factor — the sampling hot path never does.
- **A few wide native pixels silently set the solver bandwidth.** `row_support`
  is a max over native rows, and it drives a cost quadratic in the block size.
  In real spectra a wide row usually means samples were *deleted* (telluric
  window, order or chip gap) rather than genuinely wide: edges sit at midpoints,
  so removing samples makes the two bracketing pixels absorb half the gap each.
  `build_problem` now warns, names the offending pixel, and states the remedy
  (mask with `ivar = 0`, or split into separate instrument labels).

### What is left

The band tensor and its cotangent (6.2 GB) still store both triangles of a
symmetric matrix, and `bt` and its factor (6.2 GB) are both genuinely live
across the Cholesky. Folding the band to one triangle, and assembling directly
into block storage so the band tensor never exists, are the recorded next
levers — neither is needed to run the design target, which now has ~13 GB of
headroom on this machine.

## HR 6819 — the first observed dataset (2026-08-12, D30)

The maintainer approved the download, so the stack met real spectra for the first
time: the 51 public FEROS exposures of HR 6819 (ESO 073.D-0274(A), PI Rivinius,
153 MB, anonymous), the same data behind every published analysis of the system.

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
3. **`prior_logdet` returned `nan` below `eta/tau ≈ 1e-13`** — an unguarded
   `sqrt` of a difference of like-sized terms, reachable by unbounded ML-II.
4. **The row-support warning quoted `int64` minimum** as its median whenever more
   than half an epoch lay outside the model grid: rows the rebin operator never
   touched carried a sentinel instead of being excluded.
5. **A region disjoint from the model grid failed as "empty rebin operator"**,
   with no indication of which of the two ranges was wrong.

### Continuum: why the fit is done in the log

A curvature penalty applied to the flux cannot track a multiplicative response.
Measured on one exposure over 3850–4750 Å, normalizing with a 150 Å linear-space
smoother left the 97th percentile of the normalized flux at **0.55–0.63** in the
worst 50 Å bins — a 40% error. Fitting `log(flux)` instead (a pure exponential is
a straight line, and straight lines are in the penalty's nullspace):

| smoothing scale | p97 of normalized flux, per 50 Å bin, 3850–4750 Å |
|---|---|
| 80 Å | 1.006 – 1.010 |
| 150 Å | 1.007 – 1.011 |

— flat across the whole 20× gradient, and insensitive to the knob.

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

slope 0.9993 — so `frame="barycentric"` with `v_bary = ESO DRS BARYCORR` is right,
and the 0.14 km/s floor is CCF precision plus real water-vapour variability.
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

MAP + ML-II from a conjunction-phase scan: 120 L-BFGS steps, **2820 s**, peak RSS
**2.6 GB**. The gradient norm plateaus around 4×10² and oscillates — `run_map`'s
absolute `tol` is unreachable at 3.7×10⁵ good pixels, which is why it grew a
`callback` (watch the parameters, not the flag).

Profiling the marginal likelihood around the MAP, in **two independent windows**:

| | A: 4380–4600 Å | B: 4120–4330 Å | Klement et al. 2025 |
|---|---|---|---|
| lines | He I 4388/4471, Mg II 4481, Si III 4552/68/75 | He I 4144/4169, Si II, Fe II | — |
| period [d] | 40.36583 ± 0.00045 | 40.37022 ± 0.00065 | 40.3261 ± 0.0013 |
| K<sub>pre-sd</sub> [km/s] | 63.314 ± 0.013 | 63.724 ± 0.019 | 61.15 ± 0.88 |
| K<sub>Be</sub> [km/s] | 1.985 ± 0.151 | 3.022 ± 0.153 | 3.90 ± 0.27 |
| eccentricity | 0.0302 | — | 0.0289 ± 0.0058 |
| whitened residual sd | 1.674 | 1.403 | 1 if calibrated |

Read in the right order, this is a useful result and **not** a publishable orbit:

* **Eccentricity lands at 0.2σ** — 0.0302 against 0.0289 ± 0.0058, on a nearly circular
  orbit, from spectra alone.
* **The quoted ± are statistical only, and they are wrong by at least an order of
  magnitude.** The two windows differ by 0.0044 d in period (5.6σ of their combined
  internal errors) and 0.41 km/s in K<sub>pre-sd</sub> (17.8σ). Window-to-window scatter is
  the honest lower bound on the error bar; the likelihood curvature is not.
* **The residual scatter says why**: 1.4–1.7× the assumed noise, i.e. real unmodelled
  structure. Candidates, in rough order of suspicion — the pipeline resampled these
  spectra onto a common step, so the diagonal `ivar` model is optimistic by construction;
  per-epoch continuum residuals; a Gaussian LSF standing in for FEROS's real one; and the
  Be star's disc emission violating the one-static-spectrum-per-component assumption over
  a 135-day baseline.
* **K<sub>pre-sd</sub> is 3.5–4% above the literature, consistently in both windows.**
  Worth stating that the sign is the physically expected one: the published values come
  from cross-correlation and Gaussian line fits, which blend the sharp lines with the Be
  star's broad, nearly stationary ones and are therefore biased *toward* the systemic
  velocity. Deblending should raise K. That is a hypothesis this run does not test, not a
  claim.
* **K<sub>Be</sub> is detected but not measured.** The profile is a clean peak — 83 nats
  above K = 0 in window A — yet the two windows give 1.99 and 3.02 against a literature
  3.90. Expected: at v sin i ≈ 200 km/s the Be star's reflex motion is ~1/50 of a line
  width, which `math.md` §5.1 identifies as exactly the unattributable regime, and its
  recovered spectrum is prior-dominated.

The next step is not NUTS. Sampling would return the same optimistic width around the
same systematics-limited point; what the run needs first is an honest noise model, a wider
window, and a check of whether the period offset survives a per-epoch continuum treatment.
The first of those is the subject of the next section — it was built, and it did not help
in the way one would hope.

---

## The jitter site, and what it did to HR 6819 (2026-08-12, D31)

D15 always allowed a per-epoch noise-inflation factor and nothing ever built it, so the
run above had to take its estimated `ivar` at face value and report a residual scatter of
1.4–1.7 as a caveat. `forward.with_jitter` and the `log_jitter` θ site close that. This
section is the measurement of what difference it makes, which is not the difference one
would want.

### First: the marginal counts resolution elements, not pixels

Profiling a single shared α at the previous MAP, against the standard-deviation-of-the-
whitened-residuals estimator that the tutorial used to recommend:

| | window A | window B |
|---|---|---|
| weighted pixels `N` | 373,813 | 356,928 |
| residual sd (the naive estimator) | 1.6743 | 1.4030 |
| profiled `α̂` | **1.6807** | **1.4088** |
| implied `p_eff = N[1 − (sd/α̂)²]` | **2,843** | **2,930** |
| model pixels `n_comp · n_pix` | 19,876 | 20,160 |
| resolution elements (FWHM = 4.16 px) | 4,779 | 4,846 |

Two things worth keeping. The correction is real and in the predicted direction
(`math.md` §3.2a: `α̂² = χ²/(N − p_eff)`, not `χ²/N`) — but here it is only 0.4%, because
`p_eff` is **not** the model pixel count. At `dv = 1.5` km/s the grid oversamples FEROS by
4.2×, and the ML-II smoothness prior is stiffer still, so of ~20,000 nominal spectral
parameters only ~2,900 are data-determined — roughly 60% of the resolution-element count.
Inverting the two estimators for `p_eff` costs nothing and is otherwise an awkward number
to get at; in the `tests/test_jitter.py` fixture, built with a deliberately weak prior, the
same inversion gives `p_eff/N = 0.09` and the naive estimator is 4.6% low.

Per-epoch factors then buy a great deal more than one shared factor: **+19,763 nats** (A)
and **+10,020 nats** (B) over the best shared α. The exposures are genuinely not equally
good — α runs 1.11–3.61 in window A, so the worst exposure carries 1/13 of the weight the
`ERR`-free DER_SNR estimate would have given it.

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

The noise model is now self-consistent — residual sd 0.997 in both windows, which is what
a jitter is for. Everything else got worse:

| | no jitter | with jitter |
|---|---|---|
| window A vs B, period | 5.6 × combined formal σ | **13.1 ×** |
| window A vs B, K<sub>pre-sd</sub> | 17.8 × | 10.0 × |
| window A vs B, K<sub>Be</sub> | 4.8 × | 3.7 × |
| period vs literature (A) | 28.9 σ | **80.7 σ** |
| eccentricity vs literature (A) | 0.2 σ | 0.8 σ |

The error bars grew by about α, as they should (1.3–1.7×). The *central values* moved much
further than that: the period shifted by 0.078 d, which is **174× the no-jitter formal
error** on the same window, and away from the published value.

### Why — and a check that it is not a bug

Evaluating both parameter vectors under both weightings separates "the reweighting really
moved the optimum" from "the optimizer walked somewhere the objective would not go":

| | weights α = 1 | weights α = MAP |
|---|---|---|
| θ from the no-jitter fit | **1,350,406** | 1,514,430 |
| θ from the jittered fit | 1,339,784 | **1,518,265** |

Each wins under its own weights — by 10,622 and 3,835 nats respectively. Both fits are
correct; they are answering different questions. Scanning the period under the *jittered*
weights confirms the surface has structure far wider than its own curvature: from the
no-jitter θ the conditional optimum is 40.3600 ± 0.0007, from the jittered θ it is
40.4442 ± 0.0007 — two optima 125 σ apart, with the second higher.

The mechanism is visible in which exposures got downweighted. Within window A,
`corr(α, phase along the baseline) = −0.25`: the noisiest exposures concentrate early, four
of the worst five sitting below phase 0.35 of the 134.7-day baseline. Downweighting one end
of the baseline is giving up period leverage at that end, and a period fit pivots about the
weighted centre of its data — which the reweighting moved from phase 0.581 to 0.605. The two
solutions do behave like a pivot: they agree on a conjunction at BJD 2453221.7 (phase 0.615,
within 1.4 d of that weighted centroid) and diverge on either side of it.

### What to take from this

* **The jitter site works and is worth having.** It is exact (jitter α is bit-equivalent
  to being handed `ivar/α²`), it profiles to the dof-corrected estimate, and it turns "the
  residuals are 1.67× too big" from a caveat in prose into a fitted parameter.
* **It is not a repair for correlated residuals, and on this dataset it makes the point
  emphatically.** A diagonal noise model that has been rescaled is still a diagonal noise
  model. Here it whitened the residual *scale* while leaving whatever generates the
  structure untouched, and in doing so it relocated the answer by 174 formal σ.
* **The noise model selects the optimum.** That is the sharpest version of the lesson from
  the previous section. It is not only that the likelihood curvature understates the
  uncertainty — it is that two defensible noise models, fitted to the same data in the same
  window, disagree by far more than either one's stated error. Quote the spread across
  independent windows *and* across defensible noise models, or quote nothing.
* **The eccentricity survives, less impressively.** 0.0241 and 0.0213 against
  0.0289 ± 0.0058 — 0.8σ and 1.3σ, where the no-jitter run gave 0.2σ. Consistent, but the
  0.2σ was luckier than it looked.

## The response site, and the exoneration of the continuum (2026-08-12, D33)

D30's suspect list for the 1.4–1.7× residual excess had "per-epoch continuum residuals"
near the top, and its closing sentence asked for "a check of whether the period offset
survives a per-epoch continuum treatment". D33 is the treatment: the multiplicative
response coefficients D7 fixed at build time became a θ site.

### The swap

The response enters the *targets* ``z = y − r(R·1)`` and the sandwich weights ``w r²``,
not just the forward operator — the reason D7 deferred it. The swap is nonetheless exact
and cheap because ``R·1`` (the rebinned unit continuum, now stored per group) is
response-independent: ``z_new = z_old + (r_old − r_new)·R·1`` rebuilds the target with no
raw fluxes carried, re-masked so the D30 ``0·nan`` trap cannot resurface, and the
``Σ log w`` term is untouched because the noise lives on the data (math.md §7.5). The
traced Clenshaw matches ``np.polynomial.chebyshev.chebval`` operation-for-operation, so
`with_response` equals a fresh ``build_problem`` with ``r`` **bitwise** identical and the
marginal to rtol 1e-12 (`tests/test_response.py`).

### Closed loop (10 epochs, order 2, injected c ~ N(0, 0.03), SNR 130)

Joint MAP over orbit + hypers + 30 response coefficients: injected coefficient rms
0.0343, **difference-mode error rms 0.0020**, K errors +0.21% / −0.10%, and the fitted
response beats the unit response by 29,938 nats at the same orbit. The epoch-*shared*
mode comes out at its zero-centered prior, not at truth (c₀ error −0.069 against a 0.05
prior σ) — §5's response↔broad-features degeneracy, now measured rather than asserted.
The test asserts the common mode at prior scale, deliberately: tightening that assertion
would test the prior, not the data.

### The answer on HR 6819: the offsets are not the continuum's fault

`scripts/hr6819_response_run.py`: both windows, 150 L-BFGS steps per config, response
fits warm-started from the baseline MAP, order 2 per epoch (153 coefficients), prior
N(0, 0.02²). Uncertainties are conditional-orbit Laplace (nuisances at MAP), computed
identically for all four fits — they land ~2× the D30-recorded formal errors, which came
by a different route; immaterial, since every σ in this table exists to be dwarfed.

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

Three readings, in order:

* **The site works and the continuum was already good.** Thousands of nats of real,
  epoch-structured signal absorbed — by coefficients of a few *per mil*, with the
  epoch-to-epoch differences at 5×10⁻⁴ rms in both windows. `preprocess.normalize`'s
  log-space knot fit left half-a-per-mil of per-epoch continuum error on the table.
* **And nothing else moves.** The period shifts by +0.0004 / −0.0002 d (0.4σ / 0.2σ of
  the *formal* error — against the jitter site's 174σ relocation), K by < 0.001 km/s,
  the eccentricity by ≤ 0.0001, and the residual sd by 0.4–0.5% — the excess scatter is
  emphatically not continuum-shaped. Window-to-window disagreement is unchanged:
  ΔP 2.8σ → 3.1σ, ΔK<sub>pre-sd</sub> 10.5σ → 10.4σ.
* **So the continuum is crossed off D30's suspect list**, cleanly and by measurement.
  The surviving suspects for the correlated residual are the pipeline's resampling (the
  diagonal ivar is optimistic by construction), the Gaussian stand-in for FEROS's real
  LSF, and the Be star's variable disc emission. The recorded next steps are now a
  correlated-noise model and a wider window — the continuum treatment is done and keeps
  its place as a *hygiene* term, not a fix.

One more multimodality sighting, recorded because it keeps being the real lesson: window
A's baseline here reproduces the D30 record to 0.0002 d, but window B's uniform-procedure
MAP lands at P = 40.36091 — **0.0093 d below the D30-recorded 40.37022**, 6.6× the
combined formal errors, with both runs converged (parameters stationary to ~0.0002 d over
the final 20 steps). After the jitter relocation (174σ) and the two-optima period scan
(125σ apart), this is the third independent demonstration that this surface holds optima
far outside their curvature widths, selected by the optimizer's path. Every comparison in
the table above is therefore between fits sharing one procedure.

## The AR(1) chain: whiten the residuals *and* keep the orbit (2026-08-12, D34)

D31 ended with a warning — a rescaled diagonal noise model whitened the residual scale
and relocated the period by 174 formal σ — and D33 crossed the continuum off the suspect
list, leaving the pipeline's resampling correlations at the top of it. D34 models the
correlation: an AR(1) chain per epoch, `C = α²·D^(−1/2)·R_φ·D^(−1/2)` (math.md §1.4a),
with φ shared across epochs on the θ site (a property of the resampling, not of one
exposure) alongside the D31 per-epoch jitters.

### Closed forms, and the trap they avoid

The correlation matrix `R_φ` has *unit diagonal by construction* — heteroscedastic
pixels keep their supplied variances, and the residual-sd diagnostic is therefore
provably blind to φ. The discriminator is the lag-1 autocorrelation of the whitened
residuals: ~φ under diagonal whitening, ~0 under the chain whitener.

Masked pixels are exact, not approximated: a subset of a Markov chain is Markov, so a
gap becomes a single link with `ρ = φ^gap` (capped at build time, `ar1_max_gap=4` —
beyond it the chain restarts, which at |φ| ≤ 0.9 discards ρ < 0.66⁴ ≈ 0.2 in the worst
case and ~1e-3 at the fitted values below). The precision stays tridiagonal with
closed-form `log det`, and the whitener is the innovation transform
`(ε_i − ρ_i·ε_prev)/√(1 − ρ_i²)`. All of it is pinned against a dense reference that
shares nothing with the closed forms — the chain *correlation matrix* built from link
products and inverted with LAPACK: marginal log-likelihood to **rtol 1e-10** with gaps
and jitter composed, `φ = 0` reproducing the diagonal model to rtol 1e-12, gradients
against finite differences including exactly at φ = 0 (`jnp.power`'s nan-gradient at a
zero base; the gap-1 links use φ directly).

Two structural costs, both declared rather than discovered: the D28 band assembly
assumes diagonal weights, so a correlated problem auto-selects the **probe path** —
2p + 1 = 347 operator applications per evaluation with plain reverse-mode gradients,
~15× the D33 per-step cost as measured below (the tridiagonal band-sandwich extension
is a recorded lever, to be built only if AR(1) earns a permanent place); and the chain
couples pixels across masked gaps, so the solver bandwidth grows by a statically
reserved `ar_bandwidth_extra` (D21's declared-bandwidth philosophy — 5 and 6 model
pixels on the HR 6819 windows), behind an explicit `MarginalOrbitModel(ar1=True)`.

### Closed loop: φ and α jointly, neither poisons the orbit

Gate scale (10 epochs, SNR 130), injecting *both* miscalibrations at once — φ = 0.45
correlation and supplied `ivar` overstated by α² = 1.5²:

| | injected | recovered |
|---|---|---|
| φ | 0.45 | **0.4493** |
| α | 1.5 | **1.4868** |
| K errors | — | −0.05% / −0.31% |
| chain-whitened residual sd, lag-1 | 1, 0 | 0.944, −0.077 |
| same residuals, diagonal whitener: lag-1 | φ ≈ 0.45 | **+0.395** |

The 0.944 is not a miscalibration — residuals about the *fitted* spectra read low by
`√(1 − p_eff/N)` (math.md §3.2a, the D31 dof effect; p_eff/N ≈ 0.11 at gate scale),
and the marginal's own α̂ is dof-corrected anyway. The last two rows are the
discriminator working on identical residual vectors: the diagonal whitener sees the
injected correlation, the chain removes it.

### HR 6819: the noise model closes, the orbit stays

Same uniform procedure as D33 (conjunction scan, literature init, 150 L-BFGS steps),
θ = orbit + hypers + 51 per-epoch jitters + shared φ; ~53–56 s/step on the probe path
(8,369 / 7,948 s per window) against ~3.6–4.2 s/step for the D33 fits.

| | A: D33 baseline | A: D31 jitter | A: AR(1) | B: D33 baseline | B: D31 jitter | B: AR(1) |
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

Five readings, in the order they matter:

* **The noise model finally closes.** Both windows whiten in *both* moments — sd 0.997
  with lag-1 +0.041 and +0.012 — the first fits in this campaign to do so (D31 fixed the
  scale and left the structure; D33 fixed neither). The self-consistency check is almost
  embarrassing: the same residuals read through the diagonal whitener show lag-1 +0.797
  and +0.688, which is φ̂ (+0.801, +0.694) to two decimals — the model measuring its own
  necessity. In ML-II terms, window A's chain model sits **~1.7×10⁵ nats** above the D31
  diagonal-jitter model (1,691,463 vs the 1,518,265 recorded at D31's own MAP): the
  correlation term is not a refinement to the noise model, it *is* most of the
  miscalibration.
* **The D31 relocation does not recur, and α says why.** Under the chain the per-epoch
  jitters collapse from 1.11–3.61 to 1.55–1.93, and their medians — 1.66 and 1.40 —
  reproduce the D30 residual sds (1.674, 1.403) almost exactly. What D31's jitters were
  actually fitting was *correlated* structure, epoch by epoch, as if it were per-exposure
  scale — and the epochs where they over-fitted it were the ones whose downweighting
  moved the period. Modeled as correlation, the excess turns out roughly *uniform* across
  epochs — a pipeline property, exactly what the resampling hypothesis predicts — and the
  period lands within 0.006/0.009 d of the diagonal baselines instead of 0.08 d away.
* **The two windows move toward each other.** ΔP across the independent windows:
  0.00475 d at the baseline, **0.00159 d** under the chain — 3.0× closer, where the D31
  model widened it to 0.0145 d. K<sub>pre-sd</sub> spread is essentially unchanged
  (0.267 → 0.276 km/s). Every previous noise model made the windows agree less; this one
  is the first to make them agree more, which is the strongest available internal
  evidence that the chain is closer to the truth of these data.
* **K<sub>Be</sub> moves toward the literature and remains unmeasured.** 1.93 → 2.45 (A)
  and 2.95 → **3.76** (B), against 3.90 ± 0.27 — window B is now within 1σ of Klement
  et al. But the A–B spread *grew* (1.02 → 1.31 km/s) and A's value was still drifting
  at step 150. The ~1/50-linewidth reflex of the Be star (math.md §5.1's unattributable
  regime) does not yield to a noise model.
* **The literature period offset survives its third noise model.** 40.3712 and 40.3696
  against 40.3261 — 0.044 d, marginally *further* than the baseline. It is now measured
  not to be the continuum (D33), not the noise scale (D31), and not the pixel
  correlation (D34). Survivors: the Gaussian stand-in for FEROS's LSF, the Be disc's
  variability — and the published CCF analysis itself, which blends the components this
  model separates; that last cannot be adjudicated from here. What φ̂ = 0.7–0.8 does
  settle is the *scale* of D30's optimism: at these correlations a diagonal model
  overcounts low-frequency information by (1+φ)/(1−φ) ≈ 5.5–9×, which is why formal
  errors of ±0.0005 d coexisted with window disagreements of 0.005 d.

## D32 — the numpyro path stops baking the problem into the graph (2026-08-12)

D27 set the contract for `MarginalOrbitModel.marginal`: the `Problem` pytree is passed to
`jax.jit` as an *argument*, because closure-captured arrays are embedded in the graph as
constants, and XLA then evaluates every θ-independent subgraph over them at compile time,
against compile-time memory — ~80 GB of it at the design target when D27 first measured
it. The numpyro path never received that contract: the model closure from
`MarginalOrbitModel.model()` captured `self.problem`, so `run_map`'s jitted L-BFGS step
and the NUTS sample loop both compiled with the problem baked in. Recorded unfixed at
D30; fixed now, through numpyro's own machinery for exactly this case:

* the model takes the base problem as an optional argument and advertises it via a
  `model_args` attribute on the returned closure;
* `run_map` and `laplace_inverse_mass` build the potential with
  `initialize_model(..., dynamic_args=True, model_args=...)`;
* `run_nuts` runs `MCMC(..., jit_model_args=True)`, which regenerates the potential from
  the *traced* arguments inside the jitted sample loop (`hmc.py`'s `_potential_fn_gen`).

Every existing call site benefits without change (the runners resolve
`model_args` as explicit argument > model attribute > none). Calling the model with no
argument — any plain numpyro utility, e.g. `log_density` — falls back to the closure,
which is correct, just not compile-safe at scale; `model_args=()` forces that path
deliberately.

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

XLA names the mechanism unprompted — its slow-operation alarm fires during the closure
builds on exactly the predicted instructions: *"Constant folding an instruction is taking
> 1s: %scatter-add.342 = f64[50,126936] scatter(...)"* — θ-independent weight/response
subgraphs being evaluated at compile time, at 50 × native-pixels scale.

Two things stated honestly:

* **The memory cost is heuristic-gated, and that is not a defense.** At 31.7k px XLA
  folded ~2 GiB of derived constants into the executable; at 74.3k px its own folding
  guards declined the largest folds, so the memory cost happened not to materialize while
  the compile-time cost (13×) remained. Whether the D27 blow-up recurs at any given scale
  is a property of XLA's internal thresholds and version — the argument-passing contract
  removes the exposure instead of betting on the guard.
* **Folded constants are marginally *faster* at runtime** (5.9 vs 6.2 s at row 0) —
  that is the trade XLA is designed to make, and at survey scale it is the wrong one:
  the same mechanism costs tens of GB against the design target (D27), and a NUTS warmup
  recompiling per mass-matrix window would pay the folding repeatedly.

### Regression tests (`tests/test_inference.py`)

* `test_potential_with_model_args_embeds_no_problem_constants` — asserts on the jaxpr
  consts in both directions: nothing problem-sized with the problem as an argument, and
  the closure build must show the leak (so the probe is proven able to see it).
* `test_run_map_closure_and_argument_paths_agree` (float tolerance — different graphs)
  and `test_laplace_closure_and_argument_paths_agree` (exact — same eager ops).
* The M3 NUTS acceptance gate now runs through `jit_model_args=True` as its default
  path, so the gate itself regression-tests the traced sample loop.

## D35 — the band assembly learns the chain (2026-08-12)

D34 ran the HR 6819 AR(1) fits on the probe path at ~15× the D33 per-step cost and
closed with a condition: the tridiagonal band-sandwich extension gets built only if
AR(1) earns a permanent place. The science verdict — the only noise model that
whitens both moments *and* the first one to bring the two windows closer together —
earned it, and the next recorded step (a wider window) cannot be paid for on the
probe path: its gradient peaked at 23.7 GiB at *current* window scale on a 32 GB
machine. So the lever gets pulled.

### The extension

The only stage of the D28 assembly that assumed diagonal noise was the innermost
sandwich `H = RᵀW′R`. The chain adds one symmetric cross-row term per link —
`−c·√(wₙw_p)·rₙr_p·(RₙᵀR_p + R_pᵀRₙ)` — and re-weights the diagonal by the chain
diagonal `1 + Σ a` (math.md §4.5a). Both enter through the same machinery that
already carried the diagonal: a second set of static pair tables
(`operators.rebin_link_pair_tables`), built at `build_problem` time over the *union*
of links realized in any epoch, consumed by one extra `segment_sum` per epoch whose
traced weights carry φ, α and r; a per-epoch gap test selects each epoch's own links
against the shared tables, because masks differ by epoch. `H` widens by the group's
static `ar_step` — exactly the `ar_bandwidth_extra` the bandwidth reservation already
declared — and **everything downstream is untouched**: the LSF convolutions, the
T-sandwich, the band accumulation, the D29 chunking, and the custom-VJP solve see
only a slightly wider velocity-independent band image. The likelihood's auto-selection
becomes `"band"` unconditionally; probing remains the reference implementation and
the `validate` oracle.

Exactness carried over wholesale: the D34 dense-LAPACK gold test (gaps + jitter
composed) now runs the band path at the same rtol 1e-10, band = probe at rtol 1e-12
with ∂/∂φ agreeing to 1e-9, and the `epoch_chunk` batching is invariant — the AR
weight tuple (diagonal, link, gap table) pads and slices together, pinned by a test
because a batched run that dropped link terms would fail silently.

### Measured: the correlated marginal at HR-window scale

`scripts/d35_ar1_band_bench.py`, 51 epochs, 9,796 model px, 367,200 native px,
half-bandwidth 85, CPU, one assembly path per process (the peak-working-set counter
is monotone). Gradients in velocities, φ and the jitter — one L-BFGS step's work:

| | probe (D34) | band (D35) | ratio |
|---|---|---|---|
| eval, steady | 5.10 s | **0.71 s** | 7.2× |
| value+grad, steady | 58.83 s | **3.07 s** | **19.2×** |
| eval peak working set | 11.34 GiB | **1.14 GiB** | 9.9× |
| grad peak working set | 23.69 GiB | **1.85 GiB** | **12.8×** |
| log-likelihood, gradients | 1165077.330 | identical to all printed digits | — |

The gradient gap exceeding the eval gap is the D28 story repeating: the band path's
solve stage carries the closed-form custom VJP, while the probe path pays plain
reverse mode through 2p + 1 = 347 operator applications — the pre-D28 cost profile.
At 1.85 GiB, the wider window fits comfortably where 23.7 GiB was already pressing
against the machine.

### The proof on real data: window A, refit unchanged

`scripts/hr6819_ar1_run.py --windows A`, rerun with no changes beyond the assembly:
**868 s where D34 took 8,369** (9.6× end-to-end, 5.8 vs 55.8 s/step), converging to
the same optimum — log-likelihood 1,691,463.5 to the printed digit, φ̂ +0.801,
P 40.37113 vs 40.37115 (0.02 formal σ), α range 1.55–1.93 (median 1.66) and residual
diagnostics (chain sd/lag-1 0.997/+0.041, diagonal lag-1 +0.797) identical.
K<sub>Be</sub> reads 2.420 vs 2.446 — the direction both runs were still sliding
along at step 150, i.e. the flattest axis of the surface, not a path discrepancy.
The wider window stops being a budget question.

## D36 — one wide window: 4120–4600 Å, Hγ masked (2026-08-12)

The last lever recorded against this dataset (D30 asked for it; D33 repeated the
request), runnable only because of D35: `scripts/hr6819_wide_run.py` joins windows A
and B into a single fit — 22,169 model px and ~765k good native pixels, 2.26× window
A — including the 25 Å strip 4355–4380 that has never been in a fit. Hγ's core
(4325–4355 Å) is masked by `preprocess.mask_ranges`: ivar = 0 keeps the sampling
regular, the AR(1) chain restarts across the hole (a masked gap beyond
`ar1_max_gap`), the bandwidth is untouched, and the broad absorption wings — static
stellar features — stay in. Noise model and procedure are the D34 configuration
exactly. 200 L-BFGS steps, **2,447 s at 12.2 s/step** — linear-in-pixels from the
window A refit (5.8 s/step at 0.44× the size); the probe path would have priced this
at roughly 7 hours against a ~50 GB gradient, which this machine does not have.

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

Four readings:

* **The noise model closes at 2.3× the data.** sd 0.997 with lag-1 +0.019, and the
  self-consistency check lands a third time: the diagonal whitener reads +0.731
  against φ̂ = +0.737. φ̂ and the α̂ range sit between the two single-window values —
  what a pipeline property that varies mildly with wavelength should do.
* **K<sub>Be</sub> firms up toward the literature.** 3.48 against 3.90 ± 0.27
  (1.5σ), where window A alone said 2.42. The ~1/50-linewidth reflex is the
  data-starved direction (math.md §5.1), and it responded exactly to what it was
  starved of — more lines in one joint constraint.
* **The eccentricity stays put** — 0.0261, 0.5σ from the published 0.0289 ± 0.0058.
* **The period lands *below* both single-window optima** — 40.3675 against
  40.3711/40.3696 — not inside their interval: a joint fit is not an average of
  window MAPs (the cross-window coupling and new pixels are genuine information),
  and the standing multimodality lesson applies to any single optimum. It moves
  *toward* the literature and remains 0.041 d away.

**The literature period offset has now survived its fourth configuration.** Across
two independent windows, three noise models, and one joint wide fit: P ∈ [40.3675,
40.3712] — internally consistent to 0.004 d — against a published 40.3261 ± 0.0013,
with K<sub>pre-sd</sub> at 63.2–63.5, consistently 3.7–4% above the CCF value, in
the direction deblending predicts. Whatever separates this analysis from the
published orbit, it is measured to be none of: the continuum (D33), the noise scale
(D31), the pixel correlation (D34), or the window choice (D36). The surviving
suspects are the Gaussian stand-in for FEROS's real LSF — the next lever, and a v2
seam by design (D8: a tabulated LSF is a banded-operator swap) — the Be disc's
variability, and the published CCF analysis itself, which blends the components this
model separates.

## D37 — the tabulated-LSF seam opened: fitted σ(λ), and the orbit's answer (2026-08-12)

D8 reserved the seam ("tabulated LSF is v2 — a banded matrix, no structural
change"); D37 opens it. The kernel slot becomes a per-pixel profile bank realized
from per-anchor kernels through static log-λ interpolation tables
(`operators.convolve_varying`, exact adjoint pair, arbitrary asymmetric banks
accepted); the band assembly keeps its exact structure with scalar taps replaced by
row-shifted profile columns and the second sandwich application run against the
band-transpose of the first (G = Kᵀ(KᵀH)ᵀ, by G's symmetry — only left
applications broadcast on a row-major band image). Band == probe == dense at
rtol 1e-12/1e-10 under diagonal and AR(1) noise, per-anchor width gradients to
1e-9, random asymmetric banks pinned against a hidden kernel flip. 320 tests.

**What the closed loop measured first (gate scale, injected σ ramp 5.0→9.5 km/s):**
the joint fit leaves the orbit unbiased (K to 0.3%) and recovers the ramp's
*direction*, but the marginal does **not** prefer the injected truth — a flat width
beat it by ~3 nats, and the ML profile beat the truth by ~8 while sitting ~3 km/s
off one anchor. The physics: a stationary kernel change commutes with the shifts,
so the free spectra absorb it (deconvolution), and the width preference is
dominated by the smoothness prior's taste, not the instrument. Only the
anchor-to-anchor *variation* is data-identified, through the epoch-dependent
shifts. **Fitted anchor widths are diagnostics, not measurements** — the design
consequence is that the orbit's response, not σ̂(λ) itself, is the readout.

**HR 6819** (`scripts/hr6819_lsf_run.py`): the D36 configuration exactly — wide
window 4120–4600 Å, Hγ core masked, per-epoch jitters + shared AR(1) φ — plus 13
Gaussian width anchors every 40 Å (the FEROS order scale), bounds 1.5–3.5 km/s
around the nominal 2.652 (radius from the bound: half-bandwidth 91 vs 87). 200
L-BFGS steps, 5,650 s at 28.3 s/step (2.3× D36: larger radius, varying-kernel
band stages, 13-anchor VJP).

| | D36 (fixed σ = 2.652) | **D37 (fitted σ(λ) ×13)** | Klement et al. 2025 |
|---|---|---|---|
| period [d] | 40.36750 | **40.36769** | 40.3261 ± 0.0013 |
| K<sub>pre-sd</sub> [km/s] | 63.396 | **63.395** | 61.15 ± 0.88 |
| eccentricity | 0.0261 | **0.0262** | 0.0289 ± 0.0058 |
| φ̂ | +0.737 | **+0.737** | — |
| α̂ range (median) | 1.46–1.86 (1.61) | 1.46–1.86 (1.61) | — |
| whitened residual sd, lag-1 | 0.997, +0.019 | 0.998, +0.019 | 1, 0 |
| log-likelihood | 3,388,604.2 | **3,388,694.7** | — |

σ̂(λ) at the anchors [km/s]: 2.15 at 4120 Å, then 3.1–3.4 across the rest of the
window, hugging the 3.5 bound — the closed-loop behavior on real data: the marginal
buys smoother implied spectra with broader kernels, so the absolute level is
bound-limited and diagnostic only. K<sub>Be</sub> is deliberately absent from the
table: it did not converge in 200 steps here — it oscillated 1.30–4.16 across the
last 100 steps (a band that *spans* the D36 value 3.48), ending at 1.30 with
|grad| 59 where D36 ended at 3.85. The 13 near-flat width directions slow the
already-flattest axis; every tabulated quantity above was pinned over the same
trajectory (P within ±0.001, K₁ within ±0.02, φ̂ to three digits).

**The reading: +90.5 nats, and nothing moves.** The fitted width profile absorbs
real likelihood — there is wavelength structure in the effective width — and the
orbit does not respond: P +0.0002 d (0.5% of the offset, within the trajectory
wobble), K₁ −0.001 km/s, e +0.0001, φ̂ and every residual moment unchanged. This is
D33's pattern again (the response site absorbed +4,100 nats and moved nothing), now
for the LSF. **The literature period offset survives its fifth configuration**, and
LSF *width* variation joins the exonerated list: not the continuum (D33), not the
noise scale (D31), not the pixel correlation (D34), not the window (D36), not σ(λ)
(D37). The surviving LSF suspect is narrowed to profile *asymmetry* — the
first-order centroid channel, whose epoch-coupled part enters as an apparent
velocity perturbation ∝ λc′(λ)v(t)/c (math.md §1.3) — for which the operator
already accepts arbitrary banks; only a θ-parameterization (e.g. per-anchor
Gauss–Hermite h₃) would be new. Beyond the LSF: disc variability, and the published
CCF blending itself.

## D38 — the asymmetry lever, and the LSF exonerated in full (2026-08-13)

D37 left one LSF channel standing: profile *asymmetry*, the first-order centroid
effect a symmetric kernel cannot produce. D38 parameterizes it — per-anchor
Gauss–Hermite h₃ (`operators.gauss_hermite_kernel_traced`, |h₃| ≤ 0.2, h₃ = 0
bit-identical to the Gaussian machinery) behind an `lsf_h3` site — and closes it.

**The closed loop measured the identifiability first, and it is sharper than the
width case**: an injected h₃ ramp of ∓0.12 came back *flat* (fitted |h₃| ≤ 0.03)
with the orbit recovered to 1% — a free spectrum represents any static
centroid-warp field c(λ) ≈ √3·h₃(λ)·σ outright, so the data-identified remainder
is only the epoch-coupled sampling of the warp's gradient,
Δc ≈ c′(λ)·λ·(v − v_bary)/c ≈ **30 m/s** at this configuration (math.md §1.3) —
two orders below the ~4 km/s of accumulated RV signature the 0.041 d offset
represents. That estimate is also why the instrument-frame per-epoch kernel
realization is bounded out rather than built. The fixed-spectra data term *does*
prefer the injected profile (the injection is real and seen); band == probe ==
dense with h₃ anchors under diagonal and AR(1) noise; gradients in h₃ to 1e-9.
329 tests.

**HR 6819** (`scripts/hr6819_h3_run.py`): the D37 configuration plus 13 free h₃
anchors — 26 LSF parameters joint with the orbit and the D34 noise model. 300
L-BFGS steps (|grad| 44 at the end, better converged than D37's 200-step run),
10,789 s at 36.0 s/step.

| | D36 (fixed LSF) | D37 (fitted σ(λ)) | **D38 (fitted σ, h₃)** | Klement et al. 2025 |
|---|---|---|---|---|
| period [d] | 40.36750 | 40.36769 | **40.36719** | 40.3261 ± 0.0013 |
| K<sub>pre-sd</sub> [km/s] | 63.396 | 63.395 | **63.391** | 61.15 ± 0.88 |
| eccentricity | 0.0261 | 0.0262 | **0.0261** | 0.0289 ± 0.0058 |
| φ̂ | +0.737 | +0.737 | **+0.737** | — |
| whitened residual sd, lag-1 | 0.997, +0.019 | 0.998, +0.019 | **0.998, +0.020** | 1, 0 |
| log-likelihood | 3,388,604.2 | 3,388,694.7 | **3,388,721.3** | — |

ĥ₃(λ) at the anchors: interior anchors at the |h₃| ≤ 0.02 level, the largest
values 0.042–0.052 at three anchors including the data-starved blue edge — implied
centroid shifts of −0.13 to +0.31 km/s, a 0.53 km/s spread, all diagnostics by the
closed-loop measurement. +26.6 nats over D37 for 13 parameters (an order below the
widths' +90.5 — asymmetry has far less to absorb once the spectra are free, exactly
as the absorption argument predicts). K<sub>Be</sub> again did not settle on its
flat axis (2.73 at |grad| 44, inside the 1.3–4.2 band the D37 run wandered);
every tabulated quantity above was pinned.

**The reading: the LSF is exonerated in full, and the offset survives its sixth
configuration.** P moved −0.0005 d from D37 — inside the fit's own trajectory
wobble — and K₁, e, φ̂, α̂, and both residual moments are unchanged to the last
digit. Every instrumental channel this model can express has now been given a
θ-site and measured against the orbit: the continuum (D33, +4.1k nats), the noise
scale (D31), the pixel correlation (D34, +1.7e5 nats), the window choice (D36),
the LSF width (D37, +90.5 nats), the LSF asymmetry (D38, +26.6 nats) — **none
moved the period**. Across all six configurations P ∈ [40.3672, 40.3712],
internally consistent to 0.004 d, against a published 40.3261 ± 0.0013. The
surviving suspects are no longer instrumental: the Be disc's variability (a
time-variable component this static-spectrum model cannot express — and the lit
analysis's own systematic too), and the published CCF blending itself, which
measures velocities on composite line profiles this model separates. The
instrumental-systematics campaign on this dataset is complete.


---

## D40 — the nebular component, and per-pixel prior strengths (2026-08-13)

The first Tier-2 roadmap item. Everything here is from `tests/test_nebular.py` and
`examples/04_nebular.py`; the configuration is one SB2 in an H II region — 12 epochs,
SNR 220, 540 model pixels over 4838-4886 A, K = (58, 41) km/s, light fractions
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

The determinant recursion is the one that could have gone wrong quietly: `prior_logdet`
is an O(P) scalar Cholesky over the pentadiagonal prior, and generalizing `tau` and
`eta` to per-pixel changes every one of its three diagonals. It is checked against
`slogdet` of the dense construction with *random* profiles spanning 0.2-40 in curvature
and 0.1-1e4 in ridge, not against a uniform special case.

### What the contamination costs the spectra (orbit held at truth)

Two disentanglings of the same data with identical stellar priors, differing only in
whether the nebular component exists. The orbit is fixed at the injected values, so this
isolates the spectral claim.

| | truth | no nebular component | **with the component** |
|---|---|---|---|
| Hbeta core depth (light-weighted composite) | -0.506 | -0.375 (**26% shallower**) | **-0.508** |
| mean core error | 0 | **+0.154** | **+0.0015** |
| core RMS error | 0 | 0.155 | **0.0057** |
| Hbeta equivalent width [A] | 1.911 | 1.690 (**-11.5%**) | **1.908 (-0.14%)** |
| marginal log-likelihood | — | reference | **+81,424 nats** |

Both log-likelihoods are marginal, with the component spectra integrated out and the
Occam terms included, so their difference is a Bayes factor rather than a fit-quality
score: the extra component *costs* likelihood unless coherent signal pays
for it, and here it is paid 8.1e4 times over.

**Equivalent width is the number that matters**, because equivalent width is what
reaches the atmosphere code. An 11.5% error in a Balmer EW is a large error in log g,
it is systematic rather than random, and nothing in the current literature propagates
it — the disentangled spectra arrive at the next stage of the pipeline without an
uncertainty at all (roadmap.md, "where albireo sits").

`examples/04_nebular.py` adds the third treatment the literature actually uses,
**masking** the contaminated pixels (`ivar = 0` over +-150 km/s). It is honest and it is
not free: with the core deleted there is nothing behind those pixels but the prior, so
the composite comes back flat there and the product is incomplete exactly where a Balmer
gravity diagnostic is read. The single most useful figure on this page is that
three-way comparison, and it costs 9 seconds to produce.

### What the contamination costs the *orbit* (joint MAP, cold start)

The sharper result, and the one that was not expected going in. Same data, same priors,
same starting point, 300 L-BFGS steps of ML-II MAP over the orbit and the
hyperparameters (plus the 12 log-amplitudes when the component exists).

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
collapses by 59%, and the period and eccentricity go with it — a circular orbit reported
at *e* = 0.95, which is the eccentricity clip, not a fit. Only K<sub>1</sub> survives,
because 70% of the light pins it. The blind fit is also still wandering at 300 steps
where the modelled one has settled, and takes 3.6x the wall time to do it. (Neither sets
`MAPResult.converged`: that flag tests an absolute gradient-norm tolerance which, as D30
recorded, is unreachable at these pixel counts. The three orders of magnitude between
them is the readable statement.)

The per-epoch amplitudes come back with **correlation 0.99930** against the injected
ones and **0.0066 rms in log** — against an injected spread of 0.78x to 1.50x. They are
compared after centering, because only `a_j * d_neb` is observable and the geometric
mean is a convention (math.md §1.3); ML-II independently keeps the nebular component
less smooth than the stellar ones (log tau 7.9 against 11.4), which is the prior
discovering the shape it was told nothing about.

**The window profile is not cosmetic, and this is where that was measured.** The same
joint fit with the nebular component free across the whole grid — identical in every
other respect — lands K<sub>2</sub> at **+2.6%** instead of -0.29%, and the potential
250 nats worse. The freedom the profile removes was being spent absorbing stellar signal
at wavelengths where a nebula has no lines, which is the failure mode the component
exists to prevent, reappearing one level up. Measuring it also surfaced a defect that
would otherwise have been invisible: `MarginalOrbitModel` rebuilt the prior from the
sampled `log_tau`/`log_eta` and **dropped the profiles**, so a windowed component was
silently un-confined the moment ML-II was switched on. The profiles are structure, the
scalars are hyperparameters, and the merge now respects that (math.md §2).

### Readings

**The failure mode is worse than the literature describes, and the fix is cheap.** The
published concern is line-profile narrowing and biased atmospheric parameters, which is
real (-11.5% in EW). The orbit result says the contamination also propagates into the
*dynamical* answer — masses — through a 59% error in K<sub>2</sub>. Both are removed by
one extra component and twelve extra parameters, at 41 s against the blind fit's 120 s.

**Nothing downstream had to change.** The nebular column is one more column of A with a
different velocity law and a free amplitude, so the band assembly, the AR(1) link tables,
the chunking policy, the custom-VJP solve, and the D28 bandwidth contract are all
untouched; the per-pixel prior generalizes three diagonals and keeps the same O(P)
determinant recursion. That is the linear-Gaussian family paying for itself, and it is
the same reason Tier 3's time-variable component — of which this is the rank-one case —
can be a change of basis rather than a change of method.

**Two degeneracies were closed by convention rather than by data, and are recorded as
such** (math.md §5.4): the amplitude scale, pinned by centering the log-amplitudes, and
the nebular velocity, which decides where the component's lines land on the model grid
and is not a measurement. Neither is a defect; pretending either was inferred would be.



---

## D41 — calibrated faint-companion detection (2026-08-13)

The second Tier-2 roadmap item, in three pieces: vectorize the scan, marginalize K₁, and
calibrate the statistic by injection and recovery. Everything below is from
`tests/test_calibrate.py` and `examples/05_detection_limit.py`. The configuration is one
SB1/SB2 pair — 14 epochs, SNR 200, 717 model pixels over 5000-5060 A (520 native pixels),
K = (55, 40) km/s, light fractions (0.93, 0.07), P = 7.3 d, e = 0.12 — scanned on a
20-point K₂ grid from 14 to 71 km/s.

### The vectorized sweep

One batched `lax.map` over the trial grid, against the Python loop it replaces (one
jitted call and one device synchronization per point). Best of three, shared machine.

| model pixels | native | epochs | half-bandwidth | loop / point | sweep / point | speedup | relative agreement |
|---|---|---|---|---|---|---|---|
| 201 | 100 | 8 | 46 | 3.71 ms | 1.33 ms | **2.8x** | 1.1e-12 |
| 717 | 520 | 14 | 55 | 17.34 ms | 8.66 ms | **2.0x** | 1.4e-13 |
| 2,652 | 2,150 | 20 | 66 | 97.41 ms | 43.91 ms | **2.2x** | 1.8e-16 |

The factor is near-flat in problem size, which says it is the batching that pays rather
than the removal of per-point dispatch — the opposite of what a dispatch-overhead story
would predict, and the reason the acceptance gate asserts only 1.5x. It is not
bit-identical to the loop: batching re-associates the linear algebra, and the
log-likelihoods move in the last few digits. **What the factor buys is the two features
built on top of it:** a 7x20 (K₁, K₂) grid costs 0.88 s where the loop would need ~1.41 s,
and the 450-scan calibration below — 9,450 marginal solves — costs 53 s.

### Marginalizing K₁ against assuming a wrong one

The literature reports that a small error in the assumed primary semi-amplitude puts
spurious features in the recovered secondary spectrum. It does. What is not reported, and
is the more dangerous half, is the effect on the detection statistic.

| K₁ treatment | K₂ peak [km/s] | companion line-pattern correlation | D at the peak |
|---|---|---|---|
| correct, fixed | 41 | **0.961** | 40,609 |
| 5% high (57.75), fixed | 38 — one grid step low | 0.720 | **66,837** |
| 5% high, marginalized (σ = 5%) | **41** | **0.955** | 40,455 |
| 10% high (60.5), fixed | 41 | **0.486** | **135,410** |
| 10% high, marginalized (σ = 10%) | **41** | **0.931** | 41,310 |

**A wrong K₁ makes the detection look stronger while the answer gets worse.** Unremoved
primary signal is coherent across epochs; the companion's free spectrum is the only thing
that can absorb it, so it does, and D more than triples on the way to a recovered
spectrum that correlates 0.49 with truth. Marginalizing over a Gauss-Hermite rule on
N(μ₁, σ₁²) — applied to the companion *and* the no-companion model, so D stays a ratio of
two marginal likelihoods — recovers 0.93-0.96 and puts D back where the correct K₁ has it.
Correlations are offset-removed: at ℓ₂ = 0.07 the companion's smooth envelope is
prior-dominated (math.md §5.1-5.2), so the line pattern is what carries the information.

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

**The null peaks are strictly negative**, because the marginal likelihood charges an Occam
term for the companion's free spectrum and, with nothing to find, nothing pays for it. So
"D > 0" would have been a *conservative* test on this dataset — on another it might not
be, which is the whole argument for measuring the threshold instead of assuming one.

Two properties are enforced rather than hoped for. The threshold is defined through the
false-alarm estimator (1 + #{null >= D})/(N+1) rather than as a sample quantile:
`np.quantile` interpolates between order statistics and was measured leaving **8.3% of the
null above a nominal 5% threshold** on a 24-trial run — caught by a test, and
anti-conservative in exactly the direction that matters for a detection claim. And no FAP
below 1/(N+1) is reported; below that the rule degrades to "must exceed every null trial".

### One expected dependence that is not there

| injected K₂ [km/s] | 20 | 40 | 65 |
|---|---|---|---|
| limit on ℓ₂ | 0.292% | 0.296% | 0.297% |

The limit is flat in K₂. In an SB2 the components move in *antiphase*, so their relative
velocity never falls below about K₁, and at K₁ = 55 km/s the pair is well separated at
every trial K₂. A real K₂ dependence should appear only when K₁ is small enough that the
pair is barely resolved at any phase. Worth knowing before spending compute on a grid that
does not vary.

**What none of this checks is the model.** The null trials are drawn at the same K₁, orbit
and light fractions the scan assumes, so the threshold is self-consistent with those
assumptions and blind to their being wrong — the K₁ table above is exactly that failure,
and no calibrated threshold would have flagged it. The limit is likewise conditional on
the assumed companion template, since the observable is ℓ₂·d₂ and a featureless companion
is invisible at any light fraction. Both are stated wherever the numbers are.

---

## D42 — the free per-epoch radial-velocity table (2026-08-13)

Tier-2 roadmap item 3: no Keplerian, every epoch's velocity its own parameter. From
`tests/test_velocity_table.py`. One SB2 — 10 epochs, SNR 200, 400 model pixels at
dv = 6.00 km/s over 5000-5040 A (284 native pixels), K = (30, 55) km/s, light fractions
(0.6, 0.4), P = 6.31 d, e = 0.15.

### The zero point, and why it is removed

A free table has one arbitrary zero point **per component**: with no orbit tying the
stars together, each free spectrum absorbs a constant added to its own shifts. The
equality `T(d + D) x = T(d) [T(D) x]` is exact only for whole-pixel `D`, because the
model shifts by linear interpolation and a fractional shift blurs as well as translating.

| common shift applied to one component | change in log-likelihood |
|---|---|
| 1.00 model pixel | **4e-9 relative** (boundary effects only) |
| 0.10 model pixel | -7.3 nats |
| 0.01 model pixel | -0.11 nats |

So an uncentered table's absolute level is pinned by *interpolation error*, not by data —
a number that would look like a systemic velocity, move when the grid is resampled, and
mean nothing. albireo centers the pixel shifts per component, which makes the likelihood
exactly invariant:

| offset added to one component (relativistic addition) | change in log-likelihood |
|---|---|
| 5 / 50 / 200 km/s | **0.000e+00**, exactly |
| 0.5 km/s | -9.3e-10 (relative 9.8e-14, float64 round-off) |

Centering in *velocity* space instead is right only to `O(v^2/c^2)` and leaves a residual
four to six orders of magnitude larger (-9.9e-8 at 0.5 km/s, +8.7e-6 at 50 km/s). The
distinction is measured in the suite rather than asserted.

### Recovery, by starting point

250 L-BFGS steps of ML-II MAP over the 20 velocities and the four hyperparameters.

| start | per-epoch RV rms [km/s] | Wilson slope | potential |
|---|---|---|---|
| the Keplerian truth | 0.098 / 0.066 | -1.8255 | -9627.1 |
| K wrong by 10% | 0.106 / 0.063 | -1.8300 | -9627.6 |
| **K wrong by 30%** | **0.098 / 0.066** | **-1.8255** | **-9627.1** |
| truth + 15 km/s noise | 0.098 / 0.066 | -1.8255 | -9627.1 |
| cold start (every epoch at 0) | **12.94 / 28.70** | **+0.5916** | **+112,692** |

Truth is -1.8333 = -K2/K1, so the recovered mass ratio is **0.4%** off. An rms of
0.098 km/s is **1/60th of a model pixel**. Every warm start reaches the same optimum to
four decimals, including one 30% wrong in both semi-amplitudes.

**The cold start fails, and that is not a defect being hidden.** With every epoch at one
velocity the two components are indistinguishable, and the mode is documented as needing a
warm start. What matters is that the failure is *visible*: 122,000 nats worse, with a
Wilson slope of the wrong sign. A user comparing two runs cannot mistake it for a fit.

### Uncertainties — and the trap in reading them

| | mean sigma [km/s] | rms error [km/s] | error / sigma |
|---|---|---|---|
| raw Laplace diagonal | **37.95** | 0.098 / 0.066 | 0.002-0.26 |
| zero points projected out | **0.059** | 0.098 / 0.066 | 1.44 |

The raw number is `120/sqrt(10)` — the `Normal(0, 120)` prior divided by the epoch count —
**identical to four digits across both components and all ten epochs**. That is the
signature of reading a flat direction: the zero point's posterior width is the prior's, and
every epoch's marginal variance inherits it. It is 640x too large, and it would look
exactly as convincing on a useless dataset. `relative_velocity_errors` projects each
component's mean out; the projected block then has **exactly 2 zero eigenvalues** — one per
component, the identifiability claim confirmed numerically rather than argued.

The projected bars run ~1.4x optimistic against the realized errors, which is what a
Laplace approximation with the hyperparameters pinned at their MAP values should do.
Posterior samples of the `velocity_rel` deterministic need no projection and no Gaussian
assumption; that is the route the docs steer to, and this one is the fast estimate.

Per-epoch precision of 0.059 km/s is **1/102 of a model pixel**.

### The model check

`keplerian_residuals` centers both tables the same way and differences them in pixel
space, so the two arbitrary zero points cancel exactly (verified: offsetting the recovered
table by +77 and -31 km/s moves the residuals by < 1e-9).

| Keplerian tested against the recovered table | max abs residual | in units of the per-epoch sigma |
|---|---|---|
| the orbit that generated the data | 0.164 km/s | 2.8 |
| period wrong by 0.5% | 2.979 km/s | 50 |
| K_2 wrong by 5% | 2.581 km/s | 44 |

This is the mode's purpose: a Keplerian is a strong constraint, and a table fitted without
one says whether it was earned.
