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

Machine: same Windows 11 laptop (32 GB RAM), CPU only, float64. The M5 scale gate
("2×10⁵ px / 50 epochs samples in minutes on one GPU") is *projected* from CPU
measurements here — no GPU on this machine; the actual GPU run is the one open
item, flagged for hardware the maintainer controls.

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
author should be contacted before any redistribution). Head-to-head numbers
pending that build.

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
