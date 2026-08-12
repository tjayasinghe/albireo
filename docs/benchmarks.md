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
