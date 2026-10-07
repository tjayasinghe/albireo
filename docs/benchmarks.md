# Benchmarks & validation record

This page records correctness and performance results at each milestone. The numbers are
from the referenced tests, which reproduce them deterministically (fixed seeds). No claim
is made here without a test.

## The machine

Every table in this file was measured on one computer: an AMD Ryzen 9 9950X3D desktop,
16 cores / 32 threads, 32 GB, Windows 11 Pro build 26200, CPU only, float64. Tables
written before the D50 re-run label it a "Windows 11 laptop". It is the same machine.

Two statements in later sections assumed two machines and are incorrect:

* The 1.46× "residual clean-machine gap" between shift-and-add's recorded 0.018 s and the
  0.0263 s measured under the D50 protocol is not a hardware difference. Its cause cannot
  be determined. The earlier number's software stack, warmup and amount of heap
  contamination were not recorded, and contamination slows a run, so it does not explain
  a time below the uncontaminated one. The 0.018 s measurement is unreproduced.
* There was no hardware change behind "fd3 moved 12% across the hardware change". D50
  attributed part of that difference to OpenBLAS oversubscription. The remainder is
  unexplained.

The accuracy record does not depend on this. Those results are deterministic and were
reproduced exactly in the re-run, so accuracy is the durable comparison here and the
timings are not.

## The fixed-orbit linear solver (2026-08-11)

Machine: the desktop above, CPU only, float64, un-jitted (JAX 0.11). This is a correctness
milestone. Performance work (jit, custom band assembly, GPU batching) comes later.

### Exactness

| Check | Result | Test |
|---|---|---|
| marginal log-likelihood vs. dense brute-force marginalization (mixed instruments, masks, response) | rtol 1e-10 | `test_marginal_matches_dense_brute_force` |
| posterior mean & pointwise variance vs. dense | rtol 1e-7 | same |
| block-tridiagonal Cholesky / logdet / solves / Takahashi selected inverse vs. LAPACK dense | rtol 1e-10 | `test_solver.py` |
| comb-probe assembly reproduces the matrix-free operator | exact (validated per run) | `test_probe_assembly_is_exact`, `validate=True` |
| arbitrary values at masked pixels change nothing | rtol 1e-12 | `test_masked_pixel_values_do_not_affect_anything` |

### Closed-loop recovery (the acceptance test)

The simulated SB2 has SNR 100, 30 epochs (4 in partial eclipse), K = (45, 70) km/s,
e = 0.2, Gaussian LSF 4 km/s, model grid dv = 2.5 km/s (n = 2114 px) and 5% chip gaps +
cosmic-ray hits. It is solved at the injected nonlinear parameters
(`test_closed_loop_recovery_snr100_30_epochs`):

| Metric | Component 1 | Component 2 | Criterion |
|---|---|---|---|
| line-region RMS error | 0.52% | 0.73% | < 1% ✓ |
| whitened data residual variance | 0.975 (pooled) | | ~1 ✓ |
| posterior-draw std vs. Takahashi std (median ratio) | ~1.0 | | consistency ✓ |

One full marginal-likelihood evaluation (assembly + factorization + solve + logdet) at
this scale took 1.6 s of wall time (CPU, un-jitted, probe assembly = 141 operator
applications). The design-target scale (2×10⁵ px, 50 epochs) on GPU is the scale
criterion below.

### Nullspace / degeneracy analysis (empirical)

The low-frequency separation theory of [math.md §5.1](math.md) is verified
numerically. The posterior variance of the per-mode difference direction matches
`1 / (w l² (J − |g(k)|) + prior(k))` with `g(k) = Σ_j e^{ik ΔΔ_j}` within a factor of
~1.5 over a decade of spatial frequency, with the predicted ~5× variance inflation
toward low k (`test_low_frequency_degeneracy_matches_theory`, Hann-windowed modes).

The closed-loop test shows two properties, both covered by tests and documentation:

1. **The k = 0 additive indeterminacy.** The mean absorption depression differs between
   the components. With constant light ratios the data constrain only the light-weighted
   sum, and the unobservable difference (the `ℓ₁Δd₁ + ℓ₂Δd₂ = 0` direction) is set by
   the prior. This gives a ~1.5–2% systematic offset between components in this
   configuration, for any method. Four eclipse epochs (per-epoch light fractions) make
   the difference observable and remove the offset, demonstrating the degeneracy
   breaking of math.md §5.2 end to end.
2. **Sub-LSF scales are unrecoverable.** With a weak smoothness prior the posterior
   std is dominated by a flat ~5–7% contribution from deconvolution modes below the
   instrument resolution. The prior curvature scale must encode the true spectral
   smoothness (set manually here; ML-II is introduced with joint inference, below). This
   contribution is included in the reported variance.

## Joint NUTS inference (2026-08-11)

Machine: the same desktop, CPU only, float64. The whole θ → velocities → shifts → probed
marginal likelihood path (math.md §7.1) is now jit-compiled, with reverse-mode gradients
through the comb probing and the scan-based block Cholesky.

### θ-path exactness and gradients

| Check | Result | Test |
|---|---|---|
| jitted `MarginalOrbitModel.log_likelihood(θ)` vs. the fixed-orbit `build_problem` route (different but sufficient bandwidths) | rtol 1e-12 | `test_model_loglike_matches_m2_path` |
| `orbit_velocities(θ)` vs. the simulator's Kepler conventions | atol 1e-10 | `test_orbit_velocities_match_simulator` |
| ∂ log p/∂θ (K, t_conj, √e cos ω, log τ) vs. central finite differences | rtol 1e-4 (measured ~1e-7) | `test_gradient_matches_finite_differences` |
| bandwidth guard: epoch-realized shift excess ⇒ non-finite log-density | pass | `test_bandwidth_guard_rejects_out_of_bound_orbits` |
| chi-square sign guard (D63): a negative quadratic form ⇒ log-likelihood −∞ with a zero gradient; a healthy value and its gradient bit-identical | pass | `test_negative_chi_square_is_rejected_with_a_finite_gradient`, `test_sign_guard_leaves_a_healthy_evaluation_and_its_gradient_untouched` |
| smoothness bound (D63): log τ − log η above 30 ⇒ non-finite log-density | pass | `test_smoothness_bound_rejects_extreme_stiffness_ratios` |

At n = 1191 px × 2 components, 14 epochs and half-bandwidth bound 45 (probe stride 187),
the jitted marginal evaluation took 53 ms and value+gradient 222 ms, against 1.6 s
un-jitted at the larger fixed-orbit configuration. Jit alone gains about an order of
magnitude on CPU.

### The MAP → Laplace → NUTS pipeline

On the acceptance-test problem (below), NUTS with numpyro's default warmup (unit-scale
initial mass matrix) spends its early transitions at the tree-depth cap (2⁸ leapfrogs ×
~30 ms each), because the posterior scales span ~5 orders of magnitude (σ_P ~ 9×10⁻⁴ d
vs. σ_K ~ 0.06 km/s). It was still in warmup after >35 min. The shipped pipeline instead
runs three stages:

1. **MAP + ML-II** (L-BFGS on numpyro's unconstrained potential, hyperparameters
   included): 40 s, K's to 0.3% before any sampling.
2. **Laplace inverse mass matrix** at the MAP (6-dim dense Hessian, eigenvalue-floored):
   23 s.
3. **NUTS with mass adaptation off** (the supplied matrix would otherwise be
   overwritten by a poor few-sample estimate in the first adaptation window):
   150 warmup + 250 samples in 102 s, 0 divergences, mean 6.5 leapfrogs per
   transition.

The total is ~3 min for a converged orbital posterior on one desktop CPU.

### Closed-loop NUTS acceptance test

The simulated SB2 has SNR 130, 12 epochs, K = (30, 22) km/s, e = 0.2, ω = 0.7, LSF
7 km/s, grid dv = 5.5 km/s (n = 490 px), a topocentric frame with |v_bary| ≤ 25 km/s, and
gaps + cosmic-ray hits. The priors are a photometric-quality Normal on P and T_conj,
Uniform(−1, 1)² on the √e-pair (disk-constrained) and Uniform on the K's spanning ±50%.
Hyperparameters are set by ML-II (`test_nuts_gate_k_within_one_percent`):

| Parameter | Posterior mean − injected value | Posterior sd | Criterion |
|---|---|---|---|
| K₁ | −0.100 km/s (**0.33%**) | 0.058 km/s | < 1% ✓ |
| K₂ | +0.042 km/s (**0.19%**) | 0.091 km/s | < 1% ✓ |
| P | −7.1×10⁻⁵ d | 9.0×10⁻⁴ d | injected value in 95% CI ✓ |
| t_conj | +0.0021 d | 0.0023 d | ✓ |
| e | −0.0017 | 0.0011 | ✓ |
| ω | −0.0009 | 0.0072 | ✓ |

There were 0 divergences, and every injected value is inside its central 95% interval.
ML-II selected τ ≈ (330, 390) and η ≈ (0.5, 2.4), a weaker continuum anchor than the
manually set fixed-orbit value. This is expected, because without eclipse epochs the
k = 0 anchor is prior-dominated (see below).

### Posterior spectra under θ uncertainty

`posterior_spectra` mixes conditional Gaussian draws over posterior θ samples. In the
acceptance-test configuration (constant light fractions, no eclipse epochs) the
light-weighted observable combination ℓ₁d₁ + ℓ₂d₂ is recovered to < 2% RMS in line
cores, while the individual components show the expected scatter along the unobservable
k = 0 direction. This is the additive indeterminacy of the fixed-orbit section, larger
here because ML-II does not impose a tight continuum anchor that the data do not
constrain. The test asserts both properties (`test_posterior_spectra_from_samples`).

### Injection–coverage study

`scripts/m3_coverage.py` (fixed seed, reproducible) runs 24 injections with parameters
drawn from the sampling priors (disk + bandwidth-guard truncation replicated exactly).
Each has an independent random line list, so the spectral prior is misspecified by
construction and (τ, η) are refit by ML-II each time. Each is sampled with NUTS 150+250
through the MAP → Laplace → NUTS pipeline. The study took 101 min on the desktop CPU
(~4.2 min/injection).

| Site | cov68 | cov90 | mean \|z\| | rank-KS |
|---|---|---|---|---|
| period | 0.62 | 0.92 | 0.89 | 0.182 |
| t_conj | 0.62 | 0.96 | 0.82 | 0.151 |
| √e cos ω | 0.54 | 0.92 | 0.98 | 0.187 |
| √e sin ω | 0.75 | 0.92 | 0.87 | 0.166 |
| K₁ | 0.58 | 0.92 | 0.84 | 0.168 |
| K₂ | 0.67 | 0.96 | 0.82 | 0.109 |

Binomial 1σ at n = 24 is ±0.095 (cov68) and ±0.06 (cov90), N(0,1) gives mean |z| = 0.80,
and the KS critical value (α = 0.05) is 0.278. Every site is consistent with a calibrated
posterior: central-interval coverage is within 1.5σ of nominal (slightly over-covered at
90%), |z| is never above 2.05 across 144 site-checks, and the truth-rank distributions
are consistent with uniform. The underestimate of uncertainty from the empirical-Bayes
plug-in, the known cost of fixing hyperparameters at ML-II (math.md §7.3), is not
detectable at this sample size.

For point recovery across the prior, the median worst-of-(K₁, K₂) error is 0.74% and the
maximum 5.0%. The tail cases are draws with wide posteriors (K₂ ≈ 5–7 km/s at e ≈ 0.8,
where 5% is ~1.5 posterior sd), not misestimates. Divergences were 24 of 6000 post-warmup
transitions (0.4%), concentrated in the 3 injections with parameters against the hard
constraint boundaries (e = 0.75–0.95 at K₁+K₂ ≈ 70–76 km/s, the e_max and bandwidth-guard
boundaries), where reflecting trajectories are expected to diverge. Interior-of-prior
injections show none. The Laplace mass matrix remains adequate across the whole prior:
median 6.6 leapfrogs per transition, maximum 36. The strict MAP `converged` flag
(grad-norm < 10⁻²) was set in only 5/24. The tolerance is conservative, and MAP point
quality (K's to <1% typical) is unaffected.

## Realism: tellurics, SB3, per-epoch light, LSF widths, K₂ scan (2026-08-11)

Machine: the same desktop, CPU only, float64, jitted θ-path throughout. Every
number below is asserted (usually with margin) by a deterministic closed-loop test.

### θ-path exactness (new sites)

| Check | Result | Test |
|---|---|---|
| `with_light_fractions` vs. fresh `build_problem` (± telluric, constant & per-epoch) | rtol 1e-14 | `test_forward.py`, `test_realism.py` |
| `with_lsf` vs. fresh build at matched kernel radius | identical kernels; loglike rtol 1e-12 | same |
| `with_lsf` at narrower width vs. fresh (smaller-radius) build | truncation-tail level (~1e-5) | `test_with_lsf_narrower_width_agrees_to_truncation` |
| SB3 `orbit_velocities` vs. manually composed nested Keplerians | atol 1e-12 | `test_sb3_velocities_match_hand_composed` |
| ∂ log p / ∂(light, lsf_sigma) vs. finite differences | rtol 1e-4 | `test_gradients_light_lsf_match_finite_differences` |
| LSF-bound / outer-disk guards reject | non-finite log-density | `test_guards_reject_wide_lsf_and_outer_disk` |

### Closed-loop tests (one acceptance test per feature)

All are at the scale of the joint-NUTS acceptance test (n = 490 px, 10–14 epochs,
SNR 110–150, topocentric, gaps + cosmic-ray hits), with MAP/ML-II point recovery:

| Feature | Recovery | Wall time |
|---|---|---|
| **Tellurics** (3rd component, topocentric) | K₁ 0.07%, K₂ 0.41%; telluric spectrum RMS 0.047, corr 0.989 (135 core px) | 57 s |
| **SB3** (hierarchical, 14 epochs over 2.2 P_out) | K₁ 0.10%, K₂ −0.08%, K_AB 0.39%, K_C 0.50%; e_in ±0.004, e_out ±0.008, P_out rel 9e-5; observable combination RMS 0.007 | 76 s |
| **Per-epoch light** (12 epochs, 3 in partial eclipse, ℓ inferred with flat Dirichlet) | ℓ₁ per-epoch rms 0.0028 (eclipse epochs ±0.004); K's −0.77% / −0.14%; **each component individually recovered** (core RMS 0.010 / 0.012) | 48 s |
| **LSF widths** (2 instruments, reference pinned) | σ_B −0.67% (asserted <3%); K's <0.2% | 36 s |
| **K₂ scan** (ℓ₂ = 0.1 companion, 15-point grid) | peak exactly at injected K₂ = 38; contrast > 4000 in D over the scan edges; companion line pattern corr 0.977, offset-removed RMS 0.05 | ~3 s/scan |

In the per-epoch-light row, the k = 0 additive indeterminacy that limited constant-light
component recovery to the ~0.1 level is removed by eclipse epochs whose light fractions
are inferred rather than supplied. The component spectra are recovered individually at
the 0.01 level, with ℓ(t) to 0.003.

### Negative results

1. **Absolute LSF widths are unidentifiable in a template-free model.** ML-II with
   both instruments free inflates σ by +35% / +13% (a trade-off with intrinsic line
   widths; K's unaffected at <0.2%). With one reference instrument fixed, the other
   width is recovered to <1%. The policy is to fix one instrument, the same
   declared-anchor convention as the light ratio.
2. **The K₂-scan null is negative.** On a companion-free dataset, D(K₂) ∈ [−544, −465]
   over the whole grid, because the marginal likelihood's Occam term penalizes the extra
   marginalized component and no coherent signal offsets the penalty. Detection
   thresholds remain empirically calibrated (math.md §6), but the baseline is negative,
   not zero.
3. **The faint companion's envelope is prior-dominated.** At ℓ₂ = 0.1 the §5.1
   low-frequency degeneracy is amplified by ℓ₁/ℓ₂ = 9: the recovered companion has a
   ~+0.19 constant offset (its mean blanketing is absorbed by the primary) while the
   line pattern is intact. This is recorded in math.md §6. The scan delivers
   line-pattern quantities.
4. **A second exact k = 0 mode appears with tellurics** (telluric constant vs.
   common stellar constant, since Σℓ = 1). The measured offsets are +0.030 / −0.029,
   cancelling to 0.001 in the light-weighted sum. A row was added to the summary of
   degeneracies (math.md §5.4).
5. **Injected tellurics must be representable on the model grid.** Sub-pixel telluric
   lines behind a 7 km/s LSF are resolution-limited (RMS against the raw injected
   spectrum no better than ≈ 0.2 at any SNR or epoch count). This is a constraint on the
   simulator configuration, not a solver limitation.

## Scale, benchmarks, release readiness (2026-08-11)

Machine: the same desktop (32 GB RAM), CPU, float64. The scale criterion ("2×10⁵
px / 50 epochs samples in minutes on one GPU") was projected from these CPU
measurements. In the GPU run (below, 2026-08-14) the CUDA path works and scales as
predicted, but the criterion is not met on the consumer card available here. The card has
16 GB of device memory against a gradient that needs 18.24 GB, and runs fp64 at 1/50 of
fp32. The criterion remains unmet, with a specific hardware requirement.

### Three memory defects at survey scale

At survey scale (n = 31,750 px × 2 components, 50 epochs, half-bandwidth 513) the
marginal-likelihood evaluation initially failed on an 82 GB allocation. There were three
distinct causes, each now fixed and covered by the exactness suite (identical
log-likelihoods to 12 digits before/after):

1. **Closure-captured data arrays.** `jax.jit` of a method closing over the problem
   embedded every data array as an XLA constant, and the constant-folded θ-independent
   graph grew very large. `Problem`/`EpochGroup` are now registered pytrees, and the
   jitted marginal takes the problem as an argument (runtime parameter). At the failing
   size this fix alone left the planned temporaries unchanged.
2. **Unrolled probe batches.** Comb probing applied 2p+1 = 1027 matvecs in 16 unrolled
   `vmap` chunks with no data dependence between them. XLA scheduled them with
   overlapping live ranges, and its buffer analysis planned 79.6 GB of temporaries.
   Probing is now a sequential `lax.scan` over batches (combs generated from offsets
   inside the body), which forces buffer reuse: planned temporaries dropped to 2.1 GB
   (38×) and compile time fell ~5× (one scan body instead of 16 unrolled copies). The
   scatter-based block assembly (O(n·p) index maps, ~8 GB at design scale) was likewise
   replaced by per-block gathers under `lax.map` (O(B²) transients).
3. **Scan storage for the backward pass.** Reverse mode through the probe scan saved
   every batch's forward intermediates, so gradient memory returned to the unrolled
   total, with 469 GB requested at the design target. With `jax.checkpoint` on the batch
   body, the backward sweep recomputes each batch (~1.5-2× backward probing cost) and
   gradient memory stays at the outputs array plus one batch.

Unconditional rematerialization (remat) with serialized small batches cost up to 2× in
NUTS wall time at small scale (tutorial run 72 → 141 s, acceptance test 156 s vs ~102 s
at the joint-NUTS baseline, bit-identical posteriors). XLA's parallel execution of
independent unrolled probe batches, which overlapped 80 GB of live buffers at large scale,
is a multi-core speedup at small scale. Batch size and remat are therefore size-adaptive
on the probe-output footprint (64 MB threshold): small problems run all probes as one
parallel batch without remat (acceptance test back to 112 s), and large problems use the
sequential remat scan. The prior factor has its own small block size instead of the
posterior's: its bandwidth is 2 per component, and factorizing it at block 513 doubled
the Cholesky cost for a determinant of negligible cost.

### Design-target ladder (CPU, jitted, fixed bandwidth p = 513, 50 epochs, SB2)

| n (model px) | native px/epoch | eval | ∇ eval |
|---|---|---|---|
| 31,750 | 13,134 | 24.7 s | 102.7 s |
| 74,331 | 33,134 | 52.5 s | 249.9 s |
| 135,063 | 66,467 | 96.9 s | 466.0 s |
| **203,497 (design target)** | 106,467 | **149.7 s** | **729.9 s** |

Both times scale linearly in n at fixed bandwidth (~0.75 ms/px eval, ~3.4 ms/px
gradient), as the O(n·p²) flop count predicts. Log-likelihood values are bit-identical
across all three solver revisions. Peak memory stayed within the machine's 32 GB at every
size. A single design-target marginal evaluation (disentangled spectra at a given orbit)
took 2.5 min on one desktop CPU. Posterior sampling at this scale is deferred to the GPU
(below).

### GPU projection (stated as projection, not measurement)

At the design target a NUTS run needs ~2,600 gradient evaluations (150+250 transitions
× 6.5 mean leapfrogs, the pipeline numbers measured above). On this CPU that is 2600 ×
12.2 min ≈ three weeks, hence the GPU target. The dominant costs (batched probe matvecs;
800-step scanned Cholesky of 513² blocks) are dense, batched and fp64. On a single
A100-class device the same graph is expected to run the gradient in ~1-3 s (probe batches
become large GEMM-like work, the block Cholesky ~0.1 s of batched `potrf`/`trsm`). A
converged posterior would then take 1-2 hours for the widest-bandwidth massive-star
configuration, and tens of minutes at moderate bandwidths (p ~ 200: flops drop ~6×). The
tuning options are `probe_chunk` (raise on GPU) and `remat=False` (80 GB HBM holds the
stored backward pass). These projections are untested until a run on such hardware.

### First GPU run (2026-08-14)

Hardware: NVIDIA GeForce RTX 5070 Ti, 16 GB, driver 595.97, under WSL2 Ubuntu with
`jax 0.11.0` on the `cuda12` backend (`jax.default_backend()` → `gpu`,
`[CudaDevice(id=0)]`). Blackwell needs no special handling. The graph is the same
`scripts/m5_scale_bench.py` graph as in the CPU ladder above.

The CUDA path runs, and its time is linear in *n*, as the flop count predicts:

| n (model px) | native px/epoch | eval | ∇ eval |
|---|---|---|---|
| 4,877 | 1,702 | 0.232 s | 0.236 s |
| 9,526 | 3,457 | 0.447 s | 0.461 s |
| 18,221 | 6,965 | 0.915 s | 0.871 s |
| 31,734 | 13,062 | **out of memory** | — |

The scale criterion is not met on this card, for two independent hardware reasons.

**Memory.** The run failed at 31,734 model px (one sixth of the design target) on a
single 7.42 GiB request, with 13.8 GiB free and preallocation disabled. This is consistent
with the memory-pass table below: the gradient needs 18.24 GB at the design target, more
than any 16 GB card has. The GPU requests ~2.5× the CPU peak at the same size in one
contiguous buffer, so the CPU figures are a lower bound for GPU sizing, not an estimate.

**Arithmetic.** A GeForce card runs double precision at a fraction of its single-precision
rate, and albireo's solver requires float64. A 4096³ matmul gave:

| | GFLOP/s |
|---|---|
| float32 | 39,023 |
| float64 | **783** |

The fp64 rate is 50× lower. At 783 GFLOP/s it is the rate of a good desktop CPU, not of
an accelerator, so the eval times above are only about 2× faster than this desktop's CPU
rather than the projected order of magnitude. The projection assumed A100-class fp64,
which consumer hardware does not provide.

The hardware requirement is therefore ≥ 24–40 GB of device memory and a 1:2 fp64 ratio,
i.e. A100 or H100 class, not "one GPU". On that hardware neither limit applies: 80 GB
holds the 18.24 GB gradient with room to set `remat=False`, and ~10–20 TFLOP/s of vector
fp64 is 12–25× the card measured here. Nothing measured here contradicts or confirms the
1–2 hour projection above.

Independently of the throughput criterion, the run establishes portability: albireo's
graph compiles and runs correctly under CUDA with no code change, on a consumer card, on
Windows via WSL2.

### Systematic error from a manually set light ratio (`scripts/m5_light_ratio_demo.py`)

This is the LB-1/HR 6819-type failure mode, measured on a seeded simulation (the paper
asset for the planned main case, HR 6819):

1. Disentangling at a wrong, manually set ℓ rescales the recovered line depths by exactly
   ℓ_true/ℓ_assumed. The measured affine slopes are 1.44 / 0.97 / 0.59 against predictions
   of 1.50 / 1.00 / 0.60 (assumed ℓ₂ = 0.2 / 0.3 / 0.5, true value 0.3), with the separate
   additive k≈0 envelope offset isolated by the fit. Line depths enter log g /
   luminosity-class diagnostics, which is the mechanism behind the debate.
2. The marginal likelihood profiled over ℓ₁ with hyperparameters refit by ML-II at
   every trial (like for like) is flat to <0.5 log-units across ℓ₁ ∈ [0.50, 0.85]
   under constant light. The data contain no light-ratio information. With fixed
   hyperparameters the profile shows O(10–100) spurious curvature that arises entirely
   through the prior (a wrong ℓ forces rescaled spectra, which a fixed prior scale
   penalizes). This failure mode is documented. With three partial-eclipse epochs the
   same profile peaks at the true ℓ₁ = 0.70 with Δlog L = −145 at ±0.05.

### fd3 comparison harness (`scripts/fd3_bench.py`)

The fd3 v3.1 input format was reverse-engineered from the official example files and
the C source (documented in the script header). It comprises a ln-λ master matrix with a
`# ncols X nrows` header, a comment-free stdin token stream for the control file, ω in
degrees, and per-epoch σ only. There are no per-pixel weights and no masks, so the
benchmark runs gap-free and neither code is at a disadvantage. Component B's RV sign is
applied internally (identical to albireo's ω+π convention), and fd3's internal c is
299,800 km/s. The harness simulates an SB2 on the common log grid fd3 requires (no
resampling for either code), writes fd3 separation- and fit-mode inputs, runs the albireo
side, and compares component spectra (raw and mean-aligned, since both codes have a k≈0
freedom) and wall time. The fd3 side needs the binary (~1.9 MB source tarball, GSL,
builds on Linux/WSL). No license is stated on the fd3 page (v2 was GPL and v3's GPL
statement was removed), so the author should be contacted before any redistribution.

### Comparison with fd3 (2026-08-14)

The tarball's prebuilt binary is 32-bit i386 and does not run on a modern x86-64 host, so
fd3 was rebuilt from source against conda-forge GCC and GSL under WSL2 Ubuntu. It is not
vendored into this repository, because the distribution states no license.

The build was validated against the author's shipped `.mod` / `.res` / `.rvs` outputs for
four worked examples, which tests the rebuild across a different compiler, architecture
and GSL:

| example | fd3 wall time | max abs. difference from the shipped `.mod` |
|---|---|---|
| `art_single` | 0.14 s | **0** (exact) |
| `art_double` | 2.52 s | 1.3 × 10⁻⁶ (the files are written to ~6 dp) |
| `art_triple` | 3.58 s | 1.0 × 10⁻⁹ |
| **`V453_Cyg`** (1344 px, a real published system) | 62.6 s | **0** (exact) |

The comparison uses the harness's seeded SB2 (20 epochs, SNR 100, a common ln-λ grid so
that neither code resamples, no gaps and no masks so that neither is at a disadvantage,
and the orbit fixed at the injected values for both):

| | comp 1 RMS | comp 2 RMS | steady-state wall time |
|---|---|---|---|
| **albireo** | 0.0118 | 0.0165 | 0.182 s |
| **fd3** | 0.1767 | 0.2597 | **0.111 s** |
| albireo, mean-aligned | **0.0093** | **0.0116** | |
| fd3, mean-aligned | 0.0198 | 0.0223 | |

fd3 was faster: 1.64× in steady state and 5.7× from a cold start (0.630 s including JAX
compilation). It is a small C program that starts, solves and exits, and a 1200-pixel
two-component separation is the regime in which a compiled direct method is expected to
be faster. albireo is in the same class, not an order of magnitude slower. (The harness's
earlier 3.93 s figure was its un-jitted single-solve path and overstated the difference
by 20×.) Timings are the minimum of five repeats, both codes on CPU.

fd3's raw error is ~15× larger, and about nine tenths of it is a constant: mean-aligning
reduces comp 1 from 0.1767 to 0.0198. The constant is the *k* = 0 freedom that both codes
have and neither can determine from constant-light data. It is the null space of
[§5.1](math.md#51-the-low-frequency-degeneracy-the-undulations-theorem), and the reason
the literature's workflow includes a manual renormalization against an external light
ratio. albireo's smoothness prior determines the offset. fd3 leaves it to the user.

With that offset removed, albireo is about 2× more accurate in shape (0.0093 / 0.0116
against 0.0198 / 0.0223), because the prior constrains the low-*k* modes.

fd3 returns a point estimate with no uncertainty on the component spectra. [The handoff
tutorial](tutorials/downstream.md) converts albireo's uncertainty into an error bar on
log *g*.

### Clean-room shift-and-add implementation (2026-08-15)

The third code is the most widely used in the field. `scripts/shift_and_add.py` is a
clean-room implementation written from González & Levato (2006) §2.1 Eqs. (1)–(2) and §2.3,
whose recurrence is restated identically and independently by Quintero et al. (2020). No
source code was consulted. The widely used existing implementation, the one behind the
LB-1 and HR 6819 companion identifications, has no license file and was never opened.

Two details of the published method are easily implemented incorrectly. The iteration is
Gauss–Seidel: the *B* update uses the *A* produced in the same sweep. The Jacobi variant
is a different algorithm with a different convergence rate. The initialization is B = 0
with A not seeded, so the first primary estimate is the plain rest-frame co-add.

The implementation is validated against the paper's theory. §2.3 derives that the residual
is not removed but diffused: each sweep convolves it with
`f(x) = n⁻² Σ δ(x − dᵢ + d_k)`, so after *m* sweeps it has been broadened by a Gaussian of
`σ = √(2m)·σ_d`. Seeding a delta-function error and measuring the returned width
reproduces that relation (`tests/test_shift_and_add.py`), evidence that the coded
recurrence is the paper's.

The paper gives the number of sweeps as "rarely more than 5–7". This is the case on the
benchmark's data, and the comparison does not depend on the choice:

| sweeps | comp 1 aligned | comp 2 aligned | wall time |
|---|---|---|---|
| 1 | 0.0431 | 0.0458 | 0.003 s |
| **7** (published figure) | **0.0248** | **0.0302** | **0.018 s** |
| 50 | 0.0221 | 0.0239 | 0.128 s |

Seven sweeps is close to converged. Fifty sweeps improve the result by 11% for seven times
the cost and are still worse than albireo by a factor of 2.4, so stopping early does not
put the method at a disadvantage.

### Three-way comparison on identical data

The data are the same seeded SB2: 20 epochs, SNR 100, a common ln-λ grid so that no code
resamples, no gaps or masks, and the orbit fixed at its injected values for every code.
Shift-and-add uses albireo's linear shift operator, so the comparison is between
algorithms rather than between two interpolators.

| | comp 1 raw | comp 1 aligned | comp 2 raw | comp 2 aligned | wall time | uncertainty |
|---|---|---|---|---|---|---|
| **albireo** | **0.0118** | **0.0093** | **0.0165** | **0.0116** | 0.182 s | **yes, a posterior** |
| fd3 | 0.1767 | 0.0198 | 0.2597 | 0.0223 | 0.111 s | no |
| shift-and-add | 0.0317 | 0.0248 | 0.0849 | 0.0302 | **0.018 s** | no |

albireo was the slowest of the three. Shift-and-add was 10× faster than albireo's steady
state and 6× faster than fd3. It is a small number of array shifts and means, and this is
the expected ordering for a 1200-pixel two-component separation.

These timings depend on the measurement convention and do not characterize the codes. The
re-run section at the end of this file ran all three on the same desktop under one
protocol. Every accuracy value reproduced exactly. The shift-and-add time had been
contaminated by the harness's in-process timing (in both recorded measurements) and the
fd3 time inflated by its BLAS running 32 threads. The uncontaminated ranking is
shift-and-add 0.026 s, albireo 0.059 s, fd3 0.064 s with thread pinning.

albireo is about 2× more accurate, and the margin is not an artifact of the stopping
point (see the sweep table). The aligned ordering is albireo 0.0093 / 0.0116, fd3
0.0198 / 0.0223, shift-and-add 0.0248 / 0.0302.

The two established codes have different raw errors, both from the same degeneracy.
fd3's raw RMS is 15× albireo's, and nine tenths of it is a constant. Shift-and-add's raw
error is much smaller than fd3's but larger on the fainter component (0.0849 raw against
0.0302 aligned), because `B = 0` leaves the secondary's continuum level set by the
initialization rather than by the data. Both lie in the *k* = 0 null space: the per-mode
convergence factor of the shift-and-add recursion has modulus exactly 1 at zero frequency,
so that mode is a fixed point that no number of sweeps changes. All three independent
methods are subject to the degeneracy of
[§5.1](math.md#51-the-low-frequency-degeneracy-the-undulations-theorem). albireo's
smoothness prior determines that mode. The other two codes leave it to the user, which is
the purpose of the literature's manual renormalization against an external light ratio.

No adjustment to the comparison changes the last column, the presence of an uncertainty.

A comparison on an SB2 with a nebular line, between a method that masks the contaminated
pixels and methods that cannot, would be unfair to shift-and-add. González & Levato
explicitly permit "any combination algorithm... weights or some rejection algorithm", so
masking is part of the published method. `tests/test_shift_and_add.py` exercises it:
setting an epoch's weight to zero removes an unusable epoch. The method cannot produce an
uncertainty.

### AI Phoenicis: observed spectra and a published orbit as the reference (2026-08-15)

AI Phe has 36 archival HARPS spectra (ESO, *R* = 115,000, 3782–6913 Å, SNR 41–129, all ten
phase bins filled, fetched with `albireo.archive`). No true spectrum is known, but the
orbit is published to a precision no disentangling code approaches (Maxted et al. 2020,
MNRAS 498, 332):

> K₁ = 51.164 ± 0.007 km/s,  K₂ = 49.106 ± 0.010 km/s,  P = 24.5924 d,
> e = 0.1878 ± 0.0006,  ω = 110.30 ± 0.06°,  T₀ = BJD_TDB 2458362.82847

The semi-amplitude uncertainties are 0.014% and 0.020%, and several independent studies
agree to 0.1%, so the orbit is the ground truth. `scripts/aiphe_bench.py` runs the
benchmark.

The eccentricity is recovered from the spectra alone, with the optimizer started 15% from
the published eccentricity vector and 8% from both semi-amplitudes:

| | albireo | published | |
|---|---|---|---|
| e | **0.1879** | 0.1878 ± 0.0006 | +0.0001 |

A TESS light curve and 36 HARPS spectra agree on the eccentricity to 0.05% by independent
methods.

The semi-amplitudes have a reproducible ~1% systematic error. Two disjoint windows, chosen
to share no lines, give:

| | 5150–5250 Å | 5340–5440 Å | published |
|---|---|---|---|
| K₁ | 50.452 (−1.39%) | 50.440 (−1.42%) | 51.164 ± 0.007 |
| K₂ | 49.495 (+0.79%) | 49.479 (+0.76%) | 49.106 ± 0.010 |
| *q* = K₁/K₂ | 1.0193 | 1.0194 | 1.0419 |

Both runs used the period rounded to 24.5924 d and a second window starting at 5340 Å.
`aiphe_bench.py` now uses the unrounded photometric period, 24.592483 d (Kirkby-Kent
et al. 2016), and starts the cross-check window at 5341 Å so that its 3 Å pad does not
reach HARPS's inter-CCD gap, 5304.67–5337.61 Å. At 5340 Å the pad extended 0.6 Å into the
gap, over about 33 pixels per epoch, which `mask_flux_gaps` removed. Neither change
affects this table: `period` and `t_conj` are free in the K fit, and the removed pixels
have no flux. Both changes affect the runs below that hold the velocities fixed.

The two windows agree with each other to 0.02% in K₁ and 0.01% in the mass ratio, and both
differ from the published values by the same amount. Three explanations are excluded:

* **The optimizer.** The fit converged, with `|grad| = 9e-03`, and the result is the same
  for a 250-step run that reached its step limit and a 444-step run that did not.
* **Line selection or the window.** Two disjoint windows agree to 0.02%.
* **The light ratio.** Sweeping the assumed ℓ₂ from 0.38 to 0.53 changes the likelihood
  difference between albireo's K and the published K by 9 nats out of 53,306.

This leaves a ~1% systematic error in the disentangling of this system, biasing the mass
ratio 2.2% toward unity, the direction expected when the line signals of two similar
stars are partly confused. It is an unresolved discrepancy, not a measurement and not a
correction to the literature: a value good to 0.02% from several independent
cross-correlation studies of full échelle spectra is the better one.

The formal statistics do not decide between the two values. albireo's optimum is 53,306
nats above the published K, which is not decisive: with 358,265 high-SNR pixels, a
one-pixel systematic velocity offset changes the log-likelihood by that much. When formal
precision is far finer than the systematic error, likelihood ratios are not informative.
The residual z-RMS is 2.64 rather than 1. HARPS provides no error array, so the weights
are albireo's own scatter estimate, and any formal error bar from this fit is ~2.6× too
small until the weights are rescaled.

The spectra from albireo and the clean-room shift-and-add agree on these observed data,
with mean-aligned RMS differences of 0.029 and 0.037 in the line cores of the 5340–5440 Å
window, against line depths near 0.8. With no true spectrum, this is the available
consistency check. Shift-and-add again took 0.07 s against albireo's 11 s.

### Defect found on observed data

A second window at 5300–5400 Å spans the gap between HARPS's two CCDs (5304.67–5337.61 Å,
32.9 Å of exact zeros), and the gap pixels were weighted as data. They are finite, there
is no quality column, and, because HARPS provides no error array, the inverse variance is
estimated from the local scatter, which is small across a flat run of zeros. Median ivar
was 6398 across the gap against 6231 for real pixels, with `mask` empty. A window that
was 33% detector gap gave disentangled component spectra with negative flux.

`albireo.mask_flux_gaps` zero-weights contiguous runs of non-positive flux and warns with
the wavelength range. `to_epoch` calls it before the spike clip, since a flat run has no
local scatter for a running median to detect. The rule applies to runs:
`RawSpectrum.bad_pixels` does not treat a single zero flux value as missing, because one
zero can be a saturated core or a clipped cosmic ray, whereas eight in a row cannot. It
parallels the reader's rule that a zero error is not infinite precision: the reader may
leave an ambiguous value unclassified, but does not infer its meaning.

### AI Phe: the eclipse and the spectroscopic light ratio

AI Phe eclipses, but the eclipse does not give the spectroscopic light ratio directly. It
determines the fractional radii, the inclination and the surface-brightness ratio in the
photometric band (TESS, centred near 7860 Å). The light ratio in an optical spectroscopic
window is a different quantity, computed from the radii and the two temperatures. It
depends strongly on wavelength: for AI Phe's 6310 K + 5010 K pair at R₂/R₁ = 1.624, a
blackbody estimate gives ℓ₂ = 0.375 at 4000 Å and 0.510 at 6500 Å. It is far better
constrained than for a non-eclipsing system, but using the TESS-band value at 5200 Å would
be a ~10% error in the quantity by which every recovered line depth scales.

### Tutorials, examples, CI

Two executable tutorials (`examples/01_sb2_end_to_end.py`, `02_k2_scan.py`) run the
pipeline with asserts and underlie the narrative docs pages. A dedicated CI smoke job
runs both with `ALBIREO_EXAMPLE_FAST=1` (~3.6 min + 7 s measured). At tutorial scale the
SB2 example's NUTS posterior gives P to 0.013%, K₁ to 0.015% and K₂ to 0.21%, with zero
divergences.

### Release readiness (JOSS) and the real-data decision

`paper/paper.md` and `paper.bib` were drafted (claims cross-checked against this file;
"GPU-accelerated" kept out of the title until the GPU criterion is met) and
`CONTRIBUTING.md` was added. The remaining JOSS blockers are maintainer-only (affiliation,
ORCID, archive DOI at acceptance, PyPI registration). For the end-to-end run on an
observed, published SB2, the research recommendation is HR 6819 (51 public FEROS spectra,
one ESO program, ~153 MB, no login). Its manually set light ratio is central to the 2020
black-hole debate, and GRAVITY provides an interferometric ground truth,
f = 0.439 ± 0.013, to score the posterior against. AI Phoenicis is the precision backup
(60 public epochs; K's known to 0.02%). The data download is pending maintainer approval.

## Speedup pass: direct band assembly and a closed-form gradient (2026-08-11)

The machine, ladder configuration and seeds are unchanged. Log-likelihoods agree with the
scale section to machine precision (several rows bit-identical, the rest to ~1e-15
relative, since the assembly changes only the floating-point summation order).

### Cost breakdown

A stage-split profile at the 31.7k ladder row attributed 22.3 s of the 24.3 s evaluation
(92%) to comb probing. The block Cholesky took 0.86 s, and the remainder was noise.
Probing uses 2p+1 = 1027 matrix-free operator applications, covering the union of all
epochs' band offsets, although each epoch contributes only a ~50-pixel-wide band at a
velocity-determined offset. That redundancy, not the factorization, dominated the cost at
scale.

### Implementation

1. **Direct per-epoch band assembly** (`albireo/assembly.py`, math.md §4.5). Each
   epoch's (i,j) block is ℓᵢℓⱼ·T(δᵢ)ᵀ·G·T(δⱼ) with G = KᵀRᵀW′RK a narrow band. It is
   assembled by static rebin pair tables (one `segment_sum` per epoch), two
   unrolled kernel-shift passes, and a four-term tent-weighted combination of
   row-translated copies, and accumulated into a global band tensor by
   `dynamic_update_slice` (no scatters on the hot path). The work per epoch is
   O(band width) instead of O(bandwidth) matvecs, ~12× faster on the assembly stage.
   Probing remains as `assembly="probe"` (reference) and as the `validate=True` oracle,
   which now checks the band assembly against the matrix-free operator directly.
2. **Closed-form solve-stage gradient** (custom VJP). Cotangents of
   {log det, quadratic form, d̂} with respect to the precision are computed from the
   block-Takahashi banded selected inverse and d̂-outer products, so reverse mode does
   not traverse the Cholesky/solve scans. Assembly-side reverse work was reduced further
   by hoisting the velocity-independent G stage out of the rematerialized scan and by
   expressing row translation as clip-safe `dynamic_slice` (its transpose is a contiguous
   copy, where a gather's transpose is a scatter). The gradient is verified against plain
   autodiff to 1e-13 relative and by finite differences.

### Ladder, before → after (CPU, same desktop, jitted, p = 513, 50 epochs, SB2)

| n (model px) | eval before | eval after | ∇ before | ∇ after |
|---|---|---|---|---|
| 31,734 | 24.7 s | **3.0 s** (8.2×) | 102.7 s | **10.6 s** (9.7×) |
| 74,322 | 52.5 s | **8.0 s** (6.6×) | 249.9 s | **24.9 s** (10.0×) |
| 135,052 | 96.9 s | **14.5 s** (6.7×) | 466.0 s | **56.5 s** (8.2×) |
| **203,440 (design target)** | 149.7 s | **26.0 s** (5.8×) | 729.9 s | **111.3 s** (6.6×) |

The table is a single sequential run of `scripts/m5_scale_bench.py` with no external
load. A design-target marginal evaluation (disentangled spectra at a fixed orbit) took
~26 s on one desktop CPU (previously 2.5 min), and a gradient under 2 min (previously
12 min). The NUTS acceptance test took ~65 s of wall time (Laplace + warmup + 250
samples) against ~102 s at the joint-NUTS baseline and 112 s in the scale section. The
small-problem slowdown that the size-adaptive probing policy prevented therefore no
longer occurs, and the policy is no longer needed. The speedup ratios are lower at the
design-target row because the gradient's working set exceeded this machine's 32 GB
(measured 34.0 GB; the memory pass below brought it to 18.2 GB and that row to
22.2 s / 87.4 s). At realistic single-star bandwidths (HR 6819-like: p ≈ 160 rather than
the ladder's conservative 513) the same operations take seconds per gradient. Full NUTS
posteriors for real SB2 problems are then feasible on one desktop CPU, and the GPU
becomes spare capacity rather than a requirement.

### Defective Hessian in the Laplace mass matrix

Two second-order errors were found. First, the initial custom rule was exact to first
order but wrong to second order (8e-3 relative): its forward rule called the custom
function itself, so Hessians re-entered the custom boundary and lost the Cholesky-mediated
terms through the dropped cotangent. Inlining the primal in the forward rule corrects
this, and `jacrev(jacrev(...))` then agrees with plain autodiff to 1e-15. Second,
independently of the scale work, `jax.hessian` (forward-over-reverse) produces an
asymmetric Hessian on this stack even for the plain-autodiff path (off-diagonal 0.566 vs
0.855 on the diagnostic problem), while reverse-over-reverse matches central finite
differences of the gradient to 8 digits at three step sizes. `laplace_inverse_mass`, the
only forward-mode consumer in the package, had been symmetrizing a slightly wrong matrix
since joint inference was added. It now uses reverse-over-reverse (math.md §4.5). The
mass matrix is a preconditioner, so posteriors were not biased. Warmup was tuned from a
mildly wrong curvature estimate.

### GPU consequences

The hot path now consists of vmapped dense convolution-like passes, contiguous dynamic
slices, and one 50-step scan with large per-step work, all GPU-native. The
`probe_chunk`/`remat` tuning was removed with probing. The remaining GPU-specific
bottleneck is chain latency in the sequential block Cholesky/Takahashi scans (~800 steps
of 513² work at the design target). The remaining options are the associative-scan
(parallel-prefix) factorization that removes it, a custom-VJP band→block packing, a fully
analytic assembly VJP, and opt-in mixed precision. As a projection, not a measurement, the
design-target gradient takes ~0.5–1.5 s on one A100-class device and a converged
design-target posterior tens of minutes. The earlier 1–2 h projection remains the
conservative bound.

---

## Memory pass and a grid-boundary defect (2026-08-12)

After the speedup pass the design target was fast, but its gradient did not fit in
memory: XLA `memory_analysis()` of the compiled executables (buffer assignment without
allocation) gave 34.0 GB against 32 GB of RAM.

### Memory breakdown (design target: 203,440 model px, SB2, 50 epochs, p = 513)

| stage | GB | outcome |
|---|---|---|
| `G` pre-pass, `vmap`ped over all 50 epochs | ~9 | `vmap` batches every intermediate of the chain (H, the two kernel stages), not only the 4.5 GB result |
| band tensor + its cotangent | 6.2 | unchanged (next candidate) |
| `bt` + Cholesky factor | 6.2 | unchanged (both live) |
| selected inverse (`s_diag`, `s_sub`) | 3.1 | **removed**: fused into the cotangent |
| `_pack_band` gather indices/masks, stacked over K blocks | ~5 | **removed**: hoisted + rematerialized |
| prior block-tridiagonal factor | 0.8 | **removed**: scalar recursion |

### Four exact reductions

1. **Fuse the Takahashi sweep into the cotangent** (`solver.selected_inverse_cotangent`).
   Each Σ block is contracted against `d`, `u` at the step that produces it, so
   the `2K − 1` selected-inverse blocks and the outer-product temporaries are never
   materialized. `selected_inverse_blocks` remains as the test oracle.
2. **Batch the `G` pre-pass over epochs** (`epoch_chunk`). Because `G` is
   velocity-independent it is computed once per epoch either way, so any
   batching costs exactly one extra `G` pass in the rematerialized backward, and
   the size matters only for `vmap` width. By default the whole pre-pass is hoisted
   below ~1 GB (small and acceptance-test-scale problems are unaffected) and otherwise
   batched to ~0.5 GB.
3. **Prior determinant by scalar pentadiagonal recursion**
   (`assembly.prior_logdet`) instead of factorizing a bandwidth-2 matrix as
   6,358 dense 64×64 blocks.
4. **Hoist `_pack_band`'s gather indices** out of the `lax.map` body (the band
   coordinates depend only on the within-block (row, col), never on the block
   counter) and rematerialize the body, so that reverse mode no longer stacks (B, B)
   index and mask arrays over all K iterations.

| n (model px) | eval before → after | ∇ before → after |
|---|---|---|
| 31,734 | 2.93 → 2.94 GB | 5.02 → **4.00 GB** |
| 74,322 | 7.04 → **4.86 GB** | 11.92 → **11.47 GB** |
| 135,052 | 12.0 → **7.83 GB** | 16.64 → **14.37 GB** |
| **203,440 (design target)** | 20.64 → **11.06 GB** | 34.00 → **18.24 GB** |

Log-likelihoods, gradients and Hessians are unchanged. The equivalences are
regression-tested against the implementations they replaced (blocked prior determinant,
unfused selected inverse, unbatched pre-pass).

On the same desktop with the same seeds, the pass also reduced the wall time at the
design target:

| n (model px) | eval, before → after | ∇, before → after |
|---|---|---|
| 31,734 | 3.0 → **2.82 s** | 10.6 → 11.22 s |
| 74,322 | 8.0 → **6.95 s** | 24.9 → 30.67 s |
| 135,052 | 14.5 → 14.90 s | 56.5 → 59.53 s |
| **203,440 (design target)** | 26.0 → **22.16 s** | 111.3 → **87.43 s** |

The middle rows' gradients are slower by 5–23%, because of the batched pre-pass's extra
backward `G` pass, incurred where memory was not the binding constraint. At the design
target, which previously did not fit, halving the working set reduced the time because
the gradient had been thrashing against RAM. Raising `epoch_chunk` to the epoch count on
a machine with spare memory (or on GPU) recovers the lost time.

### Grid-boundary defect in the band assembly

Re-deriving the band layout exposed a correctness defect in the new assembly.
`G = Kᵀ(RᵀW′R)K` is built as a band image. `H` is exactly zero outside the model grid,
but the LSF convolution spreads in-grid entries outward, writing band entries at column
indices that correspond to no grid pixel. The T-sandwich reads those entries whenever an
epoch's shift places a component's support against a grid edge, where `T(δ)` has no row
and the contribution should be zero.

The defect occurs when the data-coverage margin, in model pixels, is smaller than the LSF
kernel radius. Band assembly was compared with probing, dense and entrywise (probing is
the symmetric reference, and only the column side was affected, so asymmetry identifies
the defect):

| coverage margin (model px) | kernel radius | max relative entry error | Δ log L |
|---|---|---|---|
| ~24 | 6 | 6.4e-15 | 2.7e-15 |
| ~8 | 6 | 6.4e-15 | 0 |
| ~4 | 6 | 6.4e-04 | 1.8e-07 |
| **0** | **6** | **6.8e-02** (asymmetry 6.8e-02) | **−57 nats** |
| 19 | 34 | 6.9e-04 | 0.16 nats |

In the last row, at a fixed grid, fitting a wider LSF is enough to trigger the defect. A
margin of zero is a usual configuration: it follows from choosing a model grid narrower
than the observed range to fit a sub-region, the documented usage. Every earlier fixture
had a margin larger than the kernel radius, so the weights were zero where the defect
occurs and 174 tests passed despite it. The largest row of `scripts/m5_scale_bench.py`
was about one pixel from the boundary.

The fix masks the out-of-grid columns of `G` at the source (one static boolean per
group), and the margin-0 case then agrees to 5.7e-15 with asymmetry 2.9e-16. Two fixtures
cover it, `edge_covered` in the equivalence set and a dedicated
model-grid-inside-the-data case. The equivalence check gained an entrywise dense
comparison and an asymmetry assertion, because the log-determinant and solve it
previously relied on average out a boundary defect, which at the 0.3 Å margin put it
below a 1e-11 scalar threshold.

### Two further silent defects

- **Gradients through the Cholesky factor were identically zero.** `_solve_stage`
  returned the factor, but its closed-form reverse rule cannot propagate a cotangent
  on it (doing so is the reverse pass through the factorization that the rule exists to
  avoid). Anything differentiating `spectra_std` or `draw_spectra` therefore received
  zero with no error. `MarginalResult` now stores the precision and rebuilds the factor
  outside the custom boundary, where plain autodiff applies. The cost is one extra block
  Cholesky, only for callers that request the factor, which the sampling hot path never
  does.
- **A few wide native pixels set the solver bandwidth without warning.** `row_support`
  is a max over native rows, and it determines a cost quadratic in the block size. In
  observed spectra a wide row usually means samples were deleted (telluric window, order
  or chip gap): pixel edges are at midpoints, so each of the two bracketing pixels
  absorbs half the gap. `build_problem` now warns, names the pixel, and states the remedy
  (mask with `ivar = 0`, or split into separate instrument labels).

### Remaining memory use

The band tensor and its cotangent (6.2 GB) still store both triangles of a
symmetric matrix, and `bt` and its factor (6.2 GB) are both live across the
Cholesky. The next options are folding the band to one triangle and assembling directly
into block storage, so that the band tensor is never stored. Neither is needed to
run the design target, which has ~13 GB of spare memory on this machine.

## HR 6819: the first observed dataset (2026-08-12)

The first observed dataset is the 51 public FEROS exposures of HR 6819
(ESO 073.D-0274(A), PI Rivinius, 153 MB, anonymous), the data used by every published
analysis of the system.

### Data properties: expected and delivered

| | expected | delivered |
|---|---|---|
| continuum | flux ~ 1 | `CONTNORM = False`; raw merged-echelle ADU, response falling **20×** over 3850–4750 Å, negative below ~3830 Å |
| uncertainties | `ivar > 0` | `ERR` column **all NaN** ("Error spectrum not available") |
| wavelength grid | one per instrument | 0.03 Å step shared, but start wavelengths spread over 0.78 Å and lengths 189621–189653, giving **28 distinct grids** |
| frame | declared | `SPECSYS = BARYCENT`, correction in `ESO DRS BARYCORR` (−25.13 to +16.12 km/s) |
| time | BJD_TDB mid-exposure | `TMID` = MJD(UTC), in the extension header, not the primary |

### Five defects found on the observed data

1. **A NaN at a zero-weight pixel made the whole likelihood `nan`.** `data.py`
   documents that a masked pixel's flux "is never read", but `build_problem` formed
   `z = flux − r·base` unmasked and every consumer multiplies by `w`, so
   `0 · nan = nan`. `normalize` writes such pixels wherever the fitted continuum
   approaches zero. The log-likelihood is identical before and after the fix when
   the masked block holds `nan`, `±inf`, or `1e300`.
2. **Per-exposure grids were rejected.** Grouping is now by (instrument, grid). The
   LSF is still keyed by instrument, so `lsf_sigma_v` has one entry per instrument,
   not 28.
3. **`prior_logdet` returned `nan` below `eta/tau ≈ 1e-13`**, which unbounded ML-II
   can reach, from an unguarded `sqrt` of a difference of like-sized terms.
4. **The row-support warning reported the `int64` minimum as its median** whenever
   more than half an epoch lay outside the model grid. Rows untouched by the rebin
   operator held a sentinel and were not excluded.
5. **A region disjoint from the model grid failed with "empty rebin operator"**,
   with no indication of which of the two ranges was wrong.

### Continuum fit in log flux

A curvature penalty applied to the flux cannot follow a multiplicative response.
On one exposure over 3850–4750 Å, normalizing with a 150 Å linear-space
smoother left the 97th percentile of the normalized flux at 0.55–0.63 in the
worst 50 Å bins, a 40% error. The fit is therefore made to `log(flux)`, where a
pure exponential is a straight line and straight lines are in the penalty's
nullspace:

| smoothing scale | p97 of normalized flux, per 50 Å bin, 3850–4750 Å |
|---|---|
| 80 Å | 1.006 – 1.010 |
| 150 Å | 1.007 – 1.011 |

The result is flat across the whole 20× gradient and insensitive to the
smoothing scale.

The second part of the fix is the knot basis. A per-pixel Whittaker smoother needs
`λ ≈ (L_px/2π)^4`. At the 5000–10000 pixel smoothing lengths that a merged echelle
spectrum requires, this is `λ ~ 10¹²`, the weight term is lost to rounding, and
`solveh_banded` fails with "leading minor not positive definite". Eight knots per
smoothing length keep `λ ≈ 2.6` at any requested scale.

### Barycentric sign convention: telluric check

No test compared albireo's `v_bary` convention with an external standard. The telluric
O₂ A band (7595–7660 Å) was cross-correlated across epochs spanning 55 km/s of correction:

| | rms residual |
|---|---|
| telluric shift = **+BARYCORR** | **0.138 km/s** |
| telluric shift = −BARYCORR | 59.3 km/s |

The slope is 0.9993, so `frame="barycentric"` with `v_bary = ESO DRS BARYCORR` is
correct. The 0.14 km/s floor is CCF precision plus water-vapour variability.
Independently, astropy's correction agrees with the pipeline's keyword to 0.017 km/s.

### Configuration for the science run

The window is 4380–4600 Å (He I 4388/4471, Mg II 4481, Si III 4552/4568/4575), with lines
photospheric in both components, no Balmer core, no variable disc emission, and the
nearest telluric band 1200 Å away. After `share_wavelength_grid` (residual relabelling
0.007 km/s, 1/300 of a pixel) the 28 groups reduce to 1.

| | |
|---|---|
| epochs / native pixels | 51 / 373,983 (100.0% good) |
| model grid | 9,938 px, 4378.45–4601.65 Å, dv = 1.50 km/s |
| operator groups | 1 |
| row support / kernel radius | 3 / 8 |
| half-bandwidth `b_nat`, `p` | 81, 163 |
| marginal log-likelihood eval | 0.5 s (after a 1.0 s compile) |

### Fit results and systematic errors

The fit was MAP + ML-II from a conjunction-phase scan: 120 L-BFGS steps, 2820 s, peak RSS
2.6 GB. The gradient norm plateaued near 4×10² and oscillated, because `run_map`'s
absolute `tol` is unreachable at 3.7×10⁵ good pixels. A `callback` allows convergence to
be judged from the parameters instead of the flag.

The marginal likelihood was profiled around the MAP in two independent windows:

| | A: 4380–4600 Å | B: 4120–4330 Å | Klement et al. 2025 |
|---|---|---|---|
| lines | He I 4388/4471, Mg II 4481, Si III 4552/68/75 | He I 4144/4169, Si II, Fe II | — |
| period [d] | 40.36583 ± 0.00045 | 40.37022 ± 0.00065 | 40.3261 ± 0.0013 |
| K<sub>pre-sd</sub> [km/s] | 63.314 ± 0.013 | 63.724 ± 0.019 | 61.15 ± 0.88 |
| K<sub>Be</sub> [km/s] | 1.985 ± 0.151 | 3.022 ± 0.153 | 3.90 ± 0.27 |
| eccentricity | 0.0302 | — | 0.0289 ± 0.0058 |
| whitened residual sd | 1.674 | 1.403 | 1 if calibrated |

The result is useful, but the orbit is not publishable:

* The eccentricity agrees to 0.2σ: 0.0302 against 0.0289 ± 0.0058, for a nearly circular
  orbit, from spectra alone.
* The quoted ± are statistical only and are too small by at least an order of magnitude.
  The two windows differ by 0.0044 d in period (5.6σ of their combined internal errors) and
  0.41 km/s in K<sub>pre-sd</sub> (17.8σ). The window-to-window scatter, not the
  likelihood curvature, is the lower bound on the error bar.
* The residual scatter, 1.4–1.7× the assumed noise, indicates unmodelled structure. In
  rough order of plausibility, the candidate causes are the pipeline's resampling of
  these spectra onto a common step (the diagonal `ivar` model then understates the
  uncertainties by construction), per-epoch continuum residuals, the Gaussian
  approximation to FEROS's LSF, and the Be star's disc emission, which violates the
  one-static-spectrum-per-component assumption over a 135-day baseline.
* K<sub>pre-sd</sub> is 3.5–4% above the literature value in both windows. The sign is
  as expected physically: the published values come from cross-correlation and Gaussian
  line fits, which blend the sharp lines with the Be star's broad, nearly stationary
  ones and are therefore biased toward the systemic velocity. Deblending should raise K.
  This run does not test that hypothesis.
* K<sub>Be</sub> is detected but not measured. The profile has a single well-defined
  peak, 83 nats above K = 0 in window A, but the two windows give 1.99 and 3.02 against a
  literature 3.90. At v sin i ≈ 200 km/s the Be star's reflex motion is ~1/50 of a line
  width, the unattributable regime of `math.md` §5.1, and its recovered spectrum is
  prior-dominated.

NUTS would return the same underestimated width around the same systematics-limited
point. The run first needs a noise model with a fitted inflation factor, a wider window,
and a check of whether the period offset persists under a per-epoch continuum treatment.

---

## The jitter site and its effect on the HR 6819 orbit (2026-08-12)

The run above used its estimated `ivar` as given and reported a residual scatter of
1.4–1.7 as a caveat. `forward.with_jitter` and the `log_jitter` θ site add a per-epoch
noise-inflation factor.

### Shared jitter: profiled α and the effective number of parameters

A single shared α was profiled at the previous MAP and compared with the standard
deviation of the whitened residuals (the naive estimator):

| | window A | window B |
|---|---|---|
| weighted pixels `N` | 373,813 | 356,928 |
| residual sd (the naive estimator) | 1.6743 | 1.4030 |
| profiled `α̂` | **1.6807** | **1.4088** |
| implied `p_eff = N[1 − (sd/α̂)²]` | **2,843** | **2,930** |
| model pixels `n_comp · n_pix` | 19,876 | 20,160 |
| resolution elements (FWHM = 4.16 px) | 4,779 | 4,846 |

The correction has the predicted sign (`math.md` §3.2a: `α̂² = χ²/(N − p_eff)`,
not `χ²/N`), but here it is only 0.4%, because `p_eff` is not the model pixel count. At
`dv = 1.5` km/s the grid oversamples FEROS by 4.2×, and the ML-II smoothness prior
constrains the spectra further, so of ~20,000 nominal spectral parameters only ~2,900 are
data-determined, roughly 60% of the resolution-element count. Inverting the two
estimators is an inexpensive route to `p_eff`, which is otherwise awkward to obtain. In
the `tests/test_jitter.py` fixture, built with a weak prior, the same inversion gives
`p_eff/N = 0.09` and the naive estimator is 4.6% low.

Per-epoch factors gain much more than one shared factor: +19,763 nats (A) and
+10,020 nats (B) over the best shared α. The exposures differ in quality: α ranges over
1.11–3.61 in window A, so the worst exposure has 1/13 of the weight the `ERR`-free
DER_SNR estimate would have given it.

### Per-epoch jitter: effect on the residuals and the orbit

A joint MAP over orbit, hyperparameters and 51 per-epoch jitters ran for 70 L-BFGS steps
(~380 s per window), with each window fitted independently:

| | A, no jitter | A, jitter | B, no jitter | B, jitter | Klement et al. 2025 |
|---|---|---|---|---|---|
| period [d] | 40.36583 ± 0.00045 | **40.44429 ± 0.00067** | 40.37022 ± 0.00065 | **40.42979 ± 0.00088** | 40.3261 ± 0.0013 |
| K<sub>pre-sd</sub> [km/s] | 63.314 ± 0.013 | **63.074 ± 0.022** | 63.724 ± 0.019 | **63.400 ± 0.024** | 61.15 ± 0.88 |
| K<sub>Be</sub> [km/s] | 1.985 ± 0.151 | **1.450 ± 0.209** | 3.022 ± 0.153 | **2.658 ± 0.248** | 3.90 ± 0.27 |
| eccentricity | 0.0302 | **0.0241** | — | **0.0213** | 0.0289 ± 0.0058 |
| whitened residual sd | 1.674 | **0.997** | 1.403 | **0.997** | 1 if calibrated |

The noise model is now self-consistent, with residual sd 0.997 in both windows. Of the
comparisons below, the two semi-amplitude agreements between windows improved and the
other three worsened:

| | no jitter | with jitter |
|---|---|---|
| window A vs B, period | 5.6 × combined formal σ | **13.1 ×** |
| window A vs B, K<sub>pre-sd</sub> | 17.8 × | 10.0 × |
| window A vs B, K<sub>Be</sub> | 4.8 × | 3.7 × |
| period vs literature (A) | 28.9 σ | **80.7 σ** |
| eccentricity vs literature (A) | 0.2 σ | 0.8 σ |

The error bars grew by about α, as expected (1.3–1.7×). The central values moved much
further: the period shifted by 0.078 d, which is 174× the no-jitter formal error on the
same window, and away from the published value.

### Cause of the period shift under jitter

The reweighting moved the optimum, and the optimizer did not fail. Both parameter vectors
were evaluated under both weightings:

| | weights α = 1 | weights α = MAP |
|---|---|---|
| θ from the no-jitter fit | **1,350,406** | 1,514,430 |
| θ from the jittered fit | 1,339,784 | **1,518,265** |

Each has the higher log-likelihood under its own weights, by 10,622 and 3,835 nats
respectively, so both fits are correct, each for its own objective. A period scan under
the jittered weights shows structure far wider than the surface's curvature. From the
no-jitter θ the conditional optimum is 40.3600 ± 0.0007 and from the jittered θ it is
40.4442 ± 0.0007, two optima 125 σ apart, with the second higher.

The cause is which exposures were downweighted. Within window A,
`corr(α, phase along the baseline) = −0.25`: the noisiest exposures are concentrated
early, with four of the worst five below phase 0.35 of the 134.7-day baseline.
Downweighting one end of the baseline weakens the period constraint there, and a period
fit pivots about the weighted centre of its data, which the reweighting moved from phase
0.581 to 0.605. Accordingly, the two solutions agree on a conjunction at BJD 2453221.7
(phase 0.615, within 1.4 d of that weighted centroid) and diverge on either side of it.

### Conclusions on the jitter site

* The jitter site is exact: jitter α is bit-equivalent to supplying `ivar/α²`. Its
  profiled value is the degrees-of-freedom (dof) corrected estimate, and it replaces a
  residual excess of 1.67× reported as a caveat with a fitted parameter.
* A jitter does not correct correlated residuals, because a rescaled diagonal noise model
  is still diagonal. Here it whitened the residual scale, left the source of the residual
  structure unmodelled, and moved the period by 174 formal σ.
* Two defensible noise models, fitted to the same data in the same window, disagree by far
  more than either one's stated error, so the noise model selects the optimum. The spread
  across independent windows and across defensible noise models should be quoted with any
  formal error.
* The eccentricity remains consistent with the literature, but less closely: 0.0241 and
  0.0213 against 0.0289 ± 0.0058, i.e. 0.8σ and 1.3σ, where the no-jitter run gave 0.2σ.
  The 0.2σ agreement was fortuitous.

## The response site and the continuum test (2026-08-12)

Per-epoch continuum residuals were a leading candidate for the 1.4–1.7× residual excess
on HR 6819. The multiplicative response coefficients, previously fixed at build time, are
now a θ site, and fitting them rules the continuum out.

### Exact response substitution

The response enters the targets ``z = y − r(R·1)`` and the sandwich weights ``w r²``, not
only the forward operator. The substitution is exact and inexpensive because ``R·1`` (the
rebinned unit continuum, stored per group) is response-independent.
``z_new = z_old + (r_old − r_new)·R·1`` rebuilds the target without the raw fluxes,
re-masked so that the ``0·nan`` defect found on HR 6819 cannot recur, and the
``Σ log w`` term is unchanged because the noise is defined on the data (math.md §7.5).
The traced Clenshaw recurrence matches ``np.polynomial.chebyshev.chebval`` operation for
operation, so `with_response` equals a fresh ``build_problem``, with ``r`` bitwise
identical and the marginal to rtol 1e-12 (`tests/test_response.py`).

### Closed loop (10 epochs, order 2, injected c ~ N(0, 0.03), SNR 130)

A joint MAP over orbit, hyperparameters and 30 response coefficients gave a
difference-mode error rms of 0.0020 for an injected coefficient rms of 0.0343, K errors of
+0.21% / −0.10%, and a fitted response 29,938 nats above the unit response at the same
orbit. The epoch-shared mode is returned at its zero-centered prior, not at the injected
value (c₀ error −0.069 against a 0.05 prior σ), which is §5's response-to-broad-features
degeneracy. The test asserts the common mode at prior scale, since a tighter assertion
would test the prior and not the data.

### HR 6819 with a fitted response

`scripts/hr6819_response_run.py` fitted both windows with 150 L-BFGS steps per
configuration, the response fits warm-started from the baseline MAP with order 2 per
epoch (153 coefficients) and prior N(0, 0.02²). Uncertainties are conditional-orbit
Laplace (nuisances at MAP), computed identically for all four fits. They are ~2× the
formal errors recorded above, which were obtained differently, and this does not affect
the comparisons below.

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

* The site works, and the continuum was already accurate. Coefficients of a few per mil
  absorb thousands of nats of epoch-structured signal, with epoch-to-epoch differences of
  5×10⁻⁴ rms in both windows. The log-space knot fit of `preprocess.normalize` had left
  half a per mil of per-epoch continuum error unmodelled.
* The orbit is nearly unchanged. The period shifts by +0.0004 / −0.0002 d (0.4σ / 0.2σ of
  the formal error, against 174σ for the jitter site), K by < 0.001 km/s, the eccentricity
  by ≤ 0.0001, and the residual sd by 0.4–0.5%, so the excess scatter is not a continuum
  error. Window-to-window disagreement is unchanged: ΔP 2.8σ → 3.1σ,
  ΔK<sub>pre-sd</sub> 10.5σ → 10.4σ.
* The continuum is ruled out by measurement. The remaining candidates for the correlated
  residual are the pipeline's resampling (the diagonal ivar understates the uncertainties
  by construction), the Gaussian approximation to FEROS's LSF, and the Be star's variable
  disc emission. The next steps are a correlated-noise model and a wider window. The
  continuum treatment remains a minor correction, not a remedy.

Window A's baseline reproduces the earlier record to 0.0002 d. Window B's
uniform-procedure MAP is P = 40.36091, 0.0093 d below the 40.37022 recorded there and
6.6× the combined formal errors, with both runs converged (parameters stationary to
~0.0002 d over the final 20 steps). With the jitter shift (174σ) and the two optima of the
period scan (125σ apart), this is the third case in which this surface has optima far
outside their curvature widths, selected by the optimizer's path. Every comparison in the
table above is therefore between fits sharing one procedure.

## The AR(1) chain: effect on the residuals and the orbit (2026-08-12)

With the continuum ruled out, the pipeline's resampling correlations are the leading
candidate. They are modelled as an AR(1) chain per epoch,
`C = α²·D^(−1/2)·R_φ·D^(−1/2)` (math.md §1.4a), with φ a θ site shared across epochs
(a property of the resampling, not of one exposure) and fitted alongside the per-epoch
jitters. The model whitens the residuals in both moments without the jitter model's
period shift.

### Closed forms and the autocorrelation diagnostic

The correlation matrix `R_φ` has unit diagonal by construction, so heteroscedastic pixels
keep their supplied variances and the residual-sd diagnostic is insensitive to φ. The
diagnostic for φ is the lag-1 autocorrelation of the whitened residuals: ~φ under diagonal
whitening, ~0 under the chain whitener.

Masked pixels are treated exactly: a subset of a Markov chain is Markov, so a gap becomes
a single link with `ρ = φ^gap`. The gap is capped at build time (`ar1_max_gap=4`) and the
chain restarts beyond it, which at |φ| ≤ 0.9 discards ρ < 0.66⁴ ≈ 0.2 in the worst case
and ~1e-3 at the fitted values below. The precision stays tridiagonal with closed-form
`log det`, and the whitener is the innovation transform
`(ε_i − ρ_i·ε_prev)/√(1 − ρ_i²)`. All of this is tested against an independent dense
reference, the chain correlation matrix built from link products and inverted with
LAPACK. The marginal log-likelihood agrees to rtol 1e-10 with gaps and jitter composed,
`φ = 0` reproduces the diagonal model to rtol 1e-12, and gradients agree with finite
differences, including at φ = 0. `jnp.power` has a nan gradient at a zero base, so the
gap-1 links use φ directly.

The chain has two structural costs. The direct band assembly assumes diagonal weights, so
a correlated problem uses the probe path: 2p + 1 = 347 operator applications per
evaluation with plain reverse-mode gradients, ~15× the response-site per-step cost as
measured below. The tridiagonal band-sandwich extension was deferred until the AR(1)
model was adopted, and is built in the band-assembly section below. The chain also
couples pixels across masked gaps, so the solver bandwidth grows by a statically reserved
`ar_bandwidth_extra` (the declared-bandwidth convention; 5 and 6 model pixels on the
HR 6819 windows), enabled by an explicit `MarginalOrbitModel(ar1=True)`.

### Closed loop: joint recovery of φ and α

At acceptance-test scale (10 epochs, SNR 130), both miscalibrations were injected at once:
correlation φ = 0.45 and a supplied `ivar` overstated by α² = 1.5²:

| | injected | recovered |
|---|---|---|
| φ | 0.45 | **0.4493** |
| α | 1.5 | **1.4868** |
| K errors | — | −0.05% / −0.31% |
| chain-whitened residual sd, lag-1 | 1, 0 | 0.944, −0.077 |
| same residuals, diagonal whitener: lag-1 | φ ≈ 0.45 | **+0.395** |

The 0.944 is not a miscalibration: residuals about the fitted spectra are low by
`√(1 − p_eff/N)` (math.md §3.2a, the dof effect of the jitter section; p_eff/N ≈ 0.11 at
this scale), and the marginal's own α̂ is dof-corrected. On the identical residual vectors
of the last two rows, the diagonal whitener retains the injected correlation and the
chain whitener removes it.

### HR 6819 with the chain noise model

The uniform procedure of the response-site fits was used (conjunction scan, literature
initialization, 150 L-BFGS steps), with θ = orbit + hyperparameters + 51 per-epoch
jitters + shared φ. The probe path took ~53–56 s/step (8,369 / 7,948 s per window)
against ~3.6–4.2 s/step for those fits.

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

* The noise model is self-consistent in both moments: both windows have sd 0.997 with
  lag-1 +0.041 and +0.012, the first fits in this campaign to do so (the jitter corrected
  the scale and left the structure, and the response site corrected neither). As a check,
  the same residuals under the diagonal whitener have lag-1 +0.797 and +0.688, which is φ̂
  (+0.801, +0.694) to two decimals. In ML-II terms, window A's chain model is ~1.7×10⁵
  nats above the diagonal-jitter model (1,691,463 vs the 1,518,265 recorded at that
  model's own MAP): the correlation term accounts for most of the miscalibration.
* The jitter model's period shift does not recur. Under the chain the per-epoch jitters
  narrow from 1.11–3.61 to 1.55–1.93, and their medians (1.66 and 1.40) are close to the
  earlier residual sds (1.674, 1.403). The diagonal jitters were fitting correlated
  structure, epoch by epoch, as per-exposure scale, and the epochs where they over-fitted
  it were the ones whose downweighting moved the period. Modelled as correlation, the
  excess is roughly uniform across epochs, as the resampling hypothesis predicts for a
  pipeline property, and the period is within 0.006/0.009 d of the diagonal baselines
  instead of 0.08 d away.
* The two windows agree better in period: ΔP is 0.00475 d at the baseline and 0.00159 d
  under the chain, 3.0× closer, where the jitter model widened it to 0.0145 d. The
  K<sub>pre-sd</sub> spread is nearly unchanged (0.267 → 0.276 km/s). This is the first
  noise model to improve the agreement between the windows, the available internal
  evidence that the chain describes these data more accurately.
* K<sub>Be</sub> moves toward the literature and remains unmeasured: 1.93 → 2.45 (A) and
  2.95 → 3.76 (B), against 3.90 ± 0.27, so window B is now within 1σ of Klement et al.
  The A–B spread grew (1.02 → 1.31 km/s) and A's value was still changing at step 150.
  A noise model does not resolve the Be star's ~1/50-linewidth reflex (math.md §5.1's
  unattributable regime).
* The period offset from the literature persists under a third noise model: 40.3712 and
  40.3696 against 40.3261, a difference of 0.044 d, marginally larger than at the
  baseline. Measurement rules out the continuum, the noise scale and the pixel
  correlation as its cause. The remaining candidates are the Gaussian approximation to
  FEROS's LSF, the Be disc's variability, and the published CCF analysis, which blends
  the components this model separates and cannot be tested here. φ̂ = 0.7–0.8 sets the
  scale by which the first fit understated its uncertainties: at these correlations a
  diagonal model overcounts low-frequency information by (1+φ)/(1−φ) ≈ 5.5–9×, which is
  why formal errors of ±0.0005 d accompanied window disagreements of 0.005 d.

## The numpyro path: the problem as a traced argument (2026-08-12)

`MarginalOrbitModel.marginal` passes the `Problem` pytree to `jax.jit` as an argument,
because closure-captured arrays are embedded in the graph as constants. XLA then
evaluates every θ-independent subgraph over them at compile time, at a cost in
compile-time memory (~80 GB at the design target when first measured). The numpyro path
did not do this: the model closure from `MarginalOrbitModel.model()` captured
`self.problem`, so `run_map`'s jitted L-BFGS step and the NUTS sample loop both compiled
with the problem embedded. The fix uses numpyro's own mechanism for this case:

* the model takes the base problem as an optional argument and exposes it through a
  `model_args` attribute on the returned closure;
* `run_map` and `laplace_inverse_mass` build the potential with
  `initialize_model(..., dynamic_args=True, model_args=...)`;
* `run_nuts` runs `MCMC(..., jit_model_args=True)`, which regenerates the potential from
  the traced arguments inside the jitted sample loop (`hmc.py`'s `_potential_fn_gen`).

Existing call sites need no change (the runners resolve `model_args` as explicit
argument > model attribute > none). Calling the model with no argument, as plain numpyro
utilities such as `log_density` do, uses the closure, which is correct but incurs the
compile-time cost at scale. `model_args=()` selects that path explicitly.

### Cost of the numpyro potential: compile time, memory and run time

`scripts/d32_model_args_bench.py` timed value+grad of the numpyro potential (the graph
that L-BFGS and NUTS compile) on the m5-ladder SB2 (50 epochs, p = 513), CPU, with one
process per cell (the peak-working-set counter is monotone) and a single run per cell:

| | 31,734 px, arg | 31,734 px, closure | 74,322 px, arg | 74,322 px, closure |
|---|---|---|---|---|
| compile | **0.8 s** | 20.9 s | **0.8 s** | 10.8 s |
| constants embedded in the executable | 0.02 GiB | **1.99 GiB** | 0.04 GiB | 0.04 GiB |
| process peak during compile | +0.00 GiB | **+1.35 GiB** | +0.00 GiB | +0.00 GiB |
| value+grad runtime | 6.2 s | 5.9 s | 20.6 s | 19.0 s |
| potential, gradient | identical to all printed digits | ← | identical | ← |

During the closure builds XLA's slow-operation alarm identifies the mechanism, on the
predicted instructions: "Constant folding an instruction is taking > 1s:
%scatter-add.342 = f64[50,126936] scatter(...)". These are θ-independent weight/response
subgraphs evaluated at compile time, at 50 × native-pixels scale.

Two qualifications apply:

* The memory cost depends on XLA's folding heuristics. At 31.7k px XLA folded ~2 GiB of
  derived constants into the executable. At 74.3k px its folding guards skipped the
  largest folds, so the memory cost did not occur while the compile-time cost (13×)
  remained. Whether it recurs at a given scale depends on XLA's internal thresholds and
  version, and passing the problem as an argument removes that dependence.
* Folded constants are marginally faster at runtime (5.9 vs 6.2 s at row 0), the
  trade-off XLA is designed to make. At survey scale it is the wrong one: the same
  mechanism costs tens of GB at the design target, and a NUTS warmup recompiling per
  mass-matrix window would repeat the folding.

### Regression tests (`tests/test_inference.py`)

* `test_potential_with_model_args_embeds_no_problem_constants` asserts on the jaxpr
  consts in both directions: nothing problem-sized with the problem as an argument, and
  problem-sized constants in the closure build, which confirms that the check detects
  them.
* `test_run_map_closure_and_argument_paths_agree` (float tolerance, different graphs) and
  `test_laplace_closure_and_argument_paths_agree` (exact, same eager ops) compare the two
  paths.
* The NUTS acceptance test now runs through `jit_model_args=True` by default, so it also
  regression-tests the traced sample loop.

## Band assembly with correlated noise (2026-08-12)

The AR(1) fits above ran on the probe path at ~15× the response-site per-step cost. AR(1)
is the only noise model that whitens both moments and the first to bring the two windows
into closer agreement, so the tridiagonal band-sandwich extension was built. The next
step, a wider window, also cannot run on the probe path, whose gradient peaked at
23.7 GiB at the current window scale on a 32 GB machine.

### The extension

The only stage of the direct band assembly that assumed diagonal noise is the innermost
sandwich `H = RᵀW′R`. The chain adds one symmetric cross-row term per link,
`−c·√(wₙw_p)·rₙr_p·(RₙᵀR_p + R_pᵀRₙ)`, and re-weights the diagonal by the chain
diagonal `1 + Σ a` (math.md §4.5a). Both use the mechanism that already handles the
diagonal. A second set of static pair tables (`operators.rebin_link_pair_tables`), built
at `build_problem` time over the union of links realized in any epoch, is consumed by one
extra `segment_sum` per epoch whose traced weights depend on φ, α and r. Because masks
differ by epoch, a per-epoch gap test selects each epoch's own links from the shared
tables. `H` widens by the group's static `ar_step`, which is the `ar_bandwidth_extra`
the bandwidth reservation already declared. All later stages are unchanged: the LSF
convolutions, the T-sandwich, the band accumulation, the chunking, and the custom-VJP
solve operate on a slightly wider velocity-independent band image. The likelihood now
always auto-selects `"band"`, and probing remains the reference implementation and the
`validate` oracle.

Exactness is preserved: the dense-LAPACK reference test (gaps + jitter composed) passes on
the band path at the same rtol 1e-10, band = probe at rtol 1e-12 with ∂/∂φ agreeing to
1e-9, and `epoch_chunk` batching leaves the result unchanged. The AR weight tuple
(diagonal, link, gap table) is padded and sliced together, which a test asserts because a
batched run that dropped link terms would fail silently.

### Correlated marginal at HR-window scale: timing and memory

`scripts/d35_ar1_band_bench.py` used 51 epochs, 9,796 model px, 367,200 native px and
half-bandwidth 85 on CPU, with one assembly path per process (the peak-working-set counter
is monotone). Gradients are in velocities, φ and the jitter, i.e. one L-BFGS step's work:

| | probe path | band path | ratio |
|---|---|---|---|
| eval, steady | 5.10 s | **0.71 s** | 7.2× |
| value+grad, steady | 58.83 s | **3.07 s** | **19.2×** |
| eval peak working set | 11.34 GiB | **1.14 GiB** | 9.9× |
| grad peak working set | 23.69 GiB | **1.85 GiB** | **12.8×** |
| log-likelihood, gradients | 1165077.330 | identical to all printed digits | — |

The gradient ratio exceeds the eval ratio, as in the first speedup pass: the band path's
solve stage has the closed-form custom VJP, while the probe path uses plain reverse mode
through 2p + 1 = 347 operator applications. At 1.85 GiB the wider window fits in memory,
where 23.7 GiB was already close to the machine's limit.

### Window A refitted on the band path

`scripts/hr6819_ar1_run.py --windows A`, rerun with only the assembly changed, took 868 s
where the probe path took 8,369 (9.6× end-to-end, 5.8 vs 55.8 s/step). It converged to the
same optimum: log-likelihood 1,691,463.5 to the printed digit, φ̂ +0.801, P 40.37113 vs
40.37115 (0.02 formal σ), with α range 1.55–1.93 (median 1.66) and residual diagnostics
(chain sd/lag-1 0.997/+0.041, diagonal lag-1 +0.797) identical. K<sub>Be</sub> is 2.420
vs 2.446, a difference along the direction in which both runs were still moving at step
150, the flattest axis of the surface, and not a path discrepancy.

## One wide window: 4120–4600 Å, Hγ masked (2026-08-12)

Feasible only with the band assembly, `scripts/hr6819_wide_run.py` joins windows A and B
into a single fit of 22,169 model px and ~765k good native pixels, 2.26× window A,
including the 25 Å strip 4355–4380 not previously fitted. Hγ's core (4325–4355 Å) is
masked by `preprocess.mask_ranges`: ivar = 0 keeps the sampling regular, the AR(1) chain
restarts across the masked range (a gap beyond `ar1_max_gap`), and the bandwidth is
unchanged. The broad absorption wings, which are static stellar features, remain in the
fit. The noise model and procedure are those of the AR(1) configuration. The fit ran 200
L-BFGS steps in 2,447 s at 12.2 s/step, linear in pixels from the window A refit
(5.8 s/step at 0.44× the size). The probe path would have taken roughly 7 hours and
needed a ~50 GB gradient, which exceeds this machine's memory.

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

* The noise model remains self-consistent at 2.3× the data: sd 0.997 with lag-1 +0.019,
  and the diagonal whitener gives +0.731 against φ̂ = +0.737. φ̂ and the α̂ range lie
  between the two single-window values, as expected of a pipeline property that varies
  mildly with wavelength.
* K<sub>Be</sub> moves toward the literature: 3.48 against 3.90 ± 0.27 (1.5σ), where
  window A alone gave 2.42. The ~1/50-linewidth reflex is the direction the data
  constrain least (math.md §5.1), and more lines in one joint fit constrain it better.
* The eccentricity is unchanged at 0.0261, 0.5σ from the published 0.0289 ± 0.0058.
* The period is below both single-window optima, 40.3675 against 40.3711/40.3696, and
  outside their interval. A joint fit is not an average of window MAPs (the cross-window
  coupling and the new pixels add information), and the multimodality noted above
  applies to any single optimum. The value moves toward the literature and remains
  0.041 d away.

The period offset from the literature persists in a fourth configuration. Across two
independent windows, three noise models, and one joint wide fit, P ∈ [40.3675, 40.3712],
internally consistent to 0.004 d, against a published 40.3261 ± 0.0013. K<sub>pre-sd</sub>
is 63.2–63.5, consistently 3.7–4% above the CCF value, in the direction deblending
predicts. Measurement rules out the continuum, the noise scale, the pixel correlation and
the window choice. The remaining candidates are the Gaussian approximation to FEROS's LSF
(a tabulated LSF is a banded-operator substitution, which the design provides for), the
Be disc's variability, and the blending in the published CCF analysis.

## Tabulated LSF: fitted σ(λ) and its effect on the orbit (2026-08-12)

The design reserved this extension (a tabulated LSF in v2, as a banded matrix with no
structural change). The kernel is now a per-pixel profile bank realized from per-anchor
kernels through static log-λ interpolation tables (`operators.convolve_varying`, exact
adjoint pair, arbitrary asymmetric banks accepted). The band assembly keeps its
structure, with scalar taps replaced by row-shifted profile columns and the second
sandwich application run against the band-transpose of the first (G = Kᵀ(KᵀH)ᵀ, by G's
symmetry, since only left applications broadcast on a row-major band image). Band, probe
and dense agree to rtol 1e-12/1e-10 under diagonal and AR(1) noise, with per-anchor width
gradients to 1e-9, and random asymmetric banks guard against an undetected kernel flip.
The suite had 320 tests.

In the closed loop (acceptance-test scale, injected σ ramp 5.0→9.5 km/s) the joint fit
leaves the orbit unbiased (K to 0.3%) and recovers the ramp's direction, but the marginal
is not highest at the injected profile. A flat width exceeded it by ~3 nats, and the
maximum-likelihood profile exceeded it by ~8 while ~3 km/s from the injected value at one
anchor. A stationary kernel change commutes with the shifts, so the free spectra absorb
it (deconvolution), and the preferred width is set mainly by the smoothness prior, not by
the instrument. Only the anchor-to-anchor variation is data-identified, through the
epoch-dependent shifts. Fitted anchor widths are therefore diagnostics, not measurements,
and the result of interest is the orbit's response.

On HR 6819, `scripts/hr6819_lsf_run.py` used the wide-window configuration (4120–4600 Å,
Hγ core masked, per-epoch jitters + shared AR(1) φ) plus 13 Gaussian width anchors every
40 Å (the FEROS order scale), bounded to 1.5–3.5 km/s around the nominal 2.652. The bound
sets the kernel radius (half-bandwidth 91 vs 87). The fit ran 200 L-BFGS steps in 5,650 s
at 28.3 s/step (2.3× the wide-window fit: larger radius, varying-kernel band stages,
13-anchor VJP).

| | fixed σ = 2.652 | **fitted σ(λ) ×13** | Klement et al. 2025 |
|---|---|---|---|
| period [d] | 40.36750 | **40.36769** | 40.3261 ± 0.0013 |
| K<sub>pre-sd</sub> [km/s] | 63.396 | **63.395** | 61.15 ± 0.88 |
| eccentricity | 0.0261 | **0.0262** | 0.0289 ± 0.0058 |
| φ̂ | +0.737 | **+0.737** | — |
| α̂ range (median) | 1.46–1.86 (1.61) | 1.46–1.86 (1.61) | — |
| whitened residual sd, lag-1 | 0.997, +0.019 | 0.998, +0.019 | 1, 0 |
| log-likelihood | 3,388,604.2 | **3,388,694.7** | — |

σ̂(λ) at the anchors is 2.15 km/s at 4120 Å and 3.1–3.4 km/s across the rest of the
window, close to the 3.5 bound. As in the closed loop, broader kernels imply smoother
spectra and a higher marginal likelihood, so the absolute level is bound-limited and
diagnostic only. K<sub>Be</sub> is omitted from the table because it did not converge in
200 steps. It oscillated over 1.30–4.16 across the last 100 steps (a range that includes
the fixed-σ value 3.48), ending at 1.30 with |grad| 59 where that fit ended at 3.85. The
13 near-flat width directions slow convergence along the already-flattest axis. Every
tabulated quantity above was stable over the same trajectory (P within ±0.001, K₁ within
±0.02, φ̂ to three digits).

The fitted width profile gains +90.5 nats, so the effective width has wavelength
structure, but the orbit does not respond: P +0.0002 d (0.5% of the offset, within the
trajectory variation), K₁ −0.001 km/s, e +0.0001, with φ̂ and the residual moments
unchanged to one unit in the last tabulated digit. The response site gave the same
pattern (+4,100 nats absorbed, orbit unchanged). The period offset persists in a fifth
configuration, and σ(λ) is ruled out. The remaining LSF candidate is profile asymmetry,
the first-order centroid channel, whose epoch-coupled part enters as an apparent velocity
perturbation ∝ λc′(λ)v(t)/c (math.md §1.3). The operator already accepts arbitrary banks,
so only a θ-parameterization (e.g. per-anchor Gauss–Hermite h₃) would be new. Beyond the
LSF, the candidates are disc variability and the published CCF blending.

## LSF asymmetry: fitted h₃ and its effect on the orbit (2026-08-13)

Profile asymmetry, the first-order centroid effect that a symmetric kernel cannot produce,
is parameterized by a per-anchor Gauss–Hermite h₃ (`operators.gauss_hermite_kernel_traced`,
|h₃| ≤ 0.2, h₃ = 0 bit-identical to the Gaussian implementation) as an `lsf_h3` site.

In the closed loop h₃ is less well identified than the widths: an injected h₃ ramp of
∓0.12 was returned flat (fitted |h₃| ≤ 0.03) with the orbit recovered to 1%. A free
spectrum can represent any static centroid-warp field c(λ) ≈ √3·h₃(λ)·σ, so the only
data-identified part is the epoch-coupled sampling of the warp's gradient,
Δc ≈ c′(λ)·λ·(v − v_bary)/c ≈ 30 m/s at this configuration (math.md §1.3). This is two
orders below the ~4 km/s of accumulated RV signature the 0.041 d offset represents, and
it also rules out the instrument-frame per-epoch kernel realization. The fixed-spectra
data term is higher for the injected profile, so the injection is present and detected.
Band, probe and dense agree with h₃ anchors under diagonal and AR(1) noise, with
gradients in h₃ to 1e-9. The suite had 329 tests.

On HR 6819, `scripts/hr6819_h3_run.py` used the fitted-σ(λ) configuration plus 13 free h₃
anchors, 26 LSF parameters joint with the orbit and the AR(1) noise model. It ran 300
L-BFGS steps in 10,789 s at 36.0 s/step and ended at |grad| 44, better converged than the
200-step width run.

| | fixed LSF | fitted σ(λ) | **fitted σ, h₃** | Klement et al. 2025 |
|---|---|---|---|---|
| period [d] | 40.36750 | 40.36769 | **40.36719** | 40.3261 ± 0.0013 |
| K<sub>pre-sd</sub> [km/s] | 63.396 | 63.395 | **63.391** | 61.15 ± 0.88 |
| eccentricity | 0.0261 | 0.0262 | **0.0261** | 0.0289 ± 0.0058 |
| φ̂ | +0.737 | +0.737 | **+0.737** | — |
| whitened residual sd, lag-1 | 0.997, +0.019 | 0.998, +0.019 | **0.998, +0.020** | 1, 0 |
| log-likelihood | 3,388,604.2 | 3,388,694.7 | **3,388,721.3** | — |

ĥ₃(λ) is at the |h₃| ≤ 0.02 level at the interior anchors. The largest values,
0.042–0.052, are at three anchors including the poorly constrained blue edge, and imply
centroid shifts of −0.13 to +0.31 km/s, a 0.53 km/s spread. All are diagnostics by the
closed-loop measurement. The fit gains +26.6 nats over the width fit for 13 parameters,
an order below the widths' +90.5, because asymmetry has far less to absorb once the
spectra are free. K<sub>Be</sub> again did not converge along its flat axis
(2.73 at |grad| 44, inside the 1.3–4.2 range of the width run). Every tabulated quantity
above was stable.

The LSF is ruled out in full, and the offset persists in a sixth configuration. P moved
−0.0005 d from the width fit, within its own trajectory variation, and K₁, e, φ̂, α̂ and
both residual moments changed by at most a few units in the last tabulated digit. Every
instrumental effect this model can represent has been given a θ site and tested against
the orbit: the continuum (+4.1k nats), the noise scale, the pixel correlation
(+1.7e5 nats), the window choice, the LSF width (+90.5 nats), and the LSF asymmetry
(+26.6 nats). None removed the period offset. Across all six configurations
P ∈ [40.3672, 40.3712], internally consistent to 0.004 d, against a published
40.3261 ± 0.0013. The remaining candidates are not instrumental: the Be disc's
variability (a time-variable component this static-spectrum model cannot represent, and
a systematic of the published analysis as well), and the published CCF blending, which
measures velocities on composite line profiles this model separates. The
instrumental-systematics campaign on this dataset is complete.


---

## The nebular component, and per-pixel prior strengths (2026-08-13)

`tests/test_nebular.py` and `examples/04_nebular.py` simulate one SB2 in an H II region:
12 epochs, SNR 220, 540 model pixels over 4838-4886 A, K = (58, 41) km/s, light fractions
(0.7, 0.3). Both stars have a broad Hbeta absorption (true composite depth -0.506,
EW 1.911 A). A static nebular Hbeta emission line of peak 0.45 varies in amplitude
by +-30% per epoch, with a factor of ~2 between the best and worst night.

### Exactness

| Check | Result |
|---|---|
| forward model vs. the simulator's injection, barycentric and topocentric | atol **1e-12** |
| `with_velocities` + `with_light_fractions` vs. a fresh `build_problem` | rtol **1e-14** (the nebular and telluric columns are kept, not rebuilt) |
| `with_nebular_amplitudes` vs. a fresh `build_problem` | bit-identical |
| band assembly vs. the matrix-free operator, nebular column + window profile | `validate=True`, rel err < 1e-10 |
| band vs. probe assembly, same configuration | log-likelihood rtol **1e-11**, spectra atol 1e-9 |
| per-pixel prior `apply` / `dense` / `prior_logdet` vs. dense NumPy | rtol **1e-12 / 1e-10** |
| uniform profile vs. the unprofiled prior | rtol 1e-14 (`apply`, `dense`, `prior_logdet`) |
| d(log L)/d(log_nebular_amp) vs. central differences | < 1e-4 relative; the gradient sums to zero, as centering requires |

The determinant recursion is the component most prone to an undetected error:
`prior_logdet` is an O(P) scalar Cholesky over the pentadiagonal prior, and per-pixel
`tau` and `eta` change all three of its diagonals. It is checked against `slogdet` of the
dense construction with random profiles spanning 0.2-40 in curvature and 0.1-1e4 in ridge,
not against a uniform special case.

### Effect of the contamination on the spectra (orbit fixed at the injected values)

The same data were disentangled twice with identical stellar priors, with and without the
nebular component, and with the orbit fixed at the injected values.

| | injected | no nebular component | **with the component** |
|---|---|---|---|
| Hbeta core depth (light-weighted composite) | -0.506 | -0.375 (**26% shallower**) | **-0.508** |
| mean core error | 0 | **+0.154** | **+0.0015** |
| core RMS error | 0 | 0.155 | **0.0057** |
| Hbeta equivalent width [A] | 1.911 | 1.690 (**-11.5%**) | **1.908 (-0.14%)** |
| marginal log-likelihood | — | reference | **+81,424 nats** |

Both log-likelihoods are marginal, with the component spectra integrated out and the
Occam terms included, so their difference is a Bayes factor rather than a fit-quality
score. The extra component lowers the marginal likelihood unless coherent signal offsets
the penalty, and here the net gain is 8.1e4 nats.

Equivalent width is the quantity passed to the atmosphere code. An 11.5% error in a
Balmer EW is a large, systematic error in log g, and nothing in the current literature
propagates it: disentangled spectra are passed to the next stage without an uncertainty.

`examples/04_nebular.py` adds the third treatment used in the literature, masking the
contaminated pixels (`ivar = 0` over +-150 km/s). It is defensible but has a cost: the
masked pixels are constrained only by the prior, so the composite is recovered flat there
and the spectrum is incomplete at the wavelengths where a Balmer gravity diagnostic is
measured. The three-way comparison takes 9 seconds to produce.

### Effect of the contamination on the orbit (joint MAP, cold start)

Both fits used the same data, priors and starting point, with 300 L-BFGS steps of ML-II
MAP over the orbit and the hyperparameters (plus the 12 log-amplitudes when the component
exists).

| | injected | no nebular component | **with the component** |
|---|---|---|---|
| K<sub>1</sub> [km/s] | 58.0 | 57.38 (-1.1%) | **57.91 (-0.15%)** |
| K<sub>2</sub> [km/s] | 41.0 | **16.77 (-59.1%)** | **40.88 (-0.29%)** |
| period [d] | 5.70000 | **5.87115 (+0.171)** | **5.69986 (-0.00014)** |
| eccentricity | 0 | **0.950 (the solver's clip)** | **0.0022** |
| potential at the end | — | +30,629 | **-19,220** |
| gradient norm at the end | — | 2.9e4 (not stationary) | 8.3 (stationary) |
| wall-clock time | — | 177 s | 49 s |

A static line is a component with K = 0, so a model with no component for it assigns it
to whichever stellar component can be made to move least. The secondary's semi-amplitude
falls by 59%, and the period and eccentricity change with it, giving a circular orbit
reported at *e* = 0.95, which is the eccentricity clip rather than a fit. Only
K<sub>1</sub> is recovered, because 70% of the light constrains it. Without the component
the fit is not stationary at 300 steps, unlike the fit with it, and takes 3.6x the
wall-clock time. Neither sets `MAPResult.converged`: that flag tests an absolute
gradient-norm tolerance that is unreachable at these pixel counts, as on HR 6819. The
three orders of magnitude between the two gradient norms is the usable indicator.

The recovered per-epoch amplitudes have correlation 0.99930 with the injected ones and
0.0066 rms difference in log, against an injected spread of 0.78x to 1.50x. They are
compared after centering, because only `a_j * d_neb` is observable and the geometric mean
is a convention (math.md §1.3). ML-II independently makes the nebular component less
smooth than the stellar ones (log tau 7.9 against 11.4), recovering a shape about which
the prior had no information.

The window profile affects the fit. The same joint fit with the nebular component free
across the whole grid gives K<sub>2</sub> at +2.6% instead of -0.29%, with the potential
250 nats worse: without the profile the component absorbs stellar signal at wavelengths
where a nebula has no lines. The measurement also exposed a defect: `MarginalOrbitModel`
rebuilt the prior from the sampled `log_tau`/`log_eta` and dropped the profiles, so a
windowed component silently lost its confinement whenever ML-II was enabled. The profiles
are structure and the scalars are hyperparameters, and the merge now keeps the profiles
(math.md §2).

### Conclusions on the nebular component

Nebular contamination is more damaging than the literature describes, and correcting it
is inexpensive. The published concern is line-profile narrowing and biased atmospheric
parameters, which is confirmed (-11.5% in EW). The contamination also propagates into
the dynamical result, the masses, through a 59% error in K<sub>2</sub>. One extra
component and twelve extra parameters remove both, at 49 s against 177 s for the fit
without the component.

The nebular column is one more column of A with a different velocity law and a free
amplitude, so the band assembly, the AR(1) link tables, the chunking policy, the
custom-VJP solve, and the bandwidth contract are unchanged. The per-pixel prior
generalizes three diagonals and keeps the same O(P) determinant recursion. By the same
property of the linear-Gaussian family, a general time-variable component, of which this
is the rank-one case, can be a change of basis rather than a change of method.

Two degeneracies are resolved by convention, not by data (math.md §5.4): the amplitude
scale, fixed by centering the log-amplitudes, and the nebular velocity, which sets where
the component's lines fall on the model grid and is not a measurement.



---

## Calibrated faint-companion detection (2026-08-13)

This work vectorized the scan, marginalized K₁, and calibrated the statistic by injection
and recovery. `tests/test_calibrate.py` and `examples/05_detection_limit.py` use one
SB1/SB2 pair (14 epochs, SNR 200, 717 model pixels over 5000-5060 A, 520 native pixels,
K = (55, 40) km/s, light fractions (0.93, 0.07), P = 7.3 d, e = 0.12) scanned on a
20-point K₂ grid from 14 to 71 km/s.

### The vectorized sweep

One batched `lax.map` over the trial grid replaces a Python loop (one jitted call and one
device synchronization per point). Timings are best of three on a shared machine.

| model pixels | native | epochs | half-bandwidth | loop / point | sweep / point | speedup | relative agreement |
|---|---|---|---|---|---|---|---|
| 201 | 100 | 8 | 46 | 3.71 ms | 1.33 ms | **2.8x** | 1.1e-12 |
| 717 | 520 | 14 | 55 | 17.34 ms | 8.66 ms | **2.0x** | 1.4e-13 |
| 2,652 | 2,150 | 20 | 66 | 97.41 ms | 43.91 ms | **2.2x** | 1.8e-16 |

The factor is nearly independent of problem size, so the gain comes from the batching
rather than from removing per-point dispatch (a dispatch-overhead explanation would
predict the opposite). The acceptance test asserts only 1.5x. The sweep is not
bit-identical to the loop: batching re-associates the linear algebra, and the
log-likelihoods differ in the last few digits. It makes two further features practical: a
7x20 (K₁, K₂) grid costs 0.88 s where the loop would need ~1.41 s, and the 450-scan
calibration below, 9,450 marginal solves, costs 53 s.

### Marginalized K₁ compared with a fixed, incorrect K₁

The literature reports that a small error in the assumed primary semi-amplitude produces
spurious features in the recovered secondary spectrum. The effect on the detection
statistic is unreported and more consequential.

| K₁ treatment | K₂ peak [km/s] | companion line-pattern correlation | D at the peak |
|---|---|---|---|
| correct, fixed | 41 | **0.961** | 40,609 |
| 5% high (57.75), fixed | 38 (one grid step low) | 0.720 | **66,837** |
| 5% high, marginalized (σ = 5%) | **41** | **0.955** | 40,455 |
| 10% high (60.5), fixed | 41 | **0.486** | **135,410** |
| 10% high, marginalized (σ = 10%) | **41** | **0.931** | 41,310 |

A wrong K₁ increases D while the recovered spectrum degrades. Unremoved primary signal
is coherent across epochs, and the companion's free spectrum is the only component that
can absorb it, so D more than triples while the recovered spectrum correlates 0.49 with
the injected one. Marginalizing over a Gauss-Hermite rule on N(μ₁, σ₁²), applied to the
companion model and the no-companion model so that D stays a ratio of two marginal
likelihoods, recovers 0.93-0.96 and returns D to the level it has with the correct K₁.
Correlations are offset-removed: at ℓ₂ = 0.07 the companion's smooth envelope is
prior-dominated (math.md §5.1-5.2), so the information is in the line pattern.

The 7-node rule costs about 30% more wall-clock time than the fixed scan (2.2 s against
1.7 s), not 7x, because the trials share one compiled graph.

### The calibrated limit

The null distribution uses 200 companion-free trials and completeness uses 50 trials at
each of six injected light fractions, all resimulated through the observed dataset's own
operators. The 450 full scans took 71 s.

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

> Any companion contributing more than 0.30% of the light would have been detected at
> 95% confidence, against a detection threshold D > -692.2 set at a 1% false-alarm
> probability from 200 companion-free trials.

The null peaks are strictly negative, because the marginal likelihood's Occam term
penalizes the companion's free spectrum and, with no companion present, no signal offsets
the penalty. "D > 0" would have been a conservative test on this dataset. On another it
need not be, which is why the threshold is measured.

Two properties are enforced by construction. The threshold is defined through the
false-alarm estimator (1 + #{null >= D})/(N+1) rather than as a sample quantile.
`np.quantile` interpolates between order statistics, and on a 24-trial run it left 8.3%
of the null above a nominal 5% threshold (detected by a test), which is anti-conservative
in the direction that matters for a detection claim. No FAP below 1/(N+1) is reported.
Below that value the rule reduces to "must exceed every null trial".

### Dependence of the limit on K₂

| injected K₂ [km/s] | 20 | 40 | 65 |
|---|---|---|---|
| limit on ℓ₂ | 0.292% | 0.296% | 0.297% |

The limit is flat in K₂. In an SB2 the components move in antiphase, so their relative
velocity never falls below about K₁, and at K₁ = 55 km/s the pair is well separated at
every trial K₂. A K₂ dependence should appear only when K₁ is small enough that the pair
is barely resolved at any phase.

The calibration does not check the model. The null trials are drawn at the same K₁, orbit
and light fractions the scan assumes, so the threshold is self-consistent with those
assumptions and insensitive to errors in them. The K₁ table above is such a case, and no
calibrated threshold would have detected it. The limit is also conditional on the assumed
companion template: the observable is ℓ₂·d₂, and a featureless companion is undetectable
at any light fraction. Both conditions are stated wherever the numbers are.

---

## The free per-epoch radial-velocity table (2026-08-13)

No Keplerian is fitted: each epoch's velocity is a separate parameter. The numbers are from
`tests/test_velocity_table.py`. The fixture is one SB2 with 10 epochs, SNR 200, 400 model
pixels at dv = 6.00 km/s over 5000-5040 A (284 native pixels), K = (30, 55) km/s, light
fractions (0.6, 0.4), P = 6.31 d and e = 0.15.

### The zero point and its removal

A free table has one arbitrary zero point per component: with no orbit relating the stars,
each free spectrum absorbs a constant added to its own shifts. The equality
`T(d + D) x = T(d) [T(D) x]` is exact only for whole-pixel `D`, because the model shifts by
linear interpolation and a fractional shift blurs as well as translates.

| common shift applied to one component | change in log-likelihood |
|---|---|
| 1.00 model pixel | **4e-9 relative** (boundary effects only) |
| 0.10 model pixel | -7.3 nats |
| 0.01 model pixel | -0.11 nats |

An uncentered table's absolute level is therefore set by interpolation error, not by the
data. It would resemble a systemic velocity, change when the grid is resampled, and contain
no information. albireo centers the pixel shifts per component, which makes the likelihood
exactly invariant:

| offset added to one component (relativistic addition) | change in log-likelihood |
|---|---|
| 5 / 50 / 200 km/s | **0.000e+00**, exactly |
| 0.5 km/s | -9.3e-10 (relative 9.8e-14, float64 round-off) |

Centering in velocity space is correct only to `O(v^2/c^2)` and leaves a residual four to
six orders of magnitude larger (-9.9e-8 at 0.5 km/s, +8.7e-6 at 50 km/s, measured in the
test suite).

### Recovery, by starting point

Each fit ran 250 L-BFGS steps of ML-II MAP over the 20 velocities and the four
hyperparameters.

| start | per-epoch RV rms [km/s] | Wilson slope | potential |
|---|---|---|---|
| the injected Keplerian | 0.098 / 0.066 | -1.8255 | -9627.1 |
| K wrong by 10% | 0.106 / 0.063 | -1.8300 | -9627.6 |
| **K wrong by 30%** | **0.098 / 0.066** | **-1.8255** | **-9627.1** |
| injected velocities + 15 km/s noise | 0.098 / 0.066 | -1.8255 | -9627.1 |
| cold start (every epoch at 0) | **12.94 / 28.70** | **+0.5916** | **+112,692** |

The injected slope is -1.8333 = -K2/K1, so the recovered mass ratio is 0.4% off. An rms of
0.098 km/s is 1/60th of a model pixel. Every warm start reaches the same optimum to four
decimals, including one 30% wrong in both semi-amplitudes.

The cold start fails: with every epoch at one velocity the two components are
indistinguishable, and the mode is documented as needing a warm start. The failure is
detectable, at 122,000 nats worse with a Wilson slope of the wrong sign.

### Uncertainties and the zero-point projection

| | mean sigma [km/s] | rms error [km/s] | error / sigma |
|---|---|---|---|
| raw Laplace diagonal | **37.95** | 0.098 / 0.066 | 0.002-0.26 |
| zero points projected out | **0.059** | 0.098 / 0.066 | 1.44 |

The raw value is `120/sqrt(10)`, the `Normal(0, 120)` prior divided by the epoch count,
identical to four digits for both components and all ten epochs. This is characteristic of
a flat direction, whose posterior width is the prior width and which every epoch's marginal
variance inherits. The value is 640x too large and would be the same for a dataset that
constrained nothing. `relative_velocity_errors` projects out each component's mean. The
projected block then has exactly 2 zero eigenvalues, one per component, which confirms the
identifiability claim numerically.

The projected error bars are ~1.4x too small against the realized errors, as expected of a
Laplace approximation with the hyperparameters fixed at their MAP values. Posterior samples
of the `velocity_rel` deterministic need no projection and no Gaussian assumption and are
the recommended route. The Laplace value is the fast estimate.

Per-epoch precision of 0.059 km/s is 1/102 of a model pixel.

### The model check

`keplerian_residuals` centers both tables in the same way and differences them in pixel
space, so the two arbitrary zero points cancel exactly. Offsetting the recovered table by
+77 and -31 km/s changes the residuals by < 1e-9.

| Keplerian tested against the recovered table | max abs residual | in units of the per-epoch sigma |
|---|---|---|
| the orbit that generated the data | 0.164 km/s | 2.8 |
| period wrong by 0.5% | 2.979 km/s | 50 |
| K_2 wrong by 5% | 2.581 km/s | 44 |

A Keplerian is a strong constraint, and a table fitted without one indicates whether the
data support it.

---

## Second speedup pass: the assembly's reverse pass and four rejected optimizations (2026-08-15)

The harness is that of the scale and speedup ladders above, on [the machine](#the-machine),
measured at 66.8 GB/s streaming (triad) and 1188 GFLOP/s fp64 `dgemm` at n = 2000. The
earlier tables ran under an unrecorded software stack, so absolute numbers are not
comparable with them. Every before/after pair below was measured back to back and is valid.

### Evaluation cost by stage

The first speedup pass attributed 92% of an evaluation to probing and then removed probing
from the hot path, so that attribution no longer applies. The jitted stages were re-measured
at the ladder's first row (31,734 model px, SB2, 50 epochs, p = 513):

| stage | s | % |
|---|---|---|
| band assembly (incl. band to block packing) | 1.94 | 65.2 |
| block Cholesky | 0.93 | 31.4 |
| solves + quadratic form | 0.075 | 2.5 |
| `A^T W z` | 0.025 | 0.8 |
| log-determinants, weighted data terms | 0.003 | 0.1 |
| **whole marginal (jitted)** | **2.97** | |
| **gradient in the velocities** | **10.23** | 3.45x eval |

Within the assembly, 88% is the epoch band scan, divided almost equally between the
velocity-independent `G = K^T H K` pre-pass (0.80 s) and the T-sandwich accumulation
(0.76 s).

### Gradient cost by stage

NUTS computes the gradient ~2,600 times per posterior and the evaluation once, so the
gradient sets the wall-clock time. By stage, it divides as follows:

| stage | value | gradient | backward only | ratio |
|---|---|---|---|---|
| band assembly | 1.91 | **8.27** | 6.37 | 4.34x |
| solve stage (custom VJP) | 0.98 | 1.68 | 0.70 | 1.71x |
| full marginal | 3.01 | 10.15 | 7.15 | 3.38x |

The assembly is 82% of the gradient, and its backward pass takes 3.3x as long as its
forward pass. The solve stage, for which the custom VJP was written, is 16%. Of the
assembly's 6.37 s backward pass, 5.94 s is in the epoch band scan and 0.20 s in the
band-to-block packing.

### Change 1: closed-form reverse rule for the band accumulate

Each epoch adds its (i, j) block to the global band tensor as

    band_out = dus(band, ds(band, idx) + f, idx)

which is `band + place(f, idx)`, the identity in `band`. Reverse mode transposes the
`dynamic_update_slice` and the `dynamic_slice` separately and rebuilds that identity as

    band_bar = dus(out_bar, 0, idx) + dus(zeros_like(band), ds(out_bar, idx), idx)

which is algebraically `out_bar` but takes three passes over the whole 522 MB band tensor
for each (i, j) block of each epoch. This is 4 x 50 x 3 x 522 MB = 313 GB of memory
traffic to reproduce an input, and it scales with the size of the band tensor, not with
that of the slice updated. `assembly._band_accumulate` is a `custom_vjp` whose reverse rule
is the closed form: the operand's cotangent is the output cotangent, and that of `f` is the
slice of it. An isolated harness at this scale gives, with values and gradients
bit-identical:

| | forward | gradient | backward |
|---|---|---|---|
| nested dynamic slices | 0.390 s | 5.615 s | 5.224 s |
| closed-form `custom_vjp` | 0.370 s | **0.840 s** | **0.470 s** |

A forward-only profile does not show this, because the ablation has opposite effects on the
two passes: deleting the band accumulate makes the forward pass 24% slower (0.98 s against
0.79 s) and the backward pass 2.7x faster.

### Change 2: the second kernel application of `G` as one contraction

`G = K^T H K` was two unrolled accumulations over the 2r+1 kernel taps. The first shifts
rows, so it remains a loop. The second does not: it adds `kernel[s] * u` at column offset
`2r - s`, which makes it a contraction against a static `(w_u, w_g)` banded matrix,
`Q[k2, k] = kernel[k2 + 2r - k]`. The loop form read and rewrote the widest image in the
assembly once per tap:

| stage | s |
|---|---|
| first (row-shifting) loop | 0.327 |
| second loop, as written | 0.548 |
| second stage as one contraction | **0.030** |

The contraction is 18x faster on that stage and 2.45x on the whole `G` pre-pass, with no
extra memory. The two applications also compose into a single `((2r+1) w_h, w_g)` map, so
both can be one contraction. That needs a 37 MB neighbourhood stack per epoch, or 1.9 GB
across a hoisted 50-epoch pre-pass (the kind of vmapped intermediate the memory pass
removed), and measured 0.32 s against 0.36 s. Ten percent for 1.9 GB was not adopted.

### Combined effect of the two changes

The full design-target ladder was run with `scripts/m5_scale_bench.py` on the same seeds and
machine. The "before" column is a re-run of the committed code (changes stashed), not a copy
of the earlier tables:

| n (model px) | eval before | eval after | | ∇ before | ∇ after | |
|---|---|---|---|---|---|---|
| 31,734 | 2.71 s | **2.16 s** | 1.25x | 10.07 s | **5.52 s** | **1.82x** |
| 74,322 | 6.45 s | **5.24 s** | 1.23x | 27.98 s | **15.25 s** | **1.83x** |
| 135,052 | 12.64 s | **10.15 s** | 1.25x | 53.02 s | **29.09 s** | **1.82x** |
| **203,440 (design target)** | 19.11 s | **15.67 s** | 1.22x | 80.23 s | **45.88 s** | **1.75x** |

The ratios are independent of problem size, 1.22–1.25x on an evaluation and 1.75–1.83x on a
gradient, as both changes predict: each removes a fixed multiple of the per-epoch band
traffic, which is linear in `n`. A design-target gradient takes 46 s, and the
gradient/evaluation ratio falls from 4.2x to 2.9x.

Stage by stage at row 0:

| | before | after | |
|---|---|---|---|
| band assembly | 1.94 s | 1.38 s | 1.41x |
| block Cholesky | 0.93 s | 0.72 s | — (unchanged; run-to-run spread) |
| whole marginal evaluation | 2.97 s | 2.20 s | 1.35x |
| gradient in the velocities | 10.23 s | 5.19 s | 1.97x |

Repeat runs of the whole-marginal figure vary by about 10% on this machine, so the ladder
above is the figure to quote. The gradient ratio is stable.

### Exactness

The log-likelihood and its gradient are bit-identical before and after, compared as raw
IEEE-754 hex rather than to a tolerance, in all three model variants: stationary LSF,
AR(1) correlated noise, and wavelength-dependent LSF. The Hessian changes by 4e-13
relative, which is due to reassociation in the second-order path.

Bit-identity was measured. It is not guaranteed. Like the rest of the band assembly, the
contraction of change 2 equals the loop only up to summation order: increasing `k2` is
increasing `s`, so the two ideal orders coincide, but XLA may block a GEMM's accumulation in
any order. Against a random kernel it differs from the loop by 0.5 ulp. The code comments
make the weaker claim.

The earlier second-order re-entry defect recurred and its regression test detected it. A
`custom_vjp` whose forward rule calls the custom function itself is first-order exact but
returns the transpose of the true Hessian, and
`test_second_order_reverse_matches_plain_autodiff` failed on the first version. The
correction is the same: the primal is inlined in the forward rule.

### Partial recheck of the three-code comparison

On the fd3 benchmark's configuration (4,444 px, 20 epochs, `b_nat` = 63) this pass reduces
albireo's steady-state time from 0.071 s to 0.059 s, or 1.20x, consistent with the ladder.
This is the only part of the comparison measured with a single change.

The comparison now differs from the recorded one, mostly for reasons other than this pass:

| | recorded earlier | this machine |
|---|---|---|
| albireo, committed code | 0.182 s | 0.071 s |
| albireo, with this pass | — | **0.059 s** |
| fd3 (rebuilt binary, WSL, min of five) | 0.111 s | 0.099 s |
| shift-and-add, 7 sweeps | 0.018 s | 0.049 s (**does not reproduce**) |

fd3 changed by 12% and albireo by 2.6x between the two measurements, with no hardware change
(see [The machine](#the-machine)). fd3 is a single-threaded C program and albireo's XLA uses
all 32 threads. In this measurement albireo is ~1.7x faster than fd3. The earlier record has
it 1.64x slower.

The three-way table above was not updated from this partial recheck. The shift-and-add
figure did not reproduce (0.049 s against 0.018 s recorded, while every other figure was
faster or unchanged), and a fair comparison needs all three codes re-run end to end under
one protocol. Accuracy is unaffected: the recovered spectra reproduce the recorded RMS
exactly (0.0093 / 0.0116 mean-aligned).

The re-run at the end of this file did that. It reproduced all twelve accuracy values
exactly and attributed the 0.049 s to the harness's in-process convention, which timed
shift-and-add on a heap the XLA solve had just used and is shared by both recorded numbers.
It also found fd3's OpenBLAS running 32 threads (pinned to one, fd3 is 1.7× faster) and
changed the harness to fresh-process timing.

### Peak memory

A speedup must not undo the memory pass. Peak buffer-assignment bytes, from XLA
`memory_analysis()` as in that pass, are:

| n (model px) | eval, then | eval, now | ∇, then | ∇, now |
|---|---|---|---|---|
| 31,734 | 2.94 GB | 2.96 GB | 4.00 GB | 4.02 GB |
| 74,322 | 4.86 GB | 4.76 GB | 11.47 GB | **10.26 GB** |
| 135,052 | 7.83 GB | 7.81 GB | 14.37 GB | **12.84 GB** |
| **203,440** | 11.06 GB | 11.14 GB | 18.24 GB | 18.33 GB |

Peak memory is unchanged at the design target (+0.5%, the row that has to fit in 32 GB) and
8-11% lower in the middle rows, where the closed-form reverse rule no longer materializes
whole-tensor temporaries.

### Removal of the forward-mode path

`custom_vjp` does not support `jax.jvp`, and `forecast._effective_parameters` was the only
place in albireo that used forward mode. The forecast obtains `p_eff = tr[Sigma A^T W A]`
from one directional derivative of `log det` in the noise scale, because `with_jitter` is
already that one-parameter family. The full suite showed eleven failures in
`test_forecast.py` and two in `test_plotting.py`, all `TypeError: can't apply forward-mode
autodiff (jvp) to a custom_vjp function`.

Both `t` and the log-determinant are scalars, so forward and reverse mode compute the
same single number, which is now obtained with `jax.grad`:

| | committed code | with this pass |
|---|---|---|
| forward (`jax.jvp`) | 0.283 s | **rejected** |
| reverse (`jax.grad`) | 0.773 s | 0.532 s |
| `p_eff` | 3979.897960 | 3979.897960 |

The values are bit-identical (absolute difference exactly 0), and
`test_p_eff_matches_dense_trace` tests the value against a dense trace oracle at rel 1e-8
on either route. The cost is 0.283 s → 0.532 s on a call made once per forecast, against a
gain of 1.8x on a gradient evaluated ~2,600 times per posterior.

Forward mode through the marginal likelihood is now unavailable. The first speedup pass had
already removed it one stage later (`_solve_stage` is a `custom_vjp`, which is why
`laplace_inverse_mass` uses reverse-over-reverse). Second derivatives remain available, and
tested, through reverse-over-reverse.

### Rejected optimizations

Four candidates looked correct in analysis and were rejected on measurement. They bound the
remaining speedup.

1. **A blocked Cholesky.** The block factorization is 31% of an evaluation, and XLA's
   fp64 dense `cholesky` is far slower per flop than its `matmul` at the block size the
   solver uses:

   | n | `matmul` | `cholesky` | `solve_triangular` |
   |---|---|---|---|
   | 256 | 183 GF/s | 7 GF/s | 41 GF/s |
   | **513** | **249** | **13** | **103** |
   | 1026 | 390 | 26 | 164 |
   | 2048 | 921 | 69 | 281 |

   A recursive blocked factorization built from trsm + gemm was expected to be faster by a
   large factor. It gained 1.36x (2.39 ms against 3.26 ms at n = 513, best inner block
   128), because neither trsm nor the leaf factorizations parallelize at that size.
   Larger blocks are worse: the cost is `O(n B^2)`, so doubling `B` takes 4x the flops for
   about 2x the rate. The block Cholesky is at its practical limit on this stack, and is
   now the largest single item in an evaluation.
2. **j-factoring the T-sandwich.** `f_ij = sum_ab w_i[a] w_j[b] S[i][a][1+b-a]` equals
   `A_i + frac_j * B_i`, which builds the tent slices once per component instead of
   once per (i, j) pair, 16 slab operations reduced to 10. It measured slower (0.81 s
   against 0.79 s; results agree to 5e-16). XLA already fuses the four terms into one
   pass, so the restructured form only adds a materialized intermediate.
3. **`remat=False`.** The memory pass chose rematerialization of the epoch body, which is
   also faster: 7.64 s against 9.81 s for the epoch scan's gradient. The memory and time
   choices coincide.
4. **A custom-VJP band-to-block packing**, from the earlier list of remaining options.
   The effect exists (`_pack_band`'s gather transposes to a scatter), but it is 0.20 s of
   a 10 s gradient. It was not implemented, by decision.

One more candidate was rejected by arithmetic before implementation: laying out the band
tensor as `(nc, nc, n_pix, n_k)` so that each epoch's slice is contiguous rather than
strided by `nc`. The band read-modify-write is only ~0.15 s of the 0.76 s T-sandwich, and
removing it made the forward pass slower: the traffic is in building `f`, not in storing it.

## Re-run of the three-code comparison on one machine (2026-08-16)

All three codes were run under one protocol on one machine. The two anomalous wall-clock
times are properties of the environment, not of the codes.

Stack: AMD Ryzen 9 9950X3D (16 cores / 32 threads), 31.1 GiB, Windows 11 Pro build 26200,
WSL2 kernel 6.18.33.2 for fd3; Python 3.13.9, jax 0.11.0, numpy 2.5.2. The original tables
record no stack, so the earlier 0.018 s could not be tested.

### Accuracy: reproduction of the twelve recorded values

Each code ran once before any timing. All twelve recorded RMS values reproduce to the
printed precision (albireo 0.0118 / 0.0093 and 0.0165 / 0.0116, fd3 0.1767 / 0.0198 and
0.2597 / 0.0223, shift-and-add 0.0317 / 0.0248 and 0.0849 / 0.0302). The exported fd3
inputs are checksum-identical to those of the earlier session, and fd3's deterministic
output `.mod` is byte-identical. The computed spectra are unchanged. Only the timing
protocol differs.

### Wall-clock times under one protocol

Each code had two to three warmups and nine recorded repeats, strictly sequential, with
nothing else running:

| | recorded earlier | min | median | convention |
|---|---|---|---|---|
| shift-and-add, 7 sweeps | 0.018 s | **0.0263 s** | 0.0267 s | fresh process, jax never imported |
| albireo | 0.182 s | **0.0591 s** | 0.0625 s | jitted steady state; cold compile + first call 0.495 s |
| fd3, `OMP_NUM_THREADS=1` | — | **0.0636 s** | 0.0640 s | full WSL process; see below |
| fd3, environment as found | 0.111 s | 0.1042 s | 0.1113 s | full WSL process |

Shift-and-add is fastest, albireo and single-threaded fd3 are at parity (minima 0.0591
against 0.0636, medians 0.0625 against 0.0640), and fd3 as found is slowest. The earlier
statement that albireo is slower than both described one measurement. The lasting
conclusions are that shift-and-add is fastest in every measurement, as expected of a small
number of array shifts and means, and that albireo's 32-thread XLA graph offsets fd3's
single-thread advantage on current hardware. The partial recheck of the speedup pass
reproduces: its 0.059 s is this table's 0.0591, its fd3 0.099 s is inside a later control
series (minima 0.092–0.104), and its irreproducible 0.049 s is inside the contaminated
range below.

### Heap fragmentation in the in-process timing of shift-and-add

The committed harness timed shift-and-add in the same process, after the albireo solve,
the convention of both recorded numbers. In one process, the minimum time per stage was
0.0265 s with numpy only, 0.0267 s after `import jax`, 0.0267 s after backend init,
0.0275 s after a small jit and 0.0632–0.0736 s after the large jitted solve. It did not
recover: 0.0729 s after `clear_caches()` plus gc, 0.0787 s after three seconds idle. The
inner `_shift` slowed from 45.9 to 125.8 µs. Neither threads nor the CPU is the cause: the
code is pure NumPy, thread pinning has no effect, and at every stage user time ≈ wall-clock
time with sys = 0 and zero page faults. The same thread therefore runs the same
instructions ~2.7× slower, but only when it allocates.

Allocating ufuncs slow ~4× (`np.floor(a)` 0.82 → 3.46 µs, `a + b` 0.88 → 3.42 µs), while the
same calls with `out=` do not (0.62 → 0.64, 0.76 → 0.77 µs). The slowdown has the size
dependence of the Windows CRT heap. Requests of 8 KB (low-fragmentation heap) are unchanged
at 0.17 → 0.15 µs. Requests of 34.6 KB, the size of shift-and-add's full-grid temporaries,
go from 0.23 to 2.14 µs, 128 KB from 0.23 to 5.54 µs, and 800 KB from 0.24 to 9.96 µs. After
XLA's allocations, a request above 16 KB spends microseconds traversing a user-mode free
list (the same address is returned, so the traversal is the cost), and `disentangle` makes
~10⁴ such allocations per call. That is +20–45 ms, the observed range. The committed
convention reproduces it: 0.037–0.043 s here, and 0.049–0.084 s after fully jitted runs,
which contains that 0.049 s.

The earlier 0.018 s was taken under the same convention on the same machine, with an
unknown amount of contamination, so the 1.46× residual (0.0263 here against 0.018 there)
cannot be decomposed further (see [The machine](#the-machine)). It combines serial
small-array NumPy throughput on an unrecorded stack with contamination of a size that cannot
be reconstructed, and contamination only slows a run. The 0.018 s is unreproduced.
`scripts/fd3_bench.py` now times shift-and-add in a fresh interpreter (warmup, then min of
five), the convention of the table above, whose shift-and-add row is its first number under
a recorded stack.

### fd3 timing and the OpenBLAS thread count

fd3 is single-threaded C and was used as the control, on the expectation that its timing
would not depend on thread pinning. It does depend on it, because the binary as built links
GSL against OpenBLAS. In the default environment it runs at 1938% CPU, with 1.95 s of user
time in a 0.10 s wall-clock run and 32 threads active. With `OMP_NUM_THREADS=1` it runs at
93% CPU and takes 0.0636 s, 1.7× faster. On a many-core machine the OpenBLAS thread pool
increases fd3's wall-clock time by 60%. This accounts for part of the 12% change in fd3
noted above (see [The machine](#the-machine)). The pinned figure is the one to use for fd3
as an algorithm. The as-found row stays in the table because fd3 is normally run that way.

### Status of the earlier tables

The earlier tables remain a record of this same desktop, with the software stack and the
timing contamination unrecorded. The accuracy result is confirmed a second time by exact
replication: ~2× on shape, the same *k* = 0 null space in all three codes, and a posterior
from one of them.

## Feature comparison with the established shift-and-add repository (2026-08-27)

The clean-room `scripts/shift_and_add.py` implements only the recurrence of González &
Levato, so the three-way table compares methods, not codebases. This section compares
the repository most widely used in the field,
[`TomerShenar/Disentangling_Shift_And_Add`](https://github.com/TomerShenar/Disentangling_Shift_And_Add),
as software: what it provides, how it is distributed, and where it differs from albireo in
kind rather than in degree.

Provenance: everything below comes from the repository's README and the GitHub API. The
source files were never opened. The repository has no license (`"license": null` from the
API, checked 2026-08-27), so reading it would contaminate the one implementation of this
algorithm that albireo can legally maintain. Anyone who does open it should not afterwards
edit `scripts/shift_and_add.py`. Where the rule limits what this page can claim, the limit
is stated.

The repository is a set of Python research scripts, configured by editing
`Input_disentangle.py` and run as `python disentangle_shift_and_add.py`, with the core in
`Disentangling/disentangle_functions.py`. Besides the recurrence it provides the other
parts of a working research tool: a χ² grid over the semi-amplitudes (K₁ and K₂, and K₃,
since it supports triples) given (P, T0, e, ω) from another source, and a negativity
constraint against spurious emission features. It also has multi-instrument input in ASCII
or FITS, mock-data generators for SB2s and SB3s, and plotting utilities. V2.0 is dated
September 2023 and the last commit 2024-03-07. At the check date it had 10 stars, 2 forks
and 1 open issue. It cites González & Levato (2006) for the algorithm and asks users to
cite Shenar et al. 2020 (A&A 639, A6) and 2022 (A&A 665, A148), both already in
`paper/paper.bib`. It is the code behind the LB-1 and HR 6819 companion identifications.

The numbers above cover the shared core only. The three-way table measured the published
recurrence under the paper's own stopping rule with the orbit fixed at the injected values,
so its RMS and wall-clock values apply to the repository's algorithm, not to its code. The
negativity constraint and the χ² grid are built on that core, were never run here, and
under the provenance rule cannot be reimplemented from the source. A direct comparison with
the repository as distributed would be legitimate (running unlicensed code is permitted;
deriving from it is not) and has not been made.

| | `Disentangling_Shift_And_Add` | albireo |
|---|---|---|
| orbit | χ² grid over (K₁, K₂[, K₃]); P, T0, e, ω supplied | joint inference of every orbital site, gradient-based, with NUTS |
| uncertainty on K | χ² map over the grid | posterior draws |
| uncertainty on spectra | none | pointwise band, and draws from the joint posterior |
| the *k* = 0 null space | suppressed by the negativity constraint | fixed by the smoothness prior; reported as the leading mode by the forecast |
| contaminated pixels | masking/weights (in the published method) | the same masking, plus telluric and nebular components when masking would discard the pixels the science needs |
| triples | yes, K₃ in the grid | yes at the expert level (hierarchical outer orbit); the `Disentangler` interface rejects it in v1 |
| validation | mock-data generators | closed-loop acceptance tests in CI against packaged injected values, plus calibrated detection |
| distribution | scripts + a config file; no license, no package, no tests | BSD-3-Clause package on PyPI, CI, docs, tutorials |

The differences in degree are those measured above: about 2× on aligned shape, and the
shortest wall-clock time in the comparison, which belongs to the repository's algorithm, as
expected of a small number of shifts and means. The difference in kind is the column of the
three-way table that no choice of test conditions equalizes: an uncertainty on the
disentangled spectra, which no code in
[the survey of methods](science.md#2-methods-of-spectral-disentangling) produces. The
practical difference is the license. The repository cannot be vendored, forked, or legally
built upon, which is why the clean-room implementation exists and is a barrier for anyone
extending the method. This is not a criticism of its authors, for whom disentangling is a
means to an end and the software a by-product of the science.

As noted above, weights and rejection are part of the published method, so masking is not
an albireo advantage. The difference arises when masking would discard the pixels the
science needs, such as a nebular line within Hβ, where albireo models the contaminant
instead. Unmodelled, it changes K₂ by −59% and gives e = 0.95 for a circular orbit, so the
choice affects the masses and not only the atmospheres.

BLOeM, the survey used in [the BLOeM tutorial](tutorials/bloem-sb2.md), is led by the
repository's author, and its 59 published SB2s have no orbital solutions, the case
`Disentangler(velocities=...)` was written for. The two codes serve successive stages and
are not rivals: the repository's shift-and-add found several of those systems, and albireo
addresses what follows, namely the orbit, the spectra, and the error bars on both.


## Stellar labels from disentangled components (2026-08-27)

Machine: [the desktop](#the-machine), CPU only, float64; comparable with the second
speedup pass and the re-run, not with the earlier tables (unrecorded stack). Harness:
`scripts/label_bench.py`, offline and reproducible. The grid is a toy at BOSZ's own node
density (250 K in Teff, 0.5 dex in log g, 0.25 dex in [M/H]; 455 nodes x 2000 px), so that
the interpolation numbers can be compared with the published ones.

### Interpolation error and the case for an emulator

The leave-out error at doubled node spacing, in fractional normalized flux, is a pessimistic
proxy, since the real fit interpolates on the full grid:

| method | rms | p95 | max | n tested |
|---|---|---|---|---|
| multilinear | 3.88e-04 | 1.74e-06 | 8.16e-03 | 371 |
| Catmull-Rom cubic | **1.84e-04** | 2.00e-07 | 4.51e-03 | 371 |

The literature values for the same spacing on a real ATLAS9 grid (Meszaros & Allende Prieto
2013) are linear 5.1e-04, cubic-Bezier 3.1e-04, and a Payne-style network about 1e-03.

1. The cubic is worth its 4^k taps: 2.1x better than multilinear here, 1.6x in the
   published comparison.
2. On a grid at this density a learned emulator would be worse by roughly a factor of
   five, so none is built for FGK. This does not apply to the coarse, strongly non-linear
   OB grids, where the same measurement has to be repeated before an emulator is built or
   ruled out. `crossval_library` is that measurement and is part of the package.

Node reproduction on this box grid is bit-for-bit (`==`, not a tolerance), so the
warm-start node scan and the continuous fit can be compared on the same basis. The simplex
path used for incomplete grids reproduces a node only to rounding (its weights at a vertex
are 1 − ε and ε), and which nodes are reproduced exactly depends on the triangulation Qhull
chose. The test suite asserts the rounding bound under permuted node orders, so that the
comparison holds on every scipy build.

### Closed-loop recovery

Two components were injected at off-node labels and fitted with wrong assumed light
fractions (0.72/0.28 assumed against 0.62/0.38 injected), with noise at the declared level:

| S/N | star | dTeff [K] | dlog g | d[M/H] | dv sin i [km/s] | formal sigma(Teff) [K] | fitted light fraction |
|---|---|---|---|---|---|---|---|
| 100 | A | +17.9 | +0.0010 | +0.0080 | -0.887 | 11.05 | 0.621 |
| 100 | B | +12.6 | -0.0125 | +0.0080 | +0.195 | 7.68 | 0.379 |
| 250 | A | +7.2 | +0.0003 | +0.0033 | -0.333 | 4.36 | 0.620 |
| 250 | B | +5.0 | -0.0048 | +0.0033 | +0.077 | 3.09 | 0.380 |
| 500 | A | +3.7 | +0.0004 | +0.0017 | -0.169 | 2.13 | 0.620 |
| 500 | B | +2.3 | -0.0016 | +0.0017 | +0.033 | 1.57 | 0.380 |

The worst row is 0.35% in Teff against a target of 2-3% (math.md 9.6), 0.013 dex in log g
against 0.15, and 8% in v sin i against 10%. For comparison, GSSP's own simulation recovery
at S/N 150 is +-40 K and +-0.06 dex, so this is within the accuracy the mode requires, on a
toy grid (see the scope note below).

The main result is the light ratio. It is recovered as 0.621/0.379 against injected values
of 0.62/0.38, from an assumed 0.72/0.28: the joint radius-ratio fit absorbed a 16% error in
the assumed dilution in the dilution parameter and not in the temperatures. It is compared
with `FixedDilution` on the same data, where the example asserts that the frozen fit is
never closer to the injected values.

Chi-square is 1835.8 for 2020 pixels at all three signal-to-noise levels, as expected. The
quoted sigma matches the injected noise and the seed is fixed, so residual/sigma is the
same array and the reduced chi-square is 0.909 at every noise scale. The null chi-squares
change as expected: the nearest-node null is 7.6e3 / 3.8e4 / 1.5e5 and the no-template null
1.7e5 / 1.1e6 / 4.2e6, both in units of the decreasing sigma.

### Wall clock

| stage | seconds |
|---|---|
| resample the library onto the model grid | 0.05 |
| build the interpolator | 0.00 |
| `match_labels`: node scan + 4 x L-BFGS + Laplace | 27.8 |
| refit 8 posterior draws | 41.1 |

The library is 455 nodes x 2000 px, projected onto 1010 model pixels. The draws refit is the
more expensive stage, scales linearly with the draw count, and is optional.

### Formal errors and the spread over refitted draws

| star | label | formal | draws | ratio |
|---|---|---|---|---|
| A | Teff | 4.36 | 17.60 | 4.0x |
| A | v sin i | 0.182 | 0.339 | 1.9x |
| B | Teff | 3.09 | 4.33 | 1.4x |
| B | v sin i | 0.087 | 0.556 | 6.4x |
| A | log g | 0.007 | 0.007 | 0.9x |
| B | log g | 0.005 | 0.003 | 0.5x |

This run establishes that the propagation path works end to end and that the two numbers
are reported side by side with their ratio. The ratios in the table do not represent real
disentangled spectra. The harness has no disentangling, so the draws are the data plus new
white noise and have none of the correlated structure that makes the formal error too
small. The spread measures label-space non-linearity alone, and the scatter across rows
(0.5x to 6.4x) is partly the sampling error of a standard deviation over eight draws, about
27%. The literature's 5-10x (Gebruers et al. 2022: 70 K formal against 425 K realistic;
Czekala et al. 2015) is for joint posterior draws of real disentangled spectra, which
contain the low-k exchange modes. Reproducing it is left to the AI Phe validation run.

### Scope of these numbers

Every figure above is on a toy grid whose spectra are analytic Gaussian lines, with each
label controlling its own set so that the label-to-spectrum map is invertible. (On a grid
where Teff and [M/H] both scale one depth, chi-square reached 1e-26 without recovering the
injected labels.) Real grids have blends, saturated cores and a continuum that is not a
smooth exponential in Teff, so the recovery figures here are an upper bound. The acceptance
test on real data is AI Phe against Maxted et al. (2020), not run here.


## AI Phoenicis: the label fit on observed spectra (2026-08-27)

Machine: the same desktop. Harness: `scripts/aiphe_labels_bench.py` on 36 archival HARPS
spectra (R = 115,000, `scripts/download_aiphe.py`), disentangled on 5150-5250 A with the
velocities held at the published orbit, so that the run tests the label fit and not the
orbit. Library: `bosz2024-fgk-r20000`, 454 nodes. The notebook of the same run is
`docs/tutorials/aiphe-labels.ipynb`.

The velocities were held at the rounded period, 24.5924 d. `aiphe_bench.py` now uses
24.592483 d, which changes the fixed velocities by up to 0.107 km/s at the steepest phases.
The figures in this section predate that change and should be re-run.

Every quantity the mode produces has an independent published value for AI Phe: Teff
6310 K and 5010 K, log g 4.001 and 3.598, R2/R1 = 1.6237 (Maxted et al. 2020, run C). The
log g values are derived from the spectroscopic and photometric elements rather than
quoted, via `g_i = 2 pi sqrt(1-e^2) K_j / (P r_i^2 sin i)`. This expression needs no
absolute masses and reproduces the published values to 0.002 dex, which the script asserts.

### Result

| configuration | primary Teff | secondary Teff | R2/R1 | chi2 |
|---|---|---|---|---|
| log g declared, dilution fitted | **6342.5 K (+0.52%)** | **5227.0 K (+4.33%)** | 1.5416 (-5.1%) | 425,869 |
| log g free | 6019.7 K (-4.60%) | 4900.3 K (-2.19%) | 1.5459 | 412,882 |
| log g declared, dilution frozen | 6449.4 K (+2.21%) | 5179.8 K (+3.39%) | n/a | 1,853,578 |

The primary is recovered to 0.52 per cent, within the 2-3 per cent that math.md 9.6 gives
as sufficient for template selection. The secondary is not, at +217 K or 4.3 per cent. The
radius ratio, which is not given to the fit, is 5 per cent low from spectroscopy alone
against a photometric measurement.

Freeing log g reproduces on real data the failure described in the tutorial: log g goes to
the lower limit of its prior (3.000, the grid edge) and both temperatures decrease with it,
while chi-square improves. The correlation report is empty, because a parameter at a bound
does not vary and the curvature at the optimum then does not show the degeneracy that
produced the result. A flagged correlation is evidence of a degeneracy, but an empty report
is not evidence of its absence.

### Change of the default comparison mode from matched to native

`compare="matched"` convolves both the model and the data with the declared LSF before
comparing them, on the argument that `d_hat` is a regularized partial deconvolution. It was
the original default. On AI Phe it placed both components at the lower limit of their
`v sin i` prior (0.14 and 0.46 km/s) and gave a larger chi-square than `native`:

| mode | primary Teff | secondary Teff | v sin i | chi2 |
|---|---|---|---|---|
| `matched` | 6319.5 K (+0.15%) | 5280.1 K (+5.39%) | 0.14 / 0.46 km/s | 1,813,881 |
| `native` | 6342.5 K (+0.52%) | 5227.0 K (+4.33%) | 2.23 / 2.21 km/s | 425,869 |

Convolving the residuals correlates them over the kernel width while the likelihood remains
diagonal, so chi-square is over-counted by the effective-sample-size factor `1/sum(k^2)`.
At sigma_LSF = 1.38 px that predicts 4.91 and the fit measured 4.26, the whole difference
between the two modes. The mis-specified likelihood does not only inflate chi-square:
`v sin i` absorbs it. The default is now `native`. `matched` is appropriate only once a
residual-covariance model represents the correlation it creates.

The closed-loop test cannot detect this. Its injected rows do not pass through an LSF or a
disentangling, so they are intrinsic spectra and both modes recover them (matched is better
at S/N 1000: -0.1 K against -4.6 K). Only real data that have been through a deconvolution
separate the two modes.

### Microturbulence as the cause of the secondary offset (negative result)

The first candidate cause of the secondary's offset was the library's fixed
microturbulence. BOSZ provides xi in {0, 1, 2, 4} km/s and the registry fixes it at 2. A K
subgiant requires nearer 1.3, and too much microturbulence makes the model's metal lines
too strong, which a fit can compensate by raising Teff. The sign is as required: at the
t5250/g3.5 node, xi = 2 gives 8.45 per cent more equivalent width than xi = 1, so a 160-node
library was rebuilt at xi = 1.

The secondary's offset increased (+292 K against +273 K), and chi-square with it (1.97e6
against 1.81e6). [M/H] changed instead, by +0.10 dex: the documented [M/H]-xi degeneracy
absorbed the change, as math.md 9.2 describes. The test is recorded so that it is not
repeated.

Most of the secondary's offset remains unexplained. Three candidates are untested: a 100 A
window constraining temperature far more strongly for an F star than for a K subgiant, the
published 5010 K being a photometric/SED temperature rather than a spectroscopic one, and
the assumed light fractions, which the fitted radius ratio only partly absorbs.

### Caveats on these numbers

The formal errors here are sub-kelvin and contain no information: they are the curvature of
an optimum on correlated residuals, and `summary()` states this whenever it prints them.
`refit_draws` gives the number to quote and was not run for this record. HARPS provides no
error array, so the weights are albireo's own estimate (the z-RMS 2.64 noted in the
D-numbered AI Phe disentangling entry applies here too). This is a validation on one
system, one window and one library, not a survey.

## Epoch velocities by N-dimensional correlation (2026-09-01)

Machine: the same desktop, CPU only, float64. Harness: `scripts/todcor_bench.py`, offline
and reproducible. The fixture is an SB2 simulated through the package's operator stack (LSF
sigma 5 km/s, rebin onto a 0.05 Å native grid, light 0.6/0.4, barycentric motion). The
templates are the injected component spectra on a 1 km/s grid (five pixels per LSF sigma).
Every number here describes the estimator and excludes template mismatch, which a real star
adds (see the end of this section).

### Equivalence to TODCOR

On a uniform grid with uniform weights and the data on the model grid, albireo's
weighted-least-squares surface reproduces Zucker & Mazeh's (1994) symmetric two-dimensional
correlation (light ratio maximized out), their original fixed-ratio expression, and the
least squares with the scale held fixed. Each agrees to 1e-10 with an independent NumPy
transcription of the published formulae (`tests/test_todcor.py`, the three identity tests).
The rest of this section covers data the published form does not accept.

### Precision, bias and calibration against S/N

The table uses sixteen noise realizations of eight epochs each, fixed light fractions and
profiled errors (Zucker 2003). The last column takes the declared weights at face value:

| S/N | star | bias [km/s] | scatter [km/s] | mean quoted sigma [km/s] | pull rms | pull rms (ivar errors) |
|---|---|---|---|---|---|---|
| 30 | A | -0.0233 +- 0.0142 | 0.1604 | 0.1702 | 0.957 | 0.953 |
| 30 | B | +0.0208 +- 0.0168 | 0.1901 | 0.1923 | 0.999 | 0.993 |
| 100 | A | -0.0070 +- 0.0043 | 0.0484 | 0.0510 | 0.961 | 0.958 |
| 100 | B | +0.0065 +- 0.0051 | 0.0573 | 0.0577 | 1.005 | 0.999 |
| 300 | A | -0.0023 +- 0.0014 | 0.0161 | 0.0170 | 0.960 | 0.957 |
| 300 | B | +0.0021 +- 0.0017 | 0.0192 | 0.0192 | 1.007 | 1.001 |

The quoted errors are calibrated: the pull rms is between 0.96 and 1.01 at every S/N,
reproducing Zucker's (2003) Figure 4 for the weighted, projected, sub-pixel version. The
profiled and declared-weight errors agree because the noise was injected at the declared
level. The rescaling matters only on real data. The scatter scales as 1/(S/N) from 0.16 to
0.016 km/s, a sixtieth of a pixel at S/N 300. There is a bias of a few thousandths of a
km/s, of opposite sign in the two components and independent of S/N (−0.002 / +0.002 km/s
at S/N 300, about 1.5 sigma each). It is the shift-interpolation systematic of the next
table, at the 0.002–0.006 px level this grid sampling predicts, not a property of the noise.

### Pixel locking of the shift operator against template sampling

The linear shift operator blurs a template at half-pixel shifts, which adds a
one-pixel-periodic ripple to the chi-square and moves the minimum toward integer shifts
(math.md §10.3). It was measured on noiseless data simulated at 0.25 km/s. The templates
were rebinned onto coarser grids so that the template interpolation error is real and not
an inverse crime, and the injected velocities span a whole pixel of the coarsest grid in
eighths:

| template dv [km/s] | LSF sigma [px] | estimate 0.1/sigma_px^2 [px] | max abs error [px] | max abs error [km/s] | rms error [km/s] |
|---|---|---|---|---|---|
| 0.50 | 10.0 | 0.0010 | 0.0022 | 0.0011 | 0.0007 |
| 1.00 | 5.0 | 0.0040 | 0.0058 | 0.0058 | 0.0033 |
| 2.50 | 2.0 | 0.0250 | 0.0153 | 0.0382 | 0.0173 |
| 5.00 | 1.0 | 0.1000 | 0.0293 | 0.1463 | 0.0967 |

The order-of-magnitude estimate is correct within a factor of three either way. The
documentation quotes it as an estimate and this table as the measurement. Three or more
pixels per LSF sigma keep the systematic below a hundredth of a pixel, while at one pixel
per sigma it is a tenth of a km/s, the size of a good epoch error. `Fit.templates()`
therefore upsamples to three per sigma, and `todcor` warns below two.

### One-dimensional and two-dimensional correlation as the components blend

The same spectra were correlated against the primary's template alone (the one-dimensional
CCF, `todcor` with one template) and with the two-dimensional fit, over a range of injected
separations of the two stars' lines, at S/N 200:

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
two-dimensional quoted error, and it does not vanish when the lines separate. A secondary
with 40% of the light contaminates the primary's peak at every separation, with a sign set
by which of its lines fall near the primary's (+0.6 km/s at 120 km/s). The two-dimensional
fit is unbiased throughout because the contaminant is in the model, and its error does not
depend on separation. The blending flag is correctly never set here: two different spectra
at the same velocity remain separable because their line lists differ, and the flag depends
on the covariance, not on the velocity difference. It is set for twin spectra at one
velocity (`test_twin_stars_at_the_same_velocity_are_flagged_blended...`), the degenerate
case.

### Wall clock

The runs use fixed light, one instrument and a coarse search with a stride of five template
pixels (the LSF sigma). The table gives the minimum of three after a warm-up, and the
compile:

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

A whole optical range (2000 Å at 0.05 Å, forty thousand pixels, the orders of one echelle
spectrum) over ±300 km/s takes 0.13 s per epoch on the CPU. The compile takes a quarter of
a second per distinct (instrument, pixel count) shape. A BLOeM-sized survey (929 stars × 25
epochs of ~2000 pixels) takes minutes rather than hours. The cost is dominated by the pair
Gram matrix, one matrix product of size pixels × shifts², an operation suited to a GPU when
the window and the search range are both large. The example's twelve epochs of ten thousand
pixels take 0.6 s including everything but the compile.

### Three components

| star | rms error [km/s] | mean quoted sigma [km/s] | pull rms | recovered light |
|---|---|---|---|---|
| A | 0.0251 | 0.0313 | 0.816 | 0.500 |
| B | 0.0413 | 0.0389 | 1.098 | 0.300 |
| C | 0.0645 | 0.0649 | 0.962 | 0.200 |

The light fractions 0.5/0.3/0.2 were fitted freely at S/N 200 over ±80 km/s, and four
epochs took 1.3 s including the compile. The 20% component, a TRICOR tertiary, is recovered
to 0.06 km/s with a calibrated error, and its light fraction to three decimals. The search
grid grows as the cube of the shift count, so the range was kept narrow. Beyond three
components the docstring recommends narrowing `v_range` and coarsening `coarse_step`.

### Scope of these numbers

These numbers measure the estimator against its own templates. A real star adds template
mismatch (the wrong temperature, gravity or rotation, or a disentangled component's own
noise and null-space contamination, §5.1), which is outside the quoted error. According to
the literature cited in `docs/tutorials/labels.md` (Posbic et al. 2012), its main effect is
a constant zero point per component rather than a loss of precision. The one check here
that resembles real data is the self-consistent loop in `examples/12_todcor.py`. Against
the components recovered by a MAP disentangling of the packaged example, the velocities are
recovered to 0.13 and 0.10 km/s rms once each component's zero point is removed. The
Keplerian fitted to them returns *K* to 0.05%, and the reduced chi-square of the correlation
is 1.005. AI Phoenicis, where every element has a published value and the templates would
be the label fits above, is the next acceptance test and has not been run.

## The pipeline in worker processes (2026-09-01)

Machine: the same desktop, CPU only, float64. Harness: `scripts/pipeline_bench.py`,
offline. The batch is eight simulated stars (the pipeline's own toy star, a two-component
SB2 drawn from `albireo.simulate.synthetic_library` at known labels, 8 epochs of 725 native
pixels each, S/N 120), identical except for the noise seed. The label stage and the figures
are off, so the timing covers the stages every star requires: the conjunction scan and
60 L-BFGS steps of disentangling on an 893-pixel grid, the velocity table against the fit's
own components, the orbit fit, and the products. One warm-up star is run in-process first,
so that the compile precedes the in-process timing. Every worker also compiles once. Another
project's test suite was running during the first four rows and had finished before the last
two. The capped eight-worker point was measured in both states and agreed to 0.1 s.

| workers | threads per worker | batch wall-clock time [s] | mean per star [s] | speedup |
|---|---|---|---|---|
| 1 (in-process) | 32 | 132.2 | 16.5 | 1.00× |
| 2 | 16 | 101.4 | 24.0 | 1.30× |
| 4 | 8 | 66.7 | 29.9 | 1.98× |
| 8 | 4 | 54.0 | 43.5 | 2.45× |
| 8 | 32 (cap off) | 54.7 | 43.0 | 2.42× |

Workers reduce the batch time sub-linearly. Four workers finish the batch twice as fast as
one process and eight 2.5× as fast, while the mean wall-clock time per star rises from 16.5
to 43.5 s as the workers share the machine. A single in-process star is not a serial
program: XLA's CPU backend already distributes the banded solve and the operator assembly
over the cores. The extra processes therefore overlap only the serial remainder of each star
(compilation, the Python loop of the 41-point conjunction scan, the orbit fit, the writing),
which bounds the gain.

The thread cap made no measurable difference. The pipeline caps each worker's XLA and
BLAS threads at `cpu_count // jobs`, but eight workers each allowed all 32 threads finished
the same batch in 54.7 s against 54.1 s capped: on this workload the operating system's
scheduler handles the oversubscription. The cap is kept as a precaution with no measurable
cost here, because oversubscription has been observed on this machine in BLAS-heavy stages
(the re-run's 32-thread OpenBLAS). It is not the source of the speedup.

`jobs="auto"` is `cpu_count // 4`, eight on this machine, the last capped row. More workers
than cores gain nothing on a CPU: once the serial remainder is overlapped the batch is
compute-bound, and each worker holds its own XLA runtime (a few hundred megabytes) and its
own copy of the shared configuration.

Two cases were not measured. The label stage is the most expensive part of a full run
(~50 s against ~30 s of disentangling on this star in fast mode) and scales the same way,
being one more in-process JAX program per star. On a GPU the in-process star would be
faster and the workers' overlap smaller.

## Gaia RVS: disentangling and velocities on simulated double-lined binaries (2026-09-10)

Machine: the same desktop, CPU only, float64. Harness: `scripts/gaia_rvs_benchmark.py`,
which calls `albireo.benchmark.run_benchmark` and writes the full report with its figures,
`rows.csv` and `summary.json`, from which the numbers below are copied. Every star is run
as a user runs it, through `run_pipeline` from a `StarConfig`, and every metric comes from
the pipeline's truth block, so the harness adds nothing the pipeline does not report.
Research notes cited by file name in this section are in the repository under
`internal/research/2026-09-09-gaia-rvs-benchmark/`.

**Simulation.** Each system's two components are BOSZ 2024 spectra (`bosz2024-fgk-rvs`,
R = 20,000) at the labels the dwarf sequence gives for the masses, rotationally broadened,
and broadened to the RVS resolving power of 11,500 by the quadrature width. They are
shifted on the relativistic log-wavelength grid, combined at the RVS light fractions (the
library's continua times the radius ratio), and observed at the epochs of the cadence
model. Photon noise is added per 0.245 A detector pixel at the S/N the ESA predictive
formula gives for the system's G_RVS and one transit per epoch. The detector spectrum is
then delivered on the DR4 epoch grid (0.025 nm, 961 samples from 846.0 to 870.0 nm) by
linear interpolation with the variance propagated. The delivered noise is therefore
correlated, as the archive's will be (lag-1 coefficient 0.27 on this grid, 0.81 on the DR3
mean-spectrum grid), and this is not declared to the pipeline. Wavelengths are vacuum and
barycentric.

**Two populations.** The first is the field as a magnitude-limited double-lined survey
observes it: 20 systems drawn (`--n 20 --seed 3`) from the Moe & Di Stefano (2017) period,
mass-ratio and eccentricity distributions. The primary mass function is weighted by the
survey volume, each pair is kept in proportion to the volume its combined light reaches,
and the largest velocity separation has a floor of 40 km/s. G comes from the Gaia
double-lined sample's distribution and the transit count from the scanning-law model over
the DR4 span. One system with four transits was left out (the harness requires ten). The
quartiles of the 20 drawn are P 4.8 / 34 / 267 d, e 0 / 0.43 / 0.65,
K1 + K2 57 / 70 / 122 km/s, q 0.87 / 0.94 / 0.98, F2/F1 0.62 / 0.81 / 0.89,
Teff1 6100 / 6400 / 6700 K, G_RVS 8.5 / 9.2 / 10.1 and transits 28 / 40 / 53. Two eclipse.
Half the pairs are twins with q > 0.95, as in a magnitude-limited double-lined sample. One
pair has a secondary with 5 percent of the light (q 0.54, P 117 d, e 0.72), and one a
period of 844 days over the 1998-day span.

The second population is real: the first 16 double-lined orbits (`SB2` and `SB2C`
solutions) the Gaia archive returns from `nss_two_body_orbit`. Each has its
`binary_masses` where the catalogue gives them (else masses from the sequence at the Gaia
temperature and the catalogue's semi-amplitude ratio), the catalogue's period,
eccentricity, argument of periastron, semi-amplitudes and systemic velocity, the
inclination that reconciles the masses with the semi-amplitudes, the DR3 number of good
transits, and G_RVS from G and G_RP. Systems the library cannot render (a primary above
7000 K) are left out and named. The others are observed on the DR4 epoch grid with their
DR3 transit counts: the sample DR4 will contain, at a fraction of the epochs it will have.
Fourteen systems ran (46 runs). One run failed on the first pass at a starting value on the
range's bound and completed on the rerun after a guard was added. The quartiles are
P 3.1 / 5.9 / 8.6 d, e 0.013 / 0.068 / 0.36, K1 + K2 105 / 141 / 187 km/s,
q 0.93 / 0.96 / 0.98, F2/F1 0.76 / 0.85 / 0.96, Teff1 5800 / 6200 / 6700 K,
G_RVS 7.4 / 8.5 / 9.7 and transits 12 / 15 / 16. Four eclipse.

**Tiers.** Every system is simulated once and analysed under four declarations, each a
`StarConfig` in the pipeline's own vocabulary:

| tier | period | conjunction | elements | light fractions | label priors |
|---|---|---|---|---|---|
| oracle | known (Gaussian, width 1e-4 P) | known | known (e, omega held) | injected | narrow (300 K, 0.3 dex) |
| eclipsing | known | known (an eclipse ephemeris) | free | injected | narrow |
| orbit | known to a Gaia solution's precision | scanned | free | measured by correlation | wide |
| blind | searched (bootstrap from library templates) | scanned | free | measured by correlation | wide |

The eclipsing tier runs only on the eclipsing systems. The oracle tier measures the
disentangling and the velocities alone. The blind tier measures what the pipeline delivers
from the spectra alone. The semi-amplitudes are a range of 2 to 250 km/s wherever they are
not declared, the same for every system.

**The field population: 19 systems, 59 runs, none failed.** Entries are the median and the
16th to 84th percentile over each tier's systems, from `summary.json`. The semi-amplitudes
come from two sources, the Keplerian of the disentangling and the orbit fitted afterwards
to the velocity table, because they fail differently.

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
| wall-clock time per star [s], 8 workers | 476 (296 to 641) | 533 | 742 (579 to 978) | 647 (460 to 1650) |

The velocity separation determines the recovery more than the magnitude or the transit
count. Ten of the nineteen systems have a largest velocity separation above four
line-spread widths (about 100 km/s): nine with P < 25 d and the eccentric 117-day pair.
Under the oracle tier all ten had both semi-amplitudes within 1.7 percent, seven within
0.7. Under the orbit tier all were within 2.3 percent except the faint companion of the
117-day pair (4.2). Under the blind tier seven of the ten were within 2.1 percent. The
exceptions were a twin whose period was found at a third of its value (P = 3.95 d, both
semi-amplitudes then 10 percent off) and two secondaries at 3.6 and 6 percent. The
recovered spectra of the ten correlated with the injected ones at 0.85 to 0.99. The nine
below four widths (K1 + K2 of 34 to 72 km/s, seven with P > 130 d and e of 0.4 to 0.7,
11 to 62 transits, S/N of 7 to 11 per pixel) had oracle-tier primary errors of 3 to 32
percent and secondary errors of 0.1 to 76, larger under the other tiers, and spectra
correlating at 0.4 to 0.7. The lines of such a pair never separate by more than two
resolution elements, and no cadence compensates for that. G_RVS and the transit count order
the residuals only within the separated group.

The blind period search recovered 11 of 19 within 2 percent: every system with P < 50 d
but one (the 3.95-day twin above, found at 0.30 of its period), and the 117-day pair,
whose primary alone produced the periodogram signal. All seven with P > 130 d failed,
going to aliases at 0.0003 to 0.09 of the true period. The scanning law's day-scale
structure, sampled 11 to 62 times over the 1998-day span, aliases a slow orbit whose
semi-amplitudes are one resolution element.

The two sources of K fail differently. For the four systems with a faint secondary (F2/F1
of 0.05 to 0.17) the disentangling recovered K2 to 0.1 to 8 percent under the oracle tier.
Under the blind tier it recovered that of the 5 percent companion to 0.02 km/s, once the
semi-amplitudes were started from the table. The correlation table gave zero for that
companion in every tier (the template of a 5 percent secondary aligns with the primary's
lines, and the pipeline flags the disagreement with the disentangling). The table's formal
errors were too small. The semi-amplitude pull rms was 3 to 6 in the tiers where the orbit
is known, with the scatter about the Keplerian above the per-epoch errors for the poorly
separated systems. The epoch-velocity pulls were 1.2 to 1.5 (the delivered grid's noise
correlation of 0.27 was not in the noise model, which took the per-epoch errors as white).

Twins were recovered in the other order in four of the nineteen orbit-tier runs and one
blind run. At q of 0.98 with the light fractions measured rather than declared, the
mass-order convention could not distinguish the components, and the fit converged with K_A
above K_B by the 2 percent the data allow. The truth comparison recognises the exchange and
evaluates the recovery in that order, in which these runs are as good as the rest (0.1 to
2.3 percent). The eclipsing tier, where the light fractions are declared, kept the order.

The label fit's temperatures were 130 K (primary) and 180 K (secondary) from the injected
values under the oracle tier, and 250 to 300 K and 450 to 560 K under the wide priors of
the orbit and blind tiers. The primary's error was negative in fifteen of nineteen oracle
runs, a bias that needs separate study. Surface gravity was recovered to 0.2 to 0.3 dex.
The light fraction the label fit measures agreed with the declared one to 0.008 under the
oracle tier and to 0.03 to 0.05 where the fractions were themselves measured by
correlation.

Starting the semi-amplitudes from a template table was a change this run required
(decision record, `internal/design.md` section 2). Before it, the orbit tier of the same
population had a median secondary error of 44 percent over its 13 completed runs, with
both semi-amplitudes of the faint-secondary system at the floor of the range. The oracle
tier, whose semi-amplitudes are declared, is unaffected.

At 100 disentangling and 80 label steps with eight workers, the wall-clock time was 476 s
per oracle star (the velocity budget follows the declared K), 742 s per orbit-tier star
and 647 s per blind star (a range of 2 to 250 km/s sets a 998 km/s budget and a 3500-pixel
grid). The 59 runs took 110 minutes.

**The Gaia DR3 orbits: 14 systems, 46 runs.** These are the archive's first sixteen
double-lined orbits (`--gaia-sb2 16`), with two left out for a primary above the library's
7000 K. The fourteen have P of 0.54 to 30 d, K1 + K2 of 29 to 296 km/s, q of 0.53 to 1.00,
F2/F1 of 0.07 to 0.98, G_RVS of 6.7 to 10.4, and 10 to 25 transits, the DR3 counts. Four
eclipse at the inclination that reconciles their masses with their semi-amplitudes. Two
are near-contact pairs at 0.54 and 0.58 d with synchronised rotation of 100 to 120 km/s.

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
| wall-clock time per star [s], 8 workers | 369 (277 to 435) | 400 (273 to 466) | 462 (394 to 552) | 495 (378 to 811) |

This population separates the performance of the disentangling from that of its starting
values. With the orbit declared, the oracle tier recovered all fourteen systems within 4
percent in both semi-amplitudes on 10 to 25 epochs, at medians of 0.27 and 0.21 percent.
Ten were within 1 percent in the primary and thirteen in the secondary. The epoch
velocities were recovered to 1 to 2 km/s with pulls of 1.4 to 1.5. The worst system was the
0.72-day pair whose semi-amplitudes sum to 29 km/s, one line-spread width (4.1 percent on
K1). Gaia's DR3 epoch count supports this recovery when the period, the conjunction and
the elements are known.

With a semi-amplitude left as a range the same systems failed on about half the runs: 43
percent of the orbit-tier runs were within 5 percent, and 21 percent of the blind ones. The
failures come from the starting value, not from the fit that follows. A template table of
10 to 16 epochs does not give a reliable start. On the 2.81-day twins the eclipsing tier's
table had its semi-amplitudes reduced by exchanged epochs and started the fit at 37 and
32 km/s for a pair at 102 and 104. L-BFGS converged from there to 47 and 49, half the
injected values. The orbit tier's table of the same star, made with measured light
fractions, started at 130 and 127 and the fit converged to 102.2 and 103.9. On an 18.9-day
pair with a secondary rotating at 87 km/s the table started the secondary at 228 km/s for
an injected 35, and the fit stayed near it. On a 5.36-day twin the table measured neither
star, and the evenly spaced starts ended with the primary at the floor. The blind tier adds
the period search, which recovered 5 of 14 within 2 percent on 10 to 25 epochs (the field
population's 11 of 19 had 28 to 100). The rest went to aliases at 0.04 to 2.3 of the
period. The next step, not in this run, is a coarse scan over the semi-amplitudes through
the existing conjunction scan before L-BFGS on every route where they are a range.

Twins were recovered in the other order in 21 percent of the orbit-tier runs and 7 percent
of the blind, as in the field population, and are evaluated in that order. The label fit
gave temperatures 190 to 210 K from the injected values under the oracle tier on these 10
to 25 epochs, against 130 to 180 K on the field population's 28 to 100.

**Changes to albireo and open items.** Every defect the harness found is in the decision
record (D61). The changes visible to a user follow. The bootstrap of the search route
re-assigns exchanged epochs by each candidate orbit and tries the peaks of a
swap-invariant periodogram. With a degenerate bootstrap semi-amplitude the fit searches the
range and is not held at that value. A semi-amplitude declared as a range now starts at the
value from a template table fitted at the declared period, every start held strictly
inside the range. A pair recovered in the other order is recognised in the truth
comparison and flagged. The report tabulates the semi-amplitudes of the disentangling
beside the table's. The scan over the semi-amplitudes, the noise correlation in the noise
model and the temperature bias were left open and are taken up by the second run below,
which found the bias to be the library's, not the disentangling's. The scanning law's
phase and the ATLAS9 box above 7000 K, which would admit the two Gaia systems left out
here, remain open.

### The second run (D62, 2026-09-10): the epoch velocities as a product, the noise model, the scan, the library

Both populations were run again with the same simulated epochs (the same seeds) and the
same optimizer budgets, with five or six workers on the same desktop. The desktop was also
running unrelated jobs for much of the time, so the wall-clock times below are not
comparable with the first run's.

**Changes.** (1) The measured epoch velocities, one per component per epoch from TODCOR
against the disentangled components with the label fit's zero points, were a product of
every star run (`velocities.rv`), but the report gave only two statistics of them. The
benchmark now gathers them all with the injected velocity of each epoch into
`velocities.csv`, pools the usable epochs per tier, and draws every system phase-folded
against the injected orbit.

(2) The delivered grid's lag-one noise correlation (0.27 on the DR4 grid) is declared to
every tier. The disentangling runs the AR(1) noise model along the pixel index, and the
velocity table's errors include the correlation through the sandwich of math.md §10.4.

(3) A template table of a dozen epochs had started the range-K routes in the wrong basin
on half the Gaia population. A scan over the semi-amplitudes alone, at the phase the
conjunction scan had located at the wrong semi-amplitudes, found nothing on three test
systems. A phase located at semi-amplitudes a factor of three off was a quarter of a period
from the injected one. The `Disentangler` interface therefore scans the marginal
likelihood jointly over a geometric grid of every ranged semi-amplitude and eight
conjunction phases, refined twice around the best trial. The scan runs on a copy of the
declaration with the model grid at twice the pixel (a quarter of the cost at a dozen
epochs, two thirds at eighty). L-BFGS starts from the best trial when it is better than the
start. On the 2.81-day Gaia twins started at 37 and 32 km/s the joint scan returned 105.7
and 105.7 for a 102/103 pair, with the correct phase, and holding either component at its
start costs 350 to 400 nats. Two guards were added after a first rerun. A best trial with
every ranged semi-amplitude at the floor of its grid is the static-component minimum and
is rejected (it was returned for an eccentric contact pair scanned near-circular). A
component is moved from its start only when holding it there costs more than 25 nats
(holding a 7-percent secondary at its start cost 11). The template table's orbit starts a
free eccentricity where it detected one at three sigma. A dozen epochs fit an eccentricity
of 0.2 to a circular pair as readily as not, and a scan on that shape of curve lost the
same secondary.

(4) The temperature bias was not the disentangling's. A label fit on a perfect spectrum
drawn from the grid itself ended 20 to 150 K from the true labels with a chi-square above
theirs. This is because BOSZ publishes no model at 5750 K, log g 3.0, [M/H] −0.75, and that
one gap had made the whole grid fall back from the Catmull-Rom cubic to the barycentric
simplex interpolant. That interpolant is piecewise linear over an arbitrary triangulation
of the lattice, and its kinks stop L-BFGS. The gap is now filled by linear interpolation
along [M/H] between its published neighbours, so the box is complete, the cubic applies,
and the same fits end within about 30 K. The first run's formal label errors, meaningless
at a kink (a pull rms of 2300 in the oracle tier), are now curvatures of a smooth surface.

**The field population again: 19 systems, 59 runs, none failed.** The nineteen systems
and their epochs are those above. Entries are the median and the 16th to 84th percentile
over each tier's systems. The first run's value follows in brackets where it differs by
more than the spread.

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
| wall-clock time per star [s], 5 workers, shared machine | 421 | 477 | 614 | 580 |

This population's range-K routes were started from tables of 28 to 100 epochs, which gave
good starts, so the scan changed little and mostly agreed with them. The orbit-tier
medians moved within their spread (1.9 and 1.8 percent against 1.8 and 2.6). The blind
tier's secondary improved from 12 to 5.4 percent and its conjunction error from 0.084 of a
period to 0.024. The period search found 12 of 19 against 11, and the blind tier's
epoch-velocity pulls fell from 3.8 and 3.1 to 1.3 and 1.6. The oracle tier's primary
improved from 1.7 to 0.8 percent. Its epoch velocities (859 pooled epochs, of which 98
percent are usable) were within three sigma of the injected value 90 percent of the time
at pulls of 1.4 and 1.1. Their median absolute residual was 0.9 and 1.3 km/s against a
median quoted error of 1.0 and 1.2. The exchanged runs rose from four to five in the orbit
tier and from one to three in the blind, the same twins at q of 0.98, evaluated in the
order recovered.

The velocity separation still determines the recovery. The seven systems with P above 130
days and K1 + K2 below 65 km/s were the worst of every tier, the oracle included, where
their primaries were 6 to 59 percent off (three of them 54 to 59 percent low). No other
tier did better, because their lines never separate by more than two resolution elements,
and for the same reason the scan found nothing there.

The label fit's temperatures improved where the disentangled spectra are good: the
eclipsing tier's secondary from 262 to 104 K and the orbit tier's primary from 255 to 128.
They stayed at 120 to 190 K under the oracle tier's 300 K priors, where the reports again
state that the fit learned nothing about the primary's temperature in fifteen of the
nineteen runs.

**The Gaia DR3 orbits again: 14 systems, 46 runs, none failed.** The fourteen systems and
their epochs are those above. Entries are the median and the 16th to 84th percentile over
each tier's systems. The first run's value follows in brackets where it differs by more
than the spread.

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
| wall-clock time per star [s], 6 then 5 workers, shared machine | 413 | 609 | 581 | 551 |

The results changed in the range-K tiers. With the period known and the conjunction
scanned, nine of the fourteen systems were within 5 percent in both semi-amplitudes,
against six before. The median errors fell from 13 and 25 percent to 3.5 and 4.2, and the
table's from 10 and 20 to 0.9 and 1.9. The conjunction error fell from 0.076 of a period
to 0.005, the exchanged runs from three to one, and the epoch-velocity pulls from 4.5 and
7.7 to 1.5 and 1.3. The 2.81-day twins, whose eclipsing-tier fit had converged to half the
semi-amplitudes from a table start of 37 and 32 km/s, were recovered to 3 percent in that
tier and 5 in the orbit tier. In this run the table itself started them well, at 98 and
101 km/s, because the cubic interpolant had improved its templates, and the scan agreed
with that start. The four eclipsing runs improved from 16 and 34 percent to 3 and 14.

The remaining failures have causes the scan does not address. The 0.54-day contact pair at
e = 0.21, with both stars rotating at 100 km/s, has lines four resolution elements wide.
The scan from a near-circular start returned the static minimum, which was rejected, and
the fit from the starting values stayed 51 and 68 percent low. For the 7-percent secondary
of the 0.58-day pair, the coarse scan put the semi-amplitude of 193 km/s at 23 with a hold
cost of 43 nats, above the guard's 25, while the primary was recovered to 13 percent. The
0.72-day pair has K1 + K2 = 28 km/s, one line-spread width, which no start corrects. For
the 18.9-day pair at e = 0.41, whose secondary rotates at 87 km/s, a static broad secondary
fits as well as the moving one at the scan's resolution. The 8.86-day pair started the
orbit tier at the table's 54 and 53 km/s (injected 53 and 55), the scan found nothing
better, and the fit ended at 62 and 64 after its 100 L-BFGS steps. The fit moved away from
a good start, an outcome decided by the step budget or the shape of the likelihood, not by
the scan.

The oracle tier was unchanged within its spread (medians 0.45 and 0.36 percent against
0.27 and 0.21). Under the AR(1) noise model its epoch-velocity pull rms fell from 1.5 and
1.4 to 1.1 and 0.94. Of the 215 pooled epochs, 93 to 95 percent were within three sigma of
the injected velocity, with a median absolute residual of 0.6 and 0.8 km/s against a median
quoted error of 0.7 and 0.9. The pooled rms residuals (18 km/s in this tier) are dominated
by the two pairs whose lines never separate, the 13/15 km/s pair and the contact pair, so
the report quotes them beside the medians. The blind tier remained limited by the period
search: 3 of 14 were found within 2 percent (5 before, the difference within the noise of a
bootstrap whose templates changed), and every failure after it follows from an alias.

The label fit's temperatures improved where the disentangled spectra allow it: the
eclipsing tier's errors fell from 257 and 205 K to 28 and 74, the orbit tier's from 210 and
326 to 166 and 260, and the blind tier's primary from 388 to 155 K. The oracle tier stayed
at 196 K for the primary. In eight of its fourteen runs the fit reported that it
"learned nothing about teff_A" (a posterior width above 80 percent of the prior). The ten
to twenty-five epochs at S/N 14 to 100 per pixel contain little temperature information in
this band once the prior is 300 K wide, so the residual is the prior's and not the
interpolant's.

**Conclusions and open items.** The epoch velocities are a product that can be read
directly from the report. Their errors are calibrated where the noise model is the declared
one. Under the oracle and orbit tiers the pull rms of the usable epochs is 0.9 to 1.5 and
more than 80 percent of them lie within three sigma of the injected velocity, which the
first run's diagonal errors did not achieve. The range-K routes recover the systems whose
lines separate by more than about three resolution elements, from a template table of a
dozen epochs or from none. The remaining failures are of four kinds a scan cannot correct:
lines that never separate (a combined semi-amplitude of one resolution element),
companions of a few percent of the light whose semi-amplitude the coarse likelihood does
not determine, rotation at a hundred kilometres per second in an eccentric orbit, and a
period the blind search did not find. The temperature scatter of the label fit was the
interpolant's, and the eclipsing and orbit tiers now measure temperatures to a few tens
and somewhat over a hundred kelvin. Where the prior is 300 K wide and the epochs are few,
the fit correctly reports that it learned nothing. The third run below takes up the period
search (on a floating mean with the disentangling deciding), the step budget (measured,
300), the scanning law's phase (real epochs from GOST are now a cadence) and the ATLAS9
box above 7000 K (registered, not yet fetched). The lines that never separate remain. The
DR4 grid is still based on the draft data model, and no run here was made on the released
product.

### The third run (D63, 2026-09-11): the period search, the sequential semi-amplitude scan, the step budget and real epochs

Each open item of the second run was first measured on that run's archived products and
then changed. Both populations were run again with the same simulated epochs (the same
seeds), under the changed code and a step budget of 300. The wall-clock times are again
not comparable, because the desktop was shared with other jobs and fewer workers were used.

**Measured on the second run's archive before any change.** (1) The blind tier's period
search lost the period at candidate generation, not at the decision. On the oracle tier's
velocity tables of the same epochs (velocities near the injected ones with realistic
errors, 32 usable systems), the true period ranked 9, 9, 15, 25 and 103 among the distinct
peaks of the classical Lomb-Scargle periodogram on the five Gaia-like systems the search
had lost. On a floating-mean periodogram of the same series it ranked 1, 1, 1, 10 and 3.
`scipy.signal.lombscargle` fits a sinusoid with no constant term. Ten to twenty-five
epochs falling into a dozen visibility windows hundreds of days apart have a sampling
window whose mean is large at most frequencies. The constant the model cannot fit is then
absorbed into the sinusoid, and spurious peaks outrank the true period. Six candidates
were also too few (twenty is right). The point-wise scaling of the series by the square
root of its weights (wrong, but with a small effect here) and the frequency grid's density
(ten per inverse baseline; refining it to fifty changed no outcome) were not the cause.
The orbit's chi-square decides well: where the true period was among the candidates it won
in 27 of 29 systems, usually by hundreds. Twenty candidates from the floating-mean
periodogram and twenty from its two-harmonic form, with the swap-invariant peaks kept,
raised the recovery on those tables from 23 to 28 of 32. The four left are two tables of
ten or eleven epochs whose true Keplerian fits worse than an alias, a period 2.04 percent
off, and one decision lost by 4.4 in chi-square.

(2) The scan did not lose the 7-percent secondary to the coarse model. With the pixel
doubled and the primary at its injected value, the marginal likelihood peaks at 203 km/s
for an injected 193, as in the full model, and is 15 to 18 nats higher there than at the
static value. The secondary was lost because the companion's likelihood peaks at the true
semi-amplitude only while the primary's is within about ten percent of its true value,
whereas the primary's own peak falls by 50 nats within twenty percent. On the product grid
at ratio 1.8 no trial held the primary close enough, and the local refinement did not move
the companion out of its starting basin. The guard then rejected the correct joint move of
39 nats as two single-component holds of 15 and 16.

(3) The fit is limited by the step budget, not by the shape of the likelihood. The
8.86-day pair's orbit-tier fit, reproduced to the last digit at 100 steps, ended 8.8 and
8.4 km/s from the injected semi-amplitudes, at 300 steps 3.5 and 2.8, and at 1000 steps
1.6 and 0.7. Across these budgets the eccentricity error went from +0.20 to +0.04 and the
conjunction error from 1.05 to 0.16 d. The gradient norm was still 6500 times the
tolerance at the end. The potential is a poor proxy for the progress: 95 percent of its
fall occurred by step 300, and the largest gain in K_B after it. The velocity table's
orbit, which the report quotes, stopped improving by 300 steps. Steps 101 to 300 cost
about 1.1 s each on 16 epochs.

(4) The light fractions the orbit and blind tiers measure by correlation against generic
library templates were off by more than a factor 1.5 on 8 of 33 systems. The errors form
two separate groups: secondaries below 8 percent of the light were measured too bright (a
6.7-percent star at 0.40), and near-equal pairs too faint (0.42 to 0.54 at 0.18 to 0.30).
The sign of the error sets the sign of the semi-amplitude error: K_B is too small for a
secondary measured too bright and too large for one measured too faint. The label fit
inherits the error without correcting it, and the second correlation pass holds the light
fixed, so no later stage re-measures it. The velocity table's detection statistic for the
secondary is a test for this error: a value below 100 selects 11 of the 16 runs and 4 of
50 good ones.

(5) One oracle-tier table of the second run had every epoch at the search edge with no
error. Its disentangling had diverged. The line search accepted a single trial at a
smoothness precision of 2e19, where the Cholesky solve loses every digit, the quadratic
form came out above the data term, and the negative chi-square was read as a 216,000-nat
improvement. At correlations 0.0, 0.1 and 0.5 the same fit converges. The label fit had
marked its own result as unreliable and returned a zero point on its scan bound. The
correlation stage had searched a window that could not contain the true velocities, and a
fine-pass defect reported a position five pixels from any it had evaluated.

**Changes.** They are listed in the order of the measurements above. (1) The period
search, `find_period`, is the weighted floating-mean generalized Lomb-Scargle periodogram
of Zechmeister and Kurster (2009), with twenty peaks and a two-harmonic form. The pipeline
proposes the union of those two lists, the peaks of the first component alone (the source
unaffected by a companion the templates could not follow) and the swap-invariant peaks
with their doubles. It fits the eccentric Keplerian at every distinct start and sets aside
an orbit above the declared semi-amplitude ceiling or eccentricity maximum (a lost
companion's velocities had let a wrong period win at 1537 km/s and e = 0.94). It merges
fits on the fitted period and flags every rival within 25 in chi-square. The disentangling
then decides among the best four candidates, a number that a later measurement kept. This
stage ran on the Gaia population and on one field system, having been finished after the
field blind tier had run. Each candidate is declared with every semi-amplitude as the same
range so that the coarse grids match, located by the coarse scans, and compared on the
prior-free marginal log-likelihood. The stage was added because on the noisier bootstrap
tables the chi-square still chose grossly wrong orbits inside the declared ranges (twins
at 6.10 d sent to 0.248 d at K 233/239 and e 0.73). On the 32 archived tables the
candidate list and the chi-square alone recovered 28 against 23, and the flag was raised
on three of the four misses and on no recovery.

(2) The semi-amplitude scan takes its axes one at a time. It scans the first ranged
component at a factor 1.25 with eight phases and each further one at the values already
located, refines jointly, scans each component once more over its whole grid, and tries
the exchange-symmetric twin of the best trial. It moves the start only when the best trial
beats it by 25 nats jointly, and a component whose own move is worth under 5 nats returns
to its start. The scan takes about 270 trials against 446.

(3) Between two scan passes each star's prior amplitude is profiled at the located orbit
through the model's light site, and the hyperparameter starts are moved where the profile
indicates a factor of two or more. The profile is exactly a profile over the prior
amplitude, not a light measurement. The marginal likelihood is invariant, to 5e-10 nats,
under scaling a star's light by a factor and its smoothness hyperparameters by the factor
squared, and as a light estimate the profile was at 0.3 to 0.7 of the injected value. It
nevertheless rejected the declared amplitude of the 7-percent secondary by 150 nats at the
orbit the first scan had located, and the re-scan brought the secondary's semi-amplitude
error to 4 percent, where the second run had a static companion.

(4) The step budget is 300.

(5) The marginal likelihood returns minus infinity on a negative chi-square, and the model
rejects a smoothness ratio above e^30. The diverged fit now ends at z-score rms 0.97,
within 0.1 percent of the injected values.

(6) The correlation stage reports the position it evaluated, clamps its window, writes nan
for an unmeasured component, and searches one window per template, offset by its zero
point. The pipeline rejects a zero point that the label fit marked as unreliable, marks a
failed table in its file and report, stops a star whose residual z-score rms exceeds 10,
and rejects a declared table with an unmeasured epoch.

(7) Real transit times from GOST are available (`cadence="gost"`) but were not used in
this run, so that its epochs stay those of the first two.

(8) A BOSZ box from 7000 to 10,000 K is registered but not downloaded, so the two hot Gaia
systems stay out of this run as before.

(9) The library-template velocity table is a product of every run.

(10) Three guards on the velocity table come from the third run's own products and were
added after its tables were measured (the tables below predate them). The exchange of the
two components by the orbit runs only when their light fractions are within a factor 3 (on
a 95/5 pair it had swapped 19 of 80 epochs on noise and moved the primary from 5 to 56
percent off). The table as measured is kept beside the delivered one. The amplitudes are
fitted freely when any smoothness precision has moved more than a factor 10 from its start
(a secondary whose lines ML-II had made a quarter shallower was lost at the declared
fraction and half recovered with a free amplitude). A frame offset the label fit did not
constrain no longer sets that component's zero point (a mostly-noise secondary had a
46 km/s offset in every velocity).

**The Gaia DR3 orbits again: 14 systems, 46 runs, none failed.** The fourteen systems and
their epochs are those of the two runs above, at 300 steps. Entries are the median and the
16th to 84th percentile over the systems of each tier, with the second run's value in
brackets where it lies outside the spread. The blind tier was run last, after the decision
by the disentangling had been added. The eclipsing contact pair had failed at the
correlation stage's new window check and was run again once the zero points could be
dropped instead.

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
| wall-clock time per star [s], 3 workers, shared machine | 403 | 580 | 654 | 671 |

The results changed in the blind tier. With nothing declared but the spectra, the period
was found within 2 percent on seven of the fourteen systems against three. The median
semi-amplitude errors fell from 24 percent to 6, the table's from 27 and 25 to 1.4 and
2.2, the epoch-velocity pulls from 16 and 17 to 3.7 and 3.6, and the eccentricity error
from 0.20 to 0.07. The decision by the disentangling overruled the table's chi-square on
four of the seven misses and chose another alias each time. The whole chi-square ranking,
rebuilt offline from the archived bootstrap tables with their component exchange undone,
reproduces the run's own choice on 30 of the 33 blind systems and supersedes the ranks
first quoted here (4, 6, 6 and 14). In it the true period is first on every system the run
recovered, third on one miss, in the fifth to eighth places on one, in the ninth to
twentieth on one, and absent from the candidate list on four
(d64_period_candidate_ranks.md). Raising the number compared from four to eight would
reach one further system in each population, so the default stays at four.

**The decision by the disentangling flags an unreliable period and does not correct it
(D64, measured on the archive).** The comparison scans the semi-amplitudes and the
conjunction but scores each candidate at a single point in period and eccentricity, the
two quantities a sparse table measures worst. On the one Gaia system where the true period
was among the four compared, it scored 251 nats below the winner, while the same model at
the true period, eccentricity and conjunction scores 185 above it
(d64_period_decision_scan.md). Scoring each candidate over its whole declared window is
neither reliable nor affordable. The window maximum favours the shortest period: peaks are
spaced by about $`0.28\,P/T`$, so a three percent window holds about 50 independent trials
for a candidate near the true period against 550 to 1500 for its shorter rivals. When that
bias is corrected by extreme-value extrapolation, two estimators disagree on the winner.
The true period is found only when the semi-amplitudes are scanned inside the period loop,
which costs about 62 hours for four candidates against the 116 seconds of the point
comparison and scales as $`T/P`$. The stage has converted no miss into a hit on either
population and has caused no additional miss. All four of its overrules on the Gaia blind
tier were on systems that ended on a wrong period, and three further misses were not
overruled. Its flag therefore indicates that the period is unreliable, not that the
selected period is better.

In the orbit tier, with the period known and the conjunction scanned, both semi-amplitudes
were within 5 percent on ten of the fourteen systems against nine before, and the medians
went from 3.5 and 4.2 percent to 1.0 and 2.2. The 7-percent secondary, static in the
second run, was 7.5 and 2.8 percent off, because the prior-amplitude pass rejected its
declared amplitude and the sequential scan found it. Three systems still failed: the
0.54-day contact pair with both stars rotating at 100 km/s (46 and 94 percent off, in
every tier but the oracle), the 0.72-day pair whose lines never separate, and the 18.9-day
pair at e = 0.41 whose secondary rotates at 87 km/s. For the last, a static broad
secondary now has the higher marginal likelihood even under the oracle tier. At 300 steps
the fit reached a minimum 12 nats deeper than the second run's, with K_B at 2 km/s
against 35. The preference is a property of the likelihood at this resolution, not of the
optimizer.

The oracle tier was unchanged within its spread, and its epoch velocities remained
calibrated (92 and 91 percent of the 215 pooled epochs within three sigma, median absolute
residuals 0.64 and 0.79 km/s). The eclipsing tier's medians were set by the contact pair.
The other three systems were at 2 to 6 percent, and the tier's epoch-velocity pulls fell
from 2.4 and 2.8 to 1.0 and 1.2 with the correlation stage's window per template.

The label fit's temperatures worsened. The eclipsing tier's primary error rose from 28 K
to 258 and the orbit tier's secondary error from 260 to 328, and on two more stars the fit
reported that it learned nothing about the temperature. The step budget caused this
through ML-II. At 100 steps the smoothness precision had barely left its start (300 to 400
on the twins). At 300 it reached its optimum, 1e4 to 1e6 on most stars, and the
disentangled components were better correlated with the injected ones (0.82 to 0.93 in the
eclipsing tier) and had a smaller standard deviation of the normalised flux. That decrease
is not a shrunken posterior mean whose lost depth the label fit takes for dilution, which
was the reading first given here (corrected in d64_posterior_smoothing.md). The curvature
penalty is identically zero at zero frequency, and an equivalent width is the
zero-frequency component, so a component's own smoothness cannot change its equivalent
width. The transfer there is the data weight over the data weight plus the ridge, which
holds to 0.8 percent over 34 components, and the smoothness precision does not enter it.
Most of the lost standard deviation is noise the smoothing removed, not line depth: on
mixed-0008's primary the standard deviation fell to 0.797 while the equivalent width over
the same window moved by 1.021. An equivalent width is changed by the coupling between the
components, the off-diagonal blocks of the accumulated normal matrix, which account for a
median 0.952 of the archived departure against 0.003 for any per-component filter. The
orbits improved and the labels did not, and both follow from the same change.

**The field population again: 19 systems, 59 runs, none failed once the window was centred.**
The nineteen systems and their epochs are those above, at 300 steps. One orbit-tier run of
the 3.99-day twins had failed twice at the optimizer's initialisation. The semi-amplitude
scan had moved the conjunction by half a period (its phase grid contains the antipode that
the 41-point phase scan cannot sample, 672 nats better there). The second scan pass gained
exactly nothing, so the final phase scan did not run, and the one-period window was still
centred on the phase scan's best. That put the start on the window's edge, where the
unconstraining transform is infinite. With the window centred on the start of the fit, the
star converges to within 0.2 and 0.1 km/s of the injected values. The blind run of the
553-day pair had failed at the same window check and was run again with the zero points
dropped.

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
| wall-clock time per star [s], 3 workers, shared machine | 506 | 779 | 908 | 757 |

This population did not improve, for the reason given in the two runs above. The seven
systems with periods above 130 days and K1 + K2 below 65 km/s, whose lines never separate
by more than two resolution elements, are the worst of every tier and vary from run to run
(the oracle tier's primaries are 6 to 59 percent off in both). The twelve short-period
systems were already within a few percent at 100 steps and remain there.

Two results changed, both in the velocity table. First, the orbit tier's table errors rose
from 1.2 and 1.8 percent to 4.6 and 6.9 because three tables lost a component. Each loss
was traced on the archived products to a different cause. On the 95/5 pair the exchange
step swapped 19 of 80 epochs on noise draws of the faint component's velocity and moved a
primary measured to 5 percent to 56. On the 2.09-day pair the disentangled secondary,
shallower by a quarter as its smoothness precision went from 400 to 2000, was lost under
the declared light fraction and half recovered under a free amplitude. On the 553-day pair
the label fit had placed the rest frame of a mostly-noise secondary 46 km/s from the
systemic velocity. The three guards that followed (an exchange only between similar light
fractions, free amplitudes once the smoothness has moved, no zero point from an
unconstrained frame offset) were added after these tables were measured. Second, the blind
tier's period search found eleven against twelve. The difference is one long-period system
whose 2 percent window is the grid's own resolution. Those eleven come from the candidate
list and the chi-square alone. The decision by the disentangling did not run on this
population, because the stage was finished after this population's blind tier had run. The
one field system on which it ran is the 553-day pair, rerun afterwards, where it overruled
the table and chose another alias.

**Conclusions and open items.** On Gaia's own epochs, ten to twenty-five per star over
five years, the blind route now finds the period on half the double-lined orbits and
recovers both semi-amplitudes within 5 percent on half, up from a fifth. With the period
known, ten of fourteen systems are within 5 percent on both, the one with the faint
secondary among them. The epoch velocities, which are the product, remain calibrated where
the orbit is right (pulls of 1.0 to 1.3 in the oracle, eclipsing and orbit tiers) and are
reported as failed, not as numbers, where it is wrong.

The run measured three things. The fit is limited by the step budget, and the budget
cannot be inferred from the potential. The divergence recorded in the second run was
caused by the arithmetic of the marginal likelihood at a smoothness precision of 2e19, not
by the noise model, and is now guarded against. The disentangled components change with the
budget in a way that is not passed to the label fit, and its temperatures are worse as a
result. The mechanism is not a shrinking of the lines by ML-II: no per-component smoothing
can change an equivalent width, and the coupling between the components accounts for a
median 0.952 of the departure (d64_posterior_smoothing.md).

Four items remain open, in the order they should be taken. The first is the label fit
compared through the operator the disentangling applies, which is the coupled one over
both components and not a per-component smoothing. The second is the candidate periods.
The decision among them scores each candidate at a single point in period and in
eccentricity, the two quantities a sparse table measures worst, and was afterwards
measured to have converted no miss into a hit in fourteen systems. The third is a light
measurement, which cannot come from the disentangling because the likelihood depends only
on the products of light and line depth. The fourth is the lines that never separate,
which no route here recovers. A phase grid that samples its own antipode has since been
implemented. The label fit, the light measurement and the velocity-table failures were
examined afterwards on this run's archive, in the D65 subsection below.

### After the third run (D65, 2026-09-17): offline measurements on its archive

Everything in this subsection was measured offline on the third run's archived products.
The code described here postdates the tables above, and no benchmark has been rerun with
it.

**The third run's label temperatures are set by the warm start, not by the likelihood.**
Its label fits stopped near the node the warm-start scan chose, at the 80 L-BFGS steps
allowed. Started at the injected labels instead, the fit reached a lower value of the same
objective on 8 of 12 products and ended a median 7 K from the injected value
(d65_label_likelihoods.md). The oracle tier's label priors are also centred on the
injected values, and the optimiser's unconstrained-space Jacobian shifts a weakly
constrained label towards the centre of its prior, so a label the data barely constrain
appears accurate there.

**Compared against the epochs instead of the disentangled components, and converged, the
label fit halves its temperature error.** The disentangling's statistics give the
chi-square of a template pair against the epoch spectra in closed form, exact for the
correlated noise and independent of the declared light. Minimised by a bounded
Levenberg-Marquardt algorithm with restarts over 22 products, it gave a median
temperature error of 60 K against 119 K for the previous comparison converged the same
way. The corresponding errors were 0.040 and 0.134 dex against 0.060 and 0.692 in surface
gravity, for primaries and secondaries, and 0.006 against 0.024 dex in metallicity. The
formal errors were close to calibrated for temperature, gravity and metallicity
(d65_converged_labels.md). This comparison is now the default of `Fit.match_labels` and of
the pipeline's label stage.

**The same comparison measures the light fraction, which the disentangling cannot.** On
the orbit tier its light fraction differed from the injected one by a median 0.011,
against 0.043 for the fraction the light stage measured by correlation before the
disentangling, and it was closer on 10 of 11 products. The exception is the product whose
archived orbit is wrong. The pipeline reports the measured fraction beside the declared
value and flags a disagreement. It does not replace the declared value in the velocity
measurement, because the velocity templates are the disentangled components, whose line
depths only the declared light reproduces.

**Rotation was biased by smoothing that was not declared, in the data and in the model,
and both are now accounted for.** The delivered epochs include the delivery interpolation,
the detector pixel and the simulator's own grid in addition to the nominal resolving
power, a width of 11.670 km/s on the DR4 product and 11.896 on DR3 against 11.07. The
benchmark now declares that width. Separately, the disentangling's own shifts and pixel
boxes on the model grid add seven twelfths of the squared grid spacing, which the label
fit attributed to slower rotation (6.9 against 11.0 km/s on the pipeline's closed-loop
test at its default grid). The epoch comparison now removes it by default
(d65_grid_smoothing.md). On three products the median rotation error fell from +3.06
to +0.63 km/s, with about 0.9 square km/s of width still unaccounted for.

**Two of the five blind systems the period search never reached are recovered by changes
to the velocity table, and two cannot be.** The tables were examined epoch by epoch
(d65_velocity_table_failures.md). One twin at 5.36 days had a single exchanged epoch of
twelve. It is recovered by proposing the peaks of the leave-one-epoch-out periodograms on
tables of at most 25 epochs. A 7 percent companion at 553 days was undetected at 24 of 26
epochs and corrupted the joint fit. It is recovered by giving a companion's velocity no
weight where its detection statistic is under 100. Over the 33 blind tables the true
period reaches rank 1 on 22 against 20, with none lost. A pair of 100 km/s rotators at
0.54 days measured with unrotated templates, and a nine-epoch table on seven nights, are
not recoverable from their tables. The pipeline now flags a table on fewer than eight
nights. A near-twin at 310 days whose lines blend at 55 of 62 epochs is recovered only by
an assignment rule that loses two other systems, and is left open.

**The detection threshold of 100.** Measured with the code as implemented over the 33
blind tables, the detection threshold changes 9 of them, those with a companion velocity
whose statistic is below 100. The 7 percent secondary at 553 days was not detected by the
library templates at any epoch. Its velocities were spread across the whole search window
at statistics of 1.4 to 61 and, fitted jointly, had moved the orbit from the true period
to 511 d. The threshold takes that system from absent in the chi-square ranking to rank 1,
at 547.4 d against a true 552.5 d. It moves one Gaia system from rank 20 to rank 8 and one
field system from rank 24 to absent, and no system loses its rank 1. The true period ranks
first on 21 systems against 20, and the leave-one-epoch-out candidates add the 22nd. At
25, the table summary's own threshold for a weak detection, the rule fails on the 553-day
system, because six epochs with statistics between 25 and 61 keep secondary velocities up
to 212 km/s off. On the field population a statistic of 100 also separates blended epochs
from measured ones, which the table's blend flag does not mark at this resolution. On one
system all 55 epochs whose lines are less than one line width apart are below it and all 7
separated epochs above. The threshold is never applied to the first component. Applying it
to every component was also measured and lost one system. A field system whose first
template fell below 100 at 5 of 16 epochs had its true period go from rank 2 of the
chi-square ranking to absent, and the period decision compares the top four of that
ranking.

**The optimiser's change of variables does not bias the orbits it recovers, and a hard
eccentricity bound holds some of the others fixed.** Removing the log-Jacobian that
`run_map` minimises along with the posterior moves the archived orbits by a median 0.007
formal sigma and at most 0.23. The pull is larger only where a semi-amplitude has fallen
into the bottom 5 percent of its range, which is the case for 7 products, of which 6 were
failed orbits (d65_run_map_jacobian.md). Nine archived fits, all failed orbits, are on the
eccentricity limit of 0.9, where the model's bound is an infinite barrier that stops the
line search, so those fits never left it.

The open items follow, in the order they should be taken. The first is a fourth run with
these changes, which is the only way to turn the offline numbers above into benchmark
numbers. Next are a blend flag that is raised at this resolving power (the present one was
raised on none of 86 blended epochs), an eccentricity bound that does not hold the fit
fixed, and the assignment rule measured inside the period decision. The rest are the
remaining width on the RVS products, the lines that never separate, a GOST-cadence run,
the hot box, and the DR4 grid against the release.
