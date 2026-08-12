# albireo — design document (v1)

**Status:** M0 draft for review. Companion document: [math.md](math.md) (all equations live
there; this document is about decisions).

albireo is an open-source Python package for **spectral disentangling** of double- and
multiple-lined spectroscopic binaries: given a time series of composite spectra, jointly infer
the orbit and the individual component spectra, with honest uncertainties on both. The name is
the mission: Albireo is the famous gold-and-blue double in Cygnus — albireo separates the gold
from the blue. Target users: massive-star binary surveys, dormant black-hole/neutron-star
companion searches, benchmark eclipsing binaries, and multi-epoch survey spectroscopy
(APOGEE/SDSS-V, WEAVE, 4MOST). Endgame: community-standard replacement for fd3/KOREL, a
methods paper (A&A/ApJS) + JOSS paper, pip-installable with tutorials from day one.

---

## 1. Prior art

| Method / code | Domain & approach | Strengths | Failure modes | We adopt / avoid |
|---|---|---|---|---|
| **Simon & Sturm 1994** (A&A 281, 286) | wavelength space; big sparse linear system, SVD, wrapped in orbit optimization | conceptually clean: disentangling *is* a linear inverse problem; natural weights/masks; arbitrary per-epoch RVs | SVD cost/memory blow up with N; ill-conditioned low-frequency nullspace, no regularization; no UQ | **Adopt** the linear-algebra formulation as our conceptual ancestor; replace SVD with structured/banded solvers + priors |
| **KOREL** (Hadrava 1995, A&AS 114, 393; 2004 guide; VO service, Škoda+2012) | Fourier space; simultaneous orbit + spectra; line-strength variability | most feature-rich incumbent; decades of use | Fourier low-frequency degeneracy; closed-source FORTRAN behind a registration-gated VO service (KOREL11b, 2011) — not scriptable or reproducible offline; formal errors long criticized; no UQ on spectra | **Avoid** Fourier space entirely; explain undulations as nullspace (math.md §5.1) |
| **fd3 / FDBinary** (Ilijić+2004, ASPC 318, 111; ascl:1705.012) | Fourier space, C + GSL, CLI | fast, template-free, triple-capable (hierarchical), scriptable, heavily cited | requires equidistant common log-λ sampling; Keplerian-only RVs; low-frequency "bias progression" (Hensberge+2008); no UQ anywhere; page last updated 2014, no license statement | **Avoid**; fd3 is the explicit benchmark target for accuracy + wall-time (M5) |
| **CRES** (Ilijić) / **Spectangular** (Sablowski & Weber 2017, A&A 597, A125; 2019, A&A 623, A31; C++/Qt, Apache-2.0) | wavelength space; SVD + global optimization of orbit or per-epoch RVs | wavelength-space flexibility (weights/masks); global optimizer; handles line-strength variability | CRES abandoned (2006), takes RVs as input; Spectangular GUI-centric, hard to batch, tiny community, point estimates only | direction validated; we add scalability + Bayesian inference |
| **Shift-and-add grid disentangling** (González & Levato 2006; Shenar et al. 2020, A&A 639, L6 & 641, A162 — LB-1/HR 6819; code: TomerShenar/Disentangling_Shift_And_Add) | iterative shift-and-add over a (K₁,K₂) χ² grid | simple, robust at low SNR, de facto standard in the massive-star/BH-candidate community; χ² maps show the K degeneracies | grid search scales exponentially in dimension; slow/stalling iteration = implicit regularizer nobody controls; **light ratios set by hand** (the crux of the LB-1/HR 6819 debate); no posterior; repo has **no license** (cannot vendor) | **Adopt** the use cases and diagnostics (K₂ scans, detection maps) as first-class features; replace the machinery |
| **PSOAP** (Czekala et al. 2017, ApJ 840, 49; MIT, dormant since 2017-12) | GP joint inference in wavelength space; component spectra marginalized analytically | principled joint Bayesian posterior over orbit *and* spectra — closest ancestor of albireo's design | **dense** GP covariances over epochs×pixels → cubic cost, quadratic memory → forced tiny chunks; NumPy/Cython, no autodiff/GPU; unmaintained | **Adopt** the philosophy; **avoid** dense kernels — banded *precision* priors + structured solves (math.md §2, §4) |
| **Seeburger et al. 2024** (MNRAS 530, 1935 — "autonomous disentangling for surveys") | wavelength space; second-derivative Tikhonov + sparse LSMR solve; Nelder–Mead over RVs/q | closest methodological competitor; survey-oriented; validated on the Unicorn/Giraffe | explicitly not Bayesian, no posteriors, no GPU/autodiff; **no public code** as of 2026-08 | their curvature penalty is exactly the deterministic limit of our τ·D₂ᵀD₂ prior — we get it for free *plus* the posterior; benchmark against it |
| **DOLBY** (Sairam et al. 2024, BEBOP VI, MNRAS) | sum of two Doppler-shifted GPs (wavelength- or CCF-space) | the one live GP/Bayesian effort since PSOAP | aimed at RV precision for circumbinary planets, not spectrum recovery; code availability unverified | adjacent niche; cite and compare |
| **BiSpeD** (González, Martínez & Alejo 2024, A&A 690, A124; Martínez & González 2025, A&A 694, A32; MIT, actively maintained) | SB1: grid over q, subtract primary, cross-correlate residuals vs. templates | owns the low-q (≲0.5) faint-companion regime; pip-friendly | not a general SB2 joint solver; template-grid dependent; grid search, no posterior | overlapping niche covered by our SB1 K₂-scan mode (template-free, marginalized) |

Also on the radar: QER20 (Quintero & Eenens 2024, MNRAS 532, 2604; code unverified) and
single-exposure NN parameter recovery (Binnenfeld et al. 2025) which *bypasses* disentangling.
A targeted search (2026-08) found **no JAX/GPU/differentiable/Bayesian disentangling code** —
the niche is unoccupied.

**The gap all of these leave:** no code offers (a) joint Bayesian orbit + spectra inference
with uncertainties on *both* — the field-wide workaround is bootstrap-over-epochs for orbit
errors and *nothing* for spectra — (b) at survey scale, (c) with modern engineering
(GPU, autodiff, tests, docs, pip). That triple is the product, and (a) is the sharpest edge.

**The differentiator** (math.md §3): conditional on the nonlinear parameters θ (orbit, light
fractions, LSF, response, hypers), the component spectra enter *linearly*. With Gaussian noise
and Gaussian smoothness priors they marginalize **analytically** → NUTS samples only ~10–200
nonlinear parameters regardless of spectrum size; spectra + full covariance are recovered
afterwards as a conditional Gaussian. This is the celerite/starry linear-marginalization trick
applied to disentangling; it is what makes the package categorically better than iterative
shift-and-add rather than incrementally better.

---

## 2. Decisions & recorded defaults

No blocking questions were open after the project brief; the following defaults were assumed
and are the ledger for review. Each is cheap to change now and expensive later.

| # | Decision | Default | Rationale / alternative |
|---|---|---|---|
| D1 | Model grid | uniform in ln λ; pixel `dv` = ~half the finest instrument pixel | shift = translation; oversampling controls interp error |
| D2 | Doppler mapping | relativistic, ξ(v) = artanh(v/c) | exactly antisymmetric ⇒ exact composition/inversion; classical off by ~0.6 km/s at 600 km/s; classical available as option |
| D3 | Shift interpolation | linear (2-tap), zero-fill, on *deviation* spectra d = s − 1 | sparse, exact adjoint; cubic 4-tap behind a flag later (math.md §1.1) |
| D4 | Data resampling | **never resample data**; model is projected to each epoch's native grid (pixel-integral rebin operator) | keeps noise diagonal; mixed-instrument for free |
| D5 | Frames | data frame declared per dataset (`topocentric` default); barycentric correction composed inside the model; tellurics = static component in topocentric frame | one convention, one function, tested |
| D6 | Tellurics | optional additive linear component (first-order approx of multiplicative transmission) | keeps linearity; error ~d★·d_tell, deep tellurics masked anyway; exact treatment is v2 |
| D7 | Response | multiplicative per-epoch Chebyshev, order 2 default, coefficients in θ (build-time constants through M4 — they enter the weights and targets, not just the operator, so their θ-swap is deferred; math.md §7.5) | absorbs normalization errors; low order to protect broad features |
| D8 | LSF | per-instrument Gaussian σ_v (constant-R); tabulated LSF is v2 (banded matrix, no structural change) | mixed-resolution datasets supported from day one |
| D9 | Spectra priors | banded-precision smoothness (τ D₂ᵀD₂ + η I) per component; τ, η fixed/ML-II/sampled | scalable (no dense kernels); nullspace made proper *explicitly* |
| D10 | Marginal-likelihood engine | Strategy A: block-tridiagonal (banded) Cholesky, `lax.scan` + `vmap` over chunks/systems | ~10¹¹ flops at design target ⇒ ~0.1 s/eval on GPU; B (CG+SLQ) for benchmarks; C (frozen logdet) quick-look only (math.md §4) |
| D11 | Sampler | numpyro NUTS via `numpyro.factor` for the marginal likelihood (keeps priors/transforms/summaries); blackjax kept compatible | mature, well-documented, custom log-density path confirmed in numpyro docs |
| D12 | MAP | optax/jaxopt on the same objective; `fit_map()` is the quick-look path | |
| D13 | Light ratio | **no default** — API requires explicit `Fixed` / `Free(prior)` / `PerEpoch` | exact degeneracy with line depths (math.md §5.2); silence would be dishonest |
| D14 | Systemic velocity | γ ≡ 0 internally (spectra in systemic frame); free-γ only with informative prior | γ is unidentified by disentangling (math.md §5.3) |
| D15 | Noise model | diagonal ivar, mask = zero weight, optional per-epoch jitter factor (built in D31) | |
| D16 | Precision | float64 mandatory; enabled at `import albireo` (opt-out env var `ALBIREO_DISABLE_X64` for experiments) | adjoint/logdet tests need it; spectra work needs it |
| D17 | Stack | Python **≥3.12** (deviation from the brief's ≥3.11: current jaxlib 0.11.x ships no cp311 wheels), JAX, numpyro, optax; hatchling; src layout; ruff; pytest; GH Actions (ubuntu+windows, 3.12/3.13); mkdocs-material; BSD-3 | verified against PyPI 2026-08-11 |
| D18 | Package name | `albireo` — **verified free on PyPI** (2026-08-11); register early | fallback `albireo-spectra` if upload is admin-blocked |
| D19 | Versioning | SemVer, `0.1.0.dev0` now; public API = documented API | |
| D20 | Orbit parameterization | sampled sites (M3): `period`, `t_conj`, `secosw = √e cos ω`, `sesinw = √e sin ω`, `k` (vector); user-chosen numpyro priors supply the scale transforms (e.g. LogNormal for P, K); disk constraint e < 1 via −∞ factor with e clipped at 0.95 (Kepler solver's verified range); γ ≡ 0 (D14) | √e-pair smooth through e = 0 and uniform-disk ⇒ e ~ U(0,1); t_conj well-defined for circular orbits (math.md §7.2) |
| D21 | Static solver bandwidth | user declares `v_rel_max_kms`; bandwidth bound fixed at build time (jit-static); numpyro model adds a **bandwidth guard** (−∞ when realized shifts exceed the budget) | probing with too small a band is *silently* wrong — the guard makes wide priors safe by construction (math.md §7.1) |
| D22 | Spectral hyperparameters (τ, η) | empirical Bayes default: MAP jointly over (θ, log τ, log η) = ML-II (spectra already marginalized); NUTS runs with hypers conditioned at ML-II values; full sampling available by putting them in the priors dict | the prior scale *is* the answer for sub-LSF modes (math.md §5.1, §7.3) — it must be estimated deliberately, and the marginal's log det Λp Occam term makes ML-II well-posed |
| D23 | SB3 = hierarchical nested Keplerians | optional θ sites `period_out`, `t_conj_out`, `secosw_out`, `sesinw_out`, `k_out = (K_AB, K_C)` (all five or none); inner components ride the outer center-of-mass RV, tertiary opposes with ω_out + π; outer light-time effect neglected (v2 seam) | matches fd3's triple topology; the tertiary is just one more linear component (math.md §7.5) |
| D24 | Light fractions in θ | optional `light` site: `(n_stellar,)` constant or `(n_epochs, n_stellar)` per-epoch, rows on the simplex (Dirichlet priors); swaps into the static graph like the shifts | ℓ enters A(θ) linearly; the D13 eclipse breaker becomes *inferred*, closing the loop on math.md §5.2 |
| D25 | LSF widths in θ | optional `lsf_sigma` site (one per instrument, model-group order); construction-time widths fix the kernel radii and act as strict upper bounds (−∞ guard above); **absolute widths require a reference-instrument anchor** (tight prior) — only relative widths are data-identified | template-free models cannot split LSF from intrinsic line width (measured: all-free ML-II inflates widths ~tens of %, orbit unaffected); same honest-anchor policy as D13 |
| D26 | K₂-scan mode | `scan.k2_scan`: null = single-component model (ℓ₁ = 1, same primary prior); companion ℓ₂ explicit (D13); D(K₂) calibrated by injections, no χ² claim — Occam term makes D < 0 under the null (verified) | math.md §6; the marginalized matched filter is the BiSpeD/shift-and-add replacement |
| D27 | Scale architecture (M5) | `Problem` is a registered pytree passed as a jit *argument* (closure-captured arrays trigger multi-GB XLA constant folding at scale); probing runs as a **sequential scan** over probe batches with per-block gather readout — unrolled batches let XLA overlap live ranges (~80 GB temp at 2×10⁵ px, measured) and un-remat scans store every batch for the backward pass (~470 GB); batch size and backward rematerialization are **size-adaptive** (large batch / no remat below a ~64 MB probe-output footprint — remat measurably doubled gate-scale NUTS for zero benefit — small batch / remat above it); the prior factor gets its own small block size | benchmarks.md M5; `probe_chunk`/`remat` stay exposed as the CPU/GPU tuning knobs |
| D28 | Direct band assembly + closed-form gradient (M5 speedup pass) | the posterior precision is assembled **per epoch** from its analytic band structure — each epoch's (i,j) block is ℓᵢℓⱼ·T(δᵢ)ᵀ·(KᵀRᵀW′RK)·T(δⱼ), a band of width ~2·(rebin support + 2·kernel radius) at a velocity-set offset — via static rebin *pair tables* (`operators.rebin_pair_tables`, stored on `EpochGroup`), a vmapped kernel-sandwich pre-pass, and a `lax.scan` accumulating tent-weighted `dynamic_slice` translations into a band tensor (`assembly.band_block_tridiagonal`; math.md §4.5). Replaces 2p+1 global probe matvecs with O(band width) work per epoch (~8× eval measured); comb probing is retained as the reference path (`assembly="probe"`) and as the `validate=True` oracle. The solve stage carries a **custom VJP** (`likelihood._solve_stage`): cotangents from the block-Takahashi banded selected inverse (`solver.selected_inverse_blocks`) and d̂-outer-products, removing reverse-mode through the Cholesky/solve scans; gradients flow through `log_likelihood`, `quad`, and `d_hat` but deliberately **not** through the returned Cholesky factor (documented contract). Row translations use clip-safe `dynamic_slice` on double-padded arrays — exact zero-fill for any shift, and the reverse pass is a contiguous copy instead of the scatter a gather-based translation would pay. Second derivatives: reverse-over-reverse only (the fwd rule recomputes its primal inline so `jacrev(jacrev(...))` stays exact; forward mode is rejected by `custom_vjp`, and `jax.hessian`'s forward-over-reverse was independently measured asymmetric on this stack even for plain autodiff — `laplace_inverse_mass` switched to rev-over-rev, fixing a defect present since M3). Remaining levers on the ledger, not implemented: analytic VJP through the assembly itself, custom-VJP band→block packing, associative-scan (parallel-prefix) Cholesky for GPU chain-latency, opt-in mixed precision, fusing the Takahashi sweep into cotangent assembly (design-target grad is memory-bound at 32 GB) | benchmarks.md M5 speedup pass; band-vs-probe and VJP-vs-autodiff equality are regression-tested in tests/test_assembly.py |
| D29 | Memory pass + boundary correctness (post-D28) | The design-target gradient measured **34.0 GB** against 32 GB of RAM — it did not fit. Four reductions, all exact: (1) the Takahashi sweep now emits the cotangent blocks directly (`solver.selected_inverse_cotangent`), so the selected inverse is never materialized; (2) the velocity-independent `G` pre-pass is **batched over epochs** (`epoch_chunk`) — `vmap` batches every intermediate of the chain, not just the result, so the un-batched pre-pass alone was ~9 GB; batching costs one extra `G` pass in the backward and is therefore applied only above a ~1 GB threshold, leaving small and gate-scale problems on the fast path; (3) the prior determinant uses a scalar pentadiagonal recursion (`assembly.prior_logdet`) instead of a block-tridiagonal factorization of a bandwidth-2 matrix; (4) `_pack_band`'s gather indices are hoisted out of the `lax.map` body (they never depended on the block counter) and the body is rematerialized, so reverse mode stops stacking (B,B) index and mask arrays over all K blocks. Result at the design target: eval 20.6 → 11.1 GB, gradient 34.0 → 18.2 GB, and — because the gradient had been thrashing against RAM — eval 26.0 → 22.2 s and gradient 111.3 → 87.4 s with it. (Mid-ladder gradients pay 5–23% for the batched pre-pass's extra backward `G` pass, where memory was not the binding constraint; `epoch_chunk` trades it back.) **Correctness fixed in the same pass**: the LSF convolution wrote `G` band entries at column indices outside the model grid and the T-sandwich read them, because only the sandwich's *row* index was zero-filled. Trigger: data-coverage margin (in model pixels) below the kernel radius — which includes choosing a model grid narrower than the observed range to fit a sub-region, and is reachable at a *fixed* grid just by fitting a wider LSF. At zero margin the assembled precision was wrong by 6.8e-2 relative and **asymmetric** by the same amount (probing is symmetric), worth −57 nats of log-likelihood. Every fixture happened to leave a margin exceeding the kernel radius, so the weights vanished where the defect lived. `G`'s out-of-grid columns are now masked at the source. Also: the Cholesky factor left `_solve_stage`'s outputs — a cotangent on it cannot be honoured by the closed-form rule, so `spectra_std`/`draw_spectra` gradients were silently **zero**; `MarginalResult` now stores the precision and rebuilds the factor outside the custom boundary (one extra Cholesky, paid only by callers that want it). `build_problem` warns when a few anomalously wide native pixels set `row_support` — deleted telluric/order gaps rather than masked ones, which inflates the solver bandwidth quadratically | benchmarks.md "D29 memory pass"; the boundary case is now a fixture (`edge_covered`) compared **entrywise** against probing, since the log-determinant and solve it previously guarded average the defect away |
| D30 | Real-data ingestion path (first observed dataset: HR 6819) | Two new modules and four fixes, driven by putting 51 archival ESO Phase-3 FEROS spectra through the stack. **`albireo.preprocess`** (pure NumPy, like `data.py`): `fit_continuum` / `normalize` — an asymmetric-envelope-then-asymmetric-sigma-clip penalized fit **of `log(flux)` on a knot grid**, because a continuum is multiplicative (the measured FEROS response falls 20× over 3850–4750 Å, which a curvature penalty in linear space cannot track — normalized flux came out 30% wrong) and because a per-pixel Whittaker smoother needs `lam ~ (L_px/2π)^4 ~ 1e12` at echelle smoothing lengths, at which the weight term is lost to rounding and the SPD factorization *fails outright*; knots at 8 per smoothing length hold `lam ~ 2.6`. `estimate_ivar` — DER_SNR (Stoehr et al. 2008, ESO's own recipe) in wavelength bins, fitted to `sigma^2 = s^2/continuum`, because the ESO FEROS `ERR` column is **entirely NaN**. `select_region` / `mask_ranges` / `mask_tellurics` / `mask_spikes` — masking is always `ivar = 0`, never deletion (D29's row-support trap). `share_wavelength_grid` — index-aligned (never value-matched: a `searchsorted` at a window edge slips a whole 1.4 km/s pixel) relabelling of per-exposure grids onto one, asserting the residual is within tolerance; a *relabelling*, not a resampling, so D4 stands. **`albireo.io`** (optional `astropy`, `[io]` extra): `read_spectrum` for ESO Phase-3 binary tables and WCS image spectra, resolving frame from `SPECSYS`, `v_bary` from the pipeline's own keyword (`ESO DRS BARYCORR` and friends — the pipeline's value is what *defines* the delivered frame; astropy only as fallback, agreement measured at 0.017 km/s), and BJD_TDB at mid-exposure from the extension-header `TMID`; every assumption it has to make emits a warning naming it. Plus `LogGrid.covering`, which sizes the model-grid margin as shift + kernel radius + slack. **Fixes found by the data, all with regression tests:** (1) `build_problem` formed `z = flux - r·base` unmasked, so a non-finite flux at a zero-weight pixel — which `data.py` explicitly *documents as legal* and which `normalize` produces at a collapsed continuum — took the whole marginal likelihood to `nan` via `0 * nan`; (2) epochs of one instrument had to share a bit-identical `wave` array or `build_problem` raised, but a pipeline that shifts before resampling gives one grid per exposure (51 spectra → 28 grids), so grouping is now by (instrument, grid) with the instrument key still keying the LSF, and `_epoch_chunk_default` divides its byte budget by the group count; (3) `prior_logdet`'s Cholesky pivot returned `nan` below `eta/tau ~ 1e-13`, reachable by unbounded ML-II, now floored; (4) the row-support warning seeded its minimum with `int64` max and never restricted to rows the operator touches, so a region-selected epoch produced a warning quoting nonsense. `run_map` gained a `callback` (it was silent for hours, and its absolute `tol` is unreachable at survey pixel counts). **Not fixed, recorded:** light fractions remain an input for a non-eclipsing system (D13 — the K-band GRAVITY ratio does not transfer to 4400 Å), there is still no jitter site (D15) so estimated inverse variances must be checked against `data_residual_zscores` (**built afterwards — see D31**), and the numpyro path captures `Problem` as a closure constant rather than a jit argument, so it pays the XLA constant folding D27 avoids for `MarginalOrbitModel.marginal` | benchmarks.md "HR 6819"; `tests/test_preprocess.py`, `tests/test_io.py`, `examples/03_hr6819_real_data.py` |
| D31 | Jitter site built (closes D15) | `EpochGroup` carries a per-epoch factor α_j; **every consumer of the weights reads `effective_w = w/α²`, none reads `w`**, so the inflation reaches the normal matrix, the right-hand side and the `Σ log w` term together — which is exactly what makes α identifiable instead of a free knob. Swapped in by `forward.with_jitter` (scalar or per-epoch, traced-safe, *replaces* rather than compounds, since the raw `w` is kept) and exposed as the optional θ site `log_jitter`. **What the marginal actually estimates:** in the data-dominated limit `−½ log det(Λ + AᵀWA)` contributes `+p_eff·log α` against the weight term's `−N·log α`, so profiling gives `α² = χ²/(N − p_eff)` with `p_eff = tr[(Λ + AᵀWA)⁻¹AᵀWA]` — the *effective-dof-corrected* variance estimate, not `χ²/N`. The naive route (whiten the residuals, read off their standard deviation, rescale ivar) is low by `√(1 − p_eff/N)` — 4.6% in the weak-prior fixture. **How much it matters is a property of the run, and `p_eff` is emphatically not `n_comp · n_pix`**: on HR 6819 an oversampled grid plus a fitted smoothness prior gave `p_eff ≈ 2900` against 19,876 model pixels — about the *resolution-element* count — so there the correction was 0.4%. Inverting the two estimators, `p_eff = N[1 − (sd/α̂)²]`, is a free and otherwise awkward diagnostic of how much of the spectrum the data actually constrain. Regression-tested by injecting a known ivar scale error and recovering it to 0.8%. **Deliberate limits, documented at the function:** a diagonal inflation cannot represent a residual correlated across pixels, so a jitter fitted against systematics (continuum, LSF mismatch, line-profile variability) reports a *wider but still biased* orbit; `data_residual_zscores` now whitens by the assumed weights, jitter included, so after fitting one it reads ≈1 by construction and the diagnostic that still bites is the residual's shape, not its scale. **Measured on HR 6819, and the result is a warning, not a win**: per-epoch α runs 1.11–3.61 (the worst exposure ends up with 1/13 of its nominal weight, and it buys +19,763 nats over one shared α), the residuals whiten to sd 0.997 exactly as advertised — and the period *moves by 174× the no-jitter formal error*, away from the published value, because the noisiest exposures cluster in the first third of the baseline (`corr(α, phase) = −0.25`) and downweighting them removes the period leverage at one end. Both fits are genuine optima under their own weights (cross-evaluated: +10,622 nats one way, +3,835 the other), so this is the noise model *selecting* the optimum, not a defect. Window-to-window disagreement in period went from 5.6× to 13.1× the combined formal error. The standing lesson gains a clause: quote the spread across independent windows **and across defensible noise models** | `tests/test_jitter.py`; math.md §3.2a; benchmarks.md "The jitter site" |

---

## 3. Data model

The core takes arrays; instrument I/O is out of scope (one thin loader utility only).

```
EpochData:                      # one observed spectrum
    wave        (N_j,) float64  # native wavelength grid, Å (vacuum or air — declared, not guessed)
    flux        (N_j,) float64  # continuum-normalized
    ivar        (N_j,) float64  # inverse variance; 0 = masked
    mask        (N_j,) bool     # optional convenience; folded into ivar
    bjd         float           # BJD_TDB mid-exposure
    v_bary      float           # barycentric correction velocity [km/s]
    instrument  str             # key into the LSF/response tables

Dataset: list[EpochData] + frame flag ("topocentric" | "barycentric"), validation, summary stats
```

Internally the dataset is packed into padded/ragged JAX arrays batched by instrument. Chip
gaps, cosmics, interstellar lines are all just `ivar = 0` — no special cases anywhere
downstream (weights/masks are honored in every operator and every solve).

## 4. Model & inference architecture

One equation summary (derivations in math.md §1–3): per epoch,
`m_j = diag(r_j) · R_j · [1 + Σ_i ℓ_ij · B_j · T(δ_ij) · d_i]`, i.e. shift → LSF-convolve →
rebin-to-native → response, linear in the stacked deviation spectra `d`. Marginalize `d`
analytically → sample θ with NUTS → recover `p(d | y)` as a mixture of conditional Gaussians
over posterior θ draws.

**Parameter split.**
- Linear (marginalized): component deviation spectra `d_i` — ~10⁵–10⁶ numbers we never sample.
- Nonlinear θ (sampled): orbit (~5–8), light fractions (0–N_c·J), LSF widths (per instrument),
  response coefficients (J·(order+1)), jitters (≤J), prior hypers (≤2·N_c). Typically 20–250.

**Modes.** (all share one code path — a mode is a choice of velocity law + what's fixed)
1. SB2/SB3 Keplerian joint inference (flagship).
2. Free per-epoch velocities (diagnostic).
3. Fixed-orbit linear solve (M2 milestone; also the "give me spectra now" utility).
4. SB1 + faint companion K₂ scan → Δlogℒ detection maps + recovered secondary spectrum
   (math.md §6), vmapped over the grid.

**Outputs** (`FitResult` / `PosteriorResult`): θ posterior (arviz-compatible), conditional
spectra draws + pointwise bands + banded covariance info, whitened residual diagnostics,
degeneracy report (low-k variance spectrum, response–spectrum covariances), sensitivity
forecast utility.

## 5. Degeneracy policy

Derivations in math.md §5. Policy: **never silently regularize away a real degeneracy** —
make it proper with an explicit prior scale, report it, and where only external information
helps, require the user to choose.

| Degeneracy | Policy |
|---|---|
| Low-frequency mode exchange ("undulations") | proper priors with documented scale; posterior covariance reports inflation; `sensitivity_forecast()` for observing strategy |
| Light ratio ↔ line depth | **mandatory explicit choice**: `Fixed` / `Free(prior=...)` / `PerEpoch(eclipse model)`; docs explain what breaks it |
| γ ↔ common shift | γ ≡ 0 default; post-hoc template measurement; free-γ requires informative prior |
| Response ↔ broad features | low default order; covariance reported |

## 6. API sketch (target user code)

```python
import albireo as ab

ds = ab.Dataset(
    [ab.EpochData(wave=w, flux=f, ivar=iv, bjd=t, v_bary=vb, instrument="HERMES"), ...],
    frame="topocentric",
)

dis = ab.Disentangler(
    dataset=ds,
    grid=ab.LogGrid.from_wavelength_range(4000.0, 6800.0, dv_kms=1.0),
    components=[ab.Star("A"), ab.Star("B"), ab.Telluric()],
    orbit=ab.Keplerian(
        period=ab.Normal(11.55, 0.01),  # priors are first-class
        t_peri=...,
        ecc=...,
        omega=...,
        k1=...,
        k2=...,
    ),
    light_ratio=ab.FixedLight([0.62, 0.38]),  # explicit, or Free(prior=...) / PerEpoch(...)
    lsf={"HERMES": ab.GaussianLSF(sigma_v_kms=2.4)},
    response=ab.Chebyshev(order=2),
    spectral_prior=ab.Smoothness(tau="ml2", eta=1e-4),
)

quick = dis.fit_map()  # MAP + conditional spectra, seconds
post = dis.sample(num_warmup=1000, num_samples=1000, seed=0)  # NUTS over θ
spec = dis.conditional_spectra(post)  # draws, bands, covariance
rep = dis.degeneracy_report(post)

# diagnostic + detection modes
free = dis.replace(orbit=ab.FreeVelocities()).fit_map()
scan = ab.K2Scan(dis, k2_grid=np.linspace(1, 150, 300)).run()  # SB1 workflow
```

## 7. Package layout & engineering

```
src/albireo/
    __init__.py        # x64 enforcement, public API re-exports
    grids.py           # LogGrid, doppler log-shift mapping          [M0 — this session]
    operators.py       # shift/interp/rebin linear ops + adjoints    [M0 — this session]
    kepler.py          # differentiable Kepler solver, velocity laws [M1]
    simulate.py        # simulator = test harness                    [M1]
    forward.py         # epoch forward model assembly                [M2]
    solver.py          # banded/block-tridiag Cholesky, logdet, solves, sampling [M2]
    likelihood.py      # marginal likelihood (strategies A/B/C)      [M2–M3]
    priors.py          # spectral priors, parameter priors           [M2]
    inference.py       # MAP (optax), NUTS (numpyro), results        [M3]
    scan.py            # K2-scan mode                                [M4]
    data.py            # EpochData/Dataset, validation, thin loader  [M1]
    preprocess.py      # continuum, ivar estimation, region/masking, grid sharing [D30]
    io.py              # FITS -> RawSpectrum -> EpochData/Dataset (astropy) [D30]
tests/                 # mirrors modules; simulator-driven closed-loop tests
docs/                  # mkdocs-material; design.md, math.md, tutorials (executable, in CI)
```

Standards (per brief): JAX with x64; CPU-first correctness, GPU as accelerator; type hints +
docstrings on all public API; pytest with closed-loop tolerance tests, adjoint tests for every
linear operator, gradcheck vs. finite differences; GitHub Actions CI on ubuntu + windows;
pre-commit (ruff, ruff-format); BSD-3; SemVer. Performance work is benchmark-gated: no
optimization without a recorded baseline in `docs/benchmarks.md` (started at M2).

Test philosophy: the simulator (M1) is the oracle — every inference feature ships with a
closed-loop test against known injected truth, and calibration claims (M3+) are backed by
SBC/coverage runs, not asserted.

## 8. Milestones

| | Deliverable | Acceptance gate |
|---|---|---|
| **M0** | scaffolding, design.md, math.md, grid + shift/resample operators with adjoint/gradcheck/flux tests | **this session — stop for review** |
| M1 | simulator (templates, orbit, LSF, noise, tellurics, response) | simulated datasets reproduce all advertised pathologies (gaps, mixed instruments) |
| M2 | fixed-orbit linear solver + uncertainties | <1% RMS recovery in line regions on SNR-100/30-epoch sims; whitened residuals ~N(0,1); marginal likelihood matches dense brute force on small problems; nullspace analysis documented |
| M3 | joint NUTS inference + MAP pipeline | SBC/coverage pass on injections; K₁,K₂ to <1% at design SNR |
| M4 | tellurics, SB3, per-epoch light, multi-instrument LSF, K₂-scan | closed-loop tests per feature |
| M5 | benchmarks vs. fd3, one real published SB2 end-to-end, docs polish, JOSS checklist | definition of done: 2×10⁵ px / 50 epochs samples in minutes on one GPU; tutorials run in CI. Candidate headline case: an LB-1/HR 6819-type system, where the posterior exposes the hand-set-light-ratio systematic that drove the published debate |

## 9. Non-goals (v1) and v2 seams

Not in v1: GUI; Fourier-space methods; time-variable component spectra (pulsations/spots);
emission-line-specific models; apsidal motion; exhaustive instrument I/O.

Designed seams for v2: components carry their own velocity-law and (future) time-dependence
interface — a pulsating component would subclass the component protocol without touching the
solver; LSF is an operator slot (Gaussian → tabulated is a swap); the Keplerian core is a
standalone module shared with a future differentiable eclipsing-binary package for joint
light-curve + RV + disentangling fits; strategies A/B share one solver interface so new
structured solvers slot in.
