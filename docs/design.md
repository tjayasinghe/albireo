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
| D7 | Response | multiplicative per-epoch Chebyshev, order 2 default, coefficients in θ | absorbs normalization errors; low order to protect broad features |
| D8 | LSF | per-instrument Gaussian σ_v (constant-R); tabulated LSF is v2 (banded matrix, no structural change) | mixed-resolution datasets supported from day one |
| D9 | Spectra priors | banded-precision smoothness (τ D₂ᵀD₂ + η I) per component; τ, η fixed/ML-II/sampled | scalable (no dense kernels); nullspace made proper *explicitly* |
| D10 | Marginal-likelihood engine | Strategy A: block-tridiagonal (banded) Cholesky, `lax.scan` + `vmap` over chunks/systems | ~10¹¹ flops at design target ⇒ ~0.1 s/eval on GPU; B (CG+SLQ) for benchmarks; C (frozen logdet) quick-look only (math.md §4) |
| D11 | Sampler | numpyro NUTS via `numpyro.factor` for the marginal likelihood (keeps priors/transforms/summaries); blackjax kept compatible | mature, well-documented, custom log-density path confirmed in numpyro docs |
| D12 | MAP | optax/jaxopt on the same objective; `fit_map()` is the quick-look path | |
| D13 | Light ratio | **no default** — API requires explicit `Fixed` / `Free(prior)` / `PerEpoch` | exact degeneracy with line depths (math.md §5.2); silence would be dishonest |
| D14 | Systemic velocity | γ ≡ 0 internally (spectra in systemic frame); free-γ only with informative prior | γ is unidentified by disentangling (math.md §5.3) |
| D15 | Noise model | diagonal ivar, mask = zero weight, optional per-epoch jitter factor | |
| D16 | Precision | float64 mandatory; enabled at `import albireo` (opt-out env var `ALBIREO_DISABLE_X64` for experiments) | adjoint/logdet tests need it; spectra work needs it |
| D17 | Stack | Python **≥3.12** (deviation from the brief's ≥3.11: current jaxlib 0.11.x ships no cp311 wheels), JAX, numpyro, optax; hatchling; src layout; ruff; pytest; GH Actions (ubuntu+windows, 3.12/3.13); mkdocs-material; BSD-3 | verified against PyPI 2026-08-11 |
| D18 | Package name | `albireo` — **verified free on PyPI** (2026-08-11); register early | fallback `albireo-spectra` if upload is admin-blocked |
| D19 | Versioning | SemVer, `0.1.0.dev0` now; public API = documented API | |
| D20 | Orbit parameterization | sample in (ln P, T, √e cos ω, √e sin ω, ln K_i); T_peri and T_conj both supported | standard well-conditioned choices |

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
