# Roadmap

This page records what albireo intends to become and why, in enough detail that the ordering can be
argued with. It is a plan, not a promise: items move, and items that turn out to be answering a
question nobody asked get deleted. Decisions that have actually been *made* live in the
[decision ledger](design.md#2-decisions-recorded-defaults); this page is upstream of that.

## Where albireo sits

Three facts about the surrounding ecosystem set the strategy.

**The technique is mature; the software is not.** Spectral disentangling has been in use since
Simon & Sturm (1994) and Hadrava (1995), and the codes that implement it are, without exception,
either unmaintained or awkward to obtain. KOREL is Fortran, last released 2011, and is not
distributed as source at all — it is reachable only through a registered web service with a
16,384-bin ceiling. FDBinary is marked deprecated by its own ASCL entry; fd3, its successor, is
still used in papers published this year, and its distribution page has not been updated since 2014
and states no license. Spectangular's last papers are from 2017 and 2019. UNWIND, released in 2026,
is IDL, and its own paper describes it as lacking a manual or an installation package. The
shift-and-add code used to identify the companions in LB-1 and HR 6819 is a small research script
with no license file, which is a real barrier to anyone who would build on it. Nobody in this list
is doing anything wrong; disentangling is a means to an end for all of them, and the software is a
by-product of the science. But it means there is no maintained, installable, tested implementation
of a technique that, in the three years to mid-2026, roughly eight papers a year named in their
*abstract* alone — and many more used in a methods section.

**The one thing everybody wants is uncertainties on the disentangled spectra, and no existing code
provides them.** The disentangled spectra are almost never the final product: they are fed to an
atmosphere code to get effective temperatures, gravities, and abundances. The uncertainties on
*those* quantities are therefore systematically understated across the literature, and the papers
say so explicitly — TMBM III notes that normalization uncertainty is simply not propagated into the
quoted properties. This is not an oversight that better bookkeeping fixes. Disentangling is an
ill-conditioned linear inverse problem with a genuine low-frequency null space, and getting an
honest error band out of it means having a posterior, which means either sampling something very
high-dimensional or marginalizing it away. albireo does the second. That capability is the reason
the package exists, and everything on this page is downstream of making it usable.

**The obvious approach was tried, and it died of compute.** PSOAP (Czekala et al. 2017) put a
Gaussian-process prior on the component spectra and sampled the joint posterior. It is the closest
thing to a direct predecessor, and its last commit is from December 2017. The paper is candid about
why: the dense GP cost restricted it to narrow bandpasses and required cluster access. albireo's
analytic marginalization plus banded structure plus JAX is a response to exactly that failure — the
component spectra are integrated out in closed form rather than sampled, and the linear algebra is
banded rather than dense. Being the package that made this tractable is the positioning; keeping
that claim honest is the reason the [benchmark record](benchmarks.md) is as long as it is.

What follows is ordered by that logic: make the existing capability usable (Tier 1), then reach the
communities whose problems it already solves (Tier 2), then extend the model (Tier 3).

## Tier 1 — make it usable (done)

A capability nobody can install, cite, or plot is not a capability. These items added no new
science and were the highest-leverage work available. They are **done**; what is left of this
tier is not code but pushing, tagging and registering, and that sequence — where the ordering
genuinely matters — is written down in [Releasing](releasing.md).

| | Item | Why it came first |
|---|---|---|
| 1 | `slow` / `network` / `gpu` markers and a `conftest` | Everything below adds tests; the fast suite went from 14 minutes to 6 |
| 2 | Single-sourced version, upgraded `CITATION.cff`, `CHANGELOG.md`, a [citing](citing.md) page, a tag-driven release workflow | Nothing could say `pip install albireo` until this was true |
| 3 | Deployed docs with a rendered [API reference](api/index.md), `py.typed`, coverage | 70 exported names with good docstrings, rendered nowhere |
| 4 | `load_example()` and a [five-minute quickstart](quickstart.md) that runs offline | Time-to-first-plot is the best-attested adoption lever there is |
| 5 | `albireo.results`: arviz conversion, save/load, export to FITS/ECSV/ASCII | An hours-long fit used to produce printed text and nothing else |
| 6 | `albireo.plotting` | The three most reusable plotting functions in the repo were stranded inside example scripts |

Two of these deserve their reasoning recorded rather than asserted.

**Export is not a convenience feature.** The disentangled spectrum plus its uncertainty band is the
product. Until it could leave the process in a format an atmosphere code reads, albireo was the
middle of a pipeline with no downstream connector, and the uncertainty — the entire differentiator
— was dropped at exactly the joint where it matters. See Tier 2 item 8 for the other half of this.

**The quickstart must not require finding data.** The pattern worth copying is lightkurve's: the
first code block does science, not configuration, on data the user did not have to locate. So a
small simulated dataset ships inside the wheel — no download, no astropy, no archive account — and
a real archival example is the second step, once the ESO loader below exists.

## Tier 2 — reach the communities

Each of these is aimed at a specific group of people with a specific unmet need, ordered so that
earlier items unblock later ones.

### 1. A nebular component with a free per-epoch amplitude — **done** (D40)

Massive stars are born in H II regions, so their spectra carry nebular emission lines that do not
move with either star and vary in strength from night to night with seeing and slit losses. Left in,
they artificially narrow the disentangled line profiles and bias the derived temperatures and
gravities. The literature's handling of this is a sequence of one-off workarounds: TMBM III models
the nebular features as a third component with static velocity and variable intensity, built by
hand; a 2026 paper in the same series masks the contaminated pixels by setting their error values to
999.

In albireo this is a static component whose per-epoch amplitude is a free parameter — structurally
the telluric component with a traced amplitude instead of a fixed one, and physically correct, since
nebular flux is added on top of the total continuum and takes no light from the stars. Two details
carried the work, both as predicted: the nebular component is static in the *barycentric* frame,
which is the opposite convention from the telluric one, and confining it to the Balmer, He I and
forbidden-line windows needed the smoothness prior's regularization strength to become per-pixel
(`SmoothnessPrior(tau_profile=, eta_profile=)`, built by `priors.window_profile`).

It was the highest impact per line of code on this page, and the closed loop says by how much: an
SB2 whose Hβ absorption carries a static nebular line disentangles to a core **26% too shallow** and
an equivalent width **11.5% low** without the component, against 0.14% with it. Equivalent width is
what reaches the atmosphere code, so that is a systematic error in log *g* — of the kind nothing in
the current literature propagates. The orbit is worse and was not expected: a static line is a
component with *K* = 0, so a nebula-blind joint fit hands the emission to whichever star can be
made to move least, returning K₂ **59% low**, a period long by 0.171 d, and a circular orbit at
*e* = 0.95 (the solver's clip). The contamination reaches the masses, not only the atmospheres. Two degeneracies came with the component and are closed by
convention rather than by data (the amplitude scale, pinned by centering the log-amplitudes; the
nebular velocity, which is a placement convention for the window profile and not a measurement);
both are recorded in [the math](math.md#13-lsf-light-fractions-response).

### 2. Calibrated faint-companion detection — **done** (D41)

The K₂ scan already answered "is there a companion, and at what velocity semi-amplitude" — the
question at the center of every dormant-black-hole candidate. What it did not do is say how
often noise alone would produce the peak it found. The state of the art in the literature is
a bare χ² map with no false-alarm probability attached, and the papers are explicit about the
resulting fragility: small deviations in the assumed primary semi-amplitude produce spurious
features in the recovered secondary spectrum.

All three pieces are built. The scan is **vectorized** — `MarginalOrbitModel.log_likelihood_sweep`
runs the trial grid as one batched `lax.map` instead of a Python loop with a device
synchronization per point. **K₁ is marginalizable** (`k2_scan(k1_sigma=)`) over a Gauss–Hermite
rule applied to both the companion and the no-companion model, so `D` stays a ratio of two
marginal likelihoods. And `albireo.calibrate.detection_limit` runs the **injection–recovery
calibration**, resimulating through the observed data's own operators
(`forward.with_data`, `simulate.resimulate`) so that a few hundred scans cost under a minute.

The K₁ result was sharper than expected and reframes why marginalization matters: a K₁ 10% high
did not merely blur the answer, it took the recovered companion's line pattern from **0.96
correlation with truth to 0.49 while more than tripling `D`** — the artifact reads as a *stronger*
detection. A calibrated threshold cannot catch that, because the null trials are drawn under the
same wrong assumption; only the marginalization can. The two are complementary, which is now
stated wherever either is documented. The calibration itself delivers the quotable sentence —
"any companion contributing more than *X*% of the light would have been detected at 95%
confidence" — with a threshold that is conservative by construction and a false-alarm probability
that never claims finer resolution than `1/(n_null + 1)`.

One expected dependence turned out not to be there, which is worth knowing before anyone spends
compute on it: the limit is nearly **flat in K₂** (0.292 / 0.296 / 0.297% at K₂ = 20 / 40 / 65
km/s). In an SB2 the two components move in antiphase, so their relative velocity never drops
below about K₁, and with K₁ = 55 km/s the pair is well separated at every trial. A real K₂
dependence should appear only when K₁ is itself small. The dependence that *does* remain is on
the assumed companion template — the observable is ℓ₂·d₂, so a featureless companion is
invisible at any light fraction — and that assumption is required to be quoted with the number.
The [HR 6819 campaign](benchmarks.md) remains a ready-made validation set.

### 3. A per-epoch radial-velocity table — **done** (D42)

Free per-epoch velocities — no Keplerian — was listed in the [design](design.md) as a diagnostic
mode and had never been exposed. The low-level machinery existed and was differentiable; only the
sampling path was missing. It is now a `velocity` theta site that *replaces* the orbit, with
`relative_velocities`, `relative_velocity_errors` and `keplerian_residuals` alongside it.

Two reasons it was worth more than its size, both borne out: a per-epoch RV table with honest
uncertainties is the artifact the binary-star community expects from any spectroscopic analysis,
and it is the natural bridge for users of cross-correlation and shift-and-add codes, who can
compare against something familiar before trusting the joint fit. It is also the model check for
the Keplerian mode — fit free velocities, then ask whether a Keplerian threads the resulting
posterior.

The honest caveat in the roadmap was right but understated it. The table is differential, and it
has **one arbitrary zero point per component, not one in total**: with no orbit tying the stars
together, each free spectrum absorbs a constant added to its own shifts. Worse, left uncentered
that zero point is pinned by *shift-interpolation error* rather than by data — a whole-pixel
common shift costs 4e-9 of the log-likelihood, a 0.1-pixel one costs 7.3 nats. albireo removes it
in pixel space, where the removal is exact. The same projection turned out to be needed for the
*uncertainties*: the raw Laplace diagonal returns the prior and nothing else (37.95 km/s = 120/√10
on every entry, against a real per-epoch error of 0.059 km/s), which is the kind of number that
looks equally convincing on a good dataset and a useless one.

Measured: warm-started from a Keplerian 30% wrong in both semi-amplitudes, per-epoch RVs recover to
**0.098 / 0.066 km/s** — 1/60th of a model pixel — and the Wilson slope to 0.4%. From a cold start
the mode fails, as it must, but at a potential 122,000 nats worse, so the failure is visible.

`examples/09_rv_table.py` is the worked script, and it is built around the two properties that
are genuinely counter-intuitive rather than around the API: it demonstrates the per-component
zero point by shifting one star's velocities by 50 km/s and watching the log-likelihood not
move (0 nats for the relativistic shift, 8.7e-6 for the ordinary one that is only its
first-order approximation), and it prints the raw Laplace bars beside the projected ones —
37.947 km/s on every entry, which is `120/√10`, the prior, against 0.056-0.065 measured. The
narrative material is in [the math](math.md#76-free-per-epoch-velocities-the-rv-table) and
`tests/test_velocity_table.py`.

### 4. An ESO archive loader, and BLOeM in one line — **done** (D44, D45)

The reason to do it was BLOeM: ~929 targets in the Small Magellanic Cloud with roughly 25 epochs
each, an intrinsic binary fraction above 70%, and 59 published double-lined systems — all public,
and the survey team's own disentangling of them still listed as future work in their July 2026
review. A short fetch of a real BLOeM target, disentangled with posteriors, is simultaneously the
tutorial, the evidence of research use that software journals ask for, and a paper. It was sequenced
after the nebular component deliberately: BLOeM's targets sit in nebulosity, so the demo needs that
component to be honest.

**Both halves are built.** `albireo.archive` (D44) is the ObsCore/TAP client and resumable
downloader; `albireo.io` (D45) now reads what it fetches by dispatching on the **IVOA utypes**
rather than on column names, and `resolve_bloem` / `bloem_catalogue` / `bloem_spectra` turn a
survey identifier into that star's epochs. `ab.bloem_catalogue(binary_class="SB2")` returns the 59
double-lined targets; `examples/06_bloem.py` takes one from name to fitted spectra.

**"A single file layout" was wrong, and that is the useful part.** Thirteen real Phase 3 spectra
across seven instruments were read column by column, and no two collections agree on anything except
the utypes — flux is `FLUX`, `FLUX_REDUCED`, or both at once; the extension is `SPECTRUM` except
Gaia-ESO's `phase3spectrum`; units are angstrom, Angstrom or nm. The obvious fix, keying on UCDs, is
worse than the disease: UVES gives its **sky-background** column the same UCD HARPS gives its
**flux** column, so a UCD-keyed reader silently fits the sky. Only the utype role tells them apart.

**The second surprise reframed the work.** All thirteen files already *read*. What was wrong was the
metadata and the weights: `medium` was computed and then dropped before it reached `EpochData`, so
D43's 83 km/s guard could never fire; quality flags were ignored; a zero uncertainty was treated as
infinite precision; an all-NaN error array was swallowed rather than announced. "Make it read" would
have produced a rewrite that fixed nothing a user could see.

**Measured:** a BLOeM epoch is ~178 kB and one star ~5 MB; `112.25R7` holds 23,651 spectra over 929
targets, of which 21,716 were public on 2026-08-13 and every target has at least one public epoch.
A third programme's worth of extra data turned up unplanned — `115.28A9` re-observes the same stars
at *R* = 17000 and 23000 in two other windows, which is 1,827 more spectra that must **not** be
pooled with LR02 under one line-spread function, so the resolver defaults to the survey programme.

**The change was then reviewed adversarially, and five defects it had introduced were caught.**
All five had the same shape — a guess dressed as a reading. The worst: honouring quality flags
assumed the standard's "zero is good", but UVES_SQUAD's `STATUS` runs `{-5, 1}` and never takes the
value 0, so all 467 products in that collection read as 100% bad and raised. A flag whose convention
cannot be read is now ignored with a warning rather than inverted, which is the same principle as
refusing an undeclared air-vs-vacuum scale: the reader may decline to answer, but it may not guess.

[The tutorial](tutorials/bloem-sb2.md) now takes one from a survey identifier to disentangled
spectra with a band. Writing it turned up two things the example had wrong or left implicit.
Its window was advertised as sitting "between Hδ and Hγ without either core" and did not:
4000–4300 Å contains Hδ at 4101.7, and `nebular_windows` puts a ±300 km/s window at
4099.7–4107.9 inside it — so the script's stated reason for not modelling the nebula was
false. The window is now 4120–4300 Å, which contains no nebular line at all, and the docstring
says why the blue edge is the load-bearing number. And the ordering the survey forces turned
out to be a real gap in the façade rather than a documentation problem: with no published
period there is nothing to warm-start a Keplerian from, so the free RV table (item 3) has to
come *first* and supply the periodogram — but `Fit.free_velocities()` warm-starts from a
Keplerian fit, which needs a period. That circle is now broken. `Disentangler(velocities=...)`
declares the velocities you measured — cross-correlation lags, line splitting — in place of an
`Orbit`, and `fit()` returns the free table directly. The declaration refuses a cold table
outright and warns when the components are never resolved, so the one failure mode the mode
has stays loud. Measured: warm-started from velocities carrying 3 km/s of scatter *and* a
150 km/s systemic offset the fit cannot see, the table comes back at **0.096 / 0.070 km/s** —
the same as D42's Keplerian-warm-started 0.098 / 0.066 — with the systemic offset absent from
both the answer and the solver's bandwidth.

### 5. The `Disentangler` façade — **done** (D46)

The [design](design.md#6-api-sketch-target-user-code) sketched a friendlier API and the README
promised it. The path before this asked a new user to hand-build a dictionary of numpyro
distributions, call three functions in the right order, extract two hyperparameters for empirical
Bayes, and rebuild the model — a fine expert interface and a poor first impression.

It is sequenced *after* the nebular component and the RV mode on purpose. A façade is a vocabulary,
and a vocabulary is expensive to change once people are using it; the components and modes it names
should exist before it names them. It ships marked **experimental**, with the low-level API
documented and supported in parallel.

**The framing that made it tractable is that it is a compiler, not a shortcut.** You declare the
system — components, light fractions, instrument, what is known about the orbit — and it emits the
expert path. `dis.explain()` prints every derivation and `dis.expert()` hands back the exact
`(model, priors, init)` triple, so "supported in parallel" is structural rather than a promise.
Four things are derived, each of them something the low-level path makes you get right yourself:
the velocity budget (from the `k` priors' own support), the grid margin, the conjunction phase (by
a scan — 10⁵ nats between the best and worst phase on the packaged example), and the smoothness
hyperparameters by ML-II, reported with a **drift flag** on any that did not move from its start.

**The rejected names mattered as much as the accepted ones.** Six of the sketch's twelve had gone
stale in ways that would have shipped a lie: `Keplerian(t_peri=, ecc=, omega=, k1=, k2=)` names
five sites that no longer exist under those names; `GaussianLSF` is neither Gaussian-only (D38) nor
one number per instrument (D37); `light_ratio=` is the wrong word for a per-epoch simplex over N
components; `PerEpoch(eclipse model)` would have named a model that does not exist, which is
exactly what the ordering rule above exists to prevent. And `dis.replace(orbit=FreeVelocities())`
is the usage D42 measured failing at 122,000 nats worse — so the free-velocity mode hangs off a
completed fit instead, where it is unconstructible without its warm start.

**Measured** on the packaged example: 12 façade lines against 59 expert lines, MAP *K* = 41.978 and
62.978 against an injected 42 and 63, *e* = 0.1512 against 0.15, residual z-RMS 0.997, and NUTS
*K*₂ = 62.988 ± 0.081 with zero divergences.

Deliberately **not** in v1: jitter, AR(1), inferred light fractions and inferred LSF widths. Each is
a one-line site at the low level and a scientific claim rather than a convenience; a `jitter=True`
keyword would offer the first as the second. `fit.z_rms` is printed unconditionally so the need is
visible, and `expert()` is how it is met.

**An adversarial review of the finished module confirmed 38 defects, and the split is the lesson:
the vocabulary survived unchallenged, the wiring did not.** Not one accepted name was disputed.
What failed was every place a declaration had to be *translated* into the model's own conventions —
and that layer is exactly where nothing downstream can check the result. The smoothness rows were
assembled in declaration order while the model orders them stars-telluric-nebular, so a permuted
declaration regularized the wrong component, worth 9,900 nats and passing every existing guard
because the vectors were still the right length. And `ecc=Between(lo, hi)` was not the prior that
was fitted: the box corner reached `e = 2·hi`, and `lo` was validated then ignored. Both are fixed
and both now have regression tests; three declarations the façade could not honour — hierarchical
triples, Gauss-Hermite `h3`, and an eccentricity lower bound — are refusals rather than
approximations.

The honest risk is recorded in [the ledger](design.md#2-decisions-recorded-defaults) and in the
module's own docstring: `Star(light=0.62)` sits beside `period=Known(40.335, 0.5)` and is formatted
identically, but one is a choice and the other a measurement.

### 6. `sensitivity_forecast()` — **done** (D47)

"Will twelve more epochs at these phases break the degeneracy?" is a question about the posterior
covariance of the component spectra, and the posterior covariance does not depend on the flux
values — only on the epochs, the phases, the weights, and the prior. So it can be computed for
observations that have not been taken. Every piece needed already existed; nobody else can answer
the question at all. It was worth its two days on the strength of that asymmetry alone.

`albireo.forecast` is the module: `plan_epochs` builds the epochs of an observation that has not
happened, `sensitivity_forecast` returns what they would buy, and `baseline=` names the ones
already in hand so the answer is a difference rather than an absolute. The flux-independence is
structural rather than promised — the precision is assembled directly and the right-hand side is
never formed — and the regression test overwrites every flux with noise a hundred times the
continuum and requires the forecast back **bit-identical**. Everything is reported against the same
quantity under the prior alone, which is the D42 lesson made routine: a band that has relaxed onto
the prior looks exactly as convincing as one the data earned.

**The premise on this page was half wrong, and finding out was the main result.** §5.1 names
Var(Δ) — the spread of the differential shift — as the observing-strategy diagnostic, and it is
the right *quantity* but the wrong *objective*. A cadence aliased to the orbital period visits the
two extreme values of Δ over and over: it **maximizes** the variance and is a poor design, because
two values leave |g(k)| recurring to *J* at a whole comb of scales. Measured in
[`examples/08_forecast.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/08_forecast.py)
on a 13.7 d circular SB2 with eight epochs in hand and twelve to plan:

| twelve planned nights | RMS Δ*v* | blind fraction | 2nd mode σ | information gain |
|---|---|---|---|---|
| at P/2 (aliased) | 117.8 km/s | 58% | 0.518 | 243 nats |
| continuing the existing cadence | 115.7 km/s | 56% | 0.106 | 295 nats |
| spread over phase | **99.3 km/s** | **33%** | **0.071** | **375 nats** |

The aliased plan wins the one column §5.1 would have had you maximize and loses every other one.
The closed form is the screen and the explanation; the assembled covariance is the answer.

**The second surprise was where the worst-determined mode actually lives.** The first working
version reported it at exactly the prior width — because the model grid is deliberately wider than
the data, so its margin pixels are prior-only and are the largest eigenvalue of Σ on essentially
every real problem. It was measuring how much margin the grid had been given. Restricted to the
coordinates the design actually weights, the leading mode is what the theory says: a delocalized
see-saw across the components at *k* = 0, sitting at ~1× the prior **for every design**, since
nothing about phase sampling can constrain it. That is worth stating in the output rather than
hiding, and what distinguishes designs is how fast the rest of the ladder falls away from it.

Deliberately **not** forecast: the orbit. The Fisher information for a velocity runs through
∂(model)/∂v ∝ ℓᵢdᵢ′, so an error bar on *K*₂ needs the line depths — exactly what has not been
measured yet. Forecasting it against an assumed template would present the assumption as a result,
which is the failure mode the rest of this page exists to avoid.

### 7. A Gaia RVS loader, before December

Gaia DR4 is scheduled for 2 December 2026 and is the first release to publish epoch RVS spectra —
6,910,785,949 of them (49 TB), already normalized and already in the barycentric frame, alongside
several hundred thousand spectroscopic-binary orbit solutions. Being the package that works on the
day the data lands is a once-only opportunity.

**But the sentence this page used to carry — "a loader written against DR3's shape now becomes a
DR4 loader on release day" — is false, and finding that out was worth more than the loader would
have been.** Three things, each checked against a primary source rather than reasoned about:

**DR3's mean spectra have the velocity divided out, irreversibly.** The archive data model says
"the spectra are in the rest frame", and the shift is applied *per transit* with that transit's own
RV before co-adding. So the orbital modulation is not an offset waiting to be undone, it is smeared
away inside the stack — and for an SB2 the shift used a single blended cross-correlation RV that
was wrong for both components. DR3 `rvs_mean_spectrum` is not a disentangling dataset, not a demo,
and the loader should refuse it rather than warn.

**DR3 has no hot stars at all.** `SELECT MAX(rv_template_teff) FROM gaiadr3.gaia_source WHERE
has_rvs='true'` returns **14,500 K** over all 999,645 published spectra; above 15,000 K there are
zero. Spectra are published only for sources with a radial velocity, and the RV template grid stops
there. albireo's entire demonstrated science case — O and early-B stars — is outside it, and so is
HR 6819 itself (source 6649357561810851328, G = 5.26, `has_rvs=false`).

**The window is genuinely thin for early types.** 846–870 nm at *R* ≈ 11,500 gives a hot star the
Paschen series P13–P17 plus the Ca II triplet: one species, Stark-broadened, mutually blended, just
longward of the Paschen jump. Set against the He I / He II / Mg II / Si III of the 4380–4600 Å
window in the [benchmarks](benchmarks.md), that is a real mismatch — for *hot* stars. Gaia's own SB2
population is mostly cooler, where the triplet is the strongest thing in the spectrum.

### What DR4 actually delivers, from the draft data model

ESA pre-released a draft DR4 data model on 2026-06-26, and it answers the questions that decide the
design. Recorded here so they are not re-derived in December:

| | |
|---|---|
| Table | `rvs_epoch_spectrum`, DataLink only — not through the main TAP interface |
| DataLink retrieval type | `EPOCH_SPECTRUM_RVS` |
| Grid | **961** elements, 846–870 nm, step **0.025 nm** — *not* DR3's 2401 × 0.01 nm |
| Frame | "shifted from the Gaia reference frame to the barycentric reference frame and are normalised" |
| Time | `obs_time_rv`, **Barycentric JD in TCB − 2 455 197.5 d**, Roemer-corrected to the barycentre |
| Uncertainties | `flux_error[961]`, propagated per bin; NaN where every contributing CCD was masked |
| Per-pixel coverage | `combined_ccd_in_index`, stored only where smaller than `combined_ccds` |
| Selection | `all_source_rvs.has_epoch_rvs` — a graded byte, not a boolean (0 = none, 1 = very weak) |

Two of those are decisive. There **is** a per-transit barycentric timestamp, which was the one
unknown that could have made the product unusable to albireo regardless of everything else. And the
fluxes are linearly interpolated onto that fixed grid before publication, which walks straight into
[D4](design.md#2-decisions-recorded-defaults): albireo does not resample observations onto a common
grid *because* resampling correlates the noise and invalidates the diagonal `ivar` model. Gaia has
already done it upstream and albireo cannot undo it, so the published `flux_error` understates the
correlation. That is a systematic to state in the loader's docstring, and possibly a use for the
AR(1) machinery (D34) rather than a reason not to build.

**The strongest signal is one line in the schema.** Epoch spectra are produced for double-lined,
emission-line and contaminated transits — the ones whose radial velocity the pipeline rejects — and
`rv_assumed_sb2` marks any source where double lines were detected in at least ten transits, which
is a ready-made SB2 target list queryable by ADQL. When that flag is set, Gaia *excludes the source
from its own multi-transit RV solution* and publishes the spectra anyway. The data exists precisely
where Gaia's own pipeline gives up, which is the definition of an opening.

So: **build it in December, against the release, not now.** The DR4 documentation tree
(`archive/documentation/GDR4/`) still 404s, DR3 shares neither the grid, the frame, the table nor
the retrieval type, and this project's own D45 record is that a guess dressed as a reading is the
expensive kind of bug. What is worth doing before then is the go/no-go query on release day: join
the BLOeM DR3 `source_id`s that `resolve_bloem` already retrieves against DR4, and check whether
`rv_assumed_sb2` reaches any early-type population at all.

This also forced a small correctness fix worth doing anyway, and it has **already landed** (D43),
pulled forward because item 4's archive loader needed it first: RVS wavelengths are vacuum and most
optical spectrographs deliver air, and there was no field in which to declare which is which.
`EpochData(medium=...)` is now that field, `Dataset` refuses a mixture, and `air_to_vacuum` /
`vacuum_to_air` convert. Calling it "sub-ångström" undersold it: the offset is 0.87-2.74 Å, but as a
velocity it is a nearly constant **83 km/s**, the same order as the orbits being measured.

### 8. Downstream handoff

The current workflow in the literature is a relay race: measure velocities, fit an eclipsing-binary
model, disentangle in one code, renormalize by hand against an external light ratio, then fit
atmospheres in another. Five tools, five format conversions, and the uncertainty is dropped at the
disentangling joint because the disentangling code never produced one.

albireo should be the front half of that pipeline and should not attempt to be the back half —
GSSP, iSpec, Korg.jl and PySME are good at what they do. What it should ship is clean writers for
the formats they ingest, and a tutorial whose punchline is the thing only a posterior makes
possible: export *N* draws from the component-spectrum posterior, fit all *N*, and read the spread
in temperature and gravity. That converts a dropped uncertainty into a propagated one at the cost of
disk space.

### 9. A benchmark page against the incumbents

New codes are trusted after they reproduce old ones, not before. `scripts/fd3_bench.py` already
contains a format-verified fd3 exporter; what is missing is a built fd3 binary, a shift-and-add
reference implementation, and a system to run them on.

Three notes on doing this fairly. The comparison system should be AI Phoenicis, not HR 6819: it
eclipses, so the light ratio is externally known and the one genuinely free choice in disentangling
stops being a confound, and its semi-amplitudes are published to 0.02%. The shift-and-add comparison
must be a clean-room implementation from the published algorithm, because the existing code carries
no license. And the framing is agreement plus speed plus posteriors — never "the old code is wrong."
Where results differ, the page reports a diagnosis, not a verdict; a fair comparison also has to
handicap albireo down to fd3's conditions (common grid, uniform weights) before showing the
unhandicapped case separately.

The single most useful figure on that page is the cheapest: an SB2 with a nebular line, disentangled
by a method that can mask the contaminated pixels and by methods that structurally cannot.

## Tier 3 — later, and why later

**Time-variable component spectra.** The core assumption of disentangling is that each component's
spectrum is constant; the systems people most want to study — Be stars with variable discs,
interacting binaries, pulsators — violate it. Tier 2's nebular component is already rank-one time
variation (a fixed shape with a free per-epoch amplitude), and that captures more than it sounds
like: "the disc emission was 1.4× stronger that night" is most of what anyone models. Line *shape*
variation needs a second basis vector per variable component, with a shrinkage prior and the same
windowing machinery. The important structural point is that this stays inside the linear-Gaussian
family, so the analytic marginalization survives — this is a change of basis, not a change of
method. It waits for the dataset that demands it, which per the D38 record is HR 6819, where disc
variability is now one of only two surviving explanations for the period offset.

**specutils interoperability.** Accept and return `Spectrum` objects at the boundary; do not adopt
them internally. The reason for the asymmetry is concrete: `SpectrumCollection` requires equal-length
spectra and rejects per-epoch metadata differences, so it cannot represent a ragged multi-epoch
multi-instrument dataset, which is albireo's central input. Writing down *why* is itself worth doing,
because it names a problem the reader has already hit.

**More archives.** SOPHIE/ELODIE is the cheapest (no authentication, decades of high-resolution
planet-search cadence); LAMOST's medium-resolution time-domain survey is the largest by volume;
SDSS-V has the largest catalog of double-lined candidates but the most download plumbing, and its
visit spectra must be read in the un-shifted form.

**Survey-scale throughput.** Batch fitting and a measured systems-per-GPU-hour number. This waits on
GPU hardware, which is also the one open acceptance gate from M5 — the current scale projections are
extrapolated from CPU measurements and say so.

**A command-line interface.** Deferred until after the façade, because the façade is the
configuration schema and building a CLI first would freeze a second one. When it happens, the
subcommand with real value is `albireo fetch`, not `albireo fit`.

**A guard at zero eccentricity.** The `(√e cos ω, √e sin ω)` parameterization is singular at exactly
the origin: ω is undefined, `e·cos ω` behaves like `|x|`, and the gradient is NaN. numpyro then
reports `Cannot find valid initial parameters`, which points nowhere near the cause. Since tidally
circularized close binaries are exactly the population a user is most likely to bring, an
initialization at `secosw = sesinw = 0` is a predictable first experience, and it deserves a
message that says so rather than a documentation note. Cheap, and worth doing before the façade
freezes the initialization API.

**Community mechanics.** Discussions enabled, contributions labeled, and — the highest-conversion
and most-neglected item — writing to the handful of people whose systems albireo is built for and
offering to run one. A user who leaves with a disentangled spectrum of *their* star is a user.

## Explicit non-goals

Recorded so they stop being reconsidered.

- **No telluric radiative-transfer fitting and no molecfit wrapper.** Several wrappers already exist
  and are fragile. Masking and downweighting telluric regions inside the likelihood is already
  supported and is the right tool; exact multiplicative treatment remains the recorded v2 seam.
- **No WEAVE- or 4MOST-specific loaders.** 4MOST's processed data is released through the ESO
  archive, so it arrives free with the loader in Tier 2.
- **No replacement for GSSP, iSpec, Korg.jl, PHOEBE, or PySME.** Interoperate. Being the front half
  of everyone's pipeline is a better position than being a worse version of the back half.
- **No GUI**, per the v1 non-goals.

## What would change this page

The roadmap above assumes the bottleneck is reach rather than capability. Three observations would
falsify that and reorder everything: real BLOeM spectra failing to load or to disentangle sensibly;
the K₂ null calibration showing the detection statistic is far less powerful than the HR 6819 work
implies; or the GPU gate closing with numbers materially worse than the CPU projections, which would
make throughput the story instead of statistics.
