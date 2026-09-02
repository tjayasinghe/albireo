# Roadmap

This page records what albireo intends to become and why, in enough detail that the ordering can be
argued with. It is a plan, not a promise: items move, and items that turn out to answer a question
nobody asked are deleted. Decisions that have been made live in the
[decision ledger](design.md#2-decisions-recorded-defaults); this page is upstream of that.

## Where albireo sits

Three facts about the surrounding ecosystem set the strategy.

**The technique is mature; the software is not.** Spectral disentangling has been in use since
Simon & Sturm (1994) and Hadrava (1995), and the codes that implement it are, without exception,
either unmaintained or awkward to obtain. KOREL is Fortran, last released 2011, and is not
distributed as source at all: it is reachable only through a registered web service with a
16,384-bin ceiling. FDBinary is marked deprecated by its own ASCL entry; fd3, its successor, is
still used in papers published this year, and its distribution page has not been updated since 2014
and states no license. Spectangular's last papers are from 2017 and 2019. UNWIND, released in 2026,
is IDL, and its own paper describes it as lacking a manual or an installation package. The
shift-and-add code used to identify the companions in LB-1 and HR 6819 is a small research script
with no license file, which is a barrier to anyone who would build on it. None of this is a fault of
the authors; disentangling is a means to an end for all of them, and the software is a by-product of
the science. The consequence is that there is no maintained, installable, tested implementation of a
technique that, in the three years to mid-2026, roughly eight papers a year named in their abstract
alone, with many more using it in a methods section.

**Uncertainties on the disentangled spectra are the most requested capability, and no existing code
provides them.** The disentangled spectra are almost never the final product: they are fed to an
atmosphere code to obtain effective temperatures, gravities and abundances. The uncertainties on
those quantities are therefore systematically understated across the literature, and the papers say
so explicitly; TMBM III notes that normalization uncertainty is not propagated into the quoted
properties. Better bookkeeping does not fix this. Disentangling is an ill-conditioned linear inverse
problem with a low-frequency null space, and obtaining a reliable error band from it requires a
posterior, which means either sampling something very high-dimensional or marginalizing it away.
albireo does the second. That capability is the reason the package exists, and everything on this
page is downstream of making it usable.

**The direct approach was tried and was limited by compute.** PSOAP (Czekala et al. 2017) put a
Gaussian-process prior on the component spectra and sampled the joint posterior. It is the closest
direct predecessor, and its last commit is from December 2017. The paper is explicit about the
reason: the dense GP cost restricted it to narrow bandpasses and required cluster access. albireo's
analytic marginalization, banded structure and JAX implementation respond to that limitation
directly: the component spectra are integrated out in closed form rather than sampled, and the
linear algebra is banded rather than dense. Keeping that claim testable is why the
[benchmark record](benchmarks.md) is as long as it is.

What follows is ordered by that logic: make the existing capability usable (Tier 1), then reach the
communities whose problems it already solves (Tier 2), then extend the model (Tier 3).

## Tier 1 — make it usable (done)

A capability nobody can install, cite, or plot is not a capability. These items added no new
science and were the highest-leverage work available. They are done; what is left of this tier is
not code but pushing, tagging and registering, and that sequence, where the ordering matters, is
written down in [Releasing](releasing.md).

| | Item | Why it came first |
|---|---|---|
| 1 | `slow` / `network` / `gpu` markers and a `conftest` | Everything below adds tests; the fast suite went from 14 minutes to 6 |
| 2 | Single-sourced version, upgraded `CITATION.cff`, `CHANGELOG.md`, a [citing](citing.md) page, a tag-driven release workflow | Nothing could say `pip install albireo` until this was true |
| 3 | Deployed docs with a rendered [API reference](api/index.md), `py.typed`, coverage | 70 exported names with good docstrings, rendered nowhere |
| 4 | `load_example()` and a [five-minute quickstart](quickstart.md) that runs offline | Time-to-first-plot is the best-attested adoption lever there is |
| 5 | `albireo.results`: arviz conversion, save/load, export to FITS/ECSV/ASCII | An hours-long fit used to produce printed text and nothing else |
| 6 | `albireo.plotting` | The three most reusable plotting functions in the repo were stranded inside example scripts |

Two of these have their reasoning recorded rather than asserted.

**Export.** The disentangled spectrum plus its uncertainty band is the product. Until it could leave
the process in a format an atmosphere code reads, albireo was the middle of a pipeline with no
downstream connector, and the uncertainty, which is the main differentiator, was dropped at exactly
the joint where it matters. See Tier 2 item 8 for the other half of this.

**The quickstart must not require finding data.** The pattern worth copying is lightkurve's: the
first code block does science rather than configuration, on data the user did not have to locate. So
a small simulated dataset ships inside the wheel, with no download, no astropy and no archive
account, and a real archival example is the second step, once the ESO loader below exists.

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

In albireo this is a static component whose per-epoch amplitude is a free parameter: structurally
the telluric component with a traced amplitude instead of a fixed one, and physically correct, since
nebular flux is added on top of the total continuum and takes no light from the stars. Two details
carried the work, both as predicted: the nebular component is static in the *barycentric* frame,
which is the opposite convention from the telluric one, and confining it to the Balmer, He I and
forbidden-line windows required the smoothness prior's regularization strength to become per-pixel
(`SmoothnessPrior(tau_profile=, eta_profile=)`, built by `priors.window_profile`).

It had the highest impact per line of code on this page, and the closed loop measures by how much:
an SB2 whose Hβ absorption carries a static nebular line disentangles to a core 26% too shallow and
an equivalent width 11.5% low without the component, against 0.14% with it. Equivalent width is what
reaches the atmosphere code, so that is a systematic error in log *g*, of the kind nothing in the
current literature propagates. The effect on the orbit was larger and was not anticipated: a static
line is a component with *K* = 0, so a nebula-blind joint fit hands the emission to whichever star
can be made to move least, returning K₂ 59% low, a period long by 0.171 d, and a circular orbit at
*e* = 0.95 (the solver's clip). The contamination reaches the masses, not only the atmospheres. Two
degeneracies came with the component and are closed by convention rather than by data (the amplitude
scale, pinned by centering the log-amplitudes; the nebular velocity, which is a placement convention
for the window profile and not a measurement); both are recorded in
[the math](math.md#13-lsf-light-fractions-response).

### 2. Calibrated faint-companion detection — **done** (D41)

The K₂ scan already answered "is there a companion, and at what velocity semi-amplitude", the
question at the center of every dormant-black-hole candidate. What it did not do is say how
often noise alone would produce the peak it found. The state of the art in the literature is
a bare χ² map with no false-alarm probability attached, and the papers are explicit about the
resulting fragility: small deviations in the assumed primary semi-amplitude produce spurious
features in the recovered secondary spectrum.

All three pieces are built. The scan is vectorized: `MarginalOrbitModel.log_likelihood_sweep`
runs the trial grid as one batched `lax.map` instead of a Python loop with a device
synchronization per point. K₁ is marginalizable (`k2_scan(k1_sigma=)`) over a Gauss–Hermite
rule applied to both the companion and the no-companion model, so `D` stays a ratio of two
marginal likelihoods. And `albireo.calibrate.detection_limit` runs the injection-recovery
calibration, resimulating through the observed data's own operators
(`forward.with_data`, `simulate.resimulate`) so that a few hundred scans cost under a minute.

The K₁ result was sharper than expected and reframes why marginalization matters: a K₁ 10% high
did not merely blur the answer, it took the recovered companion's line pattern from 0.96
correlation with truth to 0.49 while more than tripling `D`, so the artifact reads as a stronger
detection. A calibrated threshold cannot catch that, because the null trials are drawn under the
same wrong assumption; only the marginalization can. The two are complementary, which is now
stated wherever either is documented. The calibration itself delivers the quotable sentence,
"any companion contributing more than *X*% of the light would have been detected at 95%
confidence", with a threshold that is conservative by construction and a false-alarm probability
that never claims finer resolution than `1/(n_null + 1)`.

One expected dependence turned out not to be there, which is worth knowing before anyone spends
compute on it: the limit is nearly flat in K₂ (0.292 / 0.296 / 0.297% at K₂ = 20 / 40 / 65
km/s). In an SB2 the two components move in antiphase, so their relative velocity never drops
below about K₁, and with K₁ = 55 km/s the pair is well separated at every trial. A real K₂
dependence should appear only when K₁ is itself small. The dependence that does remain is on
the assumed companion template, since the observable is ℓ₂·d₂ and a featureless companion is
invisible at any light fraction, and that assumption is required to be quoted with the number.
The [HR 6819 campaign](benchmarks.md) remains a ready-made validation set.

### 3. A per-epoch radial-velocity table — **done** (D42)

Free per-epoch velocities, with no Keplerian, were listed in the [design](design.md) as a diagnostic
mode and had never been exposed. The low-level machinery existed and was differentiable; only the
sampling path was missing. It is now a `velocity` theta site that replaces the orbit, with
`relative_velocities`, `relative_velocity_errors` and `keplerian_residuals` alongside it.

Two reasons it was worth more than its size, both borne out: a per-epoch RV table with quoted
uncertainties is the artifact the binary-star community expects from any spectroscopic analysis,
and it is the natural bridge for users of cross-correlation and shift-and-add codes, who can
compare against something familiar before trusting the joint fit. It is also the model check for
the Keplerian mode: fit free velocities, then ask whether a Keplerian threads the resulting
posterior.

The caveat in the roadmap was right but understated. The table is differential, and it has one
arbitrary zero point per component rather than one in total: with no orbit tying the stars
together, each free spectrum absorbs a constant added to its own shifts. Left uncentered, that zero
point is pinned by shift-interpolation error rather than by data, since a whole-pixel common shift
costs 4e-9 of the log-likelihood and a 0.1-pixel one costs 7.3 nats. albireo removes it in pixel
space, where the removal is exact. The same projection turned out to be needed for the
uncertainties: the raw Laplace diagonal returns the prior and nothing else (37.95 km/s = 120/√10
on every entry, against a real per-epoch error of 0.059 km/s), which is the kind of number that
looks equally convincing on a good dataset and a useless one.

Measured: warm-started from a Keplerian 30% wrong in both semi-amplitudes, per-epoch RVs recover to
0.098 / 0.066 km/s, 1/60th of a model pixel, and the Wilson slope to 0.4%. From a cold start the
mode fails, as it must, but at a potential 122,000 nats worse, so the failure is visible.

`examples/09_rv_table.py` is the worked script, and it is built around the two properties that
are counter-intuitive rather than around the API: it demonstrates the per-component zero point
by shifting one star's velocities by 50 km/s and watching the log-likelihood not move (0 nats
for the relativistic shift, 8.7e-6 for the ordinary one that is only its first-order
approximation), and it prints the raw Laplace bars beside the projected ones, 37.947 km/s on
every entry, which is `120/√10`, the prior, against 0.056-0.065 measured. The narrative material
is in [the math](math.md#76-free-per-epoch-velocities-the-rv-table) and
`tests/test_velocity_table.py`.

### 4. An ESO archive loader, and BLOeM in one line — **done** (D44, D45)

The reason to do it was BLOeM: ~929 targets in the Small Magellanic Cloud with roughly 25 epochs
each, an intrinsic binary fraction above 70%, and 59 published double-lined systems, all public,
with the survey team's own disentangling of them still listed as future work in their July 2026
review. A short fetch of a real BLOeM target, disentangled with posteriors, is simultaneously the
tutorial, the evidence of research use that software journals ask for, and a paper. It was sequenced
after the nebular component deliberately: BLOeM's targets sit in nebulosity, so the demo needs that
component to be reliable.

Both halves are built. `albireo.archive` (D44) is the ObsCore/TAP client and resumable downloader;
`albireo.io` (D45) now reads what it fetches by dispatching on the IVOA utypes rather than on column
names, and `resolve_bloem` / `bloem_catalogue` / `bloem_spectra` turn a survey identifier into that
star's epochs. `ab.bloem_catalogue(binary_class="SB2")` returns the 59 double-lined targets;
`examples/06_bloem.py` takes one from name to fitted spectra.

"A single file layout" was wrong, and that is the useful part. Thirteen real Phase 3 spectra across
seven instruments were read column by column, and no two collections agree on anything except the
utypes: flux is `FLUX`, `FLUX_REDUCED`, or both at once; the extension is `SPECTRUM` except for
Gaia-ESO's `phase3spectrum`; units are angstrom, Angstrom or nm. The obvious fix, keying on UCDs, is
worse than the disease: UVES gives its sky-background column the same UCD HARPS gives its flux
column, so a UCD-keyed reader fits the sky without any symptom. Only the utype role tells them
apart.

The second surprise reframed the work. All thirteen files already read. What was wrong was the
metadata and the weights: `medium` was computed and then dropped before it reached `EpochData`, so
D43's 83 km/s guard could never fire; quality flags were ignored; a zero uncertainty was treated as
infinite precision; an all-NaN error array was swallowed rather than announced. "Make it read" would
have produced a rewrite that fixed nothing a user could see.

Measured: a BLOeM epoch is ~178 kB and one star ~5 MB; `112.25R7` holds 23,651 spectra over 929
targets, of which 21,716 were public on 2026-08-13, and every target has at least one public epoch.
A third programme's worth of extra data turned up unplanned: `115.28A9` re-observes the same stars
at *R* = 17000 and 23000 in two other windows, which is 1,827 more spectra that must not be pooled
with LR02 under one line-spread function, so the resolver defaults to the survey programme.

The change was then reviewed adversarially, and five defects it had introduced were caught. All five
had the same shape, a guess presented as a reading. The worst: honouring quality flags assumed the
standard's "zero is good", but UVES_SQUAD's `STATUS` runs `{-5, 1}` and never takes the value 0, so
all 467 products in that collection read as 100% bad and raised. A flag whose convention cannot be
read is now ignored with a warning rather than inverted, which is the same principle as refusing an
undeclared air-vs-vacuum scale: the reader may decline to answer, but it may not guess.

[The tutorial](tutorials/bloem-sb2.md) now takes one target from a survey identifier to disentangled
spectra with a band. Writing it turned up two things the example had wrong or left implicit. Its
window was advertised as sitting "between Hδ and Hγ without either core" and did not: 4000–4300 Å
contains Hδ at 4101.7, and `nebular_windows` puts a ±300 km/s window at 4099.7–4107.9 inside it, so
the script's stated reason for not modelling the nebula was false. The window is now 4120–4300 Å,
which contains no nebular line at all, and the docstring says why the blue edge is the critical
number. And the ordering the survey forces turned out to be a gap in the façade rather than a
documentation problem: with no published period there is nothing to warm-start a Keplerian from, so
the free RV table (item 3) has to come first and supply the periodogram, but `Fit.free_velocities()`
warm-starts from a Keplerian fit, which needs a period. That circle is now broken.
`Disentangler(velocities=...)` declares measured velocities (cross-correlation lags, line splitting)
in place of an `Orbit`, and `fit()` returns the free table directly. The declaration refuses a cold
table outright and warns when the components are never resolved, so the one failure mode the mode
has stays visible. Measured: warm-started from velocities carrying 3 km/s of scatter and a 150 km/s
systemic offset the fit cannot see, the table comes back at 0.096 / 0.070 km/s, the same as D42's
Keplerian-warm-started 0.098 / 0.066, with the systemic offset absent from both the answer and the
solver's bandwidth.

### 5. The `Disentangler` façade — **done** (D46)

The [design](design.md#6-api-sketch-target-user-code) sketched a friendlier API and the README
promised it. The path before this asked a new user to hand-build a dictionary of numpyro
distributions, call three functions in the right order, extract two hyperparameters for empirical
Bayes, and rebuild the model: a workable expert interface and a poor first impression.

It is sequenced after the nebular component and the RV mode deliberately. A façade is a vocabulary,
and a vocabulary is expensive to change once people are using it; the components and modes it names
should exist before it names them. It ships marked experimental, with the low-level API documented
and supported in parallel.

The framing that made it tractable is that it is a compiler rather than a shortcut. The user
declares the system (components, light fractions, instrument, what is known about the orbit) and it
emits the expert path. `dis.explain()` prints every derivation and `dis.expert()` hands back the
exact `(model, priors, init)` triple, so "supported in parallel" is structural rather than a
promise. Four things are derived, each of them something the low-level path leaves to the user: the
velocity budget (from the `k` priors' own support), the grid margin, the conjunction phase (by a
scan, with 10⁵ nats between the best and worst phase on the packaged example), and the smoothness
hyperparameters by ML-II, reported with a drift flag on any that did not move from its start.

The rejected names mattered as much as the accepted ones. Six of the sketch's twelve had gone stale
in ways that would have shipped a false statement: `Keplerian(t_peri=, ecc=, omega=, k1=, k2=)`
names five sites that no longer exist under those names; `GaussianLSF` is neither Gaussian-only
(D38) nor one number per instrument (D37); `light_ratio=` is the wrong word for a per-epoch simplex
over N components; `PerEpoch(eclipse model)` would have named a model that does not exist, which is
what the ordering rule above exists to prevent. And `dis.replace(orbit=FreeVelocities())` is the
usage D42 measured failing at 122,000 nats worse, so the free-velocity mode hangs off a completed
fit instead, where it is unconstructible without its warm start.

Measured on the packaged example: 12 façade lines against 59 expert lines, MAP *K* = 41.978 and
62.978 against an injected 42 and 63, *e* = 0.1512 against 0.15, residual z-RMS 0.997, and NUTS
*K*₂ = 62.988 ± 0.081 with zero divergences.

Not in v1, by decision: jitter, AR(1), inferred light fractions and inferred LSF widths. Each is a
one-line site at the low level and a scientific claim rather than a convenience; a `jitter=True`
keyword would offer the first as the second. `fit.z_rms` is printed unconditionally so the need is
visible, and `expert()` is how it is met.

An adversarial review of the finished module confirmed 38 defects, and the split is the lesson: the
vocabulary survived unchallenged, the wiring did not. Not one accepted name was disputed. What
failed was every place a declaration had to be translated into the model's own conventions, and that
layer is where nothing downstream can check the result. The smoothness rows were assembled in
declaration order while the model orders them stars-telluric-nebular, so a permuted declaration
regularized the wrong component, worth 9,900 nats and passing every existing guard because the
vectors were still the right length. And `ecc=Between(lo, hi)` was not the prior that was fitted:
the box corner reached `e = 2·hi`, and `lo` was validated then ignored. Both are fixed and both now
have regression tests; three declarations the façade could not honour, namely hierarchical triples,
Gauss-Hermite `h3`, and an eccentricity lower bound, are refusals rather than approximations.

The risk is recorded in [the ledger](design.md#2-decisions-recorded-defaults) and in the module's
own docstring: `Star(light=0.62)` sits beside `period=Known(40.335, 0.5)` and is formatted
identically, but one is a choice and the other a measurement.

### 6. `sensitivity_forecast()` — **done** (D47)

"Will twelve more epochs at these phases break the degeneracy?" is a question about the posterior
covariance of the component spectra, and the posterior covariance does not depend on the flux
values, only on the epochs, the phases, the weights and the prior. So it can be computed for
observations that have not been taken. Every piece needed already existed, and no other code answers
the question at all, which is why it was worth its two days.

`albireo.forecast` is the module: `plan_epochs` builds the epochs of an observation that has not
happened, `sensitivity_forecast` returns what they would buy, and `baseline=` names the ones
already in hand so the answer is a difference rather than an absolute. The flux-independence is
structural rather than promised, since the precision is assembled directly and the right-hand side
is never formed, and the regression test overwrites every flux with noise a hundred times the
continuum and requires the forecast back bit-identical. Everything is reported against the same
quantity under the prior alone, which is the D42 lesson made routine: a band that has relaxed onto
the prior looks exactly as convincing as one the data earned.

The premise on this page was half wrong, and finding that out was the main result. §5.1 names
Var(Δ), the spread of the differential shift, as the observing-strategy diagnostic, and it is the
right quantity but the wrong objective. A cadence aliased to the orbital period visits the two
extreme values of Δ repeatedly: it maximizes the variance and is a poor design, because two values
leave |g(k)| recurring to *J* at a whole comb of scales. Measured in
[`examples/08_forecast.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/08_forecast.py)
on a 13.7 d circular SB2 with eight epochs in hand and twelve to plan:

| twelve planned nights | RMS Δ*v* | blind fraction | 2nd mode σ | information gain |
|---|---|---|---|---|
| at P/2 (aliased) | 117.8 km/s | 58% | 0.518 | 243 nats |
| continuing the existing cadence | 115.7 km/s | 56% | 0.106 | 295 nats |
| spread over phase | **99.3 km/s** | **33%** | **0.071** | **375 nats** |

The aliased plan wins the one column §5.1 would have had the observer maximize and loses every other
one. The closed form is the screen and the explanation; the assembled covariance is the answer.

The second surprise was where the worst-determined mode lives. The first working version reported it
at exactly the prior width, because the model grid is deliberately wider than the data, so its margin
pixels are prior-only and are the largest eigenvalue of Σ on essentially every real problem. It was
measuring how much margin the grid had been given. Restricted to the coordinates the design actually
weights, the leading mode is what the theory predicts: a delocalized see-saw across the components at
*k* = 0, sitting at ~1× the prior for every design, since nothing about phase sampling can constrain
it. That belongs in the output rather than hidden, and what distinguishes designs is how fast the
rest of the ladder falls away from it.

Not forecast, by decision: the orbit. The Fisher information for a velocity runs through
∂(model)/∂v ∝ ℓᵢdᵢ′, so an error bar on *K*₂ needs the line depths, which are exactly what has not
been measured yet. Forecasting it against an assumed template would present the assumption as a
result, which is the failure mode the rest of this page exists to avoid.

### 7. A Gaia RVS loader, before December

Gaia DR4 is scheduled for 2 December 2026 and is the first release to publish epoch RVS spectra,
6,910,785,949 of them (49 TB), already normalized and already in the barycentric frame, alongside
several hundred thousand spectroscopic-binary orbit solutions. Working on the day the data lands is
a once-only opportunity.

The sentence this page used to carry, "a loader written against DR3's shape now becomes a DR4 loader
on release day", is false, and establishing that was worth more than the loader would have been.
Three points, each checked against a primary source rather than reasoned about:

**DR3's mean spectra have the velocity divided out, irreversibly.** The archive data model says
"the spectra are in the rest frame", and the shift is applied per transit with that transit's own
RV before co-adding. The orbital modulation is therefore not an offset waiting to be undone, it is
smeared away inside the stack, and for an SB2 the shift used a single blended cross-correlation RV
that was wrong for both components. DR3 `rvs_mean_spectrum` is not a disentangling dataset, not a
demo, and the loader should refuse it rather than warn.

**DR3 has no hot stars at all.** `SELECT MAX(rv_template_teff) FROM gaiadr3.gaia_source WHERE
has_rvs='true'` returns 14,500 K over all 999,645 published spectra; above 15,000 K there are zero.
Spectra are published only for sources with a radial velocity, and the RV template grid stops there.
albireo's demonstrated science case, O and early-B stars, is outside it, and so is HR 6819 itself
(source 6649357561810851328, G = 5.26, `has_rvs=false`).

**The window is thin for early types.** 846–870 nm at *R* ≈ 11,500 gives a hot star the Paschen
series P13–P17 plus the Ca II triplet: one species, Stark-broadened, mutually blended, just longward
of the Paschen jump. Set against the He I / He II / Mg II / Si III of the 4380–4600 Å window in the
[benchmarks](benchmarks.md), that is a real mismatch for hot stars. Gaia's own SB2 population is
mostly cooler, where the triplet is the strongest feature in the spectrum.

### What DR4 actually delivers, from the draft data model

ESA pre-released a draft DR4 data model on 2026-06-26, and it answers the questions that decide the
design. Recorded here so that they are not re-derived in December:

| | |
|---|---|
| Table | `rvs_epoch_spectrum`, DataLink only, not through the main TAP interface |
| DataLink retrieval type | `EPOCH_SPECTRUM_RVS` |
| Grid | 961 elements, 846–870 nm, step 0.025 nm, not DR3's 2401 × 0.01 nm |
| Frame | "shifted from the Gaia reference frame to the barycentric reference frame and are normalised" |
| Time | `obs_time_rv`, Barycentric JD in TCB − 2 455 197.5 d, Roemer-corrected to the barycentre |
| Uncertainties | `flux_error[961]`, propagated per bin; NaN where every contributing CCD was masked |
| Per-pixel coverage | `combined_ccd_in_index`, stored only where smaller than `combined_ccds` |
| Selection | `all_source_rvs.has_epoch_rvs`, a graded byte rather than a boolean (0 = none, 1 = very weak) |

Two of those are decisive. There is a per-transit barycentric timestamp, which was the one unknown
that could have made the product unusable to albireo regardless of everything else. And the fluxes
are linearly interpolated onto that fixed grid before publication, which runs into
[D4](design.md#2-decisions-recorded-defaults): albireo does not resample observations onto a common
grid because resampling correlates the noise and invalidates the diagonal `ivar` model. Gaia has
already done it upstream and albireo cannot undo it, so the published `flux_error` understates the
correlation. That is a systematic to state in the loader's docstring, and possibly a use for the
AR(1) machinery (D34) rather than a reason not to build.

One line in the schema is the strongest signal. Epoch spectra are produced for double-lined,
emission-line and contaminated transits, the ones whose radial velocity the pipeline rejects, and
`rv_assumed_sb2` marks any source where double lines were detected in at least ten transits, which
is a ready-made SB2 target list queryable by ADQL. When that flag is set, Gaia excludes the source
from its own multi-transit RV solution and publishes the spectra anyway. The data exists precisely
where Gaia's own pipeline gives up.

The conclusion is to build it in December, against the release, rather than now. The DR4
documentation tree (`archive/documentation/GDR4/`) still 404s, DR3 shares neither the grid, the
frame, the table nor the retrieval type, and this project's own D45 record is that a guess presented
as a reading is the expensive kind of bug. What is worth doing before then is the go/no-go query on
release day: join the BLOeM DR3 `source_id`s that `resolve_bloem` already retrieves against DR4, and
check whether `rv_assumed_sb2` reaches any early-type population at all.

This also forced a small correctness fix worth doing anyway, and it has already landed (D43), pulled
forward because item 4's archive loader needed it first: RVS wavelengths are vacuum and most optical
spectrographs deliver air, and there was no field in which to declare which is which.
`EpochData(medium=...)` is now that field, `Dataset` refuses a mixture, and `air_to_vacuum` /
`vacuum_to_air` convert. Calling it "sub-ångström" understated it: the offset is 0.87-2.74 Å, but as
a velocity it is a nearly constant 83 km/s, the same order as the orbits being measured.

### 8. Downstream handoff — **done**

The current workflow in the literature is a relay: measure velocities, fit an eclipsing-binary
model, disentangle in one code, renormalize by hand against an external light ratio, then fit
atmospheres in another. Five tools, five format conversions, and the uncertainty is dropped at the
disentangling joint because the disentangling code never produced one.

albireo is the front half of that pipeline and does not attempt to be the back half; GSSP, iSpec,
Korg.jl and PySME are good at what they do. `albireo.handoff` ships the writers: `write_gssp`,
`write_ispec`, and `export_draws`, with [a tutorial](tutorials/downstream.md) and
`examples/10_downstream.py`.

The formats turned out to be the hard part, and both traps fail without a symptom. iSpec does no
unit conversion on its text path: its whole internal scale, line lists included, is nanometres, so
an ångström value lands a factor of ten outside every model grid and still fits something. And GSSP
infers its synthetic step from the supplied file: "the step width in wavelength that will be used
for the calculation of synthetic spectra is computed from the observations" (Tkachenko 2015,
Appendix B, which is the entire manual; there is no separate document and no source repository).
albireo solves on a log-wavelength grid, whose linear spacing drifts 1.32% across the packaged
example's window, so `write_gssp` resamples onto an equidistant grid rather than writing one GSSP
would mis-step. Both are regression-tested.

GSSP accepts no per-pixel uncertainty, which is what makes the draws necessary rather than
convenient. Its configuration files contain no error path, no S/N entry and no weighting entry; its
own error bars come from χ² on the fit residuals. The posterior band therefore cannot reach a
temperature through the file at all. It can only get there by fitting *N* spectra and taking the
spread, which is what `export_draws` is for.

The loop is old and the draws are new, and the difference is measured rather than argued. Kiran et
al. (2016, §3.5) added "artificial Gaussian noise with sigma = sigma_c" to a disentangled profile
and refitted 500 times; cite them. But that assumes the error is white, and disentangling error is
not: it has a low-frequency null space, which is what moves a continuum and through it a surface
gravity. `draw_spectra` returns `d_hat + L⁻ᵀz` on the vector stacked over all components, so draws
are correlated across wavelength and across the two stars. Measured on the packaged example, using
equivalent width as the stand-in for the atmosphere code (the appropriate stand-in, since D40
established that EW is what reaches it and that an 11.5% EW error is a systematic in log *g*):

| | EW (Å) | joint draws | independent per-pixel noise | ratio |
|---|---|---|---|---|
| component 1 | 0.2568 | 0.01873 | 0.01041 | **1.80×** |
| component 2 | 0.0417 | 0.02974 | 0.00881 | **3.38×** |

White noise understates the integrated uncertainty two- to three-fold. For a pointwise question the
band is sufficient; every atmospheric parameter integrates the spectrum.

The sharper result was not the one being looked for. The correlation between the two components'
equivalent widths across draws is −0.992, against −0.052 for the same statistic under independent
noise. That is D47's *k* = 0 exchange mode, the delocalized see-saw sitting at ~1× the prior for
every observing design, arriving in a derived quantity: the two stars trade line depth almost
exactly, so their difference is far better determined than either alone, and fitting the components
separately with independent error bars misstates the answer in both directions at once. It is also
the clearest argument for why the draw index belongs in the exported filename.

Not claimed: the atmosphere code's own model error is outside this posterior, and so is the light
ratio, which albireo conditions on rather than marginalizes, which is the systematic Pavlovski &
Hensberge (2011) call dominant. Both are in the tutorial's caveat list rather than its headline.

### 9. A benchmark page against the incumbents — **done**

New codes are trusted after they reproduce old ones, not before. Both comparison codes are built
and run, and so is AI Phoenicis; the numbers are in [the benchmark record](benchmarks.md).

AI Phe delivered the cross-validation and one further result. From 36 archival HARPS spectra,
started 15% off, the eccentricity comes back at 0.1879 against a published 0.1878 ± 0.0006: a TESS
light curve and a spectroscopic disentangling agreeing to 0.05% by wholly independent routes. The
semi-amplitudes carry a reproducible ~1% systematic (K₁ −1.4%, K₂ +0.8%, mass ratio 2.2% toward
unity) that is not the optimizer, not line selection (two disjoint windows agree to 0.02%) and not
the light ratio (9 nats out of 53,306). It is recorded as an open lead rather than as a correction
to a 0.02% literature value.

The run also found a defect that no simulation would have. A badly chosen window straddled the gap
between HARPS's two CCDs, 32.9 Å of exact zeros at 5304.67–5337.61 Å, which arrived weighted like
data, because the pixels are finite, carry no quality flag, and HARPS ships no error array, so their
inverse variance was estimated from the small scatter of a flat run of zeros. The result
disentangled to component spectra with negative flux. `albireo.mask_flux_gaps` now catches
contiguous runs of non-positive flux and reports them. That is D45's failure shape one level up.

`scripts/shift_and_add.py` is the clean-room implementation, written from González & Levato (2006)
§2.1–2.3 and Quintero et al. (2020) and from no source code: the incumbent implementation carries
no license, so it was never opened. It is validated against the paper's own theory rather than
against itself: §2.3 derives that the residual is diffused rather than annihilated, by a Gaussian
of `√(2m)·σ_d` after *m* sweeps, and seeding a delta-function error reproduces that law.

Head to head, all three on identical data (aligned RMS, and wall time):

| | comp 1 | comp 2 | wall | uncertainty? |
|---|---|---|---|---|
| **albireo** | **0.0093** | **0.0116** | 0.182 s | **yes** |
| fd3 | 0.0198 | 0.0223 | 0.111 s | no |
| shift-and-add | 0.0248 | 0.0302 | **0.018 s** | no |

albireo is ~2× more accurate and, on the machine this table was recorded on, the slowest of the
three. Shift-and-add is 10× faster than albireo and 6× faster than fd3, which for a 1200-pixel
separation is the expected result, since it is a handful of array shifts and means. The accuracy
margin is not a stopping artifact: at 50 sweeps instead of the published 7, shift-and-add improves
by 11% and is still 2.4× behind.

> **Speed re-run 2026-08-16; the accuracy column is the one that replicated.** All three
> codes re-run on one machine (a 16-core desktop) under one protocol: every RMS above
> reproduces exactly, and the walls become shift-and-add 0.026 s, albireo 0.059 s, fd3
> 0.064 s single-threaded, which is parity, and 0.104 s as it actually ships, its OpenBLAS
> spinning 32 threads. The 0.018 s above was itself partly an artifact: the harness timed
> shift-and-add in-process after the XLA solve, on a heap XLA leaves serving ~35 KB
> allocations through microsecond free-list walks, a convention behind both recorded
> numbers, since replaced with a fresh-process timing. So shift-and-add stays fastest
> everywhere; "slowest of the three" was a fact about single-thread hardware; and albireo
> remains ~2× more accurate and the only code returning a posterior. Full record:
> benchmarks.md "D50 re-run".

The most reusable result is that all three fail the same way. fd3's raw error is nine tenths a
constant; shift-and-add's grows on the fainter component because `B = 0` leaves its continuum to
the initialization. Both are the *k* = 0 null space, and the shift-and-add theory says so exactly:
the per-mode convergence factor has modulus 1 at zero frequency, a fixed point no number of sweeps
can touch. Three independent methods, one degeneracy.

Building fd3 was itself a finding. The tarball's prebuilt binary is 32-bit i386 and will not run on
a modern host, so it had to be rebuilt from source; and before being used for anything it was
validated against the author's own shipped outputs, reproducing the published example (V453 Cyg,
1344 px) exactly across a different compiler, architecture and GSL. It is not vendored here, since
the distribution still states no license.

The result is informative in both directions. fd3 is 1.64× faster in steady state on a 1200-pixel
SB2 separation; a small compiled C program should win that regime, and the useful correction is that
albireo is in the same class rather than an order of magnitude behind, where the harness's own
un-jitted figure would have overstated the gap by 20×. fd3's raw RMS is ~15× larger, but nine tenths
of that is a constant: mean-aligning collapses it, because the *k* = 0 offset is the null space
neither code can determine from constant-light data, and fd3 leaves it to the user's hand
renormalization while albireo's prior pins it. On shape, with that offset removed, albireo is about
2× more accurate. And fd3 returns no uncertainty at all, which is the gap this page is organized
around.

Three notes on doing this fairly. The comparison system should be AI Phoenicis rather than HR 6819:
it eclipses, so the light ratio is externally known and the one genuinely free choice in
disentangling stops being a confound, and its semi-amplitudes are published to 0.02%. The
shift-and-add comparison must be a clean-room implementation from the published algorithm, because
the existing code carries no license. And the framing is agreement plus speed plus posteriors, never
"the old code is wrong." Where results differ, the page reports a diagnosis rather than a verdict; a
fair comparison also has to handicap albireo down to fd3's conditions (common grid, uniform weights)
before showing the unhandicapped case separately.

~~The single most useful figure on that page is the cheapest: an SB2 with a nebular line,
disentangled by a method that can mask the contaminated pixels and by methods that structurally
cannot.~~ Withdrawn: the comparison was unfair, and establishing that was worth more than the
figure. González & Levato explicitly permit "any combination algorithm... weights or some rejection
algorithm", so masking is inside the published shift-and-add method rather than an extension of it,
and `tests/test_shift_and_add.py` demonstrates a zero-weighted epoch being excluded. Proposing a
figure whose conclusion is a capability the comparison code has would have been the "the old code is
wrong" framing this section forbids. What the incumbents cannot do is report an uncertainty, and
that is the comparison the page draws instead.

### 10. Epoch radial velocities for every component, by TODCOR — **done** (D56, D57)

The joint fit never measures a per-epoch velocity, which is the correct behaviour when the
component spectra are unknown. It is also the one product every eclipsing-binary analysis, survey
pipeline and orbit code starts from, and the reason a user arriving from a cross-correlation
background found nothing familiar here. `albireo.todcor` is the two-dimensional correlation of
Zucker & Mazeh (1994), in which each spectrum is correlated against a combination of two templates
with independent shifts, so that blended peaks stop pulling each other and a faint companion can be
measured from a single spectrum, generalized to any number of components and to weighted, masked,
multi-instrument data by writing it as the least-squares fit it is. On a uniform grid with uniform
weights it reproduces the published formulae to 1e-10; on real data the masks, gaps, cosmics and
per-pixel weights enter through the same operators as the forward model and change nothing.

Three things make it more than a port. The sub-pixel minimum is computed rather than read off a
parabola, because the shift operator is linear in the template and the chi-square is therefore an
exact quadratic inside each pixel cell; the errors are Zucker's (2003) maximum-likelihood ones, with
the blending and detection diagnostics that a batch has to notice; and the loop closes, since
`Fit.templates()` turns a disentangling's own components into templates and
`Fit.measure_velocities()` runs the epochs against them, with the zero point that a disentangled
frame cannot identify stated on every table rather than absorbed without comment. `albireo.rvorbit`
fits the Keplerian to the table with the same solver and conventions as the joint model, and hands
the elements back as a warm start. `todcor_batch` runs a survey's worth of stars in one call and
records failures instead of stopping on them.

This does not reopen the non-goal below on synthesis: the templates come from the disentangling,
from a label match, or from the published grids of `albireo.library`. What it adds is the front-half
job the package was missing, the table, and the position that when the spectra are known, measuring
velocities against them is faster, simpler and works on one spectrum, which is the split Zucker
himself drew between correlation and disentangling.

### 11. One command for a list of stars — **done** (D58)

Every stage above existed as a function, and a survey, or a first-time user, does not want to call
them one after another for every star. `albireo.pipeline` is the driver: read the epochs,
disentangle, fit labels to the components, measure the epoch velocities against them, fit the orbit
to the table, write the products and the figures, for every star in a TOML file, in worker
processes, with the failures recorded and the results in a table. `albireo init`, `albireo run` and
`albireo demo` are its command line, and the configuration is the façade's own vocabulary rendered
as TOML with the standard library, which is the ordering this page asked for below: the CLI came
after the façade so that it would not freeze a second schema.

The driver makes no scientific decision of its own, which was the design constraint: light fractions
are still required and the wavelength medium is still declared before a grid is consulted, and where
a request cannot be honoured the star gets a flag and the batch carries on. What it adds is the
composition the pieces had promised: the label fit's measured frame offset is applied to each
component's own template, so the epoch velocities come out absolute and the orbit fitted to them
recovers the systemic velocity the disentangling alone cannot see; the demo's toy star returns
+11.9 ± 0.2 km/s against an injected +12.

The degeneracy it found appeared in the first demo run. With a symmetric semi-amplitude prior the
conjunction scan sees the declared component assignment and its mirror, with spectra swapped and
rescaled by the light ratio, as equally deep troughs, and the toy star came back with its components
exchanged. The resolution is a convention, stated rather than hidden: components are declared in
order of decreasing mass, the fit is started with K₁ < K₂, and the label stage flags a fitted light
fraction far from the declared one as the signature of a wrong order. Finding it exposed a façade
defect (per-component `start_at` values were silently dropped), now fixed. The scaling of a batch
across worker processes is measured in [the benchmark record](benchmarks.md): 2.0× for four workers
and 2.5× for eight, and the per-worker thread cap, kept as a precaution, turned out to be worth
nothing measurable there.

## Tier 3 — later, and why later

**Time-variable component spectra.** The core assumption of disentangling is that each component's
spectrum is constant; the systems people most want to study, Be stars with variable discs,
interacting binaries and pulsators, violate it. Tier 2's nebular component is already rank-one time
variation (a fixed shape with a free per-epoch amplitude), and that captures more than it sounds
like: "the disc emission was 1.4× stronger that night" is most of what anyone models. Line shape
variation needs a second basis vector per variable component, with a shrinkage prior and the same
windowing machinery. The structural point is that this stays inside the linear-Gaussian family, so
the analytic marginalization survives: it is a change of basis rather than a change of method. It
waits for the dataset that demands it, which per the D38 record is HR 6819, where disc variability
is now one of only two surviving explanations for the period offset.

**specutils interoperability.** Accept and return `Spectrum` objects at the boundary; do not adopt
them internally. The reason for the asymmetry is concrete: `SpectrumCollection` requires
equal-length spectra and rejects per-epoch metadata differences, so it cannot represent a ragged
multi-epoch multi-instrument dataset, which is albireo's central input. Writing down why is itself
worth doing, because it names a problem the reader has already met.

**More archives.** SOPHIE/ELODIE is the cheapest (no authentication, decades of high-resolution
planet-search cadence); LAMOST's medium-resolution time-domain survey is the largest by volume;
SDSS-V has the largest catalog of double-lined candidates but the most download plumbing, and its
visit spectra must be read in the un-shifted form.

**Survey-scale throughput.** Batch fitting and a measured systems-per-GPU-hour number. This waits on
GPU hardware, which is also the one open acceptance gate from M5; the current scale projections are
extrapolated from CPU measurements and say so.

~~**A command-line interface.** Deferred until after the façade, because the façade is the
configuration schema and building a CLI first would freeze a second one. When it happens, the
subcommand with real value is `albireo fetch`, not `albireo fit`.~~ Done (Tier 2 item 11, D58):
`albireo run` is a TOML rendering of the façade's vocabulary, and `albireo fetch` is in it.

**A guard at zero eccentricity.** The `(√e cos ω, √e sin ω)` parameterization is singular at exactly
the origin: ω is undefined, `e·cos ω` behaves like `|x|`, and the gradient is NaN. numpyro then
reports `Cannot find valid initial parameters`, which points nowhere near the cause. Since tidally
circularized close binaries are exactly the population a user is most likely to bring, an
initialization at `secosw = sesinw = 0` is a predictable first experience, and it warrants a message
that says so rather than a documentation note. Cheap, and worth doing before the façade freezes the
initialization API.

**Community mechanics.** Discussions enabled, contributions labeled, and, the highest-conversion and
most-neglected item, writing to the people whose systems albireo is built for and offering to run
one. A user who leaves with a disentangled spectrum of their own star is a user.

## Explicit non-goals

Recorded so that they stop being reconsidered.

- **No telluric radiative-transfer fitting and no molecfit wrapper.** Several wrappers already exist
  and are fragile. Masking and downweighting telluric regions inside the likelihood is already
  supported and is the right tool; exact multiplicative treatment remains the recorded v2 seam.
- **No WEAVE- or 4MOST-specific loaders.** 4MOST's processed data is released through the ESO
  archive, so it arrives with the loader in Tier 2.
- **No synthesis code and no abundance pipeline; still no replacement for GSSP, iSpec, Korg.jl,
  PHOEBE, or PySME.** Interoperate instead. Being the front half of everyone's pipeline is a better
  position than being a worse version of the back half. `albireo.match` (D52–D55) does not
  reopen this. It fits four labels, Teff, log g, [M/H] and *v* sin *i*, to a disentangled
  component against established public grids, for the front-half jobs: choosing the
  right RV template, pinning the per-component velocity zero point, and checking an assumed
  flux ratio. It synthesizes no spectrum, carries no line list, solves no radiative transfer,
  and fits no individual abundances; its module docstring states that first, before it says what it
  does do. The moment a question needs bespoke synthesis, abundances, microturbulence, or an
  eclipsing-binary model, the answer remains `albireo.handoff` and the codes above. Nor does
  `albireo.todcor` (D56) reopen it: it correlates against templates the user already has, being
  the disentangled components, a label-matched grid point, or a library rendering, and synthesizes
  none of its own.
- **No GUI**, per the v1 non-goals.

## What would change this page

The roadmap above assumes the bottleneck is reach rather than capability. Three observations would
falsify that and reorder everything: real BLOeM spectra failing to load or to disentangle sensibly;
the K₂ null calibration showing the detection statistic is far less powerful than the HR 6819 work
implies; or the GPU gate closing with numbers materially worse than the CPU projections, which would
make throughput the story instead of statistics.
