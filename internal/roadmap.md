# Roadmap

This page records the planned development of albireo and the reasons for it, in enough detail for
the order of the work to be questioned. The plan is subject to change: items are reordered, and an
item that meets no user need is deleted. Decisions, once made, are recorded in the
[decision record](design.md#2-decisions-recorded-defaults).

## Position relative to existing codes

Three facts about the existing software determine the strategy.

**The technique is mature and no maintained implementation of it exists.** Spectral disentangling
has been in use since Simon & Sturm (1994) and Hadrava (1995). Every code that implements it is
either unmaintained or difficult to obtain. KOREL is written in Fortran and was last released
in 2011. Its source is not distributed: it is available only through a web service that requires
registration and has a limit of 16,384 bins. FDBinary is marked deprecated in its ASCL entry. Its
successor, fd3, is still used in papers published this year; its distribution page has not been
updated since 2014 and states no license. Spectangular's last papers are from 2017 and 2019. UNWIND,
released in 2026, is written in IDL, and its paper describes it as lacking a manual or an
installation package. The shift-and-add code used to identify the companions in LB-1 and HR 6819 is
a small research script with no license file, which is a barrier to anyone who would build on it.
For the authors of these codes, disentangling is a means to a scientific result and the software a
by-product. As a result there is no maintained, installable, tested implementation of the technique.
In the three years to mid-2026, roughly eight papers a year named it in the abstract, and many more
used it in a methods section.

**Uncertainties on the disentangled spectra are the most requested capability, and no existing code
provides them.** The disentangled spectra are almost never the final product: they are passed to an
atmosphere code to obtain effective temperatures, gravities and abundances. The uncertainties on
those quantities are therefore systematically understated across the literature. The papers state
this explicitly: TMBM III notes that normalization uncertainty is not propagated into the quoted
properties. More careful accounting of uncertainties does not fix this. Disentangling is an
ill-conditioned linear inverse problem with a low-frequency null space, and obtaining a reliable
error band from it requires a posterior, which means either sampling a high-dimensional space or
marginalizing over it. albireo marginalizes. That capability is the reason the package exists, and
every item on this page follows from making it usable.

**The direct approach was tried and was limited by computing cost.** PSOAP (Czekala et al. 2017) put
a Gaussian-process prior on the component spectra and sampled the joint posterior. It is the closest
direct predecessor, and its last commit is from December 2017. The paper states the reason: the cost
of the dense GP restricted it to narrow bandpasses and required cluster access. albireo's analytic
marginalization, banded structure and JAX implementation address that limitation: the component
spectra are integrated out in closed form rather than sampled, and the linear algebra is banded
rather than dense. The [benchmark record](../docs/benchmarks.md) is long because it keeps that claim
testable.

The plan is ordered accordingly: make the existing capability usable (Tier 1), then reach the
communities whose problems it already solves (Tier 2), then extend the model (Tier 3).

## Tier 1: usability (done)

A capability is useful only once it can be installed, cited and plotted. These items added no new
science and gave the largest available return on effort. They are done. What remains of this tier is
pushing, tagging and registering, not code. Their order matters and is given in
[Releasing](releasing.md).

| | Item | Reason it came first |
|---|---|---|
| 1 | `slow` / `network` / `gpu` markers and a `conftest` | Every later item adds tests; the fast suite went from 14 minutes to 6 |
| 2 | Single-sourced version, upgraded `CITATION.cff`, `CHANGELOG.md`, a [citing](../docs/citing.md) page, a tag-driven release workflow | The documentation could not state `pip install albireo` until this was in place |
| 3 | Deployed docs with a rendered [API reference](../docs/api/index.md), `py.typed`, coverage | 70 exported names had docstrings that were rendered nowhere |
| 4 | `load_example()` and a [five-minute quickstart](../docs/quickstart.md) that runs offline | Time to first plot is the best-attested factor in adoption |
| 5 | `albireo.results`: arviz conversion, save/load, export to FITS/ECSV/ASCII | A fit lasting hours produced only printed text |
| 6 | `albireo.plotting` | The three most reusable plotting functions in the repository existed only inside example scripts |

The reasoning for two of these items is recorded below.

**Export.** The product is the disentangled spectrum with its uncertainty band. Until it could be
written in a format that an atmosphere code reads, albireo was the middle stage of a pipeline with
no connection to the next stage. The uncertainty, its main distinguishing feature, was lost at the
step where it is needed. Tier 2 item 8 is the other part of this work.

**The quickstart must not require finding data.** It follows the pattern of lightkurve: the first
code block performs an analysis rather than configuration, on data the user did not have to locate.
A small simulated dataset is therefore included in the wheel, with no download, no astropy and no
archive account. A real archival example is the second step, once the ESO loader below exists.

## Tier 2: user communities

Each item addresses a specific group of users with a specific unmet need. The items are ordered so
that earlier ones make later ones possible.

### 1. A nebular component with a free per-epoch amplitude (done, D40)

Massive stars form in H II regions, so their spectra contain nebular emission lines that do not move
with either star and vary in strength from night to night with seeing and slit losses. Left in the
data, they artificially narrow the disentangled line profiles and bias the derived temperatures and
gravities. The literature treats this case by case. TMBM III models the nebular features as a third
component with static velocity and variable intensity, constructed manually. A 2026 paper in the
same series masks the contaminated pixels by setting their error values to 999.

In albireo this is a static component whose per-epoch amplitude is a free parameter. Structurally it
is the telluric component with a traced amplitude instead of a fixed one. It is physically correct,
since nebular flux is added to the total continuum and takes no light from the stars. Two details
required most of the work. The nebular component is static in the barycentric frame, which is the
opposite convention from the telluric one. Confining it to the Balmer, He I and forbidden-line
windows required the regularization strength of the smoothness prior to become per-pixel
(`SmoothnessPrior(tau_profile=, eta_profile=)`, built by `priors.window_profile`).

It had the largest effect per line of code on this page, measured by the closed-loop test. An SB2
whose Hβ absorption contains a static nebular line disentangles to a core 26% too shallow and an
equivalent width 11.5% low without the component, against 0.14% with it. Equivalent-width errors
propagate to the atmosphere code, so that is a systematic error in log *g*, of a kind the current
literature does not propagate. The effect on the orbit was larger. A static line is a component with
*K* = 0, so a joint fit without the nebular component assigns the emission to whichever star can be
made to move least. It returns K₂ 59% low, a period long by 0.171 d, and *e* = 0.95 (the solver's
clip) for a circular orbit. The contamination therefore biases the masses as well as the atmospheric
parameters. The component introduces two degeneracies, which are resolved by convention and not by
the data: the amplitude scale, fixed by centering the log-amplitudes, and the nebular velocity,
which is a placement convention for the window profile and not a measurement. Both are recorded in
[the mathematical foundations](../docs/math.md#13-lsf-light-fractions-response).

### 2. Calibrated faint-companion detection (done, D41)

The K₂ scan already determined whether a companion is present and at what velocity semi-amplitude,
which is the central question for every dormant-black-hole candidate. It did not give the rate at
which noise alone would produce the peak that it found. The state of the art in the literature is a
χ² map with no false-alarm probability attached, and the papers state explicitly that small
deviations in the assumed primary semi-amplitude produce spurious features in the recovered
secondary spectrum.

All three parts are built. The scan is vectorized: `MarginalOrbitModel.log_likelihood_sweep` runs
the trial grid as one batched `lax.map` instead of a Python loop with a device synchronization per
point. K₁ can be marginalized (`k2_scan(k1_sigma=)`) with a Gauss–Hermite rule applied to both the
companion and the no-companion model, so `D` remains a ratio of two marginal likelihoods.
`albireo.calibrate.detection_limit` runs the injection-recovery calibration. It resimulates through
the operators of the observed data (`forward.with_data`, `simulate.resimulate`), so a few hundred
scans take under a minute.

A K₁ that is 10% high reduced the correlation of the recovered companion's line pattern with the
injected pattern from 0.96 to 0.49 and more than tripled `D`, so the artifact appears as a stronger
detection. A calibrated threshold cannot detect this, because the null trials are drawn under the
same wrong assumption; only the marginalization can. The two are complementary, and this is stated
wherever either is documented. The calibration provides the statement "any companion contributing
more than *X*% of the light would have been detected at 95% confidence". Its threshold is
conservative by construction, and its false-alarm probability is never reported at a finer
resolution than `1/(n_null + 1)`.

The limit is nearly flat in K₂ (0.292 / 0.296 / 0.297% at K₂ = 20 / 40 / 65 km/s), where a
dependence had been expected, so computing time need not be spent on mapping it. In an SB2 the two
components move in antiphase, so their relative velocity never drops below about K₁, and with
K₁ = 55 km/s the pair is well separated at every trial. A K₂ dependence should appear only when K₁
is small. The remaining dependence is on the assumed companion template, since the observable is
ℓ₂·d₂ and a featureless companion is undetectable at any light fraction. That assumption must be
quoted with the limit. The [HR 6819 campaign](../docs/benchmarks.md) remains available as a
validation set.

### 3. A per-epoch radial-velocity table (done, D42)

Free per-epoch velocities, with no Keplerian, were listed in the [design](design.md) as a diagnostic
mode and had never been exposed. The low-level code existed and was differentiable, and only the
sampling path was missing. The mode is now a `velocity` theta site that replaces the orbit, with
`relative_velocities`, `relative_velocity_errors` and `keplerian_residuals` alongside it.

The item is small and has two benefits. A per-epoch RV table with quoted uncertainties is the
product that the binary-star community expects from any spectroscopic analysis. It also gives users
of cross-correlation and shift-and-add codes a familiar quantity to compare before they rely on the
joint fit. In addition it is the model check for the Keplerian mode: fit free velocities, then test
whether a Keplerian is consistent with the resulting posterior.

The table is differential, and it has one arbitrary zero point per component, not one in total: with
no orbit linking the stars, each free spectrum absorbs a constant added to its own shifts. Left
uncentered, that zero point is set by shift-interpolation error and not by the data, since a common
shift of a whole pixel changes the log-likelihood by 4e-9 in relative terms and one of 0.1 pixel
changes it by 7.3 nats. albireo removes the zero point in pixel space, where the removal is exact.
The same projection is needed for the uncertainties. The raw Laplace diagonal returns only the prior
(37.95 km/s = 120/√10 on every entry, against an actual per-epoch error of 0.059 km/s), a value that
does not depend on the quality of the dataset.

Warm-started from a Keplerian 30% wrong in both semi-amplitudes, the per-epoch RVs are recovered to
0.098 / 0.066 km/s, 1/60th of a model pixel, and the Wilson slope to 0.4%. From a cold start the
mode necessarily fails, at a potential 122,000 nats worse, so the failure is evident.

`examples/09_rv_table.py` is the worked script. It is organized around the two counter-intuitive
properties. It demonstrates the per-component zero point by shifting one star's velocities by
50 km/s and showing that the log-likelihood does not change (0 nats for the relativistic shift,
8.7e-6 for the ordinary shift, which is only its first-order approximation). It prints the raw
Laplace error bars beside the projected ones: 37.947 km/s on every entry, which is `120/√10`, the
prior, against a measured 0.056-0.065. The explanation is in
[the mathematical foundations](../docs/math.md#76-free-per-epoch-velocities-the-rv-table) and
`tests/test_velocity_table.py`.

### 4. An ESO archive loader and one-line access to BLOeM (done, D44, D45)

The motivation was BLOeM: ~929 targets in the Small Magellanic Cloud with roughly 25 epochs each, an
intrinsic binary fraction above 70%, and 59 published double-lined systems, all public. The survey
team's own disentangling of them is still listed as future work in their July 2026 review. A real
BLOeM target, fetched with little code and disentangled with posteriors, serves at once as the
tutorial, as the evidence of research use that software journals ask for, and as a paper. The item
was placed after the nebular component because BLOeM's targets lie in nebulosity and the
demonstration needs that component to be reliable.

Both parts are built. `albireo.archive` (D44) is the ObsCore/TAP client and resumable downloader.
`albireo.io` (D45) reads the downloaded files by dispatching on the IVOA utypes rather than on
column names, and `resolve_bloem` / `bloem_catalogue` / `bloem_spectra` convert a survey identifier
into that star's epochs. `ab.bloem_catalogue(binary_class="SB2")` returns the 59 double-lined
targets, and `examples/06_bloem.py` takes one of them from its name to fitted spectra.

The assumption of a single file layout does not hold. Thirteen real Phase 3 spectra from seven
instruments were read column by column, and no two collections agree on anything except the utypes.
Flux is `FLUX`, `FLUX_REDUCED`, or both at once. The extension is `SPECTRUM` except for Gaia-ESO's
`phase3spectrum`. Units are angstrom, Angstrom or nm. Keying on UCDs, the obvious fix, makes the
problem worse: UVES gives its sky-background column the same UCD that HARPS gives its flux column,
so a reader keyed on UCDs fits the sky background without any symptom. Only the utype role
distinguishes the two columns.

All thirteen files could already be read. The defects were in the metadata and the weights. `medium`
was computed and then dropped before it reached `EpochData`, so the 83 km/s guard of D43 could never
trigger. Quality flags were ignored. A zero uncertainty was treated as infinite precision. An
all-NaN error array was accepted with no warning. A rewrite aimed only at reading the files would
have fixed nothing visible to a user.

A BLOeM epoch is ~178 kB and one star ~5 MB. `112.25R7` holds 23,651 spectra over 929 targets, of
which 21,716 were public on 2026-08-13, and every target has at least one public epoch. There is
also a third programme's worth of additional data: `115.28A9` re-observes the same stars at
*R* = 17000 and 23000 in two other windows. These 1,827 spectra must not be pooled with LR02 under
one line-spread function, so the resolver defaults to the survey programme.

An adversarial review of the change found five defects that it had introduced. In all five, an
assumed value was treated as one read from the file. The most serious assumed the standard's
convention that a quality flag of zero is good, but `STATUS` in UVES_SQUAD takes the values
`{-5, 1}` and never 0, so all 467 products in that collection were read as 100% bad and raised an
error. A flag whose convention cannot be read is now ignored with a warning, not inverted. The same
principle rejects an undeclared air or vacuum wavelength scale: when the header is ambiguous the
reader returns no value and does not infer one.

[The tutorial](../docs/tutorials/bloem-sb2.md) takes one target from a survey identifier to
disentangled spectra with an uncertainty band. Writing it exposed two problems. First, the example's
window was described as "between Hδ and Hγ without either core", which was incorrect. The range
4000–4300 Å contains Hδ at 4101.7, and `nebular_windows` puts a ±300 km/s window at 4099.7–4107.9
inside it, so the script's stated reason for not modelling the nebula was false. The window is now
4120–4300 Å, which contains no nebular line, and the docstring explains why the blue edge is the
critical value. Second, the `Disentangler` interface did not support the order of analysis that the
survey requires. With no published period there is nothing from which to warm-start a Keplerian, so
the free RV table (item 3) has to come first and supply the periodogram. However,
`Fit.free_velocities()` warm-starts from a Keplerian fit, which needs a period.
`Disentangler(velocities=...)` now declares measured velocities (cross-correlation lags, line
splitting) in place of an `Orbit`, and `fit()` returns the free table directly. The declaration
rejects a cold table and warns when the components are never resolved, so the one failure mode of
the free-velocity mode remains visible. Warm-started from velocities with 3 km/s of scatter and a
150 km/s systemic offset that the fit cannot determine, the table is recovered to
0.096 / 0.070 km/s, the same as D42's Keplerian-warm-started 0.098 / 0.066. The systemic offset is
absent from both the result and the solver's bandwidth.

### 5. The `Disentangler` interface (done, D46)

The [design](design.md#6-api-sketch-target-user-code) sketched a simpler API and the README
announced it. Before this item a new user had to build a dictionary of numpyro distributions
manually, call three functions in the right order, extract two hyperparameters for empirical Bayes,
and rebuild the model. That is workable for an expert and a poor introduction for a new user.

It was placed after the nebular component and the RV mode because the interface defines a
vocabulary, and a vocabulary is expensive to change once it is in use. The components and modes that
it names should exist before it names them. It is released marked experimental, with the low-level
API documented and supported in parallel.

The interface is a compiler rather than a shortcut: the user declares the system (components, light
fractions, instrument, what is known about the orbit) and it emits the expert path. `dis.explain()`
prints every derivation and `dis.expert()` returns the exact `(model, priors, init)` triple, so
parallel support of the low-level API follows from the design. Four quantities are derived, each of
which the low-level path leaves to the user: the velocity budget (from the support of the `k`
priors), the grid margin, the conjunction phase and the smoothness hyperparameters. The conjunction
phase is found by a scan, with 10⁵ nats between the best and worst phase on the packaged example.
The smoothness hyperparameters are found by ML-II and reported with a flag on any that did not move
from its starting value.

The names rejected from the sketch are part of the decision. Six of the twelve were out of date, and
releasing them would have made a false statement. `Keplerian(t_peri=, ecc=, omega=, k1=, k2=)` names
five sites that no longer exist under those names. `GaussianLSF` is neither Gaussian-only (D38) nor
one number per instrument (D37). `light_ratio=` is the wrong term for a per-epoch simplex over N
components. `PerEpoch(eclipse model)` would have named a model that does not exist, which the
ordering rule above is meant to prevent. `dis.replace(orbit=FreeVelocities())` is the usage that D42
measured to fail at 122,000 nats worse, so the free-velocity mode is instead a method of a completed
fit, where it cannot be constructed without its warm start.

On the packaged example the interface takes 12 lines against 59 on the expert path. It gives MAP
*K* = 41.978 and 62.978 against an injected 42 and 63, *e* = 0.1512 against 0.15, residual z-RMS
0.997, and NUTS *K*₂ = 62.988 ± 0.081 with zero divergences.

Jitter, AR(1), inferred light fractions and inferred LSF widths are excluded from v1 by decision.
Each is a one-line site at the low level, and each is a scientific claim, which a keyword such as
`jitter=True` would present as a convenience. `fit.z_rms` is printed unconditionally so that the
need is visible, and `expert()` is the way to meet it.

An adversarial review of the finished module confirmed 38 defects. Not one accepted name was
disputed. The defects were at every place where a declaration had to be translated into the model's
own conventions, and that layer is where nothing downstream can check the result. The smoothness
rows were assembled in declaration order while the model orders them stars-telluric-nebular, so a
permuted declaration regularized the wrong component. The difference was 9,900 nats, and it passed
every existing guard because the vectors still had the right length. `ecc=Between(lo, hi)` was not
the prior that was fitted: the corner of the box reached `e = 2·hi`, and `lo` was validated and then
ignored. Both are fixed and both have regression tests. Three declarations that the interface could
not honour, namely hierarchical triples, Gauss-Hermite `h3`, and an eccentricity lower bound, are
rejected and not approximated.

The risk is recorded in [the decision record](design.md#2-decisions-recorded-defaults) and in the
module docstring: `Star(light=0.62)` appears beside `period=Known(40.335, 0.5)` and is formatted
identically, but one is a choice and the other a measurement.

### 6. `sensitivity_forecast()` (done, D47)

Whether twelve more epochs at given phases will break the degeneracy is a question about the
posterior covariance of the component spectra. The posterior covariance does not depend on the flux
values, only on the epochs, the phases, the weights and the prior, so it can be computed for
observations that have not been taken. Every part needed already existed, and no other code provides
this forecast, which justified the two days it took.

`albireo.forecast` is the module. `plan_epochs` builds the epochs of an observation that has not
been taken, `sensitivity_forecast` returns what those epochs would contribute, and `baseline=` names
the epochs already obtained, so that the result is a difference rather than an absolute value. The
independence from the flux is structural: the precision is assembled directly and the right-hand
side is never formed. The regression test overwrites every flux with noise a hundred times the
continuum and requires a bit-identical forecast. Every quantity is reported against the same
quantity under the prior alone. This applies the result of D42: a band that has relaxed to the prior
looks the same as one constrained by the data.

The main result corrects the premise of this item. §5.1 names Var(Δ), the spread of the differential
shift, as the observing-strategy diagnostic. It is the right quantity but the wrong objective. A
cadence aliased to the orbital period samples the two extreme values of Δ repeatedly: it maximizes
the variance and is a poor design, because two values leave |g(k)| recurring to *J* at a comb of
scales. The following was measured in
[`examples/08_forecast.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/08_forecast.py)
on a 13.7 d circular SB2 with eight epochs already obtained and twelve to plan:

| twelve planned nights | RMS Δ*v* | blind fraction | 2nd mode σ | information gain |
|---|---|---|---|---|
| at P/2 (aliased) | 117.8 km/s | 58% | 0.518 | 243 nats |
| continuing the existing cadence | 115.7 km/s | 56% | 0.106 | 295 nats |
| spread over phase | **99.3 km/s** | **33%** | **0.071** | **375 nats** |

The aliased plan is best in the one column that §5.1 would have had the observer maximize and worst
in every other. The closed form serves to screen designs and to explain the result; the forecast
comes from the assembled covariance.

Computed over the whole model grid, the worst-determined mode is at exactly the prior width. The
model grid is wider than the data by design, so its margin pixels are constrained only by the prior
and are the largest eigenvalue of Σ on nearly every real problem. That value measures only how much
margin the grid was given. Restricted to the coordinates that the design weights, the leading mode
is the one that the theory predicts: a delocalized exchange mode across the components at *k* = 0.
It is at ~1× the prior for every design, since no choice of phase sampling can constrain it. It is
reported in the output and not suppressed, and designs differ in how quickly the remaining modes
fall below it.

The orbit is not forecast, by decision. The Fisher information for a velocity depends on
∂(model)/∂v ∝ ℓᵢdᵢ′, so an error bar on *K*₂ needs the line depths, which have not yet been
measured. A forecast against an assumed template would present the assumption as a result.

### 7. A Gaia RVS loader, before December (the part that can be built now is built; D60, D61)

The following were built on 2026-09-09. `albireo.gaia` is a simulator of the instrument. It contains
the S/N model from G_RVS (verified against the archive to 3-8%), the detector and delivered grids
with the correlated noise that the archive's resampling introduces, the in-flight resolving powers
per CCD row, and the scanning-law cadence in distribution. `albireo.population` provides
double-lined binaries from the field's distributions, from DEBCat and from the Gaia DR3 double-lined
orbits. `albireo.benchmark` runs every system through the pipeline under knowledge tiers and
produces a report (`scripts/gaia_rvs_benchmark.py`). The reader and the DataLink client are left for
release day and will be validated against the simulator's product presets. The rest of this item
records the reasons for the ordering.

Gaia DR4 is scheduled for 2 December 2026 and is the first release to publish epoch RVS spectra:
6,910,785,949 of them (49 TB), already normalized and already in the barycentric frame, alongside
several hundred thousand spectroscopic-binary orbit solutions. Having a working loader on the day of
the release is a one-time opportunity.

An earlier version of this page stated that "a loader written against DR3's shape now becomes a DR4
loader on release day". That statement is false, for three reasons, each checked against a primary
source:

**DR3's mean spectra have the velocity removed irreversibly.** The archive data model states that
"the spectra are in the rest frame", and the shift is applied per transit with that transit's own RV
before co-adding. The orbital modulation is therefore not a recoverable offset: it is smeared out in
the co-added spectrum. For an SB2 the shift used a single blended cross-correlation RV that was
wrong for both components. DR3 `rvs_mean_spectrum` is neither a disentangling dataset nor a
demonstration, and the loader should raise an error for it rather than warn.

**DR3 has no hot stars.** `SELECT MAX(rv_template_teff) FROM gaiadr3.gaia_source WHERE
has_rvs='true'` returns 14,500 K over all 999,645 published spectra, and none are above 15,000 K.
Spectra are published only for sources with a radial velocity, and the RV template grid ends there.
albireo's demonstrated science case, O and early-B stars, is outside it, and so is HR 6819 (source
6649357561810851328, G = 5.26, `has_rvs=false`).

**The window is poorly suited to early types.** For a hot star, 846–870 nm at *R* ≈ 11,500 contains
the Paschen series P13–P17 and the Ca II triplet: one species, Stark-broadened, mutually blended,
just longward of the Paschen jump. Compared with the He I / He II / Mg II / Si III of the
4380–4600 Å window in the [benchmarks](../docs/benchmarks.md), that is a mismatch for hot stars.
Gaia's own SB2 population is mostly cooler, and in cooler stars the triplet is the strongest feature
in the spectrum.

### DR4 epoch RVS spectra in the draft data model

ESA pre-released a draft DR4 data model on 2026-06-26. It settles the questions on which the design
depends. The answers are recorded here so that they are not derived again in December:

| | |
|---|---|
| Table | `rvs_epoch_spectrum`, DataLink only, not through the main TAP interface |
| DataLink retrieval type | `EPOCH_SPECTRUM_RVS` |
| Grid | 961 elements, 846–870 nm, step 0.025 nm, not DR3's 2401 × 0.01 nm. At R ≈ 11,500 that is 1.3 bins per LSF σ against DR3's 3.2, below the ≥ 3 px/σ threshold that the TODCOR work measured for pixel-locking, so template correlation on these spectra is expected to pixel-lock. The joint model is unaffected, because it uses its own grid and not the data's. |
| Frame | "shifted from the Gaia reference frame to the barycentric reference frame". Normalisation is conditional, and `normalisation_method` records which was applied: 0 leaves the spectrum unchanged and is applied to every double-lined transit, 3 divides flux and `flux_error` by the same median, 4 marks a median-negative spectrum. The SB2 population is therefore delivered un-normalised while the reference population is median-scaled. The loader has to branch on the byte and must not assume a pseudo-continuum near 1. |
| Time | `obs_time_rv`, Barycentric JD in TCB − 2 455 197.5 d, Roemer-corrected to the barycentre |
| Uncertainties | `flux_error[961]`, propagated per bin; NaN where every contributing CCD was masked |
| Per-pixel coverage | `combined_ccd_in_index`, a sparse `short[]` paired with a second `short[]` named `index` that holds the flux-array positions it refers to; both are null when every bin used all CCDs. It is not a per-pixel array of length 961: it has to be scattered into `index`. |
| Selection | `all_source_flags.has_epoch_rvs`, a byte graded by `external_apparent_grvs` rather than by spectrum S/N: 0 no spectrum, 1 G_RVS > 14 (unusable on its own, no epoch RVs), 2 for 12 < G_RVS ≤ 14 (epoch RVs), 3 for G_RVS ≤ 12 (epoch RVs and broadening velocities). Class 3 is the usable population. The draft's text places the column in `all_source_rvs` in one section and in `all_source_flags` in another. It is defined in `all_source_flags`, and the two tables have different sizes (2.79e9 rows against 3.12e8), so the SB2 query joins both. To be re-checked on release day. |

Two of these are decisive for the design. There is a per-transit barycentric timestamp. Whether one
existed was the one unknown that could have made the product unusable for albireo. The fluxes are
linearly interpolated onto that fixed grid before publication, which conflicts with
[D4](design.md#2-decisions-recorded-defaults): albireo does not resample observations onto a common
grid, because resampling correlates the noise and invalidates the diagonal `ivar` model. Gaia has
resampled before publication and albireo cannot undo it, so the published `flux_error` understates
the correlation. That is a systematic to state in the loader's docstring, and possibly a use for the
AR(1) code (D34). It is not a reason not to build the loader.

Epoch spectra are produced for double-lined, emission-line and contaminated transits, the ones whose
radial velocity the pipeline rejects. `rv_assumed_sb2` marks any source where double lines were
detected in at least ten transits, which gives an SB2 target list that can be queried with ADQL.
When that flag is set, Gaia excludes the source from its own multi-transit RV solution and publishes
the spectra anyway.

The conclusion is to build the loader in December, against the release, and not before. The DR4
documentation tree (`archive/documentation/GDR4/`) still returns 404, and DR3 shares neither the
grid, the frame, the table nor the retrieval type. This project's own D45 record is that an assumed
value treated as one read from the data is a costly kind of defect. Before then, the step worth
taking is the go/no-go query on release day: join the BLOeM DR3 `source_id`s that `resolve_bloem`
already retrieves against DR4, and check whether `rv_assumed_sb2` includes any early-type
population.

This item also required a small correctness fix, which has already been made (D43) because the
archive loader of item 4 needed it first. RVS wavelengths are in vacuum and most optical
spectrographs deliver air wavelengths, and there was no field in which to declare the medium.
`EpochData(medium=...)` is now that field, `Dataset` rejects a mixture, and `air_to_vacuum` /
`vacuum_to_air` convert. The description "sub-ångström" understated the effect: the offset is
0.87-2.74 Å, but as a velocity it is a nearly constant 83 km/s, the same order as the orbits being
measured.

### 8. Downstream handoff (done)

The current workflow in the literature is a chain of separate tools: measure velocities, fit an
eclipsing-binary model, disentangle in one code, renormalize manually against an external light
ratio, then fit atmospheres in another. That is five tools and five format conversions, and the
uncertainty is lost at the disentangling step because the disentangling code produced none.

albireo is the first half of that pipeline and does not attempt to be the second half, which GSSP,
iSpec, Korg.jl and PySME already provide. `albireo.handoff` provides the writers `write_gssp`,
`write_ispec` and `export_draws`, with [a tutorial](../docs/tutorials/downstream.md) and
`examples/10_downstream.py`.

The file formats were the difficult part, and each has a failure mode that gives no symptom. iSpec
does no unit conversion on text input: its whole internal scale, line lists included, is in
nanometres, so a value in ångström is a factor of ten outside every model grid and a fit is still
returned. GSSP infers its synthetic step from the supplied file: "the step width in wavelength that
will be used for the calculation of synthetic spectra is computed from the observations"
(Tkachenko 2015, Appendix B, which is the entire manual; there is no separate document and no source
repository). albireo solves on a log-wavelength grid, whose linear spacing varies by 1.32% across
the window of the packaged example, so `write_gssp` resamples onto an equidistant grid rather than
writing one from which GSSP would compute the wrong step. Both are regression-tested.

GSSP accepts no per-pixel uncertainty, which makes the draws necessary. Its configuration files
contain no error path, no S/N entry and no weighting entry, and its own error bars come from χ² on
the fit residuals. The posterior band therefore cannot be propagated to a temperature through the
file. It can be propagated only by fitting *N* spectra and taking the spread, which is the purpose
of `export_draws`.

The refitting loop is established and the draws are new, and the difference between them is
measured. Kiran et al. (2016, §3.5) added "artificial Gaussian noise with sigma = sigma_c" to a
disentangled profile and refitted 500 times, and should be cited. That procedure assumes that the
error is white, and disentangling error is not: it has a low-frequency null space, which changes a
continuum and, through it, a surface gravity. `draw_spectra` returns `d_hat + L⁻ᵀz` on the vector
stacked over all components, so the draws are correlated across wavelength and across the two stars.
The equivalent width is the appropriate proxy for the atmosphere code, since D40 established that an
EW error propagates to it and that an 11.5% EW error is a systematic in log *g*. With that proxy,
the following was measured on the packaged example:

| | EW (Å) | joint draws | independent per-pixel noise | ratio |
|---|---|---|---|---|
| component 1 | 0.2568 | 0.01873 | 0.01041 | **1.80×** |
| component 2 | 0.0417 | 0.02974 | 0.00881 | **3.38×** |

White noise understates the integrated uncertainty two- to three-fold. The band suffices for a
pointwise question, but every atmospheric parameter depends on an integral over the spectrum.

The correlation between the equivalent widths of the two components across draws is −0.992, against
−0.052 for the same statistic under independent noise. This is the *k* = 0 exchange mode of D47,
which is delocalized and at ~1× the prior for every observing design, appearing in a derived
quantity. The two stars exchange line depth almost exactly, so their difference is far better
determined than either alone, and fitting the components separately with independent error bars
misstates the result in both directions. It is also the clearest reason for keeping the draw index
in the exported filename.

The posterior does not include the model error of the atmosphere code. It also does not include the
light ratio, which albireo conditions on rather than marginalizes, and which Pavlovski & Hensberge
(2011) call the dominant systematic. Both are in the caveat list of the tutorial.

### 9. A benchmark page against the established codes (done)

A new code is trusted once it has reproduced the established ones. Both comparison codes are built
and run, and the AI Phoenicis comparison is done. The numbers are in
[the benchmark record](../docs/benchmarks.md).

AI Phe provided the cross-validation and one further result. From 36 archival HARPS spectra, started
15% off, the eccentricity is recovered as 0.1879 against a published 0.1878 ± 0.0006. A TESS light
curve and a spectroscopic disentangling thus agree to 0.05% by independent methods. The
semi-amplitudes have a reproducible ~1% systematic (K₁ −1.4%, K₂ +0.8%, mass ratio 2.2% toward
unity). It is not caused by the optimizer, by line selection (two disjoint windows agree to 0.02%)
or by the light ratio (9 nats out of 53,306). It is recorded as an open question and not as a
correction to a literature value quoted to 0.02%.

The run also exposed a defect that no simulation would have. A poorly chosen window spanned the gap
between the two HARPS CCDs, 32.9 Å of exact zeros at 5304.67–5337.61 Å. These pixels were weighted
as data: they are finite and have no quality flag, and HARPS provides no error array, so their
inverse variance was estimated from the small scatter of a flat run of zeros. The disentangled
component spectra had negative flux. `albireo.mask_flux_gaps` now detects contiguous runs of
non-positive flux and reports them. This is the same kind of failure as D45, one level up.

`scripts/shift_and_add.py` is the clean-room implementation, written from González & Levato (2006)
§2.1–2.3 and Quintero et al. (2020) and from no source code: the existing implementation has no
license, so it was never opened. It is validated against the theory in the paper: §2.3 derives that
the residual is diffused rather than annihilated, by a Gaussian of `√(2m)·σ_d` after *m* sweeps, and
seeding a delta-function error reproduces that law.

Results of the three codes on identical data (aligned RMS and wall-clock time):

| | comp 1 | comp 2 | wall-clock time | uncertainty returned |
|---|---|---|---|---|
| **albireo** | **0.0093** | **0.0116** | 0.182 s | **yes** |
| fd3 | 0.0198 | 0.0223 | 0.111 s | no |
| shift-and-add | 0.0248 | 0.0302 | **0.018 s** | no |

albireo is ~2× more accurate and, on the machine on which this table was recorded, the slowest of
the three. Shift-and-add is 10× faster than albireo and 6× faster than fd3. For a 1200-pixel
separation that is the expected result, since the method consists of a few array shifts and means.
The accuracy margin is not an artifact of the stopping point: at 50 sweeps instead of the
published 7, shift-and-add improves by 11% and is still 2.4× less accurate.

> **Speed re-run of 2026-08-16.** All three codes were re-run on one machine (a 16-core
> desktop) under one protocol. Every RMS above reproduces exactly; the wall-clock times do
> not. They are shift-and-add 0.026 s, albireo 0.059 s, and fd3 0.064 s single-threaded,
> which is parity, or 0.104 s as built, with 32 OpenBLAS threads active. The 0.018 s above
> was partly an artifact. The harness timed shift-and-add in-process after the XLA solve,
> on a heap that XLA leaves serving ~35 KB allocations through microsecond free-list
> walks. That convention affected both recorded numbers and has since been replaced by
> timing in a fresh process. Shift-and-add therefore remains fastest everywhere, "slowest
> of the three" applied to single-thread hardware, and albireo remains ~2× more accurate
> and the only code returning a posterior. The full record is in benchmarks.md,
> "Re-run of the three-code comparison on one machine" (D50).

The three methods are independent and fail in the same way. fd3's raw error is nine tenths a
constant, and shift-and-add's grows on the fainter component because `B = 0` leaves its continuum to
the initialization. Both are the *k* = 0 null space, and the shift-and-add theory gives this
exactly: the per-mode convergence factor has modulus 1 at zero frequency, a fixed point that no
number of sweeps changes.

The prebuilt binary in the fd3 tarball is 32-bit i386 and does not run on a modern host, so the code
was rebuilt from source. Before it was used for anything it was validated against the author's own
shipped outputs, and it reproduced the published example (V453 Cyg, 1344 px) exactly with a
different compiler, architecture and GSL. It is not vendored here, since the distribution still
states no license.

The result is informative in both directions. fd3 is 1.64× faster in steady state on a 1200-pixel
SB2 separation, as a small compiled C program should be in that regime. albireo is in the same class
and not an order of magnitude slower; the harness's own un-jitted figure would have overstated the
difference by 20×. fd3's raw RMS is ~15× larger, but nine tenths of that is a constant, which
mean-alignment removes. The *k* = 0 offset is the null space that neither code can determine from
constant-light data: fd3 leaves it to the user's manual renormalization, and albireo's prior fixes
it. With that offset removed, albireo is about 2× more accurate in shape. fd3 returns no
uncertainty, which is the missing capability that this page addresses.

Three rules keep the comparison fair. The comparison system should be AI Phoenicis rather than
HR 6819: it eclipses, so the light ratio is known externally and the one free choice in
disentangling is no longer a confound, and its semi-amplitudes are published to 0.02%. The
shift-and-add comparison must be a clean-room implementation from the published algorithm, because
the existing code has no license. The comparison is presented as agreement, speed and posteriors,
and never as "the old code is wrong". Where results differ, the page reports a diagnosis and not a
judgement. A fair comparison also has to restrict albireo to fd3's conditions (common grid, uniform
weights) before showing the unrestricted case separately.

~~The single most useful figure on that page is the cheapest: an SB2 with a nebular line,
disentangled by a method that can mask the contaminated pixels and by methods that structurally
cannot.~~ Withdrawn: the comparison was unfair. González & Levato explicitly permit "any combination
algorithm... weights or some rejection algorithm", so masking is part of the published shift-and-add
method and not an extension of it, and `tests/test_shift_and_add.py` demonstrates a zero-weighted
epoch being excluded. A figure whose conclusion is a capability that the comparison code has would
have been the "the old code is wrong" presentation that this section rules out. What the established
codes cannot do is report an uncertainty, and the page makes that comparison instead.

### 10. Epoch radial velocities for every component, by TODCOR (done, D56, D57)

The joint fit does not measure a per-epoch velocity, which is the correct behaviour when the
component spectra are unknown. A per-epoch velocity is also the one product from which every
eclipsing-binary analysis, survey pipeline and orbit code starts, and without it a user from a
cross-correlation background found nothing familiar in albireo. `albireo.todcor` is the
two-dimensional correlation of Zucker & Mazeh (1994), in which each spectrum is correlated against a
combination of two templates with independent shifts. Blended peaks then do not bias each other, and
a faint companion can be measured from a single spectrum. albireo generalizes it to any number of
components and to weighted, masked, multi-instrument data by writing it as the least-squares fit to
which it is equivalent. On a uniform grid with uniform weights it reproduces the published formulae
to 1e-10. On real data the masks, gaps, cosmic-ray hits and per-pixel weights enter through the same
operators as the forward model and leave the estimator unchanged.

Three features go beyond a port of the published method. The sub-pixel minimum is computed rather
than read from a parabola: the shift operator is linear in the template, so the chi-square is an
exact quadratic inside each pixel cell. The errors are the maximum-likelihood errors of
Zucker (2003), with the blending and detection diagnostics that a batch run needs. The components of
a disentangling can serve as templates: `Fit.templates()` converts them and
`Fit.measure_velocities()` runs the epochs against them. The zero point, which a disentangled frame
cannot identify, is stated on every table. `albireo.rvorbit` fits the Keplerian to the table with
the same solver and conventions as the joint model, and returns the elements as a warm start.
`todcor_batch` runs the stars of a survey in one call and records failures instead of stopping on
them.

This does not reopen the non-goal on synthesis below: the templates come from the disentangling,
from a label match, or from the published grids of `albireo.library`. It adds the table, which was
the first-half task that the package lacked. It also adopts the position that when the spectra are
known, measuring velocities against them is faster, simpler and works on one spectrum, which is the
distinction that Zucker drew between correlation and disentangling.

### 11. One command for a list of stars (done, D58)

Every stage above existed as a function, and calling them one after another for every star does not
suit a survey or a first-time user. `albireo.pipeline` is the driver. For every star in a TOML file
it reads the epochs, disentangles, fits labels to the components, measures the epoch velocities
against them, fits the orbit to the table, and writes the products and the figures. It runs in
worker processes, records the failures and collects the results in a table. `albireo init`,
`albireo run` and `albireo demo` are its command line. The configuration is the vocabulary of the
`Disentangler` interface written as TOML with the standard library. This is the order that this page
specifies below: the CLI came after the interface so that it would not freeze a second schema.

The driver makes no scientific decision, which was the design constraint. Light fractions are still
required, and the wavelength medium is still declared before a grid is consulted. Where a request
cannot be honoured, the star is flagged and the batch continues. The driver adds the composition of
the stages. The frame offset measured by the label fit is applied to each component's own template,
so the epoch velocities are absolute. The orbit fitted to them recovers the systemic velocity, which
the disentangling alone cannot determine. The toy star of the demo returns +11.9 ± 0.2 km/s against
an injected +12.

With a symmetric semi-amplitude prior, the declared component assignment and its mirror, with
spectra swapped and rescaled by the light ratio, are equally deep minima of the conjunction scan, so
the scan can exchange the components. The degeneracy is resolved by a stated convention. Components
are declared in order of decreasing mass and the fit is started with K₁ < K₂. The label stage flags
a fitted light fraction far from the declared one as the signature of a wrong order. Finding the
degeneracy also exposed a defect in the `Disentangler` interface (per-component `start_at` values
were silently dropped), which is fixed. The scaling of a batch across worker processes is measured
in [the benchmark record](../docs/benchmarks.md): 2.0× for four workers and 2.5× for eight. The
per-worker thread cap, kept as a precaution, made no measurable difference there.

## Tier 3: later work

**Time-variable component spectra.** The core assumption of disentangling is that each component's
spectrum is constant. The systems of greatest interest, Be stars with variable discs, interacting
binaries and pulsators, violate it. Tier 2's nebular component is already a rank-one time variation
(a fixed shape with a free per-epoch amplitude). That covers more than it may seem: a statement such
as "the disc emission was 1.4× stronger that night" is most of what is modelled. Line-shape
variation needs a second basis vector per variable component, with a shrinkage prior and the same
windowing code. This stays inside the linear-Gaussian family, so the analytic marginalization still
applies: it is a change of basis rather than a change of method. It waits for a dataset that
requires it, which according to the D38 record is HR 6819, where disc variability is now one of only
two remaining explanations for the period offset.

**specutils interoperability.** Accept and return `Spectrum` objects at the boundary, and do not
adopt them internally. `SpectrumCollection` requires equal-length spectra and rejects per-epoch
metadata differences, so it cannot represent a ragged multi-epoch multi-instrument dataset, which is
albireo's central input. This reason should be written down, because it names a problem that the
reader has already met.

**More archives.** SOPHIE/ELODIE requires the least work (no authentication, decades of
high-resolution planet-search cadence). LAMOST's medium-resolution time-domain survey is the largest
by volume. SDSS-V has the largest catalog of double-lined candidates and requires the most download
code, and its visit spectra must be read in the un-shifted form.

**Survey-scale throughput.** This is batch fitting and a measured number of systems per GPU-hour. It
waits on GPU hardware, as does the one open acceptance test from M5. The current scale projections
are extrapolated from CPU measurements and labelled as such.

~~**A command-line interface.** Deferred until after the `Disentangler` interface, because that
interface is the configuration schema and building a CLI first would freeze a second one. When it
happens, the subcommand with real value is `albireo fetch`, not `albireo fit`.~~ Done (Tier 2
item 11, D58): `albireo run` reads the vocabulary of the `Disentangler` interface written as TOML,
and `albireo fetch` is included.

**A guard at zero eccentricity.** The `(√e cos ω, √e sin ω)` parameterization is singular exactly at
the origin: ω is undefined, `e·cos ω` behaves like `|x|`, and the gradient is NaN. numpyro then
reports `Cannot find valid initial parameters`, which gives no indication of the cause. Tidally
circularized close binaries are the population that a user is most likely to bring, so an
initialization at `secosw = sesinw = 0` is a predictable first use, and it warrants a message that
states the cause rather than a documentation note. The guard is cheap, and worth adding before the
`Disentangler` interface freezes the initialization API.

**Community mechanics.** Enable Discussions, label contributions, and write to the people whose
systems albireo is built for, offering to run one. The last has the highest conversion and is the
most neglected. A person who receives a disentangled spectrum of their own star becomes a user.

## Explicit non-goals

These are recorded so that they are not reconsidered.

- **No telluric radiative-transfer fitting and no molecfit wrapper.** Several wrappers already
  exist. Masking and downweighting telluric regions inside the likelihood is already supported and
  is the appropriate method. Exact multiplicative treatment remains the recorded extension point for
  v2.
- **No WEAVE- or 4MOST-specific loaders.** 4MOST's processed data is released through the ESO
  archive, so it is covered by the loader in Tier 2.
- **No synthesis code and no abundance pipeline; still no replacement for GSSP, iSpec, Korg.jl,
  PHOEBE, or PySME.** albireo interoperates with them instead. Being the first half of other
  people's pipelines is a better position than being a weaker version of the second half.
  `albireo.match` (D52–D55) does not reopen this. It fits four labels, Teff, log g, [M/H] and
  *v* sin *i*, to a disentangled component against established public grids, for the tasks of the
  first half: choosing the right RV template, fixing the per-component velocity zero point, and
  checking an assumed flux ratio. It synthesizes no spectrum, contains no line list, solves no
  radiative transfer, and fits no individual abundances. Its module docstring states this before it
  states what the module does. A question that needs bespoke synthesis, abundances, microturbulence,
  or an eclipsing-binary model is still referred to `albireo.handoff` and the codes above.
  `albireo.todcor` (D56) does not reopen it either: it correlates against templates that the user
  already has (the disentangled components, a label-matched grid point, or a library rendering) and
  synthesizes none.
- **No GUI**, per the v1 non-goals.

## Conditions that would change this plan

The roadmap assumes that the bottleneck is reach rather than capability. Three observations would
falsify that and reorder the plan. The first is real BLOeM spectra failing to load or to disentangle
sensibly. The second is a K₂ null calibration showing that the detection statistic is far less
powerful than the HR 6819 work implies. The third is the GPU acceptance test finishing with numbers
materially worse than the CPU projections, which would make throughput the main concern instead of
statistics.
