# Adoption plan (2026-09-03)

This page sets out what would make albireo the code that binary-star spectroscopists choose,
and in what order to do it. It is an input to [roadmap.md](roadmap.md), as the roadmap is to
the [decision record](design.md#2-decisions-recorded-defaults). It proposes a Tier 0 that the
roadmap lacks, re-orders Tier 3, and adds items requested in the literature since the roadmap
was written.

The evidence is seven investigations run on 2026-09-03 and kept in
[research/2026-09-03-adoption/](research/2026-09-03-adoption/README.md). They are cited below
by letter: **A** the field's stated needs and the competition, **B** archive access and Gaia
DR4, **C** the physics of the candidate model extensions, **D** a read-only audit of the code,
**E** the AI Phe semi-amplitude offset, **F** a fresh install by a new user, **G** release and
community mechanics. Facts marked *verified* were checked against the primary source on that
date. Everything else on this page is judgment.

## 1. Summary

The bottleneck is reach, as the roadmap assumed, but two of the obstacles are not the ones it
lists. The documentation site does not exist. GitHub Pages was never enabled, the deploy job
is conditional on a repository variable that was never set, and the README's green Docs badge
links to a 404 page (F, G, *verified*). The quickstart's first line is `pip install albireo`,
and the package is not on PyPI (F, *verified*). Nothing on the roadmap matters to a user who
cannot install the package or read its documentation, so Tier 0 below is a week of account
actions and small fixes, not code.

After that, the order is set by three findings.

1. The field's literature names as open problems the two outputs that only albireo provides.
   Tkachenko's group writes that for disentangled spectra "observational uncertainties are not
   well-defined" (Serebriakova et al. 2025). BLOeM's B-supergiant paper states that "it is
   necessary to disentangle the spectra first in order to get true RV estimates of the
   companions" (Britavskiy et al. 2025), and the survey's JAX and NumPyro toolkit has no
   disentangling module (A, *verified*). The capability exists. What is missing is a
   demonstration on a system those groups recognise, and a message to them.
2. Two model extensions are requested more widely than any other, and both are cheap within
   the linear-Gaussian model. One is a light fraction that varies with wavelength or with
   epoch under the user's declaration (fifteen verbatim requests across a dozen groups, A §7).
   The other is time-variable component spectra (the stated obstacle in the most-cited strand
   of the literature, A §7.9). Section 5 gives their designs. The first is a diagonal operator
   with no bandwidth cost, and the second begins as the existing per-epoch light mode with the
   simplex constraint released (C).
3. The largest single event in the field's calendar is Gaia DR4 on 2 December 2026, with
   787,312 double-lined RVS transits and 6.9 billion epoch spectra (B, G, *verified*), and no
   tool prepared for it exists (A §6.9). LAMOST's medium-resolution time-domain survey became
   public worldwide on 2025-09-26 with 12,426 SB2 candidates and 665 published orbits, and is
   the trial run for Gaia (B, *verified*).

Sections 3 to 7 are the plan, section 9 is the calendar, and section 11 lists what only the
maintainer can decide.

## 2. What the investigations found

### 2.1 Availability and release state of the package (F, G, D)

- `https://tjayasinghe.github.io/albireo/` returns 404. The repository has no Pages
  configuration, no `PAGES_ENABLED` variable, and every Docs run shows `Deploy: skipped`. A
  comment in `docs.yml` gives the fix: enable Pages with the Actions source and run
  `gh variable set PAGES_ENABLED --body true`.
- `albireo` is free on PyPI (both the simple index and the JSON API return 404 against a
  working control). PyPI has no reservation mechanism, and Starfish lost the name `starfish` to
  a transcriptomics package and is distributed as `astrostarfish` (G). The name should be
  claimed now.
- There are no tags, releases, issues or pull requests, and 64% of the 72 commits fall in the
  first six days. JOSS requires six months of public history "with evidence of releases,
  public issues and pull requests" and grades a single burst of commits as not acceptable
  (G, *verified*). Waiting satisfies the calendar clause on 2027-02-14 but not the evidence
  clause.
- `pyproject.toml`, `CITATION.cff`, `codemeta.json` and `mkdocs.yml` all describe the package as
  "GPU-accelerated", while `paper/paper.md` keeps that phrase out of its title until the M5
  acceptance criterion is met (D §7). The criterion is not met. The consumer GPU ran fp64 at
  1/50 of the fp32 speed and ran out of memory at one sixth of the design target, so the claim
  needs an A100- or H100-class device to be true (D §10.4).
- `paper/paper.md` was drafted to an older JOSS template. It lacks the now-required State of
  the field, Software design, Research impact and AI-usage-disclosure sections, and its
  affiliation is the literal string `TODO` (G).

### 2.2 First run by a new user (F)

A new user on this machine, under load from other jobs, recorded these timings: install 5 s
from a warm cache (107.7 MiB cold, 37 packages), import 0.8 s warm, the `Disentangler` fit
93 s, and `fit.sample()` 686 s. No warnings were raised, and the semi-amplitudes were
recovered to 0.05% and 0.03% on all three entry paths (README, manually built `Dataset`,
`albireo demo`). The recorded showcase timings are 37.8 s and 601 s on an idle machine. The
shortcomings are in what the run reports. The README quickstart prints nothing during
`sample()` and does not save its figure. `post.summary()` omits eccentricity, argument of
periastron, R-hat and effective sample size, and gives no units. The default fit reports
"stopped at the step cap" with a gradient norm of 383 while the result is correct to 0.05%.
`repr(Dataset)` is 313,693 characters long. `albireo --version` does not exist.
`Star("primary")` fails with a bare `TypeError` on the one argument whose requirement most
needs explaining. Each has a one-line fix (F, ranked list).

### 2.3 Needs stated in the literature (A)

Of 36 papers since 2023 whose method is disentangling, thirteen use shift-and-add, eight
FDBinary or fd3, six a code the authors wrote themselves, five TODCOR, two KOREL, one UNWIND,
none Spectangular and none PSOAP. The six groups that wrote their own disentangler in three
years are the potential users. The rate of such papers is about 10 per year in 2025 and 2026,
up 50% on 2019 to 2022, and eight of the ten most-cited disentangling papers of 2020 to 2026
are compact-object papers.

The relevant quotations, all verified, follow:

- "For disentangled spectra ... observational uncertainties are not well-defined. The
  disentangled spectra are derived products that depend on the orbital solution, meaning
  their quality (and thus their uncertainties) evolves dynamically throughout the optimisation
  process." Serebriakova, Tkachenko, Johnston, Pavlovski and Aerts 2025 (arXiv:2507.10096).
  albireo's conditional Gaussian at each posterior draw defines these uncertainties.
- "It is hard to trace the source of this inconsistency because the same observed spectra and
  the same disentangling code were used in both studies." Moharana et al. 2026 on V446 Cep
  (arXiv:2604.25056). Two groups with one HERMES dataset and one code, FDBinary, obtained
  different K2, traced to eclipse cuts, Balmer inclusion and segment choice, none of which
  appear in a published result. 33 of 72 spectra were discarded for being in eclipse.
- "In a future version we will implement the possibility of the subtraction of ISM emission
  lines ... being refined on an epoch-by-epoch basis." UNWIND (arXiv:2604.02111), describing
  D40's nebular component as future work.
- "Whether or not the companion is classified as 'non-degenerate' is left as a subjective
  decision, based on a visual inspection." Shenar et al. 2022, who also confirm D41's finding
  that "even small deviations in K1 can result in spurious features ... which can lead to an
  erroneous detection of a non-degenerate companion."
- "We have found a suitable [regularisation] for each (simulated) dataset by trial and error;
  automatisation of this is an endeavour for future work." Seeburger et al. 2024. ML-II on
  `(tau, eta)` provides that automation, and the docs do not state it.
- Maxted et al. 2026 (arXiv:2607.17976) wrote a new Simon and Sturm solver for AI Phe and
  treated manually every degeneracy albireo handles structurally. They zeroed in-eclipse UVES
  rows to break the light-ratio degeneracy, grid-searched the flux ratio per HARPS order,
  removed a quadratic from the A minus B difference, and quoted one scalar S/N for the
  product. The same archival data, a published result and openly hosted benchmark spectra are
  available.

### 2.4 The competition (A §6, *verified* 2026-09-02/03)

BiSpeD (Python, MIT, pushed the day before this survey) does the SB1 companion scan on a
mass-ratio and temperature grid, with a manually set flux ratio and no false-alarm
probability. UNWIND is IDL, unlicensed, and "can only be shared without support". DERVIS is a
pure-Python FDBinary rewrite with no licence and no packaging. MESS (Nachmani, Faigler and
Mazeh 2026) is multi-epoch TODCOR with joint template optimisation over the labels that
`albireo.match` fits plus the flux ratio, and announces a LAMOST DR11 application. It is the
direct competitor to `albireo.todcor` plus `albireo.match`, with a stated limit of a 10%
companion at S/N 50. No new disentangling code has been registered on ASCL since 2017. No
other JAX or PyTorch disentangler exists, and no Gaia DR4 SB2 tool exists. Only PSOAP, dormant
since 2017, ever returned a posterior on the spectra. Spectangular has at most one citation a
year.

### 2.5 Decomposition of the AI Phe offset and the design gap it indicates (E)

Gallenne et al. 2019 measured their velocities from 33 of the same 36 HARPS spectra, and
Maxted et al. 2020's reference values are a weighted mean dominated by them, so no property of
the data can explain albireo's offset. Three independent studies agree on the mass ratio to
0.02%, whereas albireo's offset is 2.17%. In the model's coordinates, a component-common
velocity error moves K1 and K2 oppositely and a component-antisymmetric one moves them
together:

| | value | effect |
|---|---|---|
| common, `(dK1 - dK2)/2` | -0.55 km/s | the entire mass-ratio offset (-2.20% predicted, -2.17% observed) |
| antisymmetric, `(dK1 + dK2)/2` | -0.16 km/s | K1 + K2 low by 0.32% |

No epoch is in or within 0.02 in phase of an eclipse, so the Rossiter-McLaughlin effect and
in-eclipse light changes are excluded. The naming convention is confirmed from the spectra
(the Mg b lines identify the K subgiant). Two disjoint windows agree in q to 0.01%, which
excludes every noise-driven cause. Three candidate causes remain, ranked as follows. (1) Six
of the 36 spectra are HARPS EGGS exposures at R = 80,000, modelled with the HAM R = 115,000
line-spread function because both modes have `INSTRUME = 'HARPS'`. Five of the six are at one
orbital phase, the positive extremum of v1. (2) A stationary third light of 0.3% at 5200 Å,
which Maxted measures from TESS at 0.5 to 1.0% and Gallenne's third body supports, gives
exactly the symmetric reduction in K. (3) The light fraction has no first-order effect on K
and was tested only at two fixed orbits. The tests are in section 4.1. The design gap is that
the reader reads `SPEC_RES = 80000` into `RawSpectrum.resolving_power` and `to_epoch` then
discards it, because the line-spread function is keyed by instrument name (D §9).

### 2.6 The archives (B, *verified*)

- LAMOST MRS DR11 v2.0 is public worldwide without registration through a documented REST API
  and SSAP. One gzipped FITS file per observation holds the coadd and every single exposure.
  Single exposures are on native pixels with an `IVAR` column and `VACUUM = T`, and the
  heliocentric shift is already applied (proven on a constant-velocity star). The coadds are
  resampled and have a measured lag-one noise autocorrelation of +0.4. Kovalev et al. 2024
  list 12,426 SB2 candidates, Guo et al. 2025 publish 665 orbits, and Wang et al. 2025 list
  126 published SB2s that are moonlight false positives.
- The HERMES archive is public through anonymous TAP and serves an IVOA Spectrum Data Model
  VOTable, the dialect the ESO reader already dispatches on. It has 119,650 spectra, real
  `BJD` and `BVCOR` keywords, resampling at 1.56 km/s and no error array.
- SB9 was superseded on 2025-06-24 by SBX, which has a TAP service with 1,326 SB2s, 1,699
  orbits with a measured K2, and 224,148 velocity measurements labelled by component.
- Gaia DR4 is still due on 2 December 2026. The roadmap's record needs four corrections.
  First, `combined_ccd_in_index` is sparse and paired with an `index` column. Second,
  `has_epoch_rvs` is in `all_source_flags`, not `all_source_rvs`, and is graded by G_RVS, with
  only class 3 (G_RVS below 12) usable. Third, the SB2 spectra are un-normalised
  (`normalisation_method = 0`) while the rest are median-scaled. Fourth, the 0.025 nm grid
  samples the line-spread function at about 1.3 pixels per sigma, which matters for template
  correlation and not for the joint model, whose grid is chosen independently of the data's.
  The wavelengths are vacuum (measured from Ca II). `rvs_epoch_parameters_double` gives
  per-transit velocities for both components, the barycentric correction, per-component
  template labels and an `is_reordered` flag (the ordering hazard of D58).
  `nss_two_body_orbit` gives SB2 solutions with `g_luminosity_ratio`, a prior on the one
  quantity disentangling cannot measure. The DataLink endpoint accepts anonymous
  standard-library requests at 5,000 sources per call, and the Gaia Observation Forecast Tool
  returns barycentric transit times for any position today, so the forecast half can be built
  and tested before release.
- SDSS-V's `mwmVisit` product is rest-frame and unusable. Only SDSS-IV DR17 `apVisit` works
  (2.66 million visits, observed frame, vacuum, Kounkel et al. 2021's 7,273 SB2s), with a
  dither combination whose uncertainty correlation SDSS documents as ignored. CFHT ESPaDOnS
  through CADC is the best-formatted product found (errors, unmerged orders). SOPHIE and
  ELODIE serve no errors, truncate the Julian date of protected epochs, and put their products
  in different velocity frames. Keck HIRES extracted spectra may have been shifted to the
  night-sky lines with no keyword recording it, so no loader should be built.

### 2.7 The physics of the extensions (C)

Every extension considered remains within the linear-Gaussian family, so the analytic
marginalization, the band structure and the Takahashi recursion are retained. What changes is
which new quantities are linear in the spectra and which are nonlinear.

- Hadrava's per-epoch line-strength factors are albireo's existing per-epoch light mode with
  the simplex constraint released. Hadrava's line photometry shows that the constraint encodes
  an assumption that is wrong in general (a weak line dims more than the continuum during an
  eclipse), so releasing it is the extension. It requires the same centering the nebular
  amplitude already has, because only the ratio of the factors between components is
  identified.
- A wavelength-dependent light fraction is a diagonal operator applied after the shift and the
  line-spread function, in the observed frame, with zero bandwidth cost. Under constant light
  its shape is unidentified to about one part in ten thousand of the line depth, so it must be
  declared and not fitted. Its benefit is a renormalisation and a label fit that are correct
  across a broad window. Under eclipses it is identified, which is Hadrava's approach to
  in-line limb darkening.
- No study marginalises the disentangling posterior over the light ratio, and sampling it
  inside the model under constant light returns the prior. The marginalisation belongs
  downstream, where the map from posterior draw to renormalised spectrum is affine, so a draw
  of the light fraction with each spectrum draw gives the exact induced distribution. The
  first-order rule is `dW/W = -dl/l` for every equivalent width.
- A broadening function or LSD profile is obtained from the same normal equations albireo
  assembles, with the latent variable moved from the spectrum to a per-epoch profile. The
  precision is then block diagonal by epoch. No published broadening function is solved with
  a smoothness or GP prior in place of SVD truncation, and Rucinski's argument against
  truncation (it is non-local in velocity and can create asymmetry) is also an argument for a
  curvature penalty.
- The shift-and-add community's K1-K2 chi-square map has no published contour level, and
  colour-bar minima range from 1.0 to 28. The profile of the marginal likelihood is the same
  figure with the spectra integrated out rather than counted as parameters. On it the
  two-parameter level `Delta = 2.30` is defensible, and `p_eff` addresses Hynes and Maxted's
  1998 objection that the separated spectra are being treated as free parameters.
- Be discs are excluded by every published study (Halpha dropped from HR 6819, LB-1, MWC 656,
  ALS 8814). The cheapest treatment that reports the variability rather than absorbing it is a
  windowed per-epoch jitter. The model extension is a rank-R basis per variable component with
  free per-epoch coefficients, bilinear in the same way as the strength factors.

## 3. Tier 0: ship (September)

These items are preconditions for anyone using the package, and none of them is science. They
are ordered so that nothing has to be undone.

1. **Enable GitHub Pages** (source: Actions, done 2026-09-03), set `PAGES_ENABLED` (done
   2026-09-03; the first deploy runs on the next push to `main`), and confirm the site
   renders. Add a step that fails the Docs workflow on `main` when deploy is skipped, so the
   badge cannot be green without a site. Enable Discussions. Set the repository homepage and
   topics (currently "Spectral disentangling" with no topics; keep "spectroscopic binaries"
   adjacent to "disentangling" everywhere, because the term has other meanings outside
   astronomy).
2. **Register the name on PyPI** through the existing trusted-publishing workflow: the TestPyPI
   rehearsal first, then the real index. Add `Documentation` and `Homepage` to
   `[project.urls]` and the `Typing :: Typed` classifier.
3. **Fill the metadata**: ORCID and affiliation in `CITATION.cff`, `codemeta.json` and
   `paper/paper.md`. Guard the `codemeta.json` version as the `CITATION.cff` version is guarded
   (`tests/test_metadata.py`), since that file feeds ASCL and ADS. Link the Zenodo webhook
   before the first tag (with no `.zenodo.json`, Zenodo reads `CITATION.cff`, which is the
   intended path).
4. **Tag `v0.1.0`** with a CHANGELOG section and an API-stability statement modelled on
   NumPy's NEP 23 (two releases or one year between a deprecation warning and removal; an
   incorrect result is worse than an error). Move the classifier from Pre-Alpha to Alpha, and
   remove "GPU-accelerated" from every description until the M5 acceptance test has been run
   on a data-centre device. Add `CODE_OF_CONDUCT.md` (pyOpenSci requires it) and issue
   templates.
5. **From this point, work through issues and pull requests**, including the maintainer's own:
   the plan items below become issues, and each change is merged as a PR. This meets the JOSS
   evidence clause and also serves as the public roadmap.
6. **Fix the first-run text** (F): one install snippet shared by README, quickstart and site;
   the README quickstart printing `dataset.summary()` and `fit.summary()`, sampling with the
   progress bar on, and saving its figure; `post.summary()` reporting eccentricity, argument of
   periastron, R-hat and effective sample size with units; a one-line `__repr__` on `Dataset`
   and `EpochData`; `albireo --version` and help on bare invocation; `Star` raising an
   explanatory error for a missing light fraction; `Dataset.medium`; the step-cap message
   stating that the orbital sites are converged and the residual gradient lies along the flat
   ML-II directions, or a higher default step cap.
7. **Correct the documents** listed in section 10. Done 2026-09-03.
8. **Propose an EAS 2027 special session** (Vienna, 21 to 25 June 2027; proposals close
   30 September 2026) on spectroscopic binaries in the Gaia DR4 era, ideally with the
   eclipsing-binary and massive-star groups named in section 7.2 as co-proposers.

## 4. Tier A: the science that adoption depends on

New codes are trusted after they reproduce old ones on systems the reader knows. Each item
here ends in a number that can be quoted.

### 4.1 Close the AI Phe offset, and fix the design gap it exposed

The tests, in the order E ranks them, each take a few minutes of computation on the existing
data:

1. Give the six HARPS EGGS epochs their own instrument key with the R = 80,000 width, and
   separately drop them. If the component-common term moves, the cause is found.
2. Add a stationary third component and sweep its light fraction over 0 to 1.5%. The symmetric
   term should vanish near 0.3%. In the low-level interface this is a tertiary row at zero
   velocity.
3. Refit the orbit at five values of the secondary's light fraction, which tests the fitted
   K rather than the likelihood at two fixed orbits.
4. Rescale the weights to the measured z-RMS of 2.64, or add the jitter site. The driver is
   deterministic, so this should narrow the answer rather than move it.
5. Use the published period at its full precision (`24.592483` d), start window 2 at 5341 Å
   so its pad does not enter the CCD gap, and report the signs of `secosw` and `sesinw` with
   `t_conj` instead of the eccentricity alone.
6. Run NUTS on AI Phe. No posterior exists on either real-data system (D §10.2), and the
   methods paper needs one on a benchmark.

The design change makes the line-spread function a property of the epoch, not of the
instrument name. `EpochData` gains an optional `lsf_sigma_kms` (or `resolving_power`),
`to_epoch` keeps the value the reader read, and operator groups are keyed on the pair
(instrument, width). The `lsf=` argument of `Disentangler` accepts per-epoch values, and
pooling two resolving powers under one key raises a warning that names the epochs. HARPS is
not the only instrument with two modes. FEROS, UVES and X-shooter settings differ between
programmes on the same target.

### 4.2 The K1-K2 map with a contour on it

`log_likelihood_sweep` already batches trial grids. A two-dimensional sweep over (K1, K2) with
the other sites profiled, plotted in the community's convention, with the `Delta = 2.30`
contour and `p_eff` printed on the panel, is the figure every shift-and-add user will look for
first. Produce it twice on one dataset: with the nuisance sites held fixed (the like-for-like
comparison) and with them marginalised (the improvement). The difference between the two
contours is a result.

### 4.3 Reproduce Maxted et al. 2026 on AI Phe, with a band

The archival HARPS and UVES data are the same, the result is published, and the paper
enumerates eight manual steps the model removes (A §4.1). The reproduction exercises three
items of section 5 (per-epoch light fractions from the eclipse, a declared
wavelength-dependent light fraction, the light-ratio uncertainty propagated into the
equivalent widths), and the author publishes his benchmark spectra openly.

### 4.4 A fourth code on ALS 8814

El-Badry et al. 2025 ran Seeburger's code, fd3 and shift-and-add on 26 public LAMOST spectra
and published the comparison. Adding the one code that returns a posterior is the cheapest
credibility gain available and is the comparison without a verdict that roadmap Tier 2 item 9
prescribes. It depends on the LAMOST loader (section 6.2), and the emission-line variability
the paper reports is the first real test of section 5.5.

## 5. Tier B: model extensions, in order

Each entry states what is linear, what is nonlinear, what is degenerate, and what would
falsify the design. Costs are estimated from D's audit of where each change is made.

### 5.1 Light-ratio uncertainty carried downstream (days)

`Fit.spectra`, `export_draws` and `Fit.match_labels` accept a distribution on the light
fractions. For each posterior draw of the spectra, a draw of the light fractions is taken and
the affine renormalisation applied. The exported spectra and the spread of the label fit's
draws then include the photometric uncertainty as well as the disentangling uncertainty.
Report `dW/W = -dl/l` and the `1/l_i` amplification beside every renormalised spectrum. This
removes the caveat `docs/science.md` §7 states today. It must not be done by sampling the
light fraction inside the model, which under constant light returns the prior (C §4).

### 5.2 Per-epoch light fractions and third light in the `Disentangler` interface (days)

The low-level interface accepts fixed `(n_stellar, n_epochs)` fractions and a traced `light`
site, but `Star.light` is one float and can express neither (D §1). Add
`Star(light=PerEpoch(array))` and `Star(light=FromLightCurve(callable of phase))`, so a
JKTEBOP, ellc or PHOEBE solution supplies the fractions, and a `ThirdLight(fraction)`
declaration. Under constant light a third light is a rescaling and the declaration has no
other effect. Under eclipses it is the quantity that fixes the scale of the per-epoch
fractions (C §3), so it must be declared and has no default, like the light fractions.
In-eclipse spectra are then modelled rather than discarded. The V446 Cep analysis dropped 46%
of its spectra, and those are the ones that measure the light ratio.

### 5.3 Per-epoch line-strength factors (days)

`Star(strength="free")` places a positive per-epoch factor on that component. It is
implemented as the existing `light` site without the simplex check (which
`with_light_fractions` already skips), centred per component so that the geometric mean over
epochs is one, with a log-normal prior of declared width. The report prints the
between-component ratio per epoch, which is Hadrava's line photometry and the only
well-identified combination. This is KOREL's line-strength mode and is what an eclipsing
binary without a light-curve solution, a Be star whose disc strengthens, or a chromospherically
active star needs at rank one. The test is the eclipse closed loop with the simplex
deliberately violated.

### 5.4 A declared wavelength-dependent light fraction (one week)

This is a per-pixel diagonal after the shift and the line-spread function, applied on the
model grid in the observed frame. The only substantial cost is a gather in
`assembly.epoch_body`, whose index array already exists two lines below (D §1.5). It is
declared as a low-order polynomial in wavelength or as the continuum ratio the label fit's
grids provide through the shared radius ratio. The latter gives a two-pass procedure:
disentangle, fit labels and radius ratio, derive `l_i(lambda)`, disentangle again. The docs
must state the identifiability result from C: under constant light the shape is not
measurable, so it is supplied as external information like the constant ratio, and `Free` is
rejected for it. Add the row to the decision record.

### 5.5 Time-variable components: jitter first, then a basis (weeks)

The first of two steps is a windowed per-epoch jitter confined to declared lines (Halpha,
He I, the Ca II triplet). It is the cheapest treatment of a variable disc that widens the
posterior where the model is wrong instead of absorbing the variability into the spectra, and
`fit.z_rms` per window measures the misfit. The second is a rank-R basis per variable
component with free per-epoch coefficients, which generalises 5.3 from a scalar to R shapes.
It adds one row per basis vector to the forward operator and one more `(tau, eta)` pair, and
the row count is then not equal to the number of stars (D §3.3). The cost grows as the square
of the row count in the band and linearly in the bandwidth, so a rank-2 SB2 costs about four
times the block work. The deciding dataset is HR 6819 or ALS 8814. Do not build the basis
before running the jitter on one of them.

### 5.6 The profile mode (weeks; a methods paper on its own)

`albireo.profile` holds the component spectra at templates, takes a per-epoch velocity profile
per component as the unknown, and solves the same normal equations with the same smoothness
prior and the same ML-II hyperparameters. The precision is block diagonal by epoch, so the
mode is cheaper than disentangling. It yields broadening functions and LSD profiles with a
posterior covariance, which no published code provides (C §5), the SB2 velocities the
contact-binary and massive-star LSD communities already trust, and a residual profile that
shows when two components are not enough. TODCOR is this problem under a delta-function prior
on the profile, and the docs should describe it in these terms.

### 5.7 Noise and triples as declarations (days)

The `Disentangler` interface rejects jitter, AR(1), inferred line-spread widths and
hierarchical triples (D §5), all of which the low-level interface supports. Expose them as
declarations rather than switches: `noise=Jitter(prior)` and `noise=AR1(prior)`, and
`orbit=Hierarchical(inner=Orbit, outer=Orbit)`, each printed by `explain()` as a scientific
claim. The pipeline currently reports "the noise model does not describe the data" and offers
only `expert()` as the remedy. Add the zero-eccentricity guard the roadmap lists while the
initialisation API can still be changed.

### 5.8 Window-parallel solving (conditional on a dataset)

The solver is one sequential scan over one contiguous grid. Disjoint windows are not
block-diagonal because the smoothness prior couples across the gap, and the gap costs the same
per pixel as data (D §4). The multi-segment grid and a `vmap` over windows are the unbuilt
option `docs/math.md` §4.2 names. Build it only when a multi-window real dataset (Maxted's
per-order HARPS, or a full HERMES range) shows that the design-target gradient of 46 s is the
bottleneck for a user. A persistent compilation cache would save about 4%, which does not
justify a setting.

## 6. Tier C: reach

### 6.1 SBX (days)

A TAP client for the SB9 successor provides a target list of 1,326 SB2s with public epoch
spectra to cross-match, 224,148 labelled velocity measurements to warm-start
`Disentangler(velocities=)` or to validate `todcor` against, and published K2 values to score
orbits. `archive.py` already supports this TAP dialect.

### 6.2 LAMOST MRS (one to two weeks)

Read single exposures, never coadds. Declare the applied heliocentric shift and do not
re-apply it. Compute BJD_TDB from the header, declare vacuum, and key the line-spread function
on the scalar R = 7500 until a per-observation value exists. Validate on Guo et al. 2025's 665
orbits and reject the 126 moonlight false positives with a contamination guard (a component at
zero velocity with a solar-like spectrum). This is the trial run for Gaia, with the same three
decisions (an applied correction not to re-apply, native versus resampled, a missing BJD) and
the largest SB2 population available today. It enables 4.4.

### 6.3 HERMES (days)

This is the cheapest loader available, its users are albireo's hot-star community, and the
ESO reader's dispatch already fits it. Record the 1.56 km/s resampling as a measured AR(1)
coefficient per epoch rather than as a note.

### 6.4 Gaia DR4: build the half that can be built now

Before 2 December, build a `gaia_rvs` module in three parts. (i) A GOST client returns
barycentric transit times for a position and passes them to `plan_epochs` and
`sensitivity_forecast` with the 961-pixel vacuum grid and an R of about 11,500. The expected
uncertainty band for any Gaia SB2 can then be forecast today from `nss_two_body_orbit`'s
elements and `g_luminosity_ratio`. (ii) A DataLink client is exercised on DR3's endpoint and
switched to the DR4 retrieval type by one constant. (iii) A reader written against the draft
data model branches on `normalisation_method`, requires class-3 `has_epoch_rvs`, reads the
`TIMESYS` element for the time origin, and rejects DR3 mean spectra. On release day, run the
go/no-go query joining `rv_assumed_sb2` with the BLOeM DR3 identifiers `resolve_bloem` already
returns, produce the first disentangled DR4 SB2s with bands, and submit a Research Note within
the week. The Ca II triplet is broader than the differential shift for most systems, so the
per-pixel prior profile must represent it while the covariance reports that its separation is
prior-dominated (C §7). Chromospheric emission in the triplet cores is a zero-velocity
variable component, which needs 5.5's windowed jitter on day one.

### 6.5 Later

The later loaders are CFHT ESPaDOnS (best-formatted product found), APOGEE DR17 `apVisit`
(7,273 SB2s; document the dither correlation as a measured AR(1) coefficient), and last SOPHIE
and ELODIE (no errors, truncated dates, two velocity frames). Keck HIRES is excluded.

### 6.6 Two rules for every loader

First, a loader dispatches on the velocity frame, not on the archive: a product is rejected
(rest-frame), declared already corrected, or corrected in the model, and SOPHIE's two products
fall under different cases. Second, every upstream resampling is reported as a measured
lag-one autocorrelation, because six of the nine archives surveyed resample and four provide
no errors. This makes D34's AR(1) model a requirement for five loaders rather than an option.

## 7. Tier D: community and papers

### 7.1 Announce, in order

Send the `v0.1.0` release note to the Massive Star Newsletter (monthly, active), the
OpenAstronomy forum, and the astronomy communities on the social networks. A Research Note is
for a dated result, not a software announcement. The talks are the Gaia meeting in Athens
(11 to 15 January 2027, five weeks after DR4) with the first DR4 SB2 bands, EAS 2027 (Vienna,
June), and the IAU General Assembly in Rome (August 2027).

### 7.2 Write to the people whose problems the package already solves

- The BLOeM team (Villaseñor, Shenar, Sana): a worked example that takes one of their SB2s
  from MINATO's `ravel` velocities through `Disentangler(velocities=)` to TODCOR templates,
  and an offer to run the 59 published SB2s with bands. They use the same stack (JAX, NumPyro)
  and licence class, and have published a statement of need.
- Maxted and Southworth: the AI Phe reproduction of 4.3, and the spectroscopic light ratio
  Southworth's series calls "unexplained", which 5.1 and 5.4 address directly.
- El-Badry: the four-code comparison on ALS 8814 of 4.4.
- Moharana, Hełminiak and Southworth: a window-sensitivity panel from `sensitivity_forecast`
  for V446 Cep, answering "it is hard to trace the source of this inconsistency" with the
  serialisable TOML record of every choice.

### 7.3 Papers

1. JOSS, no earlier than 2027-02-14, with the evidence clause met by then (Tier 0 item 5) and
   `paper.md` rewritten to the current template with the mandatory AI-usage disclosure.
2. A methods paper in A&A or MNRAS: AI Phe with a posterior on the orbit and the spectra, the
   K1-K2 contour and its marginalised counterpart, the equivalent-width correlation of -0.992
   between components, and the nebular result that contamination affects the masses. The
   framing is agreement plus posteriors, never "the old code is wrong".
3. A Research Note on the first Gaia DR4 SB2 epoch spectra disentangled with uncertainties,
   within days of release.
4. The BLOeM SB2 sample with the survey team, if 7.2 succeeds.
5. The profile mode (5.6) as its own short methods paper.

### 7.4 Registrations

Apply for pyOpenSci review once the package is on PyPI with three to six months of public
history. The review now confers both the Astropy affiliated-package status (APE 22 delegated
the review) and a JOSS fast track without a second review. Frame albireo as a new
implementation of an established method, since pyOpenSci excludes "novel or unvetted
analytical approaches". Register with ASCL only after the first refereed paper that used the
code.

### 7.5 Teach

PHOEBE has 105 stars and 263 Discussions threads, more than every other code surveyed
combined, owing to nine annual workshops. Hold one half-day online tutorial a year, built from
the executed notebooks, with the two entry points the field's two communities need. One is
the eclipsing-binary path (a light curve in hand, a light ratio to declare, in-eclipse spectra
to keep). The other is the compact-companion path (no light curve, few epochs, a calibrated
non-detection and a goodness-of-fit that shows when two components are not enough).

### 7.6 State at the top of the README what only albireo provides

The README's capability list does not connect to any problem an astronomer recognises
(A §7.1). Three sentences belong at the top. The uncertainty band is the quantity
Serebriakova et al. describe as not well-defined. The nebular component is the feature UNWIND
lists as future work. ML-II on the prior scale is the automation Seeburger et al. name as
future work.

## 8. Non-goals, reaffirmed and added

Unchanged from the roadmap: no synthesis, no abundances, no GUI, no molecfit wrapper, no
Fourier methods. Added by this page: no metre-per-second circumbinary-planet velocities (a
separate engineering programme with its own community and code); no Gaia DR3 mean spectra
(rest-frame, rejected); no Keck HIRES loader; no SDSS-V `mwmVisit`; no fitting of a
wavelength-dependent light fraction under constant light (unidentified at one part in ten
thousand); no light fraction sampled inside the model under constant light (returns the prior);
no emulator for the label grids until `crossval_library` on an OB grid shows that cubic
interpolation is less accurate; no forward-mode path through the marginal (D49's decision
stands).

## 9. Calendar

| when | what |
|---|---|
| September 2026 | Tier 0 complete: site live, name on PyPI, `v0.1.0` tagged, metadata filled, issues opened; EAS proposal by the 30th; AI Phe tests 1 to 5; SBX client |
| October | LAMOST loader and the ALS 8814 four-code comparison; per-epoch line-spread function; 5.1, 5.2, 5.3; the K1-K2 map; AI Phe NUTS |
| November | HERMES loader; 5.4; 5.7; the Gaia forecast and DataLink half; the Maxted 2026 reproduction; letters to BLOeM, Maxted and Southworth, El-Badry |
| 2 December | Gaia DR4 reader, go/no-go query, first DR4 bands, Research Note |
| January 2027 | Athens Gaia meeting; methods paper drafted; 5.5 jitter on HR 6819 or ALS 8814 |
| February 2027 | JOSS submission if releases, issues and pull requests exist; pyOpenSci submission |
| 2027 | 5.5 basis, 5.6 profile mode, first tutorial workshop, EAS (June), IAU GA (August) |

## 10. Corrections to existing documents found by the research

These were made on 2026-09-03, with three open items listed at the end.

- `docs/science.md` §9 and its reference list attribute A&A 698, A40 (the BLOeM B-supergiant
  paper) to Bodensteiner et al. It is Britavskiy, Mahy, Lennon et al. 2025 (arXiv:2502.12239,
  *verified*); Bodensteiner et al. 2025 is A&A 698, A38, on the Oe and Be stars.
- `src/albireo/kepler.py` describes `nu + omega = pi/2` as the time at which the component
  passes in front of its companion. It is superior conjunction: the component is the one
  eclipsed. The code is right and the docstring is inverted (E, confirmed from the AI Phe
  spectra).
- `scripts/aiphe_bench.py` uses the published period rounded to `24.5924`, which adds up to
  0.1 km/s of error to the fixed-velocity label run, and starts its second window 0.6 Å inside
  the HARPS CCD gap.
- `docs/quickstart.md` line 7 (`pip install albireo`) and README lines 100 and 114 disagree
  about installation.
- `internal/releasing.md` states that nothing has ever been pushed, but the remote is in sync
  and step 2's ordering advice is stale.
- `internal/roadmap.md` item 7's DR4 table needs the four corrections in section 2.6.
- The `pipeline`, `todcor`, `library`, `match`, `forecast`, `archive` and `handoff` modules
  have no stability marker, and only `facade` is marked experimental. The 0.1.0 policy
  statement should state which interfaces are covered.
- The "Windows 11 laptop" tables in `docs/benchmarks.md` and the desktop tables may be from
  the same machine (both 32 GB, both Windows 11 build 26200). Nothing in the repository
  settles it, and the 1.46x "clean-machine gap" argument rests on the distinction (D §10.5).
  This is now settled. Read from the machine itself, the development machine is an AMD Ryzen 9
  9950X3D desktop, 16 cores / 32 threads, 32 GB, Windows 11 Pro build 26200. There was one
  machine, the "laptop" label is wrong, and both arguments that rested on the distinction are
  withdrawn in `docs/benchmarks.md`.

Three items remain open:

- The AI Phe label figures in `docs/benchmarks.md` were produced with the rounded period and
  should be re-run with 24.592483 d. This is noted in place; the run itself needs the archival
  spectra.
- `internal/design.md`'s D50 row still describes the earlier tables as a different machine.
  The decision record keeps a decision as it was taken, so the correction went into
  `docs/benchmarks.md` instead. Amending the row is the maintainer's decision.
- Every install command in the documentation is now the editable clone form, which is correct
  only until the package is on PyPI. `internal/releasing.md` step 4 contains the item that
  changes them back.

## 11. Decisions only the maintainer can make

1. Affiliation and ORCID for the citation files and the paper.
2. Whether to claim `albireo` on PyPI now with `0.1.0` or with a `0.1.0.dev1` placeholder
   (claim now; the version is secondary).
3. Whether to write to the BLOeM team before or after the LAMOST and HERMES work. The letter
   needs only the existing `resolve_bloem` path and one worked example.
4. Whether to propose the EAS 2027 session, and with whom, before 30 September.
5. Whether the methods paper goes to A&A or MNRAS, which sets its length and the figure budget
   for the K1-K2 comparison.
6. Whether the research reports beside this page are kept in the repository or in the
   maintainer's notes. They are agent-written, partly unverified where marked, and about 500 kB.
