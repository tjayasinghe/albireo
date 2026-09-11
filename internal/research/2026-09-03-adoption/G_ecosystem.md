# G_ecosystem — packaging, registration and community mechanics for `albireo`

Research date: **2026-09-03**. All web facts were checked on this date unless a page date is given.
Subject: `albireo`, Bayesian spectral disentangling of spectroscopic binaries in JAX.
GitHub <https://github.com/tjayasinghe/albireo>, docs <https://tjayasinghe.github.io/albireo/>, BSD-3-Clause,
pre-alpha, public since **2026-08-14**, not yet on PyPI.
Dependencies: numpy, scipy, jax>=0.5, numpyro, optax; optional astropy, matplotlib, arviz.

Conventions used throughout:
- **[V]** = verified against a primary source, URL and date given.
- **[I]** = inference / my judgement, explicitly not a verified fact.
- Where a search could not confirm something, it says so rather than guessing.

Two constraints on this research run, stated up front so the gaps are not mistaken for findings:
the session's WebSearch budget (200 calls) was exhausted partway through, and several
sub-researchers could not be launched against a concurrency cap. Items marked
**[NOT VERIFIED]** are the residue. They are listed in §8 as explicit open questions.

---

## 0. What actually matters

Ordered by deadline pressure, not by section number.

| # | Finding | Date pressure |
|---|---|---|
| 0 | **The documentation site does not exist.** <https://tjayasinghe.github.io/albireo/> returns 404; GitHub Pages was never enabled and the `PAGES_ENABLED` deploy gate in `docs.yml` was never set. The README's green Docs badge links to that 404. Nothing else in this report matters until this is fixed (§4A.1a). | **Now — one command** |
| 1 | **`albireo` is free on PyPI.** So are all three fallbacks. PyPI has no reservation mechanism; it is first-come-first-served — and Czekala's Starfish really did lose `starfish` to a biology package (§5.1). | **Claim it today** |
| 2 | **Zero tags, zero releases, zero issues, zero PRs.** Across 15 surveyed packages, first-year *release count* — not commit count — separated the adopted from the unadopted. Starfish put 383 commits into year one, cut 0 releases, and now does 103 downloads/month; emcee shipped on 51 commits and does 1.2M (§5.5, Finding 2). | **Tag v0.1.0 this week** |
| 3 | **EAS 2027 call for special sessions closes 30 Sep 2026** (Vienna, 21–25 June 2027). | **27 days** |
| 4 | **Gaia DR4 lands 2 December 2026** with 787,312 double-lined RVS transits, 638,966 spectroscopic-binary solutions and 6.9 billion RVS epoch spectra. | 90 days |
| 5 | **Astropy affiliated status is now a pyOpenSci review** (APE 22), and pyOpenSci fast-tracks to JOSS. One review, three outcomes. But pyOpenSci requires a PyPI/conda release and 3–6 months of public history. | Gate opens ~Nov 2026 |
| 6 | **JOSS requires >6 months of public history "with evidence of releases, public issues and pull requests".** Public 2026-08-14 → earliest submission **2027-02-14**. The calendar passes automatically; the evidence clause currently fails on all three counts (§2.2). | Ongoing |
| 7 | **`paper/paper.md` is missing 4 of JOSS's 6 required 2026 sections** — no State of the field, Software design, Research impact, or AI usage disclosure; affiliation reads `name: TODO` (§2.8). | Before submission |

**The strategic finding, if you read only one thing.** The niche is empty and nearly frictionless
to win on engineering — the incumbent is a 2014 C binary served over plain HTTP with no licence
(fd3), and a Fortran engine from 2011 behind a registration wall (KOREL). But GitHub's search for
"spectral disentangling" returns **19 repositories worldwide**, of which the astronomy ones hold
under 20 stars combined. There is no ambient traffic to capture and no community to inherit.
Across every case in §5, adoption came from exactly one of three places: **an institutional home**
(lightkurve, which has no paper at all, is NASA GO Office software), **being someone else's
dependency** (dynesty and emcee inside `bilby`), or **sustained teaching** (PHOEBE 2's nine
consecutive annual workshops, 470 notebooks, 263 GitHub Discussions on 105 stars). Packaging and
documentation are necessary and are not sufficient. Gaia DR4 on 2 December 2026 is the single best
shot at the first of those three.

---

## 1. PyPI and conda-forge

### 1.1 The name `albireo` is FREE [V]

Checked 2026-09-03, three independent endpoints:

| URL | Result |
|---|---|
| <https://pypi.org/simple/albireo/> | **HTTP 404 Not Found** |
| <https://pypi.org/pypi/albireo/json> | **HTTP 404 Not Found** |
| <https://pypi.org/project/albireo/> | page did not render content (client-side error) — inconclusive on its own |

The `project/` page was inconclusive, so the 404s carry the finding. To rule out a systematic
fetch failure I ran a positive control on the same endpoint shape:

| Control URL | Result |
|---|---|
| <https://pypi.org/simple/numpyro/> | **200**, ~80 file links, `numpyro-0.1.0.tar.gz` … `0.21.0` |

The control returns files; `albireo` returns 404. **The name is unregistered.** [V]

### 1.2 Fallback names — all three are also free [V]

Checked 2026-09-03 against `https://pypi.org/simple/<name>/`:

| Candidate | Status |
|---|---|
| `albireo-astro` | **404 — free** |
| `pyalbireo` | **404 — free** |
| `albireo-disentangle` | **404 — free** |

Note on normalisation: PEP 503 normalises a project name by lowercasing and collapsing runs of
`-`, `_` and `.` to a single `-`. So `albireo` being free also means `Albireo`, `ALBIREO` and
`al.bireo` are free — they are the same name. `albireo-astro` normalises to `albireo-astro`,
which is a *different* name from `albireo` and does not protect it. [V, PEP 503]

### 1.3 PEP 541 is therefore moot — but the finding inverts the question

Since the name is free, the reclamation route is not needed. Recorded for completeness, from
<https://peps.python.org/pep-0541/> (fetched 2026-09-03):

- A project counts as **abandoned** when the owner is unreachable, there have been **no releases
  in 12 months**, and the project homepage shows no activity. [V]
- PyPI admins attempt contact **at least three times over six weeks** before declaring an owner
  unreachable. [V]
- Requests are filed as issues on <https://github.com/pypa/pypi-support> using the PEP 541 template. [V]
- PyPI **explicitly declines to arbitrate between two active projects**, even if one is more
  popular or "better". [V]

**[I]** Realistic timeline if it had been needed: three to twelve months, with a substantial
probability of refusal. Not a path to plan around. The relevant lesson is the opposite one —

**[I] Recommendation: register `albireo` on PyPI now, before anything else in this report.**
Upload a real 0.1.0 if it is ready, or a minimal placeholder that imports and reports a version
if it is not. The cost is minutes; the cost of losing the name to an unrelated project is a
rename across docs, papers, DOI metadata and every citation already in the wild. There is no
"reserve a name" feature on PyPI — publishing is the only claim. Note also that pyOpenSci
review (§3) *requires* a PyPI or conda release, so this is on the critical path regardless.

**[I] One non-blocking flag:** "Albireo" (β Cygni) is a famously beautiful visual double, which
makes it an excellent name for a binary-star package. It is also the name of a former Nasdaq
pharmaceutical company (Albireo Pharma, acquired 2023). Trademark exposure for an academic
BSD-3 package in a different field is negligible, but the name is not distinctive enough to be
searchable — "albireo python" will compete with astrophotography content. Consider whether the
PyPI *description* and GitHub topics carry the discriminating keywords ("spectral disentangling",
"spectroscopic binary", "SB2") since the name itself will not.

### 1.4 conda-forge: the JAX stack, and the Windows hole [V]

Checked on anaconda.org, 2026-09-03:

| Package | Version | Last updated | Platforms |
|---|---|---|---|
| `jax` | 0.10.2 | 2026-06-18 | **noarch** (all platforms) |
| `jaxlib` | 0.10.2 | 2026-07-03 | linux-64, linux-aarch64, osx-64, osx-arm64 — **no win-64** |
| `numpyro` | 0.21.0 | 2026-05-02 | **noarch** |
| `optax` | 0.2.8 | 2026-03-21 | **noarch** |

Sources: <https://anaconda.org/conda-forge/jax>, <https://anaconda.org/conda-forge/jaxlib>,
<https://anaconda.org/conda-forge/numpyro>, <https://anaconda.org/conda-forge/optax>.

`jax` is noarch, but it is a pure-Python front end — the compiled work is in `jaxlib`, and
**`jaxlib` has no win-64 build on conda-forge**. A Windows support request has been open on the
feedstock since December 2022 (<https://github.com/conda-forge/jaxlib-feedstock/issues/161>). [V]
conda-forge added *experimental* win-arm64 support in February 2026
(<https://conda-forge.org/blog/2026/02/09/win-arm64/>), which is a different architecture and
does not close this gap. [V]

**By contrast, PyPI does serve native Windows.** JAX's own installation page
(<https://docs.jax.dev/en/latest/installation.html>, fetched 2026-09-03) lists **Windows x86_64:
CPU (experimental)** among supported platforms with pip wheels, and separately notes that
CUDA pip installs "do not work with Windows, and may fail silently". It also describes the conda
route as **community-supported**, not maintained by the JAX team. [V]

**Consequence for albireo [I]:** pip/PyPI is the primary and only universal channel. A
conda-forge feedstock would be Linux + macOS in practice. Given the development box is Windows
(and a meaningful fraction of the target astronomer audience is on Windows), the documented
install path must be `pip install albireo`, and any conda instructions must carry the caveat.
Do not treat conda-forge as a prerequisite for launch; treat it as a later convenience for
Linux/macOS cluster users.

A nuance worth stating precisely, because it is easy to get backwards: **noarch is a property of
the build, not of dependency availability.** albireo can legitimately be built `noarch: python`
and still be unsolvable on win-64 simply because `jaxlib` is absent there. The recipe does not
need a Windows skip; the solver produces the failure on its own. The cost is a confusing error
message for a Windows conda user, which argues for documenting pip prominently.

### 1.5 What a `noarch: python` feedstock requires [V]

From <https://conda-forge.org/docs/how-to/basics/noarch/> (fetched 2026-09-03). One package is
built once, on Linux with the oldest supported Python, instead of N_python × N_platform builds.
All of the following must hold:

- "The package contents must be invariant across Python versions and platforms."
- No compiled extensions (limited `abi3` exceptions).
- "No post-link or pre-link or pre-unlink scripts may be used."
- "The recipe must not use selectors."
- "scripts argument in setup.py must not be used."
- **"console_scripts entry points … must be listed in the build section of the recipe."**
- Dependencies must be identical across Python versions.

Minimum Python declaration in current recipes:
- host and test: `- python {{ python_min }}.*`
- run: `- python >={{ python_min }}`
- override at recipe top with `{% set python_min = "3.12" %}` or via `conda_build_config.yaml`,
  then re-render.

**[I] Relevant to albireo specifically:** the package ships an `albireo` CLI (init/run/demo/fetch).
Under noarch that console script **must be declared explicitly in the recipe's build section** —
it is not inferred. This is the single most common noarch mistake for packages with a CLI.
Everything else on the list is satisfied by a pure-Python JAX package.

### 1.6 The conda-forge submission process [V]

From <https://conda-forge.org/docs/maintainer/adding_pkgs/> (fetched 2026-09-03):

1. Open a PR against <https://github.com/conda-forge/staged-recipes> with the recipe and the license file.
2. Build from a **release tarball with a SHA256**, not from a git repo — so a PyPI sdist must exist first.
3. On merge, CI automatically creates a dedicated feedstock repository and you become its maintainer.
4. Two recipe formats are supported: **v0 `meta.yaml`** (conda-build) and **v1 `recipe.yaml`**
   (CEP 13/14, rattler-build). Both are current.
5. For a Python package: `pip` in host, install via `{{ PYTHON }} -m pip install . -vv`, minimal
   test is an import check plus `pip check`.
6. Hard rule: "all dependencies have to be packaged by conda-forge as well."

No official turnaround is published. **[I]** Community experience is days to a few weeks depending
on reviewer availability; it is a queue, not an SLA. Feedstock creation failing for >1 day is
documented as abnormal and worth reporting.

---

## 2. JOSS in 2026

Sources: <https://joss.readthedocs.io/en/latest/submitting.html> and
<https://joss.readthedocs.io/en/latest/review_criteria.html>, both fetched 2026-09-03.

### 2.1 The six-month rule — verbatim, and it binds [V]

> "The repository must have been public for more than six months prior to submission, with
> active development spanning that period."

and

> "Projects developed privately are not eligible until there is a public record of open
> development: at least six months of public history prior to submission, with evidence of
> releases, public issues and pull requests."

albireo went public **2026-08-14** → **earliest JOSS submission 2027-02-14**. This confirms the
figure already in the project's notes.

### 2.2 The clause that is the real risk, not the calendar [V]

> "The development history must show ongoing iteration, not a single burst of commits."

JOSS grades development timelines explicitly:
- **Good**: commits distributed over 6+ months showing gradual development.
- **OK**: 6+ months with sporadic activity.
- **Not acceptable**: most commits concentrated in the weeks before submission.

**[V] Measured against albireo's actual public record** (read-only inspection, 2026-09-03):

| Signal JOSS asks for | Current state |
|---|---|
| Public commit history | **72 commits**, all since 2026-08-11 ("Initial commit" — the private history is *not* in the public repo) |
| Distribution of those commits | **46 of 72 (64%) fall in the first six days** (8/14/8/12/3/1 on Aug 11–16), then 5 on Aug 27, 7 on Sep 1, 1 on Sep 2, 3 on Sep 3 |
| "evidence of **releases**" | **0 releases, 0 git tags** |
| "evidence of **public issues**" | **0 issues** (issues are enabled) |
| "evidence of **pull requests**" | **0 pull requests** |

**[I] This is the sharpest finding in §2, and it is worse than the calendar suggests.** The
six-month clock is the easy part — it passes automatically on 2027-02-14. The clause that
actually gates eligibility is "with evidence of releases, public issues and pull requests"
(§2.1), and albireo currently has **zero of all three**. Meanwhile the commit shape — a 46-commit
opening burst followed by sparse activity — is the profile JOSS grades "not acceptable" when it
sits next to a submission date.

The mitigation is not retroactive; it is what happens between now and February. Concretely, and
in rough order of value per unit effort:
- **Tag releases.** `v0.1.0` now (which also mints the Zenodo DOI, §3.6), then a small cadence.
  Zero tags after six months is the single most conspicuous gap, and it is also the one JOSS's
  single-author clause names first (§2.3).
- **Work through issues rather than around them.** Issues opened and closed by the sole author
  still constitute a public development record, and they are how a reader reconstructs *why* a
  change happened. A repository with zero issues after six months reads as unused, whatever the
  commit count says.
- **Use pull requests for non-trivial changes**, even self-merged. This costs almost nothing on a
  solo project and converts invisible work into the exact artefact being asked for.
- **Keep the commit cadence non-zero.** Sporadic-but-continuous grades "OK"; a second burst
  immediately before submission grades "not acceptable".

### 2.3 The single-author clause [V]

> "For *multiple* indicators … at submission time: a meaningful public commit history over time,
> tagged releases or a changelog, tests and CI, clear documentation, a CONTRIBUTING file, and
> stated support or governance expectations."

**[I]** albireo is single-author. This is not disqualifying — many JOSS papers are single-author —
but it shifts the burden onto the artefacts in that list. A `CONTRIBUTING.md` and a stated
support expectation are cheap and are explicitly enumerated; their absence is conspicuous.

### 2.4 "Substantial scholarly effort" [V]

JOSS gives **no numeric threshold** — no LOC count, no month count beyond the six-month history
rule. It is assessed qualitatively against:
- **Research impact**: "Evidence of publications or analyses using the software, external
  adopters or integrations."
- **Design thinking**: "Meaningful architectural decisions and trade-offs considered."
- **Open development practices**: sustained development, public history, testing, documentation.
- Whether the software is "sufficiently useful that it is *likely to be cited*".
- Explicitly *not* "a one-off tool for a single analysis".

(For contrast, pyOpenSci *does* give a number — see §3.2.)

### 2.5 AI usage disclosure is now mandatory [V] — new, and directly relevant

> "The use of generative AI is permitted for most aspects of a JOSS submission (e.g., software
> creation and review, generating documentation, assisting with paper authoring), however all
> such use must be disclosed in an 'AI usage disclosure' statement which includes:
> * **Tool use:** The tools/models used (and versions) and where they were used (code, paper text, docs).
> * **The nature and scope of assistance:** e.g., code generation, refactoring, test scaffolding, copy-editing, drafting.
> * **Confirmation of review:** Authors must assert that human authors reviewed, edited, validated all AI-assisted outputs and made the core design decisions."

And from the submission guidance:

> "Whether or not you use AI assistance in your development, submissions should demonstrate the
> irreplaceable human contributions: problem framing, key design decisions, thoughtful
> architectural choices, and practices that make software usable and sustainable for others."

**[I]** Permitted, but disclosed, and the review then looks for human design judgement. For
albireo the defensible material exists and should be written down while it is fresh: the
adversarial review of the façade that found 38 defects all in the translation layer; the decision
to fit dilution jointly through a shared radius ratio; the `compare="matched"` → `native` default
change forced by AI Phe real data; the custom VJP that made gradients 1.8× faster with
bit-identical answers. Those are design decisions with recorded reasoning and measured outcomes,
which is precisely what the criterion asks for. A JOSS paper that cannot point to any is weak
regardless of AI involvement.

### 2.6 Pre-alpha / 0.x status [V, partly negative evidence]

Neither the submission page nor the review-criteria page states a minimum version number, and
neither requires 1.0. **[V by absence]** — I read both pages looking for such a rule and found
none. What JOSS *does* require is that the software be "feature-complete (i.e., no half-baked
solutions), packaged appropriately according to common community standards" and "designed for
maintainable extension".

**[I]** So 0.x is fine; *pre-alpha* is a labelling choice that works against you. "Pre-alpha"
signals "half-baked" to a reviewer reading the same page that forbids half-baked solutions. By
submission time the project should be describing itself as beta or stable-for-the-documented-API,
with an explicit stability statement (§7.3) rather than a blanket disclaimer.

### 2.7 Companion methods paper — allowed, must be declared [V]

> "Sometimes authors prepare a JOSS publication alongside a contribution describing a science
> application, details of algorithm development, and/or methods assessment. In this circumstance,
> JOSS considers submissions for which the implementation of the software itself reflects a
> substantial scientific effort. This may be represented by the design of the software, the
> implementation of the algorithms, creation of tutorials, or any other aspect of the software.
> We ask that authors indicate whether related publications (published, in review, or nearing
> submission) exist as part of submitting to JOSS."

Also: **"Submission to a preprint server is *not* considered a previous publication."** [V]

**[I] Reading for albireo:** an A&A/MNRAS methods paper (the algorithm, the HR 6819 campaign, the
AI Phe validation) plus a JOSS paper (the software) is an explicitly supported combination, not
dual publication. The division of labour that satisfies JOSS: the methods paper carries the
science and the algorithm derivation; the JOSS paper carries the *implementation* — API design,
the JAX/gradient architecture, the tutorials, the packaging. The JOSS paper must not be a
condensed version of the methods paper. Declare the companion at submission.

### 2.8 Repository requirements checklist [V]

- OSI-approved license, in the repository (BSD-3 qualifies).
- Documentation and tests; automated dependency-managed install instructions.
- Usage examples solving real problems; API documentation for core functionality.
- Community contribution and support guidelines.
- Hosted where outsiders can freely open issues and propose changes.

**The paper itself** [V], <https://joss.readthedocs.io/en/latest/paper.html> (fetched 2026-09-03):

- Files: **`paper.md`** (Markdown + YAML front matter) and **`paper.bib`**.
- YAML fields: `title`, `authors` (each with **`orcid`** and `affiliation`), `affiliations`
  (index, name, optional ROR), `tags`, `date` (`%e %B %Y`), `bibliography`.
- Length, verbatim: "The paper should be between **750-1750 words**. Authors submitting papers
  significantly longer than 1750 words may be asked to reduce the length of their paper."
- **Six required sections** (2026 set):
  1. **Summary** — "A description of the high-level functionality and purpose of the software for
     a diverse, *non-specialist audience*."
  2. **Statement of need** — research purpose, problem domain, target audience, relation to existing work.
  3. **State of the field** — comparison to competing packages, with justification for why the
     alternatives are insufficient.
  4. **Software design** — trade-offs and architectural choices, with rationale.
  5. **Research impact statement** — realized or credible near-term impact: publications, external
     use, or community signals.
  6. **AI usage disclosure** (§2.5).
- Figures need captions and "must appear alone in a paragraph to be numbered"; citations use
  pandoc/BibLaTeX `[@key]`; spell out venue names rather than discipline abbreviations.

**[V] The existing `paper/paper.md` is drafted against an older JOSS template.** Inspected
2026-09-03 (read-only): it is **1,263 words** — comfortably inside the 750–1750 band — with
top-level headings `Summary`, `Statement of need`, `Functionality`, `Acknowledgements`,
`References`. A case-insensitive search finds **no** "state of the field", "software design",
"research impact", "AI usage" or "generative AI" anywhere in the file. The YAML has `title`,
`tags`, `authors`, `affiliations`, `date: 11 August 2026`, `bibliography`, with **both `orcid` and
`affiliation` left as explicit `TODO`s** (`name: TODO`, index 1).

So against the 2026 requirement set: **2 of 6 required sections present**, `Functionality` is not
one of the six, and four are missing — `State of the field`, `Software design`,
`Research impact statement`, `AI usage disclosure`. **[I]** This is a straightforward rewrite
rather than new research (the material for three of the four exists in `docs/benchmarks.md` and
the project's design notes), but it is not a five-minute edit either, and it would have been
easy to discover only at submission. Budget it deliberately. Note also that adding four sections
to a paper already at 1,263 words means the existing `Functionality` section has to shrink to stay
under 1,750.

**[I] Three further observations.** First, the `orcid` YAML field is a stated requirement, so that
`TODO` is on the critical path, not cosmetic — and the same ORCID gap exists in `CITATION.cff`
(§4A.1). Second, **"state of the field"** is where
albireo is unusually well-placed and should not be modest: a measured three-way comparison against
fd3 and shift-and-add, on the same data, with accuracy replicating across machines, is exactly the
evidence this section asks for and most submissions cannot supply; the clean-room separation from
the incumbent repo is worth stating explicitly. Third, **"research impact" is the weakest section
today** — the honest content is near-term-credible rather than realized (no external adopters
yet), which is another argument for the Gaia DR4 timing in §6.1 and for an RNAAS or methods-paper
result (§6.5) landing before submission.

### 2.9 Review timeline — the only number I have is stale [V but dated]

The frequently cited figures — **mean 45.5 days, median 32 days** from submission to publication —
come from the JOSS design-and-first-year-review paper, <https://arxiv.org/pdf/1707.02264>, which
describes **2016–2017** data. **[NOT VERIFIED for 2026.]** I could not retrieve current JOSS
throughput statistics before the search budget ran out. Treat 1–3 months as an order of magnitude,
not a commitment, and note that JOSS editors have publicly described 2025–26 as a period of
"rapid change, navigating AI-fuelled submissions", which is unlikely to have made queues shorter.

### 2.10 Typical reasons astronomy submissions bounce

**[NOT VERIFIED]** — I could not find a dated, quantitative breakdown of JOSS rejection reasons
for astronomy specifically. What follows is **[I]**, inferred from the criteria above rather than
from data, and should be labelled as such if reused:

1. **Scope/effort**: a thin wrapper around an existing library, or a single-paper analysis script.
2. **The burst-commit profile** in §2.2 — a private project published all at once.
3. **Installation failure in the reviewer's environment** — the reviewer must install it. For a
   JAX package this is a live risk: a reviewer on Windows, or on a machine where the pinned
   jaxlib/CUDA combination does not resolve, will report a blocking failure. Pin loosely, test
   the documented install path on a clean machine, and say plainly which platforms are supported.
4. **Missing community files** (CONTRIBUTING, code of conduct, support statement).
5. **A paper that is a science paper**, not a software paper.

---

## 3. Astropy affiliated, pyOpenSci, ASCL, Zenodo, ADS

### 3.1 Astropy affiliated review is now delegated to pyOpenSci [V] — the structural finding

Per <https://www.astropy.org/affiliated/> (fetched 2026-09-03) and APE 22:

> "Astropy uses the pyOpenSci peer review process to vet affiliated packages" — "you submit your
> package directly to pyOpenSci."

The change followed the acceptance of **APE 22, "Astropy Affiliated Packages with pyOpenSci"**,
with the transition dated to **January 2024**. APE 22 is confirmed present in the APE list at
<https://github.com/astropy/astropy-APEs> (fetched 2026-09-03). The partnership page is
<https://www.pyopensci.org/software-peer-review/partners/astropy.html>. As of the 2026 sources
retrieved, **8 packages** have been accepted by pyOpenSci and thereby become Astropy affiliated. [V]

**Affiliated vs coordinated** [V]: *coordinated* packages are maintained by the Astropy Project
itself — the coordination committee holds administrative control of the repositories and
maintainers have formal project roles (astropy core, astroquery, specutils and similar).
*Affiliated* packages are community-owned, reviewed for ecosystem fit, and stay under their
authors' control. albireo would be seeking **affiliated**.

**Astropy-specific criteria layered on top of pyOpenSci's** [V]:
1. Useful to astronomers (a sub-domain is fine).
2. "Specifically use, interface with, or provide complementary capabilities to other Astropy packages."
3. Use astropy core classes and functions where appropriate, rather than duplicating them.
4. Open development, OSI license, citation instructions (a CITATION file is recommended).

**[I] Criterion 3 is the one with teeth for albireo**, and it connects directly to §4: a
disentangling package that invents its own spectrum container and its own time representation,
when `specutils.Spectrum` and `astropy.time.Time` exist, is duplicating core functionality. The
optional-astropy dependency is currently the right engineering call for a JAX-first package; the
affiliated review will want to see a *documented interoperability boundary* — a `to_specutils()` /
`from_specutils()` pair and BJD_TDB handling via `astropy.time` — rather than astropy being
merely importable. See §4 for the concrete shape.

### 3.2 pyOpenSci: scope, gates and timeline [V]

Sources: <https://www.pyopensci.org/software-peer-review/about/package-scope.html>,
<https://www.pyopensci.org/software-peer-review/how-to/author-guide.html>,
<https://www.pyopensci.org/software-peer-review/our-process/policies.html>, all fetched 2026-09-03.

**In scope.** Astronomy qualifies *through the Astropy partnership* rather than as a standalone
domain category. Data processing, scientific software wrappers, data visualisation and analysis
are all listed in-scope categories.

**Explicitly out of scope — two of these are gates for albireo** [V]:
- **"Packages not published to PyPI or community conda channels."** ← PyPI release is a hard prerequisite.
- Novel or unvetted analytical/statistical approaches **without prior journal acceptance**;
  proof-of-concept models lacking peer review.
- Also out: general-purpose services, out-of-sync forks, vendored dependencies, packages too
  complex to maintain.

**[I] The second gate deserves attention and is not obvious.** albireo implements a Bayesian
disentangling method. If the *method* is presented as new and unpublished, a pyOpenSci editor
could read it as an "unvetted analytical approach". The defence is that spectral disentangling is
long-established in the literature (Simon & Sturm 1994; Hadrava's KOREL; Ilijic's FDBinary/fd3)
and albireo is a new implementation of an accepted method class, validated against those
incumbents and against AI Phe. **That framing should be explicit in the submission**, and it is
another argument for the methods paper preceding or accompanying the review.

**Maturity requirement** [V]:
> "This package has a public development history spanning 3-6 months, with commits distributed
> over time that reflect iterative, thoughtful development rather than rapid and recent code
> generation."

→ public 2026-08-14 means the pyOpenSci window opens **~November 2026** (3 months) and is
comfortably satisfied by **February 2027** (6 months) — i.e. the same date as the JOSS gate.

**Effort threshold** [V], and unlike JOSS it is quantified in one place:
> "If the human effort put into the package is less than the effort required to review it, please
> don't submit the package."
A scope page also gives roughly **three months of development and generally ≥1,000 non-comment
lines** as the substantial-effort guide. albireo clears this by a wide margin. [I]

**Author checklist before submitting** [V]:
- Installable from **PyPI or conda**; imports work.
- User-facing docs with **quickstart tutorials**; complete API docs with docstrings.
- `README.md` with install instructions; `CONTRIBUTING.md` with dev setup; `CODE_OF_CONDUCT.md`;
  OSI `LICENSE`.
- Automated test suite with CI.
- **Disclosure of any generative AI tool usage** (same requirement as JOSS).
- Recommended: ruff, mypy, modern tooling (uv, Hatch).
- **A commitment to maintain for 1–2 years post-review.**

**Timeline** [V]:
| Stage | Stated target |
|---|---|
| Editor-in-Chief scope/infrastructure assessment | within **2 weeks** |
| Reviewer feedback | within **3 weeks** |
| Author response | within **2 weeks** of last review |
| Reviewers | typically **2** (1 on the fast-track pathway) |

Submissions on hold are revisited every 3 months; after a year with no movement the issue closes. [V]
**[I]** So a realistic end-to-end is roughly **2–4 months** if the author is responsive.

**On acceptance** [V]: Zenodo archival is required; a peer-review badge for the README; social
promotion and an invited blog spotlight; community Slack access; annual maintenance check-ins;
and the JOSS option.

### 3.3 The pyOpenSci → JOSS fast track is real and active in 2026 [V]

From <https://www.pyopensci.org/software-peer-review/partners/joss.html> (fetched 2026-09-03):

> "If your package is accepted by pyOpenSci and in scope for JOSS, JOSS will fast track your
> package through their review process given it was already reviewed by us."

> "After a package to pyOpenSci has been reviewed and accepted you can also choose to register it
> with JOSS."

> "Once accepted by JOSS, you now have both a pyOpenSci acceptance and one by JOSS. JOSS will give
> you a Crossref DOI for citation."

A caveat is stated plainly: **"JOSS has more limited scope for packages that it will review. For
instance, while pyOpenSci will review and accept API wrappers, JOSS won't."** [V] albireo is not
a wrapper, so this does not bite. The author guide describes the JOSS option as requiring
**"no second review"**. [V]

**[NOT VERIFIED]:** whether the JOSS fast track waives the six-month public-history rule. Nothing
I fetched addresses that interaction. **[I]** Assume it does **not** — the six-month rule is a
JOSS eligibility criterion, not a review-quality criterion, and pyOpenSci imposes its own 3–6
month rule anyway. Both clocks land in the same window, so it is academic here.

**[I] The resulting recommended route — one review, three outcomes:**

```
PyPI release (now)
   └─> 3-6 months visible public development (Sep 2026 - Feb 2027)
         └─> submit to pyOpenSci software-review  (Feb 2027)
               ├─> pyOpenSci accepted  ──> Astropy affiliated (APE 22 partnership)
               └─> opt in to JOSS fast track ──> JOSS paper + Crossref DOI
```

This is strictly better than submitting to JOSS directly: same clock, one review process, and it
picks up Astropy affiliated status which JOSS alone does not confer. The cost is meeting the
Astropy interoperability criterion (§3.1 item 3, §4).

### 3.4 ASCL — eligible only after a refereed paper [V]

<https://ascl.net/> (note: `ascl.net` presented a TLS chain that WebFetch could not verify on
2026-09-03, so the eligibility text below comes from the ASCL's own arXiv article
<https://arxiv.org/abs/2412.19941>, "Ten (or more!) reasons to register your software with the
Astrophysics Source Code Library", Dec 2024, and from the ADS/Wikipedia records — flagged
accordingly).

- **Eligibility**: codes that "have generated significant results for any paper published in a
  refereed astronomy or astrophysics journal"; more broadly, code "used in research that has
  appeared in, or been submitted to, peer-reviewed publications." [V, secondary source]
- Submission is via an online form; correspondence is electronic. ASCL does not claim copyright
  and will not archive without the copyright owner's permission.
- ASCL entries are **indexed by NASA ADS** (bibstem `ascl`, publication type **software**) and by
  Clarivate's Web of Science Data Citation Index; citations are tracked by ADS, Google Scholar and
  Web of Science. Each code gets a unique ASCL ID, so **software can be cited even when no paper
  describes it**. ASCL passed 2,700 entries. [V]
- ASCL IDs take the form `ascl:NNNN.NNN`; the ADS bibcode form is `YYYYascl.soft.....X`. [V, format]

**[I] Sequencing consequence:** ASCL is **not** a launch-day action for albireo, because the
refereed-paper link does not exist yet. It becomes available once the methods paper or the JOSS
paper is accepted. It is cheap then, and worth doing — it is the route by which a code becomes
citable in ADS *independently* of its paper, which matters for the "cite the software you used"
convention.

### 3.5 ADS software indexing and Zenodo [V, partly]

- ADS indexes ASCL records natively as software (§3.4). [V]
- The **Asclepias** project (AAS/Zenodo/ADS) exists specifically to bring software citations into
  the scholarly literature graph — see the AAS BAAS report "Asclepias: Software Citations Enter
  the Scholarly Literature World", <https://baas.aas.org/pub/2022i046/release/1>. [V, 2022]
- **[NOT VERIFIED]**: the precise current mechanics of whether and how ADS harvests arbitrary
  Zenodo software DOIs in 2026. The ADS data FAQ page fetched empty. Do not assert this.

**[I] Practical guidance for "cite albireo" producing an ADS-countable citation.** Rank order:
1. **A JOSS paper** — a Crossref DOI in a journal ADS indexes. This is the most reliably counted
   citation and the one to steer users toward.
2. **An ASCL entry** — an ADS software record, citable in its own right, accepted by all
   astronomy journals and by Science/Nature.
3. **A Zenodo concept DOI** — good for reproducibility and version pinning, but the weakest of the
   three for ADS-counted credit, and the least standardised in reference lists.
Point `CITATION.cff` and the docs at exactly one preferred citation (the paper, once it exists),
with the Zenodo DOI offered separately for version pinning. Users copy whatever the docs show.

### 3.6 Zenodo ↔ GitHub, CITATION.cff and codemeta.json [V for the precedence rule]

The precedence rule is the fact most often gotten wrong, so here it is verbatim from Zenodo's own
help, <https://help.zenodo.org/docs/github/describe-software/citation-file/> (fetched 2026-09-03):

> "If your repository also contains a `.zenodo.json` file, Zenodo will **only** use the
> `.zenodo.json` metadata and ignore the `CITATION.cff` entirely."

Further [V]:
- Zenodo **does** parse `CITATION.cff` from the repository root and makes "a best-effort attempt
  at parsing Zenodo-compatible metadata from it".
- Zenodo implements only a **subset** of the CITATION.cff schema: `cff-version`, `title`,
  `version`, `license`, `type`, `abstract`, `message`, `authors` (given-names, family-names,
  affiliation, **orcid**), `keywords`.
- `.zenodo.json` supports Zenodo-specific fields CITATION.cff lacks — notably **`grants`**
  (funding) and **`communities`**.
- Zenodo still recommends keeping a `CITATION.cff` because **GitHub** uses it to render the
  "Cite this repository" button.
- Zenodo's stated long-term preference is for open metadata formats like CITATION.cff to make
  `.zenodo.json` unnecessary.

**Enabling the integration** [V], <https://help.zenodo.org/docs/github/enable-repository/>
(fetched 2026-09-03). Prerequisite: a GitHub account connected in the Zenodo profile. Then:
1. "Click the profile menu in the header and click **GitHub**"
2. "Click **Sync now** in the header"
3. "Find a repository you wish to enable, then toggle the slider to connect it to Zenodo"
4. A confirmation appears; refresh to update the list.

Thereafter "new releases from the repository will be automatically ingested and archived." [V]
Note the ordering trap, which albireo's own `CITATION.cff` comment already records correctly:
**enable the integration BEFORE tagging**, or the tag is not archived and the DOI starts one
release late.

**[NOT VERIFIED]**: whether the link is implemented as a GitHub App or the legacy webhook in 2026
(the help page describes the UI, not the mechanism), and whether the repository must be public
(not stated). Also not verified from a primary source in this run: the concept-DOI/version-DOI
wording — <https://help.zenodo.org/docs/github/versioning/> returns 404. The distinction itself is
long-standing and uncontroversial: a *concept DOI* resolves to the latest version and is the one
to cite for "the software"; a *version DOI* pins one release.

**codemeta.json** [I]: a JSON-LD software-metadata standard consumed by aggregators and registries
rather than by GitHub or Zenodo's GitHub flow. It is optional. **[I] For albireo it is not worth
the maintenance burden now** — it is a third file to keep in sync with two that already have a
defined precedence, and nothing in the pyOpenSci/Astropy/JOSS path requires it.

**[I] Recommended concrete setup**, in dependency order:
1. `CITATION.cff` in the repo root — gets the GitHub "Cite this repository" button, is parsed by
   Zenodo, includes ORCID. **Do not add `.zenodo.json` unless you need `grants` or `communities`,**
   because adding it silently disables CITATION.cff parsing entirely.
2. Enable the Zenodo–GitHub integration on the public repo; cut a **tagged GitHub Release** to
   trigger the first archive and mint the concept DOI.
3. Put the **concept DOI** badge in the README and the docs landing page.
4. After the JOSS/methods paper exists, update `CITATION.cff` `preferred-citation` to the paper,
   keeping the DOI for version pinning.
5. ASCL entry once a refereed paper exists.

---

## 4. specutils 2.x and the astropy interop boundary

All URLs in this section accessed 2026-09-03.

### 4.1 specutils 2.x release history [V]

From `CHANGES.rst` on main and PyPI:

| Version | Date |
|---|---|
| 2.0.0rc1 | 2025-03-18 |
| 2.0.0rc2 | 2025-04-24 |
| **2.0.0** | **2025-06-12** |
| 2.1.0 | 2025-07-30 |
| 2.2.0 | 2025-10-08 (dropped Python 3.10; bumped minimum astropy to **7.0**) |
| 2.3.0 | 2026-02-03 |
| **2.4.0** (latest) | **2026-06-01** |

`Requires-Python: >=3.11`.

### 4.2 `Spectrum1D` → `Spectrum`, and the trap that follows [V]

Breaking changes in 2.0.0, verbatim from CHANGES.rst:
- "Spectrum1D renamed to Spectrum." (#1126)
- "Spectral axis can now be any axis, rather than being forced to be last" (#1033, #1226)
- "Spectrum now properly handles GWCS input for wcs attribute" (#1074)
- "Spectrum arithmetic now checks whether the spectral axes of the two operand Spectrum objects
  are equal, and fails if they are not" (#1211)

`Spectrum1D` still exists, in `specutils/spectra/spectrum.py`:

```python
@deprecated(since="2.0", alternative="Spectrum")
class Spectrum1D(Spectrum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
```

**No removal version is declared** — no `obsolete=` argument, nothing in the changelog. The
lifetime is officially open-ended. [V]

**The critical detail [V for the class definition, [I] for the consequence]:** the subclass
relationship *flipped* at 2.0. In 1.x, `Spectrum` was an alias for `Spectrum1D`, so
`isinstance(x, Spectrum1D)` was true for everything. In 2.x, `Spectrum1D` is a *subclass* of
`Spectrum`, so a plain modern `Spectrum` **fails** `isinstance(x, Spectrum1D)`. Third-party code
that type-checks against `Spectrum1D` silently rejects the object specutils 2.x hands its users.
(Reasoned from the verified class definition, not executed — worth a REPL confirmation.)

### 4.3 The constructor [V]

Actual signature from source — note that `uncertainty`, `mask` and `meta` are **not** explicit
parameters; they pass through `**kwargs` to the `NDCube`/`NDData` base:

```python
def __init__(self, flux=None, spectral_axis=None, spectral_axis_index=None,
             wcs=None, velocity_convention=None, rest_value=None,
             redshift=None, radial_velocity=None, bin_specification=None,
             move_spectral_axis=None, **kwargs):
```

- **`flux`** must be a `Quantity`; a bare ndarray raises `"Flux must be a Quantity object."`
- **`spectral_axis`** must be monotonic ("can be either ascending or descending, but must be
  monotonic in either case"); coerced internally to `SpectralAxis`.
- **`mask`**: "Array where values in the flux to be masked are those that `astype(bool)` converts
  to True" — **True = masked**. Also: "Specutils will recognize flux values input as NaN and set
  the mask to True for those values unless explicitly overridden."
- **`meta`**: free-form dict; loaders conventionally put the FITS header at `meta['header']`.
- **`bin_specification`**: `'edges'` or `'centers'`.
- **`move_spectral_axis`**: force the spectral axis to `'last'`/`-1` (the pre-2.0 behaviour) or
  `'first'`/`0`; "the relevant axes in the flux, mask, and uncertainty arrays are simply swapped".

### 4.4 Multi-dimensional flux and `spectral_axis_index` [V]

Verbatim:
> "Prior to `specutils` version 2.0, the flux array was always reordered such that the spectral
> axis corresponded to the last flux axis."

> "In cases where the flux axis corresponding to the spectral axis cannot be determined
> automatically (for example, if multiple flux axes have the same length as the spectral axis),
> the spectral axis must be specified with the `spectral_axis_index` argument."

From the test suite:
```python
spec = Spectrum(spectral_axis=np.linspace(100, 1000, 10) * u.nm,
                flux=np.ones((10, 10)) * u.Jy,
                spectral_axis_index=1)
```

**[I]** `spectral_axis_index` exists precisely because dropping the forced-last convention made
square/degenerate arrays ambiguous. For albireo this is a live case: a stacked
epochs × pixels array whose epoch count happens to equal the pixel count has no inferable answer.

### 4.5 `SpectrumCollection` — equal length confirmed, and it is the wrong container [V]

Class docstring:
> "A class to represent a heterogeneous set of spectra that are the same length but have
> different spectral axes."

Enforced in code:
```python
if not (flux.shape == spectral_axis.shape):
    raise ValueError("Shape of all data elements must be the same.")
```
and in `from_spectra`: `"Shape of all elements must be the same."`

The narrative docs add: "Currently, all SpectrumCollection items must be the same shape," and warn
it "makes no assumptions about the dispersion solution" — users must resample beforehand or accept
that the spectra do not share a dispersion solution.

It differs from a list of `Spectrum` in dimensionality: individual `flux.shape` `(50,)` becomes
`(2, 50)` on a collection of two. It is a vectorised container, not a sequence.

**Answer for a set of epochs of one star with differing lengths: no.** [V] Same length is a hard
constraint. Differing *grids* are fine; differing *lengths* are rejected. The sanctioned
alternative, verbatim from the `types_of_spectra` page:
> "this case should be dealt with by making lists (or numpy object-arrays) of Spectrum objects,
> and iterating over them."

`SpectrumList` is "a simple subclass of list that is integrated with the Astropy IO registry".

**[I]** Real multi-epoch spectroscopy is ragged — different nights, different orders, different
masks. `SpectrumCollection` is therefore not the interop target; a `list[Spectrum]` is.

### 4.6 `SpectralAxis`, `SpectralCoord` and the hidden-velocity hazard [V]

`SpectralAxis` **inherits from `astropy.coordinates.SpectralCoord`**:
> "Coordinate object representing spectral values corresponding to a specific spectrum. Overloads
> SpectralCoord with additional information (currently only bin edges)."

So it adds only bin-edge bookkeeping, plus a widened unit set
(`_equivalent_unit = SpectralCoord._equivalent_unit + (u.pixel,)`).

`SpectralCoord` adds over a plain `Quantity`: `observer`, `target`, `radial_velocity`, `redshift`
(the last two mutually exclusive), `doppler_rest`, `doppler_convention`; and the methods
`with_observer_stationary_relative_to()`, `with_radial_velocity_shift()`, `to_rest()`.

Astropy's own admonitions, verbatim [V]:
> "The SpectralCoord class at this time does not fully support cases where the observer and target
> are moving relativistically relative to each other, so care should be taken in those cases."

> "It is possible that there will be API changes in future versions of Astropy based on user feedback."

Pixel units block transformation entirely: `raise u.UnitsError("Cannot transform spectral
coordinates in pixel units")`, and `"u.pix spectral axes should always be ascending"`.

**[I] This is the most important design finding in §4 for albireo.** `SpectralCoord` carries a
velocity frame as *hidden state on the coordinate array*. If `observer`/`target` are populated,
frame conversions can be applied by methods that look like pure arithmetic. For a code whose
entire scientific content is a velocity shift measured at the ~100 m/s level, a second, invisible
velocity-bookkeeping system is a correctness risk, not a convenience. **Convert to plain
`Quantity`/ndarray at the boundary; never let a `SpectralCoord` frame ride into the solver.**

### 4.7 Uncertainties [V]

Accepted: `StdDevUncertainty` (`"std"`), `VarianceUncertainty` (`"var"`), `InverseVariance`
(`"ivar"`). Omitting a class gives `UnknownUncertainty`, and specutils states: "Not defining an
uncertainty class will result in an UnknownUncertainty object which will not propagate
uncertainties in arithmetic operations." A bare numpy array assigned to `.uncertainty` "is assumed
to be the `StdDevUncertainty`".

Conversion is `NDUncertainty.represent_as(...)`, which routes everything through variance
(`_convert_to_variance` → `_convert_from_variance`), carrying units: std→var squares array and
unit; ivar→var takes `1/array` and `1/unit`. `UnknownUncertainty` defines neither hook, so
`represent_as` on it raises `TypeError`. [V]

**[I]** `InverseVariance` is the natural internal form for albireo — it is what a Gaussian
likelihood wants, and it represents zero-weight pixels as `0` rather than `inf`. Accept any of the
three at the boundary, normalise with `represent_as(InverseVariance)`, and flag
`UnknownUncertainty` explicitly rather than silently treating it as std-dev.

### 4.8 Units, and the normalized-spectrum question [V/I]

`flux` must be a `Quantity`. `spectral_axis` is restricted to `SpectralCoord._equivalent_unit`
(wavelength / frequency / energy / wavenumber via `u.spectral()`) plus `u.pixel`. Conversion
helpers: `with_flux_unit(...)`, `with_spectral_axis_unit(...)`,
`with_spectral_axis_and_flux_units(...)` (added 1.18.0, 2024-10-16); `new_flux_unit` is deprecated.

**Continuum-divided / normalized flux — [I], and explicitly NOT verified.** No doc statement and
no test was found covering dimensionless flux. The inference is strong (the only verified
constraint is "must be a Quantity", which `u.dimensionless_unscaled` satisfies; and specutils'
own `equivalent_width` assumes a continuum "defaulting to 1, meaning continuum-normalized", which
only works if dimensionless flux round-trips). **Recommended form:**
`flux = norm_array * u.dimensionless_unscaled`, with a one-line smoke test in CI rather than
trusting it as documented API. This matters for albireo because its natural data product *is*
normalized.

### 4.9 How other packages draw the boundary — four verified patterns

**muler — subclassing, and the cautionary tale.** [V] `src/muler/echelle.py`:
```python
class EchelleSpectrum(Spectrum1D):
class EchelleSpectrumList(SpectrumList):
```
It subclasses the *deprecated* `Spectrum1D` while requiring `specutils>=2.1`. The cost shows in
the source — deprecation warnings silenced at module scope:
```python
warnings.filterwarnings("ignore", category=astropy.utils.exceptions.AstropyDeprecationWarning)
```
**[I]** A module-level `filterwarnings` mutates interpreter-wide filters, so importing muler
suppresses *every* `AstropyDeprecationWarning` in the user's process, including from unrelated
packages. That is the subclassing pattern's maintenance bill arriving.

**RADIS — lazy import (good) plus an `isinstance` landmine (bad).** [V]
`radis/spectrum/spectrum.py`, inside `from_specutils`:
```python
try:
    import specutils
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "Specutils is required to use this function."
        "Install it with:\n   conda install -c conda-forge specutils\nor\n   pip install specutils"
    ) from err

assert isinstance(spectrum, specutils.Spectrum1D)
```
The import placement is exemplary; the assertion is the §4.2 trap in the wild.

**pyspeckit — pure duck typing, zero imports.** [V] `from_spectrum1d` reads `.spectral_axis`,
`.flux`, `.uncertainty` and never imports specutils; the old module-level probe is commented out.
One flaw: it reaches into `spec1d._unit`, a private attribute.

**stingray / lightkurve — the `to_*`/`from_*` pair, done cleanly.** [V] `to_lightkurve()` imports
inside the function and raises a message naming the install command; `from_lightkurve()` is a
`@staticmethod` that just reads attributes. lightkurve mirrors it with `to_stingray()`/
`from_stingray()`.

**The structural insight [I]:** `from_X` needs **no import at all** — the object is handed to you,
so you read its attributes. Only `to_X` needs a lazy import, because only it constructs the
foreign type.

**Counterexample:** jdaviz lists `specutils>=2.0.0` as a *required* dependency — specutils-native,
so no optional boundary to learn from.

### 4.10 Recommended boundary for albireo [I]

Ranked, grounded in §4.9:
1. **Duck-typed `from_specutils()` + lazy-import `to_specutils()`** — stingray/lightkurve's shape
   with pyspeckit's import discipline. This survived the entire 1.x→2.x rename without a code
   change in pyspeckit's case, precisely because it never mentions a class name.
2. Conversion functions with `isinstance` guards (RADIS) — good hygiene undone by one type check.
3. Subclassing (muler) — worst: it converts specutils from optional to mandatory by definition.

**Two rules:** never `isinstance`-check a specutils class (compare duck-type attributes, or walk
`__mro__` by name if you must discriminate); and put specutils in an optional extra
(`albireo[specutils]`), never in `dependencies`.

**[I] Why this is worth doing at all:** §3.1's Astropy affiliated criterion 3 asks packages to
"use, interface with, or provide complementary capabilities to other Astropy packages". A
`to_specutils()`/`from_specutils()` pair is the cheapest possible satisfaction of that criterion —
roughly 40 lines and an optional extra — and it is currently absent (see §4A.1, gap 5).

### 4.11 astropy.timeseries: do not build on it [V]

`TimeSeries` and `BinnedTimeSeries` are `QTable` subclasses with a required `time` column holding
a `Time` object. The **provisional warning has been dropped**: the "work-in-progress (new in
v3.2) … quite possible there will be API changes" admonition present in the v4.0.dev docs is
**absent** from the current `docs/timeseries/index.rst` on main (checked directly in the reST
source). [V] Which release removed it was not bisected. [NOT VERIFIED]

Community use is split, and for RV the answer is **mostly no** [V]:
- **Photometry: yes.** lightkurve has `class LightCurve(TimeSeries):`, and its deprecated
  `to_timeseries()` now just says "LightCurve is a sub-class of Astropy TimeSeries as of
  Lightkurve v2.0".
- **RV: no.** TheJoker — the most time-aware RV fitter checked — defines its own `RVData`
  container taking "Either as an astropy `Time` object, or as a numpy array of Barycentric MJD
  (BMJD) values", and internally does `_t_bmjd = t.tcb.mjd`.

**[I]** Do not adopt `TimeSeries` as albireo's epoch container: it buys a `QTable` with a
mandatory `time` column, which fits ragged per-epoch spectra badly — the same problem as
`SpectrumCollection` (§4.5). The RV precedent is a purpose-built container holding an
`astropy.time.Time`.

### 4.12 Barycentric JD: what to store [V]

**Representation.** `Time(value, format='jd', scale='tdb')`. Astropy's own justification, verbatim:
> "Corrections to the barycenter are more precise than the heliocenter, because the barycenter is
> a fixed point where gravity is constant. For maximum accuracy you want to have your barycentric
> corrected times in a timescale that has always ticked at a uniform rate … For this reason,
> barycentric corrected times normally use the TDB timescale."

**The distinction that bites [I]:** `scale='tdb'` alone does **not** make a time barycentric. The
scale is the clock *rate*; the light-travel correction is a separate additive term. A
`Time(x, format='jd', scale='tdb')` that never had `light_travel_time` added is a TDB-scaled
*topocentric* time, and **astropy has no attribute recording that the correction was applied.**

**`Time.light_travel_time()`** [V]:
```python
light_travel_time(skycoord, kind='barycentric', location=None, ephemeris=None)
```
Returns a `TimeDelta`: "The time offset between the barycentre or Heliocentre and Earth, in TDB
seconds. **Should be added to the original time**". The canonical recipe, verbatim from the docs —
note the ordering, TDB *first*, then add:
```python
ltt_bary = times.light_travel_time(ip_peg)
time_barycentre = times.tdb + ltt_bary
```

**Eastman, Siverd & Gaudi 2010**, PASP 122:935–946, [arXiv:1005.4415](https://arxiv.org/abs/1005.4415) [V]:
> "We summarize the rationale behind our recommendation to quote the site arrival time, in
> addition to using BJD_TDB, the Barycentric Julian Date in the Barycentric Dynamical Time
> standard for any astrophysical event."

> "the (typically unspecified) time standards adopted by various groups can differ by as much as a
> minute. Left uncorrected, this ambiguity may be mistaken for transit timing variations and bias
> eccentricity measurements."

Magnitudes from the paper [V]: HJD−BJD differs by up to **~8 s**; the Einstein delay (TDB−TT) has
a 3.4 ms peak-to-peak amplitude over 1 yr; HJD "was formally deprecated by International
Astronomical Union (IAU) Resolution A4 in 1991, in favor of the BJD"; and "Any time stamp that is
quoted without a time standard should be assumed to be uncertain to at least 1 minute." The paper
cites 34 leap seconds as of Jan 2009; **[I]** the count is 37 as of 2026-09-03 (none since
2016-12-31), so a UTC-based JD now sits ~1.2 minutes from a TDB-based one — larger than the
1-minute threshold the paper is named for.

**Community convention: BJD_TDB, unambiguously.** HJD has been IAU-deprecated since 1991, and
HJD_UTC compounds two separate errors (~8 s heliocentric + ~37 s leap seconds).

**FITS keywords** [V], from Rots et al., "Representations of Time Coordinates in FITS"
([arXiv:1409.7583](https://arxiv.org/abs/1409.7583); A&A volume/page **[NOT VERIFIED]** from the
preprint):

| Keyword | Meaning | Default |
|---|---|---|
| `TIMESYS` | time scale | **UTC** |
| `MJDREF` / `MJDREFI`+`MJDREFF` / `JDREF` | reference epoch | — |
| `TREFPOS` | spatial reference position | — |
| `TREFDIR` | reference direction for pathlength correction | — |
| `TIMEUNIT` | time unit | **s** |
| `TIMEPIXR` | pixel position of the time stamp | **0.5** |

`TREFPOS` values include `TOPOCENTER, GEOCENTER, BARYCENTER, HELIOCENTER, …`. The standard notes
that if the reference position is not `TOPOCENTER` for observational data, **`TREFDIR` should also
be provided**. `TIMESYS` defaulting to **UTC** is exactly the silent-1-minute trap Eastman et al.
describe — always write it explicitly. Note that lightkurve's generic FITS reader defaults the
scale to `tdb` when `TIMESYS` is missing, the *opposite* of the standard's default; **[I]** that
disagreement is itself the argument for never relying on a default in either direction.

**Recommendation for albireo [I]:**
- Store one canonical thing: `Time(format='jd', scale='tdb')` with the light-travel correction
  already applied, and **name the attribute so the correction is not in doubt** —
  `epoch_bjd_tdb`, not `time` or `jd`.
- Because astropy cannot record that the correction was applied, carry an explicit
  `time_reference_position = 'barycenter' | 'topocenter'` alongside it, mirroring FITS `TREFPOS`.
  This is the single most valuable piece of metadata and the one astropy will not keep for you.
- Accept an `astropy.Time` in any scale (→ `.tdb`); for bare floats, **require** the caller to
  state scale and reference position. TheJoker's silent "bare array means BMJD" is convenient and
  is exactly how a 37-second error enters a dataset unnoticed.
- Offer but never silently apply the correction, via a helper that requires `SkyCoord` and
  `EarthLocation` and refuses without a `location`.
- On FITS output write `TIMESYS='TDB'`, `TREFPOS='BARYCENTER'` and `TREFDIR`; never an offset time
  without its reference keyword. Retain the site arrival time too, per Eastman et al.

---

## 4A. Repository audit against the review checklists

**[V]** Read-only inspection of `C:\Users\thari\Documents\GitHub\albireo` on 2026-09-03. No files
were modified. This is included because it converts §2–§4 from general advice into a specific
short list.

### 4A.0 What is already right

| Item | Status |
|---|---|
| `README.md`, `LICENSE`, `CONTRIBUTING.md`, `CHANGELOG.md` | **present** |
| `CITATION.cff` | **present**, with `cff-version: 1.2.0`, ORCID/affiliation/DOI already marked `TODO(release)` |
| `codemeta.json` | **present** (codemeta 3.0) |
| `.zenodo.json` | **absent** — and per §3.6 this is *correct*: adding it would silently disable CITATION.cff parsing |
| Build backend | `hatchling>=1.27`, `dynamic = ["version"]` single-sourced from `__init__.py`, with a test asserting `pyproject`/`__version__`/`CITATION.cff` agree |
| Optional deps | `io = ["astropy>=6"]`, `plots = [...]`, `dev`, `docs` — astropy correctly optional |
| CI/tooling | `.github/`, `.pre-commit-config.yaml`, ruff, mypy, pytest+cov |

The CITATION.cff already carries the right warning in a comment: "Enable the GitHub-Zenodo webhook
*before* tagging, or the tag is not archived and the DOI story starts one release late." That is
correct and matches §3.6.

### 4A.1 Concrete gaps, ordered by cost

| # | Gap | Why it matters | Cost |
|---|---|---|---|
| 1 | **Not on PyPI** | Hard prerequisite for pyOpenSci (§3.2, out-of-scope list) and the name is unclaimed (§1.1) | minutes |
| 2 | **`CODE_OF_CONDUCT.md` missing** | Explicitly on the pyOpenSci author checklist (§3.2) | minutes |
| 3 | **ORCID absent from `CITATION.cff` and `paper/paper.md`** | JOSS expects an ORCID for the corresponding author; already self-flagged as `TODO(release)` | minutes |
| 4 | **`Development Status :: 2 - Pre-Alpha` classifier**, and `codemeta.json` `"developmentStatus": "wip"` | §2.6: "pre-alpha" reads as "half-baked" against a criterion that forbids half-baked solutions | minutes |
| 5 | **No specutils interop anywhere** — `grep -rl specutils src/` returns nothing | Astropy affiliated criterion 3 (§3.1, §4.10) | ~40 lines + an extra |
| 6 | **No Zenodo DOI yet** | Needed by JOSS at acceptance and required by pyOpenSci post-acceptance (§3.2) | one tagged release |
| 7 | **No stability policy page** | §7.4; JOSS's single-author clause names "stated support or governance expectations" | one page |
| 8 | **`paper/paper.md` is missing 4 of JOSS's 6 required sections** — has Summary + Statement of need + `Functionality`; lacks State of the field, Software design, Research impact, AI usage disclosure. Affiliation is literally `name: TODO` | §2.8. Would surface only at submission otherwise | a rewrite, not an edit |

### 4A.1a LAUNCH BLOCKER: the documentation site does not exist [V]

**The docs URL in the brief, <https://tjayasinghe.github.io/albireo/>, returns HTTP 404.** This
was checked five independent ways on 2026-09-03 and every one agrees:

| Check | Result |
|---|---|
| `WebFetch https://tjayasinghe.github.io/albireo/` | **HTTP 404 Not Found** |
| `gh api repos/tjayasinghe/albireo/pages` | **404 Not Found** — GitHub Pages is not enabled |
| `gh variable list` | **empty** — the `PAGES_ENABLED` gate variable is unset |
| Latest Docs workflow run (33802138513, 2026-09-03) | `Build: success`, **`Deploy: skipped`** |
| `mkdocs.yml` `site_url` | `https://tjayasinghe.github.io/albireo/` — matches the dead URL |

The cause is not a bug; it is a deliberate gate that was never released. `.github/workflows/docs.yml`
runs `deploy` only `if: … && vars.PAGES_ENABLED == 'true'`, and its own comment explains why and
gives the fix verbatim:

> "Set the variable once Pages is on:  `gh variable set PAGES_ENABLED --body true`"

The workflow's reasoning was sound at the time — Pages from a private repo needs GitHub Pro, so
the deploy was gated to avoid a permanently red Docs badge before the repo went public. **The
repo has been public since mid-August and the gate was never opened.**

**[I] Why this matters more than anything else in this report.** The README carries a green
**Docs** badge (it reflects the `Build` job, which does pass) whose *link target* is the 404. So
the repository actively advertises documentation that cannot be read. Every recommendation in §7
about quickstarts, Diataxis and stability pages is inert until this is fixed; so is the pyOpenSci
checklist item "user-facing documentation with quickstart tutorials" (§3.2), which a reviewer
would check by clicking that badge. It is also the most likely explanation for a repository that
has been public for three weeks and has **1 star** — there is nothing for a visitor to land on.

**Fix, in order:** enable Pages with source "GitHub Actions" in repository settings →
`gh variable set PAGES_ENABLED --body true` → push to `main` (or `workflow_dispatch`) → confirm
the URL resolves → set the repo's `homepageUrl`, which is currently **empty**, to the docs URL so
the link appears in the GitHub sidebar.

### 4A.1b Other repository-level observations [V]

From `gh repo view tjayasinghe/albireo`, 2026-09-03:

| Field | Value |
|---|---|
| `visibility` | **PUBLIC** |
| `createdAt` | **2026-08-11T17:35:17Z** |
| `pushedAt` | 2026-09-03T20:25:09Z |
| `stargazerCount` | **1** |
| `hasIssuesEnabled` | true |
| `hasDiscussionsEnabled` | **false** |
| `homepageUrl` | **empty** |
| `description` | "Spectral disentangling" |
| `licenseInfo` | BSD 3-Clause |

**[I] Three notes.** (1) The repo was *created* 2026-08-11, three days before the 2026-08-14 date
in the brief; `createdAt` does not distinguish "created" from "made public", so the conservative
2026-08-14 is the right date to use for the JOSS clock (§2.1) — but if it was public from creation,
the gate opens 2027-02-11. Worth confirming, since it costs nothing. (2) **GitHub Discussions is
disabled.** It is the standard "community signals" surface that JOSS's research-impact criterion
and pyOpenSci's support-expectations item look for, and it is a one-click enable. (3) The
one-line `description` — "Spectral disentangling" — is the text that appears in GitHub search
results and on the profile. It omits every discriminating keyword (Bayesian, JAX, SB2,
spectroscopic binary) and, per §1.3, the *name* carries none of them either.

### 4A.2 Two packaging details that connect to §1

**[V]** `requires-python = ">=3.12"`, with the in-file reason: "current jaxlib (0.11.x) no longer
ships cp311 wheels". **[I]** So a conda-forge recipe would set `{% set python_min = "3.12" %}`
(§1.5). Note this is a *tighter* floor than specutils' own `>=3.11` (§4.1), which is fine.

**[V]** `[project.scripts]` declares `albireo = "albireo.cli:main"`. **[I]** Under `noarch: python`
this console script **must be listed explicitly in the recipe's build section** — §1.5 quotes the
rule, and it is the most common noarch mistake for a package with a CLI.

---

## 5. Adoption case studies — dated evidence

Two independent collections agree: GitHub metrics gathered by me via `gh api` and by a
sub-researcher via `api.github.com`, both on 2026-09-03, match exactly on every overlapping repo
(lightkurve 531, emcee 1597, dynesty 422, exoplanet 239, PHOEBE2 105, batman 105, celerite2 85,
Starfish 77, radvel 67, juliet 67, Korg 67, iSpec 38, TheJoker 34, PySME 33, PSOAP 33). Download
figures are from pypistats.

**One methodological warning that must travel with these numbers.** `ui.adsabs.harvard.edu`
returns **HTTP 405** to automated fetching and the ADS API needs a token we do not have, so
**every citation count below is OpenAlex and/or Semantic Scholar, not ADS.** For astronomy, ADS is
normally *higher* than both, and both badly undercount conference proceedings (OpenAlex reports 11
citations for the FDBinary ASP Conf. paper, certainly a large undercount). Treat the citation
numbers as ordinal, not absolute.

### 5.1 GitHub, distribution and downloads [V, 2026-09-03]

| Package | Stars | Created | Last push | PyPI | dl/month | conda-forge | Discussions |
|---|---:|---|---|---|---:|---|---:|
| emcee | **1597** | 2011-11-07 | 2026-08-31 | `emcee` 2012-02-12 | **1,206,327** | yes | off |
| lightkurve | **531** | 2018-01-22 | 2026-08-25 | `lightkurve` 2018-01-22 | 32,808 | yes | **30** |
| dynesty | 422 | 2017-05-02 | 2026-08-08 | `dynesty` 2018-02-25 | 75,377 | yes | off |
| exoplanet | 239 | 2018-06-20 | 2026-08-31 | `exoplanet` 2018-11-14 | 1,186 | yes | 1 |
| PHOEBE 2 | 105 | 2016-08-22 | 2026-08-28 | `phoebe` 2017-03-22 | 5,837 | **no** | **263** |
| batman | 105 | 2015-05-26 | **2025-05-05** | `batman-package` 2015-06-24 | 19,009 | yes | off |
| celerite2 | 85 | 2019-01-17 | 2026-08-31 | `celerite2` 2020-08-10 | 8,357 | yes | off |
| **Starfish** | 77 | 2013-07-24 | 2026-07-22 | **`astrostarfish`** 2019-05-16 | **103** | no | off |
| radvel | 67 | 2015-12-21 | 2026-09-02 | `radvel` 2017-04-27 | 1,690 | no | off |
| juliet | 67 | 2018-11-27 | 2026-02-16 | `juliet` 2019-09-03 | — | no | off |
| Korg.jl | 67 | 2021-02-18 | 2026-09-03 | Julia registry (95 versions) | — | n/a | 1 |
| **iSpec** | 38 | 2015-10-05 | 2026-04-15 | **not on PyPI** | — | no | off |
| **TheJoker** | 34 | 2016-09-04 | 2025-12-02 | `thejoker` 2019-03-01 | **85** | no | off |
| **PySME** | 33 | 2018-09-24 | **2024-12-01** | `pysme-astro` 2019-05-28 | 2,536 | no | off |
| **PSOAP** | 33 | 2016-10-08 | **2017-12-06** | **never — 404** | — | no | off |
| shift-and-add | 10 | 2022-11-23 | 2024-03-07 | never | — | no | off |
| **GSSP** | no GitHub | — | — | never | — | no | — |

Scale references [V]: `astropy` 3,085,629 dl/month; `specutils` 63,587.

**[V] The namespace finding, and it is not hypothetical.** `starfish` on PyPI belongs to
**spacetx/starfish**, "Pipelines and pipeline components for the analysis of image-based
transcriptomics data". Czekala's astronomy Starfish lost its own name and ships as
`astrostarfish`. `ispec` on PyPI is an unrelated package by `dsm-72`. This is the strongest
possible argument for §1.3: **register `albireo` today.**

### 5.2 Papers and citations [V — OpenAlex / Semantic Scholar, NOT ADS]

| Package | Venue | Date | OpenAlex | Paper vs. installable artefact |
|---|---|---|---:|---|
| **lightkurve** | **none — ASCL only** (`ascl:1812.013`) | 2018-12 | n/a | **no refereed paper, ever** |
| emcee | PASP 125, 306 | 2013-03-01 | **12,213** | code on PyPI 4 days before the preprint |
| dynesty | MNRAS 493 | 2020-01-30 | 2,299 | code **23 months** earlier |
| batman | PASP 127 | 2015-09-28 | 1,048 | code 5 weeks earlier |
| celerite | AJ 154 | 2017-11-09 | 1,029 | code earlier |
| radvel | PASP 130 | 2018-03-12 | 528 | code **10.5 months** earlier |
| iSpec | A&A 569, A111 | 2014-07-07 | 514 | — |
| juliet | MNRAS 490 | 2019-10-03 | 356 | ~simultaneous |
| PHOEBE II | ApJS 227, 29 | 2016-12-01 | 326 | before PyPI |
| exoplanet | JOSS 6(62) | 2021-06-22 | 215 | code **31 months** earlier |
| TheJoker | ApJ 837, 20 | 2017-02-28 | 137 | **paper 24 months BEFORE pip** |
| **Starfish** | ApJ 812, 128 | 2015-10-14 | 136 | **paper 43 months BEFORE pip** |
| GSSP | A&A 581, A129 | 2015-07-21 | 106 | never packaged |
| **PSOAP** | ApJ 840, 49 | 2017-05-01 | **53** | **never packaged** |
| PySME | A&A 671, A171 | 2023-01-05 | 51 | code **43 months** earlier |
| Korg | AJ 165, 68 | 2023-01-24 | 31 | code earlier |
| FDBinary | ASP Conf. 318, 111 | 2004-12 | 11 (undercount) | — |

### 5.3 The incumbents albireo is replacing [V]

**fd3 / FDBinary** (Ilijić, Univ. of Zagreb), <http://sail.zpf.fer.hr/fd3>:
- The page says verbatim: **"last update July 2014."**
- **The host has no HTTPS.** Port 80 returns 200; **port 443 refuses connections.** Any modern
  fetcher that upgrades http to https fails outright.
- Distribution is two files on a static page: a precompiled **Linux x86 binary** (2,132,894 bytes,
  `Last-Modified: Fri, 25 Jul 2014`) and `fd3.tar.gz` (1,936,653 bytes, same date).
- **No version numbers, no licence on the page, no changelog, no issue tracker, no GitHub, not on
  PyPI, not on conda.** As of 2026-09-03 the artefacts are **12 years and 1 month old.**
- The companion wavelength-space code CRES is marked **"last update 12. June 2006"** — 20 years.

**KOREL** (Hadrava, Astronomical Institute Prague), <http://space.asu.cas.cz/~had/korel.html>,
revised 29.7.2014:
- **The code is not downloadable at all.** The only download is `PREKOR17.EXE`, a Windows binary
  for the *preprocessor*. The engine is offered solely as the **VO-KOREL** web service, which
  **"requires the user to register (create unique account)"** with email activation. Engine
  version "KOREL11b - release 9. 9. 2011"; service build dated 2019-01-15. Caps: 2000 spectra,
  16384 bins, 50 regions, 3 templates.

**Shift-and-add** (`TomerShenar/Disentangling_Shift_And_Add`): 10 stars, 54 commits, 1
contributor, last commit 2024-03-07, **no licence, no releases ever, no docs, no CI**.

**PyPI check — all 404** [V]: `fd3`, `fdbinary`, `korel`, `disentangling`,
`spectral-disentangling`, `psoap`, `gssp`, `pyterpol`.

### 5.4 The size of the niche [V] — the number that reframes everything

GitHub's search API for **"spectral disentangling"** returns **19 repositories worldwide**.
Filtering to astronomy leaves six:

| Repo | Stars | Created | Last push |
|---|---:|---|---|
| `ayushmoharana/fd3_initiator` | 5 | 2021-10-21 | 2025-06-26 |
| `aplidinio/Spectral-Disentangling` | 2 | 2020-07-14 | 2020-07-14 |
| `robert-klement/KOREL_tools` | 2 | 2020-09-07 | 2020-09-07 |
| **`tjayasinghe/albireo`** | **1** | 2026-08-11 | 2026-09-03 |
| `tanmayhinge/spectral-disentangling` | 0 | 2026-07-07 | 2026-07-17 |
| `Xinlin-code/Spectral-disentangling` | 0 | 2026-06-06 | 2026-06-06 |

Everything else matching the phrase is machine learning (traffic forecasting, hyperspectral
super-resolution, LLM continual learning).

**[I] The two halves of this point face opposite directions and both are true.**
1. **There is no incumbent to out-engineer.** The bar is a 2014 C binary served over plain HTTP
   with no licence, and a Fortran engine from 2011 behind a registration wall. Any pip-installable,
   tested, documented package is instantly the best-engineered option in the field. Nobody will
   lose a bake-off on developer experience.
2. **There is also no community to inherit.** The niche's total GitHub star mass is under 20.
   Adoption cannot arrive via GitHub discovery, PyPI browsing, or "build it and they will come" —
   those channels are empty here. And note that the two most-cited disentangling codes achieved
   their citations *with none of the modern apparatus*: this field cites tools it obtains as
   tarballs. **Better packaging is necessary but demonstrably not sufficient.**

### 5.5 What the adopted ones did in year one that the unadopted ones did not

**Finding 1 — They shipped an installable artefact BEFORE or WITH the paper.** [V] This is the
sharpest single discriminator in the dataset.

*Adopted, code first*: emcee (PyPI 4 days before the preprint, 13 months before the paper);
lightkurve (PyPI on the *day of the first commit*, and never published a paper); batman (5 weeks);
radvel (10.5 months); dynesty (23 months); exoplanet (31 months); PySME (43 months).

*Unadopted, paper first*: TheJoker (paper 24 months before pip → **85 dl/month**); Starfish (paper
43 months before pip → **103 dl/month**); PSOAP (paper, then never packaged → dead).

**The control case that proves it is not simply "good code wins":** PySME has **51** citations and
**2,536** dl/month. Starfish has **136** citations and **103** dl/month. Starfish is cited ~2.7x
more often and downloaded ~25x *less*. The difference is that PySME was pip-installable 3.5 years
*before* its paper and Starfish 3.5 years *after* its.

**Finding 2 — Effort was not the difference. Releases were.** [V] First-365-day activity:

| Package | Commits yr 1 | **Releases yr 1** | Outcome today |
|---|---:|---:|---|
| lightkurve | 606 | **12** | 531 stars, ~1M downloads |
| celerite | 979 | 6 | 191 stars |
| exoplanet | 545 | **7** | 239 stars |
| Korg.jl | 535 | **18** | 95 registry versions |
| **TheJoker** | **463** | **1** | 34 stars, 3 contributors |
| **Starfish** | **383** | **0** | package 4 years stale |
| **PSOAP** | 50 | 1 (beta) | dead at 13 months |
| **shift-and-add** | 53 | **0** | 10 stars |

Starfish's author put **383 commits** into year one — more than radvel (279), batman (193),
dynesty (131) or emcee (**51**) — and cut **zero releases**. TheJoker's author put in 463 and cut
one. These projects were not under-worked; they were **unreleased**. emcee shipped on 51
first-year commits and now does 1.2M downloads a month.

**[I] This finding lands directly on §2.2: albireo currently has 72 commits and zero tags.** It is
presently on the Starfish/TheJoker side of this table, and the fix is a `git tag`.

**Finding 3 — The adopted ones had an institutional home, and that beat every other factor.** [V]
The most-downloaded package in the survey has **no refereed paper at all**. NASA's TESS Science
Support Center page states verbatim:
> "Lightkurve is a Python-based package developed by the Kepler/K2 Guest Observer (GO) Office for
> use by the community to work with Kepler and K2 data."

> "The TESS GI Office has partnered with the Kepler/K2 GO Office to adapt Lightkurve for use with
> TESS data."

The same NASA page also lists **batman, exoplanet, juliet and radvel**. It does **not** list iSpec,
GSSP, Starfish, PSOAP, fd3, KOREL, PHOEBE, TheJoker, dynesty or emcee.

The second route is **being someone else's dependency**: `bilby` 2.8.2 (the LIGO/Virgo/KAGRA
inference library, uploaded 2026-08-10) hard-requires `dynesty>=2.0.1` and `emcee`; juliet's docs
describe it as "a wrapper" of batman, radvel, george, celerite and dynesty. **Downloads flow down
dependency edges; citations do not.**

**[I] The distinction TheJoker draws is the one to internalise.** TheJoker *was* applied to a
survey — Price-Whelan et al. 2020 found 20,000 binaries in APOGEE DR16, 126 citations — and it
still has 3 contributors and 85 downloads a month. **A survey application by the author generates
citations. A survey pipeline dependency generates users.** These are different outcomes and
require different actions.

**Finding 4 — Where there was no institutional home, sustained teaching substituted for it.** [V]
PHOEBE 2 has 105 stars, is **not on conda-forge**, and has **263 GitHub Discussions** — more than
every other repo in the survey combined. Its `phoebe-project/phoebe2-workshop` repo contains
directories `2018june, 2019july, 2020june, 2021june, 2022june, 2023june, 2024june, 2025aug,
2026aug` — **nine consecutive annual workshops holding 470 Jupyter notebooks** — last pushed
2026-08-07. It also publishes a numbered paper series (PHOEBE II 2016, 326 citations; V 2020, 211).

**[I]** PHOEBE is the proof that a niche binary-star code *can* build a genuine user community
without a mission behind it. The price was a recurring annual workshop sustained for nine years,
not a launch.

**Finding 5 — Install friction is fatal even with strong citations.** [V] iSpec has **514
citations** and **~1 GitHub contributor**. GSSP has **106 citations** and **no GitHub presence at
all**. The measured friction: iSpec requires a separately downloaded **2.2 GB** `input.tar.gz`
(2,216,270,412 bytes, Last-Modified 2025-08-01), and its source path instructs the user to compile
readline 6.2, sqlite, tcl 8.5.17, tk 8.5.17 and CPython 3.8.5 by hand. GSSP is seven Fortran 90
tarballs on a **2015 conference webpage**, last updated 2016-11-29, requiring Intel Fortran +
OpenMPI, with **no licence stated**.

**[I]** Citation count measures whether someone used a tool once for a paper. Contributor count
and downloads measure whether it entered anyone's weekly workflow. These codes achieved the first
and not the second.

**Finding 6 — The small mechanical things** [V]:
- **Community files**: lightkurve and exoplanet carry CITATION + CODE_OF_CONDUCT + CONTRIBUTING.
  TheJoker, juliet, batman, radvel, PSOAP and shift-and-add carry none.
- **CI**: juliet, PSOAP, iSpec and shift-and-add have **zero** GitHub workflows.
- **conda-forge** correlates with adoption but is clearly **not necessary** — radvel and PHOEBE
  have none and do 1,690 and 5,837 downloads/month respectively. This supports §1.4's conclusion
  that the missing win-64 jaxlib is not a launch blocker.
- **Slack/Discord: no evidence that ANY of the 15 packages runs one.** This is not a lever anyone
  in this field has pulled. GitHub Discussions is the observed substitute, and only PHOEBE has
  made it work.
- **Executable notebooks in-repo**: lightkurve 38, dynesty 14, celerite 11, radvel 9, TheJoker 8,
  emcee 6, Starfish 4; **iSpec 0, shift-and-add 0**.

### 5.6 What this implies for albireo [I]

1. **The six-month JOSS wait is not a launch blocker — it is the normal case.** dynesty waited 23
   months, exoplanet 31, PySME 43, lightkurve never published at all. What matters in the window
   is a **release cadence** (the lightkurve/exoplanet/Korg band is 7–18 tagged releases in year
   one) and **PyPI presence**. albireo currently has neither (§2.2, §4A.1) and is therefore on the
   Starfish/TheJoker side of Finding 2. Both are fixable this week.
2. **The highest-value action is finding an institutional or dependency home**, because that is
   what separated every adopted package from every unadopted one, and it is the only factor that
   overcame having no paper at all. Realistic candidates here: a survey with SB2 spectra
   (BLOeM/VLT-FLAMES, SDSS-V, 4MOST, WEAVE), the Gaia DR4 double-lined RVS products (§6.1), or
   becoming an upstream dependency of an existing binary-star tool. Note Finding 3's warning: being
   *applied* to a survey by the author is measurably not the same thing.
3. **Absent that, the PHOEBE route is the only demonstrated alternative** — and it is well matched
   to a field whose users currently run a 2014 C binary. They need teaching more than they need an
   API. The PoWR code course (§6.4) and an EAS 2027 session (§6.2) are the cheap first steps on
   that path.
4. **Do not expect GitHub discovery to work.** The niche is 19 repos with near-zero stars; there is
   no ambient traffic to capture. Adoption must be pushed through the channels in §6.5.
5. **A realistic ceiling.** PHOEBE, the healthiest niche binary-star code in the survey, does 5,837
   downloads/month; radvel 1,690; TheJoker 85. albireo's success should be measured in citations,
   named users and pipeline inclusion — **not** downloads, where a good outcome looks numerically
   tiny next to emcee's 1.2M.

---

## 6. Venues, newsletters and channels, 2026–2027

### 6.1 Gaia DR4: 2 December 2026 [V] — verified against ESA directly

<https://www.cosmos.esa.int/web/gaia/release> (fetched 2026-09-03), verbatim:

> "Gaia DR4 (based on 66 months of data) 2 December 2026"

DR5 — "not before the end of 2030", the "Complete Gaia Legacy Archive of all data". [V]

**DR4 contents that matter for a disentangling package**, from
<https://www.cosmos.esa.int/web/gaia/dr4> (fetched 2026-09-03) [V]:

| Product | Count | Size |
|---|---|---|
| Non-single-star **two-body orbital solutions** ("covers astrometric binaries, spectroscopic binaries, eclipsing binaries") | **5,901,111 sources** | 2.7 GB |
| **Spectroscopic binary solutions** (Line Integral Analysis modules) | **638,966 sources** | 99 MB |
| RVS epoch parameters, **double-lined transits** (RV, line broadening, G_RVS) | **787,312 transits** | 128 MB |
| RVS epoch parameters, single-lined transits | 6,910,423,894 transits | 323 GB |
| **RVS epoch spectra** | 6.9 billion spectra | **49 TB** |
| Astrometric NSS (acceleration) solutions | 3,365,947 | 1.2 GB |
| Resolved physically-bound pairs | 22,829,105 | 8.4 GB |
| Multiple-orbit (>2 body) solutions | 16,681 | 128 MB |
| Epoch astrometry / photometry / BP-RP spectra | 129.6 G / 121.8 G / 121 G | 62 / 7.2 / 119 TB |

**[I] This is the single largest strategic fact in the report.** On 2 December 2026 the field
acquires ~788k double-lined RVS transits and ~639k spectroscopic-binary solutions, and — for the
first time — released **epoch RVS spectra**. That is albireo's addressable input, arriving on a
fixed date, 90 days out. Two implications:
- The project's existing decision to defer Gaia RVS work to DR4 is vindicated by ESA's own page.
- A DR4-ready reader and a worked DR4 example published *in the weeks after 2 December 2026* is
  worth more attention than the same work published at any other time, because the whole
  community is looking at the same data simultaneously. This is the natural anchor for a 0.1.0
  announcement.

### 6.2 EAS 2027 — the nearest actionable deadline [V]

<https://eas.unige.ch/EAS2027/> (fetched 2026-09-03):
- **21–25 June 2027**, **Vienna, Austria**, Austria Centre Vienna.
- **Call for special sessions and symposia: proposals due 30 September 2026.** [V]

**[I] This is 27 days away and is the highest-leverage single action in the report after
registering the PyPI name.** A proposed EAS 2027 special session on spectral disentangling /
binary spectroscopy in the Gaia DR4 era would (a) land six months after DR4, (b) put the author in
the room with the exact target community, and (c) cost only a proposal now. Even as a co-proposer
on someone else's session it is worth pursuing. Note that a session proposal is a *community
organising* act, which is also the kind of evidence JOSS and pyOpenSci call "research impact" and
"external adopters".

**[NOT VERIFIED]**: EAS 2026 location/dates (the 2027 page did not carry them), and the abstract
submission deadline for 2027 (typically ~March, but not confirmed here).

### 6.3 IAU [V, partial]

- **XXXIII IAU General Assembly: Rome, 10–19 August 2027**, hosting **six Symposia and twelve
  Focus Meetings**; the 2027 selection was announced in June 2026, and the Letters of Intent and
  full-proposal calls are **closed**. [V]
- One 2027 symposium identified by name: IAUS 414, "High-z galaxies and Black Holes: The
  JWST/ALMA/GRAVITY Frontier", 11–15 October 2027, Puerto Natales, Chile — not relevant.
- **[NOT VERIFIED]**: the full list of 2027 Symposia and Focus Meetings, and whether any concerns
  binaries, massive stars or stellar spectroscopy. The IAU announcement page
  <https://www.iau.org/IAU/Iau/News/Ann2026/IAU-Scientific-Meetings-2027.aspx> returned
  **HTTP 403** to WebFetch. This should be re-checked manually — it is a cheap look and a
  binary/massive-star Focus Meeting at the Rome GA would be a prime venue.
- **[I]** Since 2027 proposals are closed, IAU is an *attendance and abstract* venue for albireo,
  not an organising one. The relevant deadline will be the GA abstract call, typically ~February–March 2027.

### 6.4 Massive-star and binary meetings [V]

From <https://massivestars.org/category/events/> (the IAU Commission on Massive Stars events
feed, fetched 2026-09-03). Ordered by date; note that today is 2026-09-03, so the first two have
already happened:

| Meeting | Dates | Location | Note |
|---|---|---|---|
| "To Be or Not to Be" — **OBe Stars** meeting | 13–17 Jul 2026 | Liège, Belgium | **Past.** Focus included disks and **stripped companions** |
| AstroStat School 7 | 27–31 Jul 2026 | Rome, Italy | Past (applications closed 30 Apr 2026) |
| **Modeling Spectra of Stellar Atmospheres and Winds (PoWR code course)** | **21–25 Sep 2026** | Potsdam, Campus Golm, Germany | **18 days away.** Hands-on training, **no registration fee** |
| 8th Black Hole Nepal Meeting | 12–16 Oct 2026 | Kathmandu, Nepal | Stellar-mass black holes |
| **XXXVII Canary Islands Winter School** — "Massive stars as cosmic tools from birth to core collapse SNe and GW events" | 16–27 Nov 2026 | La Laguna, Tenerife, Spain | Application deadline **passed** (15 Jun 2026) |
| HWO2026 | 30 Nov – 4 Dec 2026 | Paris, France | Habitable Worlds Observatory; not relevant |
| **Symposium "Massive Stars: Beacons in a Stormy Universe"** | **12–16 Jun 2028** | Šibenik, Croatia | Continues the 60-year massive-star symposium series |

**[I] Two observations.** First, the **PoWR code course (21–25 Sep 2026)** is the most
albireo-adjacent event in the near term — a hands-on training week on modelling massive-star
spectra, free, with exactly the audience that disentangles SB2s. Even attending, or getting the
package mentioned to the organisers, is disproportionately cheap relative to a conference talk.
Second, the OBe-stars meeting in July 2026 explicitly covered **stripped companions**, which is
the HR 6819 problem class; that community exists and meets, and the next comparable gathering is
the 2028 Šibenik symposium. In between, **EAS 2027 (§6.2) is the realistic venue**, which is why
the 30 Sep 2026 session-proposal deadline matters.

Still **[NOT VERIFIED]** — the WebSearch budget was exhausted before these could be covered:
- **Cool Stars 22/23** dates, location and splinter-session process. Cool Stars splinters are
  proposed a few months ahead and are an unusually good fit for a software demo — worth chasing.
- The "Impact of Binaries on Stellar Evolution" series status.
- Announced Gaia DR4 workshops/hackathons, and whether the "Gaia Sprint" series still runs.
  Given the 2 Dec 2026 release (§6.1), DR4 community events are very likely to be announced in
  the coming weeks; this is worth a standing watch rather than a one-off search.

### 6.5 Newsletters and community channels

**Massive Star Newsletter — active** [V]. <https://massivestars.org/newsletter/> (fetched
2026-09-03). Run under the **IAU Commission on Massive Stars**; contact `massivestars@kuleuven.be`;
**monthly**; PDF archive at <https://massivestars.org/pdf-archive/>. Submissions via
<https://massivestars.org/submit-an-entry/>, accepting **papers, theses, proceedings, event
announcements, jobs, press releases, seminars and miscellaneous items**. No fixed deadline
pattern stated. **[I]** The "papers" and "miscellaneous" categories are a legitimate, zero-cost
route to announce a software release to precisely the right readership; this is probably the
highest signal-to-effort channel in this whole section for albireo's audience.

**Be Star Newsletter / Binary Star Newsletter** — **[NOT VERIFIED]**, budget exhausted. The
Massive Stars site does host related newsletters (it announced the Cool Evolved Stars Newsletter
opening to the community), so the newsletter ecosystem there is worth one look.

**Astropy channels** [V], from <https://www.astropy.org/help.html> (fetched 2026-09-03) — note
that all four are live; there was no wholesale migration:
| Channel | URL | Role |
|---|---|---|
| **Open Astronomy Discourse** | <https://community.openastronomy.org/c/astropy/8> | user questions; tagged, Google-searchable, good code rendering |
| **Astropy Slack** | join at <http://joinslack.astropy.org>, `#community-help` | primarily development; `#community-help` for users |
| astropy users list | astropy@python.org | general Python-in-astronomy |
| astropy-dev list | astropy-dev@googlegroups.com | "where significant announcements for contributors/developers are usually made" |

GitHub Discussions and Stack Overflow are **not** listed by Astropy. [V]

**RNAAS — Research Notes of the AAS** [V], <https://journals.aas.org/research-notes/>:
- **≤1,500 words**, and **no more than a single figure *or* table (but not both)**.
- **Not peer reviewed**; "moderated but not edited, which allows them to be rapidly published
  online within days of acceptance."
- Scope: "works in progress, comments and clarifications, null results, or timely reports of observations."
- "**searchable in ADS and fully citable**, and they are archived for perpetuity."
- **[NOT VERIFIED]**: the publication charge. RNAAS has historically carried a small fee; confirm
  before relying on it.
- **[I]** Software announcements are not named in the scope list, and RNAAS is a *thin* vehicle
  for one. Its real use for albireo is different and better: a 1,500-word note reporting a
  concrete, dated result — e.g. a DR4 disentangling result on a specific system — which is timely,
  ADS-indexed, citable, and mentions the software in passing. That builds the ADS trail that
  JOSS's "research impact" criterion asks for, without pretending to be a methods paper.

**AAS Nova** [V, partial], <https://aasnova.org/> (fetched 2026-09-03). It is described as
> "a curation service to inform astronomy researchers and enthusiasts about breakthroughs and
> discoveries they might otherwise overlook."

It covers five AAS journals: AJ, ApJ, ApJL, ApJS and the Planetary Science Journal. [V]
**[NOT VERIFIED]**: the site states no selection methodology and offers no author-submission
route; `aasnova.org/about/` returns 404. **[I]** The word "curation" and the absence of any
submission mechanism indicate editorially selected coverage of already-published AAS-journal
papers. AAS Nova is therefore a *consequence* of publishing in an AAS journal, not a channel to
submit to. Do not build a launch plan around it. It is also irrelevant to a JOSS or A&A/MNRAS
route, since it only covers AAS journals.

**astro-ph.IM** and **astronomy social media in 2026** — **[NOT VERIFIED]** in the detail
requested. What was retrieved: Astrodon (<https://astrodon.social/>) exists as a Mastodon server
for astronomy; Bluesky is around 35M registered users and has cooled since a January 2025 peak;
Threads passed X in daily mobile users in January 2026 (141.5M vs 125M); and altmetric "credit"
for scholarly content is tallied on **X and Bluesky**. **[I]** The astronomy research community's
centre of gravity is Bluesky with a Mastodon/Astrodon tail; X retains reach but declining
research-community engagement. For a package launch, post to **Bluesky and Astrodon**, and treat
the Massive Star Newsletter and the OpenAstronomy Discourse as the channels that actually reach
buyers. This should be re-verified rather than relied upon.

---

## 7. Documentation and release conventions

### 7.1 Diataxis [V]

<https://diataxis.fr/> and <https://diataxis.fr/tutorials-how-to/> (fetched 2026-09-03). Four modes:
**tutorials, how-to guides, technical reference, explanation**.

The distinction that matters, verbatim:
> A tutorial's purpose is "to help the pupil acquire basic competence." (user is **studying**)
> A how-to guide's purpose is "to help the already-competent user perform a particular task correctly." (user is **working**)

And the named failure mode:
> "the single most common conflation made in software product documentation is that between the
> tutorial and the how-to guide."

**[NOT VERIFIED]**: the two-axis derivation (practical/theoretical × study/work) — referenced on
the landing page but the defining text sits on deeper theory pages I did not fetch. The four modes
and the tutorial/how-to distinction above are verified.

**[I] Application to albireo.** The project already has tutorial-shaped material (the BLOeM
tutorial, the showcase notebook, the D42 example). The Diataxis risk for a package like this is
specific: a "tutorial" that is really a how-to (it assumes you already know what disentangling is
and which knobs matter) will lose exactly the newcomers it is meant to win. The BLOeM tutorial
episode is instructive — it caught a real docs defect (a claimed Balmer-free window containing
Hδ) *and* exposed an API gap (`Disentangler(velocities=...)` for a binary with no known period).
That is what a genuine tutorial does: it is written from the position of not knowing, so it finds
the things the author has stopped seeing.

### 7.2 Which astronomy packages actually follow the four-mode split

**[V, from the docs I fetched]** — a partial survey, since the search budget limited this:

| Package | Landing-page nav | Explicit Diataxis split? |
|---|---|---|
| **emcee** (<https://emcee.readthedocs.io/en/stable/>) | User Guide; Tutorials | Partial — tutorials separated, no how-to/explanation split |
| **lightkurve** (<https://lightkurve.github.io/lightkurve/>) | What's new; **Quickstart**; **Tutorials**; **API**; About; Developing | Close — quickstart/tutorials/reference, no explicit "how-to"/"explanation" |
| **NumPyro** (<https://num.pyro.ai/en/stable/>) | API & Developer Reference; Introductory Tutorials; Discrete Latent Variables; Applications; Other Inference Algorithms | Three-tier: Tutorials / Examples / API. Not Diataxis |
| **pyOpenSci packaging guide** (<https://www.pyopensci.org/python-package-guide/>) | Tutorials; Packaging; Documentation; Tests; Maintain | Tutorials separated; the guide discusses Sphinx/MyST but **does not explicitly recommend Diataxis** [V] |

| **astropy** (<https://docs.astropy.org/en/stable/>) | **Getting Started**; User Guide; **Learn Astropy** (a separate site, learn.astropy.org); Astropy Packages; Contributor's Guide; Project Details | **No** — it "does not explicitly follow the four-part Diataxis framework". Note it pushes tutorials *off-site* [V] |

| **JAX** (<https://docs.jax.dev/en/latest/>) | A *curriculum ladder*: JAX 101 (expressing computations), 201 (performance and scaling), 301 (advanced autodiff and extending JAX), 401 (kernels and FFI), 501 (systems topics), 601 (internals), plus API Reference | **No** — no Diataxis labels and **no "Quickstart" heading**; "Getting started" links into JAX 101 [V] |

**[NOT VERIFIED]**: PHOEBE 2 and exoplanet doc structures (the PHOEBE docs page fetched empty).

**[I] The honest conclusion: not one of the six sites surveyed adopts Diataxis by name.** That is
a real finding, not a gap in the survey — the framework is widely admired and, in this corner of
the ecosystem, not actually used as an organising vocabulary. Three shapes appear instead:

- **Quickstart → Tutorials → API** (lightkurve; astropy, which pushes tutorials off to
  learn.astropy.org) — the astronomy norm.
- **Tutorials / Examples / API** (NumPyro) — a three-tier split where "examples" carry the
  domain applications.
- **A curriculum ladder by reader expertise** (JAX 101 → 601) — organised by how deep the reader
  intends to go, not by what kind of document each page is.

**[I] For albireo the JAX ladder is worth a serious look**, because albireo's audience genuinely
stratifies that way: an astronomer who wants component spectra and an RV curve out of a stack of
FITS files; a spectroscopist who wants to control the continuum model, the masking and the
priors; and someone extending the likelihood or reading the marginalisation. Those are three
different depths of the same subject, which is exactly what the ladder encodes and what a flat
"Tutorials" folder does not. The Diataxis *distinction* (§7.1) remains worth internalising — it is
what stops a tutorial from silently becoming a how-to — but the four-mode *vocabulary* is not what
users or reviewers here expect to see in the nav. What they expect and notice is the **quickstart**
and the **executable tutorials**. Prioritise those.

### 7.3 "First cell does science" [V, partial]

| Package | Code on landing page? | Detail |
|---|---|---|
| **emcee** | **Yes — 8 lines** | Samples a 5-D Gaussian: imports, function, setup, run. `pip install` is *not* on the landing page (linked as "Installation"). Attribution section asks users to cite the paper, with arXiv/ADS/BibTeX links. [V] |
| **emcee README** | **No** | Badges only: tests, MIT, arXiv 1202.3665, Coveralls, RTD. **No PyPI or conda badge.** Prominent: "Please cite Foreman-Mackey, Hogg, Lang & Goodman (2012) if you find this code useful in your research", plus full BibTeX and a pointer to Goodman & Weare (2010). [V] |
| **lightkurve** | **No code** — an animated GIF captioned "Lightkurve example usage"; no install instructions and **no citation/DOI statement** on the landing page. [V] |
| **NumPyro** | **No** — navigation hub only, no runnable example, no install instructions. [V] |

**[I] The evidence does not support a single universal pattern, and that is itself the finding.**
Only emcee — the most-adopted of the four — puts a short runnable example on the landing page. The
useful generalisation is narrower than "first cell does science":

1. **A short, complete, copy-pasteable example that produces a result, on the first page a user
   reads.** emcee's is 8 lines. This is the pattern worth copying.
2. **Citation instructions placed prominently and repeated.** emcee does this in the README *and*
   the docs; lightkurve does it in neither place I checked. emcee is the more-cited package.
3. Install can be a link rather than inline — none of the four put `pip install` above the fold,
   so this is not the differentiator it is often claimed to be.

For albireo **[I]**: the first page should show a disentangling run on packaged data producing a
recognisable result in under ~10 lines, and it must use the *packaged* truth grid, not the
façade's grid — the project's own notes record that mismatch as the trap that catches people.

### 7.4 API stability statements

**[V]**: astropy's policy lives in its APE series — <https://github.com/astropy/astropy-APEs>
(fetched 2026-09-03) lists **APE 2 "Astropy Release Cycle and Version Numbering"**, with
APE 10, **APE 18 "Adopt NEP 29 for CPython and Numpy Version Support"** and **APE 21 "Ending Long
Term Support Releases"** covering adjacent policy. **APE 22** is the affiliated/pyOpenSci one.
(astropy's `development/api_change.html` page 404s; the APE list is the verified source.)

**numpy NEP 23 — "Backwards compatibility and deprecation policy"** [V],
<https://numpy.org/neps/nep-0023-backwards-compatibility.html> (fetched 2026-09-03). This is the
best template in scientific Python and is short enough to copy the shape of.

Two categories of incompatible change: **deprecate-and-remove** (warn, then remove) and
**behaviour changes** (same call, different result — signalled with `FutureWarning`).

The deprecation period, verbatim:
> "shall be done after at least 2 releases assuming the current 6-monthly release cycle; if that
> changes, there shall be at least 1 year between deprecation and removal."

A deprecation must [V]:
1. state "the version number of the release in which the functionality was deprecated";
2. use `DeprecationWarning` by default, `VisibleDeprecationWarning` "for changes that need
   attention again after already having been deprecated";
3. "set a `stacklevel`, so the warning appears to come from the correct place";
4. be "mentioned in the documentation for the functionality. A `.. deprecated::` directive can be
   used for this";
5. be "listed in the release notes of the release where the deprecation is first present";
6. "not be introduced in micro (bug fix) releases".

The four governing principles, verbatim:
> 1. "Changes need to benefit more than they harm users."
> 2. "NumPy is widely used, so breaking changes should be assumed by default to be harmful."
> 3. "Decisions should be based on how they affect users and downstream packages and should be
>    based on usage data where possible."
> 4. "The possibility of an incorrect result is worse than an error or even crash."

Bug fixes are exempt: "Fixes for clear bugs are exempt from this backwards compatibility policy."

**[I] Principle 4 is the one worth adopting verbatim for albireo**, because it describes the
project's own recorded experience: the `compare="matched"` default became `native` after AI Phe
showed it over-counted χ² by 1/Σk², with vsini absorbing the error. That was a silently-wrong
result corrected at the cost of a behaviour change — exactly what principle 4 licenses. Writing
the policy down *in advance* turns that class of change from an embarrassment into a documented,
principled action, and a package that expects more such corrections (a pre-1.0 inference code
validated against new real datasets) benefits more than most from having said so first.

**[NOT VERIFIED]**: specific 0.x astronomy packages' stability wording — the search budget ran out.

**[I] What a pre-alpha 0.x package should actually write, and where.** Three properties, based on
the JOSS/pyOpenSci criteria in §2–§3 rather than on a surveyed template:

- **Say what is stable, not that nothing is.** A blanket "everything may change" is read as
  "unfinished" (and collides with JOSS's "no half-baked solutions"). Name the surface you will
  hold — e.g. the top-level façade and the CLI — and name what is explicitly internal.
- **State a deprecation promise with a number in it.** "Public API changes are announced in the
  changelog and deprecated for at least one minor release before removal" is enough, and is the
  kind of "stated support or governance expectation" JOSS's single-author clause asks for by name.
- **Put it in a dedicated docs page**, linked from the README and the docs landing page — not
  buried in a README paragraph. Reviewers look for it as an artefact.

**[I]** Concretely for albireo: the façade is already the reviewed, adversarially-tested surface
(38 defects found, all in the translation layer) and the internals are explicitly not. That is a
stability policy already, waiting to be written down.

### 7.5 The 0.1.0 announcement

**[NOT VERIFIED]** as a surveyed convention — the venue research in §6.5 was cut short.
What is verified and relevant: RNAAS is ADS-indexed and citable with a days-long turnaround
(§6.5); the Massive Star Newsletter takes community submissions monthly (§6.5); OpenAstronomy
Discourse is Astropy's designated user forum (§6.5); Zenodo mints a DOI from a tagged GitHub
Release (§3.6).

**[V] One surveyed convention that is unambiguous, from §5.5:** the announcement matters far less
than the *artefact being announced*. Every adopted package in the survey was pip-installable at or
before its announcement; three unadopted ones announced a paper and packaged years later or never.
So the ordering below puts PyPI and the tag before any announcement channel.

**[I] A defensible plan, ordered, with the DR4 date as the anchor:**

| When | Action |
|---|---|
| **Today** | **Enable GitHub Pages + `gh variable set PAGES_ENABLED --body true`** so the docs actually exist (§4A.1a). Set the repo `homepageUrl`. Enable GitHub Discussions. **Register `albireo` on PyPI** (§1.1). |
| **This week** | Add `CODE_OF_CONDUCT.md`; fill the ORCID/affiliation TODOs. Enable the Zenodo integration **then** tag **v0.1.0** — in that order, or the tag is not archived (§3.6). |
| **By 30 Sep 2026** | Submit an EAS 2027 special-session proposal (§6.2). |
| **21–25 Sep 2026** | PoWR code course, Potsdam — free, and the right room (§6.4). |
| **Sep–Nov 2026** | Sustained visible development: a release cadence (target 7–18 tags in year one, §5.5 Finding 2), public issues, self-merged PRs. Write the stability page (§7.4) and the quickstart (§7.3). Add `to_specutils`/`from_specutils` (§4.10). |
| **2 Dec 2026** | Gaia DR4. Have a DR4 reader and a worked DR4 example ready to publish within days. |
| **Dec 2026** | 0.1.0 announcement: GitHub Release with a real changelog → Massive Star Newsletter entry → OpenAstronomy Discourse post → Bluesky/Astrodon. Optionally an astro-ph.IM preprint of the methods paper. |
| **Dec 2026 – Feb 2027** | Rewrite `paper/paper.md` to the six-section 2026 template (§2.8). Pursue an institutional or dependency home (§5.6 item 2) — the single highest-value action in this report. |
| **Feb 2027 onward** | pyOpenSci submission (§3.3) → Astropy affiliated + JOSS fast track. |
| **After the paper** | ASCL entry (§3.4); update `CITATION.cff` `preferred-citation`. |

A good GitHub release note **[I]**: what changed since the last tag grouped as
added/changed/fixed/deprecated; the install line; a link to the DOI; and one sentence on what the
release makes newly possible scientifically. Not a commit log.

---

## 8. Open questions — what this run could not verify

Listed so they are not mistaken for absences of fact.

1. **IAU 2027 Symposia/Focus Meeting list** — the announcement page returns HTTP 403 to automated
   fetch. Needs a manual look; a binaries or massive-stars Focus Meeting at the Rome GA (10–19 Aug
   2027) would be a prime venue.
2. **Cool Stars 22/23** dates and splinter-session process; dedicated binary-star meetings
   2026–27; Gaia DR4 workshops and the Gaia Sprint status. Start at
   <https://massivestars.org/conferences/>.
3. **EAS 2026** location/dates and the **EAS 2027 abstract deadline**.
4. **Current JOSS throughput** — the 45.5-day mean / 32-day median figures are from 2017 data.
5. **JOSS rejection-reason statistics for astronomy** — no dated source found; §2.10 is inference.
6. **Whether the pyOpenSci→JOSS fast track waives JOSS's six-month rule** — not addressed by any
   page fetched. Assume not.
7. **ADS's current Zenodo-harvesting behaviour** — the ADS data FAQ fetched empty.
8. **Whether GitHub–Zenodo is now a GitHub App vs the legacy webhook**, and primary-source wording
   for concept vs version DOI.
9. **RNAAS publication charge.**
10. **AAS Nova submission model** (about page 404) — inference in §6.5 is that it is editor-selected.
11. **Be Star Newsletter / Binary Star Newsletter** status.
12. **astro-ph.IM software-listing mechanics** and a dated read on astronomy's 2026 social platform.
13. **Doc-structure survey** for PHOEBE 2 and exoplanet (their sites are JS SPAs returning no
    server-side content). astropy, JAX, NumPyro, lightkurve, emcee and pyOpenSci are done in §7.2;
    numpy NEP 23 is verified in §7.4.
14. `ascl.net` served a TLS chain WebFetch could not verify on 2026-09-03; ASCL facts here come
    from ADS, Wikipedia and the ASCL's own arXiv article, not from ascl.net directly.
15. **All citation counts in §5.2 are OpenAlex/Semantic Scholar, not ADS** — `ui.adsabs.harvard.edu`
    returns HTTP 405 to automated fetching and the ADS API needs a token. ADS is normally higher.
    Conference proceedings are badly undercounted (FDBinary shows 11, certainly wrong).
16. **PyPI monthly downloads for lightkurve, juliet and batman-package** were rate-limited (HTTP
    429) on some attempts; the lightkurve and specutils figures quoted came through on a separate
    call and are believed good, but juliet's did not.
17. **Astropy affiliated package list** — `astropy.org/affiliated` renders its table in JavaScript
    and the page says the listing is minimal pending APE 22. The count of 8 accepted packages comes
    from a 2026 secondary source, not from the table itself.
18. **Whether albireo's repo was public from creation (2026-08-11) or from 2026-08-14** — GitHub's
    `createdAt` does not distinguish. Worth confirming; it moves the JOSS gate by three days.

### Two things worth re-checking before acting

- **§6.4's meeting list is a snapshot of one feed** (massivestars.org/category/events/). Given Gaia
  DR4 on 2 Dec 2026, DR4-focused workshops are very likely to be announced in the coming weeks and
  are not yet in it. This deserves a standing watch, not a one-off search.
- **§6.5's read on astronomy social media in 2026 is the weakest inference in the report.** The
  platform data retrieved was about general user counts, not astronomers specifically. Verify
  before choosing where to announce.
