# F — Fresh install: an astronomer's first result with albireo

Date of run: 2026-09-03. Machine: Windows 11 Pro, 32-thread desktop, CPU only, no GPU.
Interpreter: Anaconda CPython 3.13.9. Installer: `uv` (warm package cache).
All work in a throwaway venv under `…/scratchpad/fresh`; the source repository was not
modified (`git status --porcelain` empty before and after the editable install).

**Other Python jobs run on this machine, so all wall times below are indicative.**

---

## 1. Timings

| Step | Wall time | Notes |
|---|---:|---|
| `uv venv` (Anaconda 3.13.9) | 0.16 s | |
| `uv pip install -e "…albireo[io,plots]"` | **5.06 s** | warm `uv` cache; see cold-cache note |
| — cold download volume implied | — | 107.7 MiB fetched this run (jaxlib 65.4, scipy 34.9, jax 3.1, fonttools 2.4, astropy-iers-data 1.9 MiB) |
| Resulting environment | — | **37 packages, 522.9 MB, 9,070 files** |
| `import albireo` — first ever, cold `.pyc` | **4.71 s** | |
| `import albireo` — warm | **0.77 s** | |
| `ab.load_example("sb2_sim")` | 0.00 s | |
| `ab.Disentangler(...)` construction | 0.00 s | derivations are lazy |
| **`dis.fit()`** | **93.32 s** | matches docs' "about a minute"; most of it JAX compilation |
| **`fit.sample(seed=0)`** | **685.58 s (11.4 min)** | 500 warmup + 500 samples × 2 chains, 0 divergences. Notebook records 601 s for the same call. |
| `ab.plot_spectra(...)` | 3.11 s | |
| **README quickstart, end to end** | **783.07 s (13.1 min)** | |
| `dis.fit()` on a hand-built Dataset (step 5) | 96.62 s | whole probe script 100 s |
| `albireo init` | 1.59 s | |
| **`albireo demo --out demo`** | **151 s** | pipeline self-reports 146.6 s; docs say "a minute or two" |

**Time to a first scientific result.** If the user runs the README verbatim: ~13 minutes
of total silence before anything prints. If the user runs `docs/quickstart.md` (which
prints `dataset.summary()` and `fit.summary()` along the way) the first real result — the
orbit, to 0.05% — arrives at **~95 s**. If the user runs `albireo demo`, a complete
annotated result set with truth comparison arrives at **~2.5 min**. The fastest path to a
trustworthy first result is the CLI, which is not what the README leads with.

### Accuracy achieved (all three paths)

Injected truth: *P* = 6.0 d, *e* = 0.15, *K* = (42.0, 63.0) km/s.

| Path | K₁ | K₂ | error |
|---|---|---|---|
| README quickstart, NUTS posterior | 41.977 ± 0.054 | 62.980 ± 0.092 | 0.055%, 0.032% |
| Hand-built Dataset, MAP | 41.978 | 62.979 | 0.052%, 0.033% |
| `albireo demo`, orbit from TODCOR table | 41.980 ± 0.046 | 62.987 ± 0.072 | 0.048%, 0.020% |

The package does what it claims, on the first try, on all three paths. Nothing below
disputes that.

---

## 2. Warnings, verbatim

**There were none.** Every run captured stdout and stderr to separate files:

| Run | stderr bytes |
|---|---:|
| README quickstart (`import` → `fit` → `sample` → plot) | **0** |
| `albireo demo` | **0** |
| Hand-built-Dataset probe script (13 deliberate mistakes + a fit) | **0** |

No JAX platform notice, no NumPy/SciPy deprecation, no `astropy` IERS warning, no
`matplotlib` backend warning. `pip`/`uv` printed no warnings during install either. This
is unusually clean for a JAX + astropy stack and deserves to be recorded as a strength.

The only non-zero exit codes seen were the deliberate ones: `albireo --version` (exit 2,
see friction #7) and the intentional error probes.

### The one printed result a new user must interpret

```
NUTS posterior: 1000 draws over 7 sites
  divergences: 0

  period    6.000105 +/- 0.001108
  t_conj    0.625322 +/- 0.002299
  K(primary)    41.977 +/- 0.054 km/s   95% [41.867, 42.077]
  K(secondary)    62.980 +/- 0.092 km/s   95% [62.804, 63.157]

  Smoothness was fixed at its ML-II values, so these intervals do not include
  smoothness uncertainty (a plug-in approximation, not a marginalization).

Assumed, not measured:
  light fractions    primary=0.62  secondary=0.38
      only l_i * d_i is observable, so every recovered depth scales as 1/l_i.
  smoothness starts  primary=300  secondary=300
      ML-II fits these, but which basin it finds depends on where tau starts.
```

**Is this interpretable by an astronomer who has not read the docs?** Partly.

*Reads well:* the `K(name)` rows are exemplary — value, σ, unit, and a labelled 95%
interval. The "Assumed, not measured" block is better than anything the incumbent codes
print, and the plug-in-approximation caveat is honest and in plain words.

*Does not read well:*
- **"7 sites"** is numpyro jargon. Only 4 quantities are then shown.
- **`period` and `t_conj` carry no units.** `fit.summary()` writes `period 6.000167 d`;
  `post.summary()` drops the `d`. `t_conj` is never expanded to "time of conjunction",
  and its zero point (the dataset's BJD origin) is not stated.
- **Eccentricity and ω are missing entirely** although both were sampled. `fit.summary()`
  reports them; the posterior summary does not. An astronomer's headline table needs *e*.
- **No R̂ and no ESS**, although two chains were run. Divergences alone are not a
  convergence check, and the reader cannot tell whether 1000 draws were enough.
- **The truth is never shown beside the recovered K.** The README's own snippet binds
  `truth` and then never uses it, while the surrounding prose claims recovery "to better
  than 0.1% of the injected values" — a claim the script does not demonstrate. Contrast
  the pipeline's `summary.txt`, which does it properly:
  `K from the velocity table: primary 41.980+-0.046 (truth 42, -0.020)`.
- The unexplained shift between `fit.summary()`'s `conjunction scan: t_conj = 0.73171` and
  the posterior's `t_conj 0.625322` (the scan is a coarse 41-node grid; the truth is
  0.629 d, so the posterior is right at 1.6σ) will read as an inconsistency to anyone who
  does not know the scan is only a starting point.

---

## 3. Command line

`albireo --help` and `albireo demo --help` are clear and correctly scoped. `albireo init`
writes a genuinely well-annotated `albireo.toml` (3.3 KB, every required-and-undefaulted
value flagged with the reason).

`albireo demo` is **the best first-run experience in the package**: per-stage progress
lines with elapsed times, explicit flags when a stage is skipped or a result is
differential, and a truth comparison at the end. 151 s, no warnings, exit 0. It writes
1.3 MB across 37 files, 17 per star.

The per-star `summary.txt` is the strongest artefact in the whole project. It states the
velocity budget term by term, the derived grid, the sampled sites, the assumptions, the
ML-II table, the TODCOR diagnostics, the Keplerian with `q` and *M* sin³ *i*, an explicit
"Against the injected truth" block, the flags, and a manifest of every file written.

Figures: `velocities.png` is textbook (units on both axes, O−C panel, legend);
`spectra.png` is clear with a `±2σ` band and the truth overplotted.

### What a first-time user would not understand or would distrust

- **`spectra.png` flares wildly at both ends.** The model grid (4499–4561 Å) is wider than
  the data (4505–4555 Å) by the velocity-budget margin, so the band explodes where there
  are no data. Nothing on the figure marks those regions as unconstrained. This looks like
  a failed fit at a glance. (Same on the README's figure.)
- **`results.csv` has no units in any header** — `period`, `K_1`, `gamma_1`, `teff_1`,
  `vsini_1` — and prints full float64 repr: `41.979861501631916`, `0.04604423543357488`.
  `q` and `absolute` are never expanded. The `flags` column holds a long semicolon-joined
  English sentence inside quotes, which is awkward in a spreadsheet.
- **The demo's closing truth block is a raw Python dict** with 17 significant digits:
  `{'k_disentangling': {'primary': -0.02157255370973843, …}}`. The information is right;
  the presentation is a debug dump.
- **The batch `summary.txt` is three lines** and says "2 flag(s)" without showing them.
- **`summary.txt` prints the "Assumed, not measured" block twice**, verbatim, in the same
  file. A careful reader will diff the two blocks looking for a difference that is not
  there.
- **`phase_scan.png`'s legend reads `t_conj = 0.7317`** (days) on an axis labelled "trial
  conjunction phase (fraction of one period)", with the marker drawn at *x* = 0.122. The
  number and the axis are in different units and nothing says so.
- **`velocities.csv` has no header comments and no units**, whereas the sibling
  `velocities.rv` has an excellent commented header (frame, zero point per component,
  light-fraction mode). The two files disagree on how much context they carry.
- **`albireo init`'s template turns `[labels]` on by default**, whose `library =
  "bosz2024-fgk-r20000"` is annotated `downloads ~645 MB once`. The comment is there, but
  the default of the generated file is a 645 MB download.
- `demo --help` says `--out OUT  output directory (overrides the file)`; for `demo` there
  is no file. (Shared help text with `run`.)

---

## 4. Documentation

### The site does not exist

Every documentation URL 404s.

```
https://tjayasinghe.github.io/albireo/            404
https://tjayasinghe.github.io/albireo/quickstart/ 404
https://tjayasinghe.github.io/                    404
https://github.com/tjayasinghe/albireo            200
```

Cause, confirmed against the GitHub API: `has_pages: false`, no `gh-pages` branch, and the
repository has **no Actions variables at all**, so `docs.yml`'s deploy job — gated on
`vars.PAGES_ENABLED == 'true'` — has never run. The `build` job runs and passes on every
push, so **the README's Docs badge is green while the site it links to is absent**. Last
Docs run: success, 2026-09-03T20:25:12Z. This is `internal/releasing.md` step 2, never
performed after the repository went public.

Everything below therefore evaluates the sources as GitHub renders them, which is the only
documentation surface a user actually has today, plus what the site *would* show.

### Install instructions: the site and the README contradict each other

`albireo` is **not on PyPI** (`https://pypi.org/pypi/albireo/json` → 404).

| Location | Says |
|---|---|
| `docs/quickstart.md:7` — the first command on the first page | `pip install albireo` |
| `docs/tutorials/pipeline.md:11` | `pip install "albireo[io,plots]"` |
| `docs/tutorials/showcase.ipynb` cell 0 | `pip install "albireo[plots]"` |
| `README.md:100` (Command line section) | `pip install "albireo[io,plots]"` |
| `docs/api/io.md`, `docs/api/results.md`, `tutorials/real-data.md`, `tutorials/bloem-sb2.md` | extras phrased as `pip install "albireo[io]"` |
| **`README.md:114` (Installation section)** | **"albireo is not yet on PyPI. Install from a clone."** |

So the README contradicts itself twelve lines apart, and the quickstart page opens with a
command that fails with "No matching distribution found" — which reads to a new user as a
broken package rather than an unreleased one.

Related: `[project.urls]` carries only `Repository` and `Issues` — no `Documentation` or
`Homepage`, so the docs link would be missing from the PyPI sidebar too. The GitHub repo
has no homepage set, no topics, and the one-line description "Spectral disentangling".

### Page-by-page

| Page | Verdict |
|---|---|
| `docs/index.md` | Clear, well-oriented "Where to start". No install section at all, which is why the reader meets `pip install albireo` on the next page with no caveat. |
| `docs/quickstart.md` | Excellent apart from line 7. Shows the expected printed output for every step, states the injected truth in prose, explains all four derived quantities, and closes with two conventions (no default light fraction; *e* = 0 is a singular point) that are exactly the two things a user would otherwise get wrong. Strictly better than the README quickstart. |
| `docs/science.md` | The strongest page. Sober methods-section prose, every method attributed (Bagnuolo & Gies 1991, Simon & Sturm 1994, Hadrava 1995, Ilijić et al. 2004, Pavlovski & Hensberge 2010 …), ADS links promised in the reference list. |
| `docs/api/index.md` | Very good. A stage-ordered module table and a Conventions block (units, masking by `ivar == 0`, deviation spectra, `-inf` guards). |
| `docs/api/facade.md` | Very good; the "Required declarations" and "Outside the scope of v1" sections state refusals and their reasons. One inaccuracy — see friction #8. |
| `docs/tutorials/sb2-end-to-end.md` | Very good, with an explicit Runtime admonition and a frank discussion of `converged=False`. Every block is quoted from `examples/01_sb2_end_to_end.py` and it tells you to run `python examples/01_sb2_end_to_end.py` — which requires a clone (`[tool.hatch.build.targets.wheel] packages = ["src/albireo"]`, so `examples/` is not in the wheel). |
| `docs/tutorials/showcase.ipynb` | Outputs are committed: 10/10 code cells carry outputs, 6 embedded PNGs, 382 KB. `mkdocs-jupyter` is configured `execute: false`, so the page would render fully and offline. It records `fit.sample` at 601.4 s, so the ten-minute cost *is* documented — just not where a first-time reader meets it. |
| `docs/citing.md` | Honest and complete: a pre-release warning, a BibTeX entry, the AAS `\software{}` macro, and a dependency citation table. |

### Rendering

No rendering problems found, and the math handling is unusually careful. `docs/math.md`
uses 529 inline <code>$`x`$</code> expressions and 53 <code>```math</code> fences, with
**zero** bare `$…$` (which GitHub silently corrupts). `scripts/mkdocs_math_hook.py`
converts both forms back for `arithmatex`, so the same source renders correctly on GitHub
*and* on the site. Navigation is complete: all 23 `docs/api/*.md` and all 10
`docs/tutorials/*` files appear in `mkdocs.yml`; no orphans, no missing targets. CI runs
`mkdocs build --strict`, so internal links and docstring references are already verified.

Two notes on the site's `extra_javascript`: it loads MathJax and a polyfill from two CDNs
(`cdnjs.cloudflare.com`, `cdn.jsdelivr.net`), so math will not render offline or behind a
restrictive proxy.

---

## 5. The "my own data" path (numpy arrays, no FITS)

Built a 12-epoch `Dataset` from bare numpy arrays and fitted it. **It works, and it
recovers K to 0.05%.** The friction is entirely in discoverability, not behaviour.

### What a user must supply, and whether the docs say so

| Requirement | Where a user learns it | Verdict |
|---|---|---|
| `EpochData(wave, flux, ivar, bjd)` | README fragment; `EpochData` docstring | Only 4 args are required; `v_bary` defaults to 0.0 and `instrument` to `"default"`. Works positionally too. Not shown in any doc page. |
| `Dataset(epochs)` frame | `frame="topocentric"` default | Sensible default, documented in the module docstring. |
| **Light fraction** | Everywhere, repeatedly | Excellent. The single best-documented requirement in the package. |
| **LSF per instrument** | `lsf={...}`; `LSF.from_resolution(R)` | Error message names the fix. `LSF.from_resolution(20000)` → σ = 6.37 km/s. |
| **Medium (air/vacuum)** | Only enforced when a `Telluric`/`Nebular` is declared | Error is excellent; but see below. |
| **Grid** | Derived automatically by the façade | No user action needed. `dv_kms` defaults to the finest native sampling. |
| **Velocity budget** | Derived from the `k` and `ecc` priors | No user action needed; `explain()` prints the term-by-term derivation. |

### Error messages: outstanding

Of thirteen deliberate mistakes, eleven produced messages that name both the fix and the
science. Verbatim:

```
ValueError: the star light fractions must sum to 1; primary=0.6, secondary=0.5 sums to 1.1.
This is an assumption the data cannot check: with constant light fractions the likelihood
sees only l_i * d_i, so every recovered depth scales as 1/l_i.

ValueError: no LSF declared for instrument(s) ['MYSPEC']. The dataset has ['MYSPEC']; pass
one entry per instrument, e.g. lsf={'FEROS': ab.LSF.from_resolution(48_000)}.

ValueError: this dataset does not declare whether its wavelengths are air or vacuum, and a
Telluric or Nebular component is keyed to absolute line positions. The difference is a
nearly constant 83 km/s. Declare it with read_dataset(medium=...) or EpochData(medium=...).

ValueError: declare exactly one of orbit= (a Keplerian to fit) and velocities= (the
per-epoch velocities you measured, for a system whose orbit is not known yet). …

ValueError: velocity_budget_kms=10.0 is smaller than the 431.6 km/s the declared priors can
reach, so the solver bandwidth would not cover every configuration the prior allows.
Sampling stalls against that guard rather than failing.
  [followed by the full term-by-term budget table]
```

### Where the documentation leaves a user stuck

1. **No page shows the numpy-array route.** `tutorials/real-data.md` is FITS-only;
   `quickstart.md` uses the packaged example; `bloem-sb2.md` mentions
   `EpochData(..., medium="air")` only in an admonition. The only construction example is
   the README's one-liner containing a literal `...`.
2. **The README's low-level snippet does not run as written.** It passes `init=init_values`
   and `rng_key=key`; neither name is defined anywhere in the README.
3. **`Dataset` has no `medium` attribute** (public attrs: `bjd`, `epochs`, `frame`,
   `instruments`, `n_epochs`, `summary`, `v_bary`), and `ds.summary()` does not print it.
   The error above tells you to set `EpochData(medium=...)`, but nothing lets you confirm
   what an assembled `Dataset` actually has. The pipeline's `result.json` does report
   `/dataset/medium`, so the concept exists at `Dataset` level in reporting only.
4. **`Star("primary")` raises a bare `TypeError: Star.__init__() missing 1 required
   positional argument: 'light'`** — the only refusal in the set with no explanation, and
   it guards the policy the project emphasizes most.

---

## 6. `pip`-style discoverability

| Check | Result |
|---|---|
| `python -c "import albireo; print(albireo.__version__)"` | **works** → `0.1.0.dev0` |
| `albireo --version` | **fails**, exit 2: `albireo: error: the following arguments are required: command` |
| bare `albireo` | usage + exit 2 (no help) |
| `albireo --help`, `albireo demo --help` | clear and correct |
| `help(albireo.Disentangler)` | **reads well** — 343 lines, numpydoc, every parameter explained, a full `Raises` list, and an `Examples` block. Two blemishes: unrendered Sphinx roles (`:meth:`, `:class:`, `:func:`) appear as raw markup at a terminal, and it cites `docs/math.md` §7.6 by path, which a non-clone user does not have. |
| Top-level public names | 181; every name used in the README exists. |
| Console scripts installed | `albireo.exe` only (plus dependencies' own). |

---

## 7. Ranked friction list

Ranked by how much each one costs a new astronomer on day one.

| # | Friction | Suggested fix |
|---|---|---|
| **1** | **The documentation site 404s.** `has_pages: false`, no `PAGES_ENABLED` variable, so `docs.yml`'s deploy job has never run — while the `build` job passes, leaving the README's **Docs badge green and its link dead**. Every URL in the nav is unreachable. | Enable Pages with source "GitHub Actions", then `gh variable set PAGES_ENABLED --body true` (`internal/releasing.md` step 2). Until then point the badge at the repo's `docs/` folder. Add a step to `docs.yml` that emits a workflow **warning** on `main` when deploy is skipped, so a green badge can never again mean "no site". |
| **2** | **`pip install albireo` is the first command on the quickstart page, and the package is not on PyPI (404).** README §Installation says the opposite twelve lines below its own `pip install "albireo[io,plots]"`. Six locations imply PyPI. | Replace every `pip install albireo…` with the clone recipe (one shared snippet included everywhere), or reserve the name and upload `0.1.0.dev0`. Add `Documentation` and `Homepage` to `[project.urls]`, and set the GitHub homepage + topics. |
| **3** | **The README quickstart prints nothing for 13 minutes, then throws its figure away.** No progress bar (`Fit.sample(progress_bar=False)` is the default), no intermediate prints, and `ab.plot_spectra(...)` is neither shown nor saved, so a script user gets no figure at all. | Mirror `docs/quickstart.md`: print `dataset.summary()` and `fit.summary()`; state the two runtimes inline ("≈90 s" and "≈10 min"); pass `progress_bar=True` in the quickstart; end with `fig, _ = ab.plot_spectra(...)` then `fig.savefig("spectra.png")`. |
| **4** | **`post.summary()` omits *e*, ω, R̂ and ESS, gives no units on `period`/`t_conj`, and never shows the truth** — while the README claims 0.1% recovery and binds an unused `truth`. | Report *e* and ω, add `d` to `period` and `t_conj` (and name the BJD zero point), add R̂/ESS (two chains already run), and in `load_example` mode print an "Against the injected truth" block like the pipeline's `summary.txt` already does. |
| **5** | **The default fit reports `stopped at the step cap` with `\|grad\| 383`** (and `\|grad\| 9.11` in the docs' own quickstart output) while K is right to 0.05%. It reads as a failed fit. `quickstart.md` never comments on it. | Raise the façade's default `max_steps`, and/or append one line to the summary: "the orbital sites are converged; the residual gradient lies along the flat ML-II hyperparameter directions". Move the `sb2-end-to-end.md` explanation of `converged=False` into the quickstart. |
| **6** | **`repr(Dataset)` is 313,693 characters** (26,123 for one `EpochData`): typing `ds` in a REPL floods the terminal. `ds.summary()` is 195 characters and excellent. | Add `__repr__` to `EpochData` and `Dataset` returning a one-line form (`<Dataset 12 epochs, 1 instrument, 4505.0–4555.0 Å, topocentric>`), and point at `.summary()`. |
| **7** | **`albireo --version` does not exist** (exit 2), and bare `albireo` errors instead of printing help — the first two things anyone types after installing a CLI. | Add `parser.add_argument("--version", action="version", version=albireo.__version__)`; make a bare invocation print help and exit 0. |
| **8** | **The `velocity_budget_kms` guard is lazy, but documented as a constructor `Raises`.** `Disentangler(…, velocity_budget_kms=10.0)` returns an object; the (excellent) `ValueError` only fires when `.velocity_budget` is first touched. Both the class docstring and `docs/api/facade.md` place it under construction. | Validate the override in `__post_init__` where the derived terms are cheap to compute, or move it out of the constructor's `Raises` block and say explicitly when it fires. |
| **9** | **`Star("primary")` raises a bare `TypeError: … missing 1 required positional argument: 'light'`** — the one required-declaration error carrying no explanation, guarding the policy the project stresses most. | Keep `light` required but raise with the project's own voice, as the light-sum check already does; at minimum put the reason in the first line of `Star`'s docstring. |
| **10** | **No documented path from numpy arrays to a `Dataset`**, the README's low-level snippet uses two undefined names (`init_values`, `key`) and a literal `...`, and **`Dataset` exposes no `medium`** so the error that demands it cannot be verified afterwards. | Add a six-line "From arrays" section to `quickstart.md` (the path works perfectly — it only needs showing). Define the missing names in the README. Add a `Dataset.medium` property and print it in `summary()`. |

### Below the top ten, still worth fixing

11. `spectra.png` (both the façade's and the pipeline's) extends past the data into the
    grid margin, where the band explodes and looks like a divergence. Shade or annotate
    the no-data region, or default the x-limits to the data range.
12. The README's figure labels its panels `d_1`, `d_2` with no title; the pipeline's
    labels them `primary`/`secondary` with one. Pass `labels=` in the README snippet.
13. `results.csv`: no units in any header, 17-significant-digit floats, unexplained `q`
    and `absolute`, and a prose `flags` column. Add units to the header names
    (`period_d`, `K_1_kms`), round to a sensible precision, and emit flags as codes.
14. The demo's closing truth comparison is a raw Python dict; format it as the per-star
    `summary.txt` already formats the same numbers.
15. The per-star `summary.txt` prints the "Assumed, not measured" block twice.
16. `phase_scan.png`'s legend gives `t_conj` in days on an axis in phase units.
17. `velocities.csv` lacks the commented header its sibling `velocities.rv` has.
18. `albireo init`'s template enables `[labels]`, whose default library is a 645 MB
    download, in the file a first-time user is told to run next.
19. Tutorials instruct `python examples/01_sb2_end_to_end.py`, but `examples/` is not in
    the wheel; say "from a clone" or ship them as package data.
20. `help(Disentangler)` shows raw `:meth:`/`:class:` roles and cites `docs/math.md` by
    path, which a non-clone user does not have.

---

## 8. What is already excellent

Recorded so the ranked list above is not read as a verdict on the package.

- **Zero warnings** across install, import, fit, sample, plotting, the CLI, and a
  thirteen-mistake error probe.
- **Error messages** that name the fix *and* the science, with the velocity-budget error
  printing its whole derivation table.
- **The `albireo demo` command and its per-star `summary.txt`**, which is the model every
  other summary in the package should follow — truth comparison, units throughout, an
  explicit assumptions block, and a manifest of the files it wrote.
- **`docs/science.md` and `docs/math.md`**: properly referenced, carefully typeset, and
  written for astronomers rather than for developers.
- **The math syntax discipline** — 529 inline expressions and 53 display blocks in the one
  spelling that renders faithfully on both GitHub and MkDocs, with a hook and a test
  holding the convention.
- **Accuracy on the first try**, to 0.05% in K on all three paths, on a hand-built Dataset
  as well as the packaged one.
