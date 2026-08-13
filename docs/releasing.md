# Releasing

Maintainer notes. Nothing here is needed to *use* albireo.

## The first public release

albireo's history is currently local — nothing has ever been pushed, so CI has never run and
no clock has started. Two of those clocks matter:

- **JOSS** requires at least six months of *public* development history, and rejects
  submissions whose commits are all concentrated shortly before the paper. The six months
  begins at the first push, not at the first commit.
- **PyPI and astropy affiliation** both require the package to be installable before they
  can be applied for at all.

So the ordering below front-loads going public, and leaves the paper to follow. Each step is
listed separately because each is independently reversible-or-not: pushing is easy to
correct, a published PyPI version number can never be reused.

### 1. Before pushing anything

- [ ] Fill in the ORCID and affiliation TODOs in `CITATION.cff` and `paper/paper.md`.
- [ ] Decide whether the first tag is `0.1.0` or stays `0.1.0.dev0`. A dev version can be
      uploaded to PyPI but is not installed by a bare `pip install albireo`, which makes it
      a good rehearsal and a bad launch.
- [ ] `pytest` — the whole suite, gates included.
- [ ] `ruff check . && ruff format --check .`
- [ ] `mkdocs build --strict`
- [ ] Re-check that the name `albireo` is still free on PyPI. It was verified free on
      2026-08-11 (D18) and the recorded fallback is `albireo-spectra`.

### 2. Push, and let CI run for the first time

- [ ] `git push -u origin main`
- [ ] Watch the four workflows: `lint`, `bare-install`, `test` (4-way matrix), `gates`, and
      `Docs`. **Expect something to fail** — CI has never executed, so this is the first
      real evidence any of it works. In particular the Windows legs and the bare-install
      job have never run anywhere but this machine.
- [ ] In the repository settings, set **Pages** to deploy from GitHub Actions, so the
      `Docs` workflow's deploy job has somewhere to publish.
- [ ] Enable **Discussions**.

### 3. Zenodo, *before* the first tag

- [ ] Link the repository at [zenodo.org/account/settings/github](https://zenodo.org/account/settings/github)
      and switch albireo on.

The ordering is the point: Zenodo archives releases created *after* the webhook exists. Tag
first and the first release is not archived, so the DOI story starts a release late.

### 4. Tag and publish

- [ ] Set up trusted publishing at
      [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/) for
      the project `albireo`, workflow `release.yml`, environment `pypi`.
- [ ] Rehearse: run the `Release` workflow manually with **Publish to TestPyPI** checked,
      and install the result into a scratch environment.
- [ ] Update `CHANGELOG.md` — move `Unreleased` to the version and date.
- [ ] Bump `__version__` in `src/albireo/__init__.py` and the `version:` in `CITATION.cff`
      to match. `tests/test_metadata.py` fails if they disagree, and the release workflow
      refuses to publish if the tag disagrees with either.
- [ ] `git tag v0.1.0 && git push --tags`.
- [ ] Confirm the wheel installs from PyPI in a clean environment and that
      `ab.load_example()` works — that is the quickstart's first line, and it depends on the
      packaged `.npz` having made it into the wheel.

### 5. Make it citable

- [ ] Copy the Zenodo DOI into `CITATION.cff` (`doi:` and `date-released:`) and into
      `docs/citing.md`, then commit.
- [ ] Add the DOI badge to `README.md`.
- [ ] Submit to the [ASCL](https://ascl.net/submissions). `codemeta.json` in the repository
      root prefills most of the form. ASCL entries are indexed by ADS, which is where
      astronomers actually look.

### 6. Then, and only then, tell people

Announce once, when there is something to install and something to cite. The
[roadmap](roadmap.md) lists the beachhead communities and what each one needs; a general
announcement to everyone at once is weaker than a specific one to the people whose problem
albireo already solves.

## Subsequent releases

1. `CHANGELOG.md`: move `Unreleased` into the new version.
2. Bump `__version__` and `CITATION.cff` together.
3. Tag `vX.Y.Z` and push the tag; the workflow does the rest.
4. Zenodo mints a new DOI automatically. The concept DOI keeps pointing at the newest
   version, so `docs/citing.md` only needs editing when the citation itself changes — for
   example when the methods paper appears and becomes the thing to cite.

While the version is below 1.0 the API may break in any release. Say so in the changelog
entry when it does, and say what to change.
