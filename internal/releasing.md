# Releasing

Maintainer notes. Nothing here is needed in order to use albireo.

## The first public release

albireo's history is currently local: nothing has ever been pushed, so CI has never run and
no clock has started. Two of those clocks matter:

- **JOSS** requires at least six months of public development history, and rejects
  submissions whose commits are all concentrated shortly before the paper. The six months
  begins at the first push, not at the first commit.
- **PyPI and astropy affiliation** both require the package to be installable before they
  can be applied for at all.

The ordering below therefore front-loads going public and leaves the paper to follow. Each
step is listed separately because each differs in reversibility: pushing is easy to correct,
while a published PyPI version number can never be reused.

### 1. Before pushing anything

- [ ] Fill in the ORCID and affiliation TODOs in `CITATION.cff` and `paper/paper.md`.
- [ ] Decide whether the first tag is `0.1.0` or stays `0.1.0.dev0`. A dev version can be
      uploaded to PyPI but is not installed by a bare `pip install albireo`, which makes it
      a good rehearsal and a poor launch.
- [ ] `pytest`, the whole suite, gates included. This is the local stand-in for the `Full`
      workflow, and it matters because `release.yml` builds and publishes without running
      any tests: nothing between a tag and PyPI checks the science.
- [ ] `ruff check . && ruff format --check .`
- [ ] `mkdocs build --strict`
- [ ] Re-check that the name `albireo` is still free on PyPI. It was verified free on
      2026-08-11 (D18) and the recorded fallback is `albireo-spectra`.

### 2. Push, and let CI run for the first time

- [ ] `git push -u origin main`
- [ ] Watch `CI` (one job: lint, the bare-install guards, the fast suite) and `Docs`.
      Expect something to fail, since CI has never executed and this is the first
      evidence any of it works.
- [ ] Then run the `Full` workflow by hand from the Actions tab. It carries everything
      `CI` leaves out, namely the OS/Python matrix, the slow acceptance gates with coverage,
      and the example scripts, none of which has run anywhere but this machine. The Windows
      legs are the least-evidenced part, since local runs are Windows and CI's routine job is
      Linux.
- [ ] In the repository settings, set **Pages** to deploy from GitHub Actions, so the
      `Docs` workflow's deploy job has somewhere to publish. Note that Pages from a
      private repository needs GitHub Pro; on the free tier the deploy job only works
      once the repository is public.
- [ ] Then switch the deploy job on: `gh variable set PAGES_ENABLED --body true`. It is
      gated behind that variable because `deploy-pages` cannot be made to
      succeed before the two settings above exist (it calls the Pages API and 404s), and
      a workflow that is red for a known reason stops being read. The `Docs` build job
      runs from the first push regardless, so a broken docs build is still caught; only the
      publish step waits.
- [ ] Enable **Discussions**.

!!! note "Going public is also the Actions-minutes fix"
    Actions minutes are free and unlimited on public repositories and metered on private
    ones, at 2,000/month on the free tier, with Windows billing at 2x and macOS at 10x. That
    is why `CI` is one Linux job and everything expensive is manual (`full.yml`). While the
    repository is private, budget roughly 25-30 billed minutes per push (the fast suite
    measures 12-16 minutes locally, and the runner is slower per core) and 150+ per manual
    `Full` run, where the Windows legs bill at 2x. Once it is public, neither number is
    charged against anything, which is the strongest practical argument for not staying
    private long.

### 3. Zenodo, *before* the first tag

- [ ] Link the repository at [zenodo.org/account/settings/github](https://zenodo.org/account/settings/github)
      and switch albireo on.

The ordering matters: Zenodo archives releases created after the webhook exists. Tag first
and the first release is not archived, so the DOI record starts a release late.

### 4. Tag and publish

- [ ] Set up trusted publishing at
      [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/) for
      the project `albireo`, workflow `release.yml`, environment `pypi`.
- [ ] Set up a second publisher at
      [test.pypi.org/manage/account/publishing](https://test.pypi.org/manage/account/publishing/),
      same project and workflow but environment `testpypi`. Trusted publishing is per-index
      and the OIDC claim PyPI validates includes the environment name, so the pypi.org
      publisher above does not authorize the rehearsal below; without this the rehearsal
      fails at the upload with an OIDC error, which reads like a broken workflow rather
      than a missing registration.
- [ ] Rehearse: run the `Release` workflow manually with **Publish to TestPyPI** checked,
      and install the result into a scratch environment.
- [ ] Update `CHANGELOG.md`: move `Unreleased` to the version and date.
- [ ] Bump `__version__` in `src/albireo/__init__.py` and the `version:` in `CITATION.cff`
      to match. `tests/test_metadata.py` fails if they disagree, and the release workflow
      refuses to publish if the tag disagrees with either.
- [ ] `git tag v0.1.0 && git push --tags`.
- [ ] Confirm the wheel installs from PyPI in a clean environment and that
      `ab.load_example()` works. That is the quickstart's first line, and it depends on the
      packaged `.npz` having made it into the wheel.

### 5. Make it citable

- [ ] Copy the Zenodo DOI into `CITATION.cff` (`doi:` and `date-released:`) and into
      `docs/citing.md`, then commit.
- [ ] Add the DOI badge to `README.md`.
- [ ] Submit to the [ASCL](https://ascl.net/submissions). `codemeta.json` in the repository
      root prefills most of the form. ASCL entries are indexed by ADS, which is where
      astronomers look.

### 6. Then, and only then, tell people

Announce once, when there is something to install and something to cite. The
[roadmap](roadmap.md) lists the target communities and what each one needs; a general
announcement to everyone at once is weaker than a specific one to the people whose problem
albireo already solves.

## Subsequent releases

1. `CHANGELOG.md`: move `Unreleased` into the new version.
2. Bump `__version__` and `CITATION.cff` together.
3. Tag `vX.Y.Z` and push the tag; the workflow does the rest.
4. Zenodo mints a new DOI automatically. The concept DOI keeps pointing at the newest
   version, so `docs/citing.md` only needs editing when the citation itself changes, for
   example when the methods paper appears and becomes the reference to cite.

While the version is below 1.0 the API may break in any release. Say so in the changelog
entry when it does, and say what to change.