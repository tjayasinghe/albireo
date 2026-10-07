# Releasing

These are maintainer notes. Nothing here is needed to use albireo.

## The first public release

The repository became public on 2026-08-14. Since then `main` and `origin/main` have been
in sync, and CI and Docs have run on every push. Of the two requirements below, the first
is therefore under way and the second is not yet met:

- **JOSS** requires at least six months of public development history, and rejects
  submissions whose commits are all concentrated shortly before the paper. The six months
  begins at the first push, not at the first commit, which puts the earliest submission at
  2027-02-14. Elapsed time alone is not sufficient: JOSS takes releases, issues and pull
  requests as evidence of public development, and there are none yet. Waiting does not
  produce them.
- **PyPI and astropy affiliation** both require the package to be installable before they
  can be applied for. albireo is not on PyPI, so every install command in the documentation
  is a clone install and is described as one.

What remains is therefore the release itself. The steps are listed separately because they
differ in reversibility: a push is easy to correct, while a published PyPI version number
can never be reused. Steps already taken are marked with their date.


### 1. Before pushing anything

- [ ] Fill in the ORCID and affiliation TODOs in `CITATION.cff` and `paper/paper.md`.
- [ ] Decide whether the first tag is `0.1.0` or stays `0.1.0.dev0`. A dev version can be
      uploaded to PyPI but is not installed by a bare `pip install albireo`, which suits a
      rehearsal and not a launch.
- [ ] `pytest`, the whole suite, acceptance tests included. This is the local substitute
      for the `Full` workflow. It is needed because `release.yml` builds and publishes
      without running any tests, so nothing between a tag and PyPI checks the scientific
      results.
- [ ] `ruff check . && ruff format --check .`
- [ ] `mkdocs build --strict`
- [ ] Re-check that the name `albireo` is still free on PyPI. It was verified free on
      2026-08-11 (D18) and again on 2026-09-03; the recorded fallback is `albireo-spectra`,
      also free. PyPI has no reservation mechanism, so the only way to hold the name is to
      upload something under it.

### 2. Push, and let CI run (done 2026-08-14)

- [x] `git push -u origin main`. Done 2026-08-14; the first workflow run was at 2026-08-15
      01:53 UTC.
- [x] Watch `CI` (one job: lint, the bare-install guards, the fast suite) and `Docs`. Both
      have run on every push since and both pass.
- [ ] Then run the `Full` workflow manually from the Actions tab. It runs everything that
      `CI` omits, namely the OS/Python matrix, the slow acceptance tests with coverage, and
      the example scripts, none of which has run anywhere but the development machine. The
      Windows jobs are the least-tested part, since local runs are on Windows and CI's
      routine job is on Linux.
- [x] Set Pages to deploy from GitHub Actions, so the `Docs` workflow's deploy job has
      somewhere to publish. Done 2026-09-03 with
      `gh api -X POST repos/tjayasinghe/albireo/pages -f build_type=workflow`, which is the
      API equivalent of the settings page. Pages from a private repository needs GitHub
      Pro; on the free tier the deploy job only works once the repository is public, which
      it has been since 2026-08-14.
- [x] Then enable the deploy job: `gh variable set PAGES_ENABLED --body true`. Done
      2026-09-03. The job is conditional on that variable because `deploy-pages` cannot
      succeed before the two settings above exist (it calls the Pages API and receives
      a 404), and a workflow that fails for a known reason is no longer read. The `Docs`
      build job runs from the first push regardless, so a broken docs build is still
      detected. Only the publish step waits.

      Both settings had to be made together, and a passing badge does not show whether
      either is set. From 2026-08-14 to 2026-09-03 `Docs` passed on every push while
      <https://tjayasinghe.github.io/albireo/> returned 404, because the build job was the
      only one running and the README's badge links to the site rather than to the run. The
      deploy job runs on the next push to `main`, and the badge and the link agree only
      after that run succeeds.
- [ ] Enable Discussions. Not yet enabled.

!!! note "Actions minutes are no longer a constraint"
    Actions minutes are free and unlimited on public repositories and metered on private
    ones, at 2,000/month on the free tier, with Windows billed at 2x and macOS at 10x. That
    is why `CI` is one Linux job and everything expensive is manual (`full.yml`). While the
    repository was private, a push cost roughly 25-30 billed minutes (the fast suite takes
    12-16 minutes locally, and the runner is slower per core) and a manual `Full` run cost
    150+, with the Windows jobs billed at 2x. Since 2026-08-14 none of that is charged, so
    the layout of the workflows is now a choice about feedback time rather than about cost.

### 3. Zenodo, before the first tag

- [ ] Link the repository at [zenodo.org/account/settings/github](https://zenodo.org/account/settings/github)
      and switch albireo on.

The order matters: Zenodo archives releases created after the webhook exists. If the tag
comes first, the first release is not archived and the DOI record starts a release late.

### 4. Tag and publish

- [ ] Set up trusted publishing at
      [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/) for
      the project `albireo`, workflow `release.yml`, environment `pypi`.
- [ ] Set up a second publisher at
      [test.pypi.org/manage/account/publishing](https://test.pypi.org/manage/account/publishing/),
      with the same project and workflow but environment `testpypi`. Trusted publishing is
      per-index, and the OIDC claim PyPI validates includes the environment name, so the
      pypi.org publisher above does not authorize the rehearsal below. Without the second
      publisher the rehearsal fails at the upload with an OIDC error, which suggests a
      broken workflow rather than a missing registration.
- [ ] Rehearse: run the `Release` workflow manually with "Publish to TestPyPI" checked, and
      install the result into a scratch environment.
- [ ] Switch the documentation's install commands from the editable clone form back to
      `pip install "albireo[...]"`. `grep -rn 'pip install -e' README.md docs/ scripts/`
      finds every one. The three-line clone block of `docs/quickstart.md` becomes one line,
      and the README's "not yet on PyPI" paragraph and its note about the editable form are
      removed.
- [ ] Update `CHANGELOG.md`: move `Unreleased` to the version and date.
- [ ] Bump `__version__` in `src/albireo/__init__.py` and the `version:` in `CITATION.cff`
      to match. `tests/test_metadata.py` fails if they disagree, and the release workflow
      does not publish if the tag disagrees with either.
- [ ] `git tag v0.1.0 && git push --tags`.
- [ ] Confirm the wheel installs from PyPI in a clean environment and that
      `ab.load_example()` works. That is the quickstart's first line, and it depends on the
      packaged `.npz` being included in the wheel.

### 5. Make it citable

- [ ] Copy the Zenodo DOI into `CITATION.cff` (`doi:` and `date-released:`) and into
      `docs/citing.md`, then commit.
- [ ] Add the DOI badge to `README.md`.
- [ ] Submit to the [ASCL](https://ascl.net/submissions). `codemeta.json` in the repository
      root prefills most of the form. ASCL entries are indexed by ADS, which astronomers
      search.

### 6. Announce the release

Announce once, when there is something to install and something to cite. The
[roadmap](roadmap.md) lists the target communities and what each one needs. A general
announcement to everyone at once is less effective than a specific one to the people whose
problem albireo already solves.

## Subsequent releases

1. `CHANGELOG.md`: move `Unreleased` into the new version.
2. Bump `__version__` and `CITATION.cff` together.
3. Tag `vX.Y.Z` and push the tag; the workflow does the rest.
4. Zenodo mints a new DOI automatically. The concept DOI keeps pointing at the newest
   version, so `docs/citing.md` only needs editing when the citation itself changes, for
   example when the methods paper appears and becomes the reference to cite.

While the version is below 1.0 the API may break in any release. Say so in the changelog
entry when it does, and say what to change.
