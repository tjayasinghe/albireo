# Internal notes

Working notes addressed to maintainers. Nothing here is part of the published
documentation at <https://tjayasinghe.github.io/albireo/>, and nothing here is needed in
order to use albireo: the user-facing pages are in [`docs/`](../docs).

- [`design.md`](design.md): architecture, data model, and the decision ledger (D1, D2, ...).
  The source and the tests cite it by decision number, which is why it lives in the
  repository rather than outside it.
- [`roadmap.md`](roadmap.md): what albireo intends to become, in what order, and the
  non-goals. A plan, not a promise.
- [`releasing.md`](releasing.md): the release procedure and the checklists around it.

These files are excluded from `mkdocs.yml`. Adding one back to the site nav is a decision
about what the project shows its users, not a formatting fix, so make it deliberately.
