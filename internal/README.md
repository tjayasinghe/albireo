# Internal notes

Working notes addressed to maintainers. Nothing here is part of the published
documentation at <https://tjayasinghe.github.io/albireo/>, and nothing here is needed in
order to use albireo: the user-facing pages are in [`docs/`](../docs).

- [`design.md`](design.md): architecture, data model, and the decision record (D1, D2, ...).
  The source and the tests cite it by decision number, so it is kept in the repository.
- [`roadmap.md`](roadmap.md): the planned development of albireo, its order, and the
  non-goals. It is a plan and makes no commitment.
- [`releasing.md`](releasing.md): the release procedure and its checklists.

These files are excluded from `mkdocs.yml`. Adding one to the site navigation changes what
the project presents to its users, so it should be a deliberate decision.
