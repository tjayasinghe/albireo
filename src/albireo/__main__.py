"""``python -m albireo`` runs the same command as the ``albireo`` console script."""

from albireo.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
