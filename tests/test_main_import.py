"""``albireo.__main__`` runs the command line only when executed, not when imported.

Tools that import every module of a package (pydoc, pkgutil walks, documentation
builders) would otherwise parse their own ``sys.argv`` as albireo arguments and exit.
"""

from __future__ import annotations

import importlib
import runpy
import sys

import pytest


def test_importing_the_main_module_does_not_run_the_command_line(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pydoc", "--no-such-option"])
    monkeypatch.delitem(sys.modules, "albireo.__main__", raising=False)
    module = importlib.import_module("albireo.__main__")
    assert callable(module.main)


def test_python_dash_m_still_runs_the_command_line(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["albireo", "--help"])
    monkeypatch.delitem(sys.modules, "albireo.__main__", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("albireo", run_name="__main__")
    assert exit_info.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()
