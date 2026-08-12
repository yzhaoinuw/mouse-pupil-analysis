# ruff: noqa: F401,F403,I001
"""Deprecated compatibility wrapper for :mod:`mouse_pupil_analysis.run_pupil_analysis`."""

from mouse_pupil_analysis import run_pupil_analysis as _implementation
from mouse_pupil_analysis.run_pupil_analysis import *  # noqa: F401,F403


if __name__ == "__main__":  # pragma: no cover - compatibility execution path
    _implementation.main()
