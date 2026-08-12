# ruff: noqa: F401,F403,I001
"""Deprecated compatibility wrapper for :mod:`mouse_pupil_analysis.extract_frames`."""

from mouse_pupil_analysis import extract_frames as _implementation
from mouse_pupil_analysis.extract_frames import *  # noqa: F401,F403


if __name__ == "__main__":  # pragma: no cover - compatibility execution path
    _implementation.main()
