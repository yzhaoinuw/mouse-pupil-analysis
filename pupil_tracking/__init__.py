"""Deprecated compatibility package for :mod:`mouse_pupil_analysis`.

The public project name changed before the first PyPI release.  Existing local
installs and scripts may still import ``pupil_tracking``, so this package forwards
the public API without changing the long-standing console commands.
"""

from __future__ import annotations

import warnings

import mouse_pupil_analysis as _implementation

warnings.warn(
    "pupil_tracking is deprecated; import mouse_pupil_analysis instead.",
    DeprecationWarning,
    stacklevel=2,
)

__version__ = _implementation.__version__
_EXPORTS = _implementation._EXPORTS
__all__ = _implementation.__all__


def __getattr__(name: str):
    """Forward lazily exported public names to the renamed package."""
    return getattr(_implementation, name)


def __dir__():
    return sorted(__all__)
