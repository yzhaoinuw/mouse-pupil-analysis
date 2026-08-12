"""Deprecated compatibility wrapper for :mod:`mouse_pupil_analysis.dataset`."""

from mouse_pupil_analysis import dataset as _implementation

# Forwarding by attribute rather than ``import *`` so that names the renamed
# module leaves out of ``__all__`` still resolve through the legacy path.
__all__ = getattr(_implementation, "__all__", None) or [
    _name for _name in dir(_implementation) if not _name.startswith("_")
]


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__():
    return dir(_implementation)
