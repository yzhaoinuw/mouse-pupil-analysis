"""Automated mouse pupil segmentation, diameter, and pupil-center velocity analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mouse-pupil-analysis")
except PackageNotFoundError:  # pragma: no cover - source checkout without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
