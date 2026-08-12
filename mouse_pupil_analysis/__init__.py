"""Automated mouse pupil segmentation, diameter, and pupil-center velocity analysis.

Typical use::

    from mouse_pupil_analysis import analyze_video

    result = analyze_video("mouse1.avi", calculate_velocity=True, acquisition_fps=33.3333)
    print(result.analysis_table.head())

The public names are imported lazily, so ``import mouse_pupil_analysis`` stays cheap and
does not pull in PyTorch until an analysis actually runs.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("mouse-pupil-analysis")
except PackageNotFoundError:  # pragma: no cover - source checkout without an install
    __version__ = "0.0.0+unknown"

# Public name -> module that defines it.
_EXPORTS = {
    "AnalysisConfig": "mouse_pupil_analysis.api",
    "AnalysisResult": "mouse_pupil_analysis.api",
    "analyze_frames": "mouse_pupil_analysis.api",
    "analyze_video": "mouse_pupil_analysis.api",
    "run_analysis": "mouse_pupil_analysis.api",
    "ExtractedFrame": "mouse_pupil_analysis.extract_frames",
    "extract_selected_frames": "mouse_pupil_analysis.extract_frames",
    "PupilPrediction": "mouse_pupil_analysis.pupil_predictions",
    "find_default_checkpoint": "mouse_pupil_analysis.pupil_predictions",
    "iter_pupil_predictions": "mouse_pupil_analysis.pupil_predictions",
    "build_tracking_dataframe": "mouse_pupil_analysis.tracking",
    "measure_probability_map": "mouse_pupil_analysis.tracking",
}

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from mouse_pupil_analysis.api import (
        AnalysisConfig,
        AnalysisResult,
        analyze_frames,
        analyze_video,
        run_analysis,
    )
    from mouse_pupil_analysis.extract_frames import ExtractedFrame, extract_selected_frames
    from mouse_pupil_analysis.pupil_predictions import (
        PupilPrediction,
        find_default_checkpoint,
        iter_pupil_predictions,
    )
    from mouse_pupil_analysis.tracking import build_tracking_dataframe, measure_probability_map

# Kept literal so static analysis can see it; tests assert it matches _EXPORTS.
__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ExtractedFrame",
    "PupilPrediction",
    "__version__",
    "analyze_frames",
    "analyze_video",
    "build_tracking_dataframe",
    "extract_selected_frames",
    "find_default_checkpoint",
    "iter_pupil_predictions",
    "measure_probability_map",
    "run_analysis",
]


def __getattr__(name: str):
    """Resolve a public name to its defining module on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)


def __dir__():
    return sorted(__all__)
