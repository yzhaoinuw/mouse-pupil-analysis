"""Automated mouse pupil segmentation, diameter, and pupil-center velocity analysis.

Typical use::

    from pupil_tracking import analyze_video

    result = analyze_video("mouse1.avi", calculate_velocity=True, acquisition_fps=33.3333)
    print(result.analysis_table.head())

The public names are imported lazily, so ``import pupil_tracking`` stays cheap and
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
    "AnalysisConfig": "pupil_tracking.api",
    "AnalysisResult": "pupil_tracking.api",
    "analyze_frames": "pupil_tracking.api",
    "analyze_video": "pupil_tracking.api",
    "run_analysis": "pupil_tracking.api",
    "ExtractedFrame": "pupil_tracking.extract_frames",
    "extract_selected_frames": "pupil_tracking.extract_frames",
    "PupilPrediction": "pupil_tracking.pupil_predictions",
    "find_default_checkpoint": "pupil_tracking.pupil_predictions",
    "iter_pupil_predictions": "pupil_tracking.pupil_predictions",
    "build_tracking_dataframe": "pupil_tracking.tracking",
    "measure_probability_map": "pupil_tracking.tracking",
}

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from pupil_tracking.api import (
        AnalysisConfig,
        AnalysisResult,
        analyze_frames,
        analyze_video,
        run_analysis,
    )
    from pupil_tracking.extract_frames import ExtractedFrame, extract_selected_frames
    from pupil_tracking.pupil_predictions import (
        PupilPrediction,
        find_default_checkpoint,
        iter_pupil_predictions,
    )
    from pupil_tracking.tracking import build_tracking_dataframe, measure_probability_map

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
