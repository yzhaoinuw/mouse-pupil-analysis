from pathlib import Path
from runpy import run_path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKE_GIF_PATH = PROJECT_ROOT / "media" / "make_gif.py"
MAKE_GIF_MODULE = run_path(str(MAKE_GIF_PATH))
diagnostic_segment = MAKE_GIF_MODULE["diagnostic_segment"]


def test_media_defaults_follow_repository_layout():
    assert MAKE_GIF_MODULE["PROJECT_ROOT"] == PROJECT_ROOT
    assert MAKE_GIF_MODULE["DEFAULT_OUTPUT"] == (
        PROJECT_ROOT / "media" / "pupil_diameter_analysis_result_demo.gif"
    )


def test_diagnostic_segment_keeps_rejected_run_and_neighboring_endpoints():
    values = np.arange(7, dtype=float)
    rejected = np.array([False, False, True, True, False, False, False])

    result = diagnostic_segment(values, rejected)

    np.testing.assert_allclose(
        result,
        np.array([np.nan, 1.0, 2.0, 3.0, 4.0, np.nan, np.nan]),
        equal_nan=True,
    )
