"""Tests for the demo-GIF utility.

``media/make_gif.py`` is a maintainer script rather than part of the installed
package, so it is loaded from the repository by path. The load happens inside a
fixture so the suite still collects when the script is absent, such as when tests
are run against an installed wheel.
"""

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKE_GIF_PATH = PROJECT_ROOT / "media" / "make_gif.py"


@pytest.fixture(scope="module")
def make_gif():
    if not MAKE_GIF_PATH.is_file():
        pytest.skip("media/make_gif.py is only available in a source checkout.")
    return run_path(str(MAKE_GIF_PATH))


def test_media_defaults_follow_repository_layout(make_gif):
    assert make_gif["PROJECT_ROOT"] == PROJECT_ROOT
    assert make_gif["DEFAULT_RUN_DIR"] == PROJECT_ROOT / "media" / "readme_demo"
    assert make_gif["DEFAULT_CSV"] == (
        PROJECT_ROOT / "media" / "readme_demo" / "pupil_analysis_for_gif.csv"
    )
    assert make_gif["DEFAULT_OVERLAY_DIR"] == (PROJECT_ROOT / "media" / "readme_demo" / "overlays")
    assert make_gif["DEFAULT_OUTPUT"] == (
        PROJECT_ROOT / "media" / "pupil_diameter_analysis_result_demo.gif"
    )
    assert make_gif["DEFAULT_START_FRAME"] == 7107
    assert make_gif["DEFAULT_END_FRAME"] == 7375
    assert make_gif["DEFAULT_SAMPLE_EVERY"] == 1
    assert make_gif["DEFAULT_FPS"] == 5.0


def test_diagnostic_segment_keeps_rejected_run_and_neighboring_endpoints(make_gif):
    values = np.arange(7, dtype=float)
    rejected = np.array([False, False, True, True, False, False, False])

    result = make_gif["diagnostic_segment"](values, rejected)

    np.testing.assert_allclose(
        result,
        np.array([np.nan, 1.0, 2.0, 3.0, 4.0, np.nan, np.nan]),
        equal_nan=True,
    )
