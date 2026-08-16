"""Regression coverage using the committed real-pupil fixture.

`test_end_to_end.py` proves the pipeline is wired together, but a synthetic blob
segments plausibly no matter what weights are loaded. These tests run the packaged
checkpoint over real frames, so a corrupted checkpoint, a silently swapped model, or
a regression in `resize_with_pad` produces a failure rather than a plausible number.

Landmarks are asserted as ranges, not exact values. They are wide enough to tolerate
platform floating-point differences and narrow enough that a different model would
fall outside them.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mouse_pupil_analysis import analyze_frames
from mouse_pupil_analysis.preprocessing import MODEL_IMAGE_SIZE, resize_scale

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = PROJECT_ROOT / "sample_data"
LABELLED_ROOT = SAMPLE_ROOT / "labeled_data"
VELOCITY_ROOT = SAMPLE_ROOT / "velocity_frames"

# Two sessions whose frames are a single size well above the 148 model input, so the
# expected rescale back to input pixels is unambiguous.
SESSIONS = ["HQL080_sleep250625", "HQL090_sleep251012"]

VELOCITY_FPS = 97.0
VELOCITY_FRAMES = 31

pytestmark = pytest.mark.skipif(
    not SAMPLE_ROOT.is_dir(),
    reason="sample_data/ is only present in a source checkout.",
)


@pytest.mark.parametrize("session", SESSIONS)
def test_real_frames_segment_to_plausible_diameters(session, tmp_path):
    frames = LABELLED_ROOT / session / "images"
    result = analyze_frames(frames, result_dir=tmp_path / session, num_workers=0)
    table = result.analysis_table

    assert len(table) == len(list(frames.glob("*.png")))
    model_diameters = table["estimated_pupil_diameter"].to_numpy(dtype=float)
    assert np.all(np.isfinite(model_diameters))
    # A real pupil occupies a meaningful part of the model image. Near-zero would mean
    # the model found nothing; near 148 would mean it segmented the whole frame. These
    # frames sit close around the eye, so a dilated pupil legitimately fills most of the
    # width -- the pool's largest mask is 109.7 of 148.
    assert np.all(model_diameters > 5.0)
    assert np.all(model_diameters < MODEL_IMAGE_SIZE * 0.95)


@pytest.mark.parametrize("session", SESSIONS)
def test_video_pixel_diameter_undoes_the_downscale(session, tmp_path):
    frames = LABELLED_ROOT / session / "images"
    sizes = {Image.open(path).size for path in frames.glob("*.png")}
    assert len(sizes) == 1, f"{session} is not a single frame size: {sizes}"
    (expected_size,) = sizes

    result = analyze_frames(frames, result_dir=tmp_path / session, num_workers=0)
    table = result.analysis_table

    scale_x, scale_y, _, _ = resize_scale(*expected_size)
    expected_ratio = 1.0 / np.sqrt(scale_x * scale_y)
    observed = table["pupil_diameter_input_pixels"].to_numpy(dtype=float) / table[
        "estimated_pupil_diameter"
    ].to_numpy(dtype=float)

    # These frames are larger than the model image, so input-pixel diameters must be
    # strictly larger, by exactly the geometric mean of the two axis scales.
    assert expected_ratio > 1.0
    np.testing.assert_allclose(observed, expected_ratio, rtol=1e-9)


def test_already_model_sized_frames_need_no_diameter_conversion(tmp_path):
    result = analyze_frames(
        VELOCITY_ROOT,
        result_dir=tmp_path / "velocity",
        calculate_velocity=True,
        acquisition_fps=VELOCITY_FPS,
        num_workers=0,
    )
    table = result.analysis_table

    # These frames are already 148 x 148, so the two diameter columns must agree.
    np.testing.assert_allclose(
        table["pupil_diameter_input_pixels"].to_numpy(dtype=float),
        table["estimated_pupil_diameter"].to_numpy(dtype=float),
        rtol=1e-9,
    )


def test_velocity_fixture_reproduces_documented_landmarks(tmp_path):
    result = analyze_frames(
        VELOCITY_ROOT,
        result_dir=tmp_path / "velocity",
        calculate_velocity=True,
        acquisition_fps=VELOCITY_FPS,
        num_workers=0,
    )
    table = result.analysis_table

    assert len(table) == VELOCITY_FRAMES

    # Every frame is usable; three carry a temporal-area warning. Documented in
    # sample_data/README.md.
    counts = table["tracking_status"].value_counts().to_dict()
    assert counts.get("invalid", 0) == 0
    assert counts.get("warning", 0) == 3
    assert counts.get("valid", 0) == VELOCITY_FRAMES - 3
    assert set(table.loc[table["quality_reason"].ne(""), "quality_reason"]) == {
        "abrupt_area_change"
    }

    diameters = table["estimated_pupil_diameter"].to_numpy(dtype=float)
    assert diameters.min() == pytest.approx(19.54, abs=0.15)
    assert diameters.max() == pytest.approx(26.34, abs=0.15)

    # Frames are consecutive, so every adjacent pair yields a speed and no gap is bridged.
    speeds = pd.to_numeric(table["speed_pixels_per_second"], errors="coerce")
    assert speeds.notna().sum() == VELOCITY_FRAMES - 1
    assert speeds.isna().iloc[0]

    timestamps = table["timestamp_seconds"].to_numpy(dtype=float)
    np.testing.assert_allclose(np.diff(timestamps), 1.0 / VELOCITY_FPS, rtol=1e-9)
