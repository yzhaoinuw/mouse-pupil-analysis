"""End-to-end pipeline coverage against the packaged checkpoint.

These tests exercise the wiring that unit tests cannot: video decoding, frame
extraction, the real UNet forward pass, and output assembly. They deliberately use
a synthetic video, so they verify plumbing rather than segmentation accuracy. A
real-image regression test is tracked in treaty_docs/next_steps.md and needs
sample data that is not yet redistributable.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from mouse_pupil_analysis import analyze_frames, analyze_video
from mouse_pupil_analysis.results import DIAMETER_COLUMNS, VELOCITY_COLUMNS

FRAME_COUNT = 6
FRAME_WIDTH = 200
FRAME_HEIGHT = 160
VIDEO_FPS = 10.0


def _synthetic_frame(index: int) -> np.ndarray:
    """Draw a dark disc that drifts, standing in for a moving pupil."""
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH), 180, dtype=np.uint8)
    center = (FRAME_WIDTH // 2 + index * 2, FRAME_HEIGHT // 2)
    cv2.circle(frame, center, 18, 20, thickness=-1)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    video_path = tmp_path / "synthetic_eye.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        VIDEO_FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    if not writer.isOpened():
        pytest.skip("No MJPG video writer available in this OpenCV build.")
    for index in range(FRAME_COUNT):
        writer.write(_synthetic_frame(index))
    writer.release()
    return video_path


def test_video_analysis_writes_diameter_outputs(synthetic_video: Path, tmp_path: Path):
    result = analyze_video(
        synthetic_video,
        out_dir=tmp_path / "frames",
        result_dir=tmp_path / "result",
        extraction_fps=VIDEO_FPS,
        num_workers=0,
    )

    assert result.csv_path.is_file()
    assert result.plot_path.is_file()
    assert result.segmentation_dataframe is not None
    assert result.tracking_dataframe is None
    assert result.prediction_threshold == 0.5
    assert result.analysis_table.columns.tolist() == DIAMETER_COLUMNS
    assert len(result.analysis_table) == FRAME_COUNT
    assert result.analysis_table["estimated_pupil_diameter"].notna().all()
    assert (result.analysis_table["estimated_pupil_diameter"] >= 0).all()
    assert set(result.analysis_table["segmentation_status"]) <= {"valid", "warning", "invalid"}

    extracted = sorted((tmp_path / "frames").glob("*.png"))
    assert [path.name for path in extracted] == [
        f"synthetic_eye_{index + 1:05d}.png" for index in range(FRAME_COUNT)
    ]


def test_velocity_mode_adds_tracking_columns_and_overlays(synthetic_video: Path, tmp_path: Path):
    result = analyze_video(
        synthetic_video,
        out_dir=tmp_path / "frames",
        result_dir=tmp_path / "result",
        output_mask_dir=tmp_path / "overlays",
        calculate_velocity=True,
        acquisition_fps=VIDEO_FPS,
        num_workers=0,
    )

    assert result.analysis_table.columns.tolist() == VELOCITY_COLUMNS
    assert len(result.analysis_table) == FRAME_COUNT
    assert result.tracking_dataframe is not None

    timestamps = result.analysis_table["timestamp_seconds"].to_numpy(dtype=float)
    np.testing.assert_allclose(timestamps, np.arange(FRAME_COUNT) / VIDEO_FPS)

    assert set(result.analysis_table["tracking_status"]) <= {"valid", "warning", "invalid"}
    # Speed is undefined for the first frame; nothing precedes it.
    assert result.analysis_table.loc[0, "speed_pixels_per_second"] in ("", None) or np.isnan(
        float(result.analysis_table.loc[0, "speed_pixels_per_second"])
    )

    overlays = sorted((tmp_path / "overlays").glob("*.png"))
    assert len(overlays) == FRAME_COUNT


def test_image_directory_analysis_matches_video_analysis(synthetic_video: Path, tmp_path: Path):
    from_video = analyze_video(
        synthetic_video,
        out_dir=tmp_path / "frames",
        result_dir=tmp_path / "result_video",
        extraction_fps=VIDEO_FPS,
        num_workers=0,
    )
    from_frames = analyze_frames(
        tmp_path / "frames",
        result_dir=tmp_path / "result_frames",
        num_workers=0,
    )

    np.testing.assert_allclose(
        from_frames.analysis_table["estimated_pupil_diameter"].to_numpy(dtype=float),
        from_video.analysis_table["estimated_pupil_diameter"].to_numpy(dtype=float),
    )
