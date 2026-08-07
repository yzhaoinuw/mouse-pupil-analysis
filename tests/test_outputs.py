from pathlib import Path

import numpy as np
import pandas as pd

from pupil_tracking.extract_frames import ExtractedFrame
from pupil_tracking.run_pupil_analysis import (
    _frames_from_image_directory,
    save_analysis_results,
)


def _frames(tmp_path: Path) -> list[ExtractedFrame]:
    return [
        ExtractedFrame(tmp_path / f"eye_{index + 1:05d}.png", index, index) for index in range(3)
    ]


def _results() -> list[tuple[str, float]]:
    return [
        ("eye_00001.png", 10.0),
        ("eye_00002.png", 11.0),
        ("eye_00003.png", 12.0),
    ]


def test_image_directory_uses_one_based_filename_as_source_frame(tmp_path: Path):
    (tmp_path / "eye_00001.png").touch()
    (tmp_path / "eye_00020.png").touch()

    frames = _frames_from_image_directory(tmp_path)

    assert [frame.source_frame_index for frame in frames] == [0, 19]


def test_diameter_only_writes_one_compact_analysis_output(tmp_path: Path):
    csv_path, plot_path = save_analysis_results(
        _results(),
        _frames(tmp_path),
        tmp_path,
        "eye",
    )

    dataframe = pd.read_csv(csv_path)
    assert dataframe.columns.tolist() == ["image_name", "estimated_pupil_diameter"]
    assert dataframe["image_name"].tolist() == [
        "eye_00001.png",
        "eye_00002.png",
        "eye_00003.png",
    ]
    assert plot_path.name == "eye_pupil_analysis.png"
    assert plot_path.is_file()


def test_velocity_mode_appends_compact_tracking_fields_and_removes_legacy_outputs(
    tmp_path: Path,
):
    tracking = pd.DataFrame(
        {
            "image_name": ["eye_00001.png", "eye_00002.png", "eye_00003.png"],
            "source_frame_index": [0, 1, 2],
            "timestamp_seconds": [0.0, 0.03, 0.06],
            "center_x_pixels": [100.0, 101.0, np.nan],
            "center_y_pixels": [80.0, 81.0, np.nan],
            "speed_pixels_per_second": [np.nan, 47.0, np.nan],
            "estimated_pupil_diameter": [10.0, 11.0, 12.0],
            "segmentation_valid": [True, True, False],
            "quality_reason": ["", "low_component_dominance", "abrupt_area_change"],
        }
    )
    for suffix in (
        "estimated_pupil_diameter.csv",
        "estimated_pupil_diameter.png",
        "pupil_tracking.csv",
        "pupil_tracking_qc.png",
    ):
        (tmp_path / f"eye_{suffix}").touch()

    csv_path, plot_path = save_analysis_results(
        _results(),
        _frames(tmp_path),
        tmp_path,
        "eye",
        tracking_dataframe=tracking,
    )

    dataframe = pd.read_csv(csv_path, keep_default_na=False)
    assert dataframe.columns.tolist() == [
        "image_name",
        "estimated_pupil_diameter",
        "timestamp_seconds",
        "center_x_pixels",
        "center_y_pixels",
        "speed_pixels_per_second",
        "tracking_status",
        "quality_reason",
    ]
    assert dataframe["tracking_status"].tolist() == ["valid", "warning", "invalid"]
    assert dataframe.loc[2, "center_x_pixels"] == ""
    assert plot_path.name == "eye_pupil_analysis.png"
    assert plot_path.is_file()
    for suffix in (
        "estimated_pupil_diameter.csv",
        "estimated_pupil_diameter.png",
        "pupil_tracking.csv",
        "pupil_tracking_qc.png",
    ):
        assert not (tmp_path / f"eye_{suffix}").exists()
