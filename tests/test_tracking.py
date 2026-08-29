from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mouse_pupil_analysis.extract_frames import ExtractedFrame
from mouse_pupil_analysis.pupil_predictions import PupilPrediction
from mouse_pupil_analysis.tracking import (
    SegmentationAccumulator,
    TrackingAccumulator,
    build_tracking_dataframe,
    measure_probability_map,
    model_to_original_coordinates,
    pupil_visibility,
)


def _measurement(source_frame_index, center_x, center_y, *, valid=True, area=100):
    return {
        "image_name": f"frame_{source_frame_index + 1:05d}.png",
        "source_frame_index": source_frame_index,
        "raw_center_x_pixels": center_x,
        "raw_center_y_pixels": center_y,
        "selected_component_area": area,
        "segmentation_valid": valid,
        "quality_reason": "" if valid else "test_invalid",
    }


def test_model_to_original_coordinates_reverses_landscape_resize_and_pad():
    center_x, center_y = model_to_original_coordinates(
        center_x=74,
        center_y=74,
        original_width=296,
        original_height=148,
    )

    assert center_x == 148
    assert center_y == 74


def test_model_to_original_coordinates_reverses_portrait_resize_and_pad():
    center_x, center_y = model_to_original_coordinates(
        center_x=74,
        center_y=74,
        original_width=100,
        original_height=200,
    )

    assert center_x == 50
    assert center_y == 100


def test_measure_probability_map_uses_probability_weighted_component_center():
    probability_map = np.zeros((10, 10), dtype=np.float32)
    probability_map[4:6, 2] = 0.9
    probability_map[4:6, 3] = 1.0

    measurement, binary_mask, selected_mask = measure_probability_map(
        probability_map,
        pred_thresh=0.7,
        original_size=(10, 10),
    )

    assert binary_mask.sum() == 4
    assert selected_mask.sum() == 4
    assert measurement["component_count"] == 1
    assert measurement["raw_center_x_pixels"] > 2.5
    assert measurement["raw_center_y_pixels"] == pytest.approx(4.5)
    assert measurement["segmentation_valid"]


def test_measure_probability_map_warns_on_low_component_dominance():
    probability_map = np.zeros((20, 20), dtype=np.float32)
    probability_map[2:5, 2:5] = 0.99
    probability_map[12:15, 12:14] = 0.99

    measurement, _, _ = measure_probability_map(
        probability_map,
        pred_thresh=0.7,
        original_size=(20, 20),
    )

    assert measurement["component_count"] == 2
    assert measurement["component_dominance"] == 0.6
    assert measurement["segmentation_valid"]
    assert "low_component_dominance" in measurement["quality_reason"]


def test_tracking_accumulator_reuses_streamed_binary_mask(tmp_path: Path):
    image_path = tmp_path / "eye_00001.png"
    Image.fromarray(np.zeros((148, 148), dtype=np.uint8)).save(image_path)
    frame = ExtractedFrame(image_path, source_frame_index=0, extraction_index=0)

    probability_map = np.zeros((148, 148), dtype=np.float32)
    probability_map[70:79, 70:79] = 0.95
    binary_mask = np.zeros((148, 148), dtype=bool)
    binary_mask[70:79, 70:79] = True
    prediction = PupilPrediction(
        frame=frame,
        probability_map=probability_map,
        binary_mask=binary_mask,
        estimated_pupil_diameter=10.0,
        original_size=(148, 148),
    )

    accumulator = TrackingAccumulator(pred_thresh=0.99, acquisition_fps=10.0)
    accumulator.add(prediction)
    dataframe = accumulator.build_dataframe()

    assert dataframe.loc[0, "selected_component_area"] == 81
    assert dataframe.loc[0, "segmentation_valid"]
    assert dataframe.loc[0, "image_name"] == image_path.name


def test_segmentation_accumulator_reports_not_detected_without_velocity(tmp_path: Path):
    image_path = tmp_path / "eye_00001.png"
    Image.fromarray(np.zeros((148, 148), dtype=np.uint8)).save(image_path)
    prediction = PupilPrediction(
        frame=ExtractedFrame(image_path, source_frame_index=0, extraction_index=0),
        probability_map=np.zeros((148, 148), dtype=np.float32),
        binary_mask=np.zeros((148, 148), dtype=bool),
        estimated_pupil_diameter=0.0,
        original_size=(148, 148),
    )

    accumulator = SegmentationAccumulator(pred_thresh=0.7)
    accumulator.add(prediction)
    dataframe = accumulator.build_dataframe()

    assert dataframe.loc[0, "pupil_visibility"] == "not_detected"
    assert not dataframe.loc[0, "segmentation_valid"]
    assert dataframe.loc[0, "quality_reason"] == "empty_mask"


def test_low_circularity_component_is_marked_partially_visible_or_uncertain():
    probability_map = np.zeros((40, 40), dtype=np.float32)
    probability_map[20:22, 8:32] = 0.99

    measurement, _, _ = measure_probability_map(
        probability_map,
        pred_thresh=0.7,
        original_size=(40, 40),
    )

    assert not measurement["segmentation_valid"]
    assert "low_component_circularity" in measurement["quality_reason"]
    assert pupil_visibility(measurement) == "partially_visible_or_uncertain"


def test_build_tracking_dataframe_uses_actual_elapsed_time():
    dataframe = build_tracking_dataframe(
        [
            _measurement(0, 0, 0),
            _measurement(1, 3, 4),
            _measurement(2, 3, 4),
        ],
        acquisition_fps=10,
    )

    assert dataframe.loc[1, "timestamp_seconds"] == 0.1
    assert dataframe.loc[1, "displacement_x_pixels"] == 3
    assert dataframe.loc[1, "displacement_y_pixels"] == 4
    assert dataframe.loc[1, "velocity_x_pixels_per_second"] == 30
    assert dataframe.loc[1, "velocity_y_pixels_per_second"] == 40
    assert dataframe.loc[1, "speed_pixels_per_second"] == 50


def test_build_tracking_dataframe_does_not_bridge_invalid_or_missing_frames():
    invalid_dataframe = build_tracking_dataframe(
        [
            _measurement(0, 0, 0),
            _measurement(1, 3, 4, valid=False),
            _measurement(2, 6, 8),
        ],
        acquisition_fps=10,
    )
    missing_dataframe = build_tracking_dataframe(
        [
            _measurement(0, 0, 0),
            _measurement(2, 6, 8),
        ],
        acquisition_fps=10,
    )

    assert invalid_dataframe["speed_pixels_per_second"].isna().all()
    assert missing_dataframe["speed_pixels_per_second"].isna().all()


def test_build_tracking_dataframe_warns_about_abrupt_area_change():
    measurements = [
        _measurement(index, index, index, area=300 if index == 5 else 100) for index in range(11)
    ]

    dataframe = build_tracking_dataframe(measurements, acquisition_fps=10)

    assert dataframe.loc[5, "segmentation_valid"]
    assert dataframe.loc[5, "area_to_local_median_ratio"] == 3
    assert "abrupt_area_change" in dataframe.loc[5, "quality_reason"]
    assert dataframe.loc[5, "center_x_pixels"] == 5
    assert np.isfinite(dataframe.loc[5, "speed_pixels_per_second"])


def test_temporal_area_warning_does_not_restore_prior_invalid_frame():
    measurements = [
        _measurement(index, index, index, area=300 if index == 5 else 100) for index in range(11)
    ]
    measurements[5]["segmentation_valid"] = False
    measurements[5]["quality_reason"] = "low_component_confidence"

    dataframe = build_tracking_dataframe(measurements, acquisition_fps=10)

    assert not dataframe.loc[5, "segmentation_valid"]
    assert dataframe.loc[5, "quality_reason"] == "low_component_confidence"
    assert np.isnan(dataframe.loc[5, "center_x_pixels"])
    assert np.isnan(dataframe.loc[5, "speed_pixels_per_second"])


def test_temporal_area_check_uses_source_frame_distance():
    measurements = [
        _measurement(index * 100, index, index, area=300 if index == 5 else 100)
        for index in range(11)
    ]

    dataframe = build_tracking_dataframe(measurements, acquisition_fps=10)

    assert dataframe["segmentation_valid"].all()
    assert dataframe["local_area_median"].isna().all()
