"""Postprocess pupil probability maps and calculate pupil-center kinematics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import pandas as pd

from pupil_tracking.extract_frames import ExtractedFrame
from pupil_tracking.preprocessing import MODEL_IMAGE_SIZE, resize_scale

MIN_COMPONENT_DOMINANCE = 0.80
MIN_MEAN_COMPONENT_CONFIDENCE = 0.90
MIN_COMPONENT_CIRCULARITY = 0.45
MIN_LOCAL_AREA_RATIO = 0.65
MAX_LOCAL_AREA_RATIO = 2.00
TEMPORAL_AREA_WINDOW_SECONDS = 0.50
MIN_TEMPORAL_NEIGHBORS = 4


class PupilPredictionLike(Protocol):
    """Structural interface consumed by ``TrackingAccumulator``."""

    frame: ExtractedFrame
    probability_map: np.ndarray
    binary_mask: np.ndarray
    estimated_pupil_diameter: float
    original_size: tuple[int, int]

    @property
    def image_name(self) -> str: ...

    @property
    def pupil_diameter_video_pixels(self) -> float: ...


def model_to_original_coordinates(
    center_x: float,
    center_y: float,
    original_width: int,
    original_height: int,
    target_size: int = MODEL_IMAGE_SIZE,
) -> tuple[float, float]:
    """Invert the aspect-ratio-preserving resize and padding used for inference."""
    scale_x, scale_y, pad_left, pad_top = resize_scale(
        original_width,
        original_height,
        target_size,
    )

    original_x = (center_x - pad_left) / scale_x
    original_y = (center_y - pad_top) / scale_y
    return float(original_x), float(original_y)


def _component_circularity(component_mask: np.ndarray, area: int) -> float:
    contours, _ = cv2.findContours(
        component_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    perimeter = sum(cv2.arcLength(contour, closed=True) for contour in contours)
    if perimeter <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter**2))


def _join_reasons(reasons: Iterable[str]) -> str:
    return ";".join(dict.fromkeys(reason for reason in reasons if reason))


def measure_probability_map(
    probability_map: np.ndarray,
    pred_thresh: float,
    original_size: tuple[int, int],
    binary_mask: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Measure the most plausible pupil component in one probability map."""
    probabilities = np.asarray(probability_map, dtype=np.float32).squeeze()
    if probabilities.ndim != 2:
        raise ValueError("Probability map must be two-dimensional after squeezing.")
    if not 0 < pred_thresh < 1:
        raise ValueError("Prediction threshold must be between 0 and 1.")

    original_width, original_height = original_size
    if binary_mask is None:
        binary_mask = (probabilities > pred_thresh).astype(np.uint8)
    else:
        binary_mask = np.asarray(binary_mask).squeeze()
        if binary_mask.shape != probabilities.shape:
            raise ValueError("Binary mask shape must match the probability map shape.")
        binary_mask = binary_mask.astype(bool).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )
    component_count -= 1

    empty_measurement: dict[str, object] = {
        "raw_center_x_pixels": np.nan,
        "raw_center_y_pixels": np.nan,
        "raw_center_x_model_pixels": np.nan,
        "raw_center_y_model_pixels": np.nan,
        "selected_component_area": 0,
        "foreground_area": int(binary_mask.sum()),
        "component_count": int(component_count),
        "component_dominance": np.nan,
        "mean_component_confidence": np.nan,
        "component_circularity": np.nan,
        "component_touches_border": False,
        "segmentation_valid": False,
        "quality_reason": "empty_mask",
    }
    selected_mask = np.zeros_like(binary_mask)
    if component_count == 0:
        return empty_measurement, binary_mask, selected_mask

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    selected_label = 1 + int(np.argmax(component_areas))
    selected_mask = (labels == selected_label).astype(np.uint8)
    selected_area = int(selected_mask.sum())
    foreground_area = int(binary_mask.sum())
    dominance = selected_area / foreground_area
    mean_confidence = float(probabilities[selected_mask.astype(bool)].mean())

    selected_y, selected_x = np.nonzero(selected_mask)
    selected_probabilities = probabilities[selected_y, selected_x]
    probability_sum = float(selected_probabilities.sum())
    if probability_sum > 0:
        center_x_model = float(np.dot(selected_x, selected_probabilities) / probability_sum)
        center_y_model = float(np.dot(selected_y, selected_probabilities) / probability_sum)
    else:
        center_x_model = float(selected_x.mean())
        center_y_model = float(selected_y.mean())

    center_x, center_y = model_to_original_coordinates(
        center_x_model,
        center_y_model,
        original_width,
        original_height,
        target_size=probabilities.shape[1],
    )
    touches_border = bool(
        selected_mask[0].any()
        or selected_mask[-1].any()
        or selected_mask[:, 0].any()
        or selected_mask[:, -1].any()
    )
    circularity = _component_circularity(selected_mask, selected_area)

    quality_reasons: list[str] = []
    rejection_reasons: list[str] = []
    if dominance < MIN_COMPONENT_DOMINANCE:
        quality_reasons.append("low_component_dominance")
    if mean_confidence < MIN_MEAN_COMPONENT_CONFIDENCE:
        quality_reasons.append("low_component_confidence")
        rejection_reasons.append("low_component_confidence")
    if circularity < MIN_COMPONENT_CIRCULARITY:
        quality_reasons.append("low_component_circularity")
        rejection_reasons.append("low_component_circularity")
    if touches_border:
        quality_reasons.append("component_touches_border")
        rejection_reasons.append("component_touches_border")

    measurement = {
        "raw_center_x_pixels": center_x,
        "raw_center_y_pixels": center_y,
        "raw_center_x_model_pixels": center_x_model,
        "raw_center_y_model_pixels": center_y_model,
        "selected_component_area": selected_area,
        "foreground_area": foreground_area,
        "component_count": int(component_count),
        "component_dominance": float(dominance),
        "mean_component_confidence": mean_confidence,
        "component_circularity": circularity,
        "component_touches_border": touches_border,
        "segmentation_valid": not rejection_reasons,
        "quality_reason": _join_reasons(quality_reasons),
    }
    return measurement, binary_mask, selected_mask


class TrackingAccumulator:
    """Consume transient predictions and finalize temporal tracking results."""

    def __init__(self, pred_thresh: float, acquisition_fps: float) -> None:
        if not 0 < pred_thresh < 1:
            raise ValueError("Prediction threshold must be between 0 and 1.")
        if acquisition_fps <= 0:
            raise ValueError("Acquisition FPS must be positive.")
        self.pred_thresh = pred_thresh
        self.acquisition_fps = acquisition_fps
        self.measurements: list[dict[str, object]] = []

    def add(self, prediction: PupilPredictionLike) -> None:
        """Measure one prediction without retaining its float probability map."""
        measurement, _, _ = measure_probability_map(
            prediction.probability_map,
            pred_thresh=self.pred_thresh,
            original_size=prediction.original_size,
            binary_mask=prediction.binary_mask,
        )
        measurement.update(
            {
                "image_name": prediction.image_name,
                "source_frame_index": prediction.frame.source_frame_index,
                "estimated_pupil_diameter": prediction.estimated_pupil_diameter,
                "pupil_diameter_video_pixels": prediction.pupil_diameter_video_pixels,
            }
        )
        self.measurements.append(measurement)

    def build_dataframe(self) -> pd.DataFrame:
        """Apply temporal quality checks and calculate kinematics."""
        return build_tracking_dataframe(
            self.measurements,
            acquisition_fps=self.acquisition_fps,
        )


def _append_quality_reason(current: object, reason: str) -> str:
    current_text = "" if pd.isna(current) else str(current)
    return _join_reasons([current_text, reason])


def build_tracking_dataframe(
    measurements: Iterable[Mapping[str, object]],
    acquisition_fps: float,
) -> pd.DataFrame:
    """Apply temporal quality checks and calculate frame-to-frame kinematics."""
    if acquisition_fps <= 0:
        raise ValueError("Acquisition FPS must be positive.")

    dataframe = pd.DataFrame(measurements).copy()
    if dataframe.empty:
        return dataframe

    required_columns = {
        "source_frame_index",
        "raw_center_x_pixels",
        "raw_center_y_pixels",
        "selected_component_area",
        "segmentation_valid",
        "quality_reason",
    }
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Tracking measurements are missing required columns: {missing_text}")

    dataframe = dataframe.sort_values("source_frame_index", kind="stable").reset_index(drop=True)
    dataframe["timestamp_seconds"] = dataframe["source_frame_index"].astype(float) / acquisition_fps
    dataframe["local_area_median"] = np.nan
    dataframe["area_to_local_median_ratio"] = np.nan

    initial_valid = dataframe["segmentation_valid"].astype(bool).to_numpy(copy=True)
    areas = dataframe["selected_component_area"].to_numpy(dtype=float)
    source_frame_indices = dataframe["source_frame_index"].to_numpy(dtype=int)
    temporal_radius = max(1, int(round(acquisition_fps * TEMPORAL_AREA_WINDOW_SECONDS)))

    for index in range(len(dataframe)):
        if not initial_valid[index]:
            continue
        window_start = int(
            np.searchsorted(
                source_frame_indices,
                source_frame_indices[index] - temporal_radius,
                side="left",
            )
        )
        window_stop = int(
            np.searchsorted(
                source_frame_indices,
                source_frame_indices[index] + temporal_radius,
                side="right",
            )
        )
        neighbor_indices = [
            neighbor
            for neighbor in range(window_start, window_stop)
            if neighbor != index and initial_valid[neighbor] and np.isfinite(areas[neighbor])
        ]
        if len(neighbor_indices) < MIN_TEMPORAL_NEIGHBORS:
            continue

        local_median = float(np.median(areas[neighbor_indices]))
        if local_median <= 0:
            continue
        area_ratio = float(areas[index] / local_median)
        dataframe.at[index, "local_area_median"] = local_median
        dataframe.at[index, "area_to_local_median_ratio"] = area_ratio
        if area_ratio < MIN_LOCAL_AREA_RATIO or area_ratio > MAX_LOCAL_AREA_RATIO:
            dataframe.at[index, "quality_reason"] = _append_quality_reason(
                dataframe.at[index, "quality_reason"],
                "abrupt_area_change",
            )

    valid = dataframe["segmentation_valid"].astype(bool)
    dataframe["center_x_pixels"] = dataframe["raw_center_x_pixels"].where(valid, np.nan)
    dataframe["center_y_pixels"] = dataframe["raw_center_y_pixels"].where(valid, np.nan)

    # Kinematics are only defined between immediately consecutive source frames whose
    # segmentations are both usable. Everything else stays missing; the pipeline never
    # bridges a gap. Computed column-wise so long recordings stay fast.
    source_indices = dataframe["source_frame_index"].to_numpy(dtype=float)
    timestamps = dataframe["timestamp_seconds"].to_numpy(dtype=float)
    center_x = dataframe["center_x_pixels"].to_numpy(dtype=float)
    center_y = dataframe["center_y_pixels"].to_numpy(dtype=float)
    valid_pair = valid.to_numpy(dtype=bool)

    delta_time = np.diff(timestamps, prepend=np.nan)
    consecutive = np.diff(source_indices, prepend=np.nan) == 1
    both_valid = np.concatenate(([False], valid_pair[1:] & valid_pair[:-1]))
    usable = consecutive & both_valid & (delta_time > 0)

    delta_x = np.where(usable, np.diff(center_x, prepend=np.nan), np.nan)
    delta_y = np.where(usable, np.diff(center_y, prepend=np.nan), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        velocity_x = delta_x / delta_time
        velocity_y = delta_y / delta_time

    dataframe["displacement_x_pixels"] = delta_x
    dataframe["displacement_y_pixels"] = delta_y
    dataframe["velocity_x_pixels_per_second"] = velocity_x
    dataframe["velocity_y_pixels_per_second"] = velocity_y
    dataframe["speed_pixels_per_second"] = np.hypot(velocity_x, velocity_y)

    return dataframe


if __name__ == "__main__":
    # Edit these values, then run this file directly from an IDE.
    from pupil_tracking.pupil_predictions import (
        find_default_checkpoint,
        frames_from_image_directory,
        iter_pupil_predictions,
    )

    project_root = Path(__file__).resolve().parents[1]
    image_dir = project_root / "images_test_1"
    checkpoint_path = find_default_checkpoint()
    pred_thresh = 0.7
    batch_size = 32
    acquisition_fps = 10.0  # Replace with the recording's actual acquisition rate.

    image_frames = frames_from_image_directory(image_dir)
    tracking_accumulator = TrackingAccumulator(pred_thresh, acquisition_fps)
    for prediction in iter_pupil_predictions(
        checkpoint_path,
        image_frames,
        pred_thresh=pred_thresh,
        batch_size=batch_size,
    ):
        tracking_accumulator.add(prediction)

    tracking_dataframe = tracking_accumulator.build_dataframe()
    print(tracking_dataframe.head())
