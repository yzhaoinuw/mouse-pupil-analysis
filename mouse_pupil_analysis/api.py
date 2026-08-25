"""Programmatic entry points for the pupil analysis pipeline.

``run_analysis`` performs the same work as the ``run-pupil-analysis`` console
script, so a notebook or another package can drive the pipeline and receive the
results as a DataFrame instead of reading them back from CSV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import pandas as pd

from mouse_pupil_analysis.extract_frames import ExtractedFrame, extract_selected_frames
from mouse_pupil_analysis.pupil_predictions import (
    MaskOverlayAccumulator,
    find_default_checkpoint,
    frames_from_image_directory,
    iter_pupil_predictions,
    resolve_prediction_threshold,
)
from mouse_pupil_analysis.results import DiameterRow, write_analysis_outputs
from mouse_pupil_analysis.tracking import SegmentationAccumulator, TrackingAccumulator

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Every input the analysis pipeline accepts.

    Exactly one of ``video_path`` or ``image_dir`` is required. Paths left as
    ``None`` are derived from the input location when the analysis runs.
    """

    video_path: Path | None = None
    image_dir: Path | None = None
    out_dir: Path | None = None
    result_dir: Path | None = None
    checkpoint: Path | None = None
    output_mask_dir: Path | None = None
    batch_size: int = 32
    pred_thresh: float | None = None
    prefer_central_component: bool = False
    mask_transparency: float = 0.1
    extraction_fps: float = 5.0
    max_frames: int = 10000
    calculate_velocity: bool = False
    acquisition_fps: float | None = None
    num_workers: int | None = None
    # Off by default: a library should not write to stderr unless asked. The
    # console scripts set this to True.
    show_progress: bool = False

    def __post_init__(self) -> None:
        for name in (
            "video_path",
            "image_dir",
            "out_dir",
            "result_dir",
            "checkpoint",
            "output_mask_dir",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, Path(value))

    def validate(self) -> None:
        """Raise ``ValueError`` with a user-facing message for any bad combination."""
        if self.video_path is None and self.image_dir is None:
            raise ValueError("You must specify either video_path or image_dir.")
        if self.video_path is not None and self.image_dir is not None:
            raise ValueError("Specify only one of video_path or image_dir, not both.")
        if self.pred_thresh is not None and not 0 < self.pred_thresh < 1:
            raise ValueError("pred_thresh must be between 0 and 1.")
        if self.acquisition_fps is not None and self.acquisition_fps <= 0:
            raise ValueError("acquisition_fps must be positive.")
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.calculate_velocity and self.image_dir is not None and self.acquisition_fps is None:
            raise ValueError("acquisition_fps is required with image_dir in velocity mode.")


@dataclass
class AnalysisResult:
    """Everything one analysis run produced."""

    analysis_table: pd.DataFrame
    csv_path: Path
    plot_path: Path
    image_frames: list[ExtractedFrame] = field(repr=False)
    prediction_threshold: float = 0.7
    segmentation_dataframe: pd.DataFrame | None = field(default=None, repr=False)
    tracking_dataframe: pd.DataFrame | None = field(default=None, repr=False)


def read_encoded_video_fps(video_path: Path) -> float:
    """Return the frame rate recorded in a video container's header."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video_path}")
    encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if encoded_fps <= 0:
        raise ValueError(f"Video reports an invalid FPS: {encoded_fps}")
    return encoded_fps


def _resolve_frames(config: AnalysisConfig) -> tuple[list[ExtractedFrame], AnalysisConfig]:
    """Extract frames from video when needed and settle the derived paths."""
    if config.video_path is not None:
        logger.info("Video path provided - extracting frames first...")
        if config.out_dir is None:
            config.out_dir = config.video_path.parent / f"{config.video_path.stem}_frames"
            logger.info("No out_dir provided. Using default: %s", config.out_dir)

        image_frames = extract_selected_frames(
            config.video_path,
            config.out_dir,
            config.extraction_fps,
            config.max_frames,
            extract_all=config.calculate_velocity,
            show_progress=config.show_progress,
        )
        config.image_dir = config.out_dir

        if config.calculate_velocity and config.acquisition_fps is None:
            config.acquisition_fps = read_encoded_video_fps(config.video_path)
            logger.info(
                "No acquisition FPS override provided. Using encoded video rate: %.6g fps",
                config.acquisition_fps,
            )
    else:
        image_frames = frames_from_image_directory(config.image_dir)

    if not image_frames:
        raise FileNotFoundError(f"No PNG files found in {config.image_dir}")
    return image_frames, config


def _resolve_result_dir(config: AnalysisConfig) -> Path:
    if config.result_dir is None:
        if config.video_path is not None:
            config.result_dir = config.video_path.parent / f"{config.video_path.stem}_result"
        else:
            config.result_dir = Path(f"{config.image_dir}_result")
        logger.info("No result_dir provided. Using default: %s", config.result_dir)
    config.result_dir.mkdir(parents=True, exist_ok=True)
    return config.result_dir


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    """Run the full pipeline described by ``config`` and write its outputs."""
    config.validate()
    image_frames, config = _resolve_frames(config)

    if config.calculate_velocity:
        logger.info("Using acquisition rate: %.10g samples/s", config.acquisition_fps)

    checkpoint = config.checkpoint if config.checkpoint is not None else find_default_checkpoint()
    prediction_threshold = resolve_prediction_threshold(checkpoint, config.pred_thresh)
    logger.info("Prediction threshold: %.3g", prediction_threshold)

    if config.calculate_velocity:
        segmentation_accumulator = TrackingAccumulator(
            pred_thresh=prediction_threshold,
            acquisition_fps=config.acquisition_fps,
        )
    else:
        segmentation_accumulator = SegmentationAccumulator(pred_thresh=prediction_threshold)

    overlay_accumulator = None
    if config.output_mask_dir is not None:
        config.output_mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_accumulator = MaskOverlayAccumulator(
            config.output_mask_dir,
            pred_thresh=prediction_threshold,
            mask_transparency=config.mask_transparency,
        )

    results = []
    for prediction in iter_pupil_predictions(
        checkpoint,
        image_frames,
        pred_thresh=prediction_threshold,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefer_central_component=config.prefer_central_component,
        show_progress=config.show_progress,
    ):
        results.append(
            DiameterRow(
                prediction.image_name,
                prediction.estimated_pupil_diameter,
                prediction.pupil_diameter_input_pixels,
            )
        )
        segmentation_accumulator.add(prediction)
        if overlay_accumulator is not None:
            overlay_accumulator.add(prediction)

    segmentation_dataframe = segmentation_accumulator.build_dataframe()
    tracking_dataframe = segmentation_dataframe if config.calculate_velocity else None
    if overlay_accumulator is not None:
        overlay_accumulator.save(
            image_frames,
            tracking_dataframe=segmentation_dataframe,
        )

    result_dir = _resolve_result_dir(config)
    exp_name = (
        "_".join(Path(results[0].image_name).stem.split("_")[:-1]) if results else "experiment"
    )
    analysis_table, csv_path, plot_path = write_analysis_outputs(
        results,
        image_frames,
        result_dir,
        exp_name,
        segmentation_dataframe=segmentation_dataframe,
        tracking_dataframe=tracking_dataframe,
    )

    return AnalysisResult(
        analysis_table=analysis_table,
        csv_path=csv_path,
        plot_path=plot_path,
        image_frames=image_frames,
        prediction_threshold=prediction_threshold,
        segmentation_dataframe=segmentation_dataframe,
        tracking_dataframe=tracking_dataframe,
    )


def analyze_video(video_path, **kwargs) -> AnalysisResult:
    """Extract frames from a video and run the pupil analysis pipeline."""
    return run_analysis(AnalysisConfig(video_path=video_path, **kwargs))


def analyze_frames(image_dir, **kwargs) -> AnalysisResult:
    """Run the pupil analysis pipeline on a directory of extracted PNG frames."""
    return run_analysis(AnalysisConfig(image_dir=image_dir, **kwargs))
