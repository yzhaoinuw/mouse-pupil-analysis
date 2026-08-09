# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 00:07:48 2025

@author: yzhao
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pupil_tracking.extract_frames import ExtractedFrame, extract_selected_frames
from pupil_tracking.pupil_predictions import (
    DEFAULT_CHECKPOINT,
    MaskOverlayAccumulator,
    frames_from_image_directory,
    iter_pupil_predictions,
)
from pupil_tracking.pupil_predictions import (
    generate_pupil_mask_prediction as generate_pupil_mask_prediction,
)
from pupil_tracking.pupil_predictions import (
    generate_pupil_predictions as generate_pupil_predictions,
)
from pupil_tracking.tracking import TrackingAccumulator

_LEGACY_RESULT_SUFFIXES = (
    "estimated_pupil_diameter.csv",
    "estimated_pupil_diameter.png",
    "pupil_tracking.csv",
    "pupil_tracking_qc.png",
)


def _read_encoded_video_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video_path}")
    encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if encoded_fps <= 0:
        raise ValueError(f"Video reports an invalid FPS: {encoded_fps}")
    return encoded_fps


def _tracking_status(tracking_dataframe: pd.DataFrame) -> pd.Series:
    """Reduce detailed quality evidence to valid, warning, or invalid."""
    valid = tracking_dataframe["segmentation_valid"].astype(bool)
    reasons = tracking_dataframe["quality_reason"].fillna("").astype(str)
    return pd.Series(
        np.select(
            [~valid, reasons.ne("")],
            ["invalid", "warning"],
            default="valid",
        ),
        index=tracking_dataframe.index,
        dtype="object",
    )


def _remove_legacy_result_files(result_dir: Path, exp_name: str) -> None:
    """Remove superseded duplicate outputs after the unified files are saved."""
    for suffix in _LEGACY_RESULT_SUFFIXES:
        legacy_path = result_dir / f"{exp_name}_{suffix}"
        if not legacy_path.exists():
            continue
        try:
            legacy_path.unlink()
            print(f"Removed legacy result: {legacy_path}")
        except PermissionError:
            print(f"Warning: close the legacy result so it can be removed: {legacy_path}")


def save_analysis_results(
    results,
    image_frames: list[ExtractedFrame],
    result_dir: Path,
    exp_name: str,
    tracking_dataframe: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Save one compact analysis table and one frame-indexed plot."""
    source_index_by_name = {
        frame.image_path.name: frame.source_frame_index for frame in image_frames
    }
    result_rows = [
        {
            "image_name": name,
            "estimated_pupil_diameter": diameter,
            "source_frame_index": source_index_by_name[name],
        }
        for name, diameter in results
    ]
    result_dataframe = pd.DataFrame(result_rows).sort_values(
        "source_frame_index",
        kind="stable",
    )

    if tracking_dataframe is None:
        output_dataframe = result_dataframe[["image_name", "estimated_pupil_diameter"]].reset_index(
            drop=True
        )
        frame_numbers = result_dataframe["source_frame_index"].to_numpy(dtype=int) + 1
    else:
        ordered_tracking = (
            tracking_dataframe.set_index("image_name")
            .loc[result_dataframe["image_name"]]
            .reset_index()
        )
        ordered_tracking["tracking_status"] = _tracking_status(ordered_tracking)
        ordered_tracking["quality_reason"] = (
            ordered_tracking["quality_reason"].fillna("").astype(str)
        )
        output_columns = [
            "image_name",
            "estimated_pupil_diameter",
            "timestamp_seconds",
            "center_x_pixels",
            "center_y_pixels",
            "speed_pixels_per_second",
            "tracking_status",
            "quality_reason",
        ]
        output_dataframe = ordered_tracking[output_columns]
        frame_numbers = ordered_tracking["source_frame_index"].to_numpy(dtype=int) + 1

    csv_path = result_dir / f"{exp_name}_pupil_analysis.csv"
    output_dataframe.to_csv(csv_path, index=False)

    if tracking_dataframe is None:
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(frame_numbers, output_dataframe["estimated_pupil_diameter"], linewidth=1)
        axis.set_ylabel("Estimated pupil diameter\n(model pixels)")
        axis.set_title("Pupil Analysis")
        axis.set_xlabel("Frame (1-based)")
    else:
        figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
        axes[0].plot(
            frame_numbers,
            output_dataframe["estimated_pupil_diameter"],
            linewidth=0.8,
        )
        axes[0].set_ylabel("Diameter\n(model pixels)")
        axes[0].set_title("Pupil Analysis")

        axes[1].plot(
            frame_numbers,
            output_dataframe["center_x_pixels"],
            label="x",
            linewidth=0.8,
        )
        axes[1].plot(
            frame_numbers,
            output_dataframe["center_y_pixels"],
            label="y",
            linewidth=0.8,
        )
        axes[1].set_ylabel("Center\n(video pixels)")
        axes[1].legend(loc="upper right")

        axes[2].plot(
            frame_numbers,
            output_dataframe["speed_pixels_per_second"],
            linewidth=0.8,
        )
        axes[2].set_ylabel("Speed\n(pixels/s)")

        status_codes = output_dataframe["tracking_status"].map(
            {"invalid": 0, "warning": 1, "valid": 2}
        )
        axes[3].step(
            frame_numbers,
            status_codes,
            where="mid",
            color="#9CA3AF",
            linewidth=0.6,
        )
        for status, code, color in (
            ("valid", 2, "#16A34A"),
            ("warning", 1, "#F59E0B"),
            ("invalid", 0, "#DC2626"),
        ):
            selected = output_dataframe["tracking_status"].eq(status)
            axes[3].scatter(
                frame_numbers[selected],
                np.full(selected.sum(), code),
                color=color,
                s=8 if status == "valid" else 16,
                label=status,
            )
        axes[3].set_yticks([0, 1, 2], labels=["invalid", "warning", "valid"])
        axes[3].set_ylim(-0.25, 2.25)
        axes[3].set_xlabel("Frame (1-based)")
        axes[3].legend(loc="lower right", ncol=3)

    figure.tight_layout()
    plot_path = result_dir / f"{exp_name}_pupil_analysis.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)

    _remove_legacy_result_files(result_dir, exp_name)
    print(f"Saved analysis CSV:  {csv_path}")
    print(f"Saved analysis plot: {plot_path}")
    return csv_path, plot_path


def main():
    parser = argparse.ArgumentParser(description="Pupil diameter analysis pipeline")
    parser.add_argument(
        "--video_path", type=Path, help="Optional video file to extract frames from"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Directory to save extracted frames (used with --video_path)",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        help="Directory of existing PNG images (skips extraction)",
    )
    parser.add_argument(
        "--result_dir",
        type=Path,
        help="Directory to save results (auto-created if not given)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=Path,
        default=None,
        help="Optional directory to save overlay images",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pred_thresh", type=float, default=0.7)
    parser.add_argument("--mask_transparency", type=float, default=0.1)
    parser.add_argument("--extraction_fps", type=float, default=5)
    parser.add_argument("--max_frames", type=int, default=10000)
    parser.add_argument(
        "--calculate_velocity",
        action="store_true",
        help="Analyze every encoded frame and calculate pupil-center velocity.",
    )
    parser.add_argument(
        "--acquisition_fps",
        type=float,
        help=(
            "Actual acquisition rate used for timestamps and velocity. "
            "Defaults to the video header rate for video input."
        ),
    )
    args = parser.parse_args()

    if args.acquisition_fps is not None and args.acquisition_fps <= 0:
        parser.error("--acquisition_fps must be positive.")
    if args.calculate_velocity and args.image_dir and args.acquisition_fps is None:
        parser.error("--acquisition_fps is required with --image_dir in velocity mode.")

    image_frames: list[ExtractedFrame]
    if args.video_path:
        print("Video path provided — extracting frames first...")
        if args.out_dir is None:
            args.out_dir = args.video_path.parent / f"{args.video_path.stem}_frames"
            print(f"No out_dir provided. Using default: {args.out_dir}")

        image_frames = extract_selected_frames(
            args.video_path,
            args.out_dir,
            args.extraction_fps,
            args.max_frames,
            extract_all=args.calculate_velocity,
        )
        args.image_dir = args.out_dir

        if args.calculate_velocity and args.acquisition_fps is None:
            args.acquisition_fps = _read_encoded_video_fps(args.video_path)
            print(
                "No acquisition FPS override provided. "
                f"Using encoded video rate: {args.acquisition_fps:.6g} fps"
            )
    elif args.image_dir is not None:
        image_frames = frames_from_image_directory(args.image_dir)
    else:
        raise ValueError("You must specify either (--video_path) or (--image_dir).")

    if not image_frames:
        raise FileNotFoundError(f"No PNG files found in {args.image_dir}")

    if args.calculate_velocity:
        print(f"Using acquisition rate: {args.acquisition_fps:.10g} samples/s")

    if args.output_mask_dir is not None:
        args.output_mask_dir.mkdir(parents=True, exist_ok=True)

    tracking_accumulator = None
    if args.calculate_velocity:
        tracking_accumulator = TrackingAccumulator(
            pred_thresh=args.pred_thresh,
            acquisition_fps=args.acquisition_fps,
        )
    overlay_accumulator = None
    if args.output_mask_dir is not None:
        overlay_accumulator = MaskOverlayAccumulator(
            args.output_mask_dir,
            pred_thresh=args.pred_thresh,
            mask_transparency=args.mask_transparency,
        )

    results = []
    for prediction in iter_pupil_predictions(
        args.checkpoint,
        image_frames,
        pred_thresh=args.pred_thresh,
        batch_size=args.batch_size,
    ):
        results.append((prediction.image_name, prediction.estimated_pupil_diameter))
        if tracking_accumulator is not None:
            tracking_accumulator.add(prediction)
        if overlay_accumulator is not None:
            overlay_accumulator.add(prediction)

    tracking_dataframe = None
    if tracking_accumulator is not None:
        tracking_dataframe = tracking_accumulator.build_dataframe()
    if overlay_accumulator is not None:
        overlay_accumulator.save(
            image_frames,
            tracking_dataframe=tracking_dataframe,
        )

    if args.result_dir is None:
        if args.video_path:
            args.result_dir = args.video_path.parent / f"{args.video_path.stem}_result"
        else:
            args.result_dir = Path(str(args.image_dir) + "_result")
        print(f"No result_dir provided. Using default: {args.result_dir}")
    args.result_dir.mkdir(parents=True, exist_ok=True)

    exp_name = "_".join(Path(results[0][0]).stem.split("_")[:-1]) if results else "experiment"
    save_analysis_results(
        results,
        image_frames,
        args.result_dir,
        exp_name,
        tracking_dataframe=tracking_dataframe,
    )


if __name__ == "__main__":
    main()
