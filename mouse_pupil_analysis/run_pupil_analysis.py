# -*- coding: utf-8 -*-
"""Run the pupil analysis pipeline from a terminal or directly from this file.

Terminal arguments use the normal command-line workflow.
The pipeline itself lives in :mod:`mouse_pupil_analysis.api`,
so it can also be driven from Python without a shell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mouse_pupil_analysis.api import AnalysisConfig, run_analysis
from mouse_pupil_analysis.logging_utils import configure_cli_logging
from mouse_pupil_analysis.pupil_predictions import (
    generate_pupil_mask_prediction,
    generate_pupil_predictions,
)

__all__ = [
    "build_parser",
    "generate_pupil_mask_prediction",
    "generate_pupil_predictions",
    "main",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
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
        default=None,
        help="Path to model checkpoint (default: the packaged checkpoint with the highest IoU)",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=Path,
        default=None,
        help="Optional directory to save overlay images",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Dataloader worker processes (default: up to 4, capped by CPU count). Use 0 to disable.",
    )
    parser.add_argument(
        "--pred_thresh",
        type=float,
        default=None,
        help=(
            "Override the checkpoint's calibrated pixel-confidence threshold. "
            "Defaults to checkpoint metadata or its filename, then 0.7."
        ),
    )
    parser.add_argument(
        "--prefer_central_component",
        action="store_true",
        help="Keep the confidence-weighted component nearest the image center. Off by default.",
    )
    parser.add_argument("--mask_transparency", type=float, default=0.05)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse terminal arguments and run analysis."""
    configure_cli_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    config = AnalysisConfig(
        video_path=args.video_path,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        result_dir=args.result_dir,
        checkpoint=args.checkpoint,
        output_mask_dir=args.output_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pred_thresh=args.pred_thresh,
        prefer_central_component=args.prefer_central_component,
        mask_transparency=args.mask_transparency,
        extraction_fps=args.extraction_fps,
        max_frames=args.max_frames,
        calculate_velocity=args.calculate_velocity,
        acquisition_fps=args.acquisition_fps,
        show_progress=True,
    )

    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))

    run_analysis(config)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())

    video_path = None
    test_dir = PROJECT_ROOT / "images_test"
    test_folder_name = "images_test_hyunseok"
    image_dir = test_dir / test_folder_name

    # Results are ignored by Git. Set any option to None to use the pipeline default.
    out_dir = None
    result_dir = test_dir / f"results_{test_folder_name}"
    checkpoint = None  # Use the packaged checkpoint with the highest encoded IoU.
    output_mask_dir = test_dir / f"predicted_masks_{test_folder_name}"
    batch_size = 32
    num_workers = None
    pred_thresh = None  # Use checkpoint calibration; set a float to override it.
    prefer_central_component = False  # Keep one centre-favoured component when needed.
    mask_transparency = 0.05
    extraction_fps = 5.0
    max_frames = 10000
    calculate_velocity = False
    acquisition_fps = None

    run_analysis(
        AnalysisConfig(
            video_path=video_path,
            image_dir=image_dir,
            out_dir=out_dir,
            result_dir=result_dir,
            checkpoint=checkpoint,
            output_mask_dir=output_mask_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pred_thresh=pred_thresh,
            prefer_central_component=prefer_central_component,
            mask_transparency=mask_transparency,
            extraction_fps=extraction_fps,
            max_frames=max_frames,
            calculate_velocity=calculate_velocity,
            acquisition_fps=acquisition_fps,
            show_progress=True,
        )
    )
