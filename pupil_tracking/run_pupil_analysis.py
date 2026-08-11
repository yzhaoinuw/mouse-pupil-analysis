# -*- coding: utf-8 -*-
"""Command-line entry point for the pupil analysis pipeline.

This module only parses arguments and reports errors. The pipeline itself lives
in :mod:`pupil_tracking.api`, so it can be driven from Python without a shell.
"""

import argparse
from pathlib import Path

from pupil_tracking.api import AnalysisConfig, run_analysis
from pupil_tracking.logging_utils import configure_cli_logging


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
    return parser


def main():
    configure_cli_logging()
    parser = build_parser()
    args = parser.parse_args()

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
        mask_transparency=args.mask_transparency,
        extraction_fps=args.extraction_fps,
        max_frames=args.max_frames,
        calculate_velocity=args.calculate_velocity,
        acquisition_fps=args.acquisition_fps,
    )

    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))

    run_analysis(config)


if __name__ == "__main__":
    main()
