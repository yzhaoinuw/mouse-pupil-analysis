# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 15:27:03 2025

@author: yzhao
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pupil_tracking.logging_utils import configure_cli_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedFrame:
    """Metadata for one image written from a source video frame."""

    image_path: Path
    source_frame_index: int
    extraction_index: int


def source_frame_image_name(video_name: str, source_frame_index: int) -> str:
    """Build a one-based image name from a zero-based source-frame index."""
    if source_frame_index < 0:
        raise ValueError("Source-frame index cannot be negative.")
    return f"{video_name}_{source_frame_index + 1:05d}.png"


def select_frame_indices(
    frame_count,
    encoded_fps,
    extraction_fps=5,
    max_frames=10000,
    extract_all=False,
):
    """Select source-frame indices for sampled or full-frame extraction."""
    if frame_count <= 0:
        raise ValueError("Video contains no frames.")
    if encoded_fps <= 0:
        raise ValueError("Video FPS must be positive.")
    if extraction_fps <= 0:
        raise ValueError("Extraction FPS must be positive.")
    if max_frames <= 0:
        raise ValueError("Maximum frame count must be positive.")

    if extract_all and max_frames >= frame_count:
        return np.arange(frame_count, dtype=int)

    if extract_all:
        return np.arange(max_frames, dtype=int)

    duration = frame_count / encoded_fps
    effective_extraction_fps = min(extraction_fps, encoded_fps)
    selected_count = round(duration * effective_extraction_fps)
    selected_count = max(1, min(frame_count, max_frames, selected_count))
    return np.linspace(0, frame_count - 1, selected_count, dtype=int)


def extract_selected_frames(
    video_path,
    out_dir,
    extraction_fps=5,
    max_frames=10000,
    extract_all=False,
):
    """Extract video frames and return their paths and original source indices."""
    video_path = Path(video_path)
    video_name = video_path.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info("Total frames: %d. Original FPS: %.2f", frame_count, fps)

    selected_frames = select_frame_indices(
        frame_count,
        fps,
        extraction_fps=extraction_fps,
        max_frames=max_frames,
        extract_all=extract_all,
    )
    if extract_all and len(selected_frames) < frame_count:
        logger.warning(
            "--max_frames limits full-frame extraction to the first %d of %d source frames.",
            len(selected_frames),
            frame_count,
        )
    if extract_all:
        logger.info("Extracting consecutive source frames (%d frames).", len(selected_frames))
    else:
        duration = frame_count / fps
        effective_extraction_fps = len(selected_frames) / duration
        logger.info(
            "Extracting %d frames at ~%.2f encoded-video fps",
            len(selected_frames),
            effective_extraction_fps,
        )

    extracted_frames = []
    sequential_read = len(selected_frames) == frame_count and np.array_equal(
        selected_frames,
        np.arange(frame_count),
    )
    for extraction_index, frame_idx in tqdm(
        enumerate(selected_frames),
        total=len(selected_frames),
        desc="Extracting frames",
        unit="frame",
    ):
        if not sequential_read:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            logger.warning("Could not read frame %s", frame_idx)
            continue

        out_path = out_dir / source_frame_image_name(video_name, int(frame_idx))
        if not cv2.imwrite(str(out_path), frame):
            logger.warning("Could not write %s", out_path)
            continue
        extracted_frames.append(
            ExtractedFrame(
                image_path=out_path,
                source_frame_index=int(frame_idx),
                extraction_index=extraction_index,
            )
        )

    cap.release()
    logger.info("Extracted %d frames into %s", len(extracted_frames), out_dir)
    return extracted_frames


def main():
    parser = argparse.ArgumentParser(description="Extract evenly spaced frames from a video.")
    parser.add_argument(
        "--video_path",
        type=Path,
        required=True,
        help="Path to input video file (e.g., movie.avi)",
    )
    parser.add_argument(
        "--out_dir", type=Path, required=True, help="Directory to save extracted frames"
    )
    parser.add_argument(
        "--extraction_fps",
        type=float,
        default=5,
        help="Target extraction FPS (default: 5)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=10000,
        help="Maximum number of frames to extract (default: 10000)",
    )
    parser.add_argument(
        "--all_frames",
        action="store_true",
        help="Extract every encoded source frame, up to --max_frames.",
    )

    args = parser.parse_args()
    configure_cli_logging()
    extract_selected_frames(
        args.video_path,
        args.out_dir,
        args.extraction_fps,
        args.max_frames,
        extract_all=args.all_frames,
    )


if __name__ == "__main__":
    main()
