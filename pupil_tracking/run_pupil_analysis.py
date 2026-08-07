# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 00:07:48 2025

@author: yzhao
"""

import argparse
import re
from importlib import resources
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from pupil_tracking.dataset import PupilDataset, resize_with_pad
from pupil_tracking.extract_frames import ExtractedFrame, extract_selected_frames
from pupil_tracking.tracking import build_tracking_dataframe, measure_probability_map
from pupil_tracking.unet import UNet

_IOU_RE = re.compile(r"iou=(0\.\d+)")
_NUMERIC_SUFFIX_RE = re.compile(r"(\d+)$")
_CENTER_MARKER_OPACITY = 0.35
_CENTER_MARKER_RADIUS = 3
_LEGACY_RESULT_SUFFIXES = (
    "estimated_pupil_diameter.csv",
    "estimated_pupil_diameter.png",
    "pupil_tracking.csv",
    "pupil_tracking_qc.png",
)

best = None
with resources.as_file(resources.files("pupil_tracking") / "checkpoints") as ckpt_dir:
    for p in ckpt_dir.glob("*.pth"):
        m = _IOU_RE.search(p.name)
        if not m:
            continue
        iou = float(m.group(1))
        if best is None or iou > best[0]:
            best = (iou, p)
assert best is not None, "No packaged checkpoints found."
DEFAULT_CHECKPOINT = best[1]


def _numeric_suffix(path: Path) -> tuple[int, str]:
    match = _NUMERIC_SUFFIX_RE.search(path.stem)
    if match:
        return int(match.group(1)), path.name
    return 0, path.name


def _frames_from_image_directory(image_dir: Path) -> list[ExtractedFrame]:
    image_paths = sorted(image_dir.glob("*.png"), key=_numeric_suffix)
    suffix_matches = [_NUMERIC_SUFFIX_RE.search(path.stem) for path in image_paths]
    if image_paths and all(match is not None for match in suffix_matches):
        suffixes = [int(match.group(1)) for match in suffix_matches if match is not None]
        index_offset = 1 if min(suffixes) >= 1 else 0
        source_frame_indices = [suffix - index_offset for suffix in suffixes]
    else:
        source_frame_indices = list(range(len(image_paths)))

    return [
        ExtractedFrame(
            image_path=image_path,
            source_frame_index=source_frame_indices[index],
            extraction_index=index,
        )
        for index, image_path in enumerate(image_paths)
    ]


def _read_encoded_video_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video_path}")
    encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if encoded_fps <= 0:
        raise ValueError(f"Video reports an invalid FPS: {encoded_fps}")
    return encoded_fps


def _encode_thresholded_confidence(
    probability_map: np.ndarray,
    pred_thresh: float,
) -> np.ndarray:
    """Encode threshold-passing confidence from yellow (low) to red (high)."""
    probabilities = np.asarray(probability_map, dtype=np.float32).squeeze()
    if probabilities.ndim != 2:
        raise ValueError("Probability map must be two-dimensional after squeezing.")
    if not 0 < pred_thresh < 1:
        raise ValueError("Prediction threshold must be between 0 and 1.")

    passed = probabilities > pred_thresh
    encoded = np.zeros(probabilities.shape, dtype=np.uint8)
    normalized = (probabilities[passed] - pred_thresh) / (1.0 - pred_thresh)
    encoded[passed] = np.clip(np.rint(normalized * 254.0) + 1.0, 1.0, 255.0).astype(np.uint8)
    return encoded


def _confidence_heatmap_colors(encoded_confidence: np.ndarray) -> np.ndarray:
    """Map encoded confidence to yellow, orange, and red RGB anchors."""
    normalized = (encoded_confidence.astype(np.float32) - 1.0) / 254.0
    green = np.where(
        normalized <= 0.5,
        255.0 - 180.0 * normalized,
        330.0 * (1.0 - normalized),
    )
    return np.column_stack(
        [
            np.full(normalized.shape, 255.0),
            np.clip(np.rint(green), 0.0, 255.0),
            np.zeros(normalized.shape),
        ]
    ).astype(np.uint8)


def _save_mask_overlays(
    image_frames: list[ExtractedFrame],
    confidence_maps: dict[str, np.ndarray],
    output_mask_dir: Path,
    mask_transparency: float,
    tracking_dataframe: pd.DataFrame | None = None,
) -> None:
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    tracking_rows = {}
    if tracking_dataframe is not None:
        tracking_rows = tracking_dataframe.set_index("image_name").to_dict(orient="index")

    for frame in image_frames:
        image_name = frame.image_path.name
        if image_name not in confidence_maps:
            continue

        original = Image.open(frame.image_path).convert("L")
        resized = resize_with_pad(original, target_size=148)
        grayscale = np.asarray(resized, dtype=np.uint8)
        rgb = np.stack([grayscale] * 3, axis=-1)
        confidence_map = confidence_maps[image_name]
        if confidence_map.shape != grayscale.shape:
            raise ValueError("Confidence map shape must match the resized image shape.")
        mask = confidence_map > 0

        tracking_row = tracking_rows.get(image_name)
        is_valid = tracking_row is None or bool(tracking_row["segmentation_valid"])
        blended = rgb.copy()
        if mask.any():
            heatmap_colors = _confidence_heatmap_colors(confidence_map[mask])
            blended[mask] = (
                (1 - mask_transparency) * rgb[mask] + mask_transparency * heatmap_colors
            ).astype(np.uint8)
        overlay_image = Image.fromarray(blended)

        if tracking_row is not None and np.isfinite(tracking_row["raw_center_x_model_pixels"]):
            center_x = float(tracking_row["raw_center_x_model_pixels"])
            center_y = float(tracking_row["raw_center_y_model_pixels"])
            center_color = "cyan" if is_valid else "yellow"
            marker_layer = overlay_image.copy()
            draw = ImageDraw.Draw(marker_layer)
            draw.line(
                (
                    center_x - _CENTER_MARKER_RADIUS,
                    center_y,
                    center_x + _CENTER_MARKER_RADIUS,
                    center_y,
                ),
                fill=center_color,
                width=1,
            )
            draw.line(
                (
                    center_x,
                    center_y - _CENTER_MARKER_RADIUS,
                    center_x,
                    center_y + _CENTER_MARKER_RADIUS,
                ),
                fill=center_color,
                width=1,
            )
            overlay_image = Image.blend(
                overlay_image,
                marker_layer,
                _CENTER_MARKER_OPACITY,
            )

        overlay_image.save(output_mask_dir / image_name)


def generate_pupil_predictions(
    checkpoint_path,
    image_frames: list[ExtractedFrame],
    output_mask_dir: Path = None,
    pred_thresh: float = 0.7,
    batch_size: int = 32,
    mask_transparency: float = 0.1,
    calculate_velocity: bool = False,
    acquisition_fps: float = None,
):
    """Run inference and optionally build pupil-center tracking measurements."""
    if not image_frames:
        raise FileNotFoundError("No PNG images were provided for pupil analysis.")
    if calculate_velocity and (acquisition_fps is None or acquisition_fps <= 0):
        raise ValueError("A positive acquisition FPS is required for velocity calculation.")

    image_paths = [frame.image_path for frame in image_frames]
    frame_by_name = {frame.image_path.name: frame for frame in image_frames}

    print(f"Building dataloader with batch size = {batch_size}.")
    test_dataset = PupilDataset(image_paths)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Loading UNet model...")
    model = UNet(use_attention=True)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Using inference device: {device}.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    legacy_results = []
    tracking_measurements = []
    confidence_maps = {}

    progress = tqdm(total=len(test_dataset), desc="Segmenting pupil images...", unit="image")
    with torch.inference_mode():
        for images, names in test_loader:
            image_count = images.size(0)
            images = images.to(device)
            probabilities = torch.sigmoid(model(images)).cpu().numpy()
            batch_binary_masks = probabilities > pred_thresh
            pupil_diameters = np.sqrt(np.sum(batch_binary_masks, axis=(1, 2, 3)) * 1.27)

            for index, (name, diameter) in enumerate(zip(names, pupil_diameters)):
                diameter = float(diameter)
                legacy_results.append((name, diameter))
                frame = frame_by_name[name]
                probability_map = probabilities[index].squeeze()

                if calculate_velocity:
                    with Image.open(frame.image_path) as original:
                        original_size = original.size
                    measurement, _, _ = measure_probability_map(
                        probability_map,
                        pred_thresh=pred_thresh,
                        original_size=original_size,
                    )
                    measurement.update(
                        {
                            "image_name": name,
                            "source_frame_index": frame.source_frame_index,
                            "estimated_pupil_diameter": diameter,
                        }
                    )
                    tracking_measurements.append(measurement)
                if output_mask_dir is not None:
                    confidence_maps[name] = _encode_thresholded_confidence(
                        probability_map,
                        pred_thresh,
                    )

            progress.update(image_count)
    progress.close()

    tracking_dataframe = None
    if calculate_velocity:
        tracking_dataframe = build_tracking_dataframe(
            tracking_measurements,
            acquisition_fps=acquisition_fps,
        )

    if output_mask_dir is not None:
        _save_mask_overlays(
            image_frames,
            confidence_maps,
            output_mask_dir,
            mask_transparency,
            tracking_dataframe=tracking_dataframe,
        )

    return legacy_results, tracking_dataframe


def generate_pupil_mask_prediction(
    checkpoint_path,
    image_dir: Path,
    output_mask_dir: Path = None,
    pred_thresh: float = 0.7,
    batch_size: int = 32,
    mask_transparency: float = 0.1,
):
    """Compatibility wrapper for the original diameter-only inference function."""
    image_frames = _frames_from_image_directory(Path(image_dir))
    legacy_results, _ = generate_pupil_predictions(
        checkpoint_path,
        image_frames,
        output_mask_dir=output_mask_dir,
        pred_thresh=pred_thresh,
        batch_size=batch_size,
        mask_transparency=mask_transparency,
        calculate_velocity=False,
    )
    return legacy_results


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
        image_frames = _frames_from_image_directory(args.image_dir)
    else:
        raise ValueError("You must specify either (--video_path) or (--image_dir).")

    if not image_frames:
        raise FileNotFoundError(f"No PNG files found in {args.image_dir}")

    if args.calculate_velocity:
        print(f"Using acquisition rate: {args.acquisition_fps:.10g} samples/s")

    if args.output_mask_dir is not None:
        args.output_mask_dir.mkdir(parents=True, exist_ok=True)

    results, tracking_dataframe = generate_pupil_predictions(
        args.checkpoint,
        image_frames,
        output_mask_dir=args.output_mask_dir,
        pred_thresh=args.pred_thresh,
        batch_size=args.batch_size,
        mask_transparency=args.mask_transparency,
        calculate_velocity=args.calculate_velocity,
        acquisition_fps=args.acquisition_fps,
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
