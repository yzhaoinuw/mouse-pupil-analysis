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
    return [
        ExtractedFrame(
            image_path=image_path,
            source_frame_index=index,
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


def _save_mask_overlays(
    image_frames: list[ExtractedFrame],
    binary_masks: dict[str, np.ndarray],
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
        if image_name not in binary_masks:
            continue

        original = Image.open(frame.image_path).convert("L")
        resized = resize_with_pad(original, target_size=148)
        grayscale = np.asarray(resized, dtype=np.uint8)
        rgb = np.stack([grayscale] * 3, axis=-1)
        mask = binary_masks[image_name].astype(bool)

        tracking_row = tracking_rows.get(image_name)
        is_valid = tracking_row is None or bool(tracking_row["segmentation_valid"])
        mask_color = np.array([255, 0, 0] if is_valid else [255, 165, 0], dtype=np.uint8)
        colored = rgb.copy()
        colored[mask] = mask_color
        blended = ((1 - mask_transparency) * rgb + mask_transparency * colored).astype(np.uint8)
        overlay_image = Image.fromarray(blended)

        if tracking_row is not None and np.isfinite(tracking_row["raw_center_x_model_pixels"]):
            center_x = float(tracking_row["raw_center_x_model_pixels"])
            center_y = float(tracking_row["raw_center_y_model_pixels"])
            center_color = "lime" if is_valid else "yellow"
            draw = ImageDraw.Draw(overlay_image)
            radius = 3
            draw.ellipse(
                (
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ),
                outline=center_color,
                width=2,
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    legacy_results = []
    tracking_measurements = []
    binary_masks = {}

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
                    measurement, binary_mask, _ = measure_probability_map(
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
                else:
                    binary_mask = batch_binary_masks[index].squeeze().astype(np.uint8)

                if output_mask_dir is not None:
                    binary_masks[name] = binary_mask

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
            binary_masks,
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


def save_results(results, result_dir: Path, exp_name: str):
    """Save the original diameter CSV and plot outputs."""
    results.sort(key=lambda result: _numeric_suffix(Path(result[0])))
    dataframe = pd.DataFrame(results, columns=["image_name", "estimated_pupil_diameter"])
    dataframe.index = np.arange(1, len(dataframe) + 1)
    csv_path = result_dir / f"{exp_name}_estimated_pupil_diameter.csv"
    dataframe.to_csv(csv_path, index=True)

    plt.figure(figsize=(10, 6))
    plt.plot(dataframe.index, dataframe["estimated_pupil_diameter"], linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Estimated Pupil Diameter (pixels)")
    plt.title("Estimated Pupil Diameter Over Time")
    plt.tight_layout()
    plot_path = result_dir / f"{exp_name}_estimated_pupil_diameter.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved plot: {plot_path}")


def save_tracking_results(
    tracking_dataframe: pd.DataFrame,
    result_dir: Path,
    exp_name: str,
) -> tuple[Path, Path]:
    """Save pupil-center measurements and a shared-time-axis quality-control plot."""
    preferred_columns = [
        "image_name",
        "source_frame_index",
        "timestamp_seconds",
        "center_x_pixels",
        "center_y_pixels",
        "raw_center_x_pixels",
        "raw_center_y_pixels",
        "displacement_x_pixels",
        "displacement_y_pixels",
        "velocity_x_pixels_per_second",
        "velocity_y_pixels_per_second",
        "speed_pixels_per_second",
        "estimated_pupil_diameter",
        "selected_component_area",
        "foreground_area",
        "component_count",
        "component_dominance",
        "mean_component_confidence",
        "component_circularity",
        "component_touches_border",
        "local_area_median",
        "area_to_local_median_ratio",
        "segmentation_valid",
        "quality_reason",
    ]
    remaining_columns = [
        column for column in tracking_dataframe.columns if column not in preferred_columns
    ]
    output_dataframe = tracking_dataframe[preferred_columns + remaining_columns]

    csv_path = result_dir / f"{exp_name}_pupil_tracking.csv"
    output_dataframe.to_csv(csv_path, index=False)

    time = output_dataframe["timestamp_seconds"]
    valid = output_dataframe["segmentation_valid"].astype(bool)
    figure, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(time, output_dataframe["estimated_pupil_diameter"], linewidth=0.8)
    axes[0].set_ylabel("Diameter\n(model pixels)")
    axes[0].set_title("Pupil Tracking Quality Control")

    axes[1].plot(time, output_dataframe["center_x_pixels"], label="x", linewidth=0.8)
    axes[1].plot(time, output_dataframe["center_y_pixels"], label="y", linewidth=0.8)
    axes[1].set_ylabel("Center\n(video pixels)")
    axes[1].legend(loc="upper right")

    axes[2].plot(time, output_dataframe["speed_pixels_per_second"], linewidth=0.8)
    axes[2].set_ylabel("Speed\n(pixels/s)")

    axes[3].step(time, valid.astype(int), where="mid", linewidth=0.8)
    axes[3].scatter(
        time[~valid],
        np.zeros((~valid).sum()),
        color="red",
        s=8,
        label="Rejected segmentation",
    )
    axes[3].set_yticks([0, 1], labels=["rejected", "valid"])
    axes[3].set_ylim(-0.2, 1.2)
    axes[3].set_xlabel("Actual acquisition time (seconds)")
    if (~valid).any():
        axes[3].legend(loc="lower right")

    figure.tight_layout()
    plot_path = result_dir / f"{exp_name}_pupil_tracking_qc.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)

    print(f"Saved tracking CSV:  {csv_path}")
    print(f"Saved tracking QC plot: {plot_path}")
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
    save_results(results, args.result_dir, exp_name)
    if tracking_dataframe is not None:
        save_tracking_results(tracking_dataframe, args.result_dir, exp_name)


if __name__ == "__main__":
    main()
