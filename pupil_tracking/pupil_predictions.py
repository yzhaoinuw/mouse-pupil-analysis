# -*- coding: utf-8 -*-
"""Generate pupil-segmentation predictions from PNG images.

The public functions in this module can be imported by another workflow or run
directly from Spyder using the editable configuration at the bottom of the file.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from pupil_tracking.dataset import PupilDataset, resize_with_pad
from pupil_tracking.extract_frames import ExtractedFrame
from pupil_tracking.unet import UNet

_IOU_RE = re.compile(r"iou=(0\.\d+)")
_NUMERIC_SUFFIX_RE = re.compile(r"(\d+)$")
_CENTER_MARKER_OPACITY = 0.35
_CENTER_MARKER_RADIUS = 3


@dataclass(frozen=True)
class PupilPrediction:
    """One transient model prediction and its frame metadata."""

    frame: ExtractedFrame
    probability_map: np.ndarray
    binary_mask: np.ndarray
    estimated_pupil_diameter: float

    @property
    def image_name(self) -> str:
        return self.frame.image_path.name


def find_default_checkpoint() -> Path:
    """Return the packaged checkpoint with the highest IoU in its filename."""
    best: tuple[float, Path] | None = None
    with resources.as_file(resources.files("pupil_tracking") / "checkpoints") as ckpt_dir:
        for checkpoint_path in ckpt_dir.glob("*.pth"):
            match = _IOU_RE.search(checkpoint_path.name)
            if not match:
                continue
            iou = float(match.group(1))
            if best is None or iou > best[0]:
                best = (iou, checkpoint_path)

    if best is None:
        raise FileNotFoundError("No packaged checkpoints found.")
    return best[1]


DEFAULT_CHECKPOINT = find_default_checkpoint()


def _numeric_suffix(path: Path) -> tuple[int, str]:
    match = _NUMERIC_SUFFIX_RE.search(path.stem)
    if match:
        return int(match.group(1)), path.name
    return 0, path.name


def frames_from_image_directory(image_dir: Path) -> list[ExtractedFrame]:
    """Build ordered frame metadata for the PNG files in an image directory."""
    image_paths = sorted(Path(image_dir).glob("*.png"), key=_numeric_suffix)
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


class MaskOverlayAccumulator:
    """Retain compact confidence maps until tracking-aware overlays can be saved."""

    def __init__(
        self,
        output_mask_dir: Path,
        pred_thresh: float,
        mask_transparency: float,
    ) -> None:
        self.output_mask_dir = Path(output_mask_dir)
        self.pred_thresh = pred_thresh
        self.mask_transparency = mask_transparency
        self.confidence_maps: dict[str, np.ndarray] = {}

    def add(self, prediction: PupilPrediction) -> None:
        """Encode one float probability map as a compact uint8 confidence map."""
        self.confidence_maps[prediction.image_name] = _encode_thresholded_confidence(
            prediction.probability_map,
            self.pred_thresh,
        )

    def save(
        self,
        image_frames: list[ExtractedFrame],
        tracking_dataframe: pd.DataFrame | None = None,
    ) -> None:
        """Write all accumulated overlays, optionally with tracking markers."""
        _save_mask_overlays(
            image_frames,
            self.confidence_maps,
            self.output_mask_dir,
            self.mask_transparency,
            tracking_dataframe=tracking_dataframe,
        )


def iter_pupil_predictions(
    checkpoint_path: Path,
    image_frames: list[ExtractedFrame],
    pred_thresh: float = 0.7,
    batch_size: int = 32,
) -> Iterator[PupilPrediction]:
    """Yield predictions from one model pass without retaining float maps."""
    if not image_frames:
        raise FileNotFoundError("No PNG images were provided for pupil analysis.")

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

    progress = tqdm(total=len(test_dataset), desc="Segmenting pupil images...", unit="image")
    try:
        with torch.inference_mode():
            for images, names in test_loader:
                images = images.to(device)
                probabilities = torch.sigmoid(model(images)).cpu().numpy()
                batch_binary_masks = probabilities > pred_thresh
                pupil_diameters = np.sqrt(np.sum(batch_binary_masks, axis=(1, 2, 3)) * 1.27)

                for index, (name, diameter) in enumerate(zip(names, pupil_diameters)):
                    yield PupilPrediction(
                        frame=frame_by_name[name],
                        probability_map=probabilities[index].squeeze(),
                        binary_mask=batch_binary_masks[index].squeeze(),
                        estimated_pupil_diameter=float(diameter),
                    )
                    progress.update(1)
    finally:
        progress.close()


def generate_pupil_predictions(
    checkpoint_path: Path,
    image_frames: list[ExtractedFrame],
    output_mask_dir: Path | None = None,
    pred_thresh: float = 0.7,
    batch_size: int = 32,
    mask_transparency: float = 0.1,
) -> list[tuple[str, float]]:
    """Generate diameter results and optional overlays without tracking."""
    overlay_accumulator = None
    if output_mask_dir is not None:
        overlay_accumulator = MaskOverlayAccumulator(
            output_mask_dir,
            pred_thresh,
            mask_transparency,
        )

    diameter_results = []
    for prediction in iter_pupil_predictions(
        checkpoint_path,
        image_frames,
        pred_thresh=pred_thresh,
        batch_size=batch_size,
    ):
        diameter_results.append((prediction.image_name, prediction.estimated_pupil_diameter))
        if overlay_accumulator is not None:
            overlay_accumulator.add(prediction)

    if overlay_accumulator is not None:
        overlay_accumulator.save(image_frames)
    return diameter_results


def generate_pupil_mask_prediction(
    checkpoint_path: Path,
    image_dir: Path,
    output_mask_dir: Path | None = None,
    pred_thresh: float = 0.7,
    batch_size: int = 32,
    mask_transparency: float = 0.1,
) -> list[tuple[str, float]]:
    """Generate diameter results for the PNG files in an image directory."""
    image_frames = frames_from_image_directory(Path(image_dir))
    return generate_pupil_predictions(
        checkpoint_path,
        image_frames,
        output_mask_dir=output_mask_dir,
        pred_thresh=pred_thresh,
        batch_size=batch_size,
        mask_transparency=mask_transparency,
    )


if __name__ == "__main__":
    # Spyder: edit these values, then use "Run file".
    project_root = Path(__file__).resolve().parents[1]
    image_dir = project_root / "images_test_1"
    checkpoint_path = DEFAULT_CHECKPOINT
    output_mask_dir = project_root / "predictions_test"
    pred_thresh = 0.7
    batch_size = 32
    mask_transparency = 0.1

    image_frames = frames_from_image_directory(image_dir)
    predictions = generate_pupil_predictions(
        checkpoint_path,
        image_frames,
        output_mask_dir=output_mask_dir,
        pred_thresh=pred_thresh,
        batch_size=batch_size,
        mask_transparency=mask_transparency,
    )
    print(f"Generated predictions for {len(predictions)} images.")
