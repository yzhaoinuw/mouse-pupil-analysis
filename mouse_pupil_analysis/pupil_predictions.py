# -*- coding: utf-8 -*-
"""Generate pupil-segmentation predictions from PNG images.

The public functions in this module can be imported by another workflow or run
directly using the editable configuration at the bottom of the file.
"""

import json
import logging
import math
import os
import re
import warnings
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

from mouse_pupil_analysis.extract_frames import ExtractedFrame
from mouse_pupil_analysis.preprocessing import (
    MODEL_IMAGE_SIZE,
    InferenceDataset,
    model_to_input_length,
    resize_with_pad,
)
from mouse_pupil_analysis.unet import UNet

logger = logging.getLogger(__name__)

_IOU_RE = re.compile(r"iou=(0\.\d+)")
_THRESHOLD_RE = re.compile(r"(?:pred_)?thresh(?:old)?=(0(?:\.\d+)?|1(?:\.0+)?)")
_NUMERIC_SUFFIX_RE = re.compile(r"(\d+)$")
_CENTER_MARKER_OPACITY = 0.35
_CENTER_MARKER_RADIUS = 3
DEFAULT_PREDICTION_THRESHOLD = 0.7

# A binary mask gives pupil area; the reported diameter is that of the circle with
# the same area, so diameter = sqrt(4 / pi * area).
_EQUIVALENT_CIRCLE_FACTOR = 4.0 / math.pi


def default_num_workers() -> int:
    """Cap dataloader workers at the historical default without oversubscribing."""
    return min(4, os.cpu_count() or 1)


@dataclass(frozen=True)
class PupilPrediction:
    """One transient model prediction and its frame metadata."""

    frame: ExtractedFrame
    probability_map: np.ndarray
    binary_mask: np.ndarray
    estimated_pupil_diameter: float
    original_size: tuple[int, int]

    @property
    def image_name(self) -> str:
        return self.frame.image_path.name

    @property
    def pupil_diameter_input_pixels(self) -> float:
        """Equivalent-circle diameter expressed in *input-image* pixels.

        ``estimated_pupil_diameter`` is measured in the 148 x 148 model image, whose
        scale depends on the input frame's dimensions. This property inverts the
        resize-and-pad geometry to restore the scale of the image that was supplied.

        The unit is the supplied image, not necessarily the source video. Frames
        extracted from a video are full source frames, so the two coincide. Frames
        passed through ``image_dir`` are whatever the caller prepared: if they were
        already cropped or resized to 148 x 148, this equals
        ``estimated_pupil_diameter``.

        Pixels are still not physically calibrated. Comparing across recordings is
        only meaningful when their optics and working distance match, or after
        applying a per-recording scale factor this package does not infer.
        """
        width, height = self.original_size
        return model_to_input_length(self.estimated_pupil_diameter, width, height)


def find_default_checkpoint() -> Path:
    """Return the packaged checkpoint with the highest IoU in its filename."""
    checkpoint_dir = resources.files("mouse_pupil_analysis") / "checkpoints"
    best: tuple[float, str] | None = None
    for entry in checkpoint_dir.iterdir():
        if not entry.name.endswith(".pth"):
            continue
        match = _IOU_RE.search(entry.name)
        if not match:
            continue
        iou = float(match.group(1))
        if best is None or iou > best[0]:
            best = (iou, entry.name)

    if best is None:
        raise FileNotFoundError("No packaged checkpoints found.")

    # torch.load needs a real filesystem path. Resolving it here, rather than inside
    # a resources.as_file() block, avoids handing back a path whose temporary file
    # has already been cleaned up.
    checkpoint = checkpoint_dir / best[1]
    if not hasattr(checkpoint, "__fspath__"):
        raise RuntimeError(
            "The packaged checkpoint is not available as a real file, which happens "
            "when the package is imported from a zipped archive. Install "
            "mouse-pupil-analysis normally, or pass an explicit checkpoint path."
        )
    return Path(checkpoint)


def resolve_prediction_threshold(
    checkpoint_path: Path,
    requested_threshold: float | None = None,
    fallback: float = DEFAULT_PREDICTION_THRESHOLD,
) -> float:
    """Resolve an explicit, calibrated-metadata, filename, or fallback threshold."""
    checkpoint_path = Path(checkpoint_path)
    if requested_threshold is not None:
        threshold = float(requested_threshold)
        if not 0 < threshold < 1:
            raise ValueError("Prediction threshold must be between 0 and 1.")
        return threshold

    metadata_path = checkpoint_path.with_suffix(".json")
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            threshold = float(metadata["prediction_threshold"])
            if not 0 < threshold < 1:
                raise ValueError("calibrated threshold is outside (0, 1)")
            logger.info(
                "Using calibrated prediction threshold %.3g from %s", threshold, metadata_path
            )
            return threshold
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            logger.warning("Ignoring invalid threshold metadata in %s: %s", metadata_path, error)

    match = _THRESHOLD_RE.search(checkpoint_path.name)
    if match:
        threshold = float(match.group(1))
        if 0 < threshold < 1:
            logger.info("Using prediction threshold %.3g from the checkpoint filename", threshold)
            return threshold

    if not 0 < fallback < 1:
        raise ValueError("Fallback prediction threshold must be between 0 and 1.")
    logger.info(
        "Checkpoint has no calibrated threshold metadata; using fallback %.3g",
        fallback,
    )
    return float(fallback)


def __getattr__(name: str):
    """Resolve ``DEFAULT_CHECKPOINT`` on first access rather than at import time."""
    if name == "DEFAULT_CHECKPOINT":
        return find_default_checkpoint()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_unet_checkpoint(checkpoint_path: Path, device: torch.device) -> UNet:
    """Load a checkpoint into a matching UNet, in eval mode on ``device``.

    Whether the checkpoint uses spatial attention is read from its own weights
    rather than assumed, so a custom ``--checkpoint`` trained without attention
    loads correctly instead of failing on a state-dict key mismatch.
    """
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"{checkpoint_path} does not contain a state dict. Export the model with "
            "torch.save(model.state_dict(), path)."
        )

    use_attention = any(key.startswith("att.") for key in state_dict)
    model = UNet(use_attention=use_attention)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(
            f"{checkpoint_path} is not compatible with this UNet architecture. "
            "It may come from a different model version. Original error: "
            f"{error}"
        ) from error

    logger.info("Loaded checkpoint %s (attention: %s).", Path(checkpoint_path).name, use_attention)
    model.to(device)
    model.eval()
    return model


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
        resized = resize_with_pad(original, target_size=MODEL_IMAGE_SIZE)
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
    pred_thresh: float | None = None,
    batch_size: int = 32,
    num_workers: int | None = None,
    show_progress: bool = False,
) -> Iterator[PupilPrediction]:
    """Yield predictions from one model pass without retaining float maps.

    ``show_progress`` is off by default so importing code stays quiet. The console
    scripts turn it on.
    """
    pred_thresh = resolve_prediction_threshold(checkpoint_path, pred_thresh)
    if num_workers is None:
        num_workers = default_num_workers()
    if not image_frames:
        raise FileNotFoundError("No PNG images were provided for pupil analysis.")

    image_paths = [frame.image_path for frame in image_frames]
    frame_by_name = {frame.image_path.name: frame for frame in image_frames}
    # Read each header once here so tracking and diameter conversion share the
    # lookup instead of reopening every frame downstream.
    size_by_name = {}
    for frame in image_frames:
        with Image.open(frame.image_path) as original:
            size_by_name[frame.image_path.name] = original.size

    logger.info("Building dataloader with batch size = %d.", batch_size)
    test_dataset = InferenceDataset(image_paths)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logger.info("Loading UNet model...")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    logger.info("Using inference device: %s.", device)
    model = load_unet_checkpoint(checkpoint_path, device)

    progress = tqdm(
        total=len(test_dataset),
        desc="Segmenting pupil images...",
        unit="image",
        disable=not show_progress,
    )
    try:
        with torch.inference_mode():
            for images, names in test_loader:
                images = images.to(device)
                probabilities = torch.sigmoid(model(images)).cpu().numpy()
                batch_binary_masks = probabilities > pred_thresh
                pupil_areas = np.sum(batch_binary_masks, axis=(1, 2, 3))
                pupil_diameters = np.sqrt(pupil_areas * _EQUIVALENT_CIRCLE_FACTOR)

                for index, (name, diameter) in enumerate(zip(names, pupil_diameters)):
                    yield PupilPrediction(
                        frame=frame_by_name[name],
                        probability_map=probabilities[index].squeeze(),
                        binary_mask=batch_binary_masks[index].squeeze(),
                        estimated_pupil_diameter=float(diameter),
                        original_size=size_by_name[name],
                    )
                    progress.update(1)
    finally:
        progress.close()


def generate_pupil_predictions(
    checkpoint_path: Path,
    image_frames: list[ExtractedFrame],
    output_mask_dir: Path | None = None,
    pred_thresh: float | None = None,
    batch_size: int = 32,
    mask_transparency: float = 0.1,
    show_progress: bool = False,
) -> list[tuple[str, float]]:
    """Generate diameter results and optional overlays without tracking."""
    pred_thresh = resolve_prediction_threshold(checkpoint_path, pred_thresh)
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
        show_progress=show_progress,
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
    pred_thresh: float | None = None,
    batch_size: int = 32,
    mask_transparency: float = 0.1,
) -> list[tuple[str, float]]:
    """Deprecated. Prefer :func:`mouse_pupil_analysis.analyze_frames`.

    Kept as a thin wrapper over ``generate_pupil_predictions`` for existing scripts.
    ``analyze_frames`` additionally writes the analysis table and plot and returns a
    DataFrame instead of a list of tuples.
    """
    warnings.warn(
        "generate_pupil_mask_prediction is deprecated; use mouse_pupil_analysis.analyze_frames "
        "for the full pipeline, or frames_from_image_directory plus "
        "generate_pupil_predictions for diameters only.",
        DeprecationWarning,
        stacklevel=2,
    )
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
    # Edit these values, then run this file directly.
    project_root = Path(__file__).resolve().parents[1]
    image_dir = project_root / "images_test_1"
    checkpoint_path = find_default_checkpoint()
    output_mask_dir = project_root / "predictions_test"
    pred_thresh = None  # Use checkpoint calibration; set a float to override it.
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
    logger.info("Generated predictions for %d images.", len(predictions))
