"""The demo video: the overlay image at the size the camera recorded, encoded as H.264."""

import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from mouse_pupil_analysis.extract_frames import ExtractedFrame, iter_video_images
from mouse_pupil_analysis.preprocessing import MODEL_IMAGE_SIZE, resize_scale
from mouse_pupil_analysis.pupil_predictions import (
    _CENTER_MARKER_OPACITY,
    _CENTER_MARKER_RADIUS,
    PupilPrediction,
    _confidence_heatmap_colors,
    _encode_thresholded_confidence,
)

logger = logging.getLogger(__name__)

_FRAME_STEP = 3  # one video frame per three analyzed: a QC pass reads fine at a third of the disk
_CRF = 18  # visually lossless: compression must not blur the segmentation edge
_FILL = 0.15  # heatmap weight over the source: light, so the pupil under the mask stays readable
_CONTOUR_COLOR = (0, 255, 0)  # where the mask ends: the threshold boundary, one pixel wide
_CORE_COLOR = (255, 0, 0)  # where the model is sure, in the heatmap's own red
_CORE_LEVEL = 1 + round(0.8 * 254)  # 0.8 of the way from the threshold to 1.0, on the uint8 scale
_PROGRESS_EVERY = 2000


def analyzed_frame_rate(image_frames: Sequence[ExtractedFrame], acquisition_fps: float) -> float:
    """The rate the analyzed frames run at: the acquisition rate over their source spacing."""
    indices = np.array([frame.source_frame_index for frame in image_frames], dtype=float)
    if indices.size < 2:
        return float(acquisition_fps)
    spacing = float(np.median(np.diff(indices)))
    if spacing <= 0:
        return float(acquisition_fps)
    return float(acquisition_fps) / spacing


@contextmanager
def h264_writer(output_path: Path, fps: float):
    """An H.264 writer on ``<stem>.part<suffix>``, renamed to ``output_path`` on a clean exit."""
    import imageio.v2 as imageio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # a half-written .mp4 plays and looks complete; the suffix stays last for imageio's encoder choice
    partial_path = output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")
    writer = imageio.get_writer(
        str(partial_path),
        format="FFMPEG",
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", str(_CRF), "-preset", "medium"],
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    try:
        yield writer
    except BaseException:
        writer.close()
        partial_path.unlink(missing_ok=True)
        raise
    writer.close()
    partial_path.replace(output_path)


class DemoVideoWriter:
    """Keep every third confidence map, then draw and encode the demo video."""

    def __init__(
        self,
        output_path: Path,
        pred_thresh: float,
    ) -> None:
        self.output_path = Path(output_path)
        self.pred_thresh = pred_thresh
        self.confidence_maps: dict[str, np.ndarray] = {}

    def add(self, prediction: PupilPrediction) -> None:
        """Keep this map if ``extraction_index`` says the video shows its frame."""
        if prediction.frame.extraction_index % _FRAME_STEP:
            return
        self.confidence_maps[prediction.image_name] = _encode_thresholded_confidence(
            prediction.probability_map,
            self.pred_thresh,
            prediction.binary_mask,
        )

    def save(
        self,
        image_frames: Sequence[ExtractedFrame],
        segmentation_dataframe: pd.DataFrame,
        analyzed_fps: float,
        video_path: Path | None = None,
    ) -> Path | None:
        """Draw and encode the kept frames, one at a time; None if none was kept."""
        if not self.confidence_maps:
            logger.warning(
                "No frames were kept for the demo video (frame_step %d over %d frames).",
                _FRAME_STEP,
                len(image_frames),
            )
            return None

        started = time.perf_counter()
        marker_rows: Mapping[str, dict] = {}
        if segmentation_dataframe is not None and not segmentation_dataframe.empty:
            marker_rows = segmentation_dataframe.set_index("image_name").to_dict(orient="index")
        kept_frames = [
            frame for frame in image_frames if frame.image_path.name in self.confidence_maps
        ]

        fps_out = float(analyzed_fps) / _FRAME_STEP
        written = 0
        frame_shape = None
        with h264_writer(self.output_path, fps_out) as writer:
            for frame, original in _iter_frame_images(kept_frames, video_path):
                image_name = frame.image_path.name
                overlay = render_source_overlay(
                    original,
                    self.confidence_maps[image_name],
                    marker_rows.get(image_name),
                )
                video_frame = _even_sized(np.asarray(overlay.convert("RGB")))
                # one video holds one frame size, and the encoder was opened on the first
                if frame_shape is None:
                    frame_shape = video_frame.shape
                elif video_frame.shape != frame_shape:
                    raise ValueError(
                        f"{image_name} is {video_frame.shape[1]}x{video_frame.shape[0]}, but the "
                        f"video started at {frame_shape[1]}x{frame_shape[0]}. One video holds one "
                        "frame size; analyze these frames in separate runs."
                    )
                writer.append_data(video_frame)
                written += 1
                if not written % _PROGRESS_EVERY:
                    logger.info("Demo video: %d of %d frames.", written, len(kept_frames))

        elapsed = time.perf_counter() - started
        logger.info(
            "Saved demo video: %s (%d frames at %.4g fps, %.1f MB, %.0f s)",
            self.output_path,
            written,
            fps_out,
            self.output_path.stat().st_size / 1e6,
            elapsed,
        )
        return self.output_path


def _even_sized(frame: np.ndarray) -> np.ndarray:
    """Pad a row or column of black where the frame is odd; H.264 4:2:0 needs even sides."""
    height, width = frame.shape[:2]
    if not (height % 2 or width % 2):
        return frame
    padded = np.zeros((height + height % 2, width + width % 2, frame.shape[2]), dtype=frame.dtype)
    padded[:height, :width] = frame
    return padded


def _iter_frame_images(
    image_frames: Sequence[ExtractedFrame], video_path: Path | None
) -> Iterator[tuple[ExtractedFrame, Image.Image]]:
    """``(frame, grayscale image)`` pairs from the written PNGs, or decoded from the video."""
    if video_path is None:
        return ((frame, Image.open(frame.image_path).convert("L")) for frame in image_frames)
    return iter_video_images(video_path, image_frames)


def _draw_center_marker(
    overlay_image: Image.Image,
    center_x: float,
    center_y: float,
    is_valid: bool,
    radius: int,
) -> Image.Image:
    """The mask image's cross at the measured center; cyan accepted, yellow rejected."""
    center_color = "cyan" if is_valid else "yellow"
    marker_layer = overlay_image.copy()
    draw = ImageDraw.Draw(marker_layer)
    draw.line(
        (
            center_x - radius,
            center_y,
            center_x + radius,
            center_y,
        ),
        fill=center_color,
        width=1,
    )
    draw.line(
        (
            center_x,
            center_y - radius,
            center_x,
            center_y + radius,
        ),
        fill=center_color,
        width=1,
    )
    return Image.blend(
        overlay_image,
        marker_layer,
        _CENTER_MARKER_OPACITY,
    )


def render_source_overlay(
    original: Image.Image,
    confidence_map: np.ndarray,
    tracking_row: Mapping[str, object] | None = None,
) -> Image.Image:
    """A light heatmap, the mask boundary, its sure core and the center marker, on the source frame."""
    source = np.asarray(original.convert("L"), dtype=np.uint8)
    height, width = source.shape
    scale_x, scale_y, pad_left, pad_top = resize_scale(width, height, MODEL_IMAGE_SIZE)
    inner = confidence_map[
        pad_top : pad_top + int(round(height * scale_y)),
        pad_left : pad_left + int(round(width * scale_x)),
    ]
    # nearest: averaging a confidence value with the background would paint a halo
    full_map = cv2.resize(inner, (width, height), interpolation=cv2.INTER_NEAREST)

    rgb = np.stack([source] * 3, axis=-1)
    mask = full_map > 0
    blended = rgb.copy()
    if mask.any():
        heatmap_colors = _confidence_heatmap_colors(full_map[mask])
        blended[mask] = ((1 - _FILL) * rgb[mask] + _FILL * heatmap_colors).astype(np.uint8)
        # two boundaries: where the mask ends, and where the model is sure; the band is its doubt
        for level, color in ((1, _CONTOUR_COLOR), (_CORE_LEVEL, _CORE_COLOR)):
            contours, _ = cv2.findContours(
                (full_map >= level).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(blended, contours, -1, color, 1)
    overlay_image = Image.fromarray(blended)

    if tracking_row is not None and np.isfinite(tracking_row["raw_center_x_pixels"]):
        is_valid = bool(tracking_row["segmentation_valid"])
        # the same marker size relative to the eye, not the same number of pixels
        radius = max(_CENTER_MARKER_RADIUS, int(round(_CENTER_MARKER_RADIUS / scale_x)))
        overlay_image = _draw_center_marker(
            overlay_image,
            float(tracking_row["raw_center_x_pixels"]),
            float(tracking_row["raw_center_y_pixels"]),
            is_valid,
            radius,
        )

    return overlay_image
