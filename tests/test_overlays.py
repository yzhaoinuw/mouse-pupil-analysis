from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from mouse_pupil_analysis.extract_frames import ExtractedFrame
from mouse_pupil_analysis.preprocessing import resize_with_pad
from mouse_pupil_analysis.pupil_predictions import (
    MaskOverlayAccumulator,
    PupilPrediction,
    _encode_thresholded_confidence,
    _save_mask_overlays,
)


def test_confidence_heatmap_and_center_marker_remain_translucent(tmp_path: Path):
    frame_path = tmp_path / "eye_00001.png"
    Image.fromarray(np.full((148, 148), 100, dtype=np.uint8)).save(frame_path)
    frame = ExtractedFrame(frame_path, source_frame_index=0, extraction_index=0)

    probability_map = np.zeros((148, 148), dtype=np.float32)
    probability_map[70, 69] = 0.7
    probability_map[70, 70] = 0.7001
    probability_map[70, 71] = 0.85
    probability_map[70, 72] = 1.0
    confidence_map = _encode_thresholded_confidence(probability_map, pred_thresh=0.7)
    tracking = pd.DataFrame(
        [
            {
                "image_name": frame_path.name,
                "segmentation_valid": True,
                "raw_center_x_model_pixels": 74.0,
                "raw_center_y_model_pixels": 74.0,
            }
        ]
    )

    output_dir = tmp_path / "overlays"
    _save_mask_overlays(
        [frame],
        {frame_path.name: confidence_map},
        output_dir,
        mask_transparency=0.1,
        tracking_dataframe=tracking,
    )

    overlay = np.asarray(Image.open(output_dir / frame_path.name).convert("RGB"))
    np.testing.assert_array_equal(overlay[0, 0], [100, 100, 100])
    np.testing.assert_array_equal(overlay[70, 69], [100, 100, 100])
    np.testing.assert_array_equal(overlay[70, 70], [115, 115, 90])
    np.testing.assert_array_equal(overlay[70, 71], [115, 106, 90])
    np.testing.assert_array_equal(overlay[70, 72], [115, 90, 90])
    assert not np.array_equal(overlay[74, 74], [100, 100, 100])
    assert not np.array_equal(overlay[74, 74], [0, 255, 255])


def test_confidence_heatmap_can_be_limited_to_a_selected_component():
    probability_map = np.zeros((10, 10), dtype=np.float32)
    probability_map[2, 2] = 0.9
    probability_map[7, 7] = 0.9
    selected_component = np.zeros((10, 10), dtype=bool)
    selected_component[7, 7] = True

    confidence_map = _encode_thresholded_confidence(
        probability_map,
        pred_thresh=0.7,
        binary_mask=selected_component,
    )

    assert confidence_map[2, 2] == 0
    assert confidence_map[7, 7] > 0


def test_overlay_preserves_the_full_source_with_resize_and_padding(tmp_path: Path):
    source = np.indices((203, 249), dtype=np.uint8)[1]
    frame_path = tmp_path / "eye_00001.png"
    Image.fromarray(source).save(frame_path)
    frame = ExtractedFrame(frame_path, source_frame_index=0, extraction_index=0)
    confidence_map = np.zeros((148, 148), dtype=np.uint8)
    confidence_map[74, 74] = 255

    _save_mask_overlays(
        [frame],
        {frame_path.name: confidence_map},
        tmp_path / "overlays",
        mask_transparency=0.1,
    )

    overlay = np.asarray(Image.open(tmp_path / "overlays" / frame_path.name).convert("RGB"))
    assert overlay.shape == (148, 148, 3)
    background = np.array(resize_with_pad(Image.fromarray(source)))
    expected = np.repeat(background[:, :, None], 3, axis=2)
    expected[74, 74] = (0.9 * expected[74, 74] + 0.1 * np.array([255, 0, 0])).astype(np.uint8)
    np.testing.assert_array_equal(overlay, expected)
    assert np.all(overlay[:13] == 0)  # Letterbox padding, not cropped source pixels.
    assert overlay[74, 0, 0] < 5
    assert overlay[74, -1, 0] > 240  # Both source edges remain visible.


def test_all_heads_preserves_confidence_and_only_outlines_the_retained_blob(tmp_path):
    path = tmp_path / "eye_00001.png"
    Image.new("L", (148, 148), color=100).save(path)
    frame = ExtractedFrame(path, source_frame_index=0, extraction_index=0)
    probabilities = np.zeros((148, 148), dtype=np.float32)
    probabilities[20:30, 20:30] = 0.75
    probabilities[65:85, 65:85] = 0.85
    retained = np.zeros_like(probabilities, dtype=bool)
    retained[65:85, 65:85] = True
    prediction = PupilPrediction(
        frame=frame,
        probability_map=probabilities,
        binary_mask=retained,
        estimated_pupil_diameter=0,
        original_size=(148, 148),
        geometry_circle=(120, 120, 6),
    )
    all_heads = MaskOverlayAccumulator(tmp_path / "all", 0.7, 0.35, show_all_heads=True)
    all_heads.add(prediction)
    all_heads.save([frame])
    _save_mask_overlays(
        [frame],
        {path.name: _encode_thresholded_confidence(probabilities, 0.7)},
        tmp_path / "raw",
        mask_transparency=0.35,
    )
    combined = np.array(Image.open(tmp_path / "all" / path.name))
    raw = np.array(Image.open(tmp_path / "raw" / path.name))
    # The unselected blob is still completely visible with exactly the same
    # confidence colors as the standalone raw segmentation overlay.
    np.testing.assert_array_equal(combined[15:35, 15:35], raw[15:35, 15:35])
    np.testing.assert_array_equal(combined[70:80, 70:80], raw[70:80, 70:80])
    changed = np.any(combined[:100, :100] != raw[:100, :100], axis=-1)
    perimeter = retained & ~cv2.erode(retained.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
        bool
    )
    assert changed.any()
    assert not changed[~perimeter[:100, :100]].any()
    # Both drawn and untouched perimeter pixels demonstrate a dashed outline.
    assert changed[perimeter[:100, :100]].any()
    assert not changed[perimeter[:100, :100]].all()
    # The circle and center retain the standalone geometry style.
    np.testing.assert_array_equal(combined[120, 120], [255, 255, 0])


def test_empty_all_heads_matches_geometry_only_without_a_ghost_contour(tmp_path):
    path = tmp_path / "empty_00001.png"
    Image.new("L", (148, 148), color=100).save(path)
    frame = ExtractedFrame(path, source_frame_index=0, extraction_index=0)
    prediction = PupilPrediction(
        frame=frame,
        probability_map=np.zeros((148, 148)),
        binary_mask=np.zeros((148, 148), dtype=bool),
        estimated_pupil_diameter=0,
        original_size=(148, 148),
        geometry_circle=(74, 74, 10),
    )
    for name, options in [
        ("all", {"show_all_heads": True}),
        ("geometry", {"geometry_head_only": True}),
    ]:
        accumulator = MaskOverlayAccumulator(tmp_path / name, 0.7, 0.35, **options)
        accumulator.add(prediction)
        accumulator.save([frame])
    assert (tmp_path / "all" / path.name).read_bytes() == (
        tmp_path / "geometry" / path.name
    ).read_bytes()
