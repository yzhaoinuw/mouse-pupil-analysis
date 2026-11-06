from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from mouse_pupil_analysis.extract_frames import ExtractedFrame
from mouse_pupil_analysis.pupil_predictions import (
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


def test_overlay_uses_a_centered_source_image_crop(tmp_path: Path):
    source = np.indices((200, 200), dtype=np.uint8)[0]
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
    np.testing.assert_array_equal(overlay[0, 0], [26, 26, 26])
    np.testing.assert_array_equal(overlay[-1, -1], [173, 173, 173])
    np.testing.assert_array_equal(overlay[74, 74], [115, 90, 90])
