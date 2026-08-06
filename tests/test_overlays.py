from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pupil_tracking.extract_frames import ExtractedFrame
from pupil_tracking.run_pupil_analysis import _save_mask_overlays


def test_mask_and_center_marker_remain_translucent(tmp_path: Path):
    frame_path = tmp_path / "eye_00000.png"
    Image.fromarray(np.full((148, 148), 100, dtype=np.uint8)).save(frame_path)
    frame = ExtractedFrame(frame_path, source_frame_index=0, extraction_index=0)

    mask = np.zeros((148, 148), dtype=np.uint8)
    mask[70, 70] = 1
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
        {frame_path.name: mask},
        output_dir,
        mask_transparency=0.1,
        tracking_dataframe=tracking,
    )

    overlay = np.asarray(Image.open(output_dir / frame_path.name).convert("RGB"))
    np.testing.assert_array_equal(overlay[0, 0], [100, 100, 100])
    np.testing.assert_array_equal(overlay[70, 70], [115, 90, 90])
    assert not np.array_equal(overlay[74, 74], [100, 100, 100])
    assert not np.array_equal(overlay[74, 74], [0, 255, 255])
