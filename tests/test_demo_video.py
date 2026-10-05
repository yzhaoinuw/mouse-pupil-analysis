"""Unit coverage for the demo video's frame bookkeeping; the video itself is in test_end_to_end."""

from pathlib import Path

import numpy as np
import pandas as pd

from mouse_pupil_analysis.demo_video import DemoVideoWriter, _even_sized, analyzed_frame_rate
from mouse_pupil_analysis.extract_frames import ExtractedFrame
from mouse_pupil_analysis.pupil_predictions import PupilPrediction


def _prediction(extraction_index: int) -> PupilPrediction:
    name = f"eye_{extraction_index + 1:05d}.png"
    probability_map = np.zeros((148, 148), dtype=np.float32)
    probability_map[74, 74] = 0.9
    return PupilPrediction(
        frame=ExtractedFrame(Path(name), extraction_index, extraction_index),
        probability_map=probability_map,
        binary_mask=probability_map > 0.7,
        estimated_pupil_diameter=1.0,
        original_size=(200, 160),
    )


def test_frame_step_selects_on_extraction_index_not_arrival_order(tmp_path: Path):
    writer = DemoVideoWriter(tmp_path / "demo.mp4", pred_thresh=0.7)

    # deliberately out of order: a dataloader may return batches in any order
    for index in [4, 0, 5, 3, 6, 1, 2]:
        writer.add(_prediction(index))

    assert sorted(writer.confidence_maps) == ["eye_00001.png", "eye_00004.png", "eye_00007.png"]
    assert writer.confidence_maps["eye_00001.png"].dtype == np.uint8


def test_nothing_is_written_when_no_frame_was_kept(tmp_path: Path):
    writer = DemoVideoWriter(tmp_path / "demo.mp4", pred_thresh=0.7)

    path = writer.save([], pd.DataFrame(), analyzed_fps=30.0)

    assert path is None
    assert not (tmp_path / "demo.mp4").exists()


def test_even_sized_pads_an_odd_frame_and_leaves_an_even_one():
    """H.264 with 4:2:0 chroma needs even dimensions; the eye frame is whatever the ROI was."""
    odd = np.full((151, 269, 3), 7, dtype=np.uint8)
    padded = _even_sized(odd)

    assert padded.shape == (152, 270, 3)
    np.testing.assert_array_equal(padded[:151, :269], odd)
    assert padded[151:].max() == 0 and padded[:, 269:].max() == 0

    even = np.full((150, 268, 3), 7, dtype=np.uint8)
    assert _even_sized(even) is even


def test_analyzed_frame_rate_follows_the_frames_not_the_request():
    every_frame = [ExtractedFrame(Path(f"e_{i}.png"), i, i) for i in range(10)]
    every_third = [
        ExtractedFrame(Path(f"e_{i}.png"), i, index) for index, i in enumerate(range(0, 30, 3))
    ]

    assert analyzed_frame_rate(every_frame, 33.0) == 33.0
    assert analyzed_frame_rate(every_third, 33.0) == 11.0
    # one frame says nothing about spacing, so the acquisition rate stands
    assert analyzed_frame_rate(every_frame[:1], 33.0) == 33.0
