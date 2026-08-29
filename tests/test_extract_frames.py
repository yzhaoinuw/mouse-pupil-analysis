import numpy as np

from mouse_pupil_analysis.extract_frames import select_frame_indices, source_frame_image_name


def test_source_frame_image_name_is_one_based():
    assert source_frame_image_name("eye", 0) == "eye_00001.png"
    assert source_frame_image_name("eye", 19) == "eye_00020.png"


def test_select_frame_indices_preserves_existing_rate_sampling():
    selected = select_frame_indices(
        frame_count=100,
        encoded_fps=10,
        extraction_fps=5,
        max_frames=10000,
    )

    assert len(selected) == 50
    assert selected[0] == 0
    assert selected[-1] == 99
    assert np.all(np.diff(selected) > 0)


def test_select_frame_indices_returns_every_frame_when_requested():
    selected = select_frame_indices(
        frame_count=10,
        encoded_fps=100,
        extraction_fps=5,
        max_frames=10000,
        extract_all=True,
    )

    assert np.array_equal(selected, np.arange(10))


def test_select_frame_indices_keeps_capped_velocity_frames_consecutive():
    selected = select_frame_indices(
        frame_count=100,
        encoded_fps=100,
        extraction_fps=5,
        max_frames=10,
        extract_all=True,
    )

    assert np.array_equal(selected, np.arange(10))
