import numpy as np
import pytest
from PIL import Image

from pupil_tracking.preprocessing import (
    MODEL_IMAGE_SIZE,
    model_to_input_length,
    resize_scale,
    resize_with_pad,
)


def test_resize_scale_matches_resize_with_pad_geometry():
    original = Image.fromarray(np.zeros((148, 296), dtype=np.uint8))
    padded = resize_with_pad(original, target_size=MODEL_IMAGE_SIZE)
    scale_x, scale_y, pad_left, pad_top = resize_scale(296, 148)

    assert padded.size == (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)
    assert scale_x == pytest.approx(148 / 296)
    assert scale_y == pytest.approx(74 / 148)
    assert pad_left == 0
    assert pad_top == (MODEL_IMAGE_SIZE - 74) // 2


def test_model_to_input_length_undoes_uniform_downscale():
    # A 296 x 148 frame is halved to fit the 148 px model image, so a length
    # measured in model pixels is twice as long in video pixels.
    assert model_to_input_length(10.0, 296, 148) == pytest.approx(20.0)
    assert model_to_input_length(10.0, 148, 296) == pytest.approx(20.0)


def test_model_to_input_length_is_identity_at_model_size():
    assert model_to_input_length(37.0, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE) == pytest.approx(37.0)


def test_model_to_input_length_scales_area_not_edge():
    # Non-square frames scale x and y slightly differently after integer rounding.
    # An area-derived length must use the geometric mean of both scales.
    scale_x, scale_y, _, _ = resize_scale(200, 160)
    expected = 10.0 / np.sqrt(scale_x * scale_y)
    assert model_to_input_length(10.0, 200, 160) == pytest.approx(expected)


def test_resize_scale_rejects_degenerate_input():
    with pytest.raises(ValueError):
        resize_scale(0, 100)
    with pytest.raises(ValueError):
        resize_scale(100, 100, target_size=0)
