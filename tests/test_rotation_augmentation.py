"""Check paired rotation and isolation from the original affine draws."""

import random

import numpy as np
from PIL import Image

from mouse_pupil_analysis import augmentation


def test_rotation_override_preserves_scale_draws(monkeypatch):
    calls = []

    def affine(image, *, angle, scale, **kwargs):
        calls.append((angle, scale))
        return image

    monkeypatch.setattr(augmentation.TF, "affine", affine)
    image = Image.new("L", (148, 148), 128)

    def replay(transform):
        calls.clear()
        random.seed(0)
        for _ in range(100):
            transform(image, image)
        assert calls[::2] == calls[1::2]
        return calls[::2], random.getstate()

    baseline, baseline_state = replay(augmentation.RandomAffinePair(degrees=15, p=0.2))
    candidate = augmentation.RandomAffinePair(degrees=15, p=0.2)
    candidate.override_rotation(45, 0.5, seed=0)
    stronger, candidate_state = replay(candidate)

    assert candidate_state == baseline_state
    assert [scale for _, scale in baseline] == [scale for _, scale in stronger if scale != 1]
    assert 0 < candidate.rotations_applied < candidate.rotation_draws == 100
    assert 15 < max(abs(angle) for angle, _ in stronger) <= 45
    assert any(scale != 1 and angle == 0 for angle, scale in stronger)


def test_rotation_keeps_image_and_binary_mask_aligned():
    source = np.zeros((148, 148), dtype=np.uint8)
    source[38:80, 50:89] = 255
    image = Image.fromarray(source)
    transform = augmentation.RandomAffinePair(p=0)
    transform.override_rotation(45, 1.0, seed=0)

    rotated_image, rotated_mask = transform(image, image)

    pixels, mask = np.asarray(rotated_image), np.asarray(rotated_mask)
    assert not np.array_equal(mask, source)
    assert set(np.unique(mask)) == {0, 255}
    assert ((pixels >= 128) != (mask > 0)).sum() < 10
