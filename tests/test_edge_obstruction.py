"""Cover pupil safety and fresh, reproducible training-time draws."""

import numpy as np
import pytest
import torch

from training._edge_obstruction import EDGES, EdgeObstructionDataset, mottled_edge_band


def example():
    image = np.full((148, 148), 140, dtype=np.uint8)
    image[:8] = image[-8:] = 0
    pupil = np.zeros_like(image, dtype=bool)
    pupil[55:93, 55:93] = True
    return image, pupil


@pytest.mark.parametrize("edge", EDGES)
def test_band_changes_content_and_preserves_the_pupil_and_padding(edge):
    image, pupil = example()
    revised = mottled_edge_band(image, pupil, edge, np.random.default_rng(5))

    assert revised is not None
    changed = revised != image
    assert np.count_nonzero(changed) >= 220
    assert not np.any(changed & (pupil | (image == 0)))
    assert np.unique(revised[changed]).size > 2


def test_no_placement_fits_when_pupil_covers_visible_image():
    image, _ = example()
    assert mottled_edge_band(image, image > 0, "top", np.random.default_rng(0)) is None


def test_repeated_fetches_draw_new_bands_without_consuming_torch_randomness():
    image, pupil = example()
    image = torch.from_numpy(image)[None].float() / 255
    pupil = torch.from_numpy(pupil)[None].float()
    torch_state = torch.random.get_rng_state()
    outputs = []
    for _ in range(2):
        dataset = EdgeObstructionDataset([(image, pupil)], probability=0.3, seed=0)
        draws = [dataset[0] for _ in range(24)]
        assert all(target is pupil for _, target in draws)
        unchanged = [torch.equal(augmented, image) for augmented, _ in draws]
        assert any(unchanged) and not all(unchanged)
        bands = [augmented.numpy().tobytes() for augmented, _ in draws if augmented is not image]
        assert len(set(bands)) > 1
        outputs.append([augmented.numpy().tobytes() for augmented, _ in draws])

    assert outputs[0] == outputs[1]
    assert torch.equal(torch_state, torch.random.get_rng_state())
