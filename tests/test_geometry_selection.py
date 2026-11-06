import numpy as np

from training._geometry_selection import select_predicted_center_component


def test_predicted_center_picks_nearby_whole_component():
    mask = np.zeros((48, 48), dtype=bool)
    mask[4:14, 3:13] = True
    mask[25:35, 31:41] = True
    result = select_predicted_center_component(mask * 0.8, mask, (40, 30))

    expected = np.zeros_like(mask)
    expected[25:35, 31:41] = True
    np.testing.assert_array_equal(result, expected)


def test_single_or_empty_masks_pass_through():
    mask = np.zeros((48, 48), dtype=bool)
    np.testing.assert_array_equal(select_predicted_center_component(mask, mask, (0, 0)), mask)
    mask[4:14, 3:13] = True
    np.testing.assert_array_equal(select_predicted_center_component(mask, mask, (45, 45)), mask)


def test_tiny_fragment_at_center_can_lose_to_supported_blob():
    mask = np.zeros((48, 48), dtype=bool)
    mask[4:14, 3:13] = True
    mask[30, 35] = True
    result = select_predicted_center_component(mask * 0.9, mask, (35, 30))
    assert result.sum() == 100
    assert not result[30, 35]
