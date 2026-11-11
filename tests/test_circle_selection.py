import numpy as np
import pytest

from mouse_pupil_analysis.geometry import select_circle_components
from mouse_pupil_analysis.tracking import measure_probability_map


def test_circle_keeps_multiple_whole_blobs_and_rejects_larger_outside_blob():
    mask = np.zeros((60, 60), dtype=bool)
    mask[27:30, 25:35] = True
    mask[32:35, 25:35] = True
    mask[5:20, 45:60] = True
    selected = select_circle_components(mask, (30, 30, 3))
    expected = mask.copy()
    expected[5:20, 45:60] = False
    np.testing.assert_array_equal(selected, expected)
    # Accepted pixels outside the circle survive with their component.
    assert selected[27, 25]


@pytest.mark.parametrize("area, retained", [(5, True), (6, False), (20, False)])
def test_circle_overlap_includes_exact_twenty_percent(area, retained):
    mask = np.zeros((30, 30), dtype=bool)
    mask[10, :area] = True
    selected = select_circle_components(mask, (0, 10, 0.25))
    assert int(selected.sum()) == (area if retained else 0)


def test_single_outside_blob_is_rejected_and_empty_mask_stays_empty():
    mask = np.zeros((20, 20), dtype=bool)
    np.testing.assert_array_equal(select_circle_components(mask, (10, 10, 3)), mask)
    mask[1:5, 1:5] = True
    assert not select_circle_components(mask, (10, 10, 3)).any()


@pytest.mark.parametrize("circle", [(10, 10, 0), (10, 10, -1), (np.nan, 10, 5)])
def test_invalid_circle_retains_no_foreground(circle):
    assert not select_circle_components(np.ones((20, 20), dtype=bool), circle).any()


def test_circle_input_validation():
    with pytest.raises(ValueError, match="two-dimensional"):
        select_circle_components(np.zeros(3), (1, 1, 1))
    with pytest.raises(ValueError, match="overlap"):
        select_circle_components(np.zeros((3, 3)), (1, 1, 1), min_overlap=0)


def test_union_measurement_uses_all_retained_pixels_and_reports_fragmentation():
    probabilities = np.zeros((40, 40), dtype=np.float32)
    probabilities[10:14, 10:14] = 0.95
    probabilities[10:14, 20:24] = 0.99
    mask = select_circle_components(probabilities > 0.525, (17, 12, 10))
    measurement, _, selected = measure_probability_map(
        probabilities, 0.525, (40, 40), mask, measure_all_components=True
    )
    assert selected.sum() == measurement["selected_component_area"] == 32
    assert measurement["component_count"] == 2
    assert measurement["component_dominance"] == 0.5
    expected_x = (11.5 * 0.95 + 21.5 * 0.99) / (0.95 + 0.99)
    assert measurement["raw_center_x_pixels"] == pytest.approx(expected_x)
    assert "low_component_dominance" in measurement["quality_reason"]
