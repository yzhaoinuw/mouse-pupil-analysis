import numpy as np

from mouse_pupil_analysis.pupil_predictions import select_central_component
from mouse_pupil_analysis.run_pupil_analysis import build_parser


def test_central_component_beats_a_larger_upper_artifact_without_shape_filtering():
    probabilities = np.zeros((148, 148), dtype=np.float32)
    probabilities[1:23, 58:104] = 0.79  # Larger upper artifact.
    probabilities[67:92, 70:95] = 0.93  # Smaller central pupil.
    binary_mask = probabilities > 0.5

    selected = select_central_component(probabilities, binary_mask)

    assert selected.sum() == 25 * 25
    assert selected[80, 82]
    assert not selected[10, 80]


def test_central_component_leaves_a_single_component_unchanged():
    probabilities = np.zeros((20, 20), dtype=np.float32)
    probabilities[1:4, 2:8] = 0.9
    binary_mask = probabilities > 0.5

    selected = select_central_component(probabilities, binary_mask)

    np.testing.assert_array_equal(selected, binary_mask)


def test_central_component_cli_option_is_opt_in():
    parser = build_parser()

    assert not parser.parse_args(["--image_dir", "frames"]).prefer_central_component
    assert parser.parse_args(
        ["--image_dir", "frames", "--prefer_central_component"]
    ).prefer_central_component


def test_overlay_transparency_defaults_to_five_percent():
    assert build_parser().parse_args(["--image_dir", "frames"]).mask_transparency == 0.05
