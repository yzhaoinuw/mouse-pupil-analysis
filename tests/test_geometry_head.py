import pytest
import torch

from mouse_pupil_analysis.unet import UNet
from training._geometry import (
    GeometryUNet,
    circle_regression_loss,
    circle_targets,
    gate_probabilities,
)


def test_circle_targets_enclose_visible_pixels_and_skip_empty_masks():
    masks = torch.zeros(2, 1, 148, 148)
    masks[0, 0, 20, 10:21] = 1
    targets, present = circle_targets(masks)

    assert present.tolist() == [True, False]
    assert targets[0].tolist() == pytest.approx([15, 20, 5], abs=0.001)

    predicted = torch.zeros_like(targets, requires_grad=True)
    circle_regression_loss(predicted, masks).backward()
    assert predicted.grad[0].abs().sum() > 0
    assert predicted.grad[1].abs().sum() == 0
    assert circle_regression_loss(predicted, torch.zeros_like(masks)) == 0


def test_soft_gate_respects_xy_and_only_reduces_probabilities():
    probabilities = torch.full((2, 1, 48, 48), 0.75)
    circles = torch.tensor([[30, 15, 4], [9, 36, 5]])
    gated = gate_probabilities(probabilities, circles)

    assert gated.shape == probabilities.shape
    assert torch.all(gated <= probabilities)
    assert gated[0, 0, 15, 36] == 0.75
    assert gated[0, 0, 15, 38] == 0.375
    assert gated[0, 0, 15, 40] == 0
    assert gated[0, 0, 30, 15] == 0
    assert gated[1, 0, 36, 9] == 0.75


def test_geometry_model_preserves_mask_weights_and_rng():
    with torch.random.fork_rng(devices=[]):
        images = torch.zeros(1, 1, 148, 148)
        torch.manual_seed(0)
        baseline = UNet(use_attention=True).eval()
        expected_rng = torch.get_rng_state()
        torch.manual_seed(0)
        candidate = GeometryUNet(use_attention=True).eval()

        assert torch.equal(torch.get_rng_state(), expected_rng)
        with torch.inference_mode():
            logits, circles = candidate.forward_with_geometry(images)
            torch.testing.assert_close(logits, baseline(images), rtol=0, atol=0)
            torch.testing.assert_close(logits, candidate(images), rtol=0, atol=0)
        assert circles.shape == (1, 3)
        assert (circles > 0).all()
