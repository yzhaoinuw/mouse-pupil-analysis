import json

import pytest
import torch

from mouse_pupil_analysis.pupil_predictions import (
    find_default_checkpoint,
    load_unet_checkpoint,
    resolve_prediction_threshold,
)
from mouse_pupil_analysis.unet import UNet

DEVICE = torch.device("cpu")


def test_packaged_checkpoint_loads_with_attention():
    checkpoint = find_default_checkpoint()
    model = load_unet_checkpoint(checkpoint, DEVICE)

    assert model.use_attention
    assert not model.training
    assert checkpoint.name == "166pupils_thresh=0.4_iou=0.8749.pth"
    assert resolve_prediction_threshold(checkpoint) == 0.4


def test_checkpoint_without_attention_loads(tmp_path):
    checkpoint = tmp_path / "plain.pth"
    torch.save(UNet(use_attention=False).state_dict(), checkpoint)

    model = load_unet_checkpoint(checkpoint, DEVICE)

    assert not model.use_attention


def test_incompatible_checkpoint_reports_a_useful_error(tmp_path):
    checkpoint = tmp_path / "wrong.pth"
    torch.save({"unexpected": torch.zeros(3)}, checkpoint)

    try:
        load_unet_checkpoint(checkpoint, DEVICE)
    except ValueError as error:
        assert "not compatible" in str(error)
    else:
        raise AssertionError("Expected a ValueError for an incompatible checkpoint.")


def test_default_checkpoint_is_resolved_lazily():
    import mouse_pupil_analysis.pupil_predictions as module

    # DEFAULT_CHECKPOINT is served by module __getattr__, not a module-level constant,
    # so importing the module does no filesystem work.
    assert "DEFAULT_CHECKPOINT" not in vars(module)
    assert module.DEFAULT_CHECKPOINT.is_file()


def test_threshold_resolves_from_training_metadata_before_filename(tmp_path):
    checkpoint = tmp_path / "model_thresh=0.7_iou=0.9.pth"
    checkpoint.touch()
    checkpoint.with_suffix(".json").write_text(
        json.dumps({"prediction_threshold": 0.55}),
        encoding="utf-8",
    )

    assert resolve_prediction_threshold(checkpoint) == 0.55
    assert resolve_prediction_threshold(checkpoint, requested_threshold=0.65) == 0.65


def test_threshold_uses_filename_then_fallback(tmp_path):
    assert resolve_prediction_threshold(tmp_path / "model_thresh=0.6_iou=0.9.pth") == 0.6
    assert resolve_prediction_threshold(tmp_path / "model.pth") == 0.7


def test_explicit_threshold_must_be_a_probability(tmp_path):
    with pytest.raises(ValueError, match="between 0 and 1"):
        resolve_prediction_threshold(tmp_path / "model.pth", requested_threshold=1.0)
