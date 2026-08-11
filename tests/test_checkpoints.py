import torch

from pupil_tracking.pupil_predictions import find_default_checkpoint, load_unet_checkpoint
from pupil_tracking.unet import UNet

DEVICE = torch.device("cpu")


def test_packaged_checkpoint_loads_with_attention():
    model = load_unet_checkpoint(find_default_checkpoint(), DEVICE)

    assert model.use_attention
    assert not model.training


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
    import pupil_tracking.pupil_predictions as module

    # DEFAULT_CHECKPOINT is served by module __getattr__, not a module-level constant,
    # so importing the module does no filesystem work.
    assert "DEFAULT_CHECKPOINT" not in vars(module)
    assert module.DEFAULT_CHECKPOINT.is_file()
