"""Verify experiment preflight on real files before expensive training starts."""

import json

import pytest
from PIL import Image

from scripts.run_loss_experiment import snapshot
from training._trainer import TrainingConfig


def make_pool(tmp_path, *, duplicate=False):
    images = []
    for index, (session, fold, holdout) in enumerate(
        (("train", 0, False), ("validation", 2, False), ("outer", -1, True))
    ):
        directory = tmp_path / "labeled_frames" / session
        (directory / "images").mkdir(parents=True)
        (directory / "masks").mkdir()
        Image.new("L", (4, 4), color=20 if duplicate else index + 20).save(
            directory / "images/frame.png"
        )
        mask = Image.new("P", (4, 4))
        mask.putpalette([0, 0, 0, 128, 0, 0] + [0] * (768 - 6))
        if session == "train":
            mask.putpixel((2, 2), 1)
        mask.save(directory / "masks/frame.png")
        images.append(
            {
                "key": f"{session}/frame",
                "session": session,
                "fold": fold,
                "holdout": holdout,
                "image": f"labeled_frames/{session}/images/frame.png",
                "mask": f"labeled_frames/{session}/masks/frame.png",
            }
        )
    manifest = tmp_path / "training_data_split.json"
    manifest.write_text(json.dumps({"schema": 2, "n_folds": 4, "images": images}), encoding="utf-8")
    return TrainingConfig(
        labeled_frames_dir=tmp_path / "labeled_frames", split_manifest=manifest, fold=2
    )


def test_preflight_accepts_palette_masks_and_records_negatives_and_exclusions(tmp_path):
    result = snapshot(make_pool(tmp_path))
    assert result["counts"] == {"train": 1, "validation": 1, "excluded": 1}
    assert result["empty_targets"] == {"train": 0, "validation": 1, "excluded": 1}
    assert next(r for r in result["pairs"] if r["role"] == "train")["mask_gray_values"] == [0, 38]


def test_preflight_rejects_identical_images_across_sessions_and_folds(tmp_path):
    config = make_pool(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="overlap"):
        snapshot(config)


def test_snapshot_changes_when_source_mask_changes(tmp_path):
    config = make_pool(tmp_path)
    before = snapshot(config)
    Image.new("L", (4, 4), 255).save(tmp_path / "labeled_frames/train/masks/frame.png")
    assert snapshot(config) != before
