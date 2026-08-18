"""Tests for collision-safe labelled-frame compaction."""

import json
import runpy
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPACT = runpy.run_path(str(PROJECT_ROOT / "training" / "compact_frame_names.py"))
build_plan = COMPACT["build_plan"]
apply_plan = COMPACT["apply_plan"]


def _pair(root: Path, session: str, name: str, annotation: bool = True) -> None:
    images = root / session / "images"
    masks = root / session / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    (images / f"{name}.png").write_bytes(b"image")
    (masks / f"{name}.png").write_bytes(b"mask")
    if annotation:
        (images / f"{name}.json").write_text(
            json.dumps({"imagePath": f"{name}.png", "shapes": []}), encoding="utf-8"
        )


def test_compaction_renames_the_pair_and_updates_labelme(tmp_path):
    labelled = tmp_path / "labeled_frames"
    _pair(labelled, "session_a", "verbose_recording_name_0450")

    plan = build_plan(labelled)
    apply_plan(plan)

    image = labelled / "session_a/images/frame_00450.png"
    mask = labelled / "session_a/masks/frame_00450.png"
    annotation = labelled / "session_a/images/frame_00450.json"
    assert image.read_bytes() == b"image"
    assert mask.read_bytes() == b"mask"
    assert json.loads(annotation.read_text(encoding="utf-8"))["imagePath"] == image.name


def test_numeric_aliases_cannot_collide(tmp_path):
    labelled = tmp_path / "labeled_frames"
    _pair(labelled, "session_a", "one_00001")
    _pair(labelled, "session_a", "two_1")

    with pytest.raises(ValueError, match="collision"):
        build_plan(labelled)


def test_missing_mask_is_rejected_before_any_rename(tmp_path):
    labelled = tmp_path / "labeled_frames"
    _pair(labelled, "session_a", "recording_00123", annotation=False)
    (labelled / "session_a/masks/recording_00123.png").unlink()

    with pytest.raises(FileNotFoundError, match="matching mask"):
        build_plan(labelled)
    assert (labelled / "session_a/images/recording_00123.png").is_file()
