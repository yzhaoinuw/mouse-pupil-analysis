import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training import import_labelme


def _annotation(source: Path, stem: str, label: str) -> None:
    source.mkdir(parents=True, exist_ok=True)
    Image.new("L", (13, 7), color=128).save(source / f"{stem}.png")
    payload = {
        "imagePath": f"{stem}.png",
        "imageData": "embedded-copy",
        "imageWidth": 13,
        "imageHeight": 7,
        "shapes": [
            {
                "label": label,
                "points": [[4.0, 2.0], [8.0, 2.0], [6.0, 5.0]],
                "shape_type": "polygon",
            }
        ],
    }
    (source / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def _mixed_batch(source: Path) -> None:
    _annotation(source, "session_a_00001", "pupil")
    _annotation(source, "session_a_00002", "no_visible_pupil")
    _annotation(source, "session_a_00003", "uncertain")


def test_dry_run_validates_without_writing(tmp_path):
    source = tmp_path / "annotations"
    _mixed_batch(source)

    assert (
        import_labelme.main(
            ["--source", str(source), "--session", "session_a", "--data-root", str(tmp_path)]
        )
        == 0
    )
    assert not (tmp_path / "labeled_frames/session_a").exists()


def test_apply_imports_masks_and_archives_uncertain_annotation(tmp_path):
    source = tmp_path / "annotations"
    _mixed_batch(source)
    plan = import_labelme.build_import_plan(source, tmp_path, "session_a")

    target = import_labelme.apply_import(plan, tmp_path, "session_a")

    assert {path.name for path in (target / "images").glob("*.png")} == {
        "frame_00001.png",
        "frame_00002.png",
    }
    assert {path.name for path in (target / "masks").glob("*.png")} == {
        "frame_00001.png",
        "frame_00002.png",
    }
    pupil = np.asarray(Image.open(target / "masks/frame_00001.png"))
    negative = np.asarray(Image.open(target / "masks/frame_00002.png"))
    assert pupil.any()
    assert not negative.any()

    uncertain_json = json.loads((target / "uncertain/frame_00003.json").read_text(encoding="utf-8"))
    assert uncertain_json["imagePath"] == "frame_00003.png"
    assert uncertain_json["imageData"] is None
    assert (target / "uncertain/frame_00003.png").is_file()
    assert not list((target / "images").glob("*.json"))


def test_existing_session_is_rejected_before_writing(tmp_path):
    source = tmp_path / "annotations"
    _mixed_batch(source)
    (tmp_path / "labeled_frames/session_a").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="existing session"):
        import_labelme.build_import_plan(source, tmp_path, "session_a")


def test_all_uncertain_batch_stays_outside_labelled_pool(tmp_path):
    source = tmp_path / "annotations"
    _annotation(source, "session_a_00001", "uncertain")

    with pytest.raises(ValueError, match="only uncertain"):
        import_labelme.build_import_plan(source, tmp_path, "session_a")


def test_apply_refreshes_splits_after_successful_import(tmp_path, monkeypatch):
    source = tmp_path / "annotations"
    _mixed_batch(source)
    refreshed = []
    monkeypatch.setattr(
        import_labelme,
        "refresh_split_manifest",
        lambda data_root: refreshed.append(Path(data_root)) or 0,
    )

    assert (
        import_labelme.main(
            [
                "--source",
                str(source),
                "--session",
                "session_a",
                "--data-root",
                str(tmp_path),
                "--apply",
            ]
        )
        == 0
    )
    assert refreshed == [tmp_path]
