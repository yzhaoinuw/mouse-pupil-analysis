import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training import import_labelme


def _annotation(session: Path, stem: str, label: str) -> Path:
    images = session / "images"
    images.mkdir(parents=True, exist_ok=True)
    Image.new("L", (13, 7), color=128).save(images / f"{stem}.png")
    annotation = {
        "imagePath": f"{stem}.png",
        "imageWidth": 13,
        "imageHeight": 7,
        "shapes": [
            {
                "label": label,
                "points": [[5.0, 3.0], [5.1, 3.0], [5.0, 3.1]],
                "shape_type": "polygon",
            }
        ],
    }
    path = images / f"{stem}.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    return path


def test_no_visible_pupil_writes_an_image_sized_empty_mask(tmp_path):
    session = tmp_path / "session"
    _annotation(session, "frame_00001", "no_visible_pupil")

    annotation_path = session / "images/frame_00001.json"
    kind, annotation = import_labelme.annotation_kind(annotation_path)
    (session / "masks").mkdir()
    import_labelme.write_training_mask(
        annotation_path,
        annotation,
        kind,
        session / "masks/frame_00001.png",
    )

    mask = np.asarray(Image.open(session / "masks/frame_00001.png"))
    assert mask.shape == (7, 13)
    assert not mask.any()


def test_pupil_polygon_is_rasterized_as_foreground(tmp_path):
    session = tmp_path / "session"
    _annotation(session, "frame_00005", "pupil")

    annotation_path = session / "images/frame_00005.json"
    kind, annotation = import_labelme.annotation_kind(annotation_path)
    (session / "masks").mkdir()
    import_labelme.write_training_mask(
        annotation_path,
        annotation,
        kind,
        session / "masks/frame_00005.png",
    )

    mask = np.asarray(Image.open(session / "masks/frame_00005.png"))
    assert mask.shape == (7, 13)
    assert mask.max() == 255
    assert mask[3, 5] == 255


def test_uncertain_annotation_is_intentionally_excluded(tmp_path):
    session = tmp_path / "session"
    _annotation(session, "frame_00002", "uncertain")

    annotation_path = session / "images/frame_00002.json"
    assert import_labelme.annotation_kind(annotation_path)[0] == "uncertain"
    assert not (session / "masks/frame_00002.png").exists()


def test_conflicting_labels_are_rejected(tmp_path):
    annotation_path = _annotation(tmp_path / "session", "frame_00003", "pupil")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["shapes"].append(
        {
            "label": "uncertain",
            "points": [[1.0, 1.0], [1.1, 1.0], [1.0, 1.1]],
            "shape_type": "polygon",
        }
    )
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")

    with pytest.raises(ValueError, match="mixes contradictory labels"):
        import_labelme.annotation_kind(annotation_path)


def test_unknown_label_is_rejected(tmp_path):
    _annotation(tmp_path / "session", "frame_00004", "maybe_pupil")

    with pytest.raises(ValueError, match="unsupported label"):
        import_labelme.annotation_kind(tmp_path / "session/images/frame_00004.json")
