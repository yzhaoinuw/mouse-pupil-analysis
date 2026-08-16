import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = PROJECT_ROOT / "sample_data"
FRAME_SUFFIX = re.compile(r"_(\d+)\.png$")


def _png_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.png"))


def test_cropped_sample_pairs_match_and_masks_are_binary():
    sessions = sorted(p for p in (SAMPLE_ROOT / "labeled_data").iterdir() if p.is_dir())
    assert len(sessions) == 10

    total = 0
    for session in sessions:
        image_names = _png_names(session / "images")
        assert image_names, f"{session.name} holds no images"
        assert _png_names(session / "masks") == image_names
        total += len(image_names)

        for image_name in image_names:
            with Image.open(session / "images" / image_name) as image:
                image_size = image.size
            with Image.open(session / "masks" / image_name) as mask:
                mask_values = set(np.unique(np.asarray(mask.convert("L"))).tolist())
                assert mask.size == image_size
            assert 0 in mask_values
            assert len(mask_values) <= 2
            assert max(mask_values) > 0

    assert total == 32


def test_fixture_split_mirrors_the_maintained_layout():
    """The fixture exists to exercise the split, not merely to hold images.

    Each assertion here is a property the old train/validation fixture could not
    satisfy: it had no mask small enough to populate the tiny size bin, and half its
    sessions held a single image, which satisfies "no session spans a fold" vacuously.
    """
    manifest = json.loads((SAMPLE_ROOT / "splits.json").read_text(encoding="utf-8"))

    assert manifest["n_images"] == 32
    assert manifest["n_sessions"] == 10
    assert manifest["n_folds"] == 4

    fold_of_session = {e["session"]: e["fold"] for e in manifest["sessions"]}
    for entry in manifest["images"]:
        assert entry["fold"] == fold_of_session[entry["session"]]

    per_session = Counter(entry["session"] for entry in manifest["images"])
    assert max(per_session.values()) >= 5, "no session deep enough to exercise grouping"

    tiny = [e for e in manifest["images"] if e["diameter"] <= manifest["tiny_max_diameter"]]
    assert len({e["session"] for e in tiny}) >= 2, "tiny masks must span several sessions"

    # The session is structural: every key is prefixed by the folder it came from, so
    # the grouping cannot disagree with where the file actually sits.
    for entry in manifest["images"]:
        assert entry["key"].startswith(f"{entry['session']}/")
        assert (
            entry["image"] == f"labeled_data/{entry['session']}/images/{Path(entry['image']).name}"
        )
        assert entry["mask"] == f"labeled_data/{entry['session']}/masks/{Path(entry['mask']).name}"
    assert all(e["source"] == "folder" for e in manifest["sessions"])


def test_committed_folds_match_the_manifest():
    manifest = json.loads((SAMPLE_ROOT / "splits.json").read_text(encoding="utf-8"))
    folds = SAMPLE_ROOT / "folds"

    for entry in manifest["images"]:
        name = f"cv{entry['fold'] + 1}"
        assert (folds / name / "images" / f"{entry['key']}.png").is_file()
        assert (folds / name / "masks" / f"{entry['key']}.png").is_file()

    # Folds partition the pool: nothing duplicated, nothing stale left behind.
    written = list(folds.glob("cv*/images/**/*.png"))
    assert len(written) == manifest["n_images"]


def test_raw_and_velocity_fixture_contracts():
    raw_root = SAMPLE_ROOT / "raw_frames"
    assert {
        directory.name: len(list(directory.glob("*.png")))
        for directory in raw_root.iterdir()
        if directory.is_dir()
    } == {"recording_250530": 3, "recording_250616": 3}
    raw_paths = sorted(raw_root.glob("*/*.png"))
    assert len(raw_paths) == 6
    for raw_path in raw_paths:
        with Image.open(raw_path) as image:
            assert image.width > 148 or image.height > 148

    velocity_paths = sorted(
        (SAMPLE_ROOT / "velocity_frames").glob("*.png"),
        key=lambda path: int(FRAME_SUFFIX.search(path.name).group(1)),
    )
    suffixes = [int(FRAME_SUFFIX.search(path.name).group(1)) for path in velocity_paths]
    assert suffixes == list(range(7212, 7243))
    for velocity_path in velocity_paths:
        with Image.open(velocity_path) as image:
            assert image.mode == "L"
            assert image.size == (148, 148)


def test_manifest_covers_every_logical_sample():
    with (SAMPLE_ROOT / "manifest.csv").open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))

    assert len(rows) == 69
    assert {row["category"] for row in rows} == {"cropped", "raw", "velocity"}

    # Every cropped sample carries the session it came from and the fold it landed in;
    # the old `split` column recorded a train/validation division that no longer exists.
    cropped = [row for row in rows if row["category"] == "cropped"]
    assert len(cropped) == 32
    assert all(row["session"] and re.fullmatch(r"cv[1-4]", row["fold"]) for row in cropped)
    for row in rows:
        assert (PROJECT_ROOT / row["image_path"]).is_file()
        if row["mask_path"]:
            assert (PROJECT_ROOT / row["mask_path"]).is_file()

    velocity_rows = [row for row in rows if row["category"] == "velocity"]
    assert all(row["acquisition_fps_hz"] == "97" for row in velocity_rows)
