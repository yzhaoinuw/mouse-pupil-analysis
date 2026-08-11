import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = PROJECT_ROOT / "sample_data"
FRAME_SUFFIX = re.compile(r"_(\d+)\.png$")


def _png_names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.glob("*.png"))


def test_cropped_sample_pairs_match_and_masks_are_binary():
    for split, expected_count in (("train", 8), ("validation", 4)):
        image_dir = SAMPLE_ROOT / f"images_{split}"
        mask_dir = SAMPLE_ROOT / f"masks_{split}"
        image_names = _png_names(image_dir)
        mask_names = _png_names(mask_dir)

        assert len(image_names) == expected_count
        assert mask_names == image_names

        for image_name in image_names:
            with Image.open(image_dir / image_name) as image:
                image_size = image.size
            with Image.open(mask_dir / image_name) as mask:
                mask_values = set(np.unique(np.asarray(mask.convert("L"))).tolist())
                assert mask.size == image_size
            assert 0 in mask_values
            assert len(mask_values) <= 2
            assert max(mask_values) > 0


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

    assert len(rows) == 49
    assert {row["category"] for row in rows} == {"cropped", "raw", "velocity"}
    for row in rows:
        assert (PROJECT_ROOT / row["image_path"]).is_file()
        if row["mask_path"]:
            assert (PROJECT_ROOT / row["mask_path"]).is_file()

    velocity_rows = [row for row in rows if row["category"] == "velocity"]
    assert all(row["acquisition_fps_hz"] == "97" for row in velocity_rows)
