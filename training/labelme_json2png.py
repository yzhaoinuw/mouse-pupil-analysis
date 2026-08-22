# -*- coding: utf-8 -*-
"""Convert Labelme annotations into mask PNGs, one session folder at a time.

The labelled pool is organised by recording session::

    labeled_frames/<session>/images/<anything>.png   the frame, and its .json annotation
    labeled_frames/<session>/masks/<anything>.png    written here

Annotate in Labelme and save each ``.json`` beside its image, then run::

    python training/labelme_json2png.py --data-root .
    python training/labelme_json2png.py --data-root . --session HQL091_sleep260820

Each mask takes its image's filename, which is how the two are paired. A mask that
already exists is left alone, so re-running only fills in what is missing.

Originally written 2025-09-28 by yzhao.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT  # Use PROJECT_ROOT / "sample_data" for the included fixture.
LABELLED_ROOT = "labeled_frames"
PUPIL_LABEL = "pupil"
NO_VISIBLE_PUPIL_LABEL = "no_visible_pupil"
UNCERTAIN_LABEL = "uncertain"
SUPPORTED_LABELS = {PUPIL_LABEL, NO_VISIBLE_PUPIL_LABEL, UNCERTAIN_LABEL}


def annotation_kind(json_file: Path) -> tuple[str, dict]:
    """Validate one Labelme annotation and return its image-level target kind."""
    annotation = json.loads(json_file.read_text(encoding="utf-8"))
    shapes = annotation.get("shapes", [])
    labels = {str(shape.get("label", "")).strip().casefold() for shape in shapes}
    unknown = labels - SUPPORTED_LABELS
    if unknown:
        raise ValueError(f"{json_file.name} uses unsupported label(s): {sorted(unknown)}")
    if not labels:
        raise ValueError(
            f"{json_file.name} has no shapes. Mark it {NO_VISIBLE_PUPIL_LABEL!r} or "
            f"{UNCERTAIN_LABEL!r} explicitly if it has no pupil polygon."
        )
    if len(labels) != 1:
        raise ValueError(f"{json_file.name} mixes contradictory labels: {sorted(labels)}")

    kind = labels.pop()
    if kind in {NO_VISIBLE_PUPIL_LABEL, UNCERTAIN_LABEL} and len(shapes) != 1:
        raise ValueError(f"{json_file.name} must contain exactly one {kind!r} marker.")
    return kind, annotation


def source_image(json_file: Path, annotation: dict) -> Path:
    image_path = annotation.get("imagePath")
    if not image_path:
        raise ValueError(f"{json_file.name} has no imagePath.")
    source = json_file.parent / Path(image_path).name
    if not source.is_file():
        raise FileNotFoundError(f"{json_file.name} refers to missing image {source.name}.")
    return source


def _image_size(json_file: Path, annotation: dict) -> tuple[int, int]:
    """Return the source image size after checking the Labelme metadata."""
    source = source_image(json_file, annotation)
    with Image.open(source) as image:
        size = image.size
    declared = (annotation.get("imageWidth"), annotation.get("imageHeight"))
    if all(value is not None for value in declared) and tuple(declared) != size:
        raise ValueError(
            f"{json_file.name} declares image size {tuple(declared)}, but {source.name} is {size}."
        )
    return size


def _write_empty_mask(json_file: Path, annotation: dict, target: Path) -> None:
    """Write an image-sized all-background mask for an explicit negative."""
    size = _image_size(json_file, annotation)
    Image.new("L", size, color=0).save(target)


def _write_pupil_mask(json_file: Path, annotation: dict, target: Path) -> None:
    """Rasterize validated Labelme pupil polygons without an external CLI."""
    mask = Image.new("L", _image_size(json_file, annotation), color=0)
    draw = ImageDraw.Draw(mask)
    for shape in annotation["shapes"]:
        shape_type = shape.get("shape_type", "polygon")
        points = shape.get("points", [])
        if shape_type != "polygon" or len(points) < 3:
            raise ValueError(
                f"{json_file.name} pupil targets must be polygons with at least three points."
            )
        draw.polygon([tuple(point) for point in points], fill=255)
    mask.save(target)


def write_training_mask(json_file: Path, annotation: dict, kind: str, target: Path) -> None:
    """Write one validated trainable target, rejecting uncertainty as non-mask data."""
    if kind == NO_VISIBLE_PUPIL_LABEL:
        _write_empty_mask(json_file, annotation, target)
    elif kind == PUPIL_LABEL:
        _write_pupil_mask(json_file, annotation, target)
    else:
        raise ValueError(f"{kind!r} is not a segmentation-mask target.")


def export_session(session_dir: Path) -> tuple[int, int, int]:
    """Write masks for one session; return (written, existing, uncertain)."""
    images, masks = session_dir / "images", session_dir / "masks"
    if not images.is_dir():
        raise FileNotFoundError(f"{session_dir} has no images/ directory.")
    masks.mkdir(exist_ok=True, parents=True)

    written = existing = uncertain = 0
    for json_file in sorted(images.glob("*.json")):
        kind, annotation = annotation_kind(json_file)
        target = masks / f"{json_file.stem}.png"
        if kind == UNCERTAIN_LABEL:
            if target.exists():
                raise ValueError(
                    f"{json_file.name} is uncertain but already has a training mask at {target}."
                )
            print(f"  {json_file.name} (uncertain; no training mask)")
            uncertain += 1
            continue
        if target.exists():
            existing += 1
            continue

        print(f"  {json_file.name} ({kind})")
        write_training_mask(json_file, annotation, kind, target)
        written += 1
    return written, existing, uncertain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--session",
        action="append",
        default=[],
        help="Only convert these session folders. Repeatable; default is all of them.",
    )
    args = parser.parse_args(argv)

    root = Path(args.data_root) / LABELLED_ROOT
    if not root.is_dir():
        raise SystemExit(
            f"No {LABELLED_ROOT}/ under {args.data_root}. Labelled pairs live in "
            f"{LABELLED_ROOT}/<session>/images and .../masks; see training/data_collection.md."
        )

    sessions = sorted(p for p in root.iterdir() if p.is_dir())
    if args.session:
        wanted = set(args.session)
        missing = wanted - {p.name for p in sessions}
        if missing:
            raise SystemExit(f"No such session folder: {sorted(missing)}")
        sessions = [p for p in sessions if p.name in wanted]

    total_written = total_existing = total_uncertain = 0
    for session_dir in sessions:
        print(f"{session_dir.name}")
        written, existing, uncertain = export_session(session_dir)
        total_written += written
        total_existing += existing
        total_uncertain += uncertain

    print(
        f"\n{total_written} mask(s) written, {total_existing} already present, "
        f"{total_uncertain} uncertain annotation(s) intentionally excluded."
    )
    if total_written:
        print("Now refresh the split: python training/data_splits.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
