# -*- coding: utf-8 -*-
"""Import one reviewed Labelme batch into the session-grouped training pool.

The command validates the whole batch before writing anything, then imports it as a new
session. Visible-pupil polygons
and explicit ``no_visible_pupil`` negatives become compact image/mask pairs under
``labeled_frames/<session>/images|masks``. ``uncertain`` annotations are preserved under
``labeled_frames/<session>/uncertain`` and never enter segmentation training.

Import a labeled batch and refresh the frozen fold assignment::

    python training/labelme_json2png.py --source path/to/annotations --session SESSION
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUPIL_LABEL = "pupil"
NO_VISIBLE_PUPIL_LABEL = "no_visible_pupil"
UNCERTAIN_LABEL = "uncertain"
SUPPORTED_LABELS = {PUPIL_LABEL, NO_VISIBLE_PUPIL_LABEL, UNCERTAIN_LABEL}
FRAME_INDEX_PATTERN = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class ImportEntry:
    """One validated annotation and its compact destination name."""

    annotation_path: Path
    annotation: dict
    image: Path
    kind: str
    compact_stem: str


def frame_index(path: Path) -> int:
    """Return the trailing numeric source-frame index from a Labelme filename."""
    match = FRAME_INDEX_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"{path} has no trailing numeric frame index.")
    return int(match.group(1))


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
    """Return the image named by a Labelme annotation after validating its presence."""
    image_path = annotation.get("imagePath")
    if not image_path:
        raise ValueError(f"{json_file.name} has no imagePath.")
    source = json_file.parent / Path(image_path).name
    if not source.is_file():
        raise FileNotFoundError(f"{json_file.name} refers to missing image {source.name}.")
    return source


def _image_size(json_file: Path, annotation: dict) -> tuple[int, int]:
    source = source_image(json_file, annotation)
    with Image.open(source) as image:
        size = image.size
    declared = (annotation.get("imageWidth"), annotation.get("imageHeight"))
    if all(value is not None for value in declared) and tuple(declared) != size:
        raise ValueError(
            f"{json_file.name} declares image size {tuple(declared)}, but {source.name} is {size}."
        )
    return size


def write_training_mask(json_file: Path, annotation: dict, kind: str, target: Path) -> None:
    """Rasterize one validated trainable annotation into its target mask."""
    size = _image_size(json_file, annotation)
    if kind == NO_VISIBLE_PUPIL_LABEL:
        Image.new("L", size, color=0).save(target)
        return
    if kind != PUPIL_LABEL:
        raise ValueError(f"{kind!r} is not a segmentation-mask target.")
    mask = Image.new("L", size, color=0)
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


def _validate_session_name(session: str) -> str:
    session = session.strip()
    if (
        not session
        or session in {".", ".."}
        or Path(session).name != session
        or "/" in session
        or "\\" in session
    ):
        raise ValueError(f"Session must be one directory name, not {session!r}.")
    return session


def build_import_plan(source: Path, data_root: Path, session: str) -> list[ImportEntry]:
    """Validate a new session batch and return its collision-free import plan."""
    source = Path(source).resolve()
    data_root = Path(data_root).resolve()
    session = _validate_session_name(session)
    if not source.is_dir():
        raise FileNotFoundError(f"Annotation source is not a directory: {source}")

    target = data_root / "labeled_frames" / session
    if target.exists():
        raise FileExistsError(
            f"Refusing to merge into existing session {target}. "
            "Use a new session name so intake cannot overwrite labeled data."
        )

    json_files = sorted(source.glob("*.json"))
    if not json_files:
        raise ValueError(f"No Labelme JSON annotations found in {source}.")

    entries: list[ImportEntry] = []
    destinations: dict[str, Path] = {}
    for annotation_path in json_files:
        kind, annotation = annotation_kind(annotation_path)
        image = source_image(annotation_path, annotation)
        if image.stem != annotation_path.stem:
            raise ValueError(
                f"{annotation_path.name} refers to {image.name}; Labelme JSON and image "
                "must share a filename stem."
            )
        compact_stem = f"frame_{frame_index(annotation_path):05d}"
        if compact_stem in destinations:
            raise ValueError(
                f"Frame-index collision: {destinations[compact_stem].name} and "
                f"{annotation_path.name} both map to {compact_stem}."
            )
        destinations[compact_stem] = annotation_path
        entries.append(
            ImportEntry(
                annotation_path=annotation_path,
                annotation=annotation,
                image=image,
                kind=kind,
                compact_stem=compact_stem,
            )
        )

    if all(entry.kind == UNCERTAIN_LABEL for entry in entries):
        raise ValueError(
            "The batch contains only uncertain annotations and therefore no trainable pairs. "
            "Keep it outside labeled_frames until the abstention model exists."
        )
    return entries


def _write_uncertain(entry: ImportEntry, directory: Path) -> None:
    """Preserve an uncertain source image and compact, image-external Labelme JSON."""
    image_target = directory / f"{entry.compact_stem}.png"
    annotation_target = directory / f"{entry.compact_stem}.json"
    shutil.copy2(entry.image, image_target)
    annotation = dict(entry.annotation)
    annotation["imagePath"] = image_target.name
    annotation["imageData"] = None
    annotation_target.write_text(
        json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def apply_import(plan: list[ImportEntry], data_root: Path, session: str) -> Path:
    """Build a complete new session in staging, then move it into the pool atomically."""
    data_root = Path(data_root).resolve()
    session = _validate_session_name(session)
    labelled_root = data_root / "labeled_frames"
    target = labelled_root / session
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing session {target}.")
    labelled_root.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=f".{session}.import-", dir=labelled_root))
    try:
        images = staging / "images"
        masks = staging / "masks"
        uncertain = staging / "uncertain"
        images.mkdir()
        masks.mkdir()

        for entry in plan:
            if entry.kind == UNCERTAIN_LABEL:
                uncertain.mkdir(exist_ok=True)
                _write_uncertain(entry, uncertain)
                continue

            image_target = images / f"{entry.compact_stem}.png"
            mask_target = masks / image_target.name
            shutil.copy2(entry.image, image_target)
            write_training_mask(
                entry.annotation_path,
                entry.annotation,
                entry.kind,
                mask_target,
            )

        image_names = {path.name for path in images.glob("*.png")}
        mask_names = {path.name for path in masks.glob("*.png")}
        if image_names != mask_names:
            raise RuntimeError("Staged image/mask names do not match exactly.")
        staging.replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def refresh_split_manifest(data_root: Path) -> int:
    """Refresh the frozen manifest after a successful import."""
    if __package__:
        from . import prepare_splits
    else:
        import prepare_splits

    labeled_frames_dir = Path(data_root).resolve() / "labeled_frames"
    return prepare_splits.main(["--labeled_frames_dir", str(labeled_frames_dir)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="Folder holding Labelme JSONs.")
    parser.add_argument("--session", required=True, help="New recording-session folder name.")
    args = parser.parse_args(argv)

    plan = build_import_plan(args.source, PROJECT_ROOT, args.session)
    counts = Counter(entry.kind for entry in plan)
    trainable = len(plan) - counts[UNCERTAIN_LABEL]
    print(f"Session: {args.session}")
    print(f"Source: {Path(args.source).resolve()}")
    print(
        f"Annotations: {len(plan)} total; {trainable} trainable "
        f"({counts['pupil']} pupil, {counts['no_visible_pupil']} no-visible-pupil); "
        f"{counts[UNCERTAIN_LABEL]} uncertain archived outside training."
    )

    target = apply_import(plan, PROJECT_ROOT, args.session)
    print(f"Imported {trainable} image/mask pair(s) into {target}.")
    if counts[UNCERTAIN_LABEL]:
        print(
            f"Archived {counts[UNCERTAIN_LABEL]} uncertain annotation(s) in {target / 'uncertain'}."
        )
    return refresh_split_manifest(PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
