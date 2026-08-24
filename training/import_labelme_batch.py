# -*- coding: utf-8 -*-
"""Import one reviewed Labelme batch into the session-grouped training pool.

The command validates the whole batch before writing anything. Visible-pupil polygons
and explicit ``no_visible_pupil`` negatives become compact image/mask pairs under
``labeled_frames/<session>/images|masks``. ``uncertain`` annotations are preserved under
``labeled_frames/<session>/uncertain`` and never enter segmentation training.

Preview, then apply and refresh the frozen split::

    python training/import_labelme_batch.py --source path/to/annotations --session SESSION
    python training/import_labelme_batch.py --source path/to/annotations --session SESSION \
        --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .compact_frame_names import frame_index
    from .labelme_json2png import (
        UNCERTAIN_LABEL,
        annotation_kind,
        source_image,
        write_training_mask,
    )
else:
    from compact_frame_names import frame_index
    from labelme_json2png import (
        UNCERTAIN_LABEL,
        annotation_kind,
        source_image,
        write_training_mask,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ImportEntry:
    """One validated annotation and its compact destination name."""

    annotation_path: Path
    annotation: dict
    image: Path
    kind: str
    compact_stem: str


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
            "Use a new session name so intake cannot overwrite labelled data."
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
        from . import data_splits
    else:
        import data_splits

    labeled_frames_dir = Path(data_root).resolve() / "labeled_frames"
    return data_splits.main(["--labeled_frames_dir", str(labeled_frames_dir)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="Folder holding Labelme JSONs.")
    parser.add_argument("--session", required=True, help="New recording-session folder name.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Import the session and refresh training_data_split.json.",
    )
    args = parser.parse_args(argv)

    plan = build_import_plan(args.source, args.data_root, args.session)
    counts = Counter(entry.kind for entry in plan)
    trainable = len(plan) - counts[UNCERTAIN_LABEL]
    print(f"Session: {args.session}")
    print(f"Source: {Path(args.source).resolve()}")
    print(
        f"Annotations: {len(plan)} total; {trainable} trainable "
        f"({counts['pupil']} pupil, {counts['no_visible_pupil']} no-visible-pupil); "
        f"{counts[UNCERTAIN_LABEL]} uncertain archived outside training."
    )

    if not args.apply:
        print("Dry run only; pass --apply to import this new session.")
        return 0

    target = apply_import(plan, args.data_root, args.session)
    print(f"Imported {trainable} image/mask pair(s) into {target}.")
    if counts[UNCERTAIN_LABEL]:
        print(
            f"Archived {counts[UNCERTAIN_LABEL]} uncertain annotation(s) in {target / 'uncertain'}."
        )
    return refresh_split_manifest(args.data_root)


if __name__ == "__main__":
    raise SystemExit(main())
