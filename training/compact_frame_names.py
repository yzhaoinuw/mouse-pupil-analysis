# -*- coding: utf-8 -*-
"""Compact labelled-pool filenames after frames have been grouped by session.

Session folders carry the recording identity, so filenames only need the source frame
index. This utility renames matching images, masks, and optional Labelme JSON files to
``frame_<five-digit-index>`` and updates each JSON ``imagePath`` field.

The default is a dry run. Apply a fully validated plan with::

    python training/compact_frame_names.py --data-root . --apply
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATTERN = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class RenameEntry:
    """One paired image/mask rename and its optional Labelme annotation."""

    session: str
    image: Path
    mask: Path
    annotation: Path | None
    new_name: str


def frame_index(path: Path) -> int:
    """Return the numeric suffix that identifies the source frame."""
    match = INDEX_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"{path} has no trailing numeric frame index.")
    return int(match.group(1))


def build_plan(labelled_root: Path, sessions: list[str] | None = None) -> list[RenameEntry]:
    """Validate the labelled tree and return its collision-free rename plan."""
    labelled_root = Path(labelled_root)
    if sessions:
        session_dirs = [labelled_root / session for session in sessions]
    else:
        session_dirs = sorted(path for path in labelled_root.iterdir() if path.is_dir())

    plan: list[RenameEntry] = []
    for session_dir in session_dirs:
        if not session_dir.is_dir():
            raise FileNotFoundError(f"No labelled session directory: {session_dir}")
        image_dir = session_dir / "images"
        mask_dir = session_dir / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"{session_dir} must contain images/ and masks/.")

        targets: dict[str, Path] = {}
        images = sorted(image_dir.glob("*.png"))
        if not images:
            raise ValueError(f"{image_dir} contains no PNG images.")
        for image in images:
            mask = mask_dir / image.name
            if not mask.is_file():
                raise FileNotFoundError(f"No matching mask for {image}: expected {mask}.")
            index = frame_index(image)
            new_name = f"frame_{index:05d}.png"
            if new_name in targets and targets[new_name] != image:
                raise ValueError(
                    f"Frame-index collision in {session_dir.name}: {targets[new_name].name} "
                    f"and {image.name} both map to {new_name}."
                )
            targets[new_name] = image
            annotation = image.with_suffix(".json")
            annotation_target = image_dir / Path(new_name).with_suffix(".json")
            if (
                annotation_target.exists()
                and annotation_target != annotation
                and not any(
                    existing.with_suffix(".json") == annotation_target for existing in images
                )
            ):
                raise FileExistsError(
                    f"Refusing to replace unrelated annotation {annotation_target}."
                )
            plan.append(
                RenameEntry(
                    session=session_dir.name,
                    image=image,
                    mask=mask,
                    annotation=annotation if annotation.is_file() else None,
                    new_name=new_name,
                )
            )

        image_names = {path.name for path in images}
        orphan_masks = sorted(
            path.name for path in mask_dir.glob("*.png") if path.name not in image_names
        )
        if orphan_masks:
            raise ValueError(
                f"{session_dir.name} has {len(orphan_masks)} orphan mask(s): "
                f"{', '.join(orphan_masks[:5])}."
            )
    return plan


def _updated_annotation(path: Path, new_name: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "imagePath" not in payload:
        raise ValueError(f"Labelme annotation has no imagePath: {path}")
    payload["imagePath"] = new_name
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def apply_plan(plan: list[RenameEntry]) -> int:
    """Apply a prevalidated plan with temporary names to avoid collisions."""
    staged: list[tuple[Path, Path]] = []
    annotations: dict[Path, str] = {}
    for entry in plan:
        if entry.annotation is not None:
            annotations[entry.annotation] = _updated_annotation(entry.annotation, entry.new_name)

    try:
        for entry in plan:
            targets = [
                (entry.image, entry.image.with_name(entry.new_name)),
                (entry.mask, entry.mask.with_name(entry.new_name)),
            ]
            if entry.annotation is not None:
                targets.append(
                    (
                        entry.annotation,
                        entry.annotation.with_name(Path(entry.new_name).with_suffix(".json").name),
                    )
                )
            for source, target in targets:
                if source == target:
                    continue
                temporary = source.with_name(f".{source.name}.compact-{uuid.uuid4().hex}")
                source.replace(temporary)
                staged.append((temporary, target))

        for temporary, target in staged:
            temporary.replace(target)

        for entry in plan:
            if entry.annotation is not None:
                target = entry.annotation.with_name(Path(entry.new_name).with_suffix(".json").name)
                target.write_text(annotations[entry.annotation], encoding="utf-8")
    except Exception:
        for temporary, target in reversed(staged):
            if temporary.exists():
                original_name = temporary.name.split(".compact-", 1)[0].lstrip(".")
                temporary.replace(temporary.with_name(original_name))
            elif target.exists():
                # A staged file already reached its target. Leave it in place rather than
                # guessing which pre-existing path is safe to replace.
                pass
        raise
    return len(plan)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--session", action="append", help="Rename only this session; repeatable.")
    parser.add_argument("--apply", action="store_true", help="Apply the validated plan.")
    args = parser.parse_args(argv)

    labelled_root = args.data_root.resolve() / "labeled_frames"
    plan = build_plan(labelled_root, args.session)
    changed = [entry for entry in plan if entry.image.name != entry.new_name]
    by_session: dict[str, int] = {}
    for entry in changed:
        by_session[entry.session] = by_session.get(entry.session, 0) + 1
    for session, count in sorted(by_session.items()):
        print(f"{session}: {count} pair(s) -> frame_<index>")
    print(f"Validated {len(plan)} pair(s); {len(changed)} need renaming.")

    if not args.apply:
        print("Dry run only; pass --apply to rename images, masks, and Labelme JSON files.")
        return 0
    apply_plan(plan)
    print(f"Renamed {len(changed)} pair(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
