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
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT  # Use PROJECT_ROOT / "sample_data" for the included fixture.
LABELLED_ROOT = "labeled_frames"


def export_session(session_dir: Path) -> tuple[int, int]:
    """Write a mask for every annotation in one session. Returns (written, skipped)."""
    images, masks = session_dir / "images", session_dir / "masks"
    if not images.is_dir():
        raise FileNotFoundError(f"{session_dir} has no images/ directory.")
    masks.mkdir(exist_ok=True, parents=True)

    written = skipped = 0
    for json_file in sorted(images.glob("*.json")):
        target = masks / f"{json_file.stem}.png"
        if target.exists():
            skipped += 1
            continue

        print(f"  {json_file.name}")
        subprocess.run(["labelme_export_json", str(json_file)], check=True)
        # labelme_export_json writes into a folder named after the annotation.
        export_dir = images / json_file.stem
        label = export_dir / "label.png"
        if not label.exists():
            raise FileNotFoundError(f"labelme_export_json produced no label.png for {json_file}")
        shutil.move(str(label), target)
        shutil.rmtree(export_dir)
        written += 1
    return written, skipped


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

    total_written = total_skipped = 0
    for session_dir in sessions:
        print(f"{session_dir.name}")
        written, skipped = export_session(session_dir)
        total_written += written
        total_skipped += skipped

    print(f"\n{total_written} mask(s) written, {total_skipped} already present.")
    if total_written:
        print("Now refresh the split: python training/data_splits.py --data-root . --materialize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
