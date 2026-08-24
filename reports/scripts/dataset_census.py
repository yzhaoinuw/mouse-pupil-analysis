# -*- coding: utf-8 -*-
"""Census the local training data: mask sizes per pool folder and per fold.

The historical `images_train` / `images_validation` folders were populated by hand and
shared recordings across the boundary, so validation IoU measured against them reported
held-out frames rather than generalisation. That evidence is what moved this project to
the grouped manifest, and the labelled pairs now live in one flat `labeled_frames` pool.
The leakage section below therefore only prints for a checkout still laid out the old
way; it is kept because that comparison is the argument for the current design.

    python reports/scripts/dataset_census.py --data-root . --split-manifest training_data_split.json

Sessions are read from the manifest rather than parsed out of filenames, so the census
and the fold assignment cannot disagree about which recording an image came from. That
also means this script reports at the session level only: the animal and cohort
breakdowns it used to print were themselves filename-derived, and filenames are no
longer treated as a source of truth. See `training/data_collection.md`.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_data_splits():
    path = PROJECT_ROOT / "training" / "data_splits.py"
    spec = importlib.util.spec_from_file_location("training_data_splits_census", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--split-manifest", type=Path, default=Path("training_data_split.json"))
    parser.add_argument("--tiny-max-diameter", type=float, default=15.0)
    parser.add_argument("--large-min-diameter", type=float, default=80.0)
    args = parser.parse_args(argv)

    manifest = _load_data_splits().load_manifest(args.split_manifest)

    # The first path component is the pool folder an image physically sits in, which is
    # what the legacy split used to decide train from validation. One folder now.
    by_folder: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest["images"]:
        by_folder[Path(entry["image"]).parts[0]].append(entry)

    for folder in sorted(by_folder):
        entries = by_folder[folder]
        diameters = np.array([entry["diameter"] for entry in entries])
        sessions = {entry["session"] for entry in entries}
        print(
            f"{folder:<19} n={len(entries):>3}  "
            f"tiny={int((diameters <= args.tiny_max_diameter).sum()):>3}  "
            f"large={int((diameters >= args.large_min_diameter).sum()):>3}  "
            f"median d={np.median(diameters):>5.1f}  "
            f"range {diameters.min():.1f}-{diameters.max():.1f}  "
            f"sessions={len(sessions)}"
        )

    train = by_folder.get("images_train", [])
    validation = by_folder.get("images_validation", [])
    if train and validation:
        train_sessions = {entry["session"] for entry in train}
        shared = sum(entry["session"] in train_sessions for entry in validation)
        print("\nleakage in the legacy fixed-folder split")
        print(
            f"  images_validation drawn from a session that also appears in "
            f"images_train: {shared}/{len(validation)} "
            f"({100 * shared / len(validation):>3.0f}%)"
        )
        only = {entry["session"] for entry in validation} - train_sessions
        print(f"  validation-only sessions: {sorted(only) or 'NONE'}")

    print("\nfold assignment now in force (sessions never span a fold)")
    per_fold: Counter = Counter()
    for entry in manifest["images"]:
        per_fold["holdout" if entry.get("holdout") else entry["fold"]] += 1
    for fold, count in sorted(per_fold.items(), key=lambda kv: str(kv[0])):
        print(f"  fold {str(fold):<8} {count:>3} images")

    print(
        f"\n  images per session: {Counter(e['session'] for e in manifest['images']).most_common()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
