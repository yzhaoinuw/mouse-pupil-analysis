# -*- coding: utf-8 -*-
"""Census the local training data: mask sizes, and how much validation leaks.

Filenames carry the recording an image came from, so overlap between the splits
is measurable. Validation images drawn from a recording that also appears in
training measure held-out frames, not generalisation.

    python reports/scripts/dataset_census.py --data-root .

Two naming schemes are in use and both have variable-length middle segments, so
this splits on structural markers rather than tokenising:

    HQL080_whiskerb_250722_007_eye_06622  -> animal HQL080, recording ..._007
    250530_5003_Green_..._2025-05-30T09-27-57.042_0000 -> animal 5003
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np

from mouse_pupil_analysis.augmentation import mask_equivalent_diameter, paired_image_mask_paths

TIMESTAMP = re.compile(r"^(.*?\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)")


def parse_identity(stem: str) -> tuple[str, str, str]:
    """Return ``(cohort, animal, recording)`` for one image stem."""
    if stem.startswith("HQL"):
        return ("HQL", stem.split("_")[0], stem.split("_eye_")[0])
    match = TIMESTAMP.match(stem)
    if match:
        return ("date_id", stem.split("_")[1], match.group(1))
    return ("unparsed", stem, stem)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--tiny-max-diameter", type=float, default=15.0)
    parser.add_argument("--large-min-diameter", type=float, default=80.0)
    args = parser.parse_args(argv)

    splits = {}
    for split in ("train", "validation"):
        images, masks = paired_image_mask_paths(
            args.data_root / f"images_{split}", args.data_root / f"masks_{split}"
        )
        diameters = np.array([mask_equivalent_diameter(path) for path in masks])
        identities = [parse_identity(path.stem) for path in images]
        splits[split] = identities
        unparsed = sum(cohort == "unparsed" for cohort, _, _ in identities)
        print(
            f"{split:<11} n={len(images):>3}  "
            f"tiny={int((diameters <= args.tiny_max_diameter).sum()):>3}  "
            f"large={int((diameters >= args.large_min_diameter).sum()):>3}  "
            f"median d={np.median(diameters):>5.1f}  "
            f"range {diameters.min():.1f}-{diameters.max():.1f}"
        )
        print(
            f"{'':<11} cohorts={dict(Counter(c for c, _, _ in identities))}  "
            f"animals={len({a for _, a, _ in identities})}  "
            f"recordings={len({r for _, _, r in identities})}"
            + (f"  UNPARSED={unparsed}" if unparsed else "")
        )

    train_animals = {a for _, a, _ in splits["train"]}
    train_recordings = {r for _, _, r in splits["train"]}
    validation = splits["validation"]

    print("\nleakage into validation")
    for cohort in sorted({c for c, _, _ in validation}) + [None]:
        subset = [v for v in validation if cohort is None or v[0] == cohort]
        if not subset:
            continue
        by_recording = sum(r in train_recordings for _, _, r in subset)
        by_animal = sum(a in train_animals for _, a, _ in subset)
        print(
            f"  {cohort or 'ALL':<9} n={len(subset):>3}  "
            f"same recording as training {by_recording:>3}/{len(subset)} "
            f"({100 * by_recording / len(subset):>3.0f}%)  "
            f"same animal {by_animal:>3}/{len(subset)} "
            f"({100 * by_animal / len(subset):>3.0f}%)"
        )

    validation_only = {a for _, a, _ in validation} - train_animals
    print(f"\n  validation-only animals: {sorted(validation_only) or 'NONE'}")
    print(
        f"  images per training animal: {Counter(a for _, a, _ in splits['train']).most_common()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
