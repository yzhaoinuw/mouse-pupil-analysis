# -*- coding: utf-8 -*-
"""Score a checkpoint on validation by IoU *and* by the diameter it reports.

IoU alone hides a systematic measurement offset: a threshold can score 0.895
while under-measuring every diameter by 4%. It is also mechanically harsher on
small pupils, so a low tiny-bin IoU is not by itself evidence of a small-pupil
weakness. This prints both, per threshold and per size bin.

    python reports/scripts/validation_diagnostics.py --data-root . --fold 0

Scores one fold of the grouped split. There is no standing validation folder any
more -- the labelled pairs are one flat pool and the manifest decides what is held
out -- so which images this reports on is a choice, and ``--fold`` is it. Pass
``--holdout`` instead to score the gate sessions.

The "implied boundary error" column converts each bin's IoU back into pixels
(IoU ~ 1 - k/d for a disc of diameter d), which is comparable across sizes in a
way IoU is not.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training import _trainer as training_core  # noqa: E402
from training import prepare_splits  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--split-manifest", type=Path, default=Path("training_data_split.json"))
    parser.add_argument("--fold", type=int, default=0, help="Fold to score as validation.")
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Score the manifest's holdout sessions instead of a fold.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Default: the packaged checkpoint.")
    parser.add_argument("--tiny-max-diameter", type=float, default=15.0)
    parser.add_argument("--large-min-diameter", type=float, default=80.0)
    args = parser.parse_args(argv)

    from mouse_pupil_analysis.pupil_predictions import (
        find_default_checkpoint,
        load_unet_checkpoint,
        resolve_prediction_threshold,
    )

    checkpoint = args.checkpoint or find_default_checkpoint()
    device = torch.device("cpu")
    model = load_unet_checkpoint(checkpoint, device)
    manifest = prepare_splits.load_manifest(args.split_manifest)
    if args.holdout:
        images, masks = prepare_splits.holdout_paths(manifest, args.data_root)
        scope = f"holdout ({', '.join(prepare_splits.holdout_sessions(manifest))})"
    else:
        _, (images, masks) = prepare_splits.fold_paths(manifest, args.fold, args.data_root)
        scope = f"fold {args.fold}/{manifest['n_folds']}"
    dataset = training_core.SegmentationDataset(images, masks, augment=False)

    probabilities, targets = [], []
    with torch.no_grad():
        for images, masks in DataLoader(dataset, batch_size=8):
            probabilities.append(torch.sigmoid(model(images)))
            targets.append(masks)
    probabilities, targets = torch.cat(probabilities), torch.cat(targets)

    truth_area = (targets > 0.5).sum(dim=(1, 2, 3)).float().numpy()
    truth_diameter = np.sqrt(4.0 * truth_area / math.pi)

    def measured(threshold: float) -> np.ndarray:
        area = (probabilities > threshold).sum(dim=(1, 2, 3)).float().numpy()
        return np.sqrt(4.0 * area / math.pi)

    print(f"checkpoint: {checkpoint.name}   scope: {scope}   n={len(truth_diameter)}\n")
    print(f"{'thr':>5} {'macro IoU':>10} {'signed diam':>12} {'|diam|':>8}")
    for threshold in np.arange(0.20, 0.81, 0.05):
        iou, _ = training_core.per_image_overlap_scores(probabilities, targets, float(threshold))
        relative = (measured(threshold) - truth_diameter) / truth_diameter * 100
        print(
            f"{threshold:>5.2f} {float(iou.mean()):>10.4f} "
            f"{relative.mean():>11.1f}% {np.abs(relative).mean():>7.1f}%"
        )

    packaged = resolve_prediction_threshold(checkpoint)
    iou, _ = training_core.per_image_overlap_scores(probabilities, targets, packaged)
    iou = iou.numpy()
    relative = (measured(packaged) - truth_diameter) / truth_diameter * 100
    bins = {
        f"tiny (<={args.tiny_max_diameter:g})": truth_diameter <= args.tiny_max_diameter,
        "medium": (truth_diameter > args.tiny_max_diameter)
        & (truth_diameter < args.large_min_diameter),
        f"large (>={args.large_min_diameter:g})": truth_diameter >= args.large_min_diameter,
    }
    print(f"\nby size bin, at the checkpoint's own threshold ({packaged:g}):")
    print(f"{'bin':<16} {'n':>3} {'IoU':>8} {'signed diam':>12} {'implied boundary':>17}")
    for name, selected in bins.items():
        if not selected.any():
            print(f"{name:<16} {0:>3}  (empty)")
            continue
        bin_iou = iou[selected].mean()
        median_d = np.median(truth_diameter[selected])
        print(
            f"{name:<16} {int(selected.sum()):>3} {bin_iou:>8.4f} "
            f"{relative[selected].mean():>11.1f}% {(1 - bin_iou) * median_d:>15.1f} px"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
