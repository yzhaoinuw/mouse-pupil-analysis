# -*- coding: utf-8 -*-
"""Run grouped cross-validation over every fold of a split manifest.

Each fold trains on all other folds and validates on its own held-out sessions, so
every session is scored exactly once by a model that never saw that recording
setting. The headline number is the mean per-session IoU: averaging over images
instead lets the largest session dominate, and one session currently holds 28% of
the labelled pool.

    python training/data_splits.py --data-root . --out splits.json
    python training/run_cv.py --data-root . --split-manifest splits.json --out checkpoints_exp/cv

Use this to compare *configurations* -- sampling, loss, augmentation, architecture.
It is not how the shipped checkpoint is built: once a configuration wins, freeze the
schedule and threshold, refit on the non-holdout development pool, and consume the outer
holdout once with ``training/evaluate_holdout.py``. The hard-frame visual check remains a
separate promotion gate.

Cross-validation narrows sampling noise, not seed noise; repeat with ``--seed`` to
separate the two. The +/-0.0069 floor in ``reports/2026-08-14-checkpoint-noise-floor.md``
was measured on the old leaky split and understates this one. On the grouped split,
three seeds put the sd at 0.0273 for the mean per-session IoU and as high as 0.0873
for a single fold, so treat differences below roughly 0.05 on one fold as noise
(``reports/2026-08-16-selection-metric-repair.md``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    """Load a sibling training module by path.

    The module is registered in ``sys.modules`` first because dataclasses resolves
    string annotations through it.
    """
    path = PROJECT_ROOT / "training" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"training_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def per_session_iou(
    checkpoint: Path,
    manifest: dict,
    fold: int,
    config,
    trainer,
    data_splits,
) -> dict[str, float]:
    """Score one fold's held-out images and average IoU within each session.

    The checkpoint is evaluated at the threshold its own run calibrated, which is the
    threshold that would ship with it.
    """
    from mouse_pupil_analysis.pupil_predictions import load_unet_checkpoint

    _, validation = data_splits.fold_paths(manifest, fold, config.data_root)
    image_paths, mask_paths = validation
    dataset = trainer.SegmentationDataset(image_paths, mask_paths, augment=False)
    device = trainer.resolve_device(config.device)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    model = load_unet_checkpoint(Path(checkpoint), device)
    _, probabilities, targets = trainer._collect_validation(
        model, loader, torch.nn.BCEWithLogitsLoss(), device
    )
    report = trainer.evaluate_thresholds(
        probabilities,
        targets,
        config.threshold_candidates,
        config.tiny_max_diameter,
        config.large_min_diameter,
        config.low_circularity_cutoff,
        config.selection_metric,
    )
    iou, _ = trainer.per_image_overlap_scores(probabilities, targets, report.threshold)

    session_of = data_splits.session_of_path(manifest, config.data_root)
    grouped: dict[str, list[float]] = defaultdict(list)
    for path, score in zip(image_paths, iou.cpu().numpy().tolist()):
        grouped[session_of[Path(path).resolve()]].append(score)
    return {session: statistics.fmean(scores) for session, scores in grouped.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=Path("splits.json"))
    parser.add_argument("--out", type=Path, required=True, help="Directory for run folders.")
    parser.add_argument("--folds", type=int, nargs="+", help="Subset of folds (default: all).")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tag", default="cv", help="Run-name prefix.")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--balance-sizes",
        action="store_true",
        help="Sample equal mass from the tiny/medium/large bins. Off by default: natural "
        "sampling won the paired comparison on this split.",
    )
    sampling.add_argument(
        "--natural-sampling",
        action="store_true",
        help="Use the natural size distribution. Now the default; accepted so existing "
        "commands keep working.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("balanced_iou", "macro_iou"),
        default="macro_iou",
        help="Validation metric deciding the best checkpoint and early stopping. Grouped folds "
        "can leave a size bin holding one or two images, which makes 'balanced_iou' volatile; "
        "macro_iou is the default.",
    )
    parser.add_argument(
        "--scheduler-metric",
        choices=("val_loss", "balanced_iou", "macro_iou"),
        default="val_loss",
        help="Plateau signal for the learning-rate scheduler (default: val_loss).",
    )
    parser.add_argument(
        "--selection-threshold",
        default="0.5",
        help="Fixed threshold for the per-epoch selection comparison, or 'calibrated' for the "
        "previous behaviour of selecting on the best candidate (default: 0.5).",
    )
    parser.add_argument(
        "--finetune-checkpoint",
        type=Path,
        help="Fine-tune these weights instead of training fresh. Only valid if the weights "
        "were not themselves trained on this pool, or every fold leaks.",
    )
    args = parser.parse_args(argv)
    if args.selection_threshold == "calibrated":
        selection_threshold = None
    else:
        try:
            selection_threshold = float(args.selection_threshold)
        except ValueError:
            parser.error("--selection-threshold takes a probability or the word 'calibrated'.")

    trainer = _load("run_train")
    data_splits = _load("data_splits")

    manifest = data_splits.load_manifest(args.split_manifest)
    folds = args.folds if args.folds else list(range(manifest["n_folds"]))

    gate = data_splits.holdout_sessions(manifest)
    if gate:
        held = sum(e["n_images"] for e in manifest["sessions"] if e.get("holdout"))
        print(
            f"Holdout excluded from every fold: {len(gate)} session(s), {held} image(s) "
            f"({', '.join(sorted(gate))}). Refit with run_train.py --final, then score "
            "once with training/evaluate_holdout.py."
        )

    started = time.perf_counter()
    results = []
    session_scores: dict[str, float] = {}
    for fold in folds:
        name = f"{args.tag}_f{fold}_s{args.seed}"
        print(f"\n===== {name} =====", flush=True)
        config = trainer.TrainingConfig(
            data_root=args.data_root.resolve(),
            checkpoint_dir=args.out,
            run_name=name,
            split_manifest=args.split_manifest.resolve(),
            fold=fold,
            finetune_checkpoint=args.finetune_checkpoint,
            balance_training_sizes=args.balance_sizes,
            selection_metric=args.selection_metric,
            scheduler_metric=args.scheduler_metric,
            selection_threshold=selection_threshold,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            console_interval=50,
        )
        checkpoint = trainer.run_training(config)

        metadata = json.loads((Path(checkpoint).parent / "best.json").read_text(encoding="utf-8"))
        per_session = per_session_iou(checkpoint, manifest, fold, config, trainer, data_splits)
        session_scores.update(per_session)
        results.append({"fold": fold, "metadata": metadata, "per_session": per_session})

    print(f"\n{'=' * 72}\nGrouped cross-validation over {len(folds)} fold(s)\n{'=' * 72}")
    print(f"{'fold':>4} {'thresh':>7} {'macro':>7} {'balanced':>9} {'epoch':>6}  bins scored")
    for result in results:
        metadata = result["metadata"]
        bins = [name for name, value in metadata["size_iou"].items() if value is not None]
        print(
            f"{result['fold']:>4} {metadata['prediction_threshold']:>7.2f} "
            f"{metadata['macro_iou']:>7.4f} {metadata['balanced_iou']:>9.4f} "
            f"{metadata['best_epoch']:>6}  {'+'.join(bins)}"
        )

    print(f"\n{'session':<40} {'fold':>4} {'images':>7} {'IoU':>7}")
    by_session = {entry["session"]: entry for entry in manifest["sessions"]}
    for session, score in sorted(session_scores.items(), key=lambda kv: kv[1]):
        entry = by_session[session]
        print(f"{session[:38]:<40} {entry['fold']:>4} {entry['n_images']:>7} {score:>7.4f}")

    scores = list(session_scores.values())
    if scores:
        images = sum(by_session[s]["n_images"] for s in session_scores)
        pooled = sum(session_scores[s] * by_session[s]["n_images"] for s in session_scores) / images
        spread = f", sd {statistics.stdev(scores):.4f}" if len(scores) > 1 else ""
        worst = min(session_scores, key=session_scores.get)
        print(f"\nmean per-session IoU : {statistics.fmean(scores):.4f} (n={len(scores)}{spread})")
        print(f"worst session        : {worst} ({session_scores[worst]:.4f})")
        print(f"image-weighted IoU   : {pooled:.4f}  (comparable to previously reported macro IoU)")

    summary = args.out / f"{args.tag}_s{args.seed}_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "split_manifest": str(args.split_manifest),
                "folds": folds,
                "seed": args.seed,
                "per_session_iou": session_scores,
                "per_fold": [
                    {
                        "fold": r["fold"],
                        "threshold": r["metadata"]["prediction_threshold"],
                        "macro_iou": r["metadata"]["macro_iou"],
                        "balanced_iou": r["metadata"]["balanced_iou"],
                        "best_epoch": r["metadata"]["best_epoch"],
                    }
                    for r in results
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {summary}")
    print(f"Done in {(time.perf_counter() - started) / 60:.0f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
