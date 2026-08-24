# -*- coding: utf-8 -*-
"""Run grouped cross-validation over every fold of a split manifest.

Each fold trains on all other folds and validates on its own held-out sessions, so
every session is scored exactly once by a model that never saw that recording
setting. The headline number is the mean per-session IoU: averaging over images
instead lets the largest session dominate, and one session currently holds 28% of
the labelled pool.

    python training/prepare_splits.py
    python training/run_cross_validation.py

Use this to compare *configurations* -- sampling, loss, augmentation, architecture.
The validation session, when configured, is excluded from every CV fold and is reserved
for the normal training run. CV also writes a reusable all-labeled training configuration.

Cross-validation narrows sampling noise, not seed noise; repeat with ``--seed`` to
separate the two. The +/-0.0069 floor in ``reports/2026-08-14-checkpoint-noise-floor.md``
was measured on the old leaky split and understates this one. On the grouped split,
three seeds put the sd at 0.0273 for the mean per-session IoU and as high as 0.0873
for a single fold, so treat differences below roughly 0.05 on one fold as noise
(``reports/2026-08-16-selection-metric-repair.md``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __package__:
    from . import _trainer as training_core
    from . import prepare_splits
else:  # Direct ``python training/run_cross_validation.py`` execution.
    sys.path.insert(0, str(PROJECT_ROOT))
    from training import _trainer as training_core
    from training import prepare_splits


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


def all_labeled_training_config(summary: Path, results: list[dict], config, trainer) -> dict:
    """Build the reusable all-labeled recipe from successful CV folds."""
    selected_epochs = int(round(statistics.median(r["metadata"]["best_epoch"] for r in results)))
    selected_threshold = float(
        statistics.median(r["metadata"]["prediction_threshold"] for r in results)
    )
    return {
        "schema_version": 1,
        "source_cv_summary": str(summary),
        "source_cv_summary_sha256": trainer.file_sha256(summary),
        "max_epochs": selected_epochs,
        "learning_rate": trainer.initial_learning_rate(config),
        "lr_milestones": sorted(
            {
                epoch
                for epoch in (selected_epochs // 2, (3 * selected_epochs) // 4)
                if 0 < epoch < selected_epochs
            }
        ),
        "batch_size": config.batch_size,
        "seed": config.seed,
        "use_attention": config.use_attention,
        "sampling": "natural",
        "prediction_threshold": selected_threshold,
        "finetune_checkpoint": (
            None
            if config.finetune_checkpoint is None
            else str(config.finetune_checkpoint.resolve())
        ),
    }


def selected_cv_folds(requested: list[int] | None, n_folds: int) -> tuple[list[int], bool]:
    """Return the requested existing folds and whether they cover the whole CV split."""
    all_folds = list(range(n_folds))
    if requested is None:
        return all_folds, True
    if len(set(requested)) != len(requested):
        raise ValueError("--cv_folds cannot repeat a fold.")
    folds = sorted(requested)
    invalid = [fold for fold in folds if fold not in all_folds]
    if invalid:
        raise ValueError(f"--cv_folds contains invalid fold(s): {invalid}.")
    return folds, folds == all_folds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--labeled_frames_dir",
        type=Path,
        default=Path.cwd() / "labeled_frames",
        help="Folder containing session image/mask pairs (default: ./labeled_frames).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        help="Directory for CV run folders and its generated training configuration.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        nargs="+",
        help="Existing fold indices to rerun (default: every fold).",
    )
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--finetune_checkpoint",
        type=Path,
        help="Fine-tune these weights instead of training fresh. Only valid if the weights "
        "were not themselves trained on this pool, or every fold leaks.",
    )
    args = parser.parse_args(argv)
    labeled_frames_dir = args.labeled_frames_dir.resolve()
    if labeled_frames_dir.name != "labeled_frames":
        parser.error(
            f"--labeled_frames_dir must name a labeled_frames folder; got {labeled_frames_dir}."
        )
    data_root = labeled_frames_dir.parent
    data_splits = prepare_splits
    split_manifest = data_root / data_splits.TRAINING_DATA_SPLIT_FILENAME
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else data_root / "checkpoints_exp" / "cv"
    )
    if not split_manifest.is_file():
        parser.error(
            f"No {data_splits.TRAINING_DATA_SPLIT_FILENAME} beside {labeled_frames_dir}; "
            "run training/prepare_splits.py first."
        )

    trainer = training_core

    manifest = data_splits.load_manifest(split_manifest)
    try:
        folds, complete_cv = selected_cv_folds(args.cv_folds, manifest["n_folds"])
    except ValueError as error:
        parser.error(str(error))

    gate = data_splits.holdout_sessions(manifest)
    if gate:
        held = sum(e["n_images"] for e in manifest["sessions"] if e.get("holdout"))
        print(
            f"Holdout excluded from every fold: {len(gate)} session(s), {held} image(s) "
            f"({', '.join(sorted(gate))}). A generated training config later trains all "
            "labeled sessions, including these ones."
        )
    validation_holdout = data_splits.validation_holdout_sessions(manifest)
    if validation_holdout:
        held = sum(
            entry["n_images"] for entry in manifest["sessions"] if entry.get("validation_holdout")
        )
        print(
            f"Validation holdout excluded from every CV fold: {len(validation_holdout)} session(s), "
            f"{held} image(s) ({', '.join(sorted(validation_holdout))})."
        )

    started = time.perf_counter()
    results = []
    session_scores: dict[str, float] = {}
    for fold in folds:
        name = f"fold_{fold}_seed_{args.seed}"
        print(f"\n===== {name} =====", flush=True)
        config = trainer.TrainingConfig(
            labeled_frames_dir=labeled_frames_dir,
            checkpoint_dir=checkpoint_dir / name,
            split_manifest=split_manifest,
            fold=fold,
            finetune_checkpoint=args.finetune_checkpoint,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
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

    summary = checkpoint_dir / ("summary.json" if complete_cv else "partial_summary.json")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "split_manifest": str(split_manifest),
                "folds": folds,
                "complete_cv": complete_cv,
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
    if complete_cv:
        training_config = checkpoint_dir / "training_config.json"
        training_config.write_text(
            json.dumps(all_labeled_training_config(summary, results, config, trainer), indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote all-labeled training config: {training_config}")
    else:
        print("Skipped training_config.json because this was a partial CV run.")
    print(f"Done in {(time.perf_counter() - started) / 60:.0f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
