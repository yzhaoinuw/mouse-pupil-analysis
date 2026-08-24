# -*- coding: utf-8 -*-
"""Evaluate one frozen final refit on the labeled outer holdout exactly once.

This command is intentionally separate from training. ``run_train.py --final`` never
loads holdout images or masks; it writes ``final.pth`` and ``final.json`` after a fixed
development-selected schedule. Only after the model, epoch count, learning-rate
schedule, and prediction threshold are frozen should you run::

    python training/evaluate_holdout.py \
        --run-dir checkpoints_exp/final_candidate \
        --data-root . \
        --split-manifest training_data_split.json \
        --confirm-frozen

The command refuses to overwrite ``holdout.json``. If the result causes another model
or threshold change, this holdout has become development data and a new untouched
holdout is required for the next unbiased evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = PROJECT_ROOT / "training" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"training_{name}_holdout", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_holdout(
    run_dir: Path,
    data_root: Path,
    split_manifest: Path,
    device_preference: str = "auto",
) -> dict:
    """Score a frozen final checkpoint at its already-recorded threshold."""
    trainer = _load("run_train")
    data_splits = _load("data_splits")

    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "final.pth"
    metadata_path = run_dir / "final.json"
    output_path = run_dir / "holdout.json"
    for required in (checkpoint_path, metadata_path):
        if not required.is_file():
            raise FileNotFoundError(f"Final run is incomplete; missing {required}.")
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. The holdout has already been evaluated for "
            "this run and is never overwritten."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("workflow") != "final_refit":
        raise ValueError(f"{metadata_path} is not final-refit metadata.")
    if trainer.file_sha256(checkpoint_path) != metadata.get("checkpoint_sha256"):
        raise ValueError(
            "final.pth no longer matches the frozen checkpoint recorded in final.json."
        )

    split_manifest = Path(split_manifest)
    manifest_hash = trainer.file_sha256(split_manifest)
    if manifest_hash != metadata.get("split_manifest_sha256"):
        raise ValueError(
            "The split manifest changed after final training. Restore the exact frozen manifest "
            "before evaluating the holdout."
        )
    manifest = data_splits.load_manifest(split_manifest)
    recorded_sessions = sorted(metadata.get("holdout_sessions", []))
    current_sessions = sorted(data_splits.holdout_sessions(manifest))
    if current_sessions != recorded_sessions:
        raise ValueError(
            f"Holdout sessions changed after training: recorded {recorded_sessions}, "
            f"current {current_sessions}."
        )

    image_paths, mask_paths = data_splits.holdout_paths(manifest, data_root)
    if not image_paths:
        raise ValueError("The frozen manifest contains no holdout images.")
    dataset = trainer.SegmentationDataset(image_paths, mask_paths, augment=False)
    device = trainer.resolve_device(device_preference)
    loader = DataLoader(
        dataset,
        batch_size=int(metadata["config"]["batch_size"]),
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = trainer.load_unet_checkpoint(checkpoint_path, device)
    val_loss, probabilities, targets = trainer._collect_validation(
        model,
        loader,
        torch.nn.BCEWithLogitsLoss(),
        device,
    )
    threshold = float(metadata["prediction_threshold"])
    config = metadata["config"]
    report = trainer.evaluate_thresholds(
        probabilities,
        targets,
        (threshold,),
        float(config["tiny_max_diameter"]),
        float(config["large_min_diameter"]),
        float(config["low_circularity_cutoff"]),
        "macro_iou",
    )

    iou, _ = trainer.per_image_overlap_scores(probabilities, targets, threshold)
    session_by_path = data_splits.session_of_path(manifest, data_root)
    grouped: dict[str, list[float]] = defaultdict(list)
    for path, score in zip(image_paths, iou.cpu().numpy().tolist()):
        grouped[session_by_path[Path(path).resolve()]].append(float(score))
    per_session = {session: statistics.fmean(scores) for session, scores in sorted(grouped.items())}
    mean_per_session = statistics.fmean(per_session.values())
    worst_session = min(per_session, key=per_session.get)

    payload = {
        "workflow": "outer_holdout_evaluation",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": metadata["run_name"],
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "split_manifest_sha256": manifest_hash,
        "holdout_sessions": current_sessions,
        "holdout_examples": len(image_paths),
        "prediction_threshold": threshold,
        "validation_loss": val_loss,
        "macro_iou": report.macro_iou,
        "balanced_iou": report.balanced_iou,
        "macro_dice": report.macro_dice,
        "size_iou": report.size_iou,
        "low_circularity_iou": report.low_circularity_iou,
        "mean_per_session_iou": mean_per_session,
        "worst_session": worst_session,
        "worst_session_iou": per_session[worst_session],
        "per_session_iou": per_session,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=Path("training_data_split.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--confirm-frozen",
        action="store_true",
        help="Confirm that all model, schedule, and threshold choices are frozen and this is "
        "the one-shot outer-holdout evaluation.",
    )
    args = parser.parse_args(argv)
    if not args.confirm_frozen:
        parser.error(
            "Pass --confirm-frozen only after model, schedule, and threshold choices are frozen."
        )

    payload = evaluate_holdout(
        args.run_dir,
        args.data_root.resolve(),
        args.split_manifest.resolve(),
        args.device,
    )
    print(f"Holdout examples      : {payload['holdout_examples']}")
    print(f"Mean per-session IoU  : {payload['mean_per_session_iou']:.4f}")
    print(f"Image-weighted IoU    : {payload['macro_iou']:.4f}")
    print(
        f"Worst session         : {payload['worst_session']} "
        f"({payload['worst_session_iou']:.4f})"
    )
    print(f"Wrote {Path(args.run_dir) / 'holdout.json'}")
    print(
        "The holdout is now consumed. Any training change based on this result requires a new "
        "untouched holdout for the next unbiased evaluation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
