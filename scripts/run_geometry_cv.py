"""Complete grouped CV for the ungated geometry model from a verified fold-2 run."""

import argparse
import ast
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_augmentation_experiment import read_json  # noqa: E402
from scripts.run_geometry_experiment import check_initialization  # noqa: E402
from scripts.run_loss_experiment import snapshot, write_json  # noqa: E402
from training import _trainer as trainer  # noqa: E402
from training import prepare_splits  # noqa: E402
from training._geometry import (  # noqa: E402
    GEOMETRY_LOSS_WEIGHT,
    SMOOTH_L1_BETA_PIXELS,
    GeometryUNet,
)
from training.run_cross_validation import all_labeled_training_config  # noqa: E402

FOLD_2_TRAINER_SHA256 = "09745118c518942cc9b143ee99d0e3a0aeb71f0c5bf49f95587941fb8b638a8a"
FOLD_2_CV_TRAINER_SHA256 = "fc7389d09fc6331906e3df9a61d67705d49933041201e001179314118df39bc1"


def set_runtime():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)


def verify_files(root, hashes):
    for path, digest in hashes.items():
        source = root / path
        if trainer.file_sha256(source) == digest:
            continue
        # Git can change Python line endings when checking out an otherwise identical source.
        if source.suffix == ".py":
            lf = source.read_bytes().replace(b"\r\n", b"\n")
            if digest in (
                hashlib.sha256(lf).hexdigest(),
                hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest(),
            ):
                continue
        # The release refit extends the all-labeled function, which these CV folds never call.
        if Path(path) == Path("training/_trainer.py") and digest == FOLD_2_TRAINER_SHA256:
            tree = ast.parse(source.read_bytes())
            tree.body = [
                node
                for node in tree.body
                if not (
                    isinstance(node, ast.FunctionDef) and node.name == "run_all_labeled_training"
                )
            ]
            if hashlib.sha256(ast.dump(tree).encode()).hexdigest() == FOLD_2_CV_TRAINER_SHA256:
                continue
        raise ValueError(f"File differs from the recorded experiment: {path}")


def evaluate_fold(config, checkpoint, thresholds):
    """Score ungated predictions by session at each requested threshold."""
    device = trainer.resolve_device(config.device)
    model = GeometryUNet(use_attention=config.use_attention)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    _, dataset = trainer.make_split_datasets(config)
    sessions = prepare_splits.session_of_path(
        prepare_splits.load_manifest(config.split_manifest), config.data_root
    )
    _, probabilities, targets = trainer._collect_validation(
        model.to(device),
        DataLoader(dataset, batch_size=config.batch_size),
        trainer.make_criterion(config),
        device,
    )
    target_present = targets.flatten(1).any(dim=1)
    metrics = {}
    for label, threshold in thresholds.items():
        ious, _ = trainer.per_image_overlap_scores(probabilities, targets, threshold)
        predicted = (probabilities > threshold).flatten(1).any(dim=1)
        grouped = defaultdict(list)
        for path, iou in zip(dataset.image_paths, ious.tolist(), strict=True):
            grouped[sessions[Path(path).resolve()]].append(iou)
        metrics[label] = {
            "threshold": threshold,
            "image_mean_iou": float(ious.mean()),
            "per_session_iou": {key: statistics.fmean(values) for key, values in grouped.items()},
            "empty_frames": int((~target_present).sum()),
            "empty_frames_with_foreground": int((~target_present & predicted).sum()),
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-dir", type=Path, required=True, help="Completed geometry screen.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reuse, output = args.reuse_dir.resolve(), args.output_dir.resolve()
    set_runtime()
    reference = read_json(reuse / "experiment.json")
    recorded = reference["config"]
    config = trainer.TrainingConfig(
        labeled_frames_dir=Path(recorded["labeled_frames_dir"]),
        checkpoint_dir=Path(recorded["checkpoint_dir"]),
        split_manifest=Path(recorded["split_manifest"]),
        fold=recorded["fold"],
        seed=recorded["seed"],
        device=recorded["device"],
        console_interval=recorded["console_interval"],
        max_epochs=recorded["max_epochs"],
        early_stopping_patience=recorded["early_stopping_patience"],
    )
    if json.loads(json.dumps(trainer._jsonable_config(config))) != recorded:
        raise ValueError("Training settings differ from the reusable fold.")
    if (GEOMETRY_LOSS_WEIGHT, SMOOTH_L1_BETA_PIXELS) != (
        reference["geometry_loss_weight"],
        reference["smooth_l1_beta_pixels"],
    ):
        raise ValueError("Circle supervision differs from the reusable fold.")
    runtime = read_json(Path(reference["baseline"]).parent / "experiment.json")
    gpu = torch.cuda.get_device_name() if config.device == "cuda" else None
    if (torch.__version__, np.__version__, gpu) != (
        runtime["torch_version"],
        runtime["numpy_version"],
        runtime["gpu"],
    ):
        raise ValueError("Training runtime differs from the reusable fold.")
    initial_hash = check_initialization(config, reference["shared_initial_state_sha256"])
    if initial_hash != reference["candidate_initial_state_sha256"]:
        raise ValueError("Geometry initialization differs from the reusable fold.")
    split = snapshot(config)
    if split != read_json(reuse / "split.json"):
        raise ValueError("Source pairs or split assignments differ from the reusable fold.")
    sources = reference["source_sha256"]
    verify_files(ROOT, sources)
    original = read_json(reuse / "summary.json")
    checkpoint = reuse / "candidate/best.pth"
    if trainer.file_sha256(checkpoint) != original["candidate_checkpoint_sha256"]:
        raise ValueError("Reusable checkpoint differs from the verified screen.")
    if read_json(checkpoint.with_suffix(".json"))["best_epoch"] != original["candidate_best_epoch"]:
        raise ValueError("Reusable checkpoint metadata differs from the verified screen.")

    manifest = prepare_splits.load_manifest(config.split_manifest)
    folds = list(range(manifest["n_folds"]))
    record = {
        "reused_fold": config.fold,
        "reused_checkpoint": str(checkpoint),
        "reused_checkpoint_sha256": original["candidate_checkpoint_sha256"],
        "manifest_sha256": split["manifest_sha256"],
        "candidate_initial_state_sha256": initial_hash,
        "geometry_loss_weight": GEOMETRY_LOSS_WEIGHT,
        "smooth_l1_beta_pixels": SMOOTH_L1_BETA_PIXELS,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "gpu": gpu,
        "seed": config.seed,
        "source_sha256": sources,
    }
    output.mkdir(parents=True, exist_ok=True)
    run_record = output / "run.json"
    if run_record.exists():
        if read_json(run_record) != record:
            raise ValueError("The output directory belongs to a different CV run.")
    else:
        if any(output.iterdir()):
            raise FileExistsError(f"Choose a fresh CV output directory: {output}")
        write_json(run_record, record)
    print(f"Verified reusable fold {config.fold}. Completing folds {folds}.", flush=True)

    results = []
    for fold in folds:
        run_config = replace(config, fold=fold, checkpoint_dir=output / f"fold_{fold}")
        if fold == config.fold:
            weights = checkpoint
        else:
            done = run_config.checkpoint_dir / "complete.json"
            weights = run_config.checkpoint_dir / "best.pth"
            if done.exists():
                verify_files(done.parent, read_json(done))
            else:
                print(f"Training geometry fold {fold}.", flush=True)
                assert trainer.run_training(run_config, geometry_head=True) == weights
                if snapshot(config) != split:
                    raise ValueError("Source pairs or split assignments changed during training.")
                verify_files(ROOT, sources)
                write_json(
                    done,
                    {
                        name: trainer.file_sha256(done.parent / name)
                        for name in ("best.pth", "best.json", "train.log")
                    },
                )
        metadata = read_json(weights.with_suffix(".json"))
        expected = recorded if fold == config.fold else trainer._jsonable_config(run_config)
        if metadata["config"] != json.loads(json.dumps(expected)):
            raise ValueError(f"Unexpected checkpoint settings for fold {fold}.")
        results.append({"fold": fold, "metadata": metadata, "checkpoint": weights})
        print(
            f"Fold {fold}: selected epoch {metadata['best_epoch']}, "
            f"threshold {metadata['prediction_threshold']:.2f}.",
            flush=True,
        )

    threshold = statistics.median(r["metadata"]["prediction_threshold"] for r in results)
    per_session = {}
    per_fold = []
    for result in results:
        fold, metadata, weights = result["fold"], result["metadata"], result["checkpoint"]
        scores = evaluate_fold(
            replace(config, fold=fold),
            weights,
            {
                "fixed_0.5": 0.5,
                "fold_calibrated": metadata["prediction_threshold"],
                "refit": threshold,
            },
        )
        per_session.update(scores["refit"]["per_session_iou"])
        per_fold.append(
            {
                "fold": fold,
                "checkpoint": str(weights),
                "checkpoint_sha256": trainer.file_sha256(weights),
                "reused": fold == config.fold,
                "threshold": metadata["prediction_threshold"],
                "best_epoch": metadata["best_epoch"],
                "macro_iou": metadata["macro_iou"],
                "balanced_iou": metadata["balanced_iou"],
                "scores": scores,
            }
        )
    expected_sessions = {s["session"] for s in manifest["sessions"] if s["fold"] in folds}
    if set(per_session) != expected_sessions:
        raise ValueError("The four folds did not score exactly the development sessions.")
    summary = output / "summary.json"
    write_json(
        summary,
        {
            **record,
            "split_manifest": str(config.split_manifest),
            "folds": folds,
            "complete_cv": True,
            "geometry_head": True,
            "per_fold": per_fold,
            "refit_threshold": threshold,
            "per_session_iou": per_session,
            "equal_session_mean_iou": statistics.fmean(per_session.values()),
        },
    )
    recipe = all_labeled_training_config(summary, results, config, trainer)
    write_json(output / "training_config.json", {**recipe, "geometry_head": True})
    print(f"Complete CV. Refit recipe: {recipe['max_epochs']} epochs, threshold {threshold:g}.")


if __name__ == "__main__":
    main()
