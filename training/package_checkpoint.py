# -*- coding: utf-8 -*-
"""Package a selected training run as installed checkpoint data.

Development runs contain `best.pth`, `best.json`, and `train.log`. This script performs
the single transformation that turns a complete selected run folder into the three files the package
ships, so packaging is reproducible rather than hand-assembled:

- renames all three to the concise `<count>pupils_thresh=<value>_iou=<macro>` pattern
  that `find_default_checkpoint(...)` and `resolve_prediction_threshold(...)` read,
- drops local absolute paths from the metadata and the log header,
- keeps run provenance under `training`, and records the honest scope of the
  reported numbers in `validation_note`.

Packaging stays a deliberate act. Review the candidate as documented in
`training/README.md` before running this, and remove or archive superseded
packaged checkpoints yourself; this script never deletes anything.

Typical use, from the repository root:

    python training/package_checkpoint.py --run_dir checkpoints_exp/ft_natural_lr1e-4_s0 \
        --validation_note "Validation shares recording groups with training."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGED_CHECKPOINT_DIR = PROJECT_ROOT / "mouse_pupil_analysis" / "checkpoints"

# The packaged metadata schema. `tests/test_package_checkpoint.py` asserts that the
# shipped JSON and this script's output both match it exactly, so future packaging
# cannot quietly change the shape of package data.
PACKAGED_KEYS = (
    "run_name",
    "prediction_threshold",
    "best_epoch",
    "balanced_iou",
    "macro_iou",
    "macro_dice",
    "size_iou",
    "low_circularity_iou",
    "validation_note",
    "training",
)

PACKAGED_TRAINING_KEYS = (
    "mode",
    "source_checkpoint",
    "training_examples",
    "sampling",
    "use_attention",
    "batch_size",
    "learning_rate",
    "seed",
    "early_stopping_patience",
    "scheduler_patience",
    "tiny_max_diameter",
    "large_min_diameter",
    "workflow",
    "selection_metric",
    "scheduler_metric",
    "selection_threshold",
    "threshold_candidates",
    "split_manifest_sha256",
)

# Absolute local paths in a run folder describe one machine, not the model.
_LOCAL_ONLY_CONFIG_KEYS = (
    "data_root",
    "labeled_frames_dir",
    "checkpoint_dir",
    "split_manifest",
    "training_config_path",
)


def _recorded_basename(path_text: str) -> str:
    """Return the final component of a path recorded on any platform.

    Training commonly runs on Windows while packaging or CI may run elsewhere,
    and `Path("C:\\...\\best.pth").name` is the whole string on POSIX. Splitting
    on both separators keeps local paths out of package data either way.
    """
    return re.split(r"[\\/]", path_text)[-1]


def packaged_basename(metadata: dict) -> str:
    """Return the concise packaged filename stem for one run's metadata.

    The training-set size, calibrated threshold, and macro IoU are the parts
    inference and checkpoint selection actually read. Architecture and the
    148 x 148 resize are universal, so they stay out of the name.
    """
    examples = int(metadata["training"]["training_examples"])
    threshold = float(metadata["prediction_threshold"])
    macro_iou = float(metadata["macro_iou"])
    return f"{examples}pupils_thresh={threshold:g}_iou={macro_iou:.4f}"


def build_packaged_metadata(run_metadata: dict, validation_note: str = "") -> dict:
    """Transform complete development or final-run metadata into package metadata."""
    missing = {"prediction_threshold", "macro_iou", "training_examples", "config"} - set(
        run_metadata
    )
    if missing:
        raise ValueError(
            f"Run metadata is missing {sorted(missing)}. It predates "
            "training/run_train.py's current provenance fields; re-run training or add "
            "the fields by hand before packaging."
        )

    config = run_metadata["config"]
    source_checkpoint = config.get("finetune_checkpoint")
    packaged = {
        "run_name": run_metadata["run_name"],
        "prediction_threshold": float(run_metadata["prediction_threshold"]),
        "best_epoch": int(run_metadata["best_epoch"]),
        "balanced_iou": float(run_metadata["balanced_iou"]),
        "macro_iou": float(run_metadata["macro_iou"]),
        "macro_dice": float(run_metadata["macro_dice"]),
        "size_iou": run_metadata["size_iou"],
        "low_circularity_iou": run_metadata["low_circularity_iou"],
        "validation_note": validation_note,
        "training": {
            "mode": run_metadata["training_mode"],
            # Only the name travels; the training machine's path does not.
            "source_checkpoint": (
                None if source_checkpoint is None else _recorded_basename(source_checkpoint)
            ),
            "training_examples": int(run_metadata["training_examples"]),
            "sampling": "natural",
            "use_attention": bool(config["use_attention"]),
            "batch_size": int(config["batch_size"]),
            # The rate the run started from, not the decayed rate at the best epoch.
            "learning_rate": float(
                config["finetune_learning_rate"]
                if source_checkpoint is not None
                else config["scratch_learning_rate"]
            ),
            "seed": int(config["seed"]),
            "early_stopping_patience": int(config["early_stopping_patience"]),
            "scheduler_patience": int(config["scheduler_patience"]),
            "tiny_max_diameter": float(config["tiny_max_diameter"]),
            "large_min_diameter": float(config["large_min_diameter"]),
            "workflow": run_metadata.get("workflow", "development_selection"),
            "selection_metric": config.get("selection_metric", "balanced_iou"),
            "scheduler_metric": config.get("scheduler_metric", "balanced_iou"),
            "selection_threshold": (
                "calibrated"
                if config.get("selection_threshold") is None
                else float(config["selection_threshold"])
            ),
            "threshold_candidates": list(config.get("threshold_candidates", [])),
            "split_manifest_sha256": run_metadata.get("split_manifest_sha256"),
        },
    }
    return {key: packaged[key] for key in PACKAGED_KEYS}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_all_labeled_packaged_metadata(
    run_metadata: dict,
    training_config_path: Path,
    validation_note: str,
) -> dict:
    """Build honest package metadata for a fixed all-labeled refit.

    The final weights have no validation set by design.  Their filename therefore uses
    the mean macro IoU from the CV run that selected the training recipe; the note makes
    clear that this is recipe-selection evidence, not an evaluation of the final weights.
    """
    if not validation_note:
        raise ValueError(
            "all-labeled packaging requires a validation_note describing its CV scope."
        )
    required = {
        "run_name",
        "workflow",
        "training_mode",
        "training_examples",
        "trained_epochs",
        "prediction_threshold",
        "training_config_sha256",
        "config",
    }
    missing = required - set(run_metadata)
    if missing:
        raise ValueError(f"All-labeled metadata is missing {sorted(missing)}.")
    if run_metadata["workflow"] != "all_labeled_training":
        raise ValueError("Expected all_labeled_training metadata.")

    training_config_path = Path(training_config_path)
    if not training_config_path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {training_config_path}")
    if _file_sha256(training_config_path) != run_metadata["training_config_sha256"]:
        raise ValueError("Training configuration does not match the all-labeled run metadata.")
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    summary_path = training_config_path.parent / training_config["source_cv_summary"]
    if not summary_path.is_file():
        raise FileNotFoundError(f"CV summary not found: {summary_path}")
    if _file_sha256(summary_path) != training_config["source_cv_summary_sha256"]:
        raise ValueError("CV summary does not match the training configuration.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    folds = summary.get("per_fold", [])
    if not summary.get("complete_cv") or not folds:
        raise ValueError("All-labeled packaging requires a complete CV summary with fold metrics.")

    macro_iou = sum(float(fold["macro_iou"]) for fold in folds) / len(folds)
    balanced_iou = sum(float(fold["balanced_iou"]) for fold in folds) / len(folds)
    config = run_metadata["config"]
    source_checkpoint = config.get("finetune_checkpoint")
    packaged = {
        "run_name": run_metadata["run_name"],
        "prediction_threshold": float(run_metadata["prediction_threshold"]),
        "best_epoch": int(run_metadata["trained_epochs"]),
        "balanced_iou": balanced_iou,
        "macro_iou": macro_iou,
        # The CV summary deliberately contains only the promotion metrics it needs.
        "macro_dice": None,
        "size_iou": None,
        "low_circularity_iou": None,
        "validation_note": validation_note,
        "training": {
            "mode": run_metadata["training_mode"],
            "source_checkpoint": (
                None if source_checkpoint is None else _recorded_basename(source_checkpoint)
            ),
            "training_examples": int(run_metadata["training_examples"]),
            "sampling": training_config["sampling"],
            "use_attention": bool(config["use_attention"]),
            "batch_size": int(config["batch_size"]),
            "learning_rate": float(config["scratch_learning_rate"]),
            "seed": int(config["seed"]),
            "early_stopping_patience": int(config["early_stopping_patience"]),
            "scheduler_patience": int(config["scheduler_patience"]),
            "tiny_max_diameter": float(config["tiny_max_diameter"]),
            "large_min_diameter": float(config["large_min_diameter"]),
            "workflow": run_metadata["workflow"],
            "selection_metric": config["selection_metric"],
            "scheduler_metric": config["scheduler_metric"],
            "selection_threshold": float(config["selection_threshold"]),
            "threshold_candidates": list(config["threshold_candidates"]),
            "split_manifest_sha256": None,
        },
    }
    return {key: packaged[key] for key in PACKAGED_KEYS}


def redact_log_header(header_line: str) -> str:
    """Strip machine-specific paths from a training log's JSON header line."""
    header = json.loads(header_line)
    for key in _LOCAL_ONLY_CONFIG_KEYS:
        header.pop(key, None)
    if header.get("finetune_checkpoint") is not None:
        header["finetune_checkpoint"] = _recorded_basename(header["finetune_checkpoint"])
    return json.dumps(header, sort_keys=True)


def _redacted_log(log_path: Path) -> str:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{log_path} is empty.")
    return "\n".join([redact_log_header(lines[0]), *lines[1:]]) + "\n"


def _load_packagable_run(run_dir: Path) -> tuple[Path, dict, Path]:
    """Load a complete validation-selected or fixed all-labeled run."""
    all_data_metadata = run_dir / "all_data.json"
    if all_data_metadata.is_file():
        weights = run_dir / "all_data.pth"
        log_path = run_dir / "train.log"
        for required in (weights, log_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"{run_dir} is not a complete all-labeled run; missing {required.name}."
                )
        return weights, json.loads(all_data_metadata.read_text(encoding="utf-8")), log_path

    log_path = run_dir / "train.log"
    weights = run_dir / "best.pth"
    metadata_path = run_dir / "best.json"
    for required in (weights, metadata_path, log_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"{run_dir} is not a complete development run; missing {required.name}."
            )
    return weights, json.loads(metadata_path.read_text(encoding="utf-8")), log_path


def package_checkpoint(
    run_dir: Path,
    checkpoints_dir: Path = PACKAGED_CHECKPOINT_DIR,
    validation_note: str = "",
    training_config_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Path]:
    """Copy one run folder into `checkpoints_dir` under the packaged naming pattern."""
    run_dir = Path(run_dir)
    checkpoints_dir = Path(checkpoints_dir)
    weights, run_metadata, log_path = _load_packagable_run(run_dir)
    if run_metadata.get("workflow") == "all_labeled_training":
        if training_config_path is None:
            raise ValueError(
                "all-labeled packaging requires training_config_path so CV provenance can be verified."
            )
        packaged = build_all_labeled_packaged_metadata(
            run_metadata, training_config_path, validation_note
        )
    else:
        packaged = build_packaged_metadata(run_metadata, validation_note)
    basename = packaged_basename(packaged)
    targets = {
        "weights": checkpoints_dir / f"{basename}.pth",
        "metadata": checkpoints_dir / f"{basename}.json",
        "log": checkpoints_dir / f"training_log_{basename}.txt",
    }

    existing = [path for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite "
            + ", ".join(path.name for path in existing)
            + ". Remove or archive the existing package files before packaging again."
        )

    superseded = sorted(
        path.name for path in checkpoints_dir.glob("*.pth") if path != targets["weights"]
    )
    if superseded:
        print(
            "Note: the release workflow ships exactly one checkpoint. Remove or archive "
            f"these deliberately: {', '.join(superseded)}"
        )

    for label, target in targets.items():
        print(f"{'Would write' if dry_run else 'Writing'} {label}: {target}")
    if dry_run:
        return targets

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, targets["weights"])
    targets["metadata"].write_text(json.dumps(packaged, indent=2) + "\n", encoding="utf-8")
    targets["log"].write_text(_redacted_log(log_path), encoding="utf-8")
    print(
        "\nPackaged. Now update CHANGELOG.md, note any change in reported diameters, and "
        "run the checks in AGENTS.md before building the distributions."
    )
    return targets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package a selected checkpoints_exp run as installed checkpoint data.",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Validation-selected development run containing best.pth, best.json, and train.log.",
    )
    parser.add_argument(
        "--training_config_path",
        type=Path,
        help=(
            "CV-generated all-labeled recipe used by an all_data.* run. Required only when "
            "packaging that run type."
        ),
    )
    parser.add_argument(
        "--validation_note",
        default="",
        help=(
            "One sentence on the scope of the reported metrics, stored in the packaged "
            "metadata. State it when validation is not independent of training."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the planned filenames without writing anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse terminal arguments and package one selected run folder."""
    args = _build_parser().parse_args(argv)
    package_checkpoint(
        run_dir=args.run_dir,
        validation_note=args.validation_note,
        training_config_path=args.training_config_path,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
