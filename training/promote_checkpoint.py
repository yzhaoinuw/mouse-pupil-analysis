# -*- coding: utf-8 -*-
"""Promote an experimental training run to packaged checkpoint data.

`run_train.py` writes every run to `checkpoints_exp/<run-name>/` as `best.pth`,
`best.json`, and `train.log`. This script performs the single transformation
that turns one such folder into the three files the package ships, so a
promotion is reproducible rather than hand-assembled:

- renames all three to the concise `<count>pupils_thresh=<value>_iou=<macro>` pattern
  that `find_default_checkpoint(...)` and `resolve_prediction_threshold(...)` read,
- drops local absolute paths from the metadata and the log header,
- keeps run provenance under `training`, and records the honest scope of the
  reported numbers in `validation_note`.

Promotion stays a deliberate act. Review the candidate as documented in
`training/README.md` before running this, and remove or archive superseded
packaged checkpoints yourself; this script never deletes anything.

Typical use, from the repository root:

    python training/promote_checkpoint.py --run-dir checkpoints_exp/ft_natural_lr1e-4_s0 \
        --validation-note "Validation shares recording groups with training."
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGED_CHECKPOINT_DIR = PROJECT_ROOT / "mouse_pupil_analysis" / "checkpoints"

# The packaged metadata schema. `tests/test_promotion.py` asserts that the
# shipped JSON and this script's output both match it exactly, so a future
# promotion cannot quietly change the shape of package data.
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
)

# Absolute local paths in a run folder describe one machine, not the model.
_LOCAL_ONLY_CONFIG_KEYS = ("data_root", "checkpoint_dir")


def _recorded_basename(path_text: str) -> str:
    """Return the final component of a path recorded on any platform.

    Training commonly runs on Windows while promotion or CI may run elsewhere,
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
    """Transform one run's `best.json` into packaged checkpoint metadata."""
    missing = {"prediction_threshold", "macro_iou", "training_examples", "config"} - set(
        run_metadata
    )
    if missing:
        raise ValueError(
            f"Run metadata is missing {sorted(missing)}. It predates "
            "training/run_train.py's current provenance fields; re-run training or add "
            "the fields by hand before promoting."
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
            "sampling": "balanced" if config["balance_training_sizes"] else "natural",
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


def promote(
    run_dir: Path,
    checkpoints_dir: Path = PACKAGED_CHECKPOINT_DIR,
    validation_note: str = "",
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Path]:
    """Copy one run folder into `checkpoints_dir` under the packaged naming pattern."""
    run_dir = Path(run_dir)
    checkpoints_dir = Path(checkpoints_dir)
    weights = run_dir / "best.pth"
    metadata_path = run_dir / "best.json"
    log_path = run_dir / "train.log"
    for required in (weights, metadata_path, log_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"{run_dir} is not a complete run folder; missing {required.name}."
            )

    packaged = build_packaged_metadata(
        json.loads(metadata_path.read_text(encoding="utf-8")),
        validation_note,
    )
    basename = packaged_basename(packaged)
    targets = {
        "weights": checkpoints_dir / f"{basename}.pth",
        "metadata": checkpoints_dir / f"{basename}.json",
        "log": checkpoints_dir / f"training_log_{basename}.txt",
    }

    existing = [path for path in targets.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite "
            + ", ".join(path.name for path in existing)
            + ". Pass --force if you intend to replace them."
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
        "\nPromoted. Now update CHANGELOG.md, note any change in reported diameters, and "
        "run the checks in AGENTS.md before building the distributions."
    )
    return targets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote a checkpoints_exp run folder to packaged checkpoint data.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run folder containing best.pth, best.json, and train.log.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=PACKAGED_CHECKPOINT_DIR,
        help="Packaged checkpoint directory (default: mouse_pupil_analysis/checkpoints).",
    )
    parser.add_argument(
        "--validation-note",
        default="",
        help=(
            "One sentence on the scope of the reported metrics, stored in the packaged "
            "metadata. State it when validation is not independent of training."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite packaged files that already use the same names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned filenames without writing anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse terminal arguments and promote one run folder."""
    args = _build_parser().parse_args(argv)
    promote(
        run_dir=args.run_dir,
        checkpoints_dir=args.checkpoints_dir,
        validation_note=args.validation_note,
        force=args.force,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
