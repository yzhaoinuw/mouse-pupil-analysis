"""Run one fresh or fine-tuned pupil-segmentation training job.

Pass terminal arguments for an ordinary command-line run. Running this file without
arguments uses the editable configuration block at the bottom.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__:
    from . import _trainer as trainer
    from . import prepare_splits
else:  # Direct ``python training/run_train.py`` execution from a source checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training import _trainer as trainer
    from training import prepare_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELED_FRAMES_DIR = PROJECT_ROOT / "labeled_frames"


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a fresh pupil UNet or fine-tune a compatible checkpoint.",
    )
    parser.add_argument(
        "--labeled_frames_dir",
        type=Path,
        default=Path.cwd() / "labeled_frames",
        help="Folder containing one <session>/images and <session>/masks pair per recording "
        "(default: ./labeled_frames).",
    )
    parser.add_argument(
        "--training_config_path",
        type=Path,
        help="Fixed all-labeled JSON recipe. It trains every labeled session and ignores "
        "training_data_split.json.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        help="Empty directory for this run's checkpoint and metadata (default: a new directory "
        "under checkpoints_exp/).",
    )
    parser.add_argument(
        "--finetune_checkpoint",
        type=Path,
        help="Compatible .pth weights to fine-tune; omit for fresh training.",
    )
    return parser


def _training_config(
    config_path: Path, labeled_frames_dir: Path, checkpoint_dir: Path | None
) -> trainer.TrainingConfig:
    """Load the cross-validation hand-off recipe for all-labeled training."""
    config_path = Path(config_path).resolve()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Training configuration not found: {config_path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Training configuration is not valid JSON: {config_path}") from error
    if payload.get("schema_version") != 1:
        raise ValueError("Training configuration must use schema_version 1.")

    finetune_checkpoint = payload.get("finetune_checkpoint")
    learning_rate = float(payload["learning_rate"])
    learning_rate_key = (
        "finetune_learning_rate" if finetune_checkpoint is not None else "scratch_learning_rate"
    )
    return trainer.TrainingConfig(
        labeled_frames_dir=labeled_frames_dir,
        checkpoint_dir=checkpoint_dir,
        finetune_checkpoint=None if finetune_checkpoint is None else Path(finetune_checkpoint),
        train_all_labeled_frames=True,
        training_config_path=config_path,
        lr_milestones=tuple(payload["lr_milestones"]),
        prediction_threshold=float(payload["prediction_threshold"]),
        use_attention=bool(payload["use_attention"]),
        batch_size=int(payload["batch_size"]),
        max_epochs=int(payload["max_epochs"]),
        seed=int(payload["seed"]),
        **{learning_rate_key: learning_rate},
    )


def main(argv: list[str] | None = None) -> int:
    """Parse terminal arguments and run training."""
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    labeled_frames_dir = args.labeled_frames_dir.resolve()
    if labeled_frames_dir.name != "labeled_frames":
        parser.error(
            f"--labeled_frames_dir must name a labeled_frames folder; got {labeled_frames_dir}."
        )
    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir is not None else None
    if args.training_config_path is not None:
        conflicting = [
            name
            for name, value in {
                "--finetune_checkpoint": args.finetune_checkpoint,
            }.items()
            if value is not None
        ]
        if conflicting:
            parser.error(
                "--training_config_path owns model and training settings; remove "
                + ", ".join(conflicting)
                + "."
            )
        config = _training_config(
            args.training_config_path,
            labeled_frames_dir,
            checkpoint_dir,
        )
    else:
        manifest = labeled_frames_dir.parent / prepare_splits.TRAINING_DATA_SPLIT_FILENAME
        if not manifest.is_file():
            parser.error(
                f"No {prepare_splits.TRAINING_DATA_SPLIT_FILENAME} beside {labeled_frames_dir}; "
                "create it with training/prepare_splits.py or pass --training_config_path "
                "for all-labeled training."
            )
        config = trainer.TrainingConfig(
            labeled_frames_dir=labeled_frames_dir,
            checkpoint_dir=checkpoint_dir,
            finetune_checkpoint=args.finetune_checkpoint,
            split_manifest=manifest,
        )
    trainer.run_training(config)
    return 0


def _run_direct_configuration() -> None:
    """Run the editable no-argument configuration below."""
    # Set this to a compatible .pth file to fine-tune its weights. Leave it as None
    # for fresh training. Fine-tuning automatically uses the lower learning rate.
    finetune_checkpoint = None
    # Example:
    # finetune_checkpoint = (
    #     PROJECT_ROOT
    #     / "mouse_pupil_analysis"
    #     / "checkpoints"
    #     / "166pupils_thresh=0.4_iou=0.8749.pth"
    # )
    trainer.run_training(
        trainer.TrainingConfig(
            labeled_frames_dir=LABELED_FRAMES_DIR,
            finetune_checkpoint=finetune_checkpoint,
            split_manifest=(
                LABELED_FRAMES_DIR.parent / prepare_splits.TRAINING_DATA_SPLIT_FILENAME
            ),
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    _run_direct_configuration()
