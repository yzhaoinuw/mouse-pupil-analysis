"""The packaged checkpoint metadata must stay reproducible from a run folder.

The shipped `.json` used to be assembled by hand, so it drifted from what
`training/run_train.py` actually writes and could not be regenerated. These
tests pin both ends of `training/package_checkpoint.py`: the transform it
applies, and the shape of the package data it produces.
"""

import hashlib
import json
import runpy
from pathlib import Path

import pytest
import torch

from mouse_pupil_analysis.pupil_predictions import (
    find_default_checkpoint,
    resolve_prediction_threshold,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = runpy.run_path(str(PROJECT_ROOT / "training" / "package_checkpoint.py"))

build_packaged_metadata = PACKAGE["build_packaged_metadata"]
packaged_basename = PACKAGE["packaged_basename"]
redact_log_header = PACKAGE["redact_log_header"]
package_checkpoint = PACKAGE["package_checkpoint"]
PACKAGED_KEYS = PACKAGE["PACKAGED_KEYS"]
PACKAGED_TRAINING_KEYS = PACKAGE["PACKAGED_TRAINING_KEYS"]


def _scratch_run_metadata() -> dict:
    """A from-scratch run folder, which records no source checkpoint."""
    return {
        "run_name": "full_scratch_s3",
        "training_mode": "scratch",
        "training_examples": 166,
        "prediction_threshold": 0.5,
        "best_epoch": 202,
        "balanced_iou": 0.8824594219525655,
        "macro_iou": 0.8786166310310364,
        "macro_dice": 0.9276593327522278,
        "size_iou": {"tiny": 0.8278, "medium": 0.8604, "large": 0.9592},
        "low_circularity_iou": None,
        "learning_rate": 3.125e-05,
        "meets_promotion_target": True,
        "config": {
            "data_root": "/Users/yuezhao/python_projects/pupil_tracking",
            "checkpoint_dir": "/tmp/runs_full",
            "run_name": "full_scratch_s3",
            "finetune_checkpoint": None,
            "use_attention": True,
            "batch_size": 8,
            "scratch_learning_rate": 0.001,
            "finetune_learning_rate": 0.0001,
            "n_epochs": 400,
            "early_stopping_patience": 40,
            "scheduler_patience": 8,
            "balance_training_sizes": False,
            "seed": 3,
            "tiny_max_diameter": 15.0,
            "large_min_diameter": 80.0,
        },
    }


def _run_metadata() -> dict:
    """A fine-tuning `best.json` in the shape `run_train.py` writes one."""
    return {
        "run_name": "ft_natural_lr1e-4_s0",
        "training_mode": "fine_tune",
        "training_examples": 166,
        "prediction_threshold": 0.4,
        "best_epoch": 25,
        "balanced_iou": 0.8689939975738525,
        "macro_iou": 0.8749180436134338,
        "macro_dice": 0.9290293455123901,
        "size_iou": {"tiny": 0.795, "medium": 0.859, "large": 0.953},
        "low_circularity_iou": None,
        "learning_rate": 0.0001,
        "meets_promotion_target": True,
        "config": {
            "data_root": "C:\\Users\\yzhao\\pupil_tracking",
            "checkpoint_dir": "C:\\Users\\yzhao\\pupil_tracking\\checkpoints_exp",
            "run_name": "ft_natural_lr1e-4_s0",
            "finetune_checkpoint": (
                "C:\\Users\\yzhao\\pupil_tracking\\mouse_pupil_analysis\\checkpoints"
                "\\unet_atn_resize_166pupils_thresh=0.7_iou=0.9158.pth"
            ),
            "use_attention": True,
            "batch_size": 8,
            "scratch_learning_rate": 0.001,
            "finetune_learning_rate": 0.0001,
            "n_epochs": 250,
            "early_stopping_patience": 40,
            "scheduler_patience": 8,
            "balance_training_sizes": False,
            "seed": 0,
            "tiny_max_diameter": 15.0,
            "large_min_diameter": 80.0,
        },
    }


def test_packaged_metadata_matches_the_packaging_schema():
    packaged = json.loads(
        find_default_checkpoint().with_suffix(".json").read_text(encoding="utf-8")
    )

    assert tuple(packaged) == PACKAGED_KEYS
    assert tuple(packaged["training"]) == PACKAGED_TRAINING_KEYS


def test_packaging_reproduces_the_packaged_metadata_shape():
    packaged = build_packaged_metadata(_run_metadata(), validation_note="Shares recordings.")
    shipped = json.loads(find_default_checkpoint().with_suffix(".json").read_text(encoding="utf-8"))

    assert tuple(packaged) == tuple(shipped)
    assert tuple(packaged["training"]) == tuple(shipped["training"])
    assert packaged["training"]["mode"] == "fine_tune"
    assert packaged["training"]["sampling"] == "natural"
    assert packaged["prediction_threshold"] == 0.4
    assert packaged["training"]["selection_metric"] == "balanced_iou"
    assert packaged["training"]["split_manifest_sha256"] is None

    # A from-scratch run produces the identical shape; only provenance values differ.
    assert tuple(build_packaged_metadata(_scratch_run_metadata())) == tuple(shipped)


def test_packaging_strips_local_paths_from_metadata_and_log():
    packaged = build_packaged_metadata(_run_metadata())
    header = redact_log_header(
        json.dumps(
            {
                "data_root": "C:\\Users\\yzhao\\pupil_tracking",
                "checkpoint_dir": "C:\\Users\\yzhao\\pupil_tracking\\checkpoints_exp",
                "finetune_checkpoint": "C:\\Users\\yzhao\\weights\\source.pth",
                "seed": 0,
            }
        )
    )

    assert packaged["training"]["source_checkpoint"] == (
        "unet_atn_resize_166pupils_thresh=0.7_iou=0.9158.pth"
    )
    assert "yzhao" not in json.dumps(packaged)
    assert json.loads(header) == {"finetune_checkpoint": "source.pth", "seed": 0}


def test_packaged_basename_is_derived_from_its_run_metadata():
    assert (
        packaged_basename(build_packaged_metadata(_run_metadata()))
        == "166pupils_thresh=0.4_iou=0.8749"
    )


def test_scratch_packaging_records_no_source_checkpoint():
    packaged = build_packaged_metadata(_scratch_run_metadata())

    assert packaged["training"]["mode"] == "scratch"
    assert packaged["training"]["source_checkpoint"] is None
    assert packaged["training"]["learning_rate"] == pytest.approx(1e-3)
    assert packaged_basename(packaged) == "166pupils_thresh=0.5_iou=0.8786"


def test_packaging_records_the_initial_rate_not_the_decayed_rate():
    metadata = _run_metadata()
    # `learning_rate` in a run's metadata is the scheduler's rate at the best
    # epoch, which is not the rate the run started from.
    metadata["learning_rate"] = 6.25e-06

    assert build_packaged_metadata(metadata)["training"]["learning_rate"] == pytest.approx(1e-4)


def test_packaging_rejects_run_metadata_without_provenance_fields():
    metadata = _run_metadata()
    del metadata["training_examples"]

    with pytest.raises(ValueError, match="training_examples"):
        build_packaged_metadata(metadata)


def test_packaging_writes_the_three_files_and_refuses_to_clobber(tmp_path):
    run_dir = tmp_path / "ft_natural_lr1e-4_s0"
    run_dir.mkdir()
    torch.save({"weight": torch.zeros(1)}, run_dir / "best.pth")
    (run_dir / "best.json").write_text(json.dumps(_run_metadata()), encoding="utf-8")
    (run_dir / "train.log").write_text(
        json.dumps({"data_root": "C:\\Users\\yzhao", "seed": 0}) + "\nEpoch 001 | ...\n",
        encoding="utf-8",
    )
    packaged_dir = tmp_path / "checkpoints"

    targets = package_checkpoint(
        run_dir,
        checkpoints_dir=packaged_dir,
        validation_note="Shares recordings.",
    )

    assert sorted(path.name for path in packaged_dir.iterdir()) == [
        "166pupils_thresh=0.4_iou=0.8749.json",
        "166pupils_thresh=0.4_iou=0.8749.pth",
        "training_log_166pupils_thresh=0.4_iou=0.8749.txt",
    ]
    assert resolve_prediction_threshold(targets["weights"]) == 0.4
    assert "yzhao" not in targets["log"].read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="Remove or archive"):
        package_checkpoint(run_dir, checkpoints_dir=packaged_dir)


def test_all_labeled_packaging_verifies_cv_recipe_and_records_its_scope(tmp_path):
    summary = {
        "complete_cv": True,
        "per_fold": [
            {"macro_iou": 0.600, "balanced_iou": 0.610},
            {"macro_iou": 0.616, "balanced_iou": 0.630},
        ],
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    recipe = {
        "source_cv_summary": "summary.json",
        "source_cv_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "sampling": "natural",
    }
    recipe_path = tmp_path / "training_config.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")

    run_dir = tmp_path / "all_data"
    run_dir.mkdir()
    run_metadata = {
        "run_name": "all_data",
        "workflow": "all_labeled_training",
        "training_mode": "scratch",
        "training_examples": 615,
        "trained_epochs": 100,
        "prediction_threshold": 0.5,
        "training_config_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "config": {
            "finetune_checkpoint": None,
            "use_attention": True,
            "batch_size": 8,
            "scratch_learning_rate": 0.001,
            "seed": 0,
            "early_stopping_patience": 40,
            "scheduler_patience": 8,
            "tiny_max_diameter": 15.0,
            "large_min_diameter": 80.0,
            "selection_metric": "macro_iou",
            "scheduler_metric": "val_loss",
            "selection_threshold": 0.5,
            "threshold_candidates": [0.5, 0.55],
        },
    }
    (run_dir / "all_data.pth").write_bytes(b"weights")
    (run_dir / "all_data.json").write_text(json.dumps(run_metadata), encoding="utf-8")
    (run_dir / "train.log").write_text('{"checkpoint_dir": "C:\\\\local"}\n', encoding="utf-8")

    targets = package_checkpoint(
        run_dir,
        checkpoints_dir=tmp_path / "checkpoints",
        training_config_path=recipe_path,
        validation_note="Final refit; CV metrics select its recipe, not its weights.",
    )
    metadata = json.loads(targets["metadata"].read_text(encoding="utf-8"))

    assert targets["weights"].name == "615pupils_thresh=0.5_iou=0.6080.pth"
    assert metadata["macro_iou"] == pytest.approx(0.608)
    assert metadata["balanced_iou"] == pytest.approx(0.62)
    assert metadata["macro_dice"] is None
    assert metadata["training"]["workflow"] == "all_labeled_training"


def test_packaged_metadata_states_the_scope_of_its_numbers():
    packaged = json.loads(
        find_default_checkpoint().with_suffix(".json").read_text(encoding="utf-8")
    )

    assert packaged["validation_note"].strip(), "Packaged metadata must state its validation scope."
