"""Focused coverage for fine-tuning-era validation and sampling helpers."""

import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING = runpy.run_path(str(PROJECT_ROOT / "training" / "run_train.py"))

evaluate_thresholds = TRAINING["evaluate_thresholds"]
per_image_overlap_scores = TRAINING["per_image_overlap_scores"]
size_balanced_sample_weights = TRAINING["size_balanced_sample_weights"]
default_run_name = TRAINING["default_run_name"]
TrainingConfig = TRAINING["TrainingConfig"]
training_main = TRAINING["main"]


def test_per_image_iou_does_not_let_a_large_mask_hide_a_missed_small_mask():
    targets = torch.zeros((2, 1, 4, 4))
    targets[0, 0, 0, 0] = 1
    targets[1] = 1
    probabilities = torch.zeros_like(targets)
    probabilities[1] = 0.99

    iou, dice = per_image_overlap_scores(probabilities, targets, threshold=0.7)

    assert iou.tolist() == pytest.approx([0.0, 1.0])
    assert float(iou.mean()) == pytest.approx(0.5)
    assert float(dice.mean()) == pytest.approx(0.5)


def test_threshold_calibration_uses_equal_weighted_size_bins():
    targets = torch.zeros((3, 1, 12, 12))
    targets[0, 0, 5, 5] = 1
    targets[1, 0, 4:7, 4:7] = 1
    targets[2, 0, 3:8, 3:8] = 1
    probabilities = torch.full_like(targets, 0.1)
    probabilities[targets.bool()] = 0.6

    report = evaluate_thresholds(
        probabilities,
        targets,
        thresholds=(0.5, 0.7),
        tiny_max_diameter=2.0,
        large_min_diameter=4.0,
        low_circularity_cutoff=0.45,
    )

    assert report.threshold == 0.5
    assert report.macro_iou == pytest.approx(1.0)
    assert report.balanced_iou == pytest.approx(1.0)
    assert report.size_iou == pytest.approx({"tiny": 1.0, "medium": 1.0, "large": 1.0})


def test_balanced_sampling_gives_each_size_bin_equal_total_weight():
    labels = np.asarray(["tiny", "tiny", "medium", "medium", "medium", "large"])

    weights = size_balanced_sample_weights(labels)

    totals = {label: float(weights[labels == label].sum()) for label in set(labels)}
    assert totals["tiny"] == pytest.approx(totals["medium"])
    assert totals["tiny"] == pytest.approx(totals["large"])


@pytest.mark.parametrize(
    ("balance_training_sizes", "expected"),
    [(False, "ft_natural_lr5e-5_s3"), (True, "ft_bal_lr5e-5_s3")],
)
def test_default_run_name_is_concise_and_describes_the_main_choices(
    tmp_path, balance_training_sizes, expected
):
    config = TrainingConfig(
        checkpoint_dir=tmp_path,
        finetune_checkpoint=tmp_path / "source.pth",
        finetune_learning_rate=5e-5,
        balance_training_sizes=balance_training_sizes,
        seed=3,
    )

    assert default_run_name(config) == expected


def test_run_name_cannot_escape_the_experiment_directory():
    with pytest.raises(ValueError, match="run_name"):
        TrainingConfig(run_name="../outside")


def test_terminal_entry_point_maps_arguments_to_training_config(monkeypatch, tmp_path):
    captured = []
    source = tmp_path / "source.pth"
    output = tmp_path / "runs"
    monkeypatch.setitem(training_main.__globals__, "run_training", captured.append)

    exit_code = training_main(
        [
            "--data-root",
            str(tmp_path),
            "--checkpoint-dir",
            str(output),
            "--run-name",
            "terminal-smoke",
            "--finetune-checkpoint",
            str(source),
            "--learning-rate",
            "5e-5",
            "--epochs",
            "3",
            "--natural-sampling",
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    assert len(captured) == 1
    config = captured[0]
    assert config.data_root == tmp_path.resolve()
    assert config.checkpoint_dir == output.resolve()
    assert config.run_name == "terminal-smoke"
    assert config.finetune_checkpoint == source
    assert config.finetune_learning_rate == pytest.approx(5e-5)
    assert config.n_epochs == 3
    assert not config.balance_training_sizes
    assert config.seed == 7
