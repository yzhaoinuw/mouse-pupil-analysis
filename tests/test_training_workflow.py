"""Focused coverage for the compact training and CV hand-off workflow."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING = runpy.run_path(
    str(PROJECT_ROOT / "training" / "_trainer.py"), run_name="training._trainer"
)
RUN_TRAIN = runpy.run_path(str(PROJECT_ROOT / "training" / "run_train.py"))
CV = runpy.run_path(str(PROJECT_ROOT / "training" / "run_cross_validation.py"))

evaluate_thresholds = TRAINING["evaluate_thresholds"]
per_image_overlap_scores = TRAINING["per_image_overlap_scores"]
TrainingConfig = TRAINING["TrainingConfig"]
training_main = RUN_TRAIN["main"]
make_all_labeled_dataset = TRAINING["make_all_labeled_dataset"]
make_split_datasets = TRAINING["make_split_datasets"]
all_labeled_training_config = CV["all_labeled_training_config"]
selected_cv_folds = CV["selected_cv_folds"]


def test_grouped_training_defaults_to_macro_iou_selection():
    assert TrainingConfig().selection_metric == "macro_iou"


def test_bce_dice_uses_equal_weight_per_nonempty_image():
    logits = torch.zeros((3, 1, 2, 2), requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 0, 0, 0] = 1
    targets[1] = 1
    loss = TRAINING["BCESoftDiceLoss"]()(logits, targets)
    # p=0.5: Dice is 1/3 for the one-pixel target and 2/3 for the full target.
    assert float(loss.detach()) == pytest.approx(float(torch.log(torch.tensor(2.0))) + 0.5)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[targets.bool()] < 0).all()
    assert (logits.grad[~targets.bool()] > 0).all()


def test_bce_dice_empty_batch_keeps_bce_and_its_false_positive_gradient():
    logits = torch.full((2, 1, 4, 4), 10.0, requires_grad=True)
    targets = torch.zeros_like(logits)
    loss = TRAINING["BCESoftDiceLoss"]()(logits, targets)
    assert (
        loss.detach()
        == torch.nn.functional.binary_cross_entropy_with_logits(logits, targets).detach()
    )
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad > 0).all()


def test_bce_dice_rewards_correct_foreground_and_penalizes_spurious_regions():
    targets = torch.zeros((1, 1, 8, 8))
    targets[:, :, 3:5, 3:5] = 1
    perfect = torch.where(targets > 0, 20.0, -20.0)
    spurious = perfect.clone()
    spurious[:, :, :2, :2] = 20
    criterion = TRAINING["BCESoftDiceLoss"]()
    assert float(criterion(perfect, targets)) < 1e-6
    assert criterion(spurious, targets) > criterion(perfect, targets)
    assert isinstance(TRAINING["make_criterion"](TrainingConfig()), torch.nn.BCEWithLogitsLoss)
    with pytest.raises(ValueError, match="loss must be"):
        TrainingConfig(loss="invalid")


def test_normal_manifest_run_uses_the_validation_session(monkeypatch, tmp_path):
    development = ([tmp_path / "dev.png"], [tmp_path / "dev_mask.png"])
    validation = ([tmp_path / "validation.png"], [tmp_path / "validation_mask.png"])
    constructed = []

    class FakeDataset:
        def __init__(self, images, masks, augment):
            constructed.append((images, masks, augment))

    data_splits = make_split_datasets.__globals__["data_splits"]
    monkeypatch.setattr(data_splits, "load_manifest", lambda _: {})
    monkeypatch.setattr(
        data_splits, "validation_holdout_paths", lambda *_: (development, validation)
    )
    monkeypatch.setitem(make_split_datasets.__globals__, "SegmentationDataset", FakeDataset)

    train, held_out = make_split_datasets(
        TrainingConfig(
            labeled_frames_dir=tmp_path / "labeled_frames",
            split_manifest=tmp_path / "training_data_split.json",
        )
    )

    assert isinstance(train, FakeDataset)
    assert isinstance(held_out, FakeDataset)
    assert constructed == [
        (development[0], development[1], True),
        (validation[0], validation[1], False),
    ]


def test_all_labeled_dataset_ignores_splits_and_collects_every_session(monkeypatch, tmp_path):
    labeled_frames_dir = tmp_path / "labeled_frames"
    for session in ("session_a", "session_b"):
        (labeled_frames_dir / session / "images").mkdir(parents=True)
        (labeled_frames_dir / session / "masks").mkdir()
    (tmp_path / "training_data_split.json").write_text("not read", encoding="utf-8")
    paired = []

    def fake_pairs(images_dir, masks_dir):
        paired.append((images_dir, masks_dir))
        return [images_dir / "frame.png"], [masks_dir / "frame.png"]

    class FakeDataset:
        def __init__(self, images, masks, augment):
            self.images = images
            self.masks = masks
            self.augment = augment

    monkeypatch.setitem(make_all_labeled_dataset.__globals__, "paired_image_mask_paths", fake_pairs)
    monkeypatch.setitem(make_all_labeled_dataset.__globals__, "SegmentationDataset", FakeDataset)
    dataset = make_all_labeled_dataset(
        TrainingConfig(
            labeled_frames_dir=labeled_frames_dir,
            train_all_labeled_frames=True,
        )
    )

    assert len(dataset.images) == 2
    assert dataset.augment
    assert [path.parent.name for path, _ in paired] == ["session_a", "session_b"]


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


def test_terminal_entry_point_maps_normal_arguments_to_training_config(monkeypatch, tmp_path):
    captured = []
    source = tmp_path / "source.pth"
    output = tmp_path / "runs" / "normal"
    labeled_frames_dir = tmp_path / "labeled_frames"
    labeled_frames_dir.mkdir()
    (tmp_path / "training_data_split.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(RUN_TRAIN["trainer"], "run_training", captured.append)

    assert (
        training_main(
            [
                "--labeled_frames_dir",
                str(labeled_frames_dir),
                "--checkpoint_dir",
                str(output),
                "--finetune_checkpoint",
                str(source),
            ]
        )
        == 0
    )

    config = captured[0]
    assert config.labeled_frames_dir == labeled_frames_dir.resolve()
    assert config.checkpoint_dir == output.resolve()
    assert config.split_manifest == (tmp_path / "training_data_split.json").resolve()
    assert config.finetune_checkpoint == source
    assert config.finetune_learning_rate == pytest.approx(1e-4)
    assert config.max_epochs == 200
    assert config.batch_size == 8
    assert config.seed == 0


def test_training_config_path_owns_all_labeled_training_settings(monkeypatch, tmp_path):
    captured = []
    labeled_frames_dir = tmp_path / "labeled_frames"
    labeled_frames_dir.mkdir()
    config_path = tmp_path / "training_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "max_epochs": 115,
                "learning_rate": 0.001,
                "lr_milestones": [57, 86],
                "batch_size": 8,
                "seed": 0,
                "use_attention": True,
                "loss": "bce_dice",
                "prediction_threshold": 0.5,
                "finetune_checkpoint": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(RUN_TRAIN["trainer"], "run_training", captured.append)

    assert (
        training_main(
            [
                "--labeled_frames_dir",
                str(labeled_frames_dir),
                "--training_config_path",
                str(config_path),
            ]
        )
        == 0
    )

    config = captured[0]
    assert config.train_all_labeled_frames
    assert config.split_manifest is None
    assert config.max_epochs == 115
    assert config.lr_milestones == (57, 86)
    assert config.prediction_threshold == pytest.approx(0.5)
    assert config.loss == "bce_dice"


def test_tracked_default_all_labeled_recipe_loads(monkeypatch, tmp_path):
    captured = []
    labeled_frames_dir = tmp_path / "labeled_frames"
    labeled_frames_dir.mkdir()
    monkeypatch.setattr(RUN_TRAIN["trainer"], "run_training", captured.append)

    assert (
        training_main(
            [
                "--labeled_frames_dir",
                str(labeled_frames_dir),
                "--training_config_path",
                str(PROJECT_ROOT / "training" / "default_all_labeled_training_config.json"),
            ]
        )
        == 0
    )

    config = captured[0]
    assert config.train_all_labeled_frames
    assert config.max_epochs == 115
    assert config.lr_milestones == (25, 51, 62, 71, 82)
    assert config.scratch_learning_rate == pytest.approx(0.001)


def test_training_config_path_rejects_conflicting_tuning_arguments(tmp_path):
    labeled_frames_dir = tmp_path / "labeled_frames"
    labeled_frames_dir.mkdir()
    config_path = tmp_path / "training_config.json"
    config_path.write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(SystemExit):
        training_main(
            [
                "--labeled_frames_dir",
                str(labeled_frames_dir),
                "--training_config_path",
                str(config_path),
                "--finetune_checkpoint",
                "source.pth",
            ]
        )


def test_cv_writes_a_complete_all_labeled_training_recipe(tmp_path):
    summary = tmp_path / "cv_s0_summary.json"
    summary.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        batch_size=8,
        seed=0,
        use_attention=True,
        loss="bce_dice",
        finetune_checkpoint=None,
    )
    trainer = SimpleNamespace(
        initial_learning_rate=lambda _: 0.001,
        file_sha256=lambda _: "summary-hash",
    )

    recipe = all_labeled_training_config(
        summary,
        [
            {"metadata": {"best_epoch": 100, "prediction_threshold": 0.5}},
            {"metadata": {"best_epoch": 120, "prediction_threshold": 0.6}},
        ],
        config,
        trainer,
    )

    assert recipe["max_epochs"] == 200
    assert recipe["lr_milestones"] == [100, 150]
    assert recipe["prediction_threshold"] == pytest.approx(0.55)
    assert recipe["sampling"] == "natural"
    assert recipe["loss"] == "bce_dice"
    assert recipe["source_cv_summary"] == "cv_s0_summary.json"


def test_cv_recipe_rounds_epoch_budget_up_to_the_next_hundred(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        batch_size=8, seed=0, use_attention=True, loss="bce", finetune_checkpoint=None
    )
    trainer = SimpleNamespace(
        initial_learning_rate=lambda _: 0.001,
        file_sha256=lambda _: "summary-hash",
    )

    recipe = all_labeled_training_config(
        summary,
        [
            {"metadata": {"best_epoch": 54, "prediction_threshold": 0.5}},
            {"metadata": {"best_epoch": 67, "prediction_threshold": 0.55}},
        ],
        config,
        trainer,
    )

    assert recipe["max_epochs"] == 100
    assert recipe["lr_milestones"] == [50, 75]


def test_cv_fold_selection_requires_a_complete_run_for_a_production_recipe():
    assert selected_cv_folds(None, 4) == ([0, 1, 2, 3], True)
    assert selected_cv_folds([2, 0], 4) == ([0, 2], False)
    assert selected_cv_folds([2, 0, 3, 1], 4) == ([0, 1, 2, 3], True)

    with pytest.raises(ValueError, match="cannot repeat"):
        selected_cv_folds([0, 0], 4)
    with pytest.raises(ValueError, match="invalid"):
        selected_cv_folds([4], 4)


def test_cross_validation_uses_a_bounded_selection_budget():
    assert CV["CV_MAX_EPOCHS"] == 200
    assert CV["CV_EARLY_STOPPING_PATIENCE"] == 20
