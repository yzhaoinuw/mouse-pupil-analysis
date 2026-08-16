# -*- coding: utf-8 -*-
"""Train or fine-tune the pupil-segmentation UNet.

Run this script with terminal arguments for a command-line workflow, or run it
without arguments from Spyder/an IDE to use the editable configuration block at
the bottom. Importing the module is side-effect free.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from mouse_pupil_analysis.augmentation import SegmentationDataset, paired_image_mask_paths
from mouse_pupil_analysis.pupil_predictions import load_unet_checkpoint
from mouse_pupil_analysis.unet import UNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT  # Use PROJECT_ROOT / "sample_data" for the included fixture.

SIZE_BIN_NAMES = ("tiny", "medium", "large")
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")
SELECTION_METRICS = ("balanced_iou", "macro_iou")
SCHEDULER_METRICS = ("val_loss", "balanced_iou", "macro_iou")


def _load_data_splits():
    """Load the sibling split module by path.

    ``reports/scripts`` loads this trainer with ``runpy.run_path``, which does not put
    ``training/`` on ``sys.path``, so a plain ``import data_splits`` would work as a
    script and fail from those callers.
    """
    path = Path(__file__).resolve().parent / "data_splits.py"
    spec = importlib.util.spec_from_file_location("training_data_splits", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations through sys.modules, so the module has to
    # be registered before it executes or every @dataclass in it raises.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


data_splits = _load_data_splits()


def resolve_device(preference: str = "auto") -> torch.device:
    """Select the training device, preferring CUDA, then Apple MPS, then CPU.

    Apple silicon runs this model several times faster on MPS than on CPU. Its
    kernels are not bit-identical to the CPU ones, so a run reproduced on a
    different device matches closely rather than exactly; pass an explicit
    device when a run has to be repeated precisely.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class TrainingConfig:
    """Editable settings for one fresh-training or fine-tuning run."""

    data_root: Path = DATA_ROOT
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints_exp"
    run_name: str | None = None
    finetune_checkpoint: Path | None = None
    split_manifest: Path | None = None
    fold: int | None = None
    final: bool = False
    use_attention: bool = True
    batch_size: int = 8
    scratch_learning_rate: float = 1e-3
    finetune_learning_rate: float = 1e-4
    n_epochs: int = 200
    early_stopping_patience: int = 40
    scheduler_patience: int = 8
    promotion_target_iou: float = 0.85
    min_improvement: float = 1e-4
    # Floored at 0.50 deliberately. Measured over the 24 grouped-fold checkpoints of
    # 2026-08-16: on the folds the model handles, the optimum never falls below 0.50 and
    # allowing lower gains exactly 0.0000 in 10 of 12 runs. Only the failing folds want
    # lower, where a low threshold over-predicts to scrape back IoU on a bad mask -- a
    # symptom to surface, not a calibration to adopt. A shipping calibration aimed at
    # diameter bias rather than IoU is a separate question and may legitimately go lower.
    threshold_candidates: tuple[float, ...] = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    )
    tiny_max_diameter: float = 15.0
    large_min_diameter: float = 80.0
    low_circularity_cutoff: float = 0.45
    # Natural sampling won the paired comparison of 2026-08-16 by 0.0354 mean per-session
    # IoU, at every seed and in 8 of 12 matched fold-seed cells. Equal-mass balancing did
    # not improve the tiny bin it was added to protect (2-2 across folds, largest gap
    # favouring natural) while costing large-pupil IoU 0.20 and 0.27 on two folds. The
    # packaged checkpoint already records sampling: "natural".
    balance_training_sizes: bool = False
    # ``balanced_iou`` averages the represented size bins equally. Under grouped folds a bin
    # can hold one or two images, so a single noisy image swings a third of the metric;
    # ``macro_iou`` averages over images instead and is what the first grouped sweep showed
    # to be the stabler selection signal. See reports/2026-08-16-selection-metric-repair.md.
    selection_metric: str = "balanced_iou"
    # The learning-rate plateau signal. Driving it from a size-bin IoU let one spiking epoch
    # define the high-water mark and decay the rate to its floor while the model was still
    # improving; validation loss is the quantity the gradient actually descends.
    scheduler_metric: str = "val_loss"
    # Threshold used for the per-epoch selection comparison only. ``None`` restores the old
    # behaviour of selecting on the best of ``threshold_candidates``, which makes each epoch's
    # score a maximum over 11 draws before the epoch maximum is taken on top of it. The
    # metadata written for the winning epoch is always fully calibrated either way.
    selection_threshold: float | None = 0.5
    console_interval: int = 10
    seed: int = 0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.device not in DEVICE_CHOICES:
            raise ValueError(f"device must be one of {', '.join(DEVICE_CHOICES)}.")
        if self.final and self.fold is not None:
            raise ValueError(
                "final and fold are alternatives: a fold run holds one fold out for "
                "validation, a final run holds the gate sessions out instead."
            )
        if self.final and self.split_manifest is None:
            raise ValueError("final needs split_manifest: the holdout is recorded there.")
        if not self.final and (self.split_manifest is None) != (self.fold is None):
            raise ValueError(
                "split_manifest and fold must be given together: a manifest selects the "
                "grouped split, and fold selects which group is held out."
            )
        if self.fold is not None and self.fold < 0:
            raise ValueError("fold must be nonnegative.")
        if self.batch_size <= 0 or self.n_epochs <= 0:
            raise ValueError("batch_size and n_epochs must be positive.")
        if self.scratch_learning_rate <= 0 or self.finetune_learning_rate <= 0:
            raise ValueError("Learning rates must be positive.")
        if self.early_stopping_patience <= 0 or self.scheduler_patience < 0:
            raise ValueError(
                "Early-stopping patience must be positive and scheduler patience nonnegative."
            )
        if not self.threshold_candidates or any(
            not 0 < threshold < 1 for threshold in self.threshold_candidates
        ):
            raise ValueError("threshold_candidates must contain probabilities between 0 and 1.")
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {', '.join(SELECTION_METRICS)}.")
        if self.scheduler_metric not in SCHEDULER_METRICS:
            raise ValueError(f"scheduler_metric must be one of {', '.join(SCHEDULER_METRICS)}.")
        if self.selection_threshold is not None and not 0 < self.selection_threshold < 1:
            raise ValueError("selection_threshold must be a probability between 0 and 1.")
        if self.tiny_max_diameter >= self.large_min_diameter:
            raise ValueError("tiny_max_diameter must be smaller than large_min_diameter.")
        if not 0 <= self.promotion_target_iou <= 1:
            raise ValueError("promotion_target_iou must be between 0 and 1.")
        if self.min_improvement < 0:
            raise ValueError("min_improvement cannot be negative.")
        if self.console_interval <= 0:
            raise ValueError("console_interval must be positive.")
        if self.run_name is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", self.run_name
        ):
            raise ValueError(
                "run_name may contain only letters, numbers, dots, dashes, and underscores."
            )


@dataclass(frozen=True)
class ValidationReport:
    """Macro validation results at one candidate prediction threshold."""

    threshold: float
    macro_iou: float
    macro_dice: float
    balanced_iou: float
    size_iou: dict[str, float | None]
    low_circularity_iou: float | None


def make_dataset(image_dir: Path, mask_dir: Path, augment: bool = False) -> SegmentationDataset:
    """Build a dataset from a stem-paired image and mask directory."""
    image_paths, mask_paths = paired_image_mask_paths(image_dir, mask_dir)
    return SegmentationDataset(image_paths, mask_paths, augment=augment)


def make_split_datasets(
    config: TrainingConfig,
) -> tuple[SegmentationDataset, SegmentationDataset]:
    """Return the ``(train, validation)`` datasets this run's configuration selects.

    With ``split_manifest`` set, both come from the grouped fold assignment in that
    manifest and no whole session spans the boundary. Without it, the historical fixed
    ``images_train`` / ``images_validation`` folders are used, which share recordings
    across the split and therefore measure held-out frames rather than generalisation.
    Those folders no longer exist in the maintained dataset, whose labelled pairs live
    in one flat ``labeled_frames`` / ``labeled_masks`` pool, so that path now applies only
    to a dataset still laid out the old way -- ``sample_data`` among them.

    Under ``final``, training takes every image the folds were allowed to see and
    validation is the recorded holdout, which no fold ever trained or validated on.
    """
    if config.split_manifest is None:
        if not (config.data_root / "images_train").is_dir():
            raise FileNotFoundError(
                f"No images_train/ under {config.data_root}. The labelled pool is one flat "
                "labeled_frames/ folder now, and splitting it needs the grouped manifest: "
                "pass --split-manifest splits.json --fold N. See training/data_collection.md."
            )
        return (
            make_dataset(
                config.data_root / "images_train",
                config.data_root / "masks_train",
                augment=True,
            ),
            make_dataset(
                config.data_root / "images_validation",
                config.data_root / "masks_validation",
                augment=False,
            ),
        )

    manifest = data_splits.load_manifest(config.split_manifest)
    if config.final:
        train, validation = data_splits.final_paths(manifest, config.data_root)
    else:
        train, validation = data_splits.fold_paths(manifest, config.fold, config.data_root)
    return (
        SegmentationDataset(train[0], train[1], augment=True),
        SegmentationDataset(validation[0], validation[1], augment=False),
    )


def split_description(config: TrainingConfig) -> str:
    """Return a one-line description of which split a run used."""
    if config.split_manifest is None:
        return "fixed images_train/images_validation folders (recordings shared across split)"
    manifest = data_splits.load_manifest(config.split_manifest)
    name = Path(config.split_manifest).name
    if config.final:
        gate = data_splits.holdout_sessions(manifest)
        return (
            f"{name} final run: trained on every non-holdout image, validated on the "
            f"{len(gate)} gate session(s): {', '.join(sorted(gate))}"
        )
    held_out = data_splits.fold_sessions(manifest, config.fold)
    gate = data_splits.holdout_sessions(manifest)
    suffix = f", gate sessions excluded entirely: {', '.join(sorted(gate))}" if gate else ""
    return (
        f"{name} fold {config.fold}/{manifest['n_folds']}, "
        f"holding out {len(held_out)} session(s): {', '.join(sorted(held_out))}{suffix}"
    )


def size_bin_labels(
    diameters: np.ndarray,
    tiny_max_diameter: float,
    large_min_diameter: float,
) -> np.ndarray:
    """Assign model-space target diameters to tiny, medium, or large bins."""
    diameters = np.asarray(diameters, dtype=float)
    if tiny_max_diameter >= large_min_diameter:
        raise ValueError("tiny_max_diameter must be smaller than large_min_diameter.")
    return np.select(
        [diameters <= tiny_max_diameter, diameters >= large_min_diameter],
        ["tiny", "large"],
        default="medium",
    )


def size_balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Give each represented size bin equal total sampling probability."""
    labels = np.asarray(labels, dtype=str)
    if labels.size == 0:
        raise ValueError("Cannot build sampling weights for an empty dataset.")
    counts = Counter(labels.tolist())
    weights = np.asarray([1.0 / counts[label] for label in labels], dtype=float)
    return weights / weights.mean()


def make_size_balanced_sampler(
    dataset: SegmentationDataset,
    config: TrainingConfig,
) -> tuple[WeightedRandomSampler, Counter]:
    """Create a reproducible sampler that balances mask-size bins."""
    diameters = np.asarray(dataset.mask_equivalent_diameters(), dtype=float)
    labels = size_bin_labels(
        diameters,
        config.tiny_max_diameter,
        config.large_min_diameter,
    )
    weights = size_balanced_sample_weights(labels)
    generator = torch.Generator().manual_seed(config.seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    return sampler, Counter(labels.tolist())


def per_image_overlap_scores(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-image IoU and Dice instead of a batch-area-weighted score."""
    predictions = probabilities > threshold
    targets = targets > 0.5
    dimensions = tuple(range(1, predictions.ndim))
    intersection = (predictions & targets).sum(dim=dimensions).float()
    predicted_area = predictions.sum(dim=dimensions).float()
    target_area = targets.sum(dim=dimensions).float()
    union = predicted_area + target_area - intersection
    iou = torch.where(union > 0, intersection / (union + epsilon), torch.ones_like(union))
    denominator = predicted_area + target_area
    dice = torch.where(
        denominator > 0,
        2.0 * intersection / (denominator + epsilon),
        torch.ones_like(denominator),
    )
    return iou, dice


def _target_diameters(targets: torch.Tensor) -> np.ndarray:
    dimensions = tuple(range(1, targets.ndim))
    areas = (targets > 0.5).sum(dim=dimensions).cpu().numpy().astype(float)
    return np.sqrt(4.0 * areas / math.pi)


def _low_circularity_targets(targets: torch.Tensor, cutoff: float) -> np.ndarray:
    low_circularity = []
    for target in (targets > 0.5).cpu().numpy():
        mask = np.asarray(target).squeeze().astype(np.uint8)
        area = int(mask.sum())
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(contour, closed=True) for contour in contours)
        circularity = 0.0 if area == 0 or perimeter <= 0 else 4.0 * math.pi * area / perimeter**2
        low_circularity.append(circularity < cutoff)
    return np.asarray(low_circularity, dtype=bool)


def evaluate_thresholds(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    thresholds: tuple[float, ...],
    tiny_max_diameter: float,
    large_min_diameter: float,
    low_circularity_cutoff: float,
    metric: str = "balanced_iou",
) -> ValidationReport:
    """Score every candidate threshold and return the one maximising ``metric``.

    With a single candidate this is a plain evaluation at that threshold, which is how the
    per-epoch selection comparison uses it.
    """
    if not thresholds:
        raise ValueError("At least one threshold candidate is required.")
    if metric not in SELECTION_METRICS:
        raise ValueError(f"metric must be one of {', '.join(SELECTION_METRICS)}.")
    diameters = _target_diameters(targets)
    size_labels = size_bin_labels(diameters, tiny_max_diameter, large_min_diameter)
    low_circularity = _low_circularity_targets(targets, low_circularity_cutoff)
    reports = []

    for threshold in thresholds:
        iou, dice = per_image_overlap_scores(probabilities, targets, threshold)
        iou_values = iou.cpu().numpy()
        size_iou: dict[str, float | None] = {}
        represented_bin_scores = []
        for label in SIZE_BIN_NAMES:
            selected = size_labels == label
            score = float(iou_values[selected].mean()) if selected.any() else None
            size_iou[label] = score
            if score is not None:
                represented_bin_scores.append(score)

        low_circularity_iou = (
            float(iou_values[low_circularity].mean()) if low_circularity.any() else None
        )
        reports.append(
            ValidationReport(
                threshold=float(threshold),
                macro_iou=float(iou.mean()),
                macro_dice=float(dice.mean()),
                balanced_iou=float(np.mean(represented_bin_scores)),
                size_iou=size_iou,
                low_circularity_iou=low_circularity_iou,
            )
        )

    return max(
        reports,
        key=lambda report: (
            getattr(report, metric),
            report.macro_iou,
            -abs(report.threshold - 0.7),
        ),
    )


def _metric_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _jsonable_config(config: TrainingConfig) -> dict[str, object]:
    values = asdict(config)
    for key in ("data_root", "checkpoint_dir", "finetune_checkpoint", "split_manifest"):
        value = values[key]
        values[key] = None if value is None else str(value)
    return values


def training_mode(config: TrainingConfig) -> str:
    """Return ``fine_tune`` or ``scratch`` for the configured run."""
    return "fine_tune" if config.finetune_checkpoint is not None else "scratch"


def initial_learning_rate(config: TrainingConfig) -> float:
    """Return the learning rate the run starts from, selected by training mode."""
    return (
        config.finetune_learning_rate
        if config.finetune_checkpoint is not None
        else config.scratch_learning_rate
    )


def run_header(
    config: TrainingConfig,
    training_examples: int,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Return the config plus the run facts a promoted checkpoint needs.

    ``training/promote_checkpoint.py`` reads these fields, so a packaged
    checkpoint can be rebuilt from its run folder instead of hand-assembled.
    The resolved device is recorded because MPS, CUDA, and CPU kernels do not
    produce bit-identical results.
    """
    return _jsonable_config(config) | {
        "training_examples": training_examples,
        "training_mode": training_mode(config),
        "device_used": None if device is None else device.type,
    }


def default_run_name(config: TrainingConfig) -> str:
    """Return a concise folder name that captures the important run choices."""
    run_kind = "ft" if config.finetune_checkpoint is not None else "scratch"
    sampling = "bal" if config.balance_training_sizes else "natural"
    learning_rate = initial_learning_rate(config)
    learning_rate_text = (
        f"{learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
    )
    name = f"{run_kind}_{sampling}_lr{learning_rate_text}_s{config.seed}"
    # Cross-validation runs differ only by fold, so the fold has to reach the folder name.
    if config.fold is not None:
        return f"{name}_f{config.fold}"
    # A final run must not be mistaken for a plain whole-pool run: it is the only one
    # whose validation number was measured against the gate.
    return f"{name}_final" if config.final else name


def _write_metadata(
    path: Path,
    config: TrainingConfig,
    report: ValidationReport,
    epoch: int,
    learning_rate: float,
    training_examples: int,
    split: str | None = None,
) -> None:
    payload = {
        "run_name": path.parent.name,
        "training_mode": training_mode(config),
        "training_examples": training_examples,
        "split": split,
        "prediction_threshold": report.threshold,
        "best_epoch": epoch,
        "balanced_iou": report.balanced_iou,
        "macro_iou": report.macro_iou,
        "macro_dice": report.macro_dice,
        "size_iou": report.size_iou,
        "low_circularity_iou": report.low_circularity_iou,
        "learning_rate": learning_rate,
        "meets_promotion_target": report.balanced_iou >= config.promotion_target_iou,
        "config": _jsonable_config(config),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _collect_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_images = 0
    probability_batches = []
    target_batches = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            batch_size = len(images)
            total_loss += criterion(logits, masks).item() * batch_size
            total_images += batch_size
            probability_batches.append(torch.sigmoid(logits).cpu())
            target_batches.append(masks.cpu())
    return (
        total_loss / total_images,
        torch.cat(probability_batches),
        torch.cat(target_batches),
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: TrainingConfig,
) -> tuple[float, ValidationReport]:
    """Evaluate one checkpoint with the same calibrated validation used in training."""
    _, val_dataset = make_split_datasets(config)
    device = resolve_device(config.device)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = load_unet_checkpoint(Path(checkpoint_path), device)
    val_loss, probabilities, targets = _collect_validation(
        model,
        val_loader,
        nn.BCEWithLogitsLoss(),
        device,
    )
    report = evaluate_thresholds(
        probabilities,
        targets,
        config.threshold_candidates,
        config.tiny_max_diameter,
        config.large_min_diameter,
        config.low_circularity_cutoff,
    )
    return val_loss, report


def run_training(config: TrainingConfig) -> Path:
    """Run training and return the stable path holding the best model weights."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_dataset, val_dataset = make_split_datasets(config)
    split = split_description(config)
    print(f"Split: {split}")

    training_examples = len(train_dataset)

    sampler = None
    if config.balance_training_sizes:
        sampler, size_counts = make_size_balanced_sampler(train_dataset, config)
        print(f"Training target counts by size: {dict(size_counts)}")

    device = resolve_device(config.device)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    learning_rate = initial_learning_rate(config)
    if config.finetune_checkpoint is None:
        model = UNet(use_attention=config.use_attention).to(device)
    else:
        model = load_unet_checkpoint(Path(config.finetune_checkpoint), device)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name = config.run_name or default_run_name(config)
    run_dir = config.checkpoint_dir / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Training run directory is not empty: {run_dir}. Choose a new run_name or "
            "remove the old experimental run deliberately."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pth"
    metadata_path = run_dir / "best.json"
    log_path = run_dir / "train.log"
    print(f"Run directory: {run_dir}")
    print(f"Device: {device.type}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        # Validation loss is minimised; the IoU metrics are maximised.
        mode="min" if config.scheduler_metric == "val_loss" else "max",
        factor=0.5,
        patience=config.scheduler_patience,
        min_lr=learning_rate * 0.5**5,
    )

    best_selection_score = -math.inf
    patience_counter = 0
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write(
            json.dumps(run_header(config, training_examples, device), sort_keys=True) + "\n"
        )
        for epoch in range(1, config.n_epochs + 1):
            model.train()
            total_train_loss = 0.0
            total_train_images = 0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(images)
                total_train_images += len(images)

            train_loss = total_train_loss / total_train_images
            val_loss, probabilities, targets = _collect_validation(
                model,
                val_loader,
                criterion,
                device,
            )
            report = evaluate_thresholds(
                probabilities,
                targets,
                config.threshold_candidates,
                config.tiny_max_diameter,
                config.large_min_diameter,
                config.low_circularity_cutoff,
                config.selection_metric,
            )
            # The reported threshold is calibrated over every candidate, but selecting on that
            # maximum makes each epoch's score a max over 11 draws, and the epoch maximum is
            # then taken on top of it. Compare epochs at one fixed threshold instead.
            selection_report = (
                report
                if config.selection_threshold is None
                else evaluate_thresholds(
                    probabilities,
                    targets,
                    (config.selection_threshold,),
                    config.tiny_max_diameter,
                    config.large_min_diameter,
                    config.low_circularity_cutoff,
                    config.selection_metric,
                )
            )
            selection_score = getattr(selection_report, config.selection_metric)
            scheduler.step(
                val_loss
                if config.scheduler_metric == "val_loss"
                else getattr(report, config.scheduler_metric)
            )
            current_lr = optimizer.param_groups[0]["lr"]

            log_line = (
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Macro Dice: {report.macro_dice:.4f} | "
                f"Macro IoU: {report.macro_iou:.4f} | Balanced IoU: {report.balanced_iou:.4f} | "
                f"Threshold: {report.threshold:.2f} | Tiny: {_metric_text(report.size_iou['tiny'])} | "
                f"Medium: {_metric_text(report.size_iou['medium'])} | "
                f"Large: {_metric_text(report.size_iou['large'])} | "
                f"Low-circularity: {_metric_text(report.low_circularity_iou)} | LR: {current_lr:g}"
            )
            show_epoch = epoch == 1 or epoch % config.console_interval == 0
            if show_epoch:
                print(log_line)
            log_file.write(log_line + "\n")

            improved = selection_score > best_selection_score + config.min_improvement
            if improved:
                best_selection_score = selection_score
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
                _write_metadata(
                    metadata_path,
                    config,
                    report,
                    epoch,
                    current_lr,
                    training_examples,
                    split,
                )
                if show_epoch:
                    print("Best model updated.")
            else:
                patience_counter += 1
                if show_epoch:
                    print(
                        f"Patience: {patience_counter}/{config.early_stopping_patience} "
                        f"(best {config.selection_metric} {best_selection_score:.4f})"
                    )
                if patience_counter >= config.early_stopping_patience:
                    print("Early stopping triggered; the best checkpoint remains saved.")
                    break

    # The calibrated threshold landing on the grid edge means the optimum is somewhere
    # outside it. At the low edge that is a symptom rather than a calibration: a model that
    # wants a lower threshold is under-segmenting and buying IoU by predicting more pixels
    # positive, which is what the failing folds did on 2026-08-16.
    calibrated = json.loads(metadata_path.read_text(encoding="utf-8"))["prediction_threshold"]
    low, high = min(config.threshold_candidates), max(config.threshold_candidates)
    if calibrated == low:
        print(
            f"WARNING: calibrated threshold {calibrated:.2f} is the lowest candidate, so this "
            "run wanted to go lower still. That usually means the model under-segments these "
            "validation recordings rather than that the grid is too narrow. Check the "
            "per-session scores before trusting this checkpoint."
        )
    elif calibrated == high:
        print(
            f"WARNING: calibrated threshold {calibrated:.2f} is the highest candidate, so the "
            f"calibration is censored by the grid. Widen --threshold-candidates past "
            f"{calibrated:.2f} to find the optimum."
        )

    print(f"Training log: {log_path}")
    print(f"Threshold and validation metadata: {metadata_path}")
    return checkpoint_path


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a fresh pupil UNet or fine-tune a compatible checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.cwd(),
        help="Directory holding the image and mask folders, and the root that split-manifest "
        "paths are resolved against.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Experiment output directory (default: <data-root>/checkpoints_exp).",
    )
    parser.add_argument("--run-name", help="Concise experiment folder name.")
    parser.add_argument(
        "--finetune-checkpoint",
        type=Path,
        help="Compatible .pth weights to fine-tune; omit for fresh training.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Grouped split from training/data_splits.py. Requires --fold. Without it, the "
        "fixed images_train/images_validation folders are used where they exist, which "
        "share recordings across the split.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="Fold held out for validation; every other fold trains. Requires --split-manifest.",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Release-candidate run: train on every non-holdout image and validate on the "
        "manifest's gate sessions. Requires --split-manifest, excludes --fold.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override the mode-specific default (1e-4 fine-tuning; 1e-3 fresh training).",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Training device. 'auto' prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default="balanced_iou",
        help="Validation metric that decides the best checkpoint and early stopping. "
        "'balanced_iou' weights the size bins equally, so a bin holding one or two images "
        "swings a third of it; 'macro_iou' averages over images instead.",
    )
    parser.add_argument(
        "--scheduler-metric",
        choices=SCHEDULER_METRICS,
        default="val_loss",
        help="Plateau signal for the learning-rate scheduler (default: val_loss).",
    )
    parser.add_argument(
        "--selection-threshold",
        default="0.5",
        help="Fixed threshold for the per-epoch selection comparison, or 'calibrated' to "
        "select on the best of --threshold-candidates as earlier runs did (default: 0.5). "
        "The winning epoch's metadata is fully calibrated either way.",
    )
    parser.add_argument(
        "--threshold-candidates",
        type=float,
        nargs="+",
        help="Probability grid to calibrate the reported threshold over.",
    )
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--balance-sizes",
        action="store_true",
        help="Sample equal mass from the tiny/medium/large bins. Off by default: natural "
        "sampling measured better on the grouped split and is what the packaged checkpoint "
        "uses.",
    )
    sampling.add_argument(
        "--natural-sampling",
        action="store_true",
        help="Use the natural training-set distribution. This is now the default; the flag is "
        "accepted so existing commands keep working.",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Disable spatial attention for fresh training; fine-tuning detects the architecture.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse terminal arguments and run training."""
    args = _build_cli_parser().parse_args(argv)
    data_root = args.data_root.resolve()
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else data_root / "checkpoints_exp"
    )
    learning_rate_override = {}
    if args.learning_rate is not None:
        key = (
            "finetune_learning_rate"
            if args.finetune_checkpoint is not None
            else "scratch_learning_rate"
        )
        learning_rate_override[key] = args.learning_rate

    if args.selection_threshold == "calibrated":
        selection_threshold = None
    else:
        try:
            selection_threshold = float(args.selection_threshold)
        except ValueError:
            _build_cli_parser().error(
                "--selection-threshold takes a probability or the word 'calibrated'."
            )
    threshold_override = (
        {"threshold_candidates": tuple(args.threshold_candidates)}
        if args.threshold_candidates
        else {}
    )

    run_training(
        TrainingConfig(
            data_root=data_root,
            checkpoint_dir=checkpoint_dir,
            run_name=args.run_name,
            finetune_checkpoint=args.finetune_checkpoint,
            split_manifest=(
                args.split_manifest.resolve() if args.split_manifest is not None else None
            ),
            fold=args.fold,
            final=args.final,
            use_attention=not args.no_attention,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            scheduler_patience=args.scheduler_patience,
            balance_training_sizes=args.balance_sizes,
            selection_metric=args.selection_metric,
            scheduler_metric=args.scheduler_metric,
            selection_threshold=selection_threshold,
            seed=args.seed,
            device=args.device,
            **learning_rate_override,
            **threshold_override,
        )
    )
    return 0


def _run_ide_configuration() -> None:
    """Run the editable no-argument configuration used by Spyder and IDEs."""
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

    run_training(
        TrainingConfig(
            data_root=DATA_ROOT,
            finetune_checkpoint=finetune_checkpoint,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    _run_ide_configuration()
