"""Train the geometry head and compare raw and gated masks on a matched fold."""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import fmean

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mouse_pupil_analysis.pupil_predictions import (  # noqa: E402
    load_unet_checkpoint,
    select_central_component,
)
from mouse_pupil_analysis.unet import UNet  # noqa: E402
from scripts.run_augmentation_experiment import BASELINE, read_json  # noqa: E402
from scripts.run_loss_experiment import render_review, snapshot, write_json  # noqa: E402
from training import _trainer as trainer  # noqa: E402
from training import prepare_splits  # noqa: E402
from training._geometry import (  # noqa: E402
    GATE_FADE_PIXELS,
    GATE_MARGIN_PIXELS,
    GEOMETRY_LOSS_WEIGHT,
    SMOOTH_L1_BETA_PIXELS,
    GeometryUNet,
    circle_targets,
    gate_probabilities,
)


def check_initialization(config, expected_hash):
    torch.manual_seed(config.seed)
    reference = UNet(use_attention=config.use_attention).state_dict()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(config.seed)
    candidate = GeometryUNet(use_attention=config.use_attention).state_dict()
    shared = hashlib.sha256(b"".join(candidate[key].numpy().tobytes() for key in reference))
    if shared.hexdigest() != expected_hash or not torch.equal(torch.get_rng_state(), expected_rng):
        raise ValueError("Shared initial weights or the RNG stream differ from the baseline.")
    return hashlib.sha256(
        b"".join(value.numpy().tobytes() for value in candidate.values())
    ).hexdigest()


def mask_metrics(predicted, target):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        predicted.astype(np.uint8), connectivity=8
    )
    touching = set(labels[target])
    detached = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count) if i not in touching)
    union = int((predicted | target).sum())
    return {
        "iou": int((predicted & target).sum()) / union if union else 1.0,
        "target_pixels": int(target.sum()),
        "false_foreground_pixels": int((predicted & ~target).sum()),
        "missed_foreground_pixels": int((target & ~predicted).sum()),
        "detached_pixels": detached,
        "components": count - 1,
    }


def summarize(rows):
    by_session = defaultdict(list)
    for row in rows:
        by_session[row["session"]].append(row["iou"])
    session_iou = {session: fmean(scores) for session, scores in sorted(by_session.items())}
    positive = [row for row in rows if row["target_pixels"]]
    negative = [row for row in rows if not row["target_pixels"]]
    return {
        "image_mean_iou": fmean(row["iou"] for row in rows),
        "equal_session_mean_iou": fmean(session_iou.values()),
        "session_iou": session_iou,
        "positive_frames": len(positive),
        "positive_pixel_totals": {
            key: sum(row[key] for row in positive)
            for key in ("false_foreground_pixels", "missed_foreground_pixels", "detached_pixels")
        },
        "positive_frames_with_multiple_components": sum(row["components"] > 1 for row in positive),
        "empty_frames": len(negative),
        "empty_frames_with_false_foreground": sum(row["components"] > 0 for row in negative),
        "empty_false_foreground_pixels": sum(row["false_foreground_pixels"] for row in negative),
    }


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(config, baseline, checkpoint, output):
    device = trainer.resolve_device(config.device)
    base_model = load_unet_checkpoint(baseline / "best.pth", device)
    model = GeometryUNet(use_attention=config.use_attention)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    _, dataset = trainer.make_split_datasets(config)
    sessions = prepare_splits.session_of_path(
        prepare_splits.load_manifest(config.split_manifest), config.data_root
    )
    results = defaultdict(list)
    circle_rows = []
    index = 0
    for images, masks in DataLoader(dataset, batch_size=config.batch_size):
        images = images.to(device)
        baseline_prob = base_model(images).sigmoid()[:, 0].cpu().numpy()
        logits, circles = model.forward_with_geometry(images)
        prob = logits.sigmoid()
        gated = (gate_probabilities(prob, circles)[:, 0] > 0.5).cpu().numpy()
        raw = (prob[:, 0] > 0.5).cpu().numpy()
        predicted_circles = circles.cpu().numpy()
        targets, present = circle_targets(masks)
        for offset, target in enumerate(masks[:, 0].numpy() > 0.5):
            path = Path(dataset.image_paths[index])
            session = sessions[path.resolve()]
            key = f"{session}/{path.name}"
            index += 1
            base_mask = baseline_prob[offset] > 0.5
            with Image.open(baseline / "raw_masks" / key) as saved:
                if not np.array_equal(base_mask, np.asarray(saved) > 0):
                    raise RuntimeError(f"Baseline replay differs for {key}.")
            if (gated[offset] & ~raw[offset]).any():
                raise RuntimeError(f"The gate added foreground for {key}.")
            variants = {
                "bce": base_mask,
                "bce_selector": select_central_component(baseline_prob[offset], base_mask),
                "geometry_raw": raw[offset],
                "geometry_gated": gated[offset],
            }
            for variant, prediction in variants.items():
                results[variant].append(
                    {"key": key, "session": session, **mask_metrics(prediction, target)}
                )
                mask_path = output / variant / "raw_masks" / key
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(prediction.astype(np.uint8) * 255).save(mask_path)
            predicted = predicted_circles[offset]
            truth = targets[offset].numpy()
            circle_rows.append(
                {
                    "key": key,
                    "session": session,
                    "x": float(predicted[0]),
                    "y": float(predicted[1]),
                    "radius": float(predicted[2]),
                    "target_x": float(truth[0]) if present[offset] else None,
                    "target_y": float(truth[1]) if present[offset] else None,
                    "target_radius": float(truth[2]) if present[offset] else None,
                    "center_error": (
                        float(np.linalg.norm(predicted[:2] - truth[:2]))
                        if present[offset]
                        else None
                    ),
                    "radius_error": float(predicted[2] - truth[2]) if present[offset] else None,
                }
            )
    for variant, rows in results.items():
        write_rows(output / variant / "per_image.csv", rows)
    write_rows(output / "circles.csv", circle_rows)
    positive = [row for row in circle_rows if row["target_radius"] is not None]
    return results, {
        "center_mean_error_pixels": fmean(row["center_error"] for row in positive),
        "radius_mean_absolute_error_pixels": fmean(abs(row["radius_error"]) for row in positive),
        "radius_mean_signed_error_pixels": fmean(row["radius_error"] for row in positive),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline, output = args.baseline_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)

    reference = read_json(baseline / "experiment.json")
    reference_config = reference["common_config"]
    config = trainer.TrainingConfig(
        split_manifest=ROOT / "training_data_split.json",
        fold=reference_config["fold"],
        seed=reference_config["seed"],
        max_epochs=reference_config["max_epochs"],
        early_stopping_patience=reference_config["early_stopping_patience"],
        device=reference_config["device"],
        console_interval=reference_config["console_interval"],
    )
    config_json = trainer._jsonable_config(config)
    config_json.pop("edge_obstruction_probability")
    config_json.pop("rotation_override")
    if json.loads(json.dumps(config_json)) != reference_config:
        raise ValueError("The common training recipe differs from the saved baseline.")
    gpu = torch.cuda.get_device_name() if config.device == "cuda" else None
    if (torch.__version__, np.__version__, gpu) != (
        reference["torch_version"],
        reference["numpy_version"],
        reference["gpu"],
    ):
        raise ValueError("The training environment differs from the saved baseline.")
    initial_hash = check_initialization(config, reference["initial_state_sha256"])
    split = snapshot(config)
    if split != read_json(baseline / "split.json"):
        raise ValueError("The source pairs or fold assignments differ from the baseline.")
    baseline = baseline / "bce"
    baseline_hash = trainer.file_sha256(baseline / "best.pth")
    if baseline_hash != read_json(baseline / "evaluation.json")["checkpoint_sha256"]:
        raise ValueError("The saved BCE checkpoint differs from its recorded hash.")

    sources = {
        str(path): trainer.file_sha256(ROOT / path)
        for path in (
            *reference["source_sha256"],
            "training/_geometry.py",
            "mouse_pupil_analysis/pupil_predictions.py",
            "scripts/run_augmentation_experiment.py",
            str(Path(__file__).resolve().relative_to(ROOT)),
        )
    }
    for filename, digest in reference["source_sha256"].items():
        if (
            Path(filename).name in {"preprocessing.py", "prepare_splits.py"}
            and sources[filename] != digest
        ):
            raise ValueError(f"Preprocessing or fold source differs from baseline: {filename}")
    package = ROOT / "mouse_pupil_analysis/checkpoints"
    package_hashes = reference["package_checkpoint_hashes"]
    if any(trainer.file_sha256(package / name) != value for name, value in package_hashes.items()):
        raise ValueError("A packaged checkpoint file differs from the baseline's record.")
    output.mkdir(parents=True)
    config = replace(config, checkpoint_dir=output / "candidate")
    write_json(output / "split.json", split)
    write_json(
        output / "experiment.json",
        {
            "config": trainer._jsonable_config(config),
            "baseline": str(baseline),
            "baseline_checkpoint_sha256": baseline_hash,
            "shared_initial_state_sha256": reference["initial_state_sha256"],
            "candidate_initial_state_sha256": initial_hash,
            "source_sha256": sources,
            "package_checkpoint_hashes": package_hashes,
            "geometry_loss": "Smooth L1 on normalized x, y, and radius for nonempty masks",
            "geometry_loss_weight": GEOMETRY_LOSS_WEIGHT,
            "smooth_l1_beta_pixels": SMOOTH_L1_BETA_PIXELS,
            "gate_margin_pixels": GATE_MARGIN_PIXELS,
            "gate_fade_pixels": GATE_FADE_PIXELS,
            "evaluation_threshold": 0.5,
        },
    )
    print(f"Matched baseline and RNG confirmed. Split counts: {split['counts']}")
    started = time.perf_counter()
    checkpoint = trainer.run_training(config, geometry_head=True)
    if snapshot(config) != split:
        raise RuntimeError("The source pairs or split changed during training.")
    if any(trainer.file_sha256(ROOT / name) != value for name, value in sources.items()):
        raise RuntimeError("An experiment source file changed during training.")
    if any(trainer.file_sha256(package / name) != value for name, value in package_hashes.items()):
        raise RuntimeError("A packaged checkpoint file changed during training.")
    rows, geometry = evaluate(config, baseline, checkpoint, output)
    result = {
        "masks": {variant: summarize(values) for variant, values in rows.items()},
        "geometry": geometry,
        "candidate_best_epoch": read_json(checkpoint.with_suffix(".json"))["best_epoch"],
        "candidate_checkpoint_sha256": trainer.file_sha256(checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", result)
    render_review(
        output,
        config,
        rows,
        candidate="geometry_gated",
        candidate_label="Geometry-gated BCE",
        extra_prediction_labels={
            "bce_selector": "BCE + selector",
            "geometry_raw": "Two heads, raw",
        },
        mask_note="Every output uses the common threshold 0.5.",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
