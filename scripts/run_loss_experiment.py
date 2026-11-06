"""Screen BCE versus BCE + soft Dice on an unchanged session-held-out fold.

Run from a source checkout using the pupil_tracking environment. Artifacts stay
under checkpoints_exp; no checkpoint is promoted into the installed package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mouse_pupil_analysis.pupil_predictions import load_unet_checkpoint  # noqa: E402
from mouse_pupil_analysis.unet import UNet  # noqa: E402
from training import _trainer as trainer  # noqa: E402
from training import prepare_splits  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def snapshot(config: trainer.TrainingConfig) -> dict:
    """Fingerprint exact source pairs, including excluded pairs, before training."""
    manifest = prepare_splits.load_manifest(config.split_manifest)
    prepare_splits.fold_paths(manifest, config.fold, config.data_root)
    records = []
    hashes = defaultdict(set)
    sessions = defaultdict(set)
    for entry in sorted(manifest["images"], key=lambda item: item["key"]):
        role = (
            "excluded"
            if entry.get("holdout") or entry.get("validation_holdout")
            else "validation" if entry["fold"] == config.fold else "train"
        )
        image = config.data_root / entry["image"]
        mask = config.data_root / entry["mask"]
        with Image.open(image) as source, Image.open(mask) as target:
            if source.size != target.size:
                raise ValueError(f"Image/mask dimensions differ: {entry['key']}")
            # Labelme palette masks can decode to foreground gray 38. Match the
            # existing loader's >0 rule rather than requiring a 255 encoding.
            mask_values = [int(value) for value in np.unique(np.asarray(target.convert("L")))]
        image_hash = trainer.file_sha256(image)
        hashes[role].add(image_hash)
        session = entry["key"].rsplit("/", 1)[0]
        sessions[role].add(session)
        records.append(
            {
                "key": entry["key"],
                "role": role,
                "image": entry["image"],
                "mask": entry["mask"],
                "image_sha256": image_hash,
                "mask_sha256": trainer.file_sha256(mask),
                "mask_gray_values": mask_values,
                "empty_target": not any(mask_values),
            }
        )
    if hashes["train"] & hashes["validation"] or sessions["train"] & sessions["validation"]:
        raise ValueError("Train/validation image or session overlap.")
    return {
        "manifest_sha256": trainer.file_sha256(config.split_manifest),
        "fold": config.fold,
        "pairs": records,
        "counts": {role: sum(r["role"] == role for r in records) for role in sessions},
        "sessions": {role: sorted(names) for role, names in sessions.items()},
        "empty_targets": {
            role: sum(r["role"] == role and r["empty_target"] for r in records) for role in sessions
        },
    }


def evaluate(checkpoint: Path, config: trainer.TrainingConfig, output: Path) -> tuple[dict, list]:
    """Score raw masks at fixed 0.5 and the selected checkpoint's calibrated threshold."""
    _, dataset = trainer.make_split_datasets(config)
    model = load_unet_checkpoint(checkpoint, trainer.resolve_device(config.device))
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    _, probabilities, targets = trainer._collect_validation(
        model,
        DataLoader(dataset, batch_size=config.batch_size, shuffle=False),
        trainer.make_criterion(config),
        trainer.resolve_device(config.device),
    )
    sessions = prepare_splits.session_of_path(
        prepare_splits.load_manifest(config.split_manifest), config.data_root
    )
    rows = []
    scores = {}
    for name, threshold in (("fixed_0.5", 0.5), ("calibrated", metadata["prediction_threshold"])):
        predicted = probabilities > threshold
        iou, dice = trainer.per_image_overlap_scores(probabilities, targets, threshold)
        grouped = defaultdict(list)
        empty_rows = []
        for index, image_path in enumerate(dataset.image_paths):
            truth = targets[index] > 0.5
            prediction = predicted[index]
            target_area = int(truth.sum())
            predicted_area = int(prediction.sum())
            false_foreground = int((prediction & ~truth).sum())
            session = sessions[Path(image_path).resolve()]
            row = {
                "key": f"{session}/{Path(image_path).name}",
                "session": session,
                "threshold_mode": name,
                "threshold": threshold,
                "iou": float(iou[index]),
                "dice": float(dice[index]),
                "target_pixels": target_area,
                "predicted_pixels": predicted_area,
                "false_foreground_pixels": false_foreground,
                "missed_foreground_pixels": int((truth & ~prediction).sum()),
                "area_ratio": predicted_area / target_area if target_area else None,
            }
            rows.append(row)
            grouped[session].append(row)
            if target_area == 0:
                empty_rows.append(row)
            if name == "fixed_0.5":
                mask_dir = output / "raw_masks" / session
                mask_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(prediction[0].numpy().astype(np.uint8) * 255).save(
                    mask_dir / Path(image_path).name
                )
        scores[name] = {
            "threshold": threshold,
            "image_mean_iou": float(iou.mean()),
            "image_mean_dice": float(dice.mean()),
            "session_mean_iou": statistics.fmean(
                statistics.fmean(r["iou"] for r in group) for group in grouped.values()
            ),
            "per_session": {
                session: {
                    "images": len(group),
                    "iou": statistics.fmean(r["iou"] for r in group),
                    "mean_area_ratio_nonempty": (
                        statistics.fmean(r["area_ratio"] for r in group if r["target_pixels"])
                        if any(r["target_pixels"] for r in group)
                        else None
                    ),
                }
                for session, group in sorted(grouped.items())
            },
            "empty_targets": {
                "images": len(empty_rows),
                "images_with_false_foreground": sum(r["predicted_pixels"] > 0 for r in empty_rows),
                "total_false_foreground_pixels": sum(r["predicted_pixels"] for r in empty_rows),
            },
        }
    with (output / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "checkpoint_sha256": trainer.file_sha256(checkpoint),
        "best_epoch": metadata["best_epoch"],
        "scores": scores,
    }
    write_json(output / "evaluation.json", result)
    return result, [r for r in rows if r["threshold_mode"] == "fixed_0.5"]


def render_review(
    output: Path,
    config: trainer.TrainingConfig,
    rows: dict[str, list],
    *,
    candidate: str = "bce_dice",
    candidate_label: str = "BCE + soft Dice",
    extra_prediction_labels: dict[str, str] | None = None,
    mask_note: str = "Raw masks at fixed threshold 0.5; no component cleanup.",
) -> None:
    """Create labeled side-by-side overlays for every validation image."""
    _, dataset = trainer.make_split_datasets(config)
    review = output / "review"
    review.mkdir()
    panels = []
    prediction_labels = {
        "bce": "BCE",
        **(extra_prediction_labels or {}),
        candidate: candidate_label,
    }
    width = 296 * (1 + len(prediction_labels))
    row_maps = {loss: {r["key"]: r for r in values} for loss, values in rows.items()}
    for index, (image, target) in enumerate(dataset):
        path = Path(dataset.image_paths[index])
        session = path.parent.parent.name
        key = f"{session}/{path.name}"
        gray = (image[0].numpy() * 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        canvas = Image.new("RGB", (width, 348), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), key, fill="black")
        for column, loss in enumerate(("target", *prediction_labels)):
            if loss == "target":
                mask = target[0].numpy() > 0.5
                label = "Ground truth (cyan)"
                color = np.array([0, 255, 255])
            else:
                with Image.open(output / loss / "raw_masks" / session / path.name) as raw:
                    mask = np.asarray(raw) > 0
                model_label = prediction_labels[loss]
                label = f"{model_label}: IoU {row_maps[loss][key]['iou']:.3f} at 0.5"
                color = np.array([255, 80, 50])
            overlay = rgb.copy()
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)
            canvas.paste(Image.fromarray(overlay).resize((296, 296)), (column * 296, 48))
            draw.text((column * 296 + 8, 28), label, fill="black")
        filename = f"{index:03d}.png"
        canvas.save(review / filename)
        panels.append((key, filename, canvas))
    chosen = {"260807_3582_Purple_trial3/frame_00007.png"}
    for session in {row["session"] for row in rows["bce"]}:
        worst = sorted((r for r in rows["bce"] if r["session"] == session), key=lambda r: r["iou"])
        chosen.update(r["key"] for r in worst[:2])
    chosen.update(r["key"] for r in rows["bce"] if not r["target_pixels"])
    selected = [panel for panel in panels if panel[0] in chosen]
    sheet = Image.new("RGB", (width, 348 * len(selected)), "white")
    for index, (_, _, panel) in enumerate(selected):
        sheet.paste(panel, (0, 348 * index))
    sheet.save(review / "contact_sheet.png")
    write_json(review / "contact_sheet_selection.json", [p[0] for p in selected])
    (review / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>BCE versus {html.escape(candidate_label)}</title>'
        f"<style>body{{font:16px sans-serif;max-width:{max(1100, width)}px;margin:auto}}"
        "img{max-width:100%}</style>"
        f"<h1>BCE versus {html.escape(candidate_label)}</h1><p>Fold-held-out validation; selection data, "
        f"not an independent test. {html.escape(mask_note)} "
        f"Left to right: {html.escape(', '.join(('ground truth', *prediction_labels.values())))}.</p>"
        + "".join(
            f'<figure><img loading="lazy" src="{name}"><figcaption>{html.escape(key)}</figcaption></figure>'
            for key, name, _ in panels
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints_exp/loss_screen_2026-11-03_fold2",
    )
    parser.add_argument("--fold", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    config = trainer.TrainingConfig(
        split_manifest=PROJECT_ROOT / "training_data_split.json",
        fold=args.fold,
        seed=args.seed,
        max_epochs=args.max_epochs,
        early_stopping_patience=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        console_interval=5,
    )
    before = snapshot(config)
    output.mkdir(parents=True)
    write_json(output / "split.json", before)
    torch.manual_seed(config.seed)
    initial = UNet(use_attention=config.use_attention)
    initial_hash = hashlib.sha256(
        b"".join(value.numpy().tobytes() for value in initial.state_dict().values())
    ).hexdigest()
    package_hashes = {
        p.name: trainer.file_sha256(p)
        for p in sorted((PROJECT_ROOT / "mouse_pupil_analysis/checkpoints").glob("*"))
        if p.is_file()
    }
    write_json(
        output / "experiment.json",
        {
            "purpose": "Single-fold, single-seed loss screen; no independent test claim",
            "losses": ["bce", "bce_dice"],
            "dice_definition": "BCE + mean soft Dice loss on nonempty images, equal weights 1; empty targets BCE only",
            "common_config": trainer._jsonable_config(config),
            "initial_state_sha256": initial_hash,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "deterministic_algorithms": True,
            "tf32": False,
            "package_checkpoint_hashes": package_hashes,
            "source_sha256": {
                str(p.relative_to(PROJECT_ROOT)): trainer.file_sha256(p)
                for p in (
                    Path(__file__),
                    PROJECT_ROOT / "training/_trainer.py",
                    PROJECT_ROOT / "training/prepare_splits.py",
                    PROJECT_ROOT / "mouse_pupil_analysis/augmentation.py",
                    PROJECT_ROOT / "mouse_pupil_analysis/preprocessing.py",
                    PROJECT_ROOT / "mouse_pupil_analysis/unet.py",
                )
            },
        },
    )
    print(f"Split counts: {before['counts']}; initial state: {initial_hash}", flush=True)
    results, rows = {}, {}
    started = time.perf_counter()
    for loss in ("bce", "bce_dice"):
        if snapshot(config) != before:
            raise RuntimeError("Source pool or manifest changed during experiment.")
        run_config = replace(config, loss=loss, checkpoint_dir=output / loss)
        print(f"Starting {loss}", flush=True)
        checkpoint = trainer.run_training(run_config)
        results[loss], rows[loss] = evaluate(checkpoint, run_config, output / loss)
        write_json(
            output / "summary.json",
            {"results": results, "elapsed_seconds": time.perf_counter() - started},
        )
    if snapshot(config) != before:
        raise RuntimeError("Source pool or manifest changed during experiment.")
    for name, digest in package_hashes.items():
        if trainer.file_sha256(PROJECT_ROOT / "mouse_pupil_analysis/checkpoints" / name) != digest:
            raise RuntimeError("Packaged checkpoint changed during experiment.")
    render_review(output, config, rows)
    print(json.dumps(results, indent=2), flush=True)
    print(f"Review: {output / 'review/index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
