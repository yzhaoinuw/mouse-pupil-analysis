"""Evaluate learned-center blob selection and render candidate test overlays."""

import argparse
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType

import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mouse_pupil_analysis.preprocessing import InferenceDataset  # noqa: E402
from mouse_pupil_analysis.pupil_predictions import select_central_component  # noqa: E402
from scripts.run_geometry_cv import read_json, snapshot, verify_files, write_json  # noqa: E402
from scripts.run_geometry_experiment import mask_metrics, summarize, write_rows  # noqa: E402
from training import _trainer as trainer  # noqa: E402
from training import prepare_splits  # noqa: E402
from training._geometry import GeometryUNet  # noqa: E402
from training._geometry_selection import (  # noqa: E402
    AREA_SCALE_PIXELS,
    CENTER_SCALE_PIXELS,
    select_predicted_center_component,
)


def load_model(checkpoint, digest, device):
    if trainer.file_sha256(checkpoint) != digest:
        raise ValueError(f"Checkpoint differs from its training record: {checkpoint}")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = GeometryUNet(use_attention=any(key.startswith("att.") for key in weights))
    model.load_state_dict(weights)
    return model.to(device).eval()


def predict(model, images, device):
    logits, circles = model.forward_with_geometry(images.to(device))
    return logits.sigmoid()[:, 0].cpu().numpy(), circles[:, :2].cpu().numpy()


@torch.inference_mode()
def validate(cv, threshold, device):
    manifest_path = Path(cv["split_manifest"])
    if trainer.file_sha256(manifest_path) != cv["manifest_sha256"]:
        raise ValueError("The split manifest differs from the CV run.")
    verify_files(ROOT, cv["source_sha256"])
    config = trainer.TrainingConfig(split_manifest=manifest_path, fold=cv["reused_fold"])
    saved = read_json(Path(cv["reused_checkpoint"]).parents[1] / "split.json")
    if snapshot(config) != saved:
        raise ValueError("The labeled pairs differ from the CV run.")
    manifest = prepare_splits.load_manifest(manifest_path)
    sessions = prepare_splits.session_of_path(manifest, config.data_root)
    rows = defaultdict(list)
    for fold in cv["per_fold"]:
        fold_config = trainer.TrainingConfig(split_manifest=manifest_path, fold=fold["fold"])
        _, validation = trainer.make_split_datasets(fold_config)
        model = load_model(Path(fold["checkpoint"]), fold["checkpoint_sha256"], device)
        paths = iter(validation.image_paths)
        for images, masks in DataLoader(validation, batch_size=8):
            probabilities, centers = predict(model, images, device)
            for prob, center, target in zip(probabilities, centers, masks[:, 0].numpy() > 0.5):
                path = Path(next(paths))
                session = sessions[path.resolve()]
                key = f"{session}/{path.name}"
                raw = prob > threshold
                options = {
                    "raw": raw,
                    "image_center": select_central_component(prob, raw),
                    "predicted_center": select_predicted_center_component(prob, raw, tuple(center)),
                }
                for name, selected in options.items():
                    rows[name].append(
                        {
                            "variant": name,
                            "key": key,
                            "session": session,
                            **mask_metrics(selected, target),
                        }
                    )
    counts = {len(items) for items in rows.values()}
    if counts != {sum(s["n_images"] for s in manifest["sessions"] if s["fold"] in cv["folds"])}:
        raise ValueError("Incomplete validation coverage.")
    return rows, {name: summarize(items) for name, items in rows.items()}


def release_renderer(raw_run):
    """Load the saved release's rendering functions without changing the checkout."""
    commit = raw_run["renderer_commit"]
    ref = "v0.3.0^{commit}"
    if subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip() != commit:
        raise ValueError("The released renderer differs from the raw overlay run.")
    source = subprocess.check_output(
        ["git", "show", f"{commit}:mouse_pupil_analysis/pupil_predictions.py"], cwd=ROOT
    )
    module = ModuleType("_geometry_selector_released_overlay")
    sys.modules[module.__name__] = module
    exec(compile(source, "<v0.3.0 pupil_predictions>", "exec"), module.__dict__)
    return module


@torch.inference_mode()
def render(model, device, source, output, raw_overlays, raw_run, released):
    threshold = raw_run["prediction_threshold"]
    groups = defaultdict(list)
    for path in sorted(source.rglob("*.png")):
        groups[path.parent].append(path)
    expected = {p.relative_to(source).as_posix() for group in groups.values() for p in group}
    if expected != {p.relative_to(raw_overlays).as_posix() for p in raw_overlays.rglob("*.png")}:
        raise ValueError("The test inputs differ from the preceding raw overlay run.")
    rows = []
    for directory, paths in groups.items():
        frames = released.frames_from_image_directory(directory)
        loader = DataLoader(InferenceDataset([frame.image_path for frame in frames]), batch_size=32)
        maps, removed = {}, {}
        for images, names in loader:
            probabilities, centers = predict(model, images, device)
            for name, prob, center in zip(names, probabilities, centers, strict=True):
                raw = prob > threshold
                selected = select_predicted_center_component(prob, raw, tuple(center))
                maps[name] = released._encode_thresholded_confidence(prob, threshold, selected)
                removed[name] = int(raw.sum() - selected.sum())
        released._save_mask_overlays(
            frames, maps, output / directory.relative_to(source), raw_run["overlay_alpha"]
        )
        for path in paths:
            relative = path.relative_to(source)
            rendered = output / relative
            with Image.open(rendered) as image:
                if image.size != tuple(raw_run["overlay_size"]) or image.mode != "RGB":
                    raise ValueError(f"Unexpected overlay format: {rendered}")
                image.verify()
            changed = rendered.read_bytes() != (raw_overlays / relative).read_bytes()
            rows.append(
                dict(
                    source=relative.as_posix(),
                    pixels_removed=removed[path.name],
                    overlay_changed=changed,
                )
            )
        print(
            f"{len(rows):,}/{len(expected):,} verified overlays after {directory.name}", flush=True
        )
    if {p.relative_to(output).as_posix() for p in output.rglob("*.png")} != expected:
        raise ValueError("Unexpected overlay inventory.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    candidate = ROOT / "checkpoints_exp/geometry_release_candidate_2026-11-06"
    cv = read_json(ROOT / "checkpoints_exp/geometry_release_cv_2026-11-06/summary.json")
    raw = ROOT / "images_test_prediction/geometry_release_2026-11-06"
    raw_run = read_json(raw / "run.json")
    metadata = read_json(candidate / "all_data.json")
    threshold = raw_run["prediction_threshold"]
    if (metadata["checkpoint_sha256"], metadata["prediction_threshold"]) != (
        raw_run["checkpoint_sha256"],
        threshold,
    ):
        raise ValueError("The candidate differs from the raw overlay run.")
    torch.set_num_threads(4)
    device = trainer.resolve_device()
    released = release_renderer(raw_run)
    model = load_model(
        candidate / "all_data_geometry.pth", metadata["geometry_checkpoint_sha256"], device
    )
    validation_rows, metrics = validate(cv, threshold, device)
    output.mkdir(parents=True)
    write_rows(
        output / "validation.csv", [row for items in validation_rows.values() for row in items]
    )
    write_json(output / "validation.json", metrics)
    print(
        "CV mean per-session IoU: "
        + ", ".join(
            f"{name}={item['equal_session_mean_iou']:.4f}" for name, item in metrics.items()
        ),
        flush=True,
    )
    rows = render(model, device, ROOT / "images_test", output, raw, raw_run, released)
    write_rows(output / "predictions.csv", rows)
    totals = Counter((row["pixels_removed"] > 0, row["overlay_changed"]) for row in rows)
    stats = {
        **raw_run,
        "geometry_checkpoint_sha256": metadata["geometry_checkpoint_sha256"],
        "center_scale_pixels": CENTER_SCALE_PIXELS,
        "area_scale_pixels": AREA_SCALE_PIXELS,
        "masks_changed": sum(row["pixels_removed"] > 0 for row in rows),
        "overlays_changed": sum(row["overlay_changed"] for row in rows),
        "overlay_changes_without_removal": totals[False, True],
        "pixels_removed": sum(row["pixels_removed"] for row in rows),
    }
    write_json(output / "run.json", stats)
    print(f"Selected one whole blob on {stats['masks_changed']:,} of {len(rows):,} PNGs.")


if __name__ == "__main__":
    main()
