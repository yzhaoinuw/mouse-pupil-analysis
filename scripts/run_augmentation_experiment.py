"""Compare one training augmentation to the saved matched BCE baseline."""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mouse_pupil_analysis.unet import UNet  # noqa: E402
from scripts.run_loss_experiment import evaluate, render_review, snapshot, write_json  # noqa: E402
from training import _trainer as trainer  # noqa: E402

BASELINE = ROOT / "checkpoints_exp/loss_screen_2026-11-03_fold2"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def initial_state_hash(config):
    torch.manual_seed(config.seed)
    initial = UNet(use_attention=config.use_attention)
    return hashlib.sha256(
        b"".join(value.numpy().tobytes() for value in initial.state_dict().values())
    ).hexdigest()


def main(default_variant="rotation"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("edge", "rotation"), default=default_variant)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    baseline = args.baseline_dir.resolve()
    if args.variant == "edge":
        experiment_name = "edge_obstruction_screen_2026-11-05_fold2"
        candidate, label = "bce_edge", "BCE + edge bands"
        changes = {"edge_obstruction_probability": 0.30}
        stats_file, stats_key = "edge_obstruction_stats.json", "obstructions"
    else:
        experiment_name = "rotation_screen_2026-11-05_fold2"
        candidate, label = "bce_rotation", "BCE + stronger rotation"
        changes = {"rotation_override": (45.0, 0.50)}
        stats_file, stats_key = "rotation_stats.json", "rotation"
    output = (args.output_dir or ROOT / "checkpoints_exp" / experiment_name).resolve()
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
    base_config = trainer.TrainingConfig(
        split_manifest=ROOT / "training_data_split.json",
        fold=reference_config["fold"],
        seed=reference_config["seed"],
        max_epochs=reference_config["max_epochs"],
        early_stopping_patience=20,
        device="cuda" if torch.cuda.is_available() else "cpu",
        console_interval=5,
    )
    config_json = trainer._jsonable_config(base_config)
    config_json.pop("edge_obstruction_probability")
    config_json.pop("rotation_override")
    config_json = json.loads(json.dumps(config_json))
    if config_json != reference_config:
        raise ValueError("The common training recipe differs from the saved baseline.")
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    if (torch.__version__, np.__version__, gpu) != (
        reference["torch_version"],
        reference["numpy_version"],
        reference["gpu"],
    ):
        raise ValueError("The training environment differs from the saved baseline.")
    starting_hash = initial_state_hash(base_config)
    if starting_hash != reference["initial_state_sha256"]:
        raise ValueError("Initial weights differ from the saved baseline.")
    split = snapshot(base_config)
    if split != read_json(baseline / "split.json"):
        raise ValueError("The source pairs or fold assignments differ from the baseline.")
    baseline_checkpoint = baseline / "bce/best.pth"
    baseline_result = read_json(baseline / "bce/evaluation.json")
    if trainer.file_sha256(baseline_checkpoint) != baseline_result["checkpoint_sha256"]:
        raise ValueError("The saved BCE checkpoint differs from the baseline evaluation.")

    expected_edits = {
        Path("scripts/run_loss_experiment.py"),
        Path("training/_trainer.py"),
        Path("mouse_pupil_analysis/augmentation.py"),
    }
    sources = {}
    for filename, saved_hash in reference["source_sha256"].items():
        path = Path(filename)
        digest = trainer.file_sha256(ROOT / path)
        if digest != saved_hash and path not in expected_edits:
            raise ValueError(f"Baseline source changed: {path}")
        sources[filename] = digest
    extra_sources = [Path(__file__).resolve()]
    if args.variant == "edge":
        extra_sources.append(ROOT / "training/_edge_obstruction.py")
    for path in extra_sources:
        sources[str(path.relative_to(ROOT))] = trainer.file_sha256(path)
    package = ROOT / "mouse_pupil_analysis/checkpoints"
    packaged_hashes = reference["package_checkpoint_hashes"]
    if any(trainer.file_sha256(package / name) != value for name, value in packaged_hashes.items()):
        raise ValueError("A packaged checkpoint file differs from the baseline's record.")

    output.mkdir(parents=True)
    write_json(output / "split.json", split)
    config = replace(base_config, checkpoint_dir=output / candidate, **changes)
    write_json(
        output / "experiment.json",
        {
            "baseline_checkpoint": str(baseline_checkpoint),
            "baseline_checkpoint_sha256": baseline_result["checkpoint_sha256"],
            "config": trainer._jsonable_config(config),
            "initial_state_sha256": starting_hash,
            "source_sha256": sources,
            "package_checkpoint_hashes": packaged_hashes,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "gpu": gpu,
            "deterministic_algorithms": True,
            "tf32": False,
            "variant": args.variant,
            "evaluation": "Raw masks at common threshold 0.5, then each calibrated threshold",
        },
    )
    print(f"Matched baseline confirmed. Split counts: {split['counts']}", flush=True)
    start = time.perf_counter()
    checkpoint = trainer.run_training(config)
    if snapshot(base_config) != split:
        raise RuntimeError("The source pairs or split changed during training.")
    if any(trainer.file_sha256(ROOT / name) != value for name, value in sources.items()):
        raise RuntimeError("An experiment source file changed during training.")
    if any(trainer.file_sha256(package / name) != value for name, value in packaged_hashes.items()):
        raise RuntimeError("A packaged checkpoint file changed during training.")

    baseline_output = output / "bce"
    baseline_output.mkdir()
    base_result, base_rows = evaluate(baseline_checkpoint, base_config, baseline_output)
    if base_result != baseline_result:
        raise RuntimeError("Reloaded baseline evaluation differs from the saved baseline.")
    candidate_result, candidate_rows = evaluate(checkpoint, config, config.checkpoint_dir)
    stats = read_json(config.checkpoint_dir / stats_file)
    results = {"bce": base_result, candidate: candidate_result}
    write_json(
        output / "summary.json",
        {
            "results": results,
            stats_key: stats,
            "elapsed_seconds": time.perf_counter() - start,
        },
    )
    render_review(
        output,
        config,
        {"bce": base_rows, candidate: candidate_rows},
        candidate=candidate,
        candidate_label=label,
    )
    print(json.dumps(results, indent=2), flush=True)
    print(f"Review: {output / 'review/index.html'}", flush=True)


if __name__ == "__main__":
    main()
