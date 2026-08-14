# -*- coding: utf-8 -*-
"""Train the same configuration N times per arm, varying only the random seed.

A checkpoint that scores higher than its predecessor has only demonstrated an
improvement if the margin exceeds what changing the seed alone produces. This
measures that floor for both fresh training and fine-tuning from a checkpoint.

    python reports/scripts/noise_floor.py --out checkpoints_exp/noise --seeds 5

Summarise the resulting folders with ``reports/scripts/summarize_runs.py``.
"""

from __future__ import annotations

import argparse
import runpy
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_trainer() -> dict:
    """Load run_train.py without importing it as a package module."""
    return runpy.run_path(str(PROJECT_ROOT / "training" / "run_train.py"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out", type=Path, required=True, help="Directory for run folders.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tag", default="noise", help="Run-name prefix.")
    parser.add_argument(
        "--finetune-checkpoint",
        type=Path,
        help="Weights for the fine-tune arm (default: the packaged checkpoint).",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["scratch", "finetune"],
        choices=["scratch", "finetune"],
    )
    args = parser.parse_args(argv)

    trainer = _load_trainer()
    source = args.finetune_checkpoint
    if source is None and "finetune" in args.arms:
        from mouse_pupil_analysis.pupil_predictions import find_default_checkpoint

        source = find_default_checkpoint()

    started = time.perf_counter()
    for arm in args.arms:
        for seed in range(args.seeds):
            name = f"{args.tag}_{arm}_s{seed}"
            print(f"\n===== {name} =====", flush=True)
            trainer["run_training"](
                trainer["TrainingConfig"](
                    data_root=args.data_root,
                    checkpoint_dir=args.out,
                    run_name=name,
                    finetune_checkpoint=source if arm == "finetune" else None,
                    # Natural sampling mirrors how the packaged checkpoint was trained.
                    balance_training_sizes=False,
                    n_epochs=args.epochs,
                    seed=seed,
                    device=args.device,
                    console_interval=50,
                )
            )

    print(f"\nDone in {(time.perf_counter() - started) / 60:.0f} min. Summarise with:")
    print(f"  python reports/scripts/summarize_runs.py --runs {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
