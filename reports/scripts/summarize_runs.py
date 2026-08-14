# -*- coding: utf-8 -*-
"""Summarise a directory of training runs into the report's statistics table.

Reads every ``<run-dir>/best.json`` written by ``training/run_train.py`` and
reports per-arm mean, standard deviation, and range. This is the table in
``reports/2026-08-14-checkpoint-noise-floor.md``.

    python reports/scripts/summarize_runs.py --runs checkpoints_exp --markdown

Arms are inferred from ``training_mode`` in each run's metadata, so a directory
holding both fresh-training and fine-tuning runs is grouped automatically.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from itertools import product
from pathlib import Path

METRICS = ("balanced_iou", "macro_iou", "macro_dice")


def load_runs(runs_dir: Path) -> list[dict]:
    """Load every complete run folder under ``runs_dir``."""
    runs = []
    for candidate in sorted(runs_dir.iterdir()):
        metadata_path = candidate / "best.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["run_name"] = metadata.get("run_name", candidate.name)
        runs.append(metadata)
    if not runs:
        raise SystemExit(f"No run folders with a best.json found under {runs_dir}.")
    return runs


def summarize(runs: list[dict]) -> dict[str, dict]:
    """Group runs by training mode and reduce each metric to mean/sd/range."""
    arms: dict[str, dict] = {}
    for mode in sorted({run.get("training_mode", "unknown") for run in runs}):
        members = [run for run in runs if run.get("training_mode", "unknown") == mode]
        entry: dict = {"n": len(members)}
        for metric in METRICS:
            values = [run[metric] for run in members if metric in run]
            if not values:
                continue
            entry[metric] = {
                "mean": st.mean(values),
                # A single run has no spread; report it as such rather than crashing.
                "sd": st.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        entry["thresholds"] = sorted(run["prediction_threshold"] for run in members)
        entry["best_epochs"] = sorted(run["best_epoch"] for run in members)
        arms[mode] = entry
    return arms


def spread_context(runs: list[dict], metric: str, claimed_gain: float) -> str:
    """Express a claimed improvement in units of the observed seed spread."""
    values = [run[metric] for run in runs if metric in run]
    if len(values) < 2:
        return "not enough runs to estimate a spread"
    pooled = st.stdev(values)
    pairs = [abs(a - b) for a, b in product(values, repeat=2)]
    at_least = sum(gap >= claimed_gain for gap in pairs) / len(pairs)
    return (
        f"pooled sd {pooled:.4f}; a {claimed_gain:+.4f} gain is "
        f"{claimed_gain / pooled:.2f} sd, and {at_least:.0%} of run pairs "
        f"differ by at least that much"
    )


def _fmt(stats: dict) -> str:
    return f"{stats['mean']:.4f} ±{stats['sd']:.4f}"


def render_text(arms: dict[str, dict]) -> str:
    lines = []
    for mode, entry in arms.items():
        lines.append(f"\n=== {mode} (n={entry['n']}) ===")
        for metric in METRICS:
            if metric in entry:
                stats = entry[metric]
                lines.append(
                    f"  {metric:<13} {_fmt(stats)}   " f"[{stats['min']:.4f}, {stats['max']:.4f}]"
                )
        lines.append(f"  thresholds    {entry['thresholds']}")
        lines.append(f"  best epochs   {entry['best_epochs']}")
    return "\n".join(lines)


def render_markdown(arms: dict[str, dict]) -> str:
    lines = [
        "| Arm | Balanced IoU | Macro IoU | Best epoch | Thresholds |",
        "|---|---|---|---|---|",
    ]
    for mode, entry in arms.items():
        epochs = entry["best_epochs"]
        thresholds = entry["thresholds"]
        lines.append(
            f"| {mode} (n={entry['n']}) | {_fmt(entry['balanced_iou'])} | "
            f"{_fmt(entry['macro_iou'])} | {epochs[0]}–{epochs[-1]} | "
            f"{thresholds[0]:g}–{thresholds[-1]:g} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("checkpoints_exp"),
        help="Directory holding run folders (default: checkpoints_exp).",
    )
    parser.add_argument("--markdown", action="store_true", help="Emit a Markdown table.")
    parser.add_argument(
        "--claimed-gain",
        type=float,
        default=0.0112,
        help="An improvement to express in units of the observed seed spread.",
    )
    args = parser.parse_args(argv)

    runs = load_runs(args.runs)
    arms = summarize(runs)
    print(render_markdown(arms) if args.markdown else render_text(arms))
    print(f"\nbalanced_iou: {spread_context(runs, 'balanced_iou', args.claimed_gain)}")
    print(f"macro_iou   : {spread_context(runs, 'macro_iou', args.claimed_gain)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
