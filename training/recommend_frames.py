# -*- coding: utf-8 -*-
"""Recommend which frames of a new recording to label by hand.

Takes a video or a folder of frames, scores every frame without using any label, and
copies the highest-scoring ones into a folder ready for Labelme.

    python training/recommend_frames.py --video /data/HQL091_sleep251103.avi --budget 20 \
        --checkpoint_dir checkpoints_exp/cv
    python training/recommend_frames.py --frames /data/some_frames --budget 20 \
        --checkpoint_dir checkpoints_exp/cv

From a video it writes a session folder under ``frames_to_label/``::

    frames_to_label/
        HQL091_sleep251103/
            extracted_frames/     every extracted frame
            recommended/          the recommended ones, plus selection.csv

Extraction samples at 5 FPS and never exceeds 2,000 frames, falling back to equally
spaced frames across the whole recording.

Scoring is implemented privately in ``training/_frame_scoring.py``; the evidence that it beats picking at
random is in ``reports/scripts/validate_frame_selection.py``, which measured 89% of
the achievable gap over the labelled pool. Read that module's docstring before
trusting the output: ranking is weakest on recordings whose pupils are much smaller
than anything in training.

The committee must be checkpoints that never trained on this recording. Any
cross-validation fold qualifies for genuinely new footage; do not point this at a
checkpoint trained on the same session.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "frames_to_label"

if __package__:
    from . import _frame_scoring as frame_scoring
    from . import _trainer as training_core
else:  # Direct ``python training/recommend_frames.py`` execution.
    sys.path.insert(0, str(PROJECT_ROOT))
    from training import _frame_scoring as frame_scoring
    from training import _trainer as training_core


def spread_picks(
    order: list[int],
    budget: int,
    min_gap: int,
    thumbnails: list | None = None,
    cutoff: float = 0.0,
) -> list[int]:
    """Take the top-scoring frames while keeping them distinct.

    Two separate redundancies have to be avoided. Neighbouring frames of a video are
    near-duplicates and the score that flags one flags all of them, which ``min_gap``
    handles. But a resting animal also produces near-identical frames *far apart* in
    time, and those pass any spacing rule -- so appearance is checked as well.
    """
    picked: list[int] = []

    def distinct(index: int) -> bool:
        # Reject if too close to *any* pick, not all of them.
        if any(abs(index - chosen) < min_gap for chosen in picked):
            return False
        if thumbnails is None or cutoff <= 0:
            return True
        return all(
            float(np.linalg.norm(thumbnails[index] - thumbnails[chosen])) >= cutoff
            for chosen in picked
        )

    for index in order:
        if len(picked) == budget:
            break
        if not picked or distinct(index):
            picked.append(index)
    # A short or very uniform recording can be too tight to honour both rules; rather
    # than return fewer frames than asked for, fill the remainder from the ranking.
    for index in order:
        if len(picked) == budget:
            break
        if index not in picked:
            picked.append(index)
    return picked


def resolve_output_dirs(args) -> tuple[Path, Path, str]:
    """Return (frames_dir, recommended_dir, name) for either input mode."""
    output_root = DEFAULT_OUTPUT_DIR
    if args.video is not None:
        stem = args.video.stem
        frames_dir = output_root / stem / "extracted_frames"
    else:
        frames_dir = args.frames
        stem = (
            args.frames.parent.name if args.frames.name == "extracted_frames" else args.frames.name
        )
    recommended_dir = output_root / stem / "recommended"
    return frames_dir, recommended_dir, stem


def committee_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Return the fold checkpoints directly contained in one CV-run directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise ValueError(f"No such checkpoint directory: {checkpoint_dir}")

    checkpoints = sorted(path for path in checkpoint_dir.glob("*/best.pth") if path.is_file())
    if len(checkpoints) < 2:
        raise ValueError(
            f"Need at least 2 fold checkpoints under {checkpoint_dir}, found {len(checkpoints)}. "
            "Pass the complete directory created by training/run_cross_validation.py; "
            "each fold subdirectory must contain best.pth."
        )
    return checkpoints


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Video to extract frames from.")
    source.add_argument("--frames", type=Path, help="Directory of already-extracted PNGs.")
    parser.add_argument("--budget", type=int, default=20, help="How many frames to recommend.")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        required=True,
        metavar="DIR",
        help=(
            "Complete cross-validation run directory. Its immediate fold subdirectories "
            "must each contain best.pth."
        ),
    )
    args = parser.parse_args(argv)

    trainer = training_core
    selection = frame_scoring
    from mouse_pupil_analysis.extract_frames import extract_selected_frames
    from mouse_pupil_analysis.preprocessing import InferenceDataset
    from mouse_pupil_analysis.pupil_predictions import (
        frames_from_image_directory,
        load_unet_checkpoint,
    )

    try:
        checkpoints = committee_checkpoints(args.checkpoint_dir)
    except ValueError as error:
        parser.error(str(error))

    frames_dir, recommended_dir, name = resolve_output_dirs(args)
    if frames_dir.resolve() == recommended_dir.resolve():
        parser.error("The extracted/input frames folder and recommendation output must differ.")
    if args.video is not None and not args.video.is_file():
        parser.error(f"No such video: {args.video}")
    if args.video is None and not frames_dir.is_dir():
        parser.error(f"No such directory: {frames_dir}")

    # Check both destinations before extracting or scoring anything: on a long recording
    # the committee pass takes minutes, and refusing to overwrite only at the end wastes
    # all of it.
    if recommended_dir.exists() and any(recommended_dir.iterdir()):
        parser.error(f"{recommended_dir} is not empty. Remove it before rerunning the recommender.")

    if args.video is not None:
        if frames_dir.exists() and any(frames_dir.iterdir()):
            parser.error(f"{frames_dir} is not empty. Remove it before rerunning the recommender.")
        print(f"Extracting from {args.video.name} -> {frames_dir}")
        extract_selected_frames(
            args.video,
            frames_dir,
            extraction_fps=5.0,
            max_frames=2000,
            show_progress=True,
        )

    extracted = frames_from_image_directory(frames_dir)
    if not extracted:
        parser.error(f"No PNG frames in {frames_dir}")
    paths = [frame.image_path for frame in extracted]
    print(f"Scoring {len(paths)} frames with a {len(checkpoints)}-model committee")

    device = trainer.resolve_device("auto")
    members = [load_unet_checkpoint(Path(c), device) for c in checkpoints]

    # InferenceDataset applies the same resize-and-pad the trainer uses, and takes no
    # masks -- which is the whole point here, since these frames have none.
    dataset = InferenceDataset(paths)
    committee_masks, diameters, thumbnails = [], [], []
    for index in range(len(dataset)):
        image, _ = dataset[index]
        thumbnails.append(selection.thumbnail(image))
        masks = [selection.predict_masks(m, image, device, 0.5) for m in members]
        committee_masks.append(masks)
        areas = [float(m.sum()) for m in masks]
        diameters.append(2.0 * float(np.sqrt(np.mean(areas) / np.pi)))

    scored = selection.score_frames(committee_masks, paths, diameters=diameters)
    weights = selection.DEFAULT_WEIGHTS
    values = [s.combined(weights) for s in scored]

    min_gap = max(1, len(paths) // (args.budget * 2))
    cutoff = selection.duplicate_cutoff(thumbnails)
    order = sorted(range(len(scored)), key=lambda i: values[i], reverse=True)
    picked = sorted(spread_picks(order, min(args.budget, len(paths)), min_gap, thumbnails, cutoff))

    recommended_dir.mkdir(parents=True, exist_ok=True)
    for index in picked:
        shutil.copy2(paths[index], recommended_dir / paths[index].name)

    manifest = recommended_dir / "selection.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "frame",
                "source_frame_index",
                "score",
                "disagreement",
                "implausibility",
                "temporal",
                "mean_area_fraction",
            ]
        )
        for rank, index in enumerate(sorted(picked, key=lambda i: values[i], reverse=True), 1):
            s = scored[index]
            writer.writerow(
                [
                    rank,
                    paths[index].name,
                    extracted[index].source_frame_index,
                    f"{values[index]:.4f}",
                    f"{s.disagreement:.4f}",
                    f"{s.implausibility:.4f}",
                    f"{s.temporal:.4f}",
                    f"{s.mean_area_fraction:.4f}",
                ]
            )

    print(f"\nRecommended {len(picked)} of {len(paths)} frames -> {recommended_dir}")
    print(f"{'rank':>5}{'frame':>42}{'score':>8}{'disagree':>10}{'implaus':>9}")
    for rank, index in enumerate(sorted(picked, key=lambda i: values[i], reverse=True), 1):
        s = scored[index]
        print(
            f"{rank:>5}{paths[index].name[-40:]:>42}{values[index]:>8.3f}"
            f"{s.disagreement:>10.3f}{s.implausibility:>9.3f}"
        )
    median = float(np.median(values))
    print(f"\nScore of picks {np.mean([values[i] for i in picked]):.3f} vs {median:.3f} median.")
    print(f"Wrote {manifest}")
    print(
        "\nLabel source frames in Labelme, then import the batch with "
        "training/labelme_json2png.py per training/data_collection.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
