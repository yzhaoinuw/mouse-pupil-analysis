# -*- coding: utf-8 -*-
"""Recommend which frames of a new recording to label by hand.

Takes a video or a folder of frames, scores every frame without using any label, and
copies the highest-scoring ones into a folder ready for Labelme.

    python training/recommend_frames.py --video /data/HQL091_sleep251103.avi --budget 20
    python training/recommend_frames.py --frames /data/some_frames --budget 20

From a video it writes a session folder under ``frames_to_label/``::

    frames_to_label/
        HQL091_sleep251103/
            extracted_frames/     every extracted frame
            recommended/          the recommended ones, plus selection.csv

Pass ``--output_dir`` to replace the ``frames_to_label/`` root.

Extraction samples at ``--extraction-fps`` but never exceeds ``--max-extracted``,
falling back to that many equally spaced frames across the whole recording. The
default caps a recording of any length at 2000 frames.

Scoring is ``training/frame_selection.py``; the evidence that it beats picking at
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
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMITTEE = "checkpoints_exp/cvnat/*/best.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "frames_to_label"


def _load(name: str):
    path = PROJECT_ROOT / "training" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"training_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    """Return (frames_dir, promote_dir, name) for either input mode."""
    output_root = args.output_dir or DEFAULT_OUTPUT_DIR
    if args.video is not None:
        stem = args.video.stem
        frames_dir = output_root / stem / "extracted_frames"
    else:
        frames_dir = args.frames
        stem = (
            args.frames.parent.name if args.frames.name == "extracted_frames" else args.frames.name
        )
    promote_dir = output_root / stem / "recommended"
    return frames_dir, promote_dir, stem


def clear_generated_png_outputs(directory: Path, include_manifest: bool = False) -> None:
    """Remove only files this command generates, leaving unrelated files untouched."""
    if not directory.is_dir():
        return
    for path in directory.glob("*.png"):
        path.unlink()
    if include_manifest:
        manifest = directory / "selection.csv"
        if manifest.is_file():
            manifest.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Video to extract frames from.")
    source.add_argument("--frames", type=Path, help="Directory of already-extracted PNGs.")
    parser.add_argument("--budget", type=int, default=20, help="How many frames to recommend.")
    parser.add_argument(
        "--extraction-fps",
        type=float,
        default=5.0,
        help="Sampling rate for video input, capped by --max-extracted (default: 5).",
    )
    parser.add_argument(
        "--max-extracted",
        type=int,
        default=2000,
        help="Ceiling on extracted frames; equally spaced across the recording.",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        help=f"Committee members. Default: {DEFAULT_COMMITTEE} under the project root.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-gap",
        type=int,
        help="Minimum spacing between picks, in extracted-frame positions. "
        "Default spreads the budget over the whole recording.",
    )
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=0.0,
        help="Weight for the frame-to-frame consistency signal. Off by default: it needs "
        "consecutive frames, so it could not be validated on the sparsely sampled pool.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Root for <session>/extracted_frames and <session>/recommended. "
        "Default: frames_to_label under the project root.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Keep near-identical frames. By default a resting animal's repeated frames "
        "are collapsed so the budget is not spent on one posture.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite a non-empty output folder.")
    args = parser.parse_args(argv)

    trainer = _load("run_train")
    selection = _load("frame_selection")
    from mouse_pupil_analysis.extract_frames import extract_selected_frames
    from mouse_pupil_analysis.preprocessing import InferenceDataset
    from mouse_pupil_analysis.pupil_predictions import (
        frames_from_image_directory,
        load_unet_checkpoint,
    )

    checkpoints = args.checkpoints or sorted(PROJECT_ROOT.glob(DEFAULT_COMMITTEE))
    if len(checkpoints) < 2:
        # The default committee lives under the gitignored checkpoints_exp/, so a fresh
        # clone finds nothing here and the reason needs saying out loud.
        default_hint = (
            f"The default committee is {DEFAULT_COMMITTEE} under {PROJECT_ROOT}, which is "
            "gitignored -- it exists only on a machine that has run the cross-validation "
            "sweeps. Produce one with training/run_cv.py, or pass --checkpoints."
            if not args.checkpoints
            else "Pass more paths to --checkpoints."
        )
        parser.error(
            f"Need at least 2 checkpoints for a committee, found {len(checkpoints)}. "
            f"Disagreement between models is the main ranking signal, so a single "
            f"checkpoint cannot rank frames. {default_hint}"
        )

    frames_dir, promote_dir, name = resolve_output_dirs(args)
    if frames_dir.resolve() == promote_dir.resolve():
        parser.error("The extracted/input frames folder and recommendation output must differ.")
    if args.video is not None and not args.video.is_file():
        parser.error(f"No such video: {args.video}")
    if args.video is None and not frames_dir.is_dir():
        parser.error(f"No such directory: {frames_dir}")

    # Check both destinations before extracting or scoring anything: on a long recording
    # the committee pass takes minutes, and refusing to overwrite only at the end wastes
    # all of it.
    if promote_dir.exists() and any(promote_dir.iterdir()) and not args.force:
        parser.error(f"{promote_dir} is not empty. Pass --force to overwrite.")
    if args.force:
        clear_generated_png_outputs(promote_dir, include_manifest=True)

    if args.video is not None:
        if frames_dir.exists() and any(frames_dir.iterdir()) and not args.force:
            parser.error(f"{frames_dir} is not empty. Pass --force to overwrite.")
        if args.force:
            clear_generated_png_outputs(frames_dir)
        print(f"Extracting from {args.video.name} -> {frames_dir}")
        extract_selected_frames(
            args.video,
            frames_dir,
            extraction_fps=args.extraction_fps,
            max_frames=args.max_extracted,
            show_progress=True,
        )

    extracted = frames_from_image_directory(frames_dir)
    if not extracted:
        parser.error(f"No PNG frames in {frames_dir}")
    paths = [frame.image_path for frame in extracted]
    print(f"Scoring {len(paths)} frames with a {len(checkpoints)}-model committee")

    device = trainer.resolve_device(args.device)
    members = [load_unet_checkpoint(Path(c), device) for c in checkpoints]

    # InferenceDataset applies the same resize-and-pad the trainer uses, and takes no
    # masks -- which is the whole point here, since these frames have none.
    dataset = InferenceDataset(paths)
    committee_masks, diameters, thumbnails = [], [], []
    for index in range(len(dataset)):
        image, _ = dataset[index]
        thumbnails.append(selection.thumbnail(image))
        masks = [selection.predict_masks(m, image, device, args.threshold) for m in members]
        committee_masks.append(masks)
        areas = [float(m.sum()) for m in masks]
        diameters.append(2.0 * float(np.sqrt(np.mean(areas) / np.pi)))

    scored = selection.score_frames(committee_masks, paths, diameters=diameters)
    weights = dict(selection.DEFAULT_WEIGHTS) | {"temporal": args.temporal_weight}
    values = [s.combined(weights) for s in scored]

    min_gap = args.min_gap if args.min_gap is not None else max(1, len(paths) // (args.budget * 2))
    cutoff = 0.0 if args.no_dedup else selection.duplicate_cutoff(thumbnails)
    order = sorted(range(len(scored)), key=lambda i: values[i], reverse=True)
    picked = sorted(spread_picks(order, min(args.budget, len(paths)), min_gap, thumbnails, cutoff))

    promote_dir.mkdir(parents=True, exist_ok=True)
    for index in picked:
        shutil.copy2(paths[index], promote_dir / paths[index].name)

    manifest = promote_dir / "selection.csv"
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

    print(f"\nRecommended {len(picked)} of {len(paths)} frames -> {promote_dir}")
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
        "\nLabel source frames in Labelme, then preview and apply the batch with "
        "training/import_labelme_batch.py per training/data_collection.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
