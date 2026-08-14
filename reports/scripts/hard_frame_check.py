# -*- coding: utf-8 -*-
"""Gate candidate checkpoints on real frames that validation does not cover.

``sample_data/raw_frames/recording_250616`` contains a small pupil that is not in
the validation set. A checkpoint that loses it entirely is broken in a way
validation IoU cannot see -- in the 2026-08-14 seed study the three runs with the
*highest* validation IoU were exactly the three that lost it.

Use this as a pass/fail gate before promoting anything, not as a score to
maximise: selecting on it would repeat the mistake it exists to catch.

    python reports/scripts/hard_frame_check.py --checkpoints checkpoints_exp/*/best.pth
"""

from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES = PROJECT_ROOT / "sample_data" / "raw_frames" / "recording_250616"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        help="Checkpoints to test. Default: the packaged one.",
    )
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument(
        "--min-diameter",
        type=float,
        default=5.0,
        help="A detection below this is treated as a lost pupil.",
    )
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")
    from mouse_pupil_analysis.api import analyze_frames
    from mouse_pupil_analysis.pupil_predictions import find_default_checkpoint

    checkpoints = args.checkpoints or [find_default_checkpoint()]
    failures = 0
    print(f"frames: {args.frames}\n")
    for checkpoint in checkpoints:
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze_frames(
                args.frames, result_dir=Path(tmp), checkpoint=checkpoint, num_workers=0
            )
        diameters = result.analysis_table["estimated_pupil_diameter"].to_numpy(dtype=float)
        lost = diameters < args.min_diameter
        failures += int(lost.any())
        label = "LOST" if lost.any() else "ok"
        print(
            f"  [{label:>4}] {checkpoint.parent.name}/{checkpoint.name}  "
            f"thr={result.prediction_threshold:g}  "
            f"diameters={[f'{value:.1f}' for value in diameters]}"
        )

    if failures:
        print(f"\n{failures} of {len(checkpoints)} checkpoint(s) lost a pupil entirely.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
