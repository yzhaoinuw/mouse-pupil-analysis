# -*- coding: utf-8 -*-
"""Gate candidate checkpoints on real frames that validation does not cover.

A checkpoint can lose a real pupil entirely and still score well: in the
2026-08-14 seed study the three runs with the *highest* validation IoU were
exactly the three that lost a small pupil no validation image covered. Point this
at unlabelled frames from a recording and check that the pupil is still found.

Use it as a pass/fail gate before promoting anything, not as a score to maximise:
selecting on it would repeat the mistake it exists to catch.

    python reports/scripts/hard_frame_check.py \
        --frames <dir of unlabelled frames> \
        --checkpoints checkpoints_exp/*/best.pth

``--frames`` has no default. It used to point at ``sample_data/raw_frames``, a
small bundled set of unlabelled frames, which was removed on 2026-08-15. Give it
frames the candidate has never trained on -- frames from ``labeled_data/`` are
training data and would make this gate report nothing.
"""

from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        help="Checkpoints to test. Default: the packaged one.",
    )
    parser.add_argument(
        "--frames",
        type=Path,
        required=True,
        help="Directory of unlabelled frames the candidate has never trained on.",
    )
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
