# -*- coding: utf-8 -*-
"""Does frame selection actually find the frames worth labelling?

Ranks each labelled session's frames *without* using its labels, then reveals them and
asks whether the highly-ranked frames really are the ones the model gets wrong. If the
picks are no worse than random, the score is noise and there is no reason to trust it on
a new recording.

    python reports/scripts/validate_frame_selection.py --arm cvnat

The committee for a session is the checkpoints from the fold that held that session out,
so it is always scoring a recording it never trained on -- the same footing as the
cross-validation numbers.

The honest test is **within-session** ranking: the real task is being handed one new
recording and choosing frames from it. Pooling every session together also measures
"can it spot a bad session", which is easier and not the question.

``temporal`` stays at weight 0 here. It needs consecutive frames, and the labelled pool
is a sparse sample, so neighbouring entries are unrelated and any number would be
meaningless.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = PROJECT_ROOT / "training" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"training_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def spearman(xs, ys) -> float:
    if len(xs) < 3:
        return float("nan")

    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for position, index in enumerate(order):
            out[index] = float(position)
        return out

    rx, ry = rank(xs), rank(ys)
    mx, my = st.fmean(rx), st.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm", default="cvnat", help="Checkpoint directory under checkpoints_exp."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--split-manifest", type=Path, default=PROJECT_ROOT / "splits.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--budget", type=int, default=5, help="Frames a human would label.")
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    trainer = _load("run_train")
    splits = _load("data_splits")
    selection = _load("frame_selection")
    from mouse_pupil_analysis.pupil_predictions import load_unet_checkpoint

    manifest = splits.load_manifest(args.split_manifest)
    device = trainer.resolve_device(args.device)
    rng = random.Random(0)

    by_session: dict[str, list] = {}
    for entry in manifest["images"]:
        by_session.setdefault(entry["session"], []).append(entry)

    fold_of = {e["session"]: e["fold"] for e in manifest["sessions"]}
    rows, per_session = [], []

    for session, entries in sorted(by_session.items()):
        fold = fold_of[session]
        members = []
        for seed in args.seeds:
            path = PROJECT_ROOT / "checkpoints_exp" / args.arm / f"{args.arm}_f{fold}_s{seed}"
            if (path / "best.pth").exists():
                members.append(load_unet_checkpoint(path / "best.pth", device))
        if len(members) < 2:
            print(f"skipping {session}: only {len(members)} committee member(s)")
            continue

        paths = [PROJECT_ROOT / e["image"] for e in entries]
        mask_paths = [PROJECT_ROOT / e["mask"] for e in entries]
        dataset = trainer.SegmentationDataset(paths, mask_paths, augment=False)

        committee_masks, truth_iou = [], []
        for i in range(len(dataset)):
            image, target = dataset[i]
            masks = [selection.predict_masks(m, image, device, args.threshold) for m in members]
            committee_masks.append(masks)
            gt = target[0].numpy() > 0.5
            # Error to be predicted: how badly the committee does on this frame, averaged.
            ious = []
            for mask in masks:
                union = np.logical_or(mask, gt).sum()
                ious.append(np.logical_and(mask, gt).sum() / union if union else 1.0)
            truth_iou.append(float(np.mean(ious)))

        scored = selection.score_frames(committee_masks, paths)
        error = [1.0 - v for v in truth_iou]
        combined = [s.combined() for s in scored]

        rows.extend(
            {
                "session": session,
                "error": e,
                "disagreement": s.disagreement,
                "implausibility": s.implausibility,
                "combined": c,
            }
            for s, e, c in zip(scored, error, combined)
        )

        if len(entries) > args.budget:
            order = sorted(range(len(scored)), key=lambda i: combined[i], reverse=True)
            picked = st.fmean(truth_iou[i] for i in order[: args.budget])
            draws = [
                st.fmean(rng.sample(truth_iou, args.budget)) for _ in range(args.random_trials)
            ]
            per_session.append(
                {
                    "session": session,
                    "n": len(entries),
                    "rho": spearman(combined, error),
                    "picked_iou": picked,
                    "random_iou": st.fmean(draws),
                    "worst_possible": st.fmean(sorted(truth_iou)[: args.budget]),
                    # Kept so alternative weightings can be compared without re-predicting.
                    "scored": scored,
                    "truth_iou": truth_iou,
                }
            )

    print(
        f"\n{'=' * 78}\nWITHIN-SESSION (the real task: one recording, pick {args.budget})\n{'=' * 78}"
    )
    print(f"{'session':<38}{'n':>4}{'rho':>7}{'picked':>8}{'random':>8}{'best possible':>15}")
    for r in sorted(per_session, key=lambda r: r["picked_iou"] - r["random_iou"]):
        print(
            f"{r['session'][:36]:<38}{r['n']:>4}{r['rho']:>7.2f}{r['picked_iou']:>8.3f}"
            f"{r['random_iou']:>8.3f}{r['worst_possible']:>15.3f}"
        )

    if per_session:
        picked = st.fmean(r["picked_iou"] for r in per_session)
        chance = st.fmean(r["random_iou"] for r in per_session)
        floor = st.fmean(r["worst_possible"] for r in per_session)
        rhos = [r["rho"] for r in per_session if not np.isnan(r["rho"])]
        wins = sum(1 for r in per_session if r["picked_iou"] < r["random_iou"])
        print(f"\n  picked frames mean IoU : {picked:.4f}  (lower is better -- harder frames)")
        print(f"  random frames mean IoU : {chance:.4f}")
        print(f"  best possible (oracle) : {floor:.4f}")
        captured = (chance - picked) / (chance - floor) if chance > floor else float("nan")
        print(f"  fraction of the oracle gap captured: {captured:.1%}")
        print(f"  beats random in {wins}/{len(per_session)} sessions")
        print(f"  mean within-session rho: {st.fmean(rhos):+.3f}")

    print(f"\n{'=' * 78}\nSIGNAL BREAKDOWN (pooled over all frames)\n{'=' * 78}")
    error = [r["error"] for r in rows]
    for signal in ("disagreement", "implausibility", "combined"):
        print(f"  {signal:<16} rho vs error {spearman([r[signal] for r in rows], error):+.3f}")
    print(f"\n  frames scored: {len(rows)}")

    print(
        f"\n{'=' * 78}\nWEIGHTING COMPARISON (same predictions, budget {args.budget})\n{'=' * 78}"
    )
    print(f"{'weights (disagree/implaus)':<32}{'picked IoU':>12}{'oracle gap':>13}{'wins':>7}")
    chance = st.fmean(r["random_iou"] for r in per_session)
    floor = st.fmean(r["worst_possible"] for r in per_session)
    for label, weights in (
        ("1.0 / 0.0  disagreement only", {"disagreement": 1.0, "implausibility": 0.0}),
        ("1.0 / 0.25", {"disagreement": 1.0, "implausibility": 0.25}),
        ("1.0 / 0.5", {"disagreement": 1.0, "implausibility": 0.5}),
        ("1.0 / 1.0  equal (default)", {"disagreement": 1.0, "implausibility": 1.0}),
        ("0.0 / 1.0  implausibility only", {"disagreement": 0.0, "implausibility": 1.0}),
    ):
        picks, wins = [], 0
        for r in per_session:
            values = [s.combined(weights) for s in r["scored"]]
            order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
            got = st.fmean(r["truth_iou"][i] for i in order[: args.budget])
            picks.append(got)
            wins += got < r["random_iou"]
        mean_pick = st.fmean(picks)
        gap = (chance - mean_pick) / (chance - floor) if chance > floor else float("nan")
        print(f"{label:<32}{mean_pick:>12.4f}{gap:>12.1%}{wins:>5}/{len(per_session)}")
    print(f"\n  random baseline {chance:.4f}, oracle {floor:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
