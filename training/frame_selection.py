# -*- coding: utf-8 -*-
"""Score unlabelled frames by how much labelling them would be worth.

The point is to pick, out of a new recording's thousands of frames, the handful worth
labelling by hand. Every signal here is computable without labels.

**Do not rank by model confidence.** The failure this exists to catch is confident:
on 2026-08-16 the model segmented the entire dark eye aperture instead of the pupil on
two sessions, over-predicting by 4.8x and 7.8x, at p=0.99 on every frame. Entropy or
margin sampling would rank those frames as the *least* informative in the recording.

Three label-free signals replace it:

``disagreement``
    Several checkpoints trained on different session subsets each predict a mask; the
    score is how much they differ from each other. Distinct from confidence -- a
    committee can be unanimous and wrong, which is exactly what has to be tested.
``implausibility``
    A pupil covering a third of the frame, or badly non-circular, or running off the
    edge, is wrong regardless of what any label says. This is the aperture-grabbing
    signature stated as a geometric prior.
``temporal``
    Within one recording the pupil cannot jump between neighbouring frames. Needs
    consecutive frames, so it is not measurable on the sparsely-sampled labelled pool.

``reports/scripts/validate_frame_selection.py`` checks these against known labels
before any of it is trusted on a new recording. As of 2026-08-16 it picks frames
averaging IoU 0.31 against 0.51 for random picks, with an oracle floor of 0.29 --
89% of the available gap, beating random in 10 of 10 sessions.

**Why implausibility earns its small weight.** Disagreement alone is near-blind to the
aperture-grabbing failure, because committee members trained on different folds share
the bias: they agree with each other while all being wrong, so there is nothing to
detect. On ``251016_5212_purple_Day10`` disagreement alone scores rho 0.09 -- chance --
and adding implausibility at 0.25 lifts it to 0.84, picking exactly the worst frames
available. The geometric prior sees the over-prediction that unanimity hides.

**Known blind spot.** ``HQL090_sleep251012`` stays weak at rho 0.34 against 0.78
average. Its pupils are the smallest in the pool, so a 7.8x over-prediction still lands
near an area fraction of 0.23 -- just inside ``PLAUSIBLE_AREA_FRACTION``. Tightening
that bound would catch it, but the bound would then be fitted to ten sessions, so it is
left loose deliberately. Expect frame ranking to be weakest on recordings whose pupils
are much smaller than anything in training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

# A mouse pupil crop that segments sanely occupies a middling fraction of the frame.
# Bounds are deliberately loose: this flags absurdity, not suboptimality.
PLAUSIBLE_AREA_FRACTION = (0.01, 0.25)
MIN_CIRCULARITY = 0.55
BORDER_MARGIN_PX = 2

# Disagreement carries the signal; implausibility is a minor correction. Measured on the
# 2026-08-16 pool: disagreement alone captures 80.9% of the oracle gap, weighting the two
# equally drops it to 81.5% with one session lost to chance, and 0.25 reaches 89.2% with
# every session beating random. Anything in 0.25-0.5 performs about the same -- the exact
# value was picked over 10 sessions and is not tuned. Temporal needs consecutive frames.
DEFAULT_WEIGHTS = {"disagreement": 1.0, "implausibility": 0.25, "temporal": 0.0}


@dataclass(frozen=True)
class FrameScore:
    """Per-frame label-free scores. Higher means more worth labelling."""

    path: Path
    disagreement: float
    implausibility: float
    temporal: float
    mean_area_fraction: float
    committee_size: int

    def combined(self, weights: dict[str, float] | None = None) -> float:
        w = weights or DEFAULT_WEIGHTS
        return (
            w.get("disagreement", 0.0) * self.disagreement
            + w.get("implausibility", 0.0) * self.implausibility
            + w.get("temporal", 0.0) * self.temporal
        )


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        # Two empty masks agree perfectly; treating that as disagreement would rank
        # every frame where nothing was found as maximally informative.
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def disagreement_score(masks: list[np.ndarray]) -> float:
    """Mean pairwise (1 - IoU) across committee masks, in [0, 1]."""
    if len(masks) < 2:
        return 0.0
    scores = [
        1.0 - _binary_iou(masks[i], masks[j])
        for i in range(len(masks))
        for j in range(i + 1, len(masks))
    ]
    return float(np.mean(scores))


def implausibility_score(mask: np.ndarray) -> float:
    """How far one predicted mask departs from pupil-shaped, in [0, 1].

    Each component is a violation of a geometric prior that holds for any real pupil,
    so none of them needs a ground-truth mask to evaluate.
    """
    area = float(mask.sum())
    if area == 0:
        return 1.0  # found nothing at all: worth a human look

    penalties = []

    fraction = area / mask.size
    low, high = PLAUSIBLE_AREA_FRACTION
    if fraction < low:
        penalties.append(min(1.0, (low - fraction) / low))
    elif fraction > high:
        penalties.append(min(1.0, (fraction - high) / high))
    else:
        penalties.append(0.0)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, closed=True)
        circularity = (
            0.0 if perimeter <= 0 else 4.0 * math.pi * cv2.contourArea(largest) / perimeter**2
        )
        penalties.append(max(0.0, (MIN_CIRCULARITY - circularity) / MIN_CIRCULARITY))
        # More than one blob means the model found several "pupils".
        penalties.append(0.0 if len(contours) == 1 else 1.0)

    m = BORDER_MARGIN_PX
    touches = mask[:m, :].any() or mask[-m:, :].any() or mask[:, :m].any() or mask[:, -m:].any()
    penalties.append(1.0 if touches else 0.0)

    return float(np.mean(penalties))


def temporal_score(diameters: list[float], window: int = 5) -> list[float]:
    """Per-frame deviation from the local median diameter, normalised.

    Only meaningful for consecutive frames from one recording. Sparse samples make
    neighbouring entries unrelated, which is why the harness leaves this at weight 0.
    """
    if len(diameters) < 3:
        return [0.0] * len(diameters)
    values = np.asarray(diameters, dtype=float)
    scale = np.median(values) or 1.0
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
        neighbours = np.concatenate([values[lo:i], values[i + 1 : hi]])
        local = np.median(neighbours) if neighbours.size else values[i]
        out.append(float(min(1.0, abs(values[i] - local) / scale)))
    return out


def predict_masks(
    model, image_tensor: torch.Tensor, device: torch.device, threshold: float
) -> np.ndarray:
    """Return one binary mask for one already-preprocessed image tensor."""
    with torch.no_grad():
        probability = torch.sigmoid(model(image_tensor[None].to(device)))[0, 0]
    return probability.cpu().numpy() > threshold


def score_frames(
    committee_masks: list[list[np.ndarray]],
    paths: list[Path],
    diameters: list[float] | None = None,
) -> list[FrameScore]:
    """Combine committee predictions into per-frame scores.

    ``committee_masks[i]`` holds one mask per committee member for frame ``i``.
    """
    temporal = temporal_score(diameters) if diameters is not None else [0.0] * len(committee_masks)
    scores = []
    for i, masks in enumerate(committee_masks):
        areas = [float(m.sum()) / m.size for m in masks]
        scores.append(
            FrameScore(
                path=paths[i],
                disagreement=disagreement_score(masks),
                implausibility=float(np.mean([implausibility_score(m) for m in masks])),
                temporal=temporal[i],
                mean_area_fraction=float(np.mean(areas)),
                committee_size=len(masks),
            )
        )
    return scores
