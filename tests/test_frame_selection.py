# -*- coding: utf-8 -*-
"""Tests for the label-free frame scoring behind ``training/recommend_frames.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = PROJECT_ROOT / "training" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"training_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


selection = _load("_frame_scoring")
recommend = _load("recommend_frames")


def disc(size: int, radius: int, centre=None) -> np.ndarray:
    centre = centre or (size // 2, size // 2)
    yy, xx = np.ogrid[:size, :size]
    return (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius**2


def test_identical_masks_do_not_disagree():
    mask = disc(64, 10)
    assert selection.disagreement_score([mask, mask.copy(), mask.copy()]) == 0.0


def test_disagreement_grows_as_masks_diverge():
    base = disc(64, 10)
    near = selection.disagreement_score([base, disc(64, 11)])
    far = selection.disagreement_score([base, disc(64, 25)])
    assert 0.0 < near < far


def test_two_empty_masks_agree_rather_than_disagree():
    """An unanimous "nothing here" must not read as maximal disagreement."""
    empty = np.zeros((32, 32), dtype=bool)
    assert selection.disagreement_score([empty, empty.copy()]) == 0.0


def test_a_plausible_pupil_scores_near_zero():
    assert selection.implausibility_score(disc(148, 22)) < 0.1


@pytest.mark.parametrize(
    ("mask", "reason"),
    [
        (np.zeros((148, 148), dtype=bool), "found nothing"),
        (np.ones((148, 148), dtype=bool), "covers the whole frame"),
    ],
)
def test_implausible_masks_are_flagged(mask, reason):
    assert selection.implausibility_score(mask) > 0.3, reason


def test_an_aperture_sized_blob_outranks_a_pupil_sized_one():
    """Ranking is what matters, and the absolute score understates the problem.

    A big round blob violates only the area prior, and averaging over the shape terms
    divides that by four -- so this scores about 0.25, not 1.0. Taking the max instead
    reads better but measured worse end to end; see ``implausibility_score``.
    """
    aperture = selection.implausibility_score(disc(148, 70))
    pupil = selection.implausibility_score(disc(148, 22))
    assert aperture > pupil
    assert aperture < 0.5, "documents the known dilution, not an endorsement of it"


def test_two_blobs_are_less_plausible_than_one():
    single = disc(148, 18)
    double = single | disc(148, 12, centre=(120, 120))
    assert selection.implausibility_score(double) > selection.implausibility_score(single)


def test_temporal_score_flags_the_frame_that_jumps():
    scores = selection.temporal_score([20.0, 20.0, 20.0, 60.0, 20.0, 20.0, 20.0])
    assert scores[3] == max(scores)
    assert scores[3] > 0.5


def test_duplicate_cutoff_survives_a_pool_dominated_by_duplicates():
    """A low percentile collapses to zero here, silently disabling deduplication."""
    rng = np.random.default_rng(0)
    distinct = [rng.normal(size=64) for _ in range(5)]
    thumbnails = [distinct[i % len(distinct)].copy() for i in range(100)]
    assert selection.duplicate_cutoff(thumbnails) > 0.0


def test_spread_picks_rejects_near_duplicates_far_apart_in_time():
    a, b = np.zeros(16), np.ones(16) * 5.0
    # Frames 0 and 50 are identical in appearance despite being far apart.
    thumbnails = [a if i in (0, 50) else b for i in range(60)]
    picked = recommend.spread_picks(
        [0, 50, 25], budget=2, min_gap=1, thumbnails=thumbnails, cutoff=1.0
    )
    assert picked == [0, 25]


def test_spread_picks_honours_minimum_spacing():
    assert recommend.spread_picks([10, 11, 12, 40], budget=2, min_gap=5) == [10, 40]


def test_spread_picks_fills_the_budget_when_constraints_cannot_be_met():
    """Returning fewer frames than asked for would silently shrink the labelling batch."""
    assert len(recommend.spread_picks([0, 1, 2, 3], budget=3, min_gap=99)) == 3


def test_force_cleanup_removes_only_generated_files(tmp_path):
    (tmp_path / "old.png").write_bytes(b"old")
    (tmp_path / "selection.csv").write_text("old", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")

    recommend.clear_generated_png_outputs(tmp_path, include_manifest=True)

    assert not (tmp_path / "old.png").exists()
    assert not (tmp_path / "selection.csv").exists()
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_video_outputs_default_to_frames_to_label():
    args = SimpleNamespace(
        video=PROJECT_ROOT / "videos" / "HQL095_sleep260324_010_eye.avi",
        frames=None,
        output_dir=None,
    )

    frames_dir, recommended_dir, name = recommend.resolve_output_dirs(args)

    session_dir = PROJECT_ROOT / "frames_to_label" / name
    assert name == "HQL095_sleep260324_010_eye"
    assert frames_dir == session_dir / "extracted_frames"
    assert recommended_dir == session_dir / "recommended"


def test_output_dir_replaces_the_default_root(tmp_path):
    custom_root = tmp_path / "label_queue"
    args = SimpleNamespace(
        video=Path("videos/new_session.avi"),
        frames=None,
        output_dir=custom_root,
    )

    frames_dir, recommended_dir, _ = recommend.resolve_output_dirs(args)

    assert frames_dir == custom_root / "new_session" / "extracted_frames"
    assert recommended_dir == custom_root / "new_session" / "recommended"


def test_existing_frames_stay_in_place_while_recommendations_use_output_root(tmp_path):
    frames = tmp_path / "source" / "extracted_frames"
    output_root = tmp_path / "recommendations"
    args = SimpleNamespace(video=None, frames=frames, output_dir=output_root)

    frames_dir, recommended_dir, name = recommend.resolve_output_dirs(args)

    assert name == "source"
    assert frames_dir == frames
    assert recommended_dir == output_root / "source" / "recommended"
