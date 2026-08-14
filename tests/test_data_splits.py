"""Coverage for recording-grouped splitting.

The failure this guards against is silent: a split that looks grouped but still puts
two files from one sitting on opposite sides of the boundary reports a generalisation
number that is really an interpolation number. Nothing raises when that happens, so
the grouping rules and the stability guarantee are pinned explicitly here.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_data_splits():
    path = PROJECT_ROOT / "training" / "data_splits.py"
    spec = importlib.util.spec_from_file_location("training_data_splits_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


data_splits = _load_data_splits()


# Real stems from the maintained dataset, one per naming scheme.
HQL_STEM = "HQL080_whiskerb_250722_007_eye_06622"
DATE_STEM = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_0000"


def _write_pair(root: Path, split: str, stem: str, radius: int) -> None:
    """Write one image/mask pair with a mask of a known size."""
    (root / f"images_{split}").mkdir(parents=True, exist_ok=True)
    (root / f"masks_{split}").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((60, 60), dtype=np.uint8)).save(root / f"images_{split}/{stem}.png")

    mask = np.zeros((60, 60), dtype=np.uint8)
    grid_y, grid_x = np.ogrid[:60, :60]
    mask[(grid_y - 30) ** 2 + (grid_x - 30) ** 2 <= radius**2] = 255
    Image.fromarray(mask).save(root / f"masks_{split}/{stem}.png")


@pytest.fixture
def pool(tmp_path: Path) -> Path:
    """Six sessions spread across both naming schemes and both pool folders."""
    for index in range(4):
        _write_pair(tmp_path, "train", f"HQL0{index}0_sleep2506{index}0_003_eye_0000{index}", 10)
        _write_pair(tmp_path, "train", f"HQL0{index}0_sleep2506{index}0_007_eye_0000{index}", 10)
    # Same session as the first pair above, but sitting in the validation folder.
    _write_pair(tmp_path, "validation", "HQL000_sleep250600_009_eye_00009", 10)
    _write_pair(tmp_path, "train", DATE_STEM, 20)
    _write_pair(tmp_path, "validation", "250616_5120_P_sleep_2025-06-16T16-31-19.701_0000", 20)
    return tmp_path


def test_hql_recording_files_from_one_sitting_share_a_session():
    first = data_splits.parse_identity("HQL086_whiskerb250923_002_eye_01234")
    second = data_splits.parse_identity("HQL086_whiskerb250923_008_eye_09999")

    assert first.recording != second.recording
    assert first.session == second.session == "HQL086_whiskerb250923"
    assert first.animal == "HQL086"


def test_date_id_recordings_from_one_day_share_a_session():
    first = data_splits.parse_identity(DATE_STEM)
    second = data_splits.parse_identity(
        "250530_5003_Green_Training_very_dm_light_2025-05-30T11-02-03.500_0042"
    )

    assert first.recording != second.recording
    assert first.session == second.session == "250530_5003_Green_Training_very_dm_light"
    assert first.animal == "5003"
    assert first.cohort == "date_id"


def test_different_dates_are_different_sessions():
    september = data_splits.parse_identity("HQL086_sleep250909_011_eye_0001")
    december = data_splits.parse_identity("HQL086_sleep250912_006_eye_0001")

    assert september.session != december.session


def test_unrecognised_names_become_their_own_session():
    identity = data_splits.parse_identity("some_new_camera_export_frame7")

    # Over-separating is safe; silently merging two settings is not.
    assert identity.cohort == "unparsed"
    assert identity.session == "some_new_camera_export_frame7"


def test_pool_spans_both_folders_and_no_session_crosses_a_fold(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)

    assert manifest["n_images"] == 11
    assert manifest["n_sessions"] == 6

    fold_of_session = {entry["session"]: entry["fold"] for entry in manifest["sessions"]}
    for entry in manifest["images"]:
        assert entry["fold"] == fold_of_session[entry["session"]]

    # The HQL000 session has images in both images_train and images_validation.
    split_session = [e for e in manifest["images"] if e["session"] == "HQL000_sleep250600"]
    assert {Path(e["image"]).parent.name for e in split_session} == {
        "images_train",
        "images_validation",
    }
    assert len({e["fold"] for e in split_session}) == 1


def test_every_fold_is_used_and_folds_partition_the_pool(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)

    stems = set()
    for fold in range(3):
        (train_images, _), (val_images, _) = data_splits.fold_paths(manifest, fold, pool)
        assert train_images and val_images
        assert not {p.stem for p in train_images} & {p.stem for p in val_images}
        assert len(train_images) + len(val_images) == manifest["n_images"]
        stems |= {p.stem for p in val_images}

    assert len(stems) == manifest["n_images"]


def test_fold_assignment_is_deterministic(pool: Path):
    first = data_splits.build_manifest(pool, n_folds=3, generated="2026-01-01")
    second = data_splits.build_manifest(pool, n_folds=3, generated="2026-01-01")

    assert first == second


def test_adding_a_session_leaves_existing_assignments_untouched(pool: Path):
    before = data_splits.build_manifest(pool, n_folds=3)
    existing = {entry["session"]: entry["fold"] for entry in before["sessions"]}

    _write_pair(pool, "train", "HQL999_whiskerb260101_001_eye_00001", 12)
    after = data_splits.build_manifest(pool, n_folds=3, existing=existing)

    unchanged = {
        entry["session"]: entry["fold"]
        for entry in after["sessions"]
        if entry["session"] in existing
    }
    assert unchanged == existing
    assert any(entry["session"] == "HQL999_whiskerb260101" for entry in after["sessions"])


def test_reassigning_without_history_may_repack(pool: Path):
    before = data_splits.build_manifest(pool, n_folds=3)
    _write_pair(pool, "train", "HQL999_whiskerb260101_001_eye_00001", 12)
    fresh = data_splits.build_manifest(pool, n_folds=3, existing=None)

    # Not an assertion about *which* folds move -- only that a repack is unconstrained
    # by history, which is why --reassign is opt-in.
    assert fresh["n_sessions"] == before["n_sessions"] + 1


def test_more_folds_than_sessions_is_rejected(pool: Path):
    with pytest.raises(ValueError, match="every fold needs at least one whole session"):
        data_splits.build_manifest(pool, n_folds=99)


def test_duplicate_stem_across_pool_folders_is_rejected(pool: Path):
    _write_pair(pool, "validation", DATE_STEM, 20)

    with pytest.raises(ValueError, match="appears in both"):
        data_splits.build_manifest(pool, n_folds=3)


def test_manifest_round_trips_and_rejects_an_unknown_schema(pool: Path, tmp_path: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)
    path = tmp_path / "splits.json"
    data_splits.write_manifest(path, manifest)

    assert data_splits.load_manifest(path) == manifest
    assert data_splits.existing_assignment(path) == {
        entry["session"]: entry["fold"] for entry in manifest["sessions"]
    }

    manifest["schema"] = 99
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="understands"):
        data_splits.load_manifest(path)


def test_existing_assignment_of_a_missing_file_is_empty(tmp_path: Path):
    assert data_splits.existing_assignment(tmp_path / "absent.json") == {}


def test_session_lookup_covers_every_image(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)
    lookup = data_splits.session_of_stem(manifest)

    assert len(lookup) == manifest["n_images"]
    assert lookup[DATE_STEM] == "250530_5003_Green_Training_very_dm_light"
