"""Coverage for stratified, recording-grouped splitting.

Two failures are guarded here and both are silent. A split that looks grouped but
still puts two frames from one sitting on opposite sides of the boundary reports an
interpolation number as if it were a generalisation number -- worth 0.25 IoU on this
pool against a 0.02 noise floor. The required session-folder layout keeps identity
explicit, while frozen assignments keep comparisons stable as new data arrives.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = PROJECT_ROOT / "training" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"training_{name}_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


data_splits = _load("data_splits")


def _write_pair(
    root: Path,
    stem: str,
    radius: int,
    session: str,
    grey: int = 100,
) -> Path:
    """Write one labelled pair into a recording-session folder."""
    image_dir = root / "labeled_frames" / session / "images"
    mask_dir = root / "labeled_frames" / session / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.full((60, 60), grey, dtype=np.uint8)).save(image_dir / f"{stem}.png")

    mask = np.zeros((60, 60), dtype=np.uint8)
    grid_y, grid_x = np.ogrid[:60, :60]
    mask[(grid_y - 30) ** 2 + (grid_x - 30) ** 2 <= radius**2] = 255
    Image.fromarray(mask).save(mask_dir / f"{stem}.png")

    return image_dir / f"{stem}.png"


@pytest.fixture
def pool(tmp_path: Path) -> Path:
    """Five sessions recorded as intake folders.

    Radii and greys are chosen so the sessions land in different diameter and
    brightness bands rather than all collapsing into one stratum.
    """
    for index in range(5):
        session = f"rig{index}_day{index}"
        for frame in range(3):
            _write_pair(
                tmp_path,
                f"anything_at_all_{index}_{frame}",
                radius=4 + 4 * index,
                session=session,
                grey=40 + 40 * index,
            )
    return tmp_path


# --- grouping and the freeze guarantee --------------------------------------------


def test_no_session_is_split_across_folds(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)

    fold_of_session = {entry["session"]: entry["fold"] for entry in manifest["sessions"]}
    for entry in manifest["images"]:
        assert entry["fold"] == fold_of_session[entry["session"]]


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
    was = {entry["session"]: entry["fold"] for entry in before["sessions"]}

    _write_pair(pool, "brand_new", radius=11, session="rig7_day7")
    after = data_splits.build_manifest(pool, n_folds=3, previous=before)

    unchanged = {e["session"]: e["fold"] for e in after["sessions"] if e["session"] in was}
    assert unchanged == was
    assert any(entry["session"] == "rig7_day7" for entry in after["sessions"])


def test_a_new_image_joins_the_fold_its_session_already_has(pool: Path):
    before = data_splits.build_manifest(pool, n_folds=3)
    existing_fold = next(e["fold"] for e in before["sessions"] if e["session"] == "rig0_day0")

    _write_pair(pool, "late_arrival", radius=4, session="rig0_day0", grey=40)
    after = data_splits.build_manifest(pool, n_folds=3, previous=before)

    added = next(e for e in after["images"] if e["key"].endswith("late_arrival"))
    assert added["session"] == "rig0_day0"
    assert added["fold"] == existing_fold


def test_changing_fold_count_requires_explicit_reassignment(pool: Path):
    before = data_splits.build_manifest(pool, n_folds=3)

    with pytest.raises(ValueError, match="pass --reassign"):
        data_splits.build_manifest(pool, n_folds=2, previous=before)

    after = data_splits.build_manifest(pool, n_folds=2, previous=before, reassign=True)
    assert after["n_folds"] == 2


def test_terminal_flag_names_the_number_of_folds(pool: Path):
    assert (
        data_splits.main(
            [
                "--labeled_frames_dir",
                str(pool / "labeled_frames"),
                "--n_folds",
                "3",
                "--show",
            ]
        )
        == 0
    )

    with pytest.raises(SystemExit):
        data_splits.main(["--folds", "3"])


def test_more_folds_than_sessions_is_rejected(pool: Path):
    with pytest.raises(ValueError, match="every fold needs at least one whole session"):
        data_splits.build_manifest(pool, n_folds=99)


def test_two_intake_folders_may_reuse_the_same_filenames(tmp_path: Path):
    # The whole point of intake folders is that filenames need not be unique, and
    # per-recording exports routinely restart numbering at frame_0001.
    for session in ("rig1_day1", "rig2_day2"):
        for frame in range(3):
            _write_pair(tmp_path, f"frame_{frame:04d}", radius=8, session=session)

    manifest = data_splits.build_manifest(tmp_path, n_folds=2)

    assert manifest["n_images"] == 6
    assert {e["session"] for e in manifest["sessions"]} == {"rig1_day1", "rig2_day2"}
    assert "rig1_day1/frame_0000" in {e["key"] for e in manifest["images"]}
    # Same filename, two sessions, two distinct keys, and they may not share a fold.
    same_name = [e for e in manifest["images"] if e["key"].endswith("frame_0000")]
    assert len(same_name) == 2
    assert len({e["session"] for e in same_name}) == 2


def test_masks_live_beside_their_images_in_the_session_folder(tmp_path: Path):
    _write_pair(tmp_path, "frame1", radius=8, session="rig1_day1")
    _write_pair(tmp_path, "frame2", radius=12, session="rig2_day2")

    manifest = data_splits.build_manifest(tmp_path, n_folds=2)
    masks = {e["key"]: e["mask"] for e in manifest["images"]}
    assert masks["rig1_day1/frame1"] == "labeled_frames/rig1_day1/masks/frame1.png"
    assert masks["rig2_day2/frame2"] == "labeled_frames/rig2_day2/masks/frame2.png"


def test_an_image_with_no_mask_is_rejected(tmp_path: Path):
    _write_pair(tmp_path, "frame1", radius=8, session="rig1_day1")
    _write_pair(tmp_path, "frame2", radius=12, session="rig2_day2")
    (tmp_path / "labeled_frames" / "rig1_day1" / "masks" / "frame1.png").unlink()

    with pytest.raises(FileNotFoundError, match="No mask for"):
        data_splits.build_manifest(tmp_path, n_folds=2)


def test_a_session_folder_without_images_is_rejected(tmp_path: Path):
    _write_pair(tmp_path, "frame1", radius=8, session="rig1_day1")
    (tmp_path / "labeled_frames" / "stray_folder").mkdir()

    with pytest.raises(FileNotFoundError, match="has no images/ directory"):
        data_splits.build_manifest(tmp_path, n_folds=2)


# --- stratification ---------------------------------------------------------------


def test_diameter_bands_are_spread_across_folds_rather_than_concentrated(tmp_path: Path):
    # Nine sessions, three per diameter band. A grouped-but-unstratified packing can
    # put every small session in one fold; this is what stops that.
    for index in range(9):
        radius = 4 + (index % 3) * 8
        for frame in range(2):
            _write_pair(
                tmp_path,
                f"s{index}_f{frame}",
                radius=radius,
                session=f"rig{index}",
                grey=50 + (index % 3) * 50,
            )

    manifest = data_splits.build_manifest(tmp_path, n_folds=3)

    per_fold = {}
    for entry in manifest["sessions"]:
        per_fold.setdefault(entry["fold"], []).append(entry["stratum"][:2])
    for fold, bands in per_fold.items():
        assert len(set(bands)) == 3, f"fold {fold} covers only {set(bands)}"


def test_manifest_records_the_features_it_stratified_on(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)

    assert manifest["stratified_by"] == ["median_diameter", "median_brightness"]
    assert set(manifest["stratum_cutpoints"]) == {"diameter", "brightness"}
    for entry in manifest["images"]:
        assert entry["diameter"] > 0
        assert 0 <= entry["brightness"] <= 255


def test_brightness_tracks_the_background_not_the_pupil(tmp_path: Path):
    from mouse_pupil_analysis.augmentation import image_background_brightness

    _write_pair(tmp_path, "dim", radius=6, session="a", grey=40)
    _write_pair(tmp_path, "bright", radius=20, session="b", grey=200)

    dim = image_background_brightness(
        tmp_path / "labeled_frames/a/images/dim.png", tmp_path / "labeled_frames/a/masks/dim.png"
    )
    bright = image_background_brightness(
        tmp_path / "labeled_frames/b/images/bright.png",
        tmp_path / "labeled_frames/b/masks/bright.png",
    )
    # The bright image has the far larger pupil, so a whole-frame mean would narrow
    # the gap; excluding the mask keeps this a measure of lighting.
    assert dim == pytest.approx(40, abs=1)
    assert bright == pytest.approx(200, abs=1)


# --- the holdout gate -------------------------------------------------------------


def test_holdout_sessions_appear_in_no_fold(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3, final_test_sessions={"rig0_day0"})

    gate = [e for e in manifest["sessions"] if e["session"] == "rig0_day0"][0]
    assert gate["holdout"] is True
    assert gate["fold"] == data_splits.HOLDOUT_FOLD

    for fold in range(3):
        (train_images, _), (val_images, _) = data_splits.fold_paths(manifest, fold, pool)
        seen = {p.parents[1].name for p in train_images} | {p.parents[1].name for p in val_images}
        assert "rig0_day0" not in seen


def test_validation_holdout_is_excluded_from_cv_and_loaded_for_normal_training(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3, validation_sessions={"rig0_day0"})

    gate = [e for e in manifest["sessions"] if e["session"] == "rig0_day0"][0]
    assert gate["validation_holdout"] is True
    assert gate["fold"] == data_splits.VALIDATION_HOLDOUT_FOLD
    assert data_splits.validation_holdout_sessions(manifest) == ["rig0_day0"]

    for fold in range(3):
        (train_images, _), (val_images, _) = data_splits.fold_paths(manifest, fold, pool)
        seen = {p.parents[1].name for p in train_images} | {p.parents[1].name for p in val_images}
        assert "rig0_day0" not in seen

    (train_images, _), (validation_images, _) = data_splits.validation_holdout_paths(manifest, pool)
    assert {p.parents[1].name for p in validation_images} == {"rig0_day0"}
    assert "rig0_day0" not in {p.parents[1].name for p in train_images}


def test_manual_session_assignments_update_images_and_validation_holdout(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)
    sessions = [entry["session"] for entry in manifest["sessions"]]
    assignments = {entry["session"]: entry["fold"] for entry in manifest["sessions"]}
    session = next(
        entry["session"]
        for entry in manifest["sessions"]
        if sum(other["fold"] == entry["fold"] for other in manifest["sessions"]) > 1
    )
    assignments[session] = "validation_holdout"

    updated = data_splits.apply_session_assignments(manifest, assignments)

    assert data_splits.validation_holdout_sessions(updated) == [session]
    assert all(
        image["validation_holdout"] == (image["session"] == session) for image in updated["images"]
    )
    with pytest.raises(ValueError, match="cover every editable session"):
        data_splits.apply_session_assignments(manifest, {})
    with pytest.raises(ValueError, match="Every development fold"):
        data_splits.apply_session_assignments(manifest, {name: 0 for name in sessions})


def test_final_run_trains_on_everything_else_and_validates_on_the_gate(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3, final_test_sessions={"rig0_day0"})
    (train_images, _), (gate_images, _) = data_splits.final_paths(manifest, pool)

    assert {p.parents[1].name for p in gate_images} == {"rig0_day0"}
    assert "rig0_day0" not in {p.parents[1].name for p in train_images}
    assert len(train_images) + len(gate_images) == manifest["n_images"]


def test_holdout_survives_a_later_regeneration(pool: Path):
    before = data_splits.build_manifest(pool, n_folds=3, final_test_sessions={"rig0_day0"})
    _write_pair(pool, "newcomer", radius=9, session="rig8_day8")
    after = data_splits.build_manifest(pool, n_folds=3, previous=before)

    assert data_splits.holdout_sessions(after) == ["rig0_day0"]


def test_a_manifest_with_no_holdout_refuses_a_final_run(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)

    with pytest.raises(ValueError, match="sets no final-test session"):
        data_splits.final_paths(manifest, pool)


def test_holdout_naming_a_missing_session_is_rejected(pool: Path):
    with pytest.raises(ValueError, match="no such session"):
        data_splits.build_manifest(pool, n_folds=3, final_test_sessions={"never_existed"})


# --- manifest I/O -----------------------------------------------------------------


def test_manifest_round_trips_and_rejects_an_unknown_schema(pool: Path, tmp_path: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)
    path = tmp_path / "written.json"
    data_splits.write_manifest(path, manifest)

    assert data_splits.load_manifest(path) == manifest

    manifest["schema"] = 99
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="understands"):
        data_splits.load_manifest(path)


def test_session_lookup_covers_every_image(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)
    lookup = data_splits.session_of_key(manifest)

    assert len(lookup) == manifest["n_images"]
    assert lookup["rig0_day0/anything_at_all_0_0"] == "rig0_day0"
