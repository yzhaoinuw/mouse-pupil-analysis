"""Coverage for stratified, recording-grouped splitting.

Two failures are guarded here and both are silent. A split that looks grouped but
still puts two frames from one sitting on opposite sides of the boundary reports an
interpolation number as if it were a generalisation number -- worth 0.25 IoU on this
pool against a 0.02 noise floor. And a manifest that quietly re-derives its own
grouping can move an image between folds when a provenance source changes, which
invalidates every previously recorded run without raising anything.
"""

import importlib.util
import json
import shutil
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
provenance = _load("provenance")


def _write_pair(
    root: Path,
    stem: str,
    radius: int,
    split: str | None = None,
    session: str | None = None,
    grey: int = 100,
    labelme_session: str | None = None,
) -> Path:
    """Write one image/mask pair into a session folder, or into a legacy flat folder.

    With ``session``, the pair lands in ``labeled_frames/<session>/images|masks``, which
    is the layout everything uses now. With ``split`` instead, it lands in the historical
    flat ``images_<split>`` / ``masks_<split>``, which the reader still accepts.

    ``grey`` sets the background level the brightness feature reads, and ``radius``
    sets the mask's equivalent diameter, so a test can place a pair in a chosen
    stratum on purpose.
    """
    if session is not None:
        image_dir = root / "labeled_frames" / session / "images"
        mask_dir = root / "labeled_frames" / session / "masks"
    else:
        image_dir = root / f"images_{split or 'train'}"
        mask_dir = root / f"masks_{split or 'train'}"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.full((60, 60), grey, dtype=np.uint8)).save(image_dir / f"{stem}.png")

    mask = np.zeros((60, 60), dtype=np.uint8)
    grid_y, grid_x = np.ogrid[:60, :60]
    mask[(grid_y - 30) ** 2 + (grid_x - 30) ** 2 <= radius**2] = 255
    Image.fromarray(mask).save(mask_dir / f"{stem}.png")

    if labelme_session is not None:
        (image_dir / f"{stem}.json").write_text(
            json.dumps({"flags": {"session": labelme_session}}), encoding="utf-8"
        )
    return image_dir / f"{stem}.png"


def _sessions(root: Path, sidecar=None, batch_name="unknown_batch"):
    """Group a pool without packing folds, so provenance can be tested on its own.

    Fold packing needs at least as many sessions as folds, which gets in the way of
    checking cases that deliberately collapse to a single group.
    """
    images, located = data_splits.discover_pool(root)
    resolved = provenance.resolve(located, sidecar=sidecar, batch_name=batch_name)
    return data_splits.group_sessions(images, resolved)


@pytest.fixture
def pool(tmp_path: Path) -> Path:
    """Six sessions recorded as intake folders, one spanning two pool directories.

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
    # A sixth session, written the old flat way, to pin that a legacy checkout still reads.
    _write_pair(tmp_path, "legacy_a", radius=10, split="validation", grey=120)
    _write_pair(tmp_path, "legacy_b", radius=11, split="validation", grey=120)
    return tmp_path


# --- provenance sources ----------------------------------------------------------


def test_intake_folder_names_the_session_whatever_the_file_is_called(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)

    sessions = {entry["session"]: entry["source"] for entry in manifest["sessions"]}
    assert "rig0_day0" in sessions
    # Five session folders speak for themselves; the two legacy flat files cannot.
    assert sum(source == "folder" for source in sessions.values()) == 5
    # Nothing about the filenames themselves carries the grouping.
    assert manifest["n_sessions"] == 6


def test_sidecar_outranks_folder_and_labelme(tmp_path: Path):
    _write_pair(tmp_path, "frame1", radius=8, session="folder_says", labelme_session="flag_says")

    sessions = _sessions(tmp_path, sidecar={"folder_says/frame1": "sidecar_says"})
    assert set(sessions) == {"sidecar_says"}
    assert sessions["sidecar_says"].source == "sidecar"


def test_labelme_flag_outranks_the_folder(tmp_path: Path):
    _write_pair(tmp_path, "frame1", radius=8, session="folder_says", labelme_session="flag_says")

    sessions = _sessions(tmp_path)
    assert set(sessions) == {"flag_says"}
    assert sessions["flag_says"].source == "labelme"


def test_images_with_no_recorded_provenance_become_one_group(tmp_path: Path):
    for index in range(6):
        _write_pair(tmp_path, f"mystery_{index}", radius=6 + index)

    # Over-merging is the safe failure: one group cannot straddle the boundary.
    sessions = _sessions(tmp_path, batch_name="june_dump")
    assert set(sessions) == {"june_dump"}
    assert sessions["june_dump"].source == "batch"
    assert sessions["june_dump"].n_images == 6


def test_a_pool_with_no_provenance_at_all_cannot_be_folded(tmp_path: Path):
    for index in range(6):
        _write_pair(tmp_path, f"mystery_{index}", radius=6 + index)

    # The cost of the safe fallback, made loud: one group is not five folds. This is
    # the pressure to record the session at intake rather than a pipeline failure.
    with pytest.raises(ValueError, match="from 1 non-holdout sessions"):
        data_splits.build_manifest(tmp_path, n_folds=2, batch_name="june_dump")


def test_two_unknown_batches_merge_when_given_the_same_name(tmp_path: Path):
    _write_pair(tmp_path, "first", radius=8)
    _write_pair(tmp_path, "second", radius=9)

    sessions = _sessions(tmp_path, batch_name="same_rig")
    assert set(sessions) == {"same_rig"}
    assert sessions["same_rig"].n_images == 2


def test_malformed_labelme_json_falls_through_rather_than_raising(tmp_path: Path):
    image = _write_pair(tmp_path, "frame1", radius=8, session="folder_says")
    image.with_suffix(".json").write_text("{not json at all", encoding="utf-8")

    assert set(_sessions(tmp_path)) == {"folder_says"}


def test_sidecar_round_trips_through_csv_and_json(tmp_path: Path):
    mapping = {"a": "session_one", "b": "session_two"}
    for name in ("provenance.csv", "provenance.json"):
        path = tmp_path / name
        provenance.write_sidecar(path, mapping)
        assert provenance.load_sidecar(path) == mapping


def test_sidecar_with_a_blank_session_is_rejected(tmp_path: Path):
    path = tmp_path / "provenance.csv"
    path.write_text("key,session\nframe1,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="leaves the session blank"):
        provenance.load_sidecar(path)


# --- grouping and the freeze guarantee --------------------------------------------


def test_no_session_is_split_across_folds(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3, batch_name="legacy_batch")

    fold_of_session = {entry["session"]: entry["fold"] for entry in manifest["sessions"]}
    for entry in manifest["images"]:
        assert entry["fold"] == fold_of_session[entry["session"]]


def test_a_legacy_flat_folder_is_still_read(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3, batch_name="legacy_batch")

    legacy = [e for e in manifest["images"] if e["key"].startswith("legacy_")]
    assert {Path(e["image"]).parts[0] for e in legacy} == {"images_validation"}
    # Flat files state no session, so they collapse into one safe group.
    assert {e["session"] for e in legacy} == {"legacy_batch"}
    assert len({e["fold"] for e in legacy}) == 1


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


def test_a_changed_provenance_source_raises_instead_of_repacking(pool: Path):
    before = data_splits.build_manifest(pool, n_folds=3)

    # Someone edits the sidecar and reassigns an image that is already recorded.
    key = before["images"][0]["key"]
    with pytest.raises(ValueError, match="already recorded under a different session"):
        data_splits.build_manifest(
            pool, n_folds=3, previous=before, sidecar={key: "somewhere_else"}
        )


def test_reassign_repacks_deliberately(pool: Path):
    before = data_splits.build_manifest(pool, n_folds=3)
    key = before["images"][0]["key"]

    after = data_splits.build_manifest(
        pool, n_folds=3, previous=before, sidecar={key: "somewhere_else"}, reassign=True
    )
    assert "somewhere_else" in {entry["session"] for entry in after["sessions"]}


def test_changing_fold_count_requires_explicit_reassignment(pool: Path):
    before = data_splits.build_manifest(pool, n_folds=3)

    with pytest.raises(ValueError, match="pass --reassign"):
        data_splits.build_manifest(pool, n_folds=2, previous=before)

    after = data_splits.build_manifest(pool, n_folds=2, previous=before, reassign=True)
    assert after["n_folds"] == 2


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


def test_the_session_layout_takes_precedence_over_a_retained_legacy_backup(tmp_path: Path):
    _write_pair(tmp_path, "frame1", radius=8, session="rig1_day1")
    _write_pair(tmp_path, "frame2", radius=12, session="rig2_day2")
    # The legacy reader would produce the same key from a flat nested path.
    legacy = tmp_path / "images_train" / "legacy_backup"
    legacy.mkdir(parents=True)
    (tmp_path / "masks_train" / "legacy_backup").mkdir(parents=True)
    shutil.copy2(
        tmp_path / "labeled_frames/rig1_day1/images/frame1.png",
        legacy / "old_verbose_name.png",
    )
    shutil.copy2(
        tmp_path / "labeled_frames/rig1_day1/masks/frame1.png",
        tmp_path / "masks_train/legacy_backup/old_verbose_name.png",
    )

    manifest = data_splits.build_manifest(tmp_path, n_folds=2)

    assert manifest["n_images"] == 2
    assert all(entry["image"].startswith("labeled_frames/") for entry in manifest["images"])


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
    manifest = data_splits.build_manifest(pool, n_folds=3, holdout={"rig0_day0"})

    gate = [e for e in manifest["sessions"] if e["session"] == "rig0_day0"][0]
    assert gate["holdout"] is True
    assert gate["fold"] == data_splits.HOLDOUT_FOLD

    for fold in range(3):
        (train_images, _), (val_images, _) = data_splits.fold_paths(manifest, fold, pool)
        seen = {p.parents[1].name for p in train_images} | {p.parents[1].name for p in val_images}
        assert "rig0_day0" not in seen


def test_validation_holdout_is_excluded_from_cv_and_loaded_for_normal_training(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3, validation_holdout={"rig0_day0"})

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
    manifest = data_splits.build_manifest(pool, n_folds=3, holdout={"rig0_day0"})
    (train_images, _), (gate_images, _) = data_splits.final_paths(manifest, pool)

    assert {p.parents[1].name for p in gate_images} == {"rig0_day0"}
    assert "rig0_day0" not in {p.parents[1].name for p in train_images}
    assert len(train_images) + len(gate_images) == manifest["n_images"]


def test_holdout_survives_a_later_regeneration(pool: Path):
    before = data_splits.build_manifest(pool, n_folds=3, holdout={"rig0_day0"})
    _write_pair(pool, "newcomer", radius=9, session="rig8_day8")
    after = data_splits.build_manifest(pool, n_folds=3, previous=before)

    assert data_splits.holdout_sessions(after) == ["rig0_day0"]


def test_a_manifest_with_no_holdout_refuses_a_final_run(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)

    with pytest.raises(ValueError, match="sets no holdout"):
        data_splits.final_paths(manifest, pool)


def test_holdout_naming_a_missing_session_is_rejected(pool: Path):
    with pytest.raises(ValueError, match="no such session"):
        data_splits.build_manifest(pool, n_folds=3, holdout={"never_existed"})


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


def test_a_schema_one_manifest_points_at_the_migration(pool: Path, tmp_path: Path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema": 1, "images": [], "sessions": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="--migrate-from"):
        data_splits.load_manifest(path)


def test_session_lookup_covers_every_image(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)
    lookup = data_splits.session_of_key(manifest)

    assert len(lookup) == manifest["n_images"]
    assert lookup["rig0_day0/anything_at_all_0_0"] == "rig0_day0"


def test_census_flags_a_batch_fallback_group(tmp_path: Path):
    for index in range(4):
        _write_pair(tmp_path, f"mystery_{index}", radius=6 + index)
    _write_pair(tmp_path, "known_a", radius=9, session="rig1_day1")
    _write_pair(tmp_path, "known_b", radius=14, session="rig2_day2")
    manifest = data_splits.build_manifest(tmp_path, n_folds=2, batch_name="june_dump")

    census = data_splits.format_census(manifest)
    assert "no recorded provenance" in census
    assert "june_dump" in census


# --- materialized fold folders ----------------------------------------------------


def test_materialize_writes_every_image_into_its_own_fold_folder(pool: Path, tmp_path: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)
    out = tmp_path / "folds"

    counts = data_splits.materialize(manifest, pool, out)

    assert sorted(counts) == ["cv1", "cv2", "cv3"]
    for entry in manifest["images"]:
        assert (out / f"cv{entry['fold'] + 1}" / "images" / f"{entry['key']}.png").is_file()
        assert (out / f"cv{entry['fold'] + 1}" / "masks" / f"{entry['key']}.png").is_file()

    # Folds partition the pool: one copy of each image, none duplicated.
    written = list(out.glob("cv*/images/**/*.png"))
    assert len(written) == manifest["n_images"]


def test_materialize_puts_holdout_in_its_own_folder(pool: Path, tmp_path: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3, holdout={"rig0_day0"})
    out = tmp_path / "folds"

    counts = data_splits.materialize(manifest, pool, out)

    assert "holdout" in counts
    held = list((out / "holdout" / "images").rglob("*.png"))
    assert held and all("rig0_day0" in str(p) for p in held)  # keys carry the session
    # The gate must not also appear in a fold.
    assert not any("rig0_day0" in str(p) for p in out.glob("cv*/images/**/*.png"))


def test_materialize_is_regenerated_not_merged(pool: Path, tmp_path: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)
    out = tmp_path / "folds"
    data_splits.materialize(manifest, pool, out)

    stale = out / "cv1" / "images" / "left_over_from_an_older_split.png"
    stale.write_bytes(b"stale")
    data_splits.materialize(manifest, pool, out)

    # These folders are derived output, so a rebuild must not preserve anything.
    assert not stale.exists()


def test_materialize_refuses_to_replace_the_data_root(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)

    with pytest.raises(ValueError, match="data root itself"):
        data_splits.materialize(manifest, pool, pool)


def test_materialize_can_write_a_new_directory_outside_the_data_root(pool: Path, tmp_path: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)
    outside = tmp_path.parent / f"{tmp_path.name}_outside"

    counts = data_splits.materialize(manifest, pool, outside)

    assert sum(counts.values()) == manifest["n_images"]
    assert (outside / data_splits.MATERIALIZED_MARKER).is_file()


def test_materialize_refuses_to_replace_a_user_directory(pool: Path):
    manifest = data_splits.build_manifest(pool, n_folds=3)
    out = pool / "not-generated"
    out.mkdir()
    important = out / "keep-me.txt"
    important.write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="non-generated"):
        data_splits.materialize(manifest, pool, out)

    assert important.read_text(encoding="utf-8") == "user data"
