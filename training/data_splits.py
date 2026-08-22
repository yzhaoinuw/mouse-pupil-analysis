# -*- coding: utf-8 -*-
"""Assign labelled images to stratified, recording-grouped cross-validation folds.

Two independent things are going on here, and conflating them is the mistake this
module exists to avoid:

**Grouping** keeps every image from one recording session on the same side of the
train/validation boundary. Without it the reported IoU measures interpolation inside a
setting the model has already seen. The size of that effect was measured on this pool:
copying the mask of an image's nearest neighbour scores 0.652 IoU when the neighbour
comes from the same session and 0.399 when it comes from a different one -- a 0.25 gap
against a seed noise floor of 0.02.

**Stratification** spreads pupil size and lighting evenly across the folds, so each
fold's number measures the same thing. Grouping alone does not give you this: the
first grouped split of this pool left three of five folds with no small pupil at all
and a 3x spread in median diameter, which made fold-to-fold variance mostly a story
about which size regime happened to land where.

Session identity is the labelled-frame directory name; nothing here parses a filename.

Generate or refresh the manifest::

    python training/data_splits.py
    python training/data_splits.py --show          # census only, no write

Then train against the optional validation holdout with::

    python training/run_train.py --split-manifest splits.json

Once a session is in the manifest, its fold is frozen there and adding more data cannot
move it. ``--reassign`` deliberately repacks all sessions, making prior comparisons a
different experiment.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from mouse_pupil_analysis.augmentation import image_background_brightness, mask_equivalent_diameter

SCHEMA_VERSION = 2
# An outer test gate remains distinct from the validation holdout used by the normal
# all-development training run. Both stay out of cross-validation folds.
HOLDOUT_FOLD = -1
VALIDATION_HOLDOUT_FOLD = -2
TINY_MAX_DIAMETER = 15.0

# The labelled pool, one session per directory::
#
#     labeled_frames/<session>/images/<anything>.png
#     labeled_frames/<session>/masks/<anything>.png
#
# The session is the grouping unit, so it is a directory: an image cannot enter the pool
# without one, which makes provenance a consequence of where the file goes rather than a
# convention someone has to remember.
LABELLED_ROOT = "labeled_frames"


@dataclass
class PoolImage:
    """One labelled image/mask pair, located relative to the data root."""

    key: str
    image: str
    mask: str
    diameter: float
    brightness: float


@dataclass
class Session:
    """Every labelled image sharing one recording setting."""

    key: str
    source: str
    images: list[PoolImage]

    @property
    def n_images(self) -> int:
        return len(self.images)

    def n_tiny(self, tiny_max_diameter: float) -> int:
        return sum(image.diameter <= tiny_max_diameter for image in self.images)

    def median_diameter(self) -> float:
        return statistics.median(image.diameter for image in self.images)

    def median_brightness(self) -> float:
        return statistics.median(image.brightness for image in self.images)


def _pool_image(image_path: Path, mask_path: Path, key: str, data_root: Path) -> PoolImage:
    return PoolImage(
        key=key,
        image=image_path.relative_to(data_root).as_posix(),
        mask=mask_path.relative_to(data_root).as_posix(),
        diameter=mask_equivalent_diameter(mask_path),
        brightness=image_background_brightness(image_path, mask_path),
    )


def discover_pool(data_root: Path) -> list[PoolImage]:
    """Read every labelled pair under ``data_root`` as one collection.

    Reads ``labeled_frames/<session>/images``. A session folder states its own session,
    so no filename inference, sidecar, or fallback grouping is needed.

    An image is identified by ``<session>/<filename>`` without the extension. The bare
    filename will not do -- per-recording exports routinely restart their numbering, so
    two sessions can each hold a ``frame_0001.png``, and the whole point of the folders
    is that filenames need not be unique.
    """
    data_root = Path(data_root)
    found: list[PoolImage] = []
    seen: dict[str, str] = {}

    def claim(key: str, where: str) -> None:
        if key in seen:
            raise ValueError(
                f"Image {key!r} appears in both {seen[key]} and {where}. "
                "One key must map to exactly one labelled pair."
            )
        seen[key] = where

    labelled = data_root / LABELLED_ROOT
    if labelled.is_dir():
        for session_dir in sorted(p for p in labelled.iterdir() if p.is_dir()):
            images, masks = session_dir / "images", session_dir / "masks"
            if not images.is_dir():
                raise FileNotFoundError(
                    f"{session_dir.relative_to(data_root).as_posix()} has no images/ "
                    "directory. Each session folder holds images/ and masks/."
                )
            for image_path in sorted(images.rglob("*.png")):
                relative = image_path.relative_to(images).with_suffix("").as_posix()
                key = f"{session_dir.name}/{relative}"
                claim(key, LABELLED_ROOT)
                mask_path = masks / image_path.relative_to(images)
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"No mask for {key}: expected "
                        f"{mask_path.relative_to(data_root).as_posix()}."
                    )
                found.append(_pool_image(image_path, mask_path, key, data_root))

    if not found:
        raise FileNotFoundError(f"No labelled images found under {data_root} in {LABELLED_ROOT}/.")
    return found


def frozen_sessions(previous: dict | None) -> dict[str, str]:
    """Return the ``key -> session`` already recorded in a manifest."""
    if not previous:
        return {}
    return {entry["key"]: entry["session"] for entry in previous["images"]}


def frozen_folds(previous: dict | None) -> dict[str, int]:
    """Return the ``session -> fold`` already recorded in a manifest."""
    if not previous:
        return {}
    return {entry["session"]: entry["fold"] for entry in previous["sessions"]}


def frozen_validation_holdout(previous: dict | None) -> set[str]:
    """Return sessions previously set aside for validation-backed final training."""
    if not previous:
        return set()
    return {entry["session"] for entry in previous["sessions"] if entry.get("validation_holdout")}


def group_sessions(images: list[PoolImage]) -> dict[str, Session]:
    """Group labelled pairs by their required session directory."""
    grouped: dict[str, list[PoolImage]] = defaultdict(list)
    for image in images:
        session, _ = image.key.split("/", 1)
        grouped[session].append(image)
    return {key: Session(key, "folder", value) for key, value in grouped.items()}


def _terciles(values: list[float]) -> tuple[float, float]:
    """Return the two cutpoints splitting values into low/middle/high thirds."""
    if len(values) < 3:
        middle = statistics.median(values)
        return middle, middle
    ordered = sorted(values)
    return (
        ordered[len(ordered) // 3],
        ordered[2 * len(ordered) // 3],
    )


def _band(value: float, cuts: tuple[float, float]) -> int:
    return 0 if value < cuts[0] else (1 if value < cuts[1] else 2)


def stratify(sessions: dict[str, Session]) -> tuple[dict[str, tuple[int, int]], dict]:
    """Band each session by median diameter and median brightness, as terciles.

    The two bands stay separate rather than being crossed into one label. Crossing
    them fragments the axis that matters most: the small-pupil sessions split across
    several combined strata, stop repelling each other, and pile back into the same
    couple of folds. Kept separate, every ``d0`` session repels every other ``d0``
    session whatever its lighting.

    Cutpoints shift as the pool grows, which is harmless: they only steer where *new*
    sessions land, and assignments already recorded are frozen regardless.
    """
    diameters = [session.median_diameter() for session in sessions.values()]
    brightnesses = [session.median_brightness() for session in sessions.values()]
    d_cuts = _terciles(diameters)
    b_cuts = _terciles(brightnesses)

    bands = {
        key: (
            _band(session.median_diameter(), d_cuts),
            _band(session.median_brightness(), b_cuts),
        )
        for key, session in sessions.items()
    }
    cutpoints = {
        "diameter": [round(d_cuts[0], 2), round(d_cuts[1], 2)],
        "brightness": [round(b_cuts[0], 2), round(b_cuts[1], 2)],
    }
    return bands, cutpoints


def stratum_label(bands: tuple[int, int]) -> str:
    """Render a session's bands for the census table."""
    return f"d{bands[0]}b{bands[1]}"


def assign_folds(
    sessions: dict[str, Session],
    bands: dict[str, tuple[int, int]],
    n_folds: int,
    existing: dict[str, int] | None = None,
    holdout: set[str] | None = None,
) -> dict[str, int]:
    """Pack whole sessions into folds, balancing fold size and both condition bands.

    Deterministic, no seed. Sessions carrying a valid fold in ``existing`` keep it, so
    adding data never invalidates an earlier cross-validation result.

    A new session prefers a fold that holds *no* session of its diameter band; failing
    that, the smallest fold. Diameter leads because it is the axis the evaluation
    reports size bins over, and the one an unstratified packing got worst -- three of
    five folds had no small pupil at all. Sessions are placed largest first so the big
    ones set the size balance before the small ones fill in around them.

    Making only the *absence* of a band outrank size is what lets one rule serve two
    regimes. Packing from scratch, folds start empty and coverage dominates, which is
    when it is needed: ordering by size alone there gives a 4.51x spread in median
    diameter and leaves 3 of 5 folds with no small pupil. Once folds each hold a few
    sessions the bands are covered, coverage stops firing, and size leads -- which
    keeps fold sizes even as sessions trickle in one at a time. Simulated over 200
    arrival orders, that is a 1.15x size spread against 1.33x for ranking by band count
    throughout, and a 1.25x worst case against 2.05x, with band coverage identical.

    Sessions named in ``holdout`` are set aside entirely and take no fold.
    """
    holdout = holdout or set()
    unknown = holdout - set(sessions)
    if unknown:
        raise ValueError(f"Holdout names no such session: {sorted(unknown)}")

    assignable = {key: session for key, session in sessions.items() if key not in holdout}
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if n_folds > len(assignable):
        raise ValueError(
            f"Cannot build {n_folds} folds from {len(assignable)} non-holdout sessions; "
            "every fold needs at least one whole session."
        )

    assignment: dict[str, int] = {key: HOLDOUT_FOLD for key in holdout}
    sizes = Counter({fold: 0 for fold in range(n_folds)})
    diameter_counts: dict[int, Counter] = {fold: Counter() for fold in range(n_folds)}
    brightness_counts: dict[int, Counter] = {fold: Counter() for fold in range(n_folds)}

    def record(key: str, fold: int) -> None:
        assignment[key] = fold
        sizes[fold] += assignable[key].n_images
        diameter_counts[fold][bands[key][0]] += 1
        brightness_counts[fold][bands[key][1]] += 1

    for key, fold in (existing or {}).items():
        if key in assignable and isinstance(fold, int) and 0 <= fold < n_folds:
            record(key, fold)

    remaining = [key for key in assignable if key not in assignment]
    for key in sorted(remaining, key=lambda k: (-assignable[k].n_images, k)):
        diameter_band, brightness_band = bands[key]
        fold = min(
            range(n_folds),
            key=lambda f: (
                # Only the *absence* of this diameter band outranks fold size. A fold
                # that has none gets the session; past that, the smallest fold wins.
                1 if diameter_counts[f][diameter_band] else 0,
                sizes[f],
                diameter_counts[f][diameter_band],
                brightness_counts[f][brightness_band],
                f,
            ),
        )
        record(key, fold)

    empty = [fold for fold in range(n_folds) if not sizes[fold]]
    if empty:
        raise ValueError(
            f"Folds {empty} would hold no images. This usually means --folds was raised "
            f"above the {len(assignable)} session(s) the frozen assignment already covers; "
            "pass --reassign to repack every session across the new fold count."
        )
    return assignment


def build_manifest(
    data_root: Path,
    n_folds: int = 5,
    previous: dict | None = None,
    final_test_sessions: set[str] | None = None,
    validation_sessions: set[str] | None = None,
    generated: str | None = None,
    reassign: bool = False,
) -> dict:
    """Return the complete split manifest for ``data_root/labeled_frames``.

    Existing session-to-fold assignments are frozen until ``reassign`` is requested.
    Session identity itself is the required directory name, so no second provenance
    source can silently change it.
    """
    generated = generated or date.today().isoformat()
    if previous and not reassign and previous.get("n_folds") != n_folds:
        raise ValueError(
            f"Existing manifest uses {previous.get('n_folds')} folds, but {n_folds} were "
            "requested. Changing the fold count moves recorded sessions and invalidates "
            "earlier comparisons; pass --reassign to do that deliberately."
        )
    images = discover_pool(data_root)
    sessions = group_sessions(images)
    bands, cutpoints = stratify(sessions)

    final_test_sessions = set(final_test_sessions or ())
    validation_sessions = set(validation_sessions or ())
    if not reassign and previous:
        final_test_sessions |= {
            entry["session"]
            for entry in previous["sessions"]
            if entry.get("holdout") and entry["session"] in sessions
        }
        validation_sessions |= {
            entry["session"]
            for entry in previous["sessions"]
            if entry.get("validation_holdout") and entry["session"] in sessions
        }

    unknown_validation = validation_sessions - set(sessions)
    if unknown_validation:
        raise ValueError(f"Validation session names no such session: {sorted(unknown_validation)}")
    overlap = final_test_sessions & validation_sessions
    if overlap:
        raise ValueError(
            "A session cannot be both a validation session and a final-test session: "
            f"{sorted(overlap)}"
        )

    existing = {} if reassign else frozen_folds(previous)
    assignment = assign_folds(
        sessions, bands, n_folds, existing, final_test_sessions | validation_sessions
    )
    for key in validation_sessions:
        assignment[key] = VALIDATION_HOLDOUT_FOLD

    session_rows = [
        {
            "session": key,
            "fold": assignment[key],
            "holdout": key in final_test_sessions,
            "validation_holdout": key in validation_sessions,
            "source": session.source,
            "stratum": stratum_label(bands[key]),
            "n_images": session.n_images,
            "n_tiny": session.n_tiny(TINY_MAX_DIAMETER),
            "median_diameter": round(session.median_diameter(), 2),
            "median_brightness": round(session.median_brightness(), 2),
        }
        for key, session in sorted(sessions.items(), key=lambda kv: (assignment[kv[0]], kv[0]))
    ]

    return {
        "schema": SCHEMA_VERSION,
        "generated": generated,
        "grouping": "session recorded at intake (one animal, one date, one condition)",
        "stratified_by": ["median_diameter", "median_brightness"],
        "n_folds": n_folds,
        "n_images": len(images),
        "n_sessions": len(sessions),
        "n_holdout_sessions": len(final_test_sessions),
        "n_validation_holdout_sessions": len(validation_sessions),
        "tiny_max_diameter": TINY_MAX_DIAMETER,
        "stratum_cutpoints": cutpoints,
        "sessions": session_rows,
        "images": [
            {
                "key": image.key,
                "image": image.image,
                "mask": image.mask,
                "session": image.key.split("/", 1)[0],
                "fold": assignment[image.key.split("/", 1)[0]],
                "holdout": image.key.split("/", 1)[0] in final_test_sessions,
                "validation_holdout": image.key.split("/", 1)[0] in validation_sessions,
                "diameter": round(image.diameter, 2),
                "brightness": round(image.brightness, 2),
            }
            for image in sorted(images, key=lambda i: i.key)
        ],
    }


def load_manifest(path: Path) -> dict:
    """Read a manifest and reject one this code cannot interpret."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"{path} declares schema {schema!r}, but this code understands "
            f"{SCHEMA_VERSION}. Regenerate it from labeled_frames/."
        )
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def read_previous(path: Path) -> dict | None:
    """Return the manifest already at ``path``, or ``None`` if there is none."""
    return load_manifest(path) if Path(path).exists() else None


def fold_paths(
    manifest: dict,
    fold: int,
    data_root: Path,
) -> tuple[tuple[list[Path], list[Path]], tuple[list[Path], list[Path]]]:
    """Return ``((train_images, train_masks), (val_images, val_masks))`` for one fold.

    The held-out development fold is validation; every other development fold trains.
    Validation-holdout and outer-test-holdout sessions appear in neither.
    """
    n_folds = manifest["n_folds"]
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}); got {fold}.")
    data_root = Path(data_root)

    train: tuple[list[Path], list[Path]] = ([], [])
    validation: tuple[list[Path], list[Path]] = ([], [])
    for entry in sorted(manifest["images"], key=lambda e: e["key"]):
        if entry.get("holdout") or entry.get("validation_holdout"):
            continue
        target = validation if entry["fold"] == fold else train
        target[0].append(data_root / entry["image"])
        target[1].append(data_root / entry["mask"])

    if not validation[0]:
        raise ValueError(f"Fold {fold} holds no images; regenerate the manifest.")
    if not train[0]:
        raise ValueError(f"Fold {fold} leaves no training images; regenerate the manifest.")
    return train, validation


def holdout_paths(manifest: dict, data_root: Path) -> tuple[list[Path], list[Path]]:
    """Return the held-out gate set, which no fold trains or validates on."""
    data_root = Path(data_root)
    images, masks = [], []
    for entry in sorted(manifest["images"], key=lambda e: e["key"]):
        if entry.get("holdout"):
            images.append(data_root / entry["image"])
            masks.append(data_root / entry["mask"])
    return images, masks


def validation_holdout_paths(
    manifest: dict, data_root: Path
) -> tuple[tuple[list[Path], list[Path]], tuple[list[Path], list[Path]]]:
    """Return development training pairs and the optional validation-holdout pairs."""
    data_root = Path(data_root)
    train: tuple[list[Path], list[Path]] = ([], [])
    validation: tuple[list[Path], list[Path]] = ([], [])
    for entry in sorted(manifest["images"], key=lambda e: e["key"]):
        if entry.get("holdout"):
            continue
        target = validation if entry.get("validation_holdout") else train
        target[0].append(data_root / entry["image"])
        target[1].append(data_root / entry["mask"])
    if not train[0]:
        raise ValueError("The manifest leaves no development images for training.")
    return train, validation


def validation_holdout_sessions(manifest: dict) -> list[str]:
    """Return session keys used to validate a normal all-development training run."""
    return [entry["session"] for entry in manifest["sessions"] if entry.get("validation_holdout")]


def apply_session_assignments(manifest: dict, assignments: dict[str, int | str]) -> dict:
    """Apply manual development-fold/validation-holdout assignments to a manifest.

    Outer test-holdout sessions are intentionally not editable here. The caller must
    provide one assignment for every remaining whole session; image rows mirror the
    session decision so downstream consumers never infer it from paths or filenames.
    """
    editable = {entry["session"] for entry in manifest["sessions"] if not entry.get("holdout")}
    supplied = set(assignments)
    if supplied != editable:
        missing, unknown = sorted(editable - supplied), sorted(supplied - editable)
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unknown:
            parts.append(f"unknown {unknown}")
        raise ValueError(
            "Assignments must cover every editable session (" + "; ".join(parts) + ")."
        )

    normalised: dict[str, int] = {}
    for session, target in assignments.items():
        if target == "validation_holdout":
            normalised[session] = VALIDATION_HOLDOUT_FOLD
        elif (
            isinstance(target, int)
            and not isinstance(target, bool)
            and 0 <= target < manifest["n_folds"]
        ):
            normalised[session] = target
        else:
            raise ValueError(
                f"{session!r} must target a development fold in [0, {manifest['n_folds']}) "
                "or 'validation_holdout'."
            )

    empty_folds = [
        fold
        for fold in range(manifest["n_folds"])
        if not any(target == fold for target in normalised.values())
    ]
    if empty_folds:
        raise ValueError(
            f"Every development fold needs at least one session; empty fold(s): {empty_folds}."
        )

    updated = json.loads(json.dumps(manifest))
    by_session = {entry["session"]: entry for entry in updated["sessions"]}
    for session, fold in normalised.items():
        entry = by_session[session]
        entry["fold"] = fold
        entry["validation_holdout"] = fold == VALIDATION_HOLDOUT_FOLD
    for entry in updated["images"]:
        session = entry["session"]
        if session in normalised:
            fold = normalised[session]
            entry["fold"] = fold
            entry["validation_holdout"] = fold == VALIDATION_HOLDOUT_FOLD
    updated["sessions"].sort(key=lambda entry: (entry["fold"], entry["session"]))
    updated["n_validation_holdout_sessions"] = sum(
        entry.get("validation_holdout", False) for entry in updated["sessions"]
    )
    return updated


def final_paths(
    manifest: dict,
    data_root: Path,
) -> tuple[tuple[list[Path], list[Path]], tuple[list[Path], list[Path]]]:
    """Return the development and outer-holdout paths for separate workflows.

    Final refitting consumes only the first pair. The separate one-shot evaluator is
    the only caller allowed to turn the second pair into a Dataset.
    """
    data_root = Path(data_root)
    train: tuple[list[Path], list[Path]] = ([], [])
    gate: tuple[list[Path], list[Path]] = ([], [])
    for entry in sorted(manifest["images"], key=lambda e: e["key"]):
        target = gate if entry.get("holdout") else train
        target[0].append(data_root / entry["image"])
        target[1].append(data_root / entry["mask"])

    if not gate[0]:
        raise ValueError(
            "This manifest sets no final-test session, so there is nothing to gate against. "
            "Regenerate it with --final_test_session SESSION, choosing by condition rather than "
            "by animal."
        )
    return train, gate


def holdout_sessions(manifest: dict) -> list[str]:
    """Return the session keys set aside as the final gate."""
    return [entry["session"] for entry in manifest["sessions"] if entry.get("holdout")]


def fold_sessions(manifest: dict, fold: int) -> list[str]:
    """Return the session keys held out by one fold."""
    return [
        entry["session"]
        for entry in manifest["sessions"]
        if entry["fold"] == fold and not entry.get("holdout")
    ]


def session_of_key(manifest: dict) -> dict[str, str]:
    """Map every image key to its session, for per-session reporting."""
    return {entry["key"]: entry["session"] for entry in manifest["images"]}


def session_of_path(manifest: dict, data_root: Path) -> dict[Path, str]:
    """Map every image's resolved path to its session.

    Callers hold paths, not keys, and a key cannot be recovered from a path without
    knowing which pool folder it came from. Going through the manifest's own recorded
    paths avoids re-deriving it and the chance of deriving it differently.
    """
    data_root = Path(data_root)
    return {
        (data_root / entry["image"]).resolve(): entry["session"] for entry in manifest["images"]
    }


def format_census(manifest: dict) -> str:
    """Render the fold assignment and its stratification balance as a review table."""
    tiny_max = manifest["tiny_max_diameter"]
    lines = [
        f"{'session':<38} {'fold':>5} {'strat':>6} {'imgs':>5} "
        f"{'tiny':>5} {'med_d':>6} {'med_b':>6}",
    ]
    for entry in manifest["sessions"]:
        if entry.get("holdout"):
            fold = "test"
        elif entry.get("validation_holdout"):
            fold = "valid"
        else:
            fold = str(entry["fold"])
        lines.append(
            f"{entry['session'][:36]:<38} {fold:>5} "
            f"{entry['stratum']:>6} {entry['n_images']:>5} {entry['n_tiny']:>5} "
            f"{entry['median_diameter']:>6.1f} {entry['median_brightness']:>6.1f}"
        )

    active = [
        entry
        for entry in manifest["sessions"]
        if not entry.get("holdout") and not entry.get("validation_holdout")
    ]
    total = sum(entry["n_images"] for entry in active)
    lines.append("")
    lines.append(
        f"{'fold':>4} {'sessions':>9} {'images':>7} {'share':>6} {'tiny':>5} "
        f"{'med_d':>6} {'med_b':>6}  strata"
    )
    for fold in range(manifest["n_folds"]):
        held = [entry for entry in active if entry["fold"] == fold]
        n_images = sum(entry["n_images"] for entry in held)
        tiny = sum(entry["n_tiny"] for entry in held)
        med_d = statistics.median([entry["median_diameter"] for entry in held]) if held else 0.0
        med_b = statistics.median([entry["median_brightness"] for entry in held]) if held else 0.0
        strata = sorted(entry["stratum"] for entry in held)
        lines.append(
            f"{fold:>4} {len(held):>9} {n_images:>7} {100 * n_images / total:>5.0f}% {tiny:>5} "
            f"{med_d:>6.1f} {med_b:>6.1f}  {' '.join(strata)}"
        )

    holdout = [entry for entry in manifest["sessions"] if entry.get("holdout")]
    if holdout:
        images = sum(entry["n_images"] for entry in holdout)
        lines.append("")
        lines.append(
            f"test holdout: {len(holdout)} session(s), {images} image(s), in no fold. "
            "Trained on never, validated on never -- this is the final gate."
        )

    validation_holdout = [
        entry for entry in manifest["sessions"] if entry.get("validation_holdout")
    ]
    if validation_holdout:
        images = sum(entry["n_images"] for entry in validation_holdout)
        lines.append("")
        lines.append(
            f"validation holdout: {len(validation_holdout)} session(s), {images} image(s), "
            "excluded from CV and used by the normal all-development training run."
        )

    tiny_total = sum(entry["n_tiny"] for entry in active)
    empty = [
        fold
        for fold in range(manifest["n_folds"])
        if not any(entry["n_tiny"] for entry in active if entry["fold"] == fold)
    ]
    if tiny_total and empty:
        lines.append("")
        lines.append(
            f"NOTE: folds {empty} hold no masks at or below {tiny_max:g} model pixels, so their "
            "tiny-bin IoU is undefined. Stratification spreads what exists; it cannot "
            "manufacture small pupils that were never labelled."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--labeled_frames_dir",
        type=Path,
        default=Path.cwd() / LABELLED_ROOT,
        help="Folder containing one <session>/images and <session>/masks pair per recording "
        "(default: ./labeled_frames). The manifest is always its parent/splits.json.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        help="Number of cross-validation folds (default: whatever the existing manifest "
        "uses, else 5). Changing this requires --reassign.",
    )
    parser.add_argument(
        "--final_test_session",
        action="append",
        default=[],
        metavar="SESSION",
        help="Reserve a session for the final test only: it is in no fold and never used "
        "for training or model choices. Repeatable; choose by condition, not animal.",
    )
    parser.add_argument(
        "--validation_session",
        action="append",
        default=[],
        metavar="SESSION",
        help="Reserve a session for validation-backed development training. It is outside "
        "CV folds but used by run_train.py when no --fold is supplied. Repeatable.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the census without writing the manifest.",
    )
    parser.add_argument(
        "--reassign",
        action="store_true",
        help="Repack every session from scratch, discarding frozen assignments. "
        "This makes new runs incomparable to previously recorded ones.",
    )
    args = parser.parse_args(argv)
    labeled_frames_dir = Path(args.labeled_frames_dir).resolve()
    data_root = labeled_frames_dir.parent
    if labeled_frames_dir.name != LABELLED_ROOT:
        raise ValueError(
            f"--labeled_frames_dir must name a {LABELLED_ROOT!r} folder; got "
            f"{labeled_frames_dir}."
        )
    manifest_path = data_root / "splits.json"
    previous = None if args.reassign else read_previous(manifest_path)
    if args.folds is None:
        # A manifest already records its fold count; silently substituting a default
        # here would repack against a different one and leave folds empty.
        args.folds = previous["n_folds"] if previous else 5
    manifest = build_manifest(
        data_root=data_root,
        n_folds=args.folds,
        previous=previous,
        final_test_sessions=set(args.final_test_session),
        validation_sessions=set(args.validation_session),
        reassign=args.reassign,
    )
    print()
    print(format_census(manifest))
    active = (
        manifest["n_sessions"]
        - manifest["n_holdout_sessions"]
        - manifest.get("n_validation_holdout_sessions", 0)
    )
    print(
        f"\n{manifest['n_images']} images, {manifest['n_sessions']} sessions "
        f"({active} in CV folds, {manifest.get('n_validation_holdout_sessions', 0)} validation "
        f"holdout, {manifest['n_holdout_sessions']} test holdout), "
        f"{manifest['n_folds']} folds"
    )

    if args.show:
        print("\n--show given; manifest not written.")
        return 0

    kept = len(frozen_sessions(previous).keys() & {e["key"] for e in manifest["images"]})
    write_manifest(manifest_path, manifest)
    print(f"Wrote {manifest_path} ({kept} image assignment(s) carried over unchanged).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
