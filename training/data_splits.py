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

Session identity is *recorded*, never inferred -- see :mod:`training.provenance` for
why and for the four sources it comes from. Nothing here parses a filename.

Generate or refresh the manifest::

    python training/data_splits.py --data-root . --folds 5 --out splits.json
    python training/data_splits.py --data-root . --show          # census only, no write

Then train one fold with::

    python training/run_train.py --split-manifest splits.json --fold 0

Once an image is in the manifest, its session and fold are frozen there and adding
more data cannot move it. If a provenance source later disagrees with what was
recorded, that is an error rather than a silent repack, because a repack invalidates
comparison against every previously recorded run. ``--reassign`` repacks deliberately.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from mouse_pupil_analysis.augmentation import image_background_brightness, mask_equivalent_diameter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as provenance_module  # noqa: E402

SCHEMA_VERSION = 2
HOLDOUT_FOLD = -1

# The labelled pool, one session per directory::
#
#     labeled_data/<session>/images/<anything>.png
#     labeled_data/<session>/masks/<anything>.png
#
# The session is the grouping unit, so it is a directory: an image cannot enter the pool
# without one, which makes provenance a consequence of where the file goes rather than a
# convention someone has to remember. The two historical flat pairs are still read, so an
# older local checkout keeps working.
LABELLED_ROOT = "labeled_data"
LEGACY_POOL = (
    ("images_train", "masks_train"),
    ("images_validation", "masks_validation"),
)
DEFAULT_POOL = LEGACY_POOL


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


def _mask_for(image_path: Path, image_root: Path, mask_root: Path) -> Path:
    """Find the mask for an image, mirroring intake subfolders then falling back flat.

    ``labelme_json2png.py`` writes each mask as ``<image stem>.png``. When images are
    dropped in per-recording subfolders the masks usually mirror that structure, but a
    flat mask folder is still common, so both layouts are accepted.
    """
    relative = image_path.relative_to(image_root)
    mirrored = mask_root / relative
    if mirrored.is_file():
        return mirrored
    flat = mask_root / image_path.name
    if flat.is_file():
        return flat
    raise FileNotFoundError(
        f"No mask for {relative.as_posix()}: looked for {mirrored} and {flat}. "
        "Every labelled image needs a mask sharing its stem."
    )


def _pool_image(image_path: Path, mask_path: Path, key: str, data_root: Path) -> PoolImage:
    return PoolImage(
        key=key,
        image=image_path.relative_to(data_root).as_posix(),
        mask=mask_path.relative_to(data_root).as_posix(),
        diameter=mask_equivalent_diameter(mask_path),
        brightness=image_background_brightness(image_path, mask_path),
    )


def discover_pool(
    data_root: Path,
    pool: tuple[tuple[str, str], ...] = DEFAULT_POOL,
    labelled_root: str = LABELLED_ROOT,
) -> tuple[list[PoolImage], dict[str, tuple[Path, str | None]]]:
    """Read every labelled pair under ``data_root`` as one collection.

    Reads ``labeled_data/<session>/images`` plus, for an older checkout, any flat
    ``pool`` pair still present. Returns the pool alongside the
    ``key -> (image path, session or None)`` map provenance resolution needs; a session
    folder states its own session, so those entries arrive already resolved.

    An image is identified by ``<session>/<filename>`` without the extension. The bare
    filename will not do -- per-recording exports routinely restart their numbering, so
    two sessions can each hold a ``frame_0001.png``, and the whole point of the folders
    is that filenames need not be unique.
    """
    data_root = Path(data_root)
    found: list[PoolImage] = []
    located: dict[str, tuple[Path, str | None]] = {}
    seen: dict[str, str] = {}

    def claim(key: str, where: str) -> None:
        if key in seen:
            raise ValueError(
                f"Image {key!r} appears in both {seen[key]} and {where}. "
                "One key must map to exactly one labelled pair."
            )
        seen[key] = where

    labelled = data_root / labelled_root
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
                claim(key, labelled_root)
                mask_path = masks / image_path.relative_to(images)
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"No mask for {key}: expected "
                        f"{mask_path.relative_to(data_root).as_posix()}."
                    )
                found.append(_pool_image(image_path, mask_path, key, data_root))
                located[key] = (image_path, session_dir.name)

    for image_dir, mask_dir in pool:
        image_root, mask_root = data_root / image_dir, data_root / mask_dir
        if not image_root.is_dir():
            continue
        for image_path in sorted(image_root.rglob("*.png")):
            key = image_path.relative_to(image_root).with_suffix("").as_posix()
            claim(key, image_dir)
            mask_path = _mask_for(image_path, image_root, mask_root)
            found.append(_pool_image(image_path, mask_path, key, data_root))
            located[key] = (image_path, None)

    if not found:
        raise FileNotFoundError(
            f"No labelled images found under {data_root} in {labelled_root}/ "
            f"or {[d for d, _ in pool]}."
        )
    return found, located


def existing_pool(
    data_root: Path,
    pool: tuple[tuple[str, str], ...] = DEFAULT_POOL,
) -> tuple[tuple[str, str], ...]:
    """Return only the pool folder pairs that are actually present."""
    data_root = Path(data_root)
    return tuple(entry for entry in pool if (data_root / entry[0]).is_dir())


def frozen_sessions(previous: dict | None) -> dict[str, str]:
    """Return the ``key -> session`` already recorded in a manifest."""
    if not previous:
        return {}
    return {entry["key"]: entry["session"] for entry in previous["images"]}


def frozen_source(previous: dict | None) -> dict[str, str]:
    """Return the ``session -> source`` already recorded in a manifest."""
    if not previous:
        return {}
    return {entry["session"]: entry["source"] for entry in previous["sessions"]}


def frozen_folds(previous: dict | None) -> dict[str, int]:
    """Return the ``session -> fold`` already recorded in a manifest."""
    if not previous:
        return {}
    return {entry["session"]: entry["fold"] for entry in previous["sessions"]}


def group_sessions(
    images: list[PoolImage],
    assignment: dict[str, provenance_module.Provenance],
) -> dict[str, Session]:
    """Collapse the flat pool into sessions using the resolved provenance."""
    # A session resolved from several sources reports the least explicit one, so the
    # census never overstates how well its provenance is recorded. An unrecognised
    # source sorts last for the same reason.
    rank = {"frozen": 0, "sidecar": 1, "labelme": 2, "folder": 3, "batch": 4}

    grouped: dict[str, list[PoolImage]] = defaultdict(list)
    source: dict[str, str] = {}
    for image in images:
        resolved = assignment[image.key]
        grouped[resolved.session].append(image)
        current = source.get(resolved.session)
        if current is None or rank.get(resolved.source, len(rank)) > rank.get(current, len(rank)):
            source[resolved.session] = resolved.source
    return {key: Session(key, source[key], value) for key, value in grouped.items()}


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
    pool: tuple[tuple[str, str], ...] = DEFAULT_POOL,
    tiny_max_diameter: float = 15.0,
    previous: dict | None = None,
    sidecar: dict[str, str] | None = None,
    batch_name: str | None = None,
    holdout: set[str] | None = None,
    generated: str | None = None,
    reassign: bool = False,
) -> dict:
    """Return the complete split manifest for the labelled pool under ``data_root``.

    Provenance already recorded in ``previous`` wins over every live source, so a
    manifest is self-stabilising: once an image is in it, no later change to a
    sidecar, a labelme flag, or a folder layout can move that image between folds.
    A live source that *disagrees* with the record raises rather than repacking.
    """
    generated = generated or date.today().isoformat()
    images, located = discover_pool(data_root, pool)

    resolved = provenance_module.resolve(
        located,
        sidecar=sidecar,
        batch_name=batch_name or f"unknown_batch_{generated}",
    )

    already = {} if reassign else frozen_sessions(previous)
    # The batch fallback is not a claim about provenance, it is the absence of one, so
    # it never contradicts a recorded session -- otherwise deleting a sidecar row would
    # read as a deliberate reassignment.
    conflicts = [
        (key, already[key], resolved[key].session)
        for key in sorted(resolved)
        if key in already
        and resolved[key].source != "batch"
        and already[key] != resolved[key].session
    ]
    if conflicts:
        detail = "; ".join(f"{key}: {was!r} -> {now!r}" for key, was, now in conflicts[:5])
        raise ValueError(
            f"{len(conflicts)} image(s) already recorded under a different session: {detail}. "
            "Fold assignments are frozen once written, so this would silently invalidate "
            "every previously recorded run. Fix the provenance source, or pass --reassign "
            "to repack everything deliberately."
        )
    # The recorded session wins, but the *source* stays whatever the live sources say.
    # Stamping "frozen" over it made the manifest differ from itself on regeneration,
    # and worse, silently retired the census warning about sessions that have no
    # recorded provenance at all -- they would warn once and never again.
    carried = frozen_source(previous)
    for key, session in already.items():
        if key in resolved:
            source = resolved[key].source
            if source == "batch":
                source = carried.get(session, source)
            resolved[key] = provenance_module.Provenance(session, source)

    sessions = group_sessions(images, resolved)
    bands, cutpoints = stratify(sessions)

    holdout = set(holdout or ())
    if not reassign and previous:
        holdout |= {
            entry["session"]
            for entry in previous["sessions"]
            if entry.get("holdout") and entry["session"] in sessions
        }

    existing = {} if reassign else frozen_folds(previous)
    assignment = assign_folds(sessions, bands, n_folds, existing, holdout)

    session_rows = [
        {
            "session": key,
            "fold": assignment[key],
            "holdout": key in holdout,
            "source": session.source,
            "stratum": stratum_label(bands[key]),
            "n_images": session.n_images,
            "n_tiny": session.n_tiny(tiny_max_diameter),
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
        "n_holdout_sessions": len(holdout),
        "pool": [list(entry) for entry in pool],
        "tiny_max_diameter": tiny_max_diameter,
        "stratum_cutpoints": cutpoints,
        "sessions": session_rows,
        "images": [
            {
                "key": image.key,
                "image": image.image,
                "mask": image.mask,
                "session": resolved[image.key].session,
                "fold": assignment[resolved[image.key].session],
                "holdout": resolved[image.key].session in holdout,
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
        hint = (
            " Schema 1 derived sessions from filenames; migrate it with "
            "`python training/data_splits.py --migrate-from <old manifest>`."
            if schema == 1
            else ""
        )
        raise ValueError(
            f"{path} declares schema {schema!r}, but this code understands "
            f"{SCHEMA_VERSION}.{hint}"
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

    The held-out fold is validation; every other fold trains. Holdout sessions appear
    in neither: they are the final gate and must stay unseen by every fold, or the
    gate reports on data the model was trained on.
    """
    n_folds = manifest["n_folds"]
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}); got {fold}.")
    data_root = Path(data_root)

    train: tuple[list[Path], list[Path]] = ([], [])
    validation: tuple[list[Path], list[Path]] = ([], [])
    for entry in sorted(manifest["images"], key=lambda e: e["key"]):
        if entry.get("holdout"):
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


def final_paths(
    manifest: dict,
    data_root: Path,
) -> tuple[tuple[list[Path], list[Path]], tuple[list[Path], list[Path]]]:
    """Return ``((train, holdout))`` for the release-candidate run.

    Cross-validation compares configurations; this trains the winning one on every
    image the folds were allowed to see and scores it against the holdout, which no
    fold and no earlier run ever touched. That is the only number in the project
    measured on data the training procedure has never been tuned against.
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
            "This manifest sets no holdout, so there is nothing to gate against. "
            "Regenerate it with --holdout SESSION, choosing by condition rather than "
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


def materialize(
    manifest: dict,
    data_root: Path,
    out_dir: Path,
    copy: bool = True,
) -> dict[str, int]:
    """Write the fold assignment to disk as ``cv1/``, ``cv2/``, ... folders.

    Derived output, one way. The manifest is the record -- it is committed, while the
    image folders are not, so a fresh clone has the manifest and nothing else. These
    folders are rebuilt from it and are never read back: editing them changes nothing,
    and the next ``--materialize`` overwrites them.

    Folds partition the pool, so every image is written exactly once and the whole tree
    costs one copy of the dataset. Folders are numbered from 1 because that is how folds
    get talked about; ``cv1`` is manifest fold 0.
    """
    data_root, out_dir = Path(data_root), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    written: dict[str, int] = {}
    for entry in sorted(manifest["images"], key=lambda e: e["key"]):
        name = "holdout" if entry.get("holdout") else f"cv{entry['fold'] + 1}"
        for kind, source in (("images", entry["image"]), ("masks", entry["mask"])):
            destination = out_dir / name / kind / f"{entry['key']}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if copy:
                shutil.copy2(data_root / source, destination)
            else:
                destination.symlink_to(
                    Path(os.path.relpath(data_root / source, destination.parent))
                )
        written[name] = written.get(name, 0) + 1

    (out_dir / "README.txt").write_text(
        "Generated by training/data_splits.py --materialize. Do not edit.\n\n"
        "Rebuilt from the manifest, which is the record. cvN holds fold N-1:\n"
        + "".join(
            f"  cv{fold + 1}/  fold {fold}  "
            f"{sum(1 for e in manifest['images'] if e['fold'] == fold and not e.get('holdout'))}"
            " images\n"
            for fold in range(manifest["n_folds"])
        )
        + (
            f"  holdout/ in no fold  "
            f"{sum(1 for e in manifest['images'] if e.get('holdout'))} images\n"
            if any(e.get("holdout") for e in manifest["images"])
            else ""
        )
        + "\nTrain on one fold with:\n"
        "  python training/run_train.py --split-manifest splits.json --fold 0\n",
        encoding="utf-8",
    )
    return written


def format_census(manifest: dict) -> str:
    """Render the fold assignment and its stratification balance as a review table."""
    tiny_max = manifest["tiny_max_diameter"]
    lines = [
        f"{'session':<38} {'src':<8} {'fold':>4} {'strat':>6} {'imgs':>5} "
        f"{'tiny':>5} {'med_d':>6} {'med_b':>6}",
    ]
    for entry in manifest["sessions"]:
        fold = "hold" if entry.get("holdout") else str(entry["fold"])
        lines.append(
            f"{entry['session'][:36]:<38} {entry['source']:<8} {fold:>4} "
            f"{entry['stratum']:>6} {entry['n_images']:>5} {entry['n_tiny']:>5} "
            f"{entry['median_diameter']:>6.1f} {entry['median_brightness']:>6.1f}"
        )

    active = [entry for entry in manifest["sessions"] if not entry.get("holdout")]
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
            f"holdout: {len(holdout)} session(s), {images} image(s), in no fold. "
            "Trained on never, validated on never -- this is the final gate."
        )

    batch = [entry for entry in manifest["sessions"] if entry["source"] == "batch"]
    if batch:
        lines.append("")
        lines.append(
            f"NOTE: {len(batch)} session(s) had no recorded provenance and were merged into "
            "one group each by the batch fallback:"
        )
        for entry in batch:
            lines.append(f"  {entry['session']} ({entry['n_images']} images)")
        lines.append(
            "  Safe but lumpy -- the whole batch lands in one fold. Record the session at "
            "intake instead; see training/data_collection.md."
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


def _migration_sidecar(old_manifest: Path) -> dict[str, str]:
    """Read the key-to-session mapping out of a schema-1 manifest.

    This is the only place the old filename-derived grouping is ever used. It runs
    once, to carry the sessions that were already worked out into the recorded model,
    after which the regex that produced them is gone.
    """
    raw = json.loads(Path(old_manifest).read_text(encoding="utf-8"))
    if raw.get("schema") != 1:
        raise ValueError(f"{old_manifest} is schema {raw.get('schema')!r}, expected 1.")
    # Schema 1 predates intake subfolders, so its stems are already pool-relative keys.
    return {entry["stem"]: entry["session"] for entry in raw["images"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--out",
        type=Path,
        help="Manifest to write and to read frozen assignments from "
        "(default: <data-root>/splits.json). A manifest belongs to the pool it "
        "describes, so this follows --data-root rather than the working directory.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        help="Number of cross-validation folds (default: whatever the existing manifest "
        "uses, else 5). Changing this requires --reassign.",
    )
    parser.add_argument("--tiny-max-diameter", type=float, default=15.0)
    parser.add_argument(
        "--sidecar",
        type=Path,
        help="Explicit key-to-session mapping (.csv or .json). Defaults to "
        "provenance.csv or provenance.json at the data root, if present.",
    )
    parser.add_argument(
        "--batch-name",
        help="Group name for images with no recorded provenance. Pass the same name "
        "for two batches you believe share a recording.",
    )
    parser.add_argument(
        "--holdout",
        action="append",
        default=[],
        metavar="SESSION",
        help="Set a session aside as the final gate: in no fold, trained on never. "
        "Repeatable. Choose by condition, not by animal.",
    )
    parser.add_argument(
        "--migrate-from",
        type=Path,
        help="Seed sessions from a schema-1 manifest and write them to a sidecar. "
        "One-time migration off filename-derived grouping.",
    )
    parser.add_argument(
        "--materialize",
        nargs="?",
        const="folds",
        metavar="DIR",
        type=Path,
        help="Also write the folds to disk as DIR/cv1, DIR/cv2, ... (default: folds/). "
        "Derived output, rebuilt from the manifest and never read back.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink into --materialize instead of copying.",
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
    if args.out is None:
        args.out = Path(args.data_root) / "splits.json"

    sidecar_path = args.sidecar or provenance_module.find_sidecar(args.data_root)
    sidecar = provenance_module.load_sidecar(sidecar_path) if sidecar_path else {}
    if sidecar_path:
        print(f"Provenance sidecar: {sidecar_path} ({len(sidecar)} entries)")

    # A migration deliberately starts from no history: the manifest being replaced is
    # schema 1, which this code cannot read and must not carry folds over from.
    previous = None if (args.reassign or args.migrate_from) else read_previous(args.out)
    if args.folds is None:
        # A manifest already records its fold count; silently substituting a default
        # here would repack against a different one and leave folds empty.
        args.folds = previous["n_folds"] if previous else 5
    if args.migrate_from:
        sidecar = {**_migration_sidecar(args.migrate_from), **sidecar}
        print(
            f"Migrating from {args.migrate_from}: {len(sidecar)} key(s) seeded from the "
            "schema-1 filename grouping. Folds are repacked with stratification, so "
            "numbers recorded against the old manifest are a different experiment."
        )

    manifest = build_manifest(
        data_root=args.data_root,
        n_folds=args.folds,
        tiny_max_diameter=args.tiny_max_diameter,
        previous=previous,
        sidecar=sidecar,
        batch_name=args.batch_name,
        holdout=set(args.holdout),
        reassign=args.reassign,
    )
    print()
    print(format_census(manifest))
    active = manifest["n_sessions"] - manifest["n_holdout_sessions"]
    print(
        f"\n{manifest['n_images']} images, {manifest['n_sessions']} sessions "
        f"({active} in folds, {manifest['n_holdout_sessions']} held out), "
        f"{manifest['n_folds']} folds"
    )

    if args.show:
        print("\n--show given; manifest not written.")
        return 0

    kept = len(frozen_sessions(previous).keys() & {e["key"] for e in manifest["images"]})
    write_manifest(args.out, manifest)
    print(f"Wrote {args.out} ({kept} image assignment(s) carried over unchanged).")

    if args.materialize:
        out_dir = Path(args.data_root) / args.materialize
        counts = materialize(manifest, args.data_root, out_dir, copy=not args.symlink)
        how = "symlink" if args.symlink else "copy"
        print(
            f"Materialized {sum(counts.values())} pairs into {out_dir} by {how}: "
            + ", ".join(f"{name}={n}" for name, n in sorted(counts.items()))
        )

    if args.migrate_from:
        target = Path(args.data_root) / "provenance.csv"
        provenance_module.write_sidecar(
            target, {e["key"]: e["session"] for e in manifest["images"]}
        )
        print(f"Wrote {target} -- the recorded provenance, independent of any filename.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
