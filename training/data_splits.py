# -*- coding: utf-8 -*-
"""Group the labelled image pool into sessions and assign grouped cross-validation folds.

A *session* is one recording setting: one animal, on one date, under one condition.
It is the unit that must not span the train/validation boundary, because the domain
shift that breaks this model in practice is rig, camera placement, lighting, and
animal state -- not animal identity. Mice eyes are interchangeable; camera angles
are not.

Recording *files* are too fine a unit. ``HQL086_whiskerb250923_002``, ``_005``, and
``_008`` are three files from one sitting, minutes apart with the camera untouched;
splitting them across train and validation leaks the same setting under a different
name. Animals are too coarse: grouping by animal discards usable training data to
protect against a shift the data does not show.

Generate or refresh the manifest::

    python training/data_splits.py --data-root . --folds 5 --out splits.json
    python training/data_splits.py --data-root . --show          # census only, no write

Then train one fold with::

    python training/run_train.py --split-manifest splits.json --fold 0

Adding new images never reshuffles existing sessions. Sessions already present in the
manifest keep their fold, and only genuinely new sessions are packed into whichever
folds are currently smallest, so cross-validation numbers stay comparable as the
dataset grows. Pass ``--reassign`` to deliberately repack everything from scratch,
which invalidates comparison against previously recorded runs.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from mouse_pupil_analysis.augmentation import mask_equivalent_diameter, paired_image_mask_paths

SCHEMA_VERSION = 1

DEFAULT_POOL = (
    ("images_train", "masks_train"),
    ("images_validation", "masks_validation"),
)

# ``250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_0000``
TIMESTAMP = re.compile(r"^(.*?)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)")
# ``HQL080_whiskerb_250722_007_eye_06622`` -> recording index ``_007``
HQL_FILE_INDEX = re.compile(r"_(\d{2,4})$")


@dataclass(frozen=True)
class Identity:
    """Where one labelled image came from."""

    cohort: str
    animal: str
    recording: str
    session: str


def parse_identity(stem: str) -> Identity:
    """Return the cohort, animal, recording, and session for one image stem.

    Two naming schemes are in use and both have variable-length middle segments, so
    this splits on structural markers rather than tokenising. An unrecognised stem
    becomes its own single-image session under the ``unparsed`` cohort: that is the
    conservative failure mode, because it can only over-separate, never leak.
    """
    if stem.startswith("HQL"):
        animal = stem.split("_")[0]
        recording = stem.split("_eye_")[0]
        session = HQL_FILE_INDEX.sub("", recording)
        return Identity("HQL", animal, recording, session)

    match = TIMESTAMP.match(stem)
    if match:
        session, timestamp = match.group(1), match.group(2)
        parts = session.split("_")
        animal = parts[1] if len(parts) > 1 else session
        return Identity("date_id", animal, f"{session}_{timestamp}", session)

    return Identity("unparsed", stem, stem, stem)


@dataclass
class PoolImage:
    """One labelled image/mask pair, located relative to the data root."""

    stem: str
    image: str
    mask: str
    identity: Identity
    diameter: float


@dataclass
class Session:
    """Every labelled image sharing one recording setting."""

    key: str
    cohort: str
    animal: str
    images: list[PoolImage] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        return len(self.images)

    def n_tiny(self, tiny_max_diameter: float) -> int:
        return sum(image.diameter <= tiny_max_diameter for image in self.images)

    def median_diameter(self) -> float:
        return statistics.median(image.diameter for image in self.images)

    def recordings(self) -> set[str]:
        return {image.identity.recording for image in self.images}


def discover_pool(
    data_root: Path,
    pool: tuple[tuple[str, str], ...] = DEFAULT_POOL,
) -> list[PoolImage]:
    """Read every image/mask pair in the pool directories as one flat collection.

    The historical ``images_train`` / ``images_validation`` folders are treated as a
    single pool. The split is decided by the manifest, not by which folder a file
    happens to sit in, so no labelled data has to be moved to re-split it.
    """
    data_root = Path(data_root)
    found: list[PoolImage] = []
    seen: dict[str, str] = {}
    for image_dir, mask_dir in pool:
        absolute = data_root / image_dir
        if not absolute.is_dir():
            continue
        image_paths, mask_paths = paired_image_mask_paths(absolute, data_root / mask_dir)
        for image_path, mask_path in zip(image_paths, mask_paths):
            stem = image_path.stem
            if stem in seen:
                raise ValueError(
                    f"Image stem {stem!r} appears in both {seen[stem]} and {image_dir}. "
                    "One stem must map to exactly one labelled pair."
                )
            seen[stem] = image_dir
            found.append(
                PoolImage(
                    stem=stem,
                    image=f"{image_dir}/{image_path.name}",
                    mask=f"{mask_dir}/{mask_path.name}",
                    identity=parse_identity(stem),
                    diameter=mask_equivalent_diameter(mask_path),
                )
            )
    if not found:
        raise FileNotFoundError(
            f"No labelled images found under {data_root} in {[d for d, _ in pool]}."
        )
    return found


def group_sessions(images: list[PoolImage]) -> dict[str, Session]:
    """Collapse the flat pool into sessions, keyed by recording setting."""
    sessions: dict[str, Session] = {}
    for image in images:
        identity = image.identity
        session = sessions.get(identity.session)
        if session is None:
            session = Session(identity.session, identity.cohort, identity.animal)
            sessions[identity.session] = session
        session.images.append(image)
    return sessions


def assign_folds(
    sessions: dict[str, Session],
    n_folds: int,
    existing: dict[str, int] | None = None,
) -> dict[str, int]:
    """Pack whole sessions into folds, largest first, keeping folds evenly sized.

    This is deterministic and needs no seed. Sessions already carrying a valid fold in
    ``existing`` keep it, so adding data never invalidates earlier cross-validation
    results. Ties are broken toward the fold holding fewer sessions of the same cohort,
    which keeps both acquisition cohorts represented in every fold.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if n_folds > len(sessions):
        raise ValueError(
            f"Cannot build {n_folds} folds from {len(sessions)} sessions; "
            "every fold needs at least one whole session."
        )

    assignment: dict[str, int] = {}
    sizes = Counter({fold: 0 for fold in range(n_folds)})
    cohorts: dict[int, Counter] = {fold: Counter() for fold in range(n_folds)}

    for key, fold in (existing or {}).items():
        if key in sessions and isinstance(fold, int) and 0 <= fold < n_folds:
            assignment[key] = fold
            sizes[fold] += sessions[key].n_images
            cohorts[fold][sessions[key].cohort] += 1

    remaining = [key for key in sessions if key not in assignment]
    for key in sorted(remaining, key=lambda k: (-sessions[k].n_images, k)):
        cohort = sessions[key].cohort
        fold = min(range(n_folds), key=lambda f: (sizes[f], cohorts[f][cohort], f))
        assignment[key] = fold
        sizes[fold] += sessions[key].n_images
        cohorts[fold][cohort] += 1
    return assignment


def build_manifest(
    data_root: Path,
    n_folds: int = 5,
    pool: tuple[tuple[str, str], ...] = DEFAULT_POOL,
    tiny_max_diameter: float = 15.0,
    existing: dict[str, int] | None = None,
    generated: str | None = None,
) -> dict:
    """Return the complete split manifest for the labelled pool under ``data_root``."""
    images = discover_pool(data_root, pool)
    sessions = group_sessions(images)
    assignment = assign_folds(sessions, n_folds, existing)
    return {
        "schema": SCHEMA_VERSION,
        "generated": generated or date.today().isoformat(),
        "grouping": "session (one animal, one date, one condition)",
        "n_folds": n_folds,
        "n_images": len(images),
        "n_sessions": len(sessions),
        "pool": [list(entry) for entry in pool],
        "tiny_max_diameter": tiny_max_diameter,
        "sessions": [
            {
                "session": key,
                "fold": assignment[key],
                "cohort": session.cohort,
                "animal": session.animal,
                "n_recordings": len(session.recordings()),
                "n_images": session.n_images,
                "n_tiny": session.n_tiny(tiny_max_diameter),
                "median_diameter": round(session.median_diameter(), 2),
            }
            for key, session in sorted(sessions.items(), key=lambda kv: (assignment[kv[0]], kv[0]))
        ],
        "images": [
            {
                "stem": image.stem,
                "image": image.image,
                "mask": image.mask,
                "session": image.identity.session,
                "fold": assignment[image.identity.session],
                "diameter": round(image.diameter, 2),
            }
            for image in sorted(images, key=lambda i: i.stem)
        ],
    }


def load_manifest(path: Path) -> dict:
    """Read a manifest and reject one this code cannot interpret."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"{path} declares schema {schema!r}, but this code understands {SCHEMA_VERSION}. "
            "Regenerate it with training/data_splits.py."
        )
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def existing_assignment(path: Path) -> dict[str, int]:
    """Return the session-to-fold mapping already recorded at ``path``, if any."""
    if not Path(path).exists():
        return {}
    manifest = load_manifest(path)
    return {entry["session"]: entry["fold"] for entry in manifest["sessions"]}


def fold_paths(
    manifest: dict,
    fold: int,
    data_root: Path,
) -> tuple[tuple[list[Path], list[Path]], tuple[list[Path], list[Path]]]:
    """Return ``((train_images, train_masks), (val_images, val_masks))`` for one fold.

    The held-out fold is validation; every other fold trains. Paths come back in a
    stable order so a run is reproducible from the manifest alone.
    """
    n_folds = manifest["n_folds"]
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds}); got {fold}.")
    data_root = Path(data_root)

    train: tuple[list[Path], list[Path]] = ([], [])
    validation: tuple[list[Path], list[Path]] = ([], [])
    for entry in sorted(manifest["images"], key=lambda e: e["stem"]):
        target = validation if entry["fold"] == fold else train
        target[0].append(data_root / entry["image"])
        target[1].append(data_root / entry["mask"])

    if not validation[0]:
        raise ValueError(f"Fold {fold} holds no images; regenerate the manifest.")
    if not train[0]:
        raise ValueError(f"Fold {fold} leaves no training images; regenerate the manifest.")
    return train, validation


def fold_sessions(manifest: dict, fold: int) -> list[str]:
    """Return the session keys held out by one fold."""
    return [entry["session"] for entry in manifest["sessions"] if entry["fold"] == fold]


def session_of_stem(manifest: dict) -> dict[str, str]:
    """Map every image stem to its session key, for per-session reporting."""
    return {entry["stem"]: entry["session"] for entry in manifest["images"]}


def format_census(manifest: dict) -> str:
    """Render the fold assignment as a review table."""
    tiny_max = manifest["tiny_max_diameter"]
    lines = [
        f"{'session':<40} {'cohort':<8} {'fold':>4} {'recs':>5} {'imgs':>5} "
        f"{'tiny':>5} {'med_d':>6}",
    ]
    for entry in manifest["sessions"]:
        lines.append(
            f"{entry['session'][:38]:<40} {entry['cohort']:<8} {entry['fold']:>4} "
            f"{entry['n_recordings']:>5} {entry['n_images']:>5} {entry['n_tiny']:>5} "
            f"{entry['median_diameter']:>6.1f}"
        )

    lines.append("")
    lines.append(f"{'fold':>4} {'sessions':>9} {'images':>7} {'share':>6} {'tiny':>5}  cohorts")
    total = manifest["n_images"]
    for fold in range(manifest["n_folds"]):
        held = [entry for entry in manifest["sessions"] if entry["fold"] == fold]
        images = sum(entry["n_images"] for entry in held)
        tiny = sum(entry["n_tiny"] for entry in held)
        cohorts = Counter(entry["cohort"] for entry in held)
        lines.append(
            f"{fold:>4} {len(held):>9} {images:>7} {100 * images / total:>5.0f}% {tiny:>5}  "
            f"{dict(sorted(cohorts.items()))}"
        )

    unparsed = [entry for entry in manifest["sessions"] if entry["cohort"] == "unparsed"]
    if unparsed:
        lines.append("")
        lines.append(
            f"WARNING: {len(unparsed)} session(s) have unrecognised filenames and were each "
            "treated as their own session:"
        )
        for entry in unparsed[:10]:
            lines.append(f"  {entry['session']}")
        lines.append(
            "  Grouping is only as good as the filenames. See training/data_collection.md."
        )

    tiny_total = sum(entry["n_tiny"] for entry in manifest["sessions"])
    empty = [
        fold
        for fold in range(manifest["n_folds"])
        if not any(entry["n_tiny"] for entry in manifest["sessions"] if entry["fold"] == fold)
    ]
    if tiny_total and empty:
        lines.append("")
        lines.append(
            f"NOTE: folds {empty} hold no masks at or below {tiny_max:g} model pixels, so their "
            "tiny-bin IoU is undefined. Small pupils concentrate in very few sessions."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("splits.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tiny-max-diameter", type=float, default=15.0)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the census without writing the manifest.",
    )
    parser.add_argument(
        "--reassign",
        action="store_true",
        help="Repack every session from scratch, discarding existing fold assignments. "
        "This makes new runs incomparable to previously recorded ones.",
    )
    args = parser.parse_args(argv)

    existing = {} if args.reassign else existing_assignment(args.out)
    manifest = build_manifest(
        data_root=args.data_root,
        n_folds=args.folds,
        tiny_max_diameter=args.tiny_max_diameter,
        existing=existing,
    )
    print(format_census(manifest))
    print(
        f"\n{manifest['n_images']} images, {manifest['n_sessions']} sessions, "
        f"{manifest['n_folds']} folds"
    )

    if args.show:
        print("\n--show given; manifest not written.")
        return 0

    kept = sum(1 for key in existing if key in {e["session"] for e in manifest["sessions"]})
    write_manifest(args.out, manifest)
    print(f"Wrote {args.out} ({kept} session assignment(s) carried over).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
