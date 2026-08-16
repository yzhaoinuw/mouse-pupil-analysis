# -*- coding: utf-8 -*-
"""Where a labelled image came from, read from what was recorded -- never inferred.

An image's *session* is the recording setting it came from: one animal, one date, one
condition. It is the unit that must not span the train/validation boundary, and it is
the only thing the split really needs to know.

Session identity cannot be recovered from the images themselves. That is a measured
result, not a design preference; see ``training/data_collection.md`` for the three
methods that were tried and the numbers they produced. Nor can it be recovered from
filenames, which is why nothing here parses one. So it has to be *recorded* at intake,
by whoever knows the answer, and then frozen.

Four recorded sources, most explicit first:

1. **Sidecar** -- ``provenance.csv`` (``key,session``) or ``provenance.json``
   (``{key: session}``) at the data root. A *key* is an image's path within its pool
   folder without the extension: ``frame_0001`` flat, ``rig2_day3/frame_0001`` nested. Use for a batch that arrived already mixed.
2. **Labelme flag** -- ``flags.session`` in the ``<stem>.json`` beside the image. The
   labeller sets it once per batch in the UI, at the moment they know it.
3. **Session folder** -- ``labeled_data/<session>/images/``. This is the normal route
   and it asks nothing of how files are named: the session is a directory, so an image
   cannot enter the pool without one. The sources above exist for a batch that arrived
   before anyone could sort it.
4. **Batch fallback** -- everything still unresolved becomes a single group. Merging
   two settings only costs data efficiency; tearing one apart leaks. So the safe
   failure mode is to over-merge, and it needs no human input at all.

The fallback groups per intake run, so two separate unknown batches become two groups.
If you have reason to think they share a recording, pass the same ``batch_name`` for
both and they merge into one.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

SIDECAR_NAMES = ("provenance.csv", "provenance.json")


@dataclass(frozen=True)
class Provenance:
    """One image's recorded session, and which source supplied it."""

    session: str
    source: str


def load_sidecar(path: Path) -> dict[str, str]:
    """Read a key-to-session mapping from CSV or JSON.

    CSV needs a header with ``key`` and ``session`` columns; JSON is a flat object.
    A key is an image's path within its pool folder, without the extension -- a bare
    filename when it sits flat, ``<intake folder>/<filename>`` when it does not.
    Blank sessions are rejected rather than silently ignored, because a half-filled
    sidecar that quietly falls through to the batch fallback is worse than an error.
    """
    path = Path(path)
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must be a JSON object mapping stem to session.")
        mapping = {str(k): str(v) for k, v in raw.items()}
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            if not {"key", "session"} <= fields:
                raise ValueError(
                    f"{path} needs 'key' and 'session' columns; found {sorted(fields)}."
                )
            mapping = {row["key"].strip(): (row["session"] or "").strip() for row in reader}
    else:
        raise ValueError(f"Provenance sidecar must be .csv or .json; got {path.name}.")

    blank = sorted(key for key, session in mapping.items() if not session)
    if blank:
        raise ValueError(f"{path} leaves the session blank for: {blank[:5]}")
    return mapping


def find_sidecar(data_root: Path) -> Path | None:
    """Return the sidecar at the data root, if one is present."""
    for name in SIDECAR_NAMES:
        candidate = Path(data_root) / name
        if candidate.is_file():
            return candidate
    return None


def write_sidecar(path: Path, mapping: dict[str, str]) -> None:
    """Write a key-to-session sidecar, sorted so diffs stay readable."""
    path = Path(path)
    if path.suffix == ".json":
        path.write_text(
            json.dumps(dict(sorted(mapping.items())), indent=2) + "\n", encoding="utf-8"
        )
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "session"])
        for key, session in sorted(mapping.items()):
            writer.writerow([key, session])


def labelme_session(image_path: Path) -> str | None:
    """Return ``flags.session`` from the labelme JSON beside an image, if set.

    A malformed or unreadable JSON returns ``None`` rather than raising: the file is
    the annotation tool's output, and a missing flag is the normal case, not an error.
    """
    sidecar = Path(image_path).with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        flags = json.loads(sidecar.read_text(encoding="utf-8")).get("flags") or {}
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    session = flags.get("session")
    if isinstance(session, str) and session.strip():
        return session.strip()
    return None


def resolve(
    images: dict[str, tuple[Path, str | None]],
    sidecar: dict[str, str] | None = None,
    batch_name: str = "unknown_batch",
) -> dict[str, Provenance]:
    """Resolve every image's session from the recorded sources, in precedence order.

    ``images`` maps key to ``(image_path, session or None)``, where the session is
    already known for anything read out of a session folder. Returns one
    :class:`Provenance` per key; every key always resolves, because the batch fallback
    catches whatever the explicit sources missed.
    """
    sidecar = sidecar or {}
    resolved: dict[str, Provenance] = {}
    for key, (image_path, folder) in images.items():
        session = sidecar.get(key)
        if session:
            resolved[key] = Provenance(session, "sidecar")
            continue

        session = labelme_session(image_path)
        if session:
            resolved[key] = Provenance(session, "labelme")
            continue

        if folder:
            resolved[key] = Provenance(folder, "folder")
            continue

        resolved[key] = Provenance(batch_name, "batch")
    return resolved
