"""Run a local browser UI for reviewing and editing session-level data splits.

The page reads ``splits.json`` produced by the existing automatic grouping, then opens a
loopback-only interface. Drag whole sessions between folds and the validation
holdout; the Python backend validates and writes the manifest.

    python training/split_manager.py --data-root .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_splits  # noqa: E402

HOST = "127.0.0.1"
MAX_REQUEST_BYTES = 1_000_000
LARGE_MIN_DIAMETER = 80.0
PAGE_PATH = Path(__file__).with_name("split_manager.html")


def manifest_revision(manifest: dict) -> str:
    """Return a stable optimistic-lock token for a manifest response."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ui_state(manifest: dict) -> dict:
    """Return the compact session/fold statistics consumed by the browser UI."""
    tiny_max = float(manifest["tiny_max_diameter"])
    images_by_session: dict[str, list[dict]] = {}
    for image in manifest["images"]:
        images_by_session.setdefault(image["session"], []).append(image)

    sessions = []
    for entry in manifest["sessions"]:
        images = images_by_session[entry["session"]]
        diameters = [float(image["diameter"]) for image in images]
        brightness = [float(image["brightness"]) for image in images]
        sessions.append(
            {
                "session": entry["session"],
                "source": entry["source"],
                "target": (
                    "test_holdout"
                    if entry.get("holdout")
                    else "validation_holdout" if entry.get("validation_holdout") else entry["fold"]
                ),
                "n_images": len(images),
                "tiny": sum(diameter <= tiny_max for diameter in diameters),
                "medium": sum(tiny_max < diameter < LARGE_MIN_DIAMETER for diameter in diameters),
                "large": sum(diameter >= LARGE_MIN_DIAMETER for diameter in diameters),
                "median_diameter": round(statistics.median(diameters), 1),
                "median_brightness": round(statistics.median(brightness), 1),
            }
        )
    return {
        "revision": manifest_revision(manifest),
        "n_folds": manifest["n_folds"],
        "tiny_max_diameter": tiny_max,
        "large_min_diameter": LARGE_MIN_DIAMETER,
        "sessions": sorted(sessions, key=lambda item: (str(item["target"]), item["session"])),
    }


def refresh_manifest(data_root: Path, manifest_path: Path, folds: int | None) -> dict:
    """Refresh source statistics while preserving prior automatic/manual assignments."""
    previous = data_splits.read_previous(manifest_path)
    n_folds = folds if folds is not None else previous["n_folds"] if previous else 5
    manifest = data_splits.build_manifest(data_root, n_folds=n_folds, previous=previous)
    data_splits.write_manifest(manifest_path, manifest)
    return manifest


def read_page() -> bytes:
    """Return the tracked UI template, failing clearly if a source checkout is incomplete."""
    if not PAGE_PATH.is_file():
        raise FileNotFoundError(f"Split-manager HTML template is missing: {PAGE_PATH}")
    return PAGE_PATH.read_bytes()


def handler_factory(manifest_path: Path):
    """Build a loopback request handler bound to one manifest path."""

    class SplitManagerHandler(BaseHTTPRequestHandler):
        def _json(self, status: HTTPStatus, payload: dict) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                try:
                    encoded = read_page()
                except FileNotFoundError as error:
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            elif self.path == "/api/state":
                self._json(HTTPStatus.OK, ui_state(data_splits.load_manifest(manifest_path)))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/assignments":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("Request body must be between 1 and 1,000,000 bytes.")
                payload = json.loads(self.rfile.read(length))
                manifest = data_splits.load_manifest(manifest_path)
                if payload.get("revision") != manifest_revision(manifest):
                    self._json(
                        HTTPStatus.CONFLICT,
                        {"error": "The manifest changed. Reload before saving."},
                    )
                    return
                updated = data_splits.apply_session_assignments(
                    manifest, payload.get("assignments", {})
                )
                data_splits.write_manifest(manifest_path, updated)
                self._json(HTTPStatus.OK, ui_state(updated))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def log_message(self, _format: str, *_args) -> None:
            return

    return SplitManagerHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, help="Default: <data-root>/splits.json.")
    parser.add_argument("--folds", type=int, help="Used only when creating the first manifest.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh source data with automatic grouping before opening the page.",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open the browser automatically."
    )
    args = parser.parse_args(argv)
    data_root = args.data_root.resolve()
    manifest_path = (args.manifest or data_root / "splits.json").resolve()
    if args.refresh or not manifest_path.exists():
        manifest = refresh_manifest(data_root, manifest_path, args.folds)
    else:
        manifest = data_splits.load_manifest(manifest_path)
    print(
        f"Loaded {manifest['n_sessions']} sessions into {manifest['n_folds']} folds; "
        f"{manifest.get('n_validation_holdout_sessions', 0)} validation-holdout session(s)."
    )
    server = ThreadingHTTPServer((HOST, 0), handler_factory(manifest_path))
    url = f"http://{HOST}:{server.server_port}/"
    print(f"Split manager: {url}\nPress Ctrl+C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSplit manager stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
