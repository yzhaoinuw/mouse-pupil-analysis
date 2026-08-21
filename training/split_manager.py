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
import threading
import time
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
TAB_STALE_SECONDS = 8.0
TAB_CLOSE_GRACE_SECONDS = 5.0
STARTUP_GRACE_SECONDS = 120.0


class BrowserLifecycle:
    """Track manager tabs so the loopback server exits after they are gone."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.last_client_change_at = self.started_at
        self.clients: dict[str, float] = {}
        self.connected_once = False
        self.lock = threading.Lock()

    def touch(self, client_id: str) -> None:
        with self.lock:
            self.clients[client_id] = time.monotonic()
            self.connected_once = True
            self.last_client_change_at = time.monotonic()

    def close(self, client_id: str) -> None:
        with self.lock:
            self.clients.pop(client_id, None)
            self.last_client_change_at = time.monotonic()

    def should_stop(self) -> bool:
        """Return true only after startup failure or the last tab has been gone briefly."""
        with self.lock:
            now = time.monotonic()
            stale = [
                client_id
                for client_id, last_seen in self.clients.items()
                if now - last_seen > TAB_STALE_SECONDS
            ]
            for client_id in stale:
                self.clients.pop(client_id)
                self.last_client_change_at = now
            if not self.connected_once:
                return now - self.started_at > STARTUP_GRACE_SECONDS
            return not self.clients and now - self.last_client_change_at > TAB_CLOSE_GRACE_SECONDS


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
                "brightness_values": brightness,
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


def handler_factory(manifest_path: Path, lifecycle: BrowserLifecycle):
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

        def _payload(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("Request body must be between 1 and 1,000,000 bytes.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            return payload

        @staticmethod
        def _client_id(payload: dict) -> str:
            client_id = payload.get("clientId")
            if not isinstance(client_id, str) or not client_id:
                raise ValueError("A browser client id is required.")
            return client_id

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
            if self.path in {"/api/heartbeat", "/api/close"}:
                try:
                    payload = self._payload()
                    client_id = self._client_id(payload)
                    if self.path == "/api/heartbeat":
                        lifecycle.touch(client_id)
                    else:
                        lifecycle.close(client_id)
                    self._json(HTTPStatus.OK, {"ok": True})
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            if self.path != "/api/assignments":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                return
            try:
                payload = self._payload()
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


def stop_when_browser_closes(server: ThreadingHTTPServer, lifecycle: BrowserLifecycle) -> None:
    """Stop the loopback service after its last browser tab is gone."""
    while not lifecycle.should_stop():
        time.sleep(1)
    print("Split manager stopped because its browser tab(s) closed.")
    server.shutdown()


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
    lifecycle = BrowserLifecycle()
    server = ThreadingHTTPServer((HOST, 0), handler_factory(manifest_path, lifecycle))
    server.daemon_threads = True
    url = f"http://{HOST}:{server.server_port}/"
    print(f"Split manager: {url}\nThe server stops after the last browser tab closes.")
    if not args.no_open:
        webbrowser.open(url)
    threading.Thread(
        target=stop_when_browser_closes,
        args=(server, lifecycle),
        daemon=True,
    ).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSplit manager stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
