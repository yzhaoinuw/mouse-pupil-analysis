"""Run a local browser UI for reviewing and editing session-level data splits.

The page reads ``splits.json`` produced by the existing automatic grouping, then opens a
loopback-only page. Drag whole sessions between development folds and the validation
holdout; the page validates and writes the manifest through
``data_splits.py`` rather than editing JSON in the browser.

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


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pupil training split manager</title>
<style>
:root { color-scheme: light dark; font-family: system-ui, sans-serif; }
body { max-width: 1500px; margin: 0 auto; padding: 1.5rem; background: #f7f8fa; color: #18212b; }
h1 { margin: 0; } p { line-height: 1.45; } .muted { color: #56616d; }
#notice { min-height: 1.5rem; font-weight: 600; } .error { color: #a61919; }
#summary, #groups { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); }
.summary, .group { background: white; border: 1px solid #d9dee4; border-radius: 10px; padding: 1rem; box-shadow: 0 1px 2px #0000000d; }
.summary h2, .group h2 { margin: 0 0 .35rem; font-size: 1rem; }
.numbers { display: grid; grid-template-columns: repeat(4, 1fr); gap: .25rem; font-size: .82rem; }
.numbers span { background: #eef2f6; padding: .3rem; border-radius: 4px; text-align: center; }
.group { min-height: 185px; } .group.drop-target { outline: 3px solid #6aa8ff; }
.card { background: #f5f8fb; border: 1px solid #cbd6e2; border-left: 5px solid #497aa8; border-radius: 7px; margin: .55rem 0; padding: .55rem; cursor: grab; }
.card.locked { border-left-color: #8d5c23; cursor: default; background: #fbf6ee; }
.card strong { display: block; overflow-wrap: anywhere; }.meta { font-size: .78rem; color: #53606c; margin-top: .25rem; }
button { border: 0; border-radius: 6px; padding: .65rem .9rem; margin-right: .5rem; background: #175ea8; color: white; font-weight: 700; cursor: pointer; }
button.secondary { background: #697784; } .actions { margin: 1.25rem 0; }
@media (prefers-color-scheme: dark) { body { background: #17202a; color: #eaf0f6; } .summary, .group { background: #202b36; border-color: #40505f; } .card { background: #273543; border-color: #52687d; } .numbers span { background: #314252; } .meta, .muted { color: #b1c0ce; } .card.locked { background: #423928; } }
</style>
</head>
<body>
<h1>Training split manager</h1>
<p class="muted">The automatic grouping is the starting point. Drag whole sessions to change development folds or create a validation holdout. Cross-validation ignores the validation holdout. Save writes <code>splits.json</code> after validation.</p>
<div id="notice"></div>
<div class="actions"><button id="save">Save assignments</button><button id="reload" class="secondary">Discard changes</button></div>
<div id="summary"></div>
<h2>Session assignments</h2>
<div id="groups"></div>
<script>
let state;
const notice = document.querySelector('#notice');
const summary = document.querySelector('#summary');
const groups = document.querySelector('#groups');
function el(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; }
function label(target) { return target === 'validation_holdout' ? 'Validation holdout' : target === 'test_holdout' ? 'Outer test holdout' : `Development fold ${target + 1}`; }
function targets() { return [...Array(state.n_folds).keys()].concat(['validation_holdout', 'test_holdout']); }
function stats(target) { const records = state.sessions.filter(session => session.target === target); const sum = field => records.reduce((total, session) => total + session[field], 0); const median = field => { const values = records.map(session => session[field]).sort((a,b) => a-b); return values.length ? values[Math.floor(values.length / 2)].toFixed(1) : '—'; }; return { records, sessions: records.length, images: sum('n_images'), tiny: sum('tiny'), medium: sum('medium'), large: sum('large'), diameter: median('median_diameter'), brightness: median('median_brightness') }; }
function card(session) { const node = el('article', `card${session.target === 'test_holdout' ? ' locked' : ''}`); node.draggable = session.target !== 'test_holdout'; node.dataset.session = session.session; node.append(el('strong', '', session.session)); node.append(el('div', 'meta', `${session.n_images} images · tiny ${session.tiny} · medium ${session.medium} · large ${session.large}`)); node.append(el('div', 'meta', `median diameter ${session.median_diameter} · brightness ${session.median_brightness} · ${session.source}`)); node.addEventListener('dragstart', event => event.dataTransfer.setData('text/plain', session.session)); return node; }
function render() { summary.replaceChildren(); groups.replaceChildren(); targets().forEach(target => { const value = stats(target); if (target === 'test_holdout' && !value.sessions) return; const summaryCard = el('section', 'summary'); summaryCard.append(el('h2', '', label(target))); summaryCard.append(el('div', 'muted', `${value.sessions} sessions · ${value.images} images · median diameter ${value.diameter} · brightness ${value.brightness}`)); const counts = el('div', 'numbers'); [['Tiny', value.tiny], ['Medium', value.medium], ['Large', value.large], ['Images', value.images]].forEach(([name, number]) => counts.append(el('span', '', `${name}: ${number}`))); summaryCard.append(counts); summary.append(summaryCard); const group = el('section', 'group'); group.dataset.target = target; group.append(el('h2', '', label(target))); group.append(el('div', 'muted', target === 'validation_holdout' ? 'Used by normal run_train validation; excluded from CV.' : target === 'test_holdout' ? 'Read-only here; reserved for one-time final evaluation.' : 'Used by cross-validation.')); value.records.forEach(session => group.append(card(session))); if (target !== 'test_holdout') { group.addEventListener('dragover', event => { event.preventDefault(); group.classList.add('drop-target'); }); group.addEventListener('dragleave', () => group.classList.remove('drop-target')); group.addEventListener('drop', event => { event.preventDefault(); group.classList.remove('drop-target'); const session = state.sessions.find(item => item.session === event.dataTransfer.getData('text/plain')); if (session) { session.target = target === 'validation_holdout' ? target : Number(target); render(); } }); } groups.append(group); }); }
async function load() { const response = await fetch('/api/state'); state = await response.json(); notice.textContent = `Loaded ${state.sessions.length} sessions. Tiny ≤ ${state.tiny_max_diameter}; large ≥ ${state.large_min_diameter} model pixels.`; notice.className = ''; render(); }
document.querySelector('#reload').addEventListener('click', load);
document.querySelector('#save').addEventListener('click', async () => { const assignments = Object.fromEntries(state.sessions.filter(session => session.target !== 'test_holdout').map(session => [session.session, session.target])); const response = await fetch('/api/assignments', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({revision: state.revision, assignments}) }); const payload = await response.json(); if (!response.ok) { notice.textContent = payload.error || 'Could not save assignments.'; notice.className = 'error'; return; } state = payload; notice.textContent = 'Saved validated session assignments to splits.json.'; notice.className = ''; render(); });
load().catch(error => { notice.textContent = error.message; notice.className = 'error'; });
</script>
</body>
</html>"""


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
                encoded = PAGE.encode("utf-8")
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
        f"Loaded {manifest['n_sessions']} sessions into {manifest['n_folds']} development folds; "
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
