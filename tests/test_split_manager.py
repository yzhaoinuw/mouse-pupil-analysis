"""Focused coverage for the browser split-manager payload and write boundary."""

import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANAGER = runpy.run_path(str(PROJECT_ROOT / "training" / "split_manager.py"))
BrowserLifecycle = MANAGER["BrowserLifecycle"]
ui_state = MANAGER["ui_state"]
read_page = MANAGER["read_page"]
split_paths = MANAGER["split_paths"]


def test_ui_state_exposes_session_size_and_lighting_stats():
    manifest = {
        "tiny_max_diameter": 15.0,
        "n_folds": 2,
        "sessions": [
            {"session": "a", "fold": 0},
            {"session": "b", "fold": -2, "validation_holdout": True},
        ],
        "images": [
            {"session": "a", "diameter": 10, "brightness": 30},
            {"session": "a", "diameter": 90, "brightness": 50},
            {"session": "b", "diameter": 20, "brightness": 70},
        ],
    }

    state = ui_state(manifest)
    by_session = {entry["session"]: entry for entry in state["sessions"]}

    assert by_session["a"]["target"] == 0
    assert by_session["a"]["tiny"] == 1
    assert by_session["a"]["large"] == 1
    assert by_session["a"]["median_brightness"] == 40.0
    assert by_session["a"]["brightness_values"] == [30.0, 50.0]
    assert by_session["b"]["target"] == "validation_holdout"
    assert by_session["b"]["medium"] == 1


def test_split_paths_uses_the_manifest_beside_labeled_frames(tmp_path: Path):
    data_root, manifest_path = split_paths(tmp_path / "labeled_frames")

    assert data_root == tmp_path.resolve()
    assert manifest_path == tmp_path.resolve() / "splits.json"


def test_html_template_is_a_tracked_ui_asset():
    page = read_page()

    assert b"Training split manager" in page
    assert b"Fold distribution" in page
    assert b"Selected session" in page
    assert b"Background brightness: Q1" in page
    assert b"/api/heartbeat" in page
    assert b"drawChart" in page
    assert b"/api/assignments" in page


def test_browser_lifecycle_waits_briefly_after_last_tab_closes():
    lifecycle = BrowserLifecycle()

    lifecycle.touch("tab-a")
    assert not lifecycle.should_stop()

    lifecycle.close("tab-a")
    lifecycle.last_client_change_at -= 6

    assert lifecycle.should_stop()
