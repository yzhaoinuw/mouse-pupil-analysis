"""Focused coverage for the browser split-manager payload and write boundary."""

import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANAGER = runpy.run_path(str(PROJECT_ROOT / "training" / "split_manager.py"))
ui_state = MANAGER["ui_state"]


def test_ui_state_exposes_session_size_and_lighting_stats():
    manifest = {
        "tiny_max_diameter": 15.0,
        "n_folds": 2,
        "sessions": [
            {"session": "a", "source": "folder", "fold": 0},
            {"session": "b", "source": "folder", "fold": -2, "validation_holdout": True},
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
    assert by_session["b"]["target"] == "validation_holdout"
    assert by_session["b"]["medium"] == 1
