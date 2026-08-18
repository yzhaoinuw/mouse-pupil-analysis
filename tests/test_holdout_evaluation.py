"""Safety guards for the one-shot outer-holdout evaluator."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = runpy.run_path(str(PROJECT_ROOT / "training" / "evaluate_holdout.py"))
evaluate_holdout = EVALUATOR["evaluate_holdout"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_frozen_run(run_dir: Path, manifest: Path) -> None:
    run_dir.mkdir()
    checkpoint = run_dir / "final.pth"
    checkpoint.write_bytes(b"frozen checkpoint")
    (run_dir / "final.json").write_text(
        json.dumps(
            {
                "workflow": "final_refit",
                "checkpoint_sha256": _sha256(checkpoint),
                "split_manifest_sha256": _sha256(manifest),
                "holdout_sessions": ["gate"],
            }
        ),
        encoding="utf-8",
    )


def test_holdout_result_is_never_overwritten(tmp_path):
    manifest = tmp_path / "splits.json"
    manifest.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_frozen_run(run_dir, manifest)
    (run_dir / "holdout.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already been evaluated"):
        evaluate_holdout(run_dir, tmp_path, manifest)


def test_changed_manifest_is_rejected_before_holdout_loading(tmp_path):
    manifest = tmp_path / "splits.json"
    manifest.write_text("original", encoding="utf-8")
    run_dir = tmp_path / "run"
    _write_frozen_run(run_dir, manifest)
    manifest.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="split manifest changed"):
        evaluate_holdout(run_dir, tmp_path, manifest)
