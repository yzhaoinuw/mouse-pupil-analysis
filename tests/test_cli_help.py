# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:14:01 2026

@author: yzhao
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mouse_pupil_analysis import analyze_frames

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _console_script(name: str) -> Path:
    """Return a console-script path from the active Python environment."""
    executable_dir = Path(sys.executable).parent
    if sys.platform == "win32":
        return executable_dir / "Scripts" / f"{name}.exe"
    return executable_dir / name


def test_cli_help_runs():
    # Uses the console-script entrypoint advertised in the README.
    proc = subprocess.run(
        [_console_script("run-pupil-analysis"), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_training_script_help_runs():
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "training" / "run_train.py"), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("legacy", [False, True])
def test_cli_warns_for_deprecated_selector_without_changing_its_behavior(tmp_path, legacy):
    frames = PROJECT_ROOT / "sample_data" / "unlabeled_frames" / "recording_250530"
    result_dir = tmp_path / "cli_results"
    command = [
        _console_script("run-pupil-analysis"),
        "--image_dir",
        frames,
        "--result_dir",
        result_dir,
        "--num_workers",
        "0",
    ]
    if legacy:
        command.append("--prefer_central_component")
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert ("--prefer_central_component is deprecated" in proc.stderr) == legacy
    if legacy:
        assert "still use the legacy image-center selector" in proc.stderr

    expected = analyze_frames(
        frames,
        result_dir=tmp_path / "api_results",
        num_workers=0,
        prefer_central_component=legacy,
    ).analysis_table
    actual = pd.read_csv(next(result_dir.glob("*.csv")))
    assert actual.image_name.tolist() == expected.image_name.tolist()
    np.testing.assert_allclose(
        actual.estimated_pupil_diameter, expected.estimated_pupil_diameter, rtol=0, atol=1e-12
    )
