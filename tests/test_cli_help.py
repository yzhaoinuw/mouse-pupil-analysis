# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:14:01 2026

@author: yzhao
"""

import subprocess
import sys
from pathlib import Path

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
