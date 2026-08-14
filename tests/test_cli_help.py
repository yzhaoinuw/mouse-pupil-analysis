# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:14:01 2026

@author: yzhao
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_runs():
    # Uses the console script entrypoint you advertise in README
    # (works on all OS; if it fails on Windows CI later, we can adjust)
    proc = subprocess.run(
        ["run-pupil-analysis", "--help"],
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
