"""Plotting must not depend on an interactive Matplotlib backend.

A Windows CI job failed intermittently with `_tkinter.TclError` because pyplot
selected TkAgg and the runner's Tcl install was broken. The same failure hits any
headless machine without a working Tk: compute nodes, containers, SSH without X.
Building `Figure` objects directly avoids backend selection entirely, and these
tests pin that so a future edit cannot quietly reintroduce the pyplot dependency.
"""

import builtins
import subprocess
import sys

import numpy as np
import pandas as pd

from pupil_tracking.plotting import plot_analysis
from pupil_tracking.results import DIAMETER_COLUMNS


def _diameter_table() -> tuple[pd.DataFrame, np.ndarray]:
    table = pd.DataFrame(
        {
            "image_name": ["eye_00001.png", "eye_00002.png", "eye_00003.png"],
            "estimated_pupil_diameter": [10.0, 11.0, 12.0],
            "pupil_diameter_input_pixels": [20.0, 22.0, 24.0],
        }
    )[DIAMETER_COLUMNS]
    return table, np.array([1, 2, 3])


def test_package_does_not_import_pyplot():
    code = (
        "import sys, pupil_tracking.plotting, pupil_tracking.results;"
        "print('matplotlib.pyplot' in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


def test_plot_renders_without_tkinter(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name in ("tkinter", "_tkinter") or name.startswith("tkinter."):
            raise ImportError("simulated missing Tcl/Tk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv("MPLBACKEND", "TkAgg")
    monkeypatch.setattr(builtins, "__import__", blocked)

    table, frame_numbers = _diameter_table()
    figure = plot_analysis(table, frame_numbers, include_tracking=False)

    output = tmp_path / "plot.png"
    figure.savefig(output, dpi=100)
    assert output.stat().st_size > 0


def test_repeated_figures_are_not_retained_globally():
    table, frame_numbers = _diameter_table()

    # pyplot would accumulate these in its global registry and warn past 20.
    figures = [plot_analysis(table, frame_numbers, include_tracking=False) for _ in range(25)]

    assert len({id(figure) for figure in figures}) == 25
    assert (
        "matplotlib.pyplot" not in sys.modules or not sys.modules["matplotlib.pyplot"].get_fignums()
    )
