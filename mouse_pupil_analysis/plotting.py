"""Figures for pupil analysis results.

These functions return a Matplotlib figure rather than saving it, so they can be
reused to restyle or embed the standard panels elsewhere.

They construct ``Figure`` objects directly instead of going through ``pyplot``.
pyplot selects an interactive backend on import, which fails outright on headless
machines whose Tk installation is absent or broken, and it registers every figure
in a global list that a library has no business touching. Building the figure
directly avoids both, and needs no ``plt.close()`` to stay leak-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

STATUS_CODES = {"invalid": 0, "warning": 1, "valid": 2}

STATUS_STYLES = (
    ("valid", 2, "#16A34A", 8),
    ("warning", 1, "#F59E0B", 16),
    ("invalid", 0, "#DC2626", 16),
)


def plot_diameter(analysis_table: pd.DataFrame, frame_numbers: np.ndarray) -> Figure:
    """Plot pupil diameter against the one-based source-frame number."""
    figure = Figure(figsize=(10, 6))
    axis = figure.subplots()
    axis.plot(frame_numbers, analysis_table["estimated_pupil_diameter"], linewidth=1)
    if "segmentation_status" in analysis_table:
        for status, _code, color, size in STATUS_STYLES:
            selected = analysis_table["segmentation_status"].eq(status)
            axis.scatter(
                frame_numbers[selected],
                analysis_table.loc[selected, "estimated_pupil_diameter"],
                color=color,
                s=size,
                label=status,
                zorder=3,
            )
        axis.legend(loc="best", ncol=3)
    axis.set_ylabel("Estimated pupil diameter\n(model pixels)")
    axis.set_title("Pupil Analysis")
    axis.set_xlabel("Frame (1-based)")
    figure.tight_layout()
    return figure


def plot_diameter_and_tracking(analysis_table: pd.DataFrame, frame_numbers: np.ndarray) -> Figure:
    """Plot diameter, center, speed, and quality control on a shared frame axis."""
    figure = Figure(figsize=(12, 11))
    axes = figure.subplots(4, 1, sharex=True)

    axes[0].plot(frame_numbers, analysis_table["estimated_pupil_diameter"], linewidth=0.8)
    axes[0].set_ylabel("Diameter\n(model pixels)")
    axes[0].set_title("Pupil Analysis")

    axes[1].plot(frame_numbers, analysis_table["center_x_pixels"], label="x", linewidth=0.8)
    axes[1].plot(frame_numbers, analysis_table["center_y_pixels"], label="y", linewidth=0.8)
    axes[1].set_ylabel("Center\n(input pixels)")
    axes[1].legend(loc="upper right")

    axes[2].plot(frame_numbers, analysis_table["speed_pixels_per_second"], linewidth=0.8)
    axes[2].set_ylabel("Speed\n(pixels/s)")

    status_codes = analysis_table["tracking_status"].map(STATUS_CODES)
    axes[3].step(frame_numbers, status_codes, where="mid", color="#9CA3AF", linewidth=0.6)
    for status, code, color, size in STATUS_STYLES:
        selected = analysis_table["tracking_status"].eq(status)
        axes[3].scatter(
            frame_numbers[selected],
            np.full(int(selected.sum()), code),
            color=color,
            s=size,
            label=status,
        )
    axes[3].set_yticks([0, 1, 2], labels=["invalid", "warning", "valid"])
    axes[3].set_ylim(-0.25, 2.25)
    axes[3].set_xlabel("Frame (1-based)")
    axes[3].legend(loc="lower right", ncol=3)

    figure.tight_layout()
    return figure


def plot_analysis(
    analysis_table: pd.DataFrame,
    frame_numbers: np.ndarray,
    include_tracking: bool,
) -> Figure:
    """Select the diameter-only or full tracking layout."""
    if include_tracking:
        return plot_diameter_and_tracking(analysis_table, frame_numbers)
    return plot_diameter(analysis_table, frame_numbers)
