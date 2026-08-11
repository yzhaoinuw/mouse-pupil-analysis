"""Assemble and write the user-facing analysis table."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pupil_tracking.extract_frames import ExtractedFrame
from pupil_tracking.plotting import plot_analysis

logger = logging.getLogger(__name__)

DIAMETER_COLUMNS = ["image_name", "estimated_pupil_diameter"]

VELOCITY_COLUMNS = [
    "image_name",
    "estimated_pupil_diameter",
    "timestamp_seconds",
    "center_x_pixels",
    "center_y_pixels",
    "speed_pixels_per_second",
    "tracking_status",
    "quality_reason",
]


def tracking_status(tracking_dataframe: pd.DataFrame) -> pd.Series:
    """Reduce detailed quality evidence to valid, warning, or invalid."""
    valid = tracking_dataframe["segmentation_valid"].astype(bool)
    reasons = tracking_dataframe["quality_reason"].fillna("").astype(str)
    return pd.Series(
        np.select(
            [~valid, reasons.ne("")],
            ["invalid", "warning"],
            default="valid",
        ),
        index=tracking_dataframe.index,
        dtype="object",
    )


def build_analysis_table(
    results,
    image_frames: list[ExtractedFrame],
    tracking_dataframe: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the compact output table and its one-based frame numbers.

    Returns the table exactly as it should be written to CSV, plus the displayed
    source-frame numbers that every plot panel shares as its x axis.
    """
    source_index_by_name = {
        frame.image_path.name: frame.source_frame_index for frame in image_frames
    }
    missing = [name for name, _ in results if name not in source_index_by_name]
    if missing:
        raise KeyError(
            f"{len(missing)} prediction(s) have no matching extracted frame, "
            f"starting with {missing[0]!r}."
        )

    result_rows = [
        {
            "image_name": name,
            "estimated_pupil_diameter": diameter,
            "source_frame_index": source_index_by_name[name],
        }
        for name, diameter in results
    ]
    result_dataframe = pd.DataFrame(result_rows).sort_values(
        "source_frame_index",
        kind="stable",
    )

    if tracking_dataframe is None:
        output_dataframe = result_dataframe[DIAMETER_COLUMNS].reset_index(drop=True)
        frame_numbers = result_dataframe["source_frame_index"].to_numpy(dtype=int) + 1
        return output_dataframe, frame_numbers

    ordered_tracking = (
        tracking_dataframe.set_index("image_name").loc[result_dataframe["image_name"]].reset_index()
    )
    ordered_tracking["tracking_status"] = tracking_status(ordered_tracking)
    ordered_tracking["quality_reason"] = ordered_tracking["quality_reason"].fillna("").astype(str)
    output_dataframe = ordered_tracking[VELOCITY_COLUMNS]
    frame_numbers = ordered_tracking["source_frame_index"].to_numpy(dtype=int) + 1
    return output_dataframe, frame_numbers


def write_analysis_outputs(
    results,
    image_frames: list[ExtractedFrame],
    result_dir: Path,
    exp_name: str,
    tracking_dataframe: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Write one compact analysis table and one frame-indexed plot.

    Returns the written table alongside both output paths, so a caller does not
    have to read the CSV back to inspect the results.
    """
    analysis_table, frame_numbers = build_analysis_table(
        results,
        image_frames,
        tracking_dataframe=tracking_dataframe,
    )

    result_dir = Path(result_dir)
    csv_path = result_dir / f"{exp_name}_pupil_analysis.csv"
    analysis_table.to_csv(csv_path, index=False)

    figure = plot_analysis(
        analysis_table,
        frame_numbers,
        include_tracking=tracking_dataframe is not None,
    )
    plot_path = result_dir / f"{exp_name}_pupil_analysis.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)

    logger.info("Saved analysis CSV:  %s", csv_path)
    logger.info("Saved analysis plot: %s", plot_path)
    return analysis_table, csv_path, plot_path
