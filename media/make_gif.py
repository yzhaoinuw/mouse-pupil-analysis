"""Build the animated pupil-analysis demo used in the README."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from PIL import Image

# This script only writes a GIF; it never shows a window. Without this, pyplot picks
# an interactive backend at first figure creation and fails on any machine with no
# display or a broken Tk, such as a cluster node, a container, or SSH without X.
# Backend resolution is lazy, so setting it after the imports is still in time.
matplotlib.use("Agg")

MEDIA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MEDIA_DIR.parent
DEFAULT_RUN_DIR = MEDIA_DIR / "readme_demo"
DEFAULT_CSV = DEFAULT_RUN_DIR / "pupil_analysis_for_gif.csv"
DEFAULT_OVERLAY_DIR = DEFAULT_RUN_DIR / "overlays"
DEFAULT_OUTPUT = MEDIA_DIR / "pupil_diameter_analysis_result_demo.gif"
DEFAULT_START_FRAME = 7107
DEFAULT_END_FRAME = 7375
DEFAULT_SAMPLE_EVERY = 1
DEFAULT_FPS = 5.0
DEFAULT_PRED_THRESHOLD = 0.7
REQUIRED_COLUMNS = {
    "image_name",
    "estimated_pupil_diameter",
    "center_x_pixels",
    "center_y_pixels",
    "speed_pixels_per_second",
}
FRAME_NUMBER_PATTERN = re.compile(r"(\d+)(?=\.[^.]+$)")


def build_demo_gif_palette() -> Image.Image:
    """Preserve grayscale detail and the confidence continuum in the GIF."""
    grayscale = [
        (value, value, value) for value in np.rint(np.linspace(0, 255, 88)).astype(np.uint8)
    ]

    def confidence_color(normalized: float) -> tuple[int, int, int]:
        if normalized <= 0.5:
            green = 255.0 - 180.0 * normalized
        else:
            green = 330.0 * (1.0 - normalized)
        return 255, int(round(np.clip(green, 0, 255))), 0

    confidence_colors = []
    for normalized in np.linspace(0.0, 1.0, 48):
        confidence_colors.append(confidence_color(normalized))

    overlay_colors = []
    for normalized in np.linspace(0.0, 1.0, 6):
        heatmap_color = confidence_color(normalized)
        for gray in np.linspace(0.0, 200.0, 16):
            overlay_colors.append(
                tuple(
                    int(round(0.9 * gray + 0.1 * heatmap_channel))
                    for heatmap_channel in heatmap_color
                )
            )

    center_colors = [
        (
            int(round(0.65 * gray)),
            int(round(0.65 * gray + 0.35 * 255.0)),
            int(round(0.65 * gray + 0.35 * 255.0)),
        )
        for gray in np.linspace(0.0, 200.0, 16)
    ]

    plot_colors = [
        (43, 108, 176),
        (107, 70, 193),
        (221, 107, 32),
        (197, 48, 48),
        (75, 85, 99),
        (156, 163, 175),
        (0, 139, 139),
    ]
    colors = grayscale + confidence_colors + overlay_colors + center_colors + plot_colors
    colors.extend([(255, 255, 255)] * (256 - len(colors)))

    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette([channel for color in colors for channel in color])
    return palette_image


class DemoGifWriter(PillowWriter):
    """Pillow writer with a fixed palette suited to the demo figure."""

    def finish(self) -> None:
        palette_image = build_demo_gif_palette()
        frames = [
            frame.convert("RGB").quantize(
                palette=palette_image,
                dither=Image.Dither.NONE,
            )
            for frame in self._frames
        ]
        frames[0].save(
            self.outfile,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
            optimize=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the animated README demo from unified pupil-analysis outputs."
    )
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--overlay_dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start_frame", type=int, default=DEFAULT_START_FRAME)
    parser.add_argument("--end_frame", type=int, default=DEFAULT_END_FRAME)
    parser.add_argument("--sample_every", type=int, default=DEFAULT_SAMPLE_EVERY)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--pred_thresh", type=float, default=DEFAULT_PRED_THRESHOLD)
    return parser.parse_args()


def frame_number_from_name(image_name: str) -> int:
    match = FRAME_NUMBER_PATTERN.search(Path(str(image_name)).name)
    if match is None:
        raise ValueError(f"Could not read a frame number from {image_name!r}.")
    return int(match.group(1))


def padded_limits(*series: np.ndarray, padding_fraction: float = 0.08) -> tuple[float, float]:
    finite_values = np.concatenate([values[np.isfinite(values)] for values in series])
    if finite_values.size == 0:
        return 0.0, 1.0

    lower = float(finite_values.min())
    upper = float(finite_values.max())
    span = upper - lower
    padding = max(span * padding_fraction, 0.5)
    return lower - padding, upper + padding


def diagnostic_segment(values: np.ndarray, rejected: np.ndarray) -> np.ndarray:
    """Keep rejected samples and one accepted endpoint on each side."""
    visible = rejected.copy()
    visible[:-1] |= rejected[1:]
    visible[1:] |= rejected[:-1]
    return np.where(visible, values, np.nan)


def load_demo_data(
    csv_path: Path,
    overlay_dir: Path,
    start_frame: int,
    end_frame: int,
    sample_every: int,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    if sample_every < 1:
        raise ValueError("--sample_every must be at least 1.")
    if start_frame > end_frame:
        raise ValueError("--start_frame must be less than or equal to --end_frame.")

    dataframe = pd.read_csv(csv_path)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing_columns)}")

    dataframe = dataframe.copy()
    dataframe["frame_number"] = dataframe["image_name"].map(frame_number_from_name)
    dataframe = dataframe.loc[dataframe["frame_number"].between(start_frame, end_frame)].iloc[
        ::sample_every
    ]
    dataframe = dataframe.reset_index(drop=True)
    if len(dataframe) < 2:
        raise ValueError(
            "The requested frame range and sampling interval produced fewer than two frames."
        )

    images: list[np.ndarray] = []
    for image_name in dataframe["image_name"]:
        image_path = overlay_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing overlay image: {image_path}")
        with Image.open(image_path) as image:
            images.append(np.asarray(image.convert("RGB")))

    return dataframe, images


def create_demo_gif(
    dataframe: pd.DataFrame,
    images: list[np.ndarray],
    output_path: Path,
    fps: float,
    pred_thresh: float,
) -> None:
    if fps <= 0:
        raise ValueError("--fps must be positive.")
    if not 0 < pred_thresh < 1:
        raise ValueError("--pred_thresh must be between 0 and 1.")

    frame_numbers = dataframe["frame_number"].to_numpy(dtype=float)
    diameters = dataframe["estimated_pupil_diameter"].to_numpy(dtype=float)
    center_x = dataframe["center_x_pixels"].to_numpy(dtype=float)
    center_y = dataframe["center_y_pixels"].to_numpy(dtype=float)
    speeds = dataframe["speed_pixels_per_second"].to_numpy(dtype=float)
    diagnostic_columns = {
        "tracking_status",
        "raw_center_x_pixels",
        "raw_center_y_pixels",
        "raw_speed_pixels_per_second",
    }
    has_diagnostics = diagnostic_columns.issubset(dataframe.columns)
    rejected = np.zeros(len(dataframe), dtype=bool)
    rejected_center_x = np.full(len(dataframe), np.nan)
    rejected_center_y = np.full(len(dataframe), np.nan)
    rejected_speeds = np.full(len(dataframe), np.nan)
    if has_diagnostics:
        rejected = dataframe["tracking_status"].astype(str).eq("invalid").to_numpy()
        raw_center_x = dataframe["raw_center_x_pixels"].to_numpy(dtype=float)
        raw_center_y = dataframe["raw_center_y_pixels"].to_numpy(dtype=float)
        raw_speeds = dataframe["raw_speed_pixels_per_second"].to_numpy(dtype=float)
        rejected_center_x = diagnostic_segment(raw_center_x, rejected)
        rejected_center_y = diagnostic_segment(raw_center_y, rejected)
        rejected_intervals = rejected.copy()
        rejected_intervals[1:] |= rejected[:-1]
        rejected_speeds = diagnostic_segment(raw_speeds, rejected_intervals)
    has_rejected_diagnostics = bool(
        np.isfinite(rejected_center_x).any()
        or np.isfinite(rejected_center_y).any()
        or np.isfinite(rejected_speeds).any()
    )

    figure = plt.figure(figsize=(8.5, 5.2), layout="constrained")
    outer_grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.7))
    image_axis = figure.add_subplot(outer_grid[0, 0])
    trace_grid = outer_grid[0, 1].subgridspec(
        5,
        1,
        height_ratios=(0.8, 1.0, 1.0, 1.0, 0.55),
        hspace=0.10,
    )
    diameter_axis = figure.add_subplot(trace_grid[1, 0])
    center_axis = figure.add_subplot(trace_grid[2, 0], sharex=diameter_axis)
    speed_axis = figure.add_subplot(trace_grid[3, 0], sharex=diameter_axis)
    trace_axes = (diameter_axis, center_axis, speed_axis)

    image_display = image_axis.imshow(images[0])
    image_title = image_axis.set_title("", fontsize=11)
    image_axis.axis("off")
    confidence_colormap = LinearSegmentedColormap.from_list(
        "pupil_confidence",
        [(0.0, "#ffff00"), (0.5, "#ffa500"), (1.0, "#ff0000")],
    )
    confidence_axis = image_axis.inset_axes([0.08, -0.105, 0.42, 0.035])
    confidence_colorbar = figure.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=pred_thresh, vmax=1.0),
            cmap=confidence_colormap,
        ),
        cax=confidence_axis,
        orientation="horizontal",
    )
    confidence_colorbar.set_ticks([pred_thresh, 1.0])
    confidence_colorbar.set_ticklabels([f"{pred_thresh:.1f}", "1.0"])
    confidence_colorbar.ax.tick_params(labelsize=7, length=2, pad=1)
    confidence_colorbar.outline.set_linewidth(0.6)
    image_axis.text(
        0.29,
        -0.045,
        "Confidence",
        transform=image_axis.transAxes,
        ha="center",
        va="center",
        fontsize=8,
    )
    image_axis.plot(
        [0.63],
        [-0.075],
        marker="+",
        color="#008b8b",
        markersize=9,
        markeredgewidth=1.5,
        linestyle="None",
        transform=image_axis.transAxes,
        clip_on=False,
    )
    image_axis.text(
        0.68,
        -0.075,
        "Pupil center",
        transform=image_axis.transAxes,
        ha="left",
        va="center",
        fontsize=8,
    )

    (diameter_line,) = diameter_axis.plot([], [], color="#2b6cb0", linewidth=2.2)
    (diameter_dot,) = diameter_axis.plot([], [], "o", color="#2b6cb0", markersize=5)
    diameter_axis.set_ylabel("Diameter (pixel)")
    diameter_axis.set_title("Live Pupil Measurements", fontsize=12)
    diameter_axis.set_ylim(*padded_limits(diameters))

    (center_x_line,) = center_axis.plot([], [], color="#6b46c1", linewidth=2.0, label="x center")
    (center_y_line,) = center_axis.plot([], [], color="#dd6b20", linewidth=2.0, label="y center")
    (rejected_center_x_line,) = center_axis.plot(
        [],
        [],
        color="#6b46c1",
        linewidth=1.8,
        linestyle="--",
        alpha=0.85,
        label="rejected estimate" if has_rejected_diagnostics else "_nolegend_",
    )
    (rejected_center_y_line,) = center_axis.plot(
        [],
        [],
        color="#dd6b20",
        linewidth=1.8,
        linestyle="--",
        alpha=0.85,
    )
    (center_x_dot,) = center_axis.plot([], [], "o", color="#6b46c1", markersize=5)
    (center_y_dot,) = center_axis.plot([], [], "o", color="#dd6b20", markersize=5)
    center_axis.set_ylabel("Center (pixel)")
    center_axis.set_ylim(*padded_limits(center_x, center_y, rejected_center_x, rejected_center_y))
    center_axis.legend(
        loc="upper right",
        frameon=False,
        ncols=3 if has_rejected_diagnostics else 2,
        fontsize=7 if has_rejected_diagnostics else 8,
        handlelength=1.7,
        columnspacing=0.9,
    )

    (speed_line,) = speed_axis.plot([], [], color="#c53030", linewidth=2.0)
    (rejected_speed_line,) = speed_axis.plot(
        [],
        [],
        color="#c53030",
        linewidth=1.8,
        linestyle="--",
        alpha=0.85,
    )
    (speed_dot,) = speed_axis.plot([], [], "o", color="#c53030", markersize=5)
    speed_axis.set_ylabel("Speed (pixel/s)")
    speed_axis.set_xlabel("Frame")
    speed_lower, speed_upper = padded_limits(speeds, rejected_speeds)
    speed_axis.set_ylim(max(0.0, speed_lower), speed_upper)

    cursor_lines = []
    for axis in trace_axes:
        axis.set_xlim(float(frame_numbers.min()), float(frame_numbers.max()))
        axis.grid(color="#9ca3af", alpha=0.25, linewidth=0.7)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        cursor_lines.append(
            axis.axvline(frame_numbers[0], color="#4b5563", alpha=0.45, linewidth=1.0)
        )
    diameter_axis.tick_params(labelbottom=False)
    center_axis.tick_params(labelbottom=False)

    def update(frame_index: int):
        current_slice = slice(0, frame_index + 1)
        current_frame = frame_numbers[frame_index]

        image_display.set_array(images[frame_index])
        image_title.set_text(f"Frame {int(current_frame)}")

        diameter_line.set_data(frame_numbers[current_slice], diameters[current_slice])
        diameter_dot.set_data([current_frame], [diameters[frame_index]])

        center_x_line.set_data(frame_numbers[current_slice], center_x[current_slice])
        center_y_line.set_data(frame_numbers[current_slice], center_y[current_slice])
        rejected_center_x_line.set_data(
            frame_numbers[current_slice], rejected_center_x[current_slice]
        )
        rejected_center_y_line.set_data(
            frame_numbers[current_slice], rejected_center_y[current_slice]
        )
        center_x_dot.set_data([current_frame], [center_x[frame_index]])
        center_y_dot.set_data([current_frame], [center_y[frame_index]])

        speed_line.set_data(frame_numbers[current_slice], speeds[current_slice])
        rejected_speed_line.set_data(frame_numbers[current_slice], rejected_speeds[current_slice])
        speed_dot.set_data([current_frame], [speeds[frame_index]])

        for cursor_line in cursor_lines:
            cursor_line.set_xdata([current_frame, current_frame])

        return [
            image_display,
            image_title,
            diameter_line,
            diameter_dot,
            center_x_line,
            center_y_line,
            rejected_center_x_line,
            rejected_center_y_line,
            center_x_dot,
            center_y_dot,
            speed_line,
            rejected_speed_line,
            speed_dot,
            *cursor_lines,
        ]

    animation = FuncAnimation(
        figure,
        update,
        frames=len(images),
        interval=1000.0 / fps,
        blit=True,
        repeat_delay=1000,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        output_path,
        writer=DemoGifWriter(fps=fps),
        dpi=100,
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    dataframe, images = load_demo_data(
        csv_path=args.csv_path,
        overlay_dir=args.overlay_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        sample_every=args.sample_every,
    )
    create_demo_gif(dataframe, images, args.output, args.fps, args.pred_thresh)
    print(
        f"Saved {len(images)}-frame animated GIF to {args.output} "
        f"(source frames {int(dataframe['frame_number'].iloc[0])}-"
        f"{int(dataframe['frame_number'].iloc[-1])})."
    )


if __name__ == "__main__":
    main()
