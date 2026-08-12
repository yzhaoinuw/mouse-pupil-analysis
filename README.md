# mouse-pupil-analysis

[![PyPI](https://img.shields.io/pypi/v/mouse-pupil-analysis.svg)][pypi]
[![Python](https://img.shields.io/pypi/pyversions/mouse-pupil-analysis.svg)][pypi]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21897795.svg)][doi-concept]
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)][license]
[![Agent Collab Treaty adopted](https://raw.githubusercontent.com/yzhaoinuw/agent_collab_treaty/main/assets/treaty-adopted.svg)][treaty]

Measure mouse pupil diameter from a video or a folder of frames, in one command. A trained
UNet ships inside the package, so there is no model to download, no config to write, and
nothing to label. It targets the low-contrast, low-resolution frames that rodent eye
cameras actually produce. Optionally it also tracks the pupil center, its speed, and
per-frame segmentation quality.

![Pupil analysis pipeline demo][demo]

*Left: confidence-colored pupil mask and estimated center. Right: pupil diameter, center, and speed over time.*

## Contents

| If you want to... | Go to |
|---|---|
| Install it | [Install](#install) |
| Run it on a video or on frames | [Usage](#usage) |
| Know whether it will work on your recordings | [Will this work on my recordings?](#will-this-work-on-my-recordings) |
| Track the pupil center and its speed | [Pupil center and velocity](#pupil-center-and-velocity) |
| Try it before using your own data | [Sample data](#sample-data) |
| Look up a flag | [CLI reference](#cli-reference) |
| Call it from Python or a notebook | [Python API](#python-api) |
| Read the CSV, the plot, and the units | [Output](#output) |
| Pick a CPU or GPU PyTorch, or use a conda env | [Installation notes](#installation-notes) |
| Cite it | [Citation](#citation) |
| Train your own model, or contribute | [Development](#development) |

## Install

```bash
pip install mouse-pupil-analysis
```

Python 3.10 or newer. The trained checkpoint ships with the package, so that is the entire
install.

On **Linux**, `pip` pulls a CUDA-enabled PyTorch by default. See
[PyTorch CPU and GPU builds](#pytorch-cpu-and-gpu-builds) to skip the extra gigabytes, or to
target a specific CUDA version.

## Usage

Analyze a video:

```bash
run-pupil-analysis --video_path movie.avi
```

Or analyze a folder of PNG frames you already have:

```bash
run-pupil-analysis --image_dir movie_frames/
```

That is the whole thing. You get a per-frame pupil-diameter table and a plot, written next
to the input:

```text
movie.avi
movie_frames/                     # extracted frames (video input only)
    movie_00001.png
    movie_00002.png
    ...
movie_result/
    movie_pupil_analysis.csv
    movie_pupil_analysis.png
```

Frames are sampled from the video at 5 fps, up to 10,000 of them. Inference uses a GPU when
one is available and falls back to CPU otherwise. To change output locations, sampling, or
the segmentation threshold, see the [CLI reference](#cli-reference).

## Will this work on my recordings?

Probably, if the eye is roughly centered and fills most of the frame. The model was trained
on 148 x 148 crops of centered mouse eyes, and every input image is resized with its aspect
ratio preserved and center-padded to that size before inference. The pipeline does not hunt
for a small eye inside a wide scene, so crop first if that is your setup.

To check an unfamiliar recording, write the confidence overlays and look at them:

```bash
run-pupil-analysis --video_path movie.avi --output_mask_dir movie_overlays
```

Each overlay shades the pupil pixels that passed the threshold, from yellow (just over)
through orange to red (near-certain). If the mask spills past the pupil, raise
`--pred_thresh`; if it catches only part of the pupil, lower it.

## Pupil center and velocity

Add `--calculate_velocity` to also track where the pupil center is and how fast it moves:

```bash
run-pupil-analysis \
  --video_path movie.avi \
  --calculate_velocity \
  --acquisition_fps 33.3333333333
```

Two things change in this mode:

- **Every encoded frame is analyzed**, not a 5 fps sample, so the run takes longer.
- **Timestamps come from the source-frame index and `--acquisition_fps`**, not from the
  frame rate stored in the video container, which often is not experimental time. Pass the
  rate your camera actually acquired at. With `--image_dir` the flag is required; with
  `--video_path` it defaults to the container rate.

The CSV gains `timestamp_seconds`, `center_x_pixels`, `center_y_pixels`,
`speed_pixels_per_second`, `tracking_status`, and `quality_reason`, and the plot gains
matching panels. See [Output](#output) for how to read them, and
[Segmentation-to-velocity method][method] for the algorithm behind them.

## Sample data

A clone of the repository carries real images and hand-labeled masks under
[`sample_data/`][sample-data], so you can exercise the whole pipeline before pointing it at
your own recordings:

```bash
git clone https://github.com/yzhaoinuw/mouse-pupil-analysis.git
cd mouse-pupil-analysis

run-pupil-analysis \
  --image_dir sample_data/velocity_frames \
  --result_dir results/sample_velocity \
  --output_mask_dir results/sample_velocity/overlays \
  --calculate_velocity \
  --acquisition_fps 97
```

Those are 31 consecutive frames acquired at 97 Hz. The fixture also holds uncropped frames
from two recordings and paired training and validation masks; the
[sample-data guide][sample-data] covers those. It is sized for trying the workflow and for
debugging, not for scientific model evaluation or useful training.

## CLI reference

Two commands are installed: `run-pupil-analysis` (extraction plus analysis) and
`extract-frames` (extraction only). Run either with `--help` for the raw list.

```bash
# custom output locations, plus confidence overlays
run-pupil-analysis \
  --video_path data/mouse1.avi \
  --out_dir data/frames_mouse1 \
  --result_dir data/results_mouse1 \
  --output_mask_dir data/masks_mouse1

# existing frames, with a stricter segmentation threshold
run-pupil-analysis --image_dir data/mouse1_frames --pred_thresh 0.75

# extract frames without analyzing them
extract-frames --video_path data/mouse1.avi --out_dir data/frames_mouse1
```

| Flag | Description |
|---|---|
| `--video_path` | Input video. Frames are extracted automatically before analysis. Mutually exclusive with `--image_dir`. |
| `--image_dir` | Input directory of existing PNG frames, used instead of a video. |
| `--out_dir` | Where to write extracted frames. Defaults to `<video_stem>_frames/` next to the video. |
| `--result_dir` | Where to write the CSV and plot. Defaults to `<video_stem>_result/` for video input and `<image_dir>_result/` for `--image_dir`. |
| `--output_mask_dir` | Save translucent confidence-heatmap overlays for threshold-passing pupil pixels. Yellow is closest to the prediction threshold, orange is intermediate, and red is near-perfect confidence. |
| `--pred_thresh` | Confidence threshold from 0 to 1 for classifying a pixel as pupil (default `0.7`). A value of 0.7 means a pixel counts as pupil only when model confidence exceeds 0.7. Increase it when the segmentation overpredicts the pupil; reduce it when it finds only part of the pupil. |
| `--mask_transparency` | Blend weight of the heatmap color over the source image in overlays (default `0.1`). Higher values are more saturated. |
| `--extraction_fps` | Frames per second to extract from the video (default `5`). If extracting at this rate would exceed `--max_frames`, the rate is automatically reduced so that `--max_frames` frames are extracted. |
| `--max_frames` | Cap on frames extracted from a video (default `10000`). Useful for long recordings. |
| `--calculate_velocity` | Analyze every encoded source frame and append pupil-center, speed, and segmentation-quality fields and plot panels to the analysis outputs. |
| `--acquisition_fps` | Actual experimental sampling rate used for timestamps and velocity. Required with `--image_dir` in velocity mode; defaults to the video header rate for video input. |
| `--checkpoint` | Path to a custom model checkpoint. Defaults to the packaged checkpoint with the highest IoU encoded in its filename. |
| `--batch_size` | Inference batch size (default `32`). |
| `--num_workers` | Dataloader worker processes (default: up to 4, capped by CPU count). Use `0` to load frames in the main process, which is often faster for short recordings. |

## Python API

Everything the CLI does is available from Python, which is usually more convenient inside a
notebook or a larger analysis script. The results come back as a DataFrame, so there is no
need to read the CSV back in.

```python
from mouse_pupil_analysis import analyze_video

result = analyze_video("data/mouse1.avi")
print(result.analysis_table.head())
print(result.csv_path, result.plot_path)
```

Velocity mode and every CLI flag are keyword arguments:

```python
result = analyze_video(
    "data/mouse1.avi",
    calculate_velocity=True,
    acquisition_fps=33.3333333333,
    output_mask_dir="data/masks_mouse1",
)

usable = result.analysis_table.query("tracking_status != 'invalid'")
print(f"{len(usable)} of {len(result.analysis_table)} frames usable")
```

To start from frames you already extracted, use `analyze_frames` instead:

```python
from mouse_pupil_analysis import analyze_frames

result = analyze_frames(
    "data/mouse1_frames",
    calculate_velocity=True,
    acquisition_fps=33.3333333333,
)
```

`result` is an `AnalysisResult` with these fields:

| Field | Description |
|---|---|
| `analysis_table` | The same compact table written to CSV, as a DataFrame. |
| `csv_path`, `plot_path` | Locations of the written outputs. |
| `tracking_dataframe` | Detailed per-frame quality evidence in velocity mode, otherwise `None`. Retains raw centers, component areas, confidence, circularity, and temporal-area calculations that the compact table omits. |
| `image_frames` | Frame metadata linking each image name to its source-frame index. |

For repeated runs with shared settings, build an `AnalysisConfig` once and pass it to
`run_analysis`:

```python
from pathlib import Path

from mouse_pupil_analysis import AnalysisConfig, run_analysis

for video in Path("data").glob("*.avi"):
    run_analysis(AnalysisConfig(video_path=video, pred_thresh=0.75))
```

Library code logs rather than prints, so it stays quiet by default. To see the same progress
messages the CLI shows:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Output

| File | Description |
|------|--------------|
| `*_pupil_analysis.csv` | Unified table containing `image_name` and pupil diameter in both model and input-image pixels. Velocity mode appends timestamp, accepted x/y center, speed, three-state tracking status, and a concise quality reason. Generated image names contain the one-based source-frame number. |
| `*_pupil_analysis.png` | Unified frame-indexed plot. Velocity mode appends x/y center, speed, and valid/warning/invalid quality-control panels below pupil diameter. |
| Mask images in `--output_mask_dir` | Optional. PNGs with a translucent yellow-orange-red confidence heatmap over threshold-passing pupil pixels. Velocity mode also marks the raw pupil center with a thin translucent cross: cyan for accepted candidates and yellow for rejected candidates. |

### Units

Pupil diameter is reported twice, in two different units:

- `estimated_pupil_diameter` is measured in the 148 x 148 model image. Because every
  frame is rescaled to that size, this value is **not comparable between recordings**
  with different resolution or cropping. It is kept for continuity with earlier results.
- `pupil_diameter_input_pixels` inverts the resize-and-pad geometry to express the same
  measurement at the scale of **the image you supplied**. With `--video_path` that is the
  source video frame. With `--image_dir` it is whatever you prepared: if your frames were
  already cropped or resized to 148 x 148, this column equals `estimated_pupil_diameter`.

Both are equivalent-circle diameters: the diameter of a circle whose area matches the
segmented pupil mask, `sqrt(4 / pi * area)`.

Neither column is calibrated. `pupil_diameter_input_pixels` removes the model's rescaling,
but two recordings still only compare directly if their optics and working distance match.
Otherwise apply your own per-recording scale factor.

Pupil-center coordinates and speed use the same input-image pixel scale as
`pupil_diameter_input_pixels`: the source video frame with `--video_path`, and
whatever you supplied with `--image_dir`. The x coordinate increases to the right
and the y coordinate increases downward, and speed is in input-image pixels per
second. The same calibration caveat applies, so speeds are only directly
comparable between recordings whose optics and working distance match.

Neither unit is physical. Converting to millimeters requires a scale factor from your
own optics, which this package does not attempt to infer.

### Quality control

In velocity mode the CSV reports `tracking_status` as `valid`, `warning`, or `invalid`, with
`quality_reason` identifying suspicious or rejected frames.

Published center and speed fields are left empty when segmentation is rejected. Speed is
also left empty when either adjacent frame is invalid or when source frames are not
consecutive; the pipeline does not interpolate across these gaps. A warning remains usable,
such as extra foreground components when the selected pupil component is still acceptable.

## Installation notes

### A dedicated environment

Recommended but not required, for example with [Miniconda][miniconda]:

```bash
conda create -n mouse_pupil_analysis python=3.12
conda activate mouse_pupil_analysis
pip install mouse-pupil-analysis
```

Environment names are local, so an existing environment does not need to be renamed.

### PyTorch CPU and GPU builds

On **Windows and macOS**, the plain `pip install` gives you a CPU-only build of PyTorch,
which is all this package needs to run. No action required.

On **Linux**, the default PyPI wheel bundles CUDA and is several times larger. If you do not
have an NVIDIA GPU, install the CPU-only build first to avoid the download:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install mouse-pupil-analysis
```

To use an **NVIDIA GPU**, install a matching CUDA build first. This package requires
`torch>=2.8`, which is served by the `cu126`, `cu128`, and `cu129` indexes; older indexes
such as `cu124` stop at PyTorch 2.6 and will not satisfy that floor. Pick the index matching
your driver with the [official selector][pytorch-selector]:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install mouse-pupil-analysis
```

### A note on names

The repository and the distribution are `mouse-pupil-analysis`, the Python import is
`mouse_pupil_analysis`, and the console commands are `run-pupil-analysis` and
`extract-frames`. The shorter name `pupil-tracking` on PyPI belongs to an unrelated project
by a different author, and it installs its own `pupil_tracking` module, so this project
deliberately claims no `pupil_tracking` import namespace of any kind.

## Citation

If you use this software in a paper or other scholarly work, please cite the exact version
you ran, so your analysis stays reproducible against that code:

> Zhao, Y. (2026). *mouse-pupil-analysis: Automated mouse pupil segmentation, diameter, and
> pupil-center velocity analysis using UNet* (v0.2.0). Zenodo.
> https://doi.org/10.5281/zenodo.21897796

Every release is archived on Zenodo under its own [version DOI][doi-version]. The
[concept DOI][doi-concept] in the badge above always resolves to the newest version, so use
it only when you mean the project as a whole rather than a specific analysis.

GitHub's "Cite this repository" button generates BibTeX from [`CITATION.cff`][citation], and
[`CHANGELOG.md`][changelog] records what changed between versions.

## Development

```bash
git clone https://github.com/yzhaoinuw/mouse-pupil-analysis.git
cd mouse-pupil-analysis
pip install -e ".[dev]"
```

Run the same checks CI runs before submitting a change:

```bash
ruff check .
black --check .
pytest
```

Deeper documentation lives in the file that owns it:

| Topic | Document |
|---|---|
| Training, fine-tuning, and checkpoint promotion | [`training/README.md`][training] |
| Runtime architecture and the segmentation-to-velocity method | [`project_overview.md`][overview] |
| Sample-data provenance and examples | [`sample_data/README.md`][sample-data] |
| Regenerating the demo GIF | [`media/README.md`][media] |
| Release, PyPI, and Zenodo procedure | [`RELEASING.md`][releasing] |
| Conventions for agents working in this repo | [`AGENTS.md`][agents] |

## License

MIT. See [`LICENSE`][license].

[pypi]: https://pypi.org/project/mouse-pupil-analysis/
[doi-concept]: https://doi.org/10.5281/zenodo.21897795
[doi-version]: https://doi.org/10.5281/zenodo.21897796
[treaty]: https://github.com/yzhaoinuw/agent_collab_treaty
[demo]: https://raw.githubusercontent.com/yzhaoinuw/mouse-pupil-analysis/main/media/pupil_diameter_analysis_result_demo.gif
[license]: https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/LICENSE
[citation]: https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/CITATION.cff
[changelog]: https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/CHANGELOG.md
[sample-data]: https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/sample_data/README.md
[training]: https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/training/README.md
[overview]: https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/project_overview.md
[method]: https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/project_overview.md#segmentation-to-velocity-method
[media]: https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/media/README.md
[releasing]: https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/RELEASING.md
[agents]: https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/AGENTS.md
[miniconda]: https://www.anaconda.com/docs/getting-started/miniconda/install
[pytorch-selector]: https://pytorch.org/get-started/locally/
