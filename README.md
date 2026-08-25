# mouse-pupil-analysis

[![Agent Collab Treaty adopted](https://raw.githubusercontent.com/yzhaoinuw/agent_collab_treaty/main/assets/treaty-adopted.svg)][treaty]
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21897795-1682D4.svg)][doi-concept]

![Pupil analysis pipeline demo][demo]

*Left: confidence-colored pupil mask and estimated center. Right: pupil diameter, center, and speed over time.*

Measure mouse pupil diameter from a video or a folder of frames, in one command. A trained
UNet ships inside the package, so there is no model to download, no config to write, and
nothing to label. It targets the low-contrast, low-resolution frames that rodent eye cameras
actually produce, and it also tracks the pupil center, its speed, and per-frame segmentation
quality.

## Contents

- [Install](#install)
- [Usage](#usage)
- [Output](#output)
- [CLI reference](#cli-reference)
- [Python API](#python-api)
- [Sample data](#sample-data)
- [Installation options](#installation-options)
- [FAQ](#faq)
- [Citation](#citation)
- [Development](#development)
- [License](#license)

## Install

```bash
pip install mouse-pupil-analysis
```

Python 3.10 or newer. The trained checkpoint ships with the package, so that is the entire
install. On Linux, `pip` pulls a CUDA-enabled PyTorch by default; see
[Installation options](#installation-options) to skip the extra gigabytes or to target a
specific CUDA version.

## Usage

Analyze a video:

```bash
run-pupil-analysis --video_path movie.avi
```

Or a folder of PNG frames you already have:

```bash
run-pupil-analysis --image_dir movie_frames/
```

Add `--calculate_velocity` to track the pupil center and its speed alongside diameter:

```bash
run-pupil-analysis \
  --video_path movie.avi \
  --calculate_velocity \
  --acquisition_fps 33.3333333333
```

That is the whole thing. Video frames are sampled at 5 fps by default, velocity mode
analyzes every encoded frame instead, and inference uses a GPU when one is available.

> **Input requirement.** The eye should be roughly centered and fill most of the frame.
> Every image is resized with its aspect ratio preserved and center-padded to the model's
> 148 x 148 input, so a small eye inside a wide scene has to be cropped first.

## Output

Results land next to the input, in `<video_stem>_result/` for a video and
`<image_dir>_result/` for a frame directory:

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

| File | Contents |
|------|--------------|
| `*_pupil_analysis.csv` | The per-frame table, columns below. |
| `*_pupil_analysis.png` | Frame-indexed diameter plot with valid/warning/invalid points. Velocity mode adds center, speed, and a dedicated quality-control panel. |
| Mask images in `--output_mask_dir` | Optional. Confidence heatmaps over threshold-passing pupil pixels: yellow near the threshold, orange intermediate, red near-certain. A thin center cross is cyan when accepted and yellow when rejected. |

Segmentation visibility and quality are written for every run. Timestamp, center, speed,
and `tracking_status` are added in velocity mode:

| Column | Meaning |
|---|---|
| `image_name` | Source image. The number in a generated name is the one-based source-frame index. |
| `estimated_pupil_diameter` | Equivalent-circle diameter, `sqrt(4 / pi * area)`, in the 148 x 148 model image. |
| `pupil_diameter_input_pixels` | The same diameter at the scale of the image you supplied. |
| `pupil_visibility` | `visible`, `not_detected`, `uncertain`, or `partially_visible_or_uncertain`. Shape-based uncertainty does not claim to reconstruct a pupil hidden by the eyelid. |
| `segmentation_status` | Diameter-only status: `valid`, `warning`, or `invalid`. |
| `quality_reason` | Which check flagged or rejected the segmentation. |
| `timestamp_seconds` | Derived from the source-frame index and `--acquisition_fps`. |
| `center_x_pixels`, `center_y_pixels` | Pupil center in input-image pixels; x increases right, y increases down. |
| `speed_pixels_per_second` | Center speed, in input-image pixels per second. |
| `tracking_status` | `valid`, `warning`, or `invalid`. |

No column is calibrated; see the [FAQ](#faq) before comparing recordings.
[Segmentation-to-velocity method][method] documents how the center and quality fields are
derived.

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
| `--pred_thresh` | Optional confidence-threshold override from 0 to 1. By default, inference uses calibrated metadata beside the checkpoint, then a threshold encoded in its filename, and finally `0.7` for an uncalibrated custom checkpoint. Increase it when segmentation overpredicts the pupil; reduce it when it finds only part of the pupil. |
| `--prefer_central_component` | Keep one confidence-weighted component with a gentle preference for the image centre. Off by default. It preserves occluded/crescent pupil shapes and is useful when a separate off-centre dark structure is falsely segmented. |
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
| `prediction_threshold` | The explicit or checkpoint-calibrated threshold actually used. |
| `segmentation_dataframe` | Detailed per-frame component and visibility evidence for every run. |
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
from two recordings and paired training and validation masks, which the
[sample-data guide][sample-data] covers. It is sized for trying the workflow and for
debugging, not for scientific model evaluation or useful training.

## Installation options

### Virtual environments

Recommended but not required, for example with [Miniconda][miniconda]:

```bash
conda create -n mouse_pupil_analysis python=3.12
conda activate mouse_pupil_analysis
pip install mouse-pupil-analysis
```

### CPU and GPU builds of PyTorch

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

## FAQ

**The pupil mask looks wrong. What do I check first?**

Framing, before anything else. A small or off-center eye is the most common cause, so
confirm the input requirement under [Usage](#usage) first. If the framing is right, write
the confidence overlays and look at them:

```bash
run-pupil-analysis --video_path movie.avi --output_mask_dir movie_overlays
```

Each overlay shades the pupil pixels that passed the threshold, from yellow (just over the
threshold) through orange to red (near-certain). Inference normally uses the calibration
stored with the checkpoint. If the mask spills past the pupil, override it with a higher
`--pred_thresh`; if it catches only part of the pupil, try a lower value.

`pupil_visibility` and the status/reason columns distinguish a clean segmentation from an
empty, low-confidence, border-touching, or low-circularity candidate. A narrow visible sliver
may be marked `partially_visible_or_uncertain`; a fully hidden pupil is not reconstructed.

**What should I pass for `--acquisition_fps`?**

The rate your camera actually acquired at. Velocity mode derives timestamps from the
source-frame index and this value, not from the frame rate stored in the video container,
which often is not experimental time. With `--image_dir` there is no container to fall back
on, so the flag is required; with `--video_path` it defaults to the container rate.

**Can I compare diameters or speeds between recordings?**

Only when the optics and working distance match, or after applying your own per-recording
scale factor. Nothing reported here is calibrated, and converting to millimeters needs a
scale factor this package cannot infer.

Within the CSV, prefer `pupil_diameter_input_pixels`. It inverts the resize-and-pad geometry
back to the scale of the image you supplied: the source video frame with `--video_path`, or
whatever you prepared with `--image_dir`. `estimated_pupil_diameter` is measured in the
148 x 148 model image, so it is not comparable across recordings that differ in resolution
or cropping; it is kept for continuity with earlier results. If your frames were already
148 x 148, the two columns are identical. Center and speed use the same input-image scale.

**Why are `center_x_pixels` and `speed_pixels_per_second` empty for some frames?**

Because the pipeline neither publishes nor interpolates values it cannot stand behind.
Center and speed are empty when segmentation is rejected, and speed is additionally empty
when either adjacent frame is invalid or when the two source frames are not consecutive. A
`warning` frame remains usable, such as extra foreground components when the selected pupil
component is still acceptable; `quality_reason` names the check that fired.

**What is the relationship to `pupil-tracking` on PyPI?**

None. That name belongs to an unrelated project by a different author, and it installs its
own `pupil_tracking` module, so this project deliberately claims no `pupil_tracking` import
namespace of any kind. Install `mouse-pupil-analysis`, import `mouse_pupil_analysis`, and
run `run-pupil-analysis` or `extract-frames`.

**Do I need to rename an existing conda environment?**

No. Environment names are local and have no bearing on which package is installed.

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

From a source checkout, model training and fine-tuning can also be run with terminal
arguments:

```bash
python training/run_train.py --help
```

Deeper documentation lives in the file that owns it:

| Topic | Document |
|---|---|
| Training, fine-tuning, and checkpoint packaging | [`training/README.md`][training] |
| Runtime architecture and the segmentation-to-velocity method | [`project_overview.md`][overview] |
| Sample-data provenance and examples | [`sample_data/README.md`][sample-data] |
| Regenerating the demo GIF | [`media/README.md`][media] |
| Release, PyPI, and Zenodo procedure | [`RELEASING.md`][releasing] |
| Conventions for agents working in this repo | [`AGENTS.md`][agents] |

## License

MIT. See [`LICENSE`][license].

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
