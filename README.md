# mouse-pupil-analysis

[![PyPI](https://img.shields.io/pypi/v/mouse-pupil-analysis.svg)](https://pypi.org/project/mouse-pupil-analysis/)
[![Python](https://img.shields.io/pypi/pyversions/mouse-pupil-analysis.svg)](https://pypi.org/project/mouse-pupil-analysis/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21897795.svg)](https://doi.org/10.5281/zenodo.21897795)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/LICENSE)

`mouse-pupil-analysis` segments mouse pupils from video or PNG frames and reports pupil diameter, with optional pupil-center tracking, velocity, quality control, and confidence overlays. It includes a trained UNet checkpoint, a command-line interface, and a Python API.

| I want to... | Go to |
|---|---|
| Install the package and analyze a video | [Quick start](#quick-start) |
| Check whether my images are suitable | [Input requirements](#input-requirements) |
| Try the repository's real sample images | [Included sample data](#included-sample-data) |
| Track pupil-center position and velocity | [Velocity and overlays](#velocity-and-overlays) |
| Adjust paths, thresholds, or extraction | [CLI reference](#cli-reference) |
| Call the pipeline from Python | [Python API](#python-api) |
| Interpret the generated files and units | [Outputs and interpretation](#outputs-and-interpretation) |
| Install PyTorch differently or troubleshoot | [Installation options](#installation-options) |
| Cite, train, test, or contribute | [Citation](#citation) and [Development](#development) |

## Quick start

Requires Python 3.10 or newer.

```bash
pip install mouse-pupil-analysis
run-pupil-analysis --video_path /path/to/movie.avi
```

The model checkpoint is included. A default video run extracts sampled frames to `movie_frames/` and writes the analysis table and plot to `movie_result/`:

```text
movie_result/
    movie_pupil_analysis.csv
    movie_pupil_analysis.png
```

If you already have PNG frames, analyze the folder directly:

```bash
run-pupil-analysis --image_dir /path/to/frames
```

That is enough for a standard diameter analysis. See [Velocity and overlays](#velocity-and-overlays) only if you need pupil-center tracking, speed, or per-frame visual quality checks.

![Pupil analysis pipeline demo](https://raw.githubusercontent.com/yzhaoinuw/mouse-pupil-analysis/main/media/pupil_diameter_analysis_result_demo.gif)

<p align="center"><em>Confidence-colored segmentation and estimated center (left); pupil diameter, center, speed, and quality over time (right).</em></p>

## Input requirements

The packaged model was trained on centered mouse-eye images. For reliable results:

- keep the eye near the center and make it occupy most of the frame;
- avoid major changes in crop, camera angle, illumination, and focus relative to the training domain;
- use the confidence overlays and quality fields to inspect unfamiliar recordings.

Every input image is resized with its aspect ratio preserved and center-padded to the model's 148 x 148 input size. The pipeline does not locate and crop a distant eye within a much larger scene.

## Included sample data

A repository checkout includes a compact set of real images and hand-labeled masks under [`sample_data/`](https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/sample_data/README.md). From the repository root, run:

```bash
run-pupil-analysis \
  --image_dir sample_data/velocity_frames \
  --result_dir results/sample_velocity \
  --output_mask_dir results/sample_velocity/overlays \
  --calculate_velocity \
  --acquisition_fps 97
```

The sample is for workflow exploration and regression testing, not scientific model evaluation or useful model training. The [sample-data guide](https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/sample_data/README.md) includes simpler uncropped-frame and paired-mask examples.

## Velocity and overlays

Diameter mode samples video frames. Velocity mode analyzes every encoded source frame and uses the experimental acquisition rate to calculate timestamps and speed:

```bash
run-pupil-analysis \
  --video_path data/mouse1.avi \
  --calculate_velocity \
  --acquisition_fps 33.3333333333 \
  --output_mask_dir data/mouse1_overlays
```

Use the actual acquisition rate when it differs from the video container's playback rate. For `--image_dir`, `--acquisition_fps` is required in velocity mode; for video input, it defaults to the video header rate when omitted.

Overlays show threshold-passing pupil pixels from yellow through orange to red as confidence increases. In velocity mode, a cyan cross marks an accepted center and a yellow cross marks a rejected candidate.

For the method from segmentation probabilities through center quality control and velocity, see [`project_overview.md`](https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/project_overview.md#segmentation-to-velocity-method). The README intentionally does not duplicate that internal methodology.

## CLI reference

Common commands:

```bash
# Analyze a video and choose output locations
run-pupil-analysis \
  --video_path data/mouse1.avi \
  --out_dir data/mouse1_frames \
  --result_dir data/mouse1_result

# Analyze existing PNG frames with a stricter segmentation threshold
run-pupil-analysis \
  --image_dir data/mouse1_frames \
  --result_dir data/mouse1_result \
  --pred_thresh 0.75

# Extract frames without running analysis
extract-frames \
  --video_path data/mouse1.avi \
  --out_dir data/mouse1_frames
```

Key options:

| Option | Purpose |
|---|---|
| `--video_path` | Analyze a video and extract frames automatically. Mutually exclusive with `--image_dir`. |
| `--image_dir` | Analyze an existing directory of PNG frames. |
| `--out_dir` | Override the extracted-frame directory. |
| `--result_dir` | Override the CSV and plot directory. |
| `--output_mask_dir` | Write confidence-overlay PNGs for visual inspection. |
| `--pred_thresh` | Adjust the pupil-pixel confidence threshold; raise it for over-segmentation and lower it for incomplete segmentation. |
| `--extraction_fps` | Set sampled extraction rate in diameter mode; default is 5 FPS. |
| `--max_frames` | Limit extracted frames; default is 10,000. |
| `--calculate_velocity` | Analyze every source frame and add center, speed, and quality fields. |
| `--acquisition_fps` | Set the experimental sampling rate used for velocity timestamps. |
| `--checkpoint` | Use a custom compatible checkpoint instead of the packaged model. |
| `--num_workers` | Set data-loader workers; use `0` for the main process. |

Run `run-pupil-analysis --help` or `extract-frames --help` for the complete option list.

## Python API

The Python API returns the analysis table as a pandas DataFrame while writing the same files as the CLI:

```python
from mouse_pupil_analysis import analyze_video

result = analyze_video("data/mouse1.avi")
print(result.analysis_table.head())
print(result.csv_path, result.plot_path)
```

All CLI settings are available as keyword arguments:

```python
result = analyze_video(
    "data/mouse1.avi",
    calculate_velocity=True,
    acquisition_fps=33.3333333333,
    output_mask_dir="data/mouse1_overlays",
)

usable = result.analysis_table.query("tracking_status != 'invalid'")
```

For existing frames, use `analyze_frames`:

```python
from mouse_pupil_analysis import analyze_frames

result = analyze_frames(
    "data/mouse1_frames",
    calculate_velocity=True,
    acquisition_fps=33.3333333333,
)
```

`AnalysisResult` exposes:

| Field | Description |
|---|---|
| `analysis_table` | The compact DataFrame also written to CSV. |
| `csv_path`, `plot_path` | Paths to the generated outputs. |
| `tracking_dataframe` | Detailed quality evidence in velocity mode; otherwise `None`. |
| `image_frames` | Frame metadata linking image names to source-frame indices. |

For reusable configuration, import `AnalysisConfig` and `run_analysis`. Library calls are quiet by default and use standard Python logging.

## Outputs and interpretation

| Output | Contents |
|---|---|
| `*_pupil_analysis.csv` | Image name and pupil diameter. Velocity mode adds timestamp, accepted center coordinates, speed, `tracking_status`, and `quality_reason`. |
| `*_pupil_analysis.png` | Frame-indexed diameter plot. Velocity mode adds center, speed, and quality-control panels. |
| Overlay PNGs | Optional confidence heatmaps and center markers written by `--output_mask_dir`. |

Diameter columns:

- `estimated_pupil_diameter` is the equivalent-circle diameter in the 148 x 148 model image. Keep it only for continuity with earlier results.
- `pupil_diameter_input_pixels` maps the measurement back to the scale of the supplied image. With video input, that is the source frame; with `--image_dir`, it is whatever crop or resize you supplied.

Neither value is physically calibrated. Comparisons across recordings require matching optics and working distance or a per-recording scale factor. Pupil-center coordinates use input-image pixels, and speed uses input-image pixels per second.

In velocity mode, `tracking_status` is `valid`, `warning`, or `invalid`. Warnings remain usable; rejected centers and speeds are blank. Speed is also blank across invalid or nonconsecutive frames—the pipeline does not interpolate across gaps.

## Installation options

### Isolated environment

A dedicated environment is recommended but not required:

```bash
conda create -n mouse_pupil_analysis python=3.12
conda activate mouse_pupil_analysis
pip install mouse-pupil-analysis
```

The distribution is `mouse-pupil-analysis`, the import is `mouse_pupil_analysis`, and the commands are `run-pupil-analysis` and `extract-frames`. Conda environment names are local and can be anything.

### PyTorch CPU and GPU builds

The ordinary install is sufficient on Windows and macOS. On Linux without an NVIDIA GPU, installing the CPU build first avoids downloading a CUDA-enabled PyTorch wheel:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install mouse-pupil-analysis
```

For an NVIDIA GPU, use the [official PyTorch selector](https://pytorch.org/get-started/locally/) to choose a build compatible with your driver, then install this package. Inference automatically uses CUDA when available and otherwise uses the CPU.

### Quick troubleshooting

- **Command not found:** activate the environment where you installed the package, then reinstall with `python -m pip install --upgrade mouse-pupil-analysis`.
- **Poor segmentation:** first check centering, crop, illumination, and focus; then write overlays and adjust `--pred_thresh`.
- **Unexpectedly large Linux download:** use the CPU-only PyTorch installation above if you do not need CUDA.

## Citation

If you use this software in scholarly work, cite the exact version you analyzed with:

> Zhao, Y. (2026). *mouse-pupil-analysis: Automated mouse pupil segmentation, diameter, and pupil-center velocity analysis using UNet* (v0.2.0). Zenodo. https://doi.org/10.5281/zenodo.21897796

The [version DOI](https://doi.org/10.5281/zenodo.21897796) identifies this release. The [concept DOI](https://doi.org/10.5281/zenodo.21897795) resolves to the collection of all releases. GitHub can also generate BibTeX from [`CITATION.cff`](https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/CITATION.cff). See [`CHANGELOG.md`](https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/CHANGELOG.md) for differences between versions.

## Development

Install a repository checkout with contributor tools:

```bash
git clone https://github.com/yzhaoinuw/mouse-pupil-analysis.git
cd mouse-pupil-analysis
pip install -e ".[dev]"
```

Run the core checks before submitting changes:

```bash
ruff check .
black --check .
pytest -q
```

Detailed guides are kept in the documents that own them:

- [Model training and dataset preparation](https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/training/README.md)
- [Runtime architecture and repository boundaries](https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/project_overview.md)
- [Sample-data provenance and examples](https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/sample_data/README.md)
- [Demo generation and review](https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/media/README.md)
- [Release procedure](https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/RELEASING.md)

This avoids duplicating internal architecture or training procedures in the user-facing README.

## License

Released under the [MIT License](https://github.com/yzhaoinuw/mouse-pupil-analysis/blob/main/LICENSE).

[![Agent Collab Treaty adopted](https://raw.githubusercontent.com/yzhaoinuw/agent_collab_treaty/main/assets/treaty-adopted.svg)](https://github.com/yzhaoinuw/agent_collab_treaty)
