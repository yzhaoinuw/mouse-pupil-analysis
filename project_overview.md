# Project Overview

## What This Repo Is

This repository packages a mouse pupil segmentation, pupil-diameter, and opt-in pupil-center velocity analysis pipeline. It can extract sampled or consecutive full-frame images from video, run a trained attention UNet on centered eye images, and save pupil size, tracking, quality-control, and optional mask-overlay outputs.

The installable package is `pupil_tracking`. The user-facing command line tools are `run-pupil-analysis` and `extract-frames`, both declared in `pyproject.toml`.

## Active Runtime Path

### 1. `pyproject.toml`

- Declares the `pupil-tracking` package, Python `>=3.10`, runtime dependencies, dev tools, package data, and console scripts.
- The console scripts are:
  - `extract-frames = pupil_tracking.extract_frames:main`
  - `run-pupil-analysis = pupil_tracking.run_pupil_analysis:main`
- Package data includes `pupil_tracking/checkpoints/*.pth` and `pupil_tracking/checkpoints/*.txt`; archive checkpoints are excluded.

### 2. `pupil_tracking/run_pupil_analysis.py`

- Implements the main analysis CLI.
- Accepts either `--video_path` or `--image_dir`.
- If a video is provided, calls `extract_selected_frames(...)` before inference.
- Loads `UNet(use_attention=True)`, picks CUDA when available, and falls back to CPU.
- Finds the default packaged checkpoint by selecting the highest IoU encoded in a checkpoint filename under `pupil_tracking/checkpoints/`.
- Writes `*_estimated_pupil_diameter.csv` and `*_estimated_pupil_diameter.png` into the result directory.
- With `--calculate_velocity`, analyzes consecutive source frames using an explicit acquisition timebase and writes tracking CSV/QC outputs.
- Optionally writes mask overlays when `--output_mask_dir` is provided.

### 3. `pupil_tracking/extract_frames.py`

- Implements the frame-extraction CLI and reusable `extract_selected_frames(...)`.
- Samples evenly spaced frames from the input video with OpenCV.
- Can extract consecutive source frames and return source-frame metadata for velocity analysis.
- Honors `--extraction_fps` and `--max_frames`, reducing effective extraction FPS when needed.

### 4. `pupil_tracking/tracking.py`

- Postprocesses UNet probability maps without changing the model or checkpoint.
- Selects and measures pupil components, maps centers back to original-image pixels, applies explainable quality flags, and calculates frame-to-frame displacement and velocity.
- Leaves published centers and velocities missing across rejected or non-consecutive frames rather than interpolating.

### 5. `pupil_tracking/dataset.py`

- Handles image loading, preprocessing, padding/resizing, dataset construction, and training augmentations.
- The 148 x 148 centered/padded image convention is load-bearing for the current trained model.

### 6. `pupil_tracking/unet.py`

- Defines the segmentation model used by inference and training.

### 7. `run_train.py`

- Current local training script.
- Expects training and validation image/mask folders in the repository root.
- Writes experimental checkpoints to `checkpoints_exp/`.

## Repo Structure Map

```text
pupil_tracking/
|- pupil_tracking/
|  |- run_pupil_analysis.py
|  |- extract_frames.py
|  |- tracking.py
|  |- dataset.py
|  |- unet.py
|  |- checkpoints/
|- tests/
|  |- test_imports.py
|  |- test_cli_help.py
|  |- test_extract_frames.py
|  |- test_tracking.py
|- .github/workflows/ci.yml
|- pyproject.toml
|- README.md
|- run_train.py
|- labelme_json2png.py
|- check_augmentation.py
|- make_gif.py
|- AGENTS.md
|- project_overview.md
|- next_steps.md
|- work_log.md
|- work_log_archive/
```

## What Looks Active vs. Legacy

Active, package-facing files:

- `pupil_tracking/run_pupil_analysis.py`
- `pupil_tracking/extract_frames.py`
- `pupil_tracking/tracking.py`
- `pupil_tracking/dataset.py`
- `pupil_tracking/unet.py`
- `pupil_tracking/checkpoints/`
- `tests/`
- `.github/workflows/ci.yml`
- `pyproject.toml`
- `README.md`

Active local/developer scripts:

- `run_train.py`
- `labelme_json2png.py`
- `check_augmentation.py`
- `make_gif.py`

Local/generated surfaces to treat carefully:

- `images_test_*`, `images_*_result`, `predicted_masks_*`, `predictions_test/`, and `results/` are local analysis outputs.
- `images_train/`, `images_validation/`, `masks_train/`, and `masks_validation/` are local training/validation data folders.
- `checkpoints_exp/` holds experimental training checkpoints.
- `build/`, `dist/`, `*.egg-info`, `__pycache__/`, `.pytest_cache/`, and `.ruff_cache/` are generated.
- `archive/` and `sketch*.py` style files are not the active package path.
- `pupil_tracking/checkpoints/archive/` contains historical checkpoint files excluded from package data.

## Tests And Fixtures

The test suite is intentionally lightweight:

- `tests/test_imports.py` imports the package.
- `tests/test_cli_help.py` runs `run-pupil-analysis --help`.
- `tests/test_extract_frames.py` verifies sampled/full-frame source-index selection.
- `tests/test_tracking.py` verifies coordinate mapping, component measurements, quality flags, timing, and kinematics with synthetic inputs.
- `.github/workflows/ci.yml` runs Ruff, Black, Pytest, and a wheel smoke check across Python 3.10, 3.11, and 3.12 where appropriate.

There is no small canonical sample video committed for end-to-end inference. Local image, mask, and prediction folders are useful for development, but they should not be assumed to be portable fixtures.

## User Data Expectations

The model expects the eye/pupil region to be centered enough that the majority of the eye appears in a 148 x 148 pixel area in the center of each frame. This matters because the trained model uses the current 148 x 148 padded image convention.

Primary input modes:

- `--video_path`: video file accepted by OpenCV; frames are extracted before inference.
- `--image_dir`: folder of PNG frames; extraction is skipped.

Primary outputs:

- `*_estimated_pupil_diameter.csv`
- `*_estimated_pupil_diameter.png`
- Velocity-mode `*_pupil_tracking.csv`
- Velocity-mode `*_pupil_tracking_qc.png`
- Optional mask overlay PNGs under `--output_mask_dir`

## Practical Mental Model

For most maintenance tasks, read in this order:

1. `AGENTS.md`
2. `README.md`
3. `pyproject.toml`
4. `pupil_tracking/run_pupil_analysis.py`
5. `pupil_tracking/extract_frames.py`
6. `pupil_tracking/dataset.py`
7. `tests/` and `.github/workflows/ci.yml`

## Questions Worth Clarifying Later

- Whether any local image/mask folders should become small, portable test fixtures.
- Whether `archive/`, root sketch scripts, and local experiment outputs should stay in the working tree or be moved elsewhere.
- Whether the training workflow should be made package-facing or remain a local maintainer workflow.
- Whether README examples should include a tiny smoke-test dataset once one exists.
