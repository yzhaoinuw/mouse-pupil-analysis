# Pupil Tracking Collaboration Guide

This repository packages a mouse pupil segmentation and pupil-diameter analysis pipeline.
It can extract frames from video, run a trained attention UNet on centered eye images,
and save estimated pupil diameters plus optional mask overlays.

## Project Shape

- `pupil_tracking/` is the installable Python package.
- `pupil_tracking/run_pupil_analysis.py` implements the main `run-pupil-analysis` CLI.
- `pupil_tracking/extract_frames.py` implements the `extract-frames` CLI and video sampling.
- `pupil_tracking/dataset.py` contains preprocessing, padding/resizing, dataset loading, and training augmentations.
- `pupil_tracking/unet.py` defines the segmentation model.
- `pupil_tracking/checkpoints/` holds packaged model checkpoints used by the default CLI.
- `run_train.py` is the current training script for local experiments.
- `labelme_json2png.py`, `check_augmentation.py`, and `make_gif.py` are utility scripts.
- `tests/` contains lightweight package and CLI smoke tests.
- `.github/workflows/ci.yml` runs lint, formatting, tests, wheel build, and wheel smoke checks.

## Environment

Use a dedicated Python environment. The local machine keeps miniconda environments under:

```powershell
C:\Users\yzhao\miniconda3\envs\
```

The miniconda environment for this project is named `pupil_tracking`, so run Python commands
from that environment:

```powershell
conda activate pupil_tracking
```

The package requires Python 3.10 or newer. Install editable development dependencies from the
repo root:

```powershell
pip install -e .[dev]
```

For runtime-only usage, `pip install -e .` is enough.

## Main Workflows

Run the full analysis pipeline from a video:

```powershell
run-pupil-analysis --video_path C:\path\to\movie.avi
```

Run analysis from an existing PNG frame folder:

```powershell
run-pupil-analysis --image_dir C:\path\to\frames
```

Extract frames only:

```powershell
extract-frames --video_path C:\path\to\movie.avi --out_dir C:\path\to\frames
```

Train locally after preparing `images_train/`, `masks_train/`, `images_validation/`,
and `masks_validation/`:

```powershell
python run_train.py
```

Training checkpoints are written to `checkpoints_exp/`.

## Development Checks

Run the same core checks used by CI:

```powershell
ruff check .
black --check .
pytest -q
```

Build smoke checks:

```powershell
python -m build
python -m pip install dist\*.whl
run-pupil-analysis --help
```

## Data And Artifact Hygiene

This repo often has large local image, mask, prediction, checkpoint, and build artifacts.
Before committing, inspect `git status --short` carefully and avoid adding generated outputs
unless they are intentionally part of the release or training data update.

Common generated/local folders include:

- `images_test_*`, `images_*_result`, and `predicted_masks_*`
- `images_train`, `images_validation`, `masks_train`, and `masks_validation`
- `checkpoints_exp`
- `dist`, `*.egg-info`, `.pytest_cache`, and `.ruff_cache`

The tracked package checkpoint under `pupil_tracking/checkpoints/` is part of the installed
package and is used by `run-pupil-analysis` when no explicit `--checkpoint` is provided.

## Collaboration Notes

- Read `README.md` first for user-facing behavior and expected CLI outputs.
- Keep changes scoped; this repo mixes package code, training scripts, and local experiment assets.
- Prefer package imports such as `pupil_tracking.dataset` and `pupil_tracking.unet` in new code.
- Preserve the 148 x 148 padded image convention unless intentionally changing model assumptions.
- The inference path chooses the packaged checkpoint with the highest IoU encoded in its filename.
- Be careful with branch state: local branches may lag remote branches, and generated artifacts may
  make `git status` noisy.

## Commit Style

Use a short title line. Add a short body with flat bullets when a commit contains multiple
requested changes. For feature work, describe user-facing behavior and avoid mentioning internal
tests or project-memory updates unless those are the point of the commit.
