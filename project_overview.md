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

### 2. `pupil_tracking/pupil_predictions.py`

- Owns packaged-checkpoint selection, PNG frame discovery, UNet inference, pupil-diameter calculation, and confidence-heatmap overlays; it has no tracking or acquisition-time dependency.
- Streams one transient `PupilPrediction` at a time so diameter, tracking, and overlay consumers share one model pass without retaining all float probability maps.
- Exposes reusable `generate_pupil_predictions(...)` and `generate_pupil_mask_prediction(...)` functions without depending on the analysis CLI.
- Includes an editable `if __name__ == "__main__":` block for direct IDE runs with hardcoded local paths and inference settings.

### 3. `pupil_tracking/api.py`

- Owns pipeline orchestration, independent of any command line.
- `AnalysisConfig` collects every input and validates combinations in one place; `run_analysis(config)` returns an `AnalysisResult` carrying the analysis table, both output paths, the frame metadata, and the internal tracking DataFrame.
- `analyze_video(...)` and `analyze_frames(...)` are keyword wrappers intended for notebook and script use.
- If a video is provided, calls `extract_selected_frames(...)` before inference.
- Composes the streaming inference iterator with optional tracking and overlay accumulators according to the configuration.
- With `calculate_velocity`, analyzes consecutive source frames using an explicit acquisition timebase and appends accepted center, speed, and three-state quality fields/panels to the unified outputs.

### 4. `pupil_tracking/run_pupil_analysis.py`

- Implements the `run-pupil-analysis` CLI only: it parses arguments, builds an `AnalysisConfig`, and reports validation failures through `parser.error(...)` so usage mistakes print usage text rather than a traceback.
- Accepts either `--video_path` or `--image_dir`.
- Enables console logging so terminal output matches the historical `print`-based behavior.

### 5. `pupil_tracking/results.py` and `pupil_tracking/plotting.py`

- `results.py` builds the compact user-facing table, derives the three-state `tracking_status`, and writes the CSV and figure. `DIAMETER_COLUMNS` and `VELOCITY_COLUMNS` are the authoritative output schemas.
- `plotting.py` returns Matplotlib figures rather than saving them, so the standard panels can be restyled or embedded elsewhere.

### 6. `pupil_tracking/extract_frames.py`

- Implements the frame-extraction CLI and reusable `extract_selected_frames(...)`.
- Samples evenly spaced frames from the input video with OpenCV.
- Can extract consecutive source frames and return source-frame metadata for velocity analysis.
- Honors `--extraction_fps` and `--max_frames`, reducing effective extraction FPS when needed.

### 7. `pupil_tracking/tracking.py`

- Postprocesses UNet probability maps without changing the model or checkpoint.
- `TrackingAccumulator` consumes transient predictions only when velocity mode is enabled, reuses their binary masks, and retains lightweight measurements for temporal processing.
- Selects and measures pupil components, maps centers back to original-image pixels, applies explainable quality flags, and calculates frame-to-frame displacement and velocity.
- Leaves published centers and velocities missing across rejected or non-consecutive frames rather than interpolating.
- Includes an editable `if __name__ == "__main__":` block for direct IDE inspection of the detailed tracking DataFrame.

### 8. `pupil_tracking/preprocessing.py` and `pupil_tracking/augmentation.py`

- `preprocessing.py` owns `resize_with_pad`, `MODEL_IMAGE_SIZE`, and `InferenceDataset`. The 148 x 148 centered/padded convention defined here is load-bearing for the current trained model.
- `augmentation.py` owns the training-only augmentations and `SegmentationDataset`; nothing in it runs during inference.
- `dataset.py` remains only as a deprecated re-export shim. Its `PupilDataset` returned either `(image, name)` or `(image, mask)` depending on construction; the two dataset classes replace that.

### 9. `pupil_tracking/unet.py`

- Defines the segmentation model used by inference and training.

### 10. `pupil_tracking/logging_utils.py`

- Library modules log and never print, so an embedding application controls its own output. The console scripts call `configure_cli_logging()` to restore plain terminal output.

### 11. `training/`

- Contains the maintained local training workflow: model training, Labelme JSON conversion, and augmentation inspection.
- `training/README.md` documents data preparation, fresh training, checkpoint-based fine-tuning, and intentional model promotion.
- All scripts use package imports and resolve training data/output folders from the repository root, independent of the current working directory.
- `training/run_train.py` writes experimental checkpoints to `checkpoints_exp/`.

### 12. `media/`

- Contains the tracked README animation and `media/make_gif.py`, the utility that regenerates it from local analysis results and overlays.
- `media/README.md` documents the required analysis outputs, animation controls, IDE workflow, and promotion review.
- The GIF is a deliberately promoted documentation asset; other generated videos and candidate media remain local.

### 10. `sample_data/`

- Contains a public clone-and-run fixture: eight paired training crops, four paired validation crops, six uncropped frames, and 31 consecutive prepared velocity frames.
- `sample_data/README.md` documents segmentation, velocity, augmentation, and training smoke workflows; `manifest.csv` records provenance and transformations.
- The fixture is maintained example data rather than a benchmark. Its velocity sequence preserves source frames `07212`-`07242` at 97 Hz.
- `MANIFEST.in` includes the fixture and its training utilities in source distributions so the bundled guide and integrity test remain self-contained; wheels exclude them from the installed runtime package.

## Segmentation-To-Velocity Method

Velocity mode is postprocessing around the existing pupil-segmentation model; it does not alter the UNet or its checkpoint.

1. **Preserve the timebase.** The pipeline analyzes consecutive source frames and assigns each frame a timestamp from its source-frame index and the actual acquisition rate.
2. **Prepare the model image.** Each grayscale frame is resized with its aspect ratio preserved, padded to the model's 148 x 148 input size, and passed through the attention UNet.
3. **Create the existing pupil segmentation.** A sigmoid converts model output to per-pixel probabilities. The existing prediction threshold, `--pred_thresh` (default `0.7`), determines which pixels enter the binary pupil mask.
4. **Select and center the pupil candidate.** The largest 8-connected foreground component is selected. Its center is the probability-weighted centroid of its pixels, so higher-confidence pixels contribute more strongly. The resize and padding are then inverted to report the center in original-frame pixels.
5. **Apply per-frame quality control.** Empty masks are rejected. The selected component is rejected if its mean probability is below `0.90`, its circularity is below `0.45`, or it touches the model-image border. If it contains less than 80% of all foreground pixels, `low_component_dominance` is recorded as a warning but does not by itself reject the frame.
6. **Apply temporal area control.** Initially valid components are compared with the median component area within approximately +/-0.5 seconds. With at least four valid neighbors, an area below `0.65` or above `2.0` times that median is recorded as an `abrupt_area_change` warning while the center remains usable.
7. **Calculate motion without bridging gaps.** Published center coordinates are retained for usable `valid` and `warning` frames. For consecutive usable frame pairs, the pipeline calculates horizontal and vertical displacement, divides each by the actual elapsed time to obtain x/y velocity, and reports scalar speed as `sqrt(velocity_x^2 + velocity_y^2)`. It does not interpolate across rejected or missing frames.
8. **Preserve diagnostic evidence.** The internal tracking table retains raw centers and detailed component measurements for quality decisions and tests. The compact analysis CSV exposes usable centers, speed, a valid/warning/invalid status, and a concise reason. The unified plot supports frame-indexed review. Optional overlays map threshold-passing pixel probabilities from yellow just above the prediction threshold, through orange, to red near probability 1.0; thin center markers locate accepted and rejected candidates.

## Repo Structure Map

```text
pupil_tracking/
|- pupil_tracking/
|  |- __init__.py            (public API, lazily imported)
|  |- api.py                 (AnalysisConfig, run_analysis, analyze_video/frames)
|  |- run_pupil_analysis.py  (CLI only)
|  |- pupil_predictions.py
|  |- extract_frames.py
|  |- tracking.py
|  |- results.py
|  |- plotting.py
|  |- preprocessing.py
|  |- augmentation.py
|  |- dataset.py             (deprecated shim)
|  |- logging_utils.py
|  |- unet.py
|  |- checkpoints/
|- tests/
|  |- test_imports.py
|  |- test_cli_help.py
|  |- test_extract_frames.py
|  |- test_outputs.py
|  |- test_tracking.py
|  |- test_end_to_end.py
|  |- test_sample_data.py
|- training/
|  |- README.md
|  |- run_train.py
|  |- labelme_json2png.py
|  |- check_augmentation.py
|- media/
|  |- README.md
|  |- make_gif.py
|  |- pupil_diameter_analysis_result_demo.gif
|- sample_data/
|  |- README.md
|  |- manifest.csv
|  |- images_train/
|  |- masks_train/
|  |- images_validation/
|  |- masks_validation/
|  |- raw_frames/
|  |- velocity_frames/
|- .github/workflows/ci.yml
|- pyproject.toml
|- README.md
|- AGENTS.md
|- project_overview.md
|- next_steps.md
|- work_log.md
|- work_log_archive/
```

## Maintained vs. Local/Generated

Active, package-facing files:

- `pupil_tracking/__init__.py`
- `pupil_tracking/api.py`
- `pupil_tracking/run_pupil_analysis.py`
- `pupil_tracking/pupil_predictions.py`
- `pupil_tracking/extract_frames.py`
- `pupil_tracking/tracking.py`
- `pupil_tracking/results.py`
- `pupil_tracking/plotting.py`
- `pupil_tracking/preprocessing.py`
- `pupil_tracking/augmentation.py`
- `pupil_tracking/logging_utils.py`
- `pupil_tracking/unet.py`
- `pupil_tracking/checkpoints/`
- `tests/`
- `.github/workflows/ci.yml`
- `pyproject.toml`
- `README.md`
- `sample_data/`

Active local/developer scripts:

- `training/run_train.py`
- `training/labelme_json2png.py`
- `training/check_augmentation.py`
- `media/make_gif.py`

Local/generated surfaces to treat carefully:

- `images_test_*`, `images_*_result`, `predicted_masks_*`, `predictions_test/`, and `results/` are local analysis outputs.
- Root `images_train/`, `images_validation/`, `masks_train/`, and `masks_validation/` are local training/validation data folders. The nested `sample_data/` versions are intentionally tracked fixtures.
- `checkpoints_exp/` holds experimental training checkpoints.
- `build/`, `dist/`, `*.egg-info`, `__pycache__/`, `.pytest_cache/`, and `.ruff_cache/` are generated.
- `archive/` and `sketch*.py` style files are not the active package path.
- `pupil_tracking/checkpoints/archive/` contains historical checkpoint files excluded from package data.

## Authored vs. Derived

### Authored - hand-edit these

- Package Python modules, training/media utilities, sample fixtures, tests, `pyproject.toml`, `MANIFEST.in`, and CI configuration are maintained source.
- `README.md`, `AGENTS.md`, `project_overview.md`, `next_steps.md`, and `work_log.md` are maintained documentation. `treaty_conventions.md` is the exception: it is upstream-managed through `treaty update`.
- The selection and packaging policy under `pupil_tracking/checkpoints/` is curated deliberately even though model binaries originate from training.

### Derived - regenerate or intentionally promote

- `build/`, `dist/`, `*.egg-info`, Python/tool caches, and local analysis/result folders are generated and should not be edited or staged.
- `checkpoints_exp/` is produced by `training/run_train.py`; promote a model into `pupil_tracking/checkpoints/` only as an intentional, reviewed package change.
- `media/pupil_diameter_analysis_result_demo.gif` is generated by `media/make_gif.py` but intentionally tracked as the README's promoted demo asset.
- `sample_data/velocity_frames/` contains generated 148 x 148 grayscale resize-and-pad outputs, intentionally tracked as a compact temporal fixture.
- Local training images, masks, extracted frames, predictions, and plots are data or outputs, not repository source. The small curated files under `sample_data/` are the intentional exception.

## Tests And Fixtures

The test suite is intentionally lightweight:

- `tests/test_imports.py` imports the package.
- `tests/test_cli_help.py` runs `run-pupil-analysis --help`.
- `tests/test_extract_frames.py` verifies sampled/full-frame source-index selection.
- `tests/test_outputs.py` verifies 1-based source-frame naming, compact unified schemas, three-state tracking status, and plot creation.
- `tests/test_tracking.py` verifies coordinate mapping, component measurements, quality flags, timing, and kinematics with synthetic inputs.
- `tests/test_end_to_end.py` runs the real pipeline against the packaged checkpoint on a synthetic video generated in a fixture: extraction, the UNet forward pass, diameter and velocity outputs, overlays, and video-versus-image-directory agreement. Synthetic input verifies wiring rather than segmentation accuracy.
- `tests/test_sample_data.py` verifies paired sample counts, matching masks, raw-frame dimensions, the consecutive 31-frame velocity contract, and manifest coverage.
- `tests/test_real_images.py` runs the packaged checkpoint over the committed `sample_data/` frames, which is what actually detects a corrupted checkpoint or a preprocessing regression.
- `.github/workflows/ci.yml` runs Ruff, Black, Pytest, and a wheel smoke check across Python 3.10, 3.11, and 3.12 where appropriate.

`sample_data/` is the canonical portable fixture for manual end-to-end exploration. It deliberately avoids a video binary: uncropped PNGs exercise normal inference, while a short consecutive PNG sequence exercises temporal tracking with an explicit acquisition rate.

## User Data Expectations

The model expects the eye/pupil region to be centered enough that the majority of the eye appears in a 148 x 148 pixel area in the center of each frame. This matters because the trained model uses the current 148 x 148 padded image convention.

Primary input modes:

- `--video_path`: video file accepted by OpenCV; frames are extracted before inference.
- `--image_dir`: folder of PNG frames; extraction is skipped.

Primary outputs:

- `*_pupil_analysis.csv`
- `*_pupil_analysis.png`
- Optional mask overlay PNGs under `--output_mask_dir`

## Practical Mental Model

For most maintenance tasks, read in this order:

1. `AGENTS.md`
2. `README.md`
3. `pyproject.toml`
4. `pupil_tracking/api.py`
5. `pupil_tracking/pupil_predictions.py`
6. `pupil_tracking/extract_frames.py`
7. `pupil_tracking/preprocessing.py`
8. `tests/` and `.github/workflows/ci.yml`

## Questions Worth Clarifying Later

- Whether any local image/mask folders should become small, portable test fixtures.
- Whether `archive/`, root sketch scripts, and local experiment outputs should stay in the working tree or be moved elsewhere.
- Whether the training workflow should be made package-facing or remain a local maintainer workflow.
- Whether README examples should include a tiny smoke-test dataset once one exists.
