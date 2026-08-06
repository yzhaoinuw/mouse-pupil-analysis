# Next Steps

Use this checklist alongside `work_log.md`. Keep it concrete: only add work here when it is an actual follow-up, blocked thread, or decision that future agents should see before changing code.

## Currently Hot

- [Pupil-center velocity](#pupil-center-velocity) - the branch is pushed and a GPU-regenerated 3,001-frame inspection set with translucent masks and center crosses is ready under `videos/eye_manual_inspection/`; review the overlays before changing quality thresholds.
- [DOI archival](#doi-archival) - optional next step after a GitHub release exists.

When a new thread starts, add a short bullet here with a link to its section below and the single next action.

## Pupil-Center Velocity

Status: implemented, validated, and pushed on `feature/pupil-velocity`; manual inspection/tuning review pending

### Goal And Scope

Add opt-in pupil-center tracking and velocity calculation for REM analysis while preserving the current diameter-only workflow.

For every acquired frame, report:

- Source frame index and actual acquisition timestamp.
- Pupil-center x and y coordinates in original-video pixels.
- Horizontal and vertical displacement from the immediately preceding frame.
- Horizontal and vertical velocity in pixels per second.
- Scalar pupil-center speed in pixels per second.
- Existing pupil-size measurement plus segmentation quality evidence.

The first version does not include an interactive interface, phasic/tonic REM classification, model retraining, or temporal averaging of probability maps. The UNet architecture, packaged checkpoint, training workflow, and existing 0.7 threshold remain unchanged.

### Confirmed Sample-Video Contract

The supplied `C:\Users\yzhao\Desktop\eye.avi` contains 3,001 encoded frames. Its video header reports 100 fps and approximately 30 seconds, while its burned-in timestamps span 0.0 to 90.0 seconds. The collaborator confirmed that the burned-in timestamps represent actual experimental time.

For this video:

- Every encoded frame is one acquired sample.
- Acquisition interval is exactly 0.03 seconds.
- Acquisition rate is approximately 33.3333 samples per second.
- Frame 0 has timestamp 0.00 seconds and frame 3,000 has timestamp 90.00 seconds.
- The AVI header rate must not be used to calculate biological velocity.

### CLI And Backward-Compatibility Contract

Add:

- `--calculate_velocity`: opt in to full-frame center tracking, quality control, and kinematic outputs.
- `--acquisition_fps`: optional positive acquisition rate used for timestamps and velocity. With video input, default to the video header only when no override is supplied. With `--image_dir`, require this argument in velocity mode because no reliable timebase is available.

Example for the supplied video:

```powershell
run-pupil-analysis `
  --video_path C:\Users\yzhao\Desktop\eye.avi `
  --calculate_velocity `
  --acquisition_fps 33.3333333333
```

Velocity mode must analyze every encoded frame in source order. The current `--extraction_fps` behavior remains unchanged outside velocity mode. If `--max_frames` prevents full-frame analysis, analyze the first consecutive source frames up to the cap, print a clear warning, and retain original source frame indices so timestamps and velocity remain correct.

Existing diameter CSV and plot filenames and columns remain available. Velocity mode adds new outputs instead of silently changing the legacy files.

### Frame Extraction And Timing

Update `pupil_tracking/extract_frames.py` so extraction returns metadata for every successfully written image:

- Extracted image path/name.
- Original source-frame index.
- Extraction-order index.

Keep existing callers valid if they ignore the return value. In velocity mode select every source frame rather than sampling according to the encoded playback duration. Pass the returned metadata directly into inference so stale PNG files already present in an output directory cannot be accidentally included.

For an existing image folder, sort images using their numeric suffix and treat them as consecutive acquired samples unless explicit frame metadata is available.

Calculate:

```text
timestamp_seconds = source_frame_index / acquisition_fps
```

Use source-frame differences, not output-row differences, when checking whether two samples are consecutive.

### Segmentation Postprocessing

Create a focused `pupil_tracking/tracking.py` module for testable postprocessing and kinematics. Do not change `pupil_tracking/unet.py` or the checkpoint.

Modify inference so it retains both:

- The sigmoid probability map.
- The binary mask produced by the existing prediction threshold.

For each frame:

1. Find connected components in the binary mask.
2. Select the largest plausible pupil component.
3. Calculate total foreground area and selected-component area.
4. Calculate the selected component's mean probability, dominance fraction, border contact, bounding box, and circularity.
5. Calculate a probability-weighted centroid within the selected component.
6. Convert the 148 x 148 model-space centroid back into original-image coordinates by inverting the existing aspect-ratio-preserving resize and padding.
7. Preserve the existing model-space diameter calculation for backward compatibility.

Do not use centroid position or a large displacement as a standalone rejection rule. A pupil near the eye edge or a rapid movement may be the biological signal of interest.

### Segmentation Quality Control

A preliminary diagnostic on 301 evenly spaced sample frames found no empty masks at the current 0.7 threshold, but 58 masks had multiple connected components. An eyelid closure around 3.3-3.9 seconds was confidently misidentified as pupil and produced a large false center jump. Confidence weighting alone did not remove that outlier.

Use combined per-frame evidence:

- Empty mask.
- Low selected-component dominance.
- Low mean component confidence.
- Implausible component shape/circularity.
- Selected component touching the model-image border.
- Abrupt component-area change relative to a robust local temporal baseline.

Export the individual metrics, `segmentation_valid`, and a semicolon-separated `quality_reason`. Keep thresholds as named constants in `tracking.py` so they are visible, testable, and easy to tune from validation evidence.

Low component dominance is retained as a diagnostic warning rather than a standalone rejection because reflections can create extra foreground components while the selected pupil component remains accurate. Low selected-component confidence, low circularity, border contact, empty masks, and abrupt local area changes reject a segmentation.

The first version must not interpolate rejected centers or velocities. Preserve rejected raw measurements for diagnosis, but publish center and kinematic values as missing for invalid frames.

### Kinematic Calculation

For two immediately consecutive valid source frames:

```text
delta_x = center_x[i] - center_x[i - 1]
delta_y = center_y[i] - center_y[i - 1]
delta_t = timestamp[i] - timestamp[i - 1]

velocity_x = delta_x / delta_t
velocity_y = delta_y / delta_t
speed = sqrt(velocity_x^2 + velocity_y^2)
```

Use image-coordinate convention: x increases to the right and y increases downward. Document this in the CSV/README.

If the current frame or preceding frame is invalid, the source frames are not consecutive, or elapsed time is non-positive, leave displacement and velocity fields missing. Do not bridge gaps.

Start with raw valid centroids. Quantify trajectory jitter on the sample video before adding smoothing. If smoothing is later justified, make it optional and export raw and smoothed values separately.

### New Outputs

In velocity mode, add:

- `<experiment>_pupil_tracking.csv`
- `<experiment>_pupil_tracking_qc.png`

The tracking CSV should include:

```text
image_name
source_frame_index
timestamp_seconds
center_x_pixels
center_y_pixels
displacement_x_pixels
displacement_y_pixels
velocity_x_pixels_per_second
velocity_y_pixels_per_second
speed_pixels_per_second
estimated_pupil_diameter
selected_component_area
foreground_area
component_count
component_dominance
mean_component_confidence
component_circularity
component_touches_border
segmentation_valid
quality_reason
```

The QC plot should show pupil size, x/y center trajectories, speed, and rejected-frame markers on the shared actual-time axis.

When `--output_mask_dir` is supplied in velocity mode, draw the chosen center and visually distinguish rejected segmentations while preserving the existing mask overlay behavior.

### Code Map

- `pupil_tracking/tracking.py`
  - Component measurements, coordinate conversion, quality flags, temporal area validation, and kinematics.
- `pupil_tracking/extract_frames.py`
  - Full-frame selection option and returned source-frame metadata.
- `pupil_tracking/run_pupil_analysis.py`
  - CLI arguments, retained probability maps, tracking integration, CSV/QC outputs, and legacy-output preservation.
- `tests/test_tracking.py`
  - Pure synthetic tests for centroids, component selection, quality flags, coordinate conversion, timing, and velocity gaps.
- `tests/test_extract_frames.py`
  - Focused frame-selection and metadata tests.
- `README.md`
  - New arguments, sample command, output schema summary, coordinate convention, and quality-flag behavior.

### Acceptance And Verification

Automated acceptance:

- Synthetic probability maps produce expected weighted centroids and component measurements.
- Coordinate conversion correctly reverses resize-and-pad geometry for landscape and portrait frames.
- Frame timestamps use acquisition rate rather than encoded playback rate.
- Velocity uses actual elapsed time and is missing across invalid or non-consecutive frames.
- Legacy diameter result generation remains covered.
- CLI help documents the new options.

Sample-video acceptance:

- Velocity-mode output has 3,001 ordered rows when not capped.
- Timestamps run from 0.00 through 90.00 seconds in 0.03-second steps.
- Normal pupil centers visually land inside the pupil.
- Closed-eye frames 110-113 at 3.30-3.39 seconds are rejected instead of generating a published velocity spike.
- Genuine rapid or edge-position movements are not rejected solely for their position or magnitude.
- Quality metrics and overlays make every rejected frame explainable.

Observed result on the supplied video:

- 2,993 of 3,001 segmentations are valid.
- Eight segmentations are rejected: frames 110-113, 121-122, and 127-128.
- Eleven otherwise valid frames retain a `low_component_dominance` warning.
- 2,989 frame-to-frame speeds are published; values touching rejected frames remain missing.
- The largest published speed peaks at frames 739, 1,196, and 2,721 correspond to visible rapid pupil movement in adjacent source frames.
- Optional overlays show valid centers in green and rejected masks/centers in orange and yellow.

Follow-up after review:

- Review the GPU-regenerated `videos/eye_manual_inspection/overlays_with_center/` set and sample contact sheet before changing quality thresholds.
- Validate the quality thresholds on additional recordings before treating them as a universal automated rejection policy.
- Consider optional trajectory smoothing only after quantifying center jitter and confirming that filtering preserves rapid REM eye movements.

Repository verification:

```powershell
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .
$env:PATH='C:\Users\yzhao\miniconda3\envs\pupil_tracking\Scripts;' + $env:PATH
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m pytest -q
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m build --wheel --sdist
```

Verify that the packaged checkpoint and training log remain present in both wheel and source distribution outputs.

## DOI Archival

Status: ready for user/account action

The repo now has MIT license metadata and `CITATION.cff`, so GitHub can display citation metadata for tagged releases. A DOI still requires linking the GitHub repository to an archival service such as Zenodo and creating or syncing a release there.

Remaining work:

- After the citable version tag is pushed, enable Zenodo or another archive for `yzhaoinuw/pupil_tracking`.
- Mint a DOI for the release and add it to `CITATION.cff` and `README.md`.

## Background / Paused

### Portable End-To-End Fixture

Status: paused

The current tests cover package import and CLI help. There is no small committed video/frame fixture for end-to-end inference.

Resume when the project needs stronger regression coverage for `run-pupil-analysis` outputs.

Remaining work:

- Decide whether a tiny synthetic or curated frame set can be committed without bloating the repo.
- Add a focused smoke test that exercises inference without relying on large local data folders.

### Local Artifact Cleanup

Status: paused

The working tree commonly contains generated image folders, prediction outputs, build outputs, cache folders, local sketch scripts, and experimental checkpoints. `.gitignore` covers the expected generated surfaces, but the local workspace may still be visually noisy.

Resume only when the user asks for repository cleanup or release preparation.

Remaining work:

- Inspect tracked vs. ignored files before deleting anything.
- Keep `pupil_tracking/checkpoints/` package data intact.

### Training Workflow Documentation

Status: paused

`README.md` includes maintainer notes for creating masks with Labelme and training with `run_train.py`, but training remains a local workflow rather than a packaged command.

Resume if the training path becomes user-facing or needs reproducible CI coverage.

Remaining work:

- Decide whether training should stay in `run_train.py` or move behind a package CLI.
- Document any required data layout, hyperparameter, and checkpoint naming contracts if the workflow is formalized.
