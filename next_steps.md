# Next Steps

Use this checklist alongside `work_log.md`. Keep it concrete: only add work here when it is an actual follow-up, blocked thread, or decision that future agents should see before changing code.

## Currently Hot

- [Packaging and distribution](#packaging-and-distribution) - the `refactor` branch renames the distribution and adds release automation; the PyPI publisher and Zenodo webhook still need account actions.
- [Runtime modularization](#runtime-modularization) - complete; a public Python API and focused modules are in place on the `refactor` branch.
- [Pupil-center velocity](#pupil-center-velocity) - temporal area outliers are now usable warnings and the refreshed main-cadence demo is ready for review; next validate the provisional thresholds on additional recordings.
- [Treaty v0.6.0 upstream feedback](#treaty-v060-upstream-feedback) - publication with `dev` and `main` is authorized in this delivery; monitor upstream issue #18 afterward.
- [DOI archival](#doi-archival) - optional next step after a GitHub release exists.
- [Sample data for examples and regression tests](#sample-data-for-examples-and-regression-tests) - blocked on data-sharing permission; the user is emailing collaborators as of 2026-08-10.

When a new thread starts, add a short bullet here with a link to its section below and the single next action.

## Packaging And Distribution

Status: implemented on the `refactor` branch; blocked on two account actions before a release can be cut

The distribution is renamed to `mouse-pupil-analysis` because `pupil-tracking` on PyPI belongs to an unrelated project. `.github/workflows/release.yml` builds on a `v*` tag, verifies that the tag, `CITATION.cff`, and the packaged checkpoint all agree with `pyproject.toml`, and publishes through Trusted Publishing.

Remaining work:

- Register the pending PyPI publisher and enable the Zenodo webhook. Both are account actions; exact field values are in [`RELEASING.md`](RELEASING.md).
- After the first archived release, fill in the commented `identifiers` block in `CITATION.cff` with the version DOI and add the concept-DOI badge to `README.md`.
- Decide whether the packaging rename ships as `0.2.0`.

## Runtime Modularization

Status: complete on the `refactor` branch

`api.py` owns orchestration behind `AnalysisConfig`/`run_analysis`, with `analyze_video` and `analyze_frames` as the public front door. `run_pupil_analysis.py` is argument parsing only. Table assembly and plotting live in `results.py` and `plotting.py`; `dataset.py` split into `preprocessing.py` and `augmentation.py` with a deprecating shim. Library code logs instead of printing.

Remaining work:

- Remove the `dataset.py` shim and the deprecated `generate_pupil_mask_prediction` after one release.
- Decide whether the unified plot should show `pupil_diameter_video_pixels` instead of the model-pixel column, now that both are exported. This changes the appearance of the README demo, so it is deliberately deferred.

## Pupil-Center Velocity

Status: velocity calculation, confidence-heatmap overlays, 1-based source-frame names, unified user-facing outputs, and the selected 90-frame/5-fps main-cadence README demo are implemented; temporal area outliers remain usable warnings when all hard per-frame checks pass

### Goal And Scope

Add opt-in pupil-center tracking and velocity calculation for REM analysis while preserving the current diameter-only workflow.

For every acquired frame, report:

- Image name containing the one-based source-frame number and the actual acquisition timestamp.
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
- Displayed frame 1 has timestamp 0.00 seconds and displayed frame 3,001 has timestamp 90.00 seconds.
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

The accepted follow-up replaces the duplicate diameter and tracking artifacts with one compact user-facing table and one expandable plot. Keep the detailed tracking measurements internal for quality control and tests rather than exporting every intermediate metric by default.

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

Retain the individual metrics internally for quality-control decisions and tests. Export only a compact status and reason alongside the measurements users normally analyze. Keep thresholds as named constants in `tracking.py` so they remain visible, testable, and easy to tune from validation evidence.

Low component dominance and abrupt local area changes are diagnostic warnings rather than standalone rejections when the selected pupil component remains otherwise usable. Low selected-component confidence, low circularity, border contact, and empty masks reject a segmentation.

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

### Unified User-Facing Outputs

Status: implemented; the full sample-video outputs were regenerated and accepted, and the selected README demo uses the approved aligned layout with main's smoother 90-frame/5-fps cadence. Temporal area warnings retain their published center and speed values; genuinely invalid measurements remain blank as specified above.

Write one table and one plot per analysis instead of separate diameter and tracking artifacts. Prefer the broader names `<experiment>_pupil_analysis.csv` and `<experiment>_pupil_analysis.png`; if backward compatibility requires a transition, make it explicit rather than indefinitely writing duplicate files.

The base table should contain only `image_name` and `estimated_pupil_diameter`. With `--calculate_velocity`, append:

```text
timestamp_seconds
center_x_pixels
center_y_pixels
speed_pixels_per_second
tracking_status
quality_reason
```

Use `tracking_status` values `valid`, `warning`, and `invalid`. A warning remains usable but points the user to a suspicious frame; an invalid row has blank published center and speed values. Keep raw centers, x/y displacement and velocity, component areas/counts, confidence, circularity, and temporal-area calculations in the internal DataFrame unless a later advanced-diagnostics option is justified.

Export `image_name` as the only user-facing frame identifier. When input comes from a video, name each extracted frame and corresponding overlay from the real source-frame index, for example `eye_00020.png` for source frame 20, instead of using extraction order. With `--image_dir`, preserve the supplied filename as the external identifier. Keep numeric `source_frame_index` internally for timestamp, consecutiveness, and velocity calculations, but do not duplicate it in the compact exported table.

The unified plot should always show pupil diameter. With `--calculate_velocity`, append x/y center, speed, and a three-state quality subplot. Share the displayed one-based source-frame number across every subplot so sampled or missing source frames remain visible. Display clean frames in green, warnings in amber, and invalid frames in red.

When `--output_mask_dir` is supplied, render threshold-passing pixels as a translucent confidence heatmap: yellow immediately above `--pred_thresh`, orange at intermediate confidence, and red near probability 1.0. In velocity mode retain the thin center marker so rejected candidates remain easy to locate visually.

### Code Map

- `pupil_tracking/tracking.py`
  - Component measurements, coordinate conversion, quality flags, temporal area validation, and kinematics.
- `pupil_tracking/extract_frames.py`
  - Full-frame selection option and returned source-frame metadata.
- `pupil_tracking/run_pupil_analysis.py`
  - CLI arguments, retained confidence maps, tracking integration, compact unified CSV/plot outputs, and cleanup of superseded duplicate outputs.
- `tests/test_tracking.py`
  - Pure synthetic tests for centroids, component selection, quality flags, coordinate conversion, timing, and velocity gaps.
- `tests/test_extract_frames.py`
  - Focused frame-selection and metadata tests.
- `tests/test_outputs.py`
  - One-based naming, compact diameter/velocity schemas, three-state status, and unified plot tests.
- `project_overview.md`
  - Detailed segmentation-to-velocity methodology and quality-control semantics.
- `README.md`
  - User-facing arguments, sample command, output summary, and a link to the detailed methodology in `project_overview.md`.

### Acceptance And Verification

Automated acceptance:

- Synthetic probability maps produce expected weighted centroids and component measurements.
- Coordinate conversion correctly reverses resize-and-pad geometry for landscape and portrait frames.
- Frame timestamps use acquisition rate rather than encoded playback rate.
- Velocity uses actual elapsed time and is missing across invalid or non-consecutive frames.
- Diameter-only mode writes the same compact unified output without velocity-only fields or panels.
- CLI help documents the new options.
- Confidence heatmaps preserve the grayscale frame outside the mask and render threshold, intermediate, and near-perfect probabilities as yellow, orange, and red.

Sample-video acceptance:

- Velocity-mode output has 3,001 ordered rows when not capped.
- Timestamps run from 0.00 through 90.00 seconds in 0.03-second steps.
- Normal pupil centers visually land inside the pupil.
- Closed-eye displayed frames 111-114 at 3.30-3.39 seconds remain explainable: hard-invalid frames are rejected, while otherwise usable temporal area outliers remain published with warnings.
- Genuine rapid or edge-position movements are not rejected solely for their position or magnitude.
- Quality metrics and overlays make every rejected frame explainable.

Observed result on the supplied video:

- 2,995 of 3,001 segmentations are usable: 2,982 clean `valid` rows and 13 `warning` rows.
- Six segmentations are rejected: displayed frames 111, 114, 122-123, and 128-129.
- Two formerly rejected temporal-area outliers, displayed frames 112-113, retain an `abrupt_area_change` warning; 11 other usable frames retain a `low_component_dominance` warning.
- 2,990 frame-to-frame speeds are published; values touching the six rejected frames remain missing.
- The largest published speed peaks at displayed frames 740, 1,197, and 2,722 correspond to visible rapid pupil movement in adjacent source frames.
- Optional overlays retain thin center markers and use mask color to show per-pixel model confidence from yellow through orange to red.

Follow-up after review:

- After the user edits README on GitHub, synchronize those remote changes before further local edits.
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

## Treaty v0.6.0 Upstream Feedback

Status: the validated Copier-managed migration from `v0.3.2` to `v0.6.0` is included in the authorized `dev` and `main` publication

Remaining work:

- Track upstream dry-run preview feedback in https://github.com/yzhaoinuw/agent_collab_treaty/issues/18.

## DOI Archival

Status: ready for user/account action

The repo now has MIT license metadata and `CITATION.cff`, so GitHub can display citation metadata for tagged releases. A DOI still requires linking the GitHub repository to an archival service such as Zenodo and creating or syncing a release there.

Remaining work:

- After the citable version tag is pushed, enable Zenodo or another archive for `yzhaoinuw/pupil_tracking`.
- Mint a DOI for the release and add it to `CITATION.cff` and `README.md`.

## Sample Data For Examples And Regression Tests

Status: blocked on data-sharing permission; the user is emailing collaborators for approval as of 2026-08-10

A small curated sample of training images and masks should be committed so the
labeling workflow is concrete for outside users and so inference has a real-image
regression test. This supersedes the fixture question parked under
[Portable End-To-End Fixture](#portable-end-to-end-fixture).

### Why Both Kinds Of Fixture Are Needed

A synthetic frame generated in-test verifies plumbing only: array shapes, CSV
schema, and files written. It cannot detect a corrupted checkpoint, a regression
in `resize_with_pad`, or a silent architecture mismatch, because a synthetic blob
segments plausibly either way. Keep the synthetic end-to-end test for wiring and
add a real-image test that pins an expected diameter within tolerance.

### Requested Contents

- Four to six image/mask pairs, chosen for variety rather than volume: one clean
  mid-dilation pupil, one strongly constricted, one strongly dilated, and one with
  partial eyelid occlusion. The occlusion case doubles as a visual reference when a
  user tunes `--pred_thresh`.
- At least one full-resolution original video frame, not only the 148 x 148 crop.
  Verifying the model-pixel to video-pixel inverse transform for the additive
  diameter column requires a frame with real source dimensions.
- Approximately 1 MB total. At 148 x 148 grayscale PNG this is not a binding limit.

### Destination

Commit to `examples/sample_data/images/` and `examples/sample_data/masks/`.

Do not use `images_train/`, `masks_train/`, `images_validation/`, or
`masks_validation/`. All four are ignored in `.gitignore`, so the sample would be
silently dropped, and it would also mix into the local training data folders.

### Scope Boundary

The full training set stays local. Repository contents are archived permanently
into every Zenodo release and into Git history. Deposit the complete dataset
separately with its own data DOI so the software and data records cross-reference
each other.

Remaining work:

- Obtain redistribution permission before committing any frames; confirm no IACUC
  or collaborator restriction applies.
- Select and commit the pairs, then add the real-image inference regression test.
- Decide whether the sample should also support an opt-in one-epoch CPU training
  smoke test, since `training/run_train.py` currently has no coverage.

## Background / Paused

### Portable End-To-End Fixture

Status: tracked media and training utilities now have dedicated root folders; ignored local sketches and generated analysis artifacts remain outside the maintained source layout

The current tests cover package import and CLI help. There is no small committed video/frame fixture for end-to-end inference.

Superseded by [Sample Data For Examples And Regression Tests](#sample-data-for-examples-and-regression-tests), which resolves the curated-frame question and adds the real-image regression test. The synthetic-video half of the work is tracked there and remains unblocked.

Remaining work:

- Add a synthetic-video smoke test that exercises inference without relying on large local data folders.

### Local Artifact Cleanup

Status: paused

The working tree commonly contains generated image folders, prediction outputs, build outputs, cache folders, local sketch scripts, and experimental checkpoints. `.gitignore` covers the expected generated surfaces, but the local workspace may still be visually noisy.

Resume only when the user asks for repository cleanup or release preparation.

Remaining work:

- Inspect tracked vs. ignored files before deleting anything.
- Keep `pupil_tracking/checkpoints/` package data intact.

### Training Workflow Documentation

Status: documented; the workflow remains script-based

`training/README.md` now documents the local data layout, Labelme conversion, augmentation review, fresh training, weight-based fine-tuning, and checkpoint promotion. The current fine-tuning path restores model weights but starts new optimizer and scheduler state.

Resume if the training path becomes user-facing or needs reproducible CI coverage.

Remaining work:

- Decide whether the workflow under `training/` should remain script-based or move behind a package CLI.
- Add structured optimizer/scheduler checkpointing only if exact interrupted-run resume becomes necessary.
