# Sample Data

This directory is a small public fixture for trying the repository from a fresh clone or source distribution. It contains real pupil images published with permission, but it is intentionally too small to support scientific model evaluation or production training. The fixture is not installed into the runtime wheel.

## Contents

```text
sample_data/
|- images_train/          # 8 curated cropped images
|- masks_train/           # 8 matching hand-labeled masks
|- images_validation/     # 4 curated cropped images
|- masks_validation/      # 4 matching hand-labeled masks
|- raw_frames/            # 6 uncropped frames grouped by recording
|  |- recording_250530/
|  |- recording_250616/
|- velocity_frames/       # 31 consecutive prepared frames
|- manifest.csv
|- README.md
```

`manifest.csv` records the source recording, source-frame suffix, transformation, and intended role of each logical sample.

## Try segmentation on uncropped frames

From the repository root, activate an environment where the package is installed and run:

```powershell
run-pupil-analysis `
    --image_dir sample_data\raw_frames\recording_250530 `
    --result_dir results\sample_raw_250530 `
    --output_mask_dir results\sample_raw_250530\overlays
```

Repeat with `recording_250616` to try the second acquisition setup. Each folder is analyzed separately so a result plot never connects unrelated recording timelines. This exercises preprocessing, packaged-checkpoint selection, segmentation, diameter estimation, CSV/plot output, and overlays. All six inputs retain their original uncropped dimensions.

## Try the velocity pipeline

```powershell
run-pupil-analysis `
    --image_dir sample_data\velocity_frames `
    --result_dir results\sample_velocity `
    --output_mask_dir results\sample_velocity\overlays `
    --calculate_velocity `
    --acquisition_fps 97
```

The velocity fixture contains source frames `07212` through `07242`, inclusive. They are consecutive frames from the recording used for the README GIF window, not the every-third-frame display sampling used by the animation. Preserving every source suffix allows the tracking code to calculate all 30 frame-to-frame speeds without bridging gaps.

The committed velocity images are grayscale 148 x 148 outputs from the package's aspect-preserving `resize_with_pad(...)` preprocessing. On the current packaged checkpoint, all 31 segmentations are usable, three frames carry an `abrupt_area_change` warning, and the diameter ranges from approximately 19.54 to 26.34 model pixels. These frames are already 148 x 148, so their input-pixel diameters are identical. These observations are debugging landmarks rather than a permanently frozen numerical-output contract.

## Try training and augmentation

The training utilities keep an editable `DATA_ROOT` near the top of each script. Change it from:

```python
DATA_ROOT = PROJECT_ROOT
```

to:

```python
DATA_ROOT = PROJECT_ROOT / "sample_data"
```

Then inspect paired augmentation:

```powershell
python training\check_augmentation.py
```

Or run the training script with terminal arguments:

```powershell
python training\run_train.py --data-root sample_data --epochs 1
```

For an IDE plumbing check, set `DATA_ROOT = PROJECT_ROOT / "sample_data"` and `n_epochs=1` in
the final `TrainingConfig(...)` block of `training/run_train.py`. The
best checkpoint, JSON metadata, and log are written to one descriptive run folder under
`checkpoints_exp/` regardless of score. A model trained on eight images is expected to
overfit and must not be treated as a useful trained model.

## Provenance and use

- The cropped image/mask pairs were copied unchanged from the project's hand-curated local training and validation collections.
- The raw frames were copied unchanged from two original recording frame directories.
- The velocity frames were derived from source frames `07212`-`07242` of `250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042` using grayscale conversion and the package's 148 x 148 resize-and-pad convention.
- The project has permission to publish these images and masks in this repository for collaboration and reproducible examples.

Generated results belong under the ignored `results/` directory and should not be committed.
