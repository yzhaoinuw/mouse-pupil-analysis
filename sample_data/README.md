# Sample Data

This directory is a small public fixture for trying the repository from a fresh clone or source distribution. It contains real pupil images published with permission, but it is intentionally too small to support scientific model evaluation or production training. The fixture is not installed into the runtime wheel.

## Contents

```text
sample_data/
|- labeled_data/          # 32 curated cropped pairs, one folder per session
|  |- <session>/
|  |  |- images/          #   the frames, and their labelme JSON
|  |  |- masks/           #   the matching hand-labeled masks
|- splits.json            # the grouped, stratified fold assignment
|- unlabeled_frames/      # 6 frames from two recordings, no masks, never trained on
|  |- recording_250530/
|  |- recording_250616/
|- velocity_frames/       # 31 consecutive prepared frames
|- manifest.csv
|- README.md
```

This mirrors the maintained dataset's layout exactly: one directory per recording
session, each holding `images/` and `masks/`, and `splits.json` deciding what trains and
what validates. There are no train/validation folders. The session an image belongs to
is the folder it sits in, so the whole split regenerates from the layout alone.

`manifest.csv` records the session, fold, source recording, source-frame suffix, and
transformation of each logical sample.

The 32 pairs are a real subset of the maintained pool, chosen so the fixture exercises
the split rather than merely containing images:

| | fixture | maintained pool |
| --- | --- | --- |
| labelled pairs | 32 | 222 |
| sessions | 10 | 16 |
| images per session | 6 5 4 4 3 3 3 2 1 1 | 62 32 18 17 15 13 13 13 12 11 5 4 3 2 1 1 |
| mask diameter range | 8.8 - 109.7 | 8.8 - 109.7 |
| folds containing a mask <=15 px | 3 of 4 | 4 of 4 |

Several sessions are deep enough that holding one out removes a block of related frames,
which is what the grouping exists to do; a fixture of one image per session would satisfy
"no session spans a fold" vacuously. One fold holds no small pupil, which is the same
honest limitation the real pool has and worth seeing in a test.

## Try segmentation on real frames

From the repository root, activate an environment where the package is installed and run:

```powershell
run-pupil-analysis `
    --image_dir sample_data\unlabeled_frames\recording_250530 `
    --result_dir results\sample_unlabeled_250530 `
    --output_mask_dir results\sample_unlabeled_250530\overlays
```

Repeat with `recording_250616` to try the second acquisition setup. Each folder is analyzed separately so a result plot never connects unrelated recording timelines. This exercises preprocessing, packaged-checkpoint selection, segmentation, diameter estimation, CSV/plot output, and overlays. All six inputs retain their original dimensions.

These six frames carry no masks, and that is the point: they are the only frames here the packaged model has never trained on. `reports/scripts/hard_frame_check.py` defaults to `recording_250616`, which holds a small pupil that appears in no labelled image — the failure that three of five checkpoints in the 2026-08-14 seed study lost entirely while scoring the *highest* validation IoU.

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
python training\run_train.py --data-root sample_data --split-manifest sample_data\splits.json --fold 0 --epochs 1
```

For an IDE plumbing check, set `DATA_ROOT = PROJECT_ROOT / "sample_data"` and `n_epochs=1` in
the final `TrainingConfig(...)` block of `training/run_train.py`. The
best checkpoint, JSON metadata, and log are written to one descriptive run folder under
`checkpoints_exp/` regardless of score. A model trained on eight images is expected to
overfit and must not be treated as a useful trained model.

## Try the grouped, stratified split

```powershell
python training\data_splits.py --data-root sample_data --show
```

prints the per-session census and the per-fold summary. `sample_data/splits.json` is
committed, so a fresh clone can read the split without running anything. `folds/` is not:
it is derived output, rebuilt deterministically from the session folders by

```powershell
python training\data_splits.py --data-root sample_data --materialize
```

which writes `folds/cv1/` … `folds/cv4/`, each with `images/` and `masks/`. It is never
read back, so editing it changes nothing and re-running overwrites it. Train one fold
with:

```powershell
python training\run_train.py --data-root sample_data --split-manifest sample_data\splits.json --fold 0 --epochs 1
```

See [`../training/data_collection.md`](../training/data_collection.md) for how sessions are
recorded and how folds are packed. Thirty-two images is enough to check the split mechanics,
not to train anything.

## Provenance and use

- The cropped image/mask pairs were copied unchanged from the project's hand-curated local labelled pool.
- Their labelme JSON annotations were copied with `imageData` set to null. Labelme embeds a base64 copy of the whole image in that field, which duplicated every PNG and accounted for 1.5 MB of a 5.6 MB fixture; labelme reloads the image from `imagePath` when it is null, so the annotations still open. Nothing else in them was touched.
- Each pair sits in the folder of the recording session it came from, which is what the split groups on. `splits.json` and `folds/` are generated from that layout by `training/data_splits.py`.
- The unlabelled frames were copied unchanged from two original recording frame directories. They were briefly named `raw_frames/`.
- The velocity frames were derived from source frames `07212`-`07242` of `250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042` using grayscale conversion and the package's 148 x 148 resize-and-pad convention.
- The project has permission to publish these images and masks in this repository for collaboration and reproducible examples.

Generated results belong under the ignored `results/` directory and should not be committed.
