# Sample Data

This directory is a small public fixture for trying the repository from a fresh clone or source distribution. It contains real pupil images published with permission, but it is intentionally too small to support scientific model evaluation or production training. The fixture is not installed into the runtime wheel.

## Contents

```text
sample_data/
|- labeled_data/          # 32 curated cropped images (+ their labelme JSON)
|- labeled_masks/         # 32 matching hand-labeled masks
|- provenance.csv         # which recording session each image came from
|- splits.json            # the grouped, stratified fold assignment
|- folds/                 # the same split as folders, generated from splits.json
|  |- cv1/ cv2/ cv3/ cv4/ #   each with images/ and masks/
|- raw_frames/            # 6 uncropped frames grouped by recording
|  |- recording_250530/
|  |- recording_250616/
|- velocity_frames/       # 31 consecutive prepared frames
|- manifest.csv
|- README.md
```

This mirrors the maintained dataset's layout exactly: one flat labelled pool, a
`provenance.csv` recording the session behind each image, and `splits.json` deciding
what trains and what validates. There are no train/validation folders.

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

prints the per-session census and the per-fold summary. `sample_data/splits.json` and
`sample_data/folds/` are both committed, so a fresh clone can read the split without
running anything; regenerate them with:

```powershell
python training\data_splits.py --data-root sample_data --materialize
```

`folds/` is derived output rebuilt from `splits.json` and never read back, so editing it
changes nothing. Train one fold with:

```powershell
python training\run_train.py --data-root sample_data --split-manifest sample_data\splits.json --fold 0 --epochs 1
```

See [`../training/data_collection.md`](../training/data_collection.md) for how sessions are
recorded and how folds are packed. Thirty-two images is enough to check the split mechanics,
not to train anything.

## Provenance and use

- The cropped image/mask pairs were copied unchanged from the project's hand-curated local labelled pool.
- Their labelme JSON annotations were copied with `imageData` set to null. Labelme embeds a base64 copy of the whole image in that field, which duplicated every PNG and accounted for 1.5 MB of a 5.6 MB fixture; labelme reloads the image from `imagePath` when it is null, so the annotations still open. Nothing else in them was touched.
- `provenance.csv` records the session each image belongs to, carried over from the main pool's sidecar. `splits.json` and `folds/` are generated from it by `training/data_splits.py`.
- The raw frames were copied unchanged from two original recording frame directories.
- The velocity frames were derived from source frames `07212`-`07242` of `250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042` using grayscale conversion and the package's 148 x 148 resize-and-pad convention.
- The project has permission to publish these images and masks in this repository for collaboration and reproducible examples.

Generated results belong under the ignored `results/` directory and should not be committed.
