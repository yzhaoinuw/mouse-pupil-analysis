# Train or Fine-Tune a Pupil Model

Use this workflow when you already have labelled image/mask pairs and want a new
experimental pupil-segmentation checkpoint. The normal path is deliberately short:

1. Put labelled frames in session folders.
2. Refresh `training_data_split.json`.
3. Review the automatic assignment and optionally reserve a validation session.
4. Train with validation when that session exists.

`run_train.py` selects the best checkpoint, early-stops, and calibrates its prediction
threshold against that held-out group. Cross-validation is useful when comparing
configurations, but it is not required for a working training run.

## Contents

- [Core workflow](#core-workflow)
  - [Environment](#environment)
  - [1. Prepare labelled sessions](#1-prepare-labelled-sessions)
  - [2. Refresh the session split](#2-refresh-the-session-split)
  - [3. Review or adjust assignments](#3-review-or-adjust-assignments)
  - [4. Train and validate](#4-train-and-validate)
  - [5. Use the experimental checkpoint](#5-use-the-experimental-checkpoint)
- [Optional tools](#optional-tools)
  - [Fine-tune a checkpoint](#fine-tune-a-checkpoint)
  - [Label a batch with Labelme](#label-a-batch-with-labelme)
  - [Choose frames to label](#choose-frames-to-label)
  - [Inspect augmentation](#inspect-augmentation)
  - [Cross-validate a configuration](#cross-validate-a-configuration)
  - [Package an accepted checkpoint](#package-an-accepted-checkpoint)
  - [Developer fixture](#developer-fixture)

## Core workflow

### Environment

Run these source-checkout utilities from the repository root with the project environment:

```powershell
conda activate pupil_tracking
python -m pip install -e .
```

### 1. Prepare labelled sessions

The trainer is label-tool neutral. It only needs matching PNG images and masks arranged
by recording session:

```text
labeled_frames/
  <session>/                         # one animal, date, and recording condition
    images/
      frame_00001.png
      frame_00002.png
    masks/
      frame_00001.png                # same filename as its image
      frame_00002.png
    uncertain/                       # optional; never used for segmentation training
```

Keep every frame from the same recording in one `<session>` directory. A filename only
needs to be unique within that session; `session/frame_00001.png` is the data identity.
Do not create `train` and `validation` folders or move pairs between folders.

Use the original camera-frame size. The loader converts each image/mask pair to the
model's 148 x 148 input automatically: square frames are resized, while non-square
frames are resized proportionally and black-padded. Do not crop, resize, or pad data
yourself.

### 2. Refresh the session split

Run this after adding labelled frames, including a completely new session:

```powershell
python training\prepare_splits.py
```

`prepare_splits.py` keeps every session together, assigns each session to a validation fold,
and preserves existing assignments when new data arrives. It does not move images on disk.
The first manifest uses five folds by default; if the dataset has fewer than five sessions,
choose a smaller count once, for example `--n_folds 3`.

`--labeled_frames_dir` names the `labeled_frames/` folder itself when it is outside the
repository; its parent always receives `training_data_split.json`. `--validation_session <session>`
reserves a session for choosing a normal training run, while `--final_test_session <session>`
keeps a session out of cross-validation until an all-labeled production run is built.

Review the assignment without writing changes when needed:

```powershell
python training\prepare_splits.py --show
```

The automatic assignment is a safe starting point: it keeps sessions intact and balances pupil
size and lighting summaries across folds. Existing assignments are preserved when
new labels arrive.

### 3. Review or adjust assignments

Open the local split manager when you want to inspect the session statistics or change the
automatic assignment:

```powershell
python training\review_splits.py
```

Do not open `split_manager.html` directly in a browser: it is the interface asset, while
`review_splits.py` starts the local service that reads and safely writes
`training_data_split.json`.

It displays a stacked pupil-size chart for the folds, overlaid with each fold's background-
brightness interquartile range (Q1–Q3) and median (0 black–255 white). Click a session for its
own chart; click it again to hide that chart. Both charts update immediately when a session is
dragged. Drag a whole session between folds or into the **validation session**. Saving validates
the complete session assignment and updates
`training_data_split.json`. The served interface is the tracked
[`split_manager.html`](split_manager.html)
asset; `review_splits.py` provides its local manifest API.

The local server stops automatically shortly after every split-manager tab is closed. It also
stops if no browser tab connects after launch.

Folds are used only by cross-validation. The validation session is excluded from CV and is used
by the normal training command below. Assign one before starting a validation-backed run.

### 4. Train and validate

This trains from scratch using the validation session configured in
`training_data_split.json`:

```powershell
python training\run_train.py --checkpoint_dir checkpoints_exp\scratch
```

When the validation session contains sessions, they control early stopping, learning-rate
scheduling, checkpoint selection, and prediction-threshold calibration. All folds train the
model. `run_train.py` finds `labeled_frames/` and its sibling
`training_data_split.json` automatically; pass
`--labeled_frames_dir <folder>` only when the labelled pool lives elsewhere.

Use `python training\run_train.py --help` to set the maximum epochs, batch size, learning rate,
seed, device, or output directory. CUDA is selected automatically when available.

### 5. Use the experimental checkpoint

Each run writes to the specified checkpoint directory. Without `--checkpoint_dir`, the trainer
creates a collision-safe directory under `checkpoints_exp/`.

- `best.pth` — selected model weights.
- `best.json` — selected threshold, validation metrics, epoch, and full configuration.
- `train.log` — per-epoch training and validation record.

All-labeled training from a CV configuration writes `all_data.pth` and `all_data.json` instead
of `best.*`; its metadata records the recipe used.

Treat this folder as experimental output. Test the checkpoint on representative recordings
before replacing the package's default model.

## Optional tools

### Fine-tune a checkpoint

Fine-tuning is often faster than training from scratch when you added new sessions to the
existing labelled pool. It uses the same held-out-fold validation workflow:

```powershell
python training\run_train.py `
    --finetune_checkpoint "mouse_pupil_analysis\checkpoints\166pupils_thresh=0.4_iou=0.8749.pth" `
    --learning_rate 1e-4 `
    --checkpoint_dir checkpoints_exp\ft
```

Fine-tuning loads model weights but starts a new optimizer, scheduler, and training log. Keep
the prior labelled sessions in the pool alongside newly labelled difficult cases to reduce
forgetting.

### Label a batch with Labelme

Labelme is one supported intake route, not a requirement. If you use it, save the annotated
JSON files beside their source images, then preview and import the complete batch:

```powershell
python training\import_labelme.py --source <annotation-folder> --session <new-session>
python training\import_labelme.py --source <annotation-folder> --session <new-session> --apply
```

The importer creates the session's `images/`, `masks/`, and optional `uncertain/` folders,
then refreshes the split manifest. Use `pupil` for a visible pupil polygon,
`no_visible_pupil` for a confident true-negative frame, and `uncertain` for a frame that
should be retained but excluded from segmentation loss. See [data_collection.md](data_collection.md)
for the detailed annotation policy.

### Choose frames to label

You may label whichever frames make sense for your experiment. The recommender is available
when you want help prioritising a larger recording:

```powershell
python training\recommend_frames.py `
    --video D:\data\recording.avi `
    --budget 20 `
    --checkpoint_dir checkpoints_exp\cv516_nat_macro_20260819

python training\recommend_frames.py `
    --frames D:\data\already_extracted `
    --budget 20 `
    --checkpoint_dir checkpoints_exp\cv516_nat_macro_20260819
```

`--checkpoint_dir` is the complete directory from one cross-validation run, not an
individual model folder. The recommender discovers every immediate `*/best.pth` fold
checkpoint inside it and uses their disagreement to rank frames. Use a committee whose
models did not train on the recording you are selecting from.

By default, its outputs go under `frames_to_label/<session>/`. After labelling, put the
resulting image/mask pairs in `labeled_frames/<session>/` by any supported method and run
`prepare_splits.py`.

<details>
<summary>Optional recommender arguments</summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--budget N` | `20` | Number of frames to recommend. |
| `--extraction-fps FPS` | `5` | Sampling rate for video input before the extraction cap applies. |
| `--max-extracted N` | `2000` | Maximum frames extracted from a video; excess recordings are sampled evenly. |
| `--threshold P` | `0.5` | Pixel-probability threshold used while scoring committee masks. |
| `--min-gap N` | Automatic | Minimum spacing between recommended frames, measured in extracted-frame positions. |
| `--temporal-weight W` | `0` | Weight for the consecutive-frame consistency signal. |
| `--output_dir DIR` | `frames_to_label/` | Root for the session's `extracted_frames/` and `recommended/` folders. |
| `--device DEVICE` | `auto` | Inference device, such as `cpu` or `cuda`. |
| `--no-dedup` | Off | Keep visually near-identical frames instead of collapsing them. |
| `--force` | Off | Replace generated PNGs and `selection.csv` in an existing output folder. |

</details>

### Inspect augmentation

`preview_augmentation.py` is a visual diagnostic, not preprocessing:

```powershell
python training\preview_augmentation.py
```

It renders augmented image/mask pairs so you can check that the pupil remains aligned and the
transforms look plausible.

### Cross-validate a configuration

Use cross-validation when you need to compare configurations or estimate how sensitive a
result is to the held-out session group. It takes turns holding out each fold and
never loads the validation session:

```powershell
python training\run_cross_validation.py --checkpoint_dir checkpoints_exp\cv
```

Use its per-session results to compare candidate settings. It also writes
`training_config.json`, a complete all-labeled training recipe based on the median
successful-fold epoch and calibrated threshold. Train the production model from it with:

```powershell
python training\run_train.py `
    --training_config_path checkpoints_exp\cv\training_config.json `
    --checkpoint_dir checkpoints_exp\all_labeled
```

That command deliberately reads every valid image/mask pair under `labeled_frames/` and ignores
`training_data_split.json`, including any session formerly reserved as a final test. Use it when you are ready
to trust the CV-selected recipe and inspect the resulting model on representative unlabeled
recordings.

To rerun only specific existing folds, use `--cv_folds`, for example `--cv_folds 0 2`.
That writes `partial_summary.json` only; it deliberately does not create a production training
configuration.

### Package an accepted checkpoint

<details>
<summary>Package an accepted checkpoint into the installed application</summary>

Packaging is a package/release change, not a normal training step. After checking overlays and
downstream tracking on representative recordings, preview then package a validation-selected run:

```powershell
python training\package_checkpoint.py --run-dir checkpoints_exp\<run-name> --dry-run
python training\package_checkpoint.py --run-dir checkpoints_exp\<run-name> `
    --validation-note "Held-out session validation; see best.json."
```

Update `CHANGELOG.md`, run the repository checks in `AGENTS.md`, and verify the checkpoint,
metadata, and log are included in both package distributions.

</details>

### Developer fixture

`sample_data/` is a small public fixture for checking data flow and checkpoint writing, not
for training a useful model. See [sample_data/README.md](../sample_data/README.md) for its
commands and limits.
