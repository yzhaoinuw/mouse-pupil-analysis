# Train or Fine-Tune a Pupil Model

Use this workflow when you already have labelled image/mask pairs and want a new
experimental pupil-segmentation checkpoint. The normal path is deliberately short:

1. Put labelled frames in session folders.
2. Refresh `splits.json`.
3. Train while holding out one session group for validation.

`run_train.py` selects the best checkpoint, early-stops, and calibrates its prediction
threshold against that held-out group. Cross-validation is useful when comparing
configurations, but it is not required for a working training run.

## Contents

- [Core workflow](#core-workflow)
  - [Environment](#environment)
  - [1. Prepare labelled sessions](#1-prepare-labelled-sessions)
  - [2. Refresh the session split](#2-refresh-the-session-split)
  - [3. Train and validate](#3-train-and-validate)
  - [4. Use the experimental checkpoint](#4-use-the-experimental-checkpoint)
- [Optional tools](#optional-tools)
  - [Fine-tune a checkpoint](#fine-tune-a-checkpoint)
  - [Label a batch with Labelme](#label-a-batch-with-labelme)
  - [Choose frames to label](#choose-frames-to-label)
  - [Inspect augmentation](#inspect-augmentation)
  - [Cross-validate a configuration](#cross-validate-a-configuration)
  - [Advanced evaluation and promotion](#advanced-evaluation-and-promotion)
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
python training\data_splits.py --data-root . --out splits.json
```

`data_splits.py` keeps every session together, assigns each session to a validation fold,
and preserves existing assignments when new data arrives. It does not move images on disk.
The first manifest uses five folds by default; if the dataset has fewer than five sessions,
choose a smaller count once, for example `--folds 3`.

Review the assignment without writing changes when needed:

```powershell
python training\data_splits.py --data-root . --show
```

For a normal run, one fold is the validation set and all other folds train the model. The
current CLI chooses that group with `--fold N`; session-to-fold assignment is automatic and
frozen in `splits.json` once written.

### 3. Train and validate

This trains from scratch and uses fold 0 as the validation set:

```powershell
python training\run_train.py `
    --data-root . `
    --split-manifest splits.json `
    --fold 0 `
    --run-name scratch_f0
```

Choose another fold with `--fold 1`, `--fold 2`, and so on. During training, the held-out
sessions control early stopping, learning-rate scheduling, checkpoint selection, and
prediction-threshold calibration. The other sessions are the training set.

Use `python training\run_train.py --help` to set the epoch limit, batch size, learning rate,
seed, sampling mode, architecture, or output directory. CUDA is selected automatically when
available.

### 4. Use the experimental checkpoint

Each run writes a new folder under `checkpoints_exp/<run-name>/`; an existing run folder is
never overwritten.

- `best.pth` — selected model weights.
- `best.json` — selected threshold, validation metrics, epoch, and full configuration.
- `train.log` — per-epoch training and validation record.

Treat this folder as experimental output. Test the checkpoint on representative recordings
before replacing the package's default model.

## Optional tools

### Fine-tune a checkpoint

Fine-tuning is often faster than training from scratch when you added new sessions to the
existing labelled pool. It uses the same held-out-fold validation workflow:

```powershell
python training\run_train.py `
    --data-root . `
    --split-manifest splits.json `
    --fold 0 `
    --finetune-checkpoint "mouse_pupil_analysis\checkpoints\166pupils_thresh=0.4_iou=0.8749.pth" `
    --learning-rate 1e-4 `
    --run-name ft_f0
```

Fine-tuning loads model weights but starts a new optimizer, scheduler, and training log. Keep
the prior labelled sessions in the pool alongside newly labelled difficult cases to reduce
forgetting.

### Label a batch with Labelme

Labelme is one supported intake route, not a requirement. If you use it, save the annotated
JSON files beside their source images, then preview and import the complete batch:

```powershell
python training\import_labelme_batch.py --source <annotation-folder> --session <new-session>
python training\import_labelme_batch.py --source <annotation-folder> --session <new-session> --apply
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
python training\recommend_frames.py --video D:\data\recording.avi --budget 20
python training\recommend_frames.py --frames D:\data\already_extracted --budget 20
```

By default, its outputs go under `frames_to_label/<session>/`. After labelling, put the
resulting image/mask pairs in `labeled_frames/<session>/` by any supported method and run
`data_splits.py`.

### Inspect augmentation

`check_augmentation.py` is a visual diagnostic, not preprocessing:

```powershell
python training\check_augmentation.py
```

It renders augmented image/mask pairs so you can check that the pupil remains aligned and the
transforms look plausible.

### Cross-validate a configuration

Use cross-validation when you need to compare configurations or estimate how sensitive a
result is to the held-out session group. It repeats the normal training workflow across folds:

```powershell
python training\run_cv.py --data-root . --split-manifest splits.json --out checkpoints_exp\cv
```

Use its per-session results to compare candidate settings. It is not a prerequisite for the
single-fold training command above.

### Advanced evaluation and promotion

<details>
<summary>Hold out sessions for a one-time final evaluation</summary>

Use an untouched session only when you need a stricter final performance gate. Set it aside
while building the split manifest, train a fixed all-development refit, and evaluate it once:

```powershell
python training\data_splits.py --data-root . --holdout <session> --out splits.json
python training\run_train.py --split-manifest splits.json --final `
    --final-prediction-threshold <threshold> --epochs <fixed-epoch-count>
python training\evaluate_holdout.py --run-dir checkpoints_exp\<final-run> `
    --split-manifest splits.json --confirm-frozen
```

The holdout is never loaded while training. Do not use its score to tune another run; it has
then become development data.

</details>

<details>
<summary>Promote an accepted checkpoint into the installed package</summary>

Promotion is a package/release change, not a normal training step. After checking overlays and
downstream tracking on representative recordings, preview then promote the selected run:

```powershell
python training\promote_checkpoint.py --run-dir checkpoints_exp\<run-name> --dry-run
python training\promote_checkpoint.py --run-dir checkpoints_exp\<run-name> `
    --validation-note "Held-out session validation; see best.json."
```

Update `CHANGELOG.md`, run the repository checks in `AGENTS.md`, and verify the checkpoint,
metadata, and log are included in both package distributions.

</details>

### Developer fixture

`sample_data/` is a small public fixture for checking data flow and checkpoint writing, not
for training a useful model. See [sample_data/README.md](../sample_data/README.md) for its
commands and limits.
