# Train or Fine-Tune a Mouse Pupil Segmentation Model

Follow this workflow to label your own data, train or fine-tune a model. The core procedures are:

1. Organize your labeled images into recording sessions (one folder for each session).
2. Add new sessions to folds.
3. Train a model on the labeled data.

Additionally, we also provide the 

## Contents

- [Core workflow](#core-workflow)
  - [Environment](#environment)
  - [1. Organize labeled images](#1-organize-labeled-images)
  - [2. Add new sessions ](#2-add-new-sessions)
  - [3. Train and validate](#4-train-and-validate)
- [Optional tools](#optional-tools)
  - [Choose frames to label](#choose-frames-to-label)
  - [Label images with Labelme](#label-a-batch-with-labelme)
  - [Review or adjust assignments](#3-review-or-adjust-assignments)
  - [Inspect augmentation](#inspect-augmentation)
  - [Use the experimental checkpoint](#5-use-the-experimental-checkpoint)
  - [Fine-tune a checkpoint](#fine-tune-a-checkpoint)
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

### 1. Organize labelled images

Create a folder named `labeled_frames/`. In it, place matching PNG images and masks arranged by 
recording session. For example:

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

Use the original camera-frame size. During training, the data loader automatically converts 
each image/mask pair to the model's 148 x 148 input: square frames are resized, while non-square
frames are resized proportionally and black-padded. Do not crop, resize, or pad data yourself.

### 2. Add new sessions

The training data from `labeled_frames` is grouped into folds, which is tracked by `training_data_split.json` in the repo root. 
Run this to add the labeled images from new sessions (previous step) to the folds in the training data.

```powershell
python training\prepare_splits.py
```

`prepare_splits.py` assigns news sessions to folds automatically while balancing for pupil size, lighting condition in each fold.
Existing assignments are preserved when new sessions are assigned. It updates the `training_data_split.json` and does not move images 
or masks on disk. It creates four folds by default. If you'd like a different number of folds, add `--n_folds`. For example,
```bash
python training\prepare_splits.py --n_folds 5
```

<details>
<summary><strong>Click to see additional prepare_splits.py arguments</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--labeled_frames_dir` | `./labeled_frames` | Use a labelled pool outside the repository. Its parent receives the split record. |
| `--n_folds` | Existing count, otherwise `5` | Set the number of CV folds when first creating the split. |
| `--final_test_session` | None | Repeat for each session to exclude from CV while comparing configurations. |
| `--validation_session` | None | Repeat for each session reserved for normal validation-backed training. |
| `--show` | Off | Print the proposed census without writing the split record. |
| `--reassign` | Off | Deliberately repack every session and invalidate comparisons to earlier assignments. |

</details>

### 3. Train and validate
There are two approaches to training a model. You can train a model using all data in `labeled_frames/` with
the `default_all_labeled_training_config.json` included in this repo which sets the training configurations like 
maximum epochs, early stopping, and learning rate scheduling, etc., for you, based on the developer's experiments.
To train this way, supply `--training_config_path`,
```powershell
python training\run_train.py --training_config_path training\default_all_labeled_training_config.json --checkpoint_dir checkpoints_exp\scratch
```

Alternatively, you can set a validation set by following [Review or adjust assignments](#3-review-or-adjust-assignments)
```bash
python training\run_train.py --checkpoint_dir checkpoints_exp\scratch
```

> CUDA is selected automatically when available.

<details>
<summary><strong>Click to see all run_train.py arguments</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--labeled_frames_dir` | `./labeled_frames` | Train from a labelled pool outside the repository. |
| `--checkpoint_dir` | A new directory under `checkpoints_exp/` | Choose where this run writes its checkpoint and metadata. |
| `--finetune_checkpoint` | Fresh training | Start normal validation-backed training from compatible weights. |
| `--training_config_path` | Normal training | Use the all-labelled recipe emitted by cross-validation. It owns model settings and ignores the split record. |

</details>


## Optional tools

### Fine-tune a checkpoint

Fine-tuning is often faster than training from scratch when you added new sessions to the
existing labelled pool. It uses the same held-out-fold validation workflow:

```powershell
python training\run_train.py `
    --finetune_checkpoint "mouse_pupil_analysis\checkpoints\166pupils_thresh=0.4_iou=0.8749.pth" `
    --checkpoint_dir checkpoints_exp\ft
```

Fine-tuning loads model weights but starts a new optimizer, scheduler, and training log. Keep
the prior labelled sessions in the pool alongside newly labelled difficult cases to reduce
forgetting.

### Label a batch with Labelme

Labelme is one supported intake route, not a requirement. If you use it, save the annotated
JSON files beside their source images. Give every genuinely new recording session its own
session name; matching text such as `Purple_trial5` is not proof that two recordings belong
together. The importer never merges into an existing session directory.

Import the complete batch:

```powershell
python training\labelme_json2png.py --source <annotation-folder> --session <new-session>
```

The importer creates the session's `images/`, `masks/`, and optional `uncertain/` folders,
then refreshes the split manifest. Use `pupil` for a visible pupil polygon,
`no_visible_pupil` for a confident true-negative frame, and `uncertain` for a frame that
should be retained but excluded from segmentation loss. See [data_collection.md](data_collection.md)
for the detailed annotation policy.

For example, a Labelme-reviewed recommendation queue for one recording can become a separate
training session without touching any other `Purple_trial5` folder:

```powershell
python training\labelme_json2png.py `
    --source "frames_to_label\260812_3582_Purple_trial5_pupil_recording_2026-08-12T15-05-55.154-1\recommended" `
    --session "260812_3582_Purple_trial5_pupil_recording_2026-08-12T15-05-55.154-1"

```

This command writes only to
`labeled_frames/260812_3582_Purple_trial5_pupil_recording_2026-08-12T15-05-55.154-1/`
and refreshes `training_data_split.json`; it cannot overwrite or add frames to
`labeled_frames/260812_3582_Purple_trial5/`.

<details>
<summary><strong>labelme_json2png.py arguments</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--source` | Required | Folder containing Labelme JSON files and their source images. |
| `--session` | Required | New recording-session name under this repository's `labeled_frames/`. |

</details>

### Review or adjust assignments

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

<details>
<summary><strong>review_splits.py arguments</strong></summary>

This command has no command-line arguments. It opens the current repository's
`training_data_split.json`; run `prepare_splits.py` first if that record does not exist.

</details>

Folds are used only by cross-validation. The validation session is excluded from CV and is used
by the normal training command below. Assign one before starting a validation-backed run.

### Choose frames to label

You may label whichever frames make sense for your experiment. The recommender is available
when you want help prioritising a larger recording:

```powershell
python training\recommend_frames.py `
    --video D:\data\recording.avi `
    --budget 20

python training\recommend_frames.py `
    --frames D:\data\already_extracted `
    --budget 20
```

The recommender uses the current completed CV run at `checkpoints_exp/cv` by default.
`--checkpoint_dir` optionally selects another complete cross-validation run, not an individual
model folder. The recommender discovers every immediate `*/best.pth` fold checkpoint inside it
and uses their disagreement to rank frames. Use a committee whose models did not train on the
recording you are selecting from.

By default, its outputs go under `frames_to_label/<session>/`. After labelling, put the
resulting image/mask pairs in `labeled_frames/<session>/` by any supported method and run
`prepare_splits.py`.

<details>
<summary><strong>recommend_frames.py arguments</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--video` | One of `--video` or `--frames` | Video to sample and score. |
| `--frames` | One of `--video` or `--frames` | Directory of already-extracted PNG frames to score. |
| `--checkpoint_dir` | `checkpoints_exp/cv` | Optional complete CV-run directory containing fold subdirectories with `best.pth`. |
| `--budget` | `20` | Number of frames to recommend. |

</details>

### Inspect augmentation

`preview_augmentation.py` is a visual diagnostic, not preprocessing:

```powershell
python training\preview_augmentation.py
```

It renders augmented image/mask pairs so you can check that the pupil remains aligned and the
transforms look plausible.

<details>
<summary><strong>preview_augmentation.py arguments</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--data_root` | Repository root | Use a different repository-style root containing `labeled_frames/`. |

</details>

### Use the experimental checkpoint

Each run writes to the specified checkpoint directory. Without `--checkpoint_dir`, the trainer
creates a collision-safe directory under `checkpoints_exp/`.

- `best.pth` — selected model weights.
- `best.json` — selected threshold, validation metrics, epoch, and full configuration.
- `train.log` — per-epoch training and validation record.

All-labeled training from a CV configuration writes `all_data.pth` and `all_data.json` instead
of `best.*`; its metadata records the recipe used.

Treat this folder as experimental output. Test the checkpoint on representative recordings
before replacing the package's default model.


### Cross-validate a configuration

Use cross-validation when you need to compare configurations or estimate how sensitive a
result is to the held-out session group. It takes turns holding out each fold and
never loads the validation session:

```powershell
python training\run_cross_validation.py --checkpoint_dir checkpoints_exp\cv
```

Each fold has a 200-epoch ceiling and stops after 20 non-improving validation epochs; these are
selection bounds, not a claimed final-training duration. Use its per-session results to compare
candidate settings. It also writes `training_config.json`, a complete all-labeled training recipe
using the median successful-fold epoch rounded up to the next 100, its calibrated threshold, and a
portable `summary.json` provenance reference. The repository also tracks
`training/default_all_labeled_training_config.json`, the fixed settings used for the 516-pair
all-data baseline, for repeatable expansion of that baseline. Train the production model from
either recipe with:

```powershell
python training\run_train.py `
    --training_config_path training\default_all_labeled_training_config.json `
    --checkpoint_dir checkpoints_exp\all_labeled
```

That command deliberately reads every valid image/mask pair under `labeled_frames/` and ignores
`training_data_split.json`, including any session formerly reserved as a final test. Use it when you are ready
to trust the CV-selected recipe and inspect the resulting model on representative unlabeled
recordings.

To rerun only specific existing folds, use `--cv_folds`, for example `--cv_folds 0 2`.
That writes `partial_summary.json` only; it deliberately does not create a production training
configuration.

<details>
<summary><strong>run_cross_validation.py arguments</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--labeled_frames_dir` | `./labeled_frames` | Cross-validate a labelled pool outside the repository. |
| `--checkpoint_dir` | `checkpoints_exp/cv` beside the data root | Directory that receives one fold folder per CV split. |
| `--cv_folds` | Every fold | Rerun only selected existing fold indices; this writes a partial summary only. |
| `--finetune_checkpoint` | Fresh training | Evaluate a compatible starting checkpoint, provided it was not trained on this pool. |

</details>

### Package an accepted checkpoint

<details>
<summary><strong>Package an accepted checkpoint into the installed application</strong></summary>

Packaging is a package/release change, not a normal training step. After checking overlays and
downstream tracking on representative recordings, preview then package either a
validation-selected run or an all-labeled refit:

```powershell
python training\package_checkpoint.py --run_dir checkpoints_exp\<run-name> --dry_run
python training\package_checkpoint.py --run_dir checkpoints_exp\<run-name> `
    --validation_note "Held-out session validation; see best.json."

# An all_data.* refit must verify the CV recipe that selected its fixed settings.
python training\package_checkpoint.py --run_dir checkpoints_exp\<all-data-run> `
    --training_config_path checkpoints_exp\cv\training_config.json `
    --validation_note "All-labeled refit; filename score is the mean four-fold CV macro IoU, not an evaluation of the final weights."
```

Update `CHANGELOG.md`, run the repository checks in `AGENTS.md`, and verify the checkpoint,
metadata, and log are included in both package distributions.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--run_dir` | Required | Validation-selected `best.*` run or all-labeled `all_data.*` run. |
| `--training_config_path` | Empty | CV-generated recipe for an all-labeled run; its summary hashes are verified before packaging. |
| `--validation_note` | Empty | Scope note saved with the packaged metadata. |
| `--dry_run` | Off | Print intended packaged filenames without writing files. |

</details>

### Developer fixture

`sample_data/` is a small public fixture for checking data flow and checkpoint writing, not
for training a useful model. See [sample_data/README.md](../sample_data/README.md) for its
commands and limits.
