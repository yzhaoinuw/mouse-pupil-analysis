# Train or Fine-Tune a Mouse Pupil Segmentation Model

Use this guide to label your own images and train a model to segment the pupil. You do not need
to choose model settings for a first run. The core workflow is:

1. Organize labeled image and mask pairs by recording session.
2. Create or update the fold assignments, then reserve one session for validation.
3. Train a model and check its results on representative recordings.

## Contents

- [Core workflow](#core-workflow)
  - [Environment](#environment)
  - [1. Organize labeled images](#1-organize-labeled-images)
  - [2. Create and review fold assignments](#2-create-and-review-fold-assignments)
  - [3. Train with validation](#3-train-with-validation)
- [Additional tools](#additional-tools)
  - [Choose frames to label](#choose-frames-to-label)
  - [Label images with Labelme](#label-images-with-labelme)
  - [Review fold assignments](#review-fold-assignments)
  - [Inspect augmentation](#inspect-augmentation)
  - [Use the experimental checkpoint](#use-the-experimental-checkpoint)
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

### 1. Organize labeled images

Create a `labeled_frames/` folder. Inside it, place matching PNG images and masks for each
recording session:

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
Do not create separate `train` and `validation` folders or move pairs between folders.

Use the original camera-frame size. During training, the data loader automatically converts
each image/mask pair to the model's 148 x 148 input: square frames are resized, while non-square
frames are resized proportionally and black-padded. Do not crop, resize, or pad data yourself.

For pointers on choosing images to add to the training data and on labeling images, see
[Choose frames to label](#choose-frames-to-label) and [Label images with Labelme](#label-images-with-labelme).

### 2. Create and review fold assignments

The fold-assignment record is `training_data_split.json`, stored beside `labeled_frames/`.
Run this after adding labeled frames, including a new session:

```powershell
python training\prepare_splits.py
```

`prepare_splits.py` keeps every session together, assigns new sessions to folds while balancing
pupil size and lighting, and preserves existing assignments. It does not move images or masks.
The first record has five folds by default. Use `--n_folds` only when creating it; changing an
existing fold count requires `--reassign`.

Reserve one complete session for validation before the first training run. Replace `<session>`
with one of your session-folder names:

```powershell
python training\prepare_splits.py --validation_session <session>
```

The validation session stays out of cross-validation and chooses the best checkpoint, early-
stopping point, and prediction threshold for a normal training run.

<details>
<summary><strong>More prepare_splits.py options</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--labeled_frames_dir` | `./labeled_frames` | Use a labeled pool outside the repository. Its parent receives the fold-assignment record. |
| `--n_folds` | Existing count, otherwise `5` | Set the number of CV folds when first creating the record; use `--reassign` to change it later. |
| `--final_test_session` | None | Repeat for each session to exclude from CV while comparing configurations. |
| `--validation_session` | None | Repeat for each session reserved for normal validation-backed training. |
| `--show` | Off | Print the proposed census without writing the fold-assignment record. |
| `--reassign` | Off | Deliberately repack every session and invalidate comparisons to earlier assignments. |

</details>

To inspect or adjust the assignment, use [Review fold assignments](#review-fold-assignments).


### 3. Train with validation

Train from scratch with the validation session you reserved above:

```powershell
python training\run_train.py --checkpoint_dir checkpoints_exp\scratch
```

The held-out session selects the best checkpoint, controls early stopping and learning-rate
scheduling, and calibrates the prediction threshold. CUDA is selected automatically when
available.

<details>
<summary><strong>More run_train.py options</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--labeled_frames_dir` | `./labeled_frames` | Train from a labeled pool outside the repository. |
| `--checkpoint_dir` | A new directory under `checkpoints_exp/` | Choose where this run writes its checkpoint and metadata. |
| `--finetune_checkpoint` | Fresh training | Start normal validation-backed training from compatible weights. |
| `--training_config_path` | Normal training | Use the all-labeled recipe emitted by cross-validation. It owns model settings and ignores the fold-assignment record. |

</details>


## Additional tools

### Choose frames to label

You can label any frames that serve your experiment. Use the recommender when you want help
prioritizing frames from a larger recording:

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

### Label images with Labelme

[Labelme](https://github.com/wkentaro/labelme) is one recommended labeling tool, not a
requirement. Use `pupil` for a visible pupil, `no_visible_pupil` when you are certain no pupil
is visible (for example, when the eye is closed), and `uncertain` for a frame to retain outside
the segmentation loss. Labelme saves annotations as JSON files beside the source images; the
importer below creates the PNG masks. See [data_collection.md](data_collection.md) for the
detailed policy.

Give every genuinely new recording session its own session name. Matching text such as
`Purple_trial5` does not prove that two recordings belong together, and the importer never merges
into an existing session directory.

Import a session when labeling is completed:

```powershell
python training\labelme_json2png.py --source <annotation-folder> --session <new-session>
```

The importer creates the session's `images/`, `masks/`, and optional `uncertain/` folders under
`labeled_frames/` and refreshes the fold-assignment record.

<details>
<summary><strong>More labelme_json2png.py options</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--source` | Required | Folder containing Labelme JSON files and their source images. |
| `--session` | Required | New recording-session name under this repository's `labeled_frames/`. |

</details>


### Review fold assignments

Open the interactive local fold manager to inspect session statistics or change an assignment:

```powershell
python training\review_folds.py
```

Do not open `fold_manager.html` directly: `review_folds.py` starts the local service that reads
and safely updates the fold-assignment record. Drag whole sessions between folds or into the
**validation session**, then save. The charts update as you work.


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
| `--labeled_frames_dir` | `./labeled_frames` | Cross-validate a labeled pool outside the repository. |
| `--checkpoint_dir` | `checkpoints_exp/cv` beside the data root | Directory that receives one folder for each CV fold. |
| `--cv_folds` | Every fold | Rerun only selected existing fold indices; this writes a partial summary only. |
| `--finetune_checkpoint` | Fresh training | Evaluate a compatible starting checkpoint, provided it was not trained on this pool. |

</details>


### Fine-tune a checkpoint

Fine-tuning is often faster than training from scratch when you added new sessions to the
existing labeled pool. It uses the same held-out-fold validation workflow:

```powershell
python training\run_train.py `
    --finetune_checkpoint "path\to\existing_checkpoint.pth" `
    --checkpoint_dir checkpoints_exp\ft
```

Fine-tuning loads model weights but starts a new optimizer, scheduler, and training log. Keep
the prior labeled sessions in the pool alongside newly labeled difficult cases to reduce
forgetting.


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
