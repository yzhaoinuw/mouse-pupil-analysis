# Train or Fine-Tune a Mouse Pupil Segmentation Model

Use this guide to label your own images and train a model to segment the pupil. The core workflow
is:

1. Organize labeled image and mask pairs by recording session.
2. Set aside one recording session to check the model during training.
3. Train the model and check its results on representative recordings.

## Contents

- [Core workflow](#core-workflow)
  - [Environment](#environment)
  - [1. Organize labeled images](#1-organize-labeled-images)
  - [2. Set aside a session to check the model](#2-set-aside-a-session-to-check-the-model)
  - [3. Train the model](#3-train-the-model)
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

Run these commands from the folder where you downloaded this project:

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
    uncertain/                       # optional; kept outside segmentation training
```

Keep every frame from the same recording in one `<session>` directory. A filename only
needs to be unique within that session; `session/frame_00001.png` is the data identity.
`prepare_splits.py` creates the training and validation groups from these session folders.

Use the original camera-frame size. During training, the data loader automatically converts
each image/mask pair to the model's 148 x 148 input: square frames are resized, while non-square
frames are resized proportionally and black-padded. Provide the original image and mask pairs;
the data loader handles this conversion.

For pointers on choosing images to add to the training data and on labeling images, see
[Choose frames to label](#choose-frames-to-label) and [Label images with Labelme](#label-images-with-labelme).

### 2. Set aside a session to check the model

Frames from one recording session are often very similar. To check whether the model works on a
new recording, keep one complete session aside while it learns from the others. This reserved
session is called the **validation session**.

First, arrange the remaining sessions into balanced groups, called **folds**. This keeps every
recording session together, spreads different pupil sizes and lighting conditions across the
groups, and prepares the data for the optional cross-validation workflow. Run this after adding
labeled frames, including a new session:

```powershell
python training\prepare_splits.py
```

`prepare_splits.py` creates the fold-assignment record, `training_data_split.json`, beside
`labeled_frames/`. It keeps every session together, assigns new sessions to folds while balancing
pupil size and lighting, and preserves earlier assignments. The first record has five folds by
default. Use `--n_folds` only when creating it; changing an existing fold count requires
`--reassign`.

Reserve one complete session for validation before the first training run. Replace `<session>`
with one of your session-folder names:

```powershell
python training\prepare_splits.py --validation_session <session>
```

During training, the validation session chooses the best checkpoint, the stopping point, and the
prediction threshold. It also stays separate from optional cross-validation.

<details>
<summary><strong>Click here for more prepare_splits.py options</strong></summary>

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


### 3. Train the model

The trainer learns from the other sessions and checks its progress against the validation session
you reserved above. Start a new training run with:

```powershell
python training\run_train.py --checkpoint_dir checkpoints_exp\scratch
```

The run saves its selected model, settings, and training record in `checkpoints_exp\scratch`.

<details>
<summary><strong>Click here for more run_train.py options</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--labeled_frames_dir` | `./labeled_frames` | Train from a labeled pool outside the repository. |
| `--checkpoint_dir` | A new directory under `checkpoints_exp/` | Choose where this run writes its checkpoint and metadata. |
| `--finetune_checkpoint` | Fresh training | Start normal validation-backed training from compatible weights. |
| `--training_config_path` | Normal training | Use the all-labeled recipe emitted by cross-validation. It owns model settings and ignores the fold-assignment record. |

</details>


## Additional tools

### Choose frames to label

You can label any frames that serve your experiment. For a long recording, use the recommender to
prioritize a small set of frames that the current models find most informative:

```powershell
python training\recommend_frames.py `
    --video D:\data\recording.avi `
    --budget 20

python training\recommend_frames.py `
    --frames D:\data\already_extracted `
    --budget 20
```

The recommender uses the current completed CV run at `checkpoints_exp/cv` by default.
`--checkpoint_dir` can select another completed cross-validation run. It compares the models in
that run and ranks frames where their predictions differ. Choose a run trained from other
recordings when selecting frames from a new recording.

By default, its outputs go under `frames_to_label/<session>/`. After labelling, put the
resulting image/mask pairs in `labeled_frames/<session>/` by any supported method and run
`prepare_splits.py`.

<details>
<summary><strong>Click here for recommend_frames.py arguments</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--video` | One of `--video` or `--frames` | Video to sample and score. |
| `--frames` | One of `--video` or `--frames` | Directory of already-extracted PNG frames to score. |
| `--checkpoint_dir` | `checkpoints_exp/cv` | Optional complete CV-run directory containing fold subdirectories with `best.pth`. |
| `--budget` | `20` | Number of frames to recommend. |

</details>

### Label images with Labelme

[Labelme](https://github.com/wkentaro/labelme) lets you draw the pupil boundary on selected
images. Choose a small, varied set of frames. If you used the recommender, open a recommended
frame in Labelme and look at the nearby frames too: difficult cases often need that context, and
a nearby frame with a clear boundary can be a useful training example. When a recommended frame
is closed or too low-resolution to label confidently, choose a nearby frame with a clear pupil
boundary instead.

Use `pupil` for a visible pupil, `no_visible_pupil` when you are certain no pupil is visible
(for example, when the eye is closed), and `uncertain` for a frame to retain outside the
segmentation loss. Labelme saves annotations as JSON files beside the source images; the importer
below creates the PNG masks. See [data_collection.md](data_collection.md) for the detailed policy.

Give every genuinely new recording session its own session name. The importer uses that name for
the session folder under `labeled_frames/` and keeps the session's images and masks together.

Import a session when labeling is completed:

```powershell
python training\labelme_json2png.py --source <annotation-folder> --session <new-session>
```

The importer creates the session's `images/`, `masks/`, and optional `uncertain/` folders under
`labeled_frames/` and refreshes the fold-assignment record.

<details>
<summary><strong>Click here for more labelme_json2png.py options</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--source` | Required | Folder containing Labelme JSON files and their source images. |
| `--session` | Required | New recording-session name under this repository's `labeled_frames/`. |

</details>


### Review fold assignments

Before training, you can check that the folds have a similar range of pupil sizes and lighting
conditions, and choose the validation session. Open the interactive local fold manager to inspect
the session statistics or change an assignment:

```powershell
python training\review_folds.py
```

`review_folds.py` opens the manager and updates the fold-assignment record. Drag whole sessions
between folds or into the **validation session**, then save. The charts update as you work.


### Inspect augmentation

During training, the pipeline creates varied versions of each image, such as small shifts, zooms,
and padding changes. These variations help the model recognize pupils in new recordings. Use this
tool when you want to see those changes and confirm that the mask still matches the pupil:

```powershell
python training\preview_augmentation.py
```

It shows augmented image/mask pairs with the mask overlaid on the image.

<details>
<summary><strong>Click here for preview_augmentation.py arguments</strong></summary>

| Argument | Default | Purpose |
| --- | --- | --- |
| `--data_root` | Repository root | Use a different repository-style root containing `labeled_frames/`. |

</details>


### Use the experimental checkpoint

After training, the checkpoint directory holds the files you need to inspect or use the new model.
The command above writes to its chosen directory; omitting `--checkpoint_dir` creates a new
directory under `checkpoints_exp/`.

- `best.pth` — selected model weights.
- `best.json` — selected threshold, validation metrics, epoch, and full configuration.
- `train.log` — per-epoch training and validation record.

All-labeled training from a CV configuration writes `all_data.pth` and `all_data.json` instead
of `best.*`; its metadata records the recipe used.

Treat this folder as experimental output. Test the checkpoint on representative recordings
before replacing the package's default model.


### Cross-validate a configuration

Use cross-validation when you need to compare configurations or estimate how sensitive a
result is to the held-out session group. It takes turns holding out each fold while keeping the
normal validation session separate:

```powershell
python training\run_cross_validation.py --checkpoint_dir checkpoints_exp\cv
```

Each fold trains for up to 200 epochs and stops after 20 validation epochs without improvement.
Use its per-session results to compare candidate settings. It also writes `training_config.json`,
a complete all-labeled training recipe
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

That command trains from every valid image/mask pair under `labeled_frames/`. Use it when you are
ready to trust the CV-selected recipe and inspect the resulting model on representative unlabeled
recordings.

To rerun only specific existing folds, use `--cv_folds`, for example `--cv_folds 0 2`. This
writes `partial_summary.json`, a report for the selected folds.

<details>
<summary><strong>Click here for run_cross_validation.py arguments</strong></summary>

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
<summary><strong>Click here to package an accepted checkpoint into the installed application</strong></summary>

After you have accepted a checkpoint, package it to make it the model used by the installed
application. First check its overlays and downstream tracking on representative recordings, then
preview and package either a validation-selected run or an all-labeled refit:

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

`sample_data/` is a small public example data set for checking that the training workflow and
checkpoint writing work. See [sample_data/README.md](../sample_data/README.md) for its commands
and limits.
