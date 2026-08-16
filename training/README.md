# Model Training and Fine-Tuning

This folder contains the editable workflow for preparing masks, reviewing augmentation,
training the UNet, and promoting a selected checkpoint. `run_train.py` accepts terminal
arguments when they are supplied and otherwise uses its editable Spyder/IDE configuration.

Run repository scripts from the repository root. The training script treats the current
directory as its default data root unless `--data-root` is supplied.

## Environment

Use the project's environment and editable installation:

```powershell
conda activate pupil_tracking
python -m pip install -e .
```

Label creation also requires [Labelme](https://github.com/wkentaro/labelme). Install it in the same environment if `labelme.exe` and `labelme_export_json` are not already available.

## Data layout

Create these local folders at the repository root:

```text
labeled_data/
  <session>/        # one recording session: one animal, one date, one condition
    images/         # the frames, and their .json annotations
    masks/          # the masks, same filenames
```

One directory per recording session. The session is the grouping unit for the split, so it is a directory: an image cannot enter the pool without one. There is no train/validation folder split — `splits.json` decides what trains and what validates, so a labelled pair never moves on disk when the split changes. The former `images_train/`, `images_validation/`, `masks_train/`, `masks_validation/` are still read if a checkout still has them, so an older layout keeps working.

Images and masks must be PNG files whose filenames correspond one-to-one by stem within a session. Filenames need not be unique *between* sessions — an image is identified by `<session>/<filename>`. Use the original camera frames for training. For example, if the camera produces 300 x 300 frames, place those 300 x 300 PNG files in the image folders and create matching 300 x 300 masks.

For choosing which frames are worth labelling and how to record where they came from, see [`data_collection.md`](data_collection.md).

Adding data is: drop `labeled_data/<new session>/images/`, convert the annotations, refresh the split.

```powershell
python training\labelme_json2png.py --data-root . --session <new session>
python training\data_splits.py --data-root . --materialize
```

Existing sessions keep their folds; only the new one is packed.

When `run_train.py` loads an image and its mask, it automatically resizes both to the model's 148 x 148 input, so do not crop, resize, or pad them yourself. Resizing keeps the complete frame but makes it smaller; it does not cut away any part of the frame. For a non-square frame, the loader preserves the aspect ratio and adds black padding to produce a 148 x 148 square. The same operation is applied to the image and mask so they remain aligned.

The training loader may then apply small random transformations to the training pair as data augmentation. This is also automatic. Run `check_augmentation.py` only to review examples and confirm that the image and mask stay aligned; it is not a preprocessing step.

Each training utility defines an editable `DATA_ROOT` near the top. It defaults to the repository root for the full local dataset. To use the small public fixture included with a clone, change it to:

```python
DATA_ROOT = PROJECT_ROOT / "sample_data"
```

See [`sample_data/README.md`](../sample_data/README.md) for the fixture's scope and quick-start commands. Its eight training and four validation pairs are suitable for checking data flow and checkpoint writing, not for training a useful model.

## 1. Create masks with Labelme

1. Start Labelme with `labelme.exe` and annotate the pupil in each source image.
2. Save each JSON file beside its image in `labeled_data/<session>/images/`.
3. Run:

```powershell
python training\labelme_json2png.py --data-root .
python training\labelme_json2png.py --data-root . --session HQL091_sleep260820
```

The script walks every session folder (or just the ones named with `--session`), runs `labelme_export_json`, moves each generated `label.png` into that session's `masks/` under its image's filename, and removes Labelme's temporary export directory. Existing masks are skipped, so re-running only fills in what is missing.

Before training, compare the filenames and counts in each image/mask pair. A mask with the wrong image can train without an obvious file error while corrupting the model.

## Grouped splits

A *session* is one recording setting: one animal, one date, one condition. Sessions are
the unit that must not span the train/validation boundary, because the domain shift that
breaks this model is rig, camera angle, lighting, and animal state rather than animal
identity. Copying a neighbour's mask scores 0.652 IoU when the neighbour comes from the
same session and 0.399 when it comes from a different one, against a 0.02 seed noise
floor — that gap is what a non-grouped split hands the model for free.

Which session an image belongs to is **recorded, never inferred** — and the layout
records it: the session is the directory the pair sits in. A `session` flag in the
labelme JSON or a `provenance.csv` sidecar can override that for a batch that arrived
pre-mixed, and anything still unresolved collapses into one safe over-merged group.
Filenames are not parsed. [`data_collection.md`](data_collection.md) covers the sources
and the measurements that ruled out recovering the grouping from the images themselves.

Folds are also **stratified**: sessions are banded by median pupil diameter and median
background brightness, and a new session prefers a fold holding no session of its
diameter band, then the smallest fold. Grouping alone left three of five folds with no
small pupil at all; the pool now uses four folds, and every one of them holds some.

Generate the manifest once, then refresh it whenever labelled data is added:

```powershell
python training\data_splits.py --data-root . --out splits.json
python training\data_splits.py --data-root . --show   # census only, writes nothing
```

It prints a per-session table and a per-fold summary. Every image already in the
manifest keeps the session and fold it had, and a new image joins the fold its session
already sits in, so adding data does not invalidate earlier results. A provenance source
that contradicts the manifest is an error rather than a silent repack; `--reassign`
repacks everything deliberately and does invalidate earlier numbers.

`--split-manifest` is how the pool gets split. Without it the trainer looks for the old
fixed `images_train` / `images_validation` folders, which the maintained dataset no
longer has; that path remains only for a checkout still laid out the old way, such as
`sample_data`, and it shares recordings across the boundary so its IoU measures held-out
frames rather than generalisation.

### Seeing the folds on disk

The manifest is the record, but it is JSON. To look at the split as folders:

```powershell
python training\data_splits.py --data-root . --materialize
```

writes `folds/cv1/`, `folds/cv2/`, ... each with `images/` and `masks/`, plus `holdout/`
if the manifest sets one. `cvN` holds fold `N-1`. Folds partition the pool, so the whole
tree is one copy of the dataset; pass `--symlink` for none.

This is **derived output, one way**. The folders are rebuilt from the manifest and never
read back, so editing them changes nothing and the next `--materialize` overwrites them.
The manifest stays the record because it is committed and the image folders are not — a
fresh clone has `splits.json` and `provenance.csv` and no images at all.

### The holdout gate

Every fold's number feeds configuration choice, so none of them is a clean estimate of
the final model. Set one or more sessions aside to get one:

```powershell
python training\data_splits.py --data-root . --holdout HQL090_sleep251012 --out splits.json
python training\run_train.py --split-manifest splits.json --final
```

Holdout sessions appear in no fold — trained on never, validated on never — and
`--final` trains on everything else and validates against them. Choose the holdout by
condition rather than by animal, and note that at the current pool size two sessions is
15–27% of the labelled data. None is set by default.

## 2. Inspect augmentation

Open `check_augmentation.py` in your IDE or run:

```powershell
python training\check_augmentation.py
```

The script draws repeated augmented versions of training samples with the mask overlaid. Adjust `n_samples`, `n_augs_per_sample`, and `mask_transparency` in its final block as needed. The requested sample count is automatically capped at the available dataset size. Confirm that transforms keep the pupil mask aligned and do not create unrealistic crops before starting a long run.

## 3. Train a model from scratch

For an IDE run, edit the final `TrainingConfig(...)` block in `run_train.py`, especially:

- `finetune_checkpoint`: leave as `None` for fresh training.
- `early_stopping_patience` and `scheduler_patience`: how long to wait for a
  size-balanced validation improvement and when to lower the learning rate.
- `n_epochs`: maximum training epochs.
- `use_attention`: must match the desired UNet architecture.
- `threshold_candidates`: pixel-confidence thresholds evaluated on validation data.
- `tiny_max_diameter` and `large_min_diameter`: model-pixel cutoffs for stratified
  reporting and balanced sampling.
- DataLoader batch size and fresh-training Adam learning rate, currently `8` and `1e-3`.

Then run the file directly in your IDE. The editable block deliberately remains separate from
terminal argument parsing.

For a terminal run from a directory containing the four data folders, use:

```powershell
python training\run_train.py --run-name scratch_bal_lr1e-3_s0
```

Use `--data-root C:\path\to\training-data` when the folders are elsewhere. Run
`python training\run_train.py --help` for the fine-tuning checkpoint, output directory,
learning rate, epoch, batch-size, patience, seed, sampling, and attention options.

The script automatically uses CUDA when available. It oversamples training masks so the
represented tiny, medium, and large bins receive equal total sampling probability. Validation
IoU and Dice are calculated per image, and the early-stopping score gives each represented
size bin equal weight. Low-circularity masks are reported separately as a practical proxy for
partially occluded or irregular labels; that geometry alone does not prove occlusion.

Every run writes these local outputs under `checkpoints_exp/<run-name>/`, even if it stops
early or does not meet the promotion target. Set `run_name` explicitly for a planned
experiment, or let the trainer derive a concise name from the training mode, sampling mode,
learning rate, and seed. A nonempty run folder is never overwritten.

- `best.pth`: the best model weights, replaced only by a meaningful improvement.
- `best.json`: the selected prediction threshold, macro and size-stratified metrics, best epoch,
  and the complete run configuration.
- `train.log`: every completed epoch and its threshold/size metrics.

The promotion target is metadata, not a save gate. These files are experimental output and
are not installed package data.

## 4. Fine-tune an existing checkpoint

Set `finetune_checkpoint` in the final block of `run_train.py` to a compatible `.pth` file:

```python
finetune_checkpoint = (
    PROJECT_ROOT
    / "mouse_pupil_analysis"
    / "checkpoints"
    / "166pupils_thresh=0.4_iou=0.8749.pth"
)
```

The equivalent terminal command is:

```powershell
python training\run_train.py `
    --data-root . `
    --finetune-checkpoint "mouse_pupil_analysis\checkpoints\166pupils_thresh=0.4_iou=0.8749.pth" `
    --run-name ft_natural_lr1e-4_s1 `
    --learning-rate 1e-4 `
    --natural-sampling `
    --seed 1
```

The checkpoint architecture is detected from its weights. Fine-tuning automatically uses
`finetune_learning_rate` (`1e-4` by default); fresh training uses
`scratch_learning_rate` (`1e-3`). Keep the original training examples mixed with new hard
cases so adapting to one recording does not erase established behavior.

This loads model weights only. It starts a new optimizer, learning-rate scheduler, early-stopping counter, and log, so it is fine-tuning rather than an exact resume of an interrupted training run. Exact resume support would require saving and restoring those states in a structured training checkpoint.

## 5. Cross-validate a configuration

Use cross-validation to compare *configurations* — sampling, loss, augmentation,
architecture. Each fold trains on every other fold and validates on its own held-out
sessions, so every session is scored exactly once by a model that never saw that
setting:

```powershell
python training\run_cv.py --data-root . --split-manifest splits.json --out checkpoints_exp\cv
```

Add `--folds 0 2` to run a subset, and `--seed` to repeat the whole sweep. The driver
prints a per-fold table, a per-session IoU table, and three summary numbers:

- **mean per-session IoU** — the headline. Averaging over sessions rather than images
  keeps the largest session from dominating; one session is currently 28% of the pool.
- **worst session** — usually more actionable than the mean, since it names the setting
  to label or debug next.
- **image-weighted IoU** — comparable to the macro IoU this project reported historically.

Two caveats the output surfaces per fold. `balanced_iou` averages only the size bins a
fold actually contains, so it means different things in different folds and is not
comparable across them; the `bins scored` column shows which were present. And small
pupils concentrate in very few sessions, so the fold holding them trains with almost no
tiny masks — with size-balanced sampling on, that oversamples a handful of images
heavily.

This is not how the shipped checkpoint is built. Once a configuration wins, retrain it on
the whole pool and promote that, then gate the result as below.

## 6. Review and promote a checkpoint

Do not automatically place experimental output in `mouse_pupil_analysis/checkpoints/`. First:

1. Review training and validation curves in the generated log.
2. Evaluate the checkpoint on recordings that were not used for training or validation.
3. Inspect segmentation overlays and downstream diameter/center tracking, not IoU alone.
4. Compare against the currently packaged model on the same cases.

When a model is accepted, promote its run folder with:

```powershell
python training\promote_checkpoint.py `
    --run-dir checkpoints_exp\ft_natural_lr1e-4_s0 `
    --validation-note "Validation shares recording groups with training."
```

Add `--dry-run` first to see the filenames it would write. The script applies the concise
packaged naming pattern `<count>pupils_thresh=<value>_iou=<macro-value>`, strips local
absolute paths from the metadata and log header, and writes the weights, JSON metadata, and
log into `mouse_pupil_analysis/checkpoints/`. The model is always a UNet, attention is
detected from its weights, and the 148 x 148 resize-and-pad step is universal, so `unet`,
`atn`, and `resize` are intentionally omitted from the name. Seed, learning rate, sampling,
best epoch, balanced IoU, and other details stay in the matching log and JSON metadata.

Use `--validation-note` to state the honest scope of the reported numbers, especially when
validation is not independent of training. It is stored in the packaged metadata so the
caveat travels with the model rather than living only in a work log.

Default inference selects the packaged checkpoint with the highest IoU encoded in its
filename, then reads its threshold from JSON metadata (or the filename for older
checkpoints). The script never deletes anything; it lists superseded packaged checkpoints so
you can remove or archive them deliberately, which the release workflow requires.

Promotion is a package change, so also update `CHANGELOG.md`. If the calibrated threshold
differs from the superseded checkpoint's, reported diameters change for every user: measure
the difference on `sample_data/velocity_frames` and record it as a migration note. Then run
the repository checks and package build documented in `AGENTS.md` and verify that the
selected checkpoint, metadata, and log appear in both the wheel and source distribution.
