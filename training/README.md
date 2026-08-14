# Model Training and Fine-Tuning

This folder contains the maintained local workflow for preparing masks, reviewing augmentation, training the UNet, and promoting a selected checkpoint. These are editable research scripts rather than installed command-line tools, so their configuration is intentionally kept near the top of each file for terminal or IDE runs.

Run commands from the repository root. All paths in the scripts are anchored to that root.

## Environment

Use the project's environment and editable installation:

```powershell
conda activate mouse_pupil_analysis
python -m pip install -e .
```

Label creation also requires [Labelme](https://github.com/wkentaro/labelme). Install it in the same environment if `labelme.exe` and `labelme_export_json` are not already available.

## Data layout

Create these local folders at the repository root:

```text
images_train/
masks_train/
images_validation/
masks_validation/
```

Images and masks must be PNG files whose filenames correspond one-to-one by stem. Keep the validation set independent of the training set. Use the original camera frames for training. For example, if the camera produces 300 x 300 frames, place those 300 x 300 PNG files in the image folders and create matching 300 x 300 masks.  

When `run_train.py` loads an image and its mask, it automatically resizes both to the model's 148 x 148 input, so do not crop, resize, or pad them yourself. Resizing keeps the complete frame but makes it smaller; it does not cut away any part of the frame. For a non-square frame, the loader preserves the aspect ratio and adds black padding to produce a 148 x 148 square. The same operation is applied to the image and mask so they remain aligned.

The training loader may then apply small random transformations to the training pair as data augmentation. This is also automatic. Run `check_augmentation.py` only to review examples and confirm that the image and mask stay aligned; it is not a preprocessing step.

Each training utility defines an editable `DATA_ROOT` near the top. It defaults to the repository root for the full local dataset. To use the small public fixture included with a clone, change it to:

```python
DATA_ROOT = PROJECT_ROOT / "sample_data"
```

See [`sample_data/README.md`](../sample_data/README.md) for the fixture's scope and quick-start commands. Its eight training and four validation pairs are suitable for checking data flow and checkpoint writing, not for training a useful model.

## 1. Create masks with Labelme

1. Start Labelme with `labelme.exe` and annotate the pupil in each source image.
2. Save each JSON file beside its image in `images_train/` or `images_validation/`.
3. In `labelme_json2png.py`, set `dataset_type` to `"train"` or `"validation"`.
4. Run:

```powershell
python training\labelme_json2png.py
```

The script runs `labelme_export_json`, moves each generated `label.png` into the matching mask folder, and removes Labelme's temporary export directory. Existing masks with the same stem are skipped.

Before training, compare the filenames and counts in each image/mask pair. A mask with the wrong image can train without an obvious file error while corrupting the model.

## 2. Inspect augmentation

Open `check_augmentation.py` in your IDE or run:

```powershell
python training\check_augmentation.py
```

The script draws repeated augmented versions of training samples with the mask overlaid. Adjust `n_samples`, `n_augs_per_sample`, and `mask_transparency` in its final block as needed. The requested sample count is automatically capped at the available dataset size. Confirm that transforms keep the pupil mask aligned and do not create unrealistic crops before starting a long run.

## 3. Train a model from scratch

Edit the hyperparameter block in `run_train.py`, especially:

- `finetune_checkpoint`: leave as `None` for fresh training.
- `early_stopping_patience` and `scheduler_patience`: how long to wait for a
  size-balanced validation improvement and when to lower the learning rate.
- `n_epochs`: maximum training epochs.
- `use_attention`: must match the desired UNet architecture.
- `threshold_candidates`: pixel-confidence thresholds evaluated on validation data.
- `tiny_max_diameter` and `large_min_diameter`: model-pixel cutoffs for stratified
  reporting and balanced sampling.
- DataLoader batch size and fresh-training Adam learning rate, currently `8` and `1e-3`.

Then run the file in your IDE or execute:

```powershell
python training\run_train.py
```

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(use_attention=use_attention).to(device)

finetune_checkpoint = (
    PROJECT_ROOT
    / "mouse_pupil_analysis"
    / "checkpoints"
    / "unet_atn_resize_166pupils_thresh=0.7_iou=0.9158.pth"
)
```

The checkpoint architecture is detected from its weights. Fine-tuning automatically uses
`finetune_learning_rate` (`1e-4` by default); fresh training uses
`scratch_learning_rate` (`1e-3`). Keep the original training examples mixed with new hard
cases so adapting to one recording does not erase established behavior.

This loads model weights only. It starts a new optimizer, learning-rate scheduler, early-stopping counter, and log, so it is fine-tuning rather than an exact resume of an interrupted training run. Exact resume support would require saving and restoring those states in a structured training checkpoint.

## 5. Review and promote a checkpoint

Do not automatically place experimental output in `mouse_pupil_analysis/checkpoints/`. First:

1. Review training and validation curves in the generated log.
2. Evaluate the checkpoint on recordings that were not used for training or validation.
3. Inspect segmentation overlays and downstream diameter/center tracking, not IoU alone.
4. Compare against the currently packaged model on the same cases.

When a model is accepted, rename `best.pth` to preserve the packaged
`_thresh=<value>_iou=<value>` naming contract, using the calibrated threshold and reviewed IoU
from `best.json`. Copy the renamed weights, matching JSON metadata, and `train.log` into
`mouse_pupil_analysis/checkpoints/` as an intentional package change. Default inference
selects the packaged checkpoint with the highest IoU encoded in its filename, then reads its
threshold from JSON metadata (or the filename for older checkpoints). Remove or archive
superseded packaged candidates deliberately.

After promotion, run the repository checks and package build documented in `AGENTS.md`, then
verify that the selected checkpoint, metadata, and log appear in both the wheel and source
distribution.
