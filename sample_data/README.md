# Sample Data

This small public fixture lets you try the analysis and training workflow after downloading the
project. It contains 32 labeled image/mask pairs from 10 recording sessions, six full-size
unlabeled frames, and 31 consecutive frames for the velocity example.

## Contents

```text
sample_data/
|- labeled_frames/             # 32 image/mask pairs, arranged by recording session
|- training_data_split.json    # grouped fold assignment for the labeled pairs
|- unlabeled_frames/           # 6 full-size frames from two recordings
|- velocity_frames/            # 31 consecutive frames acquired at 97 Hz
|- provenance.csv              # source and transformation record for every shipped image
|- README.md
```

`training_data_split.json` tells the training tools which complete recording sessions belong to
each fold. `provenance.csv` records the source recording, source-frame suffix, and transformation
for every fixture image.

## Analyze the sample frames

The root [README's Sample data section](../README.md#sample-data) runs the 31-frame velocity
example. To analyze the full-size frames from one recording, run:

```powershell
run-pupil-analysis `
    --image_dir sample_data\unlabeled_frames\recording_250530 `
    --result_dir results\sample_unlabeled_250530 `
    --output_mask_dir results\sample_unlabeled_250530\overlays
```

Use `recording_250616` to try the second acquisition setup. Each folder is analyzed separately,
so the result plot represents one recording at a time.

## Check the training tools

The [training guide](../training/README.md) explains [augmentation](../training/README.md#inspect-augmentation),
[fold assignments](../training/README.md#2-set-aside-a-session-to-check-the-model), and
[cross-validation](../training/README.md#cross-validate-a-configuration). Use these
fixture-specific commands to try them with the bundled data:

```powershell
python training\preview_augmentation.py --data_root sample_data
python training\prepare_splits.py --labeled_frames_dir sample_data\labeled_frames --show
python training\run_cross_validation.py --labeled_frames_dir sample_data\labeled_frames --max_epochs 1 --checkpoint_dir checkpoints_exp\sample_cv
```

The cross-validation command writes its checkpoints, metadata, and log under
`checkpoints_exp\sample_cv`.

## Provenance

- The labeled pairs and unlabeled frames were copied unchanged from the project's local recordings.
- The fixture stores PNG masks directly; Labelme JSON annotations are not included.
- The velocity frames come from source frames `07212` through `07242` of
  `250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042`, prepared with grayscale
  conversion and the package's 148 x 148 resize-and-pad convention.
- The project has permission to publish these images and masks for collaboration and reproducible
  examples.

Generated results belong under the ignored `results/` directory.
