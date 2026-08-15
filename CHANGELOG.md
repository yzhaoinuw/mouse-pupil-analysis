# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Recording-grouped, condition-stratified data splits. `training/data_splits.py` groups the
  labelled pool into *sessions* — one animal, one date, one condition — and packs whole
  sessions into cross-validation folds, writing the assignment to a `splits.json` manifest.
  `images_train/` and `images_validation/` are now read as one flat pool, so re-splitting
  moves no labelled files. `run_train.py` gains `--split-manifest` and `--fold`; without
  them it keeps the previous fixed-folder behaviour. Grouping is worth 0.25 IoU against a
  0.02 seed noise floor: copying a nearest neighbour's mask scores 0.652 when the neighbour
  shares a session and 0.399 when it does not.
- Session identity is *recorded at intake, never inferred* (`training/provenance.py`). It
  comes from an intake subfolder, a `session` flag in the labelme JSON, or a
  `provenance.csv` sidecar, in that order of precedence; anything unresolved collapses into
  a single over-merged group, which costs data efficiency but cannot leak. No filename is
  parsed. Once an image is in the manifest its session and fold are frozen, and a
  provenance source that later contradicts the record raises instead of silently repacking.
  Recovering the grouping from the images was tested and does not work — crop geometry
  splits 6 of 16 sessions, and agglomerative clustering on preprocessed frames tears 3 at
  k=5 and 6 at k=10.
- Fold stratification on pupil size and lighting. Sessions are banded by median mask
  diameter and by median background brightness (new
  `mouse_pupil_analysis.augmentation.image_background_brightness`), and a new session
  prefers a fold holding no session of its diameter band, then the smallest fold.
  Grouping alone had left a 3.03x spread in median diameter across folds with only 2 of 5
  containing any small mask; at the four folds the pool now uses, it is 1.60x and 4 of 4.
  Letting only the *absence* of a
  band outrank fold size is what makes one rule work both when packing from scratch and
  when sessions arrive one at a time — over 200 simulated arrival orders it holds fold
  sizes to a 1.15x median spread against 1.33x for ranking by band count throughout, with
  identical band coverage.
- A holdout gate. `data_splits.py --holdout SESSION` sets sessions aside from every fold,
  and `run_train.py --final` trains on everything else and validates against them — the
  only number in the project measured on data the training procedure was never tuned on.
- `training/run_cv.py`, which runs every fold and reports mean per-session IoU. Averaging
  over sessions rather than images stops the largest session — currently 28% of the pool —
  from dominating the headline number.
- `training/data_collection.md`, documenting which incoming frames are worth labelling,
  how to record the session a batch came from, and how a labelled batch joins the split.
- `reports/`, holding dated analyses and the scripts that regenerate their numbers.
  `reports/2026-08-14-checkpoint-noise-floor.md` measures the run-to-run spread of the
  model-selection metric, and `reports/scripts/` provides the seed study, the run summariser,
  a dataset/leakage census, IoU-plus-diameter-bias validation diagnostics, and a promotion
  gate that runs candidates over real frames validation does not cover.
- Apple MPS support in `training/run_train.py` via `--device {auto,cuda,mps,cpu}`. `auto`
  prefers CUDA, then MPS, then CPU, which is roughly 4.6x faster than CPU on Apple silicon.
  The resolved device is recorded in the training log, since MPS, CUDA, and CPU kernels are
  not bit-identical.
- `training/promote_checkpoint.py`, which turns one `checkpoints_exp/<run-name>/` folder into
  the three packaged checkpoint files. It applies the concise naming pattern, strips local
  absolute paths from the metadata and log header, and records run provenance and a
  `validation_note` under a fixed schema, so a promotion is reproducible from its run folder
  instead of assembled by hand. `run_train.py` now records `training_examples` and
  `training_mode` in `best.json` and the log header to make that transform possible.
- Terminal arguments for `training/run_train.py`, while running the script without arguments
  retains its editable Spyder/IDE configuration.
- Fine-tuning support in `training/run_train.py`, with a lower default fine-tuning learning
  rate, size-balanced sampling, per-image and size-stratified validation, calibrated
  prediction thresholds, and best-checkpoint/metadata/log retention even when early stopping
  occurs below the promotion target. Experimental artifacts are grouped in descriptive,
  collision-safe run folders.
- Segmentation visibility and quality-control fields for ordinary diameter-only runs. Empty,
  low-confidence, low-circularity, and border-touching segmentations are now explicit without
  requiring velocity analysis; potentially partial shapes are labeled conservatively rather
  than treated as reconstructed hidden pupils.
- Zenodo DOI metadata. `CITATION.cff` records the v0.2.0 version DOI
  (`10.5281/zenodo.21897796`) and the concept DOI (`10.5281/zenodo.21897795`), so
  GitHub's "Cite this repository" button now exports a DOI. `README.md` badges the
  concept DOI and `[project.urls]` links it.

### Changed

- **Reported diameters are about 6% larger than in v0.2.0.** The packaged checkpoint's
  calibrated threshold is `0.4`; v0.2.0 used `0.7`. A lower threshold admits more boundary
  pixels, so every `estimated_pupil_diameter` and `pupil_diameter_input_pixels` value is
  systematically larger. On the `sample_data` velocity fixture the mean diameter rises 6.0%;
  the new weights evaluated at the old `0.7` threshold differ from v0.2.0 by about 1%, so
  nearly all of the shift is the threshold rather than the model. The shift is a correction,
  not an error: measured against ground-truth masks, `0.7` under-estimated diameters by about
  4%, while `0.40`-`0.45` is close to unbiased. Diameters remain uncalibrated and comparable
  only within one analysis run.

  **Do not pool or compare diameters measured before and after this release.** Re-run earlier
  recordings with this version, or pass `--pred_thresh 0.7` to reproduce v0.2.0 numbers.

- Replaced the packaged checkpoint and training log with a fine-tuned candidate. Its
  calibrated threshold is 0.4, macro IoU is 0.8749, and balanced size-bin IoU is 0.8690 on the
  maintained validation set. Its concise filename retains the training-set size, calibrated
  threshold, and macro IoU; detailed hyperparameters and secondary metrics remain in the
  matching log and JSON metadata.

  A subsequent ten-run seed study (`reports/2026-08-14-checkpoint-noise-floor.md`) found this
  candidate's margin over its predecessor is about 1.6 standard deviations of the spread
  produced by changing the random seed alone, so it is **not** a demonstrated improvement in
  the weights. It is kept rather than replaced because every fine-tuned run preserves a
  small-pupil detection on real frames that three of five higher-scoring from-scratch runs
  lose entirely. The threshold recalibration below is the part of this release with evidence
  behind it.

  The filename metric is **not** comparable with the superseded `iou=0.9158`, which was
  measured with the batch-aggregated IoU used before this release. Re-scored with the same
  per-image metric and calibrated on the same threshold grid, the superseded checkpoint
  selects `0.45` and reaches 0.8618 macro and 0.8578 balanced IoU, so the fine-tuned model
  gains roughly 0.013 macro and 0.011 balanced IoU. Both figures are validation-selected: the
  weights, the epoch, and the threshold were all chosen on one 56-image validation set whose
  recordings share recording groups with the training set. Treat them as a relative comparison
  between candidates, not as an independent generalization estimate. `validation_note` in the
  packaged JSON metadata records the same caveat.

- Inference now uses calibrated JSON metadata beside a checkpoint, then a threshold encoded
  in its filename, before falling back to `0.7`. `--pred_thresh` remains an explicit override.
- Rewrote `README.md` around the order a user actually needs: install, run, output,
  then the full CLI reference, with a plain linked contents list above them. Unit
  conventions, calibration caveats, quality-control semantics, environment setup, and
  PyTorch CPU/GPU builds moved into later sections and an FAQ, leaving the CSV column
  table as the inline reference. No documented behavior changed. `--batch_size` and
  `--mask_transparency`, which the CLI has always accepted, are now documented.

## [0.2.0] - 2026-08-12

Everything between the `v0.1.4` tag (2026-06-12) and this release, including the
pupil-center velocity feature, first ships here. Version 0.1.3 was never released.

### Added

- Opt-in pupil-center tracking and velocity analysis via `--calculate_velocity` and
  `--acquisition_fps`. Timestamps derive from the source-frame index and the actual
  acquisition rate rather than the video container's playback rate.
- Per-frame segmentation quality control with a three-state `tracking_status`
  (`valid` / `warning` / `invalid`) and a concise `quality_reason`.
- Temporal area validation that flags abrupt component-area changes as usable warnings.
- Translucent yellow-orange-red confidence-heatmap overlays via `--output_mask_dir`,
  with center markers distinguishing accepted from rejected candidates.
- A public Python API. `analyze_video(...)`, `analyze_frames(...)`, `run_analysis(...)`,
  `AnalysisConfig`, and `AnalysisResult` are importable from `mouse_pupil_analysis` and return
  the analysis table as a DataFrame instead of requiring the CSV to be read back.
  Names resolve lazily, so `import mouse_pupil_analysis` does not load PyTorch.
- `pupil_diameter_input_pixels`, reporting pupil diameter at the scale of the image that
  was supplied rather than the 148 x 148 model image. For video input that is the source
  frame; for `image_dir` input it is whatever the caller prepared. Pixels remain
  uncalibrated, so cross-recording comparison still requires matching optics or a scale
  factor this package does not infer.
- `--num_workers` on the CLI and `num_workers` in the API. The dataloader worker count
  was previously hardcoded to 4; it now defaults to at most 4, capped by CPU count.
- `show_progress`, off by default, so library callers get no stderr output. The console
  scripts opt in, leaving terminal behavior unchanged.
- Checkpoints passed with `--checkpoint` no longer have to use spatial attention.
  The architecture is read from the checkpoint's own weights, and a genuinely
  incompatible file now reports which checkpoint failed and why instead of raising a
  raw state-dict key error.
- `mouse_pupil_analysis.__version__`, resolved from installed distribution metadata so the
  package, `pyproject.toml`, and `CITATION.cff` cannot silently disagree.
- PyPI project metadata: long description, keywords, trove classifiers, and project URLs.
- A `sample_data/` fixture of real pupil images published with permission, plus
  end-to-end tests over a synthetic video and real-image regression tests over the
  fixture. Synthetic input segments plausibly regardless of which weights are loaded,
  so only the real-image tests detect a corrupted or swapped checkpoint.

### Changed

- **Breaking (packaging).** The distribution is renamed from `pupil-tracking` to
  `mouse-pupil-analysis`. The name `pupil-tracking` on PyPI belongs to an unrelated
  project by a different author. Install with `pip install mouse-pupil-analysis`.
  The import is now `mouse_pupil_analysis`, matching the permanent project identity;
  `import pupil_tracking` no longer resolves to this project. The established
  `run-pupil-analysis` and `extract-frames` commands are unchanged, so command-line
  users are unaffected. No compatibility package is shipped: the unrelated
  `pupil-tracking` distribution installs its own `pupil_tracking/__init__.py`, and
  claiming that path from here would let the two distributions overwrite and delete
  each other's files. This release predates the first PyPI publication of
  `mouse-pupil-analysis`, so no published Python API is broken.
- **Pupil diameters change by +0.1275%.** The equivalent-circle conversion factor was
  the rounded literal `1.27`; it is now derived exactly as `4 / pi` (1.273240). Reported
  diameters are therefore a factor of `sqrt(4 / pi / 1.27)` = 1.001275 larger than in
  0.1.4. The difference is far below segmentation noise, but it is not nothing: do not
  pool diameters computed across this version boundary without noting it.
- Unified the analysis outputs into one `*_pupil_analysis.csv` and one
  `*_pupil_analysis.png` per run, replacing the separate diameter and tracking artifacts.
- Extracted frames are named from the one-based source-frame number, so image names
  remain traceable to the original recording under sampled extraction.
- Split the pipeline into focused modules. Inference streams one prediction at a time
  through `pupil_predictions`, `tracking`, and `extract_frames` instead of retaining every
  float probability map. Orchestration moved to `api`, output assembly to `results` and
  `plotting`, and `dataset` split into `preprocessing` (inference) and `augmentation`
  (training). `InferenceDataset` and `SegmentationDataset` replace the polymorphic
  `PupilDataset`, which returned either `(image, name)` or `(image, mask)`.
- Library modules log through the standard `logging` module instead of printing, so
  embedding applications can silence or capture output.
- Invalid command-line argument combinations now print usage text and exit, instead of
  raising an uncaught `ValueError` traceback.
- The packaged checkpoint is now located on first use rather than at import time, so
  importing the package no longer touches the filesystem or fails when checkpoints are
  absent. Checkpoint lookup also no longer returns a path from an exited
  `importlib.resources.as_file` block.
- Frame-to-frame kinematics are computed column-wise rather than row by row, which
  matters for long recordings. Verified equivalent to the previous implementation over
  randomized inputs covering gaps, invalid frames, and varied acquisition rates.
- Source-image dimensions are read once during inference and reused, removing a
  per-frame file open from velocity mode.
- Training images and masks are paired by filename stem through one shared helper used by
  both `training/run_train.py` and `training/check_augmentation.py`, and a mismatch in
  either direction now fails loudly. Both scripts previously sorted the two directories
  independently, which silently trained against misaligned labels whenever the folders
  diverged. `run_train.py` also seeds `random`, `numpy`, and `torch`.

### Deprecated

- `mouse_pupil_analysis.dataset` and its `PupilDataset`; import from
  `mouse_pupil_analysis.preprocessing` or `mouse_pupil_analysis.augmentation` instead.
  The shim keeps the original call signature,
  but it is now a factory function rather than a class, so `isinstance` checks and
  subclassing no longer work.
- `generate_pupil_mask_prediction`; use `analyze_frames` for the full pipeline.

## [0.1.4] - 2026-06-12

### Added

- `CITATION.cff` and MIT license metadata so GitHub can display citation information.
- Collaboration and project-overview documentation.

### Changed

- The trained checkpoint ships as package data, so no separate download is required.
- Pinned Black and Ruff versions in CI.

## [0.1.2] - 2026-01-27

### Added

- Packaged the project with `pyproject.toml` and the `run-pupil-analysis` and
  `extract-frames` console scripts.
- GitHub Actions CI running Ruff, Black, and Pytest.

Changes before 0.1.2 predate this changelog; see the Git history for details.

[Unreleased]: https://github.com/yzhaoinuw/mouse-pupil-analysis/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yzhaoinuw/mouse-pupil-analysis/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/yzhaoinuw/mouse-pupil-analysis/compare/v0.1.2...v0.1.4
[0.1.2]: https://github.com/yzhaoinuw/mouse-pupil-analysis/releases/tag/v0.1.2
