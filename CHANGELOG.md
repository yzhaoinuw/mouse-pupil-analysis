# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Zenodo DOI metadata. `CITATION.cff` records the v0.2.0 version DOI
  (`10.5281/zenodo.21897796`) and the concept DOI (`10.5281/zenodo.21897795`), so
  GitHub's "Cite this repository" button now exports a DOI. `README.md` badges the
  concept DOI and `[project.urls]` links it.

### Changed

- Reorganized `README.md` so installation and a first run come before optional
  features, added a linked contents table, and moved environment setup, PyTorch
  CPU/GPU builds, and packaging-name background into a later section. No
  documented behavior changed. `--batch_size` and `--mask_transparency`, which the
  CLI has always accepted, are now documented.

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
