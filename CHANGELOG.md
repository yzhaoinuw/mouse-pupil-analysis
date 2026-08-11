# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A public Python API. `analyze_video(...)`, `analyze_frames(...)`, `run_analysis(...)`,
  `AnalysisConfig`, and `AnalysisResult` are importable from `pupil_tracking` and return
  the analysis table as a DataFrame instead of requiring the CSV to be read back.
  Names resolve lazily, so `import pupil_tracking` no longer loads PyTorch.
- `--num_workers` on the CLI and `num_workers` in the API. The dataloader worker count
  was previously hardcoded to 4; it now defaults to at most 4, capped by CPU count.
- `pupil_tracking.__version__`, resolved from installed distribution metadata so the
  package, `pyproject.toml`, and `CITATION.cff` cannot silently disagree.
- PyPI project metadata: long description, keywords, trove classifiers, and project URLs.
- End-to-end tests that run the packaged checkpoint over a synthetic video, covering
  frame extraction, inference, velocity mode, and overlay generation.
- Real-image regression tests over the committed `sample_data/` fixture. Synthetic input
  segments plausibly regardless of which weights are loaded, so these catch a corrupted
  or swapped checkpoint and preprocessing regressions that the synthetic tests cannot.
- `pupil_diameter_video_pixels`, a new output column reporting pupil diameter in
  original-video pixels. The existing `estimated_pupil_diameter` is measured in the
  148 x 148 model image and is therefore not comparable between recordings with
  different resolution or cropping; the new column is. The old column is unchanged,
  so existing analyses keep working.
- Checkpoints passed with `--checkpoint` no longer have to use spatial attention.
  The architecture is read from the checkpoint's own weights, and a genuinely
  incompatible file now reports which checkpoint failed and why instead of raising a
  raw state-dict key error.

### Changed

- **Breaking (packaging).** The distribution is renamed from `pupil-tracking` to
  `mouse-pupil-analysis`. The name `pupil-tracking` on PyPI belongs to an unrelated
  project by a different author. Install with `pip install mouse-pupil-analysis`.
  The import name is unchanged, so existing code keeps using `import pupil_tracking`.
- Library modules log through the standard `logging` module instead of printing, so
  embedding applications can silence or capture output. Console scripts configure a
  plain handler, leaving terminal output unchanged.
- Invalid command-line argument combinations now print usage text and exit, instead of
  raising an uncaught `ValueError` traceback.
- `pupil_tracking.dataset` split into `pupil_tracking.preprocessing` (inference) and
  `pupil_tracking.augmentation` (training). The polymorphic `PupilDataset`, which
  returned either `(image, name)` or `(image, mask)`, is replaced by `InferenceDataset`
  and `SegmentationDataset`.
- Output table assembly and plotting moved into `pupil_tracking.results` and
  `pupil_tracking.plotting`. Plot functions return figures so they can be reused.
- The packaged checkpoint is now located on first use rather than at import time, so
  importing the package no longer touches the filesystem or fails when checkpoints are
  absent. Checkpoint lookup also no longer returns a path from an exited
  `importlib.resources.as_file` block.
- **Pupil diameters change by +0.1275%.** The equivalent-circle conversion factor was
  the rounded literal `1.27`; it is now derived exactly as `4 / pi` (1.273240). Reported
  diameters are therefore a factor of `sqrt(4 / pi / 1.27)` = 1.001275 larger than in
  0.1.4. The difference is far below segmentation noise, but it is not nothing: do not
  pool diameters computed across this version boundary without noting it.
- Frame-to-frame kinematics are computed column-wise rather than row by row, which
  matters for long recordings. Verified equivalent to the previous implementation over
  randomized inputs covering gaps, invalid frames, and varied acquisition rates.
- Source-image dimensions are read once during inference and reused, removing a
  per-frame file open from velocity mode.
- `training/run_train.py` pairs each image with the mask sharing its filename stem and
  fails loudly on a mismatch. Previously it sorted the two directories independently,
  which silently trained against misaligned labels whenever the folders diverged. The
  script also seeds `random`, `numpy`, and `torch`.

### Deprecated

- `pupil_tracking.dataset` and its `PupilDataset`; import from `pupil_tracking.preprocessing`
  or `pupil_tracking.augmentation` instead.
- `generate_pupil_mask_prediction`; use `analyze_frames` for the full pipeline.

## [0.1.4] - 2026-06-12

### Added

- Opt-in pupil-center tracking and velocity analysis via `--calculate_velocity` and
  `--acquisition_fps`. Timestamps derive from the source-frame index and the actual
  acquisition rate rather than the video container's playback rate.
- Per-frame segmentation quality control with a three-state `tracking_status`
  (`valid` / `warning` / `invalid`) and a concise `quality_reason`.
- Temporal area validation that flags abrupt component-area changes as usable warnings.
- Translucent yellow-orange-red confidence-heatmap overlays via `--output_mask_dir`,
  with center markers distinguishing accepted from rejected candidates.
- `CITATION.cff` and MIT license metadata so GitHub can display citation information.

### Changed

- Unified the analysis outputs into one `*_pupil_analysis.csv` and one
  `*_pupil_analysis.png` per run, replacing the separate diameter and tracking artifacts.
- Extracted frames are named from the one-based source-frame number, so image names
  remain traceable to the original recording under sampled extraction.
- Split the pipeline into focused modules: `pupil_predictions`, `tracking`, and
  `extract_frames`. Inference streams one prediction at a time instead of retaining
  every float probability map.
- The trained checkpoint ships as package data, so no separate download is required.

## [0.1.2] - 2026-01-27

### Added

- Packaged the project with `pyproject.toml` and the `run-pupil-analysis` and
  `extract-frames` console scripts.
- GitHub Actions CI running Ruff, Black, and Pytest.

Changes before 0.1.2 predate this changelog; see the Git history for details.

[Unreleased]: https://github.com/yzhaoinuw/pupil_tracking/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/yzhaoinuw/pupil_tracking/compare/v0.1.2...v0.1.4
[0.1.2]: https://github.com/yzhaoinuw/pupil_tracking/releases/tag/v0.1.2
