# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking (packaging).** The distribution is renamed from `pupil-tracking` to
  `mouse-pupil-analysis`. The name `pupil-tracking` on PyPI belongs to an unrelated
  project by a different author. Install with `pip install mouse-pupil-analysis`.
  The import name is unchanged, so existing code keeps using `import pupil_tracking`.

### Added

- `pupil_tracking.__version__`, resolved from installed distribution metadata so the
  package, `pyproject.toml`, and `CITATION.cff` cannot silently disagree.
- PyPI project metadata: long description, keywords, trove classifiers, and project URLs.

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
