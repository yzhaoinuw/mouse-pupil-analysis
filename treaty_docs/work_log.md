# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

## 2026-08-19

### Clarify sequence-aware visibility labeling (Codex, GPT-5)

- Clarified that nearby frames may establish whether the pupil has disappeared from the target
  frame. Confirmed no-visible-pupil frames receive an all-black target even when the isolated dark
  aperture resembles a fully dilated pupil; unresolved transition frames remain `uncertain`, and
  genuinely large pupils are outlined only when their boundary is visible.
- Recorded the current HQL103 intake in the active planning docs and tracked split manifest: 60
  development pairs (51 visible pupils and 9 explicit negatives) plus 5 uncertain frames outside
  segmentation training. The pool is now 376 pairs across 20 sessions, with 372 development pairs
  and the original 4-image outer holdout unchanged.
- The manifest now uses the normalized `HQL097_sleep260321_010` session key rather than the
  filename-derived `_eye` suffix. Its refreshed assignment is zero-based fold 2, while HQL103 is
  in fold 3; neither session enters the outer holdout.
- Verification: the read-only split census accounts for all 376 pairs and preserves the 4-image
  holdout; Ruff and Black pass repository-wide; `git diff --check` is clean; and the full test
  suite passes. The installed treaty CLI is v0.5.0 while the repository is pinned to treaty v0.9.0,
  so it does not understand the relocated `treaty_docs/` layout and reports false missing-path
  errors; live/archive date counts were checked directly against the v0.9.0 convention.

## 2026-08-18

### Separate generated recommendations from labeling queues (Codex, GPT-5)

- Moved the four existing recommendation sessions, including the newly generated HQL095 batch,
  from `frame_recommendations/` to `frames_to_label/`. Standardized every session to
  `extracted_frames/` plus `recommended/`; all 8,216 files were preserved.
- Made `training/recommend_frames.py` write video outputs by default to
  `frames_to_label/<session>/{extracted_frames,recommended}`. Replaced the two granular
  output overrides with one `--output_dir` root override; frame-directory input remains in place.
- Verification: focused output-layout tests pass; the CLI help exposes `--output_dir`; Ruff and
  Black pass repository-wide; and the full test suite passes with the documented in-process PATH
  injection for the installed console-script test.

### Consolidate local video storage (Codex, GPT-5)

- Standardized `videos/` as the only root for raw recordings. Moved the HQL090, HQL097, and
  HQL103 AVIs out of the short-lived `movies/` folder and kept the newly added HQL095 and HQL096
  recordings there alongside the older `eye.avi`; `videos/` now contains six AVIs and no
  generated analysis subdirectories.
- Removed the six obsolete local review/demo trees that had left `videos/` looking unchanged:
  10,151 generated files totaling about 0.44 GiB. The raw `eye.avi` was preserved, and the exact
  90-frame README GIF source was retained under the gitignored `media/readme_demo/` workspace.
- Removed the now-empty root `movies/` and `gif_videos/` directories, and aligned the GIF defaults,
  training workflow, project map, ignore rules, and focused tests with the consolidated layout.
- Verification: the focused GIF tests pass, Ruff and Black pass on the touched Python files,
  `git diff --check` is clean, and a no-argument rebuild from `media/readme_demo/` remains
  byte-identical to the tracked README GIF.

### Integrate HQL097 pupil, closed-eye, and uncertain annotations (Codex, GPT-5)

- Audited all 67 Labelme annotations beside their 307 x 198 source frames. The batch contains
  exactly one supported label per frame: 47 `pupil`, 14 `no_visible_pupil`, and 6 `uncertain`.
  Contact-sheet review confirmed the pupil polygons follow the visible boundary, the negatives
  are fully closed or occluded, and the uncertain squints should not receive segmentation loss.
- Made `labelme_json2png.py` validate the three-label contract. Pupil polygons are rasterized
  directly, explicit negatives become exact image-sized all-black masks, uncertain annotations
  produce no mask, and unknown or contradictory labels fail before writing output. Direct
  rasterization removes the unavailable `labelme_export_json` console-command dependency.
- Added `training/import_labelme_batch.py` as the reproducible intake boundary that was missing
  from the documentation. It previews by default, validates the whole source batch before any
  write, refuses an existing session, creates compact image/mask pairs without redundant JSON
  copies, archives uncertain image/JSON pairs outside training, and refreshes the manifest and
  materialized folds whenever `--apply` is passed. A real-batch smoke reproduced all 61 HQL097
  image and mask hashes exactly.
- Added 61 compactly named HQL097 image/mask pairs to development: 47 nonempty pupil masks and
  14 empty negatives. Preserved the 6 uncertain image/JSON pairs under the session's
  `uncertain/` folder, outside `images/`, and removed the 61 generated-mask JSON copies after
  verification; the original 67 source annotations remain in `frames_to_label/`.
- Refreshed the frozen grouped manifest and materialized folds. The pool now has 316 pairs across
  19 sessions: 312 development pairs and the unchanged 4-image trial5 holdout. HQL097 entered
  fold 4; all 255 earlier assignments were carried over unchanged. Fold sizes are 76/58/73/105,
  so the next action is a matched 316-image baseline before changing any training configuration.
- Verification: the focused converter/importer tests pass; Ruff and Black pass repository-wide;
  `git diff --check` is clean; and the full test suite passes when the Conda Scripts path is
  injected inside the pytest process. The initial plain-shell pytest attempt hit only the known
  desktop PATH isolation for `run-pupil-analysis`; rerunning through the documented environment
  workaround passed. Wheel and source-distribution builds also pass: the wheel retains the
  packaged checkpoint/metadata and excludes training sources, while the source distribution
  contains the importer and `sample_data/splits.json`.
