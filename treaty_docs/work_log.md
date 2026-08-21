# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

## 2026-08-21

### Compact the training workflow and build a validation-holdout split manager (Codex, GPT-5)

- Fast-forwarded local `dev` from `38398e3` to the completed recording-splits work at `e0d6bf6`,
  then created `training-compaction` from it. `origin/dev` intentionally remains unchanged until
  a push is explicitly requested.
- Reframed `training/README.md` around labelled sessions, `data_splits.py`, optional split review,
  and `run_train.py --split-manifest splits.json` without a manually selected CV fold. Fine-tuning,
  Labelme, frame recommendation, augmentation inspection, CV, outer-test evaluation, promotion,
  and the developer fixture are optional; detailed selection-result prose was removed.
- Added `training/split_manager.py`, a loopback-only drag-and-drop UI that renders automatic
  session grouping plus image count, tiny/medium/large pupil count, median diameter, and median
  brightness for every session/fold. It validates and writes whole-session fold or validation-
  holdout assignments through `data_splits.py`; it leaves the outer test holdout read-only.
- Kept the browser surface as the tracked `training/split_manager.html` asset rather than a raw
  string hidden in Python; `split_manager.py` now serves that asset and remains solely responsible
  for the local manifest API and validated write path.
- Added a separate validation holdout that CV excludes. A normal manifest training run uses it for
  early stopping, scheduler/checkpoint selection, and threshold calibration. When empty, the
  trainer uses every development session with fixed mid/three-quarter learning-rate milestones and
  a fixed threshold, recording `all_data.*` rather than claiming validation-selected `best.*`.
- Verification: `python -m pytest -q --basetemp .pytest_tmp_split_manager tests/test_cli_help.py
  tests/test_data_splits.py tests/test_training_workflow.py tests/test_holdout_evaluation.py
  tests/test_promotion.py tests/test_split_manager.py` (passed); the full pytest suite also
  passed; Ruff and Black on changed
  Python files; `python training/data_splits.py --help`; `python training/run_train.py --help`;
  local browser render of the live 22-session manager; `git diff --check`.

## 2026-08-20

### Restore CI imports for source-only training utilities (Codex, GPT-5)

- Added pytest's repository-root `pythonpath` setting. The editable distribution correctly
  installs only `mouse_pupil_analysis`, while tests also need to import the repository-only
  `training/` scripts; the setting keeps that source boundary intact and makes console-script
  pytest launches portable across CI runners.
- Updated the console-script smoke test to invoke the script installed beside the active Python
  interpreter instead of assuming its script directory is on `PATH`.
- Verification: reproduced the missing `training` import from outside the checkout, then ran the
  two previously failing test modules successfully through console-script pytest from that context.

### Make the analysis entry point directly executable (Codex, GPT-5)

- Added the same no-argument direct-run pattern used by `training/run_train.py` to
  `mouse_pupil_analysis/run_pupil_analysis.py`. Direct runs use its editable configuration block;
  supplying terminal arguments still uses the existing parser and console-script workflow.
- Removed editor-specific wording from scripts, comments, public documentation, changelog, and
  archived work logs. Direct-run configuration remains available without presenting an editor as
  part of the project workflow.
- Verification: repository text scan found no editor-specific mentions; focused training workflow
  tests passed (11 tests), Ruff and Black passed on the changed Python modules, and
  `python training/run_train.py --help` passed.

## 2026-08-19

### Train and exercise the 512-image all-in model (Codex, GPT-5)

- Froze the development choices before final refit: attention U-Net, BCE loss, natural sampling,
  seed 0, batch size 8, initial LR `1e-3`, threshold 0.5, 115 epochs, and deterministic LR
  reductions at epochs 25/51/62/71/82. Trained `final_516_nat_macro_s0` from scratch on all 512
  non-holdout pairs; the four `260812_3582_Purple_trial5` labels were not loaded.
- The final training loss fell from 0.4311 to 0.0200. Saved `final.pth` plus frozen metadata and
  manifest/checkpoint hashes under `checkpoints_exp/final_516_nat_macro_s0`; the outer holdout
  remains unconsumed.
- Ran the final checkpoint at threshold 0.5 on 200 newly extracted, unlabeled frames spread across
  `HQL088_sleep250929_009_eye.avi`. Inference took about four seconds at batch size 32. Visual
  review confirmed the model followed the boundary of a genuinely large pupil rather than filling
  the eye aperture and tracked its later constriction. QC accepted 196/200 frames; the four
  rejected overlays were a nearly closed eye or irregular fragmented masks, matching their
  low-confidence/low-circularity reasons.

### Integrate HQL095/HQL096 and train the 516-image CV committee (Codex, GPT-5)

- Audited all 145 new Labelme records against their source frames. HQL095 contributes 38 pupil
  masks. HQL096 contributes 84 pupil masks, 18 sequence-confirmed no-visible-pupil masks, and 5
  uncertain transition frames outside segmentation training; one accidentally empty transition
  annotation was conservatively made explicit as `uncertain` rather than assigned an invented
  target.
- Imported 140 trainable pairs and refreshed the frozen grouped split. The pool now contains 516
  pairs across 22 sessions: 512 development pairs and the unchanged 4-image outer holdout. HQL095
  entered zero-based fold 0 and HQL096 fold 1; all 376 previous assignments stayed unchanged.
- Trained a matched seed-0, natural-sampling, macro-IoU four-fold committee under
  `checkpoints_exp/cv516_nat_macro_20260819`. Fold macro IoUs are 0.6985/0.4832/0.5313/0.6845;
  mean per-session IoU is 0.6701 and image-weighted IoU is 0.5846. On the 17 sessions shared with
  the previous 255-image run, mean session IoU improved from 0.6468 to 0.6923 (+0.0455), with 11
  of 17 sessions improving. New-session out-of-fold IoUs are 0.6207 (HQL095), 0.5070 (HQL096),
  0.4883 (HQL097), and 0.6866 (HQL103).
- The result is not promotion-ready without follow-up. The prior aperture-confusion sessions
  improved strongly, but `250616_5120_Purple_sleep_trial_1` regressed from 0.7304 to 0.2585 and
  predicts only 0.218x the labelled area. When HQL096 is entirely held out, its fold model clears
  only 2 of 18 explicit empty masks, confirming that absence does not transfer reliably without
  examples from this appearance regime.
- Verification: all 70 focused split/import/converter/pairing/training tests and the full 168-test
  suite pass using fresh repo-local pytest temp directories; the full run needs the documented
  in-process Conda Scripts `PATH` injection. Ruff and Black pass repository-wide, and
  `git diff --check` is clean. The four-fold CUDA run completed normally on the RTX 3070 and wrote
  its summary after 142 minutes; the outer holdout was excluded from every fold.

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
