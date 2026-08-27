# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

## 2026-08-26

### Compact live planning and prepare v0.3.0 (Codex, GPT-5)

- Replaced the historical `next_steps.md` narrative with the few genuine open decisions:
  Purple-sleep generalization, velocity-threshold validation, the local CV-committee
  prerequisite for frame recommendation, and two post-release API/plot decisions.
- Prepared the v0.3.0 metadata and concise release notes. The versioned DOI remains at v0.2.0
  until Zenodo creates the v0.3.0 record after the GitHub release.
- Verification: repository-wide Ruff and Black checks, full Pytest suite, wheel/sdist build,
  distribution-namespace verification, and direct inspection of the v0.3.0 wheel's packaged
  615-image checkpoint all passed. Pytest retains one existing source-import deprecation warning
  and 345 Pillow deprecation warnings; Black could not read its global Windows cache but completed
  its checks.

### Prepare the sample fixture documentation for release (Codex, GPT-5)

- Renamed the fixture's provenance ledger from `manifest.csv` to `provenance.csv` so it is clearly
  distinct from `training_data_split.json`, the session-to-fold assignment used by training.
- Reduced the fixture guide to its unique material: layout, fixture-specific commands, and image
  provenance. It now links to the root analysis guide and training guide for shared instructions,
  sends augmentation preview at `sample_data`, and removes stale maintained-pool comparison data.
- Updated the root README and project overview to describe the current session-organized fixture,
  documented the distribution-namespace verifier, and reconciled stale paths and fold-manager
  names in `next_steps.md`. The public-documentation release prerequisite is complete; the
  Purple-sleep cross-recording regression remains an open performance limitation.

### Make the training guide experimenter-first and name the fold manager (Codex, GPT-5)

- Rewrote the training guide's core path in plain language: organize sessions, create fold
  assignments, reserve validation, train, and inspect results. The all-labeled refit remains a
  deliberate post-CV route rather than the default first run.
- Renamed the interactive browser surface from split manager to fold manager:
  `review_folds.py` serves `fold_manager.html`, with tests and developer documentation updated.
  The established `prepare_splits.py` command and `training_data_split.json` record remain
  compatible paths rather than undergoing an unnecessary data migration.
- Standardized reader-facing language on `labeled` and `fold assignment`; clarified that Labelme
  saves JSON annotations and the importer creates PNG masks.
- Rephrased the guide around the actions readers should take, without introducing avoidable
  detours. Every expandable options panel now begins with “Click here” so its interactive purpose
  is clear.
- Restored the reasons behind each major action: whole-session validation makes the first training
  result more meaningful, folds balance sessions for later comparison, and the augmentation viewer
  lets users see the varied images used during training and confirm that their masks still align.
- Clarified Labelme use after frame recommendation: inspect a recommended difficult frame alongside
  its neighbors, then label the frame with a clear boundary. A closed or low-resolution candidate
  can be replaced by a nearby confidently labelable frame.
- Verification: focused fold-manager and training tests, full Pytest suite, repository-wide Ruff
  and Black checks, package wheel/sdist build, and `git diff --check`. Pytest reported one
  existing source-import deprecation warning and 345 existing Pillow deprecation warnings;
  Black could not read its global Windows cache but completed its checks.

### Tighten the public README's first-run path (Codex, GPT-5)

- Kept the main README centered on install, common commands, results, and the Python API; moved
  framing guidance into the mask-troubleshooting FAQ instead of interrupting first-run usage.
- Documented the actual timing contract: velocity mode falls back to an input video's encoded FPS,
  but frame directories require `--acquisition_fps`; users should supply the camera's real rate
  when interpreting speed. The prior `analyze_frames` example was corrected to satisfy that
  requirement.
- Replaced volatile PyTorch CUDA-index/version instructions with a link to PyTorch's maintained
  installer selector.
- Verification: full Pytest suite, repository-wide Ruff and Black checks, package wheel/sdist
  build, `git diff --check`, and an API validation call confirming the frame-directory
  missing-FPS error. Pytest reported one existing source-import deprecation warning and 345
  existing Pillow deprecation warnings; Black could not read its global Windows cache but
  completed the check.

### Promote the 615-image all-labeled model and soften confidence overlays (Codex, GPT-5)

- Promoted `checkpoints_exp/all_615_cv_e100_nat_macro_s0/all_data.pth` as the sole packaged
  checkpoint: `615pupils_thresh=0.5_iou=0.6080.pth`. It is an attention U-Net trained from
  scratch on 615 labelled pairs for 100 epochs with natural sampling and threshold 0.5.
  Maintainer visual inspection found it performs well on the unlabeled test images.
- The filename's 0.6080 is the mean macro IoU across the four grouped-CV folds that selected
  the fixed recipe, not an evaluation of the final all-labeled weights. The packaged metadata
  records that distinction and the old 166-image packaged asset is retained locally only under
  the package-excluded checkpoint archive.
- Extended `training/package_checkpoint.py` so all-labeled `all_data.*` runs require and verify
  their CV-generated recipe and sibling summary hashes before packaging; this closes the previous
  promotion gap, which accepted only validation-selected `best.*` runs.
- Reduced the analysis CLI, API, direct-run helpers, and prediction helpers' default confidence
  overlay transparency from 0.1 to 0.05. No release/tag was created; user-facing documentation
  remains intentionally in progress.
- Verification: full Pytest passed using a repository-local base temp (one existing source-import
  deprecation warning and 345 existing Pillow deprecation warnings); repository-wide Ruff, Black,
  and whitespace checks passed. A clean wheel/sdist build and distribution namespace check passed;
  both distributions contain only the new checkpoint, matching metadata, and training log, with no
  archive contents. The initial build exposed stale retired assets under ignored `build/`; those
  generated `build/` and `dist/` folders were removed before the clean verification build.

## 2026-08-25

### Bound cross-validation model selection (Codex, GPT-5)

- Set the grouped-CV selection budget to 200 epochs with 20 validation epochs of patience. The
  completed run still selects each fold's best epoch and derives the all-data schedule from their
  median; 200 is a ceiling, not a fixed refit duration.
- Completed `checkpoints_exp/cv` over 566 images, 24 sessions, and four
  grouped folds. The selected epochs were 67/54/73/62, fold macro IoUs were
  0.6842/0.4254/0.6364/0.6859, mean per-session IoU was 0.6530, and image-weighted IoU was
  0.5995. The generated recipe selects threshold 0.5 and rounds the 64-epoch fold median up to
  100 epochs, with LR milestones 50/75.
- CV recipes record their sibling `summary.json` by portable relative filename and round their
  all-data epoch budget up to the next full 100-epoch block. `checkpoints_exp/cv` is now the
  default committee for frame recommendation.
- Trained `checkpoints_exp/all_566_cv100_nat_macro_s0/all_data.pth` once from that corrected
  recipe: 566 all-labelled pairs, 100 epochs, natural sampling, threshold 0.5, and milestones
  50/75. SHA-256: `25decc4becc2f30e17413a090506167e1e7b2e8d0ca3d4f85db9d0fd808741ef`.
- Verified this change with Ruff, Black, and the full Pytest suite. Pytest reported one existing
  source-import deprecation warning and 345 existing Pillow deprecation warnings, and used a
  repository-local temporary directory because the Windows system pytest temp directory is denied
  to the sandbox account.

### Add an opt-in centre-favoured component selector (Codex, GPT-5)

- Added `--prefer_central_component` to the analysis CLI and API, defaulting to off. After normal
  thresholding it scores connected components using confidence, a saturating area term, and a soft
  image-centre preference; it applies no circularity or shape gate, so eyelid-occluded pupil
  crescents remain eligible.
- On `260812_3582_Purple_trial5_2026-08-12T15-46-08.054-1` frame 09158, the selector retained
  the real lower pupil and removed the separate upper dark-region false positive. Three local
  overlay previews are in the ignored `images_test/central_component_preview_...` directory.
- Verification: focused central-component, overlay, tracking, and end-to-end tests; Ruff, Black,
  and whitespace checks.

### Import the Purple trial-5 recording and retrain all labelled data (Codex, GPT-5)

- Imported 20 verified Labelme pairs from the `15-05-55.154-1` Purple recording into its own
  session directory. The pre-existing four-image `260812_3582_Purple_trial5` holdout was not
  changed; the refreshed manifest contains 536 pairs across 23 sessions and assigns the new
  20-pair session to CV fold 3.
- Added `training/default_all_labeled_training_config.json`, a tracked reference implementation
  of the fixed natural-sampling recipe used for the 516-pair refit: attention U-Net, seed 0,
  batch size 8, learning rate 0.001, 115 epochs, threshold 0.5, and milestones 25/51/62/71/82.
- Trained `checkpoints_exp/all_536_nat_macro_s0/all_data.pth` on all 536 pairs. Its SHA-256 is
  `fffcfcc0078be95a65d37c5b57b7df82d80a329a56cecfb17ba7f801bf8ffd2c`.

## 2026-08-24

### Restore the Labelme intake name and make session separation explicit (Codex, GPT-5)

- Restored the intake command name to `training/labelme_json2png.py`; it remains the single
  two-argument Labelme importer, including validation, polygon rasterization, atomic new-session
  creation, and split refresh. A successful validation now imports directly rather than requiring
  a separate `--apply` confirmation.
- Clarified the Labelme instructions with a concrete recommended-frame example and the essential
  rule: a shared partial label such as `Purple_trial5` does not establish that recordings are one
  session. A new recording uses a distinct `--session`; the importer refuses to merge with an
  existing session directory.
- Verification: the real 20-annotation Purple recording queue completed a dry run with no writes.

### Reduce training command options to workflow choices (Codex, GPT-5)

- Standardized every multiword training option on underscore spelling and removed per-run
  tuning knobs for epochs, batch size, seed, device, extraction, thresholds, output roots, and
  deduplication. The commands now expose only workflow choices such as the labelled pool,
  checkpoint destination, CV run directory, fold selection, and explicit fine-tuning source.
- Removed public overwrite flags. Recommendation reruns require removing their generated output
  folders first; checkpoint packaging requires removing or archiving a same-name packaged asset.
- Added bold-titled folded argument references for every training command shown in the guide.
  They list flags only, without argparse value placeholders such as `FPS`; the descriptions say
  what values each option accepts.
- Verification: all eight command help pages, the full 143-test suite, repository-wide Ruff,
  Black, and whitespace checks passed. Black emitted its known inaccessible global-cache warning.

### Make recommender committees explicit (Codex, GPT-5)

- Replaced the stale hard-coded CV checkpoint glob with required `--checkpoint_dir` input.
  The recommender now discovers the `best.pth` files from one complete CV-run directory and
  rejects incomplete directories, so it cannot silently mix unrelated experiments.
- Updated the frame-selection guide with the one-directory invocation and a folded table of
  every optional recommender argument.
- Verification: focused frame-selection tests (19 passed) plus repository-wide Ruff, Black, and
  whitespace checks passed. Black could not read its global Windows cache but completed the check.

### Make the training command surface explicit (Codex, GPT-5)

- Replaced noun- and abbreviation-based command names with action names:
  `prepare_splits.py`, `review_splits.py`, `run_cross_validation.py`,
  `import_labelme.py`, `preview_augmentation.py`, and `package_checkpoint.py`.
  No compatibility wrappers remain; these source-checkout tools are still under active development.
- Removed the completed filename-compaction migration, the standalone Labelme-mask converter,
  and the obsolete final-refit/one-shot-holdout command. Labelme validation, mask rasterization,
  compact frame naming, and atomic intake now live together in `import_labelme.py`.
- Reduced `run_train.py` from 927 to 182 lines by moving the model loop, metrics, data loading,
  and run artifacts into private `_trainer.py`. Frame scoring is likewise private in
  `_frame_scoring.py`; there are now eight runnable commands and three clearly marked support modules.
- Simplified checkpoint packaging to validation-selected `best.*` run folders only. The renamed
  `package_checkpoint.py` still writes the weights, threshold metadata, and redacted training log
  required by package/release checks, so manual checkpoint copying is not the supported path.
- Updated the training guides, project map, reports, tests, and split-manager launch text.
- Verification: all eight training-command `--help` calls passed; Ruff and Black passed on
  `training`, `tests`, and `reports/scripts`; full Pytest passed with a repository-local base temp.

### Name the training split record explicitly (Codex, GPT-5)

- Renamed the fixed grouped-split manifest to `training_data_split.json`, including the
  maintained local record and the tracked sample-data fixture. The split engine owns this
  filename, so normal training, CV, the split manager, Labelme intake, and supporting reports
  all use the same sibling file.
- Updated current training guidance and the split-manager UI to name the record plainly; the
  all-labeled recipe path remains explicitly independent of it.
- Simplified cross-validation output names and controls: `--n_folds` creates the split size,
  `--cv_folds` optionally reruns existing folds, and `--tag` is gone. A complete CV run writes
  `summary.json` plus `training_config.json`; a partial run writes only `partial_summary.json`.

## 2026-08-23

### Compact training and add the CV hand-off recipe (Codex, GPT-5)

- Replaced `run_train.py`'s split/final-refit flag maze with optional
  `--labeled_frames_dir`, `--training_config_path`, and an actual per-run
  `--checkpoint_dir`; retained only practical normal-training controls, all with underscore
  spelling. `--max_epochs` now names the epoch limit.
- Normal training reads the sibling `splits.json` and its validation session. A supplied
  `training_config.json` instead owns the model settings, ignores `splits.json`, and trains every
  valid pair under `labeled_frames/`; its output records the source-config hash for reproducibility.
- `run_cv.py` now writes that recipe using the median successful-fold epoch and calibrated
  threshold, plus a deterministic half/three-quarter learning-rate schedule. Removed the obsolete
  flat-folder, size-balancing, single-fold CLI, and final-refit training paths; retained historical
  final-refit evaluation support for existing artifacts.
- Kept promotion metadata compatible with the fixed natural-sampling policy and excluded the new
  local labeled-frame and training-config paths from packaged checkpoint metadata.
- Reframed the former final-test reservation as a CV-excluded session, including the split-manager
  label; all-labeled training is explicit that it includes those images. Rendered the real
  split-manager page to verify the updated label.
- Updated the training and intake guides, active next steps, and workflow tests. Verification:
  focused workflow, CLI-help, and split tests passed; the full pytest suite passed with a
  repository-local base temp; repository-wide Ruff passed; Black and `git diff --check` passed
  (Black emitted only its known unreadable user-cache warning).

## 2026-08-22

### Simplify the grouped-split command (Codex, GPT-5)

- Reduced `data_splits.py` to the session-folder workflow: it now accepts only
  `--labeled_frames_dir`, `--folds`, `--final_test_session`, `--validation_session`,
  `--show`, and `--reassign`. All multiword options use underscores.
- Removed legacy flat-pool, sidecar, Labelme-flag, batch-fallback, schema-migration, and
  materialized-fold support. The manifest is the sole split record and trainers read it directly.
  A labelled pair must live under `labeled_frames/<session>/images|masks`.
- Clarified the two reservations: a validation session selects a normal development run; a final
  test session is never used for training or model decisions and is evaluated only after choices
  are frozen. Updated intake, sample-data, Labelme-import, and CV guidance accordingly.
- Verification: focused split, sample-data, Labelme-import, split-manager, and training-workflow
  tests passed with a repository-local pytest base temp; the full pytest suite also passed with a
  repository-local base temp. `data_splits.py --help` and a real `sample_data/labeled_frames`
  census passed; repository-wide Ruff and Black checks passed (Black emitted its known unreadable
  user-cache warning only), and `git diff --check` passed.
- Aligned `split_manager.py` with the split engine: it now takes optional
  `--labeled_frames_dir`, uses the fixed parent `splits.json`, and exposes `--no_open` rather than
  dash-separated flags. The UI calls `data_splits.build_manifest` only on refresh and
  `data_splits.apply_session_assignments` when saving; it does not own grouping or assignment
  rules. Rendered the live sample-data manager to verify the renamed validation/final-test labels
  and removal of the now-uninformative provenance source label. Focused split-manager tests, Ruff,
  Black, and `split_manager.py --help` passed.

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
- Replaced the manager's number tiles with a stacked tiny/medium/large pupil chart for the folds.
  Clicking a session reveals its matching chart; both redraw as sessions are dragged. The UI and
  guide now use the shorter term "fold". The local server stays the supported launch/save path:
  a directly opened browser file cannot reliably obtain permission to read and overwrite the
  active manifest.
- Put the selected-session chart beside the fold chart, fixed its image-count field, and made a
  second click deselect it. Both charts now overlay a right-axis line for median background
  brightness, the existing 0–255 feature measured outside the labelled pupil and stored per image
  in `splits.json` when `data_splits.py` refreshes the manifest.
- Follow-up verification: the live 22-session page rendered 12 fold size bars and four brightness
  points; selecting a real 62-image session rendered three size bars and one brightness point
  without `NaN`, and clicking it again hid the session panel. Focused manager tests, Ruff, and
  Black pass.
- Added browser-tab heartbeats so the loopback server stops after the last manager tab closes,
  with a brief grace period for reloads. Brightness overlays now show the full within-group
  interquartile range (Q1–Q3) plus the median, calculated from the per-image values already
  stored in the manifest rather than rereading image files.
- Verification: after closing a live manager tab, the process printed its automatic shutdown
  message and exited. The same live page rendered one brightness range and median dot for each
  of four folds, then the selected session's own range and dot.
- Added a separate validation holdout that CV excludes. A normal manifest training run uses it for
  early stopping, scheduler/checkpoint selection, and threshold calibration. When empty, the
  trainer uses every development session with fixed mid/three-quarter learning-rate milestones and
  a fixed threshold, recording `all_data.*` rather than claiming validation-selected `best.*`.
- Verification: `python -m pytest -q --basetemp .pytest_tmp_split_manager tests/test_cli_help.py
  tests/test_data_splits.py tests/test_training_workflow.py tests/test_holdout_evaluation.py
  tests/test_promotion.py tests/test_split_manager.py` (passed); the full pytest suite also
  passed; Ruff and Black on changed
  Python files; `python training/data_splits.py --help`; `python training/run_train.py --help`;
  local browser render of the live 22-session manager; `git diff --check`. A direct-open guard
  now explains that `split_manager.html` needs the local `split_manager.py` service, while a
  fresh `--no-open` server returned HTTP 200 from `/api/state`; focused manager tests, Ruff, and
  Black pass after the guard.

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
