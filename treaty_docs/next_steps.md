# Next Steps

Use this checklist alongside `work_log.md`. Keep it concrete: only add work here when it is an actual follow-up, blocked thread, or decision that future agents should see before changing code.

## Currently Hot

- [Recording-grouped data splits](#recording-grouped-data-splits) - built; run the first real cross-validation sweep to establish the generalization baseline.
- [Model-selection metric fragility](#model-selection-metric-fragility) - `balanced_iou` rests a third of its weight on two validation images and the calibrated threshold varies 0.30-0.65 across seeds; fix both before the next promotion.
- [Segmentation fine-tuning and visibility](#segmentation-fine-tuning-and-visibility) - the promoted candidate's margin is inside seed noise and the packaged checkpoint is retained; now unblocked by the grouped split.
- [Pupil-center velocity](#pupil-center-velocity) - shipped; validate the provisional quality thresholds on additional recordings before treating them as a universal rejection policy.
- [Treaty v0.9.0 docs layout](#treaty-v090-docs-layout) - migrated and verified on `chore/treaty`; review and integrate the branch.

When a new thread starts, add a short bullet here with a link to its section below and the single next action. When a thread closes, drop its bullet and compress its section to a status line plus whatever genuinely remains.

## Recording-Grouped Data Splits

Status: built; no cross-validation results measured yet

The old fixed split leaked almost completely: 54 of 56 validation images came from a recording
that also supplied training images, with **no validation-only animals**. Every IoU this project
reported before 2026-08-14 therefore measures held-out frames from seen recordings.

`training/data_splits.py` now groups the pool into **sessions** (one animal, one date, one
condition) and packs whole sessions into **stratified** folds, recorded in `splits.json`.
`training/run_cv.py` runs every fold. `training/data_collection.md` documents the labelling
policy and how a session gets recorded.

**Settled: the grouping unit is the session.** Not the recording file — same-day siblings like
`HQL086_whiskerb250923_{002,005,008}` are one sitting with the camera untouched. Not the animal
— every animal appears in one cohort and mostly one condition, so animal-grouping is nearly
redundant with session-grouping and only costs training data. The 222 images are **16 sessions**,
not 25 recordings and not 10 animals. Do not re-litigate.

**Settled: the session is recorded at intake, not inferred.** Measured, not assumed — crop
geometry splits 6 of 16 sessions, agglomerative clustering on preprocessed frames tears 3 at
k=5 and 6 at k=10, and file mtime is destroyed by copying. The images are tight eye crops with
no rig context left to fingerprint. Filenames are no longer parsed at all; provenance comes from
an intake subfolder, a labelme `session` flag, or `provenance.csv`, and anything unresolved
becomes one safe over-merged group. Do not re-litigate; add a provenance *source* to
`training/provenance.py` if a new intake route appears.

Two structural facts that constrain every future comparison:

- The `5003` dim-light session is 62 images, **28% of the pool**, and is indivisible. Fold 0
  carries it plus two small sessions at 65 images; the other four hold 34-47. A single fixed
  holdout costs 15-27% of the pool, which is why none is set by default.
- `HQL080_sleep250625` holds **10 of the 14 tiny masks in the entire dataset**. Stratification
  spread the rest so 4 of 5 folds now contain some tiny mask, up from 2 of 5, and cut the
  cross-fold median-diameter spread from 3.03x to 1.78x. It cannot fix the last fold: only four
  sessions contain a tiny mask at all. Report mean per-session IoU; `run_cv.py` prints which
  bins each fold actually scored.

Next action:

- Run the first real sweep (`python training/run_cv.py --data-root . --split-manifest splits.json
  --out checkpoints_exp/cv --epochs 400`) to establish the generalization baseline. Every number
  this project has published is an interpolation number; this produces the first one that is not.

Then:

- Repeat the sweep at two or three seeds. Cross-validation narrows sampling noise, not seed
  noise, and the measured seed floor is +/-0.0069.
- Compare size-balanced against natural sampling on the grouped split. The fold holding
  `HQL080_sleep250625` out trains on only 2 tiny masks, so balanced sampling oversamples two
  images to a third of the training mass there - a hazard the old split hid.

## Model-Selection Metric Fragility

Status: measured; no change made yet. See `reports/2026-08-14-checkpoint-noise-floor.md`.

- **Two images decide a third of the metric, and the tiny bin is really one session.**
  `balanced_iou` is the mean of the tiny, medium, and large bins; the old validation set held
  exactly 2 tiny masks, and across the whole pool `HQL080_sleep250625` holds 10 of the 14. So
  the tiny bin measures one recording setting, not a size regime, and under grouped folds it is
  absent from three folds of five. Small pupils are not a training-data bottleneck: in absolute
  pixel terms the model is about twice as accurate on them as on medium pupils, and low
  tiny-bin IoU is a mechanical property of IoU on small objects. What would actually help is
  small-pupil labels from a *different* session; more from the same one would not.
- **The calibrated threshold is not a model property.** Ten runs differing only by seed selected
  thresholds spanning 0.30-0.65, wider than the 0.7 to 0.4 change that moved every user's
  reported diameters by about 6%. Consider shipping a calibration procedure users run on their
  own recordings instead of a constant baked into the filename.
- **Report signed diameter bias next to IoU.** An IoU of 0.895 hid a systematic -4% diameter
  offset in the 0.7-threshold era. `reports/scripts/validation_diagnostics.py` prints both;
  folding the diameter column into the trainer's per-epoch log would surface the next such
  offset immediately.

Parked:

- Calibrating the threshold on diameter error instead of IoU was tested and does **not** improve
  precision (bootstrap sd 0.085 either way) - the two criteria nearly coincide for convex masks.
  Report diameter bias as a diagnostic; do not switch the selection criterion.

## Segmentation Fine-Tuning And Visibility

Status: promoted candidate's margin shown to be inside seed noise; packaged checkpoint retained;
unblocked now that the grouped split exists

The maintained trainer supports lower-rate weight fine-tuning, equal-mass sampling across
tiny/medium/large mask bins, per-image macro validation, equal-weighted size-bin early stopping,
threshold calibration, and unconditional best-checkpoint/log retention. Diameter-only inference
exports conservative visibility and segmentation-QC fields.

A ten-run seed study on 2026-08-14 (`reports/2026-08-14-checkpoint-noise-floor.md`) established:

- The +0.0112 balanced-IoU margin that justified promoting the packaged checkpoint is **1.6
  standard deviations of the spread produced by changing the random seed alone**. Five
  from-scratch runs average macro IoU 0.8749, matching it exactly; its balanced IoU sits at
  their 20th percentile. Fine-tune runs peaked at epochs 1-26, so the best checkpoint was
  essentially the starting weights.
- Retraining from scratch is **not** a safe substitute. Three of five from-scratch runs lose a
  real small pupil in `sample_data/raw_frames/recording_250616` that every fine-tuned run and
  the packaged checkpoint detect, and those three are the runs with the *highest* validation
  IoU. A higher-scoring scratch candidate was promoted and then reverted for this reason.
- The fine-tuned lineage preserves something validation cannot see. Keep the packaged
  checkpoint and treat `reports/scripts/hard_frame_check.py` as a promotion gate.

Next action:

- Re-run the fine-tune-versus-scratch comparison on the grouped split, which is now available
  ([Recording-grouped data splits](#recording-grouped-data-splits)). Note that fine-tuning the
  packaged weights inside cross-validation leaks: those weights saw all 222 images, so every
  fold's validation data is already in them. Only a scratch arm is measurable this way; judging
  the packaged lineage needs the hard-frame gate and genuinely new recordings.

Then:

- Obtain raw versions and masks for the troubleshooting frames and compare the packaged
  checkpoint against the size-balanced candidate on an independent, recording-grouped test set.
- Require of any future promotion: a seed spread behind the claimed margin, a pass on the
  hard-frame gate, and a measured diameter shift for the CHANGELOG.

Parked:

- Revisit BCE plus Dice, focal, or Tversky loss only after the sampling, metric, calibration, and
  QC changes have been evaluated. A previous attempt showed no clear improvement, but a
  controlled ablation may still be worthwhile.
- Consider a higher-resolution input or two-stage eye-ROI model only if clearly visible tiny
  pupils remain unresolved. This changes the model contract and requires full retraining, so keep
  it out of the current checkpoint-compatible work.

## Pupil-Center Velocity

Status: shipped in 0.2.0 - velocity, confidence-heatmap overlays, 1-based source-frame names, and
unified outputs are implemented and documented

The implemented method and its quality-control semantics live in
[`../project_overview.md`](../project_overview.md#segmentation-to-velocity-method), which is the
authority; the user-facing surface is in `README.md`.

Remaining work:

- Validate the quality thresholds on additional recordings before treating them as a universal
  automated rejection policy. They are currently provisional constants in `tracking.py`.
- Consider optional trajectory smoothing only after quantifying center jitter and confirming that
  filtering preserves rapid REM eye movements. If added, export raw and smoothed values separately.

## Treaty v0.9.0 Docs Layout

Status: migrated and verified on `chore/treaty`

Remaining work:

- Review and integrate `chore/treaty` when the reorganized documentation layout is accepted.

## Packaging And Distribution

Status: complete; version 0.2.0 is published on PyPI and GitHub as `mouse-pupil-analysis`

**Settled: this project ships no `pupil_tracking` module, and no compatibility shim.** The
unrelated `pupil-tracking` distribution on PyPI installs its own `pupil_tracking/__init__.py`, so
claiming that path here would give two distributions one import namespace. Measured, not assumed:
installing both in either order silently overwrites one `__init__.py` with the other's. CI and the
release workflow both fail if `pupil_tracking/` reappears in a built artifact. Do not re-litigate.

Remaining work:

- Tag the next version and publish its GitHub/PyPI release only when the maintainer resumes
  release work. Do not tag or release as part of routine `dev`/`main` integration.

## Runtime Modularization

Status: complete on `dev`; `api.py` owns orchestration, with focused `results`/`plotting`/
`preprocessing`/`augmentation` modules and a deprecating `dataset.py` shim

Remaining work:

- Remove the `dataset.py` shim and the deprecated `generate_pupil_mask_prediction` after one release.
- Decide whether the unified plot should show `pupil_diameter_input_pixels` instead of the
  model-pixel column, now that both are exported. This changes the README demo, so it is
  deliberately deferred.

## DOI Archival

Status: complete for 0.2.0; concept DOI `10.5281/zenodo.21897795`, version DOI
`10.5281/zenodo.21897796`

Citation generators emit the **first** doi-type entry under `identifiers` and only fall back to the
top-level `doi` when that list is absent. Verified with `cffconvert`: with the concept DOI listed
first, exported BibTeX cited the moving concept DOI rather than the archived code. The per-version
entry must therefore stay first.

Remaining work:

- Each release: update the version DOI in `CITATION.cff` per `RELEASING.md` step 8. The concept
  DOI, badge, and `[project.urls]` entry never change.

## Sample Data And Fixtures

Status: complete; `sample_data/` holds eight paired training crops, four paired validation crops,
six uncropped frames at two resolutions, and 31 consecutive velocity frames at 97 Hz, with a
provenance manifest

`tests/test_real_images.py` runs the packaged checkpoint over the fixture. It is the only test that
can detect a corrupted or swapped checkpoint, because synthetic input segments plausibly regardless
of the weights. Two source resolutions are what make the input-pixel diameter conversion verifiable
against real geometry.

Remaining work:

- Keep the fixture compact. Expand it only for a specific uncovered behavior; it is an exploration
  and smoke-test resource, not a benchmark or a useful training dataset.

## Training Workflow Documentation

Status: complete; `training/README.md` covers data layout, Labelme conversion, augmentation review,
fresh training, fine-tuning, threshold calibration, and promotion via `promote_checkpoint.py`

Remaining work:

- Add structured optimizer/scheduler checkpointing only if exact interrupted-run resume becomes
  necessary. Fine-tuning deliberately restores weights only, starting fresh optimizer, scheduler,
  early-stopping, and logging state.

## Background / Paused

### Local Artifact Cleanup

Status: paused

The working tree commonly contains generated image folders, prediction outputs, build outputs, cache
folders, local sketch scripts, and experimental checkpoints. `.gitignore` covers the expected
generated surfaces, but the local workspace may still be visually noisy.

Resume only when the user asks for repository cleanup or release preparation.

Remaining work:

- Inspect tracked vs. ignored files before deleting anything.
- Keep `mouse_pupil_analysis/checkpoints/` package data intact.
