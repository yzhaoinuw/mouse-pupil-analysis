# Next Steps

Use this checklist alongside `work_log.md`. Keep it concrete: only add work here when it is an actual follow-up, blocked thread, or decision that future agents should see before changing code.

## Currently Hot

- [Recording-grouped data splits](#recording-grouped-data-splits) - the pool is now 316 images; establish the matched new-pool baseline before comparing configurations.
- [Model-selection metric fragility](#model-selection-metric-fragility) - fixed; macro IoU and validation loss are now the defaults.
- [Training sampling default](#training-sampling-default) - fixed; natural sampling is now the default after beating size-balanced by 0.0354.
- [Improving cross-recording generalization](#improving-cross-recording-generalization) - diagnosed: the model segments the eye aperture, not the pupil, at p=0.99. Augmentation is the wrong tool and has been withdrawn.
- [New labels and experiment sequencing](#new-labels-and-experiment-sequencing) - HQL097 adds 47 pupils and 14 true negatives to development; the 4-frame holdout remains untouched.
- [Frame recommendation](#frame-recommendation) - HQL097 is integrated; HQL103 is the highest-value remaining new session to label.
- [Sampling rate and pupil dynamics](#sampling-rate-and-pupil-dynamics) - size needs 10-30 Hz, position needs the 97 Hz rig; measure a real PLR constriction velocity to set the bound.
- [Segmentation fine-tuning and visibility](#segmentation-fine-tuning-and-visibility) - the promoted candidate's margin is inside seed noise and the packaged checkpoint is retained; now unblocked by the grouped split.
- [Pupil-center velocity](#pupil-center-velocity) - shipped; validate the provisional quality thresholds on additional recordings before treating them as a universal rejection policy.
- [Treaty v0.9.0 docs layout](#treaty-v090-docs-layout) - migrated and verified on `chore/treaty`; review and integrate the branch.

When a new thread starts, add a short bullet here with a link to its section below and the single next action. When a thread closes, drop its bullet and compress its section to a status line plus whatever genuinely remains.

## Recording-Grouped Data Splits

Status: built and measured (2026-08-16); baseline established, see
`reports/2026-08-16-selection-metric-repair.md`

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

Two structural facts that constrained the original 222-image comparison:

- The `5003` dim-light session is 62 images, **28% of the pool**, and is indivisible, so one
  fold is always at least that big. The pool uses **4 folds**, not 5: four puts a small pupil
  in every fold and tightens the condition balance. After the 2026-08-17 intake, development-fold
  sizes are now 76/58/73/105 after the HQL097 intake, and a separate 4-image session is the
  outer holdout. HQL097 alone supplies 61 images to fold 4, so keep an eye on session dominance
  when reading the next baseline.
- `HQL080_sleep250625` holds **10 of the 14 tiny masks in the entire dataset**. Stratification
  spread the rest so every fold now contains some tiny mask, up from 2 of 5, and cut the
  cross-fold median-diameter spread from 3.03x to 1.60x. Coverage is still thin: only four
  sessions contain a tiny mask at all. Report mean per-session IoU; `run_cv.py` prints which
  bins each fold actually scored.

**Measured on 2026-08-16.** See `reports/2026-08-16-selection-metric-repair.md`. Seven sweeps,
28 fold-trainings, ~1 h 40 m on MPS.

- **Cross-recording mean per-session IoU is 0.6245 +/- 0.0322** (natural sampling, three seeds),
  against the 0.8749 macro IoU the project has been publishing. The image-weighted figure, which
  is what that 0.8749 is comparable to, is 0.57-0.64.
- **Transfer is bimodal**, not uniformly mediocre: six sessions score below 0.45 and six above
  0.75. The model either transfers to a recording setting or largely fails on it.
- **Seed noise on the grouped split is ~4x the documented floor.** The +/-0.0069 in
  `reports/2026-08-14-checkpoint-noise-floor.md` was measured on the leaky split. Here the sd is
  0.0273 on the three-seed mean and up to 0.0873 on one fold. `run_cv.py`'s docstring has been
  corrected; treat single-fold differences below ~0.05 as noise.

Then:

- Treat the 2026-08-17 255-image seed-0 result only as the previous-pool baseline: fold macro IoUs
  0.7754/0.4990/0.5400/0.6701, mean per-session IoU 0.6468. Do not compare a new configuration
  on the 316-image pool against it as though the pools were matched.
- Run the same seed-0 natural-sampling, macro-IoU configuration on all 312 development images
  next. That establishes whether the HQL097 pupils and true negatives correct the aperture grab
  before changing the loss, sampler, or architecture.
- Freeze the final epoch schedule and threshold using development evidence only, refit on all
  312 development images, then consume the four-image trial5 holdout exactly once with
  `training/evaluate_holdout.py`.

## Improving Cross-Recording Generalization

Status: open; the previous 255-image seed-0 baseline is 0.6468 mean per-session IoU, and the failure
mechanism remains diagnosed. See `reports/2026-08-16-selection-metric-repair.md` section 4b for the
older 222-image experiments.

Transfer is bimodal - six sessions below 0.45, six above 0.75 - so the question is not "why is the
model mediocre everywhere" but "what makes a recording setting fall off the cliff". **Read
[New labels](#new-labels-and-experiment-sequencing) before starting a sweep and keep configuration
comparisons on the same pool version.**

**Settled: the model segments the eye aperture, not the pupil.** Measured, not inferred. On the two
failing sessions the fold-1 checkpoint predicts **4.8x and 7.8x** the labelled area, at **p = 0.99
on every frame**, while the session that works in the same fold predicts 0.72x. It has learned
"dark blob = pupil", which holds while the pupil is the only dark thing in frame and breaks where
the whole eye aperture is dark. No appearance statistic explains it: brightness correlates with
per-session IoU at rho +0.02, boundary contrast at **-0.19**, and the failing session has *better*
boundary contrast than its high-scoring near-twin. Do not re-derive this from summary statistics;
look at a prediction.

**Settled: augmentation is the wrong tool for it, and was withdrawn as a recommendation.**
Brightness and contrast jitter cannot teach a pupil-versus-aperture distinction, which is semantic
rather than photometric. `ColorJitter(brightness=0.2, contrast=0.2)` and Gaussian blur are already
in `mouse_pupil_analysis/augmentation.py`, so the proposal amounted to turning up a knob pointed at
the wrong variable. Scale jitter is present too and plausibly works *against* this failure by
eroding the size prior that would rule out a region five times too large - if anything there is a
case for reducing it. Do not re-open a photometric augmentation sweep without new evidence.

What the diagnosis actually supports, in order:

1. **Labels from sessions where the whole eye aperture is dark.** This teaches the distinction
   directly and is the one item that next week's batch can serve. See
   [New labels](#new-labels-and-experiment-sequencing).
2. **The two-stage eye-ROI then pupil model.** Parked under
   [Segmentation fine-tuning](#segmentation-fine-tuning-and-visibility) as speculative; the
   diagnosis makes it the best-motivated structural fix on the list, since finding the eye first
   removes the ambiguity the model is currently resolving wrongly.
3. **Revisit the loss function.** All runs so far use BCE alone. The BCE+Dice / focal / Tversky
   ablation was parked until sampling, metric, and calibration were settled; those are now settled,
   so it is unblocked. Dice optimises overlap directly, which is what is being measured.
4. **Re-examine fixed-threshold selection on fold 3.** Fold 3 lost 0.056 against the old
   calibrated-threshold selection - 5.7 sd of its own seed spread, and the one place the
   2026-08-16 selector repair measurably hurt. Either accept it as the price of a stable selector
   or make the selection threshold adaptive.
5. **Add seeds 3-4** only if the sampling margin needs tightening. The sign is consistent across
   every seed, but 0.0354 sits close to the three-seed sd of 0.0273.

## Frame Recommendation

Status: shipped and validated; fresh four-member committee trained 2026-08-17 under
`checkpoints_exp/cv255_nat_macro_20260817`. `training/recommend_frames.py`, scoring in
`training/frame_selection.py`, harness in `reports/scripts/validate_frame_selection.py`.

The first real recommendation batch is complete: 20 picks each from HQL090, HQL097, and HQL103,
stored under `frame_recommendations/`. HQL090 reused an already-labeled session, so it used three
fold-1 seeds that all excluded that session; the two new sessions used the four fold models.

Takes a video or a frame folder and returns the frames worth labelling by hand. Recommended frames
average IoU 0.31 against 0.51 for random picks, with an oracle floor of 0.29 - 89% of the
achievable gap, beating random in 10 of 10 sessions.

**Settled: never rank by model confidence.** The failure worth catching runs at p = 0.99, so
entropy or margin sampling ranks exactly the wrong frames as least informative. Ranking is by
committee disagreement plus a geometric plausibility prior. Disagreement alone is near-blind to the
aperture-grab, because members trained on different folds share the bias and agree while all being
wrong; the geometric term is what recovers those frames (rho 0.09 -> 0.84 on
`251016_5212_purple_Day10`).

**Re-run the harness after any change to the scoring.** Two changes that read as improvements have
already measured worse: combining the geometric penalties by their maximum rather than their mean
(89.2% -> 80.8%), and setting the near-duplicate cutoff from a low percentile, which collapses to
zero and silently disables deduplication exactly when duplicates are common.

Remaining work:

- HQL097 is complete: contextual review produced 47 pupil masks, 14 all-black true negatives,
  and 6 uncertain frames excluded from segmentation training. Label HQL103 next because it is a
  genuinely new session. HQL090 is lower priority because development already contains that
  session.
- Fold the temporal signal into `implausibility_score` as a physiological bound rather than a
  separately weighted term - see [Sampling rate](#sampling-rate-and-pupil-dynamics). It currently
  defaults to weight 0 because it could not be validated on a sparsely sampled pool.
- The default committee globs `checkpoints_exp/cvnat/*/best.pth`, which is **gitignored**. It works
  only on a machine that still holds those 24 checkpoints; a fresh clone cannot run the tool. Either
  document the prerequisite or ship a smaller committee.

## Sampling Rate And Pupil Dynamics

Status: measured 2026-08-16 on `sample_data/velocity_frames`. Report section 4c.

**Settled: pupil size and pupil position need different sampling rates.** At 97 Hz the
frame-to-frame *size* signal is segmentation noise, not physiology: the maximum observed change of
8.92% on a 24 px pupil is 2.1 px, and a one-pixel boundary shift on a disc of radius 12 accounts
for 8.3% by itself. Published mouse work samples size at 10-50 Hz, and a sleep-staging study
classifies from diameter at 10 Hz, bounding size bandwidth well under 5 Hz. Position is the fast
quantity - 1.05 px/frame is 426% of a diameter per second - and is what justifies the 97 Hz rig.
For size alone the recording is oversampled roughly tenfold.

Consequence, which is counterintuitive: **for the temporal frame-selection signal, higher fps is
worse.** Jitter is roughly constant per frame while real change shrinks with the interval, so at
97 Hz signal and noise are comparable, whereas at 5 fps a physiological ~100%/s gives ~20% per
interval against ~1% jitter. The 5 fps extraction default is in the right regime.

Remaining work:

- Measure a mouse maximum constriction velocity from a light-driven PLR on this rig. No reliable
  published mouse figure was found; the mm/s values in the literature are human and do not transfer
  across a 2-8 mm versus 0.5-2 mm range. That number sets the plausibility bound above.

## New Labels And Experiment Sequencing

Status: HQL097 integrated and outer holdout still frozen (2026-08-18)

The pool is now 316 pairs across 19 sessions: 312 development pairs and the same 4-image outer
holdout. HQL097 contributed 61 development pairs (47 visible pupils and 14 explicit empty-mask
negatives); its 6 uncertain annotations are preserved separately and carry no segmentation mask.
**No score from the 222- or 255-image pools is a matched comparison against this pool.**

**Decide before the images are merged, because merging is irreversible in evaluation terms.**
Nothing currently in the pool can judge the packaged checkpoint: it gradient-trained on 166 of the
222 images, and its own validation set drew 54 of 56 images from recordings that also fed training,
so at session granularity it has seen everything. Genuinely new sessions are the only clean test it
will ever get, and the moment they enter a training fold that value is gone.

`260807_3582_Purple_trial3` contributes 29 development pairs;
`260812_3582_Purple_trial5` contributes 4 holdout pairs and is assigned to no fold. All training
filenames use the compact `frame_<five-digit-source-index>` form. HQL097's generated-mask JSON
copies were removed after verification; the source annotations and six uncertain records remain
available outside the paired training directories.

New Labelme batches now enter through `training/import_labelme_batch.py`: preview first, then
apply with `--refresh-splits`. The importer keeps uncertainty as image-level metadata outside
segmentation training rather than inventing an "uncertain mask."

Remaining work:

- Keep trial5 untouched until the final schedule and prediction threshold are frozen from CV.
- Record the session at intake for every future batch, per `training/data_collection.md`.
- Train the matched 316-image baseline before interpreting whether the new negatives helped.
- Label HQL103 next and add it as another development session rather than modifying trial5.

## Training Sampling Default

Status: fixed on 2026-08-17; `balance_training_sizes` defaults to `False`. See
`reports/2026-08-16-selection-metric-repair.md` section 3a.

Natural sampling beat size-balanced by **0.0354** mean per-session IoU, winning at all three
seeds and in 8 of 12 matched (fold, seed) cells. The two arms differed only in
`--natural-sampling`, so every cell is a matched pair; that pairing is what makes a 0.035 effect
readable against fold difficulty spanning 0.35-0.73.

**Settled: equal-mass size balancing does not do the job it was added for.** Its purpose was to
protect small pupils. Across the four folds the tiny bin splits 2-2 between the arms, and the
largest single tiny-bin gap (fold 0, 0.2835) favours *natural*. Meanwhile balancing costs
large-pupil IoU 0.2001 on fold 1 and 0.2711 on fold 2. It pays a real price for a benefit it does
not deliver.

**The predicted failure mode was wrong.** Fold 3 holds out `HQL080_sleep250625`, leaving ~4 tiny
masks that balancing inflates to a third of the training mass - the predicted worst case. It is
instead the only fold where balancing consistently wins (~0.015, seed sd 0.0099). Balancing's cost
shows up in folds with *ordinary* tiny counts, not in the starved one. Do not re-derive the
oversampling-collapse story; it is not what the data shows.

No code follow-up remains; use `--balance-sizes` only for an intentional comparison.

## Model-Selection Metric Fragility

Status: **fixed** on 2026-08-16. See `reports/2026-08-16-selection-metric-repair.md`.

The fragility was worse than "the metric is noisy." Under `balanced_iou`, three of four folds in
the first grouped sweep selected a checkpoint at **epoch 4-6 of 400** on a one-off spike in a size
bin holding one to three images. Because the same metric also drove `ReduceLROnPlateau` on
`mode="max"`, that spike became a high-water mark no later epoch could clear, and the learning rate
decayed to near its floor while the model was still improving. Folds were mis-selected *and*
under-trained; fold 0 was one step off `min_lr` when early stopping fired.

Repairing it moved the measured baseline from 0.5378 to 0.5891 (same sampling, same seed set), and
folds 0 and 2 gained 0.38 and 0.14 macro IoU.

Three changes, all in `training/run_train.py` and forwarded by `run_cv.py`:

- `ReduceLROnPlateau` now runs on `val_loss` (`mode="min"`); `--scheduler-metric` overrides.
- `--selection-metric {balanced_iou,macro_iou}` chooses what "best" means and also ranks threshold
  candidates. Macro IoU is now the default; pass `balanced_iou` only to reproduce an older run.
- `--selection-threshold` (default 0.5) compares epochs at one fixed threshold instead of each
  epoch's maximum over 11 candidates. `calibrated` restores the old behaviour.

**Known cost:** fold 3, whose old selection was sound, lost 0.056 under fixed-threshold selection -
5.7 sd of its own seed spread. Fixed-threshold selection is not free where calibrated selection was
already working.

Background from `reports/2026-08-14-checkpoint-noise-floor.md`, retained because the reasoning
still holds and the 2026-08-16 sweeps confirmed the first point empirically:

- **Two images decide a third of the metric, and the tiny bin is really one session.**
  `balanced_iou` is the mean of the tiny, medium, and large bins; the old validation set held
  exactly 2 tiny masks, and across the whole pool `HQL080_sleep250625` holds 10 of the 14. So
  the tiny bin measures one recording setting, not a size regime, and under grouped folds it is
  thin in most folds. Small pupils are not a training-data bottleneck: in absolute
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

## Runtime Modularization

Status: complete on `dev`; `api.py` owns orchestration, with focused `results`/`plotting`/
`preprocessing`/`augmentation` modules and a deprecating `dataset.py` shim

Remaining work:

- Remove the `dataset.py` shim and the deprecated `generate_pupil_mask_prediction` after one release.
- Decide whether the unified plot should show `pupil_diameter_input_pixels` instead of the
  model-pixel column, now that both are exported. This changes the README demo, so it is
  deliberately deferred.

## Closed

Threads with no open work. Kept as one-liners because each carries a constraint that is easy to
violate by accident; the history is in `work_log.md` and `work_log_archive/`.

- **Packaging and distribution** - 0.2.0 published as `mouse-pupil-analysis`. *Settled: this
  project ships no `pupil_tracking` module and no compatibility shim.* The unrelated
  `pupil-tracking` distribution on PyPI installs its own `pupil_tracking/__init__.py`; installing
  both in either order silently overwrites one with the other. CI and the release workflow both
  fail if `pupil_tracking/` reappears in a built artifact. Do not re-litigate. Tag and release only
  when the maintainer resumes release work, never as part of routine `dev`/`main` integration.
- **DOI archival** - concept DOI `10.5281/zenodo.21897795`, version DOI `10.5281/zenodo.21897796`.
  The per-version `identifiers` entry must stay **first**: citation generators emit the first
  doi-type entry and only fall back to the top-level `doi` when the list is absent, so a
  concept-first ordering makes exported BibTeX cite the moving concept DOI rather than the archived
  code (verified with `cffconvert`). Per-release update is `RELEASING.md` step 8.
- **Sample data and fixtures** - `sample_data/` is an exploration and smoke-test resource, not a
  benchmark or a training set; keep it compact and expand only for a specific uncovered behaviour.
  `tests/test_real_images.py` is the only test that can catch a corrupted or swapped checkpoint,
  because synthetic input segments plausibly regardless of the weights. The two source resolutions
  are what make the input-pixel diameter conversion verifiable against real geometry.
- **Training workflow documentation** - `training/README.md` covers the workflow. Fine-tuning
  deliberately restores weights only, starting fresh optimizer, scheduler, early-stopping, and
  logging state; add structured optimizer/scheduler checkpointing only if exact interrupted-run
  resume becomes necessary.

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
