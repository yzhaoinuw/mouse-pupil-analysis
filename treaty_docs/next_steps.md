# Next Steps

Use this file only for active follow-ups and decisions. Historical evidence belongs in
`work_log.md` and `reports/`.

## Currently Hot

- [Targeted user failure cases](#targeted-user-failure-cases) — collect examples
  from v0.4.1 users before the next model improvement.
- [Circle-overlap blob selection](#circle-overlap-blob-selection) — review the
  accepted 20% rule on labeled predictions after integration into dev.
- [Treaty v0.9.1 feedback](#treaty-v091-feedback) — feedback posted; revisit the
  upstream fix when planning the next treaty update.
- [Cross-recording generalization](#cross-recording-generalization) — diagnose the
  Purple-sleep failure and test the component-selection guard on new sessions.
- [Velocity quality control](#velocity-quality-control) — measure a mouse PLR constriction
  bound, then validate the quality thresholds on more recordings.
- [Frame recommendation](#frame-recommendation) — make its completed-CV-committee prerequisite
  clear wherever a fresh-clone user encounters it.

## Targeted User Failure Cases

Status: the maintainer chose to release the approved pipeline as v0.4.1 and
prioritize targeted user data for a systemic model improvement. Thin-bridge
splitting and morphological opening are deferred; no extra cleanup is enabled.

- Collect original frames, all-head overlays, software version/command/checkpoint,
  expected visible-pupil boundaries, and recording/session identifiers.
- Follow `training/data_collection.md` for annotation and intake. Preserve recording
  grouping when evaluating model changes and include clean frames and tiny or
  partially occluded pupils alongside failure cases.
- Use the collected failure modes to choose the next model/data experiment.

## Circle-Overlap Blob Selection

Status: the accepted 20% circle-overlap rule is integrated into `dev`. Keep every
whole 8-connected blob with at least 20% of its area inside the predicted circle,
with no circle expansion or confidence ranking. The existing checkpoint and 0.525
threshold isolate postprocessing from the proposed training change. Measurements
use all retained pixels; QC still reports confidence and fragmentation.

- Review `images_test_prediction/blob_selection_2026-11-11/` against
  `images_test_prediction/all_heads_2026-11-10/`, particularly the HQL090 frame
  `HQL090_sleep251012_010_eye_24367.png`, empty results, and fragmented pupils.
  That completed 12,950-image batch records the original 5% setting. The 20%
  runner defaults to `images_test_prediction/blob_selection_20pct_2026-11-11/`
  for a separate rerun; the existing 5% overlays are preserved.
- Compare with raw masks and the previous center/confidence/area selector on
  labeled fold predictions before claiming an accuracy improvement. Check
  per-session scores and empty-label false foreground; tune overlap or circle
  expansion only if the comparison supports it.
- Test radius-to-zero supervision on empty masks separately. Current geometry
  training ignores those frames, so its circle is not a trained presence signal.

## Treaty v0.9.1 Feedback

Status: template update applied locally; preview and apply matched without conflicts.
Evidence and reproducible rough edges are in
[`reports/treaty_v0.9.1_test_drive_2026-10-21.md`](reports/treaty_v0.9.1_test_drive_2026-10-21.md).

Feedback posted: [bug #30](https://github.com/yzhaoinuw/agent_collab_treaty/issues/30)
and a [diagnostic follow-up on #15](https://github.com/yzhaoinuw/agent_collab_treaty/issues/15#issuecomment-6328064722).
The [`reporting record`](reports/treaty_v0.9.1_upstream_feedback_2026-10-21.md)
contains the evidence and scope. Next action: recheck #30 when planning the next
treaty update. The traceback suggestion remains deferred; no local runtime work
or upstream PR is pending from this test drive.

## Cross-Recording Generalization

Status: open. On the weak Purple-sleep fold, the model can identify a dark eye aperture as the
pupil. Augmentation alone cannot resolve this semantic error.

Next actions:

- Collect and label diverse dark-aperture recordings.
- The fold-2 geometry-head screen is complete. With the gate off, auxiliary
  circle supervision raised equal-session IoU from 0.628 to 0.661 and reduced
  detached foreground. The learned gate lowered IoU to 0.575 by clipping
  true pupil pixels, especially in HQL086. Four-fold geometry CV is now
  complete at seed 0. Its recipe selected 100 epochs and threshold 0.525.
  The accepted all-615 refit was published in v0.4.0 on 2026-11-09 at threshold
  0.525, but packaging incorrectly retained only the segmentation export and
  runtime omitted the intended geometry selector. The dev correction
  packages both heads and enables learned-center component selection by default;
  publication of that correction remains pending.
  README Usage includes independent segmentation and geometry inspection examples,
  plus combined `--show-all-heads` overlays and explicit raw-segmentation inspection.
  The local comparison is available at
  `images_test_prediction/geometry_release_review_2026-11-06/index.html`.
  Full candidate overlays in the shipped v0.3.0 style are available under
  `images_test_prediction/geometry_release_2026-11-06/`.
  A learned-center selector raised four-fold mean per-session IoU to 0.698
  versus 0.671 raw and 0.699 with the existing selector. The maintainer clarified
  that this combined pipeline was the intended release behavior. Its overlays
  remain under `images_test_prediction/geometry_center_2026-11-06/`; the manifest
  identifies changed frames.
  See the [work log](work_log.md#complete-geometry-cv-and-train-the-release-candidate-codex-gpt-6-effort-and-token-budget-not-exposed).
- Resume the geometry-selector deep dive before a new release. The predicted
  center influences blob selection; the radius is currently diagnostic only.
  Full-image resize-and-pad overlay rendering is restored, including HQL088
  frame 58779. All test PNGs were regenerated through the README console command
  with unchanged measurements. The updated overlays are available under
  `images_test_prediction/all_heads_2026-11-10/`, retaining confidence colors and
  marking the final region with a dashed contour. Its manifest records coverage
  and analysis CSV paths; `run.json` records the commands and CLI logs. Prior
  removal statistics remain in `checkpoints_exp/cli_all_heads_2026-11-10/previous_cropped_manifest.csv`.
  Review Purple trial 8 frame 03279: area support makes the 76-pixel left blob
  score 0.429 and beat the closer, more confident 49-pixel right blob at 0.373.
  It removes 68 pixels across three components and is flagged uncertain/invalid.
  Compare confidence, area, center distance, and possible radius information
  against labeled validation frames before changing the selection rule.
- Run any loss, threshold, or architecture experiment against the same labeled pool and report
  results for each session.

## Velocity Quality Control

Status: mean segmentation confidence is diagnostic only; the extra 0.90
confidence rejection/warning was removed. Pixel thresholding and weighted centers
are unchanged. Empty-mask, circularity, border, and temporal-area checks remain.

Next actions:

- Measure a mouse PLR maximum constriction velocity on this rig.
- Validate the quality thresholds on additional recordings before treating them as universal.
- Consider smoothing only after measuring center jitter and preserving fast REM movements.
- Review `images_test_prediction/HQL090_centers_2026-11-11/HQL090_sleep251012_010_eye/`
  and the matching center table/plot under `results/HQL090_centers_2026-11-11/`.
  This is the requested sparse-frame center inspection, not a velocity run.

## Frame Recommendation

Status: shipped. The recommender uses the latest completed local cross-validation committee in
`checkpoints_exp/cv`.

Next action:

- Make that local-committee prerequisite clear wherever a fresh-clone user encounters the tool.

## Deferred Until After v0.3.0

- Remove the deprecated `dataset.py` and `generate_pupil_mask_prediction` shims.
- Decide whether public plots should use `pupil_diameter_input_pixels`.

## Settled Constraints

- Keep recording sessions intact across folds; `training_data_split.json` records the training
  assignments.
- Keep the Purple trial-5 outer-holdout session separate.
- Package data must contain only `mouse_pupil_analysis`; CI and release checks protect this.
