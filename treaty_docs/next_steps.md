# Next Steps

Use this file only for active follow-ups and decisions. Historical evidence belongs in
`work_log.md` and `reports/`.

## Currently Hot

- [Cross-recording generalization](#cross-recording-generalization) — diagnose the
  Purple-sleep failure and test the component-selection guard on new sessions.
- [Velocity quality control](#velocity-quality-control) — measure a mouse PLR constriction
  bound, then validate the quality thresholds on more recordings.
- [Frame recommendation](#frame-recommendation) — make its completed-CV-committee prerequisite
  clear wherever a fresh-clone user encounters it.

## Cross-Recording Generalization

Status: open. On the weak Purple-sleep fold, the model can identify a dark eye aperture as the
pupil. Augmentation alone cannot resolve this semantic error.

Next actions:

- Collect and label diverse dark-aperture recordings.
- Compare the existing center-favored component selector across sessions.
- Run any loss, threshold, or architecture experiment against the same labeled pool and report
  results for each session.

## Velocity Quality Control

Status: provisional quality thresholds are implemented.

Next actions:

- Measure a mouse PLR maximum constriction velocity on this rig.
- Validate the quality thresholds on additional recordings before treating them as universal.
- Consider smoothing only after measuring center jitter and preserving fast REM movements.

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
