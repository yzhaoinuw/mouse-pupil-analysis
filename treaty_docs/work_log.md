# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

## 2026-11-09

### Promote the geometry-trained checkpoint (Codex, GPT-6, effort and token budget not exposed)

- The maintainer accepted the all-615 refit for the next release and requested
  delivery through `dev`. Packaged its ungated mask export with the CV-derived
  threshold 0.525, metadata, and redacted training log. Archived the prior
  packaged model locally. The filename's 0.6088 summarizes four-fold CV
  recipe selection.
- The velocity fixture now yields 25 valid frames and six area-change warnings.
  Its model-space diameter range is 17.55 to 25.26 pixels. Updated the
  fixture expectations and documented the accepted default under Unreleased.
- Verification: Ruff, Black, and the full pytest suite passed. Rebuilt wheel
  and sdist from a fresh staged snapshot after finding stale old weights in
  the local build cache. Matched both distributions' three checkpoint files
  byte for byte to the accepted export, verified namespace and archive
  exclusions, and cleared the stale cache files. Installed the wheel in a
  separate environment, analyzed three real sample frames, and checked both
  console commands. Ran `git diff --cached --check`.

## 2026-11-06

### Prepare the geometry branch for review (Codex, GPT-6, effort and token budget not exposed)

- Scoped the branch to the geometry workflow and the loss, augmentation, and
  overlay code needed to reproduce its runs. Kept generated predictions,
  weights, and unrelated MEYE work local.
- Verification: exported the staged source with `git checkout-index`, matched
  the recorded CV source hashes with Git line-ending normalization, and ran
  repository-wide Ruff, Black, and pytest in that snapshot. Ran
  `git diff --cached --check`.

### Test whole-blob selection with the learned center (Codex, GPT-6, effort and token budget not exposed)

- Ranked 8-connected components by mean mask confidence, saturating area
  support, and a soft preference for the nearest foreground pixel to the
  learned center. The center scale is 12 model pixels and area support uses
  the existing 64-pixel scale. Retained whole components and used no radius.
- Across 611 labeled validation frames from four folds at threshold 0.525,
  equal-session IoU was 0.671 raw, 0.699 with the existing image-center
  selector, and 0.698 with the learned-center selector. The learned center
  removed 18,953 detached false-positive pixels and 14,823 true-positive
  pixels from the raw masks on positive frames. The two selectors trade
  missed pixels against false foreground without a clear IoU advantage for
  the learned center.
- Ran the all-615 checkpoint on all 12,950 test PNGs. Selection changed
  1,346 masks, and the other rendered files match the previous raw overlays
  byte for byte. The v0.3.0 overlays, per-image change manifest, and CV
  metrics are under `images_test_prediction/geometry_center_2026-11-06/`.
- Verification: matched saved split, source, and checkpoint hashes, verified
  every output PNG, and matched the input inventory. Compared
  all overlay bytes to the raw run and inspected one before-and-after pair.
  Ruff, Black, focused and full pytest, and `git diff --check` passed.

### Render the release candidate with the shipped overlay (Codex, GPT-6, effort and token budget not exposed)

- Rendered all 12,950 test PNGs with the all-615 candidate at threshold
  0.525 using the Python package from the v0.3.0 Git tag. The released
  renderer uses the resized and padded 148-square grayscale view and the
  default 5% confidence heatmap. Saved overlays under
  `images_test_prediction/geometry_release_2026-11-06/`, with source-relative
  paths preserved and the seven prior previews or plots flagged.
- Verification: checked the candidate checkpoint hash and the tagged
  renderer import, verified all 12,950 RGB PNGs and their exact correspondence
  to the inputs, and inspected two overlays. Ran `git diff --check`.

### Complete geometry CV and train the release candidate (Codex, GPT-6, effort and token budget not exposed)

- Reused the verified fold-2 checkpoint and trained folds 0, 1, and 3 at seed
  0 with the original two-head setup. Selected epochs were 41, 79, 55, and 71
  and calibrated thresholds were 0.55, 0.50, 0.60, and 0.50 in fold order.
  The existing recipe rule produced 100 epochs, learning-rate drops at 50
  and 75, and threshold 0.525. The raw mask at that shared threshold averaged
  0.671 IoU across 24 CV validation sessions. The mean of the four
  fold-calibrated image-mean IoUs was 0.6088.
- Trained one scratch refit on all 615 labeled pairs using the new recipe.
  The CV results and recipe are under
  `checkpoints_exp/geometry_release_cv_2026-11-06/`. The candidate is under
  `checkpoints_exp/geometry_release_candidate_2026-11-06/`, with full
  geometry weights and a mask-only export compatible with the shipped loader.
  Both exports come from the same refit. The promotion decision is pending
  visual review of the ungated mask.
- The review under `images_test_prediction/geometry_release_review_2026-11-06/`
  compares the shipped model at 0.50 and the candidate at 0.525 on 94 sampled
  images across 32 recording prefixes. It samples near 10%, 50%, and 90%
  within each prefix and is a qualitative comparison.
- Verification: matched the fold-2 data, source, initialization, and checkpoint
  hashes, then checked the complete CV recipe and all 615 input pairs before
  and after the refit. Verified bit-identical logits between the full and
  mask-only exports, passed checkpoint packaging in dry-run mode, and checked
  the shipped checkpoint hashes. Verified all 94 panels and inspected panels
  000 and 052. Ruff, Black, full pytest, and `git diff --check` passed.

### Render separate test overlays from both heads (Codex, GPT-6, effort and token budget not exposed)

- Reused the fold-2 epoch-55 checkpoint for all 12,950 PNGs in `images_test/`,
  including 12,943 original frames and seven historical previews or plots.
  Saved two original-size overlays per source under ignored
  `images_test_prediction/geometry_head_2026-11-06_fold2/`. The cyan mask uses
  raw probabilities at threshold 0.5. The yellow circle and center use raw
  geometry on every image. The manifest records both paths and geometry in
  model and source coordinates.
- Resumed after 10,252 pairs when the Windows path limit blocked the long
  preview folder. Its three source paths map to `prior_previews/` in each
  output head. This batch is for qualitative review across recordings.
- Verification: matched the checkpoint hash to the experiment record, matched
  12,950 source SHA-256 hashes and verified all 25,900 PNGs against the
  manifest and original dimensions. Inspected square and non-square examples.
  Ran `git diff --check`.

### Train and evaluate the shared geometry head (Codex, GPT-6, effort and token budget not exposed)

- On `feature/geometry-head`, matched the fold-2 BCE reference at seed 0.
  The mask used pixel BCE, and the auxiliary head used normalized Smooth L1
  with weight 1 on circles enclosing the augmented nonempty labels. Validation
  scheduling used mask BCE. Selected epoch 55 by raw IoU at 0.5 and stopped at
  epoch 75. The inference gate passes probabilities through the predicted
  radius plus two model pixels and fades them over the next four pixels.
- At threshold 0.5, equal-session IoU was 0.628 for raw BCE, 0.636 with its
  component selector, 0.661 for the two-head model's raw mask, and 0.575 after
  gating. The two-head raw mask improved in four of six sessions and reduced
  detached foreground from 2,375 to 766 pixels on positive frames. It regressed
  on Purple trial 3 (0.800 to 0.786) and HQL086 (0.686 to 0.641), and produced
  foreground on three of 14 empty frames versus two for BCE.
- Gating removed 4,996 false-positive and 26,804 true-positive pixels from
  the raw candidate on positive frames. HQL086 accounted for 20,872 of those
  lost true pixels, with radius underestimation averaging 20.63 model pixels.
  HQL097 benefited (0.620 to 0.665); the gate lowered IoU in the other five
  sessions. Advance the auxiliary head with raw output to a separate fold and
  seed comparison. Retain the learned gate for analysis of clipping.
- Artifacts in `checkpoints_exp/geometry_head_2026-11-06_fold2/` include
  `summary.json`, `circles.csv`, the experimental checkpoint, and a five-column
  `review/index.html`. This was validation used for checkpoint selection.
- Verification: the GPU runner exited successfully after matching split and
  source-pair hashes, baseline starting weights and RNG, and every saved BCE
  mask. It checked that the gate never added foreground. Compared original and
  refactored logits bit for bit, verified all 120 enclosing circles and 14
  empty targets, audited 536 mask PNGs and 134 review panels, and visually
  inspected panels 063 and 097. Ruff, Black, `git diff --check`, and focused
  and full pytest runs passed.

### Branch for the geometry-head experiment (Codex, GPT-6, effort and token budget not exposed)

- Created and switched to local `feature/geometry-head` at the `dev` tip
  (`9b54e71`), carrying the existing working tree into the new branch. The
  mask head will use the same visible-pupil labels and pixel BCE. The proposed
  geometry targets are circles enclosing the labeled pixels and describe
  visible-pupil support. Implementation is pending.
- Verification: checked branch, commit, and working-tree status before and
  after the switch, then ran `git diff --check`.

### Screen a center-and-radius prior (Codex, GPT-6, effort and token budget not exposed)

- The maintainer proposed a shared model with visible-mask and center/radius
  outputs to suppress stray foreground. On the saved BCE fold-2 baseline,
  tested a label-derived circle as a feasibility check. At threshold 0.5,
  intersected each nonempty frame's prediction with the minimum enclosing circle
  of its visible label plus a fixed 2-model-pixel margin. Kept raw predictions
  on the 14 empty labels so the screen received no credit for detecting closure.
- Across 120 positive frames, the gate reduced false-positive pixels from 18,846
  to 6,196 and detached false-positive pixels from 2,375 to 17. Here detached
  means a predicted 8-connected component with zero label overlap. The gate
  preserved true-positive pixels by construction, but 21 positive frames still
  had multiple components, down from 72. The current image-center selector
  produced one component where present and removed 1,793 true-positive pixels.
- Equal-session mean IoU was 0.628 raw, 0.636 with the existing selector, and
  0.674 with the label-derived circle. Most of the circle's IoU gain came from
  HQL097 (0.558 to 0.757). Purple trial 3 went from 0.800 to 0.820. All three
  outputs had false foreground on two empty frames. The circle results measure
  spatial separability on one development fold using known geometry. The
  accuracy of geometry predicted from the image remains an open question.
- Verification: replayed the saved BCE checkpoint on CPU in the `pupil_tracking`
  environment, matched all 134 saved raw masks pixel for pixel and checked target,
  false-positive, and missed-pixel counts against the run CSV. Verified that each
  padded circle covered every target pixel. Compared raw, existing selector, and
  label-derived gate using the same probabilities and masks. Ran `git diff --check`.

## 2026-10-22

### Prepare the treaty update for dev delivery (Codex, GPT-6; effort and token budget not exposed)

- Scoped the requested commit/push to the treaty v0.9.1 update, published feedback,
  and preserved work-log history. Earlier overlay code/tests/README edits and the
  pupil-measurement planning section remain intentionally local.
- Retained the prior log rotation and checked all nine previously committed date
  sections against the live/archive history; their contents are unchanged.
- Refreshed `origin/dev`; it matched local HEAD before delivery. No main-branch
  merge, PR, version bump, tag, or release is part of this handoff.
- Verification:
  - Repository-wide Ruff passed; Black reported 56 files unchanged. Full Pytest
    passed: 153 tests, with 346 existing deprecation warnings. The first sandbox
    run hit `WinError 5` on a temporary-directory rename; the successful host run
    used `python -m pytest -o addopts= -q
    --basetemp=.pytest_tmp_commit_20261022_host/tmp
    -o cache_dir=.pytest_tmp_commit_20261022_host/cache` after creating its parent.
  - `treaty validate .` passed. Staged and unstaged whitespace checks and staged
    documentation links were reviewed before committing; package builds were
    not needed for this documentation-only handoff.

## 2026-10-21

### Publish the authorized treaty feedback (Codex, GPT-6; effort and token budget not exposed)

- Posted [issue #30](https://github.com/yzhaoinuw/agent_collab_treaty/issues/30)
  for non-root adopter-owned doc misclassification and a
  [verification-diagnostic follow-up on #15](https://github.com/yzhaoinuw/agent_collab_treaty/issues/15#issuecomment-6328064722)
  after the maintainer explicitly authorized publication. Kept our sandbox/setup
  friction out of the bug queue; the traceback suggestion remains deferred.
- The connector rejected issue creation with `403 Resource not accessible by
  integration`; sandboxed `gh` then returned `401 Requires authentication`.
  Retrying the same authorized issue creation with the host-authenticated CLI
  succeeded, as did the follow-up comment. No credential or Git configuration
  was changed. Updated local reports and next steps with the published links.
- Verification:
  - Refreshed the related issue search and #15 comments before posting; read back
    #30 and comment `6328064722` through the connector. Both bodies matched the
    prepared text after normalizing line endings; #30 was open at read-back.
  - `treaty validate .`, `git diff --check`, and the updated-document whitespace
    and relative-link checks passed. Documentation only; runtime tests were not run.

### Attribute update friction before proposing upstream reports (Codex, GPT-6; effort and token budget not exposed)

- Distinguished our stale CLI/PATH, sandbox permissions, dirty tree, Git newline
  handling, and noncanonical verification heading from treaty update defects.
  The standalone verification rule already exists in CLI 0.5.0; no new v0.9.1
  regression was demonstrated by this smoke test.
- Identified a concrete pre-existing bug: `treaty diff` exempts bare `work_log.md`
  and `next_steps.md` from risk, but misses the same files under `treaty_docs/`.
  Seven of the original 30 counted sections came from this misclassification.
- Prepared a focused upstream bug draft and optional diagnostic follow-up in
  [`reports/treaty_v0.9.1_upstream_feedback_2026-10-21.md`](reports/treaty_v0.9.1_upstream_feedback_2026-10-21.md).
  Reviewed issue history for overlap; #15 already discusses verification formatting
  and broad drift risk classes. Nothing was filed or posted upstream.
- Verification:
  - Compared published 0.5.0, 0.9.0, and 0.9.1 wheel sources without installing
    older versions. Reproduced root versus nested ownership classification for
    both filenames and standalone versus inline verification in all three.
  - Confirmed the ownership check remains in upstream main and reviewed the 21
    returned issue reports, including the full current issue #15.
  - `treaty validate .` and `git diff --check` passed; all five audited relative
    file links in the updated reports/planning/log resolved. Documentation only;
    runtime tests were not run.

### Test the treaty v0.9.1 update (Codex, GPT-6; effort and token budget not exposed)

- Upgraded the environment's treaty CLI from 0.5.0 to 0.9.1 and the repository's
  template pin from v0.9.0 to v0.9.1. Preview and apply agreed on exactly two files;
  no conflicts or rejected patches occurred. Existing code, planning edits, and
  untracked log archives were stashed and restored rather than committed as part
  of this update.
- Recorded reproducible feedback in
  [`reports/treaty_v0.9.1_test_drive_2026-10-21.md`](reports/treaty_v0.9.1_test_drive_2026-10-21.md):
  inline verification evidence is rejected, expected placeholder replacements
  appear among drift warnings, and sandbox Git failures produce long tracebacks.
  These observations do not establish a new merge regression in v0.9.1.
- Reformatted the October 6 verification heading without changing its evidence
  so the current validator can recognize it. Added the feedback review to
  `next_steps.md`; the update remains local on `dev`, with no commit or push.
- Verification:
  - Verified the latest GitHub/PyPI release; ran `treaty --version`, `treaty diff .`,
    `treaty update . --dry-run`, and `treaty update .` with CLI 0.9.1.
  - Confirmed the preview left the working tree clean and the apply matched it;
    compared restored content with the safety stash. `treaty validate .` and
    `git diff --check` passed. All nine audited relative file links resolved;
    no unmerged entries, conflict markers, or rejected patches remained.
  - Runtime tests and package builds were not run for this documentation update.

## 2026-10-06

### Choose direct pupil axes for partial eyelid occlusion (Codex, GPT-6; effort and token budget not exposed)

- Agreed to evaluate horizontal width and visible vertical height rather than reconstruct
  a full pupil ellipse. The maintainer reports poor results from prior contour fitting
  with imperfect masks, especially invented hooks or protrusions.
- Horizontal width is useful only while the widest horizontal section remains visible;
  height/width is a possible occlusion flag, not a direct eyelid-opening measurement or
  a classifier of sleepiness versus discomfort. Edge selection must tolerate protrusions.
- Recorded the pending measurement and validation work in `next_steps.md`; no runtime
  implementation or accuracy claim was made.
- Verification:
  Inspected the current area-to-diameter calculation in
  `mouse_pupil_analysis/pupil_predictions.py`; reviewed the documentation diff and ran
  `git diff --check -- treaty_docs/next_steps.md treaty_docs/work_log.md treaty_docs/work_log_archive`.
  Documentation only; runtime tests were not run.
