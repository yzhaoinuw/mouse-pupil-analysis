# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

## 2026-11-11

### Tighten README and selector help (Codex, GPT-6, effort and token budget not exposed)

- Replaced the opening's experiment history and implementation detail with the
  user-facing mask, circle-overlap, and combined-area behavior. The option table
  now explicitly describes multiple-or-empty results, automatic geometry selection,
  its checkpoint requirement and alias, and the current 0.525 pixel threshold.
- Aligned CLI help, including a leftover learned-center description on the
  deprecated option. Clarified retained-versus-discarded overlay colors and
  fragmentation warnings. No inference behavior or future work changed.
- Verification: all four CLI-help/API parity tests passed; Ruff and Black passed
  for the help module. Checked the README against the selector, threshold resolver,
  overlay renderer, and union measurements. Final help rendering and whitespace
  checks accompany the documentation follow-up on `experiment/blob-selection`.

### Adopt 20% circle overlap and deliver the branch (Codex, GPT-6, effort and token budget not exposed)

- The maintainer accepted whole-blob circle selection after reviewing the 5%
  batch, requested increasing the minimum overlap to 20%, and authorized commit
  and push on `experiment/blob-selection`. The circle is still not expanded;
  the 0.525 pixel threshold, model weights, and training loss are unchanged.
- The Purple frame ending in `16-31-19.701_8400.png` has a 177-pixel left blob
  with 16.95% overlap. The 20% rule rejects it, reducing the retained area from
  528 to 351 model pixels while keeping the 319-pixel main blob. Smaller blobs
  with sufficient overlap remain; all-head colors still show rejected candidates.
  HQL090 retains its 62-pixel pupil blob at both overlap settings.
- Updated the pipeline introduction, selector help, architecture/measurement
  explanation, training guide, changelog, and checkpoint metadata description.
  The 12,950-image output remains the documented 5% run; the runner now defaults
  to a separate 20% destination. Two actual CLI previews are under
  `images_test_prediction/blob_selection_20pct_examples_2026-11-11/`.
- Delivery includes the selector, union measurements, runner, and regression
  coverage from this chat. Older unpublished documentation and experiment files
  remain local. The scoped commit rotates existing committed history's five
  oldest dates into `work_log_archive/work_log_2026-10-06_to_2026-11-09.md`;
  separate local history rotations are preserved without publishing them.
- Verification:
  - All 209 tests passed in the `pupil_tracking` environment with
    `--basetemp=checkpoints_exp/blob_selection_2026-11-11/twenty_pct_tests`.
    Boundary coverage checks exactly 20% passes, lower overlap fails, and whole
    blobs and union measurements remain intact. Repository-wide Ruff and Black
    passed with the unrelated `.pre-commit-cache/` excluded.
  - Ran the actual console pipeline on both reported frames with all-head views
    and 0.35 blend. Matched its diameters to independently selected areas and
    visually inspected the Purple preview. Verification data and CLI log are in
    `checkpoints_exp/blob_selection_20pct_2026-11-11/targeted/`.
  - Wheel/sdist builds succeeded using `--no-isolation` and a repository-local
    TEMP/TMP directory after Windows sandbox temporary-directory access failures.
    Verified both archives contain byte-identical checkpoint weights, training
    log, and updated metadata (still threshold 0.525), exclude archived weights,
    and pass `scripts/verify_distribution_namespaces.py`. Git remote-ref checks
    complete the branch delivery.

### Test circle-overlap whole-blob selection (Codex, GPT-6, effort and token budget not exposed)

- Created `experiment/blob-selection` from dev at the maintainer's request,
  preserving pre-existing local documentation and experiment files. The branch
  remains local and uncommitted; no push, merge, PR, or release was requested.
- Isolated the proposed 5% blob-area overlap rule from training: retain all
  qualifying 8-connected components whole, including pixels outside the circle,
  and return empty if none qualify. No circle expansion or blob-confidence
  ranking. Kept checkpoint weights and automatic threshold 0.525 unchanged.
  Radius-to-zero supervision for empty masks remains a separate future experiment.
- QC and tracking measure the retained union rather than silently reselecting
  its largest piece. Largest-component dominance remains a fragmentation warning;
  confidence/circularity flags still describe QC without altering the final mask.
- Preserved prior predictions. The recursive batch uses the documented console
  command per PNG-containing directory with `--show-all-heads`, the prior 0.35
  color blend, full-image resize/pad, and separate result paths. The runner records
  source/checkpoint hashes, commands/logs, per-image source/output hashes, retained
  areas, and comparisons with the previous overlay manifest.
- Completed all 15 console invocations in 719.39 seconds, producing 12,950
  overlays under `images_test_prediction/blob_selection_2026-11-11/` (12,943
  original frames plus seven existing derived PNGs, matching the prior run).
  Retained area increased in 572 images and decreased in 18; 83 final masks are
  empty, compared with 70 previously. These are
  unlabeled inspection results, not evidence of an accuracy improvement.
- The motivating HQL090 frame now retains the 62-pixel inner blob and rejects
  the formerly retained 222-pixel outer blob. Verified this independently from
  the probability map and again through the completed CLI manifest; visually
  inspected the batch overlay's corrected retained contour.
- Verification:
  - Full pytest passed (208 tests) in the `pupil_tracking` environment with
    `--basetemp=checkpoints_exp/blob_selection_2026-11-11/full_retry`, including
    exact 5% overlap, multiple retained blobs, singleton rejection, invalid
    circles, and union measurements. The initial focused run lacked a temporary
    parent directory; all 43 focused tests passed after creating it. Full-suite
    testing caught argparse interpreting a literal percent sign in help; escaped
    it and passed both the four CLI tests and the complete suite on the retry.
  - Repository-wide Ruff and Black passed with unrelated `.pre-commit-cache/`
    excluded; `git diff --check` passed. No packaging assets or dependencies changed.
  - The initial batch stopped on its source-hash guard after the help-text fix;
    preserved its partial artifacts under the experiment control directory and
    restarted to produce one consistent complete output tree. Every final overlay
    is readable RGB at 148 x 148, exact input/output/manifest inventories match,
    all source-image hashes are unchanged, and code/checkpoint hashes stayed fixed
    across the completed batch. Logs/results are under
    `checkpoints_exp/blob_selection_2026-11-11/cli_verified/`.

## 2026-11-10

### Deprecate the image-center CLI override (Codex, GPT-6, effort and token budget not exposed)

- The maintainer requested deprecating `--prefer_central_component` now that
  ordinary two-head inference already selects using the learned center. The CLI
  visibly warns and labels the flag deprecated in help, while preserving its
  fixed-image-center behavior for existing scripts. The underlying selector and
  Python compatibility parameter remain available for legacy checkpoints and
  comparisons. No removal version was set.
- Removed the active recommendation from the combined-head example and marked
  the options-table entry deprecated. Added the requested single-sentence
  introduction immediately before the inspection foldouts, linking "two heads"
  to the architecture overview. Existing selector-review follow-ups are unchanged.
- The maintainer requested commit and push to dev. Scope is the deprecation,
  inspection introduction, and their verification/context; unrelated local
  experiments and older documentation edits remain local. No release is requested.
- Verification:
  - All eight focused CLI/central-selector tests passed. New tests invoke the
    actual console command on public frames, verify warning presence only when
    the flag is used, and match CLI diameters against the corresponding Python
    API behavior with and without the legacy override. Ruff and Black passed for
    changed Python files; `git diff --check` passed. Verified the README sentence's
    placement and architecture link target.
  - Commit checks re-ran repository-wide Ruff and Black successfully. Full pytest
    ran 198 cases: 197 passed and one existing annotation-import test hit Windows
    `WinError 5` during a temporary-folder rename. All four tests in that module
    passed on a fresh temporary path immediately afterward.

### Deliver overlay corrections on dev (Codex, GPT-6, effort and token budget not exposed)

- The maintainer approved the restored full-image display and requested commit
  and push. Delivery includes confidence-preserving all-head overlays, dashed
  retained contours, full-image resize/padding in all overlay modes, and the
  Usage and regression updates. The successful README CLI rerun provides review
  artifacts; generated predictions remain local. Unrelated experiment files and
  earlier local documentation edits are excluded from the commit.
- Verification:
  - Rechecked all 196 tests successfully on `.pytest_tmp_overlay_commit`, plus
    repository-wide Ruff and Black with the local pre-commit cache excluded.
    Reviewed the scoped staged diff and whitespace before committing. The
    12,950-image CLI coverage and unchanged-diameter verification are recorded
    below. This delivery targets dev without a main merge, tag, or release.

### Restore full-image overlays through the README CLI (Codex, GPT-6, effort and token budget not exposed)

- The maintainer requested deleting the cropped predictions and starting over,
  preferably through the documented Usage command. Restored the renderer's
  full-image aspect-preserving resize/pad, with probabilities, geometry, contour,
  and segmentation center drawn directly in the same 148-square model coordinates.
  Input preprocessing, learned weights, selection, and measurements are unchanged.
- Verified the absolute deletion target stayed inside this repository and was
  exactly `images_test_prediction/all_heads_2026-11-10/`, with no linked children.
  Deleted its 12,950 cropped PNGs. Preserved only its run/manifest metadata under
  `checkpoints_exp/cli_all_heads_2026-11-10/` for measurement comparisons; source
  images and other prediction directories were left untouched.
- Regeneration uses the actual `run-pupil-analysis.exe` console entry point and
  README's `--image_dir`, `--output_mask_dir`, `--show-all-heads`, and
  `--mask_transparency 0.35` arguments, once per PNG-containing directory.
  Documented `--result_dir` keeps analysis outputs outside the test inputs;
  `--num_workers 0` avoids repeated Windows worker startup. No custom inference
  or renderer is used by the batch driver. Commands and CLI logs are retained.
- Verification:
  - All 23 focused head/overlay tests passed. Full pytest ran 196 cases: 195
    passed and one unrelated annotation-import test hit Windows `WinError 5`
    while renaming its temporary directory; that test passed on a fresh short
    temporary path. Ruff and Black passed with the pre-commit cache excluded.
  - Regression coverage checks the full rectangular source survives resizing
    and padding and both heads retain model-space alignment. Ran the console on
    the reported HQL088 frame separately and inspected the corrected full eye.
  - Completed all 15 console invocations in 395.89 seconds: 12,950 readable RGB
    overlays, including seven pre-existing derived PNGs. Exact source/output/
    manifest inventories match, source hashes are unchanged, and every CLI
    diameter matches the prior retained-mask area. Inspected the actual batch's
    corrected HQL088 overlay. `run.json` records every command and log path;
    the manifest links each image to its analysis CSV. Ran `git diff --check`.

### Diagnose cropped overlay displays (Codex, GPT-6, effort and token budget not exposed)

- The maintainer flagged the apparent zoom/crop in the all-head batch using
  `HQL088_whiskerb251006_008_eye_58779.png`. Confirmed that source is 249 x 203,
  while both historical U-Net and new overlays are 148 x 148. The historical
  renderer resizes and pads the entire source; the current renderer instead takes
  source pixels from x=50..197, y=27..174 and reprojects the mask onto that crop.
- Traced the rendering change to commit `ee08fe7` on 2026-11-06. It is independent
  of the geometry head and dashed-contour feature. The latest batch inherited it.
  Inference preprocessing still resizes and pads the full image, so this display
  crop does not alter model inputs or measurements. Restoring the earlier full
  image view and regenerating overlays is recorded as a follow-up.
- Verification:
  - Inspected source, historical U-Net overlay, and new overlay. Compared the
    current renderer against v0.3.0 and the introducing commit, and checked the
    current inference dataset preprocessing. Investigation only; runtime code
    and generated predictions were left unchanged. Ran `git diff --check`.

### Render all test images with head contours (Codex, GPT-6, effort and token budget not exposed)

- Ran the updated all-head view on all 12,950 PNGs recursively under
  `images_test/`, preserving relative paths under
  `images_test_prediction/all_heads_2026-11-10/`. This includes 12,943 original
  frames and seven existing preview/plot PNGs, flagged in the manifest. Earlier
  predictions were preserved. Used the packaged full checkpoint, threshold
  0.525, and the approved 0.35 confidence blend with unchanged geometry styling
  and dashed retained contours.
- The run completed in 322.15 seconds. Selection removed 26,820 model pixels
  across 1,346 images, matching the prior selector run. `manifest.csv` records
  source/output hashes, geometry, and retained/removed areas; `run.json` records
  settings, checkpoint and source hashes, software versions, and folder coverage.
- Verification:
  - Verified every rendered PNG is readable RGB at 148 x 148. Independently
    matched the complete input, output, and manifest inventories with no missing
    or extra images, checked manifest totals, decoded a representative image
    from every folder, and inspected the actual batch output for frame 03279.
    Ran `git diff --check`. Inference only; no runtime changes or new training.

### Preserve head colors in combined overlays (Codex, GPT-6, effort and token budget not exposed)

- The maintainer requested combining each head's existing appearance rather than
  recoloring retained/discarded regions. All-head overlays now retain the full
  yellow-orange-red segmentation confidence map and the existing yellow geometry
  circle/center. A one-pixel, half-opacity white dashed contour identifies the
  retained region; discarded blobs remain visible with their original confidence.
- Contours are projected from the final model-space mask onto the source crop,
  with dash spacing measured in display pixels. Only rendering changes; selection
  and measurements are unchanged. Geometry-only and ordinary segmentation views
  keep their existing styling. The selector deep-dive follow-up remains unchanged.
- Verification:
  - The 23 targeted overlay/head tests passed after creating the missing parent
    directory for the repo-local pytest temporary path. Coverage checks confidence
    color preservation, dashed gaps only on the retained contour, empty masks,
    coordinate projection, and unchanged analysis tables. Ruff and Black passed
    with `.pre-commit-cache/` excluded; `git diff --check` passed.
  - Ran the actual CLI on Purple trial 8 frame 03279 and visually inspected the
    raw-segmentation, geometry-only, and combined comparison. Its analysis CSV
    is byte-identical to the prior rendering. Local outputs are under
    `checkpoints_exp/all_heads_contour_2026-11-10/`.
  - The full 196-test suite passed with the existing deprecation warnings.

### Introduce the model design before its heads (Codex, GPT-6, effort and token budget not exposed)

- The maintainer requested placing the architecture overview immediately before
  the segmentation-to-velocity method. Revised its opening to establish the two
  prediction tasks and why geometry accompanies segmentation before introducing
  heads and their shared encoder. This lets readers understand the motivation
  without prior knowledge of the architecture. Existing follow-ups are unchanged.
- Following the maintainer's wording correction, separated the visible-pupil
  training target from the motivation for geometry supervision. Covered pupil
  reconstruction is outside the target; reducing detached false regions is an
  intended benefit, not a claim that these failures have been eliminated.
- Simplified the opening further at the maintainer's request: visible-pupil mask,
  observed detached regions on challenging frames, then geometry prediction as
  the response to that problem.
- The maintainer approved the wording and requested commit/push to dev before
  changing tasks. Delivery includes the architecture overview and its log entries;
  unrelated local documentation and experiments remain outside this commit.
- Verification:
  - Checked heading order, source links, and the retained Mermaid diagram.
    Verified content outside the moved section stayed byte-identical and ran
    `git diff --check`. Ruff and Black passed with the local pre-commit cache
    excluded. Documentation only; no runtime tests.
