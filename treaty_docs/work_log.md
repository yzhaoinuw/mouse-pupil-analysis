# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

## 2026-11-11

### Clarify the README introduction and default heads (Codex, GPT-6, effort and token budget not exposed)

- The maintainer requested a brief opening covering diameter, center tracking,
  velocity, and quality checks, with implementation detail replaced by a How it
  works link. Usage now explicitly states that the preceding commands already
  use both packaged heads, so the inspection foldouts are clearly optional.
- Commit/push is requested. Keep dev and main synchronized for the public README;
  no new tag or package publication is needed for this documentation follow-up.
- Verification: checked the architecture anchor, Usage command context, Markdown
  structure, and scoped whitespace. Runtime is unchanged from the tested v0.4.1
  release; no additional runtime tests were needed. Unrelated local edits remain local.

### Publish v0.4.1 and record its archive DOI (Codex, GPT-6, effort and token budget not exposed)

- Published annotated tag `v0.4.1` at `92a759e17a6086da3ad60ff6f9119662cf5f3c38`
  after main CI passed. The tag-triggered release workflow built, verified and
  published both distributions to PyPI. Created the GitHub Release with those
  exact verified artifacts and user failure-report guidance.
- Zenodo archived the release as record 24148777, version DOI
  `10.5281/zenodo.24148777`, under the unchanged concept DOI
  `10.5281/zenodo.21897795`. Updated both citation DOI fields, kept the per-version
  identifier first, and updated the README citation in the required post-tag commit.
- Main and dev receive the citation follow-up; the release tag remains on the
  verified release commit. Unrelated local edits and experiments remain local.
- Verification:
  - Main CI run 43130970690 passed the supported Python/OS matrix and clean-wheel
    checks; Release run 43131494532 passed all build and PyPI publishing checks.
  - PyPI reports version 0.4.1. Both public distribution SHA-256 values match the
    downloaded GitHub Actions artifacts exactly. Installed v0.4.1 from PyPI in the
    smoke venv outside the source tree; verified version/import location, both
    model heads, 20% circle overlap, calibration 0.525, and diagnostic-only mean
    confidence. The published console help command passed.
  - Queried the published Zenodo record and verified its version and concept DOI.
    Checked matching citation/README version DOI and release date, scoped whitespace,
    remote tag target and branch refs. The local 211-test release suite passed.

### Prepare v0.4.1 for user feedback (Codex, GPT-6, effort and token budget not exposed)

- The maintainer declined thin-bridge cleanup: its limited return does not justify
  another postprocessing rule. Prioritize user failure cases and a systemic model
  improvement after targeted data collection. No opening, erosion, or splitting
  operation is enabled in this release.
- Requested a new tagged version containing all approved changes. Selected patch
  v0.4.1 to correct the v0.4.0 geometry packaging/inference issues and include the
  accepted 20% selector, inspection/overlay fixes, and mean-confidence QC removal.
  Trained weights and the calibrated pixel threshold 0.525 remain unchanged.
- Updated version/date metadata, changelog, user failure-report guidance, and
  pipeline/release documentation. Only approved changes and scoped documentation
  join the release; older local experiment files and notes remain unpublished.
  The release will reach main before tagging, then use the tag workflow for PyPI
  and a GitHub Release for Zenodo. The new version DOI is a post-release follow-up.
- Verification:
  - All 211 tests passed on a clean Git-tree export after a fresh-temp retry of
    the known Windows annotation-folder rename failure. Ruff and Black passed.
    Exported only staged files, excluding unrelated local experiments and notes.
  - Wheel and sdist built with the pupil_tracking environment, --no-isolation,
    and repository-local temporary storage. A shorter export path avoided the
    Windows path-length limit in source-distribution assembly.
  - Verified version 0.4.1 in both artifacts, byte-identical intended checkpoint,
    training log and calibration metadata, exactly one checkpoint, no archived
    weights, no unrelated namespace, and no mean-confidence rejection in runtime.
  - Installed the wheel into a fresh smoke venv using existing runtime dependencies
    and tested outside the source tree. Verified version/import location, both
    model heads, threshold 0.525, 20% overlap, and a low-confidence valid centroid.
    Both console help commands passed. The actual velocity/all-head CLI produced
    31 readable overlays and 30 finite speeds on the consecutive 97-Hz fixture.
  - Scoped whitespace checks passed. CI and publication checks follow the release
    commit; the tag will only be published after it reaches main.

### Remove mean-confidence rejection and inspect HQL090 centers (Codex, GPT-6, effort and token budget not exposed)

- The maintainer chose to trust the calibrated pixel threshold rather than add
  a fixed 0.90 mean-confidence rejection. Removed both that rejection and warning,
  retaining mean confidence as a continuous diagnostic and retaining probability
  weighting of the centroid. Circle-overlap selection, empty/border/circularity
  checks, and temporal rules are unchanged. Updated the README and pipeline docs.
- Audited 232 HQL090 PNGs across four recordings: none has a consecutive frame
  pair. No matching source video or acquisition metadata was found under videos.
  After clarification, the maintainer requested centers only on all 89
  `HQL090_sleep251012_010_eye` images, without requiring a velocity calculation.
  Kept recordings separate and did not invent a frame rate or bridge missing frames.
- Ran the public analysis API with the unchanged packaged checkpoint, threshold
  0.525, 20% circle overlap, and 0.35 confidence blend. Wrote center-marked overlays
  to `images_test_prediction/HQL090_centers_2026-11-11/HQL090_sleep251012_010_eye/`.
  The matching directory under `results/HQL090_centers_2026-11-11/` contains the
  standard analysis outputs, `estimated_centers.csv`, `center_diagnostics.csv`,
  a sparse-frame x/y scatter plot, and `run.json` with source/code/checkpoint hashes.
- Estimated centers for 87 frames; two masks are empty. Of 69 frames passing
  remaining QC, 30 would previously have failed mean confidence. Eighteen frames
  remain rejected for circularity; their raw center estimates are still exported
  and marked in yellow for inspection. These are inspection results, not an
  accuracy assessment. Changes are local on dev pending further review.
- Verification:
  - All 17 focused tracking/output tests passed, including a below-0.90 mask
    producing valid center/velocity and continued shape/border rejection. Full
    pytest ran 211 tests: 210 passed and the existing annotation-import test hit
    Windows WinError 5 during a temporary-folder rename. All four annotation-import
    tests passed on the fresh `.pytest_tmp_confidence_retry` path immediately after.
  - Repository-wide Ruff and Black passed with `.pre-commit-cache/` excluded.
    Matched all 89 output names to inputs, verified source/copied-input hashes and
    unchanged checkpoint hash, and matched run settings to the calibrated model.
    Visually inspected the center scatter plot and frame 24367's centroid overlay.
    Confirmed no low-confidence reason is generated and ran `git diff --check`.

### Integrate blob selection into dev (Codex, GPT-6, effort and token budget not exposed)

- The maintainer requested merging the accepted experiment into dev. Fetched
  origin and confirmed dev was an ancestor of `experiment/blob-selection`, then
  fast-forwarded dev through `486b07e` without conflicts. Temporarily stashed only
  the three modified tracked documents and restored each byte for byte; unrelated
  experiments and generated outputs remain local.
- Updated current pipeline references to describe the 20% selector as dev's
  default and removed the pending integration follow-up. This integration changes
  no model weights, prediction threshold, runtime code, or tests after acceptance.
- Verification: checked remote refs and fast-forward ancestry, matched the
  integrated runtime/test/script trees to the accepted branch, and checked scoped
  documentation and whitespace. The existing 209-test and four-CLI-test results
  apply to the unchanged runtime; no duplicate test run was needed. Push and
  remote-ref verification complete the dev handoff.

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
