# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

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
