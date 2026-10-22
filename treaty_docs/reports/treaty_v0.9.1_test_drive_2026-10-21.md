# Treaty v0.9.1 Test Drive

Date: 2026-10-21 (America/New_York).

The normal update path succeeded: preview and apply both changed only
`.copier-answers.yml` and `treaty_docs/treaty_conventions.md`, with the template pin
moving from v0.9.0 to v0.9.1. No merge conflicts or rejected patches occurred.
Project-specific `AGENTS.md`, `project_overview.md`, runtime code, and existing
planning content survived. After attribution review and maintainer authorization,
[issue #30](https://github.com/yzhaoinuw/agent_collab_treaty/issues/30) and a
[diagnostic follow-up on #15](https://github.com/yzhaoinuw/agent_collab_treaty/issues/15#issuecomment-6328064722)
were posted. No PR was opened.

Assessment after attribution review: the update mechanism passed this smoke test;
no v0.9.1 regression was demonstrated. One reproducible, pre-existing treaty bug
affects the drift report for non-root documentation layouts. The remaining
failures came from our environment or our document formatting, with optional
upstream improvements to their diagnostics. Published links and reporting material are in
[`treaty_v0.9.1_upstream_feedback_2026-10-21.md`](treaty_v0.9.1_upstream_feedback_2026-10-21.md).

## Attribution

| Observation | Responsibility | Disposition |
| --- | --- | --- |
| CLI 0.5.0 and missing Scripts PATH | Our environment setup | Upgraded CLI; used its absolute path. No release defect. |
| pip/temp/cache permissions and Git MSYS access denial | Our Windows sandbox | Working execution path succeeded. Traceback presentation is optional upstream polish. |
| Dirty tree required a stash; Git normalized mixed newlines | Our working state and local Git workflow | Existing edits restored. Neither is an updater defect. |
| Inline `- Verification: evidence` failed validation | Our entry did not use the required standalone subsection heading | Fixed locally. The same rule exists in CLI 0.5.0, 0.9.0, and 0.9.1; diagnostic clarity could improve. |
| Nested work-log/planning docs counted as risky drift | Treaty classification bug | Reproduced independently of the sandbox; present before v0.9.1 and in inspected upstream main. Posted as #30. |
| Other customized AGENTS/overview sections counted as drift | Expected conditional warnings | This release did not revise them. A drift count is not a prediction of actual conflicts. |
| Preview uses committed HEAD | Documented CLI behavior | Expected; our stash made preview and apply use the same baseline. |

## Setup And Procedure

- Repository: `pupil_tracking`, branch `dev`, starting HEAD
  `5c5e9c0d97a7112d39614d229e1104cb033798a5`.
- Windows 11 build 26200; Python 3.13.7 in the `pupil_tracking` conda environment;
  Copier 9.15.2. The installed treaty CLI was initially 0.5.0, despite the project
  already being pinned to template v0.9.0.
- Verified the latest GitHub release and PyPI package as v0.9.1, then upgraded the
  environment's CLI to 0.9.1. Its new `treaty --version` output correctly showed
  both CLI and installed template versions.
- Used the environment's `Scripts/treaty.exe` directly because that directory
  was absent from this PowerShell session's PATH.
- Captured the initial file hashes and planning/log originals in an ignored
  `.pytest_tmp_treaty_20261021/` directory. Temporarily stashed the existing five
  modified files and two untracked log archives, including untracked files.
- Ran `treaty diff .`, `treaty update . --dry-run`, and `treaty update .` in that
  order. The preview left the real working tree clean; the apply matched its
  two-file summary and preserved all other recorded answers.
- Restored the existing edits without conflicts and checked their contents
  against the safety stash. The unrelated README and Python edits remain local.

## Rough Edges

### Inline verification evidence is rejected

`treaty validate .` exited 1 for the existing October 6 entry:

```text
treaty_docs\work_log.md:7: work-log-missing-verification: Session entry should include a '- Verification:' subsection.
```

The entry already contained `- Verification: inspected the current ...`, followed
by its commands and results. In CLI 0.9.1, `validation.py`'s
`_is_verification_heading` accepts a standalone `- Verification:` line, but rejects
the same prefix when evidence follows the colon. Confirmed both cases directly
against the installed function. Follow-up inspection of the published 0.5.0,
0.9.0, and 0.9.1 wheels reproduced the same predicate behavior in all three.
This was our existing formatting problem, not a new validation regression.

Workaround applied here: put `- Verification:` on its own line and indent the
existing evidence below it, preserving the historical statements. Suggested
upstream improvement: accept inline evidence, or make the diagnostic explicitly
request a standalone heading and show an example.

### Non-root paths bypass the adopter-owned exemption

`treaty diff .` reported 30 sections across four files as conflict exposure. Among
the removed headings were `## YYYY-MM-DD`, the two `[Thread ...]` placeholders,
and `## What Looks Active vs. Legacy` (replaced here by a project-specific heading).
The conventions file showed eight untouched sections and no modifications.

Follow-up source inspection identified a narrower functional bug. `diff.py`
declares `work_log.md` and `next_steps.md` adopter-owned, but `compare_file`
compares their full relative path with those bare filenames. It therefore
recognizes `next_steps.md` but not `treaty_docs/next_steps.md` or
`docs/agents/next_steps.md`. Identical edits receive zero risk at the root and
nonzero risk under a folder. The intended "your content by design" notice is
also missing for the nested versions.

This explains seven incorrectly counted sections across our two project-owned
planning/history files. Applying the existing exemption would reduce the initial
headline from 30 sections/four files to 23 sections/two files. The other 23 are
conditional drift in AGENTS/overview content, not demonstrated merge conflicts.

The published `diff.py` files in CLI 0.5.0, 0.9.0, and 0.9.1 have identical hashes;
the path-dependent result was reproduced using all three. The same comparison
was also confirmed in upstream `main`. This is a pre-existing treaty reporting
bug exposed by our supported non-root layout, not a failure of the v0.9.1 merge.

### Windows sandbox failures produce long tracebacks

The initial pip upgrade failed with a temporary-file permission error and fell
back toward a user installation. Retrying the same pinned upgrade outside the
sandbox installed 0.9.1 in the intended conda environment.

For `treaty diff`, redirecting TEMP/TMP and COPIER_CACHE_DIR to the ignored local
scratch directory avoided protected temporary/cache locations, but Copier's Git
subprocess still failed at `git submodule update --checkout --init --recursive
--force`. Git's MSYS runtime reported
`NtCreateDirectoryObject(...): 0xC0000022`; the CLI emitted a long Python traceback
ending in `ProcessExecutionError` (exit code 66). Retrying outside the sandbox
succeeded. Preview and apply used that working execution path.

This is environment friction, not evidence of a treaty merge defect. Suggested
upstream improvement: retain the subprocess stderr and return status while
giving a concise top-level error, with a verbose option for the full traceback.

## Useful Boundaries

- The dry run previews committed HEAD, as its help explicitly states. Stashing
  local edits made the preview and apply use the same baseline, but means the
  preview did not exercise those uncommitted edits. Restoration was checked
  separately.
- Tested the standard noninteractive update path only. Interactive answers,
  `--data` overrides, layout relocation, and intentional merge conflicts were
  not exercised; no claim is made about those paths.
- The October 21 test left changes local on `dev`. The October 22 commit/push
  handoff is recorded separately in `../work_log.md`; no tag or release was requested.

## Final Verification

- `treaty --version`: CLI 0.9.1, template v0.9.1.
- `treaty validate .`: passed after reformatting the existing inline verification
  heading. No other validator finding remained.
- `git diff --check`: passed.
- Relative file-link audit of the root agent/overview docs and the updated treaty
  docs/report: all nine local file targets resolved. `treaty validate` also
  accepted the new Currently Hot anchor.
- Preservation audit: 193 pre-existing files outside the four intentionally
  changed treaty/planning files retained their content. Git normalized mixed
  line endings in `tests/test_overlays.py`; its text still matched the stash.
  Both originally untracked work-log archives retained their original byte hashes.
- No unmerged Git entries, conflict markers, or rejected/original patch files
  remained in the reviewed update paths.
- Runtime tests and package builds were not run: this update changes treaty
  metadata and documentation only. Existing uncommitted runtime changes belong
  to the earlier work and were preserved.
- Follow-up attribution checks reproduced the ownership and verification
  behavior against published 0.5.0, 0.9.0, and 0.9.1 sources, checked current
  upstream ownership logic, and reviewed the 21 issue reports returned by the
  repository search. See the linked reporting plan for overlap and priorities.
