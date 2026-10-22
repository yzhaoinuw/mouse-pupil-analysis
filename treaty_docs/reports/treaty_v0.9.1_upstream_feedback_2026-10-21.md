# Treaty v0.9.1: Assessment And Reporting Plan

Date: 2026-10-21 (America/New_York). Published with the maintainer's authorization:

- Bug: [issue #30](https://github.com/yzhaoinuw/agent_collab_treaty/issues/30).
- Diagnostic follow-up:
  [comment on #15](https://github.com/yzhaoinuw/agent_collab_treaty/issues/15#issuecomment-6328064722).

Both posts were read back and matched the prepared text after normalizing GitHub's
line endings. No PR was created; the sandbox traceback suggestion remains deferred.

The v0.9.0 -> v0.9.1 template update passed the standard noninteractive smoke
test. Preview and apply matched, answers were retained, customized content
survived, and final validation passed after a local formatting correction.
There is no demonstrated v0.9.1 regression or content-loss defect in this run.
This is evidence for that path only, not a full release certification.

## Reporting Order

| Priority | Candidate | Recommendation |
| --- | --- | --- |
| Medium; confirmed bug | `treaty diff` loses the adopter-owned exemption under a non-root docs_dir | Posted as #30; report material below. |
| Low; usability | Verification diagnostic does not explain its standalone-heading requirement | Posted a diagnostic-only follow-up to #15, which already records historical formatting friction. |
| Low; usability | Copier/Git failure prints a long traceback | Defer a standalone report. The cause was our sandbox, and normal execution worked. |
| No issue | Old CLI, PATH, permissions, dirty tree, newline normalization, committed-state preview | Local setup or documented behavior; exclude from the bug queue. |

Reviewed the 21 issue reports returned by a repository-wide issue search, including
[issue #15](https://github.com/yzhaoinuw/agent_collab_treaty/issues/15).
That open issue already requests clearer drift risk classes and mentions repairs
to historical verification subsections. The specific folder-dependent ownership
bug below was not described in the returned reports. Its narrow reproducer and
acceptance criteria distinguish it from the broader risk-classification request.

## Posted Bug: Non-Root Adopter-Owned Docs Misclassified By treaty diff

Suggested title: `treaty diff counts adopter-owned docs as conflict exposure when docs_dir is non-root`

Type: bug. Suggested priority: medium; misleading preflight output, no observed
merge failure or data loss. Versions reproduced: CLI 0.5.0, 0.9.0, and 0.9.1.
The same logic remains in inspected upstream main.

### Problem

`diff.py` explicitly excludes `work_log.md` and `next_steps.md` from conflict
exposure because their content is adopter-owned. That exemption only works for
root-level paths. A supported layout such as `docs_dir: treaty_docs` passes
`treaty_docs/next_steps.md` to `compare_file`, which fails the filename membership
test. Normal project-owned content then receives removed/modified-heading
warnings and inflates the conflict-exposure summary.

### Minimal Reproduction

Run with agent-collab-treaty 0.9.1 installed:

```python
from agent_collab_treaty.diff import compare_file

pristine = "## Example\n\nTemplate content\n"
local = "## Example\n\nProject content\n"

for name in ("next_steps.md", "work_log.md"):
    for path in (name, f"treaty_docs/{name}", f"docs/agents/{name}"):
        result = compare_file(path, pristine, local)
        print(path, result.adopter_owned, result.at_risk)
```

Actual results for each filename:

```text
<name>                  True  0
treaty_docs/<name>       False 1
docs/agents/<name>       False 1
```

Expected: all three paths are recognized as the same adopter-owned document
role, with `adopter_owned=True` and `at_risk=0`. Their detailed reports should
retain the "your content by design" notice.

### Cause And Real-Adopter Effect

[`compare_file` in v0.9.1](https://github.com/yzhaoinuw/agent_collab_treaty/blob/v0.9.1/src/agent_collab_treaty/diff.py)
uses `adopter_owned=path in ADOPTER_OWNED`, while `compare_trees` supplies a full
relative path. `ADOPTER_OWNED` contains only the two bare filenames.

In the Windows `pupil_tracking` adopter (`docs_dir: treaty_docs`), the initial
report counted 30 sections across four files. Seven sections belonged to these
two project-owned documents. Respecting the existing exemption would give 23
sections across two files for the same content. The actual update was clean;
this report concerns classification, not an incorrectly performed merge.

### Suggested Fix And Acceptance Criteria

Normalize the document role relative to the configured docs directory before
classifying ownership; using the basename is also sufficient if the caller
guarantees it is classifying only rendered canonical treaty documents. Preserve
the full relative path for display and file access.

- Cover both canonical filenames in root, one-folder, and multi-folder layouts.
- Assert zero risk and the expected-customization notice in each layout.
- Keep modified maintained guidance, such as `treaty_conventions.md`, in the risk
  count; avoid broadly suppressing nested paths.
- Exercise `compare_trees`/formatted output so the caller's relative paths are
  covered, not just direct root-level unit inputs.

Related: [issue #15](https://github.com/yzhaoinuw/agent_collab_treaty/issues/15)
requests broader risk-classification improvements. This is a concrete path
normalization bug in the existing exemption, so it can be fixed independently.

## Posted Follow-Up: Explain The Required Verification Syntax

Our previous entry used `- Verification: inspected ...`; the validator requires
the standalone heading `- Verification:`. We corrected our document. The same
predicate rejects inline evidence in CLI 0.5.0, 0.9.0, and 0.9.1, so this should
not be reported as a new update failure or a missing-evidence parsing regression.

The follow-up on issue #15 suggests a diagnostic such as:

> Use a standalone '- Verification:' line, then indent the evidence below it.

Accepting inline evidence is an optional policy change, not required to fix our
entry. A focused diagnostic example would resolve the confusion without changing
the validator's contract. Keep this separate from the confirmed ownership bug.

## Defer: Friendlier Subprocess Errors

The Windows MSYS access denial was caused by our restricted execution context;
retrying outside it succeeded. An upstream improvement could summarize the failed
Git command and stderr with an optional verbose traceback. This run supports a
usability suggestion only. Do not file it as "v0.9.1 fails on Windows" or ask the
project to bypass sandbox permissions.

## Evidence And Limits

- Compared the published PyPI wheels for 0.5.0, 0.9.0, and 0.9.1 without installing
  older versions. All three `diff.py` source files were byte-identical; executed
  the two-filename ownership reproducer against each version.
- Extracted and executed `_is_verification_heading` from each published version;
  standalone accepted, inline evidence rejected in all three.
- Inspected current upstream `src/agent_collab_treaty/diff.py` through GitHub;
  its full-path membership check remains. No upstream fix was attempted here.
- No interactive-answer, `--data`, relocation, or intentional-conflict tests were
  added to this assessment. Issue #30 and the linked #15 follow-up were submitted;
  no PR was created.
