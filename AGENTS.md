# Guidelines and Tips for Agents

This is the first file any agent should read when joining this repository. It names the runtime, the active code paths, the documentation map, and the project-specific rules that prevent accidental churn.

## Startup Rule

At the beginning of a new chat or agent session, read this file first. Do not automatically read every Markdown file in the repository. Use the [Documentation](#documentation) map below to choose only the files that matter for the current task.

## Runtime Environment

Use the local miniconda environment named `pupil_tracking`.

Typical startup:

```powershell
conda activate pupil_tracking
```

All miniconda environments are under:

```powershell
C:\Users\yzhao\miniconda3\envs\
```

If `conda` is not on PATH, run the environment Python directly:

```powershell
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe
```

For commands that need console scripts from the environment in a plain PowerShell session, prepend the Scripts folder:

```powershell
$env:PATH='C:\Users\yzhao\miniconda3\envs\pupil_tracking\Scripts;' + $env:PATH
```

## Common Tasks

Install for development:

```powershell
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m pip install -e .[dev]
```

Run the main analysis pipeline:

```powershell
run-pupil-analysis --video_path C:\path\to\movie.avi
```

Run analysis from an existing PNG frame folder:

```powershell
run-pupil-analysis --image_dir C:\path\to\frames
```

Extract frames only:

```powershell
extract-frames --video_path C:\path\to\movie.avi --out_dir C:\path\to\frames
```

Run the CI-equivalent local checks:

```powershell
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .
$env:PATH='C:\Users\yzhao\miniconda3\envs\pupil_tracking\Scripts;' + $env:PATH; C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m pytest -q
```

Build verification:

```powershell
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m build --wheel --sdist
```

Pre-flight checklist before committing:

- `ruff check .` is clean.
- `black --check .` is clean.
- `pytest -q` is green.
- Packaging changes still include the packaged checkpoint and training log in the wheel/source distribution.
- A new entry has been prepended to `work_log.md` with the verification commands that were actually run.

## When To Update Treaty Docs

At the end of any substantive work session, update `work_log.md` unless the user explicitly asks not to document it, says it is off the book, or the exchange was clearly trivial.

A session is substantive when it includes file edits, meaningful validation/debugging/profiling, a technical decision or reversal, reusable evidence, branch/PR/release/deployment state changes, or unfinished follow-up that belongs in `next_steps.md`.

When a session creates or changes future work, update `next_steps.md` in the same pass: add concrete follow-ups, remove completed items, and keep "Currently Hot" accurate.

## Branch Handoff Discipline

Prefer working on `dev` unless the user asks for another branch. Before switching away from an experimental or feature branch, fully resolve the work on that branch. Confirm whether the branch contains all intended changes, whether those changes are committed, and whether the user expects them merged, pushed, or intentionally left parked.

Useful checks before switching or merging:

```powershell
git status --short --branch
git log --oneline --left-right --cherry-pick main...HEAD
git merge-base --is-ancestor main HEAD
```

## Documentation

Read these documents only as needed:

- `project_overview.md`
  - Use when onboarding to the codebase structure or editing unfamiliar areas.
  - Read "What Looks Active vs. Legacy" before editing; this repo mixes package code, scripts, local data folders, and generated outputs.

- `work_log.md` and `work_log_archive/`
  - Use when the task needs recent implementation history, experiment outcomes, or verification breadcrumbs.
  - The live `work_log.md` holds at most the 5 most recent unique calendar dates.
  - Find date anchors with:
    `rg -n '^## [0-9]{4}-[0-9]{2}-[0-9]{2}' work_log.md`

- `next_steps.md`
  - Use when planning or continuing unfinished work.
  - The "Currently Hot" pointer at the top names active threads.

- `README.md`
  - Use when changing user-facing setup, packaging, usage, CLI behavior, or input/output expectations.

- `.github/workflows/ci.yml`
  - Use when changing test, lint, formatting, build, or packaging expectations.

## Git Ownership Note

If Git reports a "detected dubious ownership" warning for this repo, mark this repository as safe:

```powershell
git config --global --add safe.directory C:/Users/yzhao/python_projects/pupil_tracking
```

## Pre-commit Note

If pre-commit cannot write to its default cache location, use a repo-local cache before running it:

```powershell
$env:PRE_COMMIT_HOME = "C:\Users\yzhao\python_projects\pupil_tracking\.pre-commit-cache"
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m pre_commit run --all-files
```

## Commit Message Guidelines

Commit messages should use a short title line. Add a short body with flat bullets when a commit contains multiple requested changes. Commit message bullets should describe high-level added or changed behavior, not implementation details.

For feature commits, mention only user-facing behavior unless tests, docs, project memory updates, or other internal work are the main purpose of the commit.

## Project-Specific Reminders

- Keep changes scoped. This repo contains installable package code, training scripts, local experiment folders, generated masks/results, and build outputs.
- Do not add generated artifacts unless the user explicitly asks.
- Use package imports such as `pupil_tracking.dataset` and `pupil_tracking.unet` in new code.
- Preserve the 148 x 148 centered/padded image convention unless intentionally changing model assumptions.
- The default inference path chooses the packaged checkpoint with the highest IoU encoded in its filename.
- The tracked checkpoint under `pupil_tracking/checkpoints/` is part of the installed package. Checkpoint archive files under `pupil_tracking/checkpoints/archive/` are excluded from package data.
- For packaging changes, verify that the packaged checkpoint remains included in both wheel and source distribution outputs.
