# Agent Instructions

## Startup

- At the beginning of a new chat for this project, check the project folder for Markdown
  files that may contain instructions, project overview notes, or work logs.
- Read `PROJECT_OVERVIEW.md`, `WORK_LOG.md`, and `README.md` before making broad project
  changes.
- The miniconda environment for this project is `pupil_tracking`.
- All miniconda environments are under `C:\Users\yzhao\miniconda3\envs\`.
- If `conda` is not on PATH, run Python directly with:

```powershell
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe
```

## Execution Preferences

- Prefer working on `dev` unless the user asks for another branch.
- Check `git status --short --branch` before edits, before commits, and after pushes.
- This repository may contain ignored local data, build outputs, and sketch files. Do not
  add generated artifacts unless the user explicitly asks.
- Keep changes scoped and avoid unrelated refactors.
- Use package imports such as `pupil_tracking.dataset` and `pupil_tracking.unet` in new code.
- Preserve the 148 x 148 padded image convention unless intentionally changing model assumptions.
- For packaging changes, verify that the packaged checkpoint remains included in both the
  wheel and source distribution.

## Useful Commands

```powershell
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .
$env:PATH='C:\Users\yzhao\miniconda3\envs\pupil_tracking\Scripts;' + $env:PATH; C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m pytest -q
```

Build verification:

```powershell
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m build --wheel --sdist
```

## Commit Message Preferences

Use a short title line. Add a short body with flat bullets when a commit contains multiple
requested changes. Commit message bullets should describe high-level added or changed
behavior, not implementation details. For feature commits, mention only user-facing behavior
unless tests, docs, project memory updates, or other internal work are the main purpose of
the commit.
