# Guidelines and Tips for Agents

Read this file first when joining this repository. It is the project-specific quick reference; generic treaty mechanics live in [`treaty_conventions.md`](treaty_conventions.md).

Keep this file lean (aim for under 150 lines). Put detailed procedures in the document that owns them and link from here.

## Startup Rule

At the beginning of a new chat or agent session, read this file first. Do not automatically read every Markdown file; use the [Documentation](#documentation) map to select what the task needs.

## Runtime Environment

Use the local miniconda environment named `pupil_tracking` on every platform:

```powershell
conda activate pupil_tracking
```

Windows is the primary development platform; a macOS environment is also in use. Check which one you are on before copying any absolute path below.

**Windows.** Environments live under `C:\Users\yzhao\miniconda3\envs\`. If `conda` is unavailable, run the environment Python directly:

```powershell
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe
```

For environment console scripts in plain PowerShell, prepend `C:\Users\yzhao\miniconda3\envs\pupil_tracking\Scripts` to `PATH`.

**macOS.** Environments live under `/Users/yuezhao/miniconda3/envs/`. The equivalent direct interpreter and script directory are:

```bash
/Users/yuezhao/miniconda3/envs/pupil_tracking/bin/python
/Users/yuezhao/miniconda3/envs/pupil_tracking/bin   # prepend to PATH for console scripts
```

If the environment does not exist yet, create it with `conda create -n pupil_tracking python=3.12` followed by `pip install -e ".[dev]"`.

## Common Tasks

Install for development:

```powershell
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m pip install -e .[dev]
```

Run analysis or frame extraction:

```powershell
run-pupil-analysis --video_path C:\path\to\movie.avi
run-pupil-analysis --image_dir C:\path\to\frames
extract-frames --video_path C:\path\to\movie.avi --out_dir C:\path\to\frames
```

Run the CI-equivalent checks:

```powershell
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .
$env:PATH='C:\Users\yzhao\miniconda3\envs\pupil_tracking\Scripts;' + $env:PATH; C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m pytest -q
```

For packaging changes, also run:

```powershell
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m build --wheel --sdist
```

On macOS or Linux the same checks are:

```bash
ruff check .
black --check .
pytest
python -m build --wheel --sdist
```

Note that reinstalling under the current distribution name after an environment previously held the old `pupil-tracking` distribution requires `pip uninstall pupil-tracking` first, then `pip install -e ".[dev]"`. Both distributions declare the same console-script names, so uninstalling either one removes `run-pupil-analysis` and `extract-frames` until the remaining one is reinstalled.

Before committing, confirm Ruff, Black, and Pytest are clean; package builds still contain the tracked checkpoint and training log when relevant; and `work_log.md` records the verification actually run.

## When To Update Treaty Docs

At the end of a substantive session, update `work_log.md` and align `next_steps.md` unless the user asks not to document it or the exchange was clearly trivial. Record decisions and reusable evidence, not a prose copy of the diff. See [Work Log Discipline](treaty_conventions.md#work-log-discipline).

## Branch Handoff Discipline

Prefer `dev` for ordinary development unless the user requests another branch; `main` is the GitHub default and recorded treaty integration branch. Before switching away from experimental or feature work, confirm its intended changes are complete, committed or intentionally local, verified, and merged/pushed/parked as requested. See [Branch Handoff](treaty_conventions.md#branch-handoff).

## Release / Tag Checklist

Treat commit + push + tag, "cut a release," or "publish version X" as a release. Clear the documentation/version/verification gate before tagging, then verify remote refs. See [Release Gate](treaty_conventions.md#release-gate).

For the PyPI and Zenodo mechanics specific to this project — version metadata locations, the tag/citation/checkpoint checks enforced by `.github/workflows/release.yml`, and the DOI follow-up commit — see [`RELEASING.md`](RELEASING.md).

## Updating The Treaty

Only update the treaty when the user asks. Use the stable treaty CLI to run `treaty diff`, preview with `treaty update --dry-run`, apply with `treaty update`, resolve any conflicts, and validate. See [Updating The Treaty](treaty_conventions.md#updating-the-treaty).

## Documentation

Read only what the task needs:

- `treaty_conventions.md`: upstream-maintained logging, branch, release, and update procedures; prefer not to edit it.
- `project_overview.md`: active runtime, structure, tests, and authored-vs-derived boundaries for unfamiliar areas.
- `next_steps.md`: unfinished work; "Currently Hot" identifies active threads.
- `work_log.md` and `work_log_archive/`: recent decisions and verification evidence; read the two latest dates when history matters.
- `README.md`: user-facing installation, usage, packaging, and I/O expectations.
- `RELEASING.md`: PyPI Trusted Publishing setup, Zenodo archiving, and the per-release sequence.
- `CHANGELOG.md`: user-facing change history; update the `Unreleased` section as features land.
- `.github/workflows/ci.yml`: lint, format, test, and build expectations.
- `.github/workflows/release.yml`: tag-triggered build, metadata consistency checks, and PyPI publication.

## Commit Message Guidelines

Use a short title. Add flat body bullets only when a commit contains multiple requested changes. Describe high-level behavior, not implementation details; omit tests/docs/internal work from feature messages unless that work is the commit's purpose.

## Git Ownership Note

If Git reports dubious ownership, use command-scoped `safe.directory` when practical. If a persistent exception is needed:

```powershell
git config --global --add safe.directory C:/Users/yzhao/python_projects/pupil_tracking
```

## Pre-commit Note

If pre-commit cannot write its default cache, use the repo-local cache:

```powershell
$env:PRE_COMMIT_HOME = "C:\Users\yzhao\python_projects\pupil_tracking\.pre-commit-cache"
C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m pre_commit run --all-files
```

## Project-Specific Reminders

- Keep changes scoped: package code, training scripts, local experiments, generated results, and build outputs share this repo.
- Do not add generated artifacts unless explicitly requested. Check `project_overview.md` before editing unfamiliar or generated-looking paths.
- Use package imports such as `mouse_pupil_analysis.preprocessing` and
  `mouse_pupil_analysis.unet` in new code. `pupil_tracking` exists only for compatibility.
- Preserve the 148 x 148 centered/padded image convention unless intentionally changing model assumptions.
- Default inference selects the packaged checkpoint with the highest IoU encoded in its filename.
- `mouse_pupil_analysis/checkpoints/` is installed package data;
  `mouse_pupil_analysis/checkpoints/archive/` is excluded. Verify wheel and
  source-distribution contents after packaging changes.
