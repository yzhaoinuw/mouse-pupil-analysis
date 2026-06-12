# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

## 2026-06-12

### Add citation and license metadata (Codex, GPT-5)

- Added an MIT `LICENSE`, `CITATION.cff` with Yue Zhao's ORCID, and README citation/license guidance so collaborators can cite the repository from GitHub.
- Bumped the package version to `0.1.4` in `pyproject.toml` for the next citable release tag.
- Added a `next_steps.md` follow-up for optional DOI archival through Zenodo or another release archive.
- Verification:
  - `git diff --check`
  - `rg -n "version =|license|license-files|Citation|License|CITATION|Version 0.1.4|doi|DOI" pyproject.toml README.md CITATION.cff LICENSE next_steps.md`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m build --wheel --sdist`
  - `$env:PATH='C:\Users\yzhao\miniconda3\envs\pupil_tracking\Scripts;' + $env:PATH; C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m pytest -q`

### Add treaty adoption badge (Codex, GPT-5)

- Added the official Agent Collab Treaty adoption badge to `README.md`, using the tri-color SVG asset and linking it back to the treaty repository.
- Verification:
  - `rg -n "Agent Collab Treaty|treaty-adopted|agent_collab_treaty" README.md`
  - No code tests were run because this was a README-only change.

### Migrate treaty docs from archive (Codex, GPT-5)

- Replaced placeholder treaty text in `AGENTS.md`, `project_overview.md`, and `next_steps.md` with repo-specific guidance migrated from `treaty_archive/` and checked against the current package, tests, and CI configuration.
- Preserved the `pupil_tracking` conda environment, direct environment-Python path, CLI entry points, packaged-checkpoint rule, 148 x 148 image convention, artifact hygiene notes, and branch/work-log discipline in active treaty docs.
- Migrated the older 2026-05-08 work history into this live log and removed the obsolete `treaty_archive/` directory after its unique content was represented in active docs.
- Verification:
  - `git ls-files | rg "(^|/)(AGENTS|PROJECT_OVERVIEW|WORK_LOG|project_overview|work_log|next_steps|treaty_archive)"` confirmed `AGENTS.md`, `project_overview.md`, and `work_log.md` are tracked under the active treaty names.
  - `rg -n "Thread A|Thread B|name of the conda|test command|app launch command|\[path/to|\[One|\[what|\[step|\[concrete|\[Same|\[Paused" AGENTS.md project_overview.md next_steps.md` returned no matches.
  - `Test-Path treaty_archive` returned `False`.
  - `git status --short --branch`
  - No code tests were run because this was a documentation-only migration.

## 2026-05-08

### Package hygiene and initial agent docs

- Created the original agent instructions to capture future Codex startup instructions, environment details, and commit-message preferences.
- Identified the active leading branch as `origin/dev` / `origin/ci-precommit`, then synced local `dev` and pushed updates so `dev` became the active leading branch.
- Added a developer collaboration guide, then split it into `PROJECT_OVERVIEW.md` so shared project context was separate from agent-only execution instructions.
- Confirmed the project miniconda environment is named `pupil_tracking` and documented the direct Python path for shells where `conda` is unavailable.
- Updated `.gitignore` to ignore generated analysis outputs, training artifacts, build outputs, egg-info folders, local sketch scripts, and local `.gitattributes`.
- Removed tracked cached bytecode from `tests/__pycache__`.
- Made checkpoint packaging explicit with `MANIFEST.in` and `pyproject.toml` package-data rules.
- Verified built wheel and source distribution include only the current packaged checkpoint and training log from `pupil_tracking/checkpoints/`.
- Removed the tracked `archive/` directory from the repository.
- Confirmed no `*sketch*.py` files remain tracked; local sketch files remain ignored.
- Updated `README.md` to remove the obsolete external checkpoint download step, note that the default checkpoint is bundled with installation, and document the optional `--checkpoint` override.
- Verification:
  - `ruff check .`
  - `black --check .`
  - `pytest -q`
  - package build inspection
