# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

## 2026-08-01

### Update Agent Collab Treaty to v0.6.0 (Codex, GPT-5, xhigh reasoning, token budget not set)

- Confirmed the repository was already Copier-managed at `v0.3.2`: `.copier-answers.yml` was present and tracked, `main` was the recorded and GitHub-default integration branch, `treaty_conventions.md` was absent, and the managed orientation docs were heavily customized.
- Used a clean-tree preview before applying the update. It wrote nothing but failed to show the promised merge diff; the real apply then reported five answer migrations, two cleanly updated files, and conflicts in `AGENTS.md`, `project_overview.md`, and `work_log.md`.
- Resolved the three conflicts without losing the `pupil_tracking` environment, CLI, CI, model, checkpoint-packaging, active-runtime, or artifact-hygiene guidance. Kept every v0.6 managed heading, reduced `AGENTS.md` to 114 lines, added the upstream-managed `treaty_conventions.md`, and documented authored-vs-derived boundaries.
- Repaired one historical work-log heading whose model/version metadata was not recorded so the v0.6 validator could assess the log without inventing provenance.
- Reviewed upstream issues #8-#15 and posted the new dry-run preview-fidelity defect as https://github.com/yzhaoinuw/agent_collab_treaty/issues/18, signed `Codex (GPT-5)`.
- Committed the validated migration and fast-forwarded local `dev` and `main` to it; no remote push was requested.
- Verification:
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe --version`
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe diff .`
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .`
  - `rg -n "^(<<<<<<<|=======|>>>>>>>)" AGENTS.md project_overview.md work_log.md treaty_conventions.md`
  - `git diff --check`
  - `git diff --cached --check`
  - `git merge-base --is-ancestor chore/treaty-v0.6.0 dev`
  - `git merge-base --is-ancestor chore/treaty-v0.6.0 main`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m pytest -q` (`2 passed`)

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

### Package hygiene and initial agent docs (Codex, model version not recorded)

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
