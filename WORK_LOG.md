# Work Log

## 2026-05-08

- Created `AGENTS.md` to capture future Codex startup instructions, environment details, and
  commit-message preferences.
- Identified the active leading branch as `origin/dev` / `origin/ci-precommit`, then synced
  local `dev` and pushed updates so `dev` became the active leading branch.
- Added a developer collaboration guide, then split it into `PROJECT_OVERVIEW.md` so shared
  project context is separate from agent-only execution instructions.
- Confirmed the project miniconda environment is named `pupil_tracking` and documented the
  direct Python path for shells where `conda` is unavailable.
- Updated `.gitignore` to ignore generated analysis outputs, training artifacts, build outputs,
  egg-info folders, local sketch scripts, and local `.gitattributes`.
- Removed tracked cached bytecode from `tests/__pycache__`.
- Made checkpoint packaging explicit with `MANIFEST.in` and `pyproject.toml` package-data rules.
- Verified built wheel and source distribution include only the current packaged checkpoint and
  training log from `pupil_tracking/checkpoints/`.
- Removed the tracked `archive/` directory from the repository.
- Confirmed no `*sketch*.py` files remain tracked; local sketch files remain ignored.
- Updated `README.md` to remove the obsolete external checkpoint download step, note that the
  default checkpoint is bundled with installation, and document the optional `--checkpoint`
  override.
- Ran verification where relevant: `ruff check .`, `black --check .`, `pytest -q`, and package
  build inspection.
