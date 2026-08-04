# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

## 2026-07-28

### Implement pupil-center velocity tracking (Codex, GPT-5)

- Created `feature/pupil-velocity` from `dev` and carried the requirements/diagnostic notes onto the feature branch.
- Added opt-in `--calculate_velocity` analysis with a separate `--acquisition_fps` timebase, consecutive full-frame extraction, original source-frame metadata, and a compatibility-preserving diameter-only path.
- Added `pupil_tracking/tracking.py` for probability-weighted component centers, inverse resize/pad coordinate mapping, explainable segmentation quality flags, temporal area checks, and non-interpolated x/y displacement and velocity.
- Added a comprehensive tracking CSV, actual-time QC plot, and optional overlays with valid centers in green and rejected masks/centers in orange and yellow.
- Preserved the UNet architecture, packaged checkpoint, prediction threshold, training workflow, and legacy diameter CSV/plot outputs.
- Added synthetic tests for frame selection, coordinate mapping, confidence-weighted centers, component warnings, temporal area rejection, actual-time velocity, and invalid/non-consecutive gaps.
- Updated `README.md`, `project_overview.md`, and `next_steps.md` with the CLI contract, output behavior, implementation map, and validation evidence.
- Full validation of the supplied `eye.avi` produced 3,001 ordered rows from 0.00 to 90.00 seconds, 2,993 valid segmentations, eight explainable rejections, and 2,989 published frame-to-frame speeds. Visual inspection confirmed the largest speed peaks correspond to visible rapid pupil movement.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .`
  - `$env:PATH='C:\Users\yzhao\miniconda3\envs\pupil_tracking\Scripts;' + $env:PATH; C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m pytest -q` (`13 passed`; pytest cache write emitted one environment-permission warning)
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m build --wheel --sdist`
  - Wheel/source-distribution content inspection confirmed `pupil_tracking/tracking.py`, the packaged checkpoint, and its training log are included.
  - Full `eye.avi` velocity run with `--acquisition_fps 33.333333333333336 --batch_size 64`.
  - Focused mask-overlay run and rendered visual inspection around the eyelid closure.
  - `git diff --check`

### Clarify pupil-center velocity requirements and inspect sample video (Codex, GPT-5)

- Confirmed the collaborator wants per-frame pupil-center coordinates, horizontal and vertical displacement, and velocity calculated using actual elapsed time.
- Recorded the request for more frequent sampling and stricter filtering of poor pupil segmentations.
- Inspected the supplied `eye.avi`: it contains 3,001 frames, is encoded at 100 fps for approximately 30 seconds, and has burned-in timestamps spanning 0.0 to 90.0 seconds.
- Recorded the collaborator's confirmation that the burned-in timestamps represent actual experimental time, making the acquisition interval 0.03 seconds (approximately 33.3 samples/s).
- Ran the packaged checkpoint on 301 evenly spaced frames. All produced masks at the current 0.7 threshold, 58 contained multiple connected components, and an eyelid closure near the burned-in 3.3-3.9 second interval was confidently misidentified as pupil.
- Found that confidence-weighted centers alone did not eliminate the blink-related center outlier, supporting combined confidence, component, geometry, and temporal quality checks.
- Added a concrete implementation and validation thread to `next_steps.md`; no production code was changed.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"`
  - Inline `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -` OpenCV/Pillow inspection of video metadata and nine representative frames.
  - Inline `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -` PyTorch/OpenCV diagnostic using the packaged `iou=0.9158` checkpoint on 301 evenly spaced frames.
  - `git status --short --branch`

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
