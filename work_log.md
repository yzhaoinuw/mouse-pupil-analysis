# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

## 2026-08-09

### Stream inference into optional tracking and overlay consumers (Codex, GPT-5)

- Replaced the tracking-aware inference branch with a streaming `PupilPrediction` record. Each record carries one frame's probability map, already-thresholded binary mask, diameter, and source metadata and is consumed before the next record is retained.
- Added `TrackingAccumulator` in `tracking.py` and `MaskOverlayAccumulator` in `pupil_predictions.py`. The CLI now constructs those consumers only when their flags are enabled; `generate_pupil_predictions(...)` no longer accepts `calculate_velocity` or `acquisition_fps`. The tracking module now also has an editable Spyder run block that leaves the detailed tracking DataFrame available for inspection.
- Reused the streamed binary mask during component measurement, preserving one UNet pass without repeating thresholding or accumulating all float probability maps. Kept the diameter-only convenience function, Spyder entry block, and original mask-prediction import path. Anchored both Spyder examples to the repository root instead of a machine-specific path or current working directory.
- Ran the packaged checkpoint through a two-image CLI smoke test in diameter-only and velocity modes. Both modes returned identical diameters; velocity mode returned valid rows at the expected 0.0 and 0.1 second timestamps.
- Verification:
  - Focused tests: `14 passed` across `tests/test_imports.py`, `tests/test_tracking.py`, and `tests/test_overlays.py`.
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .` (`19 files` unchanged; Black reported an inaccessible user-cache warning but completed successfully).
  - `C:\Users\yzhao\miniconda3\Scripts\conda.exe run -n pupil_tracking pytest -q --basetemp .pytest_tmp_streaming_refactor_full` (`23 passed`).
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m build --wheel --sdist`; the wheel and source distribution contain the refactored prediction/tracking modules, packaged checkpoint, and training log.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .`
  - `git diff --check`

## 2026-08-08

### Extract reusable pupil-prediction module (Codex, GPT-5)

- Moved packaged-checkpoint discovery, PNG frame discovery, UNet inference, optional tracking measurements, and confidence-heatmap overlay generation out of `run_pupil_analysis.py` into the sibling `pupil_tracking/pupil_predictions.py` module.
- Kept `run_pupil_analysis.py` as the CLI orchestrator and retained the original `generate_pupil_mask_prediction` import path as a compatibility re-export.
- Added a Spyder-friendly `if __name__ == "__main__":` block with editable local paths and inference settings. It runs `generate_pupil_predictions(...)` directly and writes example overlays without invoking the CLI parser.
- Updated focused tests to import frame discovery and overlay helpers from their owning module, and added coverage for the new module boundary and legacy prediction import.
- Verification:
  - Focused tests: `7 passed` across `tests/test_imports.py`, `tests/test_outputs.py`, and `tests/test_overlays.py`.
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .` (`19 files` unchanged; Black reported an inaccessible user-cache warning but completed successfully).
  - `C:\Users\yzhao\miniconda3\Scripts\conda.exe run -n pupil_tracking pytest -q --basetemp .pytest_tmp_prediction_refactor_full` (`22 passed`).
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m build --wheel --sdist`; the wheel and source distribution contain `pupil_predictions.py`, the packaged checkpoint, and its training log.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .`
  - `git diff --check`

### Keep otherwise usable temporal area outliers (Codex, GPT-5)

- Changed `abrupt_area_change` from a hard rejection to a diagnostic warning when the existing per-frame confidence, circularity, border, and nonempty-mask checks pass. Warning rows retain their measured centers and eligible frame-to-frame speeds; prior hard failures remain invalid and are not restored.
- Kept the README unchanged because it already links to the detailed method. Updated `project_overview.md` as the methodology source of truth and aligned `next_steps.md` with the warning semantics.
- Re-ran all 3,001 frames of `videos/eye.avi` at the confirmed 33.3333 Hz acquisition rate. The result contains 2,982 valid rows, 13 warnings, six invalid rows, and 2,990 published speeds. Frames 112-113 are now `abrupt_area_change` warnings; frames 111 and 114 remain hard-invalid.
- Regenerated and promoted the matched README demo with its established 850 x 520 layout, 90 frames, 5 fps cadence, confidence heatmap, and center marker. The former four-frame temporal-area run now remains continuous in the center and speed plots, and the unused rejected-estimate legend is omitted.
- Verification:
  - Focused tests: `13 passed` across `tests/test_tracking.py`, `tests/test_outputs.py`, and `tests/test_make_gif.py`.
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .` (`18 files` unchanged; Black reported an inaccessible user-cache warning but completed successfully).
  - `C:\Users\yzhao\miniconda3\Scripts\conda.exe run -n pupil_tracking pytest -q --basetemp .pytest_tmp_area_warning_full_20260808` (`20 passed`).
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .`
  - `git diff --check`
  - Pillow metadata confirmed 850 x 520, 90 frames, and 200 ms per frame; SHA-256 comparison confirmed the promoted GIF exactly matches the inspected candidate.

## 2026-08-07

### Restore portable collection of the GIF helper test (Codex, GPT-5)

- Reproduced GitHub Actions' `ModuleNotFoundError: No module named 'make_gif'` locally with the console-script form `pytest -q`; the earlier `python -m pytest` verification had implicitly placed the repository root on Python's import path.
- Updated `tests/test_make_gif.py` to load the repository-root `make_gif.py` from an explicit path derived from the test file, so collection no longer depends on the launcher's import-path behavior.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\Scripts\pytest.exe -q tests\test_make_gif.py` (`1 passed`).
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\Scripts\ruff.exe check .`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\Scripts\black.exe --check .`
  - `C:\Users\yzhao\miniconda3\Scripts\conda.exe run -n pupil_tracking pytest -q --basetemp .pytest_tmp_ci_import_conda_20260807` (`19 passed`).
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .`
  - `git diff --check`

### Show QC-rejected estimates in the README demo (Codex, GPT-5)

- Diagnosed the brief center and speed gap in the main-cadence demo. The four affected frames (7,209, 7,212, 7,215, and 7,218) retained usable raw center candidates and were rejected for `abrupt_area_change`, not low center confidence.
- Extended `make_gif.py` to recognize optional internal diagnostic columns. Accepted measurements remain solid; rejected raw center estimates and speed intervals touching rejected frames are shown as dashed traces, with one accepted endpoint on either side to make the transition legible.
- Kept the compact exported analysis contract unchanged: invalid published center and speed values remain blank. The dashed bridge is a README-demo diagnostic only and is not interpolation or a replacement measurement.
- Regenerated and promoted `pupil_diameter_analysis_result_demo.gif` with the established main cadence: 850 x 520, 90 frames, 200 ms per frame (5 fps), and 18 seconds total.
- Verification:
  - Focused unit coverage for retaining only the rejected diagnostic run and its neighboring endpoints.
  - Rendered inspection immediately before, during, and after the four rejected frames.
  - SHA-256 comparison confirmed that the promoted GIF exactly matches the inspected candidate.
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .`
  - `C:\Users\yzhao\miniconda3\Scripts\conda.exe run -n pupil_tracking python -m pytest -q --basetemp .pytest_tmp_dashed_gif_20260807` (`19 passed`).
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .`
  - `git diff --check`

### Match the current demo to main's motion cadence (Codex, GPT-5)

- Fetched `origin/main` and inspected both its historical `make_gif.py` and public README animation. Confirmed the main GIF contains 90 frames at 5 fps (200 ms per frame, 18 seconds total).
- Reproduced the exact historical row selection: diameter-filtered rows beginning at index 7,100, every third row, for 90 frames. The selected source-frame range is 7,107-7,375.
- Identified the cause of the jerkier committed candidate: `images_test_1` is already sparse, usually skipping 97 source frames, so applying the historical every-third-item sampling made most displayed jumps span 291 source frames.
- The original dense raw sequence is no longer stored locally. For demo reconstruction only, recovered the 90 eye crops from the public main GIF, approximately removed the old translucent red mask, resized them back to 148 x 148, and reran the current CUDA pipeline to obtain new confidence heatmaps and centers.
- Calculated interval-averaged speed from valid reconstructed centers and suffix-derived timestamps. The reconstructed run produced 86 valid and four invalid segmentations; invalid-frame gaps remain visible in the center and speed traces.
- Generated `videos/gif_candidates/pupil_demo_main_matched.gif` with the approved aligned layout. It contains 90 frames at 5 fps, spans source frames 7,107-7,375, is approximately 5.02 MB, and visually matches main's motion cadence much more closely.
- After user approval, promoted the main-cadence-matched candidate to `pupil_diameter_analysis_result_demo.gif` and verified its SHA-256 matches the inspected candidate exactly. The earlier sparse candidate remains local under `videos/` for comparison.
- Verification:
  - Downloaded main's public GIF and confirmed 1,152 x 576, 90 frames, 200 ms per frame, and 18,000 ms total duration.
  - Current CUDA inference completed on all 90 reconstructed frames.
  - Pillow metadata confirmed the matched candidate is 850 x 520, 90 frames, 200 ms per frame, and 18,000 ms total duration.
  - Rendered inspection of the opening, midpoint, and closing candidate frames.
  - SHA-256 comparison of the promoted repository GIF and inspected candidate.

### Promote the selected old-images README demo (Codex, GPT-5)

- Promoted the approved `images_test_1` candidate to `pupil_diameter_analysis_result_demo.gif` after the user selected it as the more interesting README animation.
- Verified the promoted repository GIF and inspected candidate have identical SHA-256 hashes. The README already references the repository-local filename.
- Kept both comparison candidates and their generated inputs under local untracked `videos/`; they are intentionally excluded from the commit.
- Recorded that the user plans to edit README on GitHub after the branch push, so the next local session should synchronize the remote README change before further edits.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .`
  - `C:\Users\yzhao\miniconda3\Scripts\conda.exe run -n pupil_tracking python -m pytest -q --basetemp .pytest_tmp_delivery_20260807` (`18 passed`).
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m build --wheel --sdist`; wheel and source distribution retained the packaged checkpoint and training log.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .`
  - `git diff --check`

### Generate two approved-layout GIF candidates (Codex, GPT-5)

- Preserved the approved square eye panel, continuous 0.7-1.0 confidence legend, aligned/narrowed trace panel, and simplified user-facing axis labels.
- Generated `videos/gif_candidates/pupil_demo_eye_frames_500_1200.gif` from displayed frames 500-1,200 at every seventh frame. The candidate contains 101 frames at 10 fps, is 850 x 520, and is approximately 4.03 MB.
- Re-ran all 945 images in `images_test_1/` through the current CUDA pipeline to create current confidence overlays and pupil centers. The run produced 943 valid, one warning, and one invalid segmentation.
- The `images_test_1` files are sparse source samples, usually 97 source frames apart. For the demo only, calculated interval-averaged speed from suffix-derived timestamps and valid centers; the production analysis CSV remained unchanged.
- Reused the old helper's `sample_every=3`, 5 fps, and 90-frame cap to generate `videos/gif_candidates/pupil_demo_images_test_1.gif`. It spans source frames 97-28,615, is 850 x 520, and is approximately 4.71 MB.
- Visually inspected the opening, midpoint, and closing states of both animations. Kept both candidates and left the current repository/README GIF unchanged for direct comparison.
- Verification:
  - Current CUDA inference on 945 `images_test_1` images completed successfully and wrote 945 confidence overlays.
  - Pillow metadata confirmed 101 frames at 100 ms per frame for the frames-500-1,200 candidate.
  - Pillow metadata confirmed 90 frames at 200 ms per frame for the `images_test_1` candidate.
  - Rendered contact-sheet inspection of the beginning, midpoint, and end of both GIFs.
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check make_gif.py`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check make_gif.py`
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .`
  - `git diff --check`

### Expand the README demo across the pupil-size transition (Codex, GPT-5)

- Expanded the default GIF window from displayed frames 680-820 to frames 200-2,100 so the demo visibly progresses from the early small pupil to the much larger pupil later in the recording.
- Changed the default sampling interval from every second frame to every twentieth frame. The resulting 96-frame animation remains smooth enough for the README, lasts 9.6 seconds at 10 fps, and is approximately 6.28 MB.
- Preserved the major frame-740 movement event in the sampled sequence while showing diameter growth from roughly 16 to 40 model pixels.
- Regenerated `pupil_diameter_analysis_result_demo.gif` at 1000 x 520 and visually inspected displayed frames 200, 740, 1,500, and 2,100. The changing pupil size, confidence heatmap, center marker, and live diameter/center/speed traces were legible throughout.
- Simplified the next layout sample to show only `Frame N` above the image, `Frame` on the x-axis, and plain `Diameter (pixel)`, `Center (pixel)`, and `Speed (pixel/s)` y-axis labels. Removed status and implementation-coordinate wording from the visible figure.
- Added a compact confidence-heatmap/pupil-center legend below the image and rendered a frame-1,500 sample for review. The repository GIF was intentionally left unchanged pending approval of this sample.
- Replaced the single-color heatmap swatch with a continuous yellow-orange-red confidence scale labeled from the 0.7 prediction threshold to 1.0. Added a fixed GIF palette that preserves the continuum, translucent colored mask, grayscale eye detail, and center marker during GIF quantization.
- Narrowed only the right-side trace panel and added dedicated vertical spacing around its three plots. The resulting 850 x 520 sample aligns the right title and x-axis label with the left frame title and confidence legend while preserving the square eye image and making changes in the traces visually steeper.
- Updated `next_steps.md` to require approval of the cleaned-label sample before regenerating the long-span demo.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe make_gif.py` (saved 96 GIF frames covering displayed frames 200-2,100).
  - Pillow metadata check confirmed a 1000 x 520 GIF with 96 frames at 100 ms per frame.
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .`
  - `C:\Users\yzhao\miniconda3\Scripts\conda.exe run -n pupil_tracking python -m pytest -q --basetemp .pytest_tmp_gif_span_20260807` (`18 passed`).
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .`
  - `git diff --check`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check make_gif.py`
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check make_gif.py`
  - Rendered inspection of the cleaned-label frame-1,500 sample.
  - Regenerated the frame-1,500 sample through the GIF writer and confirmed that the confidence bar retained 47 distinct rendered colors instead of collapsing to a few color blocks.
  - Rendered inspection confirmed that the narrowed right plot block and unchanged left image block share the same top and bottom visual boundaries.

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
