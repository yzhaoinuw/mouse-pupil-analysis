# Next Steps

Use this checklist alongside `work_log.md`. Keep it concrete: only add work here when it is an actual follow-up, blocked thread, or decision that future agents should see before changing code.

## Currently Hot

- [DOI archival](#doi-archival) - optional next step after a GitHub release exists.

When a new thread starts, add a short bullet here with a link to its section below and the single next action.

## DOI Archival

Status: ready for user/account action

The repo now has MIT license metadata and `CITATION.cff`, so GitHub can display citation metadata for tagged releases. A DOI still requires linking the GitHub repository to an archival service such as Zenodo and creating or syncing a release there.

Remaining work:

- After the citable version tag is pushed, enable Zenodo or another archive for `yzhaoinuw/pupil_tracking`.
- Mint a DOI for the release and add it to `CITATION.cff` and `README.md`.

## Background / Paused

### Portable End-To-End Fixture

Status: paused

The current tests cover package import and CLI help. There is no small committed video/frame fixture for end-to-end inference.

Resume when the project needs stronger regression coverage for `run-pupil-analysis` outputs.

Remaining work:

- Decide whether a tiny synthetic or curated frame set can be committed without bloating the repo.
- Add a focused smoke test that exercises inference without relying on large local data folders.

### Local Artifact Cleanup

Status: paused

The working tree commonly contains generated image folders, prediction outputs, build outputs, cache folders, local sketch scripts, and experimental checkpoints. `.gitignore` covers the expected generated surfaces, but the local workspace may still be visually noisy.

Resume only when the user asks for repository cleanup or release preparation.

Remaining work:

- Inspect tracked vs. ignored files before deleting anything.
- Keep `pupil_tracking/checkpoints/` package data intact.

### Training Workflow Documentation

Status: paused

`README.md` includes maintainer notes for creating masks with Labelme and training with `run_train.py`, but training remains a local workflow rather than a packaged command.

Resume if the training path becomes user-facing or needs reproducible CI coverage.

Remaining work:

- Decide whether training should stay in `run_train.py` or move behind a package CLI.
- Document any required data layout, hyperparameter, and checkpoint naming contracts if the workflow is formalized.
