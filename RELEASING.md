# Releasing

This document covers the PyPI and Zenodo mechanics specific to this project. The
general commit/push/tag discipline lives in
[Release Gate](treaty_conventions.md#release-gate).

The repository and distribution are **`mouse-pupil-analysis`**. The only import is
`mouse_pupil_analysis`, and the established console commands remain unchanged.
Nothing here may ship a `pupil_tracking` module: the unrelated PyPI distribution
`pupil-tracking` already owns that import path, and two distributions writing one
file corrupt each other on install and uninstall. The release workflow enforces
this against the built artifacts.

## One-Time Setup (Completed 2026-08-12)

Both account actions are complete for `yzhaoinuw/mouse-pupil-analysis`. The details
below remain as the authoritative configuration record.

### 1. PyPI Trusted Publisher

Trusted Publishing lets the release workflow exchange its GitHub OIDC identity for
a short-lived PyPI token, so no API token is ever stored in repository secrets.

Because `mouse-pupil-analysis` does not exist on PyPI yet, its **pending** publisher
was registered at <https://pypi.org/manage/account/publishing/> with these values:

| Field | Value |
|---|---|
| PyPI project name | `mouse-pupil-analysis` |
| Owner | `yzhaoinuw` |
| Repository name | `mouse-pupil-analysis` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The environment name must match the `environment.name` in
`.github/workflows/release.yml`. Consider adding a required reviewer to the `pypi`
environment under repository Settings → Environments, so a release cannot publish
without an explicit approval click.

Test the whole path against TestPyPI first if you want a dry run; that requires a
separate pending publisher at <https://test.pypi.org> and a temporary `repository-url`
input on the publish step.

### 2. Zenodo Archiving

The GitHub integration was enabled on 2026-08-11 and reconfirmed after the repository
rename on 2026-08-12. Zenodo archives every subsequent **GitHub Release** (not every
tag).

Zenodo mints two kinds of DOI:

- A **concept DOI** that always resolves to the newest version. Cite this in
  `README.md` so the badge never goes stale.
- A **version DOI** unique to each release. Record this in `CITATION.cff` so a
  reader can retrieve the exact code a paper used.

## Per-Release Sequence

1. Choose the version. This project follows semantic versioning; the packaging
   rename from `pupil-tracking` to `mouse-pupil-analysis` is a breaking packaging
   change and should ship as a minor bump while below 1.0.

2. Update version metadata in both places:
   - `pyproject.toml` → `[project].version`
   - `CITATION.cff` → `version` and `date-released`

   `mouse_pupil_analysis.__version__` reads from installed distribution metadata, so it
   follows `pyproject.toml` automatically and needs no edit.

3. Promote `## [Unreleased]` in `CHANGELOG.md` to the new version with its release
   date, then add the comparison link at the bottom of the file.

4. Run the local verification described in [AGENTS.md](AGENTS.md#common-tasks):
   Ruff, Black, Pytest, and `python -m build`.

5. Merge the release commit into `main`, then tag it there and push. A PyPI version
   number can never be reused, so the workflow refuses to publish a commit that has not
   reached `main`. The tag must match the `pyproject.toml` version with a leading `v`;
   the workflow fails the build if they disagree.

   ```bash
   git switch main && git merge --ff-only dev && git push origin main
   git tag v0.2.0
   git push origin v0.2.0
   ```

6. The `Release` workflow builds both distributions and refuses to publish unless:

   - the tag matches `[project].version`;
   - `CITATION.cff` records the same version;
   - both artifacts contain the exact checkpoint that inference will select, plus its
     matching training log, and no stray or archived checkpoints;
   - the tagged commit is an ancestor of `origin/main`, so a published version always
     corresponds to code that reached the default branch;
   - the wheel installs and runs in a clean environment.

   Only then does it publish to PyPI. Because the checkpoint check resolves the single
   packaged `.pth` rather than accepting any `.pth`, changing the packaging policy to
   ship more than one checkpoint requires updating that step deliberately.

7. Create the GitHub Release for that tag. This is what triggers Zenodo; pushing the
   tag alone does not.

8. Copy the newly minted version DOI into `CITATION.cff` under `identifiers`
   (the block is present but commented out until the first DOI exists), and confirm
   the concept-DOI badge in `README.md`. Commit this as a follow-up; it necessarily
   lands after the tag, because the DOI does not exist until the release does.

## Verifying A Published Release

```bash
python -m venv /tmp/verify
/tmp/verify/bin/python -m pip install mouse-pupil-analysis
/tmp/verify/bin/python -c "import mouse_pupil_analysis; print(mouse_pupil_analysis.__version__)"
/tmp/verify/bin/run-pupil-analysis --help
```

A release cannot be overwritten on PyPI. If a broken artifact ships, yank the
release on PyPI and publish a patch version; do not attempt to re-upload.
