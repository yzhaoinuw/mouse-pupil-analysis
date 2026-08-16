# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

## 2026-08-16

### Measure generalisation, and repair the selector that was corrupting it (Claude, Opus 5)

- **Ran the first grouped cross-validation sweep.** It returned mean per-session IoU 0.5378, but
  the number was contaminated: three of four folds selected a checkpoint at **epoch 4-6 of 400**.
  Under `balanced_iou`, a size bin holding one to three validation images carries a third of the
  metric, and the tiny bin spikes (fold 0: min 0.0233, max 0.6793, last-10-epoch mean 0.0707).
- **The damage was not confined to selection.** The same metric drove `ReduceLROnPlateau` on
  `mode="max"`, so the epoch-6 spike became a high-water mark that no later epoch cleared and the
  learning rate decayed 1e-3 -> 6.25e-5 by epoch 43, against a `min_lr` of 3.125e-5. Folds were
  mis-selected *and* under-trained. Validation loss over the same run fell cleanly to a minimum at
  epoch 31 -- the smooth signal found the right region while the noisy one did not.
- **Three fixes**, in `run_train.py`, forwarded by `run_cv.py`: the scheduler now runs on
  `val_loss`; `--selection-metric` chooses what "best" means and also ranks threshold candidates;
  `--selection-threshold` (default 0.5) compares epochs at a fixed threshold rather than each
  epoch's maximum over 11. Selection default left at `balanced_iou` so earlier runs reproduce.
- **Generalisation baseline: 0.6245 +/- 0.0322** mean per-session IoU (natural sampling, seeds
  0/1/2), against the 0.8749 published from the leaky split. Transfer is bimodal -- six sessions
  below 0.45, six above 0.75.
- **Natural sampling beat size-balanced by 0.0354**, winning at every seed and in 8 of 12 matched
  (fold, seed) cells. Equal-mass balancing does not deliver what it was added for: the tiny bin
  splits 2-2 and its largest gap favours natural, while balancing costs large-pupil IoU 0.20 and
  0.27 on folds 1 and 2. `balance_training_sizes` still defaults to `True`; changing a training
  default was left to the maintainer.
- **A predicted failure mode did not appear.** Fold 3, which holds out the session carrying 10 of
  14 tiny masks and was the predicted worst case for balancing, is the only fold where balancing
  consistently wins. The cost appears in folds with ordinary tiny counts instead.
- **Seed noise on the grouped split is ~4x the documented floor** (sd 0.0273 on the three-seed
  mean, up to 0.0873 on one fold, against +/-0.0069 measured on the leaky split). Corrected the
  stale +/-0.02 guidance in `run_cv.py`'s docstring.
- **Threshold calibration was censored** at the grid edge in 5 of 12 balanced and 4 of 12 natural
  folds. Fixing it needed no retraining -- selection ran at a fixed 0.5, so the grid only affects
  the threshold reported afterwards. Re-calibrating all 24 saved checkpoints over 0.05-0.95 took
  ~2 minutes and moved 7 of 24 thresholds, for +0.0082 balanced and +0.0056 natural. No conclusion
  changed.
- **Corrected a factual error made during the session**, on the user's challenge: the packaged
  checkpoint trained on **166** images, not all 222. The 222-image pool is the union of the old
  166-image train and 56-image validation folders. The CV-leakage conclusion survives on the
  stronger ground that its own validation set drew 54 of 56 images from recordings that also fed
  training, so at session granularity it has seen every session.
- **No release candidate was produced, and none could be.** These are fold models trained on three
  quarters of the pool. The gated whole-pool path is blocked: `run_train.py --final` raises because
  `splits.json` sets `n_holdout_sessions: 0`.
- Verification:
  - `ruff check .`, `black --check training/`, `pytest` (`40 passed` with the environment's `bin`
    on `PATH`; `test_cli_help.py` fails without it for the pre-existing console-script reason,
    confirmed by reproducing it on a stashed tree).
  - Smoke-ran one fold at 2-3 epochs before each long sweep, and confirmed the new options are
    recorded in `best.json` via `_jsonable_config`.
  - Full detail and tables in `reports/2026-08-16-selection-metric-repair.md`.

## 2026-08-14

### Record provenance at intake and stratify the folds (Claude, Opus 5)

- **Renamed `labeled_data/` to `labeled_frames/`** to pair with `unlabeled_frames/`: both
  hold frames, and the distinction is whether masks exist. No fold moved -- keys are
  `<session>/<filename>` and never included the pool root, which is the same property that
  made the earlier merge and restructure free.
- **Dropped the 32 labelme `.json` from the fixture.** I had added them; the original
  12-pair fixture had none. Audited what reads them: `labelme_json2png.py` skips every one
  because the masks exist, `provenance.labelme_session` finds empty `flags`, no test opens
  them, and `labelme_export_json` is not installed so a regeneration test would skip
  everywhere. 128 KB of files nothing uses. Tracked `sample_data` is 2.3 MB.
- **Restored those frames as `sample_data/unlabeled_frames/`, and untracked
  `sample_data/folds/`.** The rename is the substantive part: "raw" distinguished nothing
  once `labeled_data/` was also raw frames. What actually separates these six is that they
  carry **no masks**, which is the whole reason the promotion gate can use them --
  `hard_frame_check.py` aimed at `labeled_data/` would pass unconditionally, since those
  are training frames. Default restored, gate works out of the box. `folds/` is derived
  and deterministic, so committing it added 1.6 MB of duplicate bytes that can drift from
  `splits.json`; its test now *regenerates* the folders and checks them against the
  committed manifest, which tests the claim rather than the artefact. Tracked
  `sample_data` is 2.5 MB.
- **Removed `sample_data/raw_frames/` at the maintainer's direction.** Checked the two
  things they asked me to check first: there is **no cropping script anywhere in the
  codebase** (the only `.crop()` calls are inside `random_zoom_translate_pil` and
  `random_pad_and_crop_pil`, both random augmentation), so `labeled_data/` images are to
  be treated as raw images; and `raw_frames` had **zero stem overlap** with `labeled_data`
  (5-digit `_00000` vs 4-digit `_0000` numbering), so it was not a duplicate. Its two
  real dependents were repointed at `labeled_data` session folders rather than deleted:
  the end-to-end inference test and the input-pixel rescale test. `hard_frame_check.py`
  lost its bundled default and now requires `--frames`; **the promotion gate has no data
  of its own any more**, and pointing it at `labeled_data` would be meaningless since
  those are training frames.
- One assertion had to move with it: the inference test bounded model diameter at
  `MODEL_IMAGE_SIZE / 2`, calibrated for the wider raw frames. On `labeled_data` frames a
  dilated pupil legitimately reaches 110 of 148, so the bound is now 0.95 of the frame,
  which still catches "segmented everything".
- **The session is now a directory: `labeled_data/<session>/images|masks`.** The
  maintainer's observation that `labeled_data` + `labeled_masks` split one thing across two
  top-level names was right, and the deeper win is that this makes provenance *structural* --
  an image cannot enter the pool without landing in a session folder, so the grouping can
  never disagree with where the file sits, and the sidecar stops being something to maintain.
  **The whole split now regenerates from the layout alone**: deleting `splits.json` and
  `provenance.csv` and rebuilding gives byte-identical folds. Keys gained their session
  prefix (`<session>/<filename>`), migrated deliberately with folds carried across and
  verified unchanged for all 222 images.
- **Retired `provenance.csv` from both pools.** It is now redundant with the directory
  structure *and* actively risky: it outranks the folder, so a stale sidecar would silently
  override moving an image to a different session folder. The sidecar mechanism stays for
  batches that arrive pre-mixed, and that hazard is documented.
- **`labelme_json2png.py` was broken and I had missed it.** It still wrote to
  `masks_validation/`, deleted by the pool merge, and selected its target through an edited
  `dataset_type` module variable. Rewritten to walk session folders with `--data-root` and
  `--session`. Worth noting the pattern: the merge commit updated every *reader* of the pool
  and missed the one *writer*.
- **Mirrored the layout into `sample_data/` and expanded it 12 -> 32 real pairs.** The
  fixture now has the same shape as the maintained pool: flat `labeled_data/`, sidecar,
  `splits.json`, and a committed `folds/cv1..cv4`. Selection was the point, not the count:
  the old fixture had **no mask below 15 px**, so the tiny size bin was undefined in every
  fold, and half its sessions held one image, which satisfies "no session spans a fold"
  vacuously. The new 32 span 8.8-109.7 px across 10 sessions with several 5-6 deep, and a
  1-epoch run on fold 0 now populates all three size bins. Cannot expand `raw_frames`:
  the original recording directories are not on this machine -- **blocked pending a path
  from the maintainer**.
- **Two bugs found by round-tripping the fixture.** The recorded `source` degraded to
  `frozen` on every regeneration, so a manifest differed from itself on a no-op rerun and
  the "no recorded provenance" census warning fired once and then silently retired forever.
  The session still freezes; the source now reports the live sources, and a batch fallback
  no longer counts as contradicting a recorded session. Also updated `test_sample_data.py`
  and `test_training_pairing.py`, which still asserted the train/validation folders --
  worth noting they *caught* the layout change rather than being collateral.
- **Built `--materialize`, which writes the folds to disk as `folds/cv1..cvN`.** The
  maintainer asked for this in their first message; I argued against it (the folders are
  gitignored, so folds-on-disk would not survive a fresh clone), offered `--materialize` as
  the resolution in the same reply, and then never built it. Dropped twice over the session
  along with the `labeled_data/` merge. The objection was never a reason not to build it --
  it is a reason the *manifest* stays the record, which the one-way derivation preserves.
  **Check dropped offers against the original ask before declaring a build finished.**
- **Merged the four pool folders into one flat `labeled_data/` + `labeled_masks/`.** The
  maintainer proposed this at the start of the session; I agreed and then dropped it when
  summarising the design, and it went unbuilt for the whole build. Caught on review. The
  merge moved 222 image/mask pairs plus their labelme JSON, copy-verify-delete against
  sha256, and **no fold moved** -- an image is keyed by its path *within* its pool folder,
  so `images_train/X.png` and `labeled_data/X.png` are both key `X`. `DEFAULT_POOL` still
  lists the legacy pairs after `labeled_data`, so an older checkout and `sample_data` keep
  working with no change.
- **Two bugs the merge surfaced**, both now fixed: regenerating without `--folds` used the
  default 5 rather than the manifest's own 4 and silently wrote an empty fold; and an empty
  fold could reach the manifest at all. `--folds` now defaults to the existing manifest's
  count, and a fold that would hold no images raises and names `--reassign`.

- **The maintainer challenged filename-derived grouping: what happens when incoming images
  follow a different convention, or none?** Fair, and the answer was not a better parser.
  Tested whether session identity can be recovered from the data instead of a name, against
  the 222 images whose sessions were already known. All three methods fail in the *dangerous*
  direction — tearing one session across clusters, which is exactly the leak grouping exists
  to prevent:

  | method | result |
  | --- | --- |
  | crop geometry (exact) | 6 of 16 sessions span 2-3 crop boxes |
  | masked-thumbnail correlation + connected components | no threshold gives usable groups; chains to 219/222 in one blob |
  | agglomerative clustering (average/complete linkage) on preprocessed frames | 3 sessions torn at k=5, 6 at k=10 |
  | file mtime | uniform, destroyed by copying |

  Cause: the images are 150-283px crops around an eye, so the cage, headplate and rig framing
  that would fingerprint a recording were cropped away before anything sees them.
  **Do not retry this without new evidence.** Probes are in the session scratchpad; the
  numbers are summarised in `training/data_collection.md`.
- **Grouping is worth 0.25 IoU and stratifying does not substitute for it.** Nearest-neighbour
  mask transfer: 0.652 IoU when the neighbour comes from the same session, 0.399 from a
  different one, against a 0.02 seed noise floor. The control settles what drives it — matching
  on closest *pupil diameter* from another session recovers only 0.434, so the advantage is the
  setting, not the pupil. Also refuted the hypothesis that `resize_with_pad` normalises sessions
  away: after the real preprocessing, within-session image correlation is 0.741 against 0.192
  between, wider than on the raw crops.
- **Session identity is now recorded, never derived** (`training/provenance.py`). Precedence:
  frozen manifest > `provenance.csv` sidecar > labelme `flags.session` > intake subfolder >
  single batch fallback. Over-merging costs data efficiency; tearing leaks — so the fallback
  collapses everything unresolved into one group, and a pool with no provenance at all fails
  loudly rather than splitting badly. Once an image is in the manifest its session and fold are
  frozen; a source that later disagrees raises instead of silently repacking, which was a real
  hole in the previous design (a `parse_identity` change would have shifted every session key
  and repacked the whole pool while printing only "0 carried over").
- **Stratified the folds on pupil size and lighting**, which the maintainer pushed for and the
  data backed. The first grouped split left 3 of 5 folds with no small pupil and a 3.03x spread
  in median diameter — so fold-to-fold variance was mostly a story about which size regime
  landed where. Banding sessions by median diameter and median background brightness and packing
  new sessions into the fold thinnest in their bands gives 1.78x and 4 of 5 folds with tiny
  masks. Bands are kept *separate* rather than crossed: crossing them fragments the small-pupil
  sessions into different combined strata, they stop repelling each other, and they pile back
  into two folds.
- **The maintainer's packing suggestion beat mine and is now the rule.** I proposed a size
  "guard" (a tier penalising oversized folds); tested, it moved three images and cost a fold
  its full band coverage — dead end. The maintainer's framing was simpler: send new data to
  the emptiest fold. Tested three rules against both regimes, and the winner is *coverage-
  first* — only the **absence** of a diameter band outranks fold size:

  | rule | migration: med_d spread / tiny / all-3-bands | incremental (10 new): median / worst size spread |
  | --- | --- | --- |
  | band count throughout | 1.74x / 4-of-5 / 5-of-5 | 1.33x / 2.05x |
  | size throughout | 4.51x / 2-of-5 / 2-of-5 | 1.15x / 1.25x |
  | absence-of-band, then size | 1.74x / 4-of-5 / 5-of-5 | 1.15x / 1.25x |

  Strictly dominant, no trade-off. Coverage fires only while folds are empty, which is
  precisely when stratification is needed; once populated, size leads. Also worth recording:
  **fold-size imbalance is self-correcting** — the current 1.91x falls to ~1.35x after five
  new sessions under any rule. The residual is the indivisible 62-image session, which no
  packing rule can touch. Regenerating produced a byte-identical assignment, so nothing moved.
- **Brightness is measured outside the mask on the original image**, not the padded model input.
  `resize_with_pad` fills with black, so a padded mean would encode the crop's aspect ratio as
  if it were lighting. Session means span 71-157 with within-session spread of 1-11.
- **Added a holdout gate** (`--holdout` / `run_train.py --final`): sessions in no fold, trained
  on never. Corrected the maintainer's framing from holding out *mice* to holding out
  *conditions* — animal identity is the axis this project already established does not matter.
  None is set by default: two sessions is 15-27% of a 222-image pool, and that is the
  maintainer's call to make.
- `reports/scripts/dataset_census.py` now reads sessions from the manifest instead of importing
  the deleted `parse_identity`. It reports at session level only; its animal and cohort
  breakdowns were themselves filename-derived. It confirms the legacy fixed-folder split was
  **100% leaked** — all 56 validation images share a session with training.

### Build recording-grouped splits and cross-validation (Claude, Opus 5)

- **Chose the session, not the recording file or the animal, as the grouping unit.** The
  maintainer's argument that identity barely matters here is supported by the data: every
  animal appears in exactly one cohort and mostly one condition, so animal-grouping is
  largely redundant with session-grouping and only costs training data. Dropped the
  "hold animals out permanently for publication" follow-up as not serving this project's goal.
- **Recording files are too fine a unit, which the earlier plan had wrong.**
  `HQL086_whiskerb250923_{002,005,008}` are three files from one sitting; splitting them
  across the boundary leaks the same setting under another name. Six of 25 recording groups
  are same-day siblings like this. Collapsing on animal+date+condition gives **16 sessions,
  not 25** — the real count of independent settings behind 222 images.
- **Two measurements that change how results must be read.** One session (`5003` dim-light,
  62 images) is 28% of the pool and cannot be subdivided, so fold 0 is 62 images against
  ~40 for the others and a single fixed holdout is untenable. And `HQL080_sleep250625` holds
  10 of the 14 tiny masks in the entire dataset — "small-pupil performance" has always meant
  "performance on that one session", a sharper statement of the `balanced_iou` fragility
  already recorded. Three of five folds contain no tiny mask at all, so `balanced_iou`
  averages a different set of bins per fold and is not comparable across them; `run_cv.py`
  prints which bins each fold actually scored.
- **Fold assignment is deterministic and stable under additions.** Largest-first bin packing,
  no seed. Regenerating the manifest keeps existing sessions on their folds and packs only
  new ones, so a cross-validation number stays comparable as data arrives; `--reassign`
  repacks and is opt-in. This is the property that makes the manifest worth preserving —
  it carries the history, and `splits.json` is committed for that reason.
- Merged `images_train/` and `images_validation/` **logically, in the manifest**, rather than
  physically. Both are gitignored and are the only copy of the labelled data, so moving files
  to re-split was not worth the risk; the manifest stores data-root-relative paths instead.
- Deduplicated `parse_identity`: `reports/scripts/dataset_census.py` now loads it from
  `training/data_splits.py` so the census and the fold assignment cannot disagree. Confirmed
  the census still reproduces the published report exactly (54/56, 96%, no validation-only animals).
- Verification:
  - `ruff check .`, `black --check .` clean; `pytest -q` 100 passed, including 14 new tests in
    `tests/test_data_splits.py` covering the grouping rules, fold disjointness, determinism,
    and the stability-under-addition guarantee.
  - `python training/data_splits.py --data-root . --show` on the real 222-image pool: 16
    sessions, folds of 62/40/39/41/40 images.
  - End-to-end smoke: `run_train.py --split-manifest splits.json --fold 3 --epochs 2` trains
    and writes metadata; `run_cv.py --folds 2 3 --epochs 1` produces the per-fold and
    per-session tables and the summary JSON.
  - No cross-validation results are recorded yet — the runs above were 1-2 epoch plumbing
    checks, not measurements.

### Prune next_steps.md back to unfinished work (Claude, Opus 5)

- `next_steps.md` had grown to 474 lines and stopped matching its own governing rule in
  `treaty_conventions.md` ("remove completed items, and keep Currently Hot accurate"). Cut to
  219 lines with no real follow-up lost: 23 action bullets before, 21 after, and the only
  genuine deletion was a stale "sync the README from GitHub first" instruction from a session
  whose README has since been rewritten twice.
- Deleted the 228-line Pupil-Center Velocity design specification, 48% of the file. It was
  written in the future imperative ("Add `--calculate_velocity`", "Create a focused
  `tracking.py`") for a feature shipped in 0.2.0, and `project_overview.md` already documents
  the implemented method and its actual QC constants. That section is now a status line, a
  pointer to `project_overview.md#segmentation-to-velocity-method`, and the two follow-ups
  that are genuinely open.
- Fixed a structural bug: three threads listed under "Currently Hot" were physically filed as
  `###` children of "Background / Paused". The fine-tuning thread was already misfiled; the
  2026-08-14 session added two more beside it without noticing. All three are now `##` siblings.
- Compressed the six completed threads to a status line plus whatever genuinely remains, and
  dropped them from "Currently Hot". Kept the settled decisions that stop future re-litigation:
  the `pupil_tracking` namespace measurement, and the `CITATION.cff` identifier-ordering finding.
- Merged the duplicate "Sample Data For Examples And Regression Tests" and "Portable End-To-End
  Fixture" sections, which covered the same fixture.
- Removed machine-specific residue: the `C:\Users\yzhao\Desktop\eye.avi` timebase contract and a
  PowerShell verification block duplicating `AGENTS.md`. The timebase principle it encoded is
  already in `project_overview.md`.
- Added `.DS_Store` and `Thumbs.db` to `.gitignore` under a new OS-metadata section.
- Verification:
  - All 5 internal anchors in the rewritten file resolve to real headings; the one cross-file
    anchor (`project_overview.md#segmentation-to-velocity-method`) exists at line 100 and is the
    same anchor `README.md` links.
  - Confirmed no other tracked file links into a removed `next_steps.md` anchor.
  - Diffed action bullets mechanically between the old and new file to enumerate what was lost.

### Measure the seed noise floor and audit the promotion (Claude, Opus 5)

- Ran ten training runs on the full 166/56 dataset, five per arm, identical except the seed.
  Scratch averages 0.8745 +/- 0.0092 balanced and 0.8749 +/- 0.0079 macro IoU; fine-tuning
  from the packaged weights averages 0.8694 +/- 0.0023 and 0.8724 +/- 0.0063. The packaged
  checkpoint's macro IoU equals the five-run scratch mean to four decimals and its balanced
  IoU sits at their 20th percentile, so the +0.0112 margin that justified its promotion is
  1.6 sd of seed noise. Fine-tune runs peaked at epochs 1-26, confirming that fine-tuning on
  the data the weights were already fitted to is close to a no-op.
- Promoted the best scratch run (`full_scratch_s3`, balanced 0.8825) as authorized, then
  **reverted** it: it loses the small pupil in `sample_data/raw_frames/recording_250616`
  entirely, as do the two other highest-scoring scratch runs. All five fine-tuned runs and the
  packaged checkpoint detect it at 11.3-12.0 model px. Validation IoU is anti-correlated with
  this real-frame correctness, so the packaged checkpoint is retained and the branch ships no
  weights change.
- Measured the split: 54 of 56 validation images come from a recording that also supplies
  training images and there are no validation-only animals, so no reported number measures
  generalization. Recorded the re-split plan in `next_steps.md`.
- Established that low tiny-bin IoU is a metric artefact, not a small-pupil weakness. Implied
  boundary error is 2.4 px for tiny masks against 5.4 px for medium, so the model is roughly
  twice as accurate on small pupils; IoU merely penalises them mechanically. The tiny bin also
  holds only 2 validation masks while carrying a third of `balanced_iou`.
- Added `reports/` with the write-up and five scripts that regenerate its numbers, added
  `training/promote_checkpoint.py` so a promotion is reproducible from a run folder rather
  than hand-assembled, and added Apple MPS support to the trainer (4.6x faster than CPU here).
- Tested and rejected a proposal to calibrate the threshold on diameter error instead of IoU:
  bootstrap sd of the located threshold is 0.085 either way, and the two criteria nearly
  coincide for convex masks. Diameter bias is worth reporting as a diagnostic, not as the
  selection criterion. Recorded as parked in `next_steps.md`.
- Verification:
  - `conda run -n pupil_tracking pytest` passed the full suite including 9 new promotion
    tests; the direct-interpreter run failed only the known console-launcher PATH test.
  - `ruff check .` and `black --check .` passed.
  - A promotion test caught a real defect: `Path(...).name` does not split Windows paths on
    POSIX, so promoting a Windows-trained run from macOS or CI would have leaked
    `C:\Users\...` into published package data. Fixed with a separator-agnostic split.
  - Round-tripped the trainer and promotion script end to end on the public fixture, producing
    packaged artifacts whose schema matches the shipped metadata exactly.

### Integrate the fine-tuned model into dev and main (Codex, GPT-5)

- Fast-forwarded the accepted `feature/finetune` work into `dev` and then `main`, preserving
  the pre-existing untracked `videos/` and `.pytest_tmp_full_console_fix_20260812/` directories.
- Explicitly parked the next version tag and GitHub/PyPI release in `next_steps.md`; this
  integration creates no tag and starts no release workflow.
- Verification:
  - Refreshed the remote state and confirmed `dev` was an ancestor of `feature/finetune` and
    `main` was an ancestor of `dev`, allowing fast-forward-only integration.
  - Verified the final local, tracking, and remote `dev`/`main` refs resolve to the same
    integration commit.
  - The integrated implementation retains the previously recorded 77-test, lint, script-smoke,
    checkpoint-hash, and package-content verification; the documentation-only handoff commit
    passed the configured pre-commit hooks.

### Promote the fine-tuned model and add script arguments (Codex, GPT-5, high reasoning)

- Replaced the packaged weights and log with the authorized overall leader from
  `ft_natural_lr1e-4_s0`, and added matching calibrated-threshold JSON metadata. The promoted
  checkpoint's SHA-256 exactly matches the locally evaluated candidate.
- Chose the concise packaged stem `166pupils_thresh=0.4_iou=0.8749`. Dataset size,
  calibrated threshold, and macro per-image IoU are immediately useful selection facts;
  seed, learning rate, sampling, best epoch, balanced IoU, and other details remain in the
  log and JSON. `unet` is invariant, attention is detected from the weights, and Git history
  shows `resize` marked preprocessing/augmentation that is now universal to the 148 x 148
  model contract, so those tags are also omitted.
- Added terminal argument parsing directly to `training/run_train.py`. Supplying arguments
  enables terminal use, while running with no arguments retains the editable Spyder/IDE
  configuration. No new installed console command or wheel training module is added.
- Extended release validation to require the checkpoint's matching JSON metadata.
- Verification:
  - Confirmed the promoted checkpoint SHA-256 is
    `CA0EDA577F33FD458E302B3787E17CC6588D894E6F8048F1DDFBDC41B207D0C7`, exactly matching
    the evaluated `ft_natural_lr1e-4_s0/best.pth`; inference detects attention and resolves
    threshold 0.4 from the renamed package triplet.
  - `python training/run_train.py --help` passed, and a one-epoch fine-tuning smoke through
    that exact script invocation loaded the promoted checkpoint and wrote weights, JSON, and
    a full log. Refreshing the editable install removed the briefly introduced console script;
    only `run-pupil-analysis` and `extract-frames` remain installed.
  - Focused checkpoint/training/script/metadata/real-image/end-to-end coverage passed all 31
    tests. The full Conda-environment suite passed all 77 tests.
  - `ruff check .`, `black --check .`, and `git diff --check` passed. Black emitted only its
    known non-fatal user-cache permission warning.
  - A clean wheel and source distribution contained exactly
    `166pupils_thresh=0.4_iou=0.8749.{pth,json}` and its matching log, with no old or long-name
    checkpoint. The wheel contains no training module or new console command; the source
    distribution retains `training/run_train.py`. The namespace verifier passed both.
  - Treaty 0.5.0 remains unable to validate the repository's newer relocated
    `treaty_docs/` layout, as documented in the preceding session entry.

## 2026-08-13

### Benchmark fine-tuning candidates (Codex, GPT-5, high reasoning)

- Removed all 21 obsolete loose checkpoint files from the ignored `checkpoints_exp/`
  directory as authorized; they were untracked local artifacts and are not recoverable from
  Git. Replaced the flat naming scheme with collision-safe
  `checkpoints_exp/<run-name>/{best.pth,best.json,train.log}` folders and a local comparison
  summary.
- Benchmarked the shipped checkpoint on the maintained 56-image validation set. Its packaged
  threshold of 0.7 scored 0.8288 balanced IoU; calibrating it fairly on the same threshold
  grid selected 0.45 and raised balanced IoU to 0.8578, with 0.8618 macro, 0.7773 tiny,
  0.8421 medium, and 0.9541 large IoU.
- Ran three lower-rate fine-tunes from the shipped weights. Natural sampling at `1e-4`, seed
  0 was the overall leader at epoch 25 and threshold 0.40: 0.8690 balanced IoU and 0.8749
  macro IoU, gains of 0.0112 and 0.0131 over the calibrated shipped model. Its size-bin IoUs
  were 0.7954 tiny, 0.8587 medium, and 0.9529 large.
- The size-balanced `1e-4`, seed 0 run scored 0.8669 balanced and 0.7995 tiny IoU. The
  size-balanced `5e-5`, seed 1 run scored 0.8660 balanced and the best tiny IoU, 0.8051.
  This ablation supports keeping the sampler as an option for tiny-pupil emphasis rather than
  enabling it unconditionally.
- Re-loaded and re-evaluated all three saved `best.pth` files; their selected thresholds and
  metrics exactly matched their JSON metadata. The candidates were not promoted because
  training and validation share recording groups and the validation set has no masks below
  the configured low-circularity cutoff; independent troubleshooting masks remain required.
- Verification:
  - Focused training-workflow tests passed all 5 tests; the full Conda-environment suite
    passed all 75 tests.
  - `ruff check .`, `black --check .`, and `git diff --check` passed. Black emitted only its
    known non-fatal user-cache permission warning.
  - Fresh wheel and source-distribution builds passed. Archive inspection confirmed both
    keep the shipped checkpoint and training log, the source distribution contains the new
    training workflow and tests, and the wheel excludes repository-only training files.
  - Treaty 0.5.0 validation was attempted but cannot validate the repository's newer
    relocated `treaty_docs/` layout: validating the root expects the documents at root, while
    validating `treaty_docs/` expects `AGENTS.md` and `project_overview.md` inside it.

### Balance fine-tuning and expose visibility QC (Codex, GPT-5, high reasoning)

- Created `feature/finetune` from `dev` and kept the pre-existing untracked `videos/` and
  `.pytest_tmp_full_console_fix_20260812/` directories untouched.
- Measured the maintained masks after the actual 148 x 148 resize: only 12 of 166 training
  masks and 2 of 56 validation masks have equivalent diameter at or below 15 model pixels,
  versus 21 training and 7 validation masks at or above 90 pixels. This supports balancing
  real tiny-pupil examples while treating the observed large-pupil failures as likely
  appearance, glare, or occlusion domain shift rather than size scarcity alone.
- Chose per-image macro overlap and an equal-weighted mean of represented tiny, medium, and
  large validation bins as the training-control signal. This prevents a large mask from
  hiding a missed tiny mask and gives rare size bins influence without changing the loss.
- Fine-tuning restores weights and detects the checkpoint architecture, but deliberately
  starts new optimizer, scheduler, early-stopping, and log state at a lower learning rate.
  The best weights, full log, and calibrated-threshold metadata are retained regardless of
  whether the run clears the separate promotion target.
- Applied segmentation QC to every analysis rather than velocity mode only. Visibility labels
  are conservative: empty masks are `not_detected`, while low-circularity or border-touching
  shapes are `partially_visible_or_uncertain`; the pipeline does not claim to reconstruct a
  pupil hidden behind an eyelid.
- Parked a composite Dice/focal/Tversky loss because the maintainer's prior attempt showed no
  clear gain; also parked higher-resolution or two-stage inference because it changes the
  148 x 148 checkpoint contract. Both are recorded in `next_steps.md` for controlled follow-up.
- Verification:
  - Fresh-training and packaged-checkpoint fine-tuning smoke runs completed on the public
    fixture. Fine-tuning used `1e-4` and wrote stable `.pth`, `.json`, and `.txt` outputs; the
    fixture remains a plumbing check rather than model-quality evidence.
  - A real-image three-frame run used the packaged filename calibration (`0.7`), wrote all six
    diameter/visibility/QC columns, marked all three frames visible and valid, and produced
    three overlays.
  - `ruff check .` and `black --check .` passed; Black retained its known non-fatal user-cache
    permission warning.
  - Conda-environment Pytest passed all 73 tests. The direct-interpreter attempt passed 72 and
    failed only because its subprocess could not resolve the console launcher from `PATH`;
    `conda run -n pupil_tracking pytest` supplied the established launcher environment.
  - Fresh wheel and source-distribution builds passed. Archive inspection confirmed both keep
    the packaged checkpoint/log, the source distribution includes the new training workflow
    test and script, and the wheel remains free of repository-only training/test files.
  - `git diff --check` passed.

### Adopt the treaty v0.9 docs layout (Codex, GPT-5)

- Created `chore/treaty` from the clean tracked state of `dev`, leaving the pre-existing untracked `videos/` and `.pytest_tmp_full_console_fix_20260812/` directories untouched.
- Previewed and applied the Copier migration from treaty template `v0.6.0` to `v0.9.0`. The compatibility step recorded `docs_dir: .` before relocation and updated only `.copier-answers.yml` plus the upstream-owned conventions file.
- Used `treaty relocate` to move `treaty_conventions.md`, `next_steps.md`, `work_log.md`, and `work_log_archive/` into `treaty_docs/`, preserving Git history and keeping `AGENTS.md` and `project_overview.md` at the root.
- Repaired the project-owned references identified by the relocation preview in `RELEASING.md` and `tests/test_end_to_end.py`; aligned the root documentation map and active handoff queue with the new paths.
- Published the Windows adopter report and remaining relocation UX suggestions as agent-collab-treaty issue #22, signed `Codex (GPT-5)`, after confirming issue #21 was the related verification request rather than an equivalent adopter report.
- Put the Agent Collab Treaty adoption badge before the DOI badge in `README.md` so the collaboration contract is the first project badge shown.
- Verification:
  - `treaty validate .` passed against the relocated layout; `treaty diff` compared against template `v0.9.0` and reported the expected project-authored section drift with the upstream conventions untouched.
  - `ruff check .` and `black --check .` passed; Black reported 34 files unchanged plus its known non-fatal cache permission warning.
  - The first sandboxed Pytest attempt could not access its default Windows temp directory or launch the environment console script. The retry used the same project environment with an accessible dedicated `--basetemp` and passed all 65 tests; the only remaining messages were the established dataset deprecation warning and a non-fatal Pytest cache warning.
  - Final conflict-marker, stale-root-reference, Git-status/index, and whitespace checks passed.

## 2026-08-12

### Compress the README to the following path (Claude Code, Opus 5)

- Second review pass, on the principle the maintainer stated: **a reader who has to look up something they skipped is better off than a reader who abandons the instructions because too much was in the way.** Prose that explains a command before the reader has run it is the thing to cut.
- Merged "Pupil center and velocity" into `Usage` as a third command. It is a primary use rather than an advanced mode, so a sibling section to `Usage` overstated its distance from the basic call.
- Deleted the two explanatory bullets that followed the velocity command after checking they were redundant: the CLI reference table's `--calculate_velocity` row already states that every encoded frame is analyzed, and its `--acquisition_fps` row already states the `--image_dir` requirement and the video-header default. Only the container-rate-is-not-experimental-time warning was unique, and it became an FAQ entry.
- Replaced the `Units` and `Quality control` subsections, about 40 lines of prose between the output files and the CLI reference, with an eight-row CSV column table. The table carries what a reader needs while reading their own CSV: per-column meaning, the equivalent-circle formula, the coordinate convention, and which columns are velocity-only.
- Moved the displaced material into the FAQ as two entries, on cross-recording comparison and on empty center/speed fields. Nothing was dropped: the calibration caveat, the model-pixel versus input-pixel distinction, the continuity rationale for `estimated_pupil_diameter`, the no-interpolation rule, and the usability of `warning` rows all survive, and `Output` keeps a one-line pointer to them.
- Section line counts after the pass: Install 11, Usage 30, Output 39, CLI reference 37. The path from the top of the file to the full argument list is roughly halved.
- Verification:
  - `ruff check .`, `black --check .` (34 files unchanged), `pytest` (`65 passed`), and `git diff --check` passed.
  - Rendered through `readme_renderer[md]` as PyPI does: 28,289 characters, GIF and both badges present, five tables intact, no unresolved reference links, 15 of 15 anchors resolving.
  - Re-asserted that the contents list matches the `##` heading sequence exactly.

### Re-cut the README on maintainer review (Claude Code, Opus 5)

- The first restructure was accepted only in part. Reworked it against the maintainer's specific objections rather than patching around them.
- **The Zenodo badge endpoint is why the DOI badge rendered late.** `zenodo.org/badge/DOI/...svg` answers through a redirect in ~0.46 s against ~0.14 s for an equivalent `img.shields.io` static badge, and GitHub's image proxy caches the redirecting variant poorly on first view. Swapped to `img.shields.io/badge/DOI-10.5281%2Fzenodo.21897795-1682D4.svg`. The badge target is unchanged, so the concept DOI still resolves to the newest version.
- Cut the badge row from five to two, keeping only the DOI and the treaty badge. The PyPI version, Python version, and license badges duplicated facts the Install section and `LICENSE` already state.
- Moved the demo GIF above the prose description, per the maintainer's point that the animation carries more than the paragraph does.
- Replaced the "If you want to... / Go to" contents table with a plain nested list. The maintainer's test is worth recording: **if a section needs a gloss explaining when to read it, the heading is wrong and the section may not deserve to exist.** Applied that test to the whole file.
- Deleted the "Will this work on my recordings?" section under that test. Its load-bearing content survives in two better places: the 148 x 148 framing precondition is now a short blockquote at the end of Usage, where it is a precondition rather than an aside, and the overlay-and-threshold diagnostic became the first FAQ entry.
- Reordered the body to follow what a user does: Install, Usage, Pupil center and velocity, Output, CLI reference, then Python API. Velocity mode now follows Usage directly because it is a primary use, not an advanced option. Output moved up because it describes what exists on disk immediately after a run, and the folder tree moved with it instead of sitting inside Usage.
- Verification:
  - `ruff check .`, `black --check .` (34 files unchanged), `pytest` (`65 passed`), and `git diff --check` passed.
  - Rendered through `readme_renderer[md]` in a separate virtual environment, as PyPI does: 29,100 characters, GIF and both badges present, four tables intact, no unresolved reference links, and 18 of 18 in-document anchors resolving. A bare `twine check` still cannot substitute for this; see the previous entry.
  - Asserted programmatically that the contents list matches the `##` heading sequence exactly, so the two cannot drift.

### Record the Zenodo DOIs and restructure the README (Claude Code, Opus 5)

- Retrieved the minted identifiers from the public Zenodo API rather than waiting on the web UI: concept DOI `10.5281/zenodo.21897795` and v0.2.0 version DOI `10.5281/zenodo.21897796`. `RELEASING.md` step 8 now carries that one-line query so the next release does not have to rediscover it.
- **Citation generators ignore the top-level `doi` when `identifiers` is present.** Verified with `cffconvert` across three CFF variants: the first doi-type entry under `identifiers` always wins, and the top-level `doi` is used only when the list is absent. The first arrangement tried listed the concept DOI first, and exported BibTeX cited the moving concept DOI instead of the archived code. The per-version entry is now first, both `doi` and that entry hold the version DOI, and the ordering requirement is recorded in `CITATION.cff` and `RELEASING.md` so a future release cannot silently reintroduce it. Exported BibTeX and APA now both carry `10.5281/zenodo.21897796`.
- Restructured `README.md` around the maintainer's stated priority: a reader should reach installation and a first successful run without studying anything. Install is now four lines, usage is two commands plus the output tree, and the environment recommendation, PyTorch CPU/GPU builds, packaging-name background, and development install moved into a later `Installation notes` section. A linked contents table sits above `Install`. No technical claim was dropped: the units contract, quality-control semantics, and full argument descriptions were relocated intact, and `--batch_size` and `--mask_transparency` are documented for the first time.
- Switched README links to reference-style absolute URLs. The README is the PyPI long description, and PyPI does not rewrite relative paths, so the demo GIF and every repository-file link were broken there. Reference definitions keep the prose readable while the emitted HTML is absolute.
- **`twine check` passed without rendering the Markdown.** The environment had `readme_renderer` without its `md` extra, so `render()` returned `None` and the check reported PASSED on an unrendered file. Re-ran the render in a separate virtual environment with `readme_renderer[md]`: 30,198 characters, the GIF and all four badges present, all five tables intact, no unresolved reference links, and 19 of 19 in-document anchors resolving after the renderer's `user-content-` rewrite. Treat a bare `twine check` as insufficient evidence that a README renders.
- Base branch correction: this work started from `refactor`, which predates the `pupil_tracking` to `mouse_pupil_analysis` rename and the repository rename. Rebased onto `dev`, so the README documents the published import name rather than the retired one.
- Verification:
  - `ruff check .`, `black --check .` (34 files unchanged), `pytest` (`65 passed`), and `git diff --check` passed.
  - `python -m build` produced both distributions; `cffconvert --validate` reports valid against schema 1.2.0.
  - Every absolute URL in the README resolves (the `doi.org` and Zenodo badge URLs answer 403 to a scripted user agent and 200 to `curl`).
  - The local environment held a stale `pupil-tracking` console script that made `tests/test_cli_help.py` fail before any edit; reinstalled per `../AGENTS.md` before trusting the suite.

### Finalize the package namespace and release gate (Codex, GPT-5)

- The user confirmed that the pending PyPI publisher is registered with the renamed repository and that Zenodo remains enabled after the GitHub rename. Those account actions no longer block the release.
- Audited PR #3's first namespace guard and found that it rejected wheel paths but accidentally exempted an sdist path because the archive root itself contains `mouse_pupil_analysis`. Replaced the duplicated inline checks with `scripts/verify_distribution_namespaces.py`, which checks complete path components, and added explicit wheel-layout, sdist-layout, and false-positive regressions. Included the verifier in the sdist alongside the tests that import it, while keeping it out of the installed wheel.
- The first pushed regression imported the repository-only verifier as an ordinary package. Local `python -m pytest` placed the repository root on `sys.path`, but CI's `pytest` console entry point did not, so collection failed across the matrix. The test now loads the exact verifier file explicitly and passes under both invocation styles without installing release tooling into the wheel.
- Moved the ignored historical checkpoint archive under `mouse_pupil_analysis/checkpoints/archive/`, removed the stale legacy Python cache, and merged the fully green PR #3 into `dev` as `1f9d5b2`.
- Updated the 0.2.0 changelog and citation release date to the verified local date, 2026-08-12.
- Fast-forwarded the fully verified `dev` release commit `3de4360` to `main`, confirmed the separate `main` CI run passed, and published annotated tag `v0.2.0` at that exact commit.
- The tag-triggered release workflow built and verified both distributions, passed its clean-wheel smoke test, and published `mouse-pupil-analysis==0.2.0` through PyPI Trusted Publishing. The public PyPI record exposes `mouse_pupil_analysis-0.2.0-py3-none-any.whl` and `mouse_pupil_analysis-0.2.0.tar.gz`.
- Published the curated GitHub Release for `v0.2.0`, triggering the enabled Zenodo integration. Zenodo had not exposed a matching public record during the first several minutes, so the DOI metadata update remains an explicit follow-up rather than recording an unverified identifier.
- Verification:
  - `ruff check .`, `black --check .`, Treaty validation, and `git diff --check` passed.
  - The full local Pytest suite passed (`65 passed`), including a direct `pytest` console-entry run of the namespace regressions.
  - A clean wheel and sdist passed the shared verifier, contained no `pupil_tracking/` members, and retained the packaged checkpoint and both unchanged console commands. A clean wheel installation exposed only `mouse_pupil_analysis`.
  - Both GitHub Actions runs passed across lint, wheel smoke, Ubuntu Python 3.10-3.13, Windows 3.12, and macOS 3.12.

## 2026-08-11

### Drop the legacy namespace instead of hardening it (Claude Code, Opus 5)

- Reviewed the rename commit and found that the compatibility wrappers re-exported with `import *`, which honors `__all__`. Because the rename added `__all__` to `mouse_pupil_analysis/run_pupil_analysis.py`, `from pupil_tracking.run_pupil_analysis import run_analysis` raised `ImportError` even though the commit's stated goal was preserving legacy imports. Opened #3 fixing that by forwarding through a module-level `__getattr__`.
- **Superseded that fix in favor of removing the shim, per review.** The reviewer's objection was decisive and correct: the unrelated PyPI distribution `pupil-tracking==1.0.1` installs the exact path `pupil_tracking/__init__.py`, so shipping the shim would give two distributions ownership of one import namespace. Verified against the published wheel rather than assuming, and measured the consequence in clean virtual environments:
  - Installing `pupil-tracking` then `mouse-pupil-analysis` silently overwrote the unrelated project's `__init__.py` with our shim, leaving `pip list` reporting `pupil-tracking 1.0.1` as installed.
  - Uninstalling `mouse-pupil-analysis` then deleted the shared file, leaving `pupil-tracking 1.0.1` still listed while `import pupil_tracking` raised `ModuleNotFoundError`. An unrelated third-party package was destroyed by our uninstall.
  - The reverse order broke our own shim instead: `pupil_tracking.analyze_video` raised `AttributeError`. Both orders fail, and pip reports success throughout.
- Removed `pupil_tracking/`, dropped `pupil_tracking*` from the setuptools include list, and removed the shim tests, CI/release shim smoke checks, and deprecation wording from `README.md`, `RELEASING.md`, `CHANGELOG.md`, `../AGENTS.md`, and `../project_overview.md`.
- Added regressions so the namespace cannot return: `tests/test_metadata.py` asserts the packaging config and working tree never reintroduce it and that both console scripts still target `mouse_pupil_analysis`, and CI plus the release workflow fail on any `pupil_tracking/` member in a built wheel or sdist. The release check runs before publication, since a PyPI version can never be reused.
- Reverted the `media/` and `training/` Conda environment names to `mouse_pupil_analysis` at the maintainer's direction. Those examples target contributors following `README.md`, whereas the `pupil_tracking` name in `../AGENTS.md` and `.copier-answers.yml` describes this checkout's existing local environment; the two intentionally do not match.
- Left commit `23cab15`'s literal `\n` characters alone, per the maintainer's decision not to rewrite the already-pushed shared `dev` branch.
- Verification:
  - `ruff check .`, `black --check .`, and the full Pytest suite passed (`62 passed`).
  - A clean wheel and sdist build contained no `pupil_tracking/` members, kept the checkpoint and training log under `mouse_pupil_analysis/checkpoints/`, and retained both console entry points targeting `mouse_pupil_analysis`.
  - Installed the wheel into a clean environment alongside `pupil-tracking==1.0.1` in both orders and confirmed the two distributions no longer share any file.

### Adopt the permanent mouse-pupil-analysis identity (Codex, GPT-5)

- Renamed the primary Python package from `pupil_tracking` to `mouse_pupil_analysis`, matching the already-selected `mouse-pupil-analysis` distribution and the forthcoming GitHub repository name. Kept `run-pupil-analysis` and `extract-frames` unchanged so existing command-line habits continue to work.
- Retained `pupil_tracking` as a deprecated forwarding package, including the former module paths and direct `python -m` entry points. New code, tests, training utilities, CI, release automation, package data, documentation, and citation metadata use `mouse_pupil_analysis`.
- Updated release guidance for the repository rename. The remaining account work is to register the pending PyPI publisher with repository `mouse-pupil-analysis` and, after the rename, confirm that Zenodo still has the renamed repository enabled.
- Verification:
  - `ruff check .`, `black --check .` (45 files unchanged), and `git diff --check` passed.
  - Full Pytest suite passed (`61 passed`); the expected `mouse_pupil_analysis.dataset` deprecation warning remains.
  - A clean wheel and source distribution build each contained the checkpoint and matching training log only under `mouse_pupil_analysis/checkpoints/`. The source distribution also retained all 71 sample-data entries and four training entries.
  - The wheel entry points remain `run-pupil-analysis` and `extract-frames`, both targeting `mouse_pupil_analysis`; both `--help` invocations passed from a clean wheel installation.
  - Clean-install checks confirmed the primary import, the deprecated `pupil_tracking` forwarding import and warning, and version `0.2.0`.

### Make the demo GIF script headless-safe (Claude Code, Opus 5)

- `media/make_gif.py` still created figures through `pyplot`, so it carried the same interactive-backend dependency that was removed from the package. It writes a GIF and never calls `plt.show()`, so it was requesting a GUI backend it never used and would fail on any machine without a display.
- Pinned the backend with `matplotlib.use("Agg")`. Placed after the import block rather than between imports, because this repository's Ruff configuration enables `E402`; verified that backend resolution is lazy enough for that to still take effect before the first figure is created.
- Chose the one-line backend pin over converting the script to direct `Figure` construction, as the user selected. It is a maintainer script rather than library code, and the script never displays anything, so nothing observable changes.
- Verification:
  - Loaded the script through `runpy` with `tkinter` and `_tkinter` imports blocked and `MPLBACKEND=TkAgg` forced, then rendered a figure at the same call site that previously failed. Backend resolved to `Agg`.
  - `ruff check .`, `black --check .`, `pytest` (`60 passed`), and `make_gif.py --help` unchanged.

### Align center and speed unit wording with the diameter contract (Claude Code, Opus 5)

- Applied the same input-image qualification to pupil-center coordinates and speed that the diameter column already carried. `model_to_original_coordinates` maps through `original_size`, which is the supplied image, so centers are source-video pixels only for video input; with `--image_dir` they follow whatever the caller prepared. Updated the README output table and units section, the tracking plot's center axis label, the `DiameterRow` and `resize_scale` docstrings, `sample_data/README.md`, `next_steps.md`, and three test comments. The calculation is unchanged; only the claims about it were wrong.
- **Correcting the previous entry's dismissal of the `git diff --check` warning.** That entry reported `git diff --check` clean, which was true but answered the wrong question: `git diff --check` compares the worktree to the index, whereas the pull-request diff is `git diff origin/dev...refactor`. Against the base there was a genuine `new blank line at EOF` warning on `work_log.md`, now removed. The earlier reasoning about the `\\ No newline at end of file` markers appearing on removed lines remains correct and is a separate matter.
- Tightened the release ancestry gate to `origin/main` only, per the user's decision. A PyPI version number can never be reused, so a published release must correspond to code that reached the default branch. `RELEASING.md` now shows the `dev` to `main` merge as an explicit step before tagging.
- Verification:
  - `ruff check .`, `black --check .`, `pytest` (`60 passed`).
  - `git diff origin/dev...HEAD --check` clean after committing.
  - Confirmed no `video pixel` or `original-video` wording remains outside `work_log.md`, which is retained as an append-only historical record.

### Address the pull-request review on #2 (Claude Code, Opus 5)

- Renamed `pupil_diameter_video_pixels` to `pupil_diameter_input_pixels`. The reviewer was right that the value derives from the supplied PNG's dimensions, not the source video; with `--image_dir` and already-preprocessed 148 x 148 frames it equals the model-pixel column, which one of this branch's own tests asserts. Narrowed the README claim: removing the model rescaling does not make recordings comparable unless their optics match.
- Rebuilt the changelog against actual tag contents. The previous backfill credited the velocity feature, unified outputs, overlays, and the module split to `v0.1.4`, but that tag was cut 2026-06-12 and all of that work landed in August. Everything post-tag now sits under 0.2.0. Confirmed from `pyproject.toml` history that the version went 0.1.2 straight to 0.1.4, so 0.1.3 was never released and is noted as such rather than invented.
- Extracted `paired_image_mask_paths(...)` into `pupil_tracking/augmentation.py` and used it from both `training/run_train.py` and `training/check_augmentation.py`. The earlier fix covered only `run_train.py`; the augmentation viewer still sorted the two directories independently, which is the same silent-mispairing bug. Orphan masks are now rejected by default with an `allow_orphan_masks` escape hatch, and the helper lives in the package so it is importable and testable without executing a training script.
- Restored the deprecated `PupilDataset` shim's original positional signature. As written it accepted only `(image_paths, mask_paths, **kwargs)`, so `PupilDataset(images, masks, True)` raised `TypeError`. Documented that it is now a factory function, so `isinstance` and subclassing no longer work, rather than implying full compatibility.
- Added `show_progress`, defaulting to off, to inference and frame extraction. Converting `print` to logging had left `tqdm` writing to stderr unconditionally, so the claim that library callers get no output was false. Verified a library call now emits zero bytes on both streams while the CLI is unchanged.
- Corrected the CUDA install instructions. The `cu124` index stops at PyTorch 2.6.0, below this package's `torch>=2.8` floor, so that command could not have worked; `cu126`, `cu128`, and `cu129` carry 2.8 or newer. Removed the hardcoded wheel-size figures, which drift between releases.
- Hardened the release workflow: it now resolves the single packaged checkpoint and requires that exact file plus its matching training log in both artifacts, rejects stray checkpoints, and refuses to publish from a commit that is not an ancestor of `origin/main` or `origin/dev`. Checkout uses full history so the ancestry check can resolve merge bases.
- Smaller documentation corrections: default result directory for video input is `<stem>_result`, not `<stem>_frames_result`; the `analyze_frames` velocity example was missing `calculate_velocity=True`, without which `acquisition_fps` does nothing; recommended citation updated to 0.2.0; the Zenodo claim is now conditional because archiving is not yet enabled; removed the settled 0.2.0 version decision from `next_steps.md`.
- Did not change anything for the reported `git diff --check` warning. `git diff --check` is clean on this branch. The `\\ No newline at end of file` markers in the pull-request diff appear on removed lines, from the old `.pre-commit-config.yaml` and the deleted `requirements.txt`; both are already fixed by this branch. The only tracked file still lacking a trailing newline is the packaged training log, which this branch does not modify.
- Verification:
  - `ruff check .`, `black --check .` (31 files unchanged), `pytest` (`57 passed`), `git diff --check` clean.
  - New `tests/test_training_pairing.py` covers stem pairing, missing masks, orphan masks in both policies, the empty-directory case, the committed fixture, and both deprecated-shim call shapes.
  - Confirmed empirically that `https://download.pytorch.org/whl/cu124` serves at most torch 2.6.0 while `cu126` and `cu129` serve 2.13.0.
  - Ran the hardened release checkpoint verification locally against freshly built artifacts; both the wheel and the source distribution carry the expected checkpoint and training log.

### Merge the sample fixture and add real-image regression coverage (Claude Code, Opus 5)

- Merged `origin/dev` into `refactor` ahead of a pull request. `training/run_train.py` and `training/check_augmentation.py` auto-merged cleanly: dev's editable `DATA_ROOT` and this branch's stem-based image/mask pairing and seeding compose without conflict. Both branches had independently performed the same five-date work-log rotation and produced byte-identical archive files, so that did not conflict either. The three documentation conflicts were additive and resolved by keeping both sides.
- Added `tests/test_real_images.py`, which runs the packaged checkpoint over `sample_data/`. This is the coverage that `test_end_to_end.py` cannot provide: a synthetic blob segments plausibly regardless of which weights are loaded, so only real frames detect a corrupted or swapped checkpoint or a `resize_with_pad` regression. Landmarks are asserted as ranges rather than exact values so the tests survive platform floating-point differences.
- The two uncropped fixture recordings have different source resolutions, 284 x 156 and 304 x 176, which is what makes the video-pixel diameter conversion verifiable against real geometry rather than a synthetic frame.
- **Corrected an inaccurate changelog claim.** The 0.1.4 entry stated that deriving the equivalent-circle constant from `4 / pi` instead of the rounded literal `1.27` left results unchanged. It does not: reported diameters are a factor of `sqrt(4 / pi / 1.27)` = 1.001275 larger, or +0.1275%. The discrepancy surfaced because the fixture's documented diameter range of 18.38 to 25.38 model pixels reproduced as 18.40 to 25.41, which is exactly that factor. The changelog now states the change and warns against pooling diameters across the version boundary; `sample_data/README.md` records the updated range and why it moved.
- Verification:
  - `ruff check .`, `black --check .` (30 files unchanged), `pytest` (`49 passed`).
  - Reproduced every other documented velocity landmark after the refactor: 31 rows, 27 `valid` and 4 `warning`, all warnings `abrupt_area_change`, 30 of 30 possible speeds published, and timestamp spacing of exactly 1/97 s.
  - `python -m build`; the source distribution carries the checkpoint and all 61 sample PNGs at 3.09 MB, while the wheel carries the checkpoint and no sample data at 1.78 MB.

### Add a portable real-data fixture (Codex, GPT-5)

- Added `sample_data/` with eight curated training image/mask pairs, four validation pairs, six uncropped frames grouped by recording, and 31 consecutive velocity frames from source frames `07212`-`07242` at 97 Hz.
- Kept the uncropped examples unchanged. Prepared the velocity inputs with the package's grayscale 148 x 148 resize-and-pad convention, preserving consecutive source suffixes so all frame-to-frame velocities remain eligible.
- Added clone-and-run segmentation, overlay, velocity, augmentation, and training guidance plus a provenance/transformation manifest. The images and masks are published with permission and are explicitly scoped as workflow fixtures rather than benchmark or useful training data.
- Added an editable `DATA_ROOT` to the three training utilities and capped augmentation inspection at the available dataset size, so the public fixture can be used directly without disturbing the full local training collection.
- Added fixture integrity coverage for image/mask pairing, nonzero-foreground label semantics, raw-frame grouping, consecutive velocity suffixes, prepared-image dimensions, acquisition rate, and manifest paths.
- Included the fixture and training utilities in source distributions so their bundled guide and sample-data test are self-contained, while keeping them outside the installed wheel.
- Rotated the prior five work-log dates into `work_log_archive/work_log_2026-08-01_to_2026-08-10.md` according to the repository logging policy.
- Verification:
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m pytest -q tests\test_sample_data.py` (`3 passed`).
  - Packaged-checkpoint runs on both three-frame raw recording groups produced six overlays and separate compact diameter outputs.
  - Packaged-checkpoint velocity run produced 31 overlays, 27 valid rows, four warning rows, and all 30 possible speeds; the rendered plot and representative opening, peak-speed, and closing overlays were inspected.
  - One augmented four-image training step and one four-image validation batch completed on CUDA using the included paired fixture.
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m ruff check .`.
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m black --check .` (`19 files` unchanged; Black reported an inaccessible user-cache warning but completed successfully).
  - The direct-environment full Pytest run passed 25 tests and failed only because its subprocess could not resolve the installed `run-pupil-analysis` launcher. The established Conda invocation supplied the console-script environment and passed all `26` tests.
  - `C:\Users\yzhao\miniconda3\envs\pupil_tracking\python.exe -m build --wheel --sdist`; inspection confirmed the wheel retains the packaged checkpoint/log and excludes sample/training extras, while the source distribution retains the checkpoint/log, all 61 sample PNGs, their documentation/manifest, and all four training files.
  - `C:\Users\yzhao\python_projects\agent_collab_treaty\.venv\Scripts\treaty.exe validate .`.
  - `git diff --check`.
### Broaden CI coverage and guard release metadata (Claude Code, Opus 5)

- Added Windows and macOS jobs to CI and extended the Python range to 3.13. Windows is the primary development platform and had never been covered by CI.
- Aligned the pre-commit black and ruff pins with the dev extra (they had drifted to 24.10.0 and v0.12.9 against 25.1.0 and 0.14.14) and added a pre-commit job so the hooks and CI can no longer disagree about formatting.
- Added `tests/test_metadata.py` asserting that `__version__`, `pyproject.toml`, `CITATION.cff`, and `CHANGELOG.md` agree, and that `__all__` matches the lazy export map. A DOI is only useful if the recorded version is real.
- Caught while adding that test: it used `tomllib`, which is Python 3.11+, while `requires-python` allows 3.10 and CI tests it. The import is now guarded and the module skips on 3.10. Verified by executing the module header with `tomllib` blocked.
- Moved the `media/make_gif.py` load into a fixture so a missing script skips instead of breaking collection.
- Deleted `requirements.txt`, which duplicated `[project].dependencies` with no declared ownership.
- Documented the macOS environment paths in `../AGENTS.md` alongside the existing Windows ones, and recorded the console-script removal hazard when uninstalling the superseded distribution.
- Verification:
  - `ruff check .`, `black --check .` (28 files unchanged), `pytest` (`40 passed`).
  - `pre-commit run --all-files` with the synced pins (`black Passed`, `ruff Passed`).
  - Confirmed the previous push's CI run succeeded before layering these changes.

### Report pupil diameter in video pixels and harden model loading (Claude Code, Opus 5)

- Added a `pupil_diameter_video_pixels` output column. `estimated_pupil_diameter` measures the 148 x 148 model image, so it is not comparable between recordings with different resolution or cropping; the new column inverts the resize-and-pad geometry. The existing column is unchanged, per the additive approach the user chose.
- Centralized the resize geometry in `preprocessing.resize_scale`, which `model_to_original_coordinates` now also uses, removing a duplicated derivation. Area-derived lengths convert by the geometric mean of the two axis scales, since areas scale by their product.
- Replaced the assumption that every checkpoint uses spatial attention with inference from the checkpoint's own state-dict keys, so a non-attention `--checkpoint` loads and a genuinely incompatible file reports what failed. Chose this over the planned JSON sidecar manifest because it needs no new files and works for checkpoints that have none.
- Fixed a silent correctness bug in `training/run_train.py`: images and masks were paired by sorting two directories independently, which trains against misaligned labels whenever the folders diverge. Pairing is now by filename stem, which `labelme_json2png.py` guarantees, with a loud error on mismatch. Added seeding for `random`, `numpy`, and `torch`.
- Deliberately did not wrap `run_train.py` in a `main()` function, despite the original plan. The user runs these scripts cell by cell in an IDE, and moving module-level state into a function breaks that workflow. The pairing bug and the missing seed were the substantive issues.
- Vectorized the frame-to-frame kinematics and moved per-frame image-size reads into the single inference pass.
- Verification:
  - `ruff check .`, `black --check .`, `pytest` (`35 passed`), `python -m build`.
  - Differential test of the vectorized kinematics against the original row loop over 300 randomized cases covering source-index gaps, invalid segmentations, and acquisition rates of 10, 33.3333, and 100 fps: zero mismatches.
  - Verified the diameter conversion on a real pipeline run: for a 200 x 160 frame the observed ratio was 1.35364 against an expected `1 / sqrt(0.7400 * 0.7375)` of 1.35364.
  - Verified checkpoint loading for the packaged attention checkpoint, a freshly saved non-attention checkpoint, and a deliberately incompatible file.
  - Verified the stem pairing and its error path against temporary directories with deliberately mismatched creation order.

### Add a public Python API and split the pipeline into focused modules (Claude Code, Opus 5)

- Extracted orchestration out of the CLI into `api.py`. `AnalysisConfig` holds and validates every input; `run_analysis` returns an `AnalysisResult` carrying the analysis table, both output paths, frame metadata, and the internal tracking DataFrame. `analyze_video` and `analyze_frames` are the keyword front door.
- Made the package's public names resolve through PEP 562 module `__getattr__`, so `import pupil_tracking` no longer imports PyTorch. Kept `__all__` literal because Ruff cannot match a computed `__all__` against `TYPE_CHECKING` imports.
- Split `dataset.py` into `preprocessing.py` (inference) and `augmentation.py` (training). `InferenceDataset` and `SegmentationDataset` replace `PupilDataset`, whose return type depended on whether `mask_paths` was supplied. `dataset.py` remains a deprecating shim.
- Moved table assembly into `results.py` and figures into `plotting.py`; plot functions return figures rather than saving them.
- Replaced every library `print` with module loggers and added `logging_utils.configure_cli_logging()`, which the console scripts call so terminal output is unchanged. Verified the CLI transcript line for line against the previous behavior.
- Invalid CLI argument combinations now route through `parser.error(...)` instead of raising `ValueError`.
- Pulled three planned correctness items forward because they touched the same lines: checkpoint lookup is lazy and no longer returns a path from an exited `resources.as_file` block; the diameter factor is derived from `4 / pi` instead of the literal `1.27`; `num_workers` is configurable rather than hardcoded to 4.
- Deprecated `generate_pupil_mask_prediction` in favor of `analyze_frames`.
- Two self-inflicted defects caught during verification and fixed: replacing `run_pupil_analysis.py`'s `__main__` guard with an experiment block would have broken direct script invocation; and module `__getattr__` does not fire for bare global lookups inside its own module, so the in-module `DEFAULT_CHECKPOINT` references would have raised `NameError`.
- Verification:
  - `ruff check .`, `black --check .` (25 files unchanged).
  - `pytest -q` (`26 passed`), including the new `tests/test_end_to_end.py`, which builds a synthetic video with `cv2.VideoWriter` and runs the packaged checkpoint through extraction, inference, velocity mode, overlays, and a video-versus-image-directory equivalence check.
  - Manual CLI run against a synthetic video, plus the missing-argument path, confirming usage text instead of a traceback.
  - `python -m build`, then installed the wheel into a clean venv and confirmed the public API resolves and PyTorch stays unimported.

### Publish as mouse-pupil-analysis with release automation (Claude Code, Opus 5)

- Renamed the distribution to `mouse-pupil-analysis`. The PyPI name `pupil-tracking` is already held by an unrelated project (v1.0.1, a different author), so the original name was unpublishable. The import name `pupil_tracking` is unchanged.
- Added `pupil_tracking.__version__` from installed distribution metadata, so the package, `pyproject.toml`, and `CITATION.cff` cannot silently disagree.
- Added `.github/workflows/release.yml`: tag-triggered, verifies the tag against the project version, the citation version against the project version, and the presence of the packaged checkpoint with no archived-checkpoint leak in both artifacts, then publishes through PyPI Trusted Publishing.
- Added `CHANGELOG.md` backfilled from Git history and `RELEASING.md` covering the one-time PyPI publisher registration and Zenodo webhook, both of which are account actions the repository cannot perform.
- Documented CPU and CUDA PyTorch install paths. Confirmed from the PyPI wheel index that Windows and macOS wheels are already CPU-only; only Linux defaults to a CUDA-bundled wheel.
- Recorded the blocked sample-data request in `next_steps.md` and folded the superseded portable-fixture thread into it.
- Environment note: this session ran on macOS with a newly created `pupil_tracking` conda environment on Python 3.12. The resolved stack was torch 2.13, OpenCV 5.0, pandas 3.0.5 — all major versions above the declared floors — and the suite passed, which is useful evidence the version pins are not too loose.
- Uninstalling the stale `pupil-tracking` distribution metadata also removes the shared `run-pupil-analysis` and `extract-frames` console scripts, because both distributions declare the same script names. Reinstalling restores them; anyone pulling this branch into an existing environment must uninstall then reinstall, in that order.
- Verification:
  - `ruff check .`, `black --check .`, `pytest -q` (`23 passed`) before and after the rename.
  - `python -m build`, then verified the checkpoint ships in both wheel and sdist with no `checkpoints/archive/` contents.
  - Installed the wheel into a clean venv; `pupil_tracking.__version__` resolved to `0.1.4` and `run-pupil-analysis --help` succeeded.
