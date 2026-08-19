# Work Log

Prepend new session notes to the top of this file. The live log holds at most the 5 most recent unique calendar dates; older groups rotate into `work_log_archive/`.

## 2026-08-18

### Integrate HQL097 pupil, closed-eye, and uncertain annotations (Codex, GPT-5)

- Audited all 67 Labelme annotations beside their 307 x 198 source frames. The batch contains
  exactly one supported label per frame: 47 `pupil`, 14 `no_visible_pupil`, and 6 `uncertain`.
  Contact-sheet review confirmed the pupil polygons follow the visible boundary, the negatives
  are fully closed or occluded, and the uncertain squints should not receive segmentation loss.
- Made `labelme_json2png.py` validate the three-label contract. Pupil polygons are rasterized
  directly, explicit negatives become exact image-sized all-black masks, uncertain annotations
  produce no mask, and unknown or contradictory labels fail before writing output. Direct
  rasterization removes the unavailable `labelme_export_json` console-command dependency.
- Added `training/import_labelme_batch.py` as the reproducible intake boundary that was missing
  from the documentation. It previews by default, validates the whole source batch before any
  write, refuses an existing session, creates compact image/mask pairs without redundant JSON
  copies, archives uncertain image/JSON pairs outside training, and refreshes the manifest and
  materialized folds whenever `--apply` is passed. A real-batch smoke reproduced all 61 HQL097
  image and mask hashes exactly.
- Added 61 compactly named HQL097 image/mask pairs to development: 47 nonempty pupil masks and
  14 empty negatives. Preserved the 6 uncertain image/JSON pairs under the session's
  `uncertain/` folder, outside `images/`, and removed the 61 generated-mask JSON copies after
  verification; the original 67 source annotations remain in `frame_recommendations/`.
- Refreshed the frozen grouped manifest and materialized folds. The pool now has 316 pairs across
  19 sessions: 312 development pairs and the unchanged 4-image trial5 holdout. HQL097 entered
  fold 4; all 255 earlier assignments were carried over unchanged. Fold sizes are 76/58/73/105,
  so the next action is a matched 316-image baseline before changing any training configuration.
- Verification: the focused converter/importer tests pass; Ruff and Black pass repository-wide;
  `git diff --check` is clean; and the full test suite passes when the Conda Scripts path is
  injected inside the pytest process. The initial plain-shell pytest attempt hit only the known
  desktop PATH isolation for `run-pupil-analysis`; rerunning through the documented environment
  workaround passed. Wheel and source-distribution builds also pass: the wheel retains the
  packaged checkpoint/metadata and excludes training sources, while the source distribution
  contains the importer and `sample_data/splits.json`.

## 2026-08-17

### Intake two new sessions and train the active-learning committee (Codex, GPT-5)

- Audited the two Desktop batches before intake. All 29 `260807_3582_Purple_trial3`
  image/mask pairs and all 4 `260812_3582_Purple_trial5` pairs had exact session prefixes,
  one-to-one stems, unique trailing frame indices, matching dimensions, binary nonempty masks,
  and no orphans. Visual review confirmed trial5's very small masks align with genuinely
  constricted pupils rather than a grouping error.
- Added trial3 to development and froze trial5 as the outer holdout. The manifest now records
  255 pairs across 18 sessions: 251 development images in 17 sessions and 4 untouched holdout
  images. Trial5 belongs to no CV fold and was neither loaded nor scored during this session.
- Added `training/compact_frame_names.py` and renamed all 255 local pairs to
  `frame_<five-digit-source-index>.png` within their session. Hash checks confirmed the existing
  222 image/mask contents were unchanged; the 33 Desktop source/destination PNG hashes also match.
  Removed the 222 obsolete Labelme JSON files after masks were verified; no JSON remains under
  `labeled_frames/`.
- Made legacy-copy suppression compare image/mask content rather than basenames, so the retained
  verbose-name legacy backups do not re-enter the census after compact renaming.
- Trained a four-model scratch committee with natural sampling and macro-IoU selection in
  `checkpoints_exp/cv255_nat_macro_20260817`. Fold macro IoUs were 0.7754, 0.4990, 0.5400, and
  0.6701; mean per-session IoU was 0.6468 and image-weighted IoU was 0.6246. Trial3 scored 0.6318
  when held out in fold 2. The two known aperture-grab sessions remain the worst (0.2605 and
  0.1678), with predicted-to-labelled area ratios of 3.82x and 6.37x.
- Frame recommendation is ready to use with the four saved `best.pth` files. No new unlabeled
  pupil video is currently visible on Desktop or in the repository; `videos/eye.avi` is an older
  July demo recording with existing review outputs, so it was intentionally not treated as the
  incoming batch.
- Verification:
  - Focused split/intake/rename checks passed (58 tests); every manifest path exists, the holdout
    is in no fold, all 255 pairs are accounted for, and no Labelme JSON remains.
  - CUDA training completed all four folds on the RTX 3070 Laptop GPU and wrote four checkpoints,
    metadata files, logs, and the CV summary in 25 minutes.
  - Ruff, Black, `git diff --check`, and the full test suite passed. Pytest required a repository-
    local temporary directory and injecting the Conda Scripts path inside Python because the
    desktop sandbox cannot access its default temp root and strips that path at startup.
  - An end-to-end smoke loaded all four checkpoints on CUDA, scored one three-frame unlabeled
    fixture recording, and wrote two recommendations plus `selection.csv`; smoke output was removed.

### Clean local media layout and recommend the next label batch (Codex, GPT-5)

- Separated local media by purpose. `movies/` now contains only the three new source AVIs;
  `gif_videos/readme_demo/` contains one 90-row analysis CSV and the exact 90 overlays needed by
  the promoted README GIF; `frame_recommendations/` contains extracted candidates and label picks.
  Added all three local-data roots to `.gitignore` and updated the GIF generator's defaults.
- Proved the GIF cleanup boundary before moving anything: rebuilding from the retained CSV and
  overlays produced SHA-256 `FDB94655...E9449A`, byte-identical to both the tracked README GIF and
  its formerly selected local candidate. Updated the no-argument defaults to the actual 7,107-
  7,375 frame window at 5 fps and reproduced the same hash again after the move. The retained
  source is about 2.3 MB.
- The safety guard refused permanent deletion of the remaining historical `videos/` tree because
  it includes the old 430 MB `eye.avi` and full analysis runs, not only disposable GIF candidates.
  Its exact unused remainder is preserved pending explicit deletion confirmation: 10,152 files,
  0.844 GiB across `eye.avi`, two full eye-review runs, old candidates, two reconstructed demo
  workspaces, and the two duplicate result files left in `main_demo_warning`.
- Detected session leakage before scoring `HQL090_sleep251012_010_eye.avi`: fold models 0, 2, and
  3 trained on labeled frames from that session. Trained fold-1 seeds 1 and 2 and combined them
  with seed 0, giving HQL090 a three-member committee that excluded the session and outer holdout.
  HQL097 and HQL103 are new sessions and used the original four-fold seed-0 committee.
- Sampled 2,000 evenly spaced frames across each approximately 30-minute movie and selected 20
  unique frames per recording. Selected-score mean versus recording median was 0.786 vs 0.443 for
  HQL090, 0.783 vs 0.490 for HQL097, and 0.873 vs 0.563 for HQL103. Each set spans at least 92% of
  its recording. Visual spot checks found both strongly occluded/closed-eye negatives and visible
  pupil examples; the occlusions explain the top-ranked near-maximal disagreement.
- Verification: Ruff, Black, `git diff --check`, and the full test suite passed. Each recommendation
  folder has exactly 2,000 extracted PNGs, 20 unique picks, and 20 matching `selection.csv` rows.

### Harden grouped training and make the outer holdout genuinely one-shot (Codex, GPT-5)

- Replaced the leaking `--final` workflow. A final refit now trains every non-holdout pair for
  an epoch count and optional learning-rate milestones frozen from development runs, at a frozen
  prediction threshold. It creates no holdout Dataset or DataLoader and has no validation-driven
  scheduler, early stopping, threshold calibration, or best-checkpoint selection.
- Added `training/evaluate_holdout.py` as the separate one-shot gate. It verifies the checkpoint
  and manifest hashes, scores exactly the frozen threshold, records per-session and worst-session
  IoU, and refuses to overwrite `holdout.json`. Promotion accepts a final run only after that file
  exists and rejects mismatched or modified final artifacts.
- Changed checkpoint selection to macro IoU by default and kept validation loss as the scheduler
  signal. Promotion metadata now records the workflow, selector, scheduler, selection threshold,
  calibration grid, and split-manifest hash; the installed checkpoint metadata was backfilled
  honestly as an older development-selected run with no grouped-manifest hash.
- Hardened split and output handling. Changing the recorded fold count now requires `--reassign`;
  materialization never replaces the data root, a file, or a nonempty unmarked directory; forced
  frame recommendation removes stale generated PNGs/CSV while preserving unrelated files; and
  exact legacy copies retained after migration are not counted twice.
- Changed natural sampling and macro-IoU selection from measured recommendations into actual
  defaults. Updated the training/data-collection guides to distinguish development, final refit,
  and one-shot holdout evaluation. Added `sample_data/splits.json` to source distributions.
- Audited the local data before starting a real run. The visible legacy folders contain 166 + 56
  pairs, exactly matching all 222 entries already recorded in `splits.json`; no additional labelled
  pair is visible and the manifest has no holdout. Copied those pairs and their Labelme JSON files
  into the ignored 16-session `labeled_frames/` layout without moving the originals; all source and
  destination hashes match. A fresh census remains 222 images because retained exact legacy copies
  are ignored.
- Verification:
  - Ruff, Black, `git diff --check`, and the full 152-test suite passed. The console launcher test
    required adding the Conda Scripts directory inside the Python process because this desktop
    sandbox strips it at interpreter startup; the launcher itself and both `--help` checks pass.
  - A one-epoch sample-data smoke completed the final refit, one-shot holdout evaluation, and final
    promotion dry run. The smoke's metric is intentionally meaningless and was not treated as a
    model result.
  - Wheel and source-distribution builds passed. Both retain the packaged checkpoint, JSON, and
    training log; the sdist also contains `sample_data/splits.json` and the holdout evaluator.

## 2026-08-16

### Diagnose the transfer failure and ship frame recommendation (Claude, Opus 5)

- **Found why the failing sessions fail, and it is not what any summary statistic suggested.**
  Brightness correlates with per-session IoU at rho +0.02 and boundary contrast at **-0.19**, with
  the worst session holding *better* boundary contrast than its high-scoring near-twin. Looking at
  a prediction settled it in one image: the model segments the whole dark eye aperture instead of
  the pupil, predicting **4.8x and 7.8x** the labelled area at **p = 0.99 on every frame**, while
  the session that works in the same fold predicts 0.72x.
- **Withdrew the photometric-augmentation recommendation** made earlier the same day. Appearance
  jitter cannot teach a pupil-versus-aperture distinction, and `ColorJitter(brightness=0.2,
  contrast=0.2)` plus blur are already in the pipeline, so the proposal amounted to turning up the
  wrong knob. Scale jitter is present too and plausibly works against this failure by eroding the
  size prior. Recorded as settled so it is not re-opened.
- **Corrected a committed code comment** claiming a low calibrated threshold means the model
  under-segments. Fold 1 disproves it: the threshold sat on the floor because the 32-image session
  under-predicts and outvotes the other two, while the sessions that actually fail were
  over-predicting and needed the opposite. One global threshold cannot serve sessions whose errors
  point in different directions.
- **Shipped frame recommendation** (`training/recommend_frames.py`, `training/frame_selection.py`)
  taking a video or a frame folder and returning the frames worth labelling. Built the validation
  harness first: hide each session's labels, rank its frames, reveal them. Picks average IoU 0.31
  against 0.51 random, oracle floor 0.29 - 89% of the achievable gap, beating random in 10 of 10
  sessions. Frame extraction reuses `extract_selected_frames`, which already implemented the
  "sample at a rate, capped at N equally spaced" rule.
- **Ranking must not use model confidence.** The failure runs at p = 0.99, so entropy or margin
  sampling would rank exactly the wrong frames as least informative. Disagreement alone is also
  near-blind to it, because committee members share the bias and agree while all being wrong; the
  geometric plausibility prior is what recovers those frames (rho 0.09 -> 0.84 on the worst
  session).
- **Three defects the harness and tests caught before release.** The near-duplicate cutoff, set
  from a low percentile, collapses to zero and disables deduplication exactly when duplicates are
  common - a resting animal. A rewritten spacing check accepted frames adjacent to a pick.
  Combining geometric penalties by their maximum reads better than a mean and measures worse
  (89.2% -> 80.8%), so it was reverted with the measurement recorded in the docstring.
- **Established the sampling rate for size against position.** At 97 Hz the frame-to-frame size
  signal is segmentation noise: the 8.92% maximum on a 24 px pupil is 2.1 px, and a one-pixel
  boundary shift on a disc of radius 12 accounts for 8.3% alone. Size needs 10-30 Hz; position, at
  426% of a diameter per second, is what justifies the 97 Hz rig. So higher fps makes the temporal
  frame-selection signal *worse*, and that signal belongs inside the plausibility prior rather than
  as a separate weighted term.
- Verification:
  - `ruff check .`, `black --check` (35 files), `pytest` (`139 passed` with the environment's `bin`
    on `PATH`), including 13 new tests in `tests/test_frame_selection.py`.
  - End-to-end run against a 600-frame video built from real frames: 100 extracted, scored with a
    12-model committee, 8 promoted, output folders placed beside the video as specified. Confirmed
    deduplication takes the picks from 4 distinct source images to 8 of 8.
  - Both input modes, and the guards for a missing video, a single-checkpoint committee, and a
    populated output folder. Moved the overwrite check ahead of scoring, which on a long recording
    took minutes before refusing.

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
