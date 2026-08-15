# Choosing And Grouping New Labelled Data

This document answers two questions that come up every time a new batch of recordings
arrives: **which frames are worth labelling**, and **how the labelled result joins the
split**. The mechanics of running Labelme and the trainer are in
[`README.md`](README.md); this is the policy that sits above them.

Read this before labelling a batch. Labelling is the expensive step and the one that
cannot be undone cheaply — a hundred frames from one recording cost the same effort as
a hundred frames spread across twenty, and are worth far less.

## The unit that matters is the session, not the frame

A **session** is one recording setting: one animal, on one date, under one condition.
It is the grouping unit throughout this project because the domain shift that breaks
this model in practice is rig, camera placement, lighting, and animal state — not
animal identity. Mouse eyes are effectively interchangeable; camera angles are not.

That is not an assumption. It was measured on the 222-image pool: take each image,
find its most similar neighbour, and copy that neighbour's mask over as the prediction.

| nearest neighbour drawn from | mean IoU |
| --- | --- |
| the same session | 0.652 |
| a different session | 0.399 |
| a different session, matched on pupil **size** instead | 0.434 |

A model that does nothing but recognise the setting and recall what it saw there gains
0.25 IoU, against a seed noise floor of 0.02. The third row is the control that matters:
matching pupil size across sessions recovers almost none of it, so what makes
same-session frames easy is the *setting*, not the pupil. Stratifying on size does not
substitute for grouping, and grouping does not substitute for stratifying.

Two consequences drive everything below:

- **Frames from one session are near-duplicates for this purpose.** The 40th frame of a
  sleep recording teaches the model little the 39th did not, even though the pupil has
  moved — within-session diameter spans up to 8.5x, and it still does not help.
- **A session must never span the train/validation boundary.** If it does, the reported
  IoU measures interpolation within a setting the model has already seen. That was true
  of every number this project reported before 2026-08-14; see
  [`../reports/2026-08-14-checkpoint-noise-floor.md`](../reports/2026-08-14-checkpoint-noise-floor.md).

## Which frames to label

In priority order. Spend the budget top-down.

1. **A session you have no labels from at all.** One new setting beats twenty more
   frames of an existing one. This is the single highest-value label you can add, and
   it stays true until the pool covers every rig and condition in routine use.
2. **Conditions that are under-represented rather than animals that are.** Check what
   the pool actually holds before labelling:

   ```bash
   python training/data_splits.py --data-root . --show
   ```

   The census prints images, tiny-mask count, median diameter, and median background
   brightness per session. If every `whiskerb` session has a large median diameter and
   every `sleep` session a small one, the model has never seen a constricted pupil under
   whisker-stimulation lighting.
3. **Small pupils from a session that is not already the small-pupil session.** Only
   four sessions in the pool contain any mask at or below 15 model pixels, and one holds
   10 of the 14. Stratification spreads what exists across folds; it cannot manufacture
   small pupils that were never labelled. Every fold currently holds at least one, but
   three of them hold only one or two, so the tiny-bin IoU is still thin. Small pupils
   from a *different* setting fix that; more from the same one do not.
4. **Frames where the packaged model visibly fails.** Run inference first and label what
   it gets wrong. Frames it already segments correctly mostly confirm what the weights
   encode.
5. **More frames from a well-covered session.** Genuinely last. Ten sessions at eight
   frames each beat one session at eighty.

Within a session, prefer frames that are far apart in time and different in pupil size
over consecutive frames. Consecutive frames at 97 Hz are the same image.

**How many per session?** Roughly 8–15 is the useful range. Below about 5 the session
contributes little and makes a noisy validation fold when it is held out; above about
20 it starts to dominate whichever fold holds it. The 62-image `5003` session is 28% of
the entire pool and forces one fold to be far larger than the others — do not repeat it.

## Recording which session a frame came from

**The session has to be recorded when the batch is created. It cannot be recovered
afterwards.** This is the part that most often gets skipped, so it is worth being blunt
about why there is no way around it.

Three methods were tried against the 222 images whose sessions were already known, to
see whether the grouping could be inferred instead of recorded:

| method | result |
| --- | --- |
| crop geometry | 6 of 16 sessions span 2–3 different crop boxes |
| masked-thumbnail correlation, connected components | no threshold gives usable groups without tearing sessions |
| agglomerative clustering on preprocessed frames | 3 sessions torn at k=5, 6 torn at k=10 |
| file mtime | uniform — destroyed by copying |

Tearing a session across clusters is the dangerous direction: it is exactly the leak the
grouping exists to prevent. Loosening any of these rules until nothing tears collapses
the pool into two or three groups, which is too few to build folds from. The images are
tight crops around an eye, so the cage, headplate and rig framing that would fingerprint
a recording have been cropped away before anything sees them.

Filenames are not the answer either. They happen to encode the session today, but a
parser is one new acquisition program away from silently regrouping the pool.

So: **record it**. Any one of these works, and they are checked in this order.

1. **An intake subfolder** — the simplest, and it asks nothing of how files are named.
   Drop each recording's frames in their own directory and the directory *is* the
   session:

   ```text
   labeled_data/HQL091_sleep260820/frame_0001.png   ->  session HQL091_sleep260820
   labeled_data/whatever_you_like/img_00042.png     ->  session whatever_you_like
   ```

   Masks may mirror the same subfolders or sit flat in `labeled_masks/`; both are paired
   by filename.

   **Filenames need not be unique between folders.** Two recordings may each contain a
   `frame_0001.png`, which is what per-recording exports usually produce. An image is
   identified by its path *within* its pool folder — `rig2_day3/frame_0001`, not
   `frame_0001` — so intake folders keep them apart.

2. **A labelme flag** — set `session` once per batch in the Labelme UI and it lands in
   the `flags` block of each `<image name>.json`, beside the image. Use this when the
   files arrive already mixed into one folder.

   ```json
   { "flags": { "session": "HQL091_sleep260820" }, "shapes": [...] }
   ```

3. **A sidecar** — `provenance.csv` (columns `key,session`) or `provenance.json`
   (`{key: session}`) at the data root, where a *key* is the identifier described
   above: `frame_0001` for a flat image, `rig2_day3/frame_0001` for a nested one. This
   is the durable record: it is committed, unlike the image folders, so it is the only
   part of the grouping that survives a fresh clone. The migration off filename grouping
   wrote the current one.

4. **Nothing at all** — every unresolved image in a run collapses into a *single* group.
   Over-merging two settings only costs data efficiency; tearing one apart leaks, so
   this is the safe failure and it needs no human input. The census prints a `NOTE`
   naming any such group.

   The cost is real and is the reason to bother with 1–3: the whole batch lands in one
   fold, exactly like the oversized `5003` session. A pool with *no* provenance anywhere
   becomes one group and cannot be folded at all, which is an error rather than a
   silently bad split.

   Two unknown batches become two groups. If you have reason to think they share a
   recording, pass the same `--batch-name` to both and they merge.

## Adding a labelled batch to the split

1. Put images and masks under `labeled_data/` and `labeled_masks/`, in a per-recording
   subfolder if you can. There is one flat pool and no train/validation folders: the
   manifest decides what trains and what validates, so adding data never moves a file
   and re-splitting never moves one either.
2. Regenerate the manifest:

   ```bash
   python training/data_splits.py --data-root . --out splits.json
   ```

   Every image already in `splits.json` keeps the session and fold it had. Only
   genuinely new sessions are packed. A new image belonging to an existing session
   inherits that session's fold. This is what keeps a cross-validation number from six
   months ago comparable to one from today.
3. Read the census it prints before training. Check that the new sessions landed in
   sensible folds, that no batch-fallback `NOTE` appeared unexpectedly, and that fold
   sizes and strata are still roughly even.
4. Re-run cross-validation ([`README.md`](README.md#5-cross-validate-a-configuration)).

**If a provenance source now disagrees with what the manifest already recorded, that is
an error, not a repack.** The manifest wins, and the command fails naming the images
involved. Fix the source, or accept the change deliberately with `--reassign`.

**Never pass `--reassign` casually.** It repacks every session from scratch, which
invalidates comparison against every previously recorded run. It is correct only when
changing `--folds`, when correcting genuinely wrong provenance, or after so much new
data has arrived that the old packing is badly unbalanced — and in that case, say so in
the work log and treat prior numbers as a different experiment.

## How folds are balanced

Grouping alone is not enough. The first grouped split of this pool put three of five
folds with no small pupil at all and left a 3x spread in median diameter across folds,
which made fold-to-fold variance mostly a story about which size regime happened to land
where rather than about the model.

Sessions are therefore banded by median diameter and median background brightness
(terciles, recorded as `stratum_cutpoints` in the manifest). A new session prefers a
fold that holds *no* session of its diameter band; failing that, the smallest fold.
Diameter leads because it is the axis the evaluation reports size bins over.

Only the **absence** of a band outranks fold size, and that detail is what lets one rule
serve both regimes. When the folds are packed from scratch they start empty, coverage
dominates, and that is exactly when it is needed — ordering by size alone there gives a
4.51x spread in median diameter and leaves 3 of 5 folds with no small pupil. Once each
fold holds a few sessions the bands are covered, coverage stops firing, and size takes
over, which is what keeps folds even as sessions trickle in one at a time:

| rule, 10 new sessions arriving one at a time | median size spread | worst case |
| --- | --- | --- |
| rank by band count throughout | 1.33x | 2.05x |
| rank by absence of band, then size | 1.15x | 1.25x |

Band coverage is identical either way (5 of 5 folds hold all three diameter bands), so
this costs nothing. Simulated over 200 arrival orders from the current assignment.

Fold sizes are uneven — currently 76/58/44/44 over four folds — and most of that is not
fixable by any rule: the 62-image `5003` session cannot be divided, so one fold is at
least 62 images whatever you do. The rest is the granularity of the remaining session
sizes. It self-corrects as data arrives; five new sessions bring the spread to about
1.35x.

The pool is split into **four** folds rather than five. Four holds a small pupil in every
fold and gives a tighter condition balance, because each fold holds more sessions and the
indivisible 62-image session is a smaller share of a larger target:

| | 5 folds | 4 folds |
| --- | --- | --- |
| folds containing any mask ≤15 px | 4 of 5 | **4 of 4** |
| median diameter spread across folds | 1.78x | 1.60x |
| fold size spread | 1.91x | 1.73x |

The cost is that each model trains on 75% of the pool instead of 80%, and four numbers
are averaged instead of five. At this pool size that is the better trade; revisit it if
the pool grows enough that five folds also cover every size band.

|  | before | after |
| --- | --- | --- |
| median diameter spread across folds | 3.03x | 1.60x |
| folds containing any mask ≤15 px | 2 of 5 | 4 of 4 |

Brightness is measured as the mean grey level outside the pupil mask, on the original
image rather than the padded model input — `resize_with_pad` fills with black, so a
padded mean would encode the crop's aspect ratio as if it were lighting. Across the pool
session means span 71 to 157 while the spread *within* a session is 1 to 11, which is
what makes it usable as a lighting axis.

## The holdout gate

Cross-validation chooses configurations. Because every fold's number feeds that choice,
none of them is a clean estimate of the final model — leakage and overfitting to the
selection both push the same way.

A **holdout** is one or more sessions set aside entirely: in no fold, trained on never,
validated on never.

```bash
python training/data_splits.py --data-root . --holdout HQL090_sleep251012 --out splits.json
python training/run_train.py --split-manifest splits.json --final
```

`--final` trains on every non-holdout image and validates on the gate sessions. It is
the only number in the project measured against data the training procedure was never
tuned on.

**Choose the holdout by condition, not by animal.** Holding out a mouse tests animal
generalisation, which the measurements above say is not the axis that breaks this model.
Hold out a lighting regime, a rig, or a recording day instead.

Note the cost before you do it: at 222 images, two typical sessions is 15–27% of the
pool, and that data trains nothing. The pool is currently small enough that this is a
real trade, which is why no holdout is set by default.

## What the split does not protect against

- **Label quality.** Grouping guards against optimistic evaluation, not wrong masks.
  Compare filenames and counts between image and mask folders after every conversion.
- **A session that is internally heterogeneous.** If the camera was physically moved
  mid-recording, one recorded session covers two settings and the split under-groups.
  Split such a recording into two differently-named sessions by hand — this is the one
  case where the recorded provenance is wrong and nothing will catch it.
- **Failures validation cannot see.** Three of five from-scratch runs in the noise-floor
  study lost a real small pupil that no validation image covered, and those were the
  runs with the *highest* IoU. `reports/scripts/hard_frame_check.py` exists for exactly
  this and remains a promotion gate regardless of what cross-validation reports.
