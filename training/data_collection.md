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
   python training/prepare_splits.py --show
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

Closed or fully occluded eyes can be high-value negative examples: the current failure mode
mistakes the dark eye aperture for a pupil. If no pupil boundary is actually visible, represent
the target as an all-black mask rather than outlining the aperture. Keep these negatives in a
mixed batch with visible pupils; do not spend the whole session budget on eye closures.

Use nearby frames to decide what is visible in the target frame. If the sequence establishes that
the pupil has fully disappeared from view, label that frame `no_visible_pupil` even when the
isolated dark eye aperture could be mistaken for one very large pupil. Sequence context improves
the annotation; it does not make the aperture a pupil target. Conversely, outline a genuinely
large pupil only when its boundary is visible in the target frame. Never fill the whole dark eye
aperture merely because it could represent maximal dilation.

In Labelme, mark that explicit negative with one small `no_visible_pupil` shape; its geometry is
ignored by `labelme_json2png.py`. If a pupil may be present but low contrast or occlusion makes
the boundary unreliable, use one `uncertain` shape instead. An uncertain annotation creates no
segmentation mask and must remain outside the session's training `images/` directory. Do not turn
annotation uncertainty into an all-black target: that would teach a confident false negative.
This includes transition frames where even the sequence cannot establish the exact point of
disappearance. Because the current model sees one frame at a time, retain representative examples
of both confirmed no-visible-pupil frames and genuinely large visible pupils; if only temporal
context can separate two otherwise indistinguishable appearances, that is a model limitation, not
a reason to change either correct label. Avoid flooding the batch with consecutive near-duplicate
negatives.

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

So: **record it**. And the layout records it for you.

### The session is a directory

```text
labeled_frames/
  HQL091_sleep260820/          <- one recording session
    images/  frame_0001.png    the frame, and its .json annotation
    masks/   frame_0001.png    the mask, same filename
  whatever_you_like/
    images/  img_00042.png
    masks/   img_00042.png
```

This is the whole mechanism. An image cannot enter the pool without landing in a session
folder, so provenance is a consequence of where the file goes rather than a convention
anyone has to remember, and the grouping can never disagree with the directory it sits in.
It asks nothing of how files are named.

**Do not infer a shared session from a partial filename.** For example, two paths that both
contain `Purple_trial5` may still be different recordings. Give them distinct session names
unless the intake record establishes that they are one session. The importer deliberately
refuses to merge a batch into an existing session directory.

**Filenames need not be unique between sessions.** Two recordings may each contain a
`frame_0001.png`, which is what per-recording exports usually produce. An image is
identified by `<session>/<filename>` — `HQL091_sleep260820/frame_0001` — so the session
folders keep them apart.

The whole split is reproducible from this layout alone: delete `training_data_split.json` and
regenerate, and the folds come back identical, because the sessions come from the
directories and the packing is deterministic. A mixed or unassigned batch is not valid
split input: sort it into its recording-session directory before importing it.

## Adding a labelled batch to the split

1. Keep each Labelme JSON beside its source image, then import the new session:

   ```bash
   python training/labelme_json2png.py --source <annotation-folder> --session <new-session>
   ```

   It validates all labels, image references, frame indices, and destinations together before
   writing. It refuses to overwrite an existing session, then refreshes the frozen split.
   `pupil` and `no_visible_pupil` become compact image/mask pairs. `uncertain` image/JSON
   pairs are archived under the session's `uncertain/` directory, outside the segmentation
   pool, and receive no mask or current training loss. Every image already in
   `training_data_split.json`
   keeps its fold; only the genuinely new session is packed. There are no train/validation
   source folders: the manifest decides what trains and validates, so re-splitting moves no
   labelled file.
2. Read the census it prints before training. Check that the new session landed in
   sensible folds and that fold sizes and strata are still roughly even.
3. Re-run cross-validation ([`README.md`](README.md#5-cross-validate-a-configuration)).

**Never pass `--reassign` casually.** It repacks every session from scratch, which
invalidates comparison against every previously recorded run. It is correct only when
changing `--n_folds` or after so much new data has arrived that the old packing is badly
unbalanced — and in that case, say so in the work log and treat prior numbers as a
different experiment.

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

## Cross-validation and all-labeled training

Cross-validation chooses a reusable training configuration. It writes a JSON recipe containing
the median successful-fold epoch and calibrated threshold, together with the fixed learning-rate
schedule and other training settings. Use that recipe to train a production model on the complete
labelled pool:

```bash
python training/run_cross_validation.py --checkpoint_dir checkpoints_exp/cv
python training/run_train.py \
    --training_config_path checkpoints_exp/cv/training_config.json \
    --checkpoint_dir checkpoints_exp/all_labeled
```

The second command ignores `training_data_split.json` and trains on every valid image/mask pair under
`labeled_frames/`. It is intentionally a trusted all-data path: inspect the resulting model on
representative unlabeled recordings before promoting it.

`--final_test_session` remains useful for keeping a difficult condition out of CV while you
compare configurations. The all-labeled training command deliberately includes it once you decide
to build the production model, so it is no longer an independent final-test gate.

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
