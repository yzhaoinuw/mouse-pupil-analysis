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

Two consequences follow, and they drive everything below:

- **Frames from one session are near-duplicates of each other.** The 40th frame of a
  sleep recording teaches the model almost nothing the 39th did not. Its marginal value
  is far below that of the first frame from a session you have never labelled.
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

   The census prints images, tiny-mask count, and median diameter per session. If every
   `whiskerb` session has a large median diameter and every `sleep` session a small one,
   the model has never seen a constricted pupil under whisker-stimulation lighting.
3. **Small pupils from a session that is not already the small-pupil session.** Tiny
   masks are currently concentrated almost entirely in one session, which makes the
   tiny size bin a measure of that one recording rather than of a size regime. Small
   pupils from a *different* setting fix that; more from the same one do not.
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

## Filenames decide the grouping

Grouping is derived from the filename. `training/data_splits.py` recognises two schemes:

```text
HQL080_whiskerb_250722_007_eye_06622
└─────┘ └──────┘ └────┘ └─┘
 animal  condition date  recording index      -> session HQL080_whiskerb_250722

250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_0000
└────┘ └──┘ └───────────────────────────┘ └──────────────────────┘ └──┘
 date  animal      condition/label              timestamp          frame
                                              -> session 250530_5003_Green_Training_very_dm_light
```

The rule in both cases: **everything that identifies the setting comes before the part
that identifies the individual recording or frame.** The HQL scheme strips a trailing
2–4 digit recording index; the timestamped scheme strips the timestamp and everything
after it.

Keep new files in one of these two schemes. A stem matching neither is not silently
mis-grouped — it becomes its own single-image session under the `unparsed` cohort, and
`--show` prints a warning listing every such name. That is safe but wasteful: a batch of
unparsed frames becomes a batch of one-image sessions that can never be grouped
correctly. Fix the names instead of training around them.

If a naming scheme genuinely changes — a new rig, a new acquisition program — add a
branch to `parse_identity` in `training/data_splits.py` and a case to
`tests/test_data_splits.py` rather than renaming existing files, which would silently
reassign historical sessions to different folds.

## Adding a labelled batch to the split

1. Put images and masks in `images_train/` and `masks_train/` as usual. **Which folder
   they land in no longer decides the split** — the manifest does. `images_train/` and
   `images_validation/` are read as one flat pool, so both folders are equivalent and
   exist only for historical reasons.
2. Regenerate the manifest:

   ```bash
   python training/data_splits.py --data-root . --out splits.json
   ```

   Sessions already in `splits.json` keep the fold they had. Only genuinely new sessions
   are packed, into whichever folds are currently smallest. This is what keeps a
   cross-validation number from six months ago comparable to one from today.
3. Read the census it prints before training. Check that the new sessions landed in
   sensible folds, that no `unparsed` warning appeared, and that fold sizes are still
   roughly even.
4. Re-run cross-validation ([`README.md`](README.md#5-cross-validate-a-configuration)).

**Never pass `--reassign` casually.** It repacks every session from scratch, which
invalidates comparison against every previously recorded run. It is correct only when
changing `--folds`, or after so much new data has arrived that the old packing is badly
unbalanced — and in that case, say so in the work log and treat prior numbers as a
different experiment.

## What the split does not protect against

- **Label quality.** Grouping guards against optimistic evaluation, not wrong masks.
  Compare filenames and counts between image and mask folders after every conversion.
- **A session that is internally heterogeneous.** If the camera was physically moved
  mid-recording, one session key covers two settings and the split under-groups. Split
  such a recording into two differently-named sessions by hand.
- **Failures validation cannot see.** Three of five from-scratch runs in the noise-floor
  study lost a real small pupil that no validation image covered, and those were the
  runs with the *highest* IoU. `reports/scripts/hard_frame_check.py` exists for exactly
  this and remains a promotion gate regardless of what cross-validation reports.
