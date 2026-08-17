# What does this model score on recordings it has never seen?

**Date:** 2026-08-16
**Data:** the maintained local pool, 222 pairs in 16 sessions, 4 stratified grouped folds (`splits.json`)
**Compute:** Apple M4 via MPS; ~9 min for the first sweep, ~15 min per repaired sweep,
seven sweeps (28 fold-trainings) in ~1 h 40 m total

## Summary

This is the first measurement of generalisation this project has produced. Every
number published before it scored held-out *frames* from recordings that also
supplied training images.

- **Cross-recording IoU is far below the published figure.** Mean per-session IoU is
  **0.5891 ± 0.0273** over three seeds, against the 0.8749 macro IoU carried by the
  packaged checkpoint. The image-weighted number, which is what the old macro IoU is
  comparable to, is 0.51–0.58.
- **The first sweep was contaminated by its own checkpoint selector**, which returned
  0.5378. Under `balanced_iou`, three of four folds selected a checkpoint at epoch
  4–6 of 400 on a one-off spike in a size bin holding one to three images.
- **The damage was not only mis-selection.** The same metric drove
  `ReduceLROnPlateau`, so the spike became a high-water mark no later epoch could
  clear, and the learning rate decayed to near its floor while the model was still
  improving. Folds were mis-selected *and* under-trained.
- **Seed noise on the grouped split is roughly 4x the previously measured floor.**
  The ±0.0069 in `2026-08-14-checkpoint-noise-floor.md` was measured on the leaky
  split. Here the seed sd is 0.0273 on the 3-seed mean and up to 0.0873 on a single
  fold.
- **Natural sampling beats size-balanced sampling**, by 0.0354 mean per-session IoU.
  It wins at all three seeds and in 8 of 12 matched (fold, seed) cells. Balanced
  sampling does not reliably improve the tiny bin it exists to protect, and it
  costs medium and large accuracy. The best honest figure is therefore the natural
  arm's **0.6245 ± 0.0322**.

## 1. The selector defect

Selection ran on `balanced_iou`, the equal-weighted mean of the tiny/medium/large
bin IoUs. Under grouped folds a bin can hold one to three validation images, so one
noisy image moves a third of the metric. The tiny bin is never actually learned —
its last-10-epoch mean is 0.02–0.17 in every fold — but it spikes:

| fold | tiny min | tiny max | tiny, last 10 epochs |
|---:|---:|---:|---:|
| 0 | 0.0233 | 0.6793 | 0.0707 |
| 1 | 0.0000 | 0.2351 | 0.1695 |
| 2 | 0.0000 | 0.5952 | 0.0226 |
| 3 | 0.0142 | 0.6792 | 0.0587 |

What those spikes selected, against the best macro IoU the same run reached:

| fold | selected epoch | its macro | its tiny | best macro epoch | best macro | its tiny |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 0.2819 | **0.6793** | 31 | 0.4072 | 0.0567 |
| 1 | 5 | 0.5119 | 0.0345 | 10 | 0.5149 | 0.0952 |
| 2 | 4 | 0.2065 | **0.5952** | 12 | 0.3304 | 0.0659 |
| 3 | 74 | 0.7501 | 0.6339 | 74 | 0.7501 | 0.6339 |

Folds 0 and 2 each discarded a checkpoint roughly 0.12 macro IoU better. Fold 3,
whose tiny bin was stably informative, selected correctly.

### The learning-rate cascade

`scheduler.step(report.balanced_iou)` on `mode="max"` benchmarks against the best
value ever seen. After the epoch-6 spike, every later epoch scored as a plateau:

```
fold 0:  ep1 lr 1e-3 -> ep19 5e-4 -> ep25 2.5e-4 -> ep37 1.25e-4 -> ep43 6.25e-5
```

`min_lr` is 3.125e-5, so the run was one step off the floor when early stopping
fired at epoch 46. Validation loss over the same run fell 0.5881 → 0.2497 with a
clean minimum at epoch 31 — the smooth signal located the right region while the
noisy one did not.

A third compounding factor: `evaluate_thresholds` re-picked the threshold each
epoch by maximising over 11 candidates, so every epoch's score was already a
maximum over 11 draws before the epoch maximum was taken on top of it.

## 2. The repair

In `training/run_train.py`, forwarded through `run_cv.py`:

1. `ReduceLROnPlateau` now runs on `val_loss` (`mode="min"`). New default;
   `--scheduler-metric` overrides.
2. `--selection-metric {balanced_iou,macro_iou}` chooses what "best" means, and also
   ranks threshold candidates. Default unchanged at `balanced_iou` so earlier runs
   stay reproducible.
3. `--selection-threshold` (default 0.5) compares epochs at one fixed threshold;
   `calibrated` restores the old max-over-candidates behaviour. The winning epoch's
   metadata is fully calibrated either way, so the shipped threshold is unaffected.

Also added: `--threshold-candidates`, and a warning when the calibrated threshold
lands on the edge of the grid.

## 3. Results

Sweeps at `--selection-metric macro_iou --scheduler-metric val_loss
--selection-threshold 0.5`, seeds 0/1/2, 400 epochs.

| | mean per-session IoU | image-weighted |
|---|---:|---:|
| old selector, seed 0 | 0.5378 | 0.4198 |
| repaired, seed 0 | 0.6146 | 0.5760 |
| repaired, seed 1 | 0.5923 | 0.5620 |
| repaired, seed 2 | 0.5603 | 0.5080 |
| **repaired, 3-seed mean** | **0.5891 ± 0.0273** | |

Selected epochs moved from 4–6 to 10–143. Per fold:

| fold | repaired macro IoU | selected epochs | old selector |
|---:|---:|---|---:|
| 0 | 0.6577 ± 0.0422 | 78, 115, 15 | 0.2819 @ ep6 |
| 1 | 0.4478 ± 0.0873 | 10, 17, 143 | 0.5119 @ ep5 |
| 2 | 0.3479 ± 0.0477 | 47, 23, 20 | 0.2065 @ ep4 |
| 3 | 0.6939 ± 0.0099 | 71, 69, 53 | 0.7501 @ ep74 |

**The comparison is not uniformly favourable.** Folds 0 and 2 — the mis-selected
ones — gained 0.38 and 0.14. Folds 1 and 3, whose old selection was sound, moved
down. Fold 3's drop is 5.7 sd of its own seed spread, so selecting at a fixed 0.5
threshold has a real cost where the calibrated-threshold selection was already
working. Worth revisiting if fold 3's pattern recurs.

### Per session, mean over three seeds

| session | IoU | sd |
|---|---:|---:|
| HQL086_whiskerb250923 | 0.2290 | 0.0073 |
| HQL090_sleep251012 | 0.3015 | 0.1348 |
| 251016_5212_purple_Day10 | 0.3520 | 0.1609 |
| HQL073_250515 | 0.3800 | 0.0677 |
| HQL080_whiskerb_250722 | 0.3866 | 0.1007 |
| HQL080_sleep250625 | 0.4415 | 0.0453 |
| 250616_5120_Purple_sleep_trial_1 | 0.5462 | 0.2632 |
| HQL086_sleep250909 | 0.5648 | 0.1776 |
| HQL086_sleep250912 | 0.6370 | 0.1958 |
| 250530_5003_Green_Training_very_dm_light | 0.6752 | 0.0265 |
| HQL088_sleep250929 | 0.6940 | 0.0543 |
| HQL073_whisker250623 | 0.7560 | 0.0206 |
| HQL085_whiskerb250916 | 0.7915 | 0.0370 |
| HQL090_whiskerb251020 | 0.8631 | 0.0155 |
| 251018_5213_Purple_awake pupil recording | 0.8991 | 0.0031 |
| HQL088_whiskerb251006 | 0.9076 | 0.0265 |

Performance is bimodal: six sessions below 0.45, six above 0.75. The model either
transfers to a recording setting or largely fails on it.

**`HQL086_whiskerb250923`'s 0.2290 ± 0.0073 is an artefact of this sampling arm, not
a property of the session.** Under natural sampling the same session scores 0.4942.
Its low seed variance made it look like an intrinsic failure; it is a
balanced-sampling failure. See section 7.

## 3a. Sampling: natural beats size-balanced

Same folds, same seeds, same threshold grid, differing only in `--natural-sampling`,
so all 12 (fold, seed) cells are matched pairs. The grid censoring in section 4
applies equally to both arms and cannot manufacture a difference between them.

| | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---:|---:|---:|---:|---:|
| balanced | 0.6146 | 0.5923 | 0.5603 | 0.5891 | 0.0273 |
| natural | 0.6617 | 0.6060 | 0.6058 | **0.6245** | 0.0322 |
| difference | −0.0471 | −0.0137 | −0.0455 | −0.0354 | 0.0189 |

Natural wins at every seed, and in **8 of 12** matched cells (mean paired difference
−0.0492, sd 0.0714). By fold, balanced loses badly on folds 0 and 2 and wins
narrowly and consistently on fold 3 (+0.0166, +0.0100, +0.0213).

### The rationale for balancing does not survive contact with the size bins

Equal-mass sampling exists to protect small pupils. It does not:

| fold | arm | tiny | medium | large |
|---:|---|---:|---:|---:|
| 0 | balanced | 0.3827 | 0.6367 | 0.7525 |
| 0 | natural | **0.6662** | **0.7245** | 0.7512 |
| 1 | balanced | **0.1321** | 0.4403 | 0.5716 |
| 1 | natural | 0.0406 | 0.4422 | **0.7717** |
| 2 | balanced | 0.0752 | 0.4079 | 0.1772 |
| 2 | natural | **0.1130** | **0.4781** | **0.4483** |
| 3 | balanced | **0.1079** | 0.8561 | 0.8946 |
| 3 | natural | 0.0693 | 0.8452 | 0.8899 |

The tiny bin splits 2-2, and its largest single gap (fold 0, 0.2835) favours
*natural*. Meanwhile balancing costs large-pupil accuracy heavily on folds 1 and 2
(−0.2001 and −0.2711). It pays a real price for a benefit it does not deliver.

**A prediction that failed.** Fold 3 holds out `HQL080_sleep250625`, leaving about
four tiny masks in training, which balancing inflates to a third of the training
mass. That was the predicted worst case for balancing. It is instead the only fold
where balancing consistently wins, though by only ~0.015 against a seed sd of
0.0099. The oversampling-collapse hazard is not what the data shows; the cost of
balancing appears in the folds with *ordinary* tiny counts.

Natural is also the arm that ships: the packaged checkpoint records
`sampling: "natural"`.

## 4. Threshold calibration was censored, and fixing it changed nothing important

The calibrated threshold hit the edge of the 0.30–0.80 grid in 5 of 12 balanced folds
and 4 of 12 natural folds. **Fixing this required no retraining.** Selection ran at a
fixed 0.5 threshold, so the grid cannot affect which epoch won — only the threshold
reported afterwards. Re-calibrating all 24 saved checkpoints over 0.05–0.95 is pure
inference and took about two minutes.

| | mean macro IoU, 0.30–0.80 | over 0.05–0.95 | change |
|---|---:|---:|---:|
| balanced | 0.5368 | 0.5450 | +0.0082 |
| natural | 0.5860 | 0.5916 | +0.0056 |

The threshold moved outside the old grid in **7 of 24** runs. The effect is small and
lands on both arms about equally, so the sampling verdict is unchanged: natural still
leads by 0.0466 after re-calibration.

**The censoring was concentrated in fold 1**, whose models want unusually low
thresholds — balanced seed 2 moved 0.30 → 0.05 (+0.0934 macro IoU), natural seed 2
moved 0.30 → 0.10 (+0.0498). A model needing a 0.05 threshold is producing very
diffuse, low-confidence masks. Fold 1 holds two of the three sessions that transfer
worst, so this is a symptom of the transfer failure rather than a calibration quirk.

The grid should still be widened in `TrainingConfig` so future runs are not censored,
but no conclusion in this report depended on it.

## 4a. No release candidate came out of this

None of these 24 models is shippable, by construction. Each trained on three folds —
about three quarters of the pool — and cross-validation exists to compare
*configurations*, not to produce weights. `run_cv.py`'s own docstring says so: once a
configuration wins, it has to be retrained on the whole pool and gated with
`reports/scripts/hard_frame_check.py`.

Nor are the numbers comparable to the shipped checkpoint's 0.8749. That figure is
interpolation on a leaky split; 0.6245 is cross-recording generalisation. They measure
different tasks, and the drop is not evidence that these models are worse.

**That final step is currently blocked.** `run_train.py --final` trains on everything
except a designated gate holdout, and `splits.json` records `n_holdout_sessions: 0`:

```
ValueError: This manifest sets no holdout, so there is nothing to gate against.
Regenerate it with --holdout SESSION, choosing by condition rather than by animal.
```

So the project cannot presently produce a gated release candidate at all. Choosing a
holdout session is the prerequisite for any future promotion, and it costs 15–27% of
the pool, which is why none was set.

## 4b. Why the failing sessions fail: the model segments the eye, not the pupil

No summary statistic explains it. Brightness correlates with per-session IoU at rho
+0.02, boundary contrast at **-0.19** — and `251016_5212_purple_Day10` (0.295) has
*better* boundary contrast than `251018_5213` (0.894), 0.85 against 0.71. Pupil size
correlates most (rho +0.55) but that is a consequence, not a cause.

Looking at a prediction answers it immediately. Predicted area divided by labelled area,
fold 1 checkpoint at its calibrated threshold:

| session | frames | label px | pred px | ratio | IoU |
|---|---:|---:|---:|---:|---:|
| HQL090_sleep251012 | 13 | 528 | 2143 | **7.8x** | 0.243 |
| 251016_5212_purple_Day10 | 13 | 1289 | 4430 | **4.8x** | 0.288 |
| 250616_5120_Purple_sleep_trial_1 | 32 | 3361 | 2475 | 0.72x | 0.684 |

The model outputs **p = 0.99 on every frame** while predicting five to eight times too
much area. It has learned "dark blob = pupil", which holds while the pupil is the only
dark thing in frame and breaks on sessions where the whole eye aperture is dark.

**This kills the photometric-augmentation plan.** Brightness and contrast jitter cannot
teach the difference between a pupil and an eye aperture — that is a semantic
distinction, not an appearance one — and `ColorJitter(brightness=0.2, contrast=0.2)`
plus Gaussian blur are already in `augmentation.py`, so the change would have amounted
to turning up a knob aimed at the wrong variable. Scale jitter is already present too,
and plausibly works *against* this failure by eroding the size prior that would rule out
a region five times too large.

What the evidence supports instead: labels from sessions where the whole aperture is
dark, and the two-stage eye-ROI model previously parked as speculative.

## 4c. Sampling rate: size and position are different problems

Measured on `sample_data/velocity_frames`, 31 consecutive frames at 97 Hz, packaged
checkpoint: mean diameter 24.0 px, frame-to-frame size change median 0.97% and max
8.92%, position step median 0.29 px and max 1.05 px.

The 8.92% maximum is **not** physiology. On a 24 px pupil that is 2.1 px of diameter,
and a one-pixel shift of the mask boundary on a disc of radius 12 changes the
area-equivalent diameter by 8.3%. The whole maximum is single-pixel boundary jitter, so
at 97 Hz the size signal measures segmentation noise.

This inverts the usual intuition: **for the temporal signal, higher fps is worse.**
Jitter is roughly constant per frame while real change shrinks with the interval. At
97 Hz signal and noise are comparable; at 5 fps a physiological ~100%/s gives ~20% per
interval against the same ~1% jitter.

Published mouse work samples pupil size at 10-50 Hz, and one sleep-staging study
classifies stages from diameter at 10 Hz, which bounds size bandwidth well under 5 Hz.
No reliable mouse maximum constriction velocity was found; the mm/s figures in the
literature are human and do not transfer across a 2-8 mm versus 0.5-2 mm range.

Position is the fast quantity — 1.05 px/frame is 426% of a pupil diameter per second —
and is what justifies the 97 Hz rig. For size alone the recording is oversampled about
tenfold.

**Consequence for frame selection.** A physiological ceiling turns the temporal signal
into a plausibility bound: at interval dt, a diameter change beyond roughly
`rate x dt` plus a jitter allowance is a model error, needing no labels. It therefore
belongs inside `implausibility_score` rather than as a separately weighted term. That it
*ranks* frames usefully still needs consecutive labelled frames to establish.

## 5. What this does not establish

- **Nothing here judges the packaged checkpoint.** It gradient-trained on 166 of
  these 222 images, and its own 56-image validation set drew 54 of 56 from
  recordings that also fed training. At session granularity it has seen every
  session, so no fold's holdout is novel to it and fine-tuning from it inside CV
  leaks. Only a from-scratch configuration is measurable this way.
- **Three seeds bound the seed noise loosely.** sd 0.0273 from n=3 is itself
  imprecise, and the sampling verdict rests on a 0.0354 difference against it. The
  sign is consistent across every seed and most cells, which is the strength of the
  evidence; the magnitude is not tightly bounded.
- **No promotion follows from any of this.** See section 4a: these are fold models,
  and the gated whole-pool path is blocked on an undesignated holdout.

## 6. Follow-ups

Ordered by value per unit of compute.

- **Widen the default threshold grid** in `TrainingConfig` to about 0.05–0.95. Done
  post-hoc for the runs above (section 4); making it the default costs nothing.
- **Change `balance_training_sizes` to default `False`.** It currently defaults to
  `True`, which this evidence does not support, and the shipped checkpoint already
  uses natural. A one-line default change is the whole fix.
- **Diagnose the transfer failures before trying to fix them.** `HQL090_sleep251012`
  (0.1865 natural) and `251016_5212_purple_Day10` (0.2952) fail under both arms, and
  fold 1 — which holds both — is also where thresholds collapse toward 0.05. Compare
  their brightness, contrast, and pupil-size distributions against the pool. Pure
  analysis, no training, and it tells the augmentation work where to aim.
- **Then test photometric augmentation.** The failures are cross-recording, which is
  mostly a rig-appearance problem: illumination, contrast, focus. This is the lever
  most likely to move the 0.6245 baseline, and it is now measurable for the first
  time.
- **Revisit the loss function.** BCE alone is what these runs used. The BCE+Dice /
  focal / Tversky ablation was parked until sampling, metric, and calibration were
  settled; two of those three now are, so it is unblocked. Dice optimises overlap
  directly, which is what is being measured.
- **Designate a holdout session** if promotion is wanted, per section 4a. This is a
  data-policy decision, not an experiment.
- **Add seeds 3-4** only if the sampling margin needs tightening; the sign is
  consistent across every seed but 0.0354 sits close to the 3-seed sd.
- **Re-examine fixed-threshold selection on fold 3.** Fold 3 lost 0.056 versus the
  old calibrated-threshold selection, 5.7 sd of its own seed spread — the one place
  fix 3 measurably hurt.
