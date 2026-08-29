# Was the promoted checkpoint actually better?

**Date:** 2026-08-14
**Data:** the maintained local dataset, 166 training / 56 validation pairs
**Compute:** Apple M4 via MPS, ~55 minutes for the ten training runs

## Summary

The v0.2.0 → `166pupils_thresh=0.4_iou=0.8749` release changed two things: the
weights (by fine-tuning the previous checkpoint on the same 166 images it was
already trained on) and the prediction threshold (0.7 → 0.4).

- **The threshold change was a real correction.** At 0.7 the model under-measured
  pupil diameters by about 4%; 0.40–0.45 is close to unbiased.
- **The weights change is not supported by the evidence offered for it.** Its
  +0.0112 balanced-IoU margin is 1.6 standard deviations of the spread produced by
  changing the random seed alone, and 28% of run pairs differ by at least that much.
- **Retraining from scratch is nevertheless not a safe substitute.** Three of five
  from-scratch runs lose a real small pupil that all five fine-tuned runs and the
  packaged checkpoint detect. The three that lose it are the three with the
  *highest* validation IoU.
- **No number in this project currently measures generalisation.** 54 of 56
  validation images come from a recording that also appears in training, and there
  are no validation-only animals.

The packaged checkpoint was **kept**. A from-scratch candidate scoring higher
(balanced IoU 0.8825 vs 0.8690) was promoted and then reverted when it failed the
small-pupil check below.

## 1. The seed noise floor

Ten runs, five per arm, identical configuration except the seed. Natural
sampling, batch size 8, 400-epoch cap with patience-40 early stopping. The
fine-tune arm starts from the packaged weights at lr 1e-4; the scratch arm from
random initialisation at lr 1e-3.

| Arm | Balanced IoU | Macro IoU | Best epoch | Thresholds |
|---|---|---|---|---|
| fine_tune (n=5) | 0.8694 ±0.0023 | 0.8724 ±0.0063 | 1–26 | 0.30–0.45 |
| scratch (n=5) | 0.8745 ±0.0092 | 0.8749 ±0.0079 | 94–302 | 0.40–0.65 |
| packaged checkpoint | 0.8690 | 0.8749 | 25 | 0.40 |

Fine-tune minus scratch is −0.0052 balanced (1.2 SE) and −0.0024 macro (0.5 SE):
indistinguishable from zero. The packaged checkpoint's macro IoU equals the mean
of five from-scratch runs to four decimals, and its balanced IoU sits at the 20th
percentile of them.

**The best-epoch column is the clearest evidence.** Fine-tune runs peaked at
epochs 1, 4, 11, 14 and 26 — the best checkpoint is essentially the starting
weights, because there was nothing to learn from images the model was already
fitted to. Scratch runs peaked at 94–302. The packaged checkpoint's best epoch of
25 sits inside the fine-tune distribution.

Pooled seed sd is 0.0069 on both metrics. **Differences below roughly 0.02 are
not evidence of an improvement.**

## 2. The calibrated threshold is not a property of the model

The ten runs selected these thresholds:

```
0.30  0.40  0.40  0.40  0.40  0.45  0.45  0.50  0.55  0.65
```

That range is wider than the 0.7 → 0.4 change that shifted every user's reported
diameters by about 6% in the last release. Because the selected threshold is
written into the checkpoint filename and metadata and read back at inference, the
seed a model happened to be trained with moves every downstream pupil measurement
by several percent.

It also fails to transfer between recording cohorts. On the `date_id` recordings
alone the IoU optimum is 0.50 and zero diameter bias sits near 0.55, where the
packaged 0.40 leaves a +2.2% bias.

## 3. Validation measures held-out frames, not generalisation

| Cohort | n | From a training recording | From a training animal |
|---|---|---|---|
| date_id | 28 | 26 (93%) | 28 (100%) |
| HQL | 28 | 28 (100%) | 28 (100%) |
| **all** | **56** | **54 (96%)** | **56 (100%)** |

There are zero validation-only animals across 10 animals and 24 recordings. Every
reported IoU in this project describes interpolation within recordings the model
has seen.

## 4. Small-pupil IoU is a metric artefact

Packaged checkpoint, 56 validation images, at its own threshold of 0.40:

| Size bin | n | IoU | Signed diameter error | Implied boundary error |
|---|---|---|---|---|
| tiny (≤15 px) | 2 | 0.7954 | −4.3% | **2.4 px** |
| medium | 43 | 0.8587 | +3.6% | 5.4 px |
| large (≥80 px) | 11 | 0.9529 | +0.4% | 4.3 px |

IoU is mechanically harsher on small objects: a boundary off by *k* pixels costs
roughly `k/d` for a disc of diameter *d*. Converting each bin's IoU back into
pixels shows the model is about **twice as accurate on small pupils** as on
everything else. A constant-boundary-error model predicts a tiny-vs-large IoU gap
of 0.259; the observed gap is 0.157.

Two consequences:

- Low tiny-bin IoU was never a small-pupil deficit, so size-balanced sampling was
  built to fix a problem the metric invented.
- `balanced_iou` gives that mechanically-penalised bin a full third of the weight
  in checkpoint selection, on the strength of **two images**.

The bin that is genuinely worst in published units is *medium*: 5.4% mean
absolute diameter error across 43 of the 56 validation images.

## 5. Why the higher-scoring candidate was reverted

`sample_data/raw_frames/recording_250616` frame 1 holds a small pupil that is not
in the validation set. Every run was scored on it at its own calibrated threshold:

| Run | Threshold | Balanced IoU | Hard-frame diameter |
|---|---|---|---|
| fine_tune s0–s4 | 0.30–0.45 | 0.866–0.872 | 11.3–12.0 |
| packaged checkpoint | 0.40 | 0.8690 | 11.9 |
| scratch s1 | 0.40 | 0.8607 | 11.5 |
| scratch s0 | 0.40 | 0.8809 | 6.2 |
| scratch s2 | 0.65 | 0.8696 | **0.0** |
| scratch s3 | 0.50 | 0.8825 | **0.0** |
| scratch s4 | 0.55 | 0.8791 | **0.0** |

Validation IoU is **anti-correlated** with correctness here: the three highest
scorers lose the pupil entirely, and the only scratch run that handles it cleanly
is the lowest scorer. Fine-tuning from the existing lineage is uniformly safe.

This revises, but does not overturn, section 1. Fine-tuning bought nothing
*measurable on validation* — that conclusion stands. But the fine-tuned lineage
preserves a capability that from-scratch training loses more often than not, and
the validation set is blind to it. "Just retrain from scratch" is not a safe
substitute for the current weights.

Note that this frame is used as a **gate**, not a score. Selecting a checkpoint by
maximising performance on it would repeat exactly the error this report documents.

## Reproducing

See [`README.md`](README.md). The statistics table in section 1 comes from:

```bash
python reports/scripts/summarize_runs.py --runs <run-dir> --markdown
```
