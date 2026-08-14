# Reports

Dated analyses of the model and the data behind it, with the scripts that
produce their numbers. These are evidence for decisions recorded elsewhere —
`CHANGELOG.md` for what shipped, `treaty_docs/next_steps.md` for what is planned.

| Report | Question it answers |
|---|---|
| [`2026-08-14-checkpoint-noise-floor.md`](2026-08-14-checkpoint-noise-floor.md) | Was the promoted checkpoint better than its predecessor, or a lucky seed? |

## Requirements

The scripts need the local training data, which is not redistributable and is not
in the repository. Create these folders at the repository root as documented in
[`training/README.md`](../training/README.md):

```text
images_train/   masks_train/   images_validation/   masks_validation/
```

`hard_frame_check.py` is the exception — it runs against the committed
`sample_data/` fixture and needs no local dataset.

Run everything from the repository root in the project environment:

```bash
conda activate pupil_tracking
python -m pip install -e .
```

## Scripts

### `summarize_runs.py` — the statistics table

Reduces a directory of run folders to the per-arm mean, standard deviation, and
range that section 1 of the report tabulates. Arms are grouped automatically from
each run's `training_mode`.

```bash
python reports/scripts/summarize_runs.py --runs checkpoints_exp --markdown
```

`--claimed-gain` expresses a proposed improvement in units of the observed seed
spread, which is the number that decides whether a result is real:

```bash
python reports/scripts/summarize_runs.py --runs checkpoints_exp --claimed-gain 0.0112
```

### `noise_floor.py` — produce the runs

Trains the same configuration N times per arm, varying only the seed. This is
what `summarize_runs.py` consumes. About 55 minutes for the default 5 seeds × 2
arms on an M4 via MPS; several times longer on CPU.

```bash
python reports/scripts/noise_floor.py --out checkpoints_exp/noise --seeds 5
python reports/scripts/summarize_runs.py --runs checkpoints_exp/noise --markdown
```

Pass `--arms scratch` to skip fine-tuning, or `--device cpu` to force a device.

### `dataset_census.py` — mask sizes and split leakage

Reports the tiny/medium/large mask distribution, and how many validation images
come from a recording or animal that also appears in training.

```bash
python reports/scripts/dataset_census.py --data-root .
```

### `validation_diagnostics.py` — IoU alongside reported diameter

Sweeps the threshold grid printing macro IoU *and* signed diameter error, then
breaks the checkpoint's own threshold down by size bin. Use it whenever a
threshold changes: IoU will not tell you that reported diameters moved.

```bash
python reports/scripts/validation_diagnostics.py --data-root .
python reports/scripts/validation_diagnostics.py --checkpoint checkpoints_exp/run/best.pth
```

### `hard_frame_check.py` — promotion gate

Runs candidates over real frames containing a small pupil that validation does
not cover, and exits non-zero if any checkpoint loses it. Exists because in the
2026-08-14 study the three runs with the highest validation IoU were the three
that failed this check.

```bash
python reports/scripts/hard_frame_check.py                          # packaged checkpoint
python reports/scripts/hard_frame_check.py --checkpoints checkpoints_exp/*/best.pth
```

Treat it as pass/fail. Selecting a checkpoint by maximising its result here would
repeat the selection-on-noise error these reports document.

## Adding a report

Name the file `YYYY-MM-DD-topic.md`, state the data and compute it used, and put
any script that produced a number in `scripts/` so the number can be regenerated
rather than trusted. Add a row to the table above.
