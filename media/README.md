# README Demo GIF

This folder owns the animated pupil-analysis demo shown in the repository README:

- `pupil_diameter_analysis_result_demo.gif` is the promoted, tracked asset.
- `make_gif.py` builds the animation from a completed velocity analysis and its overlay images.

The local default source is deliberately minimal and gitignored:

```text
media/readme_demo/
  pupil_analysis_for_gif.csv
  overlays/                    # the 90 PNGs used by the promoted animation
```

Running `python media\make_gif.py` with no arguments rebuilds the promoted GIF from this
source. Large source recordings belong in `videos/`; alternate candidates and full analysis
runs are not part of the maintained demo workspace.

Run the commands below from the repository root. The script resolves its default paths relative
to the repository rather than the current working directory.

Activate the project environment first:

```powershell
conda activate mouse_pupil_analysis
```

## 1. Create the source analysis

The GIF needs both a unified analysis CSV and one overlay PNG for every selected row. A typical video run is:

```powershell
run-pupil-analysis `
    --video_path C:\path\to\eye.avi `
    --out_dir C:\path\to\eye_demo\frames `
    --result_dir C:\path\to\eye_demo\results `
    --output_mask_dir C:\path\to\eye_demo\overlays `
    --calculate_velocity `
    --acquisition_fps 33.3333
```

Use the real acquisition rate for `--acquisition_fps`. If it is omitted for video input, the analysis uses the video's encoded frame rate, which may not match the acquisition hardware.

The CSV must contain:

- `image_name`
- `estimated_pupil_diameter`
- `center_x_pixels`
- `center_y_pixels`
- `speed_pixels_per_second`

Each `image_name` must also exist in the overlay folder. If the CSV additionally contains `tracking_status`, `raw_center_x_pixels`, `raw_center_y_pixels`, and `raw_speed_pixels_per_second`, the animation can show rejected estimates as dashed diagnostic segments.

## 2. Build a candidate GIF

Pass the analysis outputs explicitly when working on a new recording:

```powershell
python media\make_gif.py `
    --csv_path C:\path\to\eye_demo\results\eye_pupil_analysis.csv `
    --overlay_dir C:\path\to\eye_demo\overlays `
    --output C:\path\to\eye_demo\pupil_demo_candidate.gif `
    --start_frame 200 `
    --end_frame 2100 `
    --sample_every 20 `
    --fps 10 `
    --pred_thresh 0.7
```

The selected range and sampling interval must produce at least two frames. The main controls are:

| Option | Meaning | Default |
| --- | --- | ---: |
| `--start_frame` | First source-frame number eligible for the GIF | `7107` |
| `--end_frame` | Last source-frame number eligible for the GIF | `7375` |
| `--sample_every` | Keep every Nth eligible analysis row | `1` |
| `--fps` | GIF playback rate | `5` |
| `--pred_thresh` | Lower bound shown on the confidence legend | `0.7` |

Run `python media\make_gif.py --help` for the full interface.

## Use editable defaults

To use the local defaults, edit `DEFAULT_CSV`, `DEFAULT_OVERLAY_DIR`, and `DEFAULT_OUTPUT` near
the top of `make_gif.py`, then run the file directly. Its `if __name__ == "__main__":` block
invokes the same command-line workflow, so the generated animation behaves identically.

## 3. Review and promote

Inspect the entire candidate before replacing the tracked demo. Check that:

- The animation covers a useful pupil-size or motion transition.
- The mask confidence, center marker, and traces stay aligned.
- Playback is smooth and the file size is reasonable for GitHub.
- Frame sampling does not hide important quality-control events.

After review, regenerate directly to `media/pupil_diameter_analysis_result_demo.gif` or copy the approved candidate there. Keep large source videos, analysis outputs, and rejected candidates outside Git; only the selected README asset is intentionally tracked.
