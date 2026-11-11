"""Run circle-overlap selection through the CLI on every PNG under images_test.

Writes a separate all-head overlay tree, CLI logs/results, and a verified manifest.
Existing predictions and model weights are never overwritten.
"""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mouse_pupil_analysis.geometry import CIRCLE_MIN_OVERLAP  # noqa: E402
from mouse_pupil_analysis.pupil_predictions import (  # noqa: E402
    find_default_checkpoint,
    resolve_prediction_threshold,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "images_test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "images_test_prediction/blob_selection_20pct_2026-11-11",
    )
    parser.add_argument(
        "--control-dir", type=Path, default=ROOT / "checkpoints_exp/blob_selection_20pct_2026-11-11"
    )
    args = parser.parse_args()
    source, output, control = (
        path.resolve() for path in (args.input_dir, args.output_dir, args.control_dir)
    )
    console = Path(sys.executable).parent / "Scripts/run-pupil-analysis.exe"
    if output.exists():
        raise FileExistsError(f"Choose a new overlay directory: {output}")
    if (
        source == output
        or source in output.parents
        or source == control
        or source in control.parents
    ):
        raise ValueError("Outputs must be outside the input directory.")
    inputs = sorted(source.rglob("*.png"))
    if not inputs or not console.is_file():
        raise FileNotFoundError("Input PNGs and the environment's console command are required.")
    checkpoint = find_default_checkpoint()
    threshold = resolve_prediction_threshold(checkpoint)
    source_files = [
        ROOT / "mouse_pupil_analysis" / name
        for name in (
            "geometry.py",
            "pupil_predictions.py",
            "tracking.py",
            "api.py",
            "run_pupil_analysis.py",
            "preprocessing.py",
            "unet.py",
        )
    ] + [Path(__file__).resolve(), checkpoint.with_suffix(".json")]
    source_hashes = {str(path): sha(path) for path in source_files}
    input_hashes = {path.relative_to(source).as_posix(): sha(path) for path in inputs}
    groups = defaultdict(list)
    for path in inputs:
        groups[path.parent].append(path)
    baseline_path = ROOT / "images_test_prediction/all_heads_2026-11-10/manifest.csv"
    baseline = {}
    if baseline_path.is_file():
        with baseline_path.open(newline="", encoding="utf-8") as stream:
            baseline = {row["source"]: row for row in csv.DictReader(stream)}
    output.mkdir(parents=True)
    control.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record = dict(
        status="running",
        started_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source),
        output=str(output),
        expected_images=len(inputs),
        completed_images=0,
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),
        prediction_threshold=threshold,
        minimum_blob_overlap=CIRCLE_MIN_OVERLAP,
        circle_expansion_pixels=0,
        show_all_heads=True,
        mask_transparency=0.35,
        training_change=False,
        interface="run-pupil-analysis console, one input directory per call",
        source_sha256=source_hashes,
        commands=[],
        baseline_manifest=str(baseline_path) if baseline else None,
        changed_area_images=0,
        empty_masks=0,
    )
    run_path = output / "run.json"

    def save_record():
        run_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    save_record()
    try:
        with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "source",
                    "input_sha256",
                    "output_sha256",
                    "derived_input",
                    "analysis_csv",
                    "estimated_pupil_diameter",
                    "retained_area_model_pixels",
                    "previous_area_model_pixels",
                ],
            )
            writer.writeheader()
            for index, (directory, paths) in enumerate(sorted(groups.items())):
                for path in source_files:
                    assert sha(path) == source_hashes[str(path)], f"Code/metadata changed: {path}"
                assert sha(checkpoint) == record["checkpoint_sha256"]
                results, log = control / f"r{index:02d}", control / f"r{index:02d}.log"
                if results.exists():
                    raise FileExistsError(results)
                relative = directory.relative_to(source)
                command = [
                    str(console),
                    "--image_dir",
                    str(directory),
                    "--output_mask_dir",
                    str(output / relative),
                    "--show-all-heads",
                    "--mask_transparency",
                    "0.35",
                    "--result_dir",
                    str(results),
                    "--num_workers",
                    "0",
                    "--checkpoint",
                    str(checkpoint),
                ]
                record["commands"].append(dict(argv=command, log=str(log), images=len(paths)))
                save_record()
                print(
                    f"CLI {index + 1}/{len(groups)}: {relative} ({len(paths):,} images)", flush=True
                )
                with log.open("w", encoding="utf-8") as handle:
                    process = subprocess.run(
                        command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
                    )
                if process.returncode:
                    raise RuntimeError(log.read_text(encoding="utf-8")[-3000:])
                assert "Geometry-based component selection: True" in log.read_text(encoding="utf-8")
                csv_files = list(results.glob("*.csv"))
                assert len(csv_files) == 1
                with csv_files[0].open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                by_name = {row["image_name"]: row for row in rows}
                assert len(rows) == len(paths) == len(by_name)
                assert set(by_name) == {path.name for path in paths}
                for path in paths:
                    rel = path.relative_to(source).as_posix()
                    overlay = output / rel
                    assert sha(path) == input_hashes[rel], f"Input changed: {path}"
                    with Image.open(overlay) as image:
                        assert image.mode == "RGB" and image.size == (148, 148)
                        image.verify()
                    diameter = float(by_name[path.name]["estimated_pupil_diameter"])
                    area = round(diameter**2 * math.pi / 4)
                    assert math.isclose(diameter, math.sqrt(area * 4 / math.pi), abs_tol=1e-9)
                    previous = baseline.get(rel)
                    previous_area = (
                        round(float(previous["estimated_pupil_diameter"]) ** 2 * math.pi / 4)
                        if previous
                        else None
                    )
                    record["changed_area_images"] += int(
                        previous_area is not None and area != previous_area
                    )
                    record["empty_masks"] += int(area == 0)
                    writer.writerow(
                        dict(
                            source=rel,
                            input_sha256=input_hashes[rel],
                            output_sha256=sha(overlay),
                            derived_input=previous["derived_input"] if previous else "unknown",
                            analysis_csv=str(csv_files[0]),
                            estimated_pupil_diameter=diameter,
                            retained_area_model_pixels=area,
                            previous_area_model_pixels=previous_area,
                        )
                    )
                stream.flush()
                record["completed_images"] += len(paths)
                save_record()
                print(
                    f"Verified {record['completed_images']:,}/{len(inputs):,} overlays", flush=True
                )
        expected = set(input_hashes)
        assert {p.relative_to(output).as_posix() for p in output.rglob("*.png")} == expected
        assert {p.relative_to(source).as_posix() for p in source.rglob("*.png")} == expected
        for path in inputs:
            assert sha(path) == input_hashes[path.relative_to(source).as_posix()]
        record.update(
            status="complete",
            finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(time.perf_counter() - started, 2),
        )
        save_record()
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in (
                        "status",
                        "completed_images",
                        "changed_area_images",
                        "empty_masks",
                        "elapsed_seconds",
                    )
                }
            ),
            flush=True,
        )
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        save_record()
        raise


if __name__ == "__main__":
    main()
