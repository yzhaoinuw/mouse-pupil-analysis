"""Exercise head diagnostics through the public pipeline and checkpoint loader."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from mouse_pupil_analysis import analyze_frames
from mouse_pupil_analysis.api import AnalysisConfig
from mouse_pupil_analysis.geometry import GeometryUNet
from mouse_pupil_analysis.preprocessing import resize_with_pad
from mouse_pupil_analysis.pupil_predictions import (
    find_default_checkpoint,
    frames_from_image_directory,
    iter_pupil_predictions,
    load_unet_checkpoint,
    resolve_prediction_threshold,
)
from mouse_pupil_analysis.run_pupil_analysis import build_parser
from mouse_pupil_analysis.unet import UNet


def test_full_geometry_checkpoint_loads_and_preserves_mask_logits(tmp_path):
    original = GeometryUNet(use_attention=True).eval()
    checkpoint = tmp_path / "all_data_geometry.pth"
    torch.save(original.state_dict(), checkpoint)
    (tmp_path / "all_data.json").write_text('{"prediction_threshold": 0.525}')
    loaded = load_unet_checkpoint(checkpoint, torch.device("cpu"))
    images = torch.zeros(1, 1, 148, 148)
    with torch.inference_mode():
        expected_mask, expected_circle = original.forward_with_geometry(images)
        mask, circle = loaded.forward_with_geometry(images)
        torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)
        torch.testing.assert_close(circle, expected_circle, rtol=0, atol=0)
        torch.testing.assert_close(loaded(images), mask, rtol=0, atol=0)
    assert resolve_prediction_threshold(checkpoint) == 0.525


@pytest.mark.parametrize(
    "flag", ["show_all_heads", "geometry_head_only", "prefer_geometry_component"]
)
def test_geometry_options_reject_mask_only_checkpoint(tmp_path, flag):
    Image.new("L", (148, 148)).save(tmp_path / "eye_00001.png")
    checkpoint = tmp_path / "legacy.pth"
    torch.save(UNet(use_attention=False).state_dict(), checkpoint)
    with pytest.raises(ValueError, match="full geometry checkpoint"):
        analyze_frames(
            tmp_path,
            checkpoint=checkpoint,
            output_mask_dir=tmp_path / "overlays",
            num_workers=0,
            **{flag: True},
        )


def test_diagnostic_options_validate_and_parse():
    with pytest.raises(ValueError, match="requires output_mask_dir"):
        AnalysisConfig(image_dir="frames", show_all_heads=True).validate()
    with pytest.raises(ValueError, match="requires output_mask_dir"):
        AnalysisConfig(image_dir="frames", geometry_head_only=True).validate()
    with pytest.raises(ValueError, match="only one"):
        AnalysisConfig(
            image_dir="frames",
            output_mask_dir="overlays",
            show_all_heads=True,
            geometry_head_only=True,
        ).validate()
    with pytest.raises(ValueError, match="only one"):
        AnalysisConfig(
            image_dir="frames", prefer_central_component=True, prefer_geometry_component=True
        ).validate()
    for flag in ("--show-all-heads", "--show_all_heads"):
        assert build_parser().parse_args([flag]).show_all_heads
    for flag in ("--geometry-head-only", "--geometry_head_only"):
        assert build_parser().parse_args([flag]).geometry_head_only
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--geometry-head-only", "--show-all-heads"])
    assert build_parser().parse_args([]).prefer_geometry_component is None
    assert build_parser().parse_args(["--raw-segmentation"]).prefer_geometry_component is False


@pytest.mark.parametrize("size", [(148, 148), (296, 200)])
@pytest.mark.parametrize("select_component", [None, True, False])
@pytest.mark.parametrize("geometry_only", [False, True])
def test_diagnostics_show_kept_removed_and_circle_without_changing_results(
    tmp_path, monkeypatch, size, select_component, geometry_only
):
    # Known two-component prediction: a supported pupil and a detached artifact.
    # Radius excludes part of the pupil to check accepted blobs are never clipped.
    class KnownGeometry(GeometryUNet):
        def forward(self, images):
            logits = images.new_full((len(images), 1, 148, 148), -8)
            logits[:, :, 65:85, 65:85] = 8
            logits[:, :, 50:55, 50:55] = 8
            return logits

        def forward_with_geometry(self, images):
            return self(images), images.new_tensor([[74, 74, 6]]).repeat(len(images), 1)

    monkeypatch.setattr(
        "mouse_pupil_analysis.pupil_predictions.load_unet_checkpoint",
        lambda *_: KnownGeometry(use_attention=False).eval(),
    )
    frames = tmp_path / "frames"
    frames.mkdir()
    image_path = frames / "eye_00001.png"
    Image.new("L", size, color=100).save(image_path)
    common = dict(checkpoint=tmp_path / "full.pth", pred_thresh=0.525, num_workers=0)
    raw = list(
        iter_pupil_predictions(
            image_frames=frames_from_image_directory(frames),
            checkpoint_path=common["checkpoint"],
            pred_thresh=0.525,
            num_workers=0,
            prefer_geometry_component=False,
        )
    )[0]
    assert raw.binary_mask.sum() == 425
    selected = analyze_frames(
        frames,
        result_dir=tmp_path / "selected",
        prefer_geometry_component=select_component,
        **common,
    )
    diagnostic = analyze_frames(
        frames,
        result_dir=tmp_path / "diagnostic",
        output_mask_dir=tmp_path / "overlays",
        show_all_heads=not geometry_only,
        geometry_head_only=geometry_only,
        prefer_geometry_component=select_component,
        mask_transparency=1,
        **common,
    )
    pd.testing.assert_frame_equal(selected.analysis_table, diagnostic.analysis_table)
    assert diagnostic.analysis_table.estimated_pupil_diameter.iloc[0] == pytest.approx(
        np.sqrt((425 if select_component is False else 400) * 4 / np.pi)
    )
    rgb = np.array(Image.open(tmp_path / "overlays" / image_path.name))
    assert rgb.shape == (148, 148, 3)
    if geometry_only:
        # Neither segmentation fills nor the segmentation-derived QC marker leak
        # into this view, even though both are still used by the analysis table.
        yellow = np.all(rgb == [255, 255, 0], axis=-1)
        background = np.array(resize_with_pad(Image.new("L", size, color=100)))
        np.testing.assert_array_equal(
            rgb[~yellow], np.repeat(background[:, :, None], 3, axis=2)[~yellow]
        )
    else:
        # Both retained and discarded interiors retain the normal confidence
        # heatmap. Selection changes the outline, never the confidence colors.
        np.testing.assert_array_equal(rgb[75, 75], [255, 0, 0])
        np.testing.assert_array_equal(rgb[52, 52], [255, 0, 0])
    assert np.any(np.all(rgb == [255, 255, 0], axis=-1))
    # All heads and contours use the same model-image coordinates at every source size.
    assert np.all(rgb[74, 74] == [255, 255, 0])


def test_packaged_default_uses_both_heads_and_matches_selector():
    from mouse_pupil_analysis.geometry import select_circle_components

    # This public fixture has multiple components, so this catches silently
    # omitting the selector even when the geometry weights are present.
    source = find_default_checkpoint().parents[2] / "sample_data" / "labeled_frames"
    frames = [
        frame
        for directory in sorted(source.glob("*/images"))
        for frame in frames_from_image_directory(directory)
    ]
    assert frames
    raw = list(
        iter_pupil_predictions(
            find_default_checkpoint(), frames, num_workers=0, prefer_geometry_component=False
        )
    )
    final = list(iter_pupil_predictions(find_default_checkpoint(), frames, num_workers=0))
    changed = 0
    for before, after in zip(raw, final, strict=True):
        assert after.geometry_circle is not None
        expected = select_circle_components(before.binary_mask, after.geometry_circle)
        np.testing.assert_array_equal(after.binary_mask, expected)
        assert after.measure_all_components
        changed += int(not np.array_equal(before.binary_mask, after.binary_mask))
    assert changed > 0


def test_pipeline_measures_all_circle_retained_blobs(tmp_path, monkeypatch):
    class FragmentedGeometry(GeometryUNet):
        def forward_with_geometry(self, images):
            logits = images.new_full((len(images), 1, 148, 148), -8)
            logits[:, :, 65:70, 65:70] = 8
            logits[:, :, 75:80, 75:80] = 8
            logits[:, :, 20:40, 100:120] = 8
            return logits, images.new_tensor([[74, 74, 15]]).repeat(len(images), 1)

    monkeypatch.setattr(
        "mouse_pupil_analysis.pupil_predictions.load_unet_checkpoint",
        lambda *_: FragmentedGeometry(use_attention=False).eval(),
    )
    Image.new("L", (148, 148), color=100).save(tmp_path / "eye_00001.png")
    result = analyze_frames(
        tmp_path,
        checkpoint=tmp_path / "full.pth",
        pred_thresh=0.525,
        num_workers=0,
        result_dir=tmp_path / "results",
        output_mask_dir=tmp_path / "overlays",
        show_all_heads=True,
    )
    row = result.segmentation_dataframe.iloc[0]
    assert row.selected_component_area == row.foreground_area == 50
    assert row.component_count == 2
    assert result.analysis_table.estimated_pupil_diameter.iloc[0] == pytest.approx(
        np.sqrt(50 * 4 / np.pi)
    )
