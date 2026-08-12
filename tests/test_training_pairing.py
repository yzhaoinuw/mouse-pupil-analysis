"""Coverage for the shared image/mask pairing helper and the deprecated shim.

Mispairing images and masks produces no error at all -- it just trains against the
wrong labels -- so the failure modes here are worth pinning explicitly.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mouse_pupil_analysis.augmentation import SegmentationDataset, paired_image_mask_paths
from mouse_pupil_analysis.dataset import PupilDataset
from mouse_pupil_analysis.preprocessing import InferenceDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = PROJECT_ROOT / "sample_data"


def _blank(path: Path) -> None:
    Image.fromarray(np.zeros((20, 20), dtype=np.uint8)).save(path)


@pytest.fixture
def paired_dirs(tmp_path: Path) -> tuple[Path, Path]:
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for stem in ("a", "b", "c"):
        _blank(images / f"{stem}.png")
    # Deliberately created in a different order than the images.
    for stem in ("c", "a", "b"):
        _blank(masks / f"{stem}.png")
    return images, masks


def test_pairing_is_by_stem_not_by_directory_order(paired_dirs):
    images, masks = paired_dirs

    image_paths, mask_paths = paired_image_mask_paths(images, masks)

    assert [path.stem for path in image_paths] == ["a", "b", "c"]
    assert [path.stem for path in mask_paths] == ["a", "b", "c"]


def test_image_without_a_mask_is_rejected(paired_dirs):
    images, masks = paired_dirs
    (masks / "b.png").unlink()

    with pytest.raises(FileNotFoundError, match="have no mask"):
        paired_image_mask_paths(images, masks)


def test_orphan_mask_is_rejected_by_default_and_can_be_allowed(paired_dirs):
    images, masks = paired_dirs
    _blank(masks / "leftover.png")

    with pytest.raises(FileNotFoundError, match="have no image"):
        paired_image_mask_paths(images, masks)

    image_paths, mask_paths = paired_image_mask_paths(images, masks, allow_orphan_masks=True)
    assert len(image_paths) == len(mask_paths) == 3


def test_empty_image_directory_is_rejected(tmp_path: Path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    with pytest.raises(FileNotFoundError, match="No PNG images"):
        paired_image_mask_paths(images, masks)


@pytest.mark.skipif(not SAMPLE_ROOT.is_dir(), reason="sample_data/ needs a source checkout.")
@pytest.mark.parametrize(("split", "count"), [("train", 8), ("validation", 4)])
def test_committed_fixture_pairs_cleanly(split, count):
    image_paths, mask_paths = paired_image_mask_paths(
        SAMPLE_ROOT / f"images_{split}",
        SAMPLE_ROOT / f"masks_{split}",
    )

    assert len(image_paths) == count
    assert [path.stem for path in image_paths] == [path.stem for path in mask_paths]


def test_deprecated_shim_accepts_the_original_positional_signature(paired_dirs):
    images, masks = paired_dirs
    image_paths = sorted(images.glob("*.png"))
    mask_paths = sorted(masks.glob("*.png"))

    # The removed class took (image_paths, mask_paths, augment, target_size,
    # scale_range, max_pad) positionally. Existing scripts rely on that order.
    with pytest.deprecated_call():
        dataset = PupilDataset(image_paths, mask_paths, True, 148, (0.9, 1.1), 8)

    assert isinstance(dataset, SegmentationDataset)
    assert dataset.augment is True
    assert dataset.target_size == 148
    assert dataset.scale_range == (0.9, 1.1)
    assert dataset.max_pad == 8


def test_deprecated_shim_without_masks_returns_an_inference_dataset(paired_dirs):
    images, _ = paired_dirs

    with pytest.deprecated_call():
        dataset = PupilDataset(sorted(images.glob("*.png")))

    assert isinstance(dataset, InferenceDataset)
    _tensor, name = dataset[0]
    assert name == "a.png"
