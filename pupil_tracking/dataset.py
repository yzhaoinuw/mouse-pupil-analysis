"""Deprecated compatibility shim.

The contents of this module moved in favor of a clearer split:

- :mod:`pupil_tracking.preprocessing` holds ``resize_with_pad`` and
  ``InferenceDataset``, the parts that run during inference.
- :mod:`pupil_tracking.augmentation` holds the training-time augmentations and
  ``SegmentationDataset``.

``PupilDataset`` returned either ``(image, name)`` or ``(image, mask)`` depending
on whether ``mask_paths`` was supplied. The two dataset classes replace that
single polymorphic class. This shim will be removed in a future release.
"""

import warnings

from pupil_tracking.augmentation import (
    RandomAffinePair,
    SegmentationDataset,
    random_pad_and_crop_pil,
    random_zoom_translate_pil,
)
from pupil_tracking.preprocessing import MODEL_IMAGE_SIZE, InferenceDataset, resize_with_pad

__all__ = [
    "InferenceDataset",
    "MODEL_IMAGE_SIZE",
    "PupilDataset",
    "RandomAffinePair",
    "SegmentationDataset",
    "random_pad_and_crop_pil",
    "random_zoom_translate_pil",
    "resize_with_pad",
]

warnings.warn(
    "pupil_tracking.dataset is deprecated; import from pupil_tracking.preprocessing "
    "or pupil_tracking.augmentation instead.",
    DeprecationWarning,
    stacklevel=2,
)


def PupilDataset(image_paths, mask_paths=None, **kwargs):
    """Deprecated. Use ``InferenceDataset`` or ``SegmentationDataset`` directly."""
    warnings.warn(
        "PupilDataset is deprecated; use InferenceDataset for inference or "
        "SegmentationDataset for training.",
        DeprecationWarning,
        stacklevel=2,
    )
    if mask_paths is None:
        return InferenceDataset(
            image_paths,
            target_size=kwargs.get("target_size", MODEL_IMAGE_SIZE),
        )
    return SegmentationDataset(image_paths, mask_paths, **kwargs)
