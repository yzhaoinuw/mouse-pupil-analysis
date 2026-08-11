"""Image preprocessing shared by inference and training.

The 148 x 148 aspect-ratio-preserving resize-and-pad convention defined here is
load-bearing: the packaged checkpoint was trained on images prepared this way.
Changing it invalidates the model.
"""

import math
from pathlib import Path

import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset

MODEL_IMAGE_SIZE = 148


def resize_with_pad(
    img: Image.Image,
    target_size: int = MODEL_IMAGE_SIZE,
    fill: int = 0,
    resample=Image.BILINEAR,
) -> Image.Image:
    """Resize preserving aspect ratio, then center-pad to a square target size."""
    w, h = img.size
    if w >= h:
        new_w = target_size
        new_h = int(round(h * target_size / w))
    else:
        new_h = target_size
        new_w = int(round(w * target_size / h))

    img = img.resize((new_w, new_h), resample=resample)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    left = pad_w // 2
    top = pad_h // 2

    padded = Image.new("L", (target_size, target_size), color=fill)
    padded.paste(img, (left, top))
    return padded


def resize_scale(
    original_width: int,
    original_height: int,
    target_size: int = MODEL_IMAGE_SIZE,
) -> tuple[float, float, int, int]:
    """Return the x/y scale and left/top padding applied by :func:`resize_with_pad`.

    This is the single source of truth for inverting the model-space geometry, used
    both to map centers back to video pixels and to convert areas and diameters.
    """
    if original_width <= 0 or original_height <= 0:
        raise ValueError("Original image dimensions must be positive.")
    if target_size <= 0:
        raise ValueError("Target size must be positive.")

    if original_width >= original_height:
        resized_width = target_size
        resized_height = int(round(original_height * target_size / original_width))
    else:
        resized_height = target_size
        resized_width = int(round(original_width * target_size / original_height))

    pad_left = (target_size - resized_width) // 2
    pad_top = (target_size - resized_height) // 2
    return resized_width / original_width, resized_height / original_height, pad_left, pad_top


def model_to_input_length(
    model_length: float,
    original_width: int,
    original_height: int,
    target_size: int = MODEL_IMAGE_SIZE,
) -> float:
    """Convert a model-space length back to the supplied image's pixel scale.

    ``original_width`` and ``original_height`` are the dimensions of the image that
    was fed in, which is the source video frame only when frames came from video
    extraction. Areas scale by ``scale_x * scale_y``, so a length derived from an
    area (such as an equivalent-circle diameter) converts by the geometric mean of
    the two scales.
    """
    scale_x, scale_y, _, _ = resize_scale(original_width, original_height, target_size)
    return float(model_length / math.sqrt(scale_x * scale_y))


class InferenceDataset(Dataset):
    """Yield ``(image_tensor, image_name)`` pairs for segmentation inference."""

    def __init__(self, image_paths, target_size: int = MODEL_IMAGE_SIZE):
        self.image_paths = [Path(path) for path in image_paths]
        self.target_size = target_size
        self.pil_to_tensor = transforms.PILToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")
        image = resize_with_pad(image, target_size=self.target_size, resample=Image.BILINEAR)
        return self.pil_to_tensor(image).float() / 255.0, image_path.name
