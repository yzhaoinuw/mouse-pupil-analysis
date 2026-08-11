"""Image preprocessing shared by inference and training.

The 148 x 148 aspect-ratio-preserving resize-and-pad convention defined here is
load-bearing: the packaged checkpoint was trained on images prepared this way.
Changing it invalidates the model.
"""

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
