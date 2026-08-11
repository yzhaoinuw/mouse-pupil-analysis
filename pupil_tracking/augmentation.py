"""Training-time augmentation and the paired image/mask dataset.

Nothing in this module runs during inference. It is imported by the training
workflow under ``training/`` only.
"""

import random
from pathlib import Path

import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    InterpolationMode as InterpMode,
)  # note: NOT v2 InterpolationMode

from pupil_tracking.preprocessing import MODEL_IMAGE_SIZE, resize_with_pad


def paired_image_mask_paths(
    image_dir,
    mask_dir,
    allow_orphan_masks: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Pair every training image with the mask sharing its filename stem.

    Sorting the two directories independently silently misaligns images and masks as
    soon as the folders diverge, which trains the model against wrong labels without
    raising anything. ``labelme_json2png.py`` writes each mask as ``<image stem>.png``,
    so the stem is the reliable key.

    Both directions are checked. An image with no mask is always an error. A mask with
    no image usually means its image was deleted or renamed, so it is an error too
    unless ``allow_orphan_masks`` is set.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    mask_by_stem = {path.stem: path for path in mask_dir.glob("*.png")}

    missing_masks = [path.name for path in image_paths if path.stem not in mask_by_stem]
    if missing_masks:
        raise FileNotFoundError(
            f"{len(missing_masks)} image(s) in {image_dir} have no mask in {mask_dir}, "
            f"starting with {missing_masks[0]!r}."
        )

    if not allow_orphan_masks:
        image_stems = {path.stem for path in image_paths}
        orphans = sorted(name for stem, name in mask_by_stem.items() if stem not in image_stems)
        if orphans:
            raise FileNotFoundError(
                f"{len(orphans)} mask(s) in {mask_dir} have no image in {image_dir}, "
                f"starting with {Path(orphans[0]).name!r}. Pass allow_orphan_masks=True "
                "to ignore them."
            )

    return image_paths, [mask_by_stem[path.stem] for path in image_paths]


class RandomAffinePair:
    """
    Apply the SAME random affine to (img, mask), but:
      - img uses BILINEAR interpolation (smooth, natural)
      - mask uses NEAREST interpolation (keeps labels crisp)
    Works on PIL Images.
    """

    def __init__(
        self,
        degrees=8,
        translate=(0, 0),
        scale=(0.95, 1.05),
        shear=None,
        fill_img=0,
        fill_mask=0,
        p=0.7,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.fill_img = fill_img
        self.fill_mask = fill_mask
        self.p = p

    def __call__(self, img, mask):
        if random.random() > self.p:
            return img, mask

        angle = random.uniform(-self.degrees, self.degrees)

        # translate is specified as fraction of image size
        max_dx = self.translate[0] * img.size[0]
        max_dy = self.translate[1] * img.size[1]
        translations = (
            int(round(random.uniform(-max_dx, max_dx))),
            int(round(random.uniform(-max_dy, max_dy))),
        )

        sc = random.uniform(self.scale[0], self.scale[1])

        # shear can be None, float, or (min,max). Keep None unless you have a reason.
        shear = self.shear
        if isinstance(shear, (tuple, list)) and len(shear) == 2:
            shear = random.uniform(shear[0], shear[1])

        img2 = TF.affine(
            img,
            angle=angle,
            translate=translations,
            scale=sc,
            shear=shear if shear is not None else 0.0,
            interpolation=InterpMode.BILINEAR,
            fill=self.fill_img,
        )
        mask2 = TF.affine(
            mask,
            angle=angle,
            translate=translations,
            scale=sc,
            shear=shear if shear is not None else 0.0,
            interpolation=InterpMode.NEAREST,
            fill=self.fill_mask,
        )
        return img2, mask2


def random_zoom_translate_pil(
    img: Image.Image,
    mask: Image.Image,
    target_size: int = MODEL_IMAGE_SIZE,
    scale_range=(0.85, 1.15),
    fill_img: int = 0,
    fill_mask: int = 0,
    p: float = 0.5,
):
    """
    With probability p:
      - applies slight zoom in/out + random translation while keeping output size == target_size.
    Otherwise returns (img, mask) unchanged.
    """
    if random.random() > p:
        return img, mask

    s = random.uniform(*scale_range)
    new_size = int(round(target_size * s))
    new_size = max(1, new_size)

    img_rs = img.resize((new_size, new_size), resample=Image.BILINEAR)
    mask_rs = mask.resize((new_size, new_size), resample=Image.NEAREST)

    if new_size == target_size:
        return img_rs, mask_rs

    if new_size > target_size:
        # zoom in: random crop
        max_left = new_size - target_size
        max_top = new_size - target_size
        left = random.randint(0, max_left)
        top = random.randint(0, max_top)
        box = (left, top, left + target_size, top + target_size)
        return img_rs.crop(box), mask_rs.crop(box)

    # zoom out: random placement (translation + padding jitter)
    canvas_img = Image.new("L", (target_size, target_size), color=fill_img)
    canvas_mask = Image.new("L", (target_size, target_size), color=fill_mask)

    max_left = target_size - new_size
    max_top = target_size - new_size
    left = random.randint(0, max_left)
    top = random.randint(0, max_top)

    canvas_img.paste(img_rs, (left, top))
    canvas_mask.paste(mask_rs, (left, top))
    return canvas_img, canvas_mask


def random_pad_and_crop_pil(
    img: Image.Image,
    mask: Image.Image,
    target_size: int = MODEL_IMAGE_SIZE,
    max_pad: int = 12,
    fill_img: int = 0,
    fill_mask: int = 0,
    p: float = 0.5,
):
    """
    With probability p:
      - randomly pads each side by [0..max_pad], then random-crops back to target_size.
    Otherwise returns (img, mask) unchanged.
    """
    if random.random() > p:
        return img, mask

    pad_l = random.randint(0, max_pad)
    pad_r = random.randint(0, max_pad)
    pad_t = random.randint(0, max_pad)
    pad_b = random.randint(0, max_pad)

    w, h = img.size  # expected (target_size, target_size)
    new_w = w + pad_l + pad_r
    new_h = h + pad_t + pad_b

    padded_img = Image.new("L", (new_w, new_h), color=fill_img)
    padded_mask = Image.new("L", (new_w, new_h), color=fill_mask)
    padded_img.paste(img, (pad_l, pad_t))
    padded_mask.paste(mask, (pad_l, pad_t))

    max_left = new_w - target_size
    max_top = new_h - target_size
    left = random.randint(0, max_left)
    top = random.randint(0, max_top)
    box = (left, top, left + target_size, top + target_size)

    return padded_img.crop(box), padded_mask.crop(box)


class SegmentationDataset(Dataset):
    """Yield augmented ``(image_tensor, mask_tensor)`` pairs for training."""

    def __init__(
        self,
        image_paths,
        mask_paths,
        augment=False,
        target_size=MODEL_IMAGE_SIZE,
        scale_range=(0.85, 1.15),
        max_pad=12,
    ):
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Got {len(image_paths)} images but {len(mask_paths)} masks; "
                "every training image needs exactly one mask."
            )
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.target_size = target_size
        self.scale_range = scale_range
        self.max_pad = max_pad

        self.pil_to_tensor = transforms.PILToTensor()

        self.flip_h = transforms.RandomHorizontalFlip(p=0.2)
        self.flip_v = transforms.RandomVerticalFlip(p=0.2)

        # A single, paired affine that rotates/translates/scales cleanly
        self.affine_pair = RandomAffinePair(
            degrees=15,
            scale=(0.95, 1.05),
            shear=None,
            fill_img=0,
            fill_mask=0,
            p=0.2,
        )

        # Photometric transforms for image only
        self.transform_img = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2)],
                    p=0.2,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                    p=0.2,
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        img = resize_with_pad(img, target_size=self.target_size, resample=Image.BILINEAR)

        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = resize_with_pad(mask, target_size=self.target_size, resample=Image.NEAREST)

        if self.augment:
            # zoom/translate/pad jitter
            img, mask = random_zoom_translate_pil(
                img,
                mask,
                target_size=self.target_size,
                scale_range=self.scale_range,
                fill_img=0,
                fill_mask=0,
                p=0.2,
            )
            img, mask = random_pad_and_crop_pil(
                img,
                mask,
                target_size=self.target_size,
                max_pad=self.max_pad,
                fill_img=0,
                fill_mask=0,
                p=0.2,
            )

            # flips (paired, safe)
            img, mask = self.flip_h(img, mask)
            img, mask = self.flip_v(img, mask)

            # affine (paired, bilinear img + nearest mask)
            img, mask = self.affine_pair(img, mask)

            # image-only photometric
            img = self.transform_img(img)

        img = self.pil_to_tensor(img).float() / 255.0

        mask = self.pil_to_tensor(mask).float()
        mask = (mask > 0).float()  # ensure {0,1}

        return img, mask
