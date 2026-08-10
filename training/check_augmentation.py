# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 17:18:21 2026

@author: yzhao
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pupil_tracking.dataset import PupilDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def show_augmented_samples(
    dataset,
    n_samples=5,
    n_augs_per_sample=2,
    overlay_mask=True,
    mask_transparency=0.1,
):
    """
    Visualize augmented samples from a PupilDataset.

    Parameters
    ----------
    dataset : PupilDataset
        Dataset with augment=True
    n_samples : int
        Number of distinct images to visualize
    n_augs_per_sample : int
        Number of augmented versions per image
    overlay_mask : bool
        If True, overlay mask in red
    """
    fig, axes = plt.subplots(
        n_samples,
        n_augs_per_sample,
        figsize=(3 * n_augs_per_sample, 3 * n_samples),
        squeeze=False,
    )

    sample_indices = np.random.choice(np.arange(len(dataset)), n_samples, replace=False)
    for i in range(n_samples):
        for j in range(n_augs_per_sample):
            img, mask = dataset[sample_indices[i]]
            img_np = img.squeeze().numpy()
            ax = axes[i, j]

            if overlay_mask:
                mask_np = mask.squeeze().numpy()
                rgb = np.stack([img_np] * 3, axis=-1)

                # alpha = 0.35  # mask transparency (0 = invisible, 1 = solid)
                red = np.array([1.0, 0.0, 0.0])

                blended = rgb.copy()
                blended[mask_np > 0] = (1 - mask_transparency) * rgb[
                    mask_np > 0
                ] + mask_transparency * red

                ax.imshow(blended)

            else:
                ax.imshow(img_np, cmap="gray")

            ax.axis("off")

            if i == 0:
                ax.set_title(f"Aug {j+1}", fontsize=10)

    plt.tight_layout()
    plt.show()


# example paths
image_paths = sorted((PROJECT_ROOT / "images_train").glob("*.png"))
mask_paths = sorted((PROJECT_ROOT / "masks_train").glob("*.png"))

dataset = PupilDataset(
    image_paths=image_paths,
    mask_paths=mask_paths,
    augment=True,
    target_size=148,
)

show_augmented_samples(
    dataset,
    n_samples=20,
    n_augs_per_sample=2,
    overlay_mask=True,
)
