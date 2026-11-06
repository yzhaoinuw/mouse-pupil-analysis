"""Experimental circle supervision and gating for visible-pupil segmentation."""

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mouse_pupil_analysis.preprocessing import MODEL_IMAGE_SIZE
from mouse_pupil_analysis.unet import UNet

GEOMETRY_LOSS_WEIGHT = 1.0
SMOOTH_L1_BETA_PIXELS = 4.0
GATE_MARGIN_PIXELS = 2.0
GATE_FADE_PIXELS = 4.0


class GeometryUNet(UNet):
    """Add a circle head to the bottleneck of the 148-square U-Net."""

    def __init__(self, *, use_attention: bool):
        super().__init__(use_attention=use_attention)
        # Preserve the mask-only run's RNG stream for shuffling and augmentation.
        with torch.random.fork_rng(devices=[]):
            self.geometry_head = nn.Sequential(
                nn.Conv2d(128, 16, kernel_size=1),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=10, stride=9),
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, 3),
                nn.Sigmoid(),
            )

    def forward_with_geometry(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mask logits and (x, y, radius) in model pixels."""
        features = self._encode(images)
        circles = self.geometry_head(features[-1]) * MODEL_IMAGE_SIZE
        return self._decode(*features), circles


def circle_targets(masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Enclose each transformed visible mask and flag empty masks."""
    circles = np.zeros((len(masks), 3), dtype=np.float32)
    present = np.zeros(len(masks), dtype=bool)
    binary_masks = (masks.detach().cpu().numpy()[:, 0] > 0.5).astype(np.uint8)
    for index, mask in enumerate(binary_masks):
        points = cv2.findNonZero(mask)
        if points is not None:
            (x, y), radius = cv2.minEnclosingCircle(cv2.convexHull(points))
            circles[index] = (x, y, radius)
            present[index] = True
    return torch.from_numpy(circles), torch.from_numpy(present)


def circle_regression_loss(circles: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Average normalized Smooth L1 over x, y, and radius on positive labels."""
    targets, present = circle_targets(masks)
    present = present.to(circles.device)
    error = F.smooth_l1_loss(
        circles / MODEL_IMAGE_SIZE,
        targets.to(circles) / MODEL_IMAGE_SIZE,
        beta=SMOOTH_L1_BETA_PIXELS / MODEL_IMAGE_SIZE,
        reduction="none",
    ).mean(dim=1)
    return (error * present).sum() / present.sum().clamp_min(1)


def gate_probabilities(probabilities: torch.Tensor, circles: torch.Tensor) -> torch.Tensor:
    """Keep probabilities inside a padded circle and fade them outside it."""
    height, width = probabilities.shape[-2:]
    x, y, radius = circles.to(probabilities)[:, :, None, None].unbind(dim=1)
    columns = torch.arange(width, device=probabilities.device)
    rows = torch.arange(height, device=probabilities.device)[:, None]
    distance = torch.sqrt((columns - x).square() + (rows - y).square())
    weights = (radius + GATE_MARGIN_PIXELS + GATE_FADE_PIXELS - distance) / GATE_FADE_PIXELS
    return probabilities * weights.clamp(0, 1).unsqueeze(1)
