"""Use the predicted center as a soft preference between whole mask components."""

import cv2
import numpy as np

CENTER_SCALE_PIXELS = 12.0
AREA_SCALE_PIXELS = 64.0


def select_predicted_center_component(
    probability_map: np.ndarray,
    binary_mask: np.ndarray,
    center: tuple[float, float],
) -> np.ndarray:
    """Keep one 8-connected blob when the mask contains multiple blobs."""
    mask = np.asarray(binary_mask, dtype=bool)
    probabilities = np.asarray(probability_map, dtype=np.float32)
    if mask.ndim != 2 or probabilities.shape != mask.shape:
        raise ValueError("Probability map and binary mask must be matching two-dimensional arrays.")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 2 or not np.isfinite(center).all():
        return mask

    x, y = center
    rows, columns = np.nonzero(mask)
    foreground_labels = labels[rows, columns]
    squared_distance = (columns - x) ** 2 + (rows - y) ** 2
    nearest_squared = np.full(count, np.inf)
    np.minimum.at(nearest_squared, foreground_labels, squared_distance)
    area = stats[1:, cv2.CC_STAT_AREA]
    confidence = np.bincount(foreground_labels, weights=probabilities[mask], minlength=count)[1:]
    confidence /= area
    area_support = -np.expm1(-area / AREA_SCALE_PIXELS)
    # A center outside the visible mask still gives a useful distance to its nearest pixel.
    center_support = 0.5 + 0.5 / (1 + nearest_squared[1:] / CENTER_SCALE_PIXELS**2)
    scores = confidence * area_support * center_support
    return labels == int(np.argmax(scores) + 1)
