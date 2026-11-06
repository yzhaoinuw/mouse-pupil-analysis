"""On-demand mottled edge obstructions for the grouped BCE experiment."""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EDGES = ("top", "right", "bottom", "left")  # np.rot90 brings this edge to the top.
STYLES = ((3, 8, 0.55), (7, 15, 0.9), (13, 24, 1.2))  # Soft, patchy, spotty.


def _smooth_noise(rng, size, knots):
    values = rng.normal(size=knots)
    positions = np.linspace(0, knots - 1, size)
    lower = np.minimum(positions.astype(int), knots - 2)
    weight = _smoothstep(positions - lower)
    return (1 - weight) * values[lower] + weight * values[lower + 1]


def _smoothstep(value):
    value = np.clip(value, 0, 1)
    return value**2 * (3 - 2 * value)


def _content_edge(image):
    """Estimate the visible edge from the first nonblack row in each column."""
    visible = image > 5
    usable = visible[:45].any(axis=0)
    columns = np.arange(image.shape[1])
    if not usable.any():
        return np.zeros_like(columns)
    first = np.interp(columns, columns[usable], visible.argmax(axis=0)[usable])
    padded = np.pad(first, 4, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 9), axis=1)


def _field(rng, size, knots):
    grid = rng.normal(size=knots).astype(np.float32)
    return np.asarray(Image.fromarray(grid).resize(size, Image.Resampling.BICUBIC))


def mottled_edge_band(image, pupil, edge, rng):
    """Return an augmented uint8 image, or None when no usable placement fits."""
    rotation = EDGES.index(edge)
    image = np.rot90(image, rotation).astype(np.float32)
    target = np.rot90(pupil, rotation)
    height, width = image.shape
    y, x = np.ogrid[:height, :width]
    along = x[0] / (width - 1)
    anchor = _content_edge(image)
    depth = y - anchor
    low, high, strength = STYLES[int(rng.integers(len(STYLES)))]

    for attempt in range(34):
        mean_depth = rng.uniform(11, 31 if attempt < 14 else 21)
        shape = mean_depth + rng.uniform(4, 9) * _smooth_noise(rng, width, int(rng.integers(4, 8)))
        shape += rng.uniform(-12, 12) * (along - 0.5)
        shape += rng.uniform(0.6, 2.4) * _smooth_noise(rng, width, 13)
        taper = rng.random() < 0.55
        if taper:
            center, span = rng.uniform(0.10, 0.90), rng.uniform(0.19, 0.58)
            shape *= 0.13 + 0.87 * np.exp(-0.5 * ((along - center) / span) ** 2)
        shape = np.clip(shape, 1.4, 38)
        feather = np.clip(rng.uniform(2, 6) + 1.7 * _smooth_noise(rng, width, 8), 1.3, 8.5)
        coverage = _smoothstep((shape + feather - depth) / (2 * feather))
        if taper:
            coverage *= _smoothstep(shape / 9)
        coverage[image == 0] = 0
        if np.any((coverage > 0) & target):
            continue

        coarse = _field(rng, (width, height), (4, 7))
        middle = _field(rng, (width, height), (10, 17))
        detail = _field(rng, (width, height), (25, 37))
        opacity = coverage * np.clip(
            rng.uniform(0.39, 0.83) + 0.19 * coarse + 0.13 * middle, 0.10, 0.96
        )
        across = np.clip(depth / shape, 0, 1)
        gradient = rng.uniform(-54, 54) * (across - 0.5)
        waves = rng.uniform(10, 36) * np.sin(
            rng.uniform(2.5, 8) * across + 1.7 * coarse + rng.uniform(-np.pi, np.pi)
        )
        shade = rng.uniform(47, 125) + gradient + waves
        shade += rng.uniform(15, 35) * coarse + rng.uniform(6, 21) * middle
        shade += rng.uniform(1, 9) * detail
        for _ in range(int(rng.integers(low, high))):
            col = int(rng.integers(width))
            row = anchor[col] + rng.uniform(0.03, 0.93) * shape[col]
            horizontal = (x - col) / rng.uniform(1.7, 9.5)
            vertical = (y - row) / rng.uniform(1.2, 5.5)
            blob = np.exp(-0.5 * (horizontal**2 + vertical**2))
            sign = 1 if rng.random() < 0.55 else -1
            shade += sign * rng.uniform(20, 75) * strength * blob

        revised = np.rint(image + opacity * (np.clip(shade, 9, 222) - image)).astype(np.uint8)
        if np.count_nonzero(np.abs(revised.astype(np.float32) - image) >= 12) >= 220:
            return np.rot90(revised, -rotation)
    return None


class EdgeObstructionDataset(Dataset):
    """Apply a fresh obstruction draw after the ordinary paired transforms.

    The grouped trainer uses a single-process DataLoader. A separate NumPy
    stream preserves the baseline's sample order and existing transforms.
    """

    def __init__(self, dataset, probability: float, seed: int):
        self.dataset = dataset
        self.probability = probability
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 2026110519]))
        self.draws = 0
        self.attempts = 0
        self.applied = dict.fromkeys(EDGES, 0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        self.draws += 1
        if self.rng.random() >= self.probability:
            return image, target
        self.attempts += 1
        edge = EDGES[int(self.rng.integers(len(EDGES)))]
        before = np.rint(image[0].numpy() * 255).astype(np.uint8)
        revised = mottled_edge_band(before, target[0].numpy() > 0, edge, self.rng)
        if revised is None:
            return image, target
        self.applied[edge] += 1
        return torch.from_numpy(np.ascontiguousarray(revised))[None].to(image.dtype) / 255, target

    def stats(self):
        return {"draws": self.draws, "attempts": self.attempts, "applied_by_edge": self.applied}
