"""Training examples for cardiac MRI segmentation.

Each example is one short-axis slice with its immediate neighbours as extra
channels (2.5D) and the expert mask for the centre slice.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_volumes(images: np.ndarray, patient: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Standardize intensity per volume.

    MRI intensity has no physical unit and varies between scanners and sessions,
    so one global scale would teach the model scanner brightness. Clipping the
    extreme 0.5% first stops a few very bright voxels, often flowing blood, from
    compressing the useful range.
    """
    out = np.empty_like(images, dtype=np.float32)
    keys = patient.astype(np.int64) * 2 + phase
    for key in np.unique(keys):
        selected = keys == key
        volume = images[selected]
        low, high = np.percentile(volume, (0.5, 99.5))
        volume = np.clip(volume, low, high)
        out[selected] = (volume - volume.mean()) / (volume.std() + 1e-6)
    return out


def neighbour_indices(patient: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the slice above and below each slice, within the same volume.

    At the first and last slice the missing neighbour repeats the slice itself,
    so an apical slice is not given anatomy from a different volume.
    """
    n = len(patient)
    index = np.arange(n)
    same = lambda a, b: (patient[a] == patient[b]) & (phase[a] == phase[b])
    previous = np.where((index > 0) & same(np.maximum(index - 1, 0), index), index - 1, index)
    following = np.where((index < n - 1) & same(np.minimum(index + 1, n - 1), index), index + 1, index)
    return previous, following


class ACDCSlices(Dataset):
    """Slices belonging to a set of patients, optionally augmented.

    Augmentation deliberately excludes mirror flips. A flipped heart places the
    right ventricle on the wrong side, which is anatomy no scan will present, the
    same reason the ECG augmentation refuses to permute leads.
    """

    def __init__(self, images: np.ndarray, masks: np.ndarray, patient: np.ndarray, phase: np.ndarray,
                 patients: np.ndarray, augment: bool = False, seed: int = 0):
        self.images, self.masks = images, masks
        self.previous, self.following = neighbour_indices(patient, phase)
        self.rows = np.flatnonzero(np.isin(patient, patients))
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int):
        row = self.rows[item]
        stack = np.stack([self.images[self.previous[row]], self.images[row], self.images[self.following[row]]])
        mask = self.masks[row]
        if self.augment:
            stack, mask = augment_slice(stack, mask, self.rng)
        return torch.from_numpy(np.ascontiguousarray(stack)), torch.from_numpy(mask.astype(np.int64))


def augment_slice(stack: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Random affine warp shared by image and mask, then intensity changes."""
    height, width = mask.shape
    angle = rng.uniform(-15, 15)
    scale = rng.uniform(0.85, 1.15)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[:, 2] += rng.uniform(-0.06, 0.06, 2) * (width, height)

    background = float(stack.min())
    stack = np.stack([
        cv2.warpAffine(channel, matrix, (width, height), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=background)
        for channel in stack
    ])
    # Nearest-neighbour for labels: interpolating class ids would invent classes
    # at every boundary, such as a sliver of myocardium between LV and background.
    mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if rng.random() < 0.5:
        low, high = stack.min(), stack.max()
        unit = (stack - low) / (high - low + 1e-6)
        stack = unit ** rng.uniform(0.7, 1.5) * (high - low) + low
    stack = stack * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
    if rng.random() < 0.2:
        stack = stack + rng.normal(0.0, 0.05, stack.shape)
    return stack.astype(np.float32), mask
