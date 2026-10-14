"""Anatomical post-processing of predicted cardiac MRI masks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .mri_data import STRUCTURES


def keep_largest_component(volume: np.ndarray) -> np.ndarray:
    """Keep only the largest 3D connected region of each structure.

    A heart has one LV cavity, one myocardium and one RV. Small islands of a
    label elsewhere in the volume, such as in the liver or a lung, are always
    errors; they barely move Dice but dominate the Hausdorff distance.

    Connectivity is the default face-adjacency in 3D, across the [slices, H, W]
    stack. Slices are typically 5-10 mm apart, so a structure whose cross-sections
    do not overlap between neighbouring slices can be split and lose its smaller
    part; the evaluation reports results with and without this step.
    """
    cleaned = volume.copy()
    for label in STRUCTURES:
        components, count = ndimage.label(volume == label)
        if count > 1:
            largest = np.argmax(np.bincount(components.ravel())[1:]) + 1
            cleaned[(components != largest) & (volume == label)] = 0
    return cleaned
