"""Weak supervision of the MRI segmenter from clinician-measured LV volumes.

The Kaggle cardiac cohort gives, per patient, the end-diastolic and end-systolic
LV volume a clinician measured, but no masks. Two signals are taken from it:

* a differentiable volume: the softmax probability of the LV class, summed over
  every pixel of every slice and multiplied by the voxel volume, is a soft LV
  volume. Its Huber distance to the clinician's EDV and ESV, and that of the
  implied EF to the clinician's EF, can be back-propagated;
* silver-standard LV masks: where the current model's own masks already give
  EDV, ESV and EF within a tight tolerance of the clinician's values, those
  masks are kept as LV-only targets.

Both supervise the LV alone. A silver mask says nothing about RV or myocardium,
so it is scored as LV versus everything else, never as a four-class target, and
the gold-standard patients keep training all three structures.

The ED and ES frames used here are the ones `scripts/validate_mri_kaggle.py
--save-edes` chose (largest and smallest predicted LV volume), so the frame
choice itself comes from the model and is a source of error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .mri_dataset import neighbour_indices, normalize_volumes

LV = 3


class KaggleVolumes:
    """ED and ES stacks of Kaggle patients with clinician volumes, one patient at a time."""

    def __init__(self, edes_dir: Path, table: pd.DataFrame, silver: set[str]):
        self.items = []
        for _, row in table.iterrows():
            path = edes_dir / f"{row['pid']}.npz"
            if not path.exists():
                continue
            self.items.append({"path": path, "pid": str(row["pid"]), "edv": float(row["edv_true"]),
                               "esv": float(row["esv_true"]), "silver": str(row["pid"]) in silver})

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def stack(volume: np.ndarray) -> torch.Tensor:
        """[slices, H, W] uint8 -> normalized 2.5D input [slices, 3, H, W]."""
        volume = volume.astype(np.float32)
        slices = len(volume)
        ids = np.zeros(slices, dtype=np.int16)
        phase = np.zeros(slices, dtype=np.int8)
        images = normalize_volumes(volume, ids, phase)
        previous, following = neighbour_indices(ids, phase)
        return torch.from_numpy(np.stack([images[previous], images, images[following]], axis=1))

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        with np.load(item["path"]) as stored:
            spacing = stored["spacing"]
            return {**item, "voxel_ml": float(np.prod(spacing)) / 1000.0,
                    "ed": self.stack(stored["ed_image"]), "es": self.stack(stored["es_image"]),
                    "ed_lv": torch.from_numpy(stored["ed_lv"].astype(np.float32)),
                    "es_lv": torch.from_numpy(stored["es_lv"].astype(np.float32))}


def soft_volume_ml(logits: torch.Tensor, voxel_ml: float) -> torch.Tensor:
    """Differentiable LV volume of one stack of slices, in millilitres."""
    return logits.float().softmax(dim=1)[:, LV].sum() * voxel_ml


def volume_loss(ed_logits, es_logits, voxel_ml: float, edv: float, esv: float) -> tuple[torch.Tensor, dict]:
    """Huber loss on EDV and ESV (per 100 ml) and on EF (as a fraction)."""
    edv_hat, esv_hat = soft_volume_ml(ed_logits, voxel_ml), soft_volume_ml(es_logits, voxel_ml)
    ef_hat = 1 - esv_hat / edv_hat.clamp(min=1.0)
    ef = 1 - esv / edv
    target = lambda value: torch.tensor(value, dtype=torch.float32)
    loss = (F.smooth_l1_loss(edv_hat / 100, target(edv / 100), beta=0.1)
            + F.smooth_l1_loss(esv_hat / 100, target(esv / 100), beta=0.1)
            + F.smooth_l1_loss(ef_hat, target(ef), beta=0.05))
    return loss, {"edv_hat": float(edv_hat), "esv_hat": float(esv_hat)}


def lv_mask_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy plus soft Dice on the LV probability against a silver LV mask."""
    probability = logits.float().softmax(dim=1)[:, LV].clamp(1e-6, 1 - 1e-6)
    cross_entropy = F.binary_cross_entropy(probability, mask)
    dice = (2 * (probability * mask).sum() + 1e-5) / (probability.sum() + mask.sum() + 1e-5)
    return cross_entropy + (1 - dice)


def silver_patients(results: pd.DataFrame, method: str = "frustum", ef_tolerance: float = 3.0,
                    volume_tolerance: float = 7.0) -> set[str]:
    """Patients whose predicted EDV, ESV and EF already match the clinician's closely."""
    close = ((results[f"ef_{method}"] - results["ef_true"]).abs() <= ef_tolerance) \
        & ((results[f"edv_{method}"] - results["edv_true"]).abs() <= volume_tolerance) \
        & ((results[f"esv_{method}"] - results["esv_true"]).abs() <= volume_tolerance)
    return set(results.loc[close, "pid"].astype(str))
