"""Explanations for the MRI segmenter, each paired with a check that can fail.

* Uncertainty from test-time augmentation. If it means anything, the pixels the
  model is unsure about should be the pixels it gets wrong, which is measurable
  against the expert masks.
* Seg-Grad-CAM (Vinogradova et al., AAAI 2020), the segmentation adaptation of
  the Grad-CAM that Sahana NS applied to the scan-level classifier in
  notebooks/MRI.ipynb. Checked two ways: whether the map depends on learned
  weights at all (model randomization, Adebayo et al., NeurIPS 2018), and whether
  it concentrates on the heart rather than elsewhere in the chest (localisation
  against the expert masks).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from .mri_dataset import ACDCSlices

# (rotation in degrees, scale). Small enough to stay within the training
# augmentation, so every view is one the model has learned to handle.
TTA_TRANSFORMS = ((0, 1.0), (-10, 1.0), (10, 1.0), (0, 0.92), (0, 1.08), (-6, 0.96), (6, 1.04))


def warp(x: torch.Tensor, angle_degrees: float, scale: float) -> torch.Tensor:
    """Resample x at rotated and scaled coordinates.

    The sampling matrix is R(angle) / scale, whose inverse is scale * R(-angle), so
    warp(warp(x, a, s), -a, 1/s) returns x up to interpolation.
    """
    radians = math.radians(angle_degrees)
    cos, sin = math.cos(radians) / scale, math.sin(radians) / scale
    theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=x.dtype, device=x.device)
    grid = F.affine_grid(theta.expand(x.shape[0], 2, 3), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


@torch.no_grad()
def tta_predict(model: nn.Module, inputs: torch.Tensor, transforms=TTA_TRANSFORMS):
    """Mean class probabilities over augmented views, each mapped back to the original frame.

    Returns the mean probabilities [B, C, H, W] and their entropy [B, H, W], which
    is zero where every view agrees with certainty and ln(C) where they are uniform.
    """
    total = None
    for angle, scale in transforms:
        identity = angle == 0 and scale == 1.0
        probabilities = model(inputs if identity else warp(inputs, angle, scale)).softmax(dim=1)
        if not identity:
            probabilities = warp(probabilities, -angle, 1.0 / scale)
        total = probabilities if total is None else total + probabilities
    mean = total / len(transforms)
    mean = mean / mean.sum(dim=1, keepdim=True).clamp_min(1e-8)
    entropy = -(mean * mean.clamp_min(1e-8).log()).sum(dim=1)
    return mean, entropy


@torch.no_grad()
def uncertainty_for_patients(model, images, masks, patient, phase, patients, device, batch_size: int = 16):
    """TTA prediction and entropy for every slice of the given patients, in slice order."""
    dataset = ACDCSlices(images, masks, patient, phase, patients, augment=False)
    model.eval()
    predictions, entropies = [], []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        batch = torch.stack([dataset[i][0] for i in range(start, stop)]).to(device)
        mean, entropy = tta_predict(model, batch)
        predictions.append(mean.argmax(dim=1).cpu().numpy().astype(np.uint8))
        entropies.append(entropy.cpu().numpy().astype(np.float32))
    return dataset.rows, np.concatenate(predictions), np.concatenate(entropies)


def heart_region(truth: np.ndarray, prediction: np.ndarray, margin: int = 5) -> np.ndarray:
    """Pixels within `margin` in-plane of either mask.

    Scoring the whole image would reward the model for being certain about empty
    lung and air far from the heart, which says nothing about its judgement.
    """
    structure = np.ones((1, 2 * margin + 1, 2 * margin + 1), dtype=bool)
    return ndimage.binary_dilation((truth > 0) | (prediction > 0), structure=structure)


def uncertainty_checks(rows, predictions, entropy, masks, patient, phase, max_pixels: int = 2_000_000,
                       seed: int = 42) -> dict:
    """Does uncertainty point at errors?

    Pixel level: AUROC of entropy as a detector of misclassified pixels near the
    heart; 0.5 means uncertainty is unrelated to error. Volume level: Spearman
    correlation between a volume's mean uncertainty and its Dice error.
    """
    from .mri_data import STRUCTURES
    from .mri_training import dice

    truth = masks[rows]
    region = heart_region(truth, predictions)
    errors = (predictions != truth)[region]
    scores = entropy[region]
    if len(errors) > max_pixels:
        chosen = np.random.default_rng(seed).choice(len(errors), max_pixels, replace=False)
        errors, scores = errors[chosen], scores[chosen]
    pixel_auroc = float(roc_auc_score(errors, scores)) if 0 < errors.sum() < len(errors) else float("nan")

    keys = patient[rows].astype(np.int64) * 2 + phase[rows]
    volume_uncertainty, volume_error = [], []
    for key in np.unique(keys):
        selected = keys == key
        volume_region = heart_region(truth[selected], predictions[selected])
        volume_uncertainty.append(float(entropy[selected][volume_region].mean()))
        mean_dice = np.mean([dice(predictions[selected], truth[selected], label) for label in STRUCTURES])
        volume_error.append(1.0 - float(mean_dice))
    correlation = spearmanr(volume_uncertainty, volume_error)
    return {
        "pixel_error_detection_auroc": pixel_auroc,
        "pixel_error_rate_near_heart": float(errors.mean()),
        "volume_uncertainty_vs_dice_error_spearman": float(correlation.statistic),
        "volume_uncertainty_vs_dice_error_p": float(correlation.pvalue),
        "volumes": int(len(volume_error)),
    }


def seg_grad_cam(model: nn.Module, inputs: torch.Tensor, target_class: int, layer: nn.Module) -> np.ndarray:
    """Seg-Grad-CAM for one slice: which regions supported the pixels predicted as target_class.

    The score is the target logit summed over the pixels the model assigns to that
    class, so the map answers a declared question rather than an arbitrary one.
    """
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        captured["activation"] = output
        output.register_hook(lambda gradient: captured.__setitem__("gradient", gradient))

    handle = layer.register_forward_hook(forward_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(inputs)
            region = logits.argmax(dim=1)[0] == target_class
            target = logits[0, target_class]
            score = target[region].sum() if region.any() else target.sum()
            score.backward()
    finally:
        handle.remove()

    weights = captured["gradient"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * captured["activation"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return (cam / cam.max().clamp_min(1e-12)).detach().cpu().numpy()


def randomized_copy(model: nn.Module, seed: int = 0) -> nn.Module:
    """The same architecture with every convolution and linear layer re-initialised.

    BatchNorm running statistics are kept. A map that looks the same on this copy
    is describing the input or the architecture, not anything the model learned.
    """
    clone = copy.deepcopy(model)
    torch.manual_seed(seed)
    for module in clone.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.reset_parameters()
    return clone


def map_similarity(first: np.ndarray, second: np.ndarray, size: int = 48) -> float:
    """Spearman rank correlation between two maps, compared at a coarse grid."""
    def shrink(values):
        tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))[None, None]
        return F.adaptive_avg_pool2d(tensor, size).flatten().numpy()
    a, b = shrink(first), shrink(second)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def localisation(cam: np.ndarray, heart: np.ndarray) -> dict[str, float]:
    """Share of the map's mass inside the heart, against the share a uniform map would put there."""
    total = float(cam.sum())
    inside = float(cam[heart].sum()) / total if total > 0 else float("nan")
    return {"mass_inside_heart": inside, "uniform_map_baseline": float(heart.mean())}
