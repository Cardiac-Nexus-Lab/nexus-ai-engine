"""Cardiac MRI models.

The encoder follows the contract in `models.py`: it maps its input to an
EMBEDDING_DIM vector, so a later fusion head can combine it with the ECG encoder
without either being rewritten.

Segmentation needs more than that vector. A U-Net decoder rebuilds full
resolution from the encoder's intermediate feature maps, so the encoder exposes
them through `forward_features`, while `forward` returns only the embedding. The
segmenter reaches inside; a fusion or diagnosis head never has to.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .models import EMBEDDING_DIM

NUM_CLASSES = 4  # background, right ventricle, myocardium, left ventricle
WIDTHS = (32, 64, 128, 256, 512)


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MRIEncoder(nn.Module):
    """2D convolutional encoder over a short-axis slice and its neighbours.

    Input channels are adjacent slices (2.5D), which gives through-plane context
    at almost no memory cost while keeping each slice a training example.
    """

    def __init__(self, in_channels: int = 3, widths: tuple[int, ...] = WIDTHS,
                 embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        stages, channels = [], in_channels
        for width in widths:
            stages.append(_DoubleConv(channels, width))
            channels = width
        self.stages = nn.ModuleList(stages)
        self.project = nn.Linear(widths[-1], embedding_dim)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        for index, stage in enumerate(self.stages):
            if index:
                x = F.max_pool2d(x, 2)
            x = stage(x)
            skips.append(x)
        embedding = self.project(F.adaptive_avg_pool2d(x, 1).flatten(1))
        return embedding, skips

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[0]


class MRISegmenter(nn.Module):
    """U-Net: the shared encoder plus a decoder head producing per-pixel logits."""

    def __init__(self, in_channels: int = 3, num_classes: int = NUM_CLASSES,
                 widths: tuple[int, ...] = WIDTHS):
        super().__init__()
        self.encoder = MRIEncoder(in_channels, widths)
        self.decoder = nn.ModuleList(
            _DoubleConv(widths[level + 1] + widths[level], widths[level])
            for level in reversed(range(len(widths) - 1))
        )
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, skips = self.encoder.forward_features(x)
        y = skips[-1]
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            # Bilinear upsampling rather than transposed convolution: no
            # checkerboard artefacts, and it matches the skip size exactly even
            # when an input dimension is not divisible by 16.
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))
        return self.head(y)


class DiceCELoss(nn.Module):
    """Cross-entropy plus soft Dice over the three cardiac structures.

    Cross-entropy alone is dominated by background, which is most of every slice;
    Dice measures overlap per structure regardless of its size, so the thin
    myocardium counts as much as the large cavity. Dice is pooled over the batch
    rather than per slice, because apical and basal slices often contain no
    structure at all and a per-slice Dice is undefined there.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dice_weight: float = 1.0, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, target)
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).to(probabilities.dtype)
        dims = (0, 2, 3)
        overlap = (probabilities * one_hot).sum(dims)
        total = probabilities.sum(dims) + one_hot.sum(dims)
        dice = (2 * overlap + self.smooth) / (total + self.smooth)
        return cross_entropy + self.dice_weight * (1 - dice[1:].mean())
