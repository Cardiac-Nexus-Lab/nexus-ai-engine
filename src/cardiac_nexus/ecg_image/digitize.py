"""Recover a 1D waveform from an image of a printed ECG trace.

Framed as localisation rather than regression. For each pixel column of a lead
strip the model emits a distribution over rows, and the waveform value for that
column is the expectation of that distribution. Two properties follow, and both
matter:

* the output is sub-pixel accurate, because an expectation over a softmax is
  continuous while an argmax would quantise to whole rows;
* the distribution's spread is a usable confidence signal. Where the trace is
  crossed by a gridline, obscured by a crease, or simply absent, the model
  spreads its mass instead of committing, and that column can be flagged rather
  than silently filled with a confident guess.

Direct regression to a scalar per column gives neither. It also handles ambiguity
badly: with two plausible row positions, a regressor is pulled towards the
midpoint, which is a value the trace never occupied.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# A strip covers 2.5 s of one lead. Rendering at 4 px/mm and 25 mm/s gives
# 100 px per second, so at 100 Hz sampling one pixel is exactly one sample and no
# resampling is needed. Spanning the full 10 s at this width would instead put a
# whole 80 ms QRS complex into two columns, erasing the morphology that matters.
STRIP_SECONDS = 2.5
STRIP_WIDTH = 256
STRIP_HEIGHT = 96


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=stride, stride=stride) if stride != (1, 1) else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TraceLocalizer(nn.Module):
    """Predicts, for every column, a distribution over row positions of the trace.

    Input  : [batch, 3, height, width]
    Output : logits [batch, height, width]

    Width is pooled early and the logits are interpolated back at the end. Holding
    full width through every block, so that each output column came from its own
    input column, put all of the compute into convolving 128 channels across 512
    columns: 618 ms per step for a batch of 12 at 96x256, which is hours per epoch.
    Since a trace's vertical position varies smoothly along time, predicting it at
    reduced width and upsampling loses very little, and costs roughly an order of
    magnitude less.
    """

    def __init__(self, height: int = STRIP_HEIGHT, base_channels: int = 16, width_reduction: int = 4):
        super().__init__()
        self.height = height
        self.width_reduction = width_reduction
        self.encoder = nn.Sequential(
            _ConvBlock(3, base_channels, stride=(2, 2)),                        # H/2,  W/2
            _ConvBlock(base_channels, base_channels * 2, stride=(2, 2)),        # H/4,  W/4
            _ConvBlock(base_channels * 2, base_channels * 4, stride=(2, 1)),    # H/8,  W/4
            _ConvBlock(base_channels * 4, base_channels * 4, stride=(1, 1)),
        )
        # Collapse the remaining height into channels, then expand back to a
        # per-row score for every retained column.
        self.head = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, (height // 8, 1), bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, height, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = x.shape[-1]
        features = self.encoder(x)
        logits = self.head(features).squeeze(2)   # [batch, height, width/reduction]
        # Back to one column per input column, so the caller's indexing is unchanged.
        return nn.functional.interpolate(logits, size=width, mode="linear", align_corners=False)


def expected_rows(logits: torch.Tensor) -> torch.Tensor:
    """Expectation of the row distribution for each column, in pixels."""
    probabilities = torch.softmax(logits, dim=1)
    rows = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
    return torch.einsum("bhw,h->bw", probabilities, rows)


def row_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Peak probability per column: low where the model cannot find the trace."""
    return torch.softmax(logits, dim=1).max(dim=1).values


def soft_target(rows: torch.Tensor, height: int, sigma: float = 1.5) -> torch.Tensor:
    """Gaussian target centred on the true row of each column.

    A one-hot target would call a one-pixel miss as wrong as a fifty-pixel miss.
    Spreading the target over neighbouring rows encodes that being close is better
    than being far, which is what the task actually rewards.
    """
    grid = torch.arange(height, device=rows.device, dtype=rows.dtype)
    distance = grid.view(1, -1, 1) - rows.unsqueeze(1)
    weights = torch.exp(-0.5 * (distance / sigma) ** 2)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)


class TraceLoss(nn.Module):
    """Cross-entropy against the soft target, plus a penalty on the expectation.

    Cross-entropy alone shapes the distribution but does not directly constrain
    the value that is read out of it. Adding an L1 term on the expectation ties
    training to the quantity actually used at inference.
    """

    def __init__(self, expectation_weight: float = 0.1, sigma: float = 1.5):
        super().__init__()
        self.expectation_weight = expectation_weight
        self.sigma = sigma

    def forward(self, logits: torch.Tensor, true_rows: torch.Tensor) -> torch.Tensor:
        height = logits.shape[1]
        target = soft_target(true_rows, height, self.sigma)
        log_probabilities = torch.log_softmax(logits, dim=1)
        cross_entropy = -(target * log_probabilities).sum(dim=1).mean()

        predicted = expected_rows(logits)
        expectation_error = (predicted - true_rows).abs().mean() / height
        return cross_entropy + self.expectation_weight * expectation_error


def rows_to_millivolts(rows: np.ndarray, baseline_row: float, pixels_per_mv: float) -> np.ndarray:
    """Convert row positions to millivolts. Rows increase downwards, voltage upwards."""
    return (baseline_row - rows) / pixels_per_mv
