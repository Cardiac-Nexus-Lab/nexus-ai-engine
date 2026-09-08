"""1D XResNet for ECG classification.

Strodthoff et al. (IEEE JBHI 2021, arXiv:2004.13701) benchmarked architectures on
PTB-XL and found xresnet1d101 best on the diagnostic superclass task, at 0.928
macro AUC against the 0.905 reached here by a plain three-layer CNN.

This is a 1D adaptation of the XResNet refinements from "Bag of Tricks for Image
Classification" (He et al., 2018) over the original ResNet:

* a three-convolution stem instead of a single wide kernel, which sees the same
  receptive field with fewer parameters and more non-linearity;
* downsampling moved off the 1x1 convolution and onto the 3x3, so the shortcut
  path stops discarding three quarters of its input;
* average pooling in the identity path when shapes change, rather than a strided
  1x1 convolution that samples one position in four;
* the final BatchNorm of each block initialised to zero, so every block starts as
  an identity mapping and the network begins as if it were much shallower.

The encoder returns an embedding rather than logits, matching ECGEncoder, so the
two are interchangeable and either can later become one branch of a multimodal
model without changes.
"""

from __future__ import annotations

import torch
from torch import nn

from .models import EMBEDDING_DIM


def _conv_layer(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, zero_bn: bool = False,
                activation: bool = True) -> nn.Sequential:
    batch_norm = nn.BatchNorm1d(out_channels)
    # Zeroing the last BatchNorm of a block makes the residual branch output zero
    # at initialisation, so the block starts as an identity and gradients flow
    # through the shortcut. This is what allows the deeper variants to train.
    nn.init.constant_(batch_norm.weight, 0.0 if zero_bn else 1.0)

    layers: list[nn.Module] = [
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
        batch_norm,
    ]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """Basic (expansion 1) or bottleneck (expansion 4) residual block."""

    def __init__(self, expansion: int, in_channels: int, mid_channels: int, stride: int = 1, kernel_size: int = 5):
        super().__init__()
        out_channels = mid_channels * expansion
        in_channels = in_channels * expansion if expansion > 1 else in_channels

        if expansion == 1:
            layers = [
                _conv_layer(in_channels, mid_channels, kernel_size, stride=stride),
                _conv_layer(mid_channels, out_channels, kernel_size, zero_bn=True, activation=False),
            ]
        else:
            layers = [
                _conv_layer(in_channels, mid_channels, 1),
                # Stride sits on the wide convolution, not the 1x1, so the block
                # downsamples after looking at a full neighbourhood.
                _conv_layer(mid_channels, mid_channels, kernel_size, stride=stride),
                _conv_layer(mid_channels, out_channels, 1, zero_bn=True, activation=False),
            ]
        self.convolutions = nn.Sequential(*layers)

        # Identity path: pool first, then match channels with a 1x1. A strided 1x1
        # alone would read one sample in `stride` and throw the rest away.
        identity: list[nn.Module] = []
        if stride != 1:
            identity.append(nn.AvgPool1d(stride, ceil_mode=True))
        if in_channels != out_channels:
            identity.append(_conv_layer(in_channels, out_channels, 1, activation=False))
        self.identity = nn.Sequential(*identity)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.convolutions(x) + self.identity(x))


class XResNet1d(nn.Module):
    """Configurable 1D XResNet encoder producing a fixed-width embedding."""

    def __init__(
        self,
        expansion: int,
        layer_sizes: list[int],
        in_leads: int = 12,
        embedding_dim: int = EMBEDDING_DIM,
        kernel_size: int = 5,
        stem_widths: tuple[int, int, int] = (32, 32, 64),
    ):
        super().__init__()

        stem_channels = [in_leads, *stem_widths]
        stem = [
            _conv_layer(stem_channels[i], stem_channels[i + 1], kernel_size, stride=2 if i == 0 else 1)
            for i in range(3)
        ]

        block_widths = [64 // expansion, 64, 128, 256, 512]
        blocks = [
            self._make_layer(
                expansion,
                block_widths[i],
                block_widths[i + 1],
                count=layer_size,
                stride=1 if i == 0 else 2,
                kernel_size=kernel_size,
            )
            for i, layer_size in enumerate(layer_sizes)
        ]

        self.layers = nn.Sequential(
            *stem,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            *blocks,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(block_widths[len(layer_sizes)] * expansion, embedding_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _make_layer(expansion: int, in_channels: int, out_channels: int, count: int, stride: int,
                    kernel_size: int) -> nn.Sequential:
        return nn.Sequential(
            *[
                ResBlock(
                    expansion,
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride=stride if i == 0 else 1,
                    kernel_size=kernel_size,
                )
                for i in range(count)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# Depths follow the ResNet family; the paper's best performer is xresnet1d101.
_CONFIGURATIONS = {
    "xresnet1d18": (1, [2, 2, 2, 2]),
    "xresnet1d34": (1, [3, 4, 6, 3]),
    "xresnet1d50": (4, [3, 4, 6, 3]),
    "xresnet1d101": (4, [3, 4, 23, 3]),
}


def build_encoder(name: str, in_leads: int = 12, embedding_dim: int = EMBEDDING_DIM) -> XResNet1d:
    if name not in _CONFIGURATIONS:
        raise ValueError(f"unknown architecture {name!r}; available: {sorted(_CONFIGURATIONS)}")
    expansion, layer_sizes = _CONFIGURATIONS[name]
    return XResNet1d(expansion, layer_sizes, in_leads=in_leads, embedding_dim=embedding_dim)


class XResNet1dClassifier(nn.Module):
    """Encoder plus linear head, matching the ECGClassifier interface."""

    def __init__(self, num_classes: int, architecture: str = "xresnet1d101", embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.encoder = build_encoder(architecture, embedding_dim=embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
