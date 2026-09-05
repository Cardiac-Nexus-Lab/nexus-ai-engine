"""Model definitions.

Each modality gets its own encoder that maps its input to a fixed-length
embedding. Once every modality produces a vector of the same width they become
directly comparable, and a small fusion head can combine them. The ECG encoder
below is the first of those; MRI and tabular encoders follow the same contract.
"""

from __future__ import annotations

import torch
from torch import nn

EMBEDDING_DIM = 128


class ECGEncoder(nn.Module):
    """1D CNN over a 12-lead, 1000-sample recording.

    Returns an embedding rather than a prediction so the same trained weights can
    be reused unchanged as one branch of a later multimodal model.
    """

    def __init__(self, in_leads: int = 12, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_leads, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, embedding_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


class ECGClassifier(nn.Module):
    """ECG encoder plus a linear head, for single-modality training.

    The architecture matches notebooks/01 and 02 so results stay comparable; only
    the encoder/head split is new, and it does not change the computation.
    """

    def __init__(self, num_classes: int, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.encoder = ECGEncoder(embedding_dim=embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def select_device(prefer: str | None = None) -> torch.device:
    """Pick the fastest device available.

    On Apple silicon this is the MPS backend, which for this model measured
    roughly 12x faster than the CPU path on an M1.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
