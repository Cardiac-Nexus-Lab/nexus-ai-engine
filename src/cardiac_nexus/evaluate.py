"""Uncertainty and calibration for a trained classifier.

A single AUROC hides two things a reader needs. First, how much of it is sampling
noise: 2,158 test recordings, of which only 262 carry HYP, cannot pin a metric
down tightly, and a point estimate implies a precision the data does not support.
Second, whether the numbers the model emits mean anything as probabilities. A
network trained with class-weighted cross-entropy is pushed towards confident
outputs and is usually badly calibrated, so "0.82" need not correspond to 82 of
100 such recordings being positive. That distinction matters the moment a
clinician reads the number rather than the ranking.

Bootstrap intervals address the first. Temperature scaling (Guo et al., 2017)
addresses the second: a single scalar divides the logits, fitted on validation
data, which cannot change any ranking and therefore leaves AUROC untouched while
correcting over-confidence.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn


def bootstrap_metric(y_true: np.ndarray, y_score: np.ndarray, metric=roc_auc_score,
                     resamples: int = 2000, alpha: float = 0.05,
                     seed: int = 42) -> dict[str, float]:
    """Percentile bootstrap interval for a metric on one class.

    Resampling recordings with replacement approximates drawing a different test
    set of the same size from the same population.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: list[float] = []
    for _ in range(resamples):
        index = rng.integers(0, n, n)
        sample_true = y_true[index]
        # A resample containing a single class leaves the metric undefined.
        if len(np.unique(sample_true)) < 2:
            continue
        values.append(metric(sample_true, y_score[index]))

    if not values:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan"), "resamples": 0}

    values = np.asarray(values)
    return {
        "estimate": float(metric(y_true, y_score)),
        "low": float(np.percentile(values, 100 * alpha / 2)),
        "high": float(np.percentile(values, 100 * (1 - alpha / 2))),
        "resamples": len(values),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> dict:
    """Gap between confidence and observed frequency, averaged over probability bins.

    Returns the per-bin detail as well, since a single number hides whether the
    model is over-confident everywhere or only at the extremes.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    rows = []

    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        count = int(in_bin.sum())
        if count == 0:
            rows.append({"lower": float(lower), "upper": float(upper), "count": 0,
                         "confidence": None, "observed": None})
            continue
        confidence = float(y_prob[in_bin].mean())
        observed = float(y_true[in_bin].mean())
        error += (count / total) * abs(confidence - observed)
        rows.append({"lower": float(lower), "upper": float(upper), "count": count,
                     "confidence": confidence, "observed": observed})

    return {"ece": float(error), "bins": rows}


class VectorScaler(nn.Module):
    """Per-class slope and intercept fitted on held-out data.

    A single shared temperature is the usual recipe, but it assumes every class is
    over-confident by the same factor. That does not hold here: training used
    per-class positive weighting, from 1.25 for NORM to 7.06 for HYP, which biases
    each class differently. Fitting one scalar per class corrects the scale, and
    the intercept corrects the shift that the weighting introduced.

    Each transform is monotonic within its class, so ranking is preserved and
    AUROC and average precision are unchanged by construction.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    @property
    def scales(self) -> np.ndarray:
        return self.log_scale.exp().detach().numpy()

    @property
    def biases(self) -> np.ndarray:
        return self.bias.detach().numpy()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.log_scale.exp() + self.bias

    def fit(self, logits: np.ndarray, targets: np.ndarray, steps: int = 300, lr: float = 0.05) -> dict:
        logit_tensor = torch.tensor(logits, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        # Unweighted: weighting shapes the decision boundary during training, but
        # calibration must match the real class frequencies.
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.log_scale, self.bias], lr=lr, max_iter=steps)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self(logit_tensor), target_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        return {"scales": self.scales.tolist(), "biases": self.biases.tolist()}

    def apply(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self(torch.tensor(logits, dtype=torch.float32))).numpy()


class TemperatureScaler(nn.Module):
    """Single scalar dividing the logits, fitted on held-out data.

    Kept for comparison against VectorScaler; on this model it does not help,
    for the reason described there.
    """

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temperature.exp()

    def fit(self, logits: np.ndarray, targets: np.ndarray, steps: int = 200, lr: float = 0.01) -> float:
        """Optimise temperature against validation negative log-likelihood."""
        logit_tensor = torch.tensor(logits, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        # Unweighted loss here: weighting exists to shape the decision boundary
        # during training, but calibration must match the real class frequencies.
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=steps)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self(logit_tensor), target_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature

    def apply(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self(torch.tensor(logits, dtype=torch.float32))).numpy()


def evaluate_with_uncertainty(y_true: np.ndarray, y_prob: np.ndarray, classes: list[str],
                              resamples: int = 2000, seed: int = 42) -> list[dict]:
    """Per-class AUROC and average precision, each with a bootstrap interval."""
    rows = []
    for index, name in enumerate(classes):
        true_i, prob_i = y_true[:, index], y_prob[:, index]
        rows.append(
            {
                "class": name,
                "support": int(true_i.sum()),
                "auroc": bootstrap_metric(true_i, prob_i, roc_auc_score, resamples, seed=seed),
                "average_precision": bootstrap_metric(
                    true_i, prob_i, average_precision_score, resamples, seed=seed
                ),
                "calibration": expected_calibration_error(true_i, prob_i),
            }
        )
    return rows
