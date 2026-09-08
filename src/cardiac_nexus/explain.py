"""Attribution and its validation.

Attribution maps are easy to produce and easy to over-trust. Evaluating twelve
methods on ECG classifiers, Bender et al. (Eur Heart J Digital Health, 2026,
"Signal or noise?") found that methods disagree with one another, that
self-consistency between models differing only by random seed sits around
0.41-0.65, and that some methods keep producing plausible-looking maps even
after the weights are randomised. Their recommendation is to treat attribution
as exploratory and to run sanity checks rather than presenting heatmaps as
evidence.

This module therefore ships the check alongside the method:

* `attribute` computes Integrated Gradients for a chosen class;
* `smoothgrad_attribute` averages over noised copies, which the same study found
  more stable than plain gradients;
* `model_randomization_test` implements the check from Adebayo et al. (2018),
  "Sanity Checks for Saliency Maps": recompute attribution against randomised
  weights and measure how much the map changes. A map that survives
  randomisation is describing the input, not anything the model learned.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from captum.attr import IntegratedGradients, NoiseTunnel
from scipy.stats import spearmanr
from torch import nn

IG_STEPS = 64


class _ProbabilityWrapper(nn.Module):
    """Attribute the predicted probability rather than the raw logit.

    Attribution magnitudes then refer to the quantity a reader is actually shown.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


@dataclass
class Attribution:
    values: np.ndarray  # [12, 1000]
    probability: float
    convergence_delta: float
    target_class: int


def _prepare(signal: np.ndarray, device: torch.device) -> torch.Tensor:
    # Signals may arrive as a read-only view of a memory-mapped cache; copy so the
    # tensor owns writable storage and gradients can be attached without warnings.
    tensor = torch.tensor(np.ascontiguousarray(signal), dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.requires_grad_()


def attribute(model: nn.Module, signal: np.ndarray, target_class: int,
              device: torch.device | None = None, steps: int = IG_STEPS) -> Attribution:
    """Integrated Gradients for one recording and one class.

    The baseline is a zero signal, which for a standardized ECG is the absence of
    deflection -- a more meaningful reference than the arbitrary baselines used
    for images.
    """
    device = device or next(model.parameters()).device
    wrapped = _ProbabilityWrapper(model).to(device).eval()
    inputs = _prepare(signal, device)

    values, delta = IntegratedGradients(wrapped).attribute(
        inputs,
        baselines=torch.zeros_like(inputs),
        target=target_class,
        n_steps=steps,
        return_convergence_delta=True,
    )
    with torch.no_grad():
        probability = wrapped(inputs.detach())[0, target_class].item()

    return Attribution(
        values=values.squeeze(0).detach().cpu().numpy(),
        probability=probability,
        convergence_delta=float(delta.item()),
        target_class=target_class,
    )


def smoothgrad_attribute(model: nn.Module, signal: np.ndarray, target_class: int,
                         device: torch.device | None = None, samples: int = 16,
                         noise_fraction: float = 0.1) -> Attribution:
    """Integrated Gradients averaged over noised copies of the input.

    Averaging suppresses the sample-to-sample jitter that makes single-pass
    gradient maps unstable; noise is scaled to the signal's own spread so the
    perturbation stays proportionate.
    """
    device = device or next(model.parameters()).device
    wrapped = _ProbabilityWrapper(model).to(device).eval()
    inputs = _prepare(signal, device)

    values = NoiseTunnel(IntegratedGradients(wrapped)).attribute(
        inputs,
        baselines=torch.zeros_like(inputs),
        target=target_class,
        n_steps=IG_STEPS,
        nt_type="smoothgrad",
        nt_samples=samples,
        stdevs=float(noise_fraction * signal.std()),
    )
    with torch.no_grad():
        probability = wrapped(inputs.detach())[0, target_class].item()

    return Attribution(
        values=values.squeeze(0).detach().cpu().numpy(),
        probability=probability,
        convergence_delta=float("nan"),  # NoiseTunnel does not return a delta
        target_class=target_class,
    )


def similarity(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Compare two attribution maps.

    Rank correlation asks whether the same samples are ranked important; cosine
    similarity on absolute values asks whether the magnitude pattern agrees while
    ignoring sign.
    """
    a, b = first.ravel(), second.ravel()
    rank, _ = spearmanr(a, b)
    abs_a, abs_b = np.abs(a), np.abs(b)
    cosine = float(abs_a @ abs_b / (np.linalg.norm(abs_a) * np.linalg.norm(abs_b) + 1e-12))
    return {"spearman": float(rank), "abs_cosine": cosine}


def model_randomization_test(model: nn.Module, signal: np.ndarray, target_class: int,
                             device: torch.device | None = None, repeats: int = 3,
                             seed: int = 0) -> dict[str, float]:
    """Compare attribution from the trained model against randomised copies.

    If a map barely changes when the weights are destroyed, it is not explaining
    the model. Low similarity is the passing result.
    """
    device = device or next(model.parameters()).device
    trained = attribute(model, signal, target_class, device)

    scores: list[dict[str, float]] = []
    for offset in range(repeats):
        randomized = copy.deepcopy(model)
        generator = torch.Generator(device="cpu").manual_seed(seed + offset)
        for parameter in randomized.parameters():
            with torch.no_grad():
                replacement = torch.empty(parameter.shape, device="cpu")
                if parameter.dim() > 1:
                    nn.init.kaiming_uniform_(replacement, a=5 ** 0.5, generator=generator)
                else:
                    replacement.uniform_(-0.1, 0.1, generator=generator)
                parameter.copy_(replacement.to(parameter.device))
        randomized.eval()
        scores.append(similarity(trained.values, attribute(randomized, signal, target_class, device).values))

    return {
        "trained_probability": trained.probability,
        "convergence_delta": trained.convergence_delta,
        "spearman_vs_random_mean": float(np.mean([s["spearman"] for s in scores])),
        "spearman_vs_random_max": float(np.max([s["spearman"] for s in scores])),
        "abs_cosine_vs_random_mean": float(np.mean([s["abs_cosine"] for s in scores])),
        "repeats": repeats,
    }
