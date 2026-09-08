"""Training-time augmentation for 12-lead ECG.

Both scaling experiments so far overfit: the baseline CNN plateaued by epoch 14
and the larger xresnet1d18 peaked at epoch 6 and then declined for thirteen
consecutive epochs. With 17,084 training recordings against millions of
parameters, the limit is the data rather than the network, so the useful lever is
making each epoch see a different version of each recording.

Every transform below preserves diagnostic meaning. That constraint rules out
several augmentations that are standard elsewhere: reversing time, permuting
leads, or flipping amplitude would each turn a recording into something a
cardiologist would read differently, teaching the model invariances that are
clinically wrong.
"""

from __future__ import annotations

import numpy as np
import torch


class ECGAugment:
    """Randomly perturb a recording in ways a real recording could plausibly vary.

    Applied per sample on the training split only. Each transform fires
    independently with its own probability, so a recording usually receives a
    combination rather than exactly one.
    """

    def __init__(
        self,
        time_shift: float = 0.5,
        max_shift_fraction: float = 0.1,
        amplitude_scale: float = 0.5,
        scale_range: tuple[float, float] = (0.8, 1.2),
        gaussian_noise: float = 0.3,
        noise_std: float = 0.05,
        baseline_wander: float = 0.3,
        wander_amplitude: float = 0.15,
        lead_dropout: float = 0.2,
        max_leads_dropped: int = 2,
        generator: np.random.Generator | None = None,
    ):
        self.time_shift = time_shift
        self.max_shift_fraction = max_shift_fraction
        self.amplitude_scale = amplitude_scale
        self.scale_range = scale_range
        self.gaussian_noise = gaussian_noise
        self.noise_std = noise_std
        self.baseline_wander = baseline_wander
        self.wander_amplitude = wander_amplitude
        self.lead_dropout = lead_dropout
        self.max_leads_dropped = max_leads_dropped
        self.rng = generator or np.random.default_rng()

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        x = signal.clone()
        leads, length = x.shape

        # Where the 10-second window happens to start is arbitrary, so a circular
        # shift produces an equally valid recording of the same patient.
        if self.rng.random() < self.time_shift:
            shift = int(self.rng.integers(-int(length * self.max_shift_fraction),
                                          int(length * self.max_shift_fraction) + 1))
            if shift:
                x = torch.roll(x, shifts=shift, dims=1)

        # Overall gain varies between machines and electrode placement. Scaling all
        # leads together keeps the relationships between leads intact, which is
        # where much of the diagnostic information lives.
        if self.rng.random() < self.amplitude_scale:
            x = x * float(self.rng.uniform(*self.scale_range))

        # Sensor and mains noise.
        if self.rng.random() < self.gaussian_noise:
            x = x + torch.from_numpy(
                self.rng.normal(0.0, self.noise_std, size=(leads, length)).astype(np.float32)
            )

        # Slow baseline drift from respiration and electrode movement: a very
        # low-frequency sinusoid, independent per lead.
        if self.rng.random() < self.baseline_wander:
            time = np.linspace(0, 2 * np.pi, length, dtype=np.float32)
            frequencies = self.rng.uniform(0.15, 0.6, size=(leads, 1)).astype(np.float32)
            phases = self.rng.uniform(0, 2 * np.pi, size=(leads, 1)).astype(np.float32)
            wander = self.wander_amplitude * np.sin(frequencies * time + phases)
            x = x + torch.from_numpy(wander.astype(np.float32))

        # Electrodes do detach. Zeroing a lead teaches the model not to depend on
        # any single one, which also makes it more robust to a bad recording.
        if self.rng.random() < self.lead_dropout:
            count = int(self.rng.integers(1, self.max_leads_dropped + 1))
            for lead in self.rng.choice(leads, size=count, replace=False):
                x[lead] = 0.0

        return x
