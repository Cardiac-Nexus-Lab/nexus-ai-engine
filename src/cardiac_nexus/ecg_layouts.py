"""What a printed ECG layout shows of a ten-second, twelve-lead recording.

A 3x4 page prints each lead for 2.5 of its 10 seconds (I, II, III for 0-2.5 s,
aVR, aVL, aVF for 2.5-5 s, and so on) plus any full-length rhythm strips; a 6x2
page prints 5 seconds of each. Digitizing such a page can only ever return those
samples, so the classifier sees the rest as fill. These functions produce exactly
that input from a complete recording, for measuring the loss
(scripts/evaluate_ecg_layouts.py) and training a classifier that expects it
(scripts/train_ecg_layouts.py). They match ecg_image.pipeline.digitize_layout:
each lead standardized on its printed part, zeros elsewhere.
"""

from __future__ import annotations

import numpy as np

from .ecg_image.render import LAYOUTS, layout_columns, rhythm_leads

__all__ = ["LAYOUTS", "printed_mask", "reduce_to_layout"]


def printed_mask(layout: str, samples: int, leads: int = 12) -> np.ndarray:
    """[leads, samples] booleans: which samples a page in this layout prints."""
    mask = np.zeros((leads, samples), bool)
    columns = layout_columns(layout)
    width = samples // len(columns)
    for c, column in enumerate(columns):
        for lead in column:
            mask[lead, c * width:(c + 1) * width] = True
    for lead in rhythm_leads(layout):
        mask[lead] = True
    return mask


def reduce_to_layout(signals: np.ndarray, layout: str, fill: str = "zeros") -> np.ndarray:
    """Signals [..., leads, samples] reduced to what `layout` prints.

    fill "zeros" leaves unprinted time at 0 (the mean of a standardized lead);
    "repeat" repeats the printed window to fill it.
    """
    signals = np.asarray(signals, dtype=np.float32)
    mask = printed_mask(layout, signals.shape[-1], signals.shape[-2])
    out = np.zeros_like(signals)
    samples = signals.shape[-1]
    for lead in range(signals.shape[-2]):
        part = signals[..., lead, mask[lead]]
        part = (part - part.mean(axis=-1, keepdims=True)) / (part.std(axis=-1, keepdims=True) + 1e-6)
        if fill == "repeat":
            reps = -(-samples // part.shape[-1])
            out[..., lead, :] = np.tile(part, reps)[..., :samples]
        else:
            out[..., lead, mask[lead]] = part
    return out
