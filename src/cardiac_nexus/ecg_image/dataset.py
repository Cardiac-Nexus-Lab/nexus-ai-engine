"""Generate training pairs for the digitizer: strip image in, true trace rows out.

Pairs are produced on demand rather than written to disk. Rendering a strip costs
far less than a training step, and generating fresh each time means a recording is
never seen twice with the same distortion, which is the point of synthesising the
data at all.

Each sample is one lead's strip, cropped from a rendered page and then degraded.
Working per strip rather than per page keeps the images small enough to train on a
laptop, and matches how the inference pipeline consumes a page: locate the strips,
then digitize each.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .digitize import STRIP_HEIGHT, STRIP_WIDTH
from .distort import DistortionConfig, distort, transform_points
from .render import MM_PER_MV, PaperSpec, render


class StripDataset(Dataset):
    """Lead strips rendered from digital signals, with row-position ground truth.

    Returns (image [3, H, W] in [0,1], rows [W] in pixels, valid [W] mask).
    """

    def __init__(
        self,
        signals: np.ndarray,
        indices: np.ndarray,
        seed: int = 0,
        distortions: DistortionConfig | None = None,
        amplitude_range: tuple[float, float] = (0.2, 0.45),
        strip_height: int = STRIP_HEIGHT,
        strip_width: int = STRIP_WIDTH,
        deterministic: bool = False,
    ):
        self.signals = signals
        self.indices = np.asarray(indices)
        self.seed = seed
        self.distortions = distortions or DistortionConfig()
        self.amplitude_range = amplitude_range
        self.strip_height = strip_height
        self.strip_width = strip_width
        # Validation and test need a fixed appearance per item; training does not.
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.indices)

    def _rng(self, item: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + item if self.deterministic else None)

    def __getitem__(self, item: int):
        """One record yields all twelve of its strips.

        Rendering dominates the cost of building a sample, so drawing the page once
        and cropping every lead from it does the work once instead of twelve times.
        Distortion is still sampled per strip, so the twelve are not identically
        degraded.
        """
        record = self.indices[item]
        rng = self._rng(item)

        signal = np.asarray(self.signals[record], dtype=np.float32)
        amplitude = float(rng.uniform(*self.amplitude_range))

        # One lead per row: each strip then holds a single trace with a known
        # baseline, which is what the localizer is trained to read.
        spec = PaperSpec(layout="12x1", pixels_per_mm=4.0, row_height_mm=16.0, margin_mm=4.0,
                         show_labels=bool(rng.random() < 0.5))
        page = render(signal, spec, amplitude_mv_per_unit=amplitude)
        image = cv2.cvtColor(page.array, cv2.COLOR_RGB2BGR)
        row_height = spec.mm(spec.row_height_mm)

        images, all_rows, all_valid = [], [], []
        for lead in range(12):
            trace = page.traces[lead]
            top = int(max(0, trace["baseline_y"] - row_height / 2))
            bottom = int(min(image.shape[0], trace["baseline_y"] + row_height / 2))
            left, right = int(trace["x0"]), int(trace["x1"])
            crop = image[top:bottom, left:right]

            # True trace position for every sample, as (x, y) within the crop.
            samples = signal[lead]
            xs = np.linspace(0, right - left - 1, len(samples))
            ys = (trace["baseline_y"] - samples * amplitude * spec.mm(MM_PER_MV)) - top
            points = np.stack([xs, ys], axis=1).astype(np.float32)

            source_height, source_width = crop.shape[:2]
            crop, homography = distort(crop, rng, self.distortions)

            # The distortion moves the trace, so the labels move with it. Skipping
            # this leaves ground truth describing where the trace used to be.
            points = transform_points(points, homography)

            crop = cv2.resize(crop, (self.strip_width, self.strip_height), interpolation=cv2.INTER_AREA)
            points[:, 0] *= self.strip_width / source_width
            points[:, 1] *= self.strip_height / source_height

            # Warping leaves the x positions uneven, so resample onto the strip's
            # regular column grid. np.interp needs x ascending, which a rotation
            # can violate near the edges.
            order = np.argsort(points[:, 0])
            rows = np.interp(np.arange(self.strip_width), points[order, 0], points[order, 1])

            # A deflection large enough to leave the crop cannot be located; mark
            # those columns so the loss ignores them rather than learning nonsense.
            valid = (rows >= 0) & (rows <= self.strip_height - 1)
            rows = np.clip(rows, 0, self.strip_height - 1)

            images.append(crop.astype(np.float32).transpose(2, 0, 1) / 255.0)
            all_rows.append(rows.astype(np.float32))
            all_valid.append(valid)

        return (
            torch.from_numpy(np.stack(images)),      # [12, 3, H, W]
            torch.from_numpy(np.stack(all_rows)),    # [12, W]
            torch.from_numpy(np.stack(all_valid)),   # [12, W]
        )
