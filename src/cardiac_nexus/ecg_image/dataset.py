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

from .digitize import STRIP_HEIGHT, STRIP_SECONDS, STRIP_WIDTH
from .distort import DistortionConfig, distort, transform_points
from .layout import crop_box
from .render import MM_PER_MV, MM_PER_SECOND, PaperSpec, layout_columns, render, rhythm_leads


def finish_strip(crop: np.ndarray, points: np.ndarray, rng: np.random.Generator, distortions: DistortionConfig,
                 strip_width: int = STRIP_WIDTH, strip_height: int = STRIP_HEIGHT):
    """Degrade one cropped strip and carry its true trace through: image [3, H, W], rows [W], valid [W].

    `points` are the trace's (x, y) positions within the crop.
    """
    source_height, source_width = crop.shape[:2]
    crop, homography = distort(crop, rng, distortions)

    # The distortion moves the trace, so the labels move with it.
    # Skipping this leaves ground truth describing where it used to be.
    points = transform_points(points, homography)

    crop = cv2.resize(crop, (strip_width, strip_height), interpolation=cv2.INTER_AREA)
    points[:, 0] *= strip_width / source_width
    points[:, 1] *= strip_height / source_height

    # Warping leaves x uneven, so resample onto the strip's column grid.
    # np.interp needs x ascending, which a rotation can violate at the edges.
    order = np.argsort(points[:, 0])
    rows = np.interp(np.arange(strip_width), points[order, 0], points[order, 1])

    # A deflection large enough to leave the crop cannot be located; mark
    # those columns so the loss ignores them rather than learning nonsense.
    valid = (rows >= 0) & (rows <= strip_height - 1)
    rows = np.clip(rows, 0, strip_height - 1)
    return crop.astype(np.float32).transpose(2, 0, 1) / 255.0, rows.astype(np.float32), valid


class StripDataset(Dataset):
    """Lead strips rendered from digital signals, with row-position ground truth.

    Returns, per record, all lead windows: images [N, 3, H, W], rows [N, W], valid [N, W].
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

        # Each lead is cut into consecutive windows rather than squashed whole into
        # one strip. At 4 px/mm and 25 mm/s a second is 100 px, so a window is taken
        # at native scale and one pixel is one sample.
        window_samples = int(STRIP_SECONDS * spec.sampling_rate)
        windows_per_lead = signal.shape[1] // window_samples

        images, all_rows, all_valid = [], [], []
        for lead in range(12):
            trace = page.traces[lead]
            top = int(max(0, trace["baseline_y"] - row_height / 2))
            bottom = int(min(image.shape[0], trace["baseline_y"] + row_height / 2))
            pixels_per_sample = (trace["x1"] - trace["x0"]) / signal.shape[1]

            for window in range(windows_per_lead):
                first = window * window_samples
                last = first + window_samples
                left = int(trace["x0"] + first * pixels_per_sample)
                right = int(trace["x0"] + last * pixels_per_sample)
                crop = image[top:bottom, left:right]
                if crop.size == 0:
                    continue

                # True trace position for this window, as (x, y) within the crop.
                samples = signal[lead, first:last]
                xs = np.linspace(0, right - left - 1, len(samples))
                ys = (trace["baseline_y"] - samples * amplitude * spec.mm(MM_PER_MV)) - top
                points = np.stack([xs, ys], axis=1).astype(np.float32)

                image_strip, rows, valid = finish_strip(crop, points, rng, self.distortions,
                                                        self.strip_width, self.strip_height)
                images.append(image_strip)
                all_rows.append(rows)
                all_valid.append(valid)

        return (
            torch.from_numpy(np.stack(images)),      # [12, 3, H, W]
            torch.from_numpy(np.stack(all_rows)),    # [12, W]
            torch.from_numpy(np.stack(all_valid)),   # [12, W]
        )


# Layouts in rough proportion to how often they are printed; PM-ECG-ID's real images
# are 3x4 variants and 6x2 for nine in ten.
LAYOUT_WEIGHTS = {"12x1": 1.0, "6x2": 2.0, "6x2+1R": 1.0, "3x4": 2.0, "3x4+1R": 2.0, "3x4+3R": 1.0}

# (paper, minor lines, major lines, trace), RGB: pink, red, magenta, and the grey a
# photocopy or a phone's white balance can turn any of them into.
PAPER_STYLES = [
    ((255, 250, 248), (247, 196, 190), (232, 138, 128), (24, 24, 28)),
    ((255, 255, 255), (250, 200, 200), (228, 118, 118), (10, 10, 10)),
    ((252, 249, 252), (236, 206, 236), (206, 140, 206), (30, 30, 40)),
    ((248, 248, 246), (214, 214, 212), (166, 166, 164), (20, 20, 20)),
]


class LayoutStripDataset(Dataset):
    """Strips from pages in every standard layout, cut the way a real page is cut.

    `layout.lead_segments` cuts a located page into boxes one row pitch tall and one
    lead segment wide, and each 2.5 s of a segment is one strip for the localizer.
    These strips are cut the same way from rendered pages, so the localizer learns
    what it will actually be given: taller boxes than the 12x1 training strips, with
    the spikes of neighbouring rows, lead labels, pulses and column ticks inside. The
    boxes are displaced as the strip-finder's own estimates are (about 1 mm vertically
    on synthetic pages, under 1 mm horizontally), and each strip is then degraded as
    StripDataset's are.

    Returns, per record, all its strips: images [N, 3, H, W], rows [N, W], valid [N, W];
    N depends on the layout drawn, so batch with `concatenate_records`.
    """

    def __init__(self, signals: np.ndarray, indices: np.ndarray, seed: int = 0,
                 distortions: DistortionConfig | None = None,
                 amplitude_range: tuple[float, float] = (0.2, 0.5),
                 strip_height: int = STRIP_HEIGHT, strip_width: int = STRIP_WIDTH,
                 deterministic: bool = False):
        self.signals = signals
        self.indices = np.asarray(indices)
        self.seed = seed
        self.distortions = distortions or DistortionConfig()
        self.amplitude_range = amplitude_range
        self.strip_height = strip_height
        self.strip_width = strip_width
        self.deterministic = deterministic
        self.layouts = list(LAYOUT_WEIGHTS)
        weights = np.array(list(LAYOUT_WEIGHTS.values()))
        self.layout_p = weights / weights.sum()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        rng = np.random.default_rng(self.seed + item if self.deterministic else None)
        signal = np.asarray(self.signals[self.indices[item]], dtype=np.float32)
        amplitude = float(rng.uniform(*self.amplitude_range))
        layout = str(rng.choice(self.layouts, p=self.layout_p))
        rows_on_page = len(layout_columns(layout)[0]) + len(rhythm_leads(layout))
        paper, minor, major, ink = PAPER_STYLES[int(rng.integers(len(PAPER_STYLES)))]
        spec = PaperSpec(layout=layout, pixels_per_mm=float(rng.uniform(2.8, 6.0)),
                         row_height_mm=float(np.clip(rng.uniform(150, 190) / rows_on_page, 14, 42)),
                         margin_mm=float(rng.uniform(7, 14)), paper_colour=paper, minor_colour=minor,
                         major_colour=major, trace_colour=ink, trace_width=int(rng.integers(1, 4)),
                         show_labels=bool(rng.random() < 0.8), pulse_every_row=True,
                         column_ticks=bool(rng.random() < 0.7))
        page = render(signal, spec, amplitude_mv_per_unit=amplitude)
        image = cv2.cvtColor(page.array, cv2.COLOR_RGB2BGR)
        pitch = spec.mm(spec.row_height_mm)
        window_samples = int(STRIP_SECONDS * spec.sampling_rate)
        window_width = spec.mm(STRIP_SECONDS * MM_PER_SECOND)

        images, all_rows, all_valid = [], [], []
        for trace in page.traces:
            # The strip-finder's estimate of this row, not the truth.
            centre = trace["baseline_y"] + spec.mm(float(np.clip(rng.normal(0, 0.8), -2.5, 2.5)))
            half = pitch * float(rng.uniform(0.92, 1.08)) / 2
            shift = spec.mm(float(np.clip(rng.normal(0, 0.3), -1.0, 1.0)))
            samples_here = trace["sample_end"] - trace["sample_start"]
            pixels_per_sample = (trace["x1"] - trace["x0"]) / max(samples_here - 1, 1)  # as render() spaces them
            for window in range(samples_here // window_samples):
                first = window * window_samples
                left = trace["x0"] + window * window_width + shift
                box = (left, centre - half, left + window_width, centre + half)
                crop = crop_box(image, box)
                if crop.size == 0:
                    continue
                values = signal[trace["lead"], trace["sample_start"] + first: trace["sample_start"] + first + window_samples]
                xs = trace["x0"] + (first + np.arange(len(values))) * pixels_per_sample - round(box[0])
                ys = trace["baseline_y"] - values * amplitude * spec.mm(MM_PER_MV) - round(box[1])
                points = np.stack([xs, ys], axis=1).astype(np.float32)
                strip, rows, valid = finish_strip(crop, points, rng, self.distortions,
                                                  self.strip_width, self.strip_height)
                images.append(strip)
                all_rows.append(rows)
                all_valid.append(valid)

        return (torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(all_rows)),
                torch.from_numpy(np.stack(all_valid)))


def concatenate_records(batch):
    """Collate records with different strip counts into one [1, N, ...] batch."""
    return tuple(torch.cat(parts).unsqueeze(0) for parts in zip(*batch))
