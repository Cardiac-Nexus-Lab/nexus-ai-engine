"""Render a digital 12-lead ECG as a paper printout.

Digitizing a photographed ECG needs image/signal pairs to learn from, and
labelling real photographs by hand does not scale. Rendering the signals already
held instead gives exact ground truth for free: the signal that produced an image
is known precisely, so an unlimited training set can be generated.

Paper conventions followed here are the clinical standard: 25 mm/s horizontally
and 10 mm/mV vertically, on a 1 mm grid with every fifth line emphasised. One
small square is therefore 0.04 s by 0.1 mV, and one large square 0.2 s by 0.5 mV.

Layout matters more than it first appears. A routine printout is 3x4: each lead
occupies a 2.5 second column, with one continuous rhythm strip below. Only 2.5
seconds of most leads is physically on the page, so no digitizer can recover ten
seconds of twelve leads from it -- the information was never printed. The 12x1
layout, which some machines produce, does carry the full ten seconds of every
lead and is what the existing classifier expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Clinical paper standard.
MM_PER_SECOND = 25.0
MM_PER_MV = 10.0


@dataclass
class PaperSpec:
    """Physical description of the printout."""

    layout: str = "12x1"          # one of LAYOUTS ("3x4+rhythm" is kept as an alias of "3x4+1R")
    pixels_per_mm: float = 4.0    # ~100 dpi
    sampling_rate: int = 100
    grid_minor_mm: float = 1.0
    grid_major_mm: float = 5.0
    row_height_mm: float = 20.0   # vertical space per lead trace; tight spacing
                                  # makes neighbouring traces overlap, which real
                                  # printouts avoid and a digitizer would struggle with
    margin_mm: float = 8.0
    paper_colour: tuple[int, int, int] = (255, 250, 248)
    minor_colour: tuple[int, int, int] = (247, 196, 190)
    major_colour: tuple[int, int, int] = (232, 138, 128)
    trace_colour: tuple[int, int, int] = (24, 24, 28)
    trace_width: int = 2
    show_labels: bool = True
    show_calibration: bool = True
    # Real machines print the pulse at the start of every row, ending where the trace
    # begins, and a short tick where one column hands over to the next. The original
    # 12x1 training pages had one pulse in the margin and no ticks; kept as the default.
    pulse_every_row: bool = False
    column_ticks: bool = False

    def mm(self, millimetres: float) -> float:
        return millimetres * self.pixels_per_mm


@dataclass
class RenderedECG:
    """An image plus the geometry needed to recover the signal from it."""

    image: Image.Image
    spec: PaperSpec
    # For each drawn trace: lead index, sample range, and the pixel box it occupies.
    traces: list[dict] = field(default_factory=list)

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.image)


def _draw_grid(draw: ImageDraw.ImageDraw, width: int, height: int, spec: PaperSpec) -> None:
    minor = spec.mm(spec.grid_minor_mm)
    major = spec.mm(spec.grid_major_mm)

    # Minor lines first so major lines draw over them.
    x = 0.0
    while x < width:
        draw.line([(x, 0), (x, height)], fill=spec.minor_colour, width=1)
        x += minor
    y = 0.0
    while y < height:
        draw.line([(0, y), (width, y)], fill=spec.minor_colour, width=1)
        y += minor

    x = 0.0
    while x < width:
        draw.line([(x, 0), (x, height)], fill=spec.major_colour, width=1)
        x += major
    y = 0.0
    while y < height:
        draw.line([(0, y), (width, y)], fill=spec.major_colour, width=1)
        y += major


LAYOUTS = ("12x1", "6x2", "6x2+1R", "3x4", "3x4+1R", "3x4+3R")
# Rhythm strips run the full ten seconds under the grid: lead II alone, or V1, II and V5.
RHYTHM_LEADS = {"1R": [1], "3R": [6, 1, 10], "rhythm": [1]}


def layout_columns(layout: str) -> list[list[int]]:
    """The leads in each column of the grid part of a layout, top to bottom, in standard machine order."""
    grid = layout.split("+")[0]
    if grid == "12x1":
        return [list(range(12))]
    if grid == "6x2":
        return [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    if grid == "3x4":
        return [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


def rhythm_leads(layout: str) -> list[int]:
    return RHYTHM_LEADS[layout.split("+")[1]] if "+" in layout else []


def _layout_rows(layout: str, total_samples: int, sampling_rate: int) -> list[list[tuple[int, int, int]]]:
    """Rows of (lead index, start sample, end sample) describing what to draw where.

    Each column of the grid shows its leads for its own slice of the recording (a 3x4
    page shows I, II, III for 0-2.5 s, aVR, aVL, aVF for 2.5-5 s, and so on), as
    printing machines do; rhythm strips below show one lead for the whole duration.
    """
    columns = layout_columns(layout)
    width = total_samples // len(columns)
    rows: list[list[tuple[int, int, int]]] = []
    for row_index in range(len(columns[0])):
        rows.append([(column[row_index], c * width, (c + 1) * width) for c, column in enumerate(columns)])
    rows += [[(lead, 0, total_samples)] for lead in rhythm_leads(layout)]
    return rows


def render(signal: np.ndarray, spec: PaperSpec | None = None,
           amplitude_mv_per_unit: float = 1.0) -> RenderedECG:
    """Draw a [12, samples] signal onto simulated ECG paper.

    The signal is treated as millivolts scaled by amplitude_mv_per_unit. Standardized
    signals are unitless, so the scale factor sets how tall the traces appear; it is
    recorded in the returned geometry so the mapping stays invertible.
    """
    spec = spec or PaperSpec()
    if signal.ndim != 2 or signal.shape[0] != 12:
        raise ValueError(f"expected a [12, samples] signal, got {signal.shape}")

    total_samples = signal.shape[1]
    rows = _layout_rows(spec.layout, total_samples, spec.sampling_rate)

    # A row's width is the total of the segments laid side by side within it, not
    # the longest single segment: a 3x4 row holds four 2.5 s columns and so spans
    # the same ten seconds as one full-width row.
    widest_row = max(sum(end - start for _, start, end in row) for row in rows)
    seconds = widest_row / spec.sampling_rate
    plot_width = spec.mm(seconds * MM_PER_SECOND)
    row_height = spec.mm(spec.row_height_mm)
    margin = spec.mm(spec.margin_mm)

    width = int(plot_width + 2 * margin)
    height = int(len(rows) * row_height + 2 * margin)

    image = Image.new("RGB", (width, height), spec.paper_colour)
    draw = ImageDraw.Draw(image)
    _draw_grid(draw, width, height, spec)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(spec.mm(3)))
    except OSError:
        font = ImageFont.load_default()

    traces: list[dict] = []
    for row_index, row in enumerate(rows):
        baseline_y = margin + row_index * row_height + row_height / 2
        # Columns are laid out left to right; track the running offset rather than
        # searching the row, which would break on a repeated segment.
        column_offset = 0.0
        for lead, start, end in row:
            segment = signal[lead, start:end]
            samples = len(segment)
            seconds_here = samples / spec.sampling_rate
            segment_width = spec.mm(seconds_here * MM_PER_SECOND)
            x0 = margin + column_offset
            column_offset += segment_width

            xs = x0 + np.linspace(0, segment_width, samples)
            ys = baseline_y - segment * amplitude_mv_per_unit * spec.mm(MM_PER_MV)

            draw.line(list(zip(xs.tolist(), ys.tolist())), fill=spec.trace_colour,
                      width=spec.trace_width, joint="curve")

            if spec.show_labels:
                # Sit the label just above the baseline rather than at the row's
                # top edge, where the first row would clip it off the page.
                draw.text((x0 + spec.mm(1.5), baseline_y - spec.mm(9)),
                          LEAD_NAMES[lead], fill=spec.trace_colour, font=font)

            traces.append(
                {
                    "lead": lead,
                    "lead_name": LEAD_NAMES[lead],
                    "sample_start": int(start),
                    "sample_end": int(end),
                    "x0": float(x0),
                    "x1": float(x0 + segment_width),
                    "baseline_y": float(baseline_y),
                    "pixels_per_mv": float(spec.mm(MM_PER_MV) * amplitude_mv_per_unit),
                    "row": row_index,
                }
            )

    if spec.column_ticks:
        for trace in traces:
            if trace["x0"] > margin + 1:
                draw.line([(trace["x0"], trace["baseline_y"] - spec.mm(4)), (trace["x0"], trace["baseline_y"] + spec.mm(4))],
                          fill=spec.trace_colour, width=spec.trace_width)

    if spec.show_calibration:
        # The 1 mV, 200 ms calibration pulse that machines print so a reader can
        # verify the scale. A digitizer can use it the same way.
        pulse_height = spec.mm(MM_PER_MV)
        if spec.pulse_every_row:
            placements = [(margin - spec.mm(5.5), margin + r * row_height + row_height / 2) for r in range(len(rows))]
        else:
            placements = [(margin / 2, margin + row_height / 2)]
        for pulse_x, base in placements:
            draw.line(
                [
                    (pulse_x, base),
                    (pulse_x, base - pulse_height),
                    (pulse_x + spec.mm(5), base - pulse_height),
                    (pulse_x + spec.mm(5), base),
                ],
                fill=spec.trace_colour,
                width=spec.trace_width,
            )

    return RenderedECG(image=image, spec=spec, traces=traces)
