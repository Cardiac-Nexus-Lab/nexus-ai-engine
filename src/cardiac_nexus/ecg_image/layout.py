"""Find the lead strips on a real ECG printout, for any of the standard layouts.

`pipeline.extract_strips` cuts a page by the geometry the synthetic renderer uses,
which a real printout does not follow: it has margins, a calibration pulse and a
label at the start of each row, separator ticks, captions and QR codes. This
module reads the page instead:

1. ink and grid are separated by colour: traces are dark and grey, the grid is
   pink, red or magenta;
2. the grid's period gives the paper scale in pixels per millimetre, and standard
   paper runs at 25 mm/s, so ten seconds of trace is 250 mm wide;
3. trace rows are where dark ink spans most of the page width (captions, labels and
   QR codes are narrow);
4. each row's trace starts where its calibration pulse ends.

Every step reports what it found, and the caller gets a reason instead of strips
when a step fails, so a page that cannot be read is refused rather than read wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .render import MM_PER_SECOND, layout_columns, rhythm_leads

SECONDS = 10.0


@dataclass
class PageGeometry:
    layout: str
    pixels_per_mm: float
    row_centres: list[float]
    row_pitch: float
    x_start: float
    x_end: float
    notes: list[str] = field(default_factory=list)


def ink_mask(page: np.ndarray, pixels_per_mm: float | None = None) -> np.ndarray:
    """Dark, unsaturated pixels: the traces, labels and pulses, but not the grid.

    Traces are often one pixel wide, so the mask is grown by hysteresis rather than
    cleaned by morphology: fainter pixels are kept only where they connect to clearly
    dark ones. Phone photos can turn a pink grid grey, so with the paper scale known,
    long straight runs (grid lines; a trace never runs straight for 15 mm) are removed.
    """
    hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
    value, saturation = hsv[..., 2].astype(np.float32), hsv[..., 1].astype(np.float32)
    background = cv2.medianBlur(hsv[..., 2], 31).astype(np.float32)  # evens out shadows and lighting
    grey = saturation < 110
    weak = ((value < 0.72 * background) & grey).astype(np.uint8)
    core = (value < 0.55 * background) & grey
    _, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    keep = np.zeros(len(stats), bool)
    keep[np.unique(labels[core])] = True
    keep &= stats[:, cv2.CC_STAT_AREA] >= 6
    keep[0] = False
    mask = keep[labels].astype(np.uint8)
    if pixels_per_mm:
        run = max(5, int(15 * pixels_per_mm))
        lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, run), np.uint8)) | \
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((run, 1), np.uint8))
        mask &= 1 - lines
    return mask


def grid_mask(page: np.ndarray) -> np.ndarray:
    """Coloured grid lines (pink, red, magenta) that are lighter than the ink."""
    hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[..., 0], hsv[..., 1]
    reddish = (hue < 15) | (hue > 135)
    return (reddish & (saturation > 25)).astype(np.uint8)


def grid_pitch(page: np.ndarray) -> float | None:
    """Pixels per millimetre from the period of the grid's vertical lines.

    Every multiple of the grid's period echoes in the autocorrelation, and lines
    drawn or photographed at a fractional pitch land on irregular whole pixels, so
    picking one peak is fragile. Instead each candidate scale p is scored as a comb:
    the mean correlation at every 5 mm multiple (5p, 10p, ...) minus the mean halfway
    between them. Only the true scale gains at the major lines and loses in between;
    a wrong scale that happens to land on 1 mm lines gains and loses alike.

    Candidates are limited by the page: ten seconds of trace is 250 mm, so a page W
    pixels wide is between about W/500 and W/240 px/mm. Within that range, twice or
    half the true scale cannot occur, which removes the remaining ambiguity.
    Returns None when no grid is visible.
    """
    width = page.shape[1]
    # Weighted by saturation: major lines are printed stronger than minor ones, and that
    # emphasis is what separates the true scale from 0.6 of it, whose comb lands on
    # 3 mm multiples (minor lines) with troughs between, just as the true comb does.
    saturation = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float64)
    profile = (grid_mask(page) * saturation).mean(axis=0)
    if profile.std() < 1e-3:
        return None
    profile = profile - profile.mean()
    correlation = np.correlate(profile, profile, mode="full")[len(profile) - 1:]
    correlation /= correlation[0] + 1e-9
    lags = np.arange(len(correlation))
    limit = width / 3

    def comb(step: float) -> float:
        k = np.arange(1, max(2, int(limit / step)) + 1)
        return float((np.interp(step * k, lags, correlation) - np.interp(step * (k - 0.5), lags, correlation)).mean())

    # The 1 mm comb settles what the 5 mm comb leaves close: when minor lines are visible
    # (the only case where 0.6 of the scale competes), only the true scale lands on them.
    candidates = np.arange(width / 500, width / 240, 0.002)
    scores = [comb(5 * p) + comb(p) for p in candidates]
    best = int(np.argmax(scores))
    if scores[best] < 0.05:
        return None
    return float(candidates[best])


def find_rows(ink: np.ndarray, count: int) -> tuple[list[float], float] | None:
    """Centres of the `count` trace rows: bands where ink spans most of the width."""
    height, width = ink.shape
    bins = 64
    coverage = np.zeros(height)
    for b in range(bins):
        part = ink[:, b * width // bins:(b + 1) * width // bins]
        coverage += part.max(axis=1)
    coverage /= bins
    # Machines print rows evenly spaced, so fit an evenly spaced comb of `count` rows to
    # the coverage profile rather than picking peaks one by one, which grabs the QRS
    # spikes of a tall trace on a dense page. Each row may then move to its own peak
    # nearby, since a rhythm strip often sits a little further below the grid.
    smooth = cv2.GaussianBlur(coverage.reshape(-1, 1), (1, 0), sigmaX=0.1,
                              sigmaY=max(1.0, height / (count + 1) / 12)).ravel()
    best, fit = -1.0, None
    for pitch in np.arange(height / (count + 3), height / max(count - 0.5, 1), 0.5):
        span = (count - 1) * pitch
        for top in range(0, int(height - span)):
            score = smooth[(top + pitch * np.arange(count)).astype(int)].sum()
            if score > best:
                best, fit = score, (top, pitch)
    if fit is None:
        return None
    top, pitch = fit
    peaks = []
    for k in range(count):
        guess = top + k * pitch
        a, b = int(max(0, guess - 0.35 * pitch)), int(min(height, guess + 0.35 * pitch + 1))
        peaks.append(a + int(np.argmax(smooth[a:b])))
    if min(smooth[p] for p in peaks) < max(0.1, 0.25 * smooth.max()):
        return None  # a 'row' with almost no ink across the page: wrong layout or a row is missing
    centres = []
    for y in peaks:
        # The median height of the band's ink is the baseline: flat stretches between
        # beats hold most of a trace's pixels, while QRS spikes pull a mean off it.
        top, bottom = int(max(0, y - 0.4 * pitch)), int(min(height, y + 0.4 * pitch))
        ys = np.nonzero(ink[top:bottom])[0]
        centres.append(float(top + np.median(ys)) if len(ys) else float(y))
    return centres, pitch


def trace_start(ink: np.ndarray, centres: list[float], pitch: float, pixels_per_mm: float) -> float | None:
    """Where traces begin: just after the calibration pulse at the left of the rows.

    The pulse is the leftmost tall ink feature of a row (1 mV = 10 mm high); its right
    edge is the start of the trace in every standard layout.
    """
    starts = []
    for centre in centres:
        top, bottom = int(max(0, centre - pitch / 2)), int(min(ink.shape[0], centre + pitch / 2))
        column_height = ink[top:bottom].sum(axis=0)
        tall = np.flatnonzero(column_height > 6 * pixels_per_mm)  # a vertical edge at least 6 mm long
        if len(tall) == 0:
            continue
        # Neighbouring tall columns are one edge (a thick printed line spans several).
        edges = [[tall[0], tall[0]]]
        for x in tall[1:]:
            if x - edges[-1][1] <= max(2, pixels_per_mm / 2):
                edges[-1][1] = x
            else:
                edges.append([x, x])
        # The pulse is 5 mm (0.2 s) wide: two edges that far apart, joined by a flat top.
        # Grid fragments and the page border also make tall columns, but not that shape.
        band = ink[top:bottom]
        for left, right in ((a, b) for a in edges for b in edges if 4 <= (b[0] - a[1]) / pixels_per_mm <= 6):
            span = band[:, left[1]:right[0] + 1]
            if span.shape[1] and span.mean(axis=1).max() > 0.7:
                starts.append(float(right[1] + 1))
                break
    return float(np.median(starts)) if starts else None


def locate(page: np.ndarray, layout: str) -> tuple[PageGeometry | None, str]:
    """Geometry of a dewarped page in the given layout, or None and the reason it failed."""
    columns = layout_columns(layout)
    rows = len(columns[0]) + len(rhythm_leads(layout))
    ppm = grid_pitch(page)
    if ppm is None:
        return None, "no ECG grid was found, so the paper scale is unknown"
    ink = ink_mask(page, ppm)
    found = find_rows(ink, rows)
    if found is None:
        return None, f"fewer than {rows} trace rows were found for a {layout} layout"
    centres, pitch = found
    x0 = trace_start(ink, centres, pitch, ppm)
    if x0 is None:
        return None, "no calibration pulse was found at the start of the rows"
    x1 = x0 + SECONDS * MM_PER_SECOND * ppm
    if x1 > page.shape[1] * 1.03:
        return None, "the ten-second trace would run past the page; the grid scale looks wrong"
    return PageGeometry(layout, ppm, centres, pitch, x0, min(x1, page.shape[1] - 1)), ""


def crop_box(page: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """The page inside a box, padded with edge pixels where the box leaves the page.

    Clipping instead would make a strip at the page edge shorter than the others, and
    resizing it to the localizer's input would then stretch its trace."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    height, width = page.shape[:2]
    pad = max(0, -x0, -y0, x1 - width, y1 - height)
    if pad:
        page = cv2.copyMakeBorder(page, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return page[y0:y1, x0:x1]


def lead_segments(geometry: PageGeometry) -> list[dict]:
    """Every printed lead segment: lead, its time window in seconds, and its pixel box."""
    columns = layout_columns(geometry.layout)
    width = (geometry.x_end - geometry.x_start) / len(columns)
    seconds = SECONDS / len(columns)
    half = geometry.row_pitch / 2
    out = []
    for c, column in enumerate(columns):
        for r, lead in enumerate(column):
            y = geometry.row_centres[r]
            out.append({"lead": lead, "t0": c * seconds, "t1": (c + 1) * seconds,
                        "box": (geometry.x_start + c * width, y - half, geometry.x_start + (c + 1) * width, y + half)})
    for k, lead in enumerate(rhythm_leads(geometry.layout)):
        y = geometry.row_centres[len(columns[0]) + k]
        out.append({"lead": lead, "t0": 0.0, "t1": SECONDS, "rhythm": True,
                    "box": (geometry.x_start, y - half, geometry.x_end, y + half)})
    return out
