"""Photograph of an ECG printout to prediction, end to end.

Four stages, each of which can fail in its own way and reports so rather than
passing a confident wrong answer to the next:

1. locate the page in the photograph and remove the camera's perspective;
2. cut the dewarped page into lead strips using the layout's known geometry;
3. read a waveform off each strip with the trace localizer;
4. rescale to millivolts and hand the assembled 12-lead signal to the classifier.

Stage 1 is deliberately classical rather than learned. Finding a bright
quadrilateral against a darker background is a solved problem, needs no training
data, and fails visibly when the assumption breaks, which is easier to debug and
to explain than a network that quietly mislocates the page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .digitize import (
    STRIP_HEIGHT,
    STRIP_SECONDS,
    STRIP_WIDTH,
    TraceLocalizer,
    expected_rows,
    row_confidence,
)
from .render import MM_PER_MV, MM_PER_SECOND, PaperSpec  # noqa: F401 - paper constants

# The layout the digitizer was trained on. Strip extraction must reproduce the
# geometry `StripDataset` used, or the model reads the wrong part of the page.
TRAINING_SPEC = PaperSpec(layout="12x1", pixels_per_mm=4.0, row_height_mm=16.0, margin_mm=4.0)


def layout_margins(spec: PaperSpec = TRAINING_SPEC, leads: int = 12,
                   seconds: float = 10.0) -> tuple[float, float]:
    """Page margins as fractions of height and width.

    These are two different numbers even though the paper margin is one number,
    because the page is wider than it is tall. Using a single fraction for both
    axes was the original bug: it put the horizontal cut 25 px late on a 1000 px
    trace, which time-shifts the recovered signal and drives correlation to zero
    no matter how well the localizer reads a strip.
    """
    margin = spec.mm(spec.margin_mm)
    height = leads * spec.mm(spec.row_height_mm) + 2 * margin
    width = spec.mm(seconds * MM_PER_SECOND) + 2 * margin
    return margin / height, margin / width


@dataclass
class DigitizedECG:
    signal: np.ndarray                  # [12, samples], standardized like training data
    confidence: np.ndarray              # [12, samples], per-sample peak probability
    page: np.ndarray                    # the dewarped page, for display
    warnings: list[str] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float:
        return float(self.confidence.mean())


def find_page(image: np.ndarray, min_area_fraction: float = 0.5,
              expected_aspect: float | None = None,
              aspect_tolerance: float = 0.35) -> tuple[np.ndarray, bool]:
    """Locate the printout and rectify it to a front-on view.

    Returns the dewarped page and whether a quadrilateral was actually found. When
    none is, the original image is passed through: a photograph cropped tightly to
    the page has no border to detect, and is already usable.

    A candidate must cover most of the frame and have roughly the layout's aspect
    ratio. Without the aspect test, Canny on an ECG printout readily finds a
    quadrilateral that is not the page: the grid, a shadow edge, or the band of
    traces itself. One such false positive was measured returning 325x998 for an
    800x1032 page, discarding 60% of the leads while reporting success. Passing
    the image through unchanged is a far better failure than confidently
    rectifying the wrong rectangle.
    """
    if expected_aspect is None:
        vertical, horizontal = layout_margins()
        expected_aspect = (1.0 / horizontal) / (1.0 / vertical)  # width/height of the layout
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Close small gaps so a slightly broken page border still forms one contour.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, False

    height, width = image.shape[:2]
    minimum_area = min_area_fraction * height * width

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contour) < minimum_area:
            break
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) != 4:
            continue

        corners = _order_corners(approximation.reshape(4, 2).astype(np.float32))
        target_width = int(max(np.linalg.norm(corners[1] - corners[0]),
                               np.linalg.norm(corners[2] - corners[3])))
        target_height = int(max(np.linalg.norm(corners[3] - corners[0]),
                                np.linalg.norm(corners[2] - corners[1])))
        if target_width < 50 or target_height < 50:
            continue

        aspect = target_width / target_height
        if abs(aspect - expected_aspect) / expected_aspect > aspect_tolerance:
            continue

        destination = np.float32([[0, 0], [target_width, 0],
                                  [target_width, target_height], [0, target_height]])
        matrix = cv2.getPerspectiveTransform(corners, destination)
        return cv2.warpPerspective(image, matrix, (target_width, target_height)), True

    return image, False


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Order four corners as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    totals = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(totals)]
    ordered[2] = points[np.argmax(totals)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def extract_strips(page: np.ndarray, leads: int = 12, windows: int = 4,
                   margins: tuple[float, float] | None = None) -> list[np.ndarray]:
    """Cut a 12x1 page into per-lead, per-window strips.

    The layout is assumed rather than detected. That is a real limitation: a 3x4
    printout cut this way yields nonsense. It is stated here so the caller can
    check the layout instead of trusting the output.
    """
    vertical_fraction, horizontal_fraction = margins or layout_margins(leads=leads)

    height, width = page.shape[:2]
    top_margin = int(height * vertical_fraction)
    usable = height - 2 * top_margin
    row_height = usable / leads

    left_margin = int(width * horizontal_fraction)
    usable_width = width - 2 * left_margin
    window_width = usable_width / windows

    strips = []
    for lead in range(leads):
        row_top = int(top_margin + lead * row_height)
        row_bottom = int(top_margin + (lead + 1) * row_height)
        for window in range(windows):
            left = int(left_margin + window * window_width)
            right = int(left_margin + (window + 1) * window_width)
            crop = page[row_top:row_bottom, left:right]
            if crop.size == 0:
                crop = np.full((STRIP_HEIGHT, STRIP_WIDTH, 3), 255, np.uint8)
            strips.append(cv2.resize(crop, (STRIP_WIDTH, STRIP_HEIGHT), interpolation=cv2.INTER_AREA))
    return strips


@torch.no_grad()
def digitize_page(image: np.ndarray, model: TraceLocalizer, device: torch.device | None = None,
                  leads: int = 12, windows: int = 4, confidence_floor: float = 0.15,
                  margins: tuple[float, float] | None = None,
                  sampling_rate: int = 100) -> DigitizedECG:
    """Read a 12-lead signal off a photographed printout."""
    device = device or next(model.parameters()).device
    model.eval()

    page, found = find_page(image)
    warnings: list[str] = []
    if not found:
        warnings.append("No page border detected; using the image as-is. "
                        "Perspective was not corrected.")

    strips = extract_strips(page, leads, windows, margins)
    batch = torch.from_numpy(
        np.stack([s.astype(np.float32).transpose(2, 0, 1) / 255.0 for s in strips])
    ).to(device)

    logits = model(batch)
    rows = expected_rows(logits).cpu().numpy()          # [leads*windows, W]
    confidence = row_confidence(logits).cpu().numpy()

    rows = rows.reshape(leads, windows * STRIP_WIDTH)
    confidence = confidence.reshape(leads, windows * STRIP_WIDTH)

    # Rows increase downwards and voltage upwards, so deflection is measured from
    # the strip's centre. No pixels-per-millivolt factor is applied, because the
    # printout's gain is not known from the image alone and the standardization
    # below would divide any such constant out again.
    #
    # The cost is that absolute millivolts are not recovered, only the shape. That
    # is acceptable here precisely because the classifier was itself trained on
    # per-lead standardized signals and has never seen absolute voltages, so the
    # digitized input matches its training distribution. A model that relied on
    # true voltage criteria would need the calibration pulse read off the page.
    signal = (STRIP_HEIGHT / 2.0) - rows

    # Put the signal back on the recording's own time base.
    #
    # A window holds STRIP_SECONDS of signal but is resized to STRIP_WIDTH pixels
    # for the model, and those two are not equal: 2.5 s at 100 Hz is 250 samples
    # rendered into 256 columns. Concatenating four windows therefore yields 1024
    # columns describing 1000 samples, so every window starts progressively later
    # than it should. Read column-for-sample, the trace drifts out of time by six
    # samples per window and correlation collapses even when each strip was read
    # perfectly.
    samples = int(round(windows * STRIP_SECONDS * sampling_rate))
    if signal.shape[1] != samples:
        source = np.linspace(0.0, 1.0, signal.shape[1])
        target = np.linspace(0.0, 1.0, samples)
        signal = np.stack([np.interp(target, source, lead) for lead in signal])
        confidence = np.stack([np.interp(target, source, lead) for lead in confidence])

    weak = confidence < confidence_floor
    if weak.any():
        warnings.append(
            f"{100 * weak.mean():.1f}% of samples were read with low confidence; "
            "the trace may be obscured there."
        )

    # Match the standardization the classifier was trained on.
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True) + 1e-6
    signal = (signal - mean) / std

    return DigitizedECG(signal=signal.astype(np.float32), confidence=confidence,
                        page=page, warnings=warnings)


def load_localizer(checkpoint_path, device: torch.device | None = None) -> TraceLocalizer:
    device = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TraceLocalizer(height=checkpoint.get("strip_height", STRIP_HEIGHT))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
