"""ECG paper-image rendering, distortion, and digitization."""

from .render import LEAD_NAMES, PaperSpec, RenderedECG, render

__all__ = ["render", "PaperSpec", "RenderedECG", "LEAD_NAMES"]
