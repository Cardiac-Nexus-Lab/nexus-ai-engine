"""Turn a clean rendered ECG into something resembling a photographed printout.

A digitizer trained only on crisp renders learns the render, not the page, and
collapses on the first phone photo it sees. The transforms here reproduce what
actually happens between a printout and an uploaded image: the camera is held at
an angle, the lighting is uneven, the paper is creased, the sensor adds noise,
and the file is saved as JPEG.

Each distortion returns the homography it applied where relevant, so the trace
geometry recorded at render time can be carried through and the pairing between
image and ground-truth signal survives.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DistortionConfig:
    perspective: float = 0.6
    max_corner_shift: float = 0.035   # fraction of the smaller image side
    rotation: float = 0.7
    max_rotation_degrees: float = 4.0
    lighting: float = 0.8
    shadow: float = 0.35
    crease: float = 0.25
    blur: float = 0.5
    max_blur_sigma: float = 1.2
    sensor_noise: float = 0.6
    max_noise_std: float = 6.0
    jpeg: float = 0.7
    min_jpeg_quality: int = 45
    brightness_contrast: float = 0.7


def _perspective(image: np.ndarray, rng: np.random.Generator,
                 config: DistortionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Warp as if the camera were not square to the page."""
    height, width = image.shape[:2]
    shift = config.max_corner_shift * min(height, width)
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    destination = source + rng.uniform(-shift, shift, source.shape).astype(np.float32)

    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        image, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, matrix


def _rotate(image: np.ndarray, rng: np.random.Generator,
            config: DistortionConfig) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    angle = float(rng.uniform(-config.max_rotation_degrees, config.max_rotation_degrees))
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, np.vstack([matrix, [0, 0, 1]]).astype(np.float32)


def _lighting_gradient(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Smooth brightness ramp, as from a window or overhead light."""
    height, width = image.shape[:2]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = float(rng.uniform(0, 2 * np.pi))
    ramp = np.cos(angle) * (x / width) + np.sin(angle) * (y / height)
    ramp = (ramp - ramp.min()) / (np.ptp(ramp) + 1e-9)
    strength = float(rng.uniform(0.10, 0.30))
    field = (1.0 - strength / 2) + strength * ramp
    return np.clip(image.astype(np.float32) * field[..., None], 0, 255).astype(np.uint8)


def _soft_shadow(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A blurred dark blob, as cast by the photographer or the phone itself."""
    height, width = image.shape[:2]
    mask = np.ones((height, width), np.float32)
    centre = (int(rng.uniform(0, width)), int(rng.uniform(0, height)))
    axes = (int(rng.uniform(width * 0.2, width * 0.7)), int(rng.uniform(height * 0.2, height * 0.8)))
    cv2.ellipse(mask, centre, axes, float(rng.uniform(0, 180)), 0, 360,
                float(rng.uniform(0.55, 0.85)), -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(width, height) * 0.05)
    return np.clip(image.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def _crease(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A fold line: paper that has been in a pocket."""
    height, width = image.shape[:2]
    field = np.ones((height, width), np.float32)
    if rng.random() < 0.5:
        position = int(rng.uniform(height * 0.2, height * 0.8))
        field[max(0, position - 1):position + 2, :] *= float(rng.uniform(0.75, 0.92))
    else:
        position = int(rng.uniform(width * 0.2, width * 0.8))
        field[:, max(0, position - 1):position + 2] *= float(rng.uniform(0.75, 0.92))
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=2.0)
    return np.clip(image.astype(np.float32) * field[..., None], 0, 255).astype(np.uint8)


def _jpeg(image: np.ndarray, rng: np.random.Generator, config: DistortionConfig) -> np.ndarray:
    quality = int(rng.integers(config.min_jpeg_quality, 96))
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else image


def distort(image: np.ndarray, rng: np.random.Generator | None = None,
            config: DistortionConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random subset of photographic degradations.

    Returns the distorted image and the 3x3 homography mapping original pixel
    coordinates onto the result, so rendered trace geometry can be transformed to
    match.
    """
    rng = rng or np.random.default_rng()
    config = config or DistortionConfig()

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result = image.copy()
    homography = np.eye(3, dtype=np.float32)

    # Geometry first: photometric effects should sit on the final shape.
    if rng.random() < config.rotation:
        result, matrix = _rotate(result, rng, config)
        homography = matrix @ homography
    if rng.random() < config.perspective:
        result, matrix = _perspective(result, rng, config)
        homography = matrix @ homography

    if rng.random() < config.lighting:
        result = _lighting_gradient(result, rng)
    if rng.random() < config.shadow:
        result = _soft_shadow(result, rng)
    if rng.random() < config.crease:
        result = _crease(result, rng)

    if rng.random() < config.brightness_contrast:
        alpha = float(rng.uniform(0.85, 1.15))
        beta = float(rng.uniform(-18, 18))
        result = np.clip(result.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < config.blur:
        result = cv2.GaussianBlur(result, (0, 0), sigmaX=float(rng.uniform(0.4, config.max_blur_sigma)))
    if rng.random() < config.sensor_noise:
        noise = rng.normal(0, float(rng.uniform(2, config.max_noise_std)), result.shape)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # JPEG last, as the final save the user's phone performs.
    if rng.random() < config.jpeg:
        result = _jpeg(result, rng, config)

    return result, homography


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Map (N, 2) pixel coordinates through a homography."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)
