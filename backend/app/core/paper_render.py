"""Shared paper-web rendering primitives.

Used by:
  * scripts/generate_synthetic_data.py  -- offline dataset generation
  * backend/app/core/simulator.py        -- live synthetic camera frames

The functions synthesise grayscale paper-web images and inject the defect
families targeted by the Bistun Paper Industries project:

    stain, hole, non_uniform, scratch, inclusion   (+ clean)

Design goal: defects must be *realistic* (matched to the OpenCV detector in
``detector.py``) so that a classifier trained on synthetic features transfers
to the real NEU-DET reference images.  All randomness flows from a single
``numpy.random.Generator`` so datasets and the live simulator are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

DefectClass = Literal["clean", "stain", "hole", "non_uniform", "scratch", "inclusion"]

# Ordered list of classes used everywhere in the project (label encoding).
CLASSES: list[str] = ["clean", "stain", "hole", "non_uniform", "scratch", "inclusion"]

# Human-readable English + Persian labels for the dashboard / docs.
LABELS: dict[str, dict[str, str]] = {
    "clean":       {"en": "Clean",          "fa": "سالم"},
    "stain":       {"en": "Stain",          "fa": "لکه"},
    "hole":        {"en": "Hole",           "fa": "سوراخ"},
    "non_uniform": {"en": "Non-uniformity", "fa": "نایکنواختی"},
    "scratch":     {"en": "Scratch",        "fa": "خراش"},
    "inclusion":   {"en": "Inclusion",      "fa": "آخال"},
}


# ---------------------------------------------------------------------------
# Base paper texture
# ---------------------------------------------------------------------------
def base_paper(
    h: int = 256,
    w: int = 256,
    brightness: float = 212.0,
    formation: float = 0.25,
    fiber_noise: float = 8.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render a clean paper-web grayscale image.

    Parameters mirror real paper-machine camera behaviour:
      * ``brightness`` -- mean web grey level (writing/printing paper ~210-220).
      * ``formation``  -- low-frequency mottling amplitude (fiber formation).
      * ``fiber_noise``-- high-frequency sensor/fiber noise (std, grey levels).
    """
    rng = rng or np.random.default_rng()
    # low-frequency formation mottling via upscaled small noise field
    sh, sw = max(2, h // 8), max(2, w // 8)
    mottle = rng.normal(0, 1, (sh + 2, sw + 2))
    mottle = cv2.resize(mottle, (w, h), interpolation=cv2.INTER_CUBIC)
    mottle = (mottle - mottle.mean()) / (mottle.std() + 1e-6)  # normalise
    fine = rng.normal(0, 1, (h, w))
    img = brightness + formation * 28.0 * mottle + fiber_noise * fine
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Defect injectors (each returns (image, list_of_defect_records))
# ---------------------------------------------------------------------------
@dataclass
class DefectRecord:
    kind: str
    x: int            # bbox left
    y: int            # bbox top
    w: int            # bbox width
    h: int            # bbox height
    severity: float   # 0..1
    extra: dict = field(default_factory=dict)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def add_stain(img: np.ndarray, rng: np.random.Generator, n: int = 1) -> tuple[np.ndarray, list[DefectRecord]]:
    """Dark (or偶尔 bright) blotches -- oil/water/chemical stains."""
    out = img.copy()
    recs: list[DefectRecord] = []
    h, w = out.shape
    for _ in range(n):
        cx, cy = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
        rx, ry = int(rng.integers(12, 45)), int(rng.integers(10, 38))
        dark = rng.normal(95, 25)
        dark = float(np.clip(dark, 50, 150))
        mask = np.zeros((h, w), np.uint8)
        ang = int(rng.integers(0, 180))
        cv2.ellipse(mask, (cx, cy), (rx, ry), ang, 0, 360, 255, -1)
        soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=rx / 2.2)
        blend = (soft.astype(np.float32) / 255.0)[..., None]
        layer = np.full_like(out, dark, np.float32)
        out_f = out.astype(np.float32) * (1 - blend[..., 0]) + layer * blend[..., 0]
        out = np.clip(out_f, 0, 255).astype(np.uint8)
        bb = _bbox_from_mask(mask) or (cx - rx, cy - ry, 2 * rx, 2 * ry)
        recs.append(DefectRecord("stain", *bb, severity=float(np.clip((160 - dark) / 110, 0, 1))))
    return out, recs


def add_hole(img: np.ndarray, rng: np.random.Generator, n: int = 1) -> tuple[np.ndarray, list[DefectRecord]]:
    """Small sharp punctures -- thin spots / actual holes (appear very dark)."""
    out = img.copy()
    recs: list[DefectRecord] = []
    h, w = out.shape
    for _ in range(n):
        cx, cy = int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))
        r = int(rng.integers(3, 9))
        val = float(np.clip(rng.normal(35, 12), 8, 70))
        cv2.circle(out, (cx, cy), r, int(val), -1)
        # faint dark halo
        cv2.circle(out, (cx, cy), r + 2, int(np.clip(val + 25, 0, 255)), 1)
        recs.append(DefectRecord("hole", cx - r, cy - r, 2 * r, 2 * r, severity=float(np.clip((90 - val) / 80, 0, 1))))
    return out, recs


def add_non_uniform(img: np.ndarray, rng: np.random.Generator, n: int = 1) -> tuple[np.ndarray, list[DefectRecord]]:
    """Whole-image basis-weight / formation non-uniformity (large gradient/mottle)."""
    h, w = img.shape
    out = img.astype(np.float32)
    # low-frequency shading: sum of 2 large sinusoids + a big blurred blob
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fx = rng.uniform(0.6, 1.6)
    fy = rng.uniform(0.6, 1.6)
    phx, phy = rng.uniform(0, 6.28, 2)
    amp = rng.uniform(18, 34)
    grad = amp * (np.sin(xx / w * fx * 6.28 + phx) + np.sin(yy / h * fy * 6.28 + phy)) / 2
    cx, cy = int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4))
    blob = np.zeros((h, w), np.float32)
    cv2.circle(blob, (cx, cy), int(rng.integers(60, 110)), 1.0, -1)
    blob = cv2.GaussianBlur(blob, (0, 0), 60) * rng.uniform(-22, 22)
    out = out + grad + blob
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, [DefectRecord("non_uniform", 0, 0, w, h, severity=float(np.clip((amp - 14) / 22, 0, 1)),
                              extra={"cv": float(grad.std() / (out.mean() + 1e-6))})]


def add_scratch(img: np.ndarray, rng: np.random.Generator, n: int = 1) -> tuple[np.ndarray, list[DefectRecord]]:
    """Thin dark linear defects -- wire marks / blade streaks."""
    out = img.copy()
    recs: list[DefectRecord] = []
    h, w = out.shape
    for _ in range(n):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        ang = rng.uniform(0, 6.283)
        length = int(rng.integers(90, 220))
        x1 = int(np.clip(x0 + length * np.cos(ang), 0, w - 1))
        y1 = int(np.clip(y0 + length * np.sin(ang), 0, h - 1))
        thick = int(rng.integers(1, 3))
        val = int(np.clip(rng.normal(110, 20), 60, 160))
        cv2.line(out, (x0, y0), (x1, y1), val, thick)
        bbx, bby, bbw, bbh = min(x0, x1), min(y0, y1), abs(x1 - x0) + thick, abs(y1 - y0) + thick
        recs.append(DefectRecord("scratch", bbx, bby, bbw, bbh,
                                 severity=float(np.clip((170 - val) / 110, 0, 1))))
    return out, recs


def add_inclusion(img: np.ndarray, rng: np.random.Generator, n: int = 1) -> tuple[np.ndarray, list[DefectRecord]]:
    """Tiny high-contrast specks -- embedded foreign matter / dirt."""
    out = img.copy()
    recs: list[DefectRecord] = []
    h, w = out.shape
    for _ in range(n):
        cx, cy = int(rng.integers(5, w - 5)), int(rng.integers(5, h - 5))
        r = int(rng.integers(1, 4))
        bright = rng.random() < 0.5
        val = int(np.clip(rng.normal(60 if not bright else 250, 10), 0, 255))
        cv2.circle(out, (cx, cy), r, val, -1)
        recs.append(DefectRecord("inclusion", cx - r, cy - r, 2 * r, 2 * r,
                                 severity=float(np.clip(abs(int(val) - int(img[cy, cx])) / 200, 0, 1))))
    return out, recs


_INJECT = {
    "stain": add_stain,
    "hole": add_hole,
    "non_uniform": add_non_uniform,
    "scratch": add_scratch,
    "inclusion": add_inclusion,
}


def render_defect(
    img: np.ndarray, kind: str, rng: np.random.Generator, n: int | None = None
) -> tuple[np.ndarray, list[DefectRecord]]:
    """Inject one defect family into ``img``. ``n`` defaults per family."""
    if kind == "clean":
        return img, []
    defaults = {"stain": 2, "hole": 3, "non_uniform": 1, "scratch": 2, "inclusion": 5}
    return _INJECT[kind](img, rng, n if n is not None else defaults[kind])


def render_sample(
    label: str,
    h: int = 256,
    w: int = 256,
    rng: np.random.Generator | None = None,
    brightness: float | None = None,
) -> tuple[np.ndarray, list[DefectRecord]]:
    """Render a full labelled sample: clean base + the defect family of ``label``."""
    rng = rng or np.random.default_rng()
    if brightness is None:
        brightness = float(rng.uniform(198, 224))
    img = base_paper(h, w, brightness=brightness,
                     formation=float(rng.uniform(0.18, 0.34)),
                     fiber_noise=float(rng.uniform(6, 11)), rng=rng)
    img, recs = render_defect(img, label, rng)
    return img, recs
