"""Machine-vision defect detector + feature extractor for the paper web.

This is the *vision* half of the Bistun "Integrated Quality & Energy Control"
product.  It is deliberately a **classical computer-vision** pipeline (no deep
learning dependency) so that it:

  * runs in real time on the live camera feed (a few ms per frame on CPU),
  * is fully deterministic and auditable (important for QA sign-off),
  * needs no GPU -- deployable on a factory edge box,
  * provides *localised* defects (bounding boxes) for the operator dashboard,
    not just a label.

Pipeline (per grayscale web frame):

  1.  Preprocess: denoise (bilateral), grayscale.
  2.  Defect localisation
      * dark blobs  -> stains / thin spots          (adaptive threshold + CC)
      * sharp small punctures -> holes / inclusions  (morph. black-hat + CC, size split)
      * linear features -> scratches                  (anisotropic morphology / CC eccentricity)
      * whole-image formation non-uniformity          (grid local-mean CV + entropy)
  3.  Feature vector (25 dims) summarising the frame -- used by the trained
      scikit-learn classifier (see ``classifier.py``) to assign a class label.
  4.  Aggregate ``quality_score`` (0..1, 1 = perfect web) and per-defect severity.

The detector works without the trained classifier (returns localised defects +
features); when a classifier is loaded it also returns the predicted class and
confidence.  This separation makes the detector unit-testable independent of ML.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# Feature vector schema -- exported so the classifier + trainer agree on order.
FEATURE_NAMES: list[str] = [
    "mean", "std", "cv",
    "local_mean_cv",          # non-uniformity (grid of local means)
    "entropy",
    "dark_frac",              # fraction below adaptive threshold (stain proxy)
    "bright_frac",            # fraction above high threshold (bright inclusion/hole-light)
    "dark_blob_count",
    "dark_blob_area_mean",
    "dark_blob_area_max",
    "dark_blob_area_total",
    "dark_blob_ecc_mean",     # scratch proxy via elongation of dark components
    "small_dark_count",       # holes / inclusions (sharp tiny dark specks)
    "small_dark_area_total",
    "small_bright_count",     # bright inclusions
    "small_bright_area_total",
    "scratch_score",          # anisotropic linear-feature response
    "grad_anisotropy",        # directional gradient ratio
    "laplacian_var",          # focus / sharpness
    "nonuniform_score",       # composite formation non-uniformity
    "stain_score",
    "hole_score",
    "inclusion_score",
    "nonuniform_grid_maxmin", # max-min local mean / global mean
    "defect_area_ratio",      # total defect pixels / frame pixels
]
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class Defect:
    kind: str                 # stain | hole | non_uniform | scratch | inclusion
    x: int
    y: int
    w: int
    h: int
    severity: float           # 0..1
    area: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "severity": round(float(self.severity), 4), "area": int(self.area),
            "extra": self.extra,
        }


@dataclass
class DetectionResult:
    defects: list[Defect]
    features: np.ndarray
    feature_dict: dict[str, float]
    quality_score: float          # 0..1
    defect_area_ratio: float
    pred_class: str | None = None
    confidence: float | None = None
    h: int = 0
    w: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "defects": [d.to_dict() for d in self.defects],
            "n_defects": len(self.defects),
            "quality_score": round(float(self.quality_score), 4),
            "defect_area_ratio": round(float(self.defect_area_ratio), 5),
            "features": {k: round(float(v), 5) for k, v in self.feature_dict.items()},
            "pred_class": self.pred_class,
            "confidence": None if self.confidence is None else round(float(self.confidence), 4),
            "h": self.h, "w": self.w,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _entropy(hist: np.ndarray) -> float:
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _eccentricity(stats, label_idx) -> float:
    """Approximate elongation of a connected component from its bbox."""
    w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
    h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
    if w == 0 or h == 0:
        return 0.0
    return float(max(w, h) / min(w, h))


def _components(mask: np.ndarray, min_area: int, max_area: int):
    """Label components within [min_area, max_area] -> (n, labels, stats, centroids)."""
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):  # skip bg
        a = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= a <= max_area:
            out.append(i)
    return out, labels, stats, cent


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------
class PaperDefectDetector:
    """Classical-CV defect detector + feature extractor.

    Parameters
    ----------
    patch_grid : int
        Grid size for local-mean non-uniformity (e.g. 8 -> 8x8 cells).
    small_area_max : int
        Max component area considered a "small" puncture (hole/inclusion).
    """

    def __init__(self, patch_grid: int = 8, small_area_max: int = 90,
                 min_blob_area: int = 12, stain_min_area: int = 35,
                 normalize: bool = True,
                 target_mean: float = 128.0, target_std: float = 40.0) -> None:
        self.patch_grid = patch_grid
        self.small_area_max = small_area_max
        self.min_blob_area = min_blob_area
        self.stain_min_area = stain_min_area   # only sizable blotches count as stains
        self.normalize = normalize
        self.target_mean = target_mean
        self.target_std = target_std
        self.classifier = None  # set via attach_classifier()

    # -- public --------------------------------------------------------------
    def attach_classifier(self, clf) -> None:
        """Attach a trained ``DefectClassifier`` (or any object with
        ``predict_proba(features)`` and ``classes_``)."""
        self.classifier = clf

    def detect(self, bgr_or_gray: np.ndarray) -> DetectionResult:
        img = self._to_gray(bgr_or_gray)
        if self.normalize:
            img = self._normalize(img)
        h, w = img.shape
        feats: dict[str, float] = {}

        # --- 1. global stats ---
        mean = float(img.mean())
        std = float(img.std())
        cv_ = std / (mean + 1e-6)
        feats["mean"] = mean
        feats["std"] = std
        feats["cv"] = cv_

        # --- 2. local-mean non-uniformity (grid) ---
        gh, gw = self.patch_grid, self.patch_grid
        cell_h, cell_w = h // gh, w // gw
        local_means = []
        for r in range(gh):
            for c in range(gw):
                block = img[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
                if block.size:
                    local_means.append(float(block.mean()))
        lm = np.array(local_means, dtype=np.float64)
        local_mean_cv = float(lm.std() / (lm.mean() + 1e-6))
        feats["local_mean_cv"] = local_mean_cv
        feats["nonuniform_grid_maxmin"] = float((lm.max() - lm.min()) / (lm.mean() + 1e-6))

        # --- 3. entropy / sharpness / gradients ---
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
        feats["entropy"] = _entropy(hist)
        lap = cv2.Laplacian(img, cv2.CV_64F)
        feats["laplacian_var"] = float(lap.var())
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx ** 2 + gy ** 2)
        # directional energy ratio (scratch -> high anisotropy)
        gx_e = float(np.abs(gx).sum())
        gy_e = float(np.abs(gy).sum())
        feats["grad_anisotropy"] = float(abs(gx_e - gy_e) / (gx_e + gy_e + 1e-6))

        # --- 4. denoise for blob analysis ---
        den = cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)

        # adaptive threshold for dark blobs (stains / dark defects).
        # blockSize=51 averages over several mottle cycles so low-frequency
        # formation mottle (~+/-20 grey levels post-normalisation) does not
        # register as dark blobs; C=24 rejects residual mottle while catching
        # genuine stains/holes (deviation 50-100+).
        dark_mask = cv2.adaptiveThreshold(
            den, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            blockSize=51, C=24)
        # remove tiny specks for the "stain" channel
        stain_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # black-hat for sharp small dark punctures (holes). Fixed threshold (35)
        # above the normalised-noise floor; Otsu on noisy paper sits too low and
        # flags thousands of specks. Open 3x3 kills residual 1-2px noise.
        blackhat = cv2.morphologyEx(den, cv2.MORPH_BLACKHAT,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        _, small_dark_mask = cv2.threshold(blackhat, 35, 255, cv2.THRESH_BINARY)
        small_dark_mask = cv2.morphologyEx(small_dark_mask, cv2.MORPH_OPEN,
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        # bright small specks (inclusions)
        tophat = cv2.morphologyEx(den, cv2.MORPH_TOPHAT,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        _, small_bright_mask = cv2.threshold(tophat, 35, 255, cv2.THRESH_BINARY)
        small_bright_mask = cv2.morphologyEx(small_bright_mask, cv2.MORPH_OPEN,
                                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        feats["dark_frac"] = float((dark_mask > 0).mean())
        feats["bright_frac"] = float((small_bright_mask > 0).mean())

        # --- 5. component analysis for localisation + features ---
        defects: list[Defect] = []
        # stains: medium-large dark components (sizable blotches only)
        stain_ids, _, s_stats, _ = _components(stain_mask, self.stain_min_area, 10_000)
        for i in stain_ids:
            a = int(s_stats[i, cv2.CC_STAT_AREA])
            x = int(s_stats[i, cv2.CC_STAT_LEFT]); y = int(s_stats[i, cv2.CC_STAT_TOP])
            bw = int(s_stats[i, cv2.CC_STAT_WIDTH]); bh = int(s_stats[i, cv2.CC_STAT_HEIGHT])
            sev = float(np.clip(a / (cell_h * cell_w * 1.5), 0, 1))
            defects.append(Defect("stain", x, y, bw, bh, sev, a,
                                  extra={"elong": _eccentricity(s_stats, i)}))
        feats["dark_blob_count"] = float(len(stain_ids))
        areas = [int(s_stats[i, cv2.CC_STAT_AREA]) for i in stain_ids] or [0]
        feats["dark_blob_area_mean"] = float(np.mean(areas))
        feats["dark_blob_area_max"] = float(np.max(areas))
        feats["dark_blob_area_total"] = float(np.sum(areas))
        eccs = [_eccentricity(s_stats, i) for i in stain_ids] or [0]
        feats["dark_blob_ecc_mean"] = float(np.mean(eccs))

        # small dark punctures -> holes (sharp) vs inclusions (very small)
        sdark_ids, _, sd_stats, _ = _components(small_dark_mask, 5, self.small_area_max)
        hole_ids, incl_dark_ids = [], []
        for i in sdark_ids:
            a = int(sd_stats[i, cv2.CC_STAT_AREA])
            (hole_ids if a >= 9 else incl_dark_ids).append(i)
        for i in hole_ids:
            a = int(sd_stats[i, cv2.CC_STAT_AREA])
            x = int(sd_stats[i, cv2.CC_STAT_LEFT]); y = int(sd_stats[i, cv2.CC_STAT_TOP])
            bw = int(sd_stats[i, cv2.CC_STAT_WIDTH]); bh = int(sd_stats[i, cv2.CC_STAT_HEIGHT])
            defects.append(Defect("hole", x, y, bw, bh, float(np.clip(a / 40, 0, 1)), a))
        feats["small_dark_count"] = float(len(sdark_ids))
        feats["small_dark_area_total"] = float(sum(int(sd_stats[i, cv2.CC_STAT_AREA]) for i in sdark_ids))

        # bright inclusions (only sizable specks -> Defects)
        sbright_ids, _, sb_stats, _ = _components(small_bright_mask, 5, self.small_area_max)
        for i in sbright_ids:
            a = int(sb_stats[i, cv2.CC_STAT_AREA])
            x = int(sb_stats[i, cv2.CC_STAT_LEFT]); y = int(sb_stats[i, cv2.CC_STAT_TOP])
            bw = int(sb_stats[i, cv2.CC_STAT_WIDTH]); bh = int(sb_stats[i, cv2.CC_STAT_HEIGHT])
            defects.append(Defect("inclusion", x, y, bw, bh, float(np.clip(a / 30, 0, 1)), a))
        feats["small_bright_count"] = float(len(sbright_ids))
        feats["small_bright_area_total"] = float(sum(int(sb_stats[i, cv2.CC_STAT_AREA]) for i in sbright_ids))

        # scratches: elongated dark components (eccentricity high, thin)
        scratch_mask = cv2.morphologyEx(
            dark_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
        scr_ids, _, sc_stats, _ = _components(scratch_mask, 25, 10_000)
        scratch_count = 0
        for i in scr_ids:
            a = int(sc_stats[i, cv2.CC_STAT_AREA])
            bw = int(sc_stats[i, cv2.CC_STAT_WIDTH]); bh = int(sc_stats[i, cv2.CC_STAT_HEIGHT])
            elong = max(bw, bh) / max(1, min(bw, bh))
            if elong >= 4.0 and a >= 40:   # genuinely linear
                x = int(sc_stats[i, cv2.CC_STAT_LEFT]); y = int(sc_stats[i, cv2.CC_STAT_TOP])
                defects.append(Defect("scratch", x, y, bw, bh,
                                      float(np.clip(elong / 12, 0, 1)), a, extra={"elong": elong}))
                scratch_count += 1
        # scratch score from anisotropic gradient energy along principal axes
        # (sum of horizontal vs vertical Sobel on blackhat)
        sb_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
        sb_y = cv2.Sobel(blackhat, cv2.CV_32F, 0, 1, ksize=3)
        line_score = float((np.abs(sb_x).sum() + np.abs(sb_y).sum()) / (h * w * 50))
        feats["scratch_score"] = float(max(scratch_count / 4.0, line_score))

        # non-uniformity composite
        feats["nonuniform_score"] = float(np.clip(local_mean_cv * 6.0 + feats["entropy"] / 10.0, 0, 1.2))
        if local_mean_cv > 0.09 or feats["nonuniform_grid_maxmin"] > 0.13:
            defects.append(Defect("non_uniform", 0, 0, w, h,
                                  severity=float(np.clip(local_mean_cv * 12.0, 0, 1)),
                                  area=int((lm.max() - lm.min()) > 8) * w * h,
                                  extra={"local_mean_cv": local_mean_cv}))

        # composite per-class scores (used as features + for sanity checks)
        feats["stain_score"] = float(np.clip(
            feats["dark_blob_area_total"] / (h * w * 0.05), 0, 1))
        feats["hole_score"] = float(np.clip(
            (len(hole_ids) + 0.4 * len(incl_dark_ids)) / 4.0, 0, 1))
        feats["inclusion_score"] = float(np.clip(
            (len(sbright_ids) + 0.5 * len(incl_dark_ids)) / 6.0, 0, 1))

        # defect area ratio from LOCALISED defects only (not raw masks, which
        # fire on mottle noise) -- this is the number shown on the dashboard and
        # fed to APC, so it must reflect real defects, not detection noise.
        defect_area_ratio = float(np.clip(
            sum(d.area for d in defects if d.kind != "non_uniform") / (h * w), 0.0, 1.0))
        feats["defect_area_ratio"] = defect_area_ratio

        # --- 6. assemble feature vector in canonical order ---
        fvec = np.array([feats.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float32)

        # --- 7. quality score (0..1): penalise localised area + severity ---
        sev_sum = sum(min(d.severity, 1.0) ** 2 for d in defects)
        nu_sev = next((d.severity for d in defects if d.kind == "non_uniform"), 0.0)
        quality = float(np.clip(
            1.0 - 1.6 * defect_area_ratio - 0.10 * sev_sum - 0.25 * nu_sev, 0.0, 1.0))

        # --- 8. optional classifier ---
        pred_class, conf = None, None
        if self.classifier is not None:
            try:
                pred_class, conf = self.classifier.predict_one(fvec)
            except Exception:
                pred_class, conf = None, None

        return DetectionResult(
            defects=defects, features=fvec, feature_dict=feats,
            quality_score=quality, defect_area_ratio=defect_area_ratio,
            pred_class=pred_class, confidence=conf, h=h, w=w,
        )

    # -- static helpers ------------------------------------------------------
    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Brightness normalisation (mean shift to ``target_mean``).

        Only the *mean* is shifted, NOT the standard deviation. Rationale: on
        a paper line the camera grey level drifts with lighting (the real
        physical disturbance), so removing the mean offset gives lighting
        invariance. Scaling the std to a fixed value would amplify the paper's
        own formation mottle/fiber noise up to defect-like contrast and flood
        the detector with false positives on clean web -- empirically thousands
        of specks per frame. Keeping the native contrast preserves the gap
        between genuine defects (large deviation) and mottle/noise (small).
        Local structure (blobs, gradients, non-uniformity ratios) is preserved.
        """
        m = float(img.mean())
        out = img.astype(np.float32) - m + self.target_mean
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def draw_overlay(img: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw defect boxes + labels on a copy of ``img`` (BGR). Returns BGR."""
        out = img.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        colors = {
            "stain": (0, 0, 220), "hole": (0, 80, 255), "non_uniform": (180, 120, 0),
            "scratch": (0, 200, 200), "inclusion": (200, 0, 200),
        }
        for d in result.defects:
            c = colors.get(d.kind, (255, 255, 255))
            cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), c, 2)
            label = f"{d.kind} {d.severity:.2f}"
            cv2.putText(out, label, (d.x, max(0, d.y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
        return out
