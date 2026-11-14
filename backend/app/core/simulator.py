"""Paper-machine process simulator -- closes the vision -> APC loop.

Generates a live stream of synthetic paper-web camera frames whose defect
content is *physically driven* by the current process setpoints (speed, steam,
chemical, consistency).  The detector measures each frame; the APC controller
re-optimises the setpoints; the simulator applies them with first-order actuator
lag.  This makes the live dashboard demonstrably show the product working:

    high speed / low steam  ->  more defects (vision sees them)
    APC retunes setpoints    ->  fewer defects next frames  ->  energy/emissions saved

The simulator is the TRL-5 "relevant environment": it stands in for the real
paper line with calibrated, auditable dynamics, and runs entirely on CPU.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .apc import APCController, ProcessState
from .detector import PaperDefectDetector
from .paper_render import CLASSES, render_sample


@dataclass
class CumulativeStats:
    frames: int = 0
    defect_frames: int = 0
    tonnes_saved: float = 0.0
    energy_saved_gj: float = 0.0
    co2_saved_kg: float = 0.0
    quality_sum: float = 0.0
    defect_rate_sum: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        n = max(1, self.frames)
        return {
            "frames": self.frames,
            "defect_frames": self.defect_frames,
            "defect_frame_rate": round(self.defect_frames / n, 4),
            "avg_quality": round(self.quality_sum / n, 4),
            "avg_defect_rate": round(self.defect_rate_sum / n, 5),
            "tonnes_saved": round(self.tonnes_saved, 3),
            "energy_saved_gj": round(self.energy_saved_gj, 3),
            "co2_saved_kg": round(self.co2_saved_kg, 2),
        }


class PaperMachineSimulator:
    """End-to-end simulator: render -> detect -> control -> advance."""

    def __init__(
        self,
        detector: PaperDefectDetector | None = None,
        apc: APCController | None = None,
        seed: int = 2026,
        frame_size: int = 256,
        jpeg_quality: int = 70,
        dt: float = 1.0,                 # simulated seconds per tick
        actuator_tau: float = 3.0,       # first-order lag (s)
    ) -> None:
        self.detector = detector or PaperDefectDetector()
        self.apc = apc or APCController()
        self.rng = np.random.default_rng(seed)
        self.frame_size = frame_size
        self.jpeg_quality = jpeg_quality
        self.dt = dt
        self.actuator_tau = actuator_tau
        self.reset()

    # -- state ----------------------------------------------------------------
    def reset(self) -> None:
        p = self.apc.p
        self.t = 0.0
        # start slightly off-optimal so APC has work to do
        self.speed = p["nom_speed"] + 90.0
        self.steam = p["nom_steam"] - 0.8
        self.chem = p["nom_chem"] - 0.3
        self.consistency = p["nom_consist"]
        self.cum = CumulativeStats()
        # exponential moving disturbance on consistency
        self._consist_drift = 0.0

    def _process_state(self, defect_rate: float, quality: float) -> ProcessState:
        return ProcessState(
            speed=self.speed, steam_pressure=self.steam, chem_dosage=self.chem,
            consistency=self.consistency, defect_rate=defect_rate, quality_score=quality,
        )

    # -- defect propensity (physical model) ----------------------------------
    def _propensity(self) -> float:
        """Probability the next frame is defective, driven by setpoints."""
        p = self.apc.p
        # use the APC quality model to get predicted defect fraction, then map to [0,1]
        dr = self.apc.defect_rate(self.speed, self.steam, self.chem, self.consistency)
        return float(np.clip(dr * 6.0, 0.02, 0.85))

    def _pick_defect_class(self) -> str:
        # weight classes by which lever is worst
        p = self.apc.p
        weights = {
            "stain": 1.0,
            "hole": 0.6,
            "non_uniform": 0.8 + 0.4 * max(0.0, (self.speed - p["nom_speed"]) / 200.0),
            "scratch": 0.5 + 0.5 * max(0.0, (self.speed - p["nom_speed"]) / 200.0),
            "inclusion": 0.5,
        }
        classes = list(weights.keys())
        w = np.array([weights[c] for c in classes], dtype=float)
        w /= w.sum()
        return str(self.rng.choice(classes, p=w))

    # -- one tick -------------------------------------------------------------
    def tick(self) -> dict[str, Any]:
        self.t += self.dt
        # slowly drift consistency (disturbance) so dynamics stay interesting
        self._consist_drift = 0.95 * self._consist_drift + 0.05 * float(self.rng.normal(0, 0.4))
        self.consistency = float(np.clip(self.apc.p["nom_consist"] + self._consist_drift,
                                         self.apc.p["nom_consist"] - 1.0,
                                         self.apc.p["nom_consist"] + 1.0))

        prop = self._propensity()
        if self.rng.random() < prop:
            label = self._pick_defect_class()
        else:
            label = "clean"

        img, _recs = render_sample(label, h=self.frame_size, w=self.frame_size, rng=self.rng)
        det = self.detector.detect(img)

        defect_rate = float(det.defect_area_ratio) if label != "clean" else max(0.0, det.defect_area_ratio)
        # blend the *predicted* defect fraction from the process model with the
        # measured area ratio so APC sees a smooth signal.
        measured = 0.5 * self.apc.defect_rate(self.speed, self.steam, self.chem, self.consistency) \
            + 0.5 * defect_rate
        quality = det.quality_score

        state = self._process_state(measured, quality)
        action = self.apc.step(state)

        # apply setpoints with first-order lag:  x += (target - x) * (1 - exp(-dt/tau))
        alpha = 1.0 - float(np.exp(-self.dt / self.actuator_tau))
        self.speed += (action.setpoints["speed"] - self.speed) * alpha
        self.steam += (action.setpoints["steam_pressure"] - self.steam) * alpha
        self.chem += (action.setpoints["chem_dosage"] - self.chem) * alpha

        # cumulative savings
        self.cum.frames += 1
        if det.pred_class and det.pred_class != "clean":
            self.cum.defect_frames += 1
        self.cum.quality_sum += quality
        self.cum.defect_rate_sum += measured
        self.cum.tonnes_saved += action.tonnes_saved_per_h * (self.dt / 3600.0)
        self.cum.energy_saved_gj += action.energy_saved_gj_per_h * (self.dt / 3600.0)
        self.cum.co2_saved_kg += action.co2_saved_kg_per_h * (self.dt / 3600.0)

        # encode frames
        frame_b64 = self._jpeg_b64(img)
        overlay = PaperDefectDetector.draw_overlay(img, det)
        overlay_b64 = self._jpeg_b64(overlay)

        return {
            "t": round(self.t, 2),
            "label_truth": label,
            "frame_jpeg_b64": frame_b64,
            "overlay_jpeg_b64": overlay_b64,
            "detection": det.to_dict(),
            "state": state.to_dict(),
            "apc": action.to_dict(),
            "cumulative": self.cum.to_dict(),
        }

    # -- helpers --------------------------------------------------------------
    def _jpeg_b64(self, img: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")
