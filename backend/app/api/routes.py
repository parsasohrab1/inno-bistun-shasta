"""REST API routes for inference, APC, and system metadata."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..config import METRICS_PATH, MODEL_PATH
from ..core.apc import APCController, ProcessState
from ..core.classifier import DefectClassifier
from ..core.detector import PaperDefectDetector
from ..core.paper_render import CLASSES, LABELS
from ..core.simulator import PaperMachineSimulator

router = APIRouter()

# A detector + classifier pair for synchronous REST inference (separate from the
# live engine so /detect never interferes with the live stream).
_rest_detector = PaperDefectDetector()
if MODEL_PATH.exists():
    try:
        _rest_detector.attach_classifier(DefectClassifier.load(MODEL_PATH))
    except Exception:
        pass
_apc = APCController()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "bistun-inno-shasta",
        "model_loaded": MODEL_PATH.exists(),
        "classes": CLASSES,
    }


@router.get("/classes")
def classes() -> dict:
    return {
        "classes": [
            {"id": i, "name": c, "label_en": LABELS[c]["en"], "label_fa": LABELS[c]["fa"]}
            for i, c in enumerate(CLASSES)
        ]
    }


@router.get("/metrics")
def metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {"detail": "metrics not found -- run scripts/train_classifier.py"}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
class DetectRequest(BaseModel):
    image_b64: str = Field(..., description="base64-encoded JPEG/PNG of a paper-web frame")
    return_overlay: bool = False


@router.post("/detect")
async def detect(req: DetectRequest) -> dict:
    try:
        raw = base64.b64decode(req.image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid base64 image")
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="image could not be decoded")
    result = _rest_detector.detect(img)
    out = result.to_dict()
    out["truth"] = None
    if req.return_overlay:
        overlay = PaperDefectDetector.draw_overlay(img, result)
        ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out["overlay_b64"] = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    return out


@router.post("/detect/file")
async def detect_file(file: UploadFile = File(...), overlay: bool = False) -> dict:
    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="image could not be decoded")
    result = _rest_detector.detect(img)
    out = result.to_dict()
    if overlay:
        ov = PaperDefectDetector.draw_overlay(img, result)
        ok, buf = cv2.imencode(".jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out["overlay_b64"] = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    return out


# ---------------------------------------------------------------------------
# APC
# ---------------------------------------------------------------------------
class APCRequest(BaseModel):
    speed: float
    steam_pressure: float
    chem_dosage: float
    consistency: float
    defect_rate: float = 0.0
    quality_score: float = 1.0
    moisture: float = 0.0


@router.post("/apc")
def apc_step(req: APCRequest) -> dict:
    state = ProcessState(
        speed=req.speed, steam_pressure=req.steam_pressure, chem_dosage=req.chem_dosage,
        consistency=req.consistency, defect_rate=req.defect_rate,
        quality_score=req.quality_score, moisture=req.moisture,
    )
    action = _apc.step(state)
    return action.to_dict()


# ---------------------------------------------------------------------------
# Engine snapshot
# ---------------------------------------------------------------------------
@router.get("/engine/state")
def engine_state() -> dict:
    from ..core.live import engine
    sim = engine.sim
    return {
        "running": engine._task is not None and not engine._task.done(),
        "paused": engine.paused,
        "clients": len(engine.clients),
        "n_clients": len(engine.clients),
        "sim_t": sim.t,
        "cumulative": sim.cum.to_dict(),
        "state": {
            "speed": sim.speed, "steam_pressure": sim.steam,
            "chem_dosage": sim.chem, "consistency": sim.consistency,
        },
    }


@router.post("/engine/control")
async def engine_control(action: str, seed: int | None = None) -> dict:
    from ..core.live import engine
    await engine.control({"action": action, "seed": seed})
    return {"ok": True, "action": action}
