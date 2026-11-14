"""API smoke tests via FastAPI TestClient."""
import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.core.paper_render import render_sample
from backend.app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert "classes" in j


def test_classes():
    r = client.get("/api/classes")
    assert r.status_code == 200
    assert len(r.json()["classes"]) == 6


def test_metrics():
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert "test_accuracy" in r.json() or "detail" in r.json()


def test_detect_endpoint():
    img, _ = render_sample("stain", 256, 256, rng=np.random.default_rng(1))
    ok, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode()
    r = client.post("/api/detect", json={"image_b64": b64, "return_overlay": True})
    assert r.status_code == 200
    d = r.json()
    assert "defects" in d and "quality_score" in d
    assert d["overlay_b64"]


def test_apc_endpoint():
    r = client.post("/api/apc", json={
        "speed": 820, "steam_pressure": 3.6, "chem_dosage": 1.6,
        "consistency": 4.5, "defect_rate": 0.10, "quality_score": 0.7})
    assert r.status_code == 200
    a = r.json()
    assert "setpoints" in a and "rationale" in a
    assert a["defect_rate_reduction"] >= 0


def test_engine_state():
    r = client.get("/api/engine/state")
    assert r.status_code == 200
    assert "cumulative" in r.json()


def test_dashboard_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Bistun Paper Industries" in r.text
