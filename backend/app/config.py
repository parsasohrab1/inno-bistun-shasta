"""Backend configuration (paths + tunables)."""
from __future__ import annotations

from pathlib import Path

# Repo root = parent of backend/
ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "classifier.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DASHBOARD_DIR = ROOT / "dashboard"

# Live simulator defaults
SIM_FPS = 3.0            # frames pushed to the dashboard per second
SIM_DT = 1.0             # simulated process seconds per tick
SIM_JPEG_QUALITY = 70

# CORS -- allow the static dashboard (same origin) and local dev origins
CORS_ORIGINS = ["*",]
