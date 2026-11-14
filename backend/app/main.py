"""Bistun Paper Industries -- Integrated Quality & Energy Control (MV + APC).

FastAPI application entry point.

Run:
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                      -> live monitoring dashboard (self-contained HTML)
    GET  /health                -> service + model status
    GET  /classes               -> defect classes (EN + FA)
    GET  /metrics               -> trained-model metrics
    POST /detect                -> base64 image -> detection result (+ optional overlay)
    POST /detect/file           -> multipart image -> detection result
    POST /apc                   -> process state -> APC control action + savings
    GET  /engine/state          -> live simulator snapshot
    POST /engine/control        -> pause/resume/reset/seed the live engine
    WS   /ws/live               -> live paper-machine stream (frame + detection + APC)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router as rest_router
from .api.ws import router as ws_router
from .config import CORS_ORIGINS, DASHBOARD_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .core.live import engine
    engine.start()
    try:
        yield
    finally:
        await engine.stop()


app = FastAPI(
    title="Bistun Paper Industries -- Integrated Quality & Energy Control",
    description=(
        "Machine-vision surface-defect detection + MPC-style Advanced Process "
        "Control for energy/emission optimisation on a paper machine. TRL 5 "
        "prototype (relevant-environment simulation)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rest_router, prefix="/api")
app.include_router(ws_router)


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>dashboard/index.html not found</h1>", status_code=404)


# expose any other static assets (e.g. sample images) under /static
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


@app.get("/favicon.ico")
async def favicon() -> dict:
    return {"ok": True}
