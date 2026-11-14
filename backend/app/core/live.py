"""Live engine: a shared paper-machine simulator that broadcasts ticks to all
connected dashboard clients over WebSocket.

A single simulator instance runs at a fixed cadence; every connected WebSocket
receives the same stream so multiple operator views stay in sync. The engine is
asyncio-friendly (the synchronous CV/ML work per tick is a few ms, well within
the 3 fps cadence).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

from ..config import SIM_DT, SIM_FPS, SIM_JPEG_QUALITY
from .apc import APCController
from .classifier import DefectClassifier
from .detector import PaperDefectDetector
from .simulator import PaperMachineSimulator

log = logging.getLogger("bistun.live")


class LiveEngine:
    def __init__(self, fps: float = SIM_FPS, dt: float = SIM_DT,
                 jpeg_quality: int = SIM_JPEG_QUALITY) -> None:
        self.fps = fps
        self.dt = dt
        self.jpeg_quality = jpeg_quality
        self.detector = PaperDefectDetector()
        self.apc = APCController()
        self._load_classifier()
        self.sim = PaperMachineSimulator(
            detector=self.detector, apc=self.apc, dt=dt,
            jpeg_quality=jpeg_quality)
        self.clients: set[WebSocket] = set()
        self._task: asyncio.Task | None = None
        self.paused = False
        self.last_event: dict[str, Any] | None = None

    def _load_classifier(self) -> None:
        from ..config import MODEL_PATH
        if MODEL_PATH.exists():
            try:
                clf = DefectClassifier.load(MODEL_PATH)
                self.detector.attach_classifier(clf)
                log.info("classifier loaded from %s", MODEL_PATH)
            except Exception as exc:  # pragma: no cover
                log.warning("classifier load failed: %s", exc)
        else:
            log.warning("no model at %s -- detector runs without classifier", MODEL_PATH)

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="live-engine")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    # -- clients -------------------------------------------------------------
    async def add_client(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        self.start()
        # send the last event immediately so the dashboard isn't blank
        if self.last_event:
            try:
                await ws.send_text(json.dumps(self.last_event))
            except Exception:
                pass

    def remove_client(self, ws: WebSocket) -> None:
        self.clients.discard(ws)
        if not self.clients:
            # no viewers -> stop the loop (saves CPU); restarted on next connect
            if self._task and not self._task.done():
                self._task.cancel()

    async def control(self, msg: dict[str, Any]) -> None:
        action = msg.get("action")
        if action == "pause":
            self.paused = True
        elif action == "resume":
            self.paused = False
        elif action == "reset":
            self.sim.reset()
            self.paused = False
        elif action == "seed":
            try:
                seed = int(msg.get("seed", 0))
                self.sim = PaperMachineSimulator(
                    detector=self.detector, apc=self.apc, dt=self.dt,
                    jpeg_quality=self.jpeg_quality, seed=seed)
            except Exception:
                pass

    # -- main loop -----------------------------------------------------------
    async def _run(self) -> None:
        interval = 1.0 / self.fps
        while True:
            try:
                if not self.paused:
                    # run the (synchronous) sim tick; it is fast enough to not
                    # stall the loop at the configured fps.
                    event = await asyncio.to_thread(self.sim.tick)
                    self.last_event = event
                    await self._broadcast(event)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # keep the loop alive
                log.exception("live tick failed: %s", exc)
                await asyncio.sleep(interval)

    async def _broadcast(self, event: dict[str, Any]) -> None:
        if not self.clients:
            return
        payload = json.dumps(event)
        dead: list[WebSocket] = []
        for ws in self.clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


# module-level singleton
engine = LiveEngine()
