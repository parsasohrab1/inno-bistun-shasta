"""WebSocket endpoint -- live paper-machine monitoring stream."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core.live import engine

log = logging.getLogger("bistun.ws")
router = APIRouter()


@router.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """Stream live simulator ticks to the dashboard.

    Client -> server messages (JSON, optional):
        {"action": "pause" | "resume" | "reset" | "seed", "seed": int}
    """
    await engine.add_client(ws)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            await engine.control(msg)
            if msg.get("action") == "state":
                await ws.send_text(json.dumps({
                    "type": "state",
                    "running": engine._task is not None and not engine._task.done(),
                    "paused": engine.paused,
                    "n_clients": len(engine.clients),
                }))
    except WebSocketDisconnect:
        pass
    finally:
        engine.remove_client(ws)
