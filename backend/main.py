from __future__ import annotations
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from backend.agents import Agent
from backend.characters import CHARACTERS
from backend.world import WorldState, TickScheduler

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Global state
connected_clients: set[WebSocket] = set()
scheduler: TickScheduler | None = None
scheduler_task: asyncio.Task | None = None


async def broadcast(event: dict) -> None:
    if not connected_clients:
        return
    message = json.dumps(event, ensure_ascii=False)
    dead: set[WebSocket] = set()
    for ws in list(connected_clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    connected_clients.difference_update(dead)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler, scheduler_task
    agents = [Agent(cfg) for cfg in CHARACTERS]
    world = WorldState(CHARACTERS)
    scheduler = TickScheduler(agents, world, broadcast)
    scheduler_task = asyncio.create_task(scheduler.run())
    logger.info("Simulation started — 60 ticks until 23:00")
    yield
    if scheduler_task and not scheduler_task.done():
        scheduler_task.cancel()
    logger.info("Simulation stopped")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"Client connected ({len(connected_clients)} total)")

    # Send current state to newly connected client
    if scheduler:
        await websocket.send_text(json.dumps({
            "type": "world_update",
            "sim_time": scheduler.clock.current,
            "positions": dict(scheduler.world.agent_rooms),
            "moods": dict(scheduler.world.agent_moods),
        }, ensure_ascii=False))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "force_tick" and scheduler:
                    scheduler.force_tick()
                    logger.info("Force tick triggered by client")
                elif msg.get("type") == "pause" and scheduler:
                    scheduler.pause()
                    logger.info("Simulation paused")
                    await websocket.send_text(json.dumps({"type": "paused"}, ensure_ascii=False))
                elif msg.get("type") == "resume" and scheduler:
                    scheduler.resume()
                    logger.info("Simulation resumed")
                    await websocket.send_text(json.dumps({"type": "resumed"}, ensure_ascii=False))
            except Exception:
                pass
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info(f"Client disconnected ({len(connected_clients)} total)")
