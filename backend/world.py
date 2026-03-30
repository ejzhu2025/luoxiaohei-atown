from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable, Coroutine

from backend.characters import CharacterConfig
from backend.agents import Agent
from backend.pathfinding import find_path

logger = logging.getLogger(__name__)


class SimClock:
    def __init__(self, start_hour: int = 18, start_minute: int = 0):
        self._total_minutes = start_hour * 60 + start_minute
        self._end_minutes = 23 * 60  # 23:00

    @property
    def current(self) -> str:
        h, m = divmod(self._total_minutes, 60)
        return f"{h:02d}:{m:02d}"

    def advance(self, minutes: int) -> None:
        self._total_minutes += minutes

    def is_over(self) -> bool:
        return self._total_minutes >= self._end_minutes


class WorldState:
    def __init__(self, characters: list[CharacterConfig]):
        self.agent_rooms: dict[str, str] = {c.name: c.initial_room for c in characters}
        self.agent_paths: dict[str, list[str]] = {c.name: [c.initial_room] for c in characters}
        self.agent_moods: dict[str, str] = {c.name: c.initial_mood for c in characters}
        self.agent_dialogues: dict[str, str] = {c.name: "" for c in characters}
        self.agent_actions: dict[str, str] = {c.name: c.initial_goal for c in characters}

    def get_room_occupants(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for name, room in self.agent_rooms.items():
            result.setdefault(room, []).append(name)
        return result

    def update_agent_room(self, name: str, room: str) -> None:
        self.agent_rooms[name] = room

    def build_summary(self) -> str:
        lines = ["当前各人状态："]
        for name, room in self.agent_rooms.items():
            mood = self.agent_moods.get(name, "")
            action = self.agent_actions.get(name, "")
            lines.append(f"  - {name} 在{room}，心情{mood}，正在{action}")
        return "\n".join(lines)

    def build_world_context(self, sim_time: str) -> dict[str, Any]:
        return {
            "sim_time": sim_time,
            "room_occupants": self.get_room_occupants(),
            "world_summary": self.build_summary(),
        }

    def apply_decision(self, name: str, decision: dict[str, Any]) -> list[str]:
        """Apply agent decision to world state. Returns full intended path."""
        target = decision.get("目标房间", self.agent_rooms[name])
        mood = decision.get("情绪", self.agent_moods[name])
        dialogue = decision.get("对话", "")
        action = decision.get("动作", "")

        self.agent_moods[name] = mood
        self.agent_dialogues[name] = dialogue
        self.agent_actions[name] = action

        current = self.agent_rooms[name]
        path = find_path(current, target)
        if len(path) > 1:
            # Move one room at a time per tick
            next_room = path[1]
            self.agent_rooms[name] = next_room
            self.agent_paths[name] = path
        else:
            self.agent_paths[name] = [current]
        return path


class TickScheduler:
    TICK_REAL_SECONDS = 4
    TICK_SIM_MINUTES = 5

    def __init__(
        self,
        agents: list[Agent],
        world: WorldState,
        broadcast: Callable[[dict], Coroutine],
    ):
        self.agents = agents
        self.world = world
        self.broadcast = broadcast
        self.clock = SimClock()
        self._running = False

    async def run(self) -> None:
        self._running = True
        tick_num = 0
        while self._running and not self.clock.is_over():
            tick_num += 1
            sim_time = self.clock.current
            logger.info(f"Tick {tick_num} | sim_time={sim_time}")

            await self.broadcast({"type": "tick_start", "tick": tick_num, "sim_time": sim_time})

            # Build shared world context once per tick
            world_context = self.world.build_world_context(sim_time)

            # All agents think concurrently
            decisions = await asyncio.gather(
                *[agent.think(world_context) for agent in self.agents],
                return_exceptions=True,
            )

            # Apply decisions and collect events
            tick_events = []
            spoke_in_room: set[str] = set()  # rooms where someone already spoke this tick
            for agent, decision in zip(self.agents, decisions):
                if isinstance(decision, Exception):
                    logger.error(f"{agent.name} think() failed: {decision}")
                    continue
                path = self.world.apply_decision(agent.name, decision)
                new_room = self.world.agent_rooms[agent.name]
                # Only one speaker per room per tick
                dialogue = decision.get("对话", "")
                if dialogue and new_room in spoke_in_room:
                    dialogue = ""  # silence this one — someone already spoke here
                elif dialogue:
                    spoke_in_room.add(new_room)
                tick_events.append({
                    "type": "agent_action",
                    "name": agent.name,
                    "color": agent.color,
                    "room": new_room,
                    "path": path,
                    "action": decision.get("动作", ""),
                    "dialogue": dialogue,
                    "mood": decision.get("情绪", ""),
                    "thought": decision.get("思考", ""),
                })
                # Push memories to all agents in same room
                if decision.get("动作"):
                    event_text = f"{agent.name}在{new_room}：{decision['动作']}"
                    for other in self.agents:
                        if other.current_room == new_room:
                            other.add_memory(event_text, importance=5, sim_time=sim_time)
                if decision.get("对话"):
                    dialogue_text = f"{agent.name}说：「{decision['对话']}」"
                    for other in self.agents:
                        if other.current_room == new_room:
                            other.add_memory(dialogue_text, importance=7, sim_time=sim_time)
                # Sync agent's own room reference
                agent.current_room = new_room

            # Broadcast all agent events
            for event in tick_events:
                await self.broadcast(event)

            # Advance sim clock
            self.clock.advance(self.TICK_SIM_MINUTES)
            await self.broadcast({
                "type": "world_update",
                "sim_time": self.clock.current,
                "positions": dict(self.world.agent_rooms),
                "moods": dict(self.world.agent_moods),
            })

            await asyncio.sleep(self.TICK_REAL_SECONDS)

        await self.broadcast({"type": "simulation_end", "sim_time": self.clock.current})
        self._running = False

    def stop(self) -> None:
        self._running = False
