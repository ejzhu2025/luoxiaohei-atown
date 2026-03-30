from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable, Coroutine

from backend.characters import CharacterConfig
from backend.agents import Agent
from backend.pathfinding import find_path

logger = logging.getLogger(__name__)

# Initial states of interactable objects in each room
INITIAL_OBJECT_STATES: dict[str, dict[str, str]] = {
    "厨房":   {"灶台": "空闲", "红烧肉": "未开始炖", "饺子": "未包"},
    "餐厅":   {"火锅": "未点火", "酒柜": "关闭"},
    "游戏室": {"双显示器": "开启", "VR头显": "待机", "Switch": "关闭"},
    "客厅":   {"PS5": "运行中-赛车游戏", "TV": "开启"},
    "次阳台": {"灯笼": "亮着", "藤椅": "空置"},
    "院子":   {"灯笼": "亮着", "鞭炮": "未点燃"},
    "衣帽间": {"全身镜": "正常", "战袍": "展示中"},
    "卧室":   {"台灯": "关闭", "智能面板": "待机"},
    "卫生间": {"浴缸": "空"},
}


class SimClock:
    def __init__(self, start_hour: int = 18, start_minute: int = 0):
        self._total_minutes = start_hour * 60 + start_minute
        self._end_minutes = 23 * 60

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
        import copy
        self.object_states: dict[str, dict[str, str]] = copy.deepcopy(INITIAL_OBJECT_STATES)

    def get_room_occupants(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for name, room in self.agent_rooms.items():
            result.setdefault(room, []).append(name)
        return result

    def update_agent_room(self, name: str, room: str) -> None:
        self.agent_rooms[name] = room

    def update_object_state(self, room: str, obj: str, state: str) -> None:
        if room in self.object_states:
            self.object_states[room][obj] = state

    def parse_and_apply_object_interaction(self, room: str, interaction: str) -> tuple[str, str] | None:
        """Parse '道具名:新状态' string and apply it. Returns (obj, state) or None."""
        if not interaction or ":" not in interaction:
            return None
        parts = interaction.split(":", 1)
        obj, state = parts[0].strip(), parts[1].strip()
        if obj and state:
            self.object_states.setdefault(room, {})[obj] = state
            return obj, state
        return None

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
            "object_states": self.object_states,
        }

    def apply_decision(self, name: str, decision: dict[str, Any]) -> list[str]:
        target = decision.get("目标房间", self.agent_rooms[name])
        self.agent_moods[name] = decision.get("情绪", self.agent_moods[name])
        self.agent_dialogues[name] = decision.get("对话", "")
        self.agent_actions[name] = decision.get("动作", "")

        # Object interaction
        interaction = decision.get("道具交互", "")
        if interaction:
            result = self.parse_and_apply_object_interaction(self.agent_rooms[name], interaction)
            if result:
                logger.info(f"{name} changed {result[0]} → {result[1]}")

        current = self.agent_rooms[name]
        path = find_path(current, target)
        if len(path) > 1:
            self.agent_rooms[name] = path[1]
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
        self._force_tick = asyncio.Event()
        self._agent_map: dict[str, Agent] = {a.name: a for a in agents}

    async def run_planning_phase(self) -> None:
        """Call each agent's plan_evening() concurrently before ticks start."""
        logger.info("Planning phase: agents generating evening plans...")
        await self.broadcast({"type": "planning_start"})
        results = await asyncio.gather(
            *[agent.plan_evening() for agent in self.agents],
            return_exceptions=True,
        )
        for agent, result in zip(self.agents, results):
            if isinstance(result, Exception):
                logger.error(f"{agent.name} planning failed: {result}")
            else:
                plan_str = " / ".join(agent.plan[:3]) if agent.plan else "无计划"
                logger.info(f"{agent.name} plan: {plan_str}")
                await self.broadcast({
                    "type": "agent_plan",
                    "name": agent.name,
                    "plan": agent.plan,
                })

    async def run(self) -> None:
        self._running = True

        # Phase 0: Planning
        await self.run_planning_phase()

        tick_num = 0
        while self._running and not self.clock.is_over():
            tick_num += 1
            sim_time = self.clock.current
            logger.info(f"Tick {tick_num} | sim_time={sim_time}")

            await self.broadcast({"type": "tick_start", "tick": tick_num, "sim_time": sim_time})

            world_context = self.world.build_world_context(sim_time)

            # ── Main decision round (all agents concurrent) ──────────────────
            decisions = await asyncio.gather(
                *[agent.think(world_context, tick_num) for agent in self.agents],
                return_exceptions=True,
            )

            # Apply decisions
            spoke_in_room: dict[str, tuple[str, str]] = {}  # room → (speaker_name, dialogue)
            tick_events = []
            for agent, decision in zip(self.agents, decisions):
                if isinstance(decision, Exception):
                    logger.error(f"{agent.name} think() failed: {decision}")
                    continue

                path = self.world.apply_decision(agent.name, decision)
                new_room = self.world.agent_rooms[agent.name]

                dialogue = decision.get("对话", "")
                # One speaker per room per tick
                if dialogue and new_room in spoke_in_room:
                    dialogue = ""
                elif dialogue:
                    spoke_in_room[new_room] = (agent.name, dialogue)

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
                    "obj_interaction": decision.get("道具交互", ""),
                })

                # Push memories
                if decision.get("动作"):
                    ev = f"{agent.name}在{new_room}：{decision['动作']}"
                    for other in self.agents:
                        if other.current_room == new_room:
                            other.add_memory(ev, importance=5, sim_time=sim_time, tick_num=tick_num)
                if dialogue:
                    ev = f"{agent.name}说：「{dialogue}」"
                    for other in self.agents:
                        if other.current_room == new_room:
                            other.add_memory(ev, importance=7, sim_time=sim_time, tick_num=tick_num)

                agent.current_room = new_room

            # Broadcast main events
            for event in tick_events:
                await self.broadcast(event)

            # ── Dialogue reaction sub-round ──────────────────────────────────
            # For each room where someone spoke, other co-located agents react immediately
            if spoke_in_room:
                reaction_tasks = []
                for room, (speaker_name, dialogue) in spoke_in_room.items():
                    listeners = [
                        a for a in self.agents
                        if a.current_room == room and a.name != speaker_name
                    ]
                    for listener in listeners:
                        reaction_tasks.append((listener, speaker_name, dialogue))

                if reaction_tasks:
                    reactions = await asyncio.gather(
                        *[a.react_to_dialogue(spk, dlg, sim_time) for a, spk, dlg in reaction_tasks],
                        return_exceptions=True,
                    )
                    for (listener, speaker_name, _), reaction in zip(reaction_tasks, reactions):
                        if isinstance(reaction, Exception) or reaction is None:
                            continue
                        response = reaction.get("回应", "")
                        action = reaction.get("动作", "")
                        mood = reaction.get("情绪", listener.mood)
                        if mood:
                            listener.mood = mood
                            self.world.agent_moods[listener.name] = mood
                        if response:
                            ev = f"{listener.name}回应：「{response}」"
                            for other in self.agents:
                                if other.current_room == listener.current_room:
                                    other.add_memory(ev, importance=7, sim_time=sim_time, tick_num=tick_num)
                        await self.broadcast({
                            "type": "agent_reaction",
                            "name": listener.name,
                            "color": listener.color,
                            "room": listener.current_room,
                            "dialogue": response,
                            "action": action,
                            "mood": mood,
                            "in_reply_to": speaker_name,
                        })

            # ── Reflection check ──────────────────────────────────────────────
            reflecting = [a for a in self.agents if a.should_reflect()]
            if reflecting:
                reflect_results = await asyncio.gather(
                    *[a.reflect(sim_time, tick_num) for a in reflecting],
                    return_exceptions=True,
                )
                for agent, insights in zip(reflecting, reflect_results):
                    if isinstance(insights, list) and insights:
                        logger.info(f"{agent.name} reflected: {insights[0]}")
                        await self.broadcast({
                            "type": "agent_reflection",
                            "name": agent.name,
                            "insights": insights,
                            "sim_time": sim_time,
                        })

            # Advance clock
            self.clock.advance(self.TICK_SIM_MINUTES)
            await self.broadcast({
                "type": "world_update",
                "sim_time": self.clock.current,
                "positions": dict(self.world.agent_rooms),
                "moods": dict(self.world.agent_moods),
                "object_states": self.world.object_states,
            })

            # Wait for timer or force-tick
            self._force_tick.clear()
            try:
                await asyncio.wait_for(self._force_tick.wait(), timeout=self.TICK_REAL_SECONDS)
            except asyncio.TimeoutError:
                pass

        await self.broadcast({"type": "simulation_end", "sim_time": self.clock.current})
        self._running = False

    def force_tick(self) -> None:
        self._force_tick.set()

    def stop(self) -> None:
        self._running = False
