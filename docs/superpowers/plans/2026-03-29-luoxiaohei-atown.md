# 罗小黑 AI Town Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Stanford AI Town–style simulation where 小黑/无限/哪吒/鹿野 autonomously navigate Nezha's apartment on New Year's Eve, driven by Claude LLM every 4 seconds, visualized on an HTML Canvas with speech bubbles and a side log panel.

**Architecture:** FastAPI backend runs a tick scheduler (asyncio) that calls claude-sonnet-4-6 for all 4 agents concurrently each tick, updates WorldState, and broadcasts the result over WebSocket. A single `frontend/index.html` receives events, animates character sprites on Canvas (mirroring the floorplan-v5 SVG layout), and renders speech bubbles + scrolling log.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, anthropic SDK (`>=0.40`), websockets, pytest · HTML5 Canvas 2D, vanilla JS, WebSocket API

---

## File Map

| File | Responsibility |
|------|----------------|
| `backend/pathfinding.py` | Room graph + A\* shortest path |
| `backend/characters.py` | Static config: name, color, personality, initial position/mood |
| `backend/agents.py` | `Agent` class: memory stream, prompt builder, LLM call, memory update |
| `backend/world.py` | `WorldState` (positions, emotions, sim clock) + `TickScheduler` (asyncio loop) |
| `backend/main.py` | FastAPI app, `/ws` WebSocket endpoint, startup |
| `frontend/index.html` | Canvas renderer, WebSocket client, speech bubbles, log panel |
| `tests/test_pathfinding.py` | A\* unit tests |
| `tests/test_agents.py` | Prompt builder + memory update tests (no real LLM calls) |
| `tests/test_world.py` | WorldState mutation tests |
| `requirements.txt` | Python dependencies |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `backend/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.115.6
uvicorn[standard]==0.32.1
anthropic>=0.40.0
python-dotenv==1.0.1
websockets==13.1
pytest==8.3.4
pytest-asyncio==0.24.0
```

- [ ] **Step 2: Install dependencies**

```bash
cd /Users/bytedance/luoxiaohei-atown
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: All packages installed, no errors.

- [ ] **Step 3: Create empty init files**

```bash
touch backend/__init__.py tests/__init__.py
```

- [ ] **Step 4: Create .env file**

```bash
echo "ANTHROPIC_API_KEY=your_key_here" > .env
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".venv/" >> .gitignore
```

- [ ] **Step 5: Commit**

```bash
cd /Users/bytedance/luoxiaohei-atown
git init
git add requirements.txt backend/__init__.py tests/__init__.py .gitignore
git commit -m "Project scaffolding: deps and structure"
```

---

## Task 2: Pathfinding Module

**Files:**
- Create: `backend/pathfinding.py`
- Create: `tests/test_pathfinding.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pathfinding.py`:

```python
import pytest
from backend.pathfinding import find_path, ROOMS

def test_same_room_returns_single_element():
    path = find_path("客厅", "客厅")
    assert path == ["客厅"]

def test_adjacent_rooms_direct_path():
    path = find_path("客厅", "次阳台")
    assert path == ["客厅", "次阳台"]

def test_multi_hop_path():
    # 卫生间 → 衣帽间: 卫生间→卧室→衣帽间
    path = find_path("卫生间", "衣帽间")
    assert path == ["卫生间", "卧室", "衣帽间"]

def test_long_path():
    # 卫生间 → 院子: 卫生间→卧室→客厅→餐厅→院子
    path = find_path("卫生间", "院子")
    assert len(path) <= 6
    assert path[0] == "卫生间"
    assert path[-1] == "院子"

def test_invalid_room_raises():
    with pytest.raises(ValueError):
        find_path("不存在的房间", "客厅")

def test_all_rooms_reachable():
    for src in ROOMS:
        for dst in ROOMS:
            path = find_path(src, dst)
            assert path[0] == src
            assert path[-1] == dst
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/bytedance/luoxiaohei-atown
source .venv/bin/activate
pytest tests/test_pathfinding.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — pathfinding.py doesn't exist yet.

- [ ] **Step 3: Implement pathfinding.py**

Create `backend/pathfinding.py`:

```python
from __future__ import annotations
import heapq

# Room graph: undirected edges with uniform weight 1
GRAPH: dict[str, list[str]] = {
    "院子":   ["厨房", "餐厅", "游戏室"],
    "厨房":   ["院子", "餐厅", "衣帽间"],
    "餐厅":   ["院子", "厨房", "游戏室", "客厅"],
    "游戏室": ["院子", "餐厅", "次阳台"],
    "衣帽间": ["厨房", "客厅", "卧室"],
    "客厅":   ["餐厅", "衣帽间", "次阳台", "卧室"],
    "次阳台": ["游戏室", "客厅"],
    "卧室":   ["衣帽间", "客厅", "卫生间"],
    "卫生间": ["卧室"],
}

ROOMS = set(GRAPH.keys())


def find_path(start: str, goal: str) -> list[str]:
    """Return shortest room path from start to goal using A* (uniform cost here)."""
    if start not in GRAPH:
        raise ValueError(f"Unknown room: {start!r}")
    if goal not in GRAPH:
        raise ValueError(f"Unknown room: {goal!r}")
    if start == goal:
        return [start]

    # (cost, current, path)
    heap: list[tuple[int, str, list[str]]] = [(0, start, [start])]
    visited: set[str] = set()

    while heap:
        cost, current, path = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return path
        for neighbor in GRAPH[current]:
            if neighbor not in visited:
                heapq.heappush(heap, (cost + 1, neighbor, path + [neighbor]))

    raise RuntimeError(f"No path from {start!r} to {goal!r}")  # should never happen
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pathfinding.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/pathfinding.py tests/test_pathfinding.py
git commit -m "Add A* pathfinding module with room graph"
```

---

## Task 3: Character Configs

**Files:**
- Create: `backend/characters.py`

No separate test needed — this is pure static data validated by downstream tests.

- [ ] **Step 1: Create characters.py**

```python
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class CharacterConfig:
    name: str
    emoji: str          # display label prefix
    color: str          # CSS hex for canvas sprite
    personality: str    # fed verbatim into system prompt
    initial_room: str
    initial_mood: str
    initial_goal: str


CHARACTERS: list[CharacterConfig] = [
    CharacterConfig(
        name="小黑",
        emoji="🐱",
        color="#4a90d9",
        personality=(
            "你是小黑，一只深蓝色的小猫，化身为少年。性格好奇、安静、敏感。"
            "喜欢盯着新鲜事物发呆，不太主动说话，但内心充满感受。"
            "和无限关系最亲密，对哪吒既好奇又略微警惕，对鹿野感到平静和安心。"
            "说话简短，偶尔冒出猫咪式的短语。"
        ),
        initial_room="餐厅",
        initial_mood="期待",
        initial_goal="盯着圆桌上的年夜饭食材发呆",
    ),
    CharacterConfig(
        name="无限",
        emoji="🌙",
        color="#9b59b6",
        personality=(
            "你是无限，精灵族首领，温柔体贴，习惯照顾身边的人。"
            "喜欢烹饪和为大家创造舒适的环境。会主动关心每个人的状态。"
            "对小黑有保护欲，对哪吒有些无奈但包容，对鹿野互相尊重。"
            "说话温和有礼，措辞细腻。"
        ),
        initial_room="厨房",
        initial_mood="专注",
        initial_goal="准备年夜饭，把红烧肉炖好",
    ),
    CharacterConfig(
        name="哪吒",
        emoji="🔱",
        color="#e74c3c",
        personality=(
            "你是哪吒，天界战神，随性爱玩，嘴硬心软。"
            "喜欢打游戏、吃零食，表面上不在乎任何人但其实很享受大家聚在一起。"
            "会主动挑起话题或恶作剧，但一旦有人需要帮助立刻行动。"
            "说话直接，夹杂口头禅，偶尔中二。"
        ),
        initial_room="客厅",
        initial_mood="放松",
        initial_goal="打PS5，等饭好了再去餐厅",
    ),
    CharacterConfig(
        name="鹿野",
        emoji="🦌",
        color="#27ae60",
        personality=(
            "你是鹿野，自然系精灵，沉稳内敛，话不多但每句话都有分量。"
            "喜欢安静地观察环境和夜景，偶尔说出令人回味的句子。"
            "对小黑有些许长辈式的温柔，对无限互相欣赏，对哪吒有耐心但不迁就。"
            "说话简洁，有时带着自然意象的比喻。"
        ),
        initial_room="次阳台",
        initial_mood="平静",
        initial_goal="看城市夜景，感受除夕夜的气氛",
    ),
]

CHARACTER_MAP: dict[str, CharacterConfig] = {c.name: c for c in CHARACTERS}
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/bytedance/luoxiaohei-atown
source .venv/bin/activate
python3.11 -c "from backend.characters import CHARACTERS; print([c.name for c in CHARACTERS])"
```

Expected: `['小黑', '无限', '哪吒', '鹿野']`

- [ ] **Step 3: Commit**

```bash
git add backend/characters.py
git commit -m "Add character configs for all four agents"
```

---

## Task 4: Agent Class with Memory Stream

**Files:**
- Create: `backend/agents.py`
- Create: `tests/test_agents.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agents.py`:

```python
import pytest
from unittest.mock import MagicMock
from backend.agents import Agent, MemoryEntry
from backend.characters import CHARACTERS


def make_agent(name="小黑") -> Agent:
    cfg = next(c for c in CHARACTERS if c.name == name)
    return Agent(cfg)


def test_agent_initial_state():
    agent = make_agent("小黑")
    assert agent.name == "小黑"
    assert agent.current_room == "餐厅"
    assert agent.mood == "期待"
    assert agent.memories == []


def test_add_memory_stores_entry():
    agent = make_agent("小黑")
    agent.add_memory("无限做好了红烧肉", importance=8, sim_time="18:30")
    assert len(agent.memories) == 1
    assert agent.memories[0].event == "无限做好了红烧肉"
    assert agent.memories[0].importance == 8


def test_memory_capped_at_20():
    agent = make_agent("小黑")
    for i in range(25):
        agent.add_memory(f"事件{i}", importance=i % 10, sim_time="18:00")
    assert len(agent.memories) == 20


def test_memory_evicts_lowest_importance():
    agent = make_agent("小黑")
    # Fill with importance=5
    for i in range(20):
        agent.add_memory(f"事件{i}", importance=5, sim_time="18:00")
    # Add one with importance=9 — should evict one importance=5
    agent.add_memory("重要事件", importance=9, sim_time="18:05")
    assert len(agent.memories) == 20
    importances = [m.importance for m in agent.memories]
    assert 9 in importances


def test_build_prompt_contains_key_fields():
    agent = make_agent("无限")
    world_context = {
        "sim_time": "18:15",
        "room_occupants": {"厨房": ["无限"], "客厅": ["哪吒"]},
        "world_summary": "年夜饭准备中，大家各自活动",
    }
    prompt = agent.build_prompt(world_context)
    assert "无限" in prompt
    assert "厨房" in prompt
    assert "18:15" in prompt
    assert "哪吒" in prompt  # world summary should mention who is where


def test_parse_llm_response_valid():
    agent = make_agent("小黑")
    raw = '{"思考": "想去看看无限做什么", "目标房间": "厨房", "动作": "走向厨房", "对话": "无限你在做什么？", "情绪": "好奇"}'
    result = agent.parse_response(raw)
    assert result["目标房间"] == "厨房"
    assert result["情绪"] == "好奇"


def test_parse_llm_response_falls_back_on_bad_json():
    agent = make_agent("小黑")
    raw = "这不是JSON"
    result = agent.parse_response(raw)
    assert result["目标房间"] == agent.current_room  # stay put
    assert "动作" in result
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_agents.py -v
```

Expected: `ImportError` — agents.py doesn't exist yet.

- [ ] **Step 3: Implement agents.py**

Create `backend/agents.py`:

```python
from __future__ import annotations
import json
import re
import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

import anthropic

from backend.characters import CharacterConfig


@dataclass
class MemoryEntry:
    time: str
    event: str
    importance: int  # 1-10


class Agent:
    MAX_MEMORIES = 20

    def __init__(self, config: CharacterConfig):
        self.name = config.name
        self.color = config.color
        self.personality = config.personality
        self.current_room: str = config.initial_room
        self.mood: str = config.initial_mood
        self.current_goal: str = config.initial_goal
        self.memories: list[MemoryEntry] = []
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic()
        return self._client

    def add_memory(self, event: str, importance: int, sim_time: str) -> None:
        entry = MemoryEntry(time=sim_time, event=event, importance=importance)
        if len(self.memories) >= self.MAX_MEMORIES:
            # evict the entry with the lowest importance (oldest if tie)
            min_idx = min(range(len(self.memories)), key=lambda i: self.memories[i].importance)
            self.memories.pop(min_idx)
        self.memories.append(entry)

    def build_prompt(self, world_context: dict[str, Any]) -> str:
        sim_time = world_context["sim_time"]
        room_occupants: dict[str, list[str]] = world_context["room_occupants"]
        world_summary: str = world_context["world_summary"]

        others_here = [n for n in room_occupants.get(self.current_room, []) if n != self.name]
        others_str = "、".join(others_here) if others_here else "没有其他人"

        # Build occupancy overview for world context
        location_lines = []
        for room, occupants in room_occupants.items():
            if occupants:
                location_lines.append(f"  - {room}：{'、'.join(occupants)}")
        location_overview = "\n".join(location_lines) if location_lines else "  - 大家分散在各处"

        # Recent memories (last 10)
        recent = self.memories[-10:] if self.memories else []
        memory_str = "\n".join(
            f"  [{m.time}] {m.event}" for m in recent
        ) if recent else "  （暂无记忆）"

        return f"""{self.personality}

现在是除夕夜 {sim_time}，你在哪吒的豪华大平层。

【当前状态】
- 当前位置：{self.current_room}
- 心情：{self.mood}
- 当前目标：{self.current_goal}
- 房间内还有：{others_str}

【各人位置】
{location_overview}

【最近的记忆】
{memory_str}

【整体情况】
{world_summary}

请决定接下来5分钟（到 {self._next_time(sim_time)} ）你做什么。
用JSON回复，必须包含以下字段（中文键名）：
{{
  "思考": "你内心的想法（1-2句）",
  "目标房间": "你想去或留在的房间名（必须是：院子/厨房/餐厅/游戏室/衣帽间/客厅/次阳台/卧室/卫生间 之一）",
  "动作": "你正在做的具体动作（1句）",
  "对话": "你说的话，如果不想说话就填空字符串",
  "情绪": "当前情绪（1-3个字）"
}}
只回复JSON，不要任何其他内容。"""

    @staticmethod
    def _next_time(sim_time: str) -> str:
        h, m = map(int, sim_time.split(":"))
        m += 5
        if m >= 60:
            h += 1
            m -= 60
        return f"{h:02d}:{m:02d}"

    def parse_response(self, raw: str) -> dict[str, Any]:
        """Parse LLM response JSON, falling back to stay-put on failure."""
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            data = json.loads(raw)
            # Validate target room
            valid_rooms = {"院子","厨房","餐厅","游戏室","衣帽间","客厅","次阳台","卧室","卫生间"}
            if data.get("目标房间") not in valid_rooms:
                data["目标房间"] = self.current_room
            return data
        except (json.JSONDecodeError, KeyError):
            return {
                "思考": "（解析失败）",
                "目标房间": self.current_room,
                "动作": "发呆",
                "对话": "",
                "情绪": self.mood,
            }

    async def think(self, world_context: dict[str, Any]) -> dict[str, Any]:
        """Call LLM and return parsed decision. Adds random jitter to avoid rate limits."""
        await asyncio.sleep(random.uniform(0, 0.5))
        prompt = self.build_prompt(world_context)
        client = self._get_client()
        message = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
        result = self.parse_response(raw)
        # Update agent state
        self.mood = result.get("情绪", self.mood)
        self.current_goal = result.get("动作", self.current_goal)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agents.py -v
```

Expected: All 8 tests PASS. (No real API calls — `think()` is not tested here.)

- [ ] **Step 5: Commit**

```bash
git add backend/agents.py tests/test_agents.py
git commit -m "Add Agent class with memory stream, prompt builder, and LLM think()"
```

---

## Task 5: World State and Tick Scheduler

**Files:**
- Create: `backend/world.py`
- Create: `tests/test_world.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_world.py`:

```python
import pytest
from backend.world import WorldState, SimClock
from backend.characters import CHARACTERS


def make_world() -> WorldState:
    return WorldState(CHARACTERS)


def test_initial_positions():
    world = make_world()
    assert world.agent_rooms["小黑"] == "餐厅"
    assert world.agent_rooms["无限"] == "厨房"
    assert world.agent_rooms["哪吒"] == "客厅"
    assert world.agent_rooms["鹿野"] == "次阳台"


def test_room_occupants():
    world = make_world()
    occ = world.get_room_occupants()
    assert "小黑" in occ["餐厅"]
    assert "无限" in occ["厨房"]
    assert "哪吒" in occ["客厅"]
    assert "鹿野" in occ["次阳台"]


def test_update_agent_position():
    world = make_world()
    world.update_agent_room("小黑", "客厅")
    assert world.agent_rooms["小黑"] == "客厅"
    occ = world.get_room_occupants()
    assert "小黑" in occ["客厅"]
    assert "小黑" not in occ.get("餐厅", [])


def test_world_summary_contains_names():
    world = make_world()
    summary = world.build_summary()
    for name in ["小黑", "无限", "哪吒", "鹿野"]:
        assert name in summary


def test_sim_clock_advances():
    clock = SimClock(start_hour=18, start_minute=0)
    assert clock.current == "18:00"
    clock.advance(5)
    assert clock.current == "18:05"
    clock.advance(55)
    assert clock.current == "19:00"


def test_sim_clock_caps_at_23():
    clock = SimClock(start_hour=22, start_minute=55)
    clock.advance(10)
    assert clock.current == "23:05"
    assert not clock.is_over()
    clock.advance(55)
    assert clock.is_over()
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_world.py -v
```

Expected: `ImportError` — world.py doesn't exist yet.

- [ ] **Step 3: Implement world.py**

Create `backend/world.py`:

```python
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
        occupants = self.get_room_occupants()
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
        """Apply agent decision to world state. Returns movement path."""
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
            return path
        else:
            self.agent_paths[name] = [current]
            return [current]


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
            for agent, decision in zip(self.agents, decisions):
                if isinstance(decision, Exception):
                    logger.error(f"{agent.name} think() failed: {decision}")
                    continue
                path = self.world.apply_decision(agent.name, decision)
                tick_events.append({
                    "type": "agent_action",
                    "name": agent.name,
                    "color": agent.color,
                    "room": self.world.agent_rooms[agent.name],
                    "path": path,
                    "action": decision.get("动作", ""),
                    "dialogue": decision.get("对话", ""),
                    "mood": decision.get("情绪", ""),
                    "thought": decision.get("思考", ""),
                })
                # Push memories to all agents in same room
                if decision.get("动作"):
                    event_text = f"{agent.name}在{self.world.agent_rooms[agent.name]}：{decision['动作']}"
                    for other_agent in self.agents:
                        if other_agent.current_room == self.world.agent_rooms[agent.name]:
                            other_agent.add_memory(event_text, importance=5, sim_time=sim_time)
                if decision.get("对话"):
                    dialogue_text = f"{agent.name}说："{decision['对话']}""
                    for other_agent in self.agents:
                        if other_agent.current_room == self.world.agent_rooms[agent.name]:
                            other_agent.add_memory(dialogue_text, importance=7, sim_time=sim_time)
                # Update agent's own room reference
                agent.current_room = self.world.agent_rooms[agent.name]

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_world.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/world.py tests/test_world.py
git commit -m "Add WorldState, SimClock, and TickScheduler"
```

---

## Task 6: FastAPI WebSocket Server

**Files:**
- Create: `backend/main.py`

- [ ] **Step 1: Create main.py**

```python
from __future__ import annotations
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.agents import Agent
from backend.characters import CHARACTERS
from backend.world import WorldState, TickScheduler

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# Global state
connected_clients: set[WebSocket] = set()
scheduler: TickScheduler | None = None
scheduler_task: asyncio.Task | None = None


async def broadcast(event: dict) -> None:
    if not connected_clients:
        return
    message = json.dumps(event, ensure_ascii=False)
    dead = set()
    for ws in connected_clients:
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
    logger.info("Simulation started")
    yield
    if scheduler_task:
        scheduler_task.cancel()
    logger.info("Simulation stopped")


app = FastAPI(lifespan=lifespan)

# Serve frontend
frontend_dir = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def serve_index():
    return FileResponse(frontend_dir / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"Client connected ({len(connected_clients)} total)")

    # Send current world state immediately to newly connected client
    if scheduler:
        await websocket.send_text(json.dumps({
            "type": "world_update",
            "sim_time": scheduler.clock.current,
            "positions": dict(scheduler.world.agent_rooms),
            "moods": dict(scheduler.world.agent_moods),
        }, ensure_ascii=False))

    try:
        while True:
            # Keep connection alive; client sends no messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info(f"Client disconnected ({len(connected_clients)} total)")
```

- [ ] **Step 2: Verify server starts (no API calls yet)**

```bash
cd /Users/bytedance/luoxiaohei-atown
source .venv/bin/activate
# Set a dummy key to test startup
ANTHROPIC_API_KEY=sk-test uvicorn backend.main:app --host 0.0.0.0 --port 8765 &
sleep 2
curl -s http://localhost:8765/ | head -5
kill %1
```

Expected: HTML content returned (the index.html will 404 until Task 7, but FastAPI starts cleanly).

- [ ] **Step 3: Commit**

```bash
git add backend/main.py
git commit -m "Add FastAPI WebSocket server with lifespan scheduler"
```

---

## Task 7: HTML Canvas Frontend

**Files:**
- Create: `frontend/index.html`

This is the largest file. It mirrors the floorplan-v5 SVG layout on Canvas and handles WebSocket events.

**Room center coordinates** (from floorplan-v5 grid: 740×600 SVG, walls at x=190/545, rows y=100/262/424/586):

| Room | Canvas center |
|------|--------------|
| 院子 | (370, 57) |
| 厨房 | (102, 181) |
| 餐厅 | (367, 181) |
| 游戏室 | (635, 181) |
| 衣帽间 | (102, 343) |
| 客厅 | (367, 343) |
| 次阳台 | (635, 343) |
| 卧室 | (202, 505) |
| 卫生间 | (558, 505) |

- [ ] **Step 1: Create frontend/index.html**

Create `frontend/index.html`:

```html
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>罗小黑 AI Town · 年夜饭</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d0b08;
    color: #e2d4b0;
    font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }
  #header {
    padding: 8px 16px;
    background: #1a1208;
    border-bottom: 1px solid #3a2a10;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  #header h1 { font-size: 15px; color: #e2b04a; }
  #sim-time { font-size: 13px; color: #c8a060; font-variant-numeric: tabular-nums; }
  #status { font-size: 11px; color: #806040; margin-left: auto; }
  #main {
    display: flex;
    flex: 1;
    overflow: hidden;
  }
  #canvas-wrap {
    flex: 0 0 740px;
    position: relative;
    overflow: hidden;
    background: #181410;
  }
  canvas { display: block; }
  #log-panel {
    flex: 1;
    overflow-y: auto;
    padding: 10px 12px;
    background: #100e08;
    border-left: 1px solid #2a1e0a;
    font-size: 12px;
    line-height: 1.7;
  }
  .log-entry { margin-bottom: 6px; border-bottom: 1px solid #1e1608; padding-bottom: 4px; }
  .log-time { color: #806040; margin-right: 4px; }
  .log-name { font-weight: bold; margin-right: 4px; }
  .log-thought { color: #a09060; font-style: italic; }
  .log-action { color: #c0b080; }
  .log-dialogue { color: #e8d090; }
  .log-mood { color: #90a060; font-size: 11px; }
</style>
</head>
<body>
<div id="header">
  <h1>罗小黑 AI Town · 除夕夜年夜饭</h1>
  <div id="sim-time">18:00</div>
  <div id="status">连接中...</div>
</div>
<div id="main">
  <div id="canvas-wrap">
    <canvas id="canvas" width="740" height="600"></canvas>
  </div>
  <div id="log-panel" id="log"></div>
</div>

<script>
// ─── Constants ─────────────────────────────────────────────────────────────
const ROOM_CENTERS = {
  '院子':   { x: 370, y: 57 },
  '厨房':   { x: 102, y: 181 },
  '餐厅':   { x: 367, y: 181 },
  '游戏室': { x: 635, y: 181 },
  '衣帽间': { x: 102, y: 343 },
  '客厅':   { x: 367, y: 343 },
  '次阳台': { x: 635, y: 343 },
  '卧室':   { x: 202, y: 505 },
  '卫生间': { x: 558, y: 505 },
};

const CHAR_COLORS = {
  '小黑': '#4a90d9',
  '无限': '#9b59b6',
  '哪吒': '#e74c3c',
  '鹿野': '#27ae60',
};

const SPRITE_R = 16;
const BUBBLE_DURATION = 4000; // ms

// ─── State ──────────────────────────────────────────────────────────────────
const sprites = {};      // name → { x, y, targetX, targetY, room, mood }
const bubbles = [];      // { name, text, x, y, expiry }
let simTime = '18:00';

// Initialize sprites at starting positions
const INITIAL = { '小黑': '餐厅', '无限': '厨房', '哪吒': '客厅', '鹿野': '次阳台' };
for (const [name, room] of Object.entries(INITIAL)) {
  const c = ROOM_CENTERS[room];
  const offset = offsetForName(name, room);
  sprites[name] = {
    x: c.x + offset.x,
    y: c.y + offset.y,
    targetX: c.x + offset.x,
    targetY: c.y + offset.y,
    room,
    mood: '',
  };
}

function offsetForName(name, room) {
  // Spread characters in the same room so they don't overlap
  const names = ['小黑', '无限', '哪吒', '鹿野'];
  const idx = names.indexOf(name);
  const offsets = [{ x: -18, y: -14 }, { x: 18, y: -14 }, { x: -18, y: 14 }, { x: 18, y: 14 }];
  return offsets[idx] || { x: 0, y: 0 };
}

// ─── Canvas Setup ────────────────────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// ─── Floor Plan Drawing ──────────────────────────────────────────────────────
function drawFloorplan() {
  // Room fills
  const fills = [
    { x: 14, y: 14, w: 712, h: 86,  fill: '#bccab4' }, // 院子
    { x: 14, y: 100, w: 176, h: 162, fill: '#e0ece8' }, // 厨房
    { x: 190, y: 100, w: 355, h: 162, fill: '#f0ddb8' }, // 餐厅
    { x: 545, y: 100, w: 181, h: 162, fill: '#d2c8e4' }, // 游戏室
    { x: 14, y: 262, w: 176, h: 162, fill: '#f0e4d2' }, // 衣帽间
    { x: 190, y: 262, w: 355, h: 162, fill: '#f0ddb8' }, // 客厅
    { x: 545, y: 262, w: 181, h: 162, fill: '#bccab4' }, // 次阳台
    { x: 14, y: 424, w: 376, h: 162, fill: '#dce6f2' }, // 卧室
    { x: 390, y: 424, w: 336, h: 162, fill: '#e0ece8' }, // 卫生间
  ];
  for (const r of fills) {
    ctx.fillStyle = r.fill;
    ctx.fillRect(r.x, r.y, r.w, r.h);
  }

  // Room labels
  ctx.fillStyle = 'rgba(80,50,20,0.55)';
  ctx.font = 'bold 11px PingFang SC, sans-serif';
  ctx.textAlign = 'center';
  const labels = [
    ['院子', 370, 25], ['厨房', 102, 112], ['餐厅', 367, 112],
    ['游戏室', 635, 112], ['衣帽间', 102, 274], ['客厅', 190+16, 274],
    ['次阳台', 635, 274], ['卧室', 100, 436], ['卫生间', 510, 436],
  ];
  for (const [label, lx, ly] of labels) {
    ctx.fillText(label, lx, ly);
  }

  // Outer walls
  ctx.strokeStyle = '#28140a';
  ctx.lineWidth = 8;
  ctx.strokeRect(14, 14, 712, 572);

  // Inner walls (simplified — key dividers)
  ctx.lineWidth = 6;

  // Horizontal: y=100
  _line(14, 100, 79, 100);
  _line(154, 100, 545, 100);
  _line(558, 100, 726, 100);

  // Horizontal: y=262
  _line(14, 262, 72, 262);
  _line(146, 262, 726, 262);

  // Horizontal: y=424
  _line(14, 424, 726, 424);

  // Vertical: x=190 (y=100-424)
  _line(190, 100, 190, 142);
  _line(190, 208, 190, 300);
  _line(190, 368, 190, 424);

  // Vertical: x=545 (y=100-424)
  _line(545, 100, 545, 134);
  _line(545, 200, 545, 296);
  _line(545, 370, 545, 424);

  // Vertical: x=390 (y=424-586)
  _line(390, 424, 390, 488);
  _line(390, 554, 390, 586);

  // Door arcs (swing doors)
  ctx.strokeStyle = '#8a6430';
  ctx.lineWidth = 1.5;
  // 厨→餐 door at x=190, y=142-208
  _arc(190, 142, 66, 0, Math.PI / 2);
  // 衣→客 door at x=190, y=300-368
  _arc(190, 300, 68, 0, Math.PI / 2);
  // 卧→卫 door at x=390, y=488-554
  _arc(390, 554, 66, -Math.PI / 2, 0);

  // Sliding door lattice: 院子↔餐厅 (gap 306-440 at y=100)
  _slidingDoor(306, 97, 134, 6);
  // 游戏室↔餐厅 sliding at x=545, y=136-198
  _slidingDoorV(543, 136, 62, 6);
  // 客厅↔次阳台 sliding at x=545, y=298-368
  _slidingDoorV(543, 298, 70, 6);

  // Closing stroke for outer
  ctx.strokeStyle = '#28140a';
  ctx.lineWidth = 8;
}

function _line(x1, y1, x2, y2) {
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
}

function _arc(cx, cy, r, startAngle, endAngle) {
  ctx.beginPath(); ctx.arc(cx, cy, r, startAngle, endAngle); ctx.stroke();
}

function _slidingDoor(x, y, w, cellSize) {
  // Horizontal lattice pattern
  ctx.strokeStyle = '#c49858';
  ctx.lineWidth = 1;
  for (let i = 0; i <= w; i += cellSize) {
    _line(x + i, y, x + i, y + 6);
  }
  _line(x, y + 3, x + w, y + 3);
}

function _slidingDoorV(x, y, h, cellSize) {
  ctx.strokeStyle = '#c49858';
  ctx.lineWidth = 1;
  for (let i = 0; i <= h; i += cellSize) {
    _line(x, y + i, x + 6, y + i);
  }
  _line(x + 3, y, x + 3, y + h);
}

// ─── Sprite Drawing ──────────────────────────────────────────────────────────
function drawSprites() {
  for (const [name, s] of Object.entries(sprites)) {
    // Lerp toward target
    s.x += (s.targetX - s.x) * 0.12;
    s.y += (s.targetY - s.y) * 0.12;

    const color = CHAR_COLORS[name] || '#aaa';

    // Shadow
    ctx.shadowColor = 'rgba(0,0,0,0.4)';
    ctx.shadowBlur = 6;

    // Circle
    ctx.beginPath();
    ctx.arc(s.x, s.y, SPRITE_R, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.shadowBlur = 0;

    // Name label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 10px PingFang SC, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(name, s.x, s.y + 4);

    // Mood dot below
    if (s.mood) {
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.font = '9px PingFang SC, sans-serif';
      ctx.fillText(s.mood, s.x, s.y + SPRITE_R + 10);
    }
  }
}

// ─── Bubble Drawing ──────────────────────────────────────────────────────────
function drawBubbles() {
  const now = Date.now();
  for (let i = bubbles.length - 1; i >= 0; i--) {
    const b = bubbles[i];
    if (now > b.expiry) { bubbles.splice(i, 1); continue; }
    const alpha = Math.min(1, (b.expiry - now) / 800);
    ctx.globalAlpha = alpha;

    // Measure text
    ctx.font = '12px PingFang SC, sans-serif';
    const maxWidth = 160;
    const lines = wrapText(b.text, maxWidth);
    const bw = maxWidth + 16;
    const bh = lines.length * 18 + 12;
    const bx = b.x - bw / 2;
    const by = b.y - SPRITE_R - bh - 10;

    // Bubble background
    ctx.fillStyle = 'rgba(30, 20, 8, 0.88)';
    roundRect(ctx, bx, by, bw, bh, 6);
    ctx.fill();
    ctx.strokeStyle = CHAR_COLORS[b.name] || '#ccc';
    ctx.lineWidth = 1.5;
    roundRect(ctx, bx, by, bw, bh, 6);
    ctx.stroke();

    // Tail
    ctx.beginPath();
    ctx.moveTo(b.x - 6, by + bh);
    ctx.lineTo(b.x, by + bh + 8);
    ctx.lineTo(b.x + 6, by + bh);
    ctx.fillStyle = 'rgba(30, 20, 8, 0.88)';
    ctx.fill();

    // Text
    ctx.fillStyle = '#e8d090';
    ctx.textAlign = 'left';
    lines.forEach((line, i) => {
      ctx.fillText(line, bx + 8, by + 16 + i * 18);
    });

    ctx.globalAlpha = 1;
  }
}

function wrapText(text, maxWidth) {
  const words = text.split('');
  const lines = [];
  let line = '';
  ctx.font = '12px PingFang SC, sans-serif';
  for (const ch of words) {
    const test = line + ch;
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = ch;
    } else {
      line = test;
    }
  }
  if (line) lines.push(line);
  return lines;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

// ─── Render Loop ─────────────────────────────────────────────────────────────
function render() {
  ctx.clearRect(0, 0, 740, 600);
  drawFloorplan();
  drawSprites();
  drawBubbles();
  requestAnimationFrame(render);
}
render();

// ─── WebSocket ───────────────────────────────────────────────────────────────
const statusEl = document.getElementById('status');
const simTimeEl = document.getElementById('sim-time');
const logPanel = document.querySelector('#log-panel');

function addLog(html) {
  const div = document.createElement('div');
  div.className = 'log-entry';
  div.innerHTML = html;
  logPanel.prepend(div);
  // Keep log manageable
  while (logPanel.children.length > 200) logPanel.removeChild(logPanel.lastChild);
}

function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onopen = () => { statusEl.textContent = '已连接'; };
  ws.onclose = () => {
    statusEl.textContent = '已断开，5秒后重连...';
    setTimeout(connectWS, 5000);
  };
  ws.onerror = () => { statusEl.textContent = '连接错误'; };

  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    handleEvent(data);
  };
}

function handleEvent(data) {
  switch (data.type) {
    case 'tick_start':
      simTimeEl.textContent = data.sim_time;
      break;

    case 'agent_action': {
      const { name, room, path, action, dialogue, mood, thought } = data;
      // Move sprite toward new room
      if (room && ROOM_CENTERS[room]) {
        const offset = offsetForName(name, room);
        const c = ROOM_CENTERS[room];
        if (sprites[name]) {
          sprites[name].targetX = c.x + offset.x;
          sprites[name].targetY = c.y + offset.y;
          sprites[name].room = room;
          sprites[name].mood = mood || '';
        }
      }
      // Speech bubble
      if (dialogue && sprites[name]) {
        bubbles.push({
          name,
          text: dialogue,
          x: sprites[name].x,
          y: sprites[name].y,
          expiry: Date.now() + BUBBLE_DURATION,
        });
      }
      // Log entry
      const color = CHAR_COLORS[name] || '#aaa';
      addLog(
        `<span class="log-time">${simTimeEl.textContent}</span>` +
        `<span class="log-name" style="color:${color}">${name}</span>` +
        (thought ? `<span class="log-thought">${thought}</span><br>` : '') +
        (action ? `<span class="log-action">${action}</span>` : '') +
        (dialogue ? `<br><span class="log-dialogue">"${dialogue}"</span>` : '') +
        (mood ? `<span class="log-mood"> [${mood}]</span>` : '')
      );
      break;
    }

    case 'world_update':
      simTimeEl.textContent = data.sim_time;
      // Sync moods
      if (data.moods) {
        for (const [name, mood] of Object.entries(data.moods)) {
          if (sprites[name]) sprites[name].mood = mood;
        }
      }
      break;

    case 'simulation_end':
      statusEl.textContent = `模拟结束 ${data.sim_time}`;
      addLog(`<span style="color:#e2b04a">—— 除夕夜结束 ${data.sim_time} ——</span>`);
      break;
  }
}

connectWS();
</script>
</body>
</html>
```

- [ ] **Step 2: Start the full server and open in browser**

```bash
cd /Users/bytedance/luoxiaohei-atown
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8765 --reload
```

Open `http://localhost:8765` in browser.

Expected:
- Floor plan renders with room fills and walls
- 4 colored sprites appear at correct starting positions (小黑→餐厅, 无限→厨房, 哪吒→客厅, 鹿野→次阳台)
- WebSocket connects ("已连接" status)
- Every ~4 seconds, tick fires: sprites move, speech bubbles appear, log entries populate
- Sim time advances in header (18:00 → 23:00)

- [ ] **Step 3: Commit**

```bash
git add frontend/index.html backend/main.py
git commit -m "Add Canvas frontend with speech bubbles, WebSocket client, and full server wiring"
```

---

## Task 8: Run Full Test Suite

- [ ] **Step 1: Run all tests**

```bash
cd /Users/bytedance/luoxiaohei-atown
source .venv/bin/activate
pytest tests/ -v
```

Expected output (all pass):
```
tests/test_pathfinding.py::test_same_room_returns_single_element PASSED
tests/test_pathfinding.py::test_adjacent_rooms_direct_path PASSED
tests/test_pathfinding.py::test_multi_hop_path PASSED
tests/test_pathfinding.py::test_long_path PASSED
tests/test_pathfinding.py::test_invalid_room_raises PASSED
tests/test_pathfinding.py::test_all_rooms_reachable PASSED
tests/test_agents.py::test_agent_initial_state PASSED
tests/test_agents.py::test_add_memory_stores_entry PASSED
tests/test_agents.py::test_memory_capped_at_20 PASSED
tests/test_agents.py::test_memory_evicts_lowest_importance PASSED
tests/test_agents.py::test_build_prompt_contains_key_fields PASSED
tests/test_agents.py::test_parse_llm_response_valid PASSED
tests/test_agents.py::test_parse_llm_response_falls_back_on_bad_json PASSED
tests/test_world.py::test_initial_positions PASSED
tests/test_world.py::test_room_occupants PASSED
tests/test_world.py::test_update_agent_position PASSED
tests/test_world.py::test_world_summary_contains_names PASSED
tests/test_world.py::test_sim_clock_advances PASSED
tests/test_world.py::test_sim_clock_caps_at_23 PASSED
```

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "All tests passing — project complete"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] 4 角色初始设定 → `characters.py` Task 3
- [x] Memory Stream (20条, importance eviction) → `agents.py` Task 4
- [x] LLM prompt 结构 (5字段JSON) → `build_prompt()` Task 4
- [x] asyncio.gather 4并发 + 0.5s jitter → `world.py` TickScheduler Task 5
- [x] A* 寻路图 (9节点) → `pathfinding.py` Task 2
- [x] WebSocket 4种事件 (tick_start / agent_action / dialogue / world_update) → `main.py` + `index.html`
- [x] Canvas 平面图 (floorplan-v5 layout) → `index.html` drawFloorplan()
- [x] 角色精灵 + 移动插值动画 → sprite lerp in render loop
- [x] 对话气泡 3秒淡出 → bubbles array with expiry
- [x] 右侧滚动日志 → log-panel with prepend
- [x] 底部/顶部时间轴 → sim-time in header
- [x] rate limit jitter → `asyncio.sleep(random.uniform(0, 0.5))` in `think()`
- [x] 对话只在同房间触发 → memory push checks `current_room` match
- [x] 角色不能传送，经路径移动 → `apply_decision` moves one room per tick via path
