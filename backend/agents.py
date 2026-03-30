from __future__ import annotations
import json
import re
import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

import anthropic

from backend.characters import CharacterConfig

# Room props — tells each agent what they can interact with
ROOM_PROPS: dict[str, str] = {
    "院子":   "竹林、石板路、春节红灯笼、鞭炮（未点燃）",
    "厨房":   "灶台（两个灶眼）、岛台、悬挂锅具、冰箱、食材（红烧肉、饺子皮、蔬菜）",
    "餐厅":   "大圆桌、火锅底座（可开火）、4把椅子、酒柜（白酒/红酒/饮料）",
    "游戏室": "双显示器电竞桌、VR头显、电竞椅、Nintendo Switch、游戏手柄、零食",
    "衣帽间": "哪吒战袍展示架、球鞋墙（数十双）、全身镜",
    "客厅":   "65寸TV、PS5主机（开着）、L形沙发、懒人沙发、茶几、遥控器",
    "次阳台": "藤椅两把、大型盆栽、城市夜景、春节灯笼、微风",
    "卧室":   "特大号床、智能控制面板、台灯",
    "卫生间": "独立浴缸、淋浴间、马桶",
}


@dataclass
class MemoryEntry:
    time: str
    event: str
    importance: int   # 1-10
    tick_num: int = 0  # for recency decay


class Agent:
    MAX_MEMORIES = 40
    REFLECT_EVERY = 10  # trigger reflection after this many new memories

    def __init__(self, config: CharacterConfig):
        self.name = config.name
        self.color = config.color
        self.core_traits = config.core_traits
        self.relationships = config.relationships
        self.speech_style = config.speech_style
        self.taboos = config.taboos
        self.current_room: str = config.initial_room
        self.mood: str = config.initial_mood
        self.current_goal: str = config.initial_goal
        self.memories: list[MemoryEntry] = []
        self.plan: list[str] = []            # evening plan from planning phase
        self._memories_since_reflection = 0
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic()
        return self._client

    # ── Memory ────────────────────────────────────────────────────────────────

    def add_memory(self, event: str, importance: int, sim_time: str, tick_num: int = 0) -> None:
        entry = MemoryEntry(time=sim_time, event=event, importance=importance, tick_num=tick_num)
        if len(self.memories) >= self.MAX_MEMORIES:
            # evict lowest importance (oldest if tie)
            min_idx = min(range(len(self.memories)), key=lambda i: self.memories[i].importance)
            self.memories.pop(min_idx)
        self.memories.append(entry)
        self._memories_since_reflection += 1

    def _score_memory(self, entry: MemoryEntry, current_tick: int, context_keywords: set[str]) -> float:
        """Score by recency × importance × relevance (Stanford AI Town retrieval)."""
        # Recency: exponential decay (half-life ≈ 70 ticks)
        ticks_ago = max(0, current_tick - entry.tick_num)
        recency = 0.99 ** ticks_ago
        # Importance: normalized 0-1
        importance = entry.importance / 10.0
        # Relevance: keyword overlap with context
        entry_words = set(re.split(r'[，。：、\s]+', entry.event))
        overlap = len(entry_words & context_keywords)
        relevance = min(overlap / max(len(context_keywords), 1) * 3, 1.0)
        return 0.5 * recency + 0.3 * importance + 0.2 * relevance

    def retrieve_memories(self, context: str, k: int = 10, current_tick: int = 0) -> list[MemoryEntry]:
        """Return top-k memories ranked by recency+importance+relevance."""
        if not self.memories:
            return []
        context_keywords = set(re.split(r'[，。：、\s]+', context)) - {'', '的', '了', '在', '是'}
        scored = [
            (self._score_memory(m, current_tick, context_keywords), m)
            for m in self.memories
        ]
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:k]]

    def should_reflect(self) -> bool:
        return self._memories_since_reflection >= self.REFLECT_EVERY and len(self.memories) >= 5

    # ── Planning (called once at simulation start) ────────────────────────────

    async def plan_evening(self) -> None:
        """Generate a rough evening plan. Called once at startup before ticks begin."""
        await asyncio.sleep(random.uniform(0, 0.5))
        prompt = f"""你是{self.name}。

【性格】{self.core_traits}

今晚是除夕夜18:00-23:00，你在哪吒家过年夜饭。其他人有：无限、哪吒、鹿野（或小黑，取决于你是谁）。

请为自己制定今晚的大致心愿计划（3-5件想做或想发生的事），要符合你的性格。
只回复JSON数组，格式：["计划1", "计划2", ...]
不要时间戳，直接写想做的事。只回复JSON，不要其他内容。"""
        client = self._get_client()
        try:
            msg = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _strip_fences(msg.content[0].text)
            self.plan = json.loads(raw)
        except Exception:
            self.plan = []

    # ── Reflection (triggered after REFLECT_EVERY new memories) ──────────────

    async def reflect(self, sim_time: str, current_tick: int) -> list[str]:
        """Synthesize recent memories into higher-level insights. Returns insight strings."""
        recent = self.memories[-15:]
        events_text = "\n".join(f"- [{m.time}] {m.event}" for m in recent)
        prompt = f"""你是{self.name}。

基于你最近的经历：
{events_text}

请提炼出3条最有意义的洞察或感受（关于人、关于今晚、关于自己）。每条一句话，要具体而非空泛。
只回复JSON数组：["洞察1", "洞察2", "洞察3"]"""
        client = self._get_client()
        try:
            msg = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _strip_fences(msg.content[0].text)
            insights: list[str] = json.loads(raw)
            for insight in insights[:3]:
                self.add_memory(f"[反思] {insight}", importance=9, sim_time=sim_time, tick_num=current_tick)
            self._memories_since_reflection = 0
            return insights
        except Exception:
            self._memories_since_reflection = 0
            return []

    # ── Main decision (called every tick) ────────────────────────────────────

    def build_prompt(self, world_context: dict[str, Any], current_tick: int = 0) -> str:
        sim_time = world_context["sim_time"]
        room_occupants: dict[str, list[str]] = world_context["room_occupants"]
        world_summary: str = world_context["world_summary"]
        object_states: dict[str, dict[str, str]] = world_context.get("object_states", {})

        others_here = [n for n in room_occupants.get(self.current_room, []) if n != self.name]
        others_str = "、".join(others_here) if others_here else "没有其他人"
        alone = len(others_here) == 0

        # Location overview
        location_lines = []
        for room, occupants in room_occupants.items():
            if occupants:
                location_lines.append(f"  - {room}：{'、'.join(occupants)}")
        location_overview = "\n".join(location_lines) if location_lines else "  - 大家分散在各处"

        # Room props + current object states
        room_props = ROOM_PROPS.get(self.current_room, "")
        room_obj_states = object_states.get(self.current_room, {})
        obj_state_str = "、".join(f"{k}[{v}]" for k, v in room_obj_states.items()) if room_obj_states else ""
        env_str = room_props
        if obj_state_str:
            env_str += f"\n- 当前状态：{obj_state_str}"

        # Retrieve top memories by relevance + recency + importance
        context_for_retrieval = f"{self.current_room} {sim_time} {others_str} {world_summary}"
        retrieved = self.retrieve_memories(context_for_retrieval, k=10, current_tick=current_tick)
        memory_str = "\n".join(
            f"  [{m.time}] {m.event}" for m in retrieved
        ) if retrieved else "  （暂无记忆）"

        # Plan
        plan_str = "\n".join(f"  - {p}" for p in self.plan) if self.plan else "  （随性而为）"

        return f"""你是{self.name}。

【性格】{self.core_traits}
【关系】{self.relationships}
【说话风格】{self.speech_style}
【禁忌】{self.taboos}

现在是除夕夜 {sim_time}，你在哪吒的豪华大平层。

【今晚的计划】
{plan_str}

【当前状态】
- 当前位置：{self.current_room}
- 房间里可用的东西：{env_str}
- 心情：{self.mood}
- 当前目标：{self.current_goal}
- 房间内还有：{others_str}

【各人位置】
{location_overview}

【相关记忆（按重要性和相关度检索）】
{memory_str}

【整体情况】
{world_summary}

请决定接下来5分钟（到 {_next_time(sim_time)} ）你做什么。
动作要具体利用房间里的道具（比如：打开火锅底座、拿起Switch手柄、坐进藤椅看夜景）。
用JSON回复：
{{
  "思考": "内心想法（1-2句）",
  "目标房间": "房间名（院子/厨房/餐厅/游戏室/衣帽间/客厅/次阳台/卧室/卫生间 之一）",
  "动作": "具体动作，尽量与道具互动（1句）",
  "道具交互": "你改变了哪个道具的状态，格式 道具名:新状态，没有则填空字符串",
  "对话": "{'你独处，必须填空字符串' if alone else '可选，30%概率说话，不说则填空字符串'}",
  "情绪": "当前情绪（1-3个字）"
}}
只回复JSON。"""

    async def think(self, world_context: dict[str, Any], current_tick: int = 0) -> dict[str, Any]:
        """Main decision. Adds jitter to spread out concurrent API calls."""
        await asyncio.sleep(random.uniform(0, 0.5))
        prompt = self.build_prompt(world_context, current_tick)
        client = self._get_client()
        message = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=450,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
        result = self.parse_response(raw)
        self.mood = result.get("情绪", self.mood)
        self.current_goal = result.get("动作", self.current_goal)
        return result

    # ── Dialogue reaction (sub-tick, fast) ───────────────────────────────────

    async def react_to_dialogue(self, speaker: str, dialogue: str, sim_time: str) -> dict[str, Any] | None:
        """Quick reaction to someone speaking in the same room. Much shorter prompt."""
        await asyncio.sleep(random.uniform(0, 0.3))
        prompt = f"""你是{self.name}。
【性格】{self.core_traits}
【说话风格】{self.speech_style}
【禁忌】{self.taboos}

你和{speaker}都在{self.current_room}。{speaker}刚说：「{dialogue}」

你怎么回应？（可以不回应，沉默也是回应）
只回复JSON：{{"回应": "你说的话（不回应填空字符串）", "动作": "你的动作（1句）", "情绪": "情绪（1-3字）"}}"""
        client = self._get_client()
        try:
            msg = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _strip_fences(msg.content[0].text)
            return json.loads(raw)
        except Exception:
            return None

    def parse_response(self, raw: str) -> dict[str, Any]:
        raw = _strip_fences(raw)
        try:
            data = json.loads(raw)
            valid_rooms = {"院子", "厨房", "餐厅", "游戏室", "衣帽间", "客厅", "次阳台", "卧室", "卫生间"}
            if data.get("目标房间") not in valid_rooms:
                data["目标房间"] = self.current_room
            return data
        except (json.JSONDecodeError, KeyError):
            return {
                "思考": "（解析失败）",
                "目标房间": self.current_room,
                "动作": "发呆",
                "道具交互": "",
                "对话": "",
                "情绪": self.mood,
            }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_fences(raw: str) -> str:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    return re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)


def _next_time(sim_time: str) -> str:
    h, m = map(int, sim_time.split(":"))
    m += 5
    if m >= 60:
        h += 1
        m -= 60
    return f"{h:02d}:{m:02d}"
