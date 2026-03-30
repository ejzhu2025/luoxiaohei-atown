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
    importance: int  # 1-10


class Agent:
    MAX_MEMORIES = 20

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
        alone = len(others_here) == 0

        # Build occupancy overview
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

        room_props = ROOM_PROPS.get(self.current_room, "")

        return f"""你是{self.name}。

【性格】{self.core_traits}
【关系】{self.relationships}
【说话风格】{self.speech_style}
【禁忌】{self.taboos}

现在是除夕夜 {sim_time}，你在哪吒的豪华大平层。

【当前状态】
- 当前位置：{self.current_room}
- 房间里有这些东西可以使用：{room_props}
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
  "对话": "{'你现在独处，对话必须是空字符串' if alone else '你可以说一句话，但不是每次都要开口，大约30%的时候保持沉默更自然。不开口时填空字符串'}",
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
            valid_rooms = {"院子", "厨房", "餐厅", "游戏室", "衣帽间", "客厅", "次阳台", "卧室", "卫生间"}
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
