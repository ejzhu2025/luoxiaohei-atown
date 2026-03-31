from __future__ import annotations
import json
import os
import re
import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

import anthropic

from backend.characters import CharacterConfig, WORLD_BACKGROUND

_MODEL = "claude-sonnet-4-6"

# 房间道具表 — 每个房间里可交互的物品描述
# 这些内容会注入到 build_prompt() 中，让 LLM 知道当前房间有什么可用
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
    """单条记忆条目，对应 Stanford AI Town 的 Memory Stream 中的一个节点"""
    time: str          # 仿真时间戳，如 "18:30"
    event: str         # 事件描述（自然语言）
    importance: int    # 重要性评分，1-10
    tick_num: int = 0  # 记忆发生时的全局 tick 编号，用于计算时间衰减


class Agent:
    """
    单个 NPC 智能体，实现 Stanford AI Town 的三大模块：
      1. Memory Stream  — add_memory / retrieve_memories
      2. Reflection     — reflect（每 REFLECT_EVERY 条新记忆触发一次）
      3. Planning       — plan_evening（仿真开始前调用一次）
    以及每 tick 的主决策循环 think() 和对话反应 react_to_dialogue()。
    """

    MAX_MEMORIES = 40      # 记忆上限；超出时淘汰重要性最低的条目
    REFLECT_EVERY = 10     # 每积累这么多条新记忆就触发一次反思

    def __init__(self, config: CharacterConfig):
        # 基本人物属性，从 CharacterConfig 读取
        self.name = config.name
        self.color = config.color               # 前端渲染用颜色
        self.core_traits = config.core_traits   # 核心性格描述
        self.relationships = config.relationships  # 与其他角色的关系
        self.speech_style = config.speech_style   # 说话风格
        self.taboos = config.taboos               # 禁忌行为

        # 运行时状态
        self.current_room: str = config.initial_room   # 当前所在房间
        self.mood: str = config.initial_mood            # 当前心情（LLM 每 tick 更新）
        self.current_goal: str = config.initial_goal   # 当前目标（LLM 每 tick 更新）

        # 记忆系统
        self.memories: list[MemoryEntry] = []
        self.plan: list[str] = []                  # 晚间计划，仿真开始前由 plan_evening 生成
        self._memories_since_reflection = 0        # 距上次反思新增的记忆数量

        # Anthropic 客户端（懒初始化，避免多余连接）
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """懒初始化 Anthropic 异步客户端（单例）"""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic()
        return self._client

    # ── Memory（记忆流） ──────────────────────────────────────────────────────

    def add_memory(self, event: str, importance: int, sim_time: str, tick_num: int = 0) -> None:
        """
        向记忆流添加一条新记忆。
        若已达上限 MAX_MEMORIES，则淘汰重要性最低的一条（同等重要性时淘汰最旧的）。
        同时递增反思计数器，用于判断是否触发 reflect()。
        """
        entry = MemoryEntry(time=sim_time, event=event, importance=importance, tick_num=tick_num)
        if len(self.memories) >= self.MAX_MEMORIES:
            # 找到重要性最低的索引并弹出
            min_idx = min(range(len(self.memories)), key=lambda i: self.memories[i].importance)
            self.memories.pop(min_idx)
        self.memories.append(entry)
        self._memories_since_reflection += 1

    def _score_memory(self, entry: MemoryEntry, current_tick: int, context_keywords: set[str]) -> float:
        """
        Stanford AI Town 记忆检索评分公式：
          score = 0.5 × recency + 0.3 × importance + 0.2 × relevance

        - recency：指数衰减，half-life ≈ 70 ticks（0.99^ticks_ago）
        - importance：归一化到 [0, 1]（原始值除以 10）
        - relevance：与上下文关键词的词汇重叠率，最高封顶 1.0
        """
        # 时间衰减：距离当前 tick 越久，得分越低
        ticks_ago = max(0, current_tick - entry.tick_num)
        recency = 0.99 ** ticks_ago

        # 重要性归一化
        importance = entry.importance / 10.0

        # 关键词相关性：将记忆文本拆词后与上下文关键词求交集
        entry_words = set(re.split(r'[，。：、\s]+', entry.event))
        overlap = len(entry_words & context_keywords)
        relevance = min(overlap / max(len(context_keywords), 1) * 3, 1.0)

        return 0.5 * recency + 0.3 * importance + 0.2 * relevance

    def retrieve_memories(self, context: str, k: int = 10, current_tick: int = 0) -> list[MemoryEntry]:
        """
        从记忆流中检索最相关的 k 条记忆，综合考虑时间衰减、重要性、关键词相关性。
        context：检索上下文字符串（如当前房间 + 时间 + 在场人物）
        """
        if not self.memories:
            return []
        # 分词并过滤常用虚词
        context_keywords = set(re.split(r'[，。：、\s]+', context)) - {'', '的', '了', '在', '是'}
        scored = [
            (self._score_memory(m, current_tick, context_keywords), m)
            for m in self.memories
        ]
        # 降序排列，取前 k 条
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:k]]

    def should_reflect(self) -> bool:
        """判断是否应触发反思：新记忆数量达到阈值且总记忆不少于 5 条"""
        return self._memories_since_reflection >= self.REFLECT_EVERY and len(self.memories) >= 5

    # ── Planning（规划，仿真开始前调用一次） ──────────────────────────────────

    async def plan_evening(self) -> None:
        """
        为角色生成今晚的大致心愿计划（3-5 条），在仿真正式开始前调用一次。
        结果存入 self.plan，后续每个 tick 的 build_prompt() 会将其注入提示词，
        引导角色行为更有方向感而非随机漫游。
        """
        # 随机抖动，避免 4 个角色同时发出 API 请求
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
                model=_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _strip_fences(msg.content[0].text)
            self.plan = json.loads(raw)
        except Exception:
            # 解析失败则保持空计划，角色将"随性而为"
            self.plan = []

    # ── Reflection（反思，每 REFLECT_EVERY 条新记忆触发） ─────────────────────

    async def reflect(self, sim_time: str, current_tick: int) -> list[str]:
        """
        对最近 15 条记忆进行高层次综合，提炼出 3 条洞察，
        并以 importance=9 的高分将洞察写回记忆流（带 [反思] 前缀）。
        反思完成后重置 _memories_since_reflection 计数器。
        返回洞察字符串列表（供 TickScheduler 广播到前端）。
        """
        # 取最近 15 条作为反思素材
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
                model=_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _strip_fences(msg.content[0].text)
            insights: list[str] = json.loads(raw)
            # 将洞察写入记忆流，重要性 9（仅次于最高）
            for insight in insights[:3]:
                self.add_memory(f"[反思] {insight}", importance=9, sim_time=sim_time, tick_num=current_tick)
            self._memories_since_reflection = 0  # 重置计数器
            return insights
        except Exception:
            self._memories_since_reflection = 0
            return []

    # ── Main decision（主决策，每 tick 调用） ────────────────────────────────

    def build_prompt(self, world_context: dict[str, Any], current_tick: int = 0) -> str:
        """
        构建发送给 LLM 的完整提示词，包含：
          - 角色性格 / 关系 / 说话风格 / 禁忌
          - 当前仿真时间
          - 今晚计划（plan_evening 生成）
          - 当前状态（位置、心情、目标、同房间人员）
          - 各人位置总览
          - 检索到的相关记忆
          - 整体情况描述
          - 当前房间道具 + 道具状态
          - 输出格式要求（JSON）
        """
        sim_time = world_context["sim_time"]
        room_occupants: dict[str, list[str]] = world_context["room_occupants"]
        world_summary: str = world_context["world_summary"]
        object_states: dict[str, dict[str, str]] = world_context.get("object_states", {})
        recent_event: str = world_context.get("recent_event", "")
        recent_dialogue: list[tuple[str, str, str]] = world_context.get("recent_dialogue", [])

        # 当前房间中的其他人
        others_here = [n for n in room_occupants.get(self.current_room, []) if n != self.name]
        others_str = "、".join(others_here) if others_here else "没有其他人"
        alone = len(others_here) == 0  # 独处时禁止填写对话

        # 各房间人员总览（只列出有人的房间）
        location_lines = []
        for room, occupants in room_occupants.items():
            if occupants:
                location_lines.append(f"  - {room}：{'、'.join(occupants)}")
        location_overview = "\n".join(location_lines) if location_lines else "  - 大家分散在各处"

        # 当前房间的道具描述 + 道具实时状态（如"红烧肉[炖煮中]"）
        room_props = ROOM_PROPS.get(self.current_room, "")
        room_obj_states = object_states.get(self.current_room, {})
        obj_state_str = "、".join(f"{k}[{v}]" for k, v in room_obj_states.items()) if room_obj_states else ""
        env_str = room_props
        if obj_state_str:
            env_str += f"\n- 当前状态：{obj_state_str}"

        # 检索相关记忆：以当前房间 + 时间 + 在场人物 + 整体情况为上下文
        context_for_retrieval = f"{self.current_room} {sim_time} {others_str} {world_summary}"
        retrieved = self.retrieve_memories(context_for_retrieval, k=10, current_tick=current_tick)
        memory_str = "\n".join(
            f"  [{m.time}] {m.event}" for m in retrieved
        ) if retrieved else "  （暂无记忆）"

        # 今晚计划（planning 阶段生成，如为空则提示随性而为）
        plan_str = "\n".join(f"  - {p}" for p in self.plan) if self.plan else "  （随性而为）"

        # 最近对话记录（当前房间的部分），防止重复
        room_recent = [
            f"  {spk}：「{line}」"
            for spk, line, rm in recent_dialogue
            if rm == self.current_room
        ]
        recent_dialogue_str = "\n".join(room_recent) if room_recent else ""

        return f"""你是{self.name}。

【世界背景】
{WORLD_BACKGROUND}

【性格与经历】{self.core_traits}
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
{f"【刚刚发生】{recent_event}" if recent_event else ""}
{f"【最近的对话（不要重复以下已经说过的话或问过的问题）】{chr(10)}{recent_dialogue_str}" if recent_dialogue_str else ""}
请决定接下来5分钟（到 {_next_time(sim_time)} ）你做什么。
动作要具体利用房间里的道具（比如：打开火锅底座、拿起Switch手柄、坐进藤椅看夜景）。
用JSON回复：
{{
  "思考": "内心想法（1-2句）",
  "目标房间": "房间名（院子/厨房/餐厅/游戏室/衣帽间/客厅/次阳台/卧室/卫生间 之一）",
  "动作": "具体动作，尽量与道具互动（1句）",
  "道具交互": "你改变了哪个道具的状态，格式 道具名:新状态，没有则填空字符串",
  "对话": "{'你独处，必须填空字符串' if alone else '主动说话，有话就说，沉默是少数情况（约20%），不说则填空字符串'}",
  "对话对象": "{'不适用' if alone else '说给谁听（写对方名字，或「所有人」）'}",
  "情绪": "当前情绪（1-3个字）"
}}
只回复JSON。"""

    async def think(self, world_context: dict[str, Any], current_tick: int = 0) -> dict[str, Any]:
        """
        主决策入口，每个 tick 被 TickScheduler 并发调用。
        加入 0-0.5s 随机抖动，将 4 个角色的 API 请求错开，
        降低同一秒内并发的峰值，也让角色行为更自然。
        调用完毕后更新 self.mood 和 self.current_goal。
        """
        await asyncio.sleep(random.uniform(0, 0.5))
        prompt = self.build_prompt(world_context, current_tick)
        client = self._get_client()
        message = await client.messages.create(
            model=_MODEL,
            max_tokens=450,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
        result = self.parse_response(raw)
        # 更新角色实时状态（心情 + 目标）
        self.mood = result.get("情绪", self.mood)
        self.current_goal = result.get("动作", self.current_goal)
        return result

    # ── Dialogue reaction（对话反应，同 tick 子轮次） ────────────────────────

    async def react_to_dialogue(
        self,
        speaker: str,
        dialogue: str,
        sim_time: str,
        target: str = "所有人",
        recent_event: str = "",
        history: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any] | None:
        """
        对话反应：当同房间有人说话时，决定是否开口。
        - target：说话的目标（对方名字 or "所有人"）；被点名者有更强回应压力
        - recent_event：当前叙事事件，注入提示词帮助角色想起话题
        - history：完整对话历史（最近6条），让角色了解对话上下文
        - 自动注入该角色最重要的3条记忆（importance>=7）
        """
        await asyncio.sleep(random.uniform(0, 0.3))

        # 注入高重要度记忆（importance >= 7），最多3条
        top_mems = sorted(
            [m for m in self.memories if m.importance >= 7],
            key=lambda m: -m.importance,
        )[:3]
        memory_hint = ""
        if top_mems:
            memory_hint = "\n【你记得的重要事情】\n" + "\n".join(
                f"  [{m.time}] {m.event}" for m in top_mems
            )

        # 对话历史（最近6条，让角色知道聊到哪了）
        history_hint = ""
        if history and len(history) > 1:
            history_hint = "\n【刚才的对话】\n" + "\n".join(
                f"  {spk}：「{line}」" for spk, line in history[-6:]
            )

        # 根据对话目标调整回应压力
        if target == self.name:
            target_hint = "这句话是直接对你说的，你需要回应。"
        elif target and target != "所有人":
            target_hint = f"这句话主要对{target}说，你也听到了，可以插话，也可以沉默。"
        else:
            target_hint = "说给大家听的，你可以回应，也可以沉默。"

        recent_hint = f"\n【刚刚发生的事】{recent_event}" if recent_event else ""

        prompt = f"""你是{self.name}。
【性格】{self.core_traits}
【说话风格】{self.speech_style}
【禁忌】{self.taboos}{memory_hint}{recent_hint}{history_hint}

现在是{sim_time}，{speaker}刚说：「{dialogue}」
{target_hint}

请做出回应（沉默也是合理的选择）。
只回复JSON：{{"回应": "你说的话（不回应填空字符串）", "动作": "你的动作（1句）", "情绪": "情绪（1-3字）"}}"""
        client = self._get_client()
        try:
            msg = await client.messages.create(
                model=_MODEL,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = _strip_fences(msg.content[0].text)
            return json.loads(raw)
        except Exception:
            return None

    def parse_response(self, raw: str) -> dict[str, Any]:
        """
        解析 LLM 返回的 JSON 字符串。
        - 先剥离 Markdown 代码块（```json ... ```）
        - 校验 "目标房间" 是否合法，非法则回退到当前房间
        - 若解析彻底失败，返回"发呆"的默认状态，保证程序不崩溃
        """
        raw = _strip_fences(raw)
        try:
            data = json.loads(raw)
            # 校验目标房间合法性
            valid_rooms = {"院子", "厨房", "餐厅", "游戏室", "衣帽间", "客厅", "次阳台", "卧室", "卫生间"}
            if data.get("目标房间") not in valid_rooms:
                data["目标房间"] = self.current_room
            return data
        except (json.JSONDecodeError, KeyError):
            # 兜底返回，维持当前状态
            return {
                "思考": "（解析失败）",
                "目标房间": self.current_room,
                "动作": "发呆",
                "道具交互": "",
                "对话": "",
                "情绪": self.mood,
            }


# ── Helpers（工具函数） ────────────────────────────────────────────────────────

def _strip_fences(raw: str) -> str:
    """去除 LLM 回复中可能包裹的 Markdown 代码块标记（```json / ```）"""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    return re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)


def _next_time(sim_time: str) -> str:
    """将仿真时间推进 5 分钟，用于 build_prompt 中显示「接下来 5 分钟到 XX:XX」"""
    h, m = map(int, sim_time.split(":"))
    m += 5
    if m >= 60:
        h += 1
        m -= 60
    return f"{h:02d}:{m:02d}"
