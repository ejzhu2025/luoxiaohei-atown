from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable, Coroutine

from backend.characters import CharacterConfig
from backend.agents import Agent
from backend.pathfinding import find_path

logger = logging.getLogger(__name__)

# ── 时段基准位置表 ─────────────────────────────────────────────────────────────
# 每个时段定义每个角色"应在"的房间。
# 若角色连续 3 tick 偏离基准，run() 会自动将其拉回。
POSITION_SCHEDULE: list[tuple[tuple[str, str], dict[str, str]]] = [
    (("19:00", "20:30"), {"无限": "餐厅", "小黑": "餐厅", "哪吒": "餐厅", "鹿野": "餐厅"}),
    (("20:30", "23:00"), {"无限": "客厅", "小黑": "客厅", "哪吒": "客厅", "鹿野": "次阳台"}),
    (("23:00", "24:00"), {"无限": "客厅", "小黑": "客厅", "哪吒": "客厅", "鹿野": "客厅"}),
]


def get_schedule_room(name: str, sim_time: str) -> str | None:
    """返回该角色在当前时段的基准位置，无匹配返回 None。"""
    try:
        h, m = map(int, sim_time.split(":"))
    except ValueError:
        return None
    total = h * 60 + m
    for (start, end), rooms in POSITION_SCHEDULE:
        sh, sm = map(int, start.split(":"))
        eh, em = map(int, end.split(":"))
        if sh * 60 + sm <= total < eh * 60 + em:
            return rooms.get(name)
    return None


# ── 每个房间里可交互道具的初始状态 ────────────────────────────────────────────
# 格式：{ 房间名: { 道具名: 当前状态 } }
# 角色每次 tick 可以改变这里的状态，下一轮 prompt 会把最新状态告诉所有角色
# ── 叙事触发事件表 ────────────────────────────────────────────────────────────
# 在特定仿真时间触发剧情节点：注入所有角色记忆 + 可选强制聚集到某房间
# gather_room=None 表示不强制移动，让角色自由反应
NARRATIVE_EVENTS: list[dict] = [
    {
        "sim_time": "19:00",
        "memory": "无限从厨房喊：「来吃饭了。」年夜饭正式开始。",
        "description": "年夜饭开始，无限喊大家来餐厅。",
        "gather_room": "餐厅",
        "importance": 8,
    },
    {
        "sim_time": "19:10",
        # 通用记忆：触发 trigger_speaker 之前先注入，作为"刚刚看到的事"
        "memory": "哪吒随手换台，新闻里一闪而过某处妖灵聚居地被清退的画面。",
        "description": "新闻里出现了妖灵聚居地被清退的画面。",
        "gather_room": None,
        "importance": 9,
        # 每人独立的内心反应记忆，检索时更容易浮现
        "character_memories": {
            "小黑": "电视里出现了妖灵聚居地被清退的画面——那片地方的样子和我以前住的地方有点像。心里像被什么堵了一下，很想说什么，又不知道从哪里开口。",
            "无限": "新闻里的妖灵清退画面，会馆最近处理的纠纷越来越多，这种事迟早要在桌上摊开谈。",
            "哪吒": "换台时看到妖灵被清退的新闻，不是第一次了。天界和人间都管不住这种事，会馆也管不住。",
            "鹿野": "那个画面让我想起小时候的事。手不自觉握了一下，又放开了。这种新年夜里播出来的东西，很难当作没看见。",
        },
        # 强制让小黑开口说出内心的话，触发人妖关系讨论
        "trigger_speaker": "小黑",
        "trigger_prompt": "你刚才在电视里看到了妖灵聚居地被清退的新闻画面，那个地方和你以前的家有点像。你心里堵得慌，忍不住开口了——说什么都行，哪怕只是一句感叹，但你一定要说出来。",
    },
    {
        "sim_time": "20:30",
        "memory": "窗外烟花断续响起，新年将近。",
        "description": "窗外烟花开始响起。",
        "gather_room": None,
        "importance": 6,
    },
    {
        "sim_time": "23:50",
        "memory": "距零点还有十分钟。电视里倒计时已经开始，院子里鞭炮零星作响。",
        "description": "距零点十分钟，倒计时开始。",
        "gather_room": None,
        "importance": 8,
        "trigger_speaker": "哪吒",
        "trigger_prompt": "再过十分钟就零点了，你想做点什么来迎接新年——点鞭炮、倒酒、或者说句什么都行，但这个时刻你不想就这么沉默地过去。",
    },
    {
        "sim_time": "00:00",
        "memory": "零点。窗外烟花齐放，震耳欲聋，整座城市亮了。新年到了。",
        "description": "零点，新年到了。",
        "gather_room": "院子",
        "importance": 10,
        "trigger_speaker": "无限",
        "trigger_prompt": "新年零点，烟花把院子照得通亮。你看着眼前这几个人——小黑、哪吒、鹿野——说一句新年的话，不用多，一句就够。",
    },
]


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


# ── SimClock：模拟时钟 ────────────────────────────────────────────────────────
# 把现实时间映射到模拟时间。每次 tick 推进 TICK_SIM_MINUTES 分钟。
# 模拟从 18:00 开始，到 23:00 结束（共 60 个 tick）。
class SimClock:
    def __init__(self, start_hour: int = 18, start_minute: int = 0):
        # 用"总分钟数"内部存储，方便计算
        self._total_minutes = start_hour * 60 + start_minute
        self._end_minutes = 24 * 60  # 24:00（零点）为模拟结束时间

    @property
    def current(self) -> str:
        """返回当前模拟时间，格式 HH:MM"""
        h, m = divmod(self._total_minutes, 60)
        return f"{h:02d}:{m:02d}"

    def advance(self, minutes: int) -> None:
        """推进模拟时间"""
        self._total_minutes += minutes

    def is_over(self) -> bool:
        """判断模拟是否已到终止时间"""
        return self._total_minutes >= self._end_minutes


# ── WorldState：世界快照 ──────────────────────────────────────────────────────
# 保存所有角色的位置、心情、动作、对话，以及各房间道具状态。
# 每次 tick 结束后更新，并通过 WebSocket 广播给前端。
class WorldState:
    def __init__(self, characters: list[CharacterConfig]):
        # 各角色当前所在房间
        self.agent_rooms: dict[str, str] = {c.name: c.initial_room for c in characters}
        # 各角色本 tick 的完整移动路径（用于前端动画插值）
        self.agent_paths: dict[str, list[str]] = {c.name: [c.initial_room] for c in characters}
        # 各角色当前心情
        self.agent_moods: dict[str, str] = {c.name: c.initial_mood for c in characters}
        # 各角色本 tick 说的话
        self.agent_dialogues: dict[str, str] = {c.name: "" for c in characters}
        # 各角色本 tick 正在做的动作
        self.agent_actions: dict[str, str] = {c.name: c.initial_goal for c in characters}
        # 各房间道具状态（深拷贝，避免污染初始值）
        import copy
        self.object_states: dict[str, dict[str, str]] = copy.deepcopy(INITIAL_OBJECT_STATES)
        # 当前叙事事件描述（触发后保留 4 个 tick，供 prompt 引用）
        self.recent_event: str = ""
        # 滚动对话窗口：最近 8 条发言（跨 tick 保留），格式 (说话者, 台词, 房间)
        self.recent_dialogue: list[tuple[str, str, str]] = []
        # 动态事件队列：用户注入或新闻映射的事件，fire_at="immediate" 表示下一 tick 触发
        self.pending_events: list[dict] = []

    def get_room_occupants(self) -> dict[str, list[str]]:
        """返回每个房间里当前有哪些角色"""
        result: dict[str, list[str]] = {}
        for name, room in self.agent_rooms.items():
            result.setdefault(room, []).append(name)
        return result

    def update_agent_room(self, name: str, room: str) -> None:
        """直接设置某角色的位置（不经寻路，用于初始化或强制移动）"""
        self.agent_rooms[name] = room

    def update_object_state(self, room: str, obj: str, state: str) -> None:
        """直接更新某房间某道具的状态"""
        if room in self.object_states:
            self.object_states[room][obj] = state

    def parse_and_apply_object_interaction(self, room: str, interaction: str) -> tuple[str, str] | None:
        """
        解析 LLM 返回的道具交互字段，格式为 '道具名:新状态'。
        成功则更新 object_states 并返回 (道具名, 新状态)，失败返回 None。
        """
        if not interaction or ":" not in interaction:
            return None
        parts = interaction.split(":", 1)
        obj, state = parts[0].strip(), parts[1].strip()
        if obj and state:
            self.object_states.setdefault(room, {})[obj] = state
            return obj, state
        return None

    def build_summary(self) -> str:
        """生成一段自然语言的世界状态摘要，放入每个角色的 prompt 里"""
        lines = ["当前各人状态："]
        for name, room in self.agent_rooms.items():
            mood = self.agent_moods.get(name, "")
            action = self.agent_actions.get(name, "")
            lines.append(f"  - {name} 在{room}，心情{mood}，正在{action}")
        return "\n".join(lines)

    def push_dialogue(self, speaker: str, line: str, room: str) -> None:
        """把一条发言追加到滚动对话窗口，超出8条时丢弃最旧的。"""
        self.recent_dialogue.append((speaker, line, room))
        if len(self.recent_dialogue) > 8:
            self.recent_dialogue.pop(0)

    def build_world_context(self, sim_time: str) -> dict[str, Any]:
        """
        组装每次 tick 传给所有 Agent 的共享上下文。
        包含：当前时间、房间占用情况、世界摘要、道具状态、最近对话、时段建议位置。
        """
        # 为每个角色查询当前时段的基准房间，注入 prompt 引导 LLM
        schedule_hints: dict[str, str] = {}
        for name in self.agent_rooms:
            hint = get_schedule_room(name, sim_time)
            if hint:
                schedule_hints[name] = hint
        return {
            "sim_time": sim_time,
            "room_occupants": self.get_room_occupants(),
            "world_summary": self.build_summary(),
            "object_states": self.object_states,
            "recent_event": self.recent_event,
            "recent_dialogue": list(self.recent_dialogue),
            "schedule_hints": schedule_hints,
        }

    def apply_decision(self, name: str, decision: dict[str, Any]) -> list[str]:
        """
        把 LLM 返回的决策应用到世界状态：
        1. 更新心情、对话、动作
        2. 解析并应用道具交互
        3. 用 A* 寻路，让角色朝目标房间移动一步
        返回完整的目标路径（供前端做移动动画）。
        """
        target = decision.get("目标房间", self.agent_rooms[name])
        self.agent_moods[name] = decision.get("情绪", self.agent_moods[name])
        dialogue = decision.get("对话", "")
        self.agent_dialogues[name] = dialogue
        self.agent_actions[name] = decision.get("动作", "")
        if dialogue:
            self.push_dialogue(name, dialogue, self.agent_rooms[name])

        # 道具交互：解析 "道具名:新状态" 格式并更新状态表
        interaction = decision.get("道具交互", "")
        if interaction:
            result = self.parse_and_apply_object_interaction(self.agent_rooms[name], interaction)
            if result:
                logger.info(f"{name} 改变了道具状态：{result[0]} → {result[1]}")

        # 寻路：直接移动到目标房间，保持对话和位置一致
        current = self.agent_rooms[name]
        path = find_path(current, target)
        if len(path) > 1:
            self.agent_rooms[name] = path[-1]
            self.agent_paths[name] = path
        else:
            self.agent_paths[name] = [current]
        return path


# ── TickScheduler：主循环调度器 ───────────────────────────────────────────────
# 控制整个模拟的节奏：
#   阶段0（Planning）→ 循环 tick（主决策 → 对话反应 → 反思检查 → 推进时钟）
class TickScheduler:
    TICK_REAL_SECONDS = 4   # 现实中每 tick 间隔 4 秒
    TICK_SIM_MINUTES = 5    # 模拟时间每 tick 推进 5 分钟

    def __init__(
        self,
        agents: list[Agent],
        world: WorldState,
        broadcast: Callable[[dict], Coroutine],
    ):
        self.agents = agents
        self.world = world
        self.broadcast = broadcast          # WebSocket 广播函数
        self.clock = SimClock(start_hour=19, start_minute=0)
        self._running = False
        self._force_tick = asyncio.Event()  # 前端点"快进"按钮时 set，跳过当前 sleep
        self._agent_map: dict[str, Agent] = {a.name: a for a in agents}
        self._fired_events: set[str] = set()  # 已触发的事件（用 sim_time 去重）
        self._event_clear_tick: int = 0     # 清除 recent_event 的 tick 编号
        self._paused = False                # 暂停标志
        self._resume_event = asyncio.Event()
        # 自动新闻注入状态
        self._last_news_tick = 0            # 上次拉新闻的 tick
        self._last_medium_tick = -99        # 上次注入中事件的 tick
        self._last_large_tick = -999        # 上次注入大事件的 tick
        self._large_event_count = 0         # 今晚已注入大事件数量
        self._resume_event.set()            # 默认运行中

    async def run_planning_phase(self) -> None:
        """
        阶段0：规划阶段。
        游戏开始前，4个角色并发调用 LLM 各自制定今晚的行程计划。
        计划存入 agent.plan，后续每个 tick 的 prompt 都会包含这份计划。
        """
        logger.info("规划阶段：各角色生成今晚计划...")
        await self.broadcast({"type": "planning_start"})
        results = await asyncio.gather(
            *[agent.plan_evening() for agent in self.agents],
            return_exceptions=True,
        )
        for agent, result in zip(self.agents, results):
            if isinstance(result, Exception):
                logger.error(f"{agent.name} 规划失败: {result}")
            else:
                plan_str = " / ".join(agent.plan[:3]) if agent.plan else "无计划"
                logger.info(f"{agent.name} 今晚计划: {plan_str}")
                await self.broadcast({
                    "type": "agent_plan",
                    "name": agent.name,
                    "plan": agent.plan,
                })

    async def _check_narrative_events(self, sim_time: str, tick_num: int) -> None:
        """检查当前仿真时间是否触发静态叙事事件（NARRATIVE_EVENTS 表），每个时间点只触发一次。"""
        for event in NARRATIVE_EVENTS:
            if event["sim_time"] != sim_time or sim_time in self._fired_events:
                continue
            self._fired_events.add(sim_time)
            await self._fire_event(event, sim_time, tick_num)

    async def _fire_event_trigger(
        self,
        event: dict,
        sim_time: str,
        tick_num: int,
    ) -> None:
        """
        如果事件含 trigger_speaker，让该角色第一个开口，
        然后跑一轮顺序对话。
        供 _check_narrative_events 和 _check_pending_events 共用。
        """
        trigger_name: str = event.get("trigger_speaker", "")
        if not trigger_name:
            return
        trigger_agent = self._agent_map.get(trigger_name)
        if not trigger_agent:
            return

        # 如果没有提供 trigger_prompt，自动从 memory 生成
        trigger_prompt: str = event.get(
            "trigger_prompt",
            f"刚刚发生了：{event.get('memory', '')}你是第一个注意到的，忍不住开口说了什么。",
        )

        logger.info(f"强制触发 {trigger_name} 就事件发言...")
        reaction = await trigger_agent.react_to_dialogue(
            speaker="环境",
            dialogue=trigger_prompt,
            sim_time=sim_time,
            target=trigger_name,
            recent_event=event.get("memory", ""),
        )
        if not (reaction and reaction.get("回应")):
            return

        line = reaction["回应"]
        mood = reaction.get("情绪", trigger_agent.mood)
        action = reaction.get("动作", "")
        room = trigger_agent.current_room

        if mood:
            trigger_agent.mood = mood
            self.world.agent_moods[trigger_name] = mood

        ev = f"{trigger_name}说：「{line}」"
        for other in self.agents:
            if other.current_room == room:
                other.add_memory(ev, importance=8, sim_time=sim_time, tick_num=tick_num)
        self.world.push_dialogue(trigger_name, line, room)

        logger.info(f"{trigger_name} 开口：{line}")
        await self.broadcast({
            "type": "agent_action",
            "name": trigger_name,
            "color": trigger_agent.color,
            "room": room,
            "path": [room],
            "action": action,
            "dialogue": line,
            "dialogue_target": "所有人",
            "mood": mood,
            "thought": "",
            "obj_interaction": "",
        })

        h, m = map(int, sim_time.split(":"))
        await self._run_sequential_conversation(
            room=room,
            initiator=trigger_name,
            opening_line=line,
            base_sim_seconds=h * 3600 + m * 60,
            initial_target="所有人",
            max_sim_minutes=3,
            tick_num=tick_num,
        )

    async def _fire_event(self, event: dict, sim_time: str, tick_num: int) -> None:
        """
        触发单个事件：注入所有角色记忆、可选强制聚集、设置 recent_event、广播、触发发言。
        """
        description = event.get("description", "")
        memory_text = event.get("memory", description)
        importance = event.get("importance", 7)

        logger.info(f"事件触发：{description}")

        # 注入记忆（支持 character_memories 个性化版本）
        char_mems: dict[str, str] = event.get("character_memories", {})
        for agent in self.agents:
            agent.add_memory(
                char_mems.get(agent.name, memory_text),
                importance=importance,
                sim_time=sim_time,
                tick_num=tick_num,
            )

        # 可选强制聚集
        if event.get("gather_room"):
            room = event["gather_room"]
            for agent in self.agents:
                self.world.update_agent_room(agent.name, room)
                agent.current_room = room
            logger.info(f"所有角色强制移动到 {room}")

        # 设置 recent_event（4 tick ≈ 20 分钟）
        self.world.recent_event = memory_text
        self._event_clear_tick = tick_num + 4

        await self.broadcast({
            "type": "narrative_event",
            "sim_time": sim_time,
            "description": description,
            "gather_room": event.get("gather_room"),
        })

        # 静态叙事事件立即触发发言；即时注入的动态事件（news/用户）跳过，
        # 让 recent_event 在下一 tick 的 think() 里自然触发反应，避免 tick 卡顿
        if not event.get("fire_at") == "immediate":
            await self._fire_event_trigger(event, sim_time, tick_num)

    async def _ambient_event_inject(self, tick_num: int, sim_time: str) -> bool:
        """
        基于当前场景状态生成环境触发事件，不依赖新闻 API。
        每 4 tick（20分钟）检查一次，随机选取一个符合条件的触发。
        返回 True 表示成功注入了一个事件。
        """
        import random

        AMBIENT_INTERVAL = 4
        if tick_num - getattr(self, '_last_ambient_tick', -99) < AMBIENT_INTERVAL:
            return False

        h, m = map(int, sim_time.split(":"))
        total_min = h * 60 + m
        occupants = self.world.get_room_occupants()
        obj = self.world.object_states

        candidates = []

        # ── 基于房间状态的触发 ─────────────────────────────────────────────────

        # 厨房：锅里有菜，飘出香味
        if obj.get("厨房", {}).get("红烧肉") not in ("未开始炖", None):
            if "厨房" in occupants or "餐厅" in occupants:
                candidates.append({
                    "description": "厨房飘出红烧肉的香味。",
                    "memory": "空气里隐约飘来红烧肉的香气，有些暖。",
                    "importance": 4,
                    "trigger_speaker": next(iter(occupants.get("餐厅", occupants.get("厨房", []))[:1]), None),
                    "trigger_prompt": "厨房里红烧肉的味道飘过来，你注意到了，随口说了一句。",
                })

        # 客厅：PS5 还在运行，有人路过
        if obj.get("客厅", {}).get("PS5", "").startswith("运行"):
            rooms_with_chars = [r for r in ("游戏室", "餐厅", "次阳台") if r in occupants]
            if rooms_with_chars:
                candidates.append({
                    "description": "客厅里游戏的音效从门缝里传出来。",
                    "memory": "隐约听见客厅那边游戏的音效，节奏很快。",
                    "importance": 3,
                })

        # 院子：鞭炮未点燃但到了晚些时候
        if total_min >= 22 * 60 and obj.get("院子", {}).get("鞭炮") == "未点燃":
            yard_chars = occupants.get("院子", []) + occupants.get("客厅", [])
            if yard_chars:
                candidates.append({
                    "description": "院子里的鞭炮还没点，哪吒盯着它看了一眼。",
                    "memory": "院子里那盘鞭炮安静地摆着，还没人去碰。",
                    "importance": 5,
                    "trigger_speaker": "哪吒" if "哪吒" in yard_chars else yard_chars[0],
                    "trigger_prompt": "院子里鞭炮还摆着没点，都快零点了，你忍不住发表了个意见。",
                })

        # ── 基于时间的触发 ─────────────────────────────────────────────────────

        # 除夕晚上8点：街上行人散尽的安静
        if 20 * 60 <= total_min < 20 * 60 + 10:
            candidates.append({
                "description": "窗外街道突然安静下来，除夕的夜越来越深。",
                "memory": "窗外的车声和人声都少了很多，整条街安静得出奇，除夕的感觉来了。",
                "importance": 5,
            })

        # 21点：对门邻居家传来喝酒划拳声
        if 21 * 60 <= total_min < 21 * 60 + 10:
            candidates.append({
                "description": "走廊里隐约传来对门邻居划拳喝酒的声音。",
                "memory": "走廊外传来一阵猜拳声，是对门邻居在热闹，和这边的安静形成对比。",
                "importance": 4,
                "trigger_speaker": "哪吒",
                "trigger_prompt": "走廊里传来对门邻居划拳喝酒的声音，你听了一会儿，说了句什么。",
            })

        # 22点：窗外开始有零星爆竹声
        if 22 * 60 <= total_min < 22 * 60 + 10:
            candidates.append({
                "description": "远处零星爆竹声开始响起，还有一个小时到零点。",
                "memory": "远处有人等不及了，零星的爆竹声先响起来，还早，但已经有年的味道了。",
                "importance": 5,
            })

        # 22:30：小黑打哈欠，无限注意到
        if 22 * 60 + 30 <= total_min < 22 * 60 + 40:
            if "小黑" in self.world.agent_rooms and "无限" in self.world.agent_rooms:
                if self.world.agent_rooms["小黑"] == self.world.agent_rooms["无限"]:
                    candidates.append({
                        "description": "小黑打了个哈欠，无限注意到了。",
                        "memory": "小黑打了个哈欠——都快十点半了，他还没睡，还硬撑着。",
                        "importance": 4,
                        "trigger_speaker": "无限",
                        "trigger_prompt": "小黑打了个哈欠，你注意到了，关心地说了句话。",
                    })

        # ── 基于角色互动的触发 ─────────────────────────────────────────────────

        # 鹿野独自在次阳台超过一段时间
        luye_room = self.world.agent_rooms.get("鹿野", "")
        if luye_room == "次阳台":
            room_others = [n for n in occupants.get("次阳台", []) if n != "鹿野"]
            if not room_others:  # 鹿野独自在阳台
                candidates.append({
                    "description": "鹿野一个人在次阳台，静静地看着夜景。",
                    "memory": "鹿野在阳台上站了一会儿，没人陪，也没说话。",
                    "importance": 4,
                    "trigger_speaker": "鹿野",
                    "trigger_prompt": "你一个人在阳台上看夜景，城市的灯火让你想到了什么，随口说了一句，不管有没有人听。",
                })

        # 小黑和哪吒同一个房间（哪吒爱逗小黑）
        xiaohei_room = self.world.agent_rooms.get("小黑", "")
        nezha_room = self.world.agent_rooms.get("哪吒", "")
        if xiaohei_room == nezha_room and xiaohei_room:
            candidates.append({
                "description": f"哪吒和小黑待在同一个房间，哪吒忍不住想逗逗他。",
                "memory": f"哪吒和小黑都在{xiaohei_room}，哪吒闲不住，总想找小黑说点什么。",
                "importance": 3,
                "trigger_speaker": "哪吒",
                "trigger_prompt": "你和小黑待在同一个房间，你觉得他挺有意思的，随口逗了他一句，语气轻松，不是挑衅。",
            })

        if not candidates:
            return False

        # 过滤掉没有 trigger_speaker 但文字太平淡的（importance≤3 且无 trigger_speaker 跳过）
        good = [c for c in candidates if c.get("trigger_speaker") or c.get("importance", 0) >= 5]
        pool = good if good else candidates

        chosen = random.choice(pool)
        # 没有 trigger_speaker 就用 fire_at immediate 静默注入
        if not chosen.get("trigger_speaker"):
            chosen["fire_at"] = "immediate"
        else:
            chosen["fire_at"] = "immediate"

        self.world.pending_events.append(chosen)
        self._last_ambient_tick = tick_num
        logger.info(f"环境事件注入：{chosen['description']}")
        return True

    async def _auto_news_inject(self, tick_num: int) -> None:
        """
        每 6 tick（30分钟游戏时间）自动拉一次新闻，按重要性分级注入：
          小事件 (importance≤5)：每次最多2个，无冷却
          中事件 (importance 6-7)：每次最多1个，12 tick（1小时）冷却
          大事件 (importance≥8)：每次最多1个，24 tick（2小时）冷却，整晚上限2个
        """
        NEWS_INTERVAL = 6           # 拉新闻间隔（tick）
        MEDIUM_COOLDOWN = 12        # 中事件冷却（tick）
        LARGE_COOLDOWN = 24         # 大事件冷却（tick）
        LARGE_NIGHT_CAP = 2         # 整晚大事件上限

        if tick_num - self._last_news_tick < NEWS_INTERVAL:
            return
        self._last_news_tick = tick_num

        try:
            from backend.news import fetch_headlines, translate_to_apartment_events
            headlines = await fetch_headlines()
            if not headlines:
                return
            events = await translate_to_apartment_events(headlines)
        except Exception as e:
            logger.warning(f"自动新闻拉取失败: {e}")
            return

        small, medium, large = [], [], []
        for ev in events:
            imp = ev.get("importance", 6)
            if imp >= 8:
                large.append(ev)
            elif imp >= 6:
                medium.append(ev)
            else:
                small.append(ev)

        to_inject = []

        # 每次只触发一个事件：优先级 大 > 中 > 小
        if (large
                and self._large_event_count < LARGE_NIGHT_CAP
                and (tick_num - self._last_large_tick) >= LARGE_COOLDOWN):
            to_inject.append(large[0])
            self._last_large_tick = tick_num
            self._large_event_count += 1
        elif medium and (tick_num - self._last_medium_tick) >= MEDIUM_COOLDOWN:
            to_inject.append(medium[0])
            self._last_medium_tick = tick_num
        elif small:
            to_inject.append(small[0])

        for ev in to_inject:
            self.world.pending_events.append(ev)

        if to_inject:
            logger.info(f"自动新闻注入 {len(to_inject)} 个事件：{[e.get('description') for e in to_inject]}")

    async def _check_pending_events(self, sim_time: str, tick_num: int) -> None:
        """
        检查动态事件队列，触发 fire_at=="immediate" 或时间匹配的事件。
        """
        if not self.world.pending_events:
            return

        to_fire, remaining = [], []
        for ev in self.world.pending_events:
            fire_at = ev.get("fire_at", "immediate")
            if fire_at == "immediate" or fire_at == sim_time:
                to_fire.append(ev)
            else:
                remaining.append(ev)

        self.world.pending_events = remaining
        for ev in to_fire:
            await self._fire_event(ev, sim_time, tick_num)

    async def _run_sequential_conversation(
        self,
        room: str,
        initiator: str,
        opening_line: str,
        base_sim_seconds: int,
        initial_target: str = "所有人",
        max_sim_minutes: int = 10,
        tick_num: int = 0,
    ) -> int:
        """
        在指定房间内运行顺序对话。时间轴：每句话消耗10秒模拟时间。

        流程：
          1. 向所有非上一发言者并发询问是否回应
          2. 取第一个非空回应作为本轮发言，时间 +10秒
          3. 无人回应 → 自然结束；超过 max_sim_minutes → 强制结束

        返回实际消耗的模拟秒数（供外层调整时钟推进量）。
        """
        max_seconds = max_sim_minutes * 60
        elapsed = 0
        history: list[tuple[str, str]] = [(initiator, opening_line)]
        last_speaker = initiator
        target = initial_target

        while elapsed < max_seconds:
            # 本轮话语的模拟时间戳
            elapsed += 10
            cur_sec = base_sim_seconds + elapsed
            sim_time = f"{cur_sec // 3600:02d}:{(cur_sec % 3600) // 60:02d}"

            # 找到房间内所有非上一发言者
            candidates = [
                a for a in self.agents
                if a.current_room == room and a.name != last_speaker
            ]
            if not candidates:
                break

            # 所有候选者并发决定是否开口，取第一个非空回应
            reactions = await asyncio.gather(
                *[
                    a.react_to_dialogue(
                        last_speaker,
                        history[-1][1],
                        sim_time,
                        target=target if a.name == target else "所有人",
                        recent_event=self.world.recent_event,
                        history=history,
                    )
                    for a in candidates
                ],
                return_exceptions=True,
            )

            next_agent = None
            next_reaction = None
            next_response = next_action = next_mood = ""
            for agent, reaction in zip(candidates, reactions):
                if isinstance(reaction, Exception) or not reaction:
                    continue
                resp = reaction.get("回应", "")
                if resp:
                    next_agent = agent
                    next_reaction = reaction
                    next_response = resp
                    next_action = reaction.get("动作", "")
                    next_mood = reaction.get("情绪", agent.mood)
                    break

            if not next_agent:
                break  # 沉默 → 对话自然结束

            if next_mood:
                next_agent.mood = next_mood
                self.world.agent_moods[next_agent.name] = next_mood

            # 应用移动（如果角色决定离开）
            valid_rooms = {"院子","厨房","餐厅","游戏室","衣帽间","客厅","次阳台","卧室","卫生间"}
            target_room = next_reaction.get("目标房间", "") if isinstance(next_reaction, dict) else ""
            if target_room and target_room in valid_rooms and target_room != next_agent.current_room:
                path = self.world.apply_decision(next_agent.name, {"目标房间": target_room})
                next_agent.current_room = self.world.agent_rooms[next_agent.name]
                await self.broadcast({
                    "type": "agent_action",
                    "name": next_agent.name,
                    "color": next_agent.color,
                    "room": next_agent.current_room,
                    "path": path,
                    "action": next_action,
                    "dialogue": "",
                    "dialogue_target": "",
                    "mood": next_mood,
                    "thought": "",
                    "obj_interaction": "",
                })

            ev = f"{next_agent.name}说：「{next_response}」"
            for other in self.agents:
                if other.current_room == room:
                    other.add_memory(ev, importance=7, sim_time=sim_time, tick_num=tick_num)
            self.world.push_dialogue(next_agent.name, next_response, room)

            await self.broadcast({
                "type": "agent_reaction",
                "name": next_agent.name,
                "color": next_agent.color,
                "room": next_agent.current_room,
                "dialogue": next_response,
                "action": next_action,
                "mood": next_mood,
                "in_reply_to": last_speaker,
                "sim_time": sim_time,
            })

            history.append((next_agent.name, next_response))
            last_speaker = next_agent.name
            target = "所有人"  # 首轮后开放给所有人

        logger.info(f"对话结束 [{room}]：{len(history)}轮，消耗{elapsed}秒模拟时间")
        return elapsed

    async def run(self) -> None:
        """
        主循环入口。流程：
          1. 规划阶段（一次性）
          2. 反复执行 tick，直到模拟时间到 23:00 或手动停止
        """
        self._running = True

        # 阶段0：规划
        await self.run_planning_phase()

        tick_num = 0
        while self._running and not self.clock.is_over():
            tick_num += 1
            sim_time = self.clock.current
            logger.info(f"Tick {tick_num} | 模拟时间={sim_time}")

            # 暂停：等待 resume 信号后再继续
            await self._resume_event.wait()

            await self.broadcast({"type": "tick_start", "tick": tick_num, "sim_time": sim_time})

            try:
                # 清除过期的叙事事件描述
                if self._event_clear_tick and tick_num >= self._event_clear_tick:
                    self.world.recent_event = ""

                # 检查并触发叙事事件（静态时间表）
                await self._check_narrative_events(sim_time, tick_num)

                # 检查并触发动态事件（用户注入 / 新闻映射）
                await self._check_pending_events(sim_time, tick_num)

                # 环境事件注入（基于场景/时间/角色位置，不依赖新闻）
                ambient_fired = await self._ambient_event_inject(tick_num, sim_time)

                # 自动新闻注入（按等级分频率）；若环境事件刚刚触发，跳过新闻，避免同 tick 两个事件
                if not ambient_fired:
                    await self._auto_news_inject(tick_num)

                # 构建本 tick 的共享世界上下文（所有 agent 共用同一份快照）
                world_context = self.world.build_world_context(sim_time)

                # ── 主决策轮：4个角色并发调用 LLM ───────────────────────────────
                decisions = await asyncio.gather(
                    *[agent.think(world_context, tick_num) for agent in self.agents],
                    return_exceptions=True,
                )

                # ── 应用决策，收集本 tick 事件 ────────────────────────────────────
                spoke_in_room: dict[str, tuple[str, str, str]] = {}
                tick_events = []

                for agent, decision in zip(self.agents, decisions):
                    if isinstance(decision, Exception):
                        logger.error(f"{agent.name} think() 失败: {decision}")
                        continue

                    path = self.world.apply_decision(agent.name, decision)
                    new_room = self.world.agent_rooms[agent.name]

                    dialogue = decision.get("对话", "")
                    target = decision.get("对话对象", "所有人") or "所有人"
                    if dialogue and new_room in spoke_in_room:
                        dialogue = ""
                    elif dialogue:
                        spoke_in_room[new_room] = (agent.name, dialogue, target)

                    tick_events.append({
                        "type": "agent_action",
                        "name": agent.name,
                        "color": agent.color,
                        "room": new_room,
                        "path": path,
                        "action": decision.get("动作", ""),
                        "dialogue": dialogue,
                        "dialogue_target": target if dialogue else "",
                        "mood": decision.get("情绪", ""),
                        "thought": decision.get("思考", ""),
                        "obj_interaction": decision.get("道具交互", ""),
                    })

                    if decision.get("动作"):
                        ev = f"{agent.name}在{new_room}：{decision['动作']}"
                        for other in self.agents:
                            if other.current_room == new_room:
                                other.add_memory(ev, importance=5, sim_time=sim_time, tick_num=tick_num)
                    if dialogue:
                        to_str = f"（对{target}）" if target and target != "所有人" else ""
                        ev = f"{agent.name}{to_str}说：「{dialogue}」"
                        for other in self.agents:
                            if other.current_room == new_room:
                                other.add_memory(ev, importance=7, sim_time=sim_time, tick_num=tick_num)

                    agent.current_room = new_room

                for event in tick_events:
                    await self.broadcast(event)

                # ── 顺序对话：每句话消耗10秒模拟时间，各房间并发进行 ───────────────
                conv_seconds_list: list[int] = []
                if spoke_in_room:
                    h, m = map(int, sim_time.split(":"))
                    base_sec = h * 3600 + m * 60
                    conv_results = await asyncio.gather(
                        *[
                            self._run_sequential_conversation(
                                room, spk, dlg, base_sec, tgt, max_sim_minutes=3, tick_num=tick_num
                            )
                            for room, (spk, dlg, tgt) in spoke_in_room.items()
                        ],
                        return_exceptions=True,
                    )
                    conv_seconds_list = [r for r in conv_results if isinstance(r, int)]

                # ── 反思检查 ─────────────────────────────────────────────────
                reflecting = [a for a in self.agents if a.should_reflect()]
                if reflecting:
                    reflect_results = await asyncio.gather(
                        *[a.reflect(sim_time, tick_num) for a in reflecting],
                        return_exceptions=True,
                    )
                    for agent, insights in zip(reflecting, reflect_results):
                        if isinstance(insights, list) and insights:
                            logger.info(f"{agent.name} 反思：{insights[0]}")
                            await self.broadcast({
                                "type": "agent_reflection",
                                "name": agent.name,
                                "insights": insights,
                                "sim_time": sim_time,
                            })

                # ── 推进模拟时钟并广播世界快照 ───────────────────────────────────
                max_conv_sec = max(conv_seconds_list, default=0)
                advance_min = max(self.TICK_SIM_MINUTES, (max_conv_sec + 59) // 60)
                self.clock.advance(advance_min)
                await self.broadcast({
                    "type": "world_update",
                    "sim_time": self.clock.current,
                    "positions": dict(self.world.agent_rooms),
                    "moods": dict(self.world.agent_moods),
                    "object_states": self.world.object_states,
                })

            except Exception as e:
                logger.error(f"Tick {tick_num} 异常，跳过本轮继续运行: {e}", exc_info=True)
                self.clock.advance(self.TICK_SIM_MINUTES)

            # ── 等待下一个 tick ───────────────────────────────────────────────
            # 正常等待 4 秒；若前端点了"快进"按钮则立刻跳过
            self._force_tick.clear()
            try:
                await asyncio.wait_for(self._force_tick.wait(), timeout=self.TICK_REAL_SECONDS)
            except asyncio.TimeoutError:
                pass  # 正常超时，进入下一 tick

        await self.broadcast({"type": "simulation_end", "sim_time": self.clock.current})
        self._running = False

    def force_tick(self) -> None:
        """前端点击"快进"按钮时调用，立刻跳过当前等待进入下一 tick"""
        self._force_tick.set()

    def pause(self) -> None:
        """暂停模拟（当前 tick 完成后停止，不再消耗 token）"""
        self._paused = True
        self._resume_event.clear()

    def resume(self) -> None:
        """恢复模拟"""
        self._paused = False
        self._resume_event.set()

    def stop(self) -> None:
        """停止模拟循环"""
        self._running = False
