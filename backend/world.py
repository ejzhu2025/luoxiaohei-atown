from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable, Coroutine

from backend.characters import CharacterConfig
from backend.agents import Agent
from backend.pathfinding import find_path

logger = logging.getLogger(__name__)

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
        self._end_minutes = 23 * 60  # 23:00 为模拟结束时间

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
        包含：当前时间、房间占用情况、世界摘要、道具状态、最近对话。
        """
        return {
            "sim_time": sim_time,
            "room_occupants": self.get_room_occupants(),
            "world_summary": self.build_summary(),
            "object_states": self.object_states,
            "recent_event": self.recent_event,
            "recent_dialogue": list(self.recent_dialogue),
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

        # 寻路：每次只移动一个房间（不允许瞬移）
        current = self.agent_rooms[name]
        path = find_path(current, target)
        if len(path) > 1:
            # 移动到路径中的下一个房间
            self.agent_rooms[name] = path[1]
            self.agent_paths[name] = path
        else:
            # 已在目标房间，原地活动
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
        self.clock = SimClock(start_hour=18, start_minute=55)
        self._running = False
        self._force_tick = asyncio.Event()  # 前端点"快进"按钮时 set，跳过当前 sleep
        self._agent_map: dict[str, Agent] = {a.name: a for a in agents}
        self._fired_events: set[str] = set()  # 已触发的事件（用 sim_time 去重）
        self._event_clear_tick: int = 0     # 清除 recent_event 的 tick 编号
        self._paused = False                # 暂停标志
        self._resume_event = asyncio.Event()
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
        """检查当前仿真时间是否触发叙事事件，若触发则注入记忆并可选强制聚集。"""
        for event in NARRATIVE_EVENTS:
            if event["sim_time"] != sim_time or sim_time in self._fired_events:
                continue
            self._fired_events.add(sim_time)
            logger.info(f"叙事事件触发 [{sim_time}]：{event['description']}")

            # 注入记忆：优先使用 character_memories（每人独立），否则用通用 memory
            char_mems: dict[str, str] = event.get("character_memories", {})
            for agent in self.agents:
                mem_text = char_mems.get(agent.name, event["memory"])
                agent.add_memory(
                    mem_text,
                    importance=event["importance"],
                    sim_time=sim_time,
                    tick_num=tick_num,
                )

            # 可选：强制聚集到指定房间
            if event.get("gather_room"):
                room = event["gather_room"]
                for agent in self.agents:
                    self.world.update_agent_room(agent.name, room)
                    agent.current_room = room
                logger.info(f"所有角色强制移动到 {room}")

            # 设置 recent_event（保留 4 个 tick ≈ 20 分钟模拟时间）
            self.world.recent_event = event["memory"]
            self._event_clear_tick = tick_num + 4

            # 广播给前端
            await self.broadcast({
                "type": "narrative_event",
                "sim_time": sim_time,
                "description": event["description"],
                "gather_room": event.get("gather_room"),
            })

            # 强制触发发言：让指定角色立刻对事件开口，带动其他人讨论
            trigger_name: str = event.get("trigger_speaker", "")
            trigger_prompt: str = event.get("trigger_prompt", "")
            if trigger_name and trigger_prompt:
                trigger_agent = self._agent_map.get(trigger_name)
                if trigger_agent:
                    logger.info(f"强制触发 {trigger_name} 就事件发言...")
                    reaction = await trigger_agent.react_to_dialogue(
                        speaker="内心",
                        dialogue=trigger_prompt,
                        sim_time=sim_time,
                        target=trigger_name,   # 指向自己 → 必须开口
                        recent_event=event["memory"],
                    )
                    if reaction and reaction.get("回应"):
                        line = reaction["回应"]
                        mood = reaction.get("情绪", trigger_agent.mood)
                        action = reaction.get("动作", "")
                        if mood:
                            trigger_agent.mood = mood
                            self.world.agent_moods[trigger_name] = mood
                        # 写入同房间所有人的记忆
                        ev = f"{trigger_name}说：「{line}」"
                        for other in self.agents:
                            if other.current_room == trigger_agent.current_room:
                                other.add_memory(ev, importance=8, sim_time=sim_time, tick_num=tick_num)
                        logger.info(f"{trigger_name} 开口：{line}")
                        # 广播这句话
                        await self.broadcast({
                            "type": "agent_action",
                            "name": trigger_name,
                            "color": trigger_agent.color,
                            "room": trigger_agent.current_room,
                            "path": [trigger_agent.current_room],
                            "action": action,
                            "dialogue": line,
                            "dialogue_target": "所有人",
                            "mood": mood,
                            "thought": "",
                            "obj_interaction": "",
                        })
                        # 立刻开启顺序对话，让同房间其他人依次接话
                        h, m = map(int, sim_time.split(":"))
                        base_sec = h * 3600 + m * 60
                        await self._run_sequential_conversation(
                            room=trigger_agent.current_room,
                            initiator=trigger_name,
                            opening_line=line,
                            base_sim_seconds=base_sec,
                            initial_target="所有人",
                            tick_num=tick_num,
                        )

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
            next_response = next_action = next_mood = ""
            for agent, reaction in zip(candidates, reactions):
                if isinstance(reaction, Exception) or not reaction:
                    continue
                resp = reaction.get("回应", "")
                if resp:
                    next_agent = agent
                    next_response = resp
                    next_action = reaction.get("动作", "")
                    next_mood = reaction.get("情绪", agent.mood)
                    break

            if not next_agent:
                break  # 沉默 → 对话自然结束

            if next_mood:
                next_agent.mood = next_mood
                self.world.agent_moods[next_agent.name] = next_mood

            ev = f"{next_agent.name}说：「{next_response}」"
            for other in self.agents:
                if other.current_room == room:
                    other.add_memory(ev, importance=7, sim_time=sim_time, tick_num=tick_num)
            self.world.push_dialogue(next_agent.name, next_response, room)

            await self.broadcast({
                "type": "agent_reaction",
                "name": next_agent.name,
                "color": next_agent.color,
                "room": room,
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

            # 清除过期的叙事事件描述
            if self._event_clear_tick and tick_num >= self._event_clear_tick:
                self.world.recent_event = ""

            # 检查并触发叙事事件
            await self._check_narrative_events(sim_time, tick_num)

            # 构建本 tick 的共享世界上下文（所有 agent 共用同一份快照）
            world_context = self.world.build_world_context(sim_time)

            # ── 主决策轮：4个角色并发调用 LLM ───────────────────────────────
            # asyncio.gather 同时发起 4 个 HTTP 请求，谁先回来谁先处理
            decisions = await asyncio.gather(
                *[agent.think(world_context, tick_num) for agent in self.agents],
                return_exceptions=True,
            )

            # ── 应用决策，收集本 tick 事件 ────────────────────────────────────
            # spoke_in_room：记录每个房间本 tick 谁开口说话了
            # 规则：同一房间同一 tick 只允许一人说话（避免多人同时开口）
            # 格式：room → (说话者, 台词, 对话对象)
            spoke_in_room: dict[str, tuple[str, str, str]] = {}
            tick_events = []

            for agent, decision in zip(self.agents, decisions):
                if isinstance(decision, Exception):
                    logger.error(f"{agent.name} think() 失败: {decision}")
                    continue

                # 把决策写入世界状态（移动、心情、道具），返回寻路路径
                path = self.world.apply_decision(agent.name, decision)
                new_room = self.world.agent_rooms[agent.name]

                dialogue = decision.get("对话", "")
                target = decision.get("对话对象", "所有人") or "所有人"
                # 同房间已有人说话 → 本角色本轮静默
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

                # 把本 tick 的动作和对话写入同房间所有角色的记忆
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

                # 同步 agent 内部的 current_room（用于下一轮 prompt 构建）
                agent.current_room = new_room

            # 把主决策事件广播给前端
            for event in tick_events:
                await self.broadcast(event)

            # ── 顺序对话：每句话消耗10秒模拟时间，各房间并发进行 ───────────────
            # 不同房间的对话同时进行（asyncio并发），同一房间内严格顺序
            conv_seconds_list: list[int] = []
            if spoke_in_room:
                h, m = map(int, sim_time.split(":"))
                base_sec = h * 3600 + m * 60
                conv_results = await asyncio.gather(
                    *[
                        self._run_sequential_conversation(
                            room, spk, dlg, base_sec, tgt, tick_num=tick_num
                        )
                        for room, (spk, dlg, tgt) in spoke_in_room.items()
                    ],
                    return_exceptions=True,
                )
                conv_seconds_list = [r for r in conv_results if isinstance(r, int)]

            # ── 反思检查：积累了足够多的新记忆后触发反思 ─────────────────────
            # 反思由 LLM 将近期记忆归纳为更高层的洞察，importance=9 写回记忆流
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
            # 至少推进 TICK_SIM_MINUTES 分钟；若对话更长则按实际时长推进
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
