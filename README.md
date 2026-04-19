# 罗小黑战记 AI Town

《罗小黑战记》题材的 AI 角色仿真，基于 Stanford AI Town 架构。四个角色（小黑、无限、哪吒、鹿野）在除夕夜自主行动、对话、反思，人妖关系矛盾在晚饭中自然发酵。

![screenshot](docs/screenshot.png)

## 角色

| 角色 | 身份 | 性格 |
|------|------|------|
| 小黑 | 猫妖少年，无限徒弟 | 话少敏感，对失去家园的痛苦有直接感知 |
| 无限 | 人类，会馆最强执行者，师父 | 温柔沉着，见过长时间尺度的战争与秩序 |
| 哪吒 | 天界神灵，无限旧识 | 随性嘴硬，从规则与稳定局面的视角思考 |
| 鹿野 | 鹿妖，无限大弟子 | 沉稳内敛，战争遗孤，话少但每句有重量 |

## 架构

参考 [Stanford AI Town (Park et al. 2023)](https://arxiv.org/abs/2304.03442)：

```
Memory Stream → Reflection → Planning → Think (每 tick)
                                           ↓
                              Sequential Conversation（顺序对话，每句 10 秒）
```

- **Memory Stream**：每条记忆含时间戳、重要度（1-10）、tick 编号，用于时间衰减检索
- **Reflection**：每积累 10 条新记忆触发一次，提炼高层洞察写回记忆流（importance=9）
- **Planning**：仿真开始前每人生成今晚的心愿计划（3-5 条），持续影响行为
- **顺序对话**：每句话消耗 10 秒模拟时间，自然沉默则结束，最长 10 分钟；时钟按实际对话时长推进
- **叙事事件**：19:00 年夜饭聚集、19:10 新闻画面触发人妖关系讨论（小黑被强制开口）、20:30 烟花

## 技术栈

- **后端**：FastAPI + asyncio + WebSocket
- **LLM**：Claude Sonnet 4.6（Anthropic）
- **寻路**：A* 算法，9 个房间的有向图
- **前端**：HTML5 Canvas 2D，WebSocket 实时渲染

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 填入 ANTHROPIC_API_KEY

# 启动
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# 打开 http://localhost:8000
```

## 房间地图

```
院子 ── 厨房 ── 餐厅 ── 游戏室 ── 次阳台
                  │                    │
                客厅 ────────────────────
                  │
         卧室 ── 卫生间
           │
         衣帽间
```

## 界面操作

- **暂停 / 继续**：右上角按钮，随时停止 token 消耗
- **强制推进**：Force Tick 按钮，跳过当前等待
- 角色气泡显示当前对话；反思触发时显示金色 ★

## 项目结构

```
backend/
  agents.py        # Agent 类：Memory / Reflection / Planning / Think
  characters.py    # 角色配置 + 世界背景文本
  world.py         # WorldState + TickScheduler + 顺序对话逻辑
  pathfinding.py   # A* 寻路
  news.py          # 触发新闻事件的叙事模块
  main.py          # FastAPI + WebSocket
frontend/
  index.html       # Canvas 渲染 + WebSocket 客户端
tests/             # 23 个单元/集成测试
```

---

## 迭代过程

### 第一步：项目脚手架

建立目录结构、安装依赖（FastAPI、anthropic、pytest），确定模块划分边界。没有任何逻辑，只是让后续每一步有落脚点。

### 第二步：A* 寻路模块

实现 `pathfinding.py`，定义 9 个房间的有向图 `GRAPH`，A* 算法计算最短路径。这是角色能自由移动的基础，后续所有移动决策都依赖它。

### 第三步：角色配置

编写 `characters.py`，为四个角色分别定义静态配置（背景故事、与其他角色的关系、世界观）。这一步把 IP 设定"注入"到系统，是角色行为有个性的前提。

### 第四步：Agent 核心

实现 `Agent` 类，包含：
- **记忆流（Memory Stream）**：追加写入，每条记忆带时间戳和重要度
- **Prompt Builder**：把记忆、当前场景、角色设定拼装成 LLM prompt
- **`think()` 方法**：调用 Claude，返回行动意图（移动目标 + 说话内容）

### 第五步：个性字段结构化

发现把性格写成一段自由文本会让 LLM 忽略细节。把 `personality` 拆分为四个结构化字段：`core_traits`（核心特质）、`relationships`（对各角色的态度）、`speech_style`（说话风格）、`taboos`（绝对不会做的事）。角色区分度明显提升。

### 第六步：世界引擎

实现 `world.py`：
- `SimClock`：仿真时钟，每 tick = 1 分钟模拟时间
- `WorldState`：追踪所有角色当前位置、对话状态、房间内物品
- `TickScheduler`：asyncio 驱动的 tick 循环，按顺序调度每个 agent 行动

### 第七步：FastAPI + WebSocket 服务

实现 `main.py`，用 FastAPI lifespan 启动后台 tick 循环，WebSocket 端点向前端广播每 tick 的完整世界快照（角色位置、气泡文本、时间、事件日志）。

### 第八步：Canvas 前端

实现 `frontend/index.html`：
- HTML5 Canvas 绘制平面图（9 个房间 + 走廊连线）
- WebSocket 接收快照，实时更新角色位置
- 每个角色用文字 sprite 表示，旁边显示说话气泡

### 第九步：测试覆盖

补充 23 个测试，覆盖：A* 寻路正确性、记忆流写入/检索、角色配置完整性、WorldState 状态转换。全部绿灯后才进行后续重构。

### 第十步：修复并发对话 → 顺序对话

早期实现让同一个房间的多个角色同时开口，导致对话混乱、毫无沉默感。重构为**顺序对话（Sequential Conversation）**：同一房间每轮只有一人发言，下一人的回应是对上一句话的直接反应；自然沉默（LLM 返回空内容）结束对话，最长上限 10 分钟模拟时间。同时把角色 sprite 从文字替换为头像图形。

### 第十一步：修正房间图 + 家具 Props

发现房间连通图存在不合理边（厨房直连衣帽间）。删除错误边，重新核对所有 9 个房间的邻接关系。同时在每个房间加入家具/物品列表（`room_props`），注入到 prompt，让角色的行为描述更有空间感（"坐在沙发上"、"拿了一双筷子"）。

### 第十二步：Stanford AI Town 完整特性

完成剩余三个核心模块：

- **Planning**：仿真开始前，每个 agent 独立生成今晚的 3-5 条心愿/计划，持续作为 prompt 上下文
- **Reflection**：每积累 10 条新记忆，LLM 对记忆流做元认知总结，写回 importance=9 的高层洞察
- **相关性检索**：`think()` 时不再把全部记忆塞入 prompt，改为按"时间衰减 × 重要度 × 语义相关度"打分，只取 Top-K 条，解决 prompt 过长问题
- **对话 Reaction Sub-round**：叙事事件触发后，先广播事件，再给每个角色一次短路 reaction，使角色能立即对突发事件作出情绪反应

### 第十三步：位置控制软引导

遇到角色"乱跑"（该在餐厅吃饭时跑去卧室）的问题。

尝试过两种硬性方案均被放弃：
- "只允许移动到相邻房间"——破坏了角色自主性，角色被困在出发点附近
- "每 3 tick 强制归位"——机械感强，叙事连续性断裂

最终方案：在 `world.py` 维护 `POSITION_SCHEDULE`（时段 → 每人推荐房间），通过 `build_world_context()` 计算 `schedule_hints`，注入 prompt："这个时段你通常应该在：X（除非有明确叙事理由）"。依赖 LLM 自主遵守，关键叙事节点（年夜饭）才用 `gather_room` 强制聚集。

---

## 踩过的坑

| 问题 | 原因 | 解决方式 |
|------|------|----------|
| 角色同时开口，对话混乱 | 并发 think() | 改为顺序对话，每轮单人发言 |
| Reflection 过于频繁，角色变"说教机器" | 触发阈值太低 | 阈值调高到 10 条记忆，且只在 tick 末尾触发 |
| Prompt 过长导致上下文溢出 | 全量记忆注入 | 相关性打分 Top-K 检索 |
| 角色无视房间位置乱跑 | LLM 自由移动 | prompt hint 软性引导，不做机械强制 |
| 房间连通图有错误边 | 手工维护失误 | 逐条核查后删除不合理边 |
