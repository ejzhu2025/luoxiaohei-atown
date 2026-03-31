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
- 角色气泡显示当前对话；反思触发时显示金色 ★

## 项目结构

```
backend/
  agents.py        # Agent 类：Memory / Reflection / Planning / Think
  characters.py    # 角色配置 + 世界背景文本
  world.py         # WorldState + TickScheduler + 顺序对话逻辑
  pathfinding.py   # A* 寻路
  main.py          # FastAPI + WebSocket
frontend/
  index.html       # Canvas 渲染 + WebSocket 客户端
```
