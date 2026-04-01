# 罗小黑战记 AI Town

《罗小黑战记》IP的AI角色仿真，基于Stanford AI Town架构，四个角色在除夕夜自主行动对话。

## 角色设定
| 角色 | 身份 | 性格 |
|------|------|------|
| 小黑 | 猫妖少年，无限徒弟 | 话少敏感，对失去家园有直接感知 |
| 无限 | 人类，会馆最强执行者 | 温柔沉着，见过长时间尺度的战争 |
| 哪吒 | 天界神灵 | 随性嘴硬，从规则稳定的视角思考 |
| 鹿野 | 鹿妖，无限大弟子 | 沉稳内敛，战争遗孤 |

## 核心叙事
人妖关系矛盾在除夕年夜饭中自然发酵：
- 19:00 年夜饭聚集
- 19:10 新闻画面触发人妖关系讨论（小黑被强制开口）
- 20:30 烟花

## 技术栈
- **后端**: FastAPI + asyncio + WebSocket
- **LLM**: Claude Sonnet 4.6
- **寻路**: A*算法，9个房间有向图
- **前端**: HTML5 Canvas 2D + WebSocket实时渲染

## 架构（Stanford AI Town）
```
Memory Stream → Reflection → Planning → Think（每tick）
                                          ↓
                             Sequential Conversation（顺序对话，每句10秒）
```
- Memory Stream：每条记忆含时间戳、重要度(1-10)、tick编号，时间衰减检索
- Reflection：积累10条新记忆触发，提炼高层洞察写回记忆流（importance=9）
- Planning：仿真开始前每人生成今晚心愿计划(3-5条)
- 顺序对话：每句消耗10秒模拟时间，自然沉默则结束，最长10分钟

## 启动
```bash
pip install -r requirements.txt
cp .env.example .env  # 填入ANTHROPIC_API_KEY
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# 访问 http://localhost:8000
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

## 踩过的坑
- 顺序对话（sequential conversation）比并发对话更自然，避免角色同时开口的混乱
- Reflection要控制触发频率，太频繁会导致角色行为过于"理性化"，失去自然感
