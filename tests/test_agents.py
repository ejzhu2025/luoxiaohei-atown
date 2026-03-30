import pytest
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
    assert agent.core_traits
    assert agent.speech_style
    assert agent.taboos


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
    for i in range(20):
        agent.add_memory(f"事件{i}", importance=5, sim_time="18:00")
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
    assert "哪吒" in prompt


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
    assert result["目标房间"] == agent.current_room
    assert "动作" in result


def test_parse_response_strips_markdown_fences():
    agent = make_agent("小黑")
    raw = '```json\n{"思考": "test", "目标房间": "客厅", "动作": "坐着", "对话": "", "情绪": "平静"}\n```'
    result = agent.parse_response(raw)
    assert result["目标房间"] == "客厅"
