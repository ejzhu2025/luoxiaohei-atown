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


def test_sim_clock_is_over_at_23():
    clock = SimClock(start_hour=22, start_minute=55)
    assert not clock.is_over()
    clock.advance(5)
    assert clock.current == "23:00"
    assert clock.is_over()


def test_sim_clock_not_over_before_23():
    clock = SimClock(start_hour=22, start_minute=50)
    assert not clock.is_over()
    clock.advance(5)
    assert clock.current == "22:55"
    assert not clock.is_over()


def test_apply_decision_moves_one_room():
    world = make_world()
    # 小黑 is in 餐厅, wants to go to 衣帽间 (2 hops: via 厨房 or 客厅)
    decision = {"目标房间": "衣帽间", "情绪": "好奇", "对话": "", "动作": "走路"}
    path = world.apply_decision("小黑", decision)
    # Should move exactly one step (to an adjacent room of 餐厅)
    adjacent_to_dining = {"院子", "厨房", "游戏室", "客厅"}
    assert world.agent_rooms["小黑"] in adjacent_to_dining
    assert path[0] == "餐厅"
    assert path[-1] == "衣帽间"


def test_apply_decision_stay_in_room():
    world = make_world()
    decision = {"目标房间": "厨房", "情绪": "专注", "对话": "好香", "动作": "搅拌锅"}
    path = world.apply_decision("无限", decision)
    assert world.agent_rooms["无限"] == "厨房"
    assert path == ["厨房"]
