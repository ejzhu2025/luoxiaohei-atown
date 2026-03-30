import pytest
from backend.pathfinding import find_path, ROOMS


def test_same_room_returns_single_element():
    path = find_path("客厅", "客厅")
    assert path == ["客厅"]


def test_adjacent_rooms_direct_path():
    path = find_path("客厅", "次阳台")
    assert path == ["客厅", "次阳台"]


def test_multi_hop_path():
    # 卫生间 → 衣帽间: 卫生间→卧室→衣帽间 (direct, since 衣帽间 now connects to 卧室 not 厨房)
    path = find_path("卫生间", "衣帽间")
    assert path == ["卫生间", "卧室", "衣帽间"]

def test_kitchen_not_connected_to_wardrobe():
    # 厨房 and 衣帽间 are no longer directly connected
    path = find_path("厨房", "衣帽间")
    assert "衣帽间" in path
    assert len(path) > 2  # must go through at least one intermediate room


def test_long_path():
    # 卫生间 → 院子: should be reachable
    path = find_path("卫生间", "院子")
    assert len(path) <= 6
    assert path[0] == "卫生间"
    assert path[-1] == "院子"


def test_invalid_room_raises():
    with pytest.raises(ValueError):
        find_path("不存在的房间", "客厅")


def test_all_rooms_reachable():
    for src in ROOMS:
        for dst in ROOMS:
            path = find_path(src, dst)
            assert path[0] == src
            assert path[-1] == dst
