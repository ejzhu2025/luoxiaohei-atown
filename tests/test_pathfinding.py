import pytest
from backend.pathfinding import find_path, ROOMS


def test_same_room_returns_single_element():
    path = find_path("客厅", "客厅")
    assert path == ["客厅"]


def test_adjacent_rooms_direct_path():
    path = find_path("客厅", "次阳台")
    assert path == ["客厅", "次阳台"]


def test_multi_hop_path():
    # 卫生间 → 衣帽间: 卫生间→卧室→衣帽间
    path = find_path("卫生间", "衣帽间")
    assert path == ["卫生间", "卧室", "衣帽间"]


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
