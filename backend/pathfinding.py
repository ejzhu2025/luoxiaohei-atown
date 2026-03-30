from __future__ import annotations
import heapq

# Room graph: undirected edges with uniform weight 1
GRAPH: dict[str, list[str]] = {
    "院子":   ["厨房", "餐厅", "游戏室"],
    "厨房":   ["院子", "餐厅", "衣帽间"],
    "餐厅":   ["院子", "厨房", "游戏室", "客厅"],
    "游戏室": ["院子", "餐厅", "次阳台"],
    "衣帽间": ["厨房", "客厅", "卧室"],
    "客厅":   ["餐厅", "衣帽间", "次阳台", "卧室"],
    "次阳台": ["游戏室", "客厅"],
    "卧室":   ["衣帽间", "客厅", "卫生间"],
    "卫生间": ["卧室"],
}

ROOMS = set(GRAPH.keys())


def find_path(start: str, goal: str) -> list[str]:
    """Return shortest room path from start to goal using A* (uniform cost here)."""
    if start not in GRAPH:
        raise ValueError(f"Unknown room: {start!r}")
    if goal not in GRAPH:
        raise ValueError(f"Unknown room: {goal!r}")
    if start == goal:
        return [start]

    # (cost, current, path)
    heap: list[tuple[int, str, list[str]]] = [(0, start, [start])]
    visited: set[str] = set()

    while heap:
        cost, current, path = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return path
        for neighbor in GRAPH[current]:
            if neighbor not in visited:
                heapq.heappush(heap, (cost + 1, neighbor, path + [neighbor]))

    raise RuntimeError(f"No path from {start!r} to {goal!r}")  # should never happen
