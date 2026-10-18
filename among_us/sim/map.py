"""A reduced Skeld: rooms, corridors and task locations.

Vision is the room a player stands in. Moving between two rooms is one hop per
tick along the shortest corridor path, so a walk is observable from both ends.
"""

from __future__ import annotations

from collections import deque

ROOMS: tuple[str, ...] = (
    "cafeteria", "weapons", "o2", "navigation", "shields", "communications",
    "storage", "admin", "electrical", "lower_engine", "security", "reactor",
    "upper_engine", "medbay",
)

CORRIDORS: tuple[tuple[str, str], ...] = (
    ("cafeteria", "weapons"), ("cafeteria", "medbay"), ("cafeteria", "admin"),
    ("cafeteria", "upper_engine"), ("cafeteria", "storage"),
    ("weapons", "o2"), ("o2", "navigation"), ("navigation", "shields"),
    ("weapons", "navigation"), ("shields", "communications"), ("shields", "storage"),
    ("communications", "storage"), ("storage", "admin"), ("storage", "electrical"),
    ("electrical", "lower_engine"), ("lower_engine", "security"), ("lower_engine", "reactor"),
    ("security", "reactor"), ("security", "upper_engine"), ("reactor", "upper_engine"),
    ("upper_engine", "medbay"),
)

ADJACENCY: dict[str, frozenset[str]] = {
    room: frozenset(
        {b for a, b in CORRIDORS if a == room} | {a for a, b in CORRIDORS if b == room}
    )
    for room in ROOMS
}

#: task id → room. Each crewmate is dealt a seed-determined subset.
TASKS: dict[str, str] = {
    "fix_wiring_electrical": "electrical",
    "calibrate_distributor": "electrical",
    "swipe_card": "admin",
    "upload_data_admin": "admin",
    "clear_asteroids": "weapons",
    "clean_o2_filter": "o2",
    "chart_course": "navigation",
    "prime_shields": "shields",
    "download_data_comms": "communications",
    "fuel_engines": "storage",
    "align_lower_engine": "lower_engine",
    "align_upper_engine": "upper_engine",
    "start_reactor": "reactor",
    "submit_scan": "medbay",
    "inspect_sample": "medbay",
    "empty_garbage": "cafeteria",
}

#: Vent connections between rooms, impostors only.
VENTS: tuple[tuple[str, str], ...] = (
    ("electrical", "security"), ("security", "medbay"), ("medbay", "electrical"),
    ("upper_engine", "reactor"), ("reactor", "lower_engine"),
    ("weapons", "navigation"), ("navigation", "shields"),
    ("admin", "cafeteria"),
)

#: Critical sabotages and the room where the crew fixes them.
SABOTAGE_ROOMS: dict[str, str] = {"reactor": "reactor", "o2": "o2"}

EMERGENCY_BUTTON_ROOM = "cafeteria"
SPAWN_ROOM = "cafeteria"


def next_hop(start: str, goal: str) -> str | None:
    """The first room on a shortest path, ties broken alphabetically."""

    if start == goal:
        return None
    previous: dict[str, str] = {start: start}
    queue = deque([start])
    while queue:
        room = queue.popleft()
        for neighbour in sorted(ADJACENCY[room]):
            if neighbour in previous:
                continue
            previous[neighbour] = room
            if neighbour == goal:
                step = goal
                while previous[step] != start:
                    step = previous[step]
                return step
            queue.append(neighbour)
    return None


def vent_reachable(start: str) -> list[str]:
    """Rooms an impostor can reach through the vents from ``start``, sorted."""

    reachable: set[str] = set()
    frontier = [start]
    while frontier:
        room = frontier.pop()
        for a, b in VENTS:
            for here, there in ((a, b), (b, a)):
                if here == room and there not in reachable and there != start:
                    reachable.add(there)
                    frontier.append(there)
    return sorted(reachable)
