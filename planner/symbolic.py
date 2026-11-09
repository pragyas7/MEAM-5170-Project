"""Classical task planner on the room graph.

This is the oracle. It searches symbolic pushes and navigates; it does not call
the trajectory optimizer. A plan it returns is a candidate the optimizer still
has to accept. Blocked doorways are doors a box is currently wedged in.
Pushing that box into a room opens the door.
"""

from __future__ import annotations

import heapq

from env.geometry import containing_room
from env.models import Layout
from planner.schema import DoneAction, NavigateAction, PushAction


def symbolic_plan(layout: Layout, *, max_depth: int = 10, max_expansions: int = 20000) -> list | None:
    names = [box.name for box in layout.state.boxes]
    robot = containing_room(layout, layout.state.robot.pose.x, layout.state.robot.pose.y)
    if robot is None:
        return None
    start_locs = tuple(_box_location(layout, box) for box in layout.state.boxes)
    start = (robot, start_locs)
    goals = list(layout.goal.atoms)

    def heuristic(state: tuple) -> int:
        rob, locs = state
        loc = dict(zip(names, locs))
        score = 0
        for atom in goals:
            if atom.kind == "box_in_room" and loc.get(atom.object) != atom.room:
                score += 1
            elif atom.kind == "robot_in_room" and rob != atom.room:
                score += 1
        return score

    parent: dict[tuple, tuple | None] = {start: None}
    best_g = {start: 0}
    heap: list[tuple[int, int, tuple]] = [(heuristic(start), 0, start)]
    expansions = 0
    while heap and expansions < max_expansions:
        _, g, state = heapq.heappop(heap)
        if g != best_g.get(state, 10**9):
            continue
        expansions += 1
        if heuristic(state) == 0:
            return _reconstruct(parent, state)
        if g >= max_depth:
            continue
        for action, nxt in _successors(layout, names, state, goals):
            ng = g + 1
            if ng < best_g.get(nxt, 10**9):
                best_g[nxt] = ng
                parent[nxt] = (state, action)
                heapq.heappush(heap, (ng + heuristic(nxt), ng, nxt))
    return None


def direct_plan(layout: Layout) -> list:
    """One-shot plan that ignores blockers and multi-room paths.

    This is the untrained proposer: for every unsatisfied atom, walk up to the
    object and push it straight at its goal region.
    """
    actions = []
    for atom in layout.goal.atoms:
        if atom.kind == "box_in_room":
            box = layout.state.box(atom.object)
            if _box_location(layout, box) == atom.room:
                continue
            if box.blocking:
                door = layout.doorway(box.blocking)
                approach = door.room_b if atom.room == door.room_a else door.room_a
                actions.append(NavigateAction(action="navigate", region=approach))
            else:
                room = containing_room(layout, box.pose.x, box.pose.y) or layout.rooms[0].name
                actions.append(NavigateAction(action="navigate", region=room))
            actions.append(PushAction(action="push", box=box.name, target_region=atom.room))
        elif atom.kind == "robot_in_room":
            here = containing_room(layout, layout.state.robot.pose.x, layout.state.robot.pose.y)
            if here != atom.room:
                actions.append(NavigateAction(action="navigate", region=atom.room))
    actions.append(DoneAction(action="done"))
    return actions


def feedback_plan(layout: Layout, message: str) -> list:
    """Insert an unblock for every door named in an optimizer failure, then retry."""
    actions = []
    blob = ""
    if message:
        token = "blocked_doors="
        if token in message:
            blob = message.split(token, 1)[1].split()[0].strip().strip(".")
    if blob and blob != "none":
        for item in blob.split(","):
            if ":" not in item or item.endswith(":None"):
                continue
            door_name, box_name = item.split(":", 1)
            try:
                door = layout.doorway(door_name)
                layout.state.box(box_name)
            except KeyError:
                continue
            actions.append(NavigateAction(action="navigate", region=door.room_a))
            actions.append(PushAction(action="push", box=box_name, target_region=door.room_b))
    tail = direct_plan(layout)
    if tail and tail[-1].action == "done":
        actions.extend(tail)
    else:
        actions.extend(tail)
        actions.append(DoneAction(action="done"))
    return actions


def _box_location(layout: Layout, box) -> str:
    if box.blocking:
        return f"door:{box.blocking}"
    room = containing_room(layout, box.pose.x, box.pose.y)
    return room or "outside"


def _blocked_map(names: list[str], locs: tuple[str, ...]) -> dict[str, str]:
    blocked = {}
    for name, loc in zip(names, locs):
        if loc.startswith("door:"):
            blocked[loc.split(":", 1)[1]] = name
    return blocked


def _reachable(layout: Layout, room: str, blocked: dict[str, str]) -> set[str]:
    graph = layout.adjacency()
    seen = set()
    stack = [room]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for door in layout.doorways:
            if door.name in blocked:
                continue
            if door.room_a == node and door.room_b not in seen:
                stack.append(door.room_b)
            elif door.room_b == node and door.room_a not in seen:
                stack.append(door.room_a)
    _ = graph
    return seen


def _successors(layout: Layout, names: list[str], state: tuple, goals: list):
    robot, locs = state
    blocked = _blocked_map(names, locs)
    reach = _reachable(layout, robot, blocked)
    # Navigate anywhere the open doors allow. One action crosses several rooms.
    for room in sorted(reach):
        if room == robot:
            continue
        action = NavigateAction(action="navigate", region=room)
        yield action, (room, locs)
    loc = dict(zip(names, locs))
    for index, name in enumerate(names):
        where = locs[index]
        if _frozen(name, where, goals):
            continue
        if where.startswith("door:"):
            door = layout.doorway(where.split(":", 1)[1])
            for target, approach in ((door.room_a, door.room_b), (door.room_b, door.room_a)):
                if approach not in reach:
                    continue
                yield _move_box(names, locs, index, target, robot_room=target), (
                    target,
                    _replaced(locs, index, target),
                )
        else:
            if where not in reach:
                continue
            for door in layout.doorways:
                if door.name in blocked:
                    continue
                if where not in (door.room_a, door.room_b):
                    continue
                target = door.other(where)
                yield _move_box(names, locs, index, target, robot_room=target), (
                    target,
                    _replaced(locs, index, target),
                )
    _ = loc


def _move_box(names, locs, index, target, robot_room):
    return PushAction(action="push", box=names[index], target_region=target)


def _replaced(locs: tuple[str, ...], index: int, value: str) -> tuple[str, ...]:
    seq = list(locs)
    seq[index] = value
    return tuple(seq)


def _frozen(name: str, where: str, goals: list) -> bool:
    mentioned = False
    for atom in goals:
        if atom.kind == "box_in_room" and atom.object == name:
            mentioned = True
            if where != atom.room:
                return False
    # A blocker with no goal is not frozen: someone may need that doorway.
    if where.startswith("door:"):
        return False
    if not mentioned:
        return True
    return True


def _reconstruct(parent: dict, state: tuple) -> list:
    actions = []
    cursor = state
    while parent[cursor] is not None:
        prev, action = parent[cursor]
        actions.append(action)
        cursor = prev
    actions.reverse()
    actions.append(DoneAction(action="done"))
    return actions
