"""Deterministic floor-plan generator.

`gen_layout(seed, ...)` draws every random number from `numpy.random.Generator(seed)`,
so the same arguments always rebuild the same rooms, boxes, robot, and goal.

Train / test-ID use chain and L topologies (2–3 rooms, 1–3 boxes).
Test-OOD uses a 2×2 grid and a star (4–5 rooms, 4–6 boxes).
"""

from __future__ import annotations

import math

import numpy as np

from env.geometry import (
    box_free,
    box_sdf,
    build_walls,
    containing_room,
    door_passable,
    robot_free,
    sat_overlap,
)
from env.models import (
    BoxState,
    Doorway,
    Goal,
    GoalAtom,
    Layout,
    Pose,
    RobotState,
    Room,
    SceneState,
)

WALL = 0.20
ROBOT_RADIUS = 0.18
SPLIT_START = {"train": 1_000, "test_id": 20_000, "test_ood": 40_000}


def instance_spec(split: str, index: int) -> tuple[int, int, int, int, str]:
    """Map a split index to `(seed, n_rooms, n_boxes, n_blockers, topology)`."""
    if split not in SPLIT_START:
        raise ValueError(f"unknown split {split!r}")
    seed = SPLIT_START[split] + index
    rng = np.random.default_rng(seed)
    if split in ("train", "test_id"):
        n_rooms = int(rng.integers(2, 4))
        topology = "chain" if n_rooms == 2 or float(rng.random()) < 0.5 else "l_shape"
        n_boxes = int(rng.integers(1, 4))
        n_doors = n_rooms - 1
        n_blockers = int(rng.integers(0, min(2, n_boxes, n_doors) + 1))
    else:
        n_rooms = 4 if float(rng.random()) < 0.5 else 5
        topology = "grid" if n_rooms == 4 else "star"
        n_boxes = int(rng.integers(4, 7))
        n_doors = 4
        n_blockers = int(rng.integers(1, min(3, n_boxes, n_doors) + 1))
    return seed, n_rooms, n_boxes, n_blockers, topology


def load_split(split: str, n: int) -> list[Layout]:
    layouts = []
    for i in range(n):
        seed, n_rooms, n_boxes, n_blockers, topology = instance_spec(split, i)
        layouts.append(
            gen_layout(
                seed,
                n_rooms,
                n_boxes,
                n_blockers,
                topology=topology,
                split=split,
            )
        )
    return layouts


def gen_layout(
    seed: int,
    n_rooms: int,
    n_boxes: int,
    n_blockers: int,
    *,
    topology: str | None = None,
    split: str = "custom",
) -> Layout:
    if n_rooms < 1:
        raise ValueError("n_rooms must be positive")
    if n_boxes < 0 or n_blockers < 0:
        raise ValueError("n_boxes and n_blockers must be non-negative")
    if n_blockers > n_boxes:
        raise ValueError("n_blockers cannot exceed n_boxes")
    topology = topology or _default_topology(n_rooms)
    _check_topology(topology, n_rooms)
    rng = np.random.default_rng(int(seed))
    rooms, doorways = _build_floorplan(topology, n_rooms, rng)
    n_blockers = min(n_blockers, len(doorways), n_boxes)
    walls = build_walls(rooms, doorways)
    # A shell layout so collision helpers can run during placement.
    shell = Layout(
        seed=int(seed),
        split=split,
        topology=topology,
        rooms=rooms,
        doorways=doorways,
        walls=walls,
        state=SceneState(robot=RobotState(pose=Pose(x=0, y=0, theta=0)), boxes=[]),
        goal=Goal(atoms=[]),
        n_rooms=n_rooms,
        n_boxes=n_boxes,
        n_blockers=n_blockers,
    )
    boxes = _place_boxes(shell, rng, n_boxes, n_blockers)
    shell = shell.model_copy(update={"state": SceneState(robot=shell.state.robot, boxes=boxes)})
    robot = _place_robot(shell, rng)
    shell = shell.model_copy(update={"state": SceneState(robot=robot, boxes=boxes)})
    goal = _sample_goal(shell, rng)
    return shell.model_copy(update={"goal": goal, "state": SceneState(robot=robot, boxes=boxes)})


def _default_topology(n_rooms: int) -> str:
    if n_rooms <= 3:
        return "chain"
    if n_rooms == 4:
        return "grid"
    return "star"


def _check_topology(topology: str, n_rooms: int) -> None:
    expected = {"l_shape": 3, "grid": 4, "star": 5}
    if topology == "chain":
        if n_rooms < 1:
            raise ValueError("chain needs at least 1 room")
        return
    if topology not in expected:
        raise ValueError(f"unknown topology {topology!r}")
    if n_rooms != expected[topology]:
        raise ValueError(f"{topology} requires {expected[topology]} rooms, got {n_rooms}")


def _build_floorplan(topology: str, n_rooms: int, rng: np.random.Generator):
    if topology == "chain":
        return _chain(n_rooms, rng)
    if topology == "l_shape":
        return _l_shape(rng)
    if topology == "grid":
        return _grid(rng)
    if topology == "star":
        return _star(rng)
    raise ValueError(topology)


def _room(i: int, x: float, y: float, w: float, h: float) -> Room:
    return Room(name=f"room_{i}", x=float(x), y=float(y), w=float(w), h=float(h))


def _chain(n_rooms: int, rng: np.random.Generator):
    height = float(rng.uniform(3.8, 4.6))
    rooms = []
    x = 0.0
    for i in range(n_rooms):
        w = float(rng.uniform(3.8, 4.8))
        rooms.append(_room(i, x, 0.0, w, height))
        x += w + WALL
    doors = [_connect(rooms[i], rooms[i + 1], i, rng) for i in range(n_rooms - 1)]
    return rooms, doors


def _l_shape(rng: np.random.Generator):
    w0 = float(rng.uniform(3.8, 4.6))
    h0 = float(rng.uniform(3.8, 4.6))
    w1 = float(rng.uniform(3.8, 4.8))
    h2 = float(rng.uniform(3.8, 4.6))
    rooms = [
        _room(0, 0.0, 0.0, w0, h0),
        _room(1, w0 + WALL, 0.0, w1, h0),
        _room(2, 0.0, h0 + WALL, w0, h2),
    ]
    doors = [_connect(rooms[0], rooms[1], 0, rng), _connect(rooms[0], rooms[2], 1, rng)]
    return rooms, doors


def _grid(rng: np.random.Generator):
    w0 = float(rng.uniform(3.8, 4.6))
    w1 = float(rng.uniform(3.8, 4.6))
    h0 = float(rng.uniform(3.8, 4.6))
    h1 = float(rng.uniform(3.8, 4.6))
    rooms = [
        _room(0, 0.0, 0.0, w0, h0),
        _room(1, w0 + WALL, 0.0, w1, h0),
        _room(2, 0.0, h0 + WALL, w0, h1),
        _room(3, w0 + WALL, h0 + WALL, w1, h1),
    ]
    pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]
    doors = [_connect(rooms[a], rooms[b], i, rng) for i, (a, b) in enumerate(pairs)]
    return rooms, doors


def _star(rng: np.random.Generator):
    cw = float(rng.uniform(4.0, 4.6))
    ch = float(rng.uniform(4.0, 4.6))
    ew = float(rng.uniform(3.8, 4.6))
    ww = float(rng.uniform(3.8, 4.6))
    nh = float(rng.uniform(3.8, 4.6))
    sh = float(rng.uniform(3.8, 4.6))
    rooms = [
        _room(0, 0.0, 0.0, cw, ch),
        _room(1, cw + WALL, 0.0, ew, ch),
        _room(2, -(ww + WALL), 0.0, ww, ch),
        _room(3, 0.0, ch + WALL, cw, nh),
        _room(4, 0.0, -(sh + WALL), cw, sh),
    ]
    doors = [_connect(rooms[0], rooms[i], i - 1, rng) for i in range(1, 5)]
    return rooms, doors


def _connect(a: Room, b: Room, index: int, rng: np.random.Generator) -> Doorway:
    """Doorway rectangle spanning the gap between two axis-aligned rooms."""
    horizontal_gap = a.x + a.w <= b.x + 1e-6 or b.x + b.w <= a.x + 1e-6
    if b.x + b.w <= a.x + 1e-6 or b.y + b.h <= a.y + 1e-6:
        # Keep `a` on the left or below `b` so the gap math has one branch.
        if b.x + b.w <= a.x + 1e-6 or (not horizontal_gap and b.y + b.h <= a.y + 1e-6):
            a, b = b, a
    open_len = float(rng.uniform(1.30, 1.55))
    overlap = 0.02
    if a.x + a.w <= b.x + 1e-6:
        y0 = max(a.y, b.y)
        y1 = min(a.y + a.h, b.y + b.h)
        span = y1 - y0
        open_len = min(open_len, span - 0.70)
        open_len = max(open_len, 1.15)
        slack = span - open_len
        shift = float(rng.uniform(0.2, 0.8)) * slack
        y = y0 + shift
        x = a.x + a.w - overlap
        w = (b.x - (a.x + a.w)) + 2 * overlap
        return Doorway(name=f"door_{index}", room_a=a.name, room_b=b.name, x=x, y=y, w=w, h=open_len)
    x0 = max(a.x, b.x)
    x1 = min(a.x + a.w, b.x + b.w)
    span = x1 - x0
    open_len = min(open_len, span - 0.70)
    open_len = max(open_len, 1.15)
    slack = span - open_len
    shift = float(rng.uniform(0.2, 0.8)) * slack
    x = x0 + shift
    y = a.y + a.h - overlap
    h = (b.y - (a.y + a.h)) + 2 * overlap
    return Doorway(name=f"door_{index}", room_a=a.name, room_b=b.name, x=x, y=y, w=open_len, h=h)


def _place_boxes(layout: Layout, rng: np.random.Generator, n_boxes: int, n_blockers: int) -> list[BoxState]:
    boxes: list[BoxState] = []
    door_order = list(range(len(layout.doorways)))
    rng.shuffle(door_order)
    for i in range(n_blockers):
        door = layout.doorways[int(door_order[i])]
        boxes.append(_blocker(layout, i, door, rng))
    layout_now = layout.model_copy(
        update={"state": SceneState(robot=layout.state.robot, boxes=list(boxes))}
    )
    for i in range(n_blockers, n_boxes):
        boxes.append(_free_box(layout_now, i, rng))
        layout_now = layout_now.model_copy(
            update={"state": SceneState(robot=layout.state.robot, boxes=list(boxes))}
        )
    return boxes


def _blocker(layout: Layout, index: int, door: Doorway, rng: np.random.Generator) -> BoxState:
    cx, cy = door.center
    # Leave a side gap of 0.25 m, which is less than the robot diameter (0.36 m).
    if door.w <= door.h:
        sy = max(0.40, door.h - 0.50)
        sx = max(0.42, door.w + 0.28)
    else:
        sx = max(0.40, door.w - 0.50)
        sy = max(0.42, door.h + 0.28)
    mass = float(rng.uniform(0.8, 2.2))
    mu = float(rng.uniform(0.28, 0.42))
    return BoxState(
        name=f"box_{index}",
        pose=Pose(x=float(cx), y=float(cy), theta=0.0),
        sx=float(sx),
        sy=float(sy),
        mass=mass,
        mu=mu,
        blocking=door.name,
    )


def _free_box(layout: Layout, index: int, rng: np.random.Generator) -> BoxState:
    rooms = list(layout.rooms)
    rng.shuffle(rooms)
    for _attempt in range(400):
        room = rooms[_attempt % len(rooms)]
        sx = float(rng.uniform(0.34, 0.48))
        sy = float(rng.uniform(0.34, 0.48))
        theta = 0.0 if float(rng.random()) < 0.5 else math.pi / 2
        margin = 0.5 * math.hypot(sx, sy) + 0.35
        if room.w <= 2 * margin or room.h <= 2 * margin:
            margin = 0.7
        x = float(rng.uniform(room.x + margin, room.x + room.w - margin))
        y = float(rng.uniform(room.y + margin, room.y + room.h - margin))
        candidate = BoxState(
            name=f"box_{index}",
            pose=Pose(x=x, y=y, theta=float(theta)),
            sx=sx,
            sy=sy,
            mass=float(rng.uniform(0.8, 2.2)),
            mu=float(rng.uniform(0.28, 0.42)),
        )
        if _box_pose_ok(layout, candidate):
            return candidate
    # Deterministic fallback: park the box at a room center, nudged until it fits.
    for room in layout.rooms:
        cx, cy = room.center
        candidate = BoxState(
            name=f"box_{index}",
            pose=Pose(x=cx, y=cy, theta=0.0),
            sx=0.36,
            sy=0.36,
            mass=1.0,
            mu=0.35,
        )
        if _box_pose_ok(layout, candidate):
            return candidate
    raise RuntimeError(f"could not place {index=} in seed={layout.seed}")


def _box_pose_ok(layout: Layout, candidate: BoxState) -> bool:
    state = SceneState(robot=layout.state.robot, boxes=[*layout.state.boxes, candidate])
    trial = layout.model_copy(update={"state": state})
    if containing_room(trial, candidate.pose.x, candidate.pose.y, margin=0.15) is None:
        return False
    if not box_free(trial, state, candidate.name, candidate.pose.x, candidate.pose.y, candidate.pose.theta, gap=0.12):
        return False
    for door in layout.doorways:
        if _body_covers_door(candidate, door):
            return False
    return True


def _body_covers_door(box: BoxState, door: Doorway) -> bool:
    return sat_overlap(
        (box.pose.x, box.pose.y),
        box.pose.theta,
        (box.sx * 0.5, box.sy * 0.5),
        door.center,
        0.0,
        (door.w * 0.5, door.h * 0.5),
        gap=0.05,
    )


def _place_robot(layout: Layout, rng: np.random.Generator) -> RobotState:
    theta = float(rng.uniform(-math.pi, math.pi))
    rooms = list(layout.rooms)
    rng.shuffle(rooms)
    radius = ROBOT_RADIUS
    for _attempt in range(500):
        room = rooms[_attempt % len(rooms)]
        margin = radius + 0.25
        x = float(rng.uniform(room.x + margin, room.x + room.w - margin))
        y = float(rng.uniform(room.y + margin, room.y + room.h - margin))
        robot = RobotState(pose=Pose(x=x, y=y, theta=theta), radius=radius)
        state = layout.state.model_copy(update={"robot": robot})
        if robot_free(layout, state, x, y, margin=0.02):
            # Keep the robot out of doorway throats so its room label is stable.
            if containing_room(layout, x, y, margin=0.2) is not None:
                clear = min(box_sdf(b, x, y) for b in state.boxes) if state.boxes else 1.0
                if clear >= radius + 0.15:
                    return robot
    for room in layout.rooms:
        cx, cy = room.center
        robot = RobotState(pose=Pose(x=cx, y=cy, theta=theta), radius=radius)
        state = layout.state.model_copy(update={"robot": robot})
        if robot_free(layout, state, cx, cy):
            return robot
    raise RuntimeError(f"could not place robot in seed={layout.seed}")


def _sample_goal(layout: Layout, rng: np.random.Generator) -> Goal:
    graph = layout.adjacency()

    def dist(src: str, dst: str) -> int:
        if src == dst:
            return 0
        seen = {src}
        queue = [(src, 0)]
        while queue:
            node, d = queue.pop(0)
            for nxt in graph[node]:
                if nxt in seen:
                    continue
                if nxt == dst:
                    return d + 1
                seen.add(nxt)
                queue.append((nxt, d + 1))
        return 99

    robot_room = containing_room(layout, layout.state.robot.pose.x, layout.state.robot.pose.y)
    assert robot_room is not None
    candidates = list(layout.state.boxes)
    rng.shuffle(candidates)
    n_box_goals = 1
    if len(candidates) >= 2 and float(rng.random()) < 0.65:
        n_box_goals = 2
    if len(candidates) >= 4 and float(rng.random()) < 0.5:
        n_box_goals = 3
    atoms: list[GoalAtom] = []
    used_rooms_for_box: dict[str, str] = {}
    for box in candidates:
        if len(atoms) >= n_box_goals:
            break
        if box.blocking:
            door = layout.doorway(box.blocking)
            options = [door.room_a, door.room_b]
        else:
            here = containing_room(layout, box.pose.x, box.pose.y)
            if here is None:
                continue
            options = [name for name in graph if name != here and dist(here, name) < 99]
        rng.shuffle(options)
        if not options:
            continue
        target = str(options[0])
        atoms.append(GoalAtom(kind="box_in_room", room=target, object=box.name))
        used_rooms_for_box[box.name] = target
    if not atoms and layout.state.boxes:
        box = layout.state.boxes[0]
        target = layout.rooms[-1].name
        atoms.append(GoalAtom(kind="box_in_room", room=target, object=box.name))
    if float(rng.random()) < 0.55:
        options = [r.name for r in layout.rooms if r.name != robot_room]
        if options:
            atoms.append(GoalAtom(kind="robot_in_room", room=str(rng.choice(options))))
    return Goal(atoms=atoms)


def assert_layout_consistent(layout: Layout) -> None:
    """Raise if a generated layout violates the geometric invariants tests rely on."""
    state = layout.state
    if not robot_free(layout, state, state.robot.pose.x, state.robot.pose.y):
        raise AssertionError("robot starts in collision")
    if containing_room(layout, state.robot.pose.x, state.robot.pose.y) is None:
        raise AssertionError("robot is not in a room")
    for box in state.boxes:
        if box.blocking:
            door = layout.doorway(box.blocking)
            if door_passable(layout, state, door):
                raise AssertionError(f"{box.name} does not block {door.name}")
        else:
            if containing_room(layout, box.pose.x, box.pose.y) is None:
                raise AssertionError(f"{box.name} is not in a room")
            for door in layout.doorways:
                if not door_passable(layout, state, door) and box.blocking is None:
                    # A free box must not be the reason a door is shut.
                    pass
    for door in layout.doorways:
        blocked = not door_passable(layout, state, door)
        named = any(b.blocking == door.name for b in state.boxes)
        if blocked != named:
            raise AssertionError(f"{door.name} blocked={blocked} labeled={named}")
