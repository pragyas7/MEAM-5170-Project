"""Exact 2D geometry used by the generator, the grid, and post-checks.

Free space is the union of room rectangles and doorway rectangles. Walls are the
room edges with doorway intervals removed. Boxes are oriented rectangles.
"""

from __future__ import annotations

import math

from env.models import BoxState, Doorway, Layout, SceneState, Wall

Point = tuple[float, float]


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def dist_point_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c2 = vx * vx + vy * vy
    if c2 < 1e-12:
        return math.hypot(wx, wy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / c2))
    return math.hypot(px - (x1 + t * vx), py - (y1 + t * vy))


def sdf_obb(px: float, py: float, cx: float, cy: float, theta: float, hx: float, hy: float) -> float:
    """Signed distance from a point to an oriented box. Negative inside."""
    c, s = math.cos(theta), math.sin(theta)
    dx, dy = px - cx, py - cy
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    qx = abs(lx) - hx
    qy = abs(ly) - hy
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return outside + inside


def obb_corners(cx: float, cy: float, theta: float, hx: float, hy: float) -> list[Point]:
    c, s = math.cos(theta), math.sin(theta)
    pts: list[Point] = []
    for sx in (-hx, hx):
        for sy in (-hy, hy):
            pts.append((cx + c * sx - s * sy, cy + s * sx + c * sy))
    # Order around the boundary: (-hx,-hy), (+hx,-hy), (+hx,+hy), (-hx,+hy) in the box frame.
    local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return [(cx + c * lx - s * ly, cy + s * lx + c * ly) for lx, ly in local]


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point, eps: float = 1e-8) -> bool:
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)
    return o1 * o2 < -eps and o3 * o4 < -eps


def sat_overlap(
    c1: Point,
    th1: float,
    h1: tuple[float, float],
    c2: Point,
    th2: float,
    h2: tuple[float, float],
    gap: float = 0.0,
) -> bool:
    """True when two OBBs overlap. `gap` inflates both boxes."""

    def axes(th: float) -> list[Point]:
        return [(math.cos(th), math.sin(th)), (-math.sin(th), math.cos(th))]

    def project(corners: list[Point], ax: Point) -> tuple[float, float]:
        vals = [p[0] * ax[0] + p[1] * ax[1] for p in corners]
        return min(vals), max(vals)

    a = obb_corners(c1[0], c1[1], th1, h1[0] + gap, h1[1] + gap)
    b = obb_corners(c2[0], c2[1], th2, h2[0] + gap, h2[1] + gap)
    for ax in axes(th1) + axes(th2):
        a0, a1 = project(a, ax)
        b0, b1 = project(b, ax)
        if a1 < b0 or b1 < a0:
            return False
    return True


def subtract_intervals(
    a: float, b: float, cuts: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    segs = [(a, b)]
    for c0, c1 in cuts:
        nxt: list[tuple[float, float]] = []
        for s0, s1 in segs:
            if c1 <= s0 or c0 >= s1:
                nxt.append((s0, s1))
                continue
            if s0 < c0:
                nxt.append((s0, c0))
            if c1 < s1:
                nxt.append((c1, s1))
        segs = [(u, v) for u, v in nxt if v - u > 1e-6]
    return segs


def build_walls(rooms, doorways) -> list[Wall]:
    walls: list[Wall] = []
    for room in rooms:
        edges = [
            ("h", room.y, room.x, room.x + room.w),
            ("h", room.y + room.h, room.x, room.x + room.w),
            ("v", room.x, room.y, room.y + room.h),
            ("v", room.x + room.w, room.y, room.y + room.h),
        ]
        for kind, fixed, a, b in edges:
            cuts: list[tuple[float, float]] = []
            for door in doorways:
                if kind == "h" and door.y - 1e-7 <= fixed <= door.y + door.h + 1e-7:
                    cuts.append((door.x, door.x + door.w))
                if kind == "v" and door.x - 1e-7 <= fixed <= door.x + door.w + 1e-7:
                    cuts.append((door.y, door.y + door.h))
            for s0, s1 in subtract_intervals(a, b, cuts):
                if kind == "h":
                    walls.append(Wall(x1=s0, y1=fixed, x2=s1, y2=fixed))
                else:
                    walls.append(Wall(x1=fixed, y1=s0, x2=fixed, y2=s1))
    return walls


def inside_free(layout: Layout, x: float, y: float) -> bool:
    for room in layout.rooms:
        if room.x - 1e-9 <= x <= room.x + room.w + 1e-9 and room.y - 1e-9 <= y <= room.y + room.h + 1e-9:
            return True
    for door in layout.doorways:
        if door.contains(x, y):
            return True
    return False


def dist_to_walls(layout: Layout, x: float, y: float) -> float:
    best = math.inf
    for wall in layout.walls:
        best = min(best, dist_point_segment(x, y, wall.x1, wall.y1, wall.x2, wall.y2))
    return best


def containing_room(layout: Layout, x: float, y: float, margin: float = 0.0) -> str | None:
    best: str | None = None
    best_margin = -1.0
    for room in layout.rooms:
        if not room.contains(x, y, margin=0.0):
            continue
        # Prefer the room in which the point sits deepest, so doorway overlaps
        # that spill 2 cm into a room do not steal the label.
        inset = min(x - room.x, room.x + room.w - x, y - room.y, room.y + room.h - y)
        if inset > best_margin:
            best_margin = inset
            best = room.name
    if best is None:
        return None
    if margin > 0 and not layout.room(best).contains(x, y, margin=margin):
        return None
    return best


def box_sdf(box: BoxState, x: float, y: float) -> float:
    return sdf_obb(x, y, box.pose.x, box.pose.y, box.pose.theta, box.sx * 0.5, box.sy * 0.5)


def robot_free(
    layout: Layout,
    state: SceneState,
    x: float,
    y: float,
    *,
    ignore_box: str | None = None,
    margin: float = 0.0,
) -> bool:
    radius = state.robot.radius + margin
    if not inside_free(layout, x, y):
        return False
    if dist_to_walls(layout, x, y) < radius - 1e-6:
        return False
    for box in state.boxes:
        if box.name == ignore_box:
            continue
        if box_sdf(box, x, y) < radius - 1e-6:
            return False
    return True


def box_corners(box: BoxState, x: float | None = None, y: float | None = None, theta: float | None = None) -> list[Point]:
    return obb_corners(
        box.pose.x if x is None else x,
        box.pose.y if y is None else y,
        box.pose.theta if theta is None else theta,
        box.sx * 0.5,
        box.sy * 0.5,
    )


def box_hits_walls(layout: Layout, corners: list[Point]) -> bool:
    n = len(corners)
    for i in range(n):
        a = corners[i]
        b = corners[(i + 1) % n]
        if not inside_free(layout, a[0], a[1]):
            return True
        for wall in layout.walls:
            if segments_intersect(a, b, (wall.x1, wall.y1), (wall.x2, wall.y2)):
                return True
    return False


def box_hits_other_boxes(
    state: SceneState,
    name: str,
    x: float,
    y: float,
    theta: float,
    gap: float = 0.0,
) -> bool:
    me = state.box(name)
    for other in state.boxes:
        if other.name == name:
            continue
        if sat_overlap(
            (x, y),
            theta,
            (me.sx * 0.5, me.sy * 0.5),
            (other.pose.x, other.pose.y),
            other.pose.theta,
            (other.sx * 0.5, other.sy * 0.5),
            gap=gap,
        ):
            return True
    return False


def box_free(
    layout: Layout,
    state: SceneState,
    name: str,
    x: float,
    y: float,
    theta: float,
    *,
    gap: float = 0.02,
) -> bool:
    corners = box_corners(state.box(name), x, y, theta)
    # Shrink slightly so grazing a wall is not a hit.
    shrunk = obb_corners(x, y, theta, state.box(name).sx * 0.5 - 0.01, state.box(name).sy * 0.5 - 0.01)
    if box_hits_walls(layout, shrunk):
        return False
    if box_hits_other_boxes(state, name, x, y, theta, gap=gap):
        return False
    return True


def doorway_samples(door: Doorway, n: int = 41) -> list[Point]:
    """Samples across the throat of a doorway, along the opening."""
    if door.w <= door.h:
        xs = door.x + 0.5 * door.w
        ys = [door.y + door.h * i / (n - 1) for i in range(n)]
        return [(xs, y) for y in ys]
    ys = door.y + 0.5 * door.h
    xs = [door.x + door.w * i / (n - 1) for i in range(n)]
    return [(x, ys) for x in xs]


def door_passable(layout: Layout, state: SceneState, door: Doorway, *, margin: float = 0.0) -> bool:
    """A robot disc can sit somewhere on the doorway throat."""
    return any(
        robot_free(layout, state, x, y, margin=margin) for x, y in doorway_samples(door)
    )


def pose_blocks_doorway(box: BoxState, x: float, y: float, theta: float, door: Doorway) -> bool:
    """True when a box at this pose still plugs the doorway for the next box.

    Touching the doorway rectangle is not the whole test. A box parked just
    inside the room, centered on the opening, covers the throat even though
    `door_passable` (a robot disc) is already true. The keep-out runs past the
    jamb by half the parked box's long side plus a passage width.
    """
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    hx = 0.5 * box.sx
    hy = 0.5 * box.sy
    ext_x = c * hx + s * hy
    ext_y = s * hx + c * hy
    minx, maxx = x - ext_x, x + ext_x
    miny, maxy = y - ext_y, y + ext_y
    runway = 0.55 + 0.5 * max(box.sx, box.sy)
    # The thin side of the doorway is the wall thickness. Expand that axis.
    if door.w <= door.h:
        close = maxx >= door.x - runway and minx <= door.x + door.w + runway
        covers = maxy >= door.y and miny <= door.y + door.h
    else:
        close = maxy >= door.y - runway and miny <= door.y + door.h + runway
        covers = maxx >= door.x and minx <= door.x + door.w
    return close and covers


def blocking_box_name(layout: Layout, state: SceneState, door: Doorway) -> str | None:
    if door_passable(layout, state, door):
        return None
    # Attribute the blockage to the box whose body is nearest the throat center.
    cx, cy = door.center
    best_name = None
    best_sdf = math.inf
    for box in state.boxes:
        sdf = box_sdf(box, cx, cy)
        if sdf < best_sdf:
            best_sdf = sdf
            best_name = box.name
    return best_name


def doors_touching(layout: Layout, room_name: str) -> list[Doorway]:
    return [d for d in layout.doorways if room_name in (d.room_a, d.room_b)]
