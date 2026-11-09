"""Grid search over eroded free space.

The occupancy grid is the free space (rooms ∪ doorways) eroded by a radius via
a Euclidean distance transform, with oriented boxes punched out. It is a
conservative skeleton for the trajectory optimizer, not the feasibility proof.
Exact collision is checked on the continuous trajectory afterwards.
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from env.geometry import box_sdf
from env.models import Layout, SceneState


class GridMap:
    def __init__(self, resolution: float, origin_x: float, origin_y: float, free: np.ndarray):
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.free = free

    def in_bounds(self, r: int, c: int) -> bool:
        h, w = self.free.shape
        return 0 <= r < h and 0 <= c < w

    def cell(self, x: float, y: float) -> tuple[int, int]:
        c = int(math.floor((x - self.origin_x) / self.resolution))
        r = int(math.floor((y - self.origin_y) / self.resolution))
        return r, c

    def world(self, r: int, c: int) -> tuple[float, float]:
        return (
            self.origin_x + (c + 0.5) * self.resolution,
            self.origin_y + (r + 0.5) * self.resolution,
        )


def build_grid(
    layout: Layout,
    state: SceneState,
    radius: float,
    *,
    resolution: float = 0.10,
    ignore_box: str | None = None,
) -> GridMap:
    xs = [room.x for room in layout.rooms] + [room.x + room.w for room in layout.rooms]
    ys = [room.y for room in layout.rooms] + [room.y + room.h for room in layout.rooms]
    pad = 0.4
    origin_x = min(xs) - pad
    origin_y = min(ys) - pad
    max_x = max(xs) + pad
    max_y = max(ys) + pad
    width = int(math.ceil((max_x - origin_x) / resolution))
    height = int(math.ceil((max_y - origin_y) / resolution))
    cols = origin_x + (np.arange(width) + 0.5) * resolution
    rows = origin_y + (np.arange(height) + 0.5) * resolution
    xx, yy = np.meshgrid(cols, rows)
    inside = np.zeros(xx.shape, dtype=bool)
    for room in layout.rooms:
        inside |= (xx >= room.x) & (xx <= room.x + room.w) & (yy >= room.y) & (yy <= room.y + room.h)
    for door in layout.doorways:
        inside |= (xx >= door.x) & (xx <= door.x + door.w) & (yy >= door.y) & (yy <= door.y + door.h)
    clearance = distance_transform_edt(inside) * resolution
    # Half a cell of margin keeps the path off the rasterized boundary.
    free = clearance >= (radius + 0.5 * resolution)
    for box in state.boxes:
        if box.name == ignore_box:
            continue
        reach = 0.5 * math.hypot(box.sx, box.sy) + radius + 2 * resolution
        c0 = max(0, int((box.pose.x - reach - origin_x) / resolution))
        c1 = min(width - 1, int((box.pose.x + reach - origin_x) / resolution) + 1)
        r0 = max(0, int((box.pose.y - reach - origin_y) / resolution))
        r1 = min(height - 1, int((box.pose.y + reach - origin_y) / resolution) + 1)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if not free[r, c]:
                    continue
                if box_sdf(box, float(xx[r, c]), float(yy[r, c])) < radius:
                    free[r, c] = False
    return GridMap(resolution, origin_x, origin_y, free)


def nearest_free(grid: GridMap, x: float, y: float, max_dist: float = 0.75) -> tuple[int, int] | None:
    r, c = grid.cell(x, y)
    if grid.in_bounds(r, c) and grid.free[r, c]:
        return r, c
    best = None
    best_d = max_dist
    rad = int(max_dist / grid.resolution) + 2
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            rr, cc = r + dr, c + dc
            if not grid.in_bounds(rr, cc) or not grid.free[rr, cc]:
                continue
            wx, wy = grid.world(rr, cc)
            d = math.hypot(wx - x, wy - y)
            if d < best_d:
                best_d = d
                best = (rr, cc)
    return best


def cells_in_room(grid: GridMap, layout: Layout, room_name: str, margin: float) -> list[tuple[int, int, float, float]]:
    room = layout.room(room_name)
    found = []
    rs, cs = np.where(grid.free)
    for r, c in zip(rs.tolist(), cs.tolist()):
        x, y = grid.world(int(r), int(c))
        if room.contains(x, y, margin=margin):
            found.append((int(r), int(c), x, y))
    return found


def closest_cell(
    cells: list[tuple[int, int, float, float]], xy: tuple[float, float]
) -> tuple[int, int] | None:
    if not cells:
        return None
    best = min(cells, key=lambda item: (item[2] - xy[0]) ** 2 + (item[3] - xy[1]) ** 2)
    return best[0], best[1]


_NEI = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def astar(grid: GridMap, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    if start == goal:
        return [start]
    if not grid.free[start] or not grid.free[goal]:
        return None

    def h(a: tuple[int, int]) -> float:
        return math.hypot(a[0] - goal[0], a[1] - goal[1])

    open_heap: list[tuple[float, float, tuple[int, int]]] = [(h(start), 0.0, start)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start: 0.0}
    seen = set()
    while open_heap:
        _, g, node = heapq.heappop(open_heap)
        if node in seen:
            continue
        seen.add(node)
        if node == goal:
            path = [node]
            while path[-1] != start:
                path.append(came[path[-1]])
            path.reverse()
            return path
        for dr, dc in _NEI:
            nxt = (node[0] + dr, node[1] + dc)
            if not grid.in_bounds(*nxt) or not grid.free[nxt]:
                continue
            if dr != 0 and dc != 0:
                if not grid.free[node[0] + dr, node[1]] or not grid.free[node[0], node[1] + dc]:
                    continue
            step = math.hypot(dr, dc)
            ng = g + step
            if ng + 1e-9 < cost.get(nxt, math.inf):
                cost[nxt] = ng
                came[nxt] = node
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None


def shortcut(grid: GridMap, path: list[tuple[int, int]]) -> list[tuple[float, float]]:
    if not path:
        return []
    pts = [grid.world(*cell) for cell in path]
    if len(pts) <= 2:
        return pts

    def clear(a: tuple[float, float], b: tuple[float, float]) -> bool:
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(2, int(dist / (grid.resolution * 0.5)))
        for i in range(steps + 1):
            u = i / steps
            x = a[0] + u * (b[0] - a[0])
            y = a[1] + u * (b[1] - a[1])
            r, c = grid.cell(x, y)
            if not grid.in_bounds(r, c) or not grid.free[r, c]:
                return False
        return True

    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not clear(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def plan_xy(
    grid: GridMap,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    simplify: bool = True,
) -> list[tuple[float, float]] | None:
    start = nearest_free(grid, start_xy[0], start_xy[1])
    goal = nearest_free(grid, goal_xy[0], goal_xy[1], max_dist=0.9)
    if start is None or goal is None:
        return None
    cells = astar(grid, start, goal)
    if cells is None:
        return None
    # Cell centers are the conservative skeleton. The shortcut can clip a jamb:
    # a sample can sit in a free cell and still be closer to the wall than the
    # erosion radius. Callers that fail the continuous check retry with simplify=False.
    pts = shortcut(grid, cells) if simplify else [grid.world(*cell) for cell in cells]
    # Anchor the polyline on the true endpoints. The continuous post-check, not
    # the grid, decides whether the short connecting hop is actually free.
    if math.hypot(pts[0][0] - start_xy[0], pts[0][1] - start_xy[1]) > 1e-3:
        pts = [start_xy, *pts]
    if math.hypot(pts[-1][0] - goal_xy[0], pts[-1][1] - goal_xy[1]) > 1e-3:
        pts = [*pts, goal_xy]
    # Drop near-duplicates so the timer does not spin on zero-length segments.
    cleaned = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > 1e-3:
            cleaned.append(p)
    return cleaned
