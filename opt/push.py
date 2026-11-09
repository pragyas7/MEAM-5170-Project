"""Quasi-static push under a rigid face-attachment simplification.

During a push the robot is fixed to one face of the box (no slip, no change of
face). The pair is a unicycle: the box translates only along the inward normal
of the contacted face, and yaw rate is limited by the leftover push-force
budget after Coulomb friction `mu * m * g` is paid. See `OptimizerConfig.scaled_push_yaw`.

The action includes an approach-to-contact navigate, then the push itself.
Documented simplification: contact is sticking and rigid, so the robot pose is
a function of the box pose for the whole push.
"""

from __future__ import annotations

import math
import time

import numpy as np

from env.geometry import box_free, containing_room, pose_blocks_doorway, robot_free, wrap_angle
from env.models import BoxState, Layout, SceneState
from opt.config import OptimizerConfig
from opt.grid import build_grid, plan_xy
from opt.navigate import solve_navigate
from opt.paths import time_parameterize
from opt.types import MotionResult, failure

# Inward heading of the push, in the box frame, for each contacted face.
LOCAL_INWARD = {
    "+x": math.pi,
    "-x": 0.0,
    "+y": -math.pi / 2.0,
    "-y": math.pi / 2.0,
}
OUTWARD = {
    "+x": (1.0, 0.0),
    "-x": (-1.0, 0.0),
    "+y": (0.0, 1.0),
    "-y": (0.0, -1.0),
}


def solve_push(
    layout: Layout,
    state: SceneState,
    box_name: str,
    config: OptimizerConfig,
    *,
    target_region: str | None = None,
    target_pose: tuple[float, float, float] | None = None,
) -> MotionResult:
    t0 = time.perf_counter()
    try:
        box = state.box(box_name)
    except KeyError:
        return failure("goal_not_reached", f"push box={box_name} failed: goal_not_reached. unknown box")
    if target_region is None and target_pose is None:
        return failure("goal_not_reached", f"push box={box_name} failed: goal_not_reached. no target")
    if target_region is not None and target_region not in layout.room_names():
        return failure(
            "goal_not_reached",
            f"push box={box_name} target_region={target_region} failed: goal_not_reached. unknown region",
        )

    required, w_allow = config.scaled_push_yaw(box.mu, box.mass)
    if config.max_push_force < required - 1e-6:
        return failure(
            "push_infeasible",
            (
                f"push box={box_name} target_region={target_region or 'pose'} failed: push_infeasible. "
                f"required_force={required:.2f} max_push_force={config.max_push_force:.2f}"
            ),
            solve_time=time.perf_counter() - t0,
        )

    goals = _goal_options(layout, state, box, target_region, target_pose)
    if not goals:
        return failure(
            "goal_not_reached",
            f"push box={box_name} failed: goal_not_reached. target room has no free pose",
            solve_time=time.perf_counter() - t0,
        )

    last: MotionResult | None = None
    calls = 0
    # Clear parks first. A pose that still plugs the doorway is only used when
    # every clear park failed, so a later box can still come through.
    clear = [(xy, th) for xy, th, blocks in goals if not blocks]
    plugging = [(xy, th) for xy, th, blocks in goals if blocks]
    for pool, reject_door_block in ((clear[:4], True), (plugging[:2], False)):
        for goal_xy, goal_th in pool:
            desired = math.atan2(goal_xy[1] - box.pose.y, goal_xy[0] - box.pose.x)
            for face, local in _rank_faces(box.pose.theta, desired)[:2]:
                attempt = _attempt_face(
                    layout, state, box, face, local, goal_xy, goal_th, config, w_allow,
                    target_region, calls, t0, reject_door_block=reject_door_block,
                )
                calls = attempt[1]
                if isinstance(attempt[0], MotionResult) and attempt[0].success:
                    return attempt[0]
                last = attempt[0]
    if last is None:
        last = failure("push_infeasible", f"push box={box_name} failed: push_infeasible.")
    last.solve_time = time.perf_counter() - t0
    last.optimizer_calls = max(calls, last.optimizer_calls)
    return last


def _attempt_face(
    layout, state, box, face, local, goal_xy, goal_th, config, w_allow, target_region, calls, t0,
    *,
    reject_door_block: bool = False,
):
    """Try one contact face. Returns `(result or None, updated_call_count)`."""
    box_name = box.name
    contact = contact_pose(box, face, state.robot.radius)
    if not robot_free(layout, state, contact[0], contact[1], ignore_box=box.name, margin=-0.02):
        return (
            failure("collision", f"push box={box_name} failed: collision. contact pose on face {face} is blocked"),
            calls,
        )
    approach = solve_navigate(
        layout,
        state,
        containing_room(layout, contact[0], contact[1]),
        config,
        goal_pose=contact,
        ignore_box=box.name,
    )
    calls += approach.optimizer_calls
    if not approach.success:
        approach.message = (
            f"push box={box_name} target_region={target_region or 'pose'} failed during approach: "
            + approach.message
        )
        return approach, calls
    pushed = _push_box(
        layout,
        state,
        box,
        face,
        local,
        goal_xy,
        goal_th,
        config,
        w_allow,
        target_region=target_region,
        reject_door_block=reject_door_block,
    )
    if pushed is None:
        reason = "infeasible_dynamics" if w_allow < 0.05 or config.min_turn_radius > 1e-3 else "collision"
        return (
            failure(
                reason,
                (
                    f"push box={box_name} target_region={target_region or 'pose'} failed: {reason}. "
                    f"face={face} yaw_allow={w_allow:.3f} min_turn_radius={config.min_turn_radius:.2f}"
                ),
            ),
            calls,
        )
    traj = _concat(approach.trajectory, pushed, box.name)
    return (
        MotionResult(
            success=True,
            failure_reason=None,
            message=(
                f"push box={box_name} target_region={target_region or 'pose'} ok "
                f"face={face} solver={pushed['solver']}"
            ),
            cost=float(approach.cost + pushed["cost"]),
            solve_time=time.perf_counter() - t0,
            optimizer_calls=calls + 1,
            trajectory=traj,
            end_robot=(traj["x"][-1], traj["y"][-1], traj["theta"][-1]),
            end_box=(box.name, traj["box_x"][-1], traj["box_y"][-1], traj["box_theta"][-1]),
            solver=pushed["solver"],
        ),
        calls,
    )


def contact_pose(box: BoxState, face: str, radius: float, standoff: float = 0.04) -> tuple[float, float, float]:
    theta = box.pose.theta
    c, s = math.cos(theta), math.sin(theta)
    ox, oy = OUTWARD[face]
    half = box.sx * 0.5 if face in ("+x", "-x") else box.sy * 0.5
    wx = c * ox - s * oy
    wy = s * ox + c * oy
    dist = half + radius + standoff
    heading = wrap_angle(theta + LOCAL_INWARD[face])
    return (box.pose.x + wx * dist, box.pose.y + wy * dist, heading)


def _rank_faces(theta: float, desired: float) -> list[tuple[str, float]]:
    scored = []
    for face, local in LOCAL_INWARD.items():
        heading = wrap_angle(theta + local)
        scored.append((abs(wrap_angle(heading - desired)), face, local))
    scored.sort()
    return [(face, local) for _, face, local in scored]


def _goal_options(layout, state, box, target_region, target_pose):
    """Candidate terminal poses as `(point, theta or None, blocks_doorway)`.

    Poses that leave the doorway clear are sorted ahead of ones that plug it.
    Parking on the opening's centerline makes the next push collide.
    """
    if target_pose is not None:
        return [((float(target_pose[0]), float(target_pose[1])), float(target_pose[2]), False)]
    room = layout.room(target_region)
    entry = _entry_point(layout, state, box, target_region) or room.center
    raw = [entry, room.center]
    # Spread targets across the room so successive pushes don't share a pose.
    for fx in (0.35, 0.62, 0.8):
        for fy in (0.32, 0.5, 0.72):
            raw.append((room.x + fx * room.w, room.y + fy * room.h))
    # On the doorway axis, but deep, plus short lateral offsets.
    cx, cy = room.center
    ex, ey = entry
    raw.append((ex, ey + 0.7))
    raw.append((ex, ey - 0.7))
    raw.append((cx, ey))
    seen = set()
    ranked = []
    for x, y in raw:
        key = (round(x, 2), round(y, 2))
        if key in seen:
            continue
        seen.add(key)
        x, y = _clamp(room, x, y, 0.5)
        if not room.contains(x, y, margin=0.35):
            continue
        point = (x, y)
        blocks = _blocks_target_doors(layout, box, x, y, box.pose.theta, target_region)
        free_pose = box_free(layout, state, box.name, x, y, box.pose.theta, gap=0.05)
        depth = _min_door_depth(layout, target_region, x, y)
        offset = _door_axis_offset(layout, target_region, entry, x, y)
        # Non-plugging and free first, then deeper, then closer to the doorway
        # axis so a straight push does not have to yaw inside the throat.
        ranked.append((blocks, not free_pose, -depth, offset, point))
    ranked.sort()
    return [(point, None, blocks) for blocks, _, _, _, point in ranked]


def _blocks_target_doors(layout, box, x, y, theta, region) -> bool:
    if region is None:
        return False
    for door in layout.doorways:
        if region not in (door.room_a, door.room_b):
            continue
        if pose_blocks_doorway(box, x, y, theta, door):
            return True
    return False


def _min_door_depth(layout, region, x, y) -> float:
    """How far (x, y) sits into `region` past the nearest doorway."""
    room = layout.room(region)
    depth = None
    for door in layout.doorways:
        if region not in (door.room_a, door.room_b):
            continue
        along = _along_into_room(room, door, x, y)
        depth = along if depth is None else min(depth, along)
    if depth is None:
        return math.hypot(x - room.center[0], y - room.center[1])
    return depth


def _door_axis_offset(layout, region, entry, x, y) -> float:
    """Perpendicular distance from the doorway axis the entry point came from."""
    room = layout.room(region)
    door = None
    best = math.inf
    for candidate in layout.doorways:
        if region not in (candidate.room_a, candidate.room_b):
            continue
        dist = math.hypot(entry[0] - candidate.center[0], entry[1] - candidate.center[1])
        if dist < best:
            best = dist
            door = candidate
    if door is None:
        return 0.0
    return abs(_across_door(room, door, x, y))


def _along_into_room(room, door, x, y) -> float:
    cx, cy = door.center
    rcx, rcy = room.center
    dx, dy = rcx - cx, rcy - cy
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 0.0
    return ((x - cx) * dx + (y - cy) * dy) / norm


def _across_door(room, door, x, y) -> float:
    cx, cy = door.center
    rcx, rcy = room.center
    dx, dy = rcx - cx, rcy - cy
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 0.0
    px, py = -dy / norm, dx / norm
    return (x - cx) * px + (y - cy) * py


def _entry_point(layout, state, box, region) -> tuple[float, float] | None:
    room = layout.room(region)
    door = None
    if box.blocking and region in (
        layout.doorway(box.blocking).room_a,
        layout.doorway(box.blocking).room_b,
    ):
        door = layout.doorway(box.blocking)
    else:
        src = containing_room(layout, box.pose.x, box.pose.y)
        if src is not None and src != region:
            path = _room_path(layout, src, region)
            if path and len(path) >= 2:
                door = _door_between(layout, path[-2], path[-1])
    if door is None:
        return room.center
    return _point_inside_from_door(room, door, margin=0.85)


def _point_inside_from_door(room, door, margin: float) -> tuple[float, float]:
    cx, cy = door.center
    rcx, rcy = room.center
    direction = np.array([rcx - cx, rcy - cy], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return room.center
    direction /= norm
    for dist in (margin, margin + 0.35, margin + 0.7, 1.4, 1.9):
        x = cx + direction[0] * dist
        y = cy + direction[1] * dist
        if room.contains(x, y, margin=0.45):
            return (float(x), float(y))
    return _clamp(room, rcx, rcy, 0.55)


def _clamp(room, x, y, margin) -> tuple[float, float]:
    return (
        float(min(max(x, room.x + margin), room.x + room.w - margin)),
        float(min(max(y, room.y + margin), room.y + room.h - margin)),
    )


def _room_path(layout, src: str, dst: str) -> list[str] | None:
    if src == dst:
        return [src]
    graph = layout.adjacency()
    prev = {src: None}
    queue = [src]
    while queue:
        node = queue.pop(0)
        for nxt in graph[node]:
            if nxt in prev:
                continue
            prev[nxt] = node
            if nxt == dst:
                path = [dst]
                while path[-1] != src:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            queue.append(nxt)
    return None


def _door_between(layout, a: str, b: str):
    for door in layout.doorways:
        ends = {door.room_a, door.room_b}
        if ends == {a, b}:
            return door
    return None


def _push_box(
    layout, state, box, face, local, goal_xy, goal_th, config, w_allow, target_region,
    *,
    reject_door_block: bool = False,
):
    start_xy = (box.pose.x, box.pose.y)
    paths = []
    straight = [start_xy, goal_xy]
    paths.append(straight)
    clear = 0.5 * max(box.sx, box.sy) + 0.06
    grid = build_grid(
        layout,
        state,
        clear,
        resolution=config.grid_resolution,
        ignore_box=box.name,
    )
    geometric = plan_xy(grid, start_xy, goal_xy)
    if geometric is not None:
        paths.append(geometric)
    motion_theta = wrap_angle(box.pose.theta + local)
    final_motion = None if goal_th is None else wrap_angle(goal_th + local)
    for path in paths:
        timed = time_parameterize(
            path,
            motion_theta,
            v_max=config.v_max,
            w_max=max(w_allow, 0.0),
            min_turn_radius=config.min_turn_radius,
            final_theta=final_motion,
            v_min=0.0,
        )
        if timed is None:
            continue
        stamped = _stamp_box(timed, box.name, local, state.robot.radius, face, box)
        if not _assembly_safe(layout, state, box.name, stamped):
            continue
        end_x = stamped["box_x"][-1]
        end_y = stamped["box_y"][-1]
        end_th = stamped["box_theta"][-1]
        if target_region is not None:
            if containing_room(layout, end_x, end_y) != target_region:
                continue
            # Arrival yaw can swing the long side back into the throat even
            # when the requested center looked clear at the start heading.
            if reject_door_block and _blocks_target_doors(layout, box, end_x, end_y, end_th, target_region):
                continue
        elif goal_th is not None:
            if math.hypot(stamped["box_x"][-1] - goal_xy[0], stamped["box_y"][-1] - goal_xy[1]) > config.goal_pos_tol:
                continue
            if abs(wrap_angle(stamped["box_theta"][-1] - goal_th)) > config.goal_theta_tol:
                continue
        return stamped
    return None


def _stamp_box(traj, name, local, radius, face, box: BoxState) -> dict:
    """Store the robot disc that the rigid attachment implies at every sample."""
    out = dict(traj)
    rx, ry, rth = [], [], []
    bx, by, bth = [], [], []
    # Rebuild a lightweight box pose per sample. Face geometry uses the box size.
    ghost = box.model_copy(deep=True)
    for x, y, motion_th in zip(traj["x"], traj["y"], traj["theta"]):
        # `traj` currently stores the motion heading in `theta` and the box center in x, y.
        box_theta = motion_th - local
        ghost.pose.x = float(x)
        ghost.pose.y = float(y)
        ghost.pose.theta = float(box_theta)
        crx, cry, heading = contact_pose(ghost, face, radius, standoff=0.04)
        bx.append(float(x))
        by.append(float(y))
        bth.append(float(box_theta))
        rx.append(crx)
        ry.append(cry)
        rth.append(heading)
    out["box_name"] = name
    out["box_x"] = bx
    out["box_y"] = by
    out["box_theta"] = bth
    out["x"] = rx
    out["y"] = ry
    out["theta"] = rth
    return out


def _assembly_safe(layout, state, box_name, traj) -> bool:
    for i in range(len(traj["box_x"])):
        bx, by, bt = traj["box_x"][i], traj["box_y"][i], traj["box_theta"][i]
        if not box_free(layout, state, box_name, bx, by, bt, gap=0.01):
            return False
        if not robot_free(layout, state, traj["x"][i], traj["y"][i], ignore_box=box_name, margin=-0.02):
            return False
    return True


def _concat(approach: dict, push: dict, box_name: str) -> dict:
    shift = float(approach["t"][-1])
    # Drop the duplicate knot at the junction.
    n0 = len(approach["x"])
    box_x = [push["box_x"][0]] * n0 + push["box_x"][1:]
    box_y = [push["box_y"][0]] * n0 + push["box_y"][1:]
    box_th = [push["box_theta"][0]] * n0 + push["box_theta"][1:]
    def cat(key):
        return list(approach[key]) + list(push[key][1:])

    t = list(approach["t"]) + [shift + float(v) for v in push["t"][1:]]
    traj = {
        "t": t,
        "x": cat("x"),
        "y": cat("y"),
        "theta": cat("theta"),
        "v": list(approach["v"][:-1]) + list(push["v"]),
        "omega": list(approach["omega"][:-1]) + list(push["omega"]),
        "box_name": box_name,
        "box_x": box_x,
        "box_y": box_y,
        "box_theta": box_th,
        "solver": push.get("solver", "analytic"),
        "cost": float(approach.get("cost", 0.0)) + float(push.get("cost", 0.0)),
    }
    # v/omega length should be len(t) (control on the last sample is unused).
    while len(traj["v"]) < len(traj["t"]):
        traj["v"].append(0.0)
        traj["omega"].append(0.0)
    traj["v"] = traj["v"][: len(traj["t"])]
    traj["omega"] = traj["omega"][: len(traj["t"])]
    return traj
