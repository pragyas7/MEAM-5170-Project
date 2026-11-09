"""Navigate: drive the robot disc to a room or to a specific pose.

The geometric skeleton is grid A* on free space eroded by the robot radius.
The skeleton is then timed as a unicycle trajectory (stop-and-spin when the
minimum turning radius is zero, circular fillets otherwise) and, when polish
is on, shortened by IPOPT direct collocation. Feasibility is the post-check:
controls inside the capability bounds, and the disc stays out of walls and boxes.
"""

from __future__ import annotations

import math
import time

from env.geometry import containing_room, robot_free, wrap_angle
from env.models import Layout, SceneState
from opt.config import OptimizerConfig
from opt.grid import build_grid, cells_in_room, closest_cell, plan_xy
from opt.paths import goal_reached, polish_unicycle, time_parameterize
from opt.types import MotionResult, failure


def solve_navigate(
    layout: Layout,
    state: SceneState,
    region: str | None,
    config: OptimizerConfig,
    *,
    goal_pose: tuple[float, float, float] | None = None,
    ignore_box: str | None = None,
) -> MotionResult:
    """Drive the robot to `region`, or to `goal_pose` when that is given.

    `ignore_box` lets a push approach treat the box it is about to contact as
    the target rather than as an obstacle it must stay a full radius away from.
    The contact pose is already outside that box.
    """
    t0 = time.perf_counter()
    robot = state.robot
    start = (robot.pose.x, robot.pose.y, robot.pose.theta)
    if region is not None and region not in layout.room_names():
        return failure("goal_not_reached", f"navigate region={region} failed: goal_not_reached. unknown region")

    if goal_pose is None:
        assert region is not None
        if containing_room(layout, start[0], start[1]) == region:
            traj = _stationary(start)
            return _ok(traj, time.perf_counter() - t0, calls=0, message=f"navigate region={region} already satisfied")
        goal_xy, goal_th = _room_goal(layout, state, region, start, config, ignore_box)
        if goal_xy is None:
            return failure(
                "collision",
                f"navigate region={region} failed: collision. {_blocked_text(layout, state)}",
                solve_time=time.perf_counter() - t0,
            )
    else:
        goal_xy = (goal_pose[0], goal_pose[1])
        goal_th = goal_pose[2]
        if (
            math.hypot(start[0] - goal_xy[0], start[1] - goal_xy[1]) <= config.goal_pos_tol
            and abs(wrap_angle(start[2] - goal_th)) <= config.goal_theta_tol
        ):
            traj = _stationary(start)
            return _ok(traj, time.perf_counter() - t0, calls=0, message="navigate pose already satisfied")

    grid = build_grid(
        layout,
        state,
        robot.radius,
        resolution=config.grid_resolution,
        ignore_box=ignore_box,
    )
    path = plan_xy(grid, (start[0], start[1]), goal_xy)
    if path is None:
        label = region or "pose"
        return failure(
            "collision",
            f"navigate region={label} failed: collision. {_blocked_text(layout, state)}",
            solve_time=time.perf_counter() - t0,
        )

    final_theta = goal_th if goal_pose is not None else None
    traj = _time_path(path, start[2], config, final_theta)
    if traj is None or not _robot_safe(layout, state, traj, ignore_box=ignore_box):
        # The grid shortcut can clip a jamb: a point inside a free cell may
        # still be closer to the wall than the robot radius. String-pull the
        # cell-center polyline with the same disc check the post-check uses.
        dense = plan_xy(grid, (start[0], start[1]), goal_xy, simplify=False)
        if dense is not None:
            pulled = _continuous_shortcut(layout, state, dense, ignore_box)
            retried = _time_path(pulled, start[2], config, final_theta)
            if retried is not None and _robot_safe(layout, state, retried, ignore_box=ignore_box):
                traj = retried
    if traj is None:
        return failure(
            "infeasible_dynamics",
            (
                f"navigate region={region or 'pose'} failed: infeasible_dynamics. "
                f"v_max={config.v_max:.2f} w_max={config.w_max:.2f} "
                f"min_turn_radius={config.min_turn_radius:.2f}"
            ),
            solve_time=time.perf_counter() - t0,
        )

    analytic = traj
    polished = _maybe_polish(analytic, config, v_min=0.0)
    traj = analytic
    if polished is not analytic and _accept(layout, state, polished, region, goal_pose, goal_xy, goal_th, config, ignore_box):
        traj = polished
    if not _robot_safe(layout, state, traj, ignore_box=ignore_box):
        return failure(
            "collision",
            f"navigate region={region or 'pose'} failed: collision. timed path clips an obstacle. {_blocked_text(layout, state)}",
            solve_time=time.perf_counter() - t0,
        )
    if not _accept(layout, state, traj, region, goal_pose, goal_xy, goal_th, config, ignore_box):
        return failure(
            "goal_not_reached",
            f"navigate region={region or 'pose'} failed: goal_not_reached.",
            solve_time=time.perf_counter() - t0,
        )
    return _ok(
        traj,
        time.perf_counter() - t0,
        calls=1,
        message=f"navigate region={region or 'pose'} ok solver={traj['solver']}",
    )


def _time_path(path, start_theta, config, final_theta):
    return time_parameterize(
        path,
        start_theta,
        v_max=config.v_max,
        w_max=config.w_max,
        min_turn_radius=config.min_turn_radius,
        final_theta=final_theta,
        v_min=0.0,
    )


def _continuous_shortcut(layout, state, points, ignore_box) -> list[tuple[float, float]]:
    """String-pull a polyline, accepting a segment only when the disc stays free."""
    if len(points) <= 2:
        return list(points)

    def clear(a, b) -> bool:
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(2, int(dist / 0.05))
        for i in range(steps + 1):
            u = i / steps
            x = a[0] + u * (b[0] - a[0])
            y = a[1] + u * (b[1] - a[1])
            if not robot_free(layout, state, x, y, ignore_box=ignore_box, margin=0.0):
                return False
        return True

    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not clear(points[i], points[j]):
            j -= 1
        out.append(points[j])
        i = j
    return out


def _room_goal(layout, state, region, start, config, ignore_box):
    grid = build_grid(
        layout,
        state,
        state.robot.radius,
        resolution=config.grid_resolution,
        ignore_box=ignore_box,
    )
    cells = cells_in_room(grid, layout, region, margin=0.35)
    if not cells:
        cells = cells_in_room(grid, layout, region, margin=0.05)
    cell = closest_cell(cells, (start[0], start[1]))
    if cell is None:
        return None, None
    return grid.world(*cell), None


def _maybe_polish(traj: dict, config: OptimizerConfig, *, v_min: float) -> dict:
    if not config.polish:
        return traj
    polished = polish_unicycle(
        traj,
        v_min=v_min,
        v_max=config.v_max,
        w_max=config.w_max,
        min_turn_radius=config.min_turn_radius,
        corridor=0.18,
        timeout_s=config.timeout_s,
        horizon_n=config.horizon_n,
    )
    if polished is None:
        return traj
    if polished["cost"] <= traj["cost"] + 1e-6:
        return polished
    return traj


def _robot_safe(layout, state, traj, ignore_box=None) -> bool:
    # Check knots and midpoints. The disc is rotation-invariant.
    xs = traj["x"]
    ys = traj["y"]
    for i in range(len(xs)):
        if not robot_free(layout, state, xs[i], ys[i], ignore_box=ignore_box, margin=-0.015):
            return False
        if i + 1 < len(xs):
            mx = 0.5 * (xs[i] + xs[i + 1])
            my = 0.5 * (ys[i] + ys[i + 1])
            if not robot_free(layout, state, mx, my, ignore_box=ignore_box, margin=-0.015):
                return False
    return True


def _accept(layout, state, traj, region, goal_pose, goal_xy, goal_th, config, ignore_box) -> bool:
    if not _robot_safe(layout, state, traj, ignore_box=ignore_box):
        return False
    if goal_pose is None:
        return containing_room(layout, traj["x"][-1], traj["y"][-1]) == region
    return goal_reached(
        traj,
        goal_xy,
        config.goal_pos_tol,
        theta=goal_th,
        theta_tol=config.goal_theta_tol,
    )


def _stationary(start: tuple[float, float, float]) -> dict:
    x, y, th = start
    return {
        "t": [0.0, 0.05],
        "x": [x, x],
        "y": [y, y],
        "theta": [th, th],
        "v": [0.0, 0.0],
        "omega": [0.0, 0.0],
        "solver": "analytic",
        "cost": 0.0,
    }


def _ok(traj: dict, solve_time: float, *, calls: int, message: str) -> MotionResult:
    return MotionResult(
        success=True,
        failure_reason=None,
        message=message,
        cost=float(traj.get("cost", 0.0)),
        solve_time=solve_time,
        optimizer_calls=calls,
        trajectory=traj,
        end_robot=(traj["x"][-1], traj["y"][-1], traj["theta"][-1]),
        solver=traj.get("solver", "analytic"),
    )


def _blocked_text(layout: Layout, state: SceneState) -> str:
    from env.geometry import blocking_box_name, door_passable

    parts = []
    for door in layout.doorways:
        if door_passable(layout, state, door):
            continue
        parts.append(f"{door.name}:{blocking_box_name(layout, state, door)}")
    return "blocked_doors=" + (",".join(parts) if parts else "none")
