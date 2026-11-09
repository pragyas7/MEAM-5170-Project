"""Check a symbolic plan by solving a trajectory optimization for every step.

A plan is feasible when every step succeeds and the goal holds at the end.
Failure reasons are the ones the spec asks the logger to record: collision,
infeasible_dynamics, push_infeasible, timeout, goal_not_reached. Parser
failures use malformed_plan and never reach this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from env.geometry import containing_room
from env.models import Layout, SceneState
from opt.config import OptimizerConfig
from opt.navigate import solve_navigate
from opt.push import solve_push


@dataclass
class StepRecord:
    index: int
    action: str
    success: bool
    failure_reason: str | None
    message: str
    cost: float
    solve_time: float
    optimizer_calls: int
    trajectory: dict | None = None
    end_state: SceneState | None = None

    def to_dict(self, *, include_trajectory: bool = False) -> dict:
        data = {
            "index": self.index,
            "action": self.action,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "message": self.message,
            "cost": self.cost,
            "solve_time": self.solve_time,
            "optimizer_calls": self.optimizer_calls,
        }
        if include_trajectory and self.trajectory is not None:
            data["trajectory"] = self.trajectory
        return data


@dataclass
class PlanResult:
    success: bool
    failure_reason: str | None
    failed_step: int | None
    message: str
    total_cost: float
    total_solve_time: float
    wall_time: float
    optimizer_calls: int
    plan_length: int
    attempts: int
    goal_satisfied: bool
    steps: list[StepRecord] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)

    def to_dict(self, *, include_trajectories: bool = False) -> dict:
        return {
            "success": self.success,
            "failure_reason": self.failure_reason,
            "failed_step": self.failed_step,
            "message": self.message,
            "total_cost": self.total_cost,
            "total_solve_time": self.total_solve_time,
            "wall_time": self.wall_time,
            "optimizer_calls": self.optimizer_calls,
            "plan_length": self.plan_length,
            "attempts": self.attempts,
            "goal_satisfied": self.goal_satisfied,
            "plan": self.plan,
            "steps": [s.to_dict(include_trajectory=include_trajectories) for s in self.steps],
        }


def goal_satisfied(layout: Layout, state: SceneState) -> bool:
    for atom in layout.goal.atoms:
        if atom.kind == "box_in_room":
            box = state.box(atom.object)
            if containing_room(layout, box.pose.x, box.pose.y) != atom.room:
                return False
        elif atom.kind == "robot_in_room":
            robot = state.robot.pose
            if containing_room(layout, robot.x, robot.y) != atom.room:
                return False
        else:
            return False
    return True


def check_plan(
    layout: Layout,
    plan: list,
    config: OptimizerConfig | None = None,
    *,
    store_trajectories: bool = False,
) -> PlanResult:
    config = config or OptimizerConfig()
    t0 = time.perf_counter()
    state = layout.state.copy()
    steps: list[StepRecord] = []
    calls = 0
    cost = 0.0
    solve_time = 0.0
    plan_dicts = [_action_dict(a) for a in plan]
    for index, action in enumerate(plan):
        kind = action.action if hasattr(action, "action") else action["action"]
        if kind == "done":
            break
        if kind == "wait":
            steps.append(
                StepRecord(index, "wait", True, None, "wait", 0.0, 0.0, 0, None, state.copy())
            )
            continue
        if kind == "navigate":
            region = action.region if hasattr(action, "region") else action["region"]
            motion = solve_navigate(layout, state, region, config)
            label = f"navigate({region})"
        elif kind == "push":
            box = action.box if hasattr(action, "box") else action["box"]
            region = action.target_region if hasattr(action, "target_region") else action.get("target_region")
            pose = None
            raw_pose = action.target_pose if hasattr(action, "target_pose") else action.get("target_pose")
            if raw_pose is not None:
                if hasattr(raw_pose, "x"):
                    pose = (raw_pose.x, raw_pose.y, raw_pose.theta)
                else:
                    pose = (raw_pose["x"], raw_pose["y"], raw_pose.get("theta", 0.0))
            motion = solve_push(
                layout,
                state,
                box,
                config,
                target_region=region,
                target_pose=pose,
            )
            label = f"push({box}->{region or 'pose'})"
        else:
            motion_fail = StepRecord(
                index,
                str(kind),
                False,
                "goal_not_reached",
                f"unknown action {kind}",
                0.0,
                0.0,
                0,
                None,
                state,
            )
            steps.append(motion_fail)
            return _finish(
                False,
                "goal_not_reached",
                index,
                motion_fail.message,
                cost,
                solve_time,
                time.perf_counter() - t0,
                calls,
                plan,
                steps,
                False,
                plan_dicts,
            )
        calls += motion.optimizer_calls
        cost += motion.cost
        solve_time += motion.solve_time
        if motion.success:
            state = _apply(state, motion)
        step = StepRecord(
            index=index,
            action=label,
            success=motion.success,
            failure_reason=motion.failure_reason,
            message=motion.message,
            cost=motion.cost,
            solve_time=motion.solve_time,
            optimizer_calls=motion.optimizer_calls,
            trajectory=motion.trajectory if store_trajectories or motion.success else motion.trajectory,
            end_state=state.copy(),
        )
        # Always keep trajectories of successful prefixes so a GIF can replay
        # them. Failed steps keep theirs too; they are small.
        if not store_trajectories:
            step.trajectory = motion.trajectory if motion.success else None
        steps.append(step)
        if not motion.success:
            return _finish(
                False,
                motion.failure_reason,
                index,
                motion.message,
                cost,
                solve_time,
                time.perf_counter() - t0,
                calls,
                plan,
                steps,
                False,
                plan_dicts,
            )
    ok = goal_satisfied(layout, state)
    if not ok:
        return _finish(
            False,
            "goal_not_reached",
            len(steps),
            "plan ended before the goal was satisfied",
            cost,
            solve_time,
            time.perf_counter() - t0,
            calls,
            plan,
            steps,
            False,
            plan_dicts,
        )
    return _finish(
        True,
        None,
        None,
        "goal satisfied",
        cost,
        solve_time,
        time.perf_counter() - t0,
        calls,
        plan,
        steps,
        True,
        plan_dicts,
    )


def _apply(state: SceneState, motion) -> SceneState:
    nxt = state.copy()
    if motion.end_robot is not None:
        nxt.robot.pose.x, nxt.robot.pose.y, nxt.robot.pose.theta = motion.end_robot
    if motion.end_box is not None:
        name, x, y, theta = motion.end_box
        box = nxt.box(name)
        box.pose.x = x
        box.pose.y = y
        box.pose.theta = theta
        box.blocking = None
    return nxt


def _action_dict(action) -> dict:
    if hasattr(action, "model_dump"):
        return action.model_dump(exclude_none=True)
    return dict(action)


def _finish(success, reason, step, message, cost, solve_time, wall, calls, plan, steps, goal_ok, plan_dicts) -> PlanResult:
    return PlanResult(
        success=success,
        failure_reason=reason,
        failed_step=step,
        message=message,
        total_cost=float(cost),
        total_solve_time=float(solve_time),
        wall_time=float(wall),
        optimizer_calls=int(calls),
        plan_length=sum(1 for a in plan if (a.action if hasattr(a, "action") else a["action"]) != "done"),
        attempts=1,
        goal_satisfied=goal_ok,
        steps=steps,
        plan=plan_dicts,
    )
