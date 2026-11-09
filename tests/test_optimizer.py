"""Feasibility checks the status report depends on.

polish is off here so the assertions are about the timed unicycle, not about
whether IPOPT beat the analytic solution. A separate test exercises IPOPT.
"""

from env.fixtures import two_room_layout
from env.geometry import pose_blocks_doorway
from opt.checker import check_plan
from opt.config import OptimizerConfig
from opt.paths import polish_unicycle, time_parameterize
from planner.schema import DoneAction, NavigateAction, PushAction


def _cfg(**kwargs) -> OptimizerConfig:
    base = dict(polish=False, horizon_n=21, timeout_s=3.0)
    base.update(kwargs)
    return OptimizerConfig(**base)


def test_open_navigate_succeeds():
    layout = two_room_layout()
    plan = [NavigateAction(action="navigate", region="room_1"), DoneAction(action="done")]
    result = check_plan(layout, plan, _cfg())
    assert result.success, result.message
    assert result.goal_satisfied
    assert result.total_cost >= 0.0
    traj = result.steps[0].trajectory
    assert traj["x"][-1] > 4.4
    # Finite differences stay inside the velocity bound.
    for i in range(len(traj["t"]) - 1):
        dt = traj["t"][i + 1] - traj["t"][i]
        speed = ((traj["x"][i + 1] - traj["x"][i]) ** 2 + (traj["y"][i + 1] - traj["y"][i]) ** 2) ** 0.5 / dt
        assert speed <= 1.05


def test_blocked_navigate_fails_collision():
    layout = two_room_layout(blocked=True)
    plan = [NavigateAction(action="navigate", region="room_1"), DoneAction(action="done")]
    result = check_plan(layout, plan, _cfg())
    assert not result.success
    assert result.failure_reason == "collision"
    assert result.failed_step == 0
    assert "door_0" in result.message


def test_push_moves_box_into_the_other_room():
    layout = two_room_layout(box_in_room=True, goal="push")
    plan = [PushAction(action="push", box="box_0", target_region="room_1"), DoneAction(action="done")]
    result = check_plan(layout, plan, _cfg())
    assert result.success, result.message
    end = result.steps[0].end_state
    assert end.box("box_0").pose.x > 4.4


def test_push_of_blocker_leaves_the_door_clear():
    layout = two_room_layout(blocked=True)
    plan = [PushAction(action="push", box="box_0", target_region="room_1"), DoneAction(action="done")]
    result = check_plan(layout, plan, _cfg())
    assert result.success, result.message
    parked = result.steps[0].end_state.box("box_0")
    assert not pose_blocks_doorway(
        parked, parked.pose.x, parked.pose.y, parked.pose.theta, layout.doorways[0]
    )


def test_push_force_below_friction_is_push_infeasible():
    layout = two_room_layout(box_in_room=True, goal="push")
    plan = [PushAction(action="push", box="box_0", target_region="room_1"), DoneAction(action="done")]
    # mu * m * g = 0.30 * 1.0 * 9.81 = 2.943 N.
    result = check_plan(layout, plan, _cfg(max_push_force=1.0))
    assert not result.success
    assert result.failure_reason == "push_infeasible"
    assert "required_force" in result.message


def test_ipopt_polishes_a_straight_trajectory():
    traj = time_parameterize(
        [(0.0, 0.0), (2.0, 0.0)],
        0.0,
        v_max=1.0,
        w_max=1.0,
        min_turn_radius=0.0,
    )
    assert traj is not None
    polished = polish_unicycle(
        traj,
        v_min=-1.0,
        v_max=1.0,
        w_max=1.0,
        min_turn_radius=0.0,
        corridor=0.25,
        timeout_s=5.0,
        horizon_n=15,
    )
    assert polished is not None, "IPOPT did not return a trajectory"
    assert polished["solver"] == "ipopt"
    assert polished["x"][-1] > 1.5
    assert polished["cost"] <= traj["cost"] + 0.05
