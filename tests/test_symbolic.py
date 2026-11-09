from env.fixtures import two_room_layout
from env.generator import load_split
from planner.schema import parse_plan, plan_to_json
from planner.symbolic import direct_plan, symbolic_plan


def test_oracle_finds_a_short_plan_on_the_fixture():
    layout = two_room_layout(box_in_room=True, goal="push")
    plan = symbolic_plan(layout)
    assert plan is not None
    kinds = [a.action for a in plan]
    assert kinds[-1] == "done"
    assert "push" in kinds
    # The parser accepts whatever the oracle emits.
    assert parse_plan(plan_to_json(plan))


def test_oracle_unblocks_before_crossing():
    layout = two_room_layout(blocked=True, goal="navigate")
    plan = symbolic_plan(layout)
    assert plan is not None
    kinds = [a.action for a in plan]
    # The push that clears the door finishes with the robot in room_1, so a
    # separate navigate is unnecessary.
    assert "push" in kinds
    assert kinds[-1] == "done"


def test_direct_plan_names_the_goal_box():
    layout = two_room_layout(box_in_room=True, goal="push")
    plan = direct_plan(layout)
    pushes = [a for a in plan if a.action == "push"]
    assert pushes and pushes[0].box == "box_0"
    assert pushes[0].target_region == "room_1"


def test_train_layouts_have_a_symbolic_plan():
    missing = []
    for layout in load_split("train", 8):
        if symbolic_plan(layout) is None:
            missing.append(layout.seed)
    assert missing == []
