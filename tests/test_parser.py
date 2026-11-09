import pytest

from planner.schema import PlanParseError, parse_plan, plan_to_json


def test_parse_raw_list():
    actions = parse_plan(
        '[{"action":"navigate","region":"room_1"},{"action":"push","box":"box_0","target_region":"room_2"},{"action":"done"}]'
    )
    assert [a.action for a in actions] == ["navigate", "push", "done"]
    assert actions[1].box == "box_0"


def test_parse_fenced_and_wrapped_object():
    text = """Here is the plan:
```json
[{"action": "wait"}, {"action": "done"}]
```
"""
    actions = parse_plan(text)
    assert [a.action for a in actions] == ["wait", "done"]
    wrapped = parse_plan('{"plan":[{"action":"navigate","region":"room_0"},{"action":"done"}]}')
    assert wrapped[0].region == "room_0"


def test_reject_malformed():
    with pytest.raises(PlanParseError) as missing:
        parse_plan('[{"action":"navigate"}]')
    assert missing.value.as_dict()["error"] == "malformed_plan"
    with pytest.raises(PlanParseError):
        parse_plan('[{"action":"fly","region":"room_0"}]')
    with pytest.raises(PlanParseError):
        parse_plan('[{"action":"push","box":"box_0"}]')
    with pytest.raises(PlanParseError):
        parse_plan("I think the robot should just go.")
    with pytest.raises(PlanParseError):
        parse_plan("")


def test_plan_json_roundtrip():
    actions = parse_plan('[{"action":"push","box":"box_0","target_pose":{"x":1,"y":2,"theta":0.5}},{"action":"done"}]')
    again = parse_plan(plan_to_json(actions))
    assert again[0].target_pose.x == 1
    assert again[0].target_region is None
