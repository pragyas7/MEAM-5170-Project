import math

from env.fixtures import two_room_layout
from env.geometry import (
    dist_point_segment,
    door_passable,
    pose_blocks_doorway,
    robot_free,
    sdf_obb,
    segments_intersect,
    subtract_intervals,
)


def test_sdf_obb_signs():
    assert sdf_obb(0, 0, 0, 0, 0, 0.5, 0.5) < 0
    assert sdf_obb(2, 0, 0, 0, 0, 0.5, 0.5) > 0
    # A 45-degree box still contains its center and rejects a far point.
    assert sdf_obb(0, 0, 0, 0, math.pi / 4, 0.4, 0.3) < 0
    assert sdf_obb(3, 3, 0, 0, math.pi / 4, 0.4, 0.3) > 0


def test_segment_distance_and_intersection():
    assert dist_point_segment(0, 1, 0, 0, 2, 0) == 1
    assert segments_intersect((0, 0), (1, 1), (0, 1), (1, 0))
    assert not segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))


def test_subtract_intervals_removes_a_door():
    parts = subtract_intervals(0, 4, [(1.5, 2.5)])
    assert parts == [(0, 1.5), (2.5, 4)]


def test_box_just_inside_a_door_still_blocks_it():
    layout = two_room_layout(blocked=True)
    door = layout.doorways[0]
    box = layout.state.box("box_0")
    # Centered on the opening, one long-side in from the jamb.
    assert pose_blocks_doorway(box, door.center[0] + 0.7, door.center[1], 0.0, door)
    # The same box at the far side of the room leaves the throat open.
    assert not pose_blocks_doorway(box, 7.4, 2.0, 0.0, door)


def test_open_door_is_passable_and_blocker_is_not():
    open_layout = two_room_layout()
    assert robot_free(open_layout, open_layout.state, 1.3, 2.0)
    assert door_passable(open_layout, open_layout.state, open_layout.doorways[0])
    blocked = two_room_layout(blocked=True)
    assert not door_passable(blocked, blocked.state, blocked.doorways[0])
    # The robot itself still fits in its room.
    assert robot_free(blocked, blocked.state, 1.3, 2.0)
