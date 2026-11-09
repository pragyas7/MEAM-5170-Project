"""Hand-built layouts for tests and demos. Not procedural."""

from __future__ import annotations

from env.geometry import build_walls
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


def two_room_layout(
    *,
    blocked: bool = False,
    box_in_room: bool = False,
    goal: str = "navigate",
) -> Layout:
    """Two 4.2 m rooms joined by a 1.4 m doorway.

    `goal='navigate'` asks the robot to reach room_1.
    `goal='push'` asks box_0, which starts in room_0, to reach room_1.
    """
    rooms = [
        Room(name="room_0", x=0.0, y=0.0, w=4.2, h=4.0),
        Room(name="room_1", x=4.4, y=0.0, w=4.2, h=4.0),
    ]
    door = Doorway(name="door_0", room_a="room_0", room_b="room_1", x=4.18, y=1.30, w=0.24, h=1.40)
    boxes: list[BoxState] = []
    if blocked:
        boxes.append(
            BoxState(
                name="box_0",
                pose=Pose(x=door.center[0], y=door.center[1], theta=0.0),
                sx=0.52,
                sy=0.90,
                mass=1.2,
                mu=0.35,
                blocking="door_0",
            )
        )
    elif box_in_room:
        boxes.append(
            BoxState(
                name="box_0",
                pose=Pose(x=2.0, y=2.0, theta=0.0),
                sx=0.40,
                sy=0.40,
                mass=1.0,
                mu=0.30,
                blocking=None,
            )
        )
    robot = RobotState(pose=Pose(x=1.3, y=2.0, theta=0.0), radius=0.18, v_max=1.0, w_max=1.2)
    if goal == "push":
        atoms = [GoalAtom(kind="box_in_room", room="room_1", object="box_0")]
    else:
        atoms = [GoalAtom(kind="robot_in_room", room="room_1")]
    return Layout(
        seed=0,
        split="fixture",
        topology="chain",
        rooms=rooms,
        doorways=[door],
        walls=build_walls(rooms, [door]),
        state=SceneState(robot=robot, boxes=boxes),
        goal=Goal(atoms=atoms),
        n_rooms=2,
        n_boxes=len(boxes),
        n_blockers=1 if blocked else 0,
    )
