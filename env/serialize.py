"""JSON and natural-language serializations of a layout."""

from __future__ import annotations

import json
import math

from env.geometry import containing_room
from env.models import Layout


def layout_to_json(layout: Layout, *, indent: int | None = 2) -> str:
    return layout.model_dump_json(indent=indent)


def layout_from_json(payload: str | dict) -> Layout:
    if isinstance(payload, str):
        return Layout.model_validate_json(payload)
    return Layout.model_validate(payload)


def scene_text(layout: Layout) -> str:
    """Text description consumed by the LLM planner. Stable for a given layout."""
    lines: list[str] = []
    lines.append(
        f"Floor plan ({layout.topology}, seed {layout.seed}). "
        "Coordinates are meters. theta is radians about +z."
    )
    lines.append("Rooms:")
    for room in layout.rooms:
        lines.append(
            f"- {room.name}: x=[{room.x:.2f}, {room.x + room.w:.2f}], "
            f"y=[{room.y:.2f}, {room.y + room.h:.2f}], size {room.w:.2f}x{room.h:.2f}"
        )
    lines.append("Adjacency (shared doorway):")
    for name, nbrs in layout.adjacency().items():
        lines.append(f"- {name}: {', '.join(nbrs) if nbrs else '(none)'}")
    lines.append("Doorways:")
    blockers = {b.blocking: b.name for b in layout.state.boxes if b.blocking}
    for door in layout.doorways:
        status = f"blocked by {blockers[door.name]}" if door.name in blockers else "open"
        lines.append(
            f"- {door.name}: {door.room_a}--{door.room_b}, "
            f"opening x=[{door.x:.2f}, {door.x + door.w:.2f}], "
            f"y=[{door.y:.2f}, {door.y + door.h:.2f}], status={status}"
        )
    lines.append("Boxes:")
    for box in layout.state.boxes:
        room = containing_room(layout, box.pose.x, box.pose.y)
        where = room if room else (f"in {box.blocking}" if box.blocking else "outside")
        lines.append(
            f"- {box.name}: center ({box.pose.x:.2f}, {box.pose.y:.2f}), "
            f"theta={box.pose.theta:.2f} rad ({math.degrees(box.pose.theta):.0f} deg), "
            f"size {box.sx:.2f}x{box.sy:.2f}, mass={box.mass:.2f} kg, mu={box.mu:.2f}, location={where}"
        )
    robot = layout.state.robot
    rroom = containing_room(layout, robot.pose.x, robot.pose.y)
    lines.append(
        f"Robot: pose ({robot.pose.x:.2f}, {robot.pose.y:.2f}, theta={robot.pose.theta:.2f} rad), "
        f"radius={robot.radius:.2f}, location={rroom}"
    )
    lines.append(f"Goal: {layout.goal.text()}")
    return "\n".join(lines)


def layout_to_dict(layout: Layout) -> dict:
    return json.loads(layout_to_json(layout, indent=None))
