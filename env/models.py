"""Serializable world models.

Coordinates are meters. Rooms and doorways are axis-aligned rectangles.
Box pose is the center, with theta about +z (radians). The robot is a disc.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Room(BaseModel):
    name: str
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float, margin: float = 0.0) -> bool:
        return (
            self.x + margin <= px <= self.x + self.w - margin
            and self.y + margin <= py <= self.y + self.h - margin
        )

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + 0.5 * self.w, self.y + 0.5 * self.h)


class Doorway(BaseModel):
    name: str
    room_a: str
    room_b: str
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + 0.5 * self.w, self.y + 0.5 * self.h)

    def other(self, room: str) -> str:
        if room == self.room_a:
            return self.room_b
        if room == self.room_b:
            return self.room_a
        raise KeyError(room)


class Wall(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Pose(BaseModel):
    x: float
    y: float
    theta: float = 0.0


class BoxState(BaseModel):
    name: str
    pose: Pose
    sx: float
    sy: float
    mass: float
    mu: float
    # Set only in the initial scene, when this box is wedged in a doorway.
    blocking: str | None = None


class RobotState(BaseModel):
    pose: Pose
    radius: float = 0.18
    v_max: float = 1.0
    w_max: float = 1.2


class SceneState(BaseModel):
    robot: RobotState
    boxes: list[BoxState]

    def box(self, name: str) -> BoxState:
        for b in self.boxes:
            if b.name == name:
                return b
        raise KeyError(name)

    def copy(self) -> SceneState:
        return self.model_copy(deep=True)


class GoalAtom(BaseModel):
    kind: Literal["box_in_room", "robot_in_room"]
    room: str
    object: str | None = None


class Goal(BaseModel):
    atoms: list[GoalAtom] = Field(default_factory=list)

    def text(self) -> str:
        parts: list[str] = []
        for atom in self.atoms:
            if atom.kind == "box_in_room":
                parts.append(f"{atom.object} in {atom.room}")
            else:
                parts.append(f"robot in {atom.room}")
        return " AND ".join(parts) if parts else "(none)"


class Layout(BaseModel):
    seed: int
    split: str = "custom"
    topology: str
    rooms: list[Room]
    doorways: list[Doorway]
    walls: list[Wall]
    state: SceneState
    goal: Goal
    n_rooms: int
    n_boxes: int
    n_blockers: int

    def room(self, name: str) -> Room:
        for r in self.rooms:
            if r.name == name:
                return r
        raise KeyError(name)

    def doorway(self, name: str) -> Doorway:
        for d in self.doorways:
            if d.name == name:
                return d
        raise KeyError(name)

    def room_names(self) -> list[str]:
        return [r.name for r in self.rooms]

    def adjacency(self) -> dict[str, list[str]]:
        graph = {r.name: [] for r in self.rooms}
        for door in self.doorways:
            graph[door.room_a].append(door.room_b)
            graph[door.room_b].append(door.room_a)
        for name in graph:
            graph[name] = sorted(set(graph[name]))
        return graph
