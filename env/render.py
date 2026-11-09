"""Matplotlib renderer and GIF export of executed plans."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from env.models import Layout, SceneState
from env.serialize import scene_text

BOX_COLORS = ["#c47b3a", "#3d6b8c", "#6e4a7c", "#3f7d5a", "#a33b3b", "#b0892e", "#2f5d62"]
BG = "#f6f3ec"
ROOM = "#e6eef6"
DOOR = "#e3eadf"
WALL = "#243038"
ROBOT = "#c4492c"
PATH = "#1d4e89"


def render_layout(
    layout: Layout,
    state: SceneState | None = None,
    *,
    trajectory: dict | None = None,
    title: str | None = None,
    ax=None,
    show_goal: bool = True,
):
    state = state or layout.state
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=140)
    else:
        fig = ax.figure
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    for room in layout.rooms:
        ax.add_patch(
            Rectangle((room.x, room.y), room.w, room.h, facecolor=ROOM, edgecolor="none", zorder=0)
        )
    for door in layout.doorways:
        ax.add_patch(
            Rectangle((door.x, door.y), door.w, door.h, facecolor=DOOR, edgecolor="none", zorder=1)
        )
    for wall in layout.walls:
        ax.plot([wall.x1, wall.x2], [wall.y1, wall.y2], color=WALL, lw=2.4, solid_capstyle="butt", zorder=2)
    for room in layout.rooms:
        cx, cy = room.center
        ax.text(cx, cy, room.name, ha="center", va="center", color="#5c6b7a", fontsize=8, zorder=3)
    for i, box in enumerate(state.boxes):
        _draw_box(ax, box.pose.x, box.pose.y, box.pose.theta, box.sx, box.sy, BOX_COLORS[i % len(BOX_COLORS)], box.name)
    if trajectory and trajectory.get("x"):
        ax.plot(trajectory["x"], trajectory["y"], color=PATH, lw=1.6, zorder=4)
    robot = state.robot
    ax.add_patch(Circle((robot.pose.x, robot.pose.y), robot.radius, facecolor=ROBOT, edgecolor="#3a1a12", lw=0.6, zorder=5))
    hx = robot.pose.x + math.cos(robot.pose.theta) * robot.radius * 1.4
    hy = robot.pose.y + math.sin(robot.pose.theta) * robot.radius * 1.4
    ax.add_patch(
        FancyArrowPatch(
            (robot.pose.x, robot.pose.y),
            (hx, hy),
            arrowstyle="-|>",
            mutation_scale=8,
            color="#3a1a12",
            lw=0.8,
            zorder=6,
        )
    )
    if show_goal:
        ax.text(
            0.01,
            0.99,
            "goal: " + layout.goal.text(),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#1e2a24",
            wrap=True,
        )
    xs = [r.x for r in layout.rooms] + [r.x + r.w for r in layout.rooms]
    ys = [r.y for r in layout.rooms] + [r.y + r.h for r in layout.rooms]
    pad = 0.45
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"{layout.split} · {layout.topology} · seed {layout.seed}", fontsize=11)
    if own:
        fig.tight_layout()
    return fig, ax


def _draw_box(ax, x, y, theta, sx, sy, color, name):
    c, s = math.cos(theta), math.sin(theta)
    hx, hy = sx / 2, sy / 2
    local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    world = [(x + c * lx - s * ly, y + s * lx + c * ly) for lx, ly in local]
    ax.add_patch(Polygon(world, closed=True, facecolor=color, edgecolor="#2a241c", lw=0.7, alpha=0.92, zorder=4))
    ax.text(x, y, name, ha="center", va="center", fontsize=7, color="white", zorder=5)


def save_layout_png(layout: Layout, path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = render_layout(layout, **kwargs)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_plan_gif(
    layout: Layout,
    steps: list,
    path: str | Path,
    *,
    fps: int = 12,
    max_frames: int = 80,
) -> Path:
    """Replay step trajectories from the initial state and write a GIF."""
    import imageio.v2 as imageio
    import io
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = layout.state.copy()
    frames_state: list[tuple[SceneState, dict | None]] = [(state.copy(), None)]
    for step in steps:
        traj = getattr(step, "trajectory", None) or (step.get("trajectory") if isinstance(step, dict) else None)
        if not traj or not traj.get("x"):
            if getattr(step, "end_state", None) is not None:
                state = step.end_state.copy()
            continue
        n = len(traj["x"])
        box_name = traj.get("box_name")
        for i in range(n):
            state = state.copy()
            state.robot.pose.x = float(traj["x"][i])
            state.robot.pose.y = float(traj["y"][i])
            state.robot.pose.theta = float(traj["theta"][i])
            if box_name and traj.get("box_x"):
                box = state.box(box_name)
                box.pose.x = float(traj["box_x"][i])
                box.pose.y = float(traj["box_y"][i])
                box.pose.theta = float(traj["box_theta"][i])
                box.blocking = None
            if i % max(1, n // 24) == 0 or i == n - 1:
                frames_state.append((state.copy(), traj))
        end = getattr(step, "end_state", None)
        if end is not None:
            state = end.copy()
    if len(frames_state) > max_frames:
        idx = np.linspace(0, len(frames_state) - 1, max_frames).astype(int)
        frames_state = [frames_state[i] for i in idx]
    images = []
    for st, traj in frames_state:
        fig, ax = render_layout(layout, st, trajectory=traj, show_goal=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        images.append(imageio.imread(buf))
    imageio.mimsave(path, images, duration=1.0 / fps)
    return path


def caption(layout: Layout) -> str:
    return scene_text(layout)
