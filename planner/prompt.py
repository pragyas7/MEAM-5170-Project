"""Prompt for the LLM task planner.

The scene text is geometry only. Robot limits come from the optimizer config
that will actually check the plan, so a test-time capability change is visible
in the prompt. E3 asks whether a fine-tuned model uses that text or repeats
motions that were feasible under the training optimizer.
"""

from __future__ import annotations

from env.models import Layout
from env.serialize import scene_text
from opt.config import OptimizerConfig

FEW_SHOT = """
Example 1
Scene:
Rooms:
- room_0: x=[0.00, 4.00], y=[0.00, 4.00]
- room_1: x=[4.20, 8.20], y=[0.00, 4.00]
Doorways:
- door_0: room_0--room_1, status=open
Boxes:
- box_0: center (1.50, 2.00), location=room_0
Robot: pose (1.00, 2.00), location=room_0
Goal: box_0 in room_1
Plan:
[{"action":"push","box":"box_0","target_region":"room_1"},{"action":"done"}]

Example 2
Scene:
Rooms:
- room_0: x=[0.00, 4.00], y=[0.00, 4.00]
- room_1: x=[4.20, 8.20], y=[0.00, 4.00]
Doorways:
- door_0: room_0--room_1, status=blocked by box_0
Boxes:
- box_0: center (4.10, 2.00), location=in door_0
Robot: pose (1.00, 2.00), location=room_0
Goal: robot in room_1
Plan:
[{"action":"navigate","region":"room_0"},{"action":"push","box":"box_0","target_region":"room_1"},{"action":"navigate","region":"room_1"},{"action":"done"}]
""".strip()


def build_prompt(
    layout: Layout,
    config: OptimizerConfig | None = None,
    feedback: list[dict] | None = None,
) -> str:
    config = config or OptimizerConfig()
    limits = (
        f"Robot limits for this query: v_max={config.v_max:.2f} m/s, "
        f"w_max={config.w_max:.2f} rad/s, min_turn_radius={config.min_turn_radius:.2f} m, "
        f"max_push_force={config.max_push_force:.1f} N. "
        "A push is impossible when max_push_force is below mu*mass*9.81."
    )
    parts = [
        "You are a task planner for a differential-drive robot that pushes boxes through a floor plan.",
        "Output ONLY a JSON list of actions. No prose.",
        "",
        "Action schema:",
        '- {"action":"navigate","region":"<room>"}  drive the robot to that room. Blocked doorways cannot be crossed.',
        '- {"action":"push","box":"<box>","target_region":"<room>"}  push the box into an adjacent room. The robot stays rigidly attached to one face of the box while pushing.',
        '- {"action":"push","box":"<box>","target_pose":{"x":0,"y":0,"theta":0}}',
        '- {"action":"wait"}',
        '- {"action":"done"}',
        "",
        "Rules:",
        "- A doorway marked blocked is impassable until the blocking box is pushed into a room.",
        "- Push the blocking box from the side the robot can already reach.",
        "- End every plan with done.",
        "- Use only room and box names that appear in the scene.",
        "",
        limits,
        "",
        FEW_SHOT,
        "",
        "Scene:",
        scene_text(layout),
        "",
        f"Goal: {layout.goal.text()}",
    ]
    if feedback:
        parts.append("")
        parts.append("Previous attempts were rejected by the trajectory optimizer. Fix the plan.")
        for item in feedback[-3:]:
            parts.append(
                f"- attempt {item.get('attempt')}: failure_reason={item.get('failure_reason')} "
                f"failed_step={item.get('failed_step')} message={item.get('message')}"
            )
            parts.append(f"  plan={item.get('plan')}")
    parts.append("")
    parts.append("Plan:")
    return "\n".join(parts)
