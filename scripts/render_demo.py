#!/usr/bin/env python3
"""Render a layout and, when the oracle plan is feasible, a GIF of it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.fixtures import two_room_layout  # noqa: E402
from env.generator import load_split  # noqa: E402
from env.render import save_layout_png, save_plan_gif  # noqa: E402
from opt.checker import check_plan  # noqa: E402
from opt.config import OptimizerConfig  # noqa: E402
from planner.schema import PushAction, DoneAction  # noqa: E402
from planner.symbolic import symbolic_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render layouts and plan GIFs")
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--fixture", choices=["open", "blocked", "push"], default=None)
    parser.add_argument("--out", default=str(ROOT / "results" / "demos"))
    args = parser.parse_args()
    out = Path(args.out)
    if args.fixture == "open":
        layout = two_room_layout()
    elif args.fixture == "blocked":
        layout = two_room_layout(blocked=True)
    elif args.fixture == "push":
        layout = two_room_layout(box_in_room=True, goal="push")
    else:
        layout = load_split(args.split, args.index + 1)[args.index]
    stem = f"{args.fixture}_{layout.seed}" if args.fixture else f"{layout.split}_{layout.seed}"
    png = save_layout_png(layout, out / f"{stem}.png")
    print(f"wrote {png}")
    plan = symbolic_plan(layout)
    if plan is None and args.fixture == "push":
        plan = [PushAction(action="push", box="box_0", target_region="room_1"), DoneAction(action="done")]
    if plan is None:
        print("no symbolic plan")
        return
    config = OptimizerConfig(polish=False)
    result = check_plan(layout, plan, config, store_trajectories=True)
    print(f"success={result.success} reason={result.failure_reason} message={result.message}")
    if any(step.trajectory for step in result.steps):
        gif = save_plan_gif(layout, result.steps, out / f"{stem}.gif")
        print(f"wrote {gif}")


if __name__ == "__main__":
    main()
