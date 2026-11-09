"""Label candidate plans with the trajectory optimizer.

Each record is one (scene, goal, plan, success, failure_reason, cost). Feasible
plans become SFT completions. A feasible plan paired with an infeasible plan
for the same scene becomes a DPO preference. Generation runs on CPU; the LoRA
step in `train/sft_lora.py` is the part that needs a GPU.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from env.generator import load_split
from env.models import Layout
from env.serialize import scene_text
from opt.checker import check_plan
from opt.config import OptimizerConfig
from planner.prompt import build_prompt
from planner.schema import DoneAction, NavigateAction, PushAction, plan_to_json
from planner.symbolic import direct_plan, symbolic_plan


def generate(
    split: str,
    n: int,
    out_path: str | Path,
    *,
    candidates: int = 4,
    config: OptimizerConfig | None = None,
    seed: int = 0,
) -> dict:
    config = config or OptimizerConfig(polish=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path = out_path.with_suffix(".sft.jsonl")
    dpo_path = out_path.with_suffix(".dpo.jsonl")
    rng = random.Random(seed)
    layouts = load_split(split, n)
    n_records = 0
    n_success = 0
    with out_path.open("w") as raw, sft_path.open("w") as sft, dpo_path.open("w") as dpo:
        for index, layout in enumerate(layouts):
            plans = _candidates(layout, rng, candidates)
            labeled = []
            for plan in plans:
                result = check_plan(layout, plan, config, store_trajectories=False)
                record = {
                    "split": split,
                    "index": index,
                    "seed": layout.seed,
                    "topology": layout.topology,
                    "scene_text": scene_text(layout),
                    "goal": layout.goal.text(),
                    "plan": result.plan,
                    "success": result.success,
                    "failure_reason": result.failure_reason,
                    "failed_step": result.failed_step,
                    "cost": result.total_cost,
                    "optimizer_calls": result.optimizer_calls,
                    "message": result.message,
                    "optimizer": config.model_dump(),
                }
                raw.write(json.dumps(record) + "\n")
                n_records += 1
                labeled.append(record)
                if result.success:
                    n_success += 1
                    prompt = build_prompt(layout, config)
                    completion = plan_to_json(plan)
                    sft.write(
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": completion},
                                ],
                                "prompt": prompt,
                                "completion": completion,
                                "seed": layout.seed,
                                "cost": result.total_cost,
                            }
                        )
                        + "\n"
                    )
            good = [r for r in labeled if r["success"]]
            bad = [r for r in labeled if not r["success"]]
            if good and bad:
                chosen = min(good, key=lambda r: r["cost"])
                rejected = bad[0]
                dpo.write(
                    json.dumps(
                        {
                            "prompt": build_prompt(layout, config),
                            "chosen": json.dumps(chosen["plan"]),
                            "rejected": json.dumps(rejected["plan"]),
                            "seed": layout.seed,
                        }
                    )
                    + "\n"
                )
    summary = {
        "records": n_records,
        "successes": n_success,
        "layouts": n,
        "out": str(out_path),
        "sft": str(sft_path),
        "dpo": str(dpo_path),
    }
    return summary


def _candidates(layout: Layout, rng: random.Random, m: int) -> list:
    pool = []
    oracle = symbolic_plan(layout)
    if oracle is not None:
        pool.append(oracle)
    pool.append(direct_plan(layout))
    base = oracle or direct_plan(layout)
    while len(pool) < m:
        pool.append(_mutate(base, layout, rng))
    # Deduplicate by JSON text, preserving order.
    seen = set()
    unique = []
    for plan in pool:
        key = plan_to_json(plan)
        if key in seen:
            continue
        seen.add(key)
        unique.append(plan)
    return unique


def _mutate(plan: list, layout: Layout, rng: random.Random) -> list:
    actions = [a.model_copy(deep=True) for a in plan if a.action != "done"]
    if not actions:
        actions = list(direct_plan(layout))
        actions = [a for a in actions if a.action != "done"]
    rooms = layout.room_names()
    boxes = [b.name for b in layout.state.boxes] or ["box_0"]
    roll = rng.random()
    if actions and roll < 0.4:
        action = rng.choice(actions)
        if action.action == "navigate":
            action.region = rng.choice(rooms)
        elif action.action == "push":
            action.target_region = rng.choice(rooms)
    elif actions and roll < 0.7:
        actions.pop(rng.randrange(len(actions)))
    else:
        actions.insert(0, NavigateAction(action="navigate", region=rng.choice(rooms)))
        if boxes:
            actions.append(PushAction(action="push", box=rng.choice(boxes), target_region=rng.choice(rooms)))
    actions.append(DoneAction(action="done"))
    return actions
