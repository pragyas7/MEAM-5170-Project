"""Run a configured experiment and write metrics JSON plus plots.

Planners
    naive       one-shot direct plan, no optimizer feedback
    repair      direct, then feedback, then the symbolic oracle (K retries)
    oracle      symbolic plan, one shot
    llm         zero-shot language model, one shot
    llm_repair  language model with optimizer feedback, K retries

`llm` planners are skipped, and the skip is recorded, when no model is configured.
That keeps `scripts/run_baseline.py` runnable on a laptop with no API key.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from env.generator import load_split
from eval.metrics import summarize
from eval.plots import plot_failures, plot_planner_bars
from opt.checker import check_plan
from opt.config import OptimizerConfig, load_config
from planner.llm import LLMError, build_client
from planner.repair import repair
from planner.schema import DoneAction, PlanParseError, parse_plan
from planner.symbolic import direct_plan, symbolic_plan
from planner.prompt import build_prompt


def run_experiment(spec: dict, *, config_path: str | None = None) -> dict:
    """`spec` is the JSON experiment file, already parsed."""
    if "optimizer_configs" in spec:
        return _run_shift(spec)
    config = load_config(spec.get("optimizer_config"))
    return _run_split(spec, config, tag=spec.get("name", "exp"))


def _run_split(spec: dict, config: OptimizerConfig, *, tag: str, config_name: str | None = None) -> dict:
    split = spec["split"]
    n = int(spec["n"])
    planners = list(spec.get("planners", ["naive", "repair", "oracle"]))
    K = int(spec.get("K", 3))
    out_dir = Path(spec.get("output_dir", f"results/{tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    layouts = load_split(split, n)
    client = None
    llm_error = None
    if any(name.startswith("llm") or name.startswith("tuned") for name in planners):
        try:
            client = build_client(
                spec.get("llm_backend", "auto"),
                model=spec.get("model"),
                adapter=spec.get("adapter"),
            )
        except LLMError as exc:
            llm_error = str(exc)
    by_planner: dict[str, list[dict]] = {}
    skipped = []
    for name in planners:
        if name.startswith("llm") and client is None:
            skipped.append({"planner": name, "reason": llm_error})
            continue
        if name.startswith("tuned"):
            adapter = (spec.get("adapters") or {}).get(name)
            if not adapter or not Path(adapter).exists():
                skipped.append({"planner": name, "reason": f"missing adapter for {name}"})
                continue
        records = []
        for index, layout in enumerate(layouts):
            result = run_one(layout, name, config, K=K, client=client, spec=spec)
            row = result.to_dict(include_trajectories=False)
            row["index"] = index
            row["seed"] = layout.seed
            row["topology"] = layout.topology
            records.append(row)
            print(
                f"[{tag}] {name} {index + 1}/{n} seed={layout.seed} "
                f"success={result.success} reason={result.failure_reason} calls={result.optimizer_calls}",
                flush=True,
            )
        by_planner[name] = records
    summary = {
        "name": spec.get("name", tag),
        "split": split,
        "n": n,
        "config_name": config_name,
        "optimizer": config.model_dump(),
        "llm_configured": client is not None,
        "skipped_planners": skipped,
        "planners": {
            name: summarize(records, name=spec.get("name", tag), split=split, planner=name)
            for name, records in by_planner.items()
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "records.jsonl").open("w") as handle:
        for name, records in by_planner.items():
            for row in records:
                handle.write(json.dumps({"planner": name, **row}) + "\n")
    if summary["planners"]:
        plot_planner_bars(summary, out_dir / "success.png")
        plot_failures(summary, out_dir / "failures.png")
    print(json.dumps({k: summary["planners"][k]["success_rate"] for k in summary["planners"]}, indent=2))
    return summary


def _run_shift(spec: dict) -> dict:
    """E3: evaluate the same planners under more than one optimizer config.

    Learned models, when their adapters exist, are evaluated under every
    config. A model trained on config A and tested on config B is the
    capability-shift comparison. The classical planners do not depend on a
    training config; running them under B shows how much harder B is.
    """
    out = {"name": spec.get("name", "E3"), "runs": []}
    base_out = spec.get("output_dir", "results/e3")
    for cfg_name, cfg_path in spec["optimizer_configs"].items():
        child = dict(spec)
        child["optimizer_config"] = cfg_path
        child["output_dir"] = str(Path(base_out) / cfg_name)
        child.pop("optimizer_configs", None)
        summary = _run_split(child, load_config(cfg_path), tag=f"{spec.get('name', 'E3')}-{cfg_name}", config_name=cfg_name)
        out["runs"].append({"config": cfg_name, "metrics": summary})
    Path(base_out).mkdir(parents=True, exist_ok=True)
    (Path(base_out) / "shift.json").write_text(json.dumps(out, indent=2))
    return out


def run_one(layout, planner: str, config: OptimizerConfig, *, K: int = 3, client=None, spec: dict | None = None):
    spec = spec or {}
    t0 = time.perf_counter()
    if planner == "naive":
        result = check_plan(layout, direct_plan(layout), config)
    elif planner == "oracle":
        plan = symbolic_plan(layout) or [DoneAction(action="done")]
        result = check_plan(layout, plan, config)
    elif planner == "repair":
        return repair(layout, config, K=K, mode="heuristic")
    elif planner == "llm":
        result = _llm_once(layout, config, client)
    elif planner == "llm_repair":
        return repair(layout, config, K=K, mode="llm", client=client)
    else:
        raise ValueError(f"unknown planner {planner!r}")
    result.wall_time = time.perf_counter() - t0
    return result


def _llm_once(layout, config, client):
    prompt = build_prompt(layout, config)
    try:
        text = client.complete(prompt, temperature=0.0)
        plan = parse_plan(text)
    except PlanParseError as exc:
        return _malformed(exc.message)
    except Exception as exc:  # network / runtime errors stay visible in the record
        return _malformed(str(exc))
    return check_plan(layout, plan, config)


def _malformed(message: str):
    from opt.checker import PlanResult

    return PlanResult(
        success=False,
        failure_reason="malformed_plan",
        failed_step=0,
        message=message,
        total_cost=0.0,
        total_solve_time=0.0,
        wall_time=0.0,
        optimizer_calls=0,
        plan_length=0,
        attempts=1,
        goal_satisfied=False,
    )


def load_spec(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
