"""Iterative repair.

The proposer gets the optimizer's failure reason and tries again, up to K
retries (K=3, so four proposals). The default proposer is classical:

0. direct plan (the untrained guess)
1. the same goal with explicit unblocks for doors named in the failure
2. the symbolic oracle
3. the symbolic oracle again only if the previous plan differed

`llm` mode uses the same loop with the language model as the proposer.
"""

from __future__ import annotations

import time

from env.models import Layout
from opt.checker import PlanResult, check_plan
from opt.config import OptimizerConfig
from planner.prompt import build_prompt
from planner.schema import PlanParseError, parse_plan, plan_to_json
from planner.symbolic import direct_plan, feedback_plan, symbolic_plan


def repair(
    layout: Layout,
    config: OptimizerConfig | None = None,
    *,
    K: int = 3,
    mode: str = "heuristic",
    client=None,
    store_trajectories: bool = False,
) -> PlanResult:
    config = config or OptimizerConfig()
    feedback: list[dict] = []
    calls = 0
    t0 = time.perf_counter()
    last: PlanResult | None = None
    seen: set[str] = set()
    for attempt in range(K + 1):
        try:
            plan = _propose(layout, config, feedback, attempt, mode=mode, client=client)
        except PlanParseError as exc:
            last = PlanResult(
                success=False,
                failure_reason="malformed_plan",
                failed_step=0,
                message=exc.message,
                total_cost=0.0,
                total_solve_time=0.0,
                wall_time=0.0,
                optimizer_calls=calls,
                plan_length=0,
                attempts=attempt + 1,
                goal_satisfied=False,
                steps=[],
                plan=[],
            )
            feedback.append(
                {
                    "attempt": attempt,
                    "plan": [],
                    "failure_reason": "malformed_plan",
                    "failed_step": 0,
                    "message": exc.message,
                }
            )
            continue
        signature = plan_to_json(plan)
        if signature in seen and attempt > 0:
            # Don't spend another optimizer call on a plan we already rejected.
            continue
        seen.add(signature)
        result = check_plan(layout, plan, config, store_trajectories=store_trajectories)
        calls += result.optimizer_calls
        result.attempts = attempt + 1
        result.optimizer_calls = calls
        result.wall_time = time.perf_counter() - t0
        last = result
        if result.success:
            return result
        feedback.append(
            {
                "attempt": attempt,
                "plan": result.plan,
                "failure_reason": result.failure_reason,
                "failed_step": result.failed_step,
                "message": result.message,
            }
        )
    assert last is not None
    last.optimizer_calls = calls
    last.attempts = K + 1
    last.wall_time = time.perf_counter() - t0
    return last


def _propose(layout, config, feedback, attempt: int, *, mode: str, client):
    if mode == "llm":
        if client is None:
            raise PlanParseError("no LLM client configured")
        prompt = build_prompt(layout, config, feedback)
        temperature = 0.0 if attempt == 0 else 0.4
        text = client.complete(prompt, temperature=temperature)
        return parse_plan(text)
    if attempt == 0:
        return direct_plan(layout)
    if attempt == 1 and feedback:
        return feedback_plan(layout, feedback[-1].get("message") or "")
    planned = symbolic_plan(layout)
    if planned is not None:
        return planned
    if feedback:
        return feedback_plan(layout, feedback[-1].get("message") or "")
    return direct_plan(layout)
