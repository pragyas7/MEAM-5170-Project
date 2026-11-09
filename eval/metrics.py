"""Aggregate PlanResult records into the tables the experiments report."""

from __future__ import annotations

from collections import Counter


def summarize(records: list[dict], *, name: str, split: str, planner: str) -> dict:
    n = len(records)
    successes = [r for r in records if r.get("success")]
    failures = [r for r in records if not r.get("success")]
    return {
        "name": name,
        "split": split,
        "planner": planner,
        "n": n,
        "successes": len(successes),
        "success_rate": (len(successes) / n) if n else 0.0,
        "mean_optimizer_calls": _mean([r.get("optimizer_calls", 0) for r in records]),
        "mean_plan_length": _mean([r.get("plan_length", 0) for r in records]),
        "mean_cost_success": _mean([r.get("total_cost", 0.0) for r in successes]),
        "mean_wall_time": _mean([r.get("wall_time", 0.0) for r in records]),
        "mean_attempts": _mean([r.get("attempts", 1) for r in records]),
        "failure_counts": dict(Counter(r.get("failure_reason") or "unknown" for r in failures)),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
