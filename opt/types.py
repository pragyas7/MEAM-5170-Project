"""Results returned by a single motion-planning call."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MotionResult:
    success: bool
    failure_reason: str | None
    message: str
    cost: float = 0.0
    solve_time: float = 0.0
    optimizer_calls: int = 0
    trajectory: dict | None = None
    end_robot: tuple[float, float, float] | None = None
    end_box: tuple[str, float, float, float] | None = None
    solver: str = "none"

    def to_step_dict(self, index: int, action: str) -> dict:
        return {
            "index": index,
            "action": action,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "message": self.message,
            "cost": self.cost,
            "solve_time": self.solve_time,
            "optimizer_calls": self.optimizer_calls,
            "solver": self.solver,
        }


def failure(reason: str, message: str, *, calls: int = 1, solve_time: float = 0.0) -> MotionResult:
    return MotionResult(
        success=False,
        failure_reason=reason,
        message=message,
        solve_time=solve_time,
        optimizer_calls=calls,
    )
