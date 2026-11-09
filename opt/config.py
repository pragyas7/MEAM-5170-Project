"""Capability knobs for the trajectory optimizer.

These are the Q2 levers. A layout describes the scene; a config describes what
the optimizer is allowed to do. Evaluating a model under a config it was not
trained with is the domain-vs-optimizer experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    v_max: float = 1.0
    w_max: float = 1.2
    # 0 allows the unicycle to spin in place. A positive value is a minimum
    # turning radius: paths that need a tighter turn are dynamically infeasible.
    min_turn_radius: float = 0.0
    max_push_force: float = 40.0
    horizon_n: int = 31
    timeout_s: float = 2.5
    goal_pos_tol: float = 0.20
    goal_theta_tol: float = 0.50
    grid_resolution: float = 0.10
    polish: bool = True
    g: float = 9.81

    def scaled_push_yaw(self, mu: float, mass: float) -> tuple[float, float]:
        """Return `(required_force, allowed_yaw_rate)`.

        Quasi-static Coulomb model, rigid face attachment:
        the robot must supply at least `mu * m * g` to slide the box.
        Yaw authority scales with the leftover force budget, so a planner that
        only just overcomes friction cannot also turn the box.
        """
        required = mu * mass * self.g
        if self.max_push_force <= 1e-9:
            return required, 0.0
        margin = max(0.0, (self.max_push_force - required) / self.max_push_force)
        return required, self.w_max * margin


def load_config(path: str | Path | None) -> OptimizerConfig:
    if path is None:
        return OptimizerConfig()
    data = json.loads(Path(path).read_text())
    return OptimizerConfig.model_validate(data)


# Field kept so older notes that mention a default JSON still round-trip.
DEFAULTS = OptimizerConfig()
_ = Field
