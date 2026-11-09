"""Trajectory optimizers and the plan-level feasibility checker."""

from opt.checker import check_plan
from opt.config import OptimizerConfig

__all__ = ["OptimizerConfig", "check_plan"]
