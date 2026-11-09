"""Unicycle kinematics.

State (x, y, theta), controls (v, omega). Euler step, which is also the
transcription used by the collocation polish:

    x' = x + dt * v * cos(theta)
    y' = y + dt * v * sin(theta)
    theta' = theta + dt * omega
"""

from __future__ import annotations

import math

import numpy as np


def step_unicycle(
    x: float, y: float, theta: float, v: float, omega: float, dt: float
) -> tuple[float, float, float]:
    return (
        x + dt * v * math.cos(theta),
        y + dt * v * math.sin(theta),
        theta + dt * omega,
    )


def rollout_unicycle(
    x: float,
    y: float,
    theta: float,
    controls: list[tuple[float, float]] | np.ndarray,
    dt: float,
) -> np.ndarray:
    """Integrate controls. Returns an array of shape (len(controls)+1, 3)."""
    states = [(x, y, theta)]
    for v, omega in np.asarray(controls, dtype=float):
        x, y, theta = step_unicycle(x, y, theta, float(v), float(omega), dt)
        states.append((x, y, theta))
    return np.asarray(states, dtype=float)


def trajectory_cost(t: list[float], v: list[float], omega: list[float], effort: float = 0.02) -> float:
    if len(t) < 2:
        return 0.0
    total = 0.0
    for k in range(len(t) - 1):
        dt = t[k + 1] - t[k]
        total += dt + effort * dt * (v[k] ** 2 + omega[k] ** 2)
    return float(total)


def controls_respect_bounds(
    v: list[float],
    omega: list[float],
    *,
    v_min: float,
    v_max: float,
    w_max: float,
    min_turn_radius: float,
    tol: float = 1e-3,
) -> bool:
    for vel, w in zip(v[:-1] if len(v) == len(omega) else v, omega[:-1] if len(omega) == len(v) else omega):
        if vel < v_min - tol or vel > v_max + tol:
            return False
        if abs(w) > w_max + tol:
            return False
        if min_turn_radius > 1e-6 and abs(w) * min_turn_radius > abs(vel) + 0.05:
            return False
    return True
