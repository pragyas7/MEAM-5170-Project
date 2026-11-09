import math

from opt.dynamics import controls_respect_bounds, rollout_unicycle, step_unicycle


def _close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, (a, b)


def test_straight_line_rollout():
    states = rollout_unicycle(0, 0, 0, [(1.0, 0.0)] * 5, dt=0.2)
    assert states.shape == (6, 3)
    _close(states[-1, 0], 1.0)
    _close(states[-1, 1], 0.0)


def test_spin_in_place():
    x, y, th = step_unicycle(1.0, 2.0, 0.0, 0.0, math.pi, dt=0.5)
    _close(x, 1.0)
    _close(y, 2.0)
    _close(th, math.pi / 2)


def test_bounds_helper():
    assert controls_respect_bounds([0.5, 0.0], [0.2, 0.0], v_min=0.0, v_max=1.0, w_max=1.0, min_turn_radius=0.0)
    assert not controls_respect_bounds([0.1, 0.0], [2.0, 0.0], v_min=0.0, v_max=1.0, w_max=1.0, min_turn_radius=0.0)
    assert not controls_respect_bounds(
        [0.1, 0.0], [1.0, 0.0], v_min=0.0, v_max=1.0, w_max=2.0, min_turn_radius=1.0
    )


