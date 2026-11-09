"""Turn a collision-free polyline into a dynamically feasible unicycle trajectory.

With min_turn_radius == 0 the unicycle stops and spins, then drives straight.
With a positive minimum turning radius, corners are replaced by circular fillets
of that radius. If a corner does not have room for the fillet, the path is
dynamically infeasible.

An optional IPOPT polish (CasADi) shortens the same trajectory. The analytic
timing is already a feasible solution, so a polish failure does not make a
feasible plan infeasible.
"""

from __future__ import annotations

import math

import numpy as np

from env.geometry import wrap_angle
from opt.dynamics import trajectory_cost


def time_parameterize(
    points: list[tuple[float, float]],
    start_theta: float,
    *,
    v_max: float,
    w_max: float,
    min_turn_radius: float,
    final_theta: float | None = None,
    v_min: float = 0.0,
    sample_dt: float = 0.08,
) -> dict | None:
    """Return a trajectory dict, or None if the bounds cannot track the path."""
    if not points:
        return None
    if len(points) == 1 or _path_length(points) < 1e-3:
        return _wait(points[0][0], points[0][1], start_theta)
    if v_max <= 1e-6:
        return None
    # A zero yaw rate can still track a straight line that is already aligned.
    if w_max <= 1e-8 and not _is_aligned_straight(points, start_theta, final_theta):
        return None
    cruise = max(min(0.15, v_max), min(0.8 * v_max, v_max))
    if min_turn_radius > 1e-4:
        expanded = _fillet(points, min_turn_radius)
        if expanded is None:
            return None
        return _sample_filleted(
            expanded,
            start_theta,
            v_max=v_max,
            w_max=w_max,
            cruise=min(cruise, w_max * min_turn_radius),
            final_theta=final_theta,
            sample_dt=sample_dt,
            min_turn_radius=min_turn_radius,
        )
    return _sample_spin_drive(
        points,
        start_theta,
        v_max=v_max,
        w_max=w_max,
        cruise=cruise,
        v_min=v_min,
        final_theta=final_theta,
        sample_dt=sample_dt,
    )


def _path_length(points: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])
    )


def _wait(x: float, y: float, theta: float) -> dict:
    return _pack([(0.0, x, y, theta, 0.0, 0.0), (0.05, x, y, theta, 0.0, 0.0)], "analytic")


def _pack(samples: list[tuple[float, float, float, float, float, float]], solver: str) -> dict:
    """samples are (t, x, y, theta, v, omega) with v, omega applied until the next sample."""
    t, x, y, th, v, w = zip(*samples)
    traj = {
        "t": [float(v_) for v_ in t],
        "x": [float(v_) for v_ in x],
        "y": [float(v_) for v_ in y],
        "theta": [float(v_) for v_ in th],
        "v": [float(v_) for v_ in v],
        "omega": [float(v_) for v_ in w],
        "solver": solver,
    }
    traj["cost"] = trajectory_cost(traj["t"], traj["v"], traj["omega"])
    return traj


def _sample_spin_drive(
    points,
    start_theta,
    *,
    v_max,
    w_max,
    cruise,
    v_min,
    final_theta,
    sample_dt,
) -> dict | None:
    theta = start_theta
    # Primitives: ("spin", x, y, dtheta) or ("drive", x0, y0, x1, y1, heading).
    prims: list[tuple] = []
    cursor = points[0]
    heading = start_theta
    for nxt in points[1:]:
        dx = nxt[0] - cursor[0]
        dy = nxt[1] - cursor[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            cursor = nxt
            continue
        heading = math.atan2(dy, dx)
        dth = wrap_angle(heading - theta)
        if abs(dth) > 1e-3:
            if w_max <= 1e-8:
                return None
            prims.append(("spin", cursor[0], cursor[1], dth))
            theta = heading
        prims.append(("drive", cursor[0], cursor[1], nxt[0], nxt[1], heading))
        cursor = nxt
        theta = heading
    if final_theta is not None:
        dth = wrap_angle(final_theta - theta)
        if abs(dth) > 1e-3:
            if w_max <= 1e-8:
                return None
            prims.append(("spin", cursor[0], cursor[1], dth))
            theta = final_theta
    samples = [(0.0, points[0][0], points[0][1], start_theta, 0.0, 0.0)]
    t = 0.0
    th = start_theta
    x, y = points[0]
    for prim in prims:
        if prim[0] == "spin":
            _, sx, sy, dth = prim
            duration = abs(dth) / w_max
            omega = 0.0 if duration < 1e-9 else dth / duration
            n = max(1, int(math.ceil(duration / sample_dt)))
            for i in range(1, n + 1):
                tau = duration * i / n
                samples.append((t + tau, sx, sy, th + omega * tau, 0.0, omega))
            # The control stored on a sample is the control that *produced* it,
            # except the first sample. Rebuild controls after the loop.
            t += duration
            th = th + dth
            x, y = sx, sy
        else:
            _, x0, y0, x1, y1, face = prim
            dist = math.hypot(x1 - x0, y1 - y0)
            speed = min(v_max, max(cruise, v_min))
            duration = dist / speed
            n = max(1, int(math.ceil(duration / sample_dt)))
            for i in range(1, n + 1):
                u = i / n
                samples.append(
                    (
                        t + duration * u,
                        x0 + u * (x1 - x0),
                        y0 + u * (y1 - y0),
                        face,
                        speed,
                        0.0,
                    )
                )
            t += duration
            th = face
            x, y = x1, y1
    return _with_interval_controls(samples, v_min=v_min, v_max=v_max, w_max=w_max)


def _with_interval_controls(samples, *, v_min, v_max, w_max) -> dict | None:
    """Recompute (v, omega) so each Euler step matches the next sample."""
    packed = _controls_from_states(samples, v_min=v_min, v_max=v_max, w_max=w_max)
    for k in range(len(packed) - 1):
        t0, x0, y0, th0, v, omega = packed[k]
        t1, x1, y1, _, _, _ = packed[k + 1]
        dt = t1 - t0
        if dt <= 1e-9:
            return None
        c, s = math.cos(th0), math.sin(th0)
        lateral = abs(-(x1 - x0) * s + (y1 - y0) * c) / dt
        if lateral > 2e-2:
            return None
        if v < v_min - 1e-2 or v > v_max + 1e-2 or abs(omega) > w_max + 1e-2:
            return None
    return _pack(packed, "analytic")


def _is_aligned_straight(points, start_theta, final_theta, tol: float = 0.2) -> bool:
    if _path_length(points) < 1e-3:
        return True
    for a, b in zip(points, points[1:]):
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-4:
            continue
        heading = math.atan2(b[1] - a[1], b[0] - a[0])
        if abs(wrap_angle(heading - start_theta)) > tol:
            return False
        start_theta = heading
    if final_theta is not None and abs(wrap_angle(final_theta - start_theta)) > tol:
        return False
    return True


def _controls_from_states(samples, *, v_min, v_max, w_max) -> list[tuple]:
    out = []
    thetas = []
    th = samples[0][3]
    thetas.append(th)
    for k in range(len(samples) - 1):
        th = th + wrap_angle(samples[k + 1][3] - samples[k][3])
        thetas.append(th)
    for k in range(len(samples) - 1):
        t0, x0, y0 = samples[k][0], samples[k][1], samples[k][2]
        t1, x1, y1 = samples[k + 1][0], samples[k + 1][1], samples[k + 1][2]
        dt = max(1e-6, t1 - t0)
        th0 = thetas[k]
        c, s = math.cos(th0), math.sin(th0)
        v = ((x1 - x0) * c + (y1 - y0) * s) / dt
        omega = (thetas[k + 1] - thetas[k]) / dt
        v = min(max(v, v_min), v_max)
        omega = min(max(omega, -w_max), w_max)
        out.append((t0, x0, y0, th0, v, omega))
    last = samples[-1]
    out.append((last[0], last[1], last[2], thetas[-1], 0.0, 0.0))
    return out


def _fillet(points: list[tuple[float, float]], radius: float) -> list[tuple] | None:
    """Replace corners with arcs.

    Returns a list of ("line", p0, p1) and ("arc", center, radius, a0, sweep) pieces,
    or None if some corner cannot host the requested radius.
    """
    pts = [np.asarray(p, dtype=float) for p in points]
    cleaned = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - cleaned[-1]) > 1e-3:
            cleaned.append(p)
    pts = cleaned
    if len(pts) < 2:
        return [("line", pts[0], pts[0])]
    if len(pts) == 2:
        return [("line", pts[0], pts[1])]
    pieces: list[tuple] = []
    cursor = pts[0]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        d1 = b - a
        d2 = c - b
        l1 = float(np.linalg.norm(d1))
        l2 = float(np.linalg.norm(d2))
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        u1 = d1 / l1
        u2 = d2 / l2
        cross = float(u1[0] * u2[1] - u1[1] * u2[0])
        dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        delta = math.atan2(abs(cross), dot)  # magnitude of the turn
        if delta < 1e-3:
            continue
        tdist = radius * math.tan(delta / 2.0)
        # Leave a little of the incoming segment so successive fillets do not overlap.
        avail_in = float(np.linalg.norm(b - cursor))
        if tdist > avail_in - 1e-3 or tdist > l2 - 1e-3:
            return None
        t1 = b - u1 * tdist
        t2 = b + u2 * tdist
        side = 1.0 if cross > 0 else -1.0
        normal = np.array([-u1[1], u1[0]]) * side
        center = t1 + normal * radius
        a0 = math.atan2(t1[1] - center[1], t1[0] - center[0])
        a1 = math.atan2(t2[1] - center[1], t2[0] - center[0])
        sweep = wrap_angle(a1 - a0)
        if side > 0 and sweep < 0:
            sweep += 2 * math.pi
        if side < 0 and sweep > 0:
            sweep -= 2 * math.pi
        if float(np.linalg.norm(t1 - cursor)) > 1e-3:
            pieces.append(("line", cursor, t1))
        pieces.append(("arc", center, radius, a0, sweep, side))
        cursor = t2
    if float(np.linalg.norm(pts[-1] - cursor)) > 1e-3:
        pieces.append(("line", cursor, pts[-1]))
    return pieces


def _sample_filleted(
    pieces,
    start_theta,
    *,
    v_max,
    w_max,
    cruise,
    final_theta,
    sample_dt,
    min_turn_radius,
) -> dict | None:
    speed = min(v_max, cruise)
    if min_turn_radius > 1e-6:
        speed = min(speed, w_max * min_turn_radius)
    if speed <= 1e-4:
        return None
    samples = []
    t = 0.0
    th = start_theta
    # Start point.
    p0 = _piece_start(pieces[0])
    samples.append((0.0, float(p0[0]), float(p0[1]), th, 0.0, 0.0))
    for piece in pieces:
        if piece[0] == "line":
            _, a, b = piece
            dist = float(np.linalg.norm(b - a))
            if dist < 1e-4:
                continue
            heading = math.atan2(float(b[1] - a[1]), float(b[0] - a[0]))
            dth = wrap_angle(heading - th)
            # A fillet should already align headings. A small residual is a spin,
            # which a positive turning radius forbids.
            if abs(dth) > 0.15:
                return None
            duration = dist / speed
            n = max(1, int(math.ceil(duration / sample_dt)))
            for i in range(1, n + 1):
                u = i / n
                samples.append(
                    (
                        t + duration * u,
                        float(a[0] + u * (b[0] - a[0])),
                        float(a[1] + u * (b[1] - a[1])),
                        heading,
                        speed,
                        0.0,
                    )
                )
            t += duration
            th = heading
        else:
            _, center, radius, a0, sweep, side = piece
            arc_len = abs(sweep) * radius
            duration = arc_len / speed
            omega = (sweep / duration) if duration > 1e-8 else 0.0
            if abs(omega) > w_max + 1e-3:
                return None
            n = max(2, int(math.ceil(duration / sample_dt)))
            for i in range(1, n + 1):
                u = i / n
                ang = a0 + sweep * u
                # Tangent heading is the radius angle plus/minus pi/2.
                heading = ang + side * (math.pi / 2)
                samples.append(
                    (
                        t + duration * u,
                        float(center[0] + radius * math.cos(ang)),
                        float(center[1] + radius * math.sin(ang)),
                        heading,
                        speed,
                        omega,
                    )
                )
            t += duration
            th = samples[-1][3]
    if final_theta is not None and abs(wrap_angle(final_theta - th)) > 0.2:
        # Cannot spin in place under a turning-radius constraint.
        return None
    return _pack(_controls_from_states(samples, v_min=0.0, v_max=v_max, w_max=w_max), "analytic")


def _piece_start(piece):
    if piece[0] == "line":
        return piece[1]
    _, center, radius, a0, sweep, side = piece
    return center + radius * np.array([math.cos(a0), math.sin(a0)])


def polish_unicycle(
    traj: dict,
    *,
    v_min: float,
    v_max: float,
    w_max: float,
    min_turn_radius: float,
    corridor: float,
    timeout_s: float,
    horizon_n: int,
) -> dict | None:
    """Shorten `traj` with direct collocation. None if CasADi/IPOPT cannot."""
    try:
        import casadi as ca
    except Exception:
        return None
    ref = _resample_traj(traj, horizon_n)
    if ref is None:
        return None
    n = ref.shape[0]
    opti = ca.Opti()
    x = opti.variable(n)
    y = opti.variable(n)
    th = opti.variable(n)
    v = opti.variable(n - 1)
    w = opti.variable(n - 1)
    dt = opti.variable()
    opti.subject_to(opti.bounded(0.02, dt, 1.5))
    opti.subject_to(x[0] == ref[0, 0])
    opti.subject_to(y[0] == ref[0, 1])
    opti.subject_to(th[0] == ref[0, 2])
    for k in range(n - 1):
        opti.subject_to(x[k + 1] == x[k] + dt * v[k] * ca.cos(th[k]))
        opti.subject_to(y[k + 1] == y[k] + dt * v[k] * ca.sin(th[k]))
        opti.subject_to(th[k + 1] == th[k] + dt * w[k])
        opti.subject_to(opti.bounded(v_min, v[k], v_max))
        opti.subject_to(opti.bounded(-w_max, w[k], w_max))
        if min_turn_radius > 1e-4:
            opti.subject_to(v[k] * v[k] >= (min_turn_radius * w[k]) ** 2)
    for k in range(n):
        opti.subject_to((x[k] - ref[k, 0]) ** 2 + (y[k] - ref[k, 1]) ** 2 <= corridor**2)
    opti.minimize(dt * (n - 1) + 0.02 * dt * ca.sumsqr(ca.vertcat(v, w)))
    opti.set_initial(dt, max(0.05, float(traj["t"][-1]) / (n - 1)))
    opti.set_initial(x, ref[:, 0])
    opti.set_initial(y, ref[:, 1])
    opti.set_initial(th, ref[:, 2])
    # Finite-difference controls as the guess.
    dts = np.diff(ref[:, 3]) if ref.shape[1] > 3 else None
    v_guess = np.zeros(n - 1)
    w_guess = np.zeros(n - 1)
    for k in range(n - 1):
        dtk = float(ref[k + 1, 3] - ref[k, 3]) if dts is not None else 0.1
        dtk = max(dtk, 1e-3)
        c, s = math.cos(float(ref[k, 2])), math.sin(float(ref[k, 2]))
        v_guess[k] = ((ref[k + 1, 0] - ref[k, 0]) * c + (ref[k + 1, 1] - ref[k, 1]) * s) / dtk
        w_guess[k] = (ref[k + 1, 2] - ref[k, 2]) / dtk
    opti.set_initial(v, v_guess)
    opti.set_initial(w, w_guess)
    opti.solver(
        "ipopt",
        {"print_time": False},
        {
            "print_level": 0,
            "sb": "yes",
            "max_iter": 60,
            "max_cpu_time": float(timeout_s),
            "tol": 1e-3,
            "acceptable_tol": 5e-3,
            "acceptable_iter": 4,
        },
    )
    try:
        sol = opti.solve()
    except RuntimeError:
        return None
    xs = np.array(sol.value(x)).reshape(-1)
    ys = np.array(sol.value(y)).reshape(-1)
    ths = np.array(sol.value(th)).reshape(-1)
    vs = np.array(sol.value(v)).reshape(-1)
    ws = np.array(sol.value(w)).reshape(-1)
    dtk = float(sol.value(dt))
    samples = []
    t = 0.0
    for k in range(n - 1):
        samples.append((t, float(xs[k]), float(ys[k]), float(ths[k]), float(vs[k]), float(ws[k])))
        t += dtk
    samples.append((t, float(xs[-1]), float(ys[-1]), float(ths[-1]), 0.0, 0.0))
    return _pack(samples, "ipopt")


def _resample_traj(traj: dict, n: int) -> np.ndarray | None:
    t = np.asarray(traj["t"], dtype=float)
    if t[-1] < 1e-6 or len(t) < 2:
        return None
    grid = np.linspace(0.0, float(t[-1]), n)
    cols = []
    for key in ("x", "y", "theta"):
        cols.append(np.interp(grid, t, np.asarray(traj[key], dtype=float)))
    cols.append(grid)
    return np.stack(cols, axis=1)


def goal_reached(traj: dict, xy: tuple[float, float], tol: float, theta: float | None = None, theta_tol: float = 0.5) -> bool:
    if not traj["x"]:
        return False
    if math.hypot(traj["x"][-1] - xy[0], traj["y"][-1] - xy[1]) > tol:
        return False
    if theta is not None and abs(wrap_angle(traj["theta"][-1] - theta)) > theta_tol:
        return False
    return True
