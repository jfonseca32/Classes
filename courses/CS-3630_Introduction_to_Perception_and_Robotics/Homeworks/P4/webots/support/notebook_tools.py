"""Helpers for ``project4.ipynb``. You do not need to read or change this file.

    check("rfid_likelihood")  run the public tests of one function and print a short report
    check_all()             run every public test and print one line per TODO
    run_mission(3630)       play a whole mission in the terminal engine (the same one Webots uses)
    show.<picture>(p4)      draw a picture with YOUR functions (needs matplotlib)

The tests run in a separate Python process, exactly as Gradescope runs the
public tests, so the result never depends on what the notebook did before.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent.parent
TEST_FILE = TESTS_DIR / "test_project4.py"
RUNNER = TESTS_DIR / "run_mission.py"

# (TODO number, function) in the order of project4.py; 0 is the setup row.
TODOS = (
    (0, "student_info"),
    (1, "sample_motion_model"),
    (2, "rfid_likelihood"),
    (3, "zone_likelihood"),
    (4, "mcl_predict"),
    (5, "mcl_update"),
    (6, "estimate_position"),
    (7, "q_value"),
    (8, "value_iteration"),
)
_NAMES = [name for _, name in TODOS] + ["robot_profile"]
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_RESULT = re.compile(r"^(PASSED|FAILED|ERROR)\s+\S*::(test_\w+)(?:\[[^\]]*\])?(?:\s+-\s+(.*))?$")


def _pytest(args, timeout=600):
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--no-header",
        "--color=no",
        str(TEST_FILE),
    ]
    env = dict(os.environ, COLUMNS="400")  # pytest would cut failure messages at 80 columns
    try:
        done = subprocess.run(
            command + list(args),
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return (
            None,
            f"The tests did not finish within {timeout} s. Is one of your functions stuck "
            f"in a loop?",
        )
    return done.returncode, done.stdout + done.stderr


def _function_of(test_name):
    """The project4.py function a test belongs to (tests are named test_<function>_...)."""
    rest = test_name[len("test_") :]
    for name in sorted(_NAMES, key=len, reverse=True):
        if rest == name or rest.startswith(name + "_"):
            return name
    return None


def _results(output):
    rows = []
    for line in _ANSI.sub("", output).splitlines():
        match = _RESULT.match(line.strip())
        if match:
            rows.append((match.group(2), match.group(1), match.group(3) or ""))
    return rows


def _import_problem(output):
    """The lines of pytest's report that explain why project4.py did not import."""
    lines = [line for line in output.splitlines() if line.startswith("E ") or "Error" in line]
    return "\n".join(lines[-8:]) if lines else output.strip()[-1500:]


def check(name: str, verbose: bool = False) -> None:
    """Run the public tests of one function, for example ``check("rfid_likelihood")``.

    Prints one line when everything passes, or the failing tests with their
    messages. ``verbose=True`` also prints pytest's full report (tracebacks).
    """
    if name not in _NAMES:
        print(f"Unknown function {name!r}. Use one of: {', '.join(_NAMES)}.")
        return
    code, output = _pytest(["-rfE", "--tb=short" if verbose else "--tb=no", "-k", f"test_{name}"])
    if code is None:
        print(output)
        return
    rows = [row for row in _results(output) if _function_of(row[0]) == name]
    passed = re.search(r"(\d+) passed", output)
    failed = [row for row in rows if row[1] != "PASSED"]
    n_passed = int(passed.group(1)) if passed else 0
    total = n_passed + len(failed)
    if code == 5 or total == 0:
        if "error" in output.lower():
            print(
                "The tests could not even start: project4.py does not import.\n"
                + _import_problem(output)
            )
        else:
            print(f"No public tests found for {name}.")
        return
    if not failed:
        tests = "the public test passes" if total == 1 else f"all {total} public tests pass"
        print(f"PASS  {name}: {tests}.")
    elif all("NotImplementedError" in message for _, _, message in failed):
        todo = dict((n, t) for t, n in TODOS).get(name)
        where = f"TODO {todo}, " if todo else ""
        print(f"TODO  {name}: not implemented yet ({where}{len(failed)} public tests waiting).")
    else:
        print(f"FAIL  {name}: {len(failed)} of {total} public tests fail.")
        for test, _, message in failed:
            print(f"  - {test}")
            if message:
                print(f"      {message[:400]}")
        if not verbose:
            print(f"  Run check({name!r}, verbose=True) for the full tracebacks.")
    if verbose:
        print("\n" + output)


def check_all() -> None:
    """Run every public test once and print a one-line status per TODO."""
    code, output = _pytest(["-rA", "--tb=no"])
    if code is None:
        print(output)
        return
    rows = _results(output)
    if not rows:
        print(
            "The tests could not even start: project4.py does not import.\n"
            + _import_problem(output)
        )
        return
    stats = {
        name: [0, 0, True] for name in _NAMES
    }  # passed, total, all failures are NotImplementedError
    for test, outcome, message in rows:
        name = _function_of(test)
        if name is None:
            continue
        stats[name][1] += 1
        if outcome == "PASSED":
            stats[name][0] += 1
        elif "NotImplementedError" not in message:
            stats[name][2] = False
    print(f"{'TODO':>4}  {'function':<22} {'public tests':>12}  status")
    done = 0
    for todo, name in TODOS:
        passed, total, untouched = stats[name]
        if total == 0:
            continue
        if passed == total:
            status, done = "PASS", done + 1
        elif passed == 0 and untouched:
            status = "not started"
        else:
            status = "FAIL"
        label = "0" if todo == 0 else str(todo)
        print(f"{label:>4}  {name:<22} {passed:>5} / {total:<5}  {status}")
    print(f"\n{done} of {len(TODOS)} rows pass all their public tests.")
    if done == len(TODOS):
        print("Every public test passes. Next: the Webots experiments.")


def run_mission(*codes: int, quiet: bool = True, chaos: bool = False, extra=()) -> None:
    """Play whole missions in text mode with your project4.py, for example ``run_mission(3630,
    2026)``.

    ``quiet=False`` prints the step log as well, ``chaos=True`` kidnaps the robot
    and drops an obstacle mid-mission.
    """
    command = [sys.executable, str(RUNNER)]
    if codes:
        command += ["--seeds"] + [str(int(code)) for code in codes]
    if quiet:
        command.append("--quiet")
    if chaos:
        command.append("--chaos")
    command += list(extra)
    process = subprocess.Popen(
        command, cwd=PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()


###############################################################################
#                       Pictures drawn with YOUR functions                    #
###############################################################################

EXTENT = (-0.5, 99.5, -0.5, 49.5)  # the 100 m x 50 m floor, in meters


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "These pictures need matplotlib: run  python -m pip install matplotlib  and "
            "restart the kernel."
        )
        return None
    return plt


def _guard(draw):
    """Turn NotImplementedError (and other errors) into a one-line note instead of a traceback."""

    def wrapper(p4, *args, **kwargs):
        plt = _pyplot()
        if plt is None:
            return None
        try:
            return draw(plt, p4, *args, **kwargs)
        except NotImplementedError as error:
            plt.close("all")
            print(f"Implement {error} first to see this picture.")
        except Exception as error:  # a student bug: show it briefly, the tests give the details
            plt.close("all")
            print(f"Could not draw the picture: {type(error).__name__}: {error}")
        return None

    wrapper.__doc__ = draw.__doc__
    wrapper.__name__ = draw.__name__
    return wrapper


def _floor(ax, p4, title, beacons=True):
    """Shelves in gray, beacons in cyan, meters on the axes."""
    shelves = np.ma.masked_where(np.asarray(p4.BASE_MAP) == 0, p4.BASE_MAP)
    ax.imshow(
        shelves,
        origin="lower",
        extent=EXTENT,
        cmap="Greys",
        vmin=0,
        vmax=1.6,
        alpha=0.9,
        interpolation="nearest",
    )
    if beacons:
        ax.scatter(
            p4.BEACONS[:, 0],
            p4.BEACONS[:, 1],
            s=18,
            c="#00b8d9",
            edgecolors="k",
            linewidths=0.5,
            zorder=5,
        )
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def _heat(ax, grid, cmap, alpha=1.0, vmin=None, vmax=None):
    return ax.imshow(
        grid,
        origin="lower",
        extent=EXTENT,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        interpolation="nearest",
    )


def _ellipse(ax, mean, cov, color):
    """The 2-sigma ellipse of a 2x2 covariance."""
    values, vectors = np.linalg.eigh(np.asarray(cov, float))
    values = np.clip(values, 0.0, None)
    angles = np.linspace(0, 2 * np.pi, 120)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    points = (vectors @ (2.0 * np.sqrt(values)[:, None] * circle)).T + np.asarray(mean, float)
    ax.plot(points[:, 0], points[:, 1], color=color, lw=1.5)


@_guard
def motion(plt, p4, seed=7):
    """300 robots driven by a control tape through your sample_motion_model."""
    rng = np.random.default_rng(seed)
    states = np.array([20.0, 10.0]) + rng.normal(0.0, 0.5, size=(300, 2))
    tape = [(2, 0), (2, 0), (0, 2), (0, 2)]
    clouds, nominal = [states], [np.array([20.0, 10.0])]
    for u in tape:
        states = np.asarray(p4.sample_motion_model(states, np.array(u, float), 0.2, rng))
        clouds.append(states)
        nominal.append(nominal[-1] + np.array(u, float))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(clouds)))
    for k, (cloud, color) in enumerate(zip(clouds, colors, strict=False)):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=4, color=color, alpha=0.5, label=f"x{k + 1}")
    nominal = np.array(nominal)
    ax.plot(nominal[:, 0], nominal[:, 1], "k--", lw=1, label="noise-free path")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(
        "Your sample_motion_model: 300 robots, control tape (2,0),(2,0),(0,2),(0,2)", fontsize=9
    )
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    plt.show()


@_guard
def rfid(plt, p4):
    """RFID likelihoods: a 4 m reading of beacon 0, and 'no beacon in range'."""
    points = p4.MAP_COORDS.reshape(-1, 2)
    ring = np.asarray(p4.rfid_likelihood(points, (0, 4.03))).reshape(50, 100)
    none = np.asarray(p4.rfid_likelihood(points, (None, float("inf")))).reshape(50, 100)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8.2))
    _heat(axes[0], ring, "Oranges", vmin=0, vmax=1)
    _floor(axes[0], p4, "Your rfid_likelihood for the reading (beacon 0, 4.03 m): an annulus")
    _heat(axes[1], none, "Oranges", vmin=0, vmax=1)
    _floor(axes[1], p4, "Your rfid_likelihood for (None, inf): holes around all 8 beacons")
    plt.tight_layout()
    plt.show()


def _zone_of(p4, x):
    i, j = p4.point_to_cell(x)
    return int(p4.ZONE_MAP[i, j]) if p4.in_bounds((i, j)) else -1


def _zone_reading(p4, x, rng):
    """A simulated zone reading at the true position ``x`` (like the Webots warehouse): the robot's
    zone, but within 1 m of another zone (across open floor) that zone with probability 0.3."""
    x = np.asarray(x, float)
    u = rng.random(2)
    zone = _zone_of(p4, x)
    near = []
    for offset in ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)):
        other = _zone_of(p4, x + offset)
        if other >= 0 and other != zone and other not in near:
            near.append(other)
    if near and u[0] < 0.3:
        return near[min(int(u[1] * len(near)), len(near) - 1)]
    return zone


@_guard
def mcl(plt, p4, seed=3630):
    """Global localization: 2000 uniform particles weighted by one round of readings."""
    rng = np.random.default_rng(seed)
    truth = np.array([31.6, 13.2])
    particles = rng.uniform([-0.5, -0.5], [99.5, 49.5], size=(2000, 2))
    i, j = p4.point_to_cell(truth)
    z_on = bool(p4.PROXIMITY_MAP_ON[i, j])
    distances = np.linalg.norm(p4.BEACONS - truth, axis=1)
    nearest = int(np.argmin(distances))
    if distances[nearest] <= p4.RFID_MAX_RANGE:
        z_rfid = (nearest, float(distances[nearest] + rng.normal(0.0, 0.3)))
    else:
        z_rfid = (None, float("inf"))
    z_zone = _zone_reading(p4, truth, rng)
    weights = np.asarray(p4.mcl_update(particles, z_on, z_rfid, z_zone), float)
    if not weights.sum() > 0:
        print("All particle weights are zero: check mcl_update and the three likelihoods.")
        return
    mean, cov = p4.estimate_position(particles, weights)
    w = weights / weights.max()
    fig, axes = plt.subplots(2, 1, figsize=(8, 8.2))
    axes[0].scatter(particles[:, 0], particles[:, 1], s=2, c="#d1495b", alpha=0.6)
    _floor(axes[0], p4, "Before: 2000 particles from a uniform prior")
    keep = w > 1e-3
    axes[1].scatter(
        particles[keep, 0], particles[keep, 1], s=2 + 30 * w[keep], c="#d1495b", alpha=0.55
    )
    axes[1].plot(*truth, marker="*", color="k", ms=12, label="true robot")
    axes[1].plot(*mean, marker="x", color="#c218c4", ms=10, mew=2, label="your estimate")
    _ellipse(axes[1], mean, cov, "#c218c4")
    axes[1].legend(fontsize=7, loc="upper right")
    reading = (
        f"RFID {z_rfid[0]} @ {z_rfid[1]:.1f} m" if z_rfid[0] is not None else "no beacon in range"
    )
    _floor(
        axes[1],
        p4,
        f"After one update (proximity {'ON' if z_on else 'OFF'}, {reading}, "
        f"zone {p4.ZONE_NAMES[z_zone]}): "
        "your weights, mean, and 2-sigma ellipse",
    )
    plt.tight_layout()
    plt.show()


@_guard
def value(plt, p4, start=(5, 5)):
    """Your value function for the goal (35, 50) and the greedy path from a start cell."""
    goal = np.zeros((50, 100), dtype=bool)
    goal[35, 50] = True
    V = np.asarray(p4.value_iteration(goal), float)
    path, cell = [start], tuple(start)
    for _ in range(300):
        if goal[cell]:
            break
        cell = p4.next_cell(cell, int(p4.greedy_action(V, cell, goal)))
        path.append(cell)
    free = np.ma.masked_where(np.asarray(p4.BASE_MAP) != 0, V)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    image = _heat(ax, free, "turbo", vmin=np.percentile(V, 2), vmax=V.max())
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cells = np.array(path)
    ax.plot(cells[:, 1], cells[:, 0], color="white", lw=2.2)
    ax.plot(cells[:, 1], cells[:, 0], color="k", lw=0.8, ls="--")
    ax.plot(50, 35, marker="*", color="white", ms=14, mec="k")
    _floor(
        ax, p4, "Your value_iteration and the greedy_action path from cell (5, 5)", beacons=False
    )
    plt.show()


@_guard
def sensors(plt, p4, truth=(29.6, 15.0), seed=3630):
    """One reading of each sensor at the same true position: what each one allows, and the
    product.
    """
    rng = np.random.default_rng(seed)
    truth = np.asarray(truth, float)
    points = p4.MAP_COORDS.reshape(-1, 2)
    i, j = p4.point_to_cell(truth)
    z_on = bool(p4.PROXIMITY_MAP_ON[i, j])
    distances = np.linalg.norm(p4.BEACONS - truth, axis=1)
    nearest = int(np.argmin(distances))
    z_rfid = (
        (nearest, float(distances[nearest] + rng.normal(0.0, 0.3)))
        if distances[nearest] <= p4.RFID_MAX_RANGE
        else (None, float("inf"))
    )
    z_zone = _zone_reading(p4, truth, rng)
    maps = [
        (
            "provided proximity_likelihood: " + ("ON" if z_on else "OFF"),
            np.asarray(p4.proximity_likelihood(points, z_on), float),
        ),
        (
            f"your rfid_likelihood: beacon {z_rfid[0]} at {z_rfid[1]:.1f} m"
            if z_rfid[0] is not None
            else "your rfid_likelihood: no beacon in range",
            np.asarray(p4.rfid_likelihood(points, z_rfid), float),
        ),
        (
            f"your zone_likelihood: the reader says {p4.ZONE_NAMES[z_zone]}",
            np.asarray(p4.zone_likelihood(points, z_zone), float),
        ),
    ]
    product = maps[0][1] * maps[1][1] * maps[2][1]
    maps.append(("all three multiplied (what mcl_update does)", product))
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6))
    for ax, (title, values) in zip(axes.flat, maps, strict=False):
        grid = values.reshape(50, 100)
        _heat(ax, grid / grid.max() if grid.max() > 0 else grid, "Oranges", vmin=0, vmax=1)
        ax.plot(*truth, marker="*", color="k", ms=11)
        _floor(ax, p4, title)
    plt.tight_layout(h_pad=2.5)
    plt.show()


def _bellman(p4, V, goal, gamma):
    """One optimal backup of V (the same reward model as q_value): max over the 4 actions."""
    obstacles = np.asarray(p4.BASE_MAP) != 0
    proximity = np.asarray(p4.PROXIMITY_MAP_ON) != 0
    h, w = V.shape
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    q = []
    for di, dj in p4.ACTION_DELTAS:
        r, c = np.clip(rows + di, 0, h - 1), np.clip(cols + dj, 0, w - 1)
        reward = (
            np.where(goal[r, c], p4.GOAL_REWARD, 0.0)
            - p4.PROXIMITY_PENALTY * proximity
            - p4.PROXIMITY_PENALTY * proximity[r, c]
        )
        q.append(reward + gamma * V[r, c])
    new = np.max(q, axis=0)
    new[goal | obstacles] = 0.0
    return new


def _route(p4, V, goal, start, gamma, limit=300):
    """Follow the provided greedy_action from ``start`` until the goal (or a loop)."""
    path, cell, seen = [tuple(start)], tuple(start), {tuple(start)}
    for _ in range(limit):
        if goal[cell]:
            break
        cell = p4.next_cell(
            cell, int(p4.greedy_action(V, cell, goal, p4.BASE_MAP, p4.PROXIMITY_MAP_ON, gamma))
        )
        if cell in seen:
            break
        path.append(cell)
        seen.add(cell)
    return path


def _value_axes(ax, p4, V, goal_cell, title, vmin, vmax):
    free = np.ma.masked_where(np.asarray(p4.BASE_MAP) != 0, V)
    image = _heat(ax, free, "turbo", vmin=vmin, vmax=vmax)
    ax.plot(goal_cell[1], goal_cell[0], marker="*", color="white", ms=12, mec="k")
    _floor(ax, p4, title, beacons=False)
    return image


@_guard
def planning(plt, p4, experiment="convergence", goal_cell=(35, 50), start=(5, 5)):
    """Value iteration experiments with YOUR value_iteration (and the provided greedy_action).

    experiment="convergence": the value map after 1, 10, and 40 sweeps and at convergence, and the
    Bellman residual of the converged map (the largest change from one more optimal backup).
    Returns the converged value map.

    experiment="discount": gamma = 0.90 vs 0.975: both value maps and the greedy routes from
    ``start``. Returns {gamma: {"V", "route", "length", "prox_cells", "reached"}}.
    """
    goal = np.zeros((50, 100), dtype=bool)
    goal[goal_cell] = True
    if experiment == "convergence":
        V = np.asarray(p4.value_iteration(goal), float)
        maps = [
            (
                f"after {k} sweep{'s' if k > 1 else ''}",
                np.asarray(p4.value_iteration(goal, max_iterations=k), float),
            )
            for k in (1, 10, 40)
        ] + [("at convergence", V)]
        vmin, vmax = np.percentile(V, 2), V.max()
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.6))
        for ax, (title, values) in zip(axes.flat, maps, strict=False):
            image = _value_axes(
                ax, p4, values, goal_cell, f"Your value_iteration: {title}", vmin, vmax
            )
        fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
        plt.show()
        residual = float(np.max(np.abs(_bellman(p4, V, goal, p4.GAMMA) - V)))
        print(
            f"Bellman residual of the converged map (largest change from one more optimal "
            f"backup): {residual:.2e}"
        )
        return V
    if experiment == "discount":
        results = {}
        fig, axes = plt.subplots(2, 1, figsize=(8, 8.4))
        for ax, gamma in zip(axes, (0.90, 0.975), strict=False):
            V = np.asarray(p4.value_iteration(goal, gamma=gamma), float)
            path = _route(p4, V, goal, start, gamma)
            prox = sum(int(p4.PROXIMITY_MAP_ON[c] > 0) for c in path)
            reached = bool(goal[path[-1]])
            results[gamma] = {
                "V": V,
                "route": path,
                "length": len(path) - 1,
                "prox_cells": prox,
                "reached": reached,
            }
            image = _value_axes(
                ax,
                p4,
                V,
                goal_cell,
                f"gamma = {gamma}: route {len(path) - 1} m, {prox} route cells "
                f"with proximity ON" + ("" if reached else ", goal NOT reached"),
                np.percentile(V, 2),
                V.max(),
            )
            cells = np.array(path)
            ax.plot(cells[:, 1], cells[:, 0], color="white", lw=2.2)
            ax.plot(cells[:, 1], cells[:, 0], color="k", lw=0.8, ls="--")
            fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        plt.tight_layout()
        plt.show()
        same = results[0.90]["route"] == results[0.975]["route"]
        for gamma, r in results.items():
            print(
                f"gamma {gamma}: V at the start {r['V'][tuple(start)]:8.3f}, route "
                f"{r['length']} m, "
                f"{r['prox_cells']} cells with proximity ON, goal "
                f"{'reached' if r['reached'] else 'NOT reached'}"
            )
        print("The two routes are " + ("the same." if same else "different."))
        return results
    raise ValueError('experiment must be "convergence" or "discount"')


def _recorded_drive(p4, seed):
    """A recorded drive: true positions (motion noise 0.2 m), the controls, and the readings."""
    rng = np.random.default_rng(seed)
    controls = np.array(
        [(1.0, 0.0)] * 70
    )  # east along the open floor below the shelves, past the 4 beacons
    truth = [np.array([12.0, 5.0])]
    for u in controls:
        truth.append(truth[-1] + u + rng.normal(0.0, 0.2, 2))
    truth = np.array(truth)
    readings = []
    for x in truth[1:]:
        i, j = p4.point_to_cell(x)
        z_on = bool(p4.PROXIMITY_MAP_ON[min(max(i, 0), 49), min(max(j, 0), 99)])
        d = np.linalg.norm(p4.BEACONS - x, axis=1)
        k = int(np.argmin(d))
        z_rfid = (
            (k, float(d[k] + rng.normal(0.0, 0.3)))
            if d[k] <= p4.RFID_MAX_RANGE
            else (None, float("inf"))
        )
        readings.append((z_on, z_rfid, _zone_reading(p4, x, rng)))
    return {"truth": truth, "controls": controls, "readings": readings}


@_guard
def localization(plt, p4, experiment="noise", sigmas=(0.2, 1.0), seed=3630, n=1000):
    """experiment="noise": your particle filter on ONE recorded drive (true motion noise 0.2 m), run
    twice with an assumed motion noise of 0.2 m and of 1.0 m: the estimate error and the belief
    spread (largest standard deviation) at every step. Returns {sigma: {"error", "spread"}}."""
    if experiment != "noise":
        raise ValueError('experiment must be "noise"')
    data = _recorded_drive(p4, seed)
    results = {}
    for sigma in sigmas:
        rng = np.random.default_rng(seed + 1)  # the same random stream for both runs
        particles = data["truth"][0] + rng.normal(0.0, 0.5, size=(n, 2))
        weights = np.full(n, 1.0 / n)
        errors, spreads = [], []
        for u, (z_on, z_rfid, z_zone), x in zip(
            data["controls"], data["readings"], data["truth"][1:], strict=False
        ):
            particles = np.asarray(p4.mcl_predict(particles, weights, u, sigma, rng), float)
            raw = np.asarray(p4.mcl_update(particles, z_on, z_rfid, z_zone), float)
            weights = (
                raw / raw.sum() if raw.sum() > 0 else np.full(len(particles), 1.0 / len(particles))
            )
            mean, cov = p4.estimate_position(particles, weights)
            errors.append(float(np.linalg.norm(np.asarray(mean, float) - x)))
            spreads.append(
                float(np.sqrt(max(np.max(np.linalg.eigvalsh(np.asarray(cov, float))), 0.0)))
            )
        results[sigma] = {"error": np.array(errors), "spread": np.array(spreads)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for sigma, color in zip(sigmas, ("#1f77b4", "#d62728"), strict=False):
        axes[0].plot(results[sigma]["error"], color=color, label=f"assumed sigma = {sigma} m")
        axes[1].plot(results[sigma]["spread"], color=color, label=f"assumed sigma = {sigma} m")
    for ax, title in zip(
        axes, ("Estimate error |mean - truth| (m)", "Belief spread: largest std (m)"), strict=False
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    for sigma in sigmas:
        print(
            f"assumed sigma {sigma} m: mean error {results[sigma]['error'].mean():.2f} m, "
            f"mean spread {results[sigma]['spread'].mean():.2f} m"
        )
    return results


class _Show:
    """``show.motion(p4)``, ``show.rfid(p4)``, ``show.sensors(p4)``, ``show.mcl(p4)``,
    ``show.value(p4)``,
    ``show.planning(p4, experiment=...)``, ``show.localization(p4, experiment="noise")``.
    """

    motion = staticmethod(motion)
    rfid = staticmethod(rfid)
    sensors = staticmethod(sensors)
    mcl = staticmethod(mcl)
    value = staticmethod(value)
    planning = staticmethod(planning)
    localization = staticmethod(localization)


show = _Show()
