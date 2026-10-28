"""Run the Project 4 warehouse mission in the terminal (no Webots needed).

This runs exactly the same mission engine as the Webots world, with your
``project4.py`` as the robot's brain, and prints what happens: the boot
system check of every TODO, the system identification, a step log with the
robot's true position next to its estimate, the events (LOCKED, SAFETY STOP,
MARKER ACQUIRED, ...), and a final table with the score.

Run it from the project folder (the one that contains ``project4.py``), or
with ``run_mission(...)`` in ``project4.ipynb``. Examples::

    python webots/support/run_mission.py                  # one mission, random code
    python webots/support/run_mission.py --seeds 1 2 3    # three mission codes
    python webots/support/run_mission.py --chaos          # drop an obstacle and kidnap the robot
    mid-mission
    python webots/support/run_mission.py --scenario start     # both start tests and their table
    python webots/support/run_mission.py --scenario sensors   # all five sensor tests (or penalty:
    both)
    python webots/support/run_mission.py --scenario sensors_rfid_only   # one test (as E and 1-N at
    the Webots briefing)
    python webots/support/run_mission.py --ascii 20       # ASCII map of the warehouse every 20
    steps
    python webots/support/run_mission.py --quiet          # system check and final table only

The same mission code gives the same warehouse, start, order, and sensor
noise as in Webots, so you can replay a Webots run here (and vice versa).
Students do not need to modify this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent  # the folder that contains project4.py and webots/
ENGINE_DIR = PROJECT_DIR / "webots" / "controllers" / "warehouse_mission"

# Events worth a line in the step log (RFID pings and the like are too chatty).
LOGGED_EVENTS = {
    "LOCKED",
    "LOST",
    "RESEED",
    "RECOVERED",
    "SAFETY_STOP",
    "WAITING",
    "FORKLIFT_YIELD",
    "MARKER_ACQUIRED",
    "DOCKING_FAILED",
    "PACKAGE_LOADED",
    "DELIVERED",
    "NEW_ORDER",
    "OBSTACLE_DROPPED",
    "REPLANNED",
    "KIDNAPPED",
    "FORKLIFTS",
    "GRID_FILTER_OFF",
    "WARNING",
    "SLOW_CODE",
    "MISSION_COMPLETE",
    "MISSION_FAILED",
    "FAULT",
    "SENSORS",
    "EXPERIMENT_COMPLETE",
    "SEEK_BEACON",
}
GROUPS = {  # an experiment group runs all of its tests
    "start": ("start_known", "start_unknown"),
    "sensors": (
        "sensors_all",
        "sensors_zone_only",
        "sensors_rfid_only",
        "sensors_proximity_only",
        "sensors_none",
    ),
    "penalty": ("penalty_50", "penalty_0"),
}
SCENARIOS = (
    ("mission", "obstacle_test")
    + tuple(GROUPS)
    + tuple(key for keys in GROUPS.values() for key in keys)
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Project 4 warehouse mission offline with your project4.py."
    )
    parser.add_argument(
        "--student-code",
        type=Path,
        default=PROJECT_DIR / "project4.py",
        metavar="PATH",
        help="path to your project4.py (default: the one in the project folder)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="CODE",
        help="mission codes to run, 0-9999 (default: one random code)",
    )
    parser.add_argument(
        "--chaos",
        action="store_true",
        help="drop an obstacle on the robot's path and kidnap the robot during each mission",
    )
    parser.add_argument(
        "--scenario",
        default="mission",
        choices=SCENARIOS,
        help="an experiment instead of the mission (as E at the Webots briefing): a whole "
        "group (start, sensors, penalty), one of its tests (e.g. sensors_rfid_only), "
        "or obstacle_test",
    )
    parser.add_argument("--forklifts", action="store_true", help="turn on forklift traffic")
    parser.add_argument(
        "--quiet", action="store_true", help="print only the system check and the final table"
    )
    parser.add_argument("--verbose", action="store_true", help="print one line per control step")
    parser.add_argument(
        "--ascii",
        type=int,
        default=0,
        metavar="N",
        help="print an ASCII map of the warehouse every N control steps",
    )
    parser.add_argument(
        "--no-zone",
        action="store_true",
        help="hard mode: no zone reader, only the book's proximity and RFID sensors",
    )
    return parser.parse_args(argv)


def load_engine():
    """Import ``wm.core`` from the Webots controller folder next to this script."""
    if not (ENGINE_DIR / "wm" / "core").is_dir():
        sys.exit(
            f"Cannot find the mission engine in {ENGINE_DIR}.\n"
            "Keep the project folder intact: webots/support/ and webots/ must stay next to "
            "project4.py."
        )
    if str(ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_DIR))
    import wm.core as core  # noqa: E402  (the engine lives inside the Webots controller folder)

    return core


def fmt_vec(values, digits: int = 1) -> str:
    return "(" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")"


def fmt_matrix(matrix) -> str:
    return (
        "["
        + ", ".join("[" + ", ".join(f"{float(v):6.3f}" for v in row) + "]" for row in matrix)
        + "]"
    )


def print_system_check(core, bridge) -> None:
    rows = bridge.system_check()
    print("SYSTEM CHECK (a quick probe of every TODO; MISSING = still raises NotImplementedError)")
    print(f"  {'#':>2}  {'function':<22} {'status':<8} note")
    for todo, name, status, message in rows:
        flag = "  <- needed to deploy" if status == core.MISSING and todo in core.CORE_TODOS else ""
        print(f"  {todo:>2}  {name:<22} {status:<8} {message}{flag}")
    ok = sum(1 for row in rows if row.status == core.OK)
    errors = [row for row in rows if row.status == core.ERROR]
    print(f"  {ok}/{len(rows)} TODOs OK.", end="")
    if errors:
        print(
            f" {len(errors)} ERROR(s): the probe got a wrong value; the note shows what it "
            f"expected.",
            end="",
        )
    print()
    print()


def describe_event(event) -> str:
    data = event.data
    if event.name == "LOCKED":
        how = " by the dock marker" if data.get("marker") else ""
        return f"{event.text}  confidence {100 * data.get('confidence', 0):.0f}%{how}"
    if event.name in ("LOST", "RESEED"):
        return f"{event.text}  ({data.get('reason', '')})"
    if event.name == "SAFETY_STOP":
        return f"{event.text}  after {100 * data.get('fraction', 0):.0f}% of the step"
    if event.name == "MARKER_ACQUIRED":
        return (
            f"{event.text}  arrived {data['distance']:.2f} m from the station; "
            f"estimate error {data['error_before']:.2f} m -> {data['error_after']:.2f} m"
        )
    if event.name == "DOCKING_FAILED":
        return (
            f"{event.text}  the robot was really {data['distance']:.2f} m from the station "
            f"(the marker is visible within {data['capture_radius']:.1f} m); re-localizing"
        )
    if event.name == "OBSTACLE_DROPPED":
        return f"{event.text}  at {fmt_vec((data['x'], data['y']))}"
    if event.name == "REPLANNED":
        text = f"{event.text}  (your value_iteration took {data.get('seconds', 0):.2f} s)"
        if data.get("changed_cells") is not None:
            text += (
                f"; the value changed in {data['changed_cells']} cells, "
                f"dropping by up to {data['value_drop']:.1f}"
            )
        return text
    if event.name == "KIDNAPPED":
        return (
            f"{event.text}  {fmt_vec(data['from'])} -> {fmt_vec(data['to'])} (the brain "
            f"was not told)"
        )
    if event.name in ("WARNING", "GRID_FILTER_OFF"):
        return f"{event.text}: {data.get('message', data.get('reason', ''))}"
    return event.text


def run_one(core, bridge, code: int, args, scenario: str) -> dict:
    options = core.MissionOptions(grid_filter=False, likelihood_layer=False, zone=not args.no_zone)
    mission = core.Mission(bridge, code=code, orders=1, options=options, scenario=scenario)
    loud = not args.quiet
    if loud:
        count = len(mission.orders)
        orders = "; ".join(
            f"pick up {o.slot.name}, deliver to {o.dock.name}" for o in mission.orders
        )
        if mission.scenario.group:
            orders = f"{mission.scenario.title}: {mission.scenario.option} (the same every run)"
        print("=" * 78)
        print(
            f"MISSION {mission.code:04d}   "
            + (
                orders
                if mission.scenario.is_tour
                else f"{count} order{'s' if count > 1 else ''}: {orders}"
            )
        )
        print(
            f"(replay this mission with --seeds {mission.code}; in Webots use --seed "
            f"{mission.code})"
        )
        print("-" * 78)

    def log(result, text: str) -> None:
        if loud:
            print(f"  [{result.clock:6.1f} s] {text}")

    def on_result(result) -> None:
        for event in result.events:
            if event.name == "SYSID" and loud:
                data = event.data
                print(f"  System ID from {data['samples']} calibration moves:")
                if data["A"] is not None and data["B"] is not None:
                    print(f"      A_hat = {fmt_matrix(data['A'])}")
                    print(f"      B_hat = {fmt_matrix(data['B'])}")
                print(
                    f"      {data['message']} -> the particle filter uses sigma = "
                    f"{data['sigma']:.3f} m"
                )
            elif event.name == "PHASE" and event.data.get("to") not in (
                core.BOOT,
                core.BRIEFING,
                core.COUNTDOWN,
            ):
                log(result, f"=== {event.text} ===")
            elif event.name == "DEPLOYED":
                log(
                    result,
                    event.text
                    if event.data.get("known_start")
                    else f"{event.text} spread over the warehouse (global localization)",
                )
            elif event.name == "WARNING" and event.data.get("code") == "SIGMA_FALLBACK":
                continue  # the system ID lines above already say so
            elif event.name in LOGGED_EVENTS:
                log(result, describe_event(event))
                details = event.data.get("details") if event.name == "FAULT" else None
                if details and loud:
                    print("      " + "\n      ".join(details.rstrip().splitlines()[-8:]))
        if result.kind not in ("move", "wait") or result.belief is None or not loud:
            return
        steps = result.stats["control_steps"]
        if args.verbose or steps % 20 == 0:
            belief = result.belief
            error = float(((belief.mean - result.end) ** 2).sum() ** 0.5)
            action = "-" if result.action is None else core.config.ACTION_NAMES[result.action]
            state = "LOCKED" if belief.locked else ("LOST" if belief.relocalizing else "      ")
            print(
                f"  [{result.clock:6.1f} s] step {steps:3d} {action:<5}  truth "
                f"{fmt_vec(result.end)}  "
                f"estimate {fmt_vec(belief.mean)} +/- {belief.sigma:.1f} m  error {error:4.1f} m  "
                f"confidence {100 * belief.confidence:3.0f}% {state}"
            )
        if args.ascii and steps % args.ascii == 0 and result.kind == "move":
            from wm.core.ascii_map import render

            print(render(result))

    summary = core.run_to_completion(
        mission, chaos=args.chaos, forklifts=args.forklifts, on_result=on_result
    )
    if loud:
        score = mission.score()
        if mission.success and mission.experiment is not None:
            print_experiment(core, mission.experiment)
        elif mission.success:
            print(
                f"  MISSION COMPLETE in {mission.stats.mission_time:.1f} s: {score['score']} "
                f"points, "
                f"{'*' * score['stars']} {score['title']}"
            )
            terms = score["score_terms"]
            print("  score = " + "  ".join(f"{name} {value:+.0f}" for name, value in terms.items()))
        elif mission.phase == core.FAILED:
            print(
                f"  MISSION FAILED: out of time after {mission.stats.control_steps} control steps "
                f"({mission.options.max_steps_per_order} per order)."
            )
        elif mission.fault is not None:
            print(f"  MISSION STOPPED: {mission.fault['message']}")
        print()
    return summary


def cells(core, experiment: dict) -> str:
    return "   ".join(
        f"{name.lower()} {core.format_metric(value, unit):>8}"
        for name, value, unit in experiment["metrics"]
    )


def print_experiment(core, experiment: dict) -> None:
    """One test's result."""
    print(
        f"  {experiment['title']} {experiment['label']} COMPLETE after {experiment['steps']} "
        f"control steps: " + cells(core, experiment)
    )


def print_test_table(core, experiments) -> None:
    """The tests side by side (the same numbers as on the Webots results screen)."""
    print(f"{experiments[0]['title']}S (every run is the same)")
    for test in experiments:
        print(f"    {test['label']:<14} {cells(core, test)}")


def print_summary(rows) -> None:
    print("MISSION SUMMARY")
    print(
        f"  {'code':>4}  {'result':<9} {'time(s)':>7} {'score':>6} {'stars':<5} "
        f"{'safety stops':>12} "
        f"{'dock fails':>10} {'lost':>4} {'track err(m)':>12} {'lock(s)':>7} {'steps':>5}"
    )
    for row in rows:
        result = (
            "COMPLETE" if row["success"] else ("FAULT" if row["phase"] == "FAULT" else "FAILED")
        )
        error = "-" if row["mean_error"] is None else f"{row['mean_error']:.2f}"
        lock = "-" if row["time_to_lock"] is None else f"{row['time_to_lock']:.1f}"
        stars = "*" * row["stars"] if row["stars"] else "-"
        print(
            f"  {row['code']:04d}  {result:<9} {row['time']:7.1f} {row['score']:6d} {stars:<5} "
            f"{row['safety_stops']:12d} {row['failed_dockings']:10d} {row['relocalizations']:4d} "
            f"{error:>12} {lock:>7} {row['steps']:5d}"
        )
    done = sum(1 for row in rows if row["success"])
    print(f"  {done}/{len(rows)} missions complete.")
    print(
        "  safety stops: the robot would have hit something; dock fails: it started docking "
        "more than "
        "2.5 m\n  from the station; lost: re-localizations; track err: mean |estimate - "
        "truth| while LOCKED; "
        "lock: time\n  to the first LOCKED."
    )
    if done < len(rows):
        print(
            "  A mission that does not complete although the tests pass: read the step log "
            "above, and see "
            "'Troubleshooting'\n  in README.md."
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    core = load_engine()
    print("=" * 78)
    print(" CS 3630 PROJECT 4 - AUTONOMOUS WAREHOUSE CHALLENGE (offline mission runner)")
    print("=" * 78)
    try:
        module = core.import_student_code(args.student_code)
    except core.StudentImportError as error:
        print(f"Could not load your code: {error}")
        return 1
    bridge = core.StudentBridge(module)
    profile = bridge.robot_profile()
    info = bridge.student_info() or {}
    print(f"Student code : {Path(bridge.path or args.student_code)}")
    print(f"Student      : {info.get('name', '?')} ({info.get('gt_username', '?')})")
    print(f"Robot        : {profile['name']} ({profile['color']})")
    print()
    print_system_check(core, bridge)
    scenarios = GROUPS.get(args.scenario, (args.scenario,))
    needs_planning = any(core.SCENARIOS_BY_KEY[key].needs_planning for key in scenarios)
    needs_localization = any(core.SCENARIOS_BY_KEY[key].needs_localization for key in scenarios)
    required = (core.PLANNING_TODOS if needs_planning else frozenset()) | (
        core.LOCALIZATION_TODOS if needs_localization else frozenset()
    )
    missing = bridge.missing_core(required)
    if missing:
        what = (
            "its particle filter and planner need"
            if needs_planning and needs_localization
            else "its planner needs"
            if needs_planning
            else "its particle filter needs"
        )
        print(f"The robot cannot deploy yet: {what} these TODOs:")
        for todo, name, message in missing:
            print(f"  TODO {todo:>2}  {name:<22} {message}")
        print(
            "Implement them (the check(...) cells in project4.ipynb help), then run the "
            "mission again."
        )
        return 1
    codes = args.seeds if args.seeds else [core.random_code()]
    rows = []
    try:
        for code in codes:
            for scenario in scenarios:
                rows.append(run_one(core, bridge, code, args, scenario))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    tests = [row["experiment"] for row in rows if row.get("experiment")]
    if tests:
        print_test_table(core, tests)
    elif rows:
        print_summary(rows)
    return (
        0
        if rows and len(rows) == len(codes) * len(scenarios) and all(row["success"] for row in rows)
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
