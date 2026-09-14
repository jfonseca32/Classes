# Project 2 Webots Sorting Visualization

This supplied Webots component visualizes the decisions produced by the
Project 2 probability notebook. It does not add programming TODOs and does not
replace or modify any existing NumPy, GTSAM, testing, or autograder code.

Students should follow [`STUDENT_SETUP_AND_NAVIGATION.md`](STUDENT_SETUP_AND_NAVIGATION.md)
for installation, Python environment setup, camera controls, and
troubleshooting. The required reflection prompts remain in the supplied
`project2.tex` reflection handout. This document provides the technical
overview for students and course staff.

## What the simulation shows

Five imported CAD/STL items move along an intake conveyor:

1. 3D-printed product
2. structural bracing
3. PCB board
4. wing component
5. motor

At the pickup station, an official Cyberbotics **ABB IRB 4600/40** industrial
robot moves the item with:

1. six coordinated revolute joints (**A** through **F**); and
2. a powered two-finger **claw** supplied for this visualization.

The notebook and Webots import the same completed student Project 2 module.
For each item, Webots supplies simulated sensor readings to the student's
`bayes_given_all_sensors(...)` function and passes its result to the student's
`make_decision(...)` function. Each returned action routes directly to a
matching physical channel:

| Notebook action | Physical route |
|---|---|
| Printed Bin (0) | Channel 0 — printed bin |
| Structural Bin (1) | Channel 1 — structural bin |
| Board Bin (2) | Channel 2 — board bin |
| Wing Bin (3) | Channel 3 — wing bin |
| Motor Bin (4) | Channel 4 — motor bin |

The five channels are destinations, not individual joints. Each channel has a
coordinated six-axis arm configuration that moves the ABB base, shoulder,
elbow, and three-axis wrist to reach that route; a channel does not control
only one joint. The destination joint configurations belong to the supplied
visualization controller, so students do not implement them. Course staff can
revise `CHANNEL_POSES` in `controllers/project2_sorting/project2_sorting.py`
without changing the student notebook.

The supplied controller contains no replacement implementation of either
assignment function. Import errors, runtime errors, malformed posteriors, and
actions outside `0` through `4` are shown in the Webots console.

## Python environment

Run Jupyter and Webots with the same local course Python environment. Because
`main.py` imports NumPy, GTSAM, and `gtbook`, that environment must make all
three packages available even though this visualization calls only the NumPy
inference and decision functions.

From the active environment, verify the dependencies and print its interpreter
path:

```bash
python -c "import numpy, gtsam, gtbook; print('Project 2 imports OK')"
```

Then point Webots' **Python command** preference at that same interpreter
(`which python` on macOS/Linux or `where.exe python` on Windows). If Webots is
using a different interpreter, the console's student-code error will identify
the missing dependency.

## Run the visualization

1. Complete `main.py` and use it from Jupyter as directed by the assignment.
2. Open `worlds/project2_sorting.wbt` in Webots and press **Run**. The supplied
   controller automatically imports the same `main.py` beside the student
   notebook and `webots/` directory; students do not copy or move their
   functions.
3. At each sensor station, Webots automatically runs the student's inference
   and decision functions, displays the observations/posteriors/action in the
   console, and performs the resulting pick-and-place motion.
4. Press **R** to restart the complete five-item sequence.

The compact room, conveyor, floor, walls, ceiling, ABB robot, cabinet, pallet,
box, and barrel use official Cyberbotics Webots PROTO assets. The ABB PROTO and
meshes are supplied locally so the arm appears without downloading from
GitHub. Decorative factory assets may be downloaded and cached on first open.
The scene uses one physical room rather than layering a panoramic room behind
it.

Keys **0** through **4** remain available only as a manual troubleshooting
fallback. Normal student evidence should show the automatic `main.py` path.
Course staff may pass `--student-code /absolute/path/to/module.py` as a
controller argument when testing an alternate module such as `main_sols.py`.

The five normalized meshes and their licenses are documented in
`assets/ATTRIBUTION.md`.

## Files supplied to students

```text
webots/
├── assets/
│   ├── ATTRIBUTION.md
│   └── five normalized STL meshes
├── PROJECT_2_WEBOTS_GUIDE.md
├── STUDENT_SETUP_AND_NAVIGATION.md
├── controllers/
│   └── project2_sorting/
│       └── project2_sorting.py
├── protos/
│   ├── IndustrialSortingArm.proto
│   ├── Irb4600-40.proto
│   ├── Irb4600-40/meshes/
│   ├── SortingBin.proto
│   └── WarehouseItem.proto
└── worlds/
    └── project2_sorting.wbt
```

Preserve this directory structure. Webots finds controllers and local PROTO
files relative to the world.
