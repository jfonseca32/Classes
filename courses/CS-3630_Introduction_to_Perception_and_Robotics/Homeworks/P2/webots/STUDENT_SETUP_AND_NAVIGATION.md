# Project 2 Webots: Student Setup and Navigation

Use this guide after completing and testing Project 2 in local Jupyter. Webots
is a visualization of your completed work; it does not add another programming
task. The supplied controller imports the same `main.py` file beside the
notebook and calls your `bayes_given_all_sensors(...)` and
`make_decision(...)` functions.

## 1. Install and check the software

1. Install **Webots R2025a** for your operating system.
2. Keep the supplied `webots/` directory beside `main.py` and the notebook. Do
   not move the world, controller, PROTO, or asset files independently.
3. Activate the same local Python environment that you use for the Project 2
   notebook. This activity does not use Colab.
4. In a terminal opened from the supplied student project directory, run:

   ```bash
   python -c "import numpy, gtsam, gtbook; print('Project 2 imports OK')"
   python -c "import sys; print(sys.executable)"
   ```

   If the first command fails, install the missing Project 2 dependency in this
   environment before continuing. The second command prints the exact Python
   interpreter Webots should use.
5. Open Webots Preferences/Settings, find **Python command**, and select the
   interpreter printed by the previous command. Restart Webots after changing
   this preference.

## 2. Open and run the world

1. In Webots, choose **File > Open World**.
2. Open `webots/worlds/project2_sorting.wbt`.
3. Wait for the scene to finish loading. The ABB arm and its meshes are included
   with the project. The first launch may still take longer while Webots
   downloads and caches decorative Cyberbotics factory assets.
4. Click the triangular **Run** button. Keep the Webots console visible.
5. The controller automatically loads the `main.py` beside `webots/`. For every
   item it:
   sends the simulated sensor observation to `bayes_given_all_sensors(...)`,
   sends the returned posterior to `make_decision(...)`, and routes the item to
   the bin whose number matches the returned action.

The automatic run processes five items: a 3D-printed product, structural
bracing, PCB board, wing component, and motor. The action-to-bin mapping is:

| Action | Destination |
|---:|---|
| 0 | Printed bin |
| 1 | Structural bin |
| 2 | Board bin |
| 3 | Wing bin |
| 4 | Motor bin |

## 3. Navigate the 3D view

- **Rotate/orbit:** left-click and drag in the 3D scene. Starting the drag on an
  object rotates around the selected point.
- **Pan:** right-click and drag.
- **Zoom:** use the mouse wheel or two-finger scroll. A middle-button vertical
  drag also zooms.
- **Pause/resume:** use the pause and run buttons in the top toolbar.
- **Restart the five-item run:** click inside the 3D view and press **R**, or use
  the Webots reset button and then press Run.
- **Restore the supplied camera:** use the toolbar's **Restore Viewpoint**
  button. Reopening the world also restores its saved starting view if you do
  not save your changed viewpoint.

Do not drag the robot, conveyor, items, or bins. Moving scene objects changes
the supplied pickup and drop coordinates.

## 4. Read the run and troubleshoot

Webots reports the sensor observation, posterior, selected action, physical
channel, and arm joint command in the console. The viewport intentionally has
no text overlay so the robot and bins remain unobstructed.

If the console reports `No module named 'gtbook'` (or another missing module),
Webots is either using the wrong Python interpreter or that package is absent
from the selected environment. Repeat the two commands in Section 1 and check
the Webots Python command setting.

If your code raises an error or returns an invalid action, correct and save
`main.py`, then reset the simulation. You do not copy code into the Webots
controller. Keys **0** through **4** are a manual troubleshooting fallback;
they are not valid evidence for the reflections because they bypass your
automatic decision.
