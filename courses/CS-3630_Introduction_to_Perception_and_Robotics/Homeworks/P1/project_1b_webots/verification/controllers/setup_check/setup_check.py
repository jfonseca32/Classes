"""CS 3630 Webots installation verification controller."""

import math
import sys

from controller import Supervisor

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
translation = robot.getSelf().getField("translation")

print(f"Python {sys.version.split()[0]}: {sys.executable}", flush=True)

try:
    import numpy
except ImportError:
    print("CS 3630 WEBOTS SETUP FAILURE: this Python cannot import NumPy.", flush=True)
    print(
        "Install the Project 1A requirements into cs3630_p1, confirm the "
        "Webots Python command points at cs3630_p1, then reload this world.",
        flush=True,
    )
else:
    print(f"NumPy {numpy.__version__}", flush=True)
    print("CS 3630 WEBOTS SETUP SUCCESS", flush=True)

while robot.step(timestep) != -1:
    time = robot.getTime()
    translation.setSFVec3f([0.8 * math.sin(time), 0.0, 0.12])
