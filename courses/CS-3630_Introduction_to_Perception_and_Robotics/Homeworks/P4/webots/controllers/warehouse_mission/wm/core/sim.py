"""The real robot: omni-wheel kinematics, motion noise, and the safety layer.

Book Section 4.2: the brain commands a body displacement ``u`` (1 m per 0.5 s
step). The wheel speeds come from the *student's* Jacobian,
``omega = J_student @ (u / dt)``, and the wheels move the body according to
the *real* robot, ``v = pinv(J_true) @ omega``. With a correct Jacobian the
two cancel; with a wrong one the robot really drives the wrong way.

The true motion adds Gaussian noise ``w ~ N(0, 0.2^2 I)`` and runs as a
straight segment. The safety layer stops the robot just before it would touch
a shelf, wall, obstacle, or forklift (SAFETY STOP); the brain then learns the
executed control ``u_eff = u * fraction_completed`` (bumper + odometry).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from .world import World

J_TRUE_PINV = np.linalg.pinv(C.J_TRUE)


def wheel_speeds(J: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Wheel angular speeds (rad/s) for a body velocity (m/s): ``omega = J v``."""
    return np.asarray(J, dtype=float) @ np.asarray(velocity, dtype=float)


def realized_velocity(omega: np.ndarray) -> np.ndarray:
    """Body velocity (m/s) that the real wheels produce: ``v = pinv(J_true) omega``."""
    return J_TRUE_PINV @ np.asarray(omega, dtype=float)


@dataclass
class MoveOutcome:
    start: np.ndarray
    end: np.ndarray
    intended: np.ndarray  # the noisy displacement the wheels tried to execute
    fraction: float  # completed fraction of ``intended``
    blocked_by: str | None  # SAFETY STOP cause, or None
    u_eff: np.ndarray  # what the brain is told it executed
    omega_cmd: np.ndarray  # commanded wheel speeds (rad/s)
    omega: np.ndarray  # wheel speeds that match the realized motion (rad/s)

    @property
    def safety_stop(self) -> bool:
        return self.blocked_by is not None


class TruthRobot:
    """The robot's true position and the physics of one control step."""

    def __init__(self, position: np.ndarray, jacobian: np.ndarray):
        self.position = np.asarray(position, dtype=float).copy()
        self.jacobian = np.asarray(jacobian, dtype=float)
        # The body motion the (possibly wrong) student Jacobian really produces
        # per commanded meter: B_real = pinv(J_true) J_student (I when correct).
        self.B_real = J_TRUE_PINV @ self.jacobian

    def plan(self, u: np.ndarray, rng: np.random.Generator):
        """Commanded wheel speeds and the noisy displacement for command ``u``."""
        u = np.asarray(u, dtype=float)
        omega_cmd = wheel_speeds(self.jacobian, u / C.STEP_DT)
        displacement = realized_velocity(omega_cmd) * C.STEP_DT + rng.normal(
            0.0, C.TRUE_MOTION_SIGMA, 2
        )
        return omega_cmd, displacement

    def execute(
        self,
        world: World,
        u: np.ndarray,
        omega_cmd: np.ndarray,
        displacement: np.ndarray,
        duration: float = C.MOVE_DURATION,
    ) -> MoveOutcome:
        """Run the straight segment through the safety layer and update the truth."""
        start = self.position.copy()
        sweep = world.sweep(start, displacement)
        self.position = sweep.end.copy()
        u = np.asarray(u, dtype=float)
        return MoveOutcome(
            start=start,
            end=self.position.copy(),
            intended=np.asarray(displacement, dtype=float),
            fraction=sweep.fraction,
            blocked_by=sweep.blocked_by,
            u_eff=u * sweep.fraction,
            omega_cmd=np.asarray(omega_cmd, dtype=float),
            omega=wheel_speeds(C.J_TRUE, (self.position - start) / duration),
        )

    def drive_to(self, world: World, target: np.ndarray, duration: float) -> MoveOutcome:
        """Precise, marker-guided motion (docking): no noise, still safety-checked."""
        start = self.position.copy()
        displacement = np.asarray(target, dtype=float) - start
        sweep = world.sweep(start, displacement)
        self.position = sweep.end.copy()
        velocity = (self.position - start) / duration
        return MoveOutcome(
            start=start,
            end=self.position.copy(),
            intended=displacement,
            fraction=sweep.fraction,
            blocked_by=sweep.blocked_by,
            u_eff=self.position - start,
            omega_cmd=wheel_speeds(C.J_TRUE, velocity),
            omega=wheel_speeds(C.J_TRUE, velocity),
        )

    def teleport(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=float).copy()
