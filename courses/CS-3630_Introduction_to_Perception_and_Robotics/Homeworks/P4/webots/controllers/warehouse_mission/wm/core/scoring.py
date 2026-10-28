"""Mission statistics and the score (DESIGN 5.5).

::

    score = 10000
          - 30   * mission_time_s / orders
          - 300  * safety_stops
          - 750  * failed_dockings
          - 400  * relocalizations (LOST events)
          + 2000 * max(0, 1 - mean_tracking_error_m)     # while LOCKED, vs truth
    clamped to >= 0, rounded to int.
    Stars: 3 if score >= 8000, 2 if >= 6000, else 1 (0 if the mission failed).

The constants live in ``config.py`` (``SCORE_*``). DESIGN 5.5 first suggested
40 points per second and asks to tune the constants so that a clean staff run
scores about 8000-9000; 30 per second does that (see config.py). The time
term is per order so that a 3-order "double shift" is scored on the same
scale as the default 1-order mission (identical for 1 order).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from . import config as C


@dataclass
class MissionStats:
    """Running statistics of one mission (the HUD and the final table read these)."""

    mission_time: float = 0.0  # s, from deployment
    control_steps: int = 0  # moves + waits (the step limit counts these)
    moves: int = 0
    waits: int = 0
    safety_stops: int = 0
    failed_dockings: int = 0
    relocalizations: int = 0  # LOST events
    replans: int = 0
    kidnaps: int = 0
    obstacles: int = 0
    orders_completed: int = 0
    orders_total: int = 1
    time_to_lock: float | None = None  # s from deployment to the first LOCKED
    tracking_error_sum: float = 0.0
    tracking_error_count: int = 0
    tracking_error_max: float = 0.0
    arrival_errors: list[float] = field(default_factory=list)  # |truth - station| at each docking
    faults: int = 0

    @property
    def mean_tracking_error(self) -> float | None:
        if self.tracking_error_count == 0:
            return None
        return self.tracking_error_sum / self.tracking_error_count

    def record_tracking_error(self, error: float) -> None:
        self.tracking_error_sum += float(error)
        self.tracking_error_count += 1
        self.tracking_error_max = max(self.tracking_error_max, float(error))

    def snapshot(self, success: bool = False) -> dict[str, Any]:
        """Plain-dict copy with the live score (safe to keep)."""
        data = asdict(self)
        data["mean_tracking_error"] = self.mean_tracking_error
        data.update(score_breakdown(self, success=success, final=False))
        return data


def stars_for(score: int) -> int:
    if score >= C.STAR_THRESHOLDS[0]:
        return 3
    if score >= C.STAR_THRESHOLDS[1]:
        return 2
    return 1


def score_breakdown(stats: MissionStats, success: bool, final: bool = True) -> dict[str, Any]:
    """Every term of the score, the total, the stars and the title.

    While the mission is running (``final=False``) the stars show what the
    current score would earn; a finished mission that failed earns 0 stars.
    """
    orders = max(1, stats.orders_total)
    error = stats.mean_tracking_error
    terms = {
        "base": float(C.SCORE_BASE),
        "time": -C.SCORE_PER_SECOND * stats.mission_time / orders,
        "safety_stops": -float(C.SCORE_PER_SAFETY_STOP * stats.safety_stops),
        "failed_dockings": -float(C.SCORE_PER_FAILED_DOCKING * stats.failed_dockings),
        "relocalizations": -float(C.SCORE_PER_RELOCALIZATION * stats.relocalizations),
        "tracking": C.SCORE_TRACKING_BONUS * max(0.0, 1.0 - error) if error is not None else 0.0,
    }
    score = int(round(max(0.0, sum(terms.values()))))
    stars = stars_for(score) if (success or not final) else 0
    return {"score": score, "stars": stars, "title": C.TITLES[stars], "score_terms": terms}
