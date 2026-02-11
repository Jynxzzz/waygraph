"""
Gap Acceptance Parameter Extraction
=====================================

Extract gap acceptance parameters from observed vehicle behavior at
intersections. Identifies situations where vehicles wait to enter a
traffic stream and measures the gaps they accept or reject.

Gap acceptance is a fundamental parameter in traffic engineering used
to calibrate microsimulation models (e.g., SUMO, VISSIM).

Key parameters:
    - Critical gap: The minimum gap a driver will accept (median of
      accepted gaps).
    - Follow-up time: Time between consecutive vehicles entering through
      the same gap.

Example::

    from waygraph.traffic import GapAcceptanceExtractor

    extractor = GapAcceptanceExtractor()
    gap = extractor.extract(scenario)
    print(f"Critical gap: {gap.critical_gap_s:.1f}s")
    print(f"Follow-up time: {gap.follow_up_time_s:.1f}s")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GapAcceptance:
    """Gap acceptance parameters from observed vehicle behavior.

    Attributes:
        critical_gap_s: Estimated critical gap in seconds (median of
            accepted gaps).
        follow_up_time_s: Estimated follow-up time in seconds.
        accepted_gaps: List of gap durations that were accepted (seconds).
        rejected_gaps: List of gap durations that were rejected (seconds).
    """

    critical_gap_s: float = 0.0
    follow_up_time_s: float = 0.0
    accepted_gaps: List[float] = field(default_factory=list)
    rejected_gaps: List[float] = field(default_factory=list)

    @property
    def n_accepted(self) -> int:
        """Number of accepted gaps."""
        return len(self.accepted_gaps)

    @property
    def n_rejected(self) -> int:
        """Number of rejected gaps."""
        return len(self.rejected_gaps)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "critical_gap_s": round(self.critical_gap_s, 2),
            "follow_up_time_s": round(self.follow_up_time_s, 2),
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
        }


class GapAcceptanceExtractor:
    """Extract gap acceptance parameters from WOMD scenarios.

    Detects stop-to-go transitions and measures the time gaps to the
    nearest conflicting vehicle at each transition.

    Args:
        min_speed_ms: Speed threshold below which a vehicle is considered
            stopped.
        max_gap_s: Maximum gap duration to consider (filters outliers).
        min_gap_s: Minimum gap duration to consider.
    """

    def __init__(
        self,
        min_speed_ms: float = 0.5,
        max_gap_s: float = 15.0,
        min_gap_s: float = 0.5,
    ):
        self.min_speed_ms = min_speed_ms
        self.max_gap_s = max_gap_s
        self.min_gap_s = min_gap_s

    def extract(self, scenario: Dict[str, Any]) -> GapAcceptance:
        """Extract gap acceptance parameters from a scenario.

        Identifies vehicles transitioning from stopped to moving
        and measures the gap to the nearest conflicting vehicle.

        Args:
            scenario: Loaded scenario dictionary.

        Returns:
            GapAcceptance with estimated parameters.
        """
        objects = scenario.get("objects", [])
        if not objects:
            return GapAcceptance()

        vehicles = [obj for obj in objects if obj.get("type") == "vehicle"]

        accepted_gaps: List[float] = []
        rejected_gaps: List[float] = []

        for obj in vehicles:
            positions = obj["position"]
            velocities = obj["velocity"]
            valid = obj["valid"]

            valid_indices = [i for i, v in enumerate(valid) if v]
            if len(valid_indices) < 20:
                continue

            for t_idx in range(len(valid_indices) - 1):
                t = valid_indices[t_idx]
                t_next = valid_indices[t_idx + 1]

                vel = np.array([velocities[t]["x"], velocities[t]["y"]])
                speed = np.linalg.norm(vel)

                vel_next = np.array(
                    [velocities[t_next]["x"], velocities[t_next]["y"]]
                )
                speed_next = np.linalg.norm(vel_next)

                pos = np.array([positions[t]["x"], positions[t]["y"]])

                # Detect stop-to-go transition (accepted gap)
                if speed < self.min_speed_ms and speed_next > self.min_speed_ms:
                    gap = self._find_nearest_gap(pos, t, vehicles)
                    if gap is not None and gap > 0:
                        accepted_gaps.append(gap)

                # Detect continued waiting (rejected gap)
                elif (
                    speed < self.min_speed_ms
                    and speed_next < self.min_speed_ms
                ):
                    gap = self._find_nearest_gap(pos, t, vehicles)
                    if (
                        gap is not None
                        and self.min_gap_s < gap < self.max_gap_s
                    ):
                        rejected_gaps.append(gap)

        # Estimate parameters
        result = GapAcceptance(
            accepted_gaps=accepted_gaps,
            rejected_gaps=rejected_gaps,
        )

        if accepted_gaps:
            result.critical_gap_s = float(np.median(accepted_gaps))
        if len(accepted_gaps) >= 2:
            sorted_gaps = sorted(accepted_gaps)
            if len(sorted_gaps) >= 2:
                result.follow_up_time_s = float(
                    np.mean(np.diff(sorted_gaps[:10]))
                )

        return result

    def _find_nearest_gap(
        self,
        pos: np.ndarray,
        timestep: int,
        vehicles: List[Dict[str, Any]],
    ) -> Optional[float]:
        """Find the time gap to the nearest conflicting vehicle.

        Args:
            pos: Position of the waiting vehicle.
            timestep: Current timestep index.
            vehicles: List of all vehicle objects.

        Returns:
            Time gap in seconds, or None if no valid gap found.
        """
        min_gap: Optional[float] = None

        for other in vehicles:
            positions = other["position"]
            velocities = other["velocity"]
            valid = other["valid"]

            if not valid[timestep]:
                continue

            other_pos = np.array(
                [positions[timestep]["x"], positions[timestep]["y"]]
            )
            other_vel = np.array(
                [velocities[timestep]["x"], velocities[timestep]["y"]]
            )
            other_speed = np.linalg.norm(other_vel)

            if other_speed < self.min_speed_ms:
                continue

            dist = np.linalg.norm(other_pos - pos)
            if dist < 2.0 or dist > 50.0:
                continue

            time_gap = dist / other_speed

            if min_gap is None or time_gap < min_gap:
                min_gap = time_gap

        return min_gap
