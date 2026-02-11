"""
Speed Distribution Analysis
=============================

Extract speed distributions per lane from observed vehicle trajectories
in WOMD scenarios. Provides statistics including mean speed, free-flow
speed (85th percentile), and fitted distribution parameters.

Example::

    from waygraph.traffic import SpeedExtractor

    extractor = SpeedExtractor()
    distributions = extractor.extract(scenario)

    for lane_id, sd in distributions.items():
        print(f"Lane {lane_id}: mean={sd.mean_speed_kmh:.1f} km/h, "
              f"free-flow={sd.free_flow_speed_kmh:.1f} km/h")
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial import KDTree


@dataclass
class SpeedDistribution:
    """Speed distribution for a road segment or lane.

    Attributes:
        segment_id: Identifier for the lane or segment.
        speeds_ms: Raw speed observations in m/s.
        mean_speed_ms: Mean speed in m/s.
        std_speed_ms: Standard deviation of speed in m/s.
        percentile_15: 15th percentile speed in m/s.
        percentile_85: 85th percentile speed in m/s.
        free_flow_speed_ms: Free-flow speed (85th percentile) in m/s.
    """

    segment_id: str = ""
    speeds_ms: List[float] = field(default_factory=list)
    mean_speed_ms: float = 0.0
    std_speed_ms: float = 0.0
    percentile_15: float = 0.0
    percentile_85: float = 0.0
    free_flow_speed_ms: float = 0.0

    @property
    def mean_speed_kmh(self) -> float:
        """Mean speed in km/h."""
        return self.mean_speed_ms * 3.6

    @property
    def free_flow_speed_kmh(self) -> float:
        """Free-flow speed in km/h."""
        return self.free_flow_speed_ms * 3.6

    def fit(self) -> None:
        """Fit distribution parameters from the raw speed data.

        Filters near-zero speeds (< 0.5 m/s) and computes mean,
        standard deviation, and percentiles.
        """
        if len(self.speeds_ms) < 2:
            return
        speeds = np.array(self.speeds_ms)
        speeds = speeds[speeds > 0.5]
        if len(speeds) < 2:
            return
        self.mean_speed_ms = float(np.mean(speeds))
        self.std_speed_ms = float(np.std(speeds))
        self.percentile_15 = float(np.percentile(speeds, 15))
        self.percentile_85 = float(np.percentile(speeds, 85))
        self.free_flow_speed_ms = self.percentile_85

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "segment_id": self.segment_id,
            "n_samples": len(self.speeds_ms),
            "mean_speed_ms": round(self.mean_speed_ms, 2),
            "mean_speed_kmh": round(self.mean_speed_kmh, 1),
            "std_speed_ms": round(self.std_speed_ms, 2),
            "free_flow_speed_ms": round(self.free_flow_speed_ms, 2),
            "free_flow_speed_kmh": round(self.free_flow_speed_kmh, 1),
        }


class SpeedExtractor:
    """Extract speed distributions per lane from WOMD scenarios.

    For each timestep, assigns each moving vehicle to the nearest lane
    centerline point and records its instantaneous speed. The resulting
    per-lane speed samples are used to fit distribution parameters.

    Args:
        min_speed_ms: Minimum speed to consider a vehicle as moving.
        lane_assignment_radius: Max distance to assign a vehicle to a lane.
        subsample_step: Subsample lane points every N steps for efficiency.
    """

    def __init__(
        self,
        min_speed_ms: float = 0.5,
        lane_assignment_radius: float = 5.0,
        subsample_step: int = 3,
    ):
        self.min_speed_ms = min_speed_ms
        self.lane_assignment_radius = lane_assignment_radius
        self.subsample_step = subsample_step

    def extract(
        self,
        scenario: Dict[str, Any],
        topology: Optional[Any] = None,
    ) -> Dict[str, SpeedDistribution]:
        """Extract speed distributions per lane from a scenario.

        Args:
            scenario: Loaded scenario dictionary.
            topology: Optional ScenarioTopology for context.

        Returns:
            Dictionary mapping lane_id to SpeedDistribution.
        """
        objects = scenario.get("objects", [])
        lane_graph = scenario.get("lane_graph", {})
        lanes = lane_graph.get("lanes", {})

        if not lanes or not objects:
            return {}

        # Build lane lookup with subsampling
        lane_points_list: List[np.ndarray] = []
        lane_ids_for_points: List[int] = []
        for lane_id, polyline in lanes.items():
            for pt in polyline[:: self.subsample_step]:
                lane_points_list.append(pt[:2])
                lane_ids_for_points.append(lane_id)

        if not lane_points_list:
            return {}

        lane_tree = KDTree(np.array(lane_points_list))

        # Collect speeds per lane
        lane_speeds: Dict[str, List[float]] = defaultdict(list)

        for obj in objects:
            if obj.get("type") != "vehicle":
                continue

            positions = obj["position"]
            velocities = obj["velocity"]
            valid = obj["valid"]

            for t in range(len(positions)):
                if not valid[t]:
                    continue

                pos = np.array([positions[t]["x"], positions[t]["y"]])
                vel = np.array([velocities[t]["x"], velocities[t]["y"]])
                speed = float(np.linalg.norm(vel))

                if speed < self.min_speed_ms:
                    continue

                dist, idx = lane_tree.query(pos)
                if dist > self.lane_assignment_radius:
                    continue

                lane_id = lane_ids_for_points[idx]
                lane_speeds[str(lane_id)].append(speed)

        # Build SpeedDistribution objects
        result: Dict[str, SpeedDistribution] = {}
        for lane_id, speeds in lane_speeds.items():
            sd = SpeedDistribution(segment_id=lane_id, speeds_ms=speeds)
            sd.fit()
            result[lane_id] = sd

        return result
