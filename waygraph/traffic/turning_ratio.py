"""
Turning Ratio Extraction
=========================

Extract turning movement counts and ratios at intersections from observed
vehicle trajectories in WOMD scenarios.

Each vehicle is assigned to an entry lane using a spatial KD-tree lookup,
and its turning movement (left, through, right, U-turn) is determined from
the heading change between the start and end of its trajectory.

Example::

    from waygraph.traffic import TurningRatioExtractor

    extractor = TurningRatioExtractor()
    movements = extractor.extract(scenario)

    for approach_id, tm in movements.items():
        print(f"{approach_id}: L={tm.left_ratio:.0%} T={tm.through_ratio:.0%} R={tm.right_ratio:.0%}")
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial import KDTree


@dataclass
class TurningMovement:
    """Turning movement data at an intersection approach.

    Attributes:
        approach_direction: Identifier for the approach (e.g., 'lane_42').
        approach_angle: Heading angle of the approach in degrees.
        left_count: Number of left-turning vehicles.
        through_count: Number of through-going vehicles.
        right_count: Number of right-turning vehicles.
        uturn_count: Number of U-turning vehicles.
        total_count: Total number of observed vehicles.
    """

    approach_direction: str = ""
    approach_angle: float = 0.0
    left_count: int = 0
    through_count: int = 0
    right_count: int = 0
    uturn_count: int = 0
    total_count: int = 0

    @property
    def left_ratio(self) -> float:
        """Fraction of vehicles turning left."""
        return self.left_count / max(self.total_count, 1)

    @property
    def through_ratio(self) -> float:
        """Fraction of vehicles going through."""
        return self.through_count / max(self.total_count, 1)

    @property
    def right_ratio(self) -> float:
        """Fraction of vehicles turning right."""
        return self.right_count / max(self.total_count, 1)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "approach_direction": self.approach_direction,
            "approach_angle": self.approach_angle,
            "left": self.left_count,
            "through": self.through_count,
            "right": self.right_count,
            "uturn": self.uturn_count,
            "total": self.total_count,
            "left_ratio": round(self.left_ratio, 3),
            "through_ratio": round(self.through_ratio, 3),
            "right_ratio": round(self.right_ratio, 3),
        }


class TurningRatioExtractor:
    """Extract turning movement ratios from WOMD scenarios.

    Assigns each vehicle trajectory to an entry lane using spatial
    proximity, then classifies the turn based on heading change.

    Args:
        lane_assignment_radius: Maximum distance (meters) to assign a
            vehicle to a lane centerline point.
        min_valid_timesteps: Minimum number of valid timesteps for a
            trajectory to be included.
    """

    def __init__(
        self,
        lane_assignment_radius: float = 5.0,
        min_valid_timesteps: int = 10,
    ):
        self.lane_assignment_radius = lane_assignment_radius
        self.min_valid_timesteps = min_valid_timesteps

    def extract(
        self,
        scenario: Dict[str, Any],
        topology: Optional[Any] = None,
    ) -> Dict[str, TurningMovement]:
        """Extract turning movements from a scenario.

        Args:
            scenario: Loaded scenario dictionary with 'objects' and 'lane_graph'.
            topology: Optional ScenarioTopology for additional context.

        Returns:
            Dictionary mapping approach_id to TurningMovement.
        """
        objects = scenario.get("objects", [])
        lane_graph = scenario.get("lane_graph", {})
        lanes = lane_graph.get("lanes", {})

        if not lanes or not objects:
            return {}

        # Build KD-tree of lane centerline points
        lane_points_list: List[np.ndarray] = []
        lane_ids_for_points: List[int] = []
        for lane_id, polyline in lanes.items():
            for pt in polyline:
                lane_points_list.append(pt[:2])
                lane_ids_for_points.append(lane_id)

        if not lane_points_list:
            return {}

        lane_tree = KDTree(np.array(lane_points_list))

        # Analyze vehicle trajectories
        turning_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"left": 0, "through": 0, "right": 0, "uturn": 0}
        )
        entry_angles: Dict[str, float] = {}

        for obj in objects:
            if obj.get("type") != "vehicle":
                continue

            positions = obj["position"]
            headings = obj["heading"]
            valid = obj["valid"]

            valid_indices = [i for i, v in enumerate(valid) if v]
            if len(valid_indices) < self.min_valid_timesteps:
                continue

            start_idx = valid_indices[0]
            end_idx = valid_indices[-1]

            start_pos = np.array(
                [positions[start_idx]["x"], positions[start_idx]["y"]]
            )
            start_heading = headings[start_idx]
            end_heading = headings[end_idx]

            # Assign to entry lane
            dist, idx = lane_tree.query(start_pos)
            if dist > self.lane_assignment_radius * 2:
                continue

            entry_lane = lane_ids_for_points[idx]
            heading_change = self._normalize_angle(end_heading - start_heading)
            turn = self._classify_turn(heading_change)

            approach_id = f"lane_{entry_lane}"
            turning_counts[approach_id][turn] += 1

            if approach_id not in entry_angles:
                entry_angles[approach_id] = start_heading

        # Build result
        result: Dict[str, TurningMovement] = {}
        for approach_id, counts in turning_counts.items():
            tm = TurningMovement(
                approach_direction=approach_id,
                approach_angle=entry_angles.get(approach_id, 0.0),
                left_count=counts["left"],
                through_count=counts["through"],
                right_count=counts["right"],
                uturn_count=counts["uturn"],
                total_count=sum(counts.values()),
            )
            result[approach_id] = tm

        return result

    @staticmethod
    def _normalize_angle(angle_deg: float) -> float:
        """Normalize angle to [-180, 180] degrees."""
        angle = angle_deg % 360.0
        if angle > 180.0:
            angle -= 360.0
        return angle

    @staticmethod
    def _classify_turn(heading_change_deg: float) -> str:
        """Classify turn type from heading change.

        Uses thresholds that account for road curvature in real data.

        Args:
            heading_change_deg: Heading change in degrees [-180, 180].

        Returns:
            One of 'left', 'through', 'right', 'uturn'.
        """
        abs_change = abs(heading_change_deg)
        if abs_change < 45:
            return "through"
        elif abs_change > 135:
            return "uturn"
        elif heading_change_deg > 0:
            return "left"
        else:
            return "right"

    def extract_from_sdc_paths(
        self,
        sdc_paths: List,
        lane_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, "TurningMovement"]:
        """Extract turning movements from v1.3.1 SDC paths (route choices).

        Unlike the heading-based method, this uses explicit route geometry
        from pre-computed SDC paths. The ``on_route`` flag identifies the
        actually chosen route, while other paths represent alternatives.

        This provides ground-truth turning classification for the SDC vehicle,
        which can validate the heading-based extraction method.

        Args:
            sdc_paths: List of SDCPath objects from v1.3.1 scenario loading.
            lane_graph: Optional lane graph dict for lane ID lookup.

        Returns:
            Dictionary mapping route_id to TurningMovement.
        """
        if not sdc_paths:
            return {}

        result: Dict[str, TurningMovement] = {}

        for idx, path in enumerate(sdc_paths):
            xy = path.xy
            if len(xy) < 2:
                continue

            # Compute heading at start and end of path
            start_vec = xy[min(5, len(xy) - 1)] - xy[0]
            end_vec = xy[-1] - xy[max(0, len(xy) - 6)]

            start_heading = float(np.degrees(np.arctan2(start_vec[1], start_vec[0])))
            end_heading = float(np.degrees(np.arctan2(end_vec[1], end_vec[0])))

            heading_change = self._normalize_angle(end_heading - start_heading)
            turn = self._classify_turn(heading_change)

            route_id = f"sdc_path_{idx}" + ("_on_route" if path.on_route else "")

            # Count as 1 observation per path
            counts = {"left": 0, "through": 0, "right": 0, "uturn": 0}
            counts[turn] = 1

            tm = TurningMovement(
                approach_direction=route_id,
                approach_angle=start_heading,
                left_count=counts["left"],
                through_count=counts["through"],
                right_count=counts["right"],
                uturn_count=counts["uturn"],
                total_count=1,
            )
            result[route_id] = tm

        return result

    def compare_methods(
        self,
        scenario: Dict[str, Any],
        sdc_paths: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Compare heading-based vs. sdc_paths-based turning classification.

        Useful for validating the heading-based method against the more
        reliable sdc_paths ground truth (v1.3.1+).

        Args:
            scenario: Loaded scenario dictionary.
            sdc_paths: Optional list of SDCPath objects.

        Returns:
            Dictionary with comparison metrics including agreement rate.
        """
        heading_result = self.extract(scenario)

        if not sdc_paths:
            return {
                "heading_approaches": len(heading_result),
                "sdc_paths_available": False,
                "agreement_rate": None,
            }

        sdc_result = self.extract_from_sdc_paths(sdc_paths)

        # Find the on_route path's turn classification
        sdc_on_route_turn = None
        for key, tm in sdc_result.items():
            if "on_route" in key:
                if tm.left_count > 0:
                    sdc_on_route_turn = "left"
                elif tm.through_count > 0:
                    sdc_on_route_turn = "through"
                elif tm.right_count > 0:
                    sdc_on_route_turn = "right"
                else:
                    sdc_on_route_turn = "uturn"
                break

        return {
            "heading_approaches": len(heading_result),
            "sdc_paths_available": True,
            "sdc_path_count": len(sdc_result),
            "sdc_on_route_turn": sdc_on_route_turn,
            "heading_turns": {
                k: max(
                    ("through", v.through_count),
                    ("left", v.left_count),
                    ("right", v.right_count),
                    ("uturn", v.uturn_count),
                    key=lambda x: x[1],
                )[0]
                for k, v in heading_result.items()
            },
        }
