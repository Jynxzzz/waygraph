"""
Intersection Detection and Classification
===========================================

Detects and classifies intersections from lane connectivity graphs.
Identifies approach directions, clusters them, and assigns intersection
types: T, Y, cross (4-way), multi (5+), roundabout, merge, or none.

The classification pipeline:
    1. Find terminal lanes (entry/exit lanes at the boundary)
    2. Compute approach directions from terminal lane endpoints
    3. Cluster directions into distinct approach arms
    4. Classify intersection type from approach count and angular distribution

Example::

    from waygraph.core.intersection import IntersectionClassifier

    classifier = IntersectionClassifier()
    classifier.classify(topology, lane_graph_dict)
    print(topology.intersection_type)  # 'cross'
    print(topology.num_approaches)     # 4
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from waygraph.utils.geometry import angular_diff as _angular_diff_util
from waygraph.utils.geometry import circular_mean as _circular_mean_util

if TYPE_CHECKING:
    from waygraph.core.lane_graph import ScenarioTopology


class IntersectionClassifier:
    """Classify intersection type from lane graph topology.

    Analyzes entry/exit lane directions, clusters them into distinct
    approach arms, and assigns an intersection type label.

    Args:
        angle_cluster_threshold: Angular threshold (degrees) for merging
            nearby approach directions into the same arm.
    """

    # Intersection type labels
    TYPES = ("none", "merge", "Y", "T", "cross", "multi", "roundabout")

    def __init__(self, angle_cluster_threshold: float = 30.0):
        self.angle_cluster_threshold = angle_cluster_threshold

    def classify(
        self,
        topo: "ScenarioTopology",
        lane_graph: dict,
    ) -> None:
        """Classify the intersection and populate topology fields.

        Modifies the topology in place, setting:
            - intersection_type
            - num_approaches
            - approach_angles
            - approach_lane_counts

        Args:
            topo: ScenarioTopology to classify (modified in place).
            lane_graph: Lane graph dictionary with 'lanes' key.
        """
        G = topo.connectivity_graph
        lanes = lane_graph["lanes"]

        if G is None or len(G.nodes) == 0:
            topo.intersection_type = "none"
            return

        # Step 1: Find entry and exit lanes
        entry_lanes = [
            n for n in G.nodes if G.in_degree(n) == 0 and n in lanes
        ]
        exit_lanes = [
            n for n in G.nodes if G.out_degree(n) == 0 and n in lanes
        ]

        # Step 2: Compute approach directions
        approach_directions: List[float] = []
        approach_points: List[np.ndarray] = []

        for lane_id in entry_lanes:
            if lane_id not in lanes:
                continue
            polyline = lanes[lane_id]
            if len(polyline) < 2:
                continue
            start_pt = polyline[0]
            vec = topo.centroid - start_pt
            angle = np.degrees(np.arctan2(vec[1], vec[0]))
            approach_directions.append(angle)
            approach_points.append(start_pt)

        for lane_id in exit_lanes:
            if lane_id not in lanes:
                continue
            polyline = lanes[lane_id]
            if len(polyline) < 2:
                continue
            end_pt = polyline[-1]
            vec = topo.centroid - end_pt
            angle = np.degrees(np.arctan2(vec[1], vec[0]))
            approach_directions.append(angle)
            approach_points.append(end_pt)

        if len(approach_directions) == 0:
            topo.intersection_type = "none"
            topo.num_approaches = 0
            return

        # Step 3: Cluster approach directions
        clusters = self._cluster_angles(
            approach_directions,
            approach_points,
            threshold_deg=self.angle_cluster_threshold,
        )

        topo.num_approaches = len(clusters)

        # Step 4: Compute inter-approach angles and absolute directions
        if len(clusters) >= 2:
            cluster_angles = sorted([c["mean_angle"] for c in clusters])

            # Store absolute direction of each approach (in [0, 360))
            topo.approach_directions = [
                float(a) % 360.0 for a in cluster_angles
            ]

            topo.approach_angles = []
            for i in range(len(cluster_angles)):
                next_i = (i + 1) % len(cluster_angles)
                diff = (cluster_angles[next_i] - cluster_angles[i]) % 360.0
                topo.approach_angles.append(float(diff))

            topo.approach_lane_counts = [c["count"] for c in clusters]
        elif len(clusters) == 1:
            topo.approach_directions = [
                float(clusters[0]["mean_angle"]) % 360.0
            ]

        # Step 5: Assign intersection type
        topo.intersection_type = self._classify_type(
            topo.num_approaches, topo.approach_angles
        )

    def _cluster_angles(
        self,
        angles: List[float],
        points: List[np.ndarray],
        threshold_deg: float = 30.0,
    ) -> List[Dict]:
        """Cluster approach directions into distinct approach arms.

        Uses angular distance to merge nearby directions into the same
        approach. Handles wrap-around at 360 degrees.

        Args:
            angles: Approach angles in degrees.
            points: Corresponding spatial points.
            threshold_deg: Maximum angular gap within a cluster.

        Returns:
            List of cluster dicts with keys 'mean_angle', 'count', 'points'.
        """
        if not angles:
            return []

        sorted_indices = np.argsort(angles)
        sorted_angles = [angles[i] for i in sorted_indices]
        sorted_points = [points[i] for i in sorted_indices]

        clusters = []
        current = {
            "angles": [sorted_angles[0]],
            "points": [sorted_points[0]],
        }

        for i in range(1, len(sorted_angles)):
            diff = self._angular_diff(
                sorted_angles[i], current["angles"][-1]
            )
            if diff < threshold_deg:
                current["angles"].append(sorted_angles[i])
                current["points"].append(sorted_points[i])
            else:
                clusters.append(current)
                current = {
                    "angles": [sorted_angles[i]],
                    "points": [sorted_points[i]],
                }
        clusters.append(current)

        # Check wrap-around merge
        if len(clusters) > 1:
            diff = self._angular_diff(
                clusters[0]["angles"][0], clusters[-1]["angles"][-1]
            )
            if diff < threshold_deg:
                clusters[0]["angles"] = (
                    clusters[-1]["angles"] + clusters[0]["angles"]
                )
                clusters[0]["points"] = (
                    clusters[-1]["points"] + clusters[0]["points"]
                )
                clusters.pop()

        # Compute cluster statistics
        result = []
        for c in clusters:
            mean_angle = self._circular_mean(c["angles"])
            result.append(
                {
                    "mean_angle": mean_angle,
                    "count": len(c["angles"]),
                    "points": c["points"],
                }
            )
        return result

    @staticmethod
    def _angular_diff(a: float, b: float) -> float:
        """Minimum angular difference between two angles in degrees.

        Delegates to :func:`waygraph.utils.geometry.angular_diff`.

        Args:
            a: First angle in degrees.
            b: Second angle in degrees.

        Returns:
            Minimum angular gap in [0, 180].
        """
        return _angular_diff_util(a, b)

    @staticmethod
    def _circular_mean(angles: List[float]) -> float:
        """Compute circular mean of angles in degrees.

        Delegates to :func:`waygraph.utils.geometry.circular_mean`.

        Args:
            angles: List of angles in degrees.

        Returns:
            Circular mean angle in degrees.
        """
        return _circular_mean_util(angles)

    @staticmethod
    def _classify_type(
        num_approaches: int, approach_angles: List[float]
    ) -> str:
        """Classify intersection type from approach count and angular layout.

        Args:
            num_approaches: Number of distinct approach arms.
            approach_angles: Angular gaps between consecutive approaches.

        Returns:
            One of: 'none', 'merge', 'Y', 'T', 'cross', 'multi', 'roundabout'.
        """
        if num_approaches <= 1:
            return "none"

        if num_approaches == 2:
            if approach_angles and max(approach_angles) > 150:
                return "none"  # Nearly straight -- not a real intersection
            return "merge"

        if num_approaches == 3:
            if approach_angles:
                min_angle = min(approach_angles)
                max_angle = max(approach_angles)
                if min_angle < 100 and max_angle > 140:
                    return "T"
                else:
                    return "Y"
            return "T"

        if num_approaches == 4:
            return "cross"

        if num_approaches >= 5:
            if approach_angles:
                angle_std = np.std(approach_angles)
                if angle_std < 15:
                    return "roundabout"
            return "multi"

        return "none"
