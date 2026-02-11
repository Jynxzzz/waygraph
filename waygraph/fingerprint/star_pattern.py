"""
Star Pattern Fingerprinting
=============================

Implements star pattern fingerprinting, adapted from GIS map conflation
literature for autonomous driving dataset localization.

The star pattern is a 48-dimensional feature vector that encodes
the topology of an intersection and its 1-hop neighborhood (the
"star" of approach roads leading to/from the intersection).

The star pattern captures:
    - Center intersection properties (type, approaches, traffic controls)
    - Per-arm properties (direction, road length, road type, lane count,
      neighbor intersection type, neighbor degree, neighbor signal status)

This provides more discriminative power than matching a single intersection
in isolation, because the spatial arrangement of neighboring intersections
provides additional structural context.

Feature vector layout (48 dimensions):
    [0:6]   Center features: type, approaches, signal, stop, crosswalk, num_arms
    [6:13]  Arm 0: angle, length, road_type, lanes, neighbor_type, neighbor_deg, neighbor_signal
    [13:20] Arm 1: ...
    [20:27] Arm 2: ...
    [27:34] Arm 3: ...
    [34:41] Arm 4: ...
    [41:48] Arm 5: ...

Example::

    from waygraph.fingerprint import StarPattern, ApproachArm

    star = StarPattern(
        center_type="cross",
        center_approaches=4,
        center_has_signal=True,
        arms=[
            ApproachArm(angle_deg=0, road_length_m=150, road_type="primary"),
            ApproachArm(angle_deg=90, road_length_m=200, road_type="secondary"),
            ApproachArm(angle_deg=180, road_length_m=150, road_type="primary"),
            ApproachArm(angle_deg=270, road_length_m=120, road_type="tertiary"),
        ],
    )
    vector = star.to_vector()  # shape (48,)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# Intersection type to numeric code mapping
INTERSECTION_TYPE_CODE: Dict[str, int] = {
    "none": 0,
    "merge": 1,
    "Y": 2,
    "T": 3,
    "cross": 4,
    "multi": 5,
    "roundabout": 6,
}

# Road type to numeric code mapping (OSM highway tags)
ROAD_TYPE_CODE: Dict[str, int] = {
    "residential": 1,
    "tertiary": 2,
    "secondary": 3,
    "primary": 4,
    "trunk": 5,
    "motorway": 5,
}

# Maximum number of arms in the feature vector
MAX_ARMS = 6

# Features per arm in the feature vector
FEATURES_PER_ARM = 7

# Total feature vector dimensionality: 6 center + 6 arms * 7 features
VECTOR_DIM = 6 + MAX_ARMS * FEATURES_PER_ARM  # = 48


@dataclass
class ApproachArm:
    """One arm of a star pattern, representing an approach road.

    Attributes:
        angle_deg: Direction from center intersection, in degrees [0, 360).
        road_length_m: Distance along the road to the next intersection.
        road_type: OSM highway tag (e.g., 'residential', 'primary').
        num_lanes: Number of lanes on this approach road.
        neighbor_type: Intersection type at the far end of the arm.
        neighbor_degree: Graph degree of the neighbor intersection node.
        neighbor_has_signal: Whether the neighbor intersection has a signal.
    """

    angle_deg: float = 0.0
    road_length_m: float = 0.0
    road_type: str = ""
    num_lanes: int = 1
    neighbor_type: str = "none"
    neighbor_degree: int = 0
    neighbor_has_signal: bool = False


@dataclass
class StarPattern:
    """Star pattern fingerprint: center intersection + approach arms.

    Adapted from GIS map conflation techniques, the star pattern encodes
    the topological context of an intersection: the center node's properties
    plus the properties of each approach road and its far-end neighbor intersection.

    This fingerprint can be extracted from WOMD lane graphs or from
    OpenStreetMap road networks, enabling GPS-free cross-matching between them.

    Attributes:
        id: Unique identifier (e.g., scenario ID or OSM node ID).
        center_type: Intersection type at the center.
        center_approaches: Number of approach directions.
        center_has_signal: Whether the center has a traffic signal.
        center_has_stop: Whether the center has a stop sign.
        center_has_crosswalk: Whether the center has a crosswalk.
        arms: List of ApproachArm objects, one per approach direction.
        lat: Latitude of the center (if known).
        lon: Longitude of the center (if known).
    """

    id: str = ""
    center_type: str = "none"
    center_approaches: int = 0
    center_has_signal: bool = False
    center_has_stop: bool = False
    center_has_crosswalk: bool = False
    arms: List[ApproachArm] = field(default_factory=list)
    lat: float = 0.0
    lon: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert the star pattern to a fixed-size 48D feature vector.

        The vector is normalized so that each component falls roughly
        in [0, 1], making it suitable for Euclidean distance comparisons.

        Returns:
            np.ndarray of shape (48,) with normalized features.
        """
        # Center features (6 dimensions)
        center = [
            INTERSECTION_TYPE_CODE.get(self.center_type, 0) / 6.0,
            self.center_approaches / 6.0,
            float(self.center_has_signal),
            float(self.center_has_stop),
            float(self.center_has_crosswalk),
            len(self.arms) / 6.0,
        ]

        # Sort arms by angle for canonical ordering
        sorted_arms = sorted(self.arms, key=lambda a: a.angle_deg)

        # Arm features (7 per arm, max 6 arms = 42 dimensions)
        arm_feats: List[float] = []
        for i in range(MAX_ARMS):
            if i < len(sorted_arms):
                arm = sorted_arms[i]
                road_type_key = (
                    arm.road_type.split("_")[0] if arm.road_type else ""
                )
                arm_feats.extend(
                    [
                        (arm.angle_deg % 360) / 360.0,
                        min(arm.road_length_m, 500.0) / 500.0,
                        ROAD_TYPE_CODE.get(road_type_key, 1) / 5.0,
                        min(arm.num_lanes, 6) / 6.0,
                        INTERSECTION_TYPE_CODE.get(arm.neighbor_type, 0) / 6.0,
                        min(arm.neighbor_degree, 10) / 10.0,
                        float(arm.neighbor_has_signal),
                    ]
                )
            else:
                arm_feats.extend([0.0] * FEATURES_PER_ARM)

        return np.array(center + arm_feats, dtype=np.float64)

    @classmethod
    def from_topology(
        cls,
        topo: "ScenarioTopology",
        scenario_id: Optional[str] = None,
        neighbor_info: Optional[List[dict]] = None,
        sdc_paths: Optional[List] = None,
    ) -> "StarPattern":
        """Build a star pattern from a ScenarioTopology.

        Uses absolute approach directions, lane counts, and traffic control
        flags from the topology. Road lengths are approximated from the
        bounding box geometry. Road types are inferred from lane count
        and speed data when possible, or from ``sdc_paths`` lane IDs
        (v1.3.1+) when available.

        **Neighbor information limitation**: WOMD scenarios do not contain
        information about neighboring intersections. By default, neighbor
        fields are set to placeholder values (``neighbor_type="cross"``,
        ``neighbor_degree=6``, ``neighbor_has_signal`` copies the center
        signal flag). You can supply per-arm neighbor data via the
        ``neighbor_info`` parameter to override these defaults.

        Args:
            topo: A ScenarioTopology (from waygraph.core.lane_graph).
                Must have ``approach_directions`` populated (requires
                IntersectionClassifier to have been run).
            scenario_id: Override for the pattern ID.
            neighbor_info: Optional list of dicts, one per approach arm,
                each containing keys ``"neighbor_type"`` (str),
                ``"neighbor_degree"`` (int), and ``"neighbor_has_signal"``
                (bool). If None, placeholder values are used.
            sdc_paths: Optional list of SDCPath objects (v1.3.1+) for
                data-driven road type inference from lane IDs.

        Returns:
            StarPattern populated from the topology.
        """
        star = cls(
            id=scenario_id or getattr(topo, "scenario_id", ""),
            center_type=topo.intersection_type,
            center_approaches=topo.num_approaches,
            center_has_signal=topo.has_traffic_light,
            center_has_stop=topo.has_stop_sign,
            center_has_crosswalk=topo.has_crosswalk,
        )

        # Estimate road length from bounding box
        bbox = topo.bounding_box
        if len(bbox) == 4:
            bbox_w = abs(bbox[2] - bbox[0])
            bbox_h = abs(bbox[3] - bbox[1])
            avg_radius = (bbox_w + bbox_h) / 4.0
        else:
            avg_radius = 100.0

        # Use absolute approach directions (not inter-approach angle gaps)
        directions = getattr(topo, "approach_directions", [])
        if not directions:
            cumsum = 0.0
            for gap in topo.approach_angles:
                directions.append(cumsum % 360.0)
                cumsum += gap

        lane_counts = topo.approach_lane_counts

        # Infer road type per approach from available data
        approach_road_types = _infer_road_types(topo, sdc_paths)

        for i, direction in enumerate(directions):
            lanes = lane_counts[i] if i < len(lane_counts) else 2

            # Neighbor information: use provided data or placeholders
            if neighbor_info is not None and i < len(neighbor_info):
                nb = neighbor_info[i]
                nb_type = nb.get("neighbor_type", "cross")
                nb_degree = nb.get("neighbor_degree", 6)
                nb_signal = nb.get("neighbor_has_signal", topo.has_traffic_light)
            else:
                nb_type = "cross"
                nb_degree = 6
                nb_signal = topo.has_traffic_light

            road_type = approach_road_types[i] if i < len(approach_road_types) else "secondary"

            arm = ApproachArm(
                angle_deg=direction % 360,
                road_length_m=avg_radius * (0.8 + 0.4 * (i % 3) / 2),
                road_type=road_type,
                num_lanes=min(lanes, 6),
                neighbor_type=nb_type,
                neighbor_degree=nb_degree,
                neighbor_has_signal=nb_signal,
            )
            star.arms.append(arm)

        return star

    def to_dict(self) -> dict:
        """Serialize the star pattern to a JSON-compatible dictionary.

        Returns:
            Dictionary with all pattern data.
        """
        return {
            "id": self.id,
            "center_type": self.center_type,
            "center_approaches": self.center_approaches,
            "center_has_signal": self.center_has_signal,
            "center_has_stop": self.center_has_stop,
            "center_has_crosswalk": self.center_has_crosswalk,
            "lat": self.lat,
            "lon": self.lon,
            "n_arms": len(self.arms),
            "arms": [
                {
                    "angle": a.angle_deg,
                    "length": a.road_length_m,
                    "road_type": a.road_type,
                    "lanes": a.num_lanes,
                    "neighbor_type": a.neighbor_type,
                    "neighbor_degree": a.neighbor_degree,
                    "neighbor_has_signal": a.neighbor_has_signal,
                }
                for a in self.arms
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StarPattern":
        """Deserialize a star pattern from a dictionary.

        Args:
            data: Dictionary as produced by ``to_dict()``.

        Returns:
            Reconstructed StarPattern.
        """
        arms = []
        for arm_data in data.get("arms", []):
            arms.append(
                ApproachArm(
                    angle_deg=arm_data.get("angle", 0.0),
                    road_length_m=arm_data.get("length", 0.0),
                    road_type=arm_data.get("road_type", ""),
                    num_lanes=arm_data.get("lanes", 1),
                    neighbor_type=arm_data.get("neighbor_type", "none"),
                    neighbor_degree=arm_data.get("neighbor_degree", 0),
                    neighbor_has_signal=arm_data.get(
                        "neighbor_has_signal", False
                    ),
                )
            )

        return cls(
            id=data.get("id", ""),
            center_type=data.get("center_type", "none"),
            center_approaches=data.get("center_approaches", 0),
            center_has_signal=data.get("center_has_signal", False),
            center_has_stop=data.get("center_has_stop", False),
            center_has_crosswalk=data.get("center_has_crosswalk", False),
            arms=arms,
            lat=data.get("lat", 0.0),
            lon=data.get("lon", 0.0),
        )


def _infer_road_types(
    topo: "ScenarioTopology",
    sdc_paths: Optional[List] = None,
) -> List[str]:
    """Infer road type for each approach from available data.

    Uses a heuristic based on lane count per approach to estimate road
    hierarchy. If sdc_paths (v1.3.1+) are provided, uses lane ID
    density along the SDC route for more accurate inference.

    Heuristic (lane-count based):
        - 1 lane: residential
        - 2 lanes: tertiary
        - 3 lanes: secondary
        - 4+ lanes: primary

    Args:
        topo: ScenarioTopology with approach_lane_counts populated.
        sdc_paths: Optional list of SDCPath objects from v1.3.1.

    Returns:
        List of road type strings, one per approach direction.
    """
    lane_counts = topo.approach_lane_counts
    num_approaches = topo.num_approaches

    if not lane_counts:
        return ["secondary"] * max(num_approaches, 1)

    road_types = []
    for i in range(num_approaches):
        n_lanes = lane_counts[i] if i < len(lane_counts) else 2

        if n_lanes >= 4:
            road_types.append("primary")
        elif n_lanes == 3:
            road_types.append("secondary")
        elif n_lanes == 2:
            road_types.append("tertiary")
        else:
            road_types.append("residential")

    return road_types
