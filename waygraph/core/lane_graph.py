"""
Lane Graph Extraction and Representation
==========================================

Provides the LaneGraph builder and the ScenarioTopology dataclass that
captures the full topological fingerprint of a WOMD scenario.

The topology fingerprint includes:
    - Lane polylines, curvatures, and lengths
    - Connectivity graph (successor/predecessor/lateral relationships)
    - Geometric properties (centroid, bounding box, area, dominant direction)
    - Graph-level statistics (degree sequence, branching/merge factors)

Example::

    from waygraph.core.lane_graph import LaneGraph, ScenarioTopology

    builder = LaneGraph()
    topo = builder.build_topology(lane_graph_dict, scenario_id="s001")
    print(f"Lanes: {topo.num_lanes}, Area: {topo.area:.0f} m^2")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial import ConvexHull


@dataclass
class ScenarioTopology:
    """Topological fingerprint of a WOMD scenario.

    This dataclass captures all the topological and geometric properties
    needed to match a scenario to an OSM road network location, classify
    its intersection type, and extract traffic parameters.

    Attributes:
        scenario_id: Unique identifier for the scenario.
        num_lanes: Total number of lane polylines.
        intersection_type: Classified type ('T', 'cross', 'Y', 'roundabout',
            'merge', 'multi', 'none').
        num_approaches: Number of distinct approach directions.
        approach_angles: Angular gaps between consecutive approaches (degrees).
        approach_directions: Absolute direction angle of each approach (degrees,
            [0, 360)). One value per approach, in the same order as
            approach_lane_counts. Computed from the circular mean of the
            terminal-lane bearings that belong to each approach cluster.
        approach_lane_counts: Number of lanes per approach direction.
        lane_curvatures: Per-lane curvature profiles.
        lane_lengths: Per-lane total lengths in meters.
        mean_lane_curvature: Mean absolute curvature across all lanes.
        has_traffic_light: Whether traffic lights are present.
        num_traffic_lights: Count of traffic lights.
        has_stop_sign: Whether stop signs are present.
        num_stop_signs: Count of stop signs.
        has_crosswalk: Whether crosswalks are present.
        num_crosswalks: Count of crosswalks.
        connectivity_graph: NetworkX DiGraph of lane connectivity.
        num_connected_components: Number of connected components.
        lane_polylines: Mapping of lane ID to (N, 2) numpy arrays.
        centroid: Geometric center of all lane points.
        bounding_box: (xmin, ymin, xmax, ymax) of all lane points.
        rotation_angle: Dominant road direction in degrees [0, 180).
        area: Convex hull area of all lane points in square meters.
        degree_sequence: Sorted node degree sequence of the connectivity graph.
        branching_factor: Average out-degree of nodes with successors.
        merge_factor: Average in-degree of nodes with predecessors.
        suc_pairs: Successor connectivity mapping.
        pre_pairs: Predecessor connectivity mapping.
        left_pairs: Left-neighbor lateral connectivity mapping.
        right_pairs: Right-neighbor lateral connectivity mapping.
    """

    # Scenario identification
    scenario_id: str = ""

    # Lane statistics
    num_lanes: int = 0
    lane_types: List[int] = field(default_factory=list)

    # Intersection classification
    intersection_type: str = "none"
    num_approaches: int = 0
    approach_angles: List[float] = field(default_factory=list)
    approach_directions: List[float] = field(default_factory=list)
    approach_lane_counts: List[int] = field(default_factory=list)

    # Lane geometry
    lane_curvatures: Dict[int, List[float]] = field(default_factory=dict)
    lane_lengths: Dict[int, float] = field(default_factory=dict)
    mean_lane_curvature: float = 0.0

    # Traffic control
    has_traffic_light: bool = False
    num_traffic_lights: int = 0
    has_stop_sign: bool = False
    num_stop_signs: int = 0
    has_crosswalk: bool = False
    num_crosswalks: int = 0

    # Graph structure
    connectivity_graph: Optional[nx.DiGraph] = None
    num_connected_components: int = 0

    # Geometric properties
    lane_polylines: Dict[int, np.ndarray] = field(default_factory=dict)
    centroid: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    bounding_box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    rotation_angle: float = 0.0
    area: float = 0.0

    # Derived topology features
    degree_sequence: List[int] = field(default_factory=list)
    branching_factor: float = 0.0
    merge_factor: float = 0.0

    # Raw connectivity for serialization
    suc_pairs: Dict[int, List[int]] = field(default_factory=dict)
    pre_pairs: Dict[int, List[int]] = field(default_factory=dict)
    left_pairs: Dict[int, List[int]] = field(default_factory=dict)
    right_pairs: Dict[int, List[int]] = field(default_factory=dict)

    def to_feature_vector(self) -> np.ndarray:
        """Convert topology to a fixed-size numeric feature vector.

        Creates a 20-dimensional vector suitable for distance-based
        comparison between topologies. Features are normalized to
        approximately [0, 1] range.

        Returns:
            np.ndarray of shape (20,) with normalized features.
        """
        features = [
            self.num_lanes / 200.0,
            self.num_approaches / 6.0,
            self.mean_lane_curvature,
            float(self.has_traffic_light),
            float(self.has_stop_sign),
            float(self.has_crosswalk),
            self.num_traffic_lights / 20.0,
            self.num_stop_signs / 10.0,
            self.num_crosswalks / 15.0,
            self.num_connected_components / 10.0,
            self.branching_factor / 3.0,
            self.merge_factor / 3.0,
            self.area / 100000.0,
            len(self.degree_sequence) / 200.0,
        ]
        # Pad approach angles to fixed size (max 6 approaches)
        angles = list(self.approach_angles) + [0.0] * 6
        features.extend([a / 180.0 for a in angles[:6]])

        return np.array(features, dtype=np.float64)

    def get_intersection_type_code(self) -> int:
        """Return numeric code for the intersection type.

        Returns:
            Integer code: 0=none, 1=merge, 2=Y, 3=T, 4=cross, 5=multi,
            6=roundabout.
        """
        type_map = {
            "none": 0,
            "merge": 1,
            "Y": 2,
            "T": 3,
            "cross": 4,
            "multi": 5,
            "roundabout": 6,
        }
        return type_map.get(self.intersection_type, 0)


class LaneGraph:
    """Build lane connectivity graphs and compute topological features.

    Constructs a directed graph from lane successor/predecessor relationships
    and computes geometric properties (curvature, length, centroid, area)
    for each lane polyline.

    Args:
        curvature_smoothing: Window size for smoothing curvature profiles.
        min_lane_points: Minimum number of points for a valid lane.

    Example::

        builder = LaneGraph()
        topo = builder.build_topology(lane_graph_dict, "scenario_001")
        print(f"Connected components: {topo.num_connected_components}")
    """

    def __init__(self, curvature_smoothing: int = 5, min_lane_points: int = 3):
        self.curvature_smoothing = curvature_smoothing
        self.min_lane_points = min_lane_points

    def build_topology(
        self, lane_graph: dict, scenario_id: str = ""
    ) -> ScenarioTopology:
        """Build a ScenarioTopology from a lane graph dictionary.

        Extracts lane polylines, builds the connectivity graph, and computes
        all geometric features. Does NOT classify the intersection or extract
        traffic controls (those are handled by other modules).

        Args:
            lane_graph: Dictionary with 'lanes', 'suc_pairs', 'pre_pairs',
                and optionally 'left_pairs', 'right_pairs'.
            scenario_id: Identifier for this scenario.

        Returns:
            ScenarioTopology with lane graph and geometric features populated.
        """
        topo = ScenarioTopology(scenario_id=scenario_id)
        lanes = lane_graph["lanes"]

        # Basic lane info
        topo.num_lanes = len(lanes)
        topo.lane_polylines = {k: v.copy() for k, v in lanes.items()}

        # Connectivity pairs
        topo.suc_pairs = {
            k: list(v) for k, v in lane_graph.get("suc_pairs", {}).items()
        }
        topo.pre_pairs = {
            k: list(v) for k, v in lane_graph.get("pre_pairs", {}).items()
        }
        topo.left_pairs = {
            k: list(v) for k, v in lane_graph.get("left_pairs", {}).items()
        }
        topo.right_pairs = {
            k: list(v) for k, v in lane_graph.get("right_pairs", {}).items()
        }

        # Build directed connectivity graph
        topo.connectivity_graph = self._build_connectivity_graph(lane_graph)

        # Geometric properties
        self._compute_geometry(topo, lanes)

        # Per-lane curvature and length
        self._compute_lane_features(topo, lanes)

        # Graph-level statistics
        self.compute_graph_stats(topo)

        return topo

    def compute_graph_stats(self, topo: ScenarioTopology) -> None:
        """Compute graph-level statistics from the connectivity graph.

        Populates degree_sequence, branching_factor, merge_factor, and
        num_connected_components on the topology object.

        Args:
            topo: ScenarioTopology with a populated connectivity_graph.
        """
        G = topo.connectivity_graph
        if G is None or len(G.nodes) == 0:
            return

        # Connected components (undirected)
        G_undirected = G.to_undirected()
        topo.num_connected_components = nx.number_connected_components(
            G_undirected
        )

        # Degree sequence
        degrees = [G.in_degree(n) + G.out_degree(n) for n in G.nodes]
        topo.degree_sequence = sorted(degrees, reverse=True)

        # Branching factor
        out_degrees = [
            G.out_degree(n) for n in G.nodes if G.out_degree(n) > 0
        ]
        topo.branching_factor = (
            float(np.mean(out_degrees)) if out_degrees else 0.0
        )

        # Merge factor
        in_degrees = [G.in_degree(n) for n in G.nodes if G.in_degree(n) > 0]
        topo.merge_factor = float(np.mean(in_degrees)) if in_degrees else 0.0

    def _build_connectivity_graph(self, lane_graph: dict) -> nx.DiGraph:
        """Build a directed graph from lane connectivity pairs.

        Uses successor/predecessor pairs as directed edges. Lateral
        relationships (left/right neighbors) are stored as node attributes.

        Args:
            lane_graph: Dictionary with lane data and connectivity pairs.

        Returns:
            NetworkX DiGraph representing lane connectivity.
        """
        G = nx.DiGraph()

        lanes = lane_graph["lanes"]
        suc_pairs = lane_graph.get("suc_pairs", {})
        pre_pairs = lane_graph.get("pre_pairs", {})
        left_pairs = lane_graph.get("left_pairs", {})
        right_pairs = lane_graph.get("right_pairs", {})

        # Collect all lane IDs
        all_ids = set(lanes.keys())
        all_ids.update(suc_pairs.keys())
        all_ids.update(pre_pairs.keys())

        for lane_id in all_ids:
            G.add_node(lane_id)
            if lane_id in lanes:
                polyline = lanes[lane_id]
                G.nodes[lane_id]["start"] = polyline[0].copy()
                G.nodes[lane_id]["end"] = polyline[-1].copy()
                G.nodes[lane_id]["length"] = self._polyline_length(polyline)

        # Successor edges
        for lane_id, sucs in suc_pairs.items():
            for suc in sucs:
                G.add_edge(lane_id, suc, type="suc")

        # Predecessor edges
        for lane_id, preds in pre_pairs.items():
            for pred in preds:
                G.add_edge(pred, lane_id, type="pre")

        # Lateral relationships as node attributes
        for lane_id, lefts in left_pairs.items():
            if lane_id in G.nodes:
                G.nodes[lane_id]["left_neighbors"] = lefts
        for lane_id, rights in right_pairs.items():
            if lane_id in G.nodes:
                G.nodes[lane_id]["right_neighbors"] = rights

        return G

    def _compute_geometry(
        self, topo: ScenarioTopology, lanes: Dict[int, np.ndarray]
    ) -> None:
        """Compute geometric properties: centroid, bounding box, area, direction.

        Args:
            topo: ScenarioTopology to populate.
            lanes: Mapping of lane ID to polyline arrays.
        """
        if not lanes:
            return

        all_points = np.vstack(list(lanes.values()))

        # Centroid
        topo.centroid = all_points.mean(axis=0)

        # Bounding box
        xmin, ymin = all_points.min(axis=0)
        xmax, ymax = all_points.max(axis=0)
        topo.bounding_box = (float(xmin), float(ymin), float(xmax), float(ymax))

        # Area via convex hull
        if len(all_points) >= 3:
            try:
                hull = ConvexHull(all_points)
                topo.area = float(hull.volume)  # In 2D, volume = area
            except Exception:
                topo.area = float((xmax - xmin) * (ymax - ymin))
        else:
            topo.area = float((xmax - xmin) * (ymax - ymin))

        # Dominant road direction
        topo.rotation_angle = self._compute_dominant_direction(lanes)

    def _compute_dominant_direction(
        self, lanes: Dict[int, np.ndarray]
    ) -> float:
        """Compute dominant road direction as a weighted circular mean.

        Uses the start-to-end vector of each lane, weighted by lane length.
        Returns angle in degrees in [0, 180) since direction is ambiguous.

        Args:
            lanes: Mapping of lane ID to polyline arrays.

        Returns:
            Dominant direction angle in degrees.
        """
        angle_weights = []
        for lane_id, polyline in lanes.items():
            if len(polyline) < 2:
                continue
            vec = polyline[-1] - polyline[0]
            length = np.linalg.norm(vec)
            if length < 1e-6:
                continue
            angle = np.degrees(np.arctan2(vec[1], vec[0])) % 180.0
            angle_weights.append((angle, length))

        if not angle_weights:
            return 0.0

        angles = np.array([a for a, _ in angle_weights])
        weights = np.array([w for _, w in angle_weights])
        weights /= weights.sum()

        # Circular mean on half-circle
        complex_vals = np.exp(2j * np.radians(angles))
        mean_complex = np.sum(weights * complex_vals)
        mean_angle = np.degrees(np.angle(mean_complex)) / 2.0

        return float(mean_angle % 180.0)

    def _compute_lane_features(
        self, topo: ScenarioTopology, lanes: Dict[int, np.ndarray]
    ) -> None:
        """Compute per-lane curvature profiles and lengths.

        Args:
            topo: ScenarioTopology to populate.
            lanes: Mapping of lane ID to polyline arrays.
        """
        curvatures_all = []

        for lane_id, polyline in lanes.items():
            if len(polyline) < self.min_lane_points:
                topo.lane_curvatures[lane_id] = []
                topo.lane_lengths[lane_id] = 0.0
                continue

            length = self._polyline_length(polyline)
            topo.lane_lengths[lane_id] = float(length)

            curvature = self._compute_curvature(polyline)
            topo.lane_curvatures[lane_id] = curvature.tolist()

            if len(curvature) > 0:
                curvatures_all.append(np.mean(np.abs(curvature)))

        if curvatures_all:
            topo.mean_lane_curvature = float(np.mean(curvatures_all))

    def _compute_curvature(self, polyline: np.ndarray) -> np.ndarray:
        """Compute discrete curvature along a polyline.

        Uses the formula: kappa = |dx * d2y - dy * d2x| / (dx^2 + dy^2)^1.5

        Args:
            polyline: (N, 2) array of points.

        Returns:
            Array of curvature values (length N-2).
        """
        if len(polyline) < 3:
            return np.array([])

        dx = np.diff(polyline[:, 0])
        dy = np.diff(polyline[:, 1])
        d2x = np.diff(dx)
        d2y = np.diff(dy)

        dx_mid = (dx[:-1] + dx[1:]) / 2.0
        dy_mid = (dy[:-1] + dy[1:]) / 2.0

        denom = (dx_mid**2 + dy_mid**2) ** 1.5
        denom = np.maximum(denom, 1e-10)
        kappa = np.abs(dx_mid * d2y - dy_mid * d2x) / denom

        if len(kappa) >= self.curvature_smoothing:
            kappa = uniform_filter1d(kappa, self.curvature_smoothing)

        return kappa

    @staticmethod
    def _polyline_length(polyline: np.ndarray) -> float:
        """Compute total arc length of a polyline.

        Args:
            polyline: (N, 2) array of points.

        Returns:
            Total length in the same units as the input coordinates.
        """
        if len(polyline) < 2:
            return 0.0
        diffs = np.diff(polyline, axis=0)
        return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))
