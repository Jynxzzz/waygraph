"""
OSM Road Network Download and Processing
==========================================

Download and process OpenStreetMap road networks using osmnx. Provides
abstractions for loading city-scale networks, extracting intersections
with topological properties, and getting local subgraphs.

Requires: ``pip install osmnx`` (included in ``waygraph[osm]``)

Example::

    from waygraph.osm import OSMNetwork

    # Load by city name
    net = OSMNetwork(city_name="San Francisco, California, USA")

    # Get all intersections
    intersections = net.get_intersections(min_degree=3)
    print(f"Found {len(intersections)} intersections")

    # Get local subgraph around a point
    subgraph = net.get_subgraph(37.7749, -122.4194, radius_m=500)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

try:
    import osmnx as ox

    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False

try:
    from shapely.geometry import LineString
except ImportError:
    LineString = None


@dataclass
class OSMIntersection:
    """An intersection extracted from an OSM road network.

    Attributes:
        node_id: OSM node ID.
        lat: Latitude (WGS84).
        lon: Longitude (WGS84).
        x: Projected x coordinate (meters, UTM).
        y: Projected y coordinate (meters, UTM).
        num_approaches: Number of approach road directions.
        approach_angles: Angular gaps between consecutive approaches.
        approach_road_names: Road names for each approach.
        approach_lane_counts: Lane counts for each approach.
        approach_road_types: OSM highway tags for each approach.
        has_traffic_signal: Whether the node has a traffic signal.
        has_stop_sign: Whether the node has a stop sign.
        has_crosswalk: Whether the node has a crossing.
        intersection_type: Classified type (T, cross, Y, multi, etc.).
    """

    node_id: int = 0
    lat: float = 0.0
    lon: float = 0.0
    x: float = 0.0
    y: float = 0.0
    num_approaches: int = 0
    approach_angles: List[float] = field(default_factory=list)
    approach_road_names: List[str] = field(default_factory=list)
    approach_lane_counts: List[int] = field(default_factory=list)
    approach_road_types: List[str] = field(default_factory=list)
    has_traffic_signal: bool = False
    has_stop_sign: bool = False
    has_crosswalk: bool = False
    intersection_type: str = "none"

    def to_feature_vector(self) -> np.ndarray:
        """Convert to a 10D feature vector for distance-based comparison.

        Returns:
            np.ndarray of shape (10,) with normalized features.
        """
        features = [
            self.num_approaches / 6.0,
            float(self.has_traffic_signal),
            float(self.has_stop_sign),
            float(self.has_crosswalk),
        ]
        angles = list(self.approach_angles) + [0.0] * 6
        features.extend([a / 180.0 for a in angles[:6]])
        return np.array(features, dtype=np.float64)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "node_id": int(self.node_id),
            "lat": float(self.lat),
            "lon": float(self.lon),
            "x": float(self.x),
            "y": float(self.y),
            "num_approaches": int(self.num_approaches),
            "approach_angles": [float(a) for a in self.approach_angles],
            "approach_road_names": list(self.approach_road_names),
            "approach_lane_counts": [int(c) for c in self.approach_lane_counts],
            "approach_road_types": list(self.approach_road_types),
            "has_traffic_signal": bool(self.has_traffic_signal),
            "has_stop_sign": bool(self.has_stop_sign),
            "has_crosswalk": bool(self.has_crosswalk),
            "intersection_type": str(self.intersection_type),
        }


@dataclass
class OSMRoadSegment:
    """A road segment between two intersections.

    Attributes:
        edge_id: (u, v, key) tuple from the OSM graph.
        start_node: OSM node ID of the start.
        end_node: OSM node ID of the end.
        name: Road name.
        road_type: OSM highway tag.
        num_lanes: Number of lanes.
        speed_limit_kmh: Posted speed limit in km/h.
        length_m: Segment length in meters.
        is_oneway: Whether the segment is one-way.
    """

    edge_id: Tuple[int, int, int] = (0, 0, 0)
    start_node: int = 0
    end_node: int = 0
    name: str = ""
    road_type: str = ""
    num_lanes: int = 1
    speed_limit_kmh: float = 0.0
    length_m: float = 0.0
    is_oneway: bool = False


class OSMNetwork:
    """Load and process OpenStreetMap road networks.

    Wraps osmnx to provide a clean API for downloading road networks,
    extracting intersections, and querying subgraphs.

    Args:
        city_name: City name for geocoded download (e.g., "San Francisco, CA").
        bbox: Bounding box as (north, south, east, west).
        network_type: OSM network type ("drive", "walk", "bike", "all").
        graph: Pre-loaded osmnx graph (skips download).

    Raises:
        ImportError: If osmnx is not installed.
        ValueError: If no source (city_name, bbox, or graph) is provided.

    Example::

        net = OSMNetwork(city_name="Phoenix, Arizona, USA")
        intersections = net.get_intersections(min_degree=6)
        segments = net.get_road_segments()
    """

    def __init__(
        self,
        city_name: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        network_type: str = "drive",
        graph: Optional[nx.MultiDiGraph] = None,
    ):
        if not HAS_OSMNX:
            raise ImportError(
                "osmnx is required for OSMNetwork. "
                "Install with: pip install waygraph[osm]"
            )

        self.city_name = city_name
        self.network_type = network_type

        if graph is not None:
            self.graph = graph
        elif city_name is not None:
            self.graph = ox.graph_from_place(
                city_name, network_type=network_type
            )
        elif bbox is not None:
            north, south, east, west = bbox
            self.graph = ox.graph_from_bbox(
                bbox=(west, south, east, north),
                network_type=network_type,
            )
        else:
            raise ValueError("Must provide city_name, bbox, or graph.")

        self.graph_proj = ox.project_graph(self.graph)
        self._intersections_cache: Dict[int, List[OSMIntersection]] = {}
        self._road_segments: Optional[List[OSMRoadSegment]] = None

    def get_intersections(
        self, min_degree: int = 3
    ) -> List[OSMIntersection]:
        """Extract all intersections from the road network.

        An intersection is a node with total degree >= min_degree.
        Results are cached per ``min_degree`` value so that calling with
        different thresholds returns correct results.

        Args:
            min_degree: Minimum total degree (in + out) to qualify.

        Returns:
            List of OSMIntersection objects.
        """
        if min_degree in self._intersections_cache:
            return self._intersections_cache[min_degree]

        G = self.graph_proj
        intersections: List[OSMIntersection] = []

        for node_id, node_data in G.nodes(data=True):
            total_degree = G.in_degree(node_id) + G.out_degree(node_id)
            if total_degree < min_degree:
                continue

            intersection = OSMIntersection(
                node_id=node_id,
                x=node_data.get("x", 0.0),
                y=node_data.get("y", 0.0),
            )

            if node_id in self.graph.nodes:
                orig_data = self.graph.nodes[node_id]
                intersection.lat = orig_data.get("y", 0.0)
                intersection.lon = orig_data.get("x", 0.0)

            highway_tag = node_data.get("highway", "")
            if highway_tag == "traffic_signals":
                intersection.has_traffic_signal = True
            elif highway_tag == "stop":
                intersection.has_stop_sign = True
            elif highway_tag == "crossing":
                intersection.has_crosswalk = True

            self._compute_approaches(intersection, G)
            intersections.append(intersection)

        self._intersections_cache[min_degree] = intersections
        return intersections

    def get_road_segments(self) -> List[OSMRoadSegment]:
        """Extract all road segments from the network.

        Returns:
            List of OSMRoadSegment objects.
        """
        if self._road_segments is not None:
            return self._road_segments

        G = self.graph_proj
        segments: List[OSMRoadSegment] = []

        for u, v, key, data in G.edges(data=True, keys=True):
            name = data.get("name", "")
            if isinstance(name, list):
                name = name[0] if name else ""

            highway = data.get("highway", "unclassified")
            if isinstance(highway, list):
                highway = highway[0] if highway else "unclassified"

            lanes = data.get("lanes", 1)
            if isinstance(lanes, str):
                try:
                    lanes = int(lanes)
                except ValueError:
                    lanes = 1
            elif isinstance(lanes, list):
                lanes = int(lanes[0]) if lanes else 1

            speed = data.get("maxspeed", 0)
            if isinstance(speed, str):
                try:
                    speed = float(
                        speed.replace(" mph", "").replace(" km/h", "")
                    )
                except ValueError:
                    speed = 0.0
            elif isinstance(speed, list):
                try:
                    speed = float(
                        str(speed[0])
                        .replace(" mph", "")
                        .replace(" km/h", "")
                    )
                except (ValueError, IndexError):
                    speed = 0.0

            segment = OSMRoadSegment(
                edge_id=(u, v, key),
                start_node=u,
                end_node=v,
                name=str(name),
                road_type=str(highway),
                num_lanes=int(lanes),
                speed_limit_kmh=float(speed),
                length_m=float(data.get("length", 0.0)),
                is_oneway=bool(data.get("oneway", False)),
            )
            segments.append(segment)

        self._road_segments = segments
        return segments

    def get_subgraph(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float = 200.0,
    ) -> nx.MultiDiGraph:
        """Get a local subgraph around a geographic point.

        Args:
            center_lat: Latitude of center point.
            center_lon: Longitude of center point.
            radius_m: Radius in meters.

        Returns:
            Projected subgraph (coordinates in meters).
        """
        center_node = ox.nearest_nodes(self.graph, center_lon, center_lat)
        return nx.ego_graph(
            self.graph_proj,
            center_node,
            radius=radius_m,
            distance="length",
        )

    def save_graph(self, filepath: str) -> None:
        """Save the graph to a GraphML file.

        Args:
            filepath: Output file path.
        """
        ox.save_graphml(self.graph, filepath)

    @classmethod
    def load_graph(cls, filepath: str) -> "OSMNetwork":
        """Load a network from a saved GraphML file.

        Args:
            filepath: Path to the GraphML file.

        Returns:
            OSMNetwork instance.
        """
        graph = ox.load_graphml(filepath)
        return cls(graph=graph)

    def _compute_approaches(
        self, intersection: OSMIntersection, G: nx.MultiDiGraph
    ) -> None:
        """Compute approach directions and properties for an intersection."""
        node_id = intersection.node_id
        node_x = intersection.x
        node_y = intersection.y

        approach_angles: List[float] = []
        approach_names: List[str] = []
        approach_types: List[str] = []
        approach_lane_counts: List[int] = []

        # Process outgoing and incoming edges
        for edges in [
            G.edges(node_id, data=True, keys=True),
            G.in_edges(node_id, data=True, keys=True),
        ]:
            for edge_tuple in edges:
                if len(edge_tuple) == 4:
                    u, v, key, edge_data = edge_tuple
                    neighbor = v if u == node_id else u
                else:
                    continue

                neighbor_data = G.nodes.get(neighbor, {})
                nx_ = neighbor_data.get("x", node_x)
                ny_ = neighbor_data.get("y", node_y)

                dx = nx_ - node_x
                dy = ny_ - node_y
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    continue

                angle = np.degrees(np.arctan2(dy, dx))
                approach_angles.append(angle)

                name = edge_data.get("name", "")
                if isinstance(name, list):
                    name = name[0] if name else ""
                approach_names.append(str(name))

                highway = edge_data.get("highway", "unclassified")
                if isinstance(highway, list):
                    highway = highway[0] if highway else "unclassified"
                approach_types.append(str(highway))

                lanes = edge_data.get("lanes", 1)
                if isinstance(lanes, (str, list)):
                    try:
                        lanes = int(lanes) if isinstance(lanes, str) else int(lanes[0])
                    except (ValueError, IndexError):
                        lanes = 1
                approach_lane_counts.append(int(lanes))

        if approach_angles:
            clusters = self._cluster_angles(approach_angles)
            intersection.num_approaches = len(clusters)
            n = len(clusters)
            intersection.approach_angles = (
                [360.0 / n] * n if n >= 2 else []
            )
            intersection.approach_road_names = [
                approach_names[c[0]] if c else "" for c in clusters
            ]
            intersection.approach_road_types = [
                approach_types[c[0]] if c else "" for c in clusters
            ]
            intersection.approach_lane_counts = [
                max((approach_lane_counts[i] for i in c), default=1)
                for c in clusters
            ]

        intersection.intersection_type = self._classify_type(
            intersection.num_approaches
        )

    @staticmethod
    def _cluster_angles(
        angles: List[float], threshold_deg: float = 30.0
    ) -> List[List[int]]:
        """Cluster approach angles into distinct directions."""
        if not angles:
            return []

        sorted_indices = list(np.argsort(angles))
        sorted_angles = [angles[i] for i in sorted_indices]

        clusters: List[List[int]] = [[sorted_indices[0]]]

        for i in range(1, len(sorted_angles)):
            diff = abs(sorted_angles[i] - sorted_angles[i - 1])
            diff = min(diff, 360.0 - diff)
            if diff < threshold_deg:
                clusters[-1].append(sorted_indices[i])
            else:
                clusters.append([sorted_indices[i]])

        # Wrap-around check
        if len(clusters) > 1:
            diff = abs(sorted_angles[0] - sorted_angles[-1])
            diff = min(diff, 360.0 - diff)
            if diff < threshold_deg:
                clusters[0].extend(clusters[-1])
                clusters.pop()

        return clusters

    @staticmethod
    def _classify_type(num_approaches: int) -> str:
        """Classify intersection type from approach count."""
        if num_approaches <= 1:
            return "none"
        elif num_approaches == 2:
            return "merge"
        elif num_approaches == 3:
            return "T"
        elif num_approaches == 4:
            return "cross"
        elif num_approaches >= 5:
            return "multi"
        return "none"
