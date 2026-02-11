"""
Build Star Pattern Database from OSM
======================================

Extract star pattern fingerprints from an OpenStreetMap road network
to build a reference database for matching WOMD scenarios.

Example::

    from waygraph.osm import OSMNetwork, OSMStarDatabase

    net = OSMNetwork(city_name="San Francisco, CA")
    db = OSMStarDatabase()
    patterns = db.build_from_network(net)
    print(f"Extracted {len(patterns)} star patterns")

    # Or from a saved GraphML file
    patterns = db.build_from_graphml("sf_graph.graphml")
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

try:
    import osmnx as ox

    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False

from waygraph.fingerprint.star_pattern import ApproachArm, StarPattern


def _classify_osm_type(degree: int) -> str:
    """Classify intersection type from OSM node degree.

    Args:
        degree: Total degree (in + out) of the node.

    Returns:
        Intersection type string.
    """
    if degree <= 2:
        return "merge"
    elif degree <= 4:
        return "T"
    elif degree <= 6:
        return "cross"
    elif degree <= 8:
        return "cross"
    else:
        return "multi"


class OSMStarDatabase:
    """Build and manage a star pattern database from OSM data.

    Extracts star pattern fingerprints from OSM road network nodes,
    capturing each intersection's type, approach roads, and neighboring
    intersection properties.

    Args:
        min_degree: Minimum total degree for a node to be considered
            an intersection (in the directed graph, degree 6 typically
            means >= 3 actual road approaches).
    """

    def __init__(self, min_degree: int = 6):
        if not HAS_OSMNX:
            raise ImportError(
                "osmnx is required for OSMStarDatabase. "
                "Install with: pip install waygraph[osm]"
            )
        self.min_degree = min_degree

    def build_from_graphml(self, graphml_path: Union[str, Path]) -> List[StarPattern]:
        """Extract star patterns from a saved GraphML file.

        Args:
            graphml_path: Path to the GraphML file.

        Returns:
            List of StarPattern objects.
        """
        G = ox.load_graphml(str(graphml_path))
        G_proj = ox.project_graph(G)
        return self._extract_patterns(G, G_proj)

    def build_from_network(self, osm_network: "OSMNetwork") -> List[StarPattern]:
        """Extract star patterns from an OSMNetwork instance.

        Args:
            osm_network: An initialized OSMNetwork.

        Returns:
            List of StarPattern objects.
        """
        return self._extract_patterns(osm_network.graph, osm_network.graph_proj)

    def _extract_patterns(self, G_unprojected, G_proj) -> List[StarPattern]:
        """Extract star patterns from projected and unprojected graphs.

        Args:
            G_unprojected: Original lat/lon graph (for coordinates).
            G_proj: Projected graph in meters (for geometry).

        Returns:
            List of StarPattern objects with >= 3 arms.
        """
        patterns: List[StarPattern] = []

        for node in G_proj.nodes:
            total_deg = G_proj.in_degree(node) + G_proj.out_degree(node)
            if total_deg < self.min_degree:
                continue

            node_data = G_proj.nodes[node]
            hw_tag = str(node_data.get("highway", ""))

            star = StarPattern(
                id=str(node),
                center_type=_classify_osm_type(total_deg),
                center_approaches=total_deg // 2,
                center_has_signal=(hw_tag == "traffic_signals"),
                center_has_stop=(hw_tag == "stop"),
                center_has_crosswalk=(hw_tag == "crossing"),
            )

            # Get lat/lon from unprojected graph
            if node in G_unprojected.nodes:
                orig = G_unprojected.nodes[node]
                star.lat = orig.get("y", 0.0)
                star.lon = orig.get("x", 0.0)

            node_x = node_data.get("x", 0.0)
            node_y = node_data.get("y", 0.0)

            # Collect approach arms, clustered by direction
            seen_directions = {}

            for u, v, k, d in G_proj.edges(node, data=True, keys=True):
                v_data = G_proj.nodes.get(v, {})
                vx = v_data.get("x", node_x)
                vy = v_data.get("y", node_y)

                dx = vx - node_x
                dy = vy - node_y
                if abs(dx) < 0.1 and abs(dy) < 0.1:
                    continue

                angle = np.degrees(np.arctan2(dy, dx)) % 360
                length = float(d.get("length", 0.0))

                highway = str(d.get("highway", "residential"))
                if isinstance(d.get("highway"), list):
                    highway = str(d["highway"][0])

                lanes = d.get("lanes", 1)
                if isinstance(lanes, str):
                    try:
                        lanes = int(lanes)
                    except ValueError:
                        lanes = 1
                elif isinstance(lanes, list):
                    try:
                        lanes = int(lanes[0])
                    except (ValueError, IndexError):
                        lanes = 1

                v_deg = G_proj.in_degree(v) + G_proj.out_degree(v)
                v_hw = str(G_proj.nodes.get(v, {}).get("highway", ""))

                arm = ApproachArm(
                    angle_deg=angle,
                    road_length_m=length,
                    road_type=highway,
                    num_lanes=int(lanes),
                    neighbor_type=_classify_osm_type(v_deg),
                    neighbor_degree=v_deg,
                    neighbor_has_signal=(v_hw == "traffic_signals"),
                )

                # Cluster by direction (30-degree bins)
                direction_bin = int(angle / 30) % 12
                if (
                    direction_bin not in seen_directions
                    or length > seen_directions[direction_bin].road_length_m
                ):
                    seen_directions[direction_bin] = arm

            star.arms = list(seen_directions.values())
            star.center_approaches = len(star.arms)

            # Reclassify based on actual arm count
            if len(star.arms) <= 2:
                star.center_type = "merge"
            elif len(star.arms) == 3:
                star.center_type = "T"
            elif len(star.arms) == 4:
                star.center_type = "cross"
            elif len(star.arms) >= 5:
                star.center_type = "multi"

            if len(star.arms) >= 3:
                patterns.append(star)

        return patterns

    @staticmethod
    def save_patterns(
        patterns: List[StarPattern],
        filepath: Union[str, Path],
    ) -> None:
        """Save star patterns to a JSON file.

        Args:
            patterns: List of StarPattern objects.
            filepath: Output JSON file path.
        """
        data = [p.to_dict() for p in patterns]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_patterns(filepath: Union[str, Path]) -> List[StarPattern]:
        """Load star patterns from a JSON file.

        Args:
            filepath: Input JSON file path.

        Returns:
            List of StarPattern objects.
        """
        with open(filepath) as f:
            data = json.load(f)
        return [StarPattern.from_dict(d) for d in data]
