"""
WayGraph: Structural Analysis Toolkit for Autonomous Driving Datasets
=====================================================================

WayGraph provides tools for structural analysis of road networks in autonomous
driving datasets, starting with the Waymo Open Motion Dataset (WOMD).

Key capabilities:
    - Intersection detection and classification from lane graphs
    - Star pattern fingerprinting (adapted from GIS map conflation) for topological matching
    - Traffic parameter extraction (turning ratios, speed distributions, gap acceptance)
    - OpenStreetMap integration and cross-referencing
    - Publication-quality visualization

Quick start::

    from waygraph.core import TopologyExtractor
    from waygraph.fingerprint import StarPattern

    # Extract topology from a WOMD scenario
    extractor = TopologyExtractor()
    topo = extractor.extract_topology("scenario.pkl")
    print(topo.intersection_type, topo.num_approaches)

    # Build a star pattern fingerprint
    star = StarPattern.from_topology(topo)
    vector = star.to_vector()  # 48D feature vector
"""

__version__ = "0.1.0"
__author__ = "Network Dreamer Authors"

from waygraph.core.scenario import ScenarioLoader
from waygraph.core.lane_graph import LaneGraph
from waygraph.core.intersection import IntersectionClassifier
from waygraph.fingerprint.star_pattern import StarPattern, ApproachArm
from waygraph.fingerprint.matching import StarPatternMatcher

__all__ = [
    "ScenarioLoader",
    "LaneGraph",
    "IntersectionClassifier",
    "StarPattern",
    "ApproachArm",
    "StarPatternMatcher",
]
