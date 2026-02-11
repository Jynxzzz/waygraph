"""
Star pattern fingerprinting and topological matching.

The fingerprint module provides the 48-dimensional star pattern feature
vector that encodes the topology of an intersection and its 1-hop
neighborhood. This fingerprint enables matching WOMD scenarios to
real-world OSM intersections with high accuracy.
"""

from waygraph.fingerprint.star_pattern import StarPattern, ApproachArm
from waygraph.fingerprint.matching import StarPatternMatcher, star_distance

__all__ = [
    "StarPattern",
    "ApproachArm",
    "StarPatternMatcher",
    "star_distance",
]
