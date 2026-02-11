"""
OpenStreetMap integration for WayGraph.

Provides tools for downloading OSM road networks, extracting intersection
properties, and building star pattern databases for matching.

Requires the ``osm`` extra: ``pip install waygraph[osm]``
"""

from waygraph.osm.download import OSMNetwork, OSMIntersection, OSMRoadSegment
from waygraph.osm.star_db import OSMStarDatabase

__all__ = [
    "OSMNetwork",
    "OSMIntersection",
    "OSMRoadSegment",
    "OSMStarDatabase",
]
