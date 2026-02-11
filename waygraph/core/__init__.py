"""
Core modules for WOMD scenario loading, lane graph extraction, and intersection analysis.

Supports both WOMD v1.1.0 (preprocessed .pkl files) and v1.3.1+ (TFRecord
with sdc_paths extraction).
"""

from waygraph.core.scenario import ScenarioLoader, SDCPath, ScenarioMetadata
from waygraph.core.lane_graph import LaneGraph, ScenarioTopology
from waygraph.core.intersection import IntersectionClassifier

# Convenience alias
TopologyExtractor = ScenarioLoader

__all__ = [
    "ScenarioLoader",
    "SDCPath",
    "ScenarioMetadata",
    "LaneGraph",
    "ScenarioTopology",
    "IntersectionClassifier",
    "TopologyExtractor",
]
