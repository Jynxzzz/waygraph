"""
Visualization tools for WayGraph.

Provides publication-quality plots for scenario topologies, intersection
analysis, star pattern matching results, and traffic pattern distributions.

Requires the ``viz`` extra: ``pip install waygraph[viz]``
"""

from waygraph.viz.scenario import ScenarioVisualizer
from waygraph.viz.intersection import IntersectionVisualizer
from waygraph.viz.matching import MatchVisualizer

__all__ = [
    "ScenarioVisualizer",
    "IntersectionVisualizer",
    "MatchVisualizer",
]
