#!/usr/bin/env python3
"""
Example 1: Load and Inspect a WOMD Scenario
=============================================

This example shows how to load a WOMD scenario from a Scenario Dreamer
.pkl file, extract its topology, and print a summary.

Usage:
    python 01_load_scenario.py /path/to/scenario.pkl
"""

import sys
from pprint import pprint

from waygraph.core import ScenarioLoader


def main():
    if len(sys.argv) < 2:
        print("Usage: python 01_load_scenario.py <path_to_scenario.pkl>")
        print("\nThis example loads a WOMD scenario and prints its topology.")
        return

    pkl_path = sys.argv[1]

    # Initialize the loader
    loader = ScenarioLoader()

    # Load and extract topology
    print(f"Loading scenario from: {pkl_path}")
    topo = loader.extract_topology(pkl_path)

    # Print summary
    print("\n--- Topology Summary ---")
    summary = loader.summarize(topo)
    pprint(summary)

    # Access specific properties
    print(f"\n--- Key Properties ---")
    print(f"  Scenario ID:        {topo.scenario_id}")
    print(f"  Intersection type:  {topo.intersection_type}")
    print(f"  Number of lanes:    {topo.num_lanes}")
    print(f"  Number of approaches: {topo.num_approaches}")
    print(f"  Approach angles:    {topo.approach_angles}")
    print(f"  Has traffic light:  {topo.has_traffic_light}")
    print(f"  Has stop sign:      {topo.has_stop_sign}")
    print(f"  Has crosswalk:      {topo.has_crosswalk}")
    print(f"  Area (m^2):         {topo.area:.1f}")
    print(f"  Mean curvature:     {topo.mean_lane_curvature:.6f}")
    print(f"  Connected components: {topo.num_connected_components}")
    print(f"  Branching factor:   {topo.branching_factor:.2f}")
    print(f"  Merge factor:       {topo.merge_factor:.2f}")

    # Feature vector for comparison
    vec = topo.to_feature_vector()
    print(f"\n--- Feature Vector (20D) ---")
    print(f"  Shape: {vec.shape}")
    print(f"  Values: {vec}")


if __name__ == "__main__":
    main()
