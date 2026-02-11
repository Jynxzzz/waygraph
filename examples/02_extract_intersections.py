#!/usr/bin/env python3
"""
Example 2: Batch Intersection Extraction
==========================================

This example shows how to extract intersection topologies from a directory
of WOMD scenario .pkl files and analyze the distribution of intersection
types across the dataset.

Usage:
    python 02_extract_intersections.py /path/to/scenario_directory/
"""

import sys
from collections import Counter
from pathlib import Path

from waygraph.core import ScenarioLoader


def main():
    if len(sys.argv) < 2:
        print("Usage: python 02_extract_intersections.py <scenario_directory>")
        print("\nThis example extracts intersections from multiple scenarios.")
        return

    scenario_dir = Path(sys.argv[1])
    pkl_files = sorted(scenario_dir.glob("*.pkl"))

    if not pkl_files:
        print(f"No .pkl files found in {scenario_dir}")
        return

    print(f"Found {len(pkl_files)} scenario files in {scenario_dir}")

    # Extract topologies
    loader = ScenarioLoader()
    topos = loader.extract_batch([str(p) for p in pkl_files], verbose=True)

    print(f"\nSuccessfully extracted {len(topos)} topologies")

    # Analyze intersection types
    type_counts = Counter(t.intersection_type for t in topos)
    print("\n--- Intersection Type Distribution ---")
    for itype, count in type_counts.most_common():
        pct = 100 * count / len(topos)
        print(f"  {itype:>12}: {count:4d} ({pct:5.1f}%)")

    # Approach count distribution
    approach_counts = Counter(t.num_approaches for t in topos)
    print("\n--- Approach Count Distribution ---")
    for n_approaches, count in sorted(approach_counts.items()):
        pct = 100 * count / len(topos)
        print(f"  {n_approaches} approaches: {count:4d} ({pct:5.1f}%)")

    # Traffic control statistics
    n_tl = sum(1 for t in topos if t.has_traffic_light)
    n_ss = sum(1 for t in topos if t.has_stop_sign)
    n_cw = sum(1 for t in topos if t.has_crosswalk)
    print("\n--- Traffic Controls ---")
    print(f"  With traffic lights: {n_tl} ({100*n_tl/len(topos):.1f}%)")
    print(f"  With stop signs:     {n_ss} ({100*n_ss/len(topos):.1f}%)")
    print(f"  With crosswalks:     {n_cw} ({100*n_cw/len(topos):.1f}%)")

    # Optional: visualize (requires matplotlib)
    try:
        from waygraph.viz import IntersectionVisualizer

        viz = IntersectionVisualizer(output_dir="./output")
        fig_path = viz.plot_type_distribution(topos, save_name="intersection_analysis")
        print(f"\nVisualization saved to: {fig_path}")
    except ImportError:
        print("\nInstall waygraph[viz] for visualization support.")


if __name__ == "__main__":
    main()
