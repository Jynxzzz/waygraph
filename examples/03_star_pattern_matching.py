#!/usr/bin/env python3
"""
Example 3: Star Pattern Fingerprinting and Matching
=====================================================

This example demonstrates the full star pattern matching pipeline:
1. Build an OSM star pattern database for a city
2. Extract a star pattern from a WOMD scenario
3. Match the WOMD pattern against the OSM database

Usage:
    python 03_star_pattern_matching.py [--city "San Francisco, CA"]

Note: Requires waygraph[osm] for OSM functionality.
"""

import sys
import json
from pathlib import Path

import numpy as np

from waygraph.fingerprint import StarPattern, ApproachArm, StarPatternMatcher


def demo_with_synthetic_data():
    """Demonstrate matching with synthetic data (no OSM needed)."""
    print("--- Demo: Star Pattern Matching (Synthetic Data) ---\n")

    # Create a synthetic OSM database
    rng = np.random.RandomState(42)
    db_patterns = []

    for i in range(500):
        n_arms = rng.choice([3, 4, 5], p=[0.4, 0.5, 0.1])
        itype = {3: "T", 4: "cross", 5: "multi"}[n_arms]
        has_signal = rng.random() > 0.4

        arms = []
        for j in range(n_arms):
            arm = ApproachArm(
                angle_deg=(j * 360.0 / n_arms + rng.normal(0, 10)) % 360,
                road_length_m=50 + rng.exponential(100),
                road_type=rng.choice(["residential", "secondary", "primary"]),
                num_lanes=rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2]),
                neighbor_type=rng.choice(["T", "cross", "merge"]),
                neighbor_degree=rng.choice([4, 6, 8]),
                neighbor_has_signal=rng.random() > 0.5,
            )
            arms.append(arm)

        star = StarPattern(
            id=f"osm_{i:04d}",
            center_type=itype,
            center_approaches=n_arms,
            center_has_signal=has_signal,
            arms=arms,
        )
        db_patterns.append(star)

    print(f"Created {len(db_patterns)} synthetic OSM patterns")

    # Build matcher
    matcher = StarPatternMatcher()
    matcher.build_database(db_patterns)
    print(f"Database size: {matcher.database_size}")

    # Create a query pattern (noisy copy of a database pattern)
    true_idx = 42
    true_pattern = db_patterns[true_idx]
    query = StarPattern(
        id="query_001",
        center_type=true_pattern.center_type,
        center_approaches=true_pattern.center_approaches,
        center_has_signal=true_pattern.center_has_signal,
        arms=[
            ApproachArm(
                angle_deg=(a.angle_deg + rng.normal(0, 15)) % 360,
                road_length_m=max(10, a.road_length_m + rng.normal(0, 20)),
                road_type=a.road_type,
                num_lanes=a.num_lanes,
                neighbor_type=a.neighbor_type,
                neighbor_degree=a.neighbor_degree,
                neighbor_has_signal=a.neighbor_has_signal,
            )
            for a in true_pattern.arms
        ],
    )

    # Match
    print(f"\nQuery: {query.id} (true match: {true_pattern.id})")
    print(f"  Center type: {query.center_type}, Arms: {len(query.arms)}")

    results = matcher.match(query, top_k=5)
    print(f"\nTop-5 matches:")
    for rank, (db_id, score) in enumerate(results, 1):
        marker = " <-- CORRECT" if db_id == true_pattern.id else ""
        print(f"  #{rank}: {db_id} (score={score:.4f}){marker}")

    # Feature vector inspection
    print(f"\nFeature vector (48D):")
    vec = query.to_vector()
    print(f"  Shape: {vec.shape}")
    print(f"  Center features: {vec[:6]}")
    print(f"  Arm 0 features:  {vec[6:13]}")

    # Batch evaluation with synthetic ground truth
    print("\n--- Batch Evaluation ---")
    n_test = 100
    queries = []
    ground_truth = {}

    for _ in range(n_test):
        idx = rng.randint(0, len(db_patterns))
        true_p = db_patterns[idx]
        noisy = StarPattern(
            id=f"test_{idx}",
            center_type=true_p.center_type,
            center_approaches=true_p.center_approaches,
            center_has_signal=true_p.center_has_signal,
            arms=[
                ApproachArm(
                    angle_deg=(a.angle_deg + rng.normal(0, 15)) % 360,
                    road_length_m=max(10, a.road_length_m + rng.normal(0, 30)),
                    road_type=a.road_type,
                    num_lanes=a.num_lanes,
                    neighbor_type=a.neighbor_type,
                    neighbor_degree=a.neighbor_degree,
                    neighbor_has_signal=a.neighbor_has_signal,
                )
                for a in true_p.arms
            ],
        )
        queries.append(noisy)
        ground_truth[noisy.id] = true_p.id

    metrics = matcher.evaluate(queries, ground_truth)
    print(f"  Samples: {metrics['n']}")
    print(f"  Top-1:   {metrics.get('top1', 0)}%")
    print(f"  Top-5:   {metrics.get('top5', 0)}%")
    print(f"  Top-10:  {metrics.get('top10', 0)}%")
    print(f"  MRR:     {metrics.get('mrr', 0)}")


def main():
    demo_with_synthetic_data()

    # If OSM is available, demonstrate real matching
    try:
        from waygraph.osm import OSMStarDatabase

        print("\n\n--- OSM star pattern extraction available ---")
        print("Use OSMStarDatabase to build a real pattern database:")
        print("  db = OSMStarDatabase()")
        print('  patterns = db.build_from_graphml("city_graph.graphml")')
    except ImportError:
        print("\nInstall waygraph[osm] for OSM pattern extraction.")


if __name__ == "__main__":
    main()
