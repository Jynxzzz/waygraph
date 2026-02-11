#!/usr/bin/env python3
"""
Example 4: Traffic Parameter Extraction
=========================================

This example shows how to extract traffic parameters from a WOMD
scenario, including turning ratios, speed distributions, and gap
acceptance parameters.

Usage:
    python 04_traffic_extraction.py /path/to/scenario.pkl
"""

import sys

from waygraph.core import ScenarioLoader
from waygraph.traffic import (
    TurningRatioExtractor,
    SpeedExtractor,
    GapAcceptanceExtractor,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python 04_traffic_extraction.py <path_to_scenario.pkl>")
        print("\nThis example extracts traffic parameters from a scenario.")
        return

    pkl_path = sys.argv[1]

    # Load scenario
    loader = ScenarioLoader()
    scenario = loader.load_pkl(pkl_path)
    topo = loader.extract_topology(scenario, scenario_id="demo")

    print(f"Scenario loaded: {topo.num_lanes} lanes, "
          f"type={topo.intersection_type}")

    # 1. Turning ratios
    print("\n--- Turning Ratios ---")
    turn_extractor = TurningRatioExtractor()
    movements = turn_extractor.extract(scenario, topo)

    if movements:
        for approach_id, tm in movements.items():
            print(f"  {approach_id}:")
            print(f"    Left:    {tm.left_count:3d} ({tm.left_ratio:.1%})")
            print(f"    Through: {tm.through_count:3d} ({tm.through_ratio:.1%})")
            print(f"    Right:   {tm.right_count:3d} ({tm.right_ratio:.1%})")
            print(f"    Total:   {tm.total_count}")
    else:
        print("  No turning movements detected.")

    # 2. Speed distributions
    print("\n--- Speed Distributions ---")
    speed_extractor = SpeedExtractor()
    speeds = speed_extractor.extract(scenario, topo)

    if speeds:
        # Show top 5 lanes by sample count
        sorted_lanes = sorted(
            speeds.items(),
            key=lambda x: len(x[1].speeds_ms),
            reverse=True,
        )[:5]
        for lane_id, sd in sorted_lanes:
            print(f"  Lane {lane_id}:")
            print(f"    Samples: {len(sd.speeds_ms)}")
            print(f"    Mean:    {sd.mean_speed_kmh:.1f} km/h")
            print(f"    Free-flow (85th pct): {sd.free_flow_speed_kmh:.1f} km/h")
    else:
        print("  No speed data extracted.")

    # 3. Gap acceptance
    print("\n--- Gap Acceptance ---")
    gap_extractor = GapAcceptanceExtractor()
    gap = gap_extractor.extract(scenario)

    print(f"  Accepted gaps: {gap.n_accepted}")
    print(f"  Rejected gaps: {gap.n_rejected}")
    if gap.critical_gap_s > 0:
        print(f"  Critical gap:  {gap.critical_gap_s:.1f} s")
        print(f"  Follow-up time: {gap.follow_up_time_s:.1f} s")
    else:
        print("  Insufficient data for gap estimation.")


if __name__ == "__main__":
    main()
