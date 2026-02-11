"""
Cached data loading functions for the WayGraph web app.

Provides efficient, cached loading of WOMD scenarios and topology
extraction using Streamlit's caching decorators.
"""

import os
import sys
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Add waygraph to path
WAYGRAPH_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WAYGRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(WAYGRAPH_ROOT))

from waygraph.core import ScenarioLoader, ScenarioTopology, ScenarioMetadata
from waygraph.fingerprint import StarPattern, ApproachArm, StarPatternMatcher
from waygraph.traffic import (
    TurningRatioExtractor,
    SpeedExtractor,
    GapAcceptanceExtractor,
)

# Default data directory - use environment variable or fall back to a relative path.
# On Streamlit Cloud, no .pkl files will be present, so demo mode activates automatically.
DEFAULT_DATA_DIR = os.environ.get(
    "WAYGRAPH_DATA_DIR",
    str(Path(__file__).resolve().parent.parent / "data"),
)


def get_data_dir() -> str:
    """Get the configured data directory."""
    return st.session_state.get("data_dir", DEFAULT_DATA_DIR)


@st.cache_data(ttl=600)
def list_scenario_files(data_dir: str) -> List[str]:
    """List all .pkl scenario files in the data directory.

    Args:
        data_dir: Path to the scenario directory.

    Returns:
        Sorted list of .pkl filenames.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    files = sorted([f.name for f in data_path.glob("*.pkl")])
    return files


@st.cache_data(ttl=300)
def load_scenario_raw(data_dir: str, filename: str) -> Optional[Dict[str, Any]]:
    """Load a raw scenario dictionary from a .pkl file.

    Args:
        data_dir: Path to the data directory.
        filename: Name of the .pkl file.

    Returns:
        Scenario dictionary, or None on error.
    """
    filepath = Path(data_dir) / filename
    if not filepath.exists():
        return None
    try:
        with open(filepath, "rb") as f:
            scenario = pickle.load(f)
        if not isinstance(scenario, dict) or "lane_graph" not in scenario:
            return None
        return scenario
    except Exception:
        return None


def extract_topology(scenario: Dict[str, Any], scenario_id: str = "") -> Optional[ScenarioTopology]:
    """Extract topology from a scenario dictionary.

    Not cached directly because ScenarioTopology contains non-hashable objects.
    Use extract_topology_summary for cached summaries.

    Args:
        scenario: Raw scenario dictionary.
        scenario_id: Identifier string.

    Returns:
        ScenarioTopology or None on error.
    """
    try:
        loader = ScenarioLoader()
        topo = loader.extract_topology(scenario, scenario_id=scenario_id)
        return topo
    except Exception:
        return None


@st.cache_data(ttl=300)
def extract_topology_summary(data_dir: str, filename: str) -> Optional[Dict[str, Any]]:
    """Extract and return a cacheable topology summary.

    Args:
        data_dir: Path to the data directory.
        filename: Name of the .pkl file.

    Returns:
        Dictionary with topology summary, or None on error.
    """
    scenario = load_scenario_raw(data_dir, filename)
    if scenario is None:
        return None

    scenario_id = Path(filename).stem
    topo = extract_topology(scenario, scenario_id)
    if topo is None:
        return None

    loader = ScenarioLoader()
    summary = loader.summarize(topo)

    # Add extra fields for the web app
    summary["num_objects"] = len(scenario.get("objects", []))
    summary["num_vehicles"] = len([o for o in scenario.get("objects", []) if o.get("type") == "vehicle"])
    summary["num_pedestrians"] = len([o for o in scenario.get("objects", []) if o.get("type") == "pedestrian"])
    summary["num_cyclists"] = len([o for o in scenario.get("objects", []) if o.get("type") == "cyclist"])
    summary["approach_directions"] = [round(d, 1) for d in getattr(topo, "approach_directions", [])]
    summary["approach_lane_counts"] = list(getattr(topo, "approach_lane_counts", []))
    summary["num_graph_nodes"] = len(topo.connectivity_graph.nodes) if topo.connectivity_graph else 0
    summary["num_graph_edges"] = len(topo.connectivity_graph.edges) if topo.connectivity_graph else 0

    return summary


@st.cache_data(ttl=600, show_spinner="Extracting batch summaries...")
def extract_batch_summaries(
    data_dir: str,
    filenames: List[str],
    max_files: int = 50,
) -> pd.DataFrame:
    """Extract topology summaries for multiple files as a DataFrame.

    Args:
        data_dir: Path to the data directory.
        filenames: List of .pkl filenames to process.
        max_files: Maximum number of files to process.

    Returns:
        DataFrame with one row per successfully loaded scenario.
    """
    records = []
    for fname in filenames[:max_files]:
        summary = extract_topology_summary(data_dir, fname)
        if summary is not None:
            summary["filename"] = fname
            records.append(summary)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def build_star_pattern(
    scenario: Dict[str, Any],
    scenario_id: str = "",
) -> Optional[StarPattern]:
    """Build a star pattern fingerprint from a scenario.

    Args:
        scenario: Raw scenario dictionary.
        scenario_id: Identifier string.

    Returns:
        StarPattern or None on error.
    """
    topo = extract_topology(scenario, scenario_id)
    if topo is None:
        return None
    try:
        star = StarPattern.from_topology(topo, scenario_id=scenario_id)
        return star
    except Exception:
        return None


def extract_star_pattern_data(
    scenario: Dict[str, Any],
    scenario_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Extract star pattern data as a serializable dictionary.

    Args:
        scenario: Raw scenario dictionary.
        scenario_id: Identifier string.

    Returns:
        Dictionary with star pattern data, or None.
    """
    star = build_star_pattern(scenario, scenario_id)
    if star is None:
        return None
    data = star.to_dict()
    data["vector"] = star.to_vector().tolist()
    return data


def extract_turning_ratios(
    scenario: Dict[str, Any],
) -> Dict[str, Dict]:
    """Extract turning ratios from a scenario.

    Args:
        scenario: Raw scenario dictionary.

    Returns:
        Dictionary mapping approach_id to turning movement data.
    """
    try:
        extractor = TurningRatioExtractor()
        movements = extractor.extract(scenario)
        return {k: v.to_dict() for k, v in movements.items()}
    except Exception:
        return {}


def extract_speed_distributions(
    scenario: Dict[str, Any],
) -> Dict[str, Dict]:
    """Extract speed distributions from a scenario.

    Args:
        scenario: Raw scenario dictionary.

    Returns:
        Dictionary mapping lane_id to speed distribution data.
    """
    try:
        extractor = SpeedExtractor()
        distributions = extractor.extract(scenario)
        return {k: v.to_dict() for k, v in distributions.items()}
    except Exception:
        return {}


def extract_gap_acceptance(
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract gap acceptance parameters from a scenario.

    Args:
        scenario: Raw scenario dictionary.

    Returns:
        Dictionary with gap acceptance data.
    """
    try:
        extractor = GapAcceptanceExtractor()
        gap = extractor.extract(scenario)
        result = gap.to_dict()
        result["accepted_gaps"] = gap.accepted_gaps
        result["rejected_gaps"] = gap.rejected_gaps
        return result
    except Exception:
        return {}


def get_lane_polylines_for_plot(
    scenario: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Get lane polylines formatted for Plotly visualization.

    Args:
        scenario: Raw scenario dictionary.

    Returns:
        List of dicts with 'x', 'y', 'lane_id' keys.
    """
    lanes = scenario.get("lane_graph", {}).get("lanes", {})
    result = []
    for lane_id, polyline in lanes.items():
        if len(polyline) >= 2:
            result.append({
                "lane_id": lane_id,
                "x": polyline[:, 0].tolist(),
                "y": polyline[:, 1].tolist(),
            })
    return result


def get_objects_for_plot(
    scenario: Dict[str, Any],
    timestep: int = 0,
) -> List[Dict[str, Any]]:
    """Get object positions formatted for Plotly visualization.

    Args:
        scenario: Raw scenario dictionary.
        timestep: Which timestep to extract positions for.

    Returns:
        List of dicts with 'x', 'y', 'type', 'is_av' keys.
    """
    objects = scenario.get("objects", [])
    av_idx = scenario.get("av_idx", -1)
    result = []
    for i, obj in enumerate(objects):
        valid = obj.get("valid", [])
        positions = obj.get("position", [])
        if timestep < len(valid) and valid[timestep] and timestep < len(positions):
            pos = positions[timestep]
            if pos.get("x", -1e4) > -9999 and pos.get("y", -1e4) > -9999:
                result.append({
                    "x": pos["x"],
                    "y": pos["y"],
                    "type": obj.get("type", "unknown"),
                    "is_av": i == av_idx,
                    "index": i,
                })
    return result


def get_road_edges_for_plot(
    scenario: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Get road edge polylines for Plotly visualization.

    Args:
        scenario: Raw scenario dictionary.

    Returns:
        List of dicts with 'x', 'y' keys.
    """
    edges = scenario.get("lane_graph", {}).get("road_edges", {})
    result = []
    for edge_id, polyline in edges.items():
        if len(polyline) >= 2:
            result.append({
                "edge_id": edge_id,
                "x": polyline[:, 0].tolist(),
                "y": polyline[:, 1].tolist(),
            })
    return result


def get_crosswalks_for_plot(
    scenario: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Get crosswalk polygons for Plotly visualization.

    Args:
        scenario: Raw scenario dictionary.

    Returns:
        List of dicts with 'x', 'y' keys.
    """
    crosswalks = scenario.get("lane_graph", {}).get("crosswalks", {})
    result = []
    for cw_id, polygon in crosswalks.items():
        if len(polygon) >= 3:
            # Close the polygon
            x = polygon[:, 0].tolist() + [polygon[0, 0]]
            y = polygon[:, 1].tolist() + [polygon[0, 1]]
            result.append({"cw_id": cw_id, "x": x, "y": y})
    return result


def get_trajectory_for_plot(
    scenario: Dict[str, Any],
    obj_index: int,
) -> Optional[Dict[str, Any]]:
    """Get a single object's trajectory for Plotly visualization.

    Args:
        scenario: Raw scenario dictionary.
        obj_index: Index of the object.

    Returns:
        Dict with 'x', 'y', 'valid' lists, or None.
    """
    objects = scenario.get("objects", [])
    if obj_index < 0 or obj_index >= len(objects):
        return None

    obj = objects[obj_index]
    positions = obj.get("position", [])
    valid = obj.get("valid", [])

    x, y, v = [], [], []
    for t in range(len(positions)):
        if t < len(valid) and valid[t]:
            pos = positions[t]
            if pos.get("x", -1e4) > -9999:
                x.append(pos["x"])
                y.append(pos["y"])
                v.append(True)
            else:
                x.append(None)
                y.append(None)
                v.append(False)
        else:
            x.append(None)
            y.append(None)
            v.append(False)

    return {
        "x": x,
        "y": y,
        "valid": v,
        "type": obj.get("type", "unknown"),
    }


# --- Demo / synthetic data generation for when real data is unavailable ---

def generate_demo_scenario() -> Dict[str, Any]:
    """Generate a synthetic demo scenario for testing.

    Returns:
        A scenario dictionary mimicking the .pkl format.
    """
    np.random.seed(42)

    # Create a simple cross intersection with 4 approaches
    lanes = {}
    suc_pairs = {}
    pre_pairs = {}

    # North approach (2 lanes entering)
    for i in range(2):
        lane_id = 100 + i
        offset = (i - 0.5) * 3.5
        pts = np.array([[offset, y] for y in np.linspace(80, 5, 20)])
        lanes[lane_id] = pts

    # South approach (2 lanes exiting)
    for i in range(2):
        lane_id = 200 + i
        offset = (i - 0.5) * 3.5
        pts = np.array([[offset, y] for y in np.linspace(-5, -80, 20)])
        lanes[lane_id] = pts

    # East approach (2 lanes entering)
    for i in range(2):
        lane_id = 300 + i
        offset = (i - 0.5) * 3.5
        pts = np.array([[x, offset] for x in np.linspace(80, 5, 20)])
        lanes[lane_id] = pts

    # West approach (2 lanes exiting)
    for i in range(2):
        lane_id = 400 + i
        offset = (i - 0.5) * 3.5
        pts = np.array([[x, offset] for x in np.linspace(-5, -80, 20)])
        lanes[lane_id] = pts

    # Connecting lanes (through movements)
    lane_id = 500
    for entry, exit_lane in [(100, 200), (101, 201), (300, 400), (301, 401)]:
        pts_entry = lanes[entry]
        pts_exit = lanes[exit_lane]
        conn = np.array([
            pts_entry[-1],
            (pts_entry[-1] + pts_exit[0]) / 2,
            pts_exit[0],
        ])
        lanes[lane_id] = conn
        suc_pairs[entry] = [lane_id]
        pre_pairs[lane_id] = [entry]
        suc_pairs[lane_id] = [exit_lane]
        pre_pairs[exit_lane] = [lane_id]
        lane_id += 1

    # Road edges
    road_edges = {
        1: np.array([[-8, 80], [-8, -80]]),
        2: np.array([[8, 80], [8, -80]]),
        3: np.array([[-80, -8], [80, -8]]),
        4: np.array([[-80, 8], [80, 8]]),
    }

    # Crosswalks
    crosswalks = {
        10: np.array([[-8, 12], [8, 12], [8, 15], [-8, 15]]),
        11: np.array([[-8, -12], [8, -12], [8, -15], [-8, -15]]),
    }

    # Stop signs
    stop_signs = {
        20: np.array([6, 12]),
        21: np.array([-6, -12]),
    }

    lane_graph = {
        "lanes": lanes,
        "suc_pairs": suc_pairs,
        "pre_pairs": pre_pairs,
        "left_pairs": {},
        "right_pairs": {},
        "road_edges": road_edges,
        "crosswalks": crosswalks,
        "stop_signs": stop_signs,
    }

    # Create some vehicle objects
    objects = []
    for v_idx in range(8):
        np.random.seed(v_idx + 10)
        start_x = np.random.uniform(-60, 60)
        start_y = np.random.uniform(-60, 60)
        vx = np.random.uniform(-5, 5)
        vy = np.random.uniform(-5, 5)

        positions = []
        velocities = []
        headings = []
        valids = []

        for t in range(91):
            x = start_x + vx * t * 0.1
            y = start_y + vy * t * 0.1
            heading = np.degrees(np.arctan2(vy, vx))
            positions.append({"x": float(x), "y": float(y)})
            velocities.append({"x": float(vx), "y": float(vy)})
            headings.append(float(heading))
            valids.append(True)

        objects.append({
            "position": positions,
            "velocity": velocities,
            "heading": headings,
            "valid": valids,
            "width": 2.0,
            "length": 4.5,
            "type": "vehicle",
        })

    # Add a pedestrian
    ped_positions = []
    for t in range(91):
        ped_positions.append({"x": float(-5 + t * 0.02), "y": float(13)})
    objects.append({
        "position": ped_positions,
        "velocity": [{"x": 0.2, "y": 0.0}] * 91,
        "heading": [0.0] * 91,
        "valid": [True] * 91,
        "width": 0.5,
        "length": 0.5,
        "type": "pedestrian",
    })

    # Traffic lights (per-timestep states)
    traffic_lights = []
    for t in range(91):
        frame_states = [
            {"lane": 100, "state": 3 if t < 45 else 4, "stop_point": {"x": 0, "y": 10}},
            {"lane": 300, "state": 4 if t < 45 else 3, "stop_point": {"x": 10, "y": 0}},
        ]
        traffic_lights.append(frame_states)

    return {
        "lane_graph": lane_graph,
        "objects": objects,
        "av_idx": 0,
        "traffic_lights": traffic_lights,
        "scenario_id": "demo_cross_intersection",
    }


def generate_demo_star_patterns(n: int = 20) -> List[Dict[str, Any]]:
    """Generate synthetic star patterns for demo matching.

    Args:
        n: Number of patterns to generate.

    Returns:
        List of star pattern dictionaries.
    """
    np.random.seed(123)
    types = ["T", "cross", "Y", "multi", "cross", "T"]
    patterns = []

    for i in range(n):
        center_type = types[i % len(types)]
        if center_type == "T":
            n_arms = 3
        elif center_type == "cross":
            n_arms = 4
        elif center_type == "Y":
            n_arms = 3
        else:
            n_arms = np.random.randint(5, 7)

        arms = []
        base_angle = np.random.uniform(0, 360)
        for j in range(n_arms):
            angle = (base_angle + j * 360 / n_arms + np.random.uniform(-10, 10)) % 360
            arms.append({
                "angle": round(angle, 1),
                "length": round(np.random.uniform(50, 300), 1),
                "road_type": np.random.choice(["primary", "secondary", "tertiary", "residential"]),
                "lanes": int(np.random.choice([1, 2, 3, 4])),
                "neighbor_type": np.random.choice(["T", "cross", "Y", "merge"]),
                "neighbor_degree": int(np.random.randint(3, 10)),
                "neighbor_has_signal": bool(np.random.random() > 0.5),
            })

        patterns.append({
            "id": f"osm_node_{1000 + i}",
            "center_type": center_type,
            "center_approaches": n_arms,
            "center_has_signal": bool(np.random.random() > 0.4),
            "center_has_stop": bool(np.random.random() > 0.7),
            "center_has_crosswalk": bool(np.random.random() > 0.5),
            "lat": round(37.77 + np.random.uniform(-0.05, 0.05), 5),
            "lon": round(-122.42 + np.random.uniform(-0.05, 0.05), 5),
            "n_arms": n_arms,
            "arms": arms,
        })

    return patterns
