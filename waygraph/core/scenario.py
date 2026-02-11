"""
WOMD Scenario Loading and Parsing
==================================

Provides a clean interface for loading Waymo Open Motion Dataset scenarios
from various formats (pickle files from Scenario Dreamer, TFRecord protos,
or pre-processed dictionaries).

Supports both WOMD v1.1.0 (via preprocessed .pkl files) and WOMD v1.3.1
(via direct TFRecord loading with ``sdc_paths`` extraction).

The loader extracts:
    - Lane graph (polylines + connectivity)
    - Traffic control elements (signals, stop signs, crosswalks)
    - Agent trajectories (vehicles, pedestrians, cyclists)
    - SDC paths (v1.3.1+): pre-computed valid future routes
    - Metadata (scenario ID, timestamps)

Example::

    loader = ScenarioLoader()

    # From a Scenario Dreamer .pkl file (v1.1.0)
    scenario = loader.load_pkl("path/to/scenario.pkl")

    # From a v1.3.1 TFRecord with sdc_paths
    scenario = loader.load_tfrecord("path/to/training.tfrecord", index=0)
    sdc_paths = scenario.get("sdc_paths", [])

    # Extract full topology
    topo = loader.extract_topology("path/to/scenario.pkl")
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from waygraph.core.lane_graph import LaneGraph, ScenarioTopology
from waygraph.core.intersection import IntersectionClassifier


@dataclass
class SDCPath:
    """A single pre-computed valid future route for the SDC (v1.3.1+).

    Each SDC path represents one valid route the autonomous vehicle could
    take from its current position. The ``on_route`` flag indicates whether
    this path matches the actually driven trajectory.

    Attributes:
        xy: (N, 2) array of x/y coordinates along the path.
        z: (N,) array of z (elevation) coordinates, or None.
        arc_length: (N,) array of cumulative arc length from path start (meters).
        lane_ids: (N,) array of lane/road-part integer IDs at each point.
        valid: (N,) boolean array indicating which points are valid.
        on_route: Whether this path matches the logged SDC trajectory.
    """

    xy: np.ndarray  # (N, 2)
    z: Optional[np.ndarray] = None  # (N,)
    arc_length: Optional[np.ndarray] = None  # (N,)
    lane_ids: Optional[np.ndarray] = None  # (N,) int
    valid: Optional[np.ndarray] = None  # (N,) bool
    on_route: bool = False

    @property
    def total_length_m(self) -> float:
        """Total path length in meters from arc_length, or computed from xy."""
        if self.arc_length is not None and len(self.arc_length) > 0:
            return float(self.arc_length[-1])
        if len(self.xy) >= 2:
            return float(np.sum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1)))
        return 0.0

    @property
    def unique_lane_ids(self) -> List[int]:
        """Unique lane IDs traversed by this path, in order of first appearance."""
        if self.lane_ids is None:
            return []
        seen = set()
        result = []
        for lid in self.lane_ids:
            lid = int(lid)
            if lid not in seen and lid >= 0:
                seen.add(lid)
                result.append(lid)
        return result

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "total_length_m": round(self.total_length_m, 2),
            "n_points": len(self.xy),
            "on_route": self.on_route,
            "unique_lane_ids": self.unique_lane_ids,
        }


@dataclass
class ScenarioMetadata:
    """Metadata for a WOMD scenario.

    Attributes:
        scenario_id: Unique identifier for the scenario.
        num_objects: Total number of tracked objects.
        num_vehicles: Number of vehicle agents.
        num_pedestrians: Number of pedestrian agents.
        num_cyclists: Number of cyclist agents.
        av_index: Index of the autonomous vehicle in the objects list.
        num_timesteps: Number of observation timesteps.
        dt: Time step between observations in seconds.
    """

    scenario_id: str = ""
    num_objects: int = 0
    num_vehicles: int = 0
    num_pedestrians: int = 0
    num_cyclists: int = 0
    av_index: int = -1
    num_timesteps: int = 0
    dt: float = 0.1


class ScenarioLoader:
    """Load and parse WOMD scenarios from various formats.

    Supports Scenario Dreamer .pkl files as the primary input format.
    The loader extracts lane graphs, agent trajectories, and traffic
    control elements, providing them through a clean Python API.

    Args:
        curvature_smoothing: Window size for lane curvature smoothing.
        min_lane_points: Minimum number of points for a valid lane polyline.

    Example::

        loader = ScenarioLoader()

        # Load scenario data
        scenario = loader.load_pkl("scenario.pkl")

        # Extract topology
        topo = loader.extract_topology("scenario.pkl")
        print(f"Intersection type: {topo.intersection_type}")
        print(f"Number of approaches: {topo.num_approaches}")
    """

    def __init__(
        self,
        curvature_smoothing: int = 5,
        min_lane_points: int = 3,
    ):
        self.curvature_smoothing = curvature_smoothing
        self.min_lane_points = min_lane_points
        self._lane_graph_builder = LaneGraph(
            curvature_smoothing=curvature_smoothing,
            min_lane_points=min_lane_points,
        )
        self._intersection_classifier = IntersectionClassifier()

    def load_pkl(self, pkl_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a scenario from a Scenario Dreamer .pkl file.

        Args:
            pkl_path: Path to the .pkl file containing a scenario dictionary.

        Returns:
            Scenario dictionary with keys: 'lane_graph', 'objects',
            'traffic_lights', 'av_idx'.

        Raises:
            FileNotFoundError: If the pkl file does not exist.
            ValueError: If the pkl file does not contain valid scenario data.
        """
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            scenario = pickle.load(f)

        self._validate_scenario(scenario)
        return scenario

    def load_dict(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Load a scenario from a pre-loaded dictionary.

        Args:
            scenario: Dictionary with at least a 'lane_graph' key.

        Returns:
            The validated scenario dictionary.

        Raises:
            ValueError: If the dictionary does not contain valid scenario data.
        """
        self._validate_scenario(scenario)
        return scenario

    def extract_topology(
        self,
        source: Union[str, Path, Dict[str, Any]],
        scenario_id: Optional[str] = None,
    ) -> ScenarioTopology:
        """Extract a complete topology fingerprint from a scenario.

        This is the primary entry point for topology extraction. It loads the
        scenario, builds the lane graph, computes geometric properties, extracts
        traffic controls, and classifies the intersection.

        Args:
            source: Path to a .pkl file, or a pre-loaded scenario dictionary.
            scenario_id: Optional identifier. If None, derived from filename
                or set to empty string.

        Returns:
            ScenarioTopology containing the full topological fingerprint.

        Example::

            loader = ScenarioLoader()
            topo = loader.extract_topology("scenario_001.pkl")
            print(topo.intersection_type)  # 'cross'
            print(topo.num_approaches)     # 4
            print(topo.has_traffic_light)  # True
            vec = topo.to_feature_vector() # 20D numpy array
        """
        if isinstance(source, (str, Path)):
            source = Path(source)
            scenario = self.load_pkl(source)
            if scenario_id is None:
                scenario_id = source.stem
        else:
            scenario = self.load_dict(source)
            if scenario_id is None:
                scenario_id = ""

        # Build lane graph and compute topology
        topo = self._build_topology(scenario, scenario_id)
        return topo

    def extract_batch(
        self,
        pkl_paths: List[Union[str, Path]],
        verbose: bool = True,
    ) -> List[ScenarioTopology]:
        """Extract topologies from multiple .pkl files.

        Args:
            pkl_paths: List of paths to .pkl files.
            verbose: Whether to print progress updates.

        Returns:
            List of ScenarioTopology objects. Failed extractions are skipped.
        """
        results = []
        for i, path in enumerate(pkl_paths):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processing {i + 1}/{len(pkl_paths)}...")
            try:
                topo = self.extract_topology(path)
                results.append(topo)
            except Exception as e:
                if verbose:
                    print(f"  Error processing {path}: {e}")
        return results

    def get_metadata(
        self, scenario: Dict[str, Any]
    ) -> ScenarioMetadata:
        """Extract metadata from a scenario dictionary.

        Args:
            scenario: Loaded scenario dictionary.

        Returns:
            ScenarioMetadata with counts and identifiers.
        """
        objects = scenario.get("objects", [])
        vehicles = [o for o in objects if o.get("type") == "vehicle"]
        pedestrians = [o for o in objects if o.get("type") == "pedestrian"]
        cyclists = [o for o in objects if o.get("type") == "cyclist"]

        num_timesteps = 0
        if objects and "position" in objects[0]:
            num_timesteps = len(objects[0]["position"])

        return ScenarioMetadata(
            scenario_id=scenario.get("scenario_id", ""),
            num_objects=len(objects),
            num_vehicles=len(vehicles),
            num_pedestrians=len(pedestrians),
            num_cyclists=len(cyclists),
            av_index=scenario.get("av_idx", -1),
            num_timesteps=num_timesteps,
            dt=0.1,  # WOMD is 10Hz
        )

    def _build_topology(
        self, scenario: Dict[str, Any], scenario_id: str
    ) -> ScenarioTopology:
        """Build a complete ScenarioTopology from a scenario dictionary.

        Args:
            scenario: Loaded scenario dictionary.
            scenario_id: Identifier for this scenario.

        Returns:
            Fully populated ScenarioTopology.
        """
        lg_data = scenario["lane_graph"]

        # Step 1: Build lane graph and compute features
        topo = self._lane_graph_builder.build_topology(lg_data, scenario_id)

        # Step 2: Extract traffic control information
        self._extract_traffic_controls(topo, scenario)

        # Step 3: Classify intersection
        self._intersection_classifier.classify(topo, lg_data)

        # Note: compute_graph_stats() is already called inside
        # build_topology(), so no need to call it again here.

        return topo

    def _extract_traffic_controls(
        self, topo: ScenarioTopology, scenario: Dict[str, Any]
    ) -> None:
        """Extract traffic control information from the scenario.

        Populates the traffic control fields on the topology object:
        crosswalks, stop signs, and traffic lights.

        Args:
            topo: ScenarioTopology to populate.
            scenario: Source scenario dictionary.
        """
        lg = scenario["lane_graph"]

        # Crosswalks
        crosswalks = lg.get("crosswalks", {})
        topo.num_crosswalks = len(crosswalks)
        topo.has_crosswalk = topo.num_crosswalks > 0

        # Stop signs
        stop_signs = lg.get("stop_signs", {})
        topo.num_stop_signs = len(stop_signs)
        topo.has_stop_sign = topo.num_stop_signs > 0

        # Traffic lights
        traffic_lights = scenario.get("traffic_lights", [])
        if traffic_lights and len(traffic_lights) > 0 and len(traffic_lights[0]) > 0:
            topo.has_traffic_light = True
            topo.num_traffic_lights = len(traffic_lights[0])
        else:
            topo.has_traffic_light = False
            topo.num_traffic_lights = 0

    def load_tfrecord(
        self,
        tfrecord_path: Union[str, Path],
        index: int = 0,
        extract_sdc_paths: bool = True,
    ) -> Dict[str, Any]:
        """Load a scenario from a WOMD v1.3.1 TFRecord file.

        Requires ``tensorflow`` and ``waymo-open-dataset`` to be installed.
        Extracts the same fields as .pkl loading, plus ``sdc_paths`` if
        available (v1.3.1+).

        Args:
            tfrecord_path: Path to a .tfrecord file.
            index: Which scenario within the TFRecord to load (0-based).
            extract_sdc_paths: Whether to extract sdc_paths (v1.3.1 feature).

        Returns:
            Scenario dictionary with keys: 'lane_graph', 'objects',
            'traffic_lights', 'av_idx', and optionally 'sdc_paths'.

        Raises:
            ImportError: If tensorflow or waymo SDK is not installed.
            IndexError: If the index exceeds the number of scenarios in the file.
        """
        try:
            import tensorflow as tf
            from waymo_open_dataset.protos import scenario_pb2
            from google.protobuf.json_format import MessageToDict
        except ImportError as e:
            raise ImportError(
                f"TFRecord loading requires tensorflow and waymo-open-dataset: {e}\n"
                "Install with: pip install tensorflow waymo-open-dataset-tf-2-11-0"
            ) from e

        tfrecord_path = Path(tfrecord_path)
        if not tfrecord_path.exists():
            raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")

        dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
        proto = None
        for i, data in enumerate(dataset):
            if i == index:
                proto = scenario_pb2.Scenario()
                proto.ParseFromString(data.numpy())
                break

        if proto is None:
            raise IndexError(
                f"Index {index} exceeds scenarios in {tfrecord_path.name}"
            )

        scenario_dict = MessageToDict(proto)
        scenario = self._parse_proto_to_scenario(scenario_dict, proto)

        if extract_sdc_paths:
            sdc_paths = self._extract_sdc_paths(proto)
            if sdc_paths:
                scenario["sdc_paths"] = sdc_paths

        return scenario

    def load_tfrecord_batch(
        self,
        tfrecord_path: Union[str, Path],
        max_scenarios: int = -1,
        extract_sdc_paths: bool = True,
    ) -> List[Dict[str, Any]]:
        """Load all scenarios from a TFRecord file.

        Args:
            tfrecord_path: Path to a .tfrecord file.
            max_scenarios: Maximum scenarios to load (-1 for all).
            extract_sdc_paths: Whether to extract sdc_paths.

        Returns:
            List of scenario dictionaries.
        """
        try:
            import tensorflow as tf
            from waymo_open_dataset.protos import scenario_pb2
            from google.protobuf.json_format import MessageToDict
        except ImportError as e:
            raise ImportError(
                f"TFRecord loading requires tensorflow and waymo-open-dataset: {e}"
            ) from e

        tfrecord_path = Path(tfrecord_path)
        dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")

        results = []
        for i, data in enumerate(dataset):
            if max_scenarios > 0 and i >= max_scenarios:
                break
            proto = scenario_pb2.Scenario()
            proto.ParseFromString(data.numpy())
            scenario_dict = MessageToDict(proto)
            scenario = self._parse_proto_to_scenario(scenario_dict, proto)

            if extract_sdc_paths:
                sdc_paths = self._extract_sdc_paths(proto)
                if sdc_paths:
                    scenario["sdc_paths"] = sdc_paths

            results.append(scenario)

        return results

    @staticmethod
    def _extract_sdc_paths(proto) -> List[SDCPath]:
        """Extract SDC paths from a v1.3.1+ Scenario protobuf.

        Args:
            proto: A ``scenario_pb2.Scenario`` protobuf message.

        Returns:
            List of SDCPath objects. Empty list if sdc_paths not present.
        """
        sdc_paths = []
        if not hasattr(proto, "sdc_paths") or not proto.sdc_paths:
            return sdc_paths

        for path_proto in proto.sdc_paths:
            n_points = len(path_proto.x)
            if n_points == 0:
                continue

            xy = np.column_stack([
                np.array(path_proto.x, dtype=np.float64),
                np.array(path_proto.y, dtype=np.float64),
            ])

            z = None
            if len(path_proto.z) == n_points:
                z = np.array(path_proto.z, dtype=np.float64)

            arc_length = None
            if len(path_proto.arc_length) == n_points:
                arc_length = np.array(path_proto.arc_length, dtype=np.float64)

            lane_ids = None
            if len(path_proto.ids) == n_points:
                lane_ids = np.array(path_proto.ids, dtype=np.int64)

            valid = None
            if len(path_proto.valid) == n_points:
                valid = np.array(path_proto.valid, dtype=bool)

            on_route = bool(path_proto.on_route) if hasattr(path_proto, "on_route") else False

            sdc_paths.append(SDCPath(
                xy=xy,
                z=z,
                arc_length=arc_length,
                lane_ids=lane_ids,
                valid=valid,
                on_route=on_route,
            ))

        return sdc_paths

    @staticmethod
    def _parse_proto_to_scenario(
        scenario_dict: dict, proto
    ) -> Dict[str, Any]:
        """Convert a parsed protobuf dict into the Scenario Dreamer .pkl format.

        Maintains backward compatibility with the existing .pkl format so all
        downstream code (topology extraction, traffic analysis) works unchanged.

        Args:
            scenario_dict: Dictionary from ``MessageToDict(proto)``.
            proto: The raw protobuf for traffic light extraction.

        Returns:
            Scenario dictionary in .pkl-compatible format.
        """
        import math

        ERR_VAL = -1e4
        _TYPE_MAP = {
            "TYPE_VEHICLE": "vehicle",
            "TYPE_PEDESTRIAN": "pedestrian",
            "TYPE_CYCLIST": "cyclist",
            "TYPE_OTHER": "other",
            "TYPE_UNSET": "unset",
        }

        # --- Lane graph ---
        lanes = {}
        pre_pairs = {}
        suc_pairs = {}
        left_pairs = {}
        right_pairs = {}
        road_edges = {}
        crosswalks = {}
        stop_signs = {}

        for mf in scenario_dict.get("mapFeatures", []):
            mf_id = int(mf["id"])
            if "lane" in mf:
                ln = mf["lane"]
                polyline = ln.get("polyline", [])
                if len(polyline) >= 2:
                    pts = np.array([[p["x"], p["y"]] for p in polyline], dtype=np.float64)
                    lanes[mf_id] = pts

                    for entry in ln.get("entryLanes", []):
                        eid = int(entry)
                        pre_pairs.setdefault(mf_id, []).append(eid)
                    for exit_l in ln.get("exitLanes", []):
                        sid = int(exit_l)
                        suc_pairs.setdefault(mf_id, []).append(sid)
                    for left in ln.get("leftNeighbors", []):
                        lid = int(left["featureId"])
                        left_pairs.setdefault(mf_id, []).append(lid)
                    for right in ln.get("rightNeighbors", []):
                        rid = int(right["featureId"])
                        right_pairs.setdefault(mf_id, []).append(rid)

            elif "roadEdge" in mf:
                polyline = mf["roadEdge"].get("polyline", [])
                if polyline:
                    pts = np.array([[p["x"], p["y"]] for p in polyline], dtype=np.float64)
                    road_edges[mf_id] = pts
            elif "crosswalk" in mf:
                polygon = mf["crosswalk"].get("polygon", [])
                if polygon:
                    pts = np.array([[p["x"], p["y"]] for p in polygon], dtype=np.float64)
                    crosswalks[mf_id] = pts
            elif "stopSign" in mf:
                pos = mf["stopSign"].get("position", {})
                if pos:
                    stop_signs[mf_id] = np.array([pos["x"], pos["y"]], dtype=np.float64)

        lane_graph = {
            "lanes": lanes,
            "pre_pairs": pre_pairs,
            "suc_pairs": suc_pairs,
            "left_pairs": left_pairs,
            "right_pairs": right_pairs,
            "road_edges": road_edges,
            "crosswalks": crosswalks,
            "stop_signs": stop_signs,
        }

        # --- Objects ---
        objects = []
        av_idx = int(scenario_dict.get("sdcTrackIndex", 0))
        av_objects_idx = -1

        for i, track in enumerate(scenario_dict.get("tracks", [])):
            if i == av_idx:
                av_objects_idx = len(objects)
            states = track.get("states", [])
            if not states:
                continue
            final_valid = states[-1]
            for s in reversed(states):
                if s.get("valid", False):
                    final_valid = s
                    break
            if "width" not in final_valid:
                if i == av_idx:
                    av_objects_idx = len(objects)
                continue

            obj = {
                "position": [
                    {"x": s.get("centerX", ERR_VAL), "y": s.get("centerY", ERR_VAL)}
                    if s.get("valid", False)
                    else {"x": ERR_VAL, "y": ERR_VAL}
                    for s in states
                ],
                "velocity": [
                    {"x": s.get("velocityX", 0), "y": s.get("velocityY", 0)}
                    if s.get("valid", False)
                    else {"x": ERR_VAL, "y": ERR_VAL}
                    for s in states
                ],
                "heading": [
                    math.degrees(s.get("heading", 0)) if s.get("valid", False) else ERR_VAL
                    for s in states
                ],
                "width": final_valid.get("width", 0),
                "length": final_valid.get("length", 0),
                "valid": [s.get("valid", False) for s in states],
                "type": _TYPE_MAP.get(track.get("objectType", "TYPE_UNSET"), "other"),
            }
            objects.append(obj)

        # --- Traffic lights ---
        traffic_light_states = []
        for dmap in proto.dynamic_map_states:
            frame_states = []
            for tls in dmap.lane_states:
                frame_states.append({
                    "lane": tls.lane,
                    "state": tls.state,
                    "stop_point": (
                        {"x": tls.stop_point.x, "y": tls.stop_point.y}
                        if tls.HasField("stop_point")
                        else None
                    ),
                })
            traffic_light_states.append(frame_states)

        return {
            "lane_graph": lane_graph,
            "objects": objects,
            "av_idx": av_objects_idx if av_objects_idx >= 0 else 0,
            "traffic_lights": traffic_light_states,
            "scenario_id": scenario_dict.get("scenarioId", ""),
        }

    @staticmethod
    def _validate_scenario(scenario: Dict[str, Any]) -> None:
        """Validate that a scenario dictionary has required keys.

        Args:
            scenario: Dictionary to validate.

        Raises:
            ValueError: If required keys are missing.
        """
        if not isinstance(scenario, dict):
            raise ValueError(
                f"Expected a dictionary, got {type(scenario).__name__}"
            )
        if "lane_graph" not in scenario:
            raise ValueError(
                "Scenario dictionary must contain a 'lane_graph' key. "
                "Ensure you are loading a Scenario Dreamer .pkl file."
            )
        lg = scenario["lane_graph"]
        if "lanes" not in lg:
            raise ValueError(
                "lane_graph must contain a 'lanes' key with lane polylines."
            )

    def summarize(self, topo: ScenarioTopology) -> Dict[str, Any]:
        """Create a human-readable summary of a topology.

        Args:
            topo: ScenarioTopology to summarize.

        Returns:
            Dictionary with key topology statistics.
        """
        return {
            "scenario_id": topo.scenario_id,
            "num_lanes": topo.num_lanes,
            "intersection_type": topo.intersection_type,
            "num_approaches": topo.num_approaches,
            "approach_angles": [round(a, 1) for a in topo.approach_angles],
            "has_traffic_light": topo.has_traffic_light,
            "num_traffic_lights": topo.num_traffic_lights,
            "has_stop_sign": topo.has_stop_sign,
            "has_crosswalk": topo.has_crosswalk,
            "mean_curvature": round(topo.mean_lane_curvature, 6),
            "area_m2": round(topo.area, 1),
            "num_connected_components": topo.num_connected_components,
            "branching_factor": round(topo.branching_factor, 2),
            "merge_factor": round(topo.merge_factor, 2),
            "bounding_box": tuple(round(v, 1) for v in topo.bounding_box),
        }
