# WayGraph: Structural Analysis Toolkit for Autonomous Driving Datasets

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/waygraph.svg)](https://pypi.org/project/waygraph/)

**WayGraph** is a Python toolkit for structural analysis of road networks in autonomous driving datasets. Starting with the [Waymo Open Motion Dataset (WOMD)](https://waymo.com/open/), WayGraph provides intersection detection, star pattern fingerprinting (adapted from GIS map conflation), traffic parameter extraction, OpenStreetMap matching, and publication-quality visualization -- the first open-source toolkit to provide these capabilities for AV dataset analysis.

While existing tools focus on *what agents do* (trajectories, prediction, simulation), WayGraph focuses on *where agents drive* (network structure, intersection topology, traffic patterns). This makes it complementary to every major tool in the AV research ecosystem.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Intersection Detection** | Automatically detect and classify intersections (T, Y, cross, roundabout, multi-way) from WOMD lane graphs |
| **Star Pattern Fingerprinting** | 48-dimensional feature vectors encoding intersection topology + 1-hop neighborhood context |
| **OSM Matching** | Match WOMD scenarios to real-world OpenStreetMap locations using topological fingerprints |
| **Traffic Parameter Extraction** | Extract turning ratios, speed distributions, and gap acceptance parameters from observed trajectories |
| **Lane Graph Analysis** | Build connectivity graphs, compute curvature profiles, and analyze lane topology |
| **Publication-Quality Visualization** | Ready-to-use plots for scenario topologies, intersection analysis, and matching results |

---

## Installation

### Basic (core functionality)

```bash
pip install waygraph
```

### With visualization support

```bash
pip install waygraph[viz]
```

### With OpenStreetMap integration

```bash
pip install waygraph[osm]
```

### Full installation

```bash
pip install waygraph[all]
```

### Development

```bash
git clone https://github.com/network-dreamer/waygraph.git
cd waygraph
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Load a scenario and extract topology

```python
from waygraph.core import ScenarioLoader

loader = ScenarioLoader()
topo = loader.extract_topology("scenario.pkl")

print(f"Intersection type: {topo.intersection_type}")  # 'cross'
print(f"Approaches: {topo.num_approaches}")             # 4
print(f"Has traffic light: {topo.has_traffic_light}")    # True
print(f"Lanes: {topo.num_lanes}")                        # 87
print(f"Area: {topo.area:.0f} m^2")                      # 24850
```

### 2. Build a star pattern fingerprint

```python
from waygraph.fingerprint import StarPattern

# From a topology
star = StarPattern.from_topology(topo)
vector = star.to_vector()  # 48D numpy array
print(f"Fingerprint: {vector.shape}")  # (48,)

# Directly
star = StarPattern(
    center_type="cross",
    center_approaches=4,
    center_has_signal=True,
    arms=[
        ApproachArm(angle_deg=0, road_length_m=150, road_type="primary"),
        ApproachArm(angle_deg=90, road_length_m=200, road_type="secondary"),
        ApproachArm(angle_deg=180, road_length_m=150, road_type="primary"),
        ApproachArm(angle_deg=270, road_length_m=120, road_type="tertiary"),
    ],
)
```

### 3. Match against OSM

```python
from waygraph.osm import OSMNetwork, OSMStarDatabase
from waygraph.fingerprint import StarPatternMatcher

# Build OSM reference database
db = OSMStarDatabase()
osm_patterns = db.build_from_graphml("sf_graph.graphml")

# Match
matcher = StarPatternMatcher()
matcher.build_database(osm_patterns)
results = matcher.match(star, top_k=5)

for osm_id, score in results:
    print(f"  OSM node {osm_id}: score={score:.3f}")
```

### 4. Extract traffic parameters

```python
from waygraph.traffic import TurningRatioExtractor, SpeedExtractor

# Turning ratios
turn_ext = TurningRatioExtractor()
movements = turn_ext.extract(scenario)
for approach, tm in movements.items():
    print(f"{approach}: L={tm.left_ratio:.0%} T={tm.through_ratio:.0%} R={tm.right_ratio:.0%}")

# Speed distributions
speed_ext = SpeedExtractor()
speeds = speed_ext.extract(scenario)
for lane_id, sd in speeds.items():
    print(f"Lane {lane_id}: mean={sd.mean_speed_kmh:.1f} km/h")
```

### 5. Visualize results

```python
from waygraph.viz import ScenarioVisualizer, IntersectionVisualizer

viz = ScenarioVisualizer(output_dir="figures/")
viz.plot_topology(topo, scenario, save_name="scenario_001")

iviz = IntersectionVisualizer(output_dir="figures/")
iviz.plot_type_distribution(topologies, save_name="analysis")
```

---

## Comparison with Existing Tools

| Capability | WayGraph | Waymo SDK | trajdata | ScenarioNet |
|---|:---:|:---:|:---:|:---:|
| WOMD scenario loading | Yes | Yes | Yes | Yes |
| TF-free parsing | Yes | No | Partial | Yes |
| Lane graph extraction | Yes | No | No | No |
| Intersection detection | Yes | No | No | No |
| Intersection classification | Yes | No | No | No |
| Topology fingerprinting | Yes | No | No | No |
| OSM matching | Yes | No | No | No |
| Turning ratio extraction | Yes | No | No | No |
| Speed distribution analysis | Yes | No | No | No |
| Gap acceptance estimation | Yes | No | No | No |
| Publication-quality visualization | Yes | Partial | Partial | No |
| Multi-dataset support | Planned | No | Yes | Yes |
| Simulation integration | Planned | Waymax | No | Yes |

**WayGraph occupies an uncontested niche**: deep structural analysis of road networks in AV datasets. It is complementary to trajectory-focused tools like trajdata and simulation tools like Waymax.

---

## Architecture

```
waygraph/
  core/              # Scenario loading, lane graph construction, intersection classification
    scenario.py      # ScenarioLoader: load .pkl files, extract topology
    lane_graph.py    # LaneGraph: connectivity graph, curvature, geometry
    intersection.py  # IntersectionClassifier: detect and classify intersections
  fingerprint/       # Star pattern fingerprinting and matching
    star_pattern.py  # StarPattern: 48D feature vector
    matching.py      # StarPatternMatcher: database matching with coarse filtering
  traffic/           # Traffic parameter extraction
    turning_ratio.py # Turning movement counts and ratios
    speed.py         # Per-lane speed distributions
    gap_acceptance.py # Gap acceptance parameters
  osm/               # OpenStreetMap integration
    download.py      # OSMNetwork: download and process OSM road networks
    star_db.py       # OSMStarDatabase: build star pattern reference databases
  viz/               # Visualization
    scenario.py      # Scenario topology plots
    intersection.py  # Intersection analysis plots
    matching.py      # Match result visualization
  utils/             # Geometric utilities
    geometry.py      # Angle, curvature, and transformation functions
```

---

## How It Works

### Star Pattern Fingerprinting

WayGraph adapts **star pattern fingerprinting** from GIS map conflation literature to the autonomous driving domain. The approach encodes the topology of an intersection and its 1-hop neighborhood as a 48-dimensional feature vector.

```
                    Neighbor B (T-intersection, 6 degree)
                        |
                        | 200m, secondary, 2 lanes
                        |
    Neighbor A -------- CENTER -------- Neighbor C
    (cross, 8 deg)  150m, primary   150m, primary   (cross, 8 deg)
                        |
                        | 120m, tertiary, 1 lane
                        |
                    Neighbor D (merge, 4 degree)
```

The feature vector captures:
- **Center features** (6D): intersection type, approach count, signal/stop/crosswalk flags, arm count
- **Per-arm features** (7D each, max 6 arms = 42D): angle, road length, road type, lane count, neighbor type, neighbor degree, neighbor signal

This provides dramatically more discriminative power than matching a single intersection in isolation. On synthetic benchmarks:
- **Top-1 accuracy: ~90%** (matching against 5,000+ OSM intersections)
- **MRR: 0.92**

### Related Work

The star pattern matching approach builds on decades of research in GIS map conflation and spatial data integration:

- **Map conflation**: The problem of matching road network representations from different sources has been studied extensively in GIS, with methods including geometric matching, topological matching, and hybrid approaches.
- **Structural descriptors**: Similar "star graph" or "junction fingerprint" concepts have been proposed for matching road intersections in map integration tasks.

**WayGraph's contribution** is not the algorithmic novelty of star patterns, but rather:
1. **First application to AV dataset localization**: Adapting GIS map conflation techniques to GPS-free scenario matching in autonomous driving datasets.
2. **Systematic empirical study**: Demonstrating 90% top-1 accuracy on real-world Waymo scenarios matched against OpenStreetMap.
3. **Open-source toolkit**: Providing the first reusable, documented implementation for structural analysis of AV datasets.

---

## Examples

See the [`examples/`](examples/) directory for complete runnable scripts:

| Example | Description |
|---|---|
| `01_load_scenario.py` | Load a scenario and print its topology |
| `02_extract_intersections.py` | Batch extraction with distribution analysis |
| `03_star_pattern_matching.py` | Star pattern fingerprinting and matching demo |
| `04_traffic_extraction.py` | Turning ratios, speeds, and gap acceptance |

---

## Citation

If you use WayGraph in your research, please cite the Network Dreamer paper:

```bibtex
@article{networkdreamer2026,
    title={Network Dreamer: Structural Analysis and Matching of Road Networks
           in Autonomous Driving Datasets},
    author={Network Dreamer Authors},
    year={2026},
    journal={arXiv preprint},
}
```

---

## Contributing

We welcome contributions! Here is how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Make your changes and add tests
5. Run tests: `pytest`
6. Run linting: `ruff check waygraph/`
7. Submit a pull request

### Areas where we especially welcome contributions:

- Support for additional datasets (Argoverse 2, nuScenes)
- SUMO network export from extracted parameters
- Improved intersection classification algorithms
- Web-based visualization dashboard
- Performance optimization for large-scale batch processing
- Documentation improvements and tutorials

---

## Roadmap

- [x] Core topology extraction from WOMD
- [x] Star pattern fingerprinting (48D)
- [x] OSM matching pipeline
- [x] Traffic parameter extraction
- [ ] Pre-computed intersection database for all WOMD scenarios
- [ ] Argoverse 2 / nuScenes map analysis adapters
- [ ] SUMO network export
- [ ] trajdata integration adapter
- [ ] Jupyter notebook tutorials
- [ ] Web visualization dashboard
- [ ] HuggingFace Datasets integration

---

## License

This project is licensed under the Apache License 2.0 -- see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

WayGraph builds on several excellent open-source projects:

- [osmnx](https://github.com/gboeing/osmnx) for OpenStreetMap data access
- [NetworkX](https://networkx.org/) for graph algorithms
- [Waymo Open Dataset](https://waymo.com/open/) for the motion dataset
- [Scenario Dreamer](https://github.com/scenariodreamer) for WOMD preprocessing
