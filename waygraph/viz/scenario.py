"""
Scenario Visualization
=======================

Plot WOMD scenario topologies with lanes, vehicles, traffic controls,
and connectivity graphs.

Example::

    from waygraph.viz import ScenarioVisualizer

    viz = ScenarioVisualizer(output_dir="figures/")
    viz.plot_topology(topo, scenario, save_name="scenario_001")
    viz.plot_connectivity(topo, save_name="connectivity_001")
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Color palette
LANE_COLOR = "#4A90D9"
ROAD_EDGE_COLOR = "#666666"
CROSSWALK_COLOR = "#F5A623"
STOP_SIGN_COLOR = "#D0021B"
TRAFFIC_LIGHT_COLOR = "#7ED321"
VEHICLE_COLOR = "#BD10E0"
AV_COLOR = "#FF6600"


class ScenarioVisualizer:
    """Visualization tools for WOMD scenario topologies.

    Args:
        output_dir: Directory to save figures.
        dpi: Resolution for saved figures.

    Raises:
        ImportError: If matplotlib is not installed.
    """

    def __init__(self, output_dir: str = "./figures", dpi: int = 150):
        if not HAS_MPL:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install waygraph[viz]"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_topology(
        self,
        topo: "ScenarioTopology",
        scenario: Optional[dict] = None,
        save_name: str = "topology",
        show_vehicles: bool = True,
        show_traffic_controls: bool = True,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 10),
    ) -> str:
        """Plot a scenario topology with lanes, vehicles, and controls.

        Args:
            topo: ScenarioTopology to visualize.
            scenario: Original scenario dict (for objects and controls).
            save_name: Output filename (without extension).
            show_vehicles: Whether to show vehicle positions.
            show_traffic_controls: Whether to show traffic controls.
            title: Custom plot title.
            figsize: Figure size in inches.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Draw lane polylines
        for lane_id, polyline in topo.lane_polylines.items():
            if len(polyline) < 2:
                continue
            ax.plot(
                polyline[:, 0],
                polyline[:, 1],
                color=LANE_COLOR,
                linewidth=1.5,
                alpha=0.7,
            )
            # Direction arrow at midpoint
            mid = len(polyline) // 2
            if mid > 0:
                ax.annotate(
                    "",
                    xy=(polyline[mid, 0], polyline[mid, 1]),
                    xytext=(polyline[mid - 1, 0], polyline[mid - 1, 1]),
                    arrowprops=dict(arrowstyle="->", color=LANE_COLOR, lw=1),
                )

        # Scenario elements
        if scenario is not None:
            lg = scenario.get("lane_graph", {})

            # Road edges
            for edge_id, polyline in lg.get("road_edges", {}).items():
                if len(polyline) >= 2:
                    ax.plot(
                        polyline[:, 0],
                        polyline[:, 1],
                        color=ROAD_EDGE_COLOR,
                        linewidth=0.8,
                        alpha=0.5,
                    )

            if show_traffic_controls:
                # Crosswalks
                for cw_id, polygon in lg.get("crosswalks", {}).items():
                    if len(polygon) >= 3:
                        patch = plt.Polygon(
                            polygon[:, :2],
                            fill=True,
                            facecolor=CROSSWALK_COLOR,
                            alpha=0.3,
                            edgecolor=CROSSWALK_COLOR,
                            linewidth=1,
                        )
                        ax.add_patch(patch)

                # Stop signs
                for ss_id, pos in lg.get("stop_signs", {}).items():
                    ax.plot(
                        pos[0], pos[1], "s",
                        color=STOP_SIGN_COLOR, markersize=8, zorder=5,
                    )

            # Vehicles
            if show_vehicles and "objects" in scenario:
                t = 0
                for i, obj in enumerate(scenario["objects"]):
                    if not obj["valid"][t]:
                        continue
                    pos = obj["position"][t]
                    color = (
                        AV_COLOR
                        if i == scenario.get("av_idx", -1)
                        else VEHICLE_COLOR
                    )
                    size = 6 if i == scenario.get("av_idx", -1) else 4
                    ax.plot(
                        pos["x"], pos["y"], "o",
                        color=color, markersize=size, zorder=4,
                    )

        # Centroid marker
        ax.plot(
            topo.centroid[0], topo.centroid[1], "x",
            color="red", markersize=12, markeredgewidth=2, zorder=6,
        )

        # Bounding box
        xmin, ymin, xmax, ymax = topo.bounding_box
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="gray",
            linestyle="--",
            linewidth=0.5,
        )
        ax.add_patch(rect)

        # Title
        if title is None:
            title = (
                f"Scenario: {topo.scenario_id}\n"
                f"Type: {topo.intersection_type} | "
                f"Approaches: {topo.num_approaches} | "
                f"Lanes: {topo.num_lanes}"
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = str(self.output_dir / f"{save_name}.png")
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_connectivity(
        self,
        topo: "ScenarioTopology",
        save_name: str = "connectivity",
        figsize: Tuple[float, float] = (10, 10),
    ) -> str:
        """Plot the lane connectivity graph.

        Args:
            topo: ScenarioTopology with connectivity_graph.
            save_name: Output filename.
            figsize: Figure size.

        Returns:
            Path to saved figure, or empty string if graph is empty.
        """
        G = topo.connectivity_graph
        if G is None or len(G.nodes) == 0:
            return ""

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Use real lane positions
        pos = {}
        for node in G.nodes:
            if node in topo.lane_polylines:
                poly = topo.lane_polylines[node]
                mid = len(poly) // 2
                pos[node] = (float(poly[mid, 0]), float(poly[mid, 1]))
            else:
                pos[node] = (0, 0)

        degrees = [G.in_degree(n) + G.out_degree(n) for n in G.nodes]
        max_deg = max(degrees) if degrees else 1

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=degrees, cmap=plt.cm.YlOrRd,
            node_size=80, alpha=0.8, vmin=0, vmax=max_deg,
        )

        suc_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("type") == "suc"
        ]
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=suc_edges,
            edge_color="blue", alpha=0.5, arrows=True,
            arrowsize=10, width=0.8,
        )

        ax.set_title(
            f"Connectivity: {topo.scenario_id} | "
            f"Nodes: {len(G.nodes)} | Edges: {len(G.edges)}"
        )
        ax.set_aspect("equal")

        plt.tight_layout()
        save_path = str(self.output_dir / f"{save_name}.png")
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
