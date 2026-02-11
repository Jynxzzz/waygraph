"""
Match Result Visualization
============================

Visualize WOMD-to-OSM matching results: side-by-side scenario overlays,
match score distributions, and geographic match maps.

Example::

    from waygraph.viz import MatchVisualizer

    viz = MatchVisualizer(output_dir="figures/")
    viz.plot_match(topo, osm_polylines, save_name="match_001")
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class MatchVisualizer:
    """Visualization tools for scenario-to-OSM matching results.

    Args:
        output_dir: Directory to save figures.
        dpi: Resolution for saved figures.
    """

    def __init__(self, output_dir: str = "./figures", dpi: int = 150):
        if not HAS_MPL:
            raise ImportError(
                "matplotlib is required. Install with: pip install waygraph[viz]"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_match(
        self,
        topo: "ScenarioTopology",
        osm_polylines: Optional[Dict] = None,
        match_info: Optional[Dict] = None,
        save_name: str = "matched",
        figsize: Tuple[float, float] = (14, 7),
    ) -> str:
        """Plot a WOMD scenario alongside its matched OSM location.

        Args:
            topo: ScenarioTopology to visualize.
            osm_polylines: Dict of OSM edge -> (N, 2) polyline arrays
                in local coordinates.
            match_info: Optional dict with match metadata (score, lat, lon).
            save_name: Output filename.
            figsize: Figure size.

        Returns:
            Path to saved figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left panel: WOMD scenario
        for lane_id, polyline in topo.lane_polylines.items():
            if len(polyline) < 2:
                continue
            centered = polyline - topo.centroid
            ax1.plot(
                centered[:, 0], centered[:, 1],
                color="#4A90D9", linewidth=1, alpha=0.7,
            )
        ax1.plot(0, 0, "rx", markersize=10, markeredgewidth=2)
        ax1.set_title(f"WOMD: {topo.scenario_id}")
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)

        # Right panel: OSM match
        if osm_polylines is not None:
            for edge_id, polyline in osm_polylines.items():
                if len(polyline) >= 2:
                    ax2.plot(
                        polyline[:, 0], polyline[:, 1],
                        color="green", linewidth=1.5, alpha=0.7,
                    )
            ax2.plot(0, 0, "rx", markersize=10, markeredgewidth=2)
            ax2.set_title("OSM Match")
            ax2.set_aspect("equal")
            ax2.grid(True, alpha=0.3)
        elif match_info is not None:
            info_text = "\n".join(
                f"{k}: {v}" for k, v in match_info.items()
            )
            ax2.text(
                0.1, 0.5, info_text,
                transform=ax2.transAxes, fontsize=10,
                verticalalignment="center", family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
            ax2.set_title("Match Details")
            ax2.axis("off")

        plt.suptitle("Scenario-to-OSM Match", fontsize=12)
        plt.tight_layout()

        save_path = str(self.output_dir / f"{save_name}.png")
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_score_distribution(
        self,
        scores: List[float],
        save_name: str = "score_distribution",
        figsize: Tuple[float, float] = (8, 5),
    ) -> str:
        """Plot the distribution of match scores.

        Args:
            scores: List of similarity scores.
            save_name: Output filename.
            figsize: Figure size.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.hist(scores, bins=40, color="#4A90D9", alpha=0.7, density=True)
        ax.axvline(
            np.mean(scores), color="red", linestyle="--",
            label=f"Mean: {np.mean(scores):.3f}",
        )
        ax.axvline(
            np.median(scores), color="orange", linestyle="--",
            label=f"Median: {np.median(scores):.3f}",
        )
        ax.set_xlabel("Similarity Score")
        ax.set_ylabel("Density")
        ax.set_title("Match Score Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = str(self.output_dir / f"{save_name}.png")
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
