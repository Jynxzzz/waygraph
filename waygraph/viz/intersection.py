"""
Intersection Visualization
============================

Plot intersection type distributions, approach analysis, traffic control
presence, and batch topology summaries.

Example::

    from waygraph.viz import IntersectionVisualizer

    viz = IntersectionVisualizer(output_dir="figures/")
    viz.plot_type_distribution(topologies, save_name="types")
    viz.plot_batch_summary(topologies, save_name="summary")
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

APPROACH_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"
]


class IntersectionVisualizer:
    """Visualization tools for intersection analysis results.

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

    def plot_type_distribution(
        self,
        topos: List,
        save_name: str = "intersection_types",
        figsize: Tuple[float, float] = (14, 8),
    ) -> str:
        """Plot intersection type distribution and approach analysis.

        Creates a 2x2 grid showing:
            - Intersection type counts
            - Number of approaches distribution
            - Traffic control presence
            - Lane count vs area scatter

        Args:
            topos: List of ScenarioTopology objects.
            save_name: Output filename.
            figsize: Figure size.

        Returns:
            Path to saved figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        types = [t.intersection_type for t in topos]
        type_counts: Dict[str, int] = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1

        labels = list(type_counts.keys())
        values = list(type_counts.values())
        axes[0, 0].bar(labels, values, color=APPROACH_COLORS[: len(labels)])
        axes[0, 0].set_title("Intersection Type Distribution")
        axes[0, 0].set_ylabel("Count")

        # Approaches
        approaches = [t.num_approaches for t in topos]
        approach_counts: Dict[int, int] = {}
        for a in approaches:
            approach_counts[a] = approach_counts.get(a, 0) + 1
        a_labels = sorted(approach_counts.keys())
        a_values = [approach_counts[a] for a in a_labels]
        axes[0, 1].bar([str(a) for a in a_labels], a_values, color="#4A90D9")
        axes[0, 1].set_title("Number of Approaches")

        # Traffic controls
        total = len(topos)
        tl = sum(1 for t in topos if t.has_traffic_light)
        ss = sum(1 for t in topos if t.has_stop_sign)
        cw = sum(1 for t in topos if t.has_crosswalk)
        ctrl_labels = ["Traffic Light", "Stop Sign", "Crosswalk"]
        ctrl_values = [tl, ss, cw]
        ctrl_colors = ["#7ED321", "#D0021B", "#F5A623"]
        axes[1, 0].barh(ctrl_labels, ctrl_values, color=ctrl_colors)
        axes[1, 0].set_title("Traffic Control Presence")
        for i, v in enumerate(ctrl_values):
            axes[1, 0].text(
                v + 0.5, i, f"{v} ({100 * v / total:.0f}%)",
                va="center", fontsize=9,
            )

        # Lane count vs area
        lane_counts = [t.num_lanes for t in topos]
        areas = [t.area for t in topos]
        scatter = axes[1, 1].scatter(
            lane_counts, areas,
            c=[t.num_approaches for t in topos],
            cmap="viridis", alpha=0.6, s=30,
        )
        axes[1, 1].set_xlabel("Number of Lanes")
        axes[1, 1].set_ylabel("Area (m^2)")
        axes[1, 1].set_title("Lane Count vs. Area")
        plt.colorbar(scatter, ax=axes[1, 1], label="Approaches")

        plt.suptitle(
            f"Intersection Analysis ({len(topos)} scenarios)", fontsize=12
        )
        plt.tight_layout()

        save_path = str(self.output_dir / f"{save_name}.png")
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_batch_summary(
        self,
        topos: List,
        save_name: str = "batch_summary",
        figsize: Tuple[float, float] = (16, 12),
    ) -> str:
        """Plot a comprehensive summary of batch topology extraction.

        Creates a 2x3 grid with type distribution, lane counts,
        curvature, branching/merge factors, area, and PCA projection.

        Args:
            topos: List of ScenarioTopology objects.
            save_name: Output filename.
            figsize: Figure size.

        Returns:
            Path to saved figure.
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        types = [t.intersection_type for t in topos]
        type_counts: Dict[str, int] = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        axes[0, 0].bar(type_counts.keys(), type_counts.values(), color="#4A90D9")
        axes[0, 0].set_title("Intersection Types")

        lane_counts = [t.num_lanes for t in topos]
        axes[0, 1].hist(lane_counts, bins=30, color="#4A90D9", alpha=0.7)
        axes[0, 1].set_title("Lane Count Distribution")

        curvatures = [
            t.mean_lane_curvature for t in topos if t.mean_lane_curvature > 0
        ]
        if curvatures:
            axes[0, 2].hist(curvatures, bins=30, color="orange", alpha=0.7)
            axes[0, 2].set_title("Mean Curvature")

        branching = [t.branching_factor for t in topos]
        merging = [t.merge_factor for t in topos]
        axes[1, 0].scatter(branching, merging, alpha=0.5, s=20, color="#4A90D9")
        axes[1, 0].set_xlabel("Branching Factor")
        axes[1, 0].set_ylabel("Merge Factor")
        axes[1, 0].set_title("Branching vs Merge")
        axes[1, 0].plot([0, 3], [0, 3], "r--", alpha=0.3)

        areas = [t.area for t in topos if t.area > 0]
        if areas:
            axes[1, 1].hist(areas, bins=30, color="green", alpha=0.7)
            axes[1, 1].set_title("Scenario Area (m^2)")

        # PCA projection
        if len(topos) >= 3:
            try:
                features = np.array([t.to_feature_vector() for t in topos])
                mean = features.mean(axis=0)
                centered = features - mean
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                idx = eigenvalues.argsort()[::-1]
                pc1, pc2 = eigenvectors[:, idx[0]], eigenvectors[:, idx[1]]
                proj = centered @ np.column_stack([pc1, pc2])

                type_to_num = {t: i for i, t in enumerate(set(types))}
                colors = [type_to_num[t] for t in types]
                axes[1, 2].scatter(
                    proj[:, 0], proj[:, 1], c=colors,
                    cmap="tab10", alpha=0.6, s=20,
                )
                axes[1, 2].set_title("Feature Space (PCA)")
            except Exception:
                axes[1, 2].text(
                    0.5, 0.5, "PCA unavailable",
                    ha="center", va="center", transform=axes[1, 2].transAxes,
                )

        plt.suptitle(
            f"Topology Summary ({len(topos)} scenarios)", fontsize=13
        )
        plt.tight_layout()

        save_path = str(self.output_dir / f"{save_name}.png")
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
