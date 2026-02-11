"""
Plotly visualization helpers for the WayGraph web app.

Provides interactive chart builders for scenario topology, star patterns,
traffic analysis, and intersection statistics.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


# Color palette (consistent across the app)
LANE_COLOR = "#4A90D9"
LANE_COLOR_LIGHT = "rgba(74, 144, 217, 0.5)"
ROAD_EDGE_COLOR = "#888888"
CROSSWALK_COLOR = "#F5A623"
STOP_SIGN_COLOR = "#D0021B"
TRAFFIC_LIGHT_COLOR = "#7ED321"
VEHICLE_COLOR = "#BD10E0"
AV_COLOR = "#FF6600"
PEDESTRIAN_COLOR = "#00BCD4"
CYCLIST_COLOR = "#8BC34A"

INTERSECTION_COLORS = {
    "none": "#95A5A6",
    "merge": "#3498DB",
    "Y": "#E67E22",
    "T": "#E74C3C",
    "cross": "#2ECC71",
    "multi": "#9B59B6",
    "roundabout": "#1ABC9C",
}

TYPE_ORDER = ["none", "merge", "Y", "T", "cross", "multi", "roundabout"]


def plot_scenario_topology(
    lanes: List[Dict],
    objects: List[Dict],
    road_edges: List[Dict],
    crosswalks: List[Dict],
    centroid: Optional[Tuple[float, float]] = None,
    title: str = "Scenario Topology",
    show_vehicles: bool = True,
    show_controls: bool = True,
    height: int = 600,
) -> go.Figure:
    """Create an interactive Plotly plot of a scenario topology.

    Args:
        lanes: Lane polyline data from get_lane_polylines_for_plot.
        objects: Object positions from get_objects_for_plot.
        road_edges: Road edge data from get_road_edges_for_plot.
        crosswalks: Crosswalk data from get_crosswalks_for_plot.
        centroid: Optional (x, y) centroid marker.
        title: Plot title.
        show_vehicles: Whether to show vehicle markers.
        show_controls: Whether to show crosswalks.
        height: Figure height in pixels.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Road edges (background)
    for edge in road_edges:
        fig.add_trace(go.Scatter(
            x=edge["x"], y=edge["y"],
            mode="lines",
            line=dict(color=ROAD_EDGE_COLOR, width=1),
            opacity=0.4,
            showlegend=False,
            hoverinfo="skip",
        ))

    # Crosswalks
    if show_controls:
        for cw in crosswalks:
            fig.add_trace(go.Scatter(
                x=cw["x"], y=cw["y"],
                mode="lines",
                fill="toself",
                fillcolor="rgba(245, 166, 35, 0.2)",
                line=dict(color=CROSSWALK_COLOR, width=1),
                showlegend=False,
                name=f"Crosswalk",
                hoverinfo="name",
            ))

    # Lane polylines
    for i, lane in enumerate(lanes):
        fig.add_trace(go.Scatter(
            x=lane["x"], y=lane["y"],
            mode="lines",
            line=dict(color=LANE_COLOR, width=2),
            opacity=0.7,
            showlegend=i == 0,
            name="Lanes" if i == 0 else None,
            hovertemplate=f"Lane {lane['lane_id']}<br>x=%{{x:.1f}}<br>y=%{{y:.1f}}<extra></extra>",
        ))

    # Objects
    if show_vehicles and objects:
        vehicles = [o for o in objects if o["type"] == "vehicle" and not o["is_av"]]
        av = [o for o in objects if o["is_av"]]
        pedestrians = [o for o in objects if o["type"] == "pedestrian"]
        cyclists = [o for o in objects if o["type"] == "cyclist"]

        if vehicles:
            fig.add_trace(go.Scatter(
                x=[v["x"] for v in vehicles],
                y=[v["y"] for v in vehicles],
                mode="markers",
                marker=dict(color=VEHICLE_COLOR, size=8, symbol="square"),
                name="Vehicles",
                hovertemplate="Vehicle #%{customdata}<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
                customdata=[v["index"] for v in vehicles],
            ))

        if av:
            fig.add_trace(go.Scatter(
                x=[v["x"] for v in av],
                y=[v["y"] for v in av],
                mode="markers",
                marker=dict(color=AV_COLOR, size=12, symbol="diamond"),
                name="AV (SDC)",
                hovertemplate="Autonomous Vehicle<br>x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
            ))

        if pedestrians:
            fig.add_trace(go.Scatter(
                x=[p["x"] for p in pedestrians],
                y=[p["y"] for p in pedestrians],
                mode="markers",
                marker=dict(color=PEDESTRIAN_COLOR, size=6, symbol="circle"),
                name="Pedestrians",
            ))

        if cyclists:
            fig.add_trace(go.Scatter(
                x=[c["x"] for c in cyclists],
                y=[c["y"] for c in cyclists],
                mode="markers",
                marker=dict(color=CYCLIST_COLOR, size=7, symbol="triangle-up"),
                name="Cyclists",
            ))

    # Centroid marker
    if centroid:
        fig.add_trace(go.Scatter(
            x=[centroid[0]], y=[centroid[1]],
            mode="markers",
            marker=dict(color="red", size=14, symbol="x"),
            name="Centroid",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )

    return fig


def plot_trajectory_animation(
    lanes: List[Dict],
    objects_by_time: List[List[Dict]],
    title: str = "Trajectory Animation",
    height: int = 600,
) -> go.Figure:
    """Create a trajectory animation using Plotly frames.

    Args:
        lanes: Lane polyline data.
        objects_by_time: List of object lists, one per timestep.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure with animation frames.
    """
    fig = go.Figure()

    # Static lanes
    for lane in lanes:
        fig.add_trace(go.Scatter(
            x=lane["x"], y=lane["y"],
            mode="lines",
            line=dict(color=LANE_COLOR, width=1.5),
            opacity=0.5,
            showlegend=False,
            hoverinfo="skip",
        ))

    # Initial positions
    if objects_by_time and objects_by_time[0]:
        first_frame = objects_by_time[0]
        for obj in first_frame:
            color = AV_COLOR if obj["is_av"] else VEHICLE_COLOR
            size = 12 if obj["is_av"] else 8
            fig.add_trace(go.Scatter(
                x=[obj["x"]], y=[obj["y"]],
                mode="markers",
                marker=dict(color=color, size=size),
                showlegend=False,
            ))

    fig.update_layout(
        title=title,
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        height=height,
        template="plotly_white",
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )

    return fig


def plot_star_pattern_radar(
    star_data: Dict[str, Any],
    title: str = "Star Pattern Features",
    height: int = 450,
) -> go.Figure:
    """Create a radar/spider chart for star pattern center features.

    Args:
        star_data: Star pattern dictionary from extract_star_pattern_data.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    vector = star_data.get("vector", [])
    if len(vector) < 6:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Center features (first 6)
    center_labels = [
        "Type Code",
        "Approaches",
        "Has Signal",
        "Has Stop",
        "Has Crosswalk",
        "Arm Count",
    ]
    center_values = vector[:6]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=center_values + [center_values[0]],
        theta=center_labels + [center_labels[0]],
        fill="toself",
        fillcolor="rgba(74, 144, 217, 0.3)",
        line=dict(color=LANE_COLOR, width=2),
        name="Center Features",
    ))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.1]),
        ),
        height=height,
        template="plotly_white",
        showlegend=True,
    )

    return fig


def plot_star_pattern_compass(
    star_data: Dict[str, Any],
    title: str = "Approach Arms",
    height: int = 500,
) -> go.Figure:
    """Create a compass rose diagram showing approach arm directions.

    Args:
        star_data: Star pattern dictionary.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    arms = star_data.get("arms", [])
    if not arms:
        fig = go.Figure()
        fig.add_annotation(text="No arms data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = go.Figure()

    # Center node
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text",
        marker=dict(color=INTERSECTION_COLORS.get(star_data.get("center_type", "none"), "#95A5A6"), size=30),
        text=[star_data.get("center_type", "?")],
        textposition="middle center",
        textfont=dict(color="white", size=10),
        name="Center",
        showlegend=False,
    ))

    # Arms as directional lines
    arm_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
    for i, arm in enumerate(arms):
        angle_rad = math.radians(arm["angle"])
        length = min(arm.get("length", 100), 300) / 300 * 2.5  # Normalize to plot scale
        end_x = length * math.cos(angle_rad)
        end_y = length * math.sin(angle_rad)

        color = arm_colors[i % len(arm_colors)]

        # Arm line
        fig.add_trace(go.Scatter(
            x=[0, end_x], y=[0, end_y],
            mode="lines+markers",
            line=dict(color=color, width=3 + arm.get("lanes", 1)),
            marker=dict(size=[0, 10], color=color),
            showlegend=False,
            hovertemplate=(
                f"<b>Arm {i+1}</b><br>"
                f"Angle: {arm['angle']:.1f} deg<br>"
                f"Length: {arm.get('length', 0):.0f} m<br>"
                f"Road: {arm.get('road_type', 'unknown')}<br>"
                f"Lanes: {arm.get('lanes', 1)}<br>"
                f"Neighbor: {arm.get('neighbor_type', 'none')}"
                "<extra></extra>"
            ),
        ))

        # Arm label
        label_x = end_x * 1.15
        label_y = end_y * 1.15
        fig.add_trace(go.Scatter(
            x=[label_x], y=[label_y],
            mode="text",
            text=[f"{arm.get('road_type', '?')}<br>{arm.get('lanes', '?')}L"],
            textfont=dict(size=9, color=color),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Add N/S/E/W reference
    for label, angle in [("N", 90), ("E", 0), ("S", 270), ("W", 180)]:
        ref_r = 3.2
        rx = ref_r * math.cos(math.radians(angle))
        ry = ref_r * math.sin(math.radians(angle))
        fig.add_annotation(
            x=rx, y=ry, text=label,
            showarrow=False, font=dict(size=12, color="#AAAAAA"),
        )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        xaxis=dict(range=[-4, 4], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-4, 4], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        shapes=[
            dict(
                type="circle", x0=-3, y0=-3, x1=3, y1=3,
                line=dict(color="#DDDDDD", dash="dot"),
            ),
        ],
    )

    return fig


def plot_feature_vector_heatmap(
    star_data: Dict[str, Any],
    title: str = "48D Feature Vector",
    height: int = 200,
) -> go.Figure:
    """Create a heatmap of the 48D star pattern feature vector.

    Args:
        star_data: Star pattern dictionary with 'vector' key.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    vector = star_data.get("vector", [])
    if len(vector) < 48:
        vector = vector + [0.0] * (48 - len(vector))

    # Reshape into labeled groups
    labels = (
        ["Type", "Appr", "Signal", "Stop", "CW", "Arms"]
        + [f"A{a}_{f}" for a in range(6) for f in ["Ang", "Len", "Rd", "Ln", "NbT", "NbD", "NbS"]]
    )[:48]

    fig = go.Figure(data=go.Heatmap(
        z=[vector],
        x=labels,
        y=["Feature Value"],
        colorscale="Viridis",
        zmin=0,
        zmax=1,
        hovertemplate="%{x}: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(showticklabels=False),
    )

    return fig


def plot_star_pattern_comparison(
    star1: Dict[str, Any],
    star2: Dict[str, Any],
    label1: str = "WOMD Pattern",
    label2: str = "OSM Pattern",
    height: int = 350,
) -> go.Figure:
    """Side-by-side comparison of two star pattern feature vectors.

    Args:
        star1: First star pattern data.
        star2: Second star pattern data.
        label1: Label for first pattern.
        label2: Label for second pattern.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    v1 = star1.get("vector", [0.0] * 48)
    v2 = star2.get("vector", [0.0] * 48)

    if len(v1) < 48:
        v1 = v1 + [0.0] * (48 - len(v1))
    if len(v2) < 48:
        v2 = v2 + [0.0] * (48 - len(v2))

    feature_names = (
        ["Type", "Approaches", "Signal", "Stop", "Crosswalk", "Arms"]
        + [f"Arm{a}_{f}" for a in range(6) for f in ["Angle", "Length", "Road", "Lanes", "NbType", "NbDeg", "NbSig"]]
    )[:48]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=feature_names, y=v1,
        name=label1,
        marker_color=LANE_COLOR,
        opacity=0.7,
    ))

    fig.add_trace(go.Bar(
        x=feature_names, y=v2,
        name=label2,
        marker_color="#2ECC71",
        opacity=0.7,
    ))

    fig.update_layout(
        title="Feature Vector Comparison",
        barmode="group",
        height=height,
        template="plotly_white",
        xaxis=dict(tickangle=45, tickfont=dict(size=7)),
        yaxis_title="Normalized Value",
    )

    return fig


def plot_turning_ratios(
    turning_data: Dict[str, Dict],
    title: str = "Turning Ratios",
    height: int = 400,
) -> go.Figure:
    """Create pie charts for turning ratios per approach.

    Args:
        turning_data: Dictionary of approach_id -> turning movement data.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    if not turning_data:
        fig = go.Figure()
        fig.add_annotation(text="No turning data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=height, template="plotly_white", title=title)
        return fig

    # Aggregate turning counts
    total_left = sum(d.get("left", 0) for d in turning_data.values())
    total_through = sum(d.get("through", 0) for d in turning_data.values())
    total_right = sum(d.get("right", 0) for d in turning_data.values())
    total_uturn = sum(d.get("uturn", 0) for d in turning_data.values())

    labels = ["Left", "Through", "Right", "U-Turn"]
    values = [total_left, total_through, total_right, total_uturn]
    colors = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12"]

    # Filter out zero values
    nonzero = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not nonzero:
        fig = go.Figure()
        fig.add_annotation(text="No turning movements observed", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=height, template="plotly_white", title=title)
        return fig

    labels_f, values_f, colors_f = zip(*nonzero)

    fig = go.Figure(data=[go.Pie(
        labels=labels_f,
        values=values_f,
        marker=dict(colors=colors_f),
        hole=0.35,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} vehicles (%{percent})<extra></extra>",
    )])

    fig.update_layout(
        title=f"{title} (Total: {sum(values)} vehicles)",
        height=height,
        template="plotly_white",
    )

    return fig


def plot_per_approach_turning(
    turning_data: Dict[str, Dict],
    title: str = "Per-Approach Turning Movements",
    height: int = 400,
) -> go.Figure:
    """Create stacked bar chart of turning movements per approach.

    Args:
        turning_data: Dictionary of approach_id -> turning movement data.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    if not turning_data:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=height, template="plotly_white")
        return fig

    approaches = list(turning_data.keys())
    left_vals = [turning_data[a].get("left", 0) for a in approaches]
    through_vals = [turning_data[a].get("through", 0) for a in approaches]
    right_vals = [turning_data[a].get("right", 0) for a in approaches]
    uturn_vals = [turning_data[a].get("uturn", 0) for a in approaches]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Left", x=approaches, y=left_vals, marker_color="#E74C3C"))
    fig.add_trace(go.Bar(name="Through", x=approaches, y=through_vals, marker_color="#2ECC71"))
    fig.add_trace(go.Bar(name="Right", x=approaches, y=right_vals, marker_color="#3498DB"))
    fig.add_trace(go.Bar(name="U-Turn", x=approaches, y=uturn_vals, marker_color="#F39C12"))

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Approach",
        yaxis_title="Vehicle Count",
        height=height,
        template="plotly_white",
    )

    return fig


def plot_speed_distributions(
    speed_data: Dict[str, Dict],
    title: str = "Speed Distributions",
    height: int = 400,
) -> go.Figure:
    """Create a box plot or bar chart of speed distributions by lane.

    Args:
        speed_data: Dictionary of lane_id -> speed distribution data.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    if not speed_data:
        fig = go.Figure()
        fig.add_annotation(text="No speed data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=height, template="plotly_white", title=title)
        return fig

    # Sort by mean speed
    sorted_lanes = sorted(speed_data.items(), key=lambda x: x[1].get("mean_speed_kmh", 0), reverse=True)
    # Show top 20 lanes
    sorted_lanes = sorted_lanes[:20]

    lane_ids = [f"Lane {lid}" for lid, _ in sorted_lanes]
    mean_speeds = [d.get("mean_speed_kmh", 0) for _, d in sorted_lanes]
    ff_speeds = [d.get("free_flow_speed_kmh", 0) for _, d in sorted_lanes]
    n_samples = [d.get("n_samples", 0) for _, d in sorted_lanes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=lane_ids, y=mean_speeds,
        name="Mean Speed",
        marker_color=LANE_COLOR,
        text=[f"{s:.1f}" for s in mean_speeds],
        textposition="auto",
        hovertemplate="%{x}<br>Mean: %{y:.1f} km/h<br>Samples: %{customdata}<extra></extra>",
        customdata=n_samples,
    ))

    fig.add_trace(go.Scatter(
        x=lane_ids, y=ff_speeds,
        mode="markers+lines",
        name="Free-Flow (85th pctl)",
        marker=dict(color="#E74C3C", size=8),
        line=dict(color="#E74C3C", dash="dash"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Lane",
        yaxis_title="Speed (km/h)",
        height=height,
        template="plotly_white",
        xaxis=dict(tickangle=45),
    )

    return fig


def plot_gap_acceptance(
    gap_data: Dict[str, Any],
    title: str = "Gap Acceptance Analysis",
    height: int = 400,
) -> go.Figure:
    """Create a histogram of accepted and rejected gaps.

    Args:
        gap_data: Gap acceptance data dictionary.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    accepted = gap_data.get("accepted_gaps", [])
    rejected = gap_data.get("rejected_gaps", [])

    if not accepted and not rejected:
        fig = go.Figure()
        fig.add_annotation(text="No gap acceptance data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=height, template="plotly_white", title=title)
        return fig

    fig = go.Figure()

    if accepted:
        fig.add_trace(go.Histogram(
            x=accepted,
            name=f"Accepted (n={len(accepted)})",
            marker_color="#2ECC71",
            opacity=0.7,
            nbinsx=20,
        ))

    if rejected:
        fig.add_trace(go.Histogram(
            x=rejected,
            name=f"Rejected (n={len(rejected)})",
            marker_color="#E74C3C",
            opacity=0.7,
            nbinsx=20,
        ))

    # Add critical gap line
    critical = gap_data.get("critical_gap_s", 0)
    if critical > 0:
        fig.add_vline(
            x=critical,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Critical gap: {critical:.1f}s",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Gap Duration (seconds)",
        yaxis_title="Frequency",
        barmode="overlay",
        height=height,
        template="plotly_white",
    )

    return fig


def plot_intersection_type_distribution(
    df: pd.DataFrame,
    title: str = "Intersection Type Distribution",
    height: int = 400,
) -> go.Figure:
    """Create a bar chart of intersection type counts.

    Args:
        df: DataFrame with 'intersection_type' column.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    if df.empty or "intersection_type" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    type_counts = df["intersection_type"].value_counts()
    # Order by TYPE_ORDER
    ordered_types = [t for t in TYPE_ORDER if t in type_counts.index]
    ordered_counts = [type_counts[t] for t in ordered_types]
    colors = [INTERSECTION_COLORS.get(t, "#95A5A6") for t in ordered_types]

    fig = go.Figure(data=[go.Bar(
        x=ordered_types,
        y=ordered_counts,
        marker_color=colors,
        text=ordered_counts,
        textposition="auto",
        hovertemplate="%{x}: %{y} scenarios<extra></extra>",
    )])

    fig.update_layout(
        title=title,
        xaxis_title="Intersection Type",
        yaxis_title="Count",
        height=height,
        template="plotly_white",
    )

    return fig


def plot_approach_distribution(
    df: pd.DataFrame,
    title: str = "Number of Approaches",
    height: int = 350,
) -> go.Figure:
    """Create a histogram of approach counts.

    Args:
        df: DataFrame with 'num_approaches' column.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    if df.empty or "num_approaches" not in df.columns:
        fig = go.Figure()
        return fig

    fig = px.histogram(
        df, x="num_approaches",
        nbins=10,
        color_discrete_sequence=[LANE_COLOR],
        title=title,
    )
    fig.update_layout(
        xaxis_title="Number of Approaches",
        yaxis_title="Count",
        height=height,
        template="plotly_white",
    )
    return fig


def plot_traffic_control_presence(
    df: pd.DataFrame,
    title: str = "Traffic Control Presence",
    height: int = 350,
) -> go.Figure:
    """Create a horizontal bar chart showing traffic control counts.

    Args:
        df: DataFrame with boolean columns for traffic controls.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    if df.empty:
        fig = go.Figure()
        return fig

    total = len(df)
    tl = df["has_traffic_light"].sum() if "has_traffic_light" in df.columns else 0
    ss = df["has_stop_sign"].sum() if "has_stop_sign" in df.columns else 0
    cw = df["has_crosswalk"].sum() if "has_crosswalk" in df.columns else 0

    labels = ["Traffic Light", "Stop Sign", "Crosswalk"]
    values = [int(tl), int(ss), int(cw)]
    colors = [TRAFFIC_LIGHT_COLOR, STOP_SIGN_COLOR, CROSSWALK_COLOR]
    pcts = [f"{100*v/total:.0f}%" if total > 0 else "0%" for v in values]

    fig = go.Figure(data=[go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v} ({p})" for v, p in zip(values, pcts)],
        textposition="auto",
    )])

    fig.update_layout(
        title=title,
        xaxis_title="Count",
        height=height,
        template="plotly_white",
    )

    return fig


def plot_lane_vs_area_scatter(
    df: pd.DataFrame,
    title: str = "Lanes vs. Area",
    height: int = 400,
) -> go.Figure:
    """Create a scatter plot of lane count vs scenario area.

    Args:
        df: DataFrame with 'num_lanes' and 'area_m2' columns.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    if df.empty:
        fig = go.Figure()
        return fig

    color_col = "intersection_type" if "intersection_type" in df.columns else None

    fig = px.scatter(
        df, x="num_lanes", y="area_m2",
        color=color_col,
        color_discrete_map=INTERSECTION_COLORS if color_col else None,
        hover_data=["scenario_id", "num_approaches"] if "scenario_id" in df.columns else None,
        title=title,
    )

    fig.update_layout(
        xaxis_title="Number of Lanes",
        yaxis_title="Scenario Area (m^2)",
        height=height,
        template="plotly_white",
    )

    return fig


def plot_match_scores(
    scores: List[Tuple[str, float]],
    title: str = "Top-K Match Scores",
    height: int = 350,
) -> go.Figure:
    """Create a bar chart of match scores for top-K results.

    Args:
        scores: List of (id, score) tuples.
        title: Plot title.
        height: Figure height.

    Returns:
        Plotly Figure.
    """
    if not scores:
        fig = go.Figure()
        fig.add_annotation(text="No match results", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    ids = [s[0] for s in scores]
    vals = [s[1] for s in scores]

    # Color by score
    colors = [f"rgba(74, 144, 217, {max(0.3, v)})" for v in vals]

    fig = go.Figure(data=[go.Bar(
        x=ids,
        y=vals,
        marker_color=colors,
        text=[f"{v:.3f}" for v in vals],
        textposition="auto",
        hovertemplate="ID: %{x}<br>Score: %{y:.4f}<extra></extra>",
    )])

    fig.update_layout(
        title=title,
        xaxis_title="OSM Node ID",
        yaxis_title="Similarity Score",
        yaxis=dict(range=[0, 1.05]),
        height=height,
        template="plotly_white",
        xaxis=dict(tickangle=45),
    )

    return fig
