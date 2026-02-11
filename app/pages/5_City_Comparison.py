"""
Page 5: City Comparison
========================

Compare intersection topology statistics across WOMD cities
(San Francisco, Phoenix) and potentially other regions.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Setup imports
APP_ROOT = Path(__file__).resolve().parent.parent
WAYGRAPH_ROOT = APP_ROOT.parent
if str(WAYGRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(WAYGRAPH_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from utils.data_loader import (
    get_data_dir,
    list_scenario_files,
    extract_batch_summaries,
)
from utils.viz import (
    plot_intersection_type_distribution,
    plot_approach_distribution,
    plot_traffic_control_presence,
    plot_lane_vs_area_scatter,
    INTERSECTION_COLORS,
    TYPE_ORDER,
)
from utils.map_viz import create_city_overview_map, CITY_CENTERS, HAS_FOLIUM

import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="City Comparison - WayGraph", page_icon="🌍", layout="wide")

st.title("City Comparison")
st.markdown(
    "Compare intersection topology distributions and traffic parameters "
    "across different cities covered by the WOMD dataset."
)

# --- WOMD Coverage Info ---
with st.expander("WOMD Geographic Coverage", expanded=False):
    st.markdown("""
    The Waymo Open Motion Dataset covers scenarios from:
    - **San Francisco, CA** - Dense urban grid with frequent signalized intersections
    - **Phoenix, AZ** - Suburban sprawl with wider roads and more multi-lane approaches
    - **Other locations** - Some scenarios from additional test markets

    Note: WOMD does not include explicit city labels. The `.pkl` scenario files in
    the training set are a mixed sample. The city can be inferred through
    OSM matching (matching the scenario's star pattern to OSM intersections
    in each city and checking which city gives the best match).
    """)

# --- Load data ---
data_dir = get_data_dir()
scenario_files = list_scenario_files(data_dir)

if not scenario_files:
    st.info("Comparing sample city data (San Francisco vs Phoenix).")

    # Generate synthetic city data
    np.random.seed(42)
    n_sf = 150
    n_phx = 100

    sf_data = []
    for i in range(n_sf):
        itype = np.random.choice(["cross", "T", "Y", "merge", "multi", "none"],
                                  p=[0.35, 0.25, 0.10, 0.15, 0.05, 0.10])
        approaches = {"cross": 4, "T": 3, "Y": 3, "merge": 2, "multi": 5, "none": 0}[itype]
        sf_data.append({
            "scenario_id": f"sf_{i:03d}",
            "intersection_type": itype,
            "num_approaches": approaches + int(np.random.choice([-1, 0, 0, 0, 1])),
            "num_lanes": int(np.random.normal(80, 30)),
            "has_traffic_light": bool(np.random.random() < 0.65),
            "has_stop_sign": bool(np.random.random() < 0.30),
            "has_crosswalk": bool(np.random.random() < 0.55),
            "area_m2": float(np.random.lognormal(9.5, 0.8)),
            "mean_curvature": float(np.random.exponential(0.003)),
            "branching_factor": float(np.random.normal(1.5, 0.3)),
            "merge_factor": float(np.random.normal(1.4, 0.3)),
            "city": "San Francisco",
        })

    phx_data = []
    for i in range(n_phx):
        itype = np.random.choice(["cross", "T", "Y", "merge", "multi", "none"],
                                  p=[0.40, 0.20, 0.05, 0.20, 0.05, 0.10])
        approaches = {"cross": 4, "T": 3, "Y": 3, "merge": 2, "multi": 5, "none": 0}[itype]
        phx_data.append({
            "scenario_id": f"phx_{i:03d}",
            "intersection_type": itype,
            "num_approaches": approaches + int(np.random.choice([-1, 0, 0, 0, 1])),
            "num_lanes": int(np.random.normal(100, 35)),
            "has_traffic_light": bool(np.random.random() < 0.70),
            "has_stop_sign": bool(np.random.random() < 0.20),
            "has_crosswalk": bool(np.random.random() < 0.40),
            "area_m2": float(np.random.lognormal(10.0, 0.7)),
            "mean_curvature": float(np.random.exponential(0.002)),
            "branching_factor": float(np.random.normal(1.6, 0.3)),
            "merge_factor": float(np.random.normal(1.5, 0.3)),
            "city": "Phoenix",
        })

    df_sf = pd.DataFrame(sf_data)
    df_phx = pd.DataFrame(phx_data)
    df_all = pd.concat([df_sf, df_phx], ignore_index=True)
    demo_mode = True

else:
    # Use real data - split into two groups for comparison
    st.markdown("### Data Configuration")
    st.markdown(
        "Since WOMD does not label scenarios by city, we split the dataset "
        "into two groups for comparative analysis."
    )

    total_files = len(scenario_files)
    n_analyze = st.slider(
        "Total scenarios to analyze",
        min_value=10,
        max_value=min(total_files, 100),
        value=min(total_files, 40),
        step=10,
    )

    split_point = n_analyze // 2
    st.markdown(f"**Group A:** first {split_point} files | **Group B:** next {n_analyze - split_point} files")

    with st.spinner("Extracting topologies..."):
        df_a = extract_batch_summaries(data_dir, scenario_files[:split_point], max_files=split_point)
        df_b = extract_batch_summaries(data_dir, scenario_files[split_point:n_analyze], max_files=n_analyze - split_point)

    if not df_a.empty:
        df_a["city"] = "Group A"
    if not df_b.empty:
        df_b["city"] = "Group B"

    df_sf = df_a
    df_phx = df_b
    df_all = pd.concat([df_sf, df_phx], ignore_index=True) if not df_sf.empty or not df_phx.empty else pd.DataFrame()
    demo_mode = False

# --- City Maps ---
if demo_mode and HAS_FOLIUM:
    st.markdown("### City Locations")
    map_col1, map_col2 = st.columns(2)
    with map_col1:
        st.markdown("**San Francisco, CA**")
        from streamlit_folium import st_folium
        m_sf = create_city_overview_map("San Francisco, CA")
        if m_sf:
            st_folium(m_sf, width=None, height=300, returned_objects=[])
    with map_col2:
        st.markdown("**Phoenix, AZ**")
        m_phx = create_city_overview_map("Phoenix, AZ")
        if m_phx:
            st_folium(m_phx, width=None, height=300, returned_objects=[])

# --- Summary Metrics ---
st.markdown("---")
st.markdown("### Summary Statistics")

city_label_a = "San Francisco" if demo_mode else "Group A"
city_label_b = "Phoenix" if demo_mode else "Group B"

met_cols = st.columns(2)

with met_cols[0]:
    st.markdown(f"#### {city_label_a}")
    if not df_sf.empty:
        a_col1, a_col2, a_col3 = st.columns(3)
        with a_col1:
            st.metric("Scenarios", len(df_sf))
        with a_col2:
            avg_lanes = df_sf["num_lanes"].mean() if "num_lanes" in df_sf.columns else 0
            st.metric("Avg Lanes", f"{avg_lanes:.0f}")
        with a_col3:
            tl_pct = df_sf["has_traffic_light"].mean() * 100 if "has_traffic_light" in df_sf.columns else 0
            st.metric("Traffic Light %", f"{tl_pct:.0f}%")
    else:
        st.info("No data")

with met_cols[1]:
    st.markdown(f"#### {city_label_b}")
    if not df_phx.empty:
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            st.metric("Scenarios", len(df_phx))
        with b_col2:
            avg_lanes = df_phx["num_lanes"].mean() if "num_lanes" in df_phx.columns else 0
            st.metric("Avg Lanes", f"{avg_lanes:.0f}")
        with b_col3:
            tl_pct = df_phx["has_traffic_light"].mean() * 100 if "has_traffic_light" in df_phx.columns else 0
            st.metric("Traffic Light %", f"{tl_pct:.0f}%")
    else:
        st.info("No data")

# --- Side-by-side Comparison Charts ---
st.markdown("---")
st.markdown("### Intersection Type Comparison")

if not df_all.empty and "intersection_type" in df_all.columns:
    # Grouped bar chart
    type_counts_a = df_sf["intersection_type"].value_counts() if not df_sf.empty else pd.Series(dtype=int)
    type_counts_b = df_phx["intersection_type"].value_counts() if not df_phx.empty else pd.Series(dtype=int)

    all_types = sorted(set(type_counts_a.index) | set(type_counts_b.index),
                       key=lambda t: TYPE_ORDER.index(t) if t in TYPE_ORDER else 99)

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name=city_label_a,
        x=all_types,
        y=[type_counts_a.get(t, 0) for t in all_types],
        marker_color="#4A90D9",
        text=[type_counts_a.get(t, 0) for t in all_types],
        textposition="auto",
    ))
    fig_comp.add_trace(go.Bar(
        name=city_label_b,
        x=all_types,
        y=[type_counts_b.get(t, 0) for t in all_types],
        marker_color="#E74C3C",
        text=[type_counts_b.get(t, 0) for t in all_types],
        textposition="auto",
    ))
    fig_comp.update_layout(
        title="Intersection Types by City/Group",
        barmode="group",
        xaxis_title="Intersection Type",
        yaxis_title="Count",
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Normalized comparison
    fig_norm = go.Figure()
    n_a = len(df_sf) if not df_sf.empty else 1
    n_b = len(df_phx) if not df_phx.empty else 1
    fig_norm.add_trace(go.Bar(
        name=city_label_a,
        x=all_types,
        y=[100 * type_counts_a.get(t, 0) / n_a for t in all_types],
        marker_color="#4A90D9",
    ))
    fig_norm.add_trace(go.Bar(
        name=city_label_b,
        x=all_types,
        y=[100 * type_counts_b.get(t, 0) / n_b for t in all_types],
        marker_color="#E74C3C",
    ))
    fig_norm.update_layout(
        title="Intersection Type Distribution (Normalized %)",
        barmode="group",
        xaxis_title="Intersection Type",
        yaxis_title="Percentage (%)",
        height=350,
        template="plotly_white",
    )
    st.plotly_chart(fig_norm, use_container_width=True)

# --- Additional Comparisons ---
st.markdown("### Detailed Comparisons")

comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs([
    "Lane Count",
    "Scenario Area",
    "Traffic Controls",
    "Graph Metrics",
])

with comp_tab1:
    if not df_all.empty and "num_lanes" in df_all.columns:
        fig = go.Figure()
        if not df_sf.empty:
            fig.add_trace(go.Histogram(
                x=df_sf["num_lanes"],
                name=city_label_a,
                marker_color="#4A90D9",
                opacity=0.7,
                nbinsx=25,
            ))
        if not df_phx.empty:
            fig.add_trace(go.Histogram(
                x=df_phx["num_lanes"],
                name=city_label_b,
                marker_color="#E74C3C",
                opacity=0.7,
                nbinsx=25,
            ))
        fig.update_layout(
            title="Lane Count Distribution",
            barmode="overlay",
            xaxis_title="Number of Lanes",
            yaxis_title="Count",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stats table
        stats_data = []
        for label, df_city in [(city_label_a, df_sf), (city_label_b, df_phx)]:
            if not df_city.empty and "num_lanes" in df_city.columns:
                stats_data.append({
                    "City": label,
                    "Mean Lanes": round(df_city["num_lanes"].mean(), 1),
                    "Median Lanes": df_city["num_lanes"].median(),
                    "Std Lanes": round(df_city["num_lanes"].std(), 1),
                    "Min": df_city["num_lanes"].min(),
                    "Max": df_city["num_lanes"].max(),
                })
        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

with comp_tab2:
    if not df_all.empty and "area_m2" in df_all.columns:
        fig = go.Figure()
        if not df_sf.empty:
            fig.add_trace(go.Histogram(
                x=df_sf["area_m2"],
                name=city_label_a,
                marker_color="#4A90D9",
                opacity=0.7,
                nbinsx=25,
            ))
        if not df_phx.empty:
            fig.add_trace(go.Histogram(
                x=df_phx["area_m2"],
                name=city_label_b,
                marker_color="#E74C3C",
                opacity=0.7,
                nbinsx=25,
            ))
        fig.update_layout(
            title="Scenario Area Distribution",
            barmode="overlay",
            xaxis_title="Area (m\u00b2)",
            yaxis_title="Count",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

with comp_tab3:
    if not df_all.empty:
        controls = ["has_traffic_light", "has_stop_sign", "has_crosswalk"]
        available_controls = [c for c in controls if c in df_all.columns]

        if available_controls:
            labels = [c.replace("has_", "").replace("_", " ").title() for c in available_controls]
            a_vals = [df_sf[c].mean() * 100 if not df_sf.empty and c in df_sf.columns else 0 for c in available_controls]
            b_vals = [df_phx[c].mean() * 100 if not df_phx.empty and c in df_phx.columns else 0 for c in available_controls]

            fig = go.Figure()
            fig.add_trace(go.Bar(name=city_label_a, x=labels, y=a_vals, marker_color="#4A90D9"))
            fig.add_trace(go.Bar(name=city_label_b, x=labels, y=b_vals, marker_color="#E74C3C"))
            fig.update_layout(
                title="Traffic Control Presence (%)",
                barmode="group",
                yaxis_title="Percentage (%)",
                height=350,
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

with comp_tab4:
    if not df_all.empty:
        graph_cols = ["branching_factor", "merge_factor"]
        available_graph = [c for c in graph_cols if c in df_all.columns]

        if available_graph and "branching_factor" in df_all.columns and "merge_factor" in df_all.columns:
            fig = go.Figure()
            if not df_sf.empty:
                fig.add_trace(go.Scatter(
                    x=df_sf["branching_factor"],
                    y=df_sf["merge_factor"],
                    mode="markers",
                    name=city_label_a,
                    marker=dict(color="#4A90D9", size=8, opacity=0.6),
                ))
            if not df_phx.empty:
                fig.add_trace(go.Scatter(
                    x=df_phx["branching_factor"],
                    y=df_phx["merge_factor"],
                    mode="markers",
                    name=city_label_b,
                    marker=dict(color="#E74C3C", size=8, opacity=0.6),
                ))
            fig.add_trace(go.Scatter(
                x=[0, 3], y=[0, 3],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                showlegend=False,
            ))
            fig.update_layout(
                title="Branching Factor vs. Merge Factor",
                xaxis_title="Branching Factor",
                yaxis_title="Merge Factor",
                height=400,
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Combined Data Table ---
st.markdown("---")
with st.expander("Full Data Table", expanded=False):
    if not df_all.empty:
        display_cols = [c for c in [
            "city", "scenario_id", "intersection_type", "num_lanes",
            "num_approaches", "has_traffic_light", "has_stop_sign",
            "has_crosswalk", "area_m2",
        ] if c in df_all.columns]
        st.dataframe(df_all[display_cols], use_container_width=True, height=400)
    else:
        st.info("No data to display.")
