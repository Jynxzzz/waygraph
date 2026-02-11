"""
Page 1: Scenario Explorer
==========================

Browse, load, and visualize individual WOMD scenarios with their
lane graphs, agent trajectories, traffic controls, and topology summaries.
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
    load_scenario_raw,
    extract_topology,
    extract_topology_summary,
    get_lane_polylines_for_plot,
    get_objects_for_plot,
    get_road_edges_for_plot,
    get_crosswalks_for_plot,
    get_trajectory_for_plot,
    generate_demo_scenario,
)
from utils.viz import plot_scenario_topology, INTERSECTION_COLORS

st.set_page_config(page_title="Scenario Explorer - WayGraph", page_icon="🔍", layout="wide")

st.title("Scenario Explorer")
st.markdown("Load and visualize WOMD scenarios with lane graphs, agents, and traffic controls.")

# --- Sidebar controls ---
with st.sidebar:
    st.markdown("### Explorer Settings")
    show_vehicles = st.checkbox("Show Vehicles", value=True)
    show_controls = st.checkbox("Show Traffic Controls", value=True)
    show_trajectories = st.checkbox("Show AV Trajectory", value=True)

data_dir = get_data_dir()
scenario_files = list_scenario_files(data_dir)
use_demo = len(scenario_files) == 0

# --- File Selection ---
if use_demo:
    st.info("No scenario files found. Loading a synthetic demo scenario.")
    selected_file = "demo_scenario"
    scenario = generate_demo_scenario()
    scenario_id = "demo_cross_intersection"
else:
    col_select, col_search = st.columns([3, 1])
    with col_select:
        selected_file = st.selectbox(
            "Select Scenario File",
            scenario_files,
            index=0,
            help="Choose a .pkl scenario file to load",
        )
    with col_search:
        search_term = st.text_input("Filter", placeholder="Search...")
        if search_term:
            filtered = [f for f in scenario_files if search_term.lower() in f.lower()]
            if filtered:
                selected_file = st.selectbox("Filtered Results", filtered, key="filtered_select")
            else:
                st.warning("No matches.")

    # Load scenario
    scenario = load_scenario_raw(data_dir, selected_file)
    scenario_id = Path(selected_file).stem

if scenario is None:
    st.error(f"Failed to load scenario: {selected_file}")
    st.stop()

# --- Topology Extraction ---
with st.spinner("Extracting topology..."):
    topo = extract_topology(scenario, scenario_id)

if topo is None:
    st.error("Failed to extract topology from this scenario.")
    st.stop()

# --- Summary Cards ---
st.markdown("### Scenario Overview")

summary_cols = st.columns(6)
with summary_cols[0]:
    itype = topo.intersection_type
    color = INTERSECTION_COLORS.get(itype, "#95A5A6")
    st.markdown(
        f'<div style="background-color:{color}; color:white; padding:12px; '
        f'border-radius:8px; text-align:center;">'
        f'<b>{itype.upper()}</b><br><small>Intersection</small></div>',
        unsafe_allow_html=True,
    )
with summary_cols[1]:
    st.metric("Lanes", topo.num_lanes)
with summary_cols[2]:
    st.metric("Approaches", topo.num_approaches)
with summary_cols[3]:
    st.metric("Area", f"{topo.area:,.0f} m\u00b2")
with summary_cols[4]:
    tl_icon = "Yes" if topo.has_traffic_light else "No"
    st.metric("Traffic Light", tl_icon)
with summary_cols[5]:
    n_objects = len(scenario.get("objects", []))
    st.metric("Objects", n_objects)

# --- Main Visualization ---
st.markdown("### Topology Visualization")

viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Lane Graph", "Topology Details", "Raw Data"])

with viz_tab1:
    # Prepare plot data
    lanes = get_lane_polylines_for_plot(scenario)
    objects = get_objects_for_plot(scenario, timestep=0) if show_vehicles else []
    road_edges = get_road_edges_for_plot(scenario)
    crosswalks = get_crosswalks_for_plot(scenario) if show_controls else []

    centroid = (float(topo.centroid[0]), float(topo.centroid[1]))

    fig = plot_scenario_topology(
        lanes=lanes,
        objects=objects,
        road_edges=road_edges,
        crosswalks=crosswalks,
        centroid=centroid,
        title=f"Scenario: {scenario_id} ({topo.intersection_type}, {topo.num_approaches} approaches)",
        show_vehicles=show_vehicles,
        show_controls=show_controls,
        height=650,
    )

    # Add AV trajectory if requested
    if show_trajectories:
        av_idx = scenario.get("av_idx", 0)
        traj = get_trajectory_for_plot(scenario, av_idx)
        if traj:
            # Filter valid points
            valid_x = [x for x, v in zip(traj["x"], traj["valid"]) if v and x is not None]
            valid_y = [y for y, v in zip(traj["y"], traj["valid"]) if v and y is not None]
            if valid_x:
                fig.add_scatter(
                    x=valid_x, y=valid_y,
                    mode="lines",
                    line=dict(color="#FF6600", width=3, dash="dot"),
                    name="AV Trajectory",
                    opacity=0.8,
                )

    st.plotly_chart(fig, use_container_width=True)

    # Timestep slider for object positions
    n_timesteps = 0
    if scenario.get("objects"):
        n_timesteps = len(scenario["objects"][0].get("position", []))

    if n_timesteps > 1 and show_vehicles:
        timestep = st.slider(
            "Timestep",
            min_value=0,
            max_value=n_timesteps - 1,
            value=0,
            step=1,
            help="Move through time to see vehicle positions",
        )
        if timestep > 0:
            objects_t = get_objects_for_plot(scenario, timestep=timestep)
            fig_t = plot_scenario_topology(
                lanes=lanes,
                objects=objects_t,
                road_edges=road_edges,
                crosswalks=crosswalks,
                centroid=centroid,
                title=f"Timestep {timestep} (t={timestep * 0.1:.1f}s)",
                show_vehicles=True,
                show_controls=show_controls,
                height=500,
            )
            st.plotly_chart(fig_t, use_container_width=True)

with viz_tab2:
    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.markdown("#### Intersection Properties")
        props = {
            "Scenario ID": topo.scenario_id,
            "Intersection Type": topo.intersection_type,
            "Number of Approaches": topo.num_approaches,
            "Approach Angles (gaps)": [round(a, 1) for a in topo.approach_angles],
            "Approach Directions": [round(d, 1) for d in getattr(topo, "approach_directions", [])],
            "Approach Lane Counts": list(getattr(topo, "approach_lane_counts", [])),
        }
        for k, v in props.items():
            st.markdown(f"**{k}:** `{v}`")

        st.markdown("#### Traffic Controls")
        tc_data = {
            "Traffic Lights": f"{'Yes' if topo.has_traffic_light else 'No'} ({topo.num_traffic_lights})",
            "Stop Signs": f"{'Yes' if topo.has_stop_sign else 'No'} ({topo.num_stop_signs})",
            "Crosswalks": f"{'Yes' if topo.has_crosswalk else 'No'} ({topo.num_crosswalks})",
        }
        for k, v in tc_data.items():
            st.markdown(f"**{k}:** {v}")

    with detail_col2:
        st.markdown("#### Geometric Properties")
        geo = {
            "Number of Lanes": topo.num_lanes,
            "Centroid": f"({topo.centroid[0]:.1f}, {topo.centroid[1]:.1f})",
            "Bounding Box": f"({', '.join(f'{v:.1f}' for v in topo.bounding_box)})",
            "Area": f"{topo.area:,.1f} m\u00b2",
            "Dominant Direction": f"{topo.rotation_angle:.1f} deg",
            "Mean Curvature": f"{topo.mean_lane_curvature:.6f}",
        }
        for k, v in geo.items():
            st.markdown(f"**{k}:** `{v}`")

        st.markdown("#### Graph Statistics")
        if topo.connectivity_graph:
            graph_stats = {
                "Graph Nodes": len(topo.connectivity_graph.nodes),
                "Graph Edges": len(topo.connectivity_graph.edges),
                "Connected Components": topo.num_connected_components,
                "Branching Factor": f"{topo.branching_factor:.2f}",
                "Merge Factor": f"{topo.merge_factor:.2f}",
                "Degree Sequence (top 5)": topo.degree_sequence[:5],
            }
            for k, v in graph_stats.items():
                st.markdown(f"**{k}:** `{v}`")

    st.markdown("#### Feature Vector (20D)")
    vec = topo.to_feature_vector()
    vec_labels = [
        "lanes/200", "approaches/6", "curvature", "traffic_light",
        "stop_sign", "crosswalk", "n_tl/20", "n_ss/10", "n_cw/15",
        "components/10", "branching/3", "merge/3", "area/100k", "nodes/200",
        "angle_0", "angle_1", "angle_2", "angle_3", "angle_4", "angle_5",
    ]

    import plotly.graph_objects as go
    fig_vec = go.Figure(data=[go.Bar(
        x=vec_labels,
        y=vec.tolist(),
        marker_color="#4A90D9",
        text=[f"{v:.3f}" for v in vec],
        textposition="auto",
    )])
    fig_vec.update_layout(
        title="Topology Feature Vector (20D, normalized)",
        yaxis_title="Value",
        height=350,
        template="plotly_white",
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig_vec, use_container_width=True)

with viz_tab3:
    st.markdown("#### Scenario Metadata")
    objects = scenario.get("objects", [])
    vehicles = [o for o in objects if o.get("type") == "vehicle"]
    pedestrians = [o for o in objects if o.get("type") == "pedestrian"]
    cyclists = [o for o in objects if o.get("type") == "cyclist"]

    meta_df = pd.DataFrame([{
        "Total Objects": len(objects),
        "Vehicles": len(vehicles),
        "Pedestrians": len(pedestrians),
        "Cyclists": len(cyclists),
        "AV Index": scenario.get("av_idx", -1),
        "Timesteps": len(objects[0].get("position", [])) if objects else 0,
    }])
    st.dataframe(meta_df, use_container_width=True)

    st.markdown("#### Lane Graph Summary")
    lg = scenario.get("lane_graph", {})
    lg_df = pd.DataFrame([{
        "Lanes": len(lg.get("lanes", {})),
        "Successor Pairs": sum(len(v) for v in lg.get("suc_pairs", {}).values()),
        "Predecessor Pairs": sum(len(v) for v in lg.get("pre_pairs", {}).values()),
        "Left Neighbors": sum(len(v) for v in lg.get("left_pairs", {}).values()),
        "Right Neighbors": sum(len(v) for v in lg.get("right_pairs", {}).values()),
        "Road Edges": len(lg.get("road_edges", {})),
        "Crosswalks": len(lg.get("crosswalks", {})),
        "Stop Signs": len(lg.get("stop_signs", {})),
    }])
    st.dataframe(lg_df, use_container_width=True)

    with st.expander("Lane Lengths", expanded=False):
        if topo.lane_lengths:
            lengths_df = pd.DataFrame([
                {"Lane ID": k, "Length (m)": round(v, 2)}
                for k, v in sorted(topo.lane_lengths.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(lengths_df, use_container_width=True, height=300)
