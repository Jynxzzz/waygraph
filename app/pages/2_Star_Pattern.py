"""
Page 2: Star Pattern Viewer
=============================

Visual representation of the 48D star pattern fingerprint, including
radar charts for center features, compass rose arm diagrams,
feature vector heatmaps, and side-by-side pattern comparisons.
"""

import sys
from pathlib import Path

import streamlit as st
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
    extract_star_pattern_data,
    generate_demo_scenario,
    generate_demo_star_patterns,
)
from utils.viz import (
    plot_star_pattern_radar,
    plot_star_pattern_compass,
    plot_feature_vector_heatmap,
    plot_star_pattern_comparison,
    INTERSECTION_COLORS,
)

from waygraph.fingerprint import StarPattern, star_distance

st.set_page_config(page_title="Star Pattern Viewer - WayGraph", page_icon="⭐", layout="wide")

st.title("Star Pattern Viewer")
st.markdown(
    "Visualize the **48-dimensional star pattern fingerprint** that encodes "
    "intersection topology and 1-hop neighborhood context."
)

# --- Information Box ---
with st.expander("How Star Patterns Work", expanded=False):
    st.markdown("""
    **Star Pattern Fingerprinting** adapts GIS map conflation techniques for
    autonomous driving dataset localization.

    The 48D feature vector captures:
    - **Center features (6D):** Intersection type, approach count, signal/stop/crosswalk flags, arm count
    - **Per-arm features (7D x 6 arms = 42D):** Direction angle, road length, road type, lane count,
      neighbor intersection type, neighbor degree, neighbor signal status

    ```
    Vector layout:
    [0:6]   Center:  type, approaches, signal, stop, crosswalk, num_arms
    [6:13]  Arm 0:   angle, length, road_type, lanes, nb_type, nb_degree, nb_signal
    [13:20] Arm 1:   ...
    [20:27] Arm 2:   ...
    [27:34] Arm 3:   ...
    [34:41] Arm 4:   ...
    [41:48] Arm 5:   ...
    ```
    """)

# --- Load data ---
data_dir = get_data_dir()
scenario_files = list_scenario_files(data_dir)
use_demo = len(scenario_files) == 0

st.markdown("---")

# --- Pattern Selection ---
st.markdown("### Select Scenario")

if use_demo:
    st.info("Using demo data. Connect real scenario files for full functionality.")
    scenario = generate_demo_scenario()
    scenario_id = "demo_cross_intersection"
    selected_file = "demo"
else:
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        selected_file = st.selectbox(
            "Scenario File",
            scenario_files,
            index=0,
            key="star_file_select",
        )
    with col_sel2:
        st.markdown("")  # spacing
        st.markdown(f"**{len(scenario_files)}** files available")

    scenario = load_scenario_raw(data_dir, selected_file)
    scenario_id = Path(selected_file).stem

if scenario is None:
    st.error("Failed to load scenario.")
    st.stop()

# --- Extract star pattern ---
with st.spinner("Extracting star pattern..."):
    star_data = extract_star_pattern_data(scenario, scenario_id)

if star_data is None:
    st.error("Failed to extract star pattern from this scenario.")
    st.stop()

# --- Star Pattern Summary ---
st.markdown("### Pattern Summary")

sum_cols = st.columns(6)
with sum_cols[0]:
    itype = star_data.get("center_type", "none")
    color = INTERSECTION_COLORS.get(itype, "#95A5A6")
    st.markdown(
        f'<div style="background-color:{color}; color:white; padding:10px; '
        f'border-radius:8px; text-align:center;">'
        f'<b>{itype.upper()}</b><br><small>Center Type</small></div>',
        unsafe_allow_html=True,
    )
with sum_cols[1]:
    st.metric("Approaches", star_data.get("center_approaches", 0))
with sum_cols[2]:
    st.metric("Arms", star_data.get("n_arms", 0))
with sum_cols[3]:
    sig = "Yes" if star_data.get("center_has_signal") else "No"
    st.metric("Has Signal", sig)
with sum_cols[4]:
    stop = "Yes" if star_data.get("center_has_stop") else "No"
    st.metric("Has Stop", stop)
with sum_cols[5]:
    cw = "Yes" if star_data.get("center_has_crosswalk") else "No"
    st.metric("Has Crosswalk", cw)

# --- Visual Representations ---
st.markdown("---")
st.markdown("### Visual Representations")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.markdown("#### Center Features (Radar)")
    fig_radar = plot_star_pattern_radar(star_data)
    st.plotly_chart(fig_radar, use_container_width=True)

with viz_col2:
    st.markdown("#### Approach Arms (Compass Rose)")
    fig_compass = plot_star_pattern_compass(star_data)
    st.plotly_chart(fig_compass, use_container_width=True)

# --- Feature Vector Heatmap ---
st.markdown("#### 48D Feature Vector Heatmap")
fig_heatmap = plot_feature_vector_heatmap(star_data)
st.plotly_chart(fig_heatmap, use_container_width=True)

# --- Arm Details Table ---
st.markdown("#### Arm Details")
arms = star_data.get("arms", [])
if arms:
    import pandas as pd
    arms_df = pd.DataFrame([
        {
            "Arm": i + 1,
            "Direction (deg)": round(a["angle"], 1),
            "Length (m)": round(a.get("length", 0), 1),
            "Road Type": a.get("road_type", "unknown"),
            "Lanes": a.get("lanes", 1),
            "Neighbor Type": a.get("neighbor_type", "none"),
            "Neighbor Degree": a.get("neighbor_degree", 0),
            "Neighbor Signal": "Yes" if a.get("neighbor_has_signal") else "No",
        }
        for i, a in enumerate(arms)
    ])
    st.dataframe(arms_df, use_container_width=True, hide_index=True)
else:
    st.info("No arms data available for this scenario.")

# --- Pattern Comparison ---
st.markdown("---")
st.markdown("### Pattern Comparison")
st.markdown("Compare this scenario's star pattern with a reference pattern (e.g., from OSM).")

# Generate a comparison pattern
compare_tab1, compare_tab2 = st.tabs(["Compare with OSM Demo", "Compare Two Scenarios"])

with compare_tab1:
    demo_patterns = generate_demo_star_patterns(10)

    # Find one with similar type
    same_type = [p for p in demo_patterns if p["center_type"] == star_data.get("center_type")]
    diff_type = [p for p in demo_patterns if p["center_type"] != star_data.get("center_type")]

    if same_type:
        compare_pattern = same_type[0]
    elif demo_patterns:
        compare_pattern = demo_patterns[0]
    else:
        compare_pattern = None

    if compare_pattern:
        # Compute distance
        star_obj = StarPattern.from_dict(star_data)
        osm_obj = StarPattern.from_dict(compare_pattern)
        distance = star_distance(star_obj.to_vector(), osm_obj.to_vector())
        score = max(0.0, 1.0 - distance / 15.0)

        comp_cols = st.columns([1, 2, 1])
        with comp_cols[0]:
            st.markdown(f"**WOMD Pattern**")
            st.markdown(f"Type: `{star_data.get('center_type')}`")
            st.markdown(f"Arms: `{star_data.get('n_arms')}`")
        with comp_cols[1]:
            st.markdown(f"<center><h3>Distance: {distance:.3f}</h3></center>", unsafe_allow_html=True)
            st.markdown(f"<center><h4>Similarity: {score:.3f}</h4></center>", unsafe_allow_html=True)
        with comp_cols[2]:
            st.markdown(f"**OSM Pattern**")
            st.markdown(f"Type: `{compare_pattern.get('center_type')}`")
            st.markdown(f"Arms: `{compare_pattern.get('n_arms')}`")
            st.markdown(f"ID: `{compare_pattern.get('id')}`")

        # Add vector to compare_pattern
        compare_pattern["vector"] = osm_obj.to_vector().tolist()

        fig_comp = plot_star_pattern_comparison(star_data, compare_pattern)
        st.plotly_chart(fig_comp, use_container_width=True)

with compare_tab2:
    if not use_demo and len(scenario_files) > 1:
        other_file = st.selectbox(
            "Select second scenario",
            [f for f in scenario_files if f != selected_file],
            index=0,
            key="compare_second_file",
        )

        other_scenario = load_scenario_raw(data_dir, other_file)
        if other_scenario is not None:
            other_id = Path(other_file).stem
            other_star = extract_star_pattern_data(other_scenario, other_id)
            if other_star is not None:
                star_obj1 = StarPattern.from_dict(star_data)
                star_obj2 = StarPattern.from_dict(other_star)
                dist = star_distance(star_obj1.to_vector(), star_obj2.to_vector())
                sim = max(0.0, 1.0 - dist / 15.0)

                c1, c2, c3 = st.columns([1, 2, 1])
                with c1:
                    st.markdown(f"**{scenario_id}**")
                    st.markdown(f"Type: `{star_data.get('center_type')}`")
                with c2:
                    st.markdown(f"<center>Distance: **{dist:.3f}** | Similarity: **{sim:.3f}**</center>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"**{other_id}**")
                    st.markdown(f"Type: `{other_star.get('center_type')}`")

                fig_comp2 = plot_star_pattern_comparison(star_data, other_star, scenario_id, other_id)
                st.plotly_chart(fig_comp2, use_container_width=True)
    else:
        st.info("Need at least 2 scenario files for two-scenario comparison.")

# --- Raw Vector ---
with st.expander("Raw 48D Vector", expanded=False):
    vector = star_data.get("vector", [])
    st.code(f"Shape: (48,)\nValues: {[round(v, 4) for v in vector]}")
