"""
WayGraph Web Application - Main Dashboard
==========================================

Interactive web application for visualizing autonomous driving scenario
topology matching between WOMD (Waymo Open Motion Dataset) and OpenStreetMap.

Run with:
    streamlit run app/app.py --server.port 8501
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Ensure waygraph is importable
WAYGRAPH_ROOT = Path(__file__).resolve().parent.parent
if str(WAYGRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(WAYGRAPH_ROOT))

# App-level imports
from utils.data_loader import (
    get_data_dir,
    list_scenario_files,
    extract_batch_summaries,
    DEFAULT_DATA_DIR,
)
from utils.viz import (
    plot_intersection_type_distribution,
    plot_approach_distribution,
    plot_traffic_control_presence,
    plot_lane_vs_area_scatter,
    INTERSECTION_COLORS,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="WayGraph - Topology Analysis",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 5px 0;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2.2em;
        font-weight: 700;
    }
    .metric-card p {
        margin: 5px 0 0 0;
        font-size: 0.95em;
        opacity: 0.9;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4A90D9;
        margin: 10px 0;
    }
    div[data-testid="stSidebarContent"] {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.shields.io/badge/WayGraph-v0.1.0-blue.svg", width=160)
    st.markdown("## Configuration")

    data_dir = st.text_input(
        "Scenario Data Directory",
        value=DEFAULT_DATA_DIR,
        help="Path to directory containing .pkl scenario files",
    )
    st.session_state["data_dir"] = data_dir

    # Check data availability
    scenario_files = list_scenario_files(data_dir)
    n_files = len(scenario_files)

    if n_files > 0:
        st.success(f"Loaded {n_files} WOMD scenario files")
    else:
        st.info("Running with sample data — explore the full pipeline below.")

    st.markdown("---")
    st.markdown("### Quick Links")
    st.page_link("pages/1_Scenario_Explorer.py", label="Scenario Explorer", icon="🔍")
    st.page_link("pages/2_Star_Pattern.py", label="Star Pattern Viewer", icon="⭐")
    st.page_link("pages/3_Matching.py", label="Matching Results", icon="🎯")
    st.page_link("pages/4_Traffic_Analysis.py", label="Traffic Analysis", icon="🚗")
    st.page_link("pages/5_City_Comparison.py", label="City Comparison", icon="🌍")

    st.markdown("---")
    st.markdown(
        "<small>Built with WayGraph toolkit<br>"
        "Concordia University</small>",
        unsafe_allow_html=True,
    )


# --- Main Dashboard ---
st.title("WayGraph Dashboard")
st.markdown(
    "Structural analysis and topology matching of autonomous driving scenarios "
    "from the **Waymo Open Motion Dataset** to **OpenStreetMap**."
)

# --- Key Metrics ---
st.markdown("### Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""<div class="metric-card">
        <h2>{n_files}</h2>
        <p>Scenario Files</p>
        </div>""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """<div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <h2>48D</h2>
        <p>Star Pattern Dims</p>
        </div>""",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """<div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <h2>~90%</h2>
        <p>Top-1 Accuracy</p>
        </div>""",
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """<div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <h2>7</h2>
        <p>Intersection Types</p>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("")

# --- Batch Analysis ---
if n_files > 0:
    st.markdown("### Dataset Analysis")

    batch_size = st.slider(
        "Number of scenarios to analyze",
        min_value=5,
        max_value=min(n_files, 100),
        value=min(n_files, 30),
        step=5,
        help="More scenarios = better statistics but slower loading",
    )

    df = extract_batch_summaries(data_dir, scenario_files, max_files=batch_size)

    if not df.empty:
        # Summary metrics row
        met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
        with met_col1:
            st.metric("Scenarios Loaded", len(df))
        with met_col2:
            avg_lanes = df["num_lanes"].mean() if "num_lanes" in df.columns else 0
            st.metric("Avg Lanes", f"{avg_lanes:.0f}")
        with met_col3:
            avg_approaches = df["num_approaches"].mean() if "num_approaches" in df.columns else 0
            st.metric("Avg Approaches", f"{avg_approaches:.1f}")
        with met_col4:
            tl_pct = df["has_traffic_light"].mean() * 100 if "has_traffic_light" in df.columns else 0
            st.metric("Has Traffic Light", f"{tl_pct:.0f}%")
        with met_col5:
            avg_area = df["area_m2"].mean() if "area_m2" in df.columns else 0
            st.metric("Avg Area", f"{avg_area:,.0f} m\u00b2")

        st.markdown("---")

        # Charts
        tab1, tab2, tab3, tab4 = st.tabs([
            "Type Distribution",
            "Approaches",
            "Traffic Controls",
            "Lanes vs Area",
        ])

        with tab1:
            fig = plot_intersection_type_distribution(df)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = plot_approach_distribution(df)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            fig = plot_traffic_control_presence(df)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            fig = plot_lane_vs_area_scatter(df)
            st.plotly_chart(fig, use_container_width=True)

        # Data table
        with st.expander("Raw Data Table", expanded=False):
            display_cols = [
                c for c in [
                    "filename", "scenario_id", "intersection_type", "num_lanes",
                    "num_approaches", "has_traffic_light", "has_stop_sign",
                    "has_crosswalk", "area_m2", "branching_factor", "merge_factor",
                ]
                if c in df.columns
            ]
            st.dataframe(df[display_cols], use_container_width=True, height=300)

    else:
        st.info("Could not extract any topology data. Files may be in an unexpected format.")

else:
    # Demo mode
    st.markdown("### Explore WayGraph")

    st.markdown("""
    <div class="info-box">
    <b>Welcome to WayGraph!</b> Explore the GPS-free localization pipeline with sample data.
    Navigate the pages in the sidebar to see star pattern fingerprinting, matching results,
    traffic analysis, and more.<br><br>
    For full dataset analysis, install WayGraph locally with your own WOMD <code>.pkl</code> files.
    </div>
    """, unsafe_allow_html=True)

    # Show demo content
    st.markdown("#### What WayGraph Does")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        **Star Pattern Fingerprinting**

        WayGraph extracts a 48-dimensional feature vector (star pattern) from each
        intersection, encoding:
        - Center intersection properties (type, approaches, traffic controls)
        - Per-arm road properties (direction, length, road type, lane count)
        - Neighboring intersection topology

        This enables **GPS-free** matching of WOMD scenarios to real-world OSM locations.
        """)

    with col_b:
        st.markdown("""
        **Key Capabilities**

        | Feature | Status |
        |---------|--------|
        | Intersection Detection | Available |
        | Star Pattern Extraction | Available |
        | OSM Matching | Available |
        | Turning Ratio Extraction | Available |
        | Speed Distribution Analysis | Available |
        | Gap Acceptance Estimation | Available |
        | Publication-Quality Visualization | Available |
        """)

    st.markdown("#### Supported Intersection Types")
    type_cols = st.columns(7)
    for i, (itype, color) in enumerate(INTERSECTION_COLORS.items()):
        with type_cols[i]:
            st.markdown(
                f'<div style="background-color:{color}; color:white; padding:10px; '
                f'border-radius:8px; text-align:center; font-weight:bold;">'
                f'{itype.upper()}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("""
    #### Pipeline Overview

    ```
    WOMD .pkl files
        --> ScenarioLoader (extract lane graph, agents, traffic controls)
            --> IntersectionClassifier (detect + classify intersection type)
                --> StarPattern.from_topology() (48D fingerprint)
                    --> StarPatternMatcher.match() (find best OSM match)
    ```

    Navigate to **Scenario Explorer** to load and visualize individual scenarios,
    or **Star Pattern Viewer** to see the 48D fingerprint in action.
    """)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<center><small>"
    "WayGraph v0.1.0 | "
    "Structural Analysis Toolkit for Autonomous Driving Datasets | "
    "Apache 2.0 License"
    "</small></center>",
    unsafe_allow_html=True,
)
