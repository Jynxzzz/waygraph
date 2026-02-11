"""
Page 4: Traffic Analysis
=========================

Extract and visualize traffic parameters from WOMD scenarios:
turning ratios, speed distributions, and gap acceptance parameters.
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
    extract_turning_ratios,
    extract_speed_distributions,
    extract_gap_acceptance,
    generate_demo_scenario,
)
from utils.viz import (
    plot_turning_ratios,
    plot_per_approach_turning,
    plot_speed_distributions,
    plot_gap_acceptance,
)

st.set_page_config(page_title="Traffic Analysis - WayGraph", page_icon="🚗", layout="wide")

st.title("Traffic Analysis")
st.markdown(
    "Extract turning ratios, speed distributions, and gap acceptance parameters "
    "from observed vehicle trajectories in WOMD scenarios."
)

# --- Load data ---
data_dir = get_data_dir()
scenario_files = list_scenario_files(data_dir)
use_demo = len(scenario_files) == 0

# --- Scenario Selection ---
st.markdown("### Select Scenario")

if use_demo:
    st.info("Using demo scenario data.")
    scenario = generate_demo_scenario()
    scenario_id = "demo_cross_intersection"
else:
    selected_file = st.selectbox(
        "Scenario File",
        scenario_files,
        index=0,
        key="traffic_file_select",
    )
    scenario = load_scenario_raw(data_dir, selected_file)
    scenario_id = Path(selected_file).stem

if scenario is None:
    st.error("Failed to load scenario.")
    st.stop()

# Scenario summary
objects = scenario.get("objects", [])
vehicles = [o for o in objects if o.get("type") == "vehicle"]
n_timesteps = len(objects[0].get("position", [])) if objects else 0

summary_cols = st.columns(4)
with summary_cols[0]:
    st.metric("Total Objects", len(objects))
with summary_cols[1]:
    st.metric("Vehicles", len(vehicles))
with summary_cols[2]:
    st.metric("Timesteps", n_timesteps)
with summary_cols[3]:
    st.metric("Duration", f"{n_timesteps * 0.1:.1f}s")

# --- Analysis Tabs ---
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "Turning Ratios",
    "Speed Distributions",
    "Gap Acceptance",
])

# ======= TAB 1: TURNING RATIOS =======
with tab1:
    st.markdown("### Turning Movement Analysis")
    st.markdown(
        "Each vehicle is assigned to an entry lane using spatial proximity, "
        "then its turning movement (left, through, right, U-turn) is determined "
        "from the heading change between trajectory start and end."
    )

    with st.spinner("Extracting turning ratios..."):
        turning_data = extract_turning_ratios(scenario)

    if turning_data:
        # Summary
        total_vehicles = sum(d.get("total", 0) for d in turning_data.values())
        total_left = sum(d.get("left", 0) for d in turning_data.values())
        total_through = sum(d.get("through", 0) for d in turning_data.values())
        total_right = sum(d.get("right", 0) for d in turning_data.values())
        total_uturn = sum(d.get("uturn", 0) for d in turning_data.values())

        turn_met_cols = st.columns(5)
        with turn_met_cols[0]:
            st.metric("Observed Vehicles", total_vehicles)
        with turn_met_cols[1]:
            pct = f"{100*total_left/max(total_vehicles,1):.0f}%"
            st.metric("Left Turns", f"{total_left} ({pct})")
        with turn_met_cols[2]:
            pct = f"{100*total_through/max(total_vehicles,1):.0f}%"
            st.metric("Through", f"{total_through} ({pct})")
        with turn_met_cols[3]:
            pct = f"{100*total_right/max(total_vehicles,1):.0f}%"
            st.metric("Right Turns", f"{total_right} ({pct})")
        with turn_met_cols[4]:
            pct = f"{100*total_uturn/max(total_vehicles,1):.0f}%"
            st.metric("U-Turns", f"{total_uturn} ({pct})")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_pie = plot_turning_ratios(turning_data, title="Aggregate Turning Ratios")
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            fig_bar = plot_per_approach_turning(turning_data, title="Per-Approach Movements")
            st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed table
        with st.expander("Detailed Turning Data", expanded=False):
            turn_records = []
            for approach_id, data in turning_data.items():
                turn_records.append({
                    "Approach": approach_id,
                    "Angle (deg)": round(data.get("approach_angle", 0), 1),
                    "Left": data.get("left", 0),
                    "Through": data.get("through", 0),
                    "Right": data.get("right", 0),
                    "U-Turn": data.get("uturn", 0),
                    "Total": data.get("total", 0),
                    "Left %": f"{data.get('left_ratio', 0)*100:.1f}%",
                    "Through %": f"{data.get('through_ratio', 0)*100:.1f}%",
                    "Right %": f"{data.get('right_ratio', 0)*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(turn_records), use_container_width=True, hide_index=True)
    else:
        st.info(
            "No turning movements could be extracted. This may happen if there "
            "are too few vehicles or they don't move near lane centerlines."
        )

# ======= TAB 2: SPEED DISTRIBUTIONS =======
with tab2:
    st.markdown("### Speed Distribution Analysis")
    st.markdown(
        "For each timestep, each moving vehicle is assigned to the nearest lane "
        "centerline point. Its instantaneous speed is recorded, building "
        "per-lane speed distributions."
    )

    with st.spinner("Extracting speed distributions..."):
        speed_data = extract_speed_distributions(scenario)

    if speed_data:
        # Summary stats
        all_means = [d.get("mean_speed_kmh", 0) for d in speed_data.values() if d.get("n_samples", 0) > 5]
        all_ff = [d.get("free_flow_speed_kmh", 0) for d in speed_data.values() if d.get("n_samples", 0) > 5]
        total_samples = sum(d.get("n_samples", 0) for d in speed_data.values())

        spd_met_cols = st.columns(4)
        with spd_met_cols[0]:
            st.metric("Lanes with Data", len(speed_data))
        with spd_met_cols[1]:
            st.metric("Total Samples", f"{total_samples:,}")
        with spd_met_cols[2]:
            avg_speed = np.mean(all_means) if all_means else 0
            st.metric("Avg Mean Speed", f"{avg_speed:.1f} km/h")
        with spd_met_cols[3]:
            avg_ff = np.mean(all_ff) if all_ff else 0
            st.metric("Avg Free-Flow", f"{avg_ff:.1f} km/h")

        fig_speed = plot_speed_distributions(speed_data, title="Speed by Lane (Top 20)")
        st.plotly_chart(fig_speed, use_container_width=True)

        # Speed histogram (all data combined)
        all_speeds_kmh = []
        for d in speed_data.values():
            mean = d.get("mean_speed_ms", 0)
            if mean > 0:
                all_speeds_kmh.append(mean * 3.6)

        if all_speeds_kmh:
            import plotly.graph_objects as go
            fig_hist = go.Figure(data=[go.Histogram(
                x=all_speeds_kmh,
                nbinsx=30,
                marker_color="#4A90D9",
                name="Mean Speed per Lane",
            )])
            fig_hist.update_layout(
                title="Distribution of Mean Speeds Across Lanes",
                xaxis_title="Mean Speed (km/h)",
                yaxis_title="Count",
                height=350,
                template="plotly_white",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Detailed table
        with st.expander("Speed Data Table", expanded=False):
            speed_records = []
            for lane_id, data in sorted(speed_data.items(), key=lambda x: x[1].get("mean_speed_kmh", 0), reverse=True):
                speed_records.append({
                    "Lane": lane_id,
                    "Samples": data.get("n_samples", 0),
                    "Mean (km/h)": round(data.get("mean_speed_kmh", 0), 1),
                    "Std (m/s)": round(data.get("std_speed_ms", 0), 2),
                    "Free-Flow (km/h)": round(data.get("free_flow_speed_kmh", 0), 1),
                })
            st.dataframe(pd.DataFrame(speed_records), use_container_width=True, height=300)
    else:
        st.info(
            "No speed data could be extracted. This may happen if vehicles "
            "are mostly stationary or too far from lane centerlines."
        )

# ======= TAB 3: GAP ACCEPTANCE =======
with tab3:
    st.markdown("### Gap Acceptance Analysis")
    st.markdown(
        "Identifies stop-to-go transitions where vehicles wait to enter "
        "a traffic stream, and measures the gaps they accept or reject. "
        "The **critical gap** is the median accepted gap duration."
    )

    with st.spinner("Extracting gap acceptance parameters..."):
        gap_data = extract_gap_acceptance(scenario)

    if gap_data and (gap_data.get("n_accepted", 0) > 0 or gap_data.get("n_rejected", 0) > 0):
        gap_met_cols = st.columns(4)
        with gap_met_cols[0]:
            st.metric("Critical Gap", f"{gap_data.get('critical_gap_s', 0):.2f}s")
        with gap_met_cols[1]:
            st.metric("Follow-Up Time", f"{gap_data.get('follow_up_time_s', 0):.2f}s")
        with gap_met_cols[2]:
            st.metric("Accepted Gaps", gap_data.get("n_accepted", 0))
        with gap_met_cols[3]:
            st.metric("Rejected Gaps", gap_data.get("n_rejected", 0))

        fig_gap = plot_gap_acceptance(gap_data, title="Gap Duration Distribution")
        st.plotly_chart(fig_gap, use_container_width=True)

        # Summary statistics
        accepted = gap_data.get("accepted_gaps", [])
        rejected = gap_data.get("rejected_gaps", [])

        if accepted:
            st.markdown("#### Accepted Gap Statistics")
            acc_stats = pd.DataFrame([{
                "Count": len(accepted),
                "Mean (s)": round(np.mean(accepted), 2),
                "Median (s)": round(np.median(accepted), 2),
                "Std (s)": round(np.std(accepted), 2),
                "Min (s)": round(min(accepted), 2),
                "Max (s)": round(max(accepted), 2),
            }])
            st.dataframe(acc_stats, use_container_width=True, hide_index=True)

        if rejected:
            st.markdown("#### Rejected Gap Statistics")
            rej_stats = pd.DataFrame([{
                "Count": len(rejected),
                "Mean (s)": round(np.mean(rejected), 2),
                "Median (s)": round(np.median(rejected), 2),
                "Std (s)": round(np.std(rejected), 2),
                "Min (s)": round(min(rejected), 2),
                "Max (s)": round(max(rejected), 2),
            }])
            st.dataframe(rej_stats, use_container_width=True, hide_index=True)

        st.markdown("""
        **Interpretation:**
        - A **critical gap** of 4-5 seconds is typical for unsignalized intersections
        - **Follow-up time** of 2-3 seconds is standard for gap acceptance models
        - These parameters can be used to calibrate SUMO/VISSIM microsimulations
        """)
    else:
        st.info(
            "No gap acceptance data could be extracted. This typically means "
            "there were no clear stop-to-go transitions in the observed trajectories. "
            "Gap acceptance analysis works best at unsignalized intersections."
        )

# --- Aggregate Analysis ---
if not use_demo and len(scenario_files) > 1:
    st.markdown("---")
    st.markdown("### Aggregate Traffic Statistics")
    st.markdown("Analyze traffic parameters across multiple scenarios.")

    max_agg = min(30, len(scenario_files))
    n_aggregate = st.slider("Scenarios to aggregate", 1, max(max_agg, 2), min(10, max_agg), key="agg_slider") if max_agg > 1 else max_agg

    if st.button("Run Aggregate Analysis", use_container_width=True):
        progress = st.progress(0)
        agg_speeds = []
        agg_turns = {"left": 0, "through": 0, "right": 0, "uturn": 0}
        agg_gaps = []
        n_processed = 0

        for i, fname in enumerate(scenario_files[:n_aggregate]):
            progress.progress((i + 1) / n_aggregate, text=f"Processing {i+1}/{n_aggregate}...")
            sc = load_scenario_raw(data_dir, fname)
            if sc is None:
                continue

            # Speeds
            spd = extract_speed_distributions(sc)
            for d in spd.values():
                if d.get("n_samples", 0) > 5:
                    agg_speeds.append(d.get("mean_speed_kmh", 0))

            # Turns
            turns = extract_turning_ratios(sc)
            for d in turns.values():
                agg_turns["left"] += d.get("left", 0)
                agg_turns["through"] += d.get("through", 0)
                agg_turns["right"] += d.get("right", 0)
                agg_turns["uturn"] += d.get("uturn", 0)

            # Gaps
            gaps = extract_gap_acceptance(sc)
            if gaps.get("critical_gap_s", 0) > 0:
                agg_gaps.append(gaps["critical_gap_s"])

            n_processed += 1

        progress.empty()

        if n_processed > 0:
            agg_col1, agg_col2, agg_col3 = st.columns(3)

            with agg_col1:
                st.markdown("**Aggregate Speed Stats**")
                if agg_speeds:
                    st.metric("Mean Speed", f"{np.mean(agg_speeds):.1f} km/h")
                    st.metric("Median Speed", f"{np.median(agg_speeds):.1f} km/h")
                    st.metric("Speed Range", f"{min(agg_speeds):.1f} - {max(agg_speeds):.1f} km/h")

            with agg_col2:
                st.markdown("**Aggregate Turning Ratios**")
                total = sum(agg_turns.values())
                if total > 0:
                    for turn_type, count in agg_turns.items():
                        pct = 100 * count / total
                        st.metric(turn_type.capitalize(), f"{count} ({pct:.1f}%)")

            with agg_col3:
                st.markdown("**Aggregate Gap Acceptance**")
                if agg_gaps:
                    st.metric("Mean Critical Gap", f"{np.mean(agg_gaps):.2f}s")
                    st.metric("Median Critical Gap", f"{np.median(agg_gaps):.2f}s")
                    st.metric("Scenarios with Data", len(agg_gaps))
                else:
                    st.info("No gap acceptance data found.")
