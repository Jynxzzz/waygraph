"""
Page 3: Matching Results
=========================

Match WOMD scenarios against an OSM star pattern database and
visualize the top-K results with distance metrics, map locations,
and accuracy statistics.
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
    extract_star_pattern_data,
    generate_demo_scenario,
    generate_demo_star_patterns,
)
from utils.viz import (
    plot_match_scores,
    plot_star_pattern_comparison,
    plot_star_pattern_compass,
    INTERSECTION_COLORS,
)
from utils.map_viz import create_match_map, HAS_FOLIUM

from waygraph.fingerprint import StarPattern, StarPatternMatcher, star_distance

st.set_page_config(page_title="Matching Results - WayGraph", page_icon="🎯", layout="wide")

st.title("Topology Matching Results")
st.markdown(
    "Match WOMD scenarios against an OSM star pattern database using "
    "the 48D fingerprint. View top-K matches, distance metrics, and "
    "matched locations on a map."
)

# --- Matching Pipeline Explanation ---
with st.expander("How Matching Works", expanded=False):
    st.markdown("""
    **Matching Pipeline:**
    1. **Coarse filter:** Reject candidates with incompatible intersection types
       or approach counts (eliminates 70-80% of candidates)
    2. **Feature distance:** Compute weighted Euclidean distance between 48D star
       pattern vectors
    3. **Ranking:** Return top-K matches sorted by similarity score

    **Performance benchmarks:**
    - Top-1 accuracy: ~90% (clean), ~70% (realistic noise)
    - MRR: ~0.92 (clean), ~0.78 (realistic)
    - Speed: <2 seconds for 250 queries against 5000+ OSM patterns
    """)

# --- Load data ---
data_dir = get_data_dir()
scenario_files = list_scenario_files(data_dir)
use_demo = len(scenario_files) == 0

# --- Build or load OSM database ---
st.markdown("### Reference Database")

# For now, use demo OSM patterns (in production, would load from a JSON file)
db_tab1, db_tab2 = st.tabs(["Demo Database", "Custom Database"])

with db_tab1:
    n_demo_patterns = st.slider("Demo OSM patterns", 10, 100, 50, 10)
    demo_patterns_data = generate_demo_star_patterns(n_demo_patterns)
    db_patterns = [StarPattern.from_dict(p) for p in demo_patterns_data]
    st.success(f"Loaded {len(db_patterns)} demo OSM patterns")

with db_tab2:
    uploaded_file = st.file_uploader(
        "Upload OSM star pattern database (JSON)",
        type=["json"],
        help="JSON file from OSMStarDatabase.save_patterns()",
    )
    if uploaded_file is not None:
        import json
        try:
            custom_data = json.load(uploaded_file)
            db_patterns = [StarPattern.from_dict(p) for p in custom_data]
            st.success(f"Loaded {len(db_patterns)} patterns from file")
        except Exception as e:
            st.error(f"Failed to load: {e}")

# --- Build matcher ---
matcher = StarPatternMatcher()
matcher.build_database(db_patterns)

# --- Query Pattern ---
st.markdown("---")
st.markdown("### Query Scenario")

if use_demo:
    st.info("Using demo scenario. Connect real data for full functionality.")
    scenario = generate_demo_scenario()
    scenario_id = "demo_cross_intersection"
    selected_file = "demo"
else:
    selected_file = st.selectbox(
        "Select Scenario to Match",
        scenario_files,
        index=0,
        key="match_file_select",
    )
    scenario = load_scenario_raw(data_dir, selected_file)
    scenario_id = Path(selected_file).stem

if scenario is None:
    st.error("Failed to load scenario.")
    st.stop()

# Extract star pattern
with st.spinner("Extracting star pattern..."):
    star_data = extract_star_pattern_data(scenario, scenario_id)

if star_data is None:
    st.error("Failed to extract star pattern.")
    st.stop()

query_star = StarPattern.from_dict(star_data)

# --- Matching Parameters ---
st.markdown("### Matching Parameters")
param_col1, param_col2, param_col3 = st.columns(3)
with param_col1:
    top_k = st.slider("Top-K results", 3, 20, 10)
with param_col2:
    max_approach_diff = st.slider("Max approach difference", 0, 3, 1)
with param_col3:
    score_norm = st.slider("Score normalization", 5.0, 30.0, 15.0, 1.0)

matcher.max_approach_diff = max_approach_diff
matcher.score_normalization = score_norm

# --- Run Matching ---
if st.button("Run Matching", type="primary", use_container_width=True):
    with st.spinner("Matching..."):
        results = matcher.match(query_star, top_k=top_k)

    st.session_state["match_results"] = results
    st.session_state["match_query_data"] = star_data
    st.session_state["match_db_data"] = demo_patterns_data

# --- Display Results ---
if "match_results" in st.session_state:
    results = st.session_state["match_results"]
    star_data = st.session_state.get("match_query_data", star_data)
    db_data = st.session_state.get("match_db_data", demo_patterns_data)

    st.markdown("---")
    st.markdown("### Match Results")

    if not results:
        st.warning("No matches found. Try relaxing the matching parameters.")
    else:
        # Summary metrics
        best_id, best_score = results[0]
        mean_score = np.mean([s for _, s in results])

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric("Best Match", best_id)
        with res_col2:
            st.metric("Best Score", f"{best_score:.4f}")
        with res_col3:
            st.metric("Mean Score (Top-K)", f"{mean_score:.4f}")
        with res_col4:
            st.metric("Matches Found", len(results))

        # Score chart
        fig_scores = plot_match_scores(results, title=f"Top-{len(results)} Matches for {scenario_id}")
        st.plotly_chart(fig_scores, use_container_width=True)

        # Results table
        st.markdown("#### Match Details")
        results_records = []
        db_lookup = {p["id"]: p for p in db_data}
        for rank, (match_id, score) in enumerate(results, 1):
            record = {
                "Rank": rank,
                "OSM Node ID": match_id,
                "Score": round(score, 4),
            }
            # Look up additional info from database
            if match_id in db_lookup:
                p = db_lookup[match_id]
                record["Type"] = p.get("center_type", "?")
                record["Approaches"] = p.get("center_approaches", 0)
                record["Has Signal"] = "Yes" if p.get("center_has_signal") else "No"
                record["Lat"] = p.get("lat", 0)
                record["Lon"] = p.get("lon", 0)
            results_records.append(record)

        results_df = pd.DataFrame(results_records)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # --- Map View ---
        st.markdown("#### Matched Locations on Map")

        if HAS_FOLIUM:
            map_results = []
            for match_id, score in results:
                if match_id in db_lookup:
                    p = db_lookup[match_id]
                    map_results.append({
                        "id": match_id,
                        "score": score,
                        "lat": p.get("lat", 0),
                        "lon": p.get("lon", 0),
                    })

            if map_results and any(r["lat"] != 0 for r in map_results):
                from streamlit_folium import st_folium
                m = create_match_map(map_results)
                if m is not None:
                    st_folium(m, width=None, height=450, returned_objects=[])
            else:
                st.info("No geographic coordinates available for matched patterns.")
        else:
            st.info("Install `folium` and `streamlit-folium` for map visualization.")

        # --- Detailed comparison with best match ---
        st.markdown("---")
        st.markdown("#### Detailed Comparison: Query vs Best Match")

        if best_id in db_lookup:
            best_pattern_data = db_lookup[best_id]
            # Add vector for comparison
            best_star_obj = StarPattern.from_dict(best_pattern_data)
            best_pattern_data["vector"] = best_star_obj.to_vector().tolist()

            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown(f"**Query: {scenario_id}**")
                fig_q = plot_star_pattern_compass(star_data, title=f"Query: {scenario_id}")
                st.plotly_chart(fig_q, use_container_width=True)

            with comp_col2:
                st.markdown(f"**Best Match: {best_id}**")
                fig_m = plot_star_pattern_compass(best_pattern_data, title=f"Match: {best_id}")
                st.plotly_chart(fig_m, use_container_width=True)

            # Feature comparison
            fig_comp = plot_star_pattern_comparison(
                star_data, best_pattern_data,
                label1=f"Query ({scenario_id})",
                label2=f"Match ({best_id})",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # Dimension-wise difference
            v1 = np.array(star_data.get("vector", [0] * 48))
            v2 = np.array(best_pattern_data.get("vector", [0] * 48))
            diff = np.abs(v1 - v2)

            st.markdown("#### Per-Dimension Difference")
            import plotly.graph_objects as go
            dim_labels = (
                ["Type", "Appr", "Sig", "Stop", "CW", "Arms"]
                + [f"A{a}_{f}" for a in range(6) for f in ["Ang", "Len", "Rd", "Ln", "NbT", "NbD", "NbS"]]
            )[:48]

            fig_diff = go.Figure(data=[go.Bar(
                x=dim_labels, y=diff.tolist(),
                marker_color=["#E74C3C" if d > 0.2 else "#2ECC71" if d < 0.05 else "#F39C12" for d in diff],
                hovertemplate="%{x}: diff=%{y:.4f}<extra></extra>",
            )])
            fig_diff.update_layout(
                title="Absolute Difference per Feature Dimension",
                yaxis_title="Absolute Difference",
                height=300,
                template="plotly_white",
                xaxis=dict(tickangle=45, tickfont=dict(size=7)),
            )
            st.plotly_chart(fig_diff, use_container_width=True)

else:
    st.info("Click **Run Matching** to match the selected scenario against the reference database.")

# --- Batch Matching ---
st.markdown("---")
st.markdown("### Batch Matching")

if not use_demo and len(scenario_files) > 1:
    max_batch = min(50, len(scenario_files))
    n_batch = st.slider("Scenarios to match", 1, max(max_batch, 2), min(10, max_batch)) if max_batch > 1 else max_batch

    if st.button("Run Batch Matching", use_container_width=True):
        progress = st.progress(0)
        batch_results = []

        for i, fname in enumerate(scenario_files[:n_batch]):
            progress.progress((i + 1) / n_batch, text=f"Matching {i+1}/{n_batch}...")
            sc = load_scenario_raw(data_dir, fname)
            if sc is None:
                continue
            sid = Path(fname).stem
            sp_data = extract_star_pattern_data(sc, sid)
            if sp_data is None:
                continue
            sp_obj = StarPattern.from_dict(sp_data)
            matches = matcher.match(sp_obj, top_k=5)
            if matches:
                best_id, best_score = matches[0]
                batch_results.append({
                    "Scenario": sid,
                    "Type": sp_data.get("center_type", "?"),
                    "Arms": sp_data.get("n_arms", 0),
                    "Best Match": best_id,
                    "Score": round(best_score, 4),
                    "Top-5 Avg": round(np.mean([s for _, s in matches[:5]]), 4),
                })

        progress.empty()

        if batch_results:
            batch_df = pd.DataFrame(batch_results)
            st.dataframe(batch_df, use_container_width=True, hide_index=True)

            # Summary stats
            avg_best = batch_df["Score"].mean()
            st.metric("Average Best-Match Score", f"{avg_best:.4f}")
        else:
            st.warning("No valid results from batch matching.")
else:
    st.info("Batch matching requires multiple scenario files.")
