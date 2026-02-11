# WayGraph Dashboard

Interactive web application for visualizing autonomous driving scenario topology matching between the Waymo Open Motion Dataset (WOMD) and OpenStreetMap.

## Features

- **Main Dashboard** -- Overview metrics and batch analysis of intersection topology statistics.
- **Scenario Explorer** -- Load and visualize individual WOMD scenarios with lane graphs, agent trajectories, and traffic controls.
- **Star Pattern Viewer** -- Visualize the 48-dimensional star pattern fingerprint that encodes intersection topology and 1-hop neighborhood context. Includes radar charts, compass rose diagrams, and feature vector heatmaps.
- **Matching Results** -- Match WOMD scenarios against an OSM star pattern database using the 48D fingerprint. View top-K matches, distance metrics, and matched locations on an interactive map.
- **Traffic Analysis** -- Extract and visualize turning ratios, speed distributions, and gap acceptance parameters from observed vehicle trajectories.
- **City Comparison** -- Compare intersection topology distributions and traffic parameters across different cities covered by WOMD.

## Demo Mode

When no `.pkl` scenario files are available, the app automatically runs in **demo mode** with synthetic data. This is the default behavior on Streamlit Community Cloud.

## Running Locally

```bash
# From the waygraph root directory
pip install -e .
pip install -r requirements.txt

streamlit run app/app.py
```

The app will be available at `http://localhost:8501`.

To point the app at real WOMD scenario data, either:
- Set the `WAYGRAPH_DATA_DIR` environment variable to the directory containing `.pkl` files, or
- Enter the path in the sidebar text input after launching the app.

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Create a new app pointing to this repository with:
   - **Main file path:** `app/app.py`
   - **Python version:** 3.10 or 3.11
4. The app will install dependencies from `requirements.txt` and the waygraph package from `pyproject.toml` automatically.

The app launches in demo mode on the cloud since no `.pkl` data files are present.
