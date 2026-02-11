"""
Folium map visualization helpers for the WayGraph web app.

Provides geographic map views for OSM matching results.
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    import folium
    from folium import plugins
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


# Default map center (San Francisco)
DEFAULT_CENTER = [37.7749, -122.4194]
DEFAULT_ZOOM = 12

# City centers for quick navigation
CITY_CENTERS = {
    "San Francisco, CA": {"lat": 37.7749, "lon": -122.4194, "zoom": 13},
    "Phoenix, AZ": {"lat": 33.4484, "lon": -112.0740, "zoom": 12},
    "Montreal, QC": {"lat": 45.5017, "lon": -73.5673, "zoom": 13},
}


def create_match_map(
    match_results: List[Dict[str, Any]],
    center: Optional[Tuple[float, float]] = None,
    zoom: int = DEFAULT_ZOOM,
    height: int = 500,
) -> Optional["folium.Map"]:
    """Create a folium map showing matched OSM locations.

    Args:
        match_results: List of dicts with 'id', 'score', 'lat', 'lon'.
        center: Optional (lat, lon) for map center.
        zoom: Initial zoom level.
        height: Map height in pixels (unused directly, for reference).

    Returns:
        Folium Map object, or None if folium not available.
    """
    if not HAS_FOLIUM:
        return None

    # Determine center
    if center is not None:
        map_center = list(center)
    elif match_results:
        lats = [r["lat"] for r in match_results if r.get("lat", 0) != 0]
        lons = [r["lon"] for r in match_results if r.get("lon", 0) != 0]
        if lats and lons:
            map_center = [sum(lats) / len(lats), sum(lons) / len(lons)]
        else:
            map_center = DEFAULT_CENTER
    else:
        map_center = DEFAULT_CENTER

    m = folium.Map(
        location=map_center,
        zoom_start=zoom,
        tiles="CartoDB positron",
    )

    # Add match markers
    for i, result in enumerate(match_results):
        lat = result.get("lat", 0)
        lon = result.get("lon", 0)
        if lat == 0 and lon == 0:
            continue

        score = result.get("score", 0)
        node_id = result.get("id", "unknown")

        # Color by rank
        if i == 0:
            color = "red"
            icon_name = "star"
        elif i < 3:
            color = "orange"
            icon_name = "circle"
        else:
            color = "blue"
            icon_name = "circle"

        popup_html = f"""
        <b>Rank #{i+1}</b><br>
        OSM Node: {node_id}<br>
        Score: {score:.4f}<br>
        Lat: {lat:.5f}<br>
        Lon: {lon:.5f}
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"#{i+1}: {node_id} (score={score:.3f})",
            icon=folium.Icon(color=color, icon=icon_name, prefix="fa"),
        ).add_to(m)

    return m


def create_city_overview_map(
    city: str,
    intersection_locations: Optional[List[Dict]] = None,
) -> Optional["folium.Map"]:
    """Create a city overview map with optional intersection markers.

    Args:
        city: City name (key in CITY_CENTERS).
        intersection_locations: Optional list of dicts with 'lat', 'lon', 'type'.

    Returns:
        Folium Map object, or None if folium not available.
    """
    if not HAS_FOLIUM:
        return None

    city_info = CITY_CENTERS.get(city, CITY_CENTERS["San Francisco, CA"])
    m = folium.Map(
        location=[city_info["lat"], city_info["lon"]],
        zoom_start=city_info["zoom"],
        tiles="CartoDB positron",
    )

    # Add city boundary circle
    folium.Circle(
        location=[city_info["lat"], city_info["lon"]],
        radius=5000,
        color="blue",
        fill=False,
        opacity=0.3,
        tooltip=city,
    ).add_to(m)

    # Add intersection markers if available
    if intersection_locations:
        marker_cluster = plugins.MarkerCluster(name="Intersections").add_to(m)

        type_colors = {
            "T": "red",
            "cross": "green",
            "Y": "orange",
            "multi": "purple",
            "roundabout": "darkgreen",
            "merge": "blue",
            "none": "gray",
        }

        for loc in intersection_locations:
            lat = loc.get("lat", 0)
            lon = loc.get("lon", 0)
            if lat == 0 and lon == 0:
                continue

            itype = loc.get("type", "none")
            color = type_colors.get(itype, "gray")

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                tooltip=f"{itype} intersection",
            ).add_to(marker_cluster)

        folium.LayerControl().add_to(m)

    return m
