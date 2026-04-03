"""
ISS Tracking and Pass Prediction Dashboard

Real-time analytics pipeline that tracks the International Space Station
and other NASA spacecraft, visualizing proximity to SatNOGS ground stations
for potential radio frequency interference prediction.

MIST6380 - Gleb Alikhver, Lucy Moon, Lexie-Anne Rodkey
"""

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.capture import (
    fetch_all_spacecraft,
    fetch_ground_stations,
    geocode_address,
)
from src.process import build_trajectory_dataframe, process_position
from src.analyze import (
    find_all_spacecraft_nearby,
    predict_passes_over_location,
    generate_interference_summary,
    predict_upcoming_passes,
)
from src.config import (
    NASA_SPACECRAFT,
    PROXIMITY_THRESHOLD_KM,
    REFRESH_INTERVAL_SECONDS,
)

# -- Page config --
st.set_page_config(
    page_title="ISS Tracking Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Build a color lookup from config
SPACECRAFT_COLORS = {name: color for _, name, color in NASA_SPACECRAFT}


def init_session_state() -> None:
    """Initialize session state on first load."""
    if "ground_stations" not in st.session_state:
        st.session_state.ground_stations = None
    if "spacecraft_data" not in st.session_state:
        st.session_state.spacecraft_data = {}


def load_ground_stations() -> pd.DataFrame:
    """Load ground stations once and cache in session state."""
    if st.session_state.ground_stations is None:
        with st.spinner("Loading SatNOGS ground stations..."):
            stations = fetch_ground_stations()
            st.session_state.ground_stations = (
                stations if stations is not None else pd.DataFrame()
            )
    return st.session_state.ground_stations


def build_globe(
    spacecraft_data: dict,
    stations_df: pd.DataFrame,
    nearby_df: pd.DataFrame,
    center_on: str = "ISS",
) -> go.Figure:
    """Build the Plotly globe with spacecraft, trajectories, and stations."""
    fig = go.Figure()

    # -- Ground stations (blue dots) --
    if not stations_df.empty:
        # Separate normal vs alerted stations
        alerted_ids = set()
        if not nearby_df.empty and "station_id" in nearby_df.columns:
            alerted_ids = set(nearby_df["station_id"].tolist())

        normal = stations_df[~stations_df["station_id"].isin(alerted_ids)]
        alerted = stations_df[stations_df["station_id"].isin(alerted_ids)]

        if not normal.empty:
            fig.add_trace(go.Scattergeo(
                lat=normal["latitude"],
                lon=normal["longitude"],
                mode="markers",
                marker=dict(size=3, color="#1E90FF", opacity=0.5),
                name="Ground Stations",
                hovertext=normal["name"],
                hoverinfo="text",
            ))

        if not alerted.empty:
            fig.add_trace(go.Scattergeo(
                lat=alerted["latitude"],
                lon=alerted["longitude"],
                mode="markers",
                marker=dict(
                    size=7,
                    color="#FFD700",
                    opacity=0.9,
                    symbol="diamond",
                ),
                name="Alerted Stations",
                hovertext=alerted["name"],
                hoverinfo="text",
            ))

    # -- Spacecraft trajectories and markers --
    center_lat, center_lon = 0, 0

    for sc_name, positions in spacecraft_data.items():
        if not positions:
            continue

        color = SPACECRAFT_COLORS.get(sc_name, "#FFFFFF")
        latest = positions[-1]

        if sc_name == center_on:
            center_lat = latest["latitude"]
            center_lon = latest["longitude"]

        # Trajectory line
        if len(positions) > 1:
            traj_df = build_trajectory_dataframe(positions)
            fig.add_trace(go.Scattergeo(
                lat=traj_df["latitude"],
                lon=traj_df["longitude"],
                mode="lines",
                line=dict(color=color, width=2),
                name=f"{sc_name} path",
                showlegend=False,
                hoverinfo="skip",
            ))

        # Current position marker
        fig.add_trace(go.Scattergeo(
            lat=[latest["latitude"]],
            lon=[latest["longitude"]],
            mode="markers+text",
            marker=dict(size=10, color=color, symbol="star"),
            text=[sc_name],
            textposition="top right",
            textfont=dict(color=color, size=10),
            name=sc_name,
            hovertext=(
                f"{sc_name}<br>"
                f"Lat: {latest['latitude']:.2f}<br>"
                f"Lon: {latest['longitude']:.2f}<br>"
                f"Alt: {latest['altitude']:.0f} km"
            ),
            hoverinfo="text",
        ))

    # -- Globe styling --
    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=center_lon, lat=center_lat),
        showocean=True,
        oceancolor="rgb(0, 40, 100)",
        showland=True,
        landcolor="rgb(25, 70, 35)",
        showcountries=True,
        countrycolor="rgba(150, 150, 150, 0.4)",
        showlakes=True,
        lakecolor="rgb(0, 40, 100)",
        showrivers=False,
        bgcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def render_sidebar(
    spacecraft_data: dict, summary: dict
) -> tuple[float, bool, str]:
    """Render sidebar with spacecraft info, controls, and alerts."""
    st.sidebar.title("ISS Tracker")
    st.sidebar.markdown("---")

    # Show ISS info prominently if available
    iss_positions = spacecraft_data.get("ISS", [])
    if iss_positions:
        iss = process_position(iss_positions[-1])
        st.sidebar.subheader("ISS Position")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Lat", f"{iss['latitude']:.2f}")
        c2.metric("Lon", f"{iss['longitude']:.2f}")
        c1.metric("Alt", f"{iss['altitude']:.0f} km")
        c2.metric("Speed", f"{iss['speed_kmh']:,.0f}")
        st.sidebar.caption(f"Updated: {iss['datetime_utc']}")

    st.sidebar.markdown("---")

    # Active spacecraft count
    st.sidebar.subheader("Tracking")
    st.sidebar.metric(
        "Active Spacecraft", len(spacecraft_data)
    )
    for name in spacecraft_data:
        color = SPACECRAFT_COLORS.get(name, "#FFF")
        st.sidebar.markdown(
            f"<span style='color:{color}'>&#9733;</span> {name}",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")

    # Controls
    st.sidebar.subheader("Controls")
    threshold = st.sidebar.slider(
        "Proximity threshold (km)",
        min_value=100,
        max_value=2000,
        value=PROXIMITY_THRESHOLD_KM,
        step=50,
    )
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)

    # Globe center selection
    sc_names = list(spacecraft_data.keys())
    center_on = st.sidebar.selectbox(
        "Center globe on",
        sc_names if sc_names else ["ISS"],
    )

    st.sidebar.markdown("---")

    # Alert summary
    st.sidebar.subheader("Alerts")
    if summary["current_nearby_count"] > 0:
        if summary["critical_count"] > 0:
            st.sidebar.error(
                f"{summary['critical_count']} CRITICAL alerts"
            )
        if summary["warning_count"] > 0:
            st.sidebar.warning(
                f"{summary['warning_count']} WARNING alerts"
            )
        st.sidebar.metric(
            "Stations in range", summary["current_nearby_count"]
        )
        st.sidebar.metric(
            "Predicted passes", summary["predicted_passes_count"]
        )
    else:
        st.sidebar.success("No proximity alerts")

    return threshold, auto_refresh, center_on


def render_alerts_table(nearby_df: pd.DataFrame) -> None:
    """Render the current proximity alerts table."""
    st.subheader("Current Proximity Alerts")

    if nearby_df.empty:
        st.success("No ground stations within proximity threshold.")
        return

    display_cols = [
        c for c in ["spacecraft", "name", "distance_km", "alert_level"]
        if c in nearby_df.columns
    ]
    display = nearby_df[display_cols].copy()
    if "distance_km" in display.columns:
        display["distance_km"] = display["distance_km"].round(1)

    st.dataframe(
        display,
        column_config={
            "spacecraft": "Spacecraft",
            "name": "Station",
            "distance_km": st.column_config.NumberColumn(
                "Distance (km)", format="%.1f"
            ),
            "alert_level": "Alert",
        },
        use_container_width=True,
        hide_index=True,
    )


def render_passes_table(passes_df: pd.DataFrame) -> None:
    """Render predicted passes table."""
    st.subheader("Predicted Passes (Next 90 Min)")

    if passes_df.empty:
        st.info("No predicted passes within threshold.")
        return

    display_cols = [
        c for c in [
            "spacecraft", "station_name",
            "estimated_distance_km", "projected_time_utc", "alert_level"
        ]
        if c in passes_df.columns
    ]

    st.dataframe(
        passes_df[display_cols].head(20),
        column_config={
            "spacecraft": "Spacecraft",
            "station_name": "Station",
            "estimated_distance_km": st.column_config.NumberColumn(
                "Est. Distance (km)", format="%.1f"
            ),
            "projected_time_utc": "Time (UTC)",
            "alert_level": "Alert",
        },
        use_container_width=True,
        hide_index=True,
    )


def render_speed_chart(spacecraft_data: dict) -> None:
    """Render speed over time for all tracked spacecraft."""
    st.subheader("Spacecraft Speed Over Time")

    fig = go.Figure()
    has_data = False

    for sc_name, positions in spacecraft_data.items():
        if len(positions) < 2:
            continue
        df = build_trajectory_dataframe(positions)
        if "velocity" not in df.columns or df["velocity"].sum() == 0:
            continue
        df = df[df["velocity"] > 0]
        if df.empty:
            continue

        has_data = True
        df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        color = SPACECRAFT_COLORS.get(sc_name, "#FFFFFF")
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["velocity"],
            mode="lines",
            name=sc_name,
            line=dict(color=color, width=2),
        ))

    if not has_data:
        st.info("Collecting speed data across refreshes...")
        return

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Time (UTC)",
        yaxis_title="Speed (km/h)",
        legend=dict(x=0, y=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_altitude_chart(spacecraft_data: dict) -> None:
    """Render altitude over time for all tracked spacecraft."""
    st.subheader("Spacecraft Altitude Over Time")

    fig = go.Figure()
    has_data = False

    for sc_name, positions in spacecraft_data.items():
        if len(positions) < 2:
            continue
        df = build_trajectory_dataframe(positions)
        if "altitude" not in df.columns:
            continue

        has_data = True
        df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        color = SPACECRAFT_COLORS.get(sc_name, "#FFFFFF")
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["altitude"],
            mode="lines",
            name=sc_name,
            line=dict(color=color, width=2),
        ))

    if not has_data:
        st.info("Collecting altitude data...")
        return

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Time (UTC)",
        yaxis_title="Altitude (km)",
        legend=dict(x=0, y=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Main dashboard entry point."""
    init_session_state()

    # -- Fetch all data --
    with st.spinner("Fetching spacecraft positions from NASA..."):
        spacecraft_data = fetch_all_spacecraft()

    if not spacecraft_data:
        st.error(
            "Could not connect to spacecraft tracking APIs. "
            "Check your internet connection and try again."
        )
        return

    stations_df = load_ground_stations()

    # -- Analysis --
    nearby_df = pd.DataFrame()
    passes_df = pd.DataFrame()
    summary = {
        "current_nearby_count": 0,
        "critical_count": 0,
        "warning_count": 0,
        "predicted_passes_count": 0,
        "spacecraft_with_alerts": 0,
    }

    # Sidebar (renders controls, returns user selections)
    threshold, auto_refresh, center_on = render_sidebar(
        spacecraft_data, summary
    )

    # Run analysis with user-selected threshold
    if not stations_df.empty:
        nearby_df = find_all_spacecraft_nearby(
            spacecraft_data, stations_df, threshold_km=threshold
        )

        # Predict passes for each spacecraft
        all_passes = []
        for sc_name, positions in spacecraft_data.items():
            traj_df = build_trajectory_dataframe(positions)
            sc_passes = predict_upcoming_passes(
                traj_df, stations_df, sc_name, threshold_km=threshold
            )
            if not sc_passes.empty:
                all_passes.append(sc_passes)

        passes_df = (
            pd.concat(all_passes, ignore_index=True)
            if all_passes else pd.DataFrame()
        )

        summary = generate_interference_summary(nearby_df, passes_df)

    # -- Layout --
    st.title("Real-Time ISS Tracking & Pass Prediction Dashboard")
    st.caption(
        "Tracking NASA spacecraft and predicting ground station "
        "proximity for RF interference analysis"
    )

    # Globe map
    fig = build_globe(
        spacecraft_data, stations_df, nearby_df, center_on=center_on
    )
    st.plotly_chart(fig, use_container_width=True)

    # Alerts and predictions
    col1, col2 = st.columns(2)
    with col1:
        render_alerts_table(nearby_df)
    with col2:
        render_passes_table(passes_df)

    # -- What's Passing Over Me? --
    st.markdown("---")
    st.subheader("What's Passing Over Me?")
    st.caption(
        "Enter a US address to see which spacecraft will pass "
        "near your location in the next 90 minutes."
    )

    address_input = st.text_input(
        "Enter your address",
        placeholder="e.g. 1600 Pennsylvania Ave, Washington, DC",
        key="user_address",
    )

    if address_input:
        with st.spinner("Geocoding address..."):
            location = geocode_address(address_input)

        if location is None:
            st.error(
                "Could not find that address. Please try a valid "
                "US street address (e.g. 123 Main St, Houston, TX)."
            )
        else:
            st.success(
                f"Matched: **{location['matched_address']}** "
                f"({location['latitude']:.4f}, "
                f"{location['longitude']:.4f})"
            )

            user_passes = predict_passes_over_location(
                location["latitude"],
                location["longitude"],
                spacecraft_data,
                threshold_km=threshold,
            )

            if user_passes.empty:
                st.info(
                    "No spacecraft predicted to pass within "
                    f"{threshold} km of your location in the "
                    "next 90 minutes. Try increasing the "
                    "proximity threshold in the sidebar."
                )
            else:
                st.dataframe(
                    user_passes,
                    column_config={
                        "spacecraft": "Spacecraft",
                        "estimated_distance_km":
                            st.column_config.NumberColumn(
                                "Closest Approach (km)",
                                format="%.1f",
                            ),
                        "projected_time_utc": "Estimated Time (UTC)",
                        "minutes_from_now":
                            st.column_config.NumberColumn(
                                "Minutes From Now",
                            ),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

    # Charts
    col3, col4 = st.columns(2)
    with col3:
        render_speed_chart(spacecraft_data)
    with col4:
        render_altitude_chart(spacecraft_data)

    # Pipeline info
    with st.expander("About This Pipeline"):
        st.markdown(
            """
**Data Pipeline Stages:**

1. **Capture** -- Real-time ISS position from the Where the ISS At
   API. Orbital data for additional NASA spacecraft (Aqua, Aura,
   Landsat 8/9, NOAA-20, Suomi NPP) from the NASA Satellite
   Situation Center (SSC) API. Ground station locations from the
   SatNOGS Network API.

2. **Process** -- Raw GEO cartesian coordinates (X, Y, Z in km)
   converted to geographic latitude, longitude, and altitude using
   trigonometric transformations. Timestamps parsed and
   standardized to UTC. Speed estimated from sequential positions.

3. **Store** -- Ground station data cached in session state.
   Spacecraft position histories maintained in memory across
   dashboard refreshes.

4. **Analyze** -- Haversine distance computed between every
   spacecraft and every ground station. Stations within the
   proximity threshold flagged with CRITICAL/WARNING/WATCH
   severity. Trajectory projection predicts upcoming passes
   over the next 90 minutes. User addresses geocoded via the
   US Census Bureau Geocoder API for personal pass predictions.

5. **Communicate** -- This interactive Streamlit dashboard with
   a 3D globe, proximity alerts, pass predictions, and
   speed/altitude charts.

**Data Sources:**
- [Where the ISS At API](https://wheretheiss.at/w/developer)
  -- real-time ISS tracking (no API key required)
- [NASA Satellite Situation Center](https://sscweb.gsfc.nasa.gov/)
  -- orbital data for NASA spacecraft (no API key required)
- [SatNOGS Network API](https://network.satnogs.org/api/)
  -- global ground station locations (no API key required)
- [US Census Bureau Geocoder](https://geocoding.geo.census.gov/)
  -- address-to-coordinate conversion (no API key required)

**Authors:** Gleb Alikhver, Lucy Moon, Lexie-Anne Rodkey -- MIST6380
            """
        )

    # Auto-refresh
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
