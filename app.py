"""
ISS Tracking and Pass Prediction Dashboard

Real-time analytics pipeline that tracks the International Space Station
and visualizes proximity to SatNOGS ground stations for potential
radio frequency interference prediction.

MIST6380 - Gleb Alikhver, Lucy Moon, Lexie-Anne Rodkey
"""

import time
from pathlib import Path

import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.capture import fetch_ground_stations, fetch_iss_position
from src.process import (
    append_position_to_history,
    build_trajectory_dataframe,
    load_positions_from_csv,
    process_iss_position,
    save_positions_to_csv,
)
from src.analyze import (
    find_nearby_stations,
    generate_interference_summary,
    predict_upcoming_passes,
)
from src.config import (
    PROXIMITY_THRESHOLD_KM,
    REFRESH_INTERVAL_SECONDS,
)

# -- Page configuration --
st.set_page_config(
    page_title="ISS Tracking Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "iss_history.csv"


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "position_history" not in st.session_state:
        st.session_state.position_history = load_positions_from_csv(
            HISTORY_FILE
        )
    if "ground_stations" not in st.session_state:
        st.session_state.ground_stations = None
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = 0


def load_ground_stations_cached() -> pd.DataFrame:
    """Load ground stations once and cache in session state."""
    if st.session_state.ground_stations is None:
        with st.spinner("Loading SatNOGS ground stations..."):
            stations = fetch_ground_stations()
            if stations is not None:
                st.session_state.ground_stations = stations
            else:
                st.session_state.ground_stations = pd.DataFrame()
    return st.session_state.ground_stations


def render_sidebar(iss_data: dict, summary: dict) -> float:
    """Render the sidebar with ISS info, controls, and alert summary."""
    st.sidebar.title("ISS Tracker")
    st.sidebar.markdown("---")

    # ISS current info
    st.sidebar.subheader("Current ISS Position")
    if iss_data:
        st.sidebar.metric("Latitude", f"{iss_data['latitude']:.4f}")
        st.sidebar.metric("Longitude", f"{iss_data['longitude']:.4f}")
        st.sidebar.metric("Altitude", f"{iss_data['altitude']:.1f} km")
        st.sidebar.metric("Speed", f"{iss_data['speed_kmh']:.0f} km/h")
        st.sidebar.text(f"Updated: {iss_data['datetime_utc']}")
    else:
        st.sidebar.warning("Unable to fetch ISS position")

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

    st.sidebar.markdown("---")

    # Alert summary
    st.sidebar.subheader("Alert Summary")
    if summary:
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Nearby Stations", summary["current_nearby_count"])
        col2.metric("Predicted Passes", summary["predicted_passes_count"])

        if summary["critical_count"] > 0:
            st.sidebar.error(
                f"{summary['critical_count']} CRITICAL proximity alert(s)"
            )
        if summary["warning_count"] > 0:
            st.sidebar.warning(
                f"{summary['warning_count']} WARNING proximity alert(s)"
            )
        if summary["nearest_station"] != "None":
            st.sidebar.info(
                f"Nearest: {summary['nearest_station']} "
                f"({summary['nearest_distance_km']} km)"
            )

    if auto_refresh:
        st.sidebar.caption(
            f"Refreshing every {REFRESH_INTERVAL_SECONDS}s"
        )

    return threshold


def build_map_layer_iss(iss_data: dict) -> pdk.Layer:
    """Create the pydeck layer for the ISS marker."""
    return pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame(
            [
                {
                    "lat": iss_data["latitude"],
                    "lon": iss_data["longitude"],
                    "name": "ISS",
                }
            ]
        ),
        get_position=["lon", "lat"],
        get_radius=80000,
        get_fill_color=[255, 0, 0, 200],
        pickable=True,
    )


def build_map_layer_trajectory(trajectory_df: pd.DataFrame) -> pdk.Layer:
    """Create the pydeck layer for the ISS trajectory trail."""
    if trajectory_df.empty or len(trajectory_df) < 2:
        return None

    path_data = [
        {
            "path": [
                [row["longitude"], row["latitude"]]
                for _, row in trajectory_df.iterrows()
            ],
            "name": "ISS Trajectory",
        }
    ]

    return pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        get_color=[255, 100, 100, 150],
        width_min_pixels=2,
        pickable=True,
    )


def build_map_layer_stations(
    stations_df: pd.DataFrame,
    nearby_df: pd.DataFrame,
) -> list[pdk.Layer]:
    """Create pydeck layers for ground stations (normal + alerted)."""
    layers = []

    if stations_df.empty:
        return layers

    # IDs of nearby stations for highlighting
    nearby_ids = set()
    if not nearby_df.empty and "station_id" in nearby_df.columns:
        nearby_ids = set(nearby_df["station_id"].tolist())

    # Normal stations (blue)
    normal = stations_df[
        ~stations_df["station_id"].isin(nearby_ids)
    ].copy()
    if not normal.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=normal,
                get_position=["longitude", "latitude"],
                get_radius=30000,
                get_fill_color=[30, 144, 255, 140],
                pickable=True,
            )
        )

    # Nearby stations (yellow/red based on alert level)
    if not nearby_df.empty:
        alert_colors = {
            "CRITICAL": [255, 0, 0, 220],
            "WARNING": [255, 165, 0, 200],
            "WATCH": [255, 255, 0, 180],
        }
        for level, color in alert_colors.items():
            level_df = nearby_df[nearby_df["alert_level"] == level]
            if not level_df.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=level_df,
                        get_position=["longitude", "latitude"],
                        get_radius=50000,
                        get_fill_color=color,
                        pickable=True,
                    )
                )

    return layers


def render_map(
    iss_data: dict,
    trajectory_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    nearby_df: pd.DataFrame,
) -> None:
    """Render the main world map with all layers."""
    layers = []

    # ISS marker
    layers.append(build_map_layer_iss(iss_data))

    # Trajectory trail
    traj_layer = build_map_layer_trajectory(trajectory_df)
    if traj_layer:
        layers.append(traj_layer)

    # Ground stations
    layers.extend(build_map_layer_stations(stations_df, nearby_df))

    view_state = pdk.ViewState(
        latitude=iss_data["latitude"],
        longitude=iss_data["longitude"],
        zoom=1.5,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip={"text": "{name}"},
    )

    st.pydeck_chart(deck, use_container_width=True)


def render_proximity_alerts(nearby_df: pd.DataFrame) -> None:
    """Render the proximity alerts table."""
    st.subheader("Current Proximity Alerts")

    if nearby_df.empty:
        st.success("No ground stations within proximity threshold.")
        return

    display_cols = ["name", "distance_km", "alert_level"]
    available = [c for c in display_cols if c in nearby_df.columns]
    display_df = nearby_df[available].copy()

    if "distance_km" in display_df.columns:
        display_df["distance_km"] = display_df["distance_km"].round(1)

    column_config = {
        "name": st.column_config.TextColumn("Station Name"),
        "distance_km": st.column_config.NumberColumn(
            "Distance (km)", format="%.1f"
        ),
        "alert_level": st.column_config.TextColumn("Alert Level"),
    }

    st.dataframe(
        display_df,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
    )


def render_predicted_passes(passes_df: pd.DataFrame) -> None:
    """Render the predicted passes table."""
    st.subheader("Predicted Upcoming Passes (Next 90 Minutes)")

    if passes_df.empty:
        st.info("No predicted passes within the threshold window.")
        return

    display_cols = [
        "station_name",
        "estimated_distance_km",
        "projected_time_utc",
        "alert_level",
    ]
    available = [c for c in display_cols if c in passes_df.columns]

    st.dataframe(
        passes_df[available],
        column_config={
            "station_name": "Station",
            "estimated_distance_km": st.column_config.NumberColumn(
                "Est. Distance (km)", format="%.1f"
            ),
            "projected_time_utc": "Projected Time (UTC)",
            "alert_level": "Alert Level",
        },
        use_container_width=True,
        hide_index=True,
    )


def render_speed_chart(trajectory_df: pd.DataFrame) -> None:
    """Render the speed over time chart."""
    st.subheader("ISS Speed Over Time")

    if trajectory_df.empty or "velocity" not in trajectory_df.columns:
        st.info("Collecting speed data... refresh to accumulate points.")
        return

    if len(trajectory_df) < 2:
        st.info("Need at least 2 data points for speed chart.")
        return

    df = trajectory_df.copy()
    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    fig = px.line(
        df,
        x="time",
        y="velocity",
        labels={"time": "Time (UTC)", "velocity": "Speed (km/h)"},
    )
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_altitude_chart(trajectory_df: pd.DataFrame) -> None:
    """Render the altitude over time chart."""
    st.subheader("ISS Altitude Over Time")

    if trajectory_df.empty or "altitude" not in trajectory_df.columns:
        st.info("Collecting altitude data...")
        return

    if len(trajectory_df) < 2:
        st.info("Need at least 2 data points for altitude chart.")
        return

    df = trajectory_df.copy()
    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    fig = px.area(
        df,
        x="time",
        y="altitude",
        labels={"time": "Time (UTC)", "altitude": "Altitude (km)"},
    )
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig.update_traces(
        line_color="#1E90FF",
        fillcolor="rgba(30,144,255,0.2)",
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Main dashboard entry point."""
    init_session_state()

    # -- Fetch data --
    raw_position = fetch_iss_position()

    if raw_position is None:
        st.error(
            "Could not connect to the ISS tracking API. "
            "Please check your internet connection and try again."
        )
        return

    iss_data = process_iss_position(raw_position)

    # Update position history
    st.session_state.position_history = append_position_to_history(
        st.session_state.position_history, raw_position
    )
    save_positions_to_csv(
        st.session_state.position_history, HISTORY_FILE
    )

    trajectory_df = build_trajectory_dataframe(
        st.session_state.position_history
    )

    # Load ground stations
    stations_df = load_ground_stations_cached()

    # -- Analysis --
    nearby_df = pd.DataFrame()
    passes_df = pd.DataFrame()
    summary = {
        "current_nearby_count": 0,
        "critical_count": 0,
        "warning_count": 0,
        "predicted_passes_count": 0,
        "nearest_station": "None",
        "nearest_distance_km": None,
    }

    threshold = render_sidebar(iss_data, summary)

    if not stations_df.empty:
        nearby_df = find_nearby_stations(
            iss_data["latitude"],
            iss_data["longitude"],
            stations_df,
            threshold_km=threshold,
        )
        passes_df = predict_upcoming_passes(
            trajectory_df, stations_df, threshold_km=threshold
        )
        summary = generate_interference_summary(nearby_df, passes_df)

    # -- Dashboard layout --
    st.title("Real-Time ISS Tracking & Pass Prediction Dashboard")
    st.caption(
        "Tracking the International Space Station and predicting "
        "ground station proximity for RF interference analysis"
    )

    # Map
    render_map(iss_data, trajectory_df, stations_df, nearby_df)

    # Alerts and predictions side by side
    col1, col2 = st.columns(2)
    with col1:
        render_proximity_alerts(nearby_df)
    with col2:
        render_predicted_passes(passes_df)

    # Charts side by side
    col3, col4 = st.columns(2)
    with col3:
        render_speed_chart(trajectory_df)
    with col4:
        render_altitude_chart(trajectory_df)

    # Pipeline info expander
    with st.expander("About This Pipeline"):
        st.markdown(
            """
**Data Pipeline Stages:**

1. **Capture** -- Live ISS position data from the Where the ISS At
   API; ground station locations from the SatNOGS Network API.
2. **Process** -- Timestamps converted to UTC, coordinate data
   structured into DataFrames, speed and trajectory features
   engineered.
3. **Store** -- Position history persisted to CSV; ground station
   data cached in session state.
4. **Analyze** -- Haversine distance computed between ISS and each
   ground station. Stations within the proximity threshold are
   flagged. Trajectory projection estimates upcoming passes.
5. **Communicate** -- This interactive Streamlit dashboard
   visualizes all pipeline outputs in real time.

**Data Sources:**
- [Where the ISS At API](https://wheretheiss.at/w/developer)
- [SatNOGS Network API](https://network.satnogs.org/api/)

**Authors:** Gleb Alikhver, Lucy Moon, Lexie-Anne Rodkey
-- MIST6380
            """
        )

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh enabled",
        value=True,
        key="auto_refresh_toggle",
    )
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
