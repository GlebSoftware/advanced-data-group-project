"""
ISS Tracking and Pass Prediction Dashboard

Real-time analytics pipeline that tracks the International Space Station
and other NASA spacecraft, visualizing proximity to SatNOGS ground stations
for potential radio frequency interference prediction.

MIST6380 - Gleb Alikhver, Lucy Moon, Lexie-Anne Rodkey
"""

from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.capture import (
    fetch_all_spacecraft,
    fetch_ground_stations,
    geocode_address,
)
from src.process import build_trajectory_dataframe, process_position
from src.analyze import (
    predict_passes_over_location,
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

# Eastern Time offset (UTC-4 for EDT, UTC-5 for EST)
ET_OFFSET = timedelta(hours=-4)
ET_TZ = timezone(ET_OFFSET)

# Athens, GA coordinates
ATHENS_GA = {"latitude": 33.9519, "longitude": -83.3576}


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_spacecraft_cached() -> dict:
    """Fetch spacecraft positions with 30s cache to avoid API calls on every rerun."""
    return fetch_all_spacecraft()


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


def filter_spacecraft_by_altitude(
    spacecraft_data: dict,
    alt_range: tuple[float, float],
) -> dict:
    """Filter spacecraft to only those orbiting within the altitude range."""
    min_alt, max_alt = alt_range
    filtered = {}
    for sc_name, positions in spacecraft_data.items():
        if not positions:
            continue
        latest = positions[-1]
        if min_alt <= latest["altitude"] <= max_alt:
            filtered[sc_name] = positions
    return filtered


def to_eastern(utc_dt: datetime) -> str:
    """Convert a UTC datetime to Eastern Time string."""
    eastern = utc_dt.astimezone(ET_TZ)
    return eastern.strftime("%I:%M:%S %p ET")


def build_globe(
    spacecraft_data: dict,
    stations_df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
) -> go.Figure:
    """Build the Plotly globe with spacecraft, trajectories, and stations."""
    fig = go.Figure()

    # -- Ground stations (blue dots) --
    if not stations_df.empty:
        fig.add_trace(go.Scattergeo(
            lat=stations_df["latitude"],
            lon=stations_df["longitude"],
            mode="markers",
            marker=dict(size=3, color="#1E90FF", opacity=0.5),
            name="Ground Stations",
            hovertext=stations_df["name"],
            hoverinfo="text",
        ))

    # -- Spacecraft trajectories and markers --
    for sc_name, positions in spacecraft_data.items():
        if not positions:
            continue

        color = SPACECRAFT_COLORS.get(sc_name, "#FFFFFF")
        latest = positions[-1]

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
        uirevision="globe",
    )

    return fig


def render_sidebar_controls() -> None:
    """Render sidebar controls. Widget values stored in session state."""
    st.sidebar.title("ISS Tracker")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Controls")
    st.sidebar.slider(
        "Altitude range (km)",
        min_value=400,
        max_value=900,
        value=(400, 900),
        step=50,
        key="alt_threshold",
        help="Show only spacecraft orbiting within this altitude range",
    )


def render_sidebar_data(spacecraft_data: dict) -> None:
    """Render sidebar data: ISS position and tracked spacecraft list."""
    iss_positions = spacecraft_data.get("ISS", [])
    if iss_positions:
        iss = process_position(iss_positions[-1])
        st.sidebar.subheader("ISS Position")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Lat", f"{iss['latitude']:.2f}")
        c2.metric("Lon", f"{iss['longitude']:.2f}")
        c1.metric("Alt", f"{iss['altitude']:.0f} km")
        c2.metric("Speed", f"{iss['speed_kmh']:,.0f}")
        utc_dt = datetime.fromtimestamp(
            iss_positions[-1]["timestamp"], tz=timezone.utc
        )
        st.sidebar.caption(f"Updated: {to_eastern(utc_dt)}")

    st.sidebar.markdown("---")

    st.sidebar.subheader("Tracking")
    st.sidebar.metric("Visible Spacecraft", len(spacecraft_data))
    for name in spacecraft_data:
        color = SPACECRAFT_COLORS.get(name, "#FFF")
        st.sidebar.markdown(
            f"<span style='color:{color}'>&#9733;</span> {name}",
            unsafe_allow_html=True,
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
        df["time"] = pd.to_datetime(
            df["timestamp"], unit="s", utc=True
        ).dt.tz_convert(ET_TZ)
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
        xaxis_title="Time (ET)",
        yaxis_title="Speed (km/h)",
        legend=dict(x=0, y=1),
    )
    st.plotly_chart(fig, width="stretch")


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
        df["time"] = pd.to_datetime(
            df["timestamp"], unit="s", utc=True
        ).dt.tz_convert(ET_TZ)
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
        xaxis_title="Time (ET)",
        yaxis_title="Altitude (km)",
        legend=dict(x=0, y=1),
    )
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    """Main dashboard entry point."""
    init_session_state()

    # -- Auto-refresh via JS timer --
    st_autorefresh(
        interval=REFRESH_INTERVAL_SECONDS * 1000,
        key="auto_refresh",
    )

    # -- Suppress rerun dimming --
    st.markdown(
        """<style>
        [data-testid="stStatusWidget"] { display: none !important; }
        .stApp div, .stApp section, .stApp iframe {
            opacity: 1 !important;
            transition: opacity 0s !important;
        }
        [data-stale], [data-stale="true"] {
            opacity: 1 !important;
        }
        .element-container {
            opacity: 1 !important;
            transition: none !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    stations_df = load_ground_stations()

    # -- Sidebar controls --
    render_sidebar_controls()

    # -- Fetch spacecraft data (cached 30s) --
    spacecraft_data = _fetch_spacecraft_cached()
    if not spacecraft_data:
        spacecraft_data = st.session_state.get("spacecraft_data", {})
    if not spacecraft_data:
        st.error(
            "Could not connect to spacecraft tracking APIs. "
            "Check your internet connection and try again."
        )
        return

    st.session_state.spacecraft_data = spacecraft_data

    # -- Center selection (spacecraft + Athens, GA) --
    sc_names = list(spacecraft_data.keys())
    center_options = sc_names + ["Athens, GA"]
    center_on = st.sidebar.selectbox(
        "Center globe on",
        center_options,
        key="center_select",
    )

    # Resolve center coordinates
    if center_on == "Athens, GA":
        center_lat = ATHENS_GA["latitude"]
        center_lon = ATHENS_GA["longitude"]
    else:
        center_positions = spacecraft_data.get(center_on, [])
        if center_positions:
            center_lat = center_positions[-1]["latitude"]
            center_lon = center_positions[-1]["longitude"]
        else:
            center_lat, center_lon = 0.0, 0.0

    # -- Filter spacecraft by altitude --
    alt_range = st.session_state.get("alt_threshold", (400, 900))
    visible_data = filter_spacecraft_by_altitude(spacecraft_data, alt_range)

    # -- Sidebar data --
    render_sidebar_data(visible_data)

    # -- Page header --
    st.title("Real-Time ISS Tracking & Pass Prediction Dashboard")
    st.caption(
        "Tracking NASA spacecraft and predicting ground station "
        "proximity for RF interference analysis"
    )

    # -- Globe --
    fig = build_globe(visible_data, stations_df, center_lat, center_lon)
    st.plotly_chart(fig, width="stretch", key="globe")

    # -- What's Passing Over Me? --
    st.markdown("---")
    st.subheader("What's Passing Over Me?")
    st.caption(
        "Enter a US address to see which spacecraft will pass "
        "within 500 km of your location in the next 90 minutes."
    )

    address_input = st.text_input(
        "Enter your address",
        placeholder="e.g. 1600 Pennsylvania Ave, Washington, DC",
        key="user_address",
    )

    if address_input:
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
                threshold_km=PROXIMITY_THRESHOLD_KM,
            )

            if user_passes.empty:
                st.info(
                    "No spacecraft predicted to pass within "
                    f"{PROXIMITY_THRESHOLD_KM} km of your location "
                    "in the next 90 minutes."
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
                        "projected_time_et": "Estimated Time (ET)",
                        "minutes_from_now":
                            st.column_config.NumberColumn(
                                "Minutes From Now",
                            ),
                    },
                    width="stretch",
                    hide_index=True,
                )

    # -- Charts --
    col1, col2 = st.columns(2)
    with col1:
        render_speed_chart(visible_data)
    with col2:
        render_altitude_chart(visible_data)

    # -- Pipeline info --
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


if __name__ == "__main__":
    main()
