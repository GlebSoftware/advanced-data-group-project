"""
Analysis module for the ISS Tracking Dashboard.

Handles proximity detection between spacecraft and ground stations,
pass prediction, and interference window identification.
"""

from datetime import datetime, timezone

import pandas as pd
from haversine import haversine, Unit

from src.config import PROXIMITY_THRESHOLD_KM


def compute_station_distances(
    spacecraft_lat: float,
    spacecraft_lon: float,
    stations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate distance from a spacecraft to every ground station.

    Adds a 'distance_km' column and sorts by distance ascending.
    """
    df = stations_df.copy()
    df["distance_km"] = df.apply(
        lambda row: haversine(
            (spacecraft_lat, spacecraft_lon),
            (row["latitude"], row["longitude"]),
            unit=Unit.KILOMETERS,
        ),
        axis=1,
    )
    return df.sort_values("distance_km").reset_index(drop=True)


def find_nearby_stations(
    spacecraft_lat: float,
    spacecraft_lon: float,
    stations_df: pd.DataFrame,
    threshold_km: float = PROXIMITY_THRESHOLD_KM,
) -> pd.DataFrame:
    """Identify ground stations within the proximity threshold."""
    df = compute_station_distances(
        spacecraft_lat, spacecraft_lon, stations_df
    )
    nearby = df[df["distance_km"] <= threshold_km].copy()
    nearby["alert_level"] = nearby["distance_km"].apply(
        _classify_alert_level
    )
    return nearby


def find_all_spacecraft_nearby(
    spacecraft_positions: dict,
    stations_df: pd.DataFrame,
    threshold_km: float = PROXIMITY_THRESHOLD_KM,
) -> pd.DataFrame:
    """Find nearby stations for ALL spacecraft at once.

    Returns a combined DataFrame with a 'spacecraft' column indicating
    which spacecraft triggered the proximity alert.
    """
    all_nearby = []

    for sc_name, positions in spacecraft_positions.items():
        if not positions:
            continue
        latest = positions[-1]
        nearby = find_nearby_stations(
            latest["latitude"],
            latest["longitude"],
            stations_df,
            threshold_km,
        )
        if not nearby.empty:
            nearby = nearby.copy()
            nearby["spacecraft"] = sc_name
            all_nearby.append(nearby)

    if not all_nearby:
        return pd.DataFrame()

    return pd.concat(all_nearby, ignore_index=True).sort_values(
        "distance_km"
    )


def _classify_alert_level(distance_km: float) -> str:
    """Classify proximity alert severity based on distance."""
    if distance_km <= 100:
        return "CRITICAL"
    elif distance_km <= 250:
        return "WARNING"
    return "WATCH"


def predict_upcoming_passes(
    trajectory_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    spacecraft_name: str = "ISS",
    threshold_km: float = PROXIMITY_THRESHOLD_KM,
) -> pd.DataFrame:
    """Predict which ground stations a spacecraft will pass near.

    Uses the two most recent positions to estimate heading and
    projects forward in 60-second increments for 90 minutes.
    """
    empty = pd.DataFrame(columns=[
        "spacecraft", "station_name", "station_lat", "station_lon",
        "estimated_distance_km", "projected_time_utc", "alert_level",
    ])

    if trajectory_df.empty or len(trajectory_df) < 2:
        return empty

    recent = trajectory_df.tail(2)
    lat1 = recent.iloc[0]["latitude"]
    lon1 = recent.iloc[0]["longitude"]
    t1 = recent.iloc[0]["timestamp"]
    lat2 = recent.iloc[1]["latitude"]
    lon2 = recent.iloc[1]["longitude"]
    t2 = recent.iloc[1]["timestamp"]

    dt = t2 - t1
    if dt == 0:
        return empty

    lat_rate = (lat2 - lat1) / dt
    lon_rate = (lon2 - lon1) / dt

    passes = []
    for step in range(1, 91):
        future_seconds = step * 60
        proj_lat = max(-90, min(90, lat2 + lat_rate * future_seconds))
        proj_lon = ((lon2 + lon_rate * future_seconds + 180) % 360) - 180
        proj_time = t2 + future_seconds

        for _, station in stations_df.iterrows():
            dist = haversine(
                (proj_lat, proj_lon),
                (station["latitude"], station["longitude"]),
                unit=Unit.KILOMETERS,
            )
            if dist <= threshold_km:
                passes.append({
                    "spacecraft": spacecraft_name,
                    "station_name": station["name"],
                    "station_lat": station["latitude"],
                    "station_lon": station["longitude"],
                    "estimated_distance_km": round(dist, 1),
                    "projected_time_utc": datetime.fromtimestamp(
                        proj_time, tz=timezone.utc
                    ).strftime("%H:%M:%S UTC"),
                    "alert_level": _classify_alert_level(dist),
                })

    if not passes:
        return empty

    return (
        pd.DataFrame(passes)
        .sort_values("estimated_distance_km")
        .drop_duplicates(subset=["spacecraft", "station_name"], keep="first")
        .reset_index(drop=True)
    )


def predict_passes_over_location(
    user_lat: float,
    user_lon: float,
    spacecraft_data: dict,
    threshold_km: float = PROXIMITY_THRESHOLD_KM,
) -> pd.DataFrame:
    """Predict which spacecraft will pass over a user's location.

    Projects each spacecraft's trajectory forward and checks if it
    comes within threshold_km of the user's coordinates.
    Returns a DataFrame of upcoming passes sorted by time.
    """
    passes = []

    for sc_name, positions in spacecraft_data.items():
        if not positions or len(positions) < 2:
            continue

        # Use the two most recent positions for projection
        p1, p2 = positions[-2], positions[-1]
        dt = p2["timestamp"] - p1["timestamp"]
        if dt == 0:
            continue

        lat_rate = (p2["latitude"] - p1["latitude"]) / dt
        lon_rate = (p2["longitude"] - p1["longitude"]) / dt

        # Project forward in 60-second steps for 90 minutes
        for step in range(1, 91):
            future_s = step * 60
            proj_lat = max(
                -90, min(90, p2["latitude"] + lat_rate * future_s)
            )
            proj_lon = (
                (p2["longitude"] + lon_rate * future_s + 180) % 360
            ) - 180
            proj_time = p2["timestamp"] + future_s

            dist = haversine(
                (proj_lat, proj_lon),
                (user_lat, user_lon),
                unit=Unit.KILOMETERS,
            )
            if dist <= threshold_km:
                passes.append({
                    "spacecraft": sc_name,
                    "estimated_distance_km": round(dist, 1),
                    "projected_time_utc": datetime.fromtimestamp(
                        proj_time, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC"),
                    "minutes_from_now": step,
                })

    if not passes:
        return pd.DataFrame(columns=[
            "spacecraft", "estimated_distance_km",
            "projected_time_utc", "minutes_from_now",
        ])

    return (
        pd.DataFrame(passes)
        .sort_values("minutes_from_now")
        .drop_duplicates(subset=["spacecraft"], keep="first")
        .reset_index(drop=True)
    )


def generate_interference_summary(
    nearby_df: pd.DataFrame,
    passes_df: pd.DataFrame,
) -> dict:
    """Produce a summary of current and predicted interference risks."""
    return {
        "current_nearby_count": len(nearby_df),
        "critical_count": (
            len(nearby_df[nearby_df["alert_level"] == "CRITICAL"])
            if not nearby_df.empty else 0
        ),
        "warning_count": (
            len(nearby_df[nearby_df["alert_level"] == "WARNING"])
            if not nearby_df.empty else 0
        ),
        "predicted_passes_count": len(passes_df),
        "spacecraft_with_alerts": (
            nearby_df["spacecraft"].nunique()
            if not nearby_df.empty and "spacecraft" in nearby_df.columns
            else 0
        ),
    }
