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
    """Calculate distance from spacecraft to every ground station.

    Adds a 'distance_km' column to a copy of the stations DataFrame
    and sorts by distance ascending.
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
    """Identify ground stations within the proximity threshold.

    Returns a DataFrame of stations that are within threshold_km
    of the spacecraft, flagged as potential interference risks.
    """
    df_with_dist = compute_station_distances(
        spacecraft_lat, spacecraft_lon, stations_df
    )
    nearby = df_with_dist[df_with_dist["distance_km"] <= threshold_km].copy()
    nearby["alert_level"] = nearby["distance_km"].apply(
        _classify_alert_level
    )
    return nearby


def _classify_alert_level(distance_km: float) -> str:
    """Classify proximity alert severity based on distance."""
    if distance_km <= 100:
        return "CRITICAL"
    elif distance_km <= 250:
        return "WARNING"
    else:
        return "WATCH"


def predict_upcoming_passes(
    trajectory_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    threshold_km: float = PROXIMITY_THRESHOLD_KM,
) -> pd.DataFrame:
    """Predict which ground stations the ISS will pass near.

    Uses the recent trajectory to estimate the spacecraft's heading
    and projects forward to identify upcoming proximity events.
    Returns a DataFrame of predicted passes with estimated times.
    """
    if trajectory_df.empty or len(trajectory_df) < 2:
        return pd.DataFrame(
            columns=[
                "station_name",
                "station_lat",
                "station_lon",
                "estimated_distance_km",
                "projected_time_utc",
                "alert_level",
            ]
        )

    # Use the two most recent positions to estimate velocity vector
    recent = trajectory_df.tail(2)
    lat1, lon1, t1 = (
        recent.iloc[0]["latitude"],
        recent.iloc[0]["longitude"],
        recent.iloc[0]["timestamp"],
    )
    lat2, lon2, t2 = (
        recent.iloc[1]["latitude"],
        recent.iloc[1]["longitude"],
        recent.iloc[1]["timestamp"],
    )

    dt = t2 - t1
    if dt == 0:
        return pd.DataFrame(
            columns=[
                "station_name",
                "station_lat",
                "station_lon",
                "estimated_distance_km",
                "projected_time_utc",
                "alert_level",
            ]
        )

    # Rate of change in degrees per second
    lat_rate = (lat2 - lat1) / dt
    lon_rate = (lon2 - lon1) / dt

    # Project forward in 60-second increments for the next 90 minutes
    # (roughly one ISS orbit)
    passes = []
    for step in range(1, 91):
        future_seconds = step * 60
        projected_lat = lat2 + lat_rate * future_seconds
        projected_lon = lon2 + lon_rate * future_seconds

        # Clamp latitude to valid range
        projected_lat = max(-90, min(90, projected_lat))

        # Wrap longitude to [-180, 180]
        projected_lon = ((projected_lon + 180) % 360) - 180

        projected_time = t2 + future_seconds

        for _, station in stations_df.iterrows():
            dist = haversine(
                (projected_lat, projected_lon),
                (station["latitude"], station["longitude"]),
                unit=Unit.KILOMETERS,
            )
            if dist <= threshold_km:
                passes.append(
                    {
                        "station_name": station["name"],
                        "station_lat": station["latitude"],
                        "station_lon": station["longitude"],
                        "estimated_distance_km": round(dist, 1),
                        "projected_time_utc": datetime.fromtimestamp(
                            projected_time, tz=timezone.utc
                        ).strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "alert_level": _classify_alert_level(dist),
                    }
                )

    passes_df = pd.DataFrame(passes)
    if not passes_df.empty:
        passes_df = (
            passes_df.sort_values("estimated_distance_km")
            .drop_duplicates(subset=["station_name"], keep="first")
            .reset_index(drop=True)
        )
    return passes_df


def generate_interference_summary(
    nearby_stations: pd.DataFrame,
    predicted_passes: pd.DataFrame,
) -> dict:
    """Produce a summary of current and predicted interference risks.

    Returns a dict with counts and details for dashboard display.
    """
    return {
        "current_nearby_count": len(nearby_stations),
        "critical_count": len(
            nearby_stations[nearby_stations["alert_level"] == "CRITICAL"]
        )
        if not nearby_stations.empty
        else 0,
        "warning_count": len(
            nearby_stations[nearby_stations["alert_level"] == "WARNING"]
        )
        if not nearby_stations.empty
        else 0,
        "predicted_passes_count": len(predicted_passes),
        "nearest_station": nearby_stations.iloc[0]["name"]
        if not nearby_stations.empty
        else "None",
        "nearest_distance_km": round(
            nearby_stations.iloc[0]["distance_km"], 1
        )
        if not nearby_stations.empty
        else None,
    }
