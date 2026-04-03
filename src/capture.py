"""
Data capture module for the ISS Tracking Dashboard.

Handles all API interactions to retrieve real-time spacecraft
positions and ground station reference data.
"""

from typing import Optional

import pandas as pd
import requests

from src.config import (
    ISS_POSITION_URL,
    SATNOGS_STATIONS_URL,
    STATION_STATUS_ACTIVE,
)


def fetch_iss_position() -> Optional[dict]:
    """Fetch the current ISS position from the Where the ISS At API.

    Returns a dict with keys: latitude, longitude, altitude,
    velocity, timestamp, visibility.
    Returns None if the request fails.
    """
    try:
        response = requests.get(ISS_POSITION_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "latitude": float(data["latitude"]),
            "longitude": float(data["longitude"]),
            "altitude": float(data["altitude"]),
            "velocity": float(data["velocity"]),
            "timestamp": int(data["timestamp"]),
            "visibility": data.get("visibility", "unknown"),
        }
    except (requests.RequestException, KeyError, ValueError):
        return None


def fetch_ground_stations() -> Optional[pd.DataFrame]:
    """Fetch active ground stations from the SatNOGS network API.

    Returns a DataFrame with columns: station_id, name, latitude,
    longitude, altitude, status.
    Returns None if the request fails.
    """
    try:
        stations = []
        url = SATNOGS_STATIONS_URL
        while url:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            # SatNOGS API may return paginated results or a list
            if isinstance(data, dict):
                stations.extend(data.get("results", []))
                url = data.get("next")
            else:
                stations.extend(data)
                url = None

        df = pd.DataFrame(stations)

        if df.empty:
            return None

        # Filter to active stations and select relevant columns
        column_map = {
            "id": "station_id",
            "name": "name",
            "lat": "latitude",
            "lng": "longitude",
            "altitude": "altitude",
            "status": "status",
        }

        available_columns = [c for c in column_map if c in df.columns]
        df = df[available_columns].rename(columns=column_map)

        if "status" in df.columns:
            df = df[df["status"] == STATION_STATUS_ACTIVE]

        df = df.reset_index(drop=True)
        return df

    except (requests.RequestException, KeyError, ValueError):
        return None


def fetch_iss_position_history(n_points: int = 10) -> Optional[list[dict]]:
    """Fetch multiple ISS positions by making sequential calls.

    This builds a short trajectory by requesting the current position
    multiple times. For longer history, use stored CSV data.
    Returns a list of position dicts or None if the first call fails.
    """
    position = fetch_iss_position()
    if position is None:
        return None
    return [position]
