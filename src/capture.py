"""
Data capture module for the ISS Tracking Dashboard.

Handles all API interactions: real-time ISS position from the
Where the ISS At API, NASA spacecraft positions from the Satellite
Situation Center (SSC) API, and ground station data from SatNOGS.
"""

import math
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import requests

from src.config import (
    ISS_POSITION_URL,
    NASA_SSC_BASE_URL,
    SATNOGS_STATIONS_URL,
    CENSUS_GEOCODER_URL,
    STATION_STATUS_ACTIVE,
    EARTH_RADIUS_KM,
    NASA_SPACECRAFT,
)


def fetch_iss_position() -> Optional[dict]:
    """Fetch the current ISS position from the Where the ISS At API.

    Returns a dict with latitude, longitude, altitude, velocity,
    and timestamp. Returns None if the request fails.
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


def fetch_nasa_ssc_positions(
    spacecraft_id: str, minutes: int = 60
) -> Optional[list[dict]]:
    """Fetch recent positions for a NASA spacecraft from the SSC API.

    Queries the last N minutes of orbital data and converts the
    GEO cartesian coordinates (X, Y, Z in km) to lat/lon/altitude.
    """
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=minutes)

    start_str = start.strftime("%Y%m%dT%H%M%SZ")
    end_str = now.strftime("%Y%m%dT%H%M%SZ")
    url = f"{NASA_SSC_BASE_URL}/locations/{spacecraft_id}/{start_str},{end_str}/geo/"

    try:
        response = requests.get(
            url,
            headers={"Accept": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
        raw = response.json()

        # Navigate the Java-typed JSON structure
        data_result = raw[1]["Result"][1]
        if data_result["StatusCode"] != "SUCCESS":
            return None

        sat_data = data_result["Data"][1][0][1]
        coord = sat_data["Coordinates"][1][0][1]
        times = sat_data["Time"][1]

        x_vals = coord["X"][1]
        y_vals = coord["Y"][1]
        z_vals = coord["Z"][1]

        positions = []
        for i in range(len(x_vals)):
            x, y, z = x_vals[i], y_vals[i], z_vals[i]
            r = math.sqrt(x**2 + y**2 + z**2)

            lat = math.degrees(math.asin(z / r))
            lon = math.degrees(math.atan2(y, x))
            alt = r - EARTH_RADIUS_KM

            # Parse timestamp
            time_str = times[i][1]
            dt = datetime.fromisoformat(
                time_str.replace("+00:00", "+00:00")
            )
            ts = int(dt.timestamp())

            positions.append({
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "altitude": round(alt, 1),
                "velocity": 0.0,
                "timestamp": ts,
                "visibility": "unknown",
            })

        # Estimate velocity from positions if we have enough points
        for i in range(1, len(positions)):
            prev = positions[i - 1]
            curr = positions[i]
            dt_hours = (curr["timestamp"] - prev["timestamp"]) / 3600
            if dt_hours > 0:
                from haversine import haversine, Unit
                dist = haversine(
                    (prev["latitude"], prev["longitude"]),
                    (curr["latitude"], curr["longitude"]),
                    unit=Unit.KILOMETERS,
                )
                curr["velocity"] = round(dist / dt_hours, 1)

        return positions

    except (requests.RequestException, KeyError, ValueError, IndexError):
        return None


def fetch_all_spacecraft() -> dict:
    """Fetch positions for all configured NASA spacecraft.

    Returns a dict mapping spacecraft display name to its list
    of positions. Uses the ISS-specific API for the ISS (faster)
    and the NASA SSC API for all others.
    """
    results = {}

    for sc_id, sc_name, _ in NASA_SPACECRAFT:
        if sc_id == "iss":
            # Use the dedicated ISS API for real-time position
            iss_pos = fetch_iss_position()
            if iss_pos:
                results[sc_name] = [iss_pos]
                # Also fetch SSC trajectory data for the ISS
                ssc_data = fetch_nasa_ssc_positions("iss", minutes=90)
                if ssc_data:
                    results[sc_name] = ssc_data
                    # Update the latest point with real-time data
                    results[sc_name].append(iss_pos)
        else:
            positions = fetch_nasa_ssc_positions(sc_id, minutes=90)
            if positions:
                results[sc_name] = positions

    return results


def fetch_ground_stations() -> Optional[pd.DataFrame]:
    """Fetch active ground stations from the SatNOGS network API.

    Returns a DataFrame with station_id, name, latitude, longitude,
    altitude, and status columns. Returns None if the request fails.
    """
    try:
        stations = []
        url = SATNOGS_STATIONS_URL
        while url:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict):
                stations.extend(data.get("results", []))
                url = data.get("next")
            else:
                stations.extend(data)
                url = None

        df = pd.DataFrame(stations)
        if df.empty:
            return None

        column_map = {
            "id": "station_id",
            "name": "name",
            "lat": "latitude",
            "lng": "longitude",
            "altitude": "altitude",
            "status": "status",
        }

        available = [c for c in column_map if c in df.columns]
        df = df[available].rename(columns=column_map)

        if "status" in df.columns:
            df = df[df["status"] == STATION_STATUS_ACTIVE]

        return df.reset_index(drop=True)

    except (requests.RequestException, KeyError, ValueError):
        return None


def geocode_address(address: str) -> Optional[dict]:
    """Geocode an address using the US Census Bureau Geocoder API.

    Returns a dict with latitude, longitude, and the matched address
    string. Returns None if the address cannot be resolved.
    """
    try:
        response = requests.get(
            CENSUS_GEOCODER_URL,
            params={
                "address": address,
                "benchmark": "Public_AR_Current",
                "format": "json",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        matches = data["result"]["addressMatches"]
        if not matches:
            return None

        match = matches[0]
        coords = match["coordinates"]
        return {
            "latitude": coords["y"],
            "longitude": coords["x"],
            "matched_address": match["matchedAddress"],
        }

    except (requests.RequestException, KeyError, IndexError):
        return None
