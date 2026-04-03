"""
Data processing module for the ISS Tracking Dashboard.

Handles data cleaning, timestamp conversion, feature engineering,
and DataFrame structuring for downstream analysis.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import TRAJECTORY_HISTORY_SIZE


def process_iss_position(raw_position: dict) -> dict:
    """Clean and enrich a raw ISS position record.

    Converts the Unix timestamp to a human-readable datetime string
    and ensures all numeric fields are properly typed.
    """
    processed = raw_position.copy()
    processed["datetime_utc"] = datetime.fromtimestamp(
        raw_position["timestamp"], tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")
    processed["speed_kmh"] = raw_position["velocity"]
    return processed


def build_trajectory_dataframe(
    position_history: list[dict],
) -> pd.DataFrame:
    """Convert a list of position dicts into a structured DataFrame.

    Sorts by timestamp, drops duplicates, and limits to the
    configured trajectory history size.
    """
    if not position_history:
        return pd.DataFrame()

    df = pd.DataFrame(position_history)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").drop_duplicates(
            subset=["timestamp"]
        )
        df = df.tail(TRAJECTORY_HISTORY_SIZE).reset_index(drop=True)

    return df


def append_position_to_history(
    history: list[dict], new_position: dict
) -> list[dict]:
    """Append a new position to the history list, enforcing max size.

    Prevents duplicate timestamps from consecutive API calls.
    """
    if history and history[-1]["timestamp"] == new_position["timestamp"]:
        return history

    history.append(new_position)

    if len(history) > TRAJECTORY_HISTORY_SIZE:
        history = history[-TRAJECTORY_HISTORY_SIZE:]

    return history


def save_positions_to_csv(
    positions: list[dict], filepath: Path
) -> None:
    """Save position history to a CSV file for persistence."""
    df = pd.DataFrame(positions)
    df.to_csv(filepath, index=False)


def load_positions_from_csv(filepath: Path) -> list[dict]:
    """Load position history from a CSV file if it exists."""
    if not filepath.exists():
        return []
    df = pd.read_csv(filepath)
    return df.to_dict("records")


def calculate_speed_between_points(
    lat1: float,
    lon1: float,
    timestamp1: int,
    lat2: float,
    lon2: float,
    timestamp2: int,
) -> float:
    """Calculate speed in km/h between two coordinate-timestamp pairs.

    Uses the haversine distance divided by elapsed time.
    Returns 0.0 if timestamps are identical.
    """
    from haversine import haversine, Unit

    time_diff_hours = (timestamp2 - timestamp1) / 3600.0
    if time_diff_hours == 0:
        return 0.0

    distance_km = haversine(
        (lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS
    )
    return distance_km / time_diff_hours
