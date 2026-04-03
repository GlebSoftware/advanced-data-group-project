"""
Data processing module for the ISS Tracking Dashboard.

Handles data cleaning, timestamp conversion, feature engineering,
and DataFrame structuring for downstream analysis.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import TRAJECTORY_HISTORY_SIZE


def process_position(raw_position: dict) -> dict:
    """Clean and enrich a raw position record.

    Converts the Unix timestamp to a readable datetime string
    and ensures all numeric fields are properly typed.
    """
    processed = raw_position.copy()
    processed["datetime_utc"] = datetime.fromtimestamp(
        raw_position["timestamp"], tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")
    processed["speed_kmh"] = raw_position.get("velocity", 0.0)
    return processed


def build_trajectory_dataframe(
    positions: list[dict],
) -> pd.DataFrame:
    """Convert a list of position dicts into a sorted, deduplicated DataFrame."""
    if not positions:
        return pd.DataFrame()

    df = pd.DataFrame(positions)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").drop_duplicates(
            subset=["timestamp"]
        )
        df = df.tail(TRAJECTORY_HISTORY_SIZE).reset_index(drop=True)

    return df


def save_positions_to_csv(
    positions: list[dict], filepath: Path
) -> None:
    """Save position history to CSV for persistence across refreshes."""
    df = pd.DataFrame(positions)
    df.to_csv(filepath, index=False)


def load_positions_from_csv(filepath: Path) -> list[dict]:
    """Load position history from CSV if the file exists."""
    if not filepath.exists():
        return []
    try:
        df = pd.read_csv(filepath)
        return df.to_dict("records")
    except Exception:
        return []
