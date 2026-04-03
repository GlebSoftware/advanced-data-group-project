"""Tests for the analysis module."""

import pandas as pd

from src.analyze import (
    compute_station_distances,
    find_nearby_stations,
    predict_upcoming_passes,
    generate_interference_summary,
)


def _make_stations_df() -> pd.DataFrame:
    """Create a small test stations DataFrame."""
    return pd.DataFrame(
        [
            {
                "station_id": 1,
                "name": "London Station",
                "latitude": 51.5074,
                "longitude": -0.1278,
                "altitude": 10,
                "status": 2,
            },
            {
                "station_id": 2,
                "name": "Tokyo Station",
                "latitude": 35.6762,
                "longitude": 139.6503,
                "altitude": 5,
                "status": 2,
            },
            {
                "station_id": 3,
                "name": "NYC Station",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "altitude": 10,
                "status": 2,
            },
        ]
    )


class TestComputeStationDistances:
    """Tests for compute_station_distances."""

    def test_returns_distances_for_all_stations(self) -> None:
        stations = _make_stations_df()
        result = compute_station_distances(51.5, -0.1, stations)

        assert "distance_km" in result.columns
        assert len(result) == 3

    def test_sorted_by_distance(self) -> None:
        stations = _make_stations_df()
        # ISS near London -- London station should be first
        result = compute_station_distances(51.5, -0.1, stations)
        assert result.iloc[0]["name"] == "London Station"


class TestFindNearbyStations:
    """Tests for find_nearby_stations."""

    def test_finds_station_within_threshold(self) -> None:
        stations = _make_stations_df()
        # Very close to London station
        result = find_nearby_stations(51.5, -0.1, stations, threshold_km=50)

        assert len(result) >= 1
        assert "alert_level" in result.columns

    def test_no_stations_within_threshold(self) -> None:
        stations = _make_stations_df()
        # Middle of the Pacific, far from all test stations
        result = find_nearby_stations(0, -170, stations, threshold_km=100)

        assert len(result) == 0


class TestPredictUpcomingPasses:
    """Tests for predict_upcoming_passes."""

    def test_empty_trajectory(self) -> None:
        stations = _make_stations_df()
        result = predict_upcoming_passes(
            pd.DataFrame(), stations, threshold_km=500
        )
        assert result.empty

    def test_single_point_trajectory(self) -> None:
        stations = _make_stations_df()
        traj = pd.DataFrame(
            [{"latitude": 10, "longitude": 20, "timestamp": 100}]
        )
        result = predict_upcoming_passes(traj, stations, threshold_km=500)
        assert result.empty

    def test_returns_predictions_with_valid_trajectory(self) -> None:
        stations = _make_stations_df()
        # Trajectory heading toward London from the southwest
        traj = pd.DataFrame(
            [
                {
                    "latitude": 45.0,
                    "longitude": -5.0,
                    "timestamp": 1000,
                },
                {
                    "latitude": 46.0,
                    "longitude": -4.0,
                    "timestamp": 1060,
                },
            ]
        )
        # Large threshold to ensure we get results
        result = predict_upcoming_passes(
            traj, stations, threshold_km=2000
        )
        # Should predict at least one pass
        assert not result.empty
        assert "station_name" in result.columns


class TestGenerateInterferenceSummary:
    """Tests for generate_interference_summary."""

    def test_summary_with_no_alerts(self) -> None:
        empty = pd.DataFrame()
        result = generate_interference_summary(empty, empty)

        assert result["current_nearby_count"] == 0
        assert result["nearest_station"] == "None"

    def test_summary_with_alerts(self) -> None:
        nearby = pd.DataFrame(
            [
                {
                    "name": "Test",
                    "distance_km": 50.0,
                    "alert_level": "CRITICAL",
                },
            ]
        )
        passes = pd.DataFrame(
            [{"station_name": "Test", "alert_level": "WARNING"}]
        )
        result = generate_interference_summary(nearby, passes)

        assert result["current_nearby_count"] == 1
        assert result["critical_count"] == 1
        assert result["nearest_station"] == "Test"
