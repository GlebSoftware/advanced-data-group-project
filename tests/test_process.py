"""Tests for the data processing module."""

from src.process import (
    process_iss_position,
    build_trajectory_dataframe,
    append_position_to_history,
    calculate_speed_between_points,
)


class TestProcessISSPosition:
    """Tests for process_iss_position."""

    def test_adds_datetime_string(self) -> None:
        raw = {
            "latitude": 10.0,
            "longitude": 20.0,
            "altitude": 400.0,
            "velocity": 27000.0,
            "timestamp": 1700000000,
            "visibility": "daylight",
        }
        result = process_iss_position(raw)

        assert "datetime_utc" in result
        assert "UTC" in result["datetime_utc"]
        assert result["speed_kmh"] == 27000.0

    def test_preserves_original_fields(self) -> None:
        raw = {
            "latitude": -33.8,
            "longitude": 151.2,
            "altitude": 415.0,
            "velocity": 27500.0,
            "timestamp": 1700000100,
            "visibility": "eclipsed",
        }
        result = process_iss_position(raw)

        assert result["latitude"] == -33.8
        assert result["longitude"] == 151.2


class TestBuildTrajectoryDataframe:
    """Tests for build_trajectory_dataframe."""

    def test_empty_input(self) -> None:
        result = build_trajectory_dataframe([])
        assert result.empty

    def test_builds_sorted_dataframe(self) -> None:
        positions = [
            {"latitude": 10.0, "longitude": 20.0, "timestamp": 200},
            {"latitude": 11.0, "longitude": 21.0, "timestamp": 100},
        ]
        result = build_trajectory_dataframe(positions)

        assert len(result) == 2
        assert result.iloc[0]["timestamp"] == 100

    def test_deduplicates_by_timestamp(self) -> None:
        positions = [
            {"latitude": 10.0, "longitude": 20.0, "timestamp": 100},
            {"latitude": 10.5, "longitude": 20.5, "timestamp": 100},
        ]
        result = build_trajectory_dataframe(positions)
        assert len(result) == 1


class TestAppendPositionToHistory:
    """Tests for append_position_to_history."""

    def test_appends_new_position(self) -> None:
        history = [{"timestamp": 100}]
        new = {"timestamp": 200}
        result = append_position_to_history(history, new)
        assert len(result) == 2

    def test_skips_duplicate_timestamp(self) -> None:
        history = [{"timestamp": 100}]
        new = {"timestamp": 100}
        result = append_position_to_history(history, new)
        assert len(result) == 1


class TestCalculateSpeed:
    """Tests for calculate_speed_between_points."""

    def test_zero_time_returns_zero(self) -> None:
        result = calculate_speed_between_points(
            0, 0, 100, 10, 10, 100
        )
        assert result == 0.0

    def test_positive_speed(self) -> None:
        # London to Paris, 1 hour apart
        result = calculate_speed_between_points(
            51.5074, -0.1278, 0, 48.8566, 2.3522, 3600
        )
        assert result > 0
        # Distance is roughly 340 km, so speed should be around 340 km/h
        assert 300 < result < 400
