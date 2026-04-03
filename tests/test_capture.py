"""Tests for the data capture module."""

from unittest.mock import patch, MagicMock

from src.capture import fetch_iss_position, fetch_ground_stations


class TestFetchISSPosition:
    """Tests for fetch_iss_position."""

    @patch("src.capture.requests.get")
    def test_successful_fetch(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = {
            "latitude": 51.5074,
            "longitude": -0.1278,
            "altitude": 420.5,
            "velocity": 27580.0,
            "timestamp": 1700000000,
            "visibility": "daylight",
        }
        mock_get.return_value.raise_for_status = MagicMock()

        result = fetch_iss_position()

        assert result is not None
        assert result["latitude"] == 51.5074
        assert result["longitude"] == -0.1278
        assert result["altitude"] == 420.5
        assert result["velocity"] == 27580.0
        assert result["timestamp"] == 1700000000

    @patch("src.capture.requests.get")
    def test_returns_none_on_network_error(
        self, mock_get: MagicMock
    ) -> None:
        import requests

        mock_get.side_effect = requests.RequestException("timeout")

        result = fetch_iss_position()
        assert result is None

    @patch("src.capture.requests.get")
    def test_returns_none_on_malformed_json(
        self, mock_get: MagicMock
    ) -> None:
        mock_get.return_value.json.return_value = {"bad": "data"}
        mock_get.return_value.raise_for_status = MagicMock()

        result = fetch_iss_position()
        assert result is None


class TestFetchGroundStations:
    """Tests for fetch_ground_stations."""

    @patch("src.capture.requests.get")
    def test_successful_fetch(self, mock_get: MagicMock) -> None:
        mock_get.return_value.json.return_value = [
            {
                "id": 1,
                "name": "Test Station",
                "lat": 40.7128,
                "lng": -74.0060,
                "altitude": 10,
                "status": 2,
            },
            {
                "id": 2,
                "name": "Inactive Station",
                "lat": 35.6762,
                "lng": 139.6503,
                "altitude": 5,
                "status": 0,
            },
        ]
        mock_get.return_value.raise_for_status = MagicMock()

        result = fetch_ground_stations()

        assert result is not None
        # Only active stations (status=2) should be included
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Test Station"

    @patch("src.capture.requests.get")
    def test_returns_none_on_error(self, mock_get: MagicMock) -> None:
        import requests

        mock_get.side_effect = requests.RequestException("error")

        result = fetch_ground_stations()
        assert result is None
