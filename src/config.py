# API endpoints
ISS_POSITION_URL = "https://api.wheretheiss.at/v1/satellites/25544"
ISS_TLE_URL = "https://api.wheretheiss.at/v1/satellites/25544/tles"
SATNOGS_STATIONS_URL = "https://network.satnogs.org/api/stations/"

# Proximity threshold in kilometers for interference flagging
PROXIMITY_THRESHOLD_KM = 500

# Auto-refresh interval in seconds
REFRESH_INTERVAL_SECONDS = 10

# Number of historical positions to retain for trajectory plotting
TRAJECTORY_HISTORY_SIZE = 100

# Map settings
MAP_DEFAULT_ZOOM = 1
MAP_STYLE = "mapbox://styles/mapbox/dark-v11"

# Ground station status filter (only include operational stations)
STATION_STATUS_ACTIVE = 2
