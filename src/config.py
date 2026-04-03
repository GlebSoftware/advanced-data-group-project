# API endpoints
ISS_POSITION_URL = "https://api.wheretheiss.at/v1/satellites/25544"
NASA_SSC_BASE_URL = "https://sscweb.gsfc.nasa.gov/WS/sscr/2"
SATNOGS_STATIONS_URL = "https://network.satnogs.org/api/stations/"
CENSUS_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"

# NASA spacecraft to track via the SSC API (ID, display name, color)
NASA_SPACECRAFT = [
    ("iss", "ISS", "#FF4444"),
    ("aqua", "Aqua", "#00FF88"),
    ("aura", "Aura", "#FFAA00"),
    ("landsat8", "Landsat 8", "#00DDFF"),
    ("landsat9", "Landsat 9", "#FF66FF"),
    ("noaa20", "NOAA-20", "#FFFF44"),
    ("suomi", "Suomi NPP", "#FF8844"),
]

# Earth radius in km (for coordinate conversion)
EARTH_RADIUS_KM = 6371.2

# Proximity threshold in kilometers for interference flagging
PROXIMITY_THRESHOLD_KM = 500

# Auto-refresh interval in seconds
REFRESH_INTERVAL_SECONDS = 15

# Number of historical positions to retain for trajectory plotting
TRAJECTORY_HISTORY_SIZE = 100

# Ground station status filter (only include operational stations)
STATION_STATUS_ACTIVE = 2
