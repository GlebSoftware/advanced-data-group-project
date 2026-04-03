# Real-Time ISS Tracking and Pass Prediction Dashboard

**MIST6380 -- Gleb Alikhver, Lucy Moon, Lexie-Anne Rodkey**

A real-time analytics pipeline that tracks the International Space Station (ISS) and other NASA-monitored spacecraft, visualizes their movement across Earth, and predicts pass windows over satellite ground stations for radio frequency interference analysis.

**[Launch the Dashboard](https://advanced-data-group-project.streamlit.app/)**

---

## Application Description and Motivation

This project develops a real-time analytics pipeline that tracks the location of the International Space Station and other NASA-monitored spacecraft and visualizes their movement across the Earth. The system collects live positional data through the NASA Satellite Situation Center API and cross-references it against a global network of satellite ground stations to model potential radio frequency interference windows.

The motivation behind this project is to demonstrate how real-time data from external APIs can be captured, processed, analyzed, and communicated through a full analytics pipeline. Space data is publicly available but often difficult for the average user to interpret. By building a system that cross-references satellite orbital paths with ground station locations, we transform raw coordinate data into meaningful insights.

This project also provides an engaging and educational example of how analytics pipelines can be used to monitor real-world systems in real time. It demonstrates key skills used in modern data analytics roles, including API integration, data cleaning, exploratory analysis, and dashboard-based data communication. The project addresses a real industry problem by anticipating when satellites will pass overhead and potentially disrupt active transmissions. Telecom operators, ground stations, and radio astronomers value these predictions.

Additionally, the project explores geospatial analytics and real-time data visualization. Users can see tracked spacecraft moving across a world map, observe their speed and trajectory, and see a forecast of potential upcoming interference windows over specific ground stations.

## Dashboard Features

- **3D Interactive Globe** -- Orthographic projection showing real-time spacecraft positions, orbital trajectories, and SatNOGS ground station locations
- **Multi-Spacecraft Tracking** -- Simultaneously tracks 7 NASA spacecraft: ISS, Aqua, Aura, Landsat 8, Landsat 9, NOAA-20, and Suomi NPP
- **Altitude Filtering** -- Dual-handle slider (400--900 km) to filter visible spacecraft by orbital altitude
- **Globe Centering** -- Center the view on any tracked spacecraft or Athens, GA
- **Speed and Altitude Charts** -- Time-series visualizations of ground-track speed and orbital altitude for all visible spacecraft
- **Pass Prediction ("What's Passing Over Me?")** -- Enter any US street address to see which spacecraft will pass within 500 km of your location in the next 90 minutes
- **Auto-Refresh** -- Dashboard updates every 60 seconds with fresh orbital data

## Tracked Spacecraft

| Spacecraft | Description | Approximate Altitude |
|---|---|---|
| ISS | International Space Station | ~420 km |
| Aqua | Earth-observing satellite (water cycle research) | ~705 km |
| Aura | Atmospheric chemistry satellite | ~705 km |
| Landsat 8 | Earth imaging satellite | ~705 km |
| Landsat 9 | Earth imaging satellite | ~705 km |
| NOAA-20 | Weather and environmental satellite | ~824 km |
| Suomi NPP | Earth-observing satellite (climate/weather) | ~824 km |

## Analytics Pipeline

### 1. Capture

Data is collected from four public APIs using the Python `requests` library:

- **[NASA Satellite Situation Center (SSC) API](https://sscweb.gsfc.nasa.gov/)** -- Provides GEO cartesian coordinate data (X, Y, Z in km) for NASA spacecraft over configurable time windows. The system queries the last 90 minutes of orbital data for each spacecraft.
- **[Where the ISS At API](https://wheretheiss.at/w/developer)** -- Provides real-time ISS latitude, longitude, and altitude with sub-second freshness.
- **[SatNOGS Network API](https://network.satnogs.org/api/)** -- Provides locations and metadata for a global network of satellite ground stations. Used as a static reference dataset for proximity analysis.
- **[US Census Bureau Geocoder](https://geocoding.geo.census.gov/)** -- Converts user-entered street addresses to geographic coordinates for personalized pass predictions.

No API keys are required for any of these data sources.

### 2. Process

Raw data is processed using Python and Pandas:

- **Coordinate Conversion** -- GEO cartesian coordinates (X, Y, Z in km) from the NASA SSC API are converted to geographic latitude, longitude, and altitude using trigonometric transformations: `lat = arcsin(Z/R)`, `lon = arctan2(Y, X)`, `alt = R - R_earth`
- **Timestamp Standardization** -- All timestamps are parsed, converted to UTC, and displayed in Eastern Time (ET)
- **Speed Estimation** -- Ground-track speed (km/h) is calculated from sequential position pairs using the haversine formula, providing consistent 2D surface speed across all spacecraft
- **Data Deduplication** -- Trajectory DataFrames are sorted by timestamp and deduplicated to prevent rendering artifacts
- **History Management** -- Position histories are capped at 100 points per spacecraft to balance trajectory visualization with memory usage

### 3. Store

Processed data is stored in:

- **Pandas DataFrames** -- Trajectory data structured as tabular datasets for analysis and visualization
- **Streamlit Session State** -- Ground station data cached across dashboard refreshes to minimize redundant API calls
- **Streamlit Cache** -- Spacecraft positions cached with a 30-second TTL to reduce API load while maintaining near-real-time freshness

For production deployment, the system could be expanded to store historical data in a relational database such as MySQL or PostgreSQL. This would enable long-term trend analysis, historical pass verification, and multi-day trajectory playback.

### 4. Analyze

The analysis stage examines spacecraft movement patterns and generates interference forecasts:

- **Proximity Detection** -- Haversine distance is computed between every spacecraft and every ground station. Stations within the proximity threshold are flagged with severity levels: CRITICAL (< 100 km), WARNING (< 250 km), and WATCH (< 500 km).
- **Trajectory Projection** -- Using the two most recent positions, the system estimates heading and speed, then projects each spacecraft's path forward in 60-second increments for 90 minutes. This linear projection approximates near-term orbital motion for short-term pass prediction.
- **Personal Pass Prediction** -- User addresses are geocoded to coordinates, and each spacecraft's projected trajectory is checked for passes within 500 km of the location.
- **Speed and Altitude Analysis** -- Time-series analysis of ground-track speed and orbital altitude reveals orbital characteristics and helps identify anomalous behavior.

### 5. Communicate

The final pipeline stage communicates insights through an interactive Streamlit dashboard:

- A 3D orthographic globe shows spacecraft positions, color-coded trajectories, and ground station markers
- Sidebar metrics display real-time ISS telemetry (latitude, longitude, altitude, speed)
- A spacecraft list with color-coded indicators shows all currently visible spacecraft
- Speed and altitude time-series charts visualize orbital dynamics
- A pass prediction table shows personalized upcoming spacecraft passes for any US address
- An expandable "About This Pipeline" section documents the full data flow

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| Data Capture | `requests` |
| Data Processing | `pandas` |
| Geospatial Analysis | `haversine` |
| Dashboard | `streamlit`, `plotly` |
| Auto-Refresh | `streamlit-autorefresh` |
| Deployment | Streamlit Community Cloud |

## Project Structure

```
advanced-data-group-project/
  app.py              # Main Streamlit dashboard
  requirements.txt    # Python dependencies
  runtime.txt         # Python version for Streamlit Cloud
  src/
    __init__.py
    config.py          # API endpoints, spacecraft list, constants
    capture.py         # API clients (ISS, NASA SSC, SatNOGS, Census)
    process.py         # Data cleaning, DataFrame construction
    analyze.py         # Proximity detection, pass prediction
```

## Running Locally

```bash
# Clone the repository
git clone https://github.com/GlebSoftware/advanced-data-group-project.git
cd advanced-data-group-project

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Data Sources

All data sources are publicly available and require no API keys:

| Source | URL | Data Provided |
|---|---|---|
| Where the ISS At | https://wheretheiss.at/w/developer | Real-time ISS position |
| NASA SSC | https://sscweb.gsfc.nasa.gov/ | Multi-spacecraft orbital data |
| SatNOGS Network | https://network.satnogs.org/api/ | Global ground station locations |
| US Census Geocoder | https://geocoding.geo.census.gov/ | Address-to-coordinate conversion |

## Authors

- **Gleb Alikhver** -- University of Georgia, MIST6380
- **Lucy Moon** -- University of Georgia, MIST6380
- **Lexie-Anne Rodkey** -- University of Georgia, MIST6380
