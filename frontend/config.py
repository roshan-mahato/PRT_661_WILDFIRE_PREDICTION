"""
Tunable settings for the dashboard, in one place.

Values that depend on the environment (the API address and the refresh
timeout) can be overridden with environment variables; everything else is a
plain constant. Colours and fonts live in utils.theme, styling in
assets/style.css.
"""

import os

# --- Backend API ---------------------------------------------------------------
API_BASE_URL = os.getenv("WILDFIRE_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 30  # seconds for ordinary API reads
REFRESH_TIMEOUT = int(os.getenv("WILDFIRE_REFRESH_TIMEOUT", "180"))  # a full refresh is slow
# Live fetch: weather download + predict. One state takes seconds; the whole
# country takes minutes (longer on the public API, which is paced).
LIVE_FETCH_TIMEOUT = int(os.getenv("WILDFIRE_LIVE_TIMEOUT", "900"))
# In Live mode a region is fetched again only if its last fetch is older than this.
LIVE_MAX_AGE = 600  # seconds

# How long fetched data is reused before asking the API again (seconds).
PREDICTIONS_CACHE_TTL = 300  # predictions only change when a refresh or fetch runs
HEALTH_CACHE_TTL = 60

# --- Forecast grid -------------------------------------------------------------
GRID_SIZE = 0.5  # degrees; must match the backend's prediction grid

# --- Page layout ---------------------------------------------------------------
DAY_SLOTS = 7          # day strip: a full forecast week; fewer days leave placeholders
MAP_HEIGHT = 580       # pixels
HOTSPOT_ROWS = 8       # chart 4: places shown in the timeline
TOP_PLACES = 5         # chart 3: places named beside the donut
NEAR_KM = 25           # closer than this to a town reads as "near <town>"

# --- Town search ---------------------------------------------------------------
RADIUS_OPTIONS_KM = (25, 50, 100)   # "area around the town" choices
DEFAULT_RADIUS_KM = 50
NEARBY_TOWNS_MIN_KM = 100   # chart 3 looks at least this far for neighbouring towns
FALLBACK_MAX_KM = 150       # a town with no forecast area uses the nearest one up to this far
FOCUS_MAP_VIEW_KM = 100     # the map shows at least this far around a searched town
