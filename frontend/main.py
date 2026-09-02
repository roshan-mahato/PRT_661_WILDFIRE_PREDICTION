import streamlit as st

import numpy as np
import pandas as pd
from components.headers import render_header
from components.insights import render_insights_row
from components.map_section import render_map_with_insights 

st.set_page_config(
    page_title="Wildfire Prediction Platform",
    page_icon="🔥",
    layout="wide",  # <--- Forces wide mode across the browser window
    initial_sidebar_state="collapsed",
)
# Renders header component
header = render_header()

insights = render_insights_row()

# Map Component
hotspots_json = [
    {"latitude": -33.8688, "longitude": 151.2093, "intensity": 8500},
    {"latitude": -32.5000, "longitude": 151.1000, "intensity": 4200},
    {"latitude": -34.1000, "longitude": 150.8000, "intensity": 9100},
]

render_map_with_insights(hotspots_json, region="New South Wales")
