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
df = pd.DataFrame(
    {
        "latitude": [-33.86, -32.50, -34.10],
        "longitude": [151.20, 151.10, 150.80],
        "intensity": [8500, 4200, 9100],
    }
)
map_section = render_map_with_insights(df, "New South Wales")
