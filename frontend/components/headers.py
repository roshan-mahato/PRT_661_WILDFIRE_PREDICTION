import streamlit as st 


def render_header():
    """Renders the top header and region selector"""
    title_col, region_col = st.columns([3,1])
    with title_col:
        st.title("Wildfire Prediction Platform")
        
    with region_col:
        selected_region = st.selectbox("Region", options=[
            "Australia(national)",
            "New South Wales",

        ], index = 0)
    return selected_region

    