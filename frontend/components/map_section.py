from typing import Any,Dict,List
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium

def render_map_with_insights(
        hotspots_json: List[Dict[str, Any]], region: str
):
    """Renders 80% Folium Map (left) and  20% Threat summary (Right) using raw JSON payloald."""
    map_col, right_col = st.columns([8,2])

    with map_col:
        region_views = {
            "Australia (National)": ([-25.2744, 133.7751], 4),
            "New South Wales": ([-33.8688, 151.2093], 6),
            "Victoria": ([-37.8136, 144.9631], 7),
            "Queensland": ([-27.4705, 153.0260], 5),
        }
        center, zoom = region_views.get(region, ([-25.2744, 133.7751], 4))

        attr = ('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>, &copy; <a href="https://carto.com/attributions">CARTO</a>')
        tiles = "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_2s99_1_503a3a7bf96ca39604de30b4"

        # Base map initialization
        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles=tiles,
            control_scale=False,
            attr=attr
        )

        if hotspots_json:
            # Extract [lat, lon, intensity] triples directly from JSON
            heat_data = [
                [item["latitude"], item["longitude"], item["intensity"]]
                for item in hotspots_json
            ]

            # Render HeatMap
            HeatMap(
                heat_data,
                radius=18,
                blur=22,
                max_zoom=1,
                gradient={0.4: "#0000FF", 0.7: "#FFA500", 1.0: "#FF0000"},
            ).add_to(m)

            # Draw circle markers for individual hotspots
            for item in hotspots_json:
                folium.CircleMarker(
                    location=[item["latitude"], item["longitude"]],
                    radius=4,
                    color="#FF4B4B",
                    fill=True,
                    fill_color="#FF4B4B",
                    fill_opacity=0.8,
                    popup=f"Intensity: {item['intensity']} kW/m",
                ).add_to(m)

        # Render Folium inside Streamlit
        st_folium(
            m,
            use_container_width=True,
            height=800,
            zoom=5,
            returned_objects=[],
        )

    # ---------------------------------------------------------
    # 2. RIGHT COLUMN: INSIGHT PANEL (20%)
    # ---------------------------------------------------------
    with right_col:
        st.markdown(
            """
            <div style=" 
                border: 1px solid #313642;
                border-radius: 12px;
                padding: 16px;
                height: 550px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
                <div>
                    <h4 style="margin-top:0; color: #FF4B4B; font-size: 1.05rem;">
                        🚨 Region Threat Summary
                    </h4>
                    <hr style="border-color: #313642; margin: 10px 0;">
                    <p style="font-size: 0.85rem; margin-bottom: 12px;">
                        <b>Primary Cluster:</b><br>Hunter Valley Sector
                    </p>
                    <p style="font-size: 0.85rem; margin-bottom: 12px;">
                        <b>Spread Direction:</b><br>South-East (135°)
                    </p>
                    <p style="font-size: 0.85rem; margin-bottom: 12px;">
                        <b>Growth Rate:</b><br>+2.4 km²/hr
                    </p>
                </div>
                <div>
                    <div style="
                        border-left: 3px solid #ffa500;
                        padding: 8px 10px;
                        border-radius: 4px;
                        font-size: 0.8rem;
                    ">
                        <b>Status:</b> High wind alert active for coastal perimeters.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )