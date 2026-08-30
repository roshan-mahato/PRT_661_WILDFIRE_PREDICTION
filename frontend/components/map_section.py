import pandas as pd
import pydeck as pdk
import streamlit as st


def render_map_with_insights(df: pd.DataFrame, region: str):
    """Renders a row with 80% map width and 20% right-hand insight panel."""
    # 80% left column, 20% right column ratio
    map_col, right_col = st.columns([8, 2])

    # ---------------------------------------------------------
    # LEFT COLUMN (80% WIDTH): PYDECK MAP
    # ---------------------------------------------------------
    with map_col:
        region_views = {
            "Australia (National)": (-25.2744, 133.7751, 3.8),
            "New South Wales": (-33.8688, 151.2093, 6.2),
            "Victoria": (-37.8136, 144.9631, 6.5),
        }
        lat, lon, zoom = region_views.get(
            region, (-25.2744, 133.7751, 3.8)
        )

        view_state = pdk.ViewState(
            latitude=lat, longitude=lon, zoom=zoom, pitch=30
        )

        # PyDeck Heatmap & Scatter Layers
        layers = [
            pdk.Layer(
                "HeatmapLayer",
                data=df,
                get_position=["longitude", "latitude"],
                get_weight="intensity",
                radius_pixels=60,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=["longitude", "latitude"],
                get_color="[255, 60, 0, 200]",
                get_radius=15000,
                pickable=True,
            ),
        ]

        deck = pdk.Deck(
            map_style='carto-dark',
            initial_view_state=view_state,
            layers=layers,
            tooltip={"html": "<b>Intensity:</b> {intensity} kW/m"},
        )

        # Ensures map stretches to fill the full 80% column
        st.pydeck_chart(deck, use_container_width=True, height=550)

    # ---------------------------------------------------------
    # RIGHT COLUMN (20% WIDTH): INSIGHTS PANEL
    # ---------------------------------------------------------
    with right_col:
        st.markdown(
            """
            <div style="
                background-color: #1e222a;
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
                        background-color: #262c36;
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