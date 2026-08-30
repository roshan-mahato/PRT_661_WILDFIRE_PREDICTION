import streamlit as st

def render_insight_card(
    title: str,
    body_text: str,
    border_color: str = "#313642",
    bg_color: str = "#1e222a",
    text_color: str = "#FFFFFF",
):
    """Renders a single container box with rounded borders and styled text."""
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h4 style="
                color: {text_color};
                margin-top: 0;
                margin-bottom: 8px;
                font-size: 1.1rem;
                font-weight: 600;
            ">{title}</h4>
            <p style="
                color: {text_color};
                margin: 0;
                font-size: 0.95rem;
                line-height: 1.5;
                opacity: 0.9;
            ">{body_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insights_row(df=None):
    """Renders a 4-column layout of rounded insight boxes matching your wireframe."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_insight_card(
            title=" Hotspot Count",
            body_text="80 active fire clusters detected across the region.",
            border_color="#ff4b4b",
        )

    with col2:
        render_insight_card(
            title=" Risk Level",
            body_text="Extreme spread probability due to low humidity.",
            border_color="#ffa500",
        )

    with col3:
        render_insight_card(
            title=" Peak Intensity",
            body_text="Max radiative power recorded at 9,500 kW/m.",
        )

    with col4:
        render_insight_card(
            title=  "Wind Trajectory",
            body_text="South-West gusting at 45 km/h driving perimeter.",
        )