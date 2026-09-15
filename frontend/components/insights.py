import streamlit as st

from utils.regions import ffdi_band
from utils.theme import COLORS, SEVERITY_STYLE


def render_insight_card(
    title: str,
    body_text: str,
    border_color: str = COLORS["border"],
    bg_color: str = COLORS["bg"],
    text_color: str = COLORS["text"],
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


def render_insights_row(summary: dict):
    """
    Renders the four headline cards from a region summary produced by
    utils.regions.summarise_region().

    When there is no data the cards still render, saying so plainly -- an
    empty row would look like a rendering bug rather than missing data.
    """
    col1, col2, col3, col4 = st.columns(4)

    if not summary.get("has_data"):
        for col, title in zip(
            (col1, col2, col3, col4),
            ("Cells at Risk", "Risk Level", "Peak FFDI", "Conditions"),
        ):
            with col:
                render_insight_card(
                    title=title,
                    body_text="No prediction data available for this region.",
                    border_color=COLORS["border"],
                )
        return

    risk_colour, risk_bg = SEVERITY_STYLE.get(
        summary["max_risk_level"], (COLORS["border"], COLORS["bg"])
    )

    with col1:
        render_insight_card(
            title="Cells at Risk",
            body_text=(
                f"{summary['at_risk_count']} of {summary['cell_count']} grid cells "
                f"predicted to see fire activity."
            ),
            border_color=COLORS["flame"],
        )

    with col2:
        render_insight_card(
            title="Risk Level",
            body_text=(
                f"{summary['max_risk_level']} — peak fire probability "
                f"{summary['max_probability']:.0%} across the region."
            ),
            border_color=risk_colour,
            bg_color=risk_bg,
        )

    with col3:
        render_insight_card(
            title="Peak FFDI",
            body_text=(
                f"{summary['max_ffdi']:.1f} ({ffdi_band(summary['max_ffdi'])}) — "
                f"regional average {summary['mean_ffdi']:.1f}."
            ),
            border_color=COLORS["amber"],
        )

    with col4:
        render_insight_card(
            title="Conditions",
            body_text=(
                f"{summary['mean_temp']:.0f}°C, {summary['mean_humidity']:.0f}% humidity, "
                f"wind {summary['mean_wind']:.0f} km/h."
            ),
            border_color=COLORS["spread"],
        )
