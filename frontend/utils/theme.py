"""
Design tokens shared across the app. Charts import COLORS from here so the
Plotly figures stay visually consistent with the rest of the UI (cards, map,
banners) without duplicating hex values in multiple files.

The same values are mirrored in .streamlit/config.toml (primary, background,
text and border colours) so Streamlit's own widgets match the custom HTML.
"""

COLORS = {
    "bg": "#F6F4F0",
    "surface": "#FFFFFF",
    "surface_alt": "#FBFAF7",
    "border": "#E4DFD6",
    "text": "#1C1917",
    "text_muted": "#57534E",
    "text_dim": "#8A847B",
    # dark "ink" used by the status banner at the top of the page
    "ink": "#1B1714",
    "ink_2": "#2B211C",
    "ink_text": "#F5EFE8",
    "ink_muted": "#B8ADA3",
    # brand accent -- buttons, selected states, focus rings
    "ember": "#D9480F",
    "ember_bg": "#FFF1E8",
    # severity scale — reserved strictly for risk/danger meaning
    "extreme": "#C0362C",
    "extreme_bg": "#FCEBEA",
    "high": "#D9722C",
    "high_bg": "#FDF0E4",
    "moderate": "#C99A1E",
    "moderate_bg": "#FBF3DD",
    "low": "#3E8858",
    "low_bg": "#E9F5EE",
    "spread": "#2A6FB0",
    "spread_bg": "#E9F1FA",
    # decorative accent palette — purely for visual variety, never used to
    # signal severity, so it can't be confused with the scale above
    "flame": "#F2545B",
    "amber": "#FFB347",
    "gold": "#FFD166",
    "teal": "#06B88A",
    "blue": "#4C6EF5",
    "violet": "#9B5DE5",
}

SEVERITY_STYLE = {
    "Extreme": (COLORS["extreme"], COLORS["extreme_bg"]),
    "High": (COLORS["high"], COLORS["high_bg"]),
    "Moderate": (COLORS["moderate"], COLORS["moderate_bg"]),
    "Low": (COLORS["low"], COLORS["low_bg"]),
}

FONT_BODY = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_DISPLAY = "'Space Grotesk', Inter, system-ui, sans-serif"
