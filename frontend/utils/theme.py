"""
Design tokens shared across the app. Charts import COLORS from here so the
Plotly figures stay visually consistent with the rest of the UI (cards, map,
banners) without duplicating hex values in multiple files.
"""

COLORS = {
    "bg": "#F7F7F5",
    "surface": "#FFFFFF",
    "border": "#E3E1DB",
    "text": "#1C1C1A",
    "text_muted": "#5B5B55",
    "text_dim": "#8A877E",
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
